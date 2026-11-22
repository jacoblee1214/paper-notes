"""EgoWAM (arXiv:2607.08436 v2) 표 수치 재계산과 검정.

입력: 논문 Fig. 3 / Fig. 8 / Fig. 9 / Fig. 11 / Table 1-4 / Table 3 값.
  Fig. 3의 막대 높이는 프로젝트 페이지(gatech-rl2.github.io/egowam.github.io)의 app.js에 있는
  PERF_DATA(소수 4자리)를 옮겼고, Fig. 3의 증감 라벨(그림 위 +n%)은 PDF 그림에서 직접 읽어 옮겼다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy 필요, 수 초)

점검 라벨: V1-V17 (재계산 항목).
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
RES = {}

# ---------------------------------------------------------------------------
# 0. 입력 데이터
# ---------------------------------------------------------------------------
METHODS = ["bc", "pixel", "pixelpt", "dino", "flow"]
LABEL = {"bc": "BC", "pixel": "Pixel", "pixelpt": "Pixel-PT", "dino": "DINO", "flow": "3D Flow"}
# regime: base=로봇만, cotrain=+In-Domain Human(1:1), flagship=+EgoVerse(~10:1)
# split: in_domain / ood(=OOD 물체와 OOD 장면 평균); 값은 (normalized score, success rate)
PERF = {
    "cup": {
        "bc": {"base": ((.35, .25), (.1481, .0556)), "cotrain": ((.2167, .15), (.0926, .0556)), "flagship": ((.30, .15), (.1296, .0556))},
        "pixel": {"base": ((.3667, .25), (.1481, .1111)), "cotrain": ((.45, .35), (.1296, .0556)), "flagship": ((.7167, .50), (.2315, .0556))},
        "pixelpt": {"base": ((.50, .40), (.1852, .1111)), "cotrain": ((.60, .45), (.1852, .1111)), "flagship": ((.6333, .50), (.2407, .1111))},
        "dino": {"base": ((.4167, .25), (.1852, .1111)), "cotrain": ((.65, .50), (.3611, .2167)), "flagship": ((.8333, .70), (.6241, .5389))},
        "flow": {"base": ((.80, .70), (.2222, .2222)), "cotrain": ((.90, .85), (.3704, .2222)), "flagship": ((.9333, .90), (.6611, .5389))},
    },
    "bag": {
        "bc": {"base": ((.7375, .40), (.325, .05)), "cotrain": ((.7125, .45), (.3375, .10)), "flagship": ((.8125, .50), (.3625, .10))},
        "pixel": {"base": ((.85, .45), (.325, .10)), "cotrain": ((.8125, .40), (.325, .10)), "flagship": ((.9125, .65), (.375, .15))},
        "pixelpt": {"base": ((.575, .30), (.2125, .05)), "cotrain": ((.7875, .35), (.3125, .05)), "flagship": ((.80, .35), (.7875, .35))},
        "dino": {"base": ((.825, .40), (.3125, .10)), "cotrain": ((.925, .80), (.50, .15)), "flagship": ((.9625, .85), (.8375, .50))},
        "flow": {"base": ((.875, .60), (.4375, .10)), "cotrain": ((.95, .80), (.575, .30)), "flagship": ((.9625, .90), (.7375, .55))},
    },
    "fold": {  # 논문 Fig. 3에는 fold-clothes의 In-Domain Human 막대가 없다
        "bc": {"base": ((1.0, 1.0), (.2583, .05)), "flagship": ((.9167, .85), (.2417, .0))},
        "pixel": {"base": ((1.0, 1.0), (.2833, .20)), "flagship": ((1.0, 1.0), (.575, .10))},
        "pixelpt": {"base": ((1.0, 1.0), (.30, .25)), "flagship": ((1.0, 1.0), (.3583, .25))},
        "dino": {"base": ((1.0, 1.0), (.2417, .15)), "flagship": ((1.0, 1.0), (.9167, .85))},
        "flow": {"base": ((1.0, 1.0), (.3667, .25)), "flagship": ((1.0, 1.0), (.7583, .40))},
    },
}
MAXPTS = {"cup": 3, "bag": 4, "fold": 3}  # 하위 과제 점수 만점 (부록 C.3)

# Fig. 3 그림 위 증감 라벨 (EgoVerse - 로봇만, %p). 순서: BC, Pixel, Pixel-PT, DINO, 3D Flow
FIG3_LABELS = {
    ("cup", "id", "ns"): [-5, 35, 13, 42, 13], ("cup", "id", "sr"): [-10, 25, 10, 45, 20],
    ("cup", "ood", "ns"): [-2, 8, 6, 44, 44], ("cup", "ood", "sr"): [0, -6, 0, 43, 32],
    ("bag", "id", "ns"): [7, 6, 23, 14, 9], ("bag", "id", "sr"): [10, 20, 5, 45, 30],
    ("bag", "ood", "ns"): [4, 5, 58, 52, 30], ("bag", "ood", "sr"): [5, 5, 30, 40, 45],
    ("fold", "id", "ns"): [-8, 0, 0, 0, 0], ("fold", "id", "sr"): [-15, 0, 0, 0, 0],
    ("fold", "ood", "ns"): [-2, 29, 6, 68, 39], ("fold", "ood", "sr"): [-5, -10, 0, 70, 15],
}


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def r(x, nd=4):
    return None if x is None else round(float(x), nd)


# ---------------------------------------------------------------------------
# V1. Fig. 3 라벨 대 막대 높이에서 계산한 증감
# ---------------------------------------------------------------------------
v1 = []
mismatch = 0
for (task, split, met), labels in FIG3_LABELS.items():
    for m, lab in zip(METHODS, labels):
        base = PERF[task][m]["base"]
        flag = PERF[task][m]["flagship"]
        si = 0 if split == "id" else 1
        mi = 0 if met == "ns" else 1
        d = (flag[si][mi] - base[si][mi]) * 100
        ok = abs(round(d + 1e-9) - lab) <= 1  # 반올림 차이 1%p 허용
        exact = int(math.floor(d + 0.5)) == lab if d >= 0 else -int(math.floor(-d + 0.5)) == lab
        if not ok:
            mismatch += 1
        v1.append({"task": task, "split": split, "metric": met, "method": m, "label": lab,
                   "computed": r(d, 2), "match_within_1pt": ok, "match_round": exact})
RES["V1_fig3_label_vs_bars"] = {"n": len(v1), "mismatch_gt1pt": mismatch,
                                "n_not_exact_round": sum(1 for x in v1 if not x["match_round"]),
                                "rows_not_exact_round": [x for x in v1 if not x["match_round"]]}

# ---------------------------------------------------------------------------
# V2. 정수 가능성: 시행 수 복원
# ---------------------------------------------------------------------------
def feasible_id(val, den):
    k = round(val * den)
    return abs(k / den - val) < 6e-5, k


v2 = {"id_sr_non_integer": [], "id_ns_non_integer": [], "ood_sr_pairs": {}, "ood_ns_pairs": {}}
for task in PERF:
    ood_sr_vals, ood_ns_vals = [], []
    for m in METHODS:
        for reg, ((ns_id, sr_id), (ns_ood, sr_ood)) in PERF[task][m].items():
            ok, k = feasible_id(sr_id, 20)
            if not ok:
                v2["id_sr_non_integer"].append((task, m, reg, sr_id))
            ok, k = feasible_id(ns_id, MAXPTS[task] * 20)
            if not ok:
                v2["id_ns_non_integer"].append((task, m, reg, ns_id))
            ood_sr_vals.append(sr_ood)
            ood_ns_vals.append(ns_ood)

    def pair_bad(vals, n1, n2, pts):
        # OOD 값 = (a/(pts*n1) + b/(pts*n2)) / 2 (두 분할의 평균), a, b는 정수 점수 합. 격자 밖인 값의 수.
        bad = 0
        for v in vals:
            found = False
            for a in range(pts * n1 + 1):
                for b in range(pts * n2 + 1):
                    if abs((a / (pts * n1) + b / (pts * n2)) / 2 - v) < 6e-5:
                        found = True
                        break
                if found:
                    break
            if not found:
                bad += 1
        return bad

    for (n1, n2) in [(10, 10), (9, 10), (9, 9), (20, 20)]:
        v2["ood_sr_pairs"][f"{task}_{n1}+{n2}"] = pair_bad(ood_sr_vals, n1, n2, 1)
        v2["ood_ns_pairs"][f"{task}_{n1}+{n2}"] = pair_bad(ood_ns_vals, n1, n2, MAXPTS[task])
    both = []
    for n1 in range(6, 21):
        for n2 in range(n1, 21):
            if pair_bad(ood_sr_vals, n1, n2, 1) == 0 and pair_bad(ood_ns_vals, n1, n2, MAXPTS[task]) == 0:
                both.append((n1, n2))
    v2.setdefault("ood_pairs_feasible_for_all", {})[task] = both
RES["V2_integer_feasibility"] = v2

# Fig. 9에서 읽은 OOD 물체 SR(11, 44, 78 %)은 n=9 격자(1/9 = 11.1 %, 4/9 = 44.4 %, 7/9 = 77.8 %)와 맞고,
# OOD 장면 SR(0, 10, 30 %)은 n=10 격자와 맞는다.
RES["V2b_fig9_ood_object_grid"] = {
    "values_pct": [11, 44, 78], "n9_grid_pct": [round(100 * k / 9, 1) for k in (1, 4, 7)],
    "n10_grid_pct": [10, 40, 80], "scene_values_pct": [0, 10, 30],
}

# ---------------------------------------------------------------------------
# V3. 롤아웃 수 1800 점검
# ---------------------------------------------------------------------------
cells = {"cup": 5 * 3, "bag": 5 * 3, "fold": 5 * 2}  # 방법 5 x 체제 (fold는 로봇만, +EgoVerse 두 가지)
per_cell = 20 + 20
RES["V3_rollout_count"] = {
    "cells_by_task": cells, "total_cells": sum(cells.values()), "per_cell": per_cell,
    "total_from_fig3": sum(cells.values()) * per_cell,
    "if_fold_had_in_domain": 3 * 5 * 3 * per_cell,
    "cup_ood_object_n9_total": sum(cells.values()) * per_cell - 15 * 1,  # cup OOD 물체 9회면 칸당 1회 적음 (cup 15칸)
    "ablations_extra": {
        "fig8a_bag_unaligned(BC,flow) ID20": 2 * 20,
        "fig8b_cup_aligned(4 methods) ID20": 4 * 20,
        "fig9_cup(action-only, flow-only) ID20+OODobj9+OODscene10": 2 * (20 + 9 + 10),
        "fig11_pixelpt_dense(2 regimes) ID20+OOD19": 2 * (20 + 19),
    },
    "paper_claim": 1800,
}
RES["V3_rollout_count"]["with_all_ablations"] = (RES["V3_rollout_count"]["total_from_fig3"]
                                                 + sum(RES["V3_rollout_count"]["ablations_extra"].values()))

# ---------------------------------------------------------------------------
# V4. 성공률 Wilson 구간과 Fisher 검정: 로봇만 대 +EgoVerse
# ---------------------------------------------------------------------------
def counts(task, m, reg, split):
    (ns_id, sr_id), (ns_ood, sr_ood) = PERF[task][m][reg]
    if split == "id":
        return round(sr_id * 20), 20
    n = 19 if task == "cup" else 20      # cup OOD는 물체 9 + 장면 10 (Fig. 9, V2b)
    return round(sr_ood * n), n


v4 = []
for task in PERF:
    for split in ("id", "ood"):
        for m in METHODS:
            k0, n0 = counts(task, m, "base", split)
            k1, n1 = counts(task, m, "flagship", split)
            lo0, hi0 = wilson(k0, n0)
            lo1, hi1 = wilson(k1, n1)
            v4.append({"task": task, "split": split, "method": m, "k_base": k0, "n_base": n0,
                       "k_flag": k1, "n_flag": n1, "sr_base": r(k0 / n0, 3), "sr_flag": r(k1 / n1, 3),
                       "delta_pp": r(100 * (k1 / n1 - k0 / n0), 1),
                       "wilson_base": [r(lo0, 3), r(hi0, 3)], "wilson_flag": [r(lo1, 3), r(hi1, 3)],
                       "fisher_p": r(fisher(k1, n1, k0, n0), 4)})
RES["V4_sr_base_vs_egoverse"] = v4

# BC가 로봇만보다 나빠진다는 주장 (cup ID, fold ID, cup OOD scene...)
RES["V5_bc_degradation"] = [x for x in v4 if x["method"] == "bc"]

# ---------------------------------------------------------------------------
# V6. +EgoVerse 조건에서 방법 간 비교 (대 BC), 로봇만 조건에서 방법 간 비교 (대 BC)
# ---------------------------------------------------------------------------
v6 = []
for task in PERF:
    for split in ("id", "ood"):
        for reg in ("base", "flagship"):
            kb, nb = counts(task, "bc", reg, split)
            for m in METHODS[1:]:
                k, n = counts(task, m, reg, split)
                v6.append({"task": task, "split": split, "regime": reg, "method": m,
                           "k": k, "n": n, "k_bc": kb, "n_bc": nb, "fisher_p_vs_bc": r(fisher(k, n, kb, nb), 4)})
RES["V6_method_vs_bc"] = v6

v6b = []  # DINO 대 3D Flow (EgoVerse)
for task in PERF:
    for split in ("id", "ood"):
        kd, nd = counts(task, "dino", "flagship", split)
        kf, nf = counts(task, "flow", "flagship", split)
        v6b.append({"task": task, "split": split, "dino": f"{kd}/{nd}", "flow": f"{kf}/{nf}",
                    "fisher_p": r(fisher(kd, nd, kf, nf), 4)})
RES["V6b_dino_vs_flow_egoverse"] = v6b

# ---------------------------------------------------------------------------
# V7. 초록·서론의 정량 문구 점검
# ---------------------------------------------------------------------------
dino_ratio = {}
for task in PERF:
    (ns0, sr0), (ns0o, sr0o) = PERF[task]["dino"]["base"]
    (ns1, sr1), (ns1o, sr1o) = PERF[task]["dino"]["flagship"]
    dino_ratio[task] = {"ood_ns_ratio": r(ns1o / ns0o, 2), "ood_sr_ratio": r(sr1o / sr0o, 2)}
flow_id = {t: {"sr_gain_pp": r(100 * (PERF[t]["flow"]["flagship"][0][1] - PERF[t]["flow"]["base"][0][1]), 1),
               "ns_gain_pp": r(100 * (PERF[t]["flow"]["flagship"][0][0] - PERF[t]["flow"]["base"][0][0]), 1)} for t in PERF}
avg = {}
for m in METHODS:
    avg[m] = {
        "ood_sr_mean_flagship": r(np.mean([PERF[t][m]["flagship"][1][1] for t in PERF]), 3),
        "ood_ns_mean_flagship": r(np.mean([PERF[t][m]["flagship"][1][0] for t in PERF]), 3),
        "id_sr_mean_flagship": r(np.mean([PERF[t][m]["flagship"][0][1] for t in PERF]), 3),
        "id_sr_mean_base": r(np.mean([PERF[t][m]["base"][0][1] for t in PERF]), 3),
        "ood_sr_mean_base": r(np.mean([PERF[t][m]["base"][1][1] for t in PERF]), 3),
    }
RES["V7_abstract_claims"] = {"dino_ood_ratio_by_task": dino_ratio, "flow_in_domain_gain": flow_id, "task_means": avg}

# ---------------------------------------------------------------------------
# V8. Fig. 1 수치 (+8 ... +68 %)를 Fig. 3 값에서 만들 수 있는지: 단순 가설 검사
# ---------------------------------------------------------------------------
FIG1 = {"pixel": {"Spatial": 8, "Object": 5, "Scene": 10}, "dino": {"Spatial": 33, "Object": 52, "Scene": 68},
        "flow": {"Spatial": 52, "Object": 44, "Scene": 39}}
cand = {}
for m in ("pixel", "dino", "flow"):
    cand[m] = {}
    for name, fn in {
        "ID_NS_mean3": lambda m: np.mean([FIG3_LABELS[(t, "id", "ns")][METHODS.index(m)] for t in PERF]),
        "ID_SR_mean3": lambda m: np.mean([FIG3_LABELS[(t, "id", "sr")][METHODS.index(m)] for t in PERF]),
        "OOD_NS_mean3": lambda m: np.mean([FIG3_LABELS[(t, "ood", "ns")][METHODS.index(m)] for t in PERF]),
        "OOD_SR_mean3": lambda m: np.mean([FIG3_LABELS[(t, "ood", "sr")][METHODS.index(m)] for t in PERF]),
        "ID_SR_mean_cup_bag": lambda m: np.mean([FIG3_LABELS[(t, "id", "sr")][METHODS.index(m)] for t in ("cup", "bag")]),
    }.items():
        cand[m][name] = r(fn(m), 1)
RES["V8_fig1_hypotheses"] = {"fig1": FIG1, "candidates_from_fig3_labels": cand,
                             "note": "Fig. 1의 Spatial/Object/Scene 값의 산출식은 본문에 없다. 일부 칸만 Fig. 3 한 셀과 일치한다."}

# ---------------------------------------------------------------------------
# V9. Table 3 데이터 비율
# ---------------------------------------------------------------------------
T3 = {"cup": (300, 2.5, 2.0, 20.5), "fold": (360, 3.0, 2.0, 21.0), "bag": (300, 2.5, 2.0, 7.0)}
v9 = {}
tot_r = tot_h = 0
for t, (d, hr, hi, he) in T3.items():
    v9[t] = {"robot_demo_per_hour": r(d / hr, 1), "in_domain_ratio": r(hi / hr, 2), "egoverse_ratio": r(he / hr, 2)}
    tot_r += hr
    tot_h += he
v9["overall_egoverse_ratio_by_hours"] = r(tot_h / tot_r, 2)
v9["frames_robot_cup_30hz"] = int(2.5 * 3600 * 30)
v9["frames_egoverse_cup_30hz"] = int(20.5 * 3600 * 30)
# 학습량: 100 step x 2000 epoch x 배치 32+32
steps = 100 * 2000
v9["train_steps"] = steps
v9["robot_samples_seen"] = steps * 32
v9["human_samples_seen"] = steps * 32
v9["robot_passes_cup"] = r(steps * 32 / v9["frames_robot_cup_30hz"], 1)
v9["egoverse_passes_cup"] = r(steps * 32 / v9["frames_egoverse_cup_30hz"], 2)
v9["in_domain_passes_cup"] = r(steps * 32 / int(2 * 3600 * 30), 1)
v9["steps_per_second_for_two_days"] = r(steps / (48 * 3600), 2)
v9["episodes_486_minutes_each_if_20.5h"] = r(20.5 * 60 / 486, 2)
RES["V9_data_composition"] = v9

# ---------------------------------------------------------------------------
# V10. 절제 (Fig. 8, Fig. 9, Fig. 11)
# ---------------------------------------------------------------------------
def f(k1, n1, k0, n0):
    return r(fisher(k1, n1, k0, n0), 4)


v10 = {
    "fig8a_bag_unaligned": {
        "BC_in_domain_9of20_vs_unaligned_4of20_p": f(4, 20, 9, 20),
        "BC_robot_only_8of20_vs_unaligned_4of20_p": f(4, 20, 8, 20),
        "Flow_in_domain_16of20_vs_unaligned_15of20_p": f(15, 20, 16, 20),
        "Flow_robot_only_12of20_vs_unaligned_15of20_p": f(15, 20, 12, 20),
        "BC_unaligned_4of20_vs_Flow_unaligned_15of20_p": f(15, 20, 4, 20),
    },
    "fig8b_cup_aligned": {
        "BC_in_domain_3of20_vs_aligned_7of20_p": f(7, 20, 3, 20),
        "BC_robot_only_5of20_vs_aligned_7of20_p": f(7, 20, 5, 20),
        "Pixel_7of20_vs_13of20_p": f(13, 20, 7, 20),
        "DINO_10of20_vs_14of20_p": f(14, 20, 10, 20),
        "Flow_17of20_vs_17of20_p": f(17, 20, 17, 20),
    },
    "fig9_cup": {
        "ID_action_3of20_vs_flowonly_7of20_p": f(7, 20, 3, 20),
        "OODobj_action_1of9_vs_flowonly_4of9_p": f(4, 9, 1, 9),
        "OODscene_action_0of10_vs_flowonly_1of10_p": f(1, 10, 0, 10),
        "ID_flowonly_7of20_vs_both_18of20_p": f(18, 20, 7, 20),
        "OODobj_flowonly_4of9_vs_both_7of9_p": f(7, 9, 4, 9),
        "OODscene_flowonly_1of10_vs_both_3of10_p": f(3, 10, 1, 10),
        "ID_action_3of20_vs_both_18of20_p": f(18, 20, 3, 20),
        "OODobj_action_1of9_vs_both_7of9_p": f(7, 9, 1, 9),
        "OODscene_action_0of10_vs_both_3of10_p": f(3, 10, 0, 10),
    },
    "fig11_pixelpt_dense": {
        "ID_SR_dense_robot_7of20_vs_egoverse_9of20_p": f(9, 20, 7, 20),
        "ID_SR_pixelpt_robot_8of20_vs_egoverse_10of20_p": f(10, 20, 8, 20),
        "dense_egoverse_9of20_vs_dino_14of20_p(cup ID, Fig. 3)": f(14, 20, 9, 20),
        "dense_egoverse_9of20_vs_flow_18of20_p(cup ID, Fig. 3)": f(18, 20, 9, 20),
    },
}
# Table 1: 손실 0.23 vs 0.22, 0.20 vs 0.19 (소수 2자리). 반올림만으로 생길 수 있는 상대 차이 범위.
v10["table1_rounding"] = {
    "human_rel_diff_pct_nominal": r(100 * (0.23 - 0.22) / 0.23, 1),
    "human_rel_diff_pct_range": [r(100 * (0.225 - 0.225) / 0.225, 1), r(100 * ((0.235 - 0.215) / 0.235), 1)],
    "robot_rel_diff_pct_nominal": r(100 * (0.20 - 0.19) / 0.20, 1),
    "robot_rel_diff_pct_range": [0.0, r(100 * ((0.205 - 0.185) / 0.205), 1)],
    "note": "손실은 소수 둘째 자리, 시드·표준편차 없음. 한 번의 실행이면 0.01 차이는 반올림 안에 든다.",
}
RES["V10_ablations"] = v10

# ---------------------------------------------------------------------------
# V11. RoboTwin Table 4 (100 시드) 재계산과 v1 대 v2
# ---------------------------------------------------------------------------
T4_V2 = {  # [ACT-EE, DP-EE, BC s, BC c, Pixel s, Pixel c, DINO s, DINO c, Flow s, Flow c]
    "pick": [2, 5, 2, 9, 6, 17, 3, 28, 1, 19],
    "stack": [0, 0, 0, 0, 0, 0, 0, 8, 0, 16],
    "mug": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
}
T4_V1 = {"pick": [2, 5, 2, 6, 7, 11, 4, 28, 0, 16], "stack": [0, 0, 0, 0, 0, 0, 0, 8, 0, 16], "mug": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]}
cols = ["ACT", "DP", "BC_s", "BC_c", "Pix_s", "Pix_c", "DINO_s", "DINO_c", "Flow_s", "Flow_c"]
v11 = {"single_to_cross": {}, "wm_vs_bc_cross": {}, "v1_vs_v2": {}}
for t, row in T4_V2.items():
    d = dict(zip(cols, row))
    v11["single_to_cross"][t] = {k: {"s": d[k + "_s"], "c": d[k + "_c"], "p": f(d[k + "_c"], 100, d[k + "_s"], 100)} for k in ("BC", "Pix", "DINO", "Flow")}
    v11["wm_vs_bc_cross"][t] = {k: {"c": d[k + "_c"], "bc_c": d["BC_c"], "p": f(d[k + "_c"], 100, d["BC_c"], 100)} for k in ("Pix", "DINO", "Flow")}
for t in T4_V2:
    diff = {}
    for c, a, b in zip(cols, T4_V1[t], T4_V2[t]):
        if a != b:
            diff[c] = {"v1": a, "v2": b, "p": f(b, 100, a, 100)}
    v11["v1_vs_v2"][t] = diff
v11["pick_dino_vs_flow_cross_p"] = f(28, 100, 19, 100)
v11["pick_pixel_vs_flow_cross_p"] = f(19, 100, 17, 100)
v11["claim_every_variant_improves_single_to_cross"] = {
    t: [k for k in ("BC", "Pix", "DINO", "Flow") if T4_V2[t][cols.index(k + "_c")] <= T4_V2[t][cols.index(k + "_s")]]
    for t in T4_V2}
RES["V11_robotwin"] = v11

# ---------------------------------------------------------------------------
# V12. 헤드 파라미터 수 (공식 config의 구조식으로 계산, 코드 실행 없음)
# ---------------------------------------------------------------------------
def dit_decoder_params(C, hidden=384, blocks=6, patch=2, time_dim=256, obs_dim=256, max_cams=3):
    lin = lambda i, o: i * o + o
    p = 0
    p += lin(time_dim, 4 * time_dim) + lin(4 * time_dim, time_dim)          # time_mlp
    p += max_cams * time_dim                                                 # camera_embed
    p += lin(obs_dim, time_dim) + lin(time_dim, time_dim)                    # cond_proj
    p += lin(2 * time_dim, hidden) + lin(hidden, hidden)                     # cond_vec_mlp
    p += C * patch * patch * hidden + hidden                                 # patch_embed (conv)
    attn = lin(hidden, 3 * hidden) + lin(hidden, hidden)                     # MultiheadAttention in/out proj
    mlp = lin(hidden, 4 * hidden) + lin(4 * hidden, hidden)
    ada = lin(hidden, 6 * hidden)
    p += blocks * (attn + mlp + ada)
    p += lin(hidden, 2 * hidden) + lin(hidden, C * patch * patch)            # final layer
    return p


def point_flow_params(num_h=100, D=3, hidden=256, blocks=4, cond=256, time_dim=256, mlp_ratio=4.0, mlp_layers=2):
    lin = lambda i, o: i * o + o
    p = lin(D, hidden) + lin(num_h * D, hidden)                              # anchor_proj, noise_proj
    p += lin(time_dim, hidden) + lin(hidden, hidden)                         # time_mlp
    m = int(hidden * mlp_ratio)
    mlp = lin(hidden, m) + (mlp_layers - 1) * lin(m, m) + lin(m, hidden)
    blk = 2 * lin(hidden, 3 * hidden) // 1  # placeholder, 아래서 덮어씀
    attn = lin(hidden, 3 * hidden) + lin(hidden, hidden)                     # MultiheadAttention (self와 cross 모두 in/out proj)
    blk = 2 * attn + 3 * 2 * hidden + mlp + lin(cond, hidden)                # mha + cmha + 3 LayerNorm + mlp + cond_proj
    p += blocks * blk
    p += lin(hidden, num_h * D)                                              # out_proj
    return p


def ddt_head_params(dim=2048, blocks=2, cond=384, mlp_ratio=4.0):
    # DiT-DH의 넓은 DDT 헤드를 단순 DiT 블록(attn 4d^2 + mlp 8d^2 + adaLN 6d*cond)으로 어림한 하한
    attn = 4 * dim * dim
    mlp = 2 * mlp_ratio * dim * dim
    ada = 6 * dim * cond
    return int(blocks * (attn + mlp + ada))


v12 = {
    "pixel_dit_16ch_params": dit_decoder_params(16),
    "dino_dit_768ch_params": dit_decoder_params(768),
    "flow_head_params": point_flow_params(),
    "paper_claim": {"pixel": 17.3e6, "dino": 19.6e6, "flow": 9.0e6, "pixelpt": 1.4e9},
    "wide_ddt_head_2x2048_lower_bound_params": ddt_head_params(),
}
v12["dino_minus_pixel_params"] = v12["dino_dit_768ch_params"] - v12["pixel_dit_16ch_params"]
v12["dino_with_wide_ddt_total_lower_bound"] = v12["dino_dit_768ch_params"] + v12["wide_ddt_head_2x2048_lower_bound_params"]
RES["V12_head_params"] = v12

# ---------------------------------------------------------------------------
# V13. 학습률 일정: CosineAnnealingLR(T_max=1400, eta_min=1e-5), 에폭 단위로 2000 에폭
# ---------------------------------------------------------------------------
def cos_lr(ep, base=1e-4, eta=1e-5, tmax=1400):
    return eta + (base - eta) * (1 + math.cos(math.pi * ep / tmax)) / 2


RES["V13_lr_schedule"] = {
    f"epoch_{e}": r(cos_lr(e), 7) for e in (0, 700, 999, 1399, 1400, 1799, 1999, 2000)
}
RES["V13_lr_schedule"]["note"] = "T_max=1400 이후 PyTorch CosineAnnealingLR은 다시 올라간다(주기 2800). 논문은 학습 에폭 2000과 T_max 1400을 함께 적는다."

# ---------------------------------------------------------------------------
# V14. 흐름 시간 표본 Beta(1.5, 1)의 방향: 논문 식 대 코드
# ---------------------------------------------------------------------------
a, b = 1.5, 1.0
mean_tau = a / (a + b)
RES["V14_flow_time_convention"] = {
    "beta_mean": r(mean_tau, 3),
    "paper_literal_noise_fraction_mean": r(1 - mean_tau, 3),   # a_tau = (1-tau) eps + tau a, tau가 1이면 깨끗한 데이터
    "code_noise_fraction_mean": r(mean_tau, 3),                # x_t = t*noise + (1-t)*actions, t가 1이면 잡음
    "note": "논문 식 그대로면 평균 잡음 비율 0.4, 공식 코드는 0.6 (pi0와 같은 방향).",
}

# ---------------------------------------------------------------------------
# V15. A.5 카메라 포즈 오차가 3D 흐름 임계값에 주는 영향 (기하 어림)
# ---------------------------------------------------------------------------
th_h = 0.010   # 사람 이동 임계값 10 mm
rot_err_deg = 1.9
true_rot_deg = 4.7
trans_err = 0.018
true_trans = 0.025
depths = [0.3, 0.5, 0.8, 1.2]
v15 = {
    "rot_err_displacement_mm": {str(d): r(1000 * d * math.tan(math.radians(rot_err_deg)), 1) for d in depths},
    "true_rot_raw_flow_mm": {str(d): r(1000 * d * math.tan(math.radians(true_rot_deg)), 1) for d in depths},
    "depth_where_rot_err_exceeds_10mm_m": r(th_h / math.tan(math.radians(rot_err_deg)), 3),
    "rot_residual_fraction": r(rot_err_deg / true_rot_deg, 2),
    "trans_residual_fraction": r(trans_err / true_trans, 2),
    "trans_err_vs_threshold_ratio": r(trans_err / th_h, 2),
    "ate_path_length_cm_from_6.8pct": r(1.4 / 0.068, 1),
}
RES["V15_pose_error_vs_threshold"] = v15

# ---------------------------------------------------------------------------
# V16. 표본 크기: 검출 가능한 차이 (n=20, 두 표본 비율, 정규 근사, alpha .05, 검정력 .8)
# ---------------------------------------------------------------------------
def n_per_group(p1, p2, alpha=0.05, power=0.8):
    za = stats.norm.ppf(1 - alpha / 2)
    zb = stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p2 - p1) ** 2)


def mde(n, p1=0.25, alpha=0.05, power=0.8):
    for d in np.arange(0.05, 0.95, 0.01):
        if p1 + d <= 1 and n_per_group(p1, p1 + d, alpha, power) <= n:
            return r(d, 2)
    return None


RES["V16_power"] = {
    "n_per_group_25_to_45": n_per_group(.25, .45), "n_per_group_25_to_50": n_per_group(.25, .50),
    "n_per_group_40_to_60": n_per_group(.40, .60), "n_per_group_50_to_70": n_per_group(.50, .70),
    "mde_at_n20_from_25pct": mde(20, .25), "mde_at_n20_from_50pct": mde(20, .50),
    "mde_at_n100_from_10pct": mde(100, .10),
}

# ---------------------------------------------------------------------------
# V17. 부록 A.4 문구 점검: "Pixel-PT는 EgoVerse 공동 학습에서 DINO를 넘은 적이 없고 3D Flow를 넘은 것은 열두 지표 중 하나"
# ---------------------------------------------------------------------------
beats_dino, beats_flow, ties_dino, ties_flow = [], [], 0, 0
for task in PERF:
    for si, split in enumerate(("id", "ood")):
        for mi, met in enumerate(("ns", "sr")):
            pt = PERF[task]["pixelpt"]["flagship"][si][mi]
            di = PERF[task]["dino"]["flagship"][si][mi]
            fl = PERF[task]["flow"]["flagship"][si][mi]
            if pt > di + 1e-9:
                beats_dino.append((task, split, met))
            if pt > fl + 1e-9:
                beats_flow.append((task, split, met))
            ties_dino += int(abs(pt - di) < 1e-9)
            ties_flow += int(abs(pt - fl) < 1e-9)
RES["V17_a4_pixelpt_claim"] = {"metrics": 12, "pixelpt_beats_dino": beats_dino, "pixelpt_beats_flow": beats_flow,
                               "ties_with_dino": ties_dino, "ties_with_flow": ties_flow}

# ---------------------------------------------------------------------------
# 저장과 요약
# ---------------------------------------------------------------------------
OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")


def show(title, obj):
    print("==", title)
    print(json.dumps(obj, ensure_ascii=False, indent=1)[:2600])


if __name__ == "__main__":
    show("V1", {k: v for k, v in RES["V1_fig3_label_vs_bars"].items() if k != "rows_not_exact_round"})
    print("V1 rows not exact round:", [(x["task"], x["split"], x["metric"], x["method"], x["label"], x["computed"]) for x in RES["V1_fig3_label_vs_bars"]["rows_not_exact_round"]])
    show("V2", RES["V2_integer_feasibility"])
    show("V3", RES["V3_rollout_count"])
    print("== V4 (sr, Fisher)")
    for x in RES["V4_sr_base_vs_egoverse"]:
        print(x["task"], x["split"], x["method"], f'{x["k_base"]}/{x["n_base"]} -> {x["k_flag"]}/{x["n_flag"]}', x["delta_pp"], "p=", x["fisher_p"])
    show("V6b", RES["V6b_dino_vs_flow_egoverse"])
    show("V7", RES["V7_abstract_claims"])
    show("V8", RES["V8_fig1_hypotheses"])
    show("V9", RES["V9_data_composition"])
    show("V10", RES["V10_ablations"])
    show("V11", RES["V11_robotwin"])
    show("V12", RES["V12_head_params"])
    show("V13", RES["V13_lr_schedule"])
    show("V14", RES["V14_flow_time_convention"])
    show("V15", RES["V15_pose_error_vs_threshold"])
    show("V16", RES["V16_power"])
