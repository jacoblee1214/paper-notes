"""논문 표(Table 1-5, 부록 8.5.0/8.5.3)의 일관성 검사와 재계산. numpy, scipy만 쓴다(1초 안팎).

실행: python3 analysis/table_check.py  ->  analysis/table_check_results.json
논문: Bodmer 외, arXiv:2606.24208 v1. 값은 paper/paper.txt에서 옮겼다. 검사 번호는 V1-V12.
"""
import json
import math
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("table_check_results.json")
R = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [100 * (c - h) / d, 100 * (c + h) / d]


def two_prop_p(p1, p2, n):
    """독립 두 비율(같은 n)의 양측 z 검정. p는 %."""
    a, b = p1 / 100, p2 / 100
    pool = (a + b) / 2
    se = math.sqrt(2 * pool * (1 - pool) / n)
    z = (a - b) / se
    return 2 * (1 - stats.norm.cdf(abs(z)))


def feasible_counts(pct, n, tol=0.05):
    """pct(소수 1자리 %)가 n개 중 k개의 비율로 반올림되어 나올 수 있는 k 목록."""
    ks = [k for k in range(n + 1) if abs(100 * k / n - pct) <= tol + 1e-9]
    return ks


# ---------------------------------------------------------------- Table 1
T1 = {  # (SR1, SRIK, Efc, Q1)
    "Dynaarm": {"DDIM†": (75.0, None, 0.01, 16.0), "DDIM": (27.0, 54.6, 2.37, 6.5),
                "Projection": (23.7, 99.9, 0.72, 6.8), "Gradient": (58.8, 99.3, 0.97, 12.80),
                "Theseus": (63.5, 95.8, 0.26, 14.5), "IPOPT": (69.8, 96.4, 0.20, 15.2)},
    "Franka": {"DDIM": (33.7, 83.7, 2.02, 7.1), "Projection": (37.4, 100.0, 1.19, 8.1),
               "Gradient": (50.9, 97.9, 1.04, 11.1), "Theseus": (61.0, 98.3, 0.23, 14.1),
               "IPOPT": (71.0, 99.8, 0.06, 15.7)},
}
# V1: 정수 가능성. 단위 후보 n=1200(격자 1200개), n=150(객체 30 x 자세 5)
v1 = {}
for arm, rows in T1.items():
    for m, (sr1, srik, *_rest) in rows.items():
        for name, val in (("SR1", sr1), ("SRIK", srik)):
            if val is None:
                continue
            for n in (1200, 150):
                ks = feasible_counts(val, n)
                v1[f"{arm}/{m}/{name}/n{n}"] = ks[:3] if ks else None
infeasible_150 = sorted(k for k, v in v1.items() if k.endswith("n150") and not v)
infeasible_1200 = sorted(k for k, v in v1.items() if k.endswith("n1200") and not v)
R["V1_integer_feasibility"] = {"n150_infeasible_cells": infeasible_150, "n1200_infeasible_cells": infeasible_1200}

# V2: 부록 8.5.3(쉬운/어려운 자세별 SR1)의 가중 평균 대 표 1 (쉬운 3, 어려운 2 자세 가정)
APP = {"Franka": {"Gradient": (70, 23), "Projection": (57, 9), "IPOPT": (72, 70), "Theseus": (65, 56)},
       "Dynaarm": {"Gradient": (59, 58), "Projection": (28, 17), "IPOPT": (71, 68), "Theseus": (61, 67)}}
v2 = {}
for arm, rows in APP.items():
    for m, (easy, hard) in rows.items():
        est = 0.6 * easy + 0.4 * hard
        v2[f"{arm}/{m}"] = {"easy": easy, "hard": hard, "weighted_3to2": round(est, 2),
                            "table1": T1[arm][m][0], "diff": round(est - T1[arm][m][0], 2)}
R["V2_easy_hard_vs_table1"] = v2
R["V2_max_abs_diff"] = max(abs(v["diff"]) for v in v2.values())

# V3: 실행 시간 배수
R["V3_runtime_ratio"] = {"Theseus/DDIM": round(3.63 / 0.0793, 1), "IPOPT/DDIM": round(5.47 / 0.0793, 1),
                         "gradient/DDIM": round(84.2 / 79.3, 3), "projection/DDIM": round(84.1 / 79.3, 3)}

# V4: 초록과 본문의 "최대 +pp" 주장
R["V4_gain_claims"] = {
    "grasp_Dynaarm_IPOPT_minus_Gradient": round(69.8 - 58.8, 1),
    "grasp_Franka_IPOPT_minus_Gradient": round(71.0 - 50.9, 1),
    "visuomotor_Franka_PP_ours_minus_best_baseline": round(67.0 - max(44.0, 41.0, 43.3), 1),
    "visuomotor_Dynaarm_PP_ours_minus_best_baseline": round(41.8 - max(26.0, 29.3, 26.3), 1),
    "visuomotor_Drawer_Dynaarm_ours_minus_best_baseline": round(81.8 - max(79.5, 79.8, 79.8), 1),
    "visuomotor_Drawer_Franka_ours_minus_best_baseline": round(60.0 - max(57.8, 57.8, 58.0), 1),
}

# ---------------------------------------------------------------- Table 2 / 3
M = ["DDIM†", "DDIM", "Grad(σ)", "Grad(xt)", "Ours"]
# (task, task_sd, sub, sub_sd) per arm
T2 = {"Drawer": {"Dynaarm": [(99.5, 1.0, 100.0, 0.0), (79.5, 1.9, 84.0, 0.8), (79.8, 1.7, 85.0, 1.6), (79.8, 3.3, 83.8, 3.0), (81.8, 2.9, 87.3, 2.9)],
                 "Franka": [(99.5, 1.0, 100.0, 0.0), (57.8, 2.2, 66.3, 3.1), (57.8, 7.8, 66.8, 6.8), (58.0, 2.4, 67.0, 2.8), (60.0, 3.4, 70.0, 1.6)]},
      "PickPlace": {"Dynaarm": [(62.5, 6.2, 95.3, 2.1), (26.0, 2.0, 63.3, 3.3), (29.3, 4.3, 69.3, 4.6), (26.3, 3.4, 63.3, 5.0), (41.8, 3.8, 66.0, 3.6)],
                    "Franka": [(62.5, 6.2, 95.3, 2.1), (44.0, 4.5, 89.0, 3.4), (41.0, 1.4, 92.3, 1.3), (43.3, 1.7, 90.3, 3.8), (67.0, 3.4, 98.3, 1.3)]}}
T3 = {"Drawer": {"Dynaarm": [(99.5, 1.0, 100.0, 0.0), (79.5, 1.9, 84.0, 0.8), (79.8, 1.7, 85.0, 1.6), (79.8, 3.3, 83.8, 3.0), (81.8, 2.9, 87.3, 2.9)],
                 "Franka": [(99.5, 1.0, 100.0, 0.0), (57.8, 2.2, 66.3, 3.1), (57.8, 7.8, 66.8, 6.8), (58.0, 2.4, 67.0, 2.8), (60.0, 3.4, 70.0, 1.6)]},
      "PickPlace": {"Dynaarm": [(62.5, 6.2, 95.3, 2.1), (26.0, 2.0, 63.2, 3.3), (29.3, 4.3, 69.3, 4.6), (26.3, 3.4, 63.2, 5.0), (41.8, 3.8, 66.0, 3.6)],
                    "Franka": [(62.5, 6.2, 95.3, 2.1), (44.0, 4.5, 89.0, 3.4), (41.0, 1.4, 92.3, 1.3), (43.3, 1.7, 90.3, 3.8), (67.0, 3.4, 98.3, 1.3)]}}
v5 = []
for t in T2:
    for a in T2[t]:
        for i, m in enumerate(M):
            if T2[t][a][i] != T3[t][a][i]:
                v5.append({"task": t, "arm": a, "method": m, "table2": T2[t][a][i], "table3": T3[t][a][i]})
R["V5_table2_vs_table3_diff_cells"] = v5

# V6: 4 run x 100 에피소드로 (평균, SD)가 만들어지는가
runs = np.arange(0, 101)
A, B, C = np.meshgrid(runs, runs, runs, indexing="ij", sparse=True)


def feasible_mean_sd(mean, sd, nrun=4):
    out = {"ddof0": False, "ddof1": False, "mean_only": False}
    for s in range(0, 100 * nrun + 1):
        if abs(s / nrun - mean) > 0.05 + 1e-9:
            continue
        out["mean_only"] = True
        D = s - A - B - C
        ok = (D >= 0) & (D <= 100)
        arr = np.stack(np.broadcast_arrays(A, B, C, np.where(ok, D, 0)), axis=-1)[ok].astype(float)
        if arr.size == 0:
            continue
        for ddof, key in ((0, "ddof0"), (1, "ddof1")):
            sdv = arr.std(axis=1, ddof=ddof)
            if np.any(np.abs(sdv - sd) <= 0.05 + 1e-9):
                out[key] = True
    return out


v6 = {}
bad_mean, bad_sd0, bad_sd1, bad_both = [], [], [], []
for t in T2:
    for a in T2[t]:
        for i, m in enumerate(M):
            for j, nm in ((0, "task"), (2, "sub")):
                mean, sd = T2[t][a][i][j], T2[t][a][i][j + 1]
                key = f"{t}/{a}/{m}/{nm}"
                f = feasible_mean_sd(mean, sd)
                v6[key] = f
                if not f["mean_only"]:
                    bad_mean.append(key)
                if not f["ddof0"]:
                    bad_sd0.append(key)
                if not f["ddof1"]:
                    bad_sd1.append(key)
                if not f["ddof0"] and not f["ddof1"]:
                    bad_both.append(key)
R["V6_four_run_granularity"] = {"cells": len(v6), "mean_not_multiple_of_0.25": bad_mean,
                                "sd_infeasible_ddof0": bad_sd0, "sd_infeasible_ddof1": bad_sd1,
                                "sd_infeasible_both": bad_both}

# V7: 검정. (a) 4개 run 평균의 Welch t (b) 에피소드를 400개로 합친 비율 검정(가정)
v7 = {}
for t in T2:
    for a in T2[t]:
        rows = T2[t][a]
        ours = rows[4]
        base_i = max((1, 2, 3), key=lambda i: rows[i][0])
        base = rows[base_i]
        tt = stats.ttest_ind_from_stats(ours[0], ours[1], 4, base[0], base[1], 4, equal_var=False)
        pooled_p = two_prop_p(ours[0], base[0], 400)
        v7[f"{t}/{a}/task"] = {"ours": ours[:2], "best_baseline": M[base_i], "baseline": base[:2],
                               "diff_pp": round(ours[0] - base[0], 2), "welch_p_runs_n4": round(float(tt.pvalue), 4),
                               "two_prop_p_n400_assumed": float(f"{pooled_p:.3g}")}
        base_i = max((1, 2, 3), key=lambda i: rows[i][2])
        base = rows[base_i]
        tt = stats.ttest_ind_from_stats(ours[2], ours[3], 4, base[2], base[3], 4, equal_var=False)
        v7[f"{t}/{a}/sub"] = {"ours": ours[2:], "best_baseline": M[base_i], "baseline": base[2:],
                              "diff_pp": round(ours[2] - base[2], 2), "welch_p_runs_n4": round(float(tt.pvalue), 4),
                              "two_prop_p_n400_assumed": float(f"{two_prop_p(ours[2], base[2], 400):.3g}")}
R["V7_visuomotor_tests"] = v7

# V8: Table 1 SR1 검정 (단위 n=150, n=1200 가정, 독립 비율 근사)
v8 = {}
for arm, pairs in (("Dynaarm", [("IPOPT", "Gradient"), ("Theseus", "Gradient"), ("IPOPT", "Theseus")]),
                   ("Franka", [("IPOPT", "Gradient"), ("Theseus", "Gradient"), ("IPOPT", "Theseus")])):
    for a, b in pairs:
        pa, pb = T1[arm][a][0], T1[arm][b][0]
        v8[f"{arm}/{a}-{b}"] = {"diff_pp": round(pa - pb, 1), "p_n150": float(f"{two_prop_p(pa, pb, 150):.3g}"),
                                "p_n1200": float(f"{two_prop_p(pa, pb, 1200):.3g}")}
v8["wilson95_IPOPT_Dynaarm_n150"] = [round(x, 1) for x in wilson(round(0.698 * 150), 150)]
v8["wilson95_Gradient_Dynaarm_n150"] = [round(x, 1) for x in wilson(round(0.588 * 150), 150)]
R["V8_table1_tests"] = v8

# ---------------------------------------------------------------- Table 5
ENVS = ["Floor", "Walls", "Clutter", "Tunnels"]
T5 = {  # method: [(SRTot, SR1, SRIK, CR) x4]
    "Dynaarm": {"DDIM": [(5.0, 16.25, 32.5, 60.0), (2.5, 16.25, 32.5, 62.5), (4.37, 19.99, 32.5, 67.5), (0.0, 16.25, 32.5, 80.0)],
                "Gradient": [(33.12, 33.12, 100, 0.0), (17.5, 19.99, 100, 25.0), (28.12, 28.12, 100, 0.0), (30.63, 30.62, 100, 0.0)],
                "Projection": [(19.38, 22.49, 100, 20.0), (0.63, 5.62, 100, 20.0), (0.0, 0.0, 75.0, 0.0), (14.37, 19.37, 100, 22.5)],
                "Theseus": [(36.88, 36.87, 77.5, 0.0), (35.63, 44.36, 77.5, 17.5), (23.13, 28.37, 82.5, 37.5), (24.37, 46.24, 80.0, 32.5)],
                "IPOPT": [(46.88, 51.24, 75.0, 10.0), (22.5, 43.74, 75.0, 32.5), (31.25, 39.99, 65.0, 20.0), (23.75, 26.24, 52.5, 32.5)]},
    "Franka": {"DDIM": [(11.87, 68.73, 100, 82.5), (10.0, 68.73, 100, 85.0), (6.25, 60.61, 100, 90.0), (7.5, 68.73, 100, 87.5)],
               "Gradient": [(22.5, 22.49, 100, 0.0), (20.62, 20.62, 100, 30.0), (23.13, 23.74, 100, 17.5), (18.12, 18.12, 100, 5.0)],
               "Projection": [(58.13, 60.61, 100, 2.5), (43.13, 45.61, 100, 2.5), (0.0, 0.0, 57.5, 0.0), (45.0, 47.49, 100, 10.0)],
               "Theseus": [(58.75, 58.74, 100, 0.0), (50.62, 55.61, 100, 17.5), (26.87, 38.12, 97.5, 27.5), (38.75, 59.36, 100, 30.0)],
               "IPOPT": [(65.62, 69.98, 100, 12.5), (60.62, 76.86, 100, 20.0), (55.63, 70.61, 100, 27.5), (61.88, 68.11, 100, 17.5)]},
}
v9 = {"denominators": {}, "truncation": [], "SRTot_gt_SR1": [], "SRTot_gt_100_minus_CR": [], "per_arm_best": {}, "mean_over_envs": {}}
allvals = [v for arm in T5.values() for rows in arm.values() for cell in rows for v in cell]
for n in (40, 160, 320, 400, 640):
    miss = 0
    for v in allvals:
        k = v * n / 100
        if abs(k - round(k)) > 0.02 * n / 100 + 1e-9 and v not in (0.0, 100):  # 0.02pp 허용(절단 표기)
            miss += 1
    v9["denominators"][f"n{n}_values_not_integer_count"] = miss
v9["denominators"]["total_values"] = len(allvals)
for arm, rows in T5.items():
    for m, cells in rows.items():
        for e, (tot, s1, ik, cr) in zip(ENVS, cells):
            k = tot * 160 / 100
            nearest = round(k) * 100 / 160
            if abs(nearest - tot) > 1e-9 and abs(nearest - tot) <= 0.02 + 1e-9 and nearest > tot:
                v9["truncation"].append(f"{arm}/{m}/{e}: {tot} (n=160의 {round(k)}/160 = {nearest:.3f})")
            if tot > s1 + 0.011:
                v9["SRTot_gt_SR1"].append(f"{arm}/{m}/{e}: {tot} > {s1}")
            if tot > 100 - cr + 0.011:
                v9["SRTot_gt_100_minus_CR"].append(f"{arm}/{m}/{e}: {tot} > {100 - cr}")
        v9["mean_over_envs"][f"{arm}/{m}"] = round(sum(c[0] for c in cells) / 4, 2)
    for ei, e in enumerate(ENVS):
        best = max(rows, key=lambda m: rows[m][ei][0])
        v9["per_arm_best"][f"{arm}/{e}"] = [best, rows[best][ei][0]]
R["V9_table5"] = v9
# V9b: 환경별 SRTot 검정 (n=160 가정, IPOPT 또는 Theseus 대 최고 기준선)
v9b = {}
for arm, rows in T5.items():
    for ei, e in enumerate(ENVS):
        base = max(("Gradient", "Projection"), key=lambda m: rows[m][ei][0])
        ours = max(("Theseus", "IPOPT"), key=lambda m: rows[m][ei][0])
        a, b = rows[ours][ei][0], rows[base][ei][0]
        v9b[f"{arm}/{e}"] = {"ours": [ours, a], "best_baseline": [base, b], "diff_pp": round(a - b, 2),
                             "p_n160_assumed": float(f"{two_prop_p(a, b, 160):.3g}"),
                             "p_n40_literal": float(f"{two_prop_p(a, b, 40):.3g}")}
R["V9b_table5_tests"] = v9b

# ---------------------------------------------------------------- Table 4 / 부록 8.5.0
def mlp_params(dof):
    lin = lambda i, o: i * o + o
    return (lin(9, 256) + 2 * lin(256, 256)            # 임베딩 + 은닉 2
            + lin(256, 128) + lin(128, dof)            # 관절 머리
            + lin(256, 64) + lin(64, 1)                # 도달 머리
            + lin(256, 128) + lin(128, 9))             # 자세 보정 머리


R["V10_surrogate_params"] = {"dof6": mlp_params(6), "dof7": mlp_params(7), "paper": {"Dynaarm_6dof": "≈218k", "Franka_7dof": "≈219k", "total": "≈2.2e5"}}

# V11: DDIM†가 "상한"인데 안내된 방법이 넘는 칸
v11 = []
for t, arm in (("PickPlace", "Franka"), ("PickPlace", "Dynaarm")):
    up = T2[t][arm][0]
    o = T2[t][arm][4]
    v11.append({"task": t, "arm": arm, "SRtask": {"DDIM†": up[:2], "ours": o[:2]}, "SRsub": {"DDIM†": up[2:], "ours": o[2:]},
                "welch_p_task": round(float(stats.ttest_ind_from_stats(o[0], o[1], 4, up[0], up[1], 4, equal_var=False).pvalue), 3),
                "welch_p_sub": round(float(stats.ttest_ind_from_stats(o[2], o[3], 4, up[2], up[3], 4, equal_var=False).pvalue), 3)})
R["V11_ours_vs_upper_bound"] = v11

# V12: 본문 5.1.1의 Q1 비교 (<=7.1 대 >=14)
R["V12_Q1_claim"] = {"snap_DDIM_Q1": [6.5, 7.1], "ours_Q1": [14.5, 15.2, 14.1, 15.7], "projection_Q1": [6.8, 8.1],
                     "projection_Efc_vs_DDIM": {"Dynaarm": [0.72, 2.37], "Franka": [1.19, 2.02]}}

# V13: 그림 6(원문 23쪽)에서 읽은 값(그림 판독 값, 정수 라벨 +-1). (SR1 바깥, SRall 안쪽), 쉬움/어려움
FIG6 = {"Dynaarm": {"easy": {"DDIM†": (75, 41), "DDIM": (31, 14), "Projection": (28, 9), "Gradient": (59, 28), "Theseus": (61, 30), "IPOPT": (71, 34)},
                    "hard": {"DDIM†": (75, 41), "DDIM": (21, 11), "Projection": (17, 5), "Gradient": (58, 27), "Theseus": (67, 32), "IPOPT": (68, 32)}},
        "Franka": {"easy": {"DDIM†": (76, 40), "DDIM": (51, 26), "Projection": (57, 25), "Gradient": (70, 36), "Theseus": (65, 33), "IPOPT": (72, 37)},
                   "hard": {"DDIM†": (76, 40), "DDIM": (8, 4), "Projection": (9, 3), "Gradient": (23, 12), "Theseus": (56, 26), "IPOPT": (70, 33)}}}
v13 = {}
for arm in FIG6:
    for m in FIG6[arm]["easy"]:
        e, h = FIG6[arm]["easy"][m], FIG6[arm]["hard"][m]
        v13[f"{arm}/{m}"] = {"SR1_weighted_3to2": round(0.6 * e[0] + 0.4 * h[0], 1), "SRall_weighted_3to2": round(0.6 * e[1] + 0.4 * h[1], 1)}
    for m in ("Theseus", "IPOPT"):
        v13[f"{arm}/{m}_minus_Gradient"] = {"SR1_gap_pp": round(v13[f"{arm}/{m}"]["SR1_weighted_3to2"] - v13[f"{arm}/Gradient"]["SR1_weighted_3to2"], 1),
                                           "SRall_gap_pp": round(v13[f"{arm}/{m}"]["SRall_weighted_3to2"] - v13[f"{arm}/Gradient"]["SRall_weighted_3to2"], 1)}
R["V13_fig6_strict_metric"] = v13
R["data"] = {"T1": {a: {m: list(v) for m, v in r.items()} for a, r in T1.items()}, "T2": T2, "T5": T5, "FIG6": FIG6}

# V14: 식 (25)의 상자 제약 -1 <= x_K <= 1, -1 <= delta_k <= 1 대 표준정규 표본(좌표당 P(|z| <= 1) = erf(1/sqrt2))
p1 = math.erf(1 / math.sqrt(2))
R["V14_box_vs_gaussian"] = {"p_coord_in_box": round(p1, 4), "p_xK_in_box_d21": float(f"{p1 ** 21:.3g}"),
                            "p_xK_in_box_d9": float(f"{p1 ** 9:.3g}"), "p_one_step_delta_in_box_d21": float(f"{p1 ** 21:.3g}"),
                            "note": "표본 x_K ~ N(0,I)가 상자 안에 있을 확률. 확산 모델의 정규화 공간이 N(0,I)라고 가정(논문은 정규화를 밝히지 않음)"}

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: R[k] for k in ("V1_integer_feasibility", "V2_max_abs_diff", "V3_runtime_ratio", "V4_gain_claims",
                                    "V5_table2_vs_table3_diff_cells", "V6_four_run_granularity", "V10_surrogate_params")}, ensure_ascii=False, indent=1))
print(json.dumps(R["V9_table5"], ensure_ascii=False, indent=1))
print(json.dumps(R["V7_visuomotor_tests"], ensure_ascii=False, indent=1))
print(json.dumps(R["V8_table1_tests"], ensure_ascii=False, indent=1))
print(json.dumps(R["V9b_table5_tests"], ensure_ascii=False, indent=1))
print(json.dumps(R["V11_ours_vs_upper_bound"], ensure_ascii=False, indent=1))
print(json.dumps(R["V13_fig6_strict_metric"], ensure_ascii=False, indent=1))
