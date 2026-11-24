"""BoxWAM(arXiv:2610.17526) 표 값 재계산과 일관성 검사 (numpy, scipy, 약 2초).

논문의 표 1–5, 7–9와 본문 수치를 옮겨 평균, 정수 가능성, 검정을 다시 계산한다.
시행 수는 논문이 밝힌 것(표 1: 3 시드 x 100회, 표 2·8·9: 과제당 100회 한 시드, 표 3: 50회, 절 4.3: 3 시드 x 50회)을 쓰고,
셀 단위 횟수는 "백분율 x 시행 수"로 복원한 가정값이다. 모든 검정은 시행을 독립으로 보며 학습 시드 변동은 포함하지 않는다.

실행: python3 analysis/recompute.py  -> analysis/recompute_results.json
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"

T1_TASKS = ["adjust_bottle", "beat_block_hammer", "click_alarmclock", "grab_roller", "handover_mic",
            "lift_pot", "open_laptop", "pick_dual_bottles", "turn_switch"]
# 표 1 (논문 7쪽): 방법 -> (평균, 과제 9개)
T1 = {
    "clean": {
        "DP": (55.4, [97.0, 42.0, 61.0, 98.0, 53.0, 39.0, 49.0, 24.0, 36.0]),
        "ACT": (60.4, [97.0, 56.0, 32.0, 94.0, 85.0, 88.0, 56.0, 31.0, 5.0]),
        "DP3": (81.2, [99.0, 72.0, 77.0, 98.0, 100.0, 97.0, 82.0, 60.0, 46.0]),
        "pi0": (71.4, [90.0, 43.0, 63.0, 96.0, 98.0, 84.0, 85.0, 57.0, 27.0]),
        "Flow": (73.2, [99.7, 63.0, 90.3, 99.7, 58.3, 69.3, 81.0, 47.7, 50.0]),
        "BoxWAM": (78.3, [97.7, 67.7, 95.0, 100.0, 65.3, 84.3, 80.0, 62.0, 52.3]),
    },
    "randomized": {
        "DP": (0.7, [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        "ACT": (6.3, [23.0, 3.0, 4.0, 25.0, 0.0, 0.0, 0.0, 0.0, 2.0]),
        "DP3": (5.1, [3.0, 8.0, 14.0, 2.0, 3.0, 0.0, 7.0, 1.0, 8.0]),
        "pi0": (33.1, [56.0, 21.0, 11.0, 80.0, 13.0, 36.0, 46.0, 12.0, 23.0]),
        "Flow": (32.6, [72.3, 18.7, 17.7, 77.7, 1.0, 6.3, 43.3, 18.3, 38.0]),
        "BoxWAM": (37.8, [88.7, 29.3, 29.3, 78.3, 0.0, 22.7, 40.3, 16.7, 34.7]),
    },
}
# 표 2 / 7
T2 = {"pi05": (52.6, 26.9, 55.3), "pi05+Box": (61.5, 34.0, 57.7)}
T8_CLEAN = {
    "adjust_bottle": (68.0, 94.0),
    "beat_block_hammer": (81.0, 86.0),
    "blocks_ranking_rgb": (6.0, 19.0),
    "blocks_ranking_size": (28.0, 46.0),
    "click_alarmclock": (89.0, 93.0),
    "click_bell": (78.0, 73.0),
    "dump_bin_bigbin": (30.0, 64.0),
    "grab_roller": (84.0, 87.0),
    "handover_block": (14.0, 15.0),
    "handover_mic": (61.0, 22.0),
    "hanging_mug": (13.0, 13.0),
    "lift_pot": (32.0, 64.0),
    "move_can_pot": (30.0, 39.0),
    "move_pillbottle_pad": (16.0, 36.0),
    "move_playingcard_away": (82.0, 67.0),
    "move_stapler_pad": (33.0, 26.0),
    "open_laptop": (87.0, 90.0),
    "open_microwave": (33.0, 80.0),
    "pick_diverse_bottles": (50.0, 68.0),
    "pick_dual_bottles": (66.0, 88.0),
    "place_a2b_left": (60.0, 67.0),
    "place_a2b_right": (55.0, 55.0),
    "place_bread_basket": (67.0, 84.0),
    "place_bread_skillet": (63.0, 77.0),
    "place_burger_fries": (65.0, 94.0),
    "place_can_basket": (27.0, 49.0),
    "place_cans_plasticbox": (60.0, 92.0),
    "place_container_plate": (83.0, 95.0),
    "place_dual_shoes": (2.0, 4.0),
    "place_empty_cup": (70.0, 90.0),
    "place_fan": (36.0, 36.0),
    "place_mouse_pad": (37.0, 49.0),
    "place_object_basket": (66.0, 67.0),
    "place_object_scale": (30.0, 48.0),
    "place_object_stand": (81.0, 83.0),
    "place_phone_stand": (37.0, 54.0),
    "place_shoe": (90.0, 93.0),
    "press_stapler": (89.0, 91.0),
    "put_bottles_dustbin": (58.0, 71.0),
    "put_object_cabinet": (50.0, 65.0),
    "rotate_qrcode": (39.0, 66.0),
    "scan_object": (42.0, 74.0),
    "shake_bottle_horizontally": (99.0, 100.0),
    "shake_bottle": (100.0, 100.0),
    "stack_blocks_three": (0.0, 0.0),
    "stack_blocks_two": (4.0, 7.0),
    "stack_bowls_three": (72.0, 60.0),
    "stack_bowls_two": (85.0, 84.0),
    "stamp_seal": (23.0, 18.0),
    "turn_switch": (57.0, 30.0),
}

T9_RAND = {
    "adjust_bottle": (15.0, 53.0),
    "beat_block_hammer": (20.0, 9.0),
    "blocks_ranking_rgb": (6.0, 9.0),
    "blocks_ranking_size": (10.0, 19.0),
    "click_alarmclock": (48.0, 62.0),
    "click_bell": (51.0, 50.0),
    "dump_bin_bigbin": (24.0, 39.0),
    "grab_roller": (54.0, 71.0),
    "handover_block": (1.0, 0.0),
    "handover_mic": (3.0, 0.0),
    "hanging_mug": (1.0, 0.0),
    "lift_pot": (1.0, 5.0),
    "move_can_pot": (1.0, 4.0),
    "move_pillbottle_pad": (3.0, 21.0),
    "move_playingcard_away": (12.0, 18.0),
    "move_stapler_pad": (3.0, 13.0),
    "open_laptop": (32.0, 27.0),
    "open_microwave": (21.0, 11.0),
    "pick_diverse_bottles": (26.0, 27.0),
    "pick_dual_bottles": (42.0, 40.0),
    "place_a2b_left": (31.0, 36.0),
    "place_a2b_right": (24.0, 36.0),
    "place_bread_basket": (42.0, 61.0),
    "place_bread_skillet": (23.0, 32.0),
    "place_burger_fries": (58.0, 78.0),
    "place_can_basket": (8.0, 25.0),
    "place_cans_plasticbox": (25.0, 71.0),
    "place_container_plate": (76.0, 65.0),
    "place_dual_shoes": (3.0, 1.0),
    "place_empty_cup": (50.0, 47.0),
    "place_fan": (17.0, 22.0),
    "place_mouse_pad": (8.0, 8.0),
    "place_object_basket": (28.0, 36.0),
    "place_object_scale": (17.0, 26.0),
    "place_object_stand": (51.0, 43.0),
    "place_phone_stand": (4.0, 11.0),
    "place_shoe": (70.0, 61.0),
    "press_stapler": (64.0, 73.0),
    "put_bottles_dustbin": (29.0, 48.0),
    "put_object_cabinet": (27.0, 15.0),
    "rotate_qrcode": (7.0, 11.0),
    "scan_object": (22.0, 61.0),
    "shake_bottle_horizontally": (93.0, 96.0),
    "shake_bottle": (90.0, 97.0),
    "stack_blocks_three": (0.0, 27.0),
    "stack_blocks_two": (1.0, 41.0),
    "stack_bowls_three": (33.0, 35.0),
    "stack_bowls_two": (52.0, 41.0),
    "stamp_seal": (10.0, 4.0),
    "turn_switch": (7.0, 13.0),
}
# 표 8 (청정), 표 9 (무작위): 과제 -> (pi0.5, pi0.5 + BoxWAM)  (부록 C)
# 표 3 (실로봇, 각 과제 50회): 방법 -> [(SR, 진행률) x 3과제], 논문 평균
T3_TASKS = {"Make Coffee": 5, "Water Plant": 3, "Place Pokemon": 2}  # 표 6의 하위 목표 수
T3 = {
    "DP": [(10.0, 34.0), (8.0, 9.4), (34.0, 47.0)],
    "Flow": [(24.0, 48.0), (48.0, 54.7), (60.0, 72.0)],
    "BoxWAM": [(26.0, 54.4), (66.0, 70.7), (70.0, 78.0)],
}
T3_AVG = {"DP": 23.7, "Flow": 51.1, "BoxWAM": 60.9}  # 표 3 Average 열. 본문은 BoxWAM 60.85
# 표 4 (지연 ms, 청정, 무작위) / 표 5
T4 = {
    "Flow": (168.3, 73.2, 32.6), "VAE": (250.5, 80.1, 34.5), "DINO": (251.4, 69.9, 29.0),
    "pose6d": (None, 73.2, 32.1), "obb2d": (None, 74.2, 30.7), "keypoints": (None, 75.3, 27.8),
    "ee_traj": (None, 74.0, 32.6), "Box": (172.4, 78.3, 37.8),
}
T5 = {
    "Flow": (73.2, 32.6), "separate": (76.6, 37.4), "joint": (76.7, 32.9),
    "action_then_state": (76.1, 35.5), "state_then_action": (78.3, 37.8),
}

rng = np.random.default_rng(0)
R = {}


def feasible(p, n, nd=1):
    """백분율 p(소수 nd자리)가 시행 n회의 정수 성공 수로 나올 수 있는가. 가능하면 가장 가까운 k를 돌려준다."""
    ks = [k for k in range(n + 1) if round(100.0 * k / n, nd) == round(p, nd)]
    return ks[0] if ks else None


def nearest_k(p, n):
    k = round(p * n / 100.0)
    return k, round(100.0 * k / n, 2)


def boot_ci(x, n=20000):
    x = np.asarray(x, float)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def paired(diff):
    d = np.asarray(diff, float)
    wins, losses, ties = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    nz = wins + losses
    sign_p = float(stats.binomtest(wins, nz, 0.5).pvalue) if nz else None
    try:
        w = float(stats.wilcoxon(d[d != 0]).pvalue)
    except ValueError:
        w = None
    t = stats.ttest_1samp(d, 0.0)
    return {"mean_diff": float(d.mean()), "wins": wins, "losses": losses, "ties": ties,
            "sign_p": sign_p, "wilcoxon_p": w, "t_p": float(t.pvalue), "boot95": boot_ci(d)}


def two_prop(p1, p2, n1, n2):
    """풀링 2표본 비율 z 검정. p는 백분율."""
    a, b = p1 / 100.0, p2 / 100.0
    pool = (a * n1 + b * n2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (b - a) / se
    return {"diff_pp": round(p2 - p1, 2), "z": round(z, 2), "p": float(2 * stats.norm.sf(abs(z))),
            "se_pp": round(100 * se, 2)}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


# ---------------------------------------------------------------- V1 표 1 평균
v1 = {}
for setting, rows in T1.items():
    for m, (avg, cells) in rows.items():
        calc = float(np.mean(cells))
        v1[f"{setting}/{m}"] = {"reported": avg, "recomputed": round(calc, 2), "diff": round(calc - avg, 2),
                                "ok": abs(round(calc, 1) - avg) < 0.051}
R["V1_table1_averages"] = v1
cl, rd = T1["clean"], T1["randomized"]
R["V1_claims"] = {
    "clean_BoxWAM_rank": sorted(((m, v[0]) for m, v in cl.items()), key=lambda x: -x[1]),
    "rand_BoxWAM_rank": sorted(((m, v[0]) for m, v in rd.items()), key=lambda x: -x[1]),
    "clean_Box_minus_Flow": round(78.3 - 73.2, 1), "rand_Box_minus_Flow": round(37.8 - 32.6, 1),
    "clean_Box_minus_pi0": round(78.3 - 71.4, 1), "rand_Box_minus_pi0": round(37.8 - 33.1, 1),
    "rand_Box_minus_pi0_per_task_wins": int(sum(b > p for b, p in zip(rd["BoxWAM"][1], rd["pi0"][1]))),
    "clean_Box_minus_DP3": round(78.3 - 81.2, 1),
}

# ---------------------------------------------------------------- V2 표 1 정수 가능성 (Flow, BoxWAM)
v2 = {}
for setting, rows in T1.items():
    for m in ("Flow", "BoxWAM"):
        for t, p in zip(T1_TASKS, rows[m][1]):
            v2[f"{setting}/{m}/{t}"] = {"p": p, "k_n300": feasible(p, 300), "k_n150": feasible(p, 150),
                                        "k_n100": feasible(p, 100)}
bad300 = [k for k, v in v2.items() if v["k_n300"] is None]
bad150 = [k for k, v in v2.items() if v["k_n150"] is None]
bad100 = [k for k, v in v2.items() if v["k_n100"] is None]
R["V2_table1_integer"] = {"cells": v2, "n_cells": len(v2), "infeasible_n300": bad300, "infeasible_n150": bad150,
                          "n_infeasible_n150": len(bad150), "n_infeasible_n100": len(bad100)}
# 리더보드 기준선 칸은 정수여야 한다(100회)
R["V2_baseline_ints"] = all(float(x).is_integer() for m in ("DP", "ACT", "DP3", "pi0")
                            for s in T1.values() for x in s[m][1])

# ---------------------------------------------------------------- V3 BoxWAM 대 Flow, 과제별 (n=300 가정)
v3 = {}
for setting, rows in T1.items():
    per_task, pvals = [], []
    for t, pf, pb in zip(T1_TASKS, rows["Flow"][1], rows["BoxWAM"][1]):
        kf, kb = round(pf * 3), round(pb * 3)
        tab = [[kb, 300 - kb], [kf, 300 - kf]]
        p = float(stats.fisher_exact(tab)[1])
        per_task.append({"task": t, "flow": pf, "box": pb, "diff": round(pb - pf, 1), "fisher_p": p})
        pvals.append(p)
    order = np.argsort(pvals)
    holm, run = [None] * 9, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, pvals[i] * (9 - rank)))
        holm[i] = run
    for d, h in zip(per_task, holm):
        d["holm_p"] = h
    diffs = [d["diff"] for d in per_task]
    v3[setting] = {"per_task": per_task, "paired": paired(diffs),
                   "n_sig_005": sum(d["fisher_p"] < 0.05 for d in per_task),
                   "n_sig_holm": sum(d["holm_p"] < 0.05 for d in per_task),
                   "n_box_better": sum(d > 0 for d in diffs), "n_box_worse": sum(d < 0 for d in diffs)}
R["V3_box_vs_flow_table1"] = v3

# ---------------------------------------------------------------- V4 표 2 / 7 / 8 / 9
v4 = {}
for name, tab, idx in (("clean", T8_CLEAN, 0), ("randomized", T9_RAND, 1)):
    base = np.array([v[0] for v in tab.values()])
    box = np.array([v[1] for v in tab.values()])
    d = box - base
    per, pv = [], []
    for t, (a, b) in tab.items():
        p = float(stats.fisher_exact([[int(b), 100 - int(b)], [int(a), 100 - int(a)]])[1])
        per.append({"task": t, "base": a, "box": b, "diff": b - a, "fisher_p": p})
        pv.append(p)
    order = np.argsort(pv)
    holm, run = [None] * 50, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, pv[i] * (50 - rank)))
        holm[i] = run
    for r, h in zip(per, holm):
        r["holm_p"] = h
    sig_up = [r["task"] for r in per if r["fisher_p"] < 0.05 and r["diff"] > 0]
    sig_down = [r["task"] for r in per if r["fisher_p"] < 0.05 and r["diff"] < 0]
    pooled = two_prop(base.mean(), box.mean(), 5000, 5000)
    v4[name] = {
        "n_tasks": len(tab), "mean_base": round(float(base.mean()), 3), "mean_box": round(float(box.mean()), 3),
        "reported_base": T2["pi05"][idx], "reported_box": T2["pi05+Box"][idx],
        "paired_over_tasks": paired(d), "n_sig_up_005": len(sig_up), "n_sig_down_005": len(sig_down),
        "sig_up": sig_up, "sig_down": sig_down,
        "n_sig_holm": sum(r["holm_p"] < 0.05 for r in per), "pooled_two_prop_5000": pooled,
        "largest_drops": sorted(per, key=lambda r: r["diff"])[:5],
        "largest_gains": sorted(per, key=lambda r: -r["diff"])[:5],
        "tasks_base_ge_95": [t for t, v in tab.items() if v[0] >= 95],
        "per_task": per,
    }
# 청정에서 상승한 과제 수가 많아도 평균을 정하는 과제
v4["clean"]["top5_gain_share"] = round(float(sum(sorted((v[1] - v[0] for v in T8_CLEAN.values()), reverse=True)[:5]) /
                                              sum(max(0.0, v[1] - v[0]) for v in T8_CLEAN.values())), 3)
v4["randomized"]["top5_gain_share"] = round(float(sum(sorted((v[1] - v[0] for v in T9_RAND.values()), reverse=True)[:5]) /
                                                   sum(max(0.0, v[1] - v[0]) for v in T9_RAND.values())), 3)
# 평균 차이의 과제 제외 민감도: 이득이 가장 큰 5개 과제를 빼면
for name, tab in (("clean", T8_CLEAN), ("randomized", T9_RAND)):
    d = sorted((v[1] - v[0] for v in tab.values()), reverse=True)
    v4[name]["mean_diff_without_top5_gain"] = round(float(np.mean(d[5:])), 2)
v4["latency"] = {"pi05_ms": 55.3, "box_ms": 57.7, "overhead_ms": round(57.7 - 55.3, 1),
                 "overhead_pct": round(100 * (57.7 / 55.3 - 1), 1)}
R["V4_table2_multitask"] = v4

# ---------------------------------------------------------------- V5 표 3 실로봇
v5 = {"avg": {}, "cells": {}, "sr_only_avg": {}, "progress_only_avg": {}}
for m, cells in T3.items():
    flat = [x for c in cells for x in c]
    v5["avg"][m] = {"reported": T3_AVG[m], "recomputed": round(float(np.mean(flat)), 3),
                    "text_BoxWAM_60.85": (60.85 if m == "BoxWAM" else None)}
    v5["sr_only_avg"][m] = round(float(np.mean([c[0] for c in cells])), 2)
    v5["progress_only_avg"][m] = round(float(np.mean([c[1] for c in cells])), 2)
    for (task, S), (sr, pr) in zip(T3_TASKS.items(), cells):
        s_cnt = feasible(sr, 50)
        k_pr = feasible(pr, 50 * S)
        lo = (s_cnt or 0) * S
        hi = (s_cnt or 0) * S + (50 - (s_cnt or 0)) * (S - 1)
        nk, nv = nearest_k(pr, 50 * S)
        v5["cells"][f"{m}/{task}"] = {
            "SR": sr, "SR_k": s_cnt, "SR_wilson95": wilson(s_cnt, 50) if s_cnt is not None else None,
            "progress": pr, "progress_k_feasible": k_pr, "progress_nearest_k": nk, "progress_nearest_value": nv,
            "progress_k_range_given_SR": [lo, hi], "progress_ge_SR": pr >= sr,
        }
v5["infeasible_progress_cells"] = [k for k, v in v5["cells"].items() if v["progress_k_feasible"] is None]
# Fisher: SR 50 vs 50
fis = {}
for task_i, task in enumerate(T3_TASKS):
    for a, b in (("BoxWAM", "Flow"), ("BoxWAM", "DP"), ("Flow", "DP")):
        ka, kb = feasible(T3[a][task_i][0], 50), feasible(T3[b][task_i][0], 50)
        fis[f"{task}: {a} vs {b}"] = {"k": [ka, kb], "fisher_p": float(stats.fisher_exact([[ka, 50 - ka], [kb, 50 - kb]])[1])}
tot = {m: sum(feasible(c[0], 50) for c in T3[m]) for m in T3}
for a, b in (("BoxWAM", "Flow"), ("BoxWAM", "DP"), ("Flow", "DP")):
    fis[f"3과제 합계(150회): {a} vs {b}"] = {"k": [tot[a], tot[b]], "fisher_p": float(
        stats.fisher_exact([[tot[a], 150 - tot[a]], [tot[b], 150 - tot[b]]])[1])}
v5["fisher_sr"] = fis
v5["table_vs_text_avg_BoxWAM"] = {"table": 60.9, "text": 60.85, "recomputed": round(float(np.mean([x for c in T3["BoxWAM"] for x in c])), 3)}
R["V5_table3_realworld"] = v5

# ---------------------------------------------------------------- V6 표 4·5 차이와 표준오차(시행 수 가정)
v6 = {"T4": {}, "T5": {}}
for n_cell, label in ((2700, "n2700(3x100x9)"), (1350, "n1350(3x50x9)")):
    v6["T4"][label] = {m: {"clean": two_prop(v[1], T4["Box"][1], n_cell, n_cell),
                            "rand": two_prop(v[2], T4["Box"][2], n_cell, n_cell)} for m, v in T4.items() if m != "Box"}
    v6["T5"][label] = {m: {"clean": two_prop(v[0], T5["state_then_action"][0], n_cell, n_cell),
                            "rand": two_prop(v[1], T5["state_then_action"][1], n_cell, n_cell)}
                       for m, v in T5.items() if m != "state_then_action"}
v6["claims"] = {
    "box_best_rand": T4["Box"][2] == max(v[2] for v in T4.values()),
    "box_best_clean": T4["Box"][1] == max(v[1] for v in T4.values()),
    "VAE_minus_Box_clean": round(T4["VAE"][1] - T4["Box"][1], 1),
    "box_beats_compact_both": all(T4["Box"][1] > T4[m][1] and T4["Box"][2] > T4[m][2]
                                  for m in ("pose6d", "obb2d", "keypoints", "ee_traj")),
    "separate_minus_box_rand": round(T5["separate"][1] - T5["state_then_action"][1], 1),
    "latency_box_over_flow_pct": round(100 * (172.4 / 168.3 - 1), 1),
    "latency_vae_over_flow_pct": round(100 * (250.5 / 168.3 - 1), 1),
    "latency_dino_over_flow_pct": round(100 * (251.4 / 168.3 - 1), 1),
    "latency_box_minus_flow_ms": round(172.4 - 168.3, 1),
    "steps_ratio_dense_vs_others": round(15000 / 6000, 2),
}
R["V6_ablations"] = v6

# ---------------------------------------------------------------- V7 표 간 중복과 교차 검사
R["V7_cross_table"] = {
    "T1_Box_equals_T4_Box": (T1["clean"]["BoxWAM"][0], T1["randomized"]["BoxWAM"][0]) == (T4["Box"][1], T4["Box"][2]),
    "T1_Flow_equals_T4_Flow": (T1["clean"]["Flow"][0], T1["randomized"]["Flow"][0]) == (T4["Flow"][1], T4["Flow"][2]),
    "T5_Box_equals_T4_Box": T5["state_then_action"] == (T4["Box"][1], T4["Box"][2]),
    "T5_Flow_equals_T4_Flow": T5["Flow"] == (T4["Flow"][1], T4["Flow"][2]),
    "T1_protocol_text": "3 seeds x 100 trials per task (7쪽 표 1 캡션, 6쪽 4.1)",
    "sec43_protocol_text": "3 seeds x 50 trials per task (8쪽 4.3)",
    "T2_equals_T7": T2["pi05"][:2] == (52.6, 26.9),
    "T2_means_vs_T8T9": {"clean_base": round(float(np.mean([v[0] for v in T8_CLEAN.values()])), 2),
                         "clean_box": round(float(np.mean([v[1] for v in T8_CLEAN.values()])), 2),
                         "rand_base": round(float(np.mean([v[0] for v in T9_RAND.values()])), 2),
                         "rand_box": round(float(np.mean([v[1] for v in T9_RAND.values()])), 2)},
    "T4_T5_cells_in_averages_n150_feasible": {
        k: feasible(v, 1350, 1) is not None for k, v in
        {"Box clean": 78.3, "Box rand": 37.8, "Flow clean": 73.2, "Flow rand": 32.6}.items()},
}
# 표 1의 9과제 평균이 표 4·5의 평균과 같은 값으로 두 번 나온다는 것을 근거로,
# 표 1 칸이 n=150으로는 불가능(V2)이므로 절 4.3의 "50회"가 표 1의 300회 중 일부이거나 문장 쪽 표기 차이일 가능성이 있다.

# ---------------------------------------------------------------- V8 그림 6과 본문 수치 (그림 판독)
x_px = {"50": 135.0, "71": 252.0, "200": 605.0}  # 잘라낸 그림 6의 눈금 픽셀 (판독)
def px_to_ms(px):
    s = (math.log(71) - math.log(50)) / (x_px["71"] - x_px["50"])
    return 50 * math.exp((px - x_px["50"]) * s)
R["V8_fig6_and_text"] = {
    "text_easy": 87.94, "text_hard": 87.62, "text_mean": round((87.94 + 87.62) / 2, 2),
    "fig6_read_boxwam_sr": 87.8, "fig6_read_boxwam_ms": round(px_to_ms(224), 1), "text_ms": 64.9,
    "fig6_read_pi05_ms": round(px_to_ms(212), 1), "fig6_read_pi05_sr": 80.0,
    "fig6_read_pi0_sr": 62.0, "fig6_read_fastwam_sr": 92.0,
    "IK_baseline": {"easy": 26.78, "hard": 26.30},
    "multitask_2500_vs_27500_clean": {"2500_joint": 61.5, "27500_eef_easy": 87.94, "ratio": round(87.94 / 61.5, 2)},
    "note": "그림 값은 잘라낸 그림의 픽셀에서 읽은 근삿값이다 (그림 판독 값).",
}
# 표 6(하위 목표 수)과 평가 횟수로 본 진행률의 정밀도
R["V9_real_world_granularity"] = {t: {"subgoals": S, "progress_step_pct": round(100 / (50 * S), 3)}
                                  for t, S in T3_TASKS.items()}


# ---------------------------------------------------------------- V10 검출 한계 (유의수준 0.05, 검정력 0.8, 두 표본 비율, 시행 독립 가정)
def mde(n, p=0.5):
    se = math.sqrt(2 * p * (1 - p) / n)
    return round(100 * (1.96 + 0.8416) * se, 1)
R["V10_detectable_difference_pp"] = {
    "real_world_cell_n50_p0.5": mde(50), "real_world_cell_n50_p0.25": mde(50, 0.25),
    "real_world_3tasks_n150_p0.5": mde(150),
    "sim_task_n100_p0.5": mde(100), "sim_task_n300_p0.5": mde(300),
    "table1_avg_n2700_p0.5": mde(2700), "table1_avg_n1350_p0.5": mde(1350),
    "table2_pooled_n5000_p0.4": mde(5000, 0.4),
    "note": "시행을 독립으로 본 두 표본 비율 검정의 검출 한계(백분율 포인트). 과제 군집과 학습 시드 변동은 포함하지 않아 실제 한계는 이보다 크다.",
}

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print("wrote", OUT)
