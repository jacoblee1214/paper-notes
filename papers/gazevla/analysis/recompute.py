"""GazeVLA (arXiv:2604.22615 v3) 표 재계산과 일관성 검사 (numpy, scipy). 실행: python3 analysis/recompute.py
결과: analysis/recompute_results.json

논문 표 값은 paper/paper.txt(pdftotext -layout)와 그림에서 옮겼다. 그림 6과 그림 9의 값은 막대 위 숫자 라벨을 눈으로 읽은 값이다(그림 판독 값).
원자료가 없으므로 모든 검사는 '보고된 값끼리' 맞는지를 본다. 시행 수는 논문 서술(표 5, 4.2절)을 따르며, 가정한 곳은 키 이름에 assumed를 붙였다.
검정은 시행이 서로 독립이라고 가정한 정확 검정이라서 실제 불확실성보다 낙관적일 수 있다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fisher(k1, n1, k2, n2):
    return stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def cmh(tables):
    """Mantel-Haenszel (연속성 보정 없음)과 공통 오즈비. tables: [(k1,n1,k2,n2), ...]"""
    num = den = 0.0
    r_num = r_den = 0.0
    for k1, n1, k2, n2 in tables:
        a, b, c, d = k1, n1 - k1, k2, n2 - k2
        n = n1 + n2
        m1, m0 = a + c, b + d
        ea = n1 * m1 / n
        va = n1 * n2 * m1 * m0 / (n * n * (n - 1))
        num += a - ea
        den += va
        r_num += a * d / n
        r_den += b * c / n
    chi2 = num * num / den
    return {"chi2": chi2, "p": float(stats.chi2.sf(chi2, 1)), "or_mh": r_num / r_den if r_den else float("inf")}


# ------------------------------------------------------------------ V1 표 1 평균 (AV-ALOHA, 셀당 100회)
tasks = ["Cube Transfer", "Hook Package", "Peg Insertion", "Pour Test Tube", "Slot Insertion", "Thread Needle"]
methods = ["LFA", "HRDT", "pi0.5", "Ours"]
T1 = {  # (분포, 방법) -> 6과제 값(%), 논문 표 1
    ("ID", "LFA"): [87, 23, 10, 41, 43, 56], ("ID", "HRDT"): [89, 13, 17, 24, 42, 46],
    ("ID", "pi0.5"): [94, 20, 15, 34, 54, 33], ("ID", "Ours"): [100, 32, 18, 39, 60, 43],
    ("OOD-D", "LFA"): [12, 8, 7, 11, 23, 21], ("OOD-D", "HRDT"): [28, 0, 2, 14, 19, 23],
    ("OOD-D", "pi0.5"): [25, 11, 6, 24, 47, 19], ("OOD-D", "Ours"): [35, 14, 9, 28, 56, 23],
    ("OOD-L", "LFA"): [0, 0, 0, 0, 0, 0], ("OOD-L", "HRDT"): [12, 0, 0, 3, 11, 12],
    ("OOD-L", "pi0.5"): [32, 10, 7, 25, 44, 20], ("OOD-L", "Ours"): [36, 19, 16, 22, 50, 21],
}
T1_AVG_REPORTED = {("ID", "LFA"): 43, ("ID", "HRDT"): 39, ("ID", "pi0.5"): 41, ("ID", "Ours"): 49,
                   ("OOD-D", "LFA"): 14, ("OOD-D", "HRDT"): 14, ("OOD-D", "pi0.5"): 22, ("OOD-D", "Ours"): 28,
                   ("OOD-L", "LFA"): 0, ("OOD-L", "HRDT"): 6, ("OOD-L", "pi0.5"): 23, ("OOD-L", "Ours"): 27}
v1 = {}
for key, vals in T1.items():
    m = sum(vals) / 6
    rep = T1_AVG_REPORTED[key]
    v1["/".join(key)] = {"mean": round(m, 3), "reported": rep, "rounds_to_reported": abs(m - rep) <= 0.5 + 1e-9,
                         "sum_counts_of_600": sum(vals)}
res["V1_table1_averages"] = v1

# ------------------------------------------------------------------ V2 "OOD에서 π0.5 대비 22% 상대 향상"
rel = {}
for s in ("OOD-D", "OOD-L"):
    o, p = sum(T1[(s, "Ours")]) / 6, sum(T1[(s, "pi0.5")]) / 6
    rel[s] = {"ours": o, "pi05": p, "relative_gain_pct": 100 * (o / p - 1)}
o_all = (sum(T1[("OOD-D", "Ours")]) + sum(T1[("OOD-L", "Ours")])) / 12
p_all = (sum(T1[("OOD-D", "pi0.5")]) + sum(T1[("OOD-L", "pi0.5")])) / 12
rel["OOD pooled (12 cells)"] = {"ours": o_all, "pi05": p_all, "relative_gain_pct": 100 * (o_all / p_all - 1)}
rel["from reported rounded averages (28+27)/(22+23)"] = {"relative_gain_pct": 100 * ((28 + 27) / (22 + 23) - 1)}
rel["ID relative gain"] = {"relative_gain_pct": 100 * ((sum(T1[("ID", "Ours")]) / sum(T1[("ID", "pi0.5")])) - 1)}
res["V2_ood_relative_gain"] = rel

# ------------------------------------------------------------------ V3 셀 단위 우열 (18셀)
wins = {"ours_strict_best": 0, "ours_tied_best": 0, "ours_not_best": []}
cells = []
for s in ("ID", "OOD-D", "OOD-L"):
    for i, t in enumerate(tasks):
        vals = {m: T1[(s, m)][i] for m in methods}
        best_other = max(v for m, v in vals.items() if m != "Ours")
        cells.append({"setting": s, "task": t, "ours": vals["Ours"], "best_baseline": best_other,
                      "best_baseline_name": [m for m, v in vals.items() if m != "Ours" and v == best_other]})
        if vals["Ours"] > best_other:
            wins["ours_strict_best"] += 1
        elif vals["Ours"] == best_other:
            wins["ours_tied_best"] += 1
        else:
            wins["ours_not_best"].append(f"{s}/{t}: Ours {vals['Ours']} < {best_other}")
wins["cells"] = cells
res["V3_table1_cell_wins"] = wins

# ------------------------------------------------------------------ V4 표 1 셀별 Fisher, Holm, Wilson, 풀링 (셀당 100회 가정: 4.2절)
rows, ps = [], []
for s in ("ID", "OOD-D", "OOD-L"):
    for i, t in enumerate(tasks):
        k_o, k_p = T1[(s, "Ours")][i], T1[(s, "pi0.5")][i]
        p = fisher(k_o, 100, k_p, 100)
        rows.append({"setting": s, "task": t, "ours": k_o, "pi05": k_p, "diff_pp": k_o - k_p, "fisher_p": p,
                     "ours_wilson": [round(x * 100, 1) for x in wilson(k_o, 100)]})
        ps.append(p)
adj = holm(ps)
for r, a in zip(rows, adj):
    r["holm_p"] = a
res["V4_table1_cell_tests_assumed_n100"] = {
    "cells": rows,
    "n_cells_p_lt_0.05": sum(r["fisher_p"] < 0.05 for r in rows),
    "n_cells_holm_lt_0.05": sum(r["holm_p"] < 0.05 for r in rows),
}
pooled = {}
for s in ("ID", "OOD-D", "OOD-L"):
    for base in ("pi0.5", "HRDT", "LFA"):
        ko, kb = sum(T1[(s, "Ours")]), sum(T1[(s, base)])
        pooled[f"{s} Ours vs {base}"] = {"ours": ko, "base": kb, "n_each": 600, "diff_pp": (ko - kb) / 6,
                                          "fisher_p": fisher(ko, 600, kb, 600)}
tabs = [(T1[(s, "Ours")][i], 100, T1[(s, "pi0.5")][i], 100) for s in ("ID", "OOD-D", "OOD-L") for i in range(6)]
pooled["all 18 cells Ours vs pi0.5 (CMH over cells)"] = cmh(tabs)
res["V4_table1_pooled_assumed_n100"] = pooled

# ------------------------------------------------------------------ V5 그림 6 (그림 판독 값, 과제 6개, 방법당 40회 가정: 표 5)
F6 = {  # 과제: (EgoMimic, HRDT, pi0.5, Ours) 막대 라벨(%)
    "Collect Fruit": (50, 45, 55, 68), "Place Remote": (38, 32, 30, 43), "Stack Blocks": (38, 45, 52, 62),
    "Tighten Screw": (2, 0, 5, 12), "Grasp Bottle": (72, 78, 68, 85), "Type Keys": (30, 22, 35, 48)}
names = ["EgoMimic", "HRDT", "pi0.5", "Ours"]
mean_lab = {n: sum(F6[t][j] for t in F6) / 6 for j, n in enumerate(names)}
# 40회 격자: 가능한 정수 성공 수 k의 퍼센트는 2.5의 배수
grid = {}
for t, vs in F6.items():
    for n, v in zip(names, vs):
        ks = [k for k in range(41) if abs(2.5 * k - v) <= 0.5 + 1e-9]
        grid[f"{t}/{n}"] = {"label": v, "feasible_k_of_40": ks, "on_half_boundary": any(abs(2.5 * k - v) == 0.5 for k in ks)}
exact = {}  # 가장 가까운 격자 값(경계에서는 두 후보)으로 계산한 평균
lo_hi = {}
for j, n in enumerate(names):
    ks_lo = [min(grid[f"{t}/{n}"]["feasible_k_of_40"]) for t in F6]
    ks_hi = [max(grid[f"{t}/{n}"]["feasible_k_of_40"]) for t in F6]
    lo_hi[n] = (sum(ks_lo) * 2.5 / 6, sum(ks_hi) * 2.5 / 6)
v5 = {
    "label_means": mean_lab,
    "mean_range_from_grid_k_of_40": lo_hi,
    "diff_ours_vs_pi05_pp_label": mean_lab["Ours"] - mean_lab["pi0.5"],
    "diff_ours_vs_pi05_pp_range": [lo_hi["Ours"][0] - lo_hi["pi0.5"][1], lo_hi["Ours"][1] - lo_hi["pi0.5"][0]],
    "grid_check": grid,
    "n_labels_on_half_boundary": sum(g["on_half_boundary"] for g in grid.values()),
}
best_base = [max(F6[t][:3]) for t in F6]
v5["mean_of_best_baseline_per_task"] = sum(best_base) / 6
v5["diff_ours_vs_best_per_task_baseline_pp"] = mean_lab["Ours"] - sum(best_base) / 6
v5["strongest_baseline_by_mean"] = max(names[:3], key=lambda n: mean_lab[n])
# 과제별 Fisher: 가장 가까운 격자 k (경계에서는 라벨 쪽 반올림 방향 = 큰 값 쪽으로 보수적으로 두 후보 모두 계산)
pt = {}
cnt = {}
for t, vs in F6.items():
    ko = round(vs[3] / 2.5)
    kp = round(vs[2] / 2.5)
    pt[t] = {"ours_k": ko, "pi05_k": kp, "diff_pp": 2.5 * (ko - kp), "fisher_p": fisher(ko, 40, kp, 40)}
    cnt[t] = (ko, 40, kp, 40)
adjp = holm([pt[t]["fisher_p"] for t in F6])
for t, a in zip(F6, adjp):
    pt[t]["holm_p"] = a
v5["per_task_ours_vs_pi05_assumed_n40"] = pt
ko_all, kp_all = sum(c[0] for c in cnt.values()), sum(c[2] for c in cnt.values())
v5["pooled_ours_vs_pi05_assumed_n240"] = {"ours_k": ko_all, "pi05_k": kp_all, "fisher_p": fisher(ko_all, 240, kp_all, 240),
                                          "diff_pp": (ko_all - kp_all) / 2.4, "cmh": cmh(list(cnt.values()))}
for n_idx, n in ((0, "EgoMimic"), (1, "HRDT")):
    ks = [(round(F6[t][3] / 2.5), 40, round(F6[t][n_idx] / 2.5), 40) for t in F6]
    v5[f"pooled_ours_vs_{n}"] = cmh(ks)
# 구간 폭: 칸당 40회, 풀링 240회의 Wilson
v5["wilson_cell_n40_at_50pct"] = [round(x * 100, 1) for x in wilson(20, 40)]
v5["wilson_pooled_ours_n240"] = [round(x * 100, 1) for x in wilson(ko_all, 240)]
v5["wilson_pooled_pi05_n240"] = [round(x * 100, 1) for x in wilson(kp_all, 240)]
res["V5_figure6_real_world"] = v5

# ------------------------------------------------------------------ V6 표 2 (ID 20회, OOD 10회) 절제
T2 = {"Ours": (19, 20, 6, 10, 8, 10, 6, 10), "w/o Intention": (16, 20, 5, 10, 6, 10, 5, 10),
      "w/o Human Pretrain": (13, 20, 1, 10, 2, 10, 0, 10), "w/o Human Finetune": (17, 20, 3, 10, 5, 10, 2, 10)}
v6 = {"rows": {}}
cols = ["ID", "OOD-Position", "OOD-Object", "OOD-Scene"]
for name, t in T2.items():
    ks = [t[0], t[2], t[4], t[6]]
    ns = [t[1], t[3], t[5], t[7]]
    v6["rows"][name] = {"total_k": sum(ks), "total_n": sum(ns), "ood_k": sum(ks[1:]), "ood_n": 30,
                        "ood_wilson": [round(x * 100, 1) for x in wilson(sum(ks[1:]), 30)],
                        "id_wilson": [round(x * 100, 1) for x in wilson(ks[0], 20)]}
for other in ("w/o Intention", "w/o Human Pretrain", "w/o Human Finetune"):
    o, w = T2["Ours"], T2[other]
    cell = {c: {"ours": f"{o[2*i]}/{o[2*i+1]}", other: f"{w[2*i]}/{w[2*i+1]}",
                "fisher_p": fisher(o[2 * i], o[2 * i + 1], w[2 * i], w[2 * i + 1])} for i, c in enumerate(cols)}
    tabs = [(o[2 * i], o[2 * i + 1], w[2 * i], w[2 * i + 1]) for i in range(4)]
    ood_o, ood_w = sum(o[2 * i] for i in (1, 2, 3)), sum(w[2 * i] for i in (1, 2, 3))
    v6[f"Ours vs {other}"] = {"cells": cell, "cmh_4_strata": cmh(tabs),
                              "ood_pooled_30": {"ours": ood_o, "other": ood_w, "fisher_p": fisher(ood_o, 30, ood_w, 30)}}
v6["id_cells_fisher_min_p_over_all_12_pairs"] = min(v6[f"Ours vs {o}"]["cells"][c]["fisher_p"]
                                                    for o in ("w/o Intention", "w/o Human Pretrain", "w/o Human Finetune") for c in cols)
res["V6_table2_ablation"] = v6

# ------------------------------------------------------------------ V7 표 3 (과제 하나, 5가지 의도 원천) 과 그림 6의 같은 과제 값
T3 = {"Predicted (Ours)": 65, "Human Gaze": 65, "VLM Keypoint": 50, "No Intention": 40, "Random Noise": 25}
v7 = {"table3_labels": T3}
for n in (20, 40):
    ks = {k: v * n / 100 for k, v in T3.items()}
    v7[f"integer_ok_n{n}"] = {k: float(x).is_integer() for k, x in ks.items()}
v7["fig6_collect_fruit_ours_label"] = 68
v7["fig6_collect_fruit_ours_grid_pct"] = 67.5
v7["table3_vs_fig6_diff_pp"] = 68 - 65
for n in (20, 40):
    k = {name: int(round(p * n / 100)) for name, p in T3.items()}
    v7[f"fisher_vs_predicted_assumed_n{n}"] = {name: fisher(k["Predicted (Ours)"], n, kk, n) for name, kk in k.items() if name != "Predicted (Ours)"}
v7["fisher_human_gaze_vs_vlm_keypoint_n20"] = fisher(13, 20, 10, 20)
res["V7_table3_intention_sources"] = v7

# ------------------------------------------------------------------ V8 표 6 / 그림 9 (병 옮기기, ID 5회 + OOD 15회)
T6 = {"10m robot": (4, 9), "20m robot": (4, 12), "10m human + 10m robot": (5, 12),
      "30m robot": (5, 12), "20m human + 10m robot": (5, 13)}
FIG9 = {"10m robot": 65, "20m robot": 80, "10m human + 10m robot": 85, "30m robot": 85, "20m human + 10m robot": 90}
v8 = {"rows": {}}
for k, (i, o) in T6.items():
    tot = i + o
    v8["rows"][k] = {"id": f"{i}/5", "ood": f"{o}/15", "total": f"{tot}/20", "pct": 100 * tot / 20,
                     "fig9_label": FIG9[k], "match": 100 * tot / 20 == FIG9[k], "wilson": [round(x * 100, 1) for x in wilson(tot, 20)]}
tot = {k: i + o for k, (i, o) in T6.items()}
v8["fisher_10h10r_vs_20r"] = fisher(tot["10m human + 10m robot"], 20, tot["20m robot"], 20)
v8["fisher_10h10r_vs_30r"] = fisher(tot["10m human + 10m robot"], 20, tot["30m robot"], 20)
v8["fisher_20h10r_vs_30r"] = fisher(tot["20m human + 10m robot"], 20, tot["30m robot"], 20)
v8["fisher_20h10r_vs_10r"] = fisher(tot["20m human + 10m robot"], 20, tot["10m robot"], 20)
v8["fisher_20r_vs_10r"] = fisher(tot["20m robot"], 20, tot["10m robot"], 20)
v8["trajectories_per_10min_robot_vs_human"] = {"robot": 20, "human": 100, "source": "부록 E.2"}
v8["fig6_grasp_bottle_ours_label"] = 85
v8["note"] = "그림 6 Grasp Bottle Ours 85%(40회 가정 34/40)와 표 6의 10m human + 10m robot 17/20은 같은 데이터 양(로봇 20궤적, 사람 100궤적, 표 5)에 대응"
res["V8_table6_fig9_data_scaling"] = v8

# ------------------------------------------------------------------ V9 4.1절의 4.8%와 11픽셀
diag = 224 * math.sqrt(2)
res["V9_intention_error_units"] = {"image": "224x224", "diagonal_px": diag, "4.8pct_of_diagonal_px": 0.048 * diag,
                                   "4.8pct_of_side_px": 0.048 * 224, "11px_as_pct_of_diagonal": 100 * 11 / diag,
                                   "11px_as_pct_of_side": 100 * 11 / 224,
                                   "bin_width_px": 224 / 224, "max_quantization_error_px": 0.5}

# ------------------------------------------------------------------ V10 사전학습 계산 예산 (표 4, 부록 B.2)
steps, bs, accum, gpus, gpuh = 20000, 2048, 4, 8, 1344
frames = 150e6
wall_h = gpuh / gpus
res["V10_pretraining_budget"] = {
    "gpu_hours": gpuh, "wall_hours_8gpu": wall_h, "wall_days": wall_h / 24, "sec_per_step_wall": wall_h * 3600 / steps,
    "gpu_sec_per_step": gpuh * 3600 / steps,
    "samples_seen_if_2048_is_global_batch": steps * bs, "epochs_over_150M_frames_if_2048": steps * bs / frames,
    "samples_seen_if_2048_per_update_x4_accum": steps * bs * accum, "epochs_if_x4": steps * bs * accum / frames,
    "per_gpu_batch_if_2048_global": bs / gpus, "samples_per_sec_if_2048": steps * bs / (wall_h * 3600),
    "samples_per_sec_if_x4": steps * bs * accum / (wall_h * 3600),
}

# ------------------------------------------------------------------ V11 추론 지연 (부록 E.6)
res["V11_latency"] = {"with_intention_ms": 44.384, "direct_ms": 32.205, "diff_ms": 44.384 - 32.205, "reported_diff_ms": 12.179,
                      "relative_overhead_pct": 100 * (44.384 / 32.205 - 1), "budget_ms_at_10hz": 100,
                      "fraction_of_budget_pct": 44.384}

# ------------------------------------------------------------------ V12 버전 간 비교: v2(그림 6, 20회) 대 v3(그림 6, 40회)
F6_v2 = {"Put fruit": (60, 75, 70, 85), "Stack blocks": (30, 50, 50, 55), "Put remote": (20, 25, 30, 40),
         "Tighten screw": (0, 5, 10, 20), "Put bottle": (40, 70, 65, 85), "Type VLIA": (10, 25, 30, 45)}  # DP, ACT, pi0.5, Ours
m_v2 = [sum(F6_v2[t][j] for t in F6_v2) / 6 for j in range(4)]
res["V12_v2_vs_v3"] = {
    "v2_fig6_means_DP_ACT_pi05_Ours": m_v2, "v2_ours_minus_pi05_pp": m_v2[3] - m_v2[2],
    "v3_fig6_means": mean_lab, "v3_ours_minus_pi05_pp": mean_lab["Ours"] - mean_lab["pi0.5"],
    "v2_ours_minus_best_baseline_per_task_pp": m_v2[3] - sum(max(F6_v2[t][:3]) for t in F6_v2) / 6,
    "v2_pi05_cells": {t: F6_v2[t][2] for t in F6_v2}, "v3_pi05_cells": {t: F6[t][2] for t in F6},
    "v2_ours_cells": {t: F6_v2[t][3] for t in F6_v2}, "v3_ours_cells": {t: F6[t][3] for t in F6},
    "table1_dp_column_removed_in_v3": {"DP_avg_ID": 28, "DP_avg_OOD-D": 7, "DP_avg_OOD-L": 0},
    "table2_ours_rows_identical_v2_v3": True,
    "table2_v2_baselines_removed_in_v3": {"ACT": "16/20,1/10,4/10,0/10", "DP": "13/20,2/10,3/10,0/10", "pi0.5": "17/20,3/10,4/10,3/10"},
    "table2_v2_pi05_vs_ours_fisher": {c: fisher(a, na, b, nb) for c, (a, na, b, nb) in
                                      {"ID": (19, 20, 17, 20), "OOD-Position": (6, 10, 3, 10), "OOD-Object": (8, 10, 4, 10), "OOD-Scene": (6, 10, 3, 10)}.items()},
    "table2_v2_pi05_vs_ours_cmh": cmh([(19, 20, 17, 20), (6, 10, 3, 10), (8, 10, 4, 10), (6, 10, 3, 10)]),
}
# v2 그림 6 (20회 가정) 과제별 / 풀링
tabs_v2 = [(round(F6_v2[t][3] / 5), 20, round(F6_v2[t][2] / 5), 20) for t in F6_v2]
res["V12_v2_vs_v3"]["v2_fig6_pooled_ours_vs_pi05_assumed_n20"] = {
    "k_ours": sum(t[0] for t in tabs_v2), "k_pi05": sum(t[2] for t in tabs_v2), "n_each": 120,
    "fisher_p": fisher(sum(t[0] for t in tabs_v2), 120, sum(t[2] for t in tabs_v2), 120), "cmh": cmh(tabs_v2)}

# ------------------------------------------------------------------ V13 검출력: 같은 정확 검정으로 80% 검출 가능한 최소 차이
def power_exact(n, p1, p2):
    pm1 = stats.binom.pmf(np.arange(n + 1), n, p1)
    pm2 = stats.binom.pmf(np.arange(n + 1), n, p2)
    pw = 0.0
    for a in range(n + 1):
        for b in range(n + 1):
            if pm1[a] * pm2[b] < 1e-12:
                continue
            if fisher(a, n, b, n) < 0.05:
                pw += pm1[a] * pm2[b]
    return pw


def power_normal(n, p1, p2):
    se = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    return float(stats.norm.sf(1.959964 - abs(p1 - p2) / se) + stats.norm.cdf(-1.959964 - abs(p1 - p2) / se))


pw = {}
for n in (10, 20, 40):
    row = {}
    for base in (0.4, 0.6):
        mde = None
        for d in np.arange(0.05, 0.6, 0.05):
            if base + d > 0.999:
                break
            if power_exact(n, base, base + d) >= 0.8:
                mde = round(float(d), 2)
                break
        row[f"base_{base}"] = mde
    pw[f"n_{n}_per_arm_exact_fisher"] = row
for n in (100, 240, 600):
    row = {}
    for base in (0.4, 0.6):
        mde = None
        for d in np.arange(0.01, 0.6, 0.01):
            if base + d > 0.999:
                break
            if power_normal(n, base, base + d) >= 0.8:
                mde = round(float(d), 2)
                break
        row[f"base_{base}"] = mde
    pw[f"n_{n}_per_arm_normal_approx"] = row
pw["power_example_Table2_OOD_pooled_30_per_arm_ours_vs_wo_intention_(0.67 vs 0.53)"] = power_exact(30, 0.667, 0.533)
pw["power_example_fig6_pooled_240_per_arm_(0.529 vs 0.408)"] = power_normal(240, 0.529, 0.408)
pw["power_example_table1_cell_100_per_arm_(0.49 vs 0.41)"] = power_normal(100, 0.49, 0.41)
res["V13_power"] = pw

# ------------------------------------------------------------------ V14 표 5의 데이터 양과 시간 환산 (부록 E.2: 10분에 로봇 약 20, 사람 약 100 궤적)
T5 = {"Fruit Collecting": (100, 200), "Remote Stowing": (50, 100), "Block Stacking": (50, 100),
      "Screw Tightening": (100, 200), "Bottle Transferring": (20, 100), "Keyboard Typing": (20, 100)}
res["V14_table5_data_volumes"] = {t: {"robot_traj": r, "human_traj": h, "est_robot_min": r / 2.0, "est_human_min": h / 10.0,
                                       "human_per_robot": h / r} for t, (r, h) in T5.items()}
res["V14_table5_data_volumes"]["note"] = "10분당 로봇 약 20, 사람 약 100 궤적(부록 E.2)을 모든 과제에 적용한 환산. 과제별 실제 수집 시간은 논문에 없음"

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print("wrote", OUT)
for k in ("V1_table1_averages",):
    print({a: (b["mean"], b["reported"], b["rounds_to_reported"]) for a, b in res[k].items()})
print("V3", wins["ours_strict_best"], wins["ours_tied_best"], wins["ours_not_best"])
print("V5 means", mean_lab, "diff", v5["diff_ours_vs_pi05_pp_label"], v5["diff_ours_vs_pi05_pp_range"], "best-per-task diff", v5["diff_ours_vs_best_per_task_baseline_pp"])
