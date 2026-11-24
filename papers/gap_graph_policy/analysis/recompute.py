#!/usr/bin/env python3
"""GaP (arXiv:2607.05369 v3) 표 수치 재계산과 검정.

실행: python3 analysis/recompute.py   (수 초, numpy·scipy만 사용)
결과: analysis/recompute_results.json

가정(논문에 시행 수가 없는 곳은 명시):
- 표 1: 칸당 100 task instance (논문 5.2절, 표 1 캡션).
- 표 3의 성공률: 칸당 100 instance로 가정 (논문에 명시 없음).
- 그림 2, 3의 값은 그림 판독 값(약 ±1)입니다.
"""
import json
import math
import os

from scipy.stats import fisher_exact

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recompute_results.json")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def fisher(k1, n1, k2, n2):
    return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


R = {}

# ---------------------------------------------------------------- 표 1
cols = ["LIBERO object", "LIBERO-Pro object swap", "XY 20x20", "basket swap",
        "permutation", "mixed all", "pack fixed", "pack varied"]
T1 = {
    "CaP-X":           [None, .22, .07, .05, .11, .10, .01, .01],
    "pi0.5":           [.96, .24, .78, .15, .20, .20, .17, .18],
    "MolmoAct2":       [.97, .43, .90, .26, .10, .20, .18, .18],
    "TipTop":          [.22, .22, .29, .24, .31, .24, .34, .46],
    "pi0.5 w/GaP":     [.85, .60, .79, .32, .50, .39, .67, .66],
    "MolmoAct2 w/GaP": [.70, .62, .84, .58, .39, .66, .59, .59],
    "GaP":             [.95, .95, .95, .97, .93, .97, .99, .98],
}
N = 100
cells = sum(1 for r in T1.values() for v in r if v is not None)
r1 = {"cells": cells, "trials_if_100_each": cells * N,
      "all_values_integer_counts_of_100": all(abs(v * N - round(v * N)) < 1e-9
                                              for r in T1.values() for v in r if v is not None)}

# GaP 대 순수 기준선 중 최고 (CaP-X, pi0.5, MolmoAct2, TipTop)
pure = ["CaP-X", "pi0.5", "MolmoAct2", "TipTop"]
vs_best = []
for j, c in enumerate(cols):
    cand = [(m, T1[m][j]) for m in pure if T1[m][j] is not None]
    m, v = max(cand, key=lambda x: x[1])
    g = T1["GaP"][j]
    kg, kb = round(g * N), round(v * N)
    vs_best.append({"column": c, "gap": g, "best_baseline": m, "best_value": v,
                    "diff_pp": round((g - v) * 100, 1),
                    "fisher_p": round(fisher(kg, N, kb, N), 4),
                    "wilson_gap": wilson(kg, N), "wilson_best": wilson(kb, N)})
r1["gap_vs_best_pure_baseline"] = vs_best

# "as low as 0.20 for LIBERO-PRO": 열별 VLA 최솟값
r1["vla_min_by_column"] = {c: min(T1["pi0.5"][j], T1["MolmoAct2"][j]) for j, c in enumerate(cols)}
r1["vla_overall_min"] = min(min(T1["pi0.5"]), min(T1["MolmoAct2"]))

# GaP를 붙인 VLA의 배율 (">2x" 주장)
ratios = {}
for base, hyb in [("pi0.5", "pi0.5 w/GaP"), ("MolmoAct2", "MolmoAct2 w/GaP")]:
    ratios[base] = [round(T1[hyb][j] / T1[base][j], 2) for j in range(8)]
r1["hybrid_over_vla_ratio"] = {"columns": cols, **ratios}
r1["hybrid_ratio_ge_2_count"] = {b: sum(1 for x in v if x >= 2.0) for b, v in ratios.items()}
r1["hybrid_lower_than_vla_columns"] = {
    b: [cols[j] for j in range(8) if T1[h][j] < T1[b][j]]
    for b, h in [("pi0.5", "pi0.5 w/GaP"), ("MolmoAct2", "MolmoAct2 w/GaP")]}

# 검출 한계: n=100 vs 100, 80% 검정력 근사 (정규 근사)
def mde(p, n=100):
    return 2.8 * math.sqrt(2 * p * (1 - p) / n)
r1["min_detectable_diff_n100_at_p"] = {str(p): round(mde(p), 3) for p in (0.5, 0.8, 0.9, 0.95)}

# 표 3과 대조
r1["cross_check_table3"] = {
    "I_TiPToP_table3": 0.24, "I_TiPToP_table1_mixed_all": T1["TipTop"][5],
    "II_TiPToP_table3": 0.34, "II_TiPToP_table1_pack_fixed": T1["TipTop"][6],
    "I_GaP_table3": 0.97, "I_GaP_table1_mixed_all": T1["GaP"][5],
    "II_GaP_table3": 0.97, "II_GaP_table1_pack_fixed_varied": [T1["GaP"][6], T1["GaP"][7]],
}
R["table1"] = r1

# ---------------------------------------------------------------- 표 2 (실로봇)
T2 = {"Fulfill Grocery Orders": (8, 25, 25, 25), "  object": (3, 5, 5, 5),
      "  20x20": (2, 5, 5, 5), "  basket swap": (1, 5, 5, 5),
      "  tall basket": (0, 5, 5, 5), "  permutation": (2, 5, 5, 5),
      "Pack Grocery Items": (10, 30, 28, 30), "Make Popcorn": (0, 20, 18, 20)}
r2 = {}
for name, (kt, nt, kg, ng) in T2.items():
    r2[name.strip()] = {"tiptop": f"{kt}/{nt}", "gap": f"{kg}/{ng}",
                        "tiptop_wilson": wilson(kt, nt), "gap_wilson": wilson(kg, ng),
                        "fisher_p": round(fisher(kt, nt, kg, ng), 6)}
r2["subrows_sum_tiptop"] = 3 + 2 + 1 + 0 + 2
r2["subrows_sum_gap"] = 25
r2["gap_pooled_real_trials"] = {"k": 25 + 28 + 18, "n": 25 + 30 + 20,
                                "wilson": wilson(25 + 28 + 18, 25 + 30 + 20)}
r2["time_check_s"] = {"perceive+execute": 14.1 + 36.4, "caption_avg_gap": 67, "gap_unexplained_s": round(67 - 50.5, 1)}
r2["popcorn_real_failures_text"] = 2
R["table2"] = r2

# ---------------------------------------------------------------- 표 3 / 그림 3 / 그림 2
thr = {"I": (49.7, 124.1), "II": (7.3, 16.2), "III": (30.5, 76.6)}
fig3_read = {"I": (72.5, 29.3), "II": (490.0, 218.0), "III": (118.0, 47.0)}  # 그림 판독
r3 = {}
for k, (a, b) in thr.items():
    ct0, ct1 = 3600 / a, 3600 / b
    r3[k] = {"implied_ct_initial_s": round(ct0, 1), "implied_ct_learned_s": round(ct1, 1),
             "fig3_read_initial_s": fig3_read[k][0], "fig3_read_learned_s": fig3_read[k][1],
             "ct_reduction_pct_from_table3": round((1 - ct1 / ct0) * 100, 1),
             "ct_reduction_pct_from_fig3_read": round((1 - fig3_read[k][1] / fig3_read[k][0]) * 100, 1),
             "throughput_gain_x": round(b / a, 2)}
r3["text_claim_up_to_pct"] = 57
r3["mean_reduction_table3_pct"] = round(sum(r3[k]["ct_reduction_pct_from_table3"] for k in thr) / 3, 1)
# 처리량이 success/ct 인지 확인: III 초기 성공률 0.33
r3["III_if_success_weighted"] = {
    "initial_throughput_if_sr_weighted": round(0.33 * 3600 / 118, 1),
    "reported_initial": 30.5,
    "learned_if_sr_weighted": round(0.94 * 3600 / 47, 1), "reported_learned": 76.6}
R["table3_fig3"] = r3

fig2_read = [30, 30, 43, 30, 80, 70, 80, 90, 93, 94]  # %, 그림 판독
r4 = {"read_percent": fig2_read}
r4["k_over_30_matches"] = [{"read": v, "nearest_k": round(v / 100 * 30),
                           "k/30_pct": round(round(v / 100 * 30) / 30 * 100, 1)} for v in fig2_read]
r4["iter10_94pct_as_k_of_30"] = [28 / 30 * 100, 29 / 30 * 100]
r4["iter10_94pct_as_k_of_100"] = 94 / 100 * 100
r4["text_initial_33pct_as_k_of_30"] = round(10 / 30 * 100, 1)
r4["phases_text"] = {"grasp": "iter 1-3", "perception_prompt_to_handle": "iter 4", "placement_offset": "iter 4-8"}
r4["phases_figure_shading"] = {"Grasp Improvement": "iter 1-3", "Transport Adjustment": "iter 4-5",
                               "Placement Improvement": "iter 6-10"}
r4["time_check"] = {"15min_x_10iter": 150, "text_total_min": 150,
                    "fig3_popcorn_iterations_incl_rejected_approx": 7}
r4["appendix_E2_iter1"] = {"grasp_25_of_30": round(25 / 30, 3), "place_17_of_30": round(17 / 30, 3),
                           "text_all_placements_fail": True, "terminal_coverage": 0.0,
                           "fig2_iter1_on_burner_pct": 30}
R["fig2"] = r4

# ---------------------------------------------------------------- 표 4 (USB-C)
rows = {"Individual port": (5, 5), "Ascending": (32, 35), "Descending": (32, 35),
        "Odd": (20, 20), "Even": (14, 15)}
pos = {"-15deg": (4, 5), "15deg": (5, 5), "-5cm": (5, 5), "5cm": (4, 5)}
k_goal = sum(v[0] for v in rows.values()); n_goal = sum(v[1] for v in rows.values())
k_pos = sum(v[0] for v in pos.values()); n_pos = sum(v[1] for v in pos.values())
et = {"Individual port": (26.0, 1), "Ascending": (247.0, 7), "Descending": (223.2, 7),
      "Odd": (136.8, 4), "Even": (104.6, 3)}
r5 = {"goal_condition_sum": f"{k_goal}/{n_goal}", "position_orientation_sum": f"{k_pos}/{n_pos}",
      "total": f"{k_goal + k_pos}/{n_goal + n_pos}", "text_total": "121/130",
      "rate": round((k_goal + k_pos) / (n_goal + n_pos), 4),
      "wilson": wilson(k_goal + k_pos, n_goal + n_pos),
      "per_condition_wilson": {k: wilson(*v) for k, v in {**rows, **pos}.items()},
      "seconds_per_insertion": {k: round(a / b, 1) for k, (a, b) in et.items()},
      "text_seconds_per_insertion": 30}
# 시퀀스 일관성: SR(All) 대 per-trial
seq = {"Ascending": (32, 35, 2, 5), "Descending": (32, 35, 4, 5), "Odd": (20, 20, 5, 5), "Even": (14, 15, 4, 5)}
cons = {}
for k, (ki, ni, ks, ns) in seq.items():
    fails = ni - ki
    # 실패 삽입 수 f개가 서로 다른 sequence에 흩어지면 실패 sequence 수는 최대 min(f, ns)
    cons[k] = {"failed_insertions": fails, "failed_sequences": ns - ks,
               "feasible": (ns - ks) <= fails and (fails == 0) == (ns - ks == 0) or (ns - ks) <= fails}
r5["sequence_consistency"] = cons
r5["note_sequence_level"] = {"complete_sequences": f"{2+4+5+4}/20", "wilson": wilson(15, 20)}
R["table4"] = r5

# ---------------------------------------------------------------- 표 5 (Wash Crates)
r6 = {}
r6["gap_150"] = {"k": 143, "n": 150, "rate": round(143 / 150, 4), "wilson": wilson(143, 150)}
r6["hand_150"] = {"k": 148, "n": 150, "rate": round(148 / 150, 4), "wilson": wilson(148, 150)}
r6["fisher_p"] = round(fisher(143, 150, 148, 150), 4)
r6["diff_pp"] = round((143 - 148) / 150 * 100, 2)
r6["3h_run"] = {"gap": "55/59", "hand": "58/59", "gap_rate": round(55 / 59, 4), "hand_rate": round(58 / 59, 4),
                "fisher_p": round(fisher(55, 59, 58, 59), 4),
                "per_hour": [round(55 / 3, 2), round(58 / 3, 2)],
                "table_rounded": [18, 19]}
r6["time_budget"] = {"gap_59x179.13_s": round(59 * 179.13, 0), "hand_59x176.47_s": round(59 * 176.47, 0),
                     "window_s": 10800, "implied_overhead_per_trial_gap_s": round((10800 - 59 * 179.13) / 59, 1),
                     "implied_overhead_per_trial_hand_s": round((10800 - 59 * 176.47) / 59, 1)}
r6["steady_state_throughput_per_h"] = {"gap": round(0.9533 * 3600 / 179.13, 2),
                                       "hand": round(0.9867 * 3600 / 176.47, 2)}
r6["cycle_time_diff_s"] = round(179.13 - 176.47, 2)
# 동등성: 차이 -3.3pp 의 95% 신뢰구간 (Newcombe 근사 대신 Wald)
p1, p2 = 143 / 150, 148 / 150
se = math.sqrt(p1 * (1 - p1) / 150 + p2 * (1 - p2) / 150)
r6["diff_wald_ci_pp"] = [round((p1 - p2 - 1.96 * se) * 100, 1), round((p1 - p2 + 1.96 * se) * 100, 1)]
R["table5"] = r6

# ---------------------------------------------------------------- 기타 계수
R["counts"] = {"morsl_skills_sum": 15 + 5 + 8 + 15 + 8, "morsl_claimed": 51,
               "industry_unit_per_hour_to_s": round(3600 / 500, 2)}

with open(OUT, "w") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print("saved", OUT)
for k in ("table1", "table4", "table5"):
    print(k, json.dumps(R[k], ensure_ascii=False)[:300])
