"""FingerViP (arXiv:2604.21331 v3) 보고 수치 재계산과 검정.

입력은 논문 v3의 표 1, 2, 3, 4, 5, 6, 부록 A.1·B.1, 그림 5 라벨 값, v1 표 I(열린 문 단계)을 옮겨 적은 것이다.
원시 시도 기록은 공개되지 않았으므로 시도별 쌍(pair) 정보가 필요한 검정은 하지 못한다.
대신 평가가 같은 초기 조건에서 이뤄졌다는 서술(§4)을 쓰고 McNemar 검정의 가능한 범위를 계산한다.

실행: python3 analysis/recompute.py   (numpy, scipy 필요)  → analysis/recompute_results.json
"""
import json
import math
from itertools import combinations  # noqa: F401
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
rng = np.random.default_rng(0)

TASKS = ["S1", "S2", "S3", "S4"]
N = {"S1": 42, "S2": 37, "S3": 50, "S4": 55}

# 표 1 (v3): 성공 횟수. 순서 S1, S2, S3, S4
T1 = {
    "TVC": [8, 14, 20, 30],
    "WC": [2, 10, 28, 34],
    "TVC w/ WC": [9, 17, 34, 39],
    "FTC": [24, 21, 37, 37],
    "FTC w/ WC": [18, 19, 38, 40],
    "w/o camera poses": [22, 20, 39, 37],
    "w/o joint currents": [24, 22, 42, 41],
    "Ours": [31, 28, 45, 46],
}
# 표 1의 보고 퍼센트(작업별)와 평균
T1_PCT = {
    "TVC": [19.0, 37.8, 40.0, 54.5], "WC": [4.8, 27.0, 56.0, 61.8],
    "TVC w/ WC": [21.4, 45.9, 68.0, 70.9], "FTC": [57.1, 56.8, 74.0, 67.3],
    "FTC w/ WC": [42.9, 51.4, 76.0, 72.7], "w/o camera poses": [52.4, 54.1, 78.0, 67.3],
    "w/o joint currents": [57.1, 59.5, 84.0, 74.5], "Ours": [73.8, 75.7, 90.0, 83.6],
}
T1_AVG = {"TVC": 37.8, "WC": 37.4, "TVC w/ WC": 51.6, "FTC": 63.8, "FTC w/ WC": 60.8,
          "w/o camera poses": 63.0, "w/o joint currents": 68.8, "Ours": 80.8}
# 표 6 (v3 부록 A.5) 사람 원격조작
HT = [42, 34, 48, 54]
HT_PCT = [100.0, 91.9, 96.0, 98.2]
HT_AVG = 96.5
# v1 표 I의 "Open Door" 열 (v3에서는 그림 5의 1단계로 옮겨짐)
V1_DOOR = {"TVC": 44, "WC": 47, "TVC w/ WC": 49, "FTC": 48, "FTC w/ WC": 51,
           "w/o camera poses": 44, "w/o joint currents": 47, "Ours": 50, "HT": 55}
# 그림 5 (v3) 막대 라벨, 순서 TVC, WC, TVC w/ WC, FTC, FTC w/ WC, Ours
FIG5_METHODS = ["TVC", "WC", "TVC w/ WC", "FTC", "FTC w/ WC", "Ours"]
FIG5_STAGE1 = [0.80, 0.86, 0.89, 0.87, 0.93, 0.91]
FIG5_STAGE2 = [0.55, 0.62, 0.71, 0.67, 0.73, 0.84]
# 표 4 (v3 부록 A.2) 범주별: Ours 성공/시행
T4 = {
    "S1": [(17, 18), (4, 6), (4, 6), (3, 6), (1, 3), (2, 3)],
    "S2": [(14, 18), (4, 6), (5, 6), (2, 3), (3, 4)],
    "S3": [(18, 20), (7, 10), (5, 5), (5, 5), (5, 5), (5, 5)],
    "S4": [(22, 25), (13, 15), (4, 5), (3, 5), (4, 5)],
}
T4_TOTAL = {"S1": 31, "S2": 28, "S3": 45, "S4": 46}
# 부록 A.1 범주별 시행 수
A1_COUNTS = {"S1": [18, 6, 6, 6, 3, 3], "S2": [18, 6, 6, 3, 4], "S3": [20, 10, 5, 5, 5, 5], "S4": [25, 15, 5, 5, 5]}
# 표 5 (부록 A.3): MAE x1e-3, 열 S1..S4, Avg
T5 = {
    "TVC w/ Thumb + Index": ([35.54, 34.55, 35.09, 30.30], 33.87),
    "TVC w/ Thumb + Index + Little": ([32.00, 32.10, 33.67, 28.53], 31.58),
    "TVC w/ Index + Middle + Ring": ([30.79, 32.30, 35.63, 29.38], 32.03),
    "TVC w/ Five Fingertips (Ours)": ([29.55, 30.81, 33.21, 27.90], 30.37),
}
# 표 3: 시각 인코더 MAE (첫 1, 첫 4, 첫 8 행동), 20개 궤적
T3 = {
    "Frozen CLIP": (0.0200, 0.0272, 0.0377), "Frozen DINO-v2": (0.0214, 0.0282, 0.0374),
    "Fine-tuned CLIP": (0.0232, 0.0303, 0.0396), "Fine-tuned DINO-v2": (0.0209, 0.0278, 0.0371),
    "CLIP from Scratch": (0.0225, 0.0306, 0.0428), "DINOv2 from Scratch": (0.0215, 0.0294, 0.0406),
}
# 표 2
T2 = {"acq_ms": (35.2, 34.4), "gflops": (215.01, 79.00), "fps": (10.2, 13.2), "avg_sr": (80.8, 51.6)}
# 부록 B.1
DEMOS = {"S1": (255, 7.2), "S2": (256, 6.1), "S3": (245, 8.5), "S4": (256, 13.5)}

res = {}


def pct(k, n):
    return round(100.0 * k / n, 1)


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe_diff(k1, n1, k2, n2):
    """두 비율 차(p1-p2)의 Newcombe 하이브리드 점수 95% 구간."""
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    return (d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.zeros(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def mcnemar_range(a, b, n):
    """같은 n개 초기 조건에서 방법 A의 성공 a, 방법 B의 성공 b가 주어졌을 때
    McNemar 정확검정 p값의 가능한 최소와 최대 (쌍 정보가 없으므로 범위만 낼 수 있다)."""
    def p_of(n10, n01):
        m = n10 + n01
        if m == 0:
            return 1.0
        k = min(n10, n01)
        return min(1.0, 2 * float(stats.binom.cdf(k, m, 0.5)))
    ps = []
    for n11 in range(max(0, a + b - n), min(a, b) + 1):
        ps.append(p_of(a - n11, b - n11))
    return min(ps), max(ps)


# ---------------------------------------------------------------- V1 표 1 셀 퍼센트
v1 = []
for m, ks in T1.items():
    for t, k, rep in zip(TASKS, ks, T1_PCT[m]):
        calc = pct(k, N[t])
        v1.append({"method": m, "task": t, "k": k, "n": N[t], "reported": rep, "calc": calc, "match": abs(calc - rep) < 0.051})
res["V1_cells"] = v1
res["V1_all_match"] = all(r["match"] for r in v1)
res["V1_n_cells"] = len(v1)

# ---------------------------------------------------------------- V2 평균
v2 = {}
for m, ks in T1.items():
    macro = sum(100.0 * k / N[t] for t, k in zip(TASKS, ks)) / 4
    macro_from_rounded = sum(T1_PCT[m]) / 4
    pooled = 100.0 * sum(ks) / sum(N.values())
    v2[m] = {"reported": T1_AVG[m], "macro_unrounded": round(macro, 3), "macro_from_rounded_cells": round(macro_from_rounded, 3),
             "pooled": round(pooled, 3), "pooled_k": sum(ks), "pooled_n": sum(N.values()),
             "macro_match_1dp": round(macro, 1) == T1_AVG[m] or round(macro_from_rounded, 1) == T1_AVG[m]}
res["V2_average"] = v2
ht_macro = sum(100.0 * k / N[t] for t, k in zip(TASKS, HT)) / 4
res["V2_HT"] = {"reported": HT_AVG, "macro": round(ht_macro, 3), "pooled": round(100.0 * sum(HT) / 184, 3),
                "pct_cells_match": [pct(k, N[t]) == r for t, k, r in zip(TASKS, HT, HT_PCT)]}

# ---------------------------------------------------------------- V3 상대 향상 56.6%
o, b = v2["Ours"], v2["TVC w/ WC"]
res["V3_relative_improvement"] = {
    "reported_pct": 56.6,
    "from_rounded_avgs": round(100 * (80.8 / 51.6 - 1), 2),
    "from_unrounded_macro": round(100 * (o["macro_unrounded"] / b["macro_unrounded"] - 1), 2),
    "from_pooled": round(100 * (o["pooled"] / b["pooled"] - 1), 2),
    "abs_diff_points_macro": round(o["macro_unrounded"] - b["macro_unrounded"], 2),
    "vs_best_baseline_FTC_macro_relative": round(100 * (o["macro_unrounded"] / v2["FTC"]["macro_unrounded"] - 1), 2),
    "vs_best_baseline_FTC_macro_points": round(o["macro_unrounded"] - v2["FTC"]["macro_unrounded"], 2),
}

# ---------------------------------------------------------------- V4 표 4 합산
v4 = {}
for t in TASKS:
    ks = sum(k for k, _ in T4[t])
    ns = sum(n for _, n in T4[t])
    v4[t] = {"cat_k_sum": ks, "table1_k": T1["Ours"][TASKS.index(t)], "cat_n_sum": ns, "A1_n": N[t],
             "A1_counts_sum": sum(A1_COUNTS[t]), "A1_counts_match_T4": [n for _, n in T4[t]] == A1_COUNTS[t],
             "k_match": ks == T4_TOTAL[t] == T1["Ours"][TASKS.index(t)], "n_match": ns == N[t] == sum(A1_COUNTS[t])}
res["V4_table4"] = v4

# ---------------------------------------------------------------- V5 표 5, 표 3
v5 = {}
for name, (cells, avg) in T5.items():
    v5[name] = {"avg_reported": avg, "avg_calc": round(sum(cells) / 4, 3), "match": abs(sum(cells) / 4 - avg) <= 0.0051}
best_per_task = [min(T5[n][0][i] for n in T5) for i in range(4)]
res["V5_table5"] = {"rows": v5, "ours_lowest_in_all_tasks": all(T5["TVC w/ Five Fingertips (Ours)"][0][i] == best_per_task[i] for i in range(4)),
                    "ours_margin_vs_second_best": [round(sorted(T5[n][0][i] for n in T5)[1] - T5["TVC w/ Five Fingertips (Ours)"][0][i], 2) for i in range(4)]}
col_best = {}
for j, lab in enumerate(["first", "first4", "first8"]):
    best = min(T3, key=lambda r: T3[r][j])
    col_best[lab] = {"best": best, "value": T3[best][j], "frozen_clip": T3["Frozen CLIP"][j],
                     "frozen_clip_rank": 1 + sorted(v[j] for v in T3.values()).index(T3["Frozen CLIP"][j]),
                     "rel_gap_to_best_pct": round(100 * (T3["Frozen CLIP"][j] / T3[best][j] - 1), 2)}
res["V5_table3"] = col_best
res["V5_table3_vs_table5_same_config"] = {"table3_frozenCLIP_first8": T3["Frozen CLIP"][2], "table5_ours_avg_first8": T5["TVC w/ Five Fingertips (Ours)"][1] / 1000,
                                         "ratio": round(T3["Frozen CLIP"][2] / (T5["TVC w/ Five Fingertips (Ours)"][1] / 1000), 3),
                                         "note": "표 3은 20개 궤적, 표 5는 80개 궤적(과제당 20개 가정)"}

# ---------------------------------------------------------------- V6 그림 5와 v1 표 I
v6 = {"stage2_vs_table1_S4": [], "stage1_vs_v1_open_door": []}
for m, s2 in zip(FIG5_METHODS, FIG5_STAGE2):
    k = T1[m][3]
    v6["stage2_vs_table1_S4"].append({"method": m, "fig5": s2, "k/n": f"{k}/55", "exact": round(k / 55, 4), "rounded_2dp": round(k / 55, 2), "match": round(k / 55, 2) == s2})
for m, s1 in zip(FIG5_METHODS, FIG5_STAGE1):
    k = V1_DOOR[m]
    v6["stage1_vs_v1_open_door"].append({"method": m, "fig5": s1, "v1_k": k, "exact": round(k / 55, 4), "rounded_2dp": round(k / 55, 2),
                                         "v1_pct_1dp": pct(k, 55), "from_1dp_rounded_again": round(round(100 * k / 55, 1) / 100 + 1e-9, 2), "match_exact": round(k / 55, 2) == s1})
v6["stage1_ge_stage2_all"] = all(V1_DOOR[m] >= T1[m][3] for m in V1_DOOR if m in T1)
v6["ours_stage1_rank"] = sorted(FIG5_METHODS, key=lambda m: -V1_DOOR[m]).index("Ours") + 1
v6["stage1_Ours_vs_FTCwWC"] = {"ours": V1_DOOR["Ours"], "ftc_w_wc": V1_DOOR["FTC w/ WC"]}
v6["conditional_stage2_given_stage1"] = {m: round(T1[m][3] / V1_DOOR[m], 3) for m in FIG5_METHODS}
res["V6_fig5"] = v6

# ---------------------------------------------------------------- V7 시행 수와 학습량
per_method = sum(N.values())
res["V7_counts"] = {"per_method_trials": per_method, "learned_methods_in_table1": 8, "learned_total": 8 * per_method,
                    "reported_eval_rollouts": 1472, "ht_trials": per_method, "learned_plus_ht": 9 * per_method,
                    "eight_methods_x_184_equals_1472": 8 * per_method == 1472}
eps = sum(v[0] for v in DEMOS.values())
secs = sum(v[0] * v[1] for v in DEMOS.values())
res["V7_dataset"] = {"episodes_sum": eps, "reported_total": 1012, "match": eps == 1012,
                     "total_seconds": round(secs, 1), "total_hours": round(secs / 3600, 2),
                     "frames_at_30Hz": int(secs * 30), "mean_episode_s": round(secs / eps, 2),
                     "policies_if_one_per_method_task": 8 * 4,
                     "gpu_hours_if_37h_each": 8 * 4 * 37,
                     "note": "정책 수와 학습 시간은 추정(방법 8 x 과제 4, 평균 37시간 가정). 논문은 정책 수를 쓰지 않음"}

# ---------------------------------------------------------------- V8 표 2 일관성
a_o, a_b = T2["acq_ms"]
f_o, f_b = T2["fps"]
g_o, g_b = T2["gflops"]
vit_gmacs = 17.57  # ViT-B/16 224x224 한 장의 곱셈-덧셈(MAC) 수 (아래 계산과 같은 식)
res["V8_table2"] = {
    "frame_period_ms": [round(1000 / f_o, 1), round(1000 / f_b, 1)],
    "non_acquisition_ms": [round(1000 / f_o - a_o, 1), round(1000 / f_b - a_b, 1)],
    "fps_ratio": round(f_o / f_b, 3), "flops_ratio": round(g_o / g_b, 3),
    "flops_diff_G": round(g_o - g_b, 2),
    "extra_images_per_query_if_2_obs_steps": 8,
    "diff_per_extra_image_G": round((g_o - g_b) / 8, 2),
    "vit_b16_gmacs_per_image_analytic": vit_gmacs,
    "relative_sr_gain_vs_flops_ratio": round(80.8 / 51.6, 3),
}

# ---------------------------------------------------------------- V9 Wilson 구간
wil = {}
for m, ks in T1.items():
    wil[m] = {t: [round(x, 3) for x in wilson(k, N[t])] for t, k in zip(TASKS, ks)}
    kk = sum(ks)
    wil[m]["pooled"] = [round(x, 3) for x in wilson(kk, 184)]
res["V9_wilson"] = wil

# ---------------------------------------------------------------- V10 Ours 대 각 방법: Fisher, Newcombe, Holm
base_names = [m for m in T1 if m != "Ours"]
rows = []
for m in base_names:
    for i, t in enumerate(TASKS):
        ko, km, n = T1["Ours"][i], T1[m][i], N[t]
        lo, hi = newcombe_diff(ko, n, km, n)
        pm = mcnemar_range(ko, km, n)
        rows.append({"baseline": m, "task": t, "ours": f"{ko}/{n}", "base": f"{km}/{n}", "diff_pts": round(100 * (ko - km) / n, 1),
                     "newcombe95": [round(100 * lo, 1), round(100 * hi, 1)], "p_fisher": fisher(ko, n, km, n),
                     "p_mcnemar_min": pm[0], "p_mcnemar_max": pm[1]})
adj = holm([r["p_fisher"] for r in rows])
for r, a in zip(rows, adj):
    r["p_holm_28"] = a
res["V10_pairwise_task"] = rows

pooled = []
for m in base_names:
    ko, km = sum(T1["Ours"]), sum(T1[m])
    lo, hi = newcombe_diff(ko, 184, km, 184)
    # Mantel-Haenszel 공통 오즈비와 CMH 검정 (과제 4층, 연속성 보정 없음)
    num = den = 0.0
    sum_a = sum_e = sum_v = 0.0
    for i, t in enumerate(TASKS):
        n = N[t]
        a_, b_ = T1["Ours"][i], n - T1["Ours"][i]
        c_, d_ = T1[m][i], n - T1[m][i]
        tot = 2 * n
        num += a_ * d_ / tot
        den += b_ * c_ / tot
        sum_a += a_
        sum_e += (a_ + b_) * (a_ + c_) / tot
        sum_v += (a_ + b_) * (c_ + d_) * (a_ + c_) * (b_ + d_) / (tot ** 2 * (tot - 1))
    cmh = (sum_a - sum_e) ** 2 / sum_v
    pooled.append({"baseline": m, "ours": f"{ko}/184", "base": f"{km}/184", "diff_pts": round(100 * (ko - km) / 184, 1),
                   "newcombe95": [round(100 * lo, 1), round(100 * hi, 1)], "p_fisher_pooled": fisher(ko, 184, km, 184),
                   "mh_or": round(num / den, 2) if den > 0 else None, "cmh_chi2": round(cmh, 2), "p_cmh": float(stats.chi2.sf(cmh, 1)),
                   "p_mcnemar_pooled_min": mcnemar_range(ko, km, 184)[0], "p_mcnemar_pooled_max": mcnemar_range(ko, km, 184)[1]})
adjp = holm([r["p_cmh"] for r in pooled])
for r, a in zip(pooled, adjp):
    r["p_cmh_holm_7"] = a
res["V10_pairwise_pooled"] = pooled

# 매크로 평균(과제 4개 평균) 차이의 모수적 부트스트랩 (각 칸 이항 재표집)
boot = {}
for m in base_names:
    d = []
    for _ in range(20000):
        so = [rng.binomial(N[t], T1["Ours"][i] / N[t]) / N[t] for i, t in enumerate(TASKS)]
        sm = [rng.binomial(N[t], T1[m][i] / N[t]) / N[t] for i, t in enumerate(TASKS)]
        d.append(100 * (np.mean(so) - np.mean(sm)))
    boot[m] = {"mean_diff_pts": round(float(np.mean(d)), 2), "ci95": [round(float(np.percentile(d, 2.5)), 1), round(float(np.percentile(d, 97.5)), 1)]}
res["V10_macro_bootstrap"] = boot

# ---------------------------------------------------------------- V11 본문 서술 검증
txt = {}
# (a) "removing finger current inputs significantly reduces success rates ... button pressing and object grasping"
txt["currents_S1"] = {"ours": "31/42", "noCur": "24/42", "p_fisher": fisher(31, 42, 24, 42)}
txt["currents_S3"] = {"ours": "45/50", "noCur": "42/50", "p_fisher": fisher(45, 50, 42, 50)}
txt["currents_S4"] = {"ours": "46/55", "noCur": "41/55", "p_fisher": fisher(46, 55, 41, 55)}
txt["currents_S2"] = {"ours": "28/37", "noCur": "22/37", "p_fisher": fisher(28, 37, 22, 37)}
txt["currents_pooled"] = {"ours": "150/184", "noCur": f"{sum(T1['w/o joint currents'])}/184",
                          "p_fisher": fisher(150, 184, sum(T1["w/o joint currents"]), 184)}
# (b) "incorporating camera pose encoding ... notably enhances ... unstable-support stick retrieval"
txt["poses_S2"] = {"ours": "28/37", "noPose": "20/37", "p_fisher": fisher(28, 37, 20, 37)}
txt["poses_S1"] = {"ours": "31/42", "noPose": "22/42", "p_fisher": fisher(31, 42, 22, 42)}
txt["poses_pooled"] = {"ours": "150/184", "noPose": f"{sum(T1['w/o camera poses'])}/184",
                       "p_fisher": fisher(150, 184, sum(T1["w/o camera poses"]), 184)}
# (c) 손목 카메라를 더하면 도움이 되는가
txt["wrist_added_to_TVC_pooled"] = {"TVC": f"{sum(T1['TVC'])}/184", "TVC w/ WC": f"{sum(T1['TVC w/ WC'])}/184",
                                    "p_fisher": fisher(sum(T1["TVC"]), 184, sum(T1["TVC w/ WC"]), 184)}
txt["wrist_added_to_FTC_pooled"] = {"FTC": f"{sum(T1['FTC'])}/184", "FTC w/ WC": f"{sum(T1['FTC w/ WC'])}/184",
                                    "p_fisher": fisher(sum(T1["FTC"]), 184, sum(T1["FTC w/ WC"]), 184)}
# (d) 손끝 카메라만(FTC) 대 TVC w/ WC: 손끝이 외부+손목을 대체하는가
txt["FTC_vs_TVCwWC_pooled"] = {"FTC": f"{sum(T1['FTC'])}/184", "TVC w/ WC": f"{sum(T1['TVC w/ WC'])}/184",
                               "p_fisher": fisher(sum(T1["FTC"]), 184, sum(T1["TVC w/ WC"]), 184)}
# (e) 세 번째 시점을 FTC에 더하는 효과 (Ours 대 FTC), 과제별
txt["ours_vs_FTC_by_task"] = {t: {"ours": f"{T1['Ours'][i]}/{N[t]}", "FTC": f"{T1['FTC'][i]}/{N[t]}",
                                  "p_fisher": fisher(T1["Ours"][i], N[t], T1["FTC"][i], N[t])} for i, t in enumerate(TASKS)}
# (f) 사람 원격조작 대 Ours (상한 기준이라는 서술)
txt["HT_vs_ours_pooled"] = {"HT": f"{sum(HT)}/184", "Ours": "150/184", "p_fisher": fisher(sum(HT), 184, 150, 184)}
txt["FTC_vs_TVCwWC_by_task"] = {t: {"FTC": f"{T1['FTC'][i]}/{N[t]}", "TVC w/ WC": f"{T1['TVC w/ WC'][i]}/{N[t]}",
                                    "p_fisher": fisher(T1["FTC"][i], N[t], T1["TVC w/ WC"][i], N[t])} for i, t in enumerate(TASKS)}
# (g) S4 2단계 성공 / 1단계 성공 (2단계 성공이 1단계 성공을 포함한다고 가정, 1단계 횟수는 v1 표 I)
txt["S4_stage2_given_stage1"] = {m: {"k": T1[m][3], "n": V1_DOOR[m], "p_vs_ours": (fisher(46, 50, T1[m][3], V1_DOOR[m]) if m != "Ours" else None)}
                                 for m in FIG5_METHODS}
res["V11_text_claims"] = txt

# ---------------------------------------------------------------- V12 범주별 일반화 (표 4)
ind_k = sum(T4[t][0][0] for t in TASKS)
ind_n = sum(T4[t][0][1] for t in TASKS)
ood_k = sum(sum(k for k, _ in T4[t][1:]) for t in TASKS)
ood_n = sum(sum(n for _, n in T4[t][1:]) for t in TASKS)
res["V12_generalization"] = {
    "T1_in_distribution_layout": f"{ind_k}/{ind_n}", "others_unseen": f"{ood_k}/{ood_n}",
    "pct": [pct(ind_k, ind_n), pct(ood_k, ood_n)], "wilson_T1": [round(x, 3) for x in wilson(ind_k, ind_n)],
    "wilson_others": [round(x, 3) for x in wilson(ood_k, ood_n)], "p_fisher": fisher(ind_k, ind_n, ood_k, ood_n),
    "per_task": {t: {"T1": f"{T4[t][0][0]}/{T4[t][0][1]}", "others": f"{sum(k for k,_ in T4[t][1:])}/{sum(n for _,n in T4[t][1:])}",
                     "p_fisher": fisher(T4[t][0][0], T4[t][0][1], sum(k for k, _ in T4[t][1:]), sum(n for _, n in T4[t][1:]))} for t in TASKS},
    "categories_with_n_le_6": sum(1 for t in TASKS for _, n in T4[t] if n <= 6),
    "categories_total": sum(len(T4[t]) for t in TASKS),
    "note": "표 4는 Ours만 보고함. 같은 범주에서 기준선의 성공률은 없음",
}
cats = []
labels = {"S1": ["위치", "색", "비점등", "모양", "흰 상자", "조명"], "S2": ["위치", "색", "모양", "재질", "길이"],
          "S3": ["자세", "물체", "커튼 재질", "커튼 색", "커튼 길이", "벽 외관"], "S4": ["자세", "물체", "손잡이 모양", "손잡이 색", "벽 외관"]}
for t in TASKS:
    for lab, (k, n) in zip(labels[t], T4[t]):
        cats.append({"task": t, "cat": lab, "k": k, "n": n, "wilson": [round(x, 2) for x in wilson(k, n)]})
res["V12_generalization"]["per_task_holm4"] = holm([res["V12_generalization"]["per_task"][t]["p_fisher"] for t in TASKS])
res["V12_categories"] = cats

# ---------------------------------------------------------------- V13 표본 크기 (검출력)
def n_per_group(p1, p2, alpha=0.05, power=0.8):
    h = abs(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return math.ceil((z / h) ** 2)


res["V13_sample_size"] = {
    "ours_vs_FTC_S1(0.738 vs 0.571)": n_per_group(31 / 42, 24 / 42),
    "ours_vs_noCur_S4(0.836 vs 0.745)": n_per_group(46 / 55, 41 / 55),
    "ours_vs_noPose_S2(0.757 vs 0.541)": n_per_group(28 / 37, 20 / 37),
    "ours_vs_noCur_pooled(0.815 vs 0.701)": n_per_group(150 / 184, sum(T1['w/o joint currents']) / 184),
    "ours_vs_FTC_pooled": n_per_group(150 / 184, sum(T1['FTC']) / 184),
    "min_detectable_diff_n42_from_0.75": None,
}
# n=42, 기준 0.75에서 검출력 0.8로 잡히는 최소 차이
for d in np.arange(0.05, 0.5, 0.01):
    if n_per_group(0.75, 0.75 - d) <= 42:
        res["V13_sample_size"]["min_detectable_diff_n42_from_0.75"] = round(float(d), 2)
        break

# ---------------------------------------------------------------- V14 v1 표 I과 v3 표 1 일치
v1_table_I = {"TVC": [8, 14, 20, 30], "WC": [2, 10, 28, 34], "TVC w/ WC": [9, 17, 34, 39], "FTC": [24, 21, 37, 37],
              "FTC w/ WC": [18, 19, 38, 40], "HT": [42, 34, 48, 54], "w/o camera poses": [22, 20, 39, 37],
              "w/o joint currents": [24, 22, 42, 41], "Ours": [31, 28, 45, 46]}
res["V14_v1_vs_v3"] = {"table_cells_identical": all(v1_table_I[m] == (T1[m] if m != "HT" else HT) for m in v1_table_I),
                       "note": "v1 표 I의 S4는 Open Door와 Retrieve Object 두 열. 평균은 Retrieve 열로 계산된 값(예: Ours 80.8)과 일치"}
v1_avg_with_door = {m: round((sum(100.0 * k / N[t] for t, k in zip(TASKS[:3], T1[m][:3])) + 100.0 * V1_DOOR[m] / 55) / 4, 1) for m in T1}
res["V14_avg_if_door_column_used"] = v1_avg_with_door

# ---------------------------------------------------------------- 요약 출력
print("V1 cells", res["V1_n_cells"], "all match:", res["V1_all_match"])
print("V2 averages macro match:", {m: v["macro_match_1dp"] for m, v in v2.items()})
print("V3", res["V3_relative_improvement"])
print("V4", {t: (v["k_match"], v["n_match"]) for t, v in v4.items()})
print("V5", res["V5_table5"]["ours_lowest_in_all_tasks"], {k: v["match"] for k, v in v5.items()})
print("V5 t3", col_best)
print("V6 s2", [(r["method"], r["match"]) for r in v6["stage2_vs_table1_S4"]])
print("V6 s1", [(r["method"], r["fig5"], r["rounded_2dp"], r["match_exact"]) for r in v6["stage1_vs_v1_open_door"]])
print("V8", res["V8_table2"])
print("V10 tasks (Ours vs X) p fisher / holm28")
for r in rows:
    print(f"  {r['baseline']:20s}{r['task']} {r['ours']:6s} vs {r['base']:6s} diff {r['diff_pts']:6.1f} p={r['p_fisher']:.4f} holm={r['p_holm_28']:.3f} mcn[{r['p_mcnemar_min']:.3f},{r['p_mcnemar_max']:.3f}]")
print("V10 pooled")
for r in pooled:
    print(f"  {r['baseline']:20s}{r['ours']} vs {r['base']} diff {r['diff_pts']} p_fisher={r['p_fisher_pooled']:.4g} cmh={r['p_cmh']:.4g} holm={r['p_cmh_holm_7']:.4g} OR={r['mh_or']} ci={r['newcombe95']}")
print("V10 bootstrap", boot)
print("V11", json.dumps(txt, indent=1, default=float)[:3000])
print("V12", {k: v for k, v in res["V12_generalization"].items() if k != "per_task"})
print("V12 per task", res["V12_generalization"]["per_task"])
print("V13", res["V13_sample_size"])
print("V14", res["V14_v1_vs_v3"], v1_avg_with_door)
print("V7", res["V7_counts"], res["V7_dataset"])

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print("wrote", OUT)
