"""PLaW-VLA (arXiv:2610.12285 v1) 논문 표 수치를 다시 계산하고 통계 검정을 한다.

입력: 논문 표와 본문 값을 아래에 그대로 옮겨 적었다(출처는 주석). Table 7은 paper_tables.py에 있다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- RoboTwin 과제당 평가 횟수는 논문에 없다. 표 7 값이 모두 정수 %이므로 100회로 가정한다(공식 코드의 기본값도 100).
- LIBERO는 논문 본문대로 스위트당 10과제 x 50회 x 평가 시드 3개 = 1,500회, 전체 6,000회다.
  비교 대상 기준선은 단일 값이므로 스위트당 500회(전체 2,000회)로 가정한다.
- LIBERO-Plus는 과제당 1회 평가가 원 벤치마크의 방식이다. 전체 에피소드 수는 논문에 없어, 검정에는 N을 변수로 두고 몇 가지 값을 쓴다.
- 실로봇은 논문 C.2대로 설정당 30회, 점수/최대 점수를 평균한 과제 진행도다.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import lsq_linear

from paper_tables import TABLE7, TABLE7_AVG

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def z_two_prop(p1, n1, p2, n2):
    """두 비율 차이의 z 검정(합동 분산). p는 0~1."""
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


def r2(x):
    return round(float(x), 2)


# --------------------------------------------------------------------------------------
# V1. Table 1 / Table 5 평균 재계산 (4개 스위트 단순 평균)
# --------------------------------------------------------------------------------------
# (방법, Spatial, Object, Goal, Long, 보고된 Average) -- Table 5(부록 A.1)
T5 = [
    ("TraceVLA", 84.6, 85.2, 75.1, 54.1, 74.8),
    ("Octo", 78.9, 85.7, 84.6, 51.1, 75.1),
    ("OpenVLA", 84.7, 88.4, 79.2, 53.7, 76.5),
    ("SpatialVLA", 88.2, 89.9, 78.6, 55.5, 78.1),
    ("LAPA", 83.4, 87.6, 78.2, 68.8, 79.5),
    ("WorldVLA", 87.6, 96.2, 83.4, 60.0, 81.8),
    ("CoT-VLA", 87.5, 91.6, 87.6, 69.0, 83.9),
    ("ThinkAct", 88.3, 91.4, 87.1, 70.9, 84.4),
    ("pi0+FAST", 96.4, 96.8, 88.6, 60.2, 85.5),
    ("MolmoAct", 87.0, 95.4, 87.6, 77.2, 86.8),
    ("SmolVLA", 93.0, 94.0, 91.0, 77.0, 88.8),
    ("Being-H0", 92.6, 96.8, 94.0, 77.4, 90.2),
    ("GR00T-N1", 94.4, 97.6, 93.0, 90.6, 93.9),
    ("pi0", 96.8, 98.8, 95.8, 85.2, 94.2),
    ("UniVLA", 95.4, 98.8, 93.6, 94.0, 95.5),
    ("F1", 98.2, 97.8, 95.4, 91.3, 95.7),
    ("DD-VLA", 97.2, 98.6, 97.4, 92.0, 96.3),
    ("FLOWER", 97.5, 99.1, 96.1, 94.9, 96.9),
    ("pi0.5", 98.8, 98.2, 98.0, 92.4, 96.9),
    ("JALA-dino", 96.0, 98.2, 97.4, 96.0, 96.9),
    ("CronusVLA", 97.3, 99.6, 96.9, 94.0, 97.0),
    ("OpenVLA-OFT", 97.6, 98.4, 97.9, 94.5, 97.1),
    ("VLA-JEPA", 96.2, 99.6, 97.2, 95.8, 97.2),
    ("Ours", 98.9, 99.0, 97.1, 94.6, 97.4),
]
v1 = []
for name, a, b, c, d, rep in T5:
    m = (a + b + c + d) / 4
    v1.append({"method": name, "mean4": round(m, 3), "reported": rep, "diff": round(m - rep, 3),
               "ok_at_0.1": abs(m - rep) <= 0.05 + 1e-9})
RES["V1_libero_averages"] = v1
RES["V1_mimic_video_3suite_mean"] = round((94.2 + 96.8 + 90.6) / 3, 3)  # 표에는 93.9, 3개 스위트 평균

# --------------------------------------------------------------------------------------
# V2. LIBERO 머리 비교 (+0.5 / Long +2.2 / VLA-JEPA 대비)
# --------------------------------------------------------------------------------------
pi05_avg_exact = (98.8 + 98.2 + 98.0 + 92.4) / 4  # 96.85 (표는 96.9로 반올림)
ours_avg_exact = (98.9 + 99.0 + 97.1 + 94.6) / 4  # 97.4
n_ours, n_base = 6000, 2000
v2 = {
    "pi05_mean_exact": pi05_avg_exact,
    "ours_mean_exact": ours_avg_exact,
    "gain_vs_pi05_pp": round(ours_avg_exact - pi05_avg_exact, 3),
    "gain_vs_pi05_pp_as_in_text": 97.4 - 96.9,
}
z, p = z_two_prop(ours_avg_exact / 100, n_ours, pi05_avg_exact / 100, n_base)
v2["ours_vs_pi05_z"], v2["ours_vs_pi05_p"] = r2(z), round(p, 3)
z, p = z_two_prop(0.974, n_ours, 0.972, n_base)
v2["ours_vs_vlajepa_z"], v2["ours_vs_vlajepa_p"] = r2(z), round(p, 3)
# LIBERO-Long 94.6 (1,500회) vs 92.4 (500회)
z, p = z_two_prop(0.946, 1500, 0.924, 500)
v2["long_946_vs_924_z"], v2["long_946_vs_924_p"] = r2(z), round(p, 3)
# 스위트별 평가 시드 SD 와 이항 분산 비교 (500회, p)
v2["binomial_sd_pp_per_seed_500"] = {
    s: r2(100 * math.sqrt(p / 100 * (1 - p / 100) / 500)) for s, p in
    [("Spatial", 98.9), ("Object", 99.0), ("Goal", 97.1), ("Long", 94.6)]
}
v2["reported_sd_pp"] = {"Spatial": 0.6, "Object": 0.3, "Goal": 1.0, "Long": 0.8}
# 평균 97.40의 평가 잡음만의 95% 구간 (n=6000)
lo, hi = wilson(round(0.974 * 6000), 6000)
v2["ours_avg_wilson95_pp"] = [r2(100 * lo), r2(100 * hi)]
RES["V2_libero_headline"] = v2

# --------------------------------------------------------------------------------------
# V3. LIBERO-Plus: 보고된 평균에서 범주 가중치 역산 (Table 6, 9개 행)
# --------------------------------------------------------------------------------------
cats = ["Robot", "Layout", "Light", "Background", "Language", "Noise", "Camera"]
T6 = {
    "OpenVLA": ([3.5, 28.5, 8.1, 34.8, 23.0, 15.2, 0.8], 15.6),
    "NORA": ([37.0, 62.1, 45.7, 58.6, 65.1, 12.8, 2.2], 39.0),
    "pi0": ([6.0, 68.9, 85.0, 81.4, 58.8, 79.0, 13.8], 53.6),
    "pi0-Fast": ([21.6, 68.8, 73.2, 73.2, 61.0, 74.4, 65.1], 61.6),
    "RIPT-VLA": ([31.2, 74.2, 88.4, 91.6, 77.6, 73.5, 55.2], 68.4),
    "OpenVLA-OFT": ([31.9, 74.2, 88.7, 93.3, 79.5, 75.8, 56.4], 69.6),
    "WorldVLA": ([27.9, 38.0, 43.7, 17.1, 41.6, 10.9, 0.1], 25.0),
    "UniVLA": ([46.2, 31.9, 69.0, 81.0, 69.6, 21.2, 1.8], 42.9),
    "Ours": ([73.5, 89.9, 93.1, 93.6, 75.7, 55.2, 41.5], 72.7),
}
mat = np.array([v[0] for v in T6.values()])
rhs = np.array([v[1] for v in T6.values()])
# 합 = 1 제약을 큰 가중 행으로 넣고 0 이상 제약으로 최소제곱
big = 1e4
mat2 = np.vstack([mat, big * np.ones((1, 7))])
rhs2 = np.concatenate([rhs, [big]])
sol = lsq_linear(mat2, rhs2, bounds=(0, 1))
w = sol.x
resid = mat @ w - rhs
simple_mean = mat.mean(axis=1)
RES["V3_libero_plus"] = {
    "fitted_weights": {c: round(float(x), 4) for c, x in zip(cats, w)},
    "weights_sum": round(float(w.sum()), 4),
    "max_abs_residual_pp": round(float(np.abs(resid).max()), 3),
    "residual_by_method": {k: round(float(r), 3) for k, r in zip(T6, resid)},
    "unweighted_mean_minus_reported": {k: round(float(m - v[1]), 2) for (k, v), m in zip(T6.items(), simple_mean)},
}
# 가중치가 의미 있는지: 9개 식, 6개 자유도(합=1)라 3개 식이 남는다 -> 잔차가 작으면 가중 평균이 맞다는 증거
# 전체 N 후보에서 가중치 x N 이 정수에 가까운지는 확인하지 않는다(N 미상).

# --------------------------------------------------------------------------------------
# V4. RoboTwin: Table 3 평균 vs Table 7 (수평별, 과제별) 재계산
# --------------------------------------------------------------------------------------
cols = ["XVLA_E", "XVLA_H", "pi0_E", "pi0_H", "pi05_E", "pi05_H", "Ours_E", "Ours_H"]
T = np.array([r[2] for r in TABLE7], dtype=float)
hz = np.array([r[1] for r in TABLE7])
avg50 = T.mean(axis=0)
RES["V4_table7_avg50"] = {c: {"recomputed": round(float(a), 2), "reported": rep, "diff": round(float(a) - rep, 3)}
                          for c, a, rep in zip(cols, avg50, TABLE7_AVG)}
T3 = {  # Table 3 (본문 6쪽): 수평별 평균, 열 순서 동일
    1: [81.6, 82.5, 66.5, 61.6, 85.1, 80.2, 87.9, 87.8],
    2: [59.3, 55.9, 66.1, 54.7, 79.3, 73.0, 82.1, 75.5],
    3: [61.2, 66.0, 61.6, 50.2, 78.6, 67.4, 82.4, 79.2],
}
T3_avg = [72.9, 72.8, 65.9, 58.4, 82.7, 76.8, 85.6, 83.2]
v4 = {"horizon_counts": {int(h): int((hz == h).sum()) for h in (1, 2, 3)}, "horizon_means": {}}
for h in (1, 2, 3):
    m = T[hz == h].mean(axis=0)
    v4["horizon_means"][h] = {
        "recomputed": [round(float(x), 2) for x in m], "table3": T3[h],
        "max_abs_diff": round(float(np.abs(m - np.array(T3[h])).max()), 2),
        "mismatch_cols": [cols[i] for i in range(8) if abs(round(float(m[i]), 1) - T3[h][i]) > 0.051],
    }
v4["avg_table3_vs_recomputed_max_diff"] = round(float(np.abs(avg50 - np.array(T3_avg)).max()), 3)
# Table 3 괄호 안 이득 = Ours - (두 번째 값). 두 번째로 좋은 것은 열별 최댓값 대 Ours 가 아닌 기준선 중 최댓값
gains = {}
for h in (1, 2, 3):
    for mode, (ci, oi) in {"Easy": ([0, 2, 4], 6), "Hard": ([1, 3, 5], 7)}.items():
        base = max(T3[h][c] for c in ci)
        gains[f"H{h}_{mode}"] = round(T3[h][oi] - base, 1)
for mode, (ci, oi) in {"Easy": ([0, 2, 4], 6), "Hard": ([1, 3, 5], 7)}.items():
    gains[f"Avg_{mode}"] = round(T3_avg[oi] - max(T3_avg[c] for c in ci), 1)
v4["gains_recomputed"] = gains
v4["gains_in_paper"] = {"H1_Easy": 2.8, "H1_Hard": 5.3, "H2_Easy": 2.8, "H2_Hard": 2.5, "H3_Easy": 3.8, "H3_Hard": 11.8,
                        "Avg_Easy": 2.9, "Avg_Hard": 6.4}
# 본문 "36 Hard 과제 향상, 4 동률, 10 하락" 검사 (Ours Hard vs pi0.5 Hard)
d_hard = T[:, 7] - T[:, 5]
d_easy = T[:, 6] - T[:, 4]
v4["hard_vs_pi05"] = {"up": int((d_hard > 0).sum()), "tie": int((d_hard == 0).sum()), "down": int((d_hard < 0).sum()),
                      "paper_claim": [36, 4, 10]}
v4["easy_vs_pi05"] = {"up": int((d_easy > 0).sum()), "tie": int((d_easy == 0).sum()), "down": int((d_easy < 0).sum())}
v4["hard_h3_all_improve"] = bool((d_hard[hz == 3] > 0).all())
v4["hard_h3_per_task_gain"] = {r[0]: int(d) for r, d in zip(TABLE7, d_hard) if r[1] == 3}
# 가장 큰 이득 과제와 평균에서의 비중
order = np.argsort(-d_hard)
v4["hard_top5_gains"] = [(TABLE7[i][0], int(d_hard[i])) for i in order[:5]]
v4["hard_worst5"] = [(TABLE7[i][0], int(d_hard[i])) for i in order[::-1][:5]]
# 이득 6.4 중 상위 5개 과제 몫
v4["hard_gain_total_points"] = int(d_hard.sum())
v4["hard_gain_top5_points"] = int(d_hard[order[:5]].sum())
# Horizon III Hard: 500회 vs 500회 (n=100/과제 가정)
k_o = int(T[hz == 3, 7].sum())
k_b = int(T[hz == 3, 5].sum())
lo, hi = wilson(k_o, 500)
v4["h3_hard"] = {"ours_k": k_o, "pi05_k": k_b, "n_each": 500, "ours_pct": k_o / 5, "pi05_pct": k_b / 5,
                 "fisher_p": fisher(k_o, 500, k_b, 500), "ours_wilson95": [round(100 * lo, 1), round(100 * hi, 1)]}
# 쌍체: 5개 과제 부호검정, 과제별 100회 가정 Fisher
sign5 = stats.binomtest(5, 5, 0.5, alternative="greater").pvalue
v4["h3_hard_sign_test_5of5_one_sided_p"] = round(float(sign5), 4)
v4["h3_hard_per_task_fisher_p"] = {
    r[0]: round(fisher(int(r[2][7]), 100, int(r[2][5]), 100), 4) for r in TABLE7 if r[1] == 3}
# 50과제 전체, Hard: 부호검정(동률 제외)과 대응 t, 윌콕슨
nz = d_hard[d_hard != 0]
v4["hard_sign_test_two_sided_p"] = float(stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue)
v4["hard_paired_t_p"] = float(stats.ttest_rel(T[:, 7], T[:, 5]).pvalue)
v4["hard_wilcoxon_p"] = float(stats.wilcoxon(T[:, 7], T[:, 5]).pvalue)
v4["hard_mean_diff_pp"] = round(float(d_hard.mean()), 2)
v4["hard_mean_diff_ci95_pp"] = [round(float(x), 2) for x in stats.t.interval(
    0.95, len(d_hard) - 1, loc=d_hard.mean(), scale=stats.sem(d_hard))]
v4["easy_mean_diff_pp"] = round(float(d_easy.mean()), 2)
v4["easy_paired_t_p"] = float(stats.ttest_rel(T[:, 6], T[:, 4]).pvalue)
v4["easy_mean_diff_ci95_pp"] = [round(float(x), 2) for x in stats.t.interval(
    0.95, len(d_easy) - 1, loc=d_easy.mean(), scale=stats.sem(d_easy))]
# 2-sample 정확: 전체 5000회 대 5000회
k_all_o, k_all_b = int(T[:, 7].sum() * 100 / 100), int(T[:, 5].sum())  # %를 횟수로(100회/과제)
v4["hard_all_fisher_p"] = fisher(k_all_o, 5000, k_all_b, 5000)
# 이동 가능성: Handover Mic X-VLA 0%
v4["handover_mic"] = {"xvla": [0, 0], "pi0.5": [98, 97], "ours": [86, 81]}
# Easy 수평 III 중 pi0.5 대비 하락
v4["easy_h3_per_task_delta"] = {r[0]: int(d) for r, d in zip(TABLE7, d_easy) if r[1] == 3}
# 열 합이 정수인지(소수 둘째 자리 평균 재현)
v4["integer_sum_check"] = bool(np.allclose(T.sum(axis=0) / 50, TABLE7_AVG, atol=0.006))
RES["V4_robotwin"] = v4

# --------------------------------------------------------------------------------------
# V5. 절제 표 (Table 4, 8, 9, 10) 재계산
# --------------------------------------------------------------------------------------
full = (97.40, 72.70)
T4 = [  # 이름, LIBERO, 괄호(LIBERO), Plus, 괄호(Plus)
    ("w/o Stage I", 96.28, -1.12, 69.05, -3.65),
    ("w/o world model", 96.05, -1.35, 65.49, -7.21),
    ("w/o prediction loss", 96.30, -1.10, 67.56, -5.14),
    ("w/o future conditioning", 96.85, -0.55, 69.81, -2.89),
    ("Cosmos VAE", 96.65, -0.75, 70.93, -1.77),
    ("w/o Stage II", 95.65, -1.75, 66.03, -6.67),
    ("w/o Stage I and II", 94.05, -3.35, 60.80, -11.90),
    ("w/o all future states", 93.05, -4.35, None, None),
    ("w/o far-future state", 95.45, -1.95, None, None),
    ("w/o near-future state", 95.80, -1.60, None, None),
]
v5 = []
for name, l, dl, p_, dp in T4:
    ok_l = abs(round(l - full[0], 2) - dl) < 0.006
    ok_p = True if p_ is None else abs(round(p_ - full[1], 2) - dp) < 0.006
    v5.append({"row": name, "libero_delta_ok": ok_l, "plus_delta_ok": ok_p})
RES["V5_table4_deltas"] = v5
RES["V5_text_claims"] = {
    "w/o_world_model_vs_w/o_StageI_plus_pp": round(69.05 - 65.49, 2),  # 본문 3.56
    "w/o_world_model_vs_w/o_StageI_libero_pp": round(96.28 - 96.05, 2),  # 0.23
    "cosmos_gap_plus": round(72.70 - 70.93, 2),  # 초록 +1.77
    "cosmos_gap_libero": round(97.40 - 96.65, 2),
}
# Table 10 평균, Table 8 평균·델타
T10 = {
    "w/o all": [89.60, 95.40, 97.40, 89.80, 93.05],
    "w/o far": [89.60, 97.40, 99.80, 95.00, 95.45],
    "w/o near": [92.40, 97.00, 99.00, 94.80, 95.80],
    "full": [98.90, 99.00, 97.10, 94.60, 97.40],
}
RES["V5_table10"] = {k: {"mean4": round(sum(v[:4]) / 4, 3), "reported": v[4]} for k, v in T10.items()}
RES["V5_table10_per_suite_delta_vs_full"] = {
    k: [round(a - b, 2) for a, b in zip(v[:4], T10["full"][:4])] for k, v in T10.items() if k != "full"}
T8 = {
    ("noPre", 30): [97.00, 98.40, 91.20, 89.60, 94.05],
    ("full", 30): [98.90, 99.00, 97.10, 94.60, 97.40],
    ("noPre", 50): [97.40, 98.80, 94.40, 93.20, 95.95],
    ("full", 50): [98.90, 98.90, 98.30, 91.70, 96.95],
}
RES["V5_table8"] = {f"{k[0]}_{k[1]}k": {"mean4": round(sum(v[:4]) / 4, 3), "reported": v[4]} for k, v in T8.items()}
RES["V5_table8_delta"] = {
    f"{s}k": [round(a - b, 2) for a, b in zip(T8[("full", s)], T8[("noPre", s)])] for s in (30, 50)}
RES["V5_table8_vs_pi05_long"] = {"pi05_long": 92.4, "ours_30k": 94.6, "ours_50k": 91.7}
RES["V5_checkpoint_selection"] = {
    "libero_by_ckpt_k": {"10": 78.22, "20": 94.30, "30": 97.40, "50": 96.95, "range_30_to_50": [96.47, 97.40]},
    "headline_is_max_of_ckpts_30_to_50": True,
    "gain_vs_pi05_at_30k": round(97.40 - 96.85, 2),
    "gain_vs_pi05_at_50k": round(96.95 - 96.85, 2),
    "plus_headline_30k": 72.70, "plus_50k": 70.62,
    "plus_gap_to_noPretrain_30k": round(72.70 - 60.80, 2),
    "plus_gap_to_noPretrain_50k": round(70.62 - 67.79, 2),
}
# Table 9 평균을 V3 가중치로 재계산
w_ = dict(zip(cats, w))
T9 = {"noPre": [63.23, 75.08, 86.78, 90.52, 65.91, 61.40, 44.59, 67.79],
      "full50": [68.40, 84.00, 90.20, 91.40, 75.10, 53.30, 45.10, 70.62]}
RES["V5_table9_weighted_mean"] = {
    k: {"fitted_weight_mean": round(float(np.dot(v[:7], w)), 2), "reported": v[7],
        "unweighted_mean": round(sum(v[:7]) / 7, 2)} for k, v in T9.items()}
RES["V5_table9_delta_ok"] = [round(a - b, 2) for a, b in zip(T9["full50"], T9["noPre"])]

# --------------------------------------------------------------------------------------
# V6. 절제 차이의 검정력 (LIBERO 6,000회, LIBERO-Plus N 가정)
# --------------------------------------------------------------------------------------
v6 = {}
for name, a, bb in [("w/o Stage I vs w/o world model (LIBERO)", 96.28, 96.05),
                    ("full vs w/o future conditioning (LIBERO)", 97.40, 96.85),
                    ("full vs Cosmos VAE (LIBERO)", 97.40, 96.65),
                    ("full vs w/o all future (LIBERO, inference mask)", 97.40, 93.05)]:
    z, p = z_two_prop(a / 100, 6000, bb / 100, 6000)
    v6[name] = {"diff_pp": round(a - bb, 2), "z": r2(z), "p": float(f"{p:.3g}"), "n_each": 6000}
for N in (10030, 30090):
    for name, a, bb in [("w/o Stage I vs w/o world model (Plus)", 69.05, 65.49),
                        ("full vs Cosmos VAE (Plus)", 72.70, 70.93),
                        ("full vs w/o future conditioning (Plus)", 72.70, 69.81)]:
        z, p = z_two_prop(a / 100, N, bb / 100, N)
        v6[f"{name} N={N}"] = {"diff_pp": round(a - bb, 2), "z": r2(z), "p": float(f"{p:.3g}"), "n_each": N}
# 검출 한계: 80% 검정력 최소 차이(두 비율 같은 n, p~0.97 LIBERO / 0.7 Plus)
def mde(p, n):
    return 100 * (1.959964 + 0.841621) * math.sqrt(2 * p * (1 - p) / n)
v6["mde_pp_libero_n6000_p0.97"] = r2(mde(0.97, 6000))
v6["mde_pp_plus_n10030_p0.70"] = r2(mde(0.70, 10030))
v6["mde_pp_plus_n30090_p0.70"] = r2(mde(0.70, 30090))
v6["note"] = "시행 간 분산은 평가 잡음만이다. 학습 시드가 하나라 학습 분산은 포함되지 않는다(Table 8: 체크포인트만 바꿔도 LIBERO 평균이 0.45~3.35 움직임)."
RES["V6_ablation_power"] = v6
# 학습 시드/체크포인트 분산의 대용: 30k vs 50k 에서 같은 모델 평균 차이
RES["V6_same_model_30k_vs_50k_libero_pp"] = round(97.40 - 96.95, 2)
RES["V6_same_model_suite_changes_30k_to_50k"] = {"Spatial": 0.0, "Object": -0.1, "Goal": 1.2, "Long": -2.9}

# --------------------------------------------------------------------------------------
# V7. 실로봇 (Fig. 4 막대, 프로젝트 페이지 값 = 막대와 같음)
# --------------------------------------------------------------------------------------
# 과제, 최대 점수, [pi0, pi0.5, Ours] 진행도 %
REAL = [
    ("Fold Towel", 1, [50.0, 60.0, 80.0]),
    ("Play Basketball", 2, [47.5, 42.5, 50.0]),
    ("Pack Toy", 3, [46.7, 50.0, 66.7]),
    ("Organize Stationery", 6, [43.3, 41.7, 55.0]),
    ("Fold Towel*", 1, [30.0, 50.0, 80.0]),
    ("Organize Stationery*", 6, [35.0, 40.0, 48.3]),
]
v7 = {"integrality": [], "tests": []}
for name, mx, vals in REAL:
    n_pts = 30 * mx  # 30회 x 최대 점수 = 점수 총합 분모
    row = {"task": name, "max_score": mx, "denominator_points": n_pts, "values": vals}
    row["points"] = [round(v * n_pts / 100, 2) for v in vals]
    # 가장 가까운 정수 점수로 다시 %를 만들어 소수 한 자리로 반올림했을 때 보고값과 같은지
    row["integer_ok"] = [abs(round(round(v * n_pts / 100) * 100 / n_pts, 1) - v) < 0.051 for v in vals]
    # 40회(점수 분모 80)로 읽으면 맞는지: Play Basketball 가설
    row["fits_40_trials"] = [abs(round(round(v * 40 * mx / 100) * 100 / (40 * mx), 1) - v) < 0.051 for v in vals]
    v7["integrality"].append(row)
# 이항 근사(점수를 독립 시행으로 취급, 낙관적): Fisher
for name, mx, vals in REAL:
    n_pts = 30 * mx
    ko = round(vals[2] * n_pts / 100)
    for j, bname in ((0, "pi0"), (1, "pi0.5")):
        kb = round(vals[j] * n_pts / 100)
        v7["tests"].append({"task": name, "vs": bname, "diff_pp": round(vals[2] - vals[j], 1),
                            "fisher_p_points_as_trials": round(fisher(ko, n_pts, kb, n_pts), 4),
                            "fisher_p_30_trials_binary": round(fisher(round(vals[2] * 30 / 100), 30, round(vals[j] * 30 / 100), 30), 4) if mx == 1 else None})
# 최악 분산(점수 SD <= 0.5)에서 30회 대 30회 평균 차이의 95% 반폭
v7["halfwidth95_pp_worst_sd0.5"] = r2(100 * 1.96 * 0.5 * math.sqrt(2 / 30))
v7["halfwidth95_pp_binary_p0.5"] = r2(100 * 1.96 * math.sqrt(2 * 0.25 / 30))
# 과제 6개 평균
means = np.array([v for _, _, v in REAL]).mean(axis=0)
v7["six_setting_means"] = {"pi0": r2(means[0]), "pi0.5": r2(means[1]), "ours": r2(means[2])}
means4 = np.array([v for _, _, v in REAL[:4]]).mean(axis=0)
v7["four_seen_means"] = {"pi0": r2(means4[0]), "pi0.5": r2(means4[1]), "ours": r2(means4[2])}
v7["ours_minus_pi05_each"] = {n: round(v[2] - v[1], 1) for n, _, v in REAL}
v7["trials_total_per_method_claim"] = 6 * 30  # 논문 C.2: 180
RES["V7_real_world"] = v7

# --------------------------------------------------------------------------------------
# V8. 효율 주장 산술 (그림 5 / 4.4절)
# --------------------------------------------------------------------------------------
# 그림 5의 막대 위치를 읽은 근사값 -- 논문 본문에 수치가 없다.
RES["V8_efficiency"] = {
    "text": {"added_latency_ms": 70, "added_params_B": 0.9, "ratio_vs_motus": 19, "libero_pi05": 96.9, "libero_ours": 97.4},
    "fig5_read_ours_ms_approx": 107, "fig5_read_pi05_ms_approx": 40, "fig5_read_motus_ms_approx": 1800,
    "ratio_motus_over_ours_from_fig": round(1800 / 107, 1),
    "note": "막대 위치는 그림 5의 로그 축에서 눈으로 읽은 근사값이다. Motus의 LIBERO 성공률은 그림상 Ours보다 높다(약 97.8 대 97.4).",
}

# --------------------------------------------------------------------------------------
# V9. 초록·서론 수치의 일관성
# --------------------------------------------------------------------------------------
RES["V9_abstract_claims"] = {
    "+11.8pp_H3_hard_vs_pi05": round(79.2 - 67.4, 1),
    "+1.77pp_vs_reconstruction_plus": round(72.70 - 70.93, 2),
    "claimed_1_over_19": "그림 5에서 읽은 비율 약 17, 본문 19",
    "libero_gain_vs_pi05_text_0.5": round(97.4 - 96.9, 2),
    "libero_long_gain": round(94.6 - 92.4, 2),
}

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

# --------------------------------------------------------------------------------------
# 요약 출력
# --------------------------------------------------------------------------------------
print("V1 평균 불일치 행:", [r["method"] for r in v1 if not r["ok_at_0.1"]])
print("V1 mimic-video 3개 스위트 평균:", RES["V1_mimic_video_3suite_mean"])
print("V2", json.dumps(v2, ensure_ascii=False))
print("V3 가중치", RES["V3_libero_plus"]["fitted_weights"], "최대 잔차", RES["V3_libero_plus"]["max_abs_residual_pp"])
print("V3 단순평균-보고", RES["V3_libero_plus"]["unweighted_mean_minus_reported"])
print("V4 수평별 최대 차이", {h: v4["horizon_means"][h]["max_abs_diff"] for h in (1, 2, 3)},
      "불일치 열", {h: v4["horizon_means"][h]["mismatch_cols"] for h in (1, 2, 3)})
print("V4 평균 최대 차이", v4["avg_table3_vs_recomputed_max_diff"], "이득", v4["gains_recomputed"])
print("V4 Hard vs pi0.5", v4["hard_vs_pi05"], "Easy", v4["easy_vs_pi05"])
print("V4 H3 Hard", v4["h3_hard"], "부호검정", v4["h3_hard_sign_test_5of5_one_sided_p"])
print("V4 H3 과제별 Fisher", v4["h3_hard_per_task_fisher_p"])
print("V4 Hard 평균 차이", v4["hard_mean_diff_pp"], v4["hard_mean_diff_ci95_pp"], "t p", v4["hard_paired_t_p"], "부호 p", v4["hard_sign_test_two_sided_p"])
print("V4 Easy 평균 차이", v4["easy_mean_diff_pp"], v4["easy_mean_diff_ci95_pp"], "t p", v4["easy_paired_t_p"])
print("V4 상위 이득", v4["hard_top5_gains"], "하위", v4["hard_worst5"], "총", v4["hard_gain_total_points"], "상위5", v4["hard_gain_top5_points"])
print("V5 델타", [r["row"] for r in v5 if not (r["libero_delta_ok"] and r["plus_delta_ok"])], "불일치")
print("V5 Table10 평균", RES["V5_table10"])
print("V5 Table8", RES["V5_table8"], RES["V5_table8_delta"])
print("V5 Table9 가중 평균", RES["V5_table9_weighted_mean"])
print("V6", json.dumps(v6, ensure_ascii=False, indent=1))
print("V7 정수성", [(r["task"], r["points"], r["integer_ok"], r["fits_40_trials"]) for r in v7["integrality"]])
print("V7 검정", json.dumps(v7["tests"], ensure_ascii=False))
print("V7 반폭", v7["halfwidth95_pp_worst_sd0.5"], v7["halfwidth95_pp_binary_p0.5"], v7["six_setting_means"], v7["four_seen_means"])
