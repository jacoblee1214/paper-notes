"""OPENDEXGRASP (arXiv:2609.18117 v4) 보고 수치의 재계산과 내부 일관성 검사.

실행: python3 analysis/recompute.py  → analysis/recompute_results.json (numpy, scipy, 1초 미만)
모든 입력 값은 paper/paper.txt(v4, pdftotext -layout)의 표에서 옮겼습니다.
원자료(시도별 성공 여부, 샘플별 점수)는 공개되지 않았으므로 검정은 집계값과 가정한 단위로만 합니다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}

# ---------- 표 1: 학습 데이터 ----------
t1 = {"scale_grasps": 1.24e6, "scale_functional": 557.18e3, "align_teleop": 159 * 10, "align_transfer": 25835}
res["t1_scale_functional_ratio_pct"] = round(100 * 557.18 / 1240, 2)          # 표 1: 44.93
res["t1_align_functional_ratio_pct"] = round(100 * 18.65 / 27.42, 2)          # 표 1: 68.02
res["t1_align_total_grasps"] = t1["align_teleop"] + t1["align_transfer"]      # 부록 B.2: 27,425
res["t1_align_total_vs_table_27.42k"] = round(res["t1_align_total_grasps"] / 1000, 3)
res["b1_category_size_pairs"] = 56 * 1 + 18 * 2 + 31 * 3                       # 185
res["b1_categories"] = 56 + 18 + 31                                            # 105
res["b2_teleop_grasps_check"] = 159 * 10                                       # 1,590

# ---------- 표 2: 시뮬레이션 (seen-func, unseen-func, seen-nonfunc, unseen-nonfunc) ----------
# 열: SIV, PD, SD, SR(%), Style Div.  (기능 분할만 GPT-5, 인간 점수 있음)
T2 = {
    "DGN2star": {"SIV": [5.12, 6.86, 4.98, 5.16], "PD": [1.04, 1.62, 0.87, 0.99],
                 "SD": [1.51, 4.29, 2.53, 3.87], "SR": [50.88, 43.22, 63.27, 54.49],
                 "Sty": [0.95, 0.81, 1.14, 1.03]},
    "Ours": {"SIV": [1.28, 1.39, 1.33, 1.30], "PD": [0.39, 0.48, 0.42, 0.44],
             "SD": [1.37, 1.80, 0.94, 2.29], "SR": [68.07, 62.96, 75.46, 66.73],
             "Sty": [1.39, 1.41, 1.55, 1.25]},
}
splits = ["func-seen", "func-unseen", "nonfunc-seen", "nonfunc-unseen"]
# 표 2 개선 여부: 네 분할 모두에서 SIV, PD, SD가 낮은가
res["t2_ours_lower_SIV_PD_SD_all_splits"] = all(
    T2["Ours"][k][i] < T2["DGN2star"][k][i] for k in ["SIV", "PD", "SD"] for i in range(4))
res["t2_SR_gain_pp"] = {s: round(T2["Ours"]["SR"][i] - T2["DGN2star"]["SR"][i], 2) for i, s in enumerate(splits)}
res["t2_SIV_reduction_pct"] = {s: round(100 * (1 - T2["Ours"]["SIV"][i] / T2["DGN2star"]["SIV"][i]), 1) for i, s in enumerate(splits)}

# 표 6 인스턴스 수(표 6 본문 기준). 표 3 가중 평균 재현 시도.
w_inst = np.array([75, 15, 138, 15], float)
w_inst = w_inst / w_inst.sum()
wavg = {}
for m in T2:
    for k in ["SIV", "PD", "SD", "SR", "Sty"]:
        wavg[f"{m}_{k}"] = round(float(np.dot(w_inst, T2[m][k])), 3)
res["t2_weighted_avg_by_instance_75_15_138_15"] = wavg
# 표 3 "OpenDexGrasp" 행과 "w/o pretrained VLM" 행(=CLIP 대체)과 비교
T3_full = {"SIV": 1.31, "PD": 0.41, "SD": 1.20, "SR": 71.65, "Sty": 1.47}
T3_noVLM = {"SIV": 5.15, "PD": 0.98, "SD": 2.32, "SR": 57.26, "Sty": 1.05}
res["t3_full_minus_weighted_inst"] = {k: round(T3_full[k] - wavg[f"Ours_{k}"], 3) for k in T3_full}
res["t3_noVLM_minus_DGN2star_weighted_inst"] = {k: round(T3_noVLM[k] - wavg[f"DGN2star_{k}"], 3) for k in T3_noVLM}

# ---------- 샘플 수 기반 근사 검정 (단위 = 인스턴스) ----------
# 표 6 인스턴스 수. 시도 수(10회/인스턴스)를 쓰면 독립 가정이 과장되므로 보수적으로 인스턴스 수를 n으로 쓴다.
def two_prop_z(p1, n1, p2, n2):
    p1, p2 = p1 / 100, p2 / 100
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = (p1 - p2) / se
    return z, 2 * (1 - stats.norm.cdf(abs(z)))

n_inst = [75, 15, 138, 15]
zres = {}
for i, s in enumerate(splits):
    z, p = two_prop_z(T2["Ours"]["SR"][i], n_inst[i], T2["DGN2star"]["SR"][i], n_inst[i])
    zres[s] = {"z_instance_n": round(z, 2), "p_two_sided": round(p, 4)}
res["t2_SR_z_with_instance_n"] = zres
# 시도 수 기반 (인스턴스 x 10회)는 독립 가정 상한
zres_trials = {}
for i, s in enumerate(splits):
    z, p = two_prop_z(T2["Ours"]["SR"][i], n_inst[i] * 10, T2["DGN2star"]["SR"][i], n_inst[i] * 10)
    zres_trials[s] = {"z": round(z, 2), "p": float(f"{p:.2e}")}
res["t2_SR_z_if_trials_independent_upper_bound"] = zres_trials

# ---------- SR 분모 역산: 10회 x N 에서 정수 성공 수가 나오는가 ----------
def integer_fraction(sr_pct, n_samples, trials=10):
    total = n_samples * trials
    k = sr_pct / 100 * total
    return round(k, 2), round(abs(k - round(k)), 2)

res["t2_SR_integer_check_10trials"] = {
    f"{m}_{s}": {"N": n_inst[i], "successes_expected": integer_fraction(T2[m]["SR"][i], n_inst[i])[0],
                 "distance_to_integer": integer_fraction(T2[m]["SR"][i], n_inst[i])[1]}
    for m in T2 for i, s in enumerate(splits)
}

# ---------- 표 3 단면 민감도 (분모 미기재): 243 인스턴스를 군집 수로 쓴 보수적 근사 ----------
def diff_z(p1, p2, n):
    p1, p2 = p1 / 100, p2 / 100
    se = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    z = (p1 - p2) / se
    return round(z, 2), round(2 * (1 - stats.norm.cdf(abs(z))), 4)

ab = {"w/o affordance grounding": 67.49, "w/o OpenDex-Align": 69.38, "reduced OpenDex-Scale": 60.52, "w/o pretrained VLM": 57.26}
res["t3_SR_diff_pp_and_z_n243_clusters"] = {
    k: {"diff_pp": round(71.65 - v, 2), "z_p": diff_z(71.65, v, 243)} for k, v in ab.items()
}

# ---------- 표 4: 실로봇 (객체 10개, 객체당 10회 실행 = 객체당 시도 10, 방법당 100) ----------
objects = ["Rice Paddle", "Dustpan", "Bouquet", "Pitcher", "Small Bucket",
           "Bottle", "Umbrella", "Shampoo", "Hammer", "Brush"]
sr_dgn = [40, 60, 40, 50, 50, 70, 80, 60, 80, 60]
sr_our = [60, 70, 50, 80, 50, 100, 90, 60, 90, 70]
gpt_dgn = [6.2, 7.1, 5.5, 6.7, 7.4, 7.6, 6.8, 6.4, 7.0, 6.1]
gpt_our = [8.1, 7.2, 6.3, 7.5, 7.3, 8.6, 7.2, 6.7, 8.3, 7.4]
hum_dgn = [6.5, 6.9, 4.8, 6.1, 7.9, 8.0, 5.9, 4.8, 7.6, 6.8]
hum_our = [7.7, 7.9, 7.1, 8.0, 7.1, 9.3, 7.6, 6.8, 8.1, 7.6]
res["t4_avg_check"] = {
    "SR_dgn_mean": round(np.mean(sr_dgn), 2), "SR_our_mean": round(np.mean(sr_our), 2),
    "GPT_dgn_mean": round(np.mean(gpt_dgn), 2), "GPT_our_mean": round(np.mean(gpt_our), 2),
    "HUM_dgn_mean": round(np.mean(hum_dgn), 2), "HUM_our_mean": round(np.mean(hum_our), 2),
}
succ_dgn, succ_our = sum(x for x in sr_dgn) // 10, sum(x for x in sr_our) // 10
res["t4_success_counts_of_100"] = {"dgn2star": succ_dgn, "ours": succ_our}
table = [[succ_our, 100 - succ_our], [succ_dgn, 100 - succ_dgn]]
fe_or, fe_p = stats.fisher_exact(table)
res["t4_fisher_exact_72_vs_59_of_100"] = {"odds_ratio": round(fe_or, 3), "p_two_sided": round(fe_p, 4)}
chi = stats.chi2_contingency(table, correction=False)
res["t4_chi2_72_vs_59"] = {"p": round(chi[1], 4)}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 1), round(100 * (c + h), 1)


res["t4_wilson95_ci_pct"] = {"dgn2star": wilson(succ_dgn, 100), "ours": wilson(succ_our, 100)}
d_sr = np.array(sr_our) - np.array(sr_dgn)
wins = int((d_sr > 0).sum()); ties = int((d_sr == 0).sum()); losses = int((d_sr < 0).sum())
sign_p = stats.binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else None
res["t4_per_object_SR_diff"] = {"wins": wins, "ties": ties, "losses": losses,
                                "sign_test_p_two_sided": round(float(sign_p), 4),
                                "mean_diff_pp": round(float(d_sr.mean()), 2),
                                "paired_t_p": round(float(stats.ttest_rel(sr_our, sr_dgn).pvalue), 4),
                                "wilcoxon_p": round(float(stats.wilcoxon(sr_our, sr_dgn).pvalue), 4) if np.any(d_sr != 0) else None}
for name, a, b in [("GPT5", gpt_our, gpt_dgn), ("Human", hum_our, hum_dgn)]:
    d = np.array(a) - np.array(b)
    res[f"t4_per_object_{name}_diff"] = {"wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
                                         "losses": int((d < 0).sum()), "mean_diff": round(float(d.mean()), 2),
                                         "wilcoxon_p": round(float(stats.wilcoxon(a, b).pvalue), 4),
                                         "paired_t_p": round(float(stats.ttest_rel(a, b).pvalue), 4)}

# ---------- 표 7: 추론 시간 ----------
res["t7_speedup_vs_opt_baseline"] = round(2.84 / 0.93, 2)
res["t7_ours_time_s"] = 0.93

# ---------- 기타 판정 상수 ----------
# 성공 기준: 객체 변위 < 0.2 m (2초, 손 고정). 표 2 SD는 성공 시도만 평균
res["success_threshold_m"] = 0.2
res["ratio_threshold_to_SD_ours_functional_seen"] = round(20.0 / T2["Ours"]["SD"][0], 1)  # 20 cm / 1.37 cm

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
