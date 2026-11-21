"""SCOUT (arXiv:2609.36107 v1) 보고 수치 재계산 (CPU, 수 초).

재현(학습, 시뮬 롤아웃)은 하지 않습니다. 논문 표의 요약 통계만으로 다음을 확인합니다.
  1. 표 1 정규화 점수: 작업별 평균 비율의 평균이 보고값과 맞는가
  2. 표 1 표준오차 -> 표준편차, 평균 차이의 근사 검정 (Welch 근사, 원자료 없음)
  3. 표 3 세그먼트 평균이 단순평균과 맞는가
  4. 표 2, 4, 6 성공률의 이항 신뢰구간 (시도 수는 논문에 없어 저장소 기준 34 x 5 = 170을 가정)
  5. 표 5 실로봇 U 통계량과 Psup, 정규근사 p, 위치별 평균 -> 풀링 평균 일치
  6. 부록 C.1 Holm 보정 값들의 상호 일관성
출력: recompute_results.json
"""
import json
import math
import os
from itertools import product

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}

# ── 1. 표 1 ISE 평균 시도 수 (보고값, 논문 Table 1) ─────────────────────────
tasks = ["Slide Brick", "Push Bar", "Pick Bar", "Open Box", "Turn Faucet"]
table1 = {
    "CQL":          ([9.94, 10.96, 10.22, 4.32, 6.63], [0.18, 0.20, 0.19, 0.18, 0.21], 3.17),
    "BC":           ([10.24, 11.26, 9.79, 2.59, 4.54], [0.33, 0.30, 0.32, 0.19, 0.27], 2.72),
    "AVDC":         ([8.36, 7.83, 5.41, 2.82, 2.67], [0.27, 0.27, 0.22, 0.16, 0.16], 1.91),
    "ISE":          ([7.69, 5.26, 4.84, 2.25, 2.39], [0.26, 0.20, 0.18, 0.10, 0.14], 1.61),
    "PEARL":        ([5.03, 4.51, 4.89, 2.24, 1.54], [0.23, 0.20, 0.25, 0.12, 0.03], 1.31),
    "Raw GMM":      ([6.39, 7.95, 4.96, 2.41, 2.22], [0.28, 0.25, 0.20, 0.10, 0.07], 1.67),
    "History GMM":  ([3.92, 4.65, 3.36, 1.87, 1.61], [0.23, 0.22, 0.11, 0.05, 0.03], 1.11),
    "SCOUT":        ([3.37, 4.14, 2.83, 1.83, 1.54], [0.19, 0.18, 0.17, 0.05, 0.03], 1.00),
}
scout_m = np.array(table1["SCOUT"][0])
norm_check = {}
for name, (m, se, reported) in table1.items():
    ratio = np.mean(np.array(m) / scout_m)
    norm_check[name] = {"recomputed_mean_ratio": round(float(ratio), 3), "reported": reported}
out["table1_normalized_check"] = norm_check

# 표준편차 역산 (n = 400 가정, 논문 본문 "400 rollouts"; 저장소 README는 pick/push/brick에 800을 씀)
sd_implied = {}
for name in ["SCOUT", "History GMM"]:
    m, se, _ = table1[name]
    sd_implied[name] = {t: {"sd_if_n400": round(s * math.sqrt(400), 2),
                            "sd_if_n800": round(s * math.sqrt(800), 2)}
                        for t, s in zip(tasks, se)}
out["table1_sd_implied"] = sd_implied

# Welch 근사 (정규 가정): SCOUT 대 History GMM, 대 다른 기준선 (원자료 없음, 참고용)
def welch_p(m1, s1, m2, s2):
    d = m1 - m2
    se = math.sqrt(s1 ** 2 + s2 ** 2)
    z = d / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return d, se, z, p

reported_padj = {  # 부록 C.1 (Mann-Whitney, Holm)
    "Slide Brick": 4.93e-24, "Push Bar": 7.29e-13, "Pick Bar": 3.75e-26,
    "Open Box": 7.40e-5, "Turn Faucet": 1.0,
}
hist = table1["History GMM"]
sc = table1["SCOUT"]
welch = {}
for i, t in enumerate(tasks):
    d, se, z, p = welch_p(hist[0][i], hist[1][i], sc[0][i], sc[1][i])
    welch[t] = {"diff_hist_minus_scout": round(d, 3), "se_diff": round(se, 3),
                "z": round(z, 2), "p_normal_approx": round(p, 4),
                "paper_padj_MW_holm": reported_padj[t]}
out["table1_welch_vs_history_gmm"] = welch

# Holm 일관성 (m=5): padj_(k) = max_{j<=k} (m-j+1) p_(j).
# 보고된 padj 순서가 단조 증가하는지, padj로부터 역산한 raw p를 확인
pv = sorted(reported_padj.items(), key=lambda kv: kv[1])
holm_monotone = all(pv[i][1] <= pv[i + 1][1] for i in range(len(pv) - 1))
out["appC1_holm_check"] = {
    "ordering_by_padj": [k for k, _ in pv],
    "monotone": holm_monotone,
    "turn_faucet_rank5_raw_p_implied": "padj=1.0 -> raw p 약 1 (순위 5, 승수 1)",
    "open_box_rank4_raw_p_implied": round(7.40e-5 / 2, 8),  # 승수 (5-4+1)=2
}

# ── 3. 표 3 세그먼트 평균 ─────────────────────────────────────────────────
t3 = {
    "Pick Bar":    {"train_segs": [2.8, 2.2, 1.7], "train_reported": 2.25, "held_segs": [4.0, 5.1], "held_reported": 4.56},
    "Push Bar":    {"train_segs": [3.7, 5.7, 3.9], "train_reported": 4.48, "held_segs": [5.9, 6.8], "held_reported": 6.34},
    "Slide Brick": {"train_segs": [4.6, 3.0, 2.1], "train_reported": 3.14, "held_segs": [3.0, 3.7], "held_reported": 3.36},
}
for k, v in t3.items():
    v["train_simple_mean"] = round(float(np.mean(v["train_segs"])), 3)
    v["held_simple_mean"] = round(float(np.mean(v["held_segs"])), 3)
    v["train_gap"] = round(v["train_reported"] - v["train_simple_mean"], 3)
    v["held_gap"] = round(v["held_reported"] - v["held_simple_mean"], 3)
out["table3_segment_means"] = t3

# ── 4. 문-문 수치: 표 2, 4, 6 성공률 (n 가정) ───────────────────────────
def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h

reported_sr = {
    "PNDiT (generator only)": 0.68, "Raw GMM": 0.61, "History GMM": 0.68,
    "HAVE 5 samples": 0.82, "HAVE 20 samples": 0.84, "SCOUT 1 step": 0.83, "SCOUT 5 steps": 0.97,
    "w/o recon 1 step": 0.71, "w/o recon 5 steps": 0.73,
    "estimated flow 1 step": 0.83, "estimated flow 5 steps": 0.96,
}
N_DOOR = 170  # 34 test objects x 5 trials (metadata/multimodal_door.json, rollout.trials_per_object=5), 저장소 기준 가정
door = {}
for name, v in reported_sr.items():
    k = round(v * N_DOOR)
    lo, hi = wilson(k, N_DOOR)
    door[name] = {"sr": v, "k_of_170": k, "wilson95": [round(lo, 3), round(hi, 3)]}
out["door_sr_wilson_n170"] = door

def two_prop_p(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return z, 2 * (1 - stats.norm.cdf(abs(z)))

kS1 = round(0.83 * N_DOOR)
kS5 = round(0.97 * N_DOOR)
comp = {}
for name, key in [("History GMM", round(0.68 * N_DOOR)), ("HAVE 20 samples", round(0.84 * N_DOOR)),
                  ("HAVE 5 samples", round(0.82 * N_DOOR)), ("PNDiT (generator only)", round(0.68 * N_DOOR)),
                  ("Raw GMM", round(0.61 * N_DOOR))]:
    z1, p1 = two_prop_p(kS1, N_DOOR, key, N_DOOR)
    z5, p5 = two_prop_p(kS5, N_DOOR, key, N_DOOR)
    comp[name] = {"SCOUT1_vs_p": round(p1, 4), "SCOUT5_vs_p": round(p5, 6)}
out["door_two_prop_tests_n170"] = comp
out["door_recon_ablation_1step_vs_full_1step"] = {
    "note": "w/o recon 0.71 vs full 0.83 at 1 step, n=170 assumed",
    "p": round(two_prop_p(round(0.71 * N_DOOR), N_DOOR, round(0.83 * N_DOOR), N_DOOR)[1], 6)}

# ── 5. 표 5 실로봇 U, Psup, 정규근사 p, 위치별 평균 ─────────────────────
n1 = n2 = 30
rw = {}
for task, U, psup_rep, p_rep, mean_s, mean_b in [
        ("Pick Bar", 311.5, 0.65, 0.035, 2.17, 3.27),
        ("Push Bar", 250.0, 0.72, 0.003, 2.90, 5.60)]:
    mu_U = n1 * n2 / 2
    sd_U = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)  # 동점 보정 없음
    z = (U - mu_U + 0.5) / sd_U
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    rw[task] = {
        "U_reported": U,
        "psup_if_U_counts_SCOUT_fewer": round(U / (n1 * n2), 3),
        "psup_if_U_counts_SCOUT_more": round(1 - U / (n1 * n2), 3),
        "psup_reported": psup_rep,
        "z_no_tie_correction": round(z, 3),
        "p_no_tie_correction": round(p, 4),
        "p_reported_raw": p_rep,
        "mean_scout_reported": mean_s, "mean_baseline_reported": mean_b,
    }
out["table5_mann_whitney_check"] = rw

# 위치별 표 3 (Figure 3 텍스트 값, 95% CI 반폭, n=10 -> t(9)=2.262)
t975 = stats.t.ppf(0.975, 9)
fig3 = {
    "Pick Bar": {"baseline": [(3.2, 1.9), (3.6, 1.8), (3.0, 1.6)],
                 "scout": [(2.0, 0.5), (2.7, 2.0), (1.8, 0.7)]},
    "Push Bar": {"baseline": [(6.1, 3.0), (6.1, 2.5), (4.6, 2.0)],
                 "scout": [(2.9, 0.8), (3.7, 0.5), (2.1, 2.0)]},
}
fig3_out = {}
for task, arms in fig3.items():
    rec = {}
    for arm, cells in arms.items():
        means = np.array([m for m, _ in cells])
        sds = np.array([h / t975 * math.sqrt(10) for _, h in cells])  # CI 반폭 -> 표준편차
        pooled_mean = means.mean()  # 위치별 n=10 동일 -> 풀링 평균 = 산술평균
        within = np.sum((10 - 1) * sds ** 2) / (30 - 1)
        between = np.sum(10 * (means - pooled_mean) ** 2) / (30 - 1)
        rec[arm] = {"position_means": means.tolist(), "pooled_mean": round(float(pooled_mean), 3),
                    "position_sds": [round(float(s), 3) for s in sds],
                    "pooled_sd": round(float(math.sqrt(within + between)), 3)}
    a, b = rec["scout"], rec["baseline"]
    tt = stats.ttest_ind_from_stats(b["pooled_mean"], b["pooled_sd"], 30,
                                    a["pooled_mean"], a["pooled_sd"], 30, equal_var=False)
    rec["welch_t_p_pooled"] = round(float(tt.pvalue), 4)
    fig3_out[task] = rec
out["fig3_position_pooling_check"] = fig3_out

with open(os.path.join(HERE, "recompute_results.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print('written', len(json.dumps(out)))
