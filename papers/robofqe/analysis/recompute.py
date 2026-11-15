"""RoboFQE (arXiv:2610.18699 v2) 표 재계산과 일관성 검사 (numpy, scipy). 실행: python3 analysis/recompute.py
결과: analysis/recompute_results.json
논문 표 값은 paper/paper.txt(pdftotext -layout)에서 옮겼다. 원자료가 없으므로 모든 검사는 '보고된 값끼리' 맞는지를 본다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}

# ---------------------------------------------------------------- Table 1 (LIBERO)
V = {"Spatial": 0.237, "Object": 0.166, "Goal": 0.246, "Long": 0.029}   # 표 1 캡션 (γ=0.99, 단일 정책)
SR = {"Spatial": 0.84, "Object": 0.80, "Goal": 0.77, "Long": 0.43}
# (method, suite): ARE, ARE±, MSE(e-5), MSE±, r_V, MMRV_V, r_SR, MMRV_SR
T1 = {
    ("RoboFQE", "Spatial"): (3.40, 0.08, 6.50, 0.31, 0.96, 0.03, 0.96, 0.11),
    ("RoboFQE", "Object"): (4.56, 0.17, 5.74, 0.44, 0.91, 0.02, 0.90, 0.10),
    ("RoboFQE", "Goal"): (2.34, 0.19, 3.38, 0.59, 0.90, 0.05, 0.90, 0.13),
    ("RoboFQE", "Long"): (5.63, 2.71, 0.42, 0.38, 0.88, 0.07, 0.89, 0.15),
    ("RoboFQE-scratch", "Spatial"): (9.35, 1.88, 46.7, 9.3, 0.84, 0.09, 0.84, 0.12),
    ("RoboFQE-scratch", "Object"): (13.89, 4.79, 73.2, 24.9, 0.83, 0.11, 0.83, 0.14),
    ("RoboFQE-scratch", "Goal"): (5.82, 1.72, 26.2, 7.9, 0.83, 0.09, 0.83, 0.19),
    ("RoboFQE-scratch", "Long"): (32.09, 8.83, 9.7, 2.7, 0.72, 0.17, 0.72, 0.27),
    ("WorldGym", "Spatial"): (24.20, 4.72, 337.49, 104.26, 0.76, 0.18, 0.78, 0.16),
    ("WorldGym", "Object"): (22.21, 5.66, 136.57, 64.89, 0.70, 0.20, 0.77, 0.21),
    ("WorldGym", "Goal"): (50.04, 7.47, 1572.30, 452.08, 0.62, 0.24, 0.64, 0.22),
    ("WorldGym", "Long"): (69.18, 10.63, 39.75, 12.44, -0.24, 0.68, -0.35, 0.70),
    ("Cosmos 2.5-AC", "Spatial"): (8.44, 0.19, 40.33, 2.09, 0.81, 0.10, 0.82, 0.10),
    ("Cosmos 2.5-AC", "Object"): (7.55, 0.83, 15.77, 4.30, 0.82, 0.09, 0.82, 0.09),
    ("Cosmos 2.5-AC", "Goal"): (8.03, 2.37, 39.27, 19.89, 0.84, 0.09, 0.84, 0.10),
    ("Cosmos 2.5-AC", "Long"): (54.37, 14.89, 21.65, 10.91, 0.56, 0.26, 0.60, 0.23),
    ("Ctrl-World", "Spatial"): (9.13, 1.04, 46.52, 2.15, 0.81, 0.08, 0.81, 0.09),
    ("Ctrl-World", "Object"): (8.17, 1.47, 18.45, 4.84, 0.80, 0.07, 0.82, 0.10),
    ("Ctrl-World", "Goal"): (8.71, 3.90, 46.09, 11.42, 0.81, 0.09, 0.82, 0.11),
    ("Ctrl-World", "Long"): (58.75, 7.22, 26.53, 23.42, 0.60, 0.22, 0.64, 0.22),
}

# (1) ARE 대 MSE: 같은 V와 같은 추정치라면 Jensen에 의해 mean|e|/V <= sqrt(mean e^2)/V 이다.
#     보고된 ARE가 이 상한보다 크면 기준값이나 정의가 다르다는 뜻이다.
t1_rows = []
for (m, s), (are, are_sd, mse, mse_sd, rv, mv, rs, ms) in T1.items():
    bound = math.sqrt(mse * 1e-5) / V[s] * 100          # RMS 기반 ARE 상한 (%)
    t1_rows.append({"method": m, "suite": s, "ARE_reported": are, "ARE_RMS_bound": round(bound, 2),
                    "ok": bool(are <= bound + 0.05)})
res["table1_ARE_vs_MSE"] = t1_rows
res["table1_ARE_vs_MSE_violations"] = [r for r in t1_rows if not r["ok"]]

# (2) 순위 상관: 정책 수 n에서 Pearson r의 Fisher 95% 구간과 p<0.05 임계값
def fisher_ci(r, n, alpha=0.05):
    if n <= 3:
        return None
    z = math.atanh(r)
    se = 1 / math.sqrt(n - 3)
    zc = stats.norm.ppf(1 - alpha / 2)
    return [round(math.tanh(z - zc * se), 3), round(math.tanh(z + zc * se), 3)]

def r_crit(n, alpha=0.05):
    df = n - 2
    t = stats.t.ppf(1 - alpha / 2, df)
    return round(t / math.sqrt(df + t * t), 3)

res["r_critical_p05"] = {"n=3": r_crit(3), "n=4": r_crit(4), "n=5": r_crit(5), "n=10": r_crit(10)}
res["min_one_sided_p_spearman_n4"] = round(1 / math.factorial(4), 4)
rank_ci = []
for (m, s), vals in T1.items():
    if m != "RoboFQE":
        continue
    rv = vals[4]
    rank_ci.append({"suite": s, "r_V": rv, "CI95_n4": fisher_ci(rv, 4), "significant_n4": bool(rv > r_crit(4))})
res["RoboFQE_r_CI_if_n4"] = rank_ci

# (3) 보고 순서 검사: 'RoboFQE가 모든 과제에서 r 최고, MMRV 최저'
claims = {}
for s in V:
    rf = T1[("RoboFQE", s)]
    others = [T1[(m, s)] for m in ("WorldGym", "Cosmos 2.5-AC", "Ctrl-World")]
    claims[s] = {
        "vsV_highest_r": bool(rf[4] >= max(o[4] for o in others)),
        "vsV_lowest_MMRV": bool(rf[5] <= min(o[5] for o in others)),
        "vsSR_lowest_MMRV": bool(rf[7] <= min(o[7] for o in others)),
        "vsSR_lowest_MMRV_among_cosmos_ctrl": {
            "min_value": min([("Cosmos", T1[("Cosmos 2.5-AC", s)][7]), ("Ctrl-World", T1[("Ctrl-World", s)][7])],
                             key=lambda x: x[1])},
    }
res["table1_claims_check"] = claims

# (4) 값 차이와 CI 겹침: RoboFQE 대 가장 강한 세계모델(Cosmos) ARE
res["ARE_CI_overlap_RoboFQE_vs_Cosmos"] = {
    s: {"RoboFQE": [round(T1[("RoboFQE", s)][0] - T1[("RoboFQE", s)][1], 2), round(T1[("RoboFQE", s)][0] + T1[("RoboFQE", s)][1], 2)],
        "Cosmos": [round(T1[("Cosmos 2.5-AC", s)][0] - T1[("Cosmos 2.5-AC", s)][1], 2), round(T1[("Cosmos 2.5-AC", s)][0] + T1[("Cosmos 2.5-AC", s)][1], 2)]}
    for s in V}

# 정답 V와 SR로 본 성공 궤적 길이 (V = SR * 평균 γ^(T-1), 성공이 한 길이라고 가정한 근사)
H = {"Spatial": 220, "Object": 280, "Goal": 300, "Long": 520}
implied = {}
for s in V:
    m = V[s] / SR[s]
    T_eq = 1 + math.log(m) / math.log(0.99)
    implied[s] = {"mean_discount_given_success": round(m, 4), "implied_success_step_if_constant": round(T_eq, 1),
                  "H": H[s], "ratio_to_H": round(T_eq / H[s], 2)}
res["implied_success_length"] = implied

# ---------------------------------------------------------------- Table 2 (CALVIN)
T2 = {
    "Ground Truth": ([0.96, 0.91, 0.87, 0.82, 0.76], 4.32),
    "RoboFQE": ([0.95, 0.82, 0.69, 0.61, 0.54], 3.61),
    "WorldGym": ([0.75, 0.24, 0.05, 0.00, 0.00], 1.04),
    "Cosmos 2.5-AC": ([0.98, 0.42, 0.08, 0.00, 0.00], 1.48),
    "Ctrl-World": ([0.96, 0.51, 0.10, 0.01, 0.00], 1.59),
}
t2 = {}
for k, (srk, avg) in T2.items():
    t2[k] = {"sum_SRk": round(sum(srk), 2), "reported_avg_len": avg,
             "monotone": bool(all(np.diff(srk) <= 1e-9))}
res["table2_avglen_check"] = t2
gt_avg = T2["Ground Truth"][1]
res["table2_RoboFQE_rel_gap_pct"] = round(abs(T2["RoboFQE"][1] - gt_avg) / gt_avg * 100, 1)
# 정답 샘플 수 추정: 0.96, 0.91, 0.87, 0.82, 0.76이 모두 k/n을 소수 둘째 자리로 반올림한 값인 n
gt = T2["Ground Truth"][0]
cands = []
for n in range(1, 501):
    if all(abs(round(k / n, 2) - g) < 1e-9 for k, g in zip([round(g * n) for g in gt], gt)):
        cands.append(n)
res["table2_GT_possible_n_le500"] = cands[:20]
# 20개 고정 체인에서 체인당 한 번씩이면 값은 0.05의 배수여야 한다. 실제 GT 값은 그렇지 않다.
res["table2_GT_20chain_grid_check"] = [bool(abs(g * 20 - round(g * 20)) < 1e-9) for g in gt]
# GT SR@k의 95% 이항 구간 (n=100 가정, 논문 미기재)
res["table2_GT_binomial_CI_n100"] = {f"SR@{i+1}": [round(p - 1.96 * math.sqrt(p * (1 - p) / 100), 3),
                                                  round(p + 1.96 * math.sqrt(p * (1 - p) / 100), 3)] for i, p in enumerate(gt)}
res["table2_WorldGym_SR1_gap_vs_GT"] = round(0.96 - 0.75, 2)

# ---------------------------------------------------------------- Table 3 (real world)
t3 = {"RoboFQE": (7.5, 5.0, 0.22, 0.30), "WorldGym": (17.8, 10.9, 1.24, 1.52),
      "Cosmos 2.5-AC": (24.5, 17.1, 2.35, 3.28), "Ctrl-World": (10.3, 10.2, 1.07, 1.57)}
t975_2 = stats.t.ppf(0.975, 2)  # df=2
interp = {}
for k, (m1, h1, m2, h2) in t3.items():
    sd_if_ci = h1 * math.sqrt(3) / t975_2          # 캡션 해석: ± = 95% t-CI 반폭
    interp[k] = {"ARE_mean": m1, "ARE_pm": h1, "SD_if_pm_is_CI": round(sd_if_ci, 2),
                 "CI_half_if_pm_is_SD": round(t975_2 * h1 / math.sqrt(3), 2)}
res["table3_interpretation"] = interp
# Welch t 검정: RoboFQE 대 Ctrl-World (ARE). 두 해석 모두
def welch_from_ci(m1, h1, m2, h2):
    se1 = h1 / t975_2; se2 = h2 / t975_2   # CI 해석: 평균의 표준오차
    return se1, se2
def welch_p(m1, s1m, m2, s2m):
    # s*m = 평균의 표준오차; n=3 각각 df 근사
    t = (m2 - m1) / math.sqrt(s1m ** 2 + s2m ** 2)
    df = (s1m ** 2 + s2m ** 2) ** 2 / ((s1m ** 4) / 2 + (s2m ** 4) / 2)
    return round(t, 3), round(2 * stats.t.sf(abs(t), df), 3), round(df, 2)
se_ci = welch_from_ci(7.5, 5.0, 10.3, 10.2)
se_sd = (5.0 / math.sqrt(3), 10.2 / math.sqrt(3))   # ± 가 SD인 해석: 평균의 SE = SD/sqrt(3)
res["table3_welch_RoboFQE_vs_CtrlWorld_ARE"] = {
    "if_pm_is_CI": dict(zip(["t", "p", "df"], welch_p(7.5, se_ci[0], 10.3, se_ci[1]))),
    "if_pm_is_SD": dict(zip(["t", "p", "df"], welch_p(7.5, se_sd[0], 10.3, se_sd[1]))),
}
# 이항 정확도 구간 (Wilson): LRM 라벨 정확도 83.3%(n=250 가정), 실제 정책 성공률 (n=300 = 100 x 3 과제)
def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]
# 가능한 정수 쌍 확인: 표기된 백분율이 k/n을 한 자리 소수로 반올림한 값과 맞는지
def feasible(pct, n):
    return sorted({k for k in range(n + 1) if round(100 * k / n, 1) == pct})
res["table3_feasibility"] = {
    "LRM_83.3pct_with_n250": feasible(83.3, 250),
    "LRM_83.3pct_nearest_n250_counts": [round(100 * 208 / 250, 1), round(100 * 209 / 250, 1)],
    "GT_58.5pct_with_n300": feasible(58.5, 300),
    "GT_81.0pct_with_n300": feasible(81.0, 300)[:3],
    "GT_33.0pct_with_n300": feasible(33.0, 300)[:3],
    "LRM_97.4pct_with_n1000": feasible(97.4, 1000)[:3],
    "LRM_97.4pct_with_n250": feasible(97.4, 250),
}
res["table3_wilson_LRM_83.3_n250"] = wilson(0.833, 250)
res["table3_wilson_GT_58.5_n300"] = wilson(0.585, 300)

# ---------------------------------------------------------------- Table 4, Table 5, Table 6, A.3
T4 = {"Spatial": (0.203, 6.41, 17.70), "Object": (0.154, 9.68, 23.60), "Goal": (0.256, 9.84, 82.83), "Long": (0.026, 21.85, 3.83)}
res["table4_ARE_vs_MSE"] = {s: {"ARE": a, "RMS_bound": round(math.sqrt(m * 1e-5) / v * 100, 2)} for s, (v, a, m) in T4.items()}
T5 = {"Spatial": (19.22, 159.30, 3.40, 6.50), "Object": (29.05, 212.81, 4.56, 5.74),
      "Goal": (23.07, 298.47, 2.34, 3.38), "Long": (30.85, 8.05, 5.63, 0.42)}
res["table5_demo_vs_aug"] = {s: {"ARE_demo": a, "RMS_bound_demo": round(math.sqrt(m * 1e-5) / V[s] * 100, 2),
                                 "ratio_ARE": round(a / ra, 2), "demo_ok": bool(a <= math.sqrt(m * 1e-5) / V[s] * 100 + 0.05)}
                             for s, (a, m, ra, rm) in T5.items()}
res["table5_ratio_range"] = [min(v["ratio_ARE"] for v in res["table5_demo_vs_aug"].values()),
                             max(v["ratio_ARE"] for v in res["table5_demo_vs_aug"].values())]

# A.3: 전체 LIBERO 혼합 데이터 (3000 궤적, 569k 전이)과 메모리
transitions = {"Spatial": 102_000, "Object": 121_000, "Goal": 114_000, "Long": 232_000}
bytes_per_frame = 2 * 128 * 128 * 3   # 두 카메라, uint8, 128x128 RGB (fqe/data/libero.py는 전 궤적을 RAM에 올림)
res["ram_uint8_images_GB"] = {s: round(n * bytes_per_frame / 1e9, 1) for s, n in transitions.items()}
res["ram_uint8_all_LIBERO_GB"] = round(sum(transitions.values()) * bytes_per_frame / 1e9, 1)
res["ram_uint8_all_LIBERO_GiB"] = round(sum(transitions.values()) * bytes_per_frame / 2**30, 1)
res["A3_transitions_sum"] = sum(transitions.values())

# ---------------------------------------------------------------- 요약 출력
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

print("Table1 ARE<=RMS bound violations:", [(r["method"], r["suite"], r["ARE_reported"], r["ARE_RMS_bound"]) for r in res["table1_ARE_vs_MSE_violations"]])
print("Table5 demo rows:", {s: (v["ARE_demo"], v["RMS_bound_demo"], v["demo_ok"]) for s, v in res["table5_demo_vs_aug"].items()})
print("r crit:", res["r_critical_p05"]); print("RoboFQE r CI n=4:", res["RoboFQE_r_CI_if_n4"])
print("claims:", json.dumps(claims, ensure_ascii=False)); print("implied success length:", implied)
print("Table2 sums:", t2); print("GT possible n:", cands[:10])
print("Table3 interp:", interp); print("Welch:", res["table3_welch_RoboFQE_vs_CtrlWorld_ARE"])
print("feasibility:", res["table3_feasibility"]); print("wilson:", res["table3_wilson_LRM_83.3_n250"], res["table3_wilson_GT_58.5_n300"])
print("Table4:", res["table4_ARE_vs_MSE"]); print("RAM GB:", res["ram_uint8_images_GB"], res["ram_uint8_all_LIBERO_GB"], res["ram_uint8_all_LIBERO_GiB"])
print("Table5 ratio range:", res["table5_ratio_range"]); print("CI overlap:", res["ARE_CI_overlap_RoboFQE_vs_Cosmos"])
