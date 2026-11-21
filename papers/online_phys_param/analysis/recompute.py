"""Beyond Semantic Classes (arXiv:2611.10107 v1) 본문 수치 재계산과 일관성 검사.

입력: 논문 본문과 표·그림 주석에서 옮겨 적은 값 (paper/paper.txt의 쪽 번호를 주석에 씀).
출력: analysis/recompute_results.json
실행: python3 analysis/recompute.py   (numpy, scipy, 1초 미만)

재현(학습, 시뮬레이션 실행)은 하지 않는다. 표와 본문에 있는 숫자만 다시 계산한다.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}

# ---------- 1. 9회 주행 시간과 속도 (본문 §4.3, 8쪽) ----------
# "Across nine complete traversals ... 9.304±0.316 to 8.135±0.007 s", "mean speed 10.51 to 11.78 m/s"
# 구간 길이: s = 200–300 m (8쪽). ±의 의미(SD인지 SE인지)는 논문에 없다.
N_TRAV = 9
T_BASE, S_BASE = 9.304, 0.316
T_PROP, S_PROP = 8.135, 0.007
V_BASE, V_PROP = 10.51, 11.78
L_SEG = 100.0

rel_time = (T_BASE - T_PROP) / T_BASE
rel_speed = (V_PROP - V_BASE) / V_BASE
res["traversal_time_reduction_pct"] = round(100 * rel_time, 2)  # 논문 12.6%
res["speed_increase_pct"] = round(100 * rel_speed, 2)
res["sd_ratio_base_over_prop"] = round(S_BASE / S_PROP, 1)
res["var_ratio_base_over_prop"] = round((S_BASE / S_PROP) ** 2, 0)


def welch_for(sd_or_se: str):
    """±를 SD로 읽을 때와 SE로 읽을 때의 Welch t 검정과 95% CI."""
    if sd_or_se == "SD":
        se_b, se_p = S_BASE / np.sqrt(N_TRAV), S_PROP / np.sqrt(N_TRAV)
    else:
        se_b, se_p = S_BASE, S_PROP
    diff = T_BASE - T_PROP
    se_d = np.sqrt(se_b ** 2 + se_p ** 2)
    t = diff / se_d
    df = se_d ** 4 / (se_b ** 4 / (N_TRAV - 1) + se_p ** 4 / (N_TRAV - 1))
    p = 2 * stats.t.sf(abs(t), df)
    tc = stats.t.ppf(0.975, N_TRAV - 1)
    return {
        "reading": sd_or_se,
        "welch_t": round(float(t), 2),
        "welch_df": round(float(df), 2),
        "p_two_sided": float(f"{p:.2e}"),
        "ci95_base": [round(T_BASE - tc * se_b, 3), round(T_BASE + tc * se_b, 3)],
        "ci95_prop": [round(T_PROP - tc * se_p, 3), round(T_PROP + tc * se_p, 3)]
    }


res["welch_sd_reading"] = welch_for("SD")
res["welch_se_reading"] = welch_for("SE")

# 평균 속도와 L/T 관계: 시간 평균 V는 정확히 L/T이고, 거리 평균 V는 조화평균 L/T 이상이다(AM-HM).
# 따라서 보고된 평균 속도가 L/T보다 작으면 정의나 구간 길이가 논문 설명과 다르다.
res["speed_check"] = {
    "L_over_T_base": round(L_SEG / T_BASE, 3),
    "L_over_T_prop": round(L_SEG / T_PROP, 3),
    "reported_V_base": V_BASE,
    "reported_V_prop": V_PROP,
    "gap_pct_base": round(100 * (V_BASE / (L_SEG / T_BASE) - 1), 2),
    "gap_pct_prop": round(100 * (V_PROP / (L_SEG / T_PROP) - 1), 2),
    "implied_L_if_time_average_base": round(V_BASE * T_BASE, 2),
    "implied_L_if_time_average_prop": round(V_PROP * T_PROP, 2),
}

# 추종 오차 RMS 감소율 (8쪽)
rms = {"lateral_m": (1.637, 0.121), "heading_rad": (0.173, 0.017), "sideslip_rad": (0.089, 0.027)}
res["rms_reduction_pct"] = {k: round(100 * (a - b) / a, 1) for k, (a, b) in rms.items()}

# ---------- 2. 실기 궤적 예측 (§4.4, 9쪽) ----------
ADE = (0.737, 0.628)
FDE = (2.096, 1.750)
res["real_ade_reduction_pct"] = round(100 * (ADE[0] - ADE[1]) / ADE[0], 1)  # 논문 14.8%
res["real_fde_reduction_pct"] = round(100 * (FDE[0] - FDE[1]) / FDE[0], 1)  # 논문 16.5%

# ---------- 3. Table 1 질의 지연 (4.2절, 7쪽) ----------
# 행: N, 열: S. (CPU ms, GPU ms)
T1 = {
    1e4: {10: (0.51, 0.68), 100: (0.54, 0.68), 1000: (0.81, 0.74)},
    1e5: {10: (3.62, 2.06), 100: (4.11, 2.07), 1000: (4.01, 2.07)},
    1e6: {10: (51.61, 15.39), 100: (51.45, 15.36), 1000: (49.84, 15.22)},
}
N_list = sorted(T1)
cpu_s = [T1[n][1000][0] for n in N_list]
gpu_s = [T1[n][1000][1] for n in N_list]
res["table1_cpu_growth_per_decade"] = [round(cpu_s[i + 1] / cpu_s[i], 2) for i in range(2)]
res["table1_gpu_growth_per_decade"] = [round(gpu_s[i + 1] / gpu_s[i], 2) for i in range(2)]
res["table1_cpu_loglog_slope_1e5_to_1e6"] = round(float(np.log10(cpu_s[2] / cpu_s[1])), 2)
res["table1_gpu_loglog_slope_1e5_to_1e6"] = round(float(np.log10(gpu_s[2] / gpu_s[1])), 2)
res["table1_cpu_ns_per_entry_at_1e6"] = round(cpu_s[2] * 1e6 / 1e6, 1)  # ms per 1e6 entries -> ns per entry
res["table1_gpu_ns_per_entry_at_1e6"] = round(gpu_s[2] * 1e6 / 1e6, 1)
# S 증가에 따른 변화(N=1e6): CPU 49.84 대 51.61, 차이 3% 이내 -> 샘플 수가 비용을 거의 바꾸지 않음
res["table1_S_effect_at_1e6_cpu_pct"] = round(100 * (49.84 - 51.61) / 51.61, 1)

# 임베딩 저장량 추정 (논문에 임베딩 차원이 없어 768 가정: DINOv3 ViT-B/16 특징 차원, 식 (7) 위 설명)
D_EMB = 768
res["db_embedding_gib_fp32_at_1e6"] = round(1e6 * D_EMB * 4 / 2**30, 2)
res["db_embedding_gib_fp16_at_1e6"] = round(1e6 * D_EMB * 2 / 2**30, 2)

# ---------- 4. 그림 10 주석 ρ의 일관성 (부록 C.6, 16~17쪽) ----------
# 열: 이력 길이 t. 행: 지지 기준 κ. 본문 식 ρ = M/(M+κ)이므로 M = κρ/(1-ρ).
# 두 자리 반올림(±0.005)을 허용하고, 같은 열의 모든 행에서 M 구간이 겹치는지 본다.
T_COLS = [0, 5, 10, 25, 50, 100, 315]
RHO = {
    0.1: [0.00, 0.98, 0.99, 0.99, 1.00, 1.00, 1.00],
    1.0: [0.00, 0.83, 0.90, 0.95, 0.97, 0.98, 0.99],
    5.0: [0.00, 0.49, 0.63, 0.78, 0.86, 0.92, 0.97],
    10.0: [0.00, 0.32, 0.46, 0.63, 0.76, 0.85, 0.94],
    100.0: [0.00, 0.05, 0.08, 0.15, 0.24, 0.36, 0.62],
}


def m_interval(kappa, rho):
    lo, hi = max(rho - 0.005, 0.0), min(rho + 0.005, 0.99999)
    m_lo = kappa * lo / (1 - lo) if lo > 0 else 0.0
    m_hi = kappa * hi / (1 - hi)
    return m_lo, m_hi


fig10 = []
for j, t in enumerate(T_COLS):
    if t == 0:
        fig10.append({"t": t, "M_common": [0.0, 0.0], "consistent": True})
        continue
    lo_all, hi_all = 0.0, np.inf
    per_row = {}
    for kappa, rhos in RHO.items():
        a, b = m_interval(kappa, rhos[j])
        per_row[str(kappa)] = [round(a, 2), round(b, 2)]
        lo_all, hi_all = max(lo_all, a), min(hi_all, b)
    ok = lo_all <= hi_all
    fig10.append({"t": t, "per_row_M_interval": per_row,
                  "M_common": [round(lo_all, 2), round(hi_all, 2)] if ok else None,
                  "consistent": bool(ok)})
res["fig10_column_consistency"] = fig10
res["fig10_M_over_t_at_consistent_cols"] = {
    str(c["t"]): round(float(np.mean(c["M_common"])) / c["t"], 3)
    for c in fig10 if c["t"] > 0 and c["M_common"]}

# ---------- 5. 커널 값과 지지 혼합 가중 (§3, 부록 B.4) ----------
# K = exp(γ(cos - 1)), cos < c이면 0. 실험별 γ, κ, 바닥 c는 부록 B.4 값.
SETTINGS = {
    "E1 (γ=400, κ=5e-3)": (400, 5e-3, None),
    "E2 (γ=40, κ=5e-3)": (40, 5e-3, None),
    "racing (γ=100, κ=1e-2, c=0.9)": (100, 1e-2, 0.9),
    "real vehicle (γ=10, κ=2, c=0.9)": (10, 2.0, 0.9),
}
kern = {}
for name, (g, kap, c) in SETTINGS.items():
    row = {}
    for cos in (0.90, 0.95, 0.99, 1.0):
        k = 0.0 if (c is not None and cos < c) else float(np.exp(g * (cos - 1)))
        row[str(cos)] = {"K": float(f"{k:.3g}"), "rho_single_neighbor": float(f"{k / (k + kap):.3g}")}
    kern[name] = row
res["kernel_single_neighbor"] = kern
# 레이싱: 이웃 하나로 ρ = 0.5가 되는 최소 코사인 (M = κ)
g, kap = 100, 1e-2
res["racing_min_cos_for_rho_half_single_neighbor"] = round(1 + np.log(kap) / g, 4)
# 실기: κ=2를 넘으려면 필요한 이웃 수 (cos=0.9, 0.99)
for cos in (0.9, 0.99):
    k = np.exp(10 * (cos - 1))
    res[f"real_neighbors_for_rho_half_at_cos_{cos}"] = round(2.0 / k, 1)

# ---------- 6. 표 3(부록 C.2)의 튜닝 벡터 차원 ----------
TUNE = ["c_beta_alpha_f", "c_beta_alpha_r", "c_beta_kappa_f", "c_beta_kappa_r",
        "ds_fx", "ds_fy", "ds_rx", "ds_ry",
        "log_gamma_xf", "log_gamma_yf", "log_gamma_xr", "log_gamma_yr",
        "log_k_kappa_f", "log_k_kappa_r", "log_mu_f", "log_mu_r",
        "log_rho_f", "log_rho_r", "log_tau_f", "log_tau_r", "compat_k_beta_alpha_rdot"]
UNUSED = ["compat_k_beta_alpha_rdot"]  # 표 3 "Neutral, unused interface entry"
res["tune_vector_length_table3"] = len(TUNE)
res["tune_vector_effective_length"] = len(TUNE) - len(UNUSED)
res["paper_claims_dim"] = 21

# ---------- 7. 파이프라인 계수 (부록 A.2, B.1–B.2) ----------
n_img, patches_per_img = 1500, 32 * 32
total = n_img * patches_per_img
sampled, noise_sampled = 100_000, 35_693
assigned, noise_all = 1_532_716, 3_284
res["patch_total_check"] = {"expected_total": total, "paper_total": 1_536_000, "ok": total == 1_536_000}
res["assigned_plus_noise_check"] = {"sum": assigned + noise_all, "total": total, "ok": assigned + noise_all == total}
res["hdbscan_noise_fraction_in_sample_pct"] = round(100 * noise_sampled / sampled, 1)
res["hdbscan_assigned_fraction_pct"] = round(100 * assigned / total, 2)

# ---------- 8. 표 2 NIG 매개변수 (부록 B.4) ----------
# 논문 본문은 NIG(µ0, λ, α, β)라 쓰지만 표 2 열은 (µ0, σ, λ, α)다. β 대신 σ가 있고 변환 규칙이 없다.
res["table2_columns_vs_text"] = {"text_NIG_params": ["mu0", "lambda", "alpha", "beta"],
                                 "table2_columns": ["mu0", "sigma", "lambda", "alpha"],
                                 "match": False}

# ---------- 9. 부록 E 충돌 확률 ----------
# 롤아웃 수가 보고되지 않아 신뢰구간을 만들 수 없다. 참고로 N에 따른 Wilson 하한만 계산한다.
def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(float(c - h), 3), round(float(c + h), 3)


res["appendixE_wilson_by_N"] = {
    str(n): {"bimodal_40pct_k": round(0.4 * n), "bimodal_ci95": wilson(round(0.4 * n), n),
             "gaussian_lt1pct_k0_ci95": wilson(0, n)}
    for n in (20, 50, 100, 500)}
# 그림 10 역산 M(t)의 멱지수 (t = 5 대 315)
m_lo5, m_hi5 = fig10[1]["M_common"]
m_lo315, m_hi315 = fig10[-1]["M_common"]
res["fig10_M_power_exponent_t5_to_315"] = round(float(np.log((m_lo315 + m_hi315) / (m_lo5 + m_hi5))
                                                 / np.log(315 / 5)), 3)
res["appendixE_note"] = "롤아웃 수 N 미보고. 'more than 40%'와 'fewer than 1%'는 N에 따라 달라지는 근사 주장."

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
