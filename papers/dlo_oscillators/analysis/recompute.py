"""논문 표·본문 수치를 재계산하고 일관성을 검사한다 (CPU, 수 초).

입력은 논문 원문(paper/paper.txt)에서 옮긴 값뿐이며, 원자료나 학습은 쓰지 않는다.
결과: analysis/recompute_results.json
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "recompute_results.json"
res = {}

# ---------- V1. 초록·본문 백분율 ----------
p_nodes = 710_234   # Table 4, NODE sim
p_lnn = 203_584     # Table 4, LNN sim
p_phi = 20_031      # Table 4, Φ-CON sim
res["V1_params_reduction"] = {
    "vs_NODE_pct": round(100 * (1 - p_phi / p_nodes), 2),   # 초록 97.2
    "vs_LNN_pct": round(100 * (1 - p_phi / p_lnn), 2),      # 초록 90.2
}

# Table 2 (sim, PD -> PD+FF(Φ-CON)), 평균 오차 mm
t2 = {"Baseline": (238.6, 1.5), "Heavy": (459.4, 2.7), "Low damping": (238.9, 5.9), "Compliant": (240.5, 0.8)}
res["V2_sim_reduction_pct"] = {k: round(100 * (1 - ff / pd), 2) for k, (pd, ff) in t2.items()}
# 초록은 "97.5–99.7%" (Φ-CON 기준)
vals = list(res["V2_sim_reduction_pct"].values())
res["V2_sim_reduction_range"] = [round(min(vals), 2), round(max(vals), 2)]

# 실험 평균 감소율: 초록·결론 27.9% = 두 끝점의 per-target 평균
rel_left, rel_right = 23.5, 32.2
res["V3_real_mean_of_endpoints_pct"] = round((rel_left + rel_right) / 2, 2)   # 27.9 기대
res["V3_real_aggregate_reduction_pct"] = {
    "left": round(100 * (1 - 48.4 / 63.7), 2),
    "right": round(100 * (1 - 32.3 / 48.3), 2),
}

# ---------- V4. 학습 속도 ----------
res["V4_epoch_speed_ratio_LNN_over_PhiCON"] = round(9.35 / 1.06, 2)   # 본문 8.8x 기대

# ---------- V5. Table 1 Welch 검정 (5 시드, 평균 ± SD) ----------
t1 = {  # (NODE mean, sd), (LNN), (Φ-CON)
    "Baseline": ((14.6, 0.6), (17.6, 1.3), (14.0, 0.6)),
    "Heavy": ((12.5, 1.1), (13.6, 0.7), (6.6, 0.9)),
    "Low damping": ((15.4, 1.6), (24.8, 1.6), (14.8, 1.0)),
    "Compliant": ((11.0, 1.3), (12.9, 0.6), (4.5, 0.5)),
}
n = 5
def welch(m1, s1, m2, s2):
    v1, v2 = s1**2 / n, s2**2 / n
    t = (m1 - m2) / math.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1**2 / (n - 1) + v2**2 / (n - 1))
    p = 2 * stats.t.sf(abs(t), df)
    return round(t, 2), round(df, 2), round(p, 4)
res["V5_table1_welch_phicon_vs"] = {}
for k, (nd, ln, ph) in t1.items():
    res["V5_table1_welch_phicon_vs"][k] = {
        "vs_NODE": dict(zip(["t", "df", "p"], welch(ph[0], ph[1], nd[0], nd[1]))),
        "vs_LNN": dict(zip(["t", "df", "p"], welch(ph[0], ph[1], ln[0], ln[1]))),
    }

# ---------- V6. 실험 per-target 개선율 신뢰구간 (n=76 steady-state 목표, 정규 근사) ----------
def ci(mean, sd, n_):
    se = sd / math.sqrt(n_)
    h = stats.t.ppf(0.975, n_ - 1) * se
    return round(mean - h, 2), round(mean + h, 2)
res["V6_real_per_target_ci95_n76"] = {
    "left_23.5_pm_8.8": ci(23.5, 8.8, 76),
    "right_32.2_pm_10.5": ci(32.2, 10.5, 76),
}
res["V6_real_per_target_ci95_n80"] = {
    "left": ci(23.5, 8.8, 80), "right": ci(32.2, 10.5, 80),
}
# 'n' 불명확: 76 또는 80 (4개 미지 목표 포함 여부). 결론 불변 여부만 확인

# ---------- V7. 실험 평균 오차 차이의 대략적 검정 (대응 표본 불가, 독립 가정 상한) ----------
for side, (pd, pdsd, ff, ffsd) in {"left": (63.7, 8.0, 48.4, 5.6), "right": (48.3, 8.5, 32.3, 5.2)}.items():
    n_ = 76
    t = (pd - ff) / math.sqrt(pdsd**2 / n_ + ffsd**2 / n_)
    res.setdefault("V7_real_mean_error_welch_independent", {})[side] = round(t, 2)

# ---------- V8. 시간 측정: Table 2 대 Table 8 (같은 측정이라고 본문이 적음) ----------
t2_time = {"PhiCON_CPU": (12.6, 3.4), "PhiCON_GPU": (72.1, 8.1), "LNN_CPU": (25.5, 2.9), "LNN_GPU": (114.6, 4.1)}
t8_time = {"PhiCON_CPU": (13.9, 3.0), "PhiCON_GPU": (64.8, 15.8), "LNN_CPU": (25.4, 3.9), "LNN_GPU": (105.2, 5.0)}
res["V8_timing_table2_vs_table8_pct_diff"] = {
    k: round(100 * (t2_time[k][0] - t8_time[k][0]) / t8_time[k][0], 1) for k in t2_time}

# ---------- V9. 파라미터 수 재계산 (Table 5 구조 + 대칭 행렬 자유도) ----------
def mlp(dims):
    return sum(a * b + b for a, b in zip(dims[:-1], dims[1:]))
def mlp_params(d_in, hidden, d_out):
    return mlp([d_in] + list(hidden) + [d_out])
def sym(k):
    return k * (k + 1) // 2

# NODE: 입력 (x, ẋ, τx) = 3·3N, 출력 3N
res["V9_params"] = {}
for name, N, node_h, lnn_m, lnn_u, phi_tab, fb_tab, node_tab, lnn_tab in [
    ("sim", 30, [512] * 3, (32,) * 5, (128,) * 4, 20031, 12375, 710234, 203584),
    ("real", 12, [512] * 2, (128,) * 6, (64,) * 7, 3156, 2034, 336932, 200603),
]:
    d3n = 3 * N
    node = mlp_params(3 * d3n, node_h, d3n)
    lnn = mlp_params(d3n, lnn_m, sym(d3n)) + mlp_params(d3n, lnn_u, 1)
    n_shape = d3n - 3                      # 형상 좌표 차원 (3N-3)
    # FBCON: A=I, c=0 (identity), Kr 대칭, br, Mw·Dw 대칭
    fb = n_shape + sym(n_shape) + 2 * sym(d3n)
    # Φ-CON: Ar = LrUr (자유도 n²), cr, br, Kr 대칭, Mw·Dw 대칭
    phi = n_shape**2 + n_shape + n_shape + sym(n_shape) + 2 * sym(d3n)
    res["V9_params"][name] = {
        "NODE_calc": node, "NODE_table": node_tab,
        "LNN_calc": lnn, "LNN_table": lnn_tab,
        "FBCON_calc": fb, "FBCON_table": fb_tab, "FBCON_diff": fb_tab - fb,
        "PhiCON_calc": phi, "PhiCON_table": phi_tab, "PhiCON_diff": phi_tab - phi,
        "diff_equals_9N": (fb_tab - fb) == 9 * N and (phi_tab - phi) == 9 * N,
    }

# ---------- V10. 물리량 대략 계산 (Table 3, A.5) ----------
def mass(rho, r, L):
    return rho * math.pi * r**2 * L
g = 9.81
m_base = mass(3000, 0.005, 1.0)
m_heavy = mass(7800, 0.005, 1.0)
m_cable = mass(3500, 0.009, 0.76)
res["V10_mass_kg"] = {"sim_baseline": round(m_base, 4), "sim_heavy": round(m_heavy, 4), "real_cable": round(m_cable, 4)}
res["V10_weight_N"] = {"sim_baseline": round(m_base * g, 2), "sim_heavy": round(m_heavy * g, 2), "real_cable": round(m_cable * g, 2)}
# PD 단독 잔차 힘 = Kp × 최종 오차 (정상상태에서 PD만 있으면 이 힘이 필요)
res["V10_pd_only_residual_force_N"] = {
    "sim_baseline_Kp10": round(10 * 0.2386, 2),
    "sim_heavy_Kp10": round(10 * 0.4594, 2),
    "real_left_Kp200": round(200 * 0.0637, 2),
    "real_right_Kp200": round(200 * 0.0483, 2),
}
# 실험 임피던스 게인: ζ = 0.77, Kd = 21.8 → 암묵적 질량
zeta, kd, kp = 0.77, 21.8, 200.0
res["V10_implied_mass_kg"] = round((kd / (2 * zeta)) ** 2 / kp, 3)

# ---------- V11. 감쇠비 일정 조건 (Table 9) ----------
res["V11_table9_kd_over_sqrtkp"] = [round(1.0 / math.sqrt(10), 3), round(2.24 / math.sqrt(50), 3), round(4.47 / math.sqrt(200), 3)]
res["V11_table9_reduction_pct"] = {
    str(kp_): round(100 * (1 - ff / pd), 2)
    for kp_, pd, ff in [(10, 238.6, 1.5), (50, 48.2, 0.44), (200, 12.1, 0.15)]
}
res["V11_pd_error_times_kp_N"] = [round(238.6e-3 * 10, 3), round(48.2e-3 * 50, 3), round(12.1e-3 * 200, 3)]  # 상수면 정적 힘

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
