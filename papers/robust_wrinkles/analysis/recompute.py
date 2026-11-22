"""CORD-SLS 논문(arXiv:2606.14188 v1) 표 수치 재계산과 일관성 검사.

입력은 논문 표 1(본문), 표 2와 표 3(부록 F)을 옮긴 값이고, 코드에서 읽은 설정(horizon, 시간 간격, 샘플 수)은
repo/ 정적 읽기로 얻은 상수다(커밋 77e82b4). 논문은 시도 횟수와 ±의 정의를 적지 않으므로 검정은 n 가정별 민감도로 보인다.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초) → analysis/recompute_results.json
"""
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
R = {}

# ---------------------------------------------------------------- 표 1 (본문 8쪽)
# (동역학 평가 횟수, 단계당 시간 s, SD, 안전율 %, SD, 목표 오차 m, SD)
T1 = {
    "lift_rope": {
        "Ours": (5400, 0.050, 0.001, 100.0, 0.0, 0.0548, 0.0003),
        "MPPI": (1024000, 0.177, 0.043, 100.0, 0.0, 0.0681, 0.0038),
        "PPO": (16000000, 0.0004, 0.0001, 96.5, 0.2, 0.0587, 0.0016),
        "APG": (16000000, 0.0004, 0.0001, 100.0, 0.0, 0.0573, 0.0017),
    },
    "drag_rope": {
        "Ours": (13500, 0.070, 0.012, 99.9, 0.1, 0.0360, 0.0077),
        "MPPI": (2560000, 0.234, 0.018, 1.0, 0.1, 0.0814, 0.0034),
        "PPO": (40000000, 0.0004, 0.0001, 97.0, 0.4, 0.0320, 0.0092),
        "APG": (40000000, 0.0004, 0.0001, 93.0, 0.4, 0.0302, 0.0061),
    },
    "fold_cloth": {
        "Ours": (3400, 0.758, 0.107, 100.0, 0.0, 0.0611, 0.0072),
        "MPPI": (64000, 18.81, 6.11, 64.0, 6.1, 0.0738, 0.0079),
    },
    "flatten_cloth": {
        "Ours": (3400, 0.711, 0.077, 100.0, 0.0, 0.0234, 0.0089),
        "MPPI": (64000, 11.10, 4.66, 80.0, 0.2, 0.0526, 0.0102),
    },
}
# 코드에서 읽은 상수: horizon(단계 수), 시간 간격 dt, 상태 차원, MPC horizon N
HORIZON = {"lift_rope": 200, "drag_rope": 500, "fold_cloth": 200, "flatten_cloth": 200}
DT = 0.02
NODES = {"lift_rope": 21, "drag_rope": 21, "fold_cloth": 100, "flatten_cloth": 100}
GRIPPERS = {"lift_rope": 2, "drag_rope": 1, "fold_cloth": 2, "flatten_cloth": 2}
NS = [5, 10, 30, 100]


def welch(m1, s1, m2, s2, n):
    se = math.sqrt(s1 ** 2 / n + s2 ** 2 / n)
    if se == 0:
        return None
    t = (m1 - m2) / se
    df = (s1 ** 2 / n + s2 ** 2 / n) ** 2 / ((s1 ** 2 / n) ** 2 / (n - 1) + (s2 ** 2 / n) ** 2 / (n - 1))
    return {"t": round(t, 2), "p": float(2 * stats.t.sf(abs(t), df))}


# V1: 비율, 실시간 계수
v1 = {}
for task, rows in T1.items():
    ours = rows["Ours"]
    d = {"state_dim": NODES[task] * 3 + 3 * GRIPPERS[task], "steps": HORIZON[task]}
    for name, r in rows.items():
        d[name] = {
            "eval_ratio_vs_ours": round(r[0] / ours[0], 1),
            "time_ratio_vs_ours": round(r[1] / ours[1], 2),
            "realtime_factor": round(r[1] / DT, 2),     # 단계당 시간 / 시뮬 dt (1 초과면 실시간보다 느림)
            "evals_per_step_if_total_over_horizon": round(r[0] / HORIZON[task], 2),
        }
    v1[task] = d
R["V1_ratios_realtime"] = v1

# V2: 목표 오차와 안전율의 Ours 대 기준선 검정 (n 가정별)
v2 = {}
for task, rows in T1.items():
    o = rows["Ours"]
    for name, r in rows.items():
        if name == "Ours":
            continue
        key = f"{task}:Ours_vs_{name}"
        v2[key] = {
            "goal_err_diff(Ours-other)": round(o[5] - r[5], 4),
            "goal_err_welch": {str(n): welch(o[5], o[6], r[5], r[6], n) for n in NS},
            "safety_diff_pp(Ours-other)": round(o[3] - r[3], 1),
            "safety_welch": {str(n): welch(o[3], o[4], r[3], r[4], n) for n in NS},
        }
R["V2_ours_vs_baselines"] = v2

# V3: 안전율 값의 정수 가능성
def min_n_episode(v, step=0.1):
    """에피소드 단위 성공/실패 비율의 평균일 때 v(소수 1자리 반올림)를 낼 수 있는 최소 시도 수."""
    for n in range(1, 200001):
        k = round(v / 100 * n)
        if abs(100 * k / n - v) <= step / 2 + 1e-9:
            return n
    return None


def min_n_seeds_steps(v, steps, step=0.1, nmax=2000):
    """시드 n개 평균, 시드마다 steps 단계 중 위반 단계 수가 정수일 때 v를 낼 수 있는 최소 시드 수."""
    for n in range(1, nmax + 1):
        tot = steps * n
        k = round(v / 100 * tot)
        if abs(100 * k / tot - v) <= step / 2 + 1e-9:
            return n
    return None


v3 = {}
for task, rows in T1.items():
    for name, r in rows.items():
        v = r[3]
        v3[f"{task}:{name}"] = {
            "value": v,
            "min_episodes_if_episode_level": min_n_episode(v),
            "min_seeds_if_step_level": min_n_seeds_steps(v, HORIZON[task]),
        }
R["V3_integer_feasibility"] = v3

# V4: 동역학 평가 횟수 산술 (코드 상수와의 대조)
v4 = {}
# APG: 배치 8 x horizon x 에포크 10000 (benchmark/methods/APG.py train_APG 기본값)
for task in ("lift_rope", "drag_rope"):
    v4[f"APG_{task}_batch8_x_H_x_10000"] = 8 * HORIZON[task] * 10000
    v4[f"PPO_code_total_timesteps_{task}_H_x_10000"] = HORIZON[task] * 10000
# SLS: 횟수 / horizon = N + 7 ?
for task, N in (("lift_rope", 20), ("drag_rope", 20), ("fold_cloth", 10), ("flatten_cloth", 10)):
    cnt = T1[task]["Ours"][0]
    v4[f"SLS_{task}_evals_per_step"] = cnt / HORIZON[task]
    v4[f"SLS_{task}_N_plus_7"] = N + 7
# MPPI: 표 값을 샘플 수 x N x horizon으로 나눠 본 값
for task in ("lift_rope", "drag_rope", "fold_cloth", "flatten_cloth"):
    cnt = T1[task]["MPPI"][0]
    per_step = cnt / HORIZON[task]
    v4[f"MPPI_{task}_table_per_step"] = per_step
    v4[f"MPPI_{task}_code_samples1024_x_N_per_step"] = 1024 * (20 if NODES[task] == 21 else 10)
    v4[f"MPPI_{task}_implied_samples_x_N"] = per_step
v4["MPPI_lift_implied_N_if_1024_samples"] = 1024000 / 200 / 1024
v4["MPPI_cloth_implied_samples_if_N10"] = 64000 / 200 / 10
R["V4_eval_count_arithmetic"] = v4

# V5: 표 2 속도 향상 비 (부록 F)
T2 = {
    "rope": {"Ours": (1.095, 5.198), "DEFORM": (49.03, 2128.81), "DaXBench": (786.48, 11561.57)},
    "cloth": {"Ours": (15.08, 10.06), "DaXBench": (21.71, 110.19)},
}
v5 = {}
for obj, rows in T2.items():
    o = rows["Ours"]
    for name, r in rows.items():
        if name == "Ours":
            continue
        v5[f"{obj}:{name}"] = {"forward_x": round(r[0] / o[0], 1), "jacobian_x": round(r[1] / o[1], 1)}
    v5[f"{obj}:Ours_jac_over_fwd"] = round(o[1] / o[0], 2)
R["V5_table2_speedups"] = v5

# V6: 표 3 정확도 (부록 F, 검증 30%면 n=30)
T3 = {"Ours": (0.0150, 0.0040), "DEFORM": (0.0139, 0.0029), "Ours+res": (0.0075, 0.0068), "DEFORM+res": (0.0071, 0.0032)}
v6 = {
    "rel_gap_base_%": round(100 * (T3["Ours"][0] - T3["DEFORM"][0]) / T3["Ours"][0], 1),
    "rel_gap_res_%": round(100 * (T3["Ours+res"][0] - T3["DEFORM+res"][0]) / T3["Ours+res"][0], 1),
    "residual_gain_ours_%": round(100 * (1 - T3["Ours+res"][0] / T3["Ours"][0]), 1),
    "cv_ours_res": round(T3["Ours+res"][1] / T3["Ours+res"][0], 2),
    "welch_base": {str(n): welch(T3["Ours"][0], T3["Ours"][1], T3["DEFORM"][0], T3["DEFORM"][1], n) for n in (30, 100)},
    "welch_res": {str(n): welch(T3["Ours+res"][0], T3["Ours+res"][1], T3["DEFORM+res"][0], T3["DEFORM+res"][1], n) for n in (30, 100)},
}
R["V6_table3"] = v6

# V7: CP 교정과 표 3의 정합성. 87% 가 0.03 m 이하라면 13개(n=100)가 0.03 초과.
mean, cov, thr = 0.0150, 0.87, 0.03
top_mass = 1 - cov
m2 = (mean - top_mass * thr) / cov
min_sd = math.sqrt(top_mass * cov) * (thr - m2)
v7 = {
    "assumption": "표 3의 기본 시뮬 오차(평균 0.0150 m)와 CP 교정 잔차가 같은 100개 실로봇 표본이라고 가정(논문은 둘 다 100개라고만 적음)",
    "min_sd_given_mean_and_87pct_below_0.03": round(min_sd, 4),
    "reported_sd": 0.0040,
    "z_of_0.03_under_normal": round((0.03 - 0.0150) / 0.0040, 2),
    "p_exceed_0.03_under_normal_%": round(100 * float(stats.norm.sf((0.03 - 0.0150) / 0.0040)), 4),
}
# CP 지수: ceil((n+1)(1-d)) <= n 이려면 d >= 1/(n+1)
n_cal = 100
v7["min_per_step_miscoverage_for_finite_quantile_n100"] = round(1 / (n_cal + 1), 4)
vac = {}
for T in (5, 8, 10, 20, 50, 100, 200):
    vac[str(T)] = {
        "min_delta_for_finite_quantile": round(min(T / (n_cal + 1), 99), 3),
        "union_bound_delta_with_87pct_per_step": round(T * 0.13, 2),
        "min_n_for_delta_0.1": math.ceil(T / 0.1) - 1,
    }
v7["by_horizon_T"] = vac
R["V7_conformal"] = v7

# V8: 모서리 값. 노드·자유도 산술
v8 = {
    "cloth_sim_nodes": 10 * 10, "cloth_sim_dof_nodes_only": 300, "cloth_sim_state_dim_with_2_grippers": 306,
    "cloth_hw_nodes": 8 * 8, "cloth_hw_state_dim": 64 * 3 + 3 * 2,
    "rope_bench_nodes": 21, "drag_state_dim": 21 * 3 + 3, "lift_state_dim": 21 * 3 + 6,
    "rope_hw_nodes": 11, "rope_hw_state_dim_output_feedback": 11 * 3 + 2 * 3,
    "sls_disturbance_alpha_bench": 0.003 * DT,
    "cp_radius_hw_m": 0.03,
    "cp_radius_over_bench_alpha": 0.03 / (0.003 * DT),
}
R["V8_dimension_arithmetic"] = v8

# V9: 접촉 평활화 범위(코드 exp(-dist/kappa) 대 논문 exp(-|c|^2/kappa))
kappa = 1e-3
rows = {}
for d_mm in (1, 2, 5, 10, 17, 50):
    d = d_mm * 1e-3
    rows[str(d_mm)] = {"code_exp(-d/k)": float(np.exp(-d / kappa)), "paper_exp(-d^2/k)": float(np.exp(-d * d / kappa))}
R["V9_smoothing_weight_kappa_1e-3"] = rows
R["V9_attach_radius_mm"] = {"lift_rope": (0.002 + 0.015) * 1e3, "drag_rope": (0.002 + 0.005) * 1e3}

# V10: 3개 속도 지표 (Ours 대 RL)
R["V10_ours_vs_rl_speed"] = {
    "lift_rope_x": round(0.050 / 0.0004, 0),
    "drag_rope_x": round(0.070 / 0.0004, 0),
}

OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(R, indent=1, ensure_ascii=False))
