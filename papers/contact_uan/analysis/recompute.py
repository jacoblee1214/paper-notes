#!/usr/bin/env python3
"""논문 보고 수치의 재계산과 일관성 검사 (V1-V22). 입력: figure_values.json (digitize_figures.py의 출력).

그림에서 읽은 값은 "그림 판독 값"이며 눈금 간격의 1% 안쪽 오차를 갖는다.
논문에 없는 시행 수는 문장마다 가정이라고 적는다.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초) → recompute_results.json
"""
import json
import math
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
FV = json.load(open(os.path.join(HERE, "figure_values.json")))
R = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# V1 그림 6C: 토크 사용량 상대 오차 감소율 (본문 69.2%)
f6 = FV["fig6C_torque_usage_rel_error"]
R["V1_fig6C"] = {
    "no_uan": f6["no_uan"], "uan": f6["uan"],
    "reduction_pct_from_bars": 100 * (1 - f6["uan"] / f6["no_uan"]),
    "paper_pct": 69.2,
}

# V2 그림 5C와 부록 G.1: 24-step 관절 위치 RMSE 감소율
R["V2_replay_rmse"] = {
    "estimated_init": {"without": 0.0750, "with": 0.0239, "reduction_pct": 100 * (1 - 0.0239 / 0.0750)},
    "ground_truth_init": {"without": 0.0720, "with": 0.0184, "reduction_pct": 100 * (1 - 0.0184 / 0.0720)},
    "separately_trained_G1": {"direct": 0.0318, "estimated": 0.0239, "full_oracle": 0.0120},
    "digitized_fig5C": FV["fig5C_q_rmse_rad_estimated"],
    "ratio_with_residual_gt_over_est": 0.0184 / 0.0239,
    "estimated_vs_direct_reduction_pct": 100 * (1 - 0.0239 / 0.0318),
    "oracle_vs_estimated_reduction_pct": 100 * (1 - 0.0120 / 0.0239),
}

# V3 그림 5A: 접촉 높이 RMSE 비
a = FV["fig5A_contact_height_rmse_mm"]
R["V3_fig5A"] = {"direct_mm": a["direct"], "stance_mm": a["stance"], "ratio": a["direct"] / a["stance"]}

# V4 그림 5B: MSE → RMSE, 감소율 (WTW 학습 데이터)
b = FV["fig5B_hw_q_mse_1e-3_rad2"]
R["V4_fig5B"] = {
    "mse_1e-3": b,
    "rmse_rad": {k: math.sqrt(v * 1e-3) for k, v in b.items()},
    "stance_vs_direct_mse_reduction_pct": 100 * (1 - b["stance"] / b["direct"]),
    "uan_vs_stance_mse_reduction_pct": 100 * (1 - b["contact_uan"] / b["stance"]),
    "uan_vs_direct_mse_reduction_pct": 100 * (1 - b["contact_uan"] / b["direct"]),
}

# V5 그림 7C: "42.8 ± 12.1%" 재현 (4개 명령 힘 수준의 상대 감소율 평균 ± 표준오차)
c = FV["fig7C_measured_force_N"]
cmd = np.array(c["command"], float)
eb = np.abs(np.array(c["baseline"]) - cmd)
eu = np.abs(np.array(c["uan"]) - cmd)
red = 100 * (1 - eu / eb)
R["V5_fig7C"] = {
    "command_N": c["command"],
    "baseline_abs_err_N": eb.tolist(), "uan_abs_err_N": eu.tolist(),
    "reduction_pct_per_level": red.tolist(),
    "mean_of_levels": float(red.mean()),
    "sd_of_levels": float(red.std(ddof=1)),
    "sem_of_levels": float(red.std(ddof=1) / math.sqrt(len(red))),
    "paper": "42.8 ± 12.1",
    "alt_total_abs_error_ratio_reduction_pct": float(100 * (1 - eu.sum() / eb.sum())),
    "alt_mean_without_20N_pct": float(red[:3].mean()),
    "alt_mean_abs_error_reduction_pct": float(100 * (1 - eu.mean() / eb.mean())),
    "uan_underdelivery_at_20N_pct": float(100 * (cmd[3] - c["uan"][3]) / cmd[3]),
    "baseline_underdelivery_at_20N_pct": float(100 * (cmd[3] - c["baseline"][3]) / cmd[3]),
    "baseline_underdelivery_pct_per_level": (100 * (cmd - np.array(c["baseline"])) / cmd).tolist(),
    "uan_underdelivery_pct_per_level": (100 * (cmd - np.array(c["uan"])) / cmd).tolist(),
}

# V6 그림 7B: 토크 감소율. 세 막대의 단순 평균과 칸별 t 검정 (시행 수 5 가정, 오차 막대의 정의는 논문에 없음)
tb = FV["fig7B_torque_reduction_pct"]
means = [tb[k]["mean"] for k in ("1.0", "2.0", "2.5")]
hw = [tb[k]["halfwidth"] for k in ("1.0", "2.0", "2.5")]
n = 5
tt = {}
for k, m, h in zip(("1.0", "2.0", "2.5"), means, hw):
    sem_case_t = m / h  # 오차 막대가 SEM이라면 t = 평균/SEM
    sd_case_t = m / (h / math.sqrt(n))  # 오차 막대가 SD라면
    tt[k] = {
        "mean": m, "halfwidth": h,
        "p_if_errorbar_is_SEM": float(2 * stats.t.sf(abs(sem_case_t), n - 1)),
        "p_if_errorbar_is_SD": float(2 * stats.t.sf(abs(sd_case_t), n - 1)),
    }
R["V6_fig7B"] = {
    "bars": tt,
    "simple_mean_of_three_bars": float(np.mean(means)),
    "paper_claim_pct": 20,
    "mean_of_first_two": float(np.mean(means[:2])),
    "n_trials_assumed": n,
}

# V7 그림 7A: 속도 추종과 속도 교란
a7 = FV["fig7A_achieved_speed_mps"]
cm = np.array(a7["command"][:3])
bs = np.array(a7["baseline"][:3])
us = np.array(a7["uan"][:3])
R["V7_fig7A"] = {
    "command": cm.tolist(), "baseline_speed": bs.tolist(), "uan_speed": us.tolist(),
    "baseline_abs_err": np.abs(bs - cm).tolist(), "uan_abs_err": np.abs(us - cm).tolist(),
    "baseline_mean_abs_err": float(np.abs(bs - cm).mean()),
    "uan_mean_abs_err": float(np.abs(us - cm).mean()),
    "uan_speed_ratio_to_baseline": (us / bs).tolist(),
    "uan_3.0_speed": a7["uan"][3], "uan_3.0_over_command_pct": float(100 * (a7["uan"][3] / 3.0 - 1)),
    "uan_3.0_errorbar_halfwidth": a7["uan_3.0_errorbar_halfwidth"],
    "course_30ft_m": 30 * 0.3048,
    "traversal_time_s_at_speeds": {str(s): float(30 * 0.3048 / s) for s in (1.0, 2.0, 2.5, 3.0, 3.13)},
}

# V8 그림 8A: 20 s 생존율. 시행 수는 논문에 없다. 4096 환경 1회(n = 4096 가정)와 n = 500 두 경우의 Wilson 구간
s8 = FV["fig8A_survival_pct"]
V8 = {"values_pct": s8, "paper_text": {"direct": 49.8, "estimated": 96.7, "oracle_range": [96.6, 98.4]}}
for nn in (4096, 500):
    ci = {}
    for k, v in s8.items():
        k_succ = round(v / 100 * nn)
        lo, hi = wilson(k_succ, nn)
        ci[k] = [100 * lo, 100 * hi]
    V8[f"wilson_n{nn}"] = ci
oracles = ["base_oracle", "contact_oracle", "pose_oracle", "all_oracle"]
V8["oracle_minus_estimated_pp"] = {k: s8[k] - s8["estimated"] for k in oracles}
V8["estimated_minus_direct_pp"] = s8["estimated"] - s8["direct"]
# 그림 13의 Clean(재학습 대조, 같은 구성이라면)과 그림 8A의 Est.
f13 = FV["fig13_survival_pct"]
V8["fig13_clean_minus_fig8A_estimated_pp"] = f13["clean"] - s8["estimated"]
V8["fig13_values_pct"] = f13
R["V8_fig8A_fig13"] = V8

# V9 그림 8B, 8C: 기준 정책 대 UAN 정책 대 전문가
xb, tq = FV["fig8B_xy_rmse_mps"], FV["fig8C_torque_sq"]
V9 = {}
for tg in ("easy", "medium", "hard"):
    V9[tg] = {
        "xy_baseline_over_expert": xb["baseline"][tg] / xb["expert"][tg],
        "xy_wtw_uan_over_expert": xb["wtw_uan"][tg] / xb["expert"][tg],
        "xy_wtwvel_uan_over_expert": xb["wtw_vel_uan"][tg] / xb["expert"][tg],
        "torque_baseline_over_expert": tq["baseline"][tg] / tq["expert"][tg],
        "torque_wtw_uan_over_expert": tq["wtw_uan"][tg] / tq["expert"][tg],
        "torque_wtwvel_uan_over_expert": tq["wtw_vel_uan"][tg] / tq["expert"][tg],
        "xy_wtw_uan_vs_wtwvel_diff": xb["wtw_uan"][tg] - xb["wtw_vel_uan"][tg],
        "torque_wtw_uan_vs_wtwvel_diff_pct": 100 * (tq["wtw_vel_uan"][tg] / tq["wtw_uan"][tg] - 1),
    }
R["V9_fig8BC"] = V9

# V10 그림 11 / 부록 F: 데이터량과 정규화 에피소드 길이
f11 = FV["fig11_target_env_norm_len_pct"]
sd = FV["fig11_target_env_sd_pct"]
hours = 6e6 / 50 / 3600
n_ep = 64  # 32 환경 x 최대 2 에피소드 (부록 F "Evaluation protocol"), 에피소드는 독립이 아닐 수 있음
welch = stats.ttest_ind_from_stats(f11["wm_ft_33h"], sd["wm_ft_33h"], n_ep, f11["base"], sd["base"], n_ep, equal_var=False)
R["V10_fig11"] = {
    "transitions": 6e6, "hours_at_50Hz": hours, "ratio_vs_5min": hours * 60 / 5,
    "target_env_pct": f11, "target_env_sd_pct_from_errorbars": sd,
    "wm_ft_minus_base_pp": f11["wm_ft_33h"] - f11["base"],
    "welch_t": float(welch.statistic), "welch_p": float(welch.pvalue), "n_episodes_assumed_each": n_ep,
    "wm_scratch_minus_base_pp": f11["wm_33h"] - f11["base"],
    "text": {"wm_scratch": 6.8, "wm_ft": 38.2, "baseline": 32.6},
    "uan_5min_vs_wm_data_ratio": hours * 60 / 5,
}

# V11 표 15: 행렬 비교
T15 = {
    "position_residual": dict(q=0.00349, survival=97.34, xy=0.391, yaw=0.185, power=69.29),
    "torque_residual": dict(q=0.00571, survival=98.17, xy=0.387, yaw=0.214, power=69.25),
    "supervised_torque": dict(q=0.00565, survival=97.56, xy=0.394, yaw=0.215, power=71.90),
}
pr, tr, st = T15["position_residual"], T15["torque_residual"], T15["supervised_torque"]
R["V11_table15"] = {
    "q_position_vs_torque_residual_pct": 100 * (pr["q"] / tr["q"] - 1),
    "q_position_vs_supervised_pct": 100 * (pr["q"] / st["q"] - 1),
    "survival_range_pp": max(v["survival"] for v in T15.values()) - min(v["survival"] for v in T15.values()),
    "xy_range": max(v["xy"] for v in T15.values()) - min(v["xy"] for v in T15.values()),
    "yaw_position_vs_torque_residual_pct": 100 * (pr["yaw"] / tr["yaw"] - 1),
    "power_supervised_vs_position_pct": 100 * (st["power"] / pr["power"] - 1),
    "table15_q_over_fig5C_gt_with_residual": pr["q"] / 0.0184,
    "note": "표 15 Replay q 0.0035 rad 대 그림 5C 0.0184 rad(정답 초기화, 잔차 켬): 5.3배. 두 값의 프로토콜 차이는 본문에 없음",
}

# V12 표 12: 섭동 벡터의 1:2:3 배율
R["V12_table12"] = {
    "damping_offset": [2.0, 4.0, 6.0], "frictionloss": [0.5, 1.0, 1.5], "delay_steps": [1, 2, 3],
    "encoder_bias": [0.003, 0.006, 0.009], "armature_mult": [1.5, 2.0, 2.5],
    "ratios_to_easy": {"damping": [1, 2, 3], "frictionloss": [1, 2, 3], "delay": [1, 2, 3],
                       "encoder_bias": [1, 2, 3], "armature_offset_from_1": [0.5 / 0.5, 1.0 / 0.5, 1.5 / 0.5]},
    "consistent": True,
    "delay_ms_if_policy_steps_20ms": [20, 40, 60], "delay_ms_if_physics_steps_5ms": [5, 10, 15],
}

# V13 데이터량과 학습 노출
rows_65 = 6.5 * 60 * 50
rows_5 = 5 * 60 * 50
steps_per_iter = 4096 * 24
params_group = (6 * 128 + 128) + (128 * 128 + 128) + (128 * 1 + 1)
R["V13_data_scale"] = {
    "rows_6.5min_at_50Hz": rows_65, "rows_5min": rows_5, "rows_10min_wtw_plus_vel": 2 * rows_5,
    "nonoverlapping_24step_windows_6.5min": rows_65 / 24,
    "env_steps_per_ppo_iteration": steps_per_iter,
    "uan_params_per_group": params_group, "uan_params_total_5groups": 5 * params_group,
    "iterations_in_docs": 2000,
    "env_steps_at_2000_iter": 2000 * steps_per_iter,
    "visits_per_row_at_2000_iter": 2000 * steps_per_iter / rows_65,
    "visits_per_row_at_2000_iter_if_5min": 2000 * steps_per_iter / rows_5,
    "scalar_targets_6.5min_29joints": rows_65 * 29,
    "horizon_ms": 24 * 20, "history_ms": 3 * 20,
    "joints_per_group": {"7520_14": 5, "7520_22": 4, "5020_2x": 6, "5020": 10, "4010": 4},
    "joints_total": 5 + 4 + 6 + 10 + 4,
    "obs_dim_jumping": 58 + 1 + 3 + 6 + 3 + 3 + 29 + 29 + 29,
    "estimator_input_dim": (3 + 3 + 29 + 29 + 29) * 3,
}

# V14 식 (7): 토크-속도 포락선 경계 점검
def env_bounds(r, tau=1.0):
    lo = -tau * (1 + max(min(r, 0), -1))
    hi = tau * (1 - max(min(r, 1), 0))
    return lo, hi

R["V14_eq7_bounds"] = {str(r): env_bounds(r) for r in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)}

# V15 소프트 접촉 감도의 산술 (그림 3): 발 길이 22 cm 가정 기울기와 높이 환산
R["V15_tilt_arithmetic"] = {
    "half_foot_length_m_assumed": 0.11,
    "edge_height_change_mm_per_deg": 0.11 * math.tan(math.radians(1.0)) * 1000,
    "edge_height_change_mm_at_2deg": 0.11 * math.tan(math.radians(2.0)) * 1000,
    "paper_fig3_height_read_pct": {"-2": 16, "-1": 10, "0": 0, "1": -28, "2": -50},
    "paper_fig3_moment_read_pct": {"-2": -46, "-1": -27, "0": 0, "1": 2, "2": 3},
}

# V16 수치 일치: 초록·결론의 숫자
R["V16_headline_numbers"] = {
    "abstract_20pct_torque": "그림 7B 세 막대 평균 %.1f%%, 앞 두 막대 평균 %.1f%%" % (np.mean(means), np.mean(means[:2])),
    "abstract_43pct_force": "그림 7C 판독으로 평균 %.1f%% 재현" % red.mean(),
    "abstract_6.5min": "6.5분 x 60 x 50 Hz = %d 행" % rows_65,
}

with open(os.path.join(HERE, "recompute_results.json"), "w") as f:
    json.dump(R, f, indent=2, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))
