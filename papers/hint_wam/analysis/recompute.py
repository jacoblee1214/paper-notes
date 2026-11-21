"""HINT-WAM (arXiv:2611.24789 v1) 논문 표 수치를 다시 계산하고 통계 검정을 한다.

입력: 논문 본문, 표, 보충자료(Supp.)의 값을 아래에 그대로 옮겨 적었다(출처는 주석).
      schedule 값은 공식 코드 src/fastwam/models/wan22/schedulers/scheduler_continuous.py 식으로 계산한다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 표 6의 SE는 과제 90개의 쌍체 차이의 표준오차로 읽는다. 과제 간 독립을 가정한다.
- 학습 시드는 하나(seed 42)다. 아래 검정은 과제 간 변동만 반영하고 학습 시드 변동은 반영하지 못한다.
- 평가 잡음 하한은 과제당 50시도의 이항 표준편차로 계산한다(에피소드가 독립이라고 가정).
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}


def sign_test_two_sided(wins, losses):
    n = wins + losses
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p)


# ---------- 1. 표 1, 표 2: 평균과 차이 재계산 ----------
# 출처: 표 1 (LIBERO-90 열, RoboTwin clean/random 열)
table1 = {
    "Fast-WAM": (15.56, 86.48, 83.44),
    "SFT": (78.88, None, None),
    "Full SFT": (81.24, 84.12, 83.24),
    "BC control": (71.56, 85.36, 82.08),
    "Self-imitation": (79.62, None, None),
    "Shuffled hindsight": (80.28, None, None),
    "HINT-WAM": (85.18, 87.04, 83.68),
}
d_sft = round(table1["HINT-WAM"][0] - table1["SFT"][0], 2)
d_full = round(table1["HINT-WAM"][0] - table1["Full SFT"][0], 2)
d_bc_rt = round(table1["HINT-WAM"][1] - table1["BC control"][1], 2), round(table1["HINT-WAM"][2] - table1["BC control"][2], 2)
d_full_rt = round(table1["Full SFT"][1] - table1["Fast-WAM"][1], 2), round(table1["Full SFT"][2] - table1["Fast-WAM"][2], 2)
RES["table1_differences"] = {
    "HINT_minus_SFT": d_sft,           # 본문 6.30
    "HINT_minus_FullSFT": d_full,      # 본문 3.94
    "HINT_minus_BC_robotwin_clean_random": d_bc_rt,  # 본문 1.68, 1.60
    "FullSFT_minus_released_robotwin_clean_random": d_full_rt,  # 보충 -2.36 ... 본문 -> 아래 확인
    "SFT_minus_FastWAM_LIBERO90": round(78.88 - 15.56, 2),
}

# 표 2 (원 스위트 4개 평균). 출처: 표 2
table2 = {
    "Fast-WAM": [96.8, 99.6, 95.8, 94.0, 96.55],
    "no_replay_SFT": [58.6, 33.2, 53.4, 4.6, 37.45],
    "no_replay_FullSFT": [54.4, 30.2, 44.6, 2.2, 32.85],
    "no_replay_BC": [55.4, 25.6, 47.0, 4.0, 33.00],
    "no_replay_SelfImit": [64.4, 39.8, 54.8, 2.4, 40.35],
    "no_replay_Shuffled": [66.2, 36.0, 60.2, 2.4, 41.20],
    "no_replay_HINT": [70.8, 38.4, 58.8, 8.2, 44.05],
    "replay_SFT": [97.2, 95.8, 92.0, 81.2, 91.55],
    "replay_FullSFT": [97.8, 96.0, 94.4, 83.4, 92.90],
    "replay_HINT": [98.4, 96.0, 95.8, 90.2, 95.10],
}
t2 = {}
for k, v in table2.items():
    avg = round(float(np.mean(v[:4])), 2)
    t2[k] = {"avg_recomputed": avg, "avg_reported": v[4], "match": abs(avg - v[4]) < 0.006}
RES["table2_average_check"] = t2
RES["table2_checks"] = {
    "replay_HINT_minus_SFT_newtasks": round(81.56 - 76.70, 2),   # 본문 4.86
    "replay_HINT_minus_SFT_orig": round(t2["replay_HINT"]["avg_recomputed"] - t2["replay_SFT"]["avg_recomputed"], 2),  # 3.55
    "replay_HINT_gap_to_released": round(96.55 - t2["replay_HINT"]["avg_recomputed"], 2),  # within 1.5
    "replay_HINT_minus_FullSFT_orig": round(t2["replay_HINT"]["avg_recomputed"] - t2["replay_FullSFT"]["avg_recomputed"], 2),  # 2.20
    "noreplay_HINT_minus_SFT_orig": round(t2["no_replay_HINT"]["avg_recomputed"] - t2["no_replay_SFT"]["avg_recomputed"], 2),
}

# ---------- 2. 표 6 (LIBERO-90 쌍체 검정) ----------
# 출처: 보충 표 6 (mean, SE, t, wins, losses, ties, sign-test p)
table6 = [
    ("HINT-WAM vs SFT", 6.30, 1.76, 3.58, 36, 22, 32, 0.087),
    ("HINT-WAM vs self-imitation", 5.56, 1.88, 2.96, 47, 23, 20, 0.0056),
    ("HINT-WAM vs shuffled hindsight", 4.90, 1.78, 2.76, 45, 25, 20, 0.022),
    ("HINT-WAM vs velocity matching", 6.16, 1.59, 3.62, 48, 17, 25, 0.00042),
    ("velocity matching vs SFT", 0.14, 1.52, 0.09, 30, 37, 23, 0.46),
]
df = 90 - 1
tcrit = stats.t.ppf(0.975, df)
rows = []
for name, mean, se, t_rep, w, l, tie, p_rep in table6:
    assert w + l + tie == 90, name
    t_calc = mean / se
    p_t = 2 * stats.t.sf(abs(t_rep), df)
    rows.append({
        "comparison": name,
        "mean": mean,
        "SE": se,
        "t_reported": t_rep,
        "t_from_mean_over_SE": round(t_calc, 3),
        "t_consistent_within_0.05": abs(t_calc - t_rep) < 0.05,
        "95CI": [round(mean - tcrit * se, 2), round(mean + tcrit * se, 2)],
        "p_t_two_sided_from_reported_t": float(f"{p_t:.4g}"),
        "sign_test_p_recomputed": float(f"{sign_test_two_sided(w, l):.4g}"),
        "sign_test_p_reported": p_rep,
        "wins_losses_ties": [w, l, tie],
    })
RES["table6_paired"] = rows
# 학습 시드 없이 과제 간 SD 역산 (SE * sqrt(90))
RES["table6_HINT_vs_SFT_task_level_SD_of_difference"] = round(1.76 * math.sqrt(90), 2)
# 평가 샘플링 잡음 하한: p=0.8, n=50, 과제별 차이 SD = sqrt(2) * sqrt(p(1-p)/n)
p_ = 0.80
sd_task_noise = math.sqrt(2) * math.sqrt(p_ * (1 - p_) / 50) * 100
RES["eval_noise_floor"] = {
    "per_task_SD_of_difference_pts": round(sd_task_noise, 2),
    "SE_of_mean_over_90_tasks_pts": round(sd_task_noise / math.sqrt(90), 2),
    "note": "독립 에피소드 가정. 관측 SE 1.76보다 작아, 관측 분산의 대부분은 과제 간 이질성 또는 학습 시드 변동이다.",
}

# 최소 검출 차이(검정력 80%, 유의수준 0.05, 쌍체 SE 1.76 기준)
mde = (tcrit + stats.t.ppf(0.80, df)) * 1.76
RES["MDE_80pct_power_pts_HINT_vs_SFT_design"] = round(mde, 2)

# ---------- 3. 표 8 (RoboTwin 25과제 쌍체 검정) ----------
table8 = [
    ("HINT vs BC, clean", 1.68, 0.84, 2.00, (12, 9, 4)),
    ("HINT vs BC, randomized", 1.60, 1.32, 1.21, (14, 9, 2)),
    ("HINT vs released, clean", 0.56, 1.19, 0.47, (11, 9, 5)),
    ("HINT vs released, randomized", 0.24, 1.04, 0.23, (13, 9, 3)),
    ("BC vs released, clean", -1.12, 1.08, -1.04, (9, 11, 5)),
    ("BC vs released, randomized", -1.36, 1.43, -0.95, (8, 15, 2)),
]
rob = []
for name, mean, se, t_rep, (w, l, tie) in table8:
    assert w + l + tie == 25, name
    rob.append({
        "comparison": name,
        "mean": mean,
        "SE": se,
        "t_from_mean_over_SE": round(mean / se, 3),
        "t_reported": t_rep,
        "p_t_two_sided_df24": round(float(2 * stats.t.sf(abs(t_rep), 24)), 4),
        "sign_test_p": round(sign_test_two_sided(w, l), 4),
    })
RES["table8_robotwin_paired"] = rob

# ---------- 4. 과제별 이득 구간 (보충 E) ----------
bands = {"SFT<50": (13, 32.2), "50-80": (15, 11.5), "95+": (42, -1.7)}
n_total = 90
mean_all = 6.30
implied_80_95 = (mean_all * n_total - sum(n * g for n, g in bands.values())) / 20
RES["per_task_band_consistency"] = {
    "implied_mean_gain_in_80_95_band_pts (n=20, not reported)": round(implied_80_95, 2),
    "note": "구간 평균과 전체 평균 6.30으로 역산한 값. 보고되지 않은 구간이라 대략적 일관성 확인만 가능.",
}

# ---------- 5. 학습 비용 (보충 표 4) ----------
# 출처: 보충 표 4. s/step, steps
cost = {
    "SFT_3k": (40.2, 3000),
    "SFT_5k": (40.2, 5000),
    "StageA_teacher": (7.9, 2000),
    "StageB_distill": (60.6, 2000),
    "BC_control_2k": (40.2, 2000),
    "FullSFT_5k": (91.4, 5000),
}
hours = {k: s * n * 1.0 / 3600 for k, (s, n) in cost.items()}
gpuh = {k: s * n * 8 / 3600 for k, (s, n) in cost.items()}
hint_total_h = hours["SFT_3k"] + hours["StageA_teacher"] + hours["StageB_distill"]
RES["cost"] = {
    "hours": {k: round(v, 2) for k, v in hours.items()},
    "GPU_hours": {k: round(v, 1) for k, v in gpuh.items()},
    "HINT_total_hours_incl_SFT3k": round(hint_total_h, 2),
    "HINT_total_hours_excl_SFT3k (본문 4.4 + 34)": round(hours["StageA_teacher"] + 34, 2),
    "ratio_HINT_over_SFT5k (paper 1.28)": round(hint_total_h / hours["SFT_5k"], 3),
    "ratio_HINT_over_FullSFT (paper 0.56)": round(hint_total_h / hours["FullSFT_5k"], 3),
    "ratio_FullSFT_over_SFT5k_step (paper 2.3)": round(91.4 / 40.2, 3),
    "ratio_StageB_over_BC_step (paper 1.51)": round(60.6 / 40.2, 3),
    "ratio_FullSFT_over_SFT5k_hours (paper 2.3)": round(hours["FullSFT_5k"] / hours["SFT_5k"], 3),
    "note": "본문 4.4절의 '4.4 h + 34 h'에는 3,000스텝 SFT(약 33.5 h)가 빠져 있다. 1.28배는 이를 포함해야 맞는다.",
}

# ---------- 6. 샘플러 스케줄과 K 선택 (코드 scheduler_continuous.py, schedule.py의 식) ----------
shift, N = 5.0, 10
u = np.linspace(1.0, 0.0, N + 1)
sigma = shift * u / (1.0 + (shift - 1.0) * u)
deltas = sigma[1:] - sigma[:-1]         # 음수
sig_pts = sigma[:-1]                    # 각 단계 시작 점 (sigma_i, i=0..N-1)
# schedule.py는 deltas(길이 N)와 timesteps(sigma_steps[:-1])를 쓴다.
w_full = deltas ** 2
w_norm = w_full / w_full.sum()
order = np.argsort(-w_norm)
ess = 1.0 / np.sum(w_norm ** 2)
RES["schedule"] = {
    "sigma_points_timesteps": [round(float(s), 4) for s in sig_pts],
    "weights_normalized": [round(float(x), 4) for x in w_norm],
    "last_point_sigma": round(float(sig_pts[-1]), 4),
    "last_point_mass_pct": round(float(w_norm[-1] * 100), 2),   # 63.6 (코드 주석)
    "last_three_mass_pct": round(float(w_norm[-3:].sum() * 100), 2),   # 91.1
    "ESS_full": round(float(ess), 3),                           # 2.22
    "top1_index_order": [int(i) for i in order[:3]],
}

# ---------- 7. 에너지 점수의 적합 규칙성 검사 (닫힌 형식, 이산 분포) ----------
# 교사 T: {-1, +1} 각 1/2. 학생 후보 P_a: 평균 0 한 점(δ0), P_b: {-1,+1} 각 1/2.
# ES_beta(P) = E|X-Y|^beta - 0.5 E|X-X'|^beta, X~P, Y~T 독립, X' ~P 독립.
def energy_score(support_p, prob_p, support_t, prob_t, beta):
    sp, pp = np.array(support_p, float), np.array(prob_p, float)
    st, pt = np.array(support_t, float), np.array(prob_t, float)
    exy = sum(pp[i] * pt[j] * abs(sp[i] - st[j]) ** beta for i in range(len(sp)) for j in range(len(st)))
    exx = sum(pp[i] * pp[j] * abs(sp[i] - sp[j]) ** beta for i in range(len(sp)) for j in range(len(sp)))
    return exy - 0.5 * exx

T_s, T_p = [-1.0, 1.0], [0.5, 0.5]
collapse = ([0.0], [1.0])
matched = ([-1.0, 1.0], [0.5, 0.5])
RES["energy_score_propriety_toy"] = {
    "beta1_unsquared_L2 (paper Eq.8, code)": {
        "collapse_to_mean": round(energy_score(*collapse, T_s, T_p, 1.0), 4),
        "matched_distribution": round(energy_score(*matched, T_s, T_p, 1.0), 4),
        "reading": "matched < collapse 이므로 분포 일치를 선호한다(엄격 적합 규칙과 일치).",
    },
    "beta2_squared (if the norms were squared)": {
        "collapse_to_mean": round(energy_score(*collapse, T_s, T_p, 2.0), 4),
        "matched_distribution": round(energy_score(*matched, T_s, T_p, 2.0), 4),
        "reading": "두 값이 같다. 제곱 노름(β=2)을 쓰면 평균만 맞추면 되어 적합 규칙이 아니다. 논문과 코드는 비제곱 노름을 쓴다.",
    },
    "squared_error_to_mean_(velocity-matching-like)": {
        "collapse_to_mean": round(float(np.sum(np.array(T_p) * (np.array(T_s) - 0.0) ** 2)), 4),
        "matched_distribution_single_sample_error": 0.0,
        "reading": "단일 교사 표본에 대한 제곱오차의 기대값은 학생 평균이 0일 때 최소가 된다. 평균 0은 두 모드 어느 쪽도 아니다.",
    },
}

# ---------- 8. 논문 내 기타 수치 대조 ----------
RES["misc_checks"] = {
    "velocity_vs_SFT_gain (paper 0.14 from 79.02-78.88)": round(79.02 - 78.88, 2),
    "MMD_minus_SFT (paper 1.7)": round(80.63 - 78.88, 2),
    "actionMSE_minus_SFT (paper 4.0)": round(82.93 - 78.88, 2),
    "energy_minus_velocity (paper 6.16)": round(85.18 - 79.02, 2),
    "BC_control_minus_SFT (paper -7.3)": round(71.56 - 78.88, 2),
    "distill_no_anchor_minus_SFT (paper -4.5)": round(74.4 - 78.9, 2),
    "shuffled_minus_SFT (paper 1.4)": round(80.28 - 78.88, 2),
    "offpolicy_states_minus_SFT (paper 1.8)": round(80.7 - 78.9, 2),
    "M1_to_M2_to_M3_to_M4_steps (paper 1.1-1.4 per sample)": [round(82.6 - 81.4, 2), round(84.0 - 82.6, 2), round(85.2 - 84.0, 2)],
    "fullSFT_minus_FastWAM_RoboTwin_clean/random": [round(84.12 - 86.48, 2), round(83.24 - 83.44, 2)],
}
RES["fastwam_orig_suites_vs_fastwam_paper"] = {
    "HINT_paper_released_avg_4suites": 96.55,
    "Fast-WAM_own_report_LIBERO_avg (Fast-WAM arXiv:2603.16666 표 2)": 97.6,
    "difference": round(96.55 - 97.6, 2),
    "note": "평가 프로토콜이 다를 수 있다. 본 논문이 재평가 조건을 밝히지 않았다.",
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(RES, f, ensure_ascii=False, indent=2)

# ---------- 요약 출력 ----------
print("표 2 평균 일치:", all(v["match"] for v in t2.values()))
for r in rows:
    print(f"{r['comparison']:34s} mean={r['mean']:+.2f} t_calc={r['t_from_mean_over_SE']:.2f} t_rep={r['t_reported']} "
          f"CI={r['95CI']} p_t={r['p_t_two_sided_from_reported_t']} sign_re={r['sign_test_p_recomputed']} rep={r['sign_test_p_reported']}")
for r in rob:
    print(f"{r['comparison']:32s} t_calc={r['t_from_mean_over_SE']:+.2f} t_rep={r['t_reported']:+.2f} p_t={r['p_t_two_sided_df24']} sign={r['sign_test_p']}")
print("cost:", RES["cost"]["ratio_HINT_over_SFT5k (paper 1.28)"], RES["cost"]["ratio_HINT_over_FullSFT (paper 0.56)"], RES["cost"]["HINT_total_hours_incl_SFT3k"])
print("schedule last mass %:", RES["schedule"]["last_point_mass_pct"], "last3 %:", RES["schedule"]["last_three_mass_pct"], "ESS:", RES["schedule"]["ESS_full"])
print("energy toy:", RES["energy_score_propriety_toy"]["beta1_unsquared_L2 (paper Eq.8, code)"], RES["energy_score_propriety_toy"]["beta2_squared (if the norms were squared)"])
print("MDE:", RES["MDE_80pct_power_pts_HINT_vs_SFT_design"], "eval noise SE floor:", RES["eval_noise_floor"]["SE_of_mean_over_90_tasks_pts"])
print("implied 80-95 band:", RES["per_task_band_consistency"]["implied_mean_gain_in_80_95_band_pts (n=20, not reported)"])
