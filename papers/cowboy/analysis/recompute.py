"""COWBOY (arXiv:2610.28948 v1) 보고 수치 재계산과 일관성 검사.

입력은 논문 표 1-4(본문)와 그림 5의 장면별 성공률(그림 이미지를 눈으로 읽은 값)이다.
출력: analysis/recompute_results.json, 표준 출력에 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

검사 번호는 V1-V12. 03_analysis.md 의 번호와 같다.
"""
import json
from math import sqrt, comb
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import linprog

OUT = Path(__file__).resolve().parent / "recompute_results.json"

# ---- 논문 표 (본문 표 1-4) -------------------------------------------------
CATS = ["TableTop", "Drawer", "Box", "Shelf"]
T1_MANIPGEN = [54.8, 89.2, 87.2, 21.2]
T1_MANIPGEN_AVG = 63.1
T1_COWBOY = [94.7, 92.9, 91.3, 90.5]
T1_COWBOY_AVG = 92.4
T2_TEACHER_SUCC, T2_TEACHER_SUCC_AVG = [96.0, 95.6, 93.4, 97.3], 95.6
T2_STUDENT_SUCC, T2_STUDENT_SUCC_AVG = [94.7, 92.9, 91.3, 90.5], 92.4
T2_TEACHER_COLL, T2_TEACHER_COLL_AVG = [3.9, 4.8, 6.2, 2.4], 4.3
T2_STUDENT_COLL, T2_STUDENT_COLL_AVG = [5.3, 7.2, 8.1, 3.6], 6.1
# 표 3: 순서 = No active vision, No LiDAR, No sim render, Ours
T3_NAMES = ["No active vision", "No LiDAR", "No sim render", "Ours"]
T3_SIM_SUCC = [78.8, 75.2, 94.4, 92.4]
T3_REAL_SUCC = [46.7, 71.1, 0.0, 91.1]
T3_SIM_COLL = [9.4, 16.7, 5.6, 6.4]
T3_REAL_COLL = [8.2, 31.7, 11.3, 3.9]
# 표 4: Weighted IK, Fabrics w/o collision avoidance, Full Fabrics
T4_SUCC = [43.6, 74.8, 95.6]
T4_COLL = [41.4, 32.6, 4.3]
T4_MANIP = [0.039, 0.067, 0.072]
T4_MARGIN = [1.51, 1.87, 1.99]
# 그림 5: 장면 24개의 장면별 성공률 (행 우선, 그림 이미지 판독)
FIG5 = [89, 93, 85, 97, 84, 85,
        91, 97, 98, 82, 92, 96,
        96, 81, 80, 92, 91, 88,
        84, 91, 93, 83, 98, 84]
TRIALS_PER_SCENE = (20 + 5) * 3   # 고정 20 + 무작위 5 물체, 방향 3 (본문 4절)
N_SCENES = 24


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def feasible(pct, n, decimals):
    """k/n 이 pct 로 (decimals 자리에서) 반올림되는 정수 k 가 있는가. 있으면 k 목록."""
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    return [k for k in range(n + 1) if abs(100 * k / n - pct) <= tol]


def two_prop_p(p1, p2, n):
    """동일 n 인 두 비율 차이의 양측 z 검정 (비짝지음, 보수적)."""
    p1, p2 = p1 / 100, p2 / 100
    pp = (p1 + p2) / 2
    se = sqrt(2 * pp * (1 - pp) / n)
    z = (p1 - p2) / se
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


def main():
    R = {}

    # V1 표 1-2 평균
    avg = lambda a: round(float(np.mean(a)), 3)
    R["V1_table_means"] = {
        "manipgen_mean": avg(T1_MANIPGEN), "manipgen_reported": T1_MANIPGEN_AVG,
        "cowboy_mean": avg(T1_COWBOY), "cowboy_reported": T1_COWBOY_AVG,
        "teacher_succ_mean": avg(T2_TEACHER_SUCC), "teacher_succ_reported": T2_TEACHER_SUCC_AVG,
        "student_succ_mean": avg(T2_STUDENT_SUCC), "student_succ_reported": T2_STUDENT_SUCC_AVG,
        "teacher_coll_mean": avg(T2_TEACHER_COLL), "teacher_coll_reported": T2_TEACHER_COLL_AVG,
        "student_coll_mean": avg(T2_STUDENT_COLL), "student_coll_reported": T2_STUDENT_COLL_AVG,
        "table1_cowboy_equals_table2_student": T1_COWBOY == T2_STUDENT_SUCC,
        "table4_full_equals_table2_teacher": (T4_SUCC[2], T4_COLL[2]) == (T2_TEACHER_SUCC_AVG, T2_TEACHER_COLL_AVG),
    }

    # V2 표 3 'Ours' 시뮬 열 대 표 2 학생
    R["V2_table3_vs_table2"] = {
        "succ_table3": T3_SIM_SUCC[3], "succ_table2": T2_STUDENT_SUCC_AVG,
        "coll_table3": T3_SIM_COLL[3], "coll_table2": T2_STUDENT_COLL_AVG,
        "coll_gap": round(T3_SIM_COLL[3] - T2_STUDENT_COLL_AVG, 2),
    }
    # 범주별 환경 수가 다르면 가중 평균이 달라질 수 있다. 표 2 교사·학생 네 값(성공 2, 충돌 2)과
    # 표 3의 92.4 / 6.4 를 동시에 만족하는 가중치가 있는지 선형계획으로 본다.
    A_ub, b_ub = [], []
    def band(vals, lo, hi):
        A_ub.append(list(vals)); b_ub.append(hi)
        A_ub.append([-v for v in vals]); b_ub.append(-lo)
    band(T2_TEACHER_SUCC, 95.55, 95.65)
    band(T2_TEACHER_COLL, 4.25, 4.35)
    band(T2_STUDENT_SUCC, 92.35, 92.45)
    band(T2_STUDENT_COLL, 6.35, 6.45)   # 표 3의 6.4
    res_a = linprog(np.zeros(4), A_ub=A_ub, b_ub=b_ub, A_eq=[[1, 1, 1, 1]], b_eq=[1], bounds=[(0, 1)] * 4)
    # 표 2의 6.1 을 그대로 쓴 경우 (균등 가중 6.05 도 이 구간에 있다)
    A_ub2, b_ub2 = [], []
    A_ub, b_ub = A_ub2, b_ub2
    band(T2_TEACHER_SUCC, 95.55, 95.65)
    band(T2_TEACHER_COLL, 4.25, 4.35)
    band(T2_STUDENT_SUCC, 92.35, 92.45)
    band(T2_STUDENT_COLL, 6.05, 6.15)
    res_b = linprog(np.zeros(4), A_ub=A_ub2, b_ub=b_ub2, A_eq=[[1, 1, 1, 1]], b_eq=[1], bounds=[(0, 1)] * 4)
    # 충돌 6.4 가 되려면 드로어·박스 가중이 얼마나 필요한지 (교사 값 제약 없이 학생만)
    A_c, b_c = [], []
    A_ub, b_ub = A_c, b_c
    band(T2_STUDENT_SUCC, 92.35, 92.45)
    band(T2_STUDENT_COLL, 6.35, 6.45)
    res_c = linprog(np.zeros(4), A_ub=A_c, b_ub=b_c, A_eq=[[1, 1, 1, 1]], b_eq=[1], bounds=[(0, 1)] * 4)
    # 학생만 보면 가능한가, 교사까지 같은 가중치면 가능한가
    R["V2_weight_feasibility"] = {
        "student_only_succ92.4_coll6.4_feasible": bool(res_c.status == 0),
        "student_only_weights": [round(x, 3) for x in res_c.x] if res_c.status == 0 else None,
        "teacher_and_student_same_weights_coll6.4_feasible": bool(res_a.status == 0),
        "teacher_and_student_same_weights_coll6.1_feasible": bool(res_b.status == 0),
        "equal_weight_student_coll": avg(T2_STUDENT_COLL),
    }

    # V3 그림 5 장면별 성공률: 평균, 구간, 정수 가능성
    f5 = np.array(FIG5, dtype=float)
    mean5 = float(f5.mean())
    sd5 = float(f5.std(ddof=1))
    se5 = sd5 / sqrt(N_SCENES)
    tcrit = stats.t.ppf(0.975, N_SCENES - 1)
    n_total = TRIALS_PER_SCENE * N_SCENES
    infeasible = [(i + 1, v) for i, v in enumerate(FIG5) if not feasible(v, TRIALS_PER_SCENE, 0)]
    # 75회가 아닌 n 에서 불가능한 값이 모두 가능해지는 n 이 있는가
    ok_n = []
    for n in range(20, 151):
        if all(feasible(v, n, 0) for v in FIG5):
            ok_n.append(n)
    scene_k = [feasible(v, TRIALS_PER_SCENE, 0)[0] if feasible(v, TRIALS_PER_SCENE, 0) else None for v in FIG5]
    # 균질성 검정: 장면별 성공 횟수가 풀링 비율의 이항 변동보다 큰가 (불가능한 값은 가장 가까운 정수로)
    k_near = [round(v * TRIALS_PER_SCENE / 100) for v in FIG5]
    pool = sum(k_near) / (TRIALS_PER_SCENE * N_SCENES)
    chi2 = sum((k - TRIALS_PER_SCENE * pool) ** 2 / (TRIALS_PER_SCENE * pool * (1 - pool)) for k in k_near)
    p_chi = 1 - stats.chi2.cdf(chi2, N_SCENES - 1)
    binom_sd = sqrt(pool * (1 - pool) / TRIALS_PER_SCENE) * 100
    R["V3_fig5"] = {
        "values": FIG5, "n_scenes": N_SCENES, "trials_per_scene_assumed": TRIALS_PER_SCENE,
        "total_trials": n_total,
        "mean_of_scenes": round(mean5, 3), "sd_between_scenes": round(sd5, 2),
        "t_ci95_between_scenes": [round(mean5 - tcrit * se5, 2), round(mean5 + tcrit * se5, 2)],
        "reported_headline": 91, "reported_table3_real": 91.1,
        "gap_vs_headline": round(91 - mean5, 2), "gap_vs_table3": round(91.1 - mean5, 2),
        "implied_success_count_from_fig5": round(mean5 * n_total / 100, 1),
        "table3_success_count_if_1800": round(0.911 * n_total, 1),
        "missing_successes": round(0.911 * n_total - mean5 * n_total / 100, 1),
        "wilson_95_if_1800_at_89.58": [round(x, 2) for x in wilson(round(mean5 * n_total / 100), n_total)],
        "wilson_95_if_1800_at_91.1": [round(x, 2) for x in wilson(round(0.911 * n_total), n_total)],
        "infeasible_at_75": infeasible,
        "n_values_20_150_where_all_24_feasible": ok_n,
        "heterogeneity_chi2": round(chi2, 1), "heterogeneity_df": N_SCENES - 1, "heterogeneity_p": float(p_chi),
        "binomial_sd_pts": round(binom_sd, 2),
        "min_scene": [int(f5.argmin()) + 1, int(f5.min())], "max_scene": [int(f5.argmax()) + 1, int(f5.max())],
        "fixed20_share_of_trials": 60 / 75,
    }
    # 98%, 82% 가 가능한 n (n <= 75)
    R["V3_fig5"]["feasible_n_le75_for_98"] = [n for n in range(1, 76) if feasible(98, n, 0)]
    R["V3_fig5"]["feasible_n_le75_for_82"] = [n for n in range(1, 76) if feasible(82, n, 0)]

    # V4 표 3 실로봇 열의 정수 가능성과 최소 공통 n
    real_vals = T3_REAL_SUCC + T3_REAL_COLL
    min_n = None
    for n in range(1, 4001):
        if all(feasible(v, n, 1) for v in real_vals):
            min_n = n
            break
    all_n = [n for n in range(1, 4001) if all(feasible(v, n, 1) for v in real_vals)]
    R["V4_table3_real_integer_chain"] = {
        "values": real_vals, "min_common_n": min_n, "common_n_up_to_4000_first10": all_n[:10],
        "feasible_at_1800": all(feasible(v, 1800, 1) for v in real_vals),
        "feasible_at_45": [v for v in real_vals if not feasible(v, 45, 1)],
        "feasible_at_75": [v for v in real_vals if not feasible(v, 75, 1)],
        "counts_at_1800": {str(v): feasible(v, 1800, 1)[:1] for v in real_vals},
    }

    # V5 표 3 실로봇: 성공 + 충돌 합 (성공 정의에 '충돌 없음'이 들어 있다)
    sums_real = [round(s + c, 1) for s, c in zip(T3_REAL_SUCC, T3_REAL_COLL)]
    sums_sim = [round(s + c, 1) for s, c in zip(T3_SIM_SUCC, T3_SIM_COLL)]
    sums_t4 = [round(s + c, 1) for s, c in zip(T4_SUCC, T4_COLL)]
    R["V5_success_plus_collision"] = {
        "real_names": T3_NAMES, "real_sum": sums_real, "real_over_100": [n for n, s in zip(T3_NAMES, sums_real) if s > 100],
        "sim_sum": sums_sim, "table4_sum": sums_t4,
        "note": "실로봇 성공은 '충돌 없이 10 cm 들어올림'(본문 4절)이므로 성공+충돌 환경 비율은 100을 넘을 수 없다. 시뮬 성공 정의에는 충돌 조건이 없다(5절).",
    }

    # V6 표 2 교사 대 학생 (범주당 환경 수를 2000으로 가정; 총 8000/4)
    n_cat = 2000
    rows = []
    for c, t, s in zip(CATS, T2_TEACHER_SUCC, T2_STUDENT_SUCC):
        z, p = two_prop_p(t, s, n_cat)
        rows.append({"cat": c, "teacher": t, "student": s, "gap": round(t - s, 1), "z": round(z, 2), "p_unpaired": float(p)})
    R["V6_teacher_vs_student"] = {"assumed_envs_per_category": n_cat, "rows": rows,
                                  "gap_mean": round(float(np.mean(T2_TEACHER_SUCC) - np.mean(T2_STUDENT_SUCC)), 2)}
    rows = []
    for c, m, w in zip(CATS, T1_MANIPGEN, T1_COWBOY):
        z, p = two_prop_p(w, m, n_cat)
        rows.append({"cat": c, "manipgen": m, "cowboy": w, "gap": round(w - m, 1), "z": round(z, 2), "p_unpaired": float(p)})
    R["V6_cowboy_vs_manipgen"] = {"assumed_envs_per_category": n_cat, "rows": rows}

    # V7 WBC 진입률과 조건부 성공률
    enter = 0.984
    R["V7_wbc_entry"] = {
        "entry_rate": enter, "teacher_avg_success": T2_TEACHER_SUCC_AVG,
        "success_given_entry": round(T2_TEACHER_SUCC_AVG / 100 / enter * 100, 2),
        "teacher_shelf_success": T2_TEACHER_SUCC[3],
        "shelf_success_given_entry": round(T2_TEACHER_SUCC[3] / 100 / enter * 100, 2),
        "note": "교사 성공률은 진입 실패를 포함하므로 진입 후 조건부 성공률은 더 높다. 선반 97.3/0.984 = 98.9%.",
    }

    # V8 차원·개수 일관성
    obs_actor = [16, 12, 3, 3, 6, 3, 6, 3, 6, 3]
    obs_critic = [23, 23, 3, 6, 6, 12, 3, 3, 6, 3, 6, 3, 6, 3]
    R["V8_dimensions"] = {
        "actor_obs_sum": sum(obs_actor), "actor_obs_reported": 61,
        "critic_obs_sum": sum(obs_critic), "critic_obs_reported": 106,
        "expert_action": 3 + 3 + 16, "expert_action_reported": 22,
        "student_action": 3 + 7 + 16 + 6, "student_action_reported": 32,
        "total_dof": 3 + 7 + 16 + 6, "total_dof_reported": 32,
        "token_width": 124 + 4, "token_width_reported": 128, "head_dim": 128 / 8,
        "point_cloud_total": 2 * (1024 + 512), "point_cloud_reported": 3072,
        "envs_distill_text": 8192, "envs_per_rank_x_ranks": 512 * 8,
        "distill_env_mismatch": 8192 != 512 * 8,
        "wbc_max_envs": 4096, "distill_env_exceeds_wbc_max": 8192 > 4096,
        "eval_envs_text_sec4": 8192, "eval_envs_table_caption_and_G2": 8000,
        "expert_objects_topdown_constrained": 304, "expert_objects_side": 195, "union_if_disjoint": 304 + 195,
        "distill_objects_text": 250,
        "real_objects": {"fixed": 20, "random_pool": 100, "draws_per_scene": 5, "total_random_draws": 5 * N_SCENES},
    }

    # V9 비용·처리량 산술
    wbc_ms = 0.03   # ms / env / step (3.1절)
    depth_s_1024, lidar_s_1024 = 0.07, 0.25   # s per step, 1024 envs (3.3절, 부록 F)
    R["V9_compute_arithmetic"] = {
        "wbc_batch_step_ms_4096": round(wbc_ms * 4096, 1),
        "wbc_batch_step_ms_8192": round(wbc_ms * 8192, 1),
        "wbc_max_env_steps_per_s": round(1000 / wbc_ms, 0),
        "wbc_only_upper_bound_env_steps_in_36h": round(36 * 3600 * 1000 / wbc_ms / 1e9, 2),   # 10^9 단위
        "wbc_only_upper_bound_iterations_8192env_36h": round(36 * 3600 / (wbc_ms * 8192 / 1000), 0),
        "render_ms_per_env_depth": round(depth_s_1024 / 1024 * 1000, 3),
        "render_ms_per_env_lidar": round(lidar_s_1024 / 1024 * 1000, 3),
        "render_s_per_step_512env_rank": round((depth_s_1024 + lidar_s_1024) * 512 / 1024, 3),
        "render_only_upper_bound_steps_in_60h_per_rank": round(60 * 3600 / ((depth_s_1024 + lidar_s_1024) * 512 / 1024), 0),
        "gpu_hours_experts": 3 * 36, "gpu_hours_distill": 8 * 60, "gpu_hours_total": 3 * 36 + 8 * 60,
        "ppo_batch_per_iter": 8192 * 32, "ppo_minibatches_per_epoch": 8192 * 32 // 8192,
        "ppo_grad_steps_per_iter": 8192 * 32 // 8192 * 5,
        "dr_ramp_200k_sim_steps_as_ppo_iters_if_per_env": 200000 / 32,
        "dr_ramp_if_total_env_steps_iters": round(200000 / (8192 * 32), 2),
        "teacher_forcing_steps": {"full": 10000, "decay_to_zero_after": 20000},
        "teleport_quadratic_a": 0.5 / (300 - 1 + 1 - 150) ** 2,
    }

    # V10 표 3·4 비율 간 차이 (구간): 가정 n 별 필요 시험 수
    def n_needed(p1, p2, power=0.8, alpha=0.05):
        p1, p2 = p1 / 100, p2 / 100
        za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
        pbar = (p1 + p2) / 2
        num = (za * sqrt(2 * pbar * (1 - pbar)) + zb * sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        return num / (p1 - p2) ** 2
    R["V10_real_ablation_power"] = {
        "ours_vs_no_lidar_needed_per_arm": round(n_needed(91.1, 71.1), 1),
        "ours_vs_no_active_vision_needed_per_arm": round(n_needed(91.1, 46.7), 1),
        "real_vs_sim_success_gap": round(T3_SIM_SUCC[3] - T3_REAL_SUCC[3], 1),
        "ours_real_vs_fig5_mean_needed_per_arm": round(n_needed(91.1, mean5), 0),
    }
    # 표 4 불균형: 표 4 Fabric w/o collision 74.8 + 32.6 > 100 (시뮬 성공에 충돌 조건 없음)
    R["V11_table4"] = {
        "succ_plus_coll": sums_t4,
        "gains_full_vs_ik": {"succ_pts": round(T4_SUCC[2] - T4_SUCC[0], 1), "coll_pts": round(T4_COLL[2] - T4_COLL[0], 1),
                             "manip_ratio": round(T4_MANIP[2] / T4_MANIP[0], 2), "margin_gain_rad": round(T4_MARGIN[2] - T4_MARGIN[0], 2)},
        "gains_full_vs_nocoll": {"succ_pts": round(T4_SUCC[2] - T4_SUCC[1], 1), "coll_pts": round(T4_COLL[2] - T4_COLL[1], 1),
                                 "manip_ratio": round(T4_MANIP[2] / T4_MANIP[1], 2)},
        "wilson_95_full_assuming_8000": [round(x, 2) for x in wilson(round(0.956 * 8000), 8000)],
        "note": "표 4의 환경 수는 8000(부록 G.2)으로만 알 수 있다. 시드 반복과 신뢰구간 없음.",
    }

    # V12 Wilson: 표 3 실로봇 열 (n=1800 가정, 구간)
    R["V12_wilson_real_table3"] = {
        name: [round(x, 1) for x in wilson(round(v / 100 * 1800), 1800)]
        for name, v in zip(T3_NAMES, T3_REAL_SUCC)
    }
    R["V12_wilson_real_table3"]["assumption"] = "n=1800(24 장면 x 75회). 절제 열의 실제 시도 수는 논문에 없다."

    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in R.items():
        print(k, json.dumps(v, ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()
