"""Kamino (arXiv:2603.16536) 보고 수치 재계산.

실행: python3 analysis/recompute.py   → analysis/recompute_results.json (수 초, numpy·scipy)

옮겨 적은 값의 출처는 각 블록 주석에 적는다(v2 = arXiv v2, 2026-10-20). 그림에서 읽은 값은 "판독"이라고 표시한다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}

# ---------------------------------------------------------------------------
# V1. 표 1 (v2 p.7): 모델 x 지표 x (median, p90) x 솔버. None = 보고 없음("-")
# 솔버 순서: Kamino, MJWarp, MuJoCo CPU, XPBD
# ---------------------------------------------------------------------------
SOLVERS = ["Kamino", "MJWarp", "MuJoCo", "XPBD"]
T1 = {
    "BDX": {
        "rJ": [(1.12e-4, 5.72e-4), (1.63e-7, 2.19e-7), (1.48e-7, 2.00e-7), (6.12e-3, 9.99e-3)],
        "rkin": [(5.90e-4, 1.63e-3), (9.60e-2, 2.92e-1), (9.82e-2, 2.95e-1), (6.51, 9.86)],
        "rC": [(2.95e-4, 3.62e-4), (3.05e-2, 3.43e-2), (2.55e-3, 7.75e-3), (2.06e-4, 2.22e-4)],
        "rv": [(3.05e-7, 1.84e-1), (1.02e-2, 2.10e-1), (1.51e-2, 2.01e-1), (9.06e-3, 2.00e-1)],
        "rprimal": [(0.0, 3.73e-9), (0.0, 1.49e-8), None, (0.0, 0.0)],
        "rdual": [(2.09e-7, 1.23e-1), (6.84e-3, 1.43e-1), (1.01e-2, 1.35e-1), (6.08e-3, 1.35e-1)],
        "rcmpl": [(8.35e-8, 1.29e-5), (1.24e-2, 1.34e-1), None, (2.83e-3, 2.44e-2)],
        "rvi": [(3.03e-4, 1.23e-1), (7.23e-2, 2.56e-1), None, (2.20e-2, 1.74e-1)],
    },
    "Olaf": {
        "rJ": [(1.40e-4, 8.30e-4), (9.54e-4, 2.02e-3), (1.02e-3, 2.82e-3), (5.38e-2, 1.38e-1)],
        "rkin": [(4.66e-3, 6.51e-2), (9.42e-3, 5.29e-2), (9.46e-3, 5.23e-2), (1.10, 1.56)],
        "rC": [(2.97e-4, 3.05e-4), (8.08e-2, 8.09e-2), (4.91e-2, 9.70e-2), (2.06e-4, 2.81e-4)],
        "rv": [(9.19e-7, 8.39e-6), (0.0, 5.40e-2), (9.50e-3, 6.83e-2), (7.20e-3, 1.31e-2)],
        "rprimal": [(0.0, 4.66e-10), (0.0, 1.86e-9), None, (0.0, 1.46e-11)],
        "rdual": [(6.15e-7, 5.63e-6), (0.0, 3.92e-2), (7.00e-3, 5.24e-2), (4.83e-3, 8.81e-3)],
        "rcmpl": [(3.73e-6, 1.95e-5), (2.54e-3, 1.90e-2), None, (8.38e-3, 2.48e-2)],
        "rvi": [(7.58e-5, 4.26e-4), (4.06e-2, 9.11e-2), None, (2.77e-2, 3.29e-2)],
    },
    "DR Legs": {
        "rJ": [(1.89e-4, 1.41e-3), (2.52e-2, 3.20e-2), (2.15e-2, 3.03e-2), (4.71e-3, 7.33e-3)],
        "rkin": [(1.05e-4, 4.01e-4), (3.41e-2, 1.32e-1), (1.37e-2, 8.23e-2), (1.84, 4.18)],
        "rC": [(8.97e-5, 1.25e-4), (7.56e-4, 2.73e-3), (7.52e-4, 1.78e-3), (2.43e-5, 3.11e-5)],
        "rv": [(6.92e-4, 7.92e-1), (6.73e-3, 7.90e-1), (1.45e-2, 7.90e-1), (1.62e-2, 6.16e-1)],
        "rprimal": [(0.0, 4.66e-10), (0.0, 0.0), None, (0.0, 2.91e-11)],
        "rdual": [(6.83e-4, 4.43e-1), (6.52e-3, 4.40e-1), (1.40e-2, 4.40e-1), (8.55e-3, 3.38e-1)],
        "rcmpl": [(1.50e-8, 1.18e-7), (1.39e-5, 1.48e-5), None, (2.71e-5, 3.04e-5)],
        "rvi": [(6.83e-4, 4.43e-1), (6.52e-3, 4.40e-1), None, (8.55e-3, 3.38e-1)],
    },
    "Iron Man": {
        "rJ": [(4.43e-3, 1.91e-2), (9.59e-2, 1.68e-1), (7.16e-2, 1.59e-1), (4.24e-3, 9.38e-3)],
        "rkin": [(2.99e-3, 8.66e-3), (6.08e-1, 1.85), (6.10e-1, 1.82), (26.6, 39.7)],
    },
}


def cell_stats(stat_idx):
    rows = []
    for model, metrics in T1.items():
        for metric, vals in metrics.items():
            v = [None if x is None else x[stat_idx] for x in vals]
            avail = [(i, x) for i, x in enumerate(v) if x is not None]
            best = min(x for _, x in avail)
            winners = [SOLVERS[i] for i, x in avail if x == best]
            k = v[0]
            others = [x for i, x in avail if i != 0]
            best_other = min(others)
            # Kamino 순위 (동률은 같은 순위)
            rank = 1 + sum(1 for x in others if x < k)
            if best_other == 0.0:
                ratio = None if k == 0 else float("inf")
            else:
                ratio = k / best_other
            rows.append(dict(model=model, metric=metric, kamino=k, best_other=best_other,
                             ratio_kamino_over_best_other=ratio, kamino_rank=rank,
                             n_solvers=len(avail), winners=winners))
    return rows


def summarize(rows, label):
    n = len(rows)
    first = sum(1 for r in rows if r["kamino_rank"] == 1)
    near2 = sum(1 for r in rows if r["kamino_rank"] > 1 and r["ratio_kamino_over_best_other"] is not None
                and r["ratio_kamino_over_best_other"] <= 2.0)
    worse = n - first - near2
    return dict(label=label, n_cells=n, kamino_best=first, kamino_within_2x=near2, kamino_worse_than_2x=worse)


rows_med = cell_stats(0)
rows_p90 = cell_stats(1)
res["V1_table1"] = dict(
    median=summarize(rows_med, "median"),
    p90=summarize(rows_p90, "p90"),
    median_cells=rows_med,
    p90_cells=rows_p90,
)
# 지표군별 요약
groups = {"configuration (rJ, rC)": ["rJ", "rC"], "velocity (rkin, rv)": ["rkin", "rv"],
          "NCP (rprimal, rdual, rcmpl, rvi)": ["rprimal", "rdual", "rcmpl", "rvi"]}
res["V1_table1"]["by_group"] = {}
for g, ms in groups.items():
    for stat_name, rows in (("median", rows_med), ("p90", rows_p90)):
        sub = [r for r in rows if r["metric"] in ms]
        res["V1_table1"]["by_group"][f"{g} | {stat_name}"] = summarize(sub, g)
# 중앙값 대 p90 비율 (꼬리가 두꺼운 정도) — Kamino
tail = {}
for model, metrics in T1.items():
    for metric, vals in metrics.items():
        m, p = vals[0]
        if m > 0:
            tail[f"{model}/{metric}"] = p / m
res["V1_table1"]["kamino_p90_over_median"] = tail
# rdual == rvi (DR Legs 네 솔버 모두 같은 값) 확인
dl = T1["DR Legs"]
res["V1_table1"]["drlegs_rdual_equals_rvi"] = {
    SOLVERS[i]: (dl["rdual"][i] == dl["rvi"][i]) for i in (0, 1, 3)}
bo = T1["BDX"]
res["V1_table1"]["bdx_rdual_equals_rvi"] = {SOLVERS[i]: (bo["rdual"][i] == bo["rvi"][i]) for i in (0, 1, 3)}
# XPBD rC 중앙값이 BDX와 Olaf에서 같음
res["V1_table1"]["xpbd_rC_median_BDX_vs_Olaf"] = [T1["BDX"]["rC"][3][0], T1["Olaf"]["rC"][3][0]]
res["V1_table1"]["xpbd_rC_median_all_contact_models"] = [T1[m]["rC"][3][0] for m in ("BDX", "Olaf", "DR Legs")]
res["V1_table1"]["kamino_rC_median_BDX_vs_Olaf"] = [T1["BDX"]["rC"][0][0], T1["Olaf"]["rC"][0][0]]
# 속도 수준 지표 rkin에서 p90 비교
res["V1_table1"]["rkin_p90_olaf"] = {SOLVERS[i]: T1["Olaf"]["rkin"][i][1] for i in range(4)}
# 캡션: 기준 궤적 최대 p90 3e-4 vs 표의 Kamino rJ p90 (자체 궤적)
res["V1_table1"]["kamino_rJ_p90"] = {m: T1[m]["rJ"][0][1] for m in T1}

# ---------------------------------------------------------------------------
# V2. 표 A.1 위상 항등식: 독립 루프 수 = 관절 수 - 몸체 수 + 1 (연결 그래프, 순환 수)
#     고정 베이스 모델은 월드 노드를 몸체로 센다.
# ---------------------------------------------------------------------------
TAB_A1 = {  # model: (bodies, actuated, passive, loops)  v2 표 A.1
    "DR Legs": (31, 12, 24, 6), "BDX": (15, 14, 0, 0), "Olaf": (27, 20, 9, 3),
    "Iron Man": (67, 23, 57, 13), "Hoberman Sphere": (240, 0, 421, 182),
    "Ant-3": (7, 6, 0, 0), "Ant-4": (9, 8, 0, 0), "Ant-5": (11, 10, 0, 0),
}
V1TAB = {  # v1 표 1
    "DR Legs": (31, 12, 24, 6), "BDX": (21, 14, 0, 0), "Olaf": (36, 25, 15, 5), "Iron Man": (151, 27, 169, 45),
}
v2 = {}
for k, (b, a, p, loops) in TAB_A1.items():
    j = a + p
    pred_float = j - b + 1  # 월드 노드 없이 (플로팅 베이스)
    pred_fixed = j - (b + 1) + 1  # 월드 노드 포함 (고정 베이스)
    v2[k] = dict(bodies=b, joints=j, loops_reported=loops, loops_floating_formula=pred_float,
                 loops_fixed_base_formula=pred_fixed)
res["V2_loops_v2"] = v2
v1 = {}
for k, (b, a, p, loops) in V1TAB.items():
    j = a + p
    v1[k] = dict(bodies=b, joints=j, loops_reported=loops, loops_floating_formula=j - b + 1,
                 loops_fixed_base_formula=j - b)
res["V2_loops_v1"] = v1
# 최소 Delassus 차원 (그림 3, v2 / 그림 5, v1): 회전 관절 1개당 구속 5행 가정
res["V2_min_delassus_dim"] = {
    "v2_figure3": {"BDX": 70, "Olaf": 148, "DR Legs": 180, "Iron Man": 377},
    "v1_figure5": {"BDX": 84, "Olaf": 168, "DR Legs": 192, "Iron Man": 955},
    "revolute_5_rows_v2": {k: 5 * (TAB_A1[k][1] + TAB_A1[k][2]) for k in ("BDX", "Olaf", "DR Legs", "Iron Man")},
    "revolute_5_rows_plus_actuated_v1": {k: 5 * (V1TAB[k][1] + V1TAB[k][2]) + V1TAB[k][1] for k in V1TAB},
}

# ---------------------------------------------------------------------------
# V3. 표 A.5 이종 세계 처리량 (2,048 환경)
# ---------------------------------------------------------------------------
TAB_A5 = {"3-leg": 286510, "4-leg": 130308, "5-leg": 53360, "hetero": 86246}
t = {k: 1.0 / v for k, v in TAB_A5.items() if k != "hetero"}
ideal_mix = 1.0 / np.mean(list(t.values()))  # 세 종류가 같은 비율로 섞이고 비용이 합산된다고 가정
res["V3_ant_hetero"] = dict(
    reported=TAB_A5,
    ideal_harmonic_mix_samples_per_s=float(ideal_mix),
    hetero_over_ideal=float(TAB_A5["hetero"] / ideal_mix),
    overhead_pct=float(100 * (1 - TAB_A5["hetero"] / ideal_mix)),
    ratio_hetero_over_5leg=TAB_A5["hetero"] / TAB_A5["5-leg"],
    ratio_hetero_over_4leg=TAB_A5["hetero"] / TAB_A5["4-leg"],
    ordering_text_claim_ok=bool(TAB_A5["5-leg"] < TAB_A5["hetero"] < TAB_A5["4-leg"]),
    note="모두 같은 크기(5-leg)로 패딩하면 5-leg 값(53,360) 근처일 것. 보고값이 더 높으므로 조기 종료 마스크가 작동했다고 읽을 수 있음",
)

# ---------------------------------------------------------------------------
# V4. 이항·시드 통계
# ---------------------------------------------------------------------------
def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def clopper(k, n, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


res["V4_manipulation"] = dict(k=100, n=100, wilson95=wilson(100, 100), clopper_pearson95=clopper(100, 100),
                              rule_of_three_upper_fail_rate=3 / 100)
n_seed = 10
mean, pm = 98.5, 0.4
t_crit = stats.t.ppf(0.975, n_seed - 1)
res["V4_seeds"] = dict(
    n_seeds=n_seed, mean_pct=mean, pm_pct=pm,
    ci95_if_pm_is_SD=[mean - t_crit * pm / math.sqrt(n_seed), mean + t_crit * pm / math.sqrt(n_seed)],
    ci95_if_pm_is_SE=[mean - t_crit * pm, mean + t_crit * pm],
    note="± 의 정의(SD/SE)와 에피소드 수가 본문에 없음. 평균 98.5%는 에피소드 1,000개(985/1000)만으로도 설명되지만 개수는 미보고",
)

# ---------------------------------------------------------------------------
# V5. 처리량 비율과 시간 간격 보정 (v2 §3.2, §3.3, 부록 A.6)
# ---------------------------------------------------------------------------
thr = {"Kamino(sparse LLT)": 2.4e5, "XPBD": 5.8e5, "MJWarp": 6.1e6}
res["V5_throughput_BDX"] = dict(
    reported=thr,
    xpbd_over_kamino=thr["XPBD"] / thr["Kamino(sparse LLT)"],
    mjwarp_over_kamino=thr["MJWarp"] / thr["Kamino(sparse LLT)"],
    fig_A6_reading_kamino_BDX_peak="약 2.4e5 (판독, 본문과 일치)",
)
dt = {"DR Legs": {"Kamino": 1.25e-3, "XPBD": 0.2e-3}, "Iron Man": {"Kamino": 5e-3, "MJWarp": 0.2e-3, "XPBD": 0.2e-3},
      "Hoberman": {"all": 2e-3}}
res["V5_timestep_breakeven"] = dict(
    DR_Legs_Kamino_over_XPBD_dt=dt["DR Legs"]["Kamino"] / dt["DR Legs"]["XPBD"],
    IronMan_Kamino_over_MJWarp_dt=dt["Iron Man"]["Kamino"] / dt["Iron Man"]["MJWarp"],
    note=("같은 시뮬레이션 시간을 계산할 때 Kamino는 스텝이 6.25배(DR Legs, 대 XPBD), 25배(Iron Man, 대 MJWarp·XPBD) 적다. "
          "따라서 스텝당 처리량 비가 이 값보다 작으면 시뮬레이션 시간당 속도는 Kamino가 빠르다. "
          "논문은 BDX 외에는 MJWarp·XPBD의 처리량을 보고하지 않아 이 비교를 완성할 수 없다."),
    BDX_step_speed_ratio_MJWarp=thr["MJWarp"] / thr["Kamino(sparse LLT)"],
)
# 그림 A.6 판독 (RTX Pro 6000, 세계 수 약 8,000에서의 대략값, 판독 오차 약 ±5%)
res["V5_figA6_reading"] = {
    "BDX": {"LLT sparse": 200000, "LLT dense": 190000, "CR": 40000},
    "Olaf": {"LLT sparse": 68000, "LLT dense": 59000, "CR": 18500},
    "DR Legs": {"LLT sparse": 59000, "LLT dense": 48000, "CR": 19000},
    "Iron Man": {"LLT sparse": 16700, "LLT dense": 5900, "CR": 10500, "CR peak(~1800 worlds)": 19000},
}
iron = res["V5_figA6_reading"]["Iron Man"]
res["V5_ironman_CR_vs_LLT"] = dict(
    cr_peak_over_llt_sparse_plateau=iron["CR peak(~1800 worlds)"] / iron["LLT sparse"],
    cr_plateau_over_llt_sparse_plateau=iron["CR"] / iron["LLT sparse"],
    note="Iron Man에서 CR이 가장 빠르다는 서술은 최대 처리량(약 1,800 세계) 기준이다. 4,000세계 이상에서는 LLT 희소가 더 빠르다(판독).",
)

# ---------------------------------------------------------------------------
# V6. DR Legs 학습 시간 일관성 (v1 대 v2)
# ---------------------------------------------------------------------------
envs, iters, dec = 4096, 4000, 5
v1_hours, v1_fps = 31.0, 3600.0
v1_tr = 390e6
res["V6_rl_time"] = {}
res["V6_rl_time"]["v1_steps_per_iter_implied"] = v1_tr / (iters * envs)  # 약 24
res["V6_rl_time"]["v1_total_from_fps_hours_check"] = v1_tr / v1_fps / 3600
res["V6_rl_time"]["v1_sim_steps_per_s"] = v1_fps * dec
for name, hours, bench in (("DR Legs v2", 3.5, 59000.0), ("Ant hetero v2", 2.8, 86246.0)):
    sec = hours * 3600
    # 정책 스텝 1회당 시뮬 스텝 5회. 시뮬 스텝/s 상한을 벤치 처리량으로 두면 K(반복당 롤아웃 길이) 상한
    k_max = bench * sec / (iters * envs * dec)
    k24_need = iters * envs * 24 * dec / sec
    res["V6_rl_time"][name] = dict(hours=hours, bench_samples_per_s=bench,
                                    K_max_steps_per_iter_if_all_time_is_simulation=k_max,
                                    required_sim_steps_per_s_if_K24=k24_need,
                                    ratio_required_over_bench_if_K24=k24_need / bench)
res["V6_rl_time"]["v2_over_v1_speedup_same_iters"] = v1_hours / 3.5
res["V6_rl_time"]["note"] = ("v1은 반복당 환경 스텝 약 24(= 390M / (4000 x 4096)), 반복당 약 27 s. "
                              "v2는 반복당 롤아웃 길이를 적지 않았다. 24라면 DR Legs 학습이 벤치 최대치의 약 2.6배 속도로 돌아야 하므로 불가능에 가깝다. "
                              "롤아웃이 짧아졌거나 설정이 바뀌었을 가능성이 크다(미확인).")

# ---------------------------------------------------------------------------
# V7. DR Legs 과제 설정 산술
# ---------------------------------------------------------------------------
L, g = 0.265, 9.81
res["V7_task"] = dict(
    pendulum_period_s=2 * math.pi * math.sqrt(L / g),
    sim_dt_x_decimation_s=0.004 * 5,
    policy_rate_hz=1 / (0.004 * 5),
    episode_steps_10s=int(10 / (0.004 * 5)),
    swing_ref_threshold_duty=0.5 + math.asin(1e-4 / 0.05) / math.pi,  # 접촉 판정 비율 (정지 구간 포함)
    code_contact_duration_s=0.3,
    code_gait_period_s=2 * 0.3,
)
# 관측 차원: 식 (28)의 항목 vs 공개 코드 (정적 계산)
obs_eq28 = {"g_R": 3, "omega_R": 3, "cmd_xy": 2, "cmd_wz": 1, "phase_sincos": 2, "q_actuated": 12, "a_t-1": 12, "a_t-2": 12}
obs_code = {"root_rot_matrix": 9, "path_dev(2x2)": 4, "cmd": 3, "cmd_root_lin": 3, "cmd_root_ang": 3, "phase(k=2)": 4,
            "root_lin_vel": 3, "root_ang_vel": 3, "height_cmd+err": 2, "joint_q(36 = 12 act + 24 passive)": 36,
            "action(scaled)": 12, "action_prev": 12}
res["V7_obs_dim"] = dict(eq28_sum=sum(obs_eq28.values()), code_sum=sum(obs_code.values()), v1_text=94,
                         eq28_detail=obs_eq28, code_detail=obs_code)
# 보상 상한 (표 A.3): 한 스텝 최대합과 그림 A.9의 수렴값(판독 약 40)
w_track_max = 5 + 2 + 2 + 10 * 2 + 5 * 2 + 2 + 0 + 5 + 10
res["V7_reward_ceiling"] = dict(
    drlegs_max_per_step=w_track_max, fig_A9_plateau_reading=40, plateau_over_max=40 / w_track_max,
    ant_max_per_step=20, ant_plateau_reading=16.5, ant_plateau_over_max=16.5 / 20,
    note="그림 A.9의 'Mean episode reward'가 스텝당 평균(에피소드 500스텝)이어야 상한과 맞는다. 에피소드 합이면 최대 약 28,000.",
    episode_sum_max=w_track_max * 500,
)

# ---------------------------------------------------------------------------
# V8. v1 대 v2 숫자 변경 요약
# ---------------------------------------------------------------------------
res["V8_v1_v2"] = {
    "Iron Man bodies": (151, 67), "Iron Man loops": (45, 13), "Olaf bodies": (36, 27), "Olaf loops": (5, 3),
    "BDX bodies": (21, 15), "CR iteration budget": (9, 15), "PADMM iterations": ("10-30 (dt=1/240 s)", "200 (표 1), 100 (처리량)"),
    "DR Legs training": ("31 h, RTX Pro 6000", "3.5 h, RTX 5090"),
    "pages": (14, 26),
}


# ---------------------------------------------------------------------------
# V9. 표 A.2 (4절 링크, 최대 위치 수준 구속 잔차)
# ---------------------------------------------------------------------------
TAB_A2_COLS = ["Kamino soft", "Kamino stiff", "MJWarp soft", "MJWarp medium", "MJWarp stiff", "XPBD"]
TAB_A2 = {"400 Hz": [6.0e-3, 1.8e-3, 3.7e-3, 9.4e-3, 1.1e-2, 1.4e-3],
      "1000 Hz": [1.6e-3, 3.6e-4, 1.5e-3, 1.8e-3, 1.0e-4, 2.2e-4]}
link_len = 0.06
res["V9_fourbar"] = {}
for rate, v in TAB_A2.items():
    order = sorted(range(6), key=lambda i: v[i])
    res["V9_fourbar"][rate] = dict(
        best=TAB_A2_COLS[order[0]], ranking=[TAB_A2_COLS[i] for i in order],
        sub_mm=[TAB_A2_COLS[i] for i in range(6) if v[i] < 1e-3],
        relative_to_link_length_pct={TAB_A2_COLS[i]: 100 * v[i] / link_len for i in range(6)},
        kamino_stiff_over_soft=v[1] / v[0],
    )
res["V9_fourbar"]["text_claims"] = dict(
    best_400_is_XPBD=(TAB_A2_COLS[int(np.argmin(TAB_A2["400 Hz"]))] == "XPBD"),
    best_1000_is_MJWarp=(TAB_A2_COLS[int(np.argmin(TAB_A2["1000 Hz"]))].startswith("MJWarp")),
    non_monotonic_mjwarp_400="soft 3.7e-3 < medium 9.4e-3 < stiff 1.1e-2 (잔차는 단조 증가; 에너지 주입 비단조 서술은 그림 A.5 근거)",
)

OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT)
s = res["V1_table1"]
print("median:", s["median"])
print("p90:", s["p90"])
for k, v in s["by_group"].items():
    print(k, v)
print("loops v2:", {k: (d["loops_reported"], d["loops_floating_formula"], d["loops_fixed_base_formula"]) for k, d in v2.items()})
print("loops v1:", {k: (d["loops_reported"], d["loops_floating_formula"], d["loops_fixed_base_formula"]) for k, d in v1.items()})
print(res["V3_ant_hetero"])
print(res["V4_manipulation"], res["V4_seeds"])
print(res["V6_rl_time"])
print(res["V7_task"], res["V7_obs_dim"]["eq28_sum"], res["V7_obs_dim"]["code_sum"])
print(res["V5_ironman_CR_vs_LLT"])
