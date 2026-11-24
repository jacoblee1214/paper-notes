"""OpenHLM (arXiv:2606.22174 v2) 표와 그림에 적힌 수치를 다시 계산한다.

실행: python3 analysis/recompute.py   → analysis/recompute_results.json (수 초, CPU)

원칙
- 논문은 (정책, 과제)마다 실로봇 롤아웃을 5회 하고(부록 A.2), 롤아웃 점수는 루브릭 점수/만점이다.
  따라서 5회 평균은 k/(5P)의 값만 가질 수 있다(P=루브릭 만점). 이 정수 가능성을 먼저 검사한다.
- 그림 4, 5, 12, 3의 막대 값은 그림 판독 값이다(막대 위 숫자). 논문이 표로 준 값이 아니다.
- 표준오차(SE)만 있는 칸은 "그 평균과 SE를 낼 수 있는 정수 점수 조합"을 모두 찾아, 검정 결과의 범위로 보고한다.
  롤아웃별 점수는 논문에 없으므로 이 범위가 우리가 말할 수 있는 한계다.
- 롤아웃 점수가 독립이라는 가정, 루브릭 점수를 이항 시행처럼 쓰는 Fisher 검사는 근사다(점수는 한 롤아웃 안에서 누적이라 실제 유효 표본은 더 작다).
"""
import json
import math
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}


# ---------------------------------------------------------------- 공통 도구
def feasible_vectors(n, pmax, mean_pct, se_pct, tol=0.051):
    """n회 롤아웃, 만점 P인 정수 점수 조합 중 (평균 %, SE %)가 표기 반올림과 맞는 것."""
    out = []
    for combo in combinations_with_replacement(range(pmax + 1), n):
        x = np.array(combo) / pmax * 100.0
        m = x.mean()
        if abs(m - mean_pct) > tol:
            continue
        se = x.std(ddof=1) / math.sqrt(n)
        if abs(se - se_pct) <= tol:
            out.append(combo)
    return out


def welch_p(a, b, pmax_a, pmax_b):
    xa = np.array(a) / pmax_a * 100.0
    xb = np.array(b) / pmax_b * 100.0
    if xa.std(ddof=1) == 0 and xb.std(ddof=1) == 0:
        return 0.0 if xa.mean() != xb.mean() else 1.0
    return float(stats.ttest_ind(xa, xb, equal_var=False).pvalue)


def perm_p(a, b, pmax_a, pmax_b, n_mc=4000, seed=0):
    xa = np.array(a) / pmax_a * 100.0
    xb = np.array(b) / pmax_b * 100.0
    obs = abs(xa.mean() - xb.mean())
    pool = np.concatenate([xa, xb])
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_mc):
        rng.shuffle(pool)
        if abs(pool[: len(xa)].mean() - pool[len(xa):].mean()) >= obs - 1e-9:
            cnt += 1
    return (cnt + 1) / (n_mc + 1)


def fisher_points(k1, n1, k2, n2):
    """루브릭 점수를 이항 시행으로 본 Fisher 정확 검정 (근사: 점수는 롤아웃 안에서 누적)."""
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def pair_stats(F1, F2, pmax1, pmax2):
    ps = [(welch_p(a, b, pmax1, pmax2)) for a in F1 for b in F2]
    res = {"n_feasible": [len(F1), len(F2)], "welch_p_min": min(ps), "welch_p_max": max(ps)}
    if len(F1) * len(F2) <= 400:
        pp = [perm_p(a, b, pmax1, pmax2) for a in F1 for b in F2]
        res["perm_p_min"] = min(pp)
        res["perm_p_max"] = max(pp)
    return res


# ---------------------------------------------------------------- V1 표 1
TASK_P = {1: 3, 2: 3, 3: 4, 4: 3, 5: 5, 6: 4, 7: 4, 8: 3, 9: 3, 10: 3, 11: 3, 12: 4}
T1 = [  # (과제, P, 방법, 평균%, SE%, 시간 s, 걸음)
    ("Cola", 3, "decoupled", 66.7, 14.9, 36.7, 42.3),
    ("Cola", 3, "vr3pt", 40.0, 6.7, 67.8, 12.3),
    ("Cola", 3, "joint", 86.7, 8.2, 40.2, 12.0),
    ("ShelfCup", 3, "decoupled", 93.3, 6.7, 33.6, 38.4),
    ("ShelfCup", 3, "joint", 80.0, 13.3, 41.8, 10.5),
    ("Bottle", 4, "joint", 85.0, 6.1, 29.7, 10.3),
]
v1 = {"cells": [], "pairs": {}}
feas1 = {}
for task, pmax, meth, m, se, t, s in T1:
    F = feasible_vectors(5, pmax, m, se)
    feas1[(task, meth)] = (F, pmax)
    k_exact = m / 100 * 5 * pmax
    v1["cells"].append({
        "task": task, "method": meth, "mean": m, "se": se, "P": pmax,
        "implied_points": round(k_exact, 2),
        "integer_ok": abs(k_exact - round(k_exact)) < 0.06,
        "n_feasible_score_vectors": len(F),
        "example_vectors": [list(F[0])] if F else [],
    })
v1["pairs"]["Cola joint vs decoupled"] = pair_stats(*[feas1[("Cola", "joint")][0], feas1[("Cola", "decoupled")][0]], 3, 3)
v1["pairs"]["Cola joint vs vr3pt"] = pair_stats(*[feas1[("Cola", "joint")][0], feas1[("Cola", "vr3pt")][0]], 3, 3)
v1["pairs"]["ShelfCup joint vs decoupled"] = pair_stats(*[feas1[("ShelfCup", "joint")][0], feas1[("ShelfCup", "decoupled")][0]], 3, 3)
v1["footstep_ratio_cola"] = round(42.3 / 12.0, 3)
v1["fisher_points"] = {
    "Cola joint 13/15 vs decoupled 10/15": fisher_points(13, 15, 10, 15),
    "ShelfCup decoupled 14/15 vs joint 12/15": fisher_points(14, 15, 12, 15),
}
# 롤아웃 점수의 표준편차 추정(Table 1 SE → SD): 최소 검출 효과 계산용
sd1 = [se * math.sqrt(5) for (_, _, _, _, se, _, _) in T1]
RES["V1_table1"] = v1

# ---------------------------------------------------------------- V2 표 2
imu = {"Cola": {"joint": 0.187, "vr3pt": 0.235, "decoupled": 0.439},
       "ShelfCup": {"joint": 0.178, "decoupled": 0.424}}
RES["V2_table2_imu_ratio"] = {
    "Cola decoupled/joint": round(0.439 / 0.187, 2), "Cola vr3pt/joint": round(0.235 / 0.187, 2),
    "ShelfCup decoupled/joint": round(0.424 / 0.178, 2),
}

# ---------------------------------------------------------------- V3 그림 3 (판독) 과 표 1 대조
fig3 = {"joint": {"Cola": 87, "ShelfCup": 80, "Bottle": 85, "Jar": 100, "avg": 88},
        "smpl": {"Cola": 73, "ShelfCup": 87, "Bottle": 55, "Jar": 87, "avg": 75}}
MAXPTS4 = {"Cola": 3, "ShelfCup": 3, "Bottle": 4, "Jar": 3}


def exact_pct(task, pct, n=5):
    pmax = MAXPTS4[task]
    k = round(pct / 100 * n * pmax)
    return k, k / (n * pmax) * 100


v3 = {}
for arm, d in fig3.items():
    ex = [exact_pct(t, d[t])[1] for t in MAXPTS4]
    v3[arm] = {"mean_of_tasks_exact": round(float(np.mean(ex)), 2), "printed_avg": d["avg"],
               "tasks_integer_ok": all(abs(d[t] - exact_pct(t, d[t])[1]) < 0.51 for t in MAXPTS4)}
v3["joint_vs_table1"] = {"Cola": [fig3["joint"]["Cola"], 86.7], "ShelfCup": [fig3["joint"]["ShelfCup"], 80.0],
                         "Bottle": [fig3["joint"]["Bottle"], 85.0]}
# 4과제 평균 차이의 근사 z (그림 판독 오차 막대: joint 약 ±4, SMPL 약 ±5, 그림 판독 값)
v3["approx_z_avg_gap"] = round((88 - 75) / math.sqrt(4 ** 2 + 5 ** 2), 2)
v3["approx_p_two_sided"] = float(2 * (1 - stats.norm.cdf(v3["approx_z_avg_gap"])))
RES["V3_fig3_joint_vs_smpl"] = v3

# ---------------------------------------------------------------- V4 그림 4 (판독) 과 평균 재계산
ORDER = ["default", "randinit_proj", "humanoid_order", "relative_action", "no_proprio",
         "paligemma_init", "from_scratch", "onestep_fm", "onestep_drift"]
F4 = {  # 과제 Cola, ShelfCup, Bottle, Jar
    "default": [100, 93, 85, 87], "randinit_proj": [67, 100, 75, 87], "humanoid_order": [80, 80, 95, 93],
    "relative_action": [80, 80, 85, 100], "no_proprio": [87, 67, 85, 87], "paligemma_init": [40, 53, 65, 80],
    "from_scratch": [33, 47, 60, 27], "onestep_fm": [80, 53, 65, 93], "onestep_drift": [67, 67, 70, 80]}
AVG4 = {"default": 91, "randinit_proj": 82, "humanoid_order": 87, "relative_action": 86, "no_proprio": 81,
        "paligemma_init": 60, "from_scratch": 42, "onestep_fm": 73, "onestep_drift": 71}
v4 = {"rows": {}}
tasks4 = ["Cola", "ShelfCup", "Bottle", "Jar"]
for k in ORDER:
    ex = [exact_pct(t, v)[1] for t, v in zip(tasks4, F4[k])]
    pts = [exact_pct(t, v)[0] for t, v in zip(tasks4, F4[k])]
    v4["rows"][k] = {"task_pct": F4[k], "implied_points": pts, "avg_recomputed": round(float(np.mean(ex)), 2),
                     "avg_printed": AVG4[k], "match": abs(float(np.mean(ex)) - AVG4[k]) <= 0.5 + 1e-9,
                     "gap_vs_default": round(float(np.mean(ex)) - float(np.mean([exact_pct(t, v)[1] for t, v in zip(tasks4, F4["default"])])), 2)}
# 같은 구성("default")의 재측정 차이: 그림 3 joint (87,80,85,100) 대 그림 4 default (100,93,85,87)
j3 = [87, 80, 85, 100]
d4 = F4["default"]
v4["rerun_gap_joint_fig3_vs_default_fig4"] = {
    "per_task": [b - a for a, b in zip(j3, d4)],
    "mean_abs": round(float(np.mean(np.abs(np.array(d4) - np.array(j3)))), 2),
    "avg": [88, 91]}
# 그림 4 default (100,93,85,87) 와 그림 12 "8-task baseline" 과제 1-4 (100,93,85,87)
v4["fig4_default_equals_fig12_8task_baseline_tasks1to4"] = (d4 == [100, 93, 85, 87])
# 1단계 생성이 "약 20점 낮다" 주장
v4["onestep_gap_vs_default"] = {"fm": round(91.25 - 72.75, 2), "drift": round(91.25 - 71.0, 2)}
# 추론 지연 이득이 제어 주기에서 차지하는 비중
v4["latency_budget"] = {
    "chunk_exec_period_s": round(25 / 30, 4),
    "saved_ms": 90 - 60,
    "saved_fraction_of_period": round(0.030 / (25 / 30), 4),
    "control_tick_ms": round(1000 / 30, 1)}
# 인터페이스 절제 4개(16칸)에서 기본값 이상인 칸 수
iface = ["randinit_proj", "humanoid_order", "relative_action", "no_proprio"]
v4["interface_cells_at_or_above_default"] = {
    "count": sum(1 for k in iface for a, b in zip(F4[k], F4["default"]) if a >= b), "of": 16,
    "strictly_above": sum(1 for k in iface for a, b in zip(F4[k], F4["default"]) if a > b)}
RES["V4_fig4_ablation"] = v4

# ---------------------------------------------------------------- V5 그림 12 / 5 (판독)
F12 = {  # 과제 번호 → [8-task baseline, stationary, HuMI, 12-task oracle]  (그림 12 막대 위 숫자)
    1: [100, 93, 87, 87], 2: [93, 87, 80, 93], 3: [85, 100, 95, 85], 4: [87, 100, 100, 100],
    5: [92, 92, 72, 88], 6: [85, 70, 95, 95], 7: [90, 75, 85, 80], 8: [80, 80, 80, 67],
    9: [20, 100, 80, 100], 10: [47, 93, 100, 100], 11: [40, 73, 73, 87], 12: [25, 80, 15, 90]}
AGG = {"1-8": [89, 87, 87, 87], "9-11": [36, 89, 84, 96], "9-12": [33, 87, 67, 94]}
COND = ["baseline8", "stationary", "humi", "oracle12"]
v5 = {"points": {}, "agg": {}, "integer_ok": True}
pts12 = {}
for t, vals in F12.items():
    pmax = TASK_P[t]
    pts12[t] = []
    for c, v in zip(COND, vals):
        k = round(v / 100 * 5 * pmax)
        pts12[t].append(k)
        if abs(v - k / (5 * pmax) * 100) > 0.51:
            v5["integer_ok"] = False
v5["points"] = pts12
groups = {"1-8": range(1, 9), "9-11": range(9, 12), "9-12": range(9, 13)}
for g, rng_ in groups.items():
    rec = []
    for ci in range(4):
        rec.append(round(float(np.mean([pts12[t][ci] / (5 * TASK_P[t]) * 100 for t in rng_])), 2))
    pooled = []
    for ci in range(4):
        pooled.append(round(100 * sum(pts12[t][ci] for t in rng_) / sum(5 * TASK_P[t] for t in rng_), 2))
    v5["agg"][g] = {"printed": AGG[g], "mean_of_task_pct": rec, "pooled_points_pct": pooled,
                    "match_mean_of_task_pct": all(abs(a - b) <= 0.51 for a, b in zip(rec, AGG[g]))}
# Fig 5와 Fig 12 의 9-12 값 일치
v5["fig5_vs_fig12_heldout_equal"] = True
# 9-11 총 점수(45점 만점)
v5["tasks9_11_points_of_45"] = {c: sum(pts12[t][i] for t in (9, 10, 11)) for i, c in enumerate(COND)}
# HuMI 40개 = 그림 13 의 84, 그림 13 증가분
fig13 = {5: 42, 10: 67, 20: 76, 40: 84}
v5["fig13"] = {"points_of_45": {k: round(v / 100 * 45) for k, v in fig13.items()},
               "integer_ok": all(abs(v - round(v / 100 * 45) / 45 * 100) < 0.51 for v in fig13.values()),
               "increments": [fig13[10] - fig13[5], fig13[20] - fig13[10], fig13[40] - fig13[20]],
               "text_increments_claimed": [25, 9, 8]}
base_pts = v5["tasks9_11_points_of_45"]["baseline8"]
v5["fig13_vs_baseline_fisher"] = {
    "humi5 19/45 vs baseline 16/45": fisher_points(19, 45, base_pts, 45),
    "humi10 30/45 vs baseline 16/45": fisher_points(30, 45, base_pts, 45),
    "humi40 38/45 vs baseline 16/45": fisher_points(38, 45, base_pts, 45)}
# Pouring (4점, 5롤아웃 = 20점)
pour = pts12[12]
v5["pouring_fisher_points_of_20"] = {
    "humi 3 vs stationary 16": fisher_points(pour[2], 20, pour[1], 20),
    "humi 3 vs baseline 5": fisher_points(pour[2], 20, pour[0], 20),
    "stationary 16 vs oracle 18": fisher_points(pour[1], 20, pour[3], 20)}
# 9-11 에서 HuMI 대 stationary (총 38/45 대 40/45)
v5["t9_11_humi_vs_stationary_fisher_points_of_45"] = fisher_points(38, 45, 40, 45)
# "8-task baseline" 의 1-8 대 stationary, HuMI 평균
v5["tasks1_8_pooled_points_of_165"] = {c: sum(pts12[t][i] for t in range(1, 9)) for i, c in enumerate(COND)}
RES["V5_fig5_fig12_cotrain"] = v5

# ---------------------------------------------------------------- V6 표 3 (10회, 8점)
T3 = {"psi0": (48.8, 4.4), "gr00t": (57.5, 4.6), "humi": (87.5, 3.7), "oracle": (97.5, 1.7)}
v6 = {"cells": {}, "pairs": {}}
F3 = {}
for k, (m, se) in T3.items():
    F = feasible_vectors(10, 8, m, se)
    F3[k] = F
    pts = m / 100 * 80
    v6["cells"][k] = {"mean": m, "se": se, "implied_points_of_80": round(pts, 2),
                      "integer_ok": abs(pts - round(pts)) < 0.05, "n_feasible_score_vectors": len(F),
                      "example": list(F[0]) if F else None}
for a, b in [("humi", "gr00t"), ("humi", "psi0"), ("humi", "oracle"), ("gr00t", "psi0"), ("oracle", "gr00t")]:
    v6["pairs"][f"{a} vs {b}"] = pair_stats(F3[a], F3[b], 8, 8)
v6["fisher_points_of_80"] = {
    "humi 70 vs gr00t 46": fisher_points(70, 80, 46, 80), "humi 70 vs psi0 39": fisher_points(70, 80, 39, 80),
    "humi 70 vs oracle 78": fisher_points(70, 80, 78, 80)}
v6["wilson_points_95"] = {k: wilson(round(m / 100 * 80), 80) for k, (m, _) in T3.items()}
# 평가 10쌍 중 7쌍이 레몬·토마토 포함 (표 5). HuMI 총 70/80 에서 하한 계산
v6["heldout_subset_bound"] = {
    "total_points": 70, "max_points_from_3_trained_pairs": 24,
    "min_points_from_7_heldout": 46, "min_pct_heldout_pairs": round(46 / 56 * 100, 1)}
# 표 5 구성 검사
T5 = [("banana", "peach"), ("peach", "mangosteen"), ("mangosteen", "banana"), ("lemon", "peach"), ("tomato", "banana"),
      ("banana", "lemon"), ("peach", "tomato"), ("mangosteen", "lemon"), ("lemon", "tomato"), ("tomato", "mangosteen")]
from collections import Counter
f1 = Counter(a for a, _ in T5)
f2 = Counter(b for _, b in T5)
trained = [p for p in T5 if not ({"lemon", "tomato"} & set(p))]
v6["table5"] = {"fruit1_counts": dict(f1), "fruit2_counts": dict(f2), "each_twice": set(f1.values()) == {2} and set(f2.values()) == {2},
                "pairs_without_lemon_tomato": len(trained), "pairs_with_lemon_tomato": len(T5) - len(trained),
                "pairs_total_P(5,2)": 5 * 4, "P(3,2)": 3 * 2, "demos_train_pairs": 6 * 6, "demos_humi_pairs": 14 * 6,
                "demos_all": 20 * 6}
# 최소 검출 효과 (80% 검정력, α=0.05, 양측, 정규근사): 롤아웃 점수 SD 와 n 으로
sd3 = [se * math.sqrt(10) for (_, se) in T3.values()]
sd_p3 = math.sqrt(np.mean(np.square(sd3)))
sd_p1 = math.sqrt(np.mean(np.square(sd1)))
z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
v6["mde"] = {"sd_rollout_table3_pooled_pct": round(float(sd_p3), 1), "mde_n10_pct": round(float(z * sd_p3 * math.sqrt(2 / 10)), 1),
             "sd_rollout_table1_pooled_pct": round(float(sd_p1), 1), "mde_n5_pct": round(float(z * sd_p1 * math.sqrt(2 / 5)), 1),
             "mde_n20_pct_same_sd_table1": round(float(z * sd_p1 * math.sqrt(2 / 20)), 1)}
RES["V6_table3_long_horizon"] = v6

# ---------------------------------------------------------------- V7 시간·비용 산술
demo_h = {"psi0": 2.70, "gr00t": 2.70, "humi": 1.14, "oracle": 2.73}
sec_per_full = 2.70 * 3600 / 120
v7 = {"sec_per_full_demo_gr00t": round(sec_per_full, 1), "sec_per_full_demo_oracle": round(2.73 * 3600 / 120, 1),
      "ratio_humi_over_baseline": round(1.14 / 2.70, 3),
      "full_part_h_if_same_rate": round(36 * sec_per_full / 3600, 3)}
v7["humi_remainder_h"] = round(1.14 - v7["full_part_h_if_same_rate"], 3)
v7["sec_per_humi_demo_implied"] = round(v7["humi_remainder_h"] * 3600 / 84, 1)
# 3.3절: 보류 과제 40회 기준 stationary 13분, full 21분, HuMI 7분
v7["sec33"] = {"full_sec_per_demo": round(21 * 60 / 40, 1), "stationary_sec_per_demo": round(13 * 60 / 40, 1),
               "humi_sec_per_demo": round(7 * 60 / 40, 1), "humi_over_stationary": round(7 / 13, 3),
               "stationary_over_full": round(13 / 21, 3), "humi_over_full": round(7 / 21, 3)}
v7["humi_84_demos_if_10p5s_h"] = round(84 * 10.5 / 3600, 3)
v7["humi_total_if_10p5s_h"] = round(v7["full_part_h_if_same_rate"] + 84 * 10.5 / 3600, 3)
# 학습량: 스텝×배치
v7["train_samples"] = {"openhlm": 30000 * 128, "gr00t": 50000 * 32, "psi0": 50000 * 64}
# 프레임 수 근사 (30 Hz 가정): 긴 과제 데이터 120 데모
frames_120 = 120 * sec_per_full * 30
v7["train_epochs_approx_30hz"] = {k: round(v / frames_120, 1) for k, v in v7["train_samples"].items()}
v7["note_epochs"] = "30 Hz 기록, 데모당 81초 가정(표 3 시간 / 120). 기준선의 샘플링 주기는 다를 수 있다."
v7["teleop_budget_text"] = {"E2_hours_per_task_40demos": 1.5, "pure_demo_min_40demos_heldout_full": 21}
RES["V7_cost_arithmetic"] = v7

# ---------------------------------------------------------------- V8 부록 E.1 / E.2
E1 = {0.0: (60, 34.3), 0.2: (67, 35.2), 0.4: (60, 35.4), 0.6: (13, 41.0)}
e1 = {"points_of_15": {str(k): round(v[0] / 100 * 15) for k, v in E1.items()},
      "integer_ok": all(abs(v[0] - round(v[0] / 100 * 15) / 15 * 100) < 0.51 for v in E1.values()),
      "gap_0p2_vs_0_points": round(0.67 * 15) - round(0.60 * 15),
      "fisher_0p2_10of15_vs_0_9of15": fisher_points(10, 15, 9, 15),
      "fisher_0p6_2of15_vs_0p2_10of15": fisher_points(2, 15, 10, 15),
      "single_vs_multi_task_10of15_vs_13of15": fisher_points(10, 15, 13, 15),
      "duration_increase_0p6_vs_0_s": round(41.0 - 34.3, 1)}
E2 = {5: 52, 10: 60, 20: 85, 40: 91}
e2 = {"increments": [E2[10] - E2[5], E2[20] - E2[10], E2[40] - E2[20]],
      "text_claim": "largest jump between 10 and 20; roughly 90% at 40",
      "claim_ok": (E2[20] - E2[10]) == max(E2[10] - E2[5], E2[20] - E2[10], E2[40] - E2[20]) and abs(E2[40] - 90) <= 1,
      "e2_40demo_equals_fig4_default_avg": E2[40] == 91,
      "note": "4과제 평균(과제별 %의 평균)이라 점수 정수 검사와 Fisher 검정은 하지 않았다."}
RES["V8_appendix_E"] = {"E1_latency": e1, "E2_scaling": e2}

# ---------------------------------------------------------------- V9 차원·산술 검사
dims = {
    "decoupled_21": 3 + 7 + 7 + 1 + 3, "vr3pt_24": 3 + 9 + 12, "joint_32": 3 + 14 + 12 + 3 + 0,
    "smpl_81": 3 + 6 + 72, "action_34": 32 + 2, "layout_34": 7 + 1 + 7 + 1 + 6 + 6 + 3 + 3,
    "gr00t_decoupled_vs_21": "14 arm + 3 waist + 1 height + 3 nav = 21 (+14 hand block)",
    "psi0_36": 14 + 14 + 3 + 1 + 4, "inference_every_s": round(25 / 30, 4), "chunk_len_s": round(50 / 30, 3),
    "p52_ordered_pairs": 5 * 4}
RES["V9_dimension_checks"] = dims
# 가중치 수술이 보존하는 사전학습 차원
RES["V9_dimension_checks"]["weight_surgery_note"] = {
    "pretrained_slots": 32, "meaningful_pretrained_slots_if_bimanual_16": 16,
    "slots_copied_but_zero_padded_in_pretraining": 16,
    "new_slots": 2}

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=lambda o: list(o) if isinstance(o, (set, tuple)) else str(o)), encoding="utf-8")
print("wrote", OUT)
