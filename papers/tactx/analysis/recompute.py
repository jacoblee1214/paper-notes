"""TACTX (arXiv:2606.31236) 표 수치 재계산과 검정.

입력: 논문 표 1, 표 2, 표 8과 Table 3의 값(아래 상수로 옮김).
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py  (numpy, scipy 필요, 수 초)

시행 수 가정: 표 2의 한 칸은 "10회 시행 성공 횟수의 3회 반복 평균 ± 표준편차"이다(캡션).
따라서 칸당 시행 30회, 방법당 30칸 = 900회로 두고 합산한다(독립 가정은 민감도로 따로 본다).
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
rng = np.random.default_rng(0)

TASKS = ["P&P", "P&P(OOD)", "Insertion", "Wiping", "Reorient"]
PAIRS = ["D>E", "D>F", "E>D", "E>F", "F>D", "F>E"]  # 원천 > 배치

# 표 2: 방법 -> 6방향 x 5과제 (평균, 표준편차)
T2 = {
    "Vision": [
        [(5.3, .9), (1.7, .9), (2.7, .5), (3.0, 0), (0.3, .5)],
        [(5.3, .5), (1.3, .5), (1.3, .9), (1.3, .9), (0.7, .9)],
        [(7.7, .5), (0.0, 0), (3.7, .5), (0.0, 0), (7.0, 0)],
        [(1.3, .5), (0.7, .5), (0.3, .5), (0.3, .5), (1.3, .5)],
        [(7.0, .8), (6.0, 0), (3.3, .5), (0.7, .5), (7.3, .9)],
        [(6.7, .5), (1.0, 0), (5.0, 0), (0.3, .5), (0.0, 0)],
    ],
    "Binary": [
        [(4.0, 0), (1.3, .9), (2.7, 2.1), (2.0, 2.8), (1.7, 1.7)],
        [(3.3, 1.2), (2.7, .5), (2.0, .8), (2.0, 1.4), (4.0, 1.6)],
        [(3.3, .9), (0.0, 0), (0.3, .5), (1.0, 1.4), (2.3, 2.1)],
        [(4.0, 1.6), (1.3, 1.2), (0.7, .9), (0.3, .5), (6.0, 2.2)],
        [(3.3, 1.7), (2.7, 1.7), (6.7, .5), (1.3, 1.9), (7.3, 2.4)],
        [(1.0, .8), (2.7, 1.9), (0.3, .5), (1.0, .8), (2.7, .9)],
    ],
    "TactX": [
        [(8.3, .5), (1.0, 0), (4.0, .8), (4.0, 0), (3.7, .9)],
        [(5.3, 1.2), (2.0, .8), (1.3, 1.9), (0.3, .5), (6.7, .9)],
        [(9.0, 1.4), (0.7, .5), (6.0, 1.6), (6.0, .8), (5.0, 1.4)],
        [(1.3, .9), (0.0, 0), (3.7, 1.2), (1.3, .5), (5.0, .8)],
        [(8.0, 1.4), (8.3, 1.2), (8.3, .9), (6.3, .9), (7.7, 1.2)],
        [(6.7, .9), (3.0, .8), (4.7, .5), (5.7, .5), (4.3, .5)],
    ],
}

# 표 8: 센서 -> 과제별 성공/10 (같은 센서 학습·평가). 센서 순서 D, E, F
T8 = {
    "Vision": {"D": [9, 8, 2, 4, 9], "E": [9, 6, 2, 0, 8], "F": [7, 6, 8, 9, 7]},
    "GT":     {"D": [9, 7, 7, 10, 8], "E": [10, 9, 7, 6, 10], "F": [9, 8, 8, 9, 10]},
    "TactX":  {"D": [10, 5, 5, 5, 10], "E": [10, 8, 5, 8, 9], "F": [10, 9, 8, 9, 10]},
}
# 표 1 (본문): 과제별 평균
T1 = {
    "Vision": [8.33, 6.67, 4.00, 4.33, 8.00],
    "GT":     [9.33, 8.00, 7.33, 8.33, 9.33],
    "TactX":  [10.00, 7.33, 6.00, 7.33, 9.67],
}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def triples(total):
    """합이 total인 0..10 정수 3개 조합."""
    return [t for t in itertools.combinations_with_replacement(range(11), 3) if sum(t) == total]


def feasible(mean, sd):
    """평균과 표준편차(1자리 반올림)를 만드는 정수 삼중항이 있는지: (모표준편차, 표본표준편차)."""
    total = round(mean * 3)
    if abs(total / 3 - mean) > 0.051:
        return False, False, False
    pop = smp = False
    for t in triples(total):
        a = np.array(t, float)
        if round(a.std(ddof=0), 1) == round(sd, 1):
            pop = True
        if round(a.std(ddof=1), 1) == round(sd, 1):
            smp = True
    return True, pop, smp


res = {}

# ---- V1. 표 2 정수 가능성 ----
mean_ok = pop_ok = smp_ok = 0
bad_mean, bad_pop = [], []
n_cells = 0
for m, rows in T2.items():
    for pi, row in enumerate(rows):
        for ti, (mu, sd) in enumerate(row):
            n_cells += 1
            fm, fp, fs = feasible(mu, sd)
            mean_ok += fm
            pop_ok += fp
            smp_ok += fs
            if not fm:
                bad_mean.append((m, PAIRS[pi], TASKS[ti], mu, sd))
            if fm and not fp:
                bad_pop.append((m, PAIRS[pi], TASKS[ti], mu, sd))
res["V1_integer_feasibility"] = {
    "cells": n_cells, "mean_is_k_over_3": mean_ok,
    "std_feasible_population": pop_ok, "std_feasible_sample": smp_ok,
    "mean_infeasible": bad_mean, "pop_std_infeasible": bad_pop,
}

# ---- V2. 방법별 평균 (30칸) ----
means = {m: np.array([[c[0] for c in row] for row in rows]) for m, rows in T2.items()}
sds = {m: np.array([[c[1] for c in row] for row in rows]) for m, rows in T2.items()}
avg = {m: float(means[m].mean() * 10) for m in means}  # %
avg_no_ood = {m: float(np.delete(means[m], 1, axis=1).mean() * 10) for m in means}
res["V2_method_means_pct"] = {"all30": avg, "without_ood_24": avg_no_ood,
                               "paper": {"Vision": 27.5, "TactX": 45.9}}
res["V2_pp_gain"] = avg["TactX"] - avg["Vision"]
res["V2_relative_gain_pct"] = (avg["TactX"] / avg["Vision"] - 1) * 100
res["V2_by_task"] = {m: dict(zip(TASKS, (means[m].mean(axis=0) * 10).round(2).tolist())) for m in means}
res["V2_by_pair"] = {m: dict(zip(PAIRS, (means[m].mean(axis=1) * 10).round(2).tolist())) for m in means}
res["V2_gain_by_task_pp"] = dict(zip(TASKS, ((means["TactX"] - means["Vision"]).mean(axis=0) * 10).round(2).tolist()))
res["V2_gain_by_pair_pp"] = dict(zip(PAIRS, ((means["TactX"] - means["Vision"]).mean(axis=1) * 10).round(2).tolist()))
by_dep = {}
for m in means:
    dep = {"D": [], "E": [], "F": []}
    src = {"D": [], "E": [], "F": []}
    for pi, p in enumerate(PAIRS):
        s, d = p.split(">")
        dep[d].append(means[m][pi].mean() * 10)
        src[s].append(means[m][pi].mean() * 10)
    by_dep[m] = {"deploy": {k: round(float(np.mean(v)), 2) for k, v in dep.items()},
                 "source": {k: round(float(np.mean(v)), 2) for k, v in src.items()}}
res["V2_by_source_deploy"] = by_dep

# ---- V3. 합산 시행 기반 비율 검정 (900회, 독립 가정) ----
pooled = {}
for m in means:
    k = round(float(means[m].sum() * 3))
    pooled[m] = {"successes": k, "trials": 900, "rate": k / 900, "wilson95": wilson(k, 900)}
res["V3_pooled"] = pooled
kt, kv, kb = pooled["TactX"]["successes"], pooled["Vision"]["successes"], pooled["Binary"]["successes"]
res["V3_fisher_tactx_vs_vision"] = stats.fisher_exact([[kt, 900 - kt], [kv, 900 - kv]])[1]
res["V3_fisher_tactx_vs_binary"] = stats.fisher_exact([[kt, 900 - kt], [kb, 900 - kb]])[1]
res["V3_fisher_binary_vs_vision"] = stats.fisher_exact([[kb, 900 - kb], [kv, 900 - kv]])[1]

# ---- V4. 칸 단위 쌍체 비교 (30칸 / 방향 6 / 과제 5 단위) ----
d_cells = (means["TactX"] - means["Vision"]).ravel() * 10
res["V4_cell_paired"] = {
    "n": 30, "mean_diff_pp": float(d_cells.mean()),
    "sd": float(d_cells.std(ddof=1)),
    "t_p": float(stats.ttest_1samp(d_cells, 0).pvalue),
    "wilcoxon_p": float(stats.wilcoxon(d_cells).pvalue),
    "wins": int((d_cells > 0).sum()), "ties": int((d_cells == 0).sum()), "losses": int((d_cells < 0).sum()),
    "sign_test_p": float(stats.binomtest(int((d_cells > 0).sum()), int((d_cells != 0).sum()), 0.5).pvalue),
}
boot = [rng.choice(d_cells, 30).mean() for _ in range(20000)]
res["V4_cell_paired"]["bootstrap95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
# 과제 5개 단위, 방향 6개 단위의 군집 부트스트랩(쌍체 차이를 군집 평균으로)
dm = (means["TactX"] - means["Vision"]) * 10
for name, axis, n in (("by_task_clusters", 0, 5), ("by_direction_clusters", 1, 6)):
    cl = dm.mean(axis=axis)
    b = [rng.choice(cl, n).mean() for _ in range(20000)]
    res["V4_" + name] = {"cluster_means": cl.round(2).tolist(), "boot95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                         "t_p": float(stats.ttest_1samp(cl, 0).pvalue)}
# 원천 정책 단위(원천 3 x 과제 5 = 15 정책 가정; 같은 원천 정책이 두 배치에 쓰인 경우)
dsrc = []
for s in "DEF":
    idx = [i for i, p in enumerate(PAIRS) if p.startswith(s)]
    dsrc.append(dm[idx].mean(axis=0))
dsrc = np.array(dsrc).ravel()
res["V4_source_policy_units"] = {"n": 15, "mean_diff_pp": float(dsrc.mean()),
                                  "t_p": float(stats.ttest_1samp(dsrc, 0).pvalue),
                                  "wilcoxon_p": float(stats.wilcoxon(dsrc).pvalue)}

# ---- V5. "대부분의 과제·센서쌍에서 최고" ----
best = {"TactX": 0, "Binary": 0, "Vision": 0}
tied_top = 0
tactx_best_cells = []
for pi in range(6):
    for ti in range(5):
        vals = {m: means[m][pi, ti] for m in means}
        mx = max(vals.values())
        winners = [m for m, v in vals.items() if v == mx]
        for w in winners:
            best[w] += 1
        if len(winners) > 1:
            tied_top += 1
        if "TactX" in winners:
            tactx_best_cells.append((PAIRS[pi], TASKS[ti]))
res["V5_best_cells"] = {"counts_incl_ties": best, "cells_with_tie_for_top": tied_top,
                         "tactx_best_or_tied": len(tactx_best_cells),
                         "tactx_strictly_best": sum(1 for pi in range(6) for ti in range(5)
                                                    if means["TactX"][pi, ti] > max(means["Vision"][pi, ti], means["Binary"][pi, ti]))}

# 방향별 평균에서 e>F 약함 검사
res["V5_EF_weakest"] = {m: round(float(means[m][3].mean() * 10), 2) for m in means}

# ---- V6. 표 1 vs 표 8 ----
t8_means = {m: np.array([T8[m][s] for s in "DEF"]).mean(axis=0) for m in T8}
res["V6_table1_vs_table8"] = {m: {"table1": T1[m], "from_table8": t8_means[m].round(2).tolist(),
                                    "max_abs_diff": float(np.abs(np.array(T1[m]) - t8_means[m]).max())} for m in T8}
res["V6_in_domain_avg_pct"] = {m: float(np.mean(t8_means[m]) * 10) for m in T8}
res["V6_retention_tactx"] = {t: (t8_means["TactX"][i] - t8_means["Vision"][i]) / (t8_means["GT"][i] - t8_means["Vision"][i])
                              for i, t in enumerate(TASKS)}
# 표 8의 칸별 Fisher (10회 대 10회)
cells = []
for s in "DEF":
    for i, t in enumerate(TASKS):
        a, b, c = T8["TactX"][s][i], T8["Vision"][s][i], T8["GT"][s][i]
        cells.append({"sensor": s, "task": t, "tactx": a, "vision": b, "gt": c,
                      "p_tactx_vs_vision": stats.fisher_exact([[a, 10 - a], [b, 10 - b]])[1],
                      "p_tactx_vs_gt": stats.fisher_exact([[a, 10 - a], [c, 10 - c]])[1]})
res["V6_table8_cells"] = cells
res["V6_table8_min_p_vision"] = min(c["p_tactx_vs_vision"] for c in cells)
res["V6_table8_n_p_lt_005_vision"] = sum(c["p_tactx_vs_vision"] < 0.05 for c in cells)
agg = {m: sum(sum(T8[m][s]) for s in "DEF") for m in T8}
res["V6_table8_pooled_150"] = {m: {"k": agg[m], "n": 150, "rate": agg[m] / 150, "wilson95": wilson(agg[m], 150)} for m in agg}
res["V6_table8_pooled_fisher_tactx_vs_vision"] = stats.fisher_exact([[agg["TactX"], 150 - agg["TactX"]], [agg["Vision"], 150 - agg["Vision"]]])[1]
res["V6_table8_pooled_fisher_tactx_vs_gt"] = stats.fisher_exact([[agg["TactX"], 150 - agg["TactX"]], [agg["GT"], 150 - agg["GT"]]])[1]
# 센서별 비전 전용 편차
res["V6_vision_only_by_sensor"] = {s: float(np.mean(T8["Vision"][s]) * 10) for s in "DEF"}
res["V6_tactx_by_sensor"] = {s: float(np.mean(T8["TactX"][s]) * 10) for s in "DEF"}
res["V6_gt_by_sensor"] = {s: float(np.mean(T8["GT"][s]) * 10) for s in "DEF"}

# ---- V7. 비전 전용: 동일 센서 대 교차 센서 (촉각 입력이 없는 정책) ----
res["V7_vision_same_vs_cross_pct"] = {"same_sensor_table8": res["V6_in_domain_avg_pct"]["Vision"], "cross_table2": avg["Vision"],
                                       "drop_pp": res["V6_in_domain_avg_pct"]["Vision"] - avg["Vision"]}
res["V7_tactx_same_vs_cross_pct"] = {"same_sensor_table8": res["V6_in_domain_avg_pct"]["TactX"], "cross_table2": avg["TactX"],
                                      "drop_pp": res["V6_in_domain_avg_pct"]["TactX"] - avg["TactX"]}
res["V7_tactx_cross_vs_vision_same"] = avg["TactX"] - res["V6_in_domain_avg_pct"]["Vision"]
# 같은 센서 대 교차 센서, 과제별 비전 전용
res["V7_vision_by_task_same_vs_cross"] = {t: {"same": float(t8_means["Vision"][i] * 10), "cross": float(means["Vision"][:, i].mean() * 10)}
                                           for i, t in enumerate(TASKS)}

# ---- V8. 3회 반복 표준편차와 독립 이항 시행 가정의 기대값 ----
# 칸마다 성공확률 p = 평균/10, 3회 반복이 각각 Binomial(10, p)의 독립 시행이라고 가정하고 모표준편차의 기대값을 모의한다.
sim = np.random.default_rng(1)
exp_sd, obs_sd, exp_zero = [], [], []
for m in means:
    for mu, sd in zip(means[m].ravel(), sds[m].ravel()):
        p = mu / 10
        draws = sim.binomial(10, p, size=(4000, 3))
        s_ = draws.std(axis=1)
        exp_sd.append(s_.mean())
        exp_zero.append((s_ == 0).mean())
        obs_sd.append(sd)
res["V8_sd"] = {"mean_observed_pop_sd": float(np.mean(obs_sd)), "mean_expected_pop_sd_iid_binomial": float(np.mean(exp_sd)),
                 "observed_zero_sd_cells": int((np.array(obs_sd) == 0).sum()), "expected_zero_sd_cells": float(np.sum(exp_zero)),
                 "note": "칸 평균을 성공확률로 두고 독립 이항 3회를 모의한 값. 관측과 가까우면 3회 반복의 변동이 독립 시행 수준"}

# ---- V9. 부록 수치 일관성 ----
res["V9_table3"] = {"configs": 6, "traj_per_config": [442, 448], "traj_total_reported": 2670,
                     "range_total": [6 * 442, 6 * 448], "frames_per_traj": 145000 / 2670,
                     "frames_per_config_range": [20000, 31000], "frames_6config_range": [6 * 20000, 6 * 31000]}
hf = {"daimon": [(43, 7), (44, 6), (33, 3), (41, 3)], "eflesh": [(43, 7), (43, 7), (30, 4), (44, 3)],
      "flexitac": [(43, 7), (43, 7), (30, 3), (45, 5)]}
tot_tr = sum(a for v in hf.values() for a, b in v)
tot_va = sum(b for v in hf.values() for a, b in v)
res["V9_hf_dataset"] = {"train": tot_tr, "val": tot_va, "total": tot_tr + tot_va,
                          "per_sensor_total": {k: sum(a + b for a, b in v) for k, v in hf.items()},
                          "per_cell_total_min_max": [min(a + b for v in hf.values() for a, b in v), max(a + b for v in hf.values() for a, b in v)],
                          "mean_per_cell": (tot_tr + tot_va) / 12,
                          "paper_says": "~50 episodes per task"}
# 공식 설정의 객체별 val 에피소드 수 (config/datasets_config.yaml)
val_ids = {"circle": 18, "cylinder": 17, "cylinder_vertical": 14, "ellipse": 19, "large_ball": 16, "plane": 21,
           "plane_horizontal": 18, "small_ball": 30, "square": 8, "triangle": 16}
res["V9_val_ids"] = {"sum": sum(val_ids.values()), "if_20pct_total_episodes": sum(val_ids.values()) / 0.2,
                      "times3_pairs_vs_2670": sum(val_ids.values()) / 0.2 * 3}

# ---- V10. 센서 예측 확률, 객체 분류 확률 기준선 ----
res["V10_chance"] = {"sensor_chance": 1 / 3, "object_chance": 0.1, "tactx_sensor_acc": 0.475,
                      "tactx_obj_self": 0.608, "tactx_obj_cross": 0.562,
                      "note": "그림 4 막대는 그림 판독 값: recon 약 67.5, contr 약 42, L2 약 57.5, TactX 47.5 (본문은 67.5와 47.5만 명시)"}
# 평균 상대 증가 표현
res["V10_intro_20pct"] = {"pp": res["V2_pp_gain"], "relative_pct": res["V2_relative_gain_pct"]}

OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")

# ---- 요약 출력 ----
print("V1 가능성:", res["V1_integer_feasibility"]["cells"], "평균 k/3",
      res["V1_integer_feasibility"]["mean_is_k_over_3"], "모SD", res["V1_integer_feasibility"]["std_feasible_population"],
      "표본SD", res["V1_integer_feasibility"]["std_feasible_sample"])
print("  모SD 불가:", res["V1_integer_feasibility"]["pop_std_infeasible"])
print("V2 평균:", {k: round(v, 2) for k, v in avg.items()}, "OOD 제외", {k: round(v, 2) for k, v in avg_no_ood.items()})
print("  이득 pp:", round(res["V2_pp_gain"], 2), "상대 %", round(res["V2_relative_gain_pct"], 1))
print("  과제별:", res["V2_by_task"])
print("  과제별 이득:", res["V2_gain_by_task_pp"])
print("  방향별:", res["V2_by_pair"])
print("  방향별 이득:", res["V2_gain_by_pair_pp"])
print("  원천/배치:", res["V2_by_source_deploy"])
print("V3 합산:", {m: (v["successes"], round(v["rate"], 3), [round(x, 3) for x in v["wilson95"]]) for m, v in pooled.items()})
print("  Fisher TactX/Vision", res["V3_fisher_tactx_vs_vision"], "TactX/Binary", res["V3_fisher_tactx_vs_binary"], "Binary/Vision", res["V3_fisher_binary_vs_vision"])
print("V4 칸 쌍체:", res["V4_cell_paired"])
print("  과제 군집:", res["V4_by_task_clusters"])
print("  방향 군집:", res["V4_by_direction_clusters"])
print("  원천 정책 단위:", res["V4_source_policy_units"])
print("V5 최고 칸:", res["V5_best_cells"], "E>F:", res["V5_EF_weakest"])
print("V6 표1-표8 최대 차이:", {m: round(v["max_abs_diff"], 3) for m, v in res["V6_table1_vs_table8"].items()})
print("  표 8 평균:", {m: round(v, 1) for m, v in res["V6_in_domain_avg_pct"].items()})
print("  유지율:", {k: round(v, 2) for k, v in res["V6_retention_tactx"].items()})
print("  표 8 칸별 최소 p(대 Vision):", round(res["V6_table8_min_p_vision"], 3), "p<0.05 칸 수:", res["V6_table8_n_p_lt_005_vision"])
print("  표 8 합산:", {m: (v["k"], round(v["rate"], 3), [round(x, 3) for x in v["wilson95"]]) for m, v in res["V6_table8_pooled_150"].items()})
print("  합산 Fisher TactX/Vision", res["V6_table8_pooled_fisher_tactx_vs_vision"], "TactX/GT", res["V6_table8_pooled_fisher_tactx_vs_gt"])
print("  센서별 Vision", res["V6_vision_only_by_sensor"], "TactX", res["V6_tactx_by_sensor"], "GT", res["V6_gt_by_sensor"])
print("V7:", res["V7_vision_same_vs_cross_pct"], res["V7_tactx_same_vs_cross_pct"], "TactX cross - Vision same", round(res["V7_tactx_cross_vs_vision_same"], 2))
print("  비전 과제별 동일/교차:", res["V7_vision_by_task_same_vs_cross"])
print("V8 SD:", res["V8_sd"])
print("V9:", res["V9_table3"], res["V9_hf_dataset"], res["V9_val_ids"])
