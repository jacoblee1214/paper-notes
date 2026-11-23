"""TeleDexter (arXiv:2607.11481, v1) 표 수치 재계산과 검정.

입력은 논문 Table 1-6, 부록 C의 수치, Fig. 6의 단계별 성공 시행 수(PDF 벡터 경로에서 좌표를 읽어 정수로 복원),
Fig. 5의 곡선 끝값(벡터 경로, 판독 오차 약 ±0.02 단위)이다.
시행 수는 논문 4.2절의 "과제·방법당 15회"를 쓴다. 시행 단위로는 독립이라고 가정한다(조작자 수·시행 순서는 논문에 없음).

실행: python3 analysis/recompute.py   → analysis/recompute_results.json   (scipy 필요, 수 초)
"""
import json
import math
from pathlib import Path

from scipy.stats import fisher_exact

OUT = Path(__file__).with_name("recompute_results.json")
N = 15
TASKS = ["CylinderReorient", "CuboidReorient", "BunnyReorient", "HammerUse", "BrushSweep", "ScrewdriverUse", "BulbReplace"]
STAGES = {"CylinderReorient": 3, "CuboidReorient": 3, "BunnyReorient": 3, "HammerUse": 7, "BrushSweep": 7,
          "ScrewdriverUse": 5, "BulbReplace": 6}

# Table 1 (SharpaWave): 방법 -> 과제 -> (SR %, TP %)
T1 = {
    "DexRT": {"CylinderReorient": (6.7, 37.8), "CuboidReorient": (26.7, 51.1), "BunnyReorient": (0.0, 35.6),
              "HammerUse": (0.0, 26.7), "BrushSweep": (0.0, 39.0), "ScrewdriverUse": (6.7, 37.3), "BulbReplace": (0.0, 35.6)},
    "GeoRT": {"CylinderReorient": (0.0, 24.4), "CuboidReorient": (0.0, 33.3), "BunnyReorient": (0.0, 31.1),
              "HammerUse": (0.0, 30.5), "BrushSweep": (0.0, 29.5), "ScrewdriverUse": (0.0, 25.3), "BulbReplace": (0.0, 25.6)},
    "DexGen": {"CylinderReorient": (0.0, 31.1), "CuboidReorient": (0.0, 28.9), "BunnyReorient": (0.0, 26.7),
               "HammerUse": (0.0, 26.7), "BrushSweep": (0.0, 8.6), "ScrewdriverUse": (0.0, 33.3), "BulbReplace": (0.0, 20.0)},
    "SimToolReal_cat": {"HammerUse": (0.0, 27.6), "BrushSweep": (26.7, 41.9), "ScrewdriverUse": (0.0, 17.3)},
    "SimToolReal_all": {"HammerUse": (20.0, 36.2), "BrushSweep": (0.0, 5.7), "ScrewdriverUse": (0.0, 20.0)},
    "Ours": {"CylinderReorient": (80.0, 86.7), "CuboidReorient": (80.0, 86.7), "BunnyReorient": (66.7, 77.8),
             "HammerUse": (66.7, 86.7), "BrushSweep": (73.3, 89.5), "ScrewdriverUse": (73.3, 86.7), "BulbReplace": (86.7, 95.6)},
}
T1_AVG = {"DexRT": (5.7, 37.6), "GeoRT": (0.0, 28.5), "DexGen": (0.0, 25.0), "SimToolReal_cat": (8.9, 28.9),
          "SimToolReal_all": (6.7, 20.6), "Ours": (75.2, 87.1)}
T2 = {"CylinderReorient": (60.0, 73.3), "CuboidReorient": (73.3, 82.2)}  # LeapHand
# Table 5: 과제 -> (w/ AM SR, TP, w/o AM SR, TP)
T5 = {"HammerUse": (66.7, 86.7, 33.3, 57.1), "ScrewdriverUse": (73.3, 86.7, 0.0, 36.0), "CuboidReorient": (80.0, 86.7, 26.7, 51.1)}
# Table 3 단계별 (성공/분모) 와 SR
T3 = {"BulbInstall": ([(13, 15), (12, 13), (8, 12), (7, 8)], 46.7),
      "HammerDriver": ([(15, 15), (15, 15), (11, 15)], 73.3),
      "BrushForward": ([(7, 15), (7, 7), (6, 7)], 40.0)}
# Table 4: 물체 -> (EpLen Ours, EpLen Dense, Goals Ours, Goals Dense)
T4 = {"Cuboid": (378.6, 115.8, 32.6, 2.6), "Hammer": (376.9, 131.2, 186.6, 2.7), "Screwdriver": (373.2, 88.7, 178.5, 2.7)}
# Table 6 (LeapHand, 15회): 물체 -> (연속 성공 평균, 목표당 성공률 %)
T6 = {"Cylinder": (41.1, 97.6), "Cuboid": (12.1, 92.3)}

# Fig. 6 (PDF 벡터 경로에서 복원한 정수): 과제 -> 방법 -> 단계 1..N을 마친 시행 수
FIG6 = {
    "HammerUse": {"Ours": [15, 15, 15, 13, 13, 10, 10], "DexRT": [15, 6, 5, 1, 1, 0, 0], "GeoRT": [14, 9, 9, 0, 0, 0, 0],
                  "DexGen": [14, 10, 4, 0, 0, 0, 0], "SimToolReal_cat": [15, 14, 0, 0, 0, 0, 0],
                  "SimToolReal_all": [10, 10, 6, 3, 3, 3, 3]},
    "BrushSweep": {"Ours": [15, 15, 15, 13, 13, 12, 11], "DexRT": [15, 8, 8, 5, 5, 0, 0], "GeoRT": [13, 8, 8, 1, 1, 0, 0],
                   "DexGen": [5, 4, 0, 0, 0, 0, 0], "SimToolReal_cat": [13, 7, 6, 4, 4, 4, 4],
                   "SimToolReal_all": [5, 1, 0, 0, 0, 0, 0]},
    "ScrewdriverUse": {"Ours": [15, 13, 13, 13, 11], "DexRT": [15, 8, 3, 1, 1], "GeoRT": [14, 4, 1, 1, 0],
                       "DexGen": [15, 10, 0, 0, 0], "SimToolReal_cat": [12, 1, 0, 0, 0],
                       "SimToolReal_all": [11, 4, 0, 0, 0]},
    "BulbReplace": {"Ours": [15, 15, 15, 15, 13, 13], "DexRT": [15, 12, 2, 2, 1, 0], "GeoRT": [15, 8, 0, 0, 0, 0],
                    "DexGen": [15, 3, 0, 0, 0, 0]},
}
# Fig. 5 (Hammer, 학습 곡선, x = 환경 스텝 1e9): 마지막 10개 점 평균
FIG5_FINAL = {"reward_1e4": {"Ours": 1.51, "no_curriculum": 0.57, "PPO": 0.20},
              "consecutive_successes": {"Ours": 54.8, "no_curriculum": 24.2, "PPO": 12.4}}
FIG5_AT_10 = {"reward_1e4": {"Ours": 0.55}}  # x=9 0.48, x=12 0.68 사이 보간


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(100 * (c - h), 1), round(100 * (c + h), 1)


def fisher(k1, n1, k2, n2):
    return fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out = [None] * len(ps)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = run
    return out


def main():
    res = {}
    # V1: SR이 k/15인지, TP가 정수/(15N)인지, TP-SR 범위
    v1 = []
    for m, d in T1.items():
        for t, (sr, tp) in d.items():
            n_st = STAGES[t]
            k = round(sr * N / 100)
            units = round(tp * N * n_st / 100)
            sr_ok = abs(100 * k / N - sr) < 0.06
            tp_ok = abs(100 * units / (N * n_st) - tp) < 0.06
            lo = k * n_st
            hi = k * n_st + (N - k) * (n_st - 1)
            v1.append({"method": m, "task": t, "k": k, "units": units, "units_range": [lo, hi],
                       "sr_int_ok": sr_ok, "tp_int_ok": tp_ok, "in_range": lo <= units <= hi})
    res["V1_integer_feasibility"] = v1
    res["V1_all_ok"] = all(r["sr_int_ok"] and r["tp_int_ok"] and r["in_range"] for r in v1)
    res["V1_cells"] = len(v1)

    # V2: 평균 행 재계산
    v2 = {}
    for m, (sr_avg, tp_avg) in T1_AVG.items():
        d = T1[m]
        a = sum(v[0] for v in d.values()) / len(d)
        b = sum(v[1] for v in d.values()) / len(d)
        v2[m] = {"tasks": len(d), "sr_recomputed": round(a, 2), "tp_recomputed": round(b, 2), "reported": [sr_avg, tp_avg],
                 "match": abs(a - sr_avg) < 0.06 and abs(b - tp_avg) < 0.06}
    # SimToolReal 3과제 평균에 대응하는 TeleDexter 3과제 평균
    three = ["HammerUse", "BrushSweep", "ScrewdriverUse"]
    v2["Ours_same_three_tasks"] = {"sr": round(sum(T1["Ours"][t][0] for t in three) / 3, 1),
                                   "tp": round(sum(T1["Ours"][t][1] for t in three) / 3, 1)}
    res["V2_averages"] = v2

    # V3: Fig. 6 단계별 시행 수 ↔ Table 1 SR, TP
    v3 = []
    for t, d in FIG6.items():
        n_st = STAGES[t]
        for m, counts in d.items():
            assert len(counts) == n_st, (t, m)
            sr_fig = 100 * counts[-1] / N
            tp_fig = 100 * sum(counts) / (N * n_st)
            sr_t, tp_t = T1[m][t]
            v3.append({"task": t, "method": m, "counts": counts, "sr_fig": round(sr_fig, 1), "sr_table": sr_t,
                       "tp_fig": round(tp_fig, 1), "tp_table": tp_t, "units_fig": sum(counts),
                       "units_table": round(tp_t * N * n_st / 100),
                       "sr_match": abs(sr_fig - sr_t) < 0.06, "tp_match": abs(tp_fig - tp_t) < 0.06,
                       "monotone": all(a >= b for a, b in zip(counts, counts[1:]))})
    res["V3_fig6_vs_table1"] = v3
    res["V3_mismatch"] = [r for r in v3 if not (r["sr_match"] and r["tp_match"])]
    res["V3_cells"] = len(v3)

    # V4: 본문 서술의 단계별 수치 점검 (그림 복원값 기준)
    f = FIG6
    res["V4_text_vs_fig6"] = {
        "hammer_DexRT_15_to_6": [15, f["HammerUse"]["DexRT"][1]],
        "hammer_DexGen_14_to_10": [f["HammerUse"]["DexGen"][0], f["HammerUse"]["DexGen"][1]],
        "hammer_stage4_baseline_nonzero": {m: f["HammerUse"][m][3] for m in ("DexRT", "GeoRT", "DexGen")},
        "hammer_SimToolReal_all_through_completion": f["HammerUse"]["SimToolReal_all"][-1],
        "brush_DexGen_pickup_5": f["BrushSweep"]["DexGen"][0],
        "brush_stage3_8_each": [f["BrushSweep"]["DexRT"][2], f["BrushSweep"]["GeoRT"][2]],
        "brush_SimToolReal_cat_completions_4": f["BrushSweep"]["SimToolReal_cat"][-1],
        "screwdriver_DexGen_10_then_0": [f["ScrewdriverUse"]["DexGen"][1], f["ScrewdriverUse"]["DexGen"][2]],
        "screwdriver_DexRT_completion_1": f["ScrewdriverUse"]["DexRT"][-1],
        "screwdriver_ours_13_then_11": [f["ScrewdriverUse"]["Ours"][2], f["ScrewdriverUse"]["Ours"][-1]],
        "bulb_DexRT_12_to_2": [f["BulbReplace"]["DexRT"][1], f["BulbReplace"]["DexRT"][2]],
        "bulb_ours_15_through_stage4_then_13": [f["BulbReplace"]["Ours"][3], f["BulbReplace"]["Ours"][4], f["BulbReplace"]["Ours"][5]],
        "bulb_ours_trials_stopping_after_stage4": f["BulbReplace"]["Ours"][3] - f["BulbReplace"]["Ours"][4],
    }

    # V5: TeleDexter 대 과제별 최고 기준선 (SR 시행 수), 합산, Holm
    best = {}
    ps, rows = [], []
    for t in TASKS:
        ours_k = round(T1["Ours"][t][0] * N / 100)
        cands = {m: round(T1[m][t][0] * N / 100) for m in T1 if m != "Ours" and t in T1[m]}
        m_best = max(cands, key=cands.get)
        p = fisher(ours_k, N, cands[m_best], N)
        ps.append(p)
        rows.append({"task": t, "ours": f"{ours_k}/{N}", "ours_wilson": wilson(ours_k, N), "best_baseline": m_best,
                     "best_k": f"{cands[m_best]}/{N}", "best_wilson": wilson(cands[m_best], N), "p_fisher": p})
    for r, ph in zip(rows, holm(ps)):
        r["p_holm"] = ph
    res["V5_per_task_vs_best_baseline"] = rows
    pooled = {}
    ours_tot = sum(round(T1["Ours"][t][0] * N / 100) for t in TASKS)
    pooled["ours"] = {"k": ours_tot, "n": N * 7, "pct": round(100 * ours_tot / (N * 7), 2), "wilson": wilson(ours_tot, N * 7)}
    for m in ("DexRT", "GeoRT", "DexGen"):
        k = sum(round(T1[m][t][0] * N / 100) for t in TASKS)
        pooled[m] = {"k": k, "n": N * 7, "wilson": wilson(k, N * 7), "p_vs_ours": fisher(ours_tot, N * 7, k, N * 7)}
    ours3 = sum(round(T1["Ours"][t][0] * N / 100) for t in three)
    pooled["ours_3tasks"] = {"k": ours3, "n": 45, "wilson": wilson(ours3, 45)}
    for m in ("SimToolReal_cat", "SimToolReal_all"):
        k = sum(round(T1[m][t][0] * N / 100) for t in three)
        pooled[m + "_3tasks"] = {"k": k, "n": 45, "wilson": wilson(k, 45), "p_vs_ours": fisher(ours3, 45, k, 45)}
    res["V5_pooled"] = pooled

    # V6: 두 손 비교(LeapHand vs SharpaWave, 같은 과제), Fisher
    v6 = {}
    for t, (sr, tp) in T2.items():
        kl = round(sr * N / 100)
        ks = round(T1["Ours"][t][0] * N / 100)
        v6[t] = {"leap": f"{kl}/{N}", "sharpa": f"{ks}/{N}", "leap_wilson": wilson(kl, N),
                 "p_fisher": fisher(kl, N, ks, N), "tp_units_leap": round(tp * N * 3 / 100)}
    res["V6_leap_vs_sharpa"] = v6

    # V7: Table 5 행동 마스킹 절제
    v7 = {}
    ps7 = []
    for t, (sr_w, tp_w, sr_wo, tp_wo) in T5.items():
        k1, k0 = round(sr_w * N / 100), round(sr_wo * N / 100)
        n_st = STAGES[t]
        p = fisher(k1, N, k0, N)
        ps7.append(p)
        v7[t] = {"with": f"{k1}/{N}", "without": f"{k0}/{N}", "p_fisher": p,
                 "wilson_without": wilson(k0, N), "units_without": round(tp_wo * N * n_st / 100),
                 "tp_int_ok": abs(100 * round(tp_wo * N * n_st / 100) / (N * n_st) - tp_wo) < 0.06,
                 "with_equals_table1_ours": T1["Ours"][t] == (sr_w, tp_w)}
    for t, ph in zip(T5, holm(ps7)):
        v7[t]["p_holm"] = ph
    k1 = sum(round(v[0] * N / 100) for v in T5.values())
    k0 = sum(round(v[2] * N / 100) for v in T5.values())
    v7["pooled"] = {"with": f"{k1}/45", "without": f"{k0}/45", "p_fisher": fisher(k1, 45, k0, 45), "wilson_without": wilson(k0, 45)}
    v7["three_task_mean"] = {"sr_with": round(sum(v[0] for v in T5.values()) / 3, 1), "tp_with": round(sum(v[1] for v in T5.values()) / 3, 1),
                             "sr_without": round(sum(v[2] for v in T5.values()) / 3, 1), "tp_without": round(sum(v[3] for v in T5.values()) / 3, 1)}
    v7["fig2_86_vs_57"] = {"hammer_tp_with": T5["HammerUse"][1], "hammer_tp_without": T5["HammerUse"][3]}
    v7["cuboid_without_am_equals_dexrt_cuboid"] = [T5["CuboidReorient"][2], T5["CuboidReorient"][3]] == list(T1["DexRT"]["CuboidReorient"])
    res["V7_action_mask"] = v7

    # V8: Table 3 자율 정책 단계 사슬
    v8 = {}
    for task, (stages, sr) in T3.items():
        chain_ok = all(stages[i][1] == stages[i - 1][0] for i in range(1, len(stages)))
        first_den = stages[0][1]
        final = stages[-1][0]
        v8[task] = {"chain_denominators_match_previous_numerators": chain_ok, "sr_recomputed": round(100 * final / first_den, 1),
                    "sr_reported": sr, "sr_wilson": wilson(final, first_den),
                    "stage_pass_rates": [round(100 * a / b, 1) for a, b in stages],
                    "stage_wilson": [wilson(a, b) for a, b in stages]}
    k_all = sum(T3[t][0][-1][0] for t in T3)
    v8["pooled"] = {"k": k_all, "n": 45, "wilson": wilson(k_all, 45)}
    res["V8_autonomous"] = v8

    # V9: Table 4 비율
    v9 = {o: {"eplen_ratio": round(a / b, 2), "goals_ratio": round(c / d, 1)} for o, (a, b, c, d) in T4.items()}
    res["V9_table4_ratios"] = v9
    res["V9_eplen_sparse_range"] = [min(v[0] for v in T4.values()), max(v[0] for v in T4.values())]

    # V10: Table 6 목표당 성공률과 연속 성공 평균의 관계 (시행이 첫 실패에서 끝난다고 가정)
    v10 = {}
    for o, (mean_cons, per) in T6.items():
        succ = N * mean_cons
        implied = 100 * succ / (succ + N)  # 시행마다 실패 시도 1회
        v10[o] = {"mean_consecutive": mean_cons, "reported_per_target_sr": per, "implied_if_one_failure_per_trial": round(implied, 2),
                  "total_successes_est": round(succ, 1), "geometric_p_from_mean": round(100 * mean_cons / (mean_cons + 1), 2)}
    res["V10_any_to_any"] = v10

    # V11: 부록 수치 산술
    dims = {"SharpaWave": [22, 22, 22, 3, 4, 3, 15, 15, 3, 3, 4, 4, 22], "LeapHand": [16, 16, 16, 3, 4, 3, 12, 12, 3, 3, 4, 4, 16]}
    envs = 4 * 15600
    res["V11_appendix_arithmetic"] = {
        "obs_dim_sum": {k: sum(v) for k, v in dims.items()}, "obs_dim_reported": {"SharpaWave": 142, "LeapHand": 112},
        "envs_total": envs, "minibatch_vs_envs": 15600 * 32 / 31200,
        "reference_minutes_per_object": 150 * 20 / 60, "frames_per_traj_at_30Hz": 20 * 30,
        "step_bound_seconds_at_60Hz": [40 / 60, 80 / 60], "step_bound_seconds_at_30Hz": [40 / 30, 80 / 30],
        "curriculum_sigma_window_env_steps": 25600 * envs, "gravity_window_env_steps": 32000 * envs,
        "total_steps_1e10_per_day_per_sec": 1e10 / 86400,
        "reward_score_max": 1.5 * (5 * 0.5 + 2.0 * 2), "reward_dense_max_late": 0.1 * (1.0 + 4 * 0.8 + 0.6 + 0.4 + 0.3 * (1.5 + 1.5)),
        "reward_step_weight_max_in_traj": 80 + 5,
        "sparse_bonus_max_per_subgoal": (80 + 5) * 1.5 * (5 * 0.5 + 2.0 * 2),
        "dense_object_scale_max": 1 - 0.7,
        "retarget_stage1_iters_total": 6000, "retarget_stage2_iters_total": 160,
    }
    # V12: Fig. 5
    res["V12_fig5"] = {"final": FIG5_FINAL, "ratio_reward_vs_no_curriculum": round(1.51 / 0.57, 2), "ratio_reward_vs_ppo": round(1.51 / 0.20, 2),
                       "ratio_succ_vs_no_curriculum": round(54.8 / 24.2, 2), "ratio_succ_vs_ppo": round(54.8 / 12.4, 2),
                       "ours_reward_at_1e10_share_of_final": round(0.55 / 1.51, 2),
                       "note": "w/o curriculum 곡선이 Ours 곡선보다 위인 점은 0개(약 830점 중)"}


    # V14: Fig. 6 복원 시행 분포로 TP 구간 (시행별 도달 단계 k를 계단 수에서 복원, 부트스트랩 1만 회, seed 0)
    import numpy as np
    from scipy.stats import mannwhitneyu
    rng = np.random.default_rng(0)
    v14 = {}
    for t, d in FIG6.items():
        n_st = STAGES[t]
        per = {}
        for m, counts in d.items():
            ks = []
            full = [N] + list(counts)
            for k in range(n_st + 1):
                ks += [k] * (full[k] - (full[k + 1] if k < n_st else 0))
            assert len(ks) == N
            arr = np.array(ks) / n_st
            boots = rng.choice(arr, size=(10000, N)).mean(axis=1)
            per[m] = {"tp_mean": round(100 * float(arr.mean()), 1),
                      "tp_ci": [round(100 * float(np.percentile(boots, 2.5)), 1), round(100 * float(np.percentile(boots, 97.5)), 1)],
                      "ks": ks}
        best_m = max((m for m in d if m != "Ours"), key=lambda m: per[m]["tp_mean"])
        u = mannwhitneyu(per["Ours"]["ks"], per[best_m]["ks"], alternative="greater")
        per["_best_baseline_by_tp"] = {"method": best_m, "mannwhitney_p": float(u.pvalue)}
        v14[t] = per
    res["V14_tp_bootstrap"] = v14

    # V13: 검정력 (15회 대 15회, 양측 Fisher, 임의의 기준 성공률)
    from itertools import product
    power = {}
    for p0 in (0.0, 0.2, 0.5):
        # 15회에서 p0 기준선과 구별되는 최소 성공 횟수 (정확 검정으로 직접 찾음: 평균 기대가 아니라 관측 k로)
        k0 = round(p0 * N)
        found = None
        for k1 in range(k0, N + 1):
            if fisher(k1, N, k0, N) < 0.05:
                found = k1
                break
        power[f"baseline_{k0}_of_15"] = {"min_successes_for_p_lt_0.05": found, "as_pct": round(100 * found / N, 1) if found else None}
    res["V13_min_detectable"] = power

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    print("wrote", OUT)
    print("V1 all ok:", res["V1_all_ok"], "cells", res["V1_cells"])
    print("V2:", {m: v["match"] for m, v in v2.items() if "match" in v})
    print("V3 cells:", res["V3_cells"], "mismatches:", [(r["task"], r["method"], r["units_fig"], r["units_table"]) for r in res["V3_mismatch"]])
    for r in rows:
        print("V5", r["task"], r["ours"], r["best_baseline"], r["best_k"], "p=%.2g holm=%.2g" % (r["p_fisher"], r["p_holm"]))
    print("V5 pooled", {k: v for k, v in pooled.items()})
    print("V7", json.dumps(v7, default=float)[:600])
    print("V10", v10)


if __name__ == "__main__":
    main()
