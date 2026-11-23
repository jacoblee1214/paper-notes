"""T-Rex (arXiv:2606.17055 v2) 보고 수치 재계산과 검정. 결과는 recompute_results.json.

실행: python3 analysis/recompute.py   (numpy, scipy; 수 초)

입력은 논문 표 1–3과 그림 4–6에서 옮긴 값입니다. 그림 4–6의 값은 그림에 적힌 데이터 라벨을 읽은 것입니다(그림 판독 값).
시행 수는 §5.1의 과제당 16회를 모든 표와 그림에 적용합니다(표 2·3, 그림 4–6의 시행 수는 논문에 없어서 가정).
논문의 성공률은 부분 점수 루브릭(부록 F)이라 시행별 점수가 [0, 1] 연속값일 수 있습니다. [0, 1] 값의 분산은
평균이 같은 베르누이 분산(p(1-p))을 넘지 못하므로, 이진 성공으로 계산한 표준오차는 보수적인 상한입니다.
"""
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
N_TRIAL = 16  # 과제당 시행 수 (§5.1, 표 1 캡션)

TASKS = ["Flip Page", "Transfer Egg", "Wipe Plate", "Apply Paste", "Split Cup", "Sort Mahjong",
         "Open Lock", "Refill Tablet", "Acid-Base Neut.", "Extract Card", "Deal Poker", "Screw Bulb"]
T1 = {  # 표 1 (행: 방법, 열: 12개 과제), Avg. 열은 보고값
    "ViTacFormer": ([9, 0, 4, 1, 4, 7, 0, 0, 0, 2, 2, 1], 3),
    "RDP": ([12, 8, 18, 2, 6, 9, 2, 0, 0, 1, 2, 7], 6),
    "Tactile-VLA": ([38, 14, 24, 0, 21, 27, 8, 0, 9, 4, 11, 18], 15),
    "EgoScale": ([68, 44, 34, 38, 33, 36, 19, 12, 43, 41, 28, 18], 35),
    "pi0.5": ([36, 17, 28, 13, 18, 32, 5, 1, 24, 8, 9, 11], 17),
    "pi0.5 + tactile": ([8, 9, 27, 2, 4, 14, 2, 0, 7, 3, 0, 0], 6),
    "Ours": ([96, 75, 69, 66, 78, 65, 47, 41, 76, 70, 57, 35], 65),
}
SIX = ["Flip Page", "Apply Paste", "Split Cup", "Open Lock", "Extract Card", "Screw Bulb"]
SIX_IDX = [TASKS.index(t) for t in SIX]
T2 = {  # 표 2 (6개 과제), Average와 괄호 안 차이는 보고값
    "Full": ([96, 66, 78, 47, 70, 35], 65, None),
    "w/o Tactile": ([76, 39, 58, 23, 34, 20], 42, -23),
    "MLP Force + Deform": ([89, 58, 72, 44, 58, 29], 58, -7),
    "Deform": ([82, 57, 71, 36, 55, 25], 54, -11),
    "MLP Force + VQVAE Force": ([92, 63, 65, 38, 67, 28], 59, -6),
    "w/o Async": ([92, 61, 73, 45, 59, 30], 60, -5),
}
T3 = {  # 표 3: (사전학습, 중간학습) -> 6개 과제, Average는 보고값
    "pre-, mid-": ([46, 16, 20, 6, 14, 5], 18),
    "pre-, mid+": ([75, 34, 45, 10, 32, 9], 34),
    "pre+, mid-": ([88, 40, 52, 22, 46, 20], 45),
    "pre+, mid+": ([96, 66, 78, 47, 70, 35], 65),
}
# 그림 4: 분할 단계 K_slow = 1, 2, 4, 6, 8 (그림 판독 값)
F4 = {"Apply Toothpaste": [62, 65, 66, 66, 64], "Split Cup": [71, 74, 75, 78, 73], "Extract Card": [66, 65, 68, 70, 68]}
F4_K = [1, 2, 4, 6, 8]
# 그림 5: 시연 수 10, 20, 50, 100, 200. blue = 중간학습 있음, green = 없음 (그림 판독 값)
F5_N = [10, 20, 50, 100, 200]
F5 = {
    "Apply Toothpaste": {"blue": [29, 51, 60, 66, 71], "green": [9, 18, 30, 34, 36]},
    "Split Cup": {"blue": [36, 59, 70, 78, 85], "green": [17, 33, 42, 52, 55]},
    "Extract Card": {"blue": [26, 48, 60, 70, 78], "green": [12, 20, 36, 46, 50]},
}
# 그림 6 (그림 판독 값): 사후학습 6개 과제와 제로샷 4개 과제
F6_POST_TASKS = ["Flip Page", "Apply Toothpaste", "Split Cup", "Open Lock", "Extract Card", "Screw Lightbulb"]
F6_POST = {"T-Rex dataset": [96, 66, 78, 47, 70, 35], "Task-specific": [91, 50, 72, 39, 64, 29],
           "w/o mid-training": [88, 40, 52, 22, 46, 20]}
F6_ZS_TASKS = ["Pick Cube", "Flip Page", "Press Button", "Wipe Plate"]
F6_ZS = {"T-Rex dataset": [45, 20, 35, 10], "Task-specific": [15, 5, 10, 0], "w/o mid-training": [5, 0, 5, 0]}

# 부록 F 루브릭 배점 (과제 순서는 TASKS와 같다)
RUBRIC = {
    "Flip Page": [0.3, 0.3, 0.4], "Transfer Egg": [0.2, 0.3, 0.2, 0.3], "Wipe Plate": [0.2, 0.1, 0.2, 0.4, 0.1],
    "Apply Paste": [0.2, 0.1, 0.4, 0.2, 0.1], "Split Cup": [0.2, 0.3, 0.3, 0.2], "Sort Mahjong": [0.1, 0.5, 0.2, 0.2],
    "Open Lock": [0.2, 0.1, 0.4, 0.3], "Refill Tablet": [0.2] * 5,
    "Acid-Base Neut.": [0.1, 0.15, 0.1, 0.15, 0.15, 0.15, 0.1, 0.1], "Extract Card": [0.2, 0.3, 0.3, 0.2],
    "Deal Poker": [0.1, 0.2, 0.3, 0.3, 0.1], "Screw Bulb": [0.1, 0.2, 0.1, 0.4, 0.2],
}


def rnd(x):
    """반올림은 0.5를 올립니다(표의 평균 표기에 맞춤)."""
    return int(math.floor(x + 0.5))


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def two_prop_z(p1, p2, n1, n2):
    """베르누이 분산(상한) 가정의 비풀링 z 검정, 양측 p."""
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


def count(pct, n=N_TRIAL):
    return int(round(pct / 100 * n))


def main():
    res = {"n_trial_per_task": N_TRIAL}

    # ---- V1: 표 1 평균 ----
    rows = {}
    for name, (vals, rep) in T1.items():
        mean = float(np.mean(vals))
        rows[name] = {"mean": round(mean, 3), "reported": rep, "rounded_ok": rnd(mean) == rep}
    res["V1_table1_means"] = rows

    # ---- V2: 표 1 값이 k/16(%)로 표현되는지 ----
    feas = {}
    n_bad = 0
    bad_cells = []
    grid = [100 * k / N_TRIAL for k in range(N_TRIAL + 1)]
    for name, (vals, _) in T1.items():
        for t, v in zip(TASKS, vals):
            ok = any(abs(g - v) <= 0.5 + 1e-9 for g in grid)
            if not ok:
                n_bad += 1
                bad_cells.append((name, t, v))
    feas["cells_total"] = 12 * len(T1)
    feas["cells_not_on_k_over_16_grid"] = n_bad
    feas["examples"] = bad_cells[:12]
    res["V2_binary_grid_feasibility"] = feas

    # ---- V3: 표 2·3 평균과 차이 ----
    t2 = {}
    full_mean = np.mean(T2["Full"][0])
    for name, (vals, rep, d) in T2.items():
        m = float(np.mean(vals))
        t2[name] = {"mean": round(m, 3), "reported": rep, "rounded_ok": rnd(m) == rep,
                    "delta_vs_full_pts": round(m - full_mean, 2),
                    "delta_reported": d, "delta_ok": d is None or rnd(m) - 65 == d,
                    "tasks_below_full": int(sum(a < b for a, b in zip(vals, T2["Full"][0])))}
    res["V3_table2"] = t2
    t3 = {}
    for name, (vals, rep) in T3.items():
        m = float(np.mean(vals))
        t3[name] = {"mean": round(m, 3), "reported": rep, "rounded_ok": rnd(m) == rep}
    res["V3_table3"] = t3

    # ---- V4: 표 간 일관성 ----
    full = T2["Full"][0]
    t1_full_six = [T1["Ours"][0][i] for i in SIX_IDX]
    cons = {
        "T1_Ours_six_equals_T2_Full": t1_full_six == full,
        "T3_full_row_equals_T2_Full": T3["pre+, mid+"][0] == full,
        "F6_T-Rex_dataset_equals_T2_Full": F6_POST["T-Rex dataset"] == full,
        "F5_blue_at_100_equals_T2_Full(3 tasks)": [F5[k]["blue"][3] for k in F5] == [full[1], full[2], full[4]],
        "F4_Kslow6_equals_T2_Full(3 tasks)": [F4[k][3] for k in F4] == [full[1], full[2], full[4]],
        "F6_w/o_mid_equals_T3_pre+_mid-": F6_POST["w/o mid-training"] == T3["pre+, mid-"][0],
    }
    g100 = {k: F5[k]["green"][3] for k in F5}
    t3_row = dict(zip(["Apply Toothpaste", "Split Cup", "Extract Card"], [T3["pre+, mid-"][0][1], T3["pre+, mid-"][0][2], T3["pre+, mid-"][0][4]]))
    cons["F5_green_at_100_vs_T3_pre+_mid-"] = {k: (g100[k], t3_row[k]) for k in g100}
    cons["F5_green_at_100_vs_T3_pre-_mid+"] = {k: (g100[k], v) for k, v in zip(g100, [T3["pre-, mid+"][0][1], T3["pre-, mid+"][0][2], T3["pre-, mid+"][0][4]])}
    res["V4_cross_table_consistency"] = cons

    # ---- V5: 루브릭 배점 합 ----
    res["V5_rubric_sums"] = {t: round(sum(v), 10) for t, v in RUBRIC.items()}

    # ---- V6: 표 1 방법별 평균의 보수적 오차 (12 과제 x 16 = 192 시행) ----
    n_all = 12 * N_TRIAL
    ci = {}
    for name, (vals, _) in T1.items():
        p = np.array(vals) / 100
        se_bern = math.sqrt(float(np.sum(p * (1 - p) / N_TRIAL))) / 12  # 과제 평균의 SE, 과제 간 독립 가정
        ci[name] = {"mean_pct": round(float(p.mean() * 100), 2), "se_bernoulli_pts": round(se_bern * 100, 2),
                    "ci95_half_pts": round(1.96 * se_bern * 100, 2)}
    ci["worst_case_se_pts(0.5/sqrt(192))"] = round(0.5 / math.sqrt(n_all) * 100, 2)
    res["V6_table1_mean_uncertainty"] = ci

    # ---- V7: 표 1, Ours 대 각 기준선 (과제 쌍) ----
    cmp_res = {}
    ours = np.array(T1["Ours"][0])
    for name, (vals, _) in T1.items():
        if name == "Ours":
            continue
        base = np.array(vals)
        d = ours - base
        k1 = [count(v) for v in ours]
        k0 = [count(v) for v in base]
        fp = [stats.fisher_exact([[a, N_TRIAL - a], [b, N_TRIAL - b]])[1] for a, b in zip(k1, k0)]
        fp_h = holm(np.array(fp))
        zp = [two_prop_z(a / 100, b / 100, N_TRIAL, N_TRIAL)[1] for a, b in zip(ours, base)]
        # 합산 (192 대 192): 이진 근사
        K1, K0 = sum(k1), sum(k0)
        pooled = stats.fisher_exact([[K1, n_all - K1], [K0, n_all - K0]])[1]
        wil = stats.wilcoxon(ours, base).pvalue if np.any(d != 0) else 1.0
        sign = stats.binomtest(int((d > 0).sum()), 12, 0.5).pvalue
        cmp_res[name] = {
            "mean_diff_pts": round(float(d.mean()), 2), "wins": int((d > 0).sum()),
            "min_task_diff": int(d.min()), "task_of_min": TASKS[int(d.argmin())],
            "fisher_p_by_task_unadj": [float(f"{x:.3g}") for x in fp],
            "n_tasks_p<0.05_unadj": int(sum(x < 0.05 for x in fp)),
            "n_tasks_p<0.05_holm": int(sum(x < 0.05 for x in fp_h)),
            "tasks_not_sig_holm": [TASKS[i] for i, x in enumerate(fp_h) if x >= 0.05],
            "z_bernoulli_p_by_task": [float(f"{x:.3g}") for x in zp],
            "n_tasks_z_p<0.05_unadj": int(sum(x < 0.05 for x in zp)),
            "pooled_fisher_p_binary_approx": float(f"{pooled:.3g}"),
            "sign_test_p": float(f"{sign:.3g}"), "wilcoxon_p": float(f"{wil:.3g}"),
        }
    res["V7_table1_ours_vs_baselines"] = cmp_res

    # ---- V8: 표 1 순위/상대 크기 ----
    ego = np.array(T1["EgoScale"][0])
    res["V8_relative"] = {
        "abs_gain_vs_EgoScale_pts": round(float(ours.mean() - ego.mean()), 2),
        "relative_gain_vs_EgoScale_pct": round(float((ours.mean() / ego.mean() - 1) * 100), 1),
        "ratio_vs_EgoScale": round(float(ours.mean() / ego.mean()), 2),
        "tasks_where_ours_below_50": [t for t, v in zip(TASKS, ours) if v < 50],
        "pi05_tactile_minus_pi05_pts": round(float(np.mean(T1["pi0.5 + tactile"][0]) - np.mean(T1["pi0.5"][0])), 2),
        "pi05_tactile_worse_in_tasks": int(sum(a < b for a, b in zip(T1["pi0.5 + tactile"][0], T1["pi0.5"][0]))),
        "min_diff_task_vs_EgoScale": {"task": TASKS[int((ours - ego).argmin())], "diff": int((ours - ego).min())},
    }

    # ---- V9: 여섯 과제 부분집합과 분해 ----
    ego6 = float(np.mean([T1["EgoScale"][0][i] for i in SIX_IDX]))
    wo_tac = float(np.mean(T2["w/o Tactile"][0]))
    full6 = float(np.mean(full))
    res["V9_six_task_decomposition"] = {
        "EgoScale_six_mean": round(ego6, 2), "T-Rex_w/o_tactile_six_mean": round(wo_tac, 2), "T-Rex_full_six_mean": round(full6, 2),
        "full_minus_EgoScale": round(full6 - ego6, 2), "wo_tactile_minus_EgoScale": round(wo_tac - ego6, 2),
        "full_minus_wo_tactile": round(full6 - wo_tac, 2),
        "share_of_gap_from_tactile_pct": round(100 * (full6 - wo_tac) / (full6 - ego6), 1),
        "ours_12task_mean": round(float(ours.mean()), 2), "ours_other6_mean": round(float(np.mean([v for i, v in enumerate(ours) if i not in SIX_IDX])), 2),
        "EgoScale_other6_mean": round(float(np.mean([v for i, v in enumerate(ego) if i not in SIX_IDX])), 2),
    }

    # ---- V10: 표 2 절제 검정 (6 과제 x 16 = 96 시행, 이진 근사) ----
    n6 = 6 * N_TRIAL
    abl = {}
    ps = []
    names = []
    for name, (vals, _, _) in T2.items():
        if name == "Full":
            continue
        p1, p0 = full6 / 100, float(np.mean(vals)) / 100
        z, p = two_prop_z(p1, p0, n6, n6)
        K1, K0 = int(round(p1 * n6)), int(round(p0 * n6))
        fp = stats.fisher_exact([[K1, n6 - K1], [K0, n6 - K0]])[1]
        sign = stats.binomtest(int(sum(a > b for a, b in zip(full, vals))), 6, 0.5).pvalue
        abl[name] = {"delta_pts": round((p1 - p0) * 100, 2), "z_bernoulli": round(z, 2), "p_z": float(f"{p:.3g}"),
                     "p_fisher_binary_approx": float(f"{fp:.3g}"), "tasks_full_higher": int(sum(a > b for a, b in zip(full, vals))),
                     "sign_test_p": float(f"{sign:.3g}")}
        ps.append(p)
        names.append(name)
    for n, a in zip(names, holm(np.array(ps))):
        abl[n]["p_z_holm_over_5"] = float(f"{a:.3g}")
    res["V10_table2_ablation_tests"] = abl
    # 최소 검출 차이(80% 검정력, 양측 0.05, p = 0.5 기준 베르누이 분산)
    zq = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
    res["V10_minimum_detectable_diff_pts"] = {
        "single_task_16_vs_16": round(100 * zq * math.sqrt(2 * 0.25 / 16), 1),
        "six_tasks_96_vs_96": round(100 * zq * math.sqrt(2 * 0.25 / 96), 1),
        "twelve_tasks_192_vs_192": round(100 * zq * math.sqrt(2 * 0.25 / 192), 1),
    }

    # ---- V11: 표 3 ----
    pre_gain = np.mean(T3["pre+, mid-"][0]) - np.mean(T3["pre-, mid-"][0])
    mid_gain = np.mean(T3["pre-, mid+"][0]) - np.mean(T3["pre-, mid-"][0])
    both = np.mean(T3["pre+, mid+"][0]) - np.mean(T3["pre-, mid-"][0])
    res["V11_table3_additivity"] = {"pre_gain": round(float(pre_gain), 2), "mid_gain": round(float(mid_gain), 2),
                                    "both_gain": round(float(both), 2), "sum_of_single": round(float(pre_gain + mid_gain), 2),
                                    "interaction": round(float(both - pre_gain - mid_gain), 2),
                                    "pre+mid-_vs_EgoScale_six": round(float(np.mean(T3["pre+, mid-"][0]) - ego6), 2)}

    # ---- V12: 그림 4 분할 단계 곡선의 변동 폭 ----
    f4 = {}
    for k, v in F4.items():
        f4[k] = {"max_minus_min": max(v) - min(v), "argmax_K": [F4_K[i] for i, x in enumerate(v) if x == max(v)],
                 "se_per_point_bernoulli_pts": round(100 * math.sqrt(np.mean(v) / 100 * (1 - np.mean(v) / 100) / N_TRIAL), 1)}
    # w/o Async 행과 K_slow 곡선
    wa = dict(zip(["Apply Toothpaste", "Split Cup", "Extract Card"], [T2["w/o Async"][0][1], T2["w/o Async"][0][2], T2["w/o Async"][0][4]]))
    f4["w/o_Async_values"] = wa
    f4["pooled_range_over_K_(mean of 3 tasks)"] = [round(float(np.mean([F4[k][i] for k in F4])), 2) for i in range(5)]
    res["V12_figure4_split_step"] = f4

    # ---- V13: 그림 5 데이터 효율 ----
    f5 = {"gap_blue_minus_green": {}, "blue_wins": 0, "pairs": 0}
    for k, d in F5.items():
        gap = [b - g for b, g in zip(d["blue"], d["green"])]
        f5["gap_blue_minus_green"][k] = gap
        f5["blue_wins"] += sum(x > 0 for x in gap)
        f5["pairs"] += len(gap)
    f5["blue20_vs_green200"] = {k: (F5[k]["blue"][1], F5[k]["green"][4]) for k in F5}
    f5["blue50_vs_green200"] = {k: (F5[k]["blue"][2], F5[k]["green"][4]) for k in F5}
    f5["mean_gap_by_N"] = [round(float(np.mean([F5[k]["blue"][i] - F5[k]["green"][i] for k in F5])), 2) for i in range(5)]
    f5["sign_test_p_15_of_15"] = float(f"{stats.binomtest(15, 15, 0.5).pvalue:.3g}")
    f5["per_point_z_N10"] = {k: round(two_prop_z(F5[k]["blue"][0] / 100, F5[k]["green"][0] / 100, N_TRIAL, N_TRIAL)[0], 2) for k in F5}
    f5["per_point_p_N10"] = {k: float(f"{two_prop_z(F5[k]['blue'][0] / 100, F5[k]['green'][0] / 100, N_TRIAL, N_TRIAL)[1]:.3g}") for k in F5}
    res["V13_figure5_data_efficiency"] = f5

    # ---- V14: 그림 6 ----
    f6 = {}
    for split, tasks, dd in (("post", F6_POST_TASKS, F6_POST), ("zero_shot", F6_ZS_TASKS, F6_ZS)):
        m = {k: round(float(np.mean(v)), 2) for k, v in dd.items()}
        f6[split] = {"means": m,
                     "trex_minus_task_specific_by_task": [a - b for a, b in zip(dd["T-Rex dataset"], dd["Task-specific"])],
                     "trex_beats_task_specific_in": int(sum(a > b for a, b in zip(dd["T-Rex dataset"], dd["Task-specific"]))),
                     "n_tasks": len(tasks)}
    f6["post_task_specific_vs_wo_mid_gain"] = round(float(np.mean(F6_POST["Task-specific"]) - np.mean(F6_POST["w/o mid-training"])), 2)
    f6["post_trex_vs_task_specific_gain"] = round(float(np.mean(F6_POST["T-Rex dataset"]) - np.mean(F6_POST["Task-specific"])), 2)
    res["V14_figure6_midtraining"] = f6

    # ---- V15: 데이터셋 산술 ----
    ds = {
        "pairs_207x22": 207 * 22, "episodes_per_pair": round(7755 / 502, 2), "mean_episode_sec_from_100h_7755": round(100 * 3600 / 7755, 2),
        "paper_median_sec": 29.8, "paper_q1_q3_sec": [21.0, 41.1],
        "released_episodes": 5464, "released_frames": 5473459, "released_hours_at_30fps": round(5473459 / 30 / 3600, 2),
        "released_mean_episode_sec": round(5473459 / 30 / 5464, 2),
        "released_share_of_episodes_pct": round(100 * 5464 / 7755, 1), "released_share_of_hours_vs_100h_pct": round(100 * (5473459 / 30 / 3600) / 100, 1),
        "implied_unreleased_episodes": 7755 - 5464, "implied_unreleased_hours_if_100h": round(100 - 5473459 / 30 / 3600, 2),
        "implied_unreleased_mean_episode_sec_if_100h": round((100 - 5473459 / 30 / 3600) * 3600 / (7755 - 5464), 1),
        "collection_hours_per_week_10_weeks": 10.0, "dataset_tb_hf_usedStorage": round(1551267725664 / 1e12, 2),
    }
    res["V15_dataset_arithmetic"] = ds

    # ---- V16: 파라미터 수 (Qwen3-VL-2B 텍스트 설정은 공개 설정값을 가정) ----
    H, L, nh, nkv, hd, I, V = 2048, 28, 16, 8, 128, 6144, 151936
    attn = H * nh * hd + 2 * H * nkv * hd + nh * hd * H
    norms_layer = 2 * H + 2 * hd
    mlp = 3 * H * I
    lat = L * (attn + mlp + norms_layer)
    tac_mlp = 3 * H * 1536
    tac = L * (attn + tac_mlp + norms_layer)
    emb = V * H
    deform_proj = 28800 * H + H * H
    vit_assumed = 0.4e9
    total = 2 * lat + tac + emb + vit_assumed + deform_proj
    res["V16_param_count"] = {
        "assumed_config": {"hidden": H, "layers": L, "heads": nh, "kv_heads": nkv, "head_dim": hd, "intermediate": I, "vocab": V, "vit_params_assumed": vit_assumed},
        "latent_expert_nonembed_B": round(lat / 1e9, 3), "action_expert_nonembed_B": round(lat / 1e9, 3), "tactile_expert_B": round(tac / 1e9, 3),
        "embedding_table_B": round(emb / 1e9, 3), "deform_proj_M": round(deform_proj / 1e6, 1),
        "approx_total_B": round(total / 1e9, 2), "bf16_GB": round(total * 2 / 1e9, 2), "hf_midtrain_repo_usedStorage_GB": 8.52,
        "adamw_fp32_full_finetune_GB_(16 B/param, rough)": round(total * 16 / 1e9, 1),
    }
    # 빠른 틱 토큰 수와 FLOPs
    tokens = 10 + 10 + 10 + 1 + 16
    flops_step = 2 * tac * tokens
    res["V16_fast_tick_cost"] = {"tactile_tokens_per_tick": tokens, "tactile_expert_matmul_GFLOPs_per_euler_step": round(flops_step / 1e9, 1),
                                 "per_tick_4_steps_GFLOPs": round(4 * flops_step / 1e9, 1),
                                 "note": "KV 캐시에 대한 어텐션과 임베더는 제외한 하한. 측정하지 않음"}

    # ---- V17: 타임스텝 격자와 훈련 분포 ----
    N, K_SLOW = 10, 6
    tau_split = 1 - K_SLOW / N
    act_grid = [round(1 - k / N, 2) for k in range(K_SLOW)]
    tac_grid = [round(tau_split - k / N, 2) for k in range(N - K_SLOW)]
    cdf = lambda x: x ** 1.5  # Beta(1.5, 1)의 CDF
    res["V17_timestep"] = {
        "tau_split": tau_split, "action_eval_taus": act_grid, "tactile_eval_taus": tac_grid,
        "P_action_train_tau_in_[0.4,1]": round(1 - cdf(0.4), 3), "P_action_train_tau_<0.4": round(cdf(0.4), 3),
        "P_tactile_train_tau_<0.1": round(cdf(0.1 / tau_split), 3), "P_tactile_train_tau_>0.3": round(1 - cdf(0.3 / tau_split), 3),
        "action_expert_eval_mass_in_trained_range": "추론은 [0.5, 1]에서만 평가하고 훈련은 (0, 1]에 걸침",
    }

    # ---- V18: 주파수 산술 (코드 기본 설정) ----
    cmd_hz, chunk, offs = 30.0, 16, [0, 4, 8, 12]
    res["V18_rates"] = {
        "paper_slow_hz": 5, "paper_fast_hz": 20, "paper_ratio": 4,
        "config_command_hz": cmd_hz, "config_chunk": chunk, "config_refine_offsets_with_slow": offs,
        "slow_hz_if_chunk_fully_executed": round(cmd_hz / chunk, 3), "fast_hz_nominal": round(cmd_hz / 4, 2),
        "chunk_duration_s": round(chunk / cmd_hz, 3), "chunk_steps_per_s_needed_for_5hz_slow": 5 * chunk,
    }

    # ---- V19: 소트 마작 찍기 상한 (가정: 세 상자 균등, 나머지 단계 완벽) ----
    mj = RUBRIC["Sort Mahjong"]
    guess_score = mj[0] + (1 / 3) * (mj[1] + mj[2] + mj[3])
    res["V19_mahjong_chance"] = {"guess_expected_score_pct": round(100 * guess_score, 1),
                                 "EgoScale": T1["EgoScale"][0][5], "Tactile-VLA": T1["Tactile-VLA"][0][5], "Ours": T1["Ours"][0][5],
                                 "assumption": "세 상자를 균등 확률로 고르고 나머지 단계는 모두 성공한다고 가정한 상한"}

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT)
    for k in ("V1_table1_means", "V2_binary_grid_feasibility", "V3_table2", "V4_cross_table_consistency"):
        print(k, json.dumps(res[k], ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()
