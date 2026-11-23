"""ProxiDex (arXiv:2609.16586 v1) 표·본문 수치 재계산과 일관성 검사. CPU, 수 초.

실행: python3 analysis/recompute.py   → analysis/recompute_results.json (사람이 읽는 요약은 stdout)

범위
- V1  표 6 ↔ 표 1: 행 평균, 묶음 평균, 평균 ± 칸
- V2  표 7·8 평균, 본문 차이(7.5, 7.0, 4.0, 2.3, 1.1), 표 6과의 겹침
- V3  표 2 실로봇 횟수: 합산, 평균 칸, 본문 차이와 85/120, 67/90
- V4  표 2 검정: 설정별 Fisher, 합산 Fisher, 층화(CMH) 검정, 부호 검정, 검출력
- V5  표 6 시뮬 격차: 시드 수 가정에 따른 Welch 검정 (논문은 시드 수를 쓰지 않음)
- V6  그림 5·9에서 읽은 값의 정수 가능성 (시행 수 가정)
- V7  표 8 Hdyn 절제: Toilet 열을 빼면
- V8  표 사이 같은 평균±SD 칸 찾기
- V9  부록 D.3 데이터 크기 곡선 산술, 사용자 연구 정수 가능성
- V10 학습 스케줄 대 학습 단계 수 (시연 길이 가정)
- V11 본문 인용 부분집합(새 물체 85/120, 교란 67/90)의 검정, 표 7 절제 폭을 가르는 데 필요한 시드 수(표의 Avg ± 를 SD로 볼 때)
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "recompute_results.json"
R = {}

# ---------------------------------------------------------------- 표 6 (부록 C.2) 과제별 평균±SD
T6_TASKS = ["Hammer", "Door", "Pen", "Laptop", "Faucet", "Toilet", "Bucket", "Box", "Driller", "Banana"]
GROUP = {"Adroit": [0, 1, 2], "DexArt": [3, 4, 5, 6], "Self": [7, 8, 9]}
T6 = {
    "DP3": ([100.0, 76.7, 56.7, 89.7, 41.7, 79.7, 31.3, 90.3, 74.3, 77.8], [0.0, 4.7, 2.6, 0.9, 0.5, 0.9, 0.5, 3.2, 2.6, 1.2], (71.8, 1.7)),
    "ManiFlow": ([100.0, 80.3, 55.5, 93.0, 45.0, 79.9, 35.3, 94.7, 78.6, 79.8], [0.0, 1.2, 5.8, 1.6, 3.6, 3.3, 2.1, 2.1, 1.4, 2.3], (74.2, 2.3)),
    "AFRO": ([96.0, 83.8, 72.3, 86.3, 42.3, 82.2, 48.0, 97.2, 79.4, 81.4], [2.8, 3.4, 2.1, 2.5, 1.8, 3.4, 2.3, 1.3, 1.5, 1.8], (76.9, 2.3)),
    "CordViP": ([96.0, 84.7, 76.2, 92.6, 56.5, 81.0, 57.0, 96.3, 85.3, 84.7], [2.1, 1.7, 3.2, 2.1, 1.3, 2.2, 1.6, 2.0, 1.1, 1.6], (81.0, 1.9)),
    "ProxiDex": ([100.0, 86.5, 84.3, 93.0, 53.4, 84.3, 64.0, 96.7, 89.2, 87.3], [0.0, 1.9, 2.4, 1.2, 2.1, 2.3, 3.2, 1.6, 2.0, 1.5], (83.9, 1.8)),
}
# 표 1 (본문): Adroit, DexArt, Self, Avg  (평균, SD)
T1 = {
    "DP3": [(77.8, 2.4), (60.6, 0.7), (80.8, 2.3), (71.8, 1.7)],
    "ManiFlow": [(78.6, 2.3), (63.3, 2.7), (84.4, 1.9), (74.2, 2.3)],
    "AFRO": [(84.0, 2.8), (64.7, 2.5), (86.0, 1.5), (76.9, 2.3)],
    "CordViP": [(85.6, 2.3), (71.8, 1.8), (88.8, 1.6), (81.0, 1.9)],
    "ProxiDex": [(90.3, 1.4), (73.7, 2.2), (91.1, 1.7), (83.9, 1.8)],
}


def r1(x):
    """반올림(0.05는 올림). 표기 반올림과 맞추기 위해 half-up."""
    return math.floor(x * 10 + 0.5 + 1e-9) / 10


def v1():
    out = {}
    for m, (mu, sd, (avg, avgsd)) in T6.items():
        mu, sd = np.array(mu), np.array(sd)
        row = {}
        for gi, (g, idx) in enumerate(GROUP.items()):
            calc_m = r1(mu[idx].mean())
            calc_s = r1(sd[idx].mean())
            row[g] = {"mean_calc": calc_m, "mean_T1": T1[m][gi][0], "sd_mean_calc": calc_s, "sd_T1": T1[m][gi][1]}
        row["Avg"] = {"mean_calc": r1(mu.mean()), "mean_T6": avg, "mean_T1": T1[m][3][0],
                      "sd_mean_of_task_sd": r1(sd.mean()), "sd_T6": avgsd,
                      "sd_if_independent_tasks": round(float(np.sqrt((sd ** 2).sum()) / len(sd)), 2),
                      "mean_of_3_group_means": r1(np.mean([T1[m][i][0] for i in range(3)]))}
        out[m] = row
    # 본문 차이 (12.1, 9.7, 7.0, 2.9) 와 그룹별 CordViP 대비 (4.7, 1.9, 2.3)
    px = T1["ProxiDex"][3][0]
    out["diff_text_overall"] = {m: round(px - T1[m][3][0], 1) for m in ["DP3", "ManiFlow", "AFRO", "CordViP"]}
    out["diff_vs_cordvip_group"] = {g: round(T1["ProxiDex"][i][0] - T1["CordViP"][i][0], 1) for i, g in enumerate(GROUP)}
    # 과제별 최고
    best = {}
    for j, t in enumerate(T6_TASKS):
        vals = {m: T6[m][0][j] for m in T6}
        top = max(vals.values())
        best[t] = [m for m, v in vals.items() if v == top]
    out["best_per_task"] = best
    out["proxidex_not_best_tasks"] = [t for t, b in best.items() if "ProxiDex" not in b]
    wins = sum(1 for j in range(10) if T6["ProxiDex"][0][j] > T6["CordViP"][0][j])
    ties = sum(1 for j in range(10) if T6["ProxiDex"][0][j] == T6["CordViP"][0][j])
    out["proxidex_vs_cordvip_tasks"] = {"wins": wins, "ties": ties, "losses": 10 - wins - ties}
    d = np.array(T6["ProxiDex"][0]) - np.array(T6["CordViP"][0])
    out["paired_diffs_vs_cordvip"] = [round(float(x), 1) for x in d]
    out["sign_test_p_two_sided_excl_ties"] = stats.binomtest(wins, wins + (10 - wins - ties), 0.5).pvalue
    w = stats.wilcoxon(d, zero_method="wilcox", method="exact")
    out["wilcoxon_p_two_sided"] = float(w.pvalue)
    out["mean_diff"] = round(float(d.mean()), 2)
    out["diff_sd_across_tasks"] = round(float(d.std(ddof=1)), 2)
    out["t_test_p_across_tasks"] = float(stats.ttest_1samp(d, 0).pvalue)
    # 가중이 없는 3묶음 평균과 10과제 평균의 차이
    return out


# ---------------------------------------------------------------- 표 7, 8
T7_TASKS = ["Door", "Pen", "Laptop", "Faucet", "Toilet", "Bucket"]
T7 = {
    "w/o Prox": ([76.7, 79.4, 88.3, 47.9, 80.7, 50.6], [2.3, 1.6, 1.5, 1.5, 1.6, 1.3], (70.6, 1.6)),
    "w/o Int.PC": ([80.3, 75.9, 86.1, 48.7, 80.4, 48.9], [1.2, 2.4, 1.4, 1.4, 1.5, 2.0], (70.1, 1.7)),
    "w/o FDM": ([83.8, 80.3, 89.6, 50.5, 81.1, 56.4], [2.2, 1.7, 1.5, 1.7, 1.7, 1.8], (73.6, 1.8)),
    "w/o IDM": ([84.7, 80.9, 92.3, 51.8, 82.0, 60.1], [1.7, 2.1, 1.7, 1.3, 1.8, 1.5], (75.3, 1.7)),
    "w/o Closed-Loop": ([85.2, 82.5, 92.8, 52.6, 83.4, 62.3], [1.2, 1.2, 1.3, 1.1, 2.0, 1.6], (76.5, 1.4)),
    "ProxiDex": ([86.5, 84.3, 93.0, 53.4, 84.3, 64.0], [1.9, 2.4, 1.2, 2.1, 2.3, 3.2], (77.6, 2.2)),
}
T8 = {
    1: ([86.8, 84.0, 92.6, 53.3, 80.7, 64.1], [1.7, 2.2, 1.5, 1.9, 1.6, 1.8], (76.9, 1.8)),
    2: ([86.5, 84.3, 93.0, 53.4, 84.3, 64.0], [1.9, 2.4, 1.2, 2.1, 2.3, 3.2], (77.6, 2.2)),
    4: ([85.9, 83.6, 93.0, 53.7, 81.1, 64.3], [2.0, 1.9, 1.5, 1.4, 1.7, 1.9], (76.9, 1.7)),
    8: ([84.7, 82.8, 92.1, 53.4, 82.0, 63.9], [1.1, 1.8, 1.4, 1.5, 1.8, 1.7], (76.5, 1.6)),
    10: ([84.8, 82.6, 91.8, 52.8, 83.4, 63.2], [1.2, 1.3, 1.1, 1.3, 2.0, 2.0], (76.4, 1.5)),
}


def v2():
    out = {"T7": {}, "T8": {}}
    for k, (mu, sd, (avg, avgsd)) in T7.items():
        out["T7"][k] = {"mean_calc": r1(np.mean(mu)), "mean_T7": avg, "sd_mean_calc": r1(np.mean(sd)), "sd_T7": avgsd}
    full = np.mean(T7["ProxiDex"][0])
    out["T7_drops_calc_from_rounded_table_avg"] = {k: round(77.6 - v[2][0], 1) for k, v in T7.items() if k != "ProxiDex"}
    out["T7_drops_calc_from_unrounded"] = {k: round(float(full - np.mean(v[0])), 2) for k, v in T7.items() if k != "ProxiDex"}
    out["text_drops"] = {"w/o Int.PC": 7.5, "w/o Prox": 7.0, "w/o FDM": 4.0, "w/o IDM": 2.3, "w/o Closed-Loop": 1.1}
    out["Bucket_drops"] = {"w/o Int.PC": round(64.0 - 48.9, 1), "w/o Prox": round(64.0 - 50.6, 1)}
    out["Pen_drop_wo_IntPC"] = round(84.3 - 75.9, 1)
    for h, (mu, sd, (avg, avgsd)) in T8.items():
        out["T8"][h] = {"mean_calc": r1(np.mean(mu)), "mean_T8": avg, "sd_mean_calc": r1(np.mean(sd)), "sd_T8": avgsd}
    out["T8_H2_equals_T7_full"] = T8[2][0] == T7["ProxiDex"][0] and T8[2][1] == T7["ProxiDex"][1]
    out["T7_full_equals_T6_cols"] = all(T7["ProxiDex"][0][i] == T6["ProxiDex"][0][T6_TASKS.index(t)] for i, t in enumerate(T7_TASKS))
    # 표 7 w/o 행이 1~2 점 사이 단조 순서인지
    out["T7_monotone_order_all_tasks"] = all(
        T7["w/o Prox"][0][i] <= T7["w/o FDM"][0][i] <= T7["w/o IDM"][0][i] <= T7["w/o Closed-Loop"][0][i] <= T7["ProxiDex"][0][i]
        for i in range(6))
    # 표 7: 절제가 전부 ProxiDex 이하인지
    out["T7_full_is_best_every_task"] = all(T7["ProxiDex"][0][i] >= max(T7[k][0][i] for k in T7) for i in range(6))
    return out


# ---------------------------------------------------------------- 표 2 실로봇
SET = ["ID-Pick", "ID-Pinch", "ID-Sweep", "UO-Pick", "UO-Pinch", "UO-Sweep", "PT-Pick", "PT-Pinch", "PT-Sweep", "Twist", "Flip"]
N2 = [20, 20, 20, 40, 40, 40, 30, 30, 30, 20, 20]
T2 = {
    "DP3": [14, 10, 12, 18, 9, 16, 14, 13, 14, 2, 2],
    "ManiFlow": [16, 11, 15, 24, 12, 19, 17, 18, 17, 2, 3],
    "AFRO": [17, 11, 16, 25, 15, 24, 17, 17, 16, 3, 5],
    "CordViP": [18, 13, 17, 29, 21, 27, 19, 19, 18, 6, 8],
    "ProxiDex-1": [18, 14, 17, 30, 22, 28, 20, 20, 20, 7, 6],
    "ProxiDex-2": [18, 14, 18, 31, 23, 28, 21, 22, 21, 8, 7],
    "ProxiDex": [19, 15, 18, 32, 24, 29, 22, 23, 22, 10, 11],
}
T2_AVG = {"DP3": 40.0, "ManiFlow": 49.7, "AFRO": 53.5, "CordViP": 62.9, "ProxiDex-1": 65.2, "ProxiDex-2": 68.1, "ProxiDex": 72.6}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cmh(a_list, b_list, n_list):
    """층화 2x2 CMH (연속성 보정 없음). 군 A=ProxiDex, B=비교 방법, 각 층 n 시행."""
    num, var = 0.0, 0.0
    for a, b, n in zip(a_list, b_list, n_list):
        N = 2 * n
        m1 = a + b  # 성공 합
        e = n * m1 / N
        v = n * n * m1 * (N - m1) / (N * N * (N - 1))
        num += a - e
        var += v
    chi = num * num / var
    return chi, float(stats.chi2.sf(chi, 1))


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def v3_v4():
    out = {"pooled": {}, "avg_check": {}, "text": {}, "fisher_vs_cordvip": {}, "cmh": {}}
    tot = sum(N2)
    out["total_trials_per_method"] = tot
    for m, k in T2.items():
        s = sum(k)
        out["pooled"][m] = {"success": s, "n": tot, "rate_pct": round(100 * s / tot, 2),
                            "wilson95": [round(100 * x, 1) for x in wilson(s, tot)]}
        out["avg_check"][m] = {"calc": r1(100 * s / tot), "T2": T2_AVG[m],
                               "mean_of_11_cell_rates": r1(np.mean([100 * a / n for a, n in zip(k, N2)]))}
    # 본문 인용 칸
    px, cv = T2["ProxiDex"], T2["CordViP"]
    out["text"] = {
        "unseen_ProxiDex": sum(px[3:6]), "unseen_CordViP": sum(cv[3:6]),
        "perturb_ProxiDex": sum(px[6:9]), "perturb_CordViP": sum(cv[6:9]),
        "text_unseen": "85/120 vs 77/120", "text_perturb": "67/90 vs 56/90",
        "gaps_text": {m: round(T2_AVG["ProxiDex"] - T2_AVG[m], 1) for m in ["CordViP", "AFRO", "ManiFlow", "DP3"]},
        "backbone_gain_text": {"vs ProxiDex-1": round(72.6 - 65.2, 1), "vs ProxiDex-2": round(72.6 - 68.1, 1)},
    }
    # 최고·공동 최고 칸 수 (기준선 4개 대비, 변형 2개 포함 대비)
    def best_cells(methods):
        return sum(1 for j in range(11) if T2["ProxiDex"][j] >= max(T2[m][j] for m in methods))
    out["best_or_tied_cells_vs_baselines"] = best_cells(["DP3", "ManiFlow", "AFRO", "CordViP"])
    out["best_or_tied_cells_vs_all_rows"] = best_cells([m for m in T2 if m != "ProxiDex"])
    out["strict_best_cells_vs_all_rows"] = sum(1 for j in range(11) if T2["ProxiDex"][j] > max(T2[m][j] for m in T2 if m != "ProxiDex"))
    # 검정: 설정별 Fisher (ProxiDex vs 각 기준선), 합산 Fisher, CMH
    for m in ["DP3", "ManiFlow", "AFRO", "CordViP", "ProxiDex-1", "ProxiDex-2"]:
        per = []
        for j in range(11):
            a, b, n = px[j], T2[m][j], N2[j]
            per.append(stats.fisher_exact([[a, n - a], [b, n - b]])[1])
        pooled = stats.fisher_exact([[sum(px), tot - sum(px)], [sum(T2[m]), tot - sum(T2[m])]])
        chi, pc = cmh(px, T2[m], N2)
        wins = sum(1 for j in range(11) if px[j] > T2[m][j])
        ties = sum(1 for j in range(11) if px[j] == T2[m][j])
        loss = 11 - wins - ties
        sign_p = stats.binomtest(wins, wins + loss, 0.5).pvalue if wins + loss else 1.0
        # 풀링 차이의 Newcombe-Wilson 구간
        p1, p2 = sum(px) / tot, sum(T2[m]) / tot
        l1, u1 = wilson(sum(px), tot)
        l2, u2 = wilson(sum(T2[m]), tot)
        d = p1 - p2
        lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
        hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
        out["fisher_vs_cordvip"][m] = {
            "per_setting_p": [round(x, 4) for x in per],
            "per_setting_holm": [round(x, 4) for x in holm(per)],
            "n_setting_p_lt_0.05_raw": sum(1 for x in per if x < 0.05),
            "n_setting_p_lt_0.05_holm": sum(1 for x in holm(per) if x < 0.05),
            "pooled_diff_pp": round(100 * d, 1), "pooled_diff_ci95_pp": [round(100 * lo, 1), round(100 * hi, 1)],
            "pooled_fisher_p": pooled[1], "pooled_odds_ratio": pooled[0],
            "cmh_chi2": round(chi, 2), "cmh_p": pc,
            "cells_win_tie_loss": [wins, ties, loss], "sign_test_p": sign_p,
        }
    # 검출력: 합산 310 대 310, p=0.726 대 0.629 (정규 근사)
    p1, p2, n = 0.726, 0.629, 310
    pbar = (p1 + p2) / 2
    se0 = math.sqrt(2 * pbar * (1 - pbar) / n)
    se1 = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    z = (abs(p1 - p2) - 1.959964 * se0) / se1
    out["power_pooled_vs_cordvip_normal_approx"] = float(stats.norm.cdf(z))
    out["n_per_arm_for_80pct_power"] = math.ceil(((1.959964 * math.sqrt(2 * pbar * (1 - pbar)) + 0.841621 * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / (p1 - p2)) ** 2)
    # 설정별 20회 한 칸에서 80% 검정력으로 가를 수 있는 차이(두 비율 0.6 부근)
    out["per_cell_n20_mde_pp_at_p0.5"] = round(100 * (1.959964 + 0.841621) * math.sqrt(2 * 0.25 / 20), 1)
    # Wuji 과제 두 칸 (Twist, Flip): 합산 21/40 대 14/40
    out["contact_rich_pooled"] = {
        "ProxiDex": sum(px[9:]), "CordViP": sum(cv[9:]), "n": 40,
        "fisher_p": stats.fisher_exact([[sum(px[9:]), 40 - sum(px[9:])], [sum(cv[9:]), 40 - sum(cv[9:])]])[1]}
    # 변형 (U-Net, 표준 Transformer) 대 전체 모델: 실로봇 백본 절제
    for m in ["ProxiDex-1", "ProxiDex-2"]:
        out["contact_rich_flip_" + m] = f"{T2[m][10]}/20"
    return out


# ---------------------------------------------------------------- 시뮬 시드 수 가정
def v5():
    out = {}
    mu_p, sd_p = np.array(T6["ProxiDex"][0]), np.array(T6["ProxiDex"][1])
    mu_c, sd_c = np.array(T6["CordViP"][0]), np.array(T6["CordViP"][1])
    for k in (3, 5, 10):
        # 과제별 Welch t (SD는 표본 SD라 가정)
        ps = []
        for j in range(10):
            se = math.sqrt(sd_p[j] ** 2 / k + sd_c[j] ** 2 / k)
            if se == 0:
                ps.append(1.0 if mu_p[j] == mu_c[j] else 0.0)
                continue
            t = (mu_p[j] - mu_c[j]) / se
            df = se ** 4 / ((sd_p[j] ** 2 / k) ** 2 / (k - 1) + (sd_c[j] ** 2 / k) ** 2 / (k - 1))
            ps.append(float(2 * stats.t.sf(abs(t), df)))
        # 10과제 평균: 과제가 독립이라 가정한 SD, 그리고 표의 Avg ± 칸을 SD로 본 경우
        sd_avg_p, sd_avg_c = math.sqrt((sd_p ** 2).sum()) / 10, math.sqrt((sd_c ** 2).sum()) / 10
        d = mu_p.mean() - mu_c.mean()
        def welch(d, s1, s2):
            se = math.sqrt(s1 ** 2 / k + s2 ** 2 / k)
            t = d / se
            df = se ** 4 / ((s1 ** 2 / k) ** 2 / (k - 1) + (s2 ** 2 / k) ** 2 / (k - 1))
            return float(2 * stats.t.sf(abs(t), df))
        out[f"seeds_{k}"] = {
            "per_task_p": [round(x, 4) for x in ps],
            "per_task_holm": [round(x, 4) for x in holm(ps)],
            "tasks_p_lt_0.05_raw": [T6_TASKS[j] for j in range(10) if ps[j] < 0.05],
            "tasks_p_lt_0.05_holm": [T6_TASKS[j] for j, x in enumerate(holm(ps)) if x < 0.05],
            "avg_diff": round(float(d), 2),
            "avg_p_if_sd_is_independent_tasks_combined": welch(d, sd_avg_p, sd_avg_c),
            "avg_p_if_table_avg_sd_is_the_sd": welch(d, 1.8, 1.9),
        }
    # 최소 시드 수: 표의 Avg ± 를 SD로 볼 때 p<0.05가 되는 시드 수
    need = None
    for k in range(2, 40):
        se = math.sqrt(1.8 ** 2 / k + 1.9 ** 2 / k)
        df = 2 * (k - 1)
        if 2 * stats.t.sf(2.9 / se, df) < 0.05:
            need = k
            break
    out["min_seeds_for_p_lt_0.05_avg_sd_1.8_1.9"] = need
    return out


# ---------------------------------------------------------------- 그림에서 읽은 값의 정수 가능성
FIG5 = {  # 그림 5 (Adroit Pen 가림 회복 성공률, %, 막대 위 숫자)
    "w/o FDM": [64, 52, 12], "w/o IDM": [71, 59, 23], "w/o Prox": [74, 63, 25],
    "w/o Trajectory Gating": [77, 68, 27], "ProxiDex": [80, 72, 35],
}
FIG9 = {"PointCloud": [85, 70, 85], "PointCloud+Proximity": [95, 75, 90], "PointCloud+Tactile": [96, 75, 92]}


def feasible_n(vals, nmax=200):
    return [n for n in range(1, nmax + 1) if all(abs(v * n / 100 - round(v * n / 100)) < 1e-9 for v in vals)]


def v6():
    out = {}
    all5 = [v for row in FIG5.values() for v in row]
    out["fig5_all_values"] = all5
    out["fig5_n_feasible_all_cells_le_200"] = feasible_n(all5)
    out["fig5_n_feasible_proxidex_row"] = feasible_n(FIG5["ProxiDex"])[:8]
    out["fig5_min_n_each_row"] = {k: feasible_n(v)[0] for k, v in FIG5.items()}
    out["fig5_arrow_labels_check"] = {"2s": 80 - 64, "4s": 72 - 52, "6s": 35 - 12}
    n = 100
    out["fig5_assumed_n"] = n
    fish = {}
    for i, dur in enumerate(["2s", "4s", "6s"]):
        full = FIG5["ProxiDex"][i]
        fish[dur] = {}
        for k, row in FIG5.items():
            if k == "ProxiDex":
                continue
            p = stats.fisher_exact([[full, n - full], [row[i], n - row[i]]])[1]
            fish[dur][k] = round(float(p), 4)
    out["fig5_fisher_full_vs_each_n100"] = fish
    out["fig5_wilson_full_n100"] = {d: [round(100 * x, 1) for x in wilson(FIG5["ProxiDex"][i], n)] for i, d in enumerate(["2s", "4s", "6s"])}
    out["fig5_wo_idm_wo_prox_wo_gating_vs_full_6s_diff"] = {k: 35 - FIG5[k][2] for k in FIG5 if k != "ProxiDex"}
    all9 = [v for row in FIG9.values() for v in row]
    out["fig9_values"] = FIG9
    out["fig9_n_feasible_all_cells_le_200"] = feasible_n(all9)
    out["fig9_n_feasible_tactile_row"] = feasible_n(FIG9["PointCloud+Tactile"])[:6]
    out["fig9_proxidex_row_equals_T2_in_distribution_pct"] = [round(100 * T2["ProxiDex"][j] / 20) for j in range(3)]
    out["fig9_pointcloud_row_matches_T2_row"] = {m: [round(100 * T2[m][j] / 20) for j in range(3)] for m in T2
                                                  if [round(100 * T2[m][j] / 20) for j in range(3)] == FIG9["PointCloud"]}
    out["fig9_prox_vs_tactile_diff_pp"] = [b - a for a, b in zip(FIG9["PointCloud+Proximity"], FIG9["PointCloud+Tactile"])]
    return out


# ---------------------------------------------------------------- Hdyn 절제에서 Toilet 열 제외
def v7():
    out = {}
    tasks = T7_TASKS
    no_toilet = [i for i, t in enumerate(tasks) if t != "Toilet"]
    out["T8_avg_all6"] = {h: r1(np.mean(v[0])) for h, v in T8.items()}
    out["T8_avg_excl_Toilet"] = {h: round(float(np.mean([v[0][i] for i in no_toilet])), 2) for h, v in T8.items()}
    out["T8_toilet_col"] = {h: v[0][tasks.index("Toilet")] for h, v in T8.items()}
    out["T8_H2_minus_H1_all6"] = round(float(np.mean(T8[2][0]) - np.mean(T8[1][0])), 2)
    out["T8_H2_minus_H1_excl_Toilet"] = round(float(np.mean([T8[2][0][i] for i in no_toilet]) - np.mean([T8[1][0][i] for i in no_toilet])), 2)
    out["T8_toilet_share_of_H2_minus_H1"] = round((84.3 - 80.7) / 6 / (np.mean(T8[2][0]) - np.mean(T8[1][0])), 2)
    out["T8_row_range_excl_Toilet"] = [min(out["T8_avg_excl_Toilet"].values()), max(out["T8_avg_excl_Toilet"].values())]
    out["T8_spread_pp_all6"] = round(max(out["T8_avg_all6"].values()) - min(out["T8_avg_all6"].values()), 2)
    # 표 8 열 SD 범위: 각 칸 SD는 1–3 정도라 차이 0.7을 가르려면?
    sds = [s for v in T8.values() for s in v[1]]
    out["T8_cell_sd_median"] = float(np.median(sds))
    # 평균의 평균 차이 0.7 (76.9 → 77.6)를 구분하는 데 필요한 시드 수: 칸 SD가 2 정도일 때 6과제 평균 SD 약 0.8
    sd_avg = math.sqrt(np.mean(np.array([v for v in T8[2][1]]) ** 2) / 6)
    out["T8_sd_of_6task_avg_if_independent"] = round(float(sd_avg), 2)
    return out


# ---------------------------------------------------------------- 같은 mean±SD 칸 찾기
def v8():
    cells = []  # (table, row, col, mean, sd)
    for m, (mu, sd, _) in T6.items():
        for j, t in enumerate(T6_TASKS):
            cells.append(("T6", m, t, mu[j], sd[j]))
    for m, (mu, sd, _) in T7.items():
        for j, t in enumerate(T7_TASKS):
            cells.append(("T7", m, t, mu[j], sd[j]))
    for h, (mu, sd, _) in T8.items():
        for j, t in enumerate(T7_TASKS):
            cells.append(("T8", f"Hdyn={h}", t, mu[j], sd[j]))
    groups = {}
    for c in cells:
        groups.setdefault((c[3], c[4]), []).append(c)
    dup = []
    for (mu, sd), cs in groups.items():
        if len(cs) < 2 or (mu, sd) == (100.0, 0.0):
            continue
        # 의도된 재사용(ProxiDex 행이 표 6·7·8에 같이 나옴)은 제외
        labels = [(c[0], c[1], c[2]) for c in cs]
        px = [l for l in labels if l[1] in ("ProxiDex", "Hdyn=2")]
        others = [l for l in labels if l[1] not in ("ProxiDex", "Hdyn=2")]
        if len(others) >= 2 or (others and px and False):
            dup.append({"value": f"{mu}±{sd}", "cells": [f"{a} {b} {c}" for a, b, c in others]})
        elif len(others) == 1 and len(px) >= 1 and not all(l[2] == others[0][2] and True for l in px):
            pass
    # 열(과제) 단위로 같은 값이 서로 다른 표의 서로 다른 행에서 반복되는 경우만 모은다
    out = {"duplicate_mean_sd_cells_excl_proxidex_rows": dup}
    # Toilet 열 한정: T8(H=1,4,8,10) ↔ T7(w/o Prox, w/o FDM, w/o IDM, w/o Closed-Loop)
    t7_toilet = {k: (T7[k][0][4], T7[k][1][4]) for k in T7}
    t8_toilet = {h: (T8[h][0][4], T8[h][1][4]) for h in T8}
    pairs = []
    for h, v in t8_toilet.items():
        for k, w in t7_toilet.items():
            if v == w and k != "ProxiDex":
                pairs.append(f"T8 Hdyn={h} Toilet {v[0]}±{v[1]} = T7 {k} Toilet")
    out["toilet_column_pairs_T8_vs_T7"] = pairs
    # 그 밖의 열에서도 T8 행과 T7 행이 같은 칸이 있는지
    other_pairs = []
    for h in T8:
        for k in T7:
            if k == "ProxiDex" or h == 2:
                continue
            for j, t in enumerate(T7_TASKS):
                if t == "Toilet":
                    continue
                if (T8[h][0][j], T8[h][1][j]) == (T7[k][0][j], T7[k][1][j]):
                    other_pairs.append(f"T8 Hdyn={h} {t} = T7 {k} {t}")
    out["other_column_pairs_T8_vs_T7"] = other_pairs
    # 표 6 기준선과 표 7 절제 사이
    cross = []
    for m in ["DP3", "ManiFlow", "AFRO", "CordViP"]:
        for k in T7:
            if k == "ProxiDex":
                continue
            for j, t in enumerate(T7_TASKS):
                if (T6[m][0][T6_TASKS.index(t)], T6[m][1][T6_TASKS.index(t)]) == (T7[k][0][j], T7[k][1][j]):
                    cross.append(f"T6 {m} {t} = T7 {k} {t} ({T7[k][0][j]}±{T7[k][1][j]})")
    out["T6_baseline_vs_T7_ablation_same_cells"] = cross
    # 우연 일치 기준선: 평균(0.1 간격)±SD(0.1 간격) 한 칸이 맞을 확률의 규모. 평균 범위 약 40 점(400값), SD 약 20값
    out["rough_chance_of_one_cell_match_per_pair"] = round(1 / (400 * 20), 6)
    out["n_cells_T6_T7_T8_excl_proxidex_rows"] = len([c for c in cells if c[1] not in ("ProxiDex", "Hdyn=2")])
    return out


# ---------------------------------------------------------------- 부록 D.3, 사용자 연구
def v9():
    out = {}
    curve = {5: 11.5, 10: 40.0, 20: 64.0, 30: 77.5, 50: 80.0, 100: 81.8}
    out["increments"] = {"5to10": curve[10] - curve[5], "10to20": curve[20] - curve[10], "20to30": curve[30] - curve[20], "30to100": round(curve[100] - curve[30], 1)}
    out["frac_30_of_100_pct"] = round(100 * curve[30] / curve[100], 1)
    out["30demo_vs_T7_full_avg"] = (curve[30], 77.6)
    out["integer_feasible_n_curve_values"] = feasible_n(list(curve.values()), 400)[:5]
    # 사용자 연구 (n=20)
    pies = {"helpfulness": {"very": 50, "sometimes": 45, "not": 5}, "preference": {"ProxiDex": 75, "VanillaVR": 20, "3Dmouse+glove": 5}}
    out["user_study_counts_n20"] = {k: {a: v * 20 / 100 for a, v in d.items()} for k, d in pies.items()}
    out["user_study_sum_pct"] = {k: sum(d.values()) for k, d in pies.items()}
    # 20명 중 15명 선호 (75%)의 Wilson 구간, 귀무 1/3
    out["pref_15_of_20_wilson95_pct"] = [round(100 * x, 1) for x in wilson(15, 20)]
    out["pref_15_of_20_binom_p_vs_1/3"] = stats.binomtest(15, 20, 1 / 3).pvalue
    return out


# ---------------------------------------------------------------- 학습 스케줄 대 학습 단계 수
def v10():
    out = {}
    warm, ramp, anneal = 2000, 20000, 50000
    out["schedule_steps"] = {"warmup": warm, "alpha_ramp_end": warm + ramp, "tau_anneal": anneal}
    rows = []
    for demos, label in [(30, "시뮬 30 시연"), (50, "실로봇 50 시연")]:
        for L in (100, 200, 400, 800):
            samples = demos * L
            steps_per_epoch = math.ceil(samples / 128)
            total = 301 * steps_per_epoch
            rows.append({"demos": demos, "L_frames": L, "samples": samples, "steps_per_epoch_b128": steps_per_epoch, "total_steps_301ep": total,
                         "alpha_ramp_done": total >= warm + ramp, "tau_anneal_done": total >= anneal,
                         "alpha_at_end": round(min(1.0, max(0.0, (total - warm) / ramp)), 2)})
    out["rows"] = rows
    # 총 단계 수가 스케줄 끝에 닿기 위한 최소 프레임 수 (시연 수 30, 배치 128, 301 에폭)
    out["min_frames_for_ramp_done"] = math.ceil((warm + ramp) / 301) * 128
    out["min_frames_for_anneal_done"] = math.ceil(anneal / 301) * 128
    out["min_mean_len_30demos_for_ramp_done"] = round(out["min_frames_for_ramp_done"] / 30, 1)
    out["min_mean_len_30demos_for_anneal_done"] = round(out["min_frames_for_anneal_done"] / 30, 1)
    # 게이트: g(p)=1-a+a p, a 초기 0이면 g=1, 즉 램프 초반은 게이팅이 없음
    out["gate_values"] = {"alpha0": [1.0, 1.0, 1.0], "alpha0.5_p0_0.5_1": [round(1 - 0.5 + 0.5 * p, 3) for p in (0, 0.5, 1)], "alpha1_p0_0.5_1": [0.0, 0.5, 1.0]}
    # 추론 속도 산술: 정책 28 Hz, FDM 20 Hz, FoundationPose++ 10 Hz
    out["rates_hz"] = {"policy": 28, "fdm": 20, "pose": 10, "teleop_pointcloud": 23, "teleop_hand": 30}
    out["pose_updates_per_policy_step"] = round(10 / 28, 2)
    out["dream_horizon_s_to_fdm_steps_at_20Hz"] = round(3.5 * 20)
    out["dream_horizon_s_to_policy_calls_at_28Hz"] = round(3.5 * 28)
    return out


def v11():
    out = {}
    px, cv = T2["ProxiDex"], T2["CordViP"]
    for name, sl in (("unseen", slice(3, 6)), ("perturb", slice(6, 9)), ("in_dist", slice(0, 3))):
        a, b = sum(px[sl]), sum(cv[sl])
        n = sum(N2[sl])
        out[name] = {"ProxiDex": f"{a}/{n}", "CordViP": f"{b}/{n}", "diff_pp": round(100 * (a - b) / n, 1),
                     "fisher_p": stats.fisher_exact([[a, n - a], [b, n - b]])[1]}
    # 표 7: 각 절제 행 대 ProxiDex 행, 평균 열 차이와 Avg ± 를 SD로 볼 때 p<0.05가 되는 최소 시드 수
    need = {}
    for k_, (mu, sd, (avg, avgsd)) in T7.items():
        if k_ == "ProxiDex":
            continue
        d = 77.6 - avg
        s1, s2 = avgsd, 2.2
        kmin = None
        for k in range(2, 200):
            se = math.sqrt(s1 ** 2 / k + s2 ** 2 / k)
            df = se ** 4 / ((s1 ** 2 / k) ** 2 / (k - 1) + (s2 ** 2 / k) ** 2 / (k - 1))
            if 2 * stats.t.sf(d / se, df) < 0.05:
                kmin = k
                break
        need[k_] = {"drop": round(d, 1), "min_seeds_p_lt_0.05": kmin}
    out["T7_min_seeds_avg_sd_as_sd"] = need
    # 표 7 과제 단위 부호 검정: 각 절제 행이 ProxiDex보다 낮은 과제 수
    out["T7_tasks_lower_than_full"] = {k_: sum(1 for i in range(6) if v[0][i] < T7["ProxiDex"][0][i]) for k_, v in T7.items() if k_ != "ProxiDex"}
    return out


def main():
    R["V1_table1_vs_table6"] = v1()
    R["V2_ablation_tables"] = v2()
    R["V3_V4_real_robot_table2"] = v3_v4()
    R["V5_sim_gap_seed_assumption"] = v5()
    R["V6_figure_read_feasibility"] = v6()
    R["V7_hdyn_toilet"] = v7()
    R["V8_identical_cells"] = v8()
    R["V9_scaling_user_study"] = v9()
    R["V10_schedule_vs_steps"] = v10()
    R["V11_subsets_and_ablation_power"] = v11()
    OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(R, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
