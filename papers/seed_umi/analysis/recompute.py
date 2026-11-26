"""SEED-UMI (arXiv:2609.11753 v1) 표 1과 효율 수치를 다시 계산하고 검정한다.

입력: 논문 표 1(5과제 x 3정책 x 3조건)의 값을 아래에 그대로 옮겼다. 외부 데이터는 쓰지 않는다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 표 1 캡션이 "20 autonomous robot rollouts per entry"라고 하므로 각 칸의 분모는 20이다(성공 수 = 비율 x 20).
- 세 정책(ACT, DP, pi0.5)은 같은 과제·같은 시연 100개로 학습되어 서로 독립이 아니다.
  합산(풀링) 검정은 독립을 가정한 것이라 낙관적이다. 정책을 층으로 둔 CMH 검정을 함께 낸다.
- P-와 P+가 같은 정책 체크포인트에 다른 매핑을 얹은 것인지, 따로 학습한 것인지 논문에 없다.
  그래서 P- 대 P+ 검정은 독립 표본(Fisher)과 층화(CMH) 두 가지로 낸다. 짝지은 검정(McNemar)은 롤아웃별 결과가 없어서 못 한다.
- 효율(그림 7, F3): 한 명의 작업자, 한 과제, 30분 한 번이다(18개 대 52개). 반복이 없어 분산을 알 수 없다.
"""
import json
import math
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
N = 20
TASKS = ["Screw Driving", "AirPods Case", "Ball Throwing", "Table Cleaning", "Air Freshener"]
POLICIES = ["ACT", "DP", "pi0.5"]
CONDS = ["T", "P-", "P+"]
# 표 1 (%). 행: 정책, 열: 과제 순서 TASKS, 조건 T / P- / P+
TABLE1 = {
    "ACT":   {"T": [80, 65, 60, 75, 90], "P-": [55, 65, 60, 50, 70], "P+": [70, 75, 70, 65, 85]},
    "DP":    {"T": [65, 55, 50, 60, 70], "P-": [40, 55, 50, 40, 45], "P+": [55, 60, 55, 55, 60]},
    "pi0.5": {"T": [90, 75, 70, 85, 85], "P-": [60, 75, 70, 60, 65], "P+": [80, 85, 80, 75, 80]},
}
# 본문 보고값(%)
REPORTED = {
    "overall": {"T": 71.7, "P-": 57.3, "P+": 70.0},
    "task_P+_vs_T": {"AirPods Case": (73.3, 65.0), "Ball Throwing": (68.3, 60.0)},
    "lift_P+_minus_P-": {"overall": 12.7, "Screw Driving": 16.7, "Table Cleaning": 15.0, "Air Freshener": 15.0},
}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe_diff(k1, n1, k2, n2, z=1.959964):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def cmh(strata):
    """strata: [(k1,n1,k2,n2)...]. Mantel-Haenszel 공통 오즈비와 연속성 보정 없는 CMH 검정 (양측)."""
    num = den = var = obs = exp = 0.0
    r_num = r_den = 0.0
    for k1, n1, k2, n2 in strata:
        a, b, c, d = k1, n1 - k1, k2, n2 - k2
        t = n1 + n2
        m1 = a + c
        exp += n1 * m1 / t
        obs += a
        var += n1 * n2 * m1 * (t - m1) / (t * t * (t - 1))
        r_num += a * d / t
        r_den += b * c / t
    chi2 = (obs - exp) ** 2 / var
    return {"or_mh": r_num / r_den if r_den else float("inf"), "chi2": chi2, "p": float(stats.chi2.sf(chi2, 1))}


def power_fisher(p1, p2, n, sims=4000, seed=0):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(sims):
        k1, k2 = rng.binomial(n, p1), rng.binomial(n, p2)
        hits += fisher(int(k1), n, int(k2), n) < 0.05
    return hits / sims


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def main():
    res = {"assumptions": {"n_per_cell": N}}
    # 1. 격자 가능성과 성공 수
    counts = {pol: {c: [int(round(v * N / 100)) for v in TABLE1[pol][c]] for c in CONDS} for pol in POLICIES}
    res["integer_feasible_all_cells"] = all(abs(v * N / 100 - round(v * N / 100)) < 1e-9
                                            for pol in POLICIES for c in CONDS for v in TABLE1[pol][c])
    res["counts"] = counts

    # 2. 평균 재계산
    cond_task = {c: [np.mean([TABLE1[p][c][j] for p in POLICIES]) for j in range(5)] for c in CONDS}
    cond_policy = {c: {p: float(np.mean(TABLE1[p][c])) for p in POLICIES} for c in CONDS}
    cond_overall = {c: float(np.mean(cond_task[c])) for c in CONDS}
    res["mean_by_task"] = {c: [round(x, 2) for x in cond_task[c]] for c in CONDS}
    res["mean_by_policy"] = cond_policy
    res["mean_overall"] = {c: round(v, 2) for c, v in cond_overall.items()}
    res["overall_vs_reported"] = {c: {"recomputed": round(cond_overall[c], 2), "reported": REPORTED["overall"][c],
                                      "match_1dp": round(cond_overall[c], 1) == REPORTED["overall"][c]} for c in CONDS}
    lift_task = [cond_task["P+"][j] - cond_task["P-"][j] for j in range(5)]
    res["lift_P+_minus_P-_by_task"] = {t: round(x, 2) for t, x in zip(TASKS, lift_task)}
    res["lift_P+_minus_P-_overall"] = round(cond_overall["P+"] - cond_overall["P-"], 2)
    res["P+_minus_T_by_task"] = {t: round(cond_task["P+"][j] - cond_task["T"][j], 2) for j, t in enumerate(TASKS)}
    res["P-_minus_T_by_task"] = {t: round(cond_task["P-"][j] - cond_task["T"][j], 2) for j, t in enumerate(TASKS)}
    res["P+_minus_T_overall"] = round(cond_overall["P+"] - cond_overall["T"], 2)

    # 3. 칸 단위 관찰: P- = T 가 모든 정책에서 같은 과제, P+ - P- 분포
    same = {t: all(TABLE1[p]["P-"][j] == TABLE1[p]["T"][j] for p in POLICIES) for j, t in enumerate(TASKS)}
    res["P-_equals_T_in_all_three_policies"] = same
    res["cells_P-_equals_T"] = sum(TABLE1[p]["P-"][j] == TABLE1[p]["T"][j] for p in POLICIES for j in range(5))
    res["cells_P+_gt_T"] = sum(TABLE1[p]["P+"][j] > TABLE1[p]["T"][j] for p in POLICIES for j in range(5))
    res["cells_P+_lt_T"] = sum(TABLE1[p]["P+"][j] < TABLE1[p]["T"][j] for p in POLICIES for j in range(5))
    res["cells_P+_gt_P-"] = sum(TABLE1[p]["P+"][j] > TABLE1[p]["P-"][j] for p in POLICIES for j in range(5))
    res["P+_minus_P-_by_cell"] = {p: [TABLE1[p]["P+"][j] - TABLE1[p]["P-"][j] for j in range(5)] for p in POLICIES}
    res["P+_minus_T_by_cell"] = {p: [TABLE1[p]["P+"][j] - TABLE1[p]["T"][j] for j in range(5)] for p in POLICIES}

    # 4. 풀링 검정 (과제별 60회, 전체 300회) -- 정책 간 독립 가정
    def pooled(cond, j):
        return sum(counts[p][cond][j] for p in POLICIES)

    res["pooled_task"] = {}
    ps_pm = []
    for j, t in enumerate(TASKS):
        kT, kM, kP = pooled("T", j), pooled("P-", j), pooled("P+", j)
        d_pt = newcombe_diff(kP, 60, kT, 60)
        d_pm = newcombe_diff(kP, 60, kM, 60)
        p_pt, p_pm = fisher(kP, 60, kT, 60), fisher(kP, 60, kM, 60)
        ps_pm.append(p_pm)
        strata_pm = [(counts[p]["P+"][j], 20, counts[p]["P-"][j], 20) for p in POLICIES]
        res["pooled_task"][t] = {
            "k_T": kT, "k_P-": kM, "k_P+": kP, "n": 60,
            "P+_vs_T": {"diff_pp": round(100 * d_pt[0], 1), "ci95_pp": [round(100 * d_pt[1], 1), round(100 * d_pt[2], 1)],
                        "fisher_p": round(p_pt, 4)},
            "P+_vs_P-": {"diff_pp": round(100 * d_pm[0], 1), "ci95_pp": [round(100 * d_pm[1], 1), round(100 * d_pm[2], 1)],
                         "fisher_p": round(p_pm, 4), "cmh_by_policy": {k: round(v, 4) for k, v in cmh(strata_pm).items()}},
            "P-_vs_T_fisher_p": round(fisher(kM, 60, kT, 60), 4),
        }
    for t, a in zip(TASKS, holm(ps_pm)):
        res["pooled_task"][t]["P+_vs_P-"]["holm_p_over_5_tasks"] = round(a, 4)

    tot = {c: sum(pooled(c, j) for j in range(5)) for c in CONDS}
    res["pooled_overall"] = {"k": tot, "n": 300}
    for a, b in [("P+", "T"), ("P+", "P-"), ("P-", "T")]:
        d = newcombe_diff(tot[a], 300, tot[b], 300)
        res["pooled_overall"][f"{a}_vs_{b}"] = {"diff_pp": round(100 * d[0], 1), "ci95_pp": [round(100 * d[1], 1), round(100 * d[2], 1)],
                                                "fisher_p": float(f"{fisher(tot[a], 300, tot[b], 300):.3g}")}
    strata_all = [(counts[p]["P+"][j], 20, counts[p]["P-"][j], 20) for p in POLICIES for j in range(5)]
    res["pooled_overall"]["P+_vs_P-_cmh_15_strata"] = {k: float(f"{v:.4g}") for k, v in cmh(strata_all).items()}
    strata_pt = [(counts[p]["P+"][j], 20, counts[p]["T"][j], 20) for p in POLICIES for j in range(5)]
    res["pooled_overall"]["P+_vs_T_cmh_15_strata"] = {k: float(f"{v:.4g}") for k, v in cmh(strata_pt).items()}
    # 15쌍 부호 검정 (P+ 대 T)
    gt, lt = res["cells_P+_gt_T"], res["cells_P+_lt_T"]
    res["sign_test_P+_vs_T_15_cells"] = {"gt": gt, "lt": lt, "p_two_sided": float(stats.binomtest(gt, gt + lt, 0.5).pvalue)}

    # 5. 비열등성 여지: P+ - T 전체 차이의 Newcombe 구간
    # 6. F2 이질성: "접촉 하중 우세 과제" 3개(Screw, Table, Air) 대 나머지 2개의 P+ - P- 리프트 차이
    contact = [0, 3, 4]
    other = [1, 2]
    kP_c = sum(pooled("P+", j) for j in contact); kM_c = sum(pooled("P-", j) for j in contact)
    kP_o = sum(pooled("P+", j) for j in other); kM_o = sum(pooled("P-", j) for j in other)
    lift_c = (kP_c - kM_c) / 180
    lift_o = (kP_o - kM_o) / 120
    # 리프트 차이의 Wald 검정 (독립 가정)
    def var_diff(kP, kM, n):
        pP, pM = kP / n, kM / n
        return pP * (1 - pP) / n + pM * (1 - pM) / n
    se = math.sqrt(var_diff(kP_c, kM_c, 180) + var_diff(kP_o, kM_o, 120))
    z = (lift_c - lift_o) / se
    res["F2_heterogeneity"] = {"contact_tasks": [TASKS[j] for j in contact], "other_tasks": [TASKS[j] for j in other],
                               "lift_contact_pp": round(100 * lift_c, 1), "lift_other_pp": round(100 * lift_o, 1),
                               "diff_pp": round(100 * (lift_c - lift_o), 1), "wald_z": round(z, 2),
                               "p_two_sided": round(2 * float(stats.norm.sf(abs(z))), 3)}

    # 7. 정책 비교 (P+ 조건)
    pol_tot = {p: sum(counts[p]["P+"]) for p in POLICIES}
    res["policy_P+_totals_of_100"] = pol_tot
    res["policy_P+_fisher_p"] = {"pi0.5_vs_DP": float(f"{fisher(pol_tot['pi0.5'], 100, pol_tot['DP'], 100):.3g}"),
                                 "pi0.5_vs_ACT": round(fisher(pol_tot["pi0.5"], 100, pol_tot["ACT"], 100), 3),
                                 "ACT_vs_DP": round(fisher(pol_tot["ACT"], 100, pol_tot["DP"], 100), 3)}
    res["policy_rank_same_in_all_conditions"] = all(
        cond_policy[c]["pi0.5"] > cond_policy[c]["ACT"] > cond_policy[c]["DP"] for c in CONDS)

    # 8. 검정력 (n = 20 칸, 60 풀링)
    res["power"] = {
        "cell_n20_base0.6_diff0.10": power_fisher(0.6, 0.7, 20),
        "cell_n20_base0.6_diff0.20": power_fisher(0.6, 0.8, 20),
        "cell_n20_base0.6_diff0.30": power_fisher(0.6, 0.9, 20),
        "pooled_n60_base0.6_diff0.083": power_fisher(0.6, 0.683, 60),
        "pooled_n60_base0.6_diff0.15": power_fisher(0.6, 0.75, 60),
    }
    # 80% 검정력에 필요한 팔당 시행 수 (P+ - T = 8.3 pp, 기저 0.65 -> 0.733)
    def needed(p1, p2, target=0.8):
        for n in range(20, 1200, 10):
            if power_fisher(p1, p2, n, sims=800, seed=1) >= target:
                return n
        return None
    res["n_per_arm_for_80pct_power"] = {"0.65_vs_0.733 (AirPods P+ vs T)": needed(0.65, 0.733),
                                        "0.60_vs_0.683 (Ball P+ vs T)": needed(0.60, 0.683)}

    # 9. 효율 (F3, 그림 7)
    nT, nS = 18, 52
    ratio = nS / nT
    # 같은 30분 창이므로 포아송 비율 검정 = 조건부 이항 검정 (p0 = 0.5)
    bt = stats.binomtest(nS, nS + nT, 0.5)
    ci = bt.proportion_ci(confidence_level=0.95, method="exact")
    lo_r, hi_r = ci.low / (1 - ci.low), ci.high / (1 - ci.high)
    succ_T, succ_P = cond_overall["T"] / 100, cond_overall["P+"] / 100
    res["efficiency"] = {
        "demos": {"T": nT, "SEED-UMI": nS}, "ratio": round(ratio, 3),
        "seconds_per_demo": {"T": 1800 / nT, "SEED-UMI": round(1800 / nS, 1)},
        "poisson_rate_ratio_exact_ci95": [round(lo_r, 2), round(hi_r, 2)],
        "binomial_p_equal_rate": float(f"{bt.pvalue:.3g}"),
        "paper_text": {"raw": "almost 3.0x", "normalized": "approximately 2.9x"},
        "normalized_ratio_with_overall_success (P+ 70.0 / T 71.7)": round(ratio * 0.700 / 0.717, 3),
        "normalized_ratio_with_AirPods_success (P+ 73.3 / T 65.0)": round(ratio * 0.7333 / 0.65, 3),
        "normalized_ratio_with_AirPods_success_P- (65.0 / 65.0)": round(ratio * 1.0, 3),
        "rounding_of_ratio_to_1dp": round(ratio, 1),
    }
    # 수집 시간 상쇄 시뮬: 보정 에피소드 250개 한 번 + 과제별 시연 100개
    amort = {}
    for cal_s in (30, 60, 120):      # 보정 에피소드 하나당 사람 + 로봇 재생 합산 초 (가정)
        cal_min = 250 * cal_s / 60
        saved_per_task_min = 100 * (1800 / nT - 1800 / nS) / 60
        amort[f"cal_{cal_s}s_per_episode"] = {
            "calibration_min": round(cal_min, 1), "saved_per_task_min": round(saved_per_task_min, 1),
            "tasks_to_break_even": round(cal_min / saved_per_task_min, 2)}
    res["efficiency"]["calibration_amortization_assumed"] = amort
    res["efficiency"]["demo_time_100_eps_min"] = {"T": round(100 * 1800 / nT / 60, 1), "SEED-UMI": round(100 * 1800 / nS / 60, 1)}

    # 10. 수치 읽기 점검
    res["check_text_vs_table"] = {
        "P+ AirPods 73.3 vs T 65.0": [round(cond_task["P+"][1], 1), round(cond_task["T"][1], 1)],
        "P+ Ball 68.3 vs T 60.0": [round(cond_task["P+"][2], 1), round(cond_task["T"][2], 1)],
        "lift Screw +16.7": round(lift_task[0], 1), "lift Table +15.0": round(lift_task[3], 1), "lift Air +15.0": round(lift_task[4], 1),
        "lift overall +12.7": round(cond_overall["P+"] - cond_overall["P-"], 1),
    }

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ["mean_overall", "overall_vs_reported", "lift_P+_minus_P-_by_task",
                                          "P+_minus_T_by_task", "P-_minus_T_by_task", "cells_P-_equals_T", "cells_P+_gt_T",
                                          "cells_P+_lt_T", "pooled_overall", "sign_test_P+_vs_T_15_cells",
                                          "F2_heterogeneity", "policy_P+_totals_of_100", "policy_P+_fisher_p",
                                          "power", "n_per_arm_for_80pct_power", "efficiency"]}, indent=1, ensure_ascii=False))
    for t in TASKS:
        print(t, json.dumps(res["pooled_task"][t], ensure_ascii=False))


if __name__ == "__main__":
    main()
