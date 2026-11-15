"""논문 표의 성공 수에서 평균, Wilson 95% 구간, Fisher 검정(양측), Holm 보정, 일관성 검사를 다시 계산한다.

실행: python3 analysis/table_check.py  → analysis/table_check_results.json (CPU, 수 초)
입력은 논문 TeX 표 값(paper/src_v1/PACE_camera_ready_v5.tex)을 그대로 옮겼다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import brentq

OUT = Path(__file__).with_name("table_check_results.json")
TASKS = ["Key", "2-pin", "3-pin", "USB", "Ethernet"]
N = 30

# 표 1(a): 실로봇 5과제, 과제당 30회
T1 = {
    "Vision-only ACT": [4, 4, 2, 1, 4],
    "Vision-only DP": [5, 6, 4, 3, 5],
    "pi0": [6, 8, 4, 4, 6],
    "ACT + Tactile Cross-Attn": [10, 15, 11, 9, 14],
    "ACT + Tactile Concat": [13, 17, 12, 13, 14],
    "PACE": [27, 29, 28, 27, 25],
}
T1_AVG_REPORTED = {"Vision-only ACT": 10.0, "Vision-only DP": 15.3, "pi0": 18.7,
                   "ACT + Tactile Cross-Attn": 39.3, "ACT + Tactile Concat": 46.0, "PACE": 90.7}
# 표 2: 절제 실험
T2 = {
    "PACE w/o Pred.": [25, 27, 26, 26, 24],
    "PACE w/o Phase": [24, 26, 21, 23, 24],
    "PACE w/o Event": [20, 22, 18, 19, 21],
    "Phase-only": [17, 20, 16, 18, 19],
    "PACE": [27, 29, 28, 27, 25],
}
T2_AVG_REPORTED = {"PACE w/o Pred.": 85.3, "PACE w/o Phase": 78.7, "PACE w/o Event": 66.7,
                   "Phase-only": 60.0, "PACE": 90.7}
# 표 3: 신호 대체 (2-pin, Key)
T3 = {"Tactile (default)": [29, 27], "F/T": [27, 27], "Joint current": [28, 25]}
# 표 4: DP + PACE (2-pin, Key)
T4 = {"Vision-only DP": [6, 5], "DP + PACE": [28, 25]}
# 힘 피드백 수집 (2-pin, 30회)
FF = {"PACE feedback on": 29, "PACE feedback off": 21,
      "ACT + Concat feedback on": 17, "ACT + Concat feedback off": 7}
# 표 1(b): ManipulationNet (cell별 25회)
MN = {"Cyl. 3mm": 25, "Cyl. 1mm": 22, "Cyl. 0.1mm": 21, "Hex. 3mm": 24, "Hex. 1mm": 22, "Hex. 0.1mm": 19}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def fisher2(k1, n1, k2, n2, alt="two-sided"):
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(stats.fisher_exact(table, alternative=alt)[1])


def main():
    res = {}

    # 1. 평균과 보고값 대조
    avg_check = {}
    for name, ks in T1.items():
        avg = 100 * sum(ks) / (N * len(ks))
        avg_check[name] = {"recomputed": round(avg, 2), "reported": T1_AVG_REPORTED[name],
                           "match": abs(round(avg, 1) - T1_AVG_REPORTED[name]) < 0.051}
    res["table1_average_check"] = avg_check

    ab_check = {}
    for name, ks in T2.items():
        avg = 100 * sum(ks) / (N * len(ks))
        ab_check[name] = {"recomputed": round(avg, 2), "reported": T2_AVG_REPORTED[name],
                          "match": abs(round(avg, 1) - T2_AVG_REPORTED[name]) < 0.051}
    res["table2_average_check"] = ab_check

    # 2. 실로봇 과제별 Wilson 95% 구간 (PACE와 최강 기준선)
    wil = {}
    for name, ks in T1.items():
        wil[name] = {t: {"k": k, "wilson95": [round(x, 3) for x in wilson(k, N)]}
                     for t, k in zip(TASKS, ks)}
        pooled_k = sum(ks)
        wil[name]["pooled"] = {"k": pooled_k, "n": N * len(ks),
                               "wilson95": [round(x, 3) for x in wilson(pooled_k, N * len(ks))]}
    res["table1_wilson"] = wil

    # 3. Fisher 검정: PACE 대 각 기준선 (과제별 + 5과제 합산), Holm 보정 (5 기준선 x 6검정 = 30)
    base_names = [k for k in T1 if k != "PACE"]
    rows, pvals = [], []
    for b in base_names:
        for t, kp, kb in zip(TASKS, T1["PACE"], T1[b]):
            rows.append({"baseline": b, "task": t, "pace": kp, "base": kb})
        rows.append({"baseline": b, "task": "pooled", "pace": sum(T1["PACE"]), "base": sum(T1[b]),
                     "n_each": N * len(TASKS)})
    for r in rows:
        n_each = r.get("n_each", N)
        r["p_two_sided"] = fisher2(r["pace"], n_each, r["base"], n_each)
        pvals.append(r["p_two_sided"])
    adj = holm(pvals)
    for r, a in zip(rows, adj):
        r["p_holm"] = a
    res["table1_fisher_pace_vs_baselines"] = rows

    # 4. 표 2 절제: PACE 대 각 절제, 합산 기준 (n=150) + Holm (4 x 6 = 24)
    rows2, p2 = [], []
    for b in [k for k in T2 if k != "PACE"]:
        for t, kp, kb in zip(TASKS, T2["PACE"], T2[b]):
            rows2.append({"variant": b, "task": t, "pace": kp, "variant_k": kb})
        rows2.append({"variant": b, "task": "pooled", "pace": sum(T2["PACE"]), "variant_k": sum(T2[b]),
                      "n_each": N * 5})
    for r in rows2:
        r["p_two_sided"] = fisher2(r["pace"], r.get("n_each", N), r["variant_k"], r.get("n_each", N))
        p2.append(r["p_two_sided"])
    for r, a in zip(rows2, holm(p2)):
        r["p_holm"] = a
    res["table2_fisher_ablation"] = rows2

    # 5. 표 3, 4, 힘 피드백: 쌍 비교 (두 과제 합 60회)
    t3 = {"PACE tactile vs F/T (2-pin+Key)": fisher2(56, 60, 54, 60),
          "PACE tactile vs joint current (2-pin+Key)": fisher2(56, 60, 53, 60)}
    t4 = {"DP+PACE vs vanilla DP (2-pin+Key)": fisher2(53, 60, 11, 60)}
    ff = {"PACE on vs off (2-pin)": fisher2(29, 30, 21, 30),
          "ACT+Concat on vs off (2-pin)": fisher2(17, 30, 7, 30),
          "PACE off vs ACT+Concat off (2-pin)": fisher2(21, 30, 7, 30)}
    res["table3_signal_fisher"] = t3
    res["table4_dp_fisher"] = t4
    res["feedback_fisher"] = ff
    res["table3_avgs"] = {k: round(100 * sum(v) / 60, 1) for k, v in T3.items()}

    # 6. 데이터 개수와 보고 불일치 검사
    main_avg_pace = 136 / 150
    res["manipulationnet"] = {
        "cells": MN,
        "total_k": sum(MN.values()), "total_n": 150,
        "total_rate": round(100 * sum(MN.values()) / 150, 2),
        "wilson95": [round(x, 3) for x in wilson(sum(MN.values()), 150)],
        "per_peg_avg": {"cyl": round(100 * (25 + 22 + 21) / 75, 1), "hex": round(100 * (24 + 22 + 19) / 75, 1)},
        "per_clearance": {"3mm": round(100 * (25 + 24) / 50, 1), "1mm": round(100 * (22 + 22) / 50, 1),
                          "0.1mm": round(100 * (21 + 19) / 50, 1)},
    }

    # 7. 기계적 분석 보고 값의 검사: r=0.356, p<0.001 이 되려면 n이 얼마나 필요한가
    r = 0.356

    def p_of_r(n):
        t = r * math.sqrt(n - 2) / math.sqrt(1 - r * r)
        return 2 * stats.t.sf(abs(t), n - 2)

    n_needed = next(n for n in range(4, 500) if p_of_r(n) < 0.001)
    res["mechanistic_pearson"] = {"r": r, "p_at_n30": round(p_of_r(30), 4),
                                  "min_n_for_p_below_0.001": n_needed}

    # 8. ANOVA: F=8.36, eta^2=0.236 에서 총 관측 수 N과 집단 수 k의 관계
    F, eta2 = 8.36, 0.236
    anova = {}
    for k in [2, 3, 4, 5]:
        # eta^2 = SSb/(SSb+SSw), F = (eta2/(k-1)) / ((1-eta2)/(N-k))  -> N = k + (1-eta2)(k-1)F/eta2
        N_implied = k + (1 - eta2) * (k - 1) * F / eta2
        p_at_30 = stats.f.sf(F, k - 1, 30 - k) if 30 > k else None
        anova[f"k={k}"] = {"implied_total_N": round(N_implied, 1),
                           "p_if_N_is_30": None if p_at_30 is None else float(p_at_30)}
    res["anova_implied"] = anova

    # 9. 본문 주장 대조
    res["claims"] = {
        "pace_vs_concat_pp": round(100 * (136 - 69) / 150, 2),
        "pace_vs_strongest_baseline_pp_reported": ">44",
        "dp_plus_pace_gain_pp": round(88.33 - 18.33, 2),
        "tactile_default_vs_signal_avg": round(100 * 56 / 60, 1),
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # 요약 출력
    print("평균 대조:", {k: v["match"] for k, v in avg_check.items()})
    print("표2 평균 대조:", {k: v["match"] for k, v in ab_check.items()})
    print("PACE 대 기준선 합산 p(Holm):", {r["baseline"]: round(r["p_holm"], 6) for r in rows if r["task"] == "pooled"})
    print("절제 합산 p(Holm):", {r["variant"]: round(r["p_holm"], 6) for r in rows2 if r["task"] == "pooled"})
    print("MN:", res["manipulationnet"]["total_rate"], res["manipulationnet"]["wilson95"])
    print("기계 r 검사:", res["mechanistic_pearson"])
    print("ANOVA:", anova)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
