"""ResTacVLA 표 재계산 (CPU, numpy·scipy, 수 초). 실행: python3 analysis/table_check.py

- T1: Table I의 % → 성공 횟수(25회/15회) 정수 일치, 평균(8칸) 재계산, 과제 성공 기준 5과제 평균
- T2: 초록·기여의 "최대 86.7%", "+34.6%", 본문 "+44.0%", "+19.2%" 대조, 최강 기준선(T-UniT) 대비 차이
- T3: 과제별 Fisher 정확검정 + Holm, 과제 합산 Cochran–Mantel–Haenszel (ResTacVLA vs π0.5, vs T-UniT)
- T4: Alignment(A) → Interaction(I) 분해 P(I) = P(A)·P(I|A)와 이득의 로그 분해
- T5: Table II 평균(Plug 25회 + Wiping 15회)에서 가능한 (Plug, Wiping) 성공 횟수 조합, 같은 두 과제의 T-UniT 평균
- T6: Table III 강건성 — 해당 과제의 Table I 값과 비교한 하락폭(교란이 없는 조건 대비)
- T7: "모든 기준선을 모든 과제에서 앞선다" 점검(칸별 순위)
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

# Table I (Sec. IV-B). 시행 수: Lightbulb·Plug 25회, Peg-in-Hole·Peg Transfer·Plate Wiping 15회 (Sec. IV-A)
COLS = ["Lightbulb-A", "Lightbulb-I", "Plug-A", "Plug-I", "Peg-A", "Peg-I", "Transfer", "Wiping"]
NCOL = [25, 25, 25, 25, 15, 15, 15, 15]
T1 = {
    "pi05": [28.0, 8.0, 36.0, 20.0, 46.7, 40.0, 26.7, 20.0],
    "DP_woT": [20.0, 0.0, 28.0, 16.0, 40.0, 26.7, 13.3, 6.7],
    "pi05_T-ResNet": [16.0, 0.0, 32.0, 24.0, 40.0, 40.0, 20.0, 13.3],
    "pi05_T-UniT": [28.0, 12.0, 40.0, 32.0, 73.3, 53.3, 66.7, 33.3],
    "DP_T-ResTac": [32.0, 12.0, 32.0, 28.0, 66.7, 60.0, 40.0, 33.3],
    "ResTacVLA": [56.0, 32.0, 68.0, 60.0, 86.7, 80.0, 60.0, 60.0],
}
T1_AVG = {"pi05": 28.2, "DP_woT": 18.8, "pi05_T-ResNet": 23.2, "pi05_T-UniT": 42.3, "DP_T-ResTac": 38.0, "ResTacVLA": 62.8}
# 과제 성공(최종 단계) 칸: 두 단계 과제는 I, 나머지는 단일 칸
TASK_COLS = {"Lightbulb": 1, "Plug": 3, "Peg-in-Hole": 5, "Transfer": 6, "Wiping": 7}
PHASE_TASKS = {"Lightbulb": (0, 1), "Plug": (2, 3), "Peg-in-Hole": (4, 5)}

# Table II (Sec. IV-D): Plug Insertion + Plate Wiping 평균
T2 = {"Full": 60.0, "w/o VQ": 33.3, "w/o Gating": 46.7, "pi05": 20.0}
# Table III (Sec. IV-E): Dynamic = Peg-in-Hole(15), Height = Plate Wiping(15), Grasp = Plug Insertion(25)
T3 = {
    "pi05": {"Dynamic": 26.7, "Height(+)": 33.3, "Height(-)": 0.0, "Grasp": 8.0},
    "pi05_T-UniT": {"Dynamic": 40.0, "Height(+)": 46.7, "Height(-)": 13.3, "Grasp": 20.0},
    "ResTacVLA": {"Dynamic": 66.7, "Height(+)": 53.3, "Height(-)": 40.0, "Grasp": 52.0},
}
T3_N = {"Dynamic": 15, "Height(+)": 15, "Height(-)": 15, "Grasp": 25}
# 교란 조건과 비교할 Table I의 교란 없는 칸 (과제 성공 기준)
T3_BASE_COL = {"Dynamic": 5, "Height(+)": 7, "Height(-)": 7, "Grasp": 3}


def count(pct, n):
    k = round(pct / 100 * n)
    assert abs(100 * k / n - pct) < 0.051, (pct, n, k)  # 소수 첫째 자리 반올림과 일치해야 한다
    return k


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return adj


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 3), round(c + h, 3)]


def mh_test(tables):
    """Cochran–Mantel–Haenszel (연속성 보정 없음). tables: [(k1, n1, k2, n2)]"""
    num, var, or_num, or_den = 0.0, 0.0, 0.0, 0.0
    for k1, n1, k2, n2 in tables:
        a, b, c, d = k1, n1 - k1, k2, n2 - k2
        n = a + b + c + d
        m1, m0, r1 = a + c, b + d, a + b
        num += a - r1 * m1 / n
        var += r1 * (n - r1) * m1 * m0 / (n * n * (n - 1))
        or_num += a * d / n
        or_den += b * c / n
    chi2 = num * num / var
    return {"chi2": round(chi2, 3), "p": float(f"{stats.chi2.sf(chi2, 1):.3g}"),
            "MH_odds_ratio": round(or_num / or_den, 2)}


def g3(x):
    return float(f"{x:.3g}")


def main() -> None:
    out: dict = {}
    K = {m: [count(p, n) for p, n in zip(v, NCOL)] for m, v in T1.items()}

    # T1 ---------------------------------------------------------------
    t1 = {}
    for m, v in T1.items():
        avg8 = sum(v) / 8
        task = [v[i] for i in TASK_COLS.values()]
        k_task = [K[m][i] for i in TASK_COLS.values()]
        n_task = [NCOL[i] for i in TASK_COLS.values()]
        t1[m] = {
            "counts": {c: f"{k}/{n}" for c, k, n in zip(COLS, K[m], NCOL)},
            "avg_8col_recomputed": round(avg8, 2), "avg_reported": T1_AVG[m],
            "avg_matches": abs(round(avg8, 1) - T1_AVG[m]) < 0.051,
            "task_success_5task_mean": round(sum(task) / 5, 2),
            "task_success_pooled": f"{sum(k_task)}/{sum(n_task)}",
            "task_success_pooled_pct": round(100 * sum(k_task) / sum(n_task), 1),
            "task_success_pooled_wilson95": wilson(sum(k_task), sum(n_task)),
        }
    out["T1_table1"] = t1

    # T2 ---------------------------------------------------------------
    ours, pi, unit, dp, dpres = (T1_AVG[m] for m in ["ResTacVLA", "pi05", "pi05_T-UniT", "DP_woT", "DP_T-ResTac"])
    best_col = int(np.argmax(T1["ResTacVLA"]))
    out["T2_headline_claims"] = {
        "up_to_86.7_is_column": COLS[best_col],
        "best_task_success_column(final_phase)": max(((T1["ResTacVLA"][i], c) for c, i in TASK_COLS.items())),
        "+34.6_is_pp_vs_pi05": round(ours - pi, 1),
        "relative_gain_vs_pi05_pct": round(100 * (ours / pi - 1), 1),
        "+44.0_is_pp_vs_DP_woT": round(ours - dp, 1),
        "+19.2_is_pp_DP_T-ResTac_vs_DP_woT": round(dpres - dp, 1),
        "pp_vs_strongest_baseline_T-UniT(8col)": round(ours - unit, 1),
        "pp_vs_T-UniT(5task_mean)": round(t1["ResTacVLA"]["task_success_5task_mean"] - t1["pi05_T-UniT"]["task_success_5task_mean"], 1),
        "pp_vs_pi05(5task_mean)": round(t1["ResTacVLA"]["task_success_5task_mean"] - t1["pi05"]["task_success_5task_mean"], 1),
        "note": "8칸 평균은 A(정렬)와 I(완료)를 같은 무게로 섞는다. 두 단계 과제는 A ≥ I이므로 8칸 평균이 과제 성공률보다 높다.",
    }

    # T3 ---------------------------------------------------------------
    t3 = {}
    for base in ["pi05", "pi05_T-UniT", "pi05_T-ResNet"]:
        rows, ps, tabs = {}, [], []
        for task, i in TASK_COLS.items():
            n = NCOL[i]
            a, b = K["ResTacVLA"][i], K[base][i]
            p = fisher(a, n, b, n)
            ps.append(p)
            tabs.append((a, n, b, n))
            rows[task] = {"ours": f"{a}/{n}", "base": f"{b}/{n}", "ours_wilson95": wilson(a, n),
                          "base_wilson95": wilson(b, n), "fisher_p": g3(p)}
        for task, adj in zip(TASK_COLS, holm(ps)):
            rows[task]["holm_p"] = g3(adj)
        rows["pooled_CMH(5 tasks, final phase)"] = mh_test(tabs)
        # 정렬 단계만 따로
        tabs_a = [(K["ResTacVLA"][ia], NCOL[ia], K[base][ia], NCOL[ia]) for ia, _ in PHASE_TASKS.values()]
        rows["pooled_CMH(3 tasks, Alignment)"] = mh_test(tabs_a)
        t3[f"ResTacVLA_vs_{base}"] = rows
    # DP 쌍: 잔차 촉각을 넣은 DP vs 촉각 없는 DP
    tabs = [(K["DP_T-ResTac"][i], NCOL[i], K["DP_woT"][i], NCOL[i]) for i in TASK_COLS.values()]
    t3["DP_T-ResTac_vs_DP_woT"] = {"per_task_fisher_p": {t: g3(fisher(*tb)) for t, tb in zip(TASK_COLS, tabs)},
                                   "pooled_CMH": mh_test(tabs)}
    # 본문 주장: T-ResNet이 Lightbulb에서 성능을 떨어뜨린다 (28.0 → 16.0, Lightbulb-A)
    t3["T-ResNet_degrades_Lightbulb-A"] = {"pi05": "7/25", "T-ResNet": "4/25", "fisher_p": g3(fisher(7, 25, 4, 25))}
    out["T3_significance"] = t3

    # T4 ---------------------------------------------------------------
    t4 = {}
    for task, (ia, ii) in PHASE_TASKS.items():
        t4[task] = {m: {"A": f"{K[m][ia]}/{NCOL[ia]}", "I": f"{K[m][ii]}/{NCOL[ii]}",
                        "P(I|A)": (round(K[m][ii] / K[m][ia], 3) if K[m][ia] else None)} for m in T1}
    dec = {}
    for base in ["pi05", "pi05_T-UniT"]:
        A_o = sum(K["ResTacVLA"][ia] for ia, _ in PHASE_TASKS.values())
        I_o = sum(K["ResTacVLA"][ii] for _, ii in PHASE_TASKS.values())
        A_b = sum(K[base][ia] for ia, _ in PHASE_TASKS.values())
        I_b = sum(K[base][ii] for _, ii in PHASE_TASKS.values())
        r_I, r_A = I_o / I_b, A_o / A_b
        r_c = (I_o / A_o) / (I_b / A_b)
        dec[f"ResTacVLA_vs_{base}"] = {
            "A_sum": f"{A_o} vs {A_b}", "I_sum": f"{I_o} vs {I_b}",
            "ratio_I": round(r_I, 3), "ratio_A": round(r_A, 3), "ratio_P(I|A)": round(r_c, 3),
            "share_of_log_gain_from_A": round(math.log(r_A) / math.log(r_I), 3),
            "P(I|A)_fisher_p(pooled)": g3(fisher(I_o, A_o, I_b, A_b)),
        }
    t4["log_decomposition_3tasks"] = dec
    out["T4_phase_decomposition"] = t4

    # T5 ---------------------------------------------------------------
    t5 = {}
    for cfg, avg in T2.items():
        combos = [(p, w) for p in range(26) for w in range(16)
                  if abs(round((100 * p / 25 + 100 * w / 15) / 2, 1) - avg) < 0.051]
        t5[cfg] = {"avg": avg, "possible_(plug/25, wiping/15)": combos}
    unit2 = (T1["pi05_T-UniT"][3] + T1["pi05_T-UniT"][7]) / 2
    t5["consistency"] = {
        "Full_vs_TableI(Plug-I, Wiping)": (T1["ResTacVLA"][3] + T1["ResTacVLA"][7]) / 2,
        "pi05_vs_TableI": (T1["pi05"][3] + T1["pi05"][7]) / 2,
        "T-UniT_same_two_tasks(TableI)": round(unit2, 2),
        "w/o_VQ_minus_T-UniT_pp": round(T2["w/o VQ"] - unit2, 2),
        "note": "w/o VQ(연속 잔차, 게이트 유지로 읽힘)가 같은 두 과제의 T-UniT(원시 UniT 특징)와 같은 수준이다.",
    }
    out["T5_table2_ablation"] = t5

    # T6 ---------------------------------------------------------------
    t6 = {}
    for m, row in T3.items():
        t6[m] = {}
        for cond, pct in row.items():
            n = T3_N[cond]
            k = count(pct, n)
            base_pct = T1[m][T3_BASE_COL[cond]]
            kb = K[m][T3_BASE_COL[cond]]
            t6[m][cond] = {"perturbed": f"{k}/{n}", "nominal(TableI)": f"{kb}/{n}",
                           "drop_pp": round(pct - base_pct, 1), "fisher_p_vs_nominal": g3(fisher(k, n, kb, n))}
    t6["note"] = ("Dynamic은 세 방법 모두 −13.3pp(−2/15), Height(−)는 모두 −20.0pp(−3/15)로 같다. "
                  "강건성 표의 차이는 교란 없는 성능 차이가 그대로 옮겨온 것이다.")
    out["T6_robustness_drop"] = t6

    # T7 ---------------------------------------------------------------
    t7 = {}
    for j, c in enumerate(COLS):
        col = sorted(((T1[m][j], m) for m in T1), reverse=True)
        t7[c] = {"best": col[0][1], "best_value": col[0][0], "ours_rank": 1 + [m for _, m in col].index("ResTacVLA")}
    out["T7_column_ranks"] = t7

    path = ROOT / "analysis" / "table_check_results.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    # 요약 출력
    print("T1 avg matches:", {m: t1[m]["avg_matches"] for m in t1})
    print("T1 5-task:", {m: t1[m]["task_success_5task_mean"] for m in t1})
    print("T2:", out["T2_headline_claims"])
    for k, v in t3.items():
        print("T3", k, json.dumps(v, ensure_ascii=False)[:600])
    print("T4:", dec)
    print("T5:", {k: v.get("possible_(plug/25, wiping/15)") for k, v in t5.items() if k != "consistency"}, t5["consistency"])
    print("T6:", {m: {c: d["drop_pp"] for c, d in v.items()} for m, v in t6.items() if m != "note"})
    print("T7:", {c: (v["best"], v["ours_rank"]) for c, v in t7.items()})
    print("wrote", path)


if __name__ == "__main__":
    main()
