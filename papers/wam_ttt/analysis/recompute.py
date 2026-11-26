"""WAM-TTT (arXiv:2607.06988 v3) 표 수치 재계산과 일관성 검사.

입력은 논문 v3의 표 1, C.1, 2, 3, 4, E.1, E.2, E.3과 부록 D 루브릭이다(아래 상수, 논문 쪽 번호는 03 문서에 있다).
논문은 시행별 점수를 주지 않으므로 검정은 "분산 상계" 가정을 쓴다: [0, 1] 값의 분산은 m(1-m) 이하이므로
n = 25 시행 평균의 표준오차를 sqrt(m(1-m)/n)로 놓으면 보수적(더 넓은) 구간이 된다.
시행 수는 본문이 밝힌 값(표 1, C.1, E.1, E.3: 25, 표 3, E.2: 10)을 쓰고, 표 2와 4는 밝히지 않아 25를 가정한다.

실행: python3 analysis/recompute.py  (numpy, scipy, 수 초) → analysis/recompute_results.json
"""
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"

TASKS = ["TransferBottle", "TableBussing", "DeliverDrink", "SwapPlace", "PourWater", "StampPaper",
         "FlipSteak", "Pyramid", "MultiStepSteak"]
METHODS = ["pi05", "LDA", "EgoScale", "WAM-Cotrain", "WAM-ICL", "WAM-TTT"]
# 표 C.1 (표 1의 New 블록과 같음)
C1 = {
    "pi05": {"Orig": [55.5, 80.0, 70.0, 12.2, 25.0, 31.1, 4.4, 6.1, 2.0, 31.8],
             "New": [33.4, 36.0, 15.0, 7.4, 10.0, 24.4, 2.0, 4.7, 0.3, 14.8]},
    "LDA": {"Orig": [72.0, 90.0, 80.0, 66.7, 33.6, 50.0, 33.3, 6.7, 19.2, 50.2],
            "New": [56.0, 70.0, 55.0, 44.4, 20.0, 33.3, 10.0, 0.6, 3.0, 32.5]},
    "EgoScale": {"Orig": [69.4, 80.0, 69.6, 10.0, 30.0, 2.7, 5.0, 2.0, 2.0, 30.1],
                 "New": [34.4, 44.0, 33.3, 6.0, 7.5, 1.1, 5.0, 2.0, 2.0, 15.0]},
    "WAM-Cotrain": {"Orig": [11.6, 40.0, 59.4, 74.1, 26.3, 10.0, 21.0, 9.4, 16.8, 29.8],
                    "New": [10.0, 10.0, 44.3, 48.1, 24.0, 21.7, 34.2, 12.0, 23.8, 25.3]},
    "WAM-ICL": {"Orig": [60.0, 89.0, 70.3, 68.7, 55.3, 36.0, 33.3, 4.7, 18.2, 48.4],
                "New": [10.0, 10.0, 14.2, 10.0, 0.0, 5.0, 10.0, 2.0, 2.5, 7.1]},
    "WAM-TTT": {"Orig": [77.8, 100.0, 90.0, 88.9, 63.3, 35.0, 44.0, 10.0, 40.6, 61.1],
                "New": [55.6, 100.0, 66.7, 66.7, 30.0, 8.3, 34.3, 10.4, 43.8, 46.2]},
}
# 부록 D 루브릭 (가중치, 단위 1.0)
RUBRIC = {
    "TransferBottle": [.05, .10, .05, .15, .10, .05, .50],
    "TableBussing": [.5, .5],  # N = 1 기본, 항목당 0.5/N
    "DeliverDrink": [.30, .30, .40],
    "SwapPlace": [.20, .30, .30, .20],
    "PourWater": [.10, .15, .05, .60, .10],
    "StampPaper": [.05, .15, .05, .20, .05, .50],
    "FlipSteak": [.02, .20, .03, .20, .45, .10],
    "Pyramid": [.01, .05, .02, .10, .01, .10, .02, .15, .01, .15, .02, .36],
    "MultiStepSteak": [.01, .02, .01, .07, .01, .01, .01, .07, .01, .17, .20, .15, .07, .01, .01, .01, .07, .09],
}
# 격자: 시행 점수가 가능한 가장 작은 단위(단위 0.01 기준 정수의 최대공약수). 항목 가중치의 gcd.
def gcd_units(ws):
    g = 0
    for w in ws:
        g = math.gcd(g, int(round(w * 100)))
    return g

GRID = {t: gcd_units(w) for t, w in RUBRIC.items()}  # 단위 0.01
T2 = {"WAM (no adapt.)": 44.4, "LingBot-style ICL": 50.0, "XSkill-style (+SAT)": 10.4, "UniSkill-style": 15.2, "WAM-TTT": 66.7}
T3 = {"TableBussing": {"WAM-TTT": 100.0, "WAM-LoRA": 30.0, "w/o Meta": 9.0, "w/o MemRecon": 66.7, "w/o TTT": 40.0},
      "SwapPlace": {"WAM-TTT": 88.9, "WAM-LoRA": 0.0, "w/o Meta": 0.0, "w/o MemRecon": 72.0, "w/o TTT": 74.1}}
T4 = {"Lighting": {"pi05": 28.0, "LDA": 54.0, "WAM-ICL": 12.0, "WAM-TTT": 66.0},
      "Spatial": {"pi05": 0.0, "LDA": 28.0, "WAM-ICL": 20.0, "WAM-TTT": 56.0}}
E1 = {"(100,0)": [44.1, 90.0, 44.4, 59.5], "(10,190)": [42.1, 68.0, 44.2, 51.4], "(100,100)": [55.6, 100.0, 66.7, 74.1],
      "(200,0)": [47.9, 100.0, 73.2, 73.7], "(100,200)": [58.9, 100.0, 61.0, 73.3]}
E1_TASKS = ["TransferBottle", "TableBussing", "DeliverDrink"]
E2 = {"DiT only": 72.0, "DiT+VLM (no pretrain)": 80.0, "DiT+VLM (frozen)": 54.0, "DiT+VLM (open, ours)": 100.0}
E3 = {"VG only": [55.6, 100.0, 66.7, 66.7, 72.3], "VG+FD": [14.2, 33.3, 26.8, 41.2, 28.9]}
E3_TASKS = ["TransferBottle", "TableBussing", "DeliverDrink", "SwapPlace"]

def main():
    res = {}


    def r1(x):
        return round(x + 1e-9, 1)


    # ---- V1: 평균 열 ----
    v1 = {}
    for m in METHODS:
        for s in ("Orig", "New"):
            vals = C1[m][s][:9]
            rep = C1[m][s][9]
            mean = float(np.mean(vals))
            v1[f"{m}/{s}"] = {"recomputed": round(mean, 3), "reported": rep, "diff": round(mean - rep, 3),
                              "match_1dp": abs(mean - rep) <= 0.05 + 1e-9}
    res["V1_avg_column"] = v1

    # ---- V2: 본문 증가량과 유지율 ----
    avg = {m: {s: C1[m][s][9] for s in ("Orig", "New")} for m in METHODS}
    res["V2_text_deltas"] = {
        "New_TTT_minus": {m: round(avg["WAM-TTT"]["New"] - avg[m]["New"], 1) for m in METHODS if m != "WAM-TTT"},
        "Orig_TTT_minus": {m: round(avg["WAM-TTT"]["Orig"] - avg[m]["Orig"], 1) for m in METHODS if m != "WAM-TTT"},
        "retention_New_over_Orig": {m: round(avg[m]["New"] / avg[m]["Orig"], 3) for m in METHODS},
        "delta_Orig_minus_New": {m: round(avg[m]["Orig"] - avg[m]["New"], 1) for m in METHODS},
        "New_rank_ascending": sorted(METHODS, key=lambda m: avg[m]["New"]),
    }

    # ---- V3: 과제별 승패 ----
    v3 = {}
    for s in ("New", "Orig"):
        ttt = C1["WAM-TTT"][s]
        d = {}
        for m in METHODS[:-1]:
            o = C1[m][s]
            d[m] = {"wins": sum(ttt[i] > o[i] for i in range(9)), "ties": sum(ttt[i] == o[i] for i in range(9)),
                    "losses": [(TASKS[i], ttt[i], o[i]) for i in range(9) if ttt[i] < o[i]]}
        best_other = [max(C1[m][s][i] for m in METHODS[:-1]) for i in range(9)]
        d["best_of_all_others"] = {"tasks_ttt_is_max": [TASKS[i] for i in range(9) if ttt[i] > best_other[i]],
                                   "tasks_not_max": [(TASKS[i], ttt[i], best_other[i]) for i in range(9) if ttt[i] <= best_other[i]],
                                   "margin_flip_steak": round(ttt[6] - best_other[6], 1)}
        v3[s] = d
    res["V3_win_loss"] = v3

    # ---- V4: 루브릭 합과 격자 ----
    res["V4_rubric_sums"] = {t: round(sum(w), 6) for t, w in RUBRIC.items()}
    res["V4_grid_units_0.01"] = GRID


    def feasible(value, n, g_units, tol=0.05):
        """시행 n개의 점수 합이 g_units(0.01 단위)의 정수배일 때 평균(%)이 value(소수 1자리 반올림)가 될 수 있는가."""
        # 평균(%) = 100 * (m * g/100) / n = m * g / n
        g = g_units
        m = round(value * n / g)
        for mm in (m - 1, m, m + 1):
            if mm >= 0 and abs(mm * g / n - value) <= tol + 1e-9:
                return True
        return False


    def feasible_bussing(value, n):
        """Table Bussing은 항목 수 N(기본 1)에 따라 단위가 0.5/N. N = 1..6 중 하나로 가능하면 (True, [N...])."""
        ok = [N for N in range(1, 7) if feasible(value, n, 50 / N)]
        return (len(ok) > 0, ok)


    def feas(task, value, n):
        if task == "TableBussing":
            ok, Ns = feasible_bussing(value, n)
            return ok, ({"N_items_ok": Ns} if ok else {})
        return feasible(value, n, GRID[task]), {}


    v4 = {}
    inf_cells = []
    tot = 0
    for m in METHODS:
        for s in ("Orig", "New"):
            for i, t in enumerate(TASKS):
                tot += 1
                ok, extra = feas(t, C1[m][s][i], 25)
                if not ok:
                    alt = [n for n in range(5, 31) if feas(t, C1[m][s][i], n)[0]]
                    inf_cells.append({"method": m, "setting": s, "task": t, "value": C1[m][s][i], "n_feasible_5_30": alt})
    v4["total_cells"] = tot
    v4["infeasible_at_n25"] = len(inf_cells)
    v4["infeasible_cells"] = inf_cells
    # 이 가운데 9분의 k 꼴 (n = 9, 18, 27 계열) 값이 몇 개인가
    def is_ninth(v):
        return any(abs(v - 100 * k / 9) <= 0.05 + 1e-9 for k in range(0, 10))
    ninth = [(m, s, t, C1[m][s][i]) for m in METHODS for s in ("Orig", "New") for i, t in enumerate(TASKS)
             if is_ninth(C1[m][s][i]) and C1[m][s][i] not in (0.0, 100.0)]
    v4["ninth_values_excluding_0_100"] = len(ninth)
    v4["ninth_values_infeasible_at_25"] = sum(1 for c in inf_cells if is_ninth(c["value"]) and c["value"] not in (0.0, 100.0))
    # 기대: 25 시행에서 우연히 9분의 k 격자에 걸릴 확률은 작다 (참고용: 0.1 간격 값 중 9분의 k 후보 비율)
    res["V4_feasibility_C1"] = v4

    # 각 표의 격자 검사
    def table_feas(name, entries, n):
        out = []
        for task, val in entries:
            ok, extra = feas(task, val, n)
            out.append({"task": task, "value": val, "n": n, "feasible": ok, **extra})
        return {"name": name, "cells": out, "n_infeasible": sum(1 for c in out if not c["feasible"])}


    res["V5_table2_n25_assumed"] = table_feas("Table 2 (Swap Place, n=25 가정)", [("SwapPlace", v) for v in T2.values()], 25)
    res["V6_table3_n10"] = table_feas("Table 3 (n=10)", [(t, v) for t, d in T3.items() for v in d.values()], 10)
    res["V7_table4_n25_assumed"] = table_feas("Table 4 (Deliver Drink, n=25 가정)", [("DeliverDrink", v) for d in T4.values() for v in d.values()], 25)
    res["V8_E1_n25"] = table_feas("Table E.1 (n=25)", [(E1_TASKS[j], E1[k][j]) for k in E1 for j in range(3)], 25)
    res["V9_E2_n10"] = table_feas("Table E.2 (Table Bussing, n=10)", [("TableBussing", v) for v in E2.values()], 10)
    res["V10_E3_n25"] = table_feas("Table E.3 (n=25)", [(E3_TASKS[j], E3[k][j]) for k in E3 for j in range(4)], 25)

    # Table 3의 Swap Place 값과 C.1 Orig 열의 일치
    res["V6b_table3_vs_C1"] = {
        "TTT_SwapPlace_T3": T3["SwapPlace"]["WAM-TTT"], "TTT_SwapPlace_C1_New": C1["WAM-TTT"]["New"][3], "TTT_SwapPlace_C1_Orig": C1["WAM-TTT"]["Orig"][3],
        "woTTT_SwapPlace_T3": T3["SwapPlace"]["w/o TTT"], "Cotrain_SwapPlace_C1_Orig": C1["WAM-Cotrain"]["Orig"][3],
        "LDA_SwapPlace_C1_New": C1["LDA"]["New"][3],
        "woTTT_TableBussing_T3": T3["TableBussing"]["w/o TTT"], "Cotrain_TableBussing_C1_Orig": C1["WAM-Cotrain"]["Orig"][1],
        "TTT_TableBussing_T3": T3["TableBussing"]["WAM-TTT"], "TTT_TableBussing_C1_New": C1["WAM-TTT"]["New"][1],
    }

    # ---- V11: E.1, E.2, E.3 산술 ----
    v11 = {"E1_avg": {}, "E1_deltas": {}}
    for k, v in E1.items():
        mean = float(np.mean(v[:3]))
        v11["E1_avg"][k] = {"recomputed": round(mean, 2), "reported": v[3], "match": abs(mean - v[3]) <= 0.05 + 1e-9}
    v11["E1_deltas"] = {"(100,100)-(100,0)": round(74.1 - 59.5, 1), "(100,100)-(200,0)": round(74.1 - 73.7, 1), "(100,200)-(100,100)": round(73.3 - 74.1, 1)}
    v11["E1_row100_100_equals_Table1_TTT"] = [E1["(100,100)"][0] == C1["WAM-TTT"]["New"][0], E1["(100,100)"][1] == C1["WAM-TTT"]["New"][1], E1["(100,100)"][2] == C1["WAM-TTT"]["New"][2]]
    v11["E1_row100_0_vs_LDA_New_avg3"] = [round(float(np.mean(E1["(100,0)"][:3])), 2), round(float(np.mean([C1["LDA"]["New"][i] for i in (0, 1, 2)])), 2)]
    v11["E2_deltas"] = {"DiT only": round(E2["DiT only"] - 100, 1), "no pretrain": round(E2["DiT+VLM (no pretrain)"] - 100, 1), "frozen": round(E2["DiT+VLM (frozen)"] - 100, 1),
                        "frozen_vs_dit_only": round(E2["DiT+VLM (frozen)"] - E2["DiT only"], 1)}
    for k, v in E3.items():
        v11[f"E3_{k}_avg"] = {"recomputed": round(float(np.mean(v[:4])), 3), "reported": v[4]}
    v11["E3_deltas"] = [round(a - b, 1) for a, b in zip(E3["VG only"][:4], E3["VG+FD"][:4])] + [round(72.3 - 28.9, 1)]
    v11["E3_VGonly_equals_C1_New"] = [E3["VG only"][0] == C1["WAM-TTT"]["New"][0], E3["VG only"][1] == C1["WAM-TTT"]["New"][1],
                                      E3["VG only"][2] == C1["WAM-TTT"]["New"][2], E3["VG only"][3] == C1["WAM-TTT"]["New"][3]]
    res["V11_appendix_arithmetic"] = v11

    # ---- V12: 데이터와 규모 산술 ----
    eps = {"G1": 600, "gripper": 544, "sharpa": 1142}
    res["V12_data_counts"] = {
        "sum_embodiments": sum(eps.values()), "reported_total": 2286,
        "per_task_avg_overall": round(2286 / 9, 1),
        "per_task_G1_3tasks": round(600 / 3, 1), "per_task_gripper_2tasks": round(544 / 2, 1), "per_task_sharpa_4tasks": round(1142 / 4, 1),
        "default_r_h_per_task": [100, 100], "default_pairs_if_100_per_task_x9": 900,
        "outer_samples_100k_x128": 100_000 * 128, "samples_per_episode": round(100_000 * 128 / 2286, 0),
        "robot_trials_main_table": 6 * 9 * 2 * 25,
        "d_over_H": 1536 / 32,
        "fast_weight_params_per_layer_if_per_head_separate": 32 * (48 * 128 + 128 * 48),
        "fast_weight_params_per_layer_if_shared": 48 * 128 + 128 * 48,
        "slow_proj_params_per_layer_if_full_dim": 4 * 1536 * 1536,
    }

    # ---- V13: 분산 상계 검정 ----
    def se_bound(m_pct, n):
        m = m_pct / 100
        return math.sqrt(max(m * (1 - m), 0) / n) * 100


    def z_test(a, b, n=25, k=1):
        """두 평균(%)의 차를 분산 상계로 검정. k는 평균을 이루는 셀 수(평균 열이면 9)."""
        sa, sb = se_bound(a, n), se_bound(b, n)
        z = (a - b) / math.hypot(sa, sb)
        return z, 2 * (1 - stats.norm.cdf(abs(z)))

    def z_avg(vals_a, vals_b, n=25):
        mean_a, mean_b = float(np.mean(vals_a)), float(np.mean(vals_b))
        va = sum((se_bound(x, n) ** 2) for x in vals_a) / len(vals_a) ** 2
        vb = sum((se_bound(x, n) ** 2) for x in vals_b) / len(vals_b) ** 2
        z = (mean_a - mean_b) / math.sqrt(va + vb)
        return mean_a - mean_b, z, 2 * (1 - stats.norm.cdf(abs(z)))

    v13 = {"per_cell_New": {}, "per_cell_Orig": {}, "avg_New": {}, "avg_Orig": {}, "paired_across_tasks": {}}
    for s, key in (("New", "per_cell_New"), ("Orig", "per_cell_Orig")):
        for m in ("LDA", "WAM-ICL", "WAM-Cotrain"):
            v13[key][m] = []
            for i, t in enumerate(TASKS):
                z, p = z_test(C1["WAM-TTT"][s][i], C1[m][s][i])
                v13[key][m].append({"task": t, "diff": round(C1["WAM-TTT"][s][i] - C1[m][s][i], 1), "z": round(z, 2), "p": round(p, 4)})
            d, z, p = z_avg(C1["WAM-TTT"][s][:9], C1[m][s][:9])
            v13["avg_" + s][m] = {"diff": round(d, 2), "z": round(z, 2), "p": float(f"{p:.3g}")}
            diffs = np.array(C1["WAM-TTT"][s][:9]) - np.array(C1[m][s][:9])
            t, pt = stats.ttest_1samp(diffs, 0.0)
            w = stats.wilcoxon(diffs)
            npos = int((diffs > 0).sum())
            sign_p = float(stats.binomtest(npos, 9, 0.5).pvalue)
            v13["paired_across_tasks"][f"{s}/{m}"] = {"mean_diff": round(float(diffs.mean()), 2), "sd": round(float(diffs.std(ddof=1)), 2),
                                                      "t": round(float(t), 2), "p_t": round(float(pt), 4), "p_wilcoxon": round(float(w.pvalue), 4),
                                                      "n_pos": npos, "p_sign": round(sign_p, 4)}
    res["V13_bound_tests"] = v13

    # 소수 점수 격자 (E.1 "통계적으로 구분되지 않음")
    d, z, p = z_avg(E1["(100,100)"][:3], E1["(200,0)"][:3])
    d2, z2, p2 = z_avg(E1["(100,100)"][:3], E1["(100,0)"][:3])
    d3, z3, p3 = z_avg(E1["(100,100)"][:3], E1["(10,190)"][:3])
    se_avg3 = math.sqrt(sum(se_bound(x, 25) ** 2 for x in E1["(100,100)"][:3]) / 9 + sum(se_bound(x, 25) ** 2 for x in E1["(200,0)"][:3]) / 9)
    res["V14_E1_tests"] = {
        "(100,100)_vs_(200,0)": {"diff": round(d, 2), "z": round(z, 2), "p": round(p, 3), "se_diff_bound": round(se_avg3, 2),
                                  "ci95_halfwidth_bound": round(1.96 * se_avg3, 2)},
        "(100,100)_vs_(100,0)": {"diff": round(d2, 2), "z": round(z2, 2), "p": round(p2, 3)},
        "(100,100)_vs_(10,190)": {"diff": round(d3, 2), "z": round(z3, 2), "p": round(p3, 4)},
    }
    # Table 3 (n = 10, 분산 상계), 관련 비교
    t3 = {}
    for task, d in T3.items():
        for k in ("WAM-LoRA", "w/o Meta", "w/o MemRecon", "w/o TTT"):
            z, p = z_test(d["WAM-TTT"], d[k], n=10)
            t3[f"{task}/TTT-vs-{k}"] = {"diff": round(d["WAM-TTT"] - d[k], 1), "z": round(z, 2), "p": round(p, 4)}
    res["V15_table3_tests_n10"] = t3
    # Table 4 (n = 25 가정)
    t4 = {}
    for k, d in T4.items():
        for b in ("LDA", "WAM-ICL", "pi05"):
            z, p = z_test(d["WAM-TTT"], d[b], n=25)
            t4[f"{k}/TTT-vs-{b}"] = {"diff": round(d["WAM-TTT"] - d[b], 1), "z": round(z, 2), "p": round(p, 4)}
    res["V16_table4_tests_n25_assumed"] = t4
    # Table 2 (n=25 가정)
    t2 = {}
    for k in ("WAM (no adapt.)", "LingBot-style ICL"):
        z, p = z_test(T2["WAM-TTT"], T2[k], n=25)
        t2[f"TTT-vs-{k}"] = {"diff": round(T2["WAM-TTT"] - T2[k], 1), "z": round(z, 2), "p": round(p, 4)}
    res["V17_table2_tests_n25_assumed"] = t2
    # E.2 (n=10, 분산 상계)
    e2 = {}
    for k, v in E2.items():
        if k == "DiT+VLM (open, ours)":
            continue
        z, p = z_test(100.0, v, n=10)
        e2[k] = {"diff": round(100.0 - v, 1), "z": round(z, 2), "p": round(p, 4)}
    res["V17b_E2_tests_n10"] = e2
    # E.3 (n=25)
    e3 = {}
    for j, t in enumerate(E3_TASKS):
        z, p = z_test(E3["VG only"][j], E3["VG+FD"][j], n=25)
        e3[t] = {"diff": round(E3["VG only"][j] - E3["VG+FD"][j], 1), "z": round(z, 2), "p": float(f"{p:.3g}")}
    res["V18_E3_tests"] = e3

    # ---- V19: Orig -> New에서 오른 칸 ----
    up = {}
    for m in METHODS:
        up[m] = [(TASKS[i], C1[m]["Orig"][i], C1[m]["New"][i]) for i in range(9) if C1[m]["New"][i] > C1[m]["Orig"][i]]
    res["V19_new_higher_than_orig"] = up

    # ---- V20: Table 4 vs 표 1 New (Deliver Drink) ----
    res["V20_table4_vs_table1_DD"] = {m: {"New_table1": C1[m]["New"][2], "lighting": T4["Lighting"][m if m != "pi05" else "pi05"], "spatial": T4["Spatial"][m]}
                                      for m in ("pi05", "LDA", "WAM-ICL", "WAM-TTT")}

    # ---- V21: Table 1 헤드라인 최악 가정에서의 TTT 대 LDA 평균 구간 ----
    d, z, p = z_avg(C1["WAM-TTT"]["New"][:9], C1["LDA"]["New"][:9])
    se = d / z
    res["V21_headline_interval"] = {"diff": round(d, 2), "se_bound": round(se, 2), "ci95": [round(d - 1.96 * se, 1), round(d + 1.96 * se, 1)]}

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))

    # ---- 요약 출력 ----
    print("V1 mismatches:", [k for k, v in res["V1_avg_column"].items() if not v["match_1dp"]])
    print("V2:", json.dumps(res["V2_text_deltas"], ensure_ascii=False))
    print("V3 New vs LDA:", res["V3_win_loss"]["New"]["LDA"])
    print("V3 New best-of-others:", res["V3_win_loss"]["New"]["best_of_all_others"])
    print("V3 Orig vs LDA:", res["V3_win_loss"]["Orig"]["LDA"], "ICL", res["V3_win_loss"]["Orig"]["WAM-ICL"]["losses"])
    print("V4 sums:", res["V4_rubric_sums"], res["V4_grid_units_0.01"])
    print("V4 infeasible:", v4["infeasible_at_n25"], "/", v4["total_cells"], "ninth:", v4["ninth_values_excluding_0_100"], v4["ninth_values_infeasible_at_25"])
    for k in ("V5_table2_n25_assumed", "V6_table3_n10", "V7_table4_n25_assumed", "V8_E1_n25", "V9_E2_n10", "V10_E3_n25"):
        print(k, res[k]["n_infeasible"], [(c["task"], c["value"]) for c in res[k]["cells"] if not c["feasible"]])
    print("V6b", res["V6b_table3_vs_C1"])
    print("V11", json.dumps(res["V11_appendix_arithmetic"], ensure_ascii=False))
    print("V12", res["V12_data_counts"])
    print("V13 avg New", res["V13_bound_tests"]["avg_New"], "Orig", res["V13_bound_tests"]["avg_Orig"])
    print("V13 paired", json.dumps(res["V13_bound_tests"]["paired_across_tasks"], ensure_ascii=False))
    print("V14", res["V14_E1_tests"])
    print("V15", res["V15_table3_tests_n10"])
    print("V16", res["V16_table4_tests_n25_assumed"])
    print("V17", res["V17_table2_tests_n25_assumed"])
    print("V17b", res["V17b_E2_tests_n10"])
    print("V18", res["V18_E3_tests"])
    print("V19", {k: len(v) for k, v in up.items()}, up["WAM-TTT"], up["WAM-Cotrain"])
    print("V21", res["V21_headline_interval"])


if __name__ == "__main__":
    main()
