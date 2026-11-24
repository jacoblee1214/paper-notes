"""논문 표·그림 수치를 다시 계산한다 (When Vision Overrides Language, arXiv:2602.17659 v3).

검사 번호 V1–V19는 03_analysis.md와 같다. 값은 논문 표를 옮긴 것이고, 시행 수는 논문 부록 B.1
("50 trials for each task", 스위트 과제 수 15/10/10/15)에서 가정한다. 가정은 결과 json에 적는다.

실행: python3 analysis/recompute.py   (numpy, scipy 필요, 수 초)
결과: analysis/recompute_results.json
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}

SUITES = ["CF-Spatial", "CF-Object", "CF-Long", "CF-OOD"]
TASKS = {"CF-Spatial": 15, "CF-Object": 10, "CF-Long": 10, "CF-OOD": 15}
TRIALS = 50
N_SUITE = {s: TASKS[s] * TRIALS for s in SUITES}  # 750, 500, 500, 750 (가정)


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def feasible(x, n, tol=0.0501):
    """x(%, 소수 한 자리)가 k/n*100의 반올림 값으로 나올 수 있는가."""
    k = round(x * n / 100)
    return any(abs(kk / n * 100 - x) <= tol for kk in (k - 1, k, k + 1) if 0 <= kk <= n)


# ---------------------------------------------------------------- 표 1
T1 = {  # [Spatial, Object, Goal, Long, Avg]
    "OpenVLA-OFT V+L": [97.6, 98.4, 97.9, 94.5, 97.1],
    "OpenVLA-OFT V-only": [83.2, 98.2, 10.0, 85.0, 69.1],
    "pi0 V+L": [96.8, 98.8, 95.8, 85.2, 94.2],
    "pi0 V-only": [28.0, 26.8, 3.4, 15.6, 18.5],
    "pi0.5 V+L": [98.8, 98.2, 98.0, 92.4, 96.9],
    "pi0.5 V-only": [68.2, 56.6, 10.2, 71.6, 51.7],
}
v1 = {}
for k, v in T1.items():
    m = float(np.mean(v[:4]))
    v1[k] = {"reported": v[4], "recomputed": round(m, 2), "match": abs(m - v[4]) < 0.06}
res["V1_table1_avg"] = v1
# 표 1 정수 검사: LIBERO 스위트당 500 에피소드(10과제 x 50) 가정 -> 값은 0.2의 배수여야 한다
res["V1b_table1_integer_500"] = {k: [bool(feasible(x, 500)) for x in v[:4]] for k, v in T1.items()}

# ---------------------------------------------------------------- 표 2
T2 = {  # model: (grounding spatial, focused, success spatial, focused)
    "OpenVLA-OFT": (6.8, 31.9, 1.1, 14.4),
    "pi0": (70.9, 92.7, 34.4, 51.1),
    "pi0.5": (39.3, 68.5, 24.4, 54.4),
}
avg = {"grounding_spatial": float(np.mean([T2[m][0] for m in T2])), "grounding_focused": float(np.mean([T2[m][1] for m in T2])),
       "success_spatial": float(np.mean([T2[m][2] for m in T2])), "success_focused": float(np.mean([T2[m][3] for m in T2]))}
res["V2_table2_avg"] = {"reported": {"g": [39.0, 64.4], "s": [20.0, 40.0]},
                         "recomputed": {k: round(v, 2) for k, v in avg.items()}}

# ---------------------------------------------------------------- 표 3 (VLA, CAG, 지표 -> 스위트별 [F, B] x 4 + 평균 [F, B])
CAG = ["base", "TF", "VA"]
T3 = {
    "OpenVLA-OFT": {
        "grounding": {
            "base": [6.8, 84.7, 2.0, 99.6, 4.0, 62.9, 6.0, 87.3, 4.7, 83.6],
            "TF": [11.1, 78.5, 1.6, 100.0, 4.9, 61.1, 5.6, 88.1, 5.8, 81.9],
            "VA": [26.5, 59.5, 3.8, 91.8, 6.9, 60.0, 7.9, 50.1, 11.3, 65.4]},
        "success": {
            "base": [1.1, 80.7, 0.0, 97.6, 0.2, 54.4, 0.1, 81.7, 0.4, 78.6],
            "TF": [3.3, 73.3, 0.0, 94.4, 0.2, 54.0, 0.1, 83.5, 0.9, 76.3],
            "VA": [7.9, 35.2, 0.0, 57.0, 0.4, 48.9, 0.1, 22.7, 2.1, 41.0]}},
    "pi0": {
        "grounding": {
            "base": [70.9, 37.1, 6.0, 59.4, 20.4, 83.1, 18.0, 72.4, 28.8, 63.0],
            "TF": [74.5, 32.5, 12.2, 50.6, 19.6, 81.8, 21.1, 72.4, 31.9, 59.3],
            "VA": [85.2, 24.4, 13.8, 53.8, 22.7, 82.2, 27.6, 47.5, 37.3, 52.0]},
        "success": {
            "base": [34.4, 19.1, 0.0, 45.0, 1.1, 63.1, 2.7, 52.9, 9.6, 45.0],
            "TF": [33.5, 14.4, 0.6, 15.0, 2.0, 38.7, 2.1, 42.3, 9.6, 27.6],
            "VA": [36.7, 6.1, 0.0, 29.2, 2.9, 48.7, 2.1, 19.2, 10.4, 25.8]}},
    "pi0.5": {
        "grounding": {
            "base": [39.3, 61.3, 11.4, 68.8, 51.6, 58.4, 20.7, 74.0, 30.8, 65.6],
            "TF": [52.0, 45.3, 24.6, 65.2, 60.7, 56.7, 24.8, 71.9, 40.5, 59.8],
            "VA": [50.7, 45.2, 34.0, 49.8, 64.2, 61.1, 36.4, 52.8, 46.3, 52.2]},
        "success": {
            "base": [24.4, 56.9, 5.8, 66.8, 15.8, 50.4, 6.9, 69.6, 13.2, 60.9],
            "TF": [27.3, 33.3, 9.2, 41.4, 20.9, 36.7, 9.9, 62.7, 16.8, 43.5],
            "VA": [31.6, 36.0, 18.0, 29.8, 26.7, 41.8, 10.3, 37.1, 21.7, 36.2]}},
}
# 부록 표 7 (X-VLA), 같은 형식
T7 = {
    "grounding": {
        "base": [22.1, 72.1, 1.4, 70.4, 98.2, 23.3, 27.3, 58.4, 37.3, 56.1],
        "TF": [22.3, 68.9, 0.0, 77.6, 95.6, 12.2, 40.9, 63.7, 39.7, 55.6],
        "VA": [24.4, 68.8, 2.0, 59.6, 96.7, 17.6, 40.8, 60.5, 41.0, 51.6]},
    "success": {
        "base": [3.2, 58.9, 0.0, 50.2, 47.6, 9.1, 4.4, 38.8, 13.8, 39.3],
        "TF": [8.9, 62.3, 0.0, 56.6, 59.1, 3.8, 10.1, 47.9, 19.5, 42.6],
        "VA": [8.7, 53.7, 0.0, 31.6, 51.1, 0.9, 8.7, 38.1, 17.1, 31.1]}}
ALL_T = dict(T3)
ALL_T["X-VLA"] = T7

# V3: 표 2와 표 3 기준선의 CF-Spatial 값 일치
t3_ok = {}
for m, key in (("OpenVLA-OFT", "OpenVLA-OFT"), ("pi0", "pi0"), ("pi0.5", "pi0.5")):
    t3_ok[m] = {"table2_g": T2[m][0], "table3_g": T3[m]["grounding"]["base"][0],
                "table2_s": T2[m][2], "table3_s": T3[m]["success"]["base"][0],
                "match": T2[m][0] == T3[m]["grounding"]["base"][0] and T2[m][2] == T3[m]["success"]["base"][0]}
res["V3_table2_vs_table3"] = t3_ok

# V4: 표 3 / 표 7 평균 열 = 스위트 평균의 단순 평균인가, 과제 수 가중인가
w = np.array([15, 10, 10, 15]) / 50.0
v4 = []
bad_simple = 0
cells = 0
for m, d in ALL_T.items():
    for metric, dd in d.items():
        for c, row in dd.items():
            for j, side in enumerate(("F", "B")):
                vals = np.array([row[2 * i + j] for i in range(4)])
                rep = row[8 + j]
                simple, weighted = float(vals.mean()), float((vals * w).sum())
                cells += 1
                ok = abs(simple - rep) < 0.06
                bad_simple += (not ok)
                v4.append({"model": m, "metric": metric, "cag": c, "side": side, "reported": rep,
                           "simple_mean": round(simple, 2), "task_weighted": round(weighted, 2), "simple_match": ok})
res["V4_average_column"] = {"cells": cells, "simple_mean_mismatch": bad_simple,
                            "mismatches": [r for r in v4 if not r["simple_match"]],
                            "note": "평균 열은 네 스위트 값의 단순 평균(가중 아님). 과제 수 가중(15/10/10/15)이면 값이 달라진다."}
res["V4_weighted_vs_simple_examples"] = [r for r in v4 if r["model"] == "pi0.5" and r["metric"] == "grounding" and r["side"] == "F"]

# V5: 정수 가능성. 스위트별 에피소드 수 N(가정)에서 보고 값이 k/N의 반올림으로 나오는가
v5 = {"assumed_N": N_SUITE, "by_suite": {}}
for si, s in enumerate(SUITES):
    vals = []
    for m, d in ALL_T.items():
        for metric, dd in d.items():
            for c, row in dd.items():
                for j in (0, 1):
                    vals.append((m, metric, c, "FB"[j], row[2 * si + j]))
    infeasible = [(m, me, c, sd, x) for (m, me, c, sd, x) in vals if not feasible(x, N_SUITE[s])]
    # 모든 값을 정수 비율로 설명하는 분모 후보(50의 배수와 1–3000 전체)
    ok_all = [n for n in range(1, 3001) if all(feasible(x, n) for (_, _, _, _, x) in vals)]
    ok_50 = [n for n in ok_all if n % 50 == 0]
    v5["by_suite"][s] = {"n_values": len(vals), "n_infeasible_at_assumed_N": len(infeasible),
                         "infeasible_examples": infeasible[:6],
                         "denominators_1to3000_consistent_with_all": ok_all[:12], "n_such_denominators": len(ok_all),
                         "multiples_of_50_consistent": ok_50[:12]}
res["V5_integer_feasibility"] = v5

# V6: 성공 <= 접촉(grounding), 같은 과제 F 성공 + B 성공 <= 100 (OR 종료 가정)
v6 = {"success_gt_grounding": [], "sum_success_gt_100": []}
for m, d in ALL_T.items():
    for c in CAG:
        g, s = d["grounding"][c], d["success"][c]
        for i in range(10):
            if s[i] > g[i] + 1e-9:
                v6["success_gt_grounding"].append((m, c, i, g[i], s[i]))
        for i in range(5):
            if s[2 * i] + s[2 * i + 1] > 100.0 + 1e-9:
                v6["sum_success_gt_100"].append((m, c, i, s[2 * i], s[2 * i + 1]))
res["V6_logical_bounds"] = v6
# 접촉 F + B > 100 인 칸 (한 에피소드에서 둘 다 접촉 가능하므로 허용되는 값)
res["V6b_touch_sum_gt_100"] = [(m, c, i, d["grounding"][c][2 * i], d["grounding"][c][2 * i + 1])
                              for m, d in ALL_T.items() for c in CAG for i in range(5)
                              if d["grounding"][c][2 * i] + d["grounding"][c][2 * i + 1] > 100.0 + 1e-9]

# V7: 본문 문장 수치
p05, p0, ofT = T3["pi0.5"], T3["pi0"], T3["OpenVLA-OFT"]
claims = {
    "OFT grounding F 4.7": (ofT["grounding"]["base"][8], 4.7), "OFT success F 0.4": (ofT["success"]["base"][8], 0.4),
    "pi0.5 grounding F 30.8": (p05["grounding"]["base"][8], 30.8), "pi0.5 success F 13.2": (p05["success"]["base"][8], 13.2),
    "OFT grounding B 83.6": (ofT["grounding"]["base"][9], 83.6), "OFT success B 78.6": (ofT["success"]["base"][9], 78.6),
    "pi0.5 grounding B 65.6": (p05["grounding"]["base"][9], 65.6), "pi0.5 success B 60.9": (p05["success"]["base"][9], 60.9),
}
dif = {
    "TF grounding gain 9.7": (round(p05["grounding"]["TF"][8] - p05["grounding"]["base"][8], 1), 9.7),
    "TF success gain 3.6": (round(p05["success"]["TF"][8] - p05["success"]["base"][8], 1), 3.6),
    "VA grounding gain 15.5": (round(p05["grounding"]["VA"][8] - p05["grounding"]["base"][8], 1), 15.5),
    "VA success gain 8.5": (round(p05["success"]["VA"][8] - p05["success"]["base"][8], 1), 8.5),
    "VA biased grounding drop 13.4": (round(p05["grounding"]["base"][9] - p05["grounding"]["VA"][9], 1), 13.4),
    "VA biased success drop 24.7": (round(p05["success"]["base"][9] - p05["success"]["VA"][9], 1), 24.7),
    "OOD VA grounding 36.4": (p05["grounding"]["VA"][6], 36.4), "OOD VA success 10.3": (p05["success"]["VA"][6], 10.3),
    "X-VLA TF grounding 37.3->39.7": (T7["grounding"]["TF"][8], 39.7), "X-VLA TF success 13.8->19.5": (T7["success"]["TF"][8], 19.5),
    "X-VLA VA grounding 41.0": (T7["grounding"]["VA"][8], 41.0), "X-VLA VA success 17.1": (T7["success"]["VA"][8], 17.1),
}
res["V7_text_claims"] = {k: {"recomputed": a, "text": b, "match": abs(a - b) < 0.051} for k, (a, b) in {**claims, **dif}.items()}

# V8: 표 4, 그림 4와 표 3의 대응
T4 = {"strat1": [39.3, 61.3, 51.6, 58.4, 24.4, 56.9, 15.8, 50.4], "strat2": [9.5, 89.3, 34.0, 83.1, 3.7, 86.5, 2.7, 71.3],
      "strat3": [52.0, 45.3, 60.7, 56.7, 27.3, 33.3, 20.9, 36.7], "strat4": [52.3, 47.5, 64.9, 59.6, 27.7, 37.1, 26.4, 41.6],
      "strat5": [50.7, 45.2, 64.2, 61.1, 31.6, 36.0, 26.7, 41.8]}


def from_t3(c):
    g, s = p05["grounding"][c], p05["success"][c]
    return [g[0], g[1], g[4], g[5], s[0], s[1], s[4], s[5]]


v8 = {"strat1==base": T4["strat1"] == from_t3("base"), "strat3==TF": T4["strat3"] == from_t3("TF"), "strat5==VA": T4["strat5"] == from_t3("VA"),
      "strat5-strat3 success CF-Spatial (text 4.3)": round(T4["strat5"][4] - T4["strat3"][4], 1),
      "strat5-strat4 success CF-Spatial": round(T4["strat5"][4] - T4["strat4"][4], 1)}
FIG4 = {"pi0.5_grounding": {"w": [1.0, 1.5, 2.0, 3.0], "v": [39.3, 50.7, 56.8, 58.4]},
        "pi0.5_success": {"w": [1.0, 1.5, 2.0, 3.0], "v": [24.4, 31.6, 28.8, 16.4]},
        "OFT_grounding": {"w": [1.0, 1.5, 2.0, 3.0, 4.0], "v": [6.8, 11.6, 18.4, 26.5, 30.4]},
        "OFT_success": {"w": [1.0, 1.5, 2.0, 3.0, 4.0], "v": [1.1, 2.9, 5.9, 7.9, 4.1]}}
v8["fig4_pi0.5_w1.5 == Table4 strat5 CF-Spatial (VA)"] = (FIG4["pi0.5_grounding"]["v"][1], FIG4["pi0.5_success"]["v"][1]) == (T4["strat5"][0], T4["strat5"][4])
v8["fig4_OFT_w3.0 == Table3 OFT VA CF-Spatial"] = (FIG4["OFT_grounding"]["v"][3], FIG4["OFT_success"]["v"][3]) == (ofT["grounding"]["VA"][0], ofT["success"]["VA"][0])
v8["fig4_w1.0 == base (pi0.5, OFT)"] = (FIG4["pi0.5_grounding"]["v"][0], FIG4["pi0.5_success"]["v"][0], FIG4["OFT_grounding"]["v"][0], FIG4["OFT_success"]["v"][0]) == (39.3, 24.4, 6.8, 1.1)
v8["fig4_pi0.5_best_success_w"] = FIG4["pi0.5_success"]["w"][int(np.argmax(FIG4["pi0.5_success"]["v"]))]
v8["fig4_OFT_best_success_w"] = FIG4["OFT_success"]["w"][int(np.argmax(FIG4["OFT_success"]["v"]))]
v8["fig4_note"] = "그림 4의 두 곡선은 CF-Spatial의 VA-CAG 값이다(표 3, 표 4와 일치). 같은 스위트가 보고 표에도 들어간다."
res["V8_table4_fig4"] = v8
res["FIG4_values_read_from_figure_labels"] = FIG4

# ---------------------------------------------------------------- 표 3 검정 (시행 수 가정, 비쌍 Fisher)
def counts(rate, n):
    return int(round(rate * n / 100.0))


tests = []
for m in T3:
    for metric in ("grounding", "success"):
        for c in ("TF", "VA"):
            for si, s in enumerate(SUITES):
                n = N_SUITE[s]
                b = T3[m][metric]["base"][2 * si]
                x = T3[m][metric][c][2 * si]
                kb, kx = counts(b, n), counts(x, n)
                tests.append({"model": m, "metric": metric, "cag": c, "suite": s, "n_assumed": n, "base_%": b, "cag_%": x,
                              "diff_pp": round(x - b, 1), "p_fisher": fisher(kx, n, kb, n)})
# Holm (족: 전체 48개)
ps = np.array([t["p_fisher"] for t in tests])
order = np.argsort(ps)
holm = np.empty(len(ps))
running = 0.0
for rank, idx in enumerate(order):
    running = max(running, min(1.0, (len(ps) - rank) * ps[idx]))
    holm[idx] = running
for t, h in zip(tests, holm):
    t["p_holm"] = float(h)
res["V9_table3_tests"] = {
    "assumption": "스위트당 에피소드 수 = 과제 수 x 50 (750/500/500/750), 같은 초기 상태를 쓰는 쌍체 구조는 반영하지 못해 비쌍 Fisher",
    "n_tests": len(tests),
    "n_p_lt_0.05": int((ps < 0.05).sum()), "n_holm_lt_0.05": int((holm < 0.05).sum()),
    "tests": tests}

# 합산(풀링) 헤드라인: 스위트 에피소드 수로 k를 복원해 합산 비율 (논문의 평균 열은 단순 평균이라 값이 다를 수 있다)
pooled = []
for m in ("OpenVLA-OFT", "pi0", "pi0.5"):
    for metric in ("grounding", "success"):
        for c in ("base", "TF", "VA"):
            k = sum(counts(T3[m][metric][c][2 * i], N_SUITE[s]) for i, s in enumerate(SUITES))
            n = sum(N_SUITE.values())
            lo, hi = wilson(k, n)
            pooled.append({"model": m, "metric": metric, "cag": c, "k": k, "n": n, "pooled_%": round(100 * k / n, 2),
                           "wilson95_%": [round(100 * lo, 2), round(100 * hi, 2)],
                           "reported_avg_%": T3[m][metric][c][8]})
res["V10_pooled_rates"] = pooled
pool_tests = []
for m in ("OpenVLA-OFT", "pi0", "pi0.5"):
    for metric in ("grounding", "success"):
        kb = [r for r in pooled if r["model"] == m and r["metric"] == metric and r["cag"] == "base"][0]["k"]
        for c in ("TF", "VA"):
            kx = [r for r in pooled if r["model"] == m and r["metric"] == metric and r["cag"] == c][0]["k"]
            n = 2500
            pool_tests.append({"model": m, "metric": metric, "cag": c, "k_base": kb, "k_cag": kx, "n": n,
                               "diff_pp": round(100 * (kx - kb) / n, 2), "p_fisher": fisher(kx, n, kb, n)})
res["V10_pooled_tests"] = pool_tests

# ---------------------------------------------------------------- 표 5, 6
T5 = {0: {"Left": (0, 0), "Right": (0, 0), "Plate": (3, 8), "Basket": (0, 0)},
      1: {"Left": (1, 3), "Right": (3, 4), "Plate": (2, 7), "Basket": (2, 6)},
      2: {"Left": (4, 8), "Right": (2, 5), "Plate": (8, 9), "Basket": (8, 8)}}
t5 = {}
for d, row in T5.items():
    kb = sum(v[0] for v in row.values())
    kx = sum(v[1] for v in row.values())
    t5[str(d)] = {"base_k_of_40": kb, "cag_k_of_40": kx, "base_%": 100 * kb / 40, "cag_%": 100 * kx / 40,
                  "fisher_p_unpaired": fisher(kx, 40, kb, 40)}
res["V11_table5"] = {"by_demos": t5,
                     "non_monotone_cells_base": [(t, d) for t in ("Left", "Right", "Plate", "Basket") for d in (1, 2)
                                                  if T5[d][t][0] < T5[d - 1][t][0]],
                     "non_monotone_cells_cag": [(t, d) for t in ("Left", "Right", "Plate", "Basket") for d in (1, 2)
                                                 if T5[d][t][1] < T5[d - 1][t][1]],
                     "one_demo_equals_main_experiment": {t: T5[1][t] for t in T5[1]}}
res["V12_table6"] = {"latency_ratio_TF": round(126.6 / 80.0, 3), "latency_ratio_VA": round(132.6 / 80.0, 3),
                     "memory_ratio_VA": round(16.5 / 8.5, 3),
                     "note": "표기 1.6x, 1.7x, 1.9x와 일치. TF 126.6 ms는 같은 GPU에서 순차 두 번 호출한 값으로 보이지만 논문은 병렬이라고 적는다(코드 D 항목 참조)"}

# ---------------------------------------------------------------- 실세계 표 9, 10
# 각 칸: instruction별 {실행한 대상 열: (성공, 시도)}, fail 수. 열 이름은 장면별 대상 순서(첫째가 in-domain)
RW = {
    "Coke/Sprite/Fanta": {"cols": ["Coke", "Sprite", "Fanta"],
        "base": [({"Coke": (8, 10)}, 0), ({"Sprite": (6, 10)}, 0), ({"Coke": (1, 1), "Fanta": (6, 9)}, 0)],
        "cag": [({"Coke": (9, 10)}, 0), ({"Sprite": (8, 10)}, 0), ({"Fanta": (7, 10)}, 0)]},
    "Tape/Mustard/Pringles": {"cols": ["Tape", "Mustard", "Pringles"],
        "base": [({"Tape": (10, 10)}, 0), ({"Tape": (3, 3), "Mustard": (3, 6), "Pringles": (0, 1)}, 0), ({"Mustard": (1, 2), "Pringles": (2, 8)}, 0)],
        "cag": [({"Tape": (9, 10)}, 0), ({"Mustard": (6, 10)}, 0), ({"Pringles": (4, 10)}, 0)]},
    "Middle/Left/Right": {"cols": ["Middle", "Left", "Right"],
        "base": [({"Middle": (9, 10)}, 0), ({"Middle": (8, 8), "Left": (1, 2)}, 0), ({"Middle": (7, 7), "Right": (3, 3)}, 0)],
        "cag": [({"Middle": (10, 10)}, 0), ({"Middle": (6, 7), "Left": (3, 3)}, 0), ({"Middle": (4, 5), "Left": (1, 1), "Right": (4, 4)}, 0)]},
    "Table/Plate/Bowl": {"cols": ["Table", "Plate", "Bowl"],
        "base": [({"Table": (7, 9), "Plate": (1, 1)}, 0), ({"Plate": (6, 8), "Bowl": (2, 2)}, 0), ({"Plate": (4, 5), "Bowl": (4, 5)}, 0)],
        "cag": [({"Table": (8, 10)}, 0), ({"Plate": (7, 7), "Bowl": (3, 3)}, 0), ({"Plate": (3, 4), "Bowl": (6, 6)}, 0)]},
    "Stack/Plate/Basket": {"cols": ["Stack", "Plate", "Basket"],
        "base": [({"Stack": (5, 9), "Plate": (1, 1)}, 0), ({"Stack": (0, 1), "Plate": (2, 7)}, 2), ({"Basket": (2, 10)}, 0)],
        "cag": [({"Stack": (7, 10)}, 0), ({"Plate": (7, 10)}, 0), ({"Basket": (6, 9)}, 1)]},
    "Cup/Cube/Basketball": {"cols": ["Cup", "Cube", "Basketball"],
        "base": [({"Cup": (9, 10)}, 0), ({"Cup": (0, 1), "Cube": (5, 6), "Basketball": (3, 3)}, 0), ({"Cup": (3, 3), "Basketball": (6, 7)}, 0)],
        "cag": [({"Cup": (9, 10)}, 0), ({"Cube": (8, 9), "Basketball": (1, 1)}, 0), ({"Cup": (1, 2), "Basketball": (7, 8)}, 0)]},
    "Move and Pour": {"cols": ["Coke", "Sprite", "Fanta"], "long": True,
        "base": [({"Coke": (5, 9), "Sprite": (0, 1)}, 0), ({"Coke": (1, 2), "Sprite": (5, 8)}, 0), ({"Coke": (2, 3), "Fanta": (3, 7)}, 0)],
        "cag": [({"Coke": (6, 9), "Fanta": (1, 1)}, 0), ({"Sprite": (5, 10)}, 0), ({"Sprite": (1, 1), "Fanta": (3, 9)}, 0)]},
    "Apple and Banana": {"cols": ["A+B", "B+A", "A"], "long": True,
        "base": [({"A+B": (9, 10)}, 0), ({"A+B": (4, 4), "B+A": (5, 6)}, 0), ({"A+B": (8, 8), "A": (2, 2)}, 0)],
        "cag": [({"A+B": (8, 10)}, 0), ({"A+B": (2, 3), "B+A": (4, 7)}, 0), ({"A+B": (3, 3), "A": (7, 7)}, 0)]},
}
# 그림 5, 7에서 읽은 평균 막대(그림 판독 값): (grounding base, success base, grounding cag, success cag)
FIG5_AVG = {"Coke/Sprite/Fanta": (96.7, 66.7, 100.0, 80.0), "Tape/Mustard/Pringles": (80.0, 50.0, 100.0, 63.3),
            "Middle/Left/Right": (50.0, 43.3, 56.7, 56.7), "Table/Plate/Bowl": (73.3, 56.7, 76.7, 70.0),
            "Stack/Plate/Basket": (86.7, 30.0, 96.7, 66.7), "Cup/Cube/Basketball": (76.7, 66.7, 90.0, 80.0),
            "Move and Pour": (80.0, 43.3, 93.3, 46.7), "Apple and Banana": (60.0, 53.3, 80.0, 63.3)}

rw_rows = {}
row_sum_bad = []
per_inst = {}
for scene, d in RW.items():
    cols = d["cols"]
    out = {}
    for model in ("base", "cag"):
        gs, ss, biased_succ, biased_att = [], [], [], []
        for i, (cells, fail) in enumerate(d[model]):
            tgt = cols[i]
            att = sum(v[1] for v in cells.values()) + fail
            if att != 10:
                row_sum_bad.append((scene, model, i, att))
            g = cells.get(tgt, (0, 0))[1]
            s = cells.get(tgt, (0, 0))[0]
            gs.append(g)
            ss.append(s)
            biased_succ.append(sum(v[0] for c, v in cells.items() if c != tgt))
            biased_att.append(sum(v[1] for c, v in cells.items() if c != tgt))
        out[model] = {"grounding_k": gs, "success_k": ss, "grounding_avg_%": round(sum(gs) * 10 / 3, 1),
                      "success_avg_%": round(sum(ss) * 10 / 3, 1),
                      "wrong_target_success_k": biased_succ, "wrong_target_attempts_k": biased_att}
    rw_rows[scene] = out
res["V13_realworld_row_sums_not_10"] = row_sum_bad

cmp_fig = {}
for scene, (gb, sb, gc, sc) in FIG5_AVG.items():
    r = rw_rows[scene]
    cmp_fig[scene] = {"table_9_10": [r["base"]["grounding_avg_%"], r["base"]["success_avg_%"], r["cag"]["grounding_avg_%"], r["cag"]["success_avg_%"]],
                      "figure_read": [gb, sb, gc, sc],
                      "match": all(abs(a - b) < 0.06 for a, b in zip([r["base"]["grounding_avg_%"], r["base"]["success_avg_%"], r["cag"]["grounding_avg_%"], r["cag"]["success_avg_%"]], [gb, sb, gc, sc]))}
res["V14_tables9_10_vs_fig5_fig7_averages"] = cmp_fig

main6 = [s for s in RW if not RW[s].get("long")]
all8 = list(RW)


def avg_over(scenes, model, key):
    return float(np.mean([rw_rows[s][model][key] for s in scenes]))


claims_rw = {}
for name, sc in (("6 main scenes", main6), ("8 scenes (+ long-horizon)", all8)):
    claims_rw[name] = {"grounding_base": round(avg_over(sc, "base", "grounding_avg_%"), 2), "grounding_cag": round(avg_over(sc, "cag", "grounding_avg_%"), 2),
                       "success_base": round(avg_over(sc, "base", "success_avg_%"), 2), "success_cag": round(avg_over(sc, "cag", "success_avg_%"), 2)}
    claims_rw[name]["grounding_gain"] = round(claims_rw[name]["grounding_cag"] - claims_rw[name]["grounding_base"], 2)
    claims_rw[name]["success_gain"] = round(claims_rw[name]["success_cag"] - claims_rw[name]["success_base"], 2)
claims_rw["text_claims"] = {"abstract/Fig.1d: grounding +9.4, success +17.2 (52.2 -> 69.4)": "6개 주요 장면의 세 지시 평균(in-domain 포함)과 일치"}

# 장면별 지시 단위 (under-observed 두 개만) 합산과 검정
under = {"base": {"s": 0, "g": 0}, "cag": {"s": 0, "g": 0}}
indom = {"base": {"s": 0, "g": 0}, "cag": {"s": 0, "g": 0}}
under6 = {"base": {"s": 0, "g": 0}, "cag": {"s": 0, "g": 0}}
for scene, d in RW.items():
    cols = d["cols"]
    for model in ("base", "cag"):
        for i, (cells, fail) in enumerate(d[model]):
            tgt = cols[i]
            s, g = cells.get(tgt, (0, 0))[0], cells.get(tgt, (0, 0))[1]
            tgtdict = indom if i == 0 else under
            tgtdict[model]["s"] += s
            tgtdict[model]["g"] += g
            if i > 0 and scene in main6:
                under6[model]["s"] += s
                under6[model]["g"] += g
claims_rw["under_observed_pooled_8_scenes_n160"] = {
    "success_base_k": under["base"]["s"], "success_cag_k": under["cag"]["s"], "n": 160,
    "p_fisher_success": fisher(under["cag"]["s"], 160, under["base"]["s"], 160),
    "grounding_base_k": under["base"]["g"], "grounding_cag_k": under["cag"]["g"],
    "p_fisher_grounding": fisher(under["cag"]["g"], 160, under["base"]["g"], 160),
    "wilson95_success_base": [round(x, 3) for x in wilson(under["base"]["s"], 160)],
    "wilson95_success_cag": [round(x, 3) for x in wilson(under["cag"]["s"], 160)]}
claims_rw["under_observed_pooled_6_main_n120"] = {
    "success_base_k": under6["base"]["s"], "success_cag_k": under6["cag"]["s"], "n": 120,
    "p_fisher_success": fisher(under6["cag"]["s"], 120, under6["base"]["s"], 120),
    "grounding_base_k": under6["base"]["g"], "grounding_cag_k": under6["cag"]["g"],
    "p_fisher_grounding": fisher(under6["cag"]["g"], 120, under6["base"]["g"], 120)}
claims_rw["in_domain_pooled_8_scenes_n80"] = {
    "success_base_k": indom["base"]["s"], "success_cag_k": indom["cag"]["s"], "n": 80,
    "p_fisher_success": fisher(indom["cag"]["s"], 80, indom["base"]["s"], 80)}
res["V15_realworld_claims"] = claims_rw

# 장면별 under-observed 합산(n=20)
scene_tests = {}
for scene, d in RW.items():
    cols = d["cols"]
    sb = sum(d["base"][i][0].get(cols[i], (0, 0))[0] for i in (1, 2))
    sc = sum(d["cag"][i][0].get(cols[i], (0, 0))[0] for i in (1, 2))
    gb = sum(d["base"][i][0].get(cols[i], (0, 0))[1] for i in (1, 2))
    gc = sum(d["cag"][i][0].get(cols[i], (0, 0))[1] for i in (1, 2))
    scene_tests[scene] = {"success": [sb, sc], "grounding": [gb, gc], "n": 20,
                          "p_success": fisher(sc, 20, sb, 20), "p_grounding": fisher(gc, 20, gb, 20)}
res["V16_scene_tests_under_observed"] = scene_tests

# 본문 문장 검사 (실세계)
text_rw = {}
mlr, tpb = rw_rows["Middle/Left/Right"], rw_rows["Table/Plate/Bowl"]
text_rw["Spatial: 'Plate 60% and Bowl 40% grounding' -> recomputed pi0.5 grounding Plate/Bowl, success Plate/Bowl"] = {
    "grounding_Plate_Bowl_%": [tpb["base"]["grounding_k"][1] * 10, tpb["base"]["grounding_k"][2] * 10],
    "success_Plate_Bowl_%": [tpb["base"]["success_k"][1] * 10, tpb["base"]["success_k"][2] * 10]}
text_rw["Spatial: 'Left 20% and Right 30% grounding'"] = {"grounding_Left_Right_%": [mlr["base"]["grounding_k"][1] * 10, mlr["base"]["grounding_k"][2] * 10]}
sp_g = np.mean([[mlr["cag"]["grounding_avg_%"] - mlr["base"]["grounding_avg_%"], tpb["cag"]["grounding_avg_%"] - tpb["base"]["grounding_avg_%"]]])
sp_s = np.mean([[mlr["cag"]["success_avg_%"] - mlr["base"]["success_avg_%"], tpb["cag"]["success_avg_%"] - tpb["base"]["success_avg_%"]]])
text_rw["Spatial: 'grounding +16.6, success +13.3' (두 장면 평균 증가)"] = {
    "grounding_gain_each_scene": [round(mlr["cag"]["grounding_avg_%"] - mlr["base"]["grounding_avg_%"], 1), round(tpb["cag"]["grounding_avg_%"] - tpb["base"]["grounding_avg_%"], 1)],
    "success_gain_each_scene": [round(mlr["cag"]["success_avg_%"] - mlr["base"]["success_avg_%"], 1), round(tpb["cag"]["success_avg_%"] - tpb["base"]["success_avg_%"], 1)],
    "mean_grounding_gain": round(float(sp_g), 2), "mean_success_gain": round(float(sp_s), 2)}
# 어떤 정의로 16.6이 나오는가 탐색: under-observed 지시만의 grounding 증가(장면별, 합)
cand = {}
for scene in ("Middle/Left/Right", "Table/Plate/Bowl"):
    r = rw_rows[scene]
    cand[scene] = {"under_only_grounding_gain": round(np.mean(r["cag"]["grounding_k"][1:]) * 10 - np.mean(r["base"]["grounding_k"][1:]) * 10, 1),
                   "under_only_success_gain": round(np.mean(r["cag"]["success_k"][1:]) * 10 - np.mean(r["base"]["success_k"][1:]) * 10, 1)}
text_rw["Spatial: under-observed 지시만의 증가 (탐색)"] = cand
spb = rw_rows["Stack/Plate/Basket"]
text_rw["Goal: 'grounding 96.7%, success +36.7'"] = {"cag_grounding_avg": spb["cag"]["grounding_avg_%"], "success_gain": round(spb["cag"]["success_avg_%"] - spb["base"]["success_avg_%"], 1)}
tmp = rw_rows["Tape/Mustard/Pringles"]
text_rw["Object: 'CAG 100% avg grounding, success +13.3 in both scenes'"] = {
    "Coke": [rw_rows["Coke/Sprite/Fanta"]["cag"]["grounding_avg_%"], round(rw_rows["Coke/Sprite/Fanta"]["cag"]["success_avg_%"] - rw_rows["Coke/Sprite/Fanta"]["base"]["success_avg_%"], 1)],
    "Tape": [tmp["cag"]["grounding_avg_%"], round(tmp["cag"]["success_avg_%"] - tmp["base"]["success_avg_%"], 1)]}
ccb = rw_rows["Cup/Cube/Basketball"]
text_rw["OOD: 'Cube 50%, Basketball 60% success; 4 of 5 Cube failures counterfactual'"] = {
    "success_Cube_Basketball_%": [ccb["base"]["success_k"][1] * 10, ccb["base"]["success_k"][2] * 10],
    "Cube row: wrong-object attempts (cup 1 + basketball 3)": ccb["base"]["wrong_target_attempts_k"][1],
    "Cube row failures (10 - 5 successes)": 10 - ccb["base"]["success_k"][1]}
mp, cs = rw_rows["Move and Pour"], rw_rows["Coke/Sprite/Fanta"]
text_rw["Long: 'Move and Pour failures more frequent than Coke/Sprite/Fanta'"] = {
    "Move&Pour base success_avg/grounding_avg": [mp["base"]["success_avg_%"], mp["base"]["grounding_avg_%"]],
    "Coke/Sprite/Fanta base success_avg/grounding_avg": [cs["base"]["success_avg_%"], cs["base"]["grounding_avg_%"]],
    "Move&Pour CAG success_avg": mp["cag"]["success_avg_%"], "Coke/Sprite/Fanta CAG success_avg": cs["cag"]["success_avg_%"],
    "Move&Pour in-domain (Coke) base success %": mp["base"]["success_k"][0] * 10}
# 위험: 실세계 CAG 오염: 그림 1d 'Biased' 정의 탐색
for name, sc in (("6 main scenes", main6),):
    for var in ("wrong_target_success_k", "wrong_target_attempts_k"):
        pass
bias = {}
for model in ("base", "cag"):
    tot_s = sum(sum(rw_rows[s][model]["wrong_target_success_k"][1:]) for s in main6)  # under-observed 지시에서 다른 대상 성공
    tot_a = sum(sum(rw_rows[s][model]["wrong_target_attempts_k"][1:]) for s in main6)
    tot_s_all = sum(sum(rw_rows[s][model]["wrong_target_success_k"]) for s in main6)
    bias[model] = {"wrong_target_success_under_observed_k_of_120": tot_s, "wrong_target_attempts_under_observed_k_of_120": tot_a,
                   "wrong_target_success_all_instr_k_of_180": tot_s_all,
                   "pct_under_120": round(100 * tot_s / 120, 1), "pct_all_180": round(100 * tot_s_all / 180, 1),
                   "pct_attempts_under_120": round(100 * tot_a / 120, 1)}
text_rw["Fig.1d real-world 'Biased 18.9 -> 10.6' 후보 정의 (참고, 확정 못 함)"] = bias
res["V17_realworld_text_checks"] = text_rw
res["real_world_per_scene_rows"] = rw_rows

# 그림 1(d) 대응
res["V18_fig1d"] = {
    "sim_faithful_success_13.2->21.7": (p05["success"]["base"][8], p05["success"]["VA"][8]),
    "sim_biased_success_60.9->36.2": (p05["success"]["base"][9], p05["success"]["VA"][9]),
    "real_faithful_success_52.2->69.4(6 main scenes, 3 instructions incl in-domain)": (round(claims_rw["6 main scenes"]["success_base"], 1), round(claims_rw["6 main scenes"]["success_cag"], 1)),
    "note": "시뮬은 VA-CAG, 실세계 faithful은 in-domain 지시를 포함한 6개 장면 평균"}


# V19: 지시된 과제(F)와 훈련 과제(B) 성공의 합 = 두 과제 중 하나라도 끝낸 에피소드 비율 (OR 종료라 둘은 겹치지 않는다고 가정)
v19 = {}
for m, d in ALL_T.items():
    v19[m] = {}
    for c in CAG:
        s = d["success"][c]
        v19[m][c] = {SUITES[i]: round(s[2 * i] + s[2 * i + 1], 1) for i in range(4)}
        v19[m][c]["avg"] = round(s[8] + s[9], 1)
res["V19_F_plus_B_success"] = {"values": v19,
    "note": "성공 합 = F 성공 + B 성공. CAG가 F를 올리면서 합을 낮추면 기존 과제 완수가 줄어든 만큼 지시 과제로 옮겨 가지 못한 것이다."}
res["V19_counts_model_suite_cells"] = {c: {"drops": sum(1 for m in v19 for k in SUITES if v19[m][c][k] < v19[m]["base"][k] - 1e-9), "cells": len(v19) * len(SUITES)} for c in ("TF", "VA")}
res["V19_cells_where_sum_drops_with_CAG"] = [
    {"model": m, "suite": k, "base": v19[m]["base"][k], "TF": v19[m]["TF"][k], "VA": v19[m]["VA"][k]}
    for m in v19 for k in SUITES + ["avg"] if min(v19[m]["TF"][k], v19[m]["VA"][k]) < v19[m]["base"][k] - 1e-9]

OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
print("wrote", OUT)
print("V4 mismatches:", res["V4_average_column"]["simple_mean_mismatch"], "of", res["V4_average_column"]["cells"])
for s in SUITES:
    b = res["V5_integer_feasibility"]["by_suite"][s]
    print(s, "N", N_SUITE[s], "infeasible", b["n_infeasible_at_assumed_N"], "/", b["n_values"], "consistent denominators", b["denominators_1to3000_consistent_with_all"][:6], "(", b["n_such_denominators"], ")")
print("V6", res["V6_logical_bounds"])
print("Table3 tests:", res["V9_table3_tests"]["n_tests"], "p<.05", res["V9_table3_tests"]["n_p_lt_0.05"], "Holm<.05", res["V9_table3_tests"]["n_holm_lt_0.05"])
print("RW", json.dumps(claims_rw, ensure_ascii=False, default=str)[:900])
print("row sums bad:", row_sum_bad)
print("fig cmp mismatches:", [k for k, v in cmp_fig.items() if not v["match"]])
