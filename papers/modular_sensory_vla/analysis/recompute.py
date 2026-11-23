"""MoSS (arXiv:2604.23272 v2) 보고 수치 재계산과 검정.

표 1, 2, 3, 4, 6, 7, 8, 9의 값을 옮겨 적고 (1) 정수 격자 검사 (2) 평균 재계산 (3) ± 값 검사
(4) 본문 서술 수치 검사 (5) Fisher, Wilson, CMH, Holm 검정 (6) 검정력 계산을 한다. 학습이나 실험은 하지 않는다.
시행 수는 논문 부록 A.1에 있는 값(과제당 24회)을 쓰고, 표 7(RLBench)은 시행 수가 없어
값들이 150의 배수 격자에 놓이는 점으로 150을 가정한다(문장에 "가정"으로 표시).

실행: python3 analysis/recompute.py   → analysis/recompute_results.json (numpy, scipy, statsmodels, 수 초)
"""
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable

OUT = Path(__file__).resolve().parent / "recompute_results.json"
Z = 1.959964

TASKS = ["Unstack Cup", "PnP Egg", "Board Erase", "Plug Insertion"]
# 표 1 (v2): (모델 묶음, 방법, 촉각, 토크, 4과제, 보고 평균, 보고 ±)
T1 = [
    ("GR00T", "base", 0, 0, [16.7, 45.8, 20.8, 0.0], 20.8, 4.1),
    ("GR00T", "Tactile-VLA", 1, 0, [29.2, 45.8, 33.3, 12.5], 30.2, 4.7),
    ("GR00T", "ForceVLA", 1, 0, [37.5, 54.2, 33.3, 12.5], 34.4, 4.8),
    ("GR00T", "MoSS", 1, 0, [45.8, 66.7, 41.7, 16.7], 42.7, 5.0),
    ("GR00T", "TA-VLA", 0, 1, [29.2, 45.8, 37.5, 20.8], 33.3, 4.7),
    ("GR00T", "MoSS", 0, 1, [33.3, 50.0, 41.7, 25.0], 37.5, 4.9),
    ("GR00T", "MoSS", 1, 1, [54.2, 66.7, 50.0, 25.0], 49.0, 5.1),
    ("pi0", "base", 0, 0, [12.5, 50.0, 29.2, 16.7], 27.1, 4.5),
    ("pi0", "Tactile-VLA", 1, 0, [16.7, 50.0, 37.5, 12.5], 29.2, 4.6),
    ("pi0", "ForceVLA", 1, 0, [25.0, 62.5, 50.0, 16.7], 38.6, 5.0),
    ("pi0", "MoSS", 1, 0, [29.2, 62.5, 41.7, 20.8], 38.6, 5.0),
    ("pi0", "TA-VLA", 0, 1, [16.7, 58.3, 41.7, 20.8], 34.4, 4.8),
    ("pi0", "MoSS", 0, 1, [20.8, 62.5, 54.2, 29.2], 41.7, 5.0),
    ("pi0", "MoSS", 1, 1, [29.2, 66.7, 58.3, 29.2], 45.9, 5.1),
]
# v1의 π0 기본 평균 (v2에서 27.1로 바뀜)
V1_PI0_BASE_AVG = 26.1

# 표 8: (방법, 촉각, 토크, 4과제, 평균, 괄호 안 변화 4과제 + 평균)
T8 = {
    "Tactile-VLA": ([12.5, 37.5, 16.7, 16.7], 20.9, [-16.7, -8.3, -16.6, 4.2, -9.3], [29.2, 45.8, 33.3, 12.5], 30.2),
    "ForceVLA": ([25.0, 33.3, 20.8, 4.2], 21.9, [-12.5, -20.9, -12.5, -8.3, -12.5], [37.5, 54.2, 33.3, 12.5], 34.4),
    "TA-VLA": ([20.8, 41.7, 29.2, 16.7], 27.1, [-8.4, -4.1, -8.3, -4.1, -6.2], [29.2, 45.8, 37.5, 20.8], 33.3),
    # MoSS의 비교 기준은 칸별로 더 높은 단일 모달리티(촉각 또는 토크) 값과 맞는다
    "MoSS": ([54.2, 66.7, 50.0, 25.0], 49.0, [8.4, -0.0, 8.3, -0.0, 6.3], [45.8, 66.7, 41.7, 25.0], 42.7),
}
# 표 9: GR00T/π0 별 [Small, Big, PnP, Low, Middle, High, Yellow, White, Black], 평균 (v2)
T9_COLS = ["Small", "Big", "PnP", "Low", "Middle", "High", "Yellow", "White", "Black"]
T9_N = [12, 12, 24, 8, 8, 8, 8, 8, 8]
T9 = [
    ("GR00T", "base", 0, 0, [0.0, 33.3, 45.8, 0.0, 12.5, 50.0, 0.0, 0.0, 0.0], 20.8),
    ("GR00T", "Tactile-VLA", 1, 0, [8.3, 50.0, 45.8, 12.5, 37.5, 50.0, 25.0, 12.5, 0.0], 30.2),
    ("GR00T", "ForceVLA", 1, 0, [16.7, 58.3, 54.2, 12.5, 37.5, 50.0, 12.5, 25.0, 0.0], 34.4),
    ("GR00T", "MoSS", 1, 0, [25.0, 66.7, 66.7, 37.5, 50.0, 37.5, 25.0, 12.5, 12.5], 42.7),
    ("GR00T", "TA-VLA", 0, 1, [8.3, 50.0, 45.8, 37.5, 50.0, 25.0, 25.0, 25.0, 12.5], 33.3),
    ("GR00T", "MoSS", 0, 1, [8.3, 58.3, 50.0, 37.5, 62.5, 25.0, 25.0, 37.5, 12.5], 37.5),
    ("GR00T", "MoSS", 1, 1, [25.0, 83.3, 66.7, 37.5, 62.5, 50.0, 37.5, 25.0, 12.5], 49.0),
    ("pi0", "base", 0, 0, [0.0, 25.0, 50.0, 12.5, 37.5, 37.5, 25.0, 12.5, 12.5], 27.1),
    ("pi0", "Tactile-VLA", 1, 0, [0.0, 33.3, 50.0, 25.0, 37.5, 50.0, 25.0, 12.5, 0.0], 29.2),
    ("pi0", "ForceVLA", 1, 0, [16.7, 33.3, 62.5, 50.0, 50.0, 50.0, 25.0, 12.5, 12.5], 38.6),
    ("pi0", "MoSS", 1, 0, [8.3, 50.0, 62.5, 25.0, 50.0, 50.0, 25.0, 25.0, 12.5], 38.6),
    ("pi0", "TA-VLA", 0, 1, [8.3, 25.0, 58.3, 12.5, 37.5, 75.0, 37.5, 12.5, 12.5], 34.4),
    ("pi0", "MoSS", 0, 1, [0.0, 41.7, 62.5, 50.0, 50.0, 62.5, 50.0, 25.0, 12.5], 41.7),
    ("pi0", "MoSS", 1, 1, [8.3, 50.0, 66.7, 62.5, 50.0, 62.5, 37.5, 25.0, 25.0], 45.9),
]
# v1 표 7에서 π0 + MoSS(torque)의 Big 값은 41.6 (v2는 41.7)

# RLBench 표 7: [Open Window, Lamp On, Move Hanger, Push Button, Basketball, Phone on Base, Pick Up Cup, Slide Block, Take Lid Off]
T7_COLS = ["Open Window", "Lamp On", "Move Hanger", "Push Button", "Basketball", "Phone on Base", "Pick Up Cup",
           "Slide Block", "Take Lid Off"]
T7 = {"GR00T": [8.0, 36.0, 29.3, 36.0, 72.0, 20.7, 50.0, 18.7, 72.7],
      "MoSS": [9.3, 47.3, 45.3, 50.7, 70.0, 22.7, 52.7, 21.3, 68.7]}
T7_REPORTED = {"GR00T": (27.3, 46.8, 38.1), "MoSS": (38.2, 47.1, 43.1)}
T7_N = 150  # 가정

R = {}


def r1(x):
    return round(float(x), 1)


def wilson(k, n, z=Z):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [100 * (c - h), 100 * (c + h)]


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def count(v, n):
    k = round(v * n / 100)
    return k, abs(round(100 * k / n, 1) - v) < 0.051


def hdr(row):
    return f"{row[0]}|{row[1]}|T{row[2]}|Q{row[3]}"


# ---------------- V1: 표 1 정수 격자 (24회) ----------------
v1 = {}
bad = []
for row in T1:
    cs = []
    for t, v in zip(TASKS, row[4]):
        k, ok = count(v, 24)
        cs.append(k)
        if not ok:
            bad.append((hdr(row), t, v))
    v1[hdr(row)] = cs
R["V1_table1_counts_n24"] = {"counts": v1, "off_grid_cells": bad}

# ---------------- V2: 평균 재계산 ----------------
v2 = {}
for row in T1:
    cs = v1[hdr(row)]
    tot = sum(cs)
    exact = 100 * tot / 96
    mean_of_rounded = sum(row[4]) / 4
    v2[hdr(row)] = {
        "total_of_96": tot, "exact_avg": round(exact, 2), "avg_of_rounded_cells": round(mean_of_rounded, 2),
        "reported_avg": row[5], "match_exact": abs(round(exact, 1) - row[5]) < 0.051,
        "match_rounded": abs(round(mean_of_rounded + 1e-9, 1) - row[5]) < 0.051,
    }
R["V2_table1_avg"] = v2
R["V2_v1_pi0_base"] = {"v1_reported": V1_PI0_BASE_AVG, "exact": round(100 * sum(v1["pi0|base|T0|Q0"]) / 96, 2),
                       "counts": v1["pi0|base|T0|Q0"], "note": "v2는 27.1로 바뀌어 셀 합과 일치"}

# ---------------- V3: ± 값 ----------------
v3 = {}
for row in T1:
    k = sum(v1[hdr(row)])
    p = k / 96
    se = math.sqrt(p * (1 - p) / 96) * 100
    sd_trial = math.sqrt(p * (1 - p)) * 100
    p_rep = row[5] / 100
    se_rep = math.sqrt(p_rep * (1 - p_rep) / 96) * 100
    v3[hdr(row)] = {"reported_pm": row[6], "binomial_se_n96": round(se, 2), "binomial_se_from_reported_avg": round(se_rep, 2),
                    "per_trial_sd": round(sd_trial, 1), "match_se": abs(round(se_rep, 1) - row[6]) < 0.051}
R["V3_pm"] = v3

# ---------------- V4: 표 8 ----------------
v4 = {}
for name, (both, avg_both, deltas, single, avg_single) in T8.items():
    avg_calc = sum(both) / 4
    cnt = [count(v, 24)[0] for v in both]
    avg_cnt = 100 * sum(cnt) / 96
    d_cells = [round(b - s, 1) for b, s in zip(both, single)]
    d_avg_reported = round(avg_both - avg_single, 1)
    v4[name] = {
        "both_cells": both, "reported_avg": avg_both, "avg_of_cells": round(avg_calc, 2),
        "avg_from_counts": round(avg_cnt, 2), "counts": cnt, "avg_matches": abs(round(avg_cnt, 1) - avg_both) < 0.051 or abs(round(avg_calc + 1e-9, 1) - avg_both) < 0.051,
        "reported_deltas": deltas, "recomputed_cell_deltas": d_cells, "reported_avg_delta_vs_single": deltas[-1],
        "avg_delta_using_reported_avgs": d_avg_reported,
        "avg_delta_using_recomputed_avg": round(avg_cnt - avg_single, 1),
    }
R["V4_table8"] = v4

# ---------------- V5: 표 9와 표 1의 일관성 ----------------
v5 = {}
grid_bad = []
for g, m, ta, tq, cols, avg in T9:
    cnts = []
    for c, v, n in zip(T9_COLS, cols, T9_N):
        k, ok = count(v, n)
        cnts.append(k)
        if not ok:
            grid_bad.append((g, m, ta, tq, c, v, n))
    unstack = (cnts[0] + cnts[1]) / 24
    board = sum(cnts[3:6]) / 24
    plug = sum(cnts[6:9]) / 24
    pnp = cnts[2] / 24
    t1row = next(r for r in T1 if r[0] == g and r[1] == m and r[2] == ta and r[3] == tq)
    mine = [round(100 * x, 1) for x in (unstack, pnp, board, plug)]
    v5[f"{g}|{m}|T{ta}|Q{tq}"] = {
        "counts": cnts, "task_pct_from_table9": mine, "table1_task_pct": t1row[4],
        "task_match": all(abs(a - b) < 0.051 for a, b in zip(mine, t1row[4])),
        "avg_from_counts": round(100 * sum(cnts[:2] + cnts[2:3] + cnts[3:]) / 96, 2), "reported_avg": avg,
    }
R["V5_table9_vs_table1"] = {"rows": v5, "off_grid_cells": grid_bad,
                            "all_task_match": all(r["task_match"] for r in v5.values())}

# ---------------- V6: 본문 서술 수치 ----------------
def cell(g, m, ta, tq):
    return next(r for r in T1 if r[0] == g and r[1] == m and r[2] == ta and r[3] == tq)


b = cell("GR00T", "base", 0, 0)
f = cell("GR00T", "MoSS", 1, 1)
R["V6_text_claims"] = {
    "avg_gain_GR00T (28.2)": round(f[5] - b[5], 1),
    "unstack_gain (37.5)": round(f[4][0] - b[4][0], 1),
    "board_gain (29.2)": round(f[4][2] - b[4][2], 1),
    "table2a_unstack (20.9)": round(54.2 - 33.3, 1),
    "table2b_unstack (16.7)": round(54.2 - 37.5, 1),
    "table2c_unstack (8.4)": round(54.2 - 45.8, 1),
    "pnp_moss_tactile (66.7) vs forcevla (54.2)": [cell("GR00T", "MoSS", 1, 0)[4][1], cell("GR00T", "ForceVLA", 1, 0)[4][1]],
    "table2_default_equals_which_row": "Unstack 54.2 / PnP 66.7 = 표 1의 GR00T + MoSS (촉각+토크) 행 (촉각 단독 행은 45.8 / 66.7)",
    "fig1_values": {"Tactile-VLA 30.2→20.9": [30.2, 20.9], "ForceVLA 34.4→21.9": [34.4, 21.9], "TA-VLA 33.3→27.1": [33.3, 27.1],
                    "MoSS 42.7/37.5→49.0": [42.7, 37.5, 49.0]},
}

# ---------------- V7: 표 1 검정 (24회 / 96회) ----------------
def tot(row):
    return sum(v1[hdr(row)])


v7 = {}
pairs = [
    ("GR00T", ("GR00T", "MoSS", 1, 1), ("GR00T", "base", 0, 0)),
    ("GR00T", ("GR00T", "MoSS", 1, 0), ("GR00T", "base", 0, 0)),
    ("GR00T", ("GR00T", "MoSS", 0, 1), ("GR00T", "base", 0, 0)),
    ("GR00T", ("GR00T", "MoSS", 1, 0), ("GR00T", "ForceVLA", 1, 0)),
    ("GR00T", ("GR00T", "MoSS", 1, 0), ("GR00T", "Tactile-VLA", 1, 0)),
    ("GR00T", ("GR00T", "MoSS", 0, 1), ("GR00T", "TA-VLA", 0, 1)),
    ("GR00T", ("GR00T", "MoSS", 1, 1), ("GR00T", "ForceVLA", 1, 0)),
    ("GR00T", ("GR00T", "TA-VLA", 0, 1), ("GR00T", "base", 0, 0)),
    ("GR00T", ("GR00T", "ForceVLA", 1, 0), ("GR00T", "base", 0, 0)),
    ("pi0", ("pi0", "MoSS", 1, 1), ("pi0", "base", 0, 0)),
    ("pi0", ("pi0", "MoSS", 1, 0), ("pi0", "base", 0, 0)),
    ("pi0", ("pi0", "MoSS", 0, 1), ("pi0", "base", 0, 0)),
    ("pi0", ("pi0", "MoSS", 1, 0), ("pi0", "ForceVLA", 1, 0)),
    ("pi0", ("pi0", "MoSS", 0, 1), ("pi0", "TA-VLA", 0, 1)),
    ("pi0", ("pi0", "MoSS", 1, 0), ("pi0", "Tactile-VLA", 1, 0)),
    ("pi0", ("pi0", "ForceVLA", 1, 0), ("pi0", "base", 0, 0)),
]
for g, a, bb in pairs:
    ra, rb = cell(*a), cell(*bb)
    ka, kb = tot(ra), tot(rb)
    key = f"{hdr(ra)} vs {hdr(rb)}"
    v7[key] = {"k_a": ka, "k_b": kb, "n": 96, "diff_pp": round(100 * (ka - kb) / 96, 1),
               "fisher_p": fisher(ka, 96, kb, 96)}
    # 과제별 층화 (CMH, 과제당 24회)
    tabs = []
    for t in range(4):
        ca, cb = v1[hdr(ra)][t], v1[hdr(rb)][t]
        tabs.append(np.array([[ca, 24 - ca], [cb, 24 - cb]]) + 0.0)
    tabs = [t for t in tabs if (t[0, 0] + t[1, 0]) > 0 and (t[0, 1] + t[1, 1]) > 0]
    st = StratifiedTable(tabs)
    v7[key]["cmh_p"] = float(st.test_null_odds(correction=True).pvalue)
    v7[key]["cmh_common_or"] = round(float(st.oddsratio_pooled), 2)
R["V7_table1_tests"] = v7
holm_keys = [k for k in v7 if k.startswith("GR00T|MoSS|T1|Q1 vs GR00T|base") or k.startswith("pi0|MoSS|T1|Q1 vs pi0|base")]

# 과제별 MoSS(촉각+토크) 대 기본: Fisher (24회) + Holm(8)
v7b = {}
ps, keys = [], []
for g in ("GR00T", "pi0"):
    ra, rb = cell(g, "MoSS", 1, 1), cell(g, "base", 0, 0)
    for t, name in enumerate(TASKS):
        ca, cb = v1[hdr(ra)][t], v1[hdr(rb)][t]
        p = fisher(ca, 24, cb, 24)
        ps.append(p)
        keys.append(f"{g}|{name}")
        v7b[f"{g}|{name}"] = {"k_moss": ca, "k_base": cb, "diff_pp": round(100 * (ca - cb) / 24, 1), "fisher_p": p}
for k, a in zip(keys, holm(ps)):
    v7b[k]["holm_p_of_8"] = a
R["V7b_per_task_moss_vs_base"] = v7b

# ---------------- V8: 같은 모달리티 기준선과의 칸별 비교 (24칸) ----------------
cmp = {"win": 0, "tie": 0, "loss": 0, "cells": []}
for g in ("GR00T", "pi0"):
    for base_name, ta, tq in (("Tactile-VLA", 1, 0), ("ForceVLA", 1, 0), ("TA-VLA", 0, 1)):
        rb, rm = cell(g, base_name, ta, tq), cell(g, "MoSS", ta, tq)
        for t, name in enumerate(TASKS):
            d = round(rm[4][t] - rb[4][t], 1)
            key = "win" if d > 0 else ("loss" if d < 0 else "tie")
            cmp[key] += 1
            cmp["cells"].append([g, base_name, name, rm[4][t], rb[4][t], d])
by_model = {}
for g in ("GR00T", "pi0"):
    w = sum(1 for c in cmp["cells"] if c[0] == g and c[5] > 0)
    t_ = sum(1 for c in cmp["cells"] if c[0] == g and c[5] == 0)
    l = sum(1 for c in cmp["cells"] if c[0] == g and c[5] < 0)
    by_model[g] = {"win": w, "tie": t_, "loss": l}
cmp["by_model"] = by_model
R["V8_matched_modality_cells"] = cmp
# 각 기준선 대비 평균 차이
avgs = {}
for g in ("GR00T", "pi0"):
    for base_name, ta, tq in (("Tactile-VLA", 1, 0), ("ForceVLA", 1, 0), ("TA-VLA", 0, 1)):
        rb, rm = cell(g, base_name, ta, tq), cell(g, "MoSS", ta, tq)
        avgs[f"{g}|MoSS vs {base_name}"] = round(rm[5] - rb[5], 1)
R["V8_matched_modality_avg_diff"] = avgs

# ---------------- V9: "상보적 이득" (촉각+토크 대 단일) ----------------
v9 = {}
strata_best = []
for g in ("GR00T", "pi0"):
    both, tact, torq = cell(g, "MoSS", 1, 1), cell(g, "MoSS", 1, 0), cell(g, "MoSS", 0, 1)
    best = tact if tot(tact) >= tot(torq) else torq
    for nm, other in (("tactile", tact), ("torque", torq), ("best_single", best)):
        ka, kb = tot(both), tot(other)
        v9[f"{g}|both vs {nm}"] = {"k_both": ka, "k_other": kb, "diff_pp": round(100 * (ka - kb) / 96, 1),
                                   "fisher_p": fisher(ka, 96, kb, 96)}
    strata_best.append(np.array([[tot(both), 96 - tot(both)], [tot(best), 96 - tot(best)]]) + 0.0)
    cells_cmp = [(round(both[4][t] - max(tact[4][t], torq[4][t]), 1)) for t in range(4)]
    v9[f"{g}|cell diff vs best single (per task)"] = cells_cmp
st = StratifiedTable(strata_best)
v9["pooled_both_models_vs_best_single"] = {"cmh_p": float(st.test_null_odds(correction=True).pvalue),
                                          "common_or": round(float(st.oddsratio_pooled), 2),
                                          "k_both": [tot(cell(g, "MoSS", 1, 1)) for g in ("GR00T", "pi0")]}
# 기준선에서 촉각+토크 이동: 표 8
v9["table8_baselines_drop"] = {}
for name, (both, avg_both, deltas, single, avg_single) in T8.items():
    if name == "MoSS":
        continue
    kb = sum(count(v, 24)[0] for v in both)
    ks = sum(count(v, 24)[0] for v in single)
    v9["table8_baselines_drop"][name] = {"k_both": kb, "k_single_tab1": ks, "diff_pp": round(100 * (kb - ks) / 96, 1),
                                         "fisher_p": fisher(kb, 96, ks, 96)}
v9["table8_moss_both_vs_baseline_both"] = {}
km = sum(count(v, 24)[0] for v in T8["MoSS"][0])
for name, (both, *_rest) in T8.items():
    if name == "MoSS":
        continue
    kb = sum(count(v, 24)[0] for v in both)
    v9["table8_moss_both_vs_baseline_both"][name] = {"k_moss": km, "k_baseline": kb, "diff_pp": round(100 * (km - kb) / 96, 1),
                                                     "fisher_p": fisher(km, 96, kb, 96)}
R["V9_synergy"] = v9

# ---------------- V10: 표 2 절제 (24회) ----------------
T2 = {"stream": ("Single", [33.3, 50.0], "Decoupled", [54.2, 66.7]),
      "training": ("One-stage", [37.5, 58.3], "Two-stage", [54.2, 66.7]),
      "future": ("w/o prediction", [45.8, 58.3], "w/ prediction", [54.2, 66.7])}
v10 = {}
for nm, (la, va, lb, vb) in T2.items():
    d = {}
    ks_a = [count(v, 24)[0] for v in va]
    ks_b = [count(v, 24)[0] for v in vb]
    for i, task in enumerate(("Unstack", "PnP")):
        d[task] = {"k_a": ks_a[i], "k_b": ks_b[i], "diff_pp": round(100 * (ks_b[i] - ks_a[i]) / 24, 1),
                   "fisher_p": fisher(ks_b[i], 24, ks_a[i], 24)}
    d["pooled_two_tasks"] = {"k_a": sum(ks_a), "k_b": sum(ks_b), "n": 48, "fisher_p": fisher(sum(ks_b), 48, sum(ks_a), 48)}
    st = StratifiedTable([np.array([[ks_b[i], 24 - ks_b[i]], [ks_a[i], 24 - ks_a[i]]]) + 0.0 for i in range(2)])
    d["cmh_p"] = float(st.test_null_odds(correction=True).pvalue)
    v10[nm] = d
R["V10_table2_ablation"] = v10

# ---------------- V11: λ 민감도 (표 6) ----------------
T6 = {0.1: (54.2, 66.7), 0.5: (50.0, 58.3), 1.0: (29.2, 41.7)}
T6_V1 = {0.1: (13, 18), 0.5: (12, 16), 1.0: (7, 10)}  # v1 부록 B.1 표 5 (x/24)
v11 = {"v2_counts": {}, "v1_counts": T6_V1, "v1_pct": {}, "v2_vs_v1_pnp": {}}
for lam, (u, p) in T6.items():
    ku, kp = count(u, 24)[0], count(p, 24)[0]
    v11["v2_counts"][str(lam)] = [ku, kp]
    v11["v1_pct"][str(lam)] = [round(100 * T6_V1[lam][0] / 24, 1), round(100 * T6_V1[lam][1] / 24, 1)]
    v11["v2_vs_v1_pnp"][str(lam)] = {"v1_k": T6_V1[lam][1], "v2_k": kp}
v11["unstack_0.1_vs_1.0_fisher"] = fisher(13, 24, 7, 24)
v11["pnp_0.1_vs_1.0_fisher"] = fisher(16, 24, 10, 24)
v11["unstack_0.1_vs_0.5_fisher"] = fisher(13, 24, 12, 24)
v11["pnp_0.1_vs_0.5_fisher"] = fisher(16, 24, 14, 24)
v11["pooled_0.1_vs_1.0_fisher"] = fisher(29, 48, 17, 48)
v11["note"] = "v1의 PnP Egg 열: 0.1은 18/24(75.0), 0.5는 16/24(66.7). v2는 66.7(16/24)과 58.3(14/24)로 바뀜. 0.1 행의 66.7은 표 1의 같은 모델 값과 같아짐"
R["V11_table6_lambda"] = v11

# ---------------- V12: 지연 (표 3) ----------------
lat = {"base": 21.0, "tactile": 22.4, "torque": 21.9, "both": 23.4}
v12 = {k: {"ms": v, "ratio_exact": round(v / 21.0, 4), "ratio_2dp": round(v / 21.0 + 1e-9, 2)} for k, v in lat.items()}
v12["reported_ratio"] = {"tactile": 1.06, "torque": 1.04, "both": 1.11}
v12["both_minus_base_ms"] = round(23.4 - 21.0, 1)
R["V12_table3_latency"] = v12

# ---------------- V13: 표 4 (손 임바디먼트, 24회) ----------------
T4 = {"GR00T": [62.5, 54.2], "MoSS": [83.3, 95.8]}
v13 = {}
k4 = {m: [count(v, 24)[0] for v in vs] for m, vs in T4.items()}
v13["counts"] = k4
v13["avg_from_counts"] = {m: round(100 * sum(k) / 48, 2) for m, k in k4.items()}
v13["avg_of_rounded_cells"] = {m: round(sum(vs) / 2, 2) for m, vs in T4.items()}
v13["reported_avg"] = {"GR00T": 58.4, "MoSS": 89.6}
v13["fisher"] = {
    "Pick Paper": fisher(k4["MoSS"][0], 24, k4["GR00T"][0], 24),
    "Sort by Weight": fisher(k4["MoSS"][1], 24, k4["GR00T"][1], 24),
    "pooled_48": fisher(sum(k4["MoSS"]), 48, sum(k4["GR00T"]), 48),
}
v13["holm_two"] = holm([v13["fisher"]["Pick Paper"], v13["fisher"]["Sort by Weight"]])
v13["wilson_sort_moss"] = wilson(k4["MoSS"][1], 24)
v13["wilson_sort_base"] = wilson(k4["GR00T"][1], 24)
R["V13_table4_hand"] = v13

# ---------------- V14: RLBench 표 7 (시행 수 150 가정) ----------------
v14 = {"n_assumed": T7_N, "grid_check_all": True, "counts": {}, "fisher": {}, "recomputed_avgs": {}}
for m, vals in T7.items():
    ks = []
    for c, v in zip(T7_COLS, vals):
        k = round(v * T7_N / 100)
        ks.append(k)
        if abs(round(100 * k / T7_N, 1) - v) > 0.051:
            v14["grid_check_all"] = False
    v14["counts"][m] = ks
    contact = sum(ks[:4]) / (4 * T7_N) * 100
    non = sum(ks[4:]) / (5 * T7_N) * 100
    allv = sum(ks) / (9 * T7_N) * 100
    v14["recomputed_avgs"][m] = {"contact": round(contact, 2), "non_contact": round(non, 2), "overall": round(allv, 2),
                                 "reported": T7_REPORTED[m]}
ps, labs = [], []
for i, c in enumerate(T7_COLS):
    p = fisher(v14["counts"]["MoSS"][i], T7_N, v14["counts"]["GR00T"][i], T7_N)
    ps.append(p)
    labs.append(c)
for c, p, a in zip(labs, ps, holm(ps)):
    v14["fisher"][c] = {"moss_k": v14["counts"]["MoSS"][labs.index(c)], "base_k": v14["counts"]["GR00T"][labs.index(c)],
                        "p": p, "holm_p_of_9": a}
kc = (sum(v14["counts"]["MoSS"][:4]), sum(v14["counts"]["GR00T"][:4]))
kn = (sum(v14["counts"]["MoSS"][4:]), sum(v14["counts"]["GR00T"][4:]))
v14["pooled_contact_fisher"] = {"moss": kc[0], "base": kc[1], "n_each": 4 * T7_N, "p": fisher(kc[0], 4 * T7_N, kc[1], 4 * T7_N)}
v14["pooled_noncontact_fisher"] = {"moss": kn[0], "base": kn[1], "n_each": 5 * T7_N, "p": fisher(kn[0], 5 * T7_N, kn[1], 5 * T7_N)}
v14["n_sensitivity_pooled_contact_p"] = {str(n): fisher(round(38.15 * 4 * n / 100), 4 * n, round(27.325 * 4 * n / 100), 4 * n)
                                         for n in (50, 100, 150, 300)}
R["V14_table7_rlbench"] = v14

# ---------------- V15: 정확 이항 구간과 검정력 ----------------
v15 = {"wilson_table1_avg_n96": {}, "min_detectable_diff_pp": {}}
for row in T1:
    if row[1] == "MoSS" or row[1] == "base":
        v15["wilson_table1_avg_n96"][hdr(row)] = [round(x, 1) for x in wilson(tot(row), 96)]
zb = stats.norm.ppf(0.8)


def mde(p0, n):
    """두 독립 비율을 양측 α=0.05, 검정력 80%로 구분하는 최소 차이(arcsine 근사)."""
    h = (Z + zb) * math.sqrt(2 / n)
    p1 = math.sin(math.asin(math.sqrt(p0)) + h / 2) ** 2
    return 100 * (p1 - p0)


for n in (24, 96, 192):
    v15["min_detectable_diff_pp"][f"n{n}_p0=0.3"] = round(mde(0.3, n), 1)
    v15["min_detectable_diff_pp"][f"n{n}_p0=0.5"] = round(mde(0.5, n), 1)
# 표 1의 MoSS(촉각) 대 ForceVLA(GR00T) 차이(8.3 pp)를 80% 검정력으로 보려면 필요한 칸당 시행 수
def need_n(p0, p1):
    h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p0))
    return math.ceil(2 * ((Z + zb) / h) ** 2)


v15["needed_n_per_arm_80pct"] = {"0.344_vs_0.427 (GR00T MoSS-tactile vs ForceVLA)": need_n(0.344, 0.427),
                                 "0.427_vs_0.490 (MoSS tactile vs tactile+torque)": need_n(0.427, 0.490),
                                 "0.208_vs_0.490": need_n(0.208, 0.490)}
R["V15_wilson_power"] = v15

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
print("wrote", OUT)

# 요약 출력
print("V1 off-grid:", R["V1_table1_counts_n24"]["off_grid_cells"])
print("V2 mismatch exact:", [k for k, v in v2.items() if not v["match_exact"]])
print("V2 mismatch rounded-cell mean:", [k for k, v in v2.items() if not v["match_rounded"]])
print("V3 pm mismatch:", [k for k, v in v3.items() if not v["match_se"]])
for k, v in v4.items():
    print("V4", k, v["reported_avg"], v["avg_from_counts"], v["avg_matches"], v["reported_deltas"], v["recomputed_cell_deltas"])
print("V5 all task match:", R["V5_table9_vs_table1"]["all_task_match"], R["V5_table9_vs_table1"]["off_grid_cells"])
