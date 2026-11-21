"""UWM (arXiv:2504.02792 v3) 표 수치 재계산과 검정.

실행: python3 analysis/recompute.py  → analysis/recompute_results.json (수 초, CPU)

가정 (03_analysis.md §0에 자세히 씀)
- 실로봇 표 I: 과제당 평가 조건 n = 50 (Hang-Towel·Rice-Cooker 중 Rice-Cooker는 20). 성공률은 k/n이고 k = round(rate × n).
  논문은 초기 위치마다 3번 시도한다고 쓰므로, 시도를 독립으로 셀 때의 민감도(n × 3)도 함께 본다.
- LIBERO 표 II, VII: "±"는 3개 시드의 표준편차로 가정한다 (논문에 명시 없음). 표준오차라는 대안도 함께 본다.
- 두 비율 비교는 독립 표본으로 본다. 같은 초기 위치를 쓴 쌍체 설계일 수 있으나 원자료가 없어 쌍체 검정은 못 한다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(c - h, 4), round(c + h, 4))


def fisher(k1, n1, k2, n2):
    t = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(stats.fisher_exact(t, alternative="two-sided")[1])


def welch_from_stats(m1, s1, n1, m2, s2, n2):
    t, p = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
    return float(t), float(p)


# ---------------- 표 I: 실로봇 성공률 (n=50, Rice-Cooker는 n=20) ----------------
# 값: (pretrain 또는 단독, cotrain 또는 None). 열 순서는 논문 표 I과 같다.
# 열: SB-ID, SB-OOD, BC-ID, BC-OOD, PT-ID, PT-OOD, HT-ID, HT-OOD, RC-ID
T1 = {
    "UWM": {"SB-ID": ("0.86", "0.92"), "SB-OOD": ("0.76", "0.84"), "BC-ID": ("0.76", "0.84"), "BC-OOD": ("0.60", "0.72"),
            "PT-ID": ("0.78", "0.86"), "PT-OOD": ("0.78", "0.84"), "HT-ID": ("0.82", "0.86"), "HT-OOD": ("0.64", "0.76"),
            "RC-ID": ("0.60", "0.65")},
    "DP": {"SB-ID": ("0.48", None), "SB-OOD": ("0.36", None), "BC-ID": ("0.60", None), "BC-OOD": ("0.26", None),
           "PT-ID": ("0.52", None), "PT-OOD": ("0.48", None), "HT-ID": ("0.64", None), "HT-OOD": ("0.28", None),
           "RC-ID": ("0.35", None)},
    "PAD": {"SB-ID": ("0.08", "0.20"), "SB-OOD": ("0.08", "0.12"), "BC-ID": ("0.00", "0.00"), "BC-OOD": ("0.00", "0.00"),
            "PT-ID": ("0.42", "0.42"), "PT-OOD": ("0.34", "0.44"), "HT-ID": ("0.52", "0.54"), "HT-OOD": ("0.30", "0.38"),
            "RC-ID": ("0.00", "0.00")},
    "GR1": {"SB-ID": ("0.66", "0.62"), "SB-OOD": ("0.48", "0.38"), "BC-ID": ("0.66", "0.74"), "BC-OOD": ("0.44", "0.64"),
            "PT-ID": ("0.60", "0.46"), "PT-OOD": ("0.60", "0.46"), "HT-ID": ("0.66", "0.66"), "HT-OOD": ("0.48", "0.44"),
            "RC-ID": ("0.40", "0.25")},
}
N_CELL = {c: (20 if c == "RC-ID" else 50) for c in T1["UWM"]}


def cell_k(method, cell, which):
    v = T1[method][cell][which]
    return None if v is None else int(round(float(v) * N_CELL[cell]))


t1 = {"cells": {}, "sanity": {}}
for cell in N_CELL:
    row = {}
    for m in T1:
        k_pre = cell_k(m, cell, 0)
        k_co = cell_k(m, cell, 1)
        row[m] = {"pre": {"k": k_pre, "n": N_CELL[cell], "rate": round(k_pre / N_CELL[cell], 3),
                          "wilson95": wilson(k_pre, N_CELL[cell])}}
        if k_co is not None:
            row[m]["co"] = {"k": k_co, "n": N_CELL[cell], "rate": round(k_co / N_CELL[cell], 3),
                            "wilson95": wilson(k_co, N_CELL[cell])}
    # UWM 대비 최고 기준선 (pretrain)
    best = max(["DP", "PAD", "GR1"], key=lambda m: cell_k(m, cell, 0))
    kb = cell_k(best, cell, 0)
    ku = cell_k("UWM", cell, 0)
    n = N_CELL[cell]
    row["best_baseline_pre"] = best
    row["diff_pp_UWM_minus_best"] = round(100 * (ku - kb) / n, 1)
    row["fisher_p_UWM_pre_vs_best"] = round(fisher(ku, n, kb, n), 4)
    k_uc = cell_k("UWM", cell, 1)
    if k_uc is not None:
        row["fisher_p_UWM_co_vs_pre"] = round(fisher(k_uc, n, ku, n), 4)
        row["diff_pp_co_minus_pre"] = round(100 * (k_uc - ku) / n, 1)
    t1["cells"][cell] = row

# 과제 평균 (Fig 6 대응): 과제별 비율의 단순 평균. 이 값은 논문 그림에서 읽을 수 없어 재계산만 한다.
ID_CELLS = ["SB-ID", "BC-ID", "PT-ID", "HT-ID", "RC-ID"]
OOD_CELLS = ["SB-OOD", "BC-OOD", "PT-OOD", "HT-OOD"]
avg = {}
for m in T1:
    for which, idx in [("pre", 0), ("co", 1)]:
        ids = [cell_k(m, c, idx) / N_CELL[c] for c in ID_CELLS if cell_k(m, c, idx) is not None]
        oods = [cell_k(m, c, idx) / N_CELL[c] for c in OOD_CELLS if cell_k(m, c, idx) is not None]
        if ids:
            avg[f"{m}-{which}-ID-macro(5 tasks)"] = round(float(np.mean(ids)), 3)
        if oods:
            avg[f"{m}-{which}-OOD-macro(4 tasks)"] = round(float(np.mean(oods)), 3)
t1["macro_average"] = avg

# 민감도: 시도를 독립으로 셀 때 n x 3 (UWM SB-ID 대 GR1 SB-ID)
t1["sensitivity_3x_attempts"] = {
    "SB-ID UWM vs GR1 p": round(fisher(129, 150, 99, 150), 6),
    "SB-ID UWM vs GR1 (n=50) p": round(fisher(43, 50, 33, 50), 4),
}

# 검정력: UWM 대 최고 기준선 차이를 80% 검정력으로 잡으려면 칸당 몇 시도가 필요한가 (양측 0.05)
pw = NormalIndPower()
power = {}
for label, p1, p2 in [("SB-ID UWM 0.86 vs GR1 0.66", 0.86, 0.66), ("HT-ID UWM 0.82 vs GR1 0.66", 0.82, 0.66),
                      ("SB-ID co 0.92 vs pre 0.86", 0.92, 0.86), ("BC-OOD co 0.72 vs pre 0.60", 0.72, 0.60)]:
    es = proportion_effectsize(p1, p2)
    power[label] = int(math.ceil(pw.solve_power(effect_size=abs(es), alpha=0.05, power=0.8, ratio=1)))
t1["n_per_arm_for_80pct_power"] = power
res["table1_real_robot"] = t1

# ---------------- 표 II: LIBERO (5 과제, 시드 3) ----------------
T2 = {
    "UWM": [(0.91, 0.07), (0.93, 0.01), (0.80, 0.02), (0.68, 0.02), (0.65, 0.01)],
    "DP": [(0.73, 0.10), (0.88, 0.02), (0.77, 0.02), (0.65, 0.03), (0.53, 0.05)],
    "PAD": [(0.78, 0.04), (0.47, 0.04), (0.74, 0.05), (0.59, 0.08), (0.25, 0.04)],
    "GR1": [(0.77, 0.03), (0.65, 0.05), (0.62, 0.03), (0.46, 0.04), (0.38, 0.05)],
}
REPORTED_AVG = {"UWM": (0.79, 0.11), "DP": (0.71, 0.12), "PAD": (0.57, 0.19), "GR1": (0.58, 0.14)}
t2 = {"avg_check": {}, "vs_DP": {}, "vs_DP_SE_alt": {}}
for m, vals in T2.items():
    mu = float(np.mean([v[0] for v in vals]))
    t2["avg_check"][m] = {"recomputed_mean_of_5": round(mu, 3), "reported": REPORTED_AVG[m][0]}
tasks = ["Book-Caddy", "Soup-Cheese", "Bowl-Drawer", "Moka-Moka", "Mug-Mug"]
for i, task in enumerate(tasks):
    mu_u, sd_u = T2["UWM"][i]
    mu_d, sd_d = T2["DP"][i]
    t, p = welch_from_stats(mu_u, sd_u, 3, mu_d, sd_d, 3)
    t2["vs_DP"][task] = {"diff": round(mu_u - mu_d, 3), "welch_t": round(t, 2), "p_SD": round(p, 3)}
    # 대안: ±가 표준오차이면 SD = SE * sqrt(3)
    t_, p_ = welch_from_stats(mu_u, sd_u * math.sqrt(3), 3, mu_d, sd_d * math.sqrt(3), 3)
    t2["vs_DP_SE_alt"][task] = {"p_SE": round(p_, 3)}
# 과제 평균 차이 (UWM 대 DP): 과제 간 평균을 한 표본으로 본 대략 검정
diffs = [T2["UWM"][i][0] - T2["DP"][i][0] for i in range(5)]
t2["paired_task_level_mean_diff"] = {"mean_diff": round(float(np.mean(diffs)), 3),
                                     "t_over_tasks_p": round(float(stats.ttest_1samp(diffs, 0).pvalue), 3)}
res["table2_libero"] = t2

# ---------------- 표 III: 궤적 추적 (50 궤적, 시드 3 가정 시 150) ----------------
t3 = {}
for task, pol, inv in [("Book-Caddy", 0.47, 0.65), ("Soup-Cheese", 0.26, 0.55)]:
    k_p, k_i = int(round(pol * 150)), int(round(inv * 150))
    t3[task] = {"policy_traj_len_rate": pol, "inv_dyn_rate": inv,
                "p_fisher_if_n150": float(f"{fisher(k_i, 150, k_p, 150):.2e}"),
                "p_fisher_if_n50": round(fisher(int(round(inv * 50)), 50, int(round(pol * 50)), 50), 4)}
res["table3_tracking"] = t3

# ---------------- 표 IV: 범주별 OOD (칸당 5 초기 위치; 열당 6 범주) ----------------
# 열 순서: SB-co, SB-pre, SB-DP, BC-co, BC-pre, BC-DP (행 L1..C2)
rows = {
    "L1": [4, 4, 2, 5, 5, 3], "L2": [3, 2, 2, 4, 0, 0], "B1": [4, 3, 3, 4, 3, 2],
    "B2": [3, 1, 2, 1, 0, 0], "C1": [3, 2, 2, 0, 0, 0], "C2": [4, 3, 1, 1, 0, 0],
}
reported_all = [21, 15, 12, 15, 8, 6]
sums = [sum(r[j] for r in rows.values()) for j in range(6)]
t4 = {"column_sums_from_rows": sums, "reported_all_row": reported_all,
      "mismatch_columns": [j for j in range(6) if sums[j] != reported_all[j]]}
names = ["SB-co", "SB-pre", "SB-DP", "BC-co", "BC-pre", "BC-DP"]
t4["fisher"] = {
    "SB co 21/30 vs pre 15/30": round(fisher(21, 30, 15, 30), 4),
    "SB co 21/30 vs DP 12/30": round(fisher(21, 30, 12, 30), 4),
    "BC co 15/30 vs pre 8/30": round(fisher(15, 30, 8, 30), 4),
    "BC co 15/30 vs DP 6/30 (보고값)": round(fisher(15, 30, 6, 30), 4),
    "BC co 15/30 vs DP 5/30 (행 합)": round(fisher(15, 30, 5, 30), 4),
}
res["table4_categorized_ood"] = t4

# ---------------- 표 VII: 설계 선택 (LIBERO 단일 과제, 시드 3) ----------------
T7 = {
    "UWM w/ 8 registers": [(0.88, 0.04), (0.90, 0.02)],
    "UWM w/ 4 registers": [(0.83, 0.05), (0.86, 0.03)],
    "UWM w/o registers": [(0.81, 0.07), (0.85, 0.03)],
    "Cross attention UWM": [(0.78, 0.05), (0.86, 0.04)],
}
t7 = {}
base = T7["UWM w/ 8 registers"]
for name, vals in T7.items():
    t7[name] = {}
    for i, task in enumerate(["Book-Caddy", "Soup-Cheese"]):
        m, s = vals[i]
        mb, sb = base[i]
        if name == "UWM w/ 8 registers":
            continue
        _, p = welch_from_stats(mb, sb, 3, m, s, 3)
        t7[name][task] = {"diff_vs_8reg": round(m - mb, 3), "p_SD": round(p, 3)}
res["table7_design"] = t7

# ---------------- 표 VIII, IX (칸당 n = 50 가정) ----------------
t8 = {
    "SB future 43/50 vs current 35/50": round(fisher(43, 50, 35, 50), 4),
    "SB future 43/50 vs DP 24/50": round(fisher(43, 50, 24, 50), 6),
    "BC future 38/50 vs current 33/50": round(fisher(38, 50, 33, 50), 4),
    "BC future 38/50 vs DP 30/50": round(fisher(38, 50, 30, 50), 4),
}
res["table8_objective"] = t8
t9 = {
    "SB robot+robotvideo 46/50 vs robot-only 43/50": round(fisher(46, 50, 43, 50), 4),
    "SB robot+internet 44/50 vs robot-only 43/50": round(fisher(44, 50, 43, 50), 4),
    "BC robot+robotvideo 42/50 vs robot-only 38/50": round(fisher(42, 50, 38, 50), 4),
    "BC robot+internet 40/50 vs robot-only 38/50": round(fisher(40, 50, 38, 50), 4),
}
res["table9_internet_video"] = t9

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1)[:6000])
print("\nwritten:", OUT)
