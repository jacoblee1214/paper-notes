"""FlowDAgger (arXiv:2607.08877 v1) 보고 수치 재계산, 정수 가능성 점검, 검정, 역변환 toy.

입력은 논문 표(표 1-6)의 값을 그대로 옮긴 것이다. 시행 수가 표에 없는 곳은 가정을 밝히고 두 경우를 모두 계산한다.
  - 표 1-3: "3 시드 x 25 롤아웃"이라고 하지만 값이 k/25 격자에 놓이는 칸이 많아 n = 25와 n = 75를 모두 시험한다.
  - 표 4: 30 롤아웃 (표 캡션).
실행: python3 analysis/recompute.py  (수 초, CPU, numpy와 scipy만 사용)
결과: analysis/recompute_results.json
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def feasible(v, n):
    """보고값 v(소수 둘째 자리)가 k/n의 반올림 또는 내림(버림)으로 나올 수 있는가."""
    rnd = trunc = False
    for k in range(n + 1):
        x = 100 * k / n
        if abs(math.floor(x + 0.5) - round(100 * v)) < 1e-9:
            rnd = True
        if abs(math.floor(x + 1e-9) - round(100 * v)) < 1e-9:
            trunc = True
    return rnd, trunc


# ---------------------------------------------------------------- 표 1
T1_TASKS = ["Assembly", "Bin Picking", "Box Close", "Coffee Pull", "Dial Turn", "Door Lock",
            "Hammer", "Hand Insert", "Lever Pull", "Pick Place", "Soccer", "Stick Push"]
T1 = {
    "Base": [.64, .56, .36, .84, .20, .44, .40, .84, .28, .76, .24, .84],
    "SFT": [.85, .76, .70, .96, .64, .48, .80, .88, .44, .80, .36, .84],
    "LoRA-DAg": [.81, .68, .52, .68, .88, .76, .68, .72, .52, .68, .32, .92],
    "Res-DAg": [.53, .63, .69, .95, .59, .85, .56, .71, .37, .71, .28, .84],
    "DSRL": [.64, .76, .48, .84, .43, .28, .27, .92, .35, .60, .33, .73],
    "FlowDAgger": [.89, .69, .59, 1.00, .75, .75, .84, .99, .61, .85, .44, 1.00],
}
T1_MEAN_REP = {"Base": .53, "SFT": .71, "LoRA-DAg": .68, "Res-DAg": .64, "DSRL": .55, "FlowDAgger": .78}
T1_DELTA_REP = {"SFT": .18, "LoRA-DAg": .15, "Res-DAg": .11, "DSRL": .02, "FlowDAgger": .25}

r = {}
r["mean"] = {m: float(np.mean(v)) for m, v in T1.items()}
r["mean_rep"] = T1_MEAN_REP
r["mean_match"] = {m: abs(r["mean"][m] - T1_MEAN_REP[m]) < 0.005 for m in T1}
r["delta"] = {m: float(np.mean(T1[m]) - np.mean(T1["Base"])) for m in T1 if m != "Base"}
r["delta_rep"] = T1_DELTA_REP
r["delta_match"] = {m: abs(r["delta"][m] - T1_DELTA_REP[m]) < 0.005 for m in r["delta"]}
# 행별 최고 (동률 포함) 와 승수
methods = [m for m in T1 if m != "Base"]
wins = {m: 0 for m in methods}
sole = {m: 0 for m in methods}
for i in range(12):
    vals = {m: T1[m][i] for m in methods}
    top = max(vals.values())
    ws = [m for m in methods if abs(vals[m] - top) < 1e-9]
    for m in ws:
        wins[m] += 1
    if len(ws) == 1:
        sole[ws[0]] += 1
r["row_best_incl_ties"] = wins
r["row_best_sole"] = sole
r["fd_loses_rows"] = [T1_TASKS[i] for i in range(12)
                      if max(T1[m][i] for m in methods) - T1["FlowDAgger"][i] > 1e-9]
# 짝 비교 (FlowDAgger 대 각 방법), 12개 과제를 표본 단위로
rng = np.random.default_rng(0)
pair = {}
for m in methods:
    if m == "FlowDAgger":
        continue
    d = np.array(T1["FlowDAgger"]) - np.array(T1[m])
    boot = [rng.choice(d, 12).mean() for _ in range(20000)]
    pair[m] = {
        "mean_diff": float(d.mean()),
        "fd_better_tasks": int((d > 1e-9).sum()), "ties": int((abs(d) < 1e-9).sum()),
        "paired_t_p": float(stats.ttest_rel(T1["FlowDAgger"], T1[m])[1]),
        "wilcoxon_p": float(stats.wilcoxon(d, zero_method="wilcox", method="auto")[1]) if (d != 0).any() else None,
        "sign_p": float(stats.binomtest(int((d > 1e-9).sum()), int((abs(d) > 1e-9).sum()), 0.5).pvalue),
        "boot95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    }
r["paired_across_tasks"] = pair
# 개별 과제 Fisher (FlowDAgger 대 SFT 등), n = 25와 75
fish = {}
for m in ["SFT", "LoRA-DAg", "Res-DAg", "DSRL", "Base"]:
    fish[m] = {}
    for n in (25, 75):
        ps = []
        for i in range(12):
            k1 = round(T1["FlowDAgger"][i] * n)
            k2 = round(T1[m][i] * n)
            ps.append(fisher(k1, n, k2, n))
        fish[m][f"n{n}"] = {"p_per_task": [round(p, 4) for p in ps],
                            "n_sig_p05": int(sum(p < 0.05 for p in ps))}
r["fisher_fd_vs"] = fish
# 혼합 가정: 격자 증거가 가리키는 시행 수 (기준·SFT·LoRA는 25, 나머지는 75)
mix_n = {"Base": 25, "SFT": 25, "LoRA-DAg": 25, "Res-DAg": 75, "DSRL": 75}
r["fisher_mixed_assumption"] = {}
for m, n_o in mix_n.items():
    ps = [fisher(round(T1["FlowDAgger"][i] * 75), 75, round(T1[m][i] * n_o), n_o) for i in range(12)]
    r["fisher_mixed_assumption"][m] = {"n_other": n_o, "p_per_task": [round(p, 4) for p in ps],
                                        "n_sig_p05": int(sum(p < 0.05 for p in ps))}
# 격자 점검
grid = {}
for m, vals in T1.items():
    g25 = [feasible(v, 25) for v in vals]
    g75 = [feasible(v, 75) for v in vals]
    grid[m] = {
        "k25_round": int(sum(a for a, _ in g25)), "k75_round": int(sum(a for a, _ in g75)),
        "only25_cells": [T1_TASKS[i] for i in range(12) if g25[i][0] and not g75[i][0]],
        "only75_cells": [T1_TASKS[i] for i in range(12) if g75[i][0] and not g25[i][0]],
        "neither_cells": [T1_TASKS[i] for i in range(12) if not g25[i][0] and not g75[i][0]],
    }
r["grid"] = grid
# 4의 배수(k/25 격자) 칸 수가 우연(3 시드 평균이면 k/75 중 k가 3의 배수일 확률 1/3)보다 많은가
r["multiple_of_004_cells"] = {m: int(sum(abs(v * 25 - round(v * 25)) < 1e-9 for v in vals)) for m, vals in T1.items()}
r["multiple_of_004_binom_p_ge"] = {m: float(stats.binom.sf(c - 1, 12, 1 / 3)) for m, c in r["multiple_of_004_cells"].items()}
res["table1"] = r

# ---------------------------------------------------------------- 표 2
T2_TASKS = ["Assembly", "Bin Picking", "Box Close", "Dial Turn", "Hand Insert", "Lever Pull", "Stick Push"]
T2 = {
    "pi05_base": [.64, .56, .36, .20, .84, .28, .84], "pi05_fd": [.89, .69, .59, .75, .99, .61, 1.00],
    "cos_base": [.52, .28, .36, .44, .76, .56, .76], "cos_fd": [.92, .44, .64, .68, .88, .64, .96],
}
T2_DELTA_REP = {"pi05": [.25, .13, .23, .55, .15, .33, .16], "cos": [.40, .16, .28, .24, .12, .08, .20]}
T2_MEAN_REP = {"pi05_base": .53, "pi05_fd": .79, "pi05_d": .26, "cos_base": .53, "cos_fd": .74, "cos_d": .21}
r = {}
r["mean"] = {k: float(np.mean(v)) for k, v in T2.items()}
dp = np.array(T2["pi05_fd"]) - np.array(T2["pi05_base"])
dc = np.array(T2["cos_fd"]) - np.array(T2["cos_base"])
r["delta_pi05"] = [round(float(x), 2) for x in dp]
r["delta_cos"] = [round(float(x), 2) for x in dc]
r["delta_pi05_match"] = bool(np.allclose(np.round(dp, 2), T2_DELTA_REP["pi05"], atol=0.0051))
r["delta_cos_match"] = bool(np.allclose(np.round(dc, 2), T2_DELTA_REP["cos"], atol=0.0051))
r["mean_delta"] = {"pi05": float(dp.mean()), "cos": float(dc.mean())}
r["mean_match"] = {"pi05_base": abs(np.mean(T2["pi05_base"]) - .53) < .005, "pi05_fd": abs(np.mean(T2["pi05_fd"]) - .79) < .005,
                   "cos_base": abs(np.mean(T2["cos_base"]) - .53) < .005, "cos_fd": abs(np.mean(T2["cos_fd"]) - .74) < .005}
# 표 1과 표 2의 pi0.5 열 일치
idx = [T1_TASKS.index(t) for t in T2_TASKS]
r["pi05_matches_table1"] = bool(all(abs(T2["pi05_base"][j] - T1["Base"][i]) < 1e-9 and
                                   abs(T2["pi05_fd"][j] - T1["FlowDAgger"][i]) < 1e-9 for j, i in enumerate(idx)))
r["pi05_fd_mean_table1_12"] = float(np.mean(T1["FlowDAgger"]))
# 두 계열의 개선폭 차이 (과제 짝)
r["delta_gap_paired_t_p"] = float(stats.ttest_rel(dp, dc)[1])
r["delta_gap_wilcoxon_p"] = float(stats.wilcoxon(dp - dc)[1])
r["cos_gain_positive_tasks"] = int((dc > 0).sum())
r["cos_each_task_fisher_n25"] = [round(fisher(round(T2["cos_fd"][j] * 25), 25, round(T2["cos_base"][j] * 25), 25), 4) for j in range(7)]
kb = sum(round(v * 25) for v in T2["cos_base"]); kf = sum(round(v * 25) for v in T2["cos_fd"])
r["cos_pooled_k"] = {"base": kb, "fd": kf, "n": 175, "p": fisher(kf, 175, kb, 175)}
r["cos_grid_k25_all"] = bool(all(feasible(v, 25)[0] for k in ("cos_base", "cos_fd") for v in T2[k]))
r["cos_values_multiple_of_004"] = bool(all(abs(v * 25 - round(v * 25)) < 1e-9 for k in ("cos_base", "cos_fd") for v in T2[k]))
res["table2"] = r

# ---------------------------------------------------------------- 표 3
T3_COLS = ["Hammer", "Door", "Drawer", "Faucet", "Plate", "Push"]
T3 = {
    "Base": [.40, 1.00, 1.00, .96, .88, .96],
    "FlowDAgger": [.84, .88, 1.00, .96, 1.00, .56],
    "Res-DAg": [.56, .88, .68, .80, 1.00, .08],
    "LoRA-DAg": [.68, .80, .00, .36, .00, .36],
    "SFT": [.80, .12, .00, .00, .00, .00],
}
T3_MEAN_REP = {"Base": .96, "FlowDAgger": .88, "Res-DAg": .69, "LoRA-DAg": .30, "SFT": .02}
T3_DELTA_REP = {"FlowDAgger": -.08, "Res-DAg": -.27, "LoRA-DAg": -.66, "SFT": -.94}
r = {}
r["heldout_mean"] = {m: float(np.mean(v[1:])) for m, v in T3.items()}
r["heldout_mean_match"] = {m: abs(r["heldout_mean"][m] - T3_MEAN_REP[m]) < .0051 for m in T3}
r["delta"] = {m: float(np.mean(T3[m][1:]) - np.mean(T3["Base"][1:])) for m in T3 if m != "Base"}
r["delta_match"] = {m: abs(r["delta"][m] - T3_DELTA_REP[m]) < .0051 for m in r["delta"]}
r["all_k25"] = bool(all(abs(v * 25 - round(v * 25)) < 1e-9 for vals in T3.values() for v in vals))
r["hammer_matches_table1"] = {m: abs(T3[m][0] - T1[{"Base": "Base", "FlowDAgger": "FlowDAgger", "Res-DAg": "Res-DAg",
                                                     "LoRA-DAg": "LoRA-DAg", "SFT": "SFT"}[m]][6]) < 1e-9 for m in T3}
# 보류 과제별 변화와 Fisher (n = 25)
chg = {}
for m in ["FlowDAgger", "Res-DAg", "LoRA-DAg", "SFT"]:
    chg[m] = {}
    for j in range(1, 6):
        k1, k0 = round(T3[m][j] * 25), round(T3["Base"][j] * 25)
        chg[m][T3_COLS[j]] = {"delta": round(T3[m][j] - T3["Base"][j], 2), "p": round(fisher(k1, 25, k0, 25), 4)}
r["heldout_change_fisher_n25"] = chg
r["fd_heldout_sig_drops"] = [c for c, d in chg["FlowDAgger"].items() if d["delta"] < 0 and d["p"] < 0.05]
r["fd_push_drop"] = T3["FlowDAgger"][5] - T3["Base"][5]
# 훈련 내 Hammer: FlowDAgger 대 SFT
r["hammer_fd_vs_sft_p"] = fisher(21, 25, 20, 25)
r["hammer_gain"] = {m: round(T3[m][0] - T3["Base"][0], 2) for m in T3 if m != "Base"}
# 보류 평균의 과제 단위 짝 비교 (FlowDAgger 대 Res-DAg)
a = np.array(T3["FlowDAgger"][1:]) - np.array(T3["Res-DAg"][1:])
r["fd_vs_res_heldout_paired_t_p"] = float(stats.ttest_rel(T3["FlowDAgger"][1:], T3["Res-DAg"][1:])[1])
res["table3"] = r

# ---------------------------------------------------------------- 표 4
T4_TASKS = ["Block Pick", "Glassware Stacking", "Button Push", "Slider", "Wire Pull", "Jenga Stacking",
            "Toolbox Packing", "Plug Insertion"]
T4 = {"Base": [.73, .26, .60, .36, .40, .76, .13, .60], "SFT": [.80, .53, .73, .43, .53, .90, .63, .66],
      "FD": [.90, .76, .73, .66, .70, .86, .80, .72]}
T4_DELTA_REP = [.17, .50, .13, .30, .30, .10, .67, .12]
EPS = [5, 5, 10, 5, 10, 5, 10, 20]
r = {}
r["grid_n30"] = {m: [list(feasible(v, 30)) for v in vals] for m, vals in T4.items()}
r["infeasible_n30"] = [(m, T4_TASKS[i], T4[m][i]) for m, vals in T4.items() for i, v in enumerate(vals)
                       if not any(feasible(v, 30))]
r["truncated_only_cells"] = [(m, T4_TASKS[i], T4[m][i]) for m, vals in T4.items() for i, v in enumerate(vals)
                            if feasible(v, 30)[1] and not feasible(v, 30)[0]]
r["round_only_cells"] = [(m, T4_TASKS[i], T4[m][i]) for m, vals in T4.items() for i, v in enumerate(vals)
                         if feasible(v, 30)[0] and not feasible(v, 30)[1]]
r["plug_fd_feasible_n25"] = feasible(.72, 25)
# 개선폭 재계산
d_fd = [round(T4["FD"][i] - T4["Base"][i], 2) for i in range(8)]
r["delta_fd_base"] = d_fd
r["delta_match"] = d_fd == T4_DELTA_REP
r["delta_fd_sft"] = [round(T4["FD"][i] - T4["SFT"][i], 2) for i in range(8)]
r["fd_beats_sft_tasks"] = int(sum(T4["FD"][i] > T4["SFT"][i] for i in range(8)))
r["fd_ties_sft"] = [T4_TASKS[i] for i in range(8) if abs(T4["FD"][i] - T4["SFT"][i]) < 1e-9]
r["fd_loses_sft"] = [T4_TASKS[i] for i in range(8) if T4["FD"][i] < T4["SFT"][i]]
r["means"] = {m: float(np.mean(v)) for m, v in T4.items()}
# 개별 Fisher (k = round(p*30)); 플러그 FD는 21과 22 두 경우
def kk(p):
    return round(p * 30)
pf = {}
for scen, plug_k in (("plug_fd_21", 21), ("plug_fd_22", 22)):
    rows = []
    for i in range(8):
        k_fd = plug_k if i == 7 else kk(T4["FD"][i])
        k_b = kk(T4["Base"][i])
        k_s = kk(T4["SFT"][i])
        rows.append({"task": T4_TASKS[i], "k_fd": k_fd, "k_base": k_b, "k_sft": k_s,
                     "p_vs_base": fisher(k_fd, 30, k_b, 30), "p_vs_sft": fisher(k_fd, 30, k_s, 30),
                     "wilson_fd": [round(x, 3) for x in wilson(k_fd, 30)]})
    pf[scen] = rows
    p_vb = [x["p_vs_base"] for x in rows]
    order = np.argsort(p_vb)
    holm = [None] * 8
    prev = 0
    for rank, j in enumerate(order):
        adj = min(1.0, (8 - rank) * p_vb[j])
        prev = max(prev, adj)
        holm[j] = prev
    for j in range(8):
        rows[j]["holm_vs_base"] = holm[j]
    r[f"{scen}_n_sig_vs_base_raw"] = int(sum(p < 0.05 for p in p_vb))
    r[f"{scen}_n_sig_vs_base_holm"] = int(sum(h < 0.05 for h in holm))
    r[f"{scen}_n_sig_vs_sft_raw"] = int(sum(x["p_vs_sft"] < 0.05 for x in rows))
    tot_fd = sum(x["k_fd"] for x in rows); tot_b = sum(x["k_base"] for x in rows); tot_s = sum(x["k_sft"] for x in rows)
    r[f"{scen}_pooled"] = {"fd": tot_fd, "base": tot_b, "sft": tot_s, "n": 240,
                           "p_fd_vs_base": fisher(tot_fd, 240, tot_b, 240), "p_fd_vs_sft": fisher(tot_fd, 240, tot_s, 240)}
r["fisher"] = pf
bf = [(kk(T4["FD"][i]) - kk(T4["Base"][i])) for i in range(8)]
r["sign_fd_gt_base"] = int(sum(T4["FD"][i] > T4["Base"][i] for i in range(8)))
r["sign_test_fd_vs_sft_p"] = float(stats.binomtest(6, 7, 0.5).pvalue)  # 6 승 1 패 (동률 제외)
r["toolbox_text"] = {"base": "4/30 = 13.3%", "fd": "24/30 = 80%"}
r["eps_vs_delta_corr"] = float(np.corrcoef(EPS, d_fd)[0, 1])
# 기준선이 낮을수록 향상 폭이 큰가
r["base_vs_delta_spearman"] = float(stats.spearmanr(T4["Base"], d_fd)[0])
res["table4"] = r

# ---------------------------------------------------------------- 표 5, 6
T5 = {"Euler reverse": (0.0329, 315, 0.59, 0.71), "Optimization (Adam)": (0.0275, 1350, 0.79, 0.75),
      "Trajectory FP (k=5)": (0.0228, 1662, 0.77, 0.68), "Per-step FP (M=5)": (0.00168, 456, 0.87, 0.96)}
T6 = {"Trajectory FP (k=5)": (0.0228, 110, 1662), "Trajectory FP (k=20)": (0.0237, 410, 5657),
      "Per-step FP (M=3)": (0.00397, 40, 405), "Per-step FP (M=5)": (0.00168, 60, 456),
      "Per-step FP (M=10)": (0.00147, 110, 614)}
r = {}
mse = {k: v[0] for k, v in T5.items()}
claims = {
    "euler_over_ps5_x(~20)": mse["Euler reverse"] / mse["Per-step FP (M=5)"],
    "trajfp_over_ps5_x(~14)": mse["Trajectory FP (k=5)"] / mse["Per-step FP (M=5)"],
    "adam_over_ps5_x": mse["Optimization (Adam)"] / mse["Per-step FP (M=5)"],
    "time_trajfp_over_ps5_x(~3)": T5["Trajectory FP (k=5)"][1] / T5["Per-step FP (M=5)"][1],
    "time_adam_over_ps5_x(~3)": T5["Optimization (Adam)"][1] / T5["Per-step FP (M=5)"][1],
    "time_ps5_over_euler_x": T5["Per-step FP (M=5)"][1] / T5["Euler reverse"][1],
    "mse_ps3_over_ps5_x(2.4)": T6["Per-step FP (M=3)"][0] / T6["Per-step FP (M=5)"][0],
    "mse_ps5_over_ps10_x(1.1)": T6["Per-step FP (M=5)"][0] / T6["Per-step FP (M=10)"][0],
    "time_ps10_over_ps5_x(1.3)": T6["Per-step FP (M=10)"][2] / T6["Per-step FP (M=5)"][2],
    "matched_mse_k5_over_m10_x(15)": T6["Trajectory FP (k=5)"][0] / T6["Per-step FP (M=10)"][0],
    "matched_time_k5_over_m10_x(2.7)": T6["Trajectory FP (k=5)"][2] / T6["Per-step FP (M=10)"][2],
    "trajfp_k20_over_k5_mse": T6["Trajectory FP (k=20)"][0] / T6["Trajectory FP (k=5)"][0],
}
r["claimed_ratios"] = {k: round(v, 3) for k, v in claims.items()}
# 모델 평가 수: 표 6 값 vs K*M (본문 4.1), K*(M+1)
K = 10
r["evals_table6_vs_formula"] = {
    "per_step": {f"M={m}": {"table6": T6[f"Per-step FP (M={m})"][1], "K*M": K * m, "K*(M+1)": K * (m + 1)} for m in (3, 5, 10)},
    "trajectory": {f"k={k}": {"table6": T6[f"Trajectory FP (k={k})"][1], "K+2*K*k": K + 2 * K * k} for k in (5, 20)},
}
# 시간 모델: time = c * (JIT 호출 수) + e * (속도 평가 수 + 복원 10). 호출 수: Euler 2, per-step 2, trajectory 2k+2
rows = [("Euler reverse", 2, 20, 315), ("Per-step FP (M=3)", 2, 50, 405), ("Per-step FP (M=5)", 2, 70, 456),
        ("Per-step FP (M=10)", 2, 120, 614), ("Trajectory FP (k=5)", 12, 120, 1662), ("Trajectory FP (k=20)", 42, 420, 5657)]
A = np.array([[c, e] for _, c, e, _ in rows], float)
y = np.array([t for *_, t in rows], float)
sol, resid, *_ = np.linalg.lstsq(A, y, rcond=None)
pred = A @ sol
r["time_model"] = {"c_ms_per_call": float(sol[0]), "e_ms_per_eval": float(sol[1]),
                   "pred": {rows[i][0]: round(float(pred[i]), 1) for i in range(len(rows))},
                   "obs": {rows[i][0]: rows[i][3] for i in range(len(rows))}}
# 같은 모델로 호출 수를 같게 (2회) 맞췄을 때 trajectory k=5 시간
r["time_model"]["trajfp_k5_if_2_calls_ms"] = float(sol[0] * 2 + sol[1] * 120)
# SR 비교 (n = 75 가정)
def sr_k(p):
    return round(p * 75)
fs = {}
for m, (_, _, a_sr, h_sr) in T5.items():
    fs[m] = {"assembly_k75": sr_k(a_sr), "hammer_k75": sr_k(h_sr)}
r["sr_k75"] = fs
ref_a, ref_h = sr_k(0.87), sr_k(0.96)
r["sr_fisher_vs_perstep_n75"] = {m: {"assembly_p": fisher(ref_a, 75, sr_k(T5[m][2]), 75),
                                      "hammer_p": fisher(ref_h, 75, sr_k(T5[m][3]), 75)}
                                 for m in T5 if m != "Per-step FP (M=5)"}
r["sr_grid_k75_round_ok"] = {m: [feasible(T5[m][2], 75)[0], feasible(T5[m][3], 75)[0]] for m in T5}
# 표 5 대 표 1 (같은 설정): Assembly 0.87 대 0.89, Hammer 0.96 대 0.84
r["table5_vs_table1"] = {
    "assembly": {"t5_k75": 65, "t1": .89, "t1_k75": 67, "p": fisher(65, 75, 67, 75)},
    "hammer": {"t5_k75": 72, "t1": .84, "t1_k25": 21, "t1_k75_if_3seeds": 63,
               "p_n75": fisher(72, 75, 63, 75), "p_vs_k25": fisher(72, 75, 21, 25)},
}
# Euler reverse가 기준선 이하라는 주장 (0.59 대 0.64)
r["euler_below_base_assembly_p"] = fisher(sr_k(.59), 75, 16, 25)
res["table5_6"] = r

# ---------------------------------------------------------------- 그림 5, 6 (부록 B.3-B.4)
r = {}
r["fig5_base_60pct_k15"] = abs(0.60 * 15 - round(0.60 * 15)) < 1e-9
r["fig6_base_48pct_n15_possible"] = any(abs(100 * k / 15 - 48) < 0.5 for k in range(16))
r["fig6_base_48pct_n25"] = abs(0.48 * 25 - 12) < 1e-9
r["fig6_dsrl_plateau_80_k15"] = abs(0.80 * 15 - 12) < 1e-9
r["fig6_fd_97pct_k15_possible"] = any(abs(100 * k / 15 - 97) < 0.5 for k in range(16))
r["fig6_fd_97pct_closest_k15"] = [round(100 * k / 15, 1) for k in (14, 15)]
r["fig5_dsrl_70_k15"] = [round(100 * k / 15, 1) for k in (10, 11)]  # 10/15 = 66.7, 11/15 = 73.3
# 단일 시드 15회 평가: 구간
r["fig5_fd_100_wilson_k15"] = [round(x, 3) for x in wilson(15, 15)]
r["fig5_dsrl_70_wilson_k15"] = [round(x, 3) for x in wilson(10, 15)]
r["fig5_gap_fisher_if_15_vs_10_k15"] = fisher(15, 15, 10, 15)
r["fig5_gap_fisher_if_15_vs_11_k15"] = fisher(15, 15, 11, 15)
r["fig6_gap_fisher_14_vs_12"] = fisher(14, 15, 12, 15)
r["fig5_dsrl_gap_pp"] = {"claimed": 25, "read_from_text": "100 - 70 = 30"}
res["fig5_6"] = r

# ---------------------------------------------------------------- 차원과 산술 점검
r = {}
r["cosmos_joint_latent_dim"] = 16 * 9 * 28 * 28
r["pi0_chunk_noise_dim"] = 50 * 32
r["groot_noise_dim"] = 40 * 132
r["groot_dct_k10_dim"] = 10 * 132
r["pi05_dct_k10_dim_code"] = 10 * 32
r["fig4_vram_ratio"] = {"lora_over_fd": 22.5 / 8, "sft_over_fd": 70 / 8}
r["real_world_toolbox"] = {"base_k": 4, "fd_k": 24, "n": 30, "base_pct": 100 * 4 / 30, "fd_pct": 100 * 24 / 30}
res["arith"] = r

# ---------------------------------------------------------------- toy: 역변환 정확도 (식 3, 표 5의 순서)
def make_field(D, h, L, rng):
    W1 = rng.standard_normal((h, D)) / math.sqrt(D)
    W2 = rng.standard_normal((D, h)) / math.sqrt(h)
    c = rng.standard_normal(h)
    s1 = np.linalg.norm(W1, 2)
    s2 = np.linalg.norm(W2, 2)
    scale = L / (s1 * s2)  # tanh의 기울기 <= 1 이므로 첫 인수에 대한 립시츠 상수 <= L
    b = rng.standard_normal(h) * 0.5

    def v(x, t):
        return scale * (W2 @ np.tanh(W1 @ x + b + c * t))
    return v


def forward(v, w, K):
    x = w.copy()
    dt = 1.0 / K
    for k in range(K):
        x = x + dt * v(x, k * dt)
    return x


def explicit_reverse(v, a, K):
    x = a.copy()
    dt = 1.0 / K
    for k in range(K - 1, -1, -1):
        x = x - dt * v(x, k * dt)
    return x


def perstep_fp(v, a, K, M, init="explicit"):
    """init='explicit': 코드 방식(명시적 역 1회 후 FP M회). init='paper': 식 (3) (x^(0)=x_{k+1}, FP M회)."""
    x = a.copy()
    dt = 1.0 / K
    for k in range(K - 1, -1, -1):
        t = k * dt
        xk = x - dt * v(x, t) if init == "explicit" else x.copy()
        for _ in range(M):
            xk = x - dt * v(xk, t)
        x = xk
    return x


def traj_fp(v, a, K, kiter):
    w = explicit_reverse(v, a, K)
    for _ in range(kiter):
        rec = forward(v, w, K)
        w = explicit_reverse(v, a + (a - rec), K)
    return w


def mse(v, w, a, K):
    return float(np.mean((forward(v, w, K) - a) ** 2))


def toy(seed=0, D=32, h=64, L=3.0, K=10, trials=30, off=0.3):
    rng = np.random.default_rng(seed)
    out = {}
    methods = {"euler_reverse": lambda v, a: explicit_reverse(v, a, K)}
    for kk_ in (1, 5, 20):
        methods[f"traj_fp_k{kk_}"] = (lambda kk__: (lambda v, a: traj_fp(v, a, K, kk__)))(kk_)
    for M in (1, 3, 5, 10):
        methods[f"perstep_code_M{M}"] = (lambda M_: (lambda v, a: perstep_fp(v, a, K, M_, "explicit")))(M)
        methods[f"perstep_paper_M{M}"] = (lambda M_: (lambda v, a: perstep_fp(v, a, K, M_, "paper")))(M)
    acc = {m: [] for m in methods}
    for t in range(trials):
        v = make_field(D, h, L, rng)
        w0 = rng.standard_normal(D)
        a = forward(v, w0, K) + off * rng.standard_normal(D) * 0.3  # 기본 정책 밖으로 조금 벗어난 전문가 행동
        for m, f in methods.items():
            acc[m].append(mse(v, f(v, a), a, K))
    return {m: float(np.mean(x)) for m, x in acc.items()}


toyres = {}
toyres["L3_K10"] = toy(L=3.0, K=10)
# 립시츠 상수 스윕 (dt*L < 1 조건): K = 10에서 L = 1, 3, 6, 9, 12, 20
sweep = {}
for L in (1.0, 3.0, 6.0, 9.0, 12.0, 20.0):
    t_ = toy(L=L, K=10, trials=20)
    sweep[str(L)] = {"euler": t_["euler_reverse"], "perstep_code_M5": t_["perstep_code_M5"],
                     "perstep_paper_M5": t_["perstep_paper_M5"], "traj_k5": t_["traj_fp_k5"]}
toyres["lipschitz_sweep_K10"] = sweep
# 단계 수 스윕 (L = 6 고정): 명시적 역과 FP의 dt 의존성
ksweep = {}
for K_ in (5, 10, 20, 50, 100):
    t_ = toy(L=6.0, K=K_, trials=20)
    ksweep[str(K_)] = {"euler": t_["euler_reverse"], "perstep_code_M5": t_["perstep_code_M5"], "traj_k5": t_["traj_fp_k5"]}
toyres["step_sweep_L6"] = ksweep
res["toy_inversion"] = toyres

# ---------------------------------------------------------------- toy: DCT 기저 압축 (코드 noise_basis_k = 10, 길이 50)
def dct_basis(K, N=50):
    Phi = np.zeros((N, K))
    for k in range(K):
        for n in range(N):
            Phi[n, k] = math.cos(math.pi * k * (2 * n + 1) / (2 * N))
        Phi[:, k] *= math.sqrt(1.0 / N) if k == 0 else math.sqrt(2.0 / N)
    return Phi


r = {}
for K_ in (1, 5, 10, 20, 50):
    Phi = dct_basis(K_)
    r[f"K{K_}"] = {
        "energy_fraction_of_white_noise": float(np.trace(Phi.T @ Phi) / 50),
        "rowvar_of_expanded_N01_coeffs_mean": float(np.mean(np.sum(Phi ** 2, axis=1))),
        "orthonormal_columns": bool(np.allclose(Phi.T @ Phi, np.eye(K_), atol=1e-6)),
    }
# 시간 평활한 (저주파) 신호의 K=10 잔차: 행동 청크처럼 매끄러운 경우 vs 백색 잡음
Phi = dct_basis(10)
smooth = np.cumsum(np.random.default_rng(1).standard_normal((50, 32)), axis=0)
smooth /= smooth.std()
white = np.random.default_rng(2).standard_normal((50, 32))
def resid(x):
    c = np.linalg.lstsq(Phi, x, rcond=None)[0]
    return float(np.sum((x - Phi @ c) ** 2) / np.sum(x ** 2))
r["k10_relative_residual_energy"] = {"white_noise": resid(white), "random_walk_smooth": resid(smooth)}
res["toy_dct"] = r

# ---------------------------------------------------------------- 저장
OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
print("wrote", OUT)
if __name__ == "__main__":
    print(json.dumps({k: res[k] for k in ("table1", "table2")}, indent=1, ensure_ascii=False, default=str)[:6000])
