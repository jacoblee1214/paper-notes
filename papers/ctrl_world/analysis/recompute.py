"""Ctrl-World (arXiv:2510.10125 v3, ICLR 2026) 표 재계산과 일관성 검사 (numpy, scipy).
실행: python3 analysis/recompute.py  ->  analysis/recompute_results.json
논문 표 값은 paper/paper.txt(v3, pdftotext -layout)에서 옮겼다. 원자료(시행별 결과)가 없으므로
검사는 '보고된 값끼리' 맞는지와 보고된 요약이 셀 값에서 나오는지만 본다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}

# ------------------------------------------------ Table 1 (video quality, third-view, 256 clips)
# (method, PSNR, SSIM, LPIPS, FID, FVD)
T1 = [
    ("WPE-Single-View", 20.33, 0.772, 0.131, 25.50, 156.4),
    ("IRASim-Single-View", 21.36, 0.774, 0.117, 26.46, 138.1),
    ("Ctrl-World-Single-View", 21.27, 0.793, 0.110, 23.47, 127.5),
    ("Ctrl-World (multi-view)", 23.56, 0.828, 0.091, 25.00, 97.4),
]
# Ctrl-World-Single-View 대 IRASim-Single-View: 지표별 우열 (본문은 "outperforms these prior models")
t1_cmp = {}
for k, name in zip(["PSNR", "SSIM", "LPIPS", "FID", "FVD"], range(1, 6)):
    cw, ir = T1[2][name], T1[1][name]
    higher_better = k in ("PSNR", "SSIM")
    better = (cw > ir) if higher_better else (cw < ir)
    t1_cmp[k] = {"CtrlWorld_SV": cw, "IRASim_SV": ir, "CtrlWorld_better": bool(better)}
res["table1_singleview_vs_IRASim"] = t1_cmp
res["table1_singleview_vs_IRASim_note"] = (
    "PSNR에서 Ctrl-World-Single-View(21.27)가 IRASim-Single-View(21.36)보다 낮다. "
    "본문 'outperforms these prior models'는 PSNR을 빼면 맞다.")

# Table 2 third-view 'Ctrl-World' row must equal Table 1 'Ctrl-World (ours)' row
t2_third_full = (23.56, 0.828, 0.091, 25.00, 97.4)
res["table2_row_equals_table1"] = bool(tuple(T1[3][1:]) == t2_third_full)

# ------------------------------------------------ Table 3 (real vs world model, 8 tasks x 3 policies)
# (task, policy, instr_real, instr_wm, succ_real, succ_wm)
T3 = [
    ("Pick-Place", "pi0", 0.75, 0.60, 0.70, 0.55),
    ("Pick-Place", "pi0-fast", 0.90, 0.75, 0.85, 0.70),
    ("Pick-Place", "pi05", 0.90, 0.80, 0.85, 0.70),
    ("Fold-Towel", "pi0", 0.40, 0.20, 0.40, 0.20),
    ("Fold-Towel", "pi0-fast", 0.55, 0.45, 0.55, 0.40),
    ("Fold-Towel", "pi05", 0.80, 0.75, 0.75, 0.65),
    ("Drawer", "pi0", 0.05, 0.00, 0.05, 0.00),
    ("Drawer", "pi0-fast", 0.20, 0.20, 0.15, 0.05),
    ("Drawer", "pi05", 0.80, 0.65, 0.80, 0.30),
    ("Wipe-table", "pi0", 0.20, 0.10, 0.20, 0.00),
    ("Wipe-table", "pi0-fast", 0.50, 0.35, 0.50, 0.35),
    ("Wipe-table", "pi05", 0.80, 0.70, 0.80, 0.65),
    ("Close-laptop", "pi0", 0.15, 0.20, 0.15, 0.00),
    ("Close-laptop", "pi0-fast", 0.60, 0.50, 0.50, 0.00),
    ("Close-laptop", "pi05", 0.80, 0.70, 0.70, 0.05),
    ("Pull-tissue", "pi0", 0.30, 0.20, 0.30, 0.20),
    ("Pull-tissue", "pi0-fast", 0.10, 0.00, 0.10, 0.00),
    ("Pull-tissue", "pi05", 0.85, 0.60, 0.75, 0.50),
    ("Stack", "pi0", 0.75, 0.45, 0.60, 0.40),
    ("Stack", "pi0-fast", 0.80, 0.70, 0.70, 0.55),
    ("Stack", "pi05", 0.80, 0.65, 0.65, 0.55),
]
ir = np.array([r[2] for r in T3]); iw = np.array([r[3] for r in T3])
sr = np.array([r[4] for r in T3]); sw = np.array([r[5] for r in T3])

def ols(x, y):
    s = stats.linregress(x, y)
    return {"slope": round(s.slope, 3), "intercept": round(s.intercept, 3), "r": round(s.rvalue, 3), "n": len(x)}

res["table3_fig7_regression_recomputed"] = {
    "instruction (WM on real)": ols(ir, iw),
    "success (WM on real)": ols(sr, sw),
    "figure7_labels": {"instruction": "y = 0.87x - 0.04", "success": "y = 0.81x - 0.11"},
}
res["table3_pearson_spearman"] = {
    "instruction": {"pearson": round(stats.pearsonr(ir, iw)[0], 3), "spearman": round(stats.spearmanr(ir, iw)[0], 3)},
    "success": {"pearson": round(stats.pearsonr(sr, sw)[0], 3), "spearman": round(stats.spearmanr(sr, sw)[0], 3)},
}
res["table3_means"] = {
    "instruction_real": round(ir.mean(), 3), "instruction_wm": round(iw.mean(), 3),
    "success_real": round(sr.mean(), 3), "success_wm": round(sw.mean(), 3),
    "success_gap_wm_minus_real_mean": round((sw - sr).mean(), 3),
    "success_gap_sd": round((sw - sr).std(ddof=1), 3),
    "rows_wm_below_real_success": int((sw < sr).sum()), "rows_total": len(T3),
}
res["table3_success_le_instruction_violations"] = [
    f"{r[0]} {r[1]}" for r in T3 if r[4] > r[2] + 1e-9 or r[5] > r[3] + 1e-9]

# within-task policy ordering agreement (3 policies per task, 3 pairs per task, 21 pairs total)
agree_s = agree_i = pairs = ties_s = ties_i = 0
tasks = sorted(set(r[0] for r in T3), key=lambda t: [r[0] for r in T3].index(t))
for t in tasks:
    rows = [r for r in T3 if r[0] == t]
    for a in range(len(rows)):
        for b in range(a + 1, len(rows)):
            pairs += 1
            dr = rows[a][4] - rows[b][4]; dw = rows[a][5] - rows[b][5]
            if dr == 0 or dw == 0:
                ties_s += 1
            elif np.sign(dr) == np.sign(dw):
                agree_s += 1
            di = rows[a][2] - rows[b][2]; diw = rows[a][3] - rows[b][3]
            if di == 0 or diw == 0:
                ties_i += 1
            elif np.sign(di) == np.sign(diw):
                agree_i += 1
res["within_task_pair_order_agreement"] = {
    "pairs": pairs,
    "success_agree": agree_s,
    "success_ties_excluded": ties_s,
    "instruction_agree": agree_i,
    "instruction_ties_excluded": ties_i,
    "disagree_success": pairs - ties_s - agree_s,
    "disagree_instruction": pairs - ties_i - agree_i,
    "note": "Table 3은 7과제 x 3정책 = 21행, 과제 내 정책 쌍 21개. 동점 쌍은 일치로도 불일치로도 세지 않음",
}

# paired tests on Table 3 (21 rows): WM vs real. n per cell은 0.05 단위 값에서 20으로 추정 (논문 미기재)
rng_b = np.random.default_rng(7)
def boot_ci(x, reps=10000):
    idx = rng_b.integers(0, len(x), size=(reps, len(x)))
    m = x[idx].mean(axis=1)
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]
gs = sw - sr
gi = iw - ir
res["table3_paired_tests"] = {
    "success_wm_minus_real": {
        "mean": round(float(gs.mean()), 3), "bootstrap95_over_rows": boot_ci(gs),
        "paired_t_p": float(round(stats.ttest_rel(sw, sr).pvalue, 6)),
        "wilcoxon_p": float(round(stats.wilcoxon(sw, sr).pvalue, 6)),
    },
    "instruction_wm_minus_real": {
        "mean": round(float(gi.mean()), 3), "bootstrap95_over_rows": boot_ci(gi),
        "paired_t_p": float(round(stats.ttest_rel(iw, ir).pvalue, 6)),
        "wilcoxon_p": float(round(stats.wilcoxon(iw, ir).pvalue, 6)),
    },
    "note": "행 21개(과제 x 정책)를 독립 단위로 두었다. 같은 과제의 정책끼리 상관될 수 있어 p는 낙관적이다",
}

# ------------------------------------------------ Tables 4-7 (policy improvement)
CATS = {
    "Spatial (T4)": ([0.50, 0.45, 0.30, 0.45, 0.15, 0.20, 0.05, 0.20], [0.85, 0.90, 1.00, 0.80, 0.85, 0.90, 0.90, 0.80], 0.2875, 0.875),
    "Shape (T5)": ([0.40, 0.45, 0.40, 0.50], [0.85, 0.95, 0.95, 0.90], 0.4374, 0.9125),
    "Towel-dir (T6)": ([0.60, 0.50, 0.55, 0.65], [0.75, 0.80, 0.85, 0.80], 0.575, 0.80),
    "Novel-obj (T7)": ([0.20, 0.30], [0.80, 0.70], 0.25, 0.75),
}
cat_rows = {}
for name, (b, a, rb, ra) in CATS.items():
    cat_rows[name] = {
        "base_recomputed": round(float(np.mean(b)), 4), "base_reported": rb,
        "after_recomputed": round(float(np.mean(a)), 4), "after_reported": ra,
        "cells": len(b),
    }
res["policy_improvement_categories"] = cat_rows
base_avg = np.mean([np.mean(v[0]) for v in CATS.values()])
after_avg = np.mean([np.mean(v[1]) for v in CATS.values()])
res["policy_improvement_headline"] = {
    "base_avg_pct": round(100 * base_avg, 2), "after_avg_pct": round(100 * after_avg, 2),
    "absolute_gain_pp": round(100 * (after_avg - base_avg), 2),
    "relative_gain_pct": round(100 * (after_avg / base_avg - 1), 1),
    "note": "초록/결론의 '44.7%'는 절대 차이(percentage point). 상대 증가는 약 115%",
}

# Approximate 95% CI for the 4-category average gain, assuming each cell is a binomial with n=20
# (논문은 n을 명시하지 않음. 0.05 단위 값과 README의 '카테고리당 20회'로 추정).
N_ASSUMED = 20
var = 0.0
gains = []
for name, (b, a, _, _) in CATS.items():
    k = len(b)
    gains.append(np.mean(a) - np.mean(b))
    vb = sum(p * (1 - p) / N_ASSUMED for p in b) / k**2
    va = sum(p * (1 - p) / N_ASSUMED for p in a) / k**2
    var += (va + vb) / 16.0   # 4 categories: average of 4 gains -> divide by 4^2
se = math.sqrt(var)
mean_gain = float(np.mean(gains))
res["policy_improvement_ci_assumed_n20"] = {
    "mean_gain_pp": round(100 * mean_gain, 2),
    "ci95_pp": [round(100 * (mean_gain - 1.96 * se), 2), round(100 * (mean_gain + 1.96 * se), 2)],
    "assumption": "셀마다 n=20 이항, 셀 간 독립, 카테고리 평균을 동일 가중 (논문이 n을 명시하지 않아 추정)",
    "caveat": "카테고리 4개뿐이라 정규근사는 거칠다. 과제 수준 변동을 반영하면 구간은 더 넓다.",
}

# Table 5 reported 0.4374 vs recomputed 0.4375 (rounding), check
res["rounding_check_table5"] = {"recomputed": round(np.mean(CATS["Shape (T5)"][0]), 6), "reported": 0.4374}

# ------------------------------------------------ Table 3 granularity: is n=20 consistent?
vals = sorted(set([r[2] for r in T3] + [r[3] for r in T3] + [r[4] for r in T3] + [r[5] for r in T3]))
res["table3_granularity"] = {
    "all_values_multiple_of_0.05": all(abs(round(v / 0.05) * 0.05 - v) < 1e-9 for v in vals),
    "n_if_20": "모든 값이 0.05의 배수 -> 시도 수 n=20과 맞음 (README의 '카테고리당 20회'와 일치)",
}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
print(json.dumps(res, ensure_ascii=False, indent=2))
