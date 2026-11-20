"""TesserAct (arXiv:2504.20995v1, ICCV 2025) 표 재계산 (numpy, scipy). 입력은 논문에 적힌 값뿐이다."""
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}

# ---- 표 1: 데이터 규모 ----
t1 = {"RLBench(synthetic)": 20 * 1000 * 4, "Fractal(real)": 80000, "Bridge(real)": 25000, "SSv2(real, human hand)": 100000}
out["table1"] = {
    "rlbench_formula": 20 * 1000 * 4,
    "real_total": t1["Fractal(real)"] + t1["Bridge(real)"] + t1["SSv2(real, human hand)"],
    "text_claim": "only about 200k data points (real)",
}

# ---- 표 3: 행동 계획 성공률 (100 에피소드 / 과제) ----
tasks = ["close box", "open drawer", "open jar", "open microwave", "put knife", "sweep to dustpan", "lid off", "weighing off", "water plants"]
bc = np.array([53, 4, 0, 5, 0, 0, 12, 21, 0])
unipi = np.array([81, 67, 38, 72, 66, 49, 70, 68, 35])
ours = np.array([88, 80, 44, 70, 70, 56, 73, 62, 41])
n = 100


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(c - h), float(c + h)


t3 = {"tasks": tasks, "means": {}, "per_task": []}
for name, arr in [("Image-BC", bc), ("UniPi*", unipi), ("4DWM", ours)]:
    t3["means"][name] = float(arr.mean())
t3["mean_diff_4DWM_minus_UniPi_pp"] = float(ours.mean() - unipi.mean())
t3["mean_diff_4DWM_minus_BC_pp"] = float(ours.mean() - bc.mean())
t3["wins_vs_UniPi"] = int(np.sum(ours > unipi))
t3["ties_vs_UniPi"] = int(np.sum(ours == unipi))
t3["losses_vs_UniPi"] = int(np.sum(ours < unipi))
t3["wins_vs_BC"] = int(np.sum(ours > bc))
for i, t in enumerate(tasks):
    fe = stats.fisher_exact([[ours[i], n - ours[i]], [unipi[i], n - unipi[i]]])[1]
    t3["per_task"].append({
        "task": t,
        "ours": int(ours[i]), "unipi": int(unipi[i]), "bc": int(bc[i]),
        "ours_ci95": [round(v * 100, 1) for v in wilson(ours[i], n)],
        "unipi_ci95": [round(v * 100, 1) for v in wilson(unipi[i], n)],
        "fisher_p_ours_vs_unipi": round(float(fe), 3),
    })

# 과제 단위 쌍체 검정 (과제를 단위로 한 n=9)
d = ours - unipi
t3["task_level"] = {
    "paired_t_p": round(float(stats.ttest_rel(ours, unipi).pvalue), 4),
    "wilcoxon_p": round(float(stats.wilcoxon(ours, unipi).pvalue), 4),
    "sign_test_p_two_sided": round(float(stats.binomtest(int(np.sum(d > 0)), int(np.sum(d != 0)), 0.5).pvalue), 4),
    "mean_diff_pp": round(float(d.mean()), 2),
    "sd_diff_pp": round(float(d.std(ddof=1)), 2),
    "t_ci95_pp": [round(float(v), 2) for v in (d.mean() - stats.t.ppf(0.975, 8) * d.std(ddof=1) / 3, d.mean() + stats.t.ppf(0.975, 8) * d.std(ddof=1) / 3)],
}

# 풀링 두 비율 검정 (에피소드 900개를 독립으로 가정: 과대 추정 가능)
k1, k2 = int(ours.sum()), int(unipi.sum())
N = 9 * n
p1, p2 = k1 / N, k2 / N
pp = (k1 + k2) / (2 * N)
se = np.sqrt(pp * (1 - pp) * (2 / N))
z = (p1 - p2) / se
t3["pooled_two_prop"] = {
    "ours_successes": k1, "unipi_successes": k2, "N_each": N,
    "diff_pp": round((p1 - p2) * 100, 2),
    "z": round(float(z), 3),
    "p_two_sided": round(float(2 * (1 - stats.norm.cdf(abs(z)))), 4),
    "diff_ci95_unpaired_pp": [round(float((p1 - p2 - 1.96 * np.sqrt(p1 * (1 - p1) / N + p2 * (1 - p2) / N)) * 100), 2),
                              round(float((p1 - p2 + 1.96 * np.sqrt(p1 * (1 - p1) / N + p2 * (1 - p2) / N)) * 100), 2)],
}
# 과제별 Fisher 결과 중 유의한 것
t3["any_task_significant_005"] = any(r["fisher_p_ours_vs_unipi"] < 0.05 for r in t3["per_task"])
out["table3"] = t3

# ---- 표 2: 4D 장면 (Chamfer 등) 상대 차이 ----
chamfer = {
    "real": {"4D Point-E": 0.2211, "OpenSora": 0.3013, "CogVideoX": 0.2191, "TesserAct": 0.2030},
    "synthetic": {"4D Point-E": 0.1086, "OpenSora": 0.2570, "CogVideoX": 0.2884, "TesserAct": 0.0811},
}
out["table2_chamfer_rel"] = {
    dom: {m: round(100 * (v - vals["TesserAct"]) / v, 1) for m, v in vals.items() if m != "TesserAct"}
    for dom, vals in chamfer.items()
}
# TesserAct가 각 지표에서 1위인지 (낮을수록 좋은 지표는 min, 높을수록 좋은 지표는 max)
t2 = {
    "real": {"FVD": (21.59, "min", [23.67, 20.64, 21.59]), "SSIM": (75.86, "max", [71.31, 79.38, 75.86]),
             "PSNR": (20.27, "max", [19.25, 22.39, 20.27]), "AbsRel": (22.07, "min", [31.41, 26.17, 22.07]),
             "d1": (66.80, "max", [60.39, 64.82, 66.80]), "d2": (82.60, "max", [79.97, 81.62, 82.60]),
             "NormMean": (15.74, "min", [41.82, 19.53, 15.74]), "NormMedian": (7.32, "min", [32.15, 10.09, 7.32]),
             "11.25": (27.80, "max", [13.58, 22.70, 27.80]), "Chamfer": (0.2030, "min", [0.3013, 0.2191, 0.2030])},
    "synthetic": {"FVD": (40.01, "min", [54.11, 41.23, 40.01]), "SSIM": (77.59, "max", [65.90, 76.60, 77.59]),
                  "PSNR": (19.73, "max", [19.28, 20.87, 19.73]), "AbsRel": (16.02, "min", [18.40, 19.81, 16.02]),
                  "d1": (69.26, "max", [65.02, 60.07, 69.26]), "d2": (93.03, "max", [91.20, 80.16, 93.03]),
                  "NormMean": (14.75, "min", [12.94, 20.36, 14.75]), "NormMedian": (6.34, "min", [7.58, 10.47, 6.34]),
                  "11.25": (36.85, "max", [25.02, 26.04, 36.85]), "Chamfer": (0.0811, "min", [0.2570, 0.2884, 0.0811])},
}
best = {}
for dom, rows in t2.items():
    best[dom] = {}
    for m, (val, sense, col) in rows.items():
        ref = min(col) if sense == "min" else max(col)
        best[dom][m] = "best" if val == ref else "not best"
out["table2_best_check"] = best
out["table2_real_rgb_gap_vs_cogvideox"] = {"FVD_delta": round(21.59 - 20.64, 2), "PSNR_delta_dB": round(20.27 - 22.39, 2), "SSIM_delta": round(75.86 - 79.38, 2)}

# ---- 표 4: 시간 ----
out["table4"] = {"SoM_hours": 2, "ours_minutes": 1, "ratio_approx": 120, "note": "시간 측정 하드웨어 미기재"}

# 샘플 수 (Sec. 5.1.1)
out["sample_counts"] = {"real_samples": 400, "synthetic_samples": 200, "runs_per_sample": 10}

with open(os.path.join(HERE, "recompute_results.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
