"""DreamGen (arXiv:2505.12705 v2) 보고 수치 재계산 및 표 간 일관성 점검.

입력: 논문 표에서 옮긴 숫자 (표 번호와 쪽을 주석으로 표시). 네트워크·GPU 불필요.
출력: recompute_results.json (같은 폴더)
"""
import json
import os
import numpy as np
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recompute_results.json")
res = {}

# ---------- Table 1 (p.7): GR00T N1 vs GR00T N1 + DreamGen, success rate (%) ----------
t1_beh_names = ["Open Microwave", "Open Macbook", "Close Lunchbox", "Hit Tambourine", "Hit Keyboard",
                "Grab button", "Pour Water", "Water flowers", "Light Candle", "Use Vacuum",
                "Iron shirt", "Take Spoon Out", "Unroll mat", "Move Mouse"]
t1_beh_base = [0, 0, 0, 5, 0, 45, 40, 50, 10, 0, 0, 7, 0, 0]
t1_beh_dg = [23, 45, 10, 15, 90, 75, 55, 95, 15, 55, 20, 17, 55, 35]
t1_env_names = ["Pick up Tangerine", "Box sandwich", "Weigh the Orange", "Put cup in trash",
                "Put pear in basket", "Put sauce on tray", "Water Flowers", "Lift Basket",
                "Swirl Around Spoon", "Use Whisk", "Close soup container", "Uncover Pot", "Cover Pot"]
t1_env_base = [0] * 13
t1_env_dg = [30, 10, 20, 45, 35, 45, 15, 55, 15, 25, 55, 30, 35]
res["table1"] = {
    "behavior_mean_base": round(np.mean(t1_beh_base), 2),       # reported 11.2
    "behavior_mean_dg": round(np.mean(t1_beh_dg), 2),           # reported 43.2
    "behavior_n": len(t1_beh_dg),
    "env_n": len(t1_env_dg),
    "env_mean_base": round(np.mean(t1_env_base), 2),            # reported 0.0
    "env_mean_dg_recomputed": round(np.mean(t1_env_dg), 2),     # reported 28.5
    "env_sum_dg": int(sum(t1_env_dg)),
    "env_mean_dg_if_denominator_14": round(sum(t1_env_dg) / 14, 2),
    "note": "Table 1 env 평균 28.5는 13개 과제 단순평균(31.9)과 다름. 본문 11.8% (p.7 l.563) vs 표 11.2 불일치.",
}

# Table 1 is score-based (partial credit), so per-cell binomial tests are not valid.
# Paired Wilcoxon over the 14 behavior tasks (cell-level, descriptive only):
w = stats.wilcoxon(np.array(t1_beh_dg) - np.array(t1_beh_base))
res["table1"]["wilcoxon_behavior_p"] = float(w.pvalue)

# ---------- Table 4 (RoboCasa, p.17): 24 tasks, 7 columns ----------
cols = ["30", "100", "300", "30+NT", "100+NT", "300+NT", "ONLY_NT"]
t4 = np.array([
 [0.93,3.92,19.61,5.77,13.46,25.00,1.96],[1.85,6.86,36.27,3.85,19.23,50.96,16.67],
 [0.00,0.00,12.75,0.00,9.62,19.23,0.00],[0.00,0.98,9.80,0.00,12.50,33.65,1.96],
 [0.00,0.00,23.53,0.00,12.50,42.31,8.82],[0.00,0.00,15.69,0.00,14.42,28.85,0.00],
 [0.00,5.88,33.33,3.85,28.85,60.58,0.98],[0.00,0.00,29.41,0.96,9.62,58.65,5.88],
 [0.00,43.14,74.51,9.62,52.88,82.69,2.94],[0.00,12.75,14.71,0.00,8.65,28.85,0.00],
 [49.07,67.65,83.33,51.92,80.77,94.23,52.94],[20.37,54.90,58.82,44.23,55.77,47.12,15.69],
 [76.85,96.08,99.02,88.46,98.08,98.08,82.35],[9.26,42.16,79.41,33.65,68.27,74.04,33.33],
 [14.81,25.49,55.88,21.15,27.88,51.92,17.65],[4.63,15.69,26.47,7.69,13.46,25.96,6.86],
 [49.07,67.65,72.55,51.92,69.23,95.19,59.80],[24.07,42.16,52.94,37.50,45.19,59.62,28.43],
 [33.33,59.80,62.75,48.08,67.31,72.12,25.49],[47.22,57.84,70.59,55.77,75.96,76.92,29.41],
 [55.56,73.53,78.43,49.04,52.88,72.12,48.04],[27.78,56.86,85.29,34.62,63.46,83.65,48.04],
 [3.70,34.31,72.55,11.54,48.08,74.04,2.94],[0.00,1.96,22.55,0.00,10.58,26.92,2.94]])
reported_avg = [17.44, 32.07, 49.59, 23.32, 39.94, 57.61, 20.55]
res["table4"] = {
    "n_tasks": int(t4.shape[0]),
    "recomputed_avg": [round(float(x), 2) for x in t4.mean(axis=0)],
    "reported_avg": reported_avg,
}
pairs = {"300 vs 300+NT": (2, 5), "30 vs 30+NT": (0, 3), "100 vs 100+NT": (1, 4), "300 vs ONLY_NT": (2, 6)}
res["table4"]["paired_tests"] = {}
for name, (a, b) in pairs.items():
    d = t4[:, b] - t4[:, a]
    tt = stats.ttest_rel(t4[:, b], t4[:, a])
    wx = stats.wilcoxon(t4[:, b], t4[:, a])
    res["table4"]["paired_tests"][name] = {
        "mean_diff": round(float(d.mean()), 2),
        "sd_diff": round(float(d.std(ddof=1)), 2),
        "ci95_t": [round(float(x), 2) for x in (d.mean() + np.array([-1, 1]) * stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))],
        "t_p": float(tt.pvalue), "wilcoxon_p": float(wx.pvalue),
        "tasks_improved": int((d > 0).sum()), "tasks_worse": int((d < 0).sum()),
    }
# Note: per-task trial counts for simulation are not stated; the task-level test treats each task as one unit.

# ---------- Table 5 (p.18): real-world, GR1 averages and per-cell changes ----------
t5 = {
 "DP": {"GR1": [35.0, 23.3, 6.6, 25.0], "avg": 22.0, "Franka": [20.0, 0.0, 10.0], "SO100": None},
 "DP+NT": {"GR1": [15.0, 33.3, 26.4, 35.0], "avg": 27.0, "Franka": [30.0, 20.0, 10.0], "SO100": None},
 "GR00T": {"GR1": [60.0, 36.6, 27.0, 25.0], "avg": 37.0, "Franka": [40.0, 10.0, 20.0], "SO100": [17.0, 25.0]},
 "GR00T+NT": {"GR1": [65.0, 49.0, 37.0, 35.0], "avg": 46.0, "Franka": [60.0, 20.0, 30.0], "SO100": [26.0, 65.0]},
 "pi0": {"GR1": None, "Franka": [30.0, 10.0, 20.0], "SO100": None},
 "pi0+NT": {"GR1": None, "Franka": [40.0, 20.0, 20.0], "SO100": None},
}
res["table5"] = {"gr1_avg_recomputed": {k: round(float(np.mean(v["GR1"])), 2) for k, v in t5.items() if v.get("GR1")}}
res["table5"]["gr1_avg_reported"] = {k: v["avg"] for k, v in t5.items() if "avg" in v}
res["table5"]["gr1_avg_text_claim"] = {"baseline": 37.0, "with_NT": 46.4}
# GR1 pooled over 4 tasks, 10 rollouts per checkpoint per task (p.20 App. I.1), approximate
# two-proportion z-test on the four-task mean (treating cell values as proportions of 10 trials).
n_per_arm = 4 * 10
p1 = np.mean(t5["GR00T"]["GR1"]) / 100
p2 = np.mean(t5["GR00T+NT"]["GR1"]) / 100
se = np.sqrt(p1 * (1 - p1) / n_per_arm + p2 * (1 - p2) / n_per_arm)
z = (p2 - p1) / se
res["table5"]["gr1_pooled_z"] = {"diff_pp": round(100 * (p2 - p1), 2), "z": round(float(z), 2),
                                  "p_two_sided": float(2 * (1 - stats.norm.cdf(abs(z)))),
                                  "note": "cell 값을 10회 비율로 가정한 근사. 부분점수 사용 셀이 있어 근사치."}

# Per-cell paired changes (policy x task), counts of improved / flat / worse
changes = []
for pol, base, nt in [("DP", "DP", "DP+NT"), ("pi0", "pi0", "pi0+NT"), ("GR00T", "GR00T", "GR00T+NT")]:
    for emb in ["GR1", "Franka", "SO100"]:
        b = t5[base].get(emb); n = t5[nt].get(emb)
        if b is None or n is None:
            continue
        for i, (x, y) in enumerate(zip(b, n)):
            changes.append((pol, emb, i, round(y - x, 1)))
res["table5"]["paired_cell_changes"] = {
    "n_cells": len(changes),
    "improved": sum(1 for c in changes if c[3] > 0),
    "flat": sum(1 for c in changes if c[3] == 0),
    "worse": [c for c in changes if c[3] < 0],
}
# Fisher exact on Franka cells that are exact multiples of 10 (= integer successes out of 10, App. I.2)
fisher = {}
names = {"Franka": ["Pick&Place", "CubeStacking", "ToolUsage"]}
for emb in ["Franka"]:
    for i, name in enumerate(names[emb]):
        x = t5["GR00T"][emb][i]; y = t5["GR00T+NT"][emb][i]
        if x % 10 or y % 10:
            continue
        sx, sy = int(round(x / 10)), int(round(y / 10))
        fisher[f"GR00T->GR00T+NT {emb} {name}"] = {"successes_of_10": [sx, sy],
            "fisher_p": float(stats.fisher_exact([[sy, 10 - sy], [sx, 10 - sx]])[1])}
res["table5"]["fisher_gr00t_franka_so100"] = fisher

# Wilson 95% CI for a few GR00T N1 cells treated as 10-trial proportions
def wilson(p, n=10, z=1.96):
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(centre - half, 3), round(centre + half, 3)
res["table5"]["wilson_examples_gr00t_nt_franka_pnp_0.6"] = wilson(0.6)

# ---------- Table 2 + Figure 6 (p.8) ----------
# RoboCasa IF(GPT) and PA columns, sft and zero-shot rows.
t2 = {
 "Hunyuan-zero": (1.0, 0.0), "CogVideoX-zero": (0.0, 0.0), "WAN2.1-zero": (0.0, 0.0), "Cosmos-zero": (4.2, 22.9),
 "Hunyuan-sft": (68.8, 44.8), "CogVideoX-sft": (72.9, 44.8), "WAN2.1-sft": (77.1, 55.3), "Cosmos-sft": (79.2, 61.5),
}
avg_ifpa = {k: round((a + b) / 2, 2) for k, (a, b) in t2.items()}
res["table2_fig6"] = {"x_avg_IF_GPT_PA_RoboCasa": avg_ifpa,
                      "note": "그림 6의 x값(DreamGen Bench score)이 RoboCasa IF(GPT)와 PA 평균과 맞는지 확인."}
# y values (RoboCasa score) are read from Figure 6 by eye (approximate, not in any table).
y_read = {"WAN2.1-sft": 11.5, "CogVideoX-sft": 11.0, "Hunyuan-sft": 9.7, "Cosmos-sft": 21.0}
xs = np.array([avg_ifpa[k] for k in y_read]); ys = np.array([y_read[k] for k in y_read])
r_all = stats.pearsonr(xs, ys)
r_no_cosmos = stats.pearsonr(xs[:3], ys[:3])
res["table2_fig6"]["pearson_sft4_yread"] = {"r": round(float(r_all[0]), 3), "p": float(r_all[1]), "n": 4}
res["table2_fig6"]["pearson_sft3_without_cosmos"] = {"r": round(float(r_no_cosmos[0]), 3), "p": float(r_no_cosmos[1]), "n": 3}
res["table2_fig6"]["caveat"] = "n=4, y는 그림에서 눈으로 읽은 근사값. 제로샷 점은 제외. 상관은 Cosmos 한 점에 크게 의존."

# Table 6/7: Pearson r between automatic IF and human IF (reported 0.94, 0.93, 0.96, 1.00 and 0.92, 0.95, 0.97, 0.96)
models = ["Hunyuan", "CogVideoX", "WAN2.1", "Cosmos"]
ifhu_t6_t7 = {  # IF-human as printed in Table 6/7
  "RoboCasa": [81.3, 79.2, 91.7, 93.8],
  "GR1-Object": [52.0, 72.0, 80.0, 84.0],
  "GR1-Behavior_t6": [14.9, 21.3, 74.5, 68.1],
  "GR1-Behavior_t7": [14.9, 21.3, 70.2, 68.1],
  "GR1-Env": [20.0, 30.0, 43.3, 53.3],
}
ifgpt = {"RoboCasa": [68.8, 72.9, 77.1, 79.2], "GR1-Object": [38.0, 72.0, 72.0, 90.0],
         "GR1-Behavior": [38.3, 44.0, 72.3, 59.6], "GR1-Env": [27.6, 55.2, 48.3, 69.0]}
ifqwen = {"RoboCasa": [8.3, 10.4, 18.8, 29.2], "GR1-Object": [26.0, 38.0, 58.0, 62.0],
          "GR1-Behavior": [10.6, 28.0, 55.3, 61.7], "GR1-Env": [27.6, 41.4, 65.5, 65.5]}
hu_table2 = {"RoboCasa": [81.3, 79.2, 91.7, 93.8], "GR1-Object": [52.0, 72.0, 80.0, 84.0],
             "GR1-Behavior": [14.9, 21.3, 74.5, 68.1], "GR1-Env": [43.2, 61.1, 67.4, 53.3]}
corr = {}
for ds in ["RoboCasa", "GR1-Object", "GR1-Behavior", "GR1-Env"]:
    corr[ds] = {"r_IFgpt_vs_hu_table6_7": round(float(stats.pearsonr(ifgpt[ds], ifhu_t6_t7.get(ds, ifhu_t6_t7.get("GR1-Behavior_t6") if ds == "GR1-Behavior" else None))[0]), 3)}
corr["GR1-Behavior"]["r_IFgpt_vs_hu_table6"] = round(float(stats.pearsonr(ifgpt["GR1-Behavior"], ifhu_t6_t7["GR1-Behavior_t6"])[0]), 3)
corr["GR1-Behavior"]["r_IFgpt_vs_hu_table7"] = round(float(stats.pearsonr(ifgpt["GR1-Behavior"], ifhu_t6_t7["GR1-Behavior_t7"])[0]), 3)
corr["GR1-Behavior"].pop("r_IFgpt_vs_hu_table6_7", None)
corr["RoboCasa"]["r_IFqwen_vs_hu"] = round(float(stats.pearsonr(ifqwen["RoboCasa"], ifhu_t6_t7["RoboCasa"])[0]), 3)
corr["GR1-Object"]["r_IFqwen_vs_hu"] = round(float(stats.pearsonr(ifqwen["GR1-Object"], ifhu_t6_t7["GR1-Object"])[0]), 3)
corr["GR1-Env"]["r_IFqwen_vs_hu"] = round(float(stats.pearsonr(ifqwen["GR1-Env"], ifhu_t6_t7["GR1-Env"])[0]), 3)
corr["GR1-Env"]["r_IFqwen_vs_hu"] = round(float(stats.pearsonr(ifqwen["GR1-Env"], ifhu_t6_t7["GR1-Env"])[0]), 3)
corr["GR1-Env"]["r_IFgpt_vs_hu_table2_Hu_column"] = round(float(stats.pearsonr(ifgpt["GR1-Env"], hu_table2["GR1-Env"])[0]), 3)
corr["GR1-Env"]["r_IFgpt_vs_hu_table6_7"] = round(float(stats.pearsonr(ifgpt["GR1-Env"], ifhu_t6_t7["GR1-Env"])[0]), 3)
corr["GR1-Object"]["r_IFgpt_vs_hu_table6"] = round(float(stats.pearsonr(ifgpt["GR1-Object"], ifhu_t6_t7["GR1-Object"])[0]), 3)
corr["RoboCasa"]["r_IFgpt_vs_hu_table6"] = round(float(stats.pearsonr(ifgpt["RoboCasa"], ifhu_t6_t7["RoboCasa"])[0]), 3)
res["table6_7_consistency"] = {"pearson_recomputed": corr,
    "reported_GPT": {"RoboCasa": 0.94, "GR1-Object": 0.93, "GR1-Behavior": 0.96, "GR1-Env": 1.00},
    "reported_Qwen": {"RoboCasa": 0.92, "GR1-Object": 0.95, "GR1-Behavior": 0.97, "GR1-Env": 0.96},
    "table2_Hu_vs_table6_IFhuman_GR1-Env": {"table2_Hu": hu_table2["GR1-Env"], "table6_IFhuman": ifhu_t6_t7["GR1-Env"]},
    "note": "n=4 모델 점 기준 r. Table 2의 GR1-Env Hu 열과 Table 6/7의 IF-human 열이 다름. Table 7의 WAN2.1 GR1-Behavior IF-human(70.2)은 Table 6(74.5)·Table 2(74.5)와 다름."}

# ---------- Table 2 'Hu' column vs text claim (human PA ranking) and Table 2 zero-shot cell check ----------
# Text p.8: "average Pearson correlation of > 90%" -> average of reported 8 values
rep = [0.94, 0.93, 0.96, 1.00, 0.92, 0.95, 0.97, 0.96]
res["text_claim_avg_pearson"] = round(float(np.mean(rep)), 4)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(res, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))[:6000])
