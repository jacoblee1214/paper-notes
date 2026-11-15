"""Recompute reported numbers from tables in arXiv:2609.14156v2 (CPU, numpy/scipy only)."""
import json, math, os
import numpy as np
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "table_check_results.json")
R = {}

# --- LIBERO BC-Transformer, 2-view, per-seed tables (App. C.5) ---
suites = ["OBJECT", "GOAL", "SPATIAL", "LONG"]
seed = {
    "plain2v": np.array([[84.5, 73.5, 90.5], [90.0, 89.5, 81.0], [90.5, 82.0, 82.0], [50.5, 52.5, 55.5]]),
    "proprio2v": np.array([[82.5, 77.5, 80.0], [84.0, 75.5, 85.0], [82.0, 79.0, 79.0], [50.0, 38.5, 40.0]]),
    "acro2v": np.array([[99.0, 99.0, 96.5], [96.0, 92.5, 90.0], [96.0, 91.5, 89.5], [84.0, 86.0, 89.5]]),
}
reported_mean = {"plain2v": [82.8, 86.8, 84.8, 52.8], "proprio2v": [80.0, 81.5, 80.0, 42.8], "acro2v": [98.2, 92.8, 92.3, 86.5]}
reported_sd = {"plain2v": [7.0, 4.1, 4.0, 2.1], "proprio2v": [2.0, 4.3, 1.4, 5.1], "acro2v": [1.2, 2.5, 2.7, 2.3]}
R["seed_means_match"] = {}
R["sd_convention"] = {}
for k, v in seed.items():
    m = v.mean(axis=1)
    sd0 = v.std(axis=1, ddof=0)
    sd1 = v.std(axis=1, ddof=1)
    R["seed_means_match"][k] = bool(np.allclose(np.round(m, 1), reported_mean[k], atol=0.06))
    R["sd_convention"][k] = {
        "reported_sd": reported_sd[k],
        "ddof0": np.round(sd0, 1).tolist(),
        "ddof1": np.round(sd1, 1).tolist(),
        "matches_ddof0": bool(np.allclose(np.round(sd0, 1), reported_sd[k], atol=0.11)),
        "matches_ddof1": bool(np.allclose(np.round(sd1, 1), reported_sd[k], atol=0.11)),
    }

# Suite-level averages and deltas (2-view)
avg = {k: v.mean(axis=1).mean() for k, v in seed.items()}  # mean of suite means
R["avg_2v"] = {k: round(float(x), 2) for k, x in avg.items()}
R["delta_acro_minus_plain_2v"] = round(float(avg["acro2v"] - avg["plain2v"]), 2)
R["delta_acro_minus_proprio_2v"] = round(float(avg["acro2v"] - avg["proprio2v"]), 2)
R["delta_proprio_minus_plain_2v"] = round(float(avg["proprio2v"] - avg["plain2v"]), 2)

# Welch t and Welch-Satterthwaite df per suite (n=3 seeds), and an overall test on the 4 suite means
def welch(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    se = math.sqrt(va + vb)
    d = a.mean() - b.mean()
    tc = stats.t.ppf(0.975, df)
    return dict(diff=round(float(d), 2), ci95=[round(float(d - tc * se), 2), round(float(d + tc * se), 2)],
                df=round(float(df), 2), p=round(float(p), 4))

R["welch_acro_vs_plain_2v"] = {s: welch(seed["acro2v"][i], seed["plain2v"][i]) for i, s in enumerate(suites)}
R["welch_acro_vs_proprio_2v"] = {s: welch(seed["acro2v"][i], seed["proprio2v"][i]) for i, s in enumerate(suites)}
# Paired by seed (seed IDs are shared across conditions in the table): per-seed difference
dpl = seed["acro2v"] - seed["plain2v"]
R["paired_seed_diff_acro_minus_plain_2v"] = {
    "per_seed_avg_over_suites": np.round(dpl.mean(axis=0), 2).tolist(),
    "mean": round(float(dpl.mean()), 2),
}
tp = stats.ttest_rel(dpl.mean(axis=0), np.zeros(3))
R["paired_seed_diff_acro_minus_plain_2v"]["one_sample_t_over_3_seeds_p"] = round(float(tp.pvalue), 4)

# Task-level Pearson (headroom) from summary is not recoverable; illustrate mechanical correlation in toy (see toy file)

# --- Real-world pi0.5 (Table 3, 30 trials per cell, single run per condition) ---
rw = {  # subtask order: Lift-Pick, Tube-Pick, Tube-Insert, Dish-Pull, Dish-Pick, Dish-Place, Charger-Pick, Charger-Insert
    "Baseline": [20, 5, 0, 21, 11, 11, 9, 0],
    "Tac-View": [16, 9, 0, 22, 18, 16, 17, 0],
    "Position-Only": [22, 21, 7, 17, 14, 11, 9, 0],
    "Binbars": [23, 10, 1, 24, 15, 15, 9, 0],
    "Binary Contact (1x)": [19, 17, 2, 13, 5, 4, 16, 0],
    "Avg. Arrow": [23, 17, 3, 23, 14, 12, 13, 0],
    "Binary Contact (9x)": [9, 6, 0, 25, 13, 12, 11, 0],
    "Multi-arrows": [24, 24, 16, 28, 22, 21, 18, 2],
}
tasks = {"Lift": [0], "Tube": [1, 2], "Dishwasher": [3, 4, 5], "Charger": [6, 7]}
N = 30
base = np.array(rw["Baseline"])
multi = np.array(rw["Multi-arrows"])
task_diffs = []
for t, idx in tasks.items():
    task_diffs.append(float(np.mean(multi[idx] / N - base[idx] / N) * 100))
R["realworld_macro_multi_minus_baseline_pp"] = round(float(np.mean(task_diffs)), 2)
R["realworld_task_diffs_pp"] = {t: round(d, 1) for t, d in zip(tasks, task_diffs)}
# End-to-end (final stage per task: Lift-Pick, Tube-Insert, Dish-Place, Charger-Insert)
final = [0, 2, 5, 7]
R["realworld_final_stage_success"] = {
    k: {"per_task": [v[i] for i in final], "macro_pct": round(float(np.mean([v[i] / N for i in final]) * 100), 1)}
    for k, v in rw.items()}
R["realworld_final_stage_macro_diff_pp"] = round(
    R["realworld_final_stage_success"]["Multi-arrows"]["macro_pct"] - R["realworld_final_stage_success"]["Baseline"]["macro_pct"], 1)

# Fisher exact one-sided and two-sided for each sub-task Multi vs Baseline, and Multi vs Tac-View
subtasks = ["Lift-Pick", "Tube-Pick", "Tube-Insert", "Dish-Pull", "Dish-Pick", "Dish-Place", "Charger-Pick", "Charger-Insert"]
fis = {}
for j, s in enumerate(subtasks):
    a, b = rw["Multi-arrows"][j], rw["Baseline"][j]
    _, p2 = stats.fisher_exact([[a, N - a], [b, N - b]])
    fis[s] = {"multi": a, "baseline": b, "p_two_sided": round(float(p2), 4)}
R["fisher_multi_vs_baseline"] = fis
# Holm across 8 sub-tasks
ps = sorted([(v["p_two_sided"], k) for k, v in fis.items()])
m = len(ps); holm = {}; running = 0
for i, (p, k) in enumerate(ps):
    adj = min(1.0, (m - i) * p); running = max(running, adj); holm[k] = round(running, 4)
R["holm_multi_vs_baseline"] = holm

# Pooled trial-level counts: Multi vs Baseline over all 8 sub-task cells (not independent: same trials)
pool_m = int(multi.sum()); pool_b = int(base.sum())
_, p_pool = stats.fisher_exact([[pool_m, 8 * N - pool_m], [pool_b, 8 * N - pool_b]])
R["pooled_subtask_counts"] = {"multi": pool_m, "baseline": pool_b, "of": 8 * N, "fisher_p_two_sided_NOT_independent": round(float(p_pool), 6)}

# Tac-View vs Multi
tv = np.array(rw["Tac-View"])
R["multi_vs_tacview_subtask_wins"] = int(np.sum(multi > tv)), int(np.sum(multi == tv)), int(np.sum(multi < tv))

# Trial-count sanity: episode counts
R["demo_counts"] = {"cube": 100, "tube": 97, "charger": 100, "dishwasher": 100, "sum": 397, "stated_collected": 400}

with open(OUT, "w") as f:
    json.dump(R, f, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(R, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))[:6000])
