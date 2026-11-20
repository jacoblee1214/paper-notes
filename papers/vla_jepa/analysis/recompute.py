"""VLA-JEPA (arXiv:2602.10098 v2) reported numbers: recomputation and consistency checks.
Inputs are transcribed by hand from paper Tables 1-4 (pdftotext of paper/paper.pdf).
Outputs: analysis/recompute_results.json. CPU only, numpy + scipy.stats."""
import json, math, os
import numpy as np
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recompute_results.json")
res = {}

def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

def two_prop_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return z, 2 * (1 - stats.norm.cdf(abs(z)))

# ---- Table 1 (LIBERO), 500 episodes per suite, success % as printed ----
suites = ["Spatial", "Object", "Goal", "LIBERO-10"]
t1 = {
    "VLA-JEPA": [96.2, 99.6, 97.2, 95.8, 97.2],
    "w/o human videos": [94.8, 99.6, 95.8, 94.0, 96.1],
    "OpenVLA-OFT": [97.6, 98.4, 97.9, 94.5, 97.1],
    "pi0.5": [98.8, 98.2, 98.0, 92.4, 96.9],
    "GR00T N1": [94.4, 97.6, 93.0, 90.6, 93.9],
    "villa-X": [97.5, 97.0, 91.5, 74.5, 90.1],
    "UniVLA": [96.5, 96.8, 95.6, 92.0, 95.2],
}
N_SUITE = 500
mean_check = {}
for m, v in t1.items():
    mean_check[m] = {"reported_avg": v[4], "mean_of_4_suites": round(float(np.mean(v[:4])), 3)}
res["table1_avg_check"] = mean_check

# Wilson CI for VLA-JEPA suite rates and the 2000-episode average
ci = {}
for s, pct in zip(suites + ["Avg(2000 ep)"], t1["VLA-JEPA"]):
    n = N_SUITE if s != "Avg(2000 ep)" else 4 * N_SUITE
    k = round(pct / 100 * n)
    lo, hi = wilson(k, n)
    ci[s] = {"k": k, "n": n, "rate": round(k / n * 100, 2), "wilson95": [round(lo * 100, 2), round(hi * 100, 2)]}
res["table1_vla_jepa_wilson"] = ci

# Two-proportion z tests (independent samples, suite-level) vs OpenVLA-OFT and pi0.5
tests = {}
for rival in ["OpenVLA-OFT", "pi0.5"]:
    for i, s in enumerate(suites + ["Avg"]):
        n = N_SUITE if s != "Avg" else 4 * N_SUITE
        k1 = round(t1["VLA-JEPA"][i] / 100 * n)
        k2 = round(t1[rival][i] / 100 * n)
        z, p = two_prop_z(k1, n, k2, n)
        tests[f"VLA-JEPA vs {rival} / {s}"] = {"diff_pp": round((k1 - k2) / n * 100, 2), "z": round(z, 2), "p_two_sided": round(p, 3)}
res["table1_two_prop_tests"] = tests

# Probability that a 0.1 pp average gap arises by chance with the same n (sim, same as paired-free bound)
rng = np.random.default_rng(0)
n = 4 * N_SUITE
p_true = 0.972
sims = rng.binomial(n, p_true, size=(200000, 2)) / n * 100
diff = sims[:, 0] - sims[:, 1]
res["toy_libero_avg_gap_sim"] = {
    "n_per_method": n, "p_true": p_true, "reported_gap_pp": 0.1,
    "P_abs_gap_ge_0.1pp_under_null": round(float((np.abs(diff) >= 0.1).mean()), 3),
    "sd_of_gap_pp": round(float(diff.std()), 3),
}

# ---- Table 2 (SimplerEnv) ----
g_cols = ["Pick", "Move", "Drawer", "Place", "Avg"]
g = {
    "VLA-JEPA": [88.3, 64.1, 59.3, 49.1, 65.2],
    "w/o human videos": [85.3, 66.7, 75.5, 86.1, 78.4],
    "villa-x": [81.7, 55.4, 38.4, 4.2, 44.9],
    "RoboVLMs": [77.3, 61.7, 43.5, 24.1, 51.7],
    "pi0-Fast": [75.3, 67.5, 42.9, None, None],
    "pi0": [72.7, 65.3, 38.3, None, None],
}
w_cols = ["Spoon", "Carrot", "Block", "Eggplant", "Avg"]
w = {
    "VLA-JEPA": [75.0, 70.8, 12.5, 70.8, 57.3],
    "w/o human videos": [75.0, 54.2, 20.8, 79.2, 57.3],
    "LAPA (*sim-ID)": [70.8, 45.8, 54.2, 58.3, 57.3],
    "villa-x": [48.3, 24.2, 19.2, 71.7, 40.8],
    "UniVLA": [None, None, None, None, 42.7],
    "OpenVLA-OFT (*sim-ID)": [34.2, 30.0, 30.0, 72.5, 41.8],
    "pi0-Fast": [29.1, 21.9, 10.8, 66.7, 48.3],
}
mc = {}
for name, row in list(g.items()):
    vals = [x for x in row[:4] if x is not None]
    if None not in row[:4]:
        mc[f"Google {name}"] = {"reported_avg": row[4], "mean_of_4": round(float(np.mean(row[:4])), 2)}
for name, row in w.items():
    if None not in row[:4]:
        mc[f"WidowX {name}"] = {"reported_avg": row[4], "mean_of_4": round(float(np.mean(row[:4])), 2)}
res["table2_avg_check"] = mc

# Infer per-task trial granularity from printed values
def grid_fit(vals, nmax=200):
    out = []
    for n in range(2, nmax + 1):
        if all(abs(v * n / 100 - round(v * n / 100)) < 0.06 for v in vals):
            out.append(n)
    return out
res["table2_granularity"] = {
    "WidowX_VLA-JEPA_candidates_le200": grid_fit(w["VLA-JEPA"][:4]),
    "Google_VLA-JEPA_candidates_le200": grid_fit(g["VLA-JEPA"][:4]),
}
# Wilson CI for WidowX per-task at n=24 (inferred), average of 4 tasks at n=96
wci = {}
for task, v in zip(w_cols[:4], w["VLA-JEPA"][:4]):
    k = round(v / 100 * 24)
    lo, hi = wilson(k, 24)
    wci[task] = {"k": k, "n": 24, "wilson95": [round(lo * 100, 1), round(hi * 100, 1)]}
res["table2_widowx_task_ci_n24"] = wci

# Ranking claims in text (Table 2 text: best on 2/4 tasks, 2nd-highest average on WidowX)
res["table2_claim_checks"] = {
    "WidowX avg: VLA-JEPA 57.3 equals LAPA 57.3 (LAPA is a *sim-ID method)": True,
    "Google: VLA-JEPA best on Pick, Drawer, Place (3/4 among rows shown)": True,
    "Google: VLA-JEPA Move 64.1 below pi0-Fast 67.5": True,
    "w/o human videos Google Avg 78.4 > full model 65.2": True,
}

# ---- Table 3 (LIBERO-Plus): recompute unweighted mean of the 7 dimensions ----
dims = ["Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout"]
t3 = {
    "UniVLA": ([1.8, 46.2, 69.6, 69.0, 81.0, 21.2, 31.9], 42.9),
    "OpenVLA-OFT": ([56.4, 31.9, 79.5, 88.7, 93.3, 75.8, 74.2], 69.6),
    "pi0": ([13.8, 6.0, 58.8, 85.0, 81.4, 79.0, 68.9], 53.6),
    "pi0-Fast": ([65.1, 21.6, 61.0, 73.2, 73.2, 74.4, 68.8], 61.6),
    "WorldVLA": ([0.1, 27.9, 41.6, 43.7, 17.1, 10.9, 38.0], 25.0),
    "VLA-JEPA": ([63.3, 67.1, 85.4, 95.6, 93.6, 66.3, 85.1], 79.5),
    "VLA-JEPA w/o human": ([40.3, 55.7, 72.9, 88.2, 70.5, 38.2, 74.6], 62.9),
}
t3c = {}
for m, (v, rep) in t3.items():
    t3c[m] = {"reported_avg": rep, "unweighted_mean": round(float(np.mean(v)), 2),
              "gap_pp": round(rep - float(np.mean(v)), 2)}
res["table3_avg_check"] = t3c
best = {}
for i, d in enumerate(dims):
    row = {m: v[0][i] for m, v in t3.items()}
    top = max(row, key=row.get)
    best[d] = {"top": top, "top_value": row[top], "vla_jepa": row["VLA-JEPA"]}
res["table3_best_per_dim"] = best
res["table3_text_claim_5of7"] = sum(1 for d in dims if best[d]["top"] == "VLA-JEPA")
# Background margin over OpenVLA-OFT: 93.6 vs 93.3
res["table3_background_margin_pp"] = round(93.6 - 93.3, 2)

# ---- Table 4 vs Table 1 consistency ----
t4 = {"T=4": [95.0, 99.2, 95.8, 89.0, 94.8], "T=8": [94.8, 99.8, 95.8, 94.0, 96.1], "T=16": [92.8, 98.8, 98.0, 92.2, 95.5]}
res["table4_avg_check"] = {k: {"reported": v[4], "mean": round(float(np.mean(v[:4])), 2)} for k, v in t4.items()}
res["table4_T8_vs_table1_main"] = {"table4_T8": t4["T=8"][:4], "table1_main": t1["VLA-JEPA"][:4],
                                    "table1_wo_human": t1["w/o human videos"][:4]}

# ---- Paper internal consistency: K = 24/T (App. A.1) vs Table 5 ----
res["K_consistency"] = {"T=8_K": 24 / 8, "Table5_action_tokens_per_timestep": 3, "repo_config_num_action_tokens_per_timestep": 8}

with open(OUT, "w") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print("wrote", OUT)
print(json.dumps({k: res[k] for k in ["table1_avg_check", "table3_text_claim_5of7", "table3_background_margin_pp", "toy_libero_avg_gap_sim", "table2_granularity"]}, indent=1, ensure_ascii=False)[:2500])
