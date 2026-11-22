"""Recompute the numbers reported in arXiv:2608.00337v1 (tables, figure labels, text claims).

Usage: python3 recompute.py        (needs numpy, scipy; reads paper_tables.py and fig4_extracted.json)
Writes recompute_results.json. Check labels V1... refer to analysis/03_analysis.md.
"""
import json
import math
import pathlib

import numpy as np
from scipy import stats

from paper_tables import (N_FAST, N_SLOW, SEEDS_REAL, T1_H100, T1_L40S, T2_GR00T, T3, T3_N, T4, T4_SCHED, T5)

HERE = pathlib.Path(__file__).parent
R = {}


def rnd(x, k=3):
    return None if x is None else round(float(x), k)


# ---------------------------------------------------------------- V1 Table 5 arithmetic
v1 = []
for (cfg, sch), (tf, ts, tot, avg, sf, ss) in T5.items():
    nf, ns = N_FAST[cfg], N_SLOW[cfg]
    tot_calc = nf * tf + ns * (ts or 0)
    avg_calc = (nf * sf + ns * (ss or 0)) / (nf + ns)
    # integer feasibility: a per-robot mean over 3 seeds and nf robots is k/(3*nf)
    fast_cnt = tf * nf * SEEDS_REAL
    slow_cnt = (ts or 0) * ns * SEEDS_REAL
    tot_cnt = tot * SEEDS_REAL
    v1.append({
        "cfg": cfg, "sched": sch, "total_reported": tot, "total_from_tiers": rnd(tot_calc, 2),
        "total_diff": rnd(tot - tot_calc, 2), "avg_starv_reported": avg, "avg_starv_from_tiers": rnd(avg_calc, 2),
        "avg_starv_diff": rnd(avg - avg_calc, 2),
        "fast_count_3seeds": rnd(fast_cnt, 2), "slow_count_3seeds": rnd(slow_cnt, 2), "total_count_3seeds": rnd(tot_cnt, 2),
        # per-robot means are printed with 2 decimals -> rounding slack 0.005 * robots * seeds
        "total_count_integer": abs(tot_cnt - round(tot_cnt)) <= 0.005 * SEEDS_REAL + 0.01,
        "fast_count_integer": abs(fast_cnt - round(fast_cnt)) <= 0.005 * nf * SEEDS_REAL + 0.01,
        "slow_count_integer": abs(slow_cnt - round(slow_cnt)) <= 0.005 * max(ns, 1) * SEEDS_REAL + 0.01,
        "int_counts_sum_to_total": (round(fast_cnt) + round(slow_cnt)) == round(tot_cnt),
    })
R["V1_table5_arithmetic"] = v1

# ---------------------------------------------------------------- V2 Fig 4 vs Table 5
fig4 = json.load(open(HERE / "fig4_extracted.json"))
order = [("all_fast", "EDF"), ("all_fast", "RR"), ("all_fast", "LA w=1"),
         ("one_fast", "EDF"), ("one_fast", "RR"), ("one_fast", "LA w=1"), ("one_fast", "LA w=5"),
         ("half_fast", "EDF"), ("half_fast", "RR"), ("half_fast", "LA w=1"), ("half_fast", "LA w=5")]
v2 = []
for row, key in zip(fig4, order):
    t5 = T5[key]
    v2.append({"cell": f"{key[0]} {key[1]}", "fig4_total": row["total_from_bars"], "table5_total": t5[2],
               "diff": rnd(row["total_from_bars"] - t5[2], 2), "err_halfwidth": row["err_halfwidth"]})
R["V2_fig4_vs_table5"] = v2
R["V2_max_abs_diff"] = max(abs(x["diff"]) for x in v2)

# ---------------------------------------------------------------- V3 headline +18 %
def tot(cfg, s):
    return T5[(cfg, s)][2]


v3 = {
    "LA5_vs_EDF_one_fast_%": rnd((tot("one_fast", "LA w=5") / tot("one_fast", "EDF") - 1) * 100, 2),
    "LA5_vs_RR_one_fast_%": rnd((tot("one_fast", "LA w=5") / tot("one_fast", "RR") - 1) * 100, 2),
    "LA5_vs_LA1_one_fast_%": rnd((tot("one_fast", "LA w=5") / tot("one_fast", "LA w=1") - 1) * 100, 2),
    "LA1_vs_EDF_one_fast_%": rnd((tot("one_fast", "LA w=1") / tot("one_fast", "EDF") - 1) * 100, 2),
    "LA1_vs_EDF_all_fast_%": rnd((tot("all_fast", "LA w=1") / tot("all_fast", "EDF") - 1) * 100, 2),
    "LA1_vs_RR_all_fast_%": rnd((tot("all_fast", "LA w=1") / tot("all_fast", "RR") - 1) * 100, 2),
    "LA1_vs_EDF_half_fast_%": rnd((tot("half_fast", "LA w=1") / tot("half_fast", "EDF") - 1) * 100, 2),
    "LA5_vs_EDF_half_fast_%": rnd((tot("half_fast", "LA w=5") / tot("half_fast", "EDF") - 1) * 100, 2),
    "LA5_vs_RR_half_fast_%": rnd((tot("half_fast", "LA w=5") / tot("half_fast", "RR") - 1) * 100, 2),
    "fast_tier_LA5_over_RR_one_fast": rnd(T5[("one_fast", "LA w=5")][0] / T5[("one_fast", "RR")][0], 2),
    "fast_tier_LA5_over_EDF_one_fast": rnd(T5[("one_fast", "LA w=5")][0] / T5[("one_fast", "EDF")][0], 2),
    "all_fast_starv_increase_pp_LA1_vs_EDF": rnd(T5[("all_fast", "LA w=1")][3] - T5[("all_fast", "EDF")][3], 2),
}
# who contributes to the +18 %? counts over 3 seeds
c = lambda cfg, s, i, n: round((T5[(cfg, s)][i]) * n * SEEDS_REAL)
dec = {}
for s in ["EDF", "RR", "LA w=1", "LA w=5"]:
    dec[s] = {"fast_count": c("one_fast", s, 0, 1), "slow_count": c("one_fast", s, 1, 9)}
base = dec["EDF"]
dec["LA w=5 minus EDF"] = {"fast": dec["LA w=5"]["fast_count"] - base["fast_count"],
                           "slow": dec["LA w=5"]["slow_count"] - base["slow_count"]}
dec["LA w=1 minus EDF"] = {"fast": dec["LA w=1"]["fast_count"] - base["fast_count"],
                           "slow": dec["LA w=1"]["slow_count"] - base["slow_count"]}
dec["LA w=5 minus LA w=1"] = {"fast": dec["LA w=5"]["fast_count"] - dec["LA w=1"]["fast_count"],
                              "slow": dec["LA w=5"]["slow_count"] - dec["LA w=1"]["slow_count"]}
R["V3_headline"] = v3
R["V3_one_fast_count_decomposition_3seeds_pieces"] = dec

# ---------------------------------------------------------------- V4 significance
# (a) Poisson / conditional binomial on reconstructed integer totals over 3 x 60 s fleet-minutes
def binom_p(a, b):
    return float(stats.binomtest(a, a + b, 0.5).pvalue)


counts = {k: round(v[2] * SEEDS_REAL) for k, v in T5.items()}
v4a = {}
for cfg, base_s, new_s in [("one_fast", "EDF", "LA w=5"), ("one_fast", "RR", "LA w=5"), ("one_fast", "EDF", "LA w=1"),
                           ("one_fast", "LA w=1", "LA w=5"), ("all_fast", "EDF", "LA w=1"), ("all_fast", "RR", "LA w=1"),
                           ("half_fast", "EDF", "LA w=5"), ("half_fast", "RR", "LA w=1")]:
    a, b = counts[(cfg, new_s)], counts[(cfg, base_s)]
    v4a[f"{cfg}: {new_s} vs {base_s}"] = {"counts": [a, b], "ratio": rnd(a / b, 3), "p_poisson_exact": rnd(binom_p(a, b), 4)}
# fast tier of the single fast robot (one-fast): pieces over 3 x 60 s
fc_ = {s_: round(T5[("one_fast", s_)][0] * SEEDS_REAL) for s_ in ["EDF", "RR", "LA w=1", "LA w=5"]}
v4a["one_fast fast-robot: LA w=5 vs EDF"] = {"counts": [fc_["LA w=5"], fc_["EDF"]], "p_poisson_exact": rnd(binom_p(fc_["LA w=5"], fc_["EDF"]), 4)}
v4a["one_fast fast-robot: LA w=5 vs RR"] = {"counts": [fc_["LA w=5"], fc_["RR"]], "p_poisson_exact": rnd(binom_p(fc_["LA w=5"], fc_["RR"]), 4)}
# slow tier one-fast
sc_ = {s_: round(T5[("one_fast", s_)][1] * 9 * SEEDS_REAL) for s_ in ["EDF", "RR", "LA w=1", "LA w=5"]}
v4a["one_fast slow-tier: LA w=1 vs EDF"] = {"counts": [sc_["LA w=1"], sc_["EDF"]], "ratio": rnd(sc_["LA w=1"] / sc_["EDF"], 3), "p_poisson_exact": rnd(binom_p(sc_["LA w=1"], sc_["EDF"]), 4)}
hs_ = {s_: round(T5[("half_fast", s_)][1] * 5 * SEEDS_REAL) for s_ in ["EDF", "RR", "LA w=1", "LA w=5"]}
hf_ = {s_: round(T5[("half_fast", s_)][0] * 5 * SEEDS_REAL) for s_ in ["EDF", "RR", "LA w=1", "LA w=5"]}
v4a["half_fast fast-tier: LA w=5 vs EDF"] = {"counts": [hf_["LA w=5"], hf_["EDF"]], "ratio": rnd(hf_["LA w=5"] / hf_["EDF"], 3), "p_poisson_exact": rnd(binom_p(hf_["LA w=5"], hf_["EDF"]), 4)}
v4a["half_fast slow-tier: LA w=5 vs EDF"] = {"counts": [hs_["LA w=5"], hs_["EDF"]], "ratio": rnd(hs_["LA w=5"] / hs_["EDF"], 3), "p_poisson_exact": rnd(binom_p(hs_["LA w=5"], hs_["EDF"]), 4)}
R["V4a_poisson_exact_on_totals"] = v4a

# (b) Welch t from the Fig. 4 error bars; the caption does not say what they are -> three readings
hw = {k: v["err_halfwidth"] for k, v in zip(order, fig4)}
n = SEEDS_REAL


def welch(m1, e1, m2, e2, reading):
    if reading == "SD":
        s1, s2 = e1, e2
    elif reading == "SEM":
        s1, s2 = e1 * math.sqrt(n), e2 * math.sqrt(n)
    else:  # 95% CI half-width with t(2)=4.303
        s1, s2 = e1 / stats.t.ppf(0.975, n - 1) * math.sqrt(n), e2 / stats.t.ppf(0.975, n - 1) * math.sqrt(n)
    v1_, v2_ = s1 ** 2 / n, s2 ** 2 / n
    t = (m1 - m2) / math.sqrt(v1_ + v2_)
    df = (v1_ + v2_) ** 2 / (v1_ ** 2 / (n - 1) + v2_ ** 2 / (n - 1))
    return {"t": rnd(t, 2), "df": rnd(df, 1), "p": rnd(2 * stats.t.sf(abs(t), df), 4), "sd_a": rnd(s1, 2), "sd_b": rnd(s2, 2)}


v4b = {}
for cfg, base_s, new_s in [("one_fast", "EDF", "LA w=5"), ("one_fast", "RR", "LA w=5"), ("all_fast", "EDF", "LA w=1"),
                           ("all_fast", "RR", "LA w=1"), ("half_fast", "EDF", "LA w=5"), ("one_fast", "LA w=1", "LA w=5")]:
    m1, m2 = T5[(cfg, new_s)][2], T5[(cfg, base_s)][2]
    e1, e2 = hw[(cfg, new_s)], hw[(cfg, base_s)]
    v4b[f"{cfg}: {new_s} vs {base_s}"] = {r: welch(m1, e1, m2, e2, r) for r in ["SD", "SEM", "95CI"]}
R["V4b_welch_from_fig4_errorbars"] = v4b

# (c) the multiple-look problem: best of three LA weights, three configurations
R["V4c_note"] = ("headline = max over {one_fast LA@5, half_fast LA@5, all_fast LA@1} vs {EDF, RR}; "
                 "only one of three comparisons shows ~+18 %, n=3 seeds per cell")

# ---------------------------------------------------------------- V5 simulation (Table 4)
def t4(cfg, N, sched, i):
    return T4[cfg][N][T4_SCHED.index(sched)][i]


# noise estimate: in the all-fast fleet LA@1/3/5 are the same algorithm (a uniform weight scales every score)
noise = {}
sds = []
for N in T4["all_fast"]:
    th = [t4("all_fast", N, s, 0) for s in ["la1", "la3", "la5"]]
    st = [t4("all_fast", N, s, 1) for s in ["la1", "la3", "la5"]]
    noise[N] = {"thr": th, "thr_sd": rnd(np.std(th, ddof=1), 3), "thr_rel_%": rnd(np.std(th, ddof=1) / np.mean(th) * 100, 2),
                "starv": st, "starv_sd": rnd(np.std(st, ddof=1), 3)}
    sds.append((np.var(th, ddof=1), np.mean(th)))
pooled_rel = math.sqrt(np.mean([v / m ** 2 for v, m in sds]))
R["V5a_noise_from_weight_invariant_all_fast"] = {"per_N": noise, "pooled_relative_sd_of_throughput_%": rnd(pooled_rel * 100, 2)}

rel = {}
for cfg in ["one_fast", "half_fast"]:
    rel[cfg] = {}
    for N in T4[cfg]:
        base = np.mean([t4(cfg, N, "rr", 0), t4(cfg, N, "edf", 0)])
        row = {}
        for s in ["la1", "la3", "la5"]:
            d = t4(cfg, N, s, 0) / base - 1
            z = d / (pooled_rel * math.sqrt(2))
            row[s] = {"vs_mean(RR,EDF)_%": rnd(d * 100, 1), "z_vs_noise": rnd(z, 1)}
        rel[cfg][N] = row
R["V5b_total_throughput_relative_to_baselines"] = rel
R["V5b_note"] = ("claim A.7: 'at b=3 all schedulers achieve comparable total throughput' holds for LA@1 and mostly LA@3; "
                 "it fails for LA@5 in half-fast (N>=4) and is -6% in one-fast N=10")

# throughput vs starvation: within each (config, N) cell, rank correlation across the five schedulers
rhos = []
for cfg in T4:
    for N in T4[cfg]:
        th = [T4[cfg][N][i][0] for i in range(5)]
        sv = [T4[cfg][N][i][1] for i in range(5)]
        if len(set(sv)) > 1:
            rhos.append(stats.spearmanr(th, sv)[0])
R["V5c_within_cell_spearman_throughput_vs_starvation"] = {"mean_rho": rnd(np.mean(rhos), 3), "cells": len(rhos),
                                                          "share_negative_%": rnd(np.mean(np.array(rhos) < 0) * 100, 1)}

# ---------------------------------------------------------------- V6 Table 3 vs Table 4, depth ablation
cells = []
mono = 0
tot_cells = 0
dif34 = []
for cfg in T3:
    for k, N in enumerate(T3_N):
        row = T3[cfg][k]
        for w, base in enumerate([0, 3, 6]):
            L1, L2, L3 = row[base:base + 3]
            tot_cells += 1
            if L3 > L1:
                mono += 1
            ref = t4(cfg, N, ["la1", "la3", "la5"][w], 1)
            dif34.append(L1 - ref)
R["V6_table3"] = {
    "cells": tot_cells, "L3_gt_L1": mono, "share_%": rnd(mono / tot_cells * 100, 1),
    "mean_L3_minus_L1_pp": rnd(np.mean([T3[c][k][b + 2] - T3[c][k][b] for c in T3 for k in range(5) for b in (0, 3, 6)]), 3),
    "mean_L3_minus_L1_pp_N>=6": rnd(np.mean([T3[c][k][b + 2] - T3[c][k][b] for c in T3 for k in range(2, 5) for b in (0, 3, 6)]), 3),
    "T3(L=1)_minus_T4_starv_pp_mean": rnd(np.mean(dif34), 3), "T3(L=1)_minus_T4_starv_pp_absmean": rnd(np.mean(np.abs(dif34)), 3),
    "T3(L=1)_minus_T4_starv_pp_max_abs": rnd(np.max(np.abs(dif34)), 2),
    "all_fast_N10_L3_minus_L1": [rnd(T3["all_fast"][4][b + 2] - T3["all_fast"][4][b], 2) for b in (0, 3, 6)],
}

# ---------------------------------------------------------------- V7 latency tables
b = np.arange(1, 6)
out7 = {}
for name, arr in [("L40S_pi05", T1_L40S), ("H100_pi05", T1_H100), ("L40S_GR00T_N1.7", T2_GR00T)]:
    a = np.array(arr)
    slope, icpt = np.polyfit(b, a, 1)
    per_robot = a / b
    out7[name] = {"slope_ms_per_robot": rnd(slope, 2), "intercept_ms": rnd(icpt, 2),
                  "intercept_share_of_b1_%": rnd(icpt / a[0] * 100, 1),
                  "per_robot_ms": [rnd(x, 1) for x in per_robot],
                  "capacity_req_per_s": [rnd(1000 * k / x, 1) for k, x in zip(b, a)],
                  "amortization_b3_vs_b5_of_total_gain_%": rnd((per_robot[0] - per_robot[2]) / (per_robot[0] - per_robot[4]) * 100, 1),
                  "max_resid_ms": rnd(np.max(np.abs(np.polyval([slope, icpt], b) - a)), 2)}
out7["text_claims"] = {"b5_vs_b1_L40S": "211.1 vs 73.0 (A.4)", "ratio_b5_over_b1": rnd(T1_L40S[4] / T1_L40S[0], 2),
                       "H100_speedup_b1..5": [rnd(x / y, 2) for x, y in zip(T1_L40S, T1_H100)]}
R["V7_latency_tables"] = out7

# ---------------------------------------------------------------- V8 fluid capacity model
def lat_ms(bs):
    slope, icpt = np.polyfit(b, np.array(T1_L40S), 1)
    return icpt + slope * bs


fluid = {}
# simulation: fc=20, Hmax fast 6 / slow 10, cap b=3
cap3 = 1000 * 3 / T1_L40S[2]
for cfg in ["all_fast", "one_fast", "half_fast"]:
    row = {}
    for N in [2, 4, 6, 8, 10]:
        nf = {"all_fast": N, "one_fast": 1, "half_fast": N // 2}[cfg]
        ns = N - nf
        demand = nf * 20 / 6 + ns * 20 / 10   # requests/s if every chunk is used for Hmax steps
        util = demand / cap3
        fluid_starv = max(0.0, 1 - 1 / util) * 100 if util > 1 else 0.0
        row[N] = {"demand_req_s": rnd(demand, 1), "utilization": rnd(util, 2), "fluid_starvation_%": rnd(fluid_starv, 1),
                  "paper_EDF_starv_%": t4(cfg, N, "edf", 1)}
    fluid[f"sim_{cfg}"] = row
# real world: fc=30, Hmax fast 10 / slow 20, b=3 (all-fast) or 5
real = {}
for cfg, nf, ns, bcap in [("all_fast", 10, 0, 3), ("one_fast", 1, 9, 5), ("half_fast", 5, 5, 5)]:
    cap = 1000 * bcap / T1_L40S[bcap - 1]
    demand = nf * 30 / 10 + ns * 30 / 20
    util = demand / cap
    real[cfg] = {"demand_req_s": rnd(demand, 1), "capacity_req_s": rnd(cap, 1), "utilization": rnd(util, 2),
                 "fluid_starvation_%": rnd(max(0, 1 - 1 / util) * 100, 1),
                 "paper_EDF_avg_starv_%": T5[(cfg, "EDF")][3]}
fluid["real"] = real
R["V8_fluid_capacity_model"] = fluid

# ---------------------------------------------------------------- V9 small text/figure checks
R["V9_text_checks"] = {
    "A4_fig7_legend_means": {"RR": 4.6, "EDF": 4.7, "LA": 3.3}, "A4_text_says": "EDF and Round-Robin ... (mean 4.7 robots per call)",
    "A8_all_fast_LA_gain_over_EDF_%": v3["LA1_vs_EDF_all_fast_%"], "A8_text": "more than 7%", "A8_starv_pp": v3["all_fast_starv_increase_pp_LA1_vs_EDF"],
    "fig4_caption_fast_tier_2x": "almost 2x over EDF (bars 8 vs 4); Table 5 ratio = %s" % v3["fast_tier_LA5_over_EDF_one_fast"],
    "table2_name": "GR00T N1.7 (Table 2) vs 'GR00T-N1 [5]' (A.3 text, ref [5] = N1, arXiv:2503.14734)",
    "one_fast_slow_tier_LA5_vs_EDF_per_robot_%": rnd((T5[("one_fast", "LA w=5")][1] / T5[("one_fast", "EDF")][1] - 1) * 100, 1),
    "one_fast_slow_tier_starv_LA5_vs_EDF_pp": rnd(T5[("one_fast", "LA w=5")][5] - T5[("one_fast", "EDF")][5], 2),
    "half_fast_slow_tier_LA5_vs_EDF_per_robot_%": rnd((T5[("half_fast", "LA w=5")][1] / T5[("half_fast", "EDF")][1] - 1) * 100, 1),
    "half_fast_fast_tier_LA5_vs_EDF_per_robot_%": rnd((T5[("half_fast", "LA w=5")][0] / T5[("half_fast", "EDF")][0] - 1) * 100, 1),
}

# starvation-throughput correlation in real-world cells (Table 5, fast tier only etc.)
xs = [(v[3], v[2]) for v in T5.values()]
R["V9_real_avg_starv_vs_total_throughput_spearman"] = rnd(stats.spearmanr([a for a, _ in xs], [b for _, b in xs])[0], 3)

json.dump(R, open(HERE / "recompute_results.json", "w"), indent=1, default=str)
print(json.dumps({k: R[k] for k in ["V2_max_abs_diff", "V3_headline", "V3_one_fast_count_decomposition_3seeds_pieces",
                                    "V4a_poisson_exact_on_totals", "V4b_welch_from_fig4_errorbars", "V6_table3",
                                    "V5c_within_cell_spearman_throughput_vs_starvation"]}, indent=1))
