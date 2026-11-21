"""Recompute reported numbers of arXiv:2611.20244 (v1) from the paper's own tables.

Inputs are typed from the paper (Table 1 test IQMs, Table 2 per-task test success, text claims).
No model, simulator or official code is run. Output: recompute_results.json (same folder).
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import trim_mean

HERE = Path(__file__).parent
TASKS = ["spatial0", "spatial2", "object1", "object3", "goal1", "goal2", "10_1", "10_3"]
# Table 2 (per-task test success), (base, RG) per shift column, order of TASKS
T2 = {
    "lat250": [(.23, .99), (.54, .95), (.51, .99), (.66, .95), (.74, .95), (.74, .94), (.51, .85), (.91, .94)],
    "lat400": [(.00, .91), (.00, .83), (.14, .89), (.11, .87), (.40, .93), (.43, .95), (.00, .76), (.66, .99)],
    "comp":   [(.03, 1.00), (.00, .98), (.00, .99), (.00, .96), (.37, .95), (.00, .92), (.00, .91), (.00, .94)],
    "rot90":  [(.00, .99), (.00, 1.00), (.00, 1.00), (.00, .99), (.00, .99), (.00, .95), (.00, .88), (.00, .77)],
    "gain3":  [(.09, 1.00), (.03, 1.00), (.14, .97), (.03, .95), (.34, 1.00), (.34, .90), (.00, .90), (.34, .49)],
}
# Table 1 (pi0.5, test, IQM over 8 tasks): success and unsafe rate as printed
T1 = {
    "lat250": dict(succ=dict(base=.61, linear=.49, intent=.80, rg=.96, oracle=.97), unsafe=dict(base=.68, linear=1.00, intent=.17, rg=.29)),
    "lat400": dict(succ=dict(base=.16, linear=.07, intent=.16, rg=.90, oracle=.91), unsafe=dict(base=.86, linear=1.00, intent=.45, rg=.35)),
    "comp":   dict(succ=dict(base=.00, linear=.86, intent=.67, rg=.97, oracle=.92), unsafe=dict(base=.78, linear=.23, intent=.15, rg=.17)),
    "rot90":  dict(succ=dict(base=.00, linear=.98, intent=.98, rg=.98, oracle=.98), unsafe=dict(base=.80, linear=.76, intent=.36, rg=.29)),
    "gain3":  dict(succ=dict(base=.15, linear=.94, intent=.98, rg=.96, oracle=.98), unsafe=dict(base=1.00, linear=1.00, intent=.80, rg=.77)),
}
NOMINAL_SUCC_VAL = 0.99   # paper text, Question A (validation split; no nominal test value is reported)
rng = np.random.default_rng(0)
out = {}

def iqm(x):
    return float(trim_mean(np.asarray(x, float), 0.25))

# 1) Table 2 -> IQM and mean per column, compared with Table 1
out["table2_vs_table1"] = {}
for col, pairs in T2.items():
    b = np.array([p[0] for p in pairs]); r = np.array([p[1] for p in pairs])
    key = {"lat250": "lat250", "lat400": "lat400", "comp": "comp", "rot90": "rot90", "gain3": "gain3"}[col]
    out["table2_vs_table1"][col] = {
        "base_iqm": round(iqm(b), 4), "base_mean": round(b.mean(), 4), "base_table1": T1[key]["succ"]["base"],
        "rg_iqm": round(iqm(r), 4), "rg_mean": round(r.mean(), 4), "rg_table1": T1[key]["succ"]["rg"],
    }

# 2) Trial-count check: Table 2 values are printed to 2 decimals; does a k/n grid explain them?
#    A value passes if some integer k gives round(k/n, 2) == value. Large n always passes, so this
#    test cannot identify the trial count; it is reported only to show that.
vals = [v for pairs in T2.values() for p in pairs for v in p]
out["grid_fit"] = {}
for n in [35, 105, 175]:
    ok = sum(any(round(k / n, 2) == round(v, 2) for k in range(n + 1)) for v in vals)
    out["grid_fit"][str(n)] = f"{ok}/{len(vals)}"

# 3) Recovery (paper: "recovers 0.91-1.00 of lost success"); two reference definitions
rec = {}
for col, key in [("lat250", "lat250"), ("lat400", "lat400"), ("comp", "comp"), ("rot90", "rot90"), ("gain3", "gain3")]:
    s = T1[key]["succ"]
    rec[col] = dict(
        vs_oracle=round((s["rg"] - s["base"]) / (s["oracle"] - s["base"]), 3) if s["oracle"] != s["base"] else None,
        vs_nominal_val=round((s["rg"] - s["base"]) / (NOMINAL_SUCC_VAL - s["base"]), 3) if NOMINAL_SUCC_VAL != s["base"] else None,
    )
out["recovery"] = rec
out["recovery_range_vs_nominal_val"] = [min(v["vs_nominal_val"] for v in rec.values()), max(v["vs_nominal_val"] for v in rec.values())]
out["recovery_range_vs_oracle"] = [min(v["vs_oracle"] for v in rec.values()), max(v["vs_oracle"] for v in rec.values())]

# 4) Paired sign test (RG > base across the 8 tasks) and bootstrap CI of IQM
def sign_p(d):
    n = len(d); k = int(np.sum(d > 0)); ties = int(np.sum(d == 0))
    from math import comb
    m = n - ties
    kk = min(k, m - k)
    p = sum(comb(m, i) for i in range(kk + 1)) / 2 ** m * 2
    return min(1.0, p), k, m

out["sign_test"] = {}
out["bootstrap"] = {}
for col, pairs in T2.items():
    b = np.array([p[0] for p in pairs]); r = np.array([p[1] for p in pairs])
    p, k, m = sign_p(r - b)
    out["sign_test"][col] = dict(rg_gt_base=k, non_tied=m, p_two_sided=round(p, 4))
    for n_ep in (105, 175):    # 35 test states x 3 seeds (assumed) and 35 x 5 seeds (README)
        boots = []
        for _ in range(4000):
            pb = rng.binomial(n_ep, b) / n_ep
            pr = rng.binomial(n_ep, r) / n_ep
            boots.append((iqm(pb), iqm(pr)))
        boots = np.array(boots)
        out["bootstrap"][f"{col}_n{n_ep}"] = dict(
            base_ci95=[round(float(x), 3) for x in np.percentile(boots[:, 0], [2.5, 97.5])],
            rg_ci95=[round(float(x), 3) for x in np.percentile(boots[:, 1], [2.5, 97.5])],
        )

# 5) Text claims
claims = {
    "abstract 0.16->0.90 at 400ms": (T1["lat400"]["succ"]["base"], T1["lat400"]["succ"]["rg"]),
    "ReGround overall 0.90-0.98 (min,max)": (min(T1[k]["succ"]["rg"] for k in T1), max(T1[k]["succ"]["rg"] for k in T1)),
    "+0.16 at 250 vs best baseline": round(T1["lat250"]["succ"]["rg"] - max(T1["lat250"]["succ"][x] for x in ("base", "linear", "intent")), 2),
    "+0.74 at 400 vs best baseline": round(T1["lat400"]["succ"]["rg"] - max(T1["lat400"]["succ"][x] for x in ("base", "linear", "intent")), 2),
    "+0.11 composite vs best baseline": round(T1["comp"]["succ"]["rg"] - max(T1["comp"]["succ"][x] for x in ("base", "linear", "intent")), 2),
    "composite RG vs oracle (0.97 vs 0.92)": (T1["comp"]["succ"]["rg"], T1["comp"]["succ"]["oracle"]),
    "rot90 unsafe linear->RG (0.76->0.29)": (T1["rot90"]["unsafe"]["linear"], T1["rot90"]["unsafe"]["rg"]),
    "gain3 unsafe linear->RG (1.00->0.77)": (T1["gain3"]["unsafe"]["linear"], T1["gain3"]["unsafe"]["rg"]),
    "Intent safer than RG (unsafe) cells": [k for k in T1 if T1[k]["unsafe"]["intent"] < T1[k]["unsafe"]["rg"]],
}
out["text_claims"] = {k: v for k, v in claims.items()}
(HERE / "recompute_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=lambda o: o.item()))
print(json.dumps(out, indent=2, ensure_ascii=False, default=lambda o: o.item()))
