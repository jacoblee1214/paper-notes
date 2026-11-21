"""03 recomputation: paper statistics from published counts (no model runs).

Inputs: repo/results/budget_shift/summary.json (official repo, static read),
paper Table 1 / A1 / A2 / A4 / A6 / A7 / A8 values typed from paper/paper.pdf.
Output: analysis/03_recompute_results.json
"""
import json, math, os
from scipy.stats import binom, beta, norm
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
S = json.load(open(os.path.join(ROOT, "repo/results/budget_shift/summary.json")))
out = {}

# ---- V1: Wilson 95% intervals for Table 1 success counts (n=100) ----
def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)

arms = {"IDM uncached 2": 95, "IDM uncached 10": 95, "IDM cache tau8": 63,
        "IDM cache tau2": 97, "IDM cache tau4 K10": 89, "IDM attenuated": 81,
        "orig uncached 2": 96, "orig uncached 10": 95, "orig cache tau8": 94}
v1 = {}
for name, k in arms.items():
    lo, hi = wilson(k, 100)
    stored = None
    v1[name] = {"k": k, "n": 100, "wilson95": [round(lo, 1), round(hi, 1)]}
# cross-check Wilson vs summary.json for the frontier arms
chk = []
for ck, lab in [("idm", "uncached 2 steps"), ("idm", "10 steps, cache tau=8")]:
    pass
for arm_key, arm in S["frontier"]["idm"]["arms"].items():
    lo, hi = wilson(arm["success"]["k"], arm["success"]["n"])
    chk.append({"arm": "IDM " + arm_key, "k": arm["success"]["k"],
                "summary_wilson": arm["wilson95"],
                "recomputed": [round(lo, 1), round(hi, 1)],
                "match": abs(lo - arm["wilson95"][0]) < 0.15 and abs(hi - arm["wilson95"][1]) < 0.15})
for arm_key, arm in S["frontier"]["native"]["arms"].items():
    lo, hi = wilson(arm["success"]["k"], arm["success"]["n"])
    chk.append({"arm": "orig " + arm_key, "k": arm["success"]["k"],
                "summary_wilson": arm["wilson95"],
                "recomputed": [round(lo, 1), round(hi, 1)],
                "match": abs(lo - arm["wilson95"][0]) < 0.15 and abs(hi - arm["wilson95"][1]) < 0.15})
out["V1_wilson_table1"] = {"table": v1, "vs_summary_json": chk,
                            "all_match": all(c["match"] for c in chk)}

# ---- V2: exact McNemar from discordant counts + Holm over the 11 primary contrasts ----
def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * binom.cdf(k, n, 0.5))

prim = S["primary_contrasts"]
rows = []
for r in prim:
    p = mcnemar_exact(r["a_only"], r["b_only"])
    rows.append({"label": r["label"], "a_only": r["a_only"], "b_only": r["b_only"],
                 "p_recomputed": p, "p_summary": r["p"]})
# Holm
ps = sorted(range(len(rows)), key=lambda i: rows[i]["p_recomputed"])
m = len(rows)
adj = [0] * m
running = 0.0
for rank, i in enumerate(ps):
    val = min(1.0, (m - rank) * rows[i]["p_recomputed"])
    running = max(running, val)
    adj[i] = running
for i, r in enumerate(rows):
    r["holm_recomputed"] = adj[i]
    r["holm_summary"] = prim[i]["p_holm"]
out["V2_mcnemar_holm"] = {"rows": rows,
    "max_abs_p_diff": max(abs(r["p_recomputed"] - r["p_summary"]) for r in rows),
    "max_abs_holm_diff": max(abs(r["holm_recomputed"] - r["holm_summary"]) for r in rows)}

# ---- V3: Tango intervals: NOT reimplemented (own quick grid version disagreed by 7-13 points);
# paper values are taken from repo summary.json as reported, flagged unverified here.
out["V3_tango_check"] = {"status": "unverified", "note": "독립 재구현 실패(격자 근사 불일치 7-13점). 저장소 summary.json 값을 그대로 인용."}

# ---- V4: Eq. 1 cache share and shift (paper Sec.2, Table A2) ----
def share(s, cached, K=10):
    pass
def share_K(s, cached, K=10):
    u = 1 - cached / K
    sig = s * u / (1 + (s - 1) * u)
    return 100 * (1 - sig)
share = share_K
eq1 = []
for (s, cached, paper) in [(1, 8, 80), (5, 8, 44), (3, 8, 57), (3, 9, 75), (5, 9, 64),
                           (1, 6, 60), (1, 7, 70), (2, 7, 54), (2, 9, 82), (1, 4, 40),
                           (1, 5, 50), (1, 3, 30), (5, 2, 17)]:
    K = 4 if paper == 17 else 10
    eq1.append({"s": s, "cached": cached, "K": K, "share_recomputed": round(share(s, cached, K), 1),
                "paper": paper, "match": abs(share(s, cached, K) - paper) <= 1.0})
out["V4_eq1_share"] = {"rows": eq1, "all_match": all(r["match"] for r in eq1)}

# ---- V5: latency fixed-cost fit F + a*K (Table 1 host, Table A4) ----
import numpy as np
def fit(steps, ms):
    X = np.column_stack([np.ones(len(steps)), steps])
    coef, *_ = np.linalg.lstsq(X, np.array(ms, float), rcond=None)
    F, a = coef
    return {"F_ms": round(F, 1), "a_ms": round(a, 2), "F_over_a": round(F / a, 2)}
fits = {
    "IDM table1 host (1,2,4,10)": fit([1, 2, 4, 10], [80.9, 93.9, 119.3, 202.2]),
    "orig table1 host (1,2,4,10 from Table A4)": fit([1, 2, 4, 10], [81, 92, 118, 202]),
    "orig CUDA graphs (1,2,4,10)": fit([1, 2, 4, 10], [79, 86, 100, 143]),
    "follow-up IDM (2,10)": fit([2, 10], [192, 477]),
}
F_over_a = 4.85
two_step_fixed_share = F_over_a / (F_over_a + 2)
composed = {
    "cost_model_max_saving_F4.85": round(0.75 / (F_over_a + 2), 3),
    "cost_model_max_saving_F3.4_followup": round(0.75 / (3.4 + 2), 3),
    "cost_model_max_saving_F10_cudagraphs": round(0.75 / (10 + 2), 3),
    "asymptote_tau_inf_F4.85": round(1 / (F_over_a + 2), 3),
    "asymptote_tau_inf_F3.4": round(1 / (3.4 + 2), 3),
}
out["V5_latency_fit"] = {"fits": fits, "fixed_share_at_2_steps": round(two_step_fixed_share, 3),
                         "composed_saving_bounds": composed,
                         "paper_claim": "F~4.8-4.9a; ~70% fixed; 11% (14% follow-up, 6% CUDA graphs); tau-inf 15-19%"}
out["V5_latency_fit"]["sync_chunk"] = {"two_step_ms": round(202 + 50 * 10, 0),
                                       "note": "chunk = call + 10 x 50 ms; 10-step 702, 2-step 594 => 1.18x (paper)"}
out["V5_latency_fit"]["sync_chunk"] = {"ten_step_chunk": 702, "two_step_chunk": 594, "ratio": round(702 / 594, 3)}

# ---- V6: Table A8 per-task totals vs Table 1 counts ----
A8 = {"idm_2": [10,9,9,8,10,10,10,10,10,9], "idm_10": [10,10,7,9,10,10,10,10,10,9],
      "idm_tau8": [9,8,10,3,9,1,4,10,4,5], "orig_2": [10,10,10,9,10,9,10,10,10,8],
      "orig_10": [10,10,9,9,10,8,10,10,10,9], "orig_tau8": [10,10,8,9,10,8,10,10,10,9]}
totals = {k: sum(v) for k, v in A8.items()}
expect = {"idm_2": 95, "idm_10": 95, "idm_tau8": 63, "orig_2": 96, "orig_10": 95, "orig_tau8": 94}
out["V6_task_totals"] = {"totals": totals, "table1": expect,
                         "all_match": totals == expect,
                         "max_task_gap_2v10": max(abs(a - b) for a, b in zip(A8["idm_2"], A8["idm_10"]))}

# ---- V7: Table A2 fresh-vs-fixed pairs (50 states): feasible discordant splits ----
# For a same-state difference of d states, list (b,c) with b-c=d and exact McNemar p matching reported p
A2 = [("IDM s1 60%", 96, 82, 0.039), ("IDM s1 70%", 88, 62, 0.0023),
      ("IDM s1 80%", 54, 26, 0.0013), ("IDM s3 75%", 90, 50, 1e-4)]
feas = []
for name, fx, fr, prep in A2:
    d = (fx - fr) * 50 // 100
    cands = []
    for c in range(0, 40):
        b = c + d
        if b < 0: continue
        p = mcnemar_exact(b, c)
        if abs(p - prep) / max(prep, 1e-9) < 0.1 or (prep < 1e-3 and p < 1e-3):
            cands.append((b, c, round(p, 5)))
    feas.append({"pair": name, "diff_states": d, "candidates_(a_only,b_only,p)": cands[:4]})
out["V7_A2_feasibility"] = feas

# ---- V8: Table A7 episode-latent gains (50 states) feasibility ----
A7 = [("IDM 6-of-10 episode latent vs fresh", 14, 0.065),
      ("IDM 8-of-10 episode latent vs fresh", 38, 0.0002),
      ("IDM 8-of-10 episode latent vs fixed", 10, 0.18),
      ("IDM 2 steps delay2 vs 10 steps delay5", 10, 0.12),
      ("IDM 2 steps delay5 vs delay2", -14, 0.039)]
feasA7 = []
for name, diff_pts, prep in A7:
    d = abs(diff_pts) * 50 // 100
    cands = []
    for c in range(0, 40):
        b = c + d
        p = mcnemar_exact(b, c)
        if abs(math.log10(max(p,1e-12)) - math.log10(prep)) < 0.1 or abs(p - prep) / prep < 0.15:
            cands.append((b, c, round(p, 5)))
    feasA7.append({"row": name, "diff_states": d, "candidates": cands[:4]})
out["V8_A7_feasibility"] = feasA7

json.dump(out, open(os.path.join(HERE, "03_recompute_results.json"), "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: (v.get("all_match") if isinstance(v, dict) and "all_match" in v else "see json") for k, v in out.items()}, indent=1))
print("V2 max p diff", out["V2_mcnemar_holm"]["max_abs_p_diff"], "holm diff", out["V2_mcnemar_holm"]["max_abs_holm_diff"])
print("V5 fits:", fits, "share", round(two_step_fixed_share, 3), composed)
print("V6 totals", totals)
