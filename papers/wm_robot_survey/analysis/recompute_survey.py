"""Survey 03(b)/(c): recompute Table 5 (LIBERO) and Table 6 numbers from the survey's own cells.

Checks
 1. Avg recomputed as the mean of the four suite values vs the reported Avg (rounding to 0.1).
 2. Trial-count integrality: LIBERO uses 500 episodes per suite (10 tasks x 50), so a valid cell is k/5 for integer k.
    The survey does not report n; this is an ASSUMPTION, reported as such.
 3. Long vs Spatial drop (survey 7.3 claims larger drops on Goal/Long) over rows with both values.
 4. Pairwise two-proportion Fisher tests on pooled 2000-episode counts for top LIBERO rows (n = 500 per suite assumed).
Output: analysis/recompute_survey_results.json
"""
import json, math, os
from statistics import mean
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))

# Table 5 rows: name -> (Spatial, Object, Goal, Long, Avg_reported); None = not reported ("–")
T5 = {
 "MimicVideo (Pai 2025)": (94.2, 96.8, 90.6, 94.0, 93.9),
 "Say-Dream-ACT (Gu 2026)": (99.4, 99.2, 98.6, 95.4, 98.1),
 "UVA (Li 2025c)": (None, None, None, 90.0, None),
 "VideoPolicy (Liang 2025a)": (None, None, None, 94.0, None),
 "Cosmos Policy (Kim 2026)": (98.1, 100.0, 98.2, 97.6, 98.5),
 "UD-VLA (Chen 2026b)": (94.1, 95.7, 91.2, 89.6, 92.7),
 "Motus (Bi 2025)": (96.8, 99.8, 96.6, 97.6, 97.7),
 "LingBot-VA (Li 2026b)": (98.5, 99.6, 97.2, 98.5, 98.5),
 "RynnVLA-002 (Cen 2025)": (99.0, 99.8, 96.4, 94.4, 97.4),
 "DreamVLA (Zhang 2025e)": (97.5, 94.0, 89.5, 89.5, 92.6),
 "UniVLA (Bu 2025b)": (96.5, 96.8, 95.6, 92.0, 95.2),
 "Unified VLA (Wang 2025)": (95.4, 98.8, 93.6, 94.0, 95.5),
 "CoWVLA (Yang 2026a)": (97.2, 97.8, 94.6, 92.8, 95.6),
 "F1 (Lv 2025)": (98.2, 97.8, 95.4, 91.3, 95.7),
 "TriVLA (Liu 2025d)": (91.2, 93.8, 89.8, 73.2, 87.0),
 "VLA-JEPA (Sun 2026)": (96.2, 99.6, 97.2, 95.8, 97.2),
 "JEPA-VLA (Miao 2026)": (97.2, 98.0, 95.6, 94.8, 96.4),
}
# UniPi row in Table 5: Long 0.0, others not reported
T5_UNIPI = (None, None, None, 0.0, None)

res = {"table5_avg_check": [], "integrality_n500": [], "long_vs_spatial": {}, "fisher_pairs": []}

for name, (s, o, g, l, avg) in list(T5.items()) + [("UniPi (Du 2023)", T5_UNIPI)]:
    vals = [x for x in (s, o, g, l) if x is not None]
    if avg is not None and len(vals) == 4:
        calc = mean(vals)
        res["table5_avg_check"].append({"method": name, "calc": round(calc, 3), "reported": avg,
                                        "match_0.1": abs(round(calc, 1) - avg) < 0.051,
                                        "exact_tie_at_rounding": abs(calc - round(calc, 1)) - 0.05 < 1e-9 and abs(abs(calc - round(calc, 1)) - 0.05) < 1e-9})
    for suite, x in zip(["S", "O", "G", "L"], (s, o, g, l)):
        if x is None:
            continue
        k = x * 5  # 500 episodes -> k/5 percent
        ok = abs(k - round(k)) < 1e-6
        res["integrality_n500"].append({"method": name, "suite": suite, "value": x, "k_of_500": round(k, 2), "integer": ok})

# Long vs Spatial
diffs = [(n, l - s) for n, (s, o, g, l, a) in T5.items() if s is not None and l is not None]
gdiffs = [(n, g - s) for n, (s, o, g, l, a) in T5.items() if s is not None and g is not None]
odiffs = [(n, o - s) for n, (s, o, g, l, a) in T5.items() if s is not None and o is not None]
res["long_vs_spatial"] = {
    "n_rows_long": len(diffs), "mean_long_minus_spatial": round(mean(d for _, d in diffs), 2),
    "n_rows_goal": len(gdiffs), "mean_goal_minus_spatial": round(mean(d for _, d in gdiffs), 2),
    "n_rows_object": len(odiffs), "mean_object_minus_spatial": round(mean(d for _, d in odiffs), 2),
    "rows_long_below_spatial": sum(1 for _, d in diffs if d < 0), "rows_long_above_spatial": sum(1 for _, d in diffs if d > 0),
}

# Fisher exact (two-sided) for 2x2 with n = 500 per suite, 4 suites pooled = 2000 episodes
def fisher_two_sided(a, b, c, d):
    # table [[a,b],[c,d]] ; rows: method1 (success a, fail b), method2 (success c, fail d)
    n1, n2 = a + b, c + d
    m = a + c
    N = n1 + n2
    def p_of(x):
        return comb(n1, x) * comb(n2, m - x) / comb(N, m)
    p0 = p_of(a)
    lo, hi = max(0, m - n2), min(n1, m)
    return min(1.0, sum(p_of(x) for x in range(lo, hi + 1) if p_of(x) <= p0 * (1 + 1e-9)))

def pooled(name):
    s, o, g, l, a = T5[name]
    vals = [x for x in (s, o, g, l) if x is not None]
    return sum(round(v * 5) for v in vals), 500 * len(vals)

pairs = [("Cosmos Policy (Kim 2026)", "LingBot-VA (Li 2026b)"), ("Cosmos Policy (Kim 2026)", "Motus (Bi 2025)"),
         ("LingBot-VA (Li 2026b)", "Motus (Bi 2025)"), ("Say-Dream-ACT (Gu 2026)", "Cosmos Policy (Kim 2026)"),
         ("VLA-JEPA (Sun 2026)", "JEPA-VLA (Miao 2026)"), ("VLA-JEPA (Sun 2026)", "Motus (Bi 2025)"),
         ("F1 (Lv 2025)", "Unified VLA (Wang 2025)"), ("Motus (Bi 2025)", "F1 (Lv 2025)")]
for a_, b_ in pairs:
    ka, na = pooled(a_)
    kb, nb = pooled(b_)
    p = fisher_two_sided(ka, na - ka, kb, nb - kb)
    res["fisher_pairs"].append({"a": a_, "b": b_, "succ_a": ka, "n_a": na, "succ_b": kb, "n_b": nb,
                                "diff_pp": round(100 * (ka / na - kb / nb), 2), "fisher_p_two_sided": round(p, 4),
                                "note": "suite-level pooled, n=500/suite assumed; Table 5 per-suite cells only"})

json.dump(res, open(os.path.join(HERE, "recompute_survey_results.json"), "w"), ensure_ascii=False, indent=1)
bad_avg = [r for r in res["table5_avg_check"] if not r["match_0.1"] and not r["exact_tie_at_rounding"]]
bad_int = [r for r in res["integrality_n500"] if not r["integer"]]
print("avg rows checked:", len(res["table5_avg_check"]), "mismatch:", bad_avg)
print("cells checked:", len(res["integrality_n500"]), "non-integer under n=500:", bad_int)
print("long vs spatial:", res["long_vs_spatial"])
for f in res["fisher_pairs"]:
    print(f["a"], "vs", f["b"], f["succ_a"], f["n_a"], f["succ_b"], f["n_b"], "diff", f["diff_pp"], "p", f["fisher_p_two_sided"])
