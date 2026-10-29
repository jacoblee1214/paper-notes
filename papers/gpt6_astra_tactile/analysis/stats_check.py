#!/usr/bin/env python3
"""Claim/number verification for arXiv:2610.28457v1
("GPT-6-Astra Understands Touch and Outperform Vision-Tactile-Language-Action Models").

Re-computes Tab.2/Tab.3 statistics without scipy (numpy + stdlib only).

Usage:
    python3 stats_check.py                 # print markdown tables
    python3 stats_check.py --json out.json # also dump machine-readable results

Assumptions (must be stated wherever these numbers are quoted):
  * every cell = n=100 independent Bernoulli episodes (paper §4.1 says 100 fixed
    test episodes per task for Ours; N0-TWAM paper states 100 trials per task;
    trial counts for the other baselines are NOT stated in their sources).
  * Ours vs baselines are treated as UNPAIRED (baseline numbers are copied from
    other papers; seed identity is unverified).
  * Tab.3 (ablation) is PAIRED by the paper's own statement (same 100 configs),
    but per-episode outcomes are unavailable -> McNemar bounds only.
"""
import argparse
import json
import math

import numpy as np

N = 100
TASKS = ["Insert Hole", "Lift Can", "Grasp Classify", "Lift Bottle", "Pull-out Key"]

# ---- Tab.2 as printed (paper p.9) -------------------------------------------
TAB2 = {
    "ACT + UniVTAC Encoder": ([24, 29, 99, 71, 46], 53.8),
    "pi0.5": ([25, 6, 49, 100, 35], 43.0),
    "N0-VTLA": ([95, 88, 100, 99, 99], 96.2),
    "N0-TWAM": ([99, 93, 94, 58, 79], 84.6),
    "Ours": ([99, 95, 98, 93, 84], 93.8),
}
# ---- Fig.1C bar labels, transcribed from page-01.png (zoomed crop) ----------
FIG1C = {
    "Ours": [99, 95, 98, 93, 84, 93.8],
    "ACT + UniVTAC Encoder": [24, 29, 99, 71, 46, 53.8],
    "pi0.5": [25, 6, 49, 100, 35, 43.0],
    "N0-TWAM": [99, 93, 94, 58, 79, 84.6],
    "N0-VTLA": [95, 88, 100, 99, 99, 96.2],
}
# ---- Tab.3 ------------------------------------------------------------------
TAB3 = {"Ours (V+P+T)": 99, "w/o tactile (V+P)": 31}
REPORTED_GAIN_PP = 68

# ---- 8-task UniVTAC source tables (02_prior_work.md; spot-checked 2026-10-23 via
#      arxiv.org/html/2607.23782 Tab.1 and arxiv.org/html/2607.23783 Tab.2) -------
EXCLUDED = ["Put Bottle", "Insert HDMI", "Insert Tube"]
SRC8 = {  # 5 paper tasks (TASKS order) + 3 excluded tasks
    "ACT + UniVTAC Encoder": [24, 29, 99, 71, 46, 31, 28, 56],
    "pi0.5": [25, 6, 49, 100, 35, 34, 8, 74],
    "N0-VTLA": [95, 88, 100, 99, 99, 60, 25, 99],
    "N0-TWAM": [99, 93, 94, 58, 79, 87, 68, 98],
}
SRC8_REPORTED_AVG = {"ACT + UniVTAC Encoder": 48.0, "pi0.5": 41.4, "N0-VTLA": 83.1, "N0-TWAM": 84.5}


# ============================ numerics (no scipy) ============================
def norm_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))


def norm_ppf(p):
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _betacf(a, b, x, itmax=300, eps=3e-16):
    # Numerical Recipes continued fraction for the incomplete beta function
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c if abs(1 + aa / c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c if abs(1 + aa / c) > 1e-300 else 1e-300
        de = d * c
        h *= de
        if abs(de - 1) < eps:
            break
    return h


def betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def t_cdf(t, df):
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2, 0.5, x)
    return 1 - tail if t > 0 else tail


def t_ppf(p, df):
    lo, hi = -100.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def wilson(k, n=N, alpha=0.05):
    z = norm_ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def newcombe_diff_ci(k1, k2, n1=N, n2=N, alpha=0.05):
    """Newcombe (1998) method 10 hybrid-score CI for p1 - p2."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, alpha)
    l2, u2 = wilson(k2, n2, alpha)
    d = p1 - p2
    return d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def two_prop_z(k1, k2, n1=N, n2=N):
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return z, 2 * (1 - norm_cdf(abs(z)))


def _lchoose(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_two_sided(k1, k2, n1=N, n2=N):
    """Two-sided Fisher exact test (sum of tables with P <= P_obs)."""
    K, M = k1 + k2, n1 + n2
    lo, hi = max(0, K - n2), min(K, n1)
    lp = {x: _lchoose(n1, x) + _lchoose(n2, K - x) - _lchoose(M, K) for x in range(lo, hi + 1)}
    obs = lp[k1]
    return min(1.0, sum(math.exp(v) for v in lp.values() if v <= obs + 1e-9))


def binom_two_sided_exact(k, n, p=0.5):
    probs = [math.exp(_lchoose(n, i) + i * math.log(p) + (n - i) * math.log(1 - p)) for i in range(n + 1)]
    return min(1.0, sum(q for q in probs if q <= probs[k] * (1 + 1e-9)))


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for r, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - r) * pvals[i]))
        adj[i] = running
    return adj


def cmh(ours, base, n=N):
    """Cochran-Mantel-Haenszel test (continuity corrected) stratified by task."""
    s_dev, s_var, num, den = 0.0, 0.0, 0.0, 0.0
    for a, c in zip(ours, base):
        b, d = n - a, n - c
        tot = a + b + c + d
        s_dev += a - (a + b) * (a + c) / tot
        s_var += (a + b) * (c + d) * (a + c) * (b + d) / (tot * tot * (tot - 1))
        num += a * d / tot
        den += b * c / tot
    chi2 = (abs(s_dev) - 0.5) ** 2 / s_var
    p = math.erfc(math.sqrt(chi2 / 2))
    or_mh = num / den if den > 0 else float("inf")
    return chi2, p, or_mh


def stratified_mean_diff(ours, base, n=N):
    """Equal-weight 5-task mean difference, binomial variance per cell (tasks fixed)."""
    d = np.mean(ours) - np.mean(base)  # in pp
    var = sum((a / n) * (1 - a / n) / n + (c / n) * (1 - c / n) / n for a, c in zip(ours, base)) / len(ours) ** 2
    se = math.sqrt(var) * 100
    z = d / se
    return dict(diff_pp=d, se_pp=se, z=z, p_two=2 * (1 - norm_cdf(abs(z))),
                p_one_ours_lower=norm_cdf(z),
                ci95=(d - 1.959964 * se, d + 1.959964 * se),
                ci90=(d - 1.644854 * se, d + 1.644854 * se))


def tost(diff, se, margin):
    """Equivalence (two one-sided tests) with +/- margin in pp."""
    p_low = 1 - norm_cdf((diff + margin) / se)
    p_up = norm_cdf((diff - margin) / se)
    return max(p_low, p_up)


def task_level_paired(ours, base):
    d = np.array(ours, float) - np.array(base, float)
    m, sd = d.mean(), d.std(ddof=1)
    se = sd / math.sqrt(len(d))
    t = m / se
    df = len(d) - 1
    tq = t_ppf(0.975, df)
    nz = d[d != 0]
    k = int((nz > 0).sum())
    sign_p = binom_two_sided_exact(k, len(nz)) if len(nz) else 1.0
    return dict(diffs=d.tolist(), mean=m, sd=sd, t=t, df=df, p=2 * (1 - t_cdf(abs(t), df)),
                ci95=(m - tq * se, m + tq * se), sign_pos=k, sign_n=len(nz), sign_p=sign_p)


def bayes_prob_mean_greater(ours, base, n=N, draws=200_000, seed=0):
    """P(mean_ours > mean_base) under independent Jeffreys Beta posteriors per cell."""
    rng = np.random.default_rng(seed)
    mo = np.mean([rng.beta(k + 0.5, n - k + 0.5, draws) for k in ours], axis=0)
    mb = np.mean([rng.beta(k + 0.5, n - k + 0.5, draws) for k in base], axis=0)
    diff = (mo - mb) * 100
    return float((mo > mb).mean()), (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5)))


def n_per_arm(p1, p2, alpha=0.05, power=0.8):
    za, zb = norm_ppf(1 - alpha / 2), norm_ppf(power)
    pb = (p1 + p2) / 2
    num = (za * math.sqrt(2 * pb * (1 - pb)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)


def n_per_task_for_mean(ours, base, alpha=0.05, power=0.8):
    za, zb = norm_ppf(1 - alpha / 2), norm_ppf(power)
    s = sum((a / 100) * (1 - a / 100) + (c / 100) * (1 - c / 100) for a, c in zip(ours, base)) / len(ours) ** 2
    d = abs(np.mean(ours) - np.mean(base)) / 100
    return math.ceil((za + zb) ** 2 * s / d ** 2)


def mde_at_n(p_base, n=N, alpha=0.05, power=0.8, direction=+1):
    """Smallest detectable shift from p_base (normal approx). None if unreachable within [0, 1]."""
    lo, hi = p_base, (1.0 if direction > 0 else 0.0)
    if n_per_arm(hi, p_base, alpha, power) > n:
        return None
    for _ in range(100):
        mid = (lo + hi) / 2
        need = n_per_arm(mid, p_base, alpha, power)
        if need > n:
            lo = mid
        else:
            hi = mid
    return (hi - p_base) * 100


# ================================== checks ===================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write results to this JSON path")
    args = ap.parse_args()
    R = {}

    # 1) internal consistency ------------------------------------------------
    print("## 1. Tab.2 mean recomputation / Fig.1C cross-check\n")
    print("| Method | recomputed mean | reported | match | Fig.1C (5 tasks + avg) match |")
    print("|---|---|---|---|---|")
    R["means"] = {}
    for m, (v, rep) in TAB2.items():
        rec = sum(v) / len(v)
        ok = abs(rec - rep) < 1e-9
        fig_ok = FIG1C[m] == v + [rep]
        R["means"][m] = dict(recomputed=rec, reported=rep, match=ok, fig1c_match=fig_ok)
        print(f"| {m} | {rec:.1f} | {rep:.1f} | {'OK' if ok else 'MISMATCH'} | {'OK' if fig_ok else 'MISMATCH'} |")
        assert ok and fig_ok, m
    gain = TAB3["Ours (V+P+T)"] - TAB3["w/o tactile (V+P)"]
    assert gain == REPORTED_GAIN_PP
    print(f"\nTab.3 gain: {gain} pp (reported {REPORTED_GAIN_PP}) -> OK")
    assert TAB2["Ours"][0][0] == TAB3["Ours (V+P+T)"], "Tab.2 vs Tab.3 Insert Hole"
    print("Tab.2 Ours/Insert Hole (99) == Tab.3 Ours (99) -> OK\n")

    print("Per-task ranking of Ours (1 = best, ties share rank):")
    R["ranks"] = {}
    for i, t in enumerate(TASKS + ["Mean"]):
        vals = {m: (v[i] if i < 5 else rep) for m, (v, rep) in TAB2.items()}
        rank = 1 + sum(1 for m, x in vals.items() if m != "Ours" and x > vals["Ours"])
        better = [f"{m}={x}" for m, x in vals.items() if m != "Ours" and x > vals["Ours"]]
        R["ranks"][t] = dict(rank=rank, beaten_by=better)
        print(f"  {t}: rank {rank}/5; higher: {better or '-'}")

    print("\nClaim '§4.2 matches or outperforms N0-TWAM on all five tasks':")
    cmp_ = [(t, o, b) for t, o, b in zip(TASKS, TAB2["Ours"][0], TAB2["N0-TWAM"][0])]
    print("  ", [(t, o, b, 'OK' if o >= b else 'FAIL') for t, o, b in cmp_])

    # 2) Wilson CIs --------------------------------------------------------------
    print("\n## 2. Wilson 95% CI (n=100 per cell)\n")
    print("| Method | " + " | ".join(TASKS) + " |")
    print("|---|" + "---|" * 5)
    R["wilson"] = {}
    for m, (v, _) in TAB2.items():
        cis = [wilson(k) for k in v]
        R["wilson"][m] = [(round(l * 100, 1), round(u * 100, 1)) for l, u in cis]
        print(f"| {m} | " + " | ".join(f"{k} [{l*100:.1f}, {u*100:.1f}]" for k, (l, u) in zip(v, cis)) + " |")
    for m, k in TAB3.items():
        l, u = wilson(k)
        R["wilson"][m] = (round(l * 100, 1), round(u * 100, 1))
        print(f"Tab.3 {m}: {k} [{l*100:.1f}, {u*100:.1f}]")

    # 3) pairwise tests ----------------------------------------------------------
    print("\n## 3. Ours vs each baseline, per task (unpaired)\n")
    print("| Baseline | Task | Ours | Base | diff pp | Newcombe 95% CI | z | p (z) | p (Fisher) | Holm-Fisher (5 tasks) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    R["pairwise"] = {}
    ours = TAB2["Ours"][0]
    for b in ["N0-VTLA", "N0-TWAM", "ACT + UniVTAC Encoder", "pi0.5"]:
        base = TAB2[b][0]
        fps = [fisher_two_sided(o, c) for o, c in zip(ours, base)]
        hp = holm(fps)
        R["pairwise"][b] = []
        for t, o, c, fp, h in zip(TASKS, ours, base, fps, hp):
            z, pz = two_prop_z(o, c)
            lo, hi = newcombe_diff_ci(o, c)
            R["pairwise"][b].append(dict(task=t, ours=o, base=c, z=z, p_z=pz, p_fisher=fp, p_holm=h,
                                         ci=(lo * 100, hi * 100)))
            print(f"| {b} | {t} | {o} | {c} | {o-c:+d} | [{lo*100:+.1f}, {hi*100:+.1f}] | {z:+.2f} | {pz:.3g} | {fp:.3g} | {h:.3g} |")

    # 4) 5-task mean -------------------------------------------------------------
    print("\n## 4. Five-task mean comparisons\n")
    R["mean_tests"] = {}
    for b in ["N0-VTLA", "N0-TWAM"]:
        base = TAB2[b][0]
        s = stratified_mean_diff(ours, base)
        tl = task_level_paired(ours, base)
        chi2, pc, orr = cmh(ours, base)
        pf = fisher_two_sided(sum(ours), sum(base), 500, 500)
        pr, cri = bayes_prob_mean_greater(ours, base)
        eq = {m: tost(s["diff_pp"], s["se_pp"], m) for m in (2, 3, 5)}
        R["mean_tests"][b] = dict(stratified=s, task_level=tl, cmh=dict(chi2=chi2, p=pc, or_mh=orr),
                                  pooled_fisher_p=pf, bayes_p_ours_greater=pr, bayes_cri95=cri, tost_p=eq)
        print(f"### Ours (93.8) vs {b} ({np.mean(base):.1f})")
        print(f"- (A) episode-level, tasks fixed: diff={s['diff_pp']:+.1f}pp, SE={s['se_pp']:.2f}pp, z={s['z']:+.2f}, "
              f"p_two={s['p_two']:.3f}, one-sided p(Ours<base)={s['p_one_ours_lower']:.3f}, "
              f"95%CI=[{s['ci95'][0]:+.1f}, {s['ci95'][1]:+.1f}], 90%CI=[{s['ci90'][0]:+.1f}, {s['ci90'][1]:+.1f}]")
        print(f"- (A') TOST equivalence p: margin 2pp={eq[2]:.3f}, 3pp={eq[3]:.3f}, 5pp={eq[5]:.3f}")
        print(f"- (B) task-level paired t (df={tl['df']}): diffs={tl['diffs']}, mean={tl['mean']:+.1f}, sd={tl['sd']:.2f}, "
              f"t={tl['t']:+.2f}, p={tl['p']:.3f}, 95%CI=[{tl['ci95'][0]:+.1f}, {tl['ci95'][1]:+.1f}]; "
              f"sign test {tl['sign_pos']}/{tl['sign_n']} positive, p={tl['sign_p']:.3f}")
        print(f"- (C) CMH (task-stratified): chi2={chi2:.2f}, p={pc:.3g}, MH OR={orr:.2f}")
        print(f"- (D) pooled 500-vs-500 Fisher: p={pf:.3g} (ignores task stratification)")
        print(f"- (E) Jeffreys-Beta MC: P(mean_Ours > mean_{b})={pr:.3f}, 95% CrI of diff=[{cri[0]:+.1f}, {cri[1]:+.1f}]\n")

    # 5) ablation ---------------------------------------------------------------
    print("## 5. Tab.3 tactile ablation (99 vs 31)\n")
    pf = fisher_two_sided(99, 31)
    # paired: same 100 configs. c = #(ours fail & w/o success) in {0,1}; b = 68 + c
    mc = {c: binom_two_sided_exact(c, 68 + 2 * c) for c in (0, 1)}
    lo, hi = newcombe_diff_ci(99, 31)
    R["ablation"] = dict(fisher_p=pf, mcnemar_exact_p_by_c=mc, diff_ci=(lo * 100, hi * 100))
    print(f"- unpaired Fisher p={pf:.3g}; Newcombe 95% CI of diff=[{lo*100:.1f}, {hi*100:.1f}] pp")
    print(f"- paired McNemar exact (discordant b=68+c, c in {{0,1}}): p(c=0)={mc[0]:.3g}, p(c=1)={mc[1]:.3g}\n")

    # 6) power ------------------------------------------------------------------
    print("## 6. Power (alpha=0.05 two-sided, power=0.8)\n")
    R["power"] = {
        "IH 99 vs 95 n/arm": n_per_arm(0.99, 0.95),
        "LC 95 vs 88 n/arm": n_per_arm(0.95, 0.88),
        "LC 95 vs 93 n/arm": n_per_arm(0.95, 0.93),
        "LB 93 vs 99 n/arm": n_per_arm(0.93, 0.99),
        "mean 93.8 vs 96.2 n/task (stratified)": n_per_task_for_mean(ours, TAB2["N0-VTLA"][0]),
        "mean 93.8 vs 96.2 n/arm (pooled-proportion approx)": n_per_arm(0.938, 0.962),
        "MDE (pp) above base 0.95 at n=100 (None = even 100% is undetectable)": mde_at_n(0.95),
        "MDE (pp) below base 0.95 at n=100": round(mde_at_n(0.95, direction=-1), 1),
        "MDE (pp) below base 0.99 at n=100": round(mde_at_n(0.99, direction=-1), 1),
    }
    for k, v in R["power"].items():
        print(f"- {k}: {v}")

    # 7) 8-task scenario --------------------------------------------------------
    print("\n## 7. UniVTAC 8-task scenario (excluded: " + ", ".join(EXCLUDED) + ")\n")
    R["eight_task"] = {}
    for m, v in SRC8.items():
        a8 = sum(v) / 8
        assert v[:5] == TAB2[m][0], f"{m}: 5-task cells differ from source table"
        assert round(a8, 1) == SRC8_REPORTED_AVG[m] or abs(a8 - SRC8_REPORTED_AVG[m]) < 0.06, m
        need = (a8 * 8 - sum(ours)) / 3
        R["eight_task"][m] = dict(avg8=a8, excl_avg=sum(v[5:]) / 3, ours_needed_excl_avg=need)
        print(f"- {m}: 8-task avg={a8:.2f} (src {SRC8_REPORTED_AVG[m]}), excluded-3 avg={sum(v[5:])/3:.1f}; "
              + (f"Ours needs >= {need:.1f}% mean on excluded 3 tasks to tie this baseline's 8-task avg" if need > 0
                 else "Ours exceeds this baseline's 8-task avg even with 0% on the excluded tasks"))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(R, f, indent=1, default=float, ensure_ascii=False)
        print(f"\n[written] {args.json}")


if __name__ == "__main__":
    main()
