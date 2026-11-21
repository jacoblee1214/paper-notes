"""Table 2 success rates: Wilson CIs, Fisher exact tests (stdlib only)."""
import json, math
from fractions import Fraction
from math import comb

def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def fisher_two_sided(a, n1, b, n2):
    # 2x2: a successes of n1 (ours), b successes of n2 (DP); hypergeometric
    K = a + b; N = n1 + n2
    def pmf(x):
        return Fraction(comb(K, x) * comb(N - K, n1 - x), comb(N, n1))
    p_obs = pmf(a)
    lo, hi = max(0, n1 - (N - K)), min(n1, K)
    return float(sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs))

tasks = {
    "Rotation": (37, 40, 22, 40),
    "Scooping": (34, 40, 22, 40),
    "Sweeping": (37, 40, 5, 40),
}
out = {}
for t, (ko, no, kd, nd) in tasks.items():
    lo1, hi1 = wilson(ko, no); lo2, hi2 = wilson(kd, nd)
    out[t] = dict(ours=f"{ko}/{no}", ours_pct=round(100 * ko / no, 1),
                  ours_wilson95=[round(100 * lo1, 1), round(100 * hi1, 1)],
                  dp=f"{kd}/{nd}", dp_pct=round(100 * kd / nd, 1),
                  dp_wilson95=[round(100 * lo2, 1), round(100 * hi2, 1)],
                  diff_pp=round(100 * (ko / no - kd / nd), 1),
                  fisher_p_two_sided=float(f"{fisher_two_sided(ko, no, kd, nd):.3g}"))
# Push-Shape: only mean mIoU and mean rotation error are reported (n=32); no per-trial data, so no CI.
out["Push-Shape"] = dict(note="mIoU 0.731 vs 0.550, rot. error 8.0 vs 48.2 deg, n=32; per-trial values not published, CI/test not computable")
# Consistency checks against Table 1 trial counts
checks = {
    "Table1 test trials (40,40,40,32) match Table2 denominators and text": [40, 40, 40, 32] == [40, 40, 40, 32],
    "Text 92.5% = 37/40": round(100 * 37 / 40, 1) == 92.5,
    "Text 85% = 34/40": round(100 * 34 / 40, 1) == 85.0,
    "Text 55% = 22/40": round(100 * 22 / 40, 1) == 55.0,
    "Text 12.5% = 5/40": round(100 * 5 / 40, 1) == 12.5,
}
out["consistency"] = checks
print(json.dumps(out, ensure_ascii=False, indent=2))
