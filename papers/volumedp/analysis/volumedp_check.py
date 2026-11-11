"""VolumeDP 표 재계산과 실물 통계 (numpy + 표준 라이브러리만 사용).

(1) Table I, II, III 평균 산술, (2) Table III 열 평균과 표 값 비교,
(3) 실물 Table V: 과제별 Fisher 정확검정, 합산 Fisher, Mantel-Haenszel(과제 층화),
(4) Wilson 95% 구간, (5) 검정력(두 비율, 정규근사)과 필요 n.
결과는 volumedp_check_results.json 으로 저장합니다.
"""
import json
import math
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}

# (1) 표 평균 산술
libero = {"Spatial": (90.7, 78.3, 81.1), "Object": (95.9, 92.5, 89.8),
          "Goal": (89.9, 68.3, 67.6), "Long": (78.7, 50.5, 57.6)}
out["libero_avg"] = {"ours": round(np.mean([v[0] for v in libero.values()]), 2),
                     "dp": round(np.mean([v[1] for v in libero.values()]), 2),
                     "dit": round(np.mean([v[2] for v in libero.values()]), 2)}
maniskill = {"ours": [78.0, 54.7, 79.0, 57.3, 75.3], "dp": [64.7, 28.7, 65.3, 47.3, 42.0],
             "dit": [66.0, 36.7, 31.0, 24.0, 36.7]}
out["maniskill_avg"] = {k: round(float(np.mean(v)), 2) for k, v in maniskill.items()}

# (2) LIBERO-Plus 7항목 단순평균 vs 표의 Average 열
plus = {"ours": [23.1, 50.1, 80.8, 56.5, 51.1, 44.7, 66.6],
        "dp": [1.6, 32.3, 77.0, 25.0, 19.8, 20.3, 42.2],
        "dit": [1.6, 18.3, 58.1, 47.5, 29.2, 26.8, 41.2]}
table_plus = {"ours": 53.0, "dp": 31.7, "dit": 31.1}
out["libero_plus"] = {k: {"simple_mean": round(float(np.mean(v)), 2), "table": table_plus[k]}
                      for k, v in plus.items()}

# (3) 실물 Table V (20 trials/task). 과제: (DP 성공, 제안 성공)
real = {"Placing Bowl": (16, 17), "Microwave": (12, 15), "Door Opening": (13, 17), "Nut onto Screw": (5, 12)}
N = 20


def fisher_two_sided(a, b, c, d):
    """2x2 표 [[a,b],[c,d]] 의 양측 Fisher 정확검정 (초기하분포, 주변합 고정)."""
    r1, r2, c1 = a + b, c + d, a + c
    n = r1 + r2

    def pmf(x):
        return math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)

    p_obs = pmf(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return float(sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs * (1 + 1e-9)))


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(center - half, 4), round(center + half, 4)


per_task = {}
for k, (dp, ours) in real.items():
    per_task[k] = {"dp": dp, "ours": ours, "fisher_p": round(fisher_two_sided(ours, N - ours, dp, N - dp), 4),
                   "dp_wilson": wilson(dp, N), "ours_wilson": wilson(ours, N)}
out["real_per_task"] = per_task

dp_tot = sum(v[0] for v in real.values())
ours_tot = sum(v[1] for v in real.values())
out["real_pooled"] = {"dp": dp_tot, "ours": ours_tot, "n_per_arm": 4 * N,
                      "fisher_p": round(fisher_two_sided(ours_tot, 4 * N - ours_tot, dp_tot, 4 * N - dp_tot), 4),
                      "dp_wilson": wilson(dp_tot, 4 * N), "ours_wilson": wilson(ours_tot, 4 * N)}

# Mantel-Haenszel 층화 검정 (과제별 2x2 표, 연속성 보정 없음) 과 MH 공통 오즈비
num = den = 0.0
var = 0.0
for dp, ours in real.values():
    a, b, c, d = ours, N - ours, dp, N - dp  # 행: 제안, DP / 열: 성공, 실패
    n_s = a + b + c + d
    num += a * d / n_s
    den += b * c / n_s
    r1, r2, c1, c2 = a + b, c + d, a + c, b + d
    var += r1 * r2 * c1 * c2 / (n_s * n_s * (n_s - 1))
or_mh = num / den
e_ours = sum((ours + dp) / (2 * N) * N for dp, ours in real.values())
v_ours = sum((ours + dp) / (2 * N) * (1 - (ours + dp) / (2 * N)) * N for dp, ours in real.values())
chi2_mh = (sum(v[1] for v in real.values()) - e_ours) ** 2 / v_ours
p_mh = math.erfc(math.sqrt(chi2_mh) / math.sqrt(2))  # 자유도 1 카이제곱의 상측확률
out["mantel_haenszel"] = {"or_common": round(or_mh, 3), "chi2": round(chi2_mh, 3), "p_two_sided": round(p_mh, 4)}

# Holm 보정 (과제별 Fisher 4개)
ps = sorted(((v["fisher_p"], k) for k, v in per_task.items()))
holm = {}
running = 0.0
for i, (p, k) in enumerate(ps):
    adj = min(1.0, (len(ps) - i) * p)
    running = max(running, adj)
    holm[k] = round(running, 4)
out["holm_per_task"] = holm

# (4)-(5) 검정력: 두 비율 정규근사 (양측 alpha=0.05, power 0.8)
def n_per_arm(p1, p2, alpha_z=1.959964, power_z=0.841621):
    pbar = (p1 + p2) / 2
    num = (alpha_z * math.sqrt(2 * pbar * (1 - pbar)) + power_z * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)


p_dp, p_ours = dp_tot / (4 * N), ours_tot / (4 * N)
out["power"] = {
    "observed_p_dp": round(p_dp, 4), "observed_p_ours": round(p_ours, 4),
    "n_per_arm_for_80pct_at_observed_diff": n_per_arm(p_dp, p_ours),
    "n_per_arm_for_80pct_if_diff_10pt": n_per_arm(0.575, 0.675),
    "n_per_arm_for_80pct_if_diff_5pt": n_per_arm(0.575, 0.625),
}
# 과제별 n=20 은 검정력이 매우 낮습니다. 관측 차이를 참값으로 두면 과대평가되므로 과제별 검정력은 계산하지 않습니다.

# (6) LIBERO 4 suite Welch t: ± 를 3 시드 표준편차(SD)로 가정 (원문에 정의 없음)
try:
    from scipy.stats import ttest_ind_from_stats
    sd = {"Spatial": (1.4, 1.1), "Object": (0.7, 0.7), "Goal": (1.2, 1.2), "Long": (1.9, 1.3)}
    out["libero_welch_sd_assumed"] = {}
    for k, (o, d) in {"Spatial": (90.7, 78.3), "Object": (95.9, 92.5), "Goal": (89.9, 68.3), "Long": (78.7, 50.5)}.items():
        t, p = ttest_ind_from_stats(o, sd[k][0], 3, d, sd[k][1], 3, equal_var=False)
        out["libero_welch_sd_assumed"][k] = {"t": round(float(t), 2), "p": round(float(p), 4)}
except ImportError:
    out["libero_welch_sd_assumed"] = "scipy 없음, 건너뜀"

path = os.path.join(HERE, "volumedp_check_results.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=2))
