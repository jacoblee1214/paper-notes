"""HandUMI (arXiv:2611.22146v1) 수치 재계산: 실로봇 성공률 검정, 일관성 검사.
실행: python3 analysis/recompute.py  (수 초, 표준 라이브러리 + scipy 선택)
결과: analysis/recompute_results.json
"""
import json
import math
from pathlib import Path

from scipy.stats import fisher_exact, binomtest

HERE = Path(__file__).resolve().parent


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


arms = {"ACT_562": 12, "ACT_101": 0, "pi05_562": 7, "pi05_base": 0}
N = 20
out = {"n": N, "successes": arms, "wilson95": {}, "fisher_two_sided": {}}
for k, s in arms.items():
    lo, hi = wilson(s, N)
    out["wilson95"][k] = [round(lo * 100, 1), round(hi * 100, 1)]

pairs = [("ACT_562", "pi05_562"), ("ACT_562", "ACT_101"), ("ACT_562", "pi05_base"),
         ("pi05_562", "ACT_101"), ("pi05_562", "pi05_base"), ("ACT_101", "pi05_base")]
for a, b in pairs:
    tab = [[arms[a], N - arms[a]], [arms[b], N - arms[b]]]
    _, p = fisher_exact(tab, alternative="two-sided")
    out["fisher_two_sided"][f"{a} vs {b}"] = round(p, 6)

# 단측 근거: 전체 실험 n=20에서 ACT 562의 성공률 차이 검출력(정확 Fisher, 격자 시뮬레이션)
def power_fisher(p1, p2, n, alpha=0.05, sims=None):
    # 정확 검정의 검출력: 두 이항 분포의 결합 확률을 직접 합산
    from scipy.stats import binom
    pw = 0.0
    for k1 in range(n + 1):
        for k2 in range(n + 1):
            p_obs = binom.pmf(k1, n, p1) * binom.pmf(k2, n, p2)
            if p_obs == 0:
                continue
            _, pv = fisher_exact([[k1, n - k1], [k2, n - k2]])
            if pv < alpha:
                pw += p_obs
    return pw

out["power_fisher_n20"] = {
    "ACT562_0.6_vs_pi05_0.35": round(power_fisher(0.6, 0.35, 20), 3),
    "ACT562_0.6_vs_pi05_0.6": round(power_fisher(0.6, 0.6, 20), 3),
}

# 신뢰구간이 0.5 이하 차이를 포함하는지 확인하는 필요 n (두 비율 차 0.25, 80% 검출력)
def n_needed(p1, p2, target=0.8, nmax=400):
    for n in range(5, nmax):
        if power_fisher(p1, p2, n) >= target:
            return n
    return None
out["n_per_arm_for_power80_diff0.25"] = n_needed(0.6, 0.35)

# 일관성 검사
checks = {
    "episodes_641_minus_79_eq_562": 641 - 79 == 562,
    "dropped_fraction_pct": round(79 / 641 * 100, 1),
    "frames_per_min_at_30fps": round(201397 / 111.9 / 60, 2),  # 30 fps면 1800 fps.min
    "frames_vs_30fps_expected_min": round(201397 / 30 / 60, 1),
    "parts_plus_quest_usd": round(207.11 + 349.99, 2),
    "per_unit_parts_usd": round(207.11 / 2, 2),
    "rejected_OpenArm_4.7pct_of_641_eps": round(0.047 * 641, 1),
    "rejected_TRLC_3.1pct_of_641_eps": round(0.031 * 641, 1),
    "selfcoll_R1Lite_16.4pct_of_641_eps": round(0.164 * 641, 1),
    "sync_rows_total": 3 + 2 + 2 + 4 + 0,
    "sync_rows_fraction_pct_per_stream_max": round(4 / 198025 * 100, 4),
    "block_days_audit_note": "arXiv: 3 operators, 4 days, ~5 h; project page: 19 blocks, 4 days",
    "abstract_sub_mm_vs_table3_piper_1.17mm": 1.17 > 1.0,
    "acts_101_19.3min_vs_full_112min_ratio": round(101 / 562, 3),
    "acts_101_19.3min_vs_full_112min_time_ratio": round(19.3 / 111.9, 3),
}
out["consistency"] = checks

(HERE / "recompute_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
