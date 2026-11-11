"""실물 소표본 검정력 토이: 칸당 시도 수 n(두 집단)에서 Fisher 정확 검정(양측, α=0.05)의 정확한 검정력.

검정력은 두 이항 분포의 모든 결과 조합(k1, k2)에 대해 p값을 계산하고 확률로 가중합해 구한다(몬테카를로 없음).
ULTRA Table IV는 칸당 10회(방향별)이므로 n = 10, 20, 30, 40을 본다.
실행: python3 analysis/power_toy.py  (CPU, 수 초). 결과: analysis/power_toy_results.json
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("power_toy_results.json")
ALPHA = 0.05


def pval_table(n):
    P = np.ones((n + 1, n + 1))
    for a in range(n + 1):
        for b in range(n + 1):
            P[a, b] = stats.fisher_exact([[a, n - a], [b, n - b]])[1]
    return P


def power(n, p1, p2, P=None):
    if P is None:
        P = pval_table(n)
    pm1 = stats.binom.pmf(np.arange(n + 1), n, p1)
    pm2 = stats.binom.pmf(np.arange(n + 1), n, p2)
    sig = P < ALPHA
    return float(pm1 @ sig @ pm2)


def min_n_for_power(pairs, target=0.8, nmax=80, step=2):
    """쌍마다 검정력이 목표에 처음 도달하는 n. p값 표는 n마다 한 번만 계산한다."""
    found = {}
    for n in range(10, nmax + 1, step):
        P = pval_table(n)
        for p1, p2 in pairs:
            key = f"{p1}->{p2}"
            if key not in found and power(n, p1, p2, P) >= target:
                found[key] = n
        if len(found) == len(pairs):
            break
    return {f"{p1}->{p2}": found.get(f"{p1}->{p2}") for p1, p2 in pairs}


if __name__ == "__main__":
    res = {"alpha": ALPHA, "test": "Fisher exact, two-sided", "grid": {}}
    pairs = [(0.5, 0.8), (0.5, 0.9), (0.5, 0.6), (0.2, 0.6), (0.25, 0.45)]
    for n in (10, 20, 30, 40):
        P = pval_table(n)
        res["grid"][f"n={n}"] = {f"{p1}->{p2}": round(power(n, p1, p2, P), 3) for p1, p2 in pairs}
    res["min_n_per_arm_for_80pct_step2"] = min_n_for_power(pairs)
    # 판정 기준: 실물 칸(n=10) vs n=10 두 집단에서 가장 큰 차이(50%→90%)를 잡는 검정력
    res["note"] = ("칸당 n=10(실물 Table IV)에서 50%→80% 차이의 검정력은 약 0.13, 50%→90%는 약 0.30이다. "
                   "80% 검정력에는 두 집단당 약 24–44회가 필요하다. 실물 칸 간 차이 대부분은 이 검정으로 구분되지 않는다.")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=1))
