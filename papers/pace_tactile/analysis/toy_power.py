"""토이: 과제당 30회 비교의 검출력, 유의한 결과가 부풀려지는 정도, 보정 없는 다중 비교.

가정: 성공 확률은 이항 분포. 기준값은 논문 표 1의 ACT+Tactile Concat 평균(46.0%)과 같은 46%.
- 검정: 단측 Fisher, α=0.05 (논문과 같은 방식)
- 검출력: 참 차이 Δ에서 유의할 확률
- 부풀림: 유의한 경우의 관측 차이 평균 (Δ가 작을 때 과대추정)
- 보정 없는 다중 비교: 귀무 아래 25회 비교(기준선 5 x 과제 5) 중 하나 이상 p<.05일 확률
실행: python3 analysis/toy_power.py → analysis/toy_power_results.json (numpy, scipy, 단일 프로세스, 수십 초)
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("toy_power_results.json")
N = 30
BASE = 0.46
REPS = 2000
rng = np.random.default_rng(42)


def p_fisher_one(k1, k2, n=N):
    return stats.fisher_exact([[k1, n - k1], [k2, n - k2]], alternative="greater")[1]


def power(delta, reps=REPS):
    p1 = min(0.99, BASE + delta)
    ks1 = rng.binomial(N, p1, size=reps)
    ks0 = rng.binomial(N, BASE, size=reps)
    hits = [p_fisher_one(a, b) < 0.05 for a, b in zip(ks1, ks0)]
    return float(np.mean(hits))


def inflation(delta, reps=REPS):
    p1 = min(0.99, BASE + delta)
    ks1 = rng.binomial(N, p1, size=reps)
    ks0 = rng.binomial(N, BASE, size=reps)
    sig = [(a - b) / N for a, b in zip(ks1, ks0) if p_fisher_one(a, b) < 0.05]
    return float(np.mean(sig)) if sig else None


def fwer_uncorrected(m=25, reps=REPS):
    """귀무 아래 m개 독립 비교에서 하나 이상 p<.05일 확률 (보정 없음)."""
    anyhit = 0
    for _ in range(reps):
        ks1 = rng.binomial(N, BASE, size=m)
        ks0 = rng.binomial(N, BASE, size=m)
        if any(p_fisher_one(a, b) < 0.05 for a, b in zip(ks1, ks0)):
            anyhit += 1
    return anyhit / reps


def main():
    res = {"n_per_cell": N, "base_rate": BASE, "reps": REPS, "power": {}, "inflation": {}}
    for d in [0.10, 0.20, 0.30, 0.45]:
        res["power"][f"delta={d}"] = round(power(d), 3)
        res["inflation"][f"delta={d}"] = inflation(d)
    res["fwer_uncorrected_25_null"] = round(fwer_uncorrected(25, reps=500), 3)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
