"""토이 1: 한 시드 한 체크포인트 비교에서 Fisher 검정이 얼마나 자주 '유의'가 나오는가 (numpy, 수 초).

논문 4.4절과 부록 D는 (과제, 방법) 칸마다 시드 하나로 학습하고, 그 체크포인트를 N번(20 또는 30) 굴려 Fisher 단측 검정을 한다.
이 검정은 N번 굴린 표본 오차만 본다. 학습 시드가 바뀌면 체크포인트의 진짜 성공률이 달라지는데 그 분산은 들어 있지 않다.
가정: 두 방법의 시드별 진짜 성공률 = mu + s * 표준정규(0, 1 범위로 자름). s는 논문에 없는 값이라 0, 0.05, 0.10, 0.15를 훑는다.
  null : 두 방법의 평균 mu가 같다(0.75).  alt: WHIRL이 평균 +0.15 높다.
실행: python3 analysis/toy_single_seed.py  → analysis/toy_single_seed_results.json
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)
SIMS = 20000


def table(n):
    t = np.ones((n + 1, n + 1))
    for a in range(n + 1):
        for b in range(n + 1):
            t[a, b] = stats.fisher_exact([[a, n - a], [b, n - b]], alternative="greater")[1]
    return t


out = {"assumption": "시드별 진짜 성공률 SD s는 논문에 없는 가정값", "sims": SIMS}
for n in (20, 30):
    tb = table(n)
    rows = {}
    for s in (0.0, 0.05, 0.10, 0.15):
        res = {}
        for label, delta in (("null_same_mean", 0.0), ("alt_plus_0.15", 0.15)):
            p1 = np.clip(0.75 + delta + s * rng.standard_normal(SIMS), 0.02, 0.98)
            p2 = np.clip(0.75 + s * rng.standard_normal(SIMS), 0.02, 0.98)
            a = rng.binomial(n, p1)
            b = rng.binomial(n, p2)
            res[label] = round(float((tb[a, b] < 0.05).mean()), 3)
        rows[f"s={s}"] = res
    out[f"N={n}"] = rows
Path(ROOT / "analysis" / "toy_single_seed_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
