"""WorldVLA 평균 SR 차이의 검출력 toy (numpy, CPU, 1 프로세스, 수 초).
질문: 과제당 롤아웃 50회 x 4과제(=200회)로 논문이 보고한 크기의 차이(행 1 vs 2: +4.4pp, 행 4 vs 5: +1.5pp)가
      95% 수준에서 구별 가능한가? 가정: 과제 4개의 성공 확률은 같은 기준 p 근처에 있고 과제 간 차이는 무시.
실행: python3 analysis/toy_power.py -> analysis/toy_power_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_power_results.json"
rng = np.random.default_rng(0)
N_TASK, N_TRIAL, SIMS = 4, 50, 20000
res = {"assumptions": "4 tasks x 50 trials per arm, independent Bernoulli, pooled-variance z test on 200-trial means"}

def detect_rate(p_base, diff):
    p1, p2 = p_base + diff / 2, p_base - diff / 2  # 차이 diff를 가운데 기준으로 분할
    s1 = rng.binomial(N_TRIAL * N_TASK, p1, SIMS)
    s2 = rng.binomial(N_TRIAL * N_TASK, p2, SIMS)
    n = N_TRIAL * N_TASK
    a, b = s1 / n, s2 / n
    pooled = (a + b) / 2
    se = np.sqrt(np.maximum(pooled * (1 - pooled) * 2 / n, 1e-12))
    z = (a - b) / se
    return float(np.mean(np.abs(z) > 1.96))

rows = []
for p_base in [0.63, 0.77]:
    for diff in [0.015, 0.044, 0.10, 0.226]:
        rows.append({"p_base": p_base, "true_diff_pp": round(100 * diff, 1),
                     "power_95": round(detect_rate(p_base, diff), 3)})
res["power"] = rows
# 검출 가능한 최소 차이(대략): 기준 p=0.7에서 80% 검출력이 되는 diff 탐색
grid = np.linspace(0.02, 0.2, 37)
mdd = next((float(d) for d in grid if detect_rate(0.7, d) >= 0.8), None)
res["min_detectable_diff_pp_at_80pct_power_p0.7"] = None if mdd is None else round(100 * mdd, 1)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
