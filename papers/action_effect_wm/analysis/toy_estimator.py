"""토이: 부록 A.6의 두 추정량 비교 (CPU, 수 초).

논문 주장: 미니배치 평균의 제곱 (plug-in)은 E[d]^2 + Var(d)/N_B 로 편향되고,
서로 다른 독립 단위 곱의 평균 (U-통계량, 식 13)은 E[d]^2 로 불편이다.
여기서는 d = mu + sigma*z 에서 두 추정량의 평균과 분산을 반복 시뮬레이션으로 확인한다.
출력: toy_estimator_results.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)
MU, SIGMA = 0.3, 1.0
REPS = 200_000
out = {}
for n_b in (2, 4, 8, 32):
    d = MU + SIGMA * rng.standard_normal((REPS, n_b))
    plug = d.mean(axis=1) ** 2
    s1, s2 = d.sum(axis=1), (d ** 2).sum(axis=1)
    ustat = (s1 ** 2 - s2) / (n_b * (n_b - 1))
    out[str(n_b)] = {
        "target_E_d_squared": MU ** 2,
        "plugin_mean": round(float(plug.mean()), 4),
        "plugin_bias_theory": round(SIGMA ** 2 / n_b, 4),
        "ustat_mean": round(float(ustat.mean()), 4),
        "ustat_sd": round(float(ustat.std()), 4),
    }
(HERE / "toy_estimator_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=1))
