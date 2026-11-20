"""WorldArena 상관 검정력 toy (numpy). 실행: python3 analysis/toy_small_n_corr.py
검증 대상: 논문 4.3의 EWMScore와 데이터 엔진 성공률 r=0.600, 행동 계획 r=0.360이 모델 6개로 얻어졌다.
질문: 6개 점으로 r=0.6을 얼마나 믿을 수 있나. 참 상관 rho를 여러 값으로 두고 관측 r의 분포를 본다.
Monte Carlo 20000회, 단일 프로세스.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "toy_small_n_corr_results.json"
rng = np.random.default_rng(1)
REPS = 20000


def sample_r(rho, n, reps):
    cov = np.array([[1, rho], [rho, 1]])
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((reps, n, 2)) @ L.T
    x, y = z[..., 0], z[..., 1]
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    return (xc * yc).sum(1) / np.sqrt((xc ** 2).sum(1) * (yc ** 2).sum(1))


res = {"reps": REPS, "scenarios": {}}
for n in [6, 14]:
    rnull = sample_r(0.0, n, REPS)
    crit = float(np.quantile(np.abs(rnull), 0.95))
    tc = float(stats.t.ppf(0.975, n - 2))
    res["scenarios"][f"n{n}"] = {"null_95pct_abs_r": round(crit, 3),
                                 "analytic_critical_r_alpha05": round(tc / math.sqrt(n - 2 + tc * tc), 3)}
    for rho in [0.5, 0.7, 0.8, 0.9]:
        r = sample_r(rho, n, REPS)
        res["scenarios"][f"n{n}"][f"rho{rho}"] = {
            "P(r>=0.600)": round(float(np.mean(r >= 0.600)), 3),
            "median_r": round(float(np.median(r)), 3),
            "P(r>=0.825)": round(float(np.mean(r >= 0.825)), 3),
        }

# 관측 r=0.600 (n=6)에 대한 단측 대략 p (rho=0 귀무)
res["observed_r600_n6_one_sided_p_null"] = round(float(np.mean(sample_r(0.0, 6, REPS) >= 0.600)), 3)
res["note"] = ("n=6에서 r=0.600은 5% 수준 임계(약 0.81)보다 작아 유의하지 않다(귀무 단측 p 약 0.10). "
               "그러나 참 상관이 0.7~0.8이면 관측 r이 0.6 이상으로 나올 확률은 0.7~0.84로 높다. "
               "따라서 0.600은 '약하다'가 아니라 '모델 6개로는 구분이 안 된다'가 맞다. "
               "사람 평가 r=0.825의 n은 논문에 없어 같은 식으로 판단할 수 없다.")
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
