"""토이 T1: 논문 §4.4 "궤적 수준 1-δ" 주장과 하드웨어 교정 설정(n=100, 반지름이 잔차의 87%를 덮음)의 정합성 검사.

논문은 δ̄ = δ/T로 단계별 오류율을 나누고 q = s_(⌈(n+1)(1-δ̄)⌉)를 쓴다(§4.4). 하드웨어에서는 100개 무작위 제어로
"반올림한 경험적 반지름 0.03 m가 교정 잔차의 87%를 덮는다"고 적었다(§5 Hardware). 이 설정이 궤적 수준에서 무엇을 보장하는지
numpy 몬테카를로로 본다. 잔차 분포는 알려져 있지 않으므로 로그정규(평균 0.015 m)와 시간 상관(AR(1) 가우시안 코풀라)을 바꿔 가며 본다.

확인 항목
  (a) n=100에서 δ̄ < 1/101이면 분위수 인덱스가 n을 넘어 q=+∞ (무의미한 관 크기). T별 최소 δ.
  (b) 올바른 CP(δ=0.1, 인덱스가 유한한 경우만)의 궤적 수준 포함 확률이 1-δ 이상인지.
  (c) 논문의 87% 반지름을 쓰면 길이 T 궤적이 한 번도 관 밖으로 나가지 않을 확률. 상관 ρ=0(독립)부터 0.99까지.

실행: python3 analysis/toy_cp_coverage.py   (numpy, scipy, 약 10 s) → analysis/toy_cp_coverage_results.json
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "toy_cp_coverage_results.json"
rng = np.random.default_rng(0)
SIGMA = 0.6
MU = math.log(0.015) - SIGMA ** 2 / 2          # 평균 0.015 m인 로그정규
N_CAL = 100
N_TRIALS = 20000


def scores(rho, T, size):
    """시간 상관 ρ인 길이 T 잔차열 (주변 분포는 로그정규, 코풀라는 AR(1) 가우시안)."""
    z = np.empty((size, T))
    z[:, 0] = rng.standard_normal(size)
    for t in range(1, T):
        z[:, t] = rho * z[:, t - 1] + math.sqrt(1 - rho ** 2) * rng.standard_normal(size)
    return np.exp(MU + SIGMA * z)


res = {"setup": {"lognormal_mean_m": 0.015, "sigma": SIGMA, "n_cal": N_CAL, "trials": N_TRIALS}}

# (a) 인덱스 유한 조건
res["a_min_delta_finite_quantile"] = {str(T): round(min(T / (N_CAL + 1), 99), 3) for T in (5, 10, 20, 50, 100, 200)}

# (b) 올바른 CP, 독립/상관 모두. δ=0.1, 인덱스가 n 이하인 T만.
cal = np.exp(MU + SIGMA * rng.standard_normal((N_TRIALS, N_CAL)))
cal.sort(axis=1)
b = {}
for T in (5, 8, 10):
    db = 0.1 / T
    idx = math.ceil((N_CAL + 1) * (1 - db))
    if idx > N_CAL:
        b[str(T)] = "q=+inf (인덱스 > n)"
        continue
    q = cal[:, idx - 1]
    row = {}
    for rho in (0.0, 0.5, 0.9):
        test = scores(rho, T, N_TRIALS)
        row[f"rho={rho}"] = round(float(np.mean(np.all(test <= q[:, None], axis=1))), 3)
    b[str(T)] = {"index": idx, "traj_coverage": row, "target": 0.9}
res["b_correct_cp_delta0.1"] = b

# (c) 논문의 87% 반지름 (분포의 0.87 분위수로 고정)
r87 = float(np.exp(MU + SIGMA * stats.norm.ppf(0.87)))
c = {"radius_m": round(r87, 4), "per_step_cov": 0.87}
for T in (5, 10, 20, 50, 200):
    row = {}
    for rho in (0.0, 0.5, 0.9, 0.99):
        test = scores(rho, T, N_TRIALS)
        row[f"rho={rho}"] = round(float(np.mean(np.all(test <= r87, axis=1))), 3)
    row["independent_0.87^T"] = round(0.87 ** T, 4)
    c[f"T={T}"] = row
res["c_paper_87pct_radius"] = c

# (d) 교정 표본 수에 따른 실제 단계별 포함 확률의 변동 (Beta 분포, 분할 CP)
# n=100, 목표 87% 분위수: 실제 포함 확률은 Beta(k, n+1-k) 분포
k = math.ceil((N_CAL + 1) * 0.87)
beta = stats.beta(k, N_CAL + 1 - k)
res["d_split_cp_coverage_spread_n100_target0.87"] = {
    "k": k, "mean": round(float(beta.mean()), 3),
    "p5": round(float(beta.ppf(0.05)), 3), "p95": round(float(beta.ppf(0.95)), 3),
}

OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False))
