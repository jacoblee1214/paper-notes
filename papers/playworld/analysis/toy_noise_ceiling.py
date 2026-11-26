"""토이 T1: 정책 18개 상관 r과 RMSE에 시행 수 잡음이 만드는 천장·바닥.

질문: 그림 7의 r = 0.8766, RMSE = 0.171은 세계모델이 완벽해도 나올 수 있는 값의 얼마나 가까이에 있는가?
설정(가정):
  - 정책 18개의 진짜 성공률 p_i = 그림 7 PlayWorld 패널에서 읽은 실제 SR(그림 판독, 0.05 격자).
  - 실제 평가는 정책당 20회(본문 4.4), 세계모델 평가는 정책당 50회(본문 4.4) 이항 표본.
  - 세계모델 오차 모형: 세계모델의 진짜 성공률 = clip(p_i + e_i), e_i ~ N(bias, sigma^2).
  - 완벽한 모델은 sigma = 0, bias = 0.
CPU 수 초, numpy만 사용. 실행: python3 analysis/toy_noise_ceiling.py → toy_noise_ceiling_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_noise_ceiling_results.json"
rng = np.random.default_rng(0)
p_true = np.array([0.0, .05, .05, .15, .15, .30, .30, .45, .45, .60, .50, .65, .55, .65, .80, .80, .85, 1.0])
N_REAL, N_WM, SIMS = 20, 50, 20000
REPORTED = {"PlayWorld": (0.8766, 0.171), "Human Demo": (0.6636, 0.275), "Human Play": (0.6619, 0.297)}


def simulate(sigma, bias=0.0):
    rs, rm = [], []
    for _ in range(SIMS):
        pw = np.clip(p_true + bias + rng.normal(0, sigma, p_true.size), 0, 1)
        x = rng.binomial(N_REAL, p_true) / N_REAL
        y = rng.binomial(N_WM, pw) / N_WM
        if x.std() == 0 or y.std() == 0:
            continue
        rs.append(np.corrcoef(x, y)[0, 1])
        rm.append(np.sqrt(((x - y) ** 2).mean()))
    return float(np.mean(rs)), float(np.percentile(rs, 5)), float(np.percentile(rs, 95)), float(np.mean(rm))


res = {"assumptions": {"n_real": N_REAL, "n_wm": N_WM, "sims": SIMS, "p_true": p_true.tolist()}}
r0, lo, hi, rm0 = simulate(0.0)
res["perfect_model"] = {"r_mean": round(r0, 3), "r_5_95": [round(lo, 3), round(hi, 3)], "rmse_mean": round(rm0, 3)}
res["analytic_rmse_floor"] = round(float(np.sqrt((p_true * (1 - p_true)).mean() * (1 / N_REAL + 1 / N_WM))), 3)
grid = {}
for s in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
    r, l, h, m = simulate(s)
    grid[str(s)] = {"r_mean": round(r, 3), "r_5_95": [round(l, 3), round(h, 3)], "rmse_mean": round(m, 3)}
res["sigma_grid"] = grid
# 보고된 RMSE에 맞는 sigma를 보간으로 찾는다
ss = np.array([float(k) for k in grid]); mm = np.array([grid[k]["rmse_mean"] for k in grid]); rr = np.array([grid[k]["r_mean"] for k in grid])
res["implied_sigma_from_rmse"] = {k: round(float(np.interp(v[1], mm, ss)), 3) for k, v in REPORTED.items()}
res["implied_r_at_that_sigma"] = {k: round(float(np.interp(res["implied_sigma_from_rmse"][k], ss, rr)), 3) for k in REPORTED}
res["reported"] = REPORTED
# 편향 모형: 세계모델이 모든 정책을 bias만큼 높게 평가 (환각 성공)
res["bias_grid"] = {str(b): dict(zip(["r_mean", "r_p5", "r_p95", "rmse_mean"], [round(x, 3) for x in simulate(0.05, b)])) for b in [0.0, 0.1, 0.2, 0.3]}
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: res[k] for k in ["perfect_model", "analytic_rmse_floor", "implied_sigma_from_rmse", "implied_r_at_that_sigma"]}, indent=1))
