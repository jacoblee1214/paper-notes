"""토이 T1: 논문 부록 A.2의 MPPI 하이퍼파라미터에서 한 스텝의 보정 크기가 얼마나 되는가.

묻는 것: K = 256, H = 5, 잡음 sigma = 5 mm, 온도 lambda = 0.08 (부록 A.2)에서 식 10-13의 가중 평균이
공칭 행동을 얼마나 움직일 수 있는가. 천 시뮬레이터는 쓰지 않는다. 종단점이 3차원 위치인 운동학 점과
부록 표 A.2의 거리항·속도항으로 비용을 만든다(비용 단위는 미터로 가정. 논문에 단위가 없다).

  비용  S = sum_{t=1..H} ||x_t - p||_2 + ||x_H - p||_2 + 10 * sum_t ||a_t||^2     (표 A.2의 reach 단계 항)
  해석 근사  Delta ~= (sigma^2 / lambda) * grad_a0 S      (MPPI 가중 평균의 1차 근사)

비용 배율 c를 1, 3, 10, 30으로 바꿔 민감도를 본다. 논문은 거리항 가중 1.0이라고만 쓴다.
출력: analysis/toy_mppi_refine_results.json, 표준출력 요약. 실행: python3 analysis/toy_mppi_refine.py (numpy, 1초 미만)
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_mppi_refine_results.json"
K, H, SIGMA, LAM = 256, 5, 5e-3, 0.08     # 부록 A.2
CLAMP = 0.020                              # 부록 B.1: 스텝당 EE 이동 20 mm 상한
NOMINAL_STEP = 0.010                       # 가정: 공칭 행동 한 스텝 10 mm
rng = np.random.default_rng(0)


def cost(x0, A, target, scale):
    """A: (K, H, 3) 행동 열. 운동학 점 x_{t+1} = x_t + a_t."""
    X = x0 + np.cumsum(A, axis=1)                        # (K, H, 3)
    d = np.linalg.norm(X - target, axis=2)               # (K, H)
    S = d.sum(axis=1) + d[:, -1]                         # 단계 + 종단(식 A.2 설명)
    S = scale * S + 10.0 * (np.linalg.norm(A, axis=2) ** 2).sum(axis=1)
    return S


def one_trial(scale, dist, bias_deg):
    x0 = np.zeros(3)
    target = np.array([dist, 0.0, 0.0])
    # 공칭(사전) 행동: 목표 방향에서 bias_deg 만큼 틀어진 한 스텝 10 mm
    th = np.deg2rad(bias_deg)
    a_nom = NOMINAL_STEP * np.array([np.cos(th), np.sin(th), 0.0])
    Ubar = np.tile(a_nom, (H, 1))
    eps = rng.normal(0.0, SIGMA, size=(K, H, 3))
    A = Ubar[None] + eps
    S = cost(x0, A, target, scale)
    w = np.exp(-(S - S.min()) / LAM)
    w /= w.sum()
    delta = (w[:, None, None] * eps).sum(axis=0)[0]      # 첫 행동의 변화량
    ess = 1.0 / np.sum(w ** 2)
    # 이상적 보정 방향: 공칭 행동에서 목표 방향 단위벡터 쪽으로
    ideal = np.array([1.0, 0.0, 0.0]) * NOMINAL_STEP - a_nom
    proj = float(delta @ ideal / (np.linalg.norm(ideal) + 1e-12))   # 이상적 보정 방향 성분(m)
    # 해석 근사: 수치 기울기
    h = 1e-5
    g = np.zeros(3)
    for i in range(3):
        e = np.zeros(3); e[i] = h
        Ap = np.tile(Ubar, (1, 1, 1)).astype(float); Am = Ap.copy()
        Ap[0, 0] += e; Am[0, 0] -= e
        g[i] = (cost(x0, Ap, target, scale)[0] - cost(x0, Am, target, scale)[0]) / (2 * h)
    approx = -(SIGMA ** 2 / LAM) * g
    return np.linalg.norm(delta), ess, proj, np.linalg.norm(approx)


def main():
    res = {"params": {"K": K, "H": H, "sigma_m": SIGMA, "lambda": LAM, "clamp_m": CLAMP, "nominal_step_m": NOMINAL_STEP}}
    # 가능한 최대 이동: 표본 K개 중 최대 잡음 노름(3차원)
    mx = np.linalg.norm(rng.normal(0, SIGMA, size=(2000, K, 3)), axis=2).max(axis=1)
    res["max_sample_noise_norm_mm"] = {"mean": round(float(mx.mean() * 1e3), 2), "p05": round(float(np.percentile(mx, 5) * 1e3), 2)}
    rows = []
    for scale in (1, 3, 10, 30):
        for dist in (0.20,):
            for bias in (30.0,):
                r = np.array([one_trial(scale, dist, bias)[:4] for _ in range(300)], dtype=float)
                rows.append({
                    "cost_scale": scale, "target_dist_m": dist, "prior_bias_deg": bias,
                    "mean_correction_mm": round(float(r[:, 0].mean() * 1e3), 3),
                    "mean_correction_over_nominal_step_pct": round(float(r[:, 0].mean() / NOMINAL_STEP * 100), 1),
                    "ESS_of_256": round(float(r[:, 1].mean()), 1),
                    "mean_component_along_needed_direction_mm": round(float(r[:, 2].mean() * 1e3), 3),
                    "analytic_sigma2_over_lambda_grad_mm": round(float(r[:, 3].mean() * 1e3), 3),
                })
    res["rows"] = rows
    # 필요한 보정: 30도 어긋난 10 mm 공칭 행동을 목표 방향으로 되돌리는 데 필요한 크기
    th = np.deg2rad(30.0)
    need = np.linalg.norm(np.array([1.0, 0, 0]) * NOMINAL_STEP - NOMINAL_STEP * np.array([np.cos(th), np.sin(th), 0]))
    res["needed_correction_mm_for_30deg_bias"] = round(float(need * 1e3), 2)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
