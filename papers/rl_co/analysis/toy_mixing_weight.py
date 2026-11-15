"""토이 A: 실제 데모 손실의 가중치 β가 시뮬레이션-실제 차이(bias)와 실데이터 분산(variance)을 어떻게 맞바꾸는가.

모형 (numpy, 단일 프로세스, 수 초~1분):
- 차원 D = 20. 실제 과제의 목표 행동 y_r ~ N(0, I). 시뮬레이션 과제 y_s = y_r + g, ||g|| = G (고정 방향).
- 실제 데모 N_r개의 평균 ŷ_r = y_r + 잡음(분산 σ²/N_r). 시뮬레이션 데이터 N_s = 1000개의 평균 ŷ_s = y_s + 잡음(σ²/N_s).
- Stage II를 "시뮬레이션 RL이 수렴해서 시뮬레이션 최적 ŷ_s에 도달한다"고 가정하고, 실제 앵커를 β로 섞는다:
    θ(β) = argmin ||θ - ŷ_s||² + β ||θ - ŷ_r||²  =  (ŷ_s + β ŷ_r) / (1 + β)
- 실제 오차(차원당) E(β) = (G²/D + σ²/N_s + β² σ²/N_r) / (1+β)²  (기댓값, 해석식)
- 최적 β*  = c / b,  c = G²/D + σ²/N_s,  b = σ²/N_r   →  β* = N_r (G²/(D σ²) + 1/N_s)
  E* = c b / (c + b).

이 토이는 RL 과정 자체(탐색, 보상 잡음, 수렴 속도)와 정책 표현을 모형화하지 않는다. β를 잠정적으로 "정규화된 손실의 가중치"로 해석한다.
검증: 몬테카를로 평균 오차의 argmin이 해석식 β*와 맞는지, 기준(β=0 시뮬레이션만, β→∞ 실제만, β=1)과 비교한다.

실행: python3 analysis/toy_mixing_weight.py  → analysis/toy_mixing_results.json
"""
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_mixing_results.json"
D = 20
SIGMA = 1.0
N_S = 1000
GAPS = [0.0, 0.5, 1.0, 2.0]
N_REAL = [20, 50]
BETAS = np.logspace(-3, 3, 121)
REPS = 4000
SEED = 0


def theory(gap, n_r):
    c = gap ** 2 / D + SIGMA ** 2 / N_S
    b = SIGMA ** 2 / n_r
    beta_star = c / b
    e_star = c * b / (c + b)
    return beta_star, e_star, c, b


def simulate(gap, n_r, rng):
    g_dir = rng.standard_normal(D)
    g = gap * g_dir / np.linalg.norm(g_dir)
    yr = rng.standard_normal((REPS, D))
    ys = yr + g
    yhat_r = yr + rng.standard_normal((REPS, D)) * SIGMA / math.sqrt(n_r)
    yhat_s = ys + rng.standard_normal((REPS, D)) * SIGMA / math.sqrt(N_S)
    e_real, e_sim = [], []
    for b in BETAS:
        theta = (yhat_s + b * yhat_r) / (1 + b)
        e_real.append(float(np.mean(np.sum((theta - yr) ** 2, axis=1)) / D))
        e_sim.append(float(np.mean(np.sum((theta - ys) ** 2, axis=1)) / D))
    return np.array(e_real), np.array(e_sim)


def main():
    rng = np.random.default_rng(SEED)
    out = {"setup": {"D": D, "sigma": SIGMA, "N_sim": N_S, "gaps": GAPS, "N_real": N_REAL,
                     "reps": REPS, "beta_grid": [float(BETAS[0]), float(BETAS[-1]), len(BETAS)], "seed": SEED},
           "cases": []}
    for n_r in N_REAL:
        for gap in GAPS:
            beta_star, e_star, c, b = theory(gap, n_r)
            e_real, e_sim = simulate(gap, n_r, rng)
            i_min = int(np.argmin(e_real))
            case = {
                "N_real": n_r, "gap": gap,
                "beta_star_theory": round(beta_star, 4),
                "beta_star_mc_argmin": round(float(BETAS[i_min]), 4),
                "E_star_theory": round(e_star, 4),
                "E_star_mc": round(float(e_real[i_min]), 4),
                "E_sim_only_beta0_theory": round(c, 4),
                "E_real_only_beta_inf_theory": round(b, 4),
                "E_beta1_mc": round(float(np.interp(0.0, np.log10(BETAS), e_real)), 4),
                "E_sim_error_at_beta_star_mc": round(float(e_sim[i_min]), 4),
                "E_sim_error_at_beta1_mc": round(float(np.interp(0.0, np.log10(BETAS), e_sim)), 4),
                "curve_beta": [round(float(x), 5) for x in BETAS[::4]],
                "curve_E_real": [round(x, 4) for x in e_real[::4]],
                "curve_E_sim": [round(x, 4) for x in e_sim[::4]],
            }
            out["cases"].append(case)
            print(f"N_r={n_r:3d} G={gap:3.1f}  beta*(theory)={beta_star:8.3f}  argmin(MC)={BETAS[i_min]:8.3f}  "
                  f"E*(theory)={e_star:.4f} E*(MC)={e_real[i_min]:.4f}  E(beta=0)={c:.4f}  E(beta->inf)={b:.4f}")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
