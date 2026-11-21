"""논문 식 (1)의 부호 규약과 Euler 갱신식이 서로 맞는지 보는 1차원 토이 (numpy, 1초 안팎).

논문 2.1절: A^tau = tau*A + (1-tau)*eps, 목표 V ~ eps - A (식 1), 추론은 A <- A + (1/K) V (K=4).
공식 코드(n1-release, flow_matching_action_head.py): velocity = actions - noise, 손실 mse(pred, velocity),
추론은 actions <- actions + dt * pred.

데이터 A ~ N(m, s^2) 이면 조건부 기댓값 E[A - eps | x_tau = x] 가 닫힌 형태이므로, 이를 "완벽히 학습된 V"로 쓴다.
  Var(x) = tau^2 s^2 + (1-tau)^2,  E[A|x] = m + tau s^2 (x - tau m)/Var(x),  E[eps|x] = (1-tau)(x - tau m)/Var(x)
  v*(x,tau) = E[A - eps | x] = m + (tau s^2 - (1-tau)) (x - tau m)/Var(x)

두 규약을 비교한다.
  (a) 코드 규약: 학습 목표 A - eps = v*, 갱신 x <- x + dt v*
  (b) 논문 식 (1) 문자 그대로: 학습 목표 eps - A = -v*, 갱신 x <- x + dt (학습된 값) = x - dt v*
각각 K=4 Euler로 x0 ~ N(0,1) 에서 tau=1까지 적분하고, 결과 분포가 N(m, s^2) 에 가까운지 본다.

실행: python3 analysis/toy_flow_sign.py -> analysis/toy_flow_sign_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_flow_sign_results.json"
M, S = 1.0, 0.1          # 데이터 평균과 표준편차 (A ~ N(M, S^2))
N = 200_000
rng = np.random.default_rng(0)


def v_star(x, tau):
    var = tau ** 2 * S ** 2 + (1 - tau) ** 2
    return M + (tau * S ** 2 - (1 - tau)) * (x - tau * M) / var


def integrate(K, sign):
    x = rng.standard_normal(N)
    dt = 1.0 / K
    for k in range(K):
        tau = k / K
        x = x + dt * sign * v_star(x, tau)
    return x


res = {"target": {"mean": M, "sd": S}, "runs": {}}
for K in (4, 16, 64):
    for name, sign in (("code_convention", +1.0), ("paper_eq1_literal", -1.0)):
        x = integrate(K, sign)
        res["runs"][f"K={K}:{name}"] = {"mean": round(float(x.mean()), 4), "sd": round(float(x.std()), 4)}

json.dump(res, OUT.open("w", encoding="utf-8"), indent=1)
for k, v in res["runs"].items():
    print(k, v)
