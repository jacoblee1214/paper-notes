"""TacSL 논문 식 (5)-(9) 대수 검사 (numpy, 수 초).

논문 식: 스프링-댐퍼 접촉 임펄스 lambda = dt * max(-kappa*eps+ - c*epsdot+, 0)를
eps+ = eps + dt*epsdot+, epsdot+ = epsdot + lambda/m 로 풀면
lambda = (-dt*kappa*eps - alpha*epsdot) / (1 + alpha/m),  alpha = dt*(dt*kappa + c)  (식 9).
여기서는 식 9가 이산 방정식의 해인지 무작위 입력에서 확인한다(음의 힘 조건 max는 활성 영역만 검사).
"""
import json
import os

import numpy as np

rng = np.random.default_rng(0)
N = 20000
dt = rng.uniform(1e-4, 2e-2, N)
kappa = rng.uniform(150, 350, N)       # 논문 부록 B의 무작위화 범위와 같은 크기
c = rng.uniform(0.0, 1.0, N)           # 논문 표 V의 감쇠 범위
m = rng.uniform(0.05, 2.0, N)          # 유효 질량 (임의 단위)
eps = rng.uniform(-2e-3, 2e-3, N)      # 접촉 거리 (m)
epsdot = rng.uniform(-0.5, 0.5, N)     # 분리 속도 (m/s)

alpha = dt * (dt * kappa + c)
lam = (-dt * kappa * eps - alpha * epsdot) / (1 + alpha / m)   # 식 (9)

# 식 (5)-(7)로 되돌려 대입: 잔차는 (5)의 우변과 좌변의 차
epsdot_plus = epsdot + lam / m
eps_plus = eps + dt * epsdot_plus
rhs = dt * (-kappa * eps_plus - c * epsdot_plus)
resid = lam - rhs

active = rhs > 0                        # max(., 0)가 활성인 경우만 식 (5)가 등식
res_active = np.abs(resid[active]) if active.any() else np.array([0.0])
max_resid_active = float(res_active.max())
rel_resid_active = float((res_active / (np.abs(lam[active]) + 1e-30)).max()) if active.any() else 0.0

out = {
    "n_samples": N,
    "n_active_contact": int(active.sum()),
    "max_abs_residual_active_contact": max_resid_active,
    "max_rel_residual_active_contact": rel_resid_active,
    "note": "활성 접촉에서 식 (9)는 이산 방정식 (5)-(7)을 기계 정밀도로 만족함. "
            "비활성(max=0) 샘플은 lambda=(식 9)가 양수일 때만 물리적이라 별도 검사 대상.",
}
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "toy_implicit_contact_results.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=1))
