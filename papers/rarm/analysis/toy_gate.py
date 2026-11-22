"""토이 2: 신뢰 게이트 Δ의 민감도 (numpy, 수 초).

논문 식 (3), (4), (5): 게이트는 g(c_t, r_j*) >= m_j* - Δ 일 때만 진행을 갱신하고, 보상은 1{p_t > p_(t-1)}이다.
Δ = 0.5 하나를 모든 과제에 쓴다. 이 토이는 점수 척도가 가정이므로 결론을 논문의 실제 수치로 옮기면 안 된다.

모형(우리 가정):
- 참조 클립 M = 60개. 자기 일치 점수 m_j = 3.0 + N(0, 0.3^2) (눈에 띄는 동작일수록 높음).
- 롤아웃 클립 t의 참조상 실제 위치 u_t. 점수 g(c_t, r_j) = m_j - q - kappa*|j - u_t| + N(0, sigma^2).
  q = 롤아웃과 참조의 외형 격차(성공 롤아웃은 q_s, 그럴듯하지만 틀린 실패 롤아웃은 q_f > q_s).
- 성공: u_t가 0에서 M-1까지 선형(속도 변동). 실패: 앞쪽 30%만 따라가고, 이후 겉모습의 최근접 위치가 참조 후반부 어디든 될 수 있다.
- 이벤트 수 = 진행 갱신 횟수(상한 M). 게이트 없음 = 모든 최근접 일치를 수락(Δ = inf).
출력: 성공 이벤트 E_s, 실패 이벤트 E_f, 비율 E_f/E_s (게이트 없음, Δ = 0.25, 0.5, 1.0, 2.0).

실행: python3 analysis/toy_gate.py   → analysis/toy_gate_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("toy_gate_results.json")
M, T, KAPPA, SIGMA = 60, 120, 0.12, 0.25


def run(q, delta, kind, rng):
    m = 3.0 + 0.3 * rng.standard_normal(M)
    j = np.arange(M)
    if kind == "success":
        u = np.clip(np.cumsum(np.abs(1 + 0.5 * rng.standard_normal(T))) / T * (M - 1) * 0.97, 0, M - 1)
        u = np.sort(u)
    else:
        # 앞 30%는 따라가고, 이후에는 겉모습이 참조의 후반 클립 어디와도 닮을 수 있다(그럴듯하지만 틀림)
        u = np.where(np.arange(T) < 0.3 * T, np.arange(T) / (0.3 * T) * 0.3 * (M - 1),
                     rng.uniform(0.3 * (M - 1), M - 1, size=T))
    prev, ev = -1, 0
    for t in range(T):
        g = m - q - KAPPA * np.abs(j - u[t]) + SIGMA * rng.standard_normal(M)
        k = int(np.argmax(g))
        if g[k] >= m[k] - delta and k > prev:
            prev = k
            ev += 1
    return ev


rng = np.random.default_rng(0)
res = {"model": {"M": M, "T": T, "kappa": KAPPA, "sigma": SIGMA, "self_match_mean": 3.0}, "grid": []}
for q_s, q_f in ((0.1, 0.3), (0.1, 0.6), (0.1, 1.0), (0.3, 0.6)):
    for delta in (0.25, 0.5, 1.0, 2.0, float("inf")):
        es = np.mean([run(q_s, delta, "success", rng) for _ in range(200)])
        ef = np.mean([run(q_f, delta, "failure", rng) for _ in range(200)])
        res["grid"].append({"q_success": q_s, "q_fail": q_f, "delta": "inf" if delta == float("inf") else delta,
                            "E_success": round(float(es), 1), "E_fail": round(float(ef), 1),
                            "ratio": round(float(ef / es), 2) if es > 0 else None})
# 잡음의 최댓값이 문턱을 올린다: E[max of M N(0, sigma^2)] ≈ sigma*sqrt(2 ln M) (점수 척도 단위). Δ = 0.5와 같은 크기다.
from math import log, sqrt
res["max_noise_shift"] = {str(m_): round(SIGMA * sqrt(2 * log(m_)), 2) for m_ in (20, 60, 180)}
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("max_noise_shift", res["max_noise_shift"])
print("q_s q_f  Δ      E_s   E_f   E_f/E_s")
for g in res["grid"]:
    print(f"{g['q_success']:.1f} {g['q_fail']:.1f}  {str(g['delta']):>4}  {g['E_success']:5.1f} {g['E_fail']:5.1f}  {g['ratio']}")
