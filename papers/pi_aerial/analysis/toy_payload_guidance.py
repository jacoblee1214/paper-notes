"""AirVLA 페이로드 인식 가이던스(논문 식 2, 3, 6, 8, 9)의 CPU toy 두 개.

T1: 가중치 alpha(식 8, 9)가 파지 상태별로 어떤 값을 갖는지 계산한다.
    논문은 "closedness/openness 점수"의 정의(조리개 -> 점수 사상)를 쓰지 않아, g를 [0, 1] 개방도로 두고
    c_meas = 1 - g, o_meas = g 로 가정한다(가정).
T2: 1차원 가우시안 정책(해석적 흐름 매칭 속도장)에서 식 3의 가이던스 v - s * xi 가 식 2의 사후 분포
    p(A) * exp(-Phi(A)) 의 평균을 얼마나 만드는지 본다. 논문이 정하지 않은 s(tau)와 정책 분산 sigma가 결과를 정한다.
    기본 설정: 오일러 10단계(부록 A.3), lambda_z = 0.5(부록), z_des 오프셋 0.15(부록, 단위 가정 필요).

실행: python3 analysis/toy_payload_guidance.py   (numpy, 수 초). 결과: analysis/toy_payload_guidance_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_payload_guidance_results.json"
RES = {}


def clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def alpha_gate(u_hist, g):
    """식 8, 9. u_hist: 직전 청크 마지막 K개 그리퍼 명령(+1 닫기), g: 측정 개방도(1=열림)."""
    c_int = float(np.mean([clip01((u + 1) / 2) for u in u_hist]))
    o_int = float(np.mean([clip01(1 - (u + 1) / 2) for u in u_hist]))
    c_meas, o_meas = clip01(1 - g), clip01(g)
    o_flag = clip01(0.5 * o_int + 0.5 * o_meas)
    return clip01((1 - o_flag) * c_int * c_meas), dict(c_int=c_int, o_int=o_int, c_meas=c_meas, o_meas=o_meas, o_flag=o_flag)


def toy_alpha():
    states = {
        "열린 채 대기 (u=-1, g=1.0)": ([-1] * 4, 1.0),
        "닫기 명령 막 내림, 아직 열림 (u=+1, g=1.0)": ([1] * 4, 1.0),
        "물체를 쥐고 비행 (u=+1, g=0.3)": ([1] * 4, 0.3),
        "물체를 쥐고 비행, 명령 혼합 (u=[1,1,1,-1], g=0.3)": ([1, 1, 1, -1], 0.3),
        "빈 손으로 완전히 닫음 (u=+1, g=0.0)": ([1] * 4, 0.0),
        "놓는 중 (u=-1, g=0.3)": ([-1] * 4, 0.3),
    }
    out = {}
    for name, (u, g) in states.items():
        a, parts = alpha_gate(u, g)
        out[name] = {"alpha": round(a, 3), **{k: round(v, 3) for k, v in parts.items()}}
    return out


def guided_sample(mu, sigma, lam, zdes, s_fn, n=20000, steps=10, seed=0, w=1.0):
    """x_tau = (1-tau) x0 + tau x1, x0 ~ N(0,1), x1 ~ N(mu, sigma^2) 의 해석적 속도장 + 식 3 가이던스.

    Phi(A) = (lam * w / 2) (A - zdes)^2. A_hat = E[x1 | x_tau]. xi = dA_hat/dx_tau * dPhi/dA_hat.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    dt = 1.0 / steps
    for i in range(steps):
        tau = i * dt
        V = (1 - tau) ** 2 + (tau * sigma) ** 2
        a_hat = mu + tau * sigma**2 / V * (x - tau * mu)
        e_x0 = (1 - tau) / V * (x - tau * mu)
        v = a_hat - e_x0
        jac = tau * sigma**2 / V
        xi = jac * lam * w * (a_hat - zdes)
        x = x + dt * (v - s_fn(tau) * xi)
    return x


def toy_guidance():
    mu, zdes, lam = 0.0, 0.15, 0.5
    out = {"설정": {"mu(사전 평균 z)": mu, "z_des": zdes, "lambda_z": lam, "steps": 10}}
    rows = []
    for sigma in (0.05, 0.1, 0.3, 1.0):
        post_mean = (mu / sigma**2 + lam * zdes) / (1 / sigma**2 + lam)  # 식 2의 정확한 사후 평균
        frac_exact = (post_mean - mu) / (zdes - mu)
        r = {"sigma(정책 z 표준편차)": sigma, "식2 정확 사후 평균": round(post_mean, 5),
             "목표까지 이동 비율(정확)": round(frac_exact, 5)}
        for name, s_fn in (("s=0", lambda t: 0.0), ("s=1", lambda t: 1.0), ("s=1/(1-tau+0.1)", lambda t: 1.0 / (1 - t + 0.1)),
                           ("s=10", lambda t: 10.0)):
            x = guided_sample(mu, sigma, lam, zdes, s_fn)
            r[f"{name} 평균"] = round(float(x.mean()), 5)
            r[f"{name} 표준편차"] = round(float(x.std()), 5)
        rows.append(r)
    out["rows"] = rows
    # 정확 사후 분산도 같이: 1/(1/sigma^2 + lam)
    return out


def main():
    RES["T1_alpha"] = toy_alpha()
    RES["T2_guidance"] = toy_guidance()
    OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(RES["T1_alpha"], ensure_ascii=False, indent=1))
    for r in RES["T2_guidance"]["rows"]:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
