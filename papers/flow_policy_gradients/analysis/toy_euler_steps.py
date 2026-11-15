"""토이 1: 유한 적분 단계(Euler)가 1차원 flow 정책의 출력에 주는 오차.

목적: 논문이 주장하는 "적분 단계를 줄여도 성능 손실이 작다"(부록 B.2.2: zero-sampling 5단계, 부록 B.1.1: 8/16 단계 불안정)가
      가장 단순한 설정에서 어떤 크기의 오차인지 본다. 논문 결과의 재현이 아니다.

설정(가정, 임의 설정):
- 목표 행동 분포는 1차원 두 봉우리 혼합 a ~ 0.5 N(-1, 0.2^2) + 0.5 N(+1, 0.2^2) 이다.
  단봉 가우시안은 0 초기화 궤적이 직선이라 적분 오차가 0이 되므로(처음 시도에서 확인) 혼합으로 바꿨다.
- 논문 코드(actor_critic.py)와 같은 시간 규약: x_t = t*eps + (1-t)*a, t=1이 노이즈, t=0이 데이터.
  정답 속도장 v(x,t) = E[eps - a | x_t] 는 혼합 성분별 선형 가우시안 회귀로 닫힌 형식이다.
  성분 k에 대해 y = x_t ~ N((1-t) mu_k, V_k), V_k = (1-t)^2 s_k^2 + t^2.
  책임도 r_k(y) ∝ pi_k N(y; (1-t) mu_k, V_k),
  E[eps | y, k] = t (y - (1-t) mu_k) / V_k,  E[a | y, k] = mu_k + (1-t) s_k^2 (y - (1-t) mu_k) / V_k.
  v = sum_k r_k (E[eps|y,k] - E[a|y,k]).
- Euler 적분: t=1에서 t=0까지 K 단계, x <- x + v(x,t) dt, dt = -1/K.
- 두 초기화: 무작위(eps ~ N(0,1), 학습 때) 와 0(zero-sampling, 평가 때).

측정:
- 무작위 초기화: K 단계 출력의 평균, 표준편차, 봉우리 질량(a>0 비율)이 K=1024 기준에서 얼마나 벗어나는지.
- 0 초기화: K 단계의 결정적 출력이 K=1024 출력에서 얼마나 떨어져 있는지.
- 비교용 단봉 가우시안: 0 초기화 오차가 0인지 확인(이론: 평균 궤적이 직선이라 정확).

실행: python3 -I analysis/toy_euler_steps.py  (numpy만 사용, 1 프로세스, 수 초)
결과: analysis/toy_euler_steps_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_euler_steps_results.json"
PI = np.array([0.35, 0.65])
MU = np.array([-1.0, 1.0])
SD = np.array([0.2, 0.2])
N = 200_000
REF_K = 1024
STEPS = [1, 2, 4, 5, 8, 10, 16, 32, 64]


def velocity_mix(x, t, pi=PI, mu=MU, sd=SD):
    """정답 속도장 E[eps - a | x_t] (혼합 가우시안 목표)."""
    V = (1 - t) ** 2 * sd**2 + t**2  # (K,)
    d = x[:, None] - (1 - t) * mu[None, :]  # (N, K)
    logw = np.log(pi)[None, :] - 0.5 * d**2 / V[None, :] - 0.5 * np.log(V)[None, :]
    logw -= logw.max(axis=1, keepdims=True)
    r = np.exp(logw)
    r /= r.sum(axis=1, keepdims=True)
    e_eps = t * d / V[None, :]
    e_a = mu[None, :] + (1 - t) * sd[None, :] ** 2 * d / V[None, :]
    return (r * (e_eps - e_a)).sum(axis=1)


def integrate(x0, K):
    x = x0.copy()
    ts = np.linspace(1.0, 0.0, K + 1)
    for i in range(K):
        t = ts[i]
        dt = ts[i + 1] - ts[i]  # 음수
        x = x + velocity_mix(x, t) * dt
    return x


def gauss_zero_check():
    """단봉 가우시안(m=0.5, s=0.3)에서 0 초기화 적분 오차가 0인지 확인."""
    m, s = 0.5, 0.3

    def v(x, t):
        V = (1 - t) ** 2 * s**2 + t**2
        return (x - (1 - t) * m) * (t - (1 - t) * s**2) / V - m

    out = {}
    for K in [1, 2, 5, 10]:
        x = np.zeros(1)
        ts = np.linspace(1.0, 0.0, K + 1)
        for i in range(K):
            x = x + v(x, ts[i]) * (ts[i + 1] - ts[i])
        out[K] = float(x[0] - m)
    return out


def main():
    rng = np.random.default_rng(0)
    eps = rng.standard_normal(N)
    ref_rand = integrate(eps, REF_K)
    ref_zero = integrate(np.zeros(1), REF_K)[0]
    ref_pos = float((ref_rand > 0).mean())

    rows = []
    for K in STEPS:
        out = integrate(eps, K)
        z = integrate(np.zeros(1), K)[0]
        rows.append({
            "K": K,
            "random_mean": float(out.mean()),
            "random_std": float(out.std()),
            "random_pos_mass": float((out > 0).mean()),
            "random_mean_err": float(out.mean() - ref_rand.mean()),
            "random_std_err": float(out.std() - ref_rand.std()),
            "random_pos_mass_err": float((out > 0).mean() - ref_pos),
            "zero_action": float(z),
            "zero_err": float(z - ref_zero),
        })

    res = {
        "setup": {"pi": PI.tolist(), "mu": MU.tolist(), "sd": SD.tolist(), "N": N, "ref_K": REF_K, "steps": STEPS,
                  "time_convention": "x_t = t*eps + (1-t)*a (논문 코드 actor_critic.py와 같은 규약)",
                  "seed": 0},
        "ref_K1024": {"random_mean": float(ref_rand.mean()), "random_std": float(ref_rand.std()),
                      "random_pos_mass": ref_pos, "zero_action": float(ref_zero)},
        "gauss_zero_check_K_to_err": gauss_zero_check(),
        "rows": rows,
    }
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"ref K={REF_K}: mean={ref_rand.mean():.4f} std={ref_rand.std():.4f} pos={ref_pos:.4f} zero={ref_zero:.4f}")
    print("gauss zero-init error by K:", res["gauss_zero_check_K_to_err"])
    for r in rows:
        print(f"K={r['K']:3d} mean_err={r['random_mean_err']:+.4f} std_err={r['random_std_err']:+.4f} "
              f"pos_err={r['random_pos_mass_err']:+.4f} zero_action={r['zero_action']:+.4f} zero_err={r['zero_err']:+.4f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
