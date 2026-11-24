"""LiMA 부록 A.1의 I2SB 사후분포 주장과 식 5의 시작 잡음을 숫자로 확인하는 작은 토이 (numpy, 약 10초).

검사 항목
 (a) 식 (부록 A.1): 평균 mu_t = [sb^2 X0 + s^2 X1] / (s^2 + sb^2), 분산 Sigma_t = s^2 sb^2 / (s^2 + sb^2) 가
     t = 0 에서 X0, t = 1 에서 X1 로 가고 양 끝에서 분산이 0인지, 대칭 스케줄에서 분산 최대가 t = 0.5 이고 s_1^2 / 4 인지.
 (b) 식 (5)의 시작 상태 N(Z_int, sigma_1^2 I)의 분산 sigma_1^2 (= 누적 분산 전체)이 학습 때 t -> 1 상태의 분산(0)과
     얼마나 다른지, 중간 분산 최대값과 비교.
 (c) 1차원 가우시안 문제에서 '정확한 디노이저'를 쓴 I2SB 표본추출(시작 X1)과 '가우시안 시작 + 같은 조건' 표본추출의
     단계 수별 분포 오차. 두 변형이 같은 정보(조건 z)를 받으면 1단계 해가 같다는 것을 확인한다.
     이 토이는 학습을 하지 않으므로 '학습이 쉬워진다'는 논문의 주장을 시험하지 못한다.
실행: python3 analysis/toy_bridge_posterior.py   -> analysis/toy_bridge_posterior_results.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)
R = {}


def beta(t, b_min=1e-4, b_max=0.3):
    """I2SB식 대칭 삼각 스케줄(t = 0 청정, t = 1 시작). 논문에 스케줄 값이 없어 가정한 값."""
    return b_min + (b_max - b_min) * (1 - np.abs(2 * t - 1))


tt = np.linspace(0, 1, 100001)
cum = np.concatenate([[0], np.cumsum((beta(tt[1:]) + beta(tt[:-1])) / 2 * np.diff(tt))])
s2 = cum                      # sigma_t^2 = int_0^t beta
sb2 = cum[-1] - cum           # sigma_bar_t^2 = int_t^1 beta
Sig = s2 * sb2 / (s2 + sb2 + 1e-30)
w0 = sb2 / (s2 + sb2 + 1e-30)  # X0 의 가중
R["a_endpoints"] = {"w0_at_t0": float(w0[0]), "w0_at_t1": float(w0[-1]), "Sigma_t0": float(Sig[0]), "Sigma_t1": float(Sig[-1]),
                    "argmax_Sigma_t": float(tt[np.argmax(Sig)]), "max_Sigma": float(Sig.max()), "sigma1_sq_total": float(cum[-1]),
                    "max_Sigma_equals_sigma1_sq_over_4": bool(abs(Sig.max() - cum[-1] / 4) < 1e-4)}
R["b_eq5_start_noise"] = {"sigma1_sq": float(cum[-1]), "train_Sigma_at_t_to_1": float(Sig[-1]),
                          "ratio_sigma1_sq_to_max_intermediate_Sigma": float(cum[-1] / Sig.max()),
                          "note": "식 5가 쓰는 sigma_1^2 를 누적 분산 전체로 읽으면 중간 분산 최대값의 4배이고, 식 3의 t = 1 분산(0)과 다르다. 논문은 sigma_1 값을 적지 않았다"}

# (c) 1차원: x0 ~ N(0,1), 거친 의도 z = x0 + tau * eta (tau = 0.7). 목표 분포 x0 | z ~ N(z/(1+tau^2), tau^2/(1+tau^2)).
tau = 0.7
n = 20000
x0_true = rng.normal(size=n)
z = x0_true + tau * rng.normal(size=n)
m_t, v_t = z / (1 + tau ** 2), tau ** 2 / (1 + tau ** 2)


def exact_x0_hat(xt, a, b, Sg, c_mean, c_var, x1_given):
    """x_t = a*x0 + b*x1 + sqrt(Sg)*eps, x0 ~ N(c_mean, c_var) 일 때 E[x0 | x_t, x1] (x1은 관측값)."""
    y = xt - b * x1_given
    prec = 1 / c_var + a * a / Sg
    return (c_mean / c_var + a * y / Sg) / prec


def sample_bridge(N, start_with_x1=True):
    """I2SB: X1 = z 에서 시작 (조건과 시작점이 모두 z). 스텝 N, 예측 X0로 DDPM 사후 갱신."""
    ts = np.linspace(1, 0, N + 1)
    x = z.copy() if start_with_x1 else rng.normal(size=n)
    for i in range(N):
        t1, t0 = ts[i], ts[i + 1]
        s2_t0 = np.interp(t0, tt, s2)
        a2 = np.interp(t1, tt, s2) - s2_t0         # alpha^2 = int_{t0}^{t1} beta
        sg_t1 = np.interp(t1, tt, Sig)
        a_t1 = np.interp(t1, tt, w0)
        b_t1 = 1 - a_t1
        x0h = exact_x0_hat(x, a_t1, b_t1, max(sg_t1, 1e-12), m_t, v_t, z) if t1 < 1 else m_t
        # t1 = 1 에서는 x_t = x1 이므로 x0 | x1 = z 의 사전분포 평균을 쓴다 (정확한 디노이저)
        mean = a2 / (s2_t0 + a2 + 1e-30) * x0h + s2_t0 / (s2_t0 + a2 + 1e-30) * x
        var = s2_t0 * a2 / (s2_t0 + a2 + 1e-30)
        x = mean + np.sqrt(var) * rng.normal(size=n)
    return x


def sample_gauss_init(N):
    """가우시안 시작 + 같은 조건 z: x1 ~ N(0,1) 독립인 같은 틀(조건은 x0의 사전분포로만 들어감)."""
    ts = np.linspace(1, 0, N + 1)
    x1 = rng.normal(size=n)
    x = x1.copy()
    for i in range(N):
        t1, t0 = ts[i], ts[i + 1]
        s2_t0 = np.interp(t0, tt, s2)
        a2 = np.interp(t1, tt, s2) - s2_t0
        sg_t1 = np.interp(t1, tt, Sig)
        a_t1 = np.interp(t1, tt, w0)
        b_t1 = 1 - a_t1
        x0h = exact_x0_hat(x, a_t1, b_t1, max(sg_t1, 1e-12), m_t, v_t, x1) if t1 < 1 else m_t
        mean = a2 / (s2_t0 + a2 + 1e-30) * x0h + s2_t0 / (s2_t0 + a2 + 1e-30) * x
        var = s2_t0 * a2 / (s2_t0 + a2 + 1e-30)
        x = mean + np.sqrt(var) * rng.normal(size=n)
    return x


def err(x):
    """x0 | z 의 목표 N(m_t, v_t)에 대한 표준화 잔차의 평균과 표준편차(목표는 0, 1), x0_true 에 대한 MSE."""
    r = (x - m_t) / np.sqrt(v_t)
    return {"resid_mean": round(float(r.mean()), 3), "resid_std": round(float(r.std()), 3), "mse_to_x0": round(float(((x - x0_true) ** 2).mean()), 3)}


res = {}
for N in [1, 2, 5, 10, 20]:
    res[str(N)] = {"bridge_from_X1": err(sample_bridge(N)), "gaussian_init_same_condition": err(sample_gauss_init(N))}
R["c_exact_denoiser_steps"] = res
R["c_reference"] = {"posterior_var": round(float(v_t), 3), "mse_floor_if_sampling_exact_posterior": round(2 * float(v_t), 3),
                    "mse_posterior_mean_only": round(float(v_t), 3)}
(HERE / "toy_bridge_posterior_results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(R["a_endpoints"], indent=1))
print(json.dumps(R["b_eq5_start_noise"], indent=1, ensure_ascii=False))
for N, v in res.items():
    print(N, v)
print(R["c_reference"])
