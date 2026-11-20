"""토이: 예측된 미래 표현은 행동을 얼마나 정확히 알려 주는가 (numpy, 1 프로세스, 수 초).

질문: 정책이 현재 관측만 보고 행동을 맞추는 것보다, 예측된 다음 상태(영상 모델 출력)를 함께 보면 얼마나 나은가?
        예측 오차 σ가 커지면 그 이득은 언제 사라지는가?

모형 (2차원 점 질량, 단위 없음):
  a_t = ρ a_{t-1} + sqrt(1-ρ²) ε_t,  x_{t+1} = x_t + a_t + w_t,  w_t ~ N(0, 0.05²)
  관측 A: 현재 위치 x_t 만 (VPP처럼 한 프레임)
  관측 B: x_t 와 직전 위치 x_{t-1} (속도 정보가 있는 경우)
  예측 입력: 영상 모델이 낸 다음 위치 x̂_{t+1} = x_{t+1} + σ·N(0, I) (i.i.d. 가정)
  평가: 선형 회귀(절편 포함)로 a_t를 맞추고 테스트 MSE(두 차원 평균)를 잰다.
  두 가지 시험: (i) 학습은 σ_tr = 0.1, 시험은 σ_te 를 바꿈(분포 이동). (ii) 학습과 시험 모두 σ 로 맞춤(대조).
  기준선은 예측 입력 없이 현재 관측만 쓴 같은 회귀다.
  선형 회귀는 예측 입력의 계수를 0으로 줄일 수 있으므로, (ii)에서는 기준선을 넘지 못하는 경우가 없다고 예상한다.

이 토이는 논문의 주장을 검증하지 않는다. 역동역학 이득의 크기가 예측 오차와 행동의 예측 불가능성에
어떻게 달려 있는지 보는 용도다. 결과는 analysis/toy_future_error_results.json 에 저장한다.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("toy_future_error_results.json")
SEED = 0
N_EP, T_EP = 2000, 50          # 에피소드 수, 길이 (표본 약 10^5 전이)
W_STD = 0.05                    # 동역학 잡음
SIGMA_TR = 0.1                  # 학습 때 예측 오차 (영상 모델이 학습 분포에서 맞힌다고 가정)
SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]  # 시험 때 예측 오차
RHOS = [0.0, 0.5, 0.9, 0.99]


def simulate(rho, rng):
    a = np.zeros((N_EP, T_EP + 2, 2))
    x = np.zeros((N_EP, T_EP + 3, 2))
    x[:, 0] = rng.normal(0, 1.0, size=(N_EP, 2))
    eps = rng.normal(size=(N_EP, T_EP + 2, 2))
    for t in range(T_EP + 2):
        prev = a[:, t - 1] if t > 0 else np.zeros((N_EP, 2))
        a[:, t] = rho * prev + np.sqrt(1 - rho ** 2) * eps[:, t]
        w = rng.normal(0, W_STD, size=(N_EP, 2))
        x[:, t + 1] = x[:, t] + a[:, t] + w
    # 정책 입력 시점 t = 1..T_EP (x_{t-1}은 t≥1에서 존재)
    t_idx = np.arange(1, T_EP + 1)
    return a[:, t_idx].reshape(-1, 2), x[:, t_idx].reshape(-1, 2), x[:, t_idx - 1].reshape(-1, 2), \
        x[:, t_idx + 1].reshape(-1, 2)


def fit(F, y):
    F1 = np.hstack([F, np.ones((len(F), 1))])
    beta, *_ = np.linalg.lstsq(F1, y, rcond=None)
    return beta


def mse(beta, F, y):
    F1 = np.hstack([F, np.ones((len(F), 1))])
    return float(np.mean((F1 @ beta - y) ** 2))


def main():
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED, "episodes": N_EP, "T": T_EP, "w_std": W_STD, "sigma_train": SIGMA_TR,
           "sigmas_test": SIGMAS, "rhos": RHOS, "rows": []}
    for rho in RHOS:
        a, x_t, x_prev, x_next = simulate(rho, rng)
        cut = int(0.8 * len(a))
        tr, te = slice(0, cut), slice(cut, len(a))
        for setting, cur in [("A_current_only", x_t), ("B_current_plus_prev", np.hstack([x_t, x_prev]))]:
            base = fit(cur[tr], a[tr])
            base_mse = mse(base, cur[te], a[te])
            # 학습 때 예측이 깨끗하다고 가정(σ_tr), 시험 때 예측 오차만 키운다 (분포 이동)
            xhat_tr = x_next[tr] + SIGMA_TR * rng.normal(size=x_next[tr].shape)
            beta_mis = fit(np.hstack([cur[tr], xhat_tr]), a[tr])
            # 학습과 시험 모두 같은 σ로 맞춘 경우 (대조군)
            row = {"rho": rho, "setting": setting, "baseline_mse": base_mse, "var_a": float(np.var(a)),
                   "mismatch": [], "matched": []}
            for sig in SIGMAS:
                xhat_te = x_next[te] + sig * rng.normal(size=x_next[te].shape)
                row["mismatch"].append({"sigma_test": sig,
                                        "mse": mse(beta_mis, np.hstack([cur[te], xhat_te]), a[te])})
                xhat_tr_s = x_next[tr] + sig * rng.normal(size=x_next[tr].shape)
                beta_m = fit(np.hstack([cur[tr], xhat_tr_s]), a[tr])
                row["matched"].append({"sigma": sig,
                                       "mse": mse(beta_m, np.hstack([cur[te], xhat_te]), a[te])})
            cross = None
            for p_, q_ in zip(row["mismatch"][:-1], row["mismatch"][1:]):
                if p_["mse"] < base_mse <= q_["mse"]:
                    s0, s1, m0, m1 = p_["sigma_test"], q_["sigma_test"], p_["mse"], q_["mse"]
                    cross = round(s0 + (base_mse - m0) * (s1 - s0) / (m1 - m0), 3)
                    break
            row["crossover_sigma_test"] = cross
            row["matched_always_below_baseline"] = all(m["mse"] <= base_mse + 1e-9 for m in row["matched"])
            res["rows"].append(row)
    json.dump(res, OUT.open("w", encoding="utf-8"), indent=1)
    print("rho  setting                base_mse  crossover(sigma_test)  matched<=base")
    for r in res["rows"]:
        print(f"{r['rho']:.2f} {r['setting']:21s} {r['baseline_mse']:.4f}   {str(r['crossover_sigma_test']):>8s}"
              f"              {r['matched_always_below_baseline']}")


if __name__ == "__main__":
    main()
