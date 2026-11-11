"""UVT 선형 toy: 동역학 코드가 붙은 공동 목표가 행동 정확도를 올리는 조건 (numpy 1스레드, 수십 초).
실행: python3 analysis/uvt_toy.py  → analysis/uvt_toy_results.json

모형 (논문 UVT의 선형 축소판):
- x ~ N(0, I_p), p = 20. 행동 신호 a* = W x (rank q = 4), 관측 행동 a = a* + σ_a ε (ε는 x와 무관한 운동 잡음).
- 동역학 코드 d = sqrt(ρ) S x + sqrt(1-ρ) ξ, S는 행동 신호와 같은 부분공간에 있는 단위분산 선형 사상.
  따라서 R²(d | x) = ρ (모집단). ρ는 "동역학 코드가 관측으로부터 얼마나 예측 가능한가"를 조절한다.
- 목표 세 가지 (모두 선형 정책 OLS: x → 목표):
  (F) 전체 행동 OLS (압축 없음, 기준선)
  (A_r) 행동만 PCA로 r차원 압축 (행동 전용 목표, 압축 대조군)
  (J_r) [행동; 동역학]을 함께 PCA로 r차원 압축 (UVT의 선형 유사물: 선형 Gaussian VAE는 PPCA와 같음)
  디코더는 고정된 사영(P^T)이다. 논문은 디코더를 학습하지만, 선형 toy에서는 두 방법에 같은 디코더 구조를 준다.
- 평가: 테스트 x에서 ||â(x) - a*(x)||² 평균 (잡음 없는 목표에 대한 오차).
- 이득 = MSE(A_r) - MSE(J_r). 양수면 공동 목표가 도움. 시드 30개의 평균과 표준오차.

검증 항목:
- 압축이 없으면(r = k) 선형 OLS는 목표를 가역 선형 변환해도 같은 예측을 낸다. 따라서 r = k에서 J와 A의 차이는 0이어야 한다(수치 오차 이내).
- 공동 목표 이득은 r이 작고 ρ가 클 때 양수여야 하고, ρ = 0이면 동역학 차원이 잡음 방향을 차지해 음수일 수 있다.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "uvt_toy_results.json"

P_DIM, K_DIM, M_DIM, Q_RANK = 20, 12, 6, 4
N_TEST = 4000
SEEDS = 30
RHOS = [0.0, 0.25, 0.5, 0.75, 0.95]
NS = [30, 60, 200]
RS = [1, 2, 3, 4, 6]


def make_world(seed):
    rng = np.random.default_rng(1000 + seed)
    M = rng.standard_normal((K_DIM, Q_RANK))
    M /= np.linalg.norm(M, axis=1, keepdims=True)          # 행마다 단위 노름 -> W x 의 분산 1
    B = np.linalg.qr(rng.standard_normal((P_DIM, Q_RANK)))[0].T   # 직교 행 (q × p)
    W = M @ B                                                 # k × p
    T = rng.standard_normal((M_DIM, Q_RANK))
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    S = T @ B                                                 # m × p, 단위분산 예측 부분
    return W, S


def sample(rng, W, S, n, rho, sig_a):
    x = rng.standard_normal((n, P_DIM))
    a_star = x @ W.T
    a = a_star + sig_a * rng.standard_normal((n, K_DIM))
    d = np.sqrt(rho) * (x @ S.T) + np.sqrt(1 - rho) * rng.standard_normal((n, M_DIM))
    return x, a_star, a, d


def ols(X, Y):
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return coef


def pca_top(Y, r):
    Yc = Y - Y.mean(axis=0)
    _, _, Vt = np.linalg.svd(Yc, full_matrices=False)
    return Vt[:r].T  # (dim × r)


def fit_predict(x_tr, a_tr, d_tr, x_te, method, r, sig_scale):
    """method: 'F' 전체 OLS, 'A' 행동 PCA, 'J' 공동 PCA. 예측 행동 반환. 절편 포함."""
    x_tr = np.c_[x_tr, np.ones(len(x_tr))]
    x_te = np.c_[x_te, np.ones(len(x_te))]
    if method == "F":
        return x_te @ ols(x_tr, a_tr)
    if method == "A":
        P = pca_top(a_tr, r)
        mu = a_tr.mean(axis=0)
        U = (a_tr - mu) @ P
        Uhat = x_te @ ols(x_tr, U)
        return Uhat @ P.T + mu
    # J: 행동과 동역학을 같은 단위 스케일로 놓은 뒤 PCA
    Y = np.concatenate([a_tr / sig_scale[0], d_tr / sig_scale[1]], axis=1)
    mu = Y.mean(axis=0)
    P = pca_top(Y, r)
    U = (Y - mu) @ P
    Uhat = x_te @ ols(x_tr, U)
    Yhat = Uhat @ P.T + mu
    return Yhat[:, :K_DIM] * sig_scale[0]


def run(sig_a=1.0):
    """각 (n, r)에서 ρ를 바꿔 공동 목표(J)를 평가한다. 같은 시드에서 x, 운동 잡음, ξ는 모든 ρ에서 같다.
    동역학 고유 이득 = MSE_J(ρ=0, 잡음 동역학 블록) - MSE_J(ρ). 이것은 '차원만 같고 예측 불가능한 블록'과의 짝비교이다."""
    results = {}
    for n in NS:
        for rho in RHOS:
            key = f"n={n},rho={rho}"
            acc = {"F": [], "A": {r: [] for r in RS}, "J": {r: [] for r in RS}, "J0": {r: [] for r in RS}}
            equiv_gap = []
            for seed in range(SEEDS):
                W, S = make_world(seed)
                rng = np.random.default_rng(seed)
                x_tr, _, a_tr, d_tr = sample(rng, W, S, n, rho, sig_a)
                rng = np.random.default_rng(seed)
                _, _, _, d0_tr = sample(rng, W, S, n, 0.0, sig_a)   # 같은 x, 잡음, ξ에서 ρ=0 (동역학은 순수 잡음)
                rng = np.random.default_rng(seed + 500)
                x_te, a_te_star, _, _ = sample(rng, W, S, N_TEST, rho, sig_a)
                # 공동 목표의 블록 스케일: 각 블록 평균 분산을 1로 맞춤 (가정; 학습된 α에 해당하는 값은 알 수 없음)
                scale = (np.sqrt(np.mean(np.var(a_tr, axis=0))), np.sqrt(np.mean(np.var(d_tr, axis=0))))
                scale0 = (scale[0], np.sqrt(np.mean(np.var(d0_tr, axis=0))))
                acc["F"].append(float(np.mean((fit_predict(x_tr, a_tr, d_tr, x_te, "F", 0, scale) - a_te_star) ** 2)))
                for r in RS:
                    pa = fit_predict(x_tr, a_tr, d_tr, x_te, "A", r, scale)
                    pj = fit_predict(x_tr, a_tr, d_tr, x_te, "J", r, scale)
                    pj0 = fit_predict(x_tr, a_tr, d0_tr, x_te, "J", r, scale0)
                    acc["A"][r].append(float(np.mean((pa - a_te_star) ** 2)))
                    acc["J"][r].append(float(np.mean((pj - a_te_star) ** 2)))
                    acc["J0"][r].append(float(np.mean((pj0 - a_te_star) ** 2)))
                # 검증: 압축이 없으면(r = k+m) J의 행동 예측은 전체 OLS와 같아야 한다 (선형, 절편 포함)
                full_J = fit_predict(x_tr, a_tr, d_tr, x_te, "J", K_DIM + M_DIM, scale)
                equiv_gap.append(float(np.max(np.abs(full_J - fit_predict(x_tr, a_tr, d_tr, x_te, "F", 0, scale)))))
            res = {"F_mse": float(np.mean(acc["F"])), "full_rank_equiv_max_abs_gap": float(max(equiv_gap))}
            for r in RS:
                a_m, j_m, j0_m = (np.array(acc["A"][r]), np.array(acc["J"][r]), np.array(acc["J0"][r]))
                for name, v in (("A", a_m), ("J", j_m), ("J0", j0_m)):
                    res[f"{name}{r}_mse"] = float(v.mean())
                for name, g in (("gain_J_minus_A", a_m - j_m), ("dyn_gain", j0_m - j_m)):
                    res[f"{name}_r{r}"] = float(g.mean())
                    res[f"{name}_se_r{r}"] = float(g.std(ddof=1) / np.sqrt(SEEDS))
            results[key] = res
    return results


if __name__ == "__main__":
    out = {"settings": {"p": P_DIM, "k": K_DIM, "m": M_DIM, "signal_rank": Q_RANK, "n_test": N_TEST,
                        "seeds": SEEDS, "sig_a": 1.0, "rhos": RHOS, "ns": NS, "rs": RS},
           "results": run(1.0)}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"{'setting':<16}{'F mse':>8}  " + "  ".join(f"r={r}: A-J gain / dyn gain" for r in RS))
    for k, v in out["results"].items():
        cells = "  ".join(f"{v[f'gain_J_minus_A_r{r}']:+.3f}/{v[f'dyn_gain_r{r}']:+.3f}" for r in RS)
        print(f"{k:<16}{v['F_mse']:>8.4f}  {cells}")
    print("full-rank equivalence max gap:", max(v["full_rank_equiv_max_abs_gap"] for v in out["results"].values()))
