"""추가 검증 1: 동역학이 행동 신호와 직교할 때(정렬 α = 0) 동역학 고유 이득 (CPU numpy 1스레드, 시드 30개).

uvt_toy.py의 함수(sample, fit_predict, make_world 구조)를 그대로 가져다 쓰고, 동역학 사상 S만 바꾼다.
S의 행 = t_i · [√α B ; √(1−α) B₂]  (B: 행동 신호 부분공간의 직교 기저, B₂: 그와 직교하는 기저, t_i: 단위 벡터)
- α = 1: uvt_toy.py와 같음(동역학이 신호 부분공간 안)
- α = 0.5: 절반 정렬
- α = 0: 동역학이 행동 신호와 직교 (예측 가능하지만 행동과 무관한 방향)
결과 항목: 동역학 고유 이득 = MSE_J(ρ=0) − MSE_J(ρ=0.95), 행동 전용 대비 A−J, n ∈ {30, 60, 200}, r ∈ {1,2,3,4,6}.
실행: python3 analysis/extra_linear_orthogonal.py   (수 분 이내, 1스레드)
출력: analysis/extra_linear_orthogonal_results.json
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import uvt_toy as ut  # noqa: E402

OUT = HERE / "extra_linear_orthogonal_results.json"
ALPHAS = [1.0, 0.5, 0.0]
NS = ut.NS
RS = ut.RS
RHO_HI = 0.95
SEEDS = ut.SEEDS
t0 = time.time()


def make_world_align(seed, alpha):
    """uvt_toy.make_world와 같은 W, 그리고 정렬 α를 가진 S."""
    rng = np.random.default_rng(1000 + seed)
    M = rng.standard_normal((ut.K_DIM, ut.Q_RANK))
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    B = np.linalg.qr(rng.standard_normal((ut.P_DIM, ut.Q_RANK)))[0].T     # 신호 부분공간 기저 (q × p)
    W = M @ B
    T = rng.standard_normal((ut.M_DIM, ut.Q_RANK))
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    # 신호 부분공간과 직교하는 기저: 별도 난수로 만든 뒤 B에 대해 직교화
    G = rng.standard_normal((ut.P_DIM, ut.Q_RANK))
    G = G - B.T @ (B @ G)
    B2 = np.linalg.qr(G)[0].T
    Bm = np.sqrt(alpha) * B + np.sqrt(1 - alpha) * B2       # 행 노름 1 (B, B2 직교 정규)
    S = T @ Bm
    return W, S


def run():
    res = {}
    for alpha in ALPHAS:
        for n in NS:
            acc = {"rho0": {r: [] for r in RS}, "rho_hi": {r: [] for r in RS}, "A": {r: [] for r in RS}}
            for seed in range(SEEDS):
                W, S = make_world_align(seed, alpha)
                rng = np.random.default_rng(seed)
                x_tr, _, a_tr, d_tr = ut.sample(rng, W, S, n, RHO_HI, 1.0)
                rng = np.random.default_rng(seed)
                _, _, _, d0_tr = ut.sample(rng, W, S, n, 0.0, 1.0)
                rng = np.random.default_rng(seed + 500)
                x_te, a_te_star, _, _ = ut.sample(rng, W, S, ut.N_TEST, RHO_HI, 1.0)
                scale = (np.sqrt(np.mean(np.var(a_tr, axis=0))), np.sqrt(np.mean(np.var(d_tr, axis=0))))
                scale0 = (scale[0], np.sqrt(np.mean(np.var(d0_tr, axis=0))))
                for r in RS:
                    pa = ut.fit_predict(x_tr, a_tr, d_tr, x_te, "A", r, scale)
                    pj = ut.fit_predict(x_tr, a_tr, d_tr, x_te, "J", r, scale)
                    pj0 = ut.fit_predict(x_tr, a_tr, d0_tr, x_te, "J", r, scale0)
                    acc["A"][r].append(np.mean((pa - a_te_star) ** 2) - np.mean((pj - a_te_star) ** 2))
                    acc["rho_hi"][r].append(np.mean((pj - a_te_star) ** 2))
                    acc["rho0"][r].append(np.mean((pj0 - a_te_star) ** 2))
            cell = {}
            for r in RS:
                dyn = np.array(acc["rho0"][r]) - np.array(acc["rho_hi"][r])   # 동역학 고유 이득
                ajg = np.array(acc["A"][r])                                    # 행동 전용 대비 이득
                cell[f"r={r}"] = {
                    "dyn_gain": round(float(dyn.mean()), 4),
                    "dyn_gain_se": round(float(dyn.std(ddof=1) / np.sqrt(SEEDS)), 4),
                    "A_minus_J": round(float(ajg.mean()), 4),
                    "A_minus_J_se": round(float(ajg.std(ddof=1) / np.sqrt(SEEDS)), 4),
                }
            res[f"alpha={alpha},n={n}"] = cell
            print(f"alpha {alpha} n {n} | r=2 dyn {cell['r=2']['dyn_gain']} | r=4 dyn {cell['r=4']['dyn_gain']} "
                  f"| r=6 dyn {cell['r=6']['dyn_gain']} | {time.time() - t0:.0f}s", flush=True)
    return res


if __name__ == "__main__":
    out = {"setup": {"alphas": ALPHAS, "ns": NS, "rs": RS, "rho_hi": RHO_HI, "seeds": SEEDS,
                     "definition": "dyn_gain = MSE_J(rho=0) - MSE_J(rho=0.95); A_minus_J = MSE_A - MSE_J(rho=0.95)"},
           "results": run()}
    out["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2))
    print("wrote", OUT.name, out["runtime_s"], "s")
