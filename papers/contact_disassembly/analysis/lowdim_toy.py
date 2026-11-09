"""토이 B: 힘장(낮은 차원 통계)과 원시 촉각 영상 중 어느 쪽이 표본 효율이 높은가 (numpy, 1스레드, 약 1분).

질문: 라벨이 낮은 차원 잠재 z(2차원)에만 의존할 때, 원시 영상 I(32×32 = 1024픽셀)를 그대로 쓰는 것이
    z를 직접 쓰거나 비지도 PCA로 2차원 요약을 뽑아 쓰는 것보다 얼마나 많은 데이터를 요구하나?

가정(논문에 없음):
- y = 1[z1 + 0.5 z2 > 0] (무잡음 라벨), 평가는 무잡음 라벨.
- 영상 I = tanh(G z) + 과제와 무관한 픽셀 잡음(표준편차 s). G의 크기 g = 0.05 (픽셀당 신호 진폭).
- s ∈ {0.5, 1, 2}: 잡음이 커지면 PCA-2가 신호 부분공간을 못 찾는다(숨은 가정: 신호 분산이 잡음 분산보다 커야 한다).
- 학습기: 릿지(λ = 1, 절편 있음). 원시 영상은 1024차원이라 쌍대형(n × n)으로 푼다.

비교: oracle(z 직접), 원시 영상 릿지, PCA-2(비지도, 학습 영상만으로 적합) 후 릿지.

실행: python3 analysis/lowdim_toy.py → analysis/lowdim_toy_results.json
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "lowdim_toy_results.json"

D_Z, SIDE = 2, 32
D_PIX = SIDE * SIDE
W_TRUE = np.array([1.0, 0.5])
G_AMP = 0.05
NS = [10, 20, 40, 80, 160, 320, 640]
N_TEST = 2000
SEEDS = 30
LAM = 1.0
S_LIST = [0.5, 1.0, 2.0]

_g = np.random.default_rng(777)
G = G_AMP * _g.standard_normal((D_PIX, D_Z))  # 장치 설계(고정)


def sample(n, s, rng):
    z = rng.standard_normal((n, D_Z))
    y = np.sign(z @ W_TRUE)
    y[y == 0] = 1
    I = np.tanh(z @ G.T) + s * rng.standard_normal((n, D_PIX))
    return z, I, y


def ridge_dual_predict(Xtr, ytr, Xte, lam=LAM):
    mu = Xtr.mean(0)
    Xc = Xtr - mu
    m_y = ytr.mean()
    K = Xc @ Xc.T
    alpha = np.linalg.solve(K + lam * np.eye(len(K)), ytr - m_y)
    w = Xc.T @ alpha
    return np.sign((Xte - mu) @ w + m_y + 1e-12)


def ridge_primal_predict(Xtr, ytr, Xte, lam=LAM):
    mu = Xtr.mean(0)
    Xc = Xtr - mu
    m_y = ytr.mean()
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ (ytr - m_y))
    return np.sign((Xte - mu) @ w + m_y + 1e-12)


def pca2_fit(Xtr):
    mu = Xtr.mean(0)
    # 쌍대 SVD: 샘플 수가 작아도 빠르다
    U, S, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    comp = Vt[:D_Z]
    return mu, comp


def run():
    out = {}
    for s in S_LIST:
        rows = {"oracle_z": {}, "raw_pixels": {}, "pca2": {}}
        for n in NS:
            r = {k: [] for k in rows}
            for seed in range(SEEDS):
                rng = np.random.default_rng(9000 + seed)
                z_te, I_te, y_te = sample(N_TEST, s, rng)
                z_tr, I_tr, y_tr = sample(n, s, rng)
                # oracle: z를 직접 쓴다(상한)
                r["oracle_z"].append(np.mean(ridge_primal_predict(z_tr, y_tr, z_te) == y_te))
                # 원시 영상
                r["raw_pixels"].append(np.mean(ridge_dual_predict(I_tr, y_tr, I_te) == y_te))
                # PCA-2: 학습 영상만으로 비지도 적합 후 2차원 투영
                mu, comp = pca2_fit(I_tr)
                P_tr = (I_tr - mu) @ comp.T
                P_te = (I_te - mu) @ comp.T
                r["pca2"].append(np.mean(ridge_primal_predict(P_tr, y_tr, P_te) == y_te))
            for k in rows:
                rows[k][str(n)] = {"mean": round(float(np.mean(r[k])), 4),
                                   "sd": round(float(np.std(r[k])), 4)}
        # PCA가 잡은 부분공간과 실제 G 열공간의 정렬(0~1): 숨은 가정 점검
        rng = np.random.default_rng(42)
        _, I_big, _ = sample(2000, s, rng)
        mu, comp = pca2_fit(I_big)
        Q, _ = np.linalg.qr(G)
        align = np.linalg.norm(Q.T @ comp.T) / np.sqrt(D_Z)
        rows["pca2_subspace_alignment_n2000"] = round(float(align), 4)
        out[f"s={s}"] = rows
    return out


def main():
    t0 = time.time()
    res = {"config": {"ns": NS, "seeds": SEEDS, "n_test": N_TEST, "lambda": LAM,
                      "g_amp": G_AMP, "s_list": S_LIST, "d_pix": D_PIX},
           "results": run()}
    res["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("wrote", OUT, "in", res["runtime_s"], "s")
    for key, rows in res["results"].items():
        print(key, "PCA-2 alignment", rows["pca2_subspace_alignment_n2000"])
        for k in ("oracle_z", "raw_pixels", "pca2"):
            print(f"  {k:10s}", " ".join(f"n={n}:{rows[k][str(n)]['mean']:.3f}" for n in NS))


if __name__ == "__main__":
    main()
