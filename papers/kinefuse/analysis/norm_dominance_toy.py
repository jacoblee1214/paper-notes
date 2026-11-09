"""KineFuse 'norm dominance' toy: flat concatenation에서 촉각 토큰이 시각을 누르는 조건 (CPU, numpy, 단일 스레드).
실행: python3 analysis/norm_dominance_toy.py

논문 Abstract(iii)은 flat fusion과 joint-level 표현이 "norm dominance"로 시각을 억누른다고 쓰지만,
본문에는 특징 노름을 측정한 결과가 없다. 아래 toy는 그 메커니즘이 어떤 조건에서 생기는지 본다.
- 가정: 시각 토큰 노름 1, 촉각 토큰 노름 s (논문은 두 노름을 보고하지 않음). 차원 d=64, 시각 토큰 400개.
- A. 한 개 attention head에서 촉각 토큰이 받는 attention 질량 vs s, LN 없음 / LN 있음 (논문 Eq. 1–3의 flat 토큰 concat과 같은 구조).
  고정된 무작위 W_q, W_k 사용. 로짓은 (W_q q)·(W_k k)/√d 이므로 노름에 대해 s²로 커진다.
- B. 선형 readout 회귀: 두 신호의 실제 기여는 같다(분산 1씩). 특징을 s로 키운 채 ridge(λ)를 쓰면 촉각 몫이 커지는가?
  OLS(벌점 없음)에서는 가중치가 스케일을 보상하므로 몫이 s와 무관해야 한다. 표준화하면(LN 역할) 벌점이 스케일에 무관해진다.
- C. 같은 문제를 GD(고정 학습률)로 T 스텝 학습: 학습률은 가장 큰 스케일 특징에 맞춰 안정적으로 고른다. 작은 스케일 시각 특징이 얼마나 늦게 맞춰지는가.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)
D = 64
N_V = 400


def unit_rows(n, d, rng):
    x = rng.standard_normal((n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def part_a():
    """attention 질량: 촉각 토큰 K개가 받는 질량 (softmax 전체 대비). 시각 노름 1, 촉각 노름 s."""
    q = unit_rows(1, D, rng)[0]
    V = unit_rows(N_V, D, rng)                      # 시각 토큰 방향 (노름 1)
    G_rand = unit_rows(16, D, rng)                  # 촉각 토큰: 무작위 방향
    scales = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]
    res = {"scales": scales}
    # 두 설정: 시각 로짓 표준편차가 무작위 초기화 수준(~0.015)인 경우와 ~1인 경우(날카로운 attention)
    for tag, gain in (("initlike", 1.0), ("sharp", None)):
        Wq = rng.standard_normal((D, D)) / np.sqrt(D)
        Wk = rng.standard_normal((D, D)) / np.sqrt(D)
        u = Wq @ q
        if gain is None:
            g_ = 1.0 / np.sqrt(float(((V @ Wk.T) @ u).std() / np.sqrt(D)))  # 로짓은 g^2에 비례
            Wq, Wk = g_ * Wq, g_ * Wk
            u = Wq @ q
        g = np.linalg.pinv(Wk) @ u                  # 촉각 토큰: 키가 질의와 정렬되도록 고른 방향 (학습이 최선일 때의 상한)
        G_aln = np.tile(g / np.linalg.norm(g), (16, 1))
        res[f"{tag}_visual_logit_std"] = round(float((((V @ Wk.T) @ u) / np.sqrt(D)).std()), 4)
        for align, G in (("rand", G_rand), ("aligned", G_aln)):
            for K in (1, 4, 16):
                for norm_mode in ("raw", "LN"):
                    masses = []
                    for s in scales:
                        X = np.vstack([V, s * G[:K]])
                        if norm_mode == "LN":
                            X = X / np.linalg.norm(X, axis=1, keepdims=True)  # 모든 토큰 노름을 1로 고정(LN 근사, s=1과 같은 조건)
                        logits = ((X @ Wk.T) @ u) / np.sqrt(D)
                        w = np.exp(logits - logits.max())
                        masses.append(float(w[N_V:].sum() / w.sum()))
                    res[f"{tag}_{align}_K{K}_{norm_mode}"] = [round(m, 4) for m in masses]
    res["count_share_uniform"] = {str(K): round(K / (K + N_V), 4) for K in (1, 4, 16)}
    return res


def ridge_share(X, y, lam, p):
    """ridge 해에서 촉각 열(마지막 q열)이 예측 분산에서 차지하는 몫."""
    n = X.shape[0]
    w = np.linalg.solve(X.T @ X / n + lam * np.eye(X.shape[1]), X.T @ y / n)
    pred_v = X[:, :p] @ w[:p]
    pred_h = X[:, p:] @ w[p:]
    tot = np.var(pred_v + pred_h)
    return float(np.var(pred_h) / tot), float(1 - np.mean((y - pred_v - pred_h) ** 2) / np.var(y))


def make_data(n, p, q, rng):
    Zv = rng.standard_normal((n, p))
    Zh = rng.standard_normal((n, q))
    a = rng.standard_normal(p); a *= 1.0 / np.linalg.norm(a)
    b = rng.standard_normal(q); b *= 1.0 / np.linalg.norm(b)
    # 두 신호의 실제 기여를 같게: 분산 1씩
    y = Zv @ a * np.sqrt(1.0) + Zh @ b * np.sqrt(1.0)
    y = y + 0.1 * rng.standard_normal(n)
    return Zv, Zh, y


def part_b():
    p, q, n = 32, 8, 4000
    Zv, Zh, y = make_data(n, p, q, rng)
    rows = {}
    for s in (1, 3, 10, 30, 100):
        Xraw = np.hstack([Zv, s * Zh])               # 촉각 특징만 키움 (raw, flat concat)
        Xstd = np.hstack([Zv, Zh])                   # 표준화(LN/표준화 효과): 스케일 제거
        row = {}
        for lam in (0.0, 1e-2, 1e-1, 1.0):
            sh_raw, r2_raw = ridge_share(Xraw, y, lam, p)
            sh_std, r2_std = ridge_share(Xstd, y, lam, p)
            row[f"lam{lam}"] = {"share_h_raw": round(sh_raw, 3), "R2_raw": round(r2_raw, 3),
                                "share_h_std": round(sh_std, 3), "R2_std": round(r2_std, 3)}
        rows[f"s{s}"] = row
    return rows


def part_c():
    """고정 학습률 GD: 모든 특징을 한 학습률로 학습. 안정성 한계는 가장 큰 스케일 특징이 정한다."""
    p, q, n = 32, 8, 4000
    Zv, Zh, y = make_data(n, p, q, rng)
    out = {}
    for s in (1, 10):
        for mode in ("raw", "std"):
            Xh = s * Zh if mode == "raw" else Zh
            X = np.hstack([Zv, Xh])
            H = X.T @ X / n
            eta = 0.9 / np.linalg.eigvalsh(H).max()  # 이 문제에서 발산하지 않는 최대급 학습률
            w = np.zeros(X.shape[1])
            checkpoints = {10, 50, 200, 2000}
            res = {}
            for t in range(1, 2001):
                grad = X.T @ (X @ w - y) / n
                w -= eta * grad
                if t in checkpoints:
                    pv = X[:, :p] @ w[:p]
                    ph = X[:, p:] @ w[p:]
                    tot = np.var(pv + ph)
                    res[f"T{t}"] = {"share_h": round(float(np.var(ph) / tot), 3),
                                    "R2": round(float(1 - np.mean((y - pv - ph) ** 2) / np.var(y)), 3),
                                    "R2_vision_part_only": round(float(1 - np.mean((y - pv) ** 2) / np.var(y)), 3)}
            res["eta"] = float(eta)
            out[f"s{s}_{mode}"] = res
    return out


def main():
    out = {"A_attention_mass_vs_scale": part_a(), "B_ridge_share_vs_scale": part_b(), "C_gd_share_vs_steps": part_c()}
    path = ROOT / "analysis" / "norm_dominance_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
