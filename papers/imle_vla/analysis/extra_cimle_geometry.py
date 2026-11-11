"""추가 검증 1: 봉우리 간격·개수·잡음에 따른 cIMLE m=2 실패 경계 (CPU numpy 1스레드, 시드 3개).

imle_toy.py의 학습 루프(2층 tanh MLP, Adam 2e-3, 배치 256, 3000회)를 그대로 옮겨 쓰고,
목표만 바꾼다: K개 봉우리를 반지름 R 원 위에 두고, 봉우리 표준편차 σ.
- 기준: K=4, R=1, σ=0.05 (imle_toy.py와 같은 설정이므로 같은 값이 나와야 함)
- 간격: R ∈ {0.25, 0.5, 1}  (K=4, σ=0.05)
- 개수: K ∈ {2, 8}  (R=1, σ=0.05)
- 잡음: σ ∈ {0.02, 0.1, 0.2}  (K=4, R=1)
덮음 기준 반경과 봉우리 사이 기준 거리는 간격에 맞춰 0.25R, 0.3R로 줄인다.
방법: m ∈ {1, 2, 5}.
실행: python3 analysis/extra_cimle_geometry.py  (수 분, 1스레드)
출력: analysis/extra_cimle_geometry_results.json
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "extra_cimle_geometry_results.json"
ITERS = 3000
BATCH = 256
HID = 64
SEEDS = (0, 1, 2)
NS = 4000
MS = (1, 2, 5)
LR = 2e-3
t0 = time.time()


def centers(K, R):
    ang = np.arange(K) * 2 * np.pi / K
    return R * np.stack([np.cos(ang), np.sin(ang)], 1)


def init(sizes, rng):
    P = []
    for i, o in zip(sizes[:-1], sizes[1:]):
        P += [rng.normal(0, np.sqrt(1.0 / i), (i, o)), np.zeros(o)]
    return P


def fwd(P, X):
    hs = [X]
    h = X
    L = len(P) // 2
    for l in range(L):
        z = h @ P[2 * l] + P[2 * l + 1]
        h = np.tanh(z) if l < L - 1 else z
        hs.append(h)
    return h, hs


def bwd(P, hs, dout):
    g = [None] * len(P)
    L = len(P) // 2
    d = dout
    for l in reversed(range(L)):
        if l < L - 1:
            d = d * (1 - hs[l + 1] ** 2)
        g[2 * l] = hs[l].T @ d
        g[2 * l + 1] = d.sum(0)
        d = d @ P[2 * l].T
    return g


class Adam:
    def __init__(self, P, lr=LR):
        self.m = [np.zeros_like(p) for p in P]
        self.v = [np.zeros_like(p) for p in P]
        self.t = 0
        self.lr = lr

    def step(self, P, g):
        self.t += 1
        for i in range(len(P)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g[i]
            self.v[i] = 0.999 * self.v[i] + 0.001 * g[i] ** 2
            mh = self.m[i] / (1 - 0.9 ** self.t)
            vh = self.v[i] / (1 - 0.999 ** self.t)
            P[i] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def train_imle(rng, MU, sig, m, zd=2):
    K = MU.shape[0]
    P = init([zd, HID, HID, 2], rng)
    opt = Adam(P)
    for _ in range(ITERS):
        k = rng.integers(0, K, BATCH)
        y = MU[k] + sig * rng.normal(size=(BATCH, 2))
        z = rng.normal(size=(BATCH * m, zd))
        out, _ = fwd(P, z)
        cand = out.reshape(BATCH, m, 2)
        d = ((cand - y[:, None, :]) ** 2).sum(-1)
        j = d.argmin(1)
        zb = z.reshape(BATCH, m, zd)[np.arange(BATCH), j]
        pred, hs = fwd(P, zb)
        dout = 2 * (pred - y) / (BATCH * 2)
        opt.step(P, bwd(P, hs, dout))
    return lambda n: fwd(P, rng.normal(size=(n, zd)))[0]


def metrics(s, MU, R):
    d_all = np.sqrt(((s[:, None, :] - MU[None]) ** 2).sum(-1))
    near = d_all.argmin(1)
    dmin = d_all.min(1)
    K = MU.shape[0]
    covered = 0
    for k in range(K):
        mass = np.mean((near == k) & (dmin < 0.25 * R))
        covered += mass >= 0.05
    return {"coverage": covered / K, "between_frac": float(np.mean(dmin > 0.3 * R)),
            "mean_dist_over_R": float(dmin.mean() / R)}


def agg(runs):
    keys = runs[0].keys()
    return {k: {"mean": round(float(np.mean([r[k] for r in runs])), 4),
                "sd": round(float(np.std([r[k] for r in runs])), 4)} for k in keys}


CONDS = [
    ("base K=4 R=1 s=0.05", 4, 1.0, 0.05),
    ("R=0.5", 4, 0.5, 0.05),
    ("R=0.25", 4, 0.25, 0.05),
    ("K=2", 2, 1.0, 0.05),
    ("K=8", 8, 1.0, 0.05),
    ("s=0.02", 4, 1.0, 0.02),
    ("s=0.1", 4, 1.0, 0.1),
    ("s=0.2", 4, 1.0, 0.2),
]


def main():
    out = {"setup": {"iters": ITERS, "batch": BATCH, "hidden": HID, "seeds": list(SEEDS), "samples": NS,
                     "ms": list(MS), "cover_radius": "0.25R", "between_threshold": "0.3R"},
           "conditions": {}}
    for name, K, R, sig in CONDS:
        MU = centers(K, R)
        out["conditions"][name] = {"K": K, "R": R, "sigma": sig, "methods": {}}
        for m in MS:
            runs = []
            for sd in SEEDS:
                rng = np.random.default_rng(100 + sd)
                gen = train_imle(rng, MU, sig, m)
                runs.append(metrics(gen(NS), MU, R))
            out["conditions"][name]["methods"][f"m={m}"] = agg(runs)
            print(name, f"m={m}", round(time.time() - t0, 1), "s", flush=True)
    out["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", OUT.name, out["runtime_s"], "s")


if __name__ == "__main__":
    main()
