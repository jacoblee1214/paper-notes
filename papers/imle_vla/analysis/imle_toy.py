"""IMLE-VLA 핵심 메커니즘 toy (CPU, numpy 1스레드, 약 1–2분). 실행: python3 analysis/imle_toy.py

목표: 2차원 링 위의 4봉우리 혼합(반지름 1, 봉우리 표준편차 0.05). 평균은 원점이라 평균으로 가면 빈 공간에 놓인다.
조건 없는(unconditional) 설정이다. 논문의 cIMLE는 관측 조건부이므로, 여기서는 조건을 하나로 고정한 경우만 본다.

비교 (모두 2층 tanh MLP, Adam, 배치 256, 반복 3000, 시드 3개)
- L2회귀 = cIMLE에서 m=1 (논문 식 5–6: 조건부 평균으로 수렴)
- cIMLE, m ∈ {2, 5, 10, 20} (논문 식 3–4: 샘플 m개 중 가장 가까운 것만 갱신)
- 흐름 매칭 1단계 (직선 경로 학습 후 한 번의 Euler 점프)
- 흐름 매칭 10단계 (논문 π0.5의 10 Euler 단계에 해당)
- 추가 점검: cIMLE m=2에서 잠재 차원 zd ∈ {2, 32} (논문은 잠재 차원 = 청크 크기 × 행동 차원)
- 추가 점검: 흐름 10단계로 학습한 가중치를 cIMLE m=2의 초기값으로 씀(논문 III-A의 초기화, t=1 고정)

지표 (생성 샘플 4000개): 봉우리 덮음률(반지름 0.25 안 질량 ≥ 5%인 봉우리 비율), 봉우리 사이 비율(가장 가까운 봉우리까지 거리 > 0.3),
평균 거리(품질), 샘플 평균의 크기(0이면 평균으로 붕괴).

주의: 손으로 짠 역전파와 작은 MLP이므로 절대 수치보다 경향만 본다. 결과: analysis/imle_toy_results.json, 시드 0의 샘플 1000개씩 analysis/imle_toy_samples.npz (그림용)
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "imle_toy_results.json"

K = 4
SIG = 0.05
ITERS = 3000
BATCH = 256
HID = 64
SEEDS = (0, 1, 2)
NS = 4000
t0 = time.time()


def centers():
    ang = np.arange(K) * 2 * np.pi / K
    return np.stack([np.cos(ang), np.sin(ang)], 1)


MU = centers()


def sample_target(rng, n):
    k = rng.integers(0, K, n)
    return MU[k] + SIG * rng.normal(size=(n, 2))


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
    def __init__(self, P, lr=2e-3):
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


def train_imle(rng, m, zd, iters=ITERS):
    P = init([zd, HID, HID, 2], rng)
    opt = Adam(P)
    for _ in range(iters):
        y = sample_target(rng, BATCH)
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


def train_flow(rng):
    P = init([3, HID, HID, 2], rng)
    opt = Adam(P)
    for _ in range(ITERS):
        x1 = sample_target(rng, BATCH)
        x0 = rng.normal(size=(BATCH, 2))
        t = rng.uniform(size=(BATCH, 1))
        xt = (1 - t) * x0 + t * x1
        v = x1 - x0
        pred, hs = fwd(P, np.concatenate([xt, t], 1))
        dout = 2 * (pred - v) / (BATCH * 2)
        opt.step(P, bwd(P, hs, dout))

    def sampler(n, steps):
        x = rng.normal(size=(n, 2))
        dt = 1.0 / steps
        for k in range(steps):
            tt = np.full((n, 1), k * dt)
            v, _ = fwd(P, np.concatenate([x, tt], 1))
            x = x + dt * v
        return x
    return sampler, P


def train_imle_warm(rng, P0, m):
    """논문 III-A: G_θ를 흐름 체크포인트에서 초기화한다. 흐름 망의 입력 (x_t, t)에 맞춰 z와 t=1을 넣는다."""
    P = [p.copy() for p in P0]
    opt = Adam(P)
    one = lambda n: np.ones((n, 1))
    for _ in range(ITERS):
        y = sample_target(rng, BATCH)
        z = rng.normal(size=(BATCH * m, 2))
        out, _ = fwd(P, np.concatenate([z, one(BATCH * m)], 1))
        cand = out.reshape(BATCH, m, 2)
        d = ((cand - y[:, None, :]) ** 2).sum(-1)
        j = d.argmin(1)
        zb = z.reshape(BATCH, m, 2)[np.arange(BATCH), j]
        pred, hs = fwd(P, np.concatenate([zb, one(BATCH)], 1))
        dout = 2 * (pred - y) / (BATCH * 2)
        opt.step(P, bwd(P, hs, dout))
    return lambda n: fwd(P, np.concatenate([rng.normal(size=(n, 2)), one(n)], 1))[0]


def metrics(s):
    d_all = np.sqrt(((s[:, None, :] - MU[None]) ** 2).sum(-1))
    near = d_all.argmin(1)
    dmin = d_all.min(1)
    covered = 0
    for k in range(K):
        mass = np.mean((near == k) & (dmin < 0.25))
        covered += mass >= 0.05
    return {"coverage": covered / K, "between_frac": float(np.mean(dmin > 0.3)),
            "mean_dist": float(dmin.mean()), "sample_mean_norm": float(np.linalg.norm(s.mean(0)))}


results = {}
samples = {}
configs = [("L2 regression (m=1)", 1, 2), ("cIMLE m=2 (zd=2)", 2, 2), ("cIMLE m=5 (zd=2)", 5, 2),
           ("cIMLE m=10 (zd=2)", 10, 2), ("cIMLE m=20 (zd=2)", 20, 2), ("cIMLE m=2 (zd=32)", 2, 32)]
for name, m, zd in configs:
    runs = []
    for sd in SEEDS:
        rng = np.random.default_rng(100 + sd)
        gen = train_imle(rng, m, zd)
        smp = gen(NS)
        runs.append(metrics(smp))
        if sd == SEEDS[0]:
            samples[name] = smp[:1000]
    results[name] = runs
    print(name, "done", round(time.time() - t0, 1), "s", flush=True)

# 수렴 예산 점검: m=2를 15000회로 늘려도 덮음이 생기는가 (m=5는 3000회와 같은 경향이라 생략)
for name, m in [("cIMLE m=2 (15k iters)", 2)]:
    runs = []
    for sd in SEEDS:
        rng = np.random.default_rng(100 + sd)
        gen = train_imle(rng, m, 2, iters=5 * ITERS)
        runs.append(metrics(gen(NS)))
    results[name] = runs
    print(name, "done", round(time.time() - t0, 1), "s", flush=True)

flow_runs = {"flow 1-step": [], "flow 10-step": []}
warm_runs = []
for sd in SEEDS:
    rng = np.random.default_rng(100 + sd)
    sampler, P_flow = train_flow(rng)
    s1, s10 = sampler(NS, 1), sampler(NS, 10)
    flow_runs["flow 1-step"].append(metrics(s1))
    flow_runs["flow 10-step"].append(metrics(s10))
    if sd == SEEDS[0]:
        samples["flow 1-step"], samples["flow 10-step"] = s1[:1000], s10[:1000]
    # 흐름 10단계 가중치에서 시작한 cIMLE m=2 (논문의 초기화 방식)
    warm_gen = train_imle_warm(rng, P_flow, 2)
    sw = warm_gen(NS)
    warm_runs.append(metrics(sw))
    if sd == SEEDS[0]:
        samples["cIMLE m=2 warm"] = sw[:1000]
    print("flow seed", sd, round(time.time() - t0, 1), "s", flush=True)
results["cIMLE m=2 (warm from flow)"] = warm_runs
results.update(flow_runs)
print("warm-start done", round(time.time() - t0, 1), "s", flush=True)


def agg(runs):
    keys = runs[0].keys()
    return {k: {"mean": float(np.mean([r[k] for r in runs])), "sd": float(np.std([r[k] for r in runs]))} for k in keys}


summary = {name: agg(runs) for name, runs in results.items()}
out = {"setup": {"modes": K, "sigma": SIG, "iters": ITERS, "batch": BATCH, "hidden": HID,
                 "seeds": list(SEEDS), "samples": NS, "target": "unconditional 4-mode ring, radius 1"},
       "summary": summary, "runs": results, "elapsed_s": round(time.time() - t0, 1)}
OUT.write_text(json.dumps(out, indent=1))
np.savez(ROOT / "analysis" / "imle_toy_samples.npz", **{k.replace(" ", "_").replace("(", "").replace(")", "").replace("=", ""): v for k, v in samples.items()})
for name, s in summary.items():
    print(f"{name:24s} cov {s['coverage']['mean']:.2f}  between {s['between_frac']['mean']:.3f}  "
          f"dist {s['mean_dist']['mean']:.3f}  |mean| {s['sample_mean_norm']['mean']:.3f}")
