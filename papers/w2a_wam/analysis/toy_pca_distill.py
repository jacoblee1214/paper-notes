"""토이: PCA로 줄인 교사 잠재를 코사인 손실로 증류하면 학생의 표본 효율이 어떻게 달라지는가.

논문(W2A) 주장 중 "압축된 저차원 부분공간이 행동 생성의 지도 신호로 충분하다"(초록, 4.1절, 부록 A.1)와
"섞은 교사·무작위 교사는 도움이 안 된다"(부록 A.3)를 작은 합성 문제에서 구조만 확인한다. 논문 수치의 재현이 아니다.

설계(모두 우리가 정한 값이며 결과는 설계에 의존한다)
- 숨은 과제 요인 s (4차원)와 방해 요인 b (20차원, 표준편차 1.5). 관측 o = Q [s; b] (24차원, 무작위 직교 Q).
- 정답 행동 a = [tanh(W1 s); sin(W2 s)] (8차원). 학생은 o만 본다. 학습 표본 96개와 1024개(두 설정), 시험 4000개.
- 교사 잠재 z (128차원) = 과제 요인의 풍부한 특징 64개(분산 1 수준) + 방해 요인의 특징 64개(분산 9 수준) + 잡음.
  교사는 학생이 표본 몇 개로 배우기 어려운 구조를 이미 알고 있다고 가정한다(큰 사전학습 모델의 역할).
- 학생: MLP 24-128-128-8, 두 번째 은닉층에서 선형 읽기 헤드(Q-Former의 자리)로 교사 목표를 코사인 손실로 맞춘다.
  총 손실 = MSE(행동) + lambda (1 - cos). 논문 식 10-12를 단순화한 것이고 시간 가중 w(tau)는 없다.
- 비교: 증류 없음, PCA K = 4, 16, 32, 64, PCA 없음(원래 128차원), 섞은 교사(표본 사이 순서를 섞음), 무작위 교사.
- 시드 5개. 각 학습은 CPU에서 수 초.

실행: python3 analysis/toy_pca_distill.py   (torch, numpy. 전체 약 1분)
출력: analysis/toy_pca_distill_results.json
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
OUT = HERE / "toy_pca_distill_results.json"
DS, DB, DZ = 4, 20, 128
N_TEST = 4000
SETTINGS = {96: 400, 1024: 200}  # 학습 표본 수: 에폭
LAMBDA = 1.0


def make_world(seed):
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.normal(size=(DS + DB, DS + DB)))[0]
    W1 = rng.normal(size=(4, DS)) * 0.7
    W2 = rng.normal(size=(4, DS)) * 0.7
    U = rng.normal(size=(64, DS)) * 1.0
    V = rng.normal(size=(64, DB)) * 1.0
    return dict(Q=Q, W1=W1, W2=W2, U=U, V=V)


def sample(world, n, rng):
    s = rng.normal(size=(n, DS))
    b = rng.normal(size=(n, DB)) * 1.5
    o = np.concatenate([s, b], 1) @ world["Q"].T
    a = np.concatenate([np.tanh(s @ world["W1"].T), np.sin(s @ world["W2"].T)], 1)
    zs = np.tanh(s @ world["U"].T) * 1.0
    zb = np.tanh(b @ world["V"].T / 1.5) * 3.0
    z = np.concatenate([zs, zb], 1) + 0.1 * rng.normal(size=(n, DZ))
    return o.astype(np.float32), a.astype(np.float32), z.astype(np.float32)


def pca_fit(z, k):
    mu = z.mean(0, keepdims=True)
    u, sv, vt = np.linalg.svd(z - mu, full_matrices=False)
    return mu, vt[:k].T  # (DZ, k)


class Student(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(DS + DB, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.head = nn.Linear(128, 8)
        self.read = nn.Linear(128, k) if k else None

    def forward(self, x):
        h = self.f(x)
        return self.head(h), (self.read(h) if self.read is not None else None)


def run(world, seed, mode, k, n_train, epochs):
    rng = np.random.default_rng(1000 + seed)
    o, a, z = sample(world, n_train, rng)
    ot, at, _ = sample(world, N_TEST, np.random.default_rng(777 + seed))
    if mode == "none":
        target = None
        kk = 0
    else:
        if mode == "pca":
            mu, U = pca_fit(z, k)
            target = (z - mu) @ U
            kk = k
        elif mode == "raw":
            target = z
            kk = DZ
        elif mode == "shuffled":
            mu, U = pca_fit(z, 64)
            target = ((z - mu) @ U)[rng.permutation(n_train)]
            kk = 64
        elif mode == "random":
            target = rng.normal(size=(n_train, 64)).astype(np.float32) * ((z - z.mean(0)) @ pca_fit(z, 64)[1]).std()
            kk = 64
        target = torch.tensor(target, dtype=torch.float32)
    torch.manual_seed(seed)
    net = Student(kk)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    xo, xa = torch.tensor(o), torch.tensor(a)
    for ep in range(epochs):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, 64):
            idx = perm[i:i + 64]
            pa, rd = net(xo[idx])
            loss = F.mse_loss(pa, xa[idx])
            if target is not None:
                loss = loss + LAMBDA * (1 - F.cosine_similarity(rd, target[idx], dim=-1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pa, _ = net(torch.tensor(ot))
        mse = F.mse_loss(pa, torch.tensor(at)).item()
    return mse


def main():
    t0 = time.time()
    torch.set_num_threads(1)
    configs = [("none", 0)] + [("pca", k) for k in (4, 16, 32, 64)] + [("raw", DZ), ("shuffled", 64), ("random", 64)]
    out = {}
    for n_train, epochs in SETTINGS.items():
        res = {}
        for mode, k in configs:
            name = f"{mode}" + (f"_K{k}" if mode in ("pca", "raw") else "")
            vals = [run(make_world(s), s, mode, k, n_train, epochs) for s in range(5)]
            res[name] = {"test_mse_per_seed": [round(v, 4) for v in vals], "mean": round(float(np.mean(vals)), 4),
                         "sd": round(float(np.std(vals, ddof=1)), 4)}
            print(f"N={n_train:5d} {name:12s} test MSE {res[name]['mean']:.4f} +- {res[name]['sd']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        base = res["none"]["test_mse_per_seed"]
        for name, r in res.items():
            d = np.array(r["test_mse_per_seed"]) - np.array(base)
            r["paired_diff_vs_none_mean"] = round(float(d.mean()), 4)
            r["seeds_better_than_none"] = int((d < 0).sum())
        out[f"n_train_{n_train}"] = res
    OUT.write_text(json.dumps({"config": dict(LAMBDA=LAMBDA, settings=SETTINGS, seeds=5), "results": out},
                              indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
