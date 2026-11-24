"""합성 힘 궤적으로 '재구성으로 사전학습한 8토큰 잠재 공간이 접촉 횟수를 담는가'를 보는 작은 toy.

설계(논문 실험의 대체가 아니다)
- 힘 한 채널(30 Hz, 창 240프레임). 활성 구간 길이 L ∈ [100, 240]이고 앞쪽은 저진폭 잡음으로 채운다(논문 3.2.1의 무작위 사전 패딩).
- 활성 구간에 가우시안 펄스를 N ∈ {1, 2, 3}개 놓는다(펄스 폭 약 0.4 s, 진폭 0.7–1.3, 간격 ≥ 20프레임). 잡음 σ = 0.08, 느린 표류.
- 소형 Perceiver 구조 VAE(K = 8 잠재 토큰, 토큰 차원 16)를 재구성 손실로만 학습한 뒤 동결한다.
- 동결한 평균 µ(8×16 = 128차원)에 선형 분류기를 붙여 N을 맞힌다(라벨은 분류기에만 쓴다).
- 대조 1: 같은 구조를 학습하지 않은 무작위 초기화로 둔 채 같은 분류기(재구성 사전학습의 효과).
- 대조 2: 원신호를 8구간 평균·절댓값 평균으로 줄인 128차원 특징에 같은 분류기.
- KL 설정 세 가지: 논문(β = 1e-3, free-bits 0.5), KL 없음(β = 0), 강한 KL(β = 0.1, free-bits 0).
한계: 합성 데이터, 단일 채널, 1500 스텝(배치 32), 시드 1개. 정책 학습과 무관하다. 잠재에 횟수가 선형으로 남는지만 본다.
실행: python3 analysis/toy_count_probe.py   (torch, CPU 스레드 4개, 2분 안팎)
출력: analysis/toy_count_probe_results.json
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
OUT = Path(__file__).resolve().parent / "toy_count_probe_results.json"
T, K, DZ, D = 240, 8, 16, 48
STEPS, SEEDS = 1500, [0]


def make_batch(n, rng):
    x = np.zeros((n, T), dtype=np.float32)
    y = rng.integers(1, 4, n)
    tt = np.arange(T)
    for i in range(n):
        L = int(rng.integers(100, T + 1))
        start = T - L
        x[i, :start] = rng.normal(0, 0.05, start)
        sig = np.zeros(T)
        # 펄스 위치: 활성 구간 안에서 간격 >= 20 프레임이 되도록 거부 샘플링
        while True:
            c = np.sort(rng.integers(start + 10, T - 10, y[i]))
            if y[i] == 1 or np.diff(c).min() >= 20:
                break
        for ci in c:
            sig += rng.uniform(0.7, 1.3) * np.exp(-0.5 * ((tt - ci) / 5.0) ** 2)
        drift = 0.1 * np.sin(2 * math.pi * tt / rng.uniform(150, 400) + rng.uniform(0, 6.28))
        noise = rng.normal(0, 0.08, T)
        x[i, start:] = (sig + drift + noise)[start:]
    return torch.from_numpy(x)[..., None], torch.from_numpy(y - 1).long()


def fourier(n, bands=16):
    t = torch.arange(n, dtype=torch.float32)[:, None]
    f = torch.linspace(1, 60, bands)[None]
    ang = 2 * math.pi * t * f / (2 * n)
    return torch.cat([ang.sin(), ang.cos()], -1)


class Layer(nn.Module):
    def __init__(self, cross):
        super().__init__()
        self.att = nn.MultiheadAttention(D, 4, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
        self.cross = cross

    def forward(self, x, kv=None):
        kvn = self.ln1(kv) if self.cross else self.ln1(x)
        x = x + self.att(self.ln1(x), kvn, kvn, need_weights=False)[0]
        return x + self.ff(self.ln2(x))


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe = nn.Parameter(fourier(T), requires_grad=False)
        self.inp = nn.Sequential(nn.Linear(1 + 32, D), nn.GELU(), nn.Linear(D, D))
        self.lat = nn.Parameter(torch.randn(K, D) * 0.1)
        self.ec = nn.ModuleList([Layer(True), Layer(True)])
        self.es = nn.ModuleList([Layer(False), Layer(False)])
        self.head = nn.Linear(D, 2 * DZ)
        self.up = nn.Linear(DZ, D)
        self.q = nn.Sequential(nn.Linear(32, D), nn.GELU(), nn.Linear(D, D))
        self.dc = nn.ModuleList([Layer(True), Layer(True)])
        self.out = nn.Linear(D, 1)

    def encode(self, x):
        B = x.shape[0]
        tok = self.inp(torch.cat([x, self.pe.expand(B, -1, -1)], -1))
        z = self.lat.expand(B, -1, -1)
        for c, s in zip(self.ec, self.es):
            z = s(c(z, tok))
        return self.head(z).chunk(2, -1)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        h = self.up(z)
        q = self.q(self.pe).expand(x.shape[0], -1, -1)
        for blk in self.dc:
            q = blk(q, h)
        return self.out(q), mu, lv


def train_vae(pool, beta, lam, seed, steps=STEPS, bs=32):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    m = VAE()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    for s in range(steps):
        x = pool[torch.randint(0, pool.shape[0], (bs,), generator=g)]
        rec, mu, lv = m(x)
        kl = 0.5 * (mu ** 2 + lv.exp() - 1 - lv)
        loss = ((rec - x) ** 2).mean() + beta * torch.clamp(kl, min=lam).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return m, float(((rec - x) ** 2).mean().detach()), float(kl.mean().detach())


def probe(feat_tr, y_tr, feat_te, y_te):
    mean, std = feat_tr.mean(0), feat_tr.std(0) + 1e-6
    a, b = (feat_tr - mean) / std, (feat_te - mean) / std
    clf = nn.Linear(a.shape[1], 3)
    opt = torch.optim.Adam(clf.parameters(), lr=0.02, weight_decay=1e-3)
    for _ in range(400):
        loss = nn.functional.cross_entropy(clf(a), y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float((clf(b).argmax(1) == y_te).float().mean())


def raw_feat(x):
    seg = x[..., 0].view(x.shape[0], 8, T // 8)
    return torch.cat([seg.mean(2), seg.abs().mean(2), seg.amax(2), seg.amin(2)], 1)[:, :32]


def main():
    t0 = time.perf_counter()
    res = {"setup": {"window": T, "K": K, "dz": DZ, "width": D, "steps": STEPS, "batch": 32, "seeds": SEEDS,
                     "pretrain_pool": 1500, "probe_train_n": 1000, "probe_test_n": 1000}, "runs": {}}
    pool, _ = make_batch(1500, np.random.default_rng(999))
    cfgs = {"paper_beta1e-3_free0.5": (1e-3, 0.5), "no_kl": (0.0, 0.0), "strong_kl_beta0.1": (0.1, 0.0)}
    for name, (beta, lam) in cfgs.items():
        accs, accs_rand, accs_raw, recs, kls = [], [], [], [], []
        for seed in SEEDS:
            m, rec, kl = train_vae(pool, beta, lam, seed)
            m.eval()
            rng = np.random.default_rng(100 + seed)
            xtr, ytr = make_batch(1000, rng)
            xte, yte = make_batch(1000, rng)
            with torch.no_grad():
                ftr, fte = m.encode(xtr)[0].flatten(1), m.encode(xte)[0].flatten(1)
                torch.manual_seed(seed)
                m0 = VAE().eval()
                rtr, rte = m0.encode(xtr)[0].flatten(1), m0.encode(xte)[0].flatten(1)
            accs.append(probe(ftr, ytr, fte, yte))
            accs_rand.append(probe(rtr, ytr, rte, yte))
            accs_raw.append(probe(raw_feat(xtr), ytr, raw_feat(xte), yte))
            recs.append(rec)
            kls.append(kl)
        res["runs"][name] = {
            "probe_acc_pretrained_mean_over_seeds": round(float(np.mean(accs)), 3), "per_seed": [round(a, 3) for a in accs],
            "probe_acc_random_init_encoder": round(float(np.mean(accs_rand)), 3),
            "probe_acc_raw_summary_features": round(float(np.mean(accs_raw)), 3),
            "final_train_mse": round(float(np.mean(recs)), 4), "final_mean_kl_nats_per_dim": round(float(np.mean(kls)), 3),
        }
        print(name, res["runs"][name], f"{time.perf_counter() - t0:.0f}s", flush=True)
    res["chance"] = 1 / 3
    res["elapsed_s"] = round(time.perf_counter() - t0, 1)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
