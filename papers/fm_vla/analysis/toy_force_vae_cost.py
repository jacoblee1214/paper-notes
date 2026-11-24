"""논문 부록 E.1, 표 3의 Force-VAE(Perceiver-IO) 크기를 가정해 만들어 파라미터 수와 CPU 한 스텝 시간을 잰다.

목적: 이 맥(CPU/MPS)에서 Stage 1 사전학습이 얼마나 걸릴지 어림한다. 논문 수치를 재현하는 것이 아니다.
논문에 없는 값은 가정이다: 피드포워드 비율 4, 입력 MLP는 두 층, 디코더 질의 MLP는 두 층,
교차 어텐션 한 헤드의 차원 64(인코더), 디코더 헤드 수 8.
실행: python3 analysis/toy_force_vae_cost.py   (torch, CPU 스레드 4개, 1분 이내)
출력: analysis/toy_force_vae_cost_results.json
"""
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

torch.set_num_threads(4)
OUT = Path(__file__).resolve().parent / "toy_force_vae_cost_results.json"
D, K, DZ, BANDS, FMAX = 384, 8, 96, 32, 1500.0


def fourier(t, bands=BANDS, fmax=FMAX):
    f = torch.linspace(1.0, fmax, bands)
    ang = 2 * math.pi * t[..., None] * f / (fmax * 2)
    return torch.cat([ang.sin(), ang.cos()], -1)


class Block(nn.Module):  # 사전 정규화 교차/자기 어텐션 + FFN
    def __init__(self, d, heads, kv_dim=None, head_dim=None):
        super().__init__()
        kv_dim = kv_dim or d
        self.ln_q, self.ln_kv, self.ln_f = nn.LayerNorm(d), nn.LayerNorm(kv_dim), nn.LayerNorm(d)
        inner = (head_dim * heads) if head_dim else d
        self.q, self.k, self.v = nn.Linear(d, inner), nn.Linear(kv_dim, inner), nn.Linear(kv_dim, inner)
        self.o = nn.Linear(inner, d)
        self.heads = heads
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, kv=None):
        kv = x if kv is None else kv
        B, Nq, _ = x.shape
        q = self.q(self.ln_q(x)).view(B, Nq, self.heads, -1).transpose(1, 2)
        kvn = self.ln_kv(kv)
        k = self.k(kvn).view(B, kv.shape[1], self.heads, -1).transpose(1, 2)
        v = self.v(kvn).view(B, kv.shape[1], self.heads, -1).transpose(1, 2)
        a = nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, Nq, -1)
        x = x + self.o(a)
        return x + self.ff(self.ln_f(x))


class ForceVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(6 + 2 * BANDS, D), nn.GELU(), nn.Linear(D, D))
        self.latents = nn.Parameter(torch.randn(K, D) * 0.02)
        self.enc_cross = nn.ModuleList([Block(D, 1, head_dim=64) for _ in range(2)])
        self.enc_self = nn.ModuleList([Block(D, 8, head_dim=32) for _ in range(10)])
        self.head = nn.Linear(D, 2 * DZ)
        self.up = nn.Linear(DZ, D)
        self.qmlp = nn.Sequential(nn.Linear(2 * BANDS, D), nn.GELU(), nn.Linear(D, D))
        self.dec_cross = nn.ModuleList([Block(D, 8) for _ in range(2)])
        self.out = nn.Linear(D, 6)

    def forward(self, f):
        B, T, _ = f.shape
        t = torch.arange(T, dtype=torch.float32)
        pe = fourier(t).expand(B, -1, -1)
        tok = self.inp(torch.cat([f, pe], -1))
        z = self.latents.expand(B, -1, -1)
        for i, blk in enumerate(self.enc_cross):
            z = blk(z, tok)
            if i == 0:
                z = self.enc_self[0](z)
        for blk in self.enc_self[1:]:
            z = blk(z)
        mu, logvar = self.head(z).chunk(2, -1)
        zz = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        h = self.up(zz)
        q = self.qmlp(pe)
        for blk in self.dec_cross:
            q = blk(q, h)
        return self.out(q), mu, logvar


def main():
    m = ForceVAE()
    n_params = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    res = {"params_million": round(n_params / 1e6, 2), "assumptions": "FFN 비율 4, 입력·질의 MLP 두 층"}
    for T, B in [(300, 8), (600, 8)]:
        f = torch.randn(B, T, 6)
        times = []
        for it in range(4):
            t0 = time.perf_counter()
            rec, mu, lv = m(f)
            loss = ((rec - f) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            times.append(time.perf_counter() - t0)
        ms = 1000 * sum(times[1:]) / 3
        per_sample = ms / B
        full = per_sample * 512 * 100_000 / 1000 / 3600
        res[f"T{T}"] = {"batch": B, "ms_per_step": round(ms, 1), "ms_per_sample": round(per_sample, 1),
                        "hours_for_100k_steps_batch512_extrapolated": round(full, 1)}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
