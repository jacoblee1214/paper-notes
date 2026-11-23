"""한 스텝 시간 측정: 논문 부록 표 8의 크기(4+4층, d = 512, FF 2048, 헤드 16)를 따른 GEnc+GDec 정책을
무작위 가중치·무작위 입력(캐시된 RADIO 요약 5개 + 고유감각 8차원, 청크 16)으로 만들어 학습 한 스텝의 시간을 잰다.

공식 코드는 import하지 않고 논문 설명과 코드 구조를 따라 직접 만든 모듈이다. 데이터·가중치 다운로드 없음.
RADIO 백본은 동결 + 요약 캐시이므로 학습 단계에는 들어가지 않는다(PD3).

실행: python3 analysis/bench_policy_step.py  (torch, MPS가 있으면 MPS, 약 20초)  → analysis/bench_policy_step_results.json
"""
import json
import time
from pathlib import Path

import torch
from torch import nn

D, H, FF, L, TP = 512, 16, 2048, 4, 16
OUT = Path(__file__).with_name("bench_policy_step_results.json")


class EncLayer(nn.Module):
    def __init__(s):
        super().__init__()
        s.a = nn.MultiheadAttention(D, H)
        s.l1, s.l2 = nn.Linear(D, FF), nn.Linear(FF, D)
        s.n1, s.n2 = nn.LayerNorm(D), nn.LayerNorm(D)

    def forward(s, x):
        x = s.n1(x + s.a(x, x, x)[0])
        return s.n2(x + s.l2(torch.nn.functional.gelu(s.l1(x))))


class GroupDecLayer(nn.Module):
    def __init__(s):
        super().__init__()
        s.sa = nn.MultiheadAttention(D, H)
        s.ca = nn.ModuleList([nn.MultiheadAttention(D, H) for _ in range(3)])
        s.l1, s.l2 = nn.Linear(D, FF), nn.Linear(FF, D)
        s.n = nn.ModuleList([nn.LayerNorm(D) for _ in range(5)])

    def forward(s, x, groups):
        x = s.n[0](x + s.sa(x, x, x)[0])
        d = torch.stack([ca(x, g, g)[0] for ca, g in zip(s.ca, groups)]).mean(0)
        x = s.n[1](x + d)
        return s.n[2](x + s.l2(torch.nn.functional.gelu(s.l1(x))))


class Policy(nn.Module):
    def __init__(s, n_views=5):
        super().__init__()
        s.proj = nn.ModuleList([nn.Sequential(nn.Linear(2304, 2 * D), nn.LayerNorm(2 * D), nn.GELU(), nn.Linear(2 * D, D)) for _ in range(n_views)])
        s.emb = nn.Embedding(n_views, D)
        s.enc = nn.ModuleList([EncLayer() for _ in range(L)])
        s.dec = nn.ModuleList([GroupDecLayer() for _ in range(L)])
        s.q = nn.Embedding(TP, D)
        s.state = nn.Sequential(nn.Linear(8, D), nn.LayerNorm(D), nn.GELU(), nn.Linear(D, D))
        s.head = nn.Sequential(nn.Linear(D, D // 2), nn.GELU(), nn.Linear(D // 2, 8))
        s.n_fe = 4

    def forward(s, feats, qpos):  # feats: (B, V, 2304), 앞 4개가 FE, 마지막이 손목
        tok = torch.stack([p(feats[:, i]) for i, p in enumerate(s.proj)], 1) + s.emb.weight
        fe = tok[:, : s.n_fe].transpose(0, 1)
        for layer in s.enc:
            fe = layer(fe)
        wrist = tok[:, s.n_fe :].transpose(0, 1)
        joint = s.state(qpos).unsqueeze(0)
        x = s.q.weight.unsqueeze(1).expand(-1, feats.shape[0], -1)
        for layer in s.dec:
            x = layer(x, [joint, wrist, fe])
        return s.head(x.transpose(0, 1))


dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)
model = Policy().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.95, 0.999), weight_decay=1e-6)
B = 256
feats = torch.randn(B, 5, 2304, device=dev)
qpos = torch.randn(B, 8, device=dev)
target = torch.randn(B, TP, 8, device=dev)


def sync():
    if dev == "mps":
        torch.mps.synchronize()


for _ in range(3):
    loss = (model(feats, qpos) - target).abs().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
sync()
t0 = time.time()
N = 20
for _ in range(N):
    loss = (model(feats, qpos) - target).abs().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
sync()
dt = (time.time() - t0) / N
n_params = sum(p.numel() for p in model.parameters())
res = {"device": dev, "batch": B, "seconds_per_step": round(dt, 4), "params_million": round(n_params / 1e6, 1),
       "est_hours_for_20k_steps": round(dt * 20000 / 3600, 2),
       "note": "무작위 입력·가중치. 에폭 500, 데모 50개(데모당 약 200프레임, 10 Hz x 20 s 가정)면 에폭당 약 40스텝 → 약 2만 스텝"}
OUT.write_text(json.dumps(res, indent=1))
print(res)
