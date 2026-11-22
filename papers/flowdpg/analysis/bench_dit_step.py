"""v1 부록 표 3 규모의 행동 머리(DiT 14층, 헤드 16, 차원 1024, 토큰 H = 30)의 한 스텝 시간을 맥에서 잰다.

랜덤 가중치, 비전 백본과 크리틱은 뺀 하한이다. FlowDPG 한 스텝 = 한 번의 no_grad 순전파(식 4의 â) + 한 번의 순전파와 역전파.
MPS가 있으면 MPS, 없으면 CPU. 약 30초.
실행: python3 analysis/bench_dit_step.py   (torch 필요)  -> analysis/bench_dit_step_results.json
"""
import json
import time
from pathlib import Path

import torch

torch.set_num_threads(1)
OUT = Path(__file__).with_name("bench_dit_step_results.json")
dev = "mps" if torch.backends.mps.is_available() else "cpu"
D, LAYERS, HEADS, H = 1024, 14, 16, 30


def sync():
    if dev == "mps":
        torch.mps.synchronize()


layer = torch.nn.TransformerEncoderLayer(D, HEADS, 4 * D, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
net = torch.nn.Sequential(torch.nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)).to(dev)
inp = torch.nn.Linear(14, D).to(dev)
outp = torch.nn.Linear(D, 14).to(dev)
params = sum(p.numel() for p in list(net.parameters()) + list(inp.parameters()) + list(outp.parameters()))
opt = torch.optim.AdamW(list(net.parameters()) + list(inp.parameters()) + list(outp.parameters()), lr=1e-4)
res = {"device": dev, "params_M": round(params / 1e6, 1), "per_batch": {}}
for bs in (16, 64):
    x = torch.randn(bs, H + 1, 14, device=dev)
    times = []
    for i in range(6):
        sync()
        t0 = time.time()
        with torch.no_grad():
            _ = outp(net(inp(x)))
        out = outp(net(inp(x)))
        loss = out.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sync()
        if i >= 2:
            times.append(time.time() - t0)
    mem = torch.mps.current_allocated_memory() / 1e9 if dev == "mps" else None
    res["per_batch"][bs] = {"sec_per_step": round(sum(times) / len(times), 3), "mps_alloc_GB": None if mem is None else round(mem, 2)}
    print(bs, res["per_batch"][bs], flush=True)
s64 = res["per_batch"][64]["sec_per_step"]
res["extrapolation_batch256_10k_steps_hours_lower_bound"] = round(4 * s64 * 10000 / 3600, 1)
OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
print(res)
