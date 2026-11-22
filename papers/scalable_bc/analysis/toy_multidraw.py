"""토이 T1: ABC-VLA의 '다중 diffusion draw'(3.2절)가 백본 기울기 분산을 줄이는지 CPU로 확인한다.

설정: 문맥 c -> 작은 MLP 백본 h(c) -> 헤드 g(x_t, t, h) 가 flow matching 속도 (noise - x)를 예측한다.
같은 배치 샘플을 k번 복제해 독립 (noise, t)를 뽑고 손실을 평균한다(논문 3.2절, 코드의 repeat_interleave).
고정한 모델에서 백본 기울기를 여러 번 뽑아 분산(공분산의 trace)을 k별로 잰다.
기대: Var(k) ~ V_data/B + V_noise/(B*k). draw를 늘리면 V_noise 항만 줄고 V_data 항에서 포화한다.
실행: python3 analysis/toy_multidraw.py   (torch CPU, 약 1분)
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)
OUT = Path(__file__).with_name("toy_multidraw_results.json")
DC, DH, DX, T_LEN = 6, 32, 8, 1


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(DC, 64), nn.GELU(), nn.Linear(64, DH))
        self.head = nn.Sequential(nn.Linear(DX + 1 + DH, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, DX))

    def loss(self, c, x, k, noise=None, t=None):
        h = self.backbone(c)  # 샘플당 한 번
        h = h.repeat_interleave(k, 0)
        x = x.repeat_interleave(k, 0)
        eps = torch.randn_like(x) if noise is None else noise
        t = torch.rand(x.shape[0], 1) if t is None else t
        xt = (1 - t) * x + t * eps
        v = self.head(torch.cat([xt, t, h], -1))
        return ((v - (eps - x)) ** 2).mean()


def run(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    W = torch.randn(DC, DX) / np.sqrt(DC)

    def batch(B, data_noise=0.3):
        c = torch.randn(B, DC)
        x = torch.tanh(c @ W) + data_noise * torch.randn(B, DX)  # 같은 문맥에도 행동이 흩어진다(다봉성 대용)
        return c, x

    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
    for step in range(1500):  # 기울기 분산을 잴 지점까지 학습 (k=1)
        c, x = batch(256)
        loss = net.loss(c, x, 1)
        opt.zero_grad()
        loss.backward()
        opt.step()

    def grad_vec(c, x, k):
        net.zero_grad()
        net.loss(c, x, k).backward()
        return torch.cat([p.grad.flatten() for p in net.backbone.parameters()]).clone()

    B, REPS = 32, 300
    out = {}
    for k in (1, 2, 4, 8, 16):
        gs = torch.stack([grad_vec(*batch(B), k) for _ in range(REPS)])  # 배치도 새로 뽑는 전체 분산
        c, x = batch(B)
        gs2 = torch.stack([grad_vec(c, x, k) for _ in range(REPS)])  # 배치 고정, (noise, t)만 새로
        out[k] = (gs.var(0).sum().item(), gs2.var(0).sum().item())
    return out


t0 = time.time()
SEEDS = [0, 1, 2, 3, 4]
KS = [1, 2, 4, 8, 16]
runs = [run(s) for s in SEEDS]
res = {"B": 32, "reps": 300, "seeds": SEEDS, "per_seed": {}, "ratio_total_var_vs_k1_mean": {}, "ratio_noise_only_vs_k1_mean": {}}
for s, r in zip(SEEDS, runs):
    res["per_seed"][str(s)] = {str(k): {"total_var": r[k][0], "noise_only_var_fixed_batch": r[k][1]} for k in KS}
for k in KS:
    tr = [r[k][0] / r[1][0] for r in runs]
    nr = [r[k][1] / r[1][1] for r in runs]
    res["ratio_total_var_vs_k1_mean"][str(k)] = [float(np.mean(tr)), float(np.std(tr))]
    res["ratio_noise_only_vs_k1_mean"][str(k)] = [float(np.mean(nr)), float(np.std(nr))]
    print(f"k={k:2d}  total_var/k1 = {np.mean(tr):.3f} +- {np.std(tr):.3f}   noise_only/k1 = {np.mean(nr):.3f} +- {np.std(nr):.3f}")
# 총 분산 ~ a + b/k 적합 (시드별 정규화 후 평균)
ks = np.array(KS, dtype=float)
fits = []
for r in runs:
    tot = np.array([r[k][0] for k in KS])
    A = np.stack([np.ones_like(ks), 1 / ks], 1)
    (a, b), *_ = np.linalg.lstsq(A, tot, rcond=None)
    fits.append(b / (a + b))
res["fit_noise_share_at_k1_mean_sd"] = [float(np.mean(fits)), float(np.std(fits))]
res["note"] = "백본 순/역전파는 샘플당 1회로 k와 무관하고 헤드만 k배(토이에서는 헤드가 작음). 토이이므로 절대값이 아니라 추세만 본다."
OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
print("noise share at k=1:", res["fit_noise_share_at_k1_mean_sd"])
print("wrote", OUT.name, f"({time.time()-t0:.1f}s)")
