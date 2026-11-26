"""토이 T1: 재잡음 일관성 점수(RCS, 논문 식 6–7)가 후보 선택에서 무엇을 가려내는가.

논문의 RCS는 "학습된 조건부 행동 분포와 얼마나 일관된 후보인가"를 재는 점수다(VI-A절).
이 토이는 2차원 행동 분포를 flow matching으로 학습한 작은 MLP에서 같은 절차(후보 N개 샘플 → 같은 잡음으로
재잡음 → 속도장 오차의 음수 → 최댓값 선택)를 돌려 선택된 후보의 품질을 본다.
논문의 모델(5.5B)과 실로봇 과제를 대신하지 않는다. "점수가 품질을 가리는 조건"만 본다.

설정(우리가 만든 모형): 목표 g = (0, 0), 성공은 g에서 거리 0.2 이내. 데이터는 두 성분의 혼합이다.
  A. 품질-밀도 일치: 정밀 성분(σ=0.05, 비율 0.4)은 성공, 거친 성분(σ=0.5, 비율 0.6)은 대부분 실패
  B. 품질-밀도 불일치: 성공 성분이 넓고(σ=0.15, 비율 0.4), 실패는 g에서 떨어진 좁은 성분(중심 (0.6,0), σ=0.05, 비율 0.6)
후보 N=4(논문 기본값), 재잡음 시각은 샘플러 5스텝 중 ratio 0.5(공개 코드 기본값). 시드 5개, CPU, 약 1분.
실행: python3 analysis/toy_rcs_selection.py  → analysis/toy_rcs_selection_results.json  (torch 필요)
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)
OUT = Path(__file__).with_name("toy_rcs_selection_results.json")
G = torch.tensor([0.0, 0.0])


def make_data(scenario, n, gen):
    if scenario == "A":
        z = torch.rand(n, generator=gen) < 0.4
        x = torch.where(z[:, None], 0.05 * torch.randn(n, 2, generator=gen), 0.5 * torch.randn(n, 2, generator=gen))
    else:
        z = torch.rand(n, generator=gen) < 0.4
        fail_c = torch.tensor([0.6, 0.0])
        x = torch.where(z[:, None], 0.15 * torch.randn(n, 2, generator=gen), fail_c + 0.05 * torch.randn(n, 2, generator=gen))
    return x


def success(x):
    return ((x - G).norm(dim=-1) < 0.2).float()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))

    def forward(self, x, t):
        return self.f(torch.cat([x, t[:, None]], -1))


def train(scenario, seed, steps=3000):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    data = make_data(scenario, 20000, gen)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    for _ in range(steps):
        idx = torch.randint(0, len(data), (512,), generator=gen)
        x = data[idx]
        eps = torch.randn(512, 2, generator=gen)
        t = torch.rand(512, generator=gen)
        xt = (1 - t[:, None]) * x + t[:, None] * eps          # 식 (2)와 같은 보간: x_t = (1-σ)x + σ ε, 목표 ε - x
        loss = ((net(xt, t) - (eps - x)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net


@torch.no_grad()
def sample(net, n, gen, steps=5):
    x = torch.randn(n, 2, generator=gen)
    ts = torch.linspace(1, 0, steps + 1)
    for i in range(steps):
        t = ts[i].expand(n)
        x = x + (ts[i + 1] - ts[i]) * net(x, t)
    return x


@torch.no_grad()
def rcs(net, x, gen, ratio=0.5, steps=5):
    """공개 코드와 같은 절차: 시각 인덱스 int(steps*ratio)의 σ에서, 후보 전체에 같은 새 잡음으로 재잡음."""
    ts = torch.linspace(1, 0, steps + 1)
    sigma = ts[int(steps * ratio)]
    eps = torch.randn(1, 2, generator=gen).expand(len(x), 2)
    xt = (1 - sigma) * x + sigma * eps
    pred = net(xt, sigma.expand(len(x)))
    return -((pred - (eps - x)) ** 2).mean(-1)


def run(scenario, seed, trials=4000, N=4):
    net = train(scenario, seed)
    gen = torch.Generator().manual_seed(1000 + seed)
    base = rcs_pick = oracle = 0.0
    corr = []
    for _ in range(trials):
        c = sample(net, N, gen)
        s = success(c)
        r = rcs(net, c, gen)
        base += s.mean().item()
        rcs_pick += s[r.argmax()].item()
        oracle += s.max().item()
    # 후보 단위 점수-성공 상관 (많은 후보를 한 번에)
    c = sample(net, 20000, gen)
    r = rcs(net, c, gen)
    s = success(c)
    from scipy.stats import mannwhitneyu
    auc = float(mannwhitneyu(r[s == 1].numpy(), r[s == 0].numpy()).statistic / (int(s.sum()) * int((1 - s).sum()))) if 0 < s.sum() < len(s) else float("nan")
    return {"random": base / trials, "rcs": rcs_pick / trials, "oracle": oracle / trials, "auc": auc, "p_success_model": float(s.mean())}


if __name__ == "__main__":
    t0 = time.time()
    res = {}
    for sc in ("A", "B"):
        rows = [run(sc, sd) for sd in range(5)]
        res[sc] = {k: [round(r[k], 4) for r in rows] for k in rows[0]}
        res[sc]["mean"] = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]}
        print(sc, res[sc]["mean"], f"{time.time() - t0:.0f}s")
    res["note"] = "후보 N=4, 시드 5, 시행 4000. 우리가 만든 2차원 모형이며 논문의 모델과 다름"
    OUT.write_text(json.dumps(res, indent=1))
