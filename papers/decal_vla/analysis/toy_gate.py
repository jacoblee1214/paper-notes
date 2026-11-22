"""토이: 접촉이 간헐적일 때 스칼라 촉각 게이트가 필요한가 (논문 식 2의 구조만 축소해서 검사).

논문 §4.3, 식 2: X~ = X + sigma * CrossAttn(Q_vl, K_local, V_local), sigma = Gate(z_global).
이 토이는 교차어텐션 대신 작은 MLP 두 개를 쓰고, "시각 경로 f(v)에 촉각 경로 g(tau)를 더한다"는 구조만 남겼다.
검사하는 것은 논문의 수치가 아니라 다음 조건부 질문이다: 접촉이 없는 구간에서 촉각이 잡음(센서 드리프트)일 때만 게이트가
도움이 되고, 촉각이 항상 깨끗하면 게이트가 이득이 없는가. 결과는 구성에서 거의 자명한 부분이 있고(잡음 구간) 실제 로봇의
이득 크기에는 아무 말도 하지 않는다.

설정(모두 합성 데이터)
- 시각 v in R^4, 촉각 tau in R^6, 행동 a in R^3, 접촉 상태 c in {0, 1} (비율 0.3)
- 자유 구간(c=0): a = A v. 촉각은 잡음 N(0, s_free^2). 접촉 구간(c=1): 시각은 가려져 잡음, a = B tau, 촉각 N(0, 3^2).
- 체제 A(drift): 위와 같음. 체제 B(clean): 자유 구간에서도 a = A v + B tau이고 촉각이 깨끗(s_free = 3, 정보 있음).
- 모델: vision(f(v)만), ungated(f(v) + g(tau)), gated(f(v) + sigma(h(tau)) * g(tau)). 한 은닉층 64, Adam, 1500 스텝.
- 시드 5개, 학습 4096개와 시험 4096개. CPU.
실행: python analysis/toy_gate.py   (torch CPU, 다른 작업과 병렬로 돌릴 때 약 4분)
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OUT = Path(__file__).resolve().parent / "toy_gate_results.json"
torch.set_num_threads(2)


def make_data(n, regime, s_free, gen):
    A = torch.tensor(np.random.RandomState(1).randn(3, 4), dtype=torch.float32)
    B = torch.tensor(np.random.RandomState(2).randn(3, 6), dtype=torch.float32)
    c = (torch.rand(n, generator=gen) < 0.3).float()[:, None]
    v_clean = torch.randn(n, 4, generator=gen)
    tau_c = 3.0 * torch.randn(n, 6, generator=gen)
    tau_f = s_free * torch.randn(n, 6, generator=gen)
    v = torch.where(c > 0, torch.randn(n, 4, generator=gen), v_clean)  # 접촉 중 시각은 가려져 잡음
    tau = torch.where(c > 0, tau_c, tau_f)
    if regime == "drift":
        a = torch.where(c > 0, tau_c @ B.T, v_clean @ A.T)
    else:  # clean: 자유 구간에서도 촉각이 행동에 기여
        a = torch.where(c > 0, tau_c @ B.T, v_clean @ A.T + tau_f @ B.T)
    return v, tau, a, c


class Net(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.f = nn.Sequential(nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, 3))
        self.g = nn.Sequential(nn.Linear(6, 64), nn.SiLU(), nn.Linear(64, 3))
        self.h = nn.Sequential(nn.Linear(6, 16), nn.SiLU(), nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, v, tau):
        x = self.f(v)
        if self.kind == "ungated":
            return x + self.g(tau)
        if self.kind == "gated":
            return x + self.h(tau) * self.g(tau)
        return x


def run(regime, s_free, seed, kind):
    gen = torch.Generator().manual_seed(seed)
    tr = make_data(4096, regime, s_free, gen)
    te = make_data(4096, regime, s_free, gen)
    torch.manual_seed(seed)
    net = Net(kind)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(1500):
        loss = ((net(tr[0], tr[1]) - tr[2]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        err = ((net(te[0], te[1]) - te[2]) ** 2).mean(1)
        c = te[3][:, 0] > 0
        out = {"all": float(err.mean()), "free": float(err[~c].mean()), "contact": float(err[c].mean())}
        if kind == "gated":
            g = net.h(te[1])[:, 0]
            out["gate_free"] = float(g[~c].mean())
            out["gate_contact"] = float(g[c].mean())
    return out


def main():
    t0 = time.time()
    res = {}
    for regime, s_free in (("drift", 0.3), ("drift", 1.0), ("drift", 3.0), ("clean", 3.0)):
        key = f"{regime}_sfree{s_free}"
        res[key] = {}
        for kind in ("vision", "ungated", "gated"):
            runs = [run(regime, s_free, s, kind) for s in range(5)]
            res[key][kind] = {m: [round(float(np.mean([r[m] for r in runs])), 4), round(float(np.std([r[m] for r in runs], ddof=1)), 4)] for m in runs[0]}
        print(key)
        for kind, d in res[key].items():
            print("  ", kind, {m: v[0] for m, v in d.items()})
    res["seconds"] = round(time.time() - t0, 1)
    res["note"] = "각 값은 [시드 5개 평균, 표본 SD]. all=전체 MSE, free/contact=구간별 MSE, gate_*=게이트 평균."
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("seconds", res["seconds"])


if __name__ == "__main__":
    main()
