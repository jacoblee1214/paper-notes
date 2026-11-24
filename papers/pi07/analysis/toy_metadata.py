"""토이: 에피소드 메타데이터가 혼합 품질 데이터의 '평균화'를 막는가 (논문 그림 18 왼쪽의 구조만 검사).

설정(우리 가정, 논문 값에 맞추지 않음): 1차원 상태 s, 이동 s' = s + 0.5 a. 시연자는 a = -k s + 잡음, 이득 k = 0.2 + 0.6 u,
u ~ U(0,1)은 에피소드 품질(1이 최고). 데이터는 품질 상위 30%, 50%, 80%, 100%.
모델: 작은 MLP가 a를 회귀한다. (1) 메타데이터 없음, (2) 메타데이터 u를 입력(15% 확률로 가림, 가림 표시 입력 포함).
평가: 시작 s=1에서 |s|<0.05에 도달하는 스텝 수(상한 40)의 역수를 '처리량'으로 본다.
메타데이터 모델은 u=1을 프롬프트로 주고, 선택적으로 CFG(β)로 가림 상태와 외삽한다.
이 토이는 회귀의 평균화만 본다. 논문이 보고한 '데이터를 늘리면 성능이 오르는' 효과(상태 다양성)는 만들지 않는다.
실행: python3 -I analysis/toy_metadata.py  (CPU, 약 1분) → analysis/toy_metadata_results.json
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OUT = Path(__file__).with_name("toy_metadata_results.json")
torch.set_num_threads(1)


def make_data(frac, n_total, rng):
    u = rng.uniform(1 - frac, 1.0, size=n_total)          # 품질 상위 frac
    k = 0.2 + 0.6 * u
    s = rng.uniform(-1.0, 1.0, size=n_total)
    a = -k * s + rng.normal(0, 0.02, size=n_total)
    return s, u, a


def train(s, u, a, use_meta, seed, steps=1500):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    s_t, u_t, a_t = (torch.tensor(x, dtype=torch.float32) for x in (s, u, a))
    net = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    for _ in range(steps):
        idx = torch.randint(0, len(s_t), (512,), generator=g)
        mask = (torch.rand(512, generator=g) > 0.15).float() if use_meta else torch.zeros(512)
        x = torch.stack([s_t[idx], u_t[idx] * mask, mask], 1)
        loss = ((net(x).squeeze(1) - a_t[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def rollout_steps(act, T=40):
    s = 1.0
    for t in range(T):
        if abs(s) < 0.05:
            return t
        s = s + 0.5 * act(s)
    return T


def evaluate(net, use_meta, beta=0.0):
    def act(s):
        with torch.no_grad():
            if not use_meta:
                return float(net(torch.tensor([[s, 0.0, 0.0]])))
            ac = float(net(torch.tensor([[s, 1.0, 1.0]])))
            au = float(net(torch.tensor([[s, 0.0, 0.0]])))
            return ac + beta * (ac - au)
    return rollout_steps(act)


res = {}
for frac in (0.3, 0.5, 0.8, 1.0):
    row = {"no_meta": [], "meta_u1": [], "meta_u1_cfg1.3": [], "meta_u1_cfg2.2": [], "mean_gain_data": None}
    for seed in range(5):
        rng = np.random.default_rng(seed)
        s, u, a = make_data(frac, 6000, rng)
        row["mean_gain_data"] = round(float(0.2 + 0.6 * (1 - frac / 2)), 3)
        n0 = train(s, u, a, False, seed)
        n1 = train(s, u, a, True, seed)
        row["no_meta"].append(evaluate(n0, False))
        row["meta_u1"].append(evaluate(n1, True))
        row["meta_u1_cfg1.3"].append(evaluate(n1, True, 1.3))
        row["meta_u1_cfg2.2"].append(evaluate(n1, True, 2.2))
    for k in ("no_meta", "meta_u1", "meta_u1_cfg1.3", "meta_u1_cfg2.2"):
        steps = np.array(row[k], dtype=float)
        row[k] = {"steps_mean": round(float(steps.mean()), 2), "throughput": round(float((1 / np.maximum(steps, 1)).mean()), 4)}
    res[f"top{int(frac * 100)}"] = row
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False))
