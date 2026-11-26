"""토이 T1: 재구성 목표가 깨끗한 깊이일 때와 오염된 깊이일 때, 순환 잠재 상태가 달라지는가.

논문 표 2의 "Noisy-depth supervision(708.0) 대 Full WSM(753.3)"이 겨냥한 질문의 축소판이다.
- 숨은 상태 x_t (4차원, AR(1))에서 "깊이" y_t (32 픽셀, 비선형)가 만들어진다.
- 입력은 오염된 깊이(픽셀 고장이 시간 상관을 가지고 0으로 남는 결핍 잡음 + 작은 가우시안)와 부분 고유감각 p_t (x의 앞 2차원).
- 인코더 MLP -> GRU(h) -> 디코더. 목표만 다르다: (a) 깨끗한 깊이 (b) 오염된 깊이(입력과 같은 것).
- 지표: 보류 시퀀스에서 h_t로 x_t를 선형 프로브한 R² (키 r2), 깨끗한 깊이 복원 MSE.
- 이 토이는 논문의 RSSM, PPO, 시뮬레이터가 아니다. 목표 선택이 잠재 상태에 미치는 영향만 본다.
실행: python3 analysis/toy_clean_target.py   (torch 필요, CPU 약 1분)
결과: analysis/toy_clean_target_results.json
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

OUT = Path(__file__).resolve().parent / "toy_clean_target_results.json"
torch.set_num_threads(2)
D, P, T = 4, 32, 40  # 상태 차원, 픽셀 수, 시퀀스 길이


def make_data(n, rng, p_new=0.03, corr=0.95, gauss=0.02):
    A = 0.97
    x = np.zeros((n, T, D))
    x[:, 0] = rng.normal(size=(n, D))
    for t in range(1, T):
        x[:, t] = A * x[:, t - 1] + 0.25 * rng.normal(size=(n, D))
    Wm = np.random.default_rng(7).normal(size=(D, P)) / 2.0
    clean = 1 / (1 + np.exp(-(x @ Wm)))  # 0-1 사이 "깊이"
    fail = np.zeros((n, T, P), dtype=bool)
    fail[:, 0] = rng.random((n, P)) < p_new
    for t in range(1, T):
        keep = fail[:, t - 1] & (rng.random((n, P)) < corr)
        new = rng.random((n, P)) < p_new
        fail[:, t] = keep | new
    noisy = clean + gauss * rng.normal(size=clean.shape)
    noisy = np.where(fail, 0.0, noisy)
    prop = x[:, :, :2] + 0.05 * rng.normal(size=(n, T, 2))
    f = lambda a: torch.tensor(a, dtype=torch.float32)
    return f(x), f(clean), f(noisy), f(prop)


class Net(nn.Module):
    def __init__(self, h=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(P + 2, 64), nn.ELU(), nn.Linear(64, 32), nn.ELU())
        self.gru = nn.GRU(32, h, batch_first=True)
        self.dec = nn.Sequential(nn.Linear(h, 64), nn.ELU(), nn.Linear(64, P))

    def forward(self, noisy, prop):
        e = self.enc(torch.cat([noisy, prop], -1))
        hs, _ = self.gru(e)
        return hs, self.dec(hs)


def probe_r2(h_tr, x_tr, h_te, x_te):
    H = torch.cat([h_tr.reshape(-1, h_tr.shape[-1]), torch.ones(h_tr.shape[0] * h_tr.shape[1], 1)], 1)
    W = torch.linalg.lstsq(H, x_tr.reshape(-1, D)).solution
    Ht = torch.cat([h_te.reshape(-1, h_te.shape[-1]), torch.ones(h_te.shape[0] * h_te.shape[1], 1)], 1)
    pred = Ht @ W
    tgt = x_te.reshape(-1, D)
    return float(1 - ((pred - tgt) ** 2).sum() / ((tgt - tgt.mean(0)) ** 2).sum())


def run(target, seed, steps=600, p_new=0.03, gauss=0.02):
    kw = dict(p_new=p_new, gauss=gauss)
    rng = np.random.default_rng(seed)
    x, clean, noisy, prop = make_data(512, rng, **kw)
    xe, cleane, noisye, prope = make_data(256, np.random.default_rng(1000 + seed), **kw)
    kw_s = dict(p_new=p_new * 2.67, gauss=gauss)  # 분포 이동: 고장 확률 2.67배(0이면 0 그대로)
    xs, cleans, noisys, props = make_data(256, np.random.default_rng(2000 + seed), **kw_s)
    torch.manual_seed(seed)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    tgt = clean if target == "clean" else noisy
    for _ in range(steps):
        idx = torch.randint(0, x.shape[0], (64,))
        _, rec = net(noisy[idx], prop[idx])
        loss = ((rec - tgt[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        h_tr, _ = net(noisy, prop)
        h_e, rec_e = net(noisye, prope)
        h_s, rec_s = net(noisys, props)
        return {
            "r2": probe_r2(h_tr, x, h_e, xe), "r2_shift": probe_r2(h_tr, x, h_s, xs),
            "clean_mse": float(((rec_e - cleane) ** 2).mean()), "clean_mse_shift": float(((rec_s - cleans) ** 2).mean()),
        }


def main():
    t0 = time.time()
    out = {"seeds": 5, "steps": 600, "setup": "D=4, 32 픽셀, T=40, 학습 시퀀스 512, 보류 256, 이동 256(고장 확률 0.03 -> 0.08). 대조 팔: 고장 없이 가우시안 0.15만"}
    arms = {"": dict(p_new=0.03, gauss=0.02), "_gauss_only": dict(p_new=0.0, gauss=0.15)}
    for arm, kw in arms.items():
      for target in ("clean", "noisy"):
        rows = [run(target, s, **kw) for s in range(5)]
        out[target + arm] = {k: {"mean": float(np.mean([r[k] for r in rows])), "sd": float(np.std([r[k] for r in rows], ddof=1)),
                           "values": [round(r[k], 4) for r in rows]} for k in rows[0]}
    out["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for tg in ("clean", "noisy", "clean_gauss_only", "noisy_gauss_only"):
        print(tg, {k: round(v["mean"], 4) for k, v in out[tg].items()})
    print("seconds", out["seconds"])


if __name__ == "__main__":
    main()
