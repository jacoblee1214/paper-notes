"""시간 조건화 방식 비교: 인덱스 위치 / 시간 임베딩 더하기 / 시간 회전(RoPE), 학습 무작위화 유무와 범위 (torch CPU).

1차원 컨베이어 위 물체의 위치 x(t) = x0 + v t 를 불규칙한 시각의 관측 2개로 보고,
요청 시각 이후의 위치를 예측하는 작은 transformer를 학습한다.
- 조건화: index(학습된 위치 임베딩), addtime(상대 시각의 정현파 임베딩을 토큰에 더함),
  rope(attention 안에서 상대 시각 회전)
- 학습 무작위화: fixed(20 Hz, 지연 0), rand(12.5–33.3 Hz, 지연 0–50 ms, 행동 시작 지연 혼합),
  wide(1–50 Hz, 지연 0–100 ms, 같은 혼합)
- 평가: nominal, small(학습 범위), wide(1–50 Hz), delay200(행동 시작 지연 0–200 ms)

가정(논문에 없는 부분): 속도 v는 0.05–0.3 m/s의 임의 부호, 잔차(최신 관측 대비 변위)를 예측,
스트림마다 위상이 무작위인 균일 샘플링, 요청 시각 이전에 도착한 최신 2개 관측 사용.
실행: .venv/bin/python analysis/extra_time_conditioning.py
     → analysis/extra_time_conditioning_results.json
"""
import json
import math
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cpu")

N_OBS, H, ACT_DT = 2, 16, 0.05
STEPS, BATCH, SEEDS = 1500, 128, [0, 1, 2]
D, HEADS, LAYERS = 64, 4, 2
PERIODS = torch.tensor(np.geomspace(0.1, 2.0, 8), dtype=torch.float32)  # 시간 주파수
OMEGA = 2 * math.pi / PERIODS

TRAIN = {
    "fixed": dict(rate=(20.0, 20.0), delay=(0.0, 0.0), dt="zero"),
    "rand": dict(rate=(12.5, 33.33), delay=(0.0, 0.05), dt="mix"),
    "wide": dict(rate=(1.0, 50.0), delay=(0.0, 0.1), dt="mix"),
}
EVAL = {
    "nominal": dict(rate=(20.0, 20.0), delay=(0.0, 0.0), dt="zero"),
    "small": dict(rate=(12.5, 33.33), delay=(0.0, 0.05), dt="zero"),
    "wide": dict(rate=(1.0, 50.0), delay=(0.0, 0.1), dt="zero"),
    "delay200": dict(rate=(20.0, 20.0), delay=(0.0, 0.0), dt=0.2),
}


def sample(B, cfg, g):
    """관측 시각(상대, 음수), 관측 값, 질의 시각, 목표 변위(잔차) 샘플링."""
    u = lambda lo, hi, shape: lo + (hi - lo) * torch.rand(shape, generator=g)
    rate = u(*cfg["rate"], (B,))
    delay = u(*cfg["delay"], (B,))
    period = 1.0 / rate
    phase = torch.rand(B, generator=g) * period
    latest = -delay - phase
    t_obs = latest[:, None] - period[:, None] * torch.arange(N_OBS)[None].float()  # (B, n), 최신 먼저
    if cfg["dt"] == "zero":
        dt = torch.zeros(B)
    elif cfg["dt"] == "mix":
        dt = torch.where(torch.rand(B, generator=g) < 0.5, torch.zeros(B), u(0.0, 0.1, (B,)))
    else:
        dt = u(0.0, cfg["dt"], (B,))
    t_q = dt[:, None] + ACT_DT * torch.arange(H)[None].float()  # (B, H)
    v = u(0.05, 0.3, (B,)) * torch.where(torch.rand(B, generator=g) < 0.5, -1.0, 1.0)
    x0 = u(-0.5, 0.5, (B,))
    x_obs = x0[:, None] + v[:, None] * t_obs
    x_q = x0[:, None] + v[:, None] * t_q
    resid_q = x_q - x_obs[:, :1]  # 최신 관측 대비 변위
    rel_obs = t_obs - 0.0  # 요청 시각 기준 상대 시각 (음수)
    return x_obs - x_obs[:, :1], rel_obs, t_q, resid_q


def feats(tau, n_freq=8):
    """시간 임베딩용 정현파 특징 (addtime)."""
    ph = tau[..., None] * OMEGA
    return torch.cat([torch.sin(ph), torch.cos(ph)], -1)


class RopeAttn(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.qkv = nn.Linear(D, 3 * D)
        self.out = nn.Linear(D, D)
        self.hd = D // HEADS

    def rot(self, x, tau):
        # x: (B, h, T, hd), tau: (B, T). 첫 절반과 둘째 절반을 쌍으로 회전
        ang = tau[:, None, :, None] * OMEGA[None, None, None, :self.hd // 2]
        c, s = torch.cos(ang), torch.sin(ang)
        x1, x2 = x[..., : self.hd // 2], x[..., self.hd // 2:]
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1)

    def forward(self, x, tau):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, HEADS, self.hd).permute(2, 0, 3, 1, 4)
        if self.mode == "rope":
            q, k = self.rot(q, tau), self.rot(k, tau)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.attn = RopeAttn(mode)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x, tau):
        x = x + self.attn(self.ln1(x), tau)
        return x + self.mlp(self.ln2(x))


class Policy(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.val = nn.Linear(1, D)
        self.type_emb = nn.Parameter(torch.randn(2, D) * 0.02)
        self.q_base = nn.Parameter(torch.randn(H, D) * 0.02)
        self.idx_emb = nn.Parameter(torch.randn(N_OBS + H, D) * 0.02)
        self.time_mlp = nn.Linear(16, D)
        self.blocks = nn.ModuleList([Block(mode) for _ in range(LAYERS)])
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, 1)

    def forward(self, x_obs, t_obs, t_q):
        B = x_obs.shape[0]
        obs_tok = self.val(x_obs[..., None]) + self.type_emb[0]
        q_tok = self.q_base[None].expand(B, H, D) + self.type_emb[1]
        x = torch.cat([obs_tok, q_tok], 1)
        tau = torch.cat([t_obs, t_q], 1)
        if self.mode == "index":
            x = x + self.idx_emb[None]
        elif self.mode == "addtime":
            x = x + self.time_mlp(feats(tau))
        h = x
        for blk in self.blocks:
            h = blk(h, tau)
        return self.head(self.ln(h[:, N_OBS:]))[..., 0]


def train_one(mode, train_name, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(1000 + seed)
    model = Policy(mode).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    cfg = TRAIN[train_name]
    for _ in range(STEPS):
        xo, to, tq, rq = sample(BATCH, cfg, g)
        loss = F.mse_loss(model(xo, to, tq), rq * 10)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def evaluate(model, eval_name, seed):
    g = torch.Generator().manual_seed(9000 + seed)
    xo, to, tq, rq = sample(4096, EVAL[eval_name], g)
    pred = model(xo, to, tq)
    return float((pred / 10 - rq).abs().mean() * 100)  # cm


def main():
    t0 = time.time()
    res = {"device": str(DEVICE), "torch": torch.__version__, "numpy": np.__version__,
           "python": platform.python_version(), "threads": torch.get_num_threads(),
           "config": dict(steps=STEPS, batch=BATCH, seeds=SEEDS, d_model=D, heads=HEADS,
                          layers=LAYERS, horizon=H, act_dt_s=ACT_DT, obs_per_policy=N_OBS,
                          train=TRAIN, eval=EVAL, velocity_mps="0.05-0.3 random sign (assumed)"),
           "mae_cm": {}}
    for mode in ["index", "addtime", "rope"]:
        res["mae_cm"][mode] = {}
        for tr in TRAIN:
            res["mae_cm"][mode][tr] = {}
            per_eval = {e: [] for e in EVAL}
            for seed in SEEDS:
                m = train_one(mode, tr, seed).eval()
                for e in EVAL:
                    per_eval[e].append(evaluate(m, e, seed))
            for e in EVAL:
                v = np.array(per_eval[e])
                res["mae_cm"][mode][tr][e] = {"mean": round(float(v.mean()), 3),
                                              "std": round(float(v.std()), 3),
                                              "per_seed": [round(x, 3) for x in v]}
            print(mode, tr, {e: round(float(np.mean(per_eval[e])), 2) for e in EVAL}, flush=True)
    res["wall_time_s"] = round(time.time() - t0, 1)
    (ROOT / "analysis" / "extra_time_conditioning_results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1))
    print("wall", res["wall_time_s"], "s")


if __name__ == "__main__":
    main()
