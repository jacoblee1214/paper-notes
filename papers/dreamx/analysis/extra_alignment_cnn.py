"""합성 2D 장면의 CNN encoder + 학습형 query + 첫 프레임 조건 decoder로 사람·로봇 latent 정렬을 시점 각도별로 비교 (torch CPU).

장면: 몸체(사람형: 주황 큰 원 / 로봇형: 파란 작은 원)는 영상 중앙에 고정, 손(흰 점)은 세계 좌표 p에 있다.
행동 a ∈ R²는 손의 변위다. 세계 좌표의 손 위치 p를 영상에 그릴 때 사람 카메라는 θ만큼 회전해 본다(p_img = Q(θ)p).
로봇 카메라는 θ = 0이다. 같은 a라도 사람 영상에서는 회전된 움직임으로 보인다.
- encoder: 연속 두 프레임(o_t, o_{t+1})을 채널로 쌓은 CNN(6→16→32→32, 6×6 토큰) + 학습형 query 하나의 주의 풀링 → z ∈ R²
- decoder: embodiment별, 첫 프레임 o_t와 z를 받아 o_{t+1}을 재구성
- 로봇 클립에만 행동 head 손실 ‖H z − a‖² (H 공유 선형). 사람 클립에는 행동 라벨이 없다.
조건: S 공유 encoder / I embodiment별 encoder / P 공유 encoder + 짝 클립 5%에 ‖z_h − z_r‖²(가중치 5).
짝 클립은 같은 a, 같은 시작 손 위치를 쓰고 카메라만 다르다.

지표: (1) 전이 오차 ‖H z_h − a_h‖² / ‖a_h‖²: 사람 latent를 로봇 행동 head로 풀었을 때 오차.
(2) 탐침 R²: 사람 latent → a 선형 회귀(시험 집합 안). (3) 사람 클립 재구성 MSE(첫 프레임 조건 decoder의 품질).

가정(toy 설정, 논문 값 아님): 24×24 RGB 합성 장면, 움직임 벡터 대신 영상 쌍, embodiment당 500 클립,
Adam 600 step, batch 64, θ ∈ {0, 30, 60, 90}°, 시드 3개. 시험 집합은 학습과 다른 시드로 뽑은 500 클립.
실행: .venv/bin/python analysis/extra_alignment_cnn.py → analysis/extra_alignment_cnn_results.json
"""
import json
import math
import os
import platform
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "4")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cpu")
IMG = 24
N, N_TEST, STEPS, BATCH, LR, PAIR_FRAC, PAIR_W = 500, 500, 600, 64, 2e-3, 0.05, 5.0
THETAS, SEEDS, CONDS = [0, 30, 60, 90], [0, 1, 2], ["S", "I", "P"]
BODY = {"r": (0.2, 0.6, 1.0, 0.25), "h": (1.0, 0.6, 0.2, 0.40)}  # (R, G, B, 반지름)
HAND = (0.95, 0.95, 0.95)

_g = torch.linspace(-1, 1, IMG)
GY, GX = torch.meshgrid(_g, _g, indexing="ij")


def world_motion(p0, a):
    """손의 세계 좌표 이동: 변위 a에 비선형 항을 더한 d."""
    d = a + 0.3 * torch.stack([a[:, 0] * a[:, 1], torch.sin(a[:, 0])], -1)
    return p0, p0 + d


def rot(theta_deg):
    t = math.radians(theta_deg)
    return torch.tensor([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]], dtype=torch.float32)


def render(p, emb, theta_deg, gen):
    """세계 좌표 손 위치 p(N,2)를 카메라 θ로 본 24×24 RGB 영상."""
    pi = p @ rot(theta_deg if emb == "h" else 0).T
    br, bg, bb, rad = BODY[emb]
    body = torch.exp(-(GX ** 2 + GY ** 2) / (2 * rad ** 2))
    dx = GX[None] - pi[:, 0, None, None]
    dy = GY[None] - pi[:, 1, None, None]
    hand = torch.exp(-(dx ** 2 + dy ** 2) / (2 * 0.12 ** 2))
    chans = [br * body + HAND[0] * hand, bg * body + HAND[1] * hand, bb * body + HAND[2] * hand]
    img = torch.stack(chans, 1)
    return img + 0.02 * torch.randn(img.shape, generator=gen)


def make_clips(a, p0, emb, theta_deg, gen):
    """행동 a와 시작 손 위치 p0로 (입력 쌍, 첫 프레임, 둘째 프레임)을 만든다."""
    p_start, p_end = world_motion(p0, a)
    o0 = render(p_start, emb, theta_deg, gen)
    o1 = render(p_end, emb, theta_deg, gen)
    return torch.cat([o0, o1], 1), o0, o1


def sample_clips(n, gen):
    a = -0.4 + 0.8 * torch.rand(n, 2, generator=gen)
    p0 = -0.5 + torch.rand(n, 2, generator=gen)
    return a, p0


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(6, 16, 3, 2, 1), nn.GELU(), nn.Conv2d(16, 32, 3, 2, 1), nn.GELU(),
                                  nn.Conv2d(32, 32, 3, 1, 1), nn.GELU())
        self.pos = nn.Parameter(torch.zeros(36, 32))
        self.query = nn.Parameter(0.1 * torch.randn(32))
        self.key = nn.Linear(32, 32)
        self.out = nn.Linear(32, 2)

    def forward(self, x):
        f = self.conv(x).flatten(2).transpose(1, 2) + self.pos  # (B, 36, 32)
        w = torch.softmax(self.key(f) @ self.query / math.sqrt(32), -1)
        return self.out((w[..., None] * f).sum(1))


class Dec(nn.Module):
    """첫 프레임 o_t와 잠재 z로 o_{t+1}을 재구성."""

    def __init__(self):
        super().__init__()
        self.lift = nn.Linear(2, 8 * 6 * 6)
        self.net = nn.Sequential(nn.Conv2d(3 + 8, 16, 3, 1, 1), nn.GELU(), nn.Conv2d(16, 16, 3, 1, 1), nn.GELU(),
                                 nn.Conv2d(16, 3, 3, 1, 1))

    def forward(self, o0, z):
        s = F.interpolate(self.lift(z).view(-1, 8, 6, 6), size=(IMG, IMG), mode="bilinear", align_corners=False)
        return self.net(torch.cat([o0, s], 1))


def run(cond, theta, seed):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed * 31 + theta)
    a_r, p0_r = sample_clips(N, gen)
    a_h, p0_h = sample_clips(N, gen)
    n_pair = int(PAIR_FRAC * N)
    a_h[:n_pair], p0_h[:n_pair] = a_r[:n_pair], p0_r[:n_pair]  # 짝 클립: 같은 행동·시작 위치, 카메라만 다름
    x_r, o0_r, o1_r = make_clips(a_r, p0_r, "r", 0, gen)
    x_h, o0_h, o1_h = make_clips(a_h, p0_h, "h", theta, gen)
    if cond == "S":
        enc_r = enc_h = Enc()
    else:
        enc_r, enc_h = Enc(), Enc()
    dec_r, dec_h = Dec(), Dec()
    H = nn.Linear(2, 2)
    params = list(set(list(enc_r.parameters()) + list(enc_h.parameters()) + list(dec_r.parameters())
                      + list(dec_h.parameters()) + list(H.parameters())))
    opt = torch.optim.Adam(params, lr=LR, weight_decay=1e-4)
    for _ in range(STEPS):
        b = torch.randint(0, N, (BATCH,))
        zr, zh = enc_r(x_r[b]), enc_h(x_h[b])
        loss = F.mse_loss(dec_r(o0_r[b], zr), o1_r[b]) + F.mse_loss(dec_h(o0_h[b], zh), o1_h[b])
        loss = loss + ((H(zr) - a_r[b]) ** 2).mean()
        if cond == "P":
            bp = torch.randint(0, n_pair, (16,))
            loss = loss + PAIR_W * ((enc_h(x_h[bp]) - enc_r(x_r[bp])) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # 시험 집합(학습과 다른 생성 시드)
    tg = torch.Generator().manual_seed(10_000 + seed * 31 + theta)
    a_t, p0_t = sample_clips(N_TEST, tg)
    x_t, o0_t, o1_t = make_clips(a_t, p0_t, "h", theta, tg)
    with torch.no_grad():
        zt = enc_h(x_t)
        pred = H(zt)
        transfer = float(((pred - a_t) ** 2).sum() / (a_t ** 2).sum())
        recon = float(F.mse_loss(dec_h(o0_t, zt), o1_t))
        X = torch.cat([zt, torch.ones(len(zt), 1)], -1)
        beta = torch.linalg.lstsq(X, a_t).solution
        ss_res = ((a_t - X @ beta) ** 2).sum()
        probe = float(1 - ss_res / ((a_t - a_t.mean(0)) ** 2).sum())
    return probe, transfer, recon


def main():
    t0 = time.time()
    res = {"device": str(DEVICE), "torch": torch.__version__, "numpy": np.__version__,
           "python": platform.python_version(), "threads": torch.get_num_threads(),
           "config": dict(image_px=IMG, samples_per_embodiment=N, test_clips=N_TEST, steps=STEPS, batch=BATCH,
                          lr=LR, pair_frac=PAIR_FRAC, pair_weight=PAIR_W, thetas_deg=THETAS, seeds=SEEDS),
           "transfer_error": {}, "probe_r2_human": {}, "recon_mse_human": {}}
    for cond in CONDS:
        for key in ("transfer_error", "probe_r2_human", "recon_mse_human"):
            res[key][cond] = {}
        for th in THETAS:
            vals = [run(cond, th, s) for s in SEEDS]
            for key, idx in (("probe_r2_human", 0), ("transfer_error", 1), ("recon_mse_human", 2)):
                xs = [v[idx] for v in vals]
                res[key][cond][str(th)] = {"mean": round(float(np.mean(xs)), 4),
                                           "sd": round(float(np.std(xs, ddof=1)), 4),
                                           "per_seed": [round(x, 4) for x in xs]}
            print(cond, th, "transfer", round(float(np.mean([v[1] for v in vals])), 3),
                  "probe", round(float(np.mean([v[0] for v in vals])), 3), flush=True)
    res["wall_time_s"] = round(time.time() - t0, 1)
    (ROOT / "analysis" / "extra_alignment_cnn_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("wall", res["wall_time_s"], "s")


if __name__ == "__main__":
    main()
