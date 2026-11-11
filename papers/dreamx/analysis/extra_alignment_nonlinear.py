"""비선형 latent action 정렬 toy: 공유 encoder / embodiment별 encoder / 짝 클립 5%를 시점 각도 θ에 따라 비교 (torch CPU).

모형: 행동 a ∈ R², 영상 공간의 움직임 Δo_e = tanh(R_e φ(a)) + 잡음 (8차원), φ(a) = [a1, a2, a1·a2, sin a1].
- 로봇 렌더러 R_r: 직교 열 2개. 사람 렌더러 R_h = R_r Q(θ) + η R_⊥ (열 정규화, η = 0.3).
- encoder z = E_e(Δo) (MLP 8→64→2), decoder Δo ≈ D_e(z) (embodiment별, MLP 2→64→8).
- 로봇 클립에만 행동 head 손실 ‖H z − a‖² (H 공유 선형). 사람 클립에는 없다.
조건: S 공유 encoder / I embodiment별 encoder / P embodiment별 encoder + 짝지은 사람·로봇 클립 5%에 ‖z_h − z_r‖² (가중치 5).
지표: (1) 탐침 R²: embodiment마다 z → a 선형 회귀. (2) 전이 오차 ‖H z_h − a_h‖² / ‖a_h‖²:
사람 latent를 로봇 행동 head로 풀었을 때 오차(행동 전이).
가정(toy 설정, 논문 값 아님): 선형 toy를 비선형으로 바꾼 것. CNN 대신 MLP encoder, 영상 대신 8차원 움직임 벡터,
embodiment당 1000 표본, Adam 2000 step, θ ∈ {0, 30, 60, 90}°, seed 3개. 첫 프레임 조건은 embodiment별 decoder로 대신함.
실행: .venv/bin/python analysis/extra_alignment_nonlinear.py → analysis/extra_alignment_nonlinear_results.json
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

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cpu")
M, N, STEPS, LR, PAIR_FRAC, PAIR_W = 8, 1000, 2000, 3e-3, 0.05, 5.0
THETAS, SEEDS, CONDS = [0, 30, 60, 90], [0, 1, 2], ["S", "I", "P"]


def phi(a):
    return torch.stack([a[:, 0], a[:, 1], a[:, 0] * a[:, 1], torch.sin(a[:, 0])], -1)


def orth(gen):
    q, _ = torch.linalg.qr(torch.randn(M, 2, generator=gen))
    return q


def renderers(theta_deg, seed):
    g = torch.Generator().manual_seed(1000 + seed)
    Rr = orth(g)
    Rperp = orth(g)[:, :1]
    t = math.radians(theta_deg)
    Q = torch.tensor([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]], dtype=torch.float32)
    Rh = Rr @ Q + 0.3 * torch.cat([Rperp, Rperp], 1)
    Rh = Rh / Rh.norm(dim=0, keepdim=True)
    W = torch.randn(4, 2, generator=g) * 0.7
    return Rr, Rh, W


def render(R, a, W, gen):
    f = phi(a) @ W  # 행동의 비선형 특징을 공통 2차원 잠재 방향으로 사영
    return torch.tanh(f @ R.T + 0.05 * torch.randn(len(a), M, generator=gen))


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, 64), nn.GELU(), nn.Linear(64, o))


def run(cond, theta, seed):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed * 17 + int(theta))
    Rr, Rh, W = renderers(theta, seed)
    a_r = torch.randn(N, 2, generator=gen)
    a_h = torch.randn(N, 2, generator=gen)
    o_r = render(Rr, a_r, W, gen)
    o_h = render(Rh, a_h, W, gen)
    n_pair = int(PAIR_FRAC * N)
    a_h[:n_pair] = a_r[:n_pair]
    o_h[:n_pair] = render(Rh, a_r[:n_pair], W, gen)
    if cond == "S":
        enc_r = enc_h = mlp(M, 2)
    else:
        enc_r, enc_h = mlp(M, 2), mlp(M, 2)
    dec_r, dec_h = mlp(2, M), mlp(2, M)
    H = nn.Linear(2, 2)
    params = list(set(list(enc_r.parameters()) + list(enc_h.parameters()) + list(dec_r.parameters())
                      + list(dec_h.parameters()) + list(H.parameters())))
    opt = torch.optim.Adam(params, lr=LR, weight_decay=1e-3)
    for _ in range(STEPS):
        b = torch.randint(0, N, (256,))
        zr, zh = enc_r(o_r[b]), enc_h(o_h[b])
        loss = ((dec_r(zr) - o_r[b]) ** 2).mean() + ((dec_h(zh) - o_h[b]) ** 2).mean()
        loss = loss + ((H(zr) - a_r[b]) ** 2).mean()
        if cond == "P":
            bp = torch.randint(0, n_pair, (64,))
            loss = loss + PAIR_W * ((enc_h(o_h[bp]) - enc_r(o_r[bp])) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        zr, zh = enc_r(o_r), enc_h(o_h)
        # 탐침 R²: embodiment마다 z → a 선형 회귀
        def r2(z, a):
            X = torch.cat([z, torch.ones(len(z), 1)], -1)
            beta = torch.linalg.lstsq(X, a).solution
            pred = X @ beta
            return float(1 - ((a - pred) ** 2).sum() / ((a - a.mean(0)) ** 2).sum())
        probe_h = r2(zh, a_h)
        # 전이 오차: 사람 z를 로봇 행동 head로 풀었을 때
        transfer = float(((H(zh) - a_h) ** 2).sum() / (a_h ** 2).sum())
    return probe_h, transfer


def main():
    t0 = time.time()
    res = {"device": str(DEVICE), "torch": torch.__version__, "numpy": np.__version__,
           "python": platform.python_version(), "threads": torch.get_num_threads(),
           "config": dict(latent_dim=2, obs_dim=M, samples_per_embodiment=N, steps=STEPS, lr=LR,
                          pair_frac=PAIR_FRAC, pair_weight=PAIR_W, thetas_deg=THETAS, seeds=SEEDS,
                          phi="[a1, a2, a1*a2, sin a1]", render="tanh(R phi(a) + noise)"),
           "transfer_error": {}, "probe_r2_human": {}}
    for cond in CONDS:
        res["transfer_error"][cond] = {}
        res["probe_r2_human"][cond] = {}
        for th in THETAS:
            vals = [run(cond, th, s) for s in SEEDS]
            tr = [v[1] for v in vals]
            pr = [v[0] for v in vals]
            res["transfer_error"][cond][str(th)] = {"mean": round(float(np.mean(tr)), 4),
                                                    "per_seed": [round(x, 4) for x in tr]}
            res["probe_r2_human"][cond][str(th)] = {"mean": round(float(np.mean(pr)), 4),
                                                    "per_seed": [round(x, 4) for x in pr]}
            print(cond, th, "transfer", round(float(np.mean(tr)), 3), "probeR2", round(float(np.mean(pr)), 3), flush=True)
    res["wall_time_s"] = round(time.time() - t0, 1)
    (ROOT / "analysis" / "extra_alignment_nonlinear_results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1))
    print("wall", res["wall_time_s"], "s")


if __name__ == "__main__":
    main()
