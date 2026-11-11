"""추가 검증 (dexforce): 모의 kinesthetic 시연으로 학습한 소형 조건부 확산 정책(MuJoCo 누름 과제).

실행: .venv/bin/python analysis/extra_small_diffusion.py
결과: analysis/extra_small_diffusion_results.json
(이 스크립트는 extra_kinesthetic_mujoco.py의 시연·모델을 가져다 쓴다.)

- 과제: 1 자유도 팁을 벽(강성 k_e, 무작위 1e3–1e4 N/m)에 눌러 정상상태 접촉력을 2–4 N에 맞춘다.
- 시연: 가상 leader로 3 N을 맞춘 kinesthetic 시연 100개. 목표 행동은 두 가지다.
  관측 위치(k_f = 0): 목표 = x_o(t). 힘 목표(k_f = 1/k_p): 목표 = x_o(t) + f(t)/k_p.
- 관측 세 가지: 힘 없음(위치·속도만, RGB 대체), F/T(힘 값 + 잡음 0.05 N), 0/1(접촉 여부만).
- 정책: 조건부 DDPM(T = 30 단계, MLP 128×3, 시간 임베딩). 출력은 4스텝 행동 chunk(상대 목표).
  정책은 20 ms마다 다시 계획하고 첫 행동을 유지한다.
- 평가: 시도 30회(학습 분포), 시험 시드 3개, 추가로 무른 학습 밖 강성 3e4 N/m 10회.
  성공 = 마지막 0.5 s 평균 접촉력이 2–4 N.
- 학습 시드 3개, 장치는 CPU(작은 MLP라 MPS로 옮기지 않음).
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"

import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn

import extra_kinesthetic_mujoco as K

torch.set_num_threads(4)

RESULTS = Path(__file__).resolve().parent / "extra_small_diffusion_results.json"
KP = K.KP
DT = K.DT
DEMO_STEPS = int(K.DEMO_S / DT)
PERIOD = 40                 # 정책 재계획 주기 (스텝, 20 ms)
CHUNK = 4
STRIDE = 40                 # chunk 원소 간격 (스텝)
SAMPLE_EVERY = 10
T_DIFF = 30
F_WIN = (2.0, 4.0)
N_DEMO, N_TEST, N_OOD = 100, 30, 10
SEEDS = [0, 1, 2]
VARIANTS = {
    "관측 위치 목표 + 힘 없음": ("pos", "none"),
    "관측 위치 목표 + F/T": ("pos", "ft"),
    "관측 위치 목표 + 0/1": ("pos", "flag"),
    "힘 목표 + 힘 없음": ("force", "none"),
    "힘 목표 + F/T": ("force", "ft"),
    "힘 목표 + 0/1": ("force", "flag"),
}


def sensing(x, v, F, kind, rng):
    if kind == "none":
        return np.array([x / 0.02, v / 0.05])
    if kind == "ft":
        return np.array([x / 0.02, v / 0.05, (F + rng.normal(0, 0.05)) / 3.0])
    return np.array([x / 0.02, v / 0.05, float(F + rng.normal(0, 0.05) > 0.2)])


def build_dataset(demos, target_kind, sense_kind, rng):
    X, Y = [], []
    for xo, f in demos:
        n = len(xo)
        tgt = xo if target_kind == "pos" else xo + f / KP
        v = np.gradient(xo, DT)
        for t in range(0, n - STRIDE * (CHUNK + 1) - 1, SAMPLE_EVERY):
            X.append(sensing(xo[t], v[t], f[t], sense_kind, rng))
            # chunk 원소 k는 t + STRIDE(k+1) 시점의 목표다(k = 0은 실행되는 첫 목표; 관측 위치 목표에서 현재값은 0이라 제외)
            chunk = [tgt[t + STRIDE * (k + 1)] - xo[t] for k in range(CHUNK)]
            Y.append(chunk)
    return np.array(X, np.float32), np.array(Y, np.float32)


class Denoiser(nn.Module):
    def __init__(self, dobs, dact):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dobs + dact + 16, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(),
                                 nn.Linear(128, 128), nn.GELU(), nn.Linear(128, dact))

    def forward(self, a_t, obs, t_emb):
        return self.net(torch.cat([a_t, obs, t_emb], -1))


def t_embed(t, dim=16):
    freqs = torch.exp(torch.linspace(0, np.log(100.0), dim // 2))
    ang = t[:, None].float() * freqs[None] / T_DIFF
    return torch.cat([torch.sin(ang), torch.cos(ang)], -1)


def schedule():
    betas = torch.linspace(1e-4, 0.1, T_DIFF)
    alphas = 1 - betas
    abar = torch.cumprod(alphas, 0)
    return betas, alphas, abar


def train(X, Y, seed, steps=2500, bs=256):
    torch.manual_seed(seed)
    xm, xs = X.mean(0), np.maximum(X.std(0), 0.2)
    ys = np.maximum(Y.std(0), 1e-4).astype(np.float32)  # 행동은 각 칸의 표준편차로 나눈다(mm 단위 목표 변화)
    Yn = torch.tensor(Y / ys)
    Xn = torch.tensor((X - xm) / xs)
    betas, alphas, abar = schedule()
    net = Denoiser(X.shape[1], Y.shape[1] if Y.ndim == 2 else CHUNK)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Yf = Yn.reshape(len(Yn), -1)
    for _ in range(steps):
        idx = torch.randint(0, len(Xn), (bs,))
        y0 = Yf[idx]
        t = torch.randint(0, T_DIFF, (bs,))
        eps = torch.randn_like(y0)
        ab = abar[t][:, None]
        yt = ab.sqrt() * y0 + (1 - ab).sqrt() * eps
        pred = net(yt, Xn[idx], t_embed(t))
        loss = ((pred - eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    return net, dict(xm=xm, xs=xs, ys=ys)


@torch.no_grad()
def sample(net, norm, obs):
    betas, alphas, abar = schedule()
    x = torch.randn(1, CHUNK)
    o = torch.tensor(((obs - norm["xm"]) / norm["xs"])[None], dtype=torch.float32)
    for t in reversed(range(T_DIFF)):
        te = t_embed(torch.tensor([t]))
        eps = net(x, o, te)
        a = alphas[t]
        ab = abar[t]
        x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
        x0 = x0.clamp(-5.0, 5.0)
        mean = (abar[t - 1] if t > 0 else torch.tensor(1.0))
        coef1 = betas[t] * mean.sqrt() / (1 - ab)
        coef2 = (1 - mean) * a.sqrt() / (1 - ab)
        mu = coef1 * x0 + coef2 * x
        if t > 0:
            var = betas[t] * (1 - mean) / (1 - ab)
            x = mu + var.sqrt() * torch.randn_like(x)
        else:
            x = mu
    return x.numpy()[0] * norm["ys"]


def rollout(net, norm, ke, sense_kind, rng, n_steps=DEMO_STEPS):
    m = mujoco.MjModel.from_xml_string(K.PRESS_XML)
    d = mujoco.MjData(m)
    x = -0.02
    d.qpos[0] = x
    d.ctrl[0] = x
    mujoco.mj_forward(m, d)
    Fs = np.zeros(n_steps)
    target = x
    v = 0.0
    chunk = None
    for i in range(n_steps):
        xc = d.qpos[0]
        v = d.qvel[0]
        F = ke * max(0.0, xc)
        Fs[i] = F
        if i % PERIOD == 0:
            obs = sensing(xc, v, F, sense_kind, rng)
            chunk = sample(net, norm, obs)
            target = xc + chunk[0]
        d.ctrl[0] = target
        d.qfrc_applied[0] = -F
        mujoco.mj_step(m, d)
    tail = Fs[-int(0.5 / DT):]
    return float(tail.mean()), bool(F_WIN[0] <= tail.mean() <= F_WIN[1])


def main():
    t_all = time.perf_counter()
    rng_global = np.random.default_rng(2026)
    ke_train = np.exp(rng_global.uniform(np.log(1e3), np.log(1e4), N_DEMO))
    demos = [K.press_demo(float(ke))[:2] for ke in ke_train]
    demo_steady = [float(f[-int(0.5 / DT):].mean()) for _, f in demos]
    out = {"env": {"torch": torch.__version__, "mujoco": mujoco.__version__, "numpy": np.__version__,
                   "device": "cpu", "threads": 4, "n_demo": N_DEMO, "demo_k_e_range": [1e3, 1e4],
                   "demo_force_mean_N": round(float(np.mean(demo_steady)), 3),
                   "demo_force_sd_N": round(float(np.std(demo_steady)), 3),
                   "diffusion_T": T_DIFF, "chunk": CHUNK, "policy_period_ms": PERIOD * DT * 1e3,
                   "success_window_N": list(F_WIN), "seeds": SEEDS, "n_test": N_TEST, "n_ood": N_OOD,
                   "train_steps": 2500, "batch": 256,
                   "physics": "k_e는 팁에 가하는 한쪽 스프링 외력(MuJoCo 접촉 아님)"},
           "variants": {}}
    for name, (target_kind, sense_kind) in VARIANTS.items():
        tt = time.perf_counter()
        rng_ds = np.random.default_rng(7)
        X, Y = build_dataset(demos, target_kind, sense_kind, rng_ds)
        per_seed = []
        ood_seed = []
        for seed in SEEDS:
            net, norm = train(X, Y, seed)
            rng_eval = np.random.default_rng(1000 + seed)
            ke_test = np.exp(rng_eval.uniform(np.log(1e3), np.log(1e4), N_TEST))
            res = [rollout(net, norm, float(ke), sense_kind, rng_eval) for ke in ke_test]
            per_seed.append(dict(success=float(np.mean([ok for _, ok in res])),
                                 mean_force_N=float(np.mean([F for F, _ in res]))))
            ood = [rollout(net, norm, 3e4, sense_kind, rng_eval) for _ in range(N_OOD)]
            ood_seed.append(dict(success=float(np.mean([ok for _, ok in ood])),
                                 mean_force_N=float(np.mean([F for F, _ in ood]))))
        out["variants"][name] = {
            "target": "관측 위치" if target_kind == "pos" else "힘 목표(k_f = 1/k_p)",
            "sensing": {"none": "힘 없음", "ft": "F/T", "flag": "0/1"}[sense_kind],
            "train_samples": int(len(X)),
            "success_mean": round(float(np.mean([p["success"] for p in per_seed])), 3),
            "success_by_seed": [round(p["success"], 3) for p in per_seed],
            "mean_force_N": round(float(np.mean([p["mean_force_N"] for p in per_seed])), 3),
            "ood_3e4_success_mean": round(float(np.mean([p["success"] for p in ood_seed])), 3),
            "ood_3e4_mean_force_N": round(float(np.mean([p["mean_force_N"] for p in ood_seed])), 3),
            "wall_s": round(time.perf_counter() - tt, 1),
        }
        print(name, out["variants"][name], flush=True)
    out["wall_time_s"] = round(time.perf_counter() - t_all, 1)
    RESULTS.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("wall", out["wall_time_s"], "s", flush=True)


if __name__ == "__main__":
    main()
