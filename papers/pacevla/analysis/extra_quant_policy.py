"""양자화 명령 toy: 같은 추적 이득에서 기준 / 사후 양자화 / 학습된 양자화 decoder / 물리 판정 라우터 비교 (torch CPU).

환경: 2차원 손끝이 임계감쇠 PD(이득 kp)로 명령을 따라간다. 시간 간격 1/60 s, 240 step.
시연: 시작 → 쥐는 점 g (80 step 도달, 20 step 정지) → 놓는 점 r (180 step 도달, 60 step 정지), 명령에 0.5 mm 떨림.
성공: 80 step에서 손끝과 g의 거리 < 4 mm 이고, 180 step에서 손끝과 r의 거리 < 4 mm.
명령은 q step 동안 유지한다(블록 끝값으로 양자화된 명령).

정책: 조건부 flow matching(조건: 손끝 위치·속도, 목표 g·r, 구간 시작 자세(관측 가능하다고 가정), 시간)으로 H = 8 개의 미래 명령 chunk를 낸다.
- fine: q = 1 로 학습한 정책(chunk 원소 j = 시각 t+j의 시연 위치).
- coarse(학습된 양자화 decoder): q = 2 블록 끝값을 직접 학습한 정책(원소 j = 블록 j의 끝 위치).
- 네 가지 비교: base = fine, q = 1 / posthoc = fine을 q = 2 로 실행하되 블록 끝값(원소 1) 사용 /
  decoder = coarse, q = 2 / router = 물리 판정(목표까지 거리 < 3 cm 이면 fine q = 1, 아니면 coarse q = 2).
가정(toy 설정): 목표 g, r ∈ [0.3, 0.7]² m, 학습 시연 3000개, 정책 학습 1500 step, 시드 2개.
정책 학습 상태 분포는 같은 kp에서 시연 명령을 추적한 궤적에서 얻는다.
실행: .venv/bin/python analysis/extra_quant_policy.py → analysis/extra_quant_policy_results.json
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
DT, T_END, H = 1 / 60, 240, 8
T_GRASP, T_RELEASE = 80, 180
KPS = [150.0, 300.0]
N_DEMO, N_EVAL, TRAIN_STEPS, SEEDS, JIT = 3000, 300, 8000, [0, 1], 0.0005
ROUTER_D, SUCC_TOL = 0.03, 0.004
FM_STEPS = 16
SC = 0.2  # 위치 정규화 척도(m): 목표와 손끝 위치를 평균 0.5 m, 척도 0.2 m로 표준화
NOISE = 0.02  # 학습 데이터 수집 때 명령에 더하는 잡음(m, 매 step): 벗어난 상태를 학습에 포함(DART식). 0.01·0.003에서는 닫힌 루프가 무너짐
SC_R = 0.05  # 명령 잔차(명령 − 현재 손끝 위치)의 척도(m)


def min_jerk(s, g, tau):
    w = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    return s[:, None] + (g - s)[:, None] * w[None, :, None]


def demo_ref(s, g, r, gen, kp):
    """시연 명령 (N, 240, 2). 깨끗한 경로(241점)를 PD가 정확히 따라가도록 역동역학으로 명령을 만들고 떨림을 더한다."""
    tt = torch.linspace(0, 1, 80)
    path = torch.cat([min_jerk(s, g, tt),                       # 0..79
                      g[:, None].expand(-1, 20, -1),             # 80..99
                      min_jerk(g, r, tt),                        # 100..179
                      r[:, None].expand(-1, 61, -1)], 1)         # 180..240
    kd = 2 * math.sqrt(kp)
    vel = torch.zeros_like(path)
    vel[:, 1:] = (path[:, 1:] - path[:, :-1]) / DT
    acc = (vel[:, 1:] - vel[:, :-1]) / DT                        # a_t, t = 0..239
    cmd = path[:, :T_END] + (acc + kd * vel[:, :T_END]) / kp
    return cmd + JIT * torch.randn(cmd.shape, generator=gen)


def pd_step(x, v, u, kp):
    kd = 2 * math.sqrt(kp)
    a = kp * (u - x) - kd * v
    v = v + a * DT
    x = x + v * DT
    return x, v


def cond_feat(x, v, g, r, s0, t):
    tt = torch.full((x.shape[0], 1), t / T_END)
    return torch.cat([(x - 0.5) / SC, v / 0.5, (g - 0.5) / SC, (r - 0.5) / SC, (s0 - 0.5) / SC, tt], -1)


class FlowPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        cin = 2 * 5 + 1 + 1 + H * 2  # 조건 11 + 시각 1 + 명령 16
        self.net = nn.Sequential(nn.Linear(cin, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                                 nn.Linear(256, 256), nn.GELU(), nn.Linear(256, H * 2))

    def velocity(self, cond, s, y):
        inp = torch.cat([cond, s[:, None], y.flatten(1)], -1)
        return self.net(inp).view(-1, H, 2)

    @torch.no_grad()
    def sample(self, cond, base):
        """base: 현재 손끝 위치. 정책은 base 대비 명령 차이(잔차)를 생성한다."""
        y = torch.randn(cond.shape[0], H, 2)
        for i in range(FM_STEPS):
            s = torch.full((cond.shape[0],), i / FM_STEPS)
            y = y + (1 / FM_STEPS) * self.velocity(cond, s, y)
        return base[:, None, :] + y * SC_R


def make_train_set(kp, seed):
    """이 kp에서 시연 명령을 추적한 상태와 정책 목표(fine/coarse 블록 원소)."""
    gen = torch.Generator().manual_seed(seed)
    s = 0.3 + 0.4 * torch.rand(N_DEMO, 2, generator=gen)
    g = 0.3 + 0.4 * torch.rand(N_DEMO, 2, generator=gen)
    r = 0.3 + 0.4 * torch.rand(N_DEMO, 2, generator=gen)
    ref = demo_ref(s, g, r, gen, kp)
    x, v = s.clone(), torch.zeros_like(s)
    conds, fine_y, coarse_y = [], [], []
    for t in range(T_END):
        conds.append(cond_feat(x, v, g, r, s, t))
        idx_f = torch.clamp(torch.arange(t, t + H), max=T_END - 1)
        idx_c = torch.clamp(torch.arange(t, t + 2 * H, 2) + 1, max=T_END - 1)
        fine_y.append((ref[:, idx_f] - x[:, None]) / SC_R)
        coarse_y.append((ref[:, idx_c] - x[:, None]) / SC_R)
        x, v = pd_step(x, v, ref[:, t] + NOISE * torch.randn(ref[:, t].shape, generator=gen), kp)
    return torch.cat(conds), torch.cat(fine_y), torch.cat(coarse_y)


def fit(cond, y, seed):
    torch.manual_seed(seed)
    pol = FlowPolicy()
    opt = torch.optim.AdamW(pol.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(cond)
    for _ in range(TRAIN_STEPS):
        b = torch.randint(0, n, (256,))
        z = torch.randn(256, H, 2)
        tau = torch.rand(256)
        yt = (1 - tau[:, None, None]) * z + tau[:, None, None] * y[b]
        loss = ((pol.velocity(cond[b], tau, yt) - (y[b] - z)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return pol.eval()


@torch.no_grad()
def run_variant(variant, kp, fine, coarse, seed):
    gen = torch.Generator().manual_seed(500 + seed)
    s = 0.3 + 0.4 * torch.rand(N_EVAL, 2, generator=gen)
    g = 0.3 + 0.4 * torch.rand(N_EVAL, 2, generator=gen)
    r = 0.3 + 0.4 * torch.rand(N_EVAL, 2, generator=gen)
    x, v = s.clone(), torch.zeros_like(s)
    s0 = s.clone()
    hold = torch.zeros(N_EVAL, dtype=torch.long)
    cmd = x.clone()
    n_cmd = torch.zeros(N_EVAL)
    ok_g = torch.zeros(N_EVAL, dtype=torch.bool)
    ok_r = torch.zeros(N_EVAL, dtype=torch.bool)
    for t in range(T_END):
        if t == T_GRASP:
            ok_g = (x - g).norm(dim=-1) < SUCC_TOL
        if t == T_RELEASE:
            ok_r = (x - r).norm(dim=-1) < SUCC_TOL
        upd = hold == 0
        if upd.any():
            cond = cond_feat(x, v, g, r, s0, t)
            if variant == "base":
                new, q = fine.sample(cond, x)[:, 0], torch.ones(N_EVAL, dtype=torch.long)
            elif variant == "posthoc":
                new, q = fine.sample(cond, x)[:, 1], 2 * torch.ones(N_EVAL, dtype=torch.long)
            elif variant == "decoder":
                new, q = coarse.sample(cond, x)[:, 0], 2 * torch.ones(N_EVAL, dtype=torch.long)
            else:  # router: 물리 판정
                goal = g if t < 100 else r
                near = (x - goal).norm(dim=-1) < ROUTER_D
                f_new = fine.sample(cond, x)[:, 0]
                c_new = coarse.sample(cond, x)[:, 0]
                new = torch.where(near[:, None], f_new, c_new)
                q = torch.where(near, torch.ones_like(hold), 2 * torch.ones_like(hold))
            cmd = torch.where(upd[:, None], new, cmd)
            hold = torch.where(upd, q, hold)
            n_cmd += upd.float()
        x, v = pd_step(x, v, cmd, kp)
        hold = hold - 1
    success = ok_g & ok_r
    return float(success.float().mean()), float(n_cmd.mean()) / T_END


def main():
    t0 = time.time()
    res = {"device": str(DEVICE), "torch": torch.__version__, "numpy": np.__version__,
           "python": platform.python_version(), "threads": torch.get_num_threads(),
           "config": dict(dt=DT, steps=T_END, horizon=H, grasp=T_GRASP, release=T_RELEASE,
                          kp=KPS, demos=N_DEMO, eval_episodes=N_EVAL, train_steps=TRAIN_STEPS,
                          seeds=SEEDS, jitter_m=JIT, success_tol_m=SUCC_TOL, router_dist_m=ROUTER_D,
                          fm_euler_steps=FM_STEPS),
           "results": {}}
    for kp in KPS:
        acc = {v: {"succ": [], "cmd_ratio": []} for v in ["base", "posthoc", "decoder", "router"]}
        for seed in SEEDS:
            cond, yf, yc = make_train_set(kp, seed)
            fine = fit(cond, yf, seed)
            coarse = fit(cond, yc, seed + 100)
            for v in acc:
                sc, cr = run_variant(v, kp, fine, coarse, seed)
                acc[v]["succ"].append(sc)
                acc[v]["cmd_ratio"].append(cr)
            print("kp", kp, "seed", seed, {v: round(acc[v]["succ"][-1], 3) for v in acc}, flush=True)
        res["results"][f"kp{int(kp)}"] = {
            v: {"success_mean": round(float(np.mean(a["succ"])), 3),
                "success_per_seed": [round(x, 3) for x in a["succ"]],
                "cmd_ratio_mean": round(float(np.mean(a["cmd_ratio"])), 3)} for v, a in acc.items()}
    res["wall_time_s"] = round(time.time() - t0, 1)
    (ROOT / "analysis" / "extra_quant_policy_results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1))
    print("wall", res["wall_time_s"], "s")


if __name__ == "__main__":
    main()
