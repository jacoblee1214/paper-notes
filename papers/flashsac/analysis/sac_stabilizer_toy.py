"""SAC 토이: FlashSAC의 안정화 요소(단위 노름 가중치, RMSNorm 특징, 배치 정규화 + 교차 배치)가 선형-이차 제어(LQR) 과제에서 SAC를 바꾸는지 본다.

과제: 4차원 상태, 2차원 행동(tanh로 [-1, 1]), 이산 시간 선형 동역학 x' = A x + B u + 0.05 w, 보상 -(xᵀQx + 0.1 uᵀu), 구간 100 step.
병렬 환경 16개에서 수집한다. 시스템(A, B)은 시드로 정해져서 설정끼리 같은 과제를 푼다.

설정 (모두 같은 환경 step 수 60,000):
  base      : SAC 기준선. MLP 2×256 ReLU, 배치 256, 환경 step 16개당 업데이트 16회(UTD 1)
  norm      : base + 단위 노름 가중치 투영(매 업데이트 뒤), RMSNorm 특징, 같은 배치와 UTD
  norm_bn   : norm + 은닉층 배치 정규화(BN, 모멘텀 0.01), 현재와 다음 상태를 한 배치로 묶어 계산
  fs_sched  : norm_bn + 배치 2048, 환경 step 16개당 업데이트 1회(UTD 1/16). 논문의 "큰 배치, 적은 업데이트" 방향

측정: 결정적 평가 반환(64 시작 상태, 100 step) 곡선, 벽시계 시간, 비판자 가중치 노름,
과대추정 지표 Q(s0, π(s0)) - MC 할인 반환(16 시작 상태, 같은 정책).

실행:
  python3 analysis/sac_stabilizer_toy.py --all            # 4 설정 × 3 시드, 병렬 실행 → analysis/sac_stabilizer_toy_results.json
  python3 analysis/sac_stabilizer_toy.py --config base --seed 0 --out X.json   # 한 번
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "sac_stabilizer_toy_results.json"
N, M = 4, 2
E = 16  # 병렬 환경 수
TOTAL_STEPS = 60_000
EVAL_EVERY = 4_000
GAMMA, TAU, LR = 0.99, 0.005, 3e-4
HORIZON = 100
CONFIGS = ["base", "norm", "norm_bn", "fs_sched"]


def make_system(seed):
    rng = np.random.default_rng(1000 + seed)
    A = rng.normal(size=(N, N))
    A = 0.9 * A / max(abs(np.linalg.eigvals(A)))  # 안정한 자연 동역학(스펙트럼 반지름 0.9)
    B = rng.normal(size=(N, M)) / 2
    return torch.tensor(A, dtype=torch.float32), torch.tensor(B, dtype=torch.float32)


Q_MAT = torch.eye(N)
R_MAT = 0.1 * torch.eye(M)


class LQREnv:
    """병렬 LQR. 상태 X (E, N). 보상은 음의 이차 비용."""

    def __init__(self, A, B, n_envs, gen):
        self.A, self.B, self.n = A, B, n_envs
        self.gen = gen
        self.reset()

    def reset(self):
        self.X = torch.randn(self.n, N, generator=self.gen) * 0.5
        self.t = 0
        return self.X.clone()

    def step(self, U):
        cost = (self.X @ Q_MAT * self.X).sum(-1) + (U @ R_MAT * U).sum(-1)
        Xn = self.X @ self.A.T + U @ self.B.T + 0.05 * torch.randn(self.n, N, generator=self.gen)
        self.X = Xn
        self.t += 1
        done_trunc = self.t >= HORIZON
        return Xn.clone(), -cost, done_trunc


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt((x * x).mean(-1, keepdim=True) + self.eps) * self.g


class MLP(nn.Module):
    """hidden = 256, 은닉층 2개. norm=True면 마지막 특징에 RMSNorm, bn=True면 각 은닉층에 BN(ReLU 앞)."""

    def __init__(self, din, dout, norm, bn):
        super().__init__()
        self.l1 = nn.Linear(din, 256, bias=not bn)
        self.l2 = nn.Linear(256, 256, bias=not bn)
        self.bn1 = nn.BatchNorm1d(256, momentum=0.01) if bn else None
        self.bn2 = nn.BatchNorm1d(256, momentum=0.01) if bn else None
        self.final = RMSNorm(256) if norm else None
        self.head = nn.Linear(256, dout)
        self.norm = norm

    def forward(self, x):
        h = self.l1(x)
        h = self.bn1(h) if self.bn1 is not None else h
        h = F.relu(h)
        h = self.l2(h)
        h = self.bn2(h) if self.bn2 is not None else h
        h = F.relu(h)
        if self.final is not None:
            h = self.final(h)
        return self.head(h)

    @torch.no_grad()
    def project(self):
        """단위 노름: 각 선형층의 출력 유닛 가중치 벡터(입력 방향)를 단위 노름으로 투영."""
        for m in [self.l1, self.l2, self.head]:
            m.weight.copy_(F.normalize(m.weight, dim=-1, eps=1e-8))
        if self.final is not None:
            g = self.final.g
            self.final.g.copy_(g * math.sqrt(g.numel()) / torch.sqrt((g * g).sum() + 1e-8))

    def feature_norm(self, x):
        h = self.l1(x)
        h = self.bn1(h) if self.bn1 is not None else h
        h = F.relu(h)
        h = self.l2(h)
        h = self.bn2(h) if self.bn2 is not None else h
        h = F.relu(h)
        if self.final is not None:
            h = self.final(h)
        return h.norm(dim=-1).mean().item()


class Actor(nn.Module):
    def __init__(self, norm, bn):
        super().__init__()
        self.net = MLP(N, 2 * M, norm, bn)

    def dist(self, s):
        out = self.net(s)
        mu, log_std = out[..., :M], out[..., M:].clamp(-5, 2)
        return mu, log_std.exp()

    def sample(self, s):
        mu, std = self.dist(s)
        eps = torch.randn_like(mu)
        u = mu + std * eps
        a = torch.tanh(u)
        logp = (-0.5 * eps ** 2 - std.log() - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = logp - (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return a, logp

    def act_det(self, s):
        mu, _ = self.dist(s)
        return torch.tanh(mu)


def run(config, seed, out_path):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    A, B = make_system(seed)
    gen = torch.Generator().manual_seed(seed)
    norm = config in {"norm", "norm_bn", "fs_sched"}
    bn = config in {"norm_bn", "fs_sched"}
    batch = 2048 if config == "fs_sched" else 256
    upd_per_iter = 1 if config == "fs_sched" else E  # 환경 step E개당 업데이트 수

    actor = Actor(norm, bn)
    q1, q2 = MLP(N + M, 1, norm, bn), MLP(N + M, 1, norm, bn)
    q1t, q2t = MLP(N + M, 1, norm, bn), MLP(N + M, 1, norm, bn)
    q1t.load_state_dict(q1.state_dict())
    q2t.load_state_dict(q2.state_dict())
    log_alpha = torch.zeros(1, requires_grad=True)
    opt_a = torch.optim.Adam(actor.parameters(), lr=LR)
    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=LR)
    opt_al = torch.optim.Adam([log_alpha], lr=LR)
    target_ent = -float(M)

    n_iter = TOTAL_STEPS // E
    buf_s = torch.zeros(TOTAL_STEPS, N)
    buf_a = torch.zeros(TOTAL_STEPS, M)
    buf_r = torch.zeros(TOTAL_STEPS)
    buf_s2 = torch.zeros(TOTAL_STEPS, N)
    ptr = 0
    env = LQREnv(A, B, E, gen)
    S = env.reset()

    eval_gen = torch.Generator().manual_seed(999 + seed)
    eval_s0 = torch.randn(64, N, generator=eval_gen) * 0.5
    diag_s0 = eval_s0[:16]

    def set_mode(train):
        for m in [actor, q1, q2, q1t, q2t]:
            m.train(train)

    def evaluate():
        """결정적 정책의 할인 없는 반환(64개 시작 상태, 100 step)."""
        set_mode(False)
        with torch.no_grad():
            X, total = eval_s0.clone(), torch.zeros(64)
            for _ in range(HORIZON):
                U = actor.act_det(X)
                total += -((X @ Q_MAT * X).sum(-1) + (U @ R_MAT * U).sum(-1))
                X = X @ A.T + U @ B.T
        set_mode(True)
        return float(total.mean())

    def q_bias():
        """Q(s0, π(s0)) - 같은 정책의 할인 MC 반환(16 시작 상태, 600 step: γ^600 ≈ 0.002). 양수면 과대추정."""
        set_mode(False)
        with torch.no_grad():
            X = diag_s0.clone()
            a0 = actor.act_det(X)
            x0 = torch.cat([X, a0], -1)
            qv = torch.min(q1(x0), q2(x0)).squeeze(-1)
            Y, ret, disc = X.clone(), torch.zeros(16), 1.0
            for _ in range(600):
                U = actor.act_det(Y)
                ret += disc * -((Y @ Q_MAT * Y).sum(-1) + (U @ R_MAT * U).sum(-1))
                disc *= GAMMA
                Y = Y @ A.T + U @ B.T
        set_mode(True)
        return float((qv - ret).mean())

    curve = [{"env_steps": 0, "return": evaluate(), "time_s": 0.0, "updates": 0}]
    t0 = time.time()
    cum_updates = 0
    for it in range(1, n_iter + 1):
        with torch.no_grad():  # 행동 선택은 BN 이동 평균을 쓴다(공식 코드 sample_actions의 training=False)
            actor.eval()
            U, _ = actor.sample(S)
            actor.train()
        S2, R, trunc = env.step(U)
        lo, hi = ptr, ptr + E
        buf_s[lo:hi], buf_a[lo:hi], buf_r[lo:hi], buf_s2[lo:hi] = S, U, R, S2
        ptr = hi
        S = env.reset() if trunc else S2
        steps_done = it * E
        if steps_done < 2_000:  # 워밍업
            continue
        for _ in range(upd_per_iter):
            idx = torch.randint(0, ptr, (batch,), generator=gen)
            Sb, Ab, Rb, S2b = buf_s[idx], buf_a[idx], buf_r[idx], buf_s2[idx]
            alpha = log_alpha.exp().detach()
            with torch.no_grad():  # 표적 행동도 이동 평균 모드(공식 코드 update_critic의 actor(training=False))
                actor.eval()
                A2, logp2 = actor.sample(S2b)
                actor.train()
                x_cur = torch.cat([Sb, Ab], -1)
                x_nxt = torch.cat([S2b, A2], -1)
                if bn:  # 교차 배치: 현재와 다음 쌍을 한 번에 통과시켜 BN 통계를 공유
                    both = torch.cat([x_cur, x_nxt], 0)
                    qt = torch.min(q1t(both)[batch:], q2t(both)[batch:]).squeeze(-1)
                else:
                    qt = torch.min(q1t(x_nxt), q2t(x_nxt)).squeeze(-1)
                y = Rb + GAMMA * (qt - alpha * logp2)
            if bn:
                both = torch.cat([x_cur, x_nxt], 0)
                loss_q = F.mse_loss(q1(both)[:batch].squeeze(-1), y) + F.mse_loss(q2(both)[:batch].squeeze(-1), y)
            else:
                loss_q = F.mse_loss(q1(x_cur).squeeze(-1), y) + F.mse_loss(q2(x_cur).squeeze(-1), y)
            opt_q.zero_grad()
            loss_q.backward()
            opt_q.step()
            if norm:
                q1.project()
                q2.project()
            a_new, logp = actor.sample(Sb)
            xa = torch.cat([Sb, a_new], -1)
            qn = torch.min(q1(xa), q2(xa)).squeeze(-1)
            loss_pi = (alpha * logp - qn).mean()
            opt_a.zero_grad()
            loss_pi.backward()
            opt_a.step()
            if norm:
                actor.net.project()
            loss_al = -(log_alpha * (logp.detach() + target_ent)).mean()
            opt_al.zero_grad()
            loss_al.backward()
            opt_al.step()
            with torch.no_grad():
                for pt, p in zip(list(q1t.parameters()) + list(q2t.parameters()),
                                 list(q1.parameters()) + list(q2.parameters())):
                    pt.lerp_(p, TAU)
            cum_updates += 1
        if steps_done % EVAL_EVERY == 0:
            rec = {"env_steps": steps_done, "return": evaluate(), "time_s": time.time() - t0,
                   "updates": cum_updates, "q_bias": q_bias()}
            curve.append(rec)
    res = {"config": config, "seed": seed, "curve": curve, "time_total_s": time.time() - t0,
           "updates_total": cum_updates, "batch": batch, "updates_per_iter": upd_per_iter,
           "final_return": curve[-1]["return"], "final_q_bias": curve[-1]["q_bias"]}
    Path(out_path).write_text(json.dumps(res))
    print(config, seed, "final", round(res["final_return"], 3), "time", round(res["time_total_s"], 1), flush=True)


def run_all():
    tasks = [(c, s) for c in CONFIGS for s in range(3)]
    outdir = ROOT / ".scratch" / "toy_runs"
    outdir.mkdir(exist_ok=True)
    procs, results = [], []
    for c, s in tasks:
        out = outdir / f"{c}_s{s}.json"
        cmd = [sys.executable, "-I", str(Path(__file__).resolve()), "--config", c, "--seed", str(s), "--out", str(out)]
        procs.append((c, s, out, subprocess.Popen(cmd)))
        while sum(p.poll() is None for *_, p in procs) >= 4:
            time.sleep(1)
    for c, s, out, p in procs:
        p.wait()
        results.append(json.loads(out.read_text()))
    summary = {}
    for c in CONFIGS:
        rs = [r for r in results if r["config"] == c]
        fin = [r["final_return"] for r in rs]
        summary[c] = {"final_return_per_seed": fin, "final_mean": float(np.mean(fin)), "final_sd": float(np.std(fin, ddof=1)),
                      "time_total_s_mean": float(np.mean([r["time_total_s"] for r in rs])),
                      "updates_total": rs[0]["updates_total"], "batch": rs[0]["batch"],
                      "final_q_bias_mean": float(np.mean([r["final_q_bias"] for r in rs]))}
    base0 = [r["curve"][0]["return"] for r in results if r["config"] == "base"]
    out = {"meta": {"N": N, "M": M, "E": E, "total_env_steps": TOTAL_STEPS, "horizon": HORIZON, "gamma": GAMMA,
                    "tau": TAU, "lr": LR, "seeds": [0, 1, 2], "initial_return_mean": float(np.mean(base0)),
                    "note": "시드 3개, 설정별 최종 평가 반환은 같은 60k 환경 step에서 측정. 벽시계 시간은 CPU 1스레드 기준"},
           "summary": summary, "runs": results}
    OUT.write_text(json.dumps(out, indent=1))
    for c in CONFIGS:
        print(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary[c].items() if k != "final_return_per_seed"})
    print("wrote", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--config", choices=CONFIGS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str)
    a = ap.parse_args()
    if a.all:
        run_all()
    else:
        run(a.config, a.seed, a.out)
