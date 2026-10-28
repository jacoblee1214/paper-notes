"""numpy로 만든 작은 DDPG를 진자(pendulum swing-up)에서 돌려 논문 주장과 구현 간 차이를 확인한다 (CPU).
실행: python3 analysis/ddpg_toy.py   (8개 설정 × 5 시드, 병렬로 1–2분) → analysis/ddpg_toy_results.json

구조는 논문 부록 7절을 따르되 폭만 줄였다(400-300 → 64-64): ReLU, actor 출력 tanh, critic은 행동을
두 번째 은닉층에 합류, 마지막 층 초기화 U(±3e-3), 나머지 U(±1/√fan_in), Adam(actor 1e-4, critic 1e-3),
critic L2 1e-2(가중치만), γ = 0.99, 배치 64, 환경 step마다 업데이트 1번. 배치 정규화는 넣지 않았다.
환경은 gym Pendulum-v1과 같은 동역학(200 step, 토크 ±2)이고 시간 제한 종료에서는 bootstrap한다.

비교 (논문 설정 대비 하나씩만 바꿈):
  paper        τ = 0.001, OU(θ 0.15, σ 0.2, dt 1), L2 1e-2
  no_target    τ = 1 (표적망 없음 = 원래 DPG에 해당)
  tau_0.005    τ = 0.005 (SB3·TD3 저장소 기본값)
  no_l2        critic L2 = 0 (SB3·Spinning Up·TD3 저장소)
  ou_dt_0.01   OU dt = 0.01 (OpenAI baselines·SB3의 OU 기본값)
  gauss_0.1    가우시안 σ = 0.1 (Spinning Up·TD3 저장소)
  no_noise     탐색 잡음 없음
  retuned      TD3 논문의 'OurDDPG'식: lr 3e-4/3e-4, 배치 256, τ 0.005, 가우시안 0.1, L2 0
측정: 마지막 평가 return(결정적 정책, 고정 초기 상태 10개), 학습 곡선, Q(s0, μ(s0))와 1,000 step 할인 return의 차이.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STEPS, EVAL_EVERY, WARMUP = 40_000, 2_000, 1_000
H1 = H2 = 64
GAMMA = 0.99

BASE = dict(tau=0.001, noise="ou", ou_dt=1.0, sigma=0.2, l2=1e-2, lr_a=1e-4, lr_c=1e-3, batch=64)
CONFIGS = {
    "paper": {},
    "no_target": {"tau": 1.0},
    "tau_0.005": {"tau": 0.005},
    "no_l2": {"l2": 0.0},
    "ou_dt_0.01": {"ou_dt": 0.01},
    "gauss_0.1": {"noise": "gauss", "sigma": 0.1},
    "no_noise": {"noise": "none"},
    "retuned": {"lr_a": 3e-4, "lr_c": 3e-4, "batch": 256, "tau": 0.005, "noise": "gauss", "sigma": 0.1, "l2": 0.0},
}
SEEDS = (0, 1, 2, 3, 4)


# ── 환경: gym Pendulum-v1 동역학 ─────────────────────────────────────────────
class Pendulum:
    max_speed, max_torque, dt, g, m, l = 8.0, 2.0, 0.05, 10.0, 1.0, 1.0

    def __init__(self, rng):
        self.rng = rng

    def reset(self, state=None):
        self.th, self.thdot = state if state is not None else (self.rng.uniform(-np.pi, np.pi), self.rng.uniform(-1, 1))
        return self.obs()

    def obs(self):
        return np.array([math.cos(self.th), math.sin(self.th), self.thdot])

    def step(self, a_norm):
        u = float(np.clip(a_norm, -1, 1)) * self.max_torque
        ang = ((self.th + np.pi) % (2 * np.pi)) - np.pi
        cost = ang ** 2 + 0.1 * self.thdot ** 2 + 0.001 * u ** 2
        thdot = self.thdot + (3 * self.g / (2 * self.l) * math.sin(self.th) + 3.0 / (self.m * self.l ** 2) * u) * self.dt
        self.thdot = float(np.clip(thdot, -self.max_speed, self.max_speed))
        self.th = self.th + self.thdot * self.dt
        return self.obs(), -cost


# ── 신경망 (수동 역전파) ──────────────────────────────────────────────────────
def init_layer(rng, fan_in, fan_out, final=False):
    lim = 3e-3 if final else 1 / math.sqrt(fan_in)
    return [rng.uniform(-lim, lim, (fan_in, fan_out)), rng.uniform(-lim, lim, fan_out)]


class Actor:
    def __init__(self, rng):
        self.p = init_layer(rng, 3, H1) + init_layer(rng, H1, H2) + init_layer(rng, H2, 1, final=True)

    def forward(self, x):
        W1, b1, W2, b2, W3, b3 = self.p
        z1 = x @ W1 + b1; h1 = np.maximum(z1, 0)
        z2 = h1 @ W2 + b2; h2 = np.maximum(z2, 0)
        a = np.tanh(h2 @ W3 + b3)
        self.cache = (x, z1, h1, z2, h2, a)
        return a

    def backward(self, da):
        x, z1, h1, z2, h2, a = self.cache
        W1, b1, W2, b2, W3, b3 = self.p
        dz3 = da * (1 - a ** 2)
        dh2 = dz3 @ W3.T; dz2 = dh2 * (z2 > 0)
        dh1 = dz2 @ W2.T; dz1 = dh1 * (z1 > 0)
        return [x.T @ dz1, dz1.sum(0), h1.T @ dz2, dz2.sum(0), h2.T @ dz3, dz3.sum(0)]


class Critic:
    def __init__(self, rng):
        self.p = init_layer(rng, 3, H1) + init_layer(rng, H1 + 1, H2) + init_layer(rng, H2, 1, final=True)

    def forward(self, x, a):
        W1, b1, W2, b2, W3, b3 = self.p
        z1 = x @ W1 + b1; h1 = np.maximum(z1, 0)
        c = np.concatenate([h1, a], axis=1)  # 행동은 두 번째 은닉층에 합류
        z2 = c @ W2 + b2; h2 = np.maximum(z2, 0)
        q = h2 @ W3 + b3
        self.cache = (x, z1, c, z2, h2)
        return q

    def backward(self, dq):
        x, z1, c, z2, h2 = self.cache
        W1, b1, W2, b2, W3, b3 = self.p
        dh2 = dq @ W3.T; dz2 = dh2 * (z2 > 0)
        dc = dz2 @ W2.T
        dz1 = dc[:, :H1] * (z1 > 0)
        grads = [x.T @ dz1, dz1.sum(0), c.T @ dz2, dz2.sum(0), h2.T @ dq, dq.sum(0)]
        return grads, dc[:, H1:]


class Adam:
    def __init__(self, params, lr):
        self.lr, self.t = lr, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params, grads, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1 ** self.t)
            vh = self.v[i] / (1 - b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + eps)


def soft_update(target, source, tau):
    for t, s in zip(target.p, source.p):
        t *= 1 - tau
        t += tau * s


# ── 평가 ────────────────────────────────────────────────────────────────────
EVAL_STATES = [(th, 0.0) for th in np.linspace(-np.pi, np.pi, 10, endpoint=False) + np.pi / 10]


def evaluate(actor, horizon=200):
    env = Pendulum(np.random.default_rng(0))
    rets = []
    for s in EVAL_STATES:
        o = env.reset(s); R = 0.0
        for _ in range(horizon):
            o, r = env.step(actor.forward(o[None])[0, 0]); R += r
        rets.append(R)
    return float(np.mean(rets))


def q_bias(actor, critic):
    """Q(s0, μ(s0)) − Σ_{t<1000} γ^t r_t (결정적 정책, 시간 제한 없이 굴린 할인 return)."""
    env = Pendulum(np.random.default_rng(0))
    est, true = [], []
    for s in EVAL_STATES:
        o = env.reset(s)
        est.append(float(critic.forward(o[None], actor.forward(o[None]))[0, 0]))
        G, disc = 0.0, 1.0
        for _ in range(1000):
            o, r = env.step(actor.forward(o[None])[0, 0]); G += disc * r; disc *= GAMMA
        true.append(G)
    est, true = np.array(est), np.array(true)
    return {"mean_estimate": round(float(est.mean()), 1), "mean_true": round(float(true.mean()), 1),
            "mean_bias": round(float((est - true).mean()), 1),
            "rel_bias": round(float((est - true).mean() / abs(true.mean())), 3)}


def grad_check(eps=1e-6):
    """수동 역전파를 중앙 차분과 비교한 최대 상대 오차 (critic 파라미터, critic의 행동 입력, actor 파라미터)."""
    rng = np.random.default_rng(7)
    actor, critic = Actor(rng), Critic(rng)
    for p in actor.p + critic.p:  # 마지막 층이 너무 작으면 상대 오차가 불안정하므로 키운다
        p += rng.normal(0, 0.3, p.shape)
    x, a, y = rng.normal(size=(5, 3)), rng.uniform(-1, 1, (5, 1)), rng.normal(size=(5, 1))

    def loss_c():
        return float(((critic.forward(x, a) - y) ** 2).mean())

    q = critic.forward(x, a)
    gc, da = critic.backward(2 * (q - y) / len(x))
    errs = []
    for P, G in zip(critic.p, gc):
        i = tuple(rng.integers(0, s) for s in P.shape)
        P[i] += eps; lp = loss_c(); P[i] -= 2 * eps; lm = loss_c(); P[i] += eps
        errs.append(abs((lp - lm) / (2 * eps) - G[i]) / max(1e-8, abs(G[i]) + abs((lp - lm) / (2 * eps))))
    a[2, 0] += eps; lp = loss_c(); a[2, 0] -= 2 * eps; lm = loss_c(); a[2, 0] += eps
    errs.append(abs((lp - lm) / (2 * eps) - da[2, 0]) / max(1e-8, abs(da[2, 0])))

    def loss_a():
        return float(-critic.forward(x, actor.forward(x)).mean())

    critic.forward(x, actor.forward(x))
    _, dq_da = critic.backward(-np.ones((len(x), 1)) / len(x))
    ga = actor.backward(dq_da)
    for P, G in zip(actor.p, ga):
        i = tuple(rng.integers(0, s) for s in P.shape)
        P[i] += eps; lp = loss_a(); P[i] -= 2 * eps; lm = loss_a(); P[i] += eps
        errs.append(abs((lp - lm) / (2 * eps) - G[i]) / max(1e-8, abs(G[i]) + abs((lp - lm) / (2 * eps))))
    return float(max(errs))


# ── 학습 ────────────────────────────────────────────────────────────────────
def run(job):
    name, seed = job
    cfg = {**BASE, **CONFIGS[name]}
    rng = np.random.default_rng(1000 + seed)
    env = Pendulum(rng)
    actor, critic = Actor(rng), Critic(rng)
    actor_t, critic_t = Actor(rng), Critic(rng)
    actor_t.p = [p.copy() for p in actor.p]
    critic_t.p = [p.copy() for p in critic.p]
    opt_a, opt_c = Adam(actor.p, cfg["lr_a"]), Adam(critic.p, cfg["lr_c"])

    S = np.zeros((STEPS, 3)); A = np.zeros((STEPS, 1)); R = np.zeros((STEPS, 1)); S2 = np.zeros((STEPS, 3))
    o, ep_t, noise = env.reset(), 0, 0.0
    curve, diverged = [], False
    t0 = time.time()
    for t in range(STEPS):
        a = actor.forward(o[None])[0, 0]
        if cfg["noise"] == "ou":
            dt = cfg["ou_dt"]
            noise = noise - 0.15 * noise * dt + cfg["sigma"] * math.sqrt(dt) * rng.standard_normal()
        elif cfg["noise"] == "gauss":
            noise = cfg["sigma"] * rng.standard_normal()
        a_exec = float(np.clip(a + noise, -1, 1))
        o2, r = env.step(a_exec)
        S[t], A[t], R[t], S2[t] = o, a_exec, r, o2
        o, ep_t = o2, ep_t + 1
        if ep_t == 200:  # 시간 제한: 종료가 아니므로 bootstrap 유지, 잡음 과정은 새로 시작
            o, ep_t, noise = env.reset(), 0, 0.0

        if t >= WARMUP:
            idx = rng.integers(0, t + 1, cfg["batch"])
            s, act, rew, s2 = S[idx], A[idx], R[idx], S2[idx]
            y = rew + GAMMA * critic_t.forward(s2, actor_t.forward(s2))
            q = critic.forward(s, act)
            gc, _ = critic.backward(2 * (q - y) / len(idx))
            if cfg["l2"] > 0:
                for i in (0, 2, 4):  # 가중치에만 L2 (bias 제외)
                    gc[i] = gc[i] + cfg["l2"] * critic.p[i]
            opt_c.step(critic.p, gc)
            # actor: −mean Q(s, μ(s))를 줄이는 방향 (critic 파라미터는 고정)
            a_pi = actor.forward(s)
            critic.forward(s, a_pi)
            _, da = critic.backward(-np.ones((len(idx), 1)) / len(idx))
            opt_a.step(actor.p, actor.backward(da))
            soft_update(actor_t, actor, cfg["tau"])
            soft_update(critic_t, critic, cfg["tau"])
            if not np.isfinite(critic.p[4]).all() or np.abs(critic.p[4]).max() > 1e6:
                diverged = True
                break

        if (t + 1) % EVAL_EVERY == 0:
            curve.append([t + 1, round(evaluate(actor), 1)])
    out = {"config": name, "seed": seed, "curve": curve, "diverged": diverged, "seconds": round(time.time() - t0, 1)}
    if not diverged:
        out["final_return"] = curve[-1][1]
        out["q_bias"] = q_bias(actor, critic)
    return out


def main() -> None:
    jobs = [(n, s) for n in CONFIGS for s in SEEDS]
    t0 = time.time()
    with Pool(min(len(jobs), max(1, (os.cpu_count() or 4) - 2))) as pool:
        runs = pool.map(run, jobs)
    summary = {}
    for name in CONFIGS:
        rs = [r for r in runs if r["config"] == name]
        finals = [r["final_return"] for r in rs if not r["diverged"]]
        auc = [float(np.mean([c[1] for c in r["curve"]])) for r in rs if r["curve"]]
        summary[name] = {
            "final_mean": round(float(np.mean(finals)), 1) if finals else None,
            "final_per_seed": [r.get("final_return") for r in rs],
            "curve_mean_return": round(float(np.mean(auc)), 1) if auc else None,
            "first_step_above_-400": [next((c[0] for c in r["curve"] if c[1] > -400), None) for r in rs],
            "q_rel_bias": [r["q_bias"]["rel_bias"] for r in rs if "q_bias" in r],
            "diverged": sum(r["diverged"] for r in rs),
        }
    random_ret = evaluate(type("R", (), {"forward": staticmethod(lambda x: np.zeros((1, 1)))})())
    result = {"grad_check_max_rel_error": grad_check(), "settings": {"steps": STEPS, "eval_every": EVAL_EVERY, "warmup": WARMUP, "hidden": [H1, H2],
                           "base": BASE, "configs": CONFIGS, "seeds": SEEDS},
              "zero_torque_policy_return": round(random_ret, 1),
              "summary": summary, "runs": runs, "wall_seconds": round(time.time() - t0, 1)}
    path = ROOT / "analysis" / "ddpg_toy_results.json"
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print("zero-torque return", round(random_ret, 1), "wall", result["wall_seconds"], "s; wrote", path)


if __name__ == "__main__":
    main()
