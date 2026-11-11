"""추가 검증: 배치 정규화(BN) × 표적망 교차 실험을 LQR 과제(numpy, CPU)에서 돌린다.

실행: python3 analysis/extra_lqr_bn.py  → analysis/extra_lqr_bn_results.json, report/figures/lqr_bn_target.png

과제: 이산시간 x' = Ax + Bu (결정적, 개루프 안정: A의 고유값 0.95, 0.9), 비용 x'Qx + u'Ru, 할인 γ = 0.99, 에피소드 100 step.
관측은 o = S·x, S = diag(1, c)이다. c = 1(척도 같음)과 c = 20(속도 척도 20배)을 본다.
최적 정책은 할인 Riccati 반복으로 얻은 선형 피드백 u = −Kx(클리핑)이다.

구조는 ddpg_toy.py와 같다(64-64, ReLU, actor는 tanh×UMAX 출력, 행동은 두 번째 은닉층에 합류, Adam 1e-4/1e-3,
배치 64, 환경 step마다 업데이트 1번, critic L2 없음).
- BN 모드: none(없음), input(관측 입력만), all(입력과 모든 은닉층 선형 출력, 논문 부록식).
  학습 시에는 배치 통계, 행동 선택·평가에는 온라인 망의 이동 평균(momentum 0.05)을 쓴다.
  표적망은 배치 통계로 계산하고 이동 평균을 갖지 않는다.
- 표적망: τ = 0.001(논문) 또는 없음(y를 온라인 망으로 계산).
- 탐색 잡음은 가우시안 σ = 0.5(행동 단위), 30,000 step, 시드 8개.
측정: 고정 초기 상태 20개에서 결정적 정책의 할인 비용을 같은 100 step 기준으로 재고, 최적 K의 비용으로 나눈 비용비(1 = 최적).
BN 그래디언트는 중앙 차분과 파라미터별로 비교한다.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import copy
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STEPS, EVAL_EVERY, WARMUP, BATCH = 30_000, 3_000, 1_000, 64
H1 = H2 = 64
GAMMA, UMAX, HORIZON, SIGMA = 0.99, 5.0, 100, 0.5
TAU = {"target": 0.001, "none": None}
BN_MODES = ("none", "input", "all")
SCALES = (1.0, 20.0)
SEEDS = tuple(range(8))
EPS_BN, MOM = 1e-5, 0.05

A = np.array([[0.95, 0.1], [0.0, 0.9]])
B = np.array([[0.005], [0.1]])
Qc = np.diag([1.0, 0.1])
Rc = np.array([[0.1]])


# ── 과제 ────────────────────────────────────────────────────────────────────
def optimal_gain():
    P = Qc.copy()
    for _ in range(5000):
        K = np.linalg.solve(Rc + GAMMA * B.T @ P @ B, GAMMA * B.T @ P @ A)
        P = Qc + K.T @ Rc @ K + GAMMA * (A - B @ K).T @ P @ (A - B @ K)
    return K


K_OPT = optimal_gain()
EVAL_X0 = np.random.default_rng(123).uniform(-1, 1, (20, 2))


def step_dyn(x, u):
    """x: (N,2), u: (N,1). 반환: 다음 x, 비용 (N,)."""
    cost = np.einsum("ni,ij,nj->n", x, Qc, x) + np.einsum("ni,ij,nj->n", u, Rc, u)
    return x @ A.T + u @ B.T, cost


def rollout_cost(policy_x, x0):
    """policy_x: x (N,2) → u (N,1). 같은 100 step 기준 할인 비용 (N,)."""
    x, total, disc = x0.copy(), np.zeros(len(x0)), 1.0
    for _ in range(HORIZON):
        u = np.clip(policy_x(x), -UMAX, UMAX)
        x, c = step_dyn(x, u)
        total += disc * c
        disc *= GAMMA
    return total


COST_OPT = rollout_cost(lambda x: -(x @ K_OPT.T), EVAL_X0).mean()
COST_ZERO = rollout_cost(lambda x: np.zeros((len(x), 1)), EVAL_X0).mean()


# ── BN ──────────────────────────────────────────────────────────────────────
def bn_fwd(x, P, R, key, train, update):
    if not train:
        xh = (x - R[f"m{key}"]) / np.sqrt(R[f"v{key}"] + EPS_BN)
        return P[f"g{key}"] * xh + P[f"be{key}"], None
    mu, var = x.mean(0), x.var(0)
    if update:
        R[f"m{key}"] = (1 - MOM) * R[f"m{key}"] + MOM * mu
        R[f"v{key}"] = (1 - MOM) * R[f"v{key}"] + MOM * var
    inv = 1.0 / np.sqrt(var + EPS_BN)
    xh = (x - mu) * inv
    return P[f"g{key}"] * xh + P[f"be{key}"], (xh, inv)


def bn_bwd(dy, cache, P, key, grads):
    xh, inv = cache
    n = len(dy)
    grads[f"g{key}"] = (dy * xh).sum(0)
    grads[f"be{key}"] = dy.sum(0)
    dxh = dy * P[f"g{key}"]
    return inv / n * (n * dxh - dxh.sum(0) - xh * (dxh * xh).sum(0))


# ── 망 (수동 역전파) ─────────────────────────────────────────────────────────
def init_params(rng, in_dim, out_dim, mode, final=False, action_join=0):
    P = {}
    dims = [(in_dim, H1), (H1 + action_join, H2), (H2, out_dim)]
    for i, (fi, fo) in enumerate(dims, start=1):
        lim = 3e-3 if (final and i == 3) else 1 / np.sqrt(fi)
        P[f"W{i}"] = rng.uniform(-lim, lim, (fi, fo))
        P[f"b{i}"] = rng.uniform(-lim, lim, fo)
    for key, d, on in (("0", in_dim, mode in ("input", "all")), ("1", H1, mode == "all"), ("2", H2, mode == "all")):
        if on:
            P[f"g{key}"] = np.ones(d)
            P[f"be{key}"] = np.zeros(d)
    return P


def init_run(mode):
    R = {}
    for key, d, on in (("0", 2, mode in ("input", "all")), ("1", H1, mode == "all"), ("2", H2, mode == "all")):
        if on:
            R[f"m{key}"], R[f"v{key}"] = np.zeros(d), np.ones(d)
    return R


class Actor:
    def __init__(self, rng, mode):
        self.in_bn, self.hid_bn = mode in ("input", "all"), mode == "all"
        self.P = init_params(rng, 2, 1, mode, final=True)
        self.R = init_run(mode)

    def forward(self, x, train=False, update=False):
        P, R, c = self.P, self.R, {}
        h = x
        if self.in_bn:
            h, c["bn0"] = bn_fwd(h, P, R, "0", train, update)
        c["in1"] = h
        z = h @ P["W1"] + P["b1"]
        if self.hid_bn:
            z, c["bn1"] = bn_fwd(z, P, R, "1", train, update)
        c["m1"] = z > 0
        h1 = np.maximum(z, 0)
        c["in2"] = h1
        z = h1 @ P["W2"] + P["b2"]
        if self.hid_bn:
            z, c["bn2"] = bn_fwd(z, P, R, "2", train, update)
        c["m2"] = z > 0
        h2 = np.maximum(z, 0)
        c["in3"] = h2
        t = np.tanh(h2 @ P["W3"] + P["b3"])
        c["t"] = t
        self.cache = c
        return t * UMAX

    def backward(self, dout):
        P, c, g = self.P, self.cache, {}
        dz = dout * UMAX * (1 - c["t"] ** 2)
        g["W3"], g["b3"] = c["in3"].T @ dz, dz.sum(0)
        dz = (dz @ P["W3"].T) * c["m2"]
        if self.hid_bn:
            dz = bn_bwd(dz, c["bn2"], P, "2", g)
        g["W2"], g["b2"] = c["in2"].T @ dz, dz.sum(0)
        dz = (dz @ P["W2"].T) * c["m1"]
        if self.hid_bn:
            dz = bn_bwd(dz, c["bn1"], P, "1", g)
        g["W1"], g["b1"] = c["in1"].T @ dz, dz.sum(0)
        if self.in_bn:
            bn_bwd(dz @ P["W1"].T, c["bn0"], P, "0", g)
        return g


class Critic:
    def __init__(self, rng, mode):
        self.in_bn, self.hid_bn = mode in ("input", "all"), mode == "all"
        self.P = init_params(rng, 2, 1, mode, final=True, action_join=1)
        self.R = init_run(mode)

    def forward(self, x, a, train=False, update=False):
        P, R, c = self.P, self.R, {}
        h = x
        if self.in_bn:
            h, c["bn0"] = bn_fwd(h, P, R, "0", train, update)
        c["in1"] = h
        z = h @ P["W1"] + P["b1"]
        if self.hid_bn:
            z, c["bn1"] = bn_fwd(z, P, R, "1", train, update)
        c["m1"] = z > 0
        h1 = np.maximum(z, 0)
        cat = np.concatenate([h1, a], axis=1)
        c["in2"] = cat
        z = cat @ P["W2"] + P["b2"]
        if self.hid_bn:
            z, c["bn2"] = bn_fwd(z, P, R, "2", train, update)
        c["m2"] = z > 0
        h2 = np.maximum(z, 0)
        c["in3"] = h2
        self.cache = c
        return h2 @ P["W3"] + P["b3"]

    def backward(self, dq):
        P, c, g = self.P, self.cache, {}
        g["W3"], g["b3"] = c["in3"].T @ dq, dq.sum(0)
        dz = (dq @ P["W3"].T) * c["m2"]
        if self.hid_bn:
            dz = bn_bwd(dz, c["bn2"], P, "2", g)
        g["W2"], g["b2"] = c["in2"].T @ dz, dz.sum(0)
        dc = dz @ P["W2"].T
        dz = dc[:, :H1] * c["m1"]
        da = dc[:, H1:]
        if self.hid_bn:
            dz = bn_bwd(dz, c["bn1"], P, "1", g)
        g["W1"], g["b1"] = c["in1"].T @ dz, dz.sum(0)
        if self.in_bn:
            bn_bwd(dz @ P["W1"].T, c["bn0"], P, "0", g)
        return g, da


class Adam:
    def __init__(self, P, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in P.items()}
        self.v = {k: np.zeros_like(v) for k, v in P.items()}

    def step(self, P, G):
        self.t += 1
        for k in G:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * G[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * G[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            P[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def soft_update(tgt, src, tau):
    for k in src:
        tgt[k] = (1 - tau) * tgt[k] + tau * src[k]


# ── 그래디언트 점검 (파라미터별 중앙 차분 비교) ───────────────────────────────
def grad_check(mode, eps=1e-6):
    rng = np.random.default_rng(7)
    critic = Critic(rng, mode)
    for k in critic.P:
        critic.P[k] += rng.normal(0, 0.2 if k.startswith("g") else 0.3, critic.P[k].shape)
    x, a, y = rng.normal(size=(16, 2)), rng.uniform(-1, 1, (16, 1)), rng.normal(size=(16, 1))

    def loss():
        return float(((critic.forward(x, a, train=True) - y) ** 2).mean())

    q = critic.forward(x, a, train=True)
    gc, da = critic.backward(2 * (q - y) / len(x))
    errs = {}
    for k in critic.P:
        worst = 0.0
        for _ in range(3):
            i = tuple(rng.integers(0, s) for s in critic.P[k].shape)
            critic.P[k][i] += eps; lp = loss(); critic.P[k][i] -= 2 * eps; lm = loss(); critic.P[k][i] += eps
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - gc[k][i]) / max(1e-4, abs(num) + abs(gc[k][i])))
        errs[k] = worst
    a[3, 0] += eps; lp = loss(); a[3, 0] -= 2 * eps; lm = loss(); a[3, 0] += eps
    num = (lp - lm) / (2 * eps)
    errs["action_input"] = abs(num - da[3, 0]) / max(1e-4, abs(num) + abs(da[3, 0]))
    return {"max_rel_error": float(max(errs.values())), "per_key": {k: float(v) for k, v in errs.items()}}


# ── 학습 ────────────────────────────────────────────────────────────────────
def run(job):
    mode, tgt_name, scale, seed = job
    tau = TAU[tgt_name]
    s_obs = np.array([1.0, scale])
    rng = np.random.default_rng(1000 + seed)
    actor, critic = Actor(rng, mode), Critic(rng, mode)
    actor_t = {k: v.copy() for k, v in actor.P.items()}
    critic_t = {k: v.copy() for k, v in critic.P.items()}
    opt_a, opt_c = Adam(actor.P, 1e-4), Adam(critic.P, 1e-3)

    def evaluate():
        cost = rollout_cost(lambda x: actor.forward(s_obs * x, train=False), EVAL_X0).mean()
        return float(cost / COST_OPT)

    S = np.zeros((STEPS, 2)); U = np.zeros((STEPS, 1)); Rw = np.zeros((STEPS, 1)); S2 = np.zeros((STEPS, 2))
    x = rng.uniform(-1, 1, (1, 2)); ep_t = 0
    curve, diverged = [], False
    t0 = time.time()
    for t in range(STEPS):
        o = s_obs * x
        u = np.clip(actor.forward(o) + SIGMA * rng.standard_normal((1, 1)), -UMAX, UMAX)
        x2, c = step_dyn(x, u)
        S[t], U[t], Rw[t], S2[t] = o[0], u[0], -c[0], (s_obs * x2)[0]
        x, ep_t = x2, ep_t + 1
        if ep_t == HORIZON:
            x, ep_t = rng.uniform(-1, 1, (1, 2)), 0

        if t >= WARMUP:
            idx = rng.integers(0, t + 1, BATCH)
            s, u_b, r, s2 = S[idx], U[idx], Rw[idx], S2[idx]
            if tau is None:  # 표적망 없음: 온라인 망으로 목표 계산
                a2 = actor.forward(s2, train=True)
                q2 = critic.forward(s2, a2, train=True)
            else:
                ta = copy.copy(actor); ta.P, ta.R = actor_t, {}
                tc = copy.copy(critic); tc.P, tc.R = critic_t, {}
                a2 = ta.forward(s2, train=True)
                q2 = tc.forward(s2, a2, train=True)
            y = r + GAMMA * q2
            q = critic.forward(s, u_b, train=True, update=True)
            gc, _ = critic.backward(2 * (q - y) / BATCH)
            opt_c.step(critic.P, gc)
            # actor: −mean Q(s, μ(s)); 이 단계에서는 critic 파라미터를 갱신하지 않는다
            a_pi = actor.forward(s, train=True, update=True)
            critic.forward(s, a_pi, train=True)
            _, da = critic.backward(-np.ones((BATCH, 1)) / BATCH)
            opt_a.step(actor.P, actor.backward(da))
            if tau is not None:
                soft_update(actor_t, actor.P, tau)
                soft_update(critic_t, critic.P, tau)
            if not np.isfinite(critic.P["W3"]).all() or np.abs(critic.P["W3"]).max() > 1e6:
                diverged = True
                break

        if (t + 1) % EVAL_EVERY == 0:
            curve.append([t + 1, round(evaluate(), 4)])
    out = {"bn": mode, "target": tgt_name, "scale": scale, "seed": seed, "curve": curve,
           "diverged": diverged, "seconds": round(time.time() - t0, 1)}
    if curve and not diverged:
        out["final_ratio"] = curve[-1][1]
    return out


def main():
    t0 = time.time()
    gc_err = {m: grad_check(m) for m in ("input", "all")}
    jobs = [(m_, tg, sc, s) for sc in SCALES for m_ in BN_MODES for tg in TAU for s in SEEDS]
    with Pool(min(len(jobs), max(1, (os.cpu_count() or 4) - 2))) as pool:
        runs = pool.map(run, jobs)

    from scipy import stats

    def finals(mode, tg, sc):
        return [r["final_ratio"] for r in runs if r["bn"] == mode and r["target"] == tg and r["scale"] == sc
                and "final_ratio" in r]

    summary = {}
    for sc in SCALES:
        for mode in BN_MODES:
            for tg in TAU:
                fin = finals(mode, tg, sc)
                rs = [r for r in runs if r["bn"] == mode and r["target"] == tg and r["scale"] == sc]
                summary[f"scale={sc:g}, bn={mode}, target={tg}"] = {
                    "final_ratio_median": round(float(np.median(fin)), 3) if fin else None,
                    "final_ratio_mean": round(float(np.mean(fin)), 3) if fin else None,
                    "final_ratio_per_seed": fin,
                    "seeds_ratio_below_1.5": sum(v < 1.5 for v in fin),
                    "diverged": sum(r["diverged"] for r in rs),
                }
    pairs = {}
    for sc in SCALES:
        pairs[f"scale={sc:g}: BN(all) vs none, target"] = (finals("all", "target", sc), finals("none", "target", sc))
        pairs[f"scale={sc:g}: BN(all) vs none, no target"] = (finals("all", "none", sc), finals("none", "none", sc))
        pairs[f"scale={sc:g}: BN(input) vs none, target"] = (finals("input", "target", sc), finals("none", "target", sc))
        pairs[f"scale={sc:g}: target vs none, BN none"] = (finals("none", "target", sc), finals("none", "none", sc))
        pairs[f"scale={sc:g}: target vs none, BN(all)"] = (finals("all", "target", sc), finals("all", "none", sc))
    pvals = {k: round(float(stats.mannwhitneyu(a, b).pvalue), 4) for k, (a, b) in pairs.items() if a and b}

    result = {
        "settings": {"A": A.tolist(), "B": B.tolist(), "Q": Qc.tolist(), "R": Rc.tolist(), "S_obs": "diag(1, scale)",
                     "scales": list(SCALES), "gamma": GAMMA, "umax": UMAX, "horizon": HORIZON, "explore_sigma": SIGMA,
                     "steps": STEPS, "eval_every": EVAL_EVERY, "warmup": WARMUP, "batch": BATCH, "hidden": [H1, H2],
                     "bn_momentum": MOM, "tau": TAU, "bn_modes": list(BN_MODES), "seeds": list(SEEDS),
                     "eval_states": len(EVAL_X0)},
        "K_opt": K_OPT.round(4).tolist(),
        "cost_opt_eval": round(float(COST_OPT), 3),
        "cost_zero_policy_ratio": round(float(COST_ZERO / COST_OPT), 3),
        "grad_check": gc_err,
        "summary": summary,
        "mannwhitney_p": pvals,
        "runs": runs,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (ROOT / "analysis" / "extra_lqr_bn_results.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=1, ensure_ascii=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=160, sharey=True)
    styles = {("none", "target"): ("#9a9a9f", "BN 없음, τ = 0.001"), ("none", "none"): ("#9a9a9f", "BN 없음, 표적망 없음"),
              ("input", "target"): ("#2a78d6", "BN(입력), τ = 0.001"), ("all", "target"): ("#1baf7a", "BN(전체), τ = 0.001"),
              ("all", "none"): ("#eb6834", "BN(전체), 표적망 없음")}
    for ax, sc in zip(axes, SCALES):
        for (mode, tg), (col, lab) in styles.items():
            rs = [r for r in runs if r["bn"] == mode and r["target"] == tg and r["scale"] == sc and r["curve"]]
            steps = np.array([c[0] for c in rs[0]["curve"]])
            M = np.array([[c[1] for c in r["curve"]] for r in rs])
            ax.plot(steps, np.median(M, 0), color=col, lw=1.6, ls="--" if tg == "none" else "-", label=lab)
        ax.axhline(1.0, color="#0b0b0b", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_xlabel("환경 step")
        ax.set_title(f"관측 척도 1 : {sc:g}", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("평가 비용 / 최적 비용 (로그, 1 = 최적)")
    axes[1].legend(fontsize=6.8, frameon=False, loc="upper right")
    fig.suptitle("LQR toy: BN × 표적망 (중앙값, 시드 8개)", fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "report" / "figures" / "lqr_bn_target.png")
    print("wrote", ROOT / "analysis" / "extra_lqr_bn_results.json", "wall", result["wall_seconds"], "s")


if __name__ == "__main__":
    main()
