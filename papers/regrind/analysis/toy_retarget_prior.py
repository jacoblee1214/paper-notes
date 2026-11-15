"""토이 B: 리타게팅 기준(prior)이 잔차 RL의 탐색을 줄이는가, 그리고 리타게팅 오차에 얼마나 민감한가.

CPU 단독 실행 (numpy만 사용, 단일 프로세스, 수 분 이내). 실행: python3 analysis/toy_retarget_prior.py
결과: analysis/toy_retarget_prior_results.json

설정 (논문과 같은 구조를 가장 작게 옮긴 것, 실제 손이나 접촉 동역학은 없음)
- 로봇: 관절 q ∈ R^6, 키포인트 사상 f(q) = W2 tanh(W1 q + b1) ∈ R^6 (비볼록이라 국소 최소가 있음).
- 사람 시연: 관절 경로 q*_t로부터 K*_t = f(q*_t), T = 200 단계. 로봇 기준이 실제로 도달 가능함을 가정.
- 리타게팅: 키포인트 목표 K_ref_t = K*_t + ε g_t (g_t는 부드러운 고정 오차장)에 대해 Gauss-Newton IK를 시간 순서로 풀어
  기준 관절 q̄_t를 얻는다. 리타게팅 오차는 ε로 조절한다. 기준이 만든 키포인트 f(q̄_t)와 실제 사람 K*_t의 차이가 ε에 따라 커진다.
- S (처음부터 RL): 정책 입력 [q_t, K*_t − f(q_t), t/T], 보상 exp(−||f(q_t) − K*_t||²/σ²). 기준을 쓰지 않는다.
- R (리타게팅 기준 + 잔차 RL): 위치 목표 = q̄_{t+1} + 0.1·π(s_t), 입력은 기준 키포인트 f(q̄_t) 쪽 오차,
  보상은 exp(−||f(q_t) − f(q̄_t)||²/σ²). 사람 시연 K*_t가 아니라 기준을 따라간다(논문 Sec. 3.3 구조).
- 평가 성공: 마지막 절반 구간의 평균 ||f(q_t) − K*_t|| < 0.1 (실제 과제 기준은 사람 시연 K*_t).
- 학습: 진화 전략(ES), 대칭 샘플링, 순위 정규화. 정책은 1층 tanh MLP(16 은닉).
"""
import json
import os
import time
from pathlib import Path

# 단일 프로세스, 단일 스레드 (BLAS 스레드 금지)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_retarget_prior_results.json"
D, K, H, T = 6, 6, 16, 200
U_MAX = 0.05
RES_SCALE = 0.1
KAPPA = 0.5
SIGMA = 0.1
SUCCESS_TAU = 0.1
POP, GENS, LR, SIG_ES = 32, 150, 0.05, 0.05
S_LONG_GENS = 600  # 처음부터 RL이 언젠가 성공하는지 보는 한 시드 긴 실행
EVAL_EPISODES = 64
SEEDS = list(range(10))
EPS_LIST = [0.0, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]


class World:
    def __init__(self, seed):
        rng = np.random.default_rng(1000 + seed)
        self.W1 = rng.normal(size=(H, D)) * 1.5 / np.sqrt(D)
        self.b1 = rng.normal(size=H) * 0.5
        self.W2 = rng.normal(size=(K, H)) / np.sqrt(H)
        # 사람 시연 경로: q0에서 목표 q_goal까지 smoothstep 보간 + 작은 굴곡
        self.q0 = rng.normal(size=D) * 0.3
        self.q_goal = self.q0 + rng.normal(size=D) * 1.2
        s = np.linspace(0, 1, T)
        ss = 3 * s**2 - 2 * s**3
        bend = np.outer(np.sin(np.pi * s), rng.normal(size=D)) * 0.3
        self.q_true = self.q0 + np.outer(ss, self.q_goal - self.q0) + bend
        self.K_star = self.f(self.q_true)
        # 리타게팅 오차장: 부드러운 임의 궤적, 크기는 ε로 곱함
        self.g = np.cumsum(rng.normal(size=(T, K)), axis=0)
        self.g = self.g / (np.abs(self.g).max() + 1e-9)

    def f(self, q):
        return np.tanh(q @ self.W1.T + self.b1) @ self.W2.T

    def jac(self, q):
        a = 1 - np.tanh(q @ self.W1.T + self.b1) ** 2  # (n, H)
        return np.einsum("nh,kh,hd->nkd", a, self.W2, self.W1)

    def retarget(self, eps):
        """시간 순서 Levenberg-Marquardt IK. 목표는 K*_t + ε g_t, 이전 프레임 정규화와 스텝 제한을 둔다."""
        q_prev = self.q_true[0].copy()
        out = np.zeros((T, D))
        mu = 1e-2
        for t in range(T):
            tgt = self.K_star[t] + eps * self.g[t]
            q = q_prev.copy()
            for _ in range(30):
                r = self.f(q[None])[0] - tgt
                J = self.jac(q[None])[0]
                A = J.T @ J + mu * np.eye(D)
                b = J.T @ r + mu * (q - q_prev)
                step = np.linalg.solve(A, b)
                n = np.linalg.norm(step)
                if n > 0.2:
                    step = step * (0.2 / n)
                q = q - step
            out[t] = q
            q_prev = q
        return out


def init_mlp(rng, in_dim, out_dim):
    return np.concatenate([(rng.normal(size=(in_dim, H)) / np.sqrt(in_dim)).ravel(),
                           np.zeros(H),
                           (rng.normal(size=(H, out_dim)) / np.sqrt(H) * 0.1).ravel(),
                           np.zeros(out_dim)])


def mlp_forward(theta, x, in_dim, out_dim):
    """theta: (P,) 또는 (B,P), x: (B, in_dim). 출력 tanh 범위 [-1,1]."""
    n1, n2, n3 = in_dim * H, H, H * out_dim
    if theta.ndim == 1:
        theta = theta[None]
    W1 = theta[:, :n1].reshape(-1, in_dim, H)
    b1 = theta[:, n1:n1 + n2]
    W2 = theta[:, n1 + n2:n1 + n2 + n3].reshape(-1, H, out_dim)
    b2 = theta[:, n1 + n2 + n3:]
    h = np.tanh(np.einsum("bi,bih->bh", x, W1) + b1)
    return np.tanh(np.einsum("bh,bho->bo", h, W2) + b2)


def rollout(w, th, mode, ref_q=None, ref_K=None, q_start=None):
    """배치 롤아웃. th: (B, P). 반환: 평균 보상(B,), 마지막 절반 구간 평균 추적 오차(B,).

    제어는 위치 목표 a_t를 1차 지연으로 따라간다: q_{t+1} = q_t + κ(a_t − q_t).
    S: a_t = q_t + U_MAX·π(s_t)          (기준 없음, 목표는 사람 키포인트 K*_t)
    R: a_t = q̄_{t+1} + RES_SCALE·π(s_t)  (기준 관절 + 잔차, 목표는 기준 키포인트)
    """
    B = th.shape[0]
    in_dim = D + K + 1
    q = np.repeat(w.q0[None], B, axis=0) if q_start is None else q_start.copy()
    total_r = np.zeros(B)
    errs = np.zeros((B, T))
    for t in range(T):
        Ktrue = w.K_star[t][None]
        ft = w.f(q)
        if mode == "S":
            x = np.concatenate([q, Ktrue - ft, np.full((B, 1), t / T)], axis=1)
            a = q + U_MAX * mlp_forward(th, x, in_dim, D)
            target = Ktrue
        else:
            tgt = ref_K[t][None]
            x = np.concatenate([q, tgt - ft, np.full((B, 1), t / T)], axis=1)
            base = ref_q[t + 1] if t + 1 < T else ref_q[-1]
            a = base[None] + RES_SCALE * mlp_forward(th, x, in_dim, D)
            target = tgt
        total_r += np.exp(-np.sum((ft - target) ** 2, axis=1) / SIGMA**2) / T
        errs[:, t] = np.linalg.norm(ft - Ktrue, axis=1)
        q = q + KAPPA * (a - q)
    return total_r, errs[:, T // 2:].mean(axis=1)


def es_train(w, mode, seed, ref_q=None, ref_K=None, log_every=25, gens=None):
    rng = np.random.default_rng(seed)
    in_dim = D + K + 1
    theta = init_mlp(rng, in_dim, D)
    P = theta.size
    steps_per_gen = POP * T
    curve = []
    for g in range(gens or GENS):
        eps_noise = rng.normal(size=(POP // 2, P))
        noise = np.concatenate([eps_noise, -eps_noise])
        cand = theta[None] + SIG_ES * noise
        R, _ = rollout(w, cand, mode, ref_q, ref_K)
        ranks = np.argsort(np.argsort(R)).astype(float)
        adv = ranks / (POP - 1) - 0.5
        grad = noise.T @ adv / (POP * SIG_ES)
        theta = theta + LR * grad
        if (g + 1) % log_every == 0 or g == (gens or GENS) - 1:
            succ, tail = evaluate(w, theta, mode, ref_q, ref_K, seed=10_000 + seed)
            curve.append({"gen": g + 1, "env_steps": (g + 1) * steps_per_gen, "success": succ, "mean_tail_err": tail})
    final, final_tail = evaluate(w, theta, mode, ref_q, ref_K, seed=20_000 + seed)
    return theta, (final, final_tail), curve


def evaluate(w, theta, mode, ref_q, ref_K, seed):
    """결정적 정책을 EVAL_EPISODES번 굴린다. 시작 관절에 작은 섭동(0.02 rad)을 준다."""
    rng = np.random.default_rng(seed)
    th = np.repeat(theta[None], EVAL_EPISODES, axis=0)
    q_start = np.repeat(w.q0[None], EVAL_EPISODES, axis=0) + 0.02 * rng.normal(size=(EVAL_EPISODES, D))
    _, tail = rollout(w, th, mode, ref_q, ref_K, q_start=q_start)
    return float(np.mean(tail < SUCCESS_TAU)), float(np.mean(tail))


def agg(vals):
    v = np.array(vals, dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "per_seed": [float(x) for x in v]}


def main():
    t0 = time.time()
    cfg = {"D": D, "K": K, "H": H, "T": T, "U_MAX": U_MAX, "RES_SCALE": RES_SCALE, "KAPPA": KAPPA,
           "SIGMA": SIGMA, "SUCCESS_TAU": SUCCESS_TAU, "POP": POP, "GENS": GENS, "LR": LR, "SIG_ES": SIG_ES,
           "EVAL_EPISODES": EVAL_EPISODES, "SEEDS": SEEDS, "EPS_LIST": EPS_LIST, "S_LONG_GENS": S_LONG_GENS}
    results = {"config": cfg, "scratch": [], "residual": {}}
    for seed in SEEDS:
        w = World(seed)
        _, (succ, tail), curve = es_train(w, "S", seed)
        results["scratch"].append({"seed": seed, "final_success": succ, "final_mean_tail_err": tail, "curve": curve})
    for eps in EPS_LIST:
        rows = []
        for seed in SEEDS:
            w = World(seed)
            ref_q = w.retarget(eps)
            ref_K = w.f(ref_q)
            ref_err = float(np.mean(np.linalg.norm(ref_K - w.K_star, axis=1)))
            _, (succ, tail), curve = es_train(w, "R", seed, ref_q, ref_K)
            rows.append({"seed": seed, "ref_keypoint_err_mean": ref_err, "final_success": succ,
                         "final_mean_tail_err": tail, "curve": curve})
        results["residual"][str(eps)] = rows
    w = World(0)
    _, (succ_long, tail_long), curve_long = es_train(w, "S", 0, gens=S_LONG_GENS, log_every=50)
    results["scratch_long_seed0"] = {"gens": S_LONG_GENS, "final_success": succ_long,
                                     "final_mean_tail_err": tail_long, "curve": curve_long}
    results["summary"] = {
        "scratch_final_success": agg([r["final_success"] for r in results["scratch"]]),
        "scratch_final_tail_err": agg([r["final_mean_tail_err"] for r in results["scratch"]]),
        "residual_final_success": {e: agg([r["final_success"] for r in rows]) for e, rows in results["residual"].items()},
        "residual_final_tail_err": {e: agg([r["final_mean_tail_err"] for r in rows]) for e, rows in results["residual"].items()},
        "residual_ref_keypoint_err": {e: agg([r["ref_keypoint_err_mean"] for r in rows]) for e, rows in results["residual"].items()},
    }
    results["runtime_s"] = time.time() - t0
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT, "runtime", round(results["runtime_s"], 1), "s")
    sm = results["summary"]
    print("scratch success", sm["scratch_final_success"]["mean"], "tail", round(sm["scratch_final_tail_err"]["mean"], 3))
    for e in results["residual"]:
        print("residual eps", e, "success", sm["residual_final_success"][e]["mean"],
              "tail", round(sm["residual_final_tail_err"][e]["mean"], 3),
              "ref err", round(sm["residual_ref_keypoint_err"][e]["mean"], 3))
    print("scratch long seed0", [(c["env_steps"], c["success"], round(c["mean_tail_err"], 3)) for c in curve_long][-3:])


if __name__ == "__main__":
    main()
