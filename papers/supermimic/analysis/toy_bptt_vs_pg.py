"""토이 S: 학습된 미분 가능 모델을 통한 BPTT(경로 미분)와 모델 없는 정책 기울기(REINFORCE)의
분산·편향·정책 개선 비용을 비교한다. 논문 §3.3의 "BPTT의 기울기는 분산이 훨씬 작다"와
§3.2의 "world model 오차가 있어도 같은 목표를 학습한다"를 시험하는 장치다. (numpy만 사용, 단일 프로세스)

설정 (직접 작성):
  진짜 시스템(결정론적): p' = p + dt v,  v' = v + dt (u − κ v − c sin p),  dt = 0.02, κ = 0.5
    c = 0 이면 선형, c > 0 이면 비선형(중력과 비슷한 sin 항). c는 진짜 시스템과 모델의 차이를 조절한다.
    진짜 시스템 자체가 c에 따라 달라지므로, 분산 표는 c = 0(선형 진짜 시스템) 기준으로 읽는다.
  학습된 모델: 같은 자료(무작위 상태·행동 50,000쌍)에 선형 회귀 v' − v = dt (a0 + a1 p + a2 v + a3 u).
    c > 0이면 sin 항을 놓치므로 모델 오차가 생긴다. c = 0이면 정확하다(편향 0이 나와야 함: 검사 항목).
  정책: μ_t = k (p_t − r_t) + k_v v_t + b,  u_t = μ_t + σ ε_t,  σ = 0.3,  r_t = 0.5 sin(π t dt)
  보상: r_t = −(p_t − r_t)² − 0.1 v_t² − 0.01 u_t²,  목적 J(θ) = E[Σ_{t<H} r_t]  (γ = 1, 유한 지평)
  초기 상태: p_0 ~ U(−1.5, 1.5), v_0 ~ N(0, 0.5²)

추정량 (같은 (x_0, ε) 표본에서 비교, 공통 난수):
  g_true : 진짜 시스템을 통한 경로 미분  ∇_θ Σ r_t  (재매개변수화된 잡음이 고정되어 있으므로 기대값이 ∇J와 같다)
  g_model: 학습된 모델을 통한 경로 미분 (BPTT와 같은 양; 모델 기울기)
  g_pg   : 진짜 시스템의 REINFORCE, reward-to-go와 표본 평균 기준선   ∑_t (G_t − b_t) ε_t φ_t / σ
  경로 미분은 중심 유한차분(h = 1e-6)으로 계산한다. 결정론적 미분 가능 모델에서는 BPTT와 같은 값이다.

측정: H ∈ {4, 8, 16, 32, 64}, c ∈ {0, 0.5, 1, 2}
  - E[g_true] 대비 g_pg의 편향(표준오차 안에 있어야 함), 분산(추정량당 trace)
  - E[g_model] − E[g_true]의 편향, 방향 코사인, 상대 편향
  - 모델 롤아웃 오차 |p̂_t − p_t|, |v̂_t − v_t| 의 누적 (c = 1, H = 64)
  - 정책 개선 실험 (H = 32): 진짜 J를 최대화하는 θ*와 모델 J를 최대화하는 θ̂*를 Adam으로 구하고,
    진짜 J에서 두 θ의 차이(후회)를 잰다. 두 최적화는 같은 고정 표본의 평균으로 이뤄지므로 후회의 부호는 표본 잡음 안에서 읽는다.
    모델 기반은 모델 자료 50,000쌍만 쓰고, PG는 진짜 롤아웃 300 × 20,000을 쓴다(학습률을 튜닝하지 않은 기준선).

결과: analysis/toy_bptt_vs_pg_results.json   실행: python3 analysis/toy_bptt_vs_pg.py  (수 분 이내, CPU)
"""
import json
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_bptt_vs_pg_results.json"
DT, KAP, SIG = 0.02, 0.5, 0.3
THETA0 = np.array([-1.0, -0.5, 0.2])  # 평가 지점 (k, k_v, b)
N_EVAL = 200_000
N_OPT = 20_000
H_LIST = [4, 8, 16, 32, 64]
C_LIST = [0.0, 0.5, 1.0, 2.0]
FD_H = 1e-6
RNG = np.random.default_rng(20261115)


def dyn_true(c):
    def f(p, v, u):
        return p + DT * v, v + DT * (u - KAP * v - c * np.sin(p))
    return f


def fit_model(c, n=50_000):
    """같은 시스템에서 무작위 (p, v, u)를 뽑아 선형 모델 v' − v = dt (a0 + a1 p + a2 v + a3 u)를 최소제곱으로 맞춘다."""
    p = RNG.uniform(-2, 2, n)
    v = RNG.uniform(-2, 2, n)
    u = RNG.uniform(-1, 1, n)
    y = (u - KAP * v - c * np.sin(p))  # 진짜 (v' − v)/dt (결정론)
    X = np.stack([np.ones(n), p, v, u], axis=1)
    a, *_ = np.linalg.lstsq(X, y, rcond=None)
    def f(p_, v_, u_):
        return p_ + DT * v_, v_ + DT * (a[0] + a[1] * p_ + a[2] * v_ + a[3] * u_)
    return f, a


def rollout(theta, p0, v0, eps, H, dyn, want_traj=False):
    """반환: 표본별 총 보상 (N,), 필요하면 보상 행렬 (N,H), φ 행렬 (N,H,3), 상태 궤적."""
    k, kv, b = theta
    p, v = p0.copy(), v0.copy()
    N = p0.shape[0]
    R = np.zeros((N, H)) if want_traj else None
    PHI = np.zeros((N, H, 3)) if want_traj else None
    PS, VS = (np.zeros((N, H)), np.zeros((N, H))) if want_traj else (None, None)
    total = np.zeros(N)
    for t in range(H):
        r_ref = 0.5 * np.sin(np.pi * t * DT)
        e = p - r_ref
        mu = k * e + kv * v + b
        u = mu + SIG * eps[:, t]
        rew = -(e ** 2) - 0.1 * v ** 2 - 0.01 * u ** 2
        total += rew
        if want_traj:
            R[:, t] = rew
            PHI[:, t, 0] = e
            PHI[:, t, 1] = v
            PHI[:, t, 2] = 1.0
            PS[:, t] = p
            VS[:, t] = v
        p, v = dyn(p, v, u)
    if want_traj:
        return total, R, PHI, PS, VS
    return total


def pathwise(theta, p0, v0, eps, H, dyn):
    """중심 차분으로 표본별 ∇_θ Σ r_t (3,) 를 계산. 반환 (N,3)."""
    g = np.zeros((p0.shape[0], 3))
    for j in range(3):
        d = np.zeros(3)
        d[j] = FD_H
        jp = rollout(theta + d, p0, v0, eps, H, dyn)
        jm = rollout(theta - d, p0, v0, eps, H, dyn)
        g[:, j] = (jp - jm) / (2 * FD_H)
    return g


def pg_grad(theta, p0, v0, eps, H, dyn):
    """REINFORCE with reward-to-go and per-time sample-mean baseline (진짜 시스템)."""
    _, R, PHI, _, _ = rollout(theta, p0, v0, eps, H, dyn, want_traj=True)
    G = np.cumsum(R[:, ::-1], axis=1)[:, ::-1]  # reward-to-go
    base = G.mean(axis=0)
    adv = G - base  # (N,H)
    score = (eps[:, :H] / SIG)[:, :, None] * PHI  # (N,H,3)
    return (adv[:, :, None] * score).sum(axis=1)  # (N,3)


def adam_ascent(grad_fn, theta, iters=300, lr=0.02):
    m = np.zeros(3)
    v = np.zeros(3)
    for i in range(1, iters + 1):
        g = grad_fn(theta)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        mh = m / (1 - 0.9 ** i)
        vh = v / (1 - 0.999 ** i)
        theta = theta + lr * mh / (np.sqrt(vh) + 1e-8)
    return theta


def sample(N, H):
    p0 = RNG.uniform(-1.5, 1.5, N)
    v0 = RNG.normal(0, 0.5, N)
    eps = RNG.standard_normal((N, H))
    return p0, v0, eps


def main():
    t0 = time.time()
    out = {"config": {"dt": DT, "kappa": KAP, "sigma": SIG, "theta0": THETA0.tolist(),
                      "N_eval": N_EVAL, "N_opt": N_OPT, "H_list": H_LIST, "c_list": C_LIST}}
    models = {c: fit_model(c) for c in C_LIST}
    out["fitted_model_coeffs"] = {str(c): models[c][1].tolist() for c in C_LIST}

    rows = []
    for H in H_LIST:
        p0, v0, eps = sample(N_EVAL, H)
        for c in C_LIST:
            dyn_t = dyn_true(c)
            dyn_m = models[c][0]
            g_t = pathwise(THETA0, p0, v0, eps, H, dyn_t)
            g_m = pathwise(THETA0, p0, v0, eps, H, dyn_m)
            g_p = pg_grad(THETA0, p0, v0, eps, H, dyn_t)
            mt, mm, mp = g_t.mean(0), g_m.mean(0), g_p.mean(0)
            se_t = g_t.std(0) / np.sqrt(N_EVAL)
            se_p = g_p.std(0) / np.sqrt(N_EVAL)
            diff = g_m - g_t
            se_diff = diff.std(0) / np.sqrt(N_EVAL)
            rows.append({
                "H": H, "c": c,
                "grad_true_mean": mt.tolist(),
                "grad_pg_mean": mp.tolist(),
                "pg_minus_true_z": float(np.max(np.abs(mp - mt) / np.sqrt(se_p ** 2 + se_t ** 2))),
                "var_trace_pathwise_true": float(g_t.var(0).sum()),
                "var_trace_pg": float(g_p.var(0).sum()),
                "var_ratio_pg_over_pathwise": float(g_p.var(0).sum() / g_t.var(0).sum()),
                "bias_model_minus_true": (mm - mt).tolist(),
                "bias_significant_z_max": float(np.max(np.abs((mm - mt)) / np.maximum(se_diff, 1e-300))),
                "rel_bias_norm": float(np.linalg.norm(mm - mt) / np.linalg.norm(mt)),
                "cos_model_true": float(mm @ mt / (np.linalg.norm(mm) * np.linalg.norm(mt))),
                "var_trace_pathwise_model": float(g_m.var(0).sum()),
            })
            print(f"H={H:3d} c={c:3.1f}  relbias={rows[-1]['rel_bias_norm']:.3f}  cos={rows[-1]['cos_model_true']:.3f}  "
                  f"var_pg/var_pw={rows[-1]['var_ratio_pg_over_pathwise']:.1f}  zPG={rows[-1]['pg_minus_true_z']:.2f}  "
                  f"({time.time()-t0:.0f}s)")
    out["gradient_rows"] = rows

    # 모델 오차의 누적 (c = 1, H = 64): 같은 정책·같은 초기 상태에서 진짜와 모델 롤아웃의 차이
    H = 64
    p0, v0, eps = sample(5000, H)
    _, _, _, PS_t, VS_t = rollout(THETA0, p0, v0, eps, H, dyn_true(1.0), want_traj=True)
    _, _, _, PS_m, VS_m = rollout(THETA0, p0, v0, eps, H, models[1.0][0], want_traj=True)
    out["compounding_c1_H64"] = {
        "mean_abs_err_p": np.abs(PS_m - PS_t).mean(0)[[0, 3, 7, 15, 31, 63]].tolist(),
        "mean_abs_err_v": np.abs(VS_m - VS_t).mean(0)[[0, 3, 7, 15, 31, 63]].tolist(),
        "steps_index": [1, 4, 8, 16, 32, 64],
    }

    # 정책 개선 실험 (H = 32): 진짜 J 최대화 vs 모델 J 최대화 vs PG (같은 진짜 롤아웃 예산)
    H = 32
    p0, v0, eps = sample(N_OPT, H)
    opt_rows = []
    p_test, v_test, eps_test = sample(200_000, H)
    for c in C_LIST:
        dyn_t, dyn_m = dyn_true(c), models[c][0]
        th_true = adam_ascent(lambda th: pathwise(th, p0, v0, eps, H, dyn_t).mean(0), THETA0.copy())
        th_model = adam_ascent(lambda th: pathwise(th, p0, v0, eps, H, dyn_m).mean(0), THETA0.copy())
        J_true_at_true = rollout(th_true, p_test, v_test, eps_test, H, dyn_t).mean()
        J_true_at_model = rollout(th_model, p_test, v_test, eps_test, H, dyn_t).mean()
        opt_rows.append({"c": c, "theta_true_opt": th_true.tolist(), "theta_model_opt": th_model.tolist(),
                         "J_true_at_true_opt": float(J_true_at_true), "J_true_at_model_opt": float(J_true_at_model),
                         "regret_from_model_error": float(J_true_at_true - J_true_at_model)})
        print(f"opt c={c}: regret={opt_rows[-1]['regret_from_model_error']:.4f} ({time.time()-t0:.0f}s)")
    out["policy_improvement_H32"] = opt_rows

    # 모델 없는 PG: 진짜 롤아웃 300 iter × 20,000 (= 6,000,000 episode) 을 쓴다 (c 무관, 진짜 시스템 c=1 기준)
    th_pg = THETA0.copy()
    m = np.zeros(3)
    v2 = np.zeros(3)
    for i in range(1, 301):
        p0, v0, eps = sample(N_OPT, H)
        g = pg_grad(th_pg, p0, v0, eps, H, dyn_true(1.0)).mean(0)
        m = 0.9 * m + 0.1 * g
        v2 = 0.999 * v2 + 0.001 * g * g
        th_pg = th_pg + 0.02 * (m / (1 - 0.9 ** i)) / (np.sqrt(v2 / (1 - 0.999 ** i)) + 1e-8)
    out["pg_baseline_c1_H32"] = {
        "theta_pg": th_pg.tolist(),
        "true_rollouts_used": 300 * N_OPT,
        "J_true_at_pg": float(rollout(th_pg, p_test, v_test, eps_test, H, dyn_true(1.0)).mean()),
        "J_true_at_true_opt_c1": opt_rows[C_LIST.index(1.0)]["J_true_at_true_opt"],
    }
    out["pg_baseline_c1_H32"]["regret_pg"] = out["pg_baseline_c1_H32"]["J_true_at_true_opt_c1"] - out["pg_baseline_c1_H32"]["J_true_at_pg"]
    out["model_fit_transitions"] = 50_000
    out["runtime_s"] = time.time() - t0
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT, f"{time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
