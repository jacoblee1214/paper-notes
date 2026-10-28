"""Eq.1 (guided flow-matching tracking)의 성질을 CPU toy로 확인한다.

1) 항등식: 마지막 Euler 스텝(t: 1-Δ → 1) 뒤 blending 결과는
   x_1 = (1-α)·x̂1 + α·x_ref  (x̂1 = x_{1-Δ} + Δ·v, 네트워크의 x1-예측)
   → 임의의 velocity field에 대해 기계 정밀도로 성립하는지 확인.
2) 네트워크 예측이 현재 상태 x_t에 일부 의존하면(γ>0), 이전 프레임 pose에 걸리는
   실효 가중치 w_eff가 α보다 커진다 → α_p는 '하한'이고 실제 평활화는 더 강하다.
3) 프레임 간 관점: pose_k ≈ (1-w)·obs_k + w·pose_{k-1} = EMA(지수이동평균).
   고정 α vs 적응형 α(|Δθ|) 의 오차/지터/지연을 1D 회전 궤적으로 비교.

실행: python3 analysis/guidance_toy.py  → analysis/guidance_toy_results.json, report/figures/toy_*.png
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)
T_STEPS = 25  # 논문/코드 euler_steps=25


def guided_sample(x1_pred_fn, x_ref, alpha, eps, steps=T_STEPS):
    """track_object.py batched_guided_sample_sparse_structure 의 pose 블록 루프를 그대로 옮김."""
    t_seq = np.linspace(0, 1, steps + 1)
    x = eps.copy()
    last = None
    for t0, t1 in zip(t_seq[:-1], t_seq[1:]):
        dt = t1 - t0
        x1_hat = x1_pred_fn(x, t0)
        v = (x1_hat - x) / max(1 - t0, 1e-8)  # x1-예측을 velocity로 (임의의 v도 이 형태로 표현 가능)
        x_next = x + v * dt
        last = (x, v, dt)
        z_ref = (1 - t1) * eps + t1 * x_ref
        x = (1 - alpha) * x_next + alpha * z_ref
    return x, last


# ---------------- 1) 항등식 확인 ----------------
def check_identity(n=200):
    errs = []
    for _ in range(n):
        d = 13  # pose latent 차원 (6D rot + t3 + s3 + translation_scale 1)
        W = rng.normal(size=(d, d)) * 0.3
        y = rng.normal(size=d)
        fn = lambda x, t, W=W, y=y: y + np.tanh(W @ x) * (1 - t)  # 비선형, 상태 의존 예측
        eps, x_ref = rng.normal(size=d), rng.normal(size=d)
        a = rng.uniform(0, 1)
        x1, (x_prev, v, dt) = guided_sample(fn, x_ref, a, eps)
        x1_hat = x_prev + dt * v
        errs.append(np.abs(x1 - ((1 - a) * x1_hat + a * x_ref)).max())
    return float(np.max(errs))


# ---------------- 2) 실효 가중치 w_eff ----------------
def effective_weight(alpha, gamma, steps=T_STEPS):
    """예측 x̂1 = (1-γ)·y + γ·(현재 상태에서 외삽한 x1).  y=0, x_ref=1 로 두면 결과값이 곧 w_eff."""
    eps = np.zeros(1)

    def fn(x, t):
        extrap = (x - (1 - t) * eps) / max(t, 1e-8) if t > 0 else np.zeros(1)
        return (1 - gamma) * 0.0 + gamma * extrap

    x1, _ = guided_sample(fn, np.ones(1), alpha, eps, steps)
    return float(x1[0])


# ---------------- 3) 프레임 단위 추적 ----------------
def make_traj():
    vel = np.concatenate([np.full(60, 0.5), np.full(30, 8.0), np.full(30, 0.0), np.full(30, -3.0)])
    theta = np.concatenate([[0.0], np.cumsum(vel)])[:-1]
    return theta, vel


def track(obs, alpha_fn, gamma=0.3):
    est = np.zeros_like(obs)
    est[0] = obs[0]
    for k in range(1, len(obs)):
        a = alpha_fn(k)
        w = effective_weight(a, gamma)
        est[k] = (1 - w) * obs[k] + w * est[k - 1]
    return est


def metrics(est, theta):
    err = est - theta
    jitter = np.std(np.diff(est) - np.diff(theta))
    fast = slice(60, 90)
    return {"rmse": float(np.sqrt(np.mean(err ** 2))),
            "rmse_fast_phase": float(np.sqrt(np.mean(err[fast] ** 2))),
            "max_lag_fast_phase": float(np.max(np.abs(err[fast]))),
            "jitter": float(jitter)}


def main():
    out = {}
    out["identity_max_abs_err"] = check_identity()

    alphas = np.linspace(0, 0.95, 20)
    gammas = [0.0, 0.3, 0.6]
    out["w_eff"] = {f"gamma={g}": [round(effective_weight(a, g), 4) for a in alphas] for g in gammas}
    out["alphas"] = [round(float(a), 4) for a in alphas]

    theta, vel = make_traj()
    sigma = 4.0
    obs = theta + rng.normal(0, sigma, size=theta.shape)
    # 회전 속도 추정치(TAPIR 대용): 참값 + 잡음
    vel_hat = np.abs(vel + rng.normal(0, 0.5, size=vel.shape))
    adaptive = lambda k: max(0.1, 0.7 - 0.09 * vel_hat[k - 1])
    runs = {
        "no_guidance(α=0)": track(obs, lambda k: 0.0),
        "fixed α=0.5": track(obs, lambda k: 0.5),
        "fixed α=0.7": track(obs, lambda k: 0.7),
        "adaptive α(|Δθ|)": track(obs, adaptive),
    }
    out["tracking"] = {name: metrics(est, theta) for name, est in runs.items()}
    out["tracking_setup"] = {"sigma_deg": sigma, "gamma": 0.3,
                              "phases": "0-59: 0.5°/f, 60-89: 8°/f, 90-119: 0, 120-149: -3°/f"}
    (ROOT / "analysis" / "guidance_toy_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    figdir = ROOT / "report" / "figures"
    from matplotlib import font_manager
    font_path = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                                  "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
    if font_path:
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=200)
    for g, c in zip(gammas, ["#1b6f6a", "#d9822b", "#7a4fa3"]):
        ax.plot(alphas, out["w_eff"][f"gamma={g}"], color=c, lw=2, label=f"γ={g}")
    ax.plot(alphas, alphas, color="#999", ls="--", lw=1, label="y = α")
    ax.set_xlabel("pose guidance α_p")
    ax.set_ylabel("이전 프레임 pose 실효 가중치 w_eff")
    ax.set_title("α_p는 평활화 강도의 하한이다")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figdir / "toy_w_eff.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
    k = np.arange(len(theta))
    ax.plot(k, obs, ".", color="#bbb", ms=3, label="프레임별 독립 추정 (σ=4°)")
    ax.plot(k, theta, color="#222", lw=1.5, label="참값")
    ax.plot(k, runs["fixed α=0.7"], color="#d9822b", lw=1.8, label="고정 α=0.7")
    ax.plot(k, runs["adaptive α(|Δθ|)"], color="#1b6f6a", lw=1.8, label="적응형 α")
    ax.axvspan(60, 90, color="#1b6f6a", alpha=0.07)
    ax.text(62, theta.max() * 0.82, "빠른 회전 구간\n(8°/frame)", fontsize=8, color="#1b6f6a")
    ax.set_xlabel("frame")
    ax.set_ylabel("회전각 (deg)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figdir / "toy_tracking.png")
    plt.close(fig)
    print(json.dumps({k_: v for k_, v in out.items() if k_ != "w_eff"}, indent=2, ensure_ascii=False))
    print("w_eff(γ=0.3) at α=0.1/0.5/0.7:",
          [round(effective_weight(a, 0.3), 3) for a in (0.1, 0.5, 0.7)])


if __name__ == "__main__":
    main()
