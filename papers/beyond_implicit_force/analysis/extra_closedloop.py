"""추가 검증 1 (beyond_implicit_force): 1-DoF 폐루프 toy. 실행: python3 analysis/extra_closedloop.py

사람 시연(leader를 밀어 접촉력 6 N을 만듦)을 numpy로 만들고, 정책 네 변형을 RBF 능선 회귀로 학습한다.
  Base ACT   : 관측 qpos(z), 행동 = leader 목표
  ACT + τ    : 관측 (z, τ), 행동 = leader 목표
  ACT-o      : 관측 z, 행동 = follower 목표
  ACT-o + τ  : 관측 (z, τ), 행동 = follower 목표
행동은 미래 H 스텝의 목표(ACT식 chunk 대신 H 스텝 앞선 목표)이고, 서보는 P만 있는 준정적 접촉 모델(1차 지연 포함)이다.
확인: (a) ACT-o + τ가 접촉은 찾아도 시연 힘(6 N)을 내지 못하는가, (b) leader 목표가 힘을 복원하는가,
(c) 표면 강성 변화(OOD)와 예측 편향·잡음에 따라 어떻게 달라지는가.
강성·질량·마찰은 toy 가정이므로 절대값보다 경향을 본다. 폐루프는 매 스텝 정책을 다시 호출한다(ACT의 chunk 실행은 흉내 내지 않음).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

KP, MG, F_DEMO, F_MIN = 800.0, 1.5, 6.0, 2.0     # 가정 (force_proxy_toy.py와 같은 값)
K_NOM, K_SOFT = 2e4, 2e3                          # 학습은 지우개+보드(nominal), 평가는 무른 패드(soft)도
DT, WC, H = 0.02, 12.0, 10                        # 50 Hz, follower 1차 지연 대역폭 [rad/s], 미래 목표 H = 0.2 s
T_EP = 8.0
N = int(T_EP / DT)
N_DEMO, N_SEED = 6, 20
TAU_NOISE = 0.15                                  # 토크 대리값 잡음 [N] (table: T2와 같음)
VARIANTS = {"Base ACT": (False, "leader"), "ACT + τ": (True, "leader"),
            "ACT-o": (False, "follower"), "ACT-o + τ": (True, "follower")}


def eq_follow(z_cmd, k_e):
    """P 서보 + 중력 + 표면의 준정적 평형 위치 (force_proxy_toy.contact_force와 같은 식)."""
    z_free = z_cmd + MG / KP
    return np.where(z_free > 0, (KP * z_cmd + MG) / (KP + k_e), z_free)


def servo_step(z_prev, z_cmd, k_e):
    return z_prev + (eq_follow(z_cmd, k_e) - z_prev) * min(1.0, WC * DT)


def demo(rng, z_start, v):
    """사람: 표면 위에서는 v로 내리고, 표면에 닿으면 보이는 힘(k_e·z_f)을 6 N에 맞추도록 leader를 민다."""
    k_e = K_NOM
    zL, zf = np.empty(N), np.empty(N)
    zL[0] = z_start
    zf[0] = eq_follow(z_start, k_e)
    for i in range(1, N):
        f_seen = k_e * max(zf[i - 1], 0.0)
        if zf[i - 1] < -0.002:
            zL[i] = zL[i - 1] + v * DT
        else:
            zL[i] = zL[i - 1] + 0.05 * (F_DEMO - f_seen) / F_DEMO * DT
        zf[i] = servo_step(zf[i - 1], zL[i], k_e)
    F = k_e * np.maximum(zf, 0.0)
    tau = F + rng.normal(0, TAU_NOISE, N)
    return zL, zf, F, tau


def chunk(x):
    idx = np.minimum(np.arange(N) + H, N - 1)
    return x[idx]


CZ = np.linspace(-80, 15, 48)      # mm
WZ = 2.0
CT = np.linspace(0, 8, 9)          # N
WT = 1.0


def phi(z_mm, tau=None):
    fz = np.exp(-0.5 * ((z_mm[:, None] - CZ[None, :]) / WZ) ** 2)
    if tau is None:
        return fz
    ft = np.exp(-0.5 * ((tau[:, None] - CT[None, :]) / WT) ** 2)
    return (fz[:, :, None] * ft[:, None, :]).reshape(len(z_mm), -1)


def fit(demos, use_tau, target_kind, lam=1e-2):
    Z = np.concatenate([d["zf"] for d in demos]) * 1e3
    T = np.concatenate([d["tau"] for d in demos])
    Y = np.concatenate([chunk(d["zL"] if target_kind == "leader" else d["zf"]) for d in demos])
    P = phi(Z, T if use_tau else None)
    mu = Y.mean()
    W = np.linalg.solve(P.T @ P + lam * np.eye(P.shape[1]), P.T @ (Y - mu))
    return lambda z_mm, tau: float((phi(np.array([z_mm]), np.array([tau]) if use_tau else None) @ W)[0] + mu)


def rollout(policy, use_tau, k_e, seed, bias=0.0, sig=0.0, sens_mm=0.2, z0=-0.06):
    rng = np.random.default_rng(seed)
    zf = eq_follow(z0, k_e)
    F = np.zeros(N)
    for i in range(N):
        z_obs = zf * 1e3 + rng.normal(0, sens_mm)
        tau_obs = k_e * max(zf, 0.0) + rng.normal(0, TAU_NOISE)
        a = policy(z_obs, tau_obs) + bias + sig * rng.normal()
        zf = servo_step(zf, a, k_e)
        F[i] = k_e * max(zf, 0.0)
    hold = float(F[-int(3 / DT):].mean())
    contact_t = np.flatnonzero(F > 0.5)
    return {"hold_N": hold, "contact": bool(contact_t.size > 0),
            "contact_time_s": float(contact_t[0] * DT) if contact_t.size else None}


def summarize(runs):
    h = np.array([r["hold_N"] for r in runs])
    return {"hold_mean_N": round(float(h.mean()), 2), "hold_median_N": round(float(np.median(h)), 2),
            "P_contact": round(float(np.mean([r["contact"] for r in runs])), 3),
            "P_hold_ge_F_min": round(float((h >= F_MIN).mean()), 3),
            "P_restore_ge_0.8F": round(float((h >= 0.8 * F_DEMO).mean()), 3)}


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                              "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
    if font:
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
    colors = {"Base ACT": "#2a78d6", "ACT + τ": "#1baf7a", "ACT-o": "#9a9a9f", "ACT-o + τ": "#eb6834"}
    biases = [-2, -1, 0, 1, 2]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, k_name, title in zip(axes, ("nominal", "soft"),
                                 ("학습 표면 (K = 2×10⁴ N/m)", "무른 패드 (K = 2×10³ N/m, 분포 밖)")):
        for name, c in colors.items():
            med = [out["E3_sweep"][k_name][name][f"bias{b:+d}mm_sig0mm"]["hold_median_N"] for b in biases]
            ax.plot(biases, med, "o-", color=c, label=name, lw=1.6, ms=4)
        ax.axhline(F_DEMO, color="#0b0b0b", ls="--", lw=0.8)
        ax.axhline(F_MIN, color="#52514e", ls=":", lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("출력 편향 [mm] (+는 더 깊게)")
        ax.grid(color="#e4e4e7", lw=0.6)
    axes[0].set_ylabel("유지 힘 중앙값 [N]")
    axes[0].text(-1.9, F_DEMO + 0.25, "시연 6 N", fontsize=8)
    axes[0].text(-1.9, F_MIN + 0.2, "최소 2 N", fontsize=8)
    axes[1].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(ROOT / "report" / "figures" / "extra_closedloop.png", dpi=200)
    plt.close(fig)


def main():
    drng = np.random.default_rng(0)
    demos = []
    for j in range(N_DEMO):
        z_start = -0.06 + 0.005 * j
        v = drng.uniform(0.02, 0.035)
        zL, zf, F, tau = demo(drng, z_start, v)
        demos.append({"zL": zL, "zf": zf, "F": F, "tau": tau})
    hold_L = float(np.mean([d["zL"][-int(3 / DT):].mean() for d in demos]))
    hold_f = float(np.mean([d["zf"][-int(3 / DT):].mean() for d in demos]))
    demo_F = float(np.mean([d["F"][-int(3 / DT):].mean() for d in demos]))

    # 이상적 재생(정책 없이 목표 위치를 그대로 명령): 힘이 얼마나 나는지
    ideal = {}
    for k_name, k_e in (("nominal", K_NOM), ("soft", K_SOFT)):
        ideal[k_name] = {
            "leader_target_F_N": round(float(k_e * max(float(eq_follow(hold_L, k_e)), 0.0)), 2),
            "follower_target_F_N": round(float(k_e * max(float(eq_follow(hold_f, k_e)), 0.0)), 2),
        }
    # 정책 학습 (nominal 표면 시연만 사용)
    pols = {name: fit(demos, use_tau, kind) for name, (use_tau, kind) in VARIANTS.items()}
    use_tau_of = {name: v[0] for name, v in VARIANTS.items()}

    def run(name, k_e, bias=0.0, sig=0.0, n=N_SEED):
        return [rollout(pols[name], use_tau_of[name], k_e, seed=1000 + s, bias=bias, sig=sig) for s in range(n)]

    out = {"assumptions": {"Kp_eef_N_per_m": KP, "mg_N": MG, "F_demo_N": F_DEMO, "F_min_N": F_MIN,
                           "K_nominal": K_NOM, "K_soft": K_SOFT, "servo_bw_rad_s": WC, "chunk_lead_steps": H,
                           "episode_s": T_EP, "tau_noise_N": TAU_NOISE, "n_demo": N_DEMO, "n_seed": N_SEED,
                           "policy": "RBF 능선 회귀(ridge, λ=1e-2), 행동 = 미래 H 스텝 목표, 매 스텝 재호출"},
           "demo": {"hold_leader_target_mm": round(hold_L * 1e3, 2), "hold_follower_target_mm": round(hold_f * 1e3, 3),
                    "hold_force_N": round(demo_F, 2)},
           "ideal_replay": ideal, "E1_nominal": {}, "E2_soft_OOD": {}, "E3_sweep": {}}

    for name in VARIANTS:
        out["E1_nominal"][name] = summarize(run(name, K_NOM))
        out["E2_soft_OOD"][name] = summarize(run(name, K_SOFT))

    biases = [-2, -1, 0, 1, 2]
    sigmas = [0, 1, 2]
    for k_name, k_e in (("nominal", K_NOM), ("soft", K_SOFT)):
        out["E3_sweep"][k_name] = {}
        for name in VARIANTS:
            grid = {}
            for b in biases:
                for s in sigmas:
                    rs = run(name, k_e, bias=b * 1e-3, sig=s * 1e-3, n=10)
                    grid[f"bias{b:+d}mm_sig{s}mm"] = summarize(rs)
            out["E3_sweep"][k_name][name] = grid

    path = ROOT / "analysis" / "extra_closedloop_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    plot(out)
    print(json.dumps({k: out[k] for k in ("demo", "ideal_replay", "E1_nominal", "E2_soft_OOD")},
                     indent=1, ensure_ascii=False))
    print("wrote", path)


if __name__ == "__main__":
    main()
