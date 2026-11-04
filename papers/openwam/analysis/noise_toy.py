"""OpenWAM 결합 noise 평면 toy: Fig. 9 비동기 스케줄은 왜 sync보다 나쁜가.

실행: python3 analysis/noise_toy.py  → analysis/noise_toy_results.json (단일 스레드 numpy, 수십 초)

A. 학습 분포(코드 그대로): 두 스트림이 1000칸 grid의 index를 서로 독립·균등하게 뽑고(architectures/base.py),
   σ = f5(1 − i/1000)이다(action_backbone/scheduler.py). 손실 가중치는 σ·1000에 대한 종 모양에서 최솟값을 뺀 값이다.
   → index(u) 좌표에서 학습 표본은 평면 전체에 균등하다. 스케줄 경로도 같은 f5 warp를 거치므로 "경로가 학습 분포 밖으로
   나간다"는 설명은 성립하지 않는다. 남는 차이는 각 경로의 단계 배분(10 step)과 손실 가중치다.
B. 이상적 denoiser toy: (미래 영상, 행동)이 다봉 Gaussian 혼합인 세계에서 정확한 사후 평균으로 velocity를 만들고,
   deploy/denoise_schedule.py와 같은 식의 17개 스케줄(N = 10)로 Euler 적분한다. 모델 오차가 없을 때 남는
   이산화 오차만으로 Fig. 9의 순서가 나오는지 본다.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
SHIFT = 5.0
N_STEPS = 10

# Fig. 9 보고값 (RoboTwin2.0-Full 평균 성공률 %)
FIG9 = {("sync", None, 1.0, 0.0): 93.0}
for a, v in zip((4, 8, 16, 32), (91.0, 90.7, 89.7, 88.5)):
    FIG9[("VS", "action", float(a), 0.0)] = v
for a, v in zip((4, 8, 16, 32), (91.8, 91.1, 91.9, 91.5)):
    FIG9[("VS", "video", float(a), 0.0)] = v
for o, v in zip((0.2, 0.4, 0.6, 0.8), (92.2, 92.0, 92.3, 92.3)):
    FIG9[("LO", "action", 1.0, o)] = v
for o, v in zip((0.2, 0.4, 0.6, 0.8), (91.7, 91.1, 91.5, 91.2)):
    FIG9[("LO", "video", 1.0, o)] = v


def alpha_shift(u, shift):
    return shift * u / (1.0 + (shift - 1.0) * u)


def schedule(lead, alpha, offset, n=N_STEPS):
    """deploy/denoise_schedule.py schedule_variance_shift를 numpy float32로 옮김. σ 쌍 (n+1, 2), 마지막은 (0, 0)."""
    s = np.linspace(1.0, 0.0, n + 1, dtype=np.float32)[:-1]
    lead_sigma = alpha_shift(s, np.float32(1.0 / alpha))
    lag_sigma = np.minimum(s / (1.0 - offset), 1.0) if offset > 0 else s
    if lead == "video" or lead is None:
        v, a = lead_sigma, lag_sigma
    else:
        v, a = lag_sigma, lead_sigma
    sv, sa = alpha_shift(v, SHIFT), alpha_shift(a, SHIFT)
    return np.stack([np.append(sv, 0.0), np.append(sa, 0.0)], 1).astype(np.float64)


def name_of(key):
    fam, lead, a, o = key
    if fam == "sync":
        return "sync"
    return f"{fam} {lead}-lead " + (f"α={int(a)}" if fam == "VS" else f"o={o}")


# ---------------------------------------------------------------- A. 학습 분포와 손실 가중치
grid_sigma = alpha_shift(np.linspace(1, 0, 1001)[:-1], SHIFT)  # 학습 grid σ (index 0 = 순수 noise)
x = grid_sigma * 1000
y = np.exp(-2 * ((x - 500) / 1000) ** 2)
w = (y - y.min()) * (1000 / (y - y.min()).sum())  # scheduler._set_training_weight


def w_of(sig):
    return np.interp(sig, grid_sigma[::-1], w[::-1])


def u_of(sig):
    """σ → index 좌표 u ∈ [0, 1) (f5의 역함수, 0 = noise 쪽)."""
    return 1 - sig / (SHIFT - (SHIFT - 1) * sig)


bins = [(0.9, 1.01), (0.5, 0.9), (0.1, 0.5), (0.0, 0.1)]
marg = {f"σ∈[{lo},{min(hi, 1)}]": {"sample_frac": float(((grid_sigma >= lo) & (grid_sigma < hi)).mean()),
                                    "weighted_frac": float(w[(grid_sigma >= lo) & (grid_sigma < hi)].sum() / w.sum())}
        for lo, hi in bins}
du = 0.05
diag_band = 1 - (1 - du) ** 2  # |u_v − u_a| < δ 인 균등 질량
A = {
    "sigma_grid_note": "u = index/1000 균등, σ = f5(1 − u). index 쌍 (i_v, i_a)는 1000×1000 격자 위에서 균등",
    "marginal": marg,
    "weight_at_sigma": {str(s): float(w_of(s)) for s in (1.0, 0.95, 0.9, 0.833, 0.5, 0.2, 0.05, 0.005)},
    "diagonal_band_mass_du_0.05": float(diag_band),
}

# ---------------------------------------------------------------- B. 다봉 Gaussian 혼합 toy
rng = np.random.default_rng(0)
K, DA, DV = 4, 4, 8
mu_a = rng.normal(size=(K, DA))
mu_a = 1.5 * mu_a / np.linalg.norm(mu_a, axis=1, keepdims=True)
G = np.linalg.qr(rng.normal(size=(DV, DA)))[0] * 1.5  # 미래 영상이 행동을 반영


def make_world(s_a, s_v):
    mu = np.concatenate([mu_a @ G.T, mu_a], 1)  # (K, DV+DA): [video, action]
    Sa = s_a ** 2 * np.eye(DA)
    Sig = np.block([[G @ Sa @ G.T + s_v ** 2 * np.eye(DV), G @ Sa], [Sa @ G.T, Sa]])
    return mu, Sig


def posterior_velocity(xt, sv, sa, mu, Sig):
    """정확한 E[ε − x | x_t] (code 부호: target = noise − original)."""
    d = np.concatenate([np.full(DV, sv), np.full(DA, sa)])
    one_m = 1 - d
    Kc = one_m[:, None] * Sig * one_m[None, :] + np.diag(d ** 2)
    Kinv = np.linalg.inv(Kc)
    diff = xt[:, None, :] - (one_m * mu)[None]  # (M, K, D)
    maha = np.einsum("mkd,de,mke->mk", diff, Kinv, diff)
    logr = -0.5 * maha
    logr -= logr.max(1, keepdims=True)
    rsp = np.exp(logr)
    rsp /= rsp.sum(1, keepdims=True)
    gain = (Sig * one_m[None, :]) @ Kinv  # Σ(1−S)K⁻¹
    post = mu[None] + np.einsum("de,mke->mkd", gain, diff)
    ex = (rsp[..., None] * post).sum(1)
    eeps = (xt - one_m * ex) / d
    return eeps - ex


def sample(path, mu, Sig, M, seed):
    r = np.random.default_rng(seed)
    xt = r.normal(size=(M, DV + DA))
    for k in range(len(path) - 1):
        (sv, sa), (nv, na) = path[k], path[k + 1]
        if sv == nv and sa == na:
            continue
        vel = posterior_velocity(xt, sv, sa, mu, Sig)
        step = np.concatenate([np.full(DV, nv - sv), np.full(DA, na - sa)])
        xt = xt + step * vel
    return xt


def metrics(xs, mu, s_a, s_v):
    v, a = xs[:, :DV], xs[:, DV:]
    da2 = ((a[:, None, :] - mu_a[None]) ** 2).sum(-1) / s_a ** 2
    ka = da2.argmin(1)
    valid = da2.min(1) < stats.chi2.ppf(0.99, DA)
    dv2 = ((v[:, None, :] - (mu_a @ G.T)[None]) ** 2).sum(-1)
    kv = dv2.argmin(1)
    freq = np.bincount(ka, minlength=K) / len(ka)
    return {"action_valid": float(valid.mean()), "action_video_same_mode": float((ka == kv).mean()),
            "mode_freq_max_dev": float(np.abs(freq - 1 / K).max()),
            "action_dist_to_mode_sd": float(np.sqrt(da2.min(1)).mean())}


keys = list(FIG9)
paths = {name_of(k): schedule(k[1], k[2], k[3]) for k in keys}
M = 20000
worlds = {"tight (s=0.10)": (0.10, 0.10), "base (s=0.15)": (0.15, 0.15), "loose (s=0.25)": (0.25, 0.25)}
B = {}
for wname, (s_a, s_v) in worlds.items():
    mu, Sig = make_world(s_a, s_v)
    res = {}
    for k in keys:
        nm = name_of(k)
        res[nm] = metrics(sample(paths[nm], mu, Sig, M, 1), mu, s_a, s_v)
    ref = metrics(sample(schedule(None, 1.0, 0.0, n=400), mu, Sig, M, 1), mu, s_a, s_v)
    ref_a32 = metrics(sample(schedule("action", 32.0, 0.0, n=400), mu, Sig, M, 1), mu, s_a, s_v)
    xv = [res[name_of(k)]["action_valid"] for k in keys]
    yv = [FIG9[k] for k in keys]
    rho, p = stats.spearmanr(xv, yv)
    B[wname] = {"per_schedule": res, "sync_400step": ref, "action_lead_a32_400step": ref_a32,
                "spearman_valid_vs_fig9": {"rho": round(float(rho), 3), "p": float(f"{p:.3g}"), "n": len(keys)}}

# 경로 통계 (u 좌표, 손실 가중치)
pstats = {}
for k in keys:
    nm = name_of(k)
    pth = paths[nm]
    ua = u_of(np.clip(pth[:, 1], 0, 1))
    uv = u_of(np.clip(pth[:, 0], 0, 1))
    upd = np.diff(pth[:, 1]) < 0  # 행동이 갱신되는 step
    # 행동이 절반(σa ≤ 0.5)을 지날 때 영상 σ
    j = int(np.argmax(pth[:, 1] <= 0.5))
    pstats[nm] = {
        "fig9": FIG9[k],
        "action_updates": int(upd.sum()),
        "max_action_du": round(float(np.max(np.diff(ua))), 3),
        "first_action_step_sigma": [round(float(pth[0, 1]), 3), round(float(pth[1, 1]), 3)],
        "video_sigma_when_action_half": round(float(pth[j, 0]), 3),
        "mean_action_loss_weight_on_updates": round(float(w_of(pth[:-1, 1][upd]).mean()), 3),
        "max_u_gap_between_streams": round(float(np.max(np.abs(ua - uv))), 3),
    }
yfig = [FIG9[k] for k in keys]
corr = {}
for f in ("action_updates", "max_action_du", "video_sigma_when_action_half", "mean_action_loss_weight_on_updates",
          "max_u_gap_between_streams"):
    rho, p = stats.spearmanr([pstats[name_of(k)][f] for k in keys], yfig)
    corr[f] = {"rho": round(float(rho), 3), "p": float(f"{p:.3g}")}

OUT = {"A_training_distribution": A, "B_gmm_ideal_denoiser": B, "path_stats": pstats,
       "spearman_path_stat_vs_fig9": corr,
       "paths_sigma": {nm: [[round(float(a), 4), round(float(b), 4)] for a, b in p] for nm, p in paths.items()}}
(HERE / "noise_toy_results.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1))
for wname in B:
    print(wname, B[wname]["spearman_valid_vs_fig9"], "sync400", B[wname]["sync_400step"]["action_valid"],
          "a32_400", B[wname]["action_lead_a32_400step"]["action_valid"])
    for nm, m in B[wname]["per_schedule"].items():
        print(f"   {nm:24s} fig9={pstats[nm]['fig9']:5.1f} valid={m['action_valid']:.3f} same={m['action_video_same_mode']:.3f}")
print(json.dumps(corr, ensure_ascii=False))
print(json.dumps(A["marginal"], ensure_ascii=False), A["diagonal_band_mass_du_0.05"])
