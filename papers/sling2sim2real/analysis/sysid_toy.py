"""1자유도 탄성 밴드 toy: 한 번의 당김–놓기 기록으로 θ=(k, β, c, μ)를 맞추고, 식별성과 착탄 거리 오차 전파를 잰다.

실행: python3 -I analysis/sysid_toy.py  -> analysis/sysid_toy_results.json (CPU, 단일 스레드)

이 모델은 논문의 Isaac Sim 모델이 아니다(가정이 많다). 논문의 5개 파라미터 중 포아송 비는 1차원 장력 모델에서
식별되지 않으므로 뺐고, 감쇠 계수와 감쇠 스케일은 c 하나로 합쳤다. 따라서 4개 파라미터다.
- 장력: F(x, v) = k x (1 + β x) + c v + μ tanh(v / 0.01),  x > 0일 때만 작용(아니면 느슨함 → 0)
- 식별 기록: 손(질량 M_H = 0.5 kg, 가정)을 0–1 s에 0.40 m까지 3차 보간으로 당기고 1.5 s까지 유지한 뒤 놓는다.
  기록은 100 Hz로 위치 x(잡음 σ_x), 장력 f(잡음 σ_f)를 본다. 속도는 손 엔코더에서 얻는다고 가정하고 손실에는 쓰지 않는다.
- 손실(논문 Eq. 2–5의 1D 대응): L_geom = Σ((x_o−x_s)/σ_x)²/N (점군 Chamfer 대응), L_force = Σ((f_o−f_s)/σ_f)²/N,
  L_work = |W_o − W_s|/σ_W (W = ∫ f v dt, 사다리꼴). 가중치는 노이즈 단위로 맞춘 가정이다.
- 최적화: DE(scipy, 벡터화) → 히스토리 NMS(K=5, δ=0.1) → 최근접 15개로 공분산 → 5개 시작점에서 공분산 기반 CEM(시작점당 40개 표본, 상위 10개, 40 반복; CMA 대용) → Nelder-Mead 다듬기(두 변형 공통).
- 발사: 발사체(질량 M_P = 0.039 kg, 논문 값) 를 x = d 까지 당겼다 놓고, x = 0을 지날 때의 속도 v0로 R = v0² sin 2α / g.
  α = 35° (가정). 정책은 R(d; θ̂) = g를 풀어 d̂를 고른다(정책 대신 보정된 모델의 역함수).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "sysid_toy_results.json"

M_H, M_P, G_ACC, ALPHA = 0.5, 0.039, 9.81, np.deg2rad(35.0)
X_PULL, T_RAMP, T_REL, T_END = 0.40, 1.0, 1.5, 3.5
DT_INT, OBS_EVERY = 1e-3, 10          # 적분 1 ms, 관측 100 Hz
SIG_X, SIG_V, SIG_F, SIG_W = 3e-3, 5e-3, 0.05, 0.015  # 가정: 위치(점군 대응) 3 mm, 속도(엔코더) 5 mm/s, 힘 0.05 N; 일 0.015 J = 참 값 주위 W의 잡음 SD(400회 모의)
TRUE = np.array([9.0, 3.0, 0.15, 0.10])   # k [N/m], β [1/m], c [N s/m], μ [N]
GOALS = (1.725, 1.925, 2.125)              # 목표 거리 [m] (논문 172.5 / 192.5 / 212.5 cm)
NAMES = ["k", "beta", "c", "mu"]
REPS = int(os.environ.get("TOY_REPS", "20"))


# ---------- 물리 ----------
def to_phys(u):
    """단위 상자 u∈[0,1]^4 → 물리 θ. k는 log 척도 3–30, β 0–10, c는 log 척도 1e-3–1, μ 0–0.5."""
    u = np.asarray(u, float)
    k = 10 ** (np.log10(3.0) + u[..., 0] * (np.log10(30.0) - np.log10(3.0)))
    b = 10.0 * u[..., 1]
    c = 10 ** (-3.0 + 3.0 * u[..., 2])
    mu = 0.5 * u[..., 3]
    return np.stack([k, b, c, mu], axis=-1)


def to_unit(th):
    th = np.asarray(th, float)
    u0 = (np.log10(th[..., 0]) - np.log10(3.0)) / (np.log10(30.0) - np.log10(3.0))
    u1 = th[..., 1] / 10.0
    u2 = (np.log10(th[..., 2]) + 3.0) / 3.0
    u3 = th[..., 3] / 0.5
    return np.stack([u0, u1, u2, u3], axis=-1)


def tension(th, x, v):
    k, b, c, mu = th[:, 0], th[:, 1], th[:, 2], th[:, 3]
    if np.ndim(x) == 2:  # (P, N) 배열이면 θ를 열 방향으로
        k, b, c, mu = k[:, None], b[:, None], c[:, None], mu[:, None]
    F = k * x * (1 + b * x) + c * v + mu * np.tanh(v / 0.01)
    return np.where(x > 0, F, 0.0)


def pull_prescribed(t):
    """0–T_RAMP 동안 3차 보간(smoothstep)으로 0 → X_PULL, 그 뒤 유지(속도 0)."""
    s = np.clip(t / T_RAMP, 0, 1)
    x = X_PULL * (3 * s ** 2 - 2 * s ** 3)
    v = X_PULL * (6 * s - 6 * s ** 2) / T_RAMP
    v = np.where((t >= 0) & (t < T_RAMP), v, 0.0)
    return x, v


def id_record(th):
    """식별 기록. 반환: x, f (P × N), N = 350 (0–3.49 s, 100 Hz). 놓은 뒤는 적분."""
    t = 0.01 * np.arange(int(round(T_END / 0.01)))
    n_pre = int(round(T_REL / 0.01))
    xp, vp = pull_prescribed(t[:n_pre])
    P = th.shape[0]
    X = np.zeros((P, len(t)))
    V = np.zeros((P, len(t)))
    Fo = np.zeros((P, len(t)))
    X[:, :n_pre] = xp
    V[:, :n_pre] = vp
    Fo[:, :n_pre] = tension(th, np.broadcast_to(xp, (P, n_pre)), np.broadcast_to(vp, (P, n_pre)))
    x = np.full(P, X_PULL)
    v = np.zeros(P)
    j = n_pre
    for n in range(int(round((T_END - T_REL) / DT_INT))):
        if n % OBS_EVERY == 0:
            X[:, j] = x
            V[:, j] = v
            Fo[:, j] = tension(th, x, v)
            j += 1
        a = -tension(th, x, v) / M_H
        v = v + a * DT_INT
        x = x + v * DT_INT
    return X, Fo, V


def launch_speed(th, d):
    """발사체를 x=d까지 당겼다 놓고 x=0을 지나는 순간 속도(선형 보간). th: (P,4), d: (P,)."""
    P = th.shape[0]
    x = d.astype(float).copy()
    v = np.zeros(P)
    v0 = np.full(P, np.nan)
    done = np.zeros(P, bool)
    for _ in range(1200):
        a = -tension(th, x, v) / M_P
        v_new = v + a * 5e-4
        x_new = x + v_new * 5e-4
        cross = (~done) & (x_new <= 0)
        if cross.any():
            frac = x / np.maximum(x - x_new, 1e-12)
            v0[cross] = (v + frac * (v_new - v))[cross]
            done |= cross
        x, v = x_new, v_new
        if done.all():
            break
    return np.where(done, v0, 0.0)


def landing(th, d):
    v0 = launch_speed(th, d)
    return v0 ** 2 * np.sin(2 * ALPHA) / G_ACC


def solve_pull(th, goal, lo=0.10, hi=0.45):
    """R(d; θ) = goal을 이분법으로 풀어 pull 거리 d를 얻는다(R은 d에 단조 증가로 가정)."""
    lo = np.full(th.shape[0], lo)
    hi = np.full(th.shape[0], hi)
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        below = landing(th, mid) < goal
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


# ---------- 손실 ----------
def loss_fn(th, obs, variant):
    Xo, Fo_obs, Vo = (np.atleast_2d(a) for a in obs)
    Xs, Fs, Vs = id_record(th)
    N = Xo.shape[1]
    out = np.zeros(th.shape[0])
    if variant in ("full", "geom_force_work"):
        out += np.sum(((Xo - Xs) / SIG_X) ** 2, axis=1) / N
    out += np.sum(((Fo_obs - Fs) / SIG_F) ** 2, axis=1) / N
    if variant == "full":
        dt = 0.01
        Wo = np.trapz(Fo_obs * Vo, dx=dt, axis=1)
        Ws = np.trapz(Fs * Vs, dx=dt, axis=1)
        out += np.abs(Wo - Ws) / SIG_W
    return out


def fit(obs, variant, rng):
    hist_u, hist_L = [], []

    def obj(u_batch):  # scipy vectorized: (4, S) -> (S,)
        U = np.clip(np.asarray(u_batch).T, 0, 1)
        L = loss_fn(to_phys(U), obs, variant)
        hist_u.append(U.copy())
        hist_L.append(L.copy())
        return L

    res = differential_evolution(obj, bounds=[(0, 1)] * 4, vectorized=True, popsize=10, maxiter=30,
                                 mutation=(0.5, 1.0), recombination=0.7, tol=0, polish=False,
                                 seed=int(rng.integers(1 << 31)), init="latinhypercube")
    H = np.vstack(hist_u)
    L = np.concatenate(hist_L)
    best_de = np.clip(res.x, 0, 1)
    # NMS: 손실 순서로 훑으며 δ=0.1 밖의 중심만 K=5개
    order = np.argsort(L)
    centers = []
    for i in order:
        if all(np.linalg.norm(H[i] - H[j]) > 0.1 for j in centers):
            centers.append(i)
        if len(centers) == 5:
            break
    best_u, best_L = best_de, float(res.fun)
    cand = []
    for ci in centers:
        mu = H[ci].copy()
        d = np.linalg.norm(H - mu, axis=1)
        nn = np.argsort(d)[:15]
        cov = np.cov(H[nn].T) + 1e-6 * np.eye(4)
        cand.append((mu, cov))
    # 공분산 기반 CEM 5개 시작점, 한 번에 벡터화
    for it in range(40):
        newc = []
        for mu, cov in cand:
            Lc = np.linalg.cholesky(cov)
            Z = rng.standard_normal((40, 4))
            U = np.clip(mu + Z @ Lc.T, 0, 1)
            newc.append(U)
        allU = np.vstack(newc)
        Ls = loss_fn(to_phys(allU), obs, variant)
        k = 0
        new_cand = []
        for (mu, cov), U_block in zip(cand, newc):
            Lb = Ls[k:k + 40]
            k += 40
            el = U_block[np.argsort(Lb)[:10]]
            if Lb.min() < best_L:
                best_L = float(Lb.min())
                best_u = U_block[np.argmin(Lb)]
            new_cand.append((el.mean(0), np.cov(el.T) + 1e-5 * np.eye(4)))
        cand = new_cand
    # 마지막 국소 다듬기: 이 toy의 손실은 일 항(절댓값) 때문에 골짜기가 좁아 CEM만으로는 부족했다.
    # 두 손실 변형 모두에 같은 Nelder-Mead 단계를 적용한다(공정성). 단계 수는 고정.
    def scalar(u):
        u = np.clip(u, 0, 1)
        return float(loss_fn(to_phys(u[None]), obs, variant)[0])

    nm = minimize(scalar, best_u, method="Nelder-Mead", options={"maxfev": 300, "xatol": 1e-4, "fatol": 1e-4})
    if nm.fun < best_L:
        best_L, best_u = float(nm.fun), np.clip(nm.x, 0, 1)
    return to_phys(best_u), to_phys(best_de), best_L, float(res.fun), int(H.shape[0])


# ---------- 식별성 ----------
def fisher(th0):
    """log θ에 대한 Fisher 정보(위치, 힘, 속도 채널, 노이즈 σ로 정규화). 중앙차분, 상대 step 1e-3."""
    h = 1e-3
    batch = [th0]
    for j in range(4):
        for s in (+1, -1):
            t = th0.copy()
            t[j] = th0[j] * np.exp(s * h)
            batch.append(t)
    B = np.array(batch)
    X, F, Vv = id_record(B)
    J = []
    for j in range(4):
        dX = (X[1 + 2 * j] - X[2 + 2 * j]) / (2 * h)
        dF = (F[1 + 2 * j] - F[2 + 2 * j]) / (2 * h)
        dV = (Vv[1 + 2 * j] - Vv[2 + 2 * j]) / (2 * h)
        J.append(np.concatenate([dX / SIG_X, dF / SIG_F, dV / SIG_V]))
    J = np.array(J)
    Fi = J @ J.T
    return Fi


def landing_grad(th0, d0, dh=1e-3):
    g = []
    for j in range(4):
        tp, tm = th0.copy(), th0.copy()
        tp[j] *= np.exp(dh)
        tm[j] *= np.exp(-dh)
        g.append((landing(np.array([tp, tm]), np.array([d0, d0]))))
    return np.array([(x[0] - x[1]) / (2 * dh) for x in g])


def main():
    t0 = time.time()
    rng = np.random.default_rng(20260711)
    out = {"assumptions": dict(M_H=M_H, M_P=M_P, ALPHA_deg=float(np.rad2deg(ALPHA)), X_PULL=X_PULL,
                               SIG_X=SIG_X, SIG_F=SIG_F, SIG_W=SIG_W, TRUE=TRUE.tolist(),
                               names=NAMES, reps=REPS, goals=GOALS)}

    # 1) 식별성: Fisher 행렬(정규화), 상관, 조건수, 1 cm 민감도
    Fi = fisher(TRUE)
    C = np.linalg.inv(Fi + 1e-12 * np.eye(4))
    corr = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    eig = np.linalg.eigvalsh(Fi)
    Dh = np.diag(1 / np.sqrt(np.diag(Fi)))
    out["identifiability"] = {
        "fisher_log_param_diag": np.diag(Fi).tolist(),
        "std_log_param_CRLB": np.sqrt(np.diag(C)).tolist(),
        "rel_std_CRLB_pct": (100 * np.sqrt(np.diag(C))).round(3).tolist(),
        "corr_CRLB": corr.round(3).tolist(),
        "cond_scaled_fisher": float(np.linalg.cond(Dh @ Fi @ Dh)),
        "eig_scaled_fisher": (np.linalg.eigvalsh(Dh @ Fi @ Dh)).tolist(),
    }

    # 2) 발사 민감도: 목표 거리 1.925 m에서 d*, ∂R/∂ln θ, 1 cm 당김 오차
    d_star = solve_pull(TRUE[None], np.array([GOALS[1]]))
    gl = landing_grad(TRUE, float(d_star[0]))
    dR_dd = (landing(TRUE[None], d_star + 0.001) - landing(TRUE[None], d_star - 0.001))[0] / 0.002
    out["sensitivity"] = {
        "d_star_m_at_1925": float(d_star[0]),
        "dR_dlnθ_m": dict(zip(NAMES, gl.round(4).tolist())),
        "dR_dd_m_per_m": float(dR_dd),
        "R_error_per_1cm_pullback_cm": float(100 * abs(dR_dd) * 0.01),
        "R_curve_cm": {f"{d:.3f}": round(float(landing(TRUE[None], np.array([d]))[0]) * 100, 1)
                       for d in np.arange(0.18, 0.31, 0.02)},
    }
    gvar = np.array(gl)
    C_R = gvar @ C @ gvar
    out["sensitivity"]["CRLB_landing_std_cm_from_fisher"] = float(100 * np.sqrt(max(C_R, 0)))

    # 3) 반복 적합: (a) 손실 full, (b) 힘만(L_force), 각각 DE만 / DE+공분산 CEM
    # 기본 보정 없이 쓰는 설정 θ0(상자 중간값 근처)
    theta0 = np.array([10.0, 5.0, 0.5, 0.25])
    truth_th = np.repeat(TRUE[None], REPS, axis=0)
    obs_list = []
    for r in range(REPS):
        X, F, V = id_record(TRUE[None])
        obs_list.append((X[0] + rng.normal(0, SIG_X, X.shape[1]), F[0] + rng.normal(0, SIG_F, F.shape[1]),
                         V[0] + rng.normal(0, SIG_V, V.shape[1])))
    fits = {}
    for variant in ("full", "force_only"):
        vname = "full" if variant == "full" else "force_only"
        rec = {"theta_cma": [], "theta_de": [], "loss": [], "n_eval": []}
        for r in range(REPS):
            obs = obs_list[r]  # force_only 는 loss_fn에서 힘 항만 쓴다
            th_cma, th_de, Lbest, Lde, n_eval = fit(obs, "full" if vname == "full" else "force", rng)
            rec["theta_cma"].append(th_cma.tolist())
            rec["theta_de"].append(th_de.tolist())
            rec["loss"].append(Lbest)
            rec["n_eval"].append(n_eval)
        fits[vname] = rec
    out["fits_meta"] = {"n_reps": REPS, "runtime_s_so_far": round(time.time() - t0, 1)}

    # 4) 착탄 거리 오차: 기본값, DE만, CEM 포함, 목표 3개
    Lk = {}
    for vname, rec in fits.items():
        th_c = np.array(rec["theta_cma"])
        th_d = np.array(rec["theta_de"])
        for label, TH in (("cma", th_c), ("de", th_d)):
            for g in GOALS:
                d_hat = solve_pull(TH, np.full(REPS, g))
                err = 100 * (landing(np.repeat(TRUE[None], REPS, axis=0), d_hat) - g)
                key = f"{vname}_{label}_g{g}"
                Lk[key] = {"abs_err_cm_mean": round(float(np.mean(np.abs(err))), 2),
                           "abs_err_cm_median": round(float(np.median(np.abs(err))), 2),
                           "abs_err_cm_p90": round(float(np.percentile(np.abs(err), 90)), 2)}
    d_def = solve_pull(np.repeat(theta0[None], REPS, axis=0), np.full(REPS, GOALS[1]))
    err_def = 100 * (landing(np.repeat(TRUE[None], REPS, axis=0), d_def) - GOALS[1])
    Lk["default_theta_g1.925"] = {"abs_err_cm_mean": round(float(np.mean(np.abs(err_def))), 2),
                                  "abs_err_cm_median": round(float(np.median(np.abs(err_def))), 2),
                                  "abs_err_cm_p90": round(float(np.percentile(np.abs(err_def), 90)), 2)}
    out["landing_error"] = Lk

    # 5) θ 추정 오차(상대, %)
    rel = {}
    for vname, rec in fits.items():
        th = np.array(rec["theta_cma"])
        rel[vname] = {NAMES[j]: {"median_rel_err_pct": round(float(np.median(100 * np.abs(th[:, j] / TRUE[j] - 1))), 1),
                                 "p90_rel_err_pct": round(float(np.percentile(100 * np.abs(th[:, j] / TRUE[j] - 1), 90)), 1)}
                      for j in range(4)}
    out["theta_error"] = rel
    out["fits_raw"] = fits
    out["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("fits_raw",)}, ensure_ascii=False)[:2500])


if __name__ == "__main__":
    main()
