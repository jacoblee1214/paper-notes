"""ResTacVLA 핵심 가정 두 개를 numpy toy로 점검한다 (CPU 1스레드, 약 30초).
실행: python3 analysis/gate_toy.py → analysis/gate_toy_results.json

A. 게이트 입력: 논문 Eq. 3은 g = Sigmoid(MLP(σ_t))이고 σ_t는 손목 영상만으로 예측한 불확실성이다.
   1차원 접근 → 접촉 → 해제 에피소드에 "시각으로는 보이지 않는 자유 공간 충돌"을 넣고,
   σ 게이트 / 원시 예측 오차 게이트 / 정밀도 가중 오차(고전적 surprise) 게이트 / σ+오차 결합 게이트를
   같은 로지스틱 보정으로 맞춰 비교한다. 목표 라벨은 "촉각에 정보가 있는 순간(접촉 중)"이다.
B. 잔차 vs 원시 촉각: r = z − Ĉv는 [v, z]의 선형 재매개화라 OLS에서는 예측이 같다.
   차이는 정규화(모달리티별 표준화) + ridge + 유한 데이터에서만 생길 수 있다.
   예측 가능 비율 ρ²와 데이터 수, CMP 학습 데이터 수를 바꿔 가며 검증 세트로 λ를 고른 뒤 비교한다.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
T = 200          # 스텝 수
DZ = 8           # 촉각 잠재 차원
T_ON = 120       # 평균 접촉 시작
T_OFF = 175      # 해제(성공) 시작


# ---------------------------------------------------------------------------
# A. 게이트 toy
# ---------------------------------------------------------------------------
def episodes(rng, n, p_collide, sigma_n, collide_mag=1.0):
    """반환: v (n,T) 시각 거리 추정, z (n,T,DZ) 촉각 잠재, contact (n,T) 기대 접촉, collide (n,T) 예상 밖 충돌, t_on (n,)"""
    t = np.arange(T)
    t_on = T_ON + rng.integers(-8, 9, size=n)                 # 접촉 시점 흔들림(시각으로 알 수 없음)
    d = np.clip((t_on[:, None] - t[None, :]) / t_on[:, None], 0, None)  # 실제 간극 1 → 0
    rel = np.clip((t[None, :] - T_OFF) / 25, 0, None) * 0.3    # 해제 뒤 다시 벌어짐
    d = np.where(t[None, :] >= T_OFF, rel, d)
    sig_v = 0.03 + 0.07 * np.exp(-d / 0.08)                    # 접촉 근처 가림으로 시각 오차 증가
    v = d + rng.normal(size=d.shape) * sig_v
    contact = (t[None, :] >= t_on[:, None]) & (t[None, :] < T_OFF)
    force = rng.uniform(0.5, 1.5, size=n)                      # 에피소드마다 다른 접촉력
    proto = np.linalg.qr(np.random.default_rng(7).normal(size=(DZ, DZ)))[0][:, :3].T  # 과제 접촉 패턴 3개
    u = proto[rng.integers(0, 3, size=n)]
    ramp = np.clip(1 - np.exp(-(t[None, :] - t_on[:, None]) / 5.0), 0, None) * contact
    z = (force[:, None] * ramp)[:, :, None] * u[:, None, :]
    collide = np.zeros_like(contact)
    has = rng.random(n) < p_collide
    tc = rng.integers(20, 90, size=n)
    for i in np.where(has)[0]:
        collide[i, tc[i]:tc[i] + 6] = True
        dirc = rng.normal(size=DZ)
        z[i, tc[i]:tc[i] + 6] += collide_mag * dirc / np.linalg.norm(dirc)
    z = z + rng.normal(size=z.shape) * sigma_n
    return v, z, contact, collide, t_on


def fit_cmp(v, z, nbins=40):
    """손목 영상 → 촉각 잠재 가우시안 예측(μ, 스칼라 σ)을 분위수 구간 평균으로 근사 (논문 Eq. 1의 최적해)."""
    edges = np.quantile(v.ravel(), np.linspace(0, 1, nbins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.clip(np.searchsorted(edges, v.ravel(), side="right") - 1, 0, nbins - 1)
    zf = z.reshape(-1, DZ)
    mu = np.stack([zf[b == k].mean(0) for k in range(nbins)])
    var = np.array([((zf[b == k] - mu[k]) ** 2).sum(1).mean() / DZ for k in range(nbins)])
    return edges, mu, np.sqrt(var)


def apply_cmp(cmp, v, z):
    edges, mu, sig = cmp
    b = np.clip(np.searchsorted(edges, v.ravel(), side="right") - 1, 0, len(mu) - 1).reshape(v.shape)
    m = mu[b]
    s = sig[b]
    err = ((z - m) ** 2).sum(-1) / DZ                          # 원시 예측 오차(차원당)
    return s, err


def logistic_fit(X, y, l2=1e-3, iters=30):
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w


def logistic_apply(w, X):
    X = np.column_stack([np.ones(len(X)), X])
    return 1 / (1 + np.exp(-X @ w))


def gate_features(s, err):
    ls, le = np.log(s), np.log(err + 1e-12)
    return {
        "sigma_gate(paper Eq.3)": ls[..., None],
        "raw_error_gate": le[..., None],
        "precision_weighted_surprise_gate": (le - 2 * ls)[..., None],
        "sigma+error_gate": np.stack([ls, le], -1),
    }


def run_gate(seed, sigma_n, p_train_collide=0.1, n_train=1000, n_cal=400, n_test=600):
    rng = np.random.default_rng(seed)
    v, z, *_ = episodes(rng, n_train, p_train_collide, sigma_n)
    cmp = fit_cmp(v, z)
    # 게이트 보정: 충돌이 절반인 별도 세트에서 "접촉 중(기대 + 예상 밖)" 라벨에 로지스틱 회귀
    vc, zc, cc, kc, _ = episodes(rng, n_cal, 0.5, sigma_n)
    sc, ec = apply_cmp(cmp, vc, zc)
    yc = (cc | kc).ravel().astype(float)
    feats_c = gate_features(sc, ec)
    ws = {k: logistic_fit(f.reshape(-1, f.shape[-1]), yc) for k, f in feats_c.items()}
    vt, zt, ct, kt, ton = episodes(rng, n_test, 0.5, sigma_n)
    st, et = apply_cmp(cmp, vt, zt)
    feats_t = gate_features(st, et)
    t = np.arange(T)
    pre = (t[None, :] >= ton[:, None] - 10) & (t[None, :] < ton[:, None])
    free = ~(ct | kt | pre) & (t[None, :] < T_OFF)
    res = {}
    gates = {}
    for k, f in feats_t.items():
        g = logistic_apply(ws[k], f.reshape(-1, f.shape[-1])).reshape(n_test, T)
        gates[k] = g
        op = g > 0.5
        lag = []
        for i in range(n_test):
            after = np.where(op[i, ton[i]:T_OFF])[0]
            lag.append(after[0] if len(after) else T_OFF - ton[i])
        res[k] = {
            "expected_contact_recall": float(op[ct].mean()),
            "unexpected_collision_recall": float(op[kt].mean()),
            "free_space_false_open": float(op[free].mean()),
            "pre_contact_mean_g(10 steps before onset)": float(g[pre].mean()),
            "free_space_mean_g": float(g[free].mean()),
            "onset_lag_steps_median": float(np.median(lag)),
        }
    return res, gates, (vt, st, et, ct, kt, ton)


def part_a():
    out = {}
    res, gates, (vt, st, et, ct, kt, ton) = run_gate(seed=0, sigma_n=0.05)
    out["main(sigma_n=0.05, collide_mag=1.0)"] = res
    # 타임라인: 충돌 없는 에피소드를 접촉 시점 기준으로 정렬해 평균 (논문 Fig. 5와 비교)
    no_col = np.where(~kt.any(1))[0]
    rel = np.arange(-110, 70)
    tl = {}
    for k, g in gates.items():
        rows = []
        for i in no_col:
            idx = ton[i] + rel
            ok = (idx >= 0) & (idx < T)
            row = np.full(len(rel), np.nan)
            row[ok] = g[i, idx[ok]]
            rows.append(row)
        tl[k] = np.round(np.nanmean(rows, 0), 4).tolist()
    out["timeline_aligned_to_contact"] = {"rel_steps": rel.tolist(), "mean_g": tl}
    # 충돌이 있는 예시 에피소드 하나
    ex = int(np.where(kt.any(1))[0][0])
    out["example_episode"] = {
        "t_on": int(ton[ex]), "collide_steps": np.where(kt[ex])[0].tolist(),
        "sigma": np.round(st[ex], 4).tolist(), "raw_error": np.round(et[ex], 4).tolist(),
        "g": {k: np.round(g[ex], 4).tolist() for k, g in gates.items()},
    }
    # 센서 잡음 스윕 (원시 오차 게이트의 오경보가 늘어나는지)
    sweep = {}
    for sn in [0.03, 0.1, 0.2, 0.3]:
        r, _, _ = run_gate(seed=1, sigma_n=sn)
        sweep[str(sn)] = {k: {m: round(x, 3) for m, x in v.items()} for k, v in r.items()}
    out["sensor_noise_sweep"] = sweep
    # 훈련 데이터의 충돌 비율을 0.5로 올려도 σ 게이트가 충돌을 못 보는지
    r, _, _ = run_gate(seed=2, sigma_n=0.05, p_train_collide=0.5)
    out["train_collide_0.5"] = {k: {m: round(x, 3) for m, x in v.items()} for k, v in r.items()}
    return out


# ---------------------------------------------------------------------------
# B. 잔차 vs 원시 촉각 (선형-가우시안, ridge, 모달리티별 표준화)
# ---------------------------------------------------------------------------
DZB = 8
LAMBDAS = np.logspace(-4, 3, 15)


def make_world(rng, rho2, dv):
    C = rng.normal(size=(DZB, dv)) / np.sqrt(dv) * np.sqrt(rho2)   # 예측 가능 분산(차원당) ≈ ρ²
    s_u = np.sqrt(1 - rho2)                                       # 예측 불가능 분산 = 1 − ρ²
    w_v = rng.normal(size=dv) / np.sqrt(dv)
    w_u = rng.normal(size=DZB) / np.sqrt(DZB)
    return C, s_u, w_v, w_u


def sample(rng, world, n, sigma_y=0.3):
    C, s_u, w_v, w_u = world
    v = rng.normal(size=(n, C.shape[1]))
    u = rng.normal(size=(n, DZB))                          # 표준화된 촉각 고유 정보
    z = v @ C.T + s_u * u
    y = v @ w_v + u @ w_u + sigma_y * rng.normal(size=n)   # 행동은 시각 부분과 촉각 고유 부분에 같은 크기로 의존
    return v, z, y


def ridge_path(Xtr, ytr, Xevals, lams):
    """열마다 표준화한 ridge를 λ 여러 개로 한 번에 푼다. 반환: [λ마다 [평가 세트마다 예측]]"""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    A = (Xtr - mu) / sd
    Bs = [(X - mu) / sd for X in Xevals]
    ym = ytr.mean()
    G, b = A.T @ A, A.T @ (ytr - ym)
    out = []
    for lam in lams:
        Gl = G + lam * len(A) * np.eye(A.shape[1])
        w = np.linalg.solve(Gl, b) if lam > 0 else np.linalg.lstsq(Gl, b, rcond=None)[0]
        out.append([B @ w + ym for B in Bs])
    return out


def fit_cmp_linear(v, z, lam=1e-2):
    """CMP(선형, 약한 ridge): z ≈ Ĉ v"""
    return np.linalg.solve(v.T @ v + lam * len(v) * np.eye(v.shape[1]), v.T @ z).T


def run_b_cell(seed, rho2, dv, n_train, cmp_mode, n_val=500, n_test=4000, sigma_y=0.3):
    rng = np.random.default_rng(seed)
    world = make_world(rng, rho2, dv)
    vtr, ztr, ytr = sample(rng, world, n_train, sigma_y)
    if cmp_mode == "in-sample":                            # 정책 데이터와 같은 데이터로 CMP 학습 (논문 설정에 가까움)
        Chat = fit_cmp_linear(vtr, ztr)
    else:
        n_cmp = int(cmp_mode.split("=")[1])
        vc, zc, _ = sample(rng, world, n_cmp, sigma_y)
        Chat = fit_cmp_linear(vc, zc)
    vva, zva, yva = sample(rng, world, n_val, sigma_y)
    vte, zte, yte = sample(rng, world, n_test, sigma_y)
    feats = {
        "raw": lambda v, z: np.hstack([v, z]),
        "residual": lambda v, z: np.hstack([v, z - v @ Chat.T]),
    }
    out, ols = {}, {}
    lams = list(LAMBDAS) + ([0.0] if n_train > dv + DZB else [])
    for name, f in feats.items():
        path = ridge_path(f(vtr, ztr), ytr, [f(vva, zva), f(vte, zte)], lams)
        val = [np.mean((pv - yva) ** 2) for pv, _ in path[:len(LAMBDAS)]]
        j = int(np.argmin(val))
        out[name] = (float(np.mean((path[j][1] - yte) ** 2)), float(LAMBDAS[j]))
        if len(lams) > len(LAMBDAS):
            ols[name] = path[-1][1]
    # OLS 동치 확인: 표준화만 하고 λ = 0 (특징 수보다 데이터가 많을 때만 의미가 있다)
    if ols:
        out["ols_max_abs_diff"] = float(np.max(np.abs(ols["raw"] - ols["residual"])))
    # 잔차의 학습/시험 분산 비 (in-sample CMP면 학습 쪽 잔차가 작아진다)
    rtr = ztr - vtr @ Chat.T
    rte = zte - vte @ Chat.T
    out["resid_var_train_over_test"] = float(rtr.var() / rte.var())
    return out


def part_b():
    out = {"setup": {"d_z": DZB, "sigma_y": 0.3, "irreducible_mse": 0.09, "seeds": 20,
                     "note": "y = w_v·v + w_u·u + 잡음, z = Cv + sqrt(1−ρ²)u. 특징 열마다 표준화, λ는 검증 세트(500)로 선택, 시험 4000."}}
    grid = {}
    for dv in [32, 256]:
        for n_train in [150, 1500]:
            for cmp_mode in ["sep=5000", "sep=300", "in-sample"]:
                for rho2 in [0.0, 0.5, 0.9, 0.99]:
                    raw, res, lam_raw, lam_res, ols, vr = [], [], [], [], [], []
                    for seed in range(20):
                        o = run_b_cell(1000 + seed, rho2, dv, n_train, cmp_mode)
                        raw.append(o["raw"][0]); res.append(o["residual"][0])
                        lam_raw.append(o["raw"][1]); lam_res.append(o["residual"][1])
                        ols.append(o.get("ols_max_abs_diff", np.nan)); vr.append(o["resid_var_train_over_test"])
                    raw, res = np.array(raw), np.array(res)
                    d = raw - res
                    grid[f"dv={dv}|N={n_train}|cmp={cmp_mode}|rho2={rho2}"] = {
                        "mse_raw": round(float(raw.mean()), 4), "mse_residual": round(float(res.mean()), 4),
                        "raw_minus_residual_mean": round(float(d.mean()), 4),
                        "raw_minus_residual_se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 4),
                        "relative_gain_pct": round(float(100 * d.mean() / raw.mean()), 2),
                        "residual_better_in_seeds": int((d > 0).sum()),
                        "median_lambda_raw": float(np.median(lam_raw)), "median_lambda_residual": float(np.median(lam_res)),
                        "ols_max_abs_diff_max": (None if np.all(np.isnan(ols)) else float(np.nanmax(ols))),
                        "resid_var_train_over_test": round(float(np.mean(vr)), 3),
                    }
    out["grid"] = grid
    return out


def main():
    t0 = time.time()
    out = {"A_gate": part_a()}
    t1 = time.time()
    out["B_residual_vs_raw"] = part_b()
    out["runtime_s"] = {"A": round(t1 - t0, 1), "B": round(time.time() - t1, 1)}
    path = ROOT / "analysis" / "gate_toy_results.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    a = out["A_gate"]
    for k, v in a["main(sigma_n=0.05, collide_mag=1.0)"].items():
        print("A", k, {m: round(x, 3) for m, x in v.items()})
    for sn, r in a["sensor_noise_sweep"].items():
        print("A sweep", sn, {k: (v["expected_contact_recall"], v["unexpected_collision_recall"], v["free_space_false_open"]) for k, v in r.items()})
    print("A train0.5", {k: (v["unexpected_collision_recall"]) for k, v in a["train_collide_0.5"].items()})
    for k, v in out["B_residual_vs_raw"]["grid"].items():
        print("B", k, v["mse_raw"], v["mse_residual"], f'{v["relative_gain_pct"]}%', "±", v["raw_minus_residual_se"],
              v["residual_better_in_seeds"], f'{v["median_lambda_raw"]:.0e}', f'{v["median_lambda_residual"]:.0e}', v["ols_max_abs_diff_max"], v["resid_var_train_over_test"])
    print("runtime", out["runtime_s"], "wrote", path)


if __name__ == "__main__":
    main()
