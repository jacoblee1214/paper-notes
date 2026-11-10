"""PDE 장 기반 위치추정 toy (CPU, numpy 1스레드, 약 1분 이내).

모형: 2차원 이류–확산 방정식  ∂φ/∂t + v·∇φ = κ∇²φ + s  (논문 식 4와 같은 형태)
  - 유한차분: 확산은 5점 라플라스(명시적, zero-flux 경계), 이류는 1차 풍상(유입 경계값 0)
  - 참값 장: 80×80 격자, κ = 0.2 m²/s (솔버 격자를 다르게 해서 '같은 솔버로 생성·추정' 문제를 피함)
  - 필터용 지도: 40×40 격자, κ̂ (불일치 시험)
로봇: 주기 경로(타원), 이동 잡음이 있는 주행기록계(논문 식 8과 같은 가법 잡음), 1 Hz 점 센서.
필터: 입자 필터(N = 500), 측정 우도 N(y; φ̂(p, t), σ_s²), 쌍선형 보간, 체계적 재표본추출.
비교 설정:
  match    : 시간 의존 지도, κ̂ = κ
  kmis_*   : κ̂ = 배율 × κ (0.25, 0.5, 2, 4)
  static   : 지도를 t = 0 스냅샷으로 고정 (시간 변화 무시)
  odo      : 센서 없이 주행기록계만 (시작점은 참값을 알려 줌, 유리한 기준선)
  sym_*    : 반경 대칭 장(중심 원천, 이류 0)에서 같은 비교
실행: python3 analysis/pde_toy.py → analysis/pde_toy_results.json
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
L = 20.0          # 영역 한 변 [m]
T = 120           # 길이 [s], 센서 주기 1 s
SIG_O = 0.08      # 이동 잡음 표준편차 [m/스텝]
SIG_S = 0.05      # 점 센서 잡음 표준편차 (장의 단위)
N_PART = 500
N_RUNS = 30
KAPPA = 0.2       # 참값 κ [m²/s]
N_TRUE, N_MAP = 80, 40
W0 = 2.5          # 초기 장의 가우시안 폭 [m]
SIG_INIT = 1.5    # 입자 초기 분포: 알려진 출발점 주변 가우시안 [m] (논문 Fig. 2a의 흩어진 입자)
RNG_BASE = 1000


def lap(f, dx):
    p = np.pad(f, 1, mode="edge")
    return (p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - 4 * f) / dx**2


def upwind(f, u, v, dx):
    # 이류: 유입 경계값 0 (영역 밖에서 들어오는 질량 없음). edge 복제를 쓰면 유입면에서 질량이 생긴다.
    p = np.pad(f, 1, mode="constant")
    dfdx = (f - p[:-2, 1:-1]) / dx if u >= 0 else (p[2:, 1:-1] - f) / dx
    dfdy = (f - p[1:-1, :-2]) / dx if v >= 0 else (p[1:-1, 2:] - f) / dx
    return u * dfdx + v * dfdy


def scenario(n, scen):
    dx = L / n
    xs = (np.arange(n) + 0.5) * dx
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    if scen == "asym":
        phi0 = np.exp(-((X - 6.0) ** 2 + (Y - 12.0) ** 2) / (2 * W0**2))
        vel, src = (0.25, 0.10), (4.0, 6.0)
    else:  # sym: 중심 원천, 초기 장도 중심 대칭, 이류 없음
        phi0 = np.exp(-((X - L / 2) ** 2 + (Y - L / 2) ** 2) / (2 * W0**2))
        vel, src = (0.0, 0.0), (L / 2, L / 2)
    forcing = 0.02 * np.exp(-((X - src[0]) ** 2 + (Y - src[1]) ** 2) / (2 * 0.8**2))
    return dx, phi0, vel, forcing


def solve(n, kappa, scen):
    """0..T초 스냅샷(1 Hz)을 만든다. 반환: (snapshots, dt_used, 스텝 수, 벽시계 시간)."""
    dx, phi, (u, v), s = scenario(n, scen)
    dt_max = 0.5 * min(dx**2 / (4 * kappa), dx / max(abs(u) + abs(v), 1e-12))
    steps_per_s = int(np.ceil(1.0 / dt_max))
    dt = 1.0 / steps_per_s
    snaps = [phi.copy()]
    t0 = time.perf_counter()
    for _ in range(T):
        for _ in range(steps_per_s):
            phi = phi + dt * (-upwind(phi, u, v, dx) + kappa * lap(phi, dx) + s)
        snaps.append(phi.copy())
    wall = time.perf_counter() - t0
    courant = (abs(u) + abs(v)) * dt / dx
    fourier = kappa * dt / dx**2
    return np.array(snaps), dict(dt=dt, steps=steps_per_s * T, courant=courant, fourier=fourier,
                                 wall_s=wall, per_update_ms=1000 * wall / T)


def sample(field, p):
    """쌍선형 보간. field: (n, n) 셀 중심 격자, p: (N, 2) [m]."""
    n = field.shape[0]
    dx = L / n
    gx = np.clip(p[:, 0] / dx - 0.5, 0, n - 1 - 1e-9)
    gy = np.clip(p[:, 1] / dx - 0.5, 0, n - 1 - 1e-9)
    i0, j0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
    i1, j1 = np.minimum(i0 + 1, n - 1), np.minimum(j0 + 1, n - 1)
    ax, ay = gx - i0, gy - j0
    return ((1 - ax) * (1 - ay) * field[i0, j0] + ax * (1 - ay) * field[i1, j0]
            + (1 - ax) * ay * field[i0, j1] + ax * ay * field[i1, j1])


def path_disp(scen_center=(10.0, 10.0), rx=6.0, ry=4.0, omega=0.1):
    t = np.arange(T + 1)
    c = np.array(scen_center)
    pts = np.stack([c[0] + rx * np.cos(omega * t), c[1] + ry * np.sin(omega * t)], axis=1)
    return pts, np.diff(pts, axis=0)


def run_one(seed, truth, maps, d, p_path, sig_s=SIG_S):
    # maps: None = 주행기록계만, "nosensor" = 입자는 움직이되 측정으로 가중치를 갱신하지 않음, 그 밖 = 장 지도
    rng = np.random.default_rng(RNG_BASE + seed)
    # 참 궤적: 명령 변위 + 이동 잡음 (논문 식 8과 같은 가법 잡음)
    p_true = np.empty((T + 1, 2))
    p_true[0] = p_path[0]
    for t in range(1, T + 1):
        p_true[t] = np.clip(p_true[t - 1] + d[t - 1] + rng.normal(0, SIG_O, 2), 0, L)
    y = np.array([sample(truth[t], p_true[t:t + 1])[0] for t in range(T + 1)])
    y = y + rng.normal(0, sig_s, T + 1)

    if maps is None:  # 주행기록계만
        est = np.empty((T + 1, 2))
        est[0] = p_true[0]
        for t in range(1, T + 1):
            est[t] = est[t - 1] + d[t - 1]
        return est, p_true

    p = np.clip(p_true[0] + rng.normal(0, SIG_INIT, (N_PART, 2)), 0, L)
    w = np.full(N_PART, 1.0 / N_PART)
    est = np.empty((T + 1, 2))
    est[0] = p.mean(axis=0)
    for t in range(1, T + 1):
        p = np.clip(p + d[t - 1] + rng.normal(0, SIG_O, (N_PART, 2)), 0, L)
        if not isinstance(maps, str):
            pred = sample(maps[t], p)
            logw = np.log(w + 1e-300) - 0.5 * ((y[t] - pred) / sig_s) ** 2
            logw -= logw.max()
            w = np.exp(logw)
            w /= w.sum()
        est[t] = w @ p
        if not isinstance(maps, str) and 1.0 / np.sum(w**2) < N_PART / 2:
            u0 = (rng.random() + np.arange(N_PART)) / N_PART
            idx = np.minimum(np.searchsorted(np.cumsum(w), u0), N_PART - 1)
            p = p[idx]
            w = np.full(N_PART, 1.0 / N_PART)
    return est, p_true


def metrics(est, p_true):
    e = np.linalg.norm(est[1:] - p_true[1:], axis=1)
    return float(np.sqrt(np.mean(e**2))), float(e[-1])


def main():
    t_all = time.perf_counter()
    p_path, d = path_disp()

    solves = {}
    truth = {}
    truth_info = {}
    for scen in ("asym", "sym"):
        truth[scen], truth_info[scen] = solve(N_TRUE, KAPPA, scen)

    # 지도: asym은 κ 배율 5개, sym은 일치(1배)만
    cfgs = {}
    for f in (0.25, 0.5, 1.0, 2.0, 4.0):
        m, info = solve(N_MAP, KAPPA * f, "asym")
        solves[f"asym_kfac{f}"] = info
        cfgs[f"asym_kfac{f}"] = m
    m_sym, info = solve(N_MAP, KAPPA, "sym")
    solves["sym_kfac1.0"] = info
    cfgs["sym_kfac1.0"] = m_sym

    # 참값(80×80)과 필터 지도(40×40)의 격자 차이: 같은 κ에서 영역 전체 RMS
    xs = (np.arange(N_MAP) + 0.5) * (L / N_MAP)
    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
    rms_grid = []
    for t in (0, 60, 120):
        a = sample(truth["asym"][t], pts)
        b = sample(cfgs["asym_kfac1.0"][t], pts)
        rms_grid.append(float(np.sqrt(np.mean((a - b) ** 2))))
    static_map = np.repeat(cfgs["asym_kfac1.0"][0:1], T + 1, axis=0)
    sym_static = np.repeat(cfgs["sym_kfac1.0"][0:1], T + 1, axis=0)

    configs = {
        "asym_match": (cfgs["asym_kfac1.0"], "asym"),
        "asym_kmis_0.25x": (cfgs["asym_kfac0.25"], "asym"),
        "asym_kmis_0.5x": (cfgs["asym_kfac0.5"], "asym"),
        "asym_kmis_2x": (cfgs["asym_kfac2.0"], "asym"),
        "asym_kmis_4x": (cfgs["asym_kfac4.0"], "asym"),
        "asym_static_t0": (static_map, "asym"),
        "sym_match": (cfgs["sym_kfac1.0"], "sym"),
        "sym_static_t0": (sym_static, "sym"),
        "odometry_only": (None, "asym"),
        "asym_pf_nosensor": ("nosensor", "asym"),
        "sym_pf_nosensor": ("nosensor", "sym"),
        "asym_match_sigs0.02": (cfgs["asym_kfac1.0"], "asym", 0.02),
        "asym_match_sigs0.10": (cfgs["asym_kfac1.0"], "asym", 0.10),
    }
    results = {}
    raw = {}
    for name, cfg in configs.items():
        maps, scen = cfg[0], cfg[1]
        sig_s = cfg[2] if len(cfg) > 2 else SIG_S
        rm, fe = [], []
        for s in range(N_RUNS):
            tr = truth[scen]
            est, p_true = run_one(s, tr, maps, d, p_path, sig_s=sig_s)
            r, f = metrics(est, p_true)
            rm.append(r)
            fe.append(f)
        rm, fe = np.array(rm), np.array(fe)
        raw[name] = rm
        results[name] = dict(
            rmse_mean=float(rm.mean()), rmse_sd=float(rm.std(ddof=1)), rmse_median=float(np.median(rm)),
            final_mean=float(fe.mean()), final_sd=float(fe.std(ddof=1)),
            frac_rmse_gt_2m=float(np.mean(rm > 2.0)),
        )
    # 일치 지도와의 차이 검정 (Welch)
    for name in configs:
        if name in ("asym_match", "sym_match"):
            continue
        a = raw["sym_match"] if name.startswith("sym") else raw["asym_match"]
        t_stat, p_val = stats.ttest_ind(raw[name], a, equal_var=False)
        results[name]["welch_vs_match_p"] = float(p_val)

    diag = {}
    for scen, fld in (("asym", truth["asym"]), ("sym", truth["sym"])):
        for t in (0, 60, 120):
            gy, gx = np.gradient(fld[t], L / N_TRUE)
            diag[f"{scen}_t{t}"] = dict(max=float(fld[t].max()), spatial_std=float(fld[t].std()),
                                        grad_rms=float(np.sqrt(np.mean(gx**2 + gy**2))))
    out = dict(
        field_diagnostics=diag,
        setup=dict(L=L, T=T, sig_odo=SIG_O, sig_sensor=SIG_S, n_particles=N_PART, n_runs=N_RUNS,
                   kappa_true=KAPPA, grid_true=N_TRUE, grid_map=N_MAP,
                   bc="확산: zero-flux(가장자리 복제), 이류: 유입 0, 풍상 1차", estimator="weighted mean"),
        solve_info=dict(truth_asym=truth_info["asym"], truth_sym=truth_info["sym"], maps=solves),
        grid_discretization_rms_asym=dict(zip(["t0", "t60", "t120"], rms_grid)),
        results=results,
        total_wall_s=round(time.perf_counter() - t_all, 2),
    )
    (HERE / "pde_toy_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for name, r in results.items():
        print(f"{name:20s} RMSE {r['rmse_mean']:.3f}±{r['rmse_sd']:.3f} (median {r['rmse_median']:.3f})"
              f"  final {r['final_mean']:.3f}  >2m {r['frac_rmse_gt_2m']:.2f}"
              f"  p_vs_match {r.get('welch_vs_match_p', float('nan')):.3g}")
    print("solve (truth asym):", truth_info["asym"])
    print("solve (map asym):", solves["asym_kfac1.0"])
    print("grid RMS diff (80 vs 40):", out["grid_discretization_rms_asym"])
    print("field diag:", json.dumps(diag, indent=1))
    print("total wall", out["total_wall_s"], "s")


if __name__ == "__main__":
    main()
