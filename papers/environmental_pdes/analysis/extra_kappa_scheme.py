"""추가 검증 1: 수치 방식(1차 풍상 대 2차 Lax–Wendroff)에 따른 κ_eff와 필터 오차 (CPU, numpy 1스레드, 약 1분).

질문: 같은 κ_eff로 맞춘 지도를 넣으면 필터 오차가 수치 방식에 따라 달라지는가?
방법:
  1) κ_eff 측정: 1차원 가우시안 펄스를 x 방향 이류(u = 0.25)로 보내고 분산 증가율 d(σ²)/dt = 2κ_eff를 잰다.
     (κ_add = 0일 때의 수치 확산도 같은 식으로 잰다. 측정은 정방향 이류–확산 전체에 대해 한다.)
  2) 참값: pde_toy.solve(80, 0.2, "asym")와 같은 1차 풍상 80×80 장(κ = 0.2). 목표 κ_eff*를 이 장에서 잰다.
  3) 지도(40×40) 네 종:
       U_nom : 1차 풍상, κ = 0.2 (pde_toy의 match 설정과 같음, 기준선)
       U_cal : 1차 풍상, κ_add를 κ_eff = κ_eff* 가 되도록 반복 보정
       L_nom : Lax–Wendroff(2차, 이류를 분할), κ = 0.2
       L_cal : Lax–Wendroff, κ_add를 κ_eff = κ_eff* 가 되도록 보정
  4) 필터는 pde_toy.run_one 그대로(같은 N = 500, 같은 시드 30개, 같은 경로). U_nom은 pde_toy의 asym_match와 같은 숫자가 나와야 한다(검산).
가정·한계: Lax–Wendroff는 분할(x 다음 y) 1차 시간 적분이고, 분산(위상 오차)을 만들며 κ는 2차로 0이다.
실행: python3 analysis/extra_kappa_scheme.py -> analysis/extra_kappa_scheme_results.json
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pde_toy as P  # noqa: E402  (기존 toy의 필터·경로·참값 함수를 그대로 재사용)

L, T = P.L, P.T


def lw_axis(f, c, axis):
    """Lax–Wendroff 1차원 한 단계. c = u·dt/dx (Courant 수), 경계는 0으로 채움(유입 0)."""
    pad = [(0, 0), (0, 0)]
    pad[axis] = (1, 1)
    p = np.pad(f, pad, mode="constant")
    if axis == 0:
        fp, fm = p[2:, :], p[:-2, :]
    else:
        fp, fm = p[:, 2:], p[:, :-2]
    return f - 0.5 * c * (fp - fm) + 0.5 * c * c * (fp - 2 * f + fm)


def step_scheme(f, u, v, dt, dx, kappa, s, scheme):
    if scheme == "upwind":
        # 기존 pde_toy와 같은 식: 풍상 이류 + 확산·원천은 같은 시점의 장에서 계산(u, v ≥ 0 가정)
        adv = u * (f - np.pad(f, ((1, 0), (0, 0)))[:-1, :]) / dx + v * (f - np.pad(f, ((0, 0), (1, 0)))[:, :-1]) / dx
        return f - dt * adv + dt * (kappa * P.lap(f, dx) + s)
    # Lax–Wendroff: x 이류, y 이류를 분할한 뒤 확산·원천을 더한다(분할 1차 시간 적분)
    g = lw_axis(f, u * dt / dx, 0) if u != 0 else f
    g = lw_axis(g, v * dt / dx, 1) if v != 0 else g
    return g + dt * (kappa * P.lap(g, dx) + s)


def solve_scheme(n, kappa, scen, scheme, phi0=None):
    dx, phi, (u, v), s = P.scenario(n, scen)
    if phi0 is not None:
        phi = phi0
    dt_max = 0.5 * min(dx**2 / (4 * kappa), dx / max(abs(u) + abs(v), 1e-12))
    spsec = int(np.ceil(1.0 / dt_max))
    dt = 1.0 / spsec
    snaps = [phi.copy()]
    for _ in range(T):
        for _ in range(spsec):
            phi = step_scheme(phi, u, v, dt, dx, kappa, s, scheme)
        snaps.append(phi.copy())
    return np.array(snaps)


def kappa_eff(dx, kappa, scheme, t0=5, t1=20, Lm=60.0):
    """κ_eff 측정: 큰 영역(Lm = 60 m)에서 1차원 펄스(y 균일, 강제 0, u = 0.25, x = 10 m 시작)의
    분산 증가율 d(σ²)/dt를 5–20 s에서 재고 κ_eff = (d σ²/dt)/2로 한다.
    영역을 크게 잡아 경계(zero-flux 벽과 유출)가 분산 증가에 들어가지 않게 했다."""
    n = int(round(Lm / dx))
    xs = (np.arange(n) + 0.5) * dx
    X, _ = np.meshgrid(xs, xs, indexing="ij")
    phi = np.exp(-((X - 10.0) ** 2) / (2 * P.W0**2)) * np.ones_like(X)
    u, v = 0.25, 0.0
    dt_max = 0.5 * min(dx**2 / (4 * max(kappa, 1e-12)), dx / abs(u))
    spsec = int(np.ceil(1.0 / dt_max))
    dt = 1.0 / spsec
    var = {}
    for sec in range(t1 + 1):
        if sec in (t0, t1):
            m = phi.sum(axis=1)
            mu = (m * xs).sum() / m.sum()
            var[sec] = float((m * (xs - mu) ** 2).sum() / m.sum())
        if sec == t1:
            break
        for _ in range(spsec):
            phi = step_scheme(phi, u, v, dt, dx, kappa, np.zeros_like(phi), scheme)
    return (var[t1] - var[t0]) / (2 * (t1 - t0))


def calibrate(dx, scheme, target, iters=4):
    """κ_add를 κ_eff = target이 되도록 반복 보정한다(κ_eff ≈ κ_add + κ_num이므로 차이만큼 이동)."""
    k_add = target
    hist = []
    for _ in range(iters):
        ke = kappa_eff(dx, k_add, scheme)
        hist.append((k_add, ke))
        k_add = max(k_add + (target - ke), 1e-4)
    return k_add, hist


def main():
    t_all = time.perf_counter()
    KAP = P.KAPPA
    pdisp, d = P.path_disp()
    truth = P.solve(P.N_TRUE, KAP, "asym")[0]

    dx_true, dx_map = P.L / P.N_TRUE, P.L / P.N_MAP
    ke_truth = kappa_eff(dx_true, KAP, "upwind")
    target = ke_truth
    out = {"setup": dict(kappa_nominal=KAP, u=(0.25, 0.10), kappa_eff_measure="1D pulse, x-advection, d(var)/dt/2 over 5-20 s",
                         target_kappa_eff_from="upwind 80x80 truth with kappa=0.2", N_RUNS=P.N_RUNS, N_PART=P.N_PART),
           "kappa_eff": {"truth_upwind80_nominal": ke_truth}}

    maps = {}
    for name, scheme, cal in [("U_nom", "upwind", False), ("U_cal", "upwind", True),
                              ("L_nom", "lw", False), ("L_cal", "lw", True)]:
        if cal:
            k_add, hist = calibrate(dx_map, scheme, target)
        else:
            k_add, hist = KAP, None
        ke = kappa_eff(dx_map, k_add, scheme)
        maps[name] = solve_scheme(P.N_MAP, k_add, "asym", scheme)
        out["kappa_eff"][name] = {"kappa_add": float(k_add), "kappa_eff_measured": float(ke),
                                  "calibration_history": hist}
    out["kappa_eff"]["ratio_nominal_vs_truth"] = {
        "U_nom": out["kappa_eff"]["U_nom"]["kappa_eff_measured"] / target,
        "L_nom": out["kappa_eff"]["L_nom"]["kappa_eff_measured"] / target,
    }

    rows = {}
    for name in ["U_nom", "U_cal", "L_nom", "L_cal"]:
        rm, fe = [], []
        for s in range(P.N_RUNS):
            est, p_true = P.run_one(s, truth, maps[name], d, pdisp)
            r, f = P.metrics(est, p_true)
            rm.append(r)
            fe.append(f)
        rm, fe = np.array(rm), np.array(fe)
        rows[name] = dict(rmse_mean=float(rm.mean()), rmse_sd=float(rm.std(ddof=1)), rmse_median=float(np.median(rm)),
                          final_mean=float(fe.mean()), raw_rmse=rm.round(4).tolist())
    from scipy import stats
    for name in ["U_cal", "L_nom", "L_cal"]:
        rows[name]["welch_p_vs_U_nom"] = float(stats.ttest_ind(rows[name]["raw_rmse"], rows["U_nom"]["raw_rmse"],
                                                              equal_var=False).pvalue)
    out["filter"] = rows
    out["check_U_nom_equals_pde_toy_asym_match"] = "pde_toy_results.json asym_match rmse_mean 0.786 (SD 0.278) 와 같아야 함"
    out["total_wall_s"] = round(time.perf_counter() - t_all, 2)
    (HERE / "extra_kappa_scheme_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("kappa_eff truth (upwind 80, nominal 0.2):", round(ke_truth, 4))
    for k in ["U_nom", "U_cal", "L_nom", "L_cal"]:
        r = rows[k]
        print(f"{k:6s} kappa_add={out['kappa_eff'][k]['kappa_add']:.4f} kappa_eff={out['kappa_eff'][k]['kappa_eff_measured']:.4f} "
              f"RMSE {r['rmse_mean']:.3f}±{r['rmse_sd']:.3f} median {r['rmse_median']:.3f} final {r['final_mean']:.3f}"
              + (f" p_vs_U_nom {r['welch_p_vs_U_nom']:.3g}" if 'welch_p_vs_U_nom' in r else ""))
    print("wall", out["total_wall_s"], "s")


if __name__ == "__main__":
    main()
