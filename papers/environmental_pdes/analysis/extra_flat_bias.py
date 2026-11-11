"""추가 검증 2: 평평한 장에서 편향–장 수준 혼동 (CPU, numpy 1스레드, 수십 초).

질문: 편향 b를 미지수로 넣은 필터에서, 장 진폭 A를 줄이면 위치 오차가 얼마나 빨리 커지는가?
        편향 사전 분포 P_0에 따라 결과가 얼마나 달라지는가?
모형 (pde_toy와 독립인 해석 모형, 가정):
  - 장: φ(p) = A·f(p), f(p) = sin(0.3x) + cos(0.25y) + 0.5 sin(0.2(x+y)). A가 작을수록 장이 평평해진다.
  - 측정: y_t = φ(p_t) + b + ε_t, ε ~ N(0, σ²), σ = 0.05, 1 Hz, 121 점. 참 b ~ N(0, P_true), P_true = 0.25.
  - 주행기록계는 상대 경로를 정확히 안다고 둔다(이동 잡음 없음). 미지수는 출발점 x_0 (2차원)과 b.
  - 경로: pde_toy와 같은 타원 모양(반지름 6 m × 4 m, ω = 0.1 rad/s), 출발점 x_0 = (9.0, 7.5).
필터(해석적 주변화, 격자 근사):
  - x_0를 [6, 14]² 격자(0.05 m)에서 균등 사전 분포로 두고, b를 N(b̂_0 = 0, P_0)로 주변화한다.
    잔차 r_t = y_t − φ(x_0 + Δp_t)일 때 b에 대해 주변화한 이차형식은
      Q = (1/σ²)[Σ r_t² − (P_0/(σ² + nP_0)) (Σ r_t)²]  (n = 121, Sherman–Morrison)
    이고 사후 분포는 exp(−Q/2)에 비례한다. 사후 평균을 추정치로 쓴다.
  - P_0 = 0: 편향을 무시한 필터(편향이 있어도 0으로 가정).
  - P_0 = 0.25: 참 분포와 같은 사전 분포(일치).
  - P_0 = 25: 편향을 거의 무정보로 둠. P_0 = 0.25와 25는 P_0 ≫ σ²/n(≈ 2·10⁻⁵)이면 결과가 같다(포화).
  - P_0 = 10⁻⁴: 편향 사전 분포를 sd 0.01로 좁게 둔 잘못된 설정(참 sd 0.5). 민감도를 보려고 넣었다.
  - oracle: 편향 b를 알고 있다고 두고 빼서 씀(하한 기준).
  - 기준: 사후가 평평하면(정보 없음) 추정치는 사전 중심 (10, 10)이 된다. 그때 오차를 "무정보 기준"으로 표시.
결과 단위: 위치 RMSE(m), 30회 평균 ± SD, 중앙값, 오차 > 1 m 비율. 장 진폭은 "장 변동/σ" = sd_t(A f(p_t))/σ로도 적는다.
실행: python3 analysis/extra_flat_bias.py -> analysis/extra_flat_bias_results.json
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIG = 0.05
T = 120
N = T + 1
X0_TRUE = np.array([9.0, 7.5])
P_TRUE = 0.25
AMPS = [1.0, 0.3, 0.1, 0.05, 0.03, 0.01, 0.003]
P0_LIST = [0.0, 1e-4, 0.25, 25.0]
N_RUNS = 30
GRID = np.arange(6.0, 14.0001, 0.05)
RNG_BASE = 5000


def f(px, py):
    return np.sin(0.3 * px) + np.cos(0.25 * py) + 0.5 * np.sin(0.2 * (px + py))


def rel_path():
    t = np.arange(N)
    w = 0.1
    dx = 6.0 * (np.cos(w * t) - 1.0)
    dy = 4.0 * np.sin(w * t)
    return np.stack([dx, dy], axis=1)  # Δp_t, Δp_0 = 0


def main():
    t_all = time.perf_counter()
    dp = rel_path()
    # 후보 위치별 φ(x0 + Δp_t) (A = 1). A는 곱으로만 들어간다.
    GX, GY = np.meshgrid(GRID, GRID, indexing="ij")
    cand = np.stack([GX.ravel(), GY.ravel()], axis=1)                 # (C, 2)
    P_cand = cand[:, None, :] + dp[None, :, :]                         # (C, N, 2)
    F = f(P_cand[..., 0], P_cand[..., 1])                              # (C, N)
    f_true = f(X0_TRUE[0] + dp[:, 0], X0_TRUE[1] + dp[:, 1])          # (N,)
    sd_path = float(np.std(f_true))

    rng = np.random.default_rng(RNG_BASE)
    # 같은 시드로 참 b와 잡음을 모든 설정에서 공유한다(공통 난수)
    b_true = rng.normal(0, np.sqrt(P_TRUE), N_RUNS)
    eps = rng.normal(0, SIG, (N_RUNS, N))

    prior_center_err = float(np.linalg.norm(np.array([10.0, 10.0]) - X0_TRUE))
    res = {}
    for A in AMPS:
        for P0 in P0_LIST + ["oracle"]:
            errs = []
            for k in range(N_RUNS):
                y = A * f_true + b_true[k] + eps[k]
                if P0 == "oracle":
                    r = (y[None, :] - b_true[k]) - A * F                 # (C, N)
                    logpost = -0.5 * np.sum(r**2, axis=1) / SIG**2
                else:
                    r = y[None, :] - A * F
                    S1 = r.sum(axis=1)
                    S2 = (r**2).sum(axis=1)
                    if P0 == 0.0:
                        Q = S2 / SIG**2
                    else:
                        Q = (S2 - (P0 / (SIG**2 + N * P0)) * S1**2) / SIG**2
                    logpost = -0.5 * Q
                w = np.exp(logpost - logpost.max())
                w /= w.sum()
                est = w @ cand
                errs.append(float(np.linalg.norm(est - X0_TRUE)))
            errs = np.array(errs)
            key = f"amp{A}__P0{P0}"
            res[key] = dict(A=A, P0=P0, rmse_mean=float(np.sqrt(np.mean(errs**2))), err_mean=float(errs.mean()),
                            err_sd=float(errs.std(ddof=1)), err_median=float(np.median(errs)),
                            frac_err_gt_1m=float(np.mean(errs > 1.0)))
    out = dict(
        setup=dict(sigma=SIG, n_time=N, bias_true_prior_var=P_TRUE, x0_true=X0_TRUE.tolist(), grid=[6.0, 14.0, 0.05],
                   n_runs=N_RUNS, amps=AMPS, P0_list=P0_LIST, field="sin(0.3x)+cos(0.25y)+0.5 sin(0.2(x+y))",
                   odometry="exact relative path (no odometry noise)"),
        field_sd_over_path_at_A1=sd_path,
        field_sd_over_sigma_at_A1=sd_path / SIG,
        uninformative_reference_err_m=prior_center_err,
        results=res,
        total_wall_s=None,
    )
    for A in AMPS:
        out.setdefault("ratio_table", {})[f"A={A}"] = round(sd_path * A / SIG, 3)
    out["total_wall_s"] = round(time.perf_counter() - t_all, 2)
    (HERE / "extra_flat_bias_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"field sd along path (A=1) = {sd_path:.3f}, /sigma = {sd_path / SIG:.1f}; uninformative ref = {prior_center_err:.2f} m")
    print(f"{'A':>6} {'A*sd/sig':>9} | " + " | ".join(f"P0={p:<6}" for p in P0_LIST + ['oracle']))
    for A in AMPS:
        cells = []
        for P0 in P0_LIST + ["oracle"]:
            r = res[f"amp{A}__P0{P0}"]
            cells.append(f"{r['rmse_mean']:.3f}±{r['err_sd']:.2f} ({r['frac_err_gt_1m']:.2f})")
        print(f"{A:>6} {sd_path * A / SIG:>9.2f} | " + " | ".join(cells))
    print("wall", out["total_wall_s"], "s")


if __name__ == "__main__":
    main()
