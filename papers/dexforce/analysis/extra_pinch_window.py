"""추가 검증 1: 두 손가락 집기에서 k_f 허용 구간과 접촉점 오차의 영향, 이분 탐색 횟수.

모형 (정상 상태, 가정)
- 재생 힘(손가락 i): F_i = k_p (x_f,i − x_c,i) 에서 x_f,i = x_o,i + k_f f_i, x_c,i = x_o,i + e_i 이므로
  F_i = k_p k_f f_i − k_p e_i.  (e_i = 재생 때 접촉 위치가 시연 관측 위치에서 벗어난 양, 독립 정규분포 N(0, σ_e))
- 과제 창: 두 손가락 힘 모두 [N_lo, N_hi] = [0.5, 5] N  (N_lo: 쥐고 있기, N_hi: 부품 파손 방지; 가정)
- 시연: 시연 힘 f_i = 3 N · exp(σZ − σ²/2), CV = 0.3 (가정). 시연 5개 묶음 대신 시연 힘을 독립 표본으로 M = 20000 개 만든다.
- k_p = 310 N/m (논문 값). 각 시연의 허용 k_f 구간은 두 손가락 제약의 교집합(선형이므로 구간).

측정
(a) σ_e ∈ {0, 0.5, 1, 2} mm에서 허용 구간이 비어 있는 비율, k_f 하한의 중앙값, 하한이 0.0001 미만(0 근처)인 비율.
(b) 시연 집단의 성공률 R(k_f) = P(k_f가 구간 안). R ≥ 0.9가 되는 하한·상한을 (i) 0.0001 간격 격자 훑기와 (ii) 이분 탐색(허용 오차 1e-5)으로 찾고, 평가 횟수와 결과 일치를 비교.
결과: analysis/extra_pinch_window_results.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import time
from pathlib import Path

import numpy as np

OUT_JSON = Path(__file__).with_name("extra_pinch_window_results.json")
KP = 310.0
N_LO, N_HI = 0.5, 5.0
F_MED, CV = 3.0, 0.3
M = 20000
SIGMA_E_MM = [0.0, 0.5, 1.0, 2.0]
TARGET = 0.9
GRID = np.arange(0.0, 0.02 + 1e-12, 0.0001)


def sample(sigma_e_m, rng):
    sig = np.sqrt(np.log(1 + CV * CV))
    f = F_MED * np.exp(sig * rng.standard_normal((M, 2)) - sig * sig / 2)
    e = sigma_e_m * rng.standard_normal((M, 2))
    lo = np.max((N_LO + KP * e) / (KP * f), axis=1)
    hi = np.min((N_HI + KP * e) / (KP * f), axis=1)
    return f, lo, hi


def rate(kf, lo, hi):
    return float(np.mean((kf >= lo) & (kf <= hi)))


def bisect_first(pred, a, b, tol):
    """pred(a)=False, pred(b)=True 인 구간에서 첫 True 지점을 이분 탐색. 평가 횟수 반환."""
    n = 0
    while b - a > tol:
        m = 0.5 * (a + b)
        n += 1
        if pred(m):
            b = m
        else:
            a = m
    return 0.5 * (a + b), n


def main():
    t0 = time.time()
    rng = np.random.default_rng(20261111)
    out = {"setup": {"k_p": KP, "N_lo": N_LO, "N_hi": N_HI, "demo_force_median_N": F_MED, "demo_force_cv": CV,
                     "n_demos": M, "sigma_e_mm_list": SIGMA_E_MM, "target_rate": TARGET,
                     "grid_step_kf": 0.0001, "bisection_tol_kf": 1e-5, "seed": 20261111},
           "by_sigma_e": {}}

    for s_mm in SIGMA_E_MM:
        f, lo, hi = sample(s_mm * 1e-3, rng)
        feas = lo <= hi
        # 집단 성공률 곡선 (격자)
        R = np.array([rate(k, lo, hi) for k in GRID])
        grid_ok = GRID[R >= TARGET]
        grid_lo90 = float(grid_ok.min()) if grid_ok.size else None
        grid_hi90 = float(grid_ok.max()) if grid_ok.size else None
        # 이분 탐색: 하한 = 0에서 R<0.9, 피크 근처 k_f에서 R≥0.9 인 구간으로 가정하고 찾음
        peak = float(GRID[np.argmax(R)])
        bis = {}
        if R.max() >= TARGET:
            lo90, n1 = bisect_first(lambda k: rate(k, lo, hi) >= TARGET, 0.0, peak, 1e-5)
            # 상한: 피크에서 격자 끝(0.02)까지 — 반대 방향 이분 (R ≥ 0.9 가 True에서 False로 바뀌는 지점)
            a, b = peak, 0.02
            n2 = 0
            if rate(b, lo, hi) < TARGET:
                while b - a > 1e-5:
                    m = 0.5 * (a + b)
                    n2 += 1
                    if rate(m, lo, hi) >= TARGET:
                        a = m
                    else:
                        b = m
                hi90 = 0.5 * (a + b)
            else:
                hi90 = 0.02
            bis = {"lower90_kf": round(lo90, 6), "upper90_kf": round(hi90, 6),
                   "evals_lower": n1, "evals_upper": n2,
                   "grid_evals": int(len(GRID)),
                   "lower_abs_diff_vs_grid": None if grid_lo90 is None else round(abs(lo90 - grid_lo90), 6),
                   "upper_abs_diff_vs_grid": None if grid_hi90 is None else round(abs(hi90 - grid_hi90), 6)}
        out["by_sigma_e"][f"sigma_e={s_mm}mm"] = {
            "feasible_fraction_any_kf": round(float(feas.mean()), 4),
            "kf_lo_median": round(float(np.median(lo[feas])), 5) if feas.any() else None,
            "kf_hi_median": round(float(np.median(hi[feas])), 5) if feas.any() else None,
            "fraction_lo_below_1e-4": round(float(np.mean(lo[feas] < 1e-4)), 4) if feas.any() else None,
            "max_rate_over_grid": round(float(R.max()), 3),
            "grid_lower90_kf": grid_lo90, "grid_upper90_kf": grid_hi90,
            "bisection": bis,
        }
        print(s_mm, out["by_sigma_e"][f"sigma_e={s_mm}mm"], flush=True)

    # 시연 힘과 하한의 관계: σ_e = 0이면 하한 = N_lo/(k_p f), 힘이 셀수록 하한이 작다 ("과하게 쥔 시연" 설명)
    f, lo, hi = sample(0.0, np.random.default_rng(1))
    fmax = np.max(f, axis=1)
    q = np.quantile(fmax, [0.25, 0.75])
    out["demo_force_vs_lower_bound_sigma0"] = {
        "lo_median_when_max_force_below_p25": round(float(np.median(lo[fmax <= q[0]])), 5),
        "lo_median_when_max_force_above_p75": round(float(np.median(lo[fmax >= q[1]])), 5),
    }
    out["runtime_s"] = round(time.time() - t0, 2)
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
