"""추가 검증 (DexTouch-WM R2): 시뮬레이터 점수 보정이 MMRV를 바꾸는가 (정책 수 3과 10에서).

논문 식 (7): MMRV_k = (1/N) sum_i max_j |R_ik - R_jk| v_ijk,  v_ijk = 1 if (R_ik - R_jk)(S_ik - S_jk) < 0.
  (03 §6.3의 N=3 닫힌 형식 2d/3과 일치한다. 논문 본문 인용과 같은 식.)

가정 모델 (논문 실험이 아님):
  - 과제 4개. 정책 N개(3 또는 10)의 실제 점수 R_i ~ U(0.05, 0.95).
  - 시뮬 점수 S_i = clip(R_i + mu + 0.05 * eta_i, 0, 1), mu in {0.12, 0.15, 0.18}(논문 12셀 평균 편향 범위),
    eta_i ~ N(0,1). 편향은 정책마다 다르다(0.05 sd는 가정).
  - 보정 자료: 실제 점수를 아는 독립 정책 k개(k = 3 또는 10)를 같은 과제에서 따로 뽑아 보정 계수를 맞춘다.
    평가 정책 N개에 보정을 적용하고 MMRV를 잰다.
  - 보정 방법 두 가지:
      (a) shift: S' = S - mean(S_cal - R_cal)   (단조, 전 정책 동일 이동)
      (b) linear: S' = a + b S, (a, b)는 보정 자료에 최소제곱으로 맞춘다. b < 0이면 순위가 뒤집힌다.
  - 반복: 과제 4 x 반복 2000회 (시드 고정).

결과 해석의 핵심: 전 정책에 같은 단조 함수를 적용하면 v_ijk는 바뀌지 않으므로 MMRV는 정확히 0만큼 변한다.
MMRV가 바뀌려면 보정이 순서를 바꿔야 한다 (linear에서 b < 0인 경우).

실행: python3 analysis/extra_sim_bias_mmrv.py -> analysis/extra_sim_bias_mmrv_results.json
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "extra_sim_bias_mmrv_results.json"
MUS = [0.12, 0.15, 0.18]
ETA_SD = 0.05
TASKS = 4
REPS = 2000
N_EVAL = [3, 10]
K_CAL = [3, 10]


def mmrv(R, S):
    N = len(R)
    dR = np.abs(R[:, None] - R[None, :])
    rev = ((R[:, None] - R[None, :]) * (S[:, None] - S[None, :])) < 0
    per_i = np.max(np.where(rev, dR, 0.0), axis=1)
    return float(per_i.mean())


def rank_corr(R, S):
    if np.std(S) < 1e-12 or np.std(R) < 1e-12:
        return float("nan")
    return float(np.corrcoef(R, S)[0, 1])


def simulate(rng, mu, n_eval, k_cal):
    R = rng.uniform(0.05, 0.95, n_eval)
    S = np.clip(R + mu + ETA_SD * rng.standard_normal(n_eval), 0, 1)
    Rc = rng.uniform(0.05, 0.95, k_cal)
    Sc = np.clip(Rc + mu + ETA_SD * rng.standard_normal(k_cal), 0, 1)
    # shift
    shift = float(np.mean(Sc - Rc))
    S_shift = S - shift
    # linear
    if k_cal >= 2 and np.var(Sc) > 0:
        b = float(np.cov(Sc, Rc, ddof=0)[0, 1] / np.var(Sc))
        a = float(Rc.mean() - b * Sc.mean())
    else:
        b, a = 1.0, 0.0
    S_lin = a + b * S
    m_raw = mmrv(R, S)
    return {
        "mmrv_raw": m_raw,
        "mmrv_shift": mmrv(R, S_shift),
        "mmrv_linear": mmrv(R, S_lin),
        "corr_raw": rank_corr(R, S),
        "corr_shift": rank_corr(R, S_shift),
        "corr_linear": rank_corr(R, S_lin),
        "mae_raw": float(np.mean(np.abs(S - R))),
        "mae_shift": float(np.mean(np.abs(S_shift - R))),
        "mae_linear": float(np.mean(np.abs(S_lin - R))),
        "linear_slope_negative": b < 0,
    }


def main():
    out = {"settings": {"mus": MUS, "eta_sd": ETA_SD, "tasks_per_rep": TASKS, "reps": REPS,
                        "eval_policies": N_EVAL, "calibration_policies": K_CAL,
                        "note": "가정 모델. 편향의 정책별 변동 0.05는 가정이다. MMRV 식은 논문 식 (7)"},
           "cells": {}}
    for n_eval in N_EVAL:
        for k_cal in K_CAL:
            rng = np.random.default_rng(1000 * n_eval + k_cal)
            per_mu = {}
            for mu in MUS:
                rows = [simulate(rng, mu, n_eval, k_cal) for _ in range(TASKS * REPS)]
                arr = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
                d_shift = arr["mmrv_shift"] - arr["mmrv_raw"]
                d_lin = arr["mmrv_linear"] - arr["mmrv_raw"]
                per_mu[f"mu={mu:g}"] = {
                    "mean_MMRV_raw": round(float(arr["mmrv_raw"].mean()), 4),
                    "mean_MMRV_shift": round(float(arr["mmrv_shift"].mean()), 4),
                    "mean_MMRV_linear": round(float(arr["mmrv_linear"].mean()), 4),
                    "max_abs_change_shift": float(np.max(np.abs(d_shift))),
                    "share_tasks_MMRV_changed_linear": round(float(np.mean(np.abs(d_lin) > 1e-12)), 4),
                    "share_tasks_MMRV_worse_linear": round(float(np.mean(d_lin > 1e-12)), 4),
                    "share_linear_slope_negative": round(float(np.mean(arr["linear_slope_negative"])), 4),
                    "mean_MAE_raw": round(float(arr["mae_raw"].mean()), 4),
                    "mean_MAE_shift": round(float(arr["mae_shift"].mean()), 4),
                    "mean_MAE_linear": round(float(arr["mae_linear"].mean()), 4),
                    "mean_pearson_raw": round(float(np.nanmean(arr["corr_raw"])), 4),
                    "mean_pearson_linear": round(float(np.nanmean(arr["corr_linear"])), 4),
                    "n_nan_pearson": int(np.isnan(arr["corr_raw"]).sum()),
                }
            out["cells"][f"N_eval={n_eval},k_cal={k_cal}"] = per_mu
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in out["cells"].items():
        print(k)
        for mu, r in v.items():
            print("  ", mu, {x: r[x] for x in ("mean_MMRV_raw", "mean_MMRV_shift", "mean_MMRV_linear",
                                              "share_tasks_MMRV_worse_linear", "share_linear_slope_negative",
                                              "mean_MAE_raw", "mean_MAE_shift", "mean_MAE_linear")})


if __name__ == "__main__":
    main()
