"""추가 검증 (DexTouch-WM): 사람 데이터 가중을 로봇 데이터 교차검증으로 고르면 해가 되는 영역을 피하는가.

가정 모델 (논문 실험이 아님, 03 §4의 토이 A를 확장):
  - 로봇 도메인 y = w_r^T x + N(0, 0.5^2), D = 8, 로봇 샘플 N_r = 25.
  - 사람 도메인 라벨 기울기 w_h = w_r + delta * u (u = 1/sqrt(D) 방향), 사람 샘플 N_h = 400.
  - 리타게팅 오차: 사람 입력 x + N(0, s_r^2 I) (라벨은 깨끗한 x 기준).
  - 가중 최소제곱(사람 샘플 가중 lambda), 작은 ridge 1e-6.
  - 선택 절차: 로봇 샘플만으로 5-fold 교차검증 MSE가 가장 낮은 lambda를 고른다 (사람 데이터 시험 없음).
    후보 lambda = {0, 0.1, 0.3, 1.0}. lambda=0은 로봇만 쓰는 기준선.
  - 평가: 로봇 도메인 새 샘플 4000개의 MSE. 반복 200회(시드 0..199).
  - 이득 G = MSE(lambda=0) - MSE(선택 lambda). G < 0이면 해가 된다.
  - 비교: lambda=1 고정(단순 합침), 오라클 lambda(시험 MSE 기준 최선, 선택 불가능한 상한).

실행: python3 analysis/extra_src_weight_cv.py -> analysis/extra_src_weight_cv_results.json
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "extra_src_weight_cv_results.json"
D, SIG = 8, 0.5
N_R, N_H, N_TEST = 25, 400, 4000
LAMBDAS = [0.0, 0.1, 0.3, 1.0]
DELTAS = [0.0, 0.25, 0.5, 1.0]
S_R = [0.0, 0.25, 0.5, 1.0]
REPS = 200
K = 5
U = np.ones(D) / np.sqrt(D)


def wls(X, y, w):
    A = X.T @ (X * w[:, None]) + 1e-6 * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ (w * y))


def one_rep(delta, s_r, seed):
    rng = np.random.default_rng(seed)
    w_r = rng.standard_normal(D)
    w_h = w_r + delta * U
    xr = rng.standard_normal((N_R, D))
    yr = xr @ w_r + SIG * rng.standard_normal(N_R)
    xh_clean = rng.standard_normal((N_H, D))
    xh = xh_clean + s_r * rng.standard_normal((N_H, D))
    yh = xh_clean @ w_h + SIG * rng.standard_normal(N_H)
    xt = rng.standard_normal((N_TEST, D))
    yt = xt @ w_r + SIG * rng.standard_normal(N_TEST)

    folds = np.array_split(rng.permutation(N_R), K)
    cv = {}
    for lam in LAMBDAS:
        errs = []
        for f in folds:
            tr = np.setdiff1d(np.arange(N_R), f)
            X = np.vstack([xr[tr], xh])
            y = np.concatenate([yr[tr], yh])
            w = np.concatenate([np.ones(len(tr)), lam * np.ones(N_H)])
            b = wls(X, y, w)
            errs.append(np.mean((xr[f] @ b - yr[f]) ** 2) * len(f))
        cv[lam] = sum(errs) / N_R

    test = {}
    for lam in LAMBDAS:
        X = np.vstack([xr, xh])
        y = np.concatenate([yr, yh])
        w = np.concatenate([np.ones(N_R), lam * np.ones(N_H)])
        b = wls(X, y, w)
        test[lam] = float(np.mean((xt @ b - yt) ** 2))

    lam_sel = min(LAMBDAS, key=lambda l: cv[l])
    lam_orc = min(LAMBDAS, key=lambda l: test[l])
    base = test[0.0]
    return {"lam_sel": lam_sel, "lam_oracle": lam_orc,
            "gain_selected": base - test[lam_sel],
            "gain_fixed_1": base - test[1.0],
            "gain_oracle": base - test[lam_orc]}


def main():
    grid = {}
    summary = {}
    for d in DELTAS:
        for s in S_R:
            rows = [one_rep(d, s, seed) for seed in range(REPS)]
            g_sel = np.array([r["gain_selected"] for r in rows])
            g_1 = np.array([r["gain_fixed_1"] for r in rows])
            g_or = np.array([r["gain_oracle"] for r in rows])
            lam_hist = {str(l): int(sum(r["lam_sel"] == l for r in rows)) for l in LAMBDAS}
            key = f"delta={d:g},s_r={s:g}"
            grid[key] = {
                "mean_gain_cv_selected": round(float(g_sel.mean()), 4),
                "sd_gain_cv_selected": round(float(g_sel.std(ddof=1)), 4),
                "harm_rate_cv_selected(G<0)": round(float((g_sel < 0).mean()), 3),
                "mean_gain_fixed_lambda1": round(float(g_1.mean()), 4),
                "harm_rate_fixed_lambda1": round(float((g_1 < 0).mean()), 3),
                "mean_gain_oracle": round(float(g_or.mean()), 4),
                "lambda_selection_counts": lam_hist,
            }
    # 요약: 선택 절차가 해를 피하는 영역과 못 피하는 영역
    sel_means = np.array([grid[k]["mean_gain_cv_selected"] for k in grid])
    harm = np.array([grid[k]["harm_rate_cv_selected(G<0)"] for k in grid])
    summary["cells_with_mean_gain_below_zero_cv"] = [k for k in grid if grid[k]["mean_gain_cv_selected"] < 0]
    summary["max_harm_rate_cv"] = float(harm.max())
    summary["min_mean_gain_cv"] = float(sel_means.min())
    summary["max_mean_gain_cv"] = float(sel_means.max())
    out = {"settings": {"D": D, "sigma": SIG, "N_robot": N_R, "N_human": N_H, "N_test": N_TEST,
                        "lambda_grid": LAMBDAS, "cv_folds": K, "reps": REPS,
                        "gap_direction": "1/sqrt(D) (ones)", "note": "가정 모델. 논문 수치가 아님"},
           "grid": grid, "summary": summary}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    for k, v in grid.items():
        print(k, v["mean_gain_cv_selected"], v["harm_rate_cv_selected(G<0)"], v["mean_gain_fixed_lambda1"], v["mean_gain_oracle"])


if __name__ == "__main__":
    main()
