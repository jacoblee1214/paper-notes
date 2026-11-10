"""토이 A: 사람(소스) 데이터를 로봇(타깃) 예측에 섞으면 언제 도움이 되는가.

모델: 타깃 y = w_r^T x + eps (로봇 도메인), 소스 y = (w_r + delta)^T x + eps (사람 도메인, 도메인 격차 delta).
사람 입력은 리타게팅 오차를 가진다: x_src = x + e_r, e_r ~ N(0, s_r^2 I) (입력 잡음, errors-in-variables).
로봇 데이터 N_t = 25 고정 (논문의 5 h에 해당하는 소량 가정). 소스 N_s를 0..400으로 늘림.
예측기: 소스와 타깃을 합쳐 최소제곱 (소스 가중 w_s in {0, 1}). 평가: 로봇 도메인 새 샘플 MSE.
결과: 격차 delta와 리타게팅 오차 s_r에 따라 소스 추가가 도움/해로움이 갈리는 경계.

실행: python3 analysis/transfer_toy.py -> analysis/transfer_toy_results.json (1스레드, 수 초)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("transfer_toy_results.json")
D = 8          # 입력 차원 (자세 특징 요약)
N_T = 25       # 로봇 데이터 수 (고정)
SIG = 0.5      # 관측 잡음
N_TEST = 4000
REPS = 30
N_SRC = [0, 25, 50, 100, 200, 400]
DELTA = [0.0, 0.25, 0.5, 1.0]
SR = [0.0, 0.25, 0.5, 1.0]


def run(delta, s_r, n_s, rep, rng):
    w_r = np.linspace(-1, 1, D)
    w_h = w_r + delta * np.ones(D) / np.sqrt(D)  # 격차 크기 = |delta|
    X_t = rng.normal(size=(N_T, D))
    y_t = X_t @ w_r + SIG * rng.normal(size=N_T)
    X_test = rng.normal(size=(N_TEST, D))
    y_test = X_test @ w_r + SIG * rng.normal(size=N_TEST)
    if n_s > 0:
        X_s_true = rng.normal(size=(n_s, D))
        X_s = X_s_true + s_r * rng.normal(size=(n_s, D))  # 리타게팅 입력 오차
        y_s = X_s_true @ w_h + SIG * rng.normal(size=n_s)
        X_fit = np.vstack([X_t, X_s])
        y_fit = np.concatenate([y_t, y_s])
    else:
        X_fit, y_fit = X_t, y_t
    beta = np.linalg.lstsq(X_fit, y_fit, rcond=None)[0]
    pred = X_test @ beta
    return float(np.mean((pred - y_test) ** 2))


def main():
    rng = np.random.default_rng(0)
    table = {}
    for d in DELTA:
        for s in SR:
            key = f"delta={d},s_r={s}"
            row = []
            for n_s in N_SRC:
                errs = [run(d, s, n_s, r, rng) for r in range(REPS)]
                row.append(round(float(np.mean(errs)), 4))
            table[key] = {"mse_by_N_src": dict(zip(map(str, N_SRC), row))}
    # 이득 = (소스 없음 MSE) - (소스 N=400 MSE). 양수면 도움.
    summary = {}
    for k, v in table.items():
        base = v["mse_by_N_src"]["0"]
        best = min(v["mse_by_N_src"].items(), key=lambda kv: kv[1])
        v["gain_at_400"] = round(base - v["mse_by_N_src"]["400"], 4)
        v["best_N_src"] = int(best[0])
        summary[k] = {"gain_at_400": v["gain_at_400"], "best_N_src": v["best_N_src"]}
    res = {"settings": {"D": D, "N_t": N_T, "sigma": SIG, "reps": REPS,
                        "N_src": N_SRC, "delta_norm": DELTA, "s_r": SR,
                        "note": "delta는 w 벡터에 (1/sqrt(D))-방향으로 더한 격차 크기; N_t=25 고정 가정"},
           "table": table, "summary": summary}
    OUT.write_text(json.dumps(res, indent=2))
    for k, v in summary.items():
        print(k, v)


if __name__ == "__main__":
    main()
