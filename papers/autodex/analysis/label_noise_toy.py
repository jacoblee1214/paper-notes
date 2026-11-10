"""토이 A: 자동 성공 라벨의 오차가 학습한 성공 분류기와 상위 선별에 주는 영향 (numpy, 1스레드).

실행: python3 analysis/label_noise_toy.py -> analysis/label_noise_toy_results.json (수십 초)
가정(설계값, 논문 값 아님): 특징 d=6, 참 성공 확률은 로지스틱 모형, 학습 N=3593(논문 개수), 시험 N=2000(깨끗한 라벨),
반복 30회. 라벨 오차는 (a) 균일 (e0 거짓 성공, e1 거짓 실패), (b) 특징 의존(x0이 클수록 거짓 실패 증가)을 본다.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("label_noise_toy_results.json")
D = 6
W = np.array([1.2, -0.8, 0.5, 0.0, 0.3, -0.4])
B = 0.05
N_TRAIN, N_TEST, REPS = 3593, 2000, 30


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_logreg(X, y, l2=1e-3, iters=30):
    """Newton(IRLS) 로지스틱 회귀. 절편 포함."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    beta = np.zeros(Xb.shape[1])
    reg = l2 * np.eye(Xb.shape[1])
    reg[-1, -1] = 0.0
    for _ in range(iters):
        p = sigmoid(Xb @ beta)
        g = Xb.T @ (p - y) + reg @ beta
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + reg + 1e-9 * np.eye(Xb.shape[1])
        step = np.linalg.solve(H, g)
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def auc(score, y):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def top_precision(score, y, frac=0.2):
    k = int(frac * len(score))
    idx = np.argsort(-score)[:k]
    return float(y[idx].mean())


def run(e0, e1, structured, rng, n_train=N_TRAIN):
    N_TR = n_train
    x_tr = rng.normal(size=(N_TR, D))
    p_tr = sigmoid(x_tr @ W + B)
    y_true = (rng.random(N_TR) < p_tr).astype(float)
    if structured:
        # x0이 클수록(=어려운 자세 가정) 참 성공을 거짓 실패로 잘못 적을 확률이 커진다.
        f1 = np.clip(e1 * (1 + 1.0 * np.tanh(x_tr[:, 0])), 0, 0.95)
    else:
        f1 = np.full(N_TR, e1)
    flip_fn = (y_true == 1) & (rng.random(N_TR) < f1)
    flip_fp = (y_true == 0) & (rng.random(N_TR) < e0)
    y_obs = y_true.copy()
    y_obs[flip_fn] = 0
    y_obs[flip_fp] = 1

    x_te = rng.normal(size=(N_TEST, D))
    p_te = sigmoid(x_te @ W + B)
    y_te = (rng.random(N_TEST) < p_te).astype(float)

    beta = fit_logreg(x_tr, y_obs)
    w_hat = beta[:D]
    cos = float(w_hat @ W / (np.linalg.norm(w_hat) * np.linalg.norm(W)))
    slope = float(w_hat @ W / (W @ W))  # 참 W 방향 투영 계수(1이면 감쇠 없음)
    score = x_te @ w_hat + beta[-1]
    return {
        "calib_bias_pred_minus_true": float(sigmoid(score).mean() - p_te.mean()),
        "observed_label_rate": float(y_obs.mean()),
        "true_rate": float(y_true.mean()),
        "cos_to_true_w": cos,
        "attenuation_slope": slope,
        "auc_clean_test": auc(score, y_te),
        "auc_oracle": auc(x_te @ W + B, y_te),
        "top20_precision": top_precision(score, y_te),
        "top20_precision_oracle": top_precision(x_te @ W + B, y_te),
    }


SETTINGS = [
    ("oracle (라벨 오차 없음)", 0.0, 0.0, False),
    ("균일 e0=0.02, e1=0.02", 0.02, 0.02, False),
    ("균일 e0=0.05, e1=0.05", 0.05, 0.05, False),
    ("균일 e0=0.10, e1=0.10", 0.10, 0.10, False),
    ("거짓 성공 중심 e0=0.10, e1=0.02", 0.10, 0.02, False),
    ("거짓 실패 중심 e0=0.02, e1=0.10", 0.02, 0.10, False),
    ("특징 의존 e0=0.02, e1=0.10 (x0 의존)", 0.02, 0.10, True),
]

if __name__ == "__main__":
    results = {}
    for n_tr in (N_TRAIN, 200):
     for name, e0, e1, st in SETTINGS:
        rows = []
        for r in range(REPS):
            rng = np.random.default_rng(1000 + r)
            rows.append(run(e0, e1, st, rng, n_train=n_tr))
        agg = {}
        for k in rows[0]:
            v = np.array([row[k] for row in rows])
            agg[k] = {"mean": round(float(v.mean()), 4), "sd": round(float(v.std(ddof=1)), 4)}
        results[f"N={n_tr} | {name}"] = {"n_train": n_tr, "e0": e0, "e1": e1, "feature_dependent": st, **agg}
        print(n_tr, name, {k: agg[k]["mean"] for k in ("observed_label_rate", "calib_bias_pred_minus_true", "attenuation_slope", "auc_clean_test", "top20_precision")})
    meta = {"d": D, "W": W.tolist(), "B": B, "N_train": N_TRAIN, "N_test": N_TEST, "reps": REPS,
            "note": "설계값 토이. 논문 수치가 아니다. 참 성공률은 설정에 따라 다르다."}
    OUT.write_text(json.dumps({"meta": meta, "settings": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("written", OUT)
