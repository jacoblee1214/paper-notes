"""토이: 노이즈 섞인 미래 잠재 상태에서 행동을 복원할 수 있는가 (numpy, 1초 안팎).

논문 주장(§3.3 식 (10), §3.4): 학습 때 z_{t+1}을 흐름 시간 s_aug ~ U[0.5,1]로 섞고(확률 0.5),
추론 때 s=0.5까지만 적분해도 행동 복원이 유지된다.
토이: 선형 동역학 s_{t+1} = s_t + B a_t, 관측 잠재 z = (1-s)eps + s * s_{t+1} 로 두고,
선형 역동역학 a_t = W [z, s_t] 를 (a) 깨끗한 z로만 학습, (b) 논문식 증강으로 학습한 뒤
여러 s에서 행동 오차를 비교한다. 5 시드, 합성 데이터.
실행: OMP_NUM_THREADS=2 python3 lingbot_va/analysis/toy_noisy_history.py
출력: lingbot_va/analysis/toy_noisy_history_results.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D_S, D_A, N_TRAIN, N_TEST, SEEDS = 8, 3, 4000, 2000, 5
S_LEVELS = [0.0, 0.5, 0.6, 1.0]


def make_data(rng, n, B):
    s0 = rng.normal(size=(n, D_S))
    a = rng.normal(size=(n, D_A))
    s1 = s0 + a @ B.T + 0.02 * rng.normal(size=(n, D_S))
    return s0, a, s1


def observe(rng, s1, level):
    """흐름 보간 z = (1-level) eps + level * s1 (level=1이면 깨끗한 관측)."""
    eps = rng.normal(size=s1.shape)
    return (1 - level) * eps + level * s1


def fit_ridge(X, Y, lam=1e-3):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)


def run(seed):
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(D_S, D_A)) / np.sqrt(D_A)
    s0, a, s1 = make_data(rng, N_TRAIN, B)
    Wc = fit_ridge(np.hstack([s1, s0]), a)  # (a) 깨끗한 관측만
    lv = np.where(rng.random(N_TRAIN) < 0.5, rng.uniform(0.5, 1.0, N_TRAIN), 1.0)  # (b) 논문식 증강
    z_aug = np.vstack([observe(rng, s1[i:i + 1], lv[i]) for i in range(N_TRAIN)])
    Wa = fit_ridge(np.hstack([z_aug, s0]), a)
    s0t, at, s1t = make_data(rng, N_TEST, B)
    out = {}
    for lev in S_LEVELS:
        zt = observe(rng, s1t, lev) if lev < 1.0 else s1t
        Xt = np.hstack([zt, s0t])
        out[lev] = {"clean_trained_mse": float(np.mean((Xt @ Wc - at) ** 2)),
                    "aug_trained_mse": float(np.mean((Xt @ Wa - at) ** 2))}
    return out


if __name__ == "__main__":
    runs = [run(s) for s in range(SEEDS)]
    summary = {}
    for lev in S_LEVELS:
        c = np.array([r[lev]["clean_trained_mse"] for r in runs])
        g = np.array([r[lev]["aug_trained_mse"] for r in runs])
        summary[str(lev)] = {
            "clean_trained_mse_mean": round(float(c.mean()), 4),
            "clean_trained_mse_sd": round(float(c.std(ddof=1)), 4),
            "aug_trained_mse_mean": round(float(g.mean()), 4),
            "aug_trained_mse_sd": round(float(g.std(ddof=1)), 4)}
    res = {"note": "합성 선형 토이. 논문 수치와 직접 비교하지 않는다. 방향만 본다.",
           "seeds": SEEDS, "summary_by_level": summary}
    (ROOT / "analysis" / "toy_noisy_history_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))
