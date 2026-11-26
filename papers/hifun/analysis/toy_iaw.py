"""토이: 식 (7)–(8)의 개입 이점 가중(IAW)이 비평가 오차에 얼마나 의존하는지 본다.

이 토이가 보는 것: 개입 행동 a_I에 w = sigmoid(Norm(A_I)/tau)를 곱한 모방 손실이 "품질이 섞인 개입"에서 균등 모방보다
낫고, 비평가가 부정확하면 그 이점이 줄어드는지. 이 토이가 보지 않는 것: SAC 손실, 다중 헤드 비평가, 실로봇 데이터.
논문은 tau_I와 Norm의 정의를 쓰지 않으므로 아래 값(z-점수, tau=0.3)은 모두 가정이다.

설정(가정)
- 상태 s ~ N(0, I_3), 최적 행동 a*(s) = W s (W는 2x3 고정 난수), 정책 a(s) = Theta s, 시작 Theta = 0.
- 진짜 가치 Q(s, a) = -||a - a*(s)||^2. 추정 가치 Qhat = Q + eps * sd(Q) * g(s, a), g는 무작위 푸리에 특징 평균(표준화).
- 개입: 비율 (1 - f_bad)는 좋은 개입 a* + N(0, 0.1^2), 비율 f_bad는 주저하는 약한 개입 0.4 a* + N(0, 0.3^2)(목표 쪽으로 덜 가는 보정). N = 100.
- 방법: uniform(모든 개입 같은 가중), IAW(w = sigmoid(z(A)/tau), A = Qhat(s, a_I) - Qhat(s, a_pi(s)), 정책을 20회 반복 갱신),
  oracle(좋은 개입만 사용, 상한).
- 지표: ||Theta - W||_F / ||W||_F 의 시드 200개 평균과 표준오차.

실행: python3 analysis/toy_iaw.py   (numpy, 수 초)
"""
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore', category=RuntimeWarning)
OUT = Path(__file__).resolve().parent / "toy_iaw_results.json"
TAU = 0.3
SIG_GOOD, SIG_BAD, N, SHRINK = 0.1, 0.3, 100, 0.4


def fit(S, A, w):
    sw = np.sqrt(w)[:, None]
    th, *_ = np.linalg.lstsq(S * sw, A * sw, rcond=None)
    return th.T  # 2x3


def run(seed, eps, f_bad):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((2, 3))
    S = rng.standard_normal((N, 3))
    star = S @ W.T
    bad = rng.random(N) < f_bad
    AI = np.where(bad[:, None], SHRINK * star, star) + rng.standard_normal((N, 2)) * np.where(bad, SIG_BAD, SIG_GOOD)[:, None]
    # 비평가 오차: 무작위 푸리에 특징
    K = 32
    Om = rng.standard_normal((K, 5)) * 0.7
    ph = rng.uniform(0, 2 * np.pi, K)

    def g(s, a):
        x = np.concatenate([s, a], axis=1)
        return np.cos(x @ Om.T + ph).mean(axis=1) * np.sqrt(K)

    probe_a = star + rng.standard_normal((N, 2))
    q_sd = (-(np.sum((probe_a - star) ** 2, axis=1))).std()
    g_sd = g(S, probe_a).std() + 1e-9

    def qhat(s, a):
        return -np.sum((a - s @ W.T) ** 2, axis=1) + eps * q_sd * g(s, a) / g_sd

    res = {}
    # uniform
    th = fit(S, AI, np.ones(N))
    res["uniform"] = np.linalg.norm(th - W) / np.linalg.norm(W)
    # IAW
    th = np.zeros((2, 3))
    for _ in range(20):
        Adv = qhat(S, AI) - qhat(S, S @ th.T)
        z = (Adv - Adv.mean()) / (Adv.std() + 1e-9)
        w = 1 / (1 + np.exp(-z / TAU))
        th = fit(S, AI, w)
    res["iaw"] = np.linalg.norm(th - W) / np.linalg.norm(W)
    # oracle
    th = fit(S, AI, (~bad).astype(float) + 1e-9)
    res["oracle_good_only"] = np.linalg.norm(th - W) / np.linalg.norm(W)
    # 가중치와 약한 개입의 구분(AUC 대용: 약한 개입의 평균 가중 대 좋은 개입의 평균 가중)
    res["w_good"] = float(w[~bad].mean())
    res["w_bad"] = float(w[bad].mean()) if bad.any() else float("nan")
    return res


def main():
    out = {"tau": TAU, "n": N, "sig_good": SIG_GOOD, "sig_bad": SIG_BAD, "shrink_bad": SHRINK, "grid": []}
    for f_bad in (0.0, 0.3, 0.5, 0.7):
        for eps in (0.0, 0.5, 1.0, 2.0, 4.0):
            rs = [run(s, eps, f_bad) for s in range(200)]
            row = {"f_bad": f_bad, "eps": eps}
            for k in ("uniform", "iaw", "oracle_good_only", "w_good", "w_bad"):
                v = np.array([r[k] for r in rs], dtype=float)
                row[k] = round(float(np.nanmean(v)), 4)
                row[k + "_se"] = round(float(np.nanstd(v) / np.sqrt(np.sum(~np.isnan(v)))), 4)
            out["grid"].append(row)
            print(f"f_bad={f_bad:.1f} eps={eps:.1f} uniform={row['uniform']:.3f} iaw={row['iaw']:.3f} "
                  f"oracle={row['oracle_good_only']:.3f} w_good={row['w_good']:.2f} w_bad={row['w_bad']:.2f}")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
