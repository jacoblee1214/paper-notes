"""토이: 두 갈래 행동 모드와 잡음 낀 관측에서 롤링 디노이징이 모드 유지와 반응 사이에 어떻게 놓이는가.

논문의 핵심 서술(§3.1)을 해석적 모델로 확인한다. 학습은 없다. 정책은 "관측 하나가 주는 모드 사후확률"을 아는
베이즈 최적 흐름 정책이다.
  - 행동 청크 a ∈ R^H, a_j = m + σ_a z_j, 모드 m ∈ {-1, +1}. 관측 o = m*(t) + σ_o ε (모드를 가리키는 잡음 낀 단서).
  - x_j = τ_j ε_j + (1-τ_j) a_j 일 때 최적 속도장은 v = (x - E[a|x, o]) / τ (u = ε - a 이므로).
  - E[a|x,o]는 모드 사후 (관측 우도 × 버퍼에 이미 담긴 위치별 우도)로 닫힌 형태로 계산된다.
방법: H50E50(청크 시작 관측 하나로 50스텝), H50E10(10스텝마다 새 관측으로 새 청크), REACT(K=5, S=10 롤링,
블록마다 새 관측, Euler 한 단계), REACT-stale(롤링인데 관측이 한 블록 묵음, Dual Decoupling의 관측 나이 모사).
관측 시점은 블록 경계이고 연산 지연은 없다. 참 모드 m*(t)는 평균 200스텝마다 바뀐다(최소 간격 60).

측정: 정확도(실행 행동 부호 = 참 모드인 스텝 비율), 지연(참 모드가 바뀐 뒤 새 모드로 10스텝 이상 연속 일치하는
첫 시점까지의 스텝, 다음 전환 전에 못 맞추면 미반응), 블록 부호 전환 횟수(참 전환 포함).
한계: 1차원, 모드 두 개, 관측 독립 잡음, 사후를 아는 정책이다. 실제 π0.5의 성질은 확인하지 않는다.

실행: python3 analysis/toy_rolling_modes.py  (numpy, 약 20초) → analysis/toy_rolling_modes_results.json
"""
import json
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_rolling_modes_results.json"
H, K, S, N_STEPS = 50, 5, 10, 10
SIG_A = 0.3
T_TOTAL = 600
EPISODES = 400


def make_truth(rng, n):
    m = np.ones((n, T_TOTAL))
    for e in range(n):
        t, cur = 0, rng.choice([-1.0, 1.0])
        while t < T_TOTAL:
            gap = 60 + rng.geometric(1 / 140)
            m[e, t:t + gap] = cur
            cur, t = -cur, t + gap
    return m


def log_gauss(x, mu, var):
    return -0.5 * (x - mu) ** 2 / var - 0.5 * np.log(var)


def posterior_mean(x, tau, o, sig_o, beta=1.0):
    """x: (E,H) 버퍼, tau: (H,) 위치별 흐름 시간, o: (E,) 관측. 반환 E[a|x,o]와 모드 사후 P(m=+1).
    beta는 버퍼 증거를 얼마나 믿는지(1 = 버퍼가 전방 과정의 표본이라고 믿는 정확한 베이즈, 0 = 버퍼 무시)."""
    logit = 2 * o / sig_o ** 2  # 관측 우도: log N(o; +1) - log N(o; -1)
    var = (1 - tau) ** 2 * SIG_A ** 2 + tau ** 2
    for_plus = log_gauss(x, (1 - tau), var) - log_gauss(x, -(1 - tau), var)
    logit = logit + beta * for_plus.sum(axis=1)
    p_plus = 1 / (1 + np.exp(-np.clip(logit, -50, 50)))
    gain = (1 - tau) * SIG_A ** 2 / var
    a_plus = 1 + gain * (x - (1 - tau))
    a_minus = -1 + gain * (x + (1 - tau))
    return p_plus[:, None] * a_plus + (1 - p_plus)[:, None] * a_minus, p_plus


def velocity(x, tau, o, sig_o, beta=1.0):
    a_hat, _ = posterior_mean(x, tau, o, sig_o, beta)
    return (x - a_hat) / tau


def sample_chunk(rng, o, sig_o):
    """pure noise에서 N_STEPS Euler 단계로 H 길이 청크를 뽑는다 (τ 1 -> 0 균등)."""
    n = o.shape[0]
    x = rng.standard_normal((n, H))
    for i in range(N_STEPS):
        t = 1 - i / N_STEPS
        x = x - velocity(x, np.full(H, t), o, sig_o) / N_STEPS
    return x


def run(method, sig_o, truth, rng, beta=1.0):
    n = truth.shape[0]
    exe = np.zeros((n, T_TOTAL))
    n_blocks = T_TOTAL // S
    tau = (np.arange(H) // S + 1) / K

    def obs(b, delay=0):
        t = max(b - delay, 0) * S
        return truth[:, t] + sig_o * rng.standard_normal(n)

    if method == "H50E50":
        for b in range(0, n_blocks, H // S):
            chunk = sample_chunk(rng, obs(b), sig_o)
            exe[:, b * S:b * S + H] = chunk[:, :min(H, T_TOTAL - b * S)]
    elif method == "H50E10":
        for b in range(n_blocks):
            exe[:, b * S:(b + 1) * S] = sample_chunk(rng, obs(b), sig_o)[:, :S]
    else:  # REACT 롤링
        delay = 1 if method == "REACT_stale" else 0
        o0 = obs(0)
        x = (1 - tau) * 0.0 + tau * rng.standard_normal((n, H))  # 초기 자세 s0 = 0 에서 계단 잡음
        for _ in range(K - 1):  # 워밍업: 첫 관측에서 K-1 회 갱신, 앞 블록은 버림
            x1 = x - velocity(x, tau, o0, sig_o, beta) / K
            x = np.concatenate([x1[:, S:], rng.standard_normal((n, S))], axis=1)
        for b in range(n_blocks):
            x1 = x - velocity(x, tau, obs(b, delay), sig_o, beta) / K
            exe[:, b * S:(b + 1) * S] = x1[:, :S]
            x = np.concatenate([x1[:, S:], rng.standard_normal((n, S))], axis=1)
    return exe


def metrics(exe, truth):
    acc = float((np.sign(exe) == truth).mean())
    blk = np.sign(exe.reshape(exe.shape[0], -1, S).mean(axis=2))
    flips = float((np.diff(blk, axis=1) != 0).sum(axis=1).mean())
    true_flips = float((np.diff(truth[:, ::S], axis=1) != 0).sum(axis=1).mean())
    lags, missed, total = [], 0, 0
    for e in range(exe.shape[0]):
        sw = np.where(np.diff(truth[e]) != 0)[0] + 1
        for k, ts in enumerate(sw):
            te = sw[k + 1] if k + 1 < len(sw) else T_TOTAL
            ok = np.sign(exe[e, ts:te]) == truth[e, ts]
            total += 1
            run_len, hit = 0, None
            for i, v in enumerate(ok):
                run_len = run_len + 1 if v else 0
                if run_len >= 10:
                    hit = i - 9
                    break
            if hit is None:
                missed += 1
            else:
                lags.append(hit)
    return {"accuracy": round(acc, 4), "block_sign_flips_per_episode": round(flips, 2),
            "true_switches_per_episode": round(true_flips, 2),
            "median_lag_steps": float(np.median(lags)) if lags else None,
            "mean_lag_steps": round(float(np.mean(lags)), 1) if lags else None,
            "missed_fraction": round(missed / total, 3)}


def main():
    t0 = time.time()
    betas = (1.0, 0.3, 0.1, 0.03, 0.01, 0.0)
    res = {"settings": {"H": H, "K": K, "S": S, "sigma_a": SIG_A, "episodes": EPISODES, "T": T_TOTAL,
                        "euler_steps_chunk": N_STEPS, "betas": betas}}
    for sig_o in (0.5, 1.0, 2.0):
        rng = np.random.default_rng(0)
        truth = make_truth(rng, EPISODES)
        cur = res[f"sigma_o={sig_o}"] = {}
        for method in ("H50E50", "H50E10"):
            r = np.random.default_rng(1)
            cur[method] = metrics(run(method, sig_o, truth, r), truth)
        for method in ("REACT", "REACT_stale"):
            for beta in betas:
                r = np.random.default_rng(1)
                cur[f"{method}|beta={beta}"] = metrics(run(method, sig_o, truth, r, beta), truth)
        print(f"sigma_o={sig_o}")
        for m, v in cur.items():
            print(f"  {m:20s} acc={v['accuracy']:.3f} flips={v['block_sign_flips_per_episode']:5.1f} "
                  f"lag(med/mean)={v['median_lag_steps']}/{v['mean_lag_steps']} missed={v['missed_fraction']}")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
