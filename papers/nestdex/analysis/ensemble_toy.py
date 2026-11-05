"""Eq. 2 temporal ensembling toy (CPU, numpy). 실행: python3 analysis/ensemble_toy.py

- E1: 가중치 w_i = exp(-m i), P_t[0]이 가장 오래된 예측 → 겹친 예측 K개에서 예측의 평균 나이와 최신 예측의 몫
- E2: 손가락 닫기 궤적을 따라가는 잡음 섞인 chunk 정책(100 Hz, chunk 30)에서
      무앙상블(새 chunk의 첫 행동) vs 앙상블(m 여러 값)의 P95 jerk(Savitzky–Golay 3차, 0.21 s)와
      접촉 사건 뒤 명령이 새 정보를 반영하는 데 걸리는 시간
가정: 잡음은 질의마다 독립인 오프셋 + 기울기 + 백색 잡음. 실제 정책의 오차 구조는 모른다.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
HZ, H = 100, 30          # 내부 정책: 100 Hz, chunk 30 (Sec. V)
T_SEC = 3.0
SEEDS = 20


def weights(m, k):
    i = np.arange(k)                    # 0 = 가장 오래된 예측
    w = np.exp(-m * i)
    return w / w.sum()


def e1():
    rows = {}
    for m in (0.0, 0.01, 0.05, 0.1, 0.3, 1.0):
        w = weights(m, H)
        age = (H - 1) - np.arange(H)       # 오래된 예측일수록 나이가 많다
        rows[f"m={m}"] = {"mean_age_steps": round(float((w * age).sum()), 2),
                          "mean_age_ms": round(float((w * age).sum()) * 1000 / HZ, 1),
                          "newest_weight": round(float(w[-1]), 4), "oldest_weight": round(float(w[0]), 4)}
    return rows


def target(t, contact_t=None, stop_level=0.6):
    """0→1 닫기 (1.2 s 동안 smoothstep). contact_t 뒤에는 물체에 막혀 stop_level에서 멈춘다."""
    s = np.clip((t - 0.5) / 1.2, 0, 1)
    x = s * s * (3 - 2 * s)
    if contact_t is not None:
        x = np.where(t >= contact_t, np.minimum(x, stop_level), x)
    return x


def simulate(mode, m, rng, contact_t=1.1, sigma=0.02, slope_sigma=0.15):
    n = int(T_SEC * HZ)
    t = np.arange(n) / HZ
    preds = {}                           # 실행 시각 k -> 예측 목록(오래된 것부터)
    cmd = np.zeros(n)
    for q in range(n):
        # 질의 시각 q에서 정책이 아는 미래: 접촉이 이미 일어났으면 그 사실을 반영, 아니면 모른다
        known_contact = contact_t if t[q] >= contact_t else None
        fut_t = t[q] + np.arange(1, H + 1) / HZ
        clean = target(fut_t, known_contact)
        noise = rng.normal(0, sigma) + rng.normal(0, slope_sigma) * (fut_t - t[q]) + rng.normal(0, sigma * 0.3, H)
        chunk = clean + noise
        for j in range(H):
            k = q + 1 + j
            if k < n:
                preds.setdefault(k, []).append(chunk[j])
        if q + 1 < n:
            P = preds[q + 1]
            if mode == "none":
                cmd[q + 1] = P[-1] if len(P) else 0.0      # 가장 새 chunk의 첫 행동
            else:
                w = weights(m, len(P))
                cmd[q + 1] = float(np.dot(w, P))
    return t, cmd


def p95_jerk(x):
    win = int(round(0.21 * HZ)) | 1
    d3 = savgol_filter(x, win, 3, deriv=3, delta=1 / HZ)
    return float(np.percentile(np.abs(d3), 95))


def settle_time(t, mean_cmd, contact_t=1.1, stop_level=0.6, tol=0.01):
    """시드 평균 명령이 정지 수준 ± tol 안으로 들어와 머무는 첫 시각까지의 지연(ms). 잡음은 평균에서 상쇄된다."""
    ok = np.abs(mean_cmd - stop_level) <= tol
    for i in np.where(t >= contact_t)[0]:
        if ok[i:].all():
            return round((t[i] - contact_t) * 1000, 1)
    return None


def e2():
    res = {}
    for mode, m in [("none", None), ("ens", 0.01), ("ens", 0.05), ("ens", 0.1), ("ens", 0.3)]:
        jerks, traces, overshoot, jitter = [], [], [], []
        for s in range(SEEDS):
            t, c = simulate(mode, m, np.random.default_rng(1000 + s))
            jerks.append(p95_jerk(c))
            traces.append(c)
            overshoot.append(float(c[t >= 1.1].max() - 0.6))
            jitter.append(float(np.std(c[(t >= 1.8)] - 0.6)))
        # 체계적 응답은 잡음 없는 정책으로 잰다(잡음이 있으면 정착 판정이 흔들림)
        t0, c0 = simulate(mode, m, np.random.default_rng(0), sigma=0.0, slope_sigma=0.0)
        key = "no_ensemble" if mode == "none" else f"ensemble_m={m}"
        res[key] = {"p95_jerk_median": round(float(np.median(jerks)), 1),
                    "settle_after_contact_ms_noise_free": settle_time(t0, c0, tol=0.005),
                    "overshoot_past_stop_noise_free": round(float(c0[t0 >= 1.1].max() - 0.6), 4),
                    "steady_jitter_std": round(float(np.median(jitter)), 4)}
    base = res["no_ensemble"]["p95_jerk_median"]
    for k, v in res.items():
        v["jerk_ratio_vs_no_ensemble"] = round(base / v["p95_jerk_median"], 2) if k != "no_ensemble" else 1.0
    return res


def main():
    out = {"setup": {"hz": HZ, "chunk": H, "seeds": SEEDS, "contact_at_s": 1.1, "stop_level": 0.6,
                     "noise": "질의마다 오프셋 N(0,0.02) + 기울기 N(0,0.15)/s + 백색 N(0,0.006)"},
           "E1_weights": e1(), "E2_jerk_and_response": e2(),
           "reading": "앙상블은 jerk를 줄이지만, 가장 오래된 예측에 가장 큰 가중을 주므로 접촉 뒤 새 정보 반영이 늦어진다. m은 논문에 없다."}
    p = ROOT / "analysis" / "ensemble_toy_results.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
