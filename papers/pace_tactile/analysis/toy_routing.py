"""토이: 촉각 신호를 상태(느림)와 사건(빠름)으로 나누면 사건 검출이 나아지는가.

가정 (토이이며 논문 실험이 아님):
- 1차원 촉각 신호 x_t = 상태 s_t + 사건 e_t + 잡음 n_t. 20 Hz, 600 스텝, 400 에피소드.
- 상태: 접근(0) → 첫 접촉(1.0, t=150) → 안정 삽입(힘 3.0, t=380). 상태 변화는 EMA(α=0.05)로 매끄럽다.
- 사건: 충돌 또는 걸림 스파이크(진폭 ±1.5, 3스텝, 확률 0.01/스텝). 안정 삽입 구간에도 나온다.
- 잡음은 상태에 따라 달라진다(안정 삽입 구간의 잡음이 2배). 이 가정이 결과를 만든다.

비교 검출기 (같은 전역 허위경보율, 1000스텝당 1회로 맞춤):
- 단일 임베딩: 하나의 특징 f_t = |x_t - x_{t-4}|에 고정 임계값.
- 분리(상태 조건부 척도): 상태 ŝ_t = EMA(x), 잔차 r_t = x_t - ŝ_t, 사건 점수 |r_t - r_{t-4}|를
  상태에 따라 갱신되는 척도(EMA of |점수|, α=0.02)로 나눈다.

지표: 사건 후 10스텝 안에 검출한 비율, 중앙 지연(스텝), 구간별(접근/첫 접촉/안정 삽입) 검출률.
실행: python3 analysis/toy_routing.py → analysis/toy_routing_results.json (numpy 단일 프로세스, 수 초)
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("toy_routing_results.json")
T, E, TW = 600, 400, 4
CONTACT, INSERT = 150, 380
RATE, AMP, DUR = 0.01, 1.5, 3
ALPHA_STATE, ALPHA_SCALE = 0.05, 0.02
TARGET_FA = 1e-3  # 스텝당 허위경보율


def simulate(rng, sigma, n_ep=E, with_events=True, hetero=True):
    t = np.arange(T)
    state = np.zeros(T)
    state[CONTACT:INSERT] = 1.0
    state[INSERT:] = 3.0
    # 상태 변화를 EMA로 매끄럽게 (피크 이동 방지)
    s = np.zeros((n_ep, T))
    for k in range(1, T):
        s[:, k] = (1 - ALPHA_STATE) * s[:, k - 1] + ALPHA_STATE * state[k]
    noise_scale = np.where((t >= INSERT) & hetero, 2.0 * sigma, sigma)
    noise = rng.standard_normal((n_ep, T)) * noise_scale
    ev = np.zeros((n_ep, T))
    onset = np.zeros((n_ep, T), dtype=bool)
    if with_events:
        starts = rng.random((n_ep, T)) < RATE
        starts[:, :10] = False
        for d in range(DUR):
            shifted = np.zeros_like(starts)
            shifted[:, d:] = starts[:, : T - d]
            sign = rng.choice([-1.0, 1.0], size=(n_ep, T))
            ev += shifted * AMP * sign
        onset = starts
    x = s + ev + noise
    return x, onset, s


def feat_single(x):
    f = np.zeros_like(x)
    f[:, TW:] = np.abs(x[:, TW:] - x[:, :-TW])
    return f


def feat_split(x):
    s_hat = np.zeros_like(x)
    for k in range(1, x.shape[1]):
        s_hat[:, k] = (1 - ALPHA_STATE) * s_hat[:, k - 1] + ALPHA_STATE * x[:, k]
    r = x - s_hat
    d = np.zeros_like(x)
    d[:, TW:] = np.abs(r[:, TW:] - r[:, :-TW])
    scale = np.ones_like(x)
    for k in range(1, x.shape[1]):
        scale[:, k] = (1 - ALPHA_SCALE) * scale[:, k - 1] + ALPHA_SCALE * d[:, k]
    return d / np.maximum(scale, 1e-6)


def calibrate(feat_noise):
    """허위경보율이 TARGET_FA가 되도록 임계값을 정한다(사건 없는 에피소드 기준)."""
    flat = feat_noise[:, TW + 5:].ravel()
    return float(np.quantile(flat, 1 - TARGET_FA))


def evaluate(feat, thr, onset):
    alarm = feat > thr
    ep, tt = np.nonzero(onset)
    hits, lat = 0, []
    phase_hit = {"approach": [0, 0], "contact": [0, 0], "insert": [0, 0]}
    for e_i, t0 in zip(ep, tt):
        if t0 < TW or t0 > T - 12:
            continue
        window = alarm[e_i, t0 : t0 + 10]
        ph = "approach" if t0 < CONTACT else ("contact" if t0 < INSERT else "insert")
        phase_hit[ph][1] += 1
        if window.any():
            hits += 1
            lat.append(int(np.argmax(window)))
            phase_hit[ph][0] += 1
    n = sum(v[1] for v in phase_hit.values())
    out = {"detect_within_10": round(hits / n, 3), "n_events": n,
           "median_latency_steps": float(np.median(lat)) if lat else None,
           "by_phase": {k: round(v[0] / v[1], 3) if v[1] else None for k, v in phase_hit.items()}}
    return out


def main():
    rng = np.random.default_rng(0)
    results = {}
    for sigma in [0.1, 0.3, 0.6]:
        x_clean, _, _ = simulate(rng, sigma, with_events=False)
        x, onset, _ = simulate(rng, sigma, with_events=True)
        row = {}
        for name, fn in [("single", feat_single), ("split_state_scaled", feat_split)]:
            thr = calibrate(fn(x_clean))
            fa = float(np.mean(fn(x_clean)[:, TW + 5:] > thr))
            row[name] = {"threshold": round(thr, 4), "measured_fa_per_step": round(fa, 5),
                         **evaluate(fn(x), thr, onset)}
        results[f"sigma={sigma}"] = row

    # 민감도: 잡음 이분산을 끈 경우 (상태 조건부 척도가 필요 없는지 확인)
    rng2 = np.random.default_rng(1)
    homo = {}
    xc, _, _ = simulate(rng2, 0.3, with_events=False, hetero=False)
    x, onset, _ = simulate(rng2, 0.3, with_events=True, hetero=False)
    for name, fn in [("single", feat_single), ("split_state_scaled", feat_split)]:
        thr = calibrate(fn(xc))
        homo[name] = evaluate(fn(x), thr, onset)
    results["homoscedastic_check_sigma0.3"] = homo

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in results.items():
        print(k)
        for name in ["single", "split_state_scaled"] if k.startswith("sigma") else ["single", "split_state_scaled"]:
            r = v[name]
            print(f"  {name:20s} detect10={r.get('detect_within_10')} lat={r.get('median_latency_steps')} by_phase={r.get('by_phase')}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
