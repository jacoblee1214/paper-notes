"""토이: 시간 가치 위험(TVR, 식 5)과 오프라인 개입 라벨 규칙(부록 E)이 이상적인 가치 곡선에서 어떻게 동작하는가.

공식 코드를 실행하지 않았다. 논문 식 5와 부록 E의 규칙, 그리고 코드를 읽고 이해한 정의(창 8, 최소-최대 정규화,
decline/plateau/abrupt 세 규칙, 창 앞쪽을 첫 값으로 채우는 이력)를 numpy로 다시 썼다.
가치 곡선은 NTTG 라벨(v_t = t/(T-1), 부록 C.1)을 따라 선형으로 오르는 이상적인 성공 궤적과, 거기에 정체 구간을 넣은 궤적이다.

출력: analysis/toy_tvr_labels_results.json
실행: python3 analysis/toy_tvr_labels.py   (numpy, 1초 안팎)

주의: 실제 프록시 가치는 선형이 아니고 잡음이 있다. 이 토이는 "규칙이 이상적인 건강한 궤적을 어떻게 취급하는가"만 본다.
"""
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_tvr_labels_results.json"
K, EPS, GAM_R = 8, 0.005, 0.9
RES = {}


# ---------------------------------------------------------------- 식 5
def tvr(v, t, pad=True):
    """R_t = (1 - V_t) * sum_{i<K} gamma_r^i * (eps - dV_{t-i}). 창이 모자라면 첫 값으로 채운다(pad=True)."""
    idx = [max(0, t - k) for k in range(K + 1)] if pad else [t - k for k in range(K + 1)]
    vals = [v[i] for i in idx]          # V_t, V_{t-1}, ...
    s = 0.0
    for i in range(K):
        d = vals[i] - vals[i + 1]       # dV_{t-i}
        s += GAM_R ** i * (EPS - d)
    return (1.0 - v[t]) * s


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------- 부록 E 규칙 (최소-최대 정규화 후)
def minmax(v):
    lo, hi = float(v.min()), float(v.max())
    return np.zeros_like(v) if hi - lo < 1e-8 else (v - lo) / (hi - lo)


def decline(vals, t, window=8, slope_thr=-0.01, min_drop=0.05, min_down_ratio=0.7):
    s = max(0, t - window + 1)
    seg = vals[s:t + 1]
    if len(seg) < 2:
        return False
    slope = np.polyfit(np.arange(len(seg)), seg, 1)[0]
    diffs = np.diff(seg)
    return bool(slope <= slope_thr and (seg[0] - seg[-1]) >= min_drop and (diffs < 0).mean() >= min_down_ratio)


def plateau(vals, t, window=8, flat_ratio=0.6, delta=0.03, amp_mult=4.0):
    s = max(0, t - window + 1)
    seg = vals[s:t + 1]
    if len(seg) < 2:
        return False
    fr = (np.abs(np.diff(seg)) <= delta).mean()
    return bool(fr >= flat_ratio and (seg.max() - seg.min()) <= delta * amp_mult)


def abrupt(vals, t, window=8, min_drop=0.12, horizon=3, frac=0.5):
    s = max(0, t - window + 1)
    seg = vals[s:t + 1]
    if len(seg) < 2:
        return False
    diffs = np.diff(seg)
    big = np.where(diffs <= -min_drop)[0]
    if big.size == 0:
        return False
    rel = int(big[np.argmin(diffs[big])])
    if s + rel + 1 != t:
        return False
    drop = seg[rel] - seg[rel + 1]
    fut = vals[t + 1:t + 1 + horizon]
    if fut.size == 0:
        return True
    return not bool(np.any(fut >= seg[rel + 1] + frac * drop))


def mined_labels(v):
    n = minmax(v)
    return np.array([decline(n, t) or plateau(n, t) or abrupt(n, t) for t in range(len(v))])


# ---------------------------------------------------------------- A. 건강한 선형 궤적
A = {}
for T in (20, 30, 42, 60, 95, 150, 200, 300):
    v = np.arange(T) / (T - 1)
    lab = mined_labels(v)
    r = np.array([tvr(v, t) for t in range(T)])
    A[T] = {"step_dV": round(1 / (T - 1), 5),
            "mined_positive_frac": round(float(lab.mean()), 3),
            "R_positive_frac(sign)": round(float((r > 0).mean()), 3),
            "R_min": round(float(r.min()), 4), "R_max": round(float(r.max()), 4)}
RES["A_healthy_linear_ramp"] = A
# 플래토 규칙이 모든 단계를 양성으로 만드는 최소 길이(해석식): 7*dV <= 4*0.03 이고 dV <= 0.03
RES["A_plateau_all_positive_min_T"] = int(math.ceil(1 + 7 / 0.12))
RES["A_eps_matches_episode_len"] = 1 / EPS

# ---------------------------------------------------------------- B. 점수 범위: R이 회귀 목표일 때 sigmoid(R)은 어디에 있는가
grid = np.linspace(-0.3, 0.3, 61)
r_vals = []
for dV in grid:
    # 창 전체가 일정한 dV, V_t = 0
    r_vals.append(sum(GAM_R ** i * (EPS - dV) for i in range(K)))
RES["B_score_range"] = {"R_min": round(min(r_vals), 3), "R_max": round(max(r_vals), 3),
                        "sigmoid_min": round(sigmoid(min(r_vals)), 3), "sigmoid_max": round(sigmoid(max(r_vals)), 3),
                        "threshold_0.5_equals_R_sign": True,
                        "R_zero_when_window_mean_dV_equals_eps": True}

# ---------------------------------------------------------------- C. 정체 구간이 있는 궤적에서 두 신호의 일치도
rng = np.random.default_rng(0)


def episode(T, stall_start, stall_len, noise=0.01):
    """T-L 단계 동안 0에서 1로 선형 상승하는 곡선에 길이 L의 정체(값 고정)를 끼워 넣는다."""
    r = np.arange(T - stall_len) / (T - stall_len - 1)
    base = np.concatenate([r[:stall_start], np.full(stall_len, r[stall_start]), r[stall_start:]])
    n = np.cumsum(rng.normal(0, noise, T)) * 0.3
    truth = np.array([stall_start <= t < stall_start + stall_len for t in range(T)])
    return np.clip(base + n, 0, 1.2), truth


def confusion(pred, truth):
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum()); tn = int((~pred & ~truth).sum())
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(prec, 3), "recall": round(rec, 3),
            "pos_rate": round(float(pred.mean()), 3)}


C = {}
for T, stall_len in ((42, 8), (95, 14)):
    agg = {"mined": [], "Rpos": [], "truth": [], "Rpos_vs_mined": []}
    for _ in range(200):
        v, truth = episode(T, stall_start=T // 2 - stall_len // 2, stall_len=stall_len)
        mined = mined_labels(v)
        rpos = np.array([tvr(v, t) > 0 for t in range(T)])
        agg["mined"].append(mined); agg["Rpos"].append(rpos); agg["truth"].append(truth)
    mined = np.concatenate(agg["mined"]); rpos = np.concatenate(agg["Rpos"]); truth = np.concatenate(agg["truth"])
    C[T] = {"stall_len": stall_len,
            "mined_vs_truth_stall": confusion(mined, truth),
            "R_sign_vs_truth_stall": confusion(rpos, truth),
            "R_sign_vs_mined": confusion(rpos, mined)}
RES["C_noisy_ramp_with_stall"] = C

# ---------------------------------------------------------------- D. 초기 단계: 창 채움(첫 값 복제)의 효과
D = {}
for T in (42, 95):
    v = np.arange(T) / (T - 1)
    D[T] = {"R_sign_positive_steps_in_first_10": [t for t in range(10) if tvr(v, t) > 0]}
RES["D_early_steps_with_padding"] = D

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
print("A. 건강한 선형 궤적")
for T, r in A.items():
    print(f"  T={T:3d} dV/step={r['step_dV']:.4f} 채굴 양성 {r['mined_positive_frac']:.2f}  R>0 {r['R_positive_frac(sign)']:.2f}")
print("플래토가 전부 양성이 되는 최소 길이:", RES["A_plateau_all_positive_min_T"])
print("B.", RES["B_score_range"])
for T, c in C.items():
    print(f"C. T={T}: 채굴 라벨 대 정체 {c['mined_vs_truth_stall']}")
    print(f"          R>0 대 정체 {c['R_sign_vs_truth_stall']}")
    print(f"          R>0 대 채굴 {c['R_sign_vs_mined']}")
print("D.", D)
