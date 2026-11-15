"""토이 1: 정답(시뮬레이터 몬테카를로)의 표본 노이즈가 측정 가능한 정확도와 순위 상관의 천장을 정한다.
numpy만 쓴다. 실행: python3 analysis/toy_gt_noise.py   → analysis/toy_gt_noise_results.json (수 초)

가정(논문에 없는 값은 모두 표시):
- 정답 V^π = γ^(T-1) R 의 평균 (식 1). 성공 여부는 Bernoulli(SR), 성공 시점 T는 정규분포(평균 μ, 표준편차 c·μ)를 H-1 이하로 자른 분포.
- μ는 V/SR = 성공 궤적의 평균 γ^(T-1)이 표 1 캡션 값과 맞도록 정했다(이분 탐색). c는 0.15, 0.25, 0.35로 바꿔 본다.
- 정답은 LIBERO에서 궤적 200개(논문 4.1)로 추정한다. CALVIN과 실로봇은 다루지 않는다.
"""
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_gt_noise_results.json"
GAMMA = 0.99
N_GT = 200                      # 정답 몬테카를로 궤적 수 (논문 4.1)
SUITES = {  # V (표 1 캡션), SR, H (논문 4.1, 코드 rollouts.py MAX_STEPS와 같음)
    "Spatial": dict(V=0.237, SR=0.84, H=220),
    "Object": dict(V=0.166, SR=0.80, H=280),
    "Goal": dict(V=0.246, SR=0.77, H=300),
    "Long": dict(V=0.029, SR=0.43, H=520),
}
REPORTED_ARE = {"Spatial": 3.40, "Object": 4.56, "Goal": 2.34, "Long": 5.63}   # RoboFQE, 표 1


def time_pmf(mu, c, H):
    """성공 시점 T in 1..H-1 의 pmf (정규분포를 자름)."""
    t = np.arange(1, H)
    w = np.exp(-0.5 * ((t - mu) / (c * mu)) ** 2)
    return t, w / w.sum()


def mean_disc(mu, c, H):
    t, p = time_pmf(mu, c, H)
    return float(np.sum(p * GAMMA ** (t - 1)))


def solve_mu(target, c, H):
    """E[γ^(T-1) | 성공] = target 이 되는 μ를 이분 탐색한다 (γ^(T-1)은 μ가 커질수록 줄어든다)."""
    lo, hi = 2.0, H - 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if mean_disc(mid, c, H) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def episode_stats(V, SR, H, c):
    """한 궤적의 X = 성공이면 γ^(T-1), 아니면 0. E[X]=V 와 E[X^2]를 돌려준다."""
    mu = solve_mu(V / SR, c, H)
    t, p = time_pmf(mu, c, H)
    m1 = SR * np.sum(p * GAMMA ** (t - 1))
    m2 = SR * np.sum(p * GAMMA ** (2 * (t - 1)))
    return mu, m1, m2


def main():
    rng = np.random.default_rng(0)
    res = {"assumptions": {"N_GT": N_GT, "gamma": GAMMA, "timing": "truncated normal, sd = c*mu", "c_values": [0.15, 0.25, 0.35]}}

    # (A) 정답 노이즈의 상대 표준오차와 측정 바닥(floor)
    floors = {}
    for s, d in SUITES.items():
        floors[s] = {}
        for c in (0.15, 0.25, 0.35):
            mu, m1, m2 = episode_stats(d["V"], d["SR"], d["H"], c)
            var = m2 - m1 ** 2
            se = math.sqrt(var / N_GT)
            rel = se / d["V"]
            # 몬테카를로 확인: 정답 추정을 여러 번 반복한 표준편차
            draws = 4000
            ok = rng.random((draws, N_GT)) < d["SR"]
            t_all = np.clip(np.round(rng.normal(mu, c * mu, size=(draws, N_GT))), 1, d["H"] - 1)
            x = ok * GAMMA ** (t_all - 1)
            est = x.mean(axis=1)
            floors[s][f"c={c}"] = {
                "mu_steps": round(mu, 1),
                "SE_GT_abs": round(se, 5),
                "SE_GT_rel_pct": round(100 * rel, 1),
                "MC_SE_rel_pct": round(100 * est.std(ddof=1) / d["V"], 1),
                "ARE_floor_mean_abs_pct": round(100 * 0.798 * rel, 1),   # E|N(0,σ)| = 0.798 σ
                "RoboFQE_ARE_reported": REPORTED_ARE[s],
            }
    res["gt_noise_floor"] = floors

    # (B) 순위 천장: 체크포인트 4개의 SR이 주어졌을 때 정답 200궤적으로 순위를 얼마나 맞추나
    spreads = {
        "close (SR 0.84/0.80/0.77/0.73)": [0.84, 0.80, 0.77, 0.73],
        "medium (SR 0.84/0.70/0.55/0.30)": [0.84, 0.70, 0.55, 0.30],
        "wide (SR 0.90/0.60/0.30/0.05)": [0.90, 0.60, 0.30, 0.05],
    }
    rank = {}
    for s, d in SUITES.items():
        rank[s] = {}
        c = 0.25
        mu, m1, m2 = episode_stats(d["V"], d["SR"], d["H"], c)
        m = d["V"] / d["SR"]                      # 성공 시 평균 할인 (정책 간 같다고 가정)
        for name, srs in spreads.items():
            srs = np.array(srs)
            v_true = srs * m
            # 정책별 정답 표준오차: 같은 시점 분포라고 가정
            ex2 = srs * (m2 / d["SR"])            # E[X^2|정책] = SR * E[γ^(2(T-1))|성공]
            var_k = ex2 - v_true ** 2
            se_k = np.sqrt(np.maximum(var_k, 0) / N_GT)
            draws = 4000
            noisy = v_true[None, :] + se_k[None, :] * rng.standard_normal((draws, len(srs)))
            nc = noisy - noisy.mean(axis=1, keepdims=True)
            tc = v_true - v_true.mean()
            r = (nc @ tc) / (np.sqrt((nc ** 2).sum(axis=1)) * np.sqrt((tc ** 2).sum()))
            # 참값에서 가장 가까운 인접 두 정책: 정답 추정이 순서를 맞출 확률 Φ(Δ / sqrt(se_i² + se_j²))
            order = np.argsort(-v_true)
            gaps = [(v_true[order[k]] - v_true[order[k + 1]], order[k], order[k + 1]) for k in range(len(order) - 1)]
            dgap, i, j = min(gaps)
            z = dgap / math.sqrt(se_k[i] ** 2 + se_k[j] ** 2)
            pair_acc_closest = 0.5 * (1 + math.erf(z / math.sqrt(2)))
            rank[s][name] = {
                "V_true": [round(x, 3) for x in v_true],
                "SE_GT_per_policy_abs": [round(x, 4) for x in se_k],
                "ceiling_r_median_of_noisy_GT_vs_true": round(float(np.median(r)), 3),
                "closest_pair_gap_in_V": round(float(dgap), 3),
                "closest_pair_correct_order_prob": round(float(pair_acc_closest), 3),
            }
    res["rank_ceiling"] = rank
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    for s in SUITES:
        print(s, {k: (v["SE_GT_rel_pct"], v["ARE_floor_mean_abs_pct"], v["MC_SE_rel_pct"]) for k, v in floors[s].items()})
    for s in SUITES:
        for name in spreads:
            print(s, name, rank[s][name]["ceiling_r_median_of_noisy_GT_vs_true"], rank[s][name]["closest_pair_correct_order_prob"])


if __name__ == "__main__":
    main()
