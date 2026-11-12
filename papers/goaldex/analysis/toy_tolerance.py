"""토이 B: 성공 허용오차(τ)와 보고된 성공률·예측 실패율의 관계.

1) 무작위 목표의 기준 성공률: Haar 측도에서 θ의 밀도는 (1 - cos θ)/π, 따라서 P(θ < τ) = (τ - sin τ)/π.
   τ = 0.4 rad에서 약 0.34%. Monte Carlo로 확인한다.
2) 예측 실패율 모형: 실제 최종 오차 e ~ |N(0, s²)| (s는 가정), 머리 예측 ê = e + N(0, σ_h²).
   선언 성공은 ê < τ, 예측 실패는 선언 성공 중 e ≥ τ. PFR = P(e ≥ τ | ê < τ).
   논문의 PFR(시뮬레이션 8.80%, 실로봇 12/100)에서 σ_h를 역산해 τ에 따른 값이 어떻게 변하는지 본다.
   s는 논문에 없는 가정이므로 세 값(0.15, 0.25, 0.35 rad)을 모두 낸다.

설계 가정 토이이며 논문 값에 맞추지 않았다. 실행: python3 analysis/toy_tolerance.py (numpy, 1초 미만)
출력: analysis/toy_tolerance_results.json
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import erf

OUT = Path(__file__).resolve().parent / "toy_tolerance_results.json"
rng = np.random.default_rng(1)
TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
S_VALUES = [0.15, 0.25, 0.35]
N = 2_000_000


def chance_formula(tau):
    return (tau - math.sin(tau)) / math.pi


def chance_mc():
    q = rng.normal(size=(N, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    th = 2 * np.arccos(np.clip(np.abs(q[:, 0]), 0, 1))
    return {t: float(np.mean(th < t)) for t in TAUS}


def pfr(sig_h, s, tau, n=N):
    e = np.abs(rng.normal(0, s, n))
    eh = e + rng.normal(0, sig_h, n)
    declared = eh < tau
    if declared.sum() == 0:
        return float("nan")
    return float(np.mean(e[declared] >= tau))


def implied_sigma(target, s, tau):
    """PFR(σ_h) 가 target이 되는 σ_h를 이분법으로 찾는다(단조 증가 가정)."""
    lo, hi = 1e-3, 1.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        # 같은 난수 씨를 써서 단조성을 안정시킨다
        r = np.random.default_rng(7)
        e = np.abs(r.normal(0, s, N))
        eh = e + r.normal(0, mid, N)
        d = eh < tau
        v = float(np.mean(e[d] >= tau))
        if v < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    res = {"setup": {"n": N, "seed": "1 (MC), 7 (fit)", "taus": TAUS, "s_values": S_VALUES}}
    res["chance_formula"] = {str(t): chance_formula(t) for t in TAUS}
    res["chance_mc"] = {str(k): v for k, v in chance_mc().items()}

    # 공식 대 MC 확인 (τ=0.4)
    res["chance_formula_check_tau0.4"] = {"formula": chance_formula(0.4), "mc": res["chance_mc"]["0.4"]}

    # 논문 PFR 역산 (τ = 0.4 기준)
    targets = {"sim_image_goal_8.80pct": 0.0880, "real_12of100_prediction_failures": 0.12}
    implied = {}
    for name, tgt in targets.items():
        implied[name] = {}
        for s in S_VALUES:
            # σ_h → ∞ 일 때 PFR의 상한은 P(e ≥ τ)이다. 이 상한이 목표보다 작으면 도달 불가
            ceiling = float(np.mean(np.abs(rng.normal(0, s, N)) >= 0.4))
            if ceiling < tgt:
                implied[name][f"s={s}"] = {"reachable": False, "ceiling_pfr": ceiling}
                continue
            sig = implied_sigma(tgt, s, 0.4)
            implied[name][f"s={s}"] = {"reachable": True, "implied_sigma_h_rad": sig,
                                        "check_pfr_at_tau0.4": pfr(sig, s, 0.4)}
    res["implied_head_noise_at_tau0.4"] = implied

    # 같은 σ_h에서 τ를 바꿨을 때 PFR, 선언 성공률 (σ_h = 0.17 ≈ 상대 방향 오차 0.136 rad의 평균절대오차를 가우스로 환산한 값)
    sig_ref = 0.136 * math.sqrt(math.pi / 2)
    sweep = {}
    for s in S_VALUES:
        rows = {}
        for tau in TAUS:
            e = np.abs(rng.normal(0, s, N))
            eh = e + rng.normal(0, sig_ref, N)
            d = eh < tau
            rows[str(tau)] = {
                "declared_rate": float(np.mean(d)),
                "PFR": float(np.mean(e[d] >= tau)) if d.any() else None,
                "true_success_given_declared": float(np.mean(e[d] < tau)) if d.any() else None,
            }
        sweep[f"s={s}"] = rows
    res["tau_sweep_sigma_h_ref"] = {"sigma_h_rad": sig_ref, "rows": sweep}

    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
