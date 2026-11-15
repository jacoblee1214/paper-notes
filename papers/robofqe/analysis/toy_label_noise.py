"""토이 2: 종료 성공 라벨(LRM)의 오류가 할인 반환 추정의 절대값과 순위를 어떻게 바꾸는지.
numpy만 쓴다. 실행: python3 analysis/toy_label_noise.py   → analysis/toy_label_noise_results.json (수 초)

모형: FQE가 잘 맞았다고 가정하면 추정 목표는 E[γ^(T-1) r̂]이다(식 5의 종료 항). 종료 라벨 r̂은
- 성공 궤적: TPR = P(r̂=1 | 성공)
- 실패 궤적(시간 제한 H에서 종료): 1 - TNR = P(r̂=1 | 실패)
로 오류가 난다. 그러면 V̂ = TPR·V + (1-TNR)·γ^(H-1)·(1-SR) 이다(닫힌 형태, 아래 MC로 확인).
- LIBERO: γ=0.99, 실패는 H에서 끝난다(코드 rollouts.py: 성공 또는 max_steps). 성공 시점 분포는 토이 1과 같다.
- 실로봇: γ=1이라 V̂ = TPR·SR + (1-TNR)·(1-SR). 시간 항이 없다.
정책 4개(SR 0.84/0.70/0.55/0.30)에서 순위와 절대값을 본다. 논문의 정확도(97.4%, 83.3%)는 클래스별 값이 없어
대칭 오류(TPR = TNR = 정확도)를 기본으로 두고, 비대칭과 정책 의존 오류를 따로 본다.
"""
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_label_noise_results.json"
GAMMA = 0.99
SR_POL = np.array([0.84, 0.70, 0.55, 0.30])
SUITES = {"Spatial": (220, 0.282), "Object": (280, 0.2075), "Goal": (300, 0.3195), "Long": (520, 0.0674)}  # (H, 성공 시 평균 할인)


def episodes(rng, sr, H, mu_c, n):
    """성공 여부와 종료 시점. 성공 시점은 정규분포(평균 μ, 표준편차 0.25 μ)를 1..H-1로 자른 값."""
    succ = rng.random(n) < sr
    t = np.clip(np.round(rng.normal(mu_c, 0.25 * mu_c, n)), 1, H - 1)
    t = np.where(succ, t, H)           # 실패는 H에서 종료
    return succ, t


def mu_for_mean_disc(m, H):
    lo, hi = 2.0, H - 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        t = np.arange(1, H)
        w = np.exp(-0.5 * ((t - mid) / (0.25 * mid)) ** 2)
        w /= w.sum()
        if np.sum(w * GAMMA ** (t - 1)) > m:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def run_pol(rng, H, m, tpr, tnr, n=400_000, gamma=GAMMA):
    """정책별 기대 추정(MC, n 궤적)과 참값을 돌려준다. tpr, tnr은 정책 배열 또는 스칼라."""
    mu = mu_for_mean_disc(m, H)
    v_true, v_hat = [], []
    for k, sr in enumerate(SR_POL):
        a = tpr[k] if np.ndim(tpr) else tpr
        b = tnr[k] if np.ndim(tnr) else tnr
        succ, t = episodes(rng, sr, H, mu, n)
        disc = gamma ** (np.where(succ, t, H) - 1)
        v_true.append(float(np.mean(disc * succ)))
        # 라벨: 성공이면 TPR로 1, 실패이면 1-TNR로 1
        lab = np.where(succ, rng.random(n) < a, rng.random(n) < (1 - b))
        v_hat.append(float(np.mean(disc * lab)))
    return np.array(v_true), np.array(v_hat)


def kendall_tau(x, y):
    n = len(x)
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            c += s > 0
            d += s < 0
    return (c - d) / (n * (n - 1) / 2)


def closed_form(V, SR, H, a, b, gamma=GAMMA):
    return a * V + (1 - b) * gamma ** (H - 1) * (1 - SR)


def main():
    rng = np.random.default_rng(1)
    res = {"model": "V_hat = TPR*V + (1-TNR)*gamma^(H-1)*(1-SR); MC with n=400k per policy",
           "policies_SR": SR_POL.tolist()}

    # (A) LIBERO, 대칭 오류: 논문 정확도 97.4%(LIBERO), 가정 상한
    libero = {}
    for s, (H, m) in SUITES.items():
        libero[s] = {}
        for acc in (0.974, 0.90):
            vt, vh = run_pol(rng, H, m, acc, acc)
            V_cf = m * SR_POL
            bias_cf = closed_form(V_cf, SR_POL, H, acc, acc) - V_cf
            libero[s][f"acc={acc}"] = {
                "V_true_mc": [round(x, 4) for x in vt],
                "bias_relative_pct_mc_mean": round(100 * float(np.mean((vh - vt) / vt)), 2),
                "bias_relative_pct_closed_form_mean": round(100 * float(np.mean(bias_cf / V_cf)), 2),
                "kendall_tau_vs_true": round(float(kendall_tau(vt, vh)), 3),
            }
    res["libero_symmetric"] = libero

    # (B) 실로봇 설정(γ=1, 논문 4.4의 세 정책 SR 58.5/81.0/33.0%): 정확도 83.3%를 대칭으로 두면
    #     γ=1이면 V = SR이고 V̂ = TPR·SR + FPR·(1-SR) 이다(닫힌 형태).
    SR_REAL = np.array([0.585, 0.81, 0.33])
    real = {}
    for acc in (0.833, 0.95):
        V = SR_REAL
        vh = acc * V + (1 - acc) * (1 - V)
        real[f"acc={acc}"] = {
            "V_hat": [round(x, 3) for x in vh],
            "abs_bias_pp": [round(100 * x, 1) for x in (vh - V)],
            "relative_error_pct": [round(100 * x, 1) for x in np.abs(vh - V) / V],
            "mean_relative_error_pct": round(100 * float(np.mean(np.abs(vh - V) / V)), 1),
        }
    # 정책 의존 오류: 33% 정책의 실패만 자주 성공으로 본다(FPR 0.30), 나머지는 FPR 0.05, TPR 0.95
    tpr, fpr = 0.95, np.array([0.05, 0.05, 0.30])
    vh_dep = tpr * SR_REAL + fpr * (1 - SR_REAL)
    real["policy_dependent_FPR"] = {
        "V_true": SR_REAL.tolist(), "V_hat": [round(x, 3) for x in vh_dep],
        "rel_error_pct": [round(100 * x, 1) for x in np.abs(vh_dep - SR_REAL) / SR_REAL],
        "order_preserved": bool(np.all(np.argsort(vh_dep) == np.argsort(SR_REAL))),
    }
    res["real_world_gamma1"] = real

    # (C) 비대칭: 실패를 잘 잡는 검출기(TNR 0.99)와 성공을 자주 놓치는 검출기(TPR 0.75), 그리고 반대
    H, m = SUITES["Spatial"]
    asym = {}
    for name, (a, b) in {"TPR=0.75,TNR=0.99": (0.75, 0.99), "TPR=0.99,TNR=0.75": (0.99, 0.75),
                         "TPR=0.90,TNR=0.90": (0.90, 0.90)}.items():
        vt, vh = run_pol(rng, H, m, a, b)
        asym[name] = {"V_true": [round(x, 3) for x in vt], "V_hat": [round(x, 3) for x in vh],
                      "bias_pp": [round(100 * x, 1) for x in (vh - vt)],
                      "kendall_tau": round(float(kendall_tau(vt, vh)), 3)}
    res["spatial_asymmetric"] = asym

    # (D) 정책 의존 오류: 정책 3(SR 0.55)의 실패를 검출기가 더 자주 속인다(TNR 0.70). 나머지는 TNR 0.97, TPR 0.97
    H, m = SUITES["Spatial"]
    tpr = np.array([0.97, 0.97, 0.97, 0.97])
    tnr = np.array([0.97, 0.97, 0.70, 0.97])
    vt, vh = run_pol(rng, H, m, tpr, tnr)
    tnr_flat = np.full(4, 0.97)
    vt0, vh0 = run_pol(rng, H, m, tpr, tnr_flat)
    res["spatial_policy_dependent"] = {
        "V_true": [round(x, 3) for x in vt], "V_hat_policy_dependent": [round(x, 3) for x in vh],
        "V_hat_uniform_TNR": [round(x, 3) for x in vh0],
        "kendall_tau_policy_dependent": round(float(kendall_tau(vt, vh)), 3),
        "kendall_tau_uniform": round(float(kendall_tau(vt0, vh0)), 3),
    }

    # (E) 필요한 오류 수준: LIBERO Spatial에서 대칭 정확도 acc가 바뀔 때 (닫힌 형태) 상대 편향
    H, m = SUITES["Spatial"]
    V = m * SR_POL
    curve = []
    for acc in (1.0, 0.99, 0.974, 0.95, 0.90, 0.85, 0.833, 0.80):
        bias = closed_form(V, SR_POL, H, acc, acc) - V
        curve.append({"acc": acc, "mean_rel_bias_pct": round(100 * float(np.mean(bias / V)), 2)})
    res["spatial_bias_vs_acc_closed_form"] = curve

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    for s in libero:
        print(s, {k: (v["bias_relative_pct_mc_mean"], v["bias_relative_pct_closed_form_mean"]) for k, v in libero[s].items()})
    print("real", {k: v for k, v in real.items()})
    print("asym", {k: (v["bias_pp"], v["kendall_tau"]) for k, v in asym.items()})
    print("policy-dep", res["spatial_policy_dependent"])
    print("curve", curve)


if __name__ == "__main__":
    main()
