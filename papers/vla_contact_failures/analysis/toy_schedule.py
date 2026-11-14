"""토이 1: Logit-Normal(LN) 시간 샘플링이 노이즈 수준 τ의 질량을 어디로 옮기는지 계산한다(논문 §4.1, 그림 2a, 부록 A.2).

확인하는 주장: (a) LN(m=1.5, s=1)은 Beta(1.5, 1)보다 τ<0.2 질량을 약 6배 준다. (b) 이 이득의 대가로 τ가 큰(고노이즈) 구간의 학습 질량이 크게 줄어든다.
방법: 해석식(scipy) + 몬테카를로로 논문 식 τ = σ(s·z − m)을 직접 뽑아 해석식과 맞춘다. 설계 가정은 없고 논문 식과 Beta(1.5, 1)만 쓴다.
실행: python3 analysis/toy_schedule.py → analysis/toy_schedule_results.json (CPU, 1초 미만)
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "toy_schedule_results.json"
rng = np.random.default_rng(0)
BINS = [(0.0, 0.2), (0.2, 0.8), (0.8, 1.0)]


def beta_cdf(x):
    # 논문 식: τ ~ Beta(1.5, 1). openpi 코드는 0.999*t + 0.001로 바꾸지만 질량 차이는 무시할 만하다.
    return stats.beta.cdf(np.clip(x, 0, 1), 1.5, 1.0)


def ln_cdf(x, m=1.5, s=1.0):
    # τ = σ(s z − m)이면 P(τ ≤ x) = P(z ≤ (logit(x) + m)/s) = Φ((logit x + m)/s)
    x = np.clip(x, 1e-12, 1 - 1e-12)
    return stats.norm.cdf((np.log(x / (1 - x)) + m) / s)


def mass(cdf, lo, hi, **kw):
    return float(cdf(hi, **kw) - cdf(lo, **kw)) if cdf is ln_cdf else float(cdf(hi) - cdf(lo))


def main():
    res = {}
    # (a) 해석식
    beta_m = [mass(beta_cdf, a, b) for a, b in BINS]
    ln_m = [mass(ln_cdf, a, b) for a, b in BINS]
    res["analytic_mass_bins_0_0.2_0.2_0.8_0.8_1"] = {"beta_1.5_1": [round(x, 4) for x in beta_m],
                                                      "LN_m1.5_s1": [round(x, 4) for x in ln_m]}
    res["ratio_tau_lt_0.2"] = round(ln_m[0] / beta_m[0], 2)
    res["ratio_tau_gt_0.8"] = round(ln_m[2] / beta_m[2], 4)

    # (b) 몬테카를로로 논문 식을 직접 뽑아 확인 (τ = σ(z − 1.5), z ~ N(0,1))
    n = 1_000_000
    z = rng.standard_normal(n)
    tau_ln = 1 / (1 + np.exp(-(z - 1.5)))
    tau_beta = rng.beta(1.5, 1.0, n)
    res["mc_mass_bins_LN"] = [round(float(np.mean((tau_ln >= a) & (tau_ln < b))), 4) for a, b in BINS]
    res["mc_mass_bins_beta"] = [round(float(np.mean((tau_beta >= a) & (tau_beta < b))), 4) for a, b in BINS]
    res["mc_ratio_tau_lt_0.2"] = round(float(np.mean(tau_ln < 0.2) / np.mean(tau_beta < 0.2)), 2)

    # (c) 민감도: m과 s를 바꾸면 질량이 어떻게 움직이는가 (논문 v1 부록 m 스윕과 같은 식)
    grid = []
    for s in [0.5, 1.0, 1.5]:
        for m in [-1.5, -0.5, 0.5, 1.5]:
            lo = ln_cdf(0.2, m=m, s=s)
            hi8 = 1 - ln_cdf(0.8, m=m, s=s)
            grid.append({"m": m, "s": s, "P_tau_lt_0.2": round(float(lo), 4),
                         "P_tau_gt_0.8": round(float(hi8), 4),
                         "ratio_vs_beta_lt_0.2": round(float(lo / beta_m[0]), 2)})
    res["sensitivity_m_s"] = grid

    # (d) 고노이즈 구간: 추론은 τ=1에서 시작하므로 τ>0.8 학습 질량이 얼마나 남는지 본다
    res["note"] = ("LN(m=1.5, s=1)은 τ>0.8 질량이 Beta의 약 1/150이다. 논문 부록 A.4 진단은 τ 0.1–1.0에서 "
                   "흐름 오차를 재지만 보고는 낮은 노이즈(<5 mm) 구간만 쓴다. 고노이즈 구간 오차는 보고되지 않았다.")

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("wrote", OUT)
    print("analytic beta:", res["analytic_mass_bins_0_0.2_0.2_0.8_0.8_1"]["beta_1.5_1"])
    print("analytic LN  :", res["analytic_mass_bins_0_0.2_0.2_0.8_0.8_1"]["LN_m1.5_s1"])
    print("ratio tau<0.2:", res["ratio_tau_lt_0.2"], " MC:", res["mc_ratio_tau_lt_0.2"])
    print("ratio tau>0.8 (LN/Beta):", res["ratio_tau_gt_0.8"])
    print("MC bins LN:", res["mc_mass_bins_LN"], "beta:", res["mc_mass_bins_beta"])
    for g in grid:
        print(g)


if __name__ == "__main__":
    main()
