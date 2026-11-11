"""VNB toy (CPU numpy, 단일 스레드, 수 초): 접촉 마찰 믿음의 모양이 선택되는 파지를 바꾸는가.

가정(논문과 다른 점은 모두 명시):
  - 잠재 변수는 물체 마찰 µ_o 하나. 믿음은 논문 bimodal 체제와 같은 두 봉우리
    0.5 N(0.18, 0.03²) + 0.5 N(1.0, 0.05²)를 쓴다(§V-D).
    단일 가우시안 믿음은 같은 평균·분산으로 적합한다(Gauss 기준선의 K=1에 해당).
  - 유효 마찰 µ_eff 는 두 규칙을 비교한다.
      (a) 논문 식 (22): µ_eff = sqrt(µ_f · µ_o)
      (b) MuJoCo 문서(modeling, Contact parameters)의 friction 혼합 규칙: 두 geom 값의 원소별 최댓값
  - 파지 후보 g: 필요 마찰 r_g 이상이면 성공, 아니면 실패. 비용 C_g(µ) = c_g + L·1[µ_eff < r_g].
    L = 5 (실패 벌점), c_g 는 파지 노력(비용)이다. 논문의 C(θ,a) 를 단순화한 것.
  - 선택은 CVaR_β(C_g) 최소 (λc=1, 논문 Eq. 11–12 의 샘플 근사). 비교 대상은 기대비용(λc=0).
출력: analysis/cvar_toy_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "cvar_toy_results.json"
RNG = np.random.default_rng(20260424)
N = 200_000
L_FAIL = 5.0

# 파지 후보: (이름, 필요 마찰 r_g, 파지 노력 c_g)
GRASPS = [
    ("A 넓은 파지", 0.15, 1.0),
    ("B 빠른 파지", 0.50, 0.2),
    ("C 중간 파지", 0.30, 0.5),
    ("D 강한 파지", 0.90, 0.7),
]


def sample_mu_o(w_low, n, rng):
    """두 봉우리 혼합에서 µ_o 표본. w_low 는 낮은 마찰(0.18) 봉우리의 질량."""
    low = rng.random(n) < w_low
    mu = np.where(low, rng.normal(0.18, 0.03, n), rng.normal(1.0, 0.05, n))
    return np.clip(mu, 0.01, None)


def mu_eff(mu_o, rule, mu_f=1.0):
    if rule == "geom":          # 논문 식 (22)
        return np.sqrt(mu_f * mu_o)
    if rule == "max":           # MuJoCo 원소별 최댓값 (mu_f 와 µ_o 중 큰 값)
        return np.maximum(mu_f, mu_o)
    raise ValueError(rule)


def cost(mu_eff_vals, r_g, c_g):
    return c_g + L_FAIL * (mu_eff_vals < r_g).astype(float)


def cvar(c, beta):
    """Rockafellar–Uryasev 형태의 샘플 근사: VaR 분위수 + 꼬리 평균 초과분 / (1-β)."""
    eta = np.quantile(c, beta)
    return eta + np.mean(np.maximum(c - eta, 0.0)) / (1.0 - beta)


def choose(mu_samples, beta, rule, mu_f=1.0, use_cvar=True):
    scores = []
    for _, r, c_g in GRASPS:
        cc = cost(mu_eff(mu_samples, rule, mu_f), r, c_g)
        scores.append(cvar(cc, beta) if use_cvar else cc.mean())
    return int(np.argmin(scores)), [round(float(s), 3) for s in scores]


def gaussian_fit(mu_samples):
    m, s = mu_samples.mean(), mu_samples.std()
    return np.clip(m + s * RNG.standard_normal(N), 0.01, None)


def part1_multimodal(out):
    """(1) 믿음 모양(GMM 대 단일 가우시안)과 목적(CVaR 대 기대비용)에 따른 선택."""
    global RNG
    rows = []
    for beta in (0.5, 0.9, 0.95, 0.99):
        for w in (0.5,):
            mu_gmm = sample_mu_o(w, N, RNG)
            mu_g = gaussian_fit(mu_gmm)
            i_gmm, s_gmm = choose(mu_gmm, beta, "geom", use_cvar=True)
            i_g, s_g = choose(mu_g, beta, "geom", use_cvar=True)
            i_gmean, _ = choose(mu_g, beta, "geom", use_cvar=False)
            # 실제 실패 확률(진짜 믿음 기준)
            true_fail_list = []
            for _, r, _ in GRASPS:
                true_fail_list.append(round(float(np.mean(mu_eff(mu_gmm, "geom") < r)), 3))
            rows.append({
                "beta": beta, "w_low": w, "mu_f": 1.0, "rule": "geom (식 22)",
                "GMM_CVaR_choice": GRASPS[i_gmm][0], "GMM_CVaR_scores": s_gmm,
                "Gauss_CVaR_choice": GRASPS[i_g][0], "Gauss_CVaR_scores": s_g,
                "Gauss_expected_cost_choice": GRASPS[i_gmean][0],
                "true_failure_prob_per_grasp_(A,B,C,D)": true_fail_list,
            })
    out["part1_beta_sweep_w0.5_muf1.0_geom"] = rows

    # 낮은 봉우리 질량 w 에 따른 선택 전이. 단일 가우시안 적합은 표본 잡음에 민감하므로 10개 시드의 최빈값과 비율을 보고
    sweep = []
    for beta in (0.9, 0.95):
        for w in np.linspace(0.0, 1.0, 11):
            picks = {"GMM_CVaR": [], "Gauss_CVaR": [], "Gauss_mean": []}
            for seed in range(10):
                RNG = np.random.default_rng(1000 + seed)
                mu_gmm = sample_mu_o(w, N // 2, RNG)
                mu_g = gaussian_fit(mu_gmm)
                picks["GMM_CVaR"].append(GRASPS[choose(mu_gmm, beta, "geom")[0]][0])
                picks["Gauss_CVaR"].append(GRASPS[choose(mu_g, beta, "geom")[0]][0])
                picks["Gauss_mean"].append(GRASPS[choose(mu_g, beta, "geom", use_cvar=False)[0]][0])
            row = {"beta": beta, "w_low": round(float(w), 2)}
            for k, v in picks.items():
                vals, cnt = np.unique(v, return_counts=True)
                row[k] = f"{vals[np.argmax(cnt)]} ({cnt.max()}/10)"
            sweep.append(row)
    RNG = np.random.default_rng(20260424)
    out["part1_w_sweep_muf1.0_geom_10seeds"] = sweep


def part2_mu_f(out):
    """(2) 계획의 µ_f 민감도. 같은 믿음, 혼합 규칙·µ_f 를 바꾼다. β=0.95, w=0.5."""
    mu_o = sample_mu_o(0.5, N, RNG)
    rows = []
    for rule, mu_f in [("geom", 0.35), ("geom", 1.0), ("max", 1.0)]:
        i, s = choose(mu_o, 0.95, rule, mu_f)
        eff = mu_eff(mu_o, rule, mu_f)
        rows.append({"rule": rule, "mu_f": mu_f,
                     "mu_eff_5-95pct": [round(float(np.quantile(eff, q)), 3) for q in (0.05, 0.95)],
                     "chosen": GRASPS[i][0], "CVaR_scores": s,
                     "failure_prob_per_grasp_(A,B,C,D)": [round(float(np.mean(eff < r)), 3) for _, r, _ in GRASPS]})
    out["part2_mu_f_sensitivity"] = rows

    # 마찰 낮추기 교란(µ→0.05~0.15)이 같은 규칙에서 무엇을 바꾸는가
    drop = []
    for rule in ("geom", "max"):
        for mu_drop in (0.05, 0.10, 0.15):
            eff = mu_eff(np.array([mu_drop]), rule, mu_f=1.0)
            drop.append({"rule": rule, "mu_object_after_drop": mu_drop,
                         "mu_eff_if_mu_f=1.0": round(float(eff[0]), 3)})
    out["part2_friction_drop_effect"] = drop


def part3_softplus_bias(out):
    """(3) 논문 Eq. (12) 의 softplus 근사 CVaR 편향. κ=5, N=256, 비용 ~ N(0, 1)."""
    kappa = 5.0
    rows = []
    for beta in (0.5, 0.9, 0.95, 0.99):
        diffs, exact_vals = [], []
        for _ in range(400):
            c = RNG.standard_normal(256)
            eta = np.quantile(c, beta)
            exact = eta + np.mean(np.maximum(c - eta, 0.0)) / (1.0 - beta)
            soft = eta + np.mean(np.log1p(np.exp(kappa * (c - eta))) / kappa) / (1.0 - beta)
            diffs.append(soft - exact)
            exact_vals.append(exact)
        rows.append({"beta": beta, "exact_CVaR_mean": round(float(np.mean(exact_vals)), 3),
                     "softplus_minus_exact_mean": round(float(np.mean(diffs)), 4),
                     "softplus_minus_exact_sd": round(float(np.std(diffs)), 4),
                     "worst_case_bound_log2/(kappa*(1-beta))": round(float(np.log(2) / (kappa * (1 - beta))), 3)})
    out["part3_softplus_bias"] = rows


def part4_eq8_sampling(out):
    """(4) 논문 Eq.(8)–(9) 샘플은 혼합분포의 표본인가. 두 봉우리(0.18, 1.0)의 간격 [0.4, 0.8] 질량으로 확인.
    Eq.(8): θ = Σ_k ζ_k (µ_k + σ_k ε_k), ζ 는 Gumbel-softmax (코드는 τ=0.5 고정, rsample 참조).
    참조: 실제 혼합에서 간격 질량은 0 에 가깝다(두 봉우리 사이 밀도 거의 없음)."""
    mu = np.array([0.18, 1.0])
    sd = np.array([0.03, 0.05])
    logits = np.zeros(2)
    rows = []
    for tau in (1.0, 0.5, 0.1):
        rng = np.random.default_rng(7)
        n = 200_000
        g = -np.log(-np.log(rng.random((n, 2))))
        z = np.exp((logits + g) / tau)
        zeta = z / z.sum(axis=1, keepdims=True)
        eps = rng.standard_normal((n, 2))
        theta = (zeta * (mu + sd * eps)).sum(axis=1)
        gap = np.mean((theta > 0.4) & (theta < 0.8))
        rows.append({"tau": tau, "Eq8_fraction_in_gap_[0.4,0.8]": round(float(gap), 4),
                     "Eq8_mean": round(float(theta.mean()), 3)})
    # 실제 혼합에서 같은 구간 질량
    rng = np.random.default_rng(7)
    comp = rng.random(200_000) < 0.5
    mix = np.where(comp, rng.normal(0.18, 0.03, 200_000), rng.normal(1.0, 0.05, 200_000))
    rows.append({"tau": "true mixture", "Eq8_fraction_in_gap_[0.4,0.8]": round(float(np.mean((mix > 0.4) & (mix < 0.8))), 4),
                 "Eq8_mean": round(float(mix.mean()), 3)})
    out["part4_eq8_gap_mass"] = rows


def main():
    out = {}
    part1_multimodal(out)
    part2_mu_f(out)
    part3_softplus_bias(out)
    part4_eq8_sampling(out)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
