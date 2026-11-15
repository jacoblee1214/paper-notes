"""토이 2: FPO 대리 비율 exp(L_old - L_new)가 정확한 가능도 비율을 얼마나 따라가는가.

목적: 논문 식 (3), (8), (9)의 핵심 가정 "CFM 손실 차이가 로그 가능도 차이의 근사"를 1차원 가우시안에서 확인한다.
      또 N_mc(샘플 수)와 per-sample / per-action 평균이 대리 비율의 분산과 클리핑 빈도에 주는 영향을 본다.
      논문 결과의 재현이 아니다. 정확한 장(field)을 쓰는 이상적 설정이다.

설정(가정):
- 행동 a는 옛 정책 N(0, s^2)에서 뽑는다(on-policy). s = 0.3.
- 새 정책은 평균이 delta 이동한 N(delta, s^2). delta in {0.05, 0.1, 0.2}.
- 각 정책의 속도장은 해당 가우시안의 정답 장이다(본문 toy_euler_steps.py와 같은 닫힌 형식).
  v_mu(x, t) = (x - (1-t) mu)(t - (1-t) s^2)/V(t) - mu,  V(t) = (1-t)^2 s^2 + t^2.
- CFM 손실(논문 식 (6)-(7), 코드 actor_critic.py의 u 모드, 제곱 오차):
  L(a; mu) = || v_mu(x_t, t) - (eps - a) ||^2,  x_t = t*eps + (1-t)*a,  eps ~ N(0,1), t ~ U(0,1).
  같은 (eps, t)를 옛 정책과 새 정책에 쓴다(논문 식 (9)의 샘플별 짝).
- 정확한 로그 가능도 비율: log p_new(a) - log p_old(a).

측정:
- 대리 비율의 기댓값(M = 4000개 (eps,t) 평균): E[L_old - L_new | a]을 정확한 로그 비율과 비교. 기울기, 상관계수.
- 분산: 한 행동 a에서 N_mc in {1, 4, 16, 64}개 (eps,t) 쌍으로 만든 로그 비율의 표준편차.
  per-sample은 샘플별 비율의 평균 로그가 아니라 샘플마다 비율을 따로 만든 뒤 그 비율을 쓴다.
  여기서는 분산 비교를 위해 두 방식의 로그 비율을 각각 기록한다.
- 클리핑 빈도: |ratio - 1| > 0.05(논문 클리핑 파라미터 0.05, 표 A.1)인 비율의 비중. 각 방식에 대해.

실행: python3 -I analysis/toy_ratio_vs_likelihood.py  (numpy만, 1 프로세스, 수 초)
결과: analysis/toy_ratio_vs_likelihood_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_ratio_vs_likelihood_results.json"
S = 0.3
DELTAS = [0.05, 0.1, 0.2]
N_ACT = 2000
M_POP = 4000
NMC_LIST = [1, 4, 16, 64]
CLIP = 0.05


def field(x, t, mu, s):
    V = (1 - t) ** 2 * s**2 + t**2
    return (x - (1 - t) * mu) * (t - (1 - t) * s**2) / V - mu


def cfm_loss(a, eps, t, mu, s=S):
    """a: (A,1), eps,t: (A,M). 반환: (A,M) 제곱 CFM 손실."""
    x = t * eps + (1 - t) * a
    v = field(x, t, mu, s)
    return (v - (eps - a)) ** 2


def logp(a, mu, s):
    return -0.5 * ((a - mu) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)


def main():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, S, size=(N_ACT, 1))
    res = {"setup": {"s": S, "deltas": DELTAS, "N_act": N_ACT, "M_pop": M_POP, "Nmc": NMC_LIST,
                     "clip": CLIP, "seed": 1,
                     "loss": "squared CFM, x_t = t*eps + (1-t)*a, target eps - a, t~U(0,1)"},
           "by_delta": {}}
    for delta in DELTAS:
        eps = rng.standard_normal((N_ACT, M_POP))
        t = rng.uniform(size=(N_ACT, M_POP))
        d_sample = cfm_loss(a, eps, t, 0.0, S) - cfm_loss(a, eps, t, delta, S)  # L_old - L_new, 샘플별
        surrogate_mean = d_sample.mean(axis=1)  # 기대 대리 로그 비율
        exact = logp(a[:, 0], delta, S) - logp(a[:, 0], 0.0, S)

        corr = float(np.corrcoef(surrogate_mean, exact)[0, 1])
        slope = float(np.polyfit(exact, surrogate_mean, 1)[0])
        resid = surrogate_mean - slope * exact
        # 기울기 1이면 대리 로그 비율이 정확한 로그 비율과 같은 크기라는 뜻
        var_rows = []
        for nmc in NMC_LIST:
            # 샘플별: 각 쌍의 비율 exp(d_i)을 따로 만든 뒤 클리핑 판정. 로그 비율 분산은 d_i의 분산.
            per_sample_logratio = d_sample[:, :nmc]
            # per-action: 손실을 nmc개 평균한 뒤 비율을 만든다(vanilla FPO, 식 (8))
            per_action_logratio = d_sample[:, :nmc].mean(axis=1)
            ps_ratio = np.exp(per_sample_logratio)
            pa_ratio = np.exp(per_action_logratio)
            # 한 행동 a 안의 샘플 간 표준편차를 평균낸 값(같은 a에서 비율이 얼마나 흔들리는지)
            var_rows.append({
                "Nmc": nmc,
                "per_sample_logratio_sd_within_a": float(np.sqrt(per_sample_logratio.var(axis=1, ddof=1)).mean()) if nmc > 1 else float("nan"),
                "per_action_logratio_sd_across_a": float(per_action_logratio.std(ddof=1)),
                "per_sample_clip_frac": float((np.abs(ps_ratio - 1) > CLIP).mean()),
                "per_action_clip_frac": float((np.abs(pa_ratio - 1) > CLIP).mean()),
            })
        res["by_delta"][str(delta)] = {
            "exact_logratio_sd": float(exact.std(ddof=1)),
            "exact_logratio_mean": float(exact.mean()),
            "surrogate_mean_of_means": float(surrogate_mean.mean()),
            "corr_surrogate_exact": corr,
            "slope_surrogate_on_exact": slope,
            "resid_sd": float(resid.std(ddof=1)),
            "per_sample_single_logratio_sd_within_a": float(d_sample.std(axis=1, ddof=1).mean()),
            "variance_by_Nmc": var_rows,
        }
        print(f"delta={delta}: corr={corr:.4f} slope={slope:.3f} resid_sd={resid.std(ddof=1):.4f} "
              f"exact_sd={exact.std(ddof=1):.4f} single-sample sd={d_sample.std(axis=1, ddof=1).mean():.4f}")
        for r in var_rows:
            print(f"   Nmc={r['Nmc']:3d} per-action sd={r['per_action_logratio_sd_across_a']:.4f} "
                  f"clip ps={r['per_sample_clip_frac']:.3f} pa={r['per_action_clip_frac']:.3f}")
    # 척도 점검: 기울기가 행동 척도 s에 따라 어떻게 바뀌는가 (delta는 0.1*s/0.3, 같은 상대 이동).
    # 결과: 기울기 ≈ (pi/2)*s. s^2가 아니라 s에 비례한다(eps는 단위 분산, a는 척도 s라서 손실의 단위가 섞임).
    sweep = []
    for s_ in [0.1, 0.3, 1.0]:
        rng_s = np.random.default_rng(2)
        a_s = rng_s.normal(0.0, s_, size=(1000, 1))
        delta_s = 0.1 * s_ / 0.3
        eps_s = rng_s.standard_normal((1000, M_POP))
        t_s = rng_s.uniform(size=(1000, M_POP))
        d_s = (cfm_loss(a_s, eps_s, t_s, 0.0, s_) - cfm_loss(a_s, eps_s, t_s, delta_s, s_)).mean(axis=1)
        ex_s = logp(a_s[:, 0], delta_s, s_) - logp(a_s[:, 0], 0.0, s_)
        slope_s = float(np.polyfit(ex_s, d_s, 1)[0])
        sweep.append({"s": s_, "slope": slope_s, "slope_over_s": slope_s / s_,
                      "corr": float(np.corrcoef(d_s, ex_s)[0, 1])})
        print(f"scale sweep s={s_}: slope={slope_s:.4f} slope/s={slope_s / s_:.3f} corr={sweep[-1]['corr']:.5f}")
    res["scale_sweep"] = sweep
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
