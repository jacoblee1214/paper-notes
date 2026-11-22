"""토이: 논문 5.3의 "관측 차이 > 2.49 x SE" 판정 규칙이 시드 6-8개에서 어떻게 동작하는지 몬테카를로로 본다.

실행: python3 analysis/toy_rule_check.py   → analysis/toy_rule_check_results.json (수 초, numpy, scipy)

설정
  시드별 참 성공확률 p_s ~ Beta(평균 mu, 분산 sigma2), 시드마다 n_s번 시도해 이항 표본으로 시드 점수를 만든다.
  mu, 시드 수 K, 시드당 시도 수 n_s는 논문 5.1·표 2 값, sigma2는 analysis/recompute.py의 역산(같은 분산 가정)이다.
  두 로봇을 독립 시드로 비교하고 Welch SE 를 쓴다 (논문 5.3과 같은 SE).
확인하는 것
  (a) 귀무(참 차이 0, 두 로봇 평균을 Stretch 평균으로 맞춤)에서 "차이 > 2.49 SE" 규칙(한쪽)이 틀리게 "구분 가능"이라
      할 확률. 정규 가정이면 한쪽 0.64%(양쪽 1.29%). rule_t 는 z 대신 중심 t 분위수 (t.95 + t.80)를 쓴 규칙
  (b) 참 차이 = 관측 차이일 때 규칙이 "구분 가능"이라 할 확률 (검정력)
  (c) Welch 95% 구간의 포함 확률
  (d) K = 30으로 늘리면 (a)가 정규 가정 값에 가까워지는지
한계: 시드 분산은 두 로봇이 같다고 가정, Beta 분포 가정, 시드 수와 시도 수 고정. 참고용 토이이며 논문 데이터의 재현이 아니다.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("toy_rule_check_results.json")
RNG = np.random.default_rng(20261123)
ZSUM = stats.norm.ppf(0.95) + stats.norm.ppf(0.80)
REPS = 40000

# (평균 A, 평균 B, K_A, K_B, n_A, n_B, sigma2)  A = Spot, B = Stretch. n 은 시드당 시도 수(반올림)
SETUPS = {
    "door": dict(muA=0.7109, muB=0.1802, KA=6, KB=7, nA=10, nB=19, s2=0.0474),
    "drawer": dict(muA=0.8792, muB=0.6352, KA=6, KB=8, nA=15, nB=36, s2=0.0225),
}


def seed_scores(mu, s2, K, n, size):
    s2 = min(s2, 0.95 * mu * (1 - mu))
    k = mu * (1 - mu) / s2 - 1
    a, b = mu * k, (1 - mu) * k
    p = RNG.beta(a, b, size=(size, K))
    return RNG.binomial(n, p) / n


def run(muA, muB, KA, KB, nA, nB, s2, true_gap_override=None):
    if true_gap_override is not None:
        muA, muB = muB + true_gap_override, muB
    xa = seed_scores(muA, s2, KA, nA, REPS)
    xb = seed_scores(muB, s2, KB, nB, REPS)
    ma, mb = xa.mean(1), xb.mean(1)
    va, vb = xa.var(1, ddof=1) / KA, xb.var(1, ddof=1) / KB
    se = np.sqrt(va + vb)
    gap = ma - mb
    with np.errstate(invalid="ignore", divide="ignore"):
        nu = (va + vb) ** 2 / (va ** 2 / (KA - 1) + vb ** 2 / (KB - 1))
    ok = se > 0
    rule = (gap > ZSUM * se) & ok
    tcrit = stats.t.ppf(0.975, np.where(ok, nu, 1))
    sig = (np.abs(gap) > tcrit * se) & ok
    cover = (np.abs(gap - (muA - muB)) <= tcrit * se) & ok
    # t 분위수로 바꾼 규칙 (중심 t 근사): (t_.95,nu + t_.80,nu) SE
    rule_t = (gap > (stats.t.ppf(0.95, np.where(ok, nu, 1)) + stats.t.ppf(0.80, np.where(ok, nu, 1))) * se) & ok
    return {"rule_z_yes": float(rule.mean()), "rule_t_yes": float(rule_t.mean()),
            "welch_p_lt_05": float(sig.mean()), "ci95_coverage": float(cover.mean()),
            "mean_se": float(se.mean()), "median_df": float(np.median(nu[ok]))}


def main():
    res = {"reps": REPS, "zsum": float(ZSUM), "normal_one_sided_p_at_zsum": float(stats.norm.sf(ZSUM))}
    for name, cfg in SETUPS.items():
        r = {}
        r["null_true_gap0"] = run(**cfg, true_gap_override=0.0)
        r["alt_true_gap_observed"] = run(**cfg)
        big = dict(cfg, KA=30, KB=30)
        r["null_K30"] = run(**big, true_gap_override=0.0)
        r["alt_K30"] = run(**big)
        res[name] = r
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
