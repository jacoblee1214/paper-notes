"""토이 2: 40회 롤아웃 두 팔의 검정에서 '유의하다'는 결론이 얼마나 믿을 만한지 본다(논문 §5, 부록 A.5).

확인하는 주장: 단측 Fisher 검정, 보정 없는 p, 과제·방법당 40회라는 설계에서
  (a) 표 값이 실제 차이를 과장하는가(유의한 결과만 골랐을 때의 크기 편향),
  (b) 여러 비교에서 우연히 p<.05가 나올 확률(귀무 가설 아래 FWER),
  (c) 실제 차이가 있을 때 검출 확률(검정력).
설계 가정: 기준 성공률 0.30(표의 π0.5 plug 근처), 참 차이 0.20, 시행 40회, 비교 20개, 2000회 반복.
실행: python3 analysis/toy_power.py → analysis/toy_power_results.json (CPU, 수 초)
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "toy_power_results.json"
N = 40
REPS = 2000
rng = np.random.default_rng(1)


def p_greater(k1, k0):
    # 배열 입력을 받기 위해 루프를 쓴다(표 40칸짜리라 빠르다)
    out = np.empty(len(k1))
    for i, (a, b) in enumerate(zip(k1, k0)):
        out[i] = stats.fisher_exact([[a, N - a], [b, N - b]], alternative="greater")[1]
    return out


def main():
    res = {"n_per_arm": N, "reps": REPS}

    # (c)+(a): 참 차이 0.20(0.30 -> 0.50)에서 검정력과 유의한 표본의 추정 차이
    k_base = rng.binomial(N, 0.30, REPS)
    k_alt = rng.binomial(N, 0.50, REPS)
    p = p_greater(k_alt, k_base)
    sig = p <= 0.05
    res["power_true_diff_20pp"] = round(float(sig.mean()), 3)
    est = (k_alt - k_base)[sig] / N * 100
    res["mean_observed_diff_given_significant_pp"] = round(float(est.mean()), 1)
    res["true_diff_pp"] = 20.0
    res["winners_curse_bias_pp"] = round(float(est.mean() - 20.0), 1)

    # 참 차이 0.45 (표의 π0.5 button 12.5 -> FACT 57.5 수준이 아니라 기준 0.125 -> 0.575 사례)
    k_base2 = rng.binomial(N, 0.125, REPS)
    k_alt2 = rng.binomial(N, 0.575, REPS)
    p2 = p_greater(k_alt2, k_base2)
    res["power_true_diff_45pp_base12.5"] = round(float((p2 <= 0.05).mean()), 3)

    # 참 차이 0 (귀무): 한 비교의 1종 오류율과 20개 비교 중 하나라도 유의할 확률(FWER, 보정 없음)
    null_p = np.empty((REPS, 20))
    for j in range(20):
        kb = rng.binomial(N, 0.30, REPS)
        ka = rng.binomial(N, 0.30, REPS)
        null_p[:, j] = p_greater(ka, kb)
    res["type1_single_comparison"] = round(float((null_p <= 0.05).mean()), 3)
    res["fwer_any_of_20_uncorrected"] = round(float((null_p.min(axis=1) <= 0.05).mean()), 3)
    res["fwer_any_of_20_holm_approx_bonferroni"] = round(float((null_p.min(axis=1) <= 0.05 / 20).mean()), 3)

    # 논문 표의 LN 대 π0.5 plug처럼 p가 0.05 근처인 사례: 같은 참 차이라도 유의 결과가 나올 확률
    k_b3 = rng.binomial(N, 0.30, REPS)
    k_a3 = rng.binomial(N, 0.30 + 0.20, REPS)
    p3 = p_greater(k_a3, k_b3)
    res["sign_flip_rate_true_20pp_p_between_0.05_0.10"] = round(float(((p3 > 0.05) & (p3 <= 0.10)).mean()), 3)

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("wrote", OUT)
    for k, v in res.items():
        print(k, v)


if __name__ == "__main__":
    main()
