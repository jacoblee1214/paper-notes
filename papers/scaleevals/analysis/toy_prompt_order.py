"""토이 2: 프롬프트당 8회 롤아웃으로 세 정책의 순서를 맞힐 수 있는가 (논문 표 6, 부록 B.3).

질문: 표 6의 "10개 중 9개 정확한 순서, 쌍별 일치 96%"는, 태그별 평균 TP 격차(부록 표 11b)가 그대로 프롬프트에도 성립한다고 할 때
얼마나 자주 나오는가. 프롬프트마다 롤아웃 8회(표 6 요약)의 평균 TP 순서를 본다.

가정(논문에 없음):
  - 정책 i의 프롬프트 TP는 Beta(평균 m_i, 집중도 kappa)에서 독립 추출. kappa 0.7 / 1.5 / 3 (롤아웃 표준편차 약 0.35 / 0.27 / 0.20).
  - Place 계열의 평균은 표 11b의 transfer 행(π0.5 37, N1.7 23, N1.6 15), Rotate 계열은 reposing 행(N1.7 33, π0.5 29, N1.6 17).
  - 한 계열의 5개 프롬프트는 같은 평균을 가진 독립 시행으로 본다(프롬프트별 효과 없음).
  - g는 계열 평균을 중심으로 격차를 g배 키우는 배율(g = 1이면 태그 수준 격차 그대로).
출력: P(정확한 순서), 기대 쌍별 일치(동점 없음), 5개 프롬프트 중 4개 이상 맞을 확률.

실행: python3 analysis/toy_prompt_order.py   (numpy, 수 초)
"""
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_prompt_order_results.json"
FAMILIES = {
    "place(transfer 행)": {"order": ["pi0.5", "N1.7", "N1.6"], "mean": [0.37, 0.23, 0.15]},
    "rotate(reposing 행)": {"order": ["N1.7", "pi0.5", "N1.6"], "mean": [0.33, 0.29, 0.17]},
}
N_ROLL, TRIALS = 8, 40000


def simulate(means, kappa, rng):
    m = np.clip(np.array(means), 0.02, 0.98)
    x = np.stack([rng.beta(mi * kappa, (1 - mi) * kappa, (TRIALS, N_ROLL)) for mi in m], axis=1).mean(axis=2)  # (TRIALS, 3)
    exact = np.mean((x[:, 0] > x[:, 1]) & (x[:, 1] > x[:, 2]))
    pair = np.mean([np.mean(x[:, a] > x[:, b]) for a, b in combinations(range(3), 2)])
    return float(exact), float(pair)


def binom_ge(k, n, p):
    return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))


def main():
    rng = np.random.default_rng(0)
    res = {"assumptions": "Beta, 프롬프트당 8회, 계열 평균 = 표 11b 태그 행, 프롬프트 독립", "rows": []}
    for fam, d in FAMILIES.items():
        mean = np.array(d["mean"])
        for kappa in (0.7, 1.5, 3.0):
            for g in (1.0, 2.0, 3.0):
                m = mean.mean() + g * (mean - mean.mean())
                m = np.clip(m, 0.02, 0.98)
                p_exact, p_pair = simulate(m, kappa, rng)
                res["rows"].append({"family": fam, "kappa": kappa, "g": g, "means": np.round(m, 3).tolist(),
                                    "P_exact_order": round(p_exact, 3), "E_pairwise_concordance": round(p_pair, 3),
                                    "P(>=4 of 5 prompts exact)": round(binom_ge(4, 5, p_exact), 3),
                                    "P(5 of 5 exact)": round(p_exact ** 5, 3)})
    # 전체 10개 중 9개 이상, 두 계열 합산 (g, kappa 같음)
    comb = []
    for kappa in (0.7, 1.5, 3.0):
        for g in (1.0, 2.0, 3.0):
            pe = [r["P_exact_order"] for r in res["rows"] if r["kappa"] == kappa and r["g"] == g]
            # 동일 확률이 아니므로 Poisson-binomial을 직접 계산
            probs = [pe[0]] * 5 + [pe[1]] * 5
            dist = np.zeros(11)
            dist[0] = 1
            for p in probs:
                dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
                dist[0] *= (1 - p)
            comb.append({"kappa": kappa, "g": g, "P(>=9 of 10 exact)": round(float(dist[9:].sum()), 4),
                         "P(10 of 10)": round(float(dist[10]), 4)})
    res["combined_10_prompts"] = comb
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    for r in res["rows"]:
        print(r["family"], r["kappa"], r["g"], r["P_exact_order"], r["E_pairwise_concordance"], r["P(>=4 of 5 prompts exact)"])
    for c in comb:
        print(c)


if __name__ == "__main__":
    main()
