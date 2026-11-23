"""토이: 부록 A.2의 부모 선택 가중치가 섬 안의 유효 노드 비율에 따라 어떻게 달라지는가. numpy, 1초 미만.

가중치 (부록 A.2):  alpha = median(F),  m = max(median|F - alpha|, eps),
                    w_i = sigmoid(lambda (F_i - alpha) / m) / (1 + o_i)
논문은 점수 F 와 lambda 를 정의하지 않는다. 여기서는 공식 저장소 값을 가정한다:
  F = 1 / (1 + 비용) (유효 노드), F = 0 (kinematic 실패 노드, 3.2.4절의 "점수 0"), lambda = 10, eps = 1e-6.
비용은 표 1의 MD 비용 범위(10-20)에서 균등 추출, 자식 수 o_i = 0 으로 고정해 점수 효과만 본다.
관찰 지표: (a) 선택 확률 중 유효 노드가 차지하는 몫, (b) 유효 노드 안에서 가장 싼 노드와 가장 비싼 노드의 가중치 비(중앙값).

실행: python3 analysis/toy_parent_weights.py → analysis/toy_parent_weights_results.json
이것은 부록 A.2 식을 그대로 옮긴 것이며, 저장소 코드를 실행한 것이 아니다.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_parent_weights_results.json"
LAM, EPS = 10.0, 1e-6


def sigmoid(x):
    return 0.5 * (1 + np.tanh(0.5 * x))


def weights(F, o):
    alpha = np.median(F)
    m = max(np.median(np.abs(F - alpha)), EPS)
    return sigmoid(LAM * (F - alpha) / m) / (1 + o), alpha, m


def run(valid_frac, n=40, trials=2000, seed=0):
    rng = np.random.default_rng(seed)
    mass_valid, ratio, alphas, ms = [], [], [], []
    for _ in range(trials):
        k = int(round(valid_frac * n))
        cost = rng.uniform(10, 20, size=k)
        F = np.concatenate([1 / (1 + cost), np.zeros(n - k)])
        w, alpha, m = weights(F, np.zeros(n))
        mass_valid.append(w[:k].sum() / w.sum() if k else 0.0)
        ratio.append(w[:k].max() / w[:k].min() if k > 1 else np.nan)
        alphas.append(alpha)
        ms.append(m)
    return {"valid_frac": valid_frac, "n_nodes": n,
            "alpha_median": float(np.mean(alphas)), "m_scale_median": float(np.median(ms)),
            "valid_mass_mean": float(np.nanmean(mass_valid)),
            "valid_mass_if_uniform": valid_frac,
            "weight_ratio_cheapest_vs_costliest_valid_median": float(np.nanmedian(ratio))}


def main():
    res = [run(f) for f in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("valid_frac  alpha     m        valid_mass  weight_ratio(cheap/costly valid)")
    for r in res:
        print(f"{r['valid_frac']:.1f}  {r['alpha_median']:.4f}  {r['m_scale_median']:.2e}  "
              f"{r['valid_mass_mean']:.3f}  {r['weight_ratio_cheapest_vs_costliest_valid_median']:.3f}")


if __name__ == "__main__":
    main()
