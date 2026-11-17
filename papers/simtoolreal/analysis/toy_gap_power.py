"""토이: Fig. 5(Variation #1)의 98.0 대 61.0 (차이 37 pp)이 n=5 rollout에서 얼마나 안정적인지 본다.
가정: 각 조건의 rollout 값은 평균 m, 표준편차 s인 정규분포를 0–100으로 자른 값(단순 가정). 논문은 rollout 값과 SD를 주지 않는다.
실행: python3 analysis/toy_gap_power.py → analysis/toy_gap_power_results.json (numpy, scipy, 몇 초)
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats


def sim(s, reps=20000, seed=1):
    rng = np.random.default_rng(seed)
    a = np.clip(rng.normal(98.0, s, (reps, 5)), 0, 100)
    b = np.clip(rng.normal(61.0, s, (reps, 5)), 0, 100)
    p = stats.ttest_ind(a, b, axis=1, equal_var=False).pvalue
    return float((p < 0.05).mean()), float(np.mean(a.mean(1) - b.mean(1)))


def main():
    res = {"assumption": "rollout ~ clip(N(mean, s), 0, 100), 5 rollouts per condition, Welch t, 20000 reps",
           "by_sd": {}}
    for s in (5, 10, 15, 20, 25, 30):
        power, mean_gap = sim(s)
        res["by_sd"][str(s)] = {"welch_p_lt_0.05_rate": round(power, 3), "mean_gap_pp": round(mean_gap, 1)}
    # 최소로 필요한 SD: 검정력 0.8을 얻는 s (대략 탐색)
    res["note"] = ("논문은 Variation #1의 rollout별 값과 SD를 보고하지 않는다. 이 토이는 SD 가정에 따라 37 pp 차이가 "
                   "5회 반복으로 유의하게 보일 가능성을 보여 줄 뿐이며, 실제 p값이 아니다.")
    Path(__file__).with_name("toy_gap_power_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
