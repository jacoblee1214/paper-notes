"""토이: best-of-N 계획에서 가치 집계 방식(평균 대 majority_mean)이 이상치 가치 예측에 얼마나 강한가.

논문 §4.3은 가치 예측이 bimodal이거나 분산이 크면 majority mean이 평균보다 강건하다고 주장한다.
이 토이는 그 주장을 가장 단순한 설정에서 확인한다. 실제 world model이나 로봇은 쓰지 않는다.

- 행동 제안 N = 8개, 각 제안은 실제 성공 확률 q_i를 가진다.
- 가치 예측 15개 (WM 3 x 가치 5, 논문 A.3.1). 각 예측은 확률 `out`으로 이상치(0 또는 1로 뒤집힘),
  아니면 q_i + 작은 잡음이다.
- 선택은 집계값이 가장 큰 제안, 성능은 선택된 제안의 실제 q.
- 집계: average, majority_mean (코드 cosmos_utils.py의 규칙을 옮김: 임계 0.05, 다수가 실패면 0).

실행: python3 analysis/toy_majority_mean.py  (numpy, 수 초)  → analysis/toy_majority_mean_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_majority_mean_results.json"
rng = np.random.default_rng(0)
N, M, THR = 8, 15, 0.05
TRIALS = 20000


def majority_mean(v):
    m = v > THR
    frac = m.mean()
    if frac <= 0.5:
        return 0.0
    return v[m].mean()


def run(out_rate, noise=0.05):
    gain_avg, gain_maj, gain_one = [], [], []
    for _ in range(TRIALS):
        q = rng.uniform(0.0, 1.0, size=N)  # 제안별 실제 성공 확률
        preds = q[:, None] + noise * rng.standard_normal((N, M))
        flip = rng.uniform(size=(N, M)) < out_rate
        preds[flip] = rng.integers(0, 2, size=flip.sum()).astype(float)  # 0 또는 1로 뒤집힘
        preds = np.clip(preds, 0, 1)
        avg_sel = np.argmax(preds.mean(1))
        maj_sel = np.argmax([majority_mean(p) for p in preds])
        one_sel = np.argmax(preds[:, 0])  # 단일 예측 (집계 없음)
        gain_avg.append(q[avg_sel])
        gain_maj.append(q[maj_sel])
        gain_one.append(q[one_sel])
    base = 0.5  # 제안 중 무작위 선택의 기대 q (0.5)
    return {
        "random_pick_expected_q": base,
        "avg_selected_q": round(float(np.mean(gain_avg)), 4),
        "majority_mean_selected_q": round(float(np.mean(gain_maj)), 4),
        "single_pred_selected_q": round(float(np.mean(gain_one)), 4),
    }


def oracle_mean():
    vals = []
    for _ in range(TRIALS):
        q = rng.uniform(0.0, 1.0, size=N)
        vals.append(q.max())
    return round(float(np.mean(vals)), 4)


results = {}
for out_rate in (0.0, 0.1, 0.3):
    r = run(out_rate)
    r["oracle_best_q"] = oracle_mean()
    results[f"outlier_rate_{out_rate}"] = r

OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(results, indent=2, ensure_ascii=False))
