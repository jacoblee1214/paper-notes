"""CPU toy: 샘플링 후 예측 모델로 고르는 best-of-N 선택의 성공률 (표준 라이브러리만 사용).

가정 (논문에 없는 단순화):
- 한 스텝마다 정책이 후보 행동열 N개를 샘플링한다. 후보의 실제 결과 점수 g_i ~ N(0,1).
- 성공은 선택된 후보의 실제 점수가 임계값 c 이상일 때로 본다.
- 전 모델(forward model)은 점수를 g_i + eps, eps ~ N(0, sigma^2) 로 예측하고 argmax로 고른다.
- c는 '점수를 무작위로 고르면 성공률 38%'가 되도록 맞춘다 (논문 Table V의 DP-C 값).
목적: 전 모델 오차 sigma가 성공률을 얼마나 바꾸는지, 그리고 완벽한 모델(sigma=0)의 상한이 얼마인지 본다.
"""
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
N = 100          # 후보 수 (논문: 100 trajectories)
EPISODES = 20000
SEED = 0


def calibrate_threshold(target_rate):
    # 무작위 선택(후보 1개)의 성공률이 target_rate가 되는 임계값 c: P(g > c) = target
    lo, hi = -5.0, 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        p = 0.5 * math.erfc(mid / math.sqrt(2))  # P(g > mid)
        if p > target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def run(sigma, c, rng):
    succ = 0
    for _ in range(EPISODES):
        best_pred = -1e9
        best_true = None
        for _ in range(N):
            g = rng.gauss(0.0, 1.0)
            pred = g + (rng.gauss(0.0, sigma) if sigma > 0 else 0.0)
            if pred > best_pred:
                best_pred = pred
                best_true = g
        if best_true >= c:
            succ += 1
    return succ / EPISODES


c = calibrate_threshold(0.38)
rng = random.Random(SEED)
res = {"N_candidates": N, "episodes": EPISODES, "threshold_c": round(c, 4),
       "random_pick_success_target": 0.38, "by_sigma": {}}
rand_hits = sum(1 for _ in range(EPISODES) if rng.gauss(0, 1) >= c)
res["random_pick_success_check"] = round(rand_hits / EPISODES, 4)
for sigma in [0.0, 0.25, 0.5, 1.0, 2.0]:
    res["by_sigma"][str(sigma)] = round(run(sigma, c, random.Random(SEED + int(sigma * 100) + 1)), 4)

with open(os.path.join(HERE, "toy_best_of_n_results.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(json.dumps(res, ensure_ascii=False, indent=1))
