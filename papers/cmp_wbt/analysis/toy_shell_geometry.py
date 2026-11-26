"""토이 T2: 식 5-6과 부록 B의 "z ~ N(0, R^2 I)는 반지름 sqrt(d) R 근처의 얇은 셸에 몰린다"를 차원 d별로 수치로 확인한다.

논문 표 6에서 인코더 출력은 6차원(d = 6)이다. 이 토이는 학습 없이 다음만 계산한다.
  1) 식 5: R_t = 1 / (W_target + eps)^3, eps = 1 - W_batch_mean (clip 범위 R_min, R_max는 논문에 없어 적용하지 않음)
  2) 서로 다른 안전 값 W_a > W_b 의 두 사전분포 N(0, R_a^2 I), N(0, R_b^2 I)에서 뽑은 z의 노름이
     반대로 나올 확률(노름 순서가 안전 순서와 어긋나는 비율)
  3) 안전한 상태(W 큼)의 z가 R_safe = 2.0 밖으로 나가 잘리는 비율
학습된 인코더의 출력은 사전분포의 표본이 아니라 정책 손실과 KL(계수 최대 1e-3, 표 7)의 균형으로 정해지므로,
이 토이는 "KL이 완벽히 맞았을 때의 상한적 선명도"만 보여 준다.

실행: python3 analysis/toy_shell_geometry.py   (numpy, 수 초)
"""
import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)
W_BAR = 0.9           # 가정: 배치 평균 안전 값 (논문에 없음)
EPS = 1 - W_BAR
R_SAFE = 2.0          # 표 2, 4의 선택 값
M = 200_000


def R_of(W):
    return 1.0 / (W + EPS) ** 3


Ws = [1.0, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2]
res = {"assumed_W_batch_mean": W_BAR, "eps": EPS, "R_of_W": {str(w): round(R_of(w), 3) for w in Ws}}

out = {}
for d in (2, 6, 32, 256):
    norms = {w: R_of(w) * np.linalg.norm(rng.standard_normal((M, d)), axis=1) for w in Ws}
    block = {"relative_sd_of_norm": round(float(norms[0.9].std() / norms[0.9].mean()), 3)}
    inv = {}
    for wa, wb in ((0.9, 0.8), (0.9, 0.6), (0.9, 0.4), (0.8, 0.6), (0.6, 0.4), (1.0, 0.2)):
        # 더 안전한 쪽(wa)의 노름이 덜 안전한 쪽(wb)의 노름보다 큰 확률 = 순서 뒤집힘
        inv[f"W {wa} vs {wb}"] = round(float((norms[wa] > norms[wb]).mean()), 3)
    block["P(order inverted)"] = inv
    block["P(norm > R_safe=2.0) by W"] = {str(w): round(float((norms[w] > R_SAFE).mean()), 3) for w in Ws}
    out[f"d={d}"] = block
res["by_dimension"] = out

# d=6 에서 R_safe 를 바꿨을 때 "W >= 0.8 이면 통과, W <= 0.6 이면 차단" 분류의 정확도 (두 부류 동수)
d = 6
safe_norm = np.concatenate([R_of(w) * np.linalg.norm(rng.standard_normal((M, d)), axis=1) for w in (1.0, 0.9, 0.8)])
unsafe_norm = np.concatenate([R_of(w) * np.linalg.norm(rng.standard_normal((M, d)), axis=1) for w in (0.6, 0.4, 0.2)])
best = {}
for thr in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0):
    best[str(thr)] = {"pass_rate_safe": round(float((safe_norm <= thr).mean()), 3),
                      "block_rate_unsafe": round(float((unsafe_norm > thr).mean()), 3)}
res["d6_threshold_sweep_W>=0.8_vs_W<=0.6"] = best
# 위 구분의 AUC (노름이 작을수록 안전)
idx = rng.choice(M * 3, 200_000, replace=False)
a = safe_norm[idx]
b = unsafe_norm[rng.choice(M * 3, 200_000, replace=False)]
res["d6_auc_norm_separates_safe_vs_unsafe"] = round(float((a < b).mean()), 3)

# R_safe = 2.0 을 안전 값으로 되돌리면 (식 5의 역함수 W = R^(-1/3) - eps; 노름 ~ sqrt(d) R 가정, 또는 R 자체로 읽는 경우)
implied = {}
for label, R in (("norm=sqrt(d)*R (d=6)", R_SAFE / np.sqrt(6)), ("norm read as R", R_SAFE)):
    implied[label] = {"R": round(float(R), 3), "implied_W": round(float(R ** (-1 / 3) - EPS), 3)}
res["implied_W_at_R_safe_2.0_for_W_bar_0.9"] = implied

Path(__file__).with_name("toy_shell_geometry_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
