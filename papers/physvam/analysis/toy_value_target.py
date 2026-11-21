"""PhysVAM 식 (2)–(4) 가치 목표를 그대로 구현해 성질을 검사한다 (학습 없음, CPU, 수 초).

검사하는 것
  V1. 식 (3)의 목표는 [-1, 1]에 있고, 마지막 시점 V_{Te-1} = c(e)를 만족한다(논문 §3.3 주장).
  V2. 식 (4)의 지평 h에서 Te를 넘는 시점은 종료 값을 유지한다.
  V3. 성공 에피소드(c = +1)에서 결과와 무관하게 시간만으로 정해지는 목표의 ±16 스텝 변화량.
      논문 부록 B.1은 "잡기 실패 근처에서도 목표는 모두 증가한다"고 쓴다. 목표가 시간만으로 정해진다면
      같은 시점 창에서 성공 잡기와 실패 잡기의 목표 변화 차이는 0이어야 한다. 논문 보고값은 +0.043이다.
      이 차이는 목표 식에 상호작용 정보가 없어서 생기는 것이 아니라 에피소드 길이와 시점 분포에서 온다.

가정: τ = 16, K = 4 (논문 부록 A.2). 에피소드 길이 Te는 논문에 없으므로 여러 값을 본다.
실행: python3 physvam/analysis/toy_value_target.py
출력: physvam/analysis/toy_value_target_results.json
"""
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_value_target_results.json"
K, TAU = 4, 16.0
HORIZONS = [4, 8, 16, 32]


def anchors(Te):
    # 식 (2): H_k = max(1, ceil(k Te / K))
    return [max(1, math.ceil(k * Te / K)) for k in range(1, K + 1)]


def value_target(Te, c, t):
    # 식 (3): V_t = c/K * sum_k exp(-[H_k - 1 - t]_+ / tau)
    H = anchors(Te)
    return c / K * sum(math.exp(-max(Hk - 1 - t, 0) / TAU) for Hk in H)


def horizon_target(Te, c, t, h):
    # 식 (4): V_t^h = V_{min(t+h, Te-1)}
    return value_target(Te, c, min(t + h, Te - 1))


res = {"V1_range_and_terminal": [], "V2_horizon_clip": None, "V3_window_change": []}

# V1: 범위와 종료 값
for Te in [60, 100, 150, 200, 300, 400]:
    vals = [value_target(Te, c, t) for c in (+1, -1) for t in range(Te)]
    inside = bool(np.all((np.array(vals) >= -1 - 1e-12) & (np.array(vals) <= 1 + 1e-12)))
    term = {c: abs(value_target(Te, c, Te - 1) - c) < 1e-12 for c in (+1, -1)}
    res["V1_range_and_terminal"].append({"Te": Te, "in_minus1_1": inside, "terminal_equals_c": term})
print("[V1] 범위와 종료 값:", all(r["in_minus1_1"] and all(r["terminal_equals_c"].values()) for r in res["V1_range_and_terminal"]))

# V2: 지평이 Te를 넘으면 종료 값
Te = 120
clip_ok = all(abs(horizon_target(Te, 1, t, 32) - value_target(Te, 1, Te - 1)) < 1e-12 for t in range(Te - 32, Te))
res["V2_horizon_clip"] = bool(clip_ok)
print("[V2] 지평 클리핑:", clip_ok)

# V3: 성공 에피소드에서 시간만으로 정해지는 목표의 +-16 스텝 변화 (h = 4)
rows = []
for Te in [100, 150, 200, 250, 300, 400]:
    for frac in (0.3, 0.5, 0.7):
        tc = int(frac * Te)
        if tc - 16 < 0 or tc + 16 >= Te:
            continue
        d = horizon_target(Te, +1, tc + 16, 4) - horizon_target(Te, +1, tc - 16, 4)
        rows.append({"Te": Te, "tc_frac": frac, "delta_target_pm16_h4": round(d, 4)})
res["V3_window_change"] = rows
ds = [r["delta_target_pm16_h4"] for r in rows]
print("[V3] 성공 에피소드 목표의 +-16 변화 범위:", min(ds), "~", max(ds), "(논문 보고 목표 차이 +0.043)")
print("     모든 행에서 증가:", all(d > 0 for d in ds))

# 성공 에피소드의 목표는 시점에 대해 단조 증가다(식 (3)에서 지수항이 모두 증가). 따라서 "목표가 증가한다"는 사실은
# 상호작용 정보 없이 식에서 나온다. 모든 Te에서 확인한다.
mono = all(np.all(np.diff([value_target(Te, +1, t) for t in range(Te)]) >= -1e-12) for Te in [60, 100, 200, 300, 400])
mono_neg = all(np.all(np.diff([value_target(Te, -1, t) for t in range(Te)]) <= 1e-12) for Te in [60, 100, 200, 300, 400])
res["V3_monotone_success_nondecreasing"] = bool(mono)
res["V3_monotone_failure_nonincreasing"] = bool(mono_neg)
print("[V3] 성공 에피소드 목표는 비감소:", mono, "/ 실패 에피소드 목표는 비증가:", mono_neg)

# 실패 에피소드(c = -1)의 같은 창 변화
neg = []
for Te in [100, 200, 300]:
    tc = Te // 2
    neg.append(round(horizon_target(Te, -1, tc + 16, 4) - horizon_target(Te, -1, tc - 16, 4), 4))
res["V3_failed_episode_change_examples"] = neg
print("[V3] 실패 에피소드 같은 창 변화 (Te=100,200,300 중간):", neg)

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT)
