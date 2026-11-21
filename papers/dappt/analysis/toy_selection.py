"""토이 1: "최선 도메인 파라미터 대 평균" 배율(1.5–3.8×)이 선택 편향만으로 얼마나 부풀 수 있는가.

설정: K개 도메인 파라미터 중 최선을 같은 시뮬레이터에서 고르고(논문 §III-D "random search"),
같은 시뮬레이터 값을 보고한다(표 IV). 진짜 성능이 모든 도메인에서 같다고 가정(진짜 차이 0)해도,
측정 잡음만으로 max/mean 배율이 1보다 커진다. 잡음 수준 c(에피소드 평균의 상대 SD)와 K를 바꾼다.
에피소드 수 미기재이므로 c를 격자로 둔다. 실행: python3 analysis/toy_selection.py  (1초 미만)
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("toy_selection_results.json")
rng = np.random.default_rng(0)
R = 20000  # 반복 수

rows = []
for K in (5, 20, 50):
    for c in (0.05, 0.10, 0.20):
        # 진짜 값은 모두 1. 측정값은 평균 1, 상대 SD c의 정규 잡음
        est = 1 + c * rng.standard_normal((R, K))
        ratio = est.max(axis=1) / est.mean(axis=1)
        rows.append({"K": K, "noise_cv": c, "mean_max_over_mean": round(float(ratio.mean()), 3),
                     "p95": round(float(np.quantile(ratio, 0.95)), 3)})

# 확인: 논문 배율 3.8은 잡음만으로 나오는가? 가장 큰 격자(K=50, c=0.2)에서도 약 1.6 수준
out = {"reps": R, "note": "진짜 차이 0 가정. 논문은 K와 c를 보고하지 않아 값은 격자입니다.",
       "rows": rows, "paper_factor_range": [1.5, 3.8]}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
for r in rows:
    print(r)
print("wrote", OUT)
