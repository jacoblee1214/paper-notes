"""sysid_toy.py 모델에서, 보정 오차 하나가 착탄 거리에 얼마나 전달되는지 본다(같은 당김 거리, 재계산 없음 / 재계산).
실행: python3 -I analysis/model_error_check.py -> analysis/model_error_results.json (몇 초)
재계산(pull을 θ̂로 다시 풀기)을 해도 오차가 사라지지 않는다: 정책은 같은 모델로 당김을 고르므로 모델 오차가 그대로 넘어온다.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sysid_toy as S  # noqa: E402

OUT = Path(__file__).resolve().parent / "model_error_results.json"
G = 1.925
T = S.TRUE[None]
d_star = S.solve_pull(T, np.array([G]))
cases = {"k x1.1": [1.1, 1, 1, 1], "beta x1.3": [1, 1.3, 1, 1], "c x2": [1, 1, 2, 1], "mu x2": [1, 1, 1, 2],
         "k x1.1, beta /1.1": [1.1, 1 / 1.1, 1, 1]}
rows = {}
for name, mult in cases.items():
    th = T * np.array(mult)[None]
    fixed = 100 * (S.landing(th, d_star)[0] - G)
    d_hat = S.solve_pull(th, np.array([G]))
    recal = 100 * (S.landing(T, d_hat)[0] - G)
    rows[name] = {"model_R_error_at_same_pull_cm": round(float(fixed), 1),
                  "true_error_after_recalibrated_pull_cm": round(float(recal), 1)}
out = {"goal_m": G, "d_star_m": float(d_star[0]), "rows": rows,
       "note": "fixed: 같은 당김 거리에서 보정 모델이 틀리는 양. recal: 보정 모델로 당김을 다시 골랐을 때 실제 착탄 오차."}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
