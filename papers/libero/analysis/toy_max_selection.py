"""LIBERO 코드의 체크포인트 선택 규칙을 몬테카를로로 본다 (CPU, 수 초).

근거: libero/lifelong/algos/base.py learn_one_task. 5 epoch마다 같은 초기 상태 20개로 평가하고,
가장 높은 성공률 체크포인트를 저장하고 보고한다. 논문 식 (3)의 c_{k,k} = max_e c_{k,k,e}가 같은 규칙이다.

질문: 정책 능력이 학습 내내 변하지 않아도(진짜 성공률 p 고정), 11번 평가 중 최댓값을 보고하면 얼마나 올라가나?
가정: 평가마다 롤아웃 20회의 독립 베르누이(성공률 p). 실제 롤아웃은 같은 초기 상태를 쓰므로 독립은 근사다.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "toy_max_selection_results.json"
rng = np.random.default_rng(0)
N_ROLL = 20
N_EVAL = 11  # e = 0, 5, ..., 50
DRAWS = 200_000

res = {}
for p in [0.3, 0.5, 0.7, 0.9]:
    k = rng.binomial(N_ROLL, p, size=(DRAWS, N_EVAL))
    mx = k.max(axis=1) / N_ROLL
    last = k[:, -1] / N_ROLL
    mean_all = k.mean(axis=1) / N_ROLL
    res[str(p)] = {
        "E[max 보고값] - p": round(float(mx.mean() - p), 4),
        "E[마지막 평가] - p": round(float(last.mean() - p), 4),
        "E[11회 평균] - p": round(float(mean_all.mean() - p), 4),
        "max 보고값 95% 범위 폭": round(float(np.quantile(mx, 0.975) - np.quantile(mx, 0.025)), 3),
    }
res["설명"] = ("p는 고정된 진짜 성공률. 'max'는 base.py가 저장·보고하는 값이다. "
               "'11회 평균'은 FWT_k의 평균 구조와 비슷하다. 결과는 편향 크기만 보여주며 논문 수치의 보정이 아니다.")
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
