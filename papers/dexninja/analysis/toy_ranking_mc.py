"""토이: 표 2 절제 순위와 표 1 식품별 방향이 시행 잡음만으로 얼마나 자주 나오는지 몬테카를로로 센다.

논문 주장 "궤적 정제를 빼면 성공률이 가장 크게 떨어진다"(본문 6.4절)와 "Sim+Real이 식품별로 일관되게 낫다"가
시행 수 100·25의 이항 잡음에 얼마나 견디는지 본다. 외부 데이터 없음, numpy만 쓴다(수 초).

가정: 시행은 독립 이항이고 짝 구조는 없다. 표의 관측 성공률을 참값으로 둔 경우(A)와 모든 변형이 같은 참값인 경우(B)를 본다.
실행: python3 analysis/toy_ranking_mc.py   → analysis/toy_ranking_mc_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_ranking_mc_results.json"
rng = np.random.default_rng(0)
R = 200_000
n = 100
names = ["Full", "w/o QC", "w/o PhysRand", "w/o TrajRef", "w/o Tactile", "w/o GenAssets"]
obs = np.array([50, 46, 44, 39, 43, 42]) / 100.0

# A. 관측 성공률이 참값일 때: 같은 실험을 다시 하면 순위가 유지되는가
x = rng.binomial(n, obs, size=(R, 6))
abl = x[:, 1:]
a = {
    "P(Full가 6개 중 단독 최고 또는 공동 최고)": float(np.mean(x[:, 0] >= x.max(axis=1))),
    "P(5개 절제가 모두 Full보다 낮다, 엄격)": float(np.mean((abl < x[:, [0]]).all(axis=1))),
    "P(w/o TrajRef가 절제 중 최저, 동률 포함)": float(np.mean(abl[:, 2] <= abl.min(axis=1))),
    "P(w/o TrajRef가 절제 중 단독 최저)": float(np.mean(abl[:, 2] < np.delete(abl, 2, axis=1).min(axis=1))),
}

# B. 모든 변형이 같은 참값(관측 평균 44%)일 때: 관측 같은 범위(최고 50 - 최저 39 = 11 %p)가 우연히 나오는가
p0 = obs.mean()
y = rng.binomial(n, p0, size=(R, 6))
b = {
    "공통 참값": float(p0),
    "P(범위 >= 11 %p)": float(np.mean(y.max(axis=1) - y.min(axis=1) >= 11)),
    "P(Full - 최저 절제 >= 11 %p, Full을 고정 행으로 볼 때)": float(np.mean(y[:, 0] - y[:, 1:].min(axis=1) >= 11)),
    "P(5개 절제 모두 Full 이하)": float(np.mean((y[:, 1:] <= y[:, [0]]).all(axis=1))),
}

# C. 표 1: 네 식품 모두에서 Sim+Real >= Real-only (식품당 25회) 가 같은 참값(0.40)에서 우연히 나올 확률
z1 = rng.binomial(25, 0.40, size=(R, 4))
z2 = rng.binomial(25, 0.40, size=(R, 4))
c = {"P(네 식품 모두 Sim+Real >= Real-only | 참값 같음 0.40)": float(np.mean((z1 >= z2).all(axis=1))),
     "P(네 식품 모두 엄격히 >)": float(np.mean((z1 > z2).all(axis=1)))}

res = {"R": R, "A_observed_as_truth": a, "B_all_equal_truth": b, "C_table1_direction": c}
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
