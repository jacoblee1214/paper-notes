"""토이 T1: 이진 성공 보상과 연속 반영기 보상에서 RLOO 이점(advantage)이 얼마나 퇴화하는가.

논문 §4.2 주장: 이진 보상은 한 묶음의 롤아웃이 모두 성공하거나 모두 실패하면 이점이 0이 되어 학습 신호가 사라진다.
연속 보상은 이 퇴화를 피한다. 이 토이는 메커니즘만 확인한다 (실제 세계모델과 반영기는 쓰지 않음).

설정: 초기 상태마다 성공 확률 p_i ~ Beta(a, b), 묶음당 롤아웃 N = 8 (논문 §4.4), RLOO 이점 식 (5).
연속 보상: 성공 여부 y에 잡음을 더한 sigmoid 점수 R = sigmoid(k*(y - 0.5) + eps), eps ~ N(0, s^2).
실행: python3 analysis/toy_advantage.py  →  analysis/toy_advantage_results.json  (수 초, numpy)
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_advantage_results.json"
N = 8          # 묶음당 롤아웃 수 (논문 §4.4)
G = 20000      # 시뮬 묶음 수
RNG = np.random.default_rng(0)


def rloo_adv(r):
    # r: (G, N). 식 (5): b_n = mean_{j != n} r_j, A_n = r_n - b_n
    s = r.sum(axis=1, keepdims=True)
    b = (s - r) / (N - 1)
    return r - b


def run(a, b, k=4.0, s=0.5):
    p = RNG.beta(a, b, size=G)                       # 초기 상태별 성공 확률
    y = (RNG.random((G, N)) < p[:, None]).astype(float)
    r_bin = y
    eps = RNG.normal(0.0, s, size=(G, N))
    r_cont = 1.0 / (1.0 + np.exp(-(k * (y - 0.5) + eps)))
    out = {}
    for name, r in [("binary", r_bin), ("continuous", r_cont)]:
        A = rloo_adv(r)
        zero_group = np.all(np.isclose(r, r[:, :1]), axis=1).mean()   # 모두 같은 보상
        out[name] = {
            "frac_groups_zero_advantage": round(float(zero_group), 4),
            "mean_abs_advantage": round(float(np.abs(A).mean()), 4),
        }
    out["mean_success_rate"] = round(float(p.mean()), 3)
    return out


RES = {"N_rollouts_per_group": N, "groups": G, "scenarios": {}}
# Beta 모수: 평균이 높은 과제(성공 ~0.85), 중간(~0.6), 낮음(~0.3). 두께는 a+b = 2 (양극화 정도)
for label, (a, b) in {"high_success_mean0.85": (1.7, 0.3),
                      "mid_success_mean0.6": (1.2, 0.8),
                      "low_success_mean0.3": (0.6, 1.4)}.items():
    RES["scenarios"][label] = run(a, b)

RES["note"] = ("메커니즘 확인용 토이. 실제 반영기 점수의 분포와 잡음 크기는 논문에 없어서 k, s는 가정이다. "
               "논문은 이 효과의 크기를 실험으로 보이지 않았고, 표 5의 절제가 간접 증거다.")
OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(RES, ensure_ascii=False, indent=2))
