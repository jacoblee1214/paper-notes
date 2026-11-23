"""토이 T1: 규모 적응 앵커(식 2)가 손목 이동 분포를 얼마나 좁히는가.

논문 그림 6(a)는 세계 좌표 대 앵커 좌표의 이동 분포를 보여 준다(부록 C.1.2). 이 토이는 그 메커니즘만 확인한다.
- 물체: 윗면이 열린 반구가 아닌 타원체(반축 a >= b >= c), 전체 크기 30-80 cm에 해당하는 최장 반축 0.15-0.40 m,
        종횡비 b/a, c/a는 U(0.3, 1.0). 물체 중심은 원점, 책상 위 자세는 무시한다.
- 뷰: 윗반구의 K = 16개 방향(피보나치 격자, 논문 K = 16).
- "정답 파지": 뷰 v 방향 광선이 타원체 표면에 닿는 점 + 손바닥 간격 3 cm 바깥쪽 + N(0, 1 cm) 잡음. 실제 파지 분포가 아니라
  손바닥이 표면 바로 바깥에 놓인다는 가정이다.
- 앵커: 중심에서 v 방향으로 최소 외접구(반지름 = 최장 반축 a)와 만나는 점. 상대 이동 dt = t - 앵커.
비교: (A) 세계 좌표 t, (B) 뷰 조건부 세계 좌표 t - 뷰 평균(뷰가 주어졌을 때 남는 분산), (C) 앵커 상대 dt (뷰 평균을 빼기 전과 후).
수 초, numpy만 쓴다. 결과는 인공 분포에서의 메커니즘 확인이며 논문 수치와 비교할 수 없다.
실행: python3 analysis/toy_anchor.py
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_anchor_results.json"
rng = np.random.default_rng(0)
K, N_OBJ, N_GRASP = 16, 4000, 6


def upper_hemisphere_views(k):
    i = np.arange(k) + 0.5
    z = 1 - i / k  # (0, 1)
    phi = np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z * z)
    v = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    return v  # 단위 벡터, 중심을 향하는 방향의 반대(바깥) 벡터


V = upper_hemisphere_views(K)


def make(scale_lo, scale_hi, ar_lo=0.3, ar_hi=1.0):
    t_world, dt_anchor, view_id = [], [], []
    for _ in range(N_OBJ):
        a = rng.uniform(scale_lo, scale_hi)
        b = a * rng.uniform(ar_lo, ar_hi)
        c = a * rng.uniform(ar_lo, ar_hi)
        ax = np.sort(np.array([a, b, c]))[::-1]
        axes = rng.permutation(ax)  # 좌표축과 정렬(방향은 무작위 배정)
        for vi, v in enumerate(V):
            # 광선 중심 + s v 가 타원체 (x/ax)^2 합 = 1 에 닿는 s
            s = 1.0 / np.sqrt(np.sum((v / axes) ** 2))
            for _ in range(N_GRASP):
                t = (s + 0.03) * v + rng.normal(0, 0.01, 3)
                anchor = ax[0] * v
                t_world.append(t)
                dt_anchor.append(t - anchor)
                view_id.append(vi)
    return np.array(t_world), np.array(dt_anchor), np.array(view_id)


def summarize(t, dt, vid):
    out = {}
    out["world_std_xyz_cm"] = (100 * t.std(0)).round(2).tolist()
    out["anchor_std_xyz_cm"] = (100 * dt.std(0)).round(2).tolist()
    # 뷰 조건부: 같은 뷰 안의 분산
    def within(x):
        res = x.copy()
        for k in range(K):
            m = vid == k
            res[m] -= x[m].mean(0)
        return res.std(0)
    out["world_given_view_std_xyz_cm"] = (100 * within(t)).round(2).tolist()
    out["anchor_given_view_std_xyz_cm"] = (100 * within(dt)).round(2).tolist()
    out["norm_world"] = round(float(100 * np.sqrt((t.var(0)).sum())), 2)
    out["norm_world_given_view"] = round(float(100 * np.sqrt((within(t) ** 2).sum())), 2)
    out["norm_anchor"] = round(float(100 * np.sqrt((dt.var(0)).sum())), 2)
    out["norm_anchor_given_view"] = round(float(100 * np.sqrt((within(dt) ** 2).sum())), 2)
    return out


res = {"setting": "K=16, 4000 물체 x 16 뷰 x 6 파지, 최장 반축 0.15-0.40 m, 종횡비 U(0.3,1)"}
t, dt, vid = make(0.15, 0.40)
res["scale_0.15_0.40"] = summarize(t, dt, vid)
t, dt, vid = make(0.275, 0.275)  # 규모를 고정(대조): 규모 변동이 얼마나 기여하는가
res["scale_fixed_0.275"] = summarize(t, dt, vid)
t, dt, vid = make(0.15, 0.40, 0.9, 1.0)  # 거의 구형인 물체
res["scale_0.15_0.40_near_spheres"] = summarize(t, dt, vid)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
for k, v in res.items():
    print(k, v if isinstance(v, str) else {a: v[a] for a in ("norm_world", "norm_world_given_view", "norm_anchor", "norm_anchor_given_view")})
