"""toy: 공식 코드(diffsynth/pipelines/wan_video_act_embed.py L359–364)의 행동 조건 경로가
시간축을 평균내는 것(action.float().mean(dim=1))이 가져오는 결과를 numpy로 확인한다.
서로 다른 두 행동 청크(빠른→느린, 느린→빠른)의 평균이 같으면 조건 벡터가 정확히 같아진다.
행동 차원 14, 청크 길이 50. 프로젝션은 무작위 선형 층으로 대체(동일 연산 구조). 실행: python3 analysis/toy_temporal_pool.py
"""
import json
from pathlib import Path
import numpy as np

rng = np.random.default_rng(1)
OUT = Path(__file__).resolve().parent / "toy_temporal_pool_results.json"
T, D, H = 50, 14, 1280  # 청크 길이, 관절 차원, action_dim(README 기본 1280)
t = np.linspace(0, 1, T)[:, None]
base = rng.uniform(-0.5, 0.5, (1, D))
a_fwd = base + 0.5 * t * np.ones((1, D))          # 시간에 따라 증가
a_rev = base + 0.5 * (1 - t) * np.ones((1, D))    # 시간에 따라 감소
a_fwd -= a_fwd.mean(0) - base[0]                  # 평균을 맞춘다
a_rev -= a_rev.mean(0) - base[0]
W = rng.normal(0, 1 / np.sqrt(D), (D, H))         # 무작위 프로젝션 (action_proj 자리)
alpha = 0.3                                       # README 예시 action_alpha

def cond(a):
    pooled = a.mean(axis=0)                        # dim=1 평균과 같은 역할
    return alpha * (pooled @ W)

c1, c2 = cond(a_fwd), cond(a_rev)
out = {
    "chunk_shape": [T, D],
    "traj_l2_distance": float(np.linalg.norm(a_fwd - a_rev)),
    "pooled_l2_distance": float(np.linalg.norm(a_fwd.mean(0) - a_rev.mean(0))),
    "cond_l2_distance": float(np.linalg.norm(c1 - c2)),
    "cond_max_abs_diff": float(np.max(np.abs(c1 - c2))),
    "interpretation": "두 궤적은 L2 거리가 크지만 조건 벡터 차이는 기계 오차 수준이다. 공식 코드의 이 경로는 시간 순서를 버린다. 논문 §3.2는 이 점을 언급하지 않는다.",
}
OUT.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
