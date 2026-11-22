"""토이 T1: 공식 코드의 행동 전문가 교차 어텐션은 현재 특징 h_t와 예측 특징 h_T-hat에 순서·유형 표시가 없다.

코드 근거 (정적 읽기): flowmatching_expert.py 가 torch.cat((h_t, cond_future, cond_vlm)) 으로 키를 이어 붙이고,
AlternateVLDiT 의 image_mask 는 둘을 함께 켠다. 키에는 위치·유형 임베딩을 더하지 않는다.
이 스크립트는 그 구조를 numpy 로 흉내 내어 (1) 키 블록 순서를 바꿔도 출력이 같고 (2) 유형 임베딩을 더하면 달라짐을 보인다.
단일 헤드, 임의 가중치. 학습된 모델의 거동에 대한 주장이 아니라 구조적 성질의 확인이다. 실행 1초 미만.
"""
import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)
D, Dk, N, T = 32, 24, 16, 5  # 쿼리 차원, 키 차원, 토큰 수(블록당), 행동 토큰 수
Wq, Wk, Wv = (rng.normal(size=s) / 6 for s in [(D, Dk), (Dk, Dk), (Dk, Dk)])


def xattn(q, kv):
    Q, K, V = q @ Wq, kv @ Wk, kv @ Wv
    s = Q @ K.T / np.sqrt(Dk)
    a = np.exp(s - s.max(1, keepdims=True))
    a /= a.sum(1, keepdims=True)
    return a @ V


q = rng.normal(size=(T, D))
h_t = rng.normal(size=(N, Dk))
h_T = rng.normal(size=(N, Dk)) + 0.3  # 예측 특징: 통계가 조금 다르다고 가정
out_ab = xattn(q, np.vstack([h_t, h_T]))
out_ba = xattn(q, np.vstack([h_T, h_t]))
typ = rng.normal(size=(2, Dk))  # 유형 임베딩 (코드에는 없음)
out_ab_t = xattn(q, np.vstack([h_t + typ[0], h_T + typ[1]]))
out_ba_t = xattn(q, np.vstack([h_T + typ[1], h_t + typ[0]]))
out_swap_label = xattn(q, np.vstack([h_t + typ[1], h_T + typ[0]]))  # 라벨을 맞바꿈
res = {
    "max_abs_diff_block_order_swap_no_type_embedding": float(np.abs(out_ab - out_ba).max()),
    "max_abs_diff_block_order_swap_with_type_embedding": float(np.abs(out_ab_t - out_ba_t).max()),
    "max_abs_diff_label_swap_with_type_embedding": float(np.abs(out_ab_t - out_swap_label).max()),
    "note": "유형 임베딩 없이 두 블록의 순서를 바꾸면 출력은 같다(차이 ~1e-16). 순서를 바꾸는 것과 라벨을 바꾸는 것은 다르다: "
            "유형 임베딩이 있어도 블록 순서만 바꾸면 같다(어텐션은 집합 연산). 라벨을 바꾸면 달라진다.",
}
(Path(__file__).parent / "toy_key_order_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps(res, ensure_ascii=False, indent=1))
