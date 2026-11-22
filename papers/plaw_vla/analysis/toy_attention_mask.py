"""PLaW-VLA의 블록 인과 어텐션(그림 3)이 공식 코드의 마스크 규칙에서 어떻게 나오는지 작은 수치 예로 확인한다.

공식 코드의 make_att_2d_masks(big_vision 방식: att_mask가 1인 토큰부터 새 블록, 같은 블록 안은 양방향,
뒤 블록은 앞 블록을 볼 수 있고 앞 블록은 뒤 블록을 못 본다)를 numpy로 다시 쓰고, 코드가 만드는 블록 구성
[VLM 접두부 P | 과거 H | 미래 F | 행동 A]에 적용한다. 코드를 실행하거나 가져오지 않는다.
출력: analysis/toy_attention_mask_results.json
실행: python3 analysis/toy_attention_mask.py   (numpy, 1초 미만)
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_attention_mask_results.json"


def make_att_2d_masks(pad, att, read=None):
    read = pad if read is None else read
    cs = np.cumsum(att, axis=1)
    m = cs[:, None, :] <= cs[:, :, None]
    return m & pad[:, :, None] & read[:, None, :]


sizes = {"P": 4, "H": 3, "F": 3, "A": 2}
order = ["P", "H", "F", "A"]
att = []
for name in order:
    block = [0] * sizes[name]
    if name != "P":
        block[0] = 1  # 월드 모델 과거/미래 블록과 행동 블록은 첫 토큰이 1 (코드: _build_block_att_mask, embed_suffix)
    att += block
att = np.array([att])
pad = np.ones_like(att, dtype=bool)
m = make_att_2d_masks(pad, att)[0]
idx = {}
start = 0
for name in order:
    idx[name] = slice(start, start + sizes[name])
    start += sizes[name]
table = {}
for q in order:
    table[q] = {k: bool(m[idx[q], idx[k]].all()) for k in order}
    # 부분 가시성 없음(블록 단위로 전부 보이거나 전부 안 보이는지)
    assert all(m[idx[q], idx[k]].all() or (~m[idx[q], idx[k]]).all() for k in order)
print("행=질의 블록, 열=키 블록 (True면 볼 수 있음)")
print("   " + "  ".join(f"{k:>5}" for k in order))
for q in order:
    print(f"{q}  " + "  ".join(f"{str(table[q][k]):>5}" for k in order))
res = {"attention_table": table,
       "reading": "P는 P만, H는 P·H, F는 P·H·F, A는 전부를 본다. VLM 접두부는 월드 모델 토큰을 못 본다(조건화 방향이 한쪽).",
       "note": "논문 그림 3의 'w/ attention' 패턴과 같은 블록 하삼각이다. 단, 학습 때 F 블록은 첫 번째 통과의 예측값을 되먹여 만든 토큰이다(정답 미래가 아님)."}
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
