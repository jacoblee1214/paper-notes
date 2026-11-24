"""MEM의 두 설계 주장을 작은 CPU 계산으로 확인하는 토이 두 개 (torch, numpy, 수 초).

T1 (영상 인코더, 3.3절·부록 C): 시간 위치 임베딩이 t=0에서 0이고 시간 어텐션이 인과 마스크일 때
   'K=1이면 인코더 출력이 원래 ViT와 정확히 같다'가 되는가. 층 하나를 작은 크기로 구현해 세 가지 읽기를 비교한다.
   (a) 덧셈형: 출력 = 공간 어텐션 + 시간 어텐션 (마스크가 대각 포함)
   (b) 식 3 합성형: 시간 어텐션 출력에 공간 어텐션을 적용 (대각 포함)
   (c) 덧셈형 + 엄격한 과거만 보는 마스크 (대각 제외)
   그리고 표준 사인-코사인 위치 임베딩이 t=0에서 0인지 본다.
T2 (언어 기억, 3.2절·IV-A): 학습 에피소드는 하위 과제를 한 번씩만 말하는데 추론에서는 재시도가 생긴다. 과제당 실패 확률 f일 때
   '압축하지 않은' 지시문 이력이 학습에 없던 모양(어떤 하위 과제의 반복)이 될 확률과 이력 길이를 계산한다.
   하위 과제 수 S는 부록 B의 청소 과제 평균 8을 쓴다. 가정한 f는 설계 값이며 논문 값이 아니다.
실행: python3 -I analysis/toy_memory.py  → analysis/toy_memory_results.json
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).with_name("toy_memory_results.json")
torch.manual_seed(0)
res = {}

# ---------- T1 ----------
d, H, n = 32, 4, 9  # 차원, 헤드, 이미지당 토큰


def attn(q, k, v, mask=None):
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    return torch.softmax(s, -1) @ v


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q, self.k, self.v, self.o = (torch.nn.Linear(d, d) for _ in range(4))

    def heads(self, x):  # (..., T, d) → (..., H, T, d/H)
        return x.reshape(*x.shape[:-1], H, d // H).transpose(-3, -2)

    def merge(self, x):
        return x.transpose(-3, -2).reshape(*x.shape[:-3], x.shape[-2], d)

    def spatial(self, z):  # z: (K, n, d), 각 시점 안에서 양방향
        q, k, v = (self.heads(f(z)) for f in (self.q, self.k, self.v))
        return self.o(self.merge(attn(q, k, v)))

    def temporal(self, z, strict):  # 같은 패치 위치끼리 시간 방향 인과
        zt = z.transpose(0, 1)  # (n, K, d)
        q, k, v = (self.heads(f(zt)) for f in (self.q, self.k, self.v))
        K = z.shape[0]
        mask = torch.ones(K, K, dtype=torch.bool).tril(-1 if strict else 0)
        if strict:  # 과거가 없는 행은 0 출력
            o = attn(q, k, v, mask.clone().fill_diagonal_(True))
            keep = mask.any(-1).float().view(1, 1, K, 1)
            o = self.o(self.merge(o * keep)) - (1 - keep.view(1, K, 1)) * self.o.bias  # 과거 없는 행은 바이어스까지 0
        else:
            o = self.o(self.merge(attn(q, k, v, mask)))
        return o.transpose(0, 1)


layer = Layer()
z1 = torch.randn(1, n, d)
std = layer.spatial(z1)  # 원래 ViT의 어텐션 출력 (K=1)
out = {}
with torch.no_grad():
    a = layer.spatial(z1) + layer.temporal(z1, strict=False)
    b = layer.spatial(layer.temporal(z1, strict=False))
    c = layer.spatial(z1) + layer.temporal(z1, strict=True)
    out["K1_max_abs_diff_vs_standard"] = {
        "a_additive_diagonal_included": float((a - std).abs().max()),
        "b_eq3_composition_diagonal_included": float((b - std).abs().max()),
        "c_additive_strictly_past": float((c - std).abs().max()),
        "standard_output_scale": float(std.abs().mean()),
    }
    # K=4에서 c가 현재 시점(마지막)만 쓰는 경우의 크기: 과거가 있을 때 시간 항이 0이 아닌가
    z4 = torch.randn(4, n, d)
    t4 = layer.temporal(z4, strict=True)
    out["K4_strict_temporal_term_mean_abs_by_timestep"] = [float(t4[i].abs().mean()) for i in range(4)]


def sincos(t, dim=8):
    i = np.arange(dim // 2)
    w = 1.0 / (10000 ** (2 * i / dim))
    return np.concatenate([np.sin(t * w), np.cos(t * w)])


out["sinusoid_at_t0_standard_sin_cos"] = [round(float(x), 3) for x in sincos(0)]
out["sinusoid_at_t0_sin_only"] = [round(float(x), 3) for x in np.sin(0 * (1.0 / (10000 ** (2 * np.arange(4) / 8))))]
out["note"] = "t=0에서 0이 되려면 사인 성분만 쓰거나 코사인에서 1을 빼는 변형이 필요. 이것만으로는 시간 어텐션 항이 K=1에서 사라지지 않음(마스크 규약이 따로 필요)."
res["T1_video_attention_K1"] = out

# ---------- T2 ----------
rng = np.random.default_rng(0)
S = 8
rows = []
for f in (0.05, 0.1, 0.2, 0.3, 0.4):
    # 하위 과제 s의 시도 횟수 ~ Geometric(1-f). 이력 길이 = 총 시도 수
    N = 200000
    att = rng.geometric(1 - f, size=(N, S))
    any_retry = (att > 1).any(1)
    res_row = {"f": f, "p_any_retry_closed_form": round(1 - (1 - f) ** S, 4), "p_any_retry_mc": round(float(any_retry.mean()), 4),
               "mean_history_len_naive": round(float(att.sum(1).mean()), 2), "history_len_training_style": S,
               "p_history_len_over_1.5S": round(float((att.sum(1) > 1.5 * S).mean()), 4),
               "p_same_instruction_3plus_times": round(float((att >= 3).any(1).mean()), 4)}
    rows.append(res_row)
res["T2_naive_language_history_shift"] = {"S": S, "rows": rows,
    "note": "f는 가정. 압축된 기억은 재시도가 일어나도 '완료한 하위 과제 집합'만 담으므로 같은 상태로 남는다는 설계를 가정한 계산이며, 학습된 모델이 실제로 그렇게 동작하는지는 다루지 않음."}
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1)[:2500])
