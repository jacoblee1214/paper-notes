"""토이 T1: 공식 코드의 행동 이산화(prismatic/vla/action_tokenizer.py)를 numpy로 옮겨 왕복 검사한다.

확인하는 것 (D2)
- 256 토큰이 모두 쓰이는가, 그리고 디코딩 값이 몇 개인가.
- 정규화된 행동 범위 [-1, 1]의 양 끝(±1.0)이 어떤 값으로 되돌아가는가.
- 논문 §3.2의 "1st–99th 분위수 정규화 후 256 빈"은 공식 코드에서 데이터 정규화(BOUNDS_Q99)로 구현되고,
  토크나이저 자체는 고정 [-1, 1] 빈을 쓴다(D1). 이 토이는 토크나이저 부분만 다룬다.

입력: 없음 (균등 난수, 시드 0). 출력: analysis/toy_action_bins_results.json.
실행: python3 analysis/toy_action_bins.py  (numpy, 1초 미만)
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_action_bins_results.json"
N_BINS, MIN_A, MAX_A = 256, -1.0, 1.0
VOCAB = 32064  # Llama 2 어휘 크기 (OpenVLA 모델의 어휘 확장 전 크기로 가정)


def tokenize(a):
    a = np.clip(a, MIN_A, MAX_A)
    bins = np.linspace(MIN_A, MAX_A, N_BINS)
    d = np.digitize(a, bins)  # 1..256
    return VOCAB - d  # 마지막 256개 토큰 id


def decode(ids):
    bins = np.linspace(MIN_A, MAX_A, N_BINS)
    centers = (bins[:-1] + bins[1:]) / 2.0  # 255개
    d = VOCAB - ids
    idx = np.clip(d - 1, 0, centers.shape[0] - 1)
    return centers[idx]


rng = np.random.default_rng(0)
x = rng.uniform(-1, 1, size=200_000)
ids = tokenize(x)
used = np.unique(ids)
dec = decode(ids)
err = np.abs(dec - x)
res = {
    "token_ids_used": int(used.size),
    "token_id_range": [int(used.min()), int(used.max())],
    "decoded_distinct_values": int(np.unique(decode(used)).size),
    "bin_centers_available": N_BINS - 1,
    "max_abs_roundtrip_err": float(err.max()),
    "mean_abs_roundtrip_err": float(err.mean()),
    "half_bin_width": float(1.0 / (N_BINS - 1)),
    "decode_of_plus1": float(decode(tokenize(np.array([1.0])))[0]),
    "decode_of_minus1": float(decode(tokenize(np.array([-1.0])))[0]),
    "decode_of_0p995": float(decode(tokenize(np.array([0.995])))[0]),
}
OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
print(json.dumps(res, indent=2))
