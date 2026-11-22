"""RARM 비교기와 같은 크기의 무작위 가중치 모델로 이 기기의 처리 시간을 잰다 (torch, 약 1분).

가중치와 데이터는 받지 않는다. 구조는 공식 코드(정적 읽기)의 값을 따른다:
  - 프레임 인코더: ViT-B/16 (12층, d = 768, 12 헤드), 384x384 입력이면 프레임당 576 토큰, 클립 5프레임
  - 시간 처리기 이후: 프레임당 8x8 = 64 토큰 풀링 → 클립 320 토큰 (LoRA, 시간 처리기의 작은 연산은 생략)
  - 교차 어텐션(d = 768, q/k/v/out + 512로 사영) + 비교 헤드(512차원, 2층 TransformerEncoder + MLP)
  - 한 번의 보상 질의 = 롤아웃 클립 하나의 인코딩 + 참조 클립 M개와의 쌍 비교
상대 비교용 추정이다. 실제 가중치와 LoRA, 시간 처리기가 없어 절대값이 아니고, 다른 작업이 동시에 돌면 느려진다.

실행: python3 analysis/bench_mps.py   → analysis/bench_mps_results.json
"""
import json
import platform
import time
from pathlib import Path

import torch
import torch.nn as nn

OUT = Path(__file__).with_name("bench_mps_results.json")
torch.manual_seed(0)


class ViTB(nn.Module):
    def __init__(self, d=768, depth=12, heads=12, patch=16, size=384):
        super().__init__()
        self.patch = nn.Conv2d(3, d, patch, patch)
        self.pos = nn.Parameter(torch.zeros(1, (size // patch) ** 2, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2) + self.pos
        return self.blocks(x)


class CrossHead(nn.Module):
    def __init__(self, d=768, h=512, heads=8):
        super().__init__()
        self.ca = nn.MultiheadAttention(d, heads, batch_first=True)
        self.proj = nn.Linear(d, h)
        self.norm = nn.LayerNorm(h)
        layer = nn.TransformerEncoderLayer(h, heads, 512, dropout=0.0, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.mlp = nn.Sequential(nn.Linear(h, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, a, b):
        x, _ = self.ca(a, b, b)
        x = self.norm(self.proj(x))
        x = self.head(x)
        return self.mlp(x.mean(1)).squeeze(-1)


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


def timeit(fn, dev, reps=3):
    fn(); sync(dev)
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    sync(dev)
    return (time.perf_counter() - t) / reps


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if dev == "mps" else torch.float32
    vit = ViTB().to(dev, dtype).eval()
    cmp_ = CrossHead().to(dev, dtype).eval()
    res = {"device": dev, "dtype": str(dtype), "torch": torch.__version__, "machine": platform.machine(),
           "params_vit_M": round(sum(p.numel() for p in vit.parameters()) / 1e6, 1),
           "params_compare_M": round(sum(p.numel() for p in cmp_.parameters()) / 1e6, 1)}
    with torch.inference_mode():
        frames = torch.rand(5, 3, 384, 384, device=dev, dtype=dtype)
        res["encode_5_frames_s"] = round(timeit(lambda: vit(frames), dev), 4)
        res["encode_1_frame_s"] = round(timeit(lambda: vit(frames[:1]), dev), 4)
        for M in (20, 70, 140):
            a = torch.rand(M, 320, 768, device=dev, dtype=dtype)
            b = torch.rand(M, 320, 768, device=dev, dtype=dtype)
            res[f"compare_M{M}_s"] = round(timeit(lambda: cmp_(a, b), dev), 4)
    res["one_query_M70_s"] = round(res["encode_5_frames_s"] + res["compare_M70_s"], 4)
    res["one_query_M70_with_cached_frames_s"] = round(res["encode_1_frame_s"] + res["compare_M70_s"], 4)
    res["queries_per_episode_75_images"] = 75
    res["episode_reward_time_s_M70"] = round(75 * res["one_query_M70_s"], 1)
    res["rl_1M_steps_reward_time_h_M70"] = round(1_000_000 / 4 * res["one_query_M70_s"] / 3600, 1)  # 렌더 4스텝당 질의 한 번
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
