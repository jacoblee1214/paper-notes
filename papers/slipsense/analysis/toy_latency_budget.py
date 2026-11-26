"""논문 §5.2 "모델 추론 평균 2.3 ms (RTX A4500), 프레임 예산 4.17 ms"가 이 맥(CPU, MPS)에서 어떻게 보일지 보는 토이 (torch, 1분 안팎).

논문은 구조의 일부만 적었다. 아래 크기는 가정이다.
  Piezo 인코더: 3x3 conv 3블록(채널 16, 32, 64 가정) + ReLU + 2x2 max pool, FC로 128 차원 (논문 §4.2: 블록 3개, d=128)
  XL 인코더   : 프레임당 16 bin 로그 멜 한 열을 토큰 하나로(16x1 패치), SSAST-Tiny 규모 트랜스포머(폭 192, 12층, 헤드 3: SSAST 원논문 Tiny 설정 가정)
  융합        : 센서 간 max pool, 양방향 교차 어텐션(폭 128, 헤드 4 가정), 연결(256), 인과 자기어텐션 1층(헤드 4 가정), 3-클래스 헤드
  창          : T = 60 프레임을 매 프레임 통째로 다시 계산(캐시 없음, 최악 경로). S = 2 (UMI), 3 (Tesollo)
가중치는 임의 초기화이며 학습하지 않는다. 정확도 주장과 무관하고 연산량·지연만 본다. 논문 구조와 차이가 있으면 결과도 달라진다.

실행: python3 analysis/toy_latency_budget.py  (torch 필요)  → analysis/toy_latency_budget_results.json
"""
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

OUT = Path(__file__).resolve().parent / "toy_latency_budget_results.json"
T, F, D = 60, 16, 128


class PiezoEnc(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [1, 16, 32, 64]
        layers = []
        for a, b in zip(ch[:-1], ch[1:]):
            layers += [nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(64 * 4 * 4, D)

    def forward(self, x):                      # (B, 1, 32, 32)
        return self.fc(self.conv(x).flatten(1))


class XlEnc(nn.Module):
    def __init__(self, width=192, depth=12, heads=3):
        super().__init__()
        self.inp = nn.Linear(F, width)
        self.pos = nn.Parameter(torch.zeros(1, T, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.out = nn.Linear(width, D)

    def forward(self, x):                      # (B, T, F)
        return self.out(self.enc(self.inp(x) + self.pos))


class SlipNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.p, self.x = PiezoEnc(), XlEnc()
        self.ca_p = nn.MultiheadAttention(D, 4, batch_first=True)
        self.ca_x = nn.MultiheadAttention(D, 4, batch_first=True)
        self.n_p, self.n_x = nn.LayerNorm(D), nn.LayerNorm(D)
        self.sa = nn.MultiheadAttention(2 * D, 4, batch_first=True)
        self.head = nn.Linear(2 * D, 3)
        self.register_buffer("mask", torch.triu(torch.ones(T, T, dtype=torch.bool), 1))

    def forward(self, piezo, xl):              # piezo (S, T, 32, 32), xl (S, T, F)
        S = piezo.shape[0]
        zp = self.p(piezo.reshape(S * T, 1, 32, 32)).reshape(S, T, D).max(0).values[None]
        zx = self.x(xl).max(0).values[None]
        zp2 = self.n_p(zp + self.ca_p(zp, zx, zx, attn_mask=self.mask)[0])
        zx2 = self.n_x(zx + self.ca_x(zx, zp, zp, attn_mask=self.mask)[0])
        z = torch.cat([zp2, zx2], -1)
        z = z + self.sa(z, z, z, attn_mask=self.mask)[0]
        return self.head(z[:, -1])


def bench(dev, S, threads, n=100):
    torch.set_num_threads(threads)
    m = SlipNet().to(dev).eval()
    p = torch.rand(S, T, 32, 32, device=dev)
    x = torch.randn(S, T, F, device=dev)
    sync = (lambda: torch.mps.synchronize()) if dev == "mps" else (lambda: None)
    with torch.inference_mode():
        for _ in range(10):
            m(p, x)
        sync()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            m(p, x)
            sync()
            ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {"median_ms": round(ts[len(ts) // 2], 2), "p90_ms": round(ts[int(0.9 * len(ts))], 2), "n": n}


def main():
    m = SlipNet()
    R = {"params_M": {"total": round(sum(p.numel() for p in m.parameters()) / 1e6, 2),
                      "piezo_enc": round(sum(p.numel() for p in m.p.parameters()) / 1e6, 3),
                      "xl_enc": round(sum(p.numel() for p in m.x.parameters()) / 1e6, 2)},
         "frame_budget_ms": round(1000 / 240, 2), "paper_reported_ms_RTX_A4500": 2.3,
         "assumption": "구조 크기는 가정 (docstring 참조), 임의 초기화, 창 60프레임 전체 재계산"}
    for dev in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
        for S in (2, 3):
            for th in ((1, 4) if dev == "cpu" else (1,)):
                R[f"{dev}_S{S}_threads{th}"] = bench(dev, S, th)
    OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(R, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
