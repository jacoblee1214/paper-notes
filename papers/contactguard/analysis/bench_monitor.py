"""논문 부록 A의 모니터 구조를 랜덤 가중치로 만들어 파라미터 수와 지연을 센다(가중치 다운로드 없음).

확인하는 주장:
  (a) 부록 A: 학습 부분(예측기 + 행동 임베더 + 헤드) 13.2 M, 고정 인코더 21.8 M, 전체 35.0 M
  (b) 표 5: RTX 5090에서 K = 30 전체 통과가 약 22 ms (33 ms 제어 주기 안)
구조는 논문 서술을 따른 우리 재구성이다(AdaLN-zero 4블록, 폭 384, 8헤드, FF 1536, 예측 MLP 은닉 1536, 맥락 C = 3, ViT-S/14 224 입력).
행동 임베더의 1x1 conv 폭, 위치 임베딩, LayerNorm 파라미터 등 논문에 없는 세부는 가정이다.
측정은 맥(MPS 또는 CPU)에서 한 프로세스로 한다. 논문의 GPU와 직접 비교할 수 없다.

실행: python3 analysis/bench_monitor.py   (torch 필요, 약 30초)  → analysis/bench_monitor_results.json
"""
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
DIM, HEADS, FF, DEPTH, C, K, ACT_DIM, VIEWS = 384, 8, 1536, 4, 3, 30, 14, 3


class ViTS14(nn.Module):
    """DINOv2-S(ViT-S/14) 구조: 224 입력, 패치 14, 12층, 폭 384, 6헤드. 랜덤 가중치."""

    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 384, 14, 14)
        self.cls = nn.Parameter(torch.zeros(1, 1, 384))
        self.pos = nn.Parameter(torch.zeros(1, 257, 384))
        layer = nn.TransformerEncoderLayer(384, 6, 1536, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, 12, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(384)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1) + self.pos
        return self.norm(self.blocks(x))[:, 0]


class AdaLNBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(DIM, elementwise_affine=False)
        self.n2 = nn.LayerNorm(DIM, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(DIM, HEADS, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(DIM, FF), nn.GELU(), nn.Linear(FF, DIM))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(DIM, 6 * DIM))

    def forward(self, x, c, mask):
        s1, b1, g1, s2, b2, g2 = self.mod(c).chunk(6, -1)
        h = self.n1(x) * (1 + s1) + b1
        x = x + g1 * self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        h = self.n2(x) * (1 + s2) + b2
        return x + g2 * self.mlp(h)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, C, DIM))
        self.blocks = nn.ModuleList([AdaLNBlock() for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(DIM)
        self.head = nn.Sequential(nn.Linear(DIM, FF), nn.GELU(), nn.Linear(FF, DIM))
        self.act = nn.Sequential(nn.Conv1d(ACT_DIM, DIM, 1), nn.Flatten(0, 0))  # 1x1 시간 conv(폭은 가정)
        self.act_mlp = nn.Sequential(nn.Linear(DIM, FF), nn.GELU(), nn.Linear(FF, DIM))

    def embed_action(self, a):  # a: (B, T, ACT_DIM)
        return self.act_mlp(self.act[0](a.transpose(1, 2)).transpose(1, 2))

    def forward(self, ctx, a_emb):  # ctx (B, C, DIM), a_emb (B, C, DIM)
        mask = torch.triu(torch.ones(C, C, dtype=torch.bool, device=ctx.device), 1)
        x = ctx + self.pos
        for blk in self.blocks:
            x = blk(x, a_emb, mask)
        return self.head(self.norm(x)[:, -1])


def count(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def full_pass(enc, proj, pred, probe, imgs, acts):
    """인코딩(3시점 x 3카메라) + K 스텝 롤아웃 + 프로브."""
    z = enc(imgs).view(C, VIEWS, DIM).mean(1)            # (C, DIM)
    z = proj(z).unsqueeze(0)                              # (1, C, DIM)
    a_emb = pred.embed_action(acts)                       # (1, K + C, DIM)
    ctx = z
    for k in range(K):
        nxt = pred(ctx, a_emb[:, k:k + C])
        ctx = torch.cat([ctx[:, 1:], nxt.unsqueeze(1)], 1)
    return probe(ctx[:, -1])


def bench(device, n=30):
    torch.manual_seed(0)
    enc, proj, pred = ViTS14().to(device).eval(), nn.Linear(DIM, DIM).to(device).eval(), Predictor().to(device).eval()
    probe = nn.Linear(DIM, 1).to(device).eval()
    imgs = torch.randn(C * VIEWS, 3, 224, 224, device=device)
    acts = torch.randn(1, K + C, ACT_DIM, device=device)

    def sync():
        if device == "mps":
            torch.mps.synchronize()

    for _ in range(3):
        full_pass(enc, proj, pred, probe, imgs, acts)
    sync()
    ts, enc_ts = [], []
    for _ in range(n):
        t = time.perf_counter(); full_pass(enc, proj, pred, probe, imgs, acts); sync(); ts.append(time.perf_counter() - t)
        t = time.perf_counter(); enc(imgs); sync(); enc_ts.append(time.perf_counter() - t)
    ts.sort(); enc_ts.sort()
    return dict(device=device, full_ms_median=round(1000 * ts[n // 2], 1), full_ms_p90=round(1000 * ts[int(0.9 * n)], 1),
                encoder_only_ms_median=round(1000 * enc_ts[n // 2], 1))


def main():
    enc, proj, pred = ViTS14(), nn.Linear(DIM, DIM), Predictor()
    n_pred = count(pred)
    out = dict(
        params_M=dict(encoder_vit_s14=round(count(enc) / 1e6, 2), fixed_projection=round(count(proj) / 1e6, 3),
                      trainable_total=round(n_pred / 1e6, 2),
                      blocks=round(sum(count(b) for b in pred.blocks) / 1e6, 2),
                      head=round(count(pred.head) / 1e6, 2),
                      action_embedder=round((count(pred.act) + count(pred.act_mlp)) / 1e6, 2),
                      monitor_total=round((count(enc) + count(proj) + n_pred) / 1e6, 2)),
        paper_params_M=dict(trainable=13.2, frozen_encoder=21.8, total=35.0),
        timing=[])
    torch.set_num_threads(4)
    out["timing"].append(bench("cpu", n=10))
    if torch.backends.mps.is_available():
        out["timing"].append(bench("mps"))
    out["paper_table5_rtx5090_ms"] = dict(rollout_K30=17.20, full_K30=21.48)
    out["note"] = "랜덤 가중치, FP32, 배치 1, 인코딩 9장(3시점 x 3카메라, 캐시 없음) + 30스텝 롤아웃(KV 캐시 없음, 맥락 3). 논문의 구현 세부와 다를 수 있음."
    (ROOT / "analysis" / "bench_monitor_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
