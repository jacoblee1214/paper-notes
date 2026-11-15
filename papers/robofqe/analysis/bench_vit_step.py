"""RoboFQE 한 갱신 스텝의 비용을 이 맥(MPS)에서 재 본다. 가중치는 랜덤 초기화(다운로드 없음, 새 설치 없음).
구조: ViT-S/16 크기(12블록, d=384, 헤드 6, MLP 1536), 입력 128x128 → 토큰 64개 + CLS.
공식 코드(train_fqe_dinov3.py 기본 설정)와 같은 부분 미세조정: 하위 8블록과 패치 임베딩은 동결, 상위 4블록과 최종 LN만 학습.
배치: 64 궤적 x 프레임 10 x 카메라 2 = 1280 이미지. 한 스텝 = 온라인 Q 순전파·역전파 + 목표 Q 순전파(next_obs).
정책(ResNet-T) 순전파와 시간 항 Q 헤드는 빼서 과소 추정이다.
실행: python3 analysis/bench_vit_step.py (torch 필요)   → analysis/bench_vit_step_results.json
"""
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

OUT = Path(__file__).resolve().parent / "bench_vit_step_results.json"
DIM, DEPTH, HEADS, MLP = 384, 12, 6, 1536
IMG, PATCH = 128, 16
TRAIN_BLOCKS = 4
BATCH, FRAMES, CAMS = 64, 10, 2


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(DIM)
        self.attn = nn.MultiheadAttention(DIM, HEADS, batch_first=True)
        self.n2 = nn.LayerNorm(DIM)
        self.mlp = nn.Sequential(nn.Linear(DIM, MLP), nn.GELU(), nn.Linear(MLP, DIM))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class ViTS(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, DIM, PATCH, PATCH)
        self.cls = nn.Parameter(torch.zeros(1, 1, DIM))
        n_tok = (IMG // PATCH) ** 2 + 1
        self.pos = nn.Parameter(torch.zeros(1, n_tok, DIM))
        self.blocks = nn.ModuleList([Block() for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(DIM)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1) + self.pos
        for b in self.blocks:
            x = b(x)
        return self.norm(x)[:, 0]


def main():
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)
    online, target = ViTS().to(dev), ViTS().to(dev)
    target.load_state_dict(online.state_dict())
    for p in online.parameters():
        p.requires_grad_(False)
    for b in online.blocks[-TRAIN_BLOCKS:]:
        for p in b.parameters():
            p.requires_grad_(True)
    for p in online.norm.parameters():
        p.requires_grad_(True)
    for p in target.parameters():
        p.requires_grad_(False)
    params = [p for p in online.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5)
    n_img = BATCH * FRAMES * CAMS
    x = torch.rand(n_img, 3, IMG, IMG, device=dev)
    xn = torch.rand(n_img, 3, IMG, IMG, device=dev)

    def step():
        q = online(x).pow(2).mean()          # 온라인 Q(대체 손실)
        with torch.no_grad():
            tq = target(xn).pow(2).mean()    # 목표 Q(next_obs)
        loss = (q - tq) ** 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss

    times = []
    for i in range(8):
        if dev.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        step()
        if dev.type == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
        if i >= 3:                            # 앞 3회는 준비 시간으로 버린다
            times.append(dt)
    mem = None
    if dev.type == "mps":
        try:
            mem = round(torch.mps.driver_allocated_memory() / 2**30, 2)
        except Exception:
            mem = None
    n_params = sum(p.numel() for p in online.parameters())
    res = {
        "device": dev.type, "torch": torch.__version__,
        "encoder": f"ViT-S/16 layout, {DEPTH} blocks, d={DIM}, random init, {n_params/1e6:.1f}M params",
        "images_per_step": n_img, "trainable_blocks": TRAIN_BLOCKS,
        "step_seconds_median": round(sorted(times)[len(times) // 2], 3),
        "step_seconds_all": [round(t, 3) for t in times],
        "driver_allocated_GiB_after": mem,
        "steps_50k_hours_estimate": round(50000 * sorted(times)[len(times) // 2] / 3600, 1),
        "note": "정책 순전파, 데이터 로딩, 평가(estimate)는 포함하지 않음. 랜덤 가중치라 정확도와 무관.",
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
