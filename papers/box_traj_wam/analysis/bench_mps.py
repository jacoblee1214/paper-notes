"""표준 Box-WAM 단일 과제 학습·추론의 대략적 비용을 이 맥(MPS)에서 재 본다. 가중치는 랜덤 초기화(다운로드 없음, 새 설치 없음).

구조(공식 코드의 단일 과제 설정을 정적으로 읽어 맞춘 근사, 회전 위치 임베딩과 AdaLN 세부는 생략):
  인코더: ViT-L/16 (24블록, d=1024, 16헤드, MLP 4096), 입력 224x224 -> 패치 196개, 동결(순전파만)
  예측기: 12블록, d=768, 16헤드, MLP 3072, 토큰 = 관측 196 + 행동 15 + 상자 15 = 226
측정:
  (a) 인코더 순전파 이미지당 시간 -> 배치 384 x 6,000 스텝의 인코더 시간 추정
  (b) 예측기 순전파+역전파 샘플당 시간 -> 같은 스텝 수의 예측기 시간 추정
  (c) 추론 한 번: 이미지 1장 인코딩 + 예측기 전체 시퀀스 순전파 x 50 단계 (캐시 없는 상한)
실제 학습은 데이터 증강, 로딩, 컴파일, 고정밀 연산이 더해져 더 길다. 지연 값은 논문의 GPU 지연과 비교할 수 없다.

실행: python3 analysis/bench_mps.py   (torch 필요, MPS가 있으면 1분 안팎)  -> analysis/bench_mps_results.json
"""
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

OUT = Path(__file__).resolve().parent / "bench_mps_results.json"


class Block(nn.Module):
    def __init__(self, d, heads, mlp):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, mlp), nn.GELU(), nn.Linear(mlp, d))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class Stack(nn.Module):
    def __init__(self, d, depth, heads, mlp):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d, heads, mlp) for _ in range(depth)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()


def timeit(fn, dev, warm=1, iters=3):
    for _ in range(warm):
        fn()
    sync(dev)
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        sync(dev)
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2]


def main():
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)
    R = {"device": dev.type, "torch": torch.__version__}

    enc = Stack(1024, 24, 16, 4096).to(dev).eval()
    patch = nn.Conv2d(3, 1024, 16, 16).to(dev).eval()
    n_enc_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in patch.parameters())
    R["encoder_params_M"] = round(n_enc_params / 1e6, 1)
    pred = Stack(768, 12, 16, 3072).to(dev)
    R["predictor_blocks_params_M"] = round(sum(p.numel() for p in pred.parameters()) / 1e6, 1)

    B_ENC = 16
    img = torch.rand(B_ENC, 3, 224, 224, device=dev)

    def enc_fwd():
        with torch.no_grad():
            enc(patch(img).flatten(2).transpose(1, 2))

    print('encoder...', flush=True)
    t_enc = timeit(enc_fwd, dev)
    print('encoder done', t_enc, flush=True)
    R["encoder_fwd_s_per_batch16"] = round(t_enc, 3)
    R["encoder_ms_per_image"] = round(1000 * t_enc / B_ENC, 2)

    B_PRED, TOK = 32, 196 + 15 + 15
    x = torch.randn(B_PRED, TOK, 768, device=dev)
    opt = torch.optim.AdamW(pred.parameters(), lr=1e-4)

    def pred_step():
        loss = pred(x).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    t_pred = timeit(pred_step, dev)
    print('predictor done', t_pred, flush=True)
    R["predictor_fwd_bwd_s_per_batch32"] = round(t_pred, 3)
    R["predictor_ms_per_sample"] = round(1000 * t_pred / B_PRED, 2)

    steps, batch = 6000, 384
    enc_h = steps * batch * (t_enc / B_ENC) / 3600
    pred_h = steps * batch * (t_pred / B_PRED) / 3600
    R["estimate_6000_steps_batch384_hours"] = {"encoder": round(enc_h, 1), "predictor": round(pred_h, 1),
                                                "total_lower_bound": round(enc_h + pred_h, 1)}

    one = torch.rand(1, 3, 224, 224, device=dev)
    xi = torch.randn(1, TOK, 768, device=dev)
    pred.eval()

    def infer():
        with torch.no_grad():
            enc(patch(one).flatten(2).transpose(1, 2))
            for _ in range(50):
                pred(xi)

    R["inference_ms_50_steps_no_cache_upper_bound"] = round(1000 * timeit(infer, dev, warm=1, iters=3), 1)
    R["memory_note"] = "랜덤 가중치, 데이터 증강·로딩·정밀도 차이 제외. 논문의 RTX Pro 6000 지연과 비교할 수 없다."
    if dev.type == "mps":
        R["driver_allocated_GiB"] = round(torch.mps.driver_allocated_memory() / 2**30, 2)
    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(R, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
