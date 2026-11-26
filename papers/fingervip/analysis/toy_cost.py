"""FingerViP 정책의 추론 비용 토이 (표 2 점검): 분석식과 임의 가중치 시간 측정.

공식 코드를 쓰지 않고, 논문 본문과 표 8의 구조(ViT-B/16 이미지 인코더, 7층 d=768 트랜스포머 디노이저, 16 디노이징 스텝)를
torch 기본 모듈로 직접 다시 쓴 것이다. 가중치는 임의 초기값이라 정확도와는 무관하고, 연산량과 시간만 본다.
층 수 7과 헤드 8은 공개 설정 파일의 값이다(논문 표 8은 헤드 8만 적음).

1. 분석식: ViT-B/16(224x224)의 곱셈-덧셈(MAC) 수, 디노이저 MAC 수 → 논문 표 2의 215.01, 79.00 GFLOPs와 비교
2. 시간 측정: 이미지 12장(손끝 5 + 외부 1, 관측 2스텝)과 4장(외부 1 + 손목 1, 2스텝) 인코딩, 디노이저 16스텝
   (CPU와 가능하면 MPS, 배치 1, fp32, 단일 프로세스, 약 1분)

실행: python3 analysis/toy_cost.py  (torch 필요)  → analysis/toy_cost_results.json
"""
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

OUT = Path(__file__).resolve().parent / "toy_cost_results.json"
torch.manual_seed(0)
torch.set_num_threads(4)


# ---------------------------------------------------------------- 1. 분석식
def vit_b16_macs(with_attn_scores=True):
    d, ff, tokens, layers = 768, 3072, 197, 12
    per_layer = 3 * tokens * d * d + tokens * d * d + 2 * tokens * d * ff  # qkv, 출력 투영, MLP
    if with_attn_scores:
        per_layer += 2 * tokens * tokens * d  # QK^T, AV
    patch = 196 * (16 * 16 * 3) * d
    return layers * per_layer + patch


def denoiser_macs(n_cond, steps=16, layers=7, d=768, ff=3072, horizon=16, act_dim=26):
    # 한 스텝: 자기어텐션(qkvo) + 교차어텐션(q, o는 행동 토큰, k, v는 조건 토큰) + MLP + 입출력 투영
    per_layer = (4 * horizon * d * d + 2 * horizon * horizon * d
                 + 2 * horizon * d * d + 2 * n_cond * d * d + 2 * horizon * n_cond * d
                 + 2 * horizon * d * ff)
    one = layers * per_layer + 2 * horizon * act_dim * d
    return one * steps, one


res = {}
for with_scores in (True, False):
    vit = vit_b16_macs(with_scores)
    key = "with_attn_scores" if with_scores else "without_attn_scores"
    ours_cond = 6 * 2 + 2 + 1       # 시각 토큰 6뷰 x 2스텝, 관절 위치 토큰 2, 시간 토큰 1
    base_cond = 2 * 2 + 2 + 1       # 시각 토큰 2뷰 x 2스텝
    den_o, den_o1 = denoiser_macs(ours_cond)
    den_b, den_b1 = denoiser_macs(base_cond)
    res[key] = {
        "vit_b16_gmacs_per_image": round(vit / 1e9, 3),
        "ours_encoder_gmacs": round(12 * vit / 1e9, 2), "base_encoder_gmacs": round(4 * vit / 1e9, 2),
        "ours_denoiser_16steps_gmacs": round(den_o / 1e9, 2), "base_denoiser_16steps_gmacs": round(den_b / 1e9, 2),
        "ours_denoiser_1step_gmacs": round(den_o1 / 1e9, 3), "base_denoiser_1step_gmacs": round(den_b1 / 1e9, 3),
        "ours_total_16steps": round((12 * vit + den_o) / 1e9, 2), "base_total_16steps": round((4 * vit + den_b) / 1e9, 2),
        "ours_total_1step": round((12 * vit + den_o1) / 1e9, 2), "base_total_1step": round((4 * vit + den_b1) / 1e9, 2),
        "difference_total_16steps": round(((12 * vit + den_o) - (4 * vit + den_b)) / 1e9, 2),
    }
res["paper_table2_gflops"] = {"ours": 215.01, "baseline": 79.00, "difference": round(215.01 - 79.00, 2)}
res["note"] = "곱셈-덧셈 1회를 1로 센 값(MAC). 논문 표 2의 단위(MAC인지 FLOP인지)는 정의되어 있지 않다."

# ---------------------------------------------------------------- 2. 시간 측정 (임의 가중치)
class ViTB16(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 768, 16, 16)
        self.cls = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos = nn.Parameter(torch.zeros(1, 197, 768))
        layer = nn.TransformerEncoderLayer(768, 12, 3072, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 12)
        self.ln = nn.LayerNorm(768)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1) + self.pos
        return self.ln(self.enc(x))[:, 0]


class Denoiser(nn.Module):
    def __init__(self, layers=7, d=768, heads=8, horizon=16, act_dim=26):
        super().__init__()
        self.inp = nn.Linear(act_dim, d)
        layer = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, layers)
        self.head = nn.Linear(d, act_dim)

    def forward(self, a, cond):
        return self.head(self.dec(self.inp(a), cond))


def bench(device):
    dev = torch.device(device)
    vit, den = ViTB16().eval().to(dev), Denoiser().eval().to(dev)

    def sync():
        if device == "mps":
            torch.mps.synchronize()

    out = {}
    with torch.no_grad():
        for name, n_img, n_cond in (("ours_12_images", 12, 15), ("baseline_4_images", 4, 7)):
            x = torch.randn(n_img, 3, 224, 224, device=dev)
            cond = torch.randn(1, n_cond, 768, device=dev)
            a = torch.randn(1, 16, 26, device=dev)
            for _ in range(2):
                vit(x)
                den(a, cond)
            sync()
            t0 = time.perf_counter()
            reps = 3
            for _ in range(reps):
                vit(x)
            sync()
            t_enc = (time.perf_counter() - t0) / reps * 1000
            t0 = time.perf_counter()
            for _ in range(reps):
                z = a
                for _ in range(16):
                    z = den(z, cond)
            sync()
            t_den = (time.perf_counter() - t0) / reps * 1000
            out[name] = {"encoder_ms": round(t_enc, 1), "denoiser_16steps_ms": round(t_den, 1), "total_ms": round(t_enc + t_den, 1)}
    return out


res["timing_random_weights_fp32"] = {"cpu_4threads": bench("cpu")}
if torch.backends.mps.is_available():
    res["timing_random_weights_fp32"]["mps"] = bench("mps")
res["timing_note"] = "임의 가중치, 배치 1, 전처리와 카메라 수집 시간 제외. 논문의 RTX 4080 측정(표 2)과 직접 비교하지 않는다."

print(json.dumps(res, ensure_ascii=False, indent=1))
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
