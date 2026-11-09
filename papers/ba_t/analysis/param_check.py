"""BA-T 파라미터 수를 공식 코드의 모듈 정의(repo @ 406d791)에서 손으로 세어 논문 수치와 대조한다.

실행: python3 analysis/param_check.py  → analysis/param_check_results.json
torch 없이 Linear·LayerNorm 크기만 더한다(가중치 다운로드 없음).
- P1: BundleAdjustmentLayer(디코더 본체)와 디코더 주변(appearance_embed, 헤드, 초기 카메라 토큰)
- P2: ViT-L 인코더(24층, 1024차원) — 논문 비교의 "decoder size"에 들어가지 않는 몫
- P3: ViSTA† "12층 113M → 3층 38M" 산술(Appendix B)과 DecoderBlock 크기
- P4: 논문 Table 1·5 크기 비율(16%, 6×, 5–6%)
- P5: 224×224 한 쌍(시점당 토큰 196개)의 연산량 추정(곱-덧셈 2 FLOP 기준, softmax·LN 제외)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def linear(i, o, bias=True):
    return i * o + (o if bias else 0)


def layernorm(d):
    return 2 * d


def attention(d):              # vit_blocks.Attention: qkv(d→3d, bias) + proj(d→d, bias)
    return linear(d, 3 * d) + linear(d, d)


def cross_attention(d):        # vit_blocks.CrossAttention: q,k,v(d→d, bias) + proj
    return 3 * linear(d, d) + linear(d, d)


def mlp(d, ratio=4):           # vit_blocks.Mlp: fc1, fc2
    return linear(d, ratio * d) + linear(ratio * d, d)


def crossattn_layer(d):        # ba_blocks.CrossAttnLayer(use_mlp=True)
    return layernorm(d) + layernorm(d) + cross_attention(d) + layernorm(d) + mlp(d)


def modln(d):                  # ba_blocks.ModLN: LayerNorm + SiLU + Linear(d→2d)
    return layernorm(d) + linear(d, 2 * d)


def cond_block(d):             # ba_blocks.ConditionModulationBlock
    return modln(d) + attention(d) + modln(d) + mlp(d)


def token_corr_res(d):         # ba_blocks.TokenCorrRes: q,k,v + proj(Linear-GELU-Linear) + 2 LN
    return 3 * linear(d, d) + linear(d, d) + linear(d, d) + 2 * layernorm(d)


def ba_layer(d):
    parts = {
        "self_attn_x3 (norm + attn, MLP 없음)": 3 * (layernorm(d) + attention(d)),
        "cross_attn_layer (geometry update)": crossattn_layer(d),
        "camera_inverse (MLP)": mlp(d),
        "token_transform (adaLN 조건 블록)": cond_block(d),
        "camera_attn (camera update)": crossattn_layer(d),
        "geo_norm + cam_norm": 2 * layernorm(d),
        "token_corr_res (matching + residual)": token_corr_res(d),
    }
    return parts


def main():
    D, E, P = 768, 1024, 16
    out = {}

    ba = ba_layer(D)
    ba_total = sum(ba.values())
    around = {
        "appearance_embed (1024→768)": linear(E, D),
        "pts head (768→(3+1)·16²)": linear(D, 4 * P * P),
        "pose head (768→512→512→512, rot9·t3·conf1)": linear(D, 512) + 2 * linear(512, 512)
        + linear(512, 3) + linear(512, 1) + linear(512, 9),
        "init_pose_token": D,
    }
    cam_enc = linear(8 * (1 + 2 * 4), 512) + linear(512, D)   # CameraTokenEncoder (추론에 안 씀)
    out["P1_decoder"] = {
        "ba_layer_parts_M": {k: round(v / 1e6, 3) for k, v in ba.items()},
        "ba_layer_total_M": round(ba_total / 1e6, 3),
        "around_M": {k: round(v / 1e6, 3) for k, v in around.items()},
        "ba_layer_plus_around_M": round((ba_total + sum(around.values())) / 1e6, 3),
        "camera_token_encoder_unused_M": round(cam_enc / 1e6, 3),
        "paper_decoder_M": 38,
        # 부록 A 서술대로라면: self-attn 3개에 MLP(비 4)가 각각 있고, 잔차 MLP 확장비가 4
        "appendixA_selfattn_with_mlp_M": round((ba_total + 3 * (layernorm(D) + mlp(D))) / 1e6, 2),
        "appendixA_plus_residual_mlp_ratio4_M": round((ba_total + 3 * (layernorm(D) + mlp(D))
                                                       + mlp(D) - 2 * linear(D, D)) / 1e6, 2),
    }

    enc_block = layernorm(E) + attention(E) + layernorm(E) + mlp(E)
    enc = linear(3 * P * P, E) + 24 * enc_block + layernorm(E)
    out["P2_encoder"] = {
        "vit_l_encoder_M": round(enc / 1e6, 2),
        "total_model_M": round((enc + ba_total + sum(around.values()) + cam_enc) / 1e6, 1),
        "decoder_share_of_total": round((ba_total + sum(around.values())) / (enc + ba_total + sum(around.values()) + cam_enc), 3),
    }

    dec_block = layernorm(D) + attention(D) + cross_attention(D) + layernorm(D) + layernorm(D) + mlp(D) + layernorm(D)
    out["P3_vista"] = {
        "decoderblock_768_M": round(dec_block / 1e6, 3),
        "12_layers_M": round(12 * dec_block / 1e6, 1),
        "3_layers_M": round(3 * dec_block / 1e6, 1),
        "paper_claim": "ViSTA 12층 113M, 3층 38M (Appendix B)",
        "implied_non_layer_M_if_linear": round((38 - 113 / 12 * 3), 1),
        "solve_X_plus_12L_113__X_plus_3L_38": {"L_M": round(75 / 9, 2), "X_M": round(113 - 12 * 75 / 9, 1)},
    }

    sizes = {"VGGT": 605, "MA": 171, "DA3": 765, "DUSt3R": 227, "MASt3R": 227, "ViSTA": 113, "BA-T": 38}
    out["P4_ratios"] = {
        "BA-T/DUSt3R": round(38 / 227, 3),
        "DUSt3R/BA-T": round(227 / 38, 2),
        "BA-T/VGGT": round(38 / 605, 3),
        "BA-T/DA3": round(38 / 765, 3),
        "BA-T/MapAnything": round(38 / 171, 3),
        "claims": "초록 16%(v1 본문: DUSt3R 대비), v2 본문 '최대 6배 큰 2-view 모델', '대형 multi-view의 5–6%'",
    }

    # P5: 연산량 추정. 선형층은 2·(파라미터)·(토큰 수), 어텐션 점수·가중합은 4·Nq·Nk·D
    N = (224 // P) ** 2
    lin = lambda params, tokens: 2 * params * tokens
    att = lambda nq, nk, d: 4 * nq * nk * d
    enc_flops = 2 * (lin(enc_block, N) * 24 + 24 * att(N, N, E) + lin(linear(3 * P * P, E), N))
    cb = cond_block(D)
    per_iter = {
        "token_transform x3 (이웃 2 + 잔차 1)": 3 * (lin(cb, N) + att(N, N, D)),
        "geometry cross-attn x2 view": 2 * (lin(crossattn_layer(D), N) + att(N, N, D)),
        "self-attn x3 x2 view": 2 * 3 * (lin(attention(D), N) + att(N, N, D)),
        "token_corr_res": lin(token_corr_res(D), N) + att(N, N, D),
        "camera_attn (query 1개, key N개)": lin(2 * linear(D, D), N) + lin(crossattn_layer(D), 1) + att(1, N, D),
    }
    it = sum(per_iter.values())
    heads = 2 * lin(linear(D, 4 * P * P), N) + lin(linear(E, D), 2 * N)
    out["P5_flops_224_pair"] = {
        "tokens_per_view": N,
        "encoder_2views_GFLOP": round(enc_flops / 1e9, 1),
        "ba_layer_per_iter_GFLOP": round(it / 1e9, 2),
        "ba_layer_parts_GFLOP": {k: round(v / 1e9, 2) for k, v in per_iter.items()},
        "heads_and_embed_GFLOP": round(heads / 1e9, 2),
        "total_K4_GFLOP": round((enc_flops + 4 * it + heads) / 1e9, 1),
        "total_K8_GFLOP": round((enc_flops + 8 * it + heads) / 1e9, 1),
        "note": "추정치(측정 아님). 인코더가 대부분을 차지한다",
    }

    path = ROOT / "analysis" / "param_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
