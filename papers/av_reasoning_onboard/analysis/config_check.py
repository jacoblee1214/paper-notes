"""공개 config(HF 모델 카드의 config.json)로 파라미터 수, 가중치 바이트, KV 캐시 바이트, 측정 메모리 산술을 확인한다 (CPU, 1초 이내).

실행: python3 analysis/config_check.py   → analysis/config_check_results.json
입력: paper/hf/nvidia_Alpamayo2-Edge_config.json, paper/hf/nvidia_alpajudge_config.json (gh/curl로 받은 작은 메타데이터)
가정은 코드 안 주석에 적었다. 모델 가중치는 받지 않는다.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / "paper" / "hf"
GIB = 2**30


def load(name):
    return json.loads((HF / name).read_text())


def lm_params(c, gated):
    """decoder-only LM 파라미터. gated=True면 SwiGLU(3행렬), False면 relu2 등 2행렬 MLP."""
    d, L = c["hidden_size"], c["num_hidden_layers"]
    H, KV, hd, I = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["intermediate_size"]
    attn = d * H * hd + 2 * d * KV * hd + H * hd * d
    mlp = (3 if gated else 2) * d * I
    return L * (attn + mlp), d * c["vocab_size"]  # (층 파라미터, 임베딩 1개 분량)


def main():
    edge = load("nvidia_Alpamayo2-Edge_config.json")
    vlm = edge["vlm_config"]
    tc, vc = vlm["text_config"], vlm["vision_config"]
    # Edge: 텍스트 MLP는 relu2(2행렬), 임베딩과 LM head는 tie_word_embeddings=False(따로)
    text_layers, emb = lm_params(tc, gated=False)
    text = text_layers + 2 * emb  # 입력 임베딩 + 별도 LM head
    # SigLIP2형 비전: 2행렬 MLP(gelu), 어텐션 4·d^2
    dv, Lv, Iv = vc["hidden_size"], vc["num_hidden_layers"], vc["intermediate_size"]
    vision = Lv * (4 * dv * dv + 2 * dv * Iv)
    # projector: 2x2 spatial merge로 4·dv 입력, 중간 11520, 출력 = text hidden. 두 층 MLP라고 가정(공개 config에 구조 없음)
    proj_in, proj_mid, proj_out = 4 * dv, vlm["projector_hidden_size"], tc["hidden_size"]
    projector = proj_in * proj_mid + proj_mid * proj_out
    # 액션 expert: 512 hidden, 28층, relu2 MLP, GQA 16/8 heads, head_dim 128
    ex = edge["expert_config"]["expert_update_cfg"]
    dx, Lx, Ix = ex["hidden_size"], ex["num_hidden_layers"], ex["intermediate_size"]
    Hx, KVx, hdx = ex["num_attention_heads"], ex["num_key_value_heads"], ex["head_dim"]
    expert = Lx * (dx * Hx * hdx + 2 * dx * KVx * hdx + Hx * hdx * dx + 2 * dx * Ix)

    backbone = text + vision + projector
    total = backbone + expert
    bf16_bytes = 2 * total
    kv_per_token_bf16 = tc["num_hidden_layers"] * tc["num_key_value_heads"] * tc["head_dim"] * 2 * 2  # K,V, 2 B
    prompt_tokens = 3236  # 모델 카드: 22장 장면의 프롬프트 토큰 수
    kv_prompt_bf16 = kv_per_token_bf16 * prompt_tokens

    # H100 측정치(모델 카드): peak allocated 8.40 GiB, 가중치 BF16 약 backbone+expert 분량
    peak_alloc_gib = 8.40
    weights_gib = bf16_bytes / GIB
    remainder = peak_alloc_gib - weights_gib

    # AlpaJudge RF (Qwen3-VL형, tie=True): 텍스트 gated MLP, 비전 24층 1024
    aj = load("nvidia_alpajudge_config.json")
    avlm = aj["vlm_config"]
    atc, avc = avlm["text_config"], avlm["vision_config"]
    a_text_layers, a_emb = lm_params(atc, gated=True)
    a_text = a_text_layers + a_emb  # tie=True이므로 임베딩 1회
    a_vis = avc["depth"] * (4 * avc["hidden_size"] ** 2 + 2 * avc["hidden_size"] * avc["intermediate_size"])
    a_merge = (4 * avc["hidden_size"]) * (4 * avc["hidden_size"]) + (4 * avc["hidden_size"]) * avc["out_hidden_size"]
    a_total = a_text + a_vis + a_merge

    out = {
        "edge_params_B": {
            "text_decoder_incl_embed_and_head": round(text / 1e9, 3),
            "vision_encoder": round(vision / 1e9, 3),
            "projector_assumed_2layer": round(projector / 1e9, 3),
            "backbone_total": round(backbone / 1e9, 3),
            "action_expert": round(expert / 1e9, 3),
            "total": round(total / 1e9, 3),
            "card_claims": {"backbone": 2.45, "language_model": 1.96, "siglip2": 0.41, "projector": 0.08, "expert": 0.2},
        },
        "edge_bytes": {
            "bf16_weights_GiB": round(bf16_bytes / GIB, 2),
            "fp8_weights_GB_if_all_fp8": round(total / 1e9, 2),
            "kv_per_token_bf16_bytes": kv_per_token_bf16,
            "kv_prompt_3236_tokens_bf16_MiB": round(kv_prompt_bf16 / 2**20, 1),
            "kv_prompt_3236_tokens_6_copies_GiB": round(6 * kv_prompt_bf16 / GIB, 2),
        },
        "h100_measured_check": {
            "peak_allocated_GiB_card": peak_alloc_gib,
            "weights_bf16_GiB": round(weights_gib, 2),
            "remainder_GiB": round(remainder, 2),
            "note": "잔여분은 활성값·ViT·KV·샘플링 버퍼 합. 6개 후보가 KV를 각자 가지면 6×KV가 잔여분 안에 들어간다(추정, 구조 미공개).",
        },
        "mac_fit": {
            "unified_memory_GiB": 24,
            "bf16_weights_fraction_of_24GiB": round(bf16_bytes / GIB / 24, 3),
            "note": "BF16 가중치 약 4.9 GiB는 24 GiB 통합 메모리에 여유 있게 들어간다. GPU 작업 한도는 별도 확인 필요.",
        },
        "alpajudge_rf_params_B": {
            "text_incl_tied_embed": round(a_text / 1e9, 3),
            "vision_24L_1024": round(a_vis / 1e9, 3),
            "merger_assumed": round(a_merge / 1e9, 3),
            "total": round(a_total / 1e9, 3),
            "bf16_GB": round(2 * a_total / 1e9, 2),
            "card_note": "notebook/README: 모델 로드에 가중치 약 5 GB (BF16 추정 약 4.1 GB; 차이는 미공개 구조 세부 때문일 수 있음)",
        },
    }
    (ROOT / "analysis" / "config_check_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
