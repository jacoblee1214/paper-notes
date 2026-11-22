"""공식 코드의 설정값(정적으로 읽은 숫자)에서 파라미터 수, 토큰 수, 메모리, FLOPs를 어림 계산한다.

코드를 실행하거나 가져오지 않는다. 아래 상수는 repo의 configuration_decal.py, modeling_decal.py(get_qwen_config,
Qwen3VLWithExpertModel, DeCALModel)와 launch/*.sh 에서 읽은 값이다. Qwen3-VL-2B 비전 탑 구성(깊이 24, 은닉 1024,
중간 4096, 출력 2048, deepstack 3)도 modeling_decal.py 에서 읽었다. 결과는 어림값이다(노름, 편향, 위치 임베딩 생략).
실행: python3 analysis/code_budget.py   (표준 라이브러리만, 즉시)
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "code_budget_results.json"


def attn(hidden, heads, kv_heads, head_dim):
    q = hidden * heads * head_dim
    kv = 2 * hidden * kv_heads * head_dim
    o = heads * head_dim * hidden
    return q + kv + o


def mlp(hidden, inter):
    return 3 * hidden * inter


def block(hidden, inter, heads, kv_heads, head_dim):
    return attn(hidden, heads, kv_heads, head_dim) + mlp(hidden, inter)


def main():
    r = {}
    L = 28
    # 이해 전문가: Qwen3-VL-2B 텍스트 (qwen3_vl_28l: hidden 2048, inter 6144, 16 heads, 8 kv heads, head_dim 128)
    und_text_layers = L * block(2048, 6144, 16, 8, 128)
    vocab = 151936
    und_embed = vocab * 2048  # tie_word_embeddings = True 이므로 한 번만 센다
    # 비전 탑: depth 24, hidden 1024, inter 4096 (일반 MLP 2개 행렬), heads 16
    vit = 24 * (4 * 1024 * 1024 + 2 * 1024 * 4096)
    merger = 4 * (4096 * 4096 + 4096 * 2048)  # 주 merger 1 + deepstack 3
    und = und_text_layers + und_embed + vit + merger
    # 생성·행동 전문가: qwen3_28l (hidden 1024, inter 3072, 16 heads, 8 kv heads, head_dim 128), 임베딩 없음
    expert = L * block(1024, 3072, 16, 8, 128)
    # 촉각 교차어텐션: 층마다 nn.MultiheadAttention(embed_dim=2048, heads=8) = 4 * 2048^2
    tac_xattn = L * 4 * 2048 * 2048
    tac_gate = 2048 * 2048 + 2048  # LayerNorm 제외. 입력은 hidden_size(2048) 차원
    # 촉각 인코더(MultiFingerTactileEncoder, hidden 2048): CNN 3층 + 1x1 + Transformer 2층
    cnn = 3 * 64 + 64 * 3 * 9 + 64 * 128 * 9 + 128 * 256 * 9 + 256 * 2048
    tenc = cnn + 2 * (4 * 2048 * 2048 + 2 * 2048 * 8192)
    # 촉각 생성 가지: 힘 인코더 12->1024, 토큰 투영 1024->1024, 잠재 투영 1024->256, 디코더 3개(local-token 디코더 구조)
    dec = 256 * 64 * 14 * 14 + 64 * 64 * 16 + 64 * 32 * 16 + 32 * 32 * 16 + 32 * 3 * 16
    tdec_enc = 2048 * 64 * 14 * 14 + 64 * 64 * 16 + 64 * 32 * 16 + 32 * 32 * 16 + 32 * 3 * 16  # 인코더 보조 복원 디코더(token_dim 2048)
    gen_branch = 12 * 1024 + 1024 * 1024 + 1024 * 256 + 2 * dec + 256 * 6
    # Cosmos 투영 (생성 전문가 입력/출력)
    cosmos_proj = 16 * 1024 + 1024 * 1024 * 64 + 1024 * 1024 * 64 + 1024 * 16
    # 행동 입출력
    act_io = 12 * 1024 + 44 * 1024 + 1024 * 12 + 1024 * 44 + 56 * 1024 + 2048 * 1024 + 1024 * 1024
    total = und + 2 * expert + tac_xattn + tac_gate + tenc + gen_branch + tdec_enc + cosmos_proj + act_io
    r["params"] = {
        "und_expert_qwen3vl2b": und, "gen_expert": expert, "act_expert": expert,
        "tactile_cross_attn_28_layers": tac_xattn, "tactile_gate": tac_gate, "tactile_encoder": tenc,
        "tactile_gen_branch_incl_decoders": gen_branch, "tactile_encoder_recon_decoder": tdec_enc,
        "cosmos_proj_and_down_up": cosmos_proj, "action_io": act_io, "total": total,
        "total_B": round(total / 1e9, 2),
        "note": "Cosmos 토크나이저(CI8x8, 동결, jit)와 위치 임베딩은 제외. 논문 부록 B는 InternVLA-A1-3B 초기화라고 쓴다.",
    }
    r["memory_GB"] = {
        "weights_bf16": round(total * 2 / 1e9, 1),
        "weights_fp32_deploy": round(total * 4 / 1e9, 1),
        "train_full_adamw_fp32_states_rough": round(total * (2 + 2 + 4 + 4) / 1e9, 1),
        "note": "배포 스크립트는 policy.to(torch.float32)로 올린다(deploy_policy_tac_server.py). 학습 메모리는 bf16 가중치 + bf16 기울기 + fp32 Adam 두 상태 가정의 하한 어림이며 활성값 제외.",
        "mac_unified_memory_GiB": 24,
    }
    # 토큰 수 (3 시점 중 입력 2개 사용, 3 카메라)
    views, hist = 3, 2
    cosmos_tokens_per_frame = (32 // 8) * (32 // 8)  # 256x256 -> 32x32 latent -> 8x8 conv stride 8 -> 4x4
    gen_img = views * hist * cosmos_tokens_per_frame
    gen_tac = 10
    vlm_img_per_view = (224 // 32) ** 2  # patch 16, merge 2 -> 28x28 가정이 아니라 224/32 = 7 -> 49
    prefix = views * (vlm_img_per_view + 2) + 48  # 이미지 토큰 + 시작/끝 + 최대 길이 48 언어 토큰
    suffix = 1 + 2 * 50
    seq = prefix + gen_img + gen_tac + suffix
    r["tokens"] = {"cosmos_tokens_per_frame": cosmos_tokens_per_frame, "gen_image_tokens": gen_img, "gen_tactile_tokens": gen_tac,
                   "prefix_tokens_est": prefix, "suffix_tokens(chunk50)": suffix, "total_seq_est": seq,
                   "note": "Qwen3-VL 이미지 토큰 수는 프로세서 리사이즈 규칙에 따라 달라질 수 있어 어림(224/32)^2 = 49로 둠."}
    # 추론 FLOPs 어림: 이해(prefix) 1회 + 생성(middle) 1회 + 행동 10 디노이징 스텝(KV 재사용, suffix만)
    f_und = 2 * (und_text_layers + 0) * prefix + 2 * vit * 3 * (224 // 16) ** 2
    f_gen = 2 * expert * (gen_img + gen_tac)
    f_act = 10 * 2 * expert * suffix
    f_tac_x = 2 * tac_xattn * 0  # 교차어텐션 키·값 투영은 10 토큰이라 무시
    flops = f_und + f_gen + f_act
    r["flops_per_chunk_G"] = {"und": round(f_und / 1e9, 1), "gen": round(f_gen / 1e9, 1), "act_10steps": round(f_act / 1e9, 1), "total": round(flops / 1e9, 1),
                              "note": "어텐션 점수 계산과 Cosmos 인코더 FLOPs 제외 어림. 논문 보고: RTX 4090에서 0.27 s/청크(부록 A.2)."}
    # 이 맥 어림: M4 Pro MPS fp16 유효 처리량을 3-8 TFLOPS로 가정(측정 안 함)
    r["mac_latency_guess_s"] = {"at_3_TFLOPS": round(flops / 3e12, 2), "at_8_TFLOPS": round(flops / 8e12, 2),
                                 "note": "가정값이다. 측정하지 않았다. 메모리 대역폭(약 270 GB/s)이 아니라 연산 대역으로 단순화."}
    # 원문 비교: 4090 bf16 0.27 s 에서 역산한 유효 처리량
    r["effective_tflops_implied_by_paper_latency"] = round(flops / 0.27 / 1e12, 2)
    OUT.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
