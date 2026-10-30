"""공개 체크포인트 메타데이터로 보고서의 구조·파라미터·메모리 주장을 검산한다 (CPU, numpy). 실행: python3 analysis/param_check.py

입력: paper/hf/ 의 Hugging Face 메타데이터(config.json, safetensors 헤더에서 읽은 텐서 모양, 양자화 설정).
      원본 파일은 커서 git에 넣지 않으므로(.gitignore), 첫 실행에서 analysis/hf_meta_summary.json 으로 요약해 두고
      원본이 없으면 그 요약만으로 계산한다. 가중치 파일은 받지 않는다.
결과: analysis/param_check_results.json
"""
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / "paper" / "hf"
SUMMARY = ROOT / "analysis" / "hf_meta_summary.json"
GiB = 1024 ** 3


def build_summary() -> dict:
    """원본 HF 메타데이터 → 텐서 묶음별 파라미터 수, config 일부, 인덱스 총량, 양자화 집계."""
    shapes = json.load(open(HF / "bf16_shapes.json"))
    groups = Counter()
    dtypes = Counter()
    for name, (dt, shape) in shapes.items():
        groups[re.sub(r"\.\d+\.", ".#.", name)] += math.prod(shape) if shape else 1
        dtypes[dt] += 1
    cfg = json.load(open(HF / "bf16_config.json"))
    llm = cfg["llm_config"]
    keep = ["hidden_size", "layers_block_type", "mamba_head_dim", "mamba_num_heads", "ssm_state_size", "n_groups",
            "conv_kernel", "num_attention_heads", "num_key_value_heads", "head_dim", "n_routed_experts",
            "num_experts_per_tok", "moe_intermediate_size", "moe_latent_size", "moe_shared_expert_intermediate_size",
            "n_shared_experts", "max_position_embeddings", "vocab_size", "mamba_ssm_cache_dtype",
            "num_nextn_predict_layers", "mtp_layers_block_type"]
    q = json.load(open(HF / "nvfp4_hf_quant_config.json"))["quantization"]
    qtally = Counter()
    for k, v in q["quantized_layers"].items():
        qtally[f'{re.sub(r"[.]d+[.]", ".#.", re.sub(r"[.][0-9]+[.]", ".#.", k))} :: {v.get("quant_algo")}'] += 1
    excl = Counter(re.sub(r"\.\d+", ".#", e) for e in q["exclude_modules"])
    nvcfg = json.load(open(HF / "nvfp4_config.json"))["llm_config"]
    return {
        "source": {
            "bf16_repo": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16@fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe",
            "nvfp4_repo": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4@da5b2650a835292c678994a1fa3fb653250fb508",
            "note": "텐서 모양은 BF16 safetensors 64개 샤드의 헤더(앞부분)만 읽어 얻음. 가중치는 받지 않음",
        },
        "tensor_count": len(shapes),
        "dtypes": dict(dtypes),
        "param_groups": dict(groups),
        "llm_config": {k: llm[k] for k in keep},
        "nvfp4_llm_config_ssm_cache_dtype": nvcfg.get("mamba_ssm_cache_dtype"),
        "vision": {k: cfg["vision_config"][k] for k in ("hidden_size", "num_hidden_layers", "patch_size", "max_num_patches")}
        | {"projector_hidden_size": cfg["projector_hidden_size"], "downsample_ratio": cfg["downsample_ratio"]},
        "index_total_size_bytes": {
            "bf16": json.load(open(HF / "bf16_index.json"))["metadata"]["total_size"],
            "nvfp4": json.load(open(HF / "nvfp4_index.json"))["metadata"]["total_size"],
        },
        "nvfp4_quant": {"kv_cache_quant_algo": q["kv_cache_quant_algo"], "quantized_layers": dict(qtally),
                        "exclude_modules": dict(excl)},
    }


def main():
    if (HF / "bf16_shapes.json").exists():
        s = build_summary()
        SUMMARY.write_text(json.dumps(s, indent=1, ensure_ascii=False))
    else:
        s = json.load(open(SUMMARY))
    g = s["param_groups"]
    c = s["llm_config"]
    out = {}

    def tot(pred):
        return sum(v for k, v in g.items() if pred(k))

    bb = lambda k: k.startswith("language_model.backbone.")
    routed = tot(lambda k: bb(k) and ".experts.#." in k)
    mtp = tot(lambda k: k.startswith("language_model.mtp."))
    vision = tot(lambda k: k.startswith("vision_model."))
    projector = tot(lambda k: k.startswith("mlp1.") or k.startswith("vision_projector"))
    emb = g["language_model.backbone.embeddings.weight"]
    head = g["language_model.lm_head.weight"]
    total = sum(g.values())
    backbone = tot(bb)
    llm = backbone + head  # 백본(임베딩 포함) + 출력층, MTP 제외

    types = Counter(c["layers_block_type"])
    n_moe = types["moe"]
    per_expert = 2 * c["moe_latent_size"] * c["moe_intermediate_size"]  # up + down (relu², gate 없음)
    assert abs(routed - n_moe * c["n_routed_experts"] * per_expert) < 1e3
    active_routed = n_moe * c["num_experts_per_tok"] * per_expert
    active_llm = (llm - routed) + active_routed
    out["P1_structure"] = {
        "layers": dict(types),
        "pattern": "".join({"mamba": "M", "moe": "E", "attention": "*"}[t] for t in c["layers_block_type"]),
        "routed_experts": c["n_routed_experts"], "top_k": c["num_experts_per_tok"],
        "latent_moe": f'{c["hidden_size"]} → {c["moe_latent_size"]} (latent) → expert {c["moe_intermediate_size"]}',
        "attention": f'{c["num_attention_heads"]} q heads, {c["num_key_value_heads"]} kv heads, head_dim {c["head_dim"]}',
        "mamba": f'{c["mamba_num_heads"]} heads × {c["mamba_head_dim"]}, state {c["ssm_state_size"]}, groups {c["n_groups"]}',
        "max_position_embeddings": c["max_position_embeddings"],
        "vision": s["vision"],
    }
    out["P2_params_B"] = {
        "total_all": round(total / 1e9, 3),
        "report_N": 124.453,
        "llm_without_mtp_vision": round(llm / 1e9, 3),
        "mtp": round(mtp / 1e9, 3),
        "vision_encoder": round(vision / 1e9, 3),
        "projector": round(projector / 1e9, 3),
        "routed_experts_share_of_total": round(routed / total, 4),
        "active_llm_per_token_incl_embedding": round(active_llm / 1e9, 3),
        "active_llm_per_token_excl_embedding": round((active_llm - emb) / 1e9, 3),
        "active_routed_per_token": round(active_routed / 1e9, 3),
        "active_fraction_of_llm": round(active_llm / llm, 4),
        # 토큰당 활성 파라미터를 모듈별로 (임베딩 제외, 단위 B)
        "active_breakdown_B": {
            "routed_experts(top-22)": round(active_routed / 1e9, 3),
            "mamba_in_out_proj": round(tot(lambda k: bb(k) and (".mixer.in_proj." in k or ".mixer.out_proj." in k)) / 1e9, 3),
            "shared_experts": round(tot(lambda k: bb(k) and ".shared_experts." in k) / 1e9, 3),
            "latent_proj+router": round(tot(lambda k: bb(k) and ("_latent_proj." in k or ".mixer.gate." in k)) / 1e9, 3),
            "attention_qkvo": round(tot(lambda k: bb(k) and any(f".mixer.{p}_proj." in k for p in "qkvo")) / 1e9, 3),
            "lm_head": round(head / 1e9, 3),
        },
        "automodel_doc": {
            "claim": "120.7B trainable with vision tower frozen (99.31%), ~227 GB bf16, 63 shards",
            "llm_without_mtp_vision_B": round(llm / 1e9, 3),
            "share_of_model_without_mtp": round(llm / (total - mtp), 4),
            "total_minus_mtp_bf16_GiB": round((total - mtp) * 2 / GiB, 1),
            "index_files": 64, "model_shards": 63,  # 64번째는 vision_model.summary_idxs (I64 [2]) 하나뿐인 파일
        },
    }
    # 체크포인트 크기와 BPE (보고서 §4.1: BPE = 8S/N)
    sz = s["index_total_size_bytes"]
    out["P3_checkpoint"] = {
        "bf16_GB": round(sz["bf16"] / 1e9, 2), "bf16_GiB": round(sz["bf16"] / GiB, 1),
        "nvfp4_GB": round(sz["nvfp4"] / 1e9, 3), "report_S_GB": 74.987,
        "bpe_from_index": round(8 * sz["nvfp4"] / total, 3),
        "bpe_from_report_S": round(8 * 74.987e9 / total, 3),
        "bf16_bits_per_param": round(8 * sz["bf16"] / total, 3),
        "quant_tally": s["nvfp4_quant"],
        "nvfp4_config_ssm_cache_dtype": s["nvfp4_llm_config_ssm_cache_dtype"],
        "report_says_runtime_ssm_cache": "FP16 + stochastic rounding (Table 12 Aggressive, §4.5)",
    }
    # 요청당 상태 메모리: KV 캐시(어텐션 8층) vs Mamba 상태(40층)
    n_attn, n_mamba = types["attention"], types["mamba"]
    kv_elems_per_tok = n_attn * 2 * c["num_key_value_heads"] * c["head_dim"]
    ssm_elems = n_mamba * c["mamba_num_heads"] * c["mamba_head_dim"] * c["ssm_state_size"]
    conv_dim = c["mamba_num_heads"] * c["mamba_head_dim"] + 2 * c["n_groups"] * c["ssm_state_size"]
    conv_elems = n_mamba * conv_dim * (c["conv_kernel"] - 1)
    kv_b = {"BF16": 2.0, "FP8": 1.0, "NVFP4": 4.5 / 8}
    ssm_b = {"FP32": 4, "FP16": 2}
    per_layer_ssm_fp16 = c["mamba_num_heads"] * c["mamba_head_dim"] * c["ssm_state_size"] * 2
    per_layer_kv_fp8_tok = 2 * c["num_key_value_heads"] * c["head_dim"] * 1
    out["P4_state_memory"] = {
        "kv_bytes_per_token": {k: kv_elems_per_tok * v for k, v in kv_b.items()},
        "kv_GiB_at_1M_tokens": {k: round(kv_elems_per_tok * v * 2 ** 20 / GiB, 2) for k, v in kv_b.items()},
        "ssm_state_MiB_per_request": {k: round(ssm_elems * v / 2 ** 20, 1) for k, v in ssm_b.items()},
        "conv_state_MiB_per_request_bf16": round(conv_elems * 2 / 2 ** 20, 2),
        "tokens_per_mamba_page_check": {
            "one_layer_ssm_fp16_bytes": per_layer_ssm_fp16,
            "one_layer_kv_fp8_bytes_per_token": per_layer_kv_fp8_tok,
            "tokens": per_layer_ssm_fp16 // per_layer_kv_fp8_tok,
            "report": "one Mamba cache page ≈ 4K tokens of Attention KV (§5.3)",
        },
        "crossover_tokens_ssm_fp16_vs_kv_fp8": round(ssm_elems * 2 / (kv_elems_per_tok * 1)),
    }
    # 이 맥에서 돌릴 수 있는가: 체크포인트·공개 양자화 크기 vs 통합 메모리
    mac_total_GiB = 24.0
    mac_gpu_budget_GiB = round(mac_total_GiB * 0.75, 1)  # macOS 기본 GPU 사용 한도 근사(가정)
    sizes_GB = {  # HF API(2026-10-30) 파일 크기. 시각 모듈·MTP 별도 파일 제외
        "BF16 (공식)": sz["bf16"] / 1e9, "NVFP4 (공식)": sz["nvfp4"] / 1e9,
        "GGUF Q8_0 (ggml-org)": 128.5, "GGUF Q4_K_M (lmstudio-community)": 70.1, "GGUF Q4_0 (ggml-org)": 71.9,
        "MLX 4bit (Duckets)": 69.4, "MLX 3bit (rapid-mlx)": 42.7,
        "참고: Lightning 30B-A3B GGUF Q4_0": 18.9,
    }
    out["P5_local_fit"] = {
        "assumptions": {"mac_unified_GiB": mac_total_GiB, "gpu_budget_GiB_assumed": mac_gpu_budget_GiB},
        "sizes_GiB": {k: round(v * 1e9 / GiB, 1) for k, v in sizes_GB.items()},
        "fits_budget": {k: v * 1e9 / GiB <= mac_gpu_budget_GiB for k, v in sizes_GB.items()},
        "min_bits_per_param_to_fit_weights_only": round(mac_gpu_budget_GiB * GiB * 8 / total, 2),
    }
    # 배치 1 decode에서 토큰마다 읽는 가중치 바이트 (공개 NVFP4 레시피: expert 4.5비트, Mamba·attention·lm_head FP8, latent BF16)
    ab = out["P2_params_B"]["active_breakdown_B"]
    nvfp4_bytes = ((ab["routed_experts(top-22)"] + ab["shared_experts"]) * 4.5 / 8 + (ab["mamba_in_out_proj"]
                   + ab["attention_qkvo"] + ab["lm_head"]) * 1 + ab["latent_proj+router"] * 2) * 1e9
    out["P5_local_fit"]["decode_bs1_weight_read"] = {
        "NVFP4_recipe_GB_per_token": round(nvfp4_bytes / 1e9, 2),
        "Q4_4.5bit_all_GB_per_token": round(sum(ab.values()) * 4.5 / 8, 2),
        "upper_bound_tok_s_at_273GBps": {"NVFP4_recipe": round(273e9 / nvfp4_bytes, 1),
                                          "Q4_4.5bit_all": round(273 / (sum(ab.values()) * 4.5 / 8), 1)},
        "note": "273 GB/s = M4 Pro·DGX Spark 메모리 대역폭. KV·상태 읽기와 MTP 효과 제외, 들어간다고 가정한 상한",
    }
    path = ROOT / "analysis" / "param_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in out.items():
        print(k, json.dumps(v, ensure_ascii=False)[:600])
    print("wrote", path)


if __name__ == "__main__":
    main()
