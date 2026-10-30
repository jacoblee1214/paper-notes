"""공개 config.json으로 전체·활성 파라미터, 런타임 메모리, 이 맥(24 GiB 통합 메모리)에 들어가는지를 계산한다 (CPU, 수 초).

실행: python3 analysis/config_check.py   (gguf_header_check.py를 먼저 돌려 두면 GGUF 실측값과도 대조한다)
- 입력: paper/hf/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16_config.json (HF, rev e09a4ce)
- 결과: analysis/config_check_results.json
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "paper/hf/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16_config.json").read_text())
L = CFG["llm_config"]
GB = 1e9

H = L["hidden_size"]                     # 4096
LAT = L["moe_latent_size"]               # 1024
SH = L["moe_shared_expert_intermediate_size"]  # 5376
NH, HD, NKV = L["num_attention_heads"], L["head_dim"], L["num_key_value_heads"]
MH, MHD, NG, DS, CK = L["mamba_num_heads"], L["mamba_head_dim"], L["n_groups"], L["ssm_state_size"], L["conv_kernel"]
V = L["vocab_size"]
D_INNER = MH * MHD                       # 8192
CONV_DIM = D_INNER + 2 * NG * DS         # 10240
IN_PROJ = 2 * D_INNER + 2 * NG * DS + MH  # 18560


def layer_params(b):
    """블록 하나의 파라미터를 구성 요소별로 (relu2 MLP라 gate 없음, bias 없음)."""
    t = b["block_type"]
    if t == "mamba":
        return {"mamba_inout": H * IN_PROJ + D_INNER * H, "mamba_conv": CONV_DIM * CK + CONV_DIM,
                "mamba_small": 3 * MH + D_INNER + H}
    if t == "attention":
        return {"attention": H * NH * HD + 2 * H * NKV * HD + NH * HD * H, "norm": H}
    E, I = b["n_routed_experts"], b["moe_intermediate_size"]
    return {"routed": E * 2 * LAT * I, "shared": 2 * H * SH, "latent": 2 * H * LAT, "router": H * E + E, "norm": H}


def count(blocks, top_k=None):
    tot, act = {}, {}
    for b in blocks:
        p = layer_params(b)
        for k, v in p.items():
            tot[k] = tot.get(k, 0) + v
            if k == "routed":
                k_used = top_k or b["num_experts_per_tok"]
                act[k] = act.get(k, 0) + v * k_used / b["n_routed_experts"]
            else:
                act[k] = act.get(k, 0) + v
    for d in (tot, act):
        d["embedding"] = V * H
        d["lm_head"] = V * H
    act["embedding"] = 0  # 토큰 하나는 임베딩 한 행만 읽는다
    return tot, act


def main():
    out = {}
    bolt = L["block_configs"]
    moe = [b for b in bolt if b["block_type"] == "moe"]
    out["layers"] = {t: sum(b["block_type"] == t for b in bolt) for t in ("mamba", "moe", "attention")}
    out["E_set"] = sorted({b["n_routed_experts"] for b in moe})
    out["I_set"] = sorted({b["moe_intermediate_size"] for b in moe})
    out["I_hist"] = {str(i): sum(b["moe_intermediate_size"] == i for b in moe) for i in out["I_set"]}
    out["E_hist"] = {str(e): sum(b["n_routed_experts"] == e for b in moe) for e in out["E_set"]}
    out["top_k"] = sorted({b["num_experts_per_tok"] for b in moe})
    out["mean_E"] = round(sum(b["n_routed_experts"] for b in moe) / len(moe), 1)

    tot, act = count(bolt)
    out["bolt_total_B"] = round(sum(tot.values()) / 1e9, 3)
    out["bolt_total_by_component_B"] = {k: round(v / 1e9, 4) for k, v in tot.items()}
    out["bolt_active_B"] = round(sum(act.values()) / 1e9, 3)
    out["bolt_active_by_component_B"] = {k: round(v / 1e9, 4) for k, v in act.items()}
    out["bolt_active_without_lm_head_B"] = round((sum(act.values()) - act["lm_head"]) / 1e9, 3)

    # 부모(Super VL 120B-A12B): Table 1 값(전 층 512 experts, 폭 2688, top-22)으로 같은 공식
    parent = [dict(b, n_routed_experts=512, moe_intermediate_size=2688, num_experts_per_tok=22)
              if b["block_type"] == "moe" else b for b in bolt]
    ptot, pact = count(parent)
    out["parent_total_B"] = round(sum(ptot.values()) / 1e9, 3)
    out["parent_active_B"] = round(sum(pact.values()) / 1e9, 3)
    out["routed_kept_fraction"] = round(tot["routed"] / ptot["routed"], 4)
    out["active_routed_fraction_vs_parent"] = round(act["routed"] / pact["routed"], 4)

    # GGUF 헤더 실측과 대조 (있으면)
    gpath = ROOT / "analysis/gguf_header_results.json"
    if gpath.exists():
        g = json.loads(gpath.read_text())["main"]
        out["gguf_total_params"] = g["total_params"]
        out["formula_minus_gguf_params"] = int(sum(tot.values()) - g["total_params"])

    # 런타임 상태 (토큰당, 바이트)
    kv_per_tok = lambda bytes_per: out["layers"]["attention"] * 2 * NKV * HD * bytes_per
    ssm_state = lambda bytes_per: out["layers"]["mamba"] * (MH * MHD * DS + CONV_DIM * (CK - 1)) * bytes_per
    out["kv_bytes_per_token"] = {"FP8": kv_per_tok(1), "F16": kv_per_tok(2)}
    out["kv_GB_at_ctx"] = {f"{c // 1024}K": {"FP8": round(kv_per_tok(1) * c / GB, 3), "F16": round(kv_per_tok(2) * c / GB, 3)}
                           for c in (8192, 32768, 65536, 131072, 262144)}
    out["ssm_state_GB"] = {"BF16 (Table 2 가정)": round(ssm_state(2) / GB, 4),
                           "FP32 (config mamba_ssm_cache_dtype)": round(ssm_state(4) / GB, 4)}
    out["table2_check"] = {"KV_FP8_128K_table": 0.54, "KV_FP8_128K_calc": out["kv_GB_at_ctx"]["128K"]["FP8"],
                           "SSM_BF16_table": 0.08, "SSM_BF16_calc": out["ssm_state_GB"]["BF16 (Table 2 가정)"]}
    out["bf16_size_GB"] = round(sum(tot.values()) * 2 / GB, 2)

    # 이 맥: hw.memsize 25,769,803,776 B(24 GiB). GPU 기본 한도(recommendedMaxWorkingSetSize)는 실측 안 함 → 2/3 가정
    ram = 25769803776 / GB
    gpu_default = ram * 2 / 3
    os_reserve = 4.0  # macOS + 기본 앱 예약(가정)
    compute_buf = 0.5  # llama.cpp compute buffer(가정)
    files = {"공식 24GB IQ2_XXS (main)": 22.0183, "Unsloth UD-IQ1_M": 20.3075, "Unsloth UD-IQ1_S": 18.5078,
             "커뮤니티 REAP50 16GB": 13.38, "E4B Q4_K_M (gemma4 계열)": 6.1804}
    rows = []
    for name, w in files.items():
        for ctx in (8192, 32768, 131072):
            kv = kv_per_tok(2) * ctx / GB if "E4B" not in name else None  # E4B는 구조가 달라 KV 계산 안 함
            need = w + (kv or 0) + (ssm_state(4) / GB if "E4B" not in name else 0) + compute_buf
            rows.append({"file": name, "weights_GB": w, "ctx": ctx, "kv_f16_GB": None if kv is None else round(kv, 3),
                         "need_GB": round(need, 2),
                         "fits_gpu_default_2of3": need <= gpu_default,
                         "fits_ram_minus_os": need <= ram - os_reserve})
    out["mac"] = {"ram_GB": round(ram, 3), "gpu_default_assumed_GB": round(gpu_default, 2), "os_reserve_assumed_GB": os_reserve,
                  "compute_buffer_assumed_GB": compute_buf, "vision_mmproj_GB": 1.7337, "drafter_gguf_GB": 0.9982,
                  "rows": rows}
    (ROOT / "analysis/config_check_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k in ("layers", "E_set", "I_set", "I_hist", "E_hist", "top_k", "mean_E", "bolt_total_B", "bolt_active_B",
              "bolt_active_without_lm_head_B", "parent_total_B", "parent_active_B", "routed_kept_fraction",
              "active_routed_fraction_vs_parent", "gguf_total_params", "formula_minus_gguf_params", "kv_GB_at_ctx",
              "ssm_state_GB", "table2_check", "bf16_size_GB"):
        print(k, out.get(k))
    print("by component", out["bolt_total_by_component_B"])
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
