"""MoE에서 병렬 초안 검증이 '토큰 하나 생성과 거의 같은 비용'(Sec. 6)인지 대역폭 모델로 따져 보는 toy (CPU, 수 초).

실행: python3 analysis/spec_decode_toy.py   (gguf_header_check.py, config_check.py 결과를 읽음)
가정: 디코드는 메모리 대역폭에 묶인다. 한 번의 forward는 그 forward가 건드리는 가중치 바이트를 한 번 읽는다.
     B개 토큰을 함께 검증하면 층마다 B개 토큰이 고른 expert의 합집합을 읽는다.
     라우팅은 (a) 토큰마다 독립·균등, (b) 완전 상관(모두 같은 expert) 두 극단으로 둔다. 실제는 그 사이.
결과: analysis/spec_decode_results.json
"""
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
G = json.loads((ROOT / "analysis/gguf_header_results.json").read_text())["main"]["components"]
CFG = json.loads((ROOT / "paper/hf/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16_config.json").read_text())["llm_config"]
MOE = [b for b in CFG["block_configs"] if b["block_type"] == "moe"]
K = 9
DRAFTER_GB = 0.96  # Table 2 (NVFP4/BF16 drafter)

routed_total = G["Routed-expert projections"]["GB"]
w = np.array([b["n_routed_experts"] * b["moe_intermediate_size"] for b in MOE], float)
routed_layer_GB = routed_total * w / w.sum()          # 층별 routed expert 바이트 (E_l x I_l에 비례)
E = np.array([b["n_routed_experts"] for b in MOE], float)
dense_GB = sum(v["GB"] for k, v in G.items() if k != "Routed-expert projections") - G["Token embedding"]["GB"]


def distinct_uniform(B):
    """토큰 B개가 각자 E개 중 K개를 균등하게(중복 없이) 고를 때 기대 고유 expert 수."""
    return E * (1 - (1 - K / E) ** B)


def bytes_per_forward(B, mode):
    touched = K * np.ones_like(E) if mode == "correlated" else distinct_uniform(B)
    return dense_GB + float((routed_layer_GB * touched / E).sum())


def main():
    out = {"dense_GB_per_forward": round(dense_GB, 3), "routed_GB_total": routed_total,
           "routed_GB_single_token": round(float((routed_layer_GB * K / E).sum()), 3)}
    one = bytes_per_forward(1, "uniform")
    out["GB_per_token_no_spec"] = round(one, 3)
    rows = []
    for B in (1, 2, 4, 8):
        for mode in ("uniform", "correlated"):
            rows.append({"B": B, "routing": mode, "GB_per_forward": round(bytes_per_forward(B, mode), 3),
                         "relative_to_single": round(bytes_per_forward(B, mode) / one, 3),
                         "mean_distinct_experts": round(float(distinct_uniform(B).mean()) if mode == "uniform" else K, 1)})
    out["verify_cost"] = rows
    # 속도 상한: speedup ≈ AL × (단일 토큰 바이트) / (검증 바이트 + drafter 바이트)
    spd = {}
    for name, al in (("SPEED-Bench temp0 NVFP4 drafter (Table 5)", 4.681), ("SPEED-Bench temp1 NVFP4 (Table 5)", 4.248),
                     ("AA AgentPerf natural (Table 6)", 4.18)):
        spd[name] = {mode: round(al * one / (bytes_per_forward(8, mode) + DRAFTER_GB), 2) for mode in ("uniform", "correlated")}
    out["speedup_upper_bound_B8"] = spd
    # 대역폭을 가정하면 기준 TPS 상한: DGX Spark·M4 Pro 공식 대역폭 273 GB/s(사전 지식, 확인 필요), RTX Spark 노트북은 미확인
    out["tps_bound_at_273GBps"] = {"no_spec": round(273 / one, 1),
                                   "with_spec_uniform_AL4.18": round(273 / one * spd["AA AgentPerf natural (Table 6)"]["uniform"], 1),
                                   "with_spec_correlated_AL4.18": round(273 / one * spd["AA AgentPerf natural (Table 6)"]["correlated"], 1),
                                   "reported_TPS_RTX_Spark": 83.99}
    (ROOT / "analysis/spec_decode_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
