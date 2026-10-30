"""논문 표·본문 수치 재계산 (CPU, 1초 미만). 실행: python3 analysis/table_check.py → analysis/table_check_results.json

- C1 Table 2 합계와 85.1% 크기 감소, "24GB"가 무엇을 포함하는지
- C2 압축 학습 토큰·배치 산술 (Sec. 2.3)
- C3 Table 5 Overall = 범주 평균인지, NVFP4 drafter의 AL 손실
- C4 Table 7: BF16 → 2비트 손실, 경쟁 모델 대비 승패, 블로그 주장(PinchBench #1)
- C5 Table 8 속도·재생 시간 비율
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]

TABLE2 = {  # 구성 요소: (형식, 비트, GB)
    "Attention projections": ("Q8_0", 8.5, 0.30), "Routed-expert projections": ("IQ2_XXS", 2.0625, 17.05),
    "Shared-expert projections": ("IQ2_XXS", 2.0625, 0.45), "Latent MoE projections": ("Q8_0", 8.5, 0.36),
    "Mamba input/output projections": ("NVFP4", 4.5, 2.47), "Token embedding": ("Q8_0", 8.5, 0.57),
    "LM head": ("Q8_0", 8.5, 0.57), "MoE router weights": ("F32", 32, 0.23), "Mamba convolution weights": ("F32", 32, 0.01)}
RUNTIME = {"Mamba SSM cache": 0.08, "KV cache": 0.54, "Activation buffer": 0.25}
DRAFTER, VISION = 0.96, 1.68

COLS = ["Super 120B BF16", "Bolt BF16", "Bolt 24GB", "Qwen3.6 35B-A3B BF16", "Gemma4 26B-A4B",
        "Lightning 30B-A3B", "Nemotron3 Super 120B BF16"]
T7 = {  # Table 7 (None = 표에 값 없음)
    "GDPVal-AA-V2": [34.48, 31.13, 28.47, 25.75, 15.33, 18.26, 12.27],
    "Terminal-Bench 2.1": [61.94, 53.93, 50.98, 44.38, 37.22, 23.46, 39.61],
    "Tau3 Banking": [19.85, 17.01, 17.01, 10.52, 14.02, 9.48, 12.37],
    "HLE (no tools)": [31.84, 28.73, 25.30, 19.56, 17.42, 10.47, 20.30],
    "AA-Omniscience accuracy": [25.98, 21.80, 16.90, 19.47, 22.17, 16.63, 26.68],
    "SciCode": [47.27, 44.49, 41.02, 35.33, 40.28, 31.38, 35.11],
    "GPQA Diamond": [90.78, 88.83, 86.43, 83.40, 79.61, 75.57, 78.60],
    "AA-LCR": [68.81, 67.38, 64.06, 61.06, 57.56, 49.19, 58.44],
    "AA-Omniscience non-hallucination": [69.24, 56.14, 61.27, 65.85, 14.22, 68.81, 14.64],
    "PinchBench": [90.25, 90.35, 87.82, 88.07, 74.70, 83.43, 80.36],
    "MCP-Atlas": [51.20, 55.60, 52.20, 51.2, None, 46.4, 52.8],  # Qwen은 공식 보고 62.8, 저자 재현 51.2
    "BrowseComp": [71.50, 70.00, 66.75, 48.74, 26.30, 36.81, 22.77],
    "WebArena (vision)": [57.40, 50.25, 50.86, 36.60, 38.2, None, None],
    "IFBench (instruction loose)": [73.04, 68.71, 66.17, 63.71, 77.25, 72.88, 71.92]}
T5 = {  # Table 5: 범주 → (BF16 t0, BF16 t1, NVFP4 t0, NVFP4 t1)
    "coding": (5.738, 5.275, 5.693, 5.180), "rag": (5.469, 5.117, 5.457, 5.083),
    "multilingual": (5.565, 5.029, 5.529, 4.955), "summarization": (5.406, 4.863, 5.215, 4.786),
    "math": (4.939, 4.516, 4.947, 4.457), "reasoning": (4.796, 4.447, 4.799, 4.450), "qa": (4.326, 3.900, 4.335, 3.923),
    "stem": (4.119, 3.645, 4.092, 3.629), "humanities": (4.073, 3.522, 3.986, 3.492),
    "writing": (3.725, 3.421, 3.698, 3.358), "roleplay": (3.943, 3.428, 3.747, 3.418)}
T5_OVERALL = (4.736, 4.288, 4.681, 4.248)
T8 = {"Bolt 24GB + LiLiCorr": (83.99, 600.80), "Qwen3.6 35B-A3B Q4_K_M + MTP": (96.63, 453.97),
      "Gemma4 26B-A4B Q4_K_M + MTP": (100.56, 480.68), "Qwen3.8 27B Q4_K_M + DFlash": (28.26, 1043.28)}


def main():
    out = {}
    w = sum(v[2] for v in TABLE2.values())
    rt = sum(RUNTIME.values())
    out["C1"] = {"weight_subtotal": round(w, 2), "runtime_subtotal": round(rt, 2), "combined": round(w + rt, 2),
                 "with_drafter": round(w + rt + DRAFTER, 2), "with_drafter_and_vision": round(w + rt + DRAFTER + VISION, 2),
                 "24GiB_in_GB": round(24 * 2**30 / 1e9, 2),
                 "reduction_paper_22.1": round(1 - 22.1 / 148, 4), "reduction_table2_22.01": round(1 - 22.01 / 148, 4),
                 "experts_share_of_weights": round((17.05 + 0.45) / w, 4),
                 "mamba_nvfp4_share_of_weights": round(2.47 / w, 4)}
    out["C2"] = {"short_ctx_tokens_B": 8 + 64, "total_tokens_B": 8 + 64 + 48,
                 "batch_tokens_8K": 2048 * 8192, "batch_tokens_128K": 128 * 131072,
                 "gpu_hours": 128 * 64, "tokens_per_gpu_s": round(120e9 / (128 * 64 * 3600)),
                 "jobs_hours": 16 * 4}
    cats = list(T5)
    means = [round(sum(T5[c][i] for c in cats) / len(cats), 4) for i in range(4)]
    out["C3"] = {"category_mean": means, "reported_overall": T5_OVERALL,
                 "overall_minus_mean": [round(T5_OVERALL[i] - means[i], 4) for i in range(4)],
                 "nvfp4_AL_change_t0": round(T5_OVERALL[2] / T5_OVERALL[0] - 1, 4),
                 "nvfp4_AL_change_t1": round(T5_OVERALL[3] / T5_OVERALL[1] - 1, 4),
                 "lilicorr_vs_dspark_AL": round(4.18 / 3.68 - 1, 4),
                 "best_worst_category_t0": (max(cats, key=lambda c: T5[c][0]), min(cats, key=lambda c: T5[c][0]))}
    rows = {}
    for b, v in T7.items():
        bf, q = v[1], v[2]
        rows[b] = {"bf16_to_2bit": round(q - bf, 2), "parent_to_2bit": round(q - v[0], 2)}
    wins = {}
    for j, name in [(3, "Qwen3.6"), (4, "Gemma4"), (5, "Lightning")]:
        w_, l_, t_ = [], [], []
        for b, v in T7.items():
            if v[j] is None:
                continue
            (w_ if v[2] > v[j] else l_ if v[2] < v[j] else t_).append(b)
        wins[name] = {"wins": len(w_), "losses": len(l_), "ties": len(t_), "lost_on": l_}
    common = [b for b, v in T7.items() if all(x is not None for x in v[:3])]
    mean = lambda j: round(sum(T7[b][j] for b in common) / len(common), 2)
    out["C4"] = {"per_benchmark": rows, "bolt24_vs_competitors": wins,
                 "mean_over_common_14_or_less": {"n": len(common), "parent": mean(0), "bolt_bf16": mean(1), "bolt_24gb": mean(2)},
                 "quantized_better_than_bf16": [b for b, v in T7.items() if v[2] > v[1]],
                 "pinchbench_rank_in_table7": sorted(((v, COLS[i]) for i, v in enumerate(T7["PinchBench"])), reverse=True),
                 "blog_claim": "Bolt 74B-A9B 24GB #1 open model in its class on PinchBench, #2 on BrowseComp (블로그). "
                               "Table 7 안에서는 PinchBench에서 Qwen3.6 35B-A3B BF16(88.07)이 Bolt 24GB(87.82)보다 높음"}
    bolt = T8["Bolt 24GB + LiLiCorr"]
    out["C5"] = {k: {"tps_ratio_bolt_over": round(bolt[0] / v[0], 3), "replay_ratio_bolt_over": round(bolt[1] / v[1], 3)}
                 for k, v in T8.items() if not k.startswith("Bolt")}
    (ROOT / "analysis/table_check_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in out.items():
        print(k, json.dumps(v, ensure_ascii=False)[:700])


if __name__ == "__main__":
    main()
