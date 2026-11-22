"""Genie (arXiv:2402.15391 v1) 논문 표 수치의 재계산과 내부 일관성 검사.

논문이 주는 것은 평균 성능 점 값뿐이다(시드 수, SD, 시도 수 없음). 따라서 검정은 하지 않고
차이, 비율, 단위 환산, 표와 본문의 산술 일치만 확인한다.
실행: python3 analysis/recompute.py  -> analysis/recompute_results.json (수 초, 표준 라이브러리만 사용)
"""
import json
import math
from pathlib import Path

OUT = Path(__file__).with_name("recompute_results.json")
res = {}

# 표 2: LAM 입력 ablation (Table 2, 본문 §3.4). (파라미터, FVD, ΔtPSNR)
T2 = {
    "platformers_token": (2.3e9, 38.8, 1.33),
    "platformers_pixel": (2.5e9, 40.1, 1.91),
    "robotics_token": (1.0e9, 257.8, 1.65),
    "robotics_pixel": (1.0e9, 136.4, 2.07),
}
res["V1_table2"] = {
    "platformers_dPSNR_pixel_minus_token": round(T2["platformers_pixel"][2] - T2["platformers_token"][2], 2),
    "robotics_dPSNR_pixel_minus_token": round(T2["robotics_pixel"][2] - T2["robotics_token"][2], 2),
    "platformers_dFVD_pixel_minus_token": round(T2["platformers_pixel"][1] - T2["platformers_token"][1], 1),
    "platformers_dFVD_rel_pct": round(100 * (T2["platformers_pixel"][1] / T2["platformers_token"][1] - 1), 1),
    "robotics_dFVD_pixel_minus_token": round(T2["robotics_pixel"][1] - T2["robotics_token"][1], 1),
    "robotics_dFVD_rel_pct": round(100 * (T2["robotics_pixel"][1] / T2["robotics_token"][1] - 1), 1),
    "platformers_param_mismatch_pct": round(100 * (T2["platformers_pixel"][0] / T2["platformers_token"][0] - 1), 1),
}

# 표 3: 토크나이저 구조 ablation (Table 3). (파라미터, 메모리 GB, FVD, ΔtPSNR)
T3 = {"ViT": (230e6, 0.3, 114.5, 1.39), "C-ViViT": (225e6, 1.6, 272.7, 1.37), "ST-ViViT": (205e6, 0.9, 81.4, 1.66)}
res["V2_table3"] = {
    "ST_vs_ViT_dFVD_rel_pct": round(100 * (T3["ST-ViViT"][2] / T3["ViT"][2] - 1), 1),
    "ST_vs_CViViT_dFVD_rel_pct": round(100 * (T3["ST-ViViT"][2] / T3["C-ViViT"][2] - 1), 1),
    "ST_vs_ViT_dPSNR": round(T3["ST-ViViT"][3] - T3["ViT"][3], 2),
    "ST_vs_CViViT_dPSNR": round(T3["ST-ViViT"][3] - T3["C-ViViT"][3], 2),
    "param_range_pct": round(100 * (max(v[0] for v in T3.values()) / min(v[0] for v in T3.values()) - 1), 1),
    "ST_vs_ViT_memory_ratio": round(T3["ST-ViViT"][1] / T3["ViT"][1], 2),
    "CViViT_vs_ST_memory_ratio": round(T3["C-ViViT"][1] / T3["ST-ViViT"][1], 2),
}

# 표 4: 데이터 큐레이션 (Table 4). FVD 61.4 -> 54.8
res["V3_table4"] = {
    "dFVD": round(54.8 - 61.4, 1),
    "dFVD_rel_pct": round(100 * (54.8 / 61.4 - 1), 1),
    "curated_fraction_of_original_pct": round(100 * 6.8 / 55, 1),
}

# 데이터 규모 (§3, App. B.1): 16 s 클립 10 FPS
clip_s = 16
res["V4_dataset_hours"] = {
    "original_55M_clips_hours": round(55e6 * clip_s / 3600, -2),
    "curated_6p8M_clips_hours": round(6.8e6 * clip_s / 3600, -2),
}

# 토큰 수 검사 (§3.1 배치 토큰, §3.2 942B 토큰). 프레임당 토큰 수는 본문에서 직접 주어지지 않아 역산한다.
seq_len = 16
batch_tokens = {128: 1.9e6, 256: 3.8e6, 448: 6.6e6}  # 본문 §3.1 값
tok_per_frame_implied = {b: round(t / (b * seq_len)) for b, t in batch_tokens.items()}
tpf = 920  # 160x90 영상, 패치 4 (App. C.2 Table 7)의 대략적 토큰 수. 역산값과 비교
res["V5_tokens"] = {
    "tokens_per_frame_implied_from_batch_tokens": tok_per_frame_implied,
    "batch128_tokens_check_M": round(128 * seq_len * tpf / 1e6, 2),
    "final_genie_total_tokens_B": round(512 * 125_000 * seq_len * tpf / 1e9, 1),
    "paper_final_tokens_B": 942,
    "scaling_200k_x256_tokens_B": round(256 * 200_000 * seq_len * tpf / 1e9, 1),
    "paper_scaling_tokens_B": 750,
}

# 계산량 검사 (Table 10: 750B 토큰, 200k 스텝). 6·N·D 근사와 표 값의 비율
T10 = [(41e6, 2.05e20), (96e6, 3.58e20), (192e6, 6.4e20), (404e6, 1.2e21), (811e6, 2.2e21), (1.6e9, 4.04e21), (2.7e9, 6.91e21)]
D10 = 750e9
ratios = [round(f / (6 * n * D10), 2) for n, f in T10]
res["V6_flops_vs_6ND"] = {
    "ratio_table_over_6ND_by_model": ratios,
    "min_ratio": min(ratios),
    "max_ratio": max(ratios),
    "final_model_6ND_e22": round(6 * 10.1e9 * 942e9 / 1e22, 2),
    "final_model_table12_e22": 6.6,
    "final_model_ratio": round(6.6e22 / (6 * 10.1e9 * 942e9), 2),
}

# 파라미터 합 (§3.2: 토크나이저 200M, LAM 300M, 동역학 10.1B, 합 10.7B)
res["V7_param_sum"] = {
    "components_sum_B": round(10.1 + 0.2 + 0.3, 2),
    "paper_total_B": 10.7,
    "abstract_B": 11,
}

# 배치 크기 스케일링 계산량 비율 (Table 6: 64 -> 384 배치)
res["V8_tokenizer_batch_scaling"] = {
    "flops_ratio": round(2.57e21 / 4.22e20, 2),
    "psnr_gain_db": round(36.5 - 35.7, 1),
}

# 유의성: 시드 수와 분산이 없어 검정 불가. 기록만 남긴다.
res["V9_significance"] = "not computable: no seeds, SD, CI or per-sample values for FVD and dPSNR in the paper text"

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
