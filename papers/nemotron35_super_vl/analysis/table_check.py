"""기술 보고서의 표·본문 수치와 블로그 주장을 다시 계산한다 (CPU). 실행: python3 analysis/table_check.py

출처 표기: T = 보고서 표 번호, F = 그림 번호, § = 절. 블로그는 blog_announcement(blogs.nvidia.com),
blog_technical(developer.nvidia.com). 수치는 paper/paper.txt 와 블로그 HTML에서 옮겼다.
결과: analysis/table_check_results.json
"""
import json
import math
import os
from pathlib import Path
from statistics import median

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
r = lambda x, n=3: round(x, n)

# T10: 사후학습 모델 평가 (N3.5-Super, N3-Ultra, Qwen3.5-122B, Qwen3.8-Flash-Next, DS-V4-Flash, Inkling Small, GLM-5.3 Flash)
MODELS = ["N3.5-Super", "N3-Ultra", "Qwen3.5", "Qwen3.8", "DS-V4", "Inkling", "GLM-5.3"]
T10 = {
    "SWE-bench Verified": [71.10, 73.93, 71.87, 72.60, 77.80, 75.27, 79.10],
    "SWE-bench Multilingual": [67.72, 68.78, 62.00, 79.00, 72.00, 71.44, 78.30],
    "SWE-bench Pro": [51.78, 49.66, 45.60, 60.37, 56.91, 56.54, 59.70],
    "Tau2 Retail": [69.24, 73.68, 55.37, 71.60, 73.79, 71.82, 78.18],
    "Tau2 Airline": [76.50, 79.75, 63.50, 80.75, 83.50, 62.75, 81.75],
    "Tau2 Telecom": [73.19, 83.00, 87.17, 90.35, 98.25, 90.12, 98.46],
    "Tau3 Banking": [18.88, 18.35, 14.23, 44.54, 38.56, None, 34.85],
    "Terminal-Bench 2.1": [60.67, 54.78, 48.70, 85.80, 76.80, 53.40, 82.70],
    "GDPVal-AA-v2 norm ELO": [0.356, 0.324, 0.180, 0.531, 0.441, 0.407, 0.532],
    "BrowseComp": [78.91, 67.85, 58.10, 89.02, 84.20, 74.10, 86.57],
    "BrowseComp (OpenCode)": [72.35, 61.06, 45.42, 76.07, 64.77, 55.37, 81.91],
    "SciCode (subtask)": [47.13, 41.96, 39.83, 46.62, 49.30, 48.28, 49.96],
    "SciCode (problem)": [15.10, 13.46, 10.77, 14.42, 18.27, 18.65, 17.50],
    "LiveCodeBench v6": [88.22, 89.21, 76.65, 94.13, 94.64, 88.25, 80.62],
    "GPQAD": [89.52, 86.99, 86.74, 92.68, 89.65, 88.57, 91.22],
    "CritPt": [10.86, 3.14, 1.14, 15.70, 12.57, 9.14, 9.70],
    "HLE (no tools, text)": [31.79, 29.24, 24.24, 39.57, 35.82, 33.27, 36.47],
    "HLE (with tools, text)": [46.01, 37.40, 47.50, None, 45.10, 47.80, 55.30],
    "Apex-Shortlist": [88.36, 75.00, 54.65, 91.76, 88.70, 71.54, 77.13],
    "AA-Omniscience Accuracy": [26.20, 25.30, 26.25, 27.87, 44.95, 35.38, 34.22],
    "AA-Omniscience Non-Hallucination": [69.96, 77.91, 15.5, 58.69, 4.09, 40.39, 69.75],
    "BabyVision": [29.60, None, 35.65, 46.82, None, 24.74, 39.09],
    "CharXiv-RQ": [80.82, None, 83.47, 85.77, None, 77.77, 81.93],
    "MathVision": [78.16, None, 83.29, 87.54, None, 78.77, 82.52],
    "OCR-Reasoning": [65.08, None, 64.27, 67.10, None, 54.69, 64.61],
    "OCRBench-v2 (EN)": [68.12, None, 68.27, 67.91, None, 54.21, 61.81],
    "OCRBench-v2 (ZH)": [59.15, None, 67.26, 68.28, None, 52.38, 56.81],
    "Video-MME-v2": [32.85, None, 31.20, 45.05, None, None, 14.79],
    "V*": [85.60, None, 91.10, 93.02, None, 70.51, 88.13],
    "HLE (full)": [29.20, None, 24.44, 38.76, None, 31.60, 35.08],
    "MMLongBench-Doc": [62.82, None, 63.74, 67.93, None, 49.06, 65.83],
    "OSWorld": [52.14, None, 51.47, 55.76, None, None, 68.86],
    "AA-LCR": [68.63, 75.19, 75.13, 78.38, 79.69, 77.31, 79.50],
    "GraphWalks (<=1M)": [89.71, 63.28, 86.12, 76.15, 65.38, 55.74, 88.41],
    "IFBench": [70.86, 82.94, 67.56, 75.63, 76.19, 79.31, 57.67],
    "NV Chat Benchmark": [52.67, 34.27, 46.98, 35.16, 49.78, 53.60, 61.00],
    "MMLU-ProX": [76.91, 83.10, 83.19, 82.98, 85.53, 83.97, 82.99],
    "WMT24pp COMET": [84.80, 82.06, 86.34, 86.20, 85.50, 84.60, 86.13],
}

# T23: BF16 vs 공개 NVFP4 (토큰 BF16, 토큰 NVFP4, 정확도 BF16, 정확도 NVFP4)
T23 = {
    "SWE-Bench Verified": (22353, 21166, 71.10, 71.07), "SWE-Bench Multilingual": (29710, 30610, 67.72, 67.00),
    "SWE-Bench Pro": (25630, 25036, 51.78, 51.30), "Tau2 Telecom": (5784, 5659, 73.19, 73.19),
    "Tau2 Retail": (4028, 3983, 69.24, 69.30), "Tau2-AA (Airline)": (5426, 5224, 76.50, 75.75),
    "Tau3 Banking": (12173, 12065, 18.88, 17.78), "Terminal-Bench v2.1": (65560, 65717, 60.67, 62.78),
    "GDPVal": (49705, 52688, 49.98, 55.74), "SciCode (subtask)": (6621, 6796, 47.13, 46.61),
    "SciCode (problem)": (6621, 6796, 15.10, 14.13), "LiveCodeBench v6": (15449, 15733, 88.22, 87.41),
    "BIRD": (637, 573, 59.58, 60.60), "GPQA Diamond": (14383, 14738, 89.52, 89.90),
    "APEX Shortlist": (68050, 66926, 88.36, 89.36), "HLE (text)": (39253, 40170, 31.79, 31.44),
    "AA-Omniscience Accuracy": (4373, 4549, 26.20, 25.86), "AA-Omni Non-Hallucination": (4373, 4549, 69.96, 72.33),
    "CritPt": (7347, 7111, 10.86, 11.71), "BabyVision": (11839, 11634, 29.60, 29.51),
    "CharXiv (RQ)": (4157, 3930, 80.82, 79.88), "MathVision": (7434, 7585, 78.16, 77.79),
    "OCR Reasoning": (5362, 5300, 65.08, 65.09), "OCRBenchV2 EN": (1860, 1919, 68.12, 68.34),
    "OCRBenchV2 ZH": (2688, 2867, 59.15, 59.03), "ScreenSpot Pro": (236, 324, 43.53, 42.25),
    "Video-MME-v2": (4579, 5046, 32.85, 32.54), "V*": (855, 732, 85.60, 85.95),
    "HLE (full)": (36564, 37365, 29.20, 29.00), "MMLongBench-Doc": (3365, 3484, 62.82, 62.82),
    "AA-LCR": (4601, 4228, 68.62, 67.69), "GraphWalks": (10635, 10135, 89.71, 90.34),
    "IFBench": (10016, 9027, 70.86, 71.50), "NV Chat Benchmark": (3835, 3977, 52.67, 51.93),
    "MMLU-ProX": (3478, 3460, 76.91, 73.76), "WMT24++ (en→xx)": (1175, 1194, 52.32, 51.94),
    "WMT24++ COMET": (1175, 1194, 84.80, 84.76),
}

# T22: Lightning BF16 vs W4A16 (CritPt 제외 규칙은 표 주석)
T22 = {
    "SWE-Bench Verified": (24569, 23414, 52.20, 52.20), "SWE-Bench Multilingual": (31071, 33924, 37.13, 37.00),
    "Tau2 Telecom": (7076, 7075, 59.65, 60.96), "Tau3 Banking": (19107, 17998, 9.90, 9.48),
    "Terminal-Bench v2.1": (76219, 69523, 24.44, 22.05), "GDPval": (52043, 52563, 17.19, 19.09),
    "PinchBench": (7206, 7115, 83.86, 84.78), "BrowseComp": (142589, 138151, 39.50, 39.50),
    "SciCode (subtask)": (17190, 17428, 32.74, 31.53), "GPQA Diamond": (15552, 14394, 76.45, 75.38),
    "HLE": (30352, 27270, 10.80, 10.70), "AA-Omniscience Accuracy": (2505, 2588, 17.47, 16.35),
    "AA-Omni Non-Hallucination": (2505, 2588, 68.88, 69.20), "CritPt": (4379, 3622, 0.00, 0.00),
    "AA-LCR": (7679, 7388, 52.38, 50.00), "RULER-v1 (1M)": (22, 21, 91.52, 90.83),
    "IFBench (loose)": (13069, 11422, 72.00, 74.04), "LM Arena Proxy": (3401, 3496, 12.06, 11.10),
}


def recovery_stats(table, dedup_tokens=False):
    ratios = [b / a * 100 for (_, _, a, b) in table.values() if a > 0]
    if dedup_tokens:  # 같은 실행을 지표 두 개로 나눈 행(SciCode, AA-Omni, WMT)은 토큰을 한 번만 센다
        seen, tb, tn = set(), 0, 0
        for (x, y, _, _) in table.values():
            if (x, y) in seen:
                continue
            seen.add((x, y)); tb += x; tn += y
    else:
        tb = sum(v[0] for v in table.values()); tn = sum(v[1] for v in table.values())
    return {"rows": len(table), "rows_with_ratio": len(ratios), "median_recovery_pct": r(median(ratios), 2),
            "min_recovery_pct": r(min(ratios), 2), "verbosity_ratio_of_totals_pct": r((tn / tb - 1) * 100, 2)}


def wilson_like_ci_overlap(a, b):
    """Arena Elo ± 값을 95% 구간으로 보고 겹치는지만 본다."""
    (ma, ea), (mb, eb) = a, b
    return (ma - ea) <= (mb + eb) and (mb - eb) <= (ma + ea)


def main():
    out = {}
    # C1 초록·서론의 수치
    out["C1_abstract"] = {
        "swe_pro_super_vs_ultra": [51.78, 49.66, r(51.78 - 49.66, 2)],
        "active_ratio_ultra_over_super": r(55 / 12, 2), "claim": "approximately 4.6× fewer active parameters",
        "mtp_AL_gain_pct": {"greedy": r((4.718 / 4.287 - 1) * 100, 2), "sampling": r((4.612 / 3.787 - 1) * 100, 2),
                            "claim": "up to 21.8% under probabilistic draft sampling (T8)"},
        "nvfp4_vs_bf16_gb200_F10": r(4324 / 1778, 3), "nvfp4_vs_bf16_h100_F10": r(970 / 522, 3),
        "super_vs_qwen35_gb200_F11": r(4331.88 / 1854.35, 3), "super_vs_qwen38_gb200_F11": r(4331.88 / 2282.73, 3),
        "super_vs_qwen35_h100_F11": r(988.87 / 424.94, 3), "super_vs_qwen38_h100_F11": r(988.87 / 752.95, 3),
        "peak_4856_source": "T24 row: conc 64, MTP 5, ReplaySSM ON, 89.9 tok/s/user",
        "throughput_numbers_differ": {"abstract_T24": 4856, "F10_NVFP4_GB200": 4324, "F11_GB200": 4331.88,
                                      "note": "같은 워크로드라도 ReplaySSM·MTP 길이·동시성 설정이 달라 세 값이 다름"},
        "ultra_vs_super_params": r(120 / 550, 3), "blog_claim": "roughly one-fifth the parameter count",
    }
    # C2 Indic CPT (T1, T2, T3, §2.3)
    t2_before = [72.48, 51.83, 84.17, 90.73, 72.67]
    t2_after = [77.90, 56.81, 92.83, 96.77, 82.49]
    t1_priority_native_out = [138.9, 196.4, 149.3, 180.5, 183.5, 152.8, 171.3, 320.1, 262.6, 157.5, 192.4, 168.5]
    t1_priority_roman_out = [19.0, 17.2, 18.4, 19.1, 19.3, 17.6, 19.7, 19.0, 17.6, 20.0, 18.8, 18.9]
    t1_additional_out = [31.1, 18.7, 39.9, 33.1, 28.9, 31.3, 50.1, 22.8, 40.8, 30.5]
    out["C2_indic"] = {
        "T2_deltas_recomputed": [r(a - b, 2) for a, b in zip(t2_after, t2_before)],
        "T2_reported": [5.42, 4.98, 8.66, 6.04, 9.82],
        "T1_subtotals": {"priority_native": r(sum(t1_priority_native_out), 1), "reported": 2273.8,
                         "priority_roman": r(sum(t1_priority_roman_out), 1), "reported_roman": 224.6,
                         "additional": r(sum(t1_additional_out), 1), "reported_additional": 327.2,
                         "grand_total": r(sum(t1_priority_native_out) + sum(t1_priority_roman_out) + sum(t1_additional_out), 1),
                         "reported_total": 2825.6},
        "translation_cost_reduction_pct": r((1 - 0.335 / 0.835) * 100, 1), "tokens_per_gpuh_ratio": r(0.835 / 0.335, 2),
        "T3_monotonic_10_20_30": [63.1, 64.6, 65.6],
    }
    # C3 중간학습 효과 (T5)
    ver = {"Pi": (57.4, 67.8), "OpenCode": (54.4, 65.8), "Hermes": (59.8, 63.2), "Codex": (48.4, 61.8), "Claude Code": (53.8, 58.0)}
    tb = {"Pi": (30.3, 35.2), "OpenCode": (35.2, 38.2), "Hermes": (44.2, 53.9), "Codex": (27.0, 39.7), "Claude Code": (39.7, 43.5)}
    out["C3_midtraining_T5"] = {
        "swe_verified_mean_gain_5_harness": r(sum(b - a for a, b in ver.values()) / 5, 2), "claim_verified": 8.6,
        "terminal_bench_mean_gain_5_harness": r(sum(b - a for a, b in tb.values()) / 5, 2), "claim_tb": 6.8,
        "openhands_changes": {"Multilingual": r(65.6 - 65.9, 1), "Pro": r(51.7 - 52.1, 1), "Verified": r(71.9 - 71.3, 1),
                              "run_to_run": "약 1점 (§2.5)"},
        "ruler_1m": {"after_LC": 79.9, "after_midtraining": 33.5, "after_recovery": 81.0},
    }
    # C4 테스트 시점 확장 (§3.11, F9)
    out["C4_test_time_scaling"] = {"terminal_bench": r(71.9 - 62.7, 1), "deepswe": r(34.5 - 24.6, 1),
                                   "browsecomp_subset": r(79.8 - 71.5, 1), "reported": [9.2, 9.9, 8.3]}
    # C5 오염 제거 (§3.12) → 사후학습 표본 규모 역산, 블로그 "3.8 million samples"와 비교
    removed = 14793 + 5878
    out["C5_decontamination"] = {
        "removed_sum": removed, "reported": 20671,
        "implied_post_training_samples_M": {"at_0.085pct": r(removed / 0.00085 / 1e6, 1),
                                            "at_0.09pct": r(removed / 0.0009 / 1e6, 1),
                                            "at_0.095pct": r(removed / 0.00095 / 1e6, 1)},
        "implied_benchmark_questions": round(1808 / 0.06),
        "blog_claim": "post-trained with roughly 3.8 million samples (blog_announcement)",
    }
    # C6 T10: Super 3.5 대 Ultra 3, 7개 모델 중 순위 (AA-Omni 비환각만 높을수록 좋음, 나머지도 높을수록 좋음)
    wins = losses = ties = 0
    deltas = {}
    ranks = {}
    best_rows = []
    for k, v in T10.items():
        s, u = v[0], v[1]
        if u is not None:
            deltas[k] = r(s - u, 2)
            wins += s > u; losses += s < u; ties += s == u
        vals = [x for x in v if x is not None]
        ranks[k] = f"{1 + sum(x > s for x in vals)}/{len(vals)}"
        if max(vals) == s:
            best_rows.append(k)
    mean_rank = sum(int(x.split("/")[0]) for x in ranks.values()) / len(ranks)
    out["C6_T10"] = {
        "super_vs_ultra": {"wins": int(wins), "losses": int(losses), "ties": int(ties), "rows": len(deltas)},
        "super_minus_ultra": deltas,
        "super_rank_among_available": ranks,
        "rows_where_super_is_best": best_rows,
        "mean_rank": r(mean_rank, 2),
    }
    # C7 Text Arena (T11): 전체·스타일 보정, ±를 95% 구간으로 보고 겹침
    arena = {"N3.5-Super NVFP4": (1420, 12), "N3-Ultra NVFP4": (1426, 7), "Qwen3.5": (1416, 4), "DS-V4": (1436, 4),
             "Inkling": (1404, 5), "GLM-5.3": (1474, 5), "Gemma-4 26B": (1438, 7)}
    out["C7_T11_arena_overall_style"] = {k: wilson_like_ci_overlap(arena["N3.5-Super NVFP4"], v)
                                         for k, v in arena.items() if k != "N3.5-Super NVFP4"}
    # C8 양자화 복원율 (T23, T22)
    out["C8_quantization"] = {
        "T23_all_rows": recovery_stats(T23), "T23_tokens_dedup": recovery_stats(T23, True),
        "T23_reported": {"median": 99.62, "verbosity": 0.44, "benchmarks_claimed": 36},
        "same_release_other_numbers": {"T18_fp8_kv_35_benchmarks": 99.52, "T18_nvfp4_kv": 99.36, "conclusion_§6": 99.52,
                                       "abstract_§1_§4.1": 99.62, "note": "같은 공개 NVFP4 + FP8 KV 구성인데 본문에 두 값이 있음"},
        "T23_rows_below_99": sorted([(k, r(v[3] / v[2] * 100, 2)) for k, v in T23.items() if v[3] / v[2] < 0.99],
                                    key=lambda x: x[1]),
        "T22_lightning": recovery_stats(T22), "T22_tokens_dedup": recovery_stats(T22, True),
        "T22_reported": {"median": 99.24, "verbosity": -3.63},
        "T12_bpe_check": r(8 * 74.987 / 124.453, 3),
        "kv_nvfp4_payload_saving_pct": r((1 - 4.5 / 8) * 100, 2),
    }
    # C9 서빙 (T24, T25)
    t24 = [(4374, 4628, 80.2, 86.0), (4458, 4729, 81.6, 86.9), (4632, 4856, 83.0, 89.9), (4629, 4846, 81.3, 91.5),
           (4566, 4859, 38.1, 41.9)]
    out["C9_serving"] = {
        "T24_tps_gain_pct": [r((b / a - 1) * 100, 1) for a, b, _, _ in t24],
        "T24_user_gain_pct": [r((d / c - 1) * 100, 1) for _, _, c, d in t24],
        "T24_reported_range": "4.7–6.4% / 6.4–12.5%",
        "T25_tps_gain_pct": r((3276 / 2969 - 1) * 100, 1), "T25_user_gain_pct": r((155.9 / 96.6 - 1) * 100, 1),
        "T25_ttft_ratio": r(1.12 / 0.49, 2),
    }
    # C10 블로그 주장 대조
    out["C10_blog_claims"] = {
        "26T_tokens": {"blog": "about 26 trillion tokens across pretraining and continued pretraining",
                       "reconstruction_T": r(25 + 1 + 0.1 + 0.035 + 0.0654 + 0.009, 3),
                       "note": "Nemotron 3 Super 사전학습 25T(arXiv:2604.12374 초록) + 3.5 단계(증류 1T, Indic 100B, LC 35B, 중간학습 65.4B, 회복 9B)"},
        "3.8M_samples": {"blog": "post-trained with roughly 3.8 million samples",
                         "report": "오염 제거로 20,671개(약 0.09%)를 뺐다 → 사후학습 표본 약 23M (C5)", "status": "보고서와 맞지 않거나 정의가 다름"},
        "5x_throughput": {"blog": "up to 5x higher throughput ... compared to other open models in its size category",
                          "report": "F11: Qwen3.5 대비 2.34×, Qwen3.8 대비 1.90× (≥50 tok/s/user)", "status": "보고서에 없음"},
        "8x_users_85pct": {"blog": "Dynamo, 16 B200, 50 tok/s/user: 8x as many users, 85% lower TTFT",
                           "report": "§5.6/T25: VR200 11장, 분리 서빙 +10.3% tok/s/GPU, TTFT는 오히려 0.49→1.12 s",
                           "status": "보고서에 없음(다른 실험 조건)"},
        "hermes_langchain": {"blog": "Hermes Bench 63.44 vs Ultra 56.33; LangChain Evals 87.05 vs 84.3",
                             "report": "해당 벤치마크 없음(T5의 Hermes는 하네스 이름)", "status": "보고서에 없음"},
        "charxiv": {"blog": "80.8%, Nano Omni 63%, with Python 84%", "report_T10": 80.82,
                    "status": "80.8은 일치, 63%·84%는 보고서에 없음"},
        "lilicorr_2.3x": {"blog": "up to 2.3x faster token generation (DGX Spark)",
                          "report": "§4.8: drafter 가중치 양자화로 +11%, lm_head로 +5% (Spark). 2.3×는 없음", "status": "보고서에 없음"},
        "600_tps_baseten": {"blog": "over 600 TPS on AA agentic tasks (Baseten)", "status": "보고서에 없음"},
        "4x_task_completion": {"blog": "4x faster task completion (Fig. 3, 10,000 coding tasks, B200)",
                               "report_F7_gpu_hours": {"N3.5 Super NVFP4": (51.3, 8.3), "Qwen3.8 NVFP4": (60.4, 37.1),
                                                       "DS-V4 Flash MXFP4": (56.9, 40.4), "Inkling Small NVFP4": (56.5, 98.9)},
                               "gpu_hour_ratio_vs": {"Qwen3.8": r(37.1 / 8.3, 2), "DS-V4": r(40.4 / 8.3, 2), "Inkling": r(98.9 / 8.3, 2)},
                               "accuracy_gap_pp_vs": {"Qwen3.8": r(51.3 - 60.4, 1), "DS-V4": r(51.3 - 56.9, 1), "Inkling": r(51.3 - 56.5, 1)},
                               "status": "F7(2,193 과제, B200 4장)에서 GPU-h 4.5–4.9배 적음. 단 정확도는 5–9%p 낮음. 블로그의 10,000 과제는 보고서에 없음"},
        "one_fifth_params": {"blog": "comparable with Ultra at roughly one-fifth the parameter count",
                             "check": r(120 / 550, 3), "T10_super_vs_ultra": "C6 참고"},
    }
    # C11 하네스·프로토콜에 따른 SWE-bench Pro, Terminal-Bench 값 (F8은 쪽 이미지에서 읽음)
    f8_pro = {"Claude Code": 49.45, "Mini-SWE-agent": 47.40, "Codex": 47.26, "Hermes": 47.20, "Pi": 46.92, "OpenCode": 41.04}
    f8_tb = {"Codex": 61.80, "Hermes": 61.42, "Claude Code": 59.55, "OpenCode": 55.06, "Mini-SWE-agent": 54.68, "Pi": 48.31}
    out["C11_harness_protocol"] = {
        "swe_pro_T10_opencode": 51.78, "swe_pro_F8_opencode": f8_pro["OpenCode"],
        "gap_T10_minus_F8_opencode_pp": r(51.78 - f8_pro["OpenCode"], 2),
        "swe_pro_F8_best": max(f8_pro.values()), "ultra_T10": 49.66,
        "F8_best_minus_ultra_T10_pp": r(max(f8_pro.values()) - 49.66, 2),
        "swe_pro_F7_nvfp4": 51.3, "swe_pro_F8_mean_6": r(sum(f8_pro.values()) / 6, 2),
        "tb_T10_terminus2": 60.67, "tb_F8_best": max(f8_tb.values()), "tb_F9_pass1": 62.7, "tb_T23_bf16_nvfp4": [60.67, 62.78],
        "note": ("F8: 하네스 기본 프롬프트, Git 기록 제거 + 부정행위 금지 지시, 731과제 × 2회, OpenCode v1.17.8. "
                 "T10: NeMo Gym 레시피(고정 커밋 1e66890, 731과제 × 3회, OpenCode v1.17.11). 그 레시피도 apply_anti_cheating=True로 "
                 "HEAD 조상 밖의 refs·reflog를 지우고 프롬프트에 부정행위 금지 규칙이 있음(paper/gym_recipe/). 10.7%p 차이는 보고서가 설명하지 않음."),
    }
    path = ROOT / "analysis" / "table_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in out.items():
        print(k, json.dumps(v, ensure_ascii=False)[:500])
    print("wrote", path)


if __name__ == "__main__":
    main()
