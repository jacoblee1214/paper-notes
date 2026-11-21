"""FAST (arXiv:2501.09747 v1) 본문·표 수치 재계산과 일관성 검사.

입력: 논문 본문과 표에 적힌 값만 사용한다(그림의 막대 값은 텍스트에 없어서 쓰지 않는다).
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy; 수 초)
"""
import json
import math
from pathlib import Path

from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}

# ---- V1: Table I (평균 토큰 수, 1초 청크) ----
# (데이터셋, 행동 차원 D, 제어 주파수 Hz, 평균 naive 토큰, 평균 FAST 토큰, 논문이 적은 압축률, 팔 수)
table1 = [
    ("BridgeV2", 7, 5, 35, 20, 1.75, 1),
    ("DROID", 7, 15, 105, 29, 3.6, 1),
    ("Bussing", 7, 20, 140, 28, 5.0, 1),
    ("Shirt Fold", 14, 50, 700, 53, 13.2, 2),
]
rows = []
for name, D, hz, naive_paper, fast, ratio_paper, arms in table1:
    naive_formula = D * hz * 1  # 1초 청크에서 naive는 차원 × 샘플 수
    rows.append({
        "dataset": name,
        "naive_formula_D_x_Hz": naive_formula,
        "naive_paper": naive_paper,
        "naive_match": naive_formula == naive_paper,
        "fast_paper": fast,
        "ratio_recomputed": round(naive_paper / fast, 3),
        "ratio_paper": ratio_paper,
        "ratio_match": abs(naive_paper / fast - ratio_paper) < 0.05,
        "fast_per_arm": round(fast / arms, 1),
    })
res["V1_table1"] = rows
per_arm = [r["fast_per_arm"] for r in rows]
res["V1_fast_per_arm_range"] = [min(per_arm), max(per_arm)]
res["V1_bimanual_fast_tokens_paper_text_claim"] = 60  # 본문: "i.e., 60 tokens for the bi-manual setup"
res["V1_bimanual_fast_tokens_table"] = 53

# ---- V2: Table II (DROID 평가 시행 수) ----
trials = [4, 4, 2, 2, 2, 2, 4, 2, 4, 2, 4, 2, 2, 2, 2, 2, 2]
res["V2_droid_tasks_in_table"] = len(trials)
res["V2_droid_tasks_in_text"] = 16
res["V2_droid_trials_sum"] = sum(trials)
res["V2_droid_trials_in_table_total"] = 44

# ---- V3: 학습 샘플 수 일관성 (부록 C, D, E) ----
# DROID: 75k 에피소드, 21M 샘플, 240k 반복, batch 256 -> 샘플 본 횟수와 에폭
droid_seen = 240_000 * 256
res["V3_droid_samples_seen"] = droid_seen
res["V3_droid_epochs_from_samples"] = round(droid_seen / 21e6, 2)
res["V3_droid_text_says"] = "3 epochs (부록 D), 본문 부록 E는 '≈3 episodes'로 적힘"
# LIBERO: 270k 샘플, 40k 반복, '≈40 epochs' -> 암묵적 batch 크기
libero_implied_batch = 270_000 * 40 / 40_000
res["V3_libero_implied_batch_from_40_epochs"] = libero_implied_batch
# 1M 청크 × 1초 = 전체 시간이 아니라 샘플 수
res["V3_universal_chunks_hours_if_1s_each"] = round(1_000_000 * 1 / 3600, 1)

# ---- V4: 비율 주장 ----
res["V4_inference_latency_ratio_fast_over_diffusion"] = round(750 / 100, 2)
res["V4_fast_ms_per_token_range"] = [round(750 / 60, 1), round(750 / 30, 1)]
res["V4_diffusion_ms_per_step_if_10_steps"] = 10.0

# ---- V5: 작은 표본의 검출력 (DROID: 정책당 44 시행, 두 정책 비교) ----
# 독립 두 비율, 정책당 n = 44, p = 0.5 가정(최악 분산). 차이의 95% 신뢰구간 반폭과 80% 검정력 최소 검출 차이
n = 44
se_diff = math.sqrt(2 * 0.25 / n)
res["V5_droid_trials_per_policy"] = n
res["V5_droid_diff_ci_halfwidth_pts"] = round(100 * stats.norm.ppf(0.975) * se_diff, 1)
res["V5_droid_min_detectable_diff_pts_80power"] = round(100 * (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * se_diff, 1)

# ---- V6: 'up to 5x' 와 본문 '5x fewer GPU hours' 는 같은 값인가 ----
res["V6_training_speedup_abstract"] = "up to 5x"
res["V6_training_speedup_body"] = "5x fewer GPU hours (single run), 3x fewer steps on bussing (Fig. 9)"

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
for k, v in res.items():
    print(k, v)
