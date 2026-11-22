"""SmolVLA 논문 표 값의 재계산과 일관성 검사 (numpy, scipy). 실행: python3 analysis/recompute.py -> analysis/recompute_results.json

검사: (1) 평균 열 재계산, (2) LIBERO 시행 수 가정(스위트당 100회, 평균은 400회)에서의 이항 신뢰구간과 차이 검정,
(3) 실로봇 시행 수 미보고 상황의 민감도, (4) 그림 5의 시간·고정시간 t검정, (5) 부록 A.1 데이터셋 목록 개수.
"""
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "paper.txt"
OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}


def wilson(p, n, z=1.96):
    denom = 1 + z * z / n
    c = (p + z * z / (2 * n)) / denom
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return c - h, c + h


def two_prop_z(p1, n1, p2, n2):
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


# (1) 평균 열 재계산 (표 2, 표 3, 표 5, 표 6–13). 값은 논문 표 그대로.
tables = {
    "T2_LIBERO_SmolVLA_0.45B": ([90, 96, 92, 71], 87.3),
    "T2_LIBERO_SmolVLA_0.24B": ([87, 93, 88, 63], 82.75),
    "T2_LIBERO_SmolVLA_2.25B": ([93, 94, 91, 77], 88.75),
    "T2_LIBERO_pi0_3.3B": ([90, 86, 95, 73], 86.0),
    "T2_LIBERO_pi0_Paligemma3B": ([87, 63, 89, 48], 71.8),
    "T2_LIBERO_OpenVLA": ([84.7, 88.4, 79.2, 53.7], 76.5),
    "T2_LIBERO_Octo": ([78.9, 85.7, 84.6, 51.1], 75.1),
    "T2_LIBERO_DP": ([78.3, 92.5, 68.3, 50.5], 72.4),
    "T2_MW_SmolVLA_0.24B": ([86.43, 46.36, 35, 60], 56.95),
    "T2_MW_SmolVLA_0.45B": ([82.5, 41.8, 45.0, 60.0], 57.3),
    "T2_MW_SmolVLA_2.25B": ([87.14, 51.82, 70, 64], 68.24),
    "T2_MW_pi0_3.5B_VLA": ([71.8, 48.2, 41.7, 30.0], 47.9),
    "T2_MW_pi0_3.5B_Paligemma": ([80.4, 40.9, 36.7, 44.0], 50.5),
    "T2_MW_TinyVLA": ([77.6, 21.5, 11.4, 15.8], 31.6),
    "T2_MW_DP": ([23.1, 10.7, 1.9, 6.1], 10.5),
    "T3_SO100_SmolVLA": ([75, 90, 70], 78.3),
    "T3_SO100_pi0": ([100, 40, 45], 61.7),
    "T3_SO100_ACT": ([70, 50, 25], 48.3),
    "T5_multi_pt": ([75, 90, 70], 78.3),
    "T5_multi_nopt": ([80, 40, 35], 51.7),
    "T5_single_nopt": ([55, 45, 20], 40.0),
    "T5_async_fig5": ([80, 90, 50], 73.3),
    "T6_CA": ([87, 92, 83, 54], 79.0),
    "T6_SA": ([80, 94, 84, 40], 74.5),
    "T6_CA+SA": ([86, 99, 90, 67], 85.5),
    "T8_N8": ([77, 88, 86, 49], 75.0),
    "T8_N16": ([88, 91, 91, 44], 78.5),
    "T8_N24": ([86, 97, 86, 49], 79.5),
    "T8_N32": ([89, 94, 85, 53], 80.3),
    "T8_skip2": ([84, 90, 83, 45], 75.5),
    "T8_VLM256M": ([86, 83, 75, 59], 75.8),
    "T9_x1.00": ([87, 96, 90, 56], 82.3),
    "T9_x0.75": ([82, 89, 84, 55], 77.5),
    "T9_x0.50": ([89, 94, 85, 53], 80.3),
    "T9_x0.25": ([76, 97, 83, 39], 73.8),
    "T10_FM": ([89, 94, 85, 53], 80.25),
    "T10_regression": ([92, 85, 86, 38], 75.25),
    "T11_prefix_CA": ([89, 94, 85, 53], 80.3),
    "T11_suffix_CA": ([86, 82, 78, 47], 73.3),
    "T11_prefix_SA": ([62, 74, 57, 20], 53.3),
    "T11_suffix_SA": ([80, 92, 80, 47], 74.8),
    "T12_chunk1": ([45, 77, 54, 24], 50.0),
    "T12_chunk10": ([90, 94, 94, 58], 84.0),
    "T12_chunk30": ([85, 94, 87, 48], 78.5),
    "T12_chunk50": ([89, 94, 85, 53], 80.3),
    "T12_chunk100": ([83, 88, 85, 42], 74.5),
    "T13_steps1": ([89, 94, 85, 53], 80.3),
    "T13_steps10": ([89, 94, 91, 57], 82.8),
    "T13_steps30": ([76, 91, 74, 42], 70.8),
    "T13_steps50": ([54, 70, 58, 25], 51.8),
}
mean_check = {}
for k, (vals, reported) in tables.items():
    m = float(np.mean(vals))
    mean_check[k] = {"computed": round(m, 4), "reported": reported, "diff": round(m - reported, 4),
                     "ok_within_0.05": abs(m - reported) <= 0.051}
res["V1_mean_check"] = mean_check
res["V1_mismatches"] = [k for k, v in mean_check.items() if not v["ok_within_0.05"]]

# 표 13 "50 steps" 행은 논문 표에서 54/70/58/25 로 읽었는데 평균 51.8과 맞는지 위에서 확인됨.
# 같은 기본 설정이 여러 표에서 다른 값으로 나오는지 확인 (V2)
default_rows = {
    "T6_CA+SA (ours 표기)": [86, 99, 90, 67],
    "T11_prefix_CA": [89, 94, 85, 53],
    "T10_FM": [89, 94, 85, 53],
    "T12_chunk50": [89, 94, 85, 53],
    "T13_steps1": [89, 94, 85, 53],
    "T9_x0.50": [89, 94, 85, 53],
    "T8_N32": [89, 94, 85, 53],
    "T9_x1.00": [87, 96, 90, 56],
    "T12_chunk10": [90, 94, 94, 58],
}
res["V2_same_setting_different_values"] = {k: {"row": v, "avg": round(float(np.mean(v)), 2)} for k, v in default_rows.items()}

# 표 9: 확장 폭 단조성과 본문 주장 "0.75 x d가 좋은 균형"
res["V3_expert_width"] = {
    "avg_by_width": {"1.00": 82.3, "0.75": 77.5, "0.50": 80.3, "0.25": 73.8},
    "note": "0.75가 0.50보다 2.8 p 낮다. 본문의 선택 근거와 표가 맞지 않는다.",
}

# (2) LIBERO 시행 수 가정: 스위트 100회 x 4 = 400 에피소드 평균
N = 400
pairs = {
    "T10 FM 80.25 vs regression 75.25": (80.25, 75.25),
    "T6 CA 79.0 vs SA 74.5": (79.0, 74.5),
    "T6 CA+SA 85.5 vs CA 79.0": (85.5, 79.0),
    "T2 SmolVLA 0.45B 87.3 vs pi0 3.3B 86.0": (87.3, 86.0),
    "T9 x0.75 77.5 vs x0.50 80.3": (77.5, 80.3),
    "T8 skip2 75.5 vs VLM-256M 75.8": (75.5, 75.8),
    "T13 steps1 80.3 vs steps10 82.8": (80.3, 82.8),
}
lib = {}
for k, (a, b) in pairs.items():
    z, p = two_prop_z(a / 100, N, b / 100, N)
    lib[k] = {"diff_pp": round(a - b, 2), "z": round(z, 2), "p_two_sided": round(p, 4)}
res["V4_libero_binomial_400"] = lib
res["V4_note"] = "이항 독립 가정(스위트·과제 군집 무시). 군집을 고려하면 유효 n은 더 작아 검정력이 낮다."
res["V4_wilson_CI_95_avg_400"] = {str(p): [round(100 * x, 2) for x in wilson(p / 100, N)] for p in [87.3, 80.25, 75.25, 85.5, 79.0]}
res["V4_wilson_CI_95_suite_100_SmolVLA045_Spatial90"] = [round(100 * x, 2) for x in wilson(0.90, 100)]

# (3) 실로봇: 시행 수 미보고. 과제당 n=10 가정 (그림 5 프로토콜과 같은 규모)과 사용자 제시 n=20 민감도
def real_world(p1, p2, n_each):
    z, p = two_prop_z(p1, n_each, p2, n_each)
    return {"n_each": n_each, "diff_pp": round(100 * (p1 - p2), 1), "z": round(z, 2), "p_two_sided": round(p, 4)}

res["V5_real_world_sensitivity"] = {
    "T3_avg_SmolVLA_vs_pi0_n30": real_world(0.783, 0.617, 30),
    "T3_avg_SmolVLA_vs_pi0_n60": real_world(0.783, 0.617, 60),
    "T3_avg_SmolVLA_vs_ACT_n30": real_world(0.783, 0.483, 30),
    "T5_pretrain_78.3_vs_51.7_n30": real_world(0.783, 0.517, 30),
    "T4_ID_SmolVLA_vs_ACT_n10": real_world(0.90, 0.70, 10),
    "T4_OOD_SmolVLA_vs_ACT_n10": real_world(0.50, 0.40, 10),
    "T3_avg_SmolVLA_vs_pi0_n30_note": "점수는 0.5 단위 부분 점수라 이항 가정은 근사다.",
}

# (4) 그림 5 시간·고정시간 (표 기술 통계, 시행 수는 본문에서 10과 5)
t_time = stats.ttest_ind_from_stats(13.75, 2.42, 10, 9.70, 2.95, 10, equal_var=False)
t_count = stats.ttest_ind_from_stats(1.8, 0.45, 5, 3.8, 1.3, 5, equal_var=False)
res["V6_fig5_time_welch"] = {"t": round(float(t_time.statistic), 2), "p": round(float(t_time.pvalue), 4),
                             "pct_faster": round(100 * (13.75 - 9.70) / 13.75, 1)}
res["V6_fig5_count_welch_n5"] = {"t": round(float(t_count.statistic), 2), "p": round(float(t_count.pvalue), 4),
                                 "sum_check_sync_9_async_19": [round(1.8 * 5, 1), round(3.8 * 5, 1)]}
sample_sd_sync = float(np.std([0, 2, 2, 2, 3], ddof=1))
res["V6_note"] = ("시간 표준편차 2.42/2.95와 평균 13.75/9.70은 n=10으로 Welch 검정이 가능하다. "
                  "고정시간 실험(n=5 가정)은 합계 9·19와 평균 1.8·3.8이 n=5와 맞는다. "
                  "sync SD 0.45는 n=5 표본 SD 값 가능 범위 안에 있다.")
res["V6_success_fig5_pickplace_sync_vs_async_n10"] = {
    "pick_place": real_world(0.80, 0.75, 10), "sorting": real_world(0.50, 0.70, 10)}

# (5) 부록 A.1 데이터셋 목록 개수 (논문 본문 481과 비교)
txt = PAPER.read_text(encoding="utf-8")
start = txt.find("List of datasets")
end = txt.find("\nReferences", start)
seg = txt[start + len("List of datasets."): end if end != -1 else None]
ids = []
for tok in re.split(r"[\s,]+", seg):
    t = tok.strip(".;:()")
    if re.fullmatch(r"[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-+]+", t):
        ids.append(t)
uniq = sorted(set(ids))
res["V7_dataset_list"] = {
    "unique_ids_in_appendix": len(uniq),
    "claimed_in_text_and_table1": 481,
    "difference": len(uniq) - 481,
    "so101_or_svla_ids": [x for x in uniq if re.search(r"so101|svla", x, re.I)],
    "so100_ids": sum(1 for x in uniq if re.search(r"so_?100", x, re.I)),
}
res["V7_table1"] = {"episodes": 22900, "frames": 10_600_000, "frames_per_episode": round(10_600_000 / 22_900, 1),
                    "note": "Table 1 캡션은 '~10M episodes'라고 쓰지만 표의 10.6M은 프레임 수다."}

# (6) 학습 설정 산술: 200k 스텝, 코사인 감쇠 30k 스텝 (코드 기본값), 4 GPU 기준 GPU 시간
res["V8_schedule"] = {"warmup_paper_steps": 100, "warmup_code_default_steps": 1000,
                      "decay_steps_code_default": 30000, "pretrain_steps_paper": 200000,
                      "fraction_at_floor_after_decay": round((200000 - 30000) / 200000, 3)}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print("mean mismatches:", res["V1_mismatches"])
print("dataset unique:", len(uniq))
print(json.dumps(res["V4_libero_binomial_400"], ensure_ascii=False, indent=1))
print(json.dumps(res["V5_real_world_sensitivity"], ensure_ascii=False, indent=1))
print(res["V6_fig5_time_welch"], res["V6_fig5_count_welch_n5"])
print(res["V6_success_fig5_pickplace_sync_vs_async_n10"])
print(res["V7_dataset_list"]["so101_or_svla_ids"], res["V7_dataset_list"]["so100_ids"])
