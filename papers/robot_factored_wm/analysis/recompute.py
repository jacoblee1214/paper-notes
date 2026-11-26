"""논문 표 수치의 재계산과 일관성 검사 (arXiv:2607.22535 v2 / v1).

실행: python3 analysis/recompute.py   → analysis/recompute_results.json
의존: numpy, scipy. 1초 미만.

점검 항목(V1–V10)은 analysis/03_analysis.md 와 번호가 같다.
표 번호는 v2 기준이다(v1의 표 2는 v2의 표 3). 숫자는 논문 표에서 옮겼다.
시행 수를 논문이 밝히지 않은 곳은 값 옆에 '가정'이라고 적었다.
"""
import json
from itertools import product
from math import comb, sqrt
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
R = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def mcnemar_exact(b, c):
    n = b + c
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


# ---------------------------------------------------------------- 표 값(v2)
T1 = {  # (PSNR, SSIM, LPIPS) DROID / RoboCasa-GR1
    "svd_ctrlworld": {"droid": (23.15, 0.884, 0.104)},
    "svd_ours": {"droid": (25.05, 0.899, 0.091)},
    "wan_adaln": {"droid": (18.57, 0.824, 0.224), "gr1": (17.67, 0.835, 0.194)},
    "wan_ours": {"droid": (21.87, 0.859, 0.178), "gr1": (24.61, 0.889, 0.131)},
}
T3_V2 = {"raw": (21.34, 0.848, 0.187), "nominal": (21.76, 0.858, 0.179), "depth": (21.96, 0.856, 0.179)}
T2_V1 = {"raw": (21.57, 0.860, 0.175), "nominal": (22.44, 0.872, 0.164), "depth": (23.08, 0.874, 0.161)}
T4 = {  # v1 표 4와 v2 표 4는 같다
    "droid_nom_nom": (21.74, 0.856, 0.179), "droid_log_log": (22.37, 0.861, 0.174), "droid_log_nom": (21.12, 0.846, 0.187),
    "gr1_nom_nom": (25.70, 0.904, 0.120), "gr1_log_log": (28.26, 0.918, 0.112), "gr1_log_nom": (24.69, 0.897, 0.126),
}
T7 = {  # 전체 프레임 / 로봇 제외
    "droid_adaln": ((18.57, 0.824, 0.224), (20.87, 0.872, 0.176)),
    "droid_ours": ((21.87, 0.859, 0.178), (23.71, 0.893, 0.150)),
    "gr1_adaln": ((18.05, 0.844, 0.184), (20.85, 0.893, 0.141)),
    "gr1_ours": ((24.89, 0.892, 0.129), (26.42, 0.914, 0.117)),
}
T5 = {
    "ur5_adaln_zs": (22.96, 0.946, 0.176), "ur5_ours_zs": (26.94, 0.962, 0.141),
    "arx_adaln_zs": (17.33, 0.930, 0.200), "arx_ours_zs": (22.77, 0.965, 0.136),
    "hrd_ours_zs": (21.00, 0.857, 0.201), "hrd_adaln_ft": (19.29, 0.837, 0.221), "hrd_ours_ft": (23.01, 0.885, 0.164),
}


def diff(a, b):
    return [round(a[0] - b[0], 3), round(a[1] - b[1], 3), round(a[2] - b[2], 3)]


# ---------------------------------------------------------------- V1: 표 1 차이
R["V1_table1_deltas"] = {
    "svd_droid_ours_minus_ctrlworld": diff(T1["svd_ours"]["droid"], T1["svd_ctrlworld"]["droid"]),
    "wan_droid_ours_minus_adaln": diff(T1["wan_ours"]["droid"], T1["wan_adaln"]["droid"]),
    "wan_gr1_ours_minus_adaln": diff(T1["wan_ours"]["gr1"], T1["wan_adaln"]["gr1"]),
    "lpips_rel_reduction_pct": {
        "svd_droid": round(100 * (1 - 0.091 / 0.104), 1),
        "wan_droid": round(100 * (1 - 0.178 / 0.224), 1),
        "wan_gr1": round(100 * (1 - 0.131 / 0.194), 1),
    },
    "note": "SVD 행(192x320)과 Wan 행(480x832)은 해상도가 달라 절대값 비교 불가(표 1 캡션). 차이는 같은 백본 안에서만 본다.",
}

# ---------------------------------------------------------------- V2: 표 2 정수 가능성, 신뢰구간, 검정
def feasible(pct, n):
    return [k for k in range(n + 1) if round(100 * k / n, 1) == pct]

v2 = {"robocasa_n160": {}, "robolab_n100": {}, "robocasa_full_n256": {}}
for name, pct in (("adaln", 70.6), ("ours", 75.0)):
    ks = feasible(pct, 160)
    v2["robocasa_n160"][name] = {"reported_pct": pct, "integer_k": ks, "wilson95": wilson(ks[0], 160)}
v2["robocasa_n160"]["fisher_unpaired_p"] = round(stats.fisher_exact([[120, 40], [113, 47]])[1], 4)
# 같은 160개 클립에 두 모델을 돌렸지만 클립별 정오 표가 없다. 차이 7개(120 - 113)를 만드는 맞짝 불일치 (b, c)에 대한 McNemar 정확검정
mc = {}
for n_disc in range(7, 61, 2):
    c_ = (n_disc - 7) // 2          # AdaLN만 맞춘 클립 수
    b_ = c_ + 7                      # Ours만 맞춘 클립 수
    mc[n_disc] = round(mcnemar_exact(b_, c_), 4)
v2["robocasa_n160"]["mcnemar_p_by_discordant_count(ours_only - adaln_only = 7)"] = mc
v2["robocasa_n160"]["max_discordant_for_p_lt_0.05"] = max([n for n, p in mc.items() if p < 0.05], default=None)

# RoboLab: 혼동 행렬은 부록 E에 있다 (TP, TN, FP, FN)
cm = {"adaln": (0, 49, 1, 50), "ours": (18, 48, 2, 32)}
for name, (tp, tn, fp, fn) in cm.items():
    n = tp + tn + fp + fn
    acc = (tp + tn) / n
    v2["robolab_n100"][name] = {
        "confusion_TP_TN_FP_FN": [tp, tn, fp, fn], "n": n, "accuracy_pct": round(100 * acc, 1),
        "wilson95": wilson(tp + tn, n),
        "recall_success_pct": round(100 * tp / (tp + fn), 1),
        "specificity_pct": round(100 * tn / (tn + fp), 1),
        "balanced_accuracy_pct": round(50 * (tp / (tp + fn) + tn / (tn + fp)), 1),
        "predicted_success_rate_pct": round(100 * (tp + fp) / n, 1),
    }
v2["robolab_n100"]["fisher_unpaired_p"] = round(stats.fisher_exact([[66, 34], [49, 51]])[1], 4)
# 맞짝: 양성 50개에서 AdaLN은 0개 맞힘(TP=0) → 같은 클립에서 Ours가 18개 추가. 음성 50개에서 맞힌 수 49 대 48 → 둘 다 맞힌 수 k는 47 또는 48
pairs = []
for k in (47, 48):
    b_adaln_only = 49 - k
    c_ours_only_neg = 48 - k
    ours_only = 18 + c_ours_only_neg
    pairs.append({"both_correct_on_negatives": k, "adaln_only": b_adaln_only, "ours_only": ours_only,
                  "mcnemar_exact_p": float(f"{mcnemar_exact(b_adaln_only, ours_only):.3g}")})
v2["robolab_n100"]["paired_mcnemar_bounds"] = pairs
v2["robolab_n100"]["note"] = "AdaLN은 100개 중 1개만 '성공'이라고 답했다. 정확도 49.0%는 '항상 실패' 응답(50.0%)보다 낮다."

for name, pct in (("adaln", 64.5), ("ours", 66.4), ("gt_video_judge", 81.6)):
    ks = feasible(pct, 256)
    v2["robocasa_full_n256"][name] = {"reported_pct": pct, "integer_k": ks, "wilson95": wilson(ks[0], 256) if ks else None}
v2["robocasa_full_n256"]["fisher_unpaired_p"] = round(stats.fisher_exact([[170, 86], [165, 91]])[1], 4)
v2["robocasa_full_n256"]["note"] = "209/256 = 81.64%. 선택 부분집합(160개)은 판정자가 정답 영상에서 결과를 맞힌 209개 안에서 고른 것이다."
v2["selection_effect_pct_points"] = {"gap_selected_160": round(75.0 - 70.6, 1), "gap_full_256": round(66.4 - 64.5, 1)}
R["V2_table2_task_outcome"] = v2

# ---------------------------------------------------------------- V3: 표 7 대 표 1
v3 = {"droid_full_equals_table1": T7["droid_adaln"][0] == T1["wan_adaln"]["droid"] and T7["droid_ours"][0] == T1["wan_ours"]["droid"]}
for ds in ("droid", "gr1"):
    full_gain = diff(T7[f"{ds}_ours"][0], T7[f"{ds}_adaln"][0])
    excl_gain = diff(T7[f"{ds}_ours"][1], T7[f"{ds}_adaln"][1])
    v3[ds] = {
        "full_gain_psnr_ssim_lpips": full_gain, "robot_excluded_gain": excl_gain,
        "psnr_gain_retained_pct": round(100 * excl_gain[0] / full_gain[0], 1),
        "lpips_gain_retained_pct": round(100 * excl_gain[2] / full_gain[2], 1),
    }
v3["gr1_subset82_vs_table1_n128"] = {
    "adaln_psnr": [T7["gr1_adaln"][0][0], T1["wan_adaln"]["gr1"][0]],
    "ours_psnr": [T7["gr1_ours"][0][0], T1["wan_ours"]["gr1"][0]],
    "note": "n = 82 부분집합(로봇 마스크 있는 클립)과 n = 128 전체의 값이 다르다. 부분집합 선택 규칙은 마스크 유무뿐이라고 부록 C가 쓴다.",
}
R["V3_table7_vs_table1"] = v3

# ---------------------------------------------------------------- V4: 표 3 대 표 4, v1 대 v2 절제
v4 = {
    "v2_nominal_minus_raw": diff(T3_V2["nominal"], T3_V2["raw"]),
    "v2_depth_minus_nominal": diff(T3_V2["depth"], T3_V2["nominal"]),
    "v1_nominal_minus_raw": diff(T2_V1["nominal"], T2_V1["raw"]),
    "v1_depth_minus_nominal": diff(T2_V1["depth"], T2_V1["nominal"]),
    "v2_minus_v1_same_row": {k: diff(T3_V2[k], T2_V1[k]) for k in T3_V2},
    "table3_nominal_row_vs_table4_droid_nom_nom": {
        "v2": diff(T3_V2["nominal"], T4["droid_nom_nom"]),
        "v1": diff(T2_V1["nominal"], T4["droid_nom_nom"]),
        "note": "둘 다 DROID에서 학습, 메시 RGB만, nominal/nominal이다. v1에서는 PSNR 0.70 dB 차이, v2에서는 0.02 dB 차이. 부록 B는 표 4의 DROID nominal 모델만 15,010 스텝이고 나머지는 10,000 스텝이라고 쓴다.",
    },
    "depth_row_vs_table1_wan_droid_ours": diff(T3_V2["depth"], T1["wan_ours"]["droid"]),
    "gr1_table4_mesh_only_vs_table1_joint_depth": {
        "table4_nom_nom_psnr": T4["gr1_nom_nom"][0], "table1_ours_psnr": T1["wan_ours"]["gr1"][0],
        "diff": round(T4["gr1_nom_nom"][0] - T1["wan_ours"]["gr1"][0], 2),
        "note": "표 4는 데이터셋별 학습, 메시만. 표 1은 DROID+RoboCasa 합동 학습, 메시+깊이. 학습 데이터, 깊이, 스텝 수가 함께 다르다.",
    },
}
# 효과 크기가 2 SE에 들어오려면 필요한 클립별 쌍 차이의 표준편차 상한 (n = 256 가정, 논문은 클립별 SD를 주지 않음)
n_clips = 256
v4["sensitivity_max_sd_of_paired_diff_for_2se"] = {
    f"delta_psnr_{d}": round(d * sqrt(n_clips) / 2, 2) for d in (0.20, 0.42)
}
v4["sensitivity_note"] = "클립 256개 가정. 쌍 차이의 SD가 이 값 이하면 해당 PSNR 차이가 약 2 SE 이상이다. 논문은 클립별 SD, 시드 수, 신뢰구간을 주지 않는다."
R["V4_ablation_consistency"] = v4

# ---------------------------------------------------------------- V5: 표 4 oracle 격차
v5 = {}
for ds in ("droid", "gr1"):
    nn, ll, ln = T4[f"{ds}_nom_nom"], T4[f"{ds}_log_log"], T4[f"{ds}_log_nom"]
    v5[ds] = {
        "oracle_logged_logged_minus_nominal_nominal": diff(ll, nn),
        "nominal_nominal_minus_logged_nominal": diff(nn, ln),
        "oracle_minus_logged_nominal": diff(ll, ln),
        "share_of_oracle_gap_recovered_by_training_on_nominal_psnr_pct": round(100 * (nn[0] - ln[0]) / (ll[0] - ln[0]), 1),
    }
R["V5_table4_oracle_gap"] = v5

# ---------------------------------------------------------------- V6: 표 5
v6 = {
    "ur5_ours_minus_adaln": diff(T5["ur5_ours_zs"], T5["ur5_adaln_zs"]),
    "arx_ours_minus_adaln": diff(T5["arx_ours_zs"], T5["arx_adaln_zs"]),
    "hrd_ours_ft_minus_adaln_ft": diff(T5["hrd_ours_ft"], T5["hrd_adaln_ft"]),
    "hrd_ours_zs_minus_adaln_ft": diff(T5["hrd_ours_zs"], T5["hrd_adaln_ft"]),
    "hrd_ours_ft_minus_ours_zs": diff(T5["hrd_ours_ft"], T5["hrd_ours_zs"]),
    "robotwin_clips_8x5": 8 * 5,
    "hrdexdb_train_clips_192ep_x3_offsets": 192 * 3,
    "hrdexdb_train_clips_reported": 575,
    "note": "HRDexDB 학습 클립 575 = 192 에피소드 x 오프셋 3개(0, -2, -4초)의 최대 576에서 1개 모자란다(부록 D). 클립 하나가 빠진 이유는 서술에 없다.",
}
R["V6_table5_transfer"] = v6

# ---------------------------------------------------------------- V7: 데이터 규모와 학습량(가정 명시)
droid_clips, gr1_clips = 41642, 9380
joint = droid_clips + gr1_clips
v7 = {
    "gr1_episodes_24tasks_x50": 24 * 50,
    "gr1_clips_per_episode_train_only": round(gr1_clips / (24 * 50), 2),
    "gr1_clips_incl_eval_128_and_256": gr1_clips + 128 + 256,
    "joint_train_clips": joint,
    "assumed_gpus_for_epoch_estimate": 4,
    "assumption_note": "논문 표 6은 GPU당 배치 1, 누적 4만 쓰고 GPU 수는 쓰지 않는다. 저장소 README 예시가 4 GPU라서 4로 가정했다(유효 배치 16).",
    "effective_batch": 16,
    "joint_epoch_steps_at_batch16": round(joint / 16),
    "epochs_at_10000_steps": round(10000 * 16 / joint, 2),
    "epochs_at_8000_steps_released": round(8000 * 16 / joint, 2),
    "droid_only_epochs_at_8000_steps": round(8000 * 16 / droid_clips, 2),
}
R["V7_scale_and_epochs"] = v7

# ---------------------------------------------------------------- V8: 채널 수, 입력 차원
R["V8_dimensions"] = {
    "wan_base_in_channels": 16 + 4 + 16, "claimed_base_36": 36,
    "mesh_only_total_in": 36 + 16, "mesh_depth_total_in": 36 + 48, "streams_in_depth_variant": 3, "channels_per_stream": 16,
    "adaln_input_union": 7 + 39, "adaln_stacked_184": (7 + 39) * 4,
    "gr1_action_29": 7 + 7 + 6 + 6 + 3, "gr1_state_39": 7 + 7 + 11 + 11 + 3,
    "hrdexdb_adaln_12d_x4": 12 * 4, "hrdexdb_adaln_note": "EEF 이동 3 + 회전 벡터 3 + 손 관절 6 = 12, 네 프레임 델타 = 48",
}

# ---------------------------------------------------------------- V9: 추론 비용
R["V9_inference_speedup"] = {
    "stated_reduction": 10, "from_50_steps_to_4": round(50 / 4, 2), "from_40_steps_to_4": 10.0,
    "nfe_50steps_cfg_vs_4steps_no_cfg": round(100 / 4, 1),
    "note": "논문 부록 B는 '디노이징 스텝을 10x 줄인다'고 쓴다. 저장소 README의 기본 추론은 50 스텝(CFG 6), 증류본은 4 스텝(CFG 1)이라서 50 대 4면 12.5x, CFG 두 번 호출을 센 NFE로는 25x이다. 기준 스텝 수가 40이면 10x이다.",
}

# ---------------------------------------------------------------- V10: 메모리
bytes_bf16 = 2
params_wan = 14e9
R["V10_memory_for_mac"] = {
    "wan14b_weights_bf16_gib": round(params_wan * bytes_bf16 / 2 ** 30, 1),
    "mac_unified_memory_gib": 24,
    "fits_weights_alone": params_wan * bytes_bf16 / 2 ** 30 < 24,
    "svd_1p5b_fp16_gib": round(1.5e9 * 2 / 2 ** 30, 1),
    "note": "Wan 2.1 14B 가중치만 bf16으로 약 26 GiB. 24 GiB 통합 메모리에 올릴 수 없다(양자화 없이). 파라미터 수 14e9는 모델 이름 기준 근사.",
}

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1))
print("wrote", OUT)
for k, v in R.items():
    print(k)
