"""KPGrasp 논문(arXiv:2606.09314 v2)의 표 수치를 다시 계산하고 일관성을 검사한다.

실행: python3 analysis/recompute.py  → analysis/recompute_results.json (수 초, numpy·scipy만 사용)
입력은 논문 본문과 부록의 표 값을 직접 옮긴 것이다(paper/paper.txt로 대조).
GPU, 데이터, 가중치는 쓰지 않는다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


# ---------------------------------------------------------------- V1: 표 1 (Dexonomy)
t1 = {
    "UniDexGrasp": dict(GSR=24.5, OSR=73.2, CDC=15.6, PD=11.6, Div=28.0),
    "DexGraspAnything": dict(GSR=21.8, OSR=72.2, CDC=11.0, PD=14.2, Div=23.3),
    "Dexonomy_wo_classifier": dict(GSR=28.9, OSR=78.1, CDC=14.8, PD=11.2, Div=24.6),
    "Dexonomy_w_classifier": dict(GSR=63.9, OSR=91.3, CDC=13.9, PD=8.6, Div=25.7),
    "KPGrasp": dict(GSR=76.3, OSR=97.3, CDC=5.6, PD=2.4, Div=23.3),
}
kp = t1["KPGrasp"]
best_direct = t1["Dexonomy_wo_classifier"]["GSR"]
cls = t1["Dexonomy_w_classifier"]["GSR"]
res["V1_table1"] = {
    "gsr_gain_pp_vs_wo_classifier": round(kp["GSR"] - best_direct, 1),
    "gsr_gain_relative_pct_vs_wo_classifier": round(100 * (kp["GSR"] - best_direct) / best_direct, 1),
    "gsr_gain_pp_vs_w_classifier": round(kp["GSR"] - cls, 1),
    "gsr_gain_relative_pct_vs_w_classifier": round(100 * (kp["GSR"] - cls) / cls, 1),
    "note": "초록은 '47.4%'라고 쓰지만 47.4는 퍼센트 포인트(pp)이고 상대 증가는 약 164%다.",
}

# 테스트 격자: IK 절의 2,139 객체 × 6 스케일 × 10 후보 (부록 A)
n_grasp = 2139 * 6 * 10
res["V2_test_grid"] = {"objects": 2139, "scales": 6, "proposals": 10, "grasps": n_grasp,
                       "paper_IK_attempts": 128340, "match": n_grasp == 128340}
# GSR 신뢰구간: 분모를 128,340으로 가정(논문은 GSR의 분모를 명시하지 않음). 객체 단위 군집은 반영하지 못함.
res["V3_gsr_wilson_assumed_n"] = {
    "gsr_pct": 76.3,
    "wilson95_pct": wilson(round(0.763 * n_grasp), n_grasp),
    "assumption": "분모 = 128,340 (IK 시도 수와 같은 격자로 가정). 학습 시드 분산은 포함하지 않음.",
}
# 표 3 KPGrasp 행이 표 1과 같은지
t3_keypoint = dict(GSR=76.3, OSR=97.3, CDC=5.6, PD=2.4, Div=23.3)
res["V4_table3_keypoint_row_equals_table1"] = t3_keypoint == kp

# ---------------------------------------------------------------- V5: 표 2 (DGA benchmark)
cols = ["GRAB", "RealD", "MD", "UDG", "DGN"]
t2_pd = {
    "UniDexGrasp": ([37.4, 39.0, 13.5, 24.5, 31.9], 29.3),
    "GraspTTA": ([51.4, 40.1, 19.0, 21.2, 24.5], 31.2),
    "SceneDiffuser": ([41.1, 42.0, 14.6, 25.1, 31.0], 30.8),
    "UGG": ([33.2, 34.4, 10.3, 24.5, 25.2], 25.5),
    "DexGraspAnything": ([30.4, 27.7, 11.4, 18.8, 17.8], 21.2),
    "KPGrasp": ([20.8, 9.6, 6.6, 13.0, 14.3], 12.9),
}
t2_sr = {
    "UniDexGrasp": ([20.8, 27.1, 21.6, 23.7, 33.9], 25.4),
    "GraspTTA": ([14.4, 13.3, 30.3, 21.0, 18.6], 19.5),
    "SceneDiffuser": ([39.1, 21.7, 69.8, 28.3, 26.6], 37.1),
    "UGG": ([42.7, 32.7, 55.3, 46.0, 46.9], 44.7),
    "DexGraspAnything": ([57.9, 44.8, 79.1, 53.1, 57.5], 58.5),
    "KPGrasp": ([59.8, 67.1, 62.2, 62.3, 67.5], 63.8),
}
mismatch = []
for label, table in (("PD", t2_pd), ("SR", t2_sr)):
    for method, (vals, rep) in table.items():
        m = float(np.mean(vals))
        if abs(m - rep) > 0.05 + 1e-9:
            mismatch.append({"metric": label, "method": method, "mean_of_5": round(m, 2), "reported_avg": rep})
res["V5_table2_average_check"] = {"rows_checked": 12, "mismatches_gt_0.05": mismatch}

# 부록 표 6: 각 데이터셋에서 학습한 KPGrasp의 평균 (보고값 47.8)
t6_each = [44.7, 45.6, 54.7, 33.1, 61.8]
t6_orig = [62.8, 32.7, 40.6, 22.8, 55.2]
t6_dexo = [59.8, 67.1, 62.2, 62.3, 67.5]
res["V6_table6_average_check"] = {
    "KPGrasp_each_dataset_mean": round(float(np.mean(t6_each)), 2),
    "KPGrasp_each_dataset_reported": 47.8,
    "original_DGA_datasets_mean": round(float(np.mean(t6_orig)), 2),
    "original_DGA_datasets_reported": 42.8,
    "KPGrasp_on_Dexonomy_mean": round(float(np.mean(t6_dexo)), 2),
    "KPGrasp_on_Dexonomy_reported": 63.8,
    "DGA_model_mean": round(float(np.mean([57.9, 44.8, 79.1, 53.1, 57.5])), 2),
}

# 표 2 대 표 6 교차 확인: KPGrasp(Dexonomy 학습) 행은 표 2와 같아야 함
res["V7_table6_dexonomy_row_equals_table2"] = t6_dexo == t2_sr["KPGrasp"][0]

# 본문 수치: 'DGA 평균 58.5 → 재학습 59.1'과 '46.6/46.8 대 63.8'은 표가 없어 검산 불가
res["V8_text_only_numbers"] = {
    "DGA_retrained_on_Dexonomy_avg": 59.1,
    "DGA_no_refinement_original": 46.6,
    "DGA_no_refinement_on_Dexonomy": 46.8,
    "status": "표 없음, 검산 불가",
}

# ---------------------------------------------------------------- V9: 표 3 (ablation)
t3 = {
    "PoseJoint_Euclidean": dict(GSR=62.5, OSR=92.0, CDC=10.3, PD=3.7, Div=25.7),
    "PoseJoint_SE3": dict(GSR=55.0, OSR=93.9, CDC=11.8, PD=5.7, Div=23.8),
    "PoseJoint_24tok": dict(GSR=63.5, OSR=94.6, CDC=9.7, PD=3.3, Div=25.8),
    "PoseJoint_34tok": dict(GSR=66.1, OSR=95.1, CDC=9.5, PD=3.2, Div=25.7),
    "Keypoint_1tok": dict(GSR=55.4, OSR=93.2, CDC=10.1, PD=8.6, Div=23.7),
    "Keypoint_no_rank": dict(GSR=67.0, OSR=96.4, CDC=7.4, PD=4.4, Div=23.2),
    "Keypoint_ours": dict(GSR=76.3, OSR=97.3, CDC=5.6, PD=2.4, Div=23.3),
}
res["V9_ablation"] = {
    "rank_gain_pp": round(76.3 - 67.0, 1),
    "best_poseJoint_gsr": max(t3[k]["GSR"] for k in t3 if k.startswith("PoseJoint")),
    "keypoint_minus_best_poseJoint_pp": round(76.3 - 66.1, 1),
    "note_v1_vs_v2": "v1 표 3의 'Multi-Token' 행(66.1/95.1/9.5/3.2/25.7)은 v2의 34-token 행과 같다. v2는 24-token 행(63.5)을 추가했다.",
    "seed_variance_reported": False,
}
# 가장 가까운 차이 (67.0 대 66.1 = 0.9 pp)는 시드 분산 없이는 해석할 수 없다.
# 같은 테스트 격자에서 두 비율의 2표본 검정 (독립 가정, 객체 군집 무시)
def two_prop_z(p1, p2, n1, n2):
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return z, 2 * (1 - stats.norm.cdf(abs(z)))

z, pv = two_prop_z(0.763, 0.670, n_grasp, n_grasp)
res["V10_rank_gain_two_prop_z_assumed_n"] = {"z": round(z, 2), "p": float(f"{pv:.3g}"),
                                              "caveat": "학습 시드 분산과 객체 군집을 무시한 하한 수준의 불확실성"}
z, pv = two_prop_z(0.661, 0.763, n_grasp, n_grasp)
res["V11_keypoint_vs_34tok_two_prop"] = {"z": round(z, 2), "p": float(f"{pv:.3g}"),
                                          "caveat": "같은 시드 조건 비교만 반영"}

# ---------------------------------------------------------------- V12: 시간과 처리량 (5.5절)
t_dga = 2.8
t_kp = 0.032
gsr = 0.763
dex_grasps = 9.5e6
dex_days = 2
dex_gpus = 8
res["V12_speed"] = {
    "speedup_vs_DGA": round(t_dga / t_kp, 1),
    "paper_says": "around 87",
    "kp_valid_per_s": round(gsr / t_kp, 1),
    "paper_says_kp_valid": "approximately 24",
    "dexonomy_per_gpu_per_s": round(dex_grasps / (dex_days * 86400 * dex_gpus), 2),
    "paper_says_dexonomy": "less than 7",
    "ratio_kp_over_dexonomy": round((gsr / t_kp) / (dex_grasps / (dex_days * 86400 * dex_gpus)), 2),
    "caveat": "Dexonomy 수치는 전체 9.5M 생성 격자를 전부 '유효'로 간주한 값이고, GPU 종류와 IK·필터 시간이 다르다. 비교 조건이 같지 않다.",
}

# ---------------------------------------------------------------- V13: 실제 로봇 (5.4절)
k, n = 83, 100
p_exact = stats.binomtest(k, n, 0.5, alternative="greater").pvalue
res["V13_realworld"] = {
    "successes": k, "trials": n,
    "rate_pct": k,
    "wilson95_pct": wilson(k, n),
    "binomial_p_vs_50pct_one_sided": float(f"{p_exact:.2e}"),
    "caveat": "객체 20개 × 5회. 같은 객체의 시도는 독립이 아니다. 객체별 결과가 없어 군집 구간은 계산하지 못했다.",
}

# ---------------------------------------------------------------- V14: IK (부록 A)
res["V14_ik"] = {
    "stated_objects_x_scales_x_proposals": 2139 * 6 * 10,
    "stated_IK_attempts": 128340,
    "fraction_max_err_gt_5mm_pct": round(100 - 97.37, 2),
    "fraction_clear_failures_pct": 0.2,
    "note": "5 mm 기준으로 2.63%가 실패로 분류되지만 본문의 '명확한 실패' 0.2%와 기준이 다르다. 두 값은 다른 임계값이다.",
    "text_0_7mm_vs_table": {"reported_avg_mean_err_mm": 0.669, "main_text_rounded_mm": 0.7, "ok": round(0.669, 1) == 0.7},
}

# ---------------------------------------------------------------- V15: 학습 규모 (5절, 부록)
samples_main = 200_000 * 4096
samples_512 = 1_600_000 * 512
res["V15_training_samples"] = {
    "main_200k_x_4096": samples_main,
    "scaling_1p6M_x_512": samples_512,
    "equal": samples_main == samples_512,
    "sec_per_step_main_2days": round(2 * 86400 / 200_000, 3),
    "sec_per_step_512_13days": round(13 * 86400 / 1_600_000, 3),
    "caveat": "배치가 8배 작은데 스텝당 시간은 약 0.81배다. GPU 수나 병렬 설정이 다를 가능성이 있어 확인이 필요하다.",
}

# ---------------------------------------------------------------- V16: 샘플링 단계 (부록 D)
res["V16_sampling_steps"] = {
    "5step_gsr_equal_main": 76.3,
    "7step_gsr": 76.0,
    "7step_minus_5step_pp": round(76.0 - 76.3, 1),
    "ok_no_gain_beyond_5": 76.0 <= 76.3,
    "time_per_grasp_s": {"1step": 0.01, "3step": 0.024, "5step": 0.032, "7step": 0.04},
}

# ---------------------------------------------------------------- V18: 그림 5 (스케일링) 표시 값
# 그림 5에 인쇄된 숫자를 읽었다(그림 판독). 본문 텍스트 값과 대조한다.
fig5 = {
    "data": {"x": [0.001, 0.01, 0.1, 1.0], "KP": [34.2, 54.6, 69.5, 76.3], "PJ": [33.0, 43.5, 51.9, 62.5], "DGA": [16.0, 19.3, 20.3, 20.9]},
    "model": {"x": [2, 4, 8, 16], "KP": [56.3, 71.9, 76.3, 77.6], "PJ": [48.6, 58.6, 62.5, 65.3], "DGA": [21.8, 20.9, 18.1, 19.0]},
    "batch": {"x": [8, 64, 512, 4096], "KP": [29.1, 57.2, 70.3, 76.3], "PJ": [15.1, 34.9, 53.4, 62.5], "DGA": [15.3, 19.5, 20.9, 20.4]},
}
gaps = {}
for panel, d in fig5.items():
    gaps[panel] = [round(a - b, 1) for a, b in zip(d["KP"], d["PJ"])]
res["V18_fig5_read"] = {
    "gap_KP_minus_PJ": gaps,
    "batch512_figure_KP": 70.3,
    "batch512_text_KP": 73.5,
    "batch512_diff_pp": round(70.3 - 73.5, 1),
    "KP_4096_matches_table1": fig5["batch"]["KP"][-1] == 76.3,
    "PJ_4096_matches_table3_euclid": fig5["batch"]["PJ"][-1] == 62.5,
    "gap_widens_monotonic_batch": all(x < y for x, y in zip(gaps["batch"], gaps["batch"][1:])),
    "gap_widens_monotonic_model": all(x < y for x, y in zip(gaps["model"], gaps["model"][1:])),
    "note": "그림 판독값이다. 본문은 '모든 설정에서 gap이 규모와 함께 벌어진다'고 쓴다. batch 축은 512에서 16.9 pp, 64에서 22.3 pp로 중간에 최대이고 4096에서 13.8 pp로 줄어든다. model 축도 M=8 이후 줄어든다.",
}

# ---------------------------------------------------------------- V17: 비율 기술 (% 대 pp)
res["V17_wording"] = {
    "KPGrasp_vs_classifier_text": "outperforms the classifier-based Dexonomy variant by 12.4% in GSR",
    "correct_unit": "12.4 percentage points (상대 증가 약 19.4%)",
    "relative_vs_classifier_pct": round(100 * 12.4 / 63.9, 1),
}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2)[:4000])
print("wrote", OUT)
