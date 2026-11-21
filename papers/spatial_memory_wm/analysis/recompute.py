"""논문 표 수치 재계산 (Video World Models with Long-term Spatial Memory, arXiv:2506.05284 v1 / NeurIPS 2025).

원자료(개별 샘플)는 공개되어 있지 않으므로, 표에 적힌 평균과 시험 수(논문 본문)만으로 계산한다.
- V1: 사용자 연구 순위 합 검사 (4개 방법의 평균 순위 합은 10이어야 한다)
- V2: 표 1 상대 개선폭
- V3: 사용자 연구 AHR의 검정. 분산은 모르므로 최대 분산(범위 [1,4]에서 평균 m이면 (m-1)(4-m))을 쓴다.
  * 평가 단위 독립 가정: n = 20명 x 14사례 = 280
  * 피험자 단위 군집 가정(보수): n = 20 (피험자 평균 기준. 피험자 평균의 분산 상한도 같은 식을 따른다)
- V4: 표 2, 표 3 차이 (분산과 시험 수가 없어 검정 불가, 차이 크기만 보고)
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# 표 1 (사용자 연구 AHR, 4개 방법, 1–4 척도, 낮을수록 나쁨)
AHR = {
    "Cam-Acc": {"TrajectoryCrafter": 1.6320, "DaS": 2.5660, "Wan2.1-Inpainting": 2.1760, "Ours": 3.6260},
    "Stat-Cons": {"TrajectoryCrafter": 1.7802, "DaS": 2.4396, "Wan2.1-Inpainting": 2.3956, "Ours": 3.3846},
    "Dyn-Plaus": {"TrajectoryCrafter": 1.6255, "DaS": 2.7033, "Wan2.1-Inpainting": 2.2701, "Ours": 3.4011},
}
N_SUBJ = 20
N_CASE = 14
N_RATINGS = N_SUBJ * N_CASE  # 280

# 표 1 뷰 리콜 (PSNR, SSIM, LPIPS)
VIEW = {
    "TrajectoryCrafter": (11.71, 0.4380, 0.5996),
    "DaS": (12.01, 0.4512, 0.5874),
    "Wan2.1-Inpainting": (12.16, 0.4506, 0.5875),
    "Ours": (19.10, 0.6471, 0.3069),
}

# 표 2 VBench (Aesthetic, Imaging, Temporal, Motion, Subject, Background)
VBENCH = {
    "TrajectoryCrafter": (0.5255, 0.6428, 0.6160, 0.9843, 0.8830, 0.9227),
    "DaS": (0.5635, 0.6617, 0.7520, 0.9856, 0.9325, 0.9494),
    "Wan2.1-Inpainting": (0.5661, 0.6788, 0.6433, 0.9868, 0.9357, 0.9513),
    "Ours": (0.5835, 0.6701, 0.7580, 0.9886, 0.9359, 0.9506),
}
VB_NAMES = ["Aesthetic", "Imaging", "Temporal", "Motion", "Subject", "Background"]

# 표 3 절제
ABL = {
    "w/o episodic": (0.5603, 0.6485, 0.7260, 0.9870, 0.9326, 0.9489),
    "w/o working": (0.5551, 0.6384, 0.6740, 0.9862, 0.9331, 0.9453),
    "Full": (0.5835, 0.6701, 0.7580, 0.9886, 0.9359, 0.9506),
}


def norm_two_sided_p(z):
    return math.erfc(abs(z) / math.sqrt(2))


def max_sd_1_4(m):
    """범위 [1,4]에서 평균 m인 분포의 최대 표준편차 (두 점 분포 at 1과 4)."""
    return math.sqrt((m - 1.0) * (4.0 - m))


def ahr_test(ours_mean, base_mean, n):
    se_ours = max_sd_1_4(ours_mean) / math.sqrt(n)
    se_base = max_sd_1_4(base_mean) / math.sqrt(n)
    se_diff = math.sqrt(se_ours**2 + se_base**2)
    diff = ours_mean - base_mean
    z = diff / se_diff
    return {"diff": round(diff, 4), "se_diff_upper": round(se_diff, 4), "z_lower_bound": round(z, 2),
            "p_two_sided_upper": float(f"{norm_two_sided_p(z):.2e}")}


def main():
    out = {}

    # V1: 순위 합
    sums = {}
    for crit, d in AHR.items():
        s = sum(d.values())
        sums[crit] = round(s, 4)
    out["V1_rank_sum_per_criterion"] = sums
    out["V1_rank_sum_expected"] = 10.0
    out["V1_note"] = "4개 방법의 평균 순위 합이 10이면 피험자가 4개 결과를 1–4로 순위 매긴 것과 일치한다."

    # V2: 표 1 상대 개선폭 (가장 좋은 기준선 대비)
    base = {"PSNR": 12.01, "SSIM": 0.4512, "LPIPS": 0.5874}  # DaS가 세 지표 모두 기준선 중 최선(또는 근접)
    ours = {"PSNR": 19.10, "SSIM": 0.6471, "LPIPS": 0.3069}
    best_base_psnr = max(VIEW[k][0] for k in VIEW if k != "Ours")
    best_base_ssim = max(VIEW[k][1] for k in VIEW if k != "Ours")
    best_base_lpips = min(VIEW[k][2] for k in VIEW if k != "Ours")
    out["V2_view_recall"] = {
        "PSNR_gain_dB_vs_best_baseline": round(ours["PSNR"] - best_base_psnr, 2),
        "SSIM_relative_gain_pct": round(100 * (ours["SSIM"] - best_base_ssim) / best_base_ssim, 1),
        "LPIPS_relative_reduction_pct": round(100 * (best_base_lpips - ours["LPIPS"]) / best_base_lpips, 1),
        "best_baseline_PSNR": best_base_psnr,
        "best_baseline_SSIM": best_base_ssim,
        "best_baseline_LPIPS": best_base_lpips,
        "note": "신뢰구간 불가: 샘플별 분산이 없다. 시험 수는 본문에 500 시퀀스(MiraData 테스트)로만 적혀 있다.",
    }

    # V3: AHR 검정 (피험자 단위 군집 보수 가정 포함)
    tests = {}
    for crit, d in AHR.items():
        ours_m = d["Ours"]
        tests[crit] = {}
        for b in ["TrajectoryCrafter", "DaS", "Wan2.1-Inpainting"]:
            tests[crit][b] = {
                "rating_level_n280": ahr_test(ours_m, d[b], N_RATINGS),
                "subject_level_n20": ahr_test(ours_m, d[b], N_SUBJ),
            }
    out["V3_ahr_tests_worst_case_variance"] = tests
    out["V3_assumption"] = ("분산은 범위 [1,4]에서 최대값으로 둔다(보수적). 독립 가정 시 n=280, "
                            "피험자 군집을 고려하면 n=20. 실제 분산은 알 수 없다.")

    # V4: 표 2, 표 3 차이 (검정 불가)
    diffs2 = {}
    for name in ["DaS", "Wan2.1-Inpainting", "TrajectoryCrafter"]:
        diffs2[name] = {VB_NAMES[i]: round(VBENCH["Ours"][i] - VBENCH[name][i], 4) for i in range(6)}
    out["V4_vbench_diff_ours_minus_baseline"] = diffs2
    diffs3 = {}
    for name in ["w/o episodic", "w/o working"]:
        diffs3[name] = {VB_NAMES[i]: round(ABL["Full"][i] - ABL[name][i], 4) for i in range(6)}
    out["V4_ablation_diff_full_minus_ablated"] = diffs3
    out["V4_note"] = ("표 2, 표 3은 평균만 있고 분산과 시험 수가 없다. 차이는 보고만 하고 p값은 계산하지 않는다. "
                      "표 3에 '공간 기억 제거' 행이 없다.")

    # K: 본문과 표의 일관성 검사
    checks = {}
    checks["K1_view_all_best"] = all(
        VIEW["Ours"][0] > VIEW[k][0] and VIEW["Ours"][1] > VIEW[k][1] and VIEW["Ours"][2] < VIEW[k][2]
        for k in VIEW if k != "Ours")
    checks["K2_vbench_wan_imaging_and_background_higher"] = (
        VBENCH["Wan2.1-Inpainting"][1] > VBENCH["Ours"][1] and VBENCH["Wan2.1-Inpainting"][5] > VBENCH["Ours"][5])
    checks["K3_ablation_full_best_all_six"] = all(
        ABL["Full"][i] > max(ABL["w/o episodic"][i], ABL["w/o working"][i]) for i in range(6))
    out["K_consistency_checks"] = checks

    with open(os.path.join(HERE, "recompute_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
