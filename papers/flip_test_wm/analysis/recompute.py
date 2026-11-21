"""The Flip Test (arXiv:2611.19822 v1) 표와 본문 수치를 재계산하고 내부 일관성을 검사한다.

입력: Table 3(부록, 모델별 쌍 결과 빈도)과 본문에 적힌 수치. 논문 PDF는 읽지 않는다(값은 아래에 옮겨 적었다).
출력: analysis/recompute_results.json (키 이름은 r2 등 내부 규칙을 따른다)
실행: python3 analysis/recompute.py   (numpy, scipy, 1초 미만)
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CHANCE = 0.25  # 이진 과제의 쌍 우연 수준 (한 프레임 0.5 x 두 프레임)
N = 150        # 분할별 쌍 수

# Table 3: (solved, same, swap, uncl, disagr) 순서. 시뮬레이션 150쌍, 실사 150쌍.
T3 = {
    "Gemini Omni Flash 1.1": ((8, 100, 4, 11, 27), (47, 86, 4, 2, 11)),
    "MiniMax-H3": ((5, 111, 2, 5, 27), (38, 87, 5, 0, 20)),
    "Wan2.2-I2V-A14B": ((5, 85, 5, 17, 38), (37, 82, 8, 1, 22)),
    "VBVR-Wan2.2": ((8, 95, 3, 9, 35), (34, 84, 5, 2, 25)),
    "Cosmos3-Super": ((8, 80, 6, 10, 46), (33, 71, 7, 10, 29)),
    "Cosmos3-Nano": ((8, 65, 6, 25, 46), (24, 77, 8, 4, 37)),
    "Cosmos-Predict2.5": ((5, 74, 5, 16, 50), (26, 76, 2, 7, 39)),
    "VPP2": ((2, 91, 2, 14, 41), (22, 94, 3, 1, 30)),
    "HunyuanVideo-1.5": ((4, 89, 4, 22, 31), (17, 91, 6, 2, 34)),
    "LingBot-Video-MoE-30B-A3B": ((4, 36, 4, 46, 60), (15, 90, 2, 1, 42)),
    "CogVideoX1.5-5B-I2V": ((3, 31, 1, 62, 53), (14, 74, 5, 8, 49)),
}
# 본문에 적힌 값 (퍼센트). 모델 전체 평균 = 시뮬레이션과 실사의 단순 평균.
TEXT_OVERALL = {
    "Gemini Omni Flash 1.1": 18.3, "Cosmos3-Super": 13.7, "Wan2.2-I2V-A14B": 14.0,
    "VBVR-Wan2.2": 14.0, "Cosmos-Predict2.5": 10.3, "Cosmos3-Nano": 10.7,
    "VPP2": 8.0, "LingBot-Video-MoE-30B-A3B": 6.3, "CogVideoX1.5-5B-I2V": 5.7,
}
TEXT_REAL = {"Gemini Omni Flash 1.1": 31.3, "MiniMax-H3": 25.3, "Wan2.2-I2V-A14B": 24.7, "CogVideoX1.5-5B-I2V": 9.3}


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def pct(x):
    return round(100 * x, 2)


def main():
    out = {}
    checks = {}

    # 1. 행 합계 검사 (solved+same+swap+uncl+disagr = 150)
    bad_rows = []
    for m, (s, r) in T3.items():
        for label, row in (("sim", s), ("real", r)):
            if sum(row) != N:
                bad_rows.append((m, label, sum(row)))
    checks["table3_row_sums_all_150"] = len(bad_rows) == 0
    checks["table3_bad_rows"] = bad_rows

    # 2. 쌍 정확도와 본문 값 대조
    per_model = {}
    for m, (s, r) in T3.items():
        sim_pct, real_pct = pct(s[0] / N), pct(r[0] / N)
        overall = round((sim_pct + real_pct) / 2, 2)
        per_model[m] = {
            "sim_solved": s[0], "real_solved": r[0],
            "sim_pair_acc_pct": sim_pct, "real_pair_acc_pct": real_pct,
            "overall_pct": overall,
            "text_overall_pct": TEXT_OVERALL.get(m),
            "text_real_pct": TEXT_REAL.get(m),
        }
    out["per_model"] = per_model
    checks["overall_matches_text"] = all(
        abs(v["overall_pct"] - v["text_overall_pct"]) < 0.06
        for v in per_model.values() if v["text_overall_pct"] is not None)
    checks["real_matches_text"] = all(
        abs(v["real_pair_acc_pct"] - v["text_real_pct"]) < 0.06
        for v in per_model.values() if v["text_real_pct"] is not None)
    checks["sim_max_pct"] = max(v["sim_pair_acc_pct"] for v in per_model.values())  # 본문 5.3
    checks["sim_models_at_ceiling"] = sum(1 for v in per_model.values() if v["sim_solved"] == 8)  # 본문 4개
    checks["real_range_real_minus_sim_pp"] = [
        round(min(v["real_pair_acc_pct"] - v["sim_pair_acc_pct"] for v in per_model.values()), 1),
        round(max(v["real_pair_acc_pct"] - v["sim_pair_acc_pct"] for v in per_model.values()), 1)]  # 본문 7–26

    # 3. 명백히 판정된 쌍 중 같은 결과 비율 (본문 실사 72%, 시뮬 89%)
    def same_share(idx):
        same = sum(T3[m][idx][1] for m in T3)
        solved = sum(T3[m][idx][0] for m in T3)
        swap = sum(T3[m][idx][2] for m in T3)
        return {"same": same, "solved": solved, "swap": swap,
                "same_over_clear": round(same / (same + solved + swap), 4)}
    out["same_share_sim"] = same_share(0)
    out["same_share_real"] = same_share(1)
    # 본문 "시뮬 89%, 실사 72%"는 same/(same+solved+swap)과 맞는지 확인
    checks["same_share_sim_matches_89"] = round(out["same_share_sim"]["same_over_clear"] * 100) == 89
    checks["same_share_real_matches_72"] = round(out["same_share_real"]["same_over_clear"] * 100) == 72

    # 4. 불확실 판정(시뮬) 비율: 본문 "약 14%"
    unc_sim = sum(T3[m][0][3] for m in T3)
    out["uncl_share_sim_pct"] = round(100 * unc_sim / (11 * N), 2)

    # 5. 쌍 정확도 불확실성: Wilson 95% 구간, 이항 검정(우연 0.25 대비)
    ci = {}
    for m in ("Gemini Omni Flash 1.1", "MiniMax-H3", "Wan2.2-I2V-A14B"):
        s, r = T3[m]
        for label, k in (("real", r[0]), ("sim", s[0])):
            lo, hi = wilson(k, N)
            p_two = stats.binomtest(k, N, CHANCE).pvalue
            p_up = stats.binomtest(k, N, CHANCE, alternative="greater").pvalue
            ci[f"{m} | {label}"] = {"k": k, "acc_pct": pct(k / N),
                                    "wilson95_pct": [pct(lo), pct(hi)],
                                    "binom_p_two_sided_vs_0.25": float(f"{p_two:.3g}"),
                                    "binom_p_greater_vs_0.25": float(f"{p_up:.3g}")}
    out["pair_ci_and_binomial"] = ci
    # 시뮬 전 모델 상한: 우연 수준에서 8/150 이하일 확률
    out["sim_all_models_below_chance_p"] = {
        m: float(f"{stats.binomtest(T3[m][0][0], N, CHANCE, alternative='less').pvalue:.3g}") for m in T3}

    # 6. 우연 수준 보정: 3지선다 과제(뉴턴 요람 1개 + 실사 시소) 쌍의 우연은 (1/3)^2 = 0.111
    def chance_adj(n3):
        return (N - n3) / N * CHANCE + n3 / N * (1 / 9)
    out["chance_adjusted_pct"] = {
        "real_split_if_seesaw_3way_10_of_150": pct(chance_adj(10)),
        "sim_split_if_newton_cradle_3way_15_of_150": pct(chance_adj(15)),
        "all_300_if_25_three_way": pct((275 * CHANCE + 25 / 9) / 300),
    }

    # 6b. 모델 간 차이의 검정력: 전체 평균(쌍 300개)의 차이 SE를 독립 이항으로 근사 (C3: "규모 효과 작음")
    def diff_ci(a, b):
        pa = (T3[a][0][0] + T3[a][1][0]) / (2 * N)
        pb = (T3[b][0][0] + T3[b][1][0]) / (2 * N)
        se = math.sqrt(pa * (1 - pa) / (2 * N) + pb * (1 - pb) / (2 * N))
        d = pa - pb
        return {"diff_pp": round(100 * d, 2), "approx_95ci_pp": [round(100 * (d - 1.96 * se), 2), round(100 * (d + 1.96 * se), 2)]}
    out["model_diff_power"] = {
        "Cosmos3-Super minus Cosmos-Predict2.5": diff_ci("Cosmos3-Super", "Cosmos-Predict2.5"),
        "Wan2.2 minus VBVR-Wan2.2": diff_ci("Wan2.2-I2V-A14B", "VBVR-Wan2.2"),
        "Gemini Omni minus Cosmos3-Super": diff_ci("Gemini Omni Flash 1.1", "Cosmos3-Super"),
    }

    # 7. 측정 에이전트 대 직접 VLM: 실사 쌍 140/150 대 131/150 (독립 가정 Fisher, 쌍 단위 자료가 없어 지표만)
    tab = np.array([[140, 10], [131, 19]])
    out["measure_vs_direct_vlm_fisher_p"] = float(f"{stats.fisher_exact(tab)[1]:.3g}")
    tab2 = np.array([[47, 103], [131, 19]])  # Omni 대 직접 VLM, 실사
    out["omni_vs_direct_vlm_fisher_p"] = float(f"{stats.fisher_exact(tab2)[1]:.3g}")

    # 8. 판독: 단일 판정기 대 이중 판정 규칙 (Omni 실사 34.7% 대 31.3%)
    out["omni_real_single_judge_vs_dual"] = {
        "single_3_8_pct": 34.7, "dual_pct": pct(47 / N),
        "single_k_equiv": round(0.347 * N, 1), "dual_k": 47,
        "inflation_pp": round(34.7 - pct(47 / N), 2)}

    # 9. 판독 정확도 상한: 독립 오류 가정에서 완벽한 생성기의 쌍 정확도 = a^2
    out["readout_ceiling_pair_acc_pct"] = {
        "human_agreement_gemini36_0.79": pct(0.79 ** 2),
        "judge_pair_agreement_0.858": pct(0.858 ** 2),
        "reported_told_real_omni_0.867": pct(0.867),
    }

    # 10. 제어 집단 SS의 근사 구간 (150 제어 집단 = 10 과제 x 15, CFR과 SFR은 이항으로 가정)
    cfr, sfr = 0.61, 0.15
    se = math.sqrt(cfr * (1 - cfr) / 150 + sfr * (1 - sfr) / 300)
    out["selective_sensitivity_vlm"] = {
        "ss": round(cfr - sfr, 3), "approx_se": round(se, 3),
        "approx_95ci": [round(cfr - sfr - 1.96 * se, 3), round(cfr - sfr + 1.96 * se, 3)],
        "paper_95ci": [0.36, 0.56]}
    # 비디오 모델의 SS 범위 ±0.07에서 CI가 0을 덮으려면 SE가 약 0.036 이상이어야 함
    out["video_ss_ci_halfwidth_needed"] = round(0.07 / 1.96, 3)

    # 11. 에이전트 표본 복원 (본문 수치 -> 개수)
    out["agent_counts"] = {
        "direct_vlm_real_pairs": 131, "direct_vlm_real_pairs_pct": pct(131 / N),
        "measure_real_pairs": 140, "measure_real_pairs_pct": pct(140 / N),
        "sim_agent_real_pairs": 137, "sim_agent_real_pairs_pct": pct(137 / N),
        "direct_vlm_sweep_frames": 356, "direct_vlm_sweep_pct": pct(356 / 480),
        "measure_sweep_correct": 477, "measure_sweep_pct": pct(477 / 480),
        "pre_sim_real_frames": 288, "pre_sim_real_pct": pct(288 / 300),
        "pre_sim_sweep_frames": 468, "pre_sim_sweep_pct": pct(468 / 480),
        "sim_agent_sweep_errors_reported": 9,
        "pre_sim_sweep_errors": 480 - 468,
        "rollout_sweep_corrections_reported": 3,
    }
    checks["sim_agent_sweep_error_balance"] = (480 - 468) - 3 == 9
    checks["measure_sweep_errors_3"] = (480 - 477) == 3
    out["measure_revisions_net"] = 52 - 4

    # 12. 호출 수: 6600 (L1 11 x 600) + 6600 (L2 11 x 600) + 1440 (L3 3 x 480) + 6600 (told) = 21240 (본문 "21,000 이상")
    out["rollout_count"] = 11 * 600 + 11 * 600 + 3 * 480 + 11 * 600
    # 프레임 수: 600 + 600 + 480 = 1680
    out["frame_count"] = 600 + 600 + 480
    checks["frame_count_1680"] = out["frame_count"] == 1680
    checks["rollout_over_21000"] = out["rollout_count"] > 21000

    # 13. 판정 일치율: 시뮬 83.5%, 실사 88.2% 평균과 전체 85.8%
    out["judge_agree_mean_of_splits_pct"] = round((83.5 + 88.2) / 2, 2)

    out["checks"] = checks
    (ROOT / "analysis" / "recompute_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print("same_share sim/real:", out["same_share_sim"]["same_over_clear"], out["same_share_real"]["same_over_clear"])
    print("uncl sim %:", out["uncl_share_sim_pct"])
    print("chance adj:", out["chance_adjusted_pct"])
    print("SS:", out["selective_sensitivity_vlm"])
    print("readout ceiling:", out["readout_ceiling_pair_acc_pct"])
    print("pair CI:", json.dumps(ci, ensure_ascii=False, indent=1)[:1500])
    print("sim below chance p:", out["sim_all_models_below_chance_p"])
    print("fisher measure vs vlm:", out["measure_vs_direct_vlm_fisher_p"], "omni vs vlm:", out["omni_vs_direct_vlm_fisher_p"])


if __name__ == "__main__":
    main()
