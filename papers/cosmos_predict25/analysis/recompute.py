"""논문 표 수치 재계산: 실로봇 성공률(표 13), 사람 선호도(그림 5–7), 벤치마크 합산(표 6–11),
정량 표의 차이(표 12, 14, 15, 17, 18, 19, 20), 데이터 산술(표 2, 5, 본문 2.1).

실행: python3 analysis/recompute.py  -> analysis/recompute_results.json (수 초, CPU, 네트워크 없음)
"""
import json
import math
from pathlib import Path

from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
R = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


# ---- 1. 표 13 실로봇 (10 시나리오 x 3회) ----
scen = ["Base", "Mangosteen", "Orange Bowl", "Beige Table", "Black Table", "Light On",
        "Distractors", "Black Cabinet", "Open Drawers", "Combo"]
tab13 = {
    "Base":     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Baseline": [3, 0, 2, 0, 0, 0, 0, 0, 0, 0],
    "Proposed": [3, 3, 3, 1, 1, 2, 3, 2, 3, 3],
}
tot_reported = {"Base": 1, "Baseline": 5, "Proposed": 24}
res13 = {"n_per_scenario": 3, "n_scenarios": 10}
for m, row in tab13.items():
    k, n = sum(row), 3 * len(row)
    res13[m] = {"successes": k, "trials": n, "rate": round(k / n, 4), "wilson95": wilson(k, n),
                "row_sum_matches_reported": k == tot_reported[m]}
f_pb = stats.fisher_exact([[24, 6], [5, 25]])
f_pbase = stats.fisher_exact([[24, 6], [1, 29]])
f_bb = stats.fisher_exact([[5, 25], [1, 29]])
res13["fisher_two_sided"] = {
    "Proposed_vs_Baseline_p": float(f_pb.pvalue),
    "Proposed_vs_Base_p": float(f_pbase.pvalue),
    "Baseline_vs_Base_p": float(f_bb.pvalue),
}
# 시나리오 단위 쌍체 비교 (시도가 시나리오 안에서 묶여 있어 독립 가정이 과함)
diff = [p - b for p, b in zip(tab13["Proposed"], tab13["Baseline"])]
pos = sum(d > 0 for d in diff)
neg = sum(d < 0 for d in diff)
ties = sum(d == 0 for d in diff)
sign_p = stats.binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else None
res13["scenario_sign_test_proposed_vs_baseline"] = {
    "per_scenario_diff": dict(zip(scen, diff)), "positive": pos, "negative": neg, "tied": ties,
    "two_sided_p": round(float(sign_p), 4)}
R["table13_real_robot"] = res13

# ---- 2. 사람 선호도 (그림 6, 7): N 미보고 ----
pref = {
    "2B_post vs Wan2.2-5B (Fig 6)": (30.0, 26.2, 43.8),
    "2B_post vs Wan2.1-14B (Fig 6)": (33.0, 34.8, 32.2),
    "14B_post vs Wan2.1-14B (Fig 7)": (48.6, 31.8, 19.6),
    "14B_post vs Wan2.2-27B-A14B (Fig 7)": (38.1, 35.9, 26.0),
}
pref_out = {}
for name, (w, l, t) in pref.items():
    dec = w + l
    q = w / dec  # 결정 투표 중 승률
    # 결정 투표 수 n_d에 대한 이항 검정 (무승부 제외) 가 최소 필요 개수
    z = 1.959964
    need = math.ceil((z * 0.5 / abs(q - 0.5)) ** 2)
    pref_out[name] = {
        "sum_pct": round(w + l + t, 1),
        "win_share_of_decisive": round(q, 4),
        "decisive_fraction_of_all_votes": round(dec / 100, 4),
        "decisive_votes_needed_for_p05_if_true_share_is_observed": need,
    }
    # 가상 N에서 같은 비율이면 p-value (예: N = 100, 200, 400, 1000 비교)
    for N in (100, 200, 400, 1000):
        nd = round(N * dec / 100)
        kw = round(nd * w / dec)
        pv = stats.binomtest(kw, nd, 0.5).pvalue
        pref_out[name][f"if_N={N}_p_two_sided"] = round(float(pv), 4)
R["human_preference_fig6_7"] = pref_out

# ---- 3. 그림 5 RL 승패 (색 매핑 불명, 합만 검산) ----
R["fig5_sums"] = {"pretrained_plus_RL": round(41.1 + 18.9 + 40.0, 1),
                  "merged_plus_RL": round(37.0 + 16.3 + 46.7, 1)}

# ---- 3b. 그림 3 도메인 SFT 대 사전학습 (막대 라벨을 텍스트 추출 순서대로 읽음: SFT 승, 기준 승, 무승부) ----
fig3 = {"Object Permanence": (50.9, 21.4, 27.7), "High Motion": (44.0, 21.3, 34.7),
        "Complex Scene": (42.6, 22.0, 35.4), "Driving": (47.9, 23.3, 28.8),
        "Robotic Manipulation": (72.6, 19.0, 8.3)}
R["fig3_sft_vs_base_reading"] = {
    k: {"sum_pct": round(sum(v), 1), "decisive_votes_needed_p05": math.ceil((1.959964 * 0.5 / abs(v[0] / (v[0] + v[1]) - 0.5)) ** 2)}
    for k, v in fig3.items()}

# ---- 4. 벤치마크 합산 검사 (Overall = (Domain + Quality)/2, 반올림 0.001) ----
def overall_check(rows):
    out = {}
    for name, (d, q, o) in rows.items():
        out[name] = {"recomputed": round((d + q) / 2, 4), "reported": o,
                     "abs_diff": round(abs((d + q) / 2 - o), 4)}
    return out


t10 = {
    "T2W 2B pre": (0.782, 0.720, 0.751), "T2W 2B post": (0.804, 0.732, 0.768),
    "T2W 14B pre": (0.791, 0.722, 0.757), "T2W 14B post": (0.803, 0.732, 0.768),
    "T2W Wan2.1-1.3B": (0.786, 0.726, 0.756), "T2W Wan2.1-14B": (0.794, 0.727, 0.761),
    "T2W Wan2.2-5B": (0.797, 0.730, 0.764), "T2W Wan2.2-27B": (0.810, 0.728, 0.769),
}
t11 = {
    "I2W 2B pre": (0.824, 0.775, 0.799), "I2W 2B post": (0.840, 0.779, 0.810),
    "I2W 14B pre": (0.835, 0.777, 0.806), "I2W 14B post": (0.838, 0.781, 0.810),
    "I2W Wan2.1-14B": (0.827, 0.768, 0.797), "I2W Wan2.2-5B": (0.834, 0.774, 0.804),
    "I2W Wan2.2-27B": (0.841, 0.772, 0.806),
}
t7 = {"T2W teacher 2B": (0.804, 0.732, 0.768), "T2W distilled 2B": (0.797, 0.731, 0.764)}
t8 = {"I2W teacher 2B": (0.840, 0.779, 0.810), "I2W distilled 2B": (0.842, 0.790, 0.816)}
R["benchmark_overall_check"] = {"table10": overall_check(t10), "table11": overall_check(t11),
                                "table7": overall_check(t7), "table8": overall_check(t8)}
R["distill_vs_teacher_drop"] = {
    "T2W_overall_drop": round(0.768 - 0.764, 3), "T2W_relative_pct": round(100 * (0.764 - 0.768) / 0.768, 2),
    "I2W_overall_gain": round(0.816 - 0.810, 3)}
R["wan_vs_cosmos_gaps"] = {
    "T2W_14B_post_minus_Wan2.2_27B": round(0.768 - 0.769, 3),
    "I2W_2B_post_minus_Wan2.2_27B": round(0.810 - 0.806, 3),
    "I2W_2B_post_minus_Wan2.2_5B": round(0.810 - 0.804, 3)}

# ---- 5. 표 6 RL 보상 합산 ----
t6 = {"T2W pre": (1.55, -0.43, -0.05, 1.08), "T2W pre+RL": (1.69, -0.19, 0.19, 1.69),
      "T2W merged": (1.69, -0.46, -0.01, 1.23), "T2W merged+RL": (1.75, -0.18, 0.18, 1.74),
      "I2W pre": (1.48, -0.76, -0.49, 0.23), "I2W pre+RL": (1.57, -0.70, -0.45, 0.42),
      "I2W merged": (1.57, -0.82, -0.52, 0.24), "I2W merged+RL": (1.57, -0.68, -0.44, 0.45)}
R["table6_sum_check"] = {k: {"components_sum": round(a + b + c, 2), "reported_sum": s,
                             "abs_diff": round(abs(a + b + c - s), 2)} for k, (a, b, c, s) in t6.items()}
R["table6_gain_pre_to_rl_T2W"] = round(1.69 - 1.08, 2)

# ---- 6. 정량 표의 상대 변화 ----
t12_uniform = {"Blur SSIM": (0.82, 0.87), "Edge F1": (0.26, 0.41), "Depth si-RMSE (lower better)": (0.70, 0.67),
               "Seg mIoU": (0.74, 0.76), "Overall Quality": (9.24, 9.31)}
R["table12_uniform_T1_vs_T2.5"] = {k: {"T1": a, "T2.5": b, "delta": round(b - a, 3),
                                        "rel_pct": round(100 * (b - a) / a, 1)} for k, (a, b) in t12_uniform.items()}
t14 = {"FVD StyleGAN T1 vs T2.5 (transfer)": (56.606, 24.222), "FVD I3D T1 vs T2.5": (60.660, 25.692),
       "FID T1 vs T2.5": (22.633, 20.022), "TSE T1 vs T2.5 (lower better?)": (1.017, 1.246),
       "CSE T1 vs T2.5 (lower better?)": (1.835, 2.310)}
R["table14_transfer_ratios"] = {k: {"T1": a, "T2.5": b, "ratio_T1_over_T2.5": round(a / b, 3)}
                                for k, (a, b) in t14.items()}
R["table14_predict_mv_vs_predict1"] = {"FVD StyleGAN ratio": round(63.685 / 23.060, 2),
                                       "FID ratio": round(25.341 / 12.095, 2)}
t15 = {"LET-AP": (0.243, 0.394), "LET-APL": (0.154, 0.254), "LET-APH": (0.236, 0.383),
       "F1": (0.604, 0.637), "x-error far (lower better)": (0.524, 0.487), "Category Acc": (0.899, 0.904)}
R["table15_rel_change_pct"] = {k: round(100 * (b - a) / a, 1) for k, (a, b) in t15.items()}
R["table17_sampson_reduction_pct"] = round(100 * (19.73 - 26.61) / 26.61, 1)
R["table19_bridge_deltas"] = {"PSNR_dB": round(24.95 - 21.14, 2), "SSIM": round(0.85 - 0.82, 2),
                              "LatentL2_rel_pct": round(100 * (0.28 - 0.32) / 0.32, 1),
                              "FVD_rel_pct": round(100 * (146 - 190) / 190, 1)}
R["table20_timeemb_vs_crossattn"] = {"PSNR_dB": round(24.95 - 24.41, 2), "FVD": [146, 159],
                                     "note": "단일 실행, 시드·신뢰구간 없음"}
t18 = {"Object GPT": (91.8, 90.0), "Object Qwen": (69.4, 62.0), "Behavior GPT": (70.2, 59.6),
       "Behavior Qwen": (59.6, 61.7), "Env GPT": (69.0, 69.0), "Env Qwen": (69.0, 65.5)}
R["table18_2.5_minus_2_14B"] = {k: round(a - b, 1) for k, (a, b) in t18.items()}

# ---- 7. 데이터 산술 ----
central = [194, 36, 39, 3, 17, 0.5, 16]
side = [30 + 30, 51 + 51, 6 + 7]
R["table2_central_view_sum_k"] = round(sum(central), 1)
R["table2_side_view_sum_k"] = sum(side)
R["table5_robotics_vs_table2_k"] = {"table5_robotic_manipulation_k": 730,
                                    "table2_all_views_k": round(sum(central) + sum(side), 1)}
R["survival_rate_200M_over_6B_pct"] = round(100 * 0.2 / 6.0, 2)
R["driving_clips_hours_3.1M_x_20s"] = round(3.1e6 * 20 / 3600)

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1))
print(json.dumps(R, ensure_ascii=False, indent=1)[:6000])
