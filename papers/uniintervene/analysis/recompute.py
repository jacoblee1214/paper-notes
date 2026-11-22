"""UniIntervene (arXiv:2606.12372 v1) 논문 표 수치를 다시 계산하고 일관성을 검사한다.

입력: 논문 본문과 표의 값을 아래에 그대로 옮겨 적었다(출처는 주석). 외부 데이터는 쓰지 않는다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 과제당 SR의 분모는 20회다(4.1절 "20 evaluation episodes per task").
- Table 2의 SR은 두 과제(Pick Eggplant, RAM Insertion)의 합쳐진 값이라 분모를 40회로 본다.
  Table 1의 두 과제 값 평균과 맞는지 따로 검사한다.
- IR은 세 번의 독립 학습 결과라고 쓰여 있으나 표준편차나 개별 값은 없다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe_diff(k1, n1, k2, n2, z=1.959964):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="two-sided")[1])


def comp(name, k1, n1, k2, n2):
    d, lo, hi = newcombe_diff(k1, n1, k2, n2)
    return {"comparison": name, "k1": k1, "n1": n1, "k2": k2, "n2": n2,
            "diff_pp": round(100 * d, 1),
            "newcombe95_pp": [round(100 * lo, 1), round(100 * hi, 1)],
            "fisher_p": round(fisher(k1, n1, k2, n2), 4)}


def mantel_haenszel(tables):
    """층화 2x2 표(과제별)의 Mantel-Haenszel 검정(연속성 보정 없음)과 공통 오즈비."""
    num = var = 0.0
    r_num = r_den = 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        row1, col1 = a + b, a + c
        e = row1 * col1 / n
        v = row1 * (c + d) * col1 * (b + d) / (n * n * (n - 1))
        num += a - e
        var += v
        r_num += a * d / n
        r_den += b * c / n
    chi2 = num * num / var
    p = float(1 - stats.chi2.cdf(chi2, 1))
    return chi2, p, (r_num / r_den if r_den > 0 else float("inf"))


# ------------------------------------------------------------------ 1. Table 1 (논문 7쪽)
tasks = ["Pick Eggplant", "Tube Insertion", "RAM Insertion", "Wipe Whiteboard", "Fold Towel"]
SR = {
    "pi0.5 SFT":       [95, 30, 10, 65, 70],
    "HiL-SERL":        [90, 60, 85, 85, 85],
    "HiL-SERL+FA-RL":  [85, 60, 75, 80, 85],
    "HiL-SERL+Uni":    [95, 70, 95, 90, 90],
}
SR_AVG_PAPER = {"pi0.5 SFT": 54, "HiL-SERL": 81, "HiL-SERL+FA-RL": 77, "HiL-SERL+Uni": 88}
IR = {
    "HiL-SERL":        [28.7, 30.2, 32.3, 30.5, 49.8],
    "HiL-SERL+FA-RL":  [20.4, 22.1, 27.9, 21.9, 30.9],
    "HiL-SERL+Uni":    [10.0, 15.8, 12.1, 10.9, 24.1],
}
IR_AVG_PAPER = {"HiL-SERL": 34.3, "HiL-SERL+FA-RL": 24.6, "HiL-SERL+Uni": 14.6}
N_EVAL = 20

# 1.1 정수 환산과 평균
RES["t1_counts"] = {m: [round(v * N_EVAL / 100) for v in vals] for m, vals in SR.items()}
RES["t1_integer_feasible"] = all(abs(v * N_EVAL / 100 - round(v * N_EVAL / 100)) < 1e-9
                                 for vals in SR.values() for v in vals)
RES["t1_sr_avg_recomputed"] = {m: float(np.mean(v)) for m, v in SR.items()}
RES["t1_sr_avg_matches_paper"] = {m: abs(float(np.mean(SR[m])) - SR_AVG_PAPER[m]) < 0.05 for m in SR}
RES["t1_ir_avg_recomputed"] = {m: round(float(np.mean(v)), 3) for m, v in IR.items()}
RES["t1_ir_avg_matches_paper"] = {m: abs(round(float(np.mean(IR[m])), 1) - IR_AVG_PAPER[m]) < 1e-9 for m in IR}

# 1.2 초록과 캡션의 "8.6% / 57%"
sr_u, sr_h, sr_f = np.mean(SR["HiL-SERL+Uni"]), np.mean(SR["HiL-SERL"]), np.mean(SR["HiL-SERL+FA-RL"])
ir_u, ir_h, ir_f = (np.mean(IR[k]) for k in ("HiL-SERL+Uni", "HiL-SERL", "HiL-SERL+FA-RL"))
RES["headline"] = {
    "sr_gain_abs_pp_vs_hilserl": float(sr_u - sr_h),
    "sr_gain_rel_pct_vs_hilserl": float(100 * (sr_u - sr_h) / sr_h),
    "sr_gain_abs_pp_vs_fa": float(sr_u - sr_f),
    "sr_gain_rel_pct_vs_fa": float(100 * (sr_u - sr_f) / sr_f),
    "ir_cut_rel_pct_vs_hilserl_ratio_of_means": float(100 * (ir_h - ir_u) / ir_h),
    "ir_cut_rel_pct_vs_hilserl_mean_of_taskratios": float(np.mean([100 * (h - u) / h for h, u in zip(IR["HiL-SERL"], IR["HiL-SERL+Uni"])])),
    "ir_cut_rel_pct_vs_fa": float(100 * (ir_f - ir_u) / ir_f),
    "ir_cut_abs_pp_vs_hilserl": float(ir_h - ir_u),
    "ir_ratio_uni_over_hilserl": float(ir_u / ir_h),
    "ir_cut_per_task_pct_vs_hilserl": [round(100 * (h - u) / h, 1) for h, u in zip(IR["HiL-SERL"], IR["HiL-SERL+Uni"])],
}

# 1.3 과제별 Fisher, Wilson, 합산
k = RES["t1_counts"]
RES["t1_wilson95_pct"] = {m: [[round(100 * x, 1) for x in wilson(kk, N_EVAL)] for kk in k[m]] for m in k}
per_task = []
for i, t in enumerate(tasks):
    per_task.append({
        "task": t,
        "uni_vs_hilserl": comp(t, k["HiL-SERL+Uni"][i], 20, k["HiL-SERL"][i], 20),
        "uni_vs_fa": comp(t, k["HiL-SERL+Uni"][i], 20, k["HiL-SERL+FA-RL"][i], 20),
        "uni_vs_sft": comp(t, k["HiL-SERL+Uni"][i], 20, k["pi0.5 SFT"][i], 20),
    })
RES["t1_per_task"] = per_task
tot = {m: sum(v) for m, v in k.items()}
RES["t1_pooled_counts_of_100"] = tot
RES["t1_pooled"] = [comp("Uni vs HiL-SERL (100 vs 100)", tot["HiL-SERL+Uni"], 100, tot["HiL-SERL"], 100),
                    comp("Uni vs FA-RL (100 vs 100)", tot["HiL-SERL+Uni"], 100, tot["HiL-SERL+FA-RL"], 100),
                    comp("HiL-SERL vs FA-RL (100 vs 100)", tot["HiL-SERL"], 100, tot["HiL-SERL+FA-RL"], 100),
                    comp("Uni vs pi0.5 SFT (100 vs 100)", tot["HiL-SERL+Uni"], 100, tot["pi0.5 SFT"], 100)]
for ref in ("HiL-SERL", "HiL-SERL+FA-RL"):
    tabs = [(k["HiL-SERL+Uni"][i], 20 - k["HiL-SERL+Uni"][i], k[ref][i], 20 - k[ref][i]) for i in range(5)]
    chi2, p, orr = mantel_haenszel(tabs)
    RES[f"t1_mh_uni_vs_{ref}"] = {"chi2": round(chi2, 3), "p": round(p, 4), "common_odds_ratio": round(orr, 2)}
# 과제 부호 검정: Uni >= 기준선인 과제 수
RES["t1_sign"] = {"uni_ge_hilserl_tasks": int(sum(u >= h for u, h in zip(SR["HiL-SERL+Uni"], SR["HiL-SERL"]))),
                  "uni_gt_hilserl_tasks": int(sum(u > h for u, h in zip(SR["HiL-SERL+Uni"], SR["HiL-SERL"]))),
                  "uni_gt_fa_tasks": int(sum(u > h for u, h in zip(SR["HiL-SERL+Uni"], SR["HiL-SERL+FA-RL"]))),
                  "tie_with_sft_on_eggplant": SR["HiL-SERL+Uni"][0] == SR["pi0.5 SFT"][0]}
RES["t1_sign"]["sign_test_p_5of5_one_sided"] = 0.5 ** 5
RES["t1_sft_vs_hilserl_eggplant_drop_pp"] = SR["pi0.5 SFT"][0] - SR["HiL-SERL"][0]

# 1.4 검정력: 88 대 81을 80% 검정력으로 구분하는 데 필요한 팀당 시행 수 (정규 근사, 양측 5%)
def n_per_arm(p1, p2, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return math.ceil((za + zb) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p1 - p2) ** 2)

RES["t1_power"] = {
    "n_per_arm_88_vs_81": n_per_arm(0.88, 0.81),
    "n_per_arm_95_vs_90": n_per_arm(0.95, 0.90),
    "n_per_arm_95_vs_85": n_per_arm(0.95, 0.85),
    "n_per_arm_70_vs_60": n_per_arm(0.70, 0.60),
    "n_per_arm_88_vs_77": n_per_arm(0.88, 0.77),
}

# 1.5 IR: 세 번 독립 학습(n=3) 가정에서 과제별 차이가 유의하려면 SD가 얼마 이하여야 하는가
t_crit = stats.t.ppf(0.975, 4)
sens = []
for i, t in enumerate(tasks):
    for ref in ("HiL-SERL", "HiL-SERL+FA-RL"):
        d = IR[ref][i] - IR["HiL-SERL+Uni"][i]
        sens.append({"task": t, "vs": ref, "diff_pp": round(d, 1),
                     "max_pooled_sd_for_p05": round(d / (t_crit * math.sqrt(2 / 3)), 2)})
RES["t1_ir_sd_sensitivity"] = sens
RES["t1_ir_note"] = "IR 표준편차와 개별 실행 값이 논문에 없어 검정 불가. 위 값은 'SD가 이 값 이하면 Welch p<0.05'의 상한(양측, df=4 근사)."

# ------------------------------------------------------------------ 2. Table 2 (8쪽)
abl = {  # variant: (qloss, f1, SR, IR)
    "w/o Future Pred.": (0.005, 0.878, 90, 15.8),
    "w/o Value Pred.":  (None, 0.845, 85, 18.7),
    "w/o TVR":          (0.004, 0.832, 85, 16.9),
    "w/o Memory Goal":  (0.004, 0.882, 85, 16.1),
    "UniIntervene":     (0.004, 0.882, 95, 11.1),
}
RES["t2_table"] = {m: {"qloss": v[0], "f1": v[1], "sr": v[2], "ir": v[3]} for m, v in abl.items()}
# 2.1 Table 1과의 일관성: 두 과제 평균
r2 = {"SR_mean_two_tasks": (SR["HiL-SERL+Uni"][0] + SR["HiL-SERL+Uni"][2]) / 2,
      "IR_mean_two_tasks": (IR["HiL-SERL+Uni"][0] + IR["HiL-SERL+Uni"][2]) / 2}
RES["t2_consistency_with_t1"] = {**r2, "table2_full_sr": 95, "table2_full_ir": 11.1,
                                  "ir_exact_half_rounding": "11.05 -> 11.1 (반올림 규칙에 따라 11.0 또는 11.1)"}
# 2.2 본문 문장 점검
f1s = {m: v[1] for m, v in abl.items()}
lowest_f1 = min(f1s, key=f1s.get)
RES["t2_text_checks"] = {
    "text_says_w/o_value_pred_has_lowest_F1_0.845": False,
    "actual_lowest_F1_variant": lowest_f1,
    "actual_lowest_F1": f1s[lowest_f1],
    "text_says_w/o_value_pred_has_highest_IR_18.7": abl["w/o Value Pred."][3] == max(v[3] for v in abl.values()),
    "text_says_future_and_memory_keep_F1_nearly_intact": [f1s["w/o Future Pred."], f1s["w/o Memory Goal"]],
    "text_says_SR_drops_95_to_90_and_85": [abl["w/o Future Pred."][2], abl["w/o Memory Goal"][2]],
    "full_best_on_all_metrics_strict": {"qloss_tie": abl["w/o TVR"][0] == abl["UniIntervene"][0] == abl["w/o Memory Goal"][0],
                                        "f1_tie_with_w/o_Memory": f1s["w/o Memory Goal"] == f1s["UniIntervene"]},
    "three_variants_share_SR_85": [m for m, v in abl.items() if v[2] == 85],
}
# 2.3 SR 95/90/85를 40회 환산하고 Fisher
n_abl = 40
RES["t2_counts_of_40"] = {m: round(v[2] * n_abl / 100) for m, v in abl.items()}
RES["t2_vs_full"] = []
for m, v in abl.items():
    if m == "UniIntervene":
        continue
    RES["t2_vs_full"].append(comp(f"Full vs {m}", round(95 * n_abl / 100), n_abl, round(v[2] * n_abl / 100), n_abl))
RES["t2_wilson95_full_pct"] = [round(100 * x, 1) for x in wilson(38, 40)]
RES["t2_power"] = {"n_per_arm_95_vs_85": n_per_arm(0.95, 0.85), "n_per_arm_95_vs_90": n_per_arm(0.95, 0.90)}

# 2.4 절제 변형을 두 과제 기준선(HiL-SERL, FA-RL)의 두 과제 평균과 비교
base_ir = (IR["HiL-SERL"][0] + IR["HiL-SERL"][2]) / 2
base_sr = (SR["HiL-SERL"][0] + SR["HiL-SERL"][2]) / 2
fa_sr = (SR["HiL-SERL+FA-RL"][0] + SR["HiL-SERL+FA-RL"][2]) / 2
fa_ir = (IR["HiL-SERL+FA-RL"][0] + IR["HiL-SERL+FA-RL"][2]) / 2
RES["t2_vs_baselines_two_task_mean"] = {
    "hilserl_sr": base_sr, "hilserl_ir": base_ir, "fa_sr": fa_sr, "fa_ir": fa_ir,
    "ir_cut_pct_vs_hilserl": {m: round(100 * (base_ir - v[3]) / base_ir, 1) for m, v in abl.items()},
    "sr_minus_hilserl_pp": {m: round(v[2] - base_sr, 1) for m, v in abl.items()},
}

# ------------------------------------------------------------------ 3. 부록 표 (13-20쪽)
T5 = {  # task: (candidates, items, bins, quota, span, delta)
    "Pick Eggplant": (11879, 120, 10, 12, 8, 0.40),
    "Tube Insertion": (14861, 120, 10, 12, 8, 0.40),
    "RAM Insertion": (40742, 120, 10, 12, 8, 0.40),
    "Wipe Whiteboard": (17638, 120, 10, 12, 8, 0.40),
    "Fold Towel": (3574, 240, 20, 12, 16, 0.50),
}
RES["t5_items_equal_bins_x_quota"] = {t: v[1] == v[2] * v[3] for t, v in T5.items()}
RES["t5_candidates_range"] = [min(v[0] for v in T5.values()), max(v[0] for v in T5.values())]
RES["t5_items_over_candidates_pct"] = {t: round(100 * v[1] / v[0], 2) for t, v in T5.items()}
RES["fig7_vs_t5_fold_candidates"] = {"fig7_caption": 3973, "table5": 3574, "diff": 3973 - 3574,
                                      "items_over_candidates_fig7_pct": round(100 * 240 / 3973, 2)}
T6 = {"Insertion (RAM+Tube)": (14950, 0.992, 0.452), "Pick Eggplant": (5580, 0.992, 0.563),
      "Wipe Whiteboard": (5529, 0.993, 0.365), "Fold Towel": (2360, 0.957, 0.339)}
RES["t6_text_checks"] = {
    "top1_sim_all_above_0.95": all(v[1] > 0.95 for v in T6.values()),
    "top1_sim_0.99_on_insertion_and_pick": T6["Insertion (RAM+Tube)"][1] >= 0.99 and T6["Pick Eggplant"][1] >= 0.99,
    "value_lift_range": [min(v[2] for v in T6.values()), max(v[2] for v in T6.values())],
    "text_range_0.34_to_0.56": True,
    "total_queries": sum(v[0] for v in T6.values()),
    "queries_per_memory_item_insertion": round(14950 / 240, 1),
    "queries_per_memory_item_pick": round(5580 / 120, 1),
    "queries_per_memory_item_fold": round(2360 / 240, 1),
    "note": "Table 6의 Insertion은 RAM과 Tube를 합쳤지만 Table 5는 각각 120개 기억을 따로 적었다.",
}
# 부록 C.4 값 분리와 NTTG 선형 값의 평균
RES["c4_mean_value_success_ramp"] = {"linear_ramp_mean_T42": float(np.mean(np.arange(42) / 41)),
                                      "linear_ramp_mean_T95": float(np.mean(np.arange(95) / 94)),
                                      "paper_success_mean_insertion_and_eggplant": 0.53}
# Table 3/4와 식 9의 가중치
RES["eq9_weights"] = {"lambda": [1.0, 0.3, 1.0, 0.05], "main_text_form": "L_TD + L_prog + 0.05 L_CQL",
                      "note": "본문식은 L_label의 0.3을 L_prog 안에 묶어 쓴다. 가중치 자체는 표 3과 일치."}
# 4.3: 사례 두 개의 트리거 시점
RES["fig3_claims"] = {"ram_insertion_horizon_steps": 42, "ram_trigger_step": 13, "ram_trigger_frac": round(13 / 42, 3),
                      "fold_horizon_steps": 95, "fold_trigger_step": 8, "fold_trigger_frac": round(8 / 95, 3),
                      "window_K": 8, "note": "Fold Towel의 트리거 8단계는 K=8 창이 처음 가득 차는 시점과 같다."}
# 부록 H, E: 하이퍼 값의 크기 의미
gam, K, eps = 0.9, 8, 0.005
S = sum(gam ** i for i in range(K))
RES["tvr_scale"] = {"sum_gamma_r_pow": S, "R_flat_window_V0": eps * S, "sigmoid_of_that": 1 / (1 + math.exp(-eps * S)),
                    "eps_equiv_episode_len_steps": 1 / eps}

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")

# ------------------------------------------------------------------ 요약 출력
print("== Table 1 평균 재계산")
for m in SR:
    print(f"  SR {m:16s} {np.mean(SR[m]):5.1f} (논문 {SR_AVG_PAPER[m]})")
for m in IR:
    print(f"  IR {m:16s} {np.mean(IR[m]):5.2f} (논문 {IR_AVG_PAPER[m]})")
h = RES["headline"]
print(f"== 8.6%: 절대 {h['sr_gain_abs_pp_vs_hilserl']:.1f}p, 상대 {h['sr_gain_rel_pct_vs_hilserl']:.2f}% / 57%: 상대 {h['ir_cut_rel_pct_vs_hilserl_ratio_of_means']:.1f}%")
print("== 합산 Fisher")
for c in RES["t1_pooled"]:
    print("  ", c["comparison"], c["diff_pp"], c["newcombe95_pp"], "p=", c["fisher_p"])
print("== 과제별 Uni vs HiL-SERL")
for r in per_task:
    c = r["uni_vs_hilserl"]
    print(f"  {r['task']:16s} {c['k1']}/20 vs {c['k2']}/20  p={c['fisher_p']}")
print("== MH", RES["t1_mh_uni_vs_HiL-SERL"], RES["t1_mh_uni_vs_HiL-SERL+FA-RL"])
print("== 검정력", RES["t1_power"])
print("== Table 2 Fisher")
for c in RES["t2_vs_full"]:
    print("  ", c["comparison"], c["diff_pp"], c["newcombe95_pp"], "p=", c["fisher_p"])
print("== Table 2 본문 점검:", RES["t2_text_checks"]["actual_lowest_F1_variant"], RES["t2_text_checks"]["actual_lowest_F1"])
print("== Fig7/표5 Fold 후보 수:", RES["fig7_vs_t5_fold_candidates"])
print("== TVR 규모:", RES["tvr_scale"])
