"""ControlTac(arXiv:2505.20498 v3) 보고 수치 재계산. 결과는 recompute_results.json.

점검 항목 V1-V13 (논문 표·그림·본문 수치만 사용, 학습·생성 실행 없음).
- 논문에 없는 시행 수나 표본 수를 가정한 계산에는 "assumed_n" 키로 가정을 남긴다.
- 그림 판독 값(그림 6, 그림 5, 그림 8 막대 라벨)은 "figure_read"로 표시한다.
실행: python3 analysis/recompute.py   (numpy, scipy; 수 초)
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
R = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def welch(m1, s1, n1, m2, s2, n2):
    t, p = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
    return float(t), float(p)


def cohen_d(m1, s1, m2, s2):
    sp = math.sqrt((s1 ** 2 + s2 ** 2) / 2)
    return (m1 - m2) / sp


# ---------------------------------------------------------------- V1 부록 B.2 산술
v1 = {}
v1["stage1_hours"] = 50 * 1500 / 3600
v1["stage2_hours"] = 30 * 550 / 3600
v1["total_hours"] = v1["stage1_hours"] + v1["stage2_hours"]
v1["params_sum_M"] = 147.53 + 222.93
v1["stage2_minus_stage1_M"] = 222.93 - 147.53
v1["steps"] = 75000
v1["implied_steps_per_epoch_stage1"] = 75000 / 50
v1["implied_steps_per_epoch_stage2"] = 75000 / 30
v1["implied_sec_per_step_stage1"] = 1500 / (75000 / 50)
v1["implied_sec_per_step_stage2"] = 550 / (75000 / 30)
v1["dataset_epochs_if_batch4_stage1"] = 75000 * 4 / 20000
v1["dataset_epochs_if_batch4_stage2"] = 75000 * 4 / 7000
v1["dataset_epochs_if_16_pairs_stage1"] = 75000 * 16 / 20000
v1["step_time_ratio_stage1_over_stage2"] = v1["implied_sec_per_step_stage1"] / v1["implied_sec_per_step_stage2"]
R["V1_training_arithmetic"] = v1

# ---------------------------------------------------------------- V2 표 1 대 참조 복사(E.1)
t1 = {
    "FAF_seen": {"Simulator": (1054, 19, .68, .03), "Sim2Real": (239, 17, .74, .03), "Hybrid": (31, 5, .81, .04),
                 "Separate": (157, 8, .79, .04), "Ours": (23, 2, .83, .03)},
    "FAF_unseen": {"Simulator": (1065, 23, .69, .03), "Sim2Real": (253, 25, .73, .03), "Hybrid": (37, 6, .75, .04),
                   "Separate": (199, 11, .72, .05), "Ours": (26, 3, .79, .04)},
    "AnyTouch2": {"Simulator": (2157, 14, .61, .04), "Sim2Real": (545, 31, .70, .05), "Ours": (29, 2, .81, .02)},
}
refcopy = {"FAF_unseen": (233, 25, .78, .05), "AnyTouch2": (81, 12, .80, .03)}
v2 = {"ref_copy_vs_ours": {}, "ssim_below_refcopy": {}, "mse_ratio_vs_ours": {}}
for ds, rc in refcopy.items():
    o = t1[ds]["Ours"]
    v2["ref_copy_vs_ours"][ds] = {
        "mse_ratio_refcopy_over_ours": rc[0] / o[0],
        "ssim_diff_ours_minus_refcopy": round(o[2] - rc[2], 3),
        "ssim_cohen_d": round(cohen_d(o[2], o[3], rc[2], rc[3]), 2),
    }
    v2["ssim_below_refcopy"][ds] = [k for k, v in t1[ds].items() if v[2] < rc[2]]
    v2["mse_ratio_vs_ours"][ds] = {k: round(v[0] / o[0], 1) for k, v in t1[ds].items()}
v2["mse_gt_refcopy_methods_FAF_unseen"] = [k for k, v in t1["FAF_unseen"].items() if v[0] > refcopy["FAF_unseen"][0]]
v2["note"] = "표 1의 SD가 어떤 단위(시험 영상)의 SD인지와 n은 본문에 없어 검정은 하지 않고 비와 효과 크기만 낸다."
R["V2_table1_vs_reference_copy"] = v2

# ---------------------------------------------------------------- V3 마스크 섭동(표 4, 5)
base = (23, 2, .83)
t4 = {"S0.8": (24, .83), "S0.9": (23, .83), "S1.1": (23, .83), "S1.2": (24, .83),
      "T2": (23, .83), "T4": (24, .83), "T6": (25, .82), "R2": (23, .83), "R4": (25, .82), "R6": (27, .82)}
t5 = {"S0.9+T4+R2": (23, .82), "S1.1+T4+R4": (26, .82), "S0.8+T6+R6": (28, .82), "S1.1+T6+R6": (28, .82)}
R["V3_mask_perturbation"] = {
    "mse_rel_increase_pct_table4": {k: round(100 * (v[0] / base[0] - 1), 1) for k, v in t4.items()},
    "mse_rel_increase_pct_table5": {k: round(100 * (v[0] / base[0] - 1), 1) for k, v in t5.items()},
    "baseline_matches_table1_seen": True,
    "alignment_precision_stated": {"v3_main_3.2": "±3 px, ±1°", "v3_E.5_typical_max": "4 px, 2°", "v1_3.1": "1 px, 1°"},
    "worst_single_R6_vs_base_SD_units": round((27 - 23) / 2, 1),
    "worst_combined_vs_base_SD_units": round((28 - 23) / 2, 1),
}

# ---------------------------------------------------------------- V4 표 6 (힘 추정 MAE)
real = {10: (0.061, .017), 11: (0.057, .015), 12: (0.055, .015), 13: (0.051, .014), 14: (0.054, .012), 15: (0.053, .012)}
ours = {15: (0.21, .13), 30: (0.17, .09), 45: (0.18, .09), 60: (0.16, .08)}
mix = {15: (0.055, .021), 30: (0.060, .017), 45: (0.058, .015), 60: (0.061, .014)}
N = 1200
v4 = {"assumed_n": f"Fig. 6 캡션의 N = {N}을 표 6에도 적용(독립 Welch, 같은 시험 표본이면 쌍체가 더 민감)"}
v4["real_best_k"] = min(real, key=lambda k: real[k][0])
v4["real_nonmonotone_after_13k"] = real[14][0] > real[13][0]
def cmp4(a, b):
    t, p = welch(*a, N, *b, N)
    return {"diff": round(b[0] - a[0], 3), "cohen_d": round(cohen_d(b[0], b[1], a[0], a[1]), 2), "welch_p_if_independent": p}


v4["real15_vs_real15_plus_gen15"] = cmp4(real[15], mix[15])
v4["real13_vs_real15"] = cmp4(real[13], real[15])
v4["real10_vs_real15"] = cmp4(real[10], real[15])
v4["mix15_vs_mix60"] = cmp4(mix[15], mix[60])
v4["interpretation"] = "n=1200 독립 가정에서는 0.002 N 차이도 p<0.01이 나오므로 p보다 효과 크기(d)를 본다. 차이 0.002–0.008 N은 SD 0.012–0.021의 0.1–0.5배"
v4["gen_only_vs_real15_ratio"] = {k: round(v[0] / real[15][0], 2) for k, v in ours.items()}
v4["gen_only_nonmonotone"] = [(a, b) for a, b in zip([15, 30, 45], [30, 45, 60]) if ours[b][0] > ours[a][0]]
v4["mix_nonmonotone_pairs"] = [(a, b) for a, b in zip([15, 30, 45], [30, 45, 60]) if mix[b][0] < mix[a][0]]
R["V4_table6_force_mae"] = v4

# ---------------------------------------------------------------- V5 그림 6(b) 판독 값 (그림 판독 값, 눈금 ±0.01)
fig6b = {  # (Real only, +15k, +30k) ; Gen only 는 (None, 15k, 30k)
    "Gen only": (None, 0.27, 0.23), "1/20": (0.53, 0.24, 0.21), "1/4": (0.37, 0.18, 0.16),
    "1/3": (0.29, 0.12, 0.10), "1/2": (0.24, 0.10, 0.08), "Full": (0.06, 0.065, 0.065)}
v5 = {"figure_read": True, "figure_values_N": fig6b}
v5["one_third_plus30k_over_full_real"] = round(fig6b["1/3"][2] / fig6b["Full"][0], 2)
v5["one_third_plus15k_over_full_real"] = round(fig6b["1/3"][1] / fig6b["Full"][0], 2)
v5["half_plus30k_over_full_real"] = round(fig6b["1/2"][2] / fig6b["Full"][0], 2)
v5["gen_only_values_three_sources"] = {
    "table6_v3_(Ours 15k/30k)": [0.21, 0.17], "fig6b_read_(Gen only 15k/30k)": [0.27, 0.23],
    "v1_table5_(1-10N 15k/30k)": [0.35, 0.33], "v1_table5_(4-10N 15k/30k)": [0.21, 0.17]}
v5["real_full_values"] = {"table6_real15k": 0.053, "fig6b_full_read": 0.06, "E1_real_pairs": 0.05, "E1_generated_pairs": 0.23}
v5["E1_generated_over_real_ratio"] = round(0.23 / 0.05, 1)
v5["range_statement"] = {"v3_main_4.2": "1-10 N, 0.1 N precision", "v1_4.2": "4-10 N for the full-framework test (real 20,000 -> 15,000)",
                         "v3_E.3": "Fz in [1,10] N"}
R["V5_fig6b_force_scaling"] = v5

# ---------------------------------------------------------------- V6 표 2 자세 추정
t2 = {  # (X mean, sd, Y mean, sd, A mean, sd)
    "cyl": {"PCA": (15, 6, 13, 3, 22, 4), "Real": (8, 3, 8, 2, 4, 2), "Sim": (17, 3, 15, 3, 6, 2), "Sim+Real": (12, 3, 13, 3, 6, 3),
            "fixed": (9, 2, 8, 2, 5, 1), "unfixed": (4, 1, 5, 1, 3, 1), "unfixed+Real": (3, 1, 4, 1, 3, 1)},
    "cross": {"PCA": (56, 9, 19, 5, 18, 5), "Real": (6, 2, 6, 2, 2, 2), "Sim": (19, 3, 18, 2, 5, 2), "Sim+Real": (17, 4, 16, 3, 4, 3),
              "fixed": (7, 2, 9, 2, 4, 2), "unfixed": (3, 1, 4, 1, 1, 1), "unfixed+Real": (2, 2, 4, 1, 1, 1)},
}
n_pose = 30
v6 = {"assumed_n": f"객체당 시험 자세 {n_pose}개(본문 4.2)를 독립 표본으로 가정, 정수 반올림 값이라 p는 대략적", "welch": {}}
for obj, rows in t2.items():
    for cmp_a, cmp_b in [("unfixed", "Real"), ("fixed", "Real"), ("unfixed+Real", "unfixed"), ("Sim+Real", "Real")]:
        a, b = rows[cmp_a], rows[cmp_b]
        for i, ax in enumerate(["X", "Y", "Angle"]):
            t, p = welch(a[2 * i], a[2 * i + 1], n_pose, b[2 * i], b[2 * i + 1], n_pose)
            v6["welch"][f"{obj}:{cmp_a}_vs_{cmp_b}:{ax}"] = {"diff": a[2 * i] - b[2 * i], "p": p}
v6["cells_unfixed_better_than_real_p_lt_0.05"] = sum(
    1 for k, v in v6["welch"].items() if "unfixed_vs_Real" in k and "+" not in k.split(":")[1] and v["diff"] < 0 and v["p"] < .05)
v6["cells_unfixed_vs_real_total"] = 6
v6["cells_fixed_vs_real"] = {k: v for k, v in v6["welch"].items() if ":fixed_vs_Real:" in k}
v6["v1_to_v3_changes"] = {"cyl real X": ("9", "8±3"), "cross unfixed Y": ("5", "4±1"),
                          "note": "v1 표 2는 값만(정수), v3는 평균±SD. 위 두 칸은 같은 칸에서 정수가 달라짐"}
v6["pairs_per_pose_E1"] = {"cyl_600_pairs/(3 types x 30 poses)": 600 / 90, "cross_200_pairs/30 poses": 200 / 30}
v6["unit_conflict"] = {"table2_caption_v3": "X/Y in mm", "E.1_and_Fig13_15": "pixels", "E.11": "1 pixel = 1/20 mm",
                       "cyl_real_X_8_px_in_mm": 8 / 20, "cyl_real_X_8_if_mm_vs_insertion_tolerance_3mm": "8 mm > 3 mm"}
R["V6_table2_pose"] = v6

# ---------------------------------------------------------------- V7 표 7 물체 밀기
gt = np.array([2.24, 0.96, 1.06, 1.08])
rl = np.array([2.38, 1.08, 1.18, 1.14])
ou = np.array([2.36, 1.11, 1.17, 1.16])
sd_r = np.array([.35, .14, .17, .23])
sd_o = np.array([.45, .21, .19, .27])
v7 = {"signed_err_real": (rl - gt).round(3).tolist(), "signed_err_ours": (ou - gt).round(3).tolist(),
      "mae_real": float(np.abs(rl - gt).mean()), "mae_ours": float(np.abs(ou - gt).mean()),
      "ours_minus_real": (ou - rl).round(3).tolist(), "max_abs_ours_minus_real": float(np.abs(ou - rl).max()),
      "ours_over_gt_all": bool((ou > gt).all()), "real_over_gt_all": bool((rl > gt).all()),
      "ours_minus_real_over_sd_real": (np.abs(ou - rl) / sd_r).round(2).tolist(),
      "relative_err_ours_pct": (100 * (ou - gt) / gt).round(1).tolist(),
      "assumed_n": "v1 4.5: 물체당 5회 밀기(약 15 s); v3에는 n 없음. 평균 차이는 SD의 14% 이하라 n과 무관하게 구별되지 않는 크기"}
R["V7_table7_pushing"] = v7

# ---------------------------------------------------------------- V8 삽입 성공률 Wilson
ins = {"cylinder": (18, 20), "cross": (17, 20), "T-shape": (17, 20), "Type-C": (15, 20)}
R["V8_insertion"] = {"counts": {k: list(v) for k, v in ins.items()}, "wilson95": {k: wilson(*v) for k, v in ins.items()},
                     "pooled_3_printed": {"k": 52, "n": 60, "wilson95": wilson(52, 60)},
                     "fisher_cyl_vs_typec": fisher(18, 20, 15, 20),
                     "note": "삽입 실험에는 증강 없음 대조군이 없다(본문 4.3, 부록 E.11). 성공률은 절대 수치만 보고된다."}

# ---------------------------------------------------------------- V9 모방학습 (그림 8, 막대 라벨 판독)
demos = [20, 35, 50, 70, 100]
il = {"No Aug": [24, 28, 48, 56, 60], "Aug(1)": [20, 32, 56, 64, 68], "Aug(2)": [16, 28, 60, 64, 64],
      "Aug(4)": [24, 36, 68, 68, 72], "Aug(9)": [20, 44, 72, 68, 72]}
n = 25
v9 = {"figure_read": "그림 8 막대 위 숫자", "n_per_cell_assumed_from_text": n,
      "all_values_multiple_of_4": all(v % 4 == 0 for r in il.values() for v in r)}
k = {a: [round(x * n / 100) for x in r] for a, r in il.items()}
v9["counts"] = k
v9["wilson95_noaug"] = {d: wilson(kk, n) for d, kk in zip(demos, k["No Aug"])}
rows = []
for a in ["Aug(1)", "Aug(2)", "Aug(4)", "Aug(9)"]:
    for d, ka, k0 in zip(demos, k[a], k["No Aug"]):
        rows.append(fisher(ka, n, k0, n))
hp = holm(rows)
v9["fisher_vs_noaug"] = {}
i = 0
for a in ["Aug(1)", "Aug(2)", "Aug(4)", "Aug(9)"]:
    for d, ka, k0 in zip(demos, k[a], k["No Aug"]):
        v9["fisher_vs_noaug"][f"{a}@{d}"] = {"succ": f"{ka}/{n} vs {k0}/{n}", "diff_pp": round(100 * (ka - k0) / n),
                                              "p": rows[i], "p_holm_20": hp[i]}
        i += 1
v9["n_comparisons"] = len(rows)
v9["min_p_raw"] = min(rows)
v9["min_p_holm"] = min(hp)


def cmh(tables):
    """tables: list of ((k1,n1),(k2,n2)). Mantel-Haenszel chi2 (연속성 보정 없음), 양측."""
    num = 0.0
    var = 0.0
    for (a1, n1), (a2, n2) in tables:
        N_ = n1 + n2
        m1 = a1 + a2
        m0 = N_ - m1
        exp = n1 * m1 / N_
        v = n1 * n2 * m1 * m0 / (N_ ** 2 * (N_ - 1))
        num += a1 - exp
        var += v
    chi = num ** 2 / var
    return float(chi), float(stats.chi2.sf(chi, 1)), float(num)


v9["cmh_pooled_over_5_settings"] = {}
for a in ["Aug(1)", "Aug(2)", "Aug(4)", "Aug(9)"]:
    chi, p, dev = cmh([((ka, n), (k0, n)) for ka, k0 in zip(k[a], k["No Aug"])])
    v9["cmh_pooled_over_5_settings"][a] = {"chi2": chi, "p": p, "excess_successes": dev,
                                          "pooled_rate_aug": round(100 * sum(k[a]) / (5 * n), 1),
                                          "pooled_rate_noaug": round(100 * sum(k["No Aug"]) / (5 * n), 1)}
v9["any_aug_arms_note"] = "네 증강 팔을 하나로 합친 CMH는 같은 대조군을 4번 재사용하므로 독립이 아니다. 참고용으로 두지 않는다."
# E.2 인용 값
v9["E2_claims"] = {
    "20 demos 5/25 vs 6/25": {"wilson_5": wilson(5, 25), "wilson_6": wilson(6, 25), "fisher_p": fisher(5, 25, 6, 25)},
    "35 demos 44% vs 28%": {"11/25 vs 7/25 fisher_p": fisher(11, 25, 7, 25), "wilson_11": wilson(11, 25), "wilson_7": wilson(7, 25)},
    "50 demos 72% vs 48%": {"18/25 vs 12/25 fisher_p": fisher(18, 25, 12, 25), "wilson_18": wilson(18, 25), "wilson_12": wilson(12, 25)},
}
v9["aug_monotone_in_copies"] = {d: [il[a][i] for a in ["No Aug", "Aug(1)", "Aug(2)", "Aug(4)", "Aug(9)"]] for i, d in enumerate(demos)}
v9["aug9_minus_noaug_pp"] = {d: il["Aug(9)"][i] - il["No Aug"][i] for i, d in enumerate(demos)}
v9["aug_best_minus_noaug_pp"] = {d: max(il[a][i] for a in il if a != "No Aug") - il["No Aug"][i] for i, d in enumerate(demos)}
v9["noaug_curve_20_to_100"] = il["No Aug"]
v9["power_25_per_arm"] = {}
for p0, p1 in [(0.28, 0.44), (0.48, 0.72), (0.56, 0.68)]:
    # 양측 Fisher 근사 검정력을 모의로 추정
    rng = np.random.default_rng(0)
    hits = 0
    M = 4000
    for _ in range(M):
        a = rng.binomial(25, p0)
        b = rng.binomial(25, p1)
        hits += fisher(b, 25, a, 25) < .05
    v9["power_25_per_arm"][f"{p0}->{p1}"] = hits / M


def n_needed(p0, p1, power=.8, alpha=.05):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pb = (p0 + p1) / 2
    return math.ceil(((za * math.sqrt(2 * pb * (1 - pb)) + zb * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) / (p1 - p0)) ** 2)


v9["n_per_arm_for_80pct_power"] = {"0.28->0.44": n_needed(.28, .44), "0.48->0.72": n_needed(.48, .72), "0.56->0.68": n_needed(.56, .68)}
R["V9_imitation_learning"] = v9

# ---------------------------------------------------------------- V10 분류 (그림 6a)
fig6a = {"Geo+Col": (0.69, 0.65, 0.79), "Simulator": (0.79, 0.85, 0.92), "Ours": (0.87, 0.95, 0.99), "Text2Tac": (0.22, 0.16, 0.14)}
v1_t3_4800_gc = (0.69, 0.65, 0.79)
v1_t3_4800_gen = (0.87, 0.95, 0.99)
v10 = {"chance_6_classes": 1 / 6, "text2tac_below_chance": [m for m, v in zip(["CNN", "ViT-scratch", "ViT-ImageNet"], fig6a["Text2Tac"]) if v < 1 / 6],
       "fig6a_geo_col_equals_v1_table3_4800_GC": fig6a["Geo+Col"] == v1_t3_4800_gc,
       "fig6a_ours_equals_v1_table3_4800_gen": fig6a["Ours"] == v1_t3_4800_gen,
       "v1_geometric_aug_count": {"8x4x3x9": 8 * 4 * 3 * 9, "with_6_hue_variants": 8 * 4 * 3 * 9 * 6},
       "text2tac_prompt_combos": 10 * 10 * 6 * 8,
       "real_200_per_object_samples": 200 * 6,
       "ours_vit_imagenet_vs_real_200_per_object_ref_0.80": 0.99 - 0.80,
       "accuracy_granularity": "시험 표본 수가 본문에 없어 0.01 단위가 몇 개 표본인지 정할 수 없다",
       "ours_minus_simulator": [round(a - b, 2) for a, b in zip(fig6a["Ours"], fig6a["Simulator"])],
       "ours_minus_geocol": [round(a - b, 2) for a, b in zip(fig6a["Ours"], fig6a["Geo+Col"])]}
R["V10_classification"] = v10

# ---------------------------------------------------------------- V11 처리량과 규모
v11 = {"images_per_s": 6.5, "batch": 128, "sec_per_batch": 128 / 6.5, "hours_per_million": 1e6 / 6.5 / 3600,
       "hours_for_60k": 6e4 / 6.5 / 3600, "hours_for_30k": 3e4 / 6.5 / 3600,
       "hybrid_images_per_s": 7, "separate_images_per_s": 3.7,
       "separate_over_ours_time": 6.5 / 3.7, "gpu_mem_gb": 29.97,
       "ratio_gen_to_train_stage1": 60000 / 20000,
       "mask_prep_min_per_object": 3, "align_s": 10}
R["V11_throughput_scale"] = v11

# ---------------------------------------------------------------- V12 교차 센서 (그림 5a 판독 값)
v12 = {"figure_read": True, "mse_0": 40, "mse_2000": 16, "ssim_0": 0.80, "ssim_2000": 0.85, "mse_500": 23,
       "table1_seen_mse": 23, "table1_unseen_faf_mse": 26,
       "note": "9DTact 미세 조정 2000장의 MSE(약 16)는 표 1의 학습 센서 값(23)보다 낮다. 데이터셋이 달라 직접 비교하지 않는다. 영벡터 시점 SSIM 약 0.80은 FeelAnyForce의 참조 복사 SSIM 0.78과 가깝다(데이터셋이 달라 참고용)."}
R["V12_cross_sensor"] = v12

# ---------------------------------------------------------------- V13 버전 간 수치 대조
R["V13_version_diff"] = {
    "v1_v2_text_identical_except_date": True,
    "v1_table1": {"Hybrid": (31, .81), "Separate": (157, .79), "Ours_first_stage": (18, .84), "Ours": (23, .83)},
    "v3_table1_has_same_Hybrid_Separate_Ours": True, "v3_drops_ours_first_stage_row": True,
    "v1_table2_vs_v3": "정수 → 평균±SD, USB 추가, 칸 2곳 정수 차이(V6)",
    "v1_table5_dropped_in_v3": {"1-10N": (0.35, 0.33), "4-10N": (0.21, 0.17)},
    "v1_table7_dropped_in_v3": {"100pos": (0.272, 0.077), "200pos": (0.207, 0.067), "300pos": (0.174, 0.060)},
    "v1_real_experiments": 3, "v3_real_experiments": 4,
    "v1_title": "ControlTac: Force- and Position-Controlled Tactile Data Augmentation with a Single Reference Image",
    "v3_title": "ControlTac: Scaling Tactile Data with Physically Controlled Tactile Image Generation",
    "v1_mask_precision": "1 px, 1 degree", "v3_mask_precision": "±3 px, ±1 degree",
    "added_in_v3": ["Simulator/Sim2Real 기준선", "AnyTouch2 미관측 평가", "참조 복사 기준선(E.1)", "9DTact 교차 센서(그림 5)",
                    "모방학습(그림 8, E.2, E.12)", "Type-C 삽입과 자세 추정", "실패 분석(부록 F)", "마스크 섭동(표 4, 5)", "학습 시간·추론 속도(B.2, B.3)"],
}

OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False, default=lambda o: list(o) if isinstance(o, tuple) else str(o)), encoding="utf-8")
print("wrote", OUT)
for key in ["V1_training_arithmetic", "V8_insertion"]:
    print(key, json.dumps(R[key], ensure_ascii=False)[:400])
