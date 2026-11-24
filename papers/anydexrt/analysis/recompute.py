"""AnyDexRT (arXiv:2607.08341) 논문 표 재계산과 일관성 검사.

입력은 논문 v1, v2의 표 값을 옮긴 상수입니다. 결과는 recompute_results.json에 저장합니다.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

표시 규칙
  - "가정"이 붙은 값은 논문에 없는 시행 수를 가정한 계산입니다.
  - "그림 판독"은 그림에서 눈으로 읽은 근사값입니다.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
HANDS = ["Inspire", "Ability", "XHand", "Wuji", "Allegro", "Leap", "Shadow"]

# 표 1 (x10^-2). 각 칸 (GMC 평균, GMC SD, LMC 평균, LMC SD). SD가 없으면 None.
OPT = {  # 결정적 방법
    "Inspire": (53.9, 43.2), "Ability": (47.7, 39.5), "XHand": (46.9, 33.8), "Wuji": (80.2, 74.2),
    "Allegro": (89.5, 76.1), "Leap": (70.8, 60.0), "Shadow": (45.0, 38.5)}
GEORT = {
    "Inspire": (71.9, 3.5, 50.6, 3.6), "Ability": (65.0, 4.7, 36.2, 11.2), "XHand": (75.4, 4.1, 51.3, 4.2),
    "Wuji": (87.3, 2.6, 77.0, 4.6), "Allegro": (85.5, 1.9, 74.8, 1.6), "Leap": (73.4, 3.2, 53.2, 8.8),
    "Shadow": (89.8, 1.6, 75.2, 4.1)}
ANY_V2 = {
    "Inspire": (84.7, 0.1, 92.1, 0.1), "Ability": (85.0, 0.2, 90.5, 0.1), "XHand": (89.7, 0.2, 94.8, 0.1),
    "Wuji": (93.2, 0.2, 94.1, 0.3), "Allegro": (92.2, 0.1, 94.0, 0.1), "Leap": (92.3, 0.1, 93.2, 0.3),
    "Shadow": (90.8, 0.2, 94.1, 0.1)}
ANY_V1 = {
    "Inspire": (80.7, 0.2, 89.5, 0.2), "Ability": (84.1, 0.2, 88.5, 0.1), "XHand": (76.0, 0.3, 88.5, 0.2),
    "Wuji": (93.1, 0.2, 92.3, 0.2), "Allegro": (87.4, 0.2, 92.1, 0.1), "Leap": (54.5, 0.4, 89.0, 0.2),
    "Shadow": (83.6, 0.1, 91.4, 0.3)}
REPORTED_AVG = {  # (GMC, LMC)
    "opt": (62.0, 52.2), "geort": (78.3, 59.8), "any_v2": (89.7, 93.3), "any_v1": (79.9, 90.2)}
TEXT_LMC_CLAIM = (59.8, 90.2)  # v1, v2 본문 "from 59.8% to 90.2%"

res = {}


def mean_cols(d, idx):
    return float(np.mean([d[h][idx] for h in HANDS]))


# ---- V1 평균 재계산 ----
avg = {
    "opt": (mean_cols(OPT, 0), mean_cols(OPT, 1)),
    "geort": (mean_cols(GEORT, 0), mean_cols(GEORT, 2)),
    "any_v2": (mean_cols(ANY_V2, 0), mean_cols(ANY_V2, 2)),
    "any_v1": (mean_cols(ANY_V1, 0), mean_cols(ANY_V1, 2)),
}
res["V1_table1_average"] = {
    k: {"recomputed_GMC": round(v[0], 2), "recomputed_LMC": round(v[1], 2),
        "reported_GMC": REPORTED_AVG[k][0], "reported_LMC": REPORTED_AVG[k][1],
        "match_to_0.1": bool(abs(round(v[0], 1) - REPORTED_AVG[k][0]) < 0.051 and abs(round(v[1], 1) - REPORTED_AVG[k][1]) < 0.051)}
    for k, v in avg.items()}
res["V1_text_claim"] = {
    "text_says_LMC_from_to": TEXT_LMC_CLAIM,
    "v2_table_average_LMC": REPORTED_AVG["any_v2"][1],
    "v1_table_average_LMC": REPORTED_AVG["any_v1"][1],
    "text_matches_v1_table": TEXT_LMC_CLAIM[1] == REPORTED_AVG["any_v1"][1],
    "text_matches_v2_table": TEXT_LMC_CLAIM[1] == REPORTED_AVG["any_v2"][1],
}

# ---- V2 v1 -> v2 칸별 변화 ----
delta = {}
for h in HANDS:
    delta[h] = {"dGMC": round(ANY_V2[h][0] - ANY_V1[h][0], 1), "dLMC": round(ANY_V2[h][2] - ANY_V1[h][2], 1)}
res["V2_v1_to_v2_delta"] = delta
res["V2_summary"] = {
    "max_dGMC": max(delta.values(), key=lambda x: x["dGMC"])["dGMC"],
    "max_dGMC_hand": max(HANDS, key=lambda h: delta[h]["dGMC"]),
    "mean_dGMC": round(float(np.mean([delta[h]["dGMC"] for h in HANDS])), 2),
    "mean_dLMC": round(float(np.mean([delta[h]["dLMC"] for h in HANDS])), 2),
    "v1_cells_below_best_baseline_GMC": [h for h in HANDS if ANY_V1[h][0] < max(OPT[h][0], GEORT[h][0])],
    "v2_cells_below_best_baseline_GMC": [h for h in HANDS if ANY_V2[h][0] < max(OPT[h][0], GEORT[h][0])],
    "v2_cells_below_best_baseline_LMC": [h for h in HANDS if ANY_V2[h][2] < max(OPT[h][1], GEORT[h][2])],
}

# ---- V3 시드 SD 기반 Welch 검정 (가정: ±는 표본 SD, 시드 5개) ----
n = 5
welch = {}
for h in HANDS:
    for metric, ia, ib in (("GMC", 0, 1), ("LMC", 2, 3)):
        a_m, a_s = ANY_V2[h][ia], ANY_V2[h][ib]
        g_m, g_s = GEORT[h][ia], GEORT[h][ib]
        t, p = stats.ttest_ind_from_stats(a_m, a_s, n, g_m, g_s, n, equal_var=False)
        welch[f"{h}_{metric}"] = {"diff": round(a_m - g_m, 1), "t": round(float(t), 2), "p": float(f"{p:.3g}")}
res["V3_welch_any_vs_geort_n5_assumed"] = welch
res["V3_not_significant_cells_p>=0.05"] = [k for k, v in welch.items() if v["p"] >= 0.05]
res["V3_sd_summary"] = {
    "any_v2_mean_of_sd": round(float(np.mean([ANY_V2[h][1] for h in HANDS] + [ANY_V2[h][3] for h in HANDS])), 3),
    "geort_mean_of_sd": round(float(np.mean([GEORT[h][1] for h in HANDS] + [GEORT[h][3] for h in HANDS])), 3),
    "geort_max_sd": 11.2,
    "any_v2_max_sd": 0.3,
}
# 결정적 기준선(Optimization)과는 SD가 없어 검정 불가: 차이만
res["V3_opt_diff"] = {h: {"dGMC": round(ANY_V2[h][0] - OPT[h][0], 1), "dLMC": round(ANY_V2[h][2] - OPT[h][1], 1)} for h in HANDS}

# ---- V4 표 4 (실로봇) ----
methods = ["Optimization", "GeoRT", "AnyDexRT"]
time_tab = {  # Sprink, Screw, Shovel, Pick-10 (초)
    "Optimization": [29.0, 25.3, 36.4, 150.4],
    "GeoRT": [32.1, 22.8, 38.5, 220.4],
    "AnyDexRT": [10.6, 17.0, 28.0, 105.8]}
tasks = ["Sprink", "Screw", "Shovel", "Pick-10"]
rate = {"Optimization": 39.6, "GeoRT": 29.2, "AnyDexRT": 62.0}
rel = {}
for j, t in enumerate(tasks):
    best_base = min(time_tab["Optimization"][j], time_tab["GeoRT"][j])
    a = time_tab["AnyDexRT"][j]
    rel[t] = {"best_baseline": best_base, "any": a, "abs_saving_s": round(best_base - a, 1),
              "rel_saving_pct": round(100 * (best_base - a) / best_base, 1),
              "rel_vs_opt_pct": round(100 * (time_tab["Optimization"][j] - a) / time_tab["Optimization"][j], 1),
              "rel_vs_geort_pct": round(100 * (time_tab["GeoRT"][j] - a) / time_tab["GeoRT"][j], 1)}
res["V4_time_savings"] = rel
res["V4_largest_relative_saving_task"] = max(tasks, key=lambda t: rel[t]["rel_saving_pct"])
res["V4_smallest_relative_saving_task"] = min(tasks, key=lambda t: rel[t]["rel_saving_pct"])
res["V4_trials_assumed"] = {"operators": 8, "trials_per_task": 3, "episodes_per_cell": 24}


def wilson(k, n_, z=1.959964):
    p = k / n_
    den = 1 + z * z / n_
    c = (p + z * z / (2 * n_)) / den
    h = z * np.sqrt(p * (1 - p) / n_ + z * z / (4 * n_ * n_)) / den
    return 100 * (c - h), 100 * (c + h)


# 핀치 성공률 = 10/A. 가정: 24 에피소드를 합쳐 240 성공 / 총 시도 수 A_tot (정수)
pinch = {}
for m in methods:
    r = rate[m] / 100
    cands = [a for a in range(150, 2000) if abs(round(100 * 240 / a, 1) - rate[m]) < 1e-9]
    pinch[m] = {"reported_pct": rate[m], "A_total_integer_solutions": cands,
                "A_per_episode": [round(a / 24, 1) for a in cands],
                "sec_per_attempt_if_pooled": [round(time_tab[m][3] * 24 / a, 2) for a in cands]}
    a0 = cands[len(cands) // 2]
    lo, hi = wilson(240, a0)
    pinch[m]["wilson95_at_mid_A"] = [round(lo, 1), round(hi, 1)]
    pinch[m]["mid_A"] = a0
res["V5_pinch_pooled_assumed"] = pinch


def two_prop_z(k1, n1, k2, n2):
    p1, p2, p = k1 / n1, k2 / n2, (k1 + k2) / (n1 + n2)
    z = (p1 - p2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return float(z), float(2 * stats.norm.sf(abs(z)))


def pick(m):
    c = pinch[m]["A_total_integer_solutions"]
    return c[len(c) // 2]


tests = {}
for other in ("Optimization", "GeoRT"):
    z, p = two_prop_z(240, pick("AnyDexRT"), 240, pick(other))
    tests[f"AnyDexRT_vs_{other}"] = {"z": round(z, 2), "p_independent_attempts": float(f"{p:.3g}")}
# 군집(작업자 8명) 보정: 설계 효과 deff = 1 + (m-1)*rho, 작업자당 시도 수 m
for rho in (0.1, 0.3):
    for other in ("Optimization", "GeoRT"):
        nA, nO = pick("AnyDexRT"), pick(other)
        mA, mO = nA / 8, nO / 8
        defA, defO = 1 + (mA - 1) * rho, 1 + (mO - 1) * rho
        z, p = two_prop_z(240, nA, 240, nO)
        # 두 군집 표본의 분산 팽창을 평균 deff로 근사
        z_adj = z / np.sqrt((defA + defO) / 2)
        tests[f"cluster_rho{rho}_AnyDexRT_vs_{other}"] = {
            "deff": [round(defA, 1), round(defO, 1)], "z_adj": round(float(z_adj), 2),
            "p_adj": float(f"{2 * stats.norm.sf(abs(z_adj)):.3g}")}
res["V5_pinch_tests_assumed"] = tests

# 블록·ablation 문장
claim_block = {"ball": 62.0, "block_text_pp": 30.0, "block_text_rate": 92.0,
               "ball_plus_pp": round(62.0 + 30.0, 1)}
cands_block = [a for a in range(150, 600) if abs(round(100 * 240 / a, 1) - 92.0) < 1e-9]
cands_abl = [a for a in range(150, 1000) if abs(round(100 * 240 / a, 1) - 49.8) < 1e-9]
claim_block["A_total_solutions_92.0"] = cands_block
claim_block["ablation_rate_62.0-12.2"] = round(62.0 - 12.2, 1)
claim_block["A_total_solutions_49.8"] = cands_abl
a_full, a_abl = pick("AnyDexRT"), cands_abl[len(cands_abl) // 2]
z, p = two_prop_z(240, a_full, 240, a_abl)
claim_block["full_vs_noclassifier_z"] = round(z, 2)
claim_block["full_vs_noclassifier_p_independent"] = float(f"{p:.3g}")
claim_block["note"] = "분류기 제거 시행 수는 본문에 없음. 같은 24 에피소드·성공 10개 가정."
res["V6_block_and_ablation"] = claim_block

# ---- V7 분류기 지표 일관성 ----
acc, prec, rec = 0.955, 0.987, 0.923
feas = []
for pos_frac in np.linspace(0.05, 0.95, 9001):
    # 연속 근사: 양성 비율 q 에서 정확도·정밀도·재현율
    tp = rec * pos_frac
    fn = pos_frac - tp
    fp = tp * (1 - prec) / prec
    tn = (1 - pos_frac) - fp
    if tn < 0:
        continue
    a = tp + tn
    p_ = tp / (tp + fp)
    ok = (abs(round(a, 3) - acc) < 1e-9) and (abs(round(p_, 3) - prec) < 1e-9)
    # 반올림 구간 허용: 각 지표 ±0.0005 안
    ok = abs(a - acc) <= 0.0005 and abs(p_ - prec) <= 0.0005
    if ok:
        feas.append(pos_frac)
res["V7_contact_classifier"] = {
    "reported": {"acc": acc, "precision": prec, "recall": rec},
    "f1_from_reported_P_R": round(2 * prec * rec / (prec + rec), 4),
    "positive_fraction_consistent_range_(rec=0.923_exact)": [round(min(feas), 3), round(max(feas), 3)] if feas else None,
    "note": "재현율을 0.923으로 고정하고 정확도·정밀도 반올림 구간 ±0.0005에 맞는 양성 비율 범위. 테스트 크기는 논문에 없음.",
}

# ---- V8 표 2, 표 3 ----
res["V8_table2_speed"] = {
    "ms_per_step": {"Opt": round(1000 / 93.4, 2), "GeoRT": round(1000 / 281.7, 2), "AnyDexRT": round(1000 / 293.0, 2)},
    "any_over_geort_speed_pct": round(100 * (293.0 / 281.7 - 1), 1),
    "any_over_opt_speed_x": round(293.0 / 93.4, 2),
}
abl = [4.1, 83.8, 89.1, 94.1]
res["V8_table3_increments"] = {
    "steps": [round(abl[i + 1] - abl[i], 1) for i in range(3)],
    "Wuji_table1_AnyDexRT_LMC": 94.1, "match_last_row": abl[-1] == 94.1,
    "v1_last_row": 92.3, "v1_Wuji_LMC": 92.3, "v1_match": True,
    "lalign_step_v1": round(92.3 - 89.1, 1), "lalign_step_v2": round(94.1 - 89.1, 1),
}

# ---- V9 그림 판독 (근사) ----
fig5 = {  # LMC, 각도 -90,-45,0,45,90, 그림 판독
    "angles": [-90, -45, 0, 45, 90],
    "AnyDexRT": [0.93, 0.93, 0.94, 0.93, 0.94],
    "GeoRT": [0.36, 0.55, 0.75, 0.59, 0.36],
    "Opt": [0.13, 0.43, 0.74, 0.26, 0.15],
}
fig6 = {  # (GeoRT SD, AnyDexRT SD), 그림 판독, 로그축
    "Inspire": (2.8e-3, 6.3e-4), "Ability": (1.0e-2, 8.2e-4), "XHand": (7.0e-3, 1.4e-3), "Wuji": (8.0e-3, 7.7e-4),
    "Allegro": (9.7e-3, 1.1e-3), "Leap": (2.1e-2, 1.4e-3), "Shadow": (8.0e-3, 9.2e-4)}
ratios = {h: round(v[0] / v[1], 1) for h, v in fig6.items()}
res["V9_fig5_read_values"] = fig5
res["V9_fig5_drop_at_pm90"] = {
    "GeoRT_drop": round(fig5["GeoRT"][2] - fig5["GeoRT"][0], 2),
    "Opt_drop": round(fig5["Opt"][2] - fig5["Opt"][0], 2),
    "AnyDexRT_range": round(max(fig5["AnyDexRT"]) - min(fig5["AnyDexRT"]), 2)}
res["V9_fig6_read_ratio_geort_over_any"] = ratios
res["V9_fig6_ratio_range_orders_of_magnitude"] = [round(float(np.log10(min(ratios.values()))), 2),
                                                  round(float(np.log10(max(ratios.values()))), 2)]

# ---- V10 수집 시간 하한 (코드 기본값, 인간 수집 부분의 sleep 합) ----
res["V10_collection_time_lower_bound_code_defaults"] = {
    "motion_5_fingers_x_6000_frames_x_0.01s": 5 * 6000 * 0.01,
    "contact_4_pairs_x_3000_frames_x_0.01s": 4 * 3000 * 0.01,
    "anchor_frames_per_sleep_0.1s": 0.1,
    "paper_fig1_label": "One-Time Collection <2 min (Anchors 옆 표기)",
    "note": "sleep 합계만의 하한. 글러브 읽기 시간은 포함하지 않음. 앵커 3종은 반복 횟수와 q 입력 시점에 따라 다름.",
}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT)
for k in ("V1_table1_average", "V1_text_claim", "V2_summary", "V3_not_significant_cells_p>=0.05", "V4_smallest_relative_saving_task",
          "V5_pinch_tests_assumed", "V6_block_and_ablation", "V7_contact_classifier", "V8_table3_increments",
          "V9_fig6_read_ratio_geort_over_any"):
    print(k, json.dumps(res[k], ensure_ascii=False))
