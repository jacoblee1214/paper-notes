"""BiDexGrasp (arXiv:2604.06589 v2) 논문 표 수치를 다시 계산하고 일관성을 검사한다.

입력: 논문 v1과 v2의 표와 본문 값을 아래에 그대로 옮겨 적었다(출처는 주석). 외부 데이터는 쓰지 않는다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 표 2의 SUC 분모: 합성 비교는 방법당 60,000 파지(12,000 인스턴스 x 5, 본문 6.1)다. 데이터셋별 표는 절반씩(30,000)이라고 가정했다.
- 표 3 분모: 논문에 없다. 60,000 또는 30,000 파지를 가정한 민감도로 보고한다.
- 표 4, 5, 7, 8 분모: 논문에 없다. 시험 파지 139,956개(본문 6.1)를 쓰는 4 배치와, 절제 설정(물체당 자세 1개)을 가정한 15,807개를 둘 다 둔다.
- 실로봇 분모: 본문의 260회와 "66.0%, 76.7%, 74.6%"를 정수 비로 풀어 구한다(정수 해가 하나인지 확인).
"""
import itertools
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
    return c - h, c + h


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def two_prop_p(p1, p2, n1, n2):
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return float(2 * (1 - stats.norm.cdf(abs(z))))


def min_n(p1, p2, alpha=0.05, power=0.8):
    """두 비율(p1, p2)의 차이를 양측 alpha, 검정력 power로 가리는 군당 최소 표본 수(정규 근사)."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pb = (p1 + p2) / 2
    n = ((za * math.sqrt(2 * pb * (1 - pb)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p1 - p2) ** 2
    return math.ceil(n)


# ---------------------------------------------------------------- V1 표 1
t1 = {  # 이름: (물체 수, 파지 수 M, 최대 크기 cm, 크기 범위 cm)  v2 표 1, v1 표 I 동일
    "DexGraspNet": (5355, 1.32, 30, 28), "BODex": (2397, 3.08, 24, 12),
    "DHAGrasp": (802, 1.30, 30, 20), "BimanGrasp": (900, 0.15, 40, 20), "BiDexGrasp": (6351, 9.53, 80, 50),
}
V1 = {}
for k, (o, g, mx, rg) in t1.items():
    V1[k] = {"grasps_per_object": round(g * 1e6 / o, 1), "min_size_cm": mx - rg}
V1["objects_sum"] = 2397 + 3954  # v1 3.1: DexGraspNet 2,397 + Objaverse 3,954
V1["objects_sum_equals_6351"] = V1["objects_sum"] == 6351
V1["object_scale_instances"] = 6351 * 11
V1["grasps_per_object_scale_instance"] = round(9.53e6 / (6351 * 11), 1)
V1["ratio_to_next_largest_grasps"] = round(9.53 / 3.08, 2)
V1["abstract_v1_9.7M_vs_table_9.53M_pct_diff"] = round(100 * (9.7 - 9.53) / 9.53, 2)
R["V1_table1"] = V1

# ---------------------------------------------------------------- V2 표 2 비율과 본문 배율
t2 = {  # (SUC-F, SUC-L, SUC-A, S, PD, SPD, CDC, D)
    ("DGN", "Biman"): (26.80, None, None, 0.16, 1.52, 0.26, 1.87, 0.150),
    ("DGN", "Float"): (75.84, None, None, 6.73, 0.17, 0.02, 0.59, 0.190),
    ("DGN", "Table"): (77.11, 80.83, 71.34, 6.87, 0.15, 0.02, 0.52, 0.165),
    ("OBJ", "Biman"): (30.21, None, None, 0.16, 0.87, 0.23, 1.38, 0.156),
    ("OBJ", "Float"): (70.82, None, None, 4.55, 0.20, 0.04, 0.67, 0.180),
    ("OBJ", "Table"): (62.41, 70.41, 62.40, 5.10, 0.20, 0.05, 0.67, 0.158),
}
V2 = {}
for ds in ("DGN", "OBJ"):
    b = t2[(ds, "Biman")]
    for mode in ("Float", "Table"):
        o = t2[(ds, mode)]
        V2[f"{ds}_{mode}"] = {
            "SUC-F_ratio": round(o[0] / b[0], 3), "S_ratio": round(o[3] / b[3], 1),
            "PD_ratio_biman_over_ours": round(b[4] / o[4], 1), "SPD_ratio": round(b[5] / o[5], 1),
            "CDC_ratio": round(b[6] / o[6], 2),
        }
    V2[f"{ds}_Table"]["SUC-L_over_biman_SUCF"] = round(t2[(ds, "Table")][1] / b[0], 2)
V2["text_claims"] = {"intro_ratio": "over 2.8x", "intro_speed": "30x (abstract/intro v1, v2)", "sec6.4_speed": "40x (v1, v2)"}
V2["dgn_both_ge_2.8"] = V2["DGN_Float"]["SUC-F_ratio"] >= 2.8 and V2["DGN_Table"]["SUC-F_ratio"] >= 2.8
V2["obj_both_ge_2.8"] = V2["OBJ_Float"]["SUC-F_ratio"] >= 2.8 and V2["OBJ_Table"]["SUC-F_ratio"] >= 2.8
V2["speed_ge_40_all"] = all(V2[k]["S_ratio"] >= 40 for k in ("DGN_Float", "DGN_Table", "OBJ_Float", "OBJ_Table"))
V2["speed_ge_30_all"] = all(V2[k]["S_ratio"] >= 30 for k in ("DGN_Float", "DGN_Table", "OBJ_Float", "OBJ_Table"))
V2["D_note"] = "D는 첫 주성분 설명 분산 비율이고 화살표는 낮을수록 좋음. 표 2에서 BimanGrasp(0.150, 0.156)이 Ours(0.158-0.190)보다 낮다"
R["V2_table2_ratios"] = V2

# ---------------------------------------------------------------- V3 SGS = S x SUC-L ?
t3 = {  # (SUC-L, SUC-A, SGS, PD, SPD)  v2 1자리, v1 2자리
    "Baseline": ((28.33, 12.89, 2.06, 0.18, 0.08), (28.3, 12.9, 2.1, 0.18, 0.08)),
    "+R": ((44.43, 39.79, 2.80, 0.13, 0.03), (44.4, 39.8, 2.8, 0.13, 0.03)),
    "+R+Dc": ((57.66, 49.52, 3.96, 0.15, 0.03), (57.7, 49.5, 4.0, 0.15, 0.03)),
    "+R+Dc+Ps": ((80.83, 71.34, 5.55, 0.15, 0.03), (80.8, 71.3, 5.6, 0.15, 0.03)),
}
V3 = {"implied_S_from_SGS_over_SUCL": {}, "rounding_consistent_v1_v2": True, "increments": {}}
prev = None
for k, (v1, v2) in t3.items():
    V3["implied_S_from_SGS_over_SUCL"][k] = round(v1[2] / (v1[0] / 100), 2)
    for a, b in zip(v1, v2):
        dec = 2 if b in (0.18, 0.13, 0.15, 0.08, 0.03) else 1
        if abs(math.floor(a * 10 ** dec + 0.5 + 1e-9) / 10 ** dec - b) > 1e-9:
            V3["rounding_consistent_v1_v2"] = False
    if prev:
        V3["increments"][k] = {"dSUC-L": round(v1[0] - prev[0], 2), "dSUC-A": round(v1[1] - prev[1], 2),
                               "dSGS": round(v1[2] - prev[2], 2)}
    prev = v1
V3["table3_last_row_equals_table2_DGN_Table"] = {
    "SUC-L": (80.83, t2[("DGN", "Table")][1]), "SUC-A": (71.34, t2[("DGN", "Table")][2]),
    "PD": (0.15, t2[("DGN", "Table")][4]), "SPD": (0.03, t2[("DGN", "Table")][5]),
    "S_implied": V3["implied_S_from_SGS_over_SUCL"]["+R+Dc+Ps"], "S_table2": t2[("DGN", "Table")][3]}
V3["table3_last_row_vs_obj_table"] = {"SUC-L_obj": t2[("OBJ", "Table")][1], "mean_of_two_datasets": round((80.83 + 70.41) / 2, 2)}
V3["SGS_over_S_vs_baseline_speed_claim"] = "Baseline SGS 2.06 -> 5.55 (x%.2f); implied raw S 7.27 -> 6.87 (-5.5%%)" % (5.55 / 2.06)
V3["sgs_ratio_vs_bimangrasp"] = {"biman_SGS_if_SUCF": round(0.16 * 0.268, 4), "ratio": round(5.55 / (0.16 * 0.268), 0),
                                 "note": "BimanGrasp의 SGS는 논문에 없다. S 0.16과 SUC-F 26.8%의 곱으로 추정(성공 정의가 다름), 속도 인용은 각 원논문 값(부록 D)"}
R["V3_table3_sgs"] = V3

# ---------------------------------------------------------------- V4 표 4, 5, 8, 7
t5 = [("none", 41.17, 2.25, 0.01, 2.87), ("GAF+SAGA", 44.68, 2.28, 0.01, 2.96),
      ("BCM+GAF", 52.74, 2.05, 0.02, 2.82), ("full", 61.93, 1.73, 0.02, 2.56)]
V4 = {"table5_deltas_vs_baseline": {n: round(s - 41.17, 2) for n, s, *_ in t5},
      "full_minus_BCM+GAF(SAGA effect given BCM,GAF)": round(61.93 - 52.74, 2),
      "BCM+GAF_minus_baseline": round(52.74 - 41.17, 2),
      "full_minus_GAF+SAGA(BCM effect given GAF,SAGA)": round(61.93 - 44.68, 2),
      "SPD_trend": [r[3] for r in t5], "CDC_trend": [r[4] for r in t5],
      "SPD_claim": "본문 6.5: GAF와 SAGA가 'reducing SPD and CDC'. 표 5의 SPD는 0.01, 0.01, 0.02, 0.02로 줄지 않는다",
      "table4_full_vs_table5_full": {"table4_SUC-L": 66.78, "table5_SUC-L": 61.93, "diff": round(66.78 - 61.93, 2),
                                    "note": "표 4는 5 epoch 전체 데이터, 표 5는 20 epoch 부분집합(물체당 자세 1개)"},
      "table4_ours_vs_DGTR_pp": round(66.78 - 41.45, 2), "table4_ours_vs_SceneDiffuser_pp": round(66.78 - 15.17, 2),
      "table5_baseline_vs_DGTR_pp": round(41.17 - 41.45, 2),
      "table8_direct_bi_view_vs_ours_pp": round(61.93 - 45.49, 2),
      "table8_direct_vs_noBCM_row": round(45.49 - 44.68, 2),
      "D_table4": {"SceneDiffuser": 0.174, "DGTR": 0.201, "Ours": 0.197,
                   "note": "D는 낮을수록 다양. Ours는 DGTR보다 낮고 SceneDiffuser보다 높다"}}
# 표 7 추론 시간: 5회, 평균 +- SD (SD로 가정)
tt = stats.ttest_ind_from_stats(1.67, 0.05, 5, 1.21, 0.03, 5, equal_var=False)
V4["table7_time"] = {"ratio": round(1.67 / 1.21, 3), "overhead_pct": round(100 * (1.67 / 1.21 - 1), 1),
                     "welch_t": round(float(tt.statistic), 1), "p": float(tt.pvalue),
                     "per_sample_ms_batch64": round(1.67 / 64 * 1000, 1),
                     "SUCL_ratio": round(61.93 / 41.17, 3)}
# 표 6 영역 품질 Q1 (x1e-2)
V4["table6_Q1_ratio"] = {"DGN": round(1.25 / 0.47, 2), "OBJ": round(1.41 / 0.79, 2),
                         "note": "물체 2,000개 x 영역 쌍 5개 = 10,000쌍(가정), 분산 미보고라 검정 불가"}
R["V4_framework_tables"] = V4

# ---------------------------------------------------------------- V5 정수성, 분해, 데이터 분할
V5 = {}
V5["synthesis_counts"] = {"12000_instances_if_1000_each_x6": 2 * 1000 * 6, "12000_if_1000_total_x6": 1000 * 6,
                          "60000_grasps_per_method": 12000 * 5,
                          "v2_text_ambiguity": "v2 본문은 'respectively'를 빼서 1,000개 합계로도 읽힌다(그러면 6,000 인스턴스)"}
V5["split_4to1"] = {"train": round(2397 * 0.8, 1), "test": round(2397 * 0.2, 1), "test_if_ceil": math.ceil(2397 * 0.2)}


def factor(n):
    f, d = {}, 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


V5["test_grasps_139956"] = {"factors": factor(139956), "triples_if_3_each": 139956 / 3,
                            "divisible_by_11_scales": (139956 / 3) % 11 == 0, "divisible_by_6_scales": (139956 / 3) % 6 == 0,
                            "per_test_object_479": round(139956 / 479, 2), "per_test_object_480": round(139956 / 480, 2),
                            "poses_per_object_scale_if_479x11": round(139956 / 3 / (479 * 11), 3),
                            "poses_per_object_scale_if_480x11": round(139956 / 3 / (480 * 11), 3)}
V5["train_grasps_3112117"] = {"per_train_object_1918": round(3112117 / 1918, 1),
                              "implied_total_DGN_grasps_if_uniform": round(3112117 / 0.8),
                              "implied_Objaverse_grasps": round(9.53e6 - 3112117 / 0.8),
                              "implied_per_Objaverse_object": round((9.53e6 - 3112117 / 0.8) / 3954, 1),
                              "implied_per_DGN_object": round(3112117 / 0.8 / 2397, 1)}
# 합성 시간 추정 (표 2의 S, SGS는 3090 한 장 기준, 부록 D)
sgs, s_ = 5.55, 6.87
V5["generation_budget"] = {
    "dataset_9.53M_hours_1gpu_at_SGS5.55": round(9.53e6 / sgs / 3600, 1),
    "dataset_hours_8gpu": round(9.53e6 / sgs / 3600 / 8, 1),
    "dataset_days_8gpu": round(9.53e6 / sgs / 3600 / 8 / 24, 2),
    "bimangrasp_0.15M_days_at_0.16_per_s": round(0.15e6 / 0.16 / 86400, 1),
    "concurrent_grasps_8gpu_x40x20": 8 * 40 * 20,
    "note": "SGS를 성공 파지/초로 보고 모두 성공만 센다고 가정. 병렬 GPU 효율 100% 가정"}
# 기준: BODex 초록 49 grasp/s (단일 손, 3090급)
V5["speed_vs_bodex_49"] = {"S_table_dgn_tabletop": 6.87, "bodex_abs_grasps_per_s": 49, "ratio": round(49 / 6.87, 1)}
R["V5_integer_and_budget"] = V5

# 실물 밀도 (밀도 2.5 kg/m^3, 본문 6.1)
rho = 2.5
V6 = {}
for L in (0.30, 0.50, 0.80):
    cube = rho * L ** 3
    sph = rho * 4 / 3 * math.pi * (L / 2) ** 3
    V6[f"size_{int(L*100)}cm"] = {"cube_kg": round(cube, 3), "sphere_kg": round(sph, 3)}
V6["note"] = "밀도 2.5 kg/m^3 가정의 속 찬 정육면체·구 질량. 물체 크기의 기준(최장 변, 대각선 등)은 논문에 없음. 평가 도구가 질량을 0.1 kg 등으로 고정하는지도 알 수 없음"
R["V6_object_mass"] = V6

# ---------------------------------------------------------------- V7 합성 비교 검정 (분모 가정)
V7 = {"assumed_n_per_cell": 30000, "cells": {}}
n = 30000
for name, p in (("BimanGrasp_DGN_SUCF", 26.80), ("Ours_Float_DGN_SUCF", 75.84), ("Ours_Table_DGN_SUCF", 77.11),
                ("BimanGrasp_OBJ_SUCF", 30.21), ("Ours_Float_OBJ_SUCF", 70.82), ("Ours_Table_OBJ_SUCF", 62.41),
                ("Ours_Table_DGN_SUCL", 80.83), ("Ours_Table_OBJ_SUCL", 70.41),
                ("Ours_Table_DGN_SUCA", 71.34), ("Ours_Table_OBJ_SUCA", 62.40)):
    k = round(p / 100 * n)
    lo, hi = wilson(k, n)
    V7["cells"][name] = {"p": p, "k": k, "wilson_lo": round(100 * lo, 2), "wilson_hi": round(100 * hi, 2),
                         "integer_exact_at_n": abs(k / n * 100 - p) < 0.005}
V7["tests"] = {
    "Ours_Table_vs_Biman_DGN_SUCF_p": two_prop_p(0.7711, 0.268, n, n),
    "Ours_Table_vs_Biman_OBJ_SUCF_p": two_prop_p(0.6241, 0.3021, n, n),
    "OBJ_Float_vs_Table_SUCF_p(70.82 vs 62.41)": two_prop_p(0.7082, 0.6241, n, n),
    "DGN_Float_vs_Table_SUCF_p(75.84 vs 77.11)": two_prop_p(0.7584, 0.7711, n, n),
    "OBJ_SUCF62.41_vs_SUCA62.40_p": two_prop_p(0.6241, 0.6240, n, n)}
V7["note"] = "SUC-F와 SUC-A는 같은 파지를 다른 시험으로 평가한 값이라 독립이 아니다. 같은 파지 위의 쌍체 비교는 원자료 없이 불가"
# 표 3 인접 행: 최소 표본 수
V7["table3_adjacent_min_n_per_arm"] = {}
rows3 = [("Baseline", 28.33), ("+R", 44.43), ("+R+Dc", 57.66), ("+R+Dc+Ps", 80.83)]
for (a, pa), (b, pb) in zip(rows3, rows3[1:]):
    V7["table3_adjacent_min_n_per_arm"][f"{a}->{b}"] = {"diff_pp": round(pb - pa, 2), "n_for_80pct_power": min_n(pa / 100, pb / 100),
                                                        "p_at_n30000": two_prop_p(pa / 100, pb / 100, 30000, 30000)}
# 표 5 / 표 8: 가정 n 두 개
V7["table5_min_n_per_arm"] = {}
for (a, pa), (b, pb) in [(("none", 41.17), ("GAF+SAGA", 44.68)), (("none", 41.17), ("BCM+GAF", 52.74)),
                         (("BCM+GAF", 52.74), ("full", 61.93)), (("Direct-bi-view", 45.49), ("full", 61.93)),
                         (("DGTR", 41.45), ("none", 41.17))]:
    V7["table5_min_n_per_arm"][f"{a}->{b}"] = {"diff_pp": round(pb - pa, 2), "n_for_80pct_power": min_n(pa / 100, pb / 100),
                                               "p_at_n15807": two_prop_p(pa / 100, pb / 100, 15807, 15807)}
V7["note_cluster"] = ("시험 물체는 479개(2,397의 20%)이고 한 물체에서 나온 파지는 서로 독립이 아니다. 차이 3.5 pp(GAF+SAGA 대 기준)를 "
                      f"가리려면 군당 {min_n(0.4117, 0.4468)}개 단위가 필요하다(독립 가정). 물체 479개만으로는 부족할 수 있다")
R["V7_sim_sensitivity"] = V7

# ---------------------------------------------------------------- V8 실로봇 정수 풀이
sol = []
for n1 in range(1, 260):
    n2 = 260 - n1
    for k1 in range(n1 + 1):
        if round(100 * k1 / n1, 1) != 66.0:
            continue
        for k2 in range(n2 + 1):
            if round(100 * k2 / n2, 1) == 76.7 and round(100 * (k1 + k2) / 260, 1) == 74.6:
                sol.append((n1, k1, n2, k2))
V8 = {"integer_solutions_weighted_avg_74.6": sol}
# 단순 평균
V8["simple_mean_66.0_76.7"] = round((66.0 + 76.7) / 2, 2)
# 가중이 아닌 경우: 76.7, 66.0 각각의 정수 비율 (n1+n2=260, 총 성공 수와 무관)
alt = []
for n1 in range(1, 260):
    n2 = 260 - n1
    ks1 = [k for k in range(n1 + 1) if round(100 * k / n1, 1) == 66.0]
    ks2 = [k for k in range(n2 + 1) if round(100 * k / n2, 1) == 76.7]
    if ks1 and ks2:
        alt.append((n1, n2))
V8["splits_where_both_percentages_integer_feasible"] = alt
V8["v1_tableVII"] = {"cells_percent": [100, 100, 100, 66.7, 50.0, 100, 83.3, 66.7, 66.7, 33.3],
                     "per_cell_trials": 6, "cells": 10}
succ = [round(p * 6 / 100) for p in V8["v1_tableVII"]["cells_percent"]]
V8["v1_tableVII"]["successes"] = succ
V8["v1_tableVII"]["total"] = f"{sum(succ)}/60 = {100*sum(succ)/60:.2f}%"
V8["v1_vs_v2"] = {"v1_full_pc_mesh_46_60": round(100 * 46 / 60, 2), "v2_full_pc_pct": 76.7}
if sol:
    n1, k1, n2, k2 = sol[0]
    lo1, hi1 = wilson(k1, n1)
    lo2, hi2 = wilson(k2, n2)
    lo3, hi3 = wilson(k1 + k2, 260)
    V8["unique_solution"] = {"single_view": [k1, n1, round(100 * lo1, 1), round(100 * hi1, 1)],
                             "full_pc": [k2, n2, round(100 * lo2, 1), round(100 * hi2, 1)],
                             "total": [k1 + k2, 260, round(100 * lo3, 1), round(100 * hi3, 1)],
                             "fisher_single_vs_full_p": fisher(k1, n1, k2, n2),
                             "diff_pp": round(100 * (k2 / n2 - k1 / n1), 1)}
    # 대안 구성: 전체 포인트 클라우드 60회(v1과 같은 46/60), 단일 시점 200회
    V8["alt_full60_single200"] = {"single": "132/200=66.0%", "full": "46/60=76.7%", "total": "178/260 = %.1f%%" % (100 * 178 / 260)}
V8["trials_per_object"] = round(260 / 30, 2)
V8["real_world_vs_sim_gap"] = {"sim_SUC-L_table4": 66.78, "sim_SUC-A_DGN": 71.34, "real_avg": 74.6}
R["V8_real_world"] = V8

# ---------------------------------------------------------------- V9 그림 6(b) 뷰 점수와 top-k softmax 표집
fig6b_primary = [0.12, 0.16, 0.15, 0.12, 0.25, 0.11, 0.20, 0.21, 0.10, 0.46, 0.10, 0.13, 0.54, 0.07, 0.25, 0.14]
fig6b_second = [0.01, 0.00, 0.00, 0.06, 0.00, 0.04, 0.00, 0.00, 0.19, 0.01, 0.02, 0.46, 0.00, 0.02, 0.02, 0.00]


def topk_softmax(scores, k=5, tau=0.1):
    idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    z = np.array([scores[i] for i in idx]) / tau
    p = np.exp(z - z.max())
    p /= p.sum()
    return {int(i + 1): round(float(x), 3) for i, x in zip(idx, p)}


V9 = {"primary_sum": round(sum(fig6b_primary), 2), "second_sum": round(sum(fig6b_second), 2),
      "primary_top5_p": topk_softmax(fig6b_primary), "second_top5_p": topk_softmax(fig6b_second),
      "second_tie_at_0.02": [i + 1 for i, v in enumerate(fig6b_second) if v == 0.02],
      "joint_p_13_then_12": round(topk_softmax(fig6b_primary)[13] * topk_softmax(fig6b_second)[12], 3),
      "note": "그림 6(b) 점수는 합이 1이 아니다(SmoothL1 회귀 출력). 값은 그림에서 읽은 값. top-5 + 온도 0.1에서 최상위 뷰 표집 확률이 높다"}
R["V9_view_sampling"] = V9

# ---------------------------------------------------------------- V10 버전 차이 수치 표
R["V10_v1_v2_numbers"] = {
    "abstract_grasps": ("9.7 million", "9.53 million"),
    "conclusion_grasps": ("9.7 million", "9.53M"),
    "baseline_bimangrasp_SUCF": (26.8, 26.80), "view_loss": ("L2 (식 9, 10)", "마스크 SmoothL1 (식 17, 18)"),
    "region_selection_steps": ("4단계 (오목 영역 제외 포함)", "3단계 (오목 영역 제외 문장 없음)"),
    "optimizer_batch_lr": ("Adam, batch 128, wd 1e-4, lr 1e-4 -> 1e-5 cosine", "본문과 부록 C에 없음"),
    "loss_weights": ("없음", "부록 C.2에 λ 값 있음 (10, 1, 15, 0.25, 1, 100, 50, 50)"),
    "real_world": ("표 VII: 10개 물체 x 6회, 메시 재구성만, 46/60 = 76.7%", "본문: 30개 물체 260회, 단일 시점 66.0%, 전체 점군 76.7%, 평균 74.6%"),
    "real_world_arms": ("Piper + BrainCo, G1 + Inspire", "Unitree G1, Piper, Nero + Inspire, BrainCo"),
    "authors": ("10명", "13명 (Zhizhao Liang, Xiaoyi Fan, Chengyi Xing 추가)"),
    "bimangrasp_description": ("the only publicly available open-source", "a representative open-source"),
    "table_precision": ("표 3, 4, 5 소수 둘째 자리", "소수 첫째 자리 (표 3, 4, 5), 값은 반올림과 일치")}

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

# ------------------------------------------------------------ 요약
print("V1 객체 합", V1["objects_sum"], V1["objects_sum_equals_6351"], "물체-크기 인스턴스", V1["object_scale_instances"], "파지/인스턴스", V1["grasps_per_object_scale_instance"])
print("V2", {k: V2[k] for k in ("DGN_Float", "DGN_Table", "OBJ_Float", "OBJ_Table")})
print("V2 판정", V2["dgn_both_ge_2.8"], V2["obj_both_ge_2.8"], V2["speed_ge_40_all"], V2["speed_ge_30_all"])
print("V3 implied S", V3["implied_S_from_SGS_over_SUCL"])
print("V4", V4["table5_deltas_vs_baseline"], V4["table7_time"])
print("V5", V5["test_grasps_139956"], V5["generation_budget"])
print("V6", V6)
print("V7 tests", V7["tests"])
print("V7 min n", V7["table3_adjacent_min_n_per_arm"], V7["table5_min_n_per_arm"])
print("V8", V8["integer_solutions_weighted_avg_74.6"], V8.get("unique_solution"), V8["v1_tableVII"]["total"], V8["splits_where_both_percentages_integer_feasible"])
print("V9", V9)
