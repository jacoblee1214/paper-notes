"""JEPA-VLA (arXiv:2602.11832v1) 논문 표 수치 재계산과 일관성 검사.

논문 표의 값은 paper/paper.pdf 페이지 이미지(표 1–7, 9–12)를 눈으로 읽어 아래에 옮겼다.
시행 횟수(에피소드 수)는 논문에 명시되어 있지 않으므로, 신뢰구간 계산에는 가정값을 쓰고 결과 파일에 가정을 적었다.

실행: python3 analysis/03_recompute.py  → analysis/03_recompute_results.json (CPU, 1초 미만)
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "03_recompute_results.json"
Z = 1.959964

# ---- 논문 값 (단위 %) -------------------------------------------------------
# Table 1: LIBERO, basic VLA, 1/10 데이터 (Spatial, Object, Goal, Long, Average)
T1 = {"base": [58.2, 74.8, 87.8, 25.8], "ours": [69.2, 78.2, 87.8, 41.0],
      "avg_printed": {"base": 61.65, "ours": 69.05}, "delta_printed": [11.0, 3.4, 0.0, 15.2, 7.40]}

# Table 3: LIBERO, mainstream OpenVLA-OFT (official는 원 논문 인용, OFT는 저자 재현)
T3 = {"oft_official": [96.2, 98.3, 96.2, 90.7], "oft_repro": [84.4, 91.8, 95.2, 89.8],
      "ours": [97.2, 98.0, 95.6, 94.8],
      "avg_printed": {"oft_official": 95.35, "oft_repro": 90.30, "ours": 96.40}, "delta_printed": [12.8, 6.2, 0.4, 5.0, 6.10]}

# Table 4: RoboTwin 2.0, 6개 과제 (Easy = clean, Hard = domain randomized)
T4_tasks = ["Beat Block Hammer", "Click Alarmclock", "Lift Pot", "Place Fan", "Place Phone Stand", "Stack Bowls Two"]
T4 = {"base_easy": [18, 85, 93, 25, 33, 75], "ours_easy": [65, 92, 98, 38, 60, 88],
      "base_hard": [0, 26, 18, 0, 4, 8], "ours_hard": [13, 30, 21, 0, 16, 26],
      "avg_printed": {"base_easy": 54.8, "base_hard": 9.3, "ours_easy": 73.5, "ours_hard": 17.7},
      "delta_printed": {"easy": 18.7, "hard": 8.4}}

# Table 5: 실세계 pick-and-place (Average, Layout, Light)
T5 = {"base": (50.0, 0.0, 0.0), "ours_1_5": (60.0, 33.3, 66.7), "ours_full": (80.0, 100.0, 100.0)}

# Table 2: LIBERO-plus 집계 (Camera, Robot, Language, Light, Background, Noise, Layout, Total)
DIMS = ["Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout"]
T2 = {"WorldVLA": [0.1, 27.9, 41.6, 43.7, 17.1, 10.9, 38.0], "base": [0.1, 20.8, 43.9, 26.3, 10.1, 4.1, 26.9],
      "ours": [0.4, 25.7, 54.4, 38.0, 23.9, 4.1, 32.8]}
T2_TOTAL = {"WorldVLA": 25.0, "base": 18.9, "ours": 25.6}
T2_DELTA = {"Camera": 0.3, "Robot": 4.9, "Language": 10.5, "Light": 11.7, "Background": 13.8, "Noise": 0.0,
            "Layout": 5.9, "Total": 6.7}

# Tables 9–12: LIBERO-plus, 1/10 데이터, 스위트별 (열: 7개 차원 + Total)
SUITES = ["Spatial", "Object", "Goal", "Long"]
T9_12 = {
    "Spatial": {"WorldVLA": [0.0, 44.3, 46.3, 65.1, 19.8, 11.7, 46.1, 32.5],
                "base": [0.0, 24.9, 51.5, 44.2, 10.5, 1.7, 31.4, 23.8],
                "ours": [0.8, 29.7, 53.3, 57.2, 44.2, 1.4, 44.2, 32.1]},
    "Object": {"WorldVLA": [0.0, 26.4, 57.2, 20.5, 17.3, 18.0, 53.6, 28.6],
               "base": [0.0, 25.4, 72.0, 19.5, 3.2, 5.2, 25.3, 21.7],
               "ours": [0.0, 33.0, 91.5, 35.0, 7.3, 6.4, 34.7, 29.6]},
    "Goal": {"WorldVLA": [0.3, 30.6, 42.2, 68.8, 30.3, 13.5, 47.4, 31.8],
             "base": [0.5, 25.4, 29.8, 35.1, 19.6, 9.0, 30.8, 21.1],
             "ours": [0.5, 28.9, 39.5, 40.5, 36.3, 7.7, 36.0, 26.2]},
    "Long": {"WorldVLA": [0.0, 12.2, 20.6, 20.4, 1.7, 1.6, 4.4, 8.2],
             "base": [0.0, 7.4, 22.5, 6.2, 7.3, 0.7, 19.9, 8.7],
             "ours": [0.5, 11.5, 33.2, 19.3, 8.0, 0.9, 16.4, 12.1]},
}
T9_12_DELTA_PRINTED = {  # 논문 Δ 행 (Total 포함)
    "Spatial": [0.8, 4.8, 1.8, 13.0, 33.7, -0.3, 12.8, 8.3],
    "Object": [0.0, 7.6, 19.5, 15.5, 4.1, 1.2, 9.4, 7.9],
    "Goal": [0.0, 3.5, 9.7, 5.4, 16.7, -1.3, 5.2, 5.1],
    "Long": [0.5, 4.1, 10.7, 13.1, 0.7, 0.2, -3.5, 3.4]}

# Table 6: CortexBench (MetaWorld 성공률 %, DMControl 보상)
T6_MW = {"VC-1": [88.0, 84.0, 80.0, 100.0, 92.0], "V-JEPA 2": [96.0, 80.0, 84.0, 100.0, 92.0]}
T6_MW_AVG = {"VC-1": 88.8, "V-JEPA 2": 90.4, "CLIP": 75.5}
T6_DMC = {"VC-1": [811, 338, 836, 155, 540], "V-JEPA 2": [825, 436, 821, 156, 681]}
T6_DMC_AVG = {"VC-1": 536.0, "V-JEPA 2": 583.8}

# Table 7: LIBERO-Long 표현 교체 (본문 표기 그대로)
T7 = [("BASELINE", 25.8), ("BASELINE+DINOv2", 26.0), ("BASELINE+DINOv2", 31.2),
      ("BASELINE+SigLIP", 33.6), ("BASELINE+V-JEPA 2", 41.0)]


# ---- 도구 ---------------------------------------------------------------------
def wilson(p, n, z=Z):
    """이항 비율의 Wilson 구간 (p는 0–1)."""
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def newcombe_diff(p1, n1, p2, n2):
    """두 독립 비율 차이 p2-p1의 Newcombe(1998) 구간. 입력은 0–1."""
    l1, u1 = wilson(p1, n1)
    l2, u2 = wilson(p2, n2)
    d = p2 - p1
    lo = d - np.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    hi = d + np.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return d, lo, hi


def pct(x):
    return np.asarray(x, dtype=float) / 100.0


results = {"assumptions": {
    "libero_episodes_per_suite": 500,
    "libero_episodes_note": "LIBERO 표준 프로토콜(과제당 50회 × 10과제)을 가정. 논문에는 평가 에피소드 수가 없다.",
    "robotwin_episodes_per_cell": [50, 100],
    "robotwin_episodes_note": "RoboTwin 2.0 과제당 평가 횟수를 논문에서 찾지 못해 50과 100을 가정.",
    "seed_caveat": "모든 비교는 학습 시드 1개로 보인다. 아래 구간은 에피소드 표본 변동만 반영하므로 학습 시드 변동을 빼고 있다."}}

# 1) 평균 재계산 (Table 1, 3, 4, 5)
def mean(x):
    return float(np.mean(x))

t1 = {k: mean(T1[k]) for k in ("base", "ours")}
t3 = {k: mean(T3[k]) for k in ("oft_official", "oft_repro", "ours")}
t4 = {k: mean(T4[k]) for k in ("base_easy", "base_hard", "ours_easy", "ours_hard")}
results["mean_checks"] = {
    "T1_avg_recomputed": t1, "T1_avg_printed": T1["avg_printed"],
    "T1_delta_recomputed": {"avg": t1["ours"] - t1["base"],
                            "per_suite": [b - a for a, b in zip(T1["base"], T1["ours"])]},
    "T3_avg_recomputed": t3, "T3_avg_printed": T3["avg_printed"],
    "T3_ours_minus_official_per_suite": [round(a - b, 1) for a, b in zip(T3["ours"], T3["oft_official"])],
    "T3_ours_minus_official_avg": round(t3["ours"] - t3["oft_official"], 2),
    "T4_avg_recomputed": t4, "T4_avg_printed": T4["avg_printed"],
    "T4_delta_recomputed": {"easy": round(t4["ours_easy"] - t4["base_easy"], 2),
                            "hard": round(t4["ours_hard"] - t4["base_hard"], 2)},
    "T4_task_delta": {"easy": [b - a for a, b in zip(T4["base_easy"], T4["ours_easy"])],
                      "hard": [b - a for a, b in zip(T4["base_hard"], T4["ours_hard"])]},
}

# Table 5: 평균 열이 하위 열과 맞는지 (두 조건 평균 vs 인쇄된 Average)
t5_check = {}
for k, (avg, lay, lig) in T5.items():
    t5_check[k] = {"average_printed": avg, "mean_of_layout_and_light": round((lay + lig) / 2, 2),
                   "gap": round(avg - (lay + lig) / 2, 2)}
# 값이 1/n 의 정수배인지 (n=10 이면 33.3이 불가능)
def consistent_n(values, candidates=range(2, 31)):
    ok = []
    for n in candidates:
        if all(abs(v * n / 100 - round(v * n / 100)) < 0.06 for v in values):
            ok.append(n)
    return ok
results["T5"] = {"avg_check": t5_check,
                 "trial_counts_consistent_with_layout_light_values": consistent_n([33.3, 66.7, 0.0, 100.0]),
                 "n10_consistent": 10 in consistent_n([33.3, 66.7, 0.0, 100.0]),
                 "note": "논문 부록 E는 모델당 10회 시행이라고 쓰지만 33.3과 66.7은 10회로는 나오지 않는다(1/3 단위)."}

# Fisher 정확검정: 실세계 layout·light, n=3 가정(10회 주장과 불일치하므로 민감도용)
fish = {}
for n in (3, 10):
    # baseline 0%, ours-1/5 layout 33.3%, light 66.7% -> n=3 에서 1/3, 2/3
    if n == 3:
        pairs = {"layout": (0, 1), "light": (0, 2)}
    else:
        pairs = {"layout": (0, 3), "light": (0, 7)}  # 10회일 때 가장 가까운 정수
    fish[f"n{n}"] = {}
    for name, (kb, ko) in pairs.items():
        tab = [[kb, n - kb], [ko, n - ko]]
        fish[f"n{n}"][name] = float(stats.fisher_exact(tab)[1])
results["T5_fisher_p_two_sided"] = fish

# 2) Table 1, 3: 스위트별 Newcombe 구간 (LIBERO 스위트당 500회 가정)
n = results["assumptions"]["libero_episodes_per_suite"]
def ci_table(base, new, n):
    rows = []
    for name, a, b in zip(SUITES, base, new):
        d, lo, hi = newcombe_diff(a / 100, n, b / 100, n)
        rows.append({"suite": name, "base": a, "new": b, "delta_pts": round(100 * d, 2),
                     "ci95_pts": [round(100 * lo, 2), round(100 * hi, 2)]})
    return rows
results["T1_ci"] = ci_table(T1["base"], T1["ours"], n)
results["T3_ci_vs_repro"] = ci_table(T3["oft_repro"], T3["ours"], n)
results["T3_ci_vs_official"] = ci_table(T3["oft_official"], T3["ours"], n)
results["T1_sign_test_suites_positive"] = float(stats.binomtest(4, 4, 0.5, alternative="greater").pvalue)
results["T3_sign_test_vs_repro_4of4"] = float(stats.binomtest(4, 4, 0.5, alternative="greater").pvalue)

# 3) Table 4 (RoboTwin): 과제별 부호 검정 (횟수 가정 불필요)
ed = [b - a for a, b in zip(T4["base_easy"], T4["ours_easy"])]
hd = [b - a for a, b in zip(T4["base_hard"], T4["ours_hard"])]
results["T4_sign_test"] = {
    "easy_all_positive_6of6_one_sided_p": float(stats.binomtest(6, 6, 0.5, alternative="greater").pvalue),
    "hard_positive_5_of_5_nonzero_two_sided_p": float(stats.binomtest(5, 5, 0.5, alternative="two-sided").pvalue),
    "note": "과제 6개(논문은 50개 중 6개 선택 이유를 밝히지 않음). 과제 내 시드 변동은 반영하지 않음.",
}
results["T4_ci_pooled_assumed_n"] = {}
for nn in results["assumptions"]["robotwin_episodes_per_cell"]:
    # 6과제의 평균 성공률 차이를 과제 단위 표준오차로 근사 (과제 간 분산 + 표본 분산)
    diffs = np.array(ed) / 100
    p_b = np.array(T4["base_easy"]) / 100
    p_o = np.array(T4["ours_easy"]) / 100
    var_samp = (p_b * (1 - p_b) + p_o * (1 - p_o)) / nn
    se = np.sqrt(np.sum(var_samp) / 36 + np.var(diffs, ddof=1) / 6)
    d = diffs.mean()
    results["T4_ci_pooled_assumed_n"][str(nn)] = {"easy_delta_pts": round(100 * d, 2),
                                                  "ci95_pts_approx": [round(100 * (d - Z * se), 2), round(100 * (d + Z * se), 2)]}

# 4) Table 2 집계 검사: 스위트 평균과 인쇄된 값
agg = {}
for m in ("WorldVLA", "base", "ours"):
    per_dim_mean = [mean([T9_12[s][m][i] for s in SUITES]) for i in range(7)]
    per_suite_total = [T9_12[s][m][7] for s in SUITES]
    agg[m] = {
        "per_dim_mean_over_suites": [round(x, 2) for x in per_dim_mean],
        "per_dim_printed": T2[m],
        "mean_of_7_dims_from_table2": round(mean(T2[m]), 2),
        "table2_total_printed": T2_TOTAL[m],
        "mean_of_suite_totals": round(mean(per_suite_total), 2),
    }
results["T2_aggregation_check"] = agg
results["T2_delta_check"] = {d: round(T2["ours"][i] - T2["base"][i], 2) for i, d in enumerate(DIMS)}
results["T2_delta_printed"] = T2_DELTA

# 5) 스위트 단위 비교: 논문 Δ 행 재계산과 WorldVLA 대비
suite_rows = {}
for s in SUITES:
    v = T9_12[s]
    rec = [round(o - b, 1) for o, b in zip(v["ours"], v["base"])]
    suite_rows[s] = {"delta_recomputed": rec, "delta_printed": T9_12_DELTA_PRINTED[s],
                     "delta_match": rec == T9_12_DELTA_PRINTED[s],
                     "ours_minus_worldvla_total": round(v["ours"][7] - v["WorldVLA"][7], 1)}
results["T9_12_suites"] = suite_rows
results["T9_12_raw_total_order"] = {"columns": DIMS + ["Total"], "values": T9_12}
results["T9_12_total_mean_over_suites"] = {m: round(mean([T9_12[s][m][7] for s in SUITES]), 2)
                                          for m in ("WorldVLA", "base", "ours")}
results["T9_12_ours_beats_worldvla_in_suites"] = sum(
    1 for s in SUITES if T9_12[s]["ours"][7] > T9_12[s]["WorldVLA"][7])
results["T9_12_sign_test_ours_gt_base_4of4_two_sided_p"] = float(
    stats.binomtest(4, 4, 0.5, alternative="two-sided").pvalue)

# 6) Table 6: CortexBench
mw_diff = [b - a for a, b in zip(T6_MW["VC-1"], T6_MW["V-JEPA 2"])]
dmc_diff = [b - a for a, b in zip(T6_DMC["VC-1"], T6_DMC["V-JEPA 2"])]
results["T6"] = {
    "mw_avg_recomputed": {k: round(mean(v), 2) for k, v in T6_MW.items()},
    "mw_avg_printed": T6_MW_AVG,
    "mw_diff_per_task": mw_diff,
    "dmc_avg_recomputed": {k: round(mean(v), 2) for k, v in T6_DMC.items()},
    "dmc_avg_printed": T6_DMC_AVG,
    "dmc_diff_per_task": dmc_diff,
    "dmc_relative_gain_pct": round(100 * (mean(T6_DMC["V-JEPA 2"]) / mean(T6_DMC["VC-1"]) - 1), 2),
    "mw_wins_ties_losses": [sum(d > 0 for d in mw_diff), sum(d == 0 for d in mw_diff), sum(d < 0 for d in mw_diff)],
    "dmc_wins_ties_losses": [sum(d > 0 for d in dmc_diff), sum(d == 0 for d in dmc_diff), sum(d < 0 for d in dmc_diff)],
}

# 7) Table 7 자체 일관성
t7_vals = [v for _, v in T7]
results["T7"] = {"rows": T7, "negative_gain_rows": [n_ for n_, v in T7 if v < 25.8],
                 "duplicate_label": "BASELINE+DINOv2가 두 번 나옴(26.0, 31.2)"}

OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))

# 콘솔 요약
print("T1 avg", t1, "printed", T1["avg_printed"])
print("T3 avg", {k: round(v, 2) for k, v in t3.items()}, "printed", T3["avg_printed"])
print("T4 avg", t4, "printed", T4["avg_printed"])
print("T5 checks", t5_check)
print("T5 n consistent", results["T5"]["trial_counts_consistent_with_layout_light_values"])
print("T5 fisher", fish)
print("T2 agg", json.dumps(agg, ensure_ascii=False))
print("suite rows", json.dumps(suite_rows, ensure_ascii=False))
print("T9-12 mean over suites", results["T9_12_total_mean_over_suites"],
      "ours beats WorldVLA in", results["T9_12_ours_beats_worldvla_in_suites"], "suites")
print("T6", json.dumps(results["T6"]))
print("wrote", OUT)
