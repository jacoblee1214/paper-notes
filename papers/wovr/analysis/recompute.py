"""WoVR 논문 표 수치 재계산, 일관성 검사, 유의성 검정 (numpy, scipy; 수 초).

실행: python3 analysis/recompute.py  → analysis/recompute_results.json
입력: 논문 표 값을 아래 상수로 옮겨 적었다 (paper/paper.txt, arXiv v3 기준).
가정: LIBERO 평가 에피소드 수 n은 논문에 적혀 있지 않다. 공식 RLinf 설정(eval total_num_envs=496,
      LIBERO 스위트당 초기 상태 50개 x 10과제 = 500)을 근거로 n=500을 기본 가정으로 쓴다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
RESULTS = {}

# ---- 표 2: LIBERO SR (%) ----
SUITES = ["Spatial", "Object", "Goal", "Long"]
T2 = {
    "one_traj": {
        "OpenVLA-OFT": [63.6, 36.4, 48.2, 13.8, 40.5],
        "GRPO(online)": [66.6, 45.2, 52.2, 14.6, 44.6],
        "WMPO": [67.8, 65.4, 56.6, 13.8, 50.9],
        "WoVR": [84.2, 80.8, 77.4, 35.8, 69.5],
    },
    "full_traj": {
        "OpenVLA-OFT": [93.6, 83.0, 90.0, 85.6, 88.1],
        "GRPO(online)": [94.6, 86.2, 92.2, 85.8, 89.7],
        "WMPO": [95.0, 94.8, 92.8, 87.0, 92.4],
        "WoVR": [98.8, 98.8, 94.8, 91.4, 96.0],
    },
}
N_SUITE = 500  # 가정 (위 설명)

# 1) 평균 재계산과 보고값 비교
mean_check = {}
for setting, methods in T2.items():
    for m, v in methods.items():
        calc = float(np.mean(v[:4]))
        mean_check[f"{setting}/{m}"] = {"reported_avg": v[4], "recomputed_avg": round(calc, 3),
                                        "match_0.05": abs(calc - v[4]) <= 0.05}
RESULTS["table2_mean_check"] = mean_check

# 2) 표 2 Δ 행 확인 (WoVR - OFT, 평균 기준)
RESULTS["table2_delta"] = {
    s: {"one_traj_vs_OFT": round(T2["one_traj"]["WoVR"][i] - T2["one_traj"]["OpenVLA-OFT"][i], 1),
        "one_traj_vs_WMPO": round(T2["one_traj"]["WoVR"][i] - T2["one_traj"]["WMPO"][i], 1),
        "full_vs_WMPO": round(T2["full_traj"]["WoVR"][i] - T2["full_traj"]["WMPO"][i], 1)}
    for i, s in enumerate(SUITES)
}
RESULTS["table2_delta"]["avg"] = {
    "one_traj_vs_OFT": round(69.5 - 40.5, 1), "one_traj_vs_WMPO": round(69.5 - 50.9, 1),
    "full_vs_WMPO": round(96.0 - 92.4, 1)}


def wilson(p, n, z=1.96):
    """Wilson 95% 구간 (p는 0–1)."""
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def two_prop_z(p1, n1, p2, n2):
    """두 비율 차이의 양측 z검정 (풀링 분산)."""
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else float("nan")
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


# 3) LIBERO 스위트별 two-proportion 검정 (n=500 가정)
lib = {}
for i, s in enumerate(SUITES):
    w = T2["one_traj"]["WoVR"][i] / 100
    m = T2["one_traj"]["WMPO"][i] / 100
    z, p = two_prop_z(w, N_SUITE, m, N_SUITE)
    lo, hi = wilson(w, N_SUITE)
    lib[s] = {"WoVR_one_traj_wilson95": [round(100 * lo, 1), round(100 * hi, 1)],
              "WoVR_vs_WMPO_one_traj_z": round(z, 2), "p": float(f"{p:.2g}")}
RESULTS["libero_one_traj_suite_tests_n500"] = lib

# 4) 4개 스위트 평균 차이의 근사 검정 (스위트 독립 가정, 평균의 SE = sqrt(sum var)/4)
def avg_diff_test(a, b, n=N_SUITE):
    da = [x / 100 for x in a]
    db = [x / 100 for x in b]
    diffs = [x - y for x, y in zip(da, db)]
    d = float(np.mean(diffs))
    se = math.sqrt(sum(x * (1 - x) / n + y * (1 - y) / n for x, y in zip(da, db))) / 4
    z = d / se
    return round(100 * d, 2), round(z, 2), float(f"{2 * (1 - stats.norm.cdf(abs(z))):.2g}")

RESULTS["libero_avg_diff_one_traj_WoVR_vs_WMPO"] = dict(zip(
    ["diff_pts", "z", "p"], avg_diff_test(T2["one_traj"]["WoVR"][:4], T2["one_traj"]["WMPO"][:4])))
RESULTS["libero_avg_diff_full_WoVR_vs_WMPO"] = dict(zip(
    ["diff_pts", "z", "p"], avg_diff_test(T2["full_traj"]["WoVR"][:4], T2["full_traj"]["WMPO"][:4])))

# 5) 표 3 실로봇 (30회/과제). 과제별 성공 횟수로 바꾸고 정수 검사
T3 = {
    "Franka": {"tasks": ["Pick Banana", "Pick Bread", "Open Drawer"],
               "OFT": [36.7, 70.0, 46.7], "WMPO": [56.7, 76.7, 60.0], "WoVR": [86.7, 90.0, 63.3]},
    "AgileX": {"tasks": ["Fold Towel", "Pick Cube", "Pick Tomato"],
               "OFT": [10.0, 23.3, 13.3], "WMPO": [10.0, 26.7, 20.0], "WoVR": [20.0, 33.3, 33.3]},
}
N_RW = 30
rw = {}
for plat, d in T3.items():
    counts = {}
    for m in ["OFT", "WMPO", "WoVR"]:
        k = [round(p * N_RW / 100) for p in d[m]]
        integer_ok = all(abs(kk * 100 / N_RW - p) < 0.05 for kk, p in zip(k, d[m]))
        counts[m] = {"per_task_success": k, "integer_ok": integer_ok, "pooled": f"{sum(k)}/90"}
    tests = {}
    for a, b in [("WoVR", "OFT"), ("WoVR", "WMPO"), ("WMPO", "OFT")]:
        ka, kb = sum(counts[a]["per_task_success"]), sum(counts[b]["per_task_success"])
        table = [[ka, 90 - ka], [kb, 90 - kb]]
        _, p_f = stats.fisher_exact(table)
        z, p_z = two_prop_z(ka / 90, 90, kb / 90, 90)
        tests[f"{a}_vs_{b}"] = {"fisher_p": float(f"{p_f:.2g}"), "z_p": float(f"{p_z:.2g}")}
    rw[plat] = {"counts": counts, "tests_pooled_90_trials": tests,
                "wilson95_WoVR_pooled": [round(100 * x, 1) for x in wilson(sum(counts["WoVR"]["per_task_success"]) / 90, 90)]}
RESULTS["table3_realworld"] = rw

# 6) 표 4 π0.5 (30회/과제)
b = [20.0, 60.0, 23.3]; w = [30.0, 86.7, 53.3]
kb = [round(x * 30 / 100) for x in b]; kw = [round(x * 30 / 100) for x in w]
_, p_f = stats.fisher_exact([[sum(kw), 90 - sum(kw)], [sum(kb), 90 - sum(kb)]])
RESULTS["table4_pi05"] = {"base_avg_reported": 34.4, "base_recomputed": round(np.mean(b), 2),
                          "woVR_avg_reported": 56.7, "woVR_recomputed": round(np.mean(w), 2),
                          "counts_base": kb, "counts_woVR": kw, "fisher_p_pooled": float(f"{p_f:.2g}")}

# 7) 표 6 절제 (LIBERO-Spatial, Object 각 n=500 가정, 두 스위트 평균)
T6 = {"WoVR_full": [84.2, 80.8], "w_o_KIR": [81.6, 77.8], "w_o_PACE_1500": [75.4, 76.2], "w_o_PACE_2500": [77.8, 77.2]}
T6_reported_avg = {"WoVR_full": 82.5, "w_o_KIR": 79.7, "w_o_PACE_1500": 75.8, "w_o_PACE_2500": 77.5}
RESULTS["table6_ablation"] = {
    "avg_check": {k: {"reported": T6_reported_avg[k], "recomputed": round(float(np.mean(v)), 2)} for k, v in T6.items()},
    "full_vs_w_o_KIR_pooled_two_suites_n1000": dict(zip(["z", "p"], two_prop_z(0.825, 1000, 0.797, 1000))),
    "full_vs_w_o_PACE_1500_pooled_two_suites_n1000": dict(zip(["z", "p"], two_prop_z(0.825, 1000, 0.758, 1000))),
    # 같은 총 데이터량(2,500 궤적) 비교: 기본 정책 2,500 (77.5) 대 1,500 + 정렬 1,000 (82.5)
    "full_vs_w_o_PACE_2500_base_same_budget_n1000": dict(zip(["z", "p"], two_prop_z(0.825, 1000, 0.775, 1000))),
}

# 8) 표 1 세계모델: 보고된 WoVR가 모든 지표와 모든 horizon에서 최선인지
T1 = {  # horizon: {method: (LPIPS, FID, FVD, FloLPIPS)}
    512: {"EVAC": (0.146, 46.528, 345.818, 0.205), "Cosmos-Predict2": (0.315, 165.862, 275.737, 0.265),
          "OpenSora": (0.105, 38.478, 89.391, 0.156), "WoVR": (0.091, 34.252, 68.011, 0.154)},
    256: {"EVAC": (0.130, 49.153, 354.983, 0.192), "Cosmos-Predict2": (0.226, 106.324, 203.853, 0.306),
          "OpenSora": (0.082, 33.577, 94.998, 0.122), "WoVR": (0.063, 24.378, 50.041, 0.102)},
    128: {"EVAC": (0.106, 44.337, 423.132, 0.166), "Cosmos-Predict2": (0.164, 77.555, 304.456, 0.281),
          "OpenSora": (0.069, 33.413, 111.643, 0.113), "WoVR": (0.047, 18.553, 39.047, 0.079)},
}
wins = {}
for h, row in T1.items():
    for j, metric in enumerate(["LPIPS", "FID", "FVD", "FloLPIPS"]):
        best = min(row, key=lambda k: row[k][j])
        wins[f"{h}/{metric}"] = best
RESULTS["table1_best_per_cell"] = wins
RESULTS["table1_woVR_best_everywhere"] = all(v == "WoVR" for v in wins.values())
# OpenSora 대비 FloLPIPS 512 (0.154 vs 0.156)는 차이가 0.002로 매우 작다
RESULTS["table1_margin_FloLPIPS_512_vs_OpenSora"] = round(0.156 - 0.154, 3)

# 9) 표 5 (LIBERO-Spatial, 24 held-out) 와 표 1 WoVR 행 비교 (같은 모델로 보임, 평가셋 다름)
T5_full = {512: (0.091, 36.687, 73.493, 0.154), 256: (0.069, 27.238, 63.948, 0.110), 128: (0.051, 20.780, 49.017, 0.081)}
RESULTS["table5_vs_table1_WoVR_rows"] = {
    str(h): {"table5_full": T5_full[h], "table1_WoVR": T1[h]["WoVR"]} for h in [512, 256, 128]
}

# 10) 본문 vs 표 불일치 확인
RESULTS["text_vs_table"] = {
    "sec4.2_full_avg_text": 95.9, "table2_full_avg": 96.0,
    "recomputed_from_row": round(float(np.mean(T2["full_traj"]["WoVR"][:4])), 3),
}

OUT.write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: RESULTS[k] for k in ["table2_mean_check", "table1_woVR_best_everywhere",
                                           "libero_avg_diff_one_traj_WoVR_vs_WMPO",
                                           "libero_avg_diff_full_WoVR_vs_WMPO", "table4_pi05",
                                           "table6_ablation", "text_vs_table"]}, ensure_ascii=False, indent=1))
print(json.dumps(RESULTS["table3_realworld"], ensure_ascii=False, indent=1)[:3000])
print(json.dumps(RESULTS["libero_one_traj_suite_tests_n500"], ensure_ascii=False, indent=1))
