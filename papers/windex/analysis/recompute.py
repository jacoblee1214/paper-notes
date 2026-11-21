"""WinDEX (arXiv:2611.21143 v1) 논문 보고 수치의 재계산과 검정. 표준 라이브러리만 사용한다.

실행: python3 analysis/recompute.py   → analysis/recompute_results.json (수 초)
입력은 모두 논문 본문과 표에서 옮긴 값이다(출처 절·표 번호를 주석에 적었다).
"""
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent / "recompute_results.json"
results = {}


def wilson(k, n, z):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def hyp_pmf(a, n1, n2, k):
    return math.comb(n1, a) * math.comb(n2, k - a) / math.comb(n1 + n2, k)


def fisher_two_sided(a, n1, b, n2):
    """2x2 표 (a/n1 대 b/n2)의 양측 Fisher 정확 검정 p값."""
    k = a + b
    lo, hi = max(0, k - n2), min(k, n1)
    p_obs = hyp_pmf(a, n1, n2, k)
    total = 0.0
    for x in range(lo, hi + 1):
        px = hyp_pmf(x, n1, n2, k)
        if px <= p_obs * (1 + 1e-9):
            total += px
    return min(1.0, total)


# ---- 1. Hand2Policy 시뮬 성공률 (§VII-D, 그림 12, 초록) ----
# 5 물체 x 2 손 x 10 시도 = 방법당 100 시도. 평균 61%, 8%, 17%는 동일 n이므로 합산 비율과 같다.
n = 100
policy = {"WinDEX references": 61, "WiLoR references": 8, "HaWoR references": 17}
results["policy_success"] = {}
for name, k in policy.items():
    lo68, hi68 = wilson(k, n, 0.9945)
    lo95, hi95 = wilson(k, n, 1.96)
    results["policy_success"][name] = {
        "k": k, "n": n,
        "wilson68": [round(lo68, 3), round(hi68, 3)],
        "wilson95": [round(lo95, 3), round(hi95, 3)],
    }
results["policy_fisher"] = {
    "WinDEX_vs_WiLoR_p": fisher_two_sided(61, 100, 8, 100),
    "WinDEX_vs_HaWoR_p": fisher_two_sided(61, 100, 17, 100),
    "HaWoR_vs_WiLoR_p": fisher_two_sided(17, 100, 8, 100),
}
results["policy_ratio"] = {
    "61_over_8": round(61 / 8, 3),   # 초록 7.6x
    "61_over_17": round(61 / 17, 3),  # 초록 3.6x
}
# 셀 하나(시도 10회)의 95% 구간 폭. 그림 12의 점 하나가 얼마나 흔들리는지 보여 준다.
results["single_cell_10_trials"] = {
    "k=6": [round(x, 3) for x in wilson(6, 10, 1.96)],
    "k=10": [round(x, 3) for x in wilson(10, 10, 1.96)],
}

# ---- 2. 표 V 세계 좌표 비교: 초록·표의 상대 감소율 ----
hawor_ft = {"W-MPJPE": 97.65, "WA-MPJPE": 40.91, "RTE": 2.83}
ours_world = {"W-MPJPE": 37.29, "WA-MPJPE": 17.69, "RTE": 2.00}
results["table5_reduction_pct"] = {
    k: round(100 * (hawor_ft[k] - ours_world[k]) / hawor_ft[k], 1) for k in hawor_ft
}  # 논문 본문 61.8, 56.8, 29.3

# ---- 3. 표 VI 검출 F1 = 2PR/(P+R) 재계산 ----
table6 = {
    "in_the_wild": {
        "Sapiens2-1B": (82.09, 77.43, 79.69),
        "YOLO11-pose": (70.31, 76.37, 73.21),
        "ViTPose-H": (50.00, 30.41, 37.82),
        "WinDEX-Net": (85.07, 77.09, 80.88),
    },
    "motion_capture": {
        "Sapiens2-1B": (84.60, 78.42, 81.39),
        "YOLO11-pose": (76.08, 77.37, 76.72),
        "ViTPose-H": (51.34, 50.53, 50.93),
        "WinDEX-Net": (88.24, 86.84, 87.53),
    },
}
f1_check = {}
for setname, rows in table6.items():
    f1_check[setname] = {}
    for m, (p, r, f_rep) in rows.items():
        f_calc = 2 * p * r / (p + r)
        f1_check[setname][m] = {"F1_reported": f_rep, "F1_recomputed": round(f_calc, 2),
                                "diff": round(f_calc - f_rep, 2)}
results["table6_f1"] = f1_check

# ---- 4. 표 IV 카메라 좌표 비교: 우리(P) 대 베이스라인 최선(P, FT) ----
t4 = {  # (P, FT) 쌍. 값은 표 IV 원문
    "wrist": {"METRO": (152.26, 48.34), "Mesh Graphormer": (114.89, 57.25), "MobRecon": (75.54, 53.51),
              "HaMeR": (62.43, 43.29), "WiLoR": (63.25, 38.23), "POEM": (104.45, 32.17),
              "POEM-v2": (46.26, 28.77), "UmeTrack": (42.59, 27.45), "HaWoR": (46.05, 31.87)},
    "MPJPE": {"METRO": (46.10, 34.20), "Mesh Graphormer": (36.33, 28.26), "MobRecon": (45.76, 36.86),
              "HaMeR": (36.50, 25.29), "WiLoR": (33.65, 24.53), "POEM": (137.51, 37.56),
              "POEM-v2": (34.22, 27.02), "UmeTrack": (36.80, 24.09), "HaWoR": (33.78, 23.48)},
    "PA-MPJPE": {"METRO": (16.13, 10.33), "Mesh Graphormer": (13.73, 10.41), "MobRecon": (13.99, 11.66),
                 "HaMeR": (10.90, 10.09), "WiLoR": (10.34, 10.16), "POEM": (38.21, 18.80),
                 "POEM-v2": (13.54, 11.13), "UmeTrack": (11.62, 10.82), "HaWoR": (10.61, 9.45)},
    "jitter": {"METRO": (19.66, 13.37), "Mesh Graphormer": (18.45, 12.19), "MobRecon": (9.42, 7.21),
               "HaMeR": (9.72, 7.35), "WiLoR": (9.01, 6.96), "POEM": (20.74, 9.51),
               "POEM-v2": (10.63, 8.30), "UmeTrack": (10.83, 7.82), "HaWoR": (8.68, 5.50)},
}
ours_p = {"wrist": 15.70, "MPJPE": 16.68, "PA-MPJPE": 7.82, "jitter": 3.15}
t4_out = {}
for metric, rows in t4.items():
    best_ft = min(rows.items(), key=lambda kv: kv[1][1])
    t4_out[metric] = {
        "ours_P": ours_p[metric],
        "best_FT_baseline": {"method": best_ft[0], "FT": best_ft[1][1]},
        "reduction_vs_best_FT_pct": round(100 * (best_ft[1][1] - ours_p[metric]) / best_ft[1][1], 1),
        "ours_below_all_FT": all(ours_p[metric] < v[1] for v in rows.values()),
        "ours_below_all_P": all(ours_p[metric] < v[0] for v in rows.values()),
    }
results["table4_vs_baselines"] = t4_out

# ---- 5. 표 II, III, IV의 같은 구성 값 비교 (일관성 검사) ----
results["same_config_mismatch"] = {
    "tableII_tap11_production": {"wrist": 15.70, "MPJPE": 16.68, "PA": 7.82},
    "tableIV_WinDEX_Window_P": {"wrist": 15.70, "MPJPE": 16.68, "PA": 7.82},
    "tableIII_WinDEX_Net_2it": {"wrist": 25.9, "MPJPE": 17.8, "PA": 7.8},
    "tableIII_WinDEX_Window_2it": {"wrist": 27.1, "MPJPE": 20.1, "PA": 8.1},
    "wrist_gap_Window_IV_vs_III_mm": round(27.1 - 15.70, 2),
}

# ---- 6. 처리량과 정확도 교환 (표 III, 초록) ----
results["speed"] = {
    "Window_over_Net_fps": round(43.61 / 9.50, 2),
    "Stream_over_Window_fps": round(171.00 / 43.61, 2),
    "Stream_wrist_vs_Window_2it_pct": round(100 * (42.7 - 27.1) / 27.1, 1),
    "Stream_MPJPE_vs_Window_2it_pct": round(100 * (33.2 - 20.1) / 20.1, 1),
}

# ---- 7. 반사영상 오차 (그림 10, 초록) ----
results["reproj_px"] = {
    "ours": 14.5,
    "best_FT_baseline_HaWoR": 22.42,
    "reduction_pct": round(100 * (22.42 - 14.5) / 22.42, 1),
}

# ---- 8. 학습 코퍼스 시간 합 (그림 6a의 13개 산업 시간) ----
hours = [48, 54, 62, 68, 70, 74, 78, 80, 84, 88, 92, 98, 104]
results["corpus_hours_sum"] = {"sum_of_bars": sum(hours), "stated": 1000, "n_categories": len(hours)}

# ---- 9. 사용한 보고 수치 간 검산 ----
results["tableII_gaps"] = {
    "MPJPE_layer11_vs_layer4_pct": round(100 * (17.08 - 16.68) / 17.08, 1),
    "PA_layer11_vs_layer4_pct": round(100 * (7.92 - 7.82) / 7.92, 1),
    "note": "단일 학습 실행. 반복 실험이 없어 0.2-0.4 mm 차이의 유의성은 계산할 수 없다.",
}

OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(results, ensure_ascii=False, indent=2))
