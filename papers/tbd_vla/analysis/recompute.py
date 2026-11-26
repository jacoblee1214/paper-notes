"""TBD-VLA (arXiv:2606.07895 v1) 논문 표 수치를 다시 계산하고 통계 검정을 한다.

입력: 논문 본문, 표, 부록 값을 아래에 그대로 옮겨 적었다(출처는 주석). 외부 데이터는 쓰지 않는다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, statsmodels; 수 초)

가정(논문에 없는 것은 표시한다)
- LIBERO 스위트당 시행 수는 500(10 과제 x 50 회, 논문 5.1절)이고 전체는 2,000이다. 기준선 값은 각 논문이 보고한 값이며
  시드와 시행 수가 같은지 알 수 없다. 그래서 기준선과의 검정은 "같은 시행 수 가정" 참고용이다.
- 실로봇은 표 13의 분자와 분모(셀당 20회)를 그대로 쓴다. 셀 안의 시행은 독립이라고 가정한다.
- LIBERO-Plus 차원 가중치는 공식 리더보드 11행으로 추정한 값(별도 분석에서 얻음)을 상수로 넣었다.
- Fig. 4의 pi0.5 w/ RTC (L=4) 값은 본문 "3.4% higher"에서 역산한 89.8 이다(그림 판독 값).
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def r1(x):
    return round(float(x), 2)


# ---------------------------------------------------------------- V1: LIBERO 표 2
LIBERO = {  # 표 2 왼쪽: Spatial, Object, Goal, Long, Avg(보고)
    "OpenVLA-OFT": (96.2, 98.3, 96.2, 90.7, 95.4),
    "pi0-FAST": (96.4, 96.8, 88.6, 60.2, 85.5),
    "pi0.5": (98.8, 98.2, 98.0, 92.4, 96.9),
    "GR00T-N1": (94.4, 97.6, 93.0, 90.6, 93.9),
    "MolmoAct": (87.0, 95.4, 87.6, 77.2, 86.6),
    "UniVLA": (95.4, 98.8, 93.6, 94.0, 95.5),
    "VLA-0": (97.0, 97.8, 96.2, 87.6, 94.7),
    "Disc Diff VLA": (97.2, 98.6, 97.4, 92.0, 96.3),
    "UD-VLA": (94.1, 95.7, 91.2, 89.6, 92.7),
    "dVLA": (97.4, 97.9, 98.2, 92.2, 96.4),
    "TBD-VLA": (97.6, 99.6, 97.4, 96.6, 97.7),
}
v1 = {}
for name, row in LIBERO.items():
    m = sum(row[:4]) / 4
    v1[name] = {"mean4": r1(m), "reported": row[4], "diff": r1(row[4] - m)}
RES["V1_libero_table2_avg"] = v1
RES["V1_text_avg_976"] = {"text_5.3": 97.6, "table2": 97.7, "mean_of_row": r1(sum(LIBERO["TBD-VLA"][:4]) / 4)}

# 정수 시행 수 점검 (스위트당 500)
tbd2 = LIBERO["TBD-VLA"][:4]
T11_L0 = (97.6, 99.0, 98.6, 95.6)  # 표 11 L=0: Spatial, Object, Goal, Long(LIBERO-10)
RES["V2_integer_counts_n500"] = {
    "table2_counts": [round(v * 5, 3) for v in tbd2], "table2_total": sum(round(v * 5) for v in tbd2),
    "table11_L0_counts": [round(v * 5, 3) for v in T11_L0], "table11_L0_total": sum(round(v * 5) for v in T11_L0),
    "table2_total_rate": r1(sum(round(v * 5) for v in tbd2) / 20),
    "table11_total_rate": r1(sum(round(v * 5) for v in T11_L0) / 20),
}
RES["V2_table2_vs_table11_L0"] = {
    s: {"table2": a, "table11_L0": b, "diff": r1(a - b)}
    for s, a, b in zip(["Spatial", "Object", "Goal", "Long"], tbd2, T11_L0)
}

# ---------------------------------------------------------------- V3: TBD-VLA vs 기준선 (스위트당 500, 가정)
v3 = {}
for base in ["pi0.5", "dVLA", "Disc Diff VLA", "UniVLA", "OpenVLA-OFT"]:
    row = {}
    for j, s in enumerate(["Spatial", "Object", "Goal", "Long"]):
        k1, k2 = round(LIBERO["TBD-VLA"][j] * 5), round(LIBERO[base][j] * 5)
        row[s] = {"tbd": LIBERO["TBD-VLA"][j], "base": LIBERO[base][j], "fisher_p": float(f"{fisher(k1, 500, k2, 500):.3g}")}
    k1 = sum(round(x * 5) for x in LIBERO["TBD-VLA"][:4])
    k2 = sum(round(x * 5) for x in LIBERO[base][:4])
    row["pooled_2000"] = {"tbd": r1(k1 / 20), "base": r1(k2 / 20), "fisher_p": float(f"{fisher(k1, 2000, k2, 2000):.3g}")}
    v3[base] = row
RES["V3_libero_vs_baselines_n500_assumed"] = v3
k = round(97.7 * 20)
RES["V3_tbd_total_wilson"] = [r1(100 * x) for x in wilson(k, 2000)]

# ---------------------------------------------------------------- V4: 표 11 (지연과 RTC)
T11 = {  # 스위트: L0, (L1 w/o, w/), (L2 w/o, w/), (L4 w/o, w/)
    "LIBERO-10": (95.6, (93.2, 93.6), (89.0, 94.4), (69.8, 92.6)),
    "LIBERO-Goal": (98.6, (95.6, 96.6), (94.0, 95.4), (83.2, 90.0)),
    "LIBERO-Spatial": (97.6, (95.6, 96.6), (91.8, 94.8), (54.2, 93.4)),
    "LIBERO-Object": (99.0, (99.8, 98.6), (97.2, 97.2), (82.0, 96.6)),
}
rep_overall = {"L0": 97.7, "L1": (96.1, 96.4), "L2": (93.0, 95.5), "L4": (72.3, 93.2)}
rep_delta = {"L1": 0.3, "L2": 2.5, "L4": 20.9}
v4 = {}
for li, L in enumerate(["L1", "L2", "L4"], start=1):
    wo = np.mean([T11[s][li][0] for s in T11])
    w = np.mean([T11[s][li][1] for s in T11])
    ko = sum(round(T11[s][li][0] * 5) for s in T11)
    kw = sum(round(T11[s][li][1] * 5) for s in T11)
    v4[L] = {"mean_wo": r1(wo), "mean_w": r1(w), "reported": rep_overall[L], "delta": r1(w - wo),
             "reported_delta": rep_delta[L], "counts_wo": ko, "counts_w": kw,
             "fisher_p_rtc_vs_none": float(f"{fisher(kw, 2000, ko, 2000):.3g}")}
v4["L0_mean"] = r1(np.mean([T11[s][0] for s in T11]))
# 단순 대비: L=0 대비 L=4 w/ RTC 손실
v4["L4_wRTC_vs_L0"] = {"drop_pp": r1(v4["L0_mean"] - v4["L4"]["mean_w"]),
                       "fisher_p": round(fisher(v4["L4"]["counts_w"], 2000, round(v4["L0_mean"] * 20), 2000), 4)}
RES["V4_table11"] = v4
# pi0.5 w/ RTC (L=4) = 89.8 (본문 3.4% 역산, 그림 판독)
k_tbd, k_pi = v4["L4"]["counts_w"], round(0.898 * 2000)
RES["V4_tbd_vs_pi05_L4_rtc_figure_read"] = {"tbd": r1(k_tbd / 20), "pi05_figure_read": 89.8,
                                            "fisher_p": float(f"{fisher(k_tbd, 2000, k_pi, 2000):.3g}")}

# ---------------------------------------------------------------- V5: LIBERO-Plus (표 12)
DIMS = ["Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout"]
LP_PRE = {"Spatial": (99.20, 62.57, 78.20, 98.28, 95.34, 95.15, 97.14), "Object": (93.69, 69.60, 91.24, 99.66, 89.52, 98.82, 83.87),
          "Goal": (76.71, 58.19, 65.60, 94.26, 86.83, 70.45, 66.82), "Long": (81.62, 51.14, 74.41, 90.87, 83.39, 95.32, 89.74)}
LP_PRE_AVG = (87.81, 60.38, 77.36, 95.77, 88.77, 89.94, 84.39)
LP_NOPRE = {"Spatial": (31.64, 62.28, 52.30, 98.97, 94.57, 72.36, 94.03), "Object": (43.43, 65.32, 55.93, 98.65, 97.17, 63.50, 76.42),
            "Goal": (15.93, 64.30, 44.14, 83.15, 72.95, 51.71, 58.11), "Long": (26.73, 59.54, 56.13, 76.64, 90.65, 59.02, 87.50)}
LP_NOPRE_AVG = (29.43, 62.86, 52.12, 89.35, 88.84, 61.65, 79.02)
LP_ROW_AVG_PRE = {"Spatial": 89.41, "Object": 89.49, "Goal": 74.12, "Long": 80.93}
LP_ROW_AVG_NOPRE = {"Spatial": 72.31, "Object": 71.49, "Goal": 55.76, "Long": 65.17}
LP_DELTA = (58.38, -2.48, 25.24, 6.42, -0.07, 28.29, 5.37, 17.31)
v5 = {"col_avg_pre": [r1(np.mean([LP_PRE[s][i] for s in LP_PRE])) for i in range(7)], "col_avg_pre_reported": list(LP_PRE_AVG),
      "col_avg_nopre": [r1(np.mean([LP_NOPRE[s][i] for s in LP_NOPRE])) for i in range(7)], "col_avg_nopre_reported": list(LP_NOPRE_AVG),
      "row_avg_pre": {s: r1(np.mean(LP_PRE[s])) for s in LP_PRE}, "row_avg_pre_reported": LP_ROW_AVG_PRE,
      "row_avg_nopre": {s: r1(np.mean(LP_NOPRE[s])) for s in LP_NOPRE}, "row_avg_nopre_reported": LP_ROW_AVG_NOPRE,
      "macro_pre": r1(np.mean(LP_PRE_AVG)), "macro_nopre": r1(np.mean(LP_NOPRE_AVG)),
      "delta_recomputed": [r1(a - b) for a, b in zip(LP_PRE_AVG, LP_NOPRE_AVG)] + [r1(np.mean(LP_PRE_AVG) - np.mean(LP_NOPRE_AVG))],
      "delta_reported": list(LP_DELTA)}
RES["V5_liberoplus_table12"] = v5

# 프로젝트 페이지의 기준선 행(논문 본문에는 레이더 차트만 있음)
LP_BASE = {
    "OpenVLA-OFT": ((55.6, 21.7, 81.0, 92.7, 91.0, 78.6, 68.7), 67.9),
    "UniVLA": ((1.8, 46.2, 69.6, 69.0, 81.0, 21.2, 31.9), 42.9),
    "pi0-FAST": ((65.1, 21.6, 61.0, 73.2, 73.2, 74.4, 68.8), 61.6),
    "pi0": ((13.8, 6.0, 58.8, 85.0, 81.4, 79.0, 68.9), 53.6),
    "RIPT-VLA": ((55.2, 31.2, 77.6, 88.4, 91.6, 73.5, 74.2), 68.4),
}
W = np.array([0.157, 0.158, 0.166, 0.114, 0.113, 0.161, 0.131])  # 리더보드 11행 적합 가중치(합 = 1.004)
W = W / W.sum()
v5b = {}
for name, (row, rep) in LP_BASE.items():
    v5b[name] = {"reported": rep, "simple_mean": r1(np.mean(row)), "weighted": r1(float(np.dot(W, row)))}
v5b["TBD-VLA w/ pretrain"] = {"reported": 83.5, "simple_mean": r1(np.mean(LP_PRE_AVG)), "weighted": r1(float(np.dot(W, LP_PRE_AVG)))}
v5b["TBD-VLA w/o pretrain"] = {"reported": 66.2, "simple_mean": r1(np.mean(LP_NOPRE_AVG)), "weighted": r1(float(np.dot(W, LP_NOPRE_AVG)))}
v5b["weights"] = [round(float(x), 3) for x in W]
RES["V5b_liberoplus_weighting"] = v5b
RES["V5c_gap_claim"] = {"text_83.0_minus_OFT_67.9": r1(83.0 - 67.9), "83.49_minus_RIPT_68.4": r1(83.49 - 68.4),
                        "83.49_minus_OFT_67.9": r1(83.49 - 67.9), "weighted_tbd_minus_RIPT": r1(v5b["TBD-VLA w/ pretrain"]["weighted"] - 68.4)}
# 10,030 롤아웃 가정: 차원 가중치 x 10,030 = 차원별 시행 수 추정
RES["V5d_dim_counts_est_10030"] = {d: int(round(w * 10030)) for d, w in zip(DIMS, W)}

# ---------------------------------------------------------------- V6: SimplerEnv Widow-X (표 3)
WX = {  # (Spoon grasp, success, Carrot grasp, success, Block grasp, success, Eggplant grasp, success, Avg)
    "Octo": (34.7, 12.5, 52.8, 8.3, 31.9, 0.0, 66.7, 43.1, 16.0),
    "OpenVLA": (4.1, 0.0, 33.3, 0.0, 12.5, 0.0, 8.3, 4.1, 1.0),
    "SpatialVLA": (25.0, 20.8, 41.7, 20.8, 58.3, 25.0, 79.2, 70.8, 34.4),
    "pi0": (45.8, 29.1, 25.0, 0.0, 50.0, 16.6, 91.6, 62.5, 27.1),
    "pi0-FAST": (62.5, 29.1, 58.5, 21.9, 54.0, 10.8, 83.3, 66.6, 32.1),
    "pi0.5": (65.3, 44.4, 57.0, 29.2, 75.0, 18.1, 80.5, 63.9, 38.9),
    "UniVLA": (83.3, 83.3, 74.0, 66.7, 95.8, 33.3, 100.0, 95.8, 69.8),
    "LLaDA-VLA": (None, 56.9, None, 76.3, None, 30.6, None, 58.3, 55.5),
    "Disc Diff VLA": (70.8, 29.2, 58.3, 29.2, 62.5, 20.8, 91.7, 70.8, 37.5),
    "TBD-VLA": (94.0, 52.0, 93.2, 86.8, 77.2, 31.2, 100.0, 97.2, 66.8),
}


def min_n(values, nmax=2000):
    for n in range(1, nmax + 1):
        if all(round(100 * round(v * n / 100) / n, 1) == round(v, 1) for v in values):
            return n
    return None


v6 = {}
for name, row in WX.items():
    succ = [row[1], row[3], row[5], row[7]]
    allv = [x for x in row[:8] if x is not None]
    v6[name] = {"mean_success": r1(np.mean(succ)), "reported": row[8], "diff": r1(row[8] - np.mean(succ)),
                "min_common_n": min_n(allv), "min_n_success_only": min_n(succ)}
RES["V6_widowx_table3"] = v6
RES["V6_tbd_counts_n250"] = [[x, round(x * 2.5, 3)] for x in WX["TBD-VLA"][:8]]
RES["V6_grasp_ge_success_tbd"] = all(WX["TBD-VLA"][i] >= WX["TBD-VLA"][i + 1] for i in (0, 2, 4, 6))

# ---------------------------------------------------------------- V7: Google Robot (표 4)
GR = {  # VM(pick, move, drawer, avg), VA(pick, move, drawer, avg)
    "Octo": ((17.0, 4.2, 22.7, 16.8), (0.6, 3.1, 1.1, 1.1)),
    "OpenVLA": ((16.3, 46.2, 35.6, 27.7), (54.5, 47.7, 17.7, 39.8)),
    "SpatialVLA": ((86.0, 77.9, 57.4, 73.8), (88.0, 72.7, 41.8, 70.7)),
    "pi0": ((72.7, 65.3, 38.3, 58.8), (75.2, 63.7, 25.6, 54.8)),
    "pi0-FAST": ((75.3, 67.5, 42.9, 61.9), (77.6, 68.2, 31.3, 59.0)),
    "InternVLA-M1": ((95.3, 90.0, 52.5, 79.3), (97.1, 82.0, 72.0, 83.7)),
    "TBD-VLA": ((99.2, 85.0, 88.9, 91.0), (97.2, 78.3, 83.4, 86.3)),
}
v7 = {}
for name, (vm, va) in GR.items():
    v7[name] = {"VM_mean3": r1(np.mean(vm[:3])), "VM_reported": vm[3], "VM_diff": r1(vm[3] - np.mean(vm[:3])),
                "VA_mean3": r1(np.mean(va[:3])), "VA_reported": va[3], "VA_diff": r1(va[3] - np.mean(va[:3]))}
RES["V7_google_table4"] = v7
RES["V7_table5_base_88.7"] = {"mean_VM_VA_rounded": r1((91.0 + 86.3) / 2), "mean_unrounded": r1((np.mean(GR["TBD-VLA"][0][:3]) + np.mean(GR["TBD-VLA"][1][:3])) / 2)}

# ---------------------------------------------------------------- V8: 표 5, 6 지연과 순전파 수
T5 = [("m=1,nd=2,Exp", 1, 2, 84.6, 0.223, 16), ("m=16,nd=2,Exp", 16, 2, 84.0, 0.061, 2), ("m=4,nd=1,Exp", 4, 1, 85.7, 0.060, 2),
      ("m=4,nd=2,Argmax", 4, 2, 81.6, 0.086, 4), ("m=4,nd=2,Exp", 4, 2, 88.7, 0.086, 4)]
Ha = 8
v8 = {"passes_formula": {r[0]: math.ceil(Ha / r[1]) * r[2] for r in T5}, "passes_reported": {r[0]: r[5] for r in T5}}
base = T5[-1]
v8["deltas"] = {r[0]: {"succ": r1(r[3] - base[3]), "time": round(r[4] - base[4], 3), "passes": r[5] - base[5]} for r in T5[:-1]}
x = np.array([r[5] for r in T5], float)
y = np.array([r[4] for r in T5])
A = np.vstack([x, np.ones_like(x)]).T
coef = np.linalg.lstsq(A, y, rcond=None)[0]
v8["time_vs_passes_fit"] = {"slope_s_per_pass": round(float(coef[0]), 4), "intercept_s": round(float(coef[1]), 4),
                            "pred_6_passes_Ha12": round(float(coef[0] * 6 + coef[1]), 3), "table1_latency": 0.117,
                            "residuals": [round(float(v), 4) for v in (y - A @ coef)]}
T6 = (0.185, 0.125, 0.113, 0.086)
v8["table6_chain"] = {"deltas": [round(T6[i + 1] - T6[i], 3) for i in range(3)], "reported": [-0.060, -0.012, -0.027],
                      "speedup_total": round(T6[0] / T6[3], 2)}
v8["latency_values"] = {"table1_libero_Ha12": 0.117, "appendixC3_real_Ha12": 0.119, "table5_6_Ha8_A40": 0.086, "table6_baseline": 0.185,
                        "m1_nd2_Ha8": 0.223, "project_page_Ha12": 0.095, "project_page_Ha8": 0.070,
                        "frames_at_15fps_0.119": round(0.119 * 15, 3)}
RES["V8_table5_6_latency"] = v8

# ---------------------------------------------------------------- V9: 실로봇 표 13
TASKS = ["Everything in Bin", "Bread in Toaster", "Transfer the Liquid"]
SET = ["ID", "Camera", "Language", "Background"]
TAB13 = {
    "pi0.5 w/o RTC": [[16, 14, 8, 11], [17, 0, 9, 13], [12, 0, 0, 12]],
    "pi0.5 w/ RTC": [[16, 15, 8, 12], [18, 0, 11, 11], [16, 0, 0, 13]],
    "TBD-VLA w/o RTC": [[14, 12, 7, 12], [17, 19, 13, 17], [13, 0, 8, 12]],
    "TBD-VLA w/ RTC": [[17, 12, 6, 14], [19, 19, 12, 18], [16, 0, 12, 16]],
}
REP13 = {  # 표 13 Avg(%) 열: 과제별
    "pi0.5 w/o RTC": (61.25, 48.75, 30.00), "pi0.5 w/ RTC": (63.75, 50.00, 36.25),
    "TBD-VLA w/o RTC": (56.25, 82.50, 41.25), "TBD-VLA w/ RTC": (61.25, 85.00, 55.00)}
FIG5 = {  # 그림 5 막대 라벨: 평균, ID, Background, Language, Camera
    "pi0.5 w/o RTC": (46.7, 75.0, 60.0, 28.3, 23.3), "pi0.5 w/ RTC": (50.0, 83.3, 60.0, 31.7, 25.0),
    "TBD-VLA w/o RTC": (60.0, 73.3, 68.3, 46.7, 51.7), "TBD-VLA w/ RTC": (67.1, 86.7, 80.0, 50.0, 51.7)}
v9 = {"task_avg": {}, "setting_avg": {}, "overall": {}, "fig5_check": {}}
for m, cells in TAB13.items():
    arr = np.array(cells)
    v9["task_avg"][m] = {"recomputed": [r1(100 * arr[t].sum() / 80) for t in range(3)], "reported": list(REP13[m])}
    v9["setting_avg"][m] = [r1(100 * arr[:, s].sum() / 60) for s in range(4)]
    tot = int(arr.sum())
    v9["overall"][m] = {"k": tot, "n": 240, "pct": r1(100 * tot / 240), "wilson95": [r1(100 * v) for v in wilson(tot, 240)]}
    v9["fig5_check"][m] = {"recomputed(avg,ID,Bg,Lang,Cam)": [r1(100 * tot / 240), r1(100 * arr[:, 0].sum() / 60), r1(100 * arr[:, 3].sum() / 60),
                                                         r1(100 * arr[:, 2].sum() / 60), r1(100 * arr[:, 1].sum() / 60)], "fig5_labels": list(FIG5[m])}
v9["tbd_rtc_row_avg_reported"] = {"paper": [86.67, 51.67, 50.00, 80.00, 67.08], "recomputed": [r1(100 * np.array(TAB13["TBD-VLA w/ RTC"])[:, s].sum() / 60) for s in range(4)]
                                  + [v9["overall"]["TBD-VLA w/ RTC"]["pct"]]}
RES["V9_real_world_table13"] = v9


def tot(m, tasks=(0, 1, 2), sets=(0, 1, 2, 3)):
    a = np.array(TAB13[m])
    return int(sum(a[t][s] for t in tasks for s in sets)), 20 * len(tasks) * len(sets)


v9t = {}
for a, b in [("TBD-VLA w/ RTC", "pi0.5 w/ RTC"), ("TBD-VLA w/o RTC", "pi0.5 w/o RTC")]:
    k1, n1 = tot(a)
    k2, n2 = tot(b)
    v9t[f"{a} vs {b}"] = {"k": [k1, k2], "n": [n1, n2], "fisher_p": float(f"{fisher(k1, n1, k2, n2):.3g}")}
    # 설정별
    for si, sname in enumerate(SET):
        k1, n1 = tot(a, sets=(si,))
        k2, n2 = tot(b, sets=(si,))
        v9t[f"{a} vs {b}"][sname] = {"k": [k1, k2], "n": 60, "fisher_p": round(fisher(k1, 60, k2, 60), 4)}
    for ti, tname in enumerate(TASKS):
        k1, n1 = tot(a, tasks=(ti,))
        k2, n2 = tot(b, tasks=(ti,))
        v9t[f"{a} vs {b}"][tname] = {"k": [k1, k2], "n": 80, "fisher_p": round(fisher(k1, 80, k2, 80), 4)}
    # 층화 CMH (12 셀)
    tabs = []
    A_ = np.array(TAB13[a])
    B_ = np.array(TAB13[b])
    for t in range(3):
        for s in range(4):
            tabs.append(np.array([[A_[t][s], 20 - A_[t][s]], [B_[t][s], 20 - B_[t][s]]]))
    st = StratifiedTable(tabs)
    tst = st.test_null_odds(correction=True)
    v9t[f"{a} vs {b}"]["CMH_12_cells"] = {"pooled_OR": r1(st.oddsratio_pooled), "p": float(f"{tst.pvalue:.3g}")}
    # 기여도: 차이의 분해
    diff = (A_ - B_)
    v9t[f"{a} vs {b}"]["cell_diff"] = {f"{TASKS[t]}/{SET[s]}": int(diff[t][s]) for t in range(3) for s in range(4)}
    v9t[f"{a} vs {b}"]["total_diff"] = int(diff.sum())
    top = sorted(((int(diff[t][s]), f"{TASKS[t]}/{SET[s]}") for t in range(3) for s in range(4)), reverse=True)[:3]
    v9t[f"{a} vs {b}"]["top3_cells"] = top
    ex = [(t, s) for t in range(3) for s in range(4) if not (B_[t][s] == 0)]
    k1 = sum(A_[t][s] for t, s in ex)
    k2 = sum(B_[t][s] for t, s in ex)
    v9t[f"{a} vs {b}"]["excluding_pi05_zero_cells"] = {"cells": len(ex), "k": [int(k1), int(k2)], "n": 20 * len(ex),
                                                       "pct": [r1(100 * k1 / (20 * len(ex))), r1(100 * k2 / (20 * len(ex)))],
                                                       "fisher_p": round(fisher(int(k1), 20 * len(ex), int(k2), 20 * len(ex)), 4)}
    # 호환: 과제별 ID 한 칸 (분포 안 쪽)
    k1, n1 = tot(a, sets=(0,))
    k2, n2 = tot(b, sets=(0,))
    v9t[f"{a} vs {b}"]["ID_only"] = {"k": [k1, k2], "n": 60, "fisher_p": round(fisher(k1, 60, k2, 60), 4)}
RES["V9b_tbd_vs_pi05"] = v9t

# RTC 효과 (TBD, pi0.5): 셀 12개 쌍
v9r = {}
for mname in ["TBD-VLA", "pi0.5"]:
    a = np.array(TAB13[f"{mname} w/ RTC"])
    b = np.array(TAB13[f"{mname} w/o RTC"])
    d = (a - b).flatten()
    pos, neg, tie = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    sign_p = float(stats.binomtest(pos, pos + neg, 0.5).pvalue) if pos + neg else 1.0
    k1, k2 = int(a.sum()), int(b.sum())
    tabs = [np.array([[a.flatten()[i], 20 - a.flatten()[i]], [b.flatten()[i], 20 - b.flatten()[i]]]) for i in range(12)]
    st = StratifiedTable(tabs)
    v9r[mname] = {"k_rtc": k1, "k_none": k2, "n": 240, "diff_pp": r1(100 * (k1 - k2) / 240), "fisher_p": round(fisher(k1, 240, k2, 240), 4),
                  "cells_pos_neg_tie": [pos, neg, tie], "sign_test_p": round(sign_p, 4), "CMH_p": float(f"{st.test_null_odds(correction=True).pvalue:.3g}"),
                  "tbd_rtc_ci_diff_note": "비쌍 검정. 같은 시드/초기 상태의 쌍 정보 없음"}
RES["V9c_rtc_effect"] = v9r

# ---------------------------------------------------------------- V10: 학습 규모 산수 (표 7, 9, 부록 A)
T7 = [("DROID", 53282, 14153535), ("BC-Z", 39350, 5471693), ("RoboMind", 30335, 4710134), ("RH20T", 6991, 2899179), ("RoboSet", 18300, 2551749),
      ("MolmoAct", 7902, 1110869), ("UT Austin Mutex", 1500, 361883), ("Stanford Hydra", 570, 358234), ("Austin Sailor", 240, 353094),
      ("TOTO", 902, 294139), ("Berkeley AutoLab UR5", 896, 86887)]
v10 = {"episodes_sum": sum(r[1] for r in T7), "samples_sum": sum(r[2] for r in T7), "reported": [160268, 32351396],
       "samples_per_episode": {r[0]: round(r[2] / r[1], 1) for r in T7}}
samples_seen = 80_000 * 1008
v10["pretrain"] = {"samples_seen": samples_seen, "epochs_over_mixture": round(samples_seen / 32351396, 2),
                   "samples_per_gpu_hour": round(samples_seen / 1600), "samples_per_gpu_second": round(samples_seen / 1600 / 3600, 1),
                   "wall_hours_4gpu": 1600 / 4, "wall_days_4gpu": round(1600 / 4 / 24, 1)}
v10["finetune_samples_seen"] = {"LIBERO(80K x 72)": 80_000 * 72, "WidowX(20K x 256)": 20_000 * 256, "Google(40K x 256)": 40_000 * 256, "real(20K x 72)": 20_000 * 72}
v10["finetune_gpu_hours_per_kstep"] = {"LIBERO": round(120 / 80, 2), "WidowX": round(40 / 20, 2), "Google": round(60 / 40, 2)}
v10["droid_share_samples"] = round(14153535 / 32351396, 3)
RES["V10_training_scale"] = v10

# ---------------------------------------------------------------- V11: 마스크 비율 샘플링 (식 1 대 코드)
N = 28  # m * Da = 4 * 7 토큰
cnt_block = [1 / (N + 1)] * (N + 1)  # 코드: 블록당 t ~ U(0,1), 토큰은 Bernoulli(t) -> 마스크 수 균일
cnt_token = [math.comb(N, k) * 0.5 ** N for k in range(N + 1)]  # 식 1 문자 그대로: 토큰마다 t_i ~ U, Bernoulli(t_i) -> 주변 확률 1/2, 독립
RES["V11_mask_count_distribution_N28"] = {
    "code_per_block_t": {"P_all_masked": cnt_block[N], "P_none_masked": cnt_block[0], "mean": N / 2, "sd": r1(math.sqrt((N + 2) * N / 12))},
    "paper_literal_per_token_t": {"P_all_masked": cnt_token[N], "P_none_masked": cnt_token[0], "mean": N / 2, "sd": r1(math.sqrt(N * 0.25)),
                                  "P_count_ge_24": float(sum(cnt_token[24:])), "P_count_le_4": float(sum(cnt_token[:5]))},
    "code_P_count_ge_24": float(sum(cnt_block[24:])), "code_P_count_le_4": float(sum(cnt_block[:5])),
    "note": "추론 첫 단계는 블록 28개 토큰이 모두 마스크 상태. 식 1을 문자 그대로 읽으면 학습에서 거의 보이지 않는 상태",
}
# ---------------------------------------------------------------- V12: 토큰 수와 디코딩 단계 수
Da = 7
tok = {"tokens_per_chunk_Hp16": 16 * Da, "tokens_per_block_m4": 4 * Da, "blocks_K": 4,
       "forward_passes_Ha12_nd2": math.ceil(12 / 4) * 2, "forward_passes_Ha8_nd2": math.ceil(8 / 4) * 2, "forward_passes_Ha12_rtc_d2": 1 + 3 * 2,
       "ar_passes_if_token_by_token_Ha12": 12 * Da}
RES["V12_decode_counts"] = tok

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")


def show(k):
    print("==", k)
    print(json.dumps(RES[k], ensure_ascii=False)[:1800])


for k in RES:
    show(k)
