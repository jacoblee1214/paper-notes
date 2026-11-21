"""WAMotor 논문 표의 수치를 옮겨 적고 평균, 속도비, 신뢰구간, 검정을 다시 계산한다.

실행: python3 analysis/recompute.py  → analysis/recompute_results.json
입력: 논문 Table 1(LIBERO), Table 2(RoboTwin 2.0), Table 3(LIBERO-Plus), Table 4·5·6·7·10에서 옮긴 값.
재학습·재실행 없음. CPU 계산만 한다.
"""
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent / "recompute_results.json"

METHODS = ["Vanilla", "VLA-Cache", "TeaCache", "BAC", "ParaStep", "SpeCa", "FlowCast", "w/ours"]
MODELS = ["Joint-WAM", "Fast-WAM", "Faster-WAM"]

# Table 1 (LIBERO): Spa, Obj, Goa, Lon, Avg(논문), Lat(ms). 행 순서는 METHODS.
LIBERO = {
    "Joint-WAM": [
        [99.40, 98.80, 98.60, 96.60, 98.35, 660.83],
        [98.80, 98.60, 96.80, 94.60, 97.20, 434.68],
        [98.20, 99.00, 96.80, 95.00, 97.25, 276.87],
        [98.80, 98.80, 97.40, 95.80, 97.70, 331.91],
        [97.40, 97.60, 95.80, 94.00, 96.20, 346.39],
        [97.20, 98.20, 95.20, 92.40, 95.75, 245.89],
        [98.80, 98.60, 97.20, 95.80, 97.60, 246.79],
        [99.00, 98.80, 97.80, 96.40, 98.00, 216.44]],
    "Fast-WAM": [
        [97.20, 99.80, 96.40, 95.00, 97.10, 387.54],
        [97.60, 99.20, 95.60, 93.60, 96.50, 203.43],
        [97.40, 99.20, 95.80, 94.20, 96.65, 137.78],
        [97.60, 99.40, 96.40, 94.60, 97.00, 161.10],
        [96.20, 98.40, 94.60, 92.80, 95.50, 169.63],
        [94.80, 98.00, 93.00, 90.40, 94.05, 109.58],
        [97.60, 99.40, 95.60, 95.00, 96.90, 124.07],
        [97.80, 99.60, 95.80, 95.20, 97.10, 92.57]],
    "Faster-WAM": [
        [98.60, 99.80, 98.20, 97.80, 98.60, 275.85],
        [97.40, 99.20, 96.00, 96.40, 97.25, 193.24],
        [97.40, 99.20, 96.60, 96.80, 97.50, 134.21],
        [98.80, 100.00, 98.40, 97.80, 98.75, 163.10],
        [97.40, 98.40, 96.60, 95.80, 97.05, 170.81],
        [95.60, 97.20, 95.20, 94.20, 95.55, 108.57],
        [98.60, 99.80, 98.20, 97.80, 98.60, 120.38],
        [99.40, 100.00, 99.00, 98.40, 99.20, 90.10]],
}

# Table 2 (RoboTwin 2.0): Clean, Rand, Avg(논문), Lat(ms)
ROBOTWIN = {
    "Joint-WAM": [
        [91.02, 90.38, 90.70, 661.53],
        [82.12, 84.80, 83.46, 432.05],
        [86.28, 88.32, 87.30, 269.92],
        [90.80, 89.76, 90.28, 337.29],
        [89.32, 88.28, 88.80, 347.45],
        [85.94, 84.22, 85.08, 247.33],
        [89.42, 88.82, 89.12, 246.15],
        [91.24, 90.36, 90.80, 217.87]],
    "Fast-WAM": [
        [91.52, 91.34, 91.43, 398.07],
        [82.66, 85.88, 84.27, 205.55],
        [81.24, 90.52, 85.88, 139.27],
        [91.34, 89.24, 90.29, 160.30],
        [90.34, 89.24, 89.79, 167.32],
        [86.32, 85.84, 86.08, 113.21],
        [89.60, 89.32, 89.46, 130.88],
        [91.62, 91.28, 91.45, 98.95]],
    "Faster-WAM": [
        [92.52, 92.38, 92.45, 282.09],
        [80.28, 84.68, 82.48, 196.24],
        [87.08, 88.32, 87.70, 136.55],
        [91.10, 92.08, 91.59, 163.88],
        [90.08, 90.02, 90.05, 168.36],
        [85.38, 84.26, 84.82, 113.38],
        [90.04, 90.96, 90.50, 124.65],
        [92.24, 91.86, 92.05, 98.70]],
}

# Table 3 (LIBERO-Plus): Cam, Rob, Lang, Light, Backg, Noise, Layout, Avg(논문), Lat(ms)
LIBERO_PLUS = {
    "Joint-WAM": [
        [36.44, 62.69, 91.39, 90.74, 55.15, 49.03, 79.55, 66.43, 665.66],
        [34.50, 58.14, 85.81, 87.10, 46.67, 48.20, 78.01, 62.63, 432.57],
        [35.75, 62.04, 88.61, 89.83, 52.00, 49.63, 79.48, 65.33, 285.34],
        [36.45, 61.24, 90.54, 90.30, 54.53, 49.61, 79.21, 65.98, 321.58],
        [33.51, 54.43, 86.12, 90.23, 51.09, 49.41, 78.75, 63.36, 340.32],
        [36.23, 53.24, 85.36, 88.26, 50.27, 47.22, 71.56, 61.73, 259.47],
        [35.74, 61.92, 90.37, 90.37, 54.29, 49.37, 79.00, 65.87, 245.45],
        [36.64, 64.97, 90.77, 90.32, 56.38, 50.01, 82.02, 67.30, 212.54]],
    "Fast-WAM": [
        [16.53, 46.11, 68.03, 82.72, 47.94, 30.74, 59.09, 50.17, 391.43],
        [13.99, 41.04, 59.22, 79.85, 46.99, 30.55, 58.41, 47.15, 206.15],
        [14.14, 40.15, 60.28, 78.95, 46.66, 31.67, 59.34, 47.31, 133.38],
        [15.49, 42.85, 67.48, 81.85, 48.99, 30.67, 61.41, 49.82, 160.71],
        [15.98, 41.54, 62.74, 74.29, 46.33, 29.30, 59.33, 47.07, 166.93],
        [17.11, 43.91, 68.82, 81.85, 47.53, 29.56, 57.76, 49.51, 117.00],
        [15.15, 43.47, 67.42, 80.33, 43.98, 28.42, 58.12, 48.13, 122.19],
        [18.84, 47.35, 69.05, 82.30, 48.65, 31.71, 60.20, 51.16, 91.51]],
    "Faster-WAM": [
        [52.97, 71.50, 95.08, 96.29, 61.34, 62.64, 81.82, 74.52, 282.62],
        [49.25, 67.83, 91.67, 90.38, 55.02, 60.05, 78.01, 70.32, 198.28],
        [52.19, 71.23, 94.41, 94.66, 60.31, 61.29, 79.75, 73.41, 137.59],
        [51.21, 71.64, 92.53, 94.08, 59.64, 60.29, 78.97, 72.62, 163.14],
        [50.89, 70.35, 92.21, 94.85, 60.32, 61.37, 77.60, 72.51, 170.55],
        [51.88, 63.16, 88.15, 91.66, 50.33, 59.95, 71.62, 68.11, 120.03],
        [52.98, 68.86, 95.08, 92.82, 57.51, 61.28, 81.88, 72.92, 120.07],
        [55.48, 69.31, 96.30, 95.46, 60.70, 64.99, 83.13, 75.05, 90.33]],
}

# Table 4 (ablation, LIBERO-Plus): 행 = (RAVS, TSAV) 조합
ABLATION = {
    "Joint-WAM": {"neither": [35.74, 61.92, 90.37, 90.37, 54.29, 49.37, 79.00, 65.87, 245.45],
                  "RAVS": [36.63, 64.69, 90.78, 90.55, 55.83, 49.68, 80.29, 66.92, 245.51],
                  "TSAV": [36.21, 62.29, 90.26, 90.27, 54.99, 49.90, 79.55, 66.21, 228.25],
                  "both": [36.64, 64.97, 90.77, 90.32, 56.38, 50.01, 82.02, 67.30, 212.54]},
    "Fast-WAM": {"neither": [15.15, 43.47, 67.42, 80.33, 43.98, 28.42, 58.12, 48.13, 122.19],
                 "RAVS": [17.26, 44.17, 68.01, 80.89, 45.73, 30.19, 58.06, 49.19, 121.46],
                 "TSAV": [17.47, 44.33, 67.92, 80.89, 45.99, 30.02, 58.20, 49.26, 109.24],
                 "both": [18.84, 47.35, 69.05, 82.30, 48.65, 31.71, 60.20, 51.16, 91.51]},
    "Faster-WAM": {"neither": [52.98, 68.86, 95.08, 92.82, 57.51, 61.28, 81.88, 72.92, 120.07],
                   "RAVS": [55.17, 69.02, 95.10, 96.08, 61.03, 62.65, 82.00, 74.44, 120.25],
                   "TSAV": [52.53, 69.99, 95.29, 95.06, 60.26, 61.46, 80.38, 73.57, 110.03],
                   "both": [55.48, 69.31, 96.30, 95.46, 60.70, 64.99, 83.13, 75.05, 90.33]},
}

# Table 7: Endpoint MSE share (LIBERO), mean ± sd (%), S_first, S_last, S_end
ENDPOINT_SHARE = {
    "Joint-WAM": (41.21, 6.82, 25.22, 5.52, 66.43, 6.93),
    "Fast-WAM": (43.59, 5.67, 36.29, 6.63, 79.88, 2.49),
    "Faster-WAM": (48.07, 7.79, 30.91, 6.53, 78.98, 3.76),
}

# Table 9: action entropy (bits), mean ± sd, vanilla vs w/ours
ENTROPY = {"Joint-WAM": ((2.31, 0.19), (2.01, 0.22)),
           "Fast-WAM": ((2.15, 0.17), (1.80, 0.17)),
           "Faster-WAM": ((2.09, 0.36), (1.87, 0.35))}

# 시도 수. LIBERO: 스위트당 10 과제 x 50 시행 = 500, 4 스위트 = 2000.
# RoboTwin: 50 과제 x 100 시행, clean/rand 각 5000, 합 10000.
# LIBERO-Plus: 10,030 과제 x 3 시행. 과제 단위 표본 수로 10,030을 쓴다(과제 내 상관이 있으므로 보수적 근사).
N_LIBERO = 2000
N_ROBO = 10000
N_PLUS_TASKS = 10030


def wilson(p_pct, n):
    p = p_pct / 100.0
    z = 1.959963984540054
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round((centre - half) * 100, 2), round((centre + half) * 100, 2)


def diff_test(p1_pct, p2_pct, n):
    """두 독립 비율 차이의 정규근사 검정(시행 독립 가정, 과제 군집 무시). 양측 p값."""
    p1, p2 = p1_pct / 100, p2_pct / 100
    pooled = (p1 + p2) / 2
    se = math.sqrt(2 * pooled * (1 - pooled) / n)
    z = (p2 - p1) / se if se > 0 else float("nan")
    p = math.erfc(abs(z) / math.sqrt(2))
    mde = 1.96 * se * 100
    return {"diff_pp": round((p2 - p1) * 100, 2), "z": round(z, 2),
            "p_two_sided": float(f"{p:.3g}"), "mde_95_pp": round(mde, 2)}


def main():
    res = {}

    # 1. 평균 재계산: 논문 Avg가 구성 항목의 산술평균과 맞는지
    chk = {}
    for m in MODELS:
        rows = LIBERO[m]
        chk[f"LIBERO {m}"] = [round(sum(r[:4]) / 4 - r[4], 3) for r in rows]
        rr = ROBOTWIN[m]
        chk[f"RoboTwin {m}"] = [round((r[0] + r[1]) / 2 - r[2], 3) for r in rr]
        lp = LIBERO_PLUS[m]
        chk[f"LIBERO-Plus {m} (7 category unweighted)"] = [round(sum(r[:7]) / 7 - r[7], 3) for r in lp]
    res["average_check_residual_pp"] = chk

    # 2. 속도비 (Vanilla 지연 / w/ours 지연)
    spd = {}
    for m in MODELS:
        spd[m] = {
            "LIBERO": round(LIBERO[m][0][5] / LIBERO[m][7][5], 2),
            "RoboTwin": round(ROBOTWIN[m][0][3] / ROBOTWIN[m][7][3], 2),
            "LIBERO-Plus": round(LIBERO_PLUS[m][0][8] / LIBERO_PLUS[m][7][8], 2),
        }
    res["speedup_vs_vanilla"] = spd

    # 3. 최강 가속 기준선(BAC 또는 FlowCast) 대비 지연 감소율과 정확도 차
    red = {}
    for m in MODELS:
        lp = LIBERO_PLUS[m]
        red[m] = {
            "LIBERO-Plus_vs_FlowCast_latency_reduction_pct": round((1 - lp[7][8] / lp[6][8]) * 100, 1),
            "LIBERO-Plus_vs_FlowCast_avg_gain_pp": round(lp[7][7] - lp[6][7], 2),
            "RoboTwin_vs_BAC_avg_gain_pp": round(ROBOTWIN[m][7][2] - ROBOTWIN[m][3][2], 2),
            "LIBERO_vs_SpeCa_latency_reduction_pct": round((1 - LIBERO[m][7][5] / LIBERO[m][5][5]) * 100, 1),
        }
    res["baseline_comparisons"] = red

    # 4. 신뢰구간 (Wilson)과 Vanilla 대비 차이 검정
    ci = {}
    tests = {}
    for m in MODELS:
        v = LIBERO[m][0][4]
        o = LIBERO[m][7][4]
        ci[f"LIBERO {m} w/ours"] = wilson(o, N_LIBERO)
        ci[f"LIBERO {m} Vanilla"] = wilson(v, N_LIBERO)
        tests[f"LIBERO {m} ours vs Vanilla"] = diff_test(v, o, N_LIBERO)
        bac = LIBERO[m][3][4]
        tests[f"LIBERO {m} ours vs BAC"] = diff_test(bac, o, N_LIBERO)

        rv = ROBOTWIN[m][0][2]
        ro = ROBOTWIN[m][7][2]
        ci[f"RoboTwin {m} w/ours"] = wilson(ro, N_ROBO)
        tests[f"RoboTwin {m} ours vs Vanilla"] = diff_test(rv, ro, N_ROBO)
        tests[f"RoboTwin {m} ours vs BAC"] = diff_test(ROBOTWIN[m][3][2], ro, N_ROBO)

        lv = LIBERO_PLUS[m][0][7]
        lo = LIBERO_PLUS[m][7][7]
        ci[f"LIBERO-Plus {m} w/ours"] = wilson(lo, N_PLUS_TASKS)
        tests[f"LIBERO-Plus {m} ours vs Vanilla"] = diff_test(lv, lo, N_PLUS_TASKS)
        tests[f"LIBERO-Plus {m} ours vs FlowCast"] = diff_test(LIBERO_PLUS[m][6][7], lo, N_PLUS_TASKS)
    res["wilson_ci_95_pct"] = ci
    res["difference_tests_independent_approx"] = tests

    # 5. 제거 실험 (Table 4) 기준선 'neither'가 Table 3의 FlowCast 행과 같은지
    ident = {m: ABLATION[m]["neither"] == LIBERO_PLUS[m][6] for m in MODELS}
    res["ablation_neither_equals_flowcast_row"] = ident

    # 6. 경계 전이 몫(Table 7) 검산
    share = {}
    for m, (sf, sfsd, sl, slsd, se, sesd) in ENDPOINT_SHARE.items():
        share[m] = {"sum_check": round(sf + sl - se, 2),
                    "intermediate_share_pct": round(100 - se, 2),
                    "boundary_over_uniform_ratio": round(se / 22.22, 2)}
    res["endpoint_share_checks"] = share

    # 7. 엔트로피 감소율
    ent = {m: {"relative_drop_pct": round((a[0] - b[0]) / a[0] * 100, 1),
               "pooled_sd_effect_size": round((a[0] - b[0]) / math.sqrt((a[1] ** 2 + b[1] ** 2) / 2), 2)}
           for m, (a, b) in ENTROPY.items()}
    res["entropy_checks"] = ent

    # 8. 이론의 설정값 검산: 0→1 Euler 이후 임계값 스케일이 h_mid 기준인지 (Shift-5 등 10스텝)
    res["notes"] = [
        "Table 2 RoboTwin: 평균은 clean과 rand의 산술평균으로 맞음(잔차 0.00 안팎).",
        "LIBERO-Plus Avg는 7개 교란 범주의 단순 평균과 맞음(잔차 0.00 안팎). 저장소의 summarize 스크립트는 과제 수 가중 평균을 쓰므로 범주별 과제 수가 같지 않으면 두 값이 다릅니다.",
        "CI는 Wilson 근사이며 과제 군집을 무시합니다. LIBERO-Plus는 과제 수 10,030을 표본 단위로 씀.",
    ]

    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print("speedups", json.dumps(spd, ensure_ascii=False))
    mx = max(abs(x) for v in chk.values() for x in v)
    print("max |avg residual| (pp):", mx)


if __name__ == "__main__":
    main()
