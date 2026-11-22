"""VSpark 논문 표 재계산과 일관성 검사 (CPU, numpy·scipy, 1초 미만).

입력: 논문 표 값(아래 상수, paper/paper.txt의 pdftotext -layout 출처 표 번호를 적어 둠)과
      공식 저장소 configs/ (있을 때만, repo/ 는 얕은 clone을 정적으로 읽음).
출력: analysis/recompute_results.json 과 표준출력 요약.

검사 번호 V1–V22는 이 문서 안에서만 쓰는 내부 라벨입니다.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT / "repo"

METHODS = ["EAGLE3", "P-EAGLE", "DFlash", "DFlash2", "DSpark", "VSpark"]
MODELS = ["pi0-FAST", "StarVLA-FAST", "ActionCodec", "G0.5"]
SUITES = ["Goal", "Spatial", "Object", "Long"]

# 표 1 (LIBERO): [method] -> (tau[4], speedup[4], tau_avg, speedup_avg), 값은 논문 표 1 그대로
T1 = {
    "pi0-FAST": {
        "EAGLE3": ([3.09, 3.12, 3.31, 3.17], [1.22, 1.24, 1.31, 1.27], 3.17, 1.26),
        "P-EAGLE": ([2.94, 2.97, 3.12, 3.00], [1.45, 1.49, 1.57, 1.48], 3.01, 1.50),
        "DFlash": ([2.33, 2.42, 2.44, 2.30], [1.39, 1.46, 1.46, 1.36], 2.37, 1.41),
        "DFlash2": ([2.99, 3.14, 3.20, 2.96], [1.50, 1.56, 1.59, 1.47], 3.07, 1.53),
        "DSpark": ([3.08, 3.12, 3.31, 3.14], [1.57, 1.59, 1.65, 1.60], 3.16, 1.60),
        "VSpark": ([3.86, 3.99, 4.03, 3.86], [1.91, 1.98, 1.99, 1.85], 3.94, 1.93),
    },
    "StarVLA-FAST": {
        "EAGLE3": ([2.46, 2.31, 2.88, 2.81], [1.26, 1.24, 1.41, 1.40], 2.61, 1.33),
        "P-EAGLE": ([2.13, 2.03, 2.44, 2.38], [1.30, 1.30, 1.40, 1.42], 2.25, 1.36),
        "DFlash": ([1.51, 1.44, 1.63, 1.61], [1.10, 1.09, 1.14, 1.17], 1.55, 1.12),
        "DFlash2": ([1.68, 1.63, 1.87, 1.83], [1.13, 1.12, 1.17, 1.22], 1.75, 1.16),
        "DSpark": ([2.13, 2.00, 2.46, 2.43], [1.31, 1.30, 1.46, 1.48], 2.26, 1.39),
        "VSpark": ([2.60, 2.43, 2.94, 2.91], [1.56, 1.56, 1.64, 1.68], 2.72, 1.61),
    },
    "ActionCodec": {
        "EAGLE3": ([3.92, 3.76, 4.26, 4.01], [1.58, 1.57, 1.73, 1.65], 3.99, 1.63),
        "P-EAGLE": ([3.73, 3.66, 4.09, 3.83], [1.88, 1.90, 2.07, 1.96], 3.83, 1.95),
        "DFlash": ([3.30, 3.16, 3.64, 3.35], [1.84, 1.83, 2.02, 1.90], 3.36, 1.90),
        "DFlash2": ([3.59, 3.44, 3.81, 3.56], [1.80, 1.75, 1.91, 1.77], 3.60, 1.81),
        "DSpark": ([3.60, 3.52, 3.94, 3.69], [1.85, 1.91, 2.04, 1.91], 3.69, 1.93),
        "VSpark": ([4.13, 4.03, 4.44, 4.23], [2.04, 2.06, 2.21, 2.08], 4.21, 2.10),
    },
    "G0.5": {
        "EAGLE3": ([2.32, 2.33, 2.60, 2.33], [0.83, 0.80, 0.88, 0.80], 2.39, 0.83),
        "P-EAGLE": ([2.37, 2.35, 2.64, 2.36], [1.08, 1.06, 1.16, 1.06], 2.43, 1.09),
        "DFlash": ([1.75, 1.73, 1.90, 1.73], [0.95, 0.93, 1.03, 0.94], 1.78, 0.96),
        "DFlash2": ([1.94, 1.91, 2.13, 1.92], [0.89, 0.87, 0.93, 0.88], 1.98, 0.89),
        "DSpark": ([2.26, 2.27, 2.54, 2.26], [1.03, 1.03, 1.10, 1.06], 2.33, 1.05),
        "VSpark": ([2.95, 2.95, 3.15, 2.92], [1.27, 1.27, 1.30, 1.27], 2.99, 1.28),
    },
}

# 표 3 (LIBERO-Plus 4 suites, LIBERO-Pro 4 suites, Avg = 8 suites 평균)
# 순서: Plus Goal, Spatial, Object, Long, Pro Goal, Spatial, Object, Long
T3 = {
    "pi0-FAST": {
        "EAGLE3": ([2.74, 2.78, 2.94, 2.78, 2.95, 3.03, 3.29, 3.06], [1.00, 1.01, 1.07, 1.00, 1.45, 1.48, 1.58, 1.46], 2.95, 1.26),
        "P-EAGLE": ([2.60, 2.63, 2.77, 2.64, 2.79, 2.88, 3.09, 2.89], [1.40, 1.39, 1.49, 1.33, 1.69, 1.76, 1.81, 1.75], 2.79, 1.58),
        "DFlash": ([1.92, 1.94, 1.92, 1.89, 2.25, 2.36, 2.39, 2.26], [1.23, 1.25, 1.25, 1.18, 1.68, 1.72, 1.73, 1.66], 2.12, 1.46),
        "DFlash2": ([2.46, 2.53, 2.62, 2.44, 2.73, 2.96, 3.07, 2.80], [1.32, 1.37, 1.42, 1.29, 1.68, 1.78, 1.80, 1.67], 2.70, 1.54),
        "DSpark": ([2.70, 2.70, 2.89, 2.75, 2.94, 3.02, 3.28, 3.06], [1.46, 1.46, 1.54, 1.44, 1.82, 1.85, 1.93, 1.83], 2.92, 1.67),
        "VSpark": ([3.32, 3.34, 3.49, 3.33, 3.62, 3.78, 3.92, 3.71], [1.78, 1.75, 1.78, 1.65, 2.14, 2.24, 2.21, 2.13], 3.56, 1.96),
    },
    "StarVLA-FAST": {
        "EAGLE3": ([2.10, 2.03, 2.39, 2.27, 2.25, 2.10, 2.91, 2.54], [1.12, 1.07, 1.24, 1.17, 1.44, 1.36, 1.71, 1.56], 2.32, 1.33),
        "P-EAGLE": ([1.87, 1.82, 2.08, 1.99, 1.99, 1.88, 2.46, 2.19], [1.18, 1.13, 1.27, 1.24, 1.51, 1.42, 1.74, 1.60], 2.04, 1.39),
        "DFlash": ([1.34, 1.32, 1.44, 1.41, 1.41, 1.35, 1.57, 1.49], [1.01, 0.98, 1.06, 1.03, 1.25, 1.21, 1.36, 1.31], 1.42, 1.15),
        "DFlash2": ([1.51, 1.49, 1.67, 1.61, 1.57, 1.52, 1.80, 1.69], [1.00, 0.96, 1.07, 1.04, 1.25, 1.21, 1.38, 1.33], 1.61, 1.16),
        "DSpark": ([1.79, 1.72, 2.00, 1.95, 1.98, 1.85, 2.50, 2.22], [1.14, 1.09, 1.23, 1.21, 1.50, 1.43, 1.77, 1.61], 2.00, 1.37),
        "VSpark": ([2.22, 2.13, 2.48, 2.39, 2.39, 2.22, 2.95, 2.65], [1.36, 1.31, 1.48, 1.42, 1.75, 1.65, 1.98, 1.87], 2.43, 1.60),
    },
    "ActionCodec": {
        "EAGLE3": ([2.95, 2.80, 3.24, 3.17, 3.36, 3.50, 4.24, 3.66], [1.35, 1.31, 1.51, 1.46, 1.49, 1.60, 1.81, 1.57], 3.36, 1.51),
        "P-EAGLE": ([2.85, 2.67, 3.09, 3.04, 3.20, 3.35, 4.12, 3.50], [1.64, 1.61, 1.78, 1.72, 1.70, 1.89, 2.11, 1.81], 3.23, 1.78),
        "DFlash": ([2.39, 2.25, 2.57, 2.56, 2.75, 2.91, 3.55, 3.03], [1.53, 1.47, 1.68, 1.64, 1.65, 1.79, 2.03, 1.78], 2.75, 1.70),
        "DFlash2": ([2.55, 2.39, 2.78, 2.75, 2.93, 3.10, 3.81, 3.25], [1.48, 1.38, 1.56, 1.55, 1.55, 1.69, 1.94, 1.68], 2.95, 1.60),
        "DSpark": ([2.72, 2.50, 2.91, 2.86, 3.07, 3.24, 3.93, 3.39], [1.63, 1.48, 1.70, 1.70, 1.69, 1.83, 2.08, 1.81], 3.08, 1.74),
        "VSpark": ([3.37, 3.15, 3.57, 3.55, 3.76, 3.84, 4.58, 4.06], [1.89, 1.81, 1.99, 1.98, 2.01, 2.14, 2.35, 2.12], 3.73, 2.04),
    },
    "G0.5": {
        "EAGLE3": ([1.98, 2.02, 2.22, 2.08, 2.21, 2.25, 2.57, 2.29], [0.71, 0.74, 0.80, 0.76, 0.87, 0.88, 0.97, 0.89], 2.20, 0.83),
        "P-EAGLE": ([1.92, 1.95, 2.09, 2.01, 2.19, 2.25, 2.56, 2.27], [0.90, 0.97, 1.00, 0.98, 1.10, 1.12, 1.24, 1.14], 2.15, 1.06),
        "DFlash": ([1.55, 1.56, 1.66, 1.61, 1.64, 1.66, 1.83, 1.68], [0.85, 0.89, 0.95, 0.92, 0.99, 1.00, 1.08, 1.02], 1.65, 0.96),
        "DFlash2": ([1.65, 1.65, 1.79, 1.72, 1.80, 1.83, 2.06, 1.86], [0.79, 0.80, 0.85, 0.83, 0.92, 0.92, 1.01, 0.94], 1.79, 0.88),
        "DSpark": ([1.95, 2.01, 2.21, 2.07, 2.11, 2.16, 2.46, 2.18], [0.90, 0.95, 1.03, 0.97, 1.08, 1.10, 1.21, 1.11], 2.14, 1.04),
        "VSpark": ([2.61, 2.66, 2.80, 2.70, 2.72, 2.80, 2.98, 2.81], [1.19, 1.21, 1.24, 1.25, 1.32, 1.35, 1.39, 1.35], 2.76, 1.29),
    },
}

# 표 2 (SimplerEnv WidowX, StarVLA-FAST): 과제 Spoon, Carrot, Stack, Eggplant, 그리고 Avg 칸
T2_TASKS = ["Spoon", "Carrot", "Stack", "Eggplant"]
T2 = {
    "EAGLE3": ([3.89, 3.49, 3.68, 3.84], [1.69, 1.59, 1.64, 1.66], 3.72, 1.65),
    "P-EAGLE": ([3.37, 2.95, 3.08, 3.26], [1.70, 1.53, 1.57, 1.62], 3.16, 1.61),
    "DFlash": ([2.85, 2.52, 2.59, 2.62], [1.51, 1.40, 1.41, 1.38], 2.64, 1.42),
    "DFlash2": ([3.21, 2.89, 2.97, 3.00], [1.62, 1.51, 1.51, 1.50], 3.02, 1.54),
    "DSpark": ([3.70, 3.31, 3.39, 3.50], [1.83, 1.72, 1.71, 1.74], 3.48, 1.75),
    "VSpark": ([4.01, 3.87, 3.81, 3.94], [1.99, 1.96, 1.93, 1.92], 3.90, 1.95),
}

# 표 4 (π0-FAST, DSpark 초안, LIBERO 4 suite 평균): (distance, relaxed ratio %, tau, speedup, SR %)
T4 = [
    (None, 0.00, 3.16, 1.60, 91.7),
    (0.011, 2.05, 3.18, 1.62, 91.7),
    (0.012, 12.70, 3.31, 1.67, 91.0),
    (0.013, 23.72, 3.47, 1.74, 89.2),
    (0.015, 35.79, 3.70, 1.82, 83.8),
    (0.020, 45.97, 4.02, 1.90, 71.4),
    (0.025, 47.56, 4.08, 1.93, 70.5),
]

# 표 5 (다음 토큰 일치 %): reject pos -> (pi0 draft, pi0 target, pi0 delta, AC draft, AC target, AC delta)
T5 = {
    "1": (24.70, 39.50, 14.80, 35.55, 54.24, 18.69),
    "2": (35.20, 45.96, 10.76, 38.54, 60.05, 21.51),
    "3": (28.20, 55.12, 26.92, 45.14, 65.48, 20.35),
    "4": (29.26, 53.40, 24.14, 32.68, 47.75, 15.07),
    "5-7": (46.57, 56.42, 9.85, 36.22, 54.35, 18.13),
    "Overall": (27.05, 45.38, 18.33, 36.35, 54.88, 18.52),
}

# 표 7 (구성요소 추가, τ, speedup): DSpark -> +CR -> +FA -> +Noise(VSpark)
T7 = {
    "pi0-FAST": [(3.16, 1.60), (3.22, 1.69), (3.87, 1.90), (3.94, 1.93)],
    "ActionCodec": [(3.69, 1.93), (4.04, 2.01), (4.10, 2.07), (4.21, 2.10)],
}

# 표 B.7 (LIBERO, 5 seeds BF16 AR/VSpark 평균과 95% t 구간, FP32 seed 0, 짝지은 결과 일치율 %)
# (suite, AR_bf16, AR_lo, AR_hi, VS_bf16, VS_lo, VS_hi, agree_bf16, AR_fp32, VS_fp32, agree_fp32)
B7 = {
    "pi0-FAST": [
        ("Goal", 91.64, 89.48, 93.80, 91.40, 90.16, 92.64, 88.56, 91.00, 91.00, 100.00),
        ("Spatial", 94.92, 93.89, 95.95, 95.48, 94.37, 96.59, 94.80, 95.80, 95.80, 100.00),
        ("Object", 97.40, 97.15, 97.65, 96.68, 96.35, 97.01, 95.68, 96.40, 96.40, 100.00),
        ("Long", 86.87, 86.11, 87.63, 85.33, 82.34, 88.33, 82.47, 85.20, 85.20, 100.00),
        ("Avg", 92.71, 92.17, 93.24, 92.22, 91.64, 92.80, 90.38, 92.10, 92.10, 100.00),
    ],
    "ActionCodec": [
        ("Goal", 88.12, 85.98, 90.26, 88.16, 86.54, 89.78, 85.48, 80.80, 80.80, 100.00),
        ("Spatial", 83.96, 82.89, 85.03, 84.16, 81.03, 87.29, 81.16, 79.80, 79.80, 100.00),
        ("Object", 93.24, 93.13, 93.35, 92.76, 92.44, 93.08, 93.04, 93.40, 93.40, 100.00),
        ("Long", 77.88, 76.80, 78.96, 79.36, 77.01, 81.71, 74.76, 75.00, 75.00, 100.00),
        ("Avg", 85.80, 85.06, 86.54, 86.11, 84.68, 87.54, 83.61, 82.25, 82.25, 100.00),
    ],
}

# 표 A.2 와 A.1, 표 A.6 (translation mm, rotation deg) 의 σ
A2_CHUNKS = 27548
EVAL_EPISODES = {"LIBERO": 4 * 10 * 50, "LIBERO-Plus": 2591 + 2402 + 2518 + 2519, "LIBERO-Pro": 4 * 5 * 10 * 50, "SimplerEnv": 4 * 24}
NOISE_SIGMA_TABLE_A6 = {  # [sx, sy, sz, srx, sry, srz]
    "pi0-FAST": [3.88, 7.10, 3.08, 0.24, 0.66, 0.90],
    "StarVLA-FAST": [4.88, 6.49, 6.17, 0.42, 1.36, 1.67],
    "ActionCodec": [2.94, 2.73, 3.31, 0.33, 0.49, 0.34],
    "G0.5": [1.17, 0.87, 2.30, 0.22, 0.39, 0.38],
}
CONFIG_FILES = {"pi0-FAST": "pi0_fast", "StarVLA-FAST": "starvla", "ActionCodec": "actioncodec"}

results = {"checks": {}}
fail = []


def check(key, ok, detail):
    results["checks"][key] = {"ok": bool(ok), "detail": detail}
    if not ok:
        fail.append(key)


def mean(x):
    return float(np.mean(x))


def main():
    # V1 표 1 Avg = 4 suite 평균 (반올림 허용 0.011)
    worst = 0.0
    for m in MODELS:
        for meth in METHODS:
            tau, spd, ta, sa = T1[m][meth]
            worst = max(worst, abs(mean(tau) - ta), abs(mean(spd) - sa))
    check("V1_table1_avg", worst <= 0.011, f"표 1 Avg 열과 4 suite 산술평균의 최대 차이 {worst:.4f}")

    # V2 표 3 Avg = 8 suite 평균
    worst = 0.0
    for m in MODELS:
        for meth in METHODS:
            tau, spd, ta, sa = T3[m][meth]
            worst = max(worst, abs(mean(tau) - ta), abs(mean(spd) - sa))
    check("V2_table3_avg", worst <= 0.011, f"표 3 Avg 열과 8 suite 평균의 최대 차이 {worst:.4f}")

    # V3 표 2 Avg (과제 4개 단순 평균과 비교)
    t2 = {k: (mean(v[0]), mean(v[1]), v[2], v[3]) for k, v in T2.items()}
    diffs = {k: round(a - b, 4) for k, (a, _, b, _) in t2.items()}
    check("V3_table2_avg", max(abs(d) for d in diffs.values()) <= 0.011,
          {"과제 단순 평균 τ 대 보고 Avg 차이": diffs,
           "VSpark 단순 평균 τ": round(t2["VSpark"][0], 4), "보고 값": 3.90})

    # V4 표 1: 모든 모델·suite에서 VSpark가 τ, speedup 최고 (2위와의 격차 포함)
    bad = []
    margins = []
    for m in MODELS:
        for s in range(4):
            for idx, name in ((0, "tau"), (1, "spd")):
                vals = {meth: T1[m][meth][idx][s] for meth in METHODS}
                best_other = max(v for k, v in vals.items() if k != "VSpark")
                margins.append(round(vals["VSpark"] - best_other, 3))
                if vals["VSpark"] <= best_other:
                    bad.append((m, SUITES[s], name))
    check("V4_table1_best_everywhere", not bad, {"VSpark가 이기지 못한 셀": bad,
                                              "최소 격차": min(margins)})

    # V5 표 3: 4 모델 x 8 suite 모두 VSpark 최고
    bad = []
    for m in MODELS:
        for s in range(8):
            for idx in (0, 1):
                vals = {meth: T3[m][meth][idx][s] for meth in METHODS}
                if vals["VSpark"] <= max(v for k, v in vals.items() if k != "VSpark"):
                    bad.append((m, s, idx))
    check("V5_table3_best_everywhere", not bad, {"VSpark가 이기지 못한 셀": bad})

    # V6 표 2: 과제마다 VSpark 최고
    bad = []
    for t in range(4):
        for idx in (0, 1):
            vals = {k: v[idx][t] for k, v in T2.items()}
            if vals["VSpark"] <= max(v for k, v in vals.items() if k != "VSpark"):
                bad.append((T2_TASKS[t], idx))
    check("V6_table2_best_per_task", not bad, {"실패": bad})

    # V7 본문 수치 대 표 1 (DSpark → VSpark τ)
    text_pairs = {"pi0-FAST": (3.16, 3.94), "StarVLA-FAST": (2.26, 2.72), "ActionCodec": (3.69, 4.21), "G0.5": (2.33, 2.99)}
    ok = all(abs(T1[m]["DSpark"][2] - a) < 1e-9 and abs(T1[m]["VSpark"][2] - b) < 1e-9 for m, (a, b) in text_pairs.items())
    check("V7_text_vs_table1", ok, "본문 3.16→3.94, 2.26→2.72, 3.69→4.21, 2.33→2.99와 표 1 Avg 일치")

    # V8 본문 표 3 수치 (π0 DSpark 2.92 → VSpark 3.56, AC 3.08 → 3.73)
    ok = (abs(T3["pi0-FAST"]["DSpark"][2] - 2.92) < 1e-9 and abs(T3["pi0-FAST"]["VSpark"][2] - 3.56) < 1e-9
          and abs(T3["ActionCodec"]["DSpark"][2] - 3.08) < 1e-9 and abs(T3["ActionCodec"]["VSpark"][2] - 3.73) < 1e-9)
    check("V8_text_vs_table3", ok, "본문 2.92→3.56(π0), 3.08→3.73(ActionCodec)와 표 3 일치")

    # V9 사이클 비용 c = τ / speedup (AR 토큰 단위). 속도 향상의 분해: spd_v/spd_d = (τ_v/τ_d)/(c_v/c_d)
    decomp = {}
    for m in MODELS:
        dv, dd = T1[m]["VSpark"], T1[m]["DSpark"]
        c_v, c_d = dv[2] / dv[3], dd[2] / dd[3]
        decomp[m] = dict(
            tau_gain_pct=round(100 * (dv[2] / dd[2] - 1), 1),
            cycle_cost_change_pct=round(100 * (c_v / c_d - 1), 1),
            speedup_gain_pct=round(100 * (dv[3] / dd[3] - 1), 1),
            cycle_cost_VSpark=round(c_v, 3), cycle_cost_DSpark=round(c_d, 3),
        )
    implied = {meth: {m: round(T1[m][meth][2] / T1[m][meth][3], 2) for m in MODELS} for meth in METHODS}
    results["implied_cycle_cost"] = implied
    results["decomposition_VSpark_vs_DSpark"] = decomp
    check("V9_speedup_decomposition", all(abs((1 + d["tau_gain_pct"] / 100) / (1 + d["cycle_cost_change_pct"] / 100) - 1 - d["speedup_gain_pct"] / 100) < 0.005 for d in decomp.values()),
          "spd 비 = τ 비 ÷ 사이클 비용 비 (항등식)")

    # V10 표 5 Δ = target − draft, 그리고 Overall 차이
    bad = []
    for k, (d1, t1, dl1, d2, t2_, dl2) in T5.items():
        if abs((t1 - d1) - dl1) > 0.011 or abs((t2_ - d2) - dl2) > 0.011:
            bad.append(k)
    check("V10_table5_delta", not bad, {"Δ 불일치 행": bad, "Overall 차이 π0": round(45.38 - 27.05, 2), "AC": round(54.88 - 36.35, 2)})

    # V11 표 7: 단계별 단조 증가, 마지막 단계 = 표 1 VSpark Avg
    chk = {}
    for m, steps in T7.items():
        taus = [t for t, _ in steps]
        spds = [s for _, s in steps]
        chk[m] = dict(monotone=bool(all(np.diff(taus) > 0) and all(np.diff(spds) > 0)),
                      final_matches_table1=bool(abs(steps[-1][0] - T1[m if m != "pi0-FAST" else "pi0-FAST"]["VSpark"][2]) < 1e-9
                                                and abs(steps[-1][1] - T1[m]["VSpark"][3]) < 1e-9),
                      tau_increments=[round(b - a, 2) for a, b in zip(taus[:-1], taus[1:])],
                      tau_increment_fraction_CR_FA_noise=None)
        incs = np.diff(taus)
        chk[m]["tau_increments"] = [round(float(x), 2) for x in incs]
        chk[m]["largest_step"] = ["CR", "FA", "Noise"][int(np.argmax(incs))]
    results["table7"] = chk
    check("V11_table7", all(v["monotone"] and v["final_matches_table1"] for v in chk.values()), chk)

    # V12 학습 샘플 수: 표 A.2 = 27,548 과 configs/*.json 의 target_chunks 합
    cfg = {}
    for m, f in CONFIG_FILES.items():
        p = REPO / "configs" / f"{f}.json"
        if p.exists():
            d = json.loads(p.read_text())
            per_suite = {s: sum(d["target_chunks"][s].values()) for s in d["target_chunks"]}
            cfg[m] = dict(total=sum(per_suite.values()), per_suite=per_suite)
    results["config_chunks"] = cfg
    check("V12_chunk_count", bool(cfg) and all(v["total"] == A2_CHUNKS for v in cfg.values()),
          {m: v["total"] for m, v in cfg.items()} or "repo/ 없음: 건너뜀")

    # V13 표 A.6 대 configs/*.json 의 noise_covariance (σ = sqrt(대각)).
    # 단위 환산 상수 50 mm/단위, 28.65°/단위(0.5 rad)로 곱한 값을 표 자릿수(소수 둘째 자리)로 반올림해 표 값과 맞는지 센다.
    scale, hits, total = {}, 0, 0
    for m, f in CONFIG_FILES.items():
        p = REPO / "configs" / f"{f}.json"
        if not p.exists():
            continue
        cov = np.array(json.loads(p.read_text())["noise_covariance"])
        sig = np.sqrt(np.diag(cov))
        paper = np.array(NOISE_SIGMA_TABLE_A6[m])
        pred = np.concatenate((sig[:3] * 50.0, sig[3:] * np.degrees(0.5)))
        match = [bool(abs(round(float(q), 2) - x) < 1e-9) for q, x in zip(pred, paper)]
        hits += sum(match)
        total += len(match)
        scale[m] = dict(predicted=[round(float(q), 2) for q in pred], paper=paper.tolist(), exact_2dp_matches=sum(match))
    results["noise_scale"] = scale
    check("V13_noise_units", bool(scale) and hits == total,
          f"표 A.6 σ와 config σ × (50 mm, 28.65°) 의 소수 둘째 자리 일치 {hits}/{total}" if scale else "repo/ 없음: 건너뜀")

    # V14 표 A.1 계수
    check("V14_eval_counts", EVAL_EPISODES["LIBERO-Plus"] == 10030 and EVAL_EPISODES["LIBERO"] == 2000 and EVAL_EPISODES["LIBERO-Pro"] == 10000 and EVAL_EPISODES["SimplerEnv"] == 96,
          EVAL_EPISODES)

    # V15 표 B.7 평균 = 4 suite 평균 (BF16 AR/VSpark/일치율, FP32)
    b7 = {}
    for m, rows in B7.items():
        suites = rows[:4]
        avg = rows[4]
        errs = {
            "AR_bf16": round(mean([r[1] for r in suites]) - avg[1], 3),
            "VS_bf16": round(mean([r[4] for r in suites]) - avg[4], 3),
            "agree_bf16": round(mean([r[7] for r in suites]) - avg[7], 3),
            "AR_fp32": round(mean([r[8] for r in suites]) - avg[8], 3),
            "VS_fp32": round(mean([r[9] for r in suites]) - avg[9], 3),
        }
        b7[m] = errs
    check("V15_table_b7_avg", all(abs(x) <= 0.011 for v in b7.values() for x in v.values()), b7)

    # V16 표 B.7: 신뢰구간에서 표준편차 역산(n = 5 시드, t(0.975, 4) = 2.776), Welch 차이 검정
    t975 = stats.t.ppf(0.975, 4)
    tests = {}
    for m, rows in B7.items():
        for r in rows:
            name, ar, arlo, arhi, vs, vslo, vshi = r[:7]
            s_ar = (arhi - arlo) / 2 * np.sqrt(5) / t975
            s_vs = (vshi - vslo) / 2 * np.sqrt(5) / t975
            tests.setdefault(m, {})[name] = dict(sd_AR=round(float(s_ar), 3), sd_VSpark=round(float(s_vs), 3))
        avg = rows[4]
        diff = avg[4] - avg[1]
        a = (avg[3] - avg[2]) / 2 * np.sqrt(5) / t975
        b = (avg[6] - avg[5]) / 2 * np.sqrt(5) / t975
        va, vb = a * a / 5, b * b / 5
        se = np.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / 4 + vb ** 2 / 4)
        tcrit = stats.t.ppf(0.975, df)
        tests[m]["Avg_difference"] = dict(diff_pp=round(float(diff), 2), se=round(float(se), 3), df=round(float(df), 1),
                                          ci95=[round(float(diff - tcrit * se), 2), round(float(diff + tcrit * se), 2)],
                                          p_welch=round(float(2 * stats.t.sf(abs(diff / se), df)), 3),
                                          note="두 방법 독립 가정 (쌍 정보 없음)")
    results["table_b7_tests"] = tests
    check("V16_table_b7_ci", True, tests)

    # V17 표 B.7: FP32에서 AR과 VSpark 완전 일치, BF16에서 AR 자체의 정밀도 효과와 비교
    results["precision_effect"] = dict(
        pi0_AR_bf16_minus_fp32=round(92.71 - 92.10, 2),
        AC_AR_bf16_minus_fp32=round(85.80 - 82.25, 2),
        pi0_VSpark_minus_AR_bf16=round(92.22 - 92.71, 2),
        AC_VSpark_minus_AR_bf16=round(86.11 - 85.80, 2),
        note="FP32 값은 시드 0 한 번 실행, BF16 값은 시드 0–4 평균. 서로 다른 실행 조건이므로 엄밀한 비교가 아님",
    )
    check("V17_fp32_identical", all(r[8] == r[9] and r[10] == 100.0 for m, rows in B7.items() for r in rows),
          "FP32에서 AR과 VSpark SR 동일, 일치율 100%")

    # V18 표 4: Wilson 구간 (n = 2000, 표 A.1 LIBERO 평균 가정) 과 두 비율 z 검정 (비쌍)
    n = EVAL_EPISODES["LIBERO"]

    def wilson(p, n, z=1.96):
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return centre - half, centre + half

    rows4 = []
    for dist, ratio, tau, spd, sr in T4:
        lo, hi = wilson(sr / 100, n)
        rows4.append(dict(distance=dist, relaxed_pct=ratio, tau=tau, speedup=spd, sr=sr,
                          sr_wilson95=[round(100 * lo, 1), round(100 * hi, 1)]))
    base = T4[0][4] / 100

    def ztest(p1, p2):
        pp = (p1 + p2) / 2
        se = np.sqrt(pp * (1 - pp) * (2 / n))
        z = (p1 - p2) / se
        return float(2 * stats.norm.sf(abs(z)))

    pvals = {f"SR {sr} vs 91.7 (speedup {spd})": round(ztest(0.917, sr / 100), 6) for _, _, _, spd, sr in T4[1:]}
    slope = np.polyfit([r[3] for r in T4], [r[4] for r in T4], 1)[0]
    results["table4"] = dict(rows=rows4, p_vs_baseline_unpaired=pvals, sr_drop_per_speedup_unit=round(float(slope), 1),
                             note="n = 2000 가정 (표 A.1), 쌍 정보 없음, 비쌍 근사")
    check("V18_table4_tests", True, {"SR 70.5 vs 91.7 p": pvals["SR 70.5 vs 91.7 (speedup 1.93)"]})

    # V19 스위트별 부호 검정 (VSpark > DSpark 방향): 8 suite 전부면 p = 2 * (1/2)^8 (suite가 독립이라는 가정)
    signs = {}
    for m in MODELS:
        wins3 = sum(T3[m]["VSpark"][0][s] > T3[m]["DSpark"][0][s] for s in range(8))
        wins1 = sum(T1[m]["VSpark"][0][s] > T1[m]["DSpark"][0][s] for s in range(4))
        signs[m] = dict(tau_table3_wins=f"{wins3}/8", tau_table1_wins=f"{wins1}/4")
    results["sign_counts"] = signs
    check("V19_sign_counts", True, signs)

    # V20 본문 주장: 표 3 Avg와 본문 1.96, 1.60, 2.04, 1.29
    txt3 = {"pi0-FAST": 1.96, "StarVLA-FAST": 1.60, "ActionCodec": 2.04, "G0.5": 1.29}
    check("V20_text_table3_speedups", all(abs(T3[m]["VSpark"][3] - v) < 1e-9 for m, v in txt3.items()), "본문 평균 속도 향상과 표 3 일치")

    # V21 표 2 본문: 1.95×, 3.90, 3.48/1.75 (DSpark)
    check("V21_text_table2", abs(T2["VSpark"][3] - 1.95) < 1e-9 and abs(T2["DSpark"][2] - 3.48) < 1e-9, "본문 일치")

    # V22 표 1·B.7 AR 시드 7 SR이 본문에 없음: 표 4의 91.7(DSpark, π0) 와 표 B.7 AR 92.71(5 시드 평균)의 차
    results["sr_consistency"] = dict(table4_pi0_SR=91.7, tableB7_pi0_AR_bf16_avg=92.71, tableB7_pi0_AR_fp32_seed0=92.10,
                                     note="표 4의 SR이 어느 실행인지 본문에 없음 (표 B.7과 1.0pp 차이)")
    check("V22_sr_source", True, results["sr_consistency"])

    out = HERE / "recompute_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("checks:", len(results["checks"]), "failed:", fail)
    for k, v in results["checks"].items():
        print(k, "OK" if v["ok"] else "FAIL")
    print("decomposition:", json.dumps(decomp, ensure_ascii=False))
    print("table_b7 avg diff:", json.dumps({m: tests[m]["Avg_difference"] for m in tests}, ensure_ascii=False))
    print("table4 p:", json.dumps(pvals, ensure_ascii=False))
    print("noise_scale:", json.dumps({m: v["exact_2dp_matches"] for m, v in scale.items()}, ensure_ascii=False))
    print("config_chunks:", json.dumps({m: v["total"] for m, v in cfg.items()}))


if __name__ == "__main__":
    main()
