"""WorldVLA (arXiv:2506.21539 v1) 표 재계산과 일관성 검사 (numpy, scipy).
실행: python3 analysis/recompute.py  -> analysis/recompute_results.json
값은 paper/paper.txt(pdftotext -layout)에서 옮겼다. 원자료(rollout 기록)가 없으므로
모든 검사는 '보고된 값끼리' 맞는지와, 50회 롤아웃 기준 통계적 불확실성을 본다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
SUITES = ["Spatial", "Object", "Goal", "Long"]
N_TRIAL = 50  # 과제당 롤아웃 수 (논문 4.1, Metrics)
res = {}

# ---------------------------------------------------------------- 표 2 평균 재계산
T2 = {
    "Diffusion Policy": [78.3, 92.5, 68.3, 50.5, 72.4],
    "Octo": [78.9, 85.7, 84.6, 51.1, 75.1],
    "DiT Policy": [84.2, 96.3, 85.4, 63.8, 82.4],
    "OpenVLA-OFT": [96.9, 98.1, 95.5, 91.1, 95.4],
    "OpenVLA": [84.7, 88.4, 79.2, 53.7, 76.5],
    "WorldVLA 256": [85.6, 89.0, 82.6, 59.0, 79.1],
    "WorldVLA 512": [87.6, 96.2, 83.4, 60.0, 81.8],
}
res["table2_mean_check"] = {k: {"mean_of_4": round(float(np.mean(v[:4])), 2), "reported": v[4],
                                 "match": bool(abs(np.mean(v[:4]) - v[4]) <= 0.05 + 1e-9)} for k, v in T2.items()}

# ---------------------------------------------------------------- 표 3 (ablation, 평균 재계산)
T3 = {  # index: (Goal, Object, Spatial, Long, Average_reported)
    1: (67.3, 82.9, 77.8, 23.0, 62.8),
    2: (73.1, 88.0, 80.2, 27.3, 67.2),
    3: (79.6, 82.9, 36.7, 16.9, 54.0),
    4: (84.4, 90.9, 81.8, 49.3, 76.6),
    5: (85.1, 90.9, 84.0, 52.4, 78.1),
}
res["table3_mean_check"] = {k: {"mean_of_4": round(float(np.mean(v[:4])), 2), "reported": v[4],
                                 "match": bool(abs(np.mean(v[:4]) - v[4]) <= 0.05 + 1e-9)} for k, v in T3.items()}

# ---------------------------------------------------------------- 표 4 FVD 상대 변화
T4 = {"10 frames": (250.0, 255.1), "50 frames": (718.6, 674.1)}  # (World Model, Action World Model) FVD
res["table4_fvd_change_pct"] = {k: round(100 * (awm - wm) / wm, 2) for k, (wm, awm) in T4.items()}
res["intro_fvd_claim_10pct_supported"] = bool(
    res["table4_fvd_change_pct"]["50 frames"] <= -10.0)

# ---------------------------------------------------------------- 표 6 평균 재계산
T6 = {"w/o WM pretrain": (67.3, 82.9, 77.8, 23.0, 62.8), "w/ WM pretrain": (73.1, 84.0, 79.8, 30.2, 66.8)}
res["table6_mean_check"] = {k: {"mean_of_4": round(float(np.mean(v[:4])), 2), "reported": v[4]} for k, v in T6.items()}

# ---------------------------------------------------------------- 불확실성: 롤아웃 50회 이항 표준오차
def se_p(p):
    return math.sqrt(p * (1 - p) / N_TRIAL)

res["per_task_95CI_halfwidth_pp"] = {p: round(196 * se_p(p), 1) for p in [0.5, 0.7, 0.85, 0.95]}

def two_prop(p1, p2, n1, n2):
    """두 비율 차이의 정규근사 z검정 (양측)."""
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return round(z, 2), round(2 * (1 - stats.norm.cdf(abs(z))), 4)

def pct(x):
    return x / 100.0

# 평균 SR은 4 과제 x 50회 = 200회. 과제별 평균을 전체 시도 비율로 근사한다.
cmp = {
    "T3 row1 vs row2 (WM 추가, 체킹 없음) avg": (2, 1),
    "T3 row4 vs row5 (WM 추가, 체킹+마스크) avg": (5, 4),
    "T3 row3 vs row4 (마스크 효과, 체킹) avg": (4, 3),
    "T2 WorldVLA256 vs OpenVLA avg (교차 논문 비교)": None,
}
tests = {}
for name, pair in cmp.items():
    if pair is None:
        a, b = 79.1, 76.5
    else:
        a, b = T3[pair[0]][4], T3[pair[1]][4]
    z, p = two_prop(pct(a), pct(b), 200, 200)
    tests[name] = {"diff_pp": round(a - b, 2), "z": z, "p_two_sided": p}
res["significance_avg_200_trials"] = tests

# 마스크 효과는 과제별로도 본다 (Spatial, Long)
for suite, idx in [("Spatial", 2), ("Long", 3)]:
    a, b = T3[4][idx] / 100, T3[3][idx] / 100
    z, p = two_prop(a, b, N_TRIAL, N_TRIAL)
    res.setdefault("mask_effect_per_suite", {})[suite] = {"diff_pp": round(100 * (a - b), 1), "z": z, "p": p}

# WM 추가 효과 과제별 (row2 vs row1)
for suite, idx in [("Goal", 0), ("Object", 1), ("Spatial", 2), ("Long", 3)]:
    a, b = T3[2][idx] / 100, T3[1][idx] / 100
    z, p = two_prop(a, b, N_TRIAL, N_TRIAL)
    res.setdefault("wm_effect_no_chunk_per_suite", {})[suite] = {"diff_pp": round(100 * (a - b), 1), "z": z, "p": p}

# ---------------------------------------------------------------- 본문 주장 대조
res["claims"] = {
    "intro: 'same backbone action model +4% (grasping SR)'": {
        "table3_row2_minus_row1_avg": round(T3[2][4] - T3[1][4], 2)},
    "intro: 'naive chunking decreases SR 10-50%'": {
        "row3_minus_row1_per_suite": {s: round(T3[3][i] - T3[1][i], 1) for i, s in enumerate(["Goal", "Object", "Spatial", "Long"])},
        "note": "Goal은 오히려 +12.3pp, Object는 0. 감소는 Spatial(-41.1)과 Long(-6.1)에 집중"},
    "intro: 'mask gives 4% to 23% improvement'": {
        "row4_minus_row3_avg": round(T3[4][4] - T3[3][4], 2)},
    "Table 5 default (2 frames, w/ chunking) = 84.4": {
        "matches_row4_Goal_SR": T3[4][0] == 84.4, "matches_row5_avg": T3[5][4] == 84.4},
}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
