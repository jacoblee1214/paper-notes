"""논문 표(arXiv:2606.02486 v2)의 수치 재계산과 일관성 검사.

실행: python3 analysis/recompute.py   → analysis/recompute_results.json
의존: numpy, scipy (수 초)

검사 항목(V1–V12)은 analysis/03_analysis.md 에 설명이 있다. 표 값은 paper/paper.txt 에서 옮겼다.
시뮬레이션 셀은 "5 시드 x 시드당 100 롤아웃"(부록 C.5)을 가정해 이항 모형으로 비교한다.
"""
import json
import math
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
R = {}


# ---------------------------------------------------------------- 표 값
# 각 값은 (평균, ±) 쌍. 논문 표 번호는 키에 적었다.
def tab(rows):
    return {k: [tuple(x) for x in v] for k, v in rows.items()}


T1_COLS = ["cv_conv", "cv_beam", "cv_pole", "cv_roll", "ac_conv", "ac_beam", "ac_pole", "ac_roll"]
T1 = tab({
    "Open-loop VLA": [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0.4, 0.6), (0, 0), (0, 0)],
    "Retargeting VLA": [(13.7, 3.8), (22.3, 4.0), (13.3, 3.2), (7.7, 2.9), (1.0, 1.4), (10.7, 3.1), (1.0, 1.4), (0.6, 1.1)],
    "VLA + Fast Replan": [(24.0, 4.6), (32.7, 4.2), (22.3, 3.7), (16.7, 4.0), (22.3, 4.2), (31.3, 4.6), (21.7, 4.1), (16.3, 3.7)],
    "Realtime ACT": [(51.7, 5.1), (60.3, 4.7), (48.7, 4.9), (37.0, 5.3), (31.7, 4.6), (41.0, 4.9), (40.3, 4.9), (31.0, 4.6)],
    "Streaming DP": [(40.7, 5.2), (50.3, 4.8), (39.3, 4.5), (28.7, 4.6), (21.0, 4.1), (29.7, 4.6), (29.0, 4.5), (19.7, 4.0)],
    "DreamVLA": [(58.3, 4.7), (47.7, 5.1), (57.3, 5.0), (46.0, 5.2), (40.3, 4.9), (38.7, 4.9), (48.0, 5.0), (30.7, 4.6)],
    "Track-and-Intercept VLA": [(91.2, 2.6), (85.4, 3.3), (88.6, 3.0), (82.0, 3.7), (71.4, 4.3), (74.8, 4.1), (69.2, 4.4), (63.6, 4.6)],
    "LaDi-WM": [(68.4, 4.4), (61.2, 4.7), (66.8, 4.5), (55.4, 4.9), (49.6, 4.8), (47.2, 4.9), (55.8, 4.7), (38.4, 4.6)],
    "AHEAD": [(97.3, 1.5), (91.7, 2.8), (95.0, 2.3), (90.0, 3.1), (87.7, 3.3), (91.0, 3.9), (93.7, 6.1), (89.0, 0.9)],
})
T2_COLS = ["0", "10", "20", "30", "40"]
T2 = tab({
    "Open-loop VLA": [(96.4, 1.9), (0, 0), (0, 0), (0, 0), (0, 0)],
    "Retargeting VLA": [(96.8, 1.8), (40.6, 4.9), (31.2, 4.4), (22.4, 4.1), (14.6, 3.4)],
    "Realtime ACT": [(97.2, 1.6), (78.4, 3.9), (64.8, 4.4), (51.4, 4.6), (41.0, 4.7)],
    "Streaming DP": [(95.4, 2.1), (31.6, 4.4), (27.2, 4.0), (22.4, 4.1), (18.6, 3.7)],
    "DreamVLA": [(96.8, 1.8), (79.2, 3.8), (70.4, 4.3), (60.2, 4.4), (47.2, 4.1)],
    "AHEAD": [(97.6, 1.5), (97.4, 1.7), (97.0, 1.8), (96.6, 2.0), (95.4, 2.2)],
})
T3_COLS = ["AH", "BC", "MO", "PiB", "MB", "Occ", "Mid", "LMO"]
T3 = tab({
    "Open-loop VLA": [(0, 0)] * 8,
    "Retargeting VLA": [(10.6, 3.0)] + [(0, 0)] * 7,
    "VLA + Fast Replan": [(0, 0), (0, 0), (31.6, 4.4), (22.4, 4.1), (0, 0), (0, 0), (0, 0), (0, 0)],
    "Realtime ACT": [(31.2, 4.4), (22.4, 4.1), (40.6, 4.9), (41.2, 4.8), (22.4, 4.1), (11.4, 3.0), (60.2, 4.4), (31.6, 4.4)],
    "Streaming DP": [(11.4, 3.0), (41.6, 4.8), (41.4, 4.6), (22.4, 4.1), (0, 0), (0, 0), (41.6, 4.8), (30.4, 4.3)],
    "DreamVLA": [(31.2, 4.4), (49.1, 9.1), (43.5, 4.5), (41.2, 4.8), (40.6, 4.9), (0, 0), (49.4, 4.7), (45.2, 4.8)],
    "Track-and-Intercept VLA": [(62.4, 3.5), (51.2, 4.7), (38.6, 1.6), (71.8, 6.2), (29.4, 0.3), (34.8, 4.5), (27.6, 2.2), (33.6, 2.4)],
    "LaDi-WM": [(48.2, 9.7), (44.6, 1.8), (52.4, 1.7), (50.8, 3.8), (41.2, 5.7), (18.6, 3.7), (36.4, 2.6), (49.8, 1.8)],
    "AHEAD": [(94.6, 2.3), (87.4, 3.2), (95.4, 2.2), (95.8, 2.0), (87.2, 3.3), (79.4, 3.8), (94.4, 2.4), (95.6, 2.1)],
})
# 스트레스 시험(표 9–11): 시간창 100, 200, 400, 800, 1600 ms
T9 = tab({
    "Retargeting VLA": [(0, 0), (0, 0), (0, 0), (0.6, 1.1), (2.4, 2.0)],
    "VLA + Fast Replan": [(0, 0), (0, 0), (1.2, 1.5), (4.8, 2.4), (9.2, 2.8)],
    "Realtime ACT": [(0, 0), (8.6, 2.7), (22.4, 4.1), (30.4, 4.4), (35.6, 4.6)],
    "Streaming DP": [(0, 0), (0, 0), (5.4, 2.4), (12.8, 3.2), (18.6, 3.7)],
    "DreamVLA": [(0, 0), (0, 0), (8.6, 2.8), (18.4, 3.8), (26.8, 4.2)],
    "AHEAD": [(0, 0), (11.4, 3.0), (35.2, 4.5), (58.6, 4.6), (76.4, 4.1)],
})
T10 = tab({
    "Retargeting VLA": [(0, 0), (0, 0), (0, 0), (0.4, 0.9), (1.8, 1.7)],
    "VLA + Fast Replan": [(0, 0), (0, 0), (0.6, 1.1), (3.4, 2.1), (7.2, 2.7)],
    "Realtime ACT": [(0, 0), (6.4, 2.5), (16.2, 3.6), (21.6, 4.0), (27.4, 4.3)],
    "Streaming DP": [(0, 0), (0, 0), (3.8, 2.9), (9.4, 2.8), (14.9, 3.4)],
    "DreamVLA": [(0, 0), (0, 0), (5.6, 2.5), (13.8, 3.4), (21.4, 4.0)],
    "AHEAD": [(0, 0), (10.6, 2.9), (30.4, 4.4), (52.8, 4.7), (68.2, 4.4)],
})
T11 = tab({
    "Retargeting VLA": [(0, 0), (0, 0), (0, 0), (0.6, 1.1), (1.4, 1.5)],
    "VLA + Fast Replan": [(0, 0), (0, 0), (0.4, 0.9), (1.8, 1.7), (4.2, 2.3)],
    "Realtime ACT": [(0, 0), (6.4, 2.5), (3.6, 2.9), (10.4, 3.3), (17.8, 3.8)],
    "Streaming DP": [(0, 0), (0, 0), (0.6, 1.1), (3.2, 2.0), (8.4, 2.7)],
    "DreamVLA": [(0, 0), (0, 0), (1.8, 1.7), (6.4, 2.5), (12.6, 3.2)],
    "AHEAD": [(0, 0), (9.8, 2.9), (22.6, 4.1), (38.4, 4.6), (48.6, 4.7)],
})
# 부록 E (4개 시나리오 열)
T12 = tab({
    "RAFT": [(97.3, 1.5), (91.7, 2.8), (95.0, 2.3), (90.0, 3.1)],
    "CoTracker3": [(89.3, 3.1), (87.7, 3.3), (80.7, 4.0), (78.3, 4.1)],
    "Implicit 3-frame": [(80.4, 3.9), (75.6, 2.2), (73.2, 4.4), (67.6, 3.6)],
    "Farneback": [(79.7, 4.0), (69.3, 4.6), (68.7, 4.7), (58.0, 4.9)],
    "Frame differencing": [(45.0, 5.0), (39.7, 4.9), (40.3, 2.1), (29.0, 4.5)],
    "No velocity": [(0, 0)] * 4,
})
T13 = tab({
    "Constant Vk=V0": [(50.6, 4.3), (87.2, 2.8), (58.4, 4.7), (90.6, 2.4)],
    "Exponential decay": [(42.8, 5.1), (78.4, 3.6), (50.2, 4.4), (89.8, 2.6)],
    "Kinematic only": [(78.2, 3.9), (86.8, 3.4), (74.6, 4.2), (90.4, 2.7)],
    "Accel token only": [(68.4, 4.5), (85.4, 3.5), (68.8, 4.6), (91.2, 2.5)],
    "Kinematic + accel (ours)": [(88.4, 3.1), (91.6, 2.6), (93.2, 2.3), (89.4, 2.9)],
    "Learned dynamics": [(76.8, 4.0), (89.4, 3.0), (91.8, 2.7), (56.8, 4.8)],
})
T14 = tab({
    "Language + motion (ours)": [(97.3, 1.5), (92.4, 2.6), (94.6, 2.4), (91.2, 2.8)],
    "Motion only": [(86.2, 3.4), (79.8, 3.9), (84.4, 3.5), (85.6, 3.4)],
    "Language only": [(87.4, 3.2), (68.6, 4.5), (79.2, 3.8), (76.4, 4.1)],
    "Uniform random 30%": [(49.4, 4.6), (41.6, 4.8), (51.8, 4.5), (42.8, 4.7)],
    "Full prediction": [(96.0, 2.2), (86.8, 3.3), (95.6, 2.4), (86.4, 3.4)],
})
T14_LAT = {"Language + motion (ours)": 158, "Motion only": 151, "Language only": 154,
           "Uniform random 30%": 160, "Full prediction": 250}
T15 = tab({
    "K=1": [(51.2, 4.4), (59.8, 4.5), (48.4, 4.7), (40.6, 4.9)],
    "K=2": [(58.6, 4.5), (65.4, 4.6), (60.2, 4.7), (56.8, 4.8)],
    "K=3": [(78.6, 2.9), (84.4, 3.6), (76.2, 4.1), (78.8, 3.9)],
    "K=5": [(96.8, 1.9), (92.4, 2.6), (94.6, 2.4), (91.2, 2.8)],
    "K=8": [(95.6, 2.2), (91.8, 2.8), (93.8, 2.6), (90.4, 3.0)],
    "K=12": [(88.6, 3.2), (91.4, 2.7), (87.4, 3.4), (77.8, 4.0)],
})
T15_LAT = {1: 120, 2: 128, 3: 138, 5: 158, 8: 185, 12: 218}
T16 = tab({
    "Adaptive": [(76.4, 4.1), (68.2, 4.4), (48.6, 4.7)],
    "Fixed K=5": [(63.8, 3.5), (54.6, 6.7), (31.4, 1.4)],
})
T18 = tab({
    "S=1": [(78.6, 3.8), (80.2, 3.7), (70.4, 4.4), (58.4, 4.6)],
    "S=3": [(86.4, 3.3), (88.2, 3.1), (87.6, 3.2), (78.4, 3.9)],
    "S=5 (ours)": [(97.3, 1.5), (92.4, 2.6), (94.6, 2.4), (91.2, 2.8)],
    "S=10": [(97.0, 1.8), (92.8, 2.5), (95.0, 2.3), (91.6, 2.7)],
    "Enc 2L": [(78.2, 3.9), (70.4, 4.3), (68.6, 4.5), (71.6, 4.2)],
    "Enc 4L (ours)": [(96.8, 1.9), (92.4, 2.6), (94.6, 2.4), (91.2, 2.8)],
    "Enc 6L": [(97.2, 1.7), (92.6, 2.5), (95.4, 2.2), (92.0, 2.6)],
})
T19 = tab({
    "default": [(97.3, 1.5), (87.4, 3.2)],
    "no alignment": [(88.4, 3.1), (61.1, 4.6)],
    "arm patches excluded": [(97.6, 1.4), (92.2, 2.7)],
    "best learned baseline (DreamVLA)": [(58.3, 4.7), (49.1, 9.1)],
})
# 표 5 (요약, 평균 ± ± )
T5 = {
    "motion": {"RAFT": (93.5, 1.8), "CoTracker3": (84.0, 2.7), "Implicit": (74.2, 3.1), "Farneback": (68.9, 4.5),
               "FrameDiff": (38.5, 3.4), "NoVel": (0.0, 0.0)},
    "velocity": {"Kin+accel": (90.7, 1.1), "Kin only": (82.5, 3.6), "Learned": (78.7, 7.9),
                 "Accel-token": (78.5, 5.8), "Const": (71.7, 10.4), "ExpDecay": (65.3, 11.0)},
    "masking": {"Lang+motion": (93.8, 1.2), "Full": (91.2, 2.7), "Motion": (84.0, 1.5), "Lang": (77.9, 3.9),
                "Uniform": (46.4, 2.5)},
    "horizon": {"K=5": (93.8, 1.2), "K=8": (92.9, 1.1), "K=12": (86.3, 3.0), "K=3": (79.5, 1.8),
                "K=2": (60.2, 1.9), "K=1": (50.0, 4.0)},
}
# 표 5 열을 표 12–15 행과 짝지음
T5_SRC = {
    ("motion", "RAFT"): T12["RAFT"], ("motion", "CoTracker3"): T12["CoTracker3"],
    ("motion", "Implicit"): T12["Implicit 3-frame"], ("motion", "Farneback"): T12["Farneback"],
    ("motion", "FrameDiff"): T12["Frame differencing"], ("motion", "NoVel"): T12["No velocity"],
    ("velocity", "Kin+accel"): T13["Kinematic + accel (ours)"], ("velocity", "Kin only"): T13["Kinematic only"],
    ("velocity", "Learned"): T13["Learned dynamics"], ("velocity", "Accel-token"): T13["Accel token only"],
    ("velocity", "Const"): T13["Constant Vk=V0"], ("velocity", "ExpDecay"): T13["Exponential decay"],
    ("masking", "Lang+motion"): T14["Language + motion (ours)"], ("masking", "Full"): T14["Full prediction"],
    ("masking", "Motion"): T14["Motion only"], ("masking", "Lang"): T14["Language only"],
    ("masking", "Uniform"): T14["Uniform random 30%"],
    ("horizon", "K=5"): T15["K=5"], ("horizon", "K=8"): T15["K=8"], ("horizon", "K=12"): T15["K=12"],
    ("horizon", "K=3"): T15["K=3"], ("horizon", "K=2"): T15["K=2"], ("horizon", "K=1"): T15["K=1"],
}

# 실로봇 표 4: (성공, 30) 과제 5개, 속도 스윕 (성공, 10) 6점
T4_TASKS = ["static_obj_moving_box", "static_box_moving_obj", "paddle_hit", "stop_ball", "catch_proj"]
T4 = {
    "Open-loop VLA": [0, 0, 0, 0, 0], "Retargeting VLA": [0, 0, 0, 0, 0], "VLA + Fast Replan": [0, 0, 0, 0, 0],
    "Realtime ACT": [9, 5, 9, 2, 0], "Streaming DP": [3, 3, 0, 0, 0], "DreamVLA": [12, 9, 1, 12, 0],
    "AHEAD": [30, 29, 23, 30, 19],
}
T4_SPEEDS = [0, 5, 10, 15, 20, 25]
T4_SWEEP = {
    "Open-loop VLA": [10, 0, 0, 0, 0, 0], "Retargeting VLA": [10, 4, 0, 0, 0, 0], "Realtime ACT": [10, 8, 0, 0, 0, 0],
    "Streaming DP": [10, 0, 3, 0, 0, 0], "DreamVLA": [10, 2, 0, 0, 0, 0], "AHEAD": [10, 10, 10, 7, 6, 2],
}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def feasible(v, n_total):
    """평균 v(%, 소수 첫째 자리 반올림)가 정수 성공 수 k / n_total 로 나올 수 있는가."""
    for k in range(n_total + 1):
        if abs(round(100.0 * k / n_total + 1e-9, 1) - v) < 1e-6:
            return True
    return False


# ---------------------------------------------------------------- 모든 시뮬레이션 셀
def all_cells():
    cells = []
    for tname, tbl, cols in [("T1", T1, T1_COLS), ("T2", T2, T2_COLS), ("T3", T3, T3_COLS),
                             ("T9", T9, range(5)), ("T10", T10, range(5)), ("T11", T11, range(5)),
                             ("T12", T12, range(4)), ("T13", T13, range(4)), ("T14", T14, range(4)),
                             ("T15", T15, range(4)), ("T16", T16, range(3)), ("T18", T18, range(4)),
                             ("T19", T19, range(2))]:
        for row, vals in tbl.items():
            for c, (m, s) in zip(cols, vals):
                cells.append(dict(table=tname, row=row, col=str(c), mean=m, pm=s))
    return cells


CELLS = all_cells()
R["n_cells_total"] = len(CELLS)

# ---------------------------------------------------------------- V1: ± 와 이항 잡음
ratios = []
rows_out = []
for c in CELLS:
    p = c["mean"] / 100
    if p <= 0 or p >= 1:
        continue
    sd1 = math.sqrt(p * (1 - p) / 100) * 100          # 시드 하나(100회)의 이항 SD (%p)
    c["binom_sd100"] = sd1
    c["ratio"] = c["pm"] / sd1
    ratios.append(c["ratio"])
ratios = np.array(ratios)
R["V1"] = dict(
    n=len(ratios), median=float(np.median(ratios)), mean=float(ratios.mean()), sd=float(ratios.std(ddof=1)),
    p10=float(np.percentile(ratios, 10)), p90=float(np.percentile(ratios, 90)),
    frac_within_10pct=float(np.mean(np.abs(ratios - 1) <= 0.10)),
    frac_within_20pct=float(np.mean(np.abs(ratios - 1) <= 0.20)),
)
# 가설 A: ± 가 5개 시드 평균의 SD(ddof=1)이고 시드 간 변동이 이항 잡음뿐이면 비율 ~ sqrt(chi2_4/4)
rng = np.random.default_rng(0)
sim = np.sqrt(rng.chisquare(4, 200000) / 4)
R["V1"]["hypA_SD_of_seed_means"] = dict(mean=float(sim.mean()), sd=float(sim.std()),
                                       p10=float(np.percentile(sim, 10)), p90=float(np.percentile(sim, 90)),
                                       frac_within_10pct=float(np.mean(np.abs(sim - 1) <= 0.10)),
                                       frac_within_20pct=float(np.mean(np.abs(sim - 1) <= 0.20)))
# 가설 B: ± 가 시드 평균의 표준오차(SD/sqrt5)이면 비율 ~ sqrt(chi2_4/4)/sqrt(5)
R["V1"]["hypB_SE_of_seed_means"] = dict(mean=float((sim / math.sqrt(5)).mean()),
                                       p10=float(np.percentile(sim / math.sqrt(5), 10)),
                                       p90=float(np.percentile(sim / math.sqrt(5), 90)))
out = sorted([c for c in CELLS if "ratio" in c and (c["ratio"] > 1.5 or c["ratio"] < 0.6)], key=lambda c: c["ratio"])
R["V1"]["outliers"] = [dict(table=c["table"], row=c["row"], col=c["col"], mean=c["mean"], pm=c["pm"],
                            expected=round(c["binom_sd100"], 2), ratio=round(c["ratio"], 2)) for c in out]
R["V1"]["n_outliers"] = len(out)

# ---------------------------------------------------------------- V2: 정수 격자
grid = {"n500": 500, "n300": 300, "n600": 600}
bad = {k: [] for k in grid}
for c in CELLS:
    for g, n in grid.items():
        if not feasible(c["mean"], n):
            bad[g].append((c["table"], c["row"], c["col"], c["mean"]))
R["V2"] = {g: dict(n_infeasible=len(v)) for g, v in bad.items()}
# 500 격자에 안 맞는 셀 중 300 격자나 600 격자로는 설명되는 것
off500 = bad["n500"]
R["V2"]["off500_total"] = len(off500)
R["V2"]["off500_by_table"] = {}
for t, r, col, m in off500:
    R["V2"]["off500_by_table"].setdefault(t, []).append(f"{r}/{col}={m}")
R["V2"]["off500_explained_by_300"] = sum(1 for (t, r, col, m) in off500 if feasible(m, 300))
R["V2"]["off500_explained_by_600"] = sum(1 for (t, r, col, m) in off500 if feasible(m, 600))
t1_rows = [c for c in CELLS if c["table"] == "T1" and c["row"] not in ("Track-and-Intercept VLA", "LaDi-WM")]
R["V2"]["T1_first7_rows_off500"] = sum(1 for c in t1_rows if not feasible(c["mean"], 500))
R["V2"]["T1_first7_rows_n"] = len(t1_rows)
R["V2"]["T1_first7_rows_off500_explained_by_300"] = sum(
    1 for c in t1_rows if not feasible(c["mean"], 500) and feasible(c["mean"], 300))
t1_new = [c for c in CELLS if c["table"] == "T1" and c["row"] in ("Track-and-Intercept VLA", "LaDi-WM")]
R["V2"]["T1_new_rows_off500"] = sum(1 for c in t1_new if not feasible(c["mean"], 500))
R["V2"]["T1_new_rows_n"] = len(t1_new)

# ---------------------------------------------------------------- V3: 표 5 = 표 12–15 의 4열 평균?
v3 = []
for (grp, name), vals in T5_SRC.items():
    m, s = T5[grp][name]
    means = np.array([v[0] for v in vals])
    pms = np.array([v[1] for v in vals])
    mean4 = means.mean()
    sd4 = means.std(ddof=1)
    v3.append(dict(group=grp, row=name, reported=m, recomputed=round(float(mean4), 2),
                   diff=round(m - float(mean4), 2),
                   pm_reported=s,
                   se_of_4_scenarios=round(float(sd4 / 2), 2),
                   rss_of_cell_pm=round(float(np.sqrt((pms ** 2).sum()) / 4), 2)))
R["V3"] = v3
R["V3_max_abs_diff"] = max(abs(x["diff"]) for x in v3)
R["V3_pm_vs_se_corr"] = float(np.corrcoef([x["pm_reported"] for x in v3],
                                        [x["se_of_4_scenarios"] for x in v3])[0, 1])

# ---------------------------------------------------------------- V4: 같은 구성(AHEAD 기본)의 값이 표마다 다른가
cv_cols = ["Conv", "Beam", "Pole", "Roll"]
same_cfg_cv = {
    "T1 (AHEAD)": [x[0] for x in T1["AHEAD"][:4]],
    "T12 (RAFT ours)": [x[0] for x in T12["RAFT"]],
    "T14 (Lang+motion ours)": [x[0] for x in T14["Language + motion (ours)"]],
    "T15 (K=5 fixed)": [x[0] for x in T15["K=5"]],
    "T18 (S=5 ours)": [x[0] for x in T18["S=5 (ours)"]],
    "T18 (Enc 4L ours)": [x[0] for x in T18["Enc 4L (ours)"]],
}
R["V4_const_vel"] = {k: dict(values=v, mean=round(float(np.mean(v)), 2)) for k, v in same_cfg_cv.items()}
same_cfg_ac = {
    "T1 (AHEAD accel)": [x[0] for x in T1["AHEAD"][4:]],
    "T13 (Kin+accel ours)": [x[0] for x in T13["Kinematic + accel (ours)"]],
}
R["V4_accel"] = {k: dict(values=v, mean=round(float(np.mean(v)), 2)) for k, v in same_cfg_ac.items()}
R["V4_accel_pm"] = {"T1": [x[1] for x in T1["AHEAD"][4:]], "T13": [x[1] for x in T13["Kinematic + accel (ours)"]]}
R["V4_BC_across_tables"] = {"T3 AHEAD BC": 87.4, "T19 default BC": 87.4, "T3 Air hockey": 94.6, "T14/T18 Pole": 94.6}

# ---------------------------------------------------------------- V5: 표 1 대 표 2 (conveyor + red cup)
# 가정: 표 1 conveyor 의 속도는 부록 C.1 대로 5–25 cm/s 균등, 성공률은 표 2 점 사이를 선형 보간
speeds = np.array([0, 10, 20, 30, 40.0])
xs = np.linspace(5, 25, 2001)
v5 = {}
t1_conv = {r: v[0][0] for r, v in T1.items()}
for name in ["Retargeting VLA", "Realtime ACT", "Streaming DP", "DreamVLA", "AHEAD"]:
    y = np.array([x[0] for x in T2[name]])
    pred = float(np.interp(xs, speeds, y).mean())
    v5[name] = dict(table1_const_vel_conveyor=t1_conv[name], table2_interp_avg_5_25=round(pred, 1),
                    gap=round(t1_conv[name] - pred, 1))
R["V5"] = v5

# ---------------------------------------------------------------- V6: 초록과 서론의 범위
learned = ["Open-loop VLA", "Retargeting VLA", "VLA + Fast Replan", "Realtime ACT", "Streaming DP", "DreamVLA"]
best_per_scn = []
scn = []
for i, c in enumerate(T1_COLS):
    scn.append(("T1", c, max(T1[r][i][0] for r in learned), T1["AHEAD"][i][0],
                max(T1[r][i][0] for r in learned + ["Track-and-Intercept VLA", "LaDi-WM"])))
for i, c in enumerate(T3_COLS):
    scn.append(("T3", c, max(T3[r][i][0] for r in learned), T3["AHEAD"][i][0],
                max(T3[r][i][0] for r in learned + ["Track-and-Intercept VLA", "LaDi-WM"])))
R["V6"] = dict(
    n_scenarios_main=len(scn),
    ahead_min=min(s[3] for s in scn), ahead_max=max(s[3] for s in scn),
    best_learned_min=min(s[2] for s in scn), best_learned_max=max(s[2] for s in scn),
    best_any_min=min(s[4] for s in scn), best_any_max=max(s[4] for s in scn),
    dreamvla_table1_min=min(x[0] for x in T1["DreamVLA"]), dreamvla_table1_max=max(x[0] for x in T1["DreamVLA"]),
    per_scenario=[dict(table=s[0], col=s[1], best_learned=s[2], ahead=s[3], best_incl_motion_informed=s[4]) for s in scn],
    n_scenarios_incl_stress=len(scn) + 3,
    table3_best_learned_max=max(max(T3[r][i][0] for r in learned) for i in range(8)),
    table3_best_any_max=max(max(T3[r][i][0] for r in learned + ["Track-and-Intercept VLA", "LaDi-WM"]) for i in range(8)),
    occ_nonzero_baselines=[r for r in T3 if r != "AHEAD" and T3[r][5][0] > 0],
    ahead_wins_all_table1=all(T1["AHEAD"][i][0] > max(T1[r][i][0] for r in T1 if r != "AHEAD") for i in range(8)),
    ahead_wins_all_table3=all(T3["AHEAD"][i][0] > max(T3[r][i][0] for r in T3 if r != "AHEAD") for i in range(8)),
)

# ---------------------------------------------------------------- V7: 본문에 인용된 수치
def avg(xs_):
    return round(float(np.mean(xs_)), 2)


R["V7"] = dict(
    TI_cv_avg=avg([x[0] for x in T1["Track-and-Intercept VLA"][:4]]),
    AHEAD_cv_avg=avg([x[0] for x in T1["AHEAD"][:4]]),
    LaDi_cv_avg=avg([x[0] for x in T1["LaDi-WM"][:4]]),
    DreamVLA_accel_range=(min(x[0] for x in T1["DreamVLA"][4:]), max(x[0] for x in T1["DreamVLA"][4:])),
    AHEAD_accel_min=min(x[0] for x in T1["AHEAD"][4:]),
    AHEAD_regime_gap_max=round(max(T1["AHEAD"][i][0] - T1["AHEAD"][i + 4][0] for i in range(4)), 2),
    residual_40cm=round(97.6 - 95.4, 2),
    stress_ratio_plinko=round(48.6 / 17.8, 2), stress_ratio_multi=round(76.4 / 35.6, 2),
    stress_ratio_occdef=round(68.2 / 27.4, 2),
    ACT_plinko_series=[x[0] for x in T11["Realtime ACT"]],
)

# ---------------------------------------------------------------- V8: 쌍별 검정 (이항 근사, 각 500회 가정)
def two_prop_p(p1, p2, n=500):
    k1, k2 = round(p1 * n / 100), round(p2 * n / 100)
    pp = (k1 + k2) / (2 * n)
    if pp in (0, 1):
        return 1.0
    z = (k1 / n - k2 / n) / math.sqrt(pp * (1 - pp) * 2 / n)
    return float(2 * stats.norm.sf(abs(z)))


v8 = []
for i, c in enumerate(T1_COLS):
    a = T1["AHEAD"][i][0]
    t = T1["Track-and-Intercept VLA"][i][0]
    v8.append(dict(cell=f"T1 {c}", ahead=a, track=t, gap=round(a - t, 1), p=two_prop_p(a, t)))
R["V8_T1_vs_track"] = v8
R["V8_T1_min_gap"] = min(v8, key=lambda x: x["gap"])
mask = []
for j, c in enumerate(cv_cols):
    a = T14["Language + motion (ours)"][j][0]
    f = T14["Full prediction"][j][0]
    pa, pf = T14["Language + motion (ours)"][j][1], T14["Full prediction"][j][1]
    z_pm = (a - f) / math.sqrt(pa ** 2 + pf ** 2)          # 표의 ± 를 표준오차로 읽은 경우
    mask.append(dict(cell=c, masked=a, full=f, diff=round(a - f, 1), p=two_prop_p(a, f),
                     z_if_pm_is_se=round(z_pm, 2), p_if_pm_is_se=float(2 * stats.norm.sf(abs(z_pm)))))
R["V8_mask_vs_full"] = mask
ad = []
for j, c in enumerate(["multi_deflection", "occlusion_deflection", "plinko"]):
    a, f = T16["Adaptive"][j][0], T16["Fixed K=5"][j][0]
    ad.append(dict(cell=c, adaptive=a, fixed5=f, diff=round(a - f, 1), p=two_prop_p(a, f)))
R["V8_adaptive_vs_fixed"] = ad
ali = []
for j, c in enumerate(["conveyor", "ballistic"]):
    a, f = T19["default"][j][0], T19["no alignment"][j][0]
    ali.append(dict(cell=c, default=a, no_align=f, diff=round(a - f, 1), p=two_prop_p(a, f)))
R["V8_alignment"] = ali
R["V8_arm_excluded_BC"] = dict(default=87.4, excluded=92.2, p=two_prop_p(87.4, 92.2))
# 가속 vs 상수 속도 (표 13, 롤링 중력)
R["V8_roll_gravity_const_vs_ours"] = dict(const=90.6, ours=89.4, p=two_prop_p(90.6, 89.4))

# ---------------------------------------------------------------- V9: 실로봇
v9 = {"tasks": [], "sweep": []}
for j, t in enumerate(T4_TASKS):
    k = T4["AHEAD"][j]
    best_name = max((r for r in T4 if r != "AHEAD"), key=lambda r: T4[r][j])
    kb = T4[best_name][j]
    lo, hi = wilson(k, 30)
    _, p = stats.fisher_exact([[k, 30 - k], [kb, 30 - kb]])
    v9["tasks"].append(dict(task=t, ahead=f"{k}/30", ahead_rate=round(100 * k / 30, 1),
                            wilson=[round(100 * lo, 1), round(100 * hi, 1)], best_baseline=best_name,
                            best=f"{kb}/30", fisher_p=float(p)))
for j, s in enumerate(T4_SPEEDS):
    k = T4_SWEEP["AHEAD"][j]
    best_name = max((r for r in T4_SWEEP if r != "AHEAD"), key=lambda r: T4_SWEEP[r][j])
    kb = T4_SWEEP[best_name][j]
    lo, hi = wilson(k, 10)
    _, p = stats.fisher_exact([[k, 10 - k], [kb, 10 - kb]])
    v9["sweep"].append(dict(speed=s, ahead=f"{k}/10", wilson=[round(100 * lo, 1), round(100 * hi, 1)],
                            best_baseline=best_name, best=f"{kb}/10", fisher_p=float(p)))
sweep_pool = sum(T4_SWEEP["AHEAD"][1:])      # 5–25 cm/s 5점
v9["ahead_sweep_5_25"] = f"{sweep_pool}/50"
v9["ahead_sweep_5_25_rate"] = sweep_pool / 50
v9["p_30of30_if_rate_0.7"] = 0.7 ** 30
v9["p_at_least_59_of_60_if_rate_0.7"] = float(stats.binom.sf(58, 60, 0.7))
# 합산 정수 검사
v9["failures_projectile"] = 30 - 19
v9["sim_vs_real"] = {
    "conveyor": dict(sim=97.3, real=100.0), "paddle_vs_air_hockey": dict(sim=94.6, real=round(100 * 23 / 30, 1)),
    "projectile_vs_ballistic": dict(sim=87.4, real=round(100 * 19 / 30, 1)),
    "speed_20cm_sim_table2": 97.0, "speed_20cm_real": "6/10",
    "speed_25cm_real": "2/10",
}
R["V9"] = v9

# ---------------------------------------------------------------- V10: 산술 (파라미터, 시간, 계산량, 지연)
lat = [(k, v) for k, v in T15_LAT.items()]
ks = np.array([k for k, _ in lat], float)
vs = np.array([v for _, v in lat], float)
slope, icpt = np.polyfit(ks, vs, 1)
R["V10"] = dict(
    param_sum_M=round(2.6 + 1.1 + 0.8 + 0.3 + 0.1, 2), param_fraction_of_7B_pct=round(4.9e6 / 7e9 * 100, 3),
    latency_sum_ms=20 + 2 + 40 + 6 + 70 + 20,
    wrapper_over_reactive_ms=158 - 90,
    gpu_hours_total=96 + 4 + 2,
    pretrain_gpu_hours_check=dict(gpus=8, wall_hours=96 / 8, steps=200000, steps_per_s=round(200000 / (96 / 8 * 3600), 2)),
    corpus_sum_K=50 + 220 + 76 + 60 + 100 + 150, corpus_with_finetune_K=50 + 220 + 76 + 60 + 100 + 150 + 0.2,
    samples_seen_M=round(200000 * 256 / 1e6, 1), epochs_over_corpus=round(200000 * 256 / 656e3, 1),
    latency_fit_slope_ms_per_step=round(float(slope), 2), latency_fit_intercept_ms=round(float(icpt), 1),
    latency_fit_residual_max=round(float(np.abs(vs - (slope * ks + icpt)).max()), 2),
    est_latency_K9_1=round(float(slope * 9.1 + icpt), 0), est_latency_K9_6=round(float(slope * 9.6 + icpt), 0),
    est_latency_K4_2=round(float(slope * 4.2 + icpt), 0), est_latency_K1_8=round(float(slope * 1.8 + icpt), 0),
    lookahead_ms_at_30Hz={k: round(k * 1000 / 30, 0) for k in (1, 2, 3, 5, 8, 12)},
    realized_K_table17={"const": 4.2, "accel": 3.8, "ballistic": 9.1, "occlusion": 9.6, "plinko": 1.8},
    wrapper_latency_real_ms=171,
    # 정렬 층 0.1M 이 두 층 MLP(Linear d->h, GELU, Linear h->d)라면 은닉 폭 h (d=2176 가정: OpenVLA SigLIP+DINOv2 연결)
    align_hidden_if_d2176=round((0.1e6 - 2176) / (2 * 2176 + 1), 1),
    align_params_if_full_dxd_M=round((2 * 2176 * 2176 + 2 * 2176) / 1e6, 2),
    # 인코더: 4층, 폭 256 (FFN 배율 r), 입력 사영 (d+4)->256
    encoder_params_M={f"ffn_x{r}": round((4 * (4 * 256 * 256 + 2 * r * 256 * 256) + (2176 + 4) * 256) / 1e6, 2) for r in (1, 2, 4)},
    tokens_N=196, grid_196=14 * 14, grid_256=16 * 16,
    # 1단계(부록 B.8)에서 모든 프레임의 패치 특징을 저장한다고 읽었을 때 저장량 (가정: 16프레임/시퀀스, d=2176, bf16)
    feature_storage_TB_N196=round(656e3 * 16 * 196 * 2176 * 2 / 1e12, 1),
    feature_storage_TB_N256=round(656e3 * 16 * 256 * 2176 * 2 / 1e12, 1),
    openvla_7p5B_bf16_GB=round(7.5e9 * 2 / 1e9, 1),
)

# ---------------------------------------------------------------- V11: 시간창 단조성(표 9–11)
nonmono = []
for tn, tbl in [("T9", T9), ("T10", T10), ("T11", T11)]:
    for r, v in tbl.items():
        ms = [x[0] for x in v]
        if any(ms[i + 1] < ms[i] for i in range(4)):
            nonmono.append(f"{tn} {r}: {ms}")
R["V11_nonmonotone_time_windows"] = nonmono
R["V11_identical_cells"] = [
    "T10 ACT 200ms = T11 ACT 200ms = 6.4 ± 2.5",
]

# ---------------------------------------------------------------- V12: 표 2 단조성
r12 = {}
for r, v in T2.items():
    ms = [x[0] for x in v]
    r12[r] = ms
R["V12_table2"] = r12

OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False, default=lambda o: o if not isinstance(o, tuple) else list(o)), encoding="utf-8")
print("V1", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in R["V1"].items() if k in ("n", "median", "mean", "sd", "frac_within_10pct", "frac_within_20pct", "n_outliers")})
print("V1 hypA", R["V1"]["hypA_SD_of_seed_means"])
print("V1 outliers", R["V1"]["outliers"])
print("V2", R["V2"]["n500"], R["V2"]["n300"], R["V2"]["off500_total"], R["V2"]["off500_explained_by_300"],
      R["V2"]["T1_first7_rows_off500"], "/", R["V2"]["T1_first7_rows_n"], R["V2"]["T1_new_rows_off500"], "/", R["V2"]["T1_new_rows_n"])
print("V3 max diff", R["V3_max_abs_diff"], "corr", round(R["V3_pm_vs_se_corr"], 2))
print("V5", R["V5"])
print("V6", {k: v for k, v in R["V6"].items() if k != "per_scenario"})
print("V9", R["V9"]["ahead_sweep_5_25"], R["V9"]["p_30of30_if_rate_0.7"])
print("V10", R["V10"])
