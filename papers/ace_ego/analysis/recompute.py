"""ACE-Ego-0 (arXiv:2606.17200 v1) 표 재계산과 일관성 검사 (V1-V10).

입력은 논문 표 값을 옮긴 상수(아래 T1, T9, T10, FIG5A)이다. 그림 5(a)의 막대 값은 그림에 적힌 숫자를 읽은 것이다(그림 판독 값).
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초) -> analysis/recompute_results.json
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
R = {}

# ---------------------------------------------------------------- 논문 표 1: 사전학습 데이터 풀
# (이름, 에피소드, 프레임, 시간)
T1_HUMAN = [
    ("Ego4D", 948683, 23396157, 216.6),
    ("EgoExo4D", 41414, 1110275, 10.3),
    ("EPIC-KITCHENS-100", 74788, 3486432, 32.3),
    ("HOI4D", 2966, 774275, 7.2),
    ("EgoDex", 327317, 83894075, 776.8),
    ("Xperience-10M", 99027, 31370900, 435.7),
]
T1_ROBOT = [
    ("AgiBot Alpha/Beta", 116013, 209284239, 1937.8, "real"),
    ("Galaxea R1Lite", 20662, 26358560, 488.1, "real"),
    ("AgiBot DigitalWorld", 47910, 24333788, 225.3, "sim"),
    ("RoboCasa Tabletop", 24000, 6020058, 83.6, "sim"),
    ("Galbot self-collected", 60000, 194_000_000, 1800.0, "real"),  # 표의 ~60,000, ~194M, 1,800+ (근삿값)
]


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


# V1: 합계, 시간 = 프레임/(fps*3600) 에서 역산한 fps, 에피소드 평균 길이
h_hours = sum(x[3] for x in T1_HUMAN)
h_frames = sum(x[2] for x in T1_HUMAN)
h_eps = sum(x[1] for x in T1_HUMAN)
r_hours = sum(x[3] for x in T1_ROBOT)
r_frames = sum(x[2] for x in T1_ROBOT)
r_eps = sum(x[1] for x in T1_ROBOT)
rows = []
for name, eps, fr, hr, *kind in T1_HUMAN + T1_ROBOT:
    fps = fr / (hr * 3600)
    mean_s = hr * 3600 / eps
    row = {"name": name, "episodes": eps, "frames": fr, "hours": hr, "implied_fps": round(fps, 2),
           "mean_episode_seconds": round(mean_s, 2), "mean_episode_frames": round(fr / eps, 1)}
    rows.append(row)
# Stage 1: 4 s 이상 30 s 이하 클립만 남긴다. 시간이 주어졌을 때 가능한 최대 에피소드 수 = 시간*3600/4
for row, src in zip(rows[:6], T1_HUMAN):
    row["max_episodes_if_min_4s"] = int(src[3] * 3600 / 4)
    row["listed_over_max"] = round(src[1] / row["max_episodes_if_min_4s"], 2)
R["V1_table1"] = {
    "rows": rows,
    "human_hours_sum": round(h_hours, 1), "reported_human_subtotal": 1478.9,
    "human_frames_sum": h_frames, "reported_human_frames": 144032114,
    "human_episodes_sum": h_eps, "reported_human_episodes": 1494195,
    "robot_hours_sum": round(r_hours, 1), "reported_robot_hours": 4534.8,
    "robot_frames_sum_approx": r_frames, "robot_episodes_sum": r_eps, "reported_robot_episodes": 268585,
    "total_hours": round(h_hours + r_hours, 1), "reported_total_hours": 6013.7,
    "total_frames_approx": h_frames + r_frames,
    "share_human_hours_pct": round(100 * h_hours / (h_hours + r_hours), 2),
    "share_sim_hours_pct": round(100 * (225.3 + 83.6) / (h_hours + r_hours), 2),
    "share_real_robot_hours_pct": round(100 * (1937.8 + 488.1 + 1800.0) / (h_hours + r_hours), 2),
    "fig1_pie_pct": {"human": 24.9, "sim": 3.4, "robot": 71.6},
    "fig3_raw_hours": 5929, "fig3_kept_hours": 1478,
    "fig3_retention_pct": round(100 * 1478 / 5929, 2),
}

# 사전학습이 본 샘플 수 (표 8: 200K 스텝, 장치당 배치 8, 128 GPU; 샘플 = 프레임 하나로 가정)
samples = 200_000 * 8 * 128
R["V1b_pretrain_passes"] = {
    "samples_seen_assumed": samples,
    "total_frames_approx": h_frames + r_frames,
    "passes_over_frames": round(samples / (h_frames + r_frames), 3),
    "note": "샘플 = 프레임 하나, 데이터 그룹 가중치 무시, 반복 확산(4회)은 같은 샘플의 재사용이라 제외",
}

# ---------------------------------------------------------------- 표 9: RoboCasa GR1 24과제
# 열: GR00T-N1.6, Qwen3PI, FLARE, ABot-M0, JoyAI-RA, ACE-Ego-0
T9 = {
    "CupToDrawerClose": [8.5, 42.0, 46.0, 48.0, 48.0, 36.0],
    "PotatoToMicrowaveClose": [41.5, 42.0, 30.0, 50.0, 70.0, 58.0],
    "MilkToMicrowaveClose": [14.0, 50.0, 58.0, 46.0, 84.0, 56.0],
    "BottleToCabinetClose": [51.5, 26.0, 66.0, 86.0, 84.0, 78.0],
    "WineToCabinetClose": [16.5, 32.0, 38.0, 66.0, 54.0, 56.0],
    "CanToDrawerClose": [13.0, 62.0, 64.0, 74.0, 90.0, 70.0],
    "CuttingboardToBasket": [58.0, 40.0, 44.0, 70.0, 88.0, 84.0],
    "CuttingboardToCardboardbox": [46.5, 46.0, 54.0, 58.0, 46.0, 84.0],
    "CuttingboardToPan": [68.5, 60.0, 80.0, 76.0, 92.0, 92.0],
    "CuttingboardToPot": [65.0, 40.0, 64.0, 66.0, 80.0, 84.0],
    "CuttingboardToTieredbasket": [46.5, 44.0, 46.0, 38.0, 36.0, 72.0],
    "PlacematToBasket": [58.5, 44.0, 48.0, 52.0, 76.0, 86.0],
    "PlacematToBowl": [57.5, 52.0, 58.0, 66.0, 52.0, 72.0],
    "PlacematToPlate": [63.0, 50.0, 74.0, 60.0, 38.0, 80.0],
    "PlacematToTieredshelf": [28.5, 28.0, 26.0, 26.0, 14.0, 44.0],
    "PlateToBowl": [57.0, 52.0, 50.0, 54.0, 48.0, 68.0],
    "PlateToCardboardbox": [43.5, 40.0, 56.0, 48.0, 38.0, 70.0],
    "PlateToPan": [51.0, 36.0, 70.0, 66.0, 46.0, 70.0],
    "PlateToPlate": [78.7, 48.0, 76.0, 64.0, 88.0, 98.0],
    "TrayToCardboardbox": [51.5, 34.0, 52.0, 54.0, 82.0, 78.0],
    "TrayToPlate": [71.0, 64.0, 64.0, 68.0, 88.0, 90.0],
    "TrayToPot": [64.5, 44.0, 70.0, 64.0, 88.0, 98.0],
    "TrayToTieredbasket": [57.0, 50.0, 60.0, 60.0, 62.0, 74.0],
    "TrayToTieredshelf": [31.5, 28.0, 28.0, 38.0, 24.0, 50.0],
}
M9 = ["GR00T-N1.6", "Qwen3PI", "FLARE", "ABot-M0", "JoyAI-RA", "ACE-Ego-0"]
REP9 = [47.6, 43.9, 55.0, 58.3, 63.2, 72.8]
TAB9 = np.array(list(T9.values()))
res9 = {}
for j, m in enumerate(M9):
    col = TAB9[:, j]
    k50 = col / 2.0  # 과제당 50회 -> 성공 수 = % / 2
    res9[m] = {
        "mean_recomputed": round(float(col.mean()), 3), "mean_reported": REP9[j],
        "cells_not_integer_of_50": int(np.sum(np.abs(k50 - np.round(k50)) > 1e-9)),
        "cells_total": len(col),
    }
R["V2_table9_means"] = res9

# 표 3(선택 과제) 대 표 9 행 일치
T3 = {
    "CuttingboardToCardboardbox": [46.5, 46.0, 54.0, 58.0, 46.0, 84.0],
    "PlacematToTieredshelf": [28.5, 28.0, 26.0, 26.0, 14.0, 44.0],
    "PlateToPlate": [78.7, 48.0, 76.0, 64.0, 88.0, 98.0],
    "PlateToBowl": [57.0, 52.0, 50.0, 54.0, 48.0, 68.0],
    "TrayToPlate": [71.0, 64.0, 64.0, 68.0, 88.0, 90.0],
}
R["V2b_table3_vs_table9_rows_match"] = all(T3[k] == T9[k] for k in T3)

# ACE 대 각 기준선: 과제 쌍체 (24과제)
ace = TAB9[:, 5]
paired9 = {}
rng = np.random.default_rng(0)
for j, m in enumerate(M9[:-1]):
    d = ace - TAB9[:, j]
    nz = d[d != 0]
    wins, losses, ties = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    sign_p = stats.binomtest(wins, wins + losses, 0.5).pvalue
    try:
        w_p = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        w_p = None
    boots = [rng.choice(d, len(d)).mean() for _ in range(10000)]
    paired9[m] = {
        "mean_diff_pp": round(float(d.mean()), 2), "wins": wins, "losses": losses, "ties": ties,
        "sign_test_p": round(float(sign_p), 5), "wilcoxon_p": None if w_p is None else round(w_p, 5),
        "bootstrap95_task_mean_diff": [round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)],
    }
R["V3_robocasa_paired_tasks"] = paired9

# ACE 24과제 중 JoyAI-RA에 지는 과제
lose = [(k, T9[k][5], T9[k][4]) for k in T9 if T9[k][5] < T9[k][4]]
R["V3b_ace_below_joyai"] = lose

# 풀링 비율 검정: 과제당 50회, 24과제 = 1,200 에피소드, 독립 가정 (단일 시드의 이항 잡음만 반영)
def two_prop(p1, p2, n):
    k1, k2 = round(p1 * n / 100), round(p2 * n / 100)
    tab = [[k1, n - k1], [k2, n - k2]]
    _, pf = stats.fisher_exact(tab)
    se = math.sqrt(p1 / 100 * (1 - p1 / 100) / n + p2 / 100 * (1 - p2 / 100) / n)
    z = (p1 - p2) / 100 / se
    return {"p1": p1, "p2": p2, "n_each_assumed": n, "diff_pp": round(p1 - p2, 2), "se_pp": round(100 * se, 2),
            "z": round(z, 2), "fisher_p": round(float(pf), 4),
            "ci95_diff_pp": [round(p1 - p2 - 196 * se, 2), round(p1 - p2 + 196 * se, 2)]}


R["V4_robocasa_pooled"] = {
    "vs_DIAL_70.2": two_prop(72.8, 70.2, 1200),
    "vs_JoyAI_63.2": two_prop(72.8, 63.2, 1200),
    "note": "DIAL은 평균만 있고 과제별 값, 시행 수가 표에 없다. n = 24 x 50 = 1,200 가정",
}

# 절제 (표 5, 그림 5b): 모두 RoboCasa 평균, n = 1,200 가정
abl = {
    "full_vs_robot_only(72.8-68.3)": two_prop(72.8, 68.3, 1200),
    "robot_only_vs_from_qwen(68.3-65.4)": two_prop(68.3, 65.4, 1200),
    "full_vs_from_qwen(72.8-65.4)": two_prop(72.8, 65.4, 1200),
    "full_vs_wo_chunking(72.8-71.7)": two_prop(72.8, 71.7, 1200),
    "full_vs_wo_urdf(72.8-70.9)": two_prop(72.8, 70.9, 1200),
    "full_vs_wo_human_loss(72.8-69.2)": two_prop(72.8, 69.2, 1200),
}
R["V5_ablation_pooled"] = abl
R["V5b_headline_decomposition_vs_joyai"] = {
    "ace_minus_joyai": round(72.8 - 63.2, 2),
    "from_qwen_minus_joyai": round(65.4 - 63.2, 2),
    "robot_pretrain_gain": round(68.3 - 65.4, 2),
    "human_gain": round(72.8 - 68.3, 2),
    "from_qwen_beats": [m for m, v in zip(M9, REP9) if 65.4 > v],
}
# 절제 수치의 정수 가능성 (n = 1,200): 성공 수 = % * 12
R["V5c_integer_feasibility_n1200"] = {str(p): abs(p * 12 - round(p * 12)) < 1e-6 for p in (72.8, 68.3, 65.4, 71.7, 70.9, 69.2)}

# ACE 열의 과제별 이항 분산으로 본 ACE 평균의 표준오차 (과제 층화, 독립 가정)
p = ace / 100
se_strat = math.sqrt(np.sum(p * (1 - p) / 50)) / len(p)
R["V5d_ace_mean_se_stratified_pp"] = round(100 * se_strat, 3)

# ---------------------------------------------------------------- 표 10: RoboTwin 2.0 50과제
M10 = ["pi0", "pi0.5", "Motus", "LingBot", "ABot-M0", "JoyAI-RA", "ACE-Ego-0"]
REP10 = {  # 평균 (Easy, Hard)
    "pi0": (65.92, 58.40), "pi0.5": (82.74, 76.76), "Motus": (88.66, 87.02), "LingBot": (88.56, 86.68),
    "ABot-M0": (86.06, 85.08), "JoyAI-RA": (90.48, 89.28), "ACE-Ego-0": (91.12, 90.62),
}
T10 = {
    "Adjust Bottle": [99, 95, 100, 99, 89, 93, 100, 100, None, None, 100, 100, 100, 100],
    "Beat Block Hammer": [79, 84, 96, 93, 95, 88, 92, 89, None, None, 95, 91, 98, 92],
    "Blocks Ranking RGB": [80, 63, 92, 85, 99, 97, 92, 91, 90, 79, 94, 93, 98, 97],
    "Blocks Ranking Size": [14, 5, 49, 26, 75, 63, 76, 70, None, None, 81, 75, 89, 91],
    "Click Alarmclock": [77, 68, 98, 89, 100, 100, 97, 43, None, None, 64, 56, 52, 38],
    "Click Bell": [71, 48, 99, 66, 100, 100, 43, 36, None, None, 81, 70, 66, 71],
    "Dump Bin Bigbin": [88, 83, 92, 97, 95, 91, 97, 97, None, None, 97, 99, 100, 97],
    "Grab Roller": [98, 94, 100, 100, 100, 100, 100, 100, None, None, 100, 100, 100, 100],
    "Handover Block": [47, 31, 66, 57, 86, 73, 83, 95, 72, 69, 99, 93, 96, 85],
    "Handover Mic": [97, 97, 98, 97, 78, 63, 94, 99, None, None, 100, 99, 91, 94],
    "Hanging Mug": [14, 11, 18, 17, 38, 38, 34, 53, None, None, 31, 28, 29, 31],
    "Lift Pot": [80, 72, 96, 85, 96, 99, 100, 100, None, None, 100, 99, 100, 100],
    "Move Can Pot": [68, 48, 51, 55, 34, 74, 89, 87, None, None, 97, 87, 100, 98],
    "Move Pillbottle Pad": [67, 46, 84, 61, 93, 96, 92, 90, 94, 86, 98, 99, 100, 100],
    "Move Playingcard Away": [74, 65, 96, 84, 100, 96, 98, 100, None, None, 99, 95, 100, 98],
    "Move Stapler Pad": [41, 24, 56, 42, 83, 85, 74, 48, 57, 61, 93, 96, 90, 89],
    "Open Laptop": [71, 81, 90, 96, 95, 91, 98, 96, None, None, 96, 100, 100, 98],
    "Open Microwave": [4, 32, 34, 77, 95, 91, 91, 92, 88, 84, 97, 99, 91, 85],
    "Pick Diverse Bottles": [69, 31, 81, 71, 90, 91, 88, 85, 71, 65, 85, 90, 84, 86],
    "Pick Dual Bottles": [59, 37, 93, 63, 96, 90, 99, 90, 70, 61, 95, 93, 89, 88],
    "Place A2B Left": [43, 47, 87, 82, 88, 79, 89, 85, None, None, 99, 96, 95, 96],
    "Place A2B Right": [39, 34, 87, 84, 91, 87, 80, 80, None, None, 97, 92, 90, 94],
    "Place Bread Basket": [62, 46, 77, 64, 91, 94, 95, 93, 89, 86, 88, 91, 92, 93],
    "Place Bread Skillet": [66, 49, 85, 66, 86, 83, 90, 92, None, None, 92, 89, 94, 89],
    "Place Burger Fries": [81, 76, 94, 87, 98, 98, 98, 94, None, None, 99, 93, 98, 100],
    "Place Can Basket": [55, 46, 62, 62, 81, 76, 75, 72, 72, 63, 71, 73, 78, 82],
    "Place Cans Plasticbox": [63, 45, 94, 84, 98, 94, 100, 98, None, None, 100, 98, 100, 98],
    "Place Container Plate": [97, 92, 99, 95, 98, 99, 99, 100, None, None, 96, 99, 98, 100],
    "Place Dual Shoes": [59, 51, 75, 75, 93, 87, 87, 86, 80, 80, 90, 97, 95, 96],
    "Place Empty Cup": [91, 85, 100, 99, 99, 98, 100, 100, None, None, 100, 100, 100, 100],
    "Place Fan": [66, 71, 87, 85, 91, 87, 92, 87, 97, 95, 91, 92, 94, 93],
    "Place Mouse Pad": [20, 20, 60, 39, 66, 68, 86, 79, None, None, 89, 82, 96, 95],
    "Place Object Basket": [67, 70, 80, 76, 81, 87, 90, 88, 91, 88, 90, 88, 93, 89],
    "Place Object Scale": [57, 52, 86, 80, 88, 85, 90, 88, None, None, 90, 87, 95, 92],
    "Place Object Stand": [82, 68, 91, 85, 98, 97, 93, 88, 90, 91, 95, 93, 95, 94],
    "Place Phone Stand": [49, 53, 81, 81, 87, 86, 90, 87, None, None, 95, 95, 91, 98],
    "Place Shoe": [76, 76, 92, 93, 99, 97, 99, 99, None, None, 99, 100, 100, 100],
    "Press Stapler": [44, 37, 87, 83, 93, 98, 86, 93, None, None, 87, 81, 98, 98],
    "Put Bottles Dustbin": [65, 56, 84, 79, 81, 79, 92, 93, 80, 89, 95, 97, 94, 93],
    "Put Object Cabinet": [73, 60, 80, 79, 88, 71, 85, 88, None, None, 87, 86, 82, 79],
    "Rotate QRcode": [74, 70, 89, 87, 89, 73, 86, 82, None, None, 83, 82, 94, 95],
    "Scan Object": [55, 42, 72, 65, 67, 66, 92, 96, 85, 86, 98, 96, 95, 97],
    "Shake Bottle Horizontally": [98, 92, 99, 99, 100, 98, 99, 98, None, None, 100, 100, 100, 100],
    "Shake Bottle": [94, 91, 99, 97, 100, 97, 100, 99, None, None, 100, 100, 100, 100],
    "Stack Blocks Three": [72, 52, 91, 76, 91, 95, 96, 95, 84, 77, 60, 62, 87, 82],
    "Stack Blocks Two": [93, 79, 97, 100, 100, 98, 100, 99, 96, 98, 95, 93, 100, 100],
    "Stack Bowls Three": [77, 75, 77, 71, 79, 87, 71, 77, 80, 86, 80, 81, 80, 85],
    "Stack Bowls Two": [94, 95, 95, 96, 98, 98, 90, 97, None, None, 95, 93, 96, 98],
    "Stamp Seal": [46, 33, 79, 55, 93, 92, 74, 77, 72, 75, 90, 90, 94, 100],
    "Turn Switch": [41, 42, 62, 54, 84, 78, 67, 63, 55, 66, 71, 76, 59, 57],
}


def col10(mi, setting):  # setting 0 = Easy, 1 = Hard
    return [v[2 * mi + setting] for v in T10.values()]


res10 = {}
for mi, m in enumerate(M10):
    for si, s in enumerate(["Easy", "Hard"]):
        c = [x for x in col10(mi, si) if x is not None]
        res10[f"{m}/{s}"] = {
            "n_tasks_listed": len(c), "mean_over_listed": round(float(np.mean(c)), 3),
            "mean_reported": REP10[m][si],
            "match_to_0.01": abs(float(np.mean(c)) - REP10[m][si]) < 0.011,
        }
R["V6_robotwin_means"] = res10

# 평균이 50개 정수의 평균이면 0.02의 배수여야 함
R["V6b_reported_avg_is_multiple_of_0.02"] = {
    f"{m}/{s}": abs(REP10[m][si] / 0.02 - round(REP10[m][si] / 0.02)) < 1e-6
    for m in M10 for si, s in enumerate(["Easy", "Hard"])
}

# ACE 대 JoyAI-RA, Motus, LingBot, pi0.5: 50과제 쌍체
paired10 = {}
for opp in ["JoyAI-RA", "Motus", "LingBot", "pi0.5"]:
    for si, s in enumerate(["Easy", "Hard"]):
        a = np.array(col10(6, si), dtype=float)
        b = np.array(col10(M10.index(opp), si), dtype=float)
        d = a - b
        wins, losses, ties = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
        sign_p = stats.binomtest(wins, wins + losses, 0.5).pvalue
        w_p = float(stats.wilcoxon(d).pvalue)
        boots = np.array([rng.choice(d, len(d)).mean() for _ in range(10000)])
        paired10[f"ACE vs {opp} / {s}"] = {
            "mean_diff_pp": round(float(d.mean()), 3), "wins": wins, "losses": losses, "ties": ties,
            "sign_test_p": round(float(sign_p), 4), "wilcoxon_p": round(w_p, 4),
            "bootstrap95_task_mean_diff": [round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)],
        }
R["V7_robotwin_paired_tasks"] = paired10

# ACE가 낮은 과제의 큰 격차
a_easy = np.array(col10(6, 0), dtype=float)
j_easy = np.array(col10(5, 0), dtype=float)
names = list(T10.keys())
R["V7b_ace_biggest_deficits_vs_joyai_easy"] = sorted(
    [(names[i], int(a_easy[i]), int(j_easy[i]), int(a_easy[i] - j_easy[i])) for i in range(50)], key=lambda x: x[3])[:5]
best_tasks = 0
for i, v in enumerate(T10.values()):
    easy_vals = [v[2 * k] for k in range(7) if v[2 * k] is not None]
    if v[12] == max(easy_vals):
        best_tasks += 1
R["V7c_ace_best_or_tied_easy_tasks"] = best_tasks

# 풀링 이항 검정 (과제당 100회 x 50 = 5,000 시행, 독립 가정). Hy-VLA는 평균만 있음 (표 4)
R["V8_robotwin_pooled"] = {
    "Easy ACE vs JoyAI": two_prop(91.12, 90.48, 5000),
    "Hard ACE vs JoyAI": two_prop(90.62, 89.28, 5000),
    "Easy ACE vs Hy-VLA(90.9)": two_prop(91.12, 90.9, 5000),
    "Hard ACE vs Hy-VLA(90.1)": two_prop(90.62, 90.1, 5000),
    "note": "n = 100 x 50 = 5,000 시행 가정. 과제 쌍체 검정(V7)이 이 표보다 적절",
}

# 그림 1 막대: ACE와 JoyAI는 Hard, ABot-M0는 Easy 값인가
R["V8b_fig1_robotwin_bars"] = {
    "fig1": {"ACE": 90.6, "JoyAI-RA": 89.3, "ABot-M0": 86.1},
    "table_easy": {"ACE": 91.12, "JoyAI-RA": 90.48, "ABot-M0": 86.06},
    "table_hard": {"ACE": 90.62, "JoyAI-RA": 89.28, "ABot-M0": 85.08},
    "reading": "ACE 90.6와 JoyAI-RA 89.3는 Hard 값, ABot-M0 86.1은 Easy 값(Hard는 85.08)과 일치",
}

# ---------------------------------------------------------------- 그림 5(a) 실로봇: 과제당 30회 (그림 판독 값)
TASKS5 = ["Pick Tea", "Scoop Coffee", "Category Sorting", "Sweep Cubes", "Stack Bowls", "Pack Shoes"]
FIG5A = {  # 막대 위 숫자 (%)
    "ACE-Ego": [86.7, 83.3, 86.7, 73.3, 80.0, 60.0],
    "pi0.5": [86.7, 66.7, 76.7, 70.0, 76.7, 53.3],
    "GR00T-N1.7": [3.3, 36.7, 83.3, 6.7, 73.3, 10.0],
}
FIG5A_AVG = {"ACE-Ego": 78.3, "pi0.5": 71.7, "GR00T-N1.7": 35.6}
n30 = 30
counts = {m: [round(v * n30 / 100) for v in vals] for m, vals in FIG5A.items()}
real = {"counts_from_figure_values": counts,
        "integer_ok": {m: all(abs(v * n30 / 100 - round(v * n30 / 100)) < 0.06 for v in vals) for m, vals in FIG5A.items()},
        "avg_recomputed": {m: round(100 * sum(c) / (6 * n30), 2) for m, c in counts.items()},
        "avg_reported_in_figure": FIG5A_AVG}
# 본문 서술 값과 그림 값의 차이
real["text_vs_figure"] = [
    {"where": "5.3절, Scoop Coffee ACE", "text": 86.7, "figure": 83.3},
    {"where": "5.3절, Scoop Coffee pi0.5", "text": 70.0, "figure": 66.7},
    {"where": "5.3절, Category Sorting ACE", "text": 90.0, "figure": 86.7},
    {"where": "5.3절, Category Sorting pi0.5", "text": 80.0, "figure": 76.7},
    {"where": "5.3절, Category Sorting GR00T", "text": 83.3, "figure": 83.3},
    {"where": "5.3절, Scoop Coffee GR00T", "text": 36.7, "figure": 36.7},
    {"where": "5.3절, Stack Bowls GR00T", "text": 73.3, "figure": 73.3},
    {"where": "5.3절, Sweep Cubes GR00T", "text": 6.7, "figure": 6.7},
]
real["avg_if_text_values_used"] = {
    "ACE-Ego": round((86.7 + 86.7 + 90.0 + 73.3 + 80.0 + 60.0) / 6, 2),
    "pi0.5": round((86.7 + 70.0 + 80.0 + 70.0 + 76.7 + 53.3) / 6, 2),
}
tot = {m: sum(c) for m, c in counts.items()}
real["pooled_counts_of_180"] = tot
real["wilson95_pooled"] = {m: [round(x, 3) for x in wilson(k, 180)] for m, k in tot.items()}
tab = [[tot["ACE-Ego"], 180 - tot["ACE-Ego"]], [tot["pi0.5"], 180 - tot["pi0.5"]]]
real["fisher_pooled_ace_vs_pi05"] = {"counts": [tot["ACE-Ego"], tot["pi0.5"]], "p": round(float(stats.fisher_exact(tab)[1]), 4)}
tab = [[tot["ACE-Ego"], 180 - tot["ACE-Ego"]], [tot["GR00T-N1.7"], 180 - tot["GR00T-N1.7"]]]
real["fisher_pooled_ace_vs_gr00t"] = {"counts": [tot["ACE-Ego"], tot["GR00T-N1.7"]], "p": float(stats.fisher_exact(tab)[1])}
per_task = []
for i, t in enumerate(TASKS5):
    a, b = counts["ACE-Ego"][i], counts["pi0.5"][i]
    per_task.append({"task": t, "ace": a, "pi05": b, "fisher_p": round(float(stats.fisher_exact([[a, 30 - a], [b, 30 - b]])[1]), 3)})
real["per_task_fisher_ace_vs_pi05"] = per_task
d = np.array(counts["ACE-Ego"]) - np.array(counts["pi0.5"])
w, l, t = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
real["sign_test_ace_vs_pi05"] = {"wins": w, "losses": l, "ties": t, "p_two_sided": round(float(stats.binomtest(w, w + l, 0.5).pvalue), 4)}
# 가정: 3.3 pp 어긋남 두 과제가 모두 본문 값이었다면 평균이 달라짐
R["V9_real_robot"] = real

# ---------------------------------------------------------------- 5.5절 Sweep Cubes 1/10 대 4/10, 볼록 껍질 4.8배
a, b = 1, 4
R["V10_sweep_cubes_ft"] = {
    "robot_only": {"k": a, "n": 10, "wilson95": [round(x, 3) for x in wilson(a, 10)]},
    "robot_plus_human": {"k": b, "n": 10, "wilson95": [round(x, 3) for x in wilson(b, 10)]},
    "fisher_p_two_sided": round(float(stats.fisher_exact([[b, 10 - b], [a, 10 - a]])[1]), 4),
    "ratio": 4.0,
    "hull_ratio_reported": 4.8, "hull_ratio_recomputed": round(0.296 / 0.062, 3),
    "frames_per_robot_demo": round(45800 / 34, 1), "frames_per_human_episode": round(117500 / 419, 1),
    "note": "시행 10회: 4배는 1/10 대 4/10. 정확 검정으로 구분되지 않음. 그림 5(a)의 Sweep Cubes 73.3%(22/30)는 다른 미세조정 설정으로 보임(데이터 양 미기재)",
}

# ---------------------------------------------------------------- 식 6-8, 20-21, 표 6: 신뢰도 가중의 실제 크기
# 인간 보조 손실의 6개 위치 채널 대 나머지 16개 채널 (22차원 배치: 팔당 위치 3, 회전 6, 그리퍼 1, 활동 플래그 1)
rho_pos, rho_low = 1.0, 0.001
npos, nlow = 6, 22 - 6
share_pos = npos * rho_pos / (npos * rho_pos + nlow * rho_low)
R["V11_channel_weight_share"] = {
    "position_channels": npos, "low_channels": nlow, "rho_low": rho_low,
    "weight_share_on_position_pct": round(100 * share_pos, 2),
    "note": "식 8의 분모 Z가 표본별 합이면 위치 채널이 보조 손실 가중의 99.7%를 차지한다. 활동 플래그 포함 16개를 낮은 채널로 센 값(마스크에 따라 달라짐)",
}


def haux(w_data_left, w_data_right, steps=20, seed=0):
    """식 20-21의 구조만 따라 만든 표본 하나의 보조 손실. 잔차는 난수(고정). Huber(beta=1)."""
    r = np.random.default_rng(seed)
    resid = r.normal(0, 0.8, size=(steps, 22))
    hub = np.where(np.abs(resid) < 1, 0.5 * resid ** 2, np.abs(resid) - 0.5)
    rho = np.full(22, rho_low)
    pos_idx = [0, 1, 2, 11, 12, 13]  # 두 손목의 xyz (왼손 0-10, 오른손 11-21)
    rho[pos_idx] = rho_pos
    hand = np.array([0] * 11 + [1] * 11)
    wd = np.where(hand == 0, w_data_left, w_data_right)
    W = rho[None, :] * wd[None, :] * np.ones((steps, 1))
    return float((W * hub).sum() / W.sum())


R["V12_dataset_prior_cancels"] = {
    "loss_wdata_1.0_1.0": round(haux(1.0, 1.0), 6),
    "loss_wdata_0.25_0.25": round(haux(0.25, 0.25), 6),
    "loss_wdata_0.5_0.5": round(haux(0.5, 0.5), 6),
    "loss_wdata_left0.25_right1.0": round(haux(0.25, 1.0), 6),
    "reading": "표본별 정규화(식 21)에서는 한 표본 안에서 상수인 w_data가 약분된다. 값이 달라지는 것은 왼손/오른손 비율이 다를 때뿐이다",
}

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT)
print(json.dumps({k: R[k] for k in ["V2_table9_means", "V4_robocasa_pooled", "V9_real_robot"]}, ensure_ascii=False)[:3000])
