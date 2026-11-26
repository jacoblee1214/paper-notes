"""RoboFollow (arXiv:2609.25636 v1) 보고 수치의 재계산과 일관성 검사.

입력은 논문 표 1(전체 결과), 표 2(Scene 2 완화 기법), 표 3(미세조정 대조), 부록 C.1 표 10(실로봇), 본문 QA 점수이고,
스크립트에 옮겨 적은 값입니다. 시행 수는 논문에 없으므로 공식 저장소 README의 평가 크기(과제 수 x 10회)를 가정합니다.
공식 저장소 https://github.com/AutoLab-SAI-SJTU/RoboFollow 의 커밋 f642e3d19ddc0a3b0fb5795eef2cd17a12f286d8 을
`git clone <url> repo && git -C repo checkout f642e3d19ddc0a3b0fb5795eef2cd17a12f286d8` 로 받아 두면
프로젝트 사이트용 결과 CSV(repo/docs/assets/data/results.csv)와 옮겨 적은 표 1을 대조합니다(없으면 이 단계는 건너뜁니다).

실행: python3 analysis/recompute.py   (numpy, scipy 필요. 수 초)
결과: analysis/recompute_results.json
"""
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "recompute_results.json"

LEVELS = ["L0", "L1", "L2", "L3"]
METRICS = ["IS", "ES", "CR"]

# 표 1. 행마다 scenes[s] = [L0 IS, ES, CR, L1 IS, ES, CR, L2 IS, ES, CR, L3 IS, ES, CR] (S1..S4), avg = [IS, ES, CR]
TABLE1 = {
    "PI05": {
        "avg": [55.7, 53.0, 47.3],
        "scenes": [
            [99.1, 98.5, 98.8, 45.5, 43.8, 42.5, 58.0, 58.0, 58.1, 44.9, 43.8, 40.1],
            [100.0, 100.0, 100.0, 77.5, 69.4, 68.8, 32.4, 24.6, 20.6, 34.2, 26.7, 16.7],
            [85.0, 85.0, 81.3, 66.9, 66.5, 53.8, 30.0, 29.8, 0.0, 42.5, 42.5, 12.5],
            [84.1, 78.1, 79.6, 34.0, 31.3, 33.3, 33.6, 30.4, 30.0, 22.9, 20.4, 20.8],
        ],
    },
    "PI05_empty": {
        "avg": [12.3, 8.9, 5.2],
        "scenes": [
            [16.0, 16.0, 6.3, 0.5, 0.5, 0.0, 16.6, 16.6, 6.3, 17.4, 16.6, 6.7],
            [16.6, 9.4, 9.4, 18.5, 10.6, 9.6, 23.1, 14.0, 11.2, 19.3, 13.9, 12.9],
            [6.4, 2.1, 0.0, 8.2, 4.1, 0.0, 6.5, 2.0, 0.0, 7.5, 2.8, 0.0],
            [20.4, 16.5, 8.2, 4.7, 4.3, 3.3, 11.0, 10.4, 6.4, 3.7, 2.4, 2.5],
        ],
    },
    "PI0": {
        "avg": [48.4, 44.9, 38.9],
        "scenes": [
            [98.2, 98.2, 97.5, 0.0, 0.0, 0.0, 4.1, 0.0, 0.0, 15.7, 0.8, 0.9],
            [95.6, 95.3, 95.6, 77.5, 67.1, 66.3, 36.3, 27.4, 21.9, 35.2, 29.1, 18.8],
            [96.3, 96.3, 95.0, 69.8, 69.8, 58.0, 29.7, 29.7, 0.2, 45.0, 45.0, 15.3],
            [84.1, 81.9, 81.9, 26.5, 23.9, 20.0, 36.6, 33.1, 32.8, 23.2, 20.4, 17.5],
        ],
    },
    "GROOT": {
        "avg": [32.7, 25.6, 20.5],
        "scenes": [
            [71.4, 67.1, 62.5, 0.0, 0.0, 0.0, 24.0, 24.0, 19.4, 36.0, 34.0, 25.8],
            [70.9, 51.2, 49.4, 23.7, 5.8, 5.4, 16.1, 10.9, 10.3, 12.3, 7.2, 7.5],
            [56.4, 52.3, 48.8, 33.9, 30.5, 19.8, 23.7, 17.0, 0.0, 20.0, 14.4, 0.3],
            [57.8, 38.2, 34.1, 31.9, 28.7, 23.3, 25.3, 15.8, 12.4, 19.6, 13.0, 9.2],
        ],
    },
    "OFT": {
        "avg": [10.7, 4.2, 3.5],
        "scenes": [
            [9.8, 9.5, 6.3, 0.0, 0.0, 0.0, 2.2, 2.2, 0.0, 2.6, 1.8, 1.1],
            [28.9, 10.3, 10.6, 23.3, 2.1, 2.5, 10.2, 4.3, 3.8, 18.9, 10.3, 9.2],
            [7.9, 1.6, 0.6, 7.5, 1.1, 0.0, 6.9, 0.3, 0.0, 6.4, 0.3, 0.0],
            [23.5, 9.6, 7.8, 9.3, 6.7, 6.7, 5.0, 0.0, 0.0, 8.6, 7.7, 7.7],
        ],
    },
    "XVLA": {
        "avg": [34.3, 30.9, 22.3],
        "scenes": [
            [76.7, 74.5, 71.9, 0.0, 0.0, 0.0, 24.2, 23.5, 12.5, 45.7, 44.3, 33.0],
            [76.7, 65.8, 66.9, 6.8, 6.8, 0.0, 38.7, 30.1, 24.3, 16.3, 10.9, 7.5],
            [74.0, 69.9, 45.6, 22.5, 22.2, 5.0, 33.0, 32.9, 0.9, 34.8, 33.0, 4.4],
            [38.5, 34.9, 39.3, 6.9, 5.9, 3.3, 29.4, 23.2, 25.2, 24.5, 17.2, 16.7],
        ],
    },
    "ACoT": {
        "avg": [53.5, 49.3, 42.5],
        "scenes": [
            [100.0, 100.0, 100.0, 8.7, 6.2, 6.3, 60.9, 57.2, 59.4, 48.0, 45.5, 42.1],
            [96.4, 91.6, 89.4, 76.3, 53.4, 52.5, 40.0, 33.5, 26.6, 34.0, 25.0, 18.3],
            [95.6, 95.4, 89.4, 77.5, 77.0, 64.8, 30.3, 30.3, 0.3, 39.7, 39.4, 1.9],
            [78.7, 72.8, 72.6, 23.6, 23.6, 23.3, 26.2, 22.0, 18.4, 20.3, 15.2, 15.4],
        ],
    },
    "Lingbot": {
        "avg": [35.2, 32.5, 27.4],
        "scenes": [
            [80.9, 78.4, 82.5, 0.0, 0.0, 0.0, 24.2, 23.7, 23.8, 46.4, 43.5, 37.2],
            [77.1, 67.6, 74.4, 7.0, 5.8, 0.0, 32.5, 28.0, 23.5, 6.6, 4.1, 1.7],
            [99.1, 98.6, 97.5, 24.8, 24.7, 9.8, 33.2, 32.6, 0.2, 29.8, 29.2, 0.1],
            [58.0, 47.1, 51.1, 8.9, 6.3, 7.3, 18.4, 16.8, 18.4, 16.3, 14.2, 10.8],
        ],
    },
    "MOTUS": {
        "avg": [36.8, 30.8, 27.3],
        "scenes": [
            [73.4, 68.8, 70.6, 11.3, 11.1, 13.8, 26.9, 26.1, 33.1, 29.8, 26.7, 27.0],
            [78.2, 69.9, 81.3, 53.1, 43.6, 50.8, 17.9, 13.6, 8.2, 16.2, 11.0, 8.3],
            [67.6, 61.2, 45.6, 42.6, 37.0, 17.8, 30.9, 26.0, 2.0, 36.6, 31.8, 10.3],
            [51.7, 34.9, 36.3, 11.6, 4.0, 1.3, 20.6, 13.1, 15.2, 20.6, 14.4, 15.8],
        ],
    },
    "FASTWAM": {
        "avg": [39.3, 35.2, 30.3],
        "scenes": [
            [83.1, 83.1, 94.4, 0.0, 0.0, 0.0, 13.0, 4.0, 3.1, 41.9, 36.2, 31.1],
            [82.5, 76.5, 85.0, 32.0, 14.1, 11.7, 24.8, 20.4, 14.6, 18.0, 5.3, 2.9],
            [98.3, 98.1, 95.6, 60.4, 59.7, 42.0, 30.6, 30.5, 0.3, 45.0, 43.3, 4.9],
            [52.1, 48.4, 56.3, 9.1, 6.9, 4.0, 21.1, 20.6, 23.2, 16.7, 15.5, 15.4],
        ],
    },
}
MODEL_LABEL = {
    "PI05": "π0.5", "PI05_empty": "π0.5 (빈 언어)", "PI0": "π0", "GROOT": "GR00T N1.6", "OFT": "OpenVLA-OFT",
    "XVLA": "X-VLA", "ACoT": "ACoT-VLA", "Lingbot": "Lingbot-VLA", "MOTUS": "Motus", "FASTWAM": "FAST-WAM",
}
MAIN_MODELS = [m for m in TABLE1 if m != "PI05_empty"]

# 평가 과제 수(저장소 README 표, 장면 x 수준). 시행 수 = 과제 수 x 10회(README의 --rounds 10 기본값) 가정.
EVAL_TASKS = {"S1": [16, 8, 16, 76], "S2": [16, 24, 68, 24], "S3": [16, 40, 64, 96], "S4": [27, 14, 25, 24]}
ROUNDS = 10
TRAIN_TASKS_PER_GROUP = [16, 16, 16, 16, 7, 4]  # 논문 3절의 Ks
SCENE_GROUPS = {"S1": [16], "S2": [16], "S3": [16], "S4": [16, 7, 4]}

# 표 2 (Scene 2만으로 미세조정): [L0 IS, ES, CR, L1 IS, ES, CR, L2 IS, ES, CR, L3 IS, ES, CR]
TABLE2 = {
    "pi05": [93.6, 89.0, 89.4, 75.7, 42.0, 40.4, 37.8, 30.3, 25.7, 45.8, 24.9, 19.2],
    "pi05 cfg 1.2": [55.9, 44.7, 46.9, 50.4, 25.0, 25.0, 31.2, 22.4, 21.3, 31.2, 14.2, 8.3],
    "pi05 cfg 1.5": [43.6, 25.9, 25.0, 37.8, 23.5, 25.0, 28.6, 20.0, 20.6, 24.6, 12.3, 10.4],
    "qwengroot": [67.9, 63.1, 65.6, 6.3, 0.0, 0.0, 31.1, 21.7, 17.2, 14.8, 1.3, 1.7],
    "qwengroot qa1": [60.1, 46.4, 45.0, 3.5, 0.8, 0.8, 26.4, 15.6, 10.3, 18.9, 8.8, 9.2],
    "qwengroot qa2": [68.2, 61.3, 60.6, 0.0, 0.0, 0.0, 21.8, 16.0, 16.6, 0.0, 0.0, 0.0],
    "qwengroot qa3": [66.6, 50.5, 46.3, 2.9, 0.0, 0.0, 30.8, 19.3, 15.0, 0.0, 0.0, 0.0],
    "langforce": [65.1, 44.2, 35.0, 6.3, 4.0, 0.0, 18.6, 14.7, 14.1, 0.0, 0.0, 0.0],
}

# 표 3 (π0.5, Scene 2 16과제): 의도 점수 L0, L1, L2, L3
TABLE3 = {
    "demos/task=25": [88.4, 69.8, 33.5, 39.8], "demos/task=50": [93.6, 75.7, 37.8, 45.8],
    "variants=1": [93.1, 72.8, 31.6, 41.5], "variants=3": [93.6, 75.7, 37.8, 45.8], "variants=5": [93.1, 79.0, 32.0, 53.0],
    "steps=2k": [86.4, 79.2, 31.5, 38.9], "steps=4k": [92.4, 77.6, 35.7, 47.6], "steps=6k": [93.6, 75.7, 37.8, 45.8],
}

# 부록 C.1 표 10: 지시문별 성공/5회
ROBOT_TRAIN = [("pick", 5), ("pick", 3), ("pick", 5), ("pick", 1), ("stack", 2), ("stack", 3), ("stack", 0), ("stack", 1)]
ROBOT_HELD = [("pick", 4), ("pick", 2), ("pick", 0), ("pick", 0), ("pick", 0), ("pick", 0), ("stack", 0), ("stack", 0)]
ROBOT_TRIALS_PER_INSTRUCTION = 5


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def rnd(x, d=2):
    return None if x is None else round(float(x), d)


def v1_avg_block(res):
    """V1: Avg 블록(16칸 단순 평균)과 보고값의 차이."""
    rows = []
    max_diff = 0.0
    for m, v in TABLE1.items():
        cells = np.array(v["scenes"])  # 4 x 12
        calc = [float(np.mean(cells[:, i::3])) for i in range(3)]  # i = 0 IS, 1 ES, 2 CR (각 수준 열 4개 x 장면 4개)
        diff = [round(c - r, 3) for c, r in zip(calc, v["avg"])]
        max_diff = max(max_diff, max(abs(d) for d in diff))
        rows.append({"model": m, "reported": v["avg"], "recomputed": [rnd(c, 3) for c in calc], "diff": diff})
    return {"rows": rows, "max_abs_diff": round(max_diff, 3),
            "ties_rounding": "보고값은 소수 첫째 자리 반올림. 차이가 0.05 이하이면 반올림 범위"}


def v1b_best_bold(res):
    """V1b: Avg 블록 최고 모델(굵은 글씨 주장)과 2위."""
    out = {}
    for i, m in enumerate(METRICS):
        order = sorted(MAIN_MODELS, key=lambda k: -TABLE1[k]["avg"][i])
        out[m] = {"best": order[0], "value": TABLE1[order[0]]["avg"][i],
                  "second": order[1], "second_value": TABLE1[order[1]]["avg"][i]}
    return out


def level_means():
    """모델별 수준별 4장면 평균 (IS), L0 대 L1–L3 평균과 하락."""
    out = {}
    for m, v in TABLE1.items():
        cells = np.array(v["scenes"])
        lv = {}
        for li, L in enumerate(LEVELS):
            lv[L] = [float(np.mean(cells[:, 3 * li + k])) for k in range(3)]  # IS, ES, CR
        l123 = [float(np.mean([lv[L][k] for L in LEVELS[1:]])) for k in range(3)]
        out[m] = {"levels": {L: [rnd(x, 1) for x in lv[L]] for L in LEVELS},
                  "L1-L3_mean": [rnd(x, 1) for x in l123],
                  "drop_IS": rnd(lv["L0"][0] - l123[0], 1),
                  "rel_drop_IS": rnd(1 - l123[0] / lv["L0"][0], 3) if lv["L0"][0] else None}
    return out


def v2_es_gt_is():
    """V2: ES > IS 또는 CR > IS 인 칸."""
    es_gt, cr_gt = [], []
    for m, v in TABLE1.items():
        for s, row in enumerate(v["scenes"]):
            for li, L in enumerate(LEVELS):
                i, e, c = row[3 * li: 3 * li + 3]
                if e > i + 1e-9:
                    es_gt.append({"model": m, "scene": f"S{s+1}", "level": L, "IS": i, "ES": e})
                if c > i + 1e-9:
                    cr_gt.append({"model": m, "scene": f"S{s+1}", "level": L, "IS": i, "CR": c})
    return {"ES_gt_IS": es_gt, "CR_gt_IS_count": len(cr_gt), "CR_gt_IS_examples": cr_gt[:12],
            "CR_gt_IS_max_gap": max((x["CR"] - x["IS"] for x in cr_gt), default=0)}


def v3_cr_grid():
    """V3: CR 보고값이 (과제 수 x 10회) 격자에서 가능한지. CR = 과제별 완료율의 단순 평균이므로 값은 100/(과제 수 x 10)의 배수."""
    infeasible, checked, per_n = [], 0, {}
    for m, v in TABLE1.items():
        for s, row in enumerate(v["scenes"]):
            for li, L in enumerate(LEVELS):
                cr = row[3 * li + 2]
                n = EVAL_TASKS[f"S{s+1}"][li] * ROUNDS
                g = 100.0 / n
                if g < 0.1:  # 격자가 반올림 단위보다 촘촘함
                    continue
                k = round(cr / g)
                ok = abs(k * g - cr) <= 0.05 + 1e-9
                checked += 1
                d = per_n.setdefault(n, {"n": n, "grid": round(g, 3), "cells": 0, "feasible": 0})
                d["cells"] += 1
                d["feasible"] += int(ok)
                if not ok:
                    infeasible.append({"model": m, "scene": f"S{s+1}", "level": L, "CR": cr, "n": n,
                                       "nearest": round(k * g, 3)})
    # 무작위 값이 우연히 맞을 확률(칸마다 0.1/격자, 최대 1)을 곱한 값(로그)
    log10_p = sum(d["cells"] * math.log10(min(1.0, 0.1 / d["grid"])) for d in per_n.values())
    return {"checked_cells": checked, "infeasible": infeasible, "per_n": sorted(per_n.values(), key=lambda x: x["n"]),
            "log10_chance_all_feasible": round(log10_p, 1)}


def v3b_scan_s4():
    """V3b: 장면 4의 수준별 CR 10개 값(빈 언어 행 포함)이 모두 가능한 시행 수 n(10..600). 과제 수 x 10회 가정과 다른 n이 맞는 칸을 찾는다."""
    out = {}
    for li, L in enumerate(LEVELS):
        vals = [v["scenes"][3][3 * li + 2] for v in TABLE1.values()]
        ok_n = []
        for n in range(10, 601):
            g = 100.0 / n
            if all(abs(round(c / g) * g - c) <= 0.05 + 1e-9 for c in vals):
                ok_n.append(n)
        n_assumed = EVAL_TASKS["S4"][li] * ROUNDS
        out[L] = {"values": vals, "assumed_n": n_assumed,
                  "n_with_all_10_feasible(<=600)": ok_n[:12],
                  "feasible_at_assumed_n": sum(abs(round(c / (100.0 / n_assumed)) * (100.0 / n_assumed) - c) <= 0.05 + 1e-9 for c in vals)}
    return out


def v4_language_free():
    """V4: 장면 엔트로피와 언어를 쓰지 않는 정책의 L0 성공 상한(= 장면 그룹 수/과제 수)."""
    K = sum(TRAIN_TASKS_PER_GROUP)
    h = sum(k / K * math.log2(k) for k in TRAIN_TASKS_PER_GROUP)
    h_scene = {s: sum(k / sum(g) * math.log2(k) for k in g) for s, g in SCENE_GROUPS.items()}
    bound = len(TRAIN_TASKS_PER_GROUP) / K  # 모든 그룹에서 한 과제를 고르면 맞는 비율 Σ p(s) max_t p(t|s)
    bound_s = {s: len(g) / sum(g) for s, g in SCENE_GROUPS.items()}
    # LIBERO 값(논문 부록 A.1): Spatial/Object/Goal/Long = 0/0/3.322/0.200
    libero = [0.0, 0.0, math.log2(10), 0.2]
    ctrl = TABLE1["PI05_empty"]["scenes"]
    ctrl_l0_cr = [r[2] for r in ctrl]
    ctrl_l0_is = [r[0] for r in ctrl]
    # S4는 한 장면 안에 3개 그룹(16, 7, 4)이 있고 그룹당 과제 수가 달라 상한 = 3/27
    # 빈 언어 π0.5의 L0 CR이 상한과 얼마나 다른가(과제 수 x 10회 시행 가정, 이항 z)
    zs = []
    for s in range(4):
        n = EVAL_TASKS[f"S{s+1}"][0] * ROUNDS
        p0 = bound_s[f"S{s+1}"]
        obs = ctrl_l0_cr[s] / 100
        zs.append(round((obs - p0) / math.sqrt(p0 * (1 - p0) / n), 2))
    return {
        "H_scene_bits": round(h, 4), "paper_value": 3.782, "H_per_scene": {s: round(v, 3) for s, v in h_scene.items()},
        "language_free_bound_L0": round(bound, 4), "language_free_bound_per_scene": {s: round(v, 4) for s, v in bound_s.items()},
        "language_free_bound_mean_over_scenes": round(float(np.mean(list(bound_s.values()))), 4),
        "libero_values": [round(x, 3) for x in libero], "libero_equal_weight_mean": round(float(np.mean(libero)), 3), "paper_libero_mean": 0.880,
        "empty_lang_L0_CR": ctrl_l0_cr, "empty_lang_L0_IS": ctrl_l0_is,
        "empty_lang_L0_CR_z_vs_bound(n=tasks*10)": zs,
        "grouping_sensitivity": {
            "S4를 한 그룹(27)으로 본 경우 H": round(sum(k / 75 * math.log2(k) for k in [16, 16, 16, 27]), 3),
            "75개 과제가 한 장면을 공유(상한)": round(math.log2(75), 3),
        },
    }


def v5_vs_control():
    """V5: 빈 언어 π0.5(표 1 둘째 행)를 참고 기준선으로 본 L1–L3 의도 점수. 기준선은 π0.5 하나라 다른 모델에는 참고용."""
    ctrl = TABLE1["PI05_empty"]["scenes"]
    out = {}
    for m in MAIN_MODELS:
        n_le, n_cells, near = 0, 0, 0
        l0_gap = []
        for s, row in enumerate(TABLE1[m]["scenes"]):
            for li in range(1, 4):
                n_cells += 1
                d = row[3 * li] - ctrl[s][3 * li]
                n_le += int(d <= 0)
                near += int(abs(d) <= 5)
            l0_gap.append(row[0] - ctrl[s][0])
        out[m] = {"L1-L3 cells with IS <= control IS": n_le, "cells within 5pp": near, "of": n_cells,
                  "L0 IS minus control (S1..S4)": [rnd(x, 1) for x in l0_gap],
                  "L0 mean IS minus control mean": rnd(float(np.mean(l0_gap)), 1)}
    return out


def v6_s3_l2():
    """V6: Scene 3 L2의 의도 점수가 모델 사이에서 한 값 근처에 모이는지, CR이 0 근처인지."""
    rows = {m: {"IS": TABLE1[m]["scenes"][2][6], "ES": TABLE1[m]["scenes"][2][7], "CR": TABLE1[m]["scenes"][2][8]} for m in MAIN_MODELS}
    is_vals = np.array([r["IS"] for r in rows.values()])
    near30 = [m for m, r in rows.items() if 29.0 <= r["IS"] <= 34.0]
    return {"rows": rows, "n_models_IS_in_29_34": len(near30), "models_in_29_34": near30,
            "IS_range_these": [min(rows[m]["IS"] for m in near30), max(rows[m]["IS"] for m in near30)],
            "CR_max": max(r["CR"] for r in rows.values()), "CR_le_2": sum(r["CR"] <= 2.0 for r in rows.values()),
            "S3_stage_weights_in_code": [0.3, 0.3, 0.4]}


def v7_table23():
    """V7: 표 2, 표 3의 서술 수치 재계산."""
    t3 = TABLE3
    gaps = {k: {"L0-L2": rnd(v[0] - v[2], 1), "L0-L3": rnd(v[0] - v[3], 1)} for k, v in t3.items()}
    # 표 2 pi05 행의 IS = 표 3 (50 demos, 3 variants, 6k) 행
    t2_pi05_is = [TABLE2["pi05"][0], TABLE2["pi05"][3], TABLE2["pi05"][6], TABLE2["pi05"][9]]
    same_rows = [k for k, v in t3.items() if v == t2_pi05_is]
    # L0 -> L1 하락
    drops = {k: {"L0-L1_IS": rnd(v[0] - v[3], 1), "L0_IS": v[0], "L1_IS": v[3], "L0_CR": v[2], "L1_CR": v[5]} for k, v in TABLE2.items()}
    # LangForce와 qwengroot 행의 차이
    lf, qg = np.array(TABLE2["langforce"]), np.array(TABLE2["qwengroot"])
    lf_minus_qg = {f"{L} {M}": rnd(lf[3 * i + j] - qg[3 * i + j], 1) for i, L in enumerate(LEVELS) for j, M in enumerate(METRICS)}
    # Scene 2 시행 수 가정(과제 수 x 10회)에서 의도 점수의 표준오차 상한 sqrt(p(1-p)/n) (점수가 [0,1]이면 분산은 p(1-p) 이하)
    n_s2 = [t * ROUNDS for t in EVAL_TASKS["S2"]]
    def se_ub(p, n):
        p = p / 100
        return 100 * math.sqrt(p * (1 - p) / n)
    comp = {}
    for name_a, name_b in [("demos/task=25", "demos/task=50"), ("variants=1", "variants=5"), ("variants=1", "variants=3"),
                           ("steps=2k", "steps=6k"), ("steps=2k", "steps=4k")]:
        a, b = t3[name_a], t3[name_b]
        row = {}
        for li, L in enumerate(LEVELS):
            d = b[li] - a[li]
            se = math.sqrt(se_ub(a[li], n_s2[li]) ** 2 + se_ub(b[li], n_s2[li]) ** 2)
            row[L] = {"diff": rnd(d, 1), "se_upper_bound": rnd(se, 1), "z_lower_bound": rnd(d / se, 2)}
        comp[f"{name_a} -> {name_b}"] = row
    return {"gaps_table3": gaps, "table2_pi05_IS_equals_table3_rows": same_rows, "table2_drops": drops,
            "langforce_minus_qwengroot": lf_minus_qg, "n_trials_scene2_assumed": n_s2, "table3_comparisons": comp}


def v8_robot():
    """V8: 실로봇 파일럿 표 10의 합계, Wilson 구간, 검정."""
    tr = [k for _, k in ROBOT_TRAIN]
    he = [k for _, k in ROBOT_HELD]
    n = ROBOT_TRIALS_PER_INSTRUCTION
    k1, k2 = sum(tr), sum(he)
    n1 = n2 = n * 8
    w1, w2 = wilson(k1, n1), wilson(k2, n2)
    fisher = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
    # 범주별
    def cat(rows, c):
        ks = [k for cc, k in rows if cc == c]
        return sum(ks), n * len(ks)
    pick_tr, pick_he = cat(ROBOT_TRAIN, "pick"), cat(ROBOT_HELD, "pick")
    stack_tr, stack_he = cat(ROBOT_TRAIN, "stack"), cat(ROBOT_HELD, "stack")
    f_pick = stats.fisher_exact([[pick_tr[0], pick_tr[1] - pick_tr[0]], [pick_he[0], pick_he[1] - pick_he[0]]])
    f_stack = stats.fisher_exact([[stack_tr[0], stack_tr[1] - stack_tr[0]], [stack_he[0], stack_he[1] - stack_he[0]]])
    # Mantel-Haenszel (pick, stack 층)
    tables = [np.array([[pick_tr[0], pick_tr[1] - pick_tr[0]], [pick_he[0], pick_he[1] - pick_he[0]]]),
              np.array([[stack_tr[0], stack_tr[1] - stack_tr[0]], [stack_he[0], stack_he[1] - stack_he[0]]])]
    num = sum(t[0, 0] * t[1, 1] / t.sum() for t in tables)
    den = sum(t[0, 1] * t[1, 0] / t.sum() for t in tables)
    or_mh = num / den if den else float("inf")
    # CMH 통계(연속성 보정 없음)
    e = sum((t[0].sum() * t[:, 0].sum()) / t.sum() for t in tables)
    v = sum(t[0].sum() * t[1].sum() * t[:, 0].sum() * t[:, 1].sum() / (t.sum() ** 2 * (t.sum() - 1)) for t in tables)
    a = sum(t[0, 0] for t in tables)
    cmh = (a - e) ** 2 / v
    p_cmh = 1 - stats.chi2.cdf(cmh, 1)
    # 지시문 단위 순열 검정(8 대 8, 지시문별 성공률)
    r1 = np.array(tr) / n
    r2 = np.array(he) / n
    allr = np.concatenate([r1, r2])
    obs = r1.mean() - r2.mean()
    cnt = tot = 0
    for idx in itertools.combinations(range(16), 8):
        mask = np.zeros(16, bool)
        mask[list(idx)] = True
        d = allr[mask].mean() - allr[~mask].mean()
        tot += 1
        cnt += int(abs(d) >= abs(obs) - 1e-12)
    # 지시문 단위 Welch t, Mann-Whitney
    t_w = stats.ttest_ind(r1, r2, equal_var=False)
    mw = stats.mannwhitneyu(r1, r2, alternative="two-sided")
    # 군집(지시문) 내 상관을 반영한 유효 표본 크기 개략: 지시문 단위 표준편차
    return {
        "train": {"k": k1, "n": n1, "rate": k1 / n1, "wilson95": [round(100 * x, 1) for x in w1]},
        "held_out": {"k": k2, "n": n2, "rate": k2 / n2, "wilson95": [round(100 * x, 1) for x in w2]},
        "paper_wilson": {"train": [35.2, 64.8], "held_out": [7.1, 29.1]},
        "fisher_pooled": {"odds_ratio": rnd(fisher[0], 2), "p": float(f"{fisher[1]:.3g}")},
        "pick": {"train": pick_tr, "held": pick_he, "fisher_p": float(f"{f_pick[1]:.3g}")},
        "stack": {"train": stack_tr, "held": stack_he, "fisher_p": float(f"{f_stack[1]:.3g}")},
        "cmh": {"or_mh": rnd(or_mh, 2), "chi2": rnd(cmh, 2), "p": float(f"{p_cmh:.3g}")},
        "instruction_level": {"train_rates": r1.tolist(), "held_rates": r2.tolist(), "mean_diff": rnd(obs, 3),
                              "perm_p_exact(12870 splits)": round(cnt / tot, 4), "welch_t_p": float(f"{t_w.pvalue:.3g}"),
                              "mannwhitney_p": float(f"{mw.pvalue:.3g}"),
                              "sd_train": rnd(r1.std(ddof=1), 3), "sd_held": rnd(r2.std(ddof=1), 3)},
        "sums_match_text": {"train_20_of_40": k1 == 20, "held_6_of_40": k2 == 6, "pick_14_20": pick_tr == (14, 20),
                            "pick_6_30": pick_he == (6, 30), "stack_6_20": stack_tr == (6, 20), "stack_0_10": stack_he == (0, 10)},
        "held_out_zero_instructions": int(sum(1 for k in he if k == 0)),
    }


def v9_qa():
    """V9: VQA 프로브(20문항)의 정확도 비교."""
    def f(a, b):
        r = stats.fisher_exact([[a, 20 - a], [b, 20 - b]])
        return float(f"{r[1]:.3g}")
    return {"paligemma_1_vs_pi05_2": f(1, 2), "pi05_2_vs_finetuned_3": f(2, 3), "paligemma_1_vs_finetuned_3": f(1, 3),
            "finetuned_3_vs_qwen3vl4b_19": f(3, 19),
            "wilson95": {"3/20": [round(100 * x, 1) for x in wilson(3, 20)], "19/20": [round(100 * x, 1) for x in wilson(19, 20)]}}


def v10_counts():
    """V10: 저장소 README의 개수와 논문 개수의 산술 대조."""
    inst = {"S1": 48, "S2": 52, "S3": 48, "S4": 81}
    ev = {s: sum(v) for s, v in EVAL_TASKS.items()}
    levels = [sum(EVAL_TASKS[s][i] for s in EVAL_TASKS) for i in range(4)]
    return {"train_tasks": sum(TRAIN_TASKS_PER_GROUP), "train_episodes_50_each": 50 * sum(TRAIN_TASKS_PER_GROUP),
            "paper_train_episodes": 3750, "scene2_episodes_16_tasks": 16 * 50, "paper_scene2_episodes": 800,
            "instruction_variants_total": sum(inst.values()), "readme_variants_total": 229,
            "eval_tasks_per_scene": ev, "eval_tasks_per_level": levels, "eval_tasks_total": sum(ev.values()),
            "readme_eval_tasks_total": 554, "full_protocol_trials": sum(ev.values()) * ROUNDS, "readme_trials": 5540,
            "scene2_mean_variants_per_task": round(52 / 16, 3), "scene2_tasks_with_5_variants": 2, "scene2_tasks_with_3_variants": 14}


def v11_avg_se():
    """V11: 모델 평균 CR의 대략적 불확실성(시행 수 가정, 이항 근사, 과제 군집 무시)과 상위 모델 차이."""
    def se_avg(m):
        s2 = 0.0
        for s, row in enumerate(TABLE1[m]["scenes"]):
            for li in range(4):
                p = row[3 * li + 2] / 100
                n = EVAL_TASKS[f"S{s+1}"][li] * ROUNDS
                s2 += p * (1 - p) / n
        return 100 * math.sqrt(s2) / 16
    out = {m: {"avg_CR": TABLE1[m]["avg"][2], "se_binomial_lower_bound": rnd(se_avg(m), 2)} for m in MAIN_MODELS}
    a, b = "PI05", "ACoT"
    d = TABLE1[a]["avg"][2] - TABLE1[b]["avg"][2]
    se = math.sqrt(se_avg(a) ** 2 + se_avg(b) ** 2)
    # 과제 단위 군집(554과제)을 고려한 거친 상한: 평균 CR의 표준오차 ≈ sqrt(p(1-p)/554 )
    clus = {m: rnd(100 * math.sqrt(TABLE1[m]["avg"][2] / 100 * (1 - TABLE1[m]["avg"][2] / 100) / 554), 2) for m in (a, b)}
    return {"per_model": out, "pi05_vs_acot": {"diff": rnd(d, 1), "se_trial_level": rnd(se, 2), "z_trial_level": rnd(d / se, 2),
                                              "se_task_level_each(554 tasks, 거친 값)": clus,
                                              "z_task_level(독립 가정)": rnd(d / math.sqrt(sum(v ** 2 for v in clus.values())), 2)}}


def v12_weighted_avg():
    """V12: 표 1 Avg(16칸 단순 평균)와 평가 과제 수로 가중한 평균(554과제)의 비교와 순위."""
    w = np.array([[EVAL_TASKS[f"S{s+1}"][li] for li in range(4)] for s in range(4)], float)
    w = w / w.sum()
    out = {}
    for m, v in TABLE1.items():
        cells = np.array(v["scenes"])
        out[m] = [rnd(float(sum(w[s, li] * cells[s, 3 * li + k] for s in range(4) for li in range(4))), 1) for k in range(3)]
    ranks = {}
    for k, name in enumerate(METRICS):
        simple = sorted(MAIN_MODELS, key=lambda m: -TABLE1[m]["avg"][k])
        weighted = sorted(MAIN_MODELS, key=lambda m: -out[m][k])
        ranks[name] = {"simple": simple, "task_weighted": weighted, "same_order": simple == weighted}
    return {"weighted_avg": out, "ranks": ranks, "weights_by_level": [round(float(x), 3) for x in w.sum(axis=0)],
            "weights_by_scene": [round(float(x), 3) for x in w.sum(axis=1)]}


def wilson_half_widths():
    out = {}
    for n in sorted({t * ROUNDS for v in EVAL_TASKS.values() for t in v}):
        lo, hi = wilson(round(0.5 * n), n)
        out[n] = rnd(50 * (hi - lo), 1)
    return out


def site_crosscheck():
    path = HERE.parent / "repo" / "docs" / "assets" / "data" / "results.csv"
    if not path.exists():
        return {"checked": False, "note": "repo/ 없음. 건너뜀"}
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    labels = []
    for r in rows:
        if r["model"] not in labels:
            labels.append(r["model"])
    names = list(TABLE1)
    n_cells = n_bad = 0
    for mi, m in enumerate(names):
        for s in range(4):
            for li, L in enumerate(LEVELS):
                r = next(x for x in rows if x["model"] == labels[mi] and x["scene"] == f"S{s+1}" and x["level"] == L)
                vals = [float(r["intent_score"]), float(r["execution_score"]), float(r["completion_rate"])]
                n_cells += 1
                n_bad += int(vals != TABLE1[m]["scenes"][s][3 * li: 3 * li + 3])
    return {"checked": True, "cells": n_cells, "mismatch": n_bad, "csv_rows": len(rows)}


def main():
    res = {
        "V1_avg_block": v1_avg_block(None),
        "V1b_best_models": v1b_best_bold(None),
        "level_means": level_means(),
        "V2_es_cr_vs_is": v2_es_gt_is(),
        "V3_cr_grid": v3_cr_grid(),
        "V3b_scene4_trial_count_scan": v3b_scan_s4(),
        "V4_entropy_language_free": v4_language_free(),
        "V5_vs_empty_language": v5_vs_control(),
        "V6_scene3_L2": v6_s3_l2(),
        "V7_tables_2_3": v7_table23(),
        "V8_real_robot": v8_robot(),
        "V9_qa_probe": v9_qa(),
        "V10_counts": v10_counts(),
        "V11_avg_uncertainty": v11_avg_se(),
        "V12_weighted_avg": v12_weighted_avg(),
        "wilson_half_width_p50_by_n": wilson_half_widths(),
        "site_csv_crosscheck": site_crosscheck(),
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    # 요약 출력
    print("V1 Avg 블록 최대 차이:", res["V1_avg_block"]["max_abs_diff"])
    print("V2 ES>IS 칸:", len(res["V2_es_cr_vs_is"]["ES_gt_IS"]), "| CR>IS 칸:", res["V2_es_cr_vs_is"]["CR_gt_IS_count"])
    g = res["V3_cr_grid"]
    print("V3 격자 검사 칸:", g["checked_cells"], "불가능:", len(g["infeasible"]), "log10(우연히 모두 가능):", g["log10_chance_all_feasible"])
    e = res["V4_entropy_language_free"]
    print("V4 H =", e["H_scene_bits"], "상한 =", e["language_free_bound_L0"], "빈 언어 L0 CR z:", e["empty_lang_L0_CR_z_vs_bound(n=tasks*10)"])
    r = res["V8_real_robot"]
    print("V8 Fisher p =", r["fisher_pooled"]["p"], "CMH p =", r["cmh"]["p"], "지시문 순열 p =", r["instruction_level"]["perm_p_exact(12870 splits)"])
    print("사이트 CSV 대조:", res["site_csv_crosscheck"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
