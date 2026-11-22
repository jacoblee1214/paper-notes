"""HuRo (arXiv:2609.10706 v5) 표 수치 재계산, 정수 가능성 검사, 구간·검정, 데이터셋 산술.

실행: python3 analysis/recompute.py  → analysis/recompute_results.json (수 초, CPU)
입력: 논문 표 값은 이 파일 안에 옮겨 적었고, v1 ↔ v5 표 일치는 paper/paper_v1.txt, paper/paper.txt를
      읽어 확인한다(파일이 없으면 그 검사만 건너뛴다).
추정으로 채운 값은 결과 JSON의 키에 `_est`를 붙인다. 시행 수는 논문 부록 A, 4.4, E.1의 값이다.
"""
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "recompute_results.json"
res = {}

# ---------------------------------------------------------------- 표 2 (부록 A.3)
# 열: Apple ID, Apple OOD, Cup ID, Cup OOD, Noodle ID, Noodle OOD, Micro OOD, 평균 ID, 평균 OOD, Overall
TABLE2 = {
    "pi0.5":        [91.7, 41.7, 91.7, 47.9, 22.2, 22.2, 0.0, 68.5, 28.0, 48.2],
    "GR00T N1.6":   [83.3, 33.3, 50.0, 64.6, 66.7, 41.7, 10.0, 66.7, 37.4, 52.0],
    "0% PT":        [83.3, 25.0, 45.8, 58.3, 75.0, 36.1, 20.0, 68.1, 34.9, 51.5],
    "10% PT":       [75.0, 41.7, 75.0, 60.4, 80.6, 69.4, 66.7, 76.9, 59.5, 68.2],
    "50% PT":       [83.3, 50.0, 79.2, 64.6, 72.2, 94.4, 70.0, 78.2, 69.8, 74.0],
    "no-overlay":   [91.7, 33.3, 87.5, 64.6, 88.9, 75.0, 50.0, 89.4, 55.7, 72.5],
    "100% PT":      [83.3, 50.0, 87.5, 79.2, 94.4, 86.1, 73.3, 88.4, 72.2, 80.3],
}
COLS = ["apple_id", "apple_ood", "cup_id", "cup_ood", "noodle_id", "noodle_ood", "micro_ood", "avg_id", "avg_ood", "overall"]
ID_IDX, OOD_IDX = [0, 2, 4], [1, 3, 5, 6]
# 롤아웃 수 (표 1, 부록 A.2)와 과제당 만점 (사과는 이진, 나머지는 세 하위 목표)
N_ROLL = {"apple_id": 12, "apple_ood": 12, "cup_id": 12, "cup_ood": 24, "noodle_id": 12, "noodle_ood": 12, "micro_ood": 10}
MAXSC = {"apple_id": 1, "apple_ood": 1, "cup_id": 3, "cup_ood": 3, "noodle_id": 3, "noodle_ood": 3, "micro_ood": 3}


def feasible(value, denom):
    """value(%, 소수 1자리 반올림)가 k/denom (정수 k)로 나올 수 있는가. 가능하면 k, 아니면 None."""
    for k in range(denom + 1):
        if abs(round(100.0 * k / denom + 1e-9, 1) - value) < 1e-9:
            return k
    return None


def min_denoms(values, limit=200):
    """모든 값이 정수 비율로 나오는 가장 작은 분모들 (limit 이하, 앞 3개)."""
    ok = [d for d in range(1, limit + 1) if all(feasible(v, d) is not None for v in values)]
    return ok[:3]


# (a) 평균 재계산
avg_check = {}
for m, row in TABLE2.items():
    id_avg = float(np.mean([row[i] for i in ID_IDX]))
    ood_avg = float(np.mean([row[i] for i in OOD_IDX]))
    overall = (id_avg + ood_avg) / 2
    avg_check[m] = {
        "id_recomputed": round(id_avg, 2), "id_reported": row[7],
        "ood_recomputed": round(ood_avg, 2), "ood_reported": row[8],
        "overall_recomputed": round(overall, 2), "overall_from_reported_avgs": round((row[7] + row[8]) / 2, 2),
        "overall_reported": row[9],
        "match": abs(id_avg - row[7]) <= 0.1 and abs(ood_avg - row[8]) <= 0.1 and abs((row[7] + row[8]) / 2 - row[9]) <= 0.1,
    }
res["table2_average_check"] = avg_check

# (b) 정수 가능성: 시행 수 × 만점이 분모
feas = {}
infeasible = []
for m, row in TABLE2.items():
    for c, v in zip(COLS[:7], row[:7]):
        denom = N_ROLL[c] * MAXSC[c]
        k = feasible(v, denom)
        feas[f"{m}|{c}"] = {"value": v, "denom_stated": denom, "k": k}
        if k is None:
            infeasible.append(f"{m}|{c}={v} (분모 {denom})")
res["table2_integer_feasibility"] = {"cells": feas, "infeasible": infeasible, "n_infeasible": len(infeasible)}

# 열별로 모든 모델 값을 설명하는 최소 분모
col_denoms = {}
for j, c in enumerate(COLS[:7]):
    vals = [TABLE2[m][j] for m in TABLE2]
    col_denoms[c] = {"stated": N_ROLL[c] * MAXSC[c], "min_denoms_consistent_with_all_values": min_denoms(vals)}
res["table2_column_denominators"] = col_denoms

# 컵 쌓기 분모 가설: 롤아웃 × 2 (하위 목표 3개가 아니라 만점 2로 환산한 것처럼 보임)
cup_h = {}
for c, n in (("cup_id", 12), ("cup_ood", 24)):
    vals = [TABLE2[m][COLS.index(c)] for m in TABLE2]
    cup_h[c] = {"rollouts": n,
                "all_feasible_if_max3": all(feasible(v, n * 3) is not None for v in vals),
                "all_feasible_if_max2": all(feasible(v, n * 2) is not None for v in vals),
                "all_feasible_if_n_binary": all(feasible(v, n) is not None for v in vals)}
res["cup_stacking_denominator_hypotheses"] = cup_h

# (c) 과제별 단조성 (0% → 10% → 50% → 100%)
mono = {}
for j, c in enumerate(COLS[:7]):
    seq = [TABLE2[m][j] for m in ("0% PT", "10% PT", "50% PT", "100% PT")]
    mono[c] = {"seq": seq, "non_decreasing": all(b >= a for a, b in zip(seq, seq[1:]))}
res["monotonic_by_task_cell"] = {"cells": mono, "n_monotone": sum(v["non_decreasing"] for v in mono.values()), "n_cells": len(mono)}

# (d) 이득의 구성: Overall 이득에서 과제별 기여
base, full = TABLE2["0% PT"], TABLE2["100% PT"]
gain = {c: round(full[j] - base[j], 1) for j, c in enumerate(COLS[:7])}
ood3_base = np.mean([base[1], base[3], base[5]]); ood3_full = np.mean([full[1], full[3], full[5]])
res["gain_decomposition"] = {
    "gain_per_cell_100_vs_0": gain,
    "ood_avg_gain_4tasks": round(float(np.mean([full[i] for i in OOD_IDX]) - np.mean([base[i] for i in OOD_IDX])), 2),
    "ood_avg_gain_without_microwave_3tasks": round(float(ood3_full - ood3_base), 2),
    "microwave_contribution_to_ood_gain": round((full[6] - base[6]) / 4, 2),
    "id_avg_gain_3tasks": round(float(np.mean([full[i] for i in ID_IDX]) - np.mean([base[i] for i in ID_IDX])), 2),
    "ood_share_of_gain_at_10pct": round(float((TABLE2["10% PT"][8] - base[8]) / (full[8] - base[8])), 3),
    "ood_gain_50_to_100": round(full[8] - TABLE2["50% PT"][8], 1),
    "id_gain_50_to_100": round(full[7] - TABLE2["50% PT"][7], 1),
}
res["ref_models_per_task_wins"] = {
    "100pct_beats_pi05_cells": [c for j, c in enumerate(COLS[:7]) if full[j] > TABLE2["pi0.5"][j]],
    "100pct_not_beating_pi05_cells": [c for j, c in enumerate(COLS[:7]) if full[j] <= TABLE2["pi0.5"][j]],
    "100pct_not_beating_groot_cells": [c for j, c in enumerate(COLS[:7]) if full[j] <= TABLE2["GR00T N1.6"][j]],
}


# ---------------------------------------------------------------- 구간과 검정
def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * max(0, c - h), 1), round(100 * min(1, c + h), 1)]


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


# 사과(이진, 12회): 모델 간 Fisher와 Wilson
apple = {}
for m in TABLE2:
    for tag, j in (("id", 0), ("ood", 1)):
        k = round(TABLE2[m][j] * 12 / 100)
        apple[f"{m}|{tag}"] = {"k": k, "n": 12, "wilson95": wilson(k, 12)}
res["apple_binary_wilson"] = apple
res["apple_fisher_12"] = {
    "ood_0_vs_100": fisher(3, 12, 6, 12),
    "ood_0_vs_10": fisher(3, 12, 5, 12),
    "ood_100_vs_no_overlay": fisher(6, 12, 4, 12),
    "id_0_vs_100": fisher(10, 12, 10, 12),
}


def simulate(models_a, models_b, which, n_eff_mode="rollouts", draws=200000, seed=0):
    """과제 단위 비율에 Beta(k+0.5, n-k+0.5) 사후분포를 두고 비가중 평균 차이 (a − b)를 시뮬레이션한다.
    n_eff_mode='rollouts': 롤아웃 수를 유효 표본으로 (점수가 [0,1]이므로 분산 상한이라 보수적).
    n_eff_mode='subgoals': 롤아웃 × 만점 (하위 목표가 독립이라는 낙관적 가정)."""
    rng = np.random.default_rng(seed)
    idx = {"id": ID_IDX, "ood": OOD_IDX}[which]

    def draw(model):
        cols = []
        for j in idx:
            c = COLS[j]
            n = N_ROLL[c] * (MAXSC[c] if n_eff_mode == "subgoals" else 1)
            p = TABLE2[model][j] / 100.0
            k = p * n
            cols.append(rng.beta(k + 0.5, n - k + 0.5, size=draws))
        return np.mean(cols, axis=0)

    d = 100 * (draw(models_a) - draw(models_b))
    return {"mean_diff": round(float(d.mean()), 1), "ci95": [round(float(np.percentile(d, 2.5)), 1), round(float(np.percentile(d, 97.5)), 1)],
            "p_diff_gt_0": round(float((d > 0).mean()), 3)}


pairs = [("100% PT", "0% PT"), ("100% PT", "10% PT"), ("100% PT", "50% PT"), ("50% PT", "10% PT"), ("10% PT", "0% PT"),
         ("100% PT", "no-overlay"), ("no-overlay", "10% PT"), ("100% PT", "GR00T N1.6"), ("100% PT", "pi0.5"),
         ("0% PT", "GR00T N1.6")]
sim = {}
for a, b in pairs:
    for which in ("id", "ood"):
        sim[f"{a} − {b} | {which}"] = {
            "rollout_level_conservative": simulate(a, b, which, "rollouts"),
            "subgoal_level_optimistic": simulate(a, b, which, "subgoals"),
        }
res["table2_aggregate_difference_simulation"] = sim


# 모델별 평균 구간 (롤아웃 수를 유효 표본으로 둔 보수적 가정, 비가중 평균)
def model_interval(model, which, draws=100000, seed=1):
    rng = np.random.default_rng(seed)
    cols = []
    for j in {"id": ID_IDX, "ood": OOD_IDX}[which]:
        c = COLS[j]; n = N_ROLL[c]; k = TABLE2[model][j] / 100.0 * n
        cols.append(rng.beta(k + 0.5, n - k + 0.5, size=draws))
    d = 100 * np.mean(cols, axis=0)
    return {"mean": round(float(d.mean()), 1), "ci95": [round(float(np.percentile(d, 2.5)), 1), round(float(np.percentile(d, 97.5)), 1)]}


res["model_average_intervals_rollout_level"] = {m: {w: model_interval(m, w) for w in ("id", "ood")} for m in TABLE2}

# ---------------------------------------------------------------- 그림 5, 6, 표 3 (Diverse Pick-and-Place)
# 만점 2/롤아웃, ID 36회(72점), OOD 18회(36점). 물체 단위 이진(1점 = 한 물체 성공)으로 보면 분모 72, 36.
DPP = {
    "fig5_ID": {"No PT": 25.0, "Visual only": 31.9, "Visual+Action": 61.1},
    "fig5_OOD": {"No PT": 13.9, "Visual only": 22.2, "Visual+Action": 50.0},
    "table3_ID": {"No PT": 63.9, "Fixed-EEF": 75.0, "Fixed-EEF+Hand": 80.6, "HuRo-EEF+Hand": 86.1},
    "table3_OOD": {"No PT": 25.0, "Fixed-EEF": 50.0, "Fixed-EEF+Hand": 61.1, "HuRo-EEF+Hand": 63.9},
    "fig6_ID_HuRo": {"0.7M": 83.3, "3.5M": 83.3, "7M": 97.2},
    "fig6_ID_I2V": {"0.7M": 72.2, "3.5M": 75.0, "7M": 80.6},
    "fig6_OOD_HuRo": {"0.7M": 63.9, "3.5M": 69.4, "7M": 83.3},
    "fig6_OOD_I2V": {"0.7M": 41.7, "3.5M": 55.6, "7M": 55.6},
    "appE3_joint": {"ALLEX-only ID": 86.1, "ALLEX-only OOD": 63.9, "joint ID": 91.7, "joint OOD": 77.8},
}
dpp_feas = {}
for grp, d in DPP.items():
    denom = 72 if ("ID" in grp or "joint ID" in d) else 36
    for k_, v in d.items():
        dn = 72 if (("ID" in grp and "OOD" not in grp) or k_.endswith("ID")) else 36
        dpp_feas[f"{grp}|{k_}"] = {"value": v, "denom": dn, "k": feasible(v, dn)}
res["diverse_pnp_integer_feasibility"] = {"cells": dpp_feas,
                                           "infeasible": [k for k, v in dpp_feas.items() if v["k"] is None]}


def pnp_cmp(name, ka, na, kb, nb):
    return {"cmp": name, "p_object_level_fisher": round(fisher(ka, na, kb, nb), 4),
            "a_wilson": wilson(ka, na), "b_wilson": wilson(kb, nb)}


dpp_tests = [
    pnp_cmp("fig5 ID: V+A(44/72) vs No PT(18/72), 물체 단위", 44, 72, 18, 72),
    pnp_cmp("fig5 ID: V+A(44/72) vs Visual only(23/72)", 44, 72, 23, 72),
    pnp_cmp("fig5 OOD: V+A(18/36) vs No PT(5/36)", 18, 36, 5, 36),
    pnp_cmp("fig5 OOD: V+A(18/36) vs Visual only(8/36)", 18, 36, 8, 36),
    pnp_cmp("fig6 OOD 7M: HuRo(30/36) vs I2V(20/36)", 30, 36, 20, 36),
    pnp_cmp("fig6 OOD 0.7M HuRo(23/36) vs I2V 7M(20/36)", 23, 36, 20, 36),
    pnp_cmp("fig6 ID 0.7M HuRo(60/72) vs I2V 7M(58/72)", 60, 72, 58, 72),
    pnp_cmp("table3 OOD: HuRo-EEF+Hand(23/36) vs Fixed-EEF+Hand(22/36)", 23, 36, 22, 36),
    pnp_cmp("table3 OOD: HuRo-EEF+Hand(23/36) vs No PT(9/36)", 23, 36, 9, 36),
    pnp_cmp("appE3 OOD: joint(28/36) vs ALLEX-only(23/36)", 28, 36, 23, 36),
]
# 롤아웃 단위(보수적): 물체 점수를 롤아웃당 0–2점에서 상한 분산(이진)으로 취급하면 n = 36, 18
dpp_tests_rollout = [
    pnp_cmp("fig5 ID (롤아웃 36회 가정): V+A 61.1% vs No PT 25.0%", round(0.611 * 36), 36, round(0.25 * 36), 36),
    pnp_cmp("fig5 OOD (롤아웃 18회 가정): V+A 50.0% vs No PT 13.9%", round(0.5 * 18), 18, round(0.139 * 18), 18),
    pnp_cmp("fig6 OOD 7M (롤아웃 18회 가정): HuRo 83.3% vs I2V 55.6%", round(0.833 * 18), 18, round(0.556 * 18), 18),
    pnp_cmp("table3 OOD (롤아웃 18회 가정): HuRo-EEF+Hand 63.9% vs Fixed-EEF+Hand 61.1%", round(0.639 * 18), 18, round(0.611 * 18), 18),
]
res["diverse_pnp_tests_object_level"] = dpp_tests
res["diverse_pnp_tests_rollout_level_conservative"] = dpp_tests_rollout

# ---------------------------------------------------------------- 표 4 (인간 도메인 비교, OOD)
T4 = {"cup_stacking": {"No PT": 31.9, "Human-HRDT": 5.6, "Human-VITRA": 0.0, "Ours 100%": 70.8},
      "cup_noodle": {"No PT": 16.7, "Human-HRDT": 37.5, "Human-VITRA": 9.7, "Ours 100%": 66.7}}
t4 = {}
for task, d in T4.items():
    for m, v in d.items():
        t4[f"{task}|{m}"] = {"value": v, "k_over_72": feasible(v, 72), "k_over_36": feasible(v, 36)}
t4h = {}
for task, n in (("cup_stacking", 24), ("cup_noodle", 12)):
    vals = list(T4[task].values())
    t4h[task] = {"rollouts_stated": n, "min_denoms": min_denoms(vals),
                                         "feasible_if_stated_x3": all(feasible(v, n * 3) is not None for v in vals),
                                         "feasible_if_2x_rollouts_x3": all(feasible(v, 2 * n * 3) is not None for v in vals)}
res["table4_integer_feasibility"] = {"cells": t4, "column_hypotheses": t4h,
                                     "infeasible_at_36": [k for k, v in t4.items() if v["k_over_36"] is None],
                                     "infeasible_at_72": [k for k, v in t4.items() if v["k_over_72"] is None]}

# ---------------------------------------------------------------- 표 7 (OpenArm, 조건당 8회)
T7 = {  # ID kiwi, apple, orange, ID avg, OOD seen, OOD unseen, OOD avg, overall
    "No PT": [50.0, 50.0, 62.5, 54.2, 62.5, 25.0, 43.8, 49.0],
    "Human-HRDT": [50.0, 87.5, 87.5, 75.0, 75.0, 37.5, 56.3, 65.6],
    "Human-VITRA": [37.5, 37.5, 25.0, 33.3, 12.5, 12.5, 12.5, 22.9],
    "Ours (visual-only)": [50.0, 62.5, 75.0, 62.5, 37.5, 12.5, 25.0, 43.8],
    "Ours (no-overlay)": [50.0, 62.5, 75.0, 62.5, 75.0, 50.0, 62.5, 62.5],
    "Ours (HuRo PT)": [62.5, 75.0, 75.0, 70.8, 75.0, 75.0, 75.0, 72.9],
}
t7 = {}
for m, r in T7.items():
    idav = np.mean(r[0:3]); oodav = np.mean(r[4:6]); ov = (idav + oodav) / 2
    t7[m] = {"id_avg": [round(float(idav), 2), r[3]], "ood_avg": [round(float(oodav), 2), r[6]],
             "overall": [round(float(ov), 2), r[7]],
             "match": abs(idav - r[3]) <= 0.06 and abs(oodav - r[6]) <= 0.06 and abs(ov - r[7]) <= 0.06,
             "cells_multiple_of_12.5": all(abs(x / 12.5 - round(x / 12.5)) < 1e-9 for x in r[0:3] + r[4:6])}
res["table7_check"] = t7
# 풀링: ID 24회, OOD 16회
k = {m: (round(np.mean(r[0:3]) * 24 / 100), round(np.mean(r[4:6]) * 16 / 100)) for m, r in T7.items()}
res["table7_pooled_counts"] = {m: {"id_k_of_24": a, "ood_k_of_16": b, "id_wilson": wilson(a, 24), "ood_wilson": wilson(b, 16)} for m, (a, b) in k.items()}
res["table7_fisher_pooled"] = {
    "OOD HuRo(12/16) vs no-overlay(10/16)": round(fisher(12, 16, 10, 16), 3),
    "OOD HuRo(12/16) vs HRDT(9/16)": round(fisher(12, 16, 9, 16), 3),
    "OOD HuRo(12/16) vs visual-only(4/16)": round(fisher(12, 16, 4, 16), 4),
    "OOD HuRo(12/16) vs No PT(7/16)": round(fisher(12, 16, 7, 16), 3),
    "ID HuRo(17/24) vs HRDT(18/24)": round(fisher(17, 24, 18, 24), 3),
    "ID HuRo(17/24) vs No PT(13/24)": round(fisher(17, 24, 13, 24), 3),
}

# ---------------------------------------------------------------- 데이터셋 산술 (표 5, 6, 8, 9, 10, 본문)
T5 = {"EgoDex": (78.9, 730.8, 55.5), "EgoVerse": (37.7, 348.7, 26.5), "Ego4D": (14.8, 136.6, 10.4),
      "Ego10K": (8.5, 78.4, 6.0), "EPIC-Kitchens": (2.4, 22.3, 1.7)}
tot_f = sum(v[0] for v in T5.values()); tot_h = sum(v[1] for v in T5.values())
t5 = {"sum_frames_M": round(tot_f, 1), "reported_total_frames_M": 142.2, "sum_hours": round(tot_h, 1), "reported_total_hours": 1316.8}
for s, (f, h, r) in T5.items():
    t5[s] = {"hours_from_frames": round(f * 1e6 / 30 / 3600, 1), "hours_reported": h,
             "ratio_from_total_142.2": round(100 * f / 142.2, 1), "ratio_reported": r}
t5["frames_to_hours_142.2M"] = round(142.2e6 / 30 / 3600, 1)
t5["rounded_text_ratios_sum"] = 56 + 27 + 10 + 6 + 2
t5["ratio_table_sum"] = round(sum(v[2] for v in T5.values()), 1)
t5["non_egodex_share_pct"] = round(100 * (142.2 - 78.9) / 142.2, 1)
res["table5_source_composition"] = t5

res["episode_length_and_epochs"] = {
    "avg_frames_per_episode_142.2M/630K": round(142.2e6 / 630e3, 1),
    "avg_seconds_per_episode": round(142.2e6 / 630e3 / 30, 2),
    "code_default_max_window_frames_stage6": 210,
    "episodes_needed_if_all_windows_le_210": math.ceil(142.2e6 / 210),
    "pretrain_samples_80k_steps_x_2048": 80_000 * 2048,
    "epochs_over_10pct_14.22M": round(80_000 * 2048 / 14.22e6, 2),
    "epochs_over_50pct_71.1M": round(80_000 * 2048 / 71.1e6, 2),
    "epochs_over_100pct_142.2M": round(80_000 * 2048 / 142.2e6, 2),
    "epochs_over_egodex_only_78.9M": round(80_000 * 2048 / 78.9e6, 2),
    "epochs_over_0.7M_budget": round(80_000 * 2048 / 0.7e6, 1),
    "epochs_over_7M_budget": round(80_000 * 2048 / 7.0e6, 1),
    "fig6_budget_as_pct_of_full": {"0.7M": round(100 * 0.7 / 142.2, 2), "3.5M": round(100 * 3.5 / 142.2, 2), "7.0M": round(100 * 7.0 / 142.2, 2)},
    "hours_if_10pct_14.22M": round(14.22e6 / 108000, 1),
    "mixed_vs_egodex_only_frame_ratio_71.1_vs_78.9": round(71.1 / 78.9, 3),
    "epic_0.7M_ratio_2.4/0.7": round(2.4 / 0.7, 2),
}

# 표 8 (비용), 표 9 (유지율)과 표 5의 시간 대조
t8 = {"EPIC": {"parts": [89.8, 6.0, 47.3], "total_reported": 143.1, "x_reported": 9.5},
      "Ego4D": {"parts": [77.9, 3.6, 38.6], "total_reported": 120.1, "x_reported": 8.0}}
for s, d in t8.items():
    d["sum_parts"] = round(sum(d["parts"]), 1)
    d["x_recomputed_15min"] = round(d["sum_parts"] / 15, 2)
    d["action_share_pct"] = round(100 * d["parts"][1] / d["sum_parts"], 1)
t8["EPIC"]["target_specific_share_reported_pct"] = 10.6
t8["Ego4D"]["target_specific_share_reported_pct"] = 5.5
t8["EPIC"]["implied_overlay_min"] = round(0.106 * 143.1 - 6.0, 1)
t8["Ego4D"]["implied_overlay_min"] = round(0.055 * 120.1 - 3.6, 1)
t8["EPIC"]["implied_overlay_share_of_visual_pct"] = round(100 * t8["EPIC"]["implied_overlay_min"] / 47.3, 1)
t8["Ego4D"]["implied_overlay_share_of_visual_pct"] = round(100 * t8["Ego4D"]["implied_overlay_min"] / 38.6, 1)
res["table8_cost"] = t8

t9 = {"EPIC": {"src_h": 96.2, "sel": 0.173, "ret": 0.981, "table5_final_h": 22.3},
      "Ego4D": {"src_h": 1589.1, "sel": 0.069, "ret": 0.970, "table5_final_h": 136.6}}
for s, d in t9.items():
    d["derived_final_h"] = round(d["src_h"] * d["sel"] * d["ret"], 1)
    d["table5_over_derived"] = round(d["table5_final_h"] / d["derived_final_h"], 2)
    d["selected_h"] = round(d["src_h"] * d["sel"], 1)
t9["implied_egodex_input_h"] = round(730.8 / 0.889, 1)
t9["implied_egoverse_input_h"] = round(348.7 / 0.927, 1)
t9["implied_ego10k_input_h_from_3298.6"] = round(3298.6 - 96.2 - 1589.1 - 730.8 / 0.889 - 348.7 / 0.927, 1)
t9["implied_ego10k_retention_pct"] = round(100 * 78.4 / t9["implied_ego10k_input_h_from_3298.6"], 1)
res["table9_retention_vs_table5"] = t9

t10 = {"audit_counts_sum": 60 + 60 + 60 + 56 + 52, "reported_288": 288,
       "62.5pct_of_288": round(0.625 * 288, 2), "55.2pct_of_288": round(0.552 * 288, 2),
       "no_violation_k": 180, "no_selfcontact_k": 159,
       "wilson_joint_limit_ok": wilson(180, 288), "wilson_no_selfcontact": wilson(159, 288),
       "camera_scored_115_of_129_pct": round(100 * 115 / 129, 1),
       "joint_limit_violators_pct": 37.5, "selfcontact_trajectories_pct": 44.8}
res["table10_audit"] = t10

# 표 6: 증가 배율
res["table6_ratios"] = {"unique_VO_100_over_10": round(35358 / 11787, 2), "samples_100_over_10": round(710 / 71, 2),
                         "unique_verbs_100_over_10": round(959 / 521, 2), "oi_cov_gain_10_to_100": round(0.687 - 0.664, 3),
                         "mixed50_vs_egodex_oi_cov": round(0.678 - 0.616, 3)}

# ---------------------------------------------------------------- v1 ↔ v5 표 일치 (텍스트에서 읽음)
def read_rows(path, names):
    out = {}
    if not Path(path).exists():
        return None
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    for line in txt.splitlines():
        s = re.sub(r"\s+", " ", line).strip()
        for n in names:
            if s.startswith(n + " ") and re.search(r"\d+\.\d", s):
                nums = re.findall(r"–|\d+\.\d|\d+(?=\s|$)", s[len(n):])
                out.setdefault(n, []).append(nums)
    return out


names = ["π0.5", "GR00T N1.6", "Ours (0% PT)", "Ours (10% PT)", "Ours (50% PT)", "Ours (no-overlay)", "Ours (100% PT)"]
v1 = read_rows(ROOT / "paper" / "paper_v1.txt", names)
v5 = read_rows(ROOT / "paper" / "paper.txt", names)
if v1 and v5:
    res["v1_v5_table2_identical"] = {n: v1.get(n, [None])[-1] == v5.get(n, [None])[-1] for n in names}

def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


OUT.write_text(json.dumps(_clean(res), ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT)
print("표 2 평균 일치:", {m: v["match"] for m, v in avg_check.items()})
print("표 2 정수 불가 셀:", infeasible)
print("열별 최소 분모:", {c: v["min_denoms_consistent_with_all_values"] for c, v in col_denoms.items()})
print("표 4 불가(36):", res["table4_integer_feasibility"]["infeasible_at_36"])
print("DPP 불가:", res["diverse_pnp_integer_feasibility"]["infeasible"])
