"""VLA-FAIL (arXiv:2606.21386 v2) 보고 수치 재계산과 일관성 검사.

입력: analysis/tables_raw.json, tables_raw_v1.json (extract_tables.py 가 PDF에서 읽은 표 1·2·4-10; 평균과 95% 구간),
      본문 표 3, 11, 15-17은 이 파일 안에 옮겨 적었다(논문 paper/paper.txt 의 같은 이름 표).
출력: analysis/recompute_results.json 과 표준 출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy 필요)
검사 이름: V = 재계산, K = 표·본문 사이 일관성.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

A = Path(__file__).resolve().parent
RAW = json.load(open(A / "tables_raw.json"))
RAW1 = json.load(open(A / "tables_raw_v1.json"))
OUT = {}

METHODS = ["ACE", "Diff", "STAC", "ACC", "LLMD", "FAIL"]
VLAS = ["X-VLA", "pi0.5"]
RW_TASKS = ["Blocks", "Drawer", "Cups", "Kitchen", "Stack T", "Mixer"]
LP_SUITES = ["Object", "Spatial", "Goal", "10"]


def unpack(tb, tasks, ncols):
    """행 4개씩(X-VLA PR, X-VLA PDT, pi0.5 PR, pi0.5 PDT) -> {task: {vla: {metric: [cell, ...]}}}"""
    d = {}
    for i, t in enumerate(tasks):
        rows = tb[4 * i: 4 * i + 4]
        d[t] = {
            "X-VLA": {"PR": rows[0], "PDT": rows[1]},
            "pi0.5": {"PR": rows[2], "PDT": rows[3]},
        }
    return d


T1 = unpack(RAW["t1"], RW_TASKS, 6)
T2 = unpack(RAW["t2"], LP_SUITES + ["Average"], 6)
T1v1 = unpack(RAW1["t1"], RW_TASKS, 6)
T2v1 = unpack(RAW1["t2"], LP_SUITES, 6)
T4 = unpack(RAW["t4"], ["Kitchen", "Drawer", "Blocks", "Cups", "Stack T", "Mixer"], 2)
T5 = unpack(RAW["t5"], ["Blocks", "Drawer", "Cups", "Kitchen", "Stack T", "Mixer"], 2)
T6 = unpack(RAW["t6"], LP_SUITES, 2)
T7 = unpack(RAW["t7"], ["Blocks", "Drawer", "Kitchen", "Stack T", "Mixer", "Cups", "Average"], 3)
T8 = unpack(RAW["t8"], ["Drawer", "Kitchen", "Stack T", "Mixer", "Average"], 2)


def mean_of(c):
    return c[0]


# ---------------------------------------------------------------- 표 3, 15-17 (논문에서 옮김)
T3 = {"X-VLA": {"lat": (136, 139), "mem": (3.717, 3.721)}, "pi0.5": {"lat": (77, 78), "mem": (7.136, 7.165)}}
T3_PCT = {"X-VLA": (2.2, 0.1), "pi0.5": (1.3, 0.4)}
# 표 15 (pi0.5), 표 16 (X-VLA): 과제 -> (R, 겹침, 시연, 보정, ID, OOD, 성공률)
T15 = {
    "Blocks": (10, 10, 100, 20, 40, 35, 0.56), "Drawer": (10, 10, 100, 20, 80, 41, 0.72),
    "Cups": (5, 15, 100, 20, 181, 54, 0.69), "Kitchen": (10, 10, 100, 20, 41, 31, 0.47),
    "Stack T": (10, 10, 100, 20, 50, 51, 0.32), "Mixer": (10, 10, 100, 20, 50, 30, 0.64),
}
T16 = {
    "Blocks": (10, 10, 100, 20, 50, 34, 0.46), "Drawer": (10, 10, 100, 20, 49, 38, 0.71),
    "Cups": (10, 20, 100, 20, 65, 40, 0.53), "Kitchen": (10, 10, 100, 20, 41, 21, 0.55),
    "Stack T": (10, 10, 100, 20, 53, 51, 0.21), "Mixer": (10, 10, 100, 20, 51, 30, 0.34),
}
H = {"pi0.5": 20, "X-VLA": 30}
# 표 17: 스위트 -> (calib, rollouts, chunk(pi0.5, X-VLA), 성공률(pi0.5, X-VLA))
T17 = {
    "Object": (100, 2413, (5, 20), (0.88, 0.63)), "Spatial": (100, 2301, (5, 20), (0.90, 0.67)),
    "Goal": (100, 2463, (5, 20), (0.79, 0.66)), "10": (100, 2431, (5, 20), (0.77, 0.59)),
}
# 표 11 (Libero-Plus, 임계값 의존, 시드 1개): 스위트 -> VLA -> 지표 -> [ACE, Diff, STAC, ACC, LLMD, FAIL]
T11 = {
    "Object": {"X-VLA": {"WACC": [.831, .763, .813, .918, .994, .910], "F1": [.796, .693, .770, .901, .990, .885],
                         "Prec": [.965, .898, .952, .922, .979, .878], "Rec": [.677, .564, .646, .880, 1.000, .893]},
               "pi0.5": {"WACC": [.595, .751, .823, .771, .624, .832], "F1": [.309, .524, .650, .625, .353, .594],
                         "Prec": [.583, .472, .599, .679, .418, .481], "Rec": [.211, .589, .709, .579, .305, .775]}},
    "Spatial": {"X-VLA": {"WACC": [.795, .774, .807, .945, .964, .934], "F1": [.728, .704, .751, .903, .931, .884],
                          "Prec": [.769, .875, .858, .833, .872, .799], "Rec": [.691, .588, .668, .985, 1.000, .989]},
                "pi0.5": {"WACC": [.738, .780, .910, .936, .695, .926], "F1": [.577, .608, .705, .760, .470, .692],
                          "Prec": [.674, .610, .582, .642, .502, .548], "Rec": [.504, .605, .895, .933, .441, .941]}},
    "Goal": {"X-VLA": {"WACC": [.656, .756, .800, .973, .969, .965], "F1": [.507, .677, .741, .955, .943, .940],
                       "Prec": [.716, .896, .838, .923, .893, .894], "Rec": [.392, .543, .665, .988, 1.000, .990]},
             "pi0.5": {"WACC": [.701, .724, .811, .902, .579, .912], "F1": [.561, .605, .736, .841, .293, .835],
                       "Prec": [.820, .835, .835, .833, .575, .787], "Rec": [.426, .474, .658, .849, .197, .889]}},
    "10": {"X-VLA": {"WACC": [.672, .799, .656, .699, .990, .821], "F1": [.536, .755, .502, .583, .986, .786],
                     "Prec": [.845, .863, .846, .897, .973, .841], "Rec": [.392, .671, .357, .431, 1.000, .737]},
           "pi0.5": {"WACC": [.748, .739, .781, .864, .609, .873], "F1": [.632, .616, .686, .767, .374, .749],
                     "Prec": [.731, .703, .777, .717, .594, .648], "Rec": [.557, .548, .614, .824, .273, .889]}},
}


def r2(x, nd=2):
    return round(x + 1e-12, nd)


# ================================================================ V1: 표 1·2 상위 3위 집계와 본문 주장
def rank_rows(d, tasks, sel):
    """각 (과제, VLA, 지표) 행에서 방법별 순위(공동 순위, 작을수록 좋음)."""
    rows = []
    for t in tasks:
        for v in VLAS:
            for m in ["PR", "PDT"]:
                vals = [mean_of(c) for c in d[t][v][m]]
                good = [(-x if m == "PR" else x) for x in vals]
                ranks = [1 + sum(1 for g in good if g < gi - 1e-9) for gi in good]
                rows.append({"task": t, "vla": v, "metric": m, "vals": vals, "ranks": ranks})
    return rows


def top3_summary(rows):
    out = {}
    for mi, name in enumerate(METHODS):
        out[name] = {
            "top3_PR": sum(1 for r in rows if r["metric"] == "PR" and r["ranks"][mi] <= 3),
            "top3_PDT": sum(1 for r in rows if r["metric"] == "PDT" and r["ranks"][mi] <= 3),
            "best_PR": sum(1 for r in rows if r["metric"] == "PR" and r["ranks"][mi] == 1),
            "best_PDT": sum(1 for r in rows if r["metric"] == "PDT" and r["ranks"][mi] == 1),
        }
    return out


rows1 = rank_rows(T1, RW_TASKS, None)
rows2 = rank_rows(T2, LP_SUITES, None)
OUT["V1_top3_real"] = {"n_rows_per_metric": 12, "summary": top3_summary(rows1)}
OUT["V1_top3_libero"] = {"n_rows_per_metric": 8, "summary": top3_summary(rows2)}
# (과제, VLA) 단위에서 PR과 PDT 둘 다 상위 3위인 셀
def both_top3(rows, mi):
    cells = {}
    for r in rows:
        cells.setdefault((r["task"], r["vla"]), []).append(r["ranks"][mi] <= 3)
    return sum(all(v) for v in cells.values()), len(cells)


OUT["V1_both_top3_real"] = {n: both_top3(rows1, i) for i, n in enumerate(METHODS)}
OUT["V1_both_top3_libero"] = {n: both_top3(rows2, i) for i, n in enumerate(METHODS)}
# 어느 한 지표라도 상위 3위가 아닌 FAIL 셀
OUT["V1_fail_not_top3"] = [
    {"task": r["task"], "vla": r["vla"], "metric": r["metric"], "rank": r["ranks"][5], "FAIL": r["vals"][5],
     "best": (max if r["metric"] == "PR" else min)(r["vals"])}
    for r in rows1 + rows2 if r["ranks"][5] > 3
]

# FAIL이 ACC·LLMD 어느 한쪽보다 못한 정도(최악 칸), 둘 다보다 나은 칸
def fail_vs_single(rows):
    res = []
    for r in rows:
        v = r["vals"]
        a, l, f = v[3], v[4], v[5]
        if r["metric"] == "PR":
            gap = f - max(a, l)
            both_better = f > max(a, l)
        else:
            gap = min(a, l) - f
            both_better = f < min(a, l)
        res.append({"task": r["task"], "vla": r["vla"], "metric": r["metric"], "gap_vs_best_single": r2(gap),
                    "better_than_both": both_better})
    return res


fvs1 = fail_vs_single(rows1)
fvs2 = fail_vs_single(rows2)
OUT["V2_fail_vs_single_real"] = {"cells": fvs1, "n_better_than_both": sum(c["better_than_both"] for c in fvs1), "n": len(fvs1),
                                "worst_gap": min(c["gap_vs_best_single"] for c in fvs1)}
OUT["V2_fail_vs_single_libero"] = {"cells": fvs2, "n_better_than_both": sum(c["better_than_both"] for c in fvs2), "n": len(fvs2),
                                  "worst_gap": min(c["gap_vs_best_single"] for c in fvs2)}

# 본문 "FAIL이 가장 낮은 AUCPDT: Cups, Stack T, Mixer(특히 pi0.5), Libero-Plus의 X-VLA"
claim_cells = []
for t in ["Cups", "Stack T", "Mixer"]:
    for v in VLAS:
        vals = [mean_of(c) for c in T1[t][v]["PDT"]]
        claim_cells.append({"set": "real", "task": t, "vla": v, "FAIL": vals[5], "best": min(vals),
                            "FAIL_is_best": vals[5] <= min(vals) + 1e-9, "best_method": [METHODS[i] for i, x in enumerate(vals) if x == min(vals)]})
for t in LP_SUITES:
    vals = [mean_of(c) for c in T2[t]["X-VLA"]["PDT"]]
    claim_cells.append({"set": "libero", "task": t, "vla": "X-VLA", "FAIL": vals[5], "best": min(vals),
                        "FAIL_is_best": vals[5] <= min(vals) + 1e-9, "best_method": [METHODS[i] for i, x in enumerate(vals) if x == min(vals)]})
OUT["V3_fail_lowest_pdt_claim"] = {"cells": claim_cells, "n_best": sum(c["FAIL_is_best"] for c in claim_cells), "n": len(claim_cells)}

# ACC vs STAC (본문: 실제 과제 거의 전부에서 ACC가 STAC보다 좋다; Libero-Plus에서는 STAC의 AUCPR이 더 좋고 ACC의 AUCPDT가 더 낮다)
def acc_vs_stac(d, tasks):
    res = {"PR_acc_better": 0, "PDT_acc_better": 0, "n": 0, "detail": []}
    for t in tasks:
        for v in VLAS:
            pr = [mean_of(c) for c in d[t][v]["PR"]]
            pd = [mean_of(c) for c in d[t][v]["PDT"]]
            res["n"] += 1
            res["PR_acc_better"] += pr[3] > pr[2]
            res["PDT_acc_better"] += pd[3] < pd[2]
            res["detail"].append({"task": t, "vla": v, "PR_ACC": pr[3], "PR_STAC": pr[2], "PDT_ACC": pd[3], "PDT_STAC": pd[2]})
    return res


OUT["V4_acc_vs_stac_real"] = acc_vs_stac(T1, RW_TASKS)
OUT["V4_acc_vs_stac_libero"] = acc_vs_stac(T2, LP_SUITES)
OUT["V4_acc_vs_stac_libero_by_vla"] = {
    v: {"PR_STAC_better": sum(mean_of(T2[t][v]["PR"][2]) > mean_of(T2[t][v]["PR"][3]) for t in LP_SUITES),
        "PDT_ACC_better": sum(mean_of(T2[t][v]["PDT"][3]) < mean_of(T2[t][v]["PDT"][2]) for t in LP_SUITES)} for v in VLAS}

# ================================================================ V5: 구간(3시드) 검사
n_out, widths = 0, []
for t in RW_TASKS:
    for v in VLAS:
        for m in ["PR", "PDT"]:
            for mi, name in enumerate(METHODS):
                c = T1[t][v][m][mi]
                if c[1] is None:
                    continue
                if not (c[1] - 1e-9 <= c[0] <= c[2] + 1e-9):
                    n_out += 1
                widths.append((c[2] - c[1], t, v, m, name))
widths.sort(reverse=True)
OUT["V5_ci_T1"] = {"n_mean_outside_ci": n_out, "n_cells_with_ci": len(widths), "widest": [(r2(w[0]), *w[1:]) for w in widths[:6]],
                   "median_width": r2(float(np.median([w[0] for w in widths])), 3)}


def overlap_disjoint():
    """FAIL 구간이 다른 방법 구간과 겹치지 않는 칸 수(3 시드, 시드 변동만 반영)."""
    res = []
    for t in RW_TASKS:
        for v in VLAS:
            for m in ["PR", "PDT"]:
                cells = T1[t][v][m]
                f = cells[5]
                for mi, name in enumerate(METHODS[:5]):
                    c = cells[mi]
                    lo, hi = (c[1], c[2]) if c[1] is not None else (c[0], c[0])
                    if m == "PR":
                        better = f[1] > hi
                        worse = f[2] < lo
                    else:
                        better = f[2] < lo
                        worse = f[1] > hi
                    res.append({"task": t, "vla": v, "metric": m, "vs": name, "fail_clearly_better": bool(better), "fail_clearly_worse": bool(worse)})
    return res


ov = overlap_disjoint()
OUT["V5_fail_vs_baselines_disjoint_ci"] = {
    "n_pairs": len(ov), "clearly_better": sum(o["fail_clearly_better"] for o in ov), "clearly_worse": sum(o["fail_clearly_worse"] for o in ov),
    "vs_baselines_only(ACE,Diff,STAC)": {
        "pairs": sum(1 for o in ov if o["vs"] in ("ACE", "Diff", "STAC")),
        "better": sum(o["fail_clearly_better"] for o in ov if o["vs"] in ("ACE", "Diff", "STAC")),
        "worse": sum(o["fail_clearly_worse"] for o in ov if o["vs"] in ("ACE", "Diff", "STAC"))},
}
# Table 1 에서 시드 구간이 0 폭인 셀(= 같은 롤아웃 재사용으로 시드가 변동을 못 잡는 셀)의 비율
zero_w = sum(1 for w in widths if w[0] < 0.015)
OUT["V5_ci_T1"]["frac_width_le_0.01"] = r2(zero_w / len(widths), 3)

# 롤아웃 표본에 따른 구간 하한: 실제 시행 수에서 AUCPR 표준오차는 toy_metrics.py 가 계산(여기서는 시행 수만 정리)

# ================================================================ V6: 표 사이 일치
def cmp_tables(Ta, tasks_a, idx_a, Tb, tasks_b, idx_b, name):
    diff = []
    for t in tasks_a:
        for v in VLAS:
            for m in ["PR", "PDT"]:
                a = Ta[t][v][m][idx_a][0]
                b = Tb[t][v][m][idx_b][0]
                if abs(a - b) > 0.0051:
                    diff.append({"task": t, "vla": v, "metric": m, "A": a, "B": b})
    return {"name": name, "n_cells": len(tasks_a) * 4, "n_diff": len(diff), "diff": diff}


OUT["K1_T4_tokenwise_vs_T1_LLMD"] = cmp_tables(T4, RW_TASKS, 1, T1, RW_TASKS, 4, "표 4 token-wise vs 표 1 LLMD")
OUT["K1_T5_velnorm_vs_T1_ACC"] = cmp_tables(T5, RW_TASKS, 1, T1, RW_TASKS, 3, "표 5 vel norm vs 표 1 ACC")
OUT["K1_T6_velnorm_vs_T2_ACC"] = cmp_tables(T6, LP_SUITES, 1, T2, LP_SUITES, 3, "표 6 vel norm vs 표 2 ACC")
OUT["K1_T7_LLMD_vs_T1_LLMD"] = cmp_tables(T7, RW_TASKS, 2, T1, RW_TASKS, 4, "표 7 LLMD vs 표 1 LLMD")
OUT["K1_T8_FAIL_vs_T1_FAIL"] = cmp_tables(T8, ["Drawer", "Kitchen", "Stack T", "Mixer"], 1, T1, ["Drawer", "Kitchen", "Stack T", "Mixer"], 5, "표 8 FAIL vs 표 1 FAIL")

# 표 6 vs v1 표 2 (v1 값이 v2 표 6에 남아 있는지)
def t6_vs_v1():
    res = []
    for t in LP_SUITES:
        for v in VLAS:
            for m in ["PR", "PDT"]:
                v2t2 = T2[t][v][m][3][0]
                t6 = T6[t][v][m][1][0]
                v1t2 = T2v1[t][v][m][3][0]
                res.append({"task": t, "vla": v, "metric": m, "v2_T2_ACC": v2t2, "v2_T6_velnorm": t6, "v1_T2_ACC": v1t2})
    return res


OUT["K1b_T6_vs_v1"] = t6_vs_v1()
OUT["K1b_T6_matches_v1_count"] = sum(1 for r in OUT["K1b_T6_vs_v1"] if abs(r["v2_T6_velnorm"] - r["v1_T2_ACC"]) < 0.0051 and abs(r["v2_T6_velnorm"] - r["v2_T2_ACC"]) > 0.0051)
OUT["K1b_T6_matches_v2T2_count"] = sum(1 for r in OUT["K1b_T6_vs_v1"] if abs(r["v2_T6_velnorm"] - r["v2_T2_ACC"]) < 0.0051)

# 평균 행 검사
def avg_check(Td, tasks, idx_list, label, avg_key="Average"):
    res = []
    for v in VLAS:
        for m in ["PR", "PDT"]:
            for mi in idx_list:
                vals = [Td[t][v][m][mi][0] for t in tasks]
                rep = Td[avg_key][v][m][mi][0]
                res.append({"vla": v, "metric": m, "col": mi, "mean_of_tasks": r2(float(np.mean(vals)), 3), "reported": rep,
                            "diff": r2(rep - float(np.mean(vals)), 3)})
    big = [r for r in res if abs(r["diff"]) > 0.0051]
    return {"label": label, "n": len(res), "n_off_by_gt_0.005": len(big), "off": big}


OUT["K2_T2_average"] = avg_check(T2, LP_SUITES, range(6), "표 2 Average = 4개 스위트 평균")
OUT["K2_T7_average"] = avg_check(T7, RW_TASKS, range(3), "표 7 Average = 6개 과제 평균")
OUT["K2_T8_average"] = avg_check(T8, ["Drawer", "Kitchen", "Stack T", "Mixer"], range(2), "표 8 Average = 4개 과제 평균(표 8에 실린 과제)")
# 표 8 평균이 5개 과제인지 6개 과제인지: 표 1에서 RAPT 없는 열은 FAIL만 비교 가능
fail_avg6 = {v: {m: r2(float(np.mean([T1[t][v][m][5][0] for t in RW_TASKS])), 3) for m in ["PR", "PDT"]} for v in VLAS}
fail_avg4 = {v: {m: r2(float(np.mean([T1[t][v][m][5][0] for t in ["Drawer", "Kitchen", "Stack T", "Mixer"]])), 3) for m in ["PR", "PDT"]} for v in VLAS}
OUT["K2_T8_fail_avg_ref"] = {"over_all_6_tasks_from_T1": fail_avg6, "over_4_tasks_from_T1": fail_avg4,
                             "reported_T8_FAIL": {v: {m: T8["Average"][v][m][1][0] for m in ["PR", "PDT"]} for v in VLAS}}

# 표 4, 5, 6 본문 인용 값
OUT["K3_text_numbers"] = {
    "global_to_token_xvla_drawer_PDT(0.19->0.24)": (T4["Drawer"]["X-VLA"]["PDT"][0][0], T4["Drawer"]["X-VLA"]["PDT"][1][0]),
    "velnorm_xvla_spatial_PDT(0.38->0.28)": (T6["Spatial"]["X-VLA"]["PDT"][0][0], T6["Spatial"]["X-VLA"]["PDT"][1][0]),
    "LL-RND_pi05_AUCPR 0.73->0.49 / PDT 0.35->0.51 (표 7 pi0.5 평균 LL-RND)": (T7["Average"]["pi0.5"]["PR"][0][0], T7["Average"]["pi0.5"]["PDT"][0][0]),
}

# 표 4 전체 개선량(token-wise vs global)
def deltas(Td, tasks, ia, ib):
    res = {"PR": [], "PDT": []}
    for t in tasks:
        for v in VLAS:
            for m in ["PR", "PDT"]:
                res[m].append(r2(Td[t][v][m][ib][0] - Td[t][v][m][ia][0], 3))
    return {m: {"mean": r2(float(np.mean(x)), 3), "min": min(x), "max": max(x), "n_worse": sum(1 for z in x if (z < 0 if m == "PR" else z > 0)), "n": len(x)} for m, x in res.items()}


OUT["V6_T4_token_minus_global"] = deltas(T4, RW_TASKS, 0, 1)
OUT["V6_T5_velnorm_minus_none"] = deltas(T5, RW_TASKS, 0, 1)
OUT["V6_T6_velnorm_minus_none"] = deltas(T6, LP_SUITES, 0, 1)
# "표 5 속도 정규화는 실제 과제에서 모든 과제에 걸쳐 일관되게 개선"(부록 B) 대 표 5 Drawer X-VLA PR 0.90 -> 0.83
OUT["V6_T5_exceptions"] = [{"task": t, "vla": v, "m": m, "none": T5[t][v][m][0][0], "vel": T5[t][v][m][1][0]}
                           for t in RW_TASKS for v in VLAS for m in ["PR", "PDT"]
                           if (T5[t][v][m][1][0] < T5[t][v][m][0][0] - 1e-9 if m == "PR" else T5[t][v][m][1][0] > T5[t][v][m][0][0] + 1e-9)]
OUT["V6_T6_exceptions"] = [{"task": t, "vla": v, "m": m, "none": T6[t][v][m][0][0], "vel": T6[t][v][m][1][0]}
                           for t in LP_SUITES for v in VLAS for m in ["PR", "PDT"]
                           if (T6[t][v][m][1][0] < T6[t][v][m][0][0] - 1e-9 if m == "PR" else T6[t][v][m][1][0] > T6[t][v][m][0][0] + 1e-9)]

# ================================================================ V7: 표 3
V7 = {}
for v, d in T3.items():
    lat = d["lat"]
    mem = d["mem"]
    V7[v] = {"lat_pct": r2(100 * (lat[1] / lat[0] - 1), 2), "lat_reported": T3_PCT[v][0],
             "mem_pct": r2(100 * (mem[1] / mem[0] - 1), 2), "mem_reported": T3_PCT[v][1]}
OUT["V7_table3"] = V7
OUT["V7_note"] = "메모리 단위가 MB로 표기됨(X-VLA 0.9B, pi0.5 3.6B 매개변수 모델의 가중치만으로 수 GB). 표기 단위 확인 필요."
# 지연 시간: 1.3% = 1 ms (77 -> 78), 반올림 해상도
OUT["V7_latency_resolution"] = {"pi0.5_1ms_is_pct": r2(100 / 77, 2), "X-VLA_3ms_is_pct": r2(300 / 136, 2)}

# ================================================================ V8: 실패율·정수 가능성
def feas(rate, n, tol=0.005):
    """성공률 rate(소수 2자리)가 n회 중 정수 성공 수로 가능한지, 가능한 정수 목록."""
    return [k for k in range(n + 1) if abs(k / n - rate) <= tol + 1e-9]


V8 = {"rollouts": {}}
for vla, T in [("pi0.5", T15), ("X-VLA", T16)]:
    for task, (R, ov, demos, cal, idn, ood, sr) in T.items():
        n_test = idn + ood
        n_all = n_test + cal
        V8["rollouts"][f"{vla}/{task}"] = {
            "ID": idn, "OOD": ood, "calib": cal, "test_total": n_test, "with_calib": n_all, "success_rate": sr,
            "succ_counts_if_test_only": feas(sr, n_test), "succ_counts_if_with_calib": feas(sr, n_all),
            "overlap_vs_H_minus_R": [ov, H[vla] - R]}
tot = {vla: {t: V8["rollouts"][f"{vla}/{t}"]["test_total"] for t in RW_TASKS} for vla in VLAS}
V8["roughly_80_claim"] = {vla: {"min": min(tot[vla].values()), "max": max(tot[vla].values()), "by_task": tot[vla]} for vla in VLAS}
V8["calib_plus_test"] = {vla: {t: V8["rollouts"][f"{vla}/{t}"]["with_calib"] for t in RW_TASKS} for vla in VLAS}
# 평균 ID, OOD
V8["avg_ID_OOD"] = {vla: (r2(float(np.mean([V8["rollouts"][f"{vla}/{t}"]["ID"] for t in RW_TASKS])), 1),
                          r2(float(np.mean([V8["rollouts"][f"{vla}/{t}"]["OOD"] for t in RW_TASKS])), 1)) for vla in VLAS}

# STAC 가 아무 신호도 못 낼 때 AUCPDT = 1 - (실패율)(부록 A: PDT 가 구간 전체에서 1). X-VLA 의 STAC PDT 로 실패율을 되읽어 표 16과 비교
infer = {}
for t in RW_TASKS:
    pdt_stac = T1[t]["X-VLA"]["PDT"][2][0]
    pr_stac = T1[t]["X-VLA"]["PR"][2][0]
    sr = T16[t][6]
    n_test = T16[t][4] + T16[t][5]
    n_all = n_test + T16[t][3]
    infer[t] = {"STAC_PDT": pdt_stac, "pi_from_PDT": r2(1 - pdt_stac), "pi_from_T16_success_rate": r2(1 - sr),
                "STAC_AUCPR": pr_stac, "AUCPR_over_pi": r2(pr_stac / (1 - sr), 2),
                "pi_if_calib_included": [r2(1 - k / n_all, 3) for k in feas(sr, n_all)][:3]}
V8["stac_xvla_infer_failure_rate"] = infer
# pi0.5 에서도 같은 값 비교(STAC PDT 가 1에 가깝지 않으므로 하한만)
V8["stac_pi05"] = {t: {"STAC_PDT": T1[t]["pi0.5"]["PDT"][2][0], "one_minus_success": r2(1 - T15[t][6])} for t in RW_TASKS}
OUT["V8"] = V8

# AUCPDT 가 가질 수 있는 최댓값 1 - pi 로 정규화한 값(표 1)
norm = {}
for t in RW_TASKS:
    for v, T in [("X-VLA", T16), ("pi0.5", T15)]:
        pi = 1 - T[t][6]
        norm[f"{v}/{t}"] = {"pi": r2(pi, 2),
                            "AUCPDT_over_(1-pi)": {METHODS[i]: r2(T1[t][v]["PDT"][i][0] / (1 - pi), 2) for i in range(6)}}
OUT["V8_normalized_pdt_real"] = norm
normL = {}
for s in LP_SUITES:
    for vi, v in enumerate(["pi0.5", "X-VLA"]):
        pi = 1 - T17[s][3][vi]
        normL[f"{v}/{s}"] = {"pi": r2(pi, 2),
                             "AUCPDT_over_(1-pi)": {METHODS[i]: r2(T2[s][v]["PDT"][i][0] / (1 - pi), 2) for i in range(6)}}
OUT["V8_normalized_pdt_libero"] = normL

# 표 17 정수 가능성과 "큰 롤아웃 수" 서술 (4 스위트 합)
V8L = {"total_rollouts": sum(T17[s][1] for s in LP_SUITES)}
V8L["avg_success_pi05"] = r2(float(np.average([T17[s][3][0] for s in LP_SUITES], weights=[T17[s][1] for s in LP_SUITES])), 3)
V8L["avg_success_xvla"] = r2(float(np.average([T17[s][3][1] for s in LP_SUITES], weights=[T17[s][1] for s in LP_SUITES])), 3)
V8L["n_failures"] = {s: {"pi0.5": int(round(T17[s][1] * (1 - T17[s][3][0]))), "X-VLA": int(round(T17[s][1] * (1 - T17[s][3][1])))} for s in LP_SUITES}
OUT["V8_libero_counts"] = V8L

# ================================================================ V9: 표 9·10 (임계값 의존) 일관성
METRICS = ["Rec", "Prec", "Acc", "WACC", "F1", "TNR"]


def unpack_td(tb, tasks):
    d = {}
    for i, t in enumerate(tasks):
        d[t] = {}
        for vi, v in enumerate(["X-VLA", "pi0.5"]):
            block = tb[12 * i + 6 * vi: 12 * i + 6 * vi + 6]
            d[t][v] = {m: block[mi] for mi, m in enumerate(METRICS)}
    return d


TD = {}
TD.update(unpack_td(RAW["t9"], ["Blocks", "Drawer", "Cups"]))
TD.update(unpack_td(RAW["t10"], ["Kitchen", "Stack T", "Mixer"]))

V9 = {"wacc_check": [], "f1_check": [], "ci_gt1": [], "class_counts": {}, "acc_check": []}
for t in RW_TASKS:
    for v in VLAS:
        td = TD[t][v]
        for mi, name in enumerate(METHODS):
            rec, prec, acc, wacc, f1, tnr = [td[m][mi][0] for m in METRICS]
            # WACC = (Rec + TNR)/2 ?
            if abs((rec + tnr) / 2 - wacc) > 0.0101:
                V9["wacc_check"].append({"task": t, "vla": v, "method": name, "Rec": rec, "TNR": tnr, "WACC": wacc, "calc": r2((rec + tnr) / 2, 3)})
            # F1 from mean Prec,Rec (평균 값이라 시드 평균과 차이가 날 수 있음: ACC 열(결정적)만 엄격)
            if prec + rec > 0 and name == "ACC":
                f1c = 2 * prec * rec / (prec + rec)
                if abs(f1c - f1) > 0.0101:
                    V9["f1_check"].append({"task": t, "vla": v, "method": name, "Prec": prec, "Rec": rec, "F1": f1, "calc": r2(f1c, 3)})
            for m in METRICS:
                c = td[m][mi]
                if c[2] is not None and c[2] > 1.0001:
                    V9["ci_gt1"].append({"task": t, "vla": v, "method": name, "metric": m, "mean": c[0], "lo": c[1], "hi": c[2]})

# 클래스 수 복원: STAC 처럼 Rec=0, TNR=1 인 행에서 Acc = N/(N+P)  -> 시험 집합의 성공 비율
for t in RW_TASKS:
    for v, T in [("X-VLA", T16), ("pi0.5", T15)]:
        R, ov, demos, cal, idn, ood, sr = T[t]
        n_test = idn + ood
        td = TD[t][v]
        out = {"n_test_rollouts": n_test, "calib": cal}
        # 가설 H_test: 시험 집합 = ID+OOD 에서 보정 20개(성공)를 뺀 것
        for hyp, base in [("H_a_total=ID+OOD,calib in it", n_test), ("H_b_total=calib+ID+OOD", n_test + cal)]:
            succ_opts = feas(sr, base)
            out[hyp] = {"succ_total_options": succ_opts[:3]}
        # STAC 행에서 직접 복원
        rec, prec, acc, wacc, f1, tnr = [td[m][2][0] for m in METRICS]
        if rec == 0 and tnr > 0:
            share = acc / tnr  # N/(N+P)
            out["STAC_success_share_of_test"] = r2(share, 3)
        out["ACC_row"] = {m: td[m][3][0] for m in METRICS}
        V9["class_counts"][f"{v}/{t}"] = out

# 정수 복원: ACC 열(결정적)에서 (P, N)을 찾는다. 가설: N = (성공 수) - 20, P = n_test - 성공 수 (n_test = ID+OOD).
def find_counts(td, n_test, cal, tol=0.0051):
    rec, prec, acc, wacc, f1, tnr = [td[m][3][0] for m in METRICS]
    sols = []
    for N in range(1, n_test):
        P = n_test - cal - N  # 시험 집합 크기 = ID+OOD-calib (보정 샘플은 성공이라 시험에서 제외)
        if P <= 0:
            continue
        tn_opts = [k for k in range(N + 1) if abs(k / N - tnr) <= tol]
        tp_opts = [k for k in range(P + 1) if abs(k / P - rec) <= tol]
        for tn in tn_opts:
            for tp in tp_opts:
                fp = N - tn
                p_ok = (tp + fp == 0 and prec == 0) or (tp + fp > 0 and abs(tp / (tp + fp) - prec) <= tol)
                a_ok = abs((tp + tn) / (P + N) - acc) <= tol
                if p_ok and a_ok:
                    sols.append((N, P, tp, tn))
    return sols


V9["acc_integer_solutions"] = {}
for t in RW_TASKS:
    for v, T in [("X-VLA", T16), ("pi0.5", T15)]:
        R, ov, demos, cal, idn, ood, sr = T[t]
        n_test = idn + ood
        sols = find_counts(TD[t][v], n_test, cal)
        expect_succ = feas(sr, n_test)
        # 가설 H_a: 성공 합계 S in expect_succ, N=S-cal
        match = [s for s in sols if (s[0] + cal) in expect_succ]
        V9["acc_integer_solutions"][f"{v}/{t}"] = {"n_test": n_test, "n_solutions": len(sols), "matching_success_rate": match[:3],
                                                    "success_total_options": expect_succ[:3],
                                                    "solutions_sample": sols[:3]}
# 보정 20개가 성공 롤아웃에서 온다는 가정 하에 최소 성공 수 점검
V9["cal_lt_success_check"] = {f"{v}/{t}": (feas(T[t][6], T[t][4] + T[t][5])[0] if feas(T[t][6], T[t][4] + T[t][5]) else None, T[t][3])
                              for v, T in [("X-VLA", T16), ("pi0.5", T15)] for t in RW_TASKS}

# 공형 보장 대비 TNR: n=20 보정, alpha=0.05 -> 기대 거짓 경보율 <= 1/21 (단일 검출기). FAIL 은 두 임계값의 OR
tnr_rows = []
for t in RW_TASKS:
    for v in VLAS:
        td = TD[t][v]
        tnr_rows.append({"task": t, "vla": v, "ACC": td["TNR"][3][0], "LLMD": td["TNR"][4][0], "FAIL": td["TNR"][5][0],
                         "ACE": td["TNR"][0][0], "Diff": td["TNR"][1][0], "STAC": td["TNR"][2][0]})
mean_tnr = {k: r2(float(np.mean([r[k] for r in tnr_rows])), 3) for k in ["ACE", "Diff", "STAC", "ACC", "LLMD", "FAIL"]}
V9["tnr_mean"] = mean_tnr
V9["tnr_target_single"] = r2(1 - 1 / 21, 3)
V9["tnr_target_or"] = r2((1 - 1 / 21) ** 2, 3)  # 독립이면 OR 의 TNR >= (1-1/21)^2
V9["tnr_below_0.95_count"] = {k: sum(1 for r in tnr_rows if r[k] < 0.95) for k in ["ACE", "Diff", "STAC", "ACC", "LLMD", "FAIL"]}
V9["tnr_rows"] = tnr_rows

# 복원된 (N, P, TP, TN)의 Wilson 95% 구간(ACC 열, 결정적)
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (round(c - h, 3), round(c + h, 3))


W = {}
for key, x in V9["acc_integer_solutions"].items():
    if x["matching_success_rate"]:
        N, P, tp, tn = x["matching_success_rate"][0]
        W[key] = {"N": N, "P": P, "TP": tp, "TN": tn, "Rec": round(tp / P, 3), "Rec_wilson": wilson(tp, P),
                  "TNR": round(tn / N, 3), "TNR_wilson": wilson(tn, N)}
V9["wilson_reconstructed"] = W
OUT["V9_threshold_dependent_real"] = V9

# 재현율·정밀도 평균으로 본 FAIL vs ACC vs LLMD (표 9, 10)
V9b = {}
for v in VLAS:
    for m in ["Rec", "Prec", "F1", "WACC"]:
        V9b[f"{v}/{m}"] = {METHODS[i]: r2(float(np.mean([TD[t][v][m][i][0] for t in RW_TASKS])), 3) for i in range(6)}
OUT["V9b_mean_threshold_dependent"] = V9b

# ================================================================ V10: 표 11 일관성(Libero-Plus)
V10 = {"f1_mismatch": [], "implied": {}, "n_cells": 0}
for s in LP_SUITES:
    cal, nroll, chunk, sr = T17[s]
    for vi, v in enumerate(["pi0.5", "X-VLA"]):
        d = T11[s][v]
        succ = nroll * sr[vi]
        P = nroll - succ  # 실패 롤아웃 수(반올림 전)
        N = succ - cal  # 보정 100개를 성공에서 뺀 시험 성공 수
        pi = P / (P + N)
        rows = []
        for mi, name in enumerate(METHODS):
            wacc, f1, prec, rec = d["WACC"][mi], d["F1"][mi], d["Prec"][mi], d["Rec"][mi]
            V10["n_cells"] += 1
            f1c = 2 * prec * rec / (prec + rec)
            if abs(f1c - f1) > 0.0051:
                V10["f1_mismatch"].append({"suite": s, "vla": v, "method": name, "F1": f1, "from_PR": r2(f1c, 3)})
            tnr = 2 * wacc - rec  # WACC = (Rec+TNR)/2 가정
            prec_c = rec * pi / (rec * pi + (1 - tnr) * (1 - pi)) if (rec * pi + (1 - tnr) * (1 - pi)) > 0 else float("nan")
            rows.append({"method": name, "TNR_implied": r2(tnr, 3), "Prec_reported": prec, "Prec_from_prevalence": r2(prec_c, 3),
                         "diff": r2(prec - prec_c, 3)})
        V10["implied"][f"{v}/{s}"] = {"P": r2(P, 0), "N": r2(N, 0), "pi_test": r2(pi, 3), "rows": rows}
V10["tnr_gt1"] = [(k, r["method"], r["TNR_implied"]) for k, x in V10["implied"].items() for r in x["rows"] if r["TNR_implied"] > 1.0 + 1e-9]
V10["prec_diff_abs_gt_0.03"] = [(k, r["method"], r["Prec_reported"], r["Prec_from_prevalence"]) for k, x in V10["implied"].items() for r in x["rows"]
                                if abs(r["diff"]) > 0.03]
V10["prec_diff_abs_max"] = max(abs(r["diff"]) for x in V10["implied"].values() for r in x["rows"])
OUT["V10_table11"] = V10
# 표 11 최고 F1/WACC 방법 vs 표 2 최고 AUCPR 방법
best11 = {}
for s in LP_SUITES:
    for v in ["X-VLA", "pi0.5"]:
        d = T11[s][v]
        best11[f"{v}/{s}"] = {"best_WACC": METHODS[int(np.argmax(d["WACC"]))], "best_F1": METHODS[int(np.argmax(d["F1"]))],
                              "best_AUCPR_T2": METHODS[int(np.argmax([c[0] for c in T2[s][v]["PR"]]))],
                              "best_AUCPDT_T2": METHODS[int(np.argmin([c[0] for c in T2[s][v]["PDT"]]))]}
OUT["V10_best_methods"] = best11
# LLMD 가 Libero-Plus 10 스위트 X-VLA 에서 임계 평가 최상, AUCPR 표 2 에서는?
OUT["V10_llmd_xvla_10"] = {"T2_PR": [c[0] for c in T2["10"]["X-VLA"]["PR"]], "T2_PDT": [c[0] for c in T2["10"]["X-VLA"]["PDT"]],
                           "T11_WACC": T11["10"]["X-VLA"]["WACC"], "T11_Rec": T11["10"]["X-VLA"]["Rec"]}

# ================================================================ V11: v1 대 v2
def v1v2(T_v2, T_v1, tasks, label):
    diffs = []
    big = []
    for t in tasks:
        for v in VLAS:
            for m in ["PR", "PDT"]:
                for mi, name in enumerate(METHODS):
                    a = T_v1[t][v][m][mi][0]
                    b = T_v2[t][v][m][mi][0]
                    if abs(a - b) > 0.0051:
                        diffs.append({"task": t, "vla": v, "metric": m, "method": name, "v1": a, "v2": b})
    return {"label": label, "n_cells": len(tasks) * 4 * 6, "n_changed": len(diffs), "changed": diffs}


OUT["V11_v1_vs_v2_T1"] = v1v2(T1, T1v1, RW_TASKS, "표 1")
OUT["V11_v1_vs_v2_T2"] = v1v2(T2, T2v1, LP_SUITES, "표 2")
t2d = OUT["V11_v1_vs_v2_T2"]["changed"]
OUT["V11_T2_change_stats"] = {
    "max_abs": r2(max(abs(c["v1"] - c["v2"]) for c in t2d), 3),
    "median_abs": r2(float(np.median([abs(c["v1"] - c["v2"]) for c in t2d])), 3),
    "changed_by_method": {n: sum(1 for c in t2d if c["method"] == n) for n in METHODS},
    "winner_change_cells": [],
}
for t in LP_SUITES:
    for v in VLAS:
        for m in ["PR", "PDT"]:
            a = [T2v1[t][v][m][i][0] for i in range(6)]
            b = [T2[t][v][m][i][0] for i in range(6)]
            f = (np.argmax if m == "PR" else np.argmin)
            if f(a) != f(b):
                OUT["V11_T2_change_stats"]["winner_change_cells"].append({"task": t, "vla": v, "metric": m, "v1_best": METHODS[int(f(a))], "v2_best": METHODS[int(f(b))]})

# v1 -> v2 표 2 FAIL 이 상위 3위인 셀 수 비교
rows2v1 = rank_rows(T2v1, LP_SUITES, None)
OUT["V11_top3_libero_v1"] = top3_summary(rows2v1)
OUT["V11_top3_libero_v2"] = top3_summary(rows2)

# ================================================================ 저장
json.dump(OUT, open(A / "recompute_results.json", "w"), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
for k, v in OUT.items():
    s = json.dumps(v, ensure_ascii=False, default=str)
    print(k, ":", s[:600])
