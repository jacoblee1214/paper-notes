"""VISTA (arXiv:2610.03333) 보고 수치 재계산과 검정.

논문 표 1·2·6–10의 값을 옮겨 적고 평균, 정수 격자, 표끼리 일치, 본문 인용 차이, 검정을 다시 계산한다.
시행 수는 논문이 밝힌 값(시뮬 과제당 40회, 실로봇 과제당 10회)을 쓴다. 학습 시드는 1개라서 구간과 p값은
롤아웃 표집 변동만 반영한다. 결과는 recompute_results.json에 쓴다.

실행: python3 analysis/recompute.py   (numpy, scipy; 수 초)
"""
import json
import math
from decimal import ROUND_HALF_UP, Decimal
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
N_SIM = 40      # 시뮬 롤아웃 수 (부록 A.7, 표 1 캡션)
N_REAL = 10     # 실로봇 시행 수 (§4.3, 표 2)

TASKS = ["Grasp Classify", "Insert HDMI", "Insert Hole", "Insert Tube",
         "Lift Bottle", "Lift Can", "Pull Out Key", "Put Bottle in Shelf"]

# 표 7 (= 표 1): 50 데모, 40 롤아웃. 값은 %. 열 순서는 TASKS와 같다.
T7 = {
    "ManiFeel":       [100.0, 12.5, 32.5, 45.0, 55.0, 10.0, 95.0, 40.0],
    "VITAL":          [100.0,  7.5, 27.5, 55.0, 20.0, 17.5, 65.0, 57.5],
    "UniVTAC":        [ 60.0,  5.0, 17.5, 32.5,  0.0,  0.0, 37.5, 17.5],
    "EquiDiff + Tac": [ 95.0, 10.0, 42.5, 92.5, 57.5, 37.5, 97.5, 87.5],
    "VISTA":          [100.0, 40.0, 55.0, 95.0, 80.0, 92.5, 97.5, 87.5],
}
T7_MEAN = {"ManiFeel": 48.8, "VITAL": 43.8, "UniVTAC": 21.3, "EquiDiff + Tac": 65.0, "VISTA": 80.9}

# 표 8: ISP 대 VISTA
T8 = {
    "ISP":   [87.5, 32.5, 55.0, 92.5, 15.0, 92.5, 92.5, 77.5],
    "VISTA": [100.0, 40.0, 55.0, 95.0, 80.0, 92.5, 97.5, 87.5],
}
T8_MEAN = {"ISP": 68.1, "VISTA": 80.9}

# 표 9: 융합 절제
T9 = {
    "ISP":             [87.5, 32.5, 55.0, 92.5, 15.0, 92.5, 92.5, 77.5],
    "VISTA-global-l0": [100.0, 17.5, 52.5, 90.0, 77.5, 77.5, 57.5, 82.5],
    "Early Fusion":    [100.0, 42.5, 37.5, 90.0, 75.0, 77.5, 82.5, 77.5],
    "ISP+Concat-Tac":  [100.0, 37.5, 67.5, 85.0, 67.5, 92.5, 77.5, 65.0],
    "VISTA":           [100.0, 40.0, 55.0, 95.0, 80.0, 92.5, 97.5, 87.5],
}
T9_MEAN = {"ISP": 68.1, "VISTA-global-l0": 69.4, "Early Fusion": 72.8, "ISP+Concat-Tac": 74.1, "VISTA": 80.9}

# 표 10: 데모 수 (50, 100). 키는 (방법, 데모 수)
T10 = {
    ("ManiFeel", 50):       [100.0, 12.5, 32.5, 45.0, 55.0, 10.0, 95.0, 40.0],
    ("ManiFeel", 100):      [100.0, 25.0, 60.0, 92.5, 70.0, 77.5, 95.0, 67.5],
    ("VITAL", 50):          [100.0, 7.5, 27.5, 55.0, 20.0, 17.5, 65.0, 57.5],
    ("VITAL", 100):         [100.0, 10.0, 20.0, 72.5, 50.0, 45.0, 72.5, 62.5],
    ("UniVTAC", 50):        [60.0, 5.0, 17.5, 32.5, 0.0, 0.0, 37.5, 17.5],
    ("UniVTAC", 100):       [60.0, 0.0, 15.0, 42.5, 0.0, 10.0, 65.0, 60.0],
    ("EquiDiff + Tac", 50): [95.0, 10.0, 42.5, 92.5, 57.5, 37.5, 97.5, 87.5],
    ("EquiDiff + Tac", 100): [97.5, 27.5, 72.5, 95.0, 75.0, 77.5, 100.0, 60.0],
    ("VISTA", 50):          [100.0, 40.0, 55.0, 95.0, 80.0, 92.5, 97.5, 87.5],
    ("VISTA", 100):         [100.0, 42.5, 80.0, 92.5, 80.0, 100.0, 97.5, 85.0],
}
T10_MEAN = {("ManiFeel", 50): 48.8, ("ManiFeel", 100): 73.4, ("VITAL", 50): 43.8, ("VITAL", 100): 54.1,
            ("UniVTAC", 50): 21.3, ("UniVTAC", 100): 31.6, ("EquiDiff + Tac", 50): 65.0,
            ("EquiDiff + Tac", 100): 75.6, ("VISTA", 50): 80.9, ("VISTA", 100): 84.7}

# 표 2: 실로봇 (성공 횟수 / 10)
T2 = {"EquiDiff + Tac": {"Erase": 1, "Insert USB": 3, "Insert pencil": 0},
      "VISTA":          {"Erase": 9, "Insert USB": 5, "Insert pencil": 8}}
T2_MEAN_REPORTED = {"EquiDiff + Tac": 1.33, "VISTA": 7.33}

# 표 6: 추론 지연 (ms), RTX 3090, DDIM-8
T6 = {("ISP", "C8"): 61.25, ("ISP", "I60"): 65.66, ("VISTA", "C8"): 63.74,
      ("VISTA", "I60"): 71.54, ("EquiDiff + Tac", "C8"): 79.88}

# UniVTAC 원 논문(arXiv:2602.10093) 표 I의 과제별 성공 횟수/100 (같은 8과제). 보고 프로토콜이 다르다(100회, 50 데모).
UNIVTAC_PAPER = {
    "ACT + UniVTAC": [99, 28, 24, 56, 71, 29, 46, 31],
    "VITaL":         [100, 6, 25, 34, 72, 8, 47, 32],
}
UNIVTAC_PAPER_MEAN = {"ACT + UniVTAC": 48.0, "VITaL": 40.5}


# ---------------------------------------------------------------- 통계 도구
def r1(x):
    """소수점 한 자리 반올림 (0.05는 올림). 파이썬 round는 반올림 대신 짝수로 맞춘다."""
    return float(Decimal(repr(round(x, 9))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def counts(row, n=N_SIM):
    k = [v * n / 100 for v in row]
    ok = all(abs(x - round(x)) < 1e-9 for x in k)
    return [int(round(x)) for x in k], ok


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * ps[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def ac_var(k, n):
    """Agresti-Coull 보정 비율의 분산 (0%·100% 칸에서 분산이 0이 되는 것을 막는다)."""
    pt = (k + 2) / (n + 4)
    return pt * (1 - pt) / (n + 4)


def mean_diff(k1, k2, n=N_SIM):
    """8과제 단순 평균의 차이(%p), Wald 95% 구간, 양측 p. 과제 평균은 과제마다 같은 가중치를 준 값이다."""
    t = len(k1)
    d = (np.mean(k1) - np.mean(k2)) / n * 100
    var = sum(ac_var(a, n) + ac_var(b, n) for a, b in zip(k1, k2)) / t**2
    se = math.sqrt(var) * 100
    z = d / se
    p = float(2 * stats.norm.sf(abs(z)))
    return d, d - 1.959964 * se, d + 1.959964 * se, p, se


def cmh(k1, k2, n=N_SIM):
    """과제를 층으로 둔 Cochran-Mantel-Haenszel 검정 (연속성 보정 포함), 양측 p."""
    a = np.array(k1, float)
    n1 = n2 = n
    m1 = a + np.array(k2, float)
    N = n1 + n2
    e = n1 * m1 / N
    v = n1 * n2 * m1 * (N - m1) / (N**2 * (N - 1))
    num = abs(a.sum() - e.sum()) - 0.5
    chi = max(num, 0) ** 2 / v.sum()
    return float(1 - stats.chi2.cdf(chi, 1))


def per_task_table(ka, kb, n=N_SIM):
    ps = [fisher(a, n, b, n) for a, b in zip(ka, kb)]
    adj = holm(ps)
    rows = []
    for t, a, b, p, q in zip(TASKS, ka, kb, ps, adj):
        rows.append({"task": t, "a": a, "b": b, "diff_pp": 100 * (a - b) / n, "p": p, "p_holm": q})
    return rows


def compare(name_a, ka, name_b, kb, n=N_SIM):
    d, lo, hi, p, se = mean_diff(ka, kb, n)
    return {"a": name_a, "b": name_b, "mean_a": float(np.mean(ka) / n * 100), "mean_b": float(np.mean(kb) / n * 100),
            "diff_pp": d, "ci95": [lo, hi], "p_wald": p, "p_cmh": cmh(ka, kb, n), "se_pp": se,
            "per_task": per_task_table(ka, kb, n)}


def mde(n=N_SIM, tasks=8, p=0.5, power_z=0.841621):
    """과제 평균 차이의 최소 검출 차이(%p): 양측 5%, 검정력 80%, 모든 과제 p 근처."""
    se = math.sqrt(tasks * 2 * p * (1 - p) / n) / tasks
    return (1.959964 + power_z) * se * 100


def main():
    R = {"assumptions": {"n_sim": N_SIM, "n_real": N_REAL,
                         "note": "학습 시드 1개. 구간과 p는 롤아웃 표집 변동만 반영. 방법 사이 시드가 같아도 쌍별 자료가 없어 독립 표본으로 계산."}}

    # ---- V1: 정수 격자 (모든 칸이 k/40인가)
    grid = {}
    allk = {}
    for name, rows in [("T7", T7), ("T8", T8), ("T9", T9), ("T10", {f"{k[0]}@{k[1]}": v for k, v in T10.items()})]:
        for m, row in rows.items():
            kk, ok = counts(row)
            grid[f"{name}:{m}"] = ok
            allk[f"{name}:{m}"] = kk
    R["v1_integer_grid_all_ok"] = all(grid.values())
    R["v1_counts"] = allk

    # ---- V2: 평균 재계산
    means = {}
    for label, rows, rep in [("T7", T7, T7_MEAN), ("T8", T8, T8_MEAN), ("T9", T9, T9_MEAN)]:
        for m, row in rows.items():
            mean = float(np.mean(row))
            means[f"{label}:{m}"] = {"recomputed": mean, "reported": rep[m], "match_1dp": abs(r1(mean) - rep[m]) < 1e-9}
    for (m, d), row in T10.items():
        mean = float(np.mean(row))
        means[f"T10:{m}@{d}"] = {"recomputed": mean, "reported": T10_MEAN[(m, d)],
                                 "match_1dp": abs(r1(mean) - T10_MEAN[(m, d)]) < 1e-9}
    R["v2_means"] = means
    R["v2_all_match"] = all(v["match_1dp"] for v in means.values())
    # 보고 평균 소수점 처리: 21.25 -> 21.3 (올림), 48.75 -> 48.8
    R["v2_half_values"] = {k: v["recomputed"] for k, v in means.items() if abs(v["recomputed"] * 10 - round(v["recomputed"] * 10)) > 1e-9}

    # ---- V3: 표 사이 일치
    c = {}
    c["T7_VISTA == T8_VISTA == T9_VISTA == T10_VISTA@50"] = (T7["VISTA"] == T8["VISTA"] == T9["VISTA"] == T10[("VISTA", 50)])
    c["T8_ISP == T9_ISP"] = T8["ISP"] == T9["ISP"]
    c["T7 baselines == T10@50 baselines"] = all(T7[m] == T10[(m, 50)] for m in T7)
    R["v3_cross_table"] = c

    # ---- V4: 본문 인용 차이
    cl = {}
    m7 = {m: float(np.mean(v)) for m, v in T7.items()}
    cl["32.1 (VISTA - ManiFeel)"] = r1(m7["VISTA"]) - r1(m7["ManiFeel"])
    cl["37.1 (VISTA - VITAL)"] = r1(m7["VISTA"]) - r1(m7["VITAL"])
    cl["59.6 (VISTA - UniVTAC)"] = r1(m7["VISTA"]) - r1(m7["UniVTAC"])
    cl["15.9 (VISTA - EquiDiff+Tac)"] = r1(m7["VISTA"]) - r1(m7["EquiDiff + Tac"])
    cl["12.8 (VISTA - ISP)"] = r1(m7["VISTA"]) - r1(float(np.mean(T8["ISP"])))
    cl["6.8 (VISTA - ISP+Concat-Tac)"] = r1(m7["VISTA"]) - r1(float(np.mean(T9["ISP+Concat-Tac"])))
    cl["1.3 (global-l0 - ISP)"] = r1(float(np.mean(T9["VISTA-global-l0"]))) - r1(float(np.mean(T9["ISP"])))
    cl["6.0 (Concat - ISP)"] = r1(float(np.mean(T9["ISP+Concat-Tac"]))) - r1(float(np.mean(T9["ISP"])))
    cl["5.3 (VISTA@50 - EquiDiff+Tac@100)"] = round(float(np.mean(T10[("VISTA", 50)])), 1) - round(float(np.mean(T10[("EquiDiff + Tac", 100)])), 1)
    cl["3.8 (VISTA@100 - VISTA@50)"] = round(float(np.mean(T10[("VISTA", 100)])), 1) - round(float(np.mean(T10[("VISTA", 50)])), 1)
    gains = {t: T8["VISTA"][i] - T8["ISP"][i] for i, t in enumerate(TASKS)}
    cl["gains HDMI/hole/tube/key (7.5/0/2.5/5.0)"] = [gains["Insert HDMI"], gains["Insert Hole"], gains["Insert Tube"], gains["Pull Out Key"]]
    cl["Lift Bottle 15.0 -> 80.0, Put Bottle 77.5 -> 87.5"] = [T8["ISP"][4], T8["VISTA"][4], T8["ISP"][7], T8["VISTA"][7]]
    cl["ISP Grasp Classify 87.5 -> 100"] = [T8["ISP"][0], T8["VISTA"][0]]
    R["v4_text_claims"] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in cl.items()}

    # ---- V5: 실로봇 산술
    real = {}
    for m, d in T2.items():
        tot = sum(d.values())
        real[m] = {"total": tot, "mean_per_task_of_10": tot / 3, "reported": T2_MEAN_REPORTED[m],
                   "match": abs(round(tot / 3, 2) - T2_MEAN_REPORTED[m]) < 1e-9,
                   "pct": 100 * tot / 30}
    R["v5_real_arith"] = real
    rt = []
    keys = list(T2["VISTA"].keys())
    ps = []
    for k in keys:
        a, b = T2["VISTA"][k], T2["EquiDiff + Tac"][k]
        ps.append(fisher(a, N_REAL, b, N_REAL))
    adj = holm(ps)
    for k, p, q in zip(keys, ps, adj):
        a, b = T2["VISTA"][k], T2["EquiDiff + Tac"][k]
        rt.append({"task": k, "vista": a, "equi": b, "p_fisher": p, "p_holm": q,
                   "wilson_vista": wilson(a, N_REAL), "wilson_equi": wilson(b, N_REAL)})
    R["v6_real_per_task"] = rt
    tv, te = sum(T2["VISTA"].values()), sum(T2["EquiDiff + Tac"].values())
    R["v6_real_pooled"] = {"vista": tv, "equi": te, "n": 3 * N_REAL, "p_fisher": fisher(tv, 30, te, 30),
                           "wilson_vista": wilson(tv, 30), "wilson_equi": wilson(te, 30),
                           "cmh_p": cmh_real()}
    # 실로봇 최소 검출: 과제당 10회에서 p=0.5 주변 80% 검정력
    R["v6_real_mde_pp_single_task"] = (1.959964 + 0.841621) * math.sqrt(2 * 0.25 / N_REAL) * 100

    # ---- V7: 시뮬 메인 비교 (표 1)
    k7 = {m: counts(r)[0] for m, r in T7.items()}
    R["v7_main"] = {m: compare("VISTA", k7["VISTA"], m, k7[m]) for m in T7 if m != "VISTA"}
    R["v7_wilson_cells"] = {m: [wilson(k, N_SIM) for k in k7[m]] for m in T7}

    # ---- V8: 절제 (표 8, 9)
    k9 = {m: counts(r)[0] for m, r in T9.items()}
    R["v8_ablation_vs_ISP"] = {m: compare(m, k9[m], "ISP", k9["ISP"]) for m in T9 if m != "ISP"}
    R["v8_vista_vs_alt"] = {m: compare("VISTA", k9["VISTA"], m, k9[m]) for m in T9 if m not in ("VISTA",)}
    R["v8_alt_range_vs_ISP_pp"] = [R["v8_ablation_vs_ISP"][m]["diff_pp"] for m in ("VISTA-global-l0", "Early Fusion", "ISP+Concat-Tac")]
    # 과제별 최대 차이를 만드는 과제 수
    R["v8_lift_bottle_share"] = {
        "vista_minus_isp_total_diff_pp_sum": float(sum(np.array(T8["VISTA"]) - np.array(T8["ISP"]))),
        "lift_bottle_diff_pp": T8["VISTA"][4] - T8["ISP"][4],
        "mean_diff_without_lift_bottle_pp": float(np.mean(np.delete(np.array(T8["VISTA"]) - np.array(T8["ISP"]), 4))),
    }
    # 같은 계산을 알려진 과제 평균 차이에서 Lift Bottle을 뺀 7과제로
    for m in ("VISTA-global-l0", "Early Fusion", "ISP+Concat-Tac", "VISTA"):
        d = np.delete(np.array(T9[m]) - np.array(T9["ISP"]), 4)
        R["v8_lift_bottle_share"][f"mean_diff_vs_ISP_without_lift_bottle:{m}"] = float(np.mean(d))
    # VISTA와 ISP+Concat-Tac를 과제별로
    R["v8_vista_vs_concat_per_task"] = per_task_table(k9["VISTA"], k9["ISP+Concat-Tac"])


    # ---- V7b: ISP(시각 전용)와 EquiDiff+Tac, 그리고 Lift Bottle을 뺀 7과제 비교
    R["v7b_isp_vs_equi"] = compare("ISP", k9["ISP"], "EquiDiff + Tac", k7["EquiDiff + Tac"])
    def drop(k, i=4):
        return [a for j, a in enumerate(k) if j != i]
    def sub(a, b):
        d, lo, hi, p, se = mean_diff(drop(a), drop(b))
        return {"diff_pp": d, "ci95": [lo, hi], "p_wald": p}
    R["v8b_without_lift_bottle"] = {
        "VISTA_vs_ISP": sub(k9["VISTA"], k9["ISP"]),
        "VISTA_vs_Concat": sub(k9["VISTA"], k9["ISP+Concat-Tac"]),
        "VISTA_vs_Early": sub(k9["VISTA"], k9["Early Fusion"]),
        "VISTA_vs_globall0": sub(k9["VISTA"], k9["VISTA-global-l0"]),
        "Concat_vs_ISP": sub(k9["ISP+Concat-Tac"], k9["ISP"]),
        "Early_vs_ISP": sub(k9["Early Fusion"], k9["ISP"]),
        "globall0_vs_ISP": sub(k9["VISTA-global-l0"], k9["ISP"]),
        "VISTA_vs_Equi": sub(k7["VISTA"], k7["EquiDiff + Tac"]),
    }

    # ---- V9: 데모 수 (표 10)
    k10 = {key: counts(r)[0] for key, r in T10.items()}
    R["v9_vista50_vs_equi100"] = compare("VISTA@50", k10[("VISTA", 50)], "EquiDiff+Tac@100", k10[("EquiDiff + Tac", 100)])
    R["v9_vista100_vs_vista50"] = compare("VISTA@100", k10[("VISTA", 100)], "VISTA@50", k10[("VISTA", 50)])
    gap = {}
    for m in ["ManiFeel", "VITAL", "UniVTAC", "EquiDiff + Tac"]:
        d50 = mean_diff(k10[("VISTA", 50)], k10[(m, 50)])
        d100 = mean_diff(k10[("VISTA", 100)], k10[(m, 100)])
        change = d100[0] - d50[0]
        se = math.sqrt(d50[4] ** 2 + d100[4] ** 2)
        z = change / se
        gap[m] = {"gap50": d50[0], "gap100": d100[0], "change_pp": change, "ci95": [change - 1.959964 * se, change + 1.959964 * se],
                  "p": float(2 * stats.norm.sf(abs(z))), "gain_own_50_to_100_pp": mean_diff(k10[(m, 100)], k10[(m, 50)])[0],
                  "gain_own_p": mean_diff(k10[(m, 100)], k10[(m, 50)])[3]}
    R["v9_gap_shrink"] = gap
    R["v9_vista100_vs_each_baseline100"] = {m: compare("VISTA@100", k10[("VISTA", 100)], m + "@100", k10[(m, 100)])["diff_pp"]
                                            for m in ["ManiFeel", "VITAL", "UniVTAC", "EquiDiff + Tac"]}
    decl = {}
    for m in ["ManiFeel", "VITAL", "UniVTAC", "EquiDiff + Tac", "VISTA"]:
        d = [(TASKS[i], T10[(m, 50)][i], T10[(m, 100)][i]) for i in range(8) if T10[(m, 100)][i] < T10[(m, 50)][i]]
        decl[m] = d
    R["v9_declines_with_more_demos"] = decl
    R["v9_equi_put_bottle_drop_fisher_p"] = fisher(k10[("EquiDiff + Tac", 50)][7], 40, k10[("EquiDiff + Tac", 100)][7], 40)

    # ---- V10: 검출력
    R["v10_mde_task_mean_pp"] = mde()
    R["v10_mde_single_task_pp"] = (1.959964 + 0.841621) * math.sqrt(2 * 0.25 / N_SIM) * 100
    R["v10_ci_half_width_task_mean_pp"] = 1.959964 * math.sqrt(8 * 0.2 / N_SIM) / 8 * 100  # 한 방법의 평균 (p(1-p)=0.2 가정)

    # ---- V11: 지연 표
    R["v11_latency"] = {
        "VISTA C8->I60": T6[("VISTA", "I60")] - T6[("VISTA", "C8")],
        "ISP C8->I60": T6[("ISP", "I60")] - T6[("ISP", "C8")],
        "VISTA - ISP at C8": T6[("VISTA", "C8")] - T6[("ISP", "C8")],
        "VISTA - ISP at I60": T6[("VISTA", "I60")] - T6[("ISP", "I60")],
        "text_two_stream_one_stream_ms": 70.18, "text_two_stream_two_stream_ms": 73.13,
        "text_two_stream_delta": 73.13 - 70.18,
        "table6_vista_i60_vs_text_one_stream": T6[("VISTA", "I60")] - 70.18,
        "replan_window_ms": {"insert (6 actions at 10 Hz)": 600, "erase (8 actions at 10 Hz)": 800},
        "max_latency_ms": max(T6.values()),
    }

    # ---- V12: 텐서 차원 산술 (부록 A.1)
    lmax = 6
    R["v12_dims"] = {"(lmax+1)^2": (lmax + 1) ** 2, "sum_(2l+1)^2": sum((2 * l + 1) ** 2 for l in range(lmax + 1)),
                     "paper_values": [49, 455]}

    # ---- V13: UniVTAC 원 논문과 비교 (프로토콜이 다름)
    cross = {"note": "UniVTAC 원 논문은 과제당 100회 롤아웃, 보고 프로토콜이 다르다(손목 카메라 시야, 행동 인터페이스 등이 이 논문에서 바뀜). 차이만 적고 검정하지 않는다.",
             "rows": []}
    for t, a, b in zip(TASKS, UNIVTAC_PAPER["ACT + UniVTAC"], T7["UniVTAC"]):
        cross["rows"].append({"task": t, "univtac_paper_ACT_UniVTAC_pct": a, "this_paper_UniVTAC_pct": b})
    cross["mean_univtac_paper"] = float(np.mean(UNIVTAC_PAPER["ACT + UniVTAC"]))
    cross["mean_this_paper"] = float(np.mean(T7["UniVTAC"]))
    cross["vital_rows"] = [{"task": t, "univtac_paper_VITaL_pct": a, "this_paper_VITAL_pct": b}
                           for t, a, b in zip(TASKS, UNIVTAC_PAPER["VITaL"], T7["VITAL"])]
    cross["mean_vital_univtac_paper"] = float(np.mean(UNIVTAC_PAPER["VITaL"]))
    cross["mean_vital_this_paper"] = float(np.mean(T7["VITAL"]))
    R["v13_cross_paper"] = cross

    # ---- V14: 과제별로 VISTA가 ISP보다 높은 과제 수, 같은 과제 수
    R["v14_task_wins_vs_ISP"] = {"higher": int(sum(a > b for a, b in zip(T8["VISTA"], T8["ISP"]))),
                                 "equal": int(sum(a == b for a, b in zip(T8["VISTA"], T8["ISP"]))),
                                 "lower": int(sum(a < b for a, b in zip(T8["VISTA"], T8["ISP"])))}

    OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print("wrote", OUT)
    summarize(R)


def cmh_real():
    a = np.array(list(T2["VISTA"].values()), float)
    b = np.array(list(T2["EquiDiff + Tac"].values()), float)
    n1 = n2 = N_REAL
    m1 = a + b
    N = n1 + n2
    e = n1 * m1 / N
    v = n1 * n2 * m1 * (N - m1) / (N**2 * (N - 1))
    num = abs(a.sum() - e.sum()) - 0.5
    return float(1 - stats.chi2.cdf(max(num, 0) ** 2 / v.sum(), 1))


def summarize(R):
    print("V1 정수 격자:", R["v1_integer_grid_all_ok"])
    print("V2 평균 일치:", R["v2_all_match"])
    print("V3:", R["v3_cross_table"])
    print("V4:", R["v4_text_claims"])
    for m, d in R["v7_main"].items():
        print(f"V7 VISTA-{m}: {d['diff_pp']:.1f} [{d['ci95'][0]:.1f}, {d['ci95'][1]:.1f}] p_wald={d['p_wald']:.2g} p_cmh={d['p_cmh']:.2g}")
    for m, d in R["v8_ablation_vs_ISP"].items():
        print(f"V8 {m}-ISP: {d['diff_pp']:.1f} [{d['ci95'][0]:.1f}, {d['ci95'][1]:.1f}] p={d['p_wald']:.2g}")
    for m, d in R["v8_vista_vs_alt"].items():
        print(f"V8 VISTA-{m}: {d['diff_pp']:.1f} [{d['ci95'][0]:.1f}, {d['ci95'][1]:.1f}] p={d['p_wald']:.2g}")
    print("V9", R["v9_vista50_vs_equi100"]["diff_pp"], R["v9_vista50_vs_equi100"]["ci95"], R["v9_vista50_vs_equi100"]["p_wald"])
    print("V9b", R["v9_vista100_vs_vista50"]["diff_pp"], R["v9_vista100_vs_vista50"]["ci95"], R["v9_vista100_vs_vista50"]["p_wald"])
    for m, g in R["v9_gap_shrink"].items():
        print("V9 gap", m, g)
    print("V9 declines", R["v9_declines_with_more_demos"])
    print("V10", R["v10_mde_task_mean_pp"], R["v10_mde_single_task_pp"], R["v10_ci_half_width_task_mean_pp"])
    print("V5/6", R["v5_real_arith"], R["v6_real_pooled"])
    for r in R["v6_real_per_task"]:
        print(r)
    print("V11", R["v11_latency"])
    print("V12", R["v12_dims"])
    print("V13", R["v13_cross_paper"]["mean_univtac_paper"], R["v13_cross_paper"]["mean_this_paper"], R["v13_cross_paper"]["mean_vital_univtac_paper"], R["v13_cross_paper"]["mean_vital_this_paper"])
    print("V14", R["v14_task_wins_vs_ISP"], R["v8_lift_bottle_share"])


if __name__ == "__main__":
    main()
