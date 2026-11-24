"""CompVLA (arXiv:2609.23614 v2) 표 1-4 재계산. numpy/scipy만 사용, 수 초.

가정: 이진 성공 과제(Pivot, Cup Dispense, PnP)는 run당 20회, 3 run = 60회 (표 1 캡션).
Erase는 글자 단위 성공률이라 시행 수 격자로 되돌리지 않는다.
실행: python3 analysis/recompute.py -> analysis/recompute_results.json
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")

# 표 1 (v2): 평균, SD (VLA 3 run), DP/ACT는 1 run
T1 = {
    "pi0":      {"Erase": (17.8, 7.1), "Pivot": (31.7, 10.4), "Cup": (45.0, 0.0), "Avg": (31.5, 5.2), "PnP": (80.0, 10.0)},
    "ForceVLA": {"Erase": (68.1, 4.1), "Pivot": (45.0, 15.0), "Cup": (50.0, 8.7), "Avg": (54.4, 4.3), "PnP": (45.0, 15.0)},
    "TA-VLA":   {"Erase": (52.8, 9.7), "Pivot": (26.7, 10.4), "Cup": (0.0, 0.0), "Avg": (26.5, 2.0), "PnP": (83.3, 12.6)},
    "CompVLA":  {"Erase": (92.9, 3.7), "Pivot": (73.3, 10.4), "Cup": (35.0, 10.0), "Avg": (67.1, 7.2), "PnP": (76.7, 5.8)},
}
T1_SINGLE = {"DP": {"Erase": 0.0, "Pivot": 0.0, "Cup": 10.0, "Avg": 3.3, "PnP": 50.0},
             "ACT": {"Erase": 0.0, "Pivot": 15.0, "Cup": 30.0, "Avg": 15.0, "PnP": 0.0}}
T1_V1_PI0 = {"Cup": (38.0, 12.1), "Avg": (29.2, 9.2)}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def triples(mean, sd, n=20):
    """run당 n회, 3 run 이진 성공 개수 (k1<=k2<=k3) 중 평균과 SD가 반올림 일치하는 것."""
    res = {"ddof1": [], "ddof0": []}
    for ks in itertools.combinations_with_replacement(range(n + 1), 3):
        v = np.array(ks) * 100.0 / n
        if round(v.mean(), 1) != round(mean, 1):
            continue
        for key, dd in (("ddof1", 1), ("ddof0", 0)):
            if round(v.std(ddof=dd), 1) == round(sd, 1):
                res[key].append(ks)
    return res


def mean_feasible(mean, n=20, runs=3):
    tot = round(mean * n * runs / 100.0, 6)
    return abs(tot - round(tot)) < 0.05 + 1e-9 and abs(round(tot) * 100.0 / (n * runs) - mean) < 0.051


def welch(m1, s1, m2, s2, n=3):
    se = math.sqrt(s1 ** 2 / n + s2 ** 2 / n)
    t = (m1 - m2) / se
    df = (s1 ** 2 / n + s2 ** 2 / n) ** 2 / ((s1 ** 2 / n) ** 2 / (n - 1) + (s2 ** 2 / n) ** 2 / (n - 1))
    return t, df, 2 * stats.t.sf(abs(t), df)


def main():
    R = {}
    # V1 격자 가능성 (이진 과제 평균이 1/60 격자에 있는가)
    grid = {}
    for m, rows in T1.items():
        for task in ("Pivot", "Cup", "PnP"):
            mean, sd = rows[task]
            grid[f"{m}/{task}"] = {"mean": mean, "grid_ok": mean_feasible(mean),
                                   "k_of_60": round(mean * 0.6)}
    grid["pi0_v1/Cup"] = {"mean": 38.0, "grid_ok": mean_feasible(38.0), "k_of_60": 38.0 * 0.6}
    R["V1_grid"] = grid

    # V2 SD 규약 (표본 ddof=1 또는 모집단 ddof=0)로 맞는 (k1,k2,k3)가 있는가
    sdres = {}
    for m, rows in T1.items():
        for task in ("Pivot", "Cup", "PnP"):
            mean, sd = rows[task]
            t = triples(mean, sd)
            sdres[f"{m}/{task}"] = {"n_ddof1": len(t["ddof1"]), "n_ddof0": len(t["ddof0"]),
                                   "ex_ddof1": t["ddof1"][:3], "ex_ddof0": t["ddof0"][:3]}
    t = triples(*T1_V1_PI0["Cup"])
    sdres["pi0_v1/Cup"] = {"n_ddof1": len(t["ddof1"]), "n_ddof0": len(t["ddof0"])}
    R["V2_sd_convention"] = sdres

    # V3 평균 열 = 세 과제 평균
    avg = {}
    for m, rows in T1.items():
        a = (rows["Erase"][0] + rows["Pivot"][0] + rows["Cup"][0]) / 3
        ub = (rows["Erase"][1] + rows["Pivot"][1] + rows["Cup"][1]) / 3
        ind = math.sqrt(rows["Erase"][1] ** 2 + rows["Pivot"][1] ** 2 + rows["Cup"][1] ** 2) / 3
        avg[m] = {"recomputed": round(a, 2), "reported": rows["Avg"][0], "sd_reported": rows["Avg"][1],
                  "sd_upper_bound_perfect_corr": round(ub, 2), "sd_if_independent": round(ind, 2)}
    for m, r in T1_SINGLE.items():
        avg[m] = {"recomputed": round((r["Erase"] + r["Pivot"] + r["Cup"]) / 3, 2), "reported": r["Avg"]}
    a1 = (17.8 + 31.7 + 38.0) / 3
    avg["pi0_v1"] = {"recomputed": round(a1, 2), "reported": 29.2}
    R["V3_average"] = avg
    R["V3_gap"] = {"CompVLA_minus_ForceVLA_avg_reported": round(67.1 - 54.4, 1),
                   "from_task_means": round((92.9 + 73.3 + 35) / 3 - (68.1 + 45 + 50) / 3, 2)}

    # V4 Wilson (60회 풀링)과 Fisher, 과제별 CompVLA 대 기준선
    ci, fisher = {}, {}
    for m, rows in T1.items():
        for task in ("Pivot", "Cup", "PnP"):
            k = round(rows[task][0] * 0.6)
            lo, hi = wilson(k, 60)
            ci[f"{m}/{task}"] = {"k": k, "n": 60, "pct": round(100 * k / 60, 1), "wilson95": [round(100 * lo, 1), round(100 * hi, 1)]}
    for task in ("Pivot", "Cup", "PnP"):
        kc = round(T1["CompVLA"][task][0] * 0.6)
        for m in ("pi0", "ForceVLA", "TA-VLA"):
            kb = round(T1[m][task][0] * 0.6)
            _, p = stats.fisher_exact([[kc, 60 - kc], [kb, 60 - kb]])
            fisher[f"{task}: CompVLA vs {m}"] = {"k": [kc, kb], "p": float(f"{p:.3g}")}
    R["V4_wilson_pooled60"] = ci
    R["V4_fisher_pooled60"] = fisher

    # V5 run 수준 Welch (n=3, 보고된 SD를 표본 SD로 해석)
    wel = {}
    for task in ("Erase", "Pivot", "Cup", "Avg", "PnP"):
        for m in ("pi0", "ForceVLA", "TA-VLA"):
            a, b = T1["CompVLA"][task], T1[m][task]
            if a[1] == 0 and b[1] == 0:
                continue
            t_, df, p = welch(a[0], a[1], b[0], b[1])
            wel[f"{task}: CompVLA vs {m}"] = {"diff": round(a[0] - b[0], 1), "t": round(t_, 2), "df": round(df, 1), "p": float(f"{p:.3g}")}
    R["V5_welch_n3"] = wel

    # V6 표 4 (2 run x 20 = 40회)
    t4 = {"pi0": {"reach": 21, "push": 27}, "CompVLA": {"reach": 18, "push": 37}}
    out4 = {}
    for stage in ("reach", "push"):
        _, p = stats.fisher_exact([[t4["CompVLA"][stage], 40 - t4["CompVLA"][stage]], [t4["pi0"][stage], 40 - t4["pi0"][stage]]])
        out4[stage] = {"CompVLA": f"{t4['CompVLA'][stage]}/40", "pi0": f"{t4['pi0'][stage]}/40", "fisher_p": float(f"{p:.3g}"),
                       "wilson_CompVLA": [round(100 * x, 1) for x in wilson(t4["CompVLA"][stage], 40)],
                       "wilson_pi0": [round(100 * x, 1) for x in wilson(t4["pi0"][stage], 40)]}
    # 분해: 성공 = 도달 AND 밀기. 밀기 단계 평균 = r*a + (1-r)*b, 전체 성공 = r*a (가정)
    dec = {}
    for m, (r, push, full) in {"pi0": (0.525, 0.675, 0.45), "CompVLA": (0.45, 0.925, 0.35)}.items():
        a = full / r
        b = (push - full) / (1 - r)
        dec[m] = {"reach": r, "push_avg": push, "full_task_table1": full, "implied_push_given_reach": round(a, 3),
                  "implied_push_given_miss": round(b, 3), "min_full_if_b_le_1": round(push - (1 - r), 3)}
    out4["decomposition_assumption_success_eq_reach_and_push"] = dec
    R["V6_table4"] = out4

    # V7 표 2/3 (10 episodes)
    _, p3 = stats.fisher_exact([[7, 3], [4, 6]])
    R["V7_table3_pivot_10eps"] = {"CompVLA": "7/10", "fixedK": "4/10", "fisher_p": float(f"{p3:.3g}"),
                                  "wilson_70": [round(100 * x, 1) for x in wilson(7, 10)],
                                  "wilson_40": [round(100 * x, 1) for x in wilson(4, 10)]}
    # 80% 검정력: 70% 대 40% 에 필요한 에피소드 수 (정규 근사, 양측 0.05)
    p1, p2 = 0.7, 0.4
    za, zb = 1.959964, 0.841621
    n = ((za * math.sqrt(2 * (p1 + p2) / 2 * (1 - (p1 + p2) / 2)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p1 - p2) ** 2
    R["V7_table3_pivot_10eps"]["n_per_arm_80pct_power"] = math.ceil(n)
    # 표 3 Pivot 대 표 1 Pivot (73.3 = 44/60)
    R["V8_table1_vs_ablation"] = {"Table1_CompVLA_Pivot_pct": 73.3, "Table3_CompVLA_Pivot_pct": 70.0,
                                  "Table1_CompVLA_Erase_pct": 92.9, "Table2_CompVLA_Erase_pct": 96.0,
                                  "Table1_pi0_Erase_pct": 17.8, "Table2_wo_K_virtual_Erase_pct": 15.3}
    # 시행 수 격자: DP/ACT, 표 4
    R["V9_single_run_grid"] = {f"{m}/{t}": (v * 20 / 100).is_integer() for m, r in T1_SINGLE.items() for t, v in r.items() if t in ("Erase", "Pivot", "Cup", "PnP")}

    # V10 평균 열 SD: 과제별 run 개수(V2의 유일 해)를 짝짓는 방식에 따라 가능한 범위. Erase는 (m-s, m, m+s)로 근사
    pair = {}
    for m, rows in T1.items():
        tr = {}
        for task in ("Pivot", "Cup"):
            tt = triples(*rows[task])["ddof1"][0]
            tr[task] = np.array(tt) * 5.0
        e = np.array([rows["Erase"][0] - rows["Erase"][1], rows["Erase"][0], rows["Erase"][0] + rows["Erase"][1]])
        vals = []
        for pp in itertools.permutations(range(3)):
            for cc in itertools.permutations(range(3)):
                a = (e + tr["Pivot"][list(pp)] + tr["Cup"][list(cc)]) / 3
                vals.append(a.std(ddof=1))
        pair[m] = {"reported": rows["Avg"][1], "min": round(min(vals), 2), "max": round(max(vals), 2)}
    R["V10_avg_sd_pairing_range"] = pair
    OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False))
    print(json.dumps({k: R[k] for k in ("V1_grid", "V3_gap", "V4_fisher_pooled60", "V6_table4", "V7_table3_pivot_10eps")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
