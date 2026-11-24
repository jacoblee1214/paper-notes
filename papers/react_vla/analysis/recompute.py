"""REACT (arXiv:2610.12007) 표·본문 수치 재계산과 검정. 학습·실행 없이 논문 표 값만 쓴다.

실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)  → analysis/recompute_results.json

가정(문장 안에도 밝힌다):
- 시뮬레이션: 과제·조건마다 100회 (부록 C.6).
- 실세계 일반 과제: 과제당 30회 (§4.1, 부록 C.6). 표 값은 정수 %라서 k = round(p * 0.3)으로 복원한다.
- 특수 과제(표 3): 시도 수를 쓰지 않았으므로 30회로 가정한다.
- 시드는 하나이므로 모든 검정은 평가 롤아웃 표집 잡음만 본다(논문 한계 절도 같은 말).
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"

METHODS = ["H50E50", "H50E10", "H10E10", "AsyncRTC", "REACT_noDD", "REACT"]

# 표 1 (RoboTwin 2.0): 과제 -> 방법별 (Clean, Rand)
T1 = {
    "beat_block_hammer": [(17, 1), (8, 0), (0, 0), (13, 4), (15, 2), (14, 4)],
    "click_bell": [(61, 6), (71, 18), (35, 0), (64, 8), (89, 32), (61, 22)],
    "move_playingcard_away": [(59, 26), (71, 13), (7, 1), (64, 11), (71, 25), (66, 22)],
    "pick_diverse_bottles": [(20, 3), (9, 0), (0, 1), (10, 6), (27, 9), (33, 6)],
    "press_stapler": [(58, 23), (22, 3), (9, 4), (37, 13), (67, 42), (63, 39)],
    "rotate_qrcode": [(30, 3), (21, 3), (4, 1), (12, 8), (52, 6), (46, 11)],
    "turn_switch": [(17, 18), (34, 24), (21, 10), (33, 24), (27, 30), (31, 35)],
}
T1_AVG = {  # 표에 적힌 평균 (Clean, Rand)
    "H50E50": (37.43, 11.43), "H50E10": (33.71, 8.71), "H10E10": (10.86, 2.43),
    "AsyncRTC": (33.29, 10.57), "REACT_noDD": (49.71, 20.86), "REACT": (44.86, 19.86),
}

# 표 2 (실세계): 로봇, 과제 -> 방법별 %
T2 = {
    ("ARX", "Bowl Stacking by Size"): [73, 66, 70, 36, 86, 76],
    ("ARX", "Bottle Cap Unscrewing"): [36, 0, 0, 0, 56, 47],
    ("ARX", "Cable Hanging"): [56, 66, 56, 3, 63, 80],
    ("Franka", "Bowl Stacking by Size"): [60, 50, 0, 47, 66, 63],
    ("Franka", "Bottle Cap Unscrewing"): [57, 0, 0, 3, 63, 70],
    ("Franka", "Keyring Hanging"): [70, 33, 0, 17, 60, 53],
}
T2_AVG = {  # 표에 적힌 로봇별 평균과 전체 평균
    "ARX": [55.0, 44.0, 42.0, 13.0, 68.3, 67.7],
    "Franka": [62.3, 27.8, 0.0, 22.3, 63.0, 62.0],
    "Overall": [58.7, 35.9, 21.0, 17.7, 65.7, 64.8],
}

# 표 3
T3_POUR = [3, 33, 0, 0, 57, 63]
T3_REACTION = {  # 평균, 표준편차 (ms)
    "H50E50": (1505, 502), "H50E10": (768, 113), "H10E10": (772, 119), "AsyncRTC": (747, 123),
    "REACT_noDD": (754, 127), "REACT": (734, 94), "Human": (705, 73),
}

# 표 8 (평균 jerk norm). None = 성공 에피소드 없음 ("–")
T8 = {
    "Sim": {
        "beat_block_hammer": [463.4, 996.7, None, 1755.0, 664.0, 775.9],
        "click_bell": [502.4, 1087.2, 1341.8, 1946.4, 690.8, 712.0],
        "move_playingcard_away": [814.3, 2012.7, 1784.5, 1832.2, 1165.8, 1341.8],
        "pick_diverse_bottles": [857.7, 1391.2, None, 1688.0, 1253.4, 1621.8],
        "press_stapler": [567.9, 1295.5, 1425.7, 3168.2, 1136.4, 1204.7],
        "rotate_qrcode": [965.5, 1663.9, 1638.8, 1944.5, 1296.1, 1422.3],
        "turn_switch": [441.1, 1798.8, 1153.1, 1821.2, 1246.7, 1454.3],
    },
    "ARX": {
        "Bowl Stacking by Size": [55.3, 76.8, 78.5, 189.3, 70.9, 80.3],
        "Bottle Cap Unscrewing": [53.7, None, None, None, 69.4, 87.3],
        "Cable Hanging": [68.9, 95.4, 95.0, 378.8, 87.6, 102.7],
    },
    "Franka": {
        "Bowl Stacking by Size": [15.1, 27.3, None, 28.5, 24.3, 27.6],
        "Bottle Cap Unscrewing": [28.1, None, None, 34.2, 29.8, 30.4],
        "Keyring Hanging": [12.1, 29.7, None, 32.0, 25.5, 27.7],
    },
}
T8_AVG = {
    "Sim": [658.9, 1463.7, 1468.8, 2022.2, 1064.7, 1219.0],
    "ARX": [59.3, 86.1, 86.8, 284.1, 76.0, 90.1],
    "Franka": [18.4, 28.5, None, 31.6, 26.5, 28.6],
}

# 그림 12 (RoboTwin Clean 성공률, 과제별). 키 (K, S). 숨은(빗금) 칸은 없음. 원문 텍스트에서 옮김.
FIG12 = {
    "beat_block_hammer": {(50, 1): 1, (25, 2): 3, (25, 1): 0, (10, 5): 0, (10, 2): 1, (10, 1): 0,
                          (5, 10): 15, (5, 5): 0, (5, 2): 0, (5, 1): 0, (4, 10): 14, (4, 5): 0, (4, 2): 1, (4, 1): 0,
                          (3, 10): 15, (3, 5): 5, (3, 2): 0, (3, 1): 0, (2, 10): 5, (2, 5): 0, (2, 2): 0, (2, 1): 0,
                          (1, 10): 4, (1, 5): 0, (1, 2): 0, (1, 1): 0},
    "click_bell": {(50, 1): 0, (25, 2): 1, (25, 1): 3, (10, 5): 20, (10, 2): 3, (10, 1): 0,
                   (5, 10): 89, (5, 5): 32, (5, 2): 1, (5, 1): 10, (4, 10): 72, (4, 5): 19, (4, 2): 4, (4, 1): 17,
                   (3, 10): 57, (3, 5): 3, (3, 2): 1, (3, 1): 8, (2, 10): 27, (2, 5): 7, (2, 2): 3, (2, 1): 3,
                   (1, 10): 22, (1, 5): 21, (1, 2): 3, (1, 1): 15},
    "move_playingcard_away": {(50, 1): 44, (25, 2): 59, (25, 1): 9, (10, 5): 65, (10, 2): 16, (10, 1): 13,
                              (5, 10): 71, (5, 5): 54, (5, 2): 2, (5, 1): 0, (4, 10): 69, (4, 5): 50, (4, 2): 5, (4, 1): 0,
                              (3, 10): 62, (3, 5): 5, (3, 2): 0, (3, 1): 0, (2, 10): 31, (2, 5): 0, (2, 2): 0, (2, 1): 0,
                              (1, 10): 2, (1, 5): 0, (1, 2): 0, (1, 1): 1},
    "pick_diverse_bottles": {(50, 1): 0, (25, 2): 9, (25, 1): 0, (10, 5): 17, (10, 2): 0, (10, 1): 0,
                             (5, 10): 27, (5, 5): 14, (5, 2): 0, (5, 1): 0, (4, 10): 25, (4, 5): 2, (4, 2): 0, (4, 1): 0,
                             (3, 10): 12, (3, 5): 1, (3, 2): 0, (3, 1): 0, (2, 10): 4, (2, 5): 0, (2, 2): 0, (2, 1): 0,
                             (1, 10): 1, (1, 5): 0, (1, 2): 0, (1, 1): 0},
    "press_stapler": {(50, 1): 52, (25, 2): 51, (25, 1): 27, (10, 5): 64, (10, 2): 0, (10, 1): 20,
                      (5, 10): 67, (5, 5): 18, (5, 2): 22, (5, 1): 1, (4, 10): 39, (4, 5): 27, (4, 2): 29, (4, 1): 7,
                      (3, 10): 41, (3, 5): 28, (3, 2): 40, (3, 1): 0, (2, 10): 37, (2, 5): 30, (2, 2): 9, (2, 1): 0,
                      (1, 10): 48, (1, 5): 29, (1, 2): 16, (1, 1): 23},
    "rotate_qrcode": {(50, 1): 0, (25, 2): 3, (25, 1): 0, (10, 5): 33, (10, 2): 2, (10, 1): 1,
                      (5, 10): 52, (5, 5): 12, (5, 2): 1, (5, 1): 0, (4, 10): 17, (4, 5): 6, (4, 2): 0, (4, 1): 0,
                      (3, 10): 24, (3, 5): 9, (3, 2): 0, (3, 1): 0, (2, 10): 7, (2, 5): 2, (2, 2): 0, (2, 1): 0,
                      (1, 10): 2, (1, 5): 0, (1, 2): 0, (1, 1): 0},
    "turn_switch": {(50, 1): 25, (25, 2): 30, (25, 1): 40, (10, 5): 33, (10, 2): 36, (10, 1): 42,
                    (5, 10): 27, (5, 5): 37, (5, 2): 28, (5, 1): 33, (4, 10): 30, (4, 5): 19, (4, 2): 20, (4, 1): 23,
                    (3, 10): 33, (3, 5): 22, (3, 2): 25, (3, 1): 22, (2, 10): 28, (2, 5): 28, (2, 2): 25, (2, 1): 26,
                    (1, 10): 16, (1, 5): 27, (1, 2): 25, (1, 1): 30},
}
# 그림 11 (7과제 평균, Clean)
FIG11 = {(50, 1): 17.4, (25, 2): 22.3, (25, 1): 11.3, (10, 5): 33.1, (10, 2): 8.3, (10, 1): 10.9,
         (5, 10): 49.7, (5, 5): 23.9, (5, 2): 7.7, (5, 1): 6.3, (4, 10): 38.0, (4, 5): 17.6, (4, 2): 8.4, (4, 1): 6.7,
         (3, 10): 34.9, (3, 5): 10.4, (3, 2): 9.4, (3, 1): 4.3, (2, 10): 19.9, (2, 5): 9.6, (2, 2): 5.3, (2, 1): 4.1,
         (1, 10): 13.6, (1, 5): 11.0, (1, 2): 6.3, (1, 1): 9.9}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def two_prop_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return z, 2 * stats.norm.sf(abs(z))


def fisher(k1, n1, k2, n2):
    return stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]


def cmh(tables):
    """층별 2x2 (a, n1, c, n2)의 Cochran-Mantel-Haenszel 검정(연속성 보정 없음)과 공통 오즈비(MH)."""
    num = 0.0
    var = 0.0
    r_num = 0.0
    r_den = 0.0
    for a, n1, c, n2 in tables:
        n = n1 + n2
        m1 = a + c
        m0 = n - m1
        ea = n1 * m1 / n
        va = n1 * n2 * m1 * m0 / (n * n * (n - 1))
        num += a - ea
        var += va
        b, d = n1 - a, n2 - c
        r_num += a * d / n
        r_den += b * c / n
    chi2 = num * num / var
    return {"chi2": chi2, "p": float(stats.chi2.sf(chi2, 1)), "or_mh": r_num / r_den if r_den else float("inf")}


def main():
    R = {}
    mi = {m: i for i, m in enumerate(METHODS)}

    # ---------- V1: 표 1 평균 ----------
    v1 = {}
    for m in METHODS:
        i = mi[m]
        clean = [T1[t][i][0] for t in T1]
        rand = [T1[t][i][1] for t in T1]
        v1[m] = {"clean_mean": round(sum(clean) / 7, 2), "rand_mean": round(sum(rand) / 7, 2),
                 "reported": T1_AVG[m],
                 "match": (round(sum(clean) / 7, 2) == T1_AVG[m][0]) and (round(sum(rand) / 7, 2) == T1_AVG[m][1])}
    R["V1_table1_averages"] = v1

    # 과제별 최고 (굵게 표시 근거): 각 과제·조건에서 1위
    best = {}
    for cond, ci in (("clean", 0), ("rand", 1)):
        cnt = {m: 0 for m in METHODS}
        for t in T1:
            vals = [T1[t][mi[m]][ci] for m in METHODS]
            mx = max(vals)
            for m in METHODS:
                if T1[t][mi[m]][ci] == mx:
                    cnt[m] += 1
        best[cond] = cnt
    R["V1b_task_wins_incl_ties"] = best

    # ---------- V2: 표 2 평균 (표 값 정수 평균 대 시행 수 복원 평균) ----------
    v2 = {}
    for rob in ("ARX", "Franka"):
        rows = [v for (r, _), v in T2.items() if r == rob]
        for m in METHODS:
            i = mi[m]
            ints = [r[i] for r in rows]
            counts = [round(x * 0.3) for x in ints]
            v2.setdefault(rob, {})[m] = {
                "mean_of_reported_ints": round(sum(ints) / 3, 2),
                "mean_from_counts": round(100 * sum(counts) / 90, 2),
                "reported_avg": T2_AVG[rob][i],
            }
    ov = {}
    for m in METHODS:
        i = mi[m]
        ints = [v[i] for v in T2.values()]
        counts = [round(x * 0.3) for x in ints]
        ov[m] = {"mean_of_reported_ints": round(sum(ints) / 6, 2), "mean_from_counts": round(100 * sum(counts) / 180, 2),
                 "mean_of_reported_robot_avgs": round((T2_AVG["ARX"][i] + T2_AVG["Franka"][i]) / 2, 2),
                 "reported_avg": T2_AVG["Overall"][i]}
    v2["Overall"] = ov
    R["V2_table2_averages"] = v2

    # 정수 %의 표기 규칙: 30회 → k/30 의 정확값과 표기값
    conv = {}
    for (rob, task), row in T2.items():
        for m, x in zip(METHODS, row):
            k = round(x * 0.3)
            exact = 100 * k / 30
            style = "반올림" if round(exact) == x else ("내림" if math.floor(exact) == x else "기타")
            conv[f"{rob}|{task}|{m}"] = {"reported": x, "k": k, "exact": round(exact, 2), "style": style}
    # 소수 .33/.67 값만 표기 방식을 센다
    frac = {"반올림": 0, "내림": 0, "정확": 0}
    for key, d in conv.items():
        e = d["exact"]
        if abs(e - round(e)) < 0.01:
            frac["정확"] += 1
        elif d["style"] == "반올림":
            frac["반올림"] += 1
        elif d["style"] == "내림":
            frac["내림"] += 1
    R["V2b_rounding_of_30_trial_percentages"] = {"cells_by_style": frac, "detail": conv}

    # ---------- V3: 본문 z 검정 (REACT 대 Async+RTC) ----------
    def sim_pool(m, ci):
        return sum(T1[t][mi[m]][ci] for t in T1), 700

    v3 = {}
    for name, (ci) in (("clean", 0), ("rand", 1)):
        k1, n1 = sim_pool("REACT", ci)
        k2, n2 = sim_pool("AsyncRTC", ci)
        z, p = two_prop_z(k1, n1, k2, n2)
        v3[f"sim_{name}_REACT_vs_AsyncRTC"] = {"k": [k1, k2], "n": [n1, n2], "z": round(z, 2), "p": p}
    kR = sum(round(v[mi["REACT"]] * 0.3) for v in T2.values())
    kA = sum(round(v[mi["AsyncRTC"]] * 0.3) for v in T2.values())
    z, p = two_prop_z(kR, 180, kA, 180)
    v3["real_REACT_vs_AsyncRTC"] = {"k": [kR, kA], "n": [180, 180], "z": round(z, 2), "p": p}
    v3["paper_reported_z"] = {"sim_clean": 4.44, "sim_rand": 4.84, "real": 9.10}
    R["V3_paper_ztests"] = v3

    # ---------- V4: 모든 쌍 비교 (REACT 변형 대 기준선), 합산 z, CMH(과제 층별), Fisher ----------
    pairs = [("REACT", "AsyncRTC"), ("REACT", "H50E50"), ("REACT", "H50E10"), ("REACT_noDD", "H50E50"),
             ("REACT_noDD", "AsyncRTC"), ("REACT_noDD", "REACT")]
    v4 = {}
    for a, b in pairs:
        row = {}
        for cond, ci in (("sim_clean", 0), ("sim_rand", 1)):
            ka, na = sim_pool(a, ci)
            kb, nb = sim_pool(b, ci)
            z, p = two_prop_z(ka, na, kb, nb)
            tabs = [(T1[t][mi[a]][ci], 100, T1[t][mi[b]][ci], 100) for t in T1]
            diffs = [x[0] - x[2] for x in tabs]
            wins = sum(d > 0 for d in diffs)
            losses = sum(d < 0 for d in diffs)
            sign_p = stats.binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else 1.0
            row[cond] = {"pooled_pct": [round(100 * ka / na, 2), round(100 * kb / nb, 2)], "z": round(z, 2), "p_pooled": p,
                         "cmh": cmh(tabs), "task_wins_losses": [wins, losses], "sign_p": sign_p,
                         "wilson_a": [round(100 * x, 1) for x in wilson(ka, na)],
                         "wilson_b": [round(100 * x, 1) for x in wilson(kb, nb)]}
        tabs = [(round(v[mi[a]] * 0.3), 30, round(v[mi[b]] * 0.3), 30) for v in T2.values()]
        ka = sum(t[0] for t in tabs)
        kb = sum(t[2] for t in tabs)
        z, p = two_prop_z(ka, 180, kb, 180)
        diffs = [t[0] - t[2] for t in tabs]
        wins = sum(d > 0 for d in diffs)
        losses = sum(d < 0 for d in diffs)
        sign_p = stats.binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else 1.0
        row["real"] = {"k": [ka, kb], "pooled_pct": [round(100 * ka / 180, 2), round(100 * kb / 180, 2)], "z": round(z, 2),
                       "p_pooled": p, "fisher_p": fisher(ka, 180, kb, 180), "cmh": cmh(tabs),
                       "task_wins_losses": [wins, losses], "sign_p": sign_p,
                       "wilson_a": [round(100 * x, 1) for x in wilson(ka, 180)],
                       "wilson_b": [round(100 * x, 1) for x in wilson(kb, 180)]}
        v4[f"{a}_vs_{b}"] = row
    R["V4_pairwise_tests"] = v4

    # 실세계에서 REACT가 H50E50을 넘기려면 필요한 시행 수 (두 비율 z, 80% 검정력, 쌍 55~65 %p 차)
    def n_needed(p1, p2, alpha=0.05, power=0.8):
        za, zb = stats.norm.isf(alpha / 2), stats.norm.isf(1 - power)
        pb = (p1 + p2) / 2
        return math.ceil(((za * math.sqrt(2 * pb * (1 - pb)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p1 - p2) ** 2)

    R["V4b_n_needed_per_arm"] = {
        "real_REACT_vs_H50E50 (64.8 vs 58.7)": n_needed(0.648, 0.587),
        "real_REACT_noDD_vs_H50E50 (65.7 vs 58.7)": n_needed(0.657, 0.587),
        "sim_clean_REACT_vs_AsyncRTC (44.86 vs 33.29)": n_needed(0.4486, 0.3329),
    }

    # ---------- V5: 표 3 ----------
    pour_k = [round(x * 0.3) for x in T3_POUR]
    pour = {}
    for m, k in zip(METHODS, pour_k):
        pour[m] = {"k_assumed_n30": k, "wilson": [round(100 * x, 1) for x in wilson(k, 30)]}
    pour_tests = {}
    for b in ("H50E10", "REACT_noDD", "H50E50", "AsyncRTC"):
        pour_tests[f"REACT_vs_{b}"] = {"fisher_p": fisher(pour_k[mi["REACT"]], 30, pour_k[mi[b]], 30)}
    pour_tests["REACT_noDD_vs_H50E10"] = {"fisher_p": fisher(pour_k[mi["REACT_noDD"]], 30, pour_k[mi["H50E10"]], 30)}
    R["V5a_pour_rice"] = {"per_method": pour, "tests": pour_tests, "assumption": "시도 수 n=30 가정"}

    react_tests = {}
    for n in (20, 30, 50):
        mt, st = T3_REACTION["REACT"]
        out = {}
        for b, (mb, sb) in T3_REACTION.items():
            if b == "REACT":
                continue
            se = math.sqrt(st ** 2 / n + sb ** 2 / n)
            t = (mt - mb) / se
            dfree = (st ** 2 / n + sb ** 2 / n) ** 2 / ((st ** 2 / n) ** 2 / (n - 1) + (sb ** 2 / n) ** 2 / (n - 1))
            out[b] = {"diff_ms": mt - mb, "t": round(t, 2), "p_welch": float(2 * stats.t.sf(abs(t), dfree))}
        react_tests[f"n={n}"] = out
    R["V5b_reaction_game_welch"] = {"tests": react_tests, "assumption": "시도 수 n을 20/30/50으로 가정 (논문 미보고), SD는 표본 SD로 가정",
                                    "excess_vs_human_ms": 734 - 705, "H50E50_vs_human_ms": 1505 - 705}

    # ---------- V6: 표 8 평균 ----------
    v6 = {}
    for grp, rows in T8.items():
        for i, m in enumerate(METHODS):
            vals = [r[i] for r in rows.values() if r[i] is not None]
            rep = T8_AVG[grp][i]
            v6[f"{grp}|{m}"] = {"mean_over_available": round(sum(vals) / len(vals), 2) if vals else None, "n_tasks": len(vals),
                                "reported": rep,
                                "match": (rep is None and not vals) or (rep is not None and vals and abs(round(sum(vals) / len(vals), 1) - rep) < 0.06)}
    R["V6_table8_averages"] = v6
    # 공통 과제만 비교
    common = {}
    for grp, rows in T8.items():
        keys = [k for k, r in rows.items() if all(r[mi[m]] is not None for m in ("H50E50", "H50E10", "REACT"))]
        for a, b in (("REACT", "H50E50"), ("REACT", "H50E10"), ("REACT_noDD", "H50E10")):
            if keys:
                ma = sum(rows[k][mi[a]] for k in keys) / len(keys)
                mb = sum(rows[k][mi[b]] for k in keys) / len(keys)
                common[f"{grp}|{a}_vs_{b}"] = {"tasks": keys, "a": round(ma, 1), "b": round(mb, 1), "ratio": round(ma / mb, 2)}
        keys = [k for k, r in rows.items() if r[mi["AsyncRTC"]] is not None and r[mi["REACT"]] is not None]
        if keys:
            ma = sum(rows[k][mi["REACT"]] for k in keys) / len(keys)
            mb = sum(rows[k][mi["AsyncRTC"]] for k in keys) / len(keys)
            common[f"{grp}|REACT_vs_AsyncRTC"] = {"tasks": keys, "a": round(ma, 1), "b": round(mb, 1), "ratio": round(ma / mb, 2)}
    R["V6b_common_task_jerk"] = common
    # 표 8의 "–"와 표 1·2의 0% 정합: 시뮬 Clean 0%인 칸이 "–", 그리고 Rand만 성공이 있는 칸
    dash = {}
    for t, vals in T8["Sim"].items():
        for m in METHODS:
            i = mi[m]
            c, r = T1[t][i]
            if vals[i] is None:
                dash[f"{t}|{m}"] = {"clean": c, "rand": r}
    R["V6c_table8_dash_cells_sim"] = dash

    # ---------- V7: 표 5·4·7 처리량 ----------
    f_cam, f_ctrl, T_vlm, T_dit, N = 30.0, 30.0, 0.037, 0.003, 10
    T_inf = T_vlm + N * T_dit
    t5 = {}
    for He in (50, 25, 10, 5):
        cyc = T_inf + He / f_ctrl
        t5[f"sync_He{He}"] = {"Tcycle_ms": round(1000 * cyc, 1), "phi_in": round(1 / cyc, 3), "phi_out": round(He / cyc, 2)}
        cyc = 0.5 * T_inf + He / f_ctrl
        t5[f"accel_He{He}"] = {"Tcycle_ms": round(1000 * cyc, 1), "phi_in": round(1 / cyc, 3), "phi_out": round(He / cyc, 2)}
        cyc = max(T_inf, He / f_ctrl)
        t5[f"async_He{He}"] = {"Tcycle_ms": round(1000 * cyc, 1), "phi_in": round(1 / cyc, 3), "phi_out": round(He / cyc, 2)}
    for K, S in ((10, 5), (5, 10)):
        cyc = T_vlm + T_dit + S / f_ctrl
        t5[f"react_noDD_K{K}_S{S}"] = {"Tcycle_ms": round(1000 * cyc, 1), "phi_in": round(1 / cyc, 3), "phi_out": round(S / cyc, 2)}
    for S in (10, 5, 1):
        t5[f"react_DD_S{S}"] = {"phi_in": round(min(f_cam / S, 1 / T_dit), 2), "phi_out": round(min(f_cam, S / T_dit), 2),
                                "vlm_pool_needed_hz": f_cam / S, "single_worker_hz": round(1 / T_vlm, 1)}
    R["V7_table5_throughput"] = t5
    R["V7b_table7_latency"] = {"speedup": round(66.04 / 39.96, 3), "reported": 1.65,
                               "model_T_inf_ms": 1000 * T_inf, "measured_std_ms": 66.04,
                               "model_react_ms": 1000 * (T_vlm + T_dit), "measured_react_ms": 39.96,
                               "vlm_share_of_react_update": round(T_vlm / (T_vlm + T_dit), 3),
                               "dit_flops_ratio_per_executed_block": "H50E10: DiT 10단계×50토큰 / REACT: 1단계×50토큰 = 10배"}
    # 지연 보고: 임베딩 나이
    R["V7c_embedding_age"] = {"mean_ms": 233, "p95_ms": 383, "max_ms": 416, "period_ms": 1000 * 10 / 30,
                              "max_over_period": round(416 / (1000 * 10 / 30), 3),
                              "T_vlm_ms": 37, "comment": "나이가 VLM 지연(37 ms)보다 훨씬 큼 -> 두 루프의 위상차가 지배한다고 볼 수 있음"}

    # ---------- V8: 그림 11 대 그림 12 ----------
    v8 = {}
    maxdev = 0
    for kk, rep in FIG11.items():
        vals = [FIG12[t][kk] for t in FIG12]
        mean = sum(vals) / 7
        dev = abs(round(mean, 1) - rep)
        maxdev = max(maxdev, dev)
        v8[f"K{kk[0]}_S{kk[1]}"] = {"mean_of_fig12": round(mean, 2), "fig11": rep, "abs_dev": round(dev, 2)}
    R["V8_fig11_vs_fig12"] = {"cells": v8, "max_abs_dev": round(maxdev, 2), "n_cells": len(FIG11)}
    # (5,10) 열은 표 1 REACT w/o DD Clean과 같은가
    r = {t: FIG12[t][(5, 10)] for t in FIG12}
    R["V8b_fig12_K5S10_vs_table1_noDD_clean"] = {t: [r[t], T1[t][mi["REACT_noDD"]][0]] for t in T1}
    # 6/7 과제 최적 또는 공동 최적
    opt = {}
    for t in FIG12:
        mx = max(FIG12[t].values())
        opt[t] = {"K5S10": FIG12[t][(5, 10)], "max": mx, "best_or_tied": FIG12[t][(5, 10)] == mx,
                  "argmax": [list(k) for k, v in FIG12[t].items() if v == mx]}
    R["V8c_optimal_or_tied"] = opt
    # K와 H 혼입: S=10 열은 K=1..5에서 H=10..50
    R["V8d_K_H_confound"] = {"S10_column": {f"K{K}": {"H": K * 10, "success": FIG11[(K, 10)]} for K in (1, 2, 3, 4, 5)},
                             "same_H50": {"K5_S10": FIG11[(5, 10)], "K10_S5": FIG11[(10, 5)], "K25_S2": FIG11[(25, 2)],
                                          "K50_S1": FIG11[(50, 1)]}}
    # 구성별 Wilson (700회)
    R["V8e_wilson_700"] = {k: [round(100 * x, 1) for x in wilson(round(v * 7), 700)] for k, v in
                           {"K5S10": 49.7, "K10S5": 33.1, "K4S10": 38.0}.items()}

    # ---------- V9: 그림 5(b)·13, 초록 수치 ----------
    three = ["click_bell", "move_playingcard_away", "press_stapler"]
    R["V9a_fig5b_endpoints"] = {
        "H50E50_clean_3task_mean": round(sum(T1[t][mi["H50E50"]][0] for t in three) / 3, 1),
        "REACT_noDD_clean_3task_mean": round(sum(T1[t][mi["REACT_noDD"]][0] for t in three) / 3, 1),
        "REACT_clean_3task_mean": round(sum(T1[t][mi["REACT"]][0] for t in three) / 3, 1),
        "fig5b_labels": {"REACT": 75.7, "pi05": 59.3},
        "se_300_rollouts_pct": round(100 * math.sqrt(0.6 * 0.4 / 300), 1),
    }
    base = [0.312, 0.244, 0.259]
    ours = [0.058, 0.020, 0.067]
    R["V9b_open_loop_mse"] = {"baseline_mean": round(sum(base) / 3, 4), "react_mean": round(sum(ours) / 3, 4),
                              "reduction_from_rounded_means": round(1 - 0.048 / 0.272, 4),
                              "reduction_from_figure_values": round(1 - (sum(ours) / 3) / (sum(base) / 3), 4),
                              "reported": 0.824, "note": "그림 13의 막대 값 6개(그림 판독, 과제 대응은 불확실)"}
    R["V9c_text_claims"] = {
        "H50E10_real_drop": [58.7, 35.9],
        "reaction_gap_to_human_ms": 734 - 705,
        "H50E50_gap_to_human_ms": 1505 - 705,
        "jerk_ratio_REACT_vs_Async_sim": round(1219.0 / 2022.2, 3),
        "jerk_ratio_REACT_vs_demo_sim": round(1219.0 / 570.1, 2),
        "jerk_ratio_REACT_vs_H50E50_sim": round(1219.0 / 658.9, 2),
        "jerk_ratio_noDD_vs_H50E50_sim": round(1064.7 / 658.9, 2),
    }

    # ---------- V10: 헤드라인 Wilson 구간 ----------
    hl = {}
    for m in METHODS:
        i = mi[m]
        kc = sum(T1[t][i][0] for t in T1)
        kr = sum(T1[t][i][1] for t in T1)
        kw = sum(round(v[i] * 0.3) for v in T2.values())
        hl[m] = {"sim_clean": [round(100 * kc / 700, 1), [round(100 * x, 1) for x in wilson(kc, 700)]],
                 "sim_rand": [round(100 * kr / 700, 1), [round(100 * x, 1) for x in wilson(kr, 700)]],
                 "real": [round(100 * kw / 180, 1), [round(100 * x, 1) for x in wilson(kw, 180)]]}
    R["V10_headline_wilson"] = hl

    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("wrote", OUT)
    # 요약 출력
    print("V1 match:", {m: v["match"] for m, v in v1.items()})
    print("V3:", {k: (v if k == "paper_reported_z" else (v["z"], f'{v["p"]:.2e}')) for k, v in v3.items()})
    for k, v in v4.items():
        print(k, "sim_clean z", v["sim_clean"]["z"], "cmh p", f'{v["sim_clean"]["cmh"]["p"]:.3g}', "wl", v["sim_clean"]["task_wins_losses"],
              "| real", v["real"]["pooled_pct"], "z", v["real"]["z"], "fisher", f'{v["real"]["fisher_p"]:.3g}', "cmh", f'{v["real"]["cmh"]["p"]:.3g}', v["real"]["task_wins_losses"])
    print("V8 max dev:", R["V8_fig11_vs_fig12"]["max_abs_dev"])
    print("V6 mismatches:", [k for k, v in v6.items() if not v["match"]])


if __name__ == "__main__":
    main()
