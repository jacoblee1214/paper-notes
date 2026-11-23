"""TDC (arXiv:2602.14174) 표 수치 재계산과 검정.

입력은 논문(v2)의 Table 1, 2, 5와 Fig. 5·6의 값, 그리고 v1(Table II, III, IV)의 값이다.
Fig. 5·6 값은 그림에서 읽은 값이다(판독 오차 약 ±3 %p, Fig. 6은 시행 20회라 5 %p 격자에 맞춤).
시행 수는 논문 4.1절의 "과제·방법당 20회 + 교란 5회"를 쓴다(Table 5 캡션의 80, 20과 일치).

실행: python3 analysis/recompute.py   → analysis/recompute_results.json
"""
import json
import math
from itertools import combinations
from pathlib import Path

from scipy.stats import fisher_exact

OUT = Path(__file__).with_name("recompute_results.json")
N_ND, N_D = 20, 5
TASKS = ["MO", "PH", "WW", "DO"]

# 논문 v2 Table 1: 방법 -> 과제 -> (SR-ND, SR-D) %,  None은 "-"
T1 = {
    "Pi0-Real": {"PH": (30, 40), "WW": (60, 20)},
    "E2VLA-Real": {"PH": (35, 20), "WW": (90, 60)},
    "ForceVLA-Real": {"PH": (15, 40), "WW": (100, 80)},
    "ACP-Real": {"PH": (45, 20), "WW": (60, 0)},
    "Pi0-Sim+L-A": {"MO": (30, 60), "PH": (35, 60), "WW": (35, 60), "DO": (80, 100)},
    "Pi0-Sim+M-A": {"MO": (75, 60), "PH": (30, 60), "WW": (60, 60), "DO": (90, 40)},
    "Pi0-Sim+H-A": {"MO": (60, 40), "PH": (35, 20), "WW": (45, 20), "DO": (80, 40)},
    "E2VLA-Sim+L-A": {"MO": (15, 40), "PH": (55, 100), "WW": (30, 60), "DO": (65, 80)},
    "E2VLA-Sim+M-A": {"MO": (90, 20), "PH": (55, 60), "WW": (55, 80), "DO": (85, 40)},
    "E2VLA-Sim+H-A": {"MO": (70, 20), "PH": (70, 40), "WW": (65, 40), "DO": (75, 40)},
    "ForceVLA-Sim": {"MO": (0, 0), "PH": (0, 60), "WW": (5, 40), "DO": (0, 20)},
    "ACP-Sim": {"MO": (0, 0), "PH": (5, 20), "WW": (15, 0), "DO": (0, 0)},
    "Ours": {"MO": (95, 80), "PH": (90, 100), "WW": (100, 100), "DO": (95, 80)},
}
# Table 1의 Overall 열 (SR-ND, SR-D)
T1_OVERALL = {
    "Pi0-Real": (45, 30), "E2VLA-Real": (63, 40), "ForceVLA-Real": (58, 60), "ACP-Real": (53, 10),
    "Pi0-Sim+L-A": (45, 70), "Pi0-Sim+M-A": (64, 55), "Pi0-Sim+H-A": (55, 30),
    "E2VLA-Sim+L-A": (41, 70), "E2VLA-Sim+M-A": (71, 50), "E2VLA-Sim+H-A": (70, 35),
    "ForceVLA-Sim": (1, 30), "ACP-Sim": (5, 5), "Ours": (95, 90),
}
# Fig. 5 (그림 판독): 실패 시행 수와 유형 비율 %(불충분, 과도, 기타). 비율이 그림에 없는 칸은 None
FIG5 = {
    "Pi0-Sim+L-A": (50, (64, None, 34)),
    "Pi0-Sim+M-A": (38, (32, 37, 32)),
    "Pi0-Sim+H-A": (50, (30, 48, 22)),
    "E2VLA-Sim+L-A": (53, (75, 6, 19)),
    "E2VLA-Sim+M-A": (33, (42, 36, 21)),
    "E2VLA-Sim+H-A": (37, (16, 68, 16)),
    "Ours": (6, (None, 67, 33)),
}
# Table 2: 방법 -> (삽입 깊이 mm, 남은 잉크 cm)
T2 = {
    "Pi0-Real": (23.1, 1.7), "E2VLA-Real": (25.0, 0.8), "ForceVLA-Real": (22.0, 0.5), "ACP-Real": (21.0, 1.3),
    "Pi0-Sim+L-A": (21.6, 2.3), "Pi0-Sim+M-A": (21.3, 2.2), "Pi0-Sim+H-A": (22.7, 1.3),
    "E2VLA-Sim+L-A": (21.9, 2.0), "E2VLA-Sim+M-A": (23.5, 1.1), "E2VLA-Sim+H-A": (21.0, 0.8),
    "ForceVLA-Sim": (20.0, 4.0), "ACP-Sim": (17.5, 3.7), "Ours": (25.0, 0.3),
}
# Table 5: 번호 -> (사전학습, 컨트롤러, SR-ND %, SR-ND CI, SR-D %, SR-D CI)
T5 = {
    1: ("×", "적응형", 76.25, (65.86, 84.24), 95.00, (76.39, 99.11)),
    2: ("✓", "L-A", 36.25, (26.57, 47.19), 60.00, (38.66, 78.12)),
    3: ("✓", "M-A", 65.00, (54.08, 74.55), 40.00, (21.88, 61.34)),
    4: ("✓", "H-A", 51.25, (40.49, 61.89), 10.00, (2.79, 30.10)),
    5: ("✓", "적응형", 95.00, (87.84, 98.04), 90.00, (69.90, 97.21)),
}
# Fig. 6 (그림 판독, 시행 20회 = ND 10 + D 10, 5 %p 격자)
FIG6 = {
    "a_stiffness": {"x": ["25/5", "50/10", "75/15", "100/20"],
                    "MO": [80, 85, 70, 60], "PH": [90, 95, 100, 85],
                    "WW": [85, 95, 95, 85], "DO": [90, 85, 90, 65]},
    "b_force": {"PH_x": [1, 2, 4, 6, 8], "PH": [85, 95, 90, 75, 35],
                "WW_x": [2, 4, 6, 8, 10], "WW": [85, 95, 100, 65, 45]},
    "c_ratio": {"x": [1, 2, 4, 6, 8, 16], "MO": [30, 60, 85, 80, 65, 40]},
}
# v1 Table II (25회 = ND 20 + D 5): 방법 -> 과제 -> (SR-ND, SR-D, SR)
V1_T2 = {
    "Pi0-Sim+L-A": {"MO": (30, 60, 36), "PH": (35, 80, 44), "WW": (35, 60, 40), "DO": (80, 100, 84)},
    "Pi0-Sim+M-A": {"MO": (75, 60, 72), "PH": (30, 60, 36), "WW": (60, 60, 60), "DO": (90, 40, 80)},
    "Pi0-Sim+H-A": {"MO": (60, 40, 56), "PH": (35, 20, 32), "WW": (45, 20, 40), "DO": (80, 40, 72)},
    "E2VLA-Sim+L-A": {"MO": (15, 40, 20), "PH": (55, 100, 64), "WW": (30, 60, 36), "DO": (65, 80, 68)},
    "E2VLA-Sim+M-A": {"MO": (90, 20, 76), "PH": (55, 60, 56), "WW": (55, 80, 60), "DO": (85, 40, 76)},
    "E2VLA-Sim+H-A": {"MO": (70, 20, 60), "PH": (70, 40, 64), "WW": (65, 40, 60), "DO": (75, 40, 68)},
    "Ours": {"MO": (100, 100, 100), "PH": (75, 100, 80), "WW": (95, 100, 96), "DO": (90, 80, 88)},
}
V1_OVERALL = {  # (SR-ND, SR-D, SR)
    "Pi0-Sim+L-A": (45, 75, 51), "Pi0-Sim+M-A": (64, 55, 62), "Pi0-Sim+H-A": (55, 30, 50),
    "E2VLA-Sim+L-A": (41, 70, 47), "E2VLA-Sim+M-A": (71, 50, 67), "E2VLA-Sim+H-A": (70, 35, 63),
    "Ours": (90, 95, 91),
}
V1_T3 = {  # (삽입 깊이, 남은 잉크)
    "Pi0-Sim+L-A": (21.5, 2.3), "Pi0-Sim+M-A": (21.3, 2.8), "Pi0-Sim+H-A": (22.8, 1.3),
    "E2VLA-Sim+L-A": (21.9, 2.0), "E2VLA-Sim+M-A": (23.6, 1.1), "E2VLA-Sim+H-A": (21.0, 0.8),
    "Ours": (25.0, 0.8),
}
# v1 Table IV (화이트보드 절제, 25회 = ND 20 + D 5): 번호 -> (SR-ND, SR-D, SR, 남은 잉크)
V1_T4 = {1: (75, 80, 76, 2.7), 2: (30, 80, 40, 1.7), 3: (75, 80, 76, 3.7), 4: (30, 80, 40, 3.5),
         5: (40, 80, 48, 2.4), 6: (40, 40, 40, 1.8), 7: (95, 100, 96, 0.8)}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def fisher(a, na, b, nb):
    """a/na 대 b/nb 양측 Fisher 정확검정 p."""
    return fisher_exact([[a, na - a], [b, nb - b]])[1]


def cnt(pct, n):
    return pct * n / 100


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def main():
    R = {}

    # V1: Overall 열 재계산 (가용 과제 단순 평균)
    v1 = {}
    for m, cells in T1.items():
        nd = sum(c[0] for c in cells.values()) / len(cells)
        d = sum(c[1] for c in cells.values()) / len(cells)
        pub = T1_OVERALL[m]
        v1[m] = {"n_tasks": len(cells), "nd_mean": nd, "d_mean": d, "pub": pub,
                 "nd_match": round(nd) == pub[0] or abs(nd - pub[0]) <= 0.5,
                 "d_match": round(d) == pub[1] or abs(d - pub[1]) <= 0.5}
    R["V1_overall"] = v1

    # V2: 정수 시행 수 검사 (ND n=20, D n=5)
    bad = []
    for m, cells in T1.items():
        for t, (nd, d) in cells.items():
            if abs(cnt(nd, N_ND) - round(cnt(nd, N_ND))) > 1e-9:
                bad.append((m, t, "ND", nd))
            if abs(cnt(d, N_D) - round(cnt(d, N_D))) > 1e-9:
                bad.append((m, t, "D", d))
    R["V2_integer_violations"] = bad

    # V3: 성공·실패 수, Fig. 5 실패 수와 비교
    v3 = {}
    for m, cells in T1.items():
        if len(cells) < 4:
            continue
        s_nd = sum(round(cnt(c[0], N_ND)) for c in cells.values())
        s_d = sum(round(cnt(c[1], N_D)) for c in cells.values())
        v3[m] = {"succ_nd": s_nd, "succ_d": s_d, "fail": 100 - s_nd - s_d,
                 "fig5_fail": FIG5.get(m, (None,))[0]}
    R["V3_failures"] = v3

    # V4: Fig. 5 비율이 정수 개수와 맞는지
    v4 = {}
    for m, (n, pcts) in FIG5.items():
        known = [p for p in pcts if p is not None]
        sols = []
        for a in range(n + 1):
            for b in range(n + 1 - a):
                c = n - a - b
                cs = (a, b, c)
                ok = all(p is None or round(100 * x / n) == p for x, p in zip(cs, pcts))
                if ok:
                    sols.append(cs)
        v4[m] = {"n": n, "pcts": pcts, "integer_solutions": sols, "n_solutions": len(sols),
                 "pct_sum": sum(known)}
    R["V4_fig5_percent"] = v4

    # V5: 헤드라인 +24, +20 pp와 검정
    best_nd = max((v for m, v in T1_OVERALL.items() if m != "Ours"), key=lambda x: x[0])
    best_d = max((v for m, v in T1_OVERALL.items() if m != "Ours"), key=lambda x: x[1])
    ours_nd, ours_d = T1_OVERALL["Ours"]
    v5 = {"claimed_gap_nd": 24, "claimed_gap_d": 20,
          "recomputed_gap_nd": ours_nd - best_nd[0], "recomputed_gap_d": ours_d - best_d[1]}
    sim = {m: c for m, c in T1.items() if len(c) == 4}
    rows = []
    for m, cells in sim.items():
        if m == "Ours":
            continue
        k_nd = sum(round(cnt(c[0], N_ND)) for c in cells.values())
        k_d = sum(round(cnt(c[1], N_D)) for c in cells.values())
        rows.append({"method": m, "k_nd": k_nd, "k_d": k_d,
                     "p_nd": fisher(76, 80, k_nd, 80), "p_d": fisher(18, 20, k_d, 20),
                     "p_total": fisher(94, 100, k_nd + k_d, 100)})
    v5["pooled_vs_sim_baselines"] = rows
    R["V5_headline"] = v5

    # V6: 과제별 Ours 대 시뮬 기준선(ND 20회), 가장 높은 기준선과 비교
    v6 = []
    for t in TASKS:
        cands = [(m, round(cnt(c[t][0], N_ND))) for m, c in T1.items() if t in c and m != "Ours"]
        m_best, k_best = max(cands, key=lambda x: x[1])
        k_o = round(cnt(T1["Ours"][t][0], N_ND))
        v6.append({"task": t, "best_baseline": m_best, "k_best": k_best, "k_ours": k_o,
                   "p": fisher(k_o, 20, k_best, 20)})
    ps = holm([x["p"] for x in v6])
    for x, a in zip(v6, ps):
        x["p_holm"] = a
    R["V6_per_task_vs_best"] = v6
    # Ours 대 E2VLA-Sim+M-A (같은 정책, 가장 높은 등방성) 과제별
    v6b = []
    for t in TASKS:
        a = round(cnt(T1["Ours"][t][0], N_ND))
        b = round(cnt(T1["E2VLA-Sim+M-A"][t][0], N_ND))
        v6b.append({"task": t, "ours": a, "e2vla_m": b, "p": fisher(a, 20, b, 20)})
    R["V6b_vs_e2vla_m"] = v6b

    # V7: Table 5 개수·Wilson 구간·검정
    v7 = {}
    for no, (_, _, nd, ndci, d, dci) in T5.items():
        k_nd, k_d = round(nd * 80 / 100), round(d * 20 / 100)
        v7[no] = {"k_nd": k_nd, "k_d": k_d, "nd_int_ok": abs(nd * 80 / 100 - k_nd) < 1e-9,
                  "d_int_ok": abs(d * 20 / 100 - k_d) < 1e-9,
                  "ci_nd_calc": wilson(k_nd, 80), "ci_nd_pub": ndci,
                  "ci_d_calc": wilson(k_d, 20), "ci_d_pub": dci}
    R["V7_table5"] = v7
    cmp5 = {}
    for no in (1, 2, 3, 4):
        cmp5[f"5_vs_{no}"] = {"p_nd": fisher(v7[5]["k_nd"], 80, v7[no]["k_nd"], 80),
                              "p_d": fisher(v7[5]["k_d"], 20, v7[no]["k_d"], 20)}
    R["V7b_table5_tests"] = cmp5
    R["V7c_no1_vs_iso"] = {f"1_vs_{no}": {"p_nd": fisher(v7[1]["k_nd"], 80, v7[no]["k_nd"], 80),
                                         "p_d": fisher(v7[1]["k_d"], 20, v7[no]["k_d"], 20)} for no in (2, 3, 4)}

    # V8: Table 5 No.2-4 대 Table 1 E2VLA-Sim+L/M/H-A (제한 절 "성능 저하" 주장)
    v8 = {}
    for no, m in ((2, "E2VLA-Sim+L-A"), (3, "E2VLA-Sim+M-A"), (4, "E2VLA-Sim+H-A")):
        c = T1[m]
        k_nd = sum(round(cnt(x[0], N_ND)) for x in c.values())
        k_d = sum(round(cnt(x[1], N_D)) for x in c.values())
        v8[f"No{no}_vs_{m}"] = {"table1_nd": k_nd, "table5_nd": v7[no]["k_nd"],
                                "table1_d": k_d, "table5_d": v7[no]["k_d"],
                                "p_nd": fisher(k_nd, 80, v7[no]["k_nd"], 80),
                                "p_d": fisher(k_d, 20, v7[no]["k_d"], 20)}
    R["V8_action_space_degradation"] = v8

    # V9: Fig. 6 기본 설정 점과 Table 1 / 전역 풀링 대조
    t1_def = {}
    for t in TASKS:
        k = round(cnt(T1["Ours"][t][0], N_ND)) + round(cnt(T1["Ours"][t][1], N_D))
        s = {"MO": FIG6["a_stiffness"]["MO"][1], "PH": FIG6["a_stiffness"]["PH"][1],
             "WW": FIG6["a_stiffness"]["WW"][1], "DO": FIG6["a_stiffness"]["DO"][1]}[t]
        ks = round(s * 20 / 100)
        t1_def[t] = {"table1_k": k, "table1_n": 25, "sweep_k": ks, "sweep_n": 20,
                     "p": fisher(k, 25, ks, 20)}
    R["V9_default_vs_table1"] = t1_def
    # 기본 설정 점들 사이 일관성 (Fig. 6a 50/10 대 6b 기본값 대 6c 비율 4)
    R["V9b_default_consistency"] = {
        "MO_a_vs_c": (FIG6["a_stiffness"]["MO"][1], FIG6["c_ratio"]["MO"][2]),
        "PH_a_vs_b": (FIG6["a_stiffness"]["PH"][1], FIG6["b_force"]["PH"][1]),
        "WW_a_vs_b": (FIG6["a_stiffness"]["WW"][1], FIG6["b_force"]["WW"][1]),
    }

    # V10: 민감도 주장 검정 (각 점 n=20)
    def f20(a, b):
        return fisher(round(a * 20 / 100), 20, round(b * 20 / 100), 20)

    v10 = {
        "MO ratio 1 vs 4 (30 vs 85)": f20(30, 85),
        "MO ratio 4 vs 16 (85 vs 40)": f20(85, 40),
        "MO ratio 4 vs 8 (85 vs 65)": f20(85, 65),
        "MO k 50/10 vs 100/20 (85 vs 60)": f20(85, 60),
        "DO k 50/10 vs 100/20 (85 vs 65)": f20(85, 65),
        "PH k 75/15 vs 100/20 (100 vs 85)": f20(100, 85),
        "PH force 2 vs 8 N (95 vs 35)": f20(95, 35),
        "PH force 2 vs 6 N (95 vs 75)": f20(95, 75),
        "WW force 4 vs 8 N (95 vs 65)": f20(95, 65),
        "WW force 6 vs 10 N (100 vs 45)": f20(100, 45),
        "PH force 1 vs 2 N (85 vs 95)": f20(85, 95),
    }
    R["V10_sensitivity"] = v10
    # 같은 계열에서 사용 가능 범위 (기본 값 대비 -15 %p 이내인 점 수)
    rng = {}
    for t in TASKS:
        ys = FIG6["a_stiffness"][t]
        rng[f"a_{t}"] = sum(1 for y in ys if y >= ys[1] - 15)
    rng["b_PH"] = sum(1 for y in FIG6["b_force"]["PH"] if y >= 95 - 15)
    rng["b_WW"] = sum(1 for y in FIG6["b_force"]["WW"] if y >= 95 - 15)
    rng["c_MO"] = sum(1 for y in FIG6["c_ratio"]["MO"] if y >= 85 - 15)
    R["V10b_points_within_15pp_of_default"] = rng

    # V11: SR-D > SR-ND 칸 수
    gt = {}
    tot = lt = eq = 0
    for m, cells in T1.items():
        n = sum(1 for c in cells.values() if c[1] > c[0])
        gt[m] = n
        tot += n
        lt += sum(1 for c in cells.values() if c[1] < c[0])
        eq += sum(1 for c in cells.values() if c[1] == c[0])
    R["V11_sr_d_gt_nd_cells"] = {"per_method": gt, "total": tot, "lower": lt, "equal": eq,
                                 "n_cells": sum(len(c) for c in T1.values())}

    # V12: v1 대 v2
    v12 = {"changed_cells": [], "same_cells": 0}
    for m, cells in V1_T2.items():
        for t, (nd, d, sr) in cells.items():
            nd2, d2 = T1[m][t]
            if (nd, d) != (nd2, d2):
                v12["changed_cells"].append({"method": m, "task": t, "v1": (nd, d), "v2": (nd2, d2)})
            else:
                v12["same_cells"] += 1
    # v1 SR 열 = (20 ND + 5 D)/25 재계산
    sr_bad = []
    for m, cells in V1_T2.items():
        for t, (nd, d, sr) in cells.items():
            calc = (cnt(nd, 20) + cnt(d, 5)) / 25 * 100
            if abs(calc - sr) > 0.51:
                sr_bad.append((m, t, calc, sr))
    v12["v1_sr_column_mismatch"] = sr_bad
    ov_bad = []
    for m, cells in V1_T2.items():
        nd = sum(c[0] for c in cells.values()) / 4
        d = sum(c[1] for c in cells.values()) / 4
        sr = sum(c[2] for c in cells.values()) / 4
        pub = V1_OVERALL[m]
        if (round(nd), round(d), round(sr)) != pub:
            ov_bad.append({"method": m, "calc": (nd, d, sr), "pub": pub})
    v12["v1_overall_mismatch"] = ov_bad
    # Ours v1 대 v2 풀링
    def pool(cells):
        k_nd = sum(round(cnt(c[0], 20)) for c in cells.values())
        k_d = sum(round(cnt(c[1], 5)) for c in cells.values())
        return k_nd, k_d

    v1o = pool({t: V1_T2["Ours"][t][:2] for t in TASKS})
    v2o = pool(T1["Ours"])
    v12["ours_v1_pooled"] = v1o
    v12["ours_v2_pooled"] = v2o
    v12["ours_v1_vs_v2_fisher_nd"] = fisher(v1o[0], 80, v2o[0], 80)
    v12["ours_v1_vs_v2_fisher_total"] = fisher(sum(v1o), 100, sum(v2o), 100)
    v12["ours_per_task_v1_v2"] = {t: {"v1": V1_T2["Ours"][t][:2], "v2": T1["Ours"][t]} for t in TASKS}
    v12["table2_quality_changed"] = {m: {"v1": V1_T3[m], "v2": T2[m]} for m in V1_T3 if V1_T3[m] != T2[m]}
    # v1 Table IV 화이트보드 절제의 v2 Table 1 대조
    v12["v1_T4_vs_v2"] = {
        "v1_No7_full": V1_T4[7][:3], "v2_Ours_WW": T1["Ours"]["WW"],
        "v1_No4_L-A": V1_T4[4][:3], "v2_E2VLA_L-A_WW": T1["E2VLA-Sim+L-A"]["WW"],
        "v1_No5_M-A": V1_T4[5][:3], "v2_E2VLA_M-A_WW": T1["E2VLA-Sim+M-A"]["WW"],
        "v1_No6_H-A": V1_T4[6][:3], "v2_E2VLA_H-A_WW": T1["E2VLA-Sim+H-A"]["WW"],
    }
    R["V12_v1_vs_v2"] = v12

    # V13: Table 2 가능 범위 점검
    v13 = {}
    for m, (depth, ink) in T2.items():
        v13[m] = {"depth_le_25": depth <= 25.0, "depth_ge_10": depth >= 10.0, "ink_lt_5": ink < 5.0}
    R["V13_table2_ranges"] = v13

    # V14: 검출력 (n=20 대 n=20, 두 비율 차를 80 % 검출하는 데 필요한 시행 수)
    def n_needed(p1, p2, alpha=0.05, power=0.8):
        from scipy.stats import norm
        za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
        pbar = (p1 + p2) / 2
        num = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        return num / (p1 - p2) ** 2

    R["V14_n_needed"] = {
        "95 vs 71 (SR-ND 전체)": n_needed(0.95, 0.71),
        "95 vs 85 (과제별 MO 대 E2VLA-M-A 근사)": n_needed(0.95, 0.85),
        "95 vs 85 vs 사이": n_needed(0.95, 0.85),
        "90 vs 70 (SR-D 전체)": n_needed(0.90, 0.70),
        "85 vs 65 (Fig. 6 이웃 점)": n_needed(0.85, 0.65),
    }

    # V15: Ours 대 실데이터 기준선 (PH, WW의 SR-ND 20회)
    v15 = {}
    for m in ("Pi0-Real", "E2VLA-Real", "ForceVLA-Real", "ACP-Real"):
        for t in ("PH", "WW"):
            a = round(cnt(T1["Ours"][t][0], N_ND))
            b = round(cnt(T1[m][t][0], N_ND))
            v15[f"{m}_{t}"] = {"ours": a, "base": b, "p": fisher(a, 20, b, 20)}
    R["V15_vs_real_baselines_nd"] = v15

    OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False, default=lambda o: list(o) if isinstance(o, tuple) else str(o)))
    print("wrote", OUT)
    # 요약 출력
    print("V1 불일치:", [m for m, v in v1.items() if not (v["nd_match"] and v["d_match"])])
    print("V2 정수 위반:", bad)
    print("V3:", {m: (v["fail"], v["fig5_fail"]) for m, v in v3.items()})
    print("V5:", {k: v for k, v in v5.items() if k != "pooled_vs_sim_baselines"})
    for r in v5["pooled_vs_sim_baselines"]:
        print("  ", r["method"], r["k_nd"], r["k_d"], "p_nd=%.4g p_d=%.4g p_tot=%.4g" % (r["p_nd"], r["p_d"], r["p_total"]))
    print("V6:", [(x["task"], x["best_baseline"], x["k_best"], x["k_ours"], round(x["p"], 3), round(x["p_holm"], 3)) for x in v6])
    print("V6b:", [(x["task"], x["ours"], x["e2vla_m"], round(x["p"], 3)) for x in v6b])
    print("V7:", {k: (v["ci_nd_calc"], v["ci_d_calc"]) for k, v in v7.items()})
    print("V7b:", cmp5)
    print("V7c:", R["V7c_no1_vs_iso"])
    print("V8:", v8)
    print("V9:", t1_def)
    print("V10:", {k: round(v, 4) for k, v in v10.items()})
    print("V10b:", rng)
    print("V11:", R["V11_sr_d_gt_nd_cells"])
    print("V12:", json.dumps(v12, ensure_ascii=False, default=str)[:2500])
    print("V14:", R["V14_n_needed"])
    print("V15:", {k: round(v["p"], 4) for k, v in v15.items()})
    print("V4:", {m: (v["n_solutions"], v["pct_sum"]) for m, v in v4.items()})


if __name__ == "__main__":
    main()
