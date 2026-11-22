"""Fast-JEPA (arXiv:2611.25927v1) 표 수치 재계산과 검정.

입력: 논문 표 값 (Table 1–7, 본문 수치). 네트워크 접근 없음.
출력: analysis/recompute_results.json 과 표준출력 요약.
"""
import itertools
import json
import math
import os

from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- Table 1 (4-task, 평균 ± SD, %, ms) ----
T1 = {  # PushT, Reacher, Cube, TwoRoom, Avg, Latency
    "INTACT": ([87.33, 96.00, 100.00, 99.00], 95.58, 0.58, 26.19, 0.22),
    "INTACT Guarded A": ([91.33, 97.00, 99.67, 97.33], 96.33, 0.38, 127.09, 2.81),
    "Fast-JEPA": ([90.00, 96.33, 100.00, 99.67], 96.50, 0.66, 10.88, 0.04),
}
# Table 1 'Published results' 평균 검산용 (IMWM 행)
T1_PUB_IMWM = ([92.70, 83.80, 94.70, 99.20], 92.60)

# ---- Table 2 (DynaGoal-Shift) ----
T2 = {  # PushT, Reacher, Cube, TwoRoom, Avg, Avg-SD, Latency
    "INTACT": ([50.67, 73.00, 56.00, 45.33], 56.25, 1.64, 24.62),
    "INTACT Guarded A": ([36.33, 48.67, 15.00, 41.00], 35.25, 1.56, 115.43),
    "Fast-JEPA": ([57.33, 80.67, 72.00, 85.33], 73.83, 4.16, 13.42),
}
# Table 2 task-wise SD (per-task), Fast-JEPA/INTACT
T2_TASK_SD = {
    "INTACT": [2.08, 2.65, 1.73, 5.03],
    "INTACT Guarded A": [6.35, 3.06, 1.00, 4.36],
    "Fast-JEPA": [8.62, 4.16, 3.00, 5.13],
}

# ---- Table 3 (DynaGoal-Deadline) ----
T3 = {
    "INTACT": ([57.00, 82.00, 74.67, 87.67], 75.33, 1.18),
    "INTACT Guarded A": ([54.67, 60.33, 26.67, 93.33], 58.75, 0.90),
    "Fast-JEPA": ([72.33, 87.00, 93.33, 98.00], 87.67, 0.38),
}

# ---- Table 4 (ablation) ----
T4 = {
    "w/o refinement": ([84.00, 92.67, 100.00, 98.67], 93.83, 0.63, 5.53),
    "w/o proposal": ([45.00, 77.67, 56.00, 94.00], 68.17, 0.14, 5.27),
    "Fast-JEPA (Table 4 행)": ([89.00, 96.33, 100.00, 100.00], 96.33, 0.52, 10.78),
}

# ---- Table 5 ----
T5 = {  # Avg SR ± SD, scoring ms, latency ms
    "GC-IDM proposal": (94.42, 0.90, None, 132.91),
    "Oracle-state proposal": (95.77, 0.42, None, 781.34),
    "SC-augmented scoring": (92.13, 0.33, 5.85, None),
    "Recursive scoring": (96.08, 0.76, 11.16, 20.54),
    "GC-IDM + Recursive": (90.03, 0.61, 11.42, 158.30),
    "Fast-JEPA (Table 5 행)": (96.33, 0.52, 2.32, 10.78),
}

# ---- Table 6 (GPU-hours, 4 tasks total, mean ± SD over 3 seeds) ----
T6 = {"Fast-JEPA": (5.95, 0.30), "INTACT": (6.67, 0.37), "LeWM": (18.16, 1.06), "Fast-LeWM": (188.05, 3.08)}

# ---- Table 7 (DynaGoal latency, ms). 열별 4 Shift + 4 Deadline 행 ----
T7_cols = {
    "LeWM": [774.70, 754.96, 754.76, 785.61, 772.84, 755.95, 755.55, 785.66],
    "Fast-LeWM": [408.27, 416.60, 402.05, 390.79, 410.42, 419.05, 394.35, 392.74],
    "Fast-LeWM + SC": [619.61, 635.59, 621.46, 597.82, 623.45, 637.34, 636.20, 599.57],
    "GC-IDM*": [147.68, 174.83, 192.75, 166.42, 158.08, 166.42, 192.53, 151.45],
    "INTACT": [24.39, 23.88, 25.07, 25.14, 26.84, 23.66, 25.39, 23.58],
    "INTACT Guarded A": [123.60, 107.56, 113.97, 116.57, 112.33, 105.43, 117.30, 110.55],
    "Fast-JEPA": [13.58, 13.33, 13.32, 13.45, 13.96, 13.30, 14.02, 13.72],
}
T7_TABLE2_LAT = {"LeWM": 767.51, "Fast-LeWM": 404.43, "Fast-LeWM + SC": 618.62, "GC-IDM*": 170.42,
                 "INTACT": 24.62, "INTACT Guarded A": 115.43, "Fast-JEPA": 13.42}
T7_TABLE3_LAT = {"LeWM": 767.50, "Fast-LeWM": 404.14, "Fast-LeWM + SC": 624.14, "GC-IDM*": 167.12,
                 "INTACT": 24.87, "INTACT Guarded A": 111.40, "Fast-JEPA": 13.75}

# ---- Figure 4 (DynaGoal-Delay, delay 0 ms) 그림에서 읽은 근사값 (±3 pt) ----
FIG4_DELAY0 = {  # PushT, Reacher, Cube, TwoRoom
    "INTACT": [48, 81, 77, 47],
    "INTACT Guarded A": [30, 84, 96, 42],
    "Fast-JEPA": [71, 96, 93, 90],
}
T2_SR_TASKS = {"INTACT": T2["INTACT"][0], "INTACT Guarded A": T2["INTACT Guarded A"][0], "Fast-JEPA": T2["Fast-JEPA"][0]}


def mean(xs):
    return sum(xs) / len(xs)


def sample_sd(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def welch(m1, s1, n1, m2, s2, n2):
    v1, v2 = s1 ** 2 / n1, s2 ** 2 / n2
    t = (m1 - m2) / math.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    p = 2 * stats.t.sf(abs(t), df)
    return t, df, p


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def grim_triples(mean_rep, sd_rep, n=3, tol_mean=0.005, tol_sd=0.005):
    """3 seeds x 100 episodes: 정수 % 세 값으로 평균과 표본 SD가 보고값과 맞는 조합 존재 여부."""
    hits = []
    for a, b, c in itertools.combinations_with_replacement(range(101), n):
        xs = [a, b, c]
        if abs(mean(xs) - mean_rep) <= tol_mean and abs(sample_sd(xs) - sd_rep) <= tol_sd:
            hits.append(tuple(xs))
    return hits


def main():
    out = {}

    # 1) Table 1 평균 검산 (작업: 4개 과제 산술평균)
    t1 = {}
    for name, (tasks, avg, sd, lat, lsd) in T1.items():
        t1[name] = {"avg_recomputed": round(mean(tasks), 3), "avg_reported": avg}
    out["table1_avg_check"] = t1
    out["table1_speedup_INTACT_over_FastJEPA"] = round(26.19 / 10.88, 3)
    out["table1_speedup_GuardedA_over_FastJEPA"] = round(127.09 / 10.88, 2)
    out["table1_pub_IMWM_avg_check"] = round(mean(T1_PUB_IMWM[0]), 2)

    # 2) Table 2, 3 평균 검산
    out["table2_avg_check"] = {k: {"recomputed": round(mean(v[0]), 3), "reported": v[1]} for k, v in T2.items()}
    out["table3_avg_check"] = {k: {"recomputed": round(mean(v[0]), 3), "reported": v[1]} for k, v in T3.items()}

    # 3) Table 4 평균 검산, Table 1 과 Table 4/5 의 같은 컨트롤러 불일치
    out["table4_avg_check"] = {k: {"recomputed": round(mean(v[0]), 3), "reported": v[1]} for k, v in T4.items()}
    out["fastjepa_inconsistency"] = {
        "table1": {"PushT": 90.00, "TwoRoom": 99.67, "avg": 96.50, "latency_ms": 10.88},
        "table4_5": {"PushT": 89.00, "TwoRoom": 100.00, "avg": 96.33, "latency_ms": 10.78},
    }

    # 4) Table 5 비율 검산 (재귀 채점 대 직접 채점)
    out["scoring_ratio_recursive_over_direct"] = round(11.16 / 2.32, 2)
    out["scoring_ratio_expected_H5"] = 5
    out["latency_ratio_recursive_over_direct"] = round(20.54 / 10.78, 2)

    # 5) Table 7 평균 검산 (Table 2, 3 의 지연 열)
    t7 = {}
    for name, vals in T7_cols.items():
        shift = vals[:4]
        dead = vals[4:]
        t7[name] = {
            "shift_mean": round(mean(shift), 2), "table2_reported": T7_TABLE2_LAT[name],
            "deadline_mean": round(mean(dead), 2), "table3_reported": T7_TABLE3_LAT[name],
        }
    out["table7_latency_check"] = t7

    # 6) Figure 4 (delay 0) 대 Table 2 (Shift, 같은 프로토콜이어야 함)
    fig_cmp = {}
    tasks = ["PushT", "Reacher", "Cube", "TwoRoom"]
    for name in FIG4_DELAY0:
        fig_cmp[name] = {t: {"figure_delay0_approx": FIG4_DELAY0[name][i],
                             "table2_shift": T2[name][0][i],
                             "diff": round(FIG4_DELAY0[name][i] - T2[name][0][i], 1)}
                         for i, t in enumerate(tasks)}
    out["figure4_vs_table2_delay0"] = fig_cmp

    # 7) 검정 (Welch, n=3 seeds). ±는 표본 SD(3 시드)로 가정
    tests = {}
    def add(label, a, b):
        t, df, p = welch(a[0], a[1], 3, b[0], b[1], 3)
        tests[label] = {"mean_diff": round(a[0] - b[0], 2), "t": round(t, 2), "df": round(df, 2), "p_two_sided": round(p, 4)}

    add("Table1 avg: Fast-JEPA vs INTACT", (96.50, 0.66), (95.58, 0.58))
    add("Table1 avg: Fast-JEPA vs INTACT Guarded A", (96.50, 0.66), (96.33, 0.38))
    add("Table2 Shift avg: Fast-JEPA vs INTACT", (73.83, 4.16), (56.25, 1.64))
    add("Table2 Shift avg: Fast-JEPA vs INTACT Guarded A", (73.83, 4.16), (35.25, 1.56))
    add("Table3 Deadline avg: Fast-JEPA vs INTACT", (87.67, 0.38), (75.33, 1.18))
    add("Table4 refinement effect (w/o refinement vs Fast-JEPA row)", (96.33, 0.52), (93.83, 0.63))
    add("Table2 Shift PushT: Fast-JEPA vs INTACT", (57.33, 8.62), (50.67, 2.08))
    out["welch_tests_n3"] = tests

    # 8) 이항 기준 (pooled 300 trials per task) Wilson CI, Fisher exact 상한 검정
    pooled = {}
    for name in ["Fast-JEPA", "INTACT", "INTACT Guarded A"]:
        sr = T1[name][0] if name in T1 else None
    fj_sr = T1["Fast-JEPA"][0]
    ig_sr = T1["INTACT Guarded A"][0]
    pooled_rows = {}
    for i, t in enumerate(tasks):
        k_fj = round(fj_sr[i] * 3)  # 3 seeds x 100 episodes -> count out of 300
        k_ig = round(ig_sr[i] * 3)
        lo1, hi1 = wilson(k_fj, 300)
        lo2, hi2 = wilson(k_ig, 300)
        tbl = [[k_fj, 300 - k_fj], [k_ig, 300 - k_ig]]
        _, p = stats.fisher_exact(tbl)
        pooled_rows[t] = {"fastjepa_k": k_fj, "guardedA_k": k_ig,
                          "fastjepa_wilson95": [round(lo1 * 100, 1), round(hi1 * 100, 1)],
                          "guardedA_wilson95": [round(lo2 * 100, 1), round(hi2 * 100, 1)],
                          "fisher_p_pooled300": round(p, 3)}
    out["table1_pooled300_binomial"] = pooled_rows

    # 9) GRIM 형 정합성: 3 시드 x 100 에피소드 정수 조합이 평균·SD를 만드는가
    grim = {}
    for label, mean_rep, sd_rep in [
        ("T1 Fast-JEPA PushT", 90.00, 3.00), ("T1 Fast-JEPA Reacher", 96.33, 1.15),
        ("T1 Fast-JEPA Cube", 100.00, 0.00), ("T1 Fast-JEPA TwoRoom", 99.67, 0.58),
        ("T1 INTACT Guarded A PushT", 91.33, 1.53),
        ("T2 Fast-JEPA PushT-shift", 57.33, 8.62), ("T2 Fast-JEPA Reacher-shift", 80.67, 4.16),
        ("T2 Fast-JEPA Cube-shift", 72.00, 3.00), ("T2 Fast-JEPA TwoRoom-shift", 85.33, 5.13),
        ("T2 INTACT PushT-shift", 50.67, 2.08), ("T2 INTACT Guarded A Cube-shift", 15.00, 1.00),
        ("T4 w/o proposal PushT", 45.00, 3.46), ("T4 Fast-JEPA PushT", 89.00, 1.73),
    ]:
        hits = grim_triples(mean_rep, sd_rep)
        grim[label] = {"feasible_integer_triples_count": len(hits), "example": hits[:3]}
    out["grim_feasibility_3seeds"] = grim

    path = os.path.join(HERE, "recompute_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=1)[:9000])


if __name__ == "__main__":
    main()
