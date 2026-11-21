"""UniVTAC (arXiv:2602.10093 v1) 표 재계산과 검정.

논문의 보고 수치를 옮겨 적고 (1) 평균 재계산, (2) 과제별 Wilson 95% 구간,
(3) 과제별 Fisher 검정 + Holm 보정, (4) 과제 평균 차이의 Wald 구간과 z 검정,
(5) 표 사이 일관성 검사를 한다. 시행 수는 논문이 밝힌 값을 쓴다 (표 I–III 100회, 표 IV 20회).
학습 시드 반복이 보고되지 않아서 검정은 롤아웃 표집 변동만 반영한다.

실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)
출력: analysis/recompute_results.json 과 표준출력
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
TASKS = ["Lift Bottle", "Pull-out Key", "Lift Can", "Put Bottle in Shelf",
         "Insert Hole", "Insert HDMI", "Insert Tube", "Grasp Classify"]
N_SIM = 100  # 표 I–III: 과제당 100회 롤아웃 (논문 IV-A, 본문)
N_REAL = 20  # 표 IV: 조건당 20회 (논문 IV-D)

# 표 I (논문 Table I). 값은 % 단위.
T1 = {
    "ACT": [42, 28, 20, 28, 19, 15, 45, 50],
    "VITaL": [72, 47, 8, 32, 25, 6, 34, 100],
    "Ours": [71, 46, 29, 31, 24, 28, 56, 99],
}
T1_REPORTED_AVG = {"ACT": 30.9, "VITaL": 40.5, "Ours": 48.0}

# 표 II (논문 Table II). Full = 표 I의 Ours와 같아야 한다.
T2 = {
    "Scratch": [56, 40, 33, 34, 3, 16, 45, 45],
    "Contact": [59, 42, 12, 24, 32, 24, 43, 99],
    "Shape": [51, 35, 26, 25, 20, 25, 53, 98],
    "Contact+Shape": [62, 28, 18, 39, 21, 23, 60, 100],
    "Full": [71, 46, 29, 31, 24, 28, 56, 99],
}
T2_REPORTED_AVG = {"Scratch": 34.0, "Contact": 41.9, "Shape": 41.6, "Contact+Shape": 43.9, "Full": 48.0}

# 표 III (논문 Table III). 열: 0, 1K, 5K, 10K, 50K, 100K, 200K 샘플.
SCALE_COLS = ["0", "1K", "5K", "10K", "50K", "100K", "200K"]
T3 = {
    "Lift Bottle": [56, 45, 44, 51, 58, 59, 71],
    "Pull-out Key": [40, 34, 35, 46, 29, 37, 46],
    "Lift Can": [33, 20, 23, 26, 24, 20, 29],
    "Put Bottle in Shelf": [34, 34, 27, 36, 27, 40, 31],
    "Insert Hole": [3, 4, 6, 19, 32, 26, 24],
    "Insert HDMI": [16, 21, 21, 24, 38, 36, 28],
    "Insert Tube": [45, 48, 51, 51, 52, 36, 56],
    "Grasp Classify": [45, 100, 99, 100, 100, 100, 99],
}
T3_REPORTED_AVG = [34.2, 38.3, 38.3, 44.1, 45.0, 44.3, 48.0]
FIG4_LABELS = [34.0, 38.2, 38.2, 44.1, 45.0, 44.2, 48.0]  # 그림 4 점 위의 값

# 표 IV (논문 Table IV). 성공 횟수 = 비율 × 20 (정수인지 확인).
T4 = {
    "Insert Tube": (55, 85),
    "Insert USB": (15, 25),
    "Bottle Upright": (60, 95),
}
T4_REPORTED_AVG = (43.3, 68.3)


def wilson(x, n, z=1.959964):
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe_diff(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson(x1, n1)
    l2, u2 = wilson(x2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def pct_to_count(vals, n):
    counts = []
    for v in vals:
        c = round(v / 100 * n)
        assert abs(c / n * 100 - v) < 1e-9, (v, n)  # 정수 횟수인지 확인
        counts.append(c)
    return counts


def avg_diff_test(c1, c2, n1, n2):
    """과제 평균 차이 (과제마다 같은 가중치, 논문의 평균 정의). Wald SE는 과제별 이항 분산의 평균."""
    k = len(c1)
    d = [a / n1 - b / n2 for a, b in zip(c1, c2)]
    var = [(a / n1) * (1 - a / n1) / n1 + (b / n2) * (1 - b / n2) / n2 for a, b in zip(c1, c2)]
    est = sum(d) / k
    se = math.sqrt(sum(var)) / k
    z = est / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"diff_pp": 100 * est, "ci95_pp": [100 * (est - 1.96 * se), 100 * (est + 1.96 * se)],
            "z": z, "p_two_sided": p}


def pooled_naive(c1, c2, n1, n2):
    """과제를 무시하고 800회를 한 덩어리로 합친 검정 (참고용, 과제 내 상관을 무시해 과대평가될 수 있음)."""
    table = [[sum(c1), n1 * len(c1) - sum(c1)], [sum(c2), n2 * len(c2) - sum(c2)]]
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    return {"chi2": chi2, "p": p}


results = {"source": "arXiv:2602.10093 v1 (논문 Table I–IV 값을 옮겨 적음)", "checks": {}, "tables": {}}

# --- 0. 정수 횟수 확인 ---------------------------------------------------------
c1 = {k: pct_to_count(v, N_SIM) for k, v in T1.items()}
c2 = {k: pct_to_count(v, N_SIM) for k, v in T2.items()}
c3 = {k: pct_to_count(v, N_SIM) for k, v in T3.items()}
for k, (a, b) in T4.items():
    assert abs(round(a / 100 * N_REAL) / N_REAL * 100 - a) < 1e-9
    assert abs(round(b / 100 * N_REAL) / N_REAL * 100 - b) < 1e-9
results["checks"]["integer_counts"] = "모든 셀이 정수 횟수로 떨어짐 (표 I–III: 100회, 표 IV: 20회)"

# --- 1. 평균 재계산 -----------------------------------------------------------
rec = {}
for name, vals, rep in [("표 I", T1, T1_REPORTED_AVG), ("표 II", T2, T2_REPORTED_AVG)]:
    rec[name] = {k: {"recomputed": round(sum(v) / 8, 3), "reported": rep[k],
                     "match": round(sum(v) / 8, 1) == rep[k]} for k, v in vals.items()}
results["tables"]["average_recompute"] = rec
avg3 = [round(sum(T3[t][j] for t in TASKS) / 8, 3) for j in range(7)]
results["tables"]["table3_average"] = {
    "cols": SCALE_COLS, "recomputed": avg3, "reported_table": T3_REPORTED_AVG, "reported_fig4": FIG4_LABELS,
    "match_table": [round(a, 1) == b for a, b in zip(avg3, T3_REPORTED_AVG)],
}
t4_avg = (sum(a for a, _ in T4.values()) / 3, sum(b for _, b in T4.values()) / 3)
results["tables"]["table4_average"] = {"recomputed": [round(t4_avg[0], 2), round(t4_avg[1], 2)],
                                       "reported": list(T4_REPORTED_AVG)}

# --- 2. 표 I: 과제별 Wilson 구간, 방식별 쌍체가 아닌 독립 비교 -------------------
per_task_I = {}
for k in T1:
    per_task_I[k] = [{"task": t, "x": c1[k][i], "rate": c1[k][i] / N_SIM,
                      "wilson95": [100 * x for x in wilson(c1[k][i], N_SIM)]} for i, t in enumerate(TASKS)]
results["tables"]["table1_wilson"] = per_task_I


def compare_per_task(a, b, names, na=N_SIM, nb=N_SIM):
    rows, pv = [], []
    for i, t in enumerate(names):
        f = stats.fisher_exact([[a[i], na - a[i]], [b[i], nb - b[i]]])
        d, lo, hi = newcombe_diff(a[i], na, b[i], nb)
        rows.append({"task": t, "diff_pp": 100 * d, "newcombe95_pp": [100 * lo, 100 * hi], "p_fisher": f.pvalue})
        pv.append(f.pvalue)
    adj = holm(pv)
    for r, p in zip(rows, adj):
        r["p_holm"] = p
    return rows


results["tables"]["ours_vs_act_per_task"] = compare_per_task(c1["Ours"], c1["ACT"], TASKS)
results["tables"]["ours_vs_vital_per_task"] = compare_per_task(c1["Ours"], c1["VITaL"], TASKS)
results["tables"]["ours_vs_act_average"] = avg_diff_test(c1["Ours"], c1["ACT"], N_SIM, N_SIM)
results["tables"]["ours_vs_vital_average"] = avg_diff_test(c1["Ours"], c1["VITaL"], N_SIM, N_SIM)
results["tables"]["ours_vs_act_naive_pooled"] = pooled_naive(c1["Ours"], c1["ACT"], N_SIM, N_SIM)
results["tables"]["ours_vs_vital_naive_pooled"] = pooled_naive(c1["Ours"], c1["VITaL"], N_SIM, N_SIM)

# --- 3. 표 II: 절제 대조 -------------------------------------------------------
abl = {}
abl["contact_vs_scratch"] = avg_diff_test(c2["Contact"], c2["Scratch"], N_SIM, N_SIM)
abl["shape_vs_scratch"] = avg_diff_test(c2["Shape"], c2["Scratch"], N_SIM, N_SIM)
abl["full_vs_scratch"] = avg_diff_test(c2["Full"], c2["Scratch"], N_SIM, N_SIM)
abl["full_vs_contact_shape"] = avg_diff_test(c2["Full"], c2["Contact+Shape"], N_SIM, N_SIM)
abl["contact_shape_vs_scratch"] = avg_diff_test(c2["Contact+Shape"], c2["Scratch"], N_SIM, N_SIM)
results["tables"]["table2_ablation_average"] = abl
results["tables"]["full_vs_contact_shape_per_task"] = compare_per_task(c2["Full"], c2["Contact+Shape"], TASKS)

# --- 4. 표 III: 데이터 규모 -----------------------------------------------------
scale = {}
mono_steps = {}
for t in TASKS:
    v = T3[t]
    mono_steps[t] = {"down_steps": sum(1 for a, b in zip(v[:-1], v[1:]) if b < a),
                     "fraction_down_steps": sum(1 for a, b in zip(v[:-1], v[1:]) if b < a) / 6}
scale["monotone_steps_per_task"] = mono_steps
# 0 대 200K 과제별 Fisher (과제 8개 보정)
pv, rows = [], []
for t in TASKS:
    v = T3[t]
    x0, x200 = round(v[0]), round(v[-1])
    f = stats.fisher_exact([[x200, N_SIM - x200], [x0, N_SIM - x0]])
    rows.append({"task": t, "0": v[0], "200K": v[-1], "p_fisher": f.pvalue})
    pv.append(f.pvalue)
for r, p in zip(rows, holm(pv)):
    r["p_holm"] = p
scale["0_vs_200K_per_task"] = rows
# 평균 기울기: 1K–200K 여섯 점에 대한 log10(샘플 수)와 평균의 Spearman 상관
sizes = [1e3, 5e3, 1e4, 5e4, 1e5, 2e5]
rho, p_rho = stats.spearmanr(np.log10(sizes), avg3[1:])
scale["spearman_avg_1K_to_200K"] = {"rho": rho, "p": p_rho, "n_points": 6}
scale["avg_step_100K_to_50K_pp"] = avg3[5] - avg3[4]
results["tables"]["table3_scaling"] = scale

# --- 5. 표 IV: 실로봇 ----------------------------------------------------------
real_rows, pv = [], []
for t, (a_pct, b_pct) in T4.items():
    a, b = round(a_pct / 100 * N_REAL), round(b_pct / 100 * N_REAL)  # 성공 횟수 (20회 중)
    f = stats.fisher_exact([[b, N_REAL - b], [a, N_REAL - a]])
    d, lo, hi = newcombe_diff(b, N_REAL, a, N_REAL)
    real_rows.append({"task": t, "vision_x": a, "vision_plus_x": b,
                      "diff_pp": 100 * d, "newcombe95_pp": [100 * lo, 100 * hi], "p_fisher": f.pvalue,
                      "vision_wilson95": [100 * x for x in wilson(a, N_REAL)],
                      "plus_wilson95": [100 * x for x in wilson(b, N_REAL)]})
    pv.append(f.pvalue)
for r, p in zip(real_rows, holm(pv)):
    r["p_holm"] = p
xs_v = [round(a / 100 * 20) for a, _ in T4.values()]
xs_p = [round(b / 100 * 20) for _, b in T4.values()]
real_avg = avg_diff_test(xs_p, xs_v, N_REAL, N_REAL)
pooled_v, pooled_p = sum(xs_v), sum(xs_p)
f_pool = stats.fisher_exact([[pooled_p, 60 - pooled_p], [pooled_v, 60 - pooled_v]])
results["tables"]["table4_real"] = {"per_task": real_rows, "average_diff": real_avg,
                                    "pooled_60_vs_60_fisher_p": f_pool.pvalue,
                                    "pooled_counts": [pooled_v, pooled_p]}

# --- 6. 일관성 검사 ------------------------------------------------------------
checks = {
    "abstract_17.1_equals_t1_diff": round(T1_REPORTED_AVG["Ours"] - T1_REPORTED_AVG["ACT"], 1) == 17.1,
    "abstract_25_equals_t4_diff": round(t4_avg[1] - t4_avg[0], 1) == 25.0,
    "table2_full_equals_table1_ours": T2["Full"] == T1["Ours"],
    "table3_200K_equals_table1_ours": all(T3[t][-1] == T1["Ours"][i] for i, t in enumerate(TASKS)),
    "table3_0_equals_table2_scratch": all(T3[t][0] == T2["Scratch"][i] for i, t in enumerate(TASKS)),
    "table3_avg_0_recomputed_34.0_vs_reported_34.2": (round(avg3[0], 1), T3_REPORTED_AVG[0]),
    "table3_avg_1K_5K_recomputed_vs_fig4_label": (avg3[1], avg3[2], FIG4_LABELS[1], FIG4_LABELS[2]),
    "table3_avg_100K_recomputed_vs_fig4_label": (avg3[5], FIG4_LABELS[5]),
    "table3_monotone_claim_holds_for_average": all(b >= a for a, b in zip(avg3[:-1], avg3[1:])),
    "table3_monotone_claim_holds_for_100K_to_50K": avg3[5] >= avg3[4],
    "dataset_14000_x_14_shapes_vs_205826": (14000 * 14, 205826),
    "real_task_gain_insert_tube_pp": 85 - 55,
    "real_task_gain_insert_usb_pp": 25 - 15,
    "real_task_gain_bottle_upright_pp": 95 - 60,
    "text_contact_gain_7.9_equals_t2": round(T2_REPORTED_AVG["Contact"] - T2_REPORTED_AVG["Scratch"], 1),
    "text_shape_gain_7.6_equals_t2": round(T2_REPORTED_AVG["Shape"] - T2_REPORTED_AVG["Scratch"], 1),
}
results["checks"]["consistency"] = checks
results["checks"]["power_note"] = "과제당 n=100, 과제 평균 차이의 SE는 아래 avg_diff_test 결과 참고"

OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    def show(title, obj):
        print(f"\n== {title}")
        print(json.dumps(obj, ensure_ascii=False, default=float)[:1500])


    show("평균 재계산 (표 I)", rec["표 I"])
    show("표 III 평균", results["tables"]["table3_average"])
    show("Ours vs ACT 과제별", results["tables"]["ours_vs_act_per_task"])
    show("Ours vs VITaL 과제별", results["tables"]["ours_vs_vital_per_task"])
    show("Ours vs ACT 평균", results["tables"]["ours_vs_act_average"])
    show("Ours vs VITaL 평균", results["tables"]["ours_vs_vital_average"])
    show("Ours vs ACT 단순 합산(참고)", results["tables"]["ours_vs_act_naive_pooled"])
    show("표 II 절제 평균", abl)
    show("표 II Full vs C+S 과제별", results["tables"]["full_vs_contact_shape_per_task"])
    show("표 III 과제별 0 vs 200K", scale["0_vs_200K_per_task"])
    show("표 III 단조성", scale["monotone_steps_per_task"])
    show("표 III spearman", scale["spearman_avg_1K_to_200K"])
    show("표 IV", results["tables"]["table4_real"])
    show("일관성", checks)
