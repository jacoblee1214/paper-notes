"""논문 표 값(arXiv:2606.30749 v2, pdftotext로 옮김)을 다시 계산하고 검정한다.

실행: python3 analysis/recompute.py  -> analysis/recompute_results.json (numpy, scipy, 수 초)
표 값은 논문 본문에서 옮긴 것이다. 계산은 평균, 이항 신뢰구간(Wilson), 두 비율 차이의 Wald 구간,
Fisher 정확 검정, 예산 간 단조성, 표 7 합계까지 포함한다.
주의: 각 셀은 500(시뮬) 또는 18(실로봇) 에피소드의 비율이고, 시드 간 분산은 논문이 보고하지 않아 빠져 있다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

HERE = Path(__file__).resolve().parent
TASKS = ["SprayBot.", "Lighter", "Dispenser", "Pliers", "Stapler", "Pen"]
BUDGETS = [20, 50, 100]
N_SIM = 500

# Table 1: 시뮬 성공률(%). 논문 표 행을 과제 순서대로 (20, 50, 100 시연) 세 개씩 펼친 값
# 과제 순서: SprayBot., Lighter, Dispenser, Pliers, Stapler, Pen
FLAT_T1 = {
    "DP":         [0.4, 0.4, 0.8, 2.0, 2.4, 3.8, 0.6, 0.8, 1.8, 0.4, 1.8, 2.2, 0.0, 1.6, 2.2, 2.8, 3.6, 3.2],
    "DP3":        [2.4, 4.8, 21.0, 3.2, 8.0, 15.8, 1.8, 4.4, 14.2, 2.4, 7.6, 13.4, 2.4, 20.0, 39.2, 2.2, 5.6, 10.4],
    "DP3-PT":     [7.6, 15.2, 29.0, 11.2, 26.6, 35.2, 11.4, 5.8, 17.4, 10.4, 21.2, 23.2, 10.4, 38.8, 43.8, 9.2, 25.8, 29.6],
    "GraspXL-PT": [16.8, 44.0, 44.2, 7.4, 34.6, 36.8, 3.2, 42.0, 58.8, 15.8, 32.6, 38.8, 21.0, 43.2, 71.6, 13.6, 31.4, 42.0],
    "HP-Scratch": [43.6, 58.6, 57.0, 29.8, 27.8, 44.2, 43.8, 64.2, 73.8, 31.6, 50.0, 54.2, 34.2, 61.8, 70.8, 48.8, 31.0, 47.6],
    "Ours":       [59.2, 61.4, 68.4, 33.2, 46.4, 45.4, 54.0, 72.8, 78.6, 32.8, 58.0, 62.6, 49.2, 66.6, 77.2, 43.8, 56.4, 54.6],
}
# T1[method][budget] = 과제 6개의 값 리스트
T1 = {m: {n: [v[t * 3 + i] for t in range(6)] for i, n in enumerate(BUDGETS)} for m, v in FLAT_T1.items()}
REPORTED_AVG = {  # Table 1 Avg 열 (20, 50, 100)
    "DP": [1.0, 1.8, 2.3], "DP3": [2.4, 8.4, 19.1], "DP3-PT": [10.1, 22.3, 29.7],
    "GraspXL-PT": [13.0, 37.8, 48.7], "HP-Scratch": [38.7, 48.9, 58.0], "Ours": [45.4, 60.3, 64.6],
}
# Table 2 (UniDex-VLA 비교). 방법 행은 단계별 누적 성공률.
T2 = {
    "Contact": [98.8, 98.2, 100.0, 99.8, 96.6, 99.4], "Grasp": [58.2, 32.0, 55.8, 71.6, 61.0, 95.0],
    "Lift": [28.8, 3.6, 0.4, 44.4, 14.0, 65.6], "At Goal": [7.6, 0.6, 0.0, 4.4, 0.0, 12.2],
    "Articulate after Lift": [6.4, 0.0, 0.0, 2.8, 0.0, 9.2], "Full (UniDex-VLA)": [2.2, 0.0, 0.0, 0.0, 0.0, 0.6],
    "Full (Ours, 50-demo 열)": [61.4, 46.4, 72.8, 58.0, 66.6, 56.4],
}
REPORTED_T2_AVG = {"Contact": 98.8, "Grasp": 62.3, "Lift": 26.1, "At Goal": 4.1, "Articulate after Lift": 3.1,
                   "Full (UniDex-VLA)": 0.5, "Full (Ours, 50-demo 열)": 60.3}
# Table 3 (구조·데이터 절제, 50 demos, Dex1B만 100 demos)
T3 = {
    "Ours": [61.4, 46.4, 72.8, 58.0, 66.6, 56.4],
    "PT-215k": [56.7, 33.2, 65.5, 52.4, 66.4, 59.4],
    "PT w/o Appr.": [55.4, 42.7, 71.2, 44.4, 51.1, 52.2],
    "PT w/o Lift": [68.6, 40.3, 67.3, 55.6, 56.5, 53.3],
    "Dex1B (100 demos)": [1.9, 3.6, 0.0, 6.1, 2.8, 2.9],
}
# Table 4 (실로봇, 과제당 18회). 논문 표 행을 [과제 3개 x (잡기, 정렬, 작동)] 순서로 펼친 백분율을 횟수로 바꾼다
N_REAL = 18
FLAT_T4_PCT = {
    "DP3": [44.4, 5.6, 5.6, 27.8, 16.7, 5.6, 44.4, 38.9, 11.1],
    "HP-Scratch": [55.6, 44.4, 33.3, 61.1, 44.4, 22.2, 66.7, 38.9, 22.2],
    "Ours": [88.9, 72.2, 38.9, 88.9, 83.3, 38.9, 55.6, 44.4, 44.4],
}
STAGES = ["Grasp", "Align", "Trigger"]
T4 = {}
for m, v in FLAT_T4_PCT.items():
    counts = [round(x * N_REAL / 100) for x in v]
    assert all(abs(x * N_REAL / 100 - c) < 0.2 for x, c in zip(v, counts)), m  # 정수 횟수인지 확인
    T4[m] = {st: [counts[t * 3 + j] for t in range(3)] for j, st in enumerate(STAGES)}
REPORTED_T4_AVG = {"DP3": [38.9, 20.4, 7.4], "HP-Scratch": [61.1, 42.6, 25.9], "Ours": [77.8, 66.6, 40.7]}
REPORTED_T4_TRIGGER = {"DP3": 7.4, "HP-Scratch": 25.9, "Ours": 40.7}
# 표 7: 세대별 궤적 수 (33개 행 중 31개 grasp type)
T7 = {"1": 37003, "2": 15176, "3": 16453, "4": 4293, "5": 3783, "6": 2970, "7": 3323, "8": 2585, "9": 1428,
      "10": 459, "11": 23707, "12": 22407, "13": 18850, "14": 5186, "15": 11487, "16": 1511, "17": 12159,
      "18": 11590, "20": 5669, "22": 11458, "23": 2342, "24": 6382, "25": 6801, "26": 17516, "27": 423,
      "28": 24354, "29": 21628, "30": 20887, "31": 15018, "32": 17410, "33": 11419}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


def diff_ci_avg(a, b, n=N_SIM):
    """두 방법의 6개 과제 평균 차이와 95% 구간 (과제는 독립, 셀 분산만 반영)."""
    a = np.array(a) / 100
    b = np.array(b) / 100
    d = (a.mean() - b.mean())
    se = math.sqrt(np.sum(a * (1 - a) / n) + np.sum(b * (1 - b) / n)) / len(a)
    return round(100 * d, 2), [round(100 * (d - 1.96 * se), 2), round(100 * (d + 1.96 * se), 2)]


out = {}

# 1. Table 1 평균 재계산
avg_check = {}
for m, d in T1.items():
    avg_check[m] = {}
    for i, n in enumerate(BUDGETS):
        rec = float(np.mean(d[n]))
        avg_check[m][str(n)] = {"recomputed": round(rec, 3), "reported": REPORTED_AVG[m][i],
                                "abs_diff": round(abs(rec - REPORTED_AVG[m][i]), 3)}
out["table1_average_check"] = avg_check
out["table1_max_abs_diff"] = round(max(v["abs_diff"] for m in avg_check.values() for v in m.values()), 3)

# 2. Ours vs HP-Scratch, Ours vs DP3 (셀 단위, 과제 평균 차이와 구간)
out["ours_vs_hpscratch_avg"] = {str(n): {"diff_pp": diff_ci_avg(T1["Ours"][n], T1["HP-Scratch"][n])[0],
                                         "ci95_pp": diff_ci_avg(T1["Ours"][n], T1["HP-Scratch"][n])[1]}
                                for n in BUDGETS}
out["ours_vs_dp3_avg"] = {str(n): {"diff_pp": diff_ci_avg(T1["Ours"][n], T1["DP3"][n])[0],
                                   "ci95_pp": diff_ci_avg(T1["Ours"][n], T1["DP3"][n])[1]}
                          for n in BUDGETS}

# 3. 셀 단위 비교: Ours가 HP-Scratch보다 낮은 셀, 예산이 늘어도 떨어지는 셀
lower_than_hp = [(t, n) for n in BUDGETS for t, a, b in zip(TASKS, T1["Ours"][n], T1["HP-Scratch"][n]) if a < b]
out["cells_ours_below_hpscratch"] = [{"task": t, "demos": n} for t, n in lower_than_hp]
out["cells_ours_count_total"] = len(TASKS) * len(BUDGETS)
drops = []
for t_i, t in enumerate(TASKS):
    seq = [T1["Ours"][n][t_i] for n in BUDGETS]
    for j in range(1, len(BUDGETS)):
        if seq[j] < seq[j - 1]:
            drops.append({"task": t, "from": BUDGETS[j - 1], "to": BUDGETS[j], "delta_pp": round(seq[j] - seq[j - 1], 1)})
out["ours_nonmonotone_steps"] = drops
# 셀 단위 Ours vs GraspXL-PT
out["cells_ours_above_graspxl"] = sum(1 for n in BUDGETS for a, b in zip(T1["Ours"][n], T1["GraspXL-PT"][n]) if a > b)

# 4. Table 2 평균 재계산과 보고값 대조
t2 = {k: {"recomputed": round(float(np.mean(v)), 2), "reported": REPORTED_T2_AVG[k]} for k, v in T2.items()}
out["table2_average_check"] = t2

# 5. Table 3: 절제 차이와 구간 (Ours 50-demo 열 기준)
ablation = {}
for k in ["PT-215k", "PT w/o Appr.", "PT w/o Lift"]:
    d, ci = diff_ci_avg(T3["Ours"], T3[k])
    ablation[k] = {"avg": round(float(np.mean(T3[k])), 2), "diff_pp": d, "ci95_pp": ci}
ablation["Ours"] = {"avg": round(float(np.mean(T3["Ours"])), 2)}
ablation["Dex1B (100 demos)"] = {"avg": round(float(np.mean(T3["Dex1B (100 demos)"])), 2)}
out["table3"] = ablation

# 6. Table 4 실로봇: 단계별 풀링 비율, Wilson 구간, 검정
real = {}
for m, st in T4.items():
    real[m] = {}
    for s, counts in st.items():
        k, n = sum(counts), N_REAL * len(counts)
        real[m][s] = {"k": k, "n": n, "pct": round(100 * k / n, 2), "wilson95": wilson(k, n)}
out["table4_pooled"] = real
out["table4_avg_check"] = {m: REPORTED_T4_TRIGGER[m] for m in T4}


def two_prop(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    _, p = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
    return {"diff_pp": round(100 * d, 2), "wald95_pp": [round(100 * (d - 1.96 * se), 2), round(100 * (d + 1.96 * se), 2)],
            "fisher_two_sided_p": float(f"{p:.3g}")}


trig = {m: (sum(T4[m]["Trigger"]), 3 * N_REAL) for m in T4}
out["table4_trigger_tests"] = {
    "ours_vs_dp3": two_prop(*trig["Ours"], *trig["DP3"]),
    "ours_vs_hpscratch": two_prop(*trig["Ours"], *trig["HP-Scratch"]),
}
# 과제별 Ours vs HP-Scratch (Trigger 단계)
out["table4_per_task_trigger"] = {
    "syringe": {"ours": T4["Ours"]["Trigger"][0], "hp": T4["HP-Scratch"]["Trigger"][0]},
    "spray": {"ours": T4["Ours"]["Trigger"][1], "hp": T4["HP-Scratch"]["Trigger"][1]},
    "scissors": {"ours": T4["Ours"]["Trigger"][2], "hp": T4["HP-Scratch"]["Trigger"][2]},
}
stage_cells = [(s, i) for s in ["Grasp", "Align", "Trigger"] for i in range(3)]
out["table4_ours_beats_hp_cells"] = sum(1 for s, i in stage_cells if T4["Ours"][s][i] > T4["HP-Scratch"][s][i])
out["table4_ours_cells_total"] = len(stage_cells)

# 7. 표 7 합계
out["table7"] = {"n_grasp_types_listed": len(T7), "sum": sum(T7.values()), "paper_total_row": 355677,
                 "missing_type_ids": sorted(set(range(1, 34)) - {int(k) for k in T7})}

# 8. 실로봇 필요 시도 수: 차이 20pp를 80% 검정력으로 보려면 (양측 5%, 비율 0.4 vs 0.6 가정 시)
p1, p2 = 0.6, 0.4
za, zb = 1.96, 0.8416
n_per_arm = ((za * math.sqrt(2 * 0.5 * 0.5) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / (p1 - p2)) ** 2
out["power_n_per_arm_for_20pp"] = math.ceil(n_per_arm)

# 9. 오라클 부목표 비교 (부록 D.3 표 10): 오라클은 시연의 정답 부목표를 쓴다
FLAT_ORACLE = [40.4, 66.0, 71.4, 26.0, 41.0, 48.4, 32.0, 51.0, 49.0, 26.4, 38.4, 46.0, 51.4, 66.0, 74.0, 45.4, 62.4, 58.0]
ORACLE = {n: [FLAT_ORACLE[t * 3 + i] for t in range(6)] for i, n in enumerate(BUDGETS)}
out["table10_oracle"] = {
    "oracle_avg_recomputed": {str(n): round(float(np.mean(ORACLE[n])), 2) for n in BUDGETS},
    "oracle_avg_reported": {"20": 36.9, "50": 54.1, "100": 57.8},
    "ours_minus_oracle_cells_positive": sum(1 for n in BUDGETS for a, b in zip(T1["Ours"][n], ORACLE[n]) if a > b),
    "ours_minus_oracle_cells_total": len(TASKS) * len(BUDGETS),
    "ours_minus_oracle_avg_pp": {str(n): round(float(np.mean(T1["Ours"][n]) - np.mean(ORACLE[n])), 2) for n in BUDGETS},
}

if __name__ == "__main__":
    (HERE / "recompute_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", HERE / "recompute_results.json")
