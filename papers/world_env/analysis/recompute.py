"""World-Env (arXiv:2509.24948 v6) 표 수치 재계산과 검정.

실행: python3 analysis/recompute.py  →  analysis/recompute_results.json
CPU, numpy/scipy만 사용. 수 초 안에 끝난다.

가정 (논문에 명시된 것과 추정을 구분):
- LIBERO 스위트당 평가 시행 n = 500 (논문 5.: "10 tasks with 50 ... for testing per task" → 10 × 50).
- 실세계(표 3)의 시행 수는 논문에 없다. 10과 100을 민감도 시나리오로 둔다.
- 표 7의 시행 수 n = 40은 논문 수치로부터 역산한 값이다 (정수 혼동행렬 검사로 확인).
- 시드 분산은 반영하지 않는다. 논문은 시드 수와 실행 간 분산을 보고하지 않는다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
N_SUITE = 500  # 스위트당 평가 시행 (논문 §5 Benchmarks 기준)
SUITES = ["Goal", "Object", "Spatial", "Long"]
RES = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def two_prop_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return z, 2 * stats.norm.sf(abs(z))


def fisher(k1, n1, k2, n2):
    t = [[k1, n1 - k1], [k2, n2 - k2]]
    return stats.fisher_exact(t)[1]


def pct_to_k(pct, n):
    k = pct / 100 * n
    assert abs(k - round(k)) < 1e-6, (pct, n)
    return int(round(k))


# ---------------- 표 1: 5 demonstrations, LIBERO 4 스위트 ----------------
T1 = {
    "pi0": [67.6, 68.4, 80.2, 28.2],
    "pi0+FAST": [59.2, 76.8, 59.2, 24.8],
    "OpenVLA": [73.2, 55.0, 82.4, 32.2],
    "UniVLA": [82.0, 76.2, 84.4, 56.4],
    "OFT": [84.0, 74.2, 84.2, 57.0],
    "Ours": [86.4, 86.6, 87.6, 57.8],
}
T1_AVG_REPORTED = {"pi0": 61.1, "pi0+FAST": 55.0, "OpenVLA": 60.7, "UniVLA": 74.75, "OFT": 74.85, "Ours": 79.6}
avg_check = {k: round(float(np.mean(v)), 3) for k, v in T1.items()}
RES["table1_average_check"] = {
    k: {"reported": T1_AVG_REPORTED[k], "recomputed": avg_check[k], "match": abs(avg_check[k] - T1_AVG_REPORTED[k]) < 0.01}
    for k in T1
}

rows = []
for i, s in enumerate(SUITES):
    k_ours = pct_to_k(T1["Ours"][i], N_SUITE)
    k_oft = pct_to_k(T1["OFT"][i], N_SUITE)
    z, p = two_prop_z(k_ours, N_SUITE, k_oft, N_SUITE)
    rows.append({
        "suite": s,
        "ours": T1["Ours"][i], "ours_wilson95": [round(100 * x, 1) for x in wilson(k_ours, N_SUITE)],
        "oft": T1["OFT"][i], "diff_pp": round(T1["Ours"][i] - T1["OFT"][i], 2),
        "z": round(z, 2), "p_two_sided": float(f"{p:.2e}"),
    })
RES["table1_ours_vs_oft_per_suite"] = rows

# 4 스위트 평균 차이의 근사 검정 (스위트 독립 가정, 평균의 분산 = 합/16)
ours_p = np.array(T1["Ours"]) / 100
oft_p = np.array(T1["OFT"]) / 100
var_ours = ours_p * (1 - ours_p) / N_SUITE
var_oft = oft_p * (1 - oft_p) / N_SUITE
diff_avg = (ours_p.mean() - oft_p.mean()) * 100
se_avg = math.sqrt((var_ours.sum() + var_oft.sum()) / 16) * 100
RES["table1_avg_diff_OFT"] = {
    "diff_pp": round(diff_avg, 2), "se_pp": round(se_avg, 2),
    "z": round(diff_avg / se_avg, 2), "p_two_sided": float(f"{2 * stats.norm.sf(abs(diff_avg / se_avg)):.2e}"),
    "note": "스위트 독립, 시행 샘플링만 반영. 학습 시드 분산 미반영",
}
# 동일 방식으로 UniVLA, RIPT-VLA 등 다른 비교
RIPT = [86.2, 83.4, 88.6, 58.4]  # 표 2
rip_rows = []
for i, s in enumerate(SUITES):
    k_o = pct_to_k(T1["Ours"][i], N_SUITE)
    k_r = pct_to_k(RIPT[i], N_SUITE)
    z, p = two_prop_z(k_o, N_SUITE, k_r, N_SUITE)
    rip_rows.append({"suite": s, "ours": T1["Ours"][i], "ript": RIPT[i],
                     "diff_pp": round(T1["Ours"][i] - RIPT[i], 2), "p_two_sided": float(f"{p:.3f}")})
RES["table2_ours_vs_ript_per_suite"] = rip_rows
RES["table2_avg"] = {"ours": round(np.mean(T1["Ours"]), 2), "ript": round(np.mean(RIPT), 3)}

# ---------------- 표 4: 종료 신호 없이 최대 스텝까지 ----------------
T4 = {
    "pi0": [55.4, 71.0, 72.6, 20.6],
    "pi0+FAST": [21.2, 74.0, 44.8, 15.0],
    "OpenVLA": [68.4, 47.4, 59.8, 26.6],
    "UniVLA": [72.0, 75.2, 66.4, 48.0],
    "OFT": [67.4, 73.8, 71.2, 39.8],
    "Ours": [85.0, 78.4, 78.4, 57.8],
}
T4_AVG_REPORTED = {"pi0": 54.9, "pi0+FAST": 38.75, "OpenVLA": 50.55, "UniVLA": 65.4, "OFT": 63.05, "Ours": 74.9}
RES["table4_average_check"] = {
    k: {"reported": T4_AVG_REPORTED[k], "recomputed": round(float(np.mean(v)), 3),
        "match": abs(np.mean(v) - T4_AVG_REPORTED[k]) < 0.01} for k, v in T4.items()
}
RES["table4_shift_vs_table1"] = {
    "OFT_avg_T1_to_T4": [74.85, 63.05, round(63.05 - 74.85, 2)],
    "Ours_avg_T1_to_T4": [79.6, 74.9, round(74.9 - 79.6, 2)],
    "note": "표 4의 두 방법 수치는 표 1과 설정이 다른 것으로 보인다. 논문은 어떤 종료 규칙이 표 1에 쓰였는지 밝히지 않음",
}
# 표 4 내 Ours 대 OFT 차이 (두 열 모두 n=500 가정)
d4 = []
for i, s in enumerate(SUITES):
    k_o = pct_to_k(T4["Ours"][i], N_SUITE)
    k_f = pct_to_k(T4["OFT"][i], N_SUITE)
    z, p = two_prop_z(k_o, N_SUITE, k_f, N_SUITE)
    d4.append({"suite": s, "diff_pp": round(T4["Ours"][i] - T4["OFT"][i], 2), "p_two_sided": float(f"{p:.2e}")})
RES["table4_ours_vs_oft_per_suite"] = d4

# ---------------- 표 5: 절제 (2x2) ----------------
T5 = {
    "no_extra_no_head": [68.4, 75.2, 73.2, 42.2],
    "extra_no_head": [79.8, 81.8, 78.4, 44.6],
    "no_extra_head": [68.8, 76.4, 74.4, 43.8],
    "extra_head_(Ours)": [86.4, 86.6, 87.6, 57.8],
}
avg5 = {k: round(float(np.mean(v)), 3) for k, v in T5.items()}
RES["table5_avg"] = avg5
RES["table5_effects_avg_pp"] = {
    "extra_data_effect_without_head": round(avg5["extra_no_head"] - avg5["no_extra_no_head"], 2),
    "extra_data_effect_with_head": round(avg5["extra_head_(Ours)"] - avg5["no_extra_head"], 2),
    "reward_head_effect_without_extra": round(avg5["no_extra_head"] - avg5["no_extra_no_head"], 2),
    "reward_head_effect_with_extra": round(avg5["extra_head_(Ours)"] - avg5["extra_no_head"], 2),
    "interaction": round((avg5["extra_head_(Ours)"] - avg5["extra_no_head"]) - (avg5["no_extra_head"] - avg5["no_extra_no_head"]), 2),
}
# 표 5의 "둘 다 없음" 행(64.75)은 표 1의 OFT SFT(74.85)보다 낮다. 어떤 기준 행인지 논문에 없다.
RES["table5_neither_vs_T1_OFT_pp"] = round(avg5["no_extra_no_head"] - 74.85, 2)
RES["table5_ours_equals_T1_ours"] = all(
    abs(a - b) < 1e-9 for a, b in zip(T5["extra_head_(Ours)"], T1["Ours"]))

# ---------------- 표 6 (Octo), 표 8 (교란) ----------------
octo = {"Octo": [84.6, 85.7, 78.9, 51.1], "Ours": T1["Ours"]}
RES["table6_octo_diff_pp"] = {s: round(octo["Ours"][i] - octo["Octo"][i], 1) for i, s in enumerate(SUITES)}
# 표 8: Spatial만, 단일 수치. n=500 가정
k_g, k_o = pct_to_k(85.4, N_SUITE), pct_to_k(87.6, N_SUITE)
z, p = two_prop_z(k_o, N_SUITE, k_g, N_SUITE)
RES["table8_gaussian_vs_orig"] = {"diff_pp": round(87.6 - 85.4, 1), "z": round(z, 2), "p_two_sided": float(f"{p:.3f}")}
k_c = pct_to_k(87.0, N_SUITE)
z, p = two_prop_z(k_o, N_SUITE, k_c, N_SUITE)
RES["table8_color_vs_orig"] = {"diff_pp": round(87.6 - 87.0, 1), "z": round(z, 2), "p_two_sided": float(f"{p:.3f}")}

# ---------------- 표 7: 반영기(reflector) 혼동행렬 역산 ----------------
# 보고 값: (η, accuracy, precision, recall, F1)
T7 = [
    (0.2, 0.825, 0.667, 1.0, 0.8),
    (0.4, 0.875, 0.764, 0.928, 0.838),
    (0.5, 0.925, 0.867, 0.928, 0.896),
    (0.6, 0.875, 0.846, 0.785, 0.815),
    (0.8, 0.85, 0.9, 0.642, 0.75),
]
# 가정: 양성 P, 음성 N, 총 n. 혼동행렬 TP, FP, FN, TN은 정수.
best = None
t7_rows = []
for n in range(20, 201):
    for P in range(1, n):
        N = n - P
        ok = True
        details = []
        for eta, acc, pr, rc, f1 in T7:
            found = None
            for TP in range(0, P + 1):
                FN = P - TP
                if abs(TP / P - rc) > 0.001:
                    continue
                for FP in range(0, N + 1):
                    if TP + FP == 0:
                        continue
                    if abs(TP / (TP + FP) - pr) > 0.001:
                        continue
                    TN = N - FP
                    if abs((TP + TN) / n - acc) > 0.001:
                        continue
                    f1c = 2 * pr * rc / (pr + rc)
                    if abs(f1c - f1) > 0.0015:
                        continue
                    found = (TP, FP, FN, TN)
                    break
                if found:
                    break
            if not found:
                ok = False
                break
            details.append((eta, found))
        if ok:
            t7_rows.append({"n": n, "positives": P, "negatives": N, "per_eta": details})
            if best is None:
                best = (n, P, N, details)
RES["table7_reconstruction"] = {
    "consistent_solutions_found": len(t7_rows),
    "smallest_n_solution": t7_rows[0] if t7_rows else None,
    "note": "n=40, 양성 14, 음성 26이 모든 η 행과 맞는 최소 해. 논문은 시행 수를 밝히지 않음",
}
# n=40 가정으로 η=0.5 정확도의 Wilson 구간
RES["table7_acc_at_eta0.5_wilson95"] = [round(x, 3) for x in wilson(37, 40)]

# ---------------- 표 3: 실세계 (시행 수 미보고) ----------------
T3 = {
    "Clean table": {"OFT": 20, "Ours": 30},
    "Put green toy": {"OFT": 30, "Ours": 50},
    "Put red toy": {"OFT": 30, "Ours": 40},
    "Put orange toy": {"OFT": 20, "Ours": 50},
}
RES["table3_real_world"] = {}
for n_trial in (10, 100):
    kO = sum(pct_to_k(v["OFT"], n_trial) for v in T3.values())
    kU = sum(pct_to_k(v["Ours"], n_trial) for v in T3.values())
    N_tot = 4 * n_trial
    RES["table3_real_world"][f"n_per_task_{n_trial}"] = {
        "oft_pooled": [kO, N_tot], "ours_pooled": [kU, N_tot],
        "fisher_p": float(f"{fisher(kU, N_tot, kO, N_tot):.3g}"),
    }
RES["table3_real_world"]["note"] = "시행 수 미보고. 10과 100은 민감도 시나리오. 과제 4개 합산 Fisher, 과제 층화 안 함"

# ---------------- 실행 ----------------
OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(json.dumps({k: RES[k] for k in ["table1_avg_diff_OFT", "table5_effects_avg_pp",
                                       "table5_neither_vs_T1_OFT_pp"]}, ensure_ascii=False, indent=2))
print("table7 consistent n:", sorted({r['n'] for r in t7_rows})[:5], "solutions:", len(t7_rows))
print("written", OUT)
