"""FACTR 2 (arXiv:2606.12406v2) 표 재계산, 정밀도 점검, 검정.

입력은 논문 본문 표 값을 손으로 옮긴 것이다 (paper/paper.txt 기준).
출력: analysis/table_check_results.json
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("table_check_results.json")
res = {}

# ---------- Table 1 (Franka, L1 joint torque error, Nm) ----------
t1 = {
    "contact": {"FILIC": (4.395, 1.531), "DO": (1.471, 0.761), "NEXT": (0.547, 0.348)},
    "free": {"sensor": (0.449, 0.208), "FILIC": (2.460, 1.358), "DO": (2.429, 1.298), "NEXT": (0.414, 0.278)},
}
res["t1_reduction_pct"] = {
    "contact_vs_FILIC": round(100 * (1 - 0.547 / 4.395), 2),
    "contact_vs_DO": round(100 * (1 - 0.547 / 1.471), 2),
    "paper_says": {"contact_vs_FILIC": 87.6, "contact_vs_DO": 62.8},
}


def welch(m1, s1, n1, m2, s2, n2):
    t, p = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
    return float(t), float(p)


# 단위(n)가 논문에 없다. 가정 두 가지를 놓고 민감도를 본다.
#  n=6: "±"가 여섯 관절 사이 표준편차라고 가정
#  n=1: 한 번의 보류 세션(시험 단위가 하나)이라고 보면 검정 자체가 불가
res["t1_welch_n6"] = {
    "contact_NEXT_vs_DO": welch(0.547, 0.348, 6, 1.471, 0.761, 6),
    "free_NEXT_vs_sensor": welch(0.414, 0.278, 6, 0.449, 0.208, 6),
}
# 자유 공간에서 NEXT가 센서보다 낮다는 주장: 차이 0.035, 표준편차 0.2–0.3
# 시간 샘플 기준(5분, 100 Hz, n=30000)으로 계산하면 자기상관 때문에 과장된다.
res["t1_free_diff"] = {
    "diff": round(0.449 - 0.414, 3),
    "t_if_n_30000_naive": welch(0.414, 0.278, 30000, 0.449, 0.208, 30000)[0],
    "note": "n=30000은 자기상관을 무시한 값이라 근거가 되지 못함",
}

# ---------- Table 5 (Piper free-space, Nm) ----------
res["t5_welch_n6"] = {
    "NEXT_vs_DO": welch(0.018, 0.012, 6, 0.248, 0.132, 6),
    "NEXT_vs_FILIC": welch(0.018, 0.012, 6, 0.435, 0.225, 6),
    "reduction_pct_vs_DO": round(100 * (1 - 0.018 / 0.248), 1),
}

# ---------- Table 4 (LSTM/GRU/MLP ablation) : 'Δq 입력이 일관되게 낫다' 점검 ----------
t4 = {
    "MLP": {"q": [(2.35, 1.78), (2.02, 1.34), (1.76, 1.13)], "qqd": [(1.04, 0.681), (0.980, 0.908), (1.02, 0.905)],
            "qqdD": [(0.706, 1.89), (0.755, 1.56), (0.813, 1.47)]},
    "GRU": {"q": [(2.25, 1.12), (1.68, 0.665), (1.62, 0.843)], "qqd": [(0.863, 0.580), (0.814, 0.596), (0.795, 0.653)],
            "qqdD": [(0.423, 1.94), (0.473, 1.47), (0.429, 1.62)]},
    "LSTM": {"q": [(2.53, 2.61), (2.21, 1.45), (2.15, 1.09)], "qqd": [(1.15, 0.715), (1.11, 0.551), (1.12, 0.578)],
             "qqdD": [(0.504, 0.876), (0.490, 0.789), (0.414, 0.547)]},
}
res["t4_qqdD_best_in_column"] = {}
for fam, rows in t4.items():
    # H=10, 25, 50 중 Δq 입력이 q,q̇ 입력보다 자유/접촉 각각에서 낮은가
    res["t4_qqdD_best_in_column"][fam] = {
        "free_qqdD_lt_qqd_all_H": all(rows["qqdD"][i][0] < rows["qqd"][i][0] for i in range(3)),
        "contact_qqdD_lt_qqd_all_H": all(rows["qqdD"][i][1] < rows["qqd"][i][1] for i in range(3)),
    }
# 논문 표 4 캡션은 Δq 입력이 '일관되게' 낫다고 한다. 위 결과가 그 판정이다.
res["t4_lstm_H50_equals_table1"] = {"free": 0.414, "contact": 0.547,
                                    "same_as_table1_NEXT_free_and_contact": True}

# ---------- Table 2 (task progress, FIRST ablation) ----------
t2 = {  # 과제: (stages, C, PC, PC+C)
    "LEGO": (6, 0.697, 0.913, 0.875),
    "NIST Belt": (7, 0.494, 0.767, 0.649),
    "NIST Insertion": (4, 0.800, 0.933, 0.933),
    "Tool Clean Up": (4, 0.650, 0.805, 0.775),
    "Cap Screwing": (4, 0.707, 0.673, 0.825),
}
avg = {k: round(np.mean([v[i] for v in t2.values()]), 4) for i, k in enumerate(["C", "PC", "PC+C"], start=1)}
res["t2_average_recomputed"] = avg
res["t2_average_paper"] = {"C": 0.670, "PC": 0.818, "PC+C": 0.811}
# 짝비교 (n=5 과제): PC - C, PC+C - C
d_pc = [v[2] - v[1] for v in t2.values()]
d_pcc = [v[3] - v[1] for v in t2.values()]
res["t2_paired_PC_minus_C"] = {"mean_diff": round(float(np.mean(d_pc)), 4),
                               "t_p": [round(float(x), 4) for x in stats.ttest_rel([v[2] for v in t2.values()], [v[1] for v in t2.values()])],
                               "per_task": [round(x, 3) for x in d_pc]}

# 정밀도: 진행도는 (성공 단계 수)/(20 롤아웃 × 단계 수)여야 한다 (C.2)
def fits(v, d):
    return any(round(k / d, 3) == round(v, 3) for k in range(d + 1))


prog_fit = {}
for name, (S, c, pc, pcc) in t2.items():
    d = 20 * S
    prog_fit[name] = {f"{x}": fits(x, d) for x in (c, pc, pcc)}
res["t2_granularity_20xS"] = prog_fit
res["t2_fraction_fitting_20xS"] = round(sum(sum(v.values()) for v in prog_fit.values()) / 15, 3)

# ---------- Table 9 (성공률, 20 롤아웃 가정) ----------
t9 = {  # (policy, cond, task): values for None, PC, PC+C
    ("ACT", "w/o tau", "Belt"): (0.220, 0.360, 0.283), ("ACT", "w/o tau", "Tool"): (0.600, 0.600, 0.763),
    ("ACT", "w/ tau", "Belt"): (0.347, 0.687, 0.770), ("ACT", "w/ tau", "Tool"): (0.650, 0.650, 0.788),
    ("FM", "w/o tau", "Belt"): (0.150, 0.213, 0.241), ("FM", "w/o tau", "Tool"): (0.463, 0.575, 0.525),
    ("FM", "w/ tau", "Belt"): (0.431, 0.767, 0.649), ("FM", "w/ tau", "Tool"): (0.525, 0.805, 0.775),
}
vals9 = [x for v in t9.values() for x in v]
res["t9_fits_n20"] = sum(fits(x, 20) for x in vals9)
res["t9_n_values"] = len(vals9)
# 1–200 중 한 분모로 표 전체가 맞는지
common = [d for d in range(2, 201) if all(fits(x, d) for x in vals9)]
res["t9_common_denominators_le200"] = common[:10]
# 표 2의 PC, PC+C 값이 표 9의 FM w/τ 성공률과 같은지 (지표가 같은가, 복사 오류인가)
t2_pc = {"Belt": (0.767, 0.649), "Tool": (0.805, 0.775)}
res["t2_vs_t9_FMwtau_identical"] = {
    "Belt_PC_PCC": [t2_pc["Belt"] == t9[("FM", "w/ tau", "Belt")][1:]],
    "Tool_PC_PCC": [t2_pc["Tool"] == t9[("FM", "w/ tau", "Tool")][1:]],
}

# 가정 n=20에서 주요 대비 (Fisher exact, 반올림한 성공 수)
def fisher(v1, v2, n=20):
    a, b = round(v1 * n), round(v2 * n)
    _, p = stats.fisher_exact([[a, n - a], [b, n - b]])
    return {"k": [a, b], "n": n, "p": round(float(p), 4)}


res["t9_fisher_n20"] = {
    "ACT_wtau_Belt_None_vs_PC": fisher(0.347, 0.687),
    "ACT_wtau_Belt_None_vs_PCC": fisher(0.347, 0.770),
    "ACT_wotau_Belt_None_vs_PC": fisher(0.220, 0.360),
    "FM_wtau_Belt_None_vs_PC": fisher(0.431, 0.767),
    "FM_wotau_Tool_None_vs_PC": fisher(0.463, 0.575),
}
# 표 10 (ACT, NIST Insertion / Cap Screwing)
res["t10_fisher_n20"] = {
    "Insertion_base_vs_tau": fisher(0.220, 0.567),
    "Insertion_current_vs_tau": fisher(0.431, 0.567),
    "Cap_base_vs_tau": fisher(0.440, 0.525),
}
res["t10_granularity_n20"] = {"0.567": fits(0.567, 20), "0.431": fits(0.431, 20), "0.525": fits(0.525, 20), "0.440": fits(0.440, 20)}

# ---------- Table 11 (LEGO, ACT, task progress) ----------
t11 = {"None": 0.453, "WR_PC": 0.345, "WR_PCC": 0.478, "FIRST_PC": 0.573, "FIRST_PCC": 0.501}
res["t11_deltas_vs_none"] = {k: round(v - t11["None"], 3) for k, v in t11.items()}

# ---------- Wilson 구간 (n=20 가정) ----------
def wilson(p, n=20, z=1.96):
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(c - h, 3), round(c + h, 3)


res["wilson_n20_examples"] = {"0.767": wilson(0.767), "0.431": wilson(0.431), "0.647_placeholder": wilson(0.65)}

# ---------- Table 1 contact vs Table 4 contact (같은 숫자) ----------
res["table1_contact_equals_table4_LSTM_H50_contact"] = True

OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o)))
print(json.dumps(res, indent=1, ensure_ascii=False, default=str)[:6000])
