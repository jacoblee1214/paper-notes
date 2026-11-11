"""ULTRA(arXiv:2603.03279v3) 표 재계산과 유의성 검정.

입력: 논문 표의 값을 직접 옮긴 것(원문 PDF 6–7쪽을 렌더해 확인, 표 I·II·III·IV).
출력: analysis/table_check_results.json, 표준출력 요약.
실행: python3 analysis/table_check.py   (CPU, 1초 미만)

검정
- Table III: 칸별 Wilson 95% 구간, RL 전후 Fisher 정확 검정(양측), 네 검정에 Holm 보정,
  쌍 설계 가정 시 McNemar 정확 검정의 가능한 범위(에피소드별 기록이 없으므로 경계값만).
- Table IV: 분모 10인 칸의 Wilson 구간, 모달리티·방향 비교 Fisher, 19/26 구간.
- Table I·II: 표준편차만 있고 반복 수가 없으므로 Welch는 n을 가정한 민감도 분석으로만 한다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests

OUT = Path(__file__).with_name("table_check_results.json")
R = {}


def wilson(k, n, alpha=0.05):
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return [round(float(lo), 4), round(float(hi), 4)]


def newcombe_diff(k1, n1, k2, n2, alpha=0.05):
    """Newcombe(1998) 점수 구간: 두 비율 차이 p2 - p1."""
    l1, u1 = proportion_confint(k1, n1, alpha=alpha, method="wilson")
    l2, u2 = proportion_confint(k2, n2, alpha=alpha, method="wilson")
    p1, p2 = k1 / n1, k2 / n2
    d = p2 - p1
    lo = d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    hi = d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return [round(lo, 4), round(hi, 4)]


def mcnemar_bounds(k_off, k_on, n):
    """쌍 설계일 때 가능한 불일치 쌍 수(b: 꺼짐 성공·켜짐 실패, c: 꺼짐 실패·켜짐 성공).
    c - b = k_on - k_off 이고 b, c >= 0. 정확 McNemar p(양측)의 범위를 구한다."""
    d = k_on - k_off
    ps = []
    for b in range(0, n + 1):
        c = b + d
        if c < 0 or c + b > n or b > k_off or c > n - k_off:
            continue
        m = b + c
        if m == 0:
            ps.append(1.0)
            continue
        ps.append(min(1.0, 2 * stats.binom.cdf(min(b, c), m, 0.5)))
    if not ps:
        return None
    return {"min_p": round(min(ps), 4), "max_p": round(max(ps), 4)}


# ---------- Table III: 시뮬-시뮬, 칸당 n=20 ----------
T3 = {
    "ID_points": (16, 19), "ID_position": (14, 16),
    "OOD_points": (5, 9), "OOD_position": (4, 12),
}
n3 = 20
t3 = {"cells": {}, "fisher": {}, "mcnemar_paired_range": {}}
for name, (off, on) in T3.items():
    t3["cells"][name] = {"RL_off": [off, n3, round(off / n3, 3), wilson(off, n3)],
                         "RL_on": [on, n3, round(on / n3, 3), wilson(on, n3)]}
    _, p = stats.fisher_exact([[off, n3 - off], [on, n3 - on]], alternative="two-sided")
    t3["fisher"][name] = {"p_raw": round(float(p), 4), "diff_pp": round(100 * (on - off) / n3, 1),
                          "newcombe_diff_ci": newcombe_diff(off, n3, on, n3)}
    t3["mcnemar_paired_range"][name] = mcnemar_bounds(off, on, n3)
rej, p_holm, _, _ = multipletests([t3["fisher"][k]["p_raw"] for k in T3], method="holm")
for k, ph, r in zip(T3, p_holm, rej):
    t3["fisher"][k]["p_holm"] = round(float(ph), 4)
R["table3"] = t3

# ---------- Table IV: 실물, 분모 10(칸), 19/26(밀집) ----------
t4 = {"dense_19_of_26_wilson": wilson(19, 26)}
cells = {"MoCap_vertical": (8, 10), "MoCap_lateral": (9, 10),
         "Ego_vertical": (5, 10), "Ego_lateral": (6, 10)}
t4["cells_wilson"] = {k: {"k": k_, "n": n_, "ci": wilson(k_, n_)} for k, (k_, n_) in cells.items()}
pairs = {
    "MoCap_pooled_vs_Ego_pooled (17/20 vs 11/20)": ((17, 20), (11, 20)),
    "MoCap_vs_Ego_vertical (8/10 vs 5/10)": ((8, 10), (5, 10)),
    "MoCap_vs_Ego_lateral (9/10 vs 6/10)": ((9, 10), (6, 10)),
}
t4["fisher"] = {}
for name, ((a, na), (b, nb)) in pairs.items():
    _, p = stats.fisher_exact([[a, na - a], [b, nb - b]])
    t4["fisher"][name] = {"p": round(float(p), 4)}
R["table4"] = t4

# ---------- Table I, II: 반복 수 미기재 -> n 가정 민감도 ----------
def welch_sens(m1, s1, m2, s2, ns=(3, 5, 10, 20)):
    out = {}
    for n in ns:
        t, p = stats.ttest_ind_from_stats(m1, s1, n, m2, s2, n, equal_var=False)
        out[f"n={n}"] = round(float(p), 6)
    return out

R["table1_welch_sensitivity"] = {
    "ID_humanoid_succ: ULTRA(Ours) 67.30±0.12 vs ULTRA(Distillation) 85.03±3.00":
        welch_sens(67.30, 0.12, 85.03, 3.00),
    "OOD_humanoid_succ: ULTRA(Ours) 70.57±0.54 vs OmniRetarget(augmented, retrained) 46.71±0.74":
        welch_sens(70.57, 0.54, 46.71, 0.74),
    "ID_humanoid_succ: ULTRA(Ours) 67.30±0.12 vs ULTRA(RL) 54.47±0.43":
        welch_sens(67.30, 0.12, 54.47, 0.43),
}
R["table2_welch_sensitivity"] = {
    "Largebox foot skating duration: ULTRA 0.061±0.031 vs OmniRetarget 0.205±0.106":
        welch_sens(0.061, 0.031, 0.205, 0.106),
    "Largebox contact floating: ULTRA 0.015±0.063 vs OmniRetarget 0.231±0.224":
        welch_sens(0.015, 0.063, 0.231, 0.224),
    "Suitcase penetration duration: ULTRA 0.002±0.013 vs OmniRetarget 0.003±0.016":
        welch_sens(0.002, 0.013, 0.003, 0.016),
}

# ---------- 본문 수치 대조 ----------
R["text_checks"] = {
    "RL gain ID points +18.8% (19/16-1)": round(100 * (19 / 16 - 1), 1),
    "RL gain ID position +14.3% (16/14-1)": round(100 * (16 / 14 - 1), 1),
    "RL gain OOD points +80% (9/5-1)": round(100 * (9 / 5 - 1), 1),
    "RL gain OOD position +200% (12/4-1)": round(100 * (12 / 4 - 1), 1),
    "OOD points ratio 9/5 = 1.8 ('nearly doubling')": round(9 / 5, 2),
    "OOD position ratio 12/4 = 3 ('tripling')": round(12 / 4, 2),
    "Teacher-student ID gap 97.57-67.30 (pp)": round(97.57 - 67.30, 2),
    "Table IV dense 19/26 = 73.1%": round(100 * 19 / 26, 1),
}

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2))
if __name__ == "__main__":
    print(json.dumps(R, ensure_ascii=False, indent=1))
