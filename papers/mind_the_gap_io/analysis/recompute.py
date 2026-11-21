"""Mind the Gap (arXiv:2602.08776 v2) 논문 표 수치를 다시 계산하고 검정한다.

입력: 논문 v2의 표 값을 아래에 그대로 옮겼다(출처 표기). 외부 데이터 없음.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시)
- 성공률은 이항 시행으로 보고 Wilson 95% CI와 Fisher 정확 검정(양측)을 쓴다.
- 시행은 서로 독립이라 가정한다. 학습 seed 수는 논문에 없다(시도 = 롤아웃 횟수).
- Table 3 π0 행의 EC2C는 Inpainting, E2E/E2C는 Naive Async라서 추론 방식이 다르다.
  이 비교는 "I/O 효과"와 "추론 방식 효과"가 섞여 있다.
- Table 4의 Weight Sorting은 Tobs=1 분모 30, Tobs=9 분모 49로 다르다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def fisher(k1, n1, k2, n2):
    tab = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(stats.fisher_exact(tab, alternative="two-sided")[1])


def binom_vs_half(k, n):
    return float(stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue)


def cell(name, k, n):
    lo, hi = wilson(k, n)
    return {"name": name, "k": k, "n": n, "p": round(k / n, 3), "ci95": [round(lo, 3), round(hi, 3)]}


def compare(label, a, b):
    """a, b = (name, k, n). 차이와 Fisher p."""
    p = fisher(a[1], a[2], b[1], b[2])
    d = b[1] / b[2] - a[1] / a[2]
    return {"label": label, "a": cell(a[0], a[1], a[2]), "b": cell(b[0], b[1], b[2]),
            "diff_b_minus_a": round(d, 3), "fisher_p": round(p, 4)}


# ---- Table 1: stiffness (Contracted, Extended), 논문 Table 1 ----
stiff = {"X": (3501, 0.981, 2914, 0.977, -17), "Y": (791, 0.998, 314, 0.995, -60), "Z": (1709, 0.977, 328, 0.992, -81)}
RES["table1_change_pct"] = {ax: {"recomputed": round(100 * (v[2] - v[0]) / v[0], 1), "reported": v[4]}
                            for ax, v in stiff.items()}
RES["table1_min_r2"] = min(min(v[1], v[3]) for v in stiff.values())

# ---- Table 2 (논문 Table 2) ----
t2 = []
t2.append(compare("Blind plug, π0: E2C vs EC2C (Q3)", ("E2C", 8, 20), ("EC2C", 16, 20)))
t2.append(compare("Wiping strict, π0: E2C vs EC2C (Q3)", ("E2C", 4, 20), ("EC2C", 18, 20)))
t2.append(compare("Wiping strict, DP: E2C vs EC2C (Q3)", ("E2C", 4, 20), ("EC2C", 13, 20)))
t2.append(compare("Wiping effective, π0: E2C vs EC2C", ("E2C", 13, 20), ("EC2C", 20, 20)))
t2.append(compare("Microwave Open, π0: E2E vs E2C (Q2)", ("E2E", 0, 20), ("E2C", 18, 20)))
t2.append(compare("Microwave Open, DP: E2E vs E2C (Q2)", ("E2E", 0, 20), ("E2C", 14, 20)))
t2.append(compare("Microwave Open, π0: E2C vs EC2C (Q3)", ("E2C", 18, 20), ("EC2C", 20, 20)))
t2.append(compare("Microwave Close, π0: E2C vs EC2C", ("E2C", 18, 20), ("EC2C", 19, 20)))
t2.append(compare("Plug Clear, π0: E2E vs E2C (Q2 force gen.)", ("E2E", 0, 20), ("E2C", 19, 20)))
t2.append(compare("Plug Clear, π0: E2C vs EC2C", ("E2C", 19, 20), ("EC2C", 19, 20)))
t2.append(compare("Plug Blind, π0: E2E vs E2C", ("E2E", 0, 20), ("E2C", 8, 20)))
RES["table2_comparisons"] = t2

# ---- Table 3 (논문 Table 3) ----
# Conveyor sort: DP naive async. π0 E2E/E2C naive; π0 EC2C inpainting.
RES["table3_sort_vs_chance"] = {
    "DP E2E 17/33": round(binom_vs_half(17, 33), 3),
    "DP E2C 20/36": round(binom_vs_half(20, 36), 3),
    "π0 E2E 16/32": round(binom_vs_half(16, 32), 3),
    "π0 E2C 28/45": round(binom_vs_half(28, 45), 3),
    "π0 EC2C 48/49": round(binom_vs_half(48, 49), 6),
}
RES["table3_sort_rates"] = {k: round(a / b, 3) for k, a, b in
                            [("DP E2E", 17, 33), ("DP E2C", 20, 36), ("DP EC2C naive", 41, 46),
                             ("π0 E2E", 16, 32), ("π0 E2C", 28, 45), ("π0 EC2C inpainting", 48, 49)]}
t3 = []
t3.append(compare("Sort, DP: E2C vs EC2C (same naive async)", ("E2C", 20, 36), ("EC2C", 41, 46)))
t3.append(compare("Sort, π0: E2C naive vs EC2C inpainting (confounded)", ("E2C", 28, 45), ("EC2C", 48, 49)))
t3.append(compare("Grasp, π0: sync EC2C vs inpainting EC2C", ("sync", 0, 50), ("inpaint", 50, 50)))
t3.append(compare("Tossing, π0: E2C vs EC2C (inpainting)", ("E2C", 17, 20), ("EC2C", 18, 20)))
t3.append(compare("Tossing, π0: sync EC2C vs inpainting EC2C (§4.2.6)", ("sync", 14, 20), ("inpaint", 18, 20)))
RES["table3_comparisons"] = t3

# ---- Table 4: observation horizon (논문 Table 4) ----
t4 = []
t4.append(compare("Plug Clear: Tobs=1 vs Tobs=9", ("Tobs=1", 19, 20), ("Tobs=9", 12, 20)))
t4.append(compare("Plug Blind: Tobs=1 vs Tobs=9", ("Tobs=1", 16, 20), ("Tobs=9", 12, 20)))
t4.append(compare("Weight sorting: Tobs=1 (n=30) vs Tobs=9 (n=49)", ("Tobs=1", 18, 30), ("Tobs=9", 48, 49)))
RES["table4_comparisons"] = t4

# ---- Table 5: payload shift (논문 Table 5), n=10 or 20 ----
t5 = []
t5.append(compare("A1→A1 (nominal, zero-shot)", ("E2C", 18, 20), ("EC2C", 20, 20)))
t5.append(compare("A1→B2 (500 g zero-shot)", ("E2C", 0, 10), ("EC2C", 8, 10)))
t5.append(compare("B2→B2 (500 g trained)", ("E2C", 7, 10), ("EC2C", 8, 10)))
t5.append(compare("A1B2→A2 (visual recombination)", ("E2C", 0, 10), ("EC2C", 10, 10)))
t5.append(compare("A1A2B2→A2", ("E2C", 10, 10), ("EC2C", 10, 10)))
RES["table5_comparisons"] = t5

# ---- Table 6: latency (논문 Table 6), n=50, jerk는 평균만 보고 ----
t6 = []
for ms, nv, ev in [(100, 50, 50), (200, 47, 48), (300, 45, 48), (400, 40, 45)]:
    t6.append(compare(f"{ms} ms: Naive vs Inpainting success", ("Naive", nv, 50), ("Ours", ev, 50)))
RES["table6_comparisons"] = t6
RES["table6_jerk"] = {"naive": [0.75, 0.80, 1.01, 1.28], "ours": [0.75, 0.78, 0.82, 0.88],
                      "note": "SD 없음, 반복 측정 없음 -> 검정 불가"}

# ---- Summary ----
RES["summary_p_lt_0.05"] = {
    "all_sig_table2_table5": [c["label"] for c in t2 + t5 if c["fisher_p"] < 0.05],
    "not_sig": [c["label"] for c in t2 + t3 + t4 + t5 + t6 if c["fisher_p"] >= 0.05],
}

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    print("Table 1 change % recomputed:", RES["table1_change_pct"], "min R²:", RES["table1_min_r2"])
    for sec in ["table2_comparisons", "table3_comparisons", "table4_comparisons", "table5_comparisons", "table6_comparisons"]:
        print("==", sec)
        for c in RES[sec]:
            print(f"  {c['label']:<55} {c['a']['k']}/{c['a']['n']} vs {c['b']['k']}/{c['b']['n']}  diff={c['diff_b_minus_a']:+.3f}  p={c['fisher_p']}")
    print("Sort rates:", RES["table3_sort_rates"])
    print("Sort vs chance:", RES["table3_sort_vs_chance"])
    print("Not sig count:", len(RES["summary_p_lt_0.05"]["not_sig"]))
