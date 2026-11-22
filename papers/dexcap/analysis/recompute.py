"""DexCap (arXiv:2403.07788 v2, RSS 2024) 논문 표 수치를 다시 계산하고 통계 검정을 한다.

입력: 논문 본문과 표의 값을 아래에 그대로 옮겨 적었다(출처는 주석). 외부 데이터는 쓰지 않는다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 표 I의 과제당 시행 수는 본문 "Each model variant is tested for 20 trials in each task"에 따라 20이다.
- 표 II의 분모는 본문의 학습 객체 6개 x 5회 = 30(trained), 미학습 객체 9개 x 5회 = 45(unseen)다.
- 표 III(Scissor)과 표 IV(Tea)의 분모는 논문에 적히지 않았다. 본문 규칙(과제당 20회)을 따라 20으로 가정한다.
- 표 II의 "Subtask"와 "All"은 같은 시행에서 측정된 값으로 본다.
- 비율은 0.05 단위로 적혀 있어 분자는 반올림 없이 정수로 복원된다(검사 항목 K1에서 확인).
- Table VIII(IMU 드리프트)은 시행 수가 적혀 있지 않다. 민감도 분석에서 n을 가정한다.
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


def newcombe_diff(k1, n1, k2, n2, z=1.959964):
    """Newcombe (1998) 방법 10: 두 비율 차이의 Wilson 기반 구간."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def fisher(k1, n1, k2, n2):
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(stats.fisher_exact(table, alternative="two-sided")[1])


def comp(name, k1, n1, k2, n2):
    d, lo, hi = newcombe_diff(k1, n1, k2, n2)
    return {
        "comparison": name,
        "k1": k1, "n1": n1, "k2": k2, "n2": n2,
        "diff_pp": round(100 * d, 1),
        "newcombe95_pp": [round(100 * lo, 1), round(100 * hi, 1)],
        "fisher_p": round(fisher(k1, n1, k2, n2), 4),
    }


def mde(n, p0=0.5, alpha=0.05, power=0.8):
    """두 독립 비율, 같은 n, 기준 비율 p0일 때 검출 가능한 최소 차이(근사식)."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    best = None
    for d in np.linspace(0.01, 0.99, 981):
        p1, p2 = p0, min(0.999, p0 + d)
        se = math.sqrt(p1 * (1 - p1) + p2 * (1 - p2)) / math.sqrt(n)
        if abs(p2 - p1) >= (za + zb) * se:
            best = float(d)
            break
    return best


# ---------------------------------------------------------------
# 1. 원자료 복원 (정수 시행 수)
# 표 I (논문 v2 Table I): 과제당 20회. (Sponge, Ball, Plate) 성공 수
# ---------------------------------------------------------------
TABLE_I = {  # (sponge, ball, plate) out of 20 each
    "BC-RNN-img": (0, 0, 0),
    "BC-RNN-img-mask": (5, 2, 2),
    "BC-RNN-point": (9, 6, 5),
    "BC-RNN-prec": (10, 6, 7),
    "DP-img": (0, 0, 0),
    "DP-img-mask": (11, 8, 6),
    "DP-point-raw": (14, 14, 8),
    "DP-point": (15, 13, 10),
    "Ours (DP-perc)": (17, 12, 14),
}
TABLE_I_OVERALL_REPORTED = {
    "BC-RNN-img": 0.00, "BC-RNN-img-mask": 0.15, "BC-RNN-point": 0.33, "BC-RNN-prec": 0.38,
    "DP-img": 0.00, "DP-img-mask": 0.42, "DP-point-raw": 0.60, "DP-point": 0.63, "Ours (DP-perc)": 0.72,
}

# 표 II (Packaging). 열: 데이터만 (Subtask/30, All/30, Unseen/45), 교정 30회 후 (Subtask/30, All/30, Unseen/45)
TABLE_II = {
    "BC-RNN-img-mask": {"data": dict(sub=(0, 30), all=(0, 30), unseen=(0, 45)),
                        "corr": dict(sub=(7, 30), all=(2, 30), unseen=(0, 45))},
    "BC-RNN-point": {"data": dict(sub=(10, 30), all=(7, 30), unseen=(7, 45)),
                     "corr": dict(sub=(12, 30), all=(8, 30), unseen=(10, 45))},
    "DP-img-mask": {"data": dict(sub=(5, 30), all=(0, 30), unseen=(0, 45)),
                    "corr": dict(sub=(14, 30), all=(10, 30), unseen=(0, 45))},
    "Ours (DP-perc)": {"data": dict(sub=(21, 30), all=(14, 30), unseen=(18, 45)),
                       "corr": dict(sub=(25, 30), all=(17, 30), unseen=(19, 45))},
}
TABLE_II_REPORTED = {  # 보고된 값 (소수 둘째 자리)
    "BC-RNN-img-mask": {"data": (0.00, 0.00, 0.00), "corr": (0.23, 0.07, 0.00)},
    "BC-RNN-point": {"data": (0.33, 0.23, 0.16), "corr": (0.40, 0.27, 0.22)},
    "DP-img-mask": {"data": (0.17, 0.00, 0.00), "corr": (0.47, 0.33, 0.00)},
    "Ours (DP-perc)": {"data": (0.70, 0.47, 0.40), "corr": (0.83, 0.57, 0.42)},
}

# 표 III (Scissor cutting), 분모 20 가정. 열: 데이터만 (Subtask, All), 교정 30회 후 (Subtask, All)
TABLE_III = {
    "BC-RNN-point": {"data": (0, 20, 0, 20), "corr": (2, 20, 0, 20)},
    "Ours (DP-perc)": {"data": (0, 20, 0, 20), "corr": (9, 20, 4, 20)},
}
# 표 IV (Tea preparing), 분모 20 가정
TABLE_IV = {
    "Ours (DP-perc)": {"data": (6, 20, 0, 20), "corr": (13, 20, 5, 20)},
}

# ---------------------------------------------------------------
# 2. 표 I 재계산: 과제별 평균, 전체 평균 (논문 보고값과 대조)
# ---------------------------------------------------------------
t1 = {}
for m, (s, b, p) in TABLE_I.items():
    overall_k = s + b + p
    overall = overall_k / 60
    t1[m] = {
        "per_task": [round(s / 20, 2), round(b / 20, 2), round(p / 20, 2)],
        "overall_k_of_60": overall_k,
        "overall_recomputed": round(overall, 3),
        "overall_reported": TABLE_I_OVERALL_REPORTED[m],
        "match_2dp": round(overall, 2) == TABLE_I_OVERALL_REPORTED[m],
        "wilson95_overall": [round(x, 3) for x in wilson(overall_k, 60)],
    }
RES["K1_table1_overall"] = t1

# 표 II 역산 검사: 보고된 비율 x 분모 = 정수인지
k1 = {}
for m, cols in TABLE_II_REPORTED.items():
    for col, vals in cols.items():
        for i, (key, v) in enumerate(zip(("sub", "all", "unseen"), vals)):
            cnt, den = TABLE_II[m][col][key]
            ok = round(v * den) == cnt and abs(cnt / den - v) < 0.005
            k1[f"{m}|{col}|{key}"] = {"count": cnt, "den": den, "reported": v, "consistent": bool(ok)}
RES["K1_table2_integer_check"] = k1

# 표 III/IV 검사 (분모 20 가정)
k1b = {}
for name, tab in (("T3", TABLE_III), ("T4", TABLE_IV)):
    for m, cols in tab.items():
        for col, (s_k, s_n, a_k, a_n) in cols.items():
            k1b[f"{name}|{m}|{col}"] = {
                "subtask": f"{s_k}/{s_n}={s_k / s_n:.2f}", "all": f"{a_k}/{a_n}={a_k / a_n:.2f}"}
RES["K1_table34_counts"] = k1b

# ---------------------------------------------------------------
# 3. 표 I 검정 (풀링 60회 기준, 과제 내 20회)
# ---------------------------------------------------------------
pooled = {m: sum(v) for m, v in TABLE_I.items()}
tests_t1 = [
    comp("Ours vs DP-point (pooled 60)", pooled["Ours (DP-perc)"], 60, pooled["DP-point"], 60),
    comp("Ours vs DP-point-raw (pooled 60)", pooled["Ours (DP-perc)"], 60, pooled["DP-point-raw"], 60),
    comp("Ours vs DP-img-mask (pooled 60)", pooled["Ours (DP-perc)"], 60, pooled["DP-img-mask"], 60),
    comp("Ours vs BC-RNN-point (pooled 60)", pooled["Ours (DP-perc)"], 60, pooled["BC-RNN-point"], 60),
    comp("DP-point vs BC-RNN-point (pooled 60)", pooled["DP-point"], 60, pooled["BC-RNN-point"], 60),
    comp("DP-img-mask vs BC-RNN-img-mask (pooled 60)", pooled["DP-img-mask"], 60, pooled["BC-RNN-img-mask"], 60),
    comp("DP-point vs DP-img-mask (pooled 60)", pooled["DP-point"], 60, pooled["DP-img-mask"], 60),
    comp("DP-point-raw vs DP-point (pooled 60)", pooled["DP-point-raw"], 60, pooled["DP-point"], 60),
]
# 과제별 (Plate: Perceiver 대 PointNet 주장, Ball: 순위 역전 여부)
S, B, P = 0, 1, 2
def tk(m, t):
    return TABLE_I[m][t]
tests_t1_task = [
    comp("Plate: Ours vs DP-point", tk("Ours (DP-perc)", P), 20, tk("DP-point", P), 20),
    comp("Plate: Ours vs DP-point-raw", tk("Ours (DP-perc)", P), 20, tk("DP-point-raw", P), 20),
    comp("Plate: DP-point vs DP-point-raw", tk("DP-point", P), 20, tk("DP-point-raw", P), 20),
    comp("Ball: Ours vs DP-point-raw", tk("Ours (DP-perc)", B), 20, tk("DP-point-raw", B), 20),
    comp("Ball: Ours vs DP-point", tk("Ours (DP-perc)", B), 20, tk("DP-point", B), 20),
    comp("Sponge: Ours vs DP-point", tk("Ours (DP-perc)", S), 20, tk("DP-point", S), 20),
    comp("Sponge: Ours vs DP-img-mask", tk("Ours (DP-perc)", S), 20, tk("DP-img-mask", S), 20),
]
RES["T1_pooled_tests"] = tests_t1
RES["T1_per_task_tests"] = tests_t1_task

# ---------------------------------------------------------------
# 4. 표 II 검정 (Packaging)
# ---------------------------------------------------------------
def tb2(m, col, key):
    return TABLE_II[m][col][key]

tests_t2 = [
    comp("Packaging data-only all(30): Ours vs BC-RNN-point", *tb2("Ours (DP-perc)", "data", "all"), *tb2("BC-RNN-point", "data", "all")),
    comp("Packaging data-only unseen(45): Ours vs BC-RNN-point", *tb2("Ours (DP-perc)", "data", "unseen"), *tb2("BC-RNN-point", "data", "unseen")),
    comp("Packaging data-only all(30): Ours vs DP-img-mask", *tb2("Ours (DP-perc)", "data", "all"), *tb2("DP-img-mask", "data", "all")),
    comp("Packaging data-only subtask(30): Ours vs BC-RNN-point", *tb2("Ours (DP-perc)", "data", "sub"), *tb2("BC-RNN-point", "data", "sub")),
    comp("Correction effect DP-img-mask all(30): corr vs data", *tb2("DP-img-mask", "corr", "all"), *tb2("DP-img-mask", "data", "all")),
    comp("Correction effect Ours all(30): corr vs data", *tb2("Ours (DP-perc)", "corr", "all"), *tb2("Ours (DP-perc)", "data", "all")),
    comp("Correction effect Ours subtask(30): corr vs data", *tb2("Ours (DP-perc)", "corr", "sub"), *tb2("Ours (DP-perc)", "data", "sub")),
    comp("Correction effect Ours unseen(45): corr vs data", *tb2("Ours (DP-perc)", "corr", "unseen"), *tb2("Ours (DP-perc)", "data", "unseen")),
    comp("Correction effect BC-RNN-point all(30): corr vs data", *tb2("BC-RNN-point", "corr", "all"), *tb2("BC-RNN-point", "data", "all")),
]
RES["T2_tests"] = tests_t2

# 교정 후 비교 (DP-img-mask 교정 vs Ours 데이터만 등 같은 조건이 아닌 비교는 참고용)
tests_t2b = [
    comp("Corrected: Ours all(30) vs DP-img-mask all(30)", *tb2("Ours (DP-perc)", "corr", "all"), *tb2("DP-img-mask", "corr", "all")),
]
RES["T2_tests_corrected"] = tests_t2b

# ---------------------------------------------------------------
# 5. 표 III, IV 검정
# ---------------------------------------------------------------
tests_t34 = [
    comp("Scissor corrected all(20): Ours vs BC-RNN-point", 4, 20, 0, 20),
    comp("Scissor corrected subtask(20): Ours vs BC-RNN-point", 9, 20, 2, 20),
    comp("Tea uncapping subtask(20): corrected vs data-only (Ours)", 13, 20, 6, 20),
    comp("Tea all(20): corrected vs data-only (Ours)", 5, 20, 0, 20),
    comp("Scissor subtask(20): corrected vs data-only (Ours)", 9, 20, 0, 20),
    comp("Scissor all(20): corrected vs data-only (Ours)", 4, 20, 0, 20),
]
RES["T34_tests"] = tests_t34

# ---------------------------------------------------------------
# 6. 검출력: 두 독립 비율, 기준 0.5, 양측 0.05, 검정력 0.8에서 최소 검출 차이
# ---------------------------------------------------------------
RES["mde_pp"] = {f"n={n}": round(100 * mde(n), 1) for n in (20, 30, 45, 60)}

# ---------------------------------------------------------------
# 7. IMU 드리프트 (Table VIII): 평균 ± SD, 시행 수 미보고 -> 민감도
# ---------------------------------------------------------------
imu = {}
for traj, (m1, s1, m2, s2) in {"Trajectory 1": (8.0, 3.1, 0.4, 0.2), "Trajectory 2": (11.3, 4.7, 0.8, 0.3)}.items():
    rows = {}
    for n in (3, 5, 10):
        r = stats.ttest_ind_from_stats(m1, s1, n, m2, s2, n, equal_var=False)
        rows[f"n={n}"] = {"welch_t": round(float(r.statistic), 2), "p": float(f"{r.pvalue:.2e}")}
    imu[traj] = rows
RES["T8_imu_sensitivity"] = imu

# ---------------------------------------------------------------
# 8. 출력
# ---------------------------------------------------------------
def show(title, rows):
    print(f"\n== {title}")
    for r in rows:
        print(f"  {r['comparison']}: {r['k1']}/{r['n1']} vs {r['k2']}/{r['n2']}, "
              f"diff {r['diff_pp']} pp, 95% {r['newcombe95_pp']}, Fisher p={r['fisher_p']}")

print("== K1 표 I 전체 평균 대 보고값")
for m, v in t1.items():
    print(f"  {m}: per-task {v['per_task']}, overall {v['overall_k_of_60']}/60 = {v['overall_recomputed']} "
          f"(보고 {v['overall_reported']}) 일치={v['match_2dp']}, Wilson95 {v['wilson95_overall']}")
bad = [k for k, v in k1.items() if not v["consistent"]]
print(f"== K1 표 II 정수 역산 불일치 항목: {bad if bad else '없음'}")
show("T1 풀링 (60회)", tests_t1)
show("T1 과제별", tests_t1_task)
show("T2 Packaging", tests_t2)
show("T2 교정 후 (참고)", tests_t2b)
show("T3/T4", tests_t34)
print("\n== 검출력 최소 차이 (pp)", RES["mde_pp"])
print("== IMU 민감도", json.dumps(imu, ensure_ascii=False))

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n저장: {OUT}")
