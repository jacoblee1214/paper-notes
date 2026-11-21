"""TacInno 표 1-4의 수치를 논문 원자료(성공 횟수와 시도 수)에서 다시 계산하고 검정한다.

입력: 논문 표의 값과 시도 수(100회 또는 50회/과제, 표 주석 기준). 원자료(에피소드별 결과)는 공개되지 않았으므로
쌍체 검정(McNemar)은 표 3의 Lift Can 극단값(88 대 0)에만 쓰고, 나머지는 비쌍체 검정으로 근사한다.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)
출력: analysis/recompute_results.json 과 표준출력
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
Z = 1.959964
results = {}


def wilson(k, n, z=Z):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe(k1, n1, k2, n2):
    """차이 p1 - p2의 Newcombe 95% 구간 (Wilson 구간의 조합)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def two_prop_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return 2 * (1 - stats.norm.cdf(abs(z)))


def fisher(k1, n1, k2, n2):
    return stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]


def mcnemar_exact(b, c):
    m = b + c
    if m == 0:
        return 1.0
    return min(1.0, 2 * stats.binom.cdf(min(b, c), m, 0.5))


def n_per_group(p1, p2, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return math.ceil((num / abs(p1 - p2)) ** 2)


def pct(x):
    return round(100 * x, 2)


# ---------- 표 1: 시뮬레이션 6과제, 과제당 100 에피소드 ----------
T1 = {  # 과제 순서: Insert Hole, Insert Tube, Lift Bottle, Lift Can, Pull-out Key, Put Bottle
    "UniVTAC-ACT": [46, 55, 49, 27, 41, 25],
    "pi0.5 (no tactile)": [88, 70, 26, 25, 36, 40],
    "FTP-pi0.5": [77, 99, 45, 33, 19, 88],
    "FTP-1": [63, 99, 61, 94, 40, 88],
    "N0-VTLA": [99, 90, 24, 81, 77, 81],
    "SimpleTouch": [94, 93, 82, 65, 54, 73],
    "TacInno-VLA": [95, 95, 78, 88, 94, 75],
}
T1_REPORTED = {"UniVTAC-ACT": 40.5, "pi0.5 (no tactile)": 47.5, "FTP-pi0.5": 60.2, "FTP-1": 74.2,
               "N0-VTLA": 75.3, "SimpleTouch": 76.8, "TacInno-VLA": 87.5}
t1 = {}
for name, v in T1.items():
    k, n = sum(v), 6 * 100
    lo, hi = wilson(k, n)
    t1[name] = dict(avg_recomputed=round(k / 6, 2), avg_reported=T1_REPORTED[name],
                    pooled_ci95=[pct(lo), pct(hi)], min_task=min(v))
results["table1"] = t1
tac = (525, 600)
t1_tests = {}
for name in ["SimpleTouch", "N0-VTLA", "FTP-1", "FTP-pi0.5", "UniVTAC-ACT", "pi0.5 (no tactile)"]:
    k = sum(T1[name])
    t1_tests[name] = dict(diff_pp=round(pct(tac[0] / 600) - pct(k / 600), 2),
                          newcombe_ci_pp=[round(100 * x, 2) for x in newcombe(tac[0], 600, k, 600)],
                          p_two_prop=float(f"{two_prop_z(tac[0], 600, k, 600):.3g}"))
results["table1_tests_pooled_600"] = t1_tests
results["table1_min_task_check"] = {n: min(v) for n, v in T1.items()}
results["table1_n_for_80pct_power_vs_SimpleTouch"] = n_per_group(0.875, 0.768)

# 과제별 Fisher (TacInno 대 SimpleTouch, 과제당 100 대 100, 같은 시드이지만 쌍체 자료 없음)
per_task_p = [float(f"{fisher(T1['TacInno-VLA'][i], 100, T1['SimpleTouch'][i], 100):.3g}") for i in range(6)]
results["table1_fisher_per_task_vs_SimpleTouch"] = per_task_p

# ---------- 표 2: 절제, 두 과제(Lift Can, Lift Bottle) 평균, 과제당 100 ----------
T2 = {  # (Lift Can, Lift Bottle)
    "TacInno-VLA": (88, 78), "w/o conditioning": (63, 61), "vision-only conditioning": (62, 77),
    "two-frame code": (81, 72), "w/o attention mask": (66, 70), "w/o code loss": (84, 68),
    "w/o tactile input": (25, 26),
}
t2 = {}
base = sum(T2["TacInno-VLA"])
for name, v in T2.items():
    k = sum(v)
    entry = dict(avg_recomputed=k / 2, pooled_n=200)  # 과제당 100회 두 과제 = 200회; 평균 % = k/2
    if name != "TacInno-VLA":
        entry.update(diff_vs_full_pp=round(k / 2 - base / 2, 2),
                     newcombe_ci_pp=[round(100 * x, 2) for x in newcombe(base, 200, k, 200)],
                     p_two_prop_unpaired=float(f"{two_prop_z(base, 200, k, 200):.3g}"))
    t2[name] = entry
results["table2"] = t2
results["table2_reported_avg_check"] = {"TacInno-VLA": 83.0, "w/o conditioning": 62.0, "vision-only conditioning": 69.5,
                                        "two-frame code": 76.5, "w/o attention mask": 68.0,
                                        "w/o code loss": 76.0, "w/o tactile input": 25.5}

# ---------- 표 3: 촉각 끄기 검사 ----------
# 상대 감소 = (켬 - 끔) / 켬. TacInno-VLA Lift Can: 켬 88%(100회), 끔 0%(100회) -> 쌍체 극단값.
b, c = 88, 0
results["table3_mcnemar_lift_can_tacinno"] = dict(b_on_off=b, c_off_on=c, p_exact=float(f"{mcnemar_exact(b, c):.3g}"),
                                                 note="끔 0/100이면 감소 100%와 일치. 끔의 성공이 0이므로 쌍체 불일치가 88:0")
results["table3_lift_bottle_drop_check"] = dict(
    on=78, off_reported_text=4, relative_drop_recomputed=round(100 * (78 - 4) / 78, 1), reported_drop=95,
    seeds_note="표 3 주석은 Lift Bottle을 25 시드로 적고, 표 2와 4.1절은 100 에피소드라고 적어 시도 수가 모순됩니다.")

# ---------- 표 4: 실로봇, 과제당 50 시도 ----------
T4 = {  # Wipe Board, Insert Plug, Lift Can, Unscrew Cap
    "FTP-1": [88, 16, 66, 64], "SimpleTouch": [82, 28, 62, 74], "TacInno-VLA, tactile-off": [52, 10, 46, 52],
    "w/o code loss": [62, 22, 50, 56], "w/o tactile input": [62, 14, 52, 78], "TacInno-VLA": [88, 38, 74, 82],
}
# 표 4의 값은 %이고 과제당 50회이므로 성공 횟수는 값/2이다 (모든 값이 짝수여야 한다)
T4 = {name: [x // 2 for x in v] for name, v in T4.items()}
T4_REPORTED = {"FTP-1": 58.5, "SimpleTouch": 61.5, "TacInno-VLA, tactile-off": 40.0, "w/o code loss": 47.5,
               "w/o tactile input": 51.5, "TacInno-VLA": 70.5}
t4 = {}
all_even = all(True for v in T4.values())  # 입력은 이미 횟수(값/2)로 바꾸었다; 짝수 검사는 아래에서 한다
_raw_even = {"TacInno-VLA": [88, 38, 74, 82], "FTP-1": [88, 16, 66, 64], "SimpleTouch": [82, 28, 62, 74]}
all_even = all(x % 2 == 0 for v in _raw_even.values() for x in v)
for name, v in T4.items():
    k = sum(v)
    lo, hi = wilson(k, 200)
    t4[name] = dict(pooled_k=k, n=200, avg_recomputed=round(100 * k / 200, 2), avg_reported=T4_REPORTED[name],
                    pooled_ci95=[pct(lo), pct(hi)], per_task_ci95=[[pct(x) for x in wilson(t, 50)] for t in v])
results["table4"] = t4
results["table4_all_counts_even_consistent_with_n50"] = all_even
t4_tests = {}
ref = sum(T4["TacInno-VLA"])
for name in ["SimpleTouch", "FTP-1", "w/o code loss", "w/o tactile input", "TacInno-VLA, tactile-off"]:
    k = sum(T4[name])
    t4_tests[name] = dict(diff_pp=round(pct(ref / 200) - pct(k / 200), 2),
                          newcombe_ci_pp=[round(100 * x, 2) for x in newcombe(ref, 200, k, 200)],
                          p_two_prop_unpaired=float(f"{two_prop_z(ref, 200, k, 200):.3g}"),
                          p_fisher=float(f"{fisher(ref, 200, k, 200):.3g}"))
results["table4_tests"] = t4_tests
results["table4_power"] = dict(
    n_per_group_80pct_for_observed_diff_vs_SimpleTouch=n_per_group(0.705, 0.615),
    n_per_group_80pct_for_observed_diff_vs_FTP1=n_per_group(0.705, 0.585),
    n_per_group_80pct_for_observed_diff_vs_no_code_loss=n_per_group(0.705, 0.475))

# ---------- 본문 수치 일관성 ----------
checks = {
    "abstract: 87.5 - 76.8 (SimpleTouch) = 10.7 pp": round(87.5 - 76.8, 2),
    "sec 4.2: 87.5 - 75.3 (N0-VTLA) = 12.2 pp": round(87.5 - 75.3, 2),
    "sec 4.6: 70.5 - 61.5 (SimpleTouch) = 9.0 pp": round(70.5 - 61.5, 2),
    "sec 4.6: 70.5 - 58.5 (FTP-1) = 12.0 pp": round(70.5 - 58.5, 2),
    "sec 4.3: w/o conditioning drop = 83.0 - 62.0 = 21 pp": round(83.0 - 62.0, 2),
    "sec 4.3: vision-only remains 83.0 - 69.5 = 13.5 pp": round(83.0 - 69.5, 2),
    "sec 4.3: two-frame drop = 83.0 - 76.5 = 6.5 pp": round(83.0 - 76.5, 2),
    "sec 4.3: attention mask drop = 83.0 - 68.0 = 15 pp": round(83.0 - 68.0, 2),
    "sec 4.3: w/o tactile drop = 83.0 - 25.5 = 57.5 pp": round(83.0 - 25.5, 2),
    "sec 4.3: w/o code loss drop = 83.0 - 76.0 = 7 pp": round(83.0 - 76.0, 2),
    "sec 4.6: code loss gain per task (26, 16, 24, 26) range": [min(88 - 62, 38 - 22, 74 - 50, 82 - 56), max(88 - 62, 38 - 22, 74 - 50, 82 - 56)],
    "sec 4.5: 0.073 / 0.020 nats ratio": round(0.073 / 0.020, 3),
    "sec 4.5: 65+25, 78+16, 87+7 (vision+action+code, %)": [65 + 25, 78 + 16, 87 + 7],
    "sec 4.4: 'to 4% on Lift Bottle' with 78% on-tactile: relative drop %": round(100 * (78 - 4) / 78, 1),
    "sec 4.2: execution-horizon baseline averages and 91.3% (not in any table; unverifiable)": None,
}
results["consistency"] = checks
results["sec4_5_predictability_note"] = ("37% 평균은 6과제 전체 평균이고, Lift Can 값(65, 78, 87%)은 다른 분모입니다. "
                                         "두 값을 직접 비교하지 않았습니다.")

OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def show(title, obj):
    print(f"\n== {title}")
    print(json.dumps(obj, ensure_ascii=False, indent=1, default=float))


show("table1", t1)
show("table1 pooled tests", t1_tests)
show("table1 fisher per task vs SimpleTouch p", per_task_p)
show("table2", t2)
show("table3 McNemar", results["table3_mcnemar_lift_can_tacinno"])
show("table4", {k: {kk: vv for kk, vv in v.items() if kk != "per_task_ci95"} for k, v in t4.items()})
show("table4 tests", t4_tests)
show("table4 power", results["table4_power"])
show("consistency", checks)
print("\nwrote", OUT)
