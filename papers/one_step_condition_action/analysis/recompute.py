"""UCA-Flow (arXiv:2608.16153 v5) 논문 표 수치를 다시 계산하고 통계 검정을 한다.

입력: 논문 표의 값을 아래에 그대로 옮겨 적었다(출처는 주석). 외부 데이터는 쓰지 않는다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- Table 2의 "±"는 시드 3개에 대한 표본 표준편차(SD)다. 캡션이 "sample standard deviation across seeds"라고 적는다.
- Table 1의 "±"도 SD로 가정한다(캡션에 명시 없음).
- 전체 평균은 37개 과제에 같은 가중치를 준다(캡션). 그룹 평균은 시드마다 먼저 계산된 값이다.
- 전체 평균의 SD는 그룹 SD만으로 정확히 정해지지 않는다. 독립 가정과 완전 양의 상관 가정의 범위로 본다.
- 실로봇 비율의 분모는 캡션의 시도 수(20회)다.
- 기준선(DP3, Simple DP3, FlowPolicy, MP1)의 값은 논문이 "as reported"라고 적었으므로 재실행하지 않았다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
results = {}

# ---------------- Table 2 (v5): 성공률, 평균 ± SD (시드 3개) ----------------
# 열: Hammer, Door, Pen (단일 과제) / Easy(21), Medium(4), Hard(4), Very Hard(5) / 논문 Average
T2 = {
    "DP3":        dict(h=(100, 0), d=(56, 5), p=(46, 10), easy=(87.3, 2.2), med=(44.5, 8.7), hard=(32.7, 7.7), vh=(39.4, 9.0), avg=(68.7, 4.7)),
    "Simple DP3": dict(h=(98, 2), d=(40, 17), p=(36, 4), easy=(86.8, 2.3), med=(42.0, 6.5), hard=(38.7, 7.5), vh=(35.0, 11.6), avg=(67.4, 5.0)),
    "FlowPolicy": dict(h=(98, 1), d=(61, 2), p=(54, 4), easy=(84.8, 2.2), med=(58.2, 7.9), hard=(40.2, 4.5), vh=(52.2, 5.0), avg=(71.6, 3.5)),
    "MP1":        dict(h=(100, 0), d=(69, 2), p=(58, 5), easy=(88.2, 1.1), med=(68.0, 3.1), hard=(58.1, 5.0), vh=(67.2, 2.7), avg=(78.9, 2.1)),
    "UCA-Flow":   dict(h=(100, 0), d=(77.3, 2.1), p=(63.2, 3.4), easy=(92.4, 0.7), med=(79.3, 1.8), hard=(68.0, 1.9), vh=(85.3, 0.7), avg=(86.4, 0.3)),
}
# 과제 수: 단일 3개 + Easy 21 + Medium 4 + Hard 4 + Very Hard 5 = 37
W = {"h": 1, "d": 1, "p": 1, "easy": 21, "med": 4, "hard": 4, "vh": 5}
NTASK = sum(W.values())
assert NTASK == 37

# Table 2 (v1, arXiv 2026-08-17): UCA-Flow 행만 다름 (v1 표 값)
T2_V1_UCA = dict(h=(100, 0), d=(77.3, 2.0), p=(63.0, 3.4), easy=(92.4, 1.2), med=(79.3, 3.4), hard=(68.0, 2.0), vh=(85.3, 2.0), avg=(88.2, 1.7))

# Table 3 (v5): 추론 지연 (ms), 평균 ± SD. 열은 Table 2와 같다.
T3 = {
    "DP3":        dict(h=129.5, d=141.3, p=145.1, easy=129.3, med=134.7, hard=131.9, vh=138.4, avg=132.2),
    "Simple DP3": dict(h=103.1, d=111.3, p=128.2, easy=91.9, med=98.3, hard=101.3, vh=103.8, avg=97.0),
    "FlowPolicy": dict(h=15.3, d=13.2, p=12.0, easy=12.0, med=12.2, hard=13.5, vh=14.5, avg=12.6),
    "MP1":        dict(h=7.1, d=7.2, p=7.4, easy=6.7, med=6.7, hard=6.7, vh=6.8, avg=6.8),
    "UCA-Flow":   dict(h=2.9, d=2.9, p=3.0, easy=3.1, med=2.9, hard=3.0, vh=2.8, avg=3.0),
}
T3_V1_UCA_AVG = 2.9  # v1 보고값


def weighted_mean(row, keys=None):
    keys = keys or list(W)
    return sum(W[k] * row[k][0] if isinstance(row[k], tuple) else W[k] * row[k] for k in keys) / sum(W[k] for k in keys)


def group_mean_task_weighted(row):
    return sum(W[k] * (row[k][0] if isinstance(row[k], tuple) else row[k]) for k in W) / NTASK


def sd_overall_bounds(row):
    """전체 평균 SD의 범위. 독립(하한 아님, 기준값)과 완전 양의 상관(상한)."""
    ws = {k: W[k] / NTASK for k in W}
    s = {k: row[k][1] for k in W}
    indep = math.sqrt(sum((ws[k] * s[k]) ** 2 for k in W))
    comono = sum(ws[k] * s[k] for k in W)
    return indep, comono


def welch(m1, s1, m2, s2, n=3):
    t, p = stats.ttest_ind_from_stats(m1, s1, n, m2, s2, n, equal_var=False)
    return float(t), float(p)


def ci95_t2(m, s, n=3):
    half = stats.t.ppf(0.975, n - 1) * s / math.sqrt(n)
    return (m - half, m + half)


# ---------- 1. Table 2 전체 평균 재계산 ----------
t2 = {}
for name, row in T2.items():
    recomputed = group_mean_task_weighted({k: row[k] for k in W})
    indep, comono = sd_overall_bounds(row)
    t2[name] = dict(
        reported=row["avg"][0],
        recomputed=round(recomputed, 2),
        diff=round(recomputed - row["avg"][0], 2),
        sd_reported=row["avg"][1],
        sd_indep=round(indep, 2),
        sd_comonotone=round(comono, 2),
        ci95_t2=[round(x, 2) for x in ci95_t2(row["avg"][0], row["avg"][1])],
    )
results["table2_average_check"] = t2

# ---------- 2. 전체 평균 차이 검정 (UCA-Flow 대 기준선, 보고된 평균과 SD, n=3) ----------
u = T2["UCA-Flow"]["avg"]
tests = {}
for b in ["DP3", "Simple DP3", "FlowPolicy", "MP1"]:
    bb = T2[b]["avg"]
    t, p = welch(u[0], u[1], bb[0], bb[1])
    tests[f"UCA-Flow vs {b}"] = dict(diff=round(u[0] - bb[0], 2), t=round(t, 2), p=float(f"{p:.3g}"))
results["table2_overall_welch"] = tests

# ---------- 3. 그룹별 검정 (UCA-Flow 대 MP1, FlowPolicy) ----------
grp = {}
for g in ["easy", "med", "hard", "vh", "d", "p", "h"]:
    for b in ["MP1", "FlowPolicy", "DP3"]:
        a, c = T2["UCA-Flow"][g], T2[b][g]
        if a[1] == 0 and c[1] == 0:
            continue
        t, p = welch(a[0], a[1], c[0], c[1])
        grp[f"{g} UCA-Flow vs {b}"] = dict(diff=round(a[0] - c[0], 2), t=round(t, 2), p=float(f"{p:.3g}"))
results["table2_group_welch"] = grp

# ---------- 4. 버전 차이: v1 표의 UCA-Flow 행으로 같은 계산 ----------
v1_row = {k: T2_V1_UCA[k] for k in W}
v1_avg_recomp = group_mean_task_weighted(v1_row)
v1_t3 = {k: T3["UCA-Flow"][k] for k in W}
v1_lat_recomp = group_mean_task_weighted({k: (v if k != "avg" else v) for k, v in T3["UCA-Flow"].items() if k in W})
results["version_check_v1"] = dict(
    reported_avg_success=T2_V1_UCA["avg"][0],
    recomputed_avg_success_from_v1_rows=round(v1_avg_recomp, 2),
    reported_avg_latency_ms=T3_V1_UCA_AVG,
    recomputed_avg_latency_from_v1_rows=round(v1_lat_recomp, 3),
    note="v1은 평균 성공률 88.2%, 지연 2.9 ms를 보고했으나 같은 표 행으로 계산하면 86.4%, 3.01 ms다. v5가 이를 고쳤다.",
)

# ---------- 5. Table 3: 과제 가중 지연과 속도비 ----------
lat = {}
for name, row in T3.items():
    lat[name] = dict(recomputed_avg_ms=round(group_mean_task_weighted(row), 3), reported_avg_ms=row["avg"])
uca_lat = group_mean_task_weighted(T3["UCA-Flow"])
ratios = {}
for b in ["DP3", "Simple DP3", "FlowPolicy", "MP1"]:
    ratio = group_mean_task_weighted(T3[b]) / uca_lat
    ratios[b] = dict(recomputed=round(ratio, 2))
reported_ratio = {"DP3": 43.9, "Simple DP3": 32.2, "FlowPolicy": 4.2, "MP1": 2.2}
for b in ratios:
    ratios[b]["reported"] = reported_ratio[b]
    ratios[b]["rounded_match"] = round(ratios[b]["recomputed"], 1) == reported_ratio[b]
results["table3_latency"] = dict(per_method=lat, speedup=ratios,
                                 v1_speedup_check=dict(dp3_v1_reported=45.6, dp3_with_2p9=round(lat["DP3"]["recomputed_avg_ms"] / 2.9, 2)))

# ---------- 6. Table 1: 절제 검정 (UCA-Flow 대 각 변형, SD n=3) ----------
T1 = {
    "Door":        dict(full=(77.3, 2.1), dual=(74.3, 5.0), cross=(68.2, 1.8), adaln=(56.5, 2.1)),
    "Pen":         dict(full=(63.2, 3.4), dual=(57.0, 3.6), cross=(58.2, 6.9), adaln=(48.6, 4.6)),
    "Dial Turn":   dict(full=(92.3, 4.7), dual=(91.3, 7.3), cross=(87.3, 10.5), adaln=(78.6, 9.3)),
    "Peg Insert":  dict(full=(90.7, 4.0), dual=(88.0, 3.6), cross=(86.3, 3.2), adaln=(61.0, 1.7)),
    "Lever Pull":  dict(full=(89.3, 3.2), dual=(89.0, 0.0), cross=(79.0, 1.0), adaln=(63.6, 6.0)),
    "Pick Place Wall": dict(full=(90.7, 3.2), dual=(89.6, 5.8), cross=(84.0, 3.4), adaln=(60.3, 11.2)),
    "Stick Pull":  dict(full=(75.3, 2.3), dual=(70.0, 3.5), cross=(62.3, 3.2), adaln=(68.0, 3.5)),
    "Shelf Place": dict(full=(74.0, 3.5), dual=(64.0, 7.8), cross=(56.6, 3.8), adaln=(32.3, 1.5)),
}
abl = {}
for task, d in T1.items():
    for var in ["dual", "cross", "adaln"]:
        t, p = welch(d["full"][0], d["full"][1], d[var][0], d[var][1])
        abl[f"{task} | full vs {var}"] = dict(diff=round(d["full"][0] - d[var][0], 1), t=round(t, 2), p=float(f"{p:.3g}"))
results["table1_ablation_welch"] = abl

# 과제를 단위로 한 대응 비교 (8개 과제, 과제 간 차이만 본다). 방향 일관성은 부호 검정으로 본다.
paired = {}
for var in ["dual", "cross", "adaln"]:
    diffs = np.array([d["full"][0] - d[var][0] for d in T1.values()])
    t, p = stats.ttest_1samp(diffs, 0.0)
    n_pos = int((diffs > 0).sum())
    sign_p = stats.binomtest(n_pos, len(diffs), 0.5).pvalue if n_pos >= 0 else None
    paired[var] = dict(mean_diff=round(float(diffs.mean()), 2), n_positive=n_pos, n=len(diffs),
                       t=round(float(t), 2), p_t=float(f"{p:.3g}"), p_sign=float(f"{sign_p:.3g}"))
results["table1_paired_over_tasks"] = paired

# 공통: 지정한 과제 집합의 전체 작업 수(8개 과제)에서 full이 모든 변형보다 높은 과제 수
results["table1_full_is_best_count"] = sum(1 for d in T1.values() if all(d["full"][0] > d[v][0] for v in ["dual", "cross", "adaln"]))

# ---------- 7. 실로봇: 20회 시도 (Table 4, 5) ----------
def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(centre - half, 3), round(centre + half, 3))


REAL = {
    # (성공 횟수, 시도 수): 논문 %에서 20회로 환산 (모든 값이 5의 배수라 정수로 떨어짐)
    "Pick&Place": {"UCA-Flow": 13, "MP1": 11, "DP3": 10, "AdaLN": 12, "CrossAttn": 10},
    "Drawer Close": {"UCA-Flow": 20, "MP1": 19, "DP3": 19, "AdaLN": 18, "CrossAttn": 19},
}
N = 20
real = {}
for task, d in REAL.items():
    real[task] = {}
    for name, k in d.items():
        real[task][name] = dict(k=k, n=N, pct=100 * k / N, wilson95=wilson(k, N))
    for other in ["MP1", "DP3", "AdaLN", "CrossAttn"]:
        table = [[d["UCA-Flow"], N - d["UCA-Flow"]], [d[other], N - d[other]]]
        odds, p = stats.fisher_exact(table)
        real[task][f"UCA-Flow vs {other} (Fisher p)"] = float(f"{p:.3g}")
results["real_world"] = real

# 검정력: 0.65 대 0.55 (Pick&Place, UCA 대 MP1), 양측 0.05, 검정력 0.8의 대략적 필요 시도 수
def n_per_arm(p1, p2, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)


results["power_real_world"] = dict(n_per_arm_65_vs_55=n_per_arm(0.65, 0.55), n_per_arm_65_vs_50=n_per_arm(0.65, 0.50))

# ---------- 출력 ----------
OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

print("[Table 2] 전체 평균 재계산 (37과제 동일 가중)")
for k, v in t2.items():
    print(f"  {k:12s} 보고 {v['reported']:5.1f}  재계산 {v['recomputed']:6.2f}  차이 {v['diff']:+.2f}  SD 보고 {v['sd_reported']}  독립 {v['sd_indep']}  완전상관 {v['sd_comonotone']}")
print("[Table 2] 전체 평균 Welch (보고된 평균 ± SD, n=3)")
for k, v in tests.items():
    print(f"  {k:28s} 차이 {v['diff']:+.2f}  t {v['t']:6.2f}  p {v['p']}")
print("[Table 2] 그룹별 UCA 대 MP1 / FlowPolicy")
for k, v in grp.items():
    if "MP1" in k or "FlowPolicy" in k:
        print(f"  {k:32s} 차이 {v['diff']:+.2f}  t {v['t']:6.2f}  p {v['p']}")
print("[v1 확인] 보고 88.2 / 재계산 %.2f, 지연 보고 2.9 / 재계산 %.3f" % (v1_avg_recomp, v1_lat_recomp))
print("[Table 3] 과제 가중 지연 (ms):", {k: v["recomputed_avg_ms"] for k, v in lat.items()})
print("[Table 3] 속도비:", {k: (v["recomputed"], v["reported"], v["rounded_match"]) for k, v in ratios.items()})
print("[Table 1] 절제 p (full 대 변형)")
for k, v in abl.items():
    print(f"  {k:36s} 차이 {v['diff']:+.1f}  p {v['p']}")
print("[Table 1] 과제 단위 대응 비교:", results["table1_paired_over_tasks"])
print("[Table 1] full이 8개 과제 모두에서 최고:", results["table1_full_is_best_count"] == 8, results["table1_full_is_best_count"])
print("[실로봇] 20회 Wilson 95% 와 Fisher")
for task, d in real.items():
    for name in ["UCA-Flow", "MP1", "DP3", "AdaLN", "CrossAttn"]:
        if name in d:
            print(f"  {task:13s} {name:10s} {d[name]['k']:2d}/20 = {d[name]['pct']:.0f}%  Wilson {d[name]['wilson95']}")
    for k, v in d.items():
        if "Fisher" in k:
            print(f"      {k}: p = {v}")
print("[검정력] 0.65 대 0.55 필요 시도 수 (군당):", results["power_real_world"]["n_per_arm_65_vs_55"])
print("[저장]", OUT)
