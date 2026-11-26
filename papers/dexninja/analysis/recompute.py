"""DexNinja (arXiv:2610.17208 v1) 논문 표 수치를 다시 계산하고 통계 검정을 한다.

입력: 본문과 부록 표의 성공 횟수를 아래에 그대로 옮겨 적었다(출처는 주석). 외부 데이터는 쓰지 않는다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, statsmodels, 수 초)

가정(논문에 없는 것은 표시한다)
- 모든 시행은 독립인 이항 시행으로 본다(Fisher 정확 검정). 논문은 같은 held-out 설정에서 정책을 비교한다고 쓰지만
  시행별 짝 자료가 없어 짝 검정은 못 한다. 짝 구조가 있으면 실제 검정력은 이보다 높을 수 있다.
- 표 1과 부록 표 2·4의 "Full/Ours" 행이 서로 다른 시행 묶음이라고 가정한다(논문은 같은 시행인지 밝히지 않는다).
- 부록 표 3의 N(추가 시뮬 에피소드 수)은 표 1의 "50 sequences"와 같은 단위라고 읽는다(N=50 행이 표 1 Sim+Real과 같다).
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}
FOODS = ["Strawberry", "Apple", "Banana", "Potato"]


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe_diff(k1, n1, k2, n2, z=1.959964):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="two-sided")[1])


def comp(name, k1, n1, k2, n2):
    d, lo, hi = newcombe_diff(k1, n1, k2, n2)
    return {"name": name, "k1": k1, "n1": n1, "k2": k2, "n2": n2, "diff_pp": round(100 * d, 1),
            "newcombe95_pp": [round(100 * lo, 1), round(100 * hi, 1)], "fisher_p": float(f"{fisher(k1, n1, k2, n2):.3g}")}


def wil(k, n):
    lo, hi = wilson(k, n)
    return [round(100 * lo, 1), round(100 * hi, 1)]


# ------------------------------------------------------------ 1. 표 1 (본문 p.8): 식품별 25회, Progress / Success / Integrity
T1 = {
    "Sim-only": {"P": [10, 5, 16, 15], "S": [6, 4, 12, 12], "I": [10, 5, 14, 11]},
    "Real-only": {"P": [12, 6, 17, 9], "S": [8, 4, 12, 7], "I": [10, 4, 14, 9]},
    "Sim+Real": {"P": [19, 14, 24, 19], "S": [12, 8, 18, 12], "I": [14, 7, 23, 16]},
}
T1_AVG = {"Sim-only": {"P": 46, "S": 34, "I": 40}, "Real-only": {"P": 44, "S": 31, "I": 37},
          "Sim+Real": {"P": 76, "S": 50, "I": 60}}
r = {}
r["sum_check"] = {reg: {m: [sum(v[m]), T1_AVG[reg][m], sum(v[m]) == T1_AVG[reg][m]] for m in "PSI"} for reg, v in T1.items()}
r["order_checks"] = {}
viol_PS, viol_SI, viol_PI = [], [], []
for reg, v in T1.items():
    for i, f in enumerate(FOODS):
        if v["S"][i] > v["P"][i]:
            viol_PS.append((reg, f))
        if v["S"][i] > v["I"][i]:
            viol_SI.append((reg, f, v["S"][i], v["I"][i]))
        if v["I"][i] > v["P"][i]:
            viol_PI.append((reg, f))
r["order_checks"] = {"success_gt_progress": viol_PS, "success_gt_integrity": viol_SI, "integrity_gt_progress": viol_PI}
r["pooled_wilson95"] = {reg: {m: [sum(v[m]), 100, wil(sum(v[m]), 100)] for m in "PSI"} for reg, v in T1.items()}
r["per_food_success_wilson95"] = {reg: {f: [v["S"][i], 25, wil(v["S"][i], 25)] for i, f in enumerate(FOODS)} for reg, v in T1.items()}
cmp1 = []
for m in "PSI":
    for a, b in [("Sim+Real", "Real-only"), ("Sim+Real", "Sim-only"), ("Sim-only", "Real-only")]:
        cmp1.append(comp(f"{m}: {a} vs {b}", sum(T1[a][m]), 100, sum(T1[b][m]), 100))
r["pooled_comparisons"] = cmp1
pf = []
for i, f in enumerate(FOODS):
    for m in "PSI":
        pf.append(comp(f"{m} {f}: Sim+Real vs Real-only", T1["Sim+Real"][m][i], 25, T1["Real-only"][m][i], 25))
ps = [x["fisher_p"] for x in pf if x["name"].startswith("S ")]
holm = multipletests(ps, method="holm")[1]
r["per_food_success_aug_vs_real"] = [x for x in pf if x["name"].startswith("S ")]
for x, h in zip(r["per_food_success_aug_vs_real"], holm):
    x["holm_p"] = round(float(h), 4)
r["per_food_progress_integrity_aug_vs_real"] = [x for x in pf if not x["name"].startswith("S ")]
# 식품으로 층화한 Mantel-Haenszel (성공, Sim+Real vs Real-only)
tabs = [np.array([[T1["Sim+Real"]["S"][i], 25 - T1["Sim+Real"]["S"][i]],
                  [T1["Real-only"]["S"][i], 25 - T1["Real-only"]["S"][i]]]) for i in range(4)]
st = StratifiedTable(tabs)
r["cmh_success_aug_vs_real"] = {"common_OR": round(float(st.oddsratio_pooled), 2),
                                "OR_ci95": [round(float(x), 2) for x in st.oddsratio_pooled_confint()],
                                "p": round(float(st.test_null_odds().pvalue), 4),
                                "breslow_day_p": round(float(st.test_equal_odds().pvalue), 3)}
tabs2 = [np.array([[T1["Sim-only"]["S"][i], 25 - T1["Sim-only"]["S"][i]],
                   [T1["Real-only"]["S"][i], 25 - T1["Real-only"]["S"][i]]]) for i in range(4)]
r["cmh_success_sim_vs_real"] = {"p": round(float(StratifiedTable(tabs2).test_null_odds().pvalue), 3)}
# 부호 검정: 4개 식품 모두에서 Sim+Real >= Real-only
r["sign_test_4of4_two_sided"] = round(2 * 0.5 ** 4, 4)
RES["table1"] = r

# ------------------------------------------------------------ 2. 표 2 (본문 p.9): 절제, 100회, P/S/I
T2 = {"Full DexNinja": [76, 50, 60], "w/o Quality Control": [72, 46, 55], "w/o Physical Randomization": [72, 44, 51],
      "w/o Trajectory Refinement": [65, 39, 53], "w/o Tactile Observation": [73, 43, 45],
      "w/o Generated Assets": [67, 42, 50]}
T2_DELTA = {"w/o Quality Control": [-4, -4, -5], "w/o Physical Randomization": [-4, -6, -9],
            "w/o Trajectory Refinement": [-11, -11, -7], "w/o Tactile Observation": [-3, -7, -15],
            "w/o Generated Assets": [-9, -8, -10]}
r = {"delta_check": {}, "vs_full": [], "vs_real_only": []}
for k, d in T2_DELTA.items():
    calc = [T2[k][j] - T2["Full DexNinja"][j] for j in range(3)]
    r["delta_check"][k] = {"calc": calc, "paper": d, "match": calc == d}
for k, v in T2.items():
    if k == "Full DexNinja":
        continue
    for j, m in enumerate("PSI"):
        r["vs_full"].append(comp(f"{m}: {k} vs Full", v[j], 100, T2["Full DexNinja"][j], 100))
    r["vs_real_only"].append(comp(f"S: {k} vs Real-only(31/100)", v[1], 100, 31, 100))
for m in "PSI":
    rows = [c for c in r["vs_full"] if c["name"].startswith(f"{m}:")]
    hh = multipletests([c["fisher_p"] for c in rows], method="holm")[1]
    for c, h in zip(rows, hh):
        c["holm_p_over_5"] = float(f"{h:.3g}")
r["max_drop_S_pp"] = 11
RES["table2"] = r

# ------------------------------------------------------------ 3. 부록 표 3 (p.24): 시뮬 에피소드 수 N, 식품별 25회 (열 순서: Strawberry, Banana, Potato, Apple)
T3_ORDER = ["Strawberry", "Banana", "Potato", "Apple"]
T3 = {10: [9, 14, 8, 5], 50: [12, 18, 12, 8], 100: [14, 20, 13, 9], 300: [16, 22, 15, 10]}
T3_AVG = {10: 36, 50: 50, 100: 56, 300: 63}
r = {"sum_check": {str(n): [sum(v), T3_AVG[n], sum(v) == T3_AVG[n]] for n, v in T3.items()}}
# N=50 행이 표 1 Sim+Real 성공과 식품별로 같은지
t1s = dict(zip(FOODS, T1["Sim+Real"]["S"]))
r["n50_equals_table1"] = {f: [T3[50][T3_ORDER.index(f)], t1s[f]] for f in FOODS}
r["pooled_wilson95"] = {str(n): [sum(v), 100, wil(sum(v), 100)] for n, v in T3.items()}
Ns = np.array(sorted(T3))
k = np.array([sum(T3[n]) for n in Ns])
x = np.log10(Ns)
# 로그 N에 대한 Cochran-Armitage 추세 검정(점수 = log10 N)과 로지스틱 회귀 기울기
nn = np.full(4, 100)
pbar = k.sum() / nn.sum()
T = np.sum(x * (k - nn * pbar))
V = pbar * (1 - pbar) * (np.sum(nn * x ** 2) - (np.sum(nn * x)) ** 2 / nn.sum())
z = T / math.sqrt(V)
r["trend_log10N"] = {"z": round(float(z), 2), "p_two_sided": float(f"{2 * (1 - stats.norm.cdf(abs(z))):.2e}")}
# 구간별 증가량 (퍼센트 포인트 / 10배)
seg = []
for a, b in [(10, 50), (50, 100), (100, 300)]:
    dpp = sum(T3[b]) - sum(T3[a])
    dec = math.log10(b / a)
    seg.append({"from": a, "to": b, "delta_pp": dpp, "per_decade_pp": round(dpp / dec, 1)})
r["segments"] = seg
r["comparisons"] = [comp("N=300 vs N=10", 63, 100, 36, 100), comp("N=300 vs N=50", 63, 100, 50, 100),
                    comp("N=100 vs N=50", 56, 100, 50, 100), comp("N=300 vs N=100", 63, 100, 56, 100),
                    comp("N=10 vs Real-only", 36, 100, 31, 100), comp("N=10 vs Sim-only", 36, 100, 34, 100),
                    comp("N=300 vs Real-only", 63, 100, 31, 100)]
# 식품별 단조 증가
r["per_food_monotone"] = {f: all(T3[a][T3_ORDER.index(f)] <= T3[b][T3_ORDER.index(f)] for a, b in [(10, 50), (50, 100), (100, 300)]) for f in T3_ORDER}
r["abstract_gt60"] = {"k": 63, "n": 100, "wilson95_pct": wil(63, 100)}
RES["table3_scaling"] = r

# ------------------------------------------------------------ 4. 부록 표 2 (p.24): RoboNinja 대 Ours, 10회씩
# 열 순서: Strawberry, Potato, Banana, Apple
T2S_ORDER = ["Strawberry", "Potato", "Banana", "Apple"]
RN, OURS = [1, 2, 2, 7], [6, 5, 9, 3]
r = {"sum": [sum(RN), sum(OURS)], "paper_sum": [12, 23], "ratio": round(sum(OURS) / sum(RN), 3),
     "diff_pp": round(100 * (sum(OURS) - sum(RN)) / 40, 1)}
r["pooled"] = comp("Ours vs RoboNinja", 23, 40, 12, 40)
r["wilson95"] = {"RoboNinja": wil(12, 40), "Ours": wil(23, 40)}
pf = [comp(f"{f}: Ours vs RoboNinja", OURS[i], 10, RN[i], 10) for i, f in enumerate(T2S_ORDER)]
hh = multipletests([c["fisher_p"] for c in pf], method="holm")[1]
for c, h in zip(pf, hh):
    c["holm_p"] = round(float(h), 3)
r["per_food"] = pf
tabs = [np.array([[OURS[i], 10 - OURS[i]], [RN[i], 10 - RN[i]]]) for i in range(4)]
st = StratifiedTable(tabs)
r["cmh"] = {"p": round(float(st.test_null_odds().pvalue), 3), "breslow_day_p": round(float(st.test_equal_odds().pvalue), 3)}
RES["supp_table2_baseline"] = r

# ------------------------------------------------------------ 5. 부록 표 4 (p.25): 머리 구조와 청크, 10회씩. 열 순서: Apple, Banana, Strawberry, Potato
T4_ORDER = ["Apple", "Banana", "Strawberry", "Potato"]
T4 = {"Universal Head": [2, 4, 2, 3], "Separate Heads": [4, 7, 5, 5], "Fixed Action Chunk": [1, 6, 2, 4], "Full System": [5, 7, 7, 5]}
T4_PAPER = {"Universal Head": 11, "Separate Heads": 21, "Fixed Action Chunk": 13, "Full System": 24}
r = {"sum_check": {k: [sum(v), T4_PAPER[k], sum(v) == T4_PAPER[k]] for k, v in T4.items()},
     "wilson95": {k: wil(sum(v), 40) for k, v in T4.items()}}
r["comparisons"] = [comp("Full vs Universal", 24, 40, 11, 40), comp("Full vs Separate", 24, 40, 21, 40),
                    comp("Full vs Fixed", 24, 40, 13, 40), comp("Separate vs Universal", 21, 40, 11, 40),
                    comp("Separate vs Fixed", 21, 40, 13, 40)]
RES["supp_table4_heads_chunk"] = r

# ------------------------------------------------------------ 6. 같은 완성 시스템의 세 측정 (표 1 / 부록 표 2 / 부록 표 4)
# 식품별 성공: 표 1 (25회), 부록 표 2 Ours (10회), 부록 표 4 Full (10회)
t1 = dict(zip(FOODS, T1["Sim+Real"]["S"]))
t2s = dict(zip(T2S_ORDER, OURS))
t4 = dict(zip(T4_ORDER, T4["Full System"]))
r = {"per_food": {}}
for f in FOODS:
    cells = [(t1[f], 25), (t2s[f], 10), (t4[f], 10)]
    tab = np.array([[a, n - a] for a, n in cells])
    chi2, p, _, _ = stats.chi2_contingency(tab, correction=False)
    r["per_food"][f] = {"counts": [f"{a}/{n}" for a, n in cells], "rates_pct": [round(100 * a / n, 1) for a, n in cells],
                        "chi2_p": round(float(p), 3)}
r["pooled"] = {"table1": [50, 100], "supp2": [23, 40], "supp4": [24, 40],
               "table1_vs_supp2_supp4_fisher": round(fisher(50, 100, 47, 80), 3),
               "supp2_vs_supp4_fisher": round(fisher(23, 40, 24, 40), 3)}
r["pooled"]["combined_47_80_pct"] = round(100 * 47 / 80, 1)
RES["cross_table_same_system"] = r

# ------------------------------------------------------------ 7. 레몬 (본문 p.8, 그림 5): 단일 8/10, 2회 7/10, 다중 3/10
L = {"single": 8, "two": 7, "multi": 3}
r = {"wilson95": {k: wil(v, 10) for k, v in L.items()},
     "single_vs_two_fisher": round(fisher(8, 10, 7, 10), 3), "two_vs_multi_fisher": round(fisher(7, 10, 3, 10), 3),
     "single_vs_multi_fisher": round(fisher(8, 10, 3, 10), 3)}
RES["lemon"] = r

# ------------------------------------------------------------ 8. 자산 깔때기 (본문 표 4)
F = {"Strawberry": [397, 361, 326, 300], "Apple": [376, 349, 321, 300], "Banana": [453, 407, 354, 300], "Potato": [444, 396, 343, 300]}
tot = [sum(F[f][j] for f in F) for j in range(4)]
r = {"column_sums": tot, "paper_totals": [1670, 1513, 1344, 1200], "match": tot == [1670, 1513, 1344, 1200]}
r["stage_pass_rates_pct"] = {f: [round(100 * F[f][j + 1] / F[f][j], 1) for j in range(3)] for f in F}
r["total_stage_pass_rates_pct"] = [round(100 * tot[j + 1] / tot[j], 1) for j in range(3)]
r["accepted_over_proposed_pct"] = round(100 * 1200 / 1670, 1)
r["accepted_equals_300_each"] = all(F[f][3] == 300 for f in F)
r["surplus_after_3d_qc"] = {f: F[f][2] - 300 for f in F}
RES["asset_funnel"] = r

# ------------------------------------------------------------ 9. 실패 분석 (본문 표 3)
fail = {"Perception/registration": 8, "Grasp/contact control": 19, "Cutting-physics mismatch": 17, "Kinematic/safety/collision": 6}
r = {"sum": sum(fail.values()), "failed_trials_expected": 100 - 50, "match": sum(fail.values()) == 50,
     "progress_failures": 100 - 76, "progress_but_not_success": 76 - 50,
     "note": "Progress 실패 24회는 지각·파지·운동학 원인 합 33회 안에 들어가고, Progress 후 실패 26회에는 물리 불일치 17회가 포함될 수 있다(모순 없음)."}
RES["failure_table"] = r

# ------------------------------------------------------------ 10. 시행 수 합계와 검정력
trials = {"Table 1 (3 regimes x 100)": 300, "Table 2 ablations (5 variants x 100, Full is Table 1)": 500,
          "Supp Table 3 scaling (N=10,100,300 x 100; N=50 is Table 1)": 300, "Supp Table 2 (RoboNinja 40 + Ours 40)": 80,
          "Supp Table 4 (4 rows x 40)": 160, "Lemon (3 x 10)": 30}
RES["trial_budget"] = {"items": trials, "total_if_all_distinct": sum(trials.values())}


def power_two_prop(p1, p2, n, alpha=0.05):
    """정규 근사 검정력 (양측)."""
    pb = (p1 + p2) / 2
    se0 = math.sqrt(2 * pb * (1 - pb) / n)
    se1 = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    za = stats.norm.ppf(1 - alpha / 2)
    d = abs(p1 - p2)
    return float(stats.norm.cdf((d - za * se0) / se1))


def n_for_power(p1, p2, power=0.8, alpha=0.05):
    for n in range(5, 2000):
        if power_two_prop(p1, p2, n, alpha) >= power:
            return n


pw = {}
pw["success_0.50_vs_0.31_n100"] = round(power_two_prop(0.50, 0.31, 100), 2)
pw["success_0.50_vs_0.31_n_for_80pct"] = n_for_power(0.50, 0.31)
pw["success_0.48_vs_0.32_n25"] = round(power_two_prop(0.48, 0.32, 25), 2)
pw["success_0.48_vs_0.32_n_for_80pct"] = n_for_power(0.48, 0.32)
pw["success_0.50_vs_0.43_n100"] = round(power_two_prop(0.50, 0.43, 100), 2)
pw["success_0.50_vs_0.43_n_for_80pct"] = n_for_power(0.50, 0.43)
pw["success_0.50_vs_0.39_n100"] = round(power_two_prop(0.50, 0.39, 100), 2)
pw["success_0.60_vs_0.525_n40"] = round(power_two_prop(0.60, 0.525, 40), 2)
pw["success_0.575_vs_0.30_n40"] = round(power_two_prop(0.575, 0.30, 40), 2)
RES["power"] = pw

# ------------------------------------------------------------ 11. 비용 식 (부록 5.2)
cost = {"gpu_hours_per_single_cut_episode": 10 / 60, "formula_N_over_6": [round(n / 6, 2) for n in (10, 50, 100, 300)],
        "memory_GB_M1_M2": [10 + 10 * 1, 10 + 10 * 2], "paper_memory_GB": [20, 30],
        "time_min_M1_M2": [10 * 1, 10 * 2], "paper_time_min": [10, 20]}
RES["cost_formula"] = cost

# ------------------------------------------------------------ 12. 요약 주장 (초록과 프로젝트 페이지)
RES["abstract_claims"] = {
    "gain_pp_main": 50 - 31, "gain_relative_pct_main": round(100 * (50 / 31 - 1), 1),
    "gain_pp_N300_vs_real_only": 63 - 31, "abstract_says": "more than 20% (real-only 대비)",
    "main_success_pct": 50, "scaling_success_pct": 63,
}

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")


def show(title, rows):
    print("==", title)
    for x in rows:
        print("  ", x["name"], f"{x['k1']}/{x['n1']} vs {x['k2']}/{x['n2']}", f"{x['diff_pp']:+} pp", x["newcombe95_pp"], "p =", x["fisher_p"])


show("T1 pooled", RES["table1"]["pooled_comparisons"])
show("T1 per food S", RES["table1"]["per_food_success_aug_vs_real"])
print("CMH", RES["table1"]["cmh_success_aug_vs_real"])
show("T2 vs full", RES["table2"]["vs_full"])
show("T3", RES["table3_scaling"]["comparisons"])
print("trend", RES["table3_scaling"]["trend_log10N"], RES["table3_scaling"]["segments"])
show("suppT2", [RES["supp_table2_baseline"]["pooled"]] + RES["supp_table2_baseline"]["per_food"])
show("suppT4", RES["supp_table4_heads_chunk"]["comparisons"])
print(json.dumps(RES["cross_table_same_system"], ensure_ascii=False))
print(json.dumps(RES["lemon"]), json.dumps(RES["power"]))
print(json.dumps(RES["asset_funnel"], ensure_ascii=False))
print("wrote", OUT)
