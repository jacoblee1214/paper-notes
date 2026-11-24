"""논문 표·본문 수치 재계산과 일관성 점검 (V1-V5, V9). 재현이 아니라 보고된 수치의 산술 점검이다.

입력은 논문 표 값을 그대로 옮긴 것(표 번호 주석). 시행 수는 논문에 명시된 값만 쓰고,
명시되지 않은 것은 "(가정)"으로 표시한다. 결과: recompute_results.json
실행: python3 recompute.py  (CPU, 1초 미만)
"""
import json
import math
from itertools import combinations
from pathlib import Path

OUT = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def fisher_two_sided(a, b, c, d):
    """2x2 [[a,b],[c,d]] Fisher 정확검정 (양측)."""
    n1, n2, k = a + b, c + d, a + c
    N = n1 + n2

    def pmf(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(N, k)
    p0 = pmf(a)
    return sum(pmf(x) for x in range(max(0, k - n2), min(n1, k) + 1) if pmf(x) <= p0 * (1 + 1e-9))


# ---------------- 표 1 (단일 과제 정책, Iso / Ours / Oracle) ----------------
T1 = {
    "Revolute":    dict(SR=(99, 99, 99), SE=1.64, ICF=(7.18, 2.62, 2.13), PCF=(3.38, 1.56, 1.48), ICM=(.29, .22, .22), PCM=(.19, .18, .17)),
    "Cylindrical": dict(SR=(99, 100, 100), SE=1.73, ICF=(5.31, 0.83, 0.53), PCF=(1.96, 0.36, 0.31), ICM=(.32, .22, .21), PCM=(.16, .14, .14)),
    "Planar":      dict(SR=(99, 99, 99), SE=2.78, ICF=(15.3, 1.93, 2.20), PCF=(4.03, 1.18, 1.16), ICM=(.74, .48, .34), PCM=(.29, .23, .21)),
    "Universal":   dict(SR=(100, 100, 100), SE=4.36, ICF=(2.18, 0.46, 0.34), PCF=(1.25, 0.23, 0.19), ICM=None, PCM=None),
}
# 표 10 (다중 과제 정책)
T10 = {
    "Revolute":    dict(SR=(99, 99, 99), SE=1.71, ICF=(7.75, 2.87, 2.55), PCF=(3.55, 1.76, 1.81), ICM=(.31, .26, .26), PCM=(.21, .22, .22)),
    "Cylindrical": dict(SR=(96, 97, 96), SE=2.39, ICF=(4.61, 1.36, 0.64), PCF=(1.74, 0.67, 0.51), ICM=(.31, .20, .21), PCM=(.15, .13, .15)),
    "Planar":      dict(SR=(100, 100, 100), SE=2.76, ICF=(14.4, 2.02, 2.21), PCF=(3.85, 1.13, 1.13), ICM=(.73, .49, .34), PCM=(.29, .24, .22)),
    "Universal":   dict(SR=(99, 100, 100), SE=3.83, ICF=(1.98, 0.45, 0.33), PCF=(1.17, 0.23, 0.19), ICM=None, PCM=None),
}
# 표 9 n_c
NC1 = {"Revolute": 99, "Cylindrical": 99, "Planar": 99, "Universal": 100}
NC10 = {"Revolute": 99, "Cylindrical": 95, "Planar": 100, "Universal": 99}


def sim_block(T, label):
    res = {"ratio_ours_over_iso": {}, "oracle_recovery": {}, "sr_wilson95": {}}
    for task, r in T.items():
        res["ratio_ours_over_iso"][task] = {m: round(r[m][1] / r[m][0], 3) for m in ["ICF", "PCF", "ICM", "PCM"] if r[m]}
        res["oracle_recovery"][task] = {m: round((r[m][0] - r[m][1]) / (r[m][0] - r[m][2]), 3) for m in ["ICF", "PCF"]}
        res["sr_wilson95"][task] = [tuple(round(100 * x, 1) for x in wilson(k, 100)) for k in r["SR"]]
    f = [res["ratio_ours_over_iso"][t]["ICF"] for t in T]
    p = [res["ratio_ours_over_iso"][t]["PCF"] for t in T]
    res["range_ICF"] = (min(f), max(f))
    res["range_PCF"] = (min(p), max(p))
    res["min_recovery"] = min(min(v.values()) for v in res["oracle_recovery"].values())
    res["SE_mean_over_objects"] = round(sum(r["SE"] for r in T.values()) / 4, 3)
    return res


OUT["V1_table1"] = sim_block(T1, "T1")
OUT["V1_table10"] = sim_block(T10, "T10")
# SR 합계와 "398 trials" 서술
OUT["V1_sr_totals"] = {c: sum(T1[t]["SR"][i] for t in T1) for i, c in enumerate(["Iso", "Ours", "Oracle"])}
OUT["V1_nc_le_min_SR"] = {"T1": {t: (NC1[t], min(T1[t]["SR"])) for t in T1}, "T10": {t: (NC10[t], min(T10[t]["SR"])) for t in T10}}
# SR 차이 Fisher (Iso vs Ours): 100 시행씩 (논문 명시)
OUT["V1_fisher_SR"] = {t: round(fisher_two_sided(T1[t]["SR"][0], 100 - T1[t]["SR"][0], T1[t]["SR"][1], 100 - T1[t]["SR"][1]), 3) for t in T1}
OUT["V1_text_claims"] = {
    "text_ICF_range": (0.13, 0.37), "text_PCF_range": (0.18, 0.46), "text_min_recovery_T1": 0.90,
    "text_mt_ICF_range": (0.14, 0.37), "text_mt_PCF_range": (0.20, 0.49), "text_min_recovery_T10": 0.80,
}
def decimals(x):
    t = repr(x)
    return len(t.split(".")[1]) if "." in t else 0


def ratio_feasible(a, b, ratio):
    """출력된 a, b(반올림)와 비율(소수 2자리) 사이에 반올림 오차 범위 내 모순이 없는가."""
    ha, hb = 0.5 * 10 ** -decimals(a), 0.5 * 10 ** -decimals(b)
    lo, hi = (b - hb) / (a + ha), (b + hb) / (a - ha)
    return lo - 0.005 <= ratio <= hi + 0.005


# ---------------- 표 2 (실로봇, Iso, Ours) 10회씩 ----------------
T2 = {
    "Revolute":    dict(SE=13.01, ICF=(46.6, 25.2, .54), PCF=(8.54, 3.62, .42), ICM=(5.00, 3.34, .67), PCM=(.93, .37, .40)),
    "Prismatic":   dict(SE=4.74, ICF=(12.3, 8.79, .72), PCF=(4.63, 3.51, .76), ICM=(1.84, 1.46, .80), PCM=(.61, .55, .90)),
    "Helical":     dict(SE=2.89, ICF=(6.96, 6.50, .93), PCF=(4.56, 4.40, .96), ICM=(1.82, .60, .33), PCM=(1.08, .45, .41)),
    "Cylindrical": dict(SE=5.76, ICF=(8.35, 4.42, .53), PCF=(3.37, 2.20, .65), ICM=(1.09, .82, .75), PCM=(.45, .37, .81)),
    "Crank":       dict(SE=5.70, ICF=(20.8, 23.6, 1.14), PCF=(4.62, 4.28, .93), ICM=(4.99, 4.15, .83), PCM=(1.12, .82, .74)),
}
t2 = {"printed_ratio_mismatch": {}, "avg": {}}
for task, r in T2.items():
    for m in ["ICF", "PCF", "ICM", "PCM"]:
        calc = r[m][1] / r[m][0]
        if abs(calc - r[m][2]) > 0.0051:
            t2["printed_ratio_off_by_more_than_0.005"] = t2.get("printed_ratio_off_by_more_than_0.005", {})
            t2["printed_ratio_off_by_more_than_0.005"][f"{task}.{m}"] = (round(calc, 4), r[m][2])
        if not ratio_feasible(r[m][0], r[m][1], r[m][2]):
            t2["printed_ratio_mismatch"][f"{task}.{m}"] = (round(calc, 4), r[m][2])
        # 반올림 범위로 오차 허용 여부
for m, tasks in {"ICF": list(T2), "PCF": list(T2), "ICM": [t for t in T2 if t != "Prismatic"], "PCM": [t for t in T2 if t != "Prismatic"]}.items():
    iso = sum(T2[t][m][0] for t in tasks) / len(tasks)
    ours = sum(T2[t][m][1] for t in tasks) / len(tasks)
    mean_ratio = sum(T2[t][m][1] / T2[t][m][0] for t in tasks) / len(tasks)
    t2["avg"][m] = dict(ratio_of_means=round(ours / iso, 3), mean_of_ratios=round(mean_ratio, 3))
t2["text"] = {"ICF": 0.72, "PCF": 0.70, "ICM": 0.69, "PCM": 0.56}
cfg = [t for t in T2 if t != "Crank"]
t2["cfg_indep_ICF_range"] = (min(T2[t]["ICF"][2] for t in cfg), max(T2[t]["ICF"][2] for t in cfg))
t2["cfg_indep_PCF_range"] = (min(T2[t]["PCF"][2] for t in cfg), max(T2[t]["PCF"][2] for t in cfg))
t2["all_cfg_indep_improved"] = all(T2[t][m][1] < T2[t][m][0] for t in cfg for m in ["ICF", "PCF", "ICM", "PCM"])
t2["SE_text_range_4objects"] = (min(T2[t]["SE"] for t in T2 if t != "Revolute"), max(T2[t]["SE"] for t in T2 if t != "Revolute"))
OUT["V2_table2"] = t2
# 실로봇 과제당 10회 대 10회: 시행별 변동계수(CV)를 가정했을 때 비율 평균이 구분 가능한 최소 크기 (가정이며 원자료 아님)
from scipy import stats  # noqa: E402

mdr = {}
for cv in (0.2, 0.3, 0.5):
    se = math.sqrt(2 * math.log(1 + cv ** 2) / 10)  # 로그정규 가정, 독립 두 군 10회씩
    tcrit = stats.t.ppf(0.975, 18)
    mdr[f"cv={cv}"] = round(math.exp(-tcrit * se), 3)
OUT["V2_min_detectable_ratio_n10_each_assumed_cv"] = mdr
# ---------------- 표 14 ----------------
T14 = {
    "Revolute":    dict(lf=(.766, .672, .88), lm=(.734, .522, .71), bf=(.738, .843, 1.14), bm=(.717, .848, 1.18)),
    "Prismatic":   dict(lf=(.362, .321, .89), lm=(1.0, 1.0, 1.0), bf=(.848, .918, 1.08), bm=(0.0, 0.0, None)),
    "Helical":     dict(lf=(.412, .424, 1.03), lm=(.724, .569, .79), bf=(.187, .332, 1.78), bm=(.751, .890, 1.18)),
    "Cylindrical": dict(lf=(.244, .126, .52), lm=(.536, .421, .79), bf=(.708, .894, 1.26), bm=(.572, .680, 1.19)),
    "Crank":       dict(lf=(.425, .469, 1.10), lm=(.926, .850, .92), bf=(.841, .892, 1.06), bm=(.261, .550, 2.11)),
}
t14 = {"printed_ratio_mismatch": {}, "avg": {}}
for task, r in T14.items():
    for k, (a, b, c) in r.items():
        if c is not None and not ratio_feasible(a, b, c):
            t14["printed_ratio_mismatch"][f"{task}.{k}"] = (round(b / a, 4), c)
for k, text in {"lf": 0.91, "lm": 0.81, "bf": 1.17, "bm": 1.29}.items():
    tasks = [t for t in T14 if not (t == "Prismatic" and k in ("lm", "bm"))]
    iso = sum(T14[t][k][0] for t in tasks) / len(tasks)
    ours = sum(T14[t][k][1] for t in tasks) / len(tasks)
    mr = sum(T14[t][k][1] / T14[t][k][0] for t in tasks) / len(tasks)
    t14["avg"][k] = dict(text=text, ratio_of_means=round(ours / iso, 3), mean_of_ratios=round(mr, 3))
OUT["V3_table14"] = t14
# 함의된 feasible 방향 임펄스 J_par = ICF (1-lambda_f)/lambda_f (표 1/2 x 표 9/14). 평균의 비 != 비의 평균이므로 대략치.
j = {}
LAM9 = {"Revolute": (.26, .12, .10), "Cylindrical": (.15, .03, .02), "Planar": (.68, .21, .24), "Universal": (.31, .14, .11)}
for t, lam in LAM9.items():
    j[f"sim_{t}"] = [round(T1[t]["ICF"][i] * (1 - lam[i]) / lam[i], 2) for i in range(3)]
for t in T2:
    j[f"real_{t}"] = [round(T2[t]["ICF"][i] * (1 - T14[t]["lf"][i]) / T14[t]["lf"][i], 2) for i in range(2)]
OUT["V4_implied_feasible_impulse_Jpar"] = j
# ---------------- 표 11-13 서술과 표 ----------------
T12 = {"DTW-max": {"SE": [5.91, 3.32, 2.74, 2.61, 2.48, 2.49, 2.34], "pct": [16.0, 8.8, 2.8, 2.3, 1.8, 1.0, 1.0]},
       "DTW-min": {"SE": [19.73, 9.93, 4.45, 3.43, 2.58, 2.43, 2.34], "pct": [40.5, 28.0, 13.8, 9.5, 1.5, 1.5, 1.0]}}
KS = [2, 4, 8, 16, 32, 64, 128]
# 백분율이 정수 시행 수의 비율로 표현 가능한 분모 (소수 1자리, 사사오입 허용)
def feasible_denoms(values, ndigits, upto=2000):
    ok = []
    for n in range(1, upto + 1):
        if all(any(abs(100 * c / n - v) <= 0.5 * 10 ** -ndigits + 1e-9 for c in range(n + 1)) for v in values):
            ok.append(n)
    return ok
allp = T12["DTW-max"]["pct"] + T12["DTW-min"]["pct"]
OUT["V5_table12_pct_feasible_denominators_le_1000"] = feasible_denoms(allp, 1, 1000)[:12]
T11C = [64.7, 84.2, 92.8, 97.7, 99.4, 99.8, 100, 100]
OUT["V5_table11_correct_dim_feasible_denominators_le_2000"] = feasible_denoms(T11C, 1, 2000)[:12]
OUT["V5_table12_text_vs_table"] = {
    "text_diverse_se": [5.91, 3.32, 2.74], "table_diverse_se": T12["DTW-max"]["SE"][:3],
    "text_pct_diverse": [16, 9, 3], "table": T12["DTW-max"]["pct"][:3],
    "text_least_se": [19.7, 9.9, 4.4], "table": T12["DTW-min"]["SE"][:3],
    "text_pct_least": [41, 28, 14], "table_least": T12["DTW-min"]["pct"][:3],
}
# 표 13 비율
OUT["V5_table13_ratio"] = {"position": round(3.10 / 11.97, 3), "orientation": round(10.59 / 16.05, 3)}
OUT["V5_table13_clean_baselines"] = {"a_direct_none": 2.14, "a_trained_none": 2.02, "b_none": 2.10, "c_ddim10": 2.14, "c_ddim100": 2.16, "table12_k128_mean_of_median": 2.34, "table1_SE_mean": OUT["V1_table1"]["SE_mean_over_objects"]}
# ---------------- 표 6 게인 ----------------
def gains(m, I, k0, r, zeta):
    return dict(ell=round(math.sqrt(I / m), 3), k_perp=k0 * r, d_par=round(2 * zeta * math.sqrt(k0), 2), d_perp=round(2 * zeta * math.sqrt(k0 * r), 2),
                zeta_over_sqrt_m=round(zeta / math.sqrt(m), 3))
OUT["V9_table6"] = {"sim": gains(0.586, 0.00622, 250, 0.1, 0.766), "real": gains(7.70, 0.724, 400, 0.1, 2.775),
                     "paper": {"sim": dict(ell=0.103, d_par=24.21, d_perp=7.66), "real": dict(ell=0.307, d_par=111, d_perp=35.1)}}
# ---------------- 시행 수 ----------------
OUT["V9_counts"] = {"real_trials_per_controller": 5 * 10, "real_demos": 3 * 20 + 2 * 32, "sim_trials_per_controller_single": 4 * 100,
                    "sim_grasp_success_trials_ours": sum(T1[t]["SR"][1] for t in T1)}
OUT["tables"] = {"T1": T1, "T10": T10, "T2": T2, "T14": T14, "T12": T12, "T11": {"N": [1, 2, 4, 8, 16, 32, 64, 128], "se95": [78.95, 23.15, 8.59, 7.23, 6.92, 6.79, 6.68, 6.61], "correct_dim_pct": T11C}}
Path(__file__).with_name("recompute_results.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(OUT, indent=1, ensure_ascii=False))
