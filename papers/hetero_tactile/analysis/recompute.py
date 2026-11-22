"""HTT (arXiv:2606.29948 v2) 보고 수치 재계산과 일관성 검사.

실행: python3 analysis/recompute.py  → analysis/recompute_results.json (수 초, numpy/scipy/statsmodels)

입력은 논문 표 1, 2, 3, 4, 5, 6과 본문 수치, 공개 데이터셋 카드의 통계(STATS.json)를 옮긴 상수입니다.
"±"는 시드 3개(10, 20, 30, 부록 C)의 표준편차라고 가정합니다(표본 SD, ddof=1). ddof=0이면 SD가 sqrt(2/3)배라 p가 더 작아집니다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

OUT = Path(__file__).with_name("recompute_results.json")
R = {}

SENS = ["Xela", "TAC-02", "9DTact", "GSMini"]

# ---------------------------------------------------------------- 표 1 (정확도 %, 평균, SD)
T1 = {
    "Scratch": [(48.90, 1.45), (22.49, 0.78), (65.63, 1.73), (53.13, 2.60), (47.54, 1.64)],
    "T3": [None, None, (51.44, 26.25), (59.26, 5.25), (55.35, 15.75)],
    "SITR": [None, None, (81.34, 4.41), (74.31, 7.14), (77.83, 5.78)],
    "AnyTouch": [None, None, (71.07, 0.51), (76.36, 5.84), (73.72, 3.18)],
    "SPH": [(49.72, 0.99), (25.29, 1.86), (82.23, 4.05), (75.86, 3.91), (58.28, 2.70)],
    "MAE": [(56.68, 0.99), (26.16, 0.43), (90.08, 0.54), (88.59, 0.71), (65.38, 0.67)],
    "HTT": [(52.41, 0.84), (26.20, 1.60), (94.84, 1.61), (91.35, 1.08), (66.20, 1.28)],
}

# ---------------------------------------------------------------- 표 5 (힘 MAE N): 3D, 수직, 전단 (평균, SD)
T5 = {  # sensor -> method -> (3D, normal, shear)
    "Xela": {"Scratch": ((1.225, .218), (2.628, .642), (.524, .007)), "SPH": ((.838, .018), (1.779, .046), (.368, .005)),
             "MAE": ((.762, .051), (1.606, .147), (.340, .007)), "HTT": ((.695, .033), (1.376, .104), (.355, .005))},
    "TAC-02": {"Scratch": ((.705, .002), (2.125, .256), (.286, .002)), "SPH": ((.555, .007), (1.146, .020), (.260, .002)),
               "MAE": ((.516, .022), (1.040, .060), (.255, .004)), "HTT": ((.508, .018), (1.009, .049), (.257, .002))},
    "9DTact": {"Scratch": ((1.255, .258), (2.426, .741), (.670, .033)), "T3": ((2.678, .468), (6.265, 1.069), (.884, .171)),
               "SITR": ((1.085, .366), (2.330, .866), (.462, .117)), "AnyTouch": ((1.233, .273), (2.679, .617), (.511, .101)),
               "SPH": ((.780, .029), (1.572, .084), (.384, .002)), "MAE": ((.574, .005), (1.042, .013), (.339, .002)),
               "HTT": ((.606, .006), (1.086, .009), (.366, .007))},
    "GSMini": {"Scratch": ((1.260, .021), (2.503, .067), (.639, .012)), "T3": ((1.197, .164), (2.301, .353), (.645, .105)),
               "SITR": ((1.373, .148), (2.357, .467), (.881, .050)), "AnyTouch": ((1.401, .242), (2.466, .544), (.868, .093)),
               "SPH": ((.845, .009), (1.427, .018), (.554, .011)), "MAE": ((.803, .092), (1.248, .180), (.581, .048)),
               "HTT": ((.736, .072), (1.045, .171), (.582, .024))},
}
# 표 2 힘 (3D MAE) 와 overall
T2_FORCE = {
    "Scratch": [1.225, .705, 1.255, 1.260, 1.111], "T3": [None, None, 2.678, 1.197, 1.938],
    "SITR": [None, None, 1.085, 1.373, 1.229], "AnyTouch": [None, None, 1.233, 1.401, 1.317],
    "SPH": [.838, .555, .780, .845, .755], "MAE": [.762, .516, .574, .803, .664], "HTT": [.695, .508, .606, .736, .636],
}
T2_SLIP = {
    "Scratch": [29.80, 31.77, 31.00, 32.00, 31.14], "T3": [None, None, 31.27, 38.25, 34.76],
    "SITR": [None, None, 41.77, 36.36, 39.07], "AnyTouch": [None, None, 31.23, 58.75, 44.99],
    "SPH": [54.82, 31.79, 49.13, 71.86, 51.90], "MAE": [54.46, 33.45, 49.70, 68.86, 51.62], "HTT": [54.21, 45.45, 53.09, 72.65, 56.35],
}
# 표 6 (슬립 F1 %): sensor -> method -> {macro:(m,sd), static, incipient, slide}
T6 = {
    "Xela": {"Scratch": ((29.80, 0), (0, 0), (0, 0), (89.4, 0)), "SPH": ((54.82, 1.18), (68.79, 3.51), (2.38, 1.89), (93.29, 1.60)),
             "MAE": ((54.46, .36), (64.01, .76), (5.99, .15), (93.39, .18)), "HTT": ((54.21, .79), (63.7, 0.0), (6.2, 0.0), (92.7, 0.0))},
    "TAC-02": {"Scratch": ((31.77, 0), (0, 0), (0, 0), (95.3, 0)), "SPH": ((31.79, .25), (.22, .16), (6.78, .19), (88.37, .40)),
               "MAE": ((33.45, .48), (2.79, 1.30), (7.63, .08), (89.93, .09)), "HTT": ((45.45, 7.27), (41.3, 24.4), (3.8, 3.8), (91.3, .6))},
    "9DTact": {"Scratch": ((31.00, 0), (0, 0), (0, 0), (94.0, 0)), "T3": ((31.27, .03), (0, 0), (0, 0), (93.8, .1)),
               "SITR": ((41.77, 6.69), (14.7, 15.5), (19.2, 2.1), (91.5, .7)), "AnyTouch": ((31.23, 0), (0, 0), (0, 0), (93.68, 0)),
               "SPH": ((49.13, 3.45), (34.69, 9.22), (22.86, 1.09), (89.83, .20)), "MAE": ((49.70, .42), (40.62, 1.07), (19.74, 1.24), (88.74, .42)),
               "HTT": ((53.09, .63), (39.1, 3.6), (30.6, 3.8), (89.6, 1.3))},
    "GSMini": {"Scratch": ((32.00, 0), (0, 0), (0, 0), (95.0, 0)), "T3": ((38.25, .41), (4.7, .5), (15.3, 1.8), (94.8, 1.1)),
               "SITR": ((36.36, .12), (0, 0), (13.1, .2), (96.0, .1)), "AnyTouch": ((58.75, 1.23), (78.32, 5.92), (1.79, 2.53), (96.13, .31)),
               "SPH": ((71.86, .24), (81.82, .28), (37.76, .23), (96.01, .26)), "MAE": ((68.86, 1.93), (80.89, 1.79), (30.66, 4.19), (95.04, .72)),
               "HTT": ((72.65, .38), (83.02, .34), (38.62, 1.22), (96.31, .16))},
}


def welch(m1, s1, m2, s2, n=3, ddof0=False):
    if ddof0:
        s1, s2 = s1 * math.sqrt(n / (n - 1)), s2 * math.sqrt(n / (n - 1))
    if s1 == 0 and s2 == 0:
        return None
    t, p = stats.ttest_ind_from_stats(m1, s1, n, m2, s2, n, equal_var=False)
    return float(t), float(p)


# ================================================================ V1 표 1 overall 과 인용 차이
v1 = {"overall_check": {}, "std_rule": {}}
for k, rows in T1.items():
    vals = [r[0] for r in rows[:4] if r]
    sds = [r[1] for r in rows[:4] if r]
    v1["overall_check"][k] = {"reported": rows[4][0], "recomputed": round(float(np.mean(vals)), 2)}
    v1["std_rule"][k] = {"reported": rows[4][1], "mean_of_sds": round(float(np.mean(sds)), 2),
                         "sd_of_mean_if_indep": round(float(np.sqrt(np.sum(np.square(sds))) / len(sds)), 2)}
cit = {
    "HTT_vs_SPH_9DTact_pp": round(94.84 - 82.23, 2), "HTT_vs_AnyTouch_GSMini_pp": round(91.35 - 76.36, 2),
    "HTT_vs_MAE_9DTact": round(94.84 - 90.08, 2), "HTT_vs_MAE_GSMini": round(91.35 - 88.59, 2), "HTT_vs_MAE_Xela": round(52.41 - 56.68, 2),
    "v1_HTT_vs_SITR_9DTact": round(94.84 - 81.34, 2), "v1_HTT_vs_SITR_GSMini": round(91.35 - 74.31, 2),
    "slip_TAC02_HTT_vs_MAE": round(45.45 - 33.45, 2),
}
v1["cited_differences"] = cit
R["V1_table1"] = v1

# ================================================================ V2 표 2 대 표 5·6 일치, 3D = (2*shear+normal)/3
v2 = {"table2_force_vs_table5": [], "force3d_identity": [], "table2_slip_vs_table6_macro": [], "macro_equals_mean_of_classes": [],
      "table2_force_overall": {}, "table2_slip_overall": {}}
for s_i, s in enumerate(SENS):
    for meth, ((d3, _), (nm, _), (sh, _)) in T5[s].items():
        t2 = T2_FORCE[meth][s_i]
        v2["table2_force_vs_table5"].append({"sensor": s, "method": meth, "table2": t2, "table5_3d": d3, "match": abs(t2 - d3) < 1e-9})
        ident = (2 * sh + nm) / 3
        v2["force3d_identity"].append({"sensor": s, "method": meth, "table5_3d": d3, "(2*shear+normal)/3": round(ident, 3),
                                        "abs_diff": round(abs(ident - d3), 4)})
    for meth, rec in T6[s].items():
        macro = rec[0][0]
        mean_cls = (rec[1][0] + rec[2][0] + rec[3][0]) / 3
        t2 = T2_SLIP[meth][s_i]
        v2["table2_slip_vs_table6_macro"].append({"sensor": s, "method": meth, "table2": t2, "table6": macro, "match": abs(t2 - macro) < 1e-9})
        v2["macro_equals_mean_of_classes"].append({"sensor": s, "method": meth, "macro": macro, "mean_of_3_class_f1": round(mean_cls, 2),
                                                   "abs_diff": round(abs(mean_cls - macro), 2),
                                                   "macro_sd": rec[0][1], "class_sds": [rec[1][1], rec[2][1], rec[3][1]]})
for name, T in (("table2_force_overall", T2_FORCE), ("table2_slip_overall", T2_SLIP)):
    for meth, rows in T.items():
        vals = [x for x in rows[:4] if x is not None]
        v2[name][meth] = {"reported": rows[4], "recomputed": round(float(np.mean(vals)), 3 if name == "table2_force_overall" else 2)}
v2["force3d_max_abs_diff"] = max(x["abs_diff"] for x in v2["force3d_identity"])
v2["macro_max_abs_diff"] = max(x["abs_diff"] for x in v2["macro_equals_mean_of_classes"])
# 클래스 F1의 SD가 모두 0인데 macro-F1 SD가 0이 아닌 칸
v2["zero_class_sd_but_nonzero_macro_sd"] = [
    {"sensor": x["sensor"], "method": x["method"], "macro_sd": x["macro_sd"], "class_sds": x["class_sds"]}
    for x in v2["macro_equals_mean_of_classes"] if all(c == 0 for c in x["class_sds"]) and x["macro_sd"] > 0]
# 스크래치 붕괴 검사: 항상 slide 예측이면 macro = slide F1 / 3
v2["scratch_collapse_check"] = [{"sensor": s, "slide_f1": T6[s]["Scratch"][3][0], "macro_if_collapsed": round(T6[s]["Scratch"][3][0] / 3, 2),
                                  "reported_macro": T6[s]["Scratch"][0][0]} for s in SENS]
R["V2_tables_2_5_6"] = v2

# ================================================================ V3 사전학습 효과 분해 (Scratch -> SPH -> MAE -> HTT)
v3 = {"classification": {}, "force": {}, "slip": {}}
for i, s in enumerate(SENS):
    row = {m: T1[m][i][0] for m in ("Scratch", "SPH", "MAE", "HTT")}
    v3["classification"][s] = {"scratch_to_sph_arch": round(row["SPH"] - row["Scratch"], 2),
                               "sph_to_mae_pretrain": round(row["MAE"] - row["SPH"], 2),
                               "mae_to_htt_align": round(row["HTT"] - row["MAE"], 2), "total": round(row["HTT"] - row["Scratch"], 2)}
    rowf = {m: T2_FORCE[m][i] for m in ("Scratch", "SPH", "MAE", "HTT")}
    v3["force"][s] = {"scratch_to_sph_arch": round(rowf["SPH"] - rowf["Scratch"], 3), "sph_to_mae_pretrain": round(rowf["MAE"] - rowf["SPH"], 3),
                      "mae_to_htt_align": round(rowf["HTT"] - rowf["MAE"], 3), "total": round(rowf["HTT"] - rowf["Scratch"], 3)}
    rows_ = {m: T2_SLIP[m][i] for m in ("Scratch", "SPH", "MAE", "HTT")}
    v3["slip"][s] = {"scratch_to_sph_arch": round(rows_["SPH"] - rows_["Scratch"], 2), "sph_to_mae_pretrain": round(rows_["MAE"] - rows_["SPH"], 2),
                     "mae_to_htt_align": round(rows_["HTT"] - rows_["MAE"], 2), "total": round(rows_["HTT"] - rows_["Scratch"], 2)}
ov = {}
for name, tab, fmt in (("classification", {m: [(r[0] if r else None) for r in T1[m]] for m in T1}, 2), ("force", T2_FORCE, 3), ("slip", T2_SLIP, 2)):
    sc, sp, ma, ht = (tab[m][4] for m in ("Scratch", "SPH", "MAE", "HTT"))
    ov[name] = {"Scratch": sc, "SPH": sp, "MAE": ma, "HTT": ht,
                "arch": round(sp - sc, fmt), "pretrain": round(ma - sp, fmt), "align": round(ht - ma, fmt)}
    tot = ht - sc
    ov[name]["share_arch_pct"] = round(100 * (sp - sc) / tot, 1)
    ov[name]["share_pretrain_pct"] = round(100 * (ma - sp) / tot, 1)
    ov[name]["share_align_pct"] = round(100 * (ht - ma) / tot, 1)
v3["overall"] = ov
R["V3_ablation_decomposition"] = v3

# ================================================================ V4 시드 3개 가정 Welch 검정 (HTT 대 MAE, HTT 대 SPH, MAE 대 SPH)
v4 = {"assumption": "n=3 시드, ±는 표본 SD(ddof=1). ddof0 열은 ± 를 모집단 SD로 본 경우", "classification": [], "force3d": [], "slip_macro": []}
for i, s in enumerate(SENS):
    for a, b in (("HTT", "MAE"), ("HTT", "SPH"), ("MAE", "SPH")):
        r = welch(*T1[a][i], *T1[b][i])
        r0 = welch(*T1[a][i], *T1[b][i], ddof0=True)
        v4["classification"].append({"sensor": s, "cmp": f"{a} vs {b}", "diff_pp": round(T1[a][i][0] - T1[b][i][0], 2),
                                     "t": round(r[0], 2), "p": float(f"{r[1]:.3g}"), "p_ddof0": float(f"{r0[1]:.3g}")})
    for a, b in (("HTT", "MAE"), ("HTT", "SPH"), ("MAE", "SPH")):
        if a in T5[s] and b in T5[s]:
            r = welch(*T5[s][a][0], *T5[s][b][0])
            r0 = welch(*T5[s][a][0], *T5[s][b][0], ddof0=True)
            v4["force3d"].append({"sensor": s, "cmp": f"{a} vs {b}", "diff_N": round(T5[s][a][0][0] - T5[s][b][0][0], 3),
                                  "t": round(r[0], 2), "p": float(f"{r[1]:.3g}"), "p_ddof0": float(f"{r0[1]:.3g}")})
        ra, rb = T6[s][a][0], T6[s][b][0]
        r = welch(*ra, *rb)
        r0 = welch(*ra, *rb, ddof0=True)
        if r:
            v4["slip_macro"].append({"sensor": s, "cmp": f"{a} vs {b}", "diff_pp": round(ra[0] - rb[0], 2),
                                     "t": round(r[0], 2), "p": float(f"{r[1]:.3g}"), "p_ddof0": float(f"{r0[1]:.3g}")})
# Holm: HTT 대 MAE 12개 (3 과제 x 4 센서, 힘은 4 센서 모두 가능)
htt_mae = [x for lst in (v4["classification"], v4["force3d"], v4["slip_macro"]) for x in lst if x["cmp"] == "HTT vs MAE"]
ps = [x["p"] for x in htt_mae]
rej, padj, _, _ = multipletests(ps, method="holm")
v4["htt_vs_mae_holm"] = [{"task_sensor": f"{'cls' if 'diff_pp' in x and x in v4['classification'] else ''}", "p": x["p"], "p_holm": float(f"{pa:.3g}"), "reject": bool(r_)}
                         for x, pa, r_ in zip(htt_mae, padj, rej)]
v4["htt_vs_mae_holm"] = []
for x, pa, r_ in zip(htt_mae, padj, rej):
    task = "cls" if x in v4["classification"] else ("force" if x in v4["force3d"] else "slip")
    v4["htt_vs_mae_holm"].append({"task": task, "sensor": x["sensor"], "diff": x.get("diff_pp", x.get("diff_N")), "p": x["p"],
                                  "p_holm_12": float(f"{pa:.3g}"), "reject_0.05": bool(r_)})
v4["htt_vs_mae_signs"] = {"cls_htt_better": sum(1 for x in v4["classification"] if x["cmp"] == "HTT vs MAE" and x["diff_pp"] > 0),
                          "force_htt_better": sum(1 for x in v4["force3d"] if x["cmp"] == "HTT vs MAE" and x["diff_N"] < 0),
                          "slip_htt_better": sum(1 for x in v4["slip_macro"] if x["cmp"] == "HTT vs MAE" and x["diff_pp"] > 0)}
R["V4_seed_welch"] = v4

# ================================================================ V5 실로봇 성공률 (표 3, 40회)
N = 40
T3 = {"Toy screw": {"qpos": 1, "wrench": 18, "T3": 0, "SITR": 4, "HTT": 36},
      "Grasp tofu": {"qpos": 2, "wrench": 14, "T3": 2, "SITR": 7, "HTT": 22}}
REP = {"Toy screw": {"qpos": 2.5, "wrench": 45, "T3": 0, "SITR": 10, "HTT": 90}, "Grasp tofu": {"qpos": 5, "wrench": 35, "T3": 5, "SITR": 17.5, "HTT": 55}}
v5 = {"integer_feasibility": [], "wilson95": {}, "fisher": [], "slip_breakdown": {}, "v1_vs_v2": {}}
for task, d in T3.items():
    for k, c in d.items():
        v5["integer_feasibility"].append({"task": task, "cond": k, "reported_pct": REP[task][k], "count_of_40": c, "pct_from_count": 100 * c / N,
                                          "ok": abs(100 * c / N - REP[task][k]) < 1e-9})
        lo, hi = proportion_confint(c, N, method="wilson")
        v5["wilson95"][f"{task}|{k}"] = [round(100 * lo, 1), round(100 * hi, 1)]
# 20회로 불가능한 값 (표 3이 20회 기준이면 17.5%, 2.5%는 불가)
v5["pct_not_possible_with_n20"] = [x for x in (2.5, 17.5)]
rows = []
for task, d in T3.items():
    for b in ("wrench", "SITR", "T3", "qpos"):
        a_, b_ = d["HTT"], d[b]
        p = stats.fisher_exact([[a_, N - a_], [b_, N - b_]])[1]
        rows.append({"task": task, "cmp": f"HTT vs {b}", "htt": f"{a_}/{N}", "other": f"{b_}/{N}", "diff_pp": round(100 * (a_ - b_) / N, 1), "p": p})
rej, padj, _, _ = multipletests([r["p"] for r in rows], method="holm")
for r, pa, rj in zip(rows, padj, rej):
    r["p"] = float(f"{r['p']:.3g}")
    r["p_holm_8"] = float(f"{pa:.3g}")
    r["reject"] = bool(rj)
v5["fisher"] = rows
v5["wrench_vs_qpos"] = {t: float(f"{stats.fisher_exact([[d['wrench'], N - d['wrench']], [d['qpos'], N - d['qpos']]])[1]:.3g}") for t, d in T3.items()}
# 두 과제 합산 HTT 대 wrench: Cochran-Mantel-Haenszel
from statsmodels.stats.contingency_tables import StratifiedTable
tabs = [np.array([[T3[t]["HTT"], N - T3[t]["HTT"]], [T3[t]["wrench"], N - T3[t]["wrench"]]]) for t in T3]
st = StratifiedTable(tabs)
v5["cmh_htt_vs_wrench"] = {"pooled_OR": round(float(st.oddsratio_pooled), 2), "p": float(f"{st.test_null_odds().pvalue:.3g}")}
# 본문 슬립 분해: 12/20 대 8/20
v5["slip_breakdown"] = {"wrench_slip_12_of_20_vs_HTT_8_of_20_fisher_p": float(f"{stats.fisher_exact([[12, 8], [8, 12]])[1]:.3g}"),
                        "relative_reduction_pct": round(100 * (12 - 8) / 12, 1),
                        "text_counts_cover": "20 rollouts (wrench 12 slip + 1 crush + 7 success = 20; HTT 8 slip + 1 no-lift + 11 success = 20), Table 3 says 40",
                        "wrench_if_n40_unexplained_failures": 26 - 13, "HTT_if_n40_unexplained_failures": 18 - 9}
# v1 (n=20, 세 조건) 대 v2 (n=40)
v1c = {"Toy screw": {"qpos": 1, "wrench": 10, "HTT": 19}, "Grasp tofu": {"qpos": 1, "wrench": 7, "HTT": 11}}
v5["v1_vs_v2"] = {"v1_counts_of_20": v1c, "v2_counts_of_40": {t: {k: T3[t][k] for k in ("qpos", "wrench", "HTT")} for t in T3},
                  "implied_added_20_rollouts": {t: {k: T3[t][k] - v1c[t][k] for k in ("qpos", "wrench", "HTT")} for t in T3},
                  "v1_fisher_screw_HTT_vs_wrench": float(f"{stats.fisher_exact([[19, 1], [10, 10]])[1]:.3g}"),
                  "v1_fisher_tofu_HTT_vs_wrench": float(f"{stats.fisher_exact([[11, 9], [7, 13]])[1]:.3g}")}


def n_per_arm(p1, p2, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pb = (p1 + p2) / 2
    return math.ceil(((za * math.sqrt(2 * pb * (1 - pb)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p1 - p2) ** 2)


v5["n_per_arm_80pct_power"] = {"tofu_HTT55_vs_wrench35": n_per_arm(.55, .35), "tofu_HTT55_vs_SITR17.5": n_per_arm(.55, .175),
                               "screw_HTT90_vs_wrench45": n_per_arm(.90, .45)}
R["V5_real_robot_table3"] = v5

# ================================================================ V6 ManiFeel (표 4): 3 시드 x 50 회
T4 = {"peg": {"tacRGB": (.21, .02), "T3": (.23, .02), "SITR": (.35, .01), "HTT(RGB)": (.44, .04), "HTT(FF)": (.48, .12)},
      "bulb": {"tacRGB": (.72, .04), "T3": (.73, .06), "SITR": (.77, .04), "HTT(RGB)": (.77, .02), "HTT(FF)": (.76, .02)}}
v6 = {"integer_feasibility_n150": [], "welch": [], "binomial_sd_check": []}
for task, d in T4.items():
    for k, (m, sd) in d.items():
        cands = [c for c in range(0, 151) if round(c / 150, 2) == m]
        v6["integer_feasibility_n150"].append({"task": task, "method": k, "mean": m, "possible_counts": cands})
        # 한 시드(50회)의 이항 SD 대 보고 SD. 시드 SD <= 보고 SD 일 확률 (카이제곱, df=2)
        sig = math.sqrt(m * (1 - m) / 50)
        chi = 2 * sd ** 2 / sig ** 2
        v6["binomial_sd_check"].append({"task": task, "method": k, "binomial_sd_per_seed": round(sig, 3), "reported_sd": sd,
                                        "P(sample_sd<=reported)": float(f"{stats.chi2.cdf(chi, 2):.3g}")})
for task, d in T4.items():
    for a, b in (("HTT(RGB)", "SITR"), ("HTT(FF)", "SITR"), ("HTT(RGB)", "tacRGB"), ("HTT(FF)", "tacRGB"), ("HTT(RGB)", "T3"), ("HTT(RGB)", "HTT(FF)")):
        t, p = welch(*d[a], *d[b])
        v6["welch"].append({"task": task, "cmp": f"{a} vs {b}", "diff": round(d[a][0] - d[b][0], 2), "t": round(t, 2), "p": float(f"{p:.3g}")})
v6["bulb_range"] = {"min": min(v[0] for v in T4["bulb"].values()), "max": max(v[0] for v in T4["bulb"].values())}
# 150 회 이항 근사 (시드 통합) HTT(RGB) 대 SITR, 피험 시드 간 상관을 무시한 낙관적 검정
for a, b in (("HTT(RGB)", "SITR"), ("HTT(FF)", "SITR")):
    ca, cb = round(T4["peg"][a][0] * 150), round(T4["peg"][b][0] * 150)
    v6.setdefault("pooled150_fisher_peg", {})[f"{a} vs {b}"] = {"counts": [ca, cb], "p": float(f"{stats.fisher_exact([[ca, 150 - ca], [cb, 150 - cb]])[1]:.3g}")}
R["V6_manifeel_table4"] = v6

# ================================================================ V7 Procrustes 검색 (그림 3)
v7 = {"R_at_K_counts_of_99": {}, "chance": {}, "fisher_htt_vs_mae": {}}
for name, (r1, r5) in {"random_init": (3, 20), "MAE-only": (20, 61), "HTT": (41, 82)}.items():
    c1 = [c for c in range(100) if round(100 * c / 99) == r1]
    c5 = [c for c in range(100) if round(100 * c / 99) == r5]
    v7["R_at_K_counts_of_99"][name] = {"R@1_counts": c1, "R@5_counts": c5}
v7["chance"] = {"gallery99_R@1_pct": round(100 / 99, 1), "gallery99_R@5_pct": round(500 / 99, 1), "gallery50_R@1_pct": 2.0, "gallery50_R@5_pct": 10.0}
v7["fisher_htt_vs_mae"] = {"R@1_41_vs_20_of_99": float(f"{stats.fisher_exact([[41, 58], [20, 79]])[1]:.3g}"),
                           "R@5_81_vs_60_of_99": float(f"{stats.fisher_exact([[81, 18], [60, 39]])[1]:.3g}")}
v7["wilson95_R@1_htt"] = [round(100 * x, 1) for x in proportion_confint(41, 99, method="wilson")]
# 객체 식별 오라클: 99개 에피소드를 30개 객체에 무작위로 배정하고, 같은 객체 후보 사이에서 무작위로 고르는 검색자의 기대 R@1, R@5
rng = np.random.default_rng(0)
r1s, r5s = [], []
for rep in range(2000):
    obj = rng.integers(0, 30, 99)
    cnt = np.bincount(obj, minlength=30)[obj]  # 각 질의와 같은 객체인 에피소드 수 (자기 짝 포함)
    r1s.append(np.mean(1 / cnt))
    r5s.append(np.mean(np.minimum(1, 5 / cnt)))
v7["object_oracle_30_objects_uniform"] = {"R@1_pct": round(100 * float(np.mean(r1s)), 1), "R@5_pct": round(100 * float(np.mean(r5s)), 1),
                                         "note": "객체별 에피소드 수가 균등하지 않으면 R@1은 올라간다. 실제 분포는 데이터셋 카드에 없다(라벨 제거)."}
R["V7_retrieval_fig3"] = v7

# ================================================================ V8 데이터셋 규모 (HPT, 공개 STATS.json의 숫자)
S = {
    "pretrain": {"episodes": 1789, "paired_samples": 64603, "optical": 64603, "array": 646030, "raw": 710633},
    "classification": {"episodes": 599, "paired_samples": 21744, "optical": 21744, "array": 217440, "raw": 239184},
    "force_static": 209869, "slip_sliding": 213415,
    "pretrain_split": {"gsmini_tac02": {"train": 34881, "val": 3841, "test": 4839, "ep": (720, 80, 100)},
                       "9dtact_xela": {"train": 16819, "val": 1865, "test": 2358, "ep": (711, 79, 99)}},
    "classification_split": {"gsmini_tac02": {"train": 11731, "val": 958, "test": 1964, "ep": (239, 20, 40)},
                             "9dtact_xela": {"train": 5672, "val": 477, "test": 942, "ep": (240, 20, 40)}},
    "force_static_per_sensor": {"xela": 105691, "tac02": 54718, "9dtact": 24829, "gsmini": 24631},
    "slip_per_sensor": {"xela": 108649, "tac02": 54717, "9dtact": 24900, "gsmini": 25149},
    "slip_episodes_per_sensor": {"xela": 104, "tac02": 100, "9dtact": 100, "gsmini": 101},
}
v8 = {}
force_all = S["force_static"] + S["slip_sliding"]
tot_inst = S["pretrain"]["raw"] + S["classification"]["raw"] + force_all + S["slip_sliding"]
distinct = S["pretrain"]["raw"] + S["classification"]["raw"] + force_all
paired = S["pretrain"]["raw"] + S["classification"]["raw"]
v8["headline_1.59M"] = {"sum_frame_task_instances": tot_inst, "distinct_frames": distinct, "paired_corpora_frames": paired,
                        "pretrain_frames": S["pretrain"]["raw"], "pretrain_share_of_headline_pct": round(100 * S["pretrain"]["raw"] / tot_inst, 1),
                        "double_counted_sliding": S["slip_sliding"], "paired_share_of_headline_pct": round(100 * paired / tot_inst, 1)}
v8["fig1_bar_labels"] = {"paper_bars": {"Pretrain": 711, "Object": 239, "Force": 423, "Sliding": 213},
                         "dataset_card": {"pretrain": 710633, "classification": 239184, "force_static_plus_sliding": 423284, "slip": 213415},
                         "note": "막대 값은 데이터셋 카드와 일치. Force 막대(423k)가 정적 210k + 슬라이딩 213k이고 Sliding 막대(213k)가 같은 슬라이딩이다."}
# 그림 1 원그래프(센서별 프레임) 재구성: 슬라이딩을 힘과 슬립에 두 번 센 값
def sensor_total(pre_opt, cls_opt, cls_arr, pre_arr, force, slip):
    return pre_opt + cls_opt + pre_arr + cls_arr + force + 2 * slip  # 슬라이딩 2회
split = S["pretrain_split"]; csplit = S["classification_split"]
gs_pre_opt = split["gsmini_tac02"]["train"] + split["gsmini_tac02"]["val"] + split["gsmini_tac02"]["test"]
gs_cls_opt = csplit["gsmini_tac02"]["train"] + csplit["gsmini_tac02"]["val"] + csplit["gsmini_tac02"]["test"]
dt_pre_opt = split["9dtact_xela"]["train"] + split["9dtact_xela"]["val"] + split["9dtact_xela"]["test"]
dt_cls_opt = csplit["9dtact_xela"]["train"] + csplit["9dtact_xela"]["val"] + csplit["9dtact_xela"]["test"]
sens = {"GSMini": gs_pre_opt + gs_cls_opt + S["force_static_per_sensor"]["gsmini"] + 2 * S["slip_per_sensor"]["gsmini"],
        "TAC-02": 10 * (gs_pre_opt + gs_cls_opt) + S["force_static_per_sensor"]["tac02"] + 2 * S["slip_per_sensor"]["tac02"],
        "9DTact": dt_pre_opt + dt_cls_opt + S["force_static_per_sensor"]["9dtact"] + 2 * S["slip_per_sensor"]["9dtact"],
        "Xela": 10 * (dt_pre_opt + dt_cls_opt) + S["force_static_per_sensor"]["xela"] + 2 * S["slip_per_sensor"]["xela"]}
v8["fig1_pie_rebuilt_double_count"] = {k: v for k, v in sens.items()}
v8["fig1_pie_paper_k"] = {"XELA": 604, "9DTact": 103, "TAC02": 746, "GSMini": 133}
v8["fig1_pie_sum_check"] = {"rebuilt_sum": sum(sens.values()), "paper_sum_k": 604 + 103 + 746 + 133}
v8["fig1_pie_percentages"] = {k: round(100 * v / sum(sens.values()), 1) for k, v in sens.items()}
# 프레임 정의 혼용: 한 쌍 = 광학 1~2프레임 + 센서 배열 10프레임
v8["frame_unit_mix"] = {"array_to_optical_ratio_pretrain": S["pretrain"]["array"] / S["pretrain"]["optical"],
                        "array_share_of_pretrain_frames_pct": round(100 * S["pretrain"]["array"] / S["pretrain"]["raw"], 1),
                        "paired_samples_pretrain_plus_cls": S["pretrain"]["paired_samples"] + S["classification"]["paired_samples"]}
# 사전학습 에폭 수 (논문 부록 A.1과 코드 기본값)
train_pairs = {k: v["train"] for k, v in S["pretrain_split"].items()}
tot_train = sum(train_pairs.values())
p_gt = train_pairs["gsmini_tac02"] / tot_train
v8["pretrain_epochs"] = {"train_paired_samples_total": tot_train, "pair_sampling_prob_gsmini_tac02": round(p_gt, 3),
                         "paper_50k_steps_x_256": {"samples": 50000 * 256, "epochs_over_train_pairs": round(50000 * 256 / tot_train, 0)},
                         "code_34k_steps_x_512": {"samples": 34000 * 512, "epochs_over_train_pairs": round(34000 * 512 / tot_train, 0)},
                         "code_60k_steps_x_512": {"samples": 60000 * 512, "epochs_over_train_pairs": round(60000 * 512 / tot_train, 0)},
                         "note": "쌍 데이터셋 크기는 sliding 청크(overlap_step=1)라 인접 샘플이 겹친다. 독립 샘플 수는 이보다 작다."}
# 평가 분할 규모
v8["eval_split_sizes"] = {"classification_test_episodes_per_pair": 40, "classes": 20, "episodes_per_class_test": 2,
                          "classification_test_samples": {"gsmini_tac02": 1964, "9dtact_xela": 942},
                          "force_static_episodes_per_sensor": {"xela": 100, "tac02": 100, "9dtact": 100, "gsmini": 101},
                          "force_split_80_10_10_test_episodes_approx": {"xela": 10, "tac02": 10, "9dtact": 10, "gsmini": 11},
                          "slip_test_episodes_approx": {k: round(0.1 * v) for k, v in S["slip_episodes_per_sensor"].items()}}
# 슬립 라벨: 코드 기본값에서 incipient 비율의 상한 (에피소드당 최대 5프레임)
inc_max = sum(5 for _ in S["slip_episodes_per_sensor"]) and sum(5 * v for v in S["slip_episodes_per_sensor"].values())
v8["slip_label_check"] = {"paper_pct": {"static": 13.6, "incipient": 1.2, "slide": 85.2, "sum": 13.6 + 1.2 + 85.2},
                          "max_incipient_frames_default_window5": inc_max, "slip_frames": S["slip_sliding"],
                          "max_incipient_pct_pooled": round(100 * inc_max / S["slip_sliding"], 2),
                          "per_sensor_max_incipient_pct": {k: round(100 * 5 * S["slip_episodes_per_sensor"][k] / S["slip_per_sensor"][k], 2)
                                                           for k in S["slip_per_sensor"]},
                          "collapsed_all_slide_macro_f1_pct": round(100 * (2 * .852 / (1 + .852)) / 3, 1)}
R["V8_dataset_counts"] = v8

# ================================================================ V9 v1 대 v2 본문 수치
v9 = {"HTT_vs_SITR_9DTact_v1_text_13.5": cit["v1_HTT_vs_SITR_9DTact"], "HTT_vs_SITR_GSMini_v1_text_17": cit["v1_HTT_vs_SITR_GSMini"],
      "v2_text_12.61_14.99": [cit["HTT_vs_SPH_9DTact_pp"], cit["HTT_vs_AnyTouch_GSMini_pp"]],
      "v2_claim_both_pretrained_outperform_all_baselines_force_slip": {}}
# v2 5.3: MAE 와 HTT가 Scratch, T3, SITR, AnyTouch 모두를 이긴다 (SPH 제외)
chk = []
for i, s in enumerate(SENS):
    for task, T, better in (("force", T2_FORCE, lambda a, b: a < b), ("slip", T2_SLIP, lambda a, b: a > b)):
        for m in ("MAE", "HTT"):
            for b in ("Scratch", "T3", "SITR", "AnyTouch"):
                if T[b][i] is None:
                    continue
                if not better(T[m][i], T[b][i]):
                    chk.append(f"{task} {s}: {m} {T[m][i]} vs {b} {T[b][i]}")
v9["v2_claim_both_pretrained_outperform_all_baselines_force_slip"]["violations"] = chk
# "HTT outperforms MAE on three of four sensors in both tasks"
cnt_force = sum(1 for i in range(4) if T2_FORCE["HTT"][i] < T2_FORCE["MAE"][i])
cnt_slip = sum(1 for i in range(4) if T2_SLIP["HTT"][i] > T2_SLIP["MAE"][i])
v9["htt_beats_mae_count"] = {"force": cnt_force, "slip": cnt_slip}
# HTT 대 SPH
v9["htt_vs_sph_counts"] = {"cls": sum(1 for i in range(4) if T1["HTT"][i][0] > T1["SPH"][i][0]),
                           "force": sum(1 for i in range(4) if T2_FORCE["HTT"][i] < T2_FORCE["SPH"][i]),
                           "slip": sum(1 for i in range(4) if T2_SLIP["HTT"][i] > T2_SLIP["SPH"][i])}
# MAE 대 SPH 슬립 overall
v9["mae_vs_sph_slip_overall"] = {"MAE": T2_SLIP["MAE"][4], "SPH": T2_SLIP["SPH"][4]}
v9["t3_scratch_classification_9dtact_sd"] = T1["T3"][2]
R["V9_text_claims"] = v9

# ================================================================ V10 분류 정확도의 에피소드 단위 불확실성 (에피소드 40개)
v10 = {"note": "분류 테스트는 쌍당 40 에피소드(클래스당 2). 샘플 단위 정확도가 에피소드 단위로 묶여 있다고 보고, 40을 유효 표본 상한으로 둔 이항 구간",
       "wilson95_n40": {}}
for k in ("MAE", "HTT", "SPH"):
    for i, s in ((2, "9DTact"), (3, "GSMini")):
        acc = T1[k][i][0] / 100
        lo, hi = proportion_confint(round(acc * 40), 40, method="wilson")
        v10["wilson95_n40"][f"{k}|{s}"] = [round(100 * lo, 1), round(100 * hi, 1)]
v10["one_episode_in_pp"] = 100 / 40
v10["htt_minus_mae_in_episodes_9DTact"] = round((94.84 - 90.08) / 2.5, 2)
v10["htt_minus_mae_in_episodes_GSMini"] = round((91.35 - 88.59) / 2.5, 2)
R["V10_cls_uncertainty"] = v10

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT)
