"""FingerEye (arXiv:2604.20689 v4) 표 수치 재계산과 일관성 검사. numpy, scipy만 쓴다.

실행: python3 analysis/recompute.py   → analysis/recompute_results.json
시행 수가 논문에 적힌 곳(실로봇 20회, 시뮬 64회 x 평가 시드 3개 x 학습 시드 3개)만 검정하고,
적히지 않은 것(±의 정의)은 두 가지 해석으로 나눠 계산한다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
R = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def cmh(tables):
    """Cochran-Mantel-Haenszel (연속성 보정 없음). tables: [(a, b, c, d)] = (A 성공, A 실패, B 성공, B 실패)."""
    num, var = 0.0, 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        r1, c1 = a + b, a + c
        num += a - r1 * c1 / n
        var += r1 * (c + d) * c1 * (b + d) / (n * n * (n - 1))
    chi = num * num / var
    return chi, float(stats.chi2.sf(chi, 1)), num


# ---------------------------------------------------------------- V1-V3: 표 1-3 평균과 증분
sim = {  # 표 1, 2, 3 (평균 ± 표준편차) 과제 순서: Coin, Nut, Peg
    "W": [(25.5, 8.3), (38.7, 23.2), (16.0, 6.6)],
    "W+M": [(56.4, 3.7), (84.4, 5.2), (36.5, 5.5)],
    "W+FE": [(65.6, 4.8), (88.2, 4.2), (43.8, 4.2)],
    "W+T": [(27.3, 6.8), (28.6, 5.1), (16.5, 7.4)],
    "FEnc+FDec": [(43.6, 8.3), (83.7, 6.1), (18.2, 7.5)],
    "NoEnc+FDec": [(48.6, 8.3), (80.7, 11.3), (26.6, 7.6)],
    "NoEnc+GDec": [(54.9, 7.5), (88.9, 4.1), (33.5, 8.0)],
    "GEnc+FDec": [(60.2, 9.1), (85.4, 5.5), (33.9, 5.3)],
    "GEnc+GDec": [(65.6, 4.8), (88.2, 4.2), (43.8, 4.2)],
}
reported_mean = {"W": 26.7, "W+M": 59.1, "W+FE": 65.9, "W+T": 24.1, "FEnc+FDec": 48.5, "NoEnc+FDec": 52.0,
                 "NoEnc+GDec": 59.1, "GEnc+FDec": 59.8, "GEnc+GDec": 65.9}
v1 = {}
for k, cells in sim.items():
    m = float(np.mean([c[0] for c in cells]))
    v1[k] = {"mean_recomputed": round(m, 2), "mean_reported": reported_mean[k], "match": abs(round(m, 1) - reported_mean[k]) < 0.051}
R["V1_sim_means"] = v1
_m = {k: float(np.mean([c[0] for c in v])) for k, v in sim.items()}
R["V1_sim_gains_unrounded"] = {"W+M-W": round(_m["W+M"] - _m["W"], 2), "W+FE-W": round(_m["W+FE"] - _m["W"], 2), "W+FE-W+M": round(_m["W+FE"] - _m["W+M"], 2),
                               "GEnc+FDec-FEnc+FDec": round(_m["GEnc+FDec"] - _m["FEnc+FDec"], 2), "NoEnc+GDec-NoEnc+FDec": round(_m["NoEnc+GDec"] - _m["NoEnc+FDec"], 2),
                               "GEnc+GDec-NoEnc+FDec": round(_m["GEnc+GDec"] - _m["NoEnc+FDec"], 2)}
R["V1_sim_gains"] = {
    "W+M-W": round(59.1 - 26.7, 1), "W+FE-W (반올림 평균끼리 39.2, 보고 39.1은 비반올림 평균 차)": round(65.9 - 26.7, 1), "W+FE-W+M": round(65.9 - 59.1, 1),
    "GEnc+FDec-FEnc+FDec": round(59.8 - 48.5, 1), "NoEnc+GDec-NoEnc+FDec": round(59.1 - 52.0, 1),
    "GEnc+GDec-NoEnc+FDec": round(65.9 - 52.0, 1), "GEnc+GDec-GEnc+FDec": round(65.9 - 59.8, 1),
    "nut W+FE-W": round(88.2 - 38.7, 1),
}

# 실로봇: 20회, 성공 수 = 비율 x 20
real_pct = {"W": [25, 35, 30, 60], "W+M": [50, 55, 45, 75], "W+FE": [65, 70, 65, 85]}  # Coin, Chip, Letter, Syringe
real_k = {m: [int(round(p * 20 / 100)) for p in v] for m, v in real_pct.items()}
tasks = ["Coin", "Chip", "Letter", "Syringe"]
R["V2_real_counts"] = real_k
R["V2_real_means"] = {m: {"mean": float(np.mean(v)), "reported": {"W": 37.5, "W+M": 56.3, "W+FE": 71.3}[m]} for m, v in real_pct.items()}
R["V2_real_gains"] = {"W+M-W": 56.25 - 37.5, "W+FE-W": 71.25 - 37.5, "W+FE-W+M": 71.25 - 56.25}
R["V2_integer_feasible_n20"] = all((p * 20 / 100) == int(p * 20 / 100) for v in real_pct.values() for p in v)
# 표 2 실로봇 W+T
tT = {"Chip": (45, 9, "W+T"), "Syringe": (65, 13, "W+T")}

# ---------------------------------------------------------------- V4: 실로봇 검정 (20회)
def compare(a, b, name):
    out = {"per_task": {}, "tables": []}
    ps = []
    for i, t in enumerate(tasks):
        ka, kb = real_k[a][i], real_k[b][i]
        _, p = stats.fisher_exact([[kb, 20 - kb], [ka, 20 - ka]])
        out["per_task"][t] = {f"{b}": f"{kb}/20", f"{a}": f"{ka}/20", "diff_pp": 5 * (kb - ka),
                              "fisher_p": round(p, 4), "wilson_b": wilson(kb, 20), "wilson_a": wilson(ka, 20)}
        ps.append(p)
        out["tables"].append((kb, 20 - kb, ka, 20 - ka))
    # Holm
    order = np.argsort(ps)
    holm = [None] * 4
    prev = 0
    for rank, idx in enumerate(order):
        adj = min(1.0, (4 - rank) * ps[idx])
        prev = max(prev, adj)
        holm[idx] = round(prev, 4)
    for i, t in enumerate(tasks):
        out["per_task"][t]["holm_p"] = holm[i]
    KA, KB = sum(real_k[a]), sum(real_k[b])
    _, pp = stats.fisher_exact([[KB, 80 - KB], [KA, 80 - KA]])
    chi, pc, num = cmh(out["tables"])
    out["pooled"] = {f"{b}": f"{KB}/80", f"{a}": f"{KA}/80", "diff_pp": round(100 * (KB - KA) / 80, 2),
                     "fisher_p_pooled": round(pp, 5), "cmh_chi2": round(chi, 3), "cmh_p": round(pc, 5),
                     "wilson_b": wilson(KB, 80), "wilson_a": wilson(KA, 80)}
    out["tables"] = None
    return out


R["V3_real_W_vs_WFE"] = compare("W", "W+FE", "")
R["V3_real_WM_vs_WFE"] = compare("W+M", "W+FE", "")
R["V3_real_W_vs_WM"] = compare("W", "W+M", "")

# 검출력: 20회 두 군, 65% 대 45%, 70% 대 50% (양측 α=0.05, 정규근사)
def power_two_prop(p1, p2, n, alpha=0.05):
    pbar = (p1 + p2) / 2
    se0 = math.sqrt(2 * pbar * (1 - pbar) / n)
    se1 = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    z = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.cdf((abs(p1 - p2) - z * se0) / se1))


def n_per_group(p1, p2, power=0.8, alpha=0.05):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    return math.ceil(((za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p1 - p2) ** 2)


R["V3_power"] = {
    "n20_65_vs_45": round(power_two_prop(0.65, 0.45, 20), 3),
    "n20_71_vs_56": round(power_two_prop(0.7125, 0.5625, 80), 3),
    "n_per_group_for_80pct_power_65_vs_45": n_per_group(0.65, 0.45),
    "n_per_group_for_80pct_power_pooled_71_vs_56": n_per_group(0.7125, 0.5625),
}

# 표 2 실로봇 (Chip, Syringe): W 7/20, 12/20 ; W+T 9/20, 13/20 ; W+FE 14/20, 17/20
tt = {}
for name, (kW, kT, kF) in {"Chip": (7, 9, 14), "Syringe": (12, 13, 17)}.items():
    tt[name] = {
        "W+FE vs W+T fisher_p": round(stats.fisher_exact([[kF, 20 - kF], [kT, 20 - kT]])[1], 4),
        "W+T vs W fisher_p": round(stats.fisher_exact([[kT, 20 - kT], [kW, 20 - kW]])[1], 4),
    }
KF, KT, KW = 14 + 17, 9 + 13, 7 + 12
tt["pooled_W+FE_vs_W+T"] = {"counts": f"{KF}/40 vs {KT}/40", "fisher_p": round(stats.fisher_exact([[KF, 40 - KF], [KT, 40 - KT]])[1], 4)}
tt["pooled_W+T_vs_W"] = {"counts": f"{KT}/40 vs {KW}/40", "fisher_p": round(stats.fisher_exact([[KT, 40 - KT], [KW, 40 - KW]])[1], 4)}
R["V4_table2_real"] = tt

# 실로봇 그룹 융합 (Coin, Letter): GEnc+GDec 13+13 / 40 = 65.0, 비그룹 최강 NoEnc+FDec 52.5% = 21/40
KG, KN = 26, 21
R["V4_real_group_fusion"] = {"GEnc+GDec": "26/40 (65.0%)", "NoEnc+FDec": "21/40 (52.5%)", "diff_pp": 12.5,
                             "fisher_p": round(stats.fisher_exact([[KG, 40 - KG], [KN, 40 - KN]])[1], 4),
                             "wilson_G": wilson(KG, 40), "wilson_N": wilson(KN, 40),
                             "note": "52.5%는 40회 중 21회로 정수 가능. Coin 65, Letter 65는 표 1과 일치"}

# ---------------------------------------------------------------- V5: 시뮬 ± 해석과 정수 가능성
def binom_sd(p, n):
    return 100 * math.sqrt(p * (1 - p) / n)


v5 = {}
for name in ["W", "W+M", "W+FE", "W+T", "FEnc+FDec", "NoEnc+FDec", "NoEnc+GDec", "GEnc+FDec"]:
    row = []
    for t, (m, s) in zip(["Coin", "Nut", "Peg"], sim[name]):
        p = m / 100
        row.append({"task": t, "mean": m, "sd_reported": s, "binom_sd_n64": round(binom_sd(p, 64), 2),
                    "binom_sd_n192": round(binom_sd(p, 192), 2), "ratio_to_n64": round(s / binom_sd(p, 64), 2),
                    "ratio_to_n192": round(s / binom_sd(p, 192), 2)})
    v5[name] = row
R["V5_sim_sd_vs_binomial"] = v5
r64 = [c["ratio_to_n64"] for rows in v5.values() for c in rows]
r192 = [c["ratio_to_n192"] for rows in v5.values() for c in rows]
R["V5_sd_ratio_summary"] = {"median_ratio_n64": round(float(np.median(r64)), 2), "median_ratio_n192": round(float(np.median(r192)), 2),
                            "cells": len(r64), "cells_below_n64_binomial": int(sum(r < 1 for r in r64)),
                            "cells_below_n192_binomial": int(sum(r < 1 for r in r192))}


def feasible_576(x):
    ks = np.arange(0, 577)
    vals = np.round(100 * ks / 576 + 1e-9, 1)
    return bool(np.any(np.abs(vals - x) < 1e-6))


infeasible = []
for name, cells in sim.items():
    for t, (m, s) in zip(["Coin", "Nut", "Peg"], cells):
        if not feasible_576(m):
            infeasible.append(f"{name}/{t}={m}")
R["V5_grid_576_infeasible_means"] = infeasible
def grid_hits(n):
    vals = np.round(100 * np.arange(n + 1) / n + 1e-9, 1)
    allv = [m for cells in sim.values() for m, _ in cells]
    return int(sum(bool(np.any(np.abs(vals - x) < 1e-6)) for x in allv)), len(allv)


R["V5_grid_hits"] = {"n64": grid_hits(64), "n192": grid_hits(192), "n576": grid_hits(576),
                     "chance_per_value_on_576_grid": round(0.1 / (100 / 576), 3),
                     "chance_all_24_on_576_grid": float((0.1 / (100 / 576)) ** 24)}
R["V5_note"] = "평균이 9개 평가(학습 3 x 평가 3, 각 64회)의 단순 평균이면 100k/576 격자 위의 값이어야 한다. 격자 밖 값이 있으면 평균 정의(가중, 시드 선택)가 다르다."

# 두 해석의 Welch: n=3(학습 시드) 또는 n=9, 합 평균은 과제 독립 가정
def welch(m1, s1, m2, s2, n):
    se = math.sqrt(s1 ** 2 / n + s2 ** 2 / n)
    t = (m1 - m2) / se
    df = (s1 ** 2 / n + s2 ** 2 / n) ** 2 / ((s1 ** 2 / n) ** 2 / (n - 1) + (s2 ** 2 / n) ** 2 / (n - 1))
    return round(t, 2), round(float(2 * stats.t.sf(abs(t), df)), 4)


def mean_sd(cells):
    return float(np.mean([c[0] for c in cells])), math.sqrt(sum(c[1] ** 2 for c in cells)) / 3


pairs = [("W+FE", "W"), ("W+FE", "W+M"), ("W+M", "W"), ("W+FE", "W+T"), ("GEnc+GDec", "NoEnc+FDec"),
         ("GEnc+GDec", "GEnc+FDec"), ("GEnc+GDec", "NoEnc+GDec"), ("GEnc+FDec", "FEnc+FDec"), ("NoEnc+GDec", "NoEnc+FDec")]
v5w = {}
for a, b in pairs:
    row = {}
    for ti, t in enumerate(["Coin", "Nut", "Peg"]):
        (m1, s1), (m2, s2) = sim[a][ti], sim[b][ti]
        row[t] = {"diff": round(m1 - m2, 1), "welch_n3(t,p)": welch(m1, s1, m2, s2, 3), "welch_n9(t,p)": welch(m1, s1, m2, s2, 9)}
    (M1, S1), (M2, S2) = mean_sd(sim[a]), mean_sd(sim[b])
    row["mean_of_3_tasks"] = {"diff": round(M1 - M2, 1), "sd_a": round(S1, 2), "sd_b": round(S2, 2),
                              "welch_n3(t,p)": welch(M1, S1, M2, S2, 3), "welch_n9(t,p)": welch(M1, S1, M2, S2, 9)}
    v5w[f"{a} vs {b}"] = row
R["V5_sim_welch"] = v5w
R["V5_welch_note"] = "±를 3 학습 시드의 표준편차(n=3) 또는 9회 평가의 표준편차(n=9)로 읽은 두 해석. 과제 간 독립 가정으로 3과제 평균의 표준편차를 합성(근사)."

# ---------------------------------------------------------------- V6: 표 7
t7 = {  # task, policy: SR, FE, Wrist, Prop, win
    ("Coin", "NoEnc+FDec"): (48.6, 51.6, 12.9, 35.6, 18.9),
    ("Coin", "FEnc+FDec"): (43.6, 46.0, 10.5, 43.4, 37.6),
    ("Nut", "NoEnc+FDec"): (80.7, 31.3, 9.8, 59.0, 77.2),
    ("Nut", "FEnc+FDec"): (83.7, 48.8, 11.8, 39.5, 34.1),
    ("Peg", "NoEnc+FDec"): (26.6, 52.0, 10.1, 38.0, 22.8),
    ("Peg", "FEnc+FDec"): (18.2, 42.2, 7.8, 50.0, 61.0),
}
R["V6_table7_mass_sums"] = {f"{k[0]}/{k[1]}": round(v[1] + v[2] + v[3], 1) for k, v in t7.items()}
props = [v[3] for v in t7.values()]
wrists = [v[2] for v in t7.values()]
R["V6_table7_ranges"] = {"prop_min": min(props), "prop_max": max(props), "wrist_min": min(wrists), "wrist_max": max(wrists),
                         "uniform_ref_prop_pct": round(100 / 6, 1), "uniform_ref_FE_pct": round(100 * 4 / 6, 1),
                         "uniform_ref_wrist_pct": round(100 / 6, 1)}
pair_agree = 0
for t in ["Coin", "Nut", "Peg"]:
    a, b = t7[(t, "NoEnc+FDec")], t7[(t, "FEnc+FDec")]
    pair_agree += int((a[0] > b[0]) == (a[1] > b[1]))
R["V6_sr_vs_fe_mass_sign_agreement"] = {"pairs": 3, "agree": pair_agree, "sign_test_p_two_sided_if_random": 0.25,
                                       "note": "세 쌍 모두 같은 방향이어도 쌍 수가 3이라 부호 검정으로는 p = 0.25(양측)"}
R["V6_flat_dec_win_rate_vs_SR"] = {"corr_prop_win_vs_SR": round(float(np.corrcoef([v[4] for v in t7.values()], [v[0] for v in t7.values()])[0, 1]), 2),
                                   "n": 6, "note": "과제가 섞인 6점의 상관(참고용)"}

# ---------------------------------------------------------------- V7: 민감도 (A.2.2)
wtag, wimg, dR = 2.0, 37.0, 0.25
dl = wtag / wimg * dR
th, r = math.pi / 12, 18.5
dth = th / (2 * r * math.sin(th / 2)) * dR
Fmin = np.array([4.30, 4.22, 9.93, 0.32, 0.13, 8.55])  # mN, mN·m
rmse_N = np.array([0.013, 0.009, 0.047])
dLmin = np.array([dl] * 3 + [dth] * 3)
stiff = Fmin * 1e-3 / np.concatenate([np.array([dl] * 3) * 1e-3, np.array([dth] * 3)])  # N/m, N·m/rad
sim_gain = np.array([70.0, 29.17, 100.0, 0.50, 0.20, 13.33])
R["V7_sensitivity"] = {
    "dl_min_mm": round(dl, 5), "dtheta_min_rad": round(dth, 5),
    "pixels_per_mm_if_tag_37px_2mm": 18.5,
    "fov120_working_dist10mm_width_mm": round(2 * 10 * math.tan(math.radians(60)), 1),
    "px_per_mm_for_640px_across_fov": round(640 / (2 * 10 * math.tan(math.radians(60))), 1),
    "implied_stiffness_N_per_m_and_Nm_per_rad": [round(float(x), 4) for x in stiff],
    "sim_holder_gain_code": sim_gain.tolist(),
    "sim_over_paper_implied": [round(float(a / b), 3) for a, b in zip(sim_gain, stiff)],
    "force_rmse_N": rmse_N.tolist(),
    "force_rmse_over_Fmin": [round(float(r_ / (f * 1e-3)), 2) for r_, f in zip(rmse_N, Fmin[:3])],
    "z_sensitivity_single_tag_perspective_mm": round(10.0 * dR / wimg, 4),
    "z_over_lateral": round(10.0 * dR / wimg / dl, 2),
}
R["V7_note"] = ("F_min = (보정 행렬) x ΔL_min 이므로 대각 강성 F_min/ΔL_min은 비대각 항이 있으면 근사. "
                "z 병진 감도는 단일 태그의 원근 투영(Δz = Z·δ/w)이라는 단순 모델이며 다중 태그 PnP는 이보다 정밀할 수 있다.")

# ---------------------------------------------------------------- V8: 접촉 임계값
t6 = {"pencil lead": (0.02, 15), "chip": (0.10, 30), "dessert cup": (0.05, 10), "seaweed": (0.01, 150), "grape": (0.06, 10),
      "cone": (0.10, 20), "egg shell": (0.05, 20), "wafer": (0.05, 5), "balloon": (0.05, 3)}
code_thr = {"pencil lead": 0.02, "chip": 0.10, "dessert cup": 0.005, "seaweed": 0.01, "grape": 0.006, "cone": 0.10,
            "egg shell": 0.05, "wafer": 0.05, "balloon": 0.05}
jit35 = 0.012
R["V8_thresholds"] = {
    k: {"paper_mm": v[0], "code_mm": code_thr[k], "frames": v[1], "paper_over_dl_min": round(v[0] / dl, 2),
        "code_over_dl_min": round(code_thr[k] / dl, 2), "paper_over_jitter35": round(v[0] / jit35, 2),
        "code_over_jitter35": round(code_thr[k] / jit35, 2)} for k, v in t6.items()}
R["V8_summary"] = {"dl_min_mm": round(dl, 4), "jitter_35tag_mm": jit35,
                   "paper_below_dl_min": [k for k, v in t6.items() if v[0] < dl],
                   "code_below_dl_min": [k for k, v in code_thr.items() if v < dl],
                   "code_below_jitter": [k for k, v in code_thr.items() if v < jit35],
                   "code_differs_from_paper_by_10x": [k for k in t6 if abs(t6[k][0] / code_thr[k] - 10) < 1e-6],
                   "seconds_per_motion": {k: round(v[1] * 0.02, 2) for k, v in t6.items()}}

# ---------------------------------------------------------------- V9: 표 4, 표 5, A.2.4, 기타 산술
bom = [0.28, 0.42, 0.10, 0.17, 0.02, 0.44, 0.45, 1.78, 0.04, 56.49]
R["V9_bom_sum"] = {"sum": round(sum(bom), 2), "reported": 60.19, "camera_share_pct": round(100 * 56.49 / 60.19, 1),
                   "per_camera_usd": round(56.49 / 2, 2), "mold_share_pct": round(100 * 1.78 / 60.19, 1)}
R["V9_tag_table5"] = {"detected_fraction": {n: round(d / n, 3) for n, d in {1: 0.85, 5: 3.65, 20: 11.86, 35: 19.82}.items()},
                      "jitter_ratio_1_to_35_mm": round(0.426 / 0.012, 1), "jitter_ratio_5_to_35_mm": round(0.207 / 0.012, 1),
                      "jitter_ratio_20_to_35_mm": round(0.015 / 0.012, 2), "sr_20tag_vs_35tag_pp": round(68.3 - 65.9, 1)}
R["V9_durability"] = {"hyst_frac_before_pct": round(100 * 0.39 / 3, 1), "hyst_frac_after_pct": round(100 * 0.46 / 3, 1),
                      "hours_for_10000_cycles_1Hz": round(10000 / 3600, 2), "rms_diff_mm_over_3mm_pct": round(100 * 0.13 / 3, 1),
                      "task_recheck_trials": "Coin 20회 변화 없음, Chip 20회 중 성공 1 감소(기준 14/20 가정 시 13/20)"}
R["V9_time"] = {"chunk_seconds_Tp16_at10Hz": 1.6, "replan_seconds_Ta8": 0.8, "replan_seconds_Ta16": 1.6,
                "sim_peg_coin_Ta16_means_open_loop_seconds": 1.6}
R["V9_cams_params"] = {"policy_dim": 512, "radio_summary": 2304, "images_hw_paper": [192, 256]}

# ---------------------------------------------------------------- V10: 버전 간 수치 (v1 그림 13·14는 그림 판독 값)
v1real = {"Coin": ("65.0", "60 demos, 20 trials", "13/20"), "Chip": ("66.7", "27 demos, 18 trials", "12/18"),
          "Letter": ("66.7", "60 demos, 12 trials", "8/12"), "Syringe": ("85.0", "40 demos, 20 trials", "17/20")}
R["V10_version_real"] = {
    "v1_fig13_FE (그림 판독)": {"Coin 65.0 (20회)": 13, "Chip 66.7 (18회)": "12/18", "Letter 66.7 (12회)": "8/12", "Syringe 85.0 (20회)": 17, "avg": 70.9},
    "v1_fig13_Mono": {"Coin 50.0": "10/20", "Chip 55.6": "10/18", "Letter 50.0": "6/12", "Syringe 75.0": "15/20", "avg": 57.7},
    "v1_fig13_Wrist": {"Coin 25.0": "5/20", "Chip 38.9": "7/18", "Letter 33.3": "4/12", "Syringe 60.0": "12/20", "avg": 39.3},
    "v4_table1": {"W+FE": [65, 70, 65, 85, 71.3], "W+M": [50, 55, 45, 75, 56.3], "W": [25, 35, 30, 60, 37.5]},
}
R["V10_v1_avg_check"] = {"FE": round((65 + 66.7 + 66.7 + 85) / 4, 2), "Mono": round((50 + 55.6 + 50 + 75) / 4, 2), "Wrist": round((25 + 38.9 + 33.3 + 60) / 4, 2),
                          "FE-Wrist_pp": round((65 + 66.7 + 66.7 + 85) / 4 - (25 + 38.9 + 33.3 + 60) / 4, 1)}
R["V10_v1_sim_coin"] = {"v1 (30 demos, 64 trials; 그림 판독)": {"FE": 60.2, "Mono": 50.0, "Wrist": 39.1},
                        "v4 (50 demos, 3x3x64)": {"W+FE": 65.6, "W+M": 56.4, "W": 25.5},
                        "gain_FE_minus_W_v1": round(60.2 - 39.1, 1), "gain_FE_minus_W_v4": round(65.6 - 25.5, 1),
                        "v1_percent_on_64_grid": {"39.1": "25/64 = 39.06", "50.0": "32/64", "60.2": "격자 밖 (38/64=59.4, 39/64=60.9)"}}
# v1 시행 수가 적은 칸의 Wilson 구간
R["V10_wilson_v1"] = {"Letter FE 8/12": wilson(8, 12), "Chip FE 12/18": wilson(12, 18), "Letter FE v4 13/20": wilson(13, 20), "Chip FE v4 14/20": wilson(14, 20)}

# ---------------------------------------------------------------- V11: 파라미터 수 (논문 부록 설정으로 직접 계산, 공식 코드는 import하지 않음)
def mha(d):
    return 4 * d * d + 4 * d


def ln(d):
    return 2 * d


def enc_layer(d, ff):
    return mha(d) + 2 * ln(d) + d * ff + ff + ff * d + d


def dec_layer_flat(d, ff):
    return 2 * mha(d) + 3 * ln(d) + d * ff + ff + ff * d + d


def dec_layer_group(d, ff, groups=3):
    return (1 + groups) * mha(d) + (2 + groups) * ln(d) + d * ff + ff + ff * d + d


d, ff, L = 512, 2048, 4
proj = 2304 * 2 * d + 2 * d + ln(2 * d) + 2 * d * d + d  # Linear(2304,1024) + LN + Linear(1024,512)
views = 5
state = 8 * d + d + ln(d) + d * d + d
head = d * (d // 2) + d // 2 + (d // 2) * 8 + 8
base = views * proj + views * d + 16 * d + state + head
cfg = {
    "NoEnc+FDec": base + L * dec_layer_flat(d, ff) + ln(d),
    "GEnc+FDec": base + L * enc_layer(d, ff) + L * dec_layer_flat(d, ff) + ln(d),
    "NoEnc+GDec": base + L * dec_layer_group(d, ff) + ln(d),
    "GEnc+GDec": base + L * enc_layer(d, ff) + L * dec_layer_group(d, ff) + ln(d),
}
R["V11_param_counts_estimate"] = {k: int(v) for k, v in cfg.items()}
R["V11_param_ratio_to_NoEnc+FDec"] = {k: round(v / cfg["NoEnc+FDec"], 3) for k, v in cfg.items()}
R["V11_note"] = "코드 구조(공식 저장소 정적 읽기)와 표 8의 차원(512, FF 2048, 4+4층)으로 센 추정치. 코드는 실행하지 않았고 FEnc 변형은 저장소에 없어 제외."

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT)
for k in ["V1_sim_gains", "V2_real_means", "V3_real_W_vs_WFE", "V3_real_WM_vs_WFE", "V5_sd_ratio_summary", "V5_grid_576_infeasible_means",
          "V7_sensitivity", "V8_summary", "V11_param_counts_estimate", "V3_power", "V4_real_group_fusion", "V4_table2_real"]:
    print(k, json.dumps(R[k], ensure_ascii=False)[:1500])
