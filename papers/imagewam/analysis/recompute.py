"""ImageWAM (arXiv:2606.19531) 표 수치 재계산과 검정.

논문 표 1-7, 11, 12의 평균, 정수 복원, 비교 검정을 다시 계산하고 recompute_results.json에 저장한다.
CPU 수 초. 실행: python3 recompute.py (이 폴더에서)
입력 수치는 논문 표를 옮긴 것이다(표 12는 table12_data.json).
"""
import json
import math
import os

import numpy as np
from scipy import stats
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}


def r(x, n=2):
    return None if x is None else round(float(x), n)


# ---------------------------------------------------------------- 표 1 (RoboTwin 요약)
T1 = {  # clean, rand, avg(보고)
    "pi0": (65.92, 58.40, 62.16),
    "pi05": (82.74, 76.76, 79.75),
    "ABot-M0": (81.20, 80.40, 80.80),
    "Motus": (88.66, 87.02, 87.80),
    "LingBot-VA": (92.90, 91.50, 92.20),
    "FastWAM": (91.88, 91.78, 91.83),
    "ImageWAM": (93.20, 93.56, 93.38),
}
OUT["V1_table1_avg"] = {
    k: {"reported": v[2], "recomputed": r((v[0] + v[1]) / 2, 3), "diff": r(v[2] - (v[0] + v[1]) / 2, 3)}
    for k, v in T1.items()
}

# ---------------------------------------------------------------- 표 2 (LIBERO)
T2 = {  # Spatial Object Goal Long, 보고 평균
    "OpenVLA": ([84.7, 88.4, 79.2, 53.7], 76.5),
    "GR00T N1": ([84.7, 88.4, 79.2, 53.7], 76.5),
    "pi0": ([96.8, 98.8, 95.8, 85.2], 94.1),
    "pi05": ([98.8, 98.2, 98.0, 92.4], 96.9),
    "LingBot-VA": ([98.5, 99.6, 97.2, 98.5], 98.5),
    "Motus": ([96.8, 99.8, 96.6, 97.6], 97.7),
    "Fast-WAM": ([98.2, 100.0, 97.0, 95.2], 97.6),
    "ImageWAM": ([97.2, 99.2, 98.8, 98.4], 98.4),
}
o = {}
for k, (v, avg) in T2.items():
    m = float(np.mean(v))
    o[k] = {"reported": avg, "recomputed": r(m, 3), "diff": r(avg - m, 3)}
o["_duplicate_rows"] = T2["OpenVLA"] == T2["GR00T N1"]
# 500회/스위트(과제 10 x 50) 정수 복원
imw = T2["ImageWAM"][0]
o["_imagewam_counts_of_500"] = [r(x * 5, 3) for x in imw]
o["_imagewam_total_success_of_2000"] = r(sum(x * 5 for x in imw), 3)
OUT["V2_table2_libero"] = o


def two_prop(k1, n1, k2, n2):
    """두 비율의 양측 z 검정(비합동 SE가 아닌 합동)과 Fisher."""
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    pz = 2 * (1 - stats.norm.cdf(abs(z)))
    _, pf = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
    return {"diff_pp": r((p1 - p2) * 100, 2), "z": r(z, 2), "p_z": float(f"{pz:.3g}"), "p_fisher": float(f"{pf:.3g}")}


# LIBERO 평균 비교 (n=2000 가정: 10 과제 x 50회 x 4 스위트)
n = 2000
OUT["V2b_libero_tests_n2000"] = {
    "ImageWAM_vs_FastWAM(97.6)": two_prop(1968, n, 1952, n),
    "ImageWAM_vs_Motus(97.7)": two_prop(1968, n, round(0.977 * n), n),
    "ImageWAM_vs_LingBot(98.5)": two_prop(1968, n, round(0.985 * n), n),
    "ImageWAM_vs_pi05(96.85)": two_prop(1968, n, round(0.9685 * n), n),
}
# 스위트별 ImageWAM 대 Fast-WAM (각 500회)
suite = {}
for nm, a, b in zip(["Spatial", "Object", "Goal", "Long"], T2["ImageWAM"][0], T2["Fast-WAM"][0]):
    suite[nm] = two_prop(round(a * 5), 500, round(b * 5), 500)
    suite[nm]["imagewam_fastwam"] = [a, b]
OUT["V2d_libero_suite_vs_fastwam_n500"] = suite
# 검출 한계: 0.8 검정력, alpha .05, p~0.98, n=2000/arm
p0 = 0.98
se0 = math.sqrt(2 * p0 * (1 - p0) / 2000)
OUT["V2c_libero_mde_pp"] = r((1.96 + 0.84) * se0 * 100, 2)

# ---------------------------------------------------------------- 표 12 (RoboTwin 과제별)
d12 = json.load(open(os.path.join(HERE, "table12_data.json")))
V = np.array(d12["values"], dtype=float)  # 50 x 14
cols = d12["columns"]
rep = np.array(d12["reported_average"])
means = V.mean(axis=0)
t12 = {}
for j, c in enumerate(cols):
    t12[c] = {"reported": float(rep[j]), "recomputed": r(means[j], 3), "diff": r(rep[j] - means[j], 3)}
OUT["V3_table12_column_means"] = t12
OUT["V3_n_tasks"] = int(V.shape[0])
# 정수 가능성: OmniGen2 열은 과제당 50회 -> 값은 짝수여야 함
om = V[:, 2:4]
OUT["V3b_omnigen2_50trial_parity"] = {
    "odd_cells": int((om % 2 == 1).sum()),
    "total_cells": int(om.size),
    "avg_multiple_of_0.04?": [
        r(((m * 50) % 2), 3) for m in (rep[2], rep[3])
    ],
}
# Flux 열(100회 가정)에서 평균이 0.02의 배수인가
OUT["V3c_flux_avg_grid"] = {
    "clean_x50": r(rep[0] * 50, 3),
    "rand_x50": r(rep[1] * 50, 3),
}
# 표 1 대 표 12 일치
OUT["V3d_table1_vs_12"] = {
    "ImageWAM": [T1["ImageWAM"][0] == rep[0], T1["ImageWAM"][1] == rep[1]],
    "LingBot-VA": [T1["LingBot-VA"][0] == rep[8], T1["LingBot-VA"][1] == rep[9]],
    "pi05": [T1["pi05"][0] == rep[10], T1["pi05"][1] == rep[11]],
    "Motus": [T1["Motus"][0] == rep[12], T1["Motus"][1] == rep[13]],
    "FastWAM_table1(91.88/91.78)_vs_FastWAM-IDM_table12": [rep[4], rep[5]],
}

# 과제별 쌍체 비교: ImageWAM(Flux 4B) 대 각 열
names = {"fastwam_idm": (4, 5), "fastwam_noco": (6, 7), "lingbot": (8, 9), "pi05": (10, 11), "motus": (12, 13), "omnigen2": (2, 3)}
rng = np.random.default_rng(0)
pair = {}
for nm, (jc, jr) in names.items():
    for phase, (a, b) in (("clean", (0, jc)), ("rand", (1, jr))):
        diff = V[:, a] - V[:, b]
        win, loss, tie = int((diff > 0).sum()), int((diff < 0).sum()), int((diff == 0).sum())
        sign_p = stats.binomtest(win, win + loss, 0.5).pvalue if win + loss else 1.0
        try:
            w = stats.wilcoxon(diff[diff != 0])
            wp = float(w.pvalue)
        except Exception:
            wp = None
        boot = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(5000)]
        pair[f"{nm}_{phase}"] = {
            "mean_diff": r(diff.mean(), 2),
            "ci95": [r(np.percentile(boot, 2.5), 2), r(np.percentile(boot, 97.5), 2)],
            "win_loss_tie": [win, loss, tie],
            "p_sign": float(f"{sign_p:.3g}"),
            "p_wilcoxon": None if wp is None else float(f"{wp:.3g}"),
            "p_paired_t": float(f"{stats.ttest_1samp(diff, 0).pvalue:.3g}"),
        }
OUT["V4_pairwise_tasks"] = pair

# 차이가 몇 과제에 몰려 있는가: 중앙값과 상위 4개 과제를 뺀 평균
conc = {}
for nm, (a, b) in {"fastwam_idm_rand": (1, 5), "fastwam_idm_clean": (0, 4), "lingbot_rand": (1, 9), "lingbot_clean": (0, 8)}.items():
    diff = V[:, a] - V[:, b]
    order = np.argsort(-diff)
    conc[nm] = {
        "median": r(np.median(diff), 1),
        "mean": r(diff.mean(), 2),
        "mean_without_top4": r(np.delete(diff, order[:4]).mean(), 2),
        "top4_tasks": [(d12["tasks"][i], int(diff[i])) for i in order[:4]],
        "bottom2_tasks": [(d12["tasks"][i], int(diff[i])) for i in order[-2:]],
    }
OUT["V4e_concentration"] = conc

# Fast-WAM 논문(arXiv:2603.16666) 표 3의 과제별 값 (ImageWAM 표 12에는 이 두 열이 없다)
fw = json.load(open(os.path.join(HERE, "fastwam_table3_data.json")))
assert fw["tasks"] == d12["tasks"]
FW = np.array(fw["values"], dtype=float)  # 14열: Fast-WAM c/r, Joint c/r, IDM c/r, ...
OUT["V4f_baseline_columns_identical_to_fastwam_table3"] = bool(np.array_equal(V[:, 4:], FW[:, 4:]))
fwres = {}
for nm, (jc, jr) in {"fastwam": (0, 1), "fastwam_joint": (2, 3)}.items():
    for phase, (a, b) in (("clean", (0, jc)), ("rand", (1, jr))):
        diff = V[:, a] - FW[:, b]
        win, loss, tie = int((diff > 0).sum()), int((diff < 0).sum()), int((diff == 0).sum())
        boot = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(5000)]
        order = np.argsort(-diff)
        fwres[f"{nm}_{phase}"] = {
            "fw_mean": r(FW[:, b].mean(), 2),
            "mean_diff": r(diff.mean(), 2),
            "median_diff": r(np.median(diff), 1),
            "ci95": [r(np.percentile(boot, 2.5), 2), r(np.percentile(boot, 97.5), 2)],
            "win_loss_tie": [win, loss, tie],
            "p_sign": float(f"{stats.binomtest(win, win + loss, 0.5).pvalue:.3g}"),
            "p_wilcoxon": float(f"{stats.wilcoxon(diff[diff != 0]).pvalue:.3g}"),
            "p_paired_t": float(f"{stats.ttest_1samp(diff, 0).pvalue:.3g}"),
            "mean_without_top4": r(np.delete(diff, order[:4]).mean(), 2),
            "top4": [(d12["tasks"][i], int(diff[i])) for i in order[:4]],
            "bottom3": [(d12["tasks"][i], int(diff[i])) for i in order[-3:]],
        }
OUT["V4g_vs_fastwam_paired"] = fwres

# 시도 수준 검정 (표 1의 FastWAM 91.88 / 91.78, n = 50 과제 x 100회 = 5000 가정)
n5 = 5000
OUT["V4b_trial_level_n5000"] = {
    "clean_ImageWAM_vs_FastWAM": two_prop(round(0.9320 * n5), n5, round(0.9188 * n5), n5),
    "rand_ImageWAM_vs_FastWAM": two_prop(round(0.9356 * n5), n5, round(0.9178 * n5), n5),
    "clean_ImageWAM_vs_LingBot": two_prop(round(0.9320 * n5), n5, round(0.9290 * n5), n5),
    "rand_ImageWAM_vs_LingBot": two_prop(round(0.9356 * n5), n5, round(0.9150 * n5), n5),
}
# 100회/과제가 아니라 50회/과제일 때(n=2500)
n25 = 2500
OUT["V4c_trial_level_n2500"] = {
    "clean_ImageWAM_vs_FastWAM": two_prop(round(0.9320 * n25), n25, round(0.9188 * n25), n25),
    "rand_ImageWAM_vs_FastWAM": two_prop(round(0.9356 * n25), n25, round(0.9178 * n25), n25),
}
OUT["V4d_robotwin_mde_pp_n5000"] = r((1.96 + 0.84) * math.sqrt(2 * 0.92 * 0.08 / 5000) * 100, 2)

# ---------------------------------------------------------------- 표 3, 7 (LIBERO-Plus)
dims = ["Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout"]
T3 = {
    "UniVLA": ([1.8, 46.2, 69.6, 69.0, 81.0, 21.2, 31.9], 42.9),
    "OpenVLA-OFT": ([56.4, 31.9, 79.5, 88.7, 93.3, 75.8, 74.2], 69.6),
    "pi0": ([13.8, 6.0, 58.8, 85.0, 81.4, 79.0, 68.9], 53.6),
    "pi0-Fast": ([65.1, 21.6, 61.0, 73.2, 73.2, 74.4, 68.8], 61.6),
    "WorldVLA": ([0.1, 27.9, 41.6, 43.7, 17.1, 10.9, 38.0], 25.0),
    "FastWAM": ([16.4, 44.5, 68.9, 78.2, 53.7, 37.7, 60.7], 51.5),
    "ImageWAM(OmniGen2)": ([80.0, 49.2, 70.9, 82.6, 69.4, 77.1, 71.8], 71.8),
    "ImageWAM(Ovis-U1)": ([63.3, 58.4, 75.4, 86.3, 66.7, 75.2, 74.6], 71.2),
    "ImageWAM(FLUX.2 4B)": ([80.8, 50.3, 91.4, 98.1, 85.5, 93.8, 80.5], 83.1),
    "ImageWAM(FLUX.2 9B)": ([79.8, 58.7, 95.2, 96.1, 91.2, 93.3, 83.1], 85.2),
}
o = {}
for k, (v, avg) in T3.items():
    m = float(np.mean(v))
    o[k] = {"reported": avg, "simple_mean": r(m, 2), "diff": r(avg - m, 2)}
OUT["V5_table3_simple_mean"] = o
# 차원 가중치 적합: avg = sum_j w_j * v_j, w>=0, sum w = 1 (보고 평균은 소수 한 자리 반올림)
# 방정식 10개, 미지수 7개(합 1 제약 포함 6개 자유도)라 가중치가 유일하지는 않다. 일관성 점검용.
def fit_weights(keys):
    A = np.array([T3[k][0] for k in keys]) / 100.0
    b = np.array([T3[k][1] for k in keys]) / 100.0
    big = 1000.0
    w, _ = nnls(np.vstack([A, big * np.ones((1, 7))]), np.append(b, big))
    pred = {k: float(np.dot(w, T3[k][0])) for k in T3}
    return w, pred


w_all, p_all = fit_weights(list(T3))
w_nf, p_nf = fit_weights([k for k in T3 if k != "FastWAM"])
OUT["V5b_weight_fit_all_rows"] = {
    "weights": {d: r(x, 3) for d, x in zip(dims, w_all)},
    "max_abs_residual": r(max(abs(p_all[k] - T3[k][1]) for k in T3), 2),
    "residual_per_row": {k: r(p_all[k] - T3[k][1], 2) for k in T3},
}
OUT["V5b2_weight_fit_without_FastWAM"] = {
    "weights": {d: r(x, 3) for d, x in zip(dims, w_nf)},
    "max_abs_residual_fitted_rows": r(max(abs(p_nf[k] - T3[k][1]) for k in T3 if k != "FastWAM"), 2),
    "FastWAM_pred_vs_reported": [r(p_nf["FastWAM"], 2), 51.5],
    "note": "FastWAM을 뺀 9행은 한 가중치로 +-0.2 안에서 맞고, FastWAM 51.5는 단순 평균(51.44)에 가깝다",
}
# 가중치 없이 개별 행 일치 여부: 표 3과 표 7 공통 행
OUT["V5c_t3_t7_same_row"] = T3["ImageWAM(FLUX.2 4B)"][0] == [80.8, 50.3, 91.4, 98.1, 85.5, 93.8, 80.5]
# Q1 서술: FLUX.2 4B가 최고인 차원 수 (표 3 전체 행 중)
best = {}
for j, dname in enumerate(dims):
    vals = {k: T3[k][0][j] for k in T3 if k != "ImageWAM(FLUX.2 9B)"}
    best[dname] = max(vals, key=vals.get)
OUT["V5d_best_per_dim_excl9B"] = best
OUT["V5e_flux4b_best_count"] = sum(1 for x in best.values() if x == "ImageWAM(FLUX.2 4B)")
# Q3: 4B -> 9B 차이
v4, v9 = T3["ImageWAM(FLUX.2 4B)"][0], T3["ImageWAM(FLUX.2 9B)"][0]
OUT["V5f_4b_to_9b_delta"] = {d: r(b - a, 1) for d, a, b in zip(dims, v4, v9)}
OUT["V5g_text_85.21_vs_table_85.2"] = {"text": 85.21, "table": 85.2, "simple_mean_9b": r(np.mean(v9), 2)}
# 가중 근사 하에서 4B 대 9B 차이의 검출 불가능성: LIBERO-Plus 시도 수는 논문에 없음 -> 표시만
OUT["V5h_delta_9b_4b_avg"] = r(85.2 - 83.1, 1)

# ---------------------------------------------------------------- 표 4 (실로봇)
T4 = {
    "pi0": [57, 58, 54, 54],
    "pi05": [83, 77, 74, 55],
    "FastWAM": [88, 75, 77, 76],
    "ImageWAM": [94, 84, 78, 82],
}
rep4 = {"pi0": 55.8, "pi05": 72.3, "FastWAM": 79.0, "ImageWAM": 84.5}
o = {}
for k, v in T4.items():
    odd = sum(1 for x in v if x % 2)
    o[k] = {"mean": r(np.mean(v), 3), "reported": rep4[k], "odd_cells_for_50_trials": odd}
OUT["V6_table4_means_parity"] = o
OUT["V6b_infeasible_with_50_trials"] = sum(o[k]["odd_cells_for_50_trials"] for k in o)
OUT["V6c_avg_is_k_over_400"] = {k: r(np.mean(v) * 4, 3) for k, v in T4.items()}  # 100회/과제일 때 정수 k(%)의 합
OUT["V6d_avg_is_k_over_200"] = {k: r(np.mean(v) * 2, 3) for k, v in T4.items()}  # 50회/과제일 때 성공 횟수 (정수여야 함)
OUT["V6e_text_deltas"] = {
    "reported_T1-T4": [6, 9, 1, 6],
    "recomputed": [a - b for a, b in zip(T4["ImageWAM"], T4["FastWAM"])],
}
# Fisher: n=100/과제 가정, 과제별과 합계
o = {}
for t in range(4):
    a, bb = T4["ImageWAM"][t], T4["FastWAM"][t]
    _, p = stats.fisher_exact([[a, 100 - a], [bb, 100 - bb]])
    o[f"T{t + 1}"] = {"imagewam": a, "fastwam": bb, "p_fisher_n100": float(f"{p:.3g}")}
ka, kb = sum(T4["ImageWAM"]), sum(T4["FastWAM"])
o["total_n400"] = two_prop(ka, 400, kb, 400)
ka2, kb2 = ka // 2, kb // 2
o["total_n200_if_50_per_task"] = two_prop(ka2, 200, kb2, 200)
o["pi05_vs_imagewam_total_n400"] = two_prop(ka, 400, sum(T4["pi05"]), 400)
# Holm (4 과제)
ps = sorted([(o[f"T{t + 1}"]["p_fisher_n100"], t + 1) for t in range(4)])
holm = {}
mk = 4
prev = 0
for i, (p, t) in enumerate(ps):
    adj = min(1.0, max(prev, p * (mk - i)))
    holm[f"T{t}"] = float(f"{adj:.3g}")
    prev = adj
o["holm_adjusted"] = holm
OUT["V7_real_world_tests"] = o
OUT["V7b_real_mde_pp_n100_per_task"] = r((1.96 + 0.84) * math.sqrt(2 * 0.8 * 0.2 / 100) * 100, 1)
OUT["V7c_real_mde_pp_n400"] = r((1.96 + 0.84) * math.sqrt(2 * 0.8 * 0.2 / 400) * 100, 1)

# ---------------------------------------------------------------- 표 5, 11 (효율)
lat = {"FastWAM-IDM": 1081, "FastWAM(1step)": 302, "ImageWAM": 263}
fl = {"FastWAM-IDM": 63.65, "FastWAM(1step)": 13.21, "ImageWAM": 9.72}
OUT["V8_table5_ratios"] = {
    "latency_IDM/ImageWAM": r(1081 / 263, 2),
    "flops_IDM/ImageWAM": r(63.65 / 9.72, 2),
    "latency_FastWAM1step/ImageWAM": r(302 / 263, 2),
    "flops_FastWAM1step/ImageWAM": r(13.21 / 9.72, 2),
    "abstract_1/4_ok": 1081 / 263 > 4.0,
    "abstract_1/6_ok": 63.65 / 9.72 > 6.0,
}
t11 = {
    "FastWAM (1x Vid. Denoise)": (302, 1.00),
    "ImageWAM (1x Vid. Denoise)": (263, 1.15),
    "FastWAM (Prefix Only)": (194, 1.56),
    "FastWAM + Compiled": (80, 3.78),
    "ImageWAM (Prefix Only)": (198, 1.53),
    "ImageWAM + Action Loop Compile": (85, 3.55),
    "ImageWAM + Image Prefill Compile": (77, 3.92),
    "ImageWAM + Action Static Graph": (69, 4.38),
}
OUT["V9_table11_speedup"] = {k: {"reported": s, "recomputed": r(302 / ms, 3)} for k, (ms, s) in t11.items()}
OUT["V9b_prefix_only_ImageWAM_vs_FastWAM_ms"] = {"FastWAM": 194, "ImageWAM": 198, "ratio": r(198 / 194, 3)}
OUT["V9c_latency_vs_1081_cache_variants"] = {k: r(1081 / ms, 2) for k, (ms, _) in t11.items()}

# ---------------------------------------------------------------- 표 6과 README 대조
OUT["V10_table6_vs_readme"] = {
    "paper_clean_to_hard": 18.3,
    "readme_pretrained_C2R": 59.0,
    "readme_delta_vs_no_pretrain": 40.9,
    "implied_no_pretrain_C2R": r(59.0 - 40.9, 2),
    "readme_delta_vs_pi05": 11.5,
    "implied_pi05_C2R": r(59.0 - 11.5, 2),
    "gap_to_paper_18.3": r(18.3 - (59.0 - 40.9), 2),
    "naming": "논문 'Clean2Hard' 대 README 'Clean2Random'",
}
OUT["V10b_table6_imagewam_vs_bagelvla"] = {
    "clean_only: ImageWAM 84.4 vs BagelVLA w/ KF 75.3": r(84.4 - 75.3, 1),
    "clean2hard: ImageWAM 18.3 vs BagelVLA w/ KF 20.9": r(18.3 - 20.9, 1),
    "clean2hard: ImageWAM 18.3 vs BagelVLA w/o KF 15.9": r(18.3 - 15.9, 1),
}

# ---------------------------------------------------------------- 표 8-10 (학습 비용)
OUT["V11_gpu_hours_H20"] = {
    "LIBERO OmniGen2/Ovis/FLUX2-4B (8 GPU x 18 h)": 8 * 18,
    "LIBERO FLUX2-9B (8 GPU x 1.6 d)": r(8 * 1.6 * 24, 1),
    "RoboTwin OmniGen2/FLUX2-4B (8 GPU x 5 d)": 8 * 5 * 24,
    "Real-world OmniGen2 (8 GPU x 18 h)": 8 * 18,
}
OUT["V11b_effective_batch"] = {
    "LIBERO FLUX2-4B (batch 10 x 8 GPU, accum 1)": 80,
    "RoboTwin 논문 표 10 (48/GPU x 8)": 48 * 8,
    "RoboTwin 코드 (12 x accum 4 x 8)": 12 * 4 * 8,
    "LIBERO FLUX2-9B 코드 (4 x accum 3 x 8)": 4 * 3 * 8,
    "LIBERO FLUX2-9B 논문 (12 x 8)": 12 * 8,
}

with open(os.path.join(HERE, "recompute_results.json"), "w") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
print("완료: recompute_results.json")
