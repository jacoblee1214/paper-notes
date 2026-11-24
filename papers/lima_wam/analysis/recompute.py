"""LiMA (arXiv:2609.28431 v1) 논문 표 수치를 다시 계산하고 통계 검정을 한다.

입력: 표 값은 아래에 옮겨 적었다(출처는 주석, 원문은 paper/paper.txt). 그림에서 읽은 값은 "그림 판독"으로 표시한다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 표 1, 표 5: 과제당 20회(표 1 캡션, 4.1절). 단계 성공률 x 20 = 단계 성공 횟수.
- 표 2: 변형당 3시드 x 시드당 20회 = 60회(표 2 캡션). "±"는 표본 표준편차(n-1)로 가정하고 모집단 SD도 함께 검사한다.
- 표 3: 시행 수가 적혀 있지 않다. 5% 단위 값이므로 칸당 20회로 가정한다. 어느 과제인지도 적혀 있지 않다.
- 그림 5(레이더): 값은 그림에서 읽었다(그림 판독, 눈금 +-2.5 포인트). 칸당 20회(4.4절).
- 시행은 독립이라고 가정한다(같은 장면 배치를 반복하면 독립이 아닐 수 있다).
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "recompute_results.json"
RES = {}

TASKS = ["Stack Cup", "Roll T-shirt", "Cook Rice", "Make Sandwich", "Make Coffee", "Assemble Package"]
METHODS = ["GR00T N1.6", "VPP", "InternVLA-A1", "Cosmos-Policy", "LiMA"]
# 표 1: (SR, PSR) 과제 순서대로, 시간(ms)
T1 = {
    "GR00T N1.6":    [(85.0, 87.5), (70.0, 71.7), (70.0, 72.0), (65.0, 72.5), (55.0, 71.7), (50.0, 61.7)],
    "VPP":           [(80.0, 80.0), (55.0, 56.7), (55.0, 60.0), (50.0, 61.3), (50.0, 56.7), (40.0, 48.3)],
    "InternVLA-A1":  [(90.0, 90.0), (75.0, 88.3), (70.0, 78.0), (65.0, 77.5), (50.0, 66.7), (50.0, 58.3)],
    "Cosmos-Policy": [(80.0, 82.5), (60.0, 70.0), (60.0, 67.0), (60.0, 71.3), (55.0, 63.3), (45.0, 51.7)],
    "LiMA":          [(90.0, 95.0), (75.0, 83.3), (80.0, 82.0), (70.0, 80.0), (60.0, 70.0), (50.0, 63.3)],
}
TIME_MS = {"GR00T N1.6": 270, "VPP": 225, "InternVLA-A1": 360, "Cosmos-Policy": 600, "LiMA": 325}
# 표 5 (부록 C.2): 단계별 성공률(%), 단계 수는 과제마다 2/3/5/4/3/3
T5 = {
    "Stack Cup": {"LiMA": [100, 90], "Cosmos-Policy": [85, 80], "GR00T N1.6": [90, 85], "VPP": [80, 80], "InternVLA-A1": [90, 90]},
    "Roll T-shirt": {"LiMA": [95, 80, 75], "Cosmos-Policy": [80, 70, 60], "GR00T N1.6": [75, 70, 70], "VPP": [60, 55, 55], "InternVLA-A1": [100, 90, 75]},
    "Cook Rice": {"LiMA": [85, 85, 80, 80, 80], "Cosmos-Policy": [75, 70, 70, 60, 60], "GR00T N1.6": [80, 70, 70, 70, 70], "VPP": [65, 60, 60, 60, 55], "InternVLA-A1": [85, 80, 80, 75, 70]},
    "Make Sandwich": {"LiMA": [90, 85, 75, 70], "Cosmos-Policy": [90, 70, 65, 60], "GR00T N1.6": [80, 75, 70, 65], "VPP": [75, 60, 60, 50], "InternVLA-A1": [100, 75, 70, 65]},
    "Make Coffee": {"LiMA": [80, 70, 60], "Cosmos-Policy": [70, 65, 55], "GR00T N1.6": [80, 80, 55], "VPP": [60, 60, 50], "InternVLA-A1": [80, 70, 50]},
    "Assemble Package": {"LiMA": [70, 70, 50], "Cosmos-Policy": [60, 50, 45], "GR00T N1.6": [70, 65, 50], "VPP": [60, 45, 40], "InternVLA-A1": [65, 60, 50]},
}
N_TRIAL = 20


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def r1(x):
    return round(float(x), 2)


# ---------------------------------------------------------------- V1: 표 5 대 표 1 (PSR = 단계 성공률 평균, SR = 마지막 단계)
v1 = []
for t_i, task in enumerate(TASKS):
    for m in METHODS:
        st = T5[task][m]
        sr, psr = T1[m][t_i]
        mean_st = sum(st) / len(st)
        v1.append({"task": task, "method": m, "stages": st, "psr_table1": psr, "psr_from_table5": r1(mean_st),
                   "psr_match": abs(mean_st - psr) <= 0.051,
                   "sr_table1": sr, "last_stage": st[-1], "sr_equals_last_stage": sr == st[-1],
                   "monotone_nonincreasing": all(a >= b for a, b in zip(st, st[1:]))})
RES["V1_psr_vs_stages"] = {
    "rows": v1,
    "n_rows": len(v1),
    "psr_all_match": all(r["psr_match"] for r in v1),
    "sr_equals_last_stage_all": all(r["sr_equals_last_stage"] for r in v1),
    "monotone_all": all(r["monotone_nonincreasing"] for r in v1),
    "psr_mismatch_rows": [r for r in v1 if not r["psr_match"]],
    "sr_not_last_rows": [r for r in v1 if not r["sr_equals_last_stage"]],
    "non_monotone_rows": [r for r in v1 if not r["monotone_nonincreasing"]],
}

# ---------------------------------------------------------------- V2: 평균 (초록, 그림 1)
means = {}
for m in METHODS:
    sr = [x[0] for x in T1[m]]
    psr = [x[1] for x in T1[m]]
    means[m] = {"sr_mean": r1(sum(sr) / 6), "psr_mean": r1(sum(psr) / 6), "sr_sum_counts_of_120": int(round(sum(sr) * N_TRIAL / 100))}
RES["V2_means"] = {"per_method": means, "abstract_claims": {"overall_sr": 70.8, "avg_psr": 78.9},
                   "figure1_bars_sr": {"LiMA": 70.8, "Cosmos-Policy": 60.0, "InternVLA-A1": 66.7, "VPP": 55.0, "GR00T N1.6": 65.8},
                   "figure1_bars_time_s": {"LiMA": 0.325, "Cosmos-Policy": 0.6, "InternVLA-A1": 0.36, "VPP": 0.225, "GR00T N1.6": 0.27}}

# ---------------------------------------------------------------- V3: 지연
lat = {"reduction_vs_cosmos_pct": r1(100 * (1 - 325 / 600)), "claim_pct": 45.8}
lat["vs_others_ratio"] = {m: r1(TIME_MS["LiMA"] / TIME_MS[m]) for m in METHODS if m != "LiMA"}
# 평균 지연 T_avg = (T_D + 4 T_R)/4 = 325 -> T_D + 4 T_R = 1300 ms (4.2절)
lat["amortized_total_ms"] = 4 * 325
# 동기식(w/o Fast-Slow) 0.45 s = T_D + T_R 로 읽으면
td_plus_tr = 450
t_r = (4 * 325 - td_plus_tr) / 3
t_d = td_plus_tr - t_r
lat["sync_reading"] = {"T_D_plus_T_R_ms": td_plus_tr, "T_R_ms": r1(t_r), "T_D_ms": r1(t_d),
                       "note": "T_D + 4 T_R = 1300, T_D + T_R = 450 을 풀면 Refiner 한 번(약 283 ms)이 Dreamer 한 번(약 167 ms)보다 길다"}
RES["V3_latency"] = lat

# ---------------------------------------------------------------- V4: 표 3의 지연 모양 (b) 스텝 수, (c) 비율
steps = np.array([5, 10, 15, 20])
t_steps = np.array([230, 325, 430, 510])
A = np.vstack([steps, np.ones(4)]).T
coef, res, *_ = np.linalg.lstsq(A, t_steps, rcond=None)
pred = A @ coef
RES["V4_table3b_latency_vs_steps"] = {"steps": steps.tolist(), "time_ms": t_steps.tolist(), "slope_ms_per_step": r1(coef[0]),
                                       "intercept_ms": r1(coef[1]), "residuals_ms": [r1(x) for x in (t_steps - pred)],
                                       "increments_ms": np.diff(t_steps).tolist(),
                                       "r2": round(float(1 - ((t_steps - pred) ** 2).sum() / ((t_steps - t_steps.mean()) ** 2).sum()), 4)}
ns = np.array([1, 2, 3, 4])
t_ratio = np.array([280, 310, 325, 340])
slope_c = np.polyfit(ns, t_ratio, 1)
RES["V4_table3c_latency_vs_ratio"] = {"slow_steps": ns.tolist(), "time_ms": t_ratio.tolist(), "increments_ms": np.diff(t_ratio).tolist(),
                                       "slope_ms_per_slow_step": r1(slope_c[0]), "residuals_ms": [r1(x) for x in (t_ratio - np.polyval(slope_c, ns))]}
# 겉보기 비용 모델: 슬로우 한 스텝 비용 d, 패스트 한 스텝 비용 r, 총 10스텝 중 Ns는 슬로우.
#   T(Ns) = c + Ns*d/4 + (10-Ns)*r      (Dreamer는 4번에 한 번)
#   동기식(같은 10스텝) T_sync = c' + Ns*d + (10-Ns)*r
# 3:7 지점에서 T = 325, T_sync = 450 이면 (3/4)*(3*d) ... 아래에서 d를 구하고 (c' = c로 둔 경우),
# 비율 표의 기울기(d/4 - r)와 비교한다.
d = (450 - 325) / (3 * 0.75)
r_needed = d / 4 - slope_c[0]
RES["V4_cost_model"] = {"d_ms_per_slow_step_if_overhead_zero": r1(d), "implied_r_ms_per_fast_step": r1(r_needed),
                         "note": "d = 125 / (3 x 0.75) = 55.6 ms. 표 3(c)의 기울기 약 +19.5 ms/스텝 = d/4 - r 이면 r = -5.6 ms 로 음수. "
                                 "슬로우 고정 오버헤드가 있으면 d는 더 작아져 r은 더 음수. 비율 열이 '디노이징 스텝 배분'이라는 읽기와 동기식 450 ms가 양립하지 않는다"}

# ---------------------------------------------------------------- V5: 표 2 (3시드 평균 ± SD) 정수 복원
T2 = {
    "w/o Future Visual Intent": {"Cook Rice": (61.7, 7.6), "Stack Cup": (76.7, 2.9)},
    "Gaussian Init. + Cross-Attn": {"Cook Rice": (70.0, 5.0), "Stack Cup": (78.3, 2.9)},
    "Flow Matching": {"Cook Rice": (73.3, 2.9), "Stack Cup": (83.3, 5.8)},
    "LiMA": {"Cook Rice": (78.3, 2.9), "Stack Cup": (93.3, 5.8)},
}
grid = list(range(0, 101, 5))
v5 = []
for var, cells in T2.items():
    for task, (mean, sd) in cells.items():
        sols = []
        for tri in itertools.combinations_with_replacement(grid, 3):
            if abs(np.mean(tri) - mean) <= 0.051:
                s1 = float(np.std(tri, ddof=1))
                s0 = float(np.std(tri, ddof=0))
                if abs(s1 - sd) <= 0.051:
                    sols.append({"seeds": list(tri), "sd_ddof1": r1(s1), "match": "ddof1"})
                elif abs(s0 - sd) <= 0.051:
                    sols.append({"seeds": list(tri), "sd_ddof0": r1(s0), "match": "ddof0"})
        v5.append({"variant": var, "task": task, "mean": mean, "sd": sd, "k_of_60": round(mean * 60 / 100),
                   "mean_is_integer_count_of_60": abs(mean * 60 / 100 - round(mean * 60 / 100)) < 0.03,
                   "triplets_5pct_grid": sols})
RES["V5_table2_seed_reconstruction"] = v5

# ---------------------------------------------------------------- V6: 표 2 vs 표 1 vs 표 3(a)
RES["V6_cross_table"] = {
    "cook_rice": {"table1_20trials": 80.0, "table2_3seeds_mean": 78.3, "table3a_full": 80.0, "fig5_base_read": 80},
    "stack_cup": {"table1_20trials": 90.0, "table2_3seeds_mean": 93.3},
    "note": "표 1의 두 값(80.0, 90.0)은 표 2의 3시드 복원 후보(75/80/80, 90/90/100)에 들어 있는 값과 같다. 어느 시드인지, 표 1이 그 중 하나인지는 적혀 있지 않다",
}

# ---------------------------------------------------------------- V7: 표 2 변형 대 LiMA (60회 풀링, Fisher) + 시드 단위 Welch
v7 = []
for task in ["Cook Rice", "Stack Cup"]:
    kl = round(T2["LiMA"][task][0] * 60 / 100)
    for var in ["w/o Future Visual Intent", "Gaussian Init. + Cross-Attn", "Flow Matching"]:
        kv = round(T2[var][task][0] * 60 / 100)
        d, lo, hi = newcombe(kl, 60, kv, 60)
        # 시드 단위 Welch t (요약 통계만 사용)
        m1, s1 = T2["LiMA"][task]
        m2, s2 = T2[var][task]
        t, p = stats.ttest_ind_from_stats(m1, s1, 3, m2, s2, 3, equal_var=False)
        v7.append({"task": task, "variant": var, "lima_k": kl, "var_k": kv, "diff_pp": r1(100 * d),
                   "newcombe95_pp": [r1(100 * lo), r1(100 * hi)], "fisher_p_60": r1(fisher(kl, 60, kv, 60)) if False else round(fisher(kl, 60, kv, 60), 4),
                   "welch_p_seed_level": round(float(p), 4)})
RES["V7_table2_tests"] = v7

# ---------------------------------------------------------------- V8: 표 3 (칸당 20회 가정) 쌍 비교
def k20(pct):
    return int(round(pct * N_TRIAL / 100))


T3a = {"Full (LiMA)": 80, "w/o Adapt.": 60, "w/o Tokens": 55, "w/o Fast-Slow": 80}
T3b = {5: 65, 10: 80, 15: 85, 20: 80}
T3c = {"1:9": 50, "2:8": 65, "3:7": 80, "4:6": 70}
v8 = []
for name, tbl, ref in [("3a", T3a, "Full (LiMA)"), ("3b", T3b, 10), ("3c", T3c, "3:7")]:
    for k_, v in tbl.items():
        if k_ == ref:
            continue
        v8.append({"table": name, "ref": str(ref), "other": str(k_), "ref_pct": tbl[ref], "other_pct": v,
                   "diff_pp": tbl[ref] - v, "fisher_p_n20": round(fisher(k20(tbl[ref]), 20, k20(v), 20), 4)})
RES["V8_table3_tests"] = v8
RES["V8_table3_note"] = {
    "best_in_3b": "15 스텝(85%)이 10 스텝(80%)보다 높다. 본문은 10을 '실용적 절충'으로 고름",
    "best_in_3c": "3:7(80%)이 가장 높다. 기본 설정과 같다",
    "wilson95_80pct_n20": [r1(100 * x) for x in wilson(16, 20)],
}

# ---------------------------------------------------------------- V9: 표 1 SR 비교 (과제별 Fisher, 풀링, 쌍체)
v9 = {"per_task": [], "pooled": {}, "paired": {}}
for t_i, task in enumerate(TASKS):
    kl = k20(T1["LiMA"][t_i][0])
    best = max((m for m in METHODS if m != "LiMA"), key=lambda m: T1[m][t_i][0])
    for m in METHODS:
        if m == "LiMA":
            continue
        km = k20(T1[m][t_i][0])
        v9["per_task"].append({"task": task, "baseline": m, "lima": kl, "base": km, "diff_pp": 5.0 * (kl - km),
                               "fisher_p": round(fisher(kl, 20, km, 20), 4)})
lima_tot = sum(k20(x[0]) for x in T1["LiMA"])
for m in METHODS:
    if m == "LiMA":
        continue
    km = sum(k20(x[0]) for x in T1[m])
    d, lo, hi = newcombe(lima_tot, 120, km, 120)
    v9["pooled"][m] = {"lima": lima_tot, "base": km, "diff_pp": r1(100 * d), "newcombe95_pp": [r1(100 * lo), r1(100 * hi)],
                       "fisher_p": round(fisher(lima_tot, 120, km, 120), 4)}
    diffs = np.array([T1["LiMA"][i][0] - T1[m][i][0] for i in range(6)])
    nz = diffs[diffs != 0]
    wins = int((diffs > 0).sum())
    losses = int((diffs < 0).sum())
    sign_p = float(stats.binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0
    w = stats.wilcoxon(diffs[diffs != 0]) if len(nz) else None
    v9["paired"][m] = {"sr_diffs": diffs.tolist(), "wins": wins, "losses": losses, "ties": int((diffs == 0).sum()),
                       "sign_p_two_sided": round(sign_p, 4), "wilcoxon_p": round(float(w.pvalue), 4) if w is not None else None,
                       "mean_diff": r1(diffs.mean())}
RES["V9_sr_comparisons"] = v9

# PSR: 단계 단위 풀링(단계 수 x 20회). 단계 시도를 독립으로 두는 근사이며 같은 시행 내 단계는 상관된다.
v10 = {}
tot_stage = sum(len(T5[t]["LiMA"]) for t in TASKS) * N_TRIAL
for m in METHODS:
    k = int(round(sum(sum(T5[t][m]) for t in TASKS) * N_TRIAL / 100))
    v10[m] = {"stage_successes": k, "stage_trials": tot_stage, "rate_pct": r1(100 * k / tot_stage)}
for m in METHODS:
    if m == "LiMA":
        continue
    v10[m]["lima_minus_pp"] = r1(v10["LiMA"]["rate_pct"] - v10[m]["rate_pct"])
    v10[m]["fisher_p_stage_pooled"] = round(fisher(v10["LiMA"]["stage_successes"], tot_stage, v10[m]["stage_successes"], tot_stage), 4)
    diffs = np.array([T1["LiMA"][i][1] - T1[m][i][1] for i in range(6)])
    wins, losses = int((diffs > 0).sum()), int((diffs < 0).sum())
    v10[m]["psr_wins_losses"] = [wins, losses]
    v10[m]["psr_sign_p"] = round(float(stats.binomtest(wins, wins + losses, 0.5).pvalue), 4)
    v10[m]["psr_mean_diff"] = r1(diffs.mean())
RES["V10_psr_pooled"] = v10

# 본문 서술: Cook Rice와 Make Sandwich에서 PSR이 "improved", 짧은 과제는 "comparable"
txt = {}
for task in TASKS:
    i = TASKS.index(task)
    best_base_psr = max((T1[m][i][1], m) for m in METHODS if m != "LiMA")
    best_base_sr = max((T1[m][i][0], m) for m in METHODS if m != "LiMA")
    txt[task] = {"lima_sr": T1["LiMA"][i][0], "best_baseline_sr": best_base_sr, "lima_psr": T1["LiMA"][i][1],
                 "best_baseline_psr": best_base_psr, "psr_margin": r1(T1["LiMA"][i][1] - best_base_psr[0])}
RES["V11_text_vs_table"] = txt

# ---------------------------------------------------------------- V12: 일반화 (그림 5 판독, Cook Rice, 칸당 20회)
# 그림 5의 마커 좌표를 픽셀로 읽어 반지름 -> 값으로 변환했다(원점 0, 5% 단위로 반올림). 겹친 마커(Cosmos 배경)는 점선 가장자리로 추정.
FIG5 = {
    "LiMA": {"Base": 80, "Cluttered": 75, "Unseen Object": 70, "Unseen Background": 60, "Unseen Lighting": 60},
    "Cosmos-Policy": {"Base": 60, "Cluttered": 50, "Unseen Object": 50, "Unseen Background": 50, "Unseen Lighting": 55},
    "GR00T N1.6": {"Base": 70, "Cluttered": 60, "Unseen Object": 55, "Unseen Background": 50, "Unseen Lighting": 50},
}
v12 = {"read_values": FIG5, "pairs": [], "retention": {}}
for m, row in FIG5.items():
    v12["retention"][m] = {k: r1(100 * v / row["Base"]) for k, v in row.items() if k != "Base"}
    v12["retention"][m]["mean_ood_pct"] = r1(np.mean([row[k] for k in row if k != "Base"]))
    v12["retention"][m]["mean_drop_pp"] = r1(row["Base"] - np.mean([row[k] for k in row if k != "Base"]))
for setting in ["Cluttered", "Unseen Object", "Unseen Background", "Unseen Lighting"]:
    for m in ["Cosmos-Policy", "GR00T N1.6"]:
        kl, km = k20(FIG5["LiMA"][setting]), k20(FIG5[m][setting])
        v12["pairs"].append({"setting": setting, "baseline": m, "lima": FIG5["LiMA"][setting], "base": FIG5[m][setting],
                             "diff_pp": FIG5["LiMA"][setting] - FIG5[m][setting], "fisher_p_n20": round(fisher(kl, 20, km, 20), 4)})
# 네 OOD 설정 풀링 (80회)
for m in ["Cosmos-Policy", "GR00T N1.6"]:
    kl = sum(k20(FIG5["LiMA"][s]) for s in ["Cluttered", "Unseen Object", "Unseen Background", "Unseen Lighting"])
    km = sum(k20(FIG5[m][s]) for s in ["Cluttered", "Unseen Object", "Unseen Background", "Unseen Lighting"])
    d, lo, hi = newcombe(kl, 80, km, 80)
    v12["pooled_ood_" + m] = {"lima": kl, "base": km, "diff_pp": r1(100 * d), "newcombe95_pp": [r1(100 * lo), r1(100 * hi)],
                              "fisher_p": round(fisher(kl, 80, km, 80), 4)}
# 기준(Base)과 OOD 평균의 하락
RES["V12_generalization"] = v12

# ---------------------------------------------------------------- V13: 정수 점검 (표 1 모든 SR, PSR)
chk = []
for m in METHODS:
    for t_i, task in enumerate(TASKS):
        sr, psr = T1[m][t_i]
        n_stage = len(T5[task][m]) * N_TRIAL
        k_psr = psr * n_stage / 100
        chk.append({"method": m, "task": task, "sr_is_k_of_20": abs(sr * 20 / 100 - round(sr * 20 / 100)) < 1e-6,
                    "psr_k": round(k_psr, 2), "psr_integer_of_stage_trials": abs(k_psr - round(k_psr)) < 0.06, "n_stage_trials": n_stage})
RES["V13_integer_feasibility"] = {"all_sr_integer": all(c["sr_is_k_of_20"] for c in chk),
                                  "all_psr_integer": all(c["psr_integer_of_stage_trials"] for c in chk),
                                  "failures": [c for c in chk if not (c["sr_is_k_of_20"] and c["psr_integer_of_stage_trials"])]}

# ---------------------------------------------------------------- V14: 잠재 텐서 산술 (부록 A.2)
RES["V14_latent_arithmetic"] = {"C": 16, "N_slot": 8, "H": 28, "W": 28, "tokens_per_slot_patch2": (28 // 2) * (28 // 2),
                                "pixels_if_vae_stride8": 28 * 8, "elements_per_latent_tensor": 16 * 8 * 28 * 28,
                                "note": "H=W=28 은 Wan2.1 VAE 공간 압축 8배 기준 224 x 224 입력에 해당. Cosmos-Policy의 224 입력과 같다. 패치 크기 2는 Cosmos-Predict2 기본값이며 논문에는 없음"}


# ---------------------------------------------------------------- V15: 지연의 FLOPs 어림 (구성은 공개 Cosmos-Predict2 2B 설정을 정적으로 읽은 값)
# 2B DiT: 28블록, 은닉 2048, 공간 패치 2 (Cosmos-Policy 저장소 predict2 설정). 224 입력 -> 잠재 28 x 28 -> 슬롯당 14 x 14 = 196 토큰.
def dit_flops(params, tokens, layers, dim):
    return 2 * params * tokens + 4 * layers * tokens ** 2 * dim
tok_lima = 8 * 196          # 부록 A.2: 슬롯 8개
tok_cosmos_aloha = 11 * 196  # Cosmos-Policy ALOHA 설정: 잠재 프레임 11개
f_dream = dit_flops(2e9, tok_lima, 28, 2048)
f_refine = dit_flops(1e9, tok_lima, 28, 1408)   # Refiner 1B의 깊이·폭은 논문에 없다. 28블록, 은닉 1408(약 1B)로 가정
f_cosmos = dit_flops(2e9, tok_cosmos_aloha, 28, 2048)
RES["V15_flops_order_of_magnitude"] = {
    "tokens_lima": tok_lima, "tokens_cosmos_aloha": tok_cosmos_aloha,
    "dreamer_tflop_per_step": r1(f_dream / 1e12), "refiner_tflop_per_step_assumed_dims": r1(f_refine / 1e12),
    "cosmos_tflop_per_step": r1(f_cosmos / 1e12),
    "dreamer_3_steps_tflop": r1(3 * f_dream / 1e12), "refiner_7_steps_tflop": r1(7 * f_refine / 1e12),
    "cosmos_5_steps_tflop": r1(5 * f_cosmos / 1e12),
    "effective_tflops_if_TD_167ms": r1(3 * f_dream / 1e12 / 0.1667), "effective_tflops_if_TR_283ms": r1(7 * f_refine / 1e12 / 0.2833),
    "effective_tflops_cosmos_600ms_5steps": r1(5 * f_cosmos / 1e12 / 0.6),
    "note": "스텝 수는 3:7 비율과 총 10스텝을 디노이징 스텝 배분으로 읽은 가정, Cosmos 5스텝은 Cosmos-Policy 논문의 기본 계획 설정에서 온 가정. "
            "VAE 인코딩과 텍스트 인코더는 제외. 유효 처리량이 같은 자릿수(수십-백 TFLOP/s)라서 325/600 ms 값은 크기 면에서 모순이 아니다",
}

# ---------------------------------------------------------------- V16: 검출 한계 (arcsine 근사, 양측 5%, 검정력 80%)
def mdd(p0, n):
    h = (1.959964 + 0.841621) / math.sqrt(n / 2)
    phi = 2 * math.asin(math.sqrt(p0)) + h
    return 100 * (math.sin(min(phi, math.pi) / 2) ** 2 - p0)
RES["V16_detectable_gap_pp"] = {f"n{n}_base{int(100 * b)}": r1(mdd(b, n)) for n in (20, 60, 80, 120) for b in (0.5, 0.6, 0.7)}

# ---------------------------------------------------------------- 출력
OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")
print("V1 PSR 일치:", RES["V1_psr_vs_stages"]["psr_all_match"], "| SR = 마지막 단계:", RES["V1_psr_vs_stages"]["sr_equals_last_stage_all"],
      "| 단조:", RES["V1_psr_vs_stages"]["monotone_all"])
print("V2 평균:", {m: (v["sr_mean"], v["psr_mean"]) for m, v in means.items()})
print("V3 지연:", lat["reduction_vs_cosmos_pct"], lat["sync_reading"])
print("V4 3b 기울기:", RES["V4_table3b_latency_vs_steps"]["slope_ms_per_step"], RES["V4_table3b_latency_vs_steps"]["intercept_ms"],
      "| 3c 증가:", RES["V4_table3c_latency_vs_ratio"]["increments_ms"], "| r:", RES["V4_cost_model"]["implied_r_ms_per_fast_step"])
for r in v5:
    print("V5", r["variant"], r["task"], r["mean"], r["sd"], r["k_of_60"], [s["seeds"] for s in r["triplets_5pct_grid"]])
for r in v7:
    print("V7", r["task"], r["variant"], r["diff_pp"], r["newcombe95_pp"], "fisher", r["fisher_p_60"], "welch", r["welch_p_seed_level"])
for r in v8:
    print("V8", r["table"], r["ref"], "vs", r["other"], r["diff_pp"], r["fisher_p_n20"])
for m, r in v9["pooled"].items():
    print("V9 pooled", m, r, v9["paired"][m])
print("V10", {m: (v["rate_pct"], v.get("fisher_p_stage_pooled")) for m, v in v10.items()})
print("V11", txt)
print("V12 retention", v12["retention"])
for r in v12["pairs"]:
    print("V12", r)
print("V12 pooled", v12["pooled_ood_Cosmos-Policy"], v12["pooled_ood_GR00T N1.6"])
print("V13", RES["V13_integer_feasibility"]["all_sr_integer"], RES["V13_integer_feasibility"]["all_psr_integer"], RES["V13_integer_feasibility"]["failures"])
print("V16", RES["V16_detectable_gap_pp"])
print("V15", RES["V15_flops_order_of_magnitude"])
print("wrote", OUT)
