"""ELASTIC (arXiv:2606.31132) 그림 값 재계산과 검정.

논문에는 표가 거의 없고 수치는 그림에 있다. 아래 DIGITIZED_* 값은 그림 2, 5, 11, 13, 14의 래스터 이미지에서
마커 중심을 찍어 읽은 값이다(읽기 오차 성공률 약 ±0.3 %p, 지연 ±0.7 ms로 본다. 그림 11의 평균 막대와
그림 5, 11의 라벨 숫자는 논문에 적힌 값이다). 재계산은 이 값에서 시작한다.

실행: python3 analysis/recompute.py   (numpy, scipy, statsmodels, 수 초)
출력: 표준 출력과 analysis/recompute_results.json
"""
import json
import math
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_confint, proportion_effectsize

OUT = Path(__file__).with_name("recompute_results.json")
R = {}


def wilson(k, n):
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return [round(100 * lo, 1), round(100 * hi, 1)]


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def newcombe_diff(k1, n1, k2, n2):
    """두 비율 차이(1-2)의 Newcombe(Wilson 기반) 95% 구간, %p."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = proportion_confint(k1, n1, alpha=0.05, method="wilson")
    l2, u2 = proportion_confint(k2, n2, alpha=0.05, method="wilson")
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return [round(100 * lo, 1), round(100 * hi, 1)]


# ---------------------------------------------------------------- V1 LIBERO-10 성공률 (그림 11)
# 과제 0..9, 과제당 20회. 막대 값(5 %p 단위)을 읽음.
LIBERO = {
    "ELASTIC":         [95, 100, 95, 95, 95, 100, 100, 100, 65, 95],
    "pi0.5":           [95, 100, 95, 100, 100, 100, 90, 100, 50, 85],
    "BoN(10,10)":      [100, 100, 100, 95, 100, 100, 95, 100, 65, 100],
    "Sequential(18,1)": [85, 100, 95, 95, 100, 100, 100, 100, 70, 95],
    "Fixed(3,7)":      [95, 100, 95, 95, 95, 100, 95, 100, 60, 90],
}
PAPER_MEAN = {"ELASTIC": 94.0, "pi0.5": 91.5, "BoN(10,10)": 95.5, "Sequential(18,1)": 94.0, "Fixed(3,7)": 92.5}
lib = {}
for k, v in LIBERO.items():
    cnt = [round(x / 5) for x in v]                      # 20회 중 성공 횟수
    tot = sum(cnt)
    lib[k] = {"counts": cnt, "total_of_200": tot, "mean_pct": tot / 2, "paper_mean": PAPER_MEAN[k],
              "matches_paper": abs(tot / 2 - PAPER_MEAN[k]) < 1e-9, "wilson95": wilson(tot, 200)}
R["V1_libero_means"] = lib
base = lib["pi0.5"]["total_of_200"]
cmp_ = {}
for k in ("ELASTIC", "BoN(10,10)", "Sequential(18,1)", "Fixed(3,7)"):
    t = lib[k]["total_of_200"]
    cmp_[f"{k} vs pi0.5"] = {"diff_pp": (t - base) / 2, "fisher_p": round(fisher(t, 200, base, 200), 3),
                             "newcombe95": newcombe_diff(t, 200, base, 200)}
t = lib["ELASTIC"]["total_of_200"]
for k in ("BoN(10,10)", "Sequential(18,1)", "Fixed(3,7)"):
    cmp_[f"ELASTIC vs {k}"] = {"diff_pp": (t - lib[k]["total_of_200"]) / 2,
                               "fisher_p": round(fisher(t, 200, lib[k]["total_of_200"], 200), 3)}
R["V1_libero_tests"] = cmp_
# 과제 단위 쌍체 부호 반전 검정: ELASTIC - pi0.5 과제별 성공 횟수 차이
d = np.array(lib["ELASTIC"]["counts"]) - np.array(lib["pi0.5"]["counts"])
obs = d.sum()
nz = d[d != 0]
tot_ge = 0
for signs in product([1, -1], repeat=len(nz)):
    if abs(np.dot(signs, nz)) >= abs(obs):
        tot_ge += 1
R["V1_libero_signflip"] = {"per_task_diff_trials": d.tolist(), "sum": int(obs),
                           "two_sided_p_signflip": tot_ge / 2 ** len(nz), "n_nonzero_tasks": int(len(nz))}
# 과제 8 (가장 어려운 과제) 따로
R["V1_libero_task8"] = {k: lib[k]["counts"][8] for k in LIBERO} | {
    "fisher_ELASTIC_vs_base_20trials": round(fisher(13, 20, 10, 20), 3)}

# ---------------------------------------------------------------- V2 LIBERO 지연 (그림 11, 값은 논문에 라벨로 적힘)
LAT = {"ELASTIC": 64.5, "pi0.5": 68.7, "BoN(10,10)": 94.0, "Sequential(18,1)": 96.5, "Fixed(3,7)": 67.7}
v2 = {}
for k, v in LAT.items():
    if k == "pi0.5":
        continue
    # 라벨이 소수 첫째 자리 반올림이므로 ±0.05 ms 범위의 최소/최대 비율
    lo = (v - 0.05) / (LAT["pi0.5"] + 0.05) - 1
    hi = (v + 0.05) / (LAT["pi0.5"] - 0.05) - 1
    v2[k] = {"pct_vs_base": round(100 * (v / LAT["pi0.5"] - 1), 2),
             "range_given_rounding": [round(100 * lo, 2), round(100 * hi, 2)]}
v2["paper_text"] = {"BoN": "+36.8%", "Sequential": "+40.3%", "ELASTIC": "-6.1%"}
R["V2_libero_latency"] = v2

# ---------------------------------------------------------------- V3 비용 정합 (식 1, α=0.13 β=0.1)
alpha, beta = 0.13, 0.1
pts = {"ELASTIC(2.6,6.9)": (2.6, 6.9), "pi0.5(10,1)": (10, 1), "BoN(10,10)": (10, 10),
       "Sequential(18,1)": (18, 1), "Fixed(3,7)": (3, 7)}
v3 = {k: {"alphaL_plus_beta(P-1)": round(alpha * L + beta * (P - 1), 3), "alphaL_plus_betaP": round(alpha * L + beta * P, 3)}
      for k, (L, P) in pts.items()}
Lb, Pb = 2.6, 6.9
v3["sequential_L_by_App7.2.1_formula"] = round((alpha * Lb + beta * (Pb - 1)) / alpha, 2)
R["V3_libero_cost_match"] = v3

# ---------------------------------------------------------------- V4 실로봇 (n=20, 그림 5)
HW_N = 20
HW = {"pi0.5 (10,1)": 9, "ELASTIC (2.6,5.6)": 16, "ELASTIC (3.6,2.9)": 16, "ELASTIC (1.9,3.5)": 13,
      "BoN (10,10)": 16, "Fixed (2,10)": 14, "Fixed (2,4)": 9, "Fixed (2,3)": 8}
v4 = {k: {"k": c, "pct": 100 * c / HW_N, "wilson95": wilson(c, HW_N)} for k, c in HW.items()}
R["V4_hw_counts"] = v4
tests = {}
pairs = [("ELASTIC (2.6,5.6)", "pi0.5 (10,1)"), ("ELASTIC (3.6,2.9)", "pi0.5 (10,1)"),
         ("ELASTIC (1.9,3.5)", "pi0.5 (10,1)"), ("BoN (10,10)", "pi0.5 (10,1)"),
         ("ELASTIC (2.6,5.6)", "BoN (10,10)"), ("ELASTIC (2.6,5.6)", "Fixed (2,10)"),
         ("ELASTIC (1.9,3.5)", "Fixed (2,4)"), ("ELASTIC (1.9,3.5)", "Fixed (2,3)"),
         ("Fixed (2,10)", "pi0.5 (10,1)")]
for a, b in pairs:
    tests[f"{a} vs {b}"] = {"diff_pp": 100 * (HW[a] - HW[b]) / HW_N, "fisher_p": round(fisher(HW[a], 20, HW[b], 20), 3),
                            "newcombe95": newcombe_diff(HW[a], 20, HW[b], 20)}
R["V4_hw_tests"] = tests
pw = NormalIndPower()
R["V4_hw_power"] = {
    "n_per_arm_80_vs_45_power80": round(pw.solve_power(proportion_effectsize(0.80, 0.45), alpha=0.05, power=0.8)),
    "n_per_arm_80_vs_70_power80": round(pw.solve_power(proportion_effectsize(0.80, 0.70), alpha=0.05, power=0.8)),
    "n_per_arm_65_vs_45_power80": round(pw.solve_power(proportion_effectsize(0.65, 0.45), alpha=0.05, power=0.8)),
    "power_n20_80_vs_45": round(float(pw.power(proportion_effectsize(0.80, 0.45), nobs1=20, alpha=0.05)), 2),
    "power_n20_80_vs_70": round(float(pw.power(proportion_effectsize(0.80, 0.70), nobs1=20, alpha=0.05)), 2),
}
# 무승부 동등성(비열등)에 필요한 n: ELASTIC=BoN=80% 일 때 차이 95% 구간 폭
R["V4_hw_elastic_vs_bon_ci"] = newcombe_diff(16, 20, 16, 20)

# ---------------------------------------------------------------- V5 지연 감소율 (그림 5, 마커 중심 읽음)
FIG5_LAT = {"pi0.5 (10,1)": 69.1, "BoN (10,10)": 95.5, "ELASTIC (2.6,5.6)": 62.5, "ELASTIC (3.6,2.9)": 64.6,
            "ELASTIC (1.9,3.5)": 58.0, "Fixed (2,10)": 63.6, "Fixed (2,4)": 60.0, "Fixed (2,3)": 60.0}
v5 = {}
for k, v in FIG5_LAT.items():
    if k.startswith("ELASTIC") or k.startswith("Fixed"):
        lo = 100 * (1 - (v + 0.7) / (FIG5_LAT["BoN (10,10)"] - 0.7))
        hi = 100 * (1 - (v - 0.7) / (FIG5_LAT["BoN (10,10)"] + 0.7))
        v5[k] = {"latency_ms": v, "reduction_vs_BoN_pct": round(100 * (1 - v / FIG5_LAT["BoN (10,10)"]), 1),
                 "range_pm0.7ms": [round(lo, 1), round(hi, 1)],
                 "reduction_vs_base_pct": round(100 * (1 - v / FIG5_LAT["pi0.5 (10,1)"]), 1)}
R["V5_hw_latency_reduction"] = v5

# ---------------------------------------------------------------- V6 그림 14 지연 모델과 그림 5, 11 비교
F14 = {  # P=1..10, 그림 14에서 읽음(ms, ±0.7)
    1: [63.0, 64.2, 65.3, 65.5, 65.4, 66.3, 66.8, 67.1, 67.0, 67.7],
    5: [73.4, 74.6, 76.4, 77.9, 79.4, 80.1, 80.6, 81.0, 82.6, 83.7],
    10: [84.8, 86.9, 90.2, 92.0, 94.4, 96.6, 97.9, 99.0, 101.0, 102.8],
}


def f14_pred(L, P):
    Ls = sorted(F14)
    Ps = np.arange(1, 11)
    vals = [np.interp(P, Ps, F14[l]) for l in Ls]
    return float(np.interp(L, Ls, vals))


per_step = (F14[10][0] - F14[1][0]) / 9
per_sample_L1 = (F14[1][9] - F14[1][0]) / 9
per_sample_L10 = (F14[10][9] - F14[10][0]) / 9
v6 = {"ms_per_step_single_sample": round(per_step, 2), "ms_per_sample_at_one_step": round(per_sample_L1, 2),
      "ms_per_sample_at_ten_steps": round(per_sample_L10, 2),
      "ratio_step_to_sample_one_step": round(per_step / per_sample_L1, 1),
      "ratio_step_to_sample_ten_steps": round(per_step / per_sample_L10, 1),
      "alpha_over_beta_paper": round(0.13 / 0.1, 2)}
cmp6 = {}
for k, (L, P, obs) in {"pi0.5 (10,1)": (10, 1, 69.1), "BoN (10,10)": (10, 10, 95.5), "ELASTIC (2.6,5.6)": (2.6, 5.6, 62.5),
                       "ELASTIC (3.6,2.9)": (3.6, 2.9, 64.6), "ELASTIC (1.9,3.5)": (1.9, 3.5, 58.0)}.items():
    pr = f14_pred(L, P)
    cmp6[k] = {"fig5_ms": obs, "fig14_interp_ms": round(pr, 1), "gap_ms": round(obs - pr, 1)}
v6["fig5_vs_fig14"] = cmp6
v6["BoN_reduction_if_fig14_model"] = {k: round(100 * (1 - f14_pred(*pt) / f14_pred(10, 10)), 1)
                                      for k, pt in {"ELASTIC (2.6,5.6)": (2.6, 5.6), "ELASTIC (3.6,2.9)": (3.6, 2.9),
                                                    "ELASTIC (1.9,3.5)": (1.9, 3.5)}.items()}
v6["fig11_base_ms_4090"] = 68.7
v6["fig11_BoN_ms_4090"] = 94.0
R["V6_latency_model"] = v6

# ---------------------------------------------------------------- V7 확산 정책 (그림 2, 13): ELASTIC 대 Fixed
# (task, 구분, L, Fixed %, ELASTIC %, 비고)  Fixed가 ELASTIC에 가려진 3곳은 가려진 값을 가정(비고에 표시)
DP_PAIRS = [
    ("PushT", 3.5, 75.5, 80.2, "Fixed 한 점이 두 ELASTIC 점(L=3.5)과 공유된다고 가정"),
    ("PushT", 3.5, 75.5, 82.1, "위와 같음"),
    ("PushT", 6.0, 88.4, 90.3, ""),
    ("PushT Obstacle", 4.8, 85.3, 85.3, "Fixed가 ELASTIC 아래에 가려짐: 같은 값으로 가정"),
    ("PushT Obstacle", 6.3, 89.1, 91.6, ""),
    ("PushT Obstacle", 8.0, 87.4, 90.3, ""),
    ("Square PH", 5.3, 93.9, 92.8, "Fixed가 일부만 보임: 상단 가장자리로 추정"),
    ("Square PH", 5.7, 89.5, 93.5, ""),
    ("Square PH", 7.6, 88.5, 95.0, ""),
    ("Square MH", 6.0, 74.0, 81.0, ""),
    ("Square MH", 6.8, 76.0, 81.6, ""),
    ("Square MH", 7.9, 79.5, 85.5, ""),
    ("Can Paired", 2.7, 45.5, 74.5, ""),
    ("Can Paired", 2.8, 70.8, 78.9, ""),
    ("Can Paired", 4.1, 50.5, 76.4, ""),
    ("Can Reverse", 2.8, 33.0, 43.4, ""),
    ("Can Reverse", 2.9, 56.5, 61.4, ""),
    ("Can Reverse", 3.7, 33.5, 60.9, ""),
]
rows = []
for task, L, f, e, note in DP_PAIRS:
    kf, ke = round(f * 2), round(e * 2)  # 200회 -> 0.5 %p 단위
    rows.append({"task": task, "L": L, "fixed": f, "elastic": e, "gain_pp": round(e - f, 1),
                 "fisher_p": round(fisher(ke, 200, kf, 200), 4), "note": note})
R["V7_dp_elastic_vs_fixed"] = rows
gains = np.array([r["gain_pp"] for r in rows])
R["V7_summary"] = {
    "n_pairs": len(rows), "n_elastic_higher": int((gains > 0).sum()), "n_equal": int((gains == 0).sum()),
    "n_lower": int((gains < 0).sum()), "n_p_lt_0.05": int(sum(r["fisher_p"] < 0.05 for r in rows)),
    "n_p_lt_0.05_excluding_assumed": int(sum(r["fisher_p"] < 0.05 for r in rows if not r["note"])),
    "n_pairs_without_assumption": int(sum(1 for r in rows if not r["note"])),
    "mean_gain_pp": round(float(gains.mean()), 1), "median_gain_pp": round(float(np.median(gains)), 1),
    "per_task_mean_gain": {t: round(float(np.mean([r["gain_pp"] for r in rows if r["task"] == t])), 1)
                           for t in dict.fromkeys(r["task"] for r in rows)},
}

# ---------------------------------------------------------------- V8 ELASTIC 최고점 대 BoN, 학습된 Fixed (그림 2, 13)
BEST = {  # task: (BoN, BoN L, BoN N, DP, ELASTIC 최고, 그 점의 L, 학습된 Fixed 최고, 그 점의 L)
    "PushT": (91.9, 10, 4, 90.8, 90.3, 6.0, 87.6, 5.9),
    "PushT Obstacle": (91.3, 10, 4, 86.2, 91.6, 6.3, 90.1, 10.5),
    "Square PH": (94.0, 10, 4, 91.0, 95.0, 7.6, 92.3, 10.0),
    "Square MH": (86.0, 10, 4, 81.0, 85.5, 7.9, 81.6, 14.2),
    "Can Paired": (77.9, 10, 4, 47.4, 79.0, 2.8, 65.1, 3.1),
    "Can Reverse": (67.9, 10, 8, 30.0, 61.5, 2.9, 61.1, 4.9),
}
v8 = {}
for t, (bon, bl, bn, dp, e, el, lf, lfl) in BEST.items():
    kb, kd, ke, kl = round(bon * 2), round(dp * 2), round(e * 2), round(lf * 2)
    v8[t] = {"BoN": bon, "DP": dp, "ELASTIC_best": e, "ELASTIC_L": el, "LearnedFixed_best": lf, "LearnedFixed_L": lfl,
             "E_minus_BoN_pp": round(e - bon, 1), "fisher_E_vs_BoN": round(fisher(ke, 200, kb, 200), 3),
             "E_minus_DP_pp": round(e - dp, 1), "fisher_E_vs_DP": round(fisher(ke, 200, kd, 200), 4),
             "E_minus_LearnedFixed_pp": round(e - lf, 1), "fisher_E_vs_LF": round(fisher(ke, 200, kl, 200), 3)}
R["V8_dp_vs_bon_dp_learnedfixed"] = v8

# ---------------------------------------------------------------- V9 정수 가능성 (200회 -> 0.5 %p 배수)
ALL_DP = [r["fixed"] for r in rows] + [r["elastic"] for r in rows] + [b[0] for b in BEST.values()] + [b[3] for b in BEST.values()]
dev = [abs(v * 2 - round(v * 2)) / 2 for v in ALL_DP]
R["V9_integer_feasibility"] = {"n_values": len(ALL_DP), "max_dev_from_0.5pp_grid": round(max(dev), 2),
                               "mean_dev": round(float(np.mean(dev)), 2),
                               "note": "모든 읽은 값이 0.5 %p 격자(200회 이항)에서 0.25 %p 이내. 읽기 오차보다 작아 이항이 아님을 시사하는 값은 없음"}

# ---------------------------------------------------------------- V10 검정력: 200회에서 감지 가능한 최소 차이
mde = {}
for p0 in (0.50, 0.75, 0.90):
    for dlt in np.arange(0.01, 0.30, 0.005):
        es = proportion_effectsize(min(p0 + dlt, 0.999), p0)
        if NormalIndPower().power(es, nobs1=200, alpha=0.05) >= 0.8:
            mde[str(p0)] = round(100 * float(dlt), 1)
            break
R["V10_power_n200"] = {"min_detectable_diff_pp_at_power80_by_baseline": mde}

# ---------------------------------------------------------------- V11 Sequential 기준선 L의 상한 (App. 7.2.1 식)
# Sequential: L = (alpha*Lbar + beta*(Pbar-1))/alpha = Lbar + (beta/alpha)*(Pbar-1), Pbar <= N
ELASTIC_L = {"PushT": [3.5, 3.5, 6.0], "PushT Obstacle": [4.8, 6.3, 8.0], "Square PH": [5.3, 5.7, 7.6],
             "Square MH": [6.0, 6.9, 7.9], "Can Paired": [2.7, 2.8, 4.1], "Can Reverse": [2.8, 2.9, 3.7]}
SEQ_L_VISIBLE_MAX = {"PushT": 9.7, "PushT Obstacle": 16.7, "Square PH": 15.3, "Square MH": 17.6,
                     "Can Paired": 17.1, "Can Reverse": 16.8}  # 읽은 가장 큰 Sequential L (가장 작은 alpha 체제)
N_SAMPLES = {"PushT": 4, "PushT Obstacle": 4, "Square PH": 4, "Square MH": 4, "Can Paired": 4, "Can Reverse": 8}
v11 = {}
for t, ls in ELASTIC_L.items():
    bound = max(ls) + (0.1 / 0.03) * (N_SAMPLES[t] - 1)
    v11[t] = {"seq_L_max_read": SEQ_L_VISIBLE_MAX[t], "upper_bound_from_formula": round(bound, 1),
              "exceeds_bound": SEQ_L_VISIBLE_MAX[t] > bound + 0.3}
R["V11_sequential_L_bound"] = v11

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(R, ensure_ascii=False, indent=1))
