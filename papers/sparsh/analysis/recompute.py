"""Sparsh (arXiv:2410.24090) 보고 수치 재계산과 검정.

입력은 논문 본문·부록 표의 보고값뿐이며 외부 데이터·학습을 쓰지 않는다.
실행: python3 analysis/recompute.py  (수 초, numpy·scipy)
결과: analysis/recompute_results.json
"""
import json, math, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}

# ---- 1. 표 8, 9, 10, 14: 보고된 95% CI 폭과 이항 표준오차 비교 ----
def binom_halfwidth(p, n):
    return 1.96 * math.sqrt(p * (1 - p) / n)

ci_checks = {
    "T3_pose_E2E_full":    dict(p=0.812, n=630,  ci=(0.811, 0.813)),
    "T3_pose_DINO_full":   dict(p=0.913, n=630,  ci=(0.912, 0.914)),
    "T3_pose_DINOv2_full": dict(p=0.665, n=630,  ci=(0.658, 0.673)),
    "T4_grasp_IJEPA_full": dict(p=0.802, n=1300, ci=(0.800, 0.804)),
    "T4_grasp_E2E_full":   dict(p=0.784, n=1300, ci=(0.783, 0.785)),
}
out["ci_vs_binomial"] = {}
for k, v in ci_checks.items():
    rep_half = (v["ci"][1] - v["ci"][0]) / 2
    out["ci_vs_binomial"][k] = dict(
        reported_halfwidth=round(rep_half, 4),
        binomial_halfwidth=round(binom_halfwidth(v["p"], v["n"]), 4),
        n_test=v["n"],
    )

# ---- 2. 표 5, 6: 힘 RMSE CI 폭으로 본 표준편차 역산 (n = 25k) ----
n_force = 25000
force_ci = {
    "T5_DIGIT_E2E_full": (39.34, 39.21, 39.48),
    "T5_DIGIT_DINO_full": (36.09, 36.01, 36.17),
    "T6_GS_IJEPA_1over3": (35.20, 24.57, 35.82),   # 보고된 CI가 비대칭
    "T6_GS_IJEPA_full":  (27.91, 27.37, 28.44),
    "T6_GS_DINO_full":   (20.25, 19.85, 20.65),
}
out["force_ci_implied_sd"] = {}
for k, (m, lo, hi) in force_ci.items():
    half = (hi - lo) / 2
    sd = half * math.sqrt(n_force) / 1.96
    center_offset = round(((lo + hi) / 2) - m, 3)
    out["force_ci_implied_sd"][k] = dict(mean=m, lo=lo, hi=hi,
                                         implied_sd=round(sd, 2),
                                         ci_center_minus_mean=center_offset,
                                         asymmetric=bool(abs(center_offset) > 0.05))

# ---- 3. 표 11: 실로봇 bead maze 거리 (n = 10, ± 해석 두 가지) ----
rows = {
    "DINO_pre":  (10.80, 3.68), "IJEPA_pre": (9.4, 3.1), "MAE_pre": (10.2, 4.9),
    "DINO_ft":   (8.45, 3.21), "IJEPA_ft":  (10.02, 5.37), "MAE_ft": (11.25, 3.85),
    "E2E":       (6.70, 1.67),
}
e2e_m, e2e_s = rows["E2E"]
out["real_robot_table11"] = {}
for k, (m, s) in rows.items():
    if k == "E2E":
        continue
    # 해석 A: ± 는 표준편차
    t_sd, p_sd = stats.ttest_ind_from_stats(m, s, 10, e2e_m, e2e_s, 10, equal_var=False)
    # 해석 B: ± 는 95% CI 반폭 (n=10, t_{0.975,9}=2.262)
    s_b, s_e = s / 2.262 * math.sqrt(10), e2e_s / 2.262 * math.sqrt(10)
    t_b, p_b = stats.ttest_ind_from_stats(m, s_b, 10, e2e_m, s_e, 10, equal_var=False)
    out["real_robot_table11"][k] = dict(
        ratio_vs_E2E_pct=round((m / e2e_m - 1) * 100, 1),
        welch_p_if_pm_is_sd=round(float(p_sd), 4),
        welch_p_if_pm_is_ci=round(float(p_b), 4),
    )

# ---- 4. 표 14: 10개 텍스타일 DIGIT n-shot, 테스트 110 샘플 ----
n = 110
t14 = {"zero": (9.1, 3.6), "1shot": (19.1, 0.0), "5shot": (28.2, 15.5), "10shot": (61.8, 10.9)}
out["table14_fisher"] = {}
for k, (a, b) in t14.items():
    ka, kb = round(a / 100 * n), round(b / 100 * n)
    tab = [[ka, n - ka], [kb, n - kb]]
    _, p = stats.fisher_exact(tab)
    out["table14_fisher"][k] = dict(sparsh_correct=ka, e2e_correct=kb, fisher_p=round(float(p), 5))

# ---- 5. 표 13 평균 재계산 ----
vals = {"force_DIGIT": 28.31, "force_GelSight": 59.74, "slip": 242.70,
        "pose": 235.89, "grasp": 5.14, "bead_maze": 19.72}
arr = np.array(list(vals.values()))
out["table13_best_vs_e2e"] = dict(
    mean=round(float(arr.mean()), 2),
    median=round(float(np.median(arr)), 2),
    share_of_top2_in_sum=round(float((242.70 + 235.89) / arr.sum() * 100), 1),
    reported_mean=98.75,
)
dino_vs_ijepa = [26.67, 32.41, 29.08, -37.91, 8.45, -5.26]
out["table13_dino_vs_ijepa_mean"] = round(float(np.mean(dino_vs_ijepa)), 2)

# ---- 6. 표 7: 슬립 F1, 전체 데이터에서 DINO가 E2E보다 낮은지 ----
out["slip_full_data"] = dict(E2E=0.767, DINO=0.685, VJEPA=0.820,
                             dino_minus_e2e=round(0.685 - 0.767, 3))

# ---- 7. 표 12: bead maze 위치 오차 (1/2 열 기준, 본문 "약 16%") ----
e2e_half = 7.14
out["table12_half_budget_reduction_pct"] = {
    "DINO": round((e2e_half - 5.98) / e2e_half * 100, 1),
    "IJEPA": round((e2e_half - 5.72) / e2e_half * 100, 1),
}

with open(os.path.join(HERE, "recompute_results.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=2))
