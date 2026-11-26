"""표 1, 표 S1, 그림 6e, 본문 수치 재계산 (논문에 적힌 값만 사용, 수 초). 실행: python3 analysis/recompute.py
시행 수가 논문에 없는 곳은 가정을 키 이름에 적었다."""
import json, math
import numpy as np
from scipy import stats

def wilson(k, n, z=1.96):
    p = k/n; d = 1+z*z/n
    c = (p+z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [round(c-h, 4), round(c+h, 4)]

# 표 1: (bias, MAE, WAPE, excess, deficit, success) 점추정과 구간
T1 = {
 "naive":        dict(bias=1.97, mae=2.18, wape=2.11, exc=2.07, dfc=0.10, suc=1.00, bias_ci=[1.67,2.22], mae_ci=[2.02,2.33], wape_ci=[1.56,2.95], suc_ci=[0.99,1.00]),
 "class_prior":  dict(bias=-0.26, mae=0.70, wape=0.68, exc=0.22, dfc=0.48, suc=0.89, bias_ci=[-0.57,0.10], mae_ci=[0.50,0.95], wape_ci=[0.59,0.77], suc_ci=[0.77,1.00]),
 "direct":       dict(bias=0.29, mae=0.96, wape=0.93, exc=0.62, dfc=0.33, suc=0.96, bias_ci=[-0.05,0.60], mae_ci=[0.75,1.19], wape_ci=[0.74,1.21], suc_ci=[0.90,1.00]),
 "slip_shear":   dict(bias=0.77, mae=1.12, wape=1.09, exc=0.95, dfc=0.18, suc=0.96, bias_ci=[0.52,0.98], mae_ci=[0.98,1.29], wape_ci=[0.86,1.43], suc_ci=[0.91,1.00]),
 "slip_area":    dict(bias=-0.52, mae=0.71, wape=0.69, exc=0.10, dfc=0.61, suc=0.93, bias_ci=[-0.82,-0.26], mae_ci=[0.48,0.99], wape_ci=[0.60,0.75], suc_ci=[0.87,1.00]),
 "slip_gbm":     dict(bias=-0.51, mae=0.72, wape=0.70, exc=0.10, dfc=0.62, suc=0.93, bias_ci=[-0.79,-0.24], mae_ci=[0.49,1.00], wape_ci=[0.62,0.76], suc_ci=[0.86,0.99]),
 "slip_images":  dict(bias=0.16, mae=0.88, wape=0.85, exc=0.47, dfc=0.41, suc=0.97, bias_ci=[-0.29,0.51], mae_ci=[0.69,1.11], wape_ci=[0.73,1.01], suc_ci=[0.93,1.00]),
 "ours":         dict(bias=-0.05, mae=0.51, wape=0.52, exc=0.21, dfc=0.33, suc=0.95, bias_ci=[-0.44,0.24], mae_ci=[0.19,0.84], wape_ci=[0.43,0.60], suc_ci=[0.90,0.99]),
 "no_prior":     dict(bias=-0.14, mae=0.61, wape=0.59, exc=0.23, dfc=0.38, suc=0.94, bias_ci=[-0.42,0.07], mae_ci=[0.43,0.83], wape_ci=[0.49,0.68], suc_ci=[0.84,1.00]),
 "wrong_prior":  dict(bias=-0.20, mae=0.62, wape=0.59, exc=0.21, dfc=0.40, suc=0.93, bias_ci=[-0.46,0.03], mae_ci=[0.47,0.86], wape_ci=[0.50,0.67], suc_ci=[0.82,0.97]),
 "coulomb_only": dict(bias=-0.16, mae=0.62, wape=0.61, exc=0.24, dfc=0.38, suc=0.94, bias_ci=[-0.41,0.08], mae_ci=[0.45,0.84], wape_ci=[0.52,0.69], suc_ci=[0.88,0.98]),
 "ours_margin":  dict(bias=0.22, mae=1.00, wape=0.97, exc=0.61, dfc=0.39, suc=0.99, bias_ci=[-0.14,0.59], mae_ci=[0.77,1.26], wape_ci=[0.78,1.24], suc_ci=[0.96,1.00]),
}
R = {}
rows = {}
for k, v in T1.items():
    r = {}
    r["bias_minus(exc-dfc)"] = round(v["bias"]-(v["exc"]-v["dfc"]), 2)
    r["mae_minus(exc+dfc)"] = round(v["mae"]-(v["exc"]+v["dfc"]), 2)
    r["implied_mean_opt_mae/wape"] = round(v["mae"]/v["wape"], 3)
    # 성공률 구간 폭에서 유효 독립 단위 수 (Wilson 반폭 기준 근사)
    lo, hi = v["suc_ci"]; h = (hi-lo)/2; p = min(v["suc"], 0.995)
    r["n_eff_from_success_ci"] = round(1.96**2*p*(1-p)/h**2) if h > 0 else None
    r["mae_ci_halfwidth_over_wape_ci_halfwidth"] = round(((v["mae_ci"][1]-v["mae_ci"][0])/(v["wape_ci"][1]-v["wape_ci"][0])), 2)
    rows[k] = r
R["table1_checks"] = rows
R["table1_naive_mean_opt_from_bias"] = round(3-1.97, 3)
R["table1_naive_overload_S1_vs_excess"] = {"overload_S1": 2.01, "excess_T1": 2.07, "diff": 0.06}
R["table1_5k_wilson_if_independent_success0.95"] = wilson(4750, 5000)
R["table1_ours_vs_class_prior_excess_ratio"] = round(0.22/0.21, 2)
R["table1_ours_vs_naive_excess_ratio"] = round(2.07/0.21, 1)
R["table1_mae_ratio_class_prior_over_ours"] = {"mae_0.51": round(0.70/0.51, 2), "mae_0.54(exc+dfc)": round(0.70/0.54, 2)}
R["S4_percent_sum"] = round(83.0+15.4+1.6, 1)

# 그림 6e (실로봇): 29개 물체 x 3회 = 87회 가정
f6 = {}
for name, k87, k29 in (("ours", 84, 28), ("class_prior", 63, 21)):
    f6[name] = {"n87": {"k": k87, "rate": round(k87/87, 4), "wilson": wilson(k87, 87)},
                "n29": {"k": k29, "rate": round(k29/29, 4), "wilson": wilson(k29, 29)}}
for tag, (a, b, n) in {"n87": (84, 63, 87), "n29": (28, 21, 29)}.items():
    o, p = stats.fisher_exact([[a, n-a], [b, n-b]])
    f6[f"fisher_ours_vs_prior_{tag}"] = {"odds": round(o, 2), "p": float(p)}
f6["integer_feasible_0.97_n87"] = [k for k in range(88) if round(k/87, 2) == 0.97]
f6["integer_feasible_0.97_n29"] = [k for k in range(30) if round(k/29, 2) == 0.97]
f6["integer_feasible_0.72_n87"] = [k for k in range(88) if round(k/87, 2) == 0.72]
f6["integer_feasible_0.72_n29"] = [k for k in range(30) if round(k/29, 2) == 0.72]
# MAE 대비 (평균, SD) Welch: 29(물체) 또는 87(시행) 가정, 짝 없음(보수적)
for n in (29, 87):
    for nm, (m1, s1, m2, s2) in {"ours_vs_class_prior": (0.21, 0.14, 0.46, 0.40), "ours_vs_naive": (0.21, 0.14, 1.99, 0.33)}.items():
        t, p = stats.ttest_ind_from_stats(m1, s1, n, m2, s2, n, equal_var=False)
        f6[f"welch_MAE_{nm}_n{n}"] = {"t": round(float(t), 2), "p": float(p)}
f6["mae_ratio_class_prior_over_ours"] = round(0.46/0.21, 2)
f6["naive_mean_opt_from_MAE"] = round(3-1.99, 2)
# MAPE=mean(3/o)-1 -> mean(1/o) -> 조화평균
mean_inv = (2.44+1)/3
f6["naive_harmonic_mean_opt_from_MAPE"] = round(1/mean_inv, 3)
f6["objects_vs_bars"] = {"text_objects": 29, "bars_in_fig6d": 25}
R["fig6e"] = f6

# 부록 S3.2 학습량 산술
steps = 200_000/8*500
R["training_arithmetic"] = {"optimizer_steps_if_all_200k": steps, "steps_per_sec_over_48h": round(steps/(48*3600), 1),
    "steps_if_85pct": 0.85*steps, "note": "배치 8이 GPU당인지 전체인지, 200k 중 학습 비율은 논문에 없음"}
# 시뮬 타이밍
R["sim_timing"] = {"dt_ms": 4, "control_hz_real": 250, "gelsight_hz": 25, "interp_factor": 10,
   "max_servo_s": 100*0.004, "dwell_s": 0.1, "lift_s": 2e-3/10e-3, "fig6c_lift_starts_s_read": 0.32}
# 표 S1: 재료 MAPE 개선
cp = dict(E=1.07, m=1.50, mu=1.21, tau=0.77); ou = dict(E=0.56, m=0.49, mu=0.56, tau=0.44)
R["S1_mape_ratio_prior_over_ours"] = {k: round(cp[k]/ou[k], 2) for k in cp}
R["S1_mape_vs_pearson"] = {"r": dict(E=0.65, m=0.88, mu=0.28, tau=0.57), "mape_ours": ou}
# 그림 6c 판독 (네크타린)
R["fig6c_read"] = {"E_true_kPa": 42, "E_end_kPa": 37, "m_true_g": 192, "m_end_g": 181, "mu_true": 0.59, "mu_end": 0.62,
    "E_err_%": round(100*(37-42)/42, 1), "m_err_%": round(100*(181-192)/192, 1), "mu_err_%": round(100*(0.62-0.59)/0.59, 1),
    "m_prior_g_read": 142, "mu_prior_read": 0.23, "E_prior_read": 83}
# 그림 6d 판독 (25개 막대 중 22개)
raw = json.load(open("analysis/fig6d_readings_raw.json"))
idx = [i for i in range(25) if i not in (16, 17, 23)]
d = np.array([raw[i][0]-raw[i][1] for i in idx]); c = np.array([raw[i][2]-raw[i][1] for i in idx])
R["fig6d_read_22bars"] = {"ours_MAE": round(float(abs(d).mean()), 3), "ours_bias": round(float(d.mean()), 3), "ours_max_over": round(float(d.max()), 2),
   "ours_n_below_opt_by_>0.02": int((d < -0.02).sum()), "class_prior_MAE": round(float(abs(c).mean()), 3),
   "mean_opt": round(float(np.mean([raw[i][1] for i in idx])), 3), "excluded_bars_1based": [17, 18, 24]}
json.dump(R, open("analysis/recompute_results.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))
