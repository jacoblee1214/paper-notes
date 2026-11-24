"""WHIRL (arXiv:2609.06009 v1) 표 1, 부록 D 통계, 그림 5·6 판독 값의 재계산과 일관성 검사.

실행: python3 analysis/recompute.py   → analysis/recompute_results.json
의존: numpy, scipy, statsmodels. 그림 판독 값은 analysis/fig_digitized.json (digitize_fig.py가 만든다).
성공 횟수와 시행 수는 논문 표 1에서 옮겼다. 검사 이름은 V1, V2, ... 이다.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[1]
OUT = {}

# 표 1: 과제 -> (시행 수, {방법: 성공 수})
T1 = {
    "Cube":         (30, {"BC": 20, "ResFiT": 25, "Residual": 25, "WHIRL": 30}),
    "LEGO":         (30, {"BC": 19, "ResFiT": 24, "Residual": 23, "WHIRL": 29}),
    "P&P":          (30, {"BC": 13, "ResFiT": 19, "Residual": 18, "WHIRL": 27}),
    "Drawer":       (20, {"BC": 11, "ResFiT": 14, "Residual": 15, "WHIRL": 19}),
    "Long Horizon": (20, {"BC": 7,  "ResFiT": 13, "Residual": 12, "WHIRL": 17}),
}
T1_PCT = {  # 표에 인쇄된 백분율
    "Cube": [66.7, 83.3, 83.3, 100.0], "LEGO": [63.3, 80.0, 76.7, 96.7], "P&P": [43.3, 63.3, 60.0, 90.0],
    "Drawer": [55.0, 70.0, 75.0, 95.0], "Long Horizon": [35.0, 65.0, 60.0, 85.0]}
APP_CI = {"Cube": (89, 100), "LEGO": (83, 99), "P&P": (74, 97), "Drawer": (76, 99), "Long Horizon": (64, 95)}
APP_P = {"Cube": 0.026, "LEGO": 0.026, "P&P": 0.008, "Drawer": 0.091, "Long Horizon": 0.078}
METHODS = ["BC", "ResFiT", "Residual", "WHIRL"]


def fisher(a, na, b, nb, alt="greater"):
    return stats.fisher_exact([[a, na - a], [b, nb - b]], alternative=alt)[1]


# V1: 표 1 백분율이 성공 수/시행 수와 맞는가
v1 = {}
for t, (n, d) in T1.items():
    v1[t] = [round(100 * d[m] / n, 1) for m in METHODS]
    assert v1[t] == T1_PCT[t], (t, v1[t])
OUT["V1_table1_pct_match"] = {"all_match": True, "recomputed": v1}

# V2: WHIRL Wilson 95% 구간과 부록 D 표 비교
v2 = {}
for t, (n, d) in T1.items():
    lo, hi = proportion_confint(d["WHIRL"], n, alpha=0.05, method="wilson")
    v2[t] = {"wilson": [round(100 * lo, 1), round(100 * hi, 1)], "paper": list(APP_CI[t]),
             "match_rounded": [round(100 * lo) == APP_CI[t][0], round(100 * hi) == APP_CI[t][1]]}
OUT["V2_wilson"] = v2

# V3: Fisher 단측(논문 방식), 양측, 최강 기준선, ResFiT, BC 대비
v3 = {}
for t, (n, d) in T1.items():
    best = max(["BC", "ResFiT", "Residual"], key=lambda m: d[m])
    rec = {"p_one_vs_Residual": fisher(d["WHIRL"], n, d["Residual"], n),
           "paper_p": APP_P[t],
           "p_two_vs_Residual": fisher(d["WHIRL"], n, d["Residual"], n, "two-sided"),
           "p_one_vs_ResFiT": fisher(d["WHIRL"], n, d["ResFiT"], n),
           "p_two_vs_ResFiT": fisher(d["WHIRL"], n, d["ResFiT"], n, "two-sided"),
           "best_baseline": best, "p_one_vs_best": fisher(d["WHIRL"], n, d[best], n),
           "diff_pp_vs_Residual": round(100 * (d["WHIRL"] - d["Residual"]) / n, 1),
           "diff_pp_vs_best": round(100 * (d["WHIRL"] - d[best]) / n, 1),
           "p_one_ResFiT_vs_Residual": fisher(d["ResFiT"], n, d["Residual"], n),
           "p_one_Residual_vs_BC": fisher(d["Residual"], n, d["BC"], n),
           "p_one_ResFiT_vs_BC": fisher(d["ResFiT"], n, d["BC"], n)}
    v3[t] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}
OUT["V3_fisher"] = v3

# V4: 다중 비교 보정(Holm, 5과제 단측 p), 과제를 층으로 묶은 CMH와 합산
ps = [fisher(T1[t][1]["WHIRL"], T1[t][0], T1[t][1]["Residual"], T1[t][0]) for t in T1]
holm = multipletests(ps, method="holm")[1]
ps_best = [v3[t]["p_one_vs_best"] for t in T1]
holm_best = multipletests(ps_best, method="holm")[1]
tabs = []
for t, (n, d) in T1.items():
    tabs.append(np.array([[d["WHIRL"], n - d["WHIRL"]], [d["Residual"], n - d["Residual"]]]))
st = StratifiedTable(tabs)
tot_n = sum(n for n, _ in T1.values())
pool = {m: sum(d[m] for _, d in T1.values()) for m in METHODS}
OUT["V4_multiplicity"] = {
    "holm_vs_Residual": [round(x, 4) for x in holm], "holm_vs_best_baseline": [round(x, 4) for x in holm_best],
    "tasks_p_lt_0.05_holm_vs_Residual": int(sum(holm < 0.05)),
    "cmh_common_OR": round(float(st.oddsratio_pooled), 2),
    "cmh_p_two_sided": float(st.test_null_odds().pvalue),
    "pooled_counts": pool, "pooled_n": tot_n,
    "pooled_rate_pct": {m: round(100 * pool[m] / tot_n, 1) for m in METHODS},
    "pooled_fisher_one_WHIRL_vs_Residual": fisher(pool["WHIRL"], tot_n, pool["Residual"], tot_n),
    "note": "합산과 CMH는 과제 내 시행을 독립으로 보는 가정. 한 시드, 한 체크포인트라서 방법 수준 추론이 아니라 그 체크포인트의 추정이다.",
}

# V5: 본문 문장 검사
diffs_res = [v3[t]["diff_pp_vs_Residual"] for t in T1]
diffs_best = [v3[t]["diff_pp_vs_best"] for t in T1]
res_vs_fit_gap = {t: abs(T1[t][1]["ResFiT"] - T1[t][1]["Residual"]) for t in T1}
best_is_residual = {t: v3[t]["best_baseline"] == "Residual" or
                    T1[t][1]["Residual"] == max(T1[t][1][m] for m in ["BC", "ResFiT", "Residual"]) for t in T1}
OUT["V5_text_claims"] = {
    "gain_vs_Residual_pp": diffs_res, "gain_vs_best_baseline_pp": diffs_best,
    "claim_15_to_30_vs_Residual_ok": min(diffs_res) >= 15 and max(diffs_res) <= 30,
    "ResFiT_vs_Residual_trial_gap": res_vs_fit_gap, "claim_at_most_one_trial_ok": max(res_vs_fit_gap.values()) <= 1,
    "Residual_is_strongest_baseline_by_task": best_is_residual,
    "text_p_0.08_vs_appendix": {"Drawer": [0.08, v3["Drawer"]["p_one_vs_Residual"]],
                                "Long Horizon": [0.08, v3["Long Horizon"]["p_one_vs_Residual"]]},
    "paper_p_vs_recomputed_match_3dp": {t: abs(APP_P[t] - v3[t]["p_one_vs_Residual"]) < 0.0006 for t in T1},
}

# V6: 최소 검출 효과와 필요한 시행 수 (단측 0.05, 검정력 80%, 정규 근사 + Fisher 정확 모의)
def power_fisher(p1, p2, n, sims=4000, seed=0):
    rng = np.random.default_rng(seed)
    table = np.ones((n + 1, n + 1))
    for a in range(n + 1):
        for b in range(n + 1):
            table[a, b] = fisher(a, n, b, n)
    a = rng.binomial(n, p1, sims)
    b = rng.binomial(n, p2, sims)
    return float((table[a, b] < 0.05).mean())


def n_needed(p1, p2, target=0.8):
    for n in range(10, 400):
        z = stats.norm
        pbar = (p1 + p2) / 2
        num = (z.ppf(0.95) * np.sqrt(2 * pbar * (1 - pbar)) + z.ppf(target) * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        if n >= num / (p1 - p2) ** 2:
            return n
    return None


v6 = {}
for t, (n, d) in T1.items():
    p1, p2 = d["WHIRL"] / n, d["Residual"] / n
    v6[t] = {"n": n, "obs_diff": round(p1 - p2, 3), "power_at_observed_rates": round(power_fisher(p1, p2, n), 2),
             "n_per_arm_for_80pct_power_normal_approx": n_needed(min(p1, 0.99), p2)}
OUT["V6_power"] = v6

# V7: 그림 5·6 판독 값 (digitize_fig.py 결과)
dig = json.loads((ROOT / "analysis" / "fig_digitized.json").read_text())
v7 = {}
for t, r in dig["fig5"].items():
    w = np.array(r["WHIRL"], float)
    b = np.array(r["wo_WM"], float)
    h_common = min(w[-1, 0], b[-1, 0])
    grid = np.linspace(max(w[0, 0], b[0, 0]), h_common, 200)
    wi = np.interp(grid, w[:, 0], w[:, 1])
    bi = np.interp(grid, b[:, 0], b[:, 1])
    frac_above = float((wi > bi + 0.005).mean())

    def area(c, hmax):
        g = np.linspace(0, hmax, 2000)
        v = np.interp(g, c[:, 0], c[:, 1], left=c[0, 1], right=0.0)  # 끝난 뒤는 0, 시작 앞은 첫 값
        return float(np.trapezoid(v, g)) if hasattr(np, "trapezoid") else float(np.trapz(v, g))

    def first_below(c, thr=0.01):
        i = np.where(c[:, 1] <= thr)[0]
        return int(c[i[0], 0]) if len(i) else None

    hm = float(max(w[-1, 0], b[-1, 0]))
    aw, ab = area(w, hm), area(b, hm)
    aw_c, ab_c = area(w, h_common), area(b, h_common)
    v7[t] = {"last_step_WHIRL": int(w[-1, 0]), "last_step_noWM": int(b[-1, 0]),
             "final_WHIRL": round(float(w[-1, 1]), 3), "final_noWM": round(float(b[-1, 1]), 3),
             "n_points": [len(w), len(b)],
             "first_step_le_0.01": [first_below(w), first_below(b)],
             "share_of_common_range_WHIRL_above_noWM": round(frac_above, 2),
             "operator_step_proxy_to_common_horizon_ratio": round(aw_c / ab_c, 3),
             "operator_step_proxy_to_longest_trace_ratio": round(aw / ab, 3),
             "reduction_pct_common": round(100 * (1 - aw_c / ab_c), 1),
             "reduction_pct_longest": round(100 * (1 - aw / ab), 1)}
OUT["V7_fig5_figure_read"] = v7

# V8: 본문 "0.113에서 0.018, 84%" 검사
pd_ = dig["fig5"]["Pull Drawer"]
OUT["V8_drawer_84pct"] = {"arith_1_minus_0.018_over_0.113_pct": round(100 * (1 - 0.018 / 0.113), 2),
                          "fig5_final_WHIRL": pd_["WHIRL"][-1], "fig5_final_noWM": pd_["wo_WM"][-1],
                          "fig5_values_near_0.113_noWM": [p for p in pd_["wo_WM"] if 0.10 <= p[1] <= 0.13],
                          "fig5_values_near_0.018_WHIRL": [p for p in pd_["WHIRL"] if 0.01 <= p[1] <= 0.03],
                          "fig6_final_Full": dig["fig6"]["Pull Drawer"]["Full"][-1],
                          "fig6_final_noWM": dig["fig6"]["Pull Drawer"]["wo_WM"][-1]}

# V9: 그림 5와 그림 6의 같은 곡선(WHIRL 대 Full, w/o WM)이 같은가
def interp(c, x):
    c = np.array(c, float)
    return np.interp(x, c[:, 0], c[:, 1])


v9 = {}
for t in ["Pick LEGO", "Pull Drawer"]:
    for a, b in (("WHIRL", "Full"), ("wo_WM", "wo_WM")):
        c5 = np.array(dig["fig5"][t][a], float)
        c6 = np.array(dig["fig6"][t][b], float)
        lo, hi = max(c5[0, 0], c6[0, 0]), min(c5[-1, 0], c6[-1, 0])
        g = np.linspace(lo, hi, 100)
        dd = np.abs(interp(c5, g) - interp(c6, g))
        v9[f"{t}:{a}-vs-{b}"] = {"mean_abs_diff": round(float(dd.mean()), 3), "max_abs_diff": round(float(dd.max()), 3),
                                 "last_step_fig5": int(c5[-1, 0]), "last_step_fig6": int(c6[-1, 0]),
                                 "n_points": [len(c5), len(c6)]}
OUT["V9_fig5_vs_fig6_same_runs"] = v9

# V10: 그림 6 순서 (Full 대 alpha_i=0 대 w/o WM) 0 도달 스텝
v10 = {}
for t, r in dig["fig6"].items():
    def fb(c, thr=0.01):
        i = [p for p in c if p[1] <= thr]
        return int(i[0][0]) if i else None
    v10[t] = {k: {"first_step_le_0.01": fb(v), "last_step": int(v[-1][0]), "last_value": round(float(v[-1][1]), 3)}
              for k, v in r.items()}
OUT["V10_fig6_order"] = v10

# V11: 인쇄된 수치 정합성: 추정 체크포인트 예산과 트레이스 길이
OUT["V11_budget"] = {"nominal_steps_pick_drawer": 8000, "nominal_steps_long_horizon": 50000,
                     "fig5_last_steps_WHIRL_over_nominal": {t: round(v["last_step_WHIRL"] / (50000 if t == "Long Horizon" else 8000), 2)
                                                           for t, v in v7.items()},
                     "fig5_last_steps_noWM_over_nominal": {t: round(v["last_step_noWM"] / (50000 if t == "Long Horizon" else 8000), 2)
                                                          for t, v in v7.items()}}

# V12: 마커 간격과 에피소드 지평(표 3), 20 Hz 가정의 상호작용 시간 (마커 1개 = 에피소드 1개 가정)
HORIZON = {"Pick Cube": 300, "Pick LEGO": 300, "Pick & Place": 600, "Pull Drawer": 450, "Long Horizon": 2100}
v12 = {}
for t, r in dig["fig5"].items():
    for key in ("WHIRL", "wo_WM"):
        xs = np.array([p[0] for p in r[key]], float)
        sp = np.diff(xs)
        v12[f"{t}:{key}"] = {"n_markers": len(xs), "median_spacing_steps": int(np.median(sp)),
                             "spacing_over_horizon": round(float(np.median(sp)) / HORIZON[t], 2),
                             "last_step": int(xs[-1]), "interaction_minutes_at_20Hz": round(float(xs[-1]) / 20 / 60, 1)}
OUT["V12_marker_spacing_and_time"] = v12

# V13, V14: 표 3 검사 (지평 스텝/초 = 제어 주파수, 물체 무작위화 폭 대 작업공간 폭)
T3 = {"Pick Cube": dict(rand=15, ws=((0.32, 0.62), (-0.08, 0.16)), hor=(300, 15)),
      "Pick LEGO": dict(rand=15, ws=((0.31, 0.58), (-0.02, 0.12)), hor=(300, 15)),
      "Pick & Place": dict(rand=15, ws=((0.32, 0.60), (-0.03, 0.31)), hor=(600, 30)),
      "Pull Drawer": dict(rand=5, ws=((0.39, 0.51), (-0.03, 0.18)), hor=(450, 22.5)),
      "Long Horizon": dict(rand=5, ws=((0.35, 0.58), (-0.12, 0.26)), hor=(2100, 105))}
v13 = {}
for t, c in T3.items():
    wx, wy = c["ws"][0][1] - c["ws"][0][0], c["ws"][1][1] - c["ws"][1][0]
    v13[t] = {"horizon_Hz": c["hor"][0] / c["hor"][1], "ws_x_width_cm": round(100 * wx), "ws_y_width_cm": round(100 * wy),
              "rand_full_width_cm_if_plus_minus": 2 * c["rand"],
              "full_width_exceeds_ws_y": 2 * c["rand"] > 100 * wy, "full_width_exceeds_ws_x": 2 * c["rand"] > 100 * wx}
OUT["V13_V14_table3"] = v13

Path(ROOT / "analysis" / "recompute_results.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=float))
print(json.dumps(OUT, ensure_ascii=False, indent=1, default=float))
