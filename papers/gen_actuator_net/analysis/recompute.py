"""논문(arXiv:2604.09487v4)이 보고한 표·그림 수치의 재계산과 일관성 점검 (V1-V12).

입력: 논문 본문에 숫자로 적힌 값 + figure_values.json(그림에서 읽은 값; extract_figure_values.py)
출력: recompute_results.json, 표준출력 요약
실행: python3 recompute.py          (numpy, scipy 필요; CPU 1초 미만)
"""
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
FIG = json.loads((HERE / "figure_values.json").read_text())
N = 100  # 논문 Fig.5, Fig.13: "Results are computed for 100 trials"
R = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return c - h, c + h


def newcombe_diff(k1, n1, k2, n2):
    """두 비율 차(p1-p2)의 Newcombe 95% 구간(Wilson 기반)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


# ---------------------------------------------------------------- 보고된 값
TASKS = ["reacher", "ball_in_cup", "table_tennis"]
CFGS = ["GenAN", "Torque GenAN", "No disag. penalty", "No ensemble", "Low action penalty"]
FIG5 = {  # Fig.5 막대 위 숫자 (= 100회 중 성공 횟수로 해석)
    "reacher": [90, 97, 86, 93, 70],
    "ball_in_cup": [75, 76, 56, 74, 55],
    "table_tennis": [96, 75, 82, 83, 54],
}
TABLE8 = {  # 표 8: 탁구 결과 분류 (성공, 너무 김, 너무 짧음, 놓침)
    "GenAN": [96, 3, 0, 1],
    "Torque GenAN": [75, 0, 13, 12],
    "No disag. penalty": [82, 1, 12, 5],
    "No ensemble": [83, 0, 17, 0],
    "Low action penalty": [54, 16, 8, 22],
}
FIG13 = {"HySR": 88, "GenAN": 89, "Torque GenAN": 74, "No disag. penalty": 66, "No ensemble": 74, "Low action penalty": 44}
FIG11 = {"GenAN": 1.32, "Torque GenAN": 1.01, "No disag. penalty": 1.30, "No ensemble": 1.22, "Low action penalty": 1.94}

# ---------------------------------------------------------------- V1 Fig.5 오차막대 vs Wilson
fig5_fig = FIG["fig5"]
v1 = []
i = 0
for t in TASKS:
    for c, k in zip(CFGS, FIG5[t]):
        lo, hi = wilson(k, N)
        fv = fig5_fig[i]
        v1.append({
            "task": t, "cfg": c, "k": k,
            "fig_value": round(fv["value"], 2),
            "wilson": [round(100 * lo, 2), round(100 * hi, 2)],
            "fig_ci": [round(fv["ci"][0], 2), round(fv["ci"][1], 2)],
            "max_abs_diff_pt": round(max(abs(100 * lo - fv["ci"][0]), abs(100 * hi - fv["ci"][1])), 2),
        })
        i += 1
R["V1_fig5_wilson_vs_plotted"] = v1
R["V1_max_diff_pt"] = max(x["max_abs_diff_pt"] for x in v1)
R["V1_value_max_diff_pt"] = max(abs(x["fig_value"] - x["k"]) for x in v1)

# ---------------------------------------------------------------- V2 Fig.13 오차막대 (HySR은 막대만 있고 오차막대 없음)
v2 = []
for c, fv in zip(["HySR"] + CFGS, FIG["fig13"]):
    k = FIG13[c]
    row = {"cfg": c, "k": k, "fig_value": round(fv["value"], 2)}
    if fv["ci"]:
        lo, hi = wilson(k, N)
        row["wilson"] = [round(100 * lo, 2), round(100 * hi, 2)]
        row["fig_ci"] = [round(fv["ci"][0], 2), round(fv["ci"][1], 2)]
    else:
        row["note"] = "오차막대 없음(Guist et al. 원 논문의 시행 수 불명)"
    v2.append(row)
R["V2_fig13"] = v2

# ---------------------------------------------------------------- V3 표 8 합계, Fig.5/13 정합
v3 = {c: {"row_sum": sum(v), "success_matches_fig5": v[0] == FIG5["table_tennis"][CFGS.index(c)]} for c, v in TABLE8.items()}
R["V3_table8"] = v3
R["V3_fig13_le_fig5"] = {c: (FIG13[c], FIG5["table_tennis"][CFGS.index(c)], FIG13[c] <= FIG5["table_tennis"][CFGS.index(c)]) for c in CFGS}
# 원 반경 68.5 cm 가 반 테이블(137 x 152.5 cm)에서 차지하는 넓이 비율 ("most of the table half")
circle = math.pi * 0.685 ** 2
half = 1.37 * 1.525
R["V3_circle_area_fraction_of_table_half"] = round(circle / half, 3)
R["V3_circle_fits_inside_half"] = (2 * 0.685 <= 1.37 + 1e-9) and (2 * 0.685 <= 1.525)

# ---------------------------------------------------------------- V4 Fisher 정확 검정 (GenAN 대 각 기준선, 과제별) + Holm 보정
rows = []
for t in TASKS:
    base = FIG5[t][0]
    for c, k in zip(CFGS[1:], FIG5[t][1:]):
        table = [[base, N - base], [k, N - k]]
        odds, p = stats.fisher_exact(table, alternative="two-sided")
        d, dl, dh = newcombe_diff(base, N, k, N)
        rows.append({"task": t, "cfg": c, "genan": base, "other": k, "diff_pt": round(100 * d, 1),
                     "diff_ci": [round(100 * dl, 1), round(100 * dh, 1)], "fisher_p": p})
order = np.argsort([r["fisher_p"] for r in rows])
m = len(rows)
run = 0.0
for rank, idx in enumerate(order):
    adj = min(1.0, (m - rank) * rows[idx]["fisher_p"])
    run = max(run, adj)
    rows[idx]["holm_p"] = run
for r in rows:
    r["fisher_p"] = float(f"{r['fisher_p']:.3g}")
    r["holm_p"] = float(f"{r['holm_p']:.3g}")
R["V4_genan_vs_baselines"] = rows

# 과제 내 모든 쌍(GenAN 제외 포함) 중 p<0.05 개수
allpairs = []
for t in TASKS:
    for (a, ka), (b, kb) in combinations(zip(CFGS, FIG5[t]), 2):
        p = stats.fisher_exact([[ka, N - ka], [kb, N - kb]])[1]
        allpairs.append((t, a, b, ka, kb, p))
R["V4_pairs_p_lt_0.05"] = [(t, a, b, ka, kb, float(f"{p:.3g}")) for t, a, b, ka, kb, p in allpairs if p < 0.05]

# ---------------------------------------------------------------- V5 Fig.6b 앙상블 대 단일 모델 (그림 판독 값, 시행 수 n=100 가정)
f6 = FIG["fig6"]
sizes = f6["sizes"]
ens = [round(x) for x in f6["success_ensemble_pct"]]
sgl = [round(x) for x in f6["success_no_ensemble_pct"]]
v5 = []
for s, a, b in zip(sizes, ens, sgl):
    p = stats.fisher_exact([[a, N - a], [b, N - b]])[1]
    d, dl, dh = newcombe_diff(a, N, b, N)
    v5.append({"n_traj": s, "ensemble": a, "single": b, "diff_pt": round(100 * d, 1), "diff_ci": [round(100 * dl, 1), round(100 * dh, 1)], "fisher_p": float(f"{p:.3g}")})
R["V5_fig6b_ensemble_vs_single_n100_assumed"] = v5
# Fig.6b 마지막 점(2500)이 Fig.5 reacher(GenAN 90, No ensemble 93)와 같은 값인지
R["V5_fig6b_2500_vs_fig5"] = {"fig6b": [ens[-1], sgl[-1]], "fig5": [FIG5["reacher"][0], FIG5["reacher"][3]]}
# "1000개로 줄여도 성능 저하 없음": 1000 대 2500
R["V5_1000_vs_2500"] = {
    "ensemble": [ens[4], ens[-1], float(f"{stats.fisher_exact([[ens[4], N-ens[4]], [ens[-1], N-ens[-1]]])[1]:.3g}")],
    "single": [sgl[4], sgl[-1], float(f"{stats.fisher_exact([[sgl[4], N-sgl[4]], [sgl[-1], N-sgl[-1]]])[1]:.3g}")],
}
R["V5_non_monotone_points"] = {"ensemble_1000_1500_2000": ens[4:7], "single_1000_1500_2000": sgl[4:7]}

# ---------------------------------------------------------------- V6 같은 설정의 그림 간 수치 대조 (그림 판독 값)
e1 = {"fig4a": FIG["fig4a_inset"][0]["value"], "fig8a_H3": FIG["fig8a"][2]["value"], "fig9a_s1": FIG["fig9a"][0]["value"],
      "fig10a_R1": FIG["fig10a"][0]["value"], "fig6a_2500": f6["pos_err_1step_deg"][-1], "fig6a_2000": f6["pos_err_1step_deg"][-2]}
e500 = {"fig4b": FIG["fig4b_inset"][0]["value"], "fig8b_H3": FIG["fig8b"][2]["value"], "fig9b_s1": FIG["fig9b"][0]["value"],
        "fig10b_R1": FIG["fig10b"][0]["value"], "fig6a_2500": f6["pos_err_500step_deg"][-1], "fig6a_2000": f6["pos_err_500step_deg"][-2]}
R["V6_genan_1step_deg"] = {k: round(v, 5) for k, v in e1.items()}
R["V6_genan_500step_deg"] = {k: round(v, 3) for k, v in e500.items()}
R["V6_ratio_fig6a_2500_to_fig4a_1step"] = round(e1["fig6a_2500"] / e1["fig4a"], 3)
R["V6_ratio_fig6a_2500_to_fig4b_500step"] = round(e500["fig6a_2500"] / e500["fig4b"], 3)

# ---------------------------------------------------------------- V7 "6 % / 32 % lower" (그림 판독 값)
g1, t1 = FIG["fig4a_inset"][0]["value"], FIG["fig4a_inset"][1]["value"]
g5, t5 = FIG["fig4b_inset"][0]["value"], FIG["fig4b_inset"][1]["value"]
R["V7_reduction_pct"] = {"1step": round(100 * (1 - g1 / t1), 1), "500step": round(100 * (1 - g5 / t5), 1),
                        "paper_text": [6, 32], "v1_v2_text": [6, 29]}
R["V7_fig4_values"] = {"GenAN_1step": round(g1, 5), "Torque_1step": round(t1, 5),
                       "GenAN_500": round(g5, 2), "Torque_500": round(t5, 2),
                       "UAN_1step": round(FIG["fig4a_main"][2]["value"], 4), "SysID_1step": round(FIG["fig4a_main"][3]["value"], 4),
                       "UAN_500": round(FIG["fig4b_main"][2]["value"], 2), "SysID_500": round(FIG["fig4b_main"][3]["value"], 2)}
R["V7_ratios_vs_genan"] = {
    "SysID_1step": round(FIG["fig4a_main"][3]["value"] / g1, 2), "SysID_500": round(FIG["fig4b_main"][3]["value"] / g5, 2),
    "UAN_1step": round(FIG["fig4a_main"][2]["value"] / g1, 2), "UAN_500": round(FIG["fig4b_main"][2]["value"] / g5, 2),
}
# Fig.4 오차막대 겹침: 1 step Torque GenAN 대 GenAN
R["V7_fig4_ci"] = {"GenAN_1step": FIG["fig4a_inset"][0]["ci"], "Torque_1step": FIG["fig4a_inset"][1]["ci"],
                   "GenAN_500": FIG["fig4b_inset"][0]["ci"], "Torque_500": FIG["fig4b_inset"][1]["ci"]}

# ---------------------------------------------------------------- V8 Fig.8-10 본문 서술 점검 (그림 판독 값)
R["V8_history_H_1step"] = [round(x["value"], 5) for x in FIG["fig8a"]]
R["V8_history_H_500"] = [round(x["value"], 2) for x in FIG["fig8b"]]
R["V8_stride_1step"] = [round(x["value"], 5) for x in FIG["fig9a"]]
R["V8_stride_500"] = [round(x["value"], 2) for x in FIG["fig9b"]]
R["V8_rollout_1step"] = [round(x["value"], 5) for x in FIG["fig10a"]]
R["V8_rollout_500"] = [round(x["value"], 2) for x in FIG["fig10b"]]
# 본문: "단일 스텝 지표에서 s=1 이 모든 더 긴 stride 보다 낫다", "다중 스텝 지표에서만 s=2가 s=1보다 약간 낮다"
s1 = [x["value"] for x in FIG["fig9a"]]
s5 = [x["value"] for x in FIG["fig9b"]]
R["V8_claim_stride_1step_s1_best"] = bool(s1[0] == min(s1))
R["V8_claim_stride_500_s2_lower"] = bool(s5[1] < s5[0])
# 본문: "H=20 은 약간의 성능 저하" / "H=2 가 H=1 보다 큰 개선"
h1 = [x["value"] for x in FIG["fig8a"]]
R["V8_H20_worse_than_H10_1step"] = bool(h1[5] > h1[4])
# 다중 스텝 R: 본문 "slight downward trend" — 단조성 점검
r5 = [x["value"] for x in FIG["fig10b"]]
R["V8_rollout_500_monotone_decreasing"] = bool(all(a >= b for a, b in zip(r5, r5[1:])))

# ---------------------------------------------------------------- V9 Fig.11(평균 오차) 대 Fig.5(성공률): 로그정규 가정 가능성
v9 = []
for c, k in zip(CFGS, FIG5["reacher"]):
    mean = FIG11[c]
    s = k / N

    def f(sig, mean=mean, s=s):
        mu = math.log(mean) - sig * sig / 2
        return stats.norm.cdf((math.log(2.0) - mu) / sig) - s

    roots = []
    grid = np.linspace(0.05, 3.0, 600)
    vals = [f(g) for g in grid]
    for a, b, fa, fb in zip(grid, grid[1:], vals, vals[1:]):
        if fa * fb < 0:
            roots.append(round(optimize.brentq(f, a, b), 3))
    v9.append({"cfg": c, "mean_deg": mean, "success": s, "lognormal_sigma_solutions": roots})
R["V9_reacher_lognormal_feasibility"] = v9
R["V9_note"] = "평균(Fig.11)과 임계 2° 이하 비율(Fig.5)을 동시에 만족하는 로그정규 분포의 σ가 존재하는지만 점검(분포 가정은 임의)"

# ---------------------------------------------------------------- V10 시간·횟수 산술
R["V10_arithmetic"] = {
    "2500_traj_hours": round(2500 * 2 / 3600, 3),
    "1000_traj_minutes": round(1000 * 2 / 60, 2),
    "steps_per_traj_500Hz": 2.0 * 500,
    "table_tennis_steps": round(1.09 / 0.002),
    "table_tennis_steps_div_action_repeat_5": round(1.09 / 0.002) / 5,
    "reacher_steps_div_5": 2.0 * 500 / 5,
    "train_split_2500": [0.8 * 2500, 0.2 * 2500],
    "unique_samples_per_epoch_approx": 0.8 * 2500 * (1000 - 5),
    "batches_per_epoch_bs512": round(0.8 * 2500 * (1000 - 5) / 512),
}
nb = R["V10_arithmetic"]["batches_per_epoch_bs512"]
R["V10_arithmetic"]["gradient_steps_150ep"] = nb * 150
R["V10_arithmetic"]["steps_per_sec_if_25min"] = round(nb * 150 / (25 * 60), 1)
R["V10_arithmetic"]["steps_per_sec_if_12h_R30"] = round(nb * 150 / (12 * 3600), 1)

# ---------------------------------------------------------------- V11 부록 F 계산(Chou-Hannaford 초기값)
D0, Fnom, hmax, p = 0.020, 1500.0, 0.25, 6e5
A = math.pi * D0 ** 2 / 4
Ap = A * p
sin2 = 2.0 / (Fnom / Ap + 3.0)
alpha = math.degrees(math.asin(math.sqrt(sin2)))
a_ = 3 / math.tan(math.radians(alpha)) ** 2
b_ = 1 / math.sin(math.radians(alpha)) ** 2
kel = Ap * (a_ * (1 - hmax) ** 2 - b_) / hmax
F0 = Ap * (a_ - b_)
R["V11_sysid_init"] = {"alpha0_deg": round(alpha, 2), "a": round(a_, 2), "b": round(b_, 3), "kel_N": round(kel, 1), "F_at_h0_N": round(F0, 1),
                       "paper": {"alpha0": 25.3, "a": 13.44, "b": 5.48, "kel": 1568}}
R["V11_identified_a_b_relation"] = {"paper_a_b": [15.6, 6.39], "b_if_coupled(1+a/3)": round(1 + 15.6 / 3, 2), "init_coupled(1+a/3)": round(1 + a_ / 3, 2)}
# 식별 파라미터 개수: 표 7에서 stage1 1개(Tp) + stage2 (a,b,c=3, kel=1, δ=8, g=8, τc=4, µ=4, d=4, τ0=4)
R["V11_param_count"] = {"stage1": 1, "stage2": 3 + 1 + 8 + 8 + 4 + 4 + 4 + 4, "total": 1 + 3 + 1 + 8 + 8 + 4 + 4 + 4 + 4, "paper": 37}
R["V11_Tp_at_grid_edge"] = {"grid_ms": [2, 50], "identified_ms": 50}
# 풀리 반경과 최대 토크 감 잡기: 최대 하중 F_nom * r (관절 1 r=38.5mm)
R["V11_max_joint_torque_Nm_joint1"] = round(Fnom * 0.0385, 1)

# ---------------------------------------------------------------- V12 모델 크기 (코드 설정 기준은 04에서; 여기서는 논문 표 1 기준)
inp = 4 * (4 + 4)  # H=3 -> (H+1)=4 항목 x (위치, 제어) x 4 관절
W = 512
# 은닉층 2개(입력-512, 512-512, 512-출력)와 3개(입력-512, 512-512 x2, 512-출력)의 두 해석
two_hidden = inp * W + W + (W * W + W) + (W * 4 + 4)
three_hidden = inp * W + W + 2 * (W * W + W) + (W * 4 + 4)
R["V12_genan_params"] = {"input_dim": inp, "two_hidden_layers_reading": two_hidden, "three_hidden_layers_reading": three_hidden,
                         "ensemble5_two": 5 * two_hidden, "ensemble5_three": 5 * three_hidden}

# ---------------------------------------------------------------- V13 1 step 오차와 인코더 분해능
# 인코더 눈금 수는 공식 저장소 sim2real/util/encoder_simulation.py 의 상수(ENCODER_TICKS)이다. 논문 본문에는 분해능이 없다.
ticks = np.array([20000, 18400, 14400, 14400])
res_deg = 360.0 / ticks
floor_deg = res_deg / 4  # 균등 양자화 오차의 평균 절댓값(목표 하나만 양자화되고 예측이 정확하다고 볼 때)
g1 = FIG["fig4a_inset"][0]["value"]
R["V13_encoder"] = {"ticks": ticks.tolist(), "resolution_deg": [round(x, 5) for x in res_deg],
                    "uniform_quantization_mean_abs_err_deg": [round(x, 5) for x in floor_deg],
                    "mean_over_joints_deg": round(float(floor_deg.mean()), 5),
                    "genan_1step_deg": round(g1, 5), "genan_1step_over_floor": round(g1 / float(floor_deg.mean()), 2),
                    "genan_1step_over_resolution": round(g1 / float(res_deg.mean()), 2),
                    "torque_genan_1step_over_floor": round(FIG["fig4a_inset"][1]["value"] / float(floor_deg.mean()), 2),
                    "diff_genan_torque_deg": round(FIG["fig4a_inset"][1]["value"] - g1, 5),
                    "diff_over_mean_resolution": round((FIG["fig4a_inset"][1]["value"] - g1) / float(res_deg.mean()), 3)}

(HERE / "recompute_results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float))

# ---------------------------------------------------------------- 요약 출력
print("V1 Wilson 대비 오차막대 최대 차이(퍼센트포인트):", R["V1_max_diff_pt"], "| 막대 값 최대 차이:", round(R["V1_value_max_diff_pt"], 3))
print("V3 표8 합계:", {k: v["row_sum"] for k, v in v3.items()}, "| 원 넓이 비율:", R["V3_circle_area_fraction_of_table_half"])
for r in rows:
    print("V4", r["task"], r["cfg"], r["genan"], "vs", r["other"], "p=", r["fisher_p"], "holm=", r["holm_p"], "diff", r["diff_pt"], r["diff_ci"])
print("V5", [(x["n_traj"], x["ensemble"], x["single"], x["fisher_p"]) for x in v5])
print("V5 1000 vs 2500:", R["V5_1000_vs_2500"])
print("V6 1-step:", R["V6_genan_1step_deg"], "ratio", R["V6_ratio_fig6a_2500_to_fig4a_1step"])
print("V6 500-step:", R["V6_genan_500step_deg"], "ratio", R["V6_ratio_fig6a_2500_to_fig4b_500step"])
print("V7:", R["V7_reduction_pct"], R["V7_fig4_values"], R["V7_ratios_vs_genan"])
print("V8:", R["V8_claim_stride_1step_s1_best"], R["V8_claim_stride_500_s2_lower"], R["V8_H20_worse_than_H10_1step"], R["V8_rollout_500_monotone_decreasing"])
print("V9:", v9)
print("V10:", R["V10_arithmetic"])
print("V11:", R["V11_sysid_init"], R["V11_param_count"], R["V11_max_joint_torque_Nm_joint1"])
print("V12:", R["V12_genan_params"])
print("V13:", R["V13_encoder"])
