"""논문 표·본문 수치 재계산과 그림 판독 값 기반 검사 (CPU, 수 초).

입력: 논문에 적힌 수치(아래 상수), analysis/curves_fig7.json, analysis/fig45_values.json
      (이 두 파일은 extract_curves.py, extract_fig45.py가 논문 PDF 벡터 그래프에서 읽은 값)
출력: analysis/recompute_results.json
사용법: python3 analysis/extract_curves.py; python3 analysis/extract_fig45.py; python3 analysis/recompute.py
"""
import json, math, os
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}

# ---------------- V1: 표 1 (벽시계 시간) ----------------
t1 = {"Walk": dict(train=(3.57, 1.74), infer=(40.8, 1.4)), "Run": dict(train=(3.54, 1.72), infer=(39.5, 1.3))}
v1 = {}
for k, d in t1.items():
    tr = d["train"]; inf = d["infer"]
    v1[k] = dict(train_ratio_cmpo_over_tdmpc2=tr[1] / tr[0], train_speedup=tr[0] / tr[1],
                 infer_speedup=inf[0] / inf[1],
                 ms_per_train_step_tdmpc2=tr[0] * 3600e3 / 200e3, ms_per_train_step_cmpo=tr[1] * 3600e3 / 200e3)
    # 가정: 표 1의 추론 시간이 학습 중 환경 상호작용 스텝마다 쓰이고(시드 5,000 스텝 제외), 평가 시간은 학습 시간에 포함되지 않는다.
    n_act = 200e3 - 5000
    inf_h_td = n_act * inf[0] * 1e-3 / 3600
    inf_h_cm = n_act * inf[1] * 1e-3 / 3600
    v1[k].update(assumed_inference_hours_tdmpc2=inf_h_td, assumed_inference_hours_cmpo=inf_h_cm,
                 residual_hours_tdmpc2=tr[0] - inf_h_td, residual_hours_cmpo=tr[1] - inf_h_cm,
                 residual_ratio_cmpo_over_tdmpc2=(tr[1] - inf_h_cm) / (tr[0] - inf_h_td))
OUT["V1_table1"] = v1

# ---------------- V2: 표 2 (MPC, w/o CM) ----------------
t2 = {"CMPO": dict(Walk=954.1, Run=629.2, Stair=60.9, Slide=265.8),
      "w MPC": dict(Walk=931.1, Run=864.5, Stair=71.9, Slide=391.2),
      "w/o CM": dict(Walk=22.2, Run=9.6, Stair=37.1, Slide=36.1)}
v2 = {"mpc_rel_change_pct": {k: (t2["w MPC"][k] / t2["CMPO"][k] - 1) * 100 for k in t2["CMPO"]},
      "woCM_rel_pct_of_cmpo": {k: t2["w/o CM"][k] / t2["CMPO"][k] * 100 for k in t2["CMPO"]}}
OUT["V2_table2"] = v2

# ---------------- V3: 실로봇 7/10 대 4/10 ----------------
a, b, n = 7, 4, 10
tab = [[a, n - a], [b, n - b]]
fe = stats.fisher_exact(tab)
v3 = dict(fisher_two_sided_p=fe.pvalue if hasattr(fe, "pvalue") else fe[1], fisher_one_sided_p=stats.fisher_exact(tab, alternative="greater")[1],
          barnard_two_sided_p=stats.barnard_exact(tab).pvalue,
          wilson_cmpo=proportion_confint(a, n, method="wilson"), wilson_sac=proportion_confint(b, n, method="wilson"))
# Newcombe(Wilson 기반) 차이 신뢰구간
l1, u1 = v3["wilson_cmpo"]; l2, u2 = v3["wilson_sac"]
p1, p2 = a / n, b / n
v3["diff"] = p1 - p2
v3["newcombe_diff_ci"] = (p1 - p2 - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), p1 - p2 + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))
# 같은 비율(0.7 대 0.4)이 유지될 때 필요한 시행 수(80% 검정력, 양측 5%, 정규근사)
za, zb = stats.norm.ppf(0.975), stats.norm.ppf(0.8)
pbar = (0.7 + 0.4) / 2
v3["n_per_arm_for_80pct_power_0.7_vs_0.4"] = ((za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(0.7 * 0.3 + 0.4 * 0.6)) / 0.3) ** 2
# 우연히 이 이상 벌어질 확률 외에 각 성공 횟수 조합의 p값 표
v3["fisher_p_by_cmpo_successes_with_sac_4of10"] = {str(k): stats.fisher_exact([[k, n - k], [b, n - b]])[1] for k in range(5, 11)}
OUT["V3_real_robot"] = v3

# ---------------- 그림 7 판독 값 ----------------
C = json.load(open(os.path.join(HERE, "curves_fig7.json")))
S = json.load(open(os.path.join(HERE, "curves_fig7_summary.json")))
TASKS = [t for t in S if t != "Average"]
METH = ["CMPO", "MAAC", "TD-MPC2", "DreamerV3", "SAC", "PPO"]
fin = {t: {m: S[t][m]["y_final"] for m in METH} for t in S}

# V4: 평균 패널 대 19개 과제 평균
avg_cmpo = np.array(C["curves"]["Average"]["CMPO"]["y"])
per = np.array([C["curves"][t]["CMPO"]["y"] for t in TASKS])
xs = np.array(C["curves"]["Average"]["CMPO"]["x_M"])
mean_of_tasks = {m: np.mean([np.array(C["curves"][t][m]["y"]) for t in TASKS], axis=0) for m in METH}
OUT["V4_average_panel"] = {m: dict(
    max_abs_diff=float(np.max(np.abs(mean_of_tasks[m] - np.array(C["curves"]["Average"][m]["y"])))),
    final_panel=float(C["curves"]["Average"][m]["y"][-1]), final_mean_of_tasks=float(mean_of_tasks[m][-1])) for m in METH}

# V5: 그림 2와 그림 7의 같은 과제 곡선 비교
v5 = {}
for t, ms in C["curves_fig2"].items():
    for m, c in ms.items():
        y2 = np.array(c["y"]); y7 = np.array(C["curves"][t][m]["y"])
        n_ = min(len(y2), len(y7))
        v5[f"{t}/{m}"] = float(np.max(np.abs(y2[:n_] - y7[:n_])) / max(1e-9, np.max(np.abs(y7))))
OUT["V5_fig2_vs_fig7_max_rel_diff"] = dict(max=max(v5.values()), by_curve=v5)

# V6: 마지막 지점 순위와 신뢰구간 기반 검정 (가정: 시드 3개, 음영 = 95% 신뢰구간)
tcrit = stats.t.ppf(0.975, 2)
def sd_from_band(lo, hi, crit):
    return (hi - lo) / 2 * math.sqrt(3) / crit
v6 = {"assumption": "시드 3개, 음영이 평균 ± t(0.975,df=2)·SE 라고 가정(대안: z=1.96)", "tasks": {}}
wins = 0; wins_ci = 0; wins_p = 0; wins_p_z = 0
for t in TASKS:
    c = S[t]["CMPO"]
    others = {m: S[t][m]["y_final"] for m in METH if m != "CMPO"}
    best_m = max(others, key=others.get)
    best_td = S[t]["TD-MPC2"]
    row = dict(cmpo_final=c["y_final"], best_baseline=best_m, best_baseline_final=others[best_m], cmpo_over_best=c["y_final"] / others[best_m],
               cmpo_over_tdmpc2=c["y_final"] / best_td["y_final"])
    row["cmpo_best"] = c["y_final"] > others[best_m]
    wins += row["cmpo_best"]
    b = S[t][best_m]
    row["ci_separated_vs_best"] = bool(c["band_lo_end"] > b["band_hi_end"]) if row["cmpo_best"] else False
    wins_ci += row["ci_separated_vs_best"]
    for crit, key in ((tcrit, "p_welch_t"), (1.96, "p_welch_z")):
        s1 = sd_from_band(c["band_lo_end"], c["band_hi_end"], crit); s2 = sd_from_band(b["band_lo_end"], b["band_hi_end"], crit)
        if s1 == 0 and s2 == 0:
            row[key] = None; continue
        res = stats.ttest_ind_from_stats(c["y_final"], s1, 3, b["y_final"], s2, 3, equal_var=False)
        row[key] = float(res.pvalue)
    row["sd_cmpo_t"] = sd_from_band(c["band_lo_end"], c["band_hi_end"], tcrit)
    if row["cmpo_best"] and row["p_welch_t"] is not None and row["p_welch_t"] < 0.05: wins_p += 1
    if row["cmpo_best"] and row["p_welch_z"] is not None and row["p_welch_z"] < 0.05: wins_p_z += 1
    v6["tasks"][t] = row
v6.update(n_tasks=len(TASKS), cmpo_best_final=wins, ci_separated=wins_ci, welch_p_lt_005_t=wins_p, welch_p_lt_005_z=wins_p_z,
          tasks_not_best=[t for t in TASKS if not v6["tasks"][t]["cmpo_best"]])
OUT["V6_final_ranking"] = v6

# V7: 평균의 구성 (원점수 평균은 값이 큰 과제가 지배)
sc = np.array([[fin[t][m] for m in METH] for t in TASKS])  # tasks x methods
share = sc[:, 0] / sc[:, 0].sum()
order = np.argsort(-sc[:, 0])
norm_by_best = sc / sc.max(axis=1, keepdims=True)
ratio_td = sc[:, 0] / sc[:, 2]
top4 = [TASKS[i] for i in order[:4]]
rest = [i for i in range(len(TASKS)) if i not in order[:4]]
OUT["V7_average_composition"] = dict(
    mean_raw_final={m: float(sc[:, j].mean()) for j, m in enumerate(METH)},
    cmpo_top4_tasks=top4, cmpo_top4_share_of_sum=float(share[order[:4]].sum()),
    mean_raw_without_top4={m: float(sc[rest, j].mean()) for j, m in enumerate(METH)},
    mean_normalized_by_task_best={m: float(norm_by_best[:, j].mean()) for j, m in enumerate(METH)},
    median_ratio_cmpo_over_tdmpc2=float(np.median(ratio_td)), geomean_ratio_cmpo_over_tdmpc2=float(np.exp(np.mean(np.log(ratio_td)))),
    ratio_cmpo_over_tdmpc2_by_task={t: float(r) for t, r in zip(TASKS, ratio_td)},
    n_tasks_cmpo_below_tdmpc2=int((ratio_td < 1).sum()),
    maac_final_range=[float(sc[:, 1].min()), float(sc[:, 1].max())],
    n_tasks_maac_below_tdmpc2=int((sc[:, 1] < sc[:, 2]).sum()), n_tasks_maac_below_sac=int((sc[:, 1] < sc[:, 4]).sum()))

# V8: 표 2 CMPO 값 대 그림 7 평균 곡선 (그림 판독 값)
v8 = {}
for t in ("Walk", "Run", "Stair", "Slide"):
    s = S[t]["CMPO"]
    v8[t] = dict(table2=t2["CMPO"][t], fig7_curve_max=s["y_max"], fig7_final=s["y_final"], fig7_ci_hi_end=s["band_hi_end"],
                 table2_over_curve_max=t2["CMPO"][t] / s["y_max"])
OUT["V8_table2_vs_fig7"] = v8

# V9: 그림 4(b), 5(a), 5(b)
F = json.load(open(os.path.join(HERE, "fig45_values.json")))
g = F["fig4b_grad_norm_by_horizon"]
OUT["V9_fig4b"] = dict(read=g,
    mean_ratio_maac_over_cmpo={h: g["MAAC"][h]["mean"] / g["CMPO"][h]["mean"] for h in g["MAAC"]},
    max_ratio_maac_over_cmpo={h: g["MAAC"][h]["max"] / g["CMPO"][h]["max"] for h in g["MAAC"]},
    maac_mean_h2_to_h4_change_pct=(g["MAAC"]["4"]["mean"] / g["MAAC"]["2"]["mean"] - 1) * 100,
    maac_max_h2_to_h4_change_pct=(g["MAAC"]["4"]["max"] / g["MAAC"]["2"]["max"] - 1) * 100,
    cmpo_mean_h2_to_h4_change_pct=(g["CMPO"]["4"]["mean"] / g["CMPO"]["2"]["mean"] - 1) * 100)
e = F["fig5b_latent_mse_x1e-3"]
OUT["V10_fig5b"] = dict(read=e, step5_ratio_recurrent_over_chunk5=e["Recurrent"]["5"] / e["Chunk h=5"]["5"],
    step1_ratio_chunk5_over_recurrent=e["Chunk h=5"]["1"] / e["Recurrent"]["1"],
    slope_per_step_recurrent=(e["Recurrent"]["5"] - e["Recurrent"]["1"]) / 4, slope_per_step_chunk5=(e["Chunk h=5"]["5"] - e["Chunk h=5"]["1"]) / 4,
    growth_x_recurrent=e["Recurrent"]["5"] / e["Recurrent"]["1"], growth_x_chunk5=e["Chunk h=5"]["5"] / e["Chunk h=5"]["1"])
h = F["fig5a_walk_horizon_reward"]
tdm_walk = S["Walk"]["TD-MPC2"]["y_final"]
OUT["V11_fig5a"] = dict(read=h, tdmpc2_walk_final_fig7=tdm_walk,
    ratio_to_tdmpc2_final={k: v["final"] / tdm_walk for k, v in h.items()},
    h3_vs_best=h["H=3"]["final"] / max(h["H=2"]["final"], h["H=4"]["final"]),
    fig7_walk_cmpo_final=S["Walk"]["CMPO"]["y_final"])

# V12: 파라미터 수 (config.yaml 기본값, 인코더 제외, LayerNorm 포함)
def mlp_params(inp, hid, out, n_hidden=2):
    p = 0; d = inp
    for _ in range(n_hidden):
        p += d * hid + hid + 2 * hid  # Linear + LayerNorm
        d = hid
    p += d * out + out + 2 * out  # 마지막 NormedLinear(dynamics) 또는 Linear
    return p
A, L, M = 61, 512, 512
rows = {}
for H in (1, 3, 5):
    dyn = mlp_params(L + A * H, M, L * H)
    pi = mlp_params(L, M, 2 * A * H) - 2 * 2 * A * H  # 정책은 마지막이 LayerNorm 없는 Linear
    rows[f"H={H}"] = dict(dynamics=dyn, policy=pi)
rows["dyn_ratio_H3_over_H1"] = rows["H=3"]["dynamics"] / rows["H=1"]["dynamics"]
rows["policy_ratio_H3_over_H1"] = rows["H=3"]["policy"] / rows["H=1"]["policy"]
OUT["V12_param_counts_approx"] = rows

# V13: 보고된 서술의 정수 가능성/단순 산술
OUT["V13_misc"] = dict(
    shown_tasks_in_fig2=9, tasks_total=19,
    gamma_from_code=min(max((1000 / 5 - 1) / (1000 / 5), 0.95), 0.995),
    gamma_pow_Hm1_H3=0.995 ** 2, gamma_pow_Hm1_H5=0.995 ** 4,
    prop1_bound_examples={f"L={L_},k={k}": sum(L_ ** i for i in range(k)) for L_ in (0.9, 1.0, 1.1, 1.3) for k in (1, 3, 5)})

json.dump(OUT, open(os.path.join(HERE, "recompute_results.json"), "w"), indent=1, default=lambda o: list(o) if isinstance(o, tuple) else float(o))
# 요약 출력
print(json.dumps({k: OUT[k] for k in ("V1_table1", "V2_table2", "V3_real_robot")}, indent=1, default=float))
for k in ("V4_average_panel", "V5_fig2_vs_fig7_max_rel_diff"):
    print(k, json.dumps(OUT[k], default=float)[:600])
v = OUT["V6_final_ranking"]; print({k: v[k] for k in v if k != "tasks"})
print(json.dumps(OUT["V7_average_composition"], indent=0, default=float)[:2500])
print(json.dumps(OUT["V8_table2_vs_fig7"], default=float))
print(json.dumps({k: OUT[k] for k in ("V9_fig4b", "V10_fig5b", "V11_fig5a", "V12_param_counts_approx")}, default=float)[:3500])
