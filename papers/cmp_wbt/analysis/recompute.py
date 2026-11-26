"""CMP (arXiv:2604.07457 v2) 표 수치 재계산과 검정. numpy와 scipy만 쓰며 수 초 안에 끝난다.

입력은 논문 표 2, 4, 5, 8의 값을 옮긴 것이고, 결과는 recompute_results.json 에 저장한다.
논문에 없는 시행 수는 가정으로 표시한다 (실로봇 표 4는 범주당 5 과제 x 3회 = 15회, 표 2는 궤적 7,000개).

실행: python3 analysis/recompute.py
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
res = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


# ---------------------------------------------------------------- 표 2 (시뮬레이션, 궤적 7,000개)
T2 = {  # (ID SR, ID ep, ID er, Geo SR, Geo ep, Geo er, Sen SR, Sen ep, Sen er)
    "UMI-on-Legs": (85.3, 4.9, 0.105, 4.7, None, None, 6.9, None, None),
    "Latent Shielding": (90.9, 4.4, 0.096, 5.6, None, None, 17.7, None, None),
    "Neural CBF": (95.9, 5.4, 0.123, 7.1, None, None, 19.8, None, None),
    "CMP 1.0": (84.2, 15.6, 0.291, 82.5, 102.2, 1.935, 56.7, 27.9, 0.687),
    "CMP 1.5": (95.2, 6.0, 0.139, 62.4, 90.8, 1.767, 49.7, 15.9, 0.475),
    "CMP 2.0": (94.7, 4.5, 0.106, 46.9, 79.0, 1.661, 40.3, 11.8, 0.367),
    "CMP 2.5": (91.3, 4.1, 0.097, 34.1, 69.4, 1.612, 34.5, 9.9, 0.308),
    "CMP 3.0": (94.7, 3.8, 0.091, 26.3, None, None, 29.3, None, None),
    "CMP 100": (94.9, 3.5, 0.086, 5.1, None, None, 16.1, None, None),
}
r = {}
u, c2, c1, c100 = T2["UMI-on-Legs"], T2["CMP 2.0"], T2["CMP 1.0"], T2["CMP 100"]
ls = T2["Latent Shielding"]
r["V1_geo_sr_ratio_R2_vs_umi"] = round(c2[3] / u[3], 3)
r["V1_sensor_sr_ratio_R2_vs_umi"] = round(c2[6] / u[6], 3)
r["V1_geo_sr_ratio_R1_vs_umi"] = round(c1[3] / u[3], 2)
r["V1_best_baseline_geo_sr_ratio"] = round(c2[3] / T2["Neural CBF"][3], 2)
r["V1_sensor_sr_ratio_latent_shielding_vs_umi"] = round(ls[6] / u[6], 2)
r["V1_sensor_sr_ratio_neural_cbf_vs_umi"] = round(T2["Neural CBF"][6] / u[6], 2)
r["V1_geo_sr_gain_pp_latent_shielding_neural_cbf"] = [round(ls[3] - u[3], 1), round(T2["Neural CBF"][3] - u[3], 1)]
r["V1_ID_sr_neural_cbf_minus_cmp_R2_pp"] = round(T2["Neural CBF"][0] - c2[0], 1)
r["V2_ID_ep_change_vs_umi_pct"] = round(100 * (c2[1] / u[1] - 1), 1)
r["V2_ID_er_change_vs_umi_pct"] = round(100 * (c2[2] / u[2] - 1), 1)
r["V2_ID_ep_change_vs_latent_shielding_pct"] = round(100 * (c2[1] / ls[1] - 1), 1)
r["V2_ID_er_change_vs_latent_shielding_pct"] = round(100 * (c2[2] / ls[2] - 1), 1)
r["V2_ID_ep_change_vs_R100_pct"] = round(100 * (c2[1] / c100[1] - 1), 1)
r["V2_ID_er_change_vs_R100_pct"] = round(100 * (c2[2] / c100[2] - 1), 1)
r["V2_R100_vs_umi_ID_ep_pct"] = round(100 * (c100[1] / u[1] - 1), 1)
r["V2_R100_vs_umi_ID_sr_pp"] = round(c100[0] - u[0], 1)
r["V2_R100_vs_umi_geo_sr_pp"] = round(c100[3] - u[3], 1)
r["V2_R100_vs_umi_sensor_sr_pp"] = round(c100[6] - u[6], 1)
# 표의 생략 규칙: SR < 30 이면 추적 지표를 "-"로 둔다
viol = []
for name, row in T2.items():
    for sr, ep, tag in ((row[3], row[4], "geo"), (row[6], row[7], "sensor")):
        if (sr < 30) != (ep is None):
            viol.append((name, tag, sr, ep))
r["V3_omission_rule_violations"] = viol
# 정수 가능성: 궤적 7,000개 가정 시 SR 소수 첫째 자리는 모두 k/7000 격자 위에 있는가
r["V4_grid_all_feasible_n7000"] = all(
    any(abs(100 * k / 7000 - v) < 0.05 for k in range(int(v * 70) - 2, int(v * 70) + 3))
    for row in T2.values() for v in (row[0], row[3], row[6]))
# 이항 표준오차 (n = 7,000, 한 번의 평가 기준) 와 부록 K가 적은 반복 실행 표준편차 3%p 비교
r["V4_binomial_se_pp"] = {f"p={p}": round(100 * math.sqrt(p * (1 - p) / 7000), 2) for p in (0.05, 0.47, 0.85, 0.95)}
r["V4_stated_repeat_sd_pp"] = 3.0
r["V4_ID_SR_R2.5_vs_R2.0_diff_pp"] = round(T2["CMP 2.5"][0] - T2["CMP 2.0"][0], 1)
r["V4_ID_SR_R2.5_vs_R2.0_diff_in_sd3"] = round((T2["CMP 2.0"][0] - T2["CMP 2.5"][0]) / 3.0, 2)
# 표 2 주석의 "생략된 표준편차 여유 0.0002 rad" 와 부록 K 의 "er 반복 실행 s.d. 약 0.002 rad"
r["V5_er_margin_ratio_appendixK_to_table2note"] = 0.002 / 0.0002
# 평가 시간: 본문 13초, 부록 F 2,500스텝 / 200 Hz
r["V6_traj_seconds_appendixF"] = 2500 / 200
res["table2"] = r

# ---------------------------------------------------------------- 표 4 (실로봇, 범주당 5 과제 x 3회 = 15회 가정)
N = 15
T4 = {  # SR 값 -> 성공 횟수는 SR% x 15 / 100
    "UMI-on-Legs": {"ID": 80.0, "Mod": 0.0, "Ext": 0.0},
    "Latent Shielding": {"ID": 73.3, "Mod": 33.3, "Ext": 20.0},
    "Neural CBF": {"ID": 80.0, "Mod": 60.0, "Ext": 40.0},
    "CMP": {"ID": 100.0, "Mod": 93.3, "Ext": 86.7},
}
counts = {m: {k: round(v * N / 100) for k, v in d.items()} for m, d in T4.items()}
r = {"counts_n15": counts}
r["V7_all_percent_match_k_over_15"] = all(
    abs(100 * counts[m][k] / N - T4[m][k]) < 0.06 for m in T4 for k in T4[m])
r["V7_cmp_total_successes"] = sum(counts["CMP"].values())
r["V7_cmp_total_failures"] = 3 * N - sum(counts["CMP"].values())
r["V7_cmp_total_wilson95"] = wilson(sum(counts["CMP"].values()), 45)
r["V8_wilson95_cmp"] = {k: wilson(v, N) for k, v in counts["CMP"].items()}
fis = {}
raw = []
labels = []
for cat in ("ID", "Mod", "Ext"):
    for base in ("UMI-on-Legs", "Latent Shielding", "Neural CBF"):
        a, b = counts["CMP"][cat], counts[base][cat]
        p = stats.fisher_exact([[a, N - a], [b, N - b]])[1]
        fis[f"{cat}: CMP vs {base} ({a}/15 vs {b}/15)"] = round(p, 4)
        raw.append(p)
        labels.append(f"{cat}: CMP vs {base}")
adj = holm(raw)
r["V9_fisher_two_sided"] = fis
r["V9_holm_adjusted_9_tests"] = {l: round(x, 4) for l, x in zip(labels, adj)}
# OOD 합산 (중등도 + 극단, 30회)
pool = {}
for base in ("UMI-on-Legs", "Latent Shielding", "Neural CBF"):
    a = counts["CMP"]["Mod"] + counts["CMP"]["Ext"]
    b = counts[base]["Mod"] + counts[base]["Ext"]
    pool[base] = {"cmp": f"{a}/30", "base": f"{b}/30",
                  "fisher_p": float(f"{stats.fisher_exact([[a, 30 - a], [b, 30 - b]])[1]:.3g}")}
r["V10_OOD_pooled_30_trials"] = pool
# CMP 가 ID 100%라는 점 - 15/15 대 12/15 에서 필요한 시행 수 (검정력 0.8, 100% 대 80%)
from math import ceil
p1, p2 = 0.95, 0.80  # 15/15 를 0.95 로 보정해 보수적으로 계산
za, zb = 1.959964, 0.841621
nreq = ceil(((za * math.sqrt(2 * ((p1 + p2) / 2) * (1 - (p1 + p2) / 2)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p1 - p2) ** 2)
r["V11_trials_per_arm_for_95_vs_80_power80"] = nreq
# 추적 오차 비교 (평균 ± SD, 표본 크기는 성공 횟수 가정)
ep = {"ID": {"UMI-on-Legs": (4.9, 1.6, 12), "Neural CBF": (4.8, 1.9, 12), "CMP": (5.1, 1.8, 15)},
      "Mod": {"Neural CBF": (7.0, 4.3, 9), "CMP": (9.6, 7.2, 14)},
      "Ext": {"Neural CBF": (19.3, 11.9, 6), "CMP": (19.2, 11.1, 13)}}
tt = {}
for cat, d in ep.items():
    for base, v in d.items():
        if base == "CMP":
            continue
        m1, s1, n1 = d["CMP"]
        m2, s2, n2 = v
        t, p = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
        tt[f"{cat}: CMP {m1} vs {base} {m2}"] = {"n_assumed": f"{n1} vs {n2}", "welch_p": round(float(p), 3)}
r["V12_ep_welch_survivors_n_assumed"] = tt
r["V12_ext_er_ratio_cmp_vs_cbf"] = round(0.87 / 0.24, 2)
# 지연 (ms): 표본 수 미기재, 효과 크기만
lat = {"UMI-on-Legs": (2.97, 0.15), "Latent Shielding": (3.89, 0.49), "Neural CBF": (5.36, 0.54), "CMP": (2.99, 0.14)}
eff = {}
for m, (mu, sd) in lat.items():
    if m == "UMI-on-Legs":
        continue
    mu0, sd0 = lat["UMI-on-Legs"]
    pooled = math.sqrt((sd ** 2 + sd0 ** 2) / 2)
    eff[m] = {"diff_ms": round(mu - mu0, 2), "cohen_d": round((mu - mu0) / pooled, 2)}
r["V13_latency_vs_umi"] = eff
r["V13_cmp_over_umi_pct"] = round(100 * (2.99 / 2.97 - 1), 2)
r["V13_cbf_over_umi_pct"] = round(100 * (5.36 / 2.97 - 1), 1)
res["table4"] = r

# ---------------------------------------------------------------- 표 5 (500개 상태, 5명 투표)
pct = {"Salvageable": (0.4, 9.2, 57.8), "Unsalvageable": (16.8, 14.4, 1.4)}
cnt = {k: [round(x * 5) for x in v] for k, v in pct.items()}  # 500 x pct/100
r = {"counts_500": cnt, "sum_pct": round(sum(sum(v) for v in pct.values()), 1)}
r["V14_all_on_grid_of_0.2pct"] = all(abs(x * 5 - round(x * 5)) < 1e-6 for v in pct.values() for x in v)
r["V14_total_count"] = sum(sum(v) for v in cnt.values())
lo = cnt["Unsalvageable"][0] / (cnt["Unsalvageable"][0] + cnt["Salvageable"][0])
hi = cnt["Salvageable"][2] / (cnt["Salvageable"][2] + cnt["Unsalvageable"][2])
r["V15_precision_low_bin"] = {"value_pct": round(100 * lo, 1), "n": cnt["Unsalvageable"][0] + cnt["Salvageable"][0],
                              "wilson95": wilson(cnt["Unsalvageable"][0], cnt["Unsalvageable"][0] + cnt["Salvageable"][0])}
r["V15_precision_high_bin"] = {"value_pct": round(100 * hi, 1), "n": cnt["Salvageable"][2] + cnt["Unsalvageable"][2],
                               "wilson95": wilson(cnt["Salvageable"][2], cnt["Salvageable"][2] + cnt["Unsalvageable"][2])}
mid_s, mid_u = cnt["Salvageable"][1], cnt["Unsalvageable"][1]
r["V16_middle_bin"] = {"n": mid_s + mid_u, "share_of_500_pct": round(100 * (mid_s + mid_u) / 500, 1),
                       "unsalvageable_pct": round(100 * mid_u / (mid_s + mid_u), 1),
                       "wilson95_unsalvageable": wilson(mid_u, mid_s + mid_u)}
bins_share = [round(100 * (cnt["Salvageable"][i] + cnt["Unsalvageable"][i]) / 500, 1) for i in range(3)]
r["V16_bin_share_pct"] = bins_share
# 이분화 (Wmax > 0.8 이면 "구제 가능" 예측)
tp, fn = cnt["Salvageable"][2], cnt["Salvageable"][0] + cnt["Salvageable"][1]
fp, tn = cnt["Unsalvageable"][2], cnt["Unsalvageable"][0] + cnt["Unsalvageable"][1]
acc = (tp + tn) / 500
pe = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / 500 ** 2
r["V17_binary_at_0.8"] = {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "accuracy": round(acc, 3),
                          "cohen_kappa": round((acc - pe) / (1 - pe), 3),
                          "sensitivity": round(tp / (tp + fn), 3), "specificity": round(tn / (tn + fp), 3)}
# 세 구간 순서 일치: Mann-Whitney 형태의 AUC (구간을 순위로 사용, 동점 0.5)
S = cnt["Salvageable"]; U = cnt["Unsalvageable"]
auc = 0.0
for i in range(3):
    for j in range(3):
        auc += S[i] * U[j] * (1.0 if i > j else 0.5 if i == j else 0.0)
r["V17_auc_from_three_bins"] = round(auc / (sum(S) * sum(U)), 3)
res["table5"] = r

# ---------------------------------------------------------------- 표 8, 부록 K
r = {}
umi_er, clip_er, rear_er, cmp_er = 0.105, 0.107, 0.130, 0.106
r["V18_rear_aug_ID_er_change_pct"] = round(100 * (rear_er / umi_er - 1), 1)
r["V18_text_says_nearly_30pct"] = True
r["V19_geo_sr_clip_minus_cmp_pp"] = round(90.2 - 46.9, 1)
r["V19_geo_sr_clip_over_cmp_ratio"] = round(90.2 / 46.9, 2)
r["V19_rev_sr_cmp_over_clip_ratio"] = round(65.2 / 33.0, 2)
r["V19_rev_sr_cmp_over_umi_ratio"] = round(65.2 / 19.0, 2)
r["V19_geo_diff_in_sd3"] = round((90.2 - 46.9) / 3.0, 1)
# 두 OOD 열 평균 (두 종류만 보고된 범위에서)
for name, g, v in (("UMI-on-Legs", 4.7, 19.0), ("Command Clipping", 90.2, 33.0), ("Rear-augmented", 87.0, 30.2), ("CMP R=2", 46.9, 65.2)):
    r[f"V20_mean_of_two_OOD_sr[{name}]"] = round((g + v) / 2, 1)
r["V21_table8_umi_cmp_match_table2"] = (4.7 == T2["UMI-on-Legs"][3] and 46.9 == T2["CMP 2.0"][3]
                                        and umi_er == T2["UMI-on-Legs"][2] and cmp_er == T2["CMP 2.0"][2])
res["table8"] = r

# ---------------------------------------------------------------- 데이터셋 구성 (그림 3, 부록 F)
r = {}
pool_cup, pool_toss = 1090, 101 * 3
frac_toss = pool_toss / (pool_cup + pool_toss)
n_toss = 2000 * frac_toss
n_cup = 2000 - n_toss
r["V22_fig3_shares_expected_pct"] = {"tossing": round(100 * n_toss / 7000, 2), "cup": round(100 * n_cup / 7000, 2),
                                     "random": round(100 * 5000 / 7000, 2)}
r["V22_fig3_shares_reported_pct"] = {"tossing": 6.2, "cup": 22.4, "random": 71.4}
r["V23_train_per_iter_seconds"] = round(2.5 * 3600 / 2000, 2)
r["V23_gradient_steps_per_iter"] = 32 * 4
r["V23_gradient_steps_total"] = 32 * 4 * 2000
res["dataset_training"] = r

# ---------------------------------------------------------------- 잠재 공간 차원 d=6 에서 셸 두께 (부록 B, 식 6)
d = 6
r = {"latent_dim_from_table6": d}
mean_chi = math.sqrt(2) * math.gamma((d + 1) / 2) / math.gamma(d / 2)
sd_chi = math.sqrt(d - mean_chi ** 2)
r["chi_mean"] = round(mean_chi, 3)
r["chi_sd"] = round(sd_chi, 3)
r["relative_sd"] = round(sd_chi / mean_chi, 3)
r["sqrt_d"] = round(math.sqrt(d), 3)
r["P(norm > 2.0) when z~N(0,I_6)"] = round(float(stats.chi2.sf(4.0, d)), 3)
res["latent_shell_chi6"] = r

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
