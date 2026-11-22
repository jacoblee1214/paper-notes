"""LeaP (arXiv:2606.17408 v1) 표 수치 재계산과 검정. 학습이나 추론은 하지 않고 논문 표 값만 쓴다.

검사 항목(결과는 recompute_results.json):
  V1  표 1 평균 행 재계산(15과제 단순 평균)과 보고값 대조
  V2  표 1 셀의 정수 복원: 3시드 x 100회 롤아웃이면 평균은 k/3, SD는 정수 3개에서 나와야 함
  V3  표 1 과제별 승패, 부호 검정, Wilcoxon, 과제 부트스트랩 구간, 과제별 Welch 검정
  V4  표 1 평균의 SD(+-0.4 등)가 과제별 SD로 설명되는지 몬테카를로
  V5  실로봇(그림 4 막대값, 20회): 평균 재계산, Fisher, Wilson, 검정력
  V6  절제 표 6-9 평균 재계산, 과제별 이질성, 300회 합산 Fisher
  V7  시드 0 절제 값과 표 1 3시드 값의 일관성
  V8  파라미터 수 비율(그림 3a, 표 5)
실행: python3 recompute.py   (numpy, scipy, statsmodels)
"""
import json
import itertools
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

OUT = {}
M = ["LeaP", "A2A", "VITA", "NoPrior", "BridgePolicy"]

# 표 1 (mean, sd), 논문 p.6
T1 = {
 "beat_block_hammer":     [(94.0,2.6),(89.3,2.3),(89.7,2.9),(46.7,4.0),(79.3,3.1)],
 "dump_bin_bigbin":       [(92.3,6.7),(88.0,6.0),(89.0,1.7),(70.7,6.4),(77.3,9.5)],
 "click_bell":            [(100.0,0.0),(94.7,1.5),(98.0,2.6),(92.0,13.9),(77.7,4.0)],
 "handover_block":        [(91.7,1.5),(92.3,2.1),(83.0,6.6),(38.7,11.6),(60.3,3.8)],
 "place_phone_stand":     [(75.0,0.0),(61.0,4.6),(64.3,2.1),(50.3,4.0),(37.0,1.0)],
 "lift_pot":              [(98.3,1.5),(92.7,4.2),(97.0,0.0),(87.0,2.0),(91.3,2.5)],
 "move_playingcard_away": [(75.0,4.4),(72.0,2.0),(76.3,4.0),(59.0,5.0),(67.3,5.0)],
 "open_laptop":           [(86.0,6.9),(70.3,4.0),(62.7,3.8),(46.3,3.5),(50.3,3.5)],
 "pick_diverse_bottles":  [(77.0,6.1),(43.3,5.5),(63.0,4.6),(37.3,8.1),(34.0,3.0)],
 "click_alarmclock":      [(97.0,1.0),(89.7,4.0),(88.3,8.0),(81.3,6.1),(67.3,10.0)],
 "place_empty_cup":       [(86.7,4.2),(88.3,3.5),(88.0,1.7),(74.7,21.7),(67.0,3.6)],
 "press_stapler":         [(71.3,9.5),(56.0,5.3),(56.7,5.8),(68.0,8.7),(49.7,6.7)],
 "place_can_basket":      [(81.3,2.1),(73.3,7.6),(73.3,11.7),(46.3,17.8),(77.0,2.6)],
 "stack_blocks_two":      [(33.7,0.6),(31.0,2.0),(35.0,7.5),(16.7,3.8),(22.0,3.6)],
 "place_object_basket":   [(64.3,4.0),(65.7,3.2),(62.3,5.5),(27.0,6.6),(63.3,5.5)],
}
T1_AVG = [(81.6,0.4),(73.8,2.2),(75.1,0.9),(56.1,2.5),(61.4,1.3)]
tasks = list(T1)
mean = np.array([[c[0] for c in T1[t]] for t in tasks])   # 15 x 5
sd = np.array([[c[1] for c in T1[t]] for t in tasks])

# ---- V1 평균 행
avg = mean.mean(0)
OUT["V1"] = {m: dict(recomputed=round(float(a), 2), reported=r[0], diff=round(float(a) - r[0], 2))
             for m, a, r in zip(M, avg, T1_AVG)}
gaps = {m: round(float(avg[0] - avg[i]), 2) for i, m in enumerate(M) if i}
OUT["V1"]["gaps_pp_from_recomputed"] = gaps   # 7.8, 6.5, 25.5, 20.2 -> "6.5 to 25.5" 확인

# ---- V2 정수 복원 (GRIM/SPRITE 식)
grid = np.arange(101)
a, b, c = np.meshgrid(grid, grid, grid, indexing="ij")
mask = (a <= b) & (b <= c)
tri = np.stack([a[mask], b[mask], c[mask]], 1).astype(float)
tmean = tri.mean(1)
tpop = tri.std(1)
tsamp = tri.std(1, ddof=1)
def feasible(m, s):
    ok_m = np.abs(tmean - m) <= 0.05 + 1e-9
    return dict(pop=bool((ok_m & (np.abs(tpop - s) <= 0.05 + 1e-9)).any()),
                samp=bool((ok_m & (np.abs(tsamp - s) <= 0.05 + 1e-9)).any()),
                mean_only=bool(ok_m.any()))
bad = {"pop": [], "samp": [], "mean_only": []}
n_cells = 0
for t in tasks:
    for mname, (m_, s_) in zip(M, T1[t]):
        n_cells += 1
        f = feasible(m_, s_)
        for k in bad:
            if not f[k]:
                bad[k].append(f"{t}/{mname}: {m_}+-{s_}")
OUT["V2"] = {"cells": n_cells, **{f"infeasible_{k}": v for k, v in bad.items()},
             "n_infeasible": {k: len(v) for k, v in bad.items()}}

# ---- V3 과제별 승패와 검정 (과제를 단위로, LeaP 대 각 기준선)
v3 = {}
rng = np.random.default_rng(0)
for j, m in enumerate(M):
    if j == 0:
        continue
    d = mean[:, 0] - mean[:, j]
    wins = int((d > 0).sum()); ties = int((d == 0).sum()); losses = int((d < 0).sum())
    sign_p = stats.binomtest(wins, wins + losses, 0.5).pvalue
    w = stats.wilcoxon(d)
    boots = [rng.choice(d, len(d)).mean() for _ in range(20000)]
    ci = np.percentile(boots, [2.5, 97.5])
    tt = stats.ttest_1samp(d, 0)
    # 과제별 Welch (n=3 시드, 보고 SD를 표본 SD로 가정 / 모 SD로 가정)
    sig = {}
    for conv, f in (("samp", 1.0), ("pop", np.sqrt(3 / 2))):
        n_sig = 0; lost_sig = 0
        for i in range(len(tasks)):
            s1, s2 = sd[i, 0] * f, sd[i, j] * f
            if s1 == 0 and s2 == 0:
                continue
            r = stats.ttest_ind_from_stats(mean[i, 0], s1, 3, mean[i, j], s2, 3, equal_var=False)
            if r.pvalue < 0.05 and d[i] > 0: n_sig += 1
            if r.pvalue < 0.05 and d[i] < 0: lost_sig += 1
        sig[conv] = dict(leap_sig_better=n_sig, leap_sig_worse=lost_sig)
    v3[m] = dict(mean_diff=round(float(d.mean()), 2), wins=wins, ties=ties, losses=losses,
                 sign_p=round(float(sign_p), 4), wilcoxon_p=round(float(w.pvalue), 5),
                 t_p=round(float(tt.pvalue), 5), boot95=[round(float(x), 2) for x in ci],
                 welch_by_task=sig, median_diff=round(float(np.median(d)), 2))
OUT["V3"] = v3
best = mean.argmax(1)
OUT["V3"]["leap_best_tasks"] = int((best == 0).sum())
OUT["V3"]["leap_not_best"] = [f"{t}: {M[best[i]]} {mean[i, best[i]]} > LeaP {mean[i,0]}" for i, t in enumerate(tasks) if best[i] != 0]
# 과제 단위 대형 효과: NoPrior 대비 격차 상위
gap_np = mean[:, 0] - mean[:, 3]
OUT["V3"]["gap_vs_noprior_range"] = [round(float(gap_np.min()), 1), round(float(gap_np.max()), 1)]
# 상위 3과제를 뺀 평균 격차(기준선별)
OUT["V3"]["mean_gap_without_top3_tasks"] = {
    m: round(float(np.sort(mean[:, 0] - mean[:, j])[:-3].mean()), 2) for j, m in enumerate(M) if j}

# ---- V4 평균의 SD가 과제별 SD로 설명되는지
v4 = {}
for j, m in enumerate(M):
    for conv, f in (("samp", 1.0), ("pop", np.sqrt(3 / 2))):
        s_task = sd[:, j] * f      # 표본 SD 기준으로 환산
        indep = np.sqrt((s_task ** 2).sum()) / 15          # 시드 간 독립이면 평균의 SD
        sims = []
        for _ in range(4000):
            x = rng.normal(mean[:, j][None, :], np.where(s_task > 0, s_task, 1e-9)[None, :], size=(3, 15))
            sims.append(x.mean(1).std(ddof=1 if conv == "samp" else 0))
        rep = T1_AVG[j][1]
        v4[f"{m}/{conv}"] = dict(indep_sd=round(float(indep), 2), reported=rep,
                                 p_leq_reported=round(float(np.mean(np.array(sims) <= rep)), 3))
OUT["V4"] = v4

# ---- V5 실로봇 (그림 4 막대 값, 20회)
RR = {"Pick Cube": [70, 45, 50, 60], "Close Box": [80, 80, 85, 35], "Pick & Place Sandbags": [90, 45, 70, 55]}
RM = ["LeaP", "VITA", "A2A", "NoPrior"]
rep_avg = {"LeaP": 80.0, "A2A": 68.3, "VITA": 56.7, "NoPrior": 46.7}
arr = np.array(list(RR.values())).T   # 4 x 3
bar_avg = arr.mean(1)
k = np.rint(arr / 5).astype(int)      # 20회 중 성공 수
v5 = {"bar_values": RR, "avg_from_bars": {m: round(float(x), 1) for m, x in zip(RM, bar_avg)},
      "avg_reported": rep_avg,
      "successes_of_20": {m: k[i].tolist() for i, m in enumerate(RM)},
      "pooled_successes_of_60": {m: int(k[i].sum()) for i, m in enumerate(RM)}}
v5["noprior_total_needed_for_46.7"] = 28
fp = {}
for i, m in enumerate(RM):
    if m == "LeaP":
        continue
    per = {}
    for ti, tname in enumerate(RR):
        tab = [[k[0, ti], 20 - k[0, ti]], [k[i, ti], 20 - k[i, ti]]]
        per[tname] = round(float(stats.fisher_exact(tab)[1]), 3)
    tab = [[k[0].sum(), 60 - k[0].sum()], [k[i].sum(), 60 - k[i].sum()]]
    pooled = float(stats.fisher_exact(tab)[1])
    # 과제로 층화한 CMH
    from statsmodels.stats.contingency_tables import StratifiedTable
    tabs = [np.array([[k[0, ti], 20 - k[0, ti]], [k[i, ti], 20 - k[i, ti]]]) for ti in range(3)]
    st = StratifiedTable(tabs)
    fp[m] = dict(per_task_fisher_p=per, pooled_fisher_p=round(pooled, 4),
                 cmh_p=round(float(st.test_null_odds().pvalue), 4),
                 pooled_diff_pp=round(100 * (k[0].sum() - k[i].sum()) / 60, 1))
v5["tests_leap_vs"] = fp
# NoPrior를 46.7(합 28/60)로 두고 다시
tab = [[k[0].sum(), 60 - k[0].sum()], [28, 32]]
v5["pooled_fisher_p_vs_noprior_if_28of60"] = round(float(stats.fisher_exact(tab)[1]), 4)
v5["wilson95_pooled"] = {m: [round(float(x), 3) for x in proportion_confint(int(k[i].sum()), 60, method="wilson")]
                          for i, m in enumerate(RM)}
v5["wilson95_per_task_leap"] = {t: [round(float(x), 3) for x in proportion_confint(int(k[0, ti]), 20, method="wilson")]
                                for ti, t in enumerate(RR)}
# 과제별 순위/승
v5["task_leap_best"] = {t: bool(k[0, ti] >= k[:, ti].max()) for ti, t in enumerate(RR)}
v5["task_leap_strict_best"] = {t: bool(k[0, ti] > np.delete(k[:, ti], 0).max()) for ti, t in enumerate(RR)}
v5["task_noprior_last"] = {t: bool(k[3, ti] <= k[:, ti].min()) for ti, t in enumerate(RR)}
# 검정력: p1=0.80 대 p2 (각 n), 정규근사 two-sided alpha 0.05, 80% 검정력에 필요한 팔당 시행
def need_n(p1, p2, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    return float(((za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p1 - p2) ** 2)
v5["needed_n_per_arm_80pct_power"] = {"LeaP80_vs_A2A68.3": round(need_n(0.80, 0.683)), "LeaP80_vs_VITA56.7": round(need_n(0.80, 0.567)),
                                       "LeaP80_vs_NoPrior46.7": round(need_n(0.80, 0.467))}
OUT["V5"] = v5

# ---- V6 절제 표 6-9
TASK3 = ["pick_diverse_bottles", "open_laptop", "handover_block"]
AB = {
 "T6": {"state (LeaP)": [74, 90, 92], "vision": [66, 77, 55], "v+s FiLM": [77, 82, 52], "v+s MLP": [72, 80, 56], "v+s CrossAttn": [76, 69, 33]},
 "T7": {"NLL (LeaP)": [74, 90, 92], "mu+full Sigma": [68, 85, 86], "GMM k=8": [70, 85, 81], "only mu": [63, 78, 93], "MSE": [50, 87, 87], "mu+fixed sigma": [47, 63, 78], "no prior": [41, 50, 52]},
 "T8": {"Full (LeaP)": [74, 90, 92], "w/o Align": [68, 86, 70], "w/o NLL": [63, 79, 81], "w/o Align&NLL": [79, 79, 21], "direct mu": [0, 0, 0]},
 "T9": {"FM": [41, 50, 52], "FM+LeaP": [74, 90, 92], "Bridge (mu)": [64, 57, 85], "Bridge+LeaP": [74, 62, 94]},
}
REP_AVG = {"T6": [85.3, 66.0, 70.3, 69.3, 59.3], "T7": [85.3, 79.7, 78.7, 78.0, 74.7, 62.7, 47.7],
           "T8": [85.3, 74.7, 74.3, 59.7, 0.0], "T9": [47.7, 85.3, 68.7, 76.7]}
v6 = {}
for tb, rows in AB.items():
    out = {}
    ref_key = list(rows)[0] if tb != "T9" else "FM+LeaP"
    ref = np.array(rows[ref_key])
    for (name, vals), rep in zip(rows.items(), REP_AVG[tb]):
        vals = np.array(vals)
        avg_ = round(float(vals.mean()), 2)
        row = dict(recomputed_avg=avg_, reported_avg=rep, match=abs(avg_ - rep) <= 0.05 + 1e-9)
        if name != ref_key:
            tab = [[int(ref.sum()), 300 - int(ref.sum())], [int(vals.sum()), 300 - int(vals.sum())]]
            row["fisher_p_300trials_vs_ref"] = float(f"{stats.fisher_exact(tab)[1]:.2g}")
            row["tasks_where_variant_ge_ref"] = [TASK3[i] for i in range(3) if vals[i] >= ref[i]]
            row["per_task_fisher_p"] = {TASK3[i]: float(f"{stats.fisher_exact([[ref[i],100-ref[i]],[vals[i],100-vals[i]]])[1]:.2g}") for i in range(3)}
        out[name] = row
    v6[tb] = out
# Table 9 쌍(Bridge vs Bridge+LeaP) 따로
b = np.array(AB["T9"]["Bridge (mu)"]); bl = np.array(AB["T9"]["Bridge+LeaP"])
v6["T9_bridge_vs_bridgeleap"] = dict(
    pooled_p=float(f"{stats.fisher_exact([[bl.sum(),300-bl.sum()],[b.sum(),300-b.sum()]])[1]:.2g}"),
    per_task_p={TASK3[i]: float(f"{stats.fisher_exact([[bl[i],100-bl[i]],[b[i],100-b[i]]])[1]:.2g}") for i in range(3)})
# 이질성: 평균 격차에서 handover_block이 차지하는 몫
for tb, key in (("T6", "state (LeaP)"), ("T8", "Full (LeaP)")):
    ref = np.array(AB[tb][key])
    shares = {}
    for name, vals in AB[tb].items():
        if name == key:
            continue
        dd = ref - np.array(vals)
        if dd.sum() != 0:
            shares[name] = round(float(dd[2] / dd.sum()), 2)
    v6[f"{tb}_handover_share_of_gap"] = shares
OUT["V6"] = v6

# ---- V7 시드 0 절제와 표 1(3시드)의 일관성
idx = [tasks.index(t) for t in TASK3]
cons = {}
for label, col, vals in (("LeaP", 0, AB["T9"]["FM+LeaP"]), ("NoPrior", 3, AB["T9"]["FM"]), ("BridgePolicy_vs_BridgeMu", 4, AB["T9"]["Bridge (mu)"])):
    z = [(v - mean[i, col]) / sd[i, col] if sd[i, col] else None for v, i in zip(vals, idx)]
    cons[label] = dict(table1_3task_mean=round(float(mean[idx, col].mean()), 1), seed0_3task_mean=round(float(np.mean(vals)), 1),
                       z_per_task=[None if x is None else round(float(x), 2) for x in z])
cons["gap_3task_table1_LeaP_minus_NoPrior"] = round(float(mean[idx, 0].mean() - mean[idx, 3].mean()), 1)
cons["gap_3task_ablation_LeaP_minus_NoPrior"] = round(85.3 - 47.7, 1)
OUT["V7"] = cons

# ---- V8 파라미터
P = {"NoPrior": 13.01, "LeaP": 13.22, "A2A": 16.92, "VITA": 16.88, "ACT": 83.89, "DP3": 262.43, "BridgePolicy": 270.99}
OUT["V8"] = {"prior_head_share_pct": round(0.21 / 13.22 * 100, 2),
             "leap_minus_noprior_M": round(13.22 - 13.01, 2),
             "A2A_over_LeaP_pct": round((16.92 / 13.22 - 1) * 100, 1), "VITA_over_LeaP_pct": round((16.88 / 13.22 - 1) * 100, 1),
             "ratio_vs_LeaP": {k: round(v / 13.22, 2) for k, v in P.items()}}

OUT["data"] = {"T1": T1, "T1_avg": T1_AVG, "ablation": AB, "real_robot_bars": RR}
with open("recompute_results.json", "w") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print("wrote recompute_results.json")
