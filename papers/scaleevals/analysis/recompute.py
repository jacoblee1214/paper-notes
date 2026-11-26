"""ScaleEvals (arXiv:2611.28411 v1) 표 값의 재계산과 일관성 검사.

입력은 모두 논문 본문·표·그림 캡션에서 옮긴 값이다(위치는 각 블록의 주석). 그림에서 읽은 값은 없다.
출력: analysis/recompute_results.json (검사별 V1–V14 항목)

실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
R = {}

POL = ["DreamZero", "pi0.5", "GR00T N1.7", "MolmoAct2", "GR00T N1.6"]
ELO = 400 / math.log(10)  # 논문 표 1 캡션: Elo 척도 400/ln 10

# 표 1 / 표 8 (n = 350 per policy, 균형 부분집합)
BT = np.array([1580, 1539, 1500, 1456, 1453.0])
BT_SE = np.array([16, 0, 17, 16, 16.0])
TP = np.array([38.5, 30.7, 23.2, 17.3, 15.5])
SR = np.array([13.1, 7.7, 3.4, 2.6, 0.6])
M_TP = np.array([54.4, 45.0, 43.9, 27.0, 34.3])  # 표 1 수동 평가 평균 TP
M_SR = np.array([31.6, 12.5, 10.0, 5.0, 12.5])
M_N = np.array([38, 40, 40, 40, 40])
PAIRS = np.array([989, 748, 809, 950, 970])  # 표 8
# 표 4 (B = 100 균형 부분집합)
FULL_N = np.array([385, 681, 601, 440, 379])
BT100 = np.array([1594, 1524, 1500, 1470, 1453.0])
BT100_SE = np.array([15, 30, 5, 28, 25.0])
TP100 = np.array([39.1, 30.0, 23.3, 17.5, 14.7])
SR100 = np.array([14.0, 8.0, 4.0, 2.5, 0.0])


def pearson_p(r, n):
    t = r * math.sqrt(n - 2) / math.sqrt(1 - r * r)
    return float(2 * stats.t.sf(abs(t), n - 2))


def fisher_ci(r, n, alpha=0.05):
    z = math.atanh(r)
    se = 1 / math.sqrt(n - 3)
    q = stats.norm.ppf(1 - alpha / 2)
    return [math.tanh(z - q * se), math.tanh(z + q * se)]


def exact_spearman_p(rho_obs, n):
    """귀무(순열) 하에서 |rho| >= |rho_obs| 인 확률(양측), n! 순열 열거."""
    base = np.arange(n)
    cnt = tot = 0
    for perm in itertools.permutations(range(n)):
        d2 = float(np.sum((base - np.array(perm)) ** 2))
        rho = 1 - 6 * d2 / (n * (n * n - 1))
        tot += 1
        if abs(rho) >= abs(rho_obs) - 1e-12:
            cnt += 1
    return cnt / tot


# ---------------------------------------------------------------- V1 정수 가능성
def feasible(pct, n, dec=1):
    ks = [k for k in range(0, n + 1) if round(100 * k / n, dec) == round(pct, dec)]
    return ks


v1 = {"sr_balanced_n350": {}, "sr_manual": {}, "sr_b100_n100": {}}
for p, s in zip(POL, SR):
    v1["sr_balanced_n350"][p] = {"reported": s, "k_over_350": feasible(s, 350)}
for p, s, n in zip(POL, M_SR, M_N):
    v1["sr_manual"][p] = {"reported": s, "n": int(n), "k": feasible(s, int(n))}
for p, s in zip(POL, SR100):
    v1["sr_b100_n100"][p] = {"reported": s, "k_over_100": feasible(s, 100),
                              "note": "정수 k/100이 아니면 K=40 부트스트랩 중앙값(두 값의 평균)일 수 있음"}
R["V1_integer_feasibility"] = v1

# ---------------------------------------------------------------- V2 상관과 검정 (n = 5)
v2 = {}
r_bt, p_bt = stats.pearsonr(BT, M_TP)
r_tp, p_tp = stats.pearsonr(TP, M_TP)
rho_bt = stats.spearmanr(BT, M_TP)
tau = stats.kendalltau(BT, M_TP)
v2["table1_values"] = {
    "pearson_BT_vs_manualTP": {"r": r_bt, "p_t_df3": p_bt, "reported": "r=0.94, p=0.018", "fisher95": fisher_ci(r_bt, 5)},
    "pearson_rawTP_vs_manualTP": {"r": r_tp, "p_t_df3": p_tp, "reported": "r=0.92, p=0.029", "fisher95": fisher_ci(r_tp, 5)},
    "spearman_BT_vs_manualTP": {"rho": float(rho_bt.statistic), "p_exact_perm": exact_spearman_p(float(rho_bt.statistic), 5),
                                "reported": "rho=0.90 (abstract에서 p=0.018 옆에 표기)"},
    "kendall_tau": {"tau": float(tau.statistic), "p_scipy": float(tau.pvalue)},
}
# 보고된 r에서 p 역검사
v2["p_from_reported_r_n5"] = {str(r): pearson_p(r, 5) for r in (0.94, 0.92, 0.93, 0.90, 0.99)}
v2["p_from_reported_r_n4_YAM"] = {"r=0.839": pearson_p(0.839, 4), "fisher95_r0.839_n4": "n-3=1이라 구간 계산 불가(자유도 부족)"}
v2["spearman_n4_rho0.80_exact_p"] = exact_spearman_p(0.80, 4)
v2["spearman_n5_rho1.0_exact_p"] = exact_spearman_p(1.0, 5)
# 지정 r에 대한 Fisher 구간 (r = 0.94, 0.92, 0.99, n = 5)
v2["fisher95_reported_r_n5"] = {str(r): fisher_ci(r, 5) for r in (0.94, 0.92, 0.99, 0.93, 0.90)}
R["V2_correlation_n5"] = v2

# ---------------------------------------------------------------- V3 / V5 표 7 (수동 평가 과제별 평균)
cols = ["GR00T N1.6", "GR00T N1.7", "pi0.5", "DreamZero", "MolmoAct2"]
nan = float("nan")
T7 = np.array([
    [1.00, 1.00, 1.00, 1.00, 1.00],  # banana on blue plate
    [0.75, 0.50, 0.25, 1.00, 0.25],  # red cube on yellow
    [0.25, 0.00, 0.33, 0.49, 0.00],  # oven croissant
    [0.75, 0.25, 0.25, 0.62, 0.00],  # bread toaster
    [0.50, 0.75, 0.50, nan, 0.12],   # marker drawer (DreamZero 결측)
    [0.25, 0.38, 0.38, 0.50, 0.50],  # flip cup
    [0.10, 0.20, 0.60, 0.60, 0.20],  # pour
    [0.00, 0.62, 1.00, 1.00, 0.50],  # slide plate
    [0.33, 0.66, 0.33, 0.66, 0.66],  # center cup
    [0.00, 0.00, 0.33, 0.00, 0.00],  # cover blue letter
    [0.00, 0.50, 0.75, 0.38, 0.62],  # remove all objects
    [0.25, 0.25, 0.25, 0.00, 0.12],  # separate bowls
    [0.60, 0.60, 0.70, 0.50, 0.00],  # all objects into box
    [0.67, 0.49, 0.00, 0.67, 0.00],  # cube on sandwich
    [0.25, 0.25, 0.12, 0.25, 0.25],  # assemble pyramid
    [0.25, 0.75, 0.25, 0.00, 0.62],  # disassemble pyramid
    [0.25, 1.00, 0.66, 0.83, 0.17],  # fold towel
    [0.00, 0.33, 0.00, 0.67, 0.00],  # mix vegetables
    [0.17, 0.00, 0.00, 0.67, 0.00],  # assemble burger
    [0.50, 0.25, 0.50, 0.50, 0.25],  # blocks matching hole
])
T7_REPORTED_MEAN = dict(zip(cols, [0.34, 0.44, 0.41, 0.54, 0.26]))
col_mean = np.nanmean(T7, axis=0)
v3 = {"column_mean_from_rows": dict(zip(cols, [round(float(x), 4) for x in col_mean])),
      "reported_mean_row": T7_REPORTED_MEAN,
      "table1_manual_TP": dict(zip(["DreamZero", "pi0.5", "GR00T N1.7", "MolmoAct2", "GR00T N1.6"], M_TP.tolist())),
      "tasks_with_value": [int(np.sum(~np.isnan(T7[:, j]))) for j in range(5)]}
# 표 7에서 다시 만든 순위와 상관 (열 순서를 POL 순서로 재배열)
order = [cols.index(p) for p in ["DreamZero", "pi0.5", "GR00T N1.7", "MolmoAct2", "GR00T N1.6"]]
m7 = col_mean[order] * 100
v3["manual_TP_from_table7_in_POL_order"] = dict(zip(POL, [round(float(x), 2) for x in m7]))
v3["diff_table1_minus_table7"] = dict(zip(POL, [round(float(a - b), 2) for a, b in zip(M_TP, m7)]))
r7, p7 = stats.pearsonr(BT, m7)
v3["pearson_BT_vs_table7_means"] = {"r": float(r7), "p": float(p7)}
v3["spearman_BT_vs_table7_means"] = {"rho": float(stats.spearmanr(BT, m7).statistic)}
v3["rank_by_table7_means"] = [POL[i] for i in np.argsort(-m7)]
v3["rank_by_table1_manual"] = [POL[i] for i in np.argsort(-M_TP)]
R["V3_table7_means"] = v3

# V4 표 7 셀로 본 수동 성공률(SR)의 가능 범위: 셀 평균 1.00이면 두 시행 모두 만점
v4 = {}
for j, name in enumerate(cols):
    col = T7[:, j]
    min_perfect = int(2 * np.nansum(col == 1.0))
    v4[name] = {"cells_equal_1.00": int(np.nansum(col == 1.0)), "min_perfect_trials": min_perfect,
                "n_trials": int(2 * np.sum(~np.isnan(col)))}
rep_sr = dict(zip(["DreamZero", "pi0.5", "GR00T N1.7", "MolmoAct2", "GR00T N1.6"], M_SR.tolist()))
for name in cols:
    n_tr = v4[name]["n_trials"]
    k_rep = round(rep_sr[{"pi0.5": "pi0.5"}.get(name, name)] / 100 * n_tr)
    v4[name]["reported_SR_pct"] = rep_sr[name]
    v4[name]["reported_perfect_trials"] = int(k_rep)
    v4[name]["consistent_with_lower_bound"] = bool(k_rep >= v4[name]["min_perfect_trials"])
R["V4_manual_SR_lower_bound"] = v4

# V5 수동 평가의 표본 잡음: 과제 단위 쌍체 검정과 부트스트랩 (표 7, 과제를 독립 단위로 취급 — 가정)
v5 = {"paired_t_by_task": {}, "bootstrap": {}}
pairs = [("DreamZero", "pi0.5"), ("pi0.5", "GR00T N1.7"), ("GR00T N1.7", "GR00T N1.6"),
         ("GR00T N1.6", "MolmoAct2"), ("pi0.5", "MolmoAct2"), ("DreamZero", "MolmoAct2")]
for a, b in pairs:
    x, y = T7[:, cols.index(a)], T7[:, cols.index(b)]
    ok = ~np.isnan(x) & ~np.isnan(y)
    t = stats.ttest_rel(x[ok], y[ok])
    w = stats.wilcoxon(x[ok] - y[ok], zero_method="wilcox", method="auto")
    v5["paired_t_by_task"][f"{a} - {b}"] = {"n_tasks": int(ok.sum()), "mean_diff_pp": float(100 * np.mean(x[ok] - y[ok])),
                                           "t": float(t.statistic), "p_t": float(t.pvalue), "p_wilcoxon": float(w.pvalue)}
rng = np.random.default_rng(0)
B = 20000
rs, rhos, flips, order_same = [], [], 0, 0
base_rank = list(np.argsort(-M_TP))
for _ in range(B):
    idx = rng.integers(0, 20, 20)
    boot = np.array([np.nanmean(T7[idx, j]) for j in range(5)])[order] * 100
    r_, _ = stats.pearsonr(BT, boot)
    rs.append(r_)
    rhos.append(stats.spearmanr(BT, boot).statistic)
    if list(np.argsort(-boot)) == [0, 1, 2, 3, 4]:
        order_same += 1
rs, rhos = np.array(rs), np.array(rhos)
v5["bootstrap"] = {"B": B, "pearson_r_BT_vs_bootstrapped_manual": {"median": float(np.median(rs)), "p5": float(np.percentile(rs, 5)),
                                                                   "p95": float(np.percentile(rs, 95)), "frac_ge_0.94": float(np.mean(rs >= 0.94))},
                   "spearman_rho": {"median": float(np.median(rhos)), "p5": float(np.percentile(rhos, 5)), "p95": float(np.percentile(rhos, 95)),
                                    "frac_eq_1.0": float(np.mean(rhos > 0.999)), "frac_ge_0.9": float(np.mean(rhos >= 0.899))},
                   "frac_bootstrap_ranking_equals_ScaleEvals_order": order_same / B,
                   "pearson_r_hist": {"edges": np.round(np.arange(0.0, 1.0001, 0.05), 2).tolist(),
                                      "counts": np.histogram(rs, bins=np.arange(0.0, 1.0001, 0.05))[0].tolist()},
                   "note": "수동 평균은 표 7 열 평균(표 1 값과 다름, V3). 과제 20개를 재표집, DreamZero는 19개 과제"}
# 한 정책 빼기 상관 (leave-one-policy-out)
loo = {}
for k in range(5):
    keep = [i for i in range(5) if i != k]
    r_, _ = stats.pearsonr(BT[keep], M_TP[keep])
    loo[f"drop {POL[k]}"] = float(r_)
v5["leave_one_policy_out_pearson_table1"] = loo
# 수동 20과제 중 AutoPrompter 프롬프트(A.3)가 금지하거나 피하라는 행동에 해당하는 3개(붓기 = 인덱스 6, 수건 접기 = 16, 숟가락으로 섞기 = 17)를 뺀 17과제의 열 평균
keep_idx = [i for i in range(20) if i not in (6, 16, 17)]
m17 = np.array([np.nanmean(T7[keep_idx, j]) for j in range(5)])[order] * 100
v5["excluding_pour_fold_mix_17_tasks"] = {
    "means_in_POL_order": dict(zip(POL, [round(float(x), 2) for x in m17])),
    "pearson_BT": float(stats.pearsonr(BT, m17)[0]), "p": float(stats.pearsonr(BT, m17)[1]),
    "spearman_BT": float(stats.spearmanr(BT, m17).statistic),
    "rank": [POL[i] for i in np.argsort(-m17)],
    "three_task_means_in_POL_order": dict(zip(POL, [round(float(x), 2) for x in (np.array([np.nanmean(T7[[6, 16, 17], j]) for j in range(5)])[order] * 100)]))}
R["V5_manual_noise"] = v5

# ---------------------------------------------------------------- V6 BT는 평균 TP의 로그 비와 거의 같은가
v6 = {"elo_per_ln_ratio": ELO}
pred_gap = [ELO * math.log(TP[i] / TP[i + 1]) for i in range(4)]
rep_gap = [float(BT[i] - BT[i + 1]) for i in range(4)]
v6["adjacent_pairs"] = [f"{POL[i]} - {POL[i + 1]}" for i in range(4)]
v6["gap_reported_elo"] = rep_gap
v6["gap_if_BT_equals_ELO_ln_TP"] = [round(x, 1) for x in pred_gap]
# 소프트 승리 목표 y = (Pi+eps)/(Pi+Pj+2eps)의 균형값으로부터 본 Elo 차이(평균 TP 비 기준)
eps = 0.01
v6["gap_from_soft_target_of_means"] = [round(ELO * math.log(((TP[i] + eps) / (TP[i] + TP[i + 1] + 2 * eps)) /
                                                           (1 - (TP[i] + eps) / (TP[i] + TP[i + 1] + 2 * eps))), 1) for i in range(4)]
# BT와 TP의 선형 관계: Elo 대 ln TP
r_l, _ = stats.pearsonr(BT, np.log(TP))
r_t, _ = stats.pearsonr(BT, TP)
v6["pearson_BT_vs_lnTP"] = float(r_l)
v6["pearson_BT_vs_TP"] = float(r_t)
slope = np.polyfit(np.log(TP), BT, 1)
v6["fit_BT_on_lnTP"] = {"slope": float(slope[0]), "intercept": float(slope[1]), "slope_expected_ELO": ELO}
# 인접 쌍 Elo 차이의 z (독립 SE 가정, 앵커 때문에 구조적 상관은 무시)
z = []
for i in range(4):
    se = math.sqrt(BT_SE[i] ** 2 + BT_SE[i + 1] ** 2)
    z.append(rep_gap[i] / se)
v6["adjacent_gap_z_independent_SE"] = [round(x, 2) for x in z]
v6["adjacent_gap_two_sided_p_z"] = [float(2 * stats.norm.sf(abs(x))) for x in z]
# 앵커(π0.5) 기준 차이: 앵커의 SE = 0이므로 DZ-π0.5, π0.5-N1.7, π0.5-MA2 는 표의 SE가 곧 차이의 SE
v6["vs_anchor"] = {POL[i]: {"gap": float(BT[i] - 1539), "se": float(BT_SE[i]),
                            "z": (float((BT[i] - 1539) / BT_SE[i]) if BT_SE[i] else None)} for i in range(5)}
# MolmoAct2 - GR00T N1.6: 둘 다 앵커 기준 SE 16 — 공분산을 모르므로 상관 0 ~ 0.9 범위
gap = BT[3] - BT[4]
v6["MA2_vs_N16"] = {"gap": float(gap),
                    "z_if_corr_0": float(gap / math.sqrt(16 ** 2 + 16 ** 2)),
                    "z_if_corr_0.5": float(gap / math.sqrt(16 ** 2 + 16 ** 2 - 2 * 0.5 * 16 * 16)),
                    "z_if_corr_0.9": float(gap / math.sqrt(16 ** 2 + 16 ** 2 - 2 * 0.9 * 16 * 16))}
# 인간 교정 점수(그림 16)로 같은 환산을 하면
G_AUTO = np.array([43.8, 32.3, 23.7, 17.6, 15.6])  # 그림 16 왼쪽 (자동)
G_HUM = np.array([36.9, 28.6, 16.2, 10.1, 4.6])    # 그림 16 오른쪽 (사람 교정)
v6["fig16_log_ratio_elo_gaps_auto"] = [round(ELO * math.log(G_AUTO[i] / G_AUTO[i + 1]), 1) for i in range(4)]
v6["fig16_log_ratio_elo_gaps_human"] = [round(ELO * math.log(G_HUM[i] / G_HUM[i + 1]), 1) for i in range(4)]
R["V6_BT_vs_log_TP"] = v6

# ---------------------------------------------------------------- V7 표 4 / 표 1 / 표 8 일관성
v7 = {}
v7["full_n_sum"] = int(FULL_N.sum())
v7["full_n_reported_2486"] = 2486
v7["balanced_total"] = 5 * 350
v7["B100_minus_B350_BT"] = (BT100 - BT).tolist()
v7["B100_minus_B350_TP"] = np.round(TP100 - TP, 2).tolist()
v7["B100_minus_B350_SR"] = np.round(SR100 - SR, 2).tolist()
v7["max_abs_BT_shift"] = float(np.max(np.abs(BT100 - BT)))
v7["max_abs_TP_shift"] = float(np.max(np.abs(TP100 - TP)))
v7["max_abs_SR_shift"] = float(np.max(np.abs(SR100 - SR)))
v7["caption_claims"] = "BT 이동 최대 15 Elo, TP·SR 이동 1 %p 미만"
v7["anchor_rows"] = {"table1_pi05_BT": 1539, "table4_pi05_BT": 1524, "table4_pi05_SE": 30, "table4_N17_BT": 1500, "table4_N17_SE": 5,
                     "note": "표 4는 π0.5를 앵커로 표시하지만 값이 1539가 아니고 SE가 0이 아님. N1.7이 1500 ± 5(그림 4의 N1.7 = 1500 재중심과 같은 모양)"}
v7["rollout_share_kept_in_balanced_350"] = dict(zip(POL, [round(350 / n, 3) for n in FULL_N]))
v7["pairs_sum"] = int(PAIRS.sum())
v7["comparisons_implied"] = int(PAIRS.sum() // 2)
v7["max_matches_one_to_one_per_policy_pair"] = 350
v7["max_comparisons_10_pairs"] = 10 * 350
v7["pairs_share_of_max_per_policy"] = dict(zip(POL, [round(float(p) / (4 * 350), 3) for p in PAIRS]))
v7["ratio_1750_over_198"] = 1750 / 198
v7["ratio_2486_over_198"] = 2486 / 198
v7["ratio_reported_8.75_12.4_equals_over_200"] = [1750 / 200, 2486 / 200]
v7["manual_rollouts"] = {"4x40+38": 4 * 40 + 38}
v7["yam_rollouts"] = 4 * 250
v7["total_physical_reported_gt_3000"] = {"2486+1000": 2486 + 1000, "plus_240_prospective": 2486 + 1000 + 240,
                                       "plus_530_drift": 2486 + 1000 + 240 + 530,
                                       "note": "530 interleaved 롤아웃이 2,486에 들어 있는지 논문에 없음"}
# 비율 환산
v7["audit_cells"] = {"policies x tags x 15": 5 * 7 * 15}
R["V7_tables_consistency"] = v7

# ---------------------------------------------------------------- V8 처리량과 인력
v8 = {"net_rollouts_per_h": {"scaleevals": 23.0, "manual": 14.8, "ratio": 23.0 / 14.8},
      "human_minutes_per_100": {"manual": 240, "scaleevals": 5, "reduction": 1 - 5 / 240, "reported": "98%"},
      "station_hours_per_100": {"manual": 100 / 14.8, "scaleevals": 100 / 23.0},
      "manual_active_human_share_of_station_time": 4 / (100 / 14.8),
      "seconds_per_rollout_net": {"manual": 3600 / 14.8, "scaleevals": 3600 / 23.0},
      "human_hours_for_corpus": {"scaleevals_2486": 2486 / 100 * 5 / 60, "manual_equivalent_2486": 2486 / 100 * 4,
                                 "manual_suite_198": 198 / 100 * 4},
      "interventions_per_530_rollouts": {"count": 5, "per_100": 5 / 530 * 100, "minutes_per_intervention_if_5min_per_100": 5 / (5 / 530 * 100)},
      "audit_labor_not_in_5min": {"rubric_judgments": 1454, "rollouts": 525, "share_of_2486": 525 / 2486,
                                  "note": "5 min/100은 운영 중 개입 시간으로 읽히며, AutoGrader 감사(525 롤아웃)와 장면 구성 시간이 포함되는지 논문에 없음"}}
R["V8_throughput_labor"] = v8

# ---------------------------------------------------------------- V9 태그별 ANOVA (표 11b)
TAGS = [("transfer", 707, 22.0, 0.001, 0.11), ("contact_rich", 298, 9.0, 0.001, 0.11), ("state_flip", 952, 17.2, 0.001, 0.07),
        ("long_horizon", 996, 19.5, 0.001, 0.07), ("articulation", 448, 7.9, 0.001, 0.07), ("spatial", 761, 13.3, 0.001, 0.07),
        ("precision", 333, 5.4, 0.001, 0.06), ("reposing", 272, 3.3, 0.011, 0.05)]
FIG9_PCT = {"transfer": 28, "contact_rich": 12, "state_flip": 38, "articulation": 18, "spatial": 31, "precision": 13, "reposing": 11}
FIG18B_ETA = {"transfer": 0.11, "contact_rich": 0.11, "state_flip": 0.07, "articulation": 0.07, "spatial": 0.06, "precision": 0.06, "reposing": 0.05}
v9 = []
k = 5
for name, n, F, p_rep, eta_rep in TAGS:
    eta = F * (k - 1) / (F * (k - 1) + (n - k))
    p = float(stats.f.sf(F, k - 1, n - k))
    row = {"tag": name, "n": n, "F": F, "eta2_from_F": round(eta, 4), "eta2_reported": eta_rep,
           "eta2_rounds_to_reported": round(eta, 2) == eta_rep, "p_from_F": p, "p_reported": ("<0.001" if p_rep == 0.001 else p_rep)}
    if name in FIG9_PCT:
        row["fig9_pct"] = FIG9_PCT[name]
        row["n_over_2486_pct"] = round(100 * n / 2486, 1)
        row["fig18b_eta2"] = FIG18B_ETA[name]
    v9.append(row)
R["V9_anova"] = v9
# BH-FDR: 표의 8개 p (p < 0.001은 상한 0.001로 대입) — reposing의 q
ps = np.array([r["p_from_F"] for r in v9])
order_ = np.argsort(ps)
m = len(ps)
q = np.empty(m)
prev = 1.0
for rank_i, idx in reversed(list(enumerate(order_, start=1))):
    prev = min(prev, ps[idx] * m / rank_i)
    q[idx] = prev
R["V9_bh_q"] = {r["tag"]: float(qi) for r, qi in zip(v9, q)}
R["V9_note_7_tags_BH"] = {"reposing_q_if_7_tests": float(ps[-1] * 1.0)}

# ---------------------------------------------------------------- V10 AutoGrader 감사 (그림 14, 15, 16, 한계 절)
TAGN = ["state_flip", "spatial", "transfer", "articulation", "precision", "contact_rich", "reposing"]
F14 = {  # (틀린 기준 수, 기준 총수) 와 셀 정확도(그림 14 굵은 값)
    "DreamZero": ([3, 5, 2, 7, 6, 4, 3], [46, 44, 38, 47, 44, 34, 37], [93, 89, 95, 85, 86, 88, 92]),
    "GR00T N1.6": ([5, 4, 4, 4, 5, 5, 7], [40, 43, 35, 46, 48, 42, 45], [88, 91, 89, 91, 90, 88, 84]),
    "GR00T N1.7": ([6, 0, 1, 6, 8, 3, 6], [44, 42, 45, 46, 44, 40, 42], [86, 100, 98, 87, 82, 92, 86]),
    "MolmoAct2": ([2, 5, 3, 5, 2, 1, 3], [43, 43, 37, 45, 41, 44, 43], [95, 88, 92, 89, 95, 98, 93]),
    "pi0.5": ([3, 2, 4, 7, 8, 3, 6], [49, 42, 46, 47, 48, 41, 43], [94, 95, 91, 85, 83, 93, 86]),
}
v10 = {"per_policy": {}, "cells": 35}
tot_wrong = tot_n = 0
mismatch = []
for p, (w, n, a) in F14.items():
    w, n, a = np.array(w), np.array(n), np.array(a)
    v10["per_policy"][p] = {"criteria": int(n.sum()), "wrong": int(w.sum()), "accuracy_pct": round(100 * (1 - w.sum() / n.sum()), 2)}
    tot_wrong += int(w.sum())
    tot_n += int(n.sum())
    for t, wi, ni, ai in zip(TAGN, w, n, a):
        calc = 100 * (1 - wi / ni)
        if int(math.floor(calc)) != ai and round(calc) != ai:
            mismatch.append([p, t, float(round(calc, 2)), int(ai)])
v10["total_criteria_from_fig14"] = tot_n
v10["total_wrong_from_fig14"] = tot_wrong
v10["overall_accuracy_pct_from_fig14"] = round(100 * (1 - tot_wrong / tot_n), 2)
v10["reported_total_judgments"] = 1454
v10["reported_accuracy_pct"] = 90.7
v10["cell_accuracy_mismatch(not floor/round)"] = mismatch
v10["mean_criteria_per_rollout"] = tot_n / 525
# 한계 절: 정확도 90.7%, 정밀도 66.7%, 재현율 94.5%, 매크로 F1 86.2% — 클래스 정의(통과 = 양성)를 가정해 혼동행렬 복원
N_ = 1454
acc, prec, rec = 0.907, 0.667, 0.945
# 오류 수 E = (1-acc) N = FP + FN. FN = (1-rec) P, FP = TP(1/prec - 1) = rec P (1/prec - 1). => P = E / ((1-rec) + rec (1/prec - 1))
E = (1 - acc) * N_
P = E / ((1 - rec) + rec * (1 / prec - 1))
TPc, FNc = rec * P, (1 - rec) * P
FPc = TPc * (1 / prec - 1)
TNc = N_ - P - FPc
f1_pos = 2 * prec * rec / (prec + rec)
prec_n, rec_n = TNc / (TNc + FNc), TNc / (TNc + FPc)
f1_neg = 2 * prec_n * rec_n / (prec_n + rec_n)
po = (TPc + TNc) / N_
pe = ((TPc + FNc) * (TPc + FPc) + (TNc + FPc) * (TNc + FNc)) / N_ ** 2
v10["implied_confusion_matrix(assume 'pass' is positive, N=1454)"] = {
    "TP": round(TPc, 1), "FN": round(FNc, 1), "FP": round(FPc, 1), "TN": round(TNc, 1),
    "human_positive_rate": round(P / N_, 4), "grader_positive_rate": round((TPc + FPc) / N_, 4),
    "F1_pass": round(f1_pos, 4), "F1_fail": round(f1_neg, 4), "macro_F1": round((f1_pos + f1_neg) / 2, 4), "reported_macro_F1": 0.862,
    "always_fail_baseline_accuracy": round((TNc + FPc) / N_, 4), "cohen_kappa": round((po - pe) / (1 - pe), 4),
    "FP_to_FN_ratio": round(FPc / FNc, 2)}
# 그림 16: 자동 대 사람 교정 TP
r16, _ = stats.pearsonr(G_AUTO, G_HUM)
v10["fig16"] = {"auto": G_AUTO.tolist(), "human": G_HUM.tolist(), "pearson_r": float(r16), "reported_r": 0.99,
                "spearman_rho": float(stats.spearmanr(G_AUTO, G_HUM).statistic),
                "exact_perm_p_rho1": exact_spearman_p(1.0, 5),
                "drop_pp": (G_AUTO - G_HUM).round(1).tolist(), "ratio_auto_over_human": (G_AUTO / G_HUM).round(2).tolist(),
                "gaps_auto": np.diff(-G_AUTO).round(1).tolist(), "gaps_human": np.diff(-G_HUM).round(1).tolist()}
# 감사 표본(그림 16 자동 점수) 대 표 1 균형 TP
v10["fig16_auto_vs_table1_TP"] = (G_AUTO - TP).round(1).tolist()
# F1 pass (그림 13) 정책별 평균
F13 = {"DreamZero": [90, 71, 97, 64, 67, 86, 88], "GR00T N1.6": [40, 29, 29, 18, 20, 0, 22], "GR00T N1.7": [44, 100, 93, 75, 50, 87, 92],
       "MolmoAct2": [67, 74, 67, 50, 57, 50, 83], "pi0.5": [83, 92, 94, 53, 75, 85, 44]}
v10["fig13_mean_F1_pass_by_policy"] = {p: round(float(np.mean(v)), 1) for p, v in F13.items()}
R["V10_autograder_audit"] = v10

# ---------------------------------------------------------------- V11 표 6 (예측 검증)
rho = [1.00, 1.00, 1.00, 1.00, 0.87, 1.00, 1.00, 1.00, 1.00, 0.00]
v11 = {"mean_rho": float(np.mean(rho)), "place_mean": float(np.mean(rho[:5])), "rotate_mean": float(np.mean(rho[5:])),
       "reported": {"overall": 0.887, "place": 0.974, "rotate": 0.800, "intro": "ρ=0.88"},
       "rollouts": 10 * 8 * 3}
# ρ = 0.87 은 3개 중 한 쌍 동점일 때(순위 1, 2.5, 2.5)의 값
v11["rho_with_one_tie_n3"] = float(stats.spearmanr([1, 2, 3], [1, 2.5, 2.5]).statistic)
v11["rho_red5"] = float(stats.spearmanr([2, 1, 3], [1, 2.5, 2.5]).statistic)
# 쌍별 일치율 (프롬프트 10개 x 쌍 3개 = 30쌍)
conc = 8 * 3 + 2 + 1  # 완전 일치 8개 프롬프트 + blue bowl(동점 1쌍 제외 2쌍) + Red 5 (일치 1쌍)
disc = 1
ties = 2
v11["pairwise"] = {"pairs_total": 30, "concordant": conc, "discordant": disc, "tied": ties,
                   "ties_excluded": conc / (conc + disc), "ties_half_credit": (conc + 0.5 * ties) / 30,
                   "ties_as_miss": conc / 30, "reported": "96%",
                   "k_over_30_that_rounds_to_96": [k for k in range(31) if round(100 * k / 30) == 96]}
v11["exact_order"] = {"reported": "9/10", "strict_no_ties": 8, "note": "blue bowl은 동점(ρ=0.87)인데 Exact ✓로 표기"}
# 집계 순위를 그대로 예측하는 기준선: 표 1 BT는 π0.5 > N1.7 > N1.6 (place 예측과 같음)
v11["baseline_global_rank"] = {"order": "pi0.5 > N1.7 > N1.6", "place_exact": "5/5 (예측과 동일)", "rotate_exact": "0/5 (관측 4/5는 N1.7 > π0.5)"}
v11["fisher_rotate_4of5_vs_0of5_two_sided"] = float(stats.fisher_exact([[4, 1], [0, 5]])[1])
v11["chance_exact_order_3policies_per_prompt"] = 1 / 6
v11["binom_P(>=9_of_10)_if_p=1/6"] = float(stats.binom.sf(8, 10, 1 / 6))
R["V11_prospective"] = v11

# ---------------------------------------------------------------- V12 표 16 YAM
v12 = {"pearson_p_n4": pearson_p(0.839, 4), "spearman_exact_p_n4_rho0.8": exact_spearman_p(0.8, 4),
       "rollouts": 4 * 250,
       "note": "BT와 평균 TP는 같은 롤아웃에서 나온 두 요약이라 독립 검증이 아님(V6). 수동 평가 기준 없음(논문 그림 7 캡션)"}
R["V12_yam"] = v12

# ---------------------------------------------------------------- V13 본문 내 표기 일치
v13 = {"items": [
    {"what": "서론의 ρ=0.88과 표 6 ρ̄=0.887", "diff": "소수 셋째 자리 반올림(0.887 → 0.89)과 '감사 전' 표기가 다름"},
    {"what": "표 7 캡션의 'Figure 6 calibration, Figure 7 radar'", "diff": "그림 7은 YAM, 수동 평가 레이더는 그림 6과 18a"},
    {"what": "본문 3.3 '일곱 개 주요 태그'와 부록 표 11b 여덟 행", "diff": "여덟째는 long horizon(이분 변수)으로 표 캡션에 설명"},
    {"what": "AutoGrader 지표: '그림 13과 6절'(4.2절)", "diff": "그림 13은 정책×태그 F1(통과), 일반 정확도는 그림 14"},
    {"what": "그림 17 캡션 8.75배, 12.4배", "diff": "1750/198 = 8.84, 2486/198 = 12.56 (200 기준이면 8.75, 12.43)"},
]}
R["V13_text_consistency"] = v13

OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT)
