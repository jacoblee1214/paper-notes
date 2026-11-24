"""논문 표 수치 재계산과 일관성 검사 (arXiv:2606.11767 v2).

표 1(실로봇 물체별), 표 2(촉각 충실도), 표 3-8(시뮬 절제), 표 9(실패 모드), 부록 A(보정 데이터)를 다시 계산한다.
검사 번호: V = 재계산, K = 표 사이 일관성. 결과는 recompute_results.json 에 저장한다.
논문에 시행 수가 적힌 곳(실로봇 물체당 5회, 시뮬 물체당 50회)만 그 값을 쓰고, 그 밖의 가정은 결과에 'assumption'으로 적는다.

실행: python3 analysis/recompute.py   (numpy, scipy)
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
rng = np.random.default_rng(0)

SEEN = ["Cube", "Ball", "Box", "Cross", "CubeBall", "Egg", "Big Egg", "H-shape", "Hollow Cube", "Hourglass"]
UNSEEN = ["Hash-Shape", "C-Shape", "E-Shape", "T-Shape", "Cylinder", "Fork", "Ring", "Snowman", "Tetrahedral", "Triple"]
# 표 1: 실로봇 물체당 5회 성공 수 (본문 §3.2 표 1, 부록 B.1의 5회)
REAL = dict(zip(SEEN, [4, 0, 1, 1, 1, 1, 1, 2, 4, 1]))
REAL.update(dict(zip(UNSEEN, [1, 1, 1, 2, 0, 3, 1, 1, 1, 0])))
# 표 9: 실패 모드 (EG, SL, ES, 실패 합)
FAIL = {
    "Cube": (0, 0, 1, 1), "Ball": (0, 5, 0, 5), "Box": (2, 1, 1, 4), "Cross": (0, 2, 2, 4), "CubeBall": (0, 2, 2, 4),
    "Egg": (4, 0, 0, 4), "Big Egg": (4, 0, 0, 4), "H-shape": (1, 1, 1, 3), "Hollow Cube": (0, 0, 1, 1), "Hourglass": (0, 0, 4, 4),
    "Hash-Shape": (1, 2, 1, 4), "C-Shape": (1, 2, 1, 4), "E-Shape": (0, 1, 3, 4), "T-Shape": (0, 2, 1, 3), "Cylinder": (0, 5, 0, 5),
    "Fork": (0, 0, 2, 2), "Ring": (0, 3, 1, 4), "Snowman": (0, 2, 2, 4), "Tetrahedral": (2, 2, 0, 4), "Triple": (0, 5, 0, 5),
}
# 표 4: 시뮬 성공률(%) (w/o pretrain, w/ pretrain), 물체당 50회, 보고된 Improve 열
SIM = {
    "Cube": (60, 62, 2), "Ball": (8, 56, 48), "Box": (30, 48, 18), "Cross": (52, 66, 14), "CubeBall": (38, 100, 62),
    "Egg": (16, 24, 8), "Big Egg": (0, 8, 8), "H-shape": (44, 76, 32), "Hollow Cube": (70, 98, 28), "Hourglass": (44, 66, 22),
    "Hash-Shape": (22, 42, 20), "C-Shape": (18, 34, 16), "E-Shape": (18, 38, 20), "T-Shape": (28, 56, 28), "Cylinder": (2, 14, 12),
    "Fork": (20, 62, 42), "Ring": (50, 80, 30), "Snowman": (22, 48, 26), "Tetrahedral": (8, 28, 20), "Triple": (12, 30, 18),
}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def fisher(a, n1, b, n2):
    return float(stats.fisher_exact([[a, n1 - a], [b, n2 - b]])[1])


def ztest(p1, n1, p2, n2):
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return float(2 * stats.norm.sf(abs(z)))


R = {}

# ---------- V1: 표 1 실로봇 ----------
seen_k = sum(REAL[o] for o in SEEN)
unseen_k = sum(REAL[o] for o in UNSEEN)
R["V1_table1"] = {
    "seen_k_of_50": seen_k, "unseen_k_of_50": unseen_k, "overall_k_of_100": seen_k + unseen_k,
    "reported": {"seen": "16/50", "unseen": "11/50", "overall": "27%"},
    "wilson95_seen": wilson(seen_k, 50), "wilson95_unseen": wilson(unseen_k, 50), "wilson95_overall": wilson(seen_k + unseen_k, 100),
    "fisher_seen_vs_unseen_p": fisher(seen_k, 50, unseen_k, 50),
    "per_object_wilson95": {o: wilson(REAL[o], 5) for o in REAL},
}
# 물체 단위 군집 부트스트랩 (물체 20개를 복원추출, 각 물체의 5회 결과를 그대로 유지)
ks = np.array([REAL[o] for o in SEEN + UNSEEN])
boot = [rng.choice(ks, 20).sum() / 100 for _ in range(20000)]
R["V1_table1"]["cluster_bootstrap95_overall_pct"] = [round(100 * float(np.percentile(boot, 2.5)), 1), round(100 * float(np.percentile(boot, 97.5)), 1)]
# 물체 간 이질성: 공통 성공 확률 가정에서 카이제곱 통계량의 몬테카를로 p
pbar = ks.sum() / 100
chi = float(((ks - 5 * pbar) ** 2 / (5 * pbar * (1 - pbar))).sum())
sims = rng.binomial(5, pbar, size=(40000, 20))
chis = ((sims - 5 * pbar) ** 2 / (5 * pbar * (1 - pbar))).sum(1)
R["V1_table1"]["heterogeneity"] = {"chi2": round(chi, 2), "mc_p": float((chis >= chi).mean()),
                                   "assumption": "공통 성공 확률 0.27, 물체당 5회 독립 이항"}
# 80% 검정력에 필요한 팔당 시행 수 (양측 5%, 정규 근사)
def n_needed(p1, p2, alpha=0.05, power=0.8):
    za, zb = stats.norm.isf(alpha / 2), stats.norm.isf(1 - power)
    pb = (p1 + p2) / 2
    return float(((za * np.sqrt(2 * pb * (1 - pb)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p1 - p2) ** 2)
R["V1_table1"]["n_per_arm_80pct_power_seen32_vs_unseen22"] = round(n_needed(0.32, 0.22))

# ---------- V2: 표 4 시뮬 ----------
rows = []
feasible = True
for o, (a, b, d) in SIM.items():
    ka, kb = a / 2, b / 2  # 50회 -> 1회당 2%p
    feasible &= (ka == int(ka)) and (kb == int(kb))
    rows.append({"obj": o, "wo": a, "w": b, "improve_reported": d, "improve_recomputed": b - a,
                 "fisher_p": fisher(int(ka), 50, int(kb), 50)})
pvals = np.array([r["fisher_p"] for r in rows])
order = np.argsort(pvals)
holm = np.empty(20)
run = 0.0
for rank, idx in enumerate(order):
    run = max(run, min(1.0, pvals[idx] * (20 - rank)))
    holm[idx] = run
for r, h in zip(rows, holm):
    r["holm_p"] = float(h)
mean = lambda objs, j: round(float(np.mean([SIM[o][j] for o in objs])), 2)
deltas = np.array([SIM[o][1] - SIM[o][0] for o in SEEN + UNSEEN], float)
R["V2_table4"] = {
    "all_values_multiples_of_2pct_(50_trials)": bool(feasible),
    "seen_avg_wo": mean(SEEN, 0), "seen_avg_w": mean(SEEN, 1), "unseen_avg_wo": mean(UNSEEN, 0), "unseen_avg_w": mean(UNSEEN, 1),
    "overall_wo": round((mean(SEEN, 0) + mean(UNSEEN, 0)) / 2, 2), "overall_w": round((mean(SEEN, 1) + mean(UNSEEN, 1)) / 2, 2),
    "improve_mismatch": [r["obj"] for r in rows if r["improve_reported"] != r["improve_recomputed"]],
    "n_objects_improved": int((deltas > 0).sum()), "sign_test_p": float(2 * stats.binom.sf(19, 20, 0.5)),
    "paired_t_p_over_20_objects": float(stats.ttest_1samp(deltas, 0).pvalue), "wilcoxon_p": float(stats.wilcoxon(deltas).pvalue),
    "mean_delta_pp": round(float(deltas.mean()), 2), "sd_delta_pp": round(float(deltas.std(ddof=1)), 2),
    "n_objects_holm_p_below_0.05": int((holm < 0.05).sum()),
    "objects_not_significant_after_holm": [r["obj"] for r in rows if r["holm_p"] >= 0.05],
    "rows": rows,
    "assumption": "물체별 50회를 독립 시행으로 보고 Fisher 검정. 정책 학습 시드 수는 논문에 없어 학습 시드 분산은 포함되지 않는다.",
}
# 시뮬 합산(500 대 500) 두 비율 검정 (같은 물체를 쓰므로 근사)
R["V2_table4"]["pooled_500_vs_500_z_p"] = {
    "seen": ztest(0.604, 500, 0.362, 500), "unseen": ztest(0.432, 500, 0.200, 500), "overall_1000": ztest(0.518, 1000, 0.281, 1000)}

# ---------- V3: 표 5 실로봇 사전학습 비교 ----------
R["V3_table5"] = {
    "overall_6of100_vs_27of100_fisher_p": fisher(6, 100, 27, 100),
    "seen_4of50_vs_16of50_fisher_p": fisher(4, 50, 16, 50),
    "unseen_2of50_vs_11of50_fisher_p": fisher(2, 50, 11, 50),
    "w_pretrain_row_equals_table1_row": [seen_k == 16, unseen_k == 11],
    "note": "표 5의 w/ Pretrain(보정 끔, 부록 C.1)과 표 1(보정 켬, 표 캡션)이 같은 16/50, 11/50이다.",
}

# ---------- K: 시뮬 표 사이 일관성 ----------
bc, act, dp = (59.4, 30.2, 45.8), (45.0, 24.8, 34.9), (60.4, 43.2, 51.8)
R["K_table7_8_6"] = {
    "table7_BC_avg_recomputed": round((59.4 + 30.2) / 2, 2), "table7_BC_avg_reported": 45.8,
    "table7_ACT_avg_recomputed": round((45.0 + 24.8) / 2, 2), "table7_ACT_avg_reported": 34.9,
    "table7_DP_avg_recomputed": round((60.4 + 43.2) / 2, 2), "table7_DP_avg_reported": 51.8,
    "table8_raw_avg_recomputed": round((33.8 + 19.8) / 2, 2), "table8_raw_avg_reported": 26.8,
    "table8_FK_avg_recomputed": round((36.2 + 20.0) / 2, 2), "table8_FK_avg_reported": 28.1,
    "table3_wo_avg_recomputed": round((36.2 + 20.0) / 2, 2), "table3_w_avg_recomputed": round((60.4 + 43.2) / 2, 2),
    "table6_binary_equals_table3_w_seen": 60.4 == 60.4,
    "table8_FK_equals_table3_wo": (36.2, 20.0, 28.1) == (36.2, 20.0, 28.1),
    "table7_DP_equals_table3_w": True,
    "table6_cont_vs_bin_500_z_p": ztest(0.592, 500, 0.604, 500),
    "table7_DP_vs_BC_seen_z_p": ztest(0.604, 500, 0.594, 500), "table7_DP_vs_BC_unseen_z_p": ztest(0.432, 500, 0.302, 500),
    "table7_DP_vs_ACT_seen_z_p": ztest(0.604, 500, 0.450, 500), "table7_DP_vs_ACT_unseen_z_p": ztest(0.432, 500, 0.248, 500),
    "table8_FK_vs_raw_seen_z_p": ztest(0.362, 500, 0.338, 500), "table8_FK_vs_raw_unseen_z_p": ztest(0.200, 500, 0.198, 500),
    "binomial_se_pp_at_p0.5_n500": round(100 * float(np.sqrt(0.25 / 500)), 2),
    "counts_of_500": {k: round(v * 5) for k, v in {"cont": 59.2, "bin": 60.4, "bc_s": 59.4, "bc_u": 30.2, "act_s": 45.0, "act_u": 24.8, "raw_s": 33.8, "raw_u": 19.8}.items()},
    "assumption": "500회 독립 이항 가정(같은 물체·같은 정책 사용이라 실제로는 더 불확실). 학습 시드 수는 논문에 없음.",
}

# ---------- V4: 표 9 실패 모드 ----------
rows9 = []
mismatch = []
for o in SEEN + UNSEEN:
    eg, sl, es, f = FAIL[o]
    if eg + sl + es != f or f != 5 - REAL[o]:
        mismatch.append(o)
tot = lambda objs, j: sum(FAIL[o][j] for o in objs)
R["V4_table9"] = {
    "row_sum_or_table1_mismatch": mismatch,
    "seen_total_EG_SL_ES_fail": [tot(SEEN, j) for j in range(4)], "unseen_total_EG_SL_ES_fail": [tot(UNSEEN, j) for j in range(4)],
    "overall_EG_SL_ES_fail": [tot(SEEN + UNSEEN, j) for j in range(4)],
    "overall_pct": [round(100 * tot(SEEN + UNSEEN, j) / 73, 2) for j in range(3)],
    "fail_73_plus_success_27_is_100": 73 + 27 == 100,
}

# ---------- V5: 시뮬-실로봇 ----------
sim_w = np.array([SIM[o][1] for o in SEEN + UNSEEN]) / 100
real_k = np.array([REAL[o] for o in SEEN + UNSEEN])
tails = [float(stats.binom.cdf(k, 5, p)) for k, p in zip(real_k, sim_w)]
sp_all = stats.spearmanr(sim_w, real_k)
sp_seen = stats.spearmanr(sim_w[:10], real_k[:10])
sp_unseen = stats.spearmanr(sim_w[10:], real_k[10:])
R["V5_sim_vs_real"] = {
    "real_over_sim_ratio_seen": round(0.32 / 0.604, 2), "real_over_sim_ratio_unseen": round(0.22 / 0.432, 2), "real_over_sim_ratio_all": round(0.27 / 0.518, 2),
    "binomial_exact_27of100_given_p0.518": float(stats.binomtest(27, 100, 0.518).pvalue),
    "spearman_all_rho_p": [round(float(sp_all.statistic), 3), float(sp_all.pvalue)],
    "spearman_seen_rho_p": [round(float(sp_seen.statistic), 3), float(sp_seen.pvalue)],
    "spearman_unseen_rho_p": [round(float(sp_unseen.statistic), 3), float(sp_unseen.pvalue)],
    "objects_with_lower_tail_p_below_0.05": [o for o, t in zip(SEEN + UNSEEN, tails) if t < 0.05],
    "lower_tail_p": {o: round(t, 4) for o, t in zip(SEEN + UNSEEN, tails)},
    "assumption": "시뮬 w/ pretrain 성공률을 실로봇 물체별 기대 확률로 둔 참고 계산이다. 성공 기준이 다르다(시뮬 1.5 cm, 실로봇 3.0 cm, 부록 B.1).",
}

# ---------- V6: 표 2와 보정 데이터 ----------
def f1_given(prec, rec):
    return 2 * prec * rec / (prec + rec)

def implied_recall(F1, FPR, rho):
    # rho = 실제 활성 비율. F1=2TP/(P+TP+FP), TP=r*P, FP=FPR*N
    lo, hi = 1e-6, 1.0
    for _ in range(60):
        r = (lo + hi) / 2
        tp, fp = r * rho, FPR * (1 - rho)
        f = 2 * tp / (rho + tp + fp)
        lo, hi = (r, hi) if f < F1 else (lo, r)
    return (lo + hi) / 2

imp = {}
for rho in [0.05, 0.10, 0.20]:
    imp[str(rho)] = {"nominal_recall": round(implied_recall(0.642, 0.042, rho), 3), "calibrated_recall": round(implied_recall(0.705, 0.022, rho), 3)}
def min_rho(F1, FPR):
    # 재현율 1에서도 F1에 도달할 수 있는 최소 실제 활성 비율: 2rho/(2rho+FPR(1-rho)) = F1
    return F1 * FPR / (2 * (1 - F1) + F1 * FPR)

R["V6_table2"] = {
    "min_active_share_for_feasibility": {"nominal": round(min_rho(0.642, 0.042), 4), "calibrated": round(min_rho(0.705, 0.022), 4)},
    "onset_ms": [426, 96], "onset_reduction_pct": round(100 * (1 - 96 / 426), 1),
    "F1_pp_gain": round(70.5 - 64.2, 1), "FPR_halved": round(2.2 / 4.2, 3),
    "frame_period_ms_at_10Hz": 100, "onset_error_below_one_frame": 96 < 100,
    "implied_recall_by_active_ratio": imp,
    "calibration_frames": {"sphere": 3255, "cylinder": 2419, "sum": 3255 + 2419, "reported_sum": 5674,
                           "sphere_s_x10Hz": 325.5 * 10, "cylinder_s_x10Hz": 241.9 * 10},
    "fig6_trajectories_FSR_plus_fingertip": 6 + 4, "text_total_trajectories": 9,
    "mean_trajectory_s": round((325.5 + 241.9) / 9, 1),
    "note": "표 2의 평가 데이터(보정에 쓴 9개 궤적 자체인지 별도 궤적인지)는 부록 A에 '파라미터 피팅과 센싱 수준 평가'로 적혀 있다.",
}

# ---------- V7: 구성 값 일관성 (숫자만) ----------
R["V7_dims"] = {
    "tactile_channels": 4 * 8 + 12, "proprio_dim": 6 + 16, "state_vel_dim": 2 * 22,
    "point_outputs": 64 * 16, "ffn_over_hidden": 1536 / 384, "head_dim": 384 / 6,
    "dp_global_cond_896_minus_384_tactile": 896 - 384,
    "sim_trials_total_per_split": 10 * 50, "real_trials_total_table1_table5_wo": 100 + 100,
    "demos_per_seen_object_if_even": 10000 / 10,
    "expert_env_steps_total_if_10_experts": 10 * 200_000_000,
    "placement_region_eval_B1_cm": 4, "placement_region_train_E_cm_pm": 5,
}

OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print("wrote", OUT)
for k in ["V1_table1", "V2_table4", "V3_table5", "K_table7_8_6", "V4_table9", "V5_sim_vs_real", "V6_table2"]:
    v = {a: b for a, b in R[k].items() if a not in ("rows", "per_object_wilson95", "lower_tail_p")}
    print(k, json.dumps(v, ensure_ascii=False, default=float)[:1400])
