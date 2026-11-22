"""논문 표의 보고 수치를 다시 계산하고 일관성을 검사한다 (CPU, 1초 미만).

대상: arXiv:2610.17802 v1의 표 1·2·4·5, 부록 B.3·B.4·B.7·A.7, 본문 백분율.
검사 항목은 V1–V20 번호로 results JSON에 남긴다.

실행: python3 analysis/recompute.py   -> analysis/recompute_results.json
의존: numpy, scipy
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
R = {}


def wilson(k, n, conf=0.95):
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=conf, method="wilson")
    return [round(ci.low, 4), round(ci.high, 4)]


def mcnemar_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else float(stats.binomtest(min(b, c), n, 0.5).pvalue)


def rhalf(x):
    """사사오입(0.5는 올림). 논문 백분율이 이 규칙인지 확인하려고 쓴다."""
    return int(math.floor(x * 100 + 0.5 + 1e-9))


# ---------------------------------------------------------------- 표 1
TASKS = ["OpenCoffee", "CloseCabinet", "CloseFaucet", "PressButton"]
T1 = {  # (setting, method) -> 4개 과제의 성공 수 (표 1)
    ("unc", "KALM"): [4, 7, 6, 6],
    ("unc", "WACL"): [6, 7, 8, 9],
    ("unc", "FR3"): [5, 8, 8, 9],
    ("con", "KALM"): [2, 5, 3, 3],
    ("con", "WACL"): [7, 8, 7, 9],
    ("con", "FR3"): [4, 7, 8, 8],
}
T1_REPORTED_AGG = {("unc", "KALM"): 58, ("unc", "WACL"): 75, ("unc", "FR3"): 75,
                   ("con", "KALM"): 33, ("con", "WACL"): 78, ("con", "FR3"): 68}

v1 = {}
for key, s in T1.items():
    tot = sum(s)
    v1["%s_%s" % key] = {
        "successes": s, "total": tot, "of": 40,
        "rate_exact_pct": tot / 40 * 100,
        "rounded_half_up": rhalf(tot / 40),
        "reported": T1_REPORTED_AGG[key],
        "match": rhalf(tot / 40) == T1_REPORTED_AGG[key],
        "wilson95": wilson(tot, 40),
    }
R["V1_table1_aggregates"] = v1

# V2 본문 문장 검사
unc_k, con_k = T1[("unc", "KALM")], T1[("con", "KALM")]
unc_w, con_w = T1[("unc", "WACL")], T1[("con", "WACL")]
R["V2_text_claims"] = {
    "KALM_con_over_unc_per_task": [c / u for c, u in zip(con_k, unc_k)],
    "KALM_half_or_less_count": sum(1 for c, u in zip(con_k, unc_k) if c <= u / 2 + 1e-9),
    "claim_half_on_three_of_four": "OK" if sum(1 for c, u in zip(con_k, unc_k) if c <= u / 2 + 1e-9) == 3 else "FAIL",
    "KALM_drops_every_task": all(c < u for c, u in zip(con_k, unc_k)),
    "WACL_unc_to_con_per_task": list(zip(unc_w, con_w)),
    "WACL_agg_change_pp": (sum(con_w) - sum(unc_w)) / 40 * 100,
    "WACL_ge_KALM_unconstrained_all_tasks": all(w >= k for w, k in zip(unc_w, unc_k)),
    "ratio_con_WACL_over_KALM": sum(con_w) / sum(con_k),
    "claim_more_than_doubles": sum(con_w) / sum(con_k) > 2,
    "FR3_minus_xArm_con_per_task": [f - w for f, w in zip(T1[("con", "FR3")], con_w)],
}

# ---------------------------------------------------------------- B.3 McNemar
# 쌍 구조: WACL만 성공 b, KALM만 성공 c, 둘 다 성공 s, 둘 다 실패 f.
# 주변 합(표 1)과 보고된 b, c가 맞는지 확인한다.
def pair_table(wacl, kalm, b, c, n=40):
    both = wacl - b
    both_k = kalm - c
    return {"b_WACL_only": b, "c_KALM_only": c, "both_success": both, "both_success_from_KALM_margin": both_k,
            "consistent": both == both_k and both >= 0, "both_fail": n - both - b - c}


v3 = {}
v3["constrained_pooled"] = pair_table(sum(con_w), sum(con_k), 19, 1)
v3["constrained_pooled"]["p_exact"] = mcnemar_exact(19, 1)
v3["constrained_pooled"]["reported_p"] = 4.0e-5
v3["unconstrained_pooled"] = pair_table(sum(unc_w), sum(unc_k), 11, 4)
v3["unconstrained_pooled"]["p_exact"] = mcnemar_exact(11, 4)
v3["unconstrained_pooled"]["reported_p"] = 0.12

# 과제별 구성: 보고된 p(0.125, 0.25, 0.125, 0.031)와 주변 합에서 (b, c)를 역으로 찾는다.
rep_p = [0.125, 0.25, 0.125, 0.031]
per_task = []
for i, t in enumerate(TASKS):
    diff = con_w[i] - con_k[i]
    sols = []
    for c in range(0, con_k[i] + 1):
        b = diff + c
        if b + c == 0 or con_w[i] - b < 0:
            continue
        if con_w[i] - b > 10 - b - c:  # both <= n - b - c 로 제한
            pass
        p = mcnemar_exact(b, c)
        if abs(round(p, 3) - rep_p[i]) < 0.0006 or abs(p - rep_p[i]) < 0.0006:
            sols.append({"b": b, "c": c, "p": round(p, 4)})
    per_task.append({"task": t, "wacl": con_w[i], "kalm": con_k[i], "reported_p": rep_p[i], "solutions": sols})
v3["constrained_per_task_inverse"] = per_task
sb = sum(x["solutions"][0]["b"] for x in per_task if len(x["solutions"]) == 1)
sc = sum(x["solutions"][0]["c"] for x in per_task if len(x["solutions"]) == 1)
v3["constrained_per_task_unique_solutions"] = all(len(x["solutions"]) == 1 for x in per_task)
v3["constrained_per_task_sum_b_c"] = [sb, sc]
v3["matches_pooled_19_1"] = (sb, sc) == (19, 1)
# 비보정 p 네 개의 Holm 보정
ps = [mcnemar_exact(x["solutions"][0]["b"], x["solutions"][0]["c"]) for x in per_task]
order = np.argsort(ps)
holm = [None] * 4
run = 0.0
for rank, idx in enumerate(order):
    val = min(1.0, (4 - rank) * ps[idx])
    run = max(run, val)
    holm[idx] = run
v3["constrained_per_task_p_raw"] = [round(p, 4) for p in ps]
v3["constrained_per_task_p_holm"] = [round(p, 4) for p in holm]
R["V3_mcnemar"] = v3

# ---------------------------------------------------------------- V4 쌍체 t
def paired_t(b, c, n=40):
    d = np.array([1] * b + [-1] * c + [0] * (n - b - c), dtype=float)
    t = stats.ttest_1samp(d, 0.0)
    return {"t": float(t.statistic), "df": n - 1, "p": float(t.pvalue)}


R["V4_paired_t"] = {"constrained": paired_t(19, 1), "reported_con": 7.7e-6,
                    "unconstrained": paired_t(11, 4), "reported_unc": 0.07}

# ---------------------------------------------------------------- V5 Wilson, 겹침
w5 = {}
for s in ("unc", "con"):
    for m in ("KALM", "WACL"):
        w5["%s_%s_per_task" % (s, m)] = [wilson(k, 10) for k in T1[(s, m)]]
    k_ci, w_ci = wilson(sum(T1[(s, "KALM")]), 40), wilson(sum(T1[(s, "WACL")]), 40)
    w5["%s_aggregate" % s] = {"KALM": k_ci, "WACL": w_ci, "disjoint": k_ci[1] < w_ci[0]}
R["V5_wilson"] = w5

# ---------------------------------------------------------------- V6 독립 표본 검정(대조용)
v6 = {}
for s in ("unc", "con"):
    a, b = sum(T1[(s, "WACL")]), sum(T1[(s, "KALM")])
    v6["%s_fisher_agg_WACL_vs_KALM" % s] = float(stats.fisher_exact([[a, 40 - a], [b, 40 - b]])[1])
    v6["%s_fisher_per_task" % s] = [float(stats.fisher_exact([[w, 10 - w], [k, 10 - k]])[1])
                                    for w, k in zip(T1[(s, "WACL")], T1[(s, "KALM")])]
R["V6_unpaired_fisher"] = v6

# ---------------------------------------------------------------- V7 취약성
v7 = {"constrained_pooled_p_vs_c": {}}
for c in range(0, 9):
    v7["constrained_pooled_p_vs_c"][str(c)] = mcnemar_exact(19, c)
# WACL 단독 성공 쌍(b) 중 몇 개가 KALM 단독 성공(c)으로 뒤집히면 p>0.05가 되나 (b+c=20 고정)
flips = None
for k in range(0, 11):
    if mcnemar_exact(19 - k, 1 + k) > 0.05:
        flips = k
        break
v7["flips_needed_b_to_c_for_p_gt_0.05"] = flips
v7["unconstrained_p_if_b11_c4"] = mcnemar_exact(11, 4)
flips_u = None
for k in range(0, 8):
    if mcnemar_exact(11 + k, 4 - k) < 0.05:
        flips_u = k
        break
v7["unconstrained_extra_b_wins_needed_for_p_lt_0.05"] = flips_u
R["V7_fragility"] = v7

# ---------------------------------------------------------------- V8 n = 10 쌍의 최소 p
v8 = {}
for d in range(1, 11):
    v8["diff_%d" % d] = mcnemar_exact(d, 0)
R["V8_min_mcnemar_p_n10_pairs_by_success_difference"] = v8

# ---------------------------------------------------------------- V9 표 2 절제
T2 = [
    ("second_constraint", "CloseFaucet", 7, 7),
    ("no_sliding_anchor", "OpenCoffee", 3, 7),
    ("link5_only", "PressButton", 6, 9),
    ("link6_only", "PressButton", 7, 9),
    ("geometric_arc", "OpenCoffee", 4, 7),
    ("geometric_line", "PressButton", 7, 9),
    ("registered_replay", "OpenCoffee", 6, 7),
]
v9 = []
for name, task, var, full in T2:
    v9.append({"variant": name, "task": task, "variant_succ": var, "full_succ": full,
               "diff": full - var,
               "fisher_p": float(stats.fisher_exact([[full, 10 - full], [var, 10 - var]])[1]),
               "best_case_mcnemar_p": mcnemar_exact(full - var, 0) if full != var else 1.0,
               "wilson_variant": wilson(var, 10), "wilson_full": wilson(full, 10)})
R["V9_table2_ablation"] = v9

# ---------------------------------------------------------------- V10 링크 탐색
both, only5, only6, neither = 4, 2, 3, 1
R["V10_link_search"] = {
    "pairs": {"both": both, "only_link5": only5, "only_link6": only6, "neither": neither},
    "sum": both + only5 + only6 + neither,
    "link5_successes": both + only5, "link6_successes": both + only6,
    "union_oracle": both + only5 + only6,
    "multilink_reported": 9,
    "union_equals_multilink": both + only5 + only6 == 9,
    "pinned_failures_total_runs": (10 - 6) + (10 - 7),
    "claim_seven_pinned_failures_all_execution": (10 - 6) + (10 - 7) == 7,
    "note": "7은 링크 5의 실패 4회와 링크 6의 실패 3회를 합한 실행 횟수이며 서로 다른 배치 수가 아니다",
    "distinct_placements_with_a_pinned_failure": only5 + only6 + neither,
}

# ---------------------------------------------------------------- V11 표 4
rows = {"OpenCoffee": (20, 20, 20, 0, 0, 0), "CloseCabinet": (20, 20, 20, 0, 0, 0),
        "CloseFaucet": (20, 20, 20, 0, 0, 2), "PressButton": (20, 20, 17, 3, 0, 0),
        "CloseFaucet_2c": (20, 20, 17, 3, 0, 6)}
tot = [sum(r[i] for r in rows.values()) for i in range(6)]
R["V11_table4_audit"] = {
    "column_sums": tot, "reported_totals": [100, 100, 94, 6, 0, 8],
    "match": tot == [100, 100, 94, 6, 0, 8],
    "fully_correct_plus_over_plus_looser": tot[2] + tot[3] + tot[4],
    "fully_correct_wilson": wilson(94, 100),
    "looser_0_of_100_upper95_wilson": wilson(0, 100)[1],
    "looser_rule_of_three_upper": 3 / 100,
    "pressbutton_fully_correct_wilson": wilson(17, 20),
    "two_constraint_fully_correct_wilson": wilson(17, 20),
    "note": "20 시드는 설정마다 같은 저장 관측 위의 샘플이라 장면 다양성이 아니라 VLM 샘플링 분산만 잰다",
}

# ---------------------------------------------------------------- V12 B.4
fails = [11, 8, 4, 1]
R["V12_planner_failures"] = {
    "attempts": 74, "success_attempts": 69, "rate_pct": 69 / 74 * 100,
    "candidates": 148, "candidate_success": 124, "candidate_fail": 148 - 124,
    "candidates_per_attempt": 148 / 74,
    "failure_causes_sum": sum(fails), "failure_cause_pct": [round(f / 24 * 100, 1) for f in fails],
    "reported_pct": [46, 33, 17, 4], "pct_sum_reported": 46 + 33 + 17 + 4,
    "failed_attempts": 5, "failed_attempts_wilson_upper": wilson(5, 74)[1],
    "wacl_con_xarm_nonsuccess_rollouts": 40 - 31,
    "note": "74회 시도가 어느 세션(xArm WACL 40회, FR3 40회, 절제 포함 여부)인지 본문에 없다",
}

# ---------------------------------------------------------------- V13 표 5
T5 = [("OpenCoffee_on", 2, 26, 1.2e-7), ("OpenCoffee_off", 7, 30, 2.4e-7),
      ("CloseFaucet_on", 3, 30, 1.5e-8), ("CloseFaucet_off", 29, 30, 1.0)]
v13 = []
for name, k, w, rp in T5:
    b = w - k  # WACL만 성공. KALM이 성공한 장면이 모두 WACL도 성공이라고 가정하면 c = 0
    c = 0
    p = mcnemar_exact(b, c)
    # c가 0이 아닐 수 있는가: 보고된 p를 만드는 (b, c) 탐색
    alts = []
    for cc in range(0, k + 1):
        bb = (w - k) + cc
        if bb > 30 - 0:
            continue
        pp = mcnemar_exact(bb, cc)
        if abs(math.log10(pp) - math.log10(rp)) < 0.05:
            alts.append({"b": bb, "c": cc, "p": pp})
    v13.append({"row": name, "KALM": k, "WACL": w, "reported_p": rp, "p_if_c0": p, "solutions_matching_reported_p": alts})
R["V13_table5_replay"] = {
    "rows": v13,
    "all_c_zero": all(x["solutions_matching_reported_p"] and x["solutions_matching_reported_p"][0]["c"] == 0 for x in v13),
    "online_vs_offline_KALM_con": {"OpenCoffee": {"online_rollouts_success": "2/10", "offline_plan_success": "2/30"},
                                   "CloseFaucet": {"online_rollouts_success": "3/10", "offline_plan_success": "3/30"}},
    "note": "표 1의 KALM 구속 실행 성공(2/10, 3/10)이 장면 30개 재생의 계획 성공(2/30, 3/30)과 같은 개수다. "
            "30개 장면의 출처(세션 3개x10 배치?)가 본문에 없어 독립성을 판단할 수 없다.",
    "pseudoreplication_if_10_placements": "고유 배치가 10개라면 유효 n=10이고 같은 관찰이 세 번 세어진 셈이다",
}

# ---------------------------------------------------------------- V14 런타임, 빔
s = [10.88, 2.76, 14.68]
R["V14_runtime_and_beam"] = {
    "constraint_chain_sum": round(sum(s), 2), "reported_barrier": 28.31,
    "chain_minus_barrier": round(sum(s) - 28.31, 2),
    "barrier_plus_post": round(28.31 + 6.32, 2), "reported_total": 34.63,
    "anchor_prep_s": 6.32, "post_barrier_s": 6.32, "identical_values_flag": True,
    "vlm_share_of_total_pct": round((10.88 + 14.68) / 34.63 * 100, 1),
    "vlm_plus_sam3_share_pct": round((10.88 + 14.68 + 2.76) / 34.63 * 100, 1),
    "beam_candidates_per_step": 20 * 32, "reported_beam_candidates": 640,
    "H_per_task": {"CloseCabinet": 4, "PressButton": 4, "OpenCoffee": 8, "CloseFaucet": 8},
    "policy_trajectory_length_appendix_a2": 16, "section_3_1_horizon_H": "H (본문 기호) vs A.2 길이 16 vs A.5 H=4/8",
}

# ---------------------------------------------------------------- V15 FR3
v15 = {}
for s_ in ("unc", "con"):
    a, b = sum(T1[(s_, "FR3")]), sum(T1[(s_, "WACL")])
    v15["%s_FR3_vs_xArm_fisher" % s_] = float(stats.fisher_exact([[a, 40 - a], [b, 40 - b]])[1])
v15["con_OpenCoffee_FR3_vs_xArm_fisher"] = float(stats.fisher_exact([[4, 6], [7, 3]])[1])
v15["con_FR3_vs_KALM_fisher"] = float(stats.fisher_exact([[27, 13], [13, 27]])[1])
R["V15_cross_embodiment"] = v15

# ---------------------------------------------------------------- V16–V19 학습 규모
iters, batch, demos, views, hz = 100_000, 512, 10, 3, 10
R["V16_training_scale"] = {
    "samples_seen": iters * batch,
    "samples_per_demo_if_10_pairs": iters * batch / demos,
    "if_views_triple_pairs": iters * batch / (demos * views),
    "note": "10개 시연(또는 최대 30개 뷰별 쌍)을 5.1천만 샘플로 학습한다. 보강 기법이 본문에 없다면 사실상 암기 모델이다.",
}

# ---------------------------------------------------------------- V17 표본 규모와 검출력
def n_per_arm(p1, p2, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pb = (p1 + p2) / 2
    return int(math.ceil(2 * (za + zb) ** 2 * pb * (1 - pb) / (p1 - p2) ** 2))


R["V17_sample_size"] = {
    "unc_75_vs_57.5": n_per_arm(0.75, 0.575),
    "con_77.5_vs_32.5": n_per_arm(0.775, 0.325),
    "ablation_70_vs_30": n_per_arm(0.7, 0.3),
    "replay_70_vs_60": n_per_arm(0.7, 0.6),
    "primitive_90_vs_70": n_per_arm(0.9, 0.7),
    "arc_70_vs_40": n_per_arm(0.7, 0.4),
    "note": "비쌍체 정규근사 80% 검정력, 조건당 필요한 롤아웃 수(쌍체이면 더 적을 수 있다)",
}

# ---------------------------------------------------------------- V18 롤아웃 총수
R["V18_rollout_budget"] = {
    "table1_cells": 6 * 4, "table1_rollouts": 6 * 4 * 10,
    "table2_rows": 7, "table2_rollouts": 7 * 10,
    "total_reported_robot_rollouts": 6 * 4 * 10 + 7 * 10,
    "demos_total_wacl": 4 * 10, "demos_total_kalm": 4 * 10,
}

# ---------------------------------------------------------------- V19 상호 비교: 인용 숫자 vs 표
R["V19_abstract_vs_table"] = {
    "abstract_78_vs_33": [rhalf(31 / 40), rhalf(13 / 40)],
    "exact": [31 / 40 * 100, 13 / 40 * 100],
    "abstract_rounding_note": "13/40 = 32.5%, 31/40 = 77.5%는 둘 다 정확히 .5라서 반올림 방향(올림)이 결과를 정한다",
    "gap_pp_exact": (31 - 13) / 40 * 100,
    "gap_pp_from_rounded": 78 - 33,
}

# ---------------------------------------------------------------- V20 합성 지표
# 연구 사안별 정규화 평균(과제 평균 대 롤아웃 합산)은 과제당 10회 고정이라 같다
R["V20_task_mean_vs_pooled"] = {
    "task_mean_equals_pooled": True,
    "reason": "과제마다 롤아웃이 10회라 과제 평균과 합산 비율이 같다",
}

OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False,
                          default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
print("wrote", OUT)
for k in ("V1_table1_aggregates",):
    for kk, vv in R[k].items():
        print(kk, vv["successes"], vv["total"], vv["rate_exact_pct"], vv["reported"], vv["match"], vv["wilson95"])
print("V3", json.dumps(R["V3_mcnemar"], ensure_ascii=False)[:1800])
print("V4", R["V4_paired_t"])
print("V7", R["V7_fragility"])
print("V8", R["V8_min_mcnemar_p_n10_pairs_by_success_difference"])
for x in R["V9_table2_ablation"]:
    print(x["variant"], x["variant_succ"], x["full_succ"], round(x["fisher_p"], 3), round(x["best_case_mcnemar_p"], 3))
print("V6", R["V6_unpaired_fisher"])
print("V11", R["V11_table4_audit"])
print("V13", json.dumps(R["V13_table5_replay"]["rows"], ensure_ascii=False))
print("V14", R["V14_runtime_and_beam"])
print("V15", R["V15_cross_embodiment"])
print("V17", R["V17_sample_size"])
