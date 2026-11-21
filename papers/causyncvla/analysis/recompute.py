"""CauSyncVLA (arXiv:2611.25667 v1) 논문 표 수치를 다시 계산하고 통계 검정을 한다.

입력: 논문 본문과 표의 값을 아래에 그대로 옮겨 적었다(출처는 주석, paper/paper.txt 기준).
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초. 외부 데이터·가중치 사용 안 함)

가정(논문에 없는 것은 표시한다)
- 성공률의 분모는 표 캡션의 시도 수다(LIBERO 500 starts, 실로봇 과제당 23회 × 3과제 = 69회 per arm).
- 쌍 비교(McNemar)는 Table 10의 wins/losses만 쓸 수 있다. 공개되지 않은 쌍 비교(Table 1 native 469 vs 456,
  Table 4, Table 15)는 불연속 두 비율 검정으로 근사하고, 그 사실을 결과에 표시한다.
- Table 2(시간 고정 20 Hz, 100 starts)는 같은 시작 상태를 쓰지만 쌍 별 결과가 없어 비쌍 검정으로만 본다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe_diff(k1, n1, k2, n2):
    """Newcombe(1998) 방법 10: 두 비율 차이(p1 - p2)의 Wilson 기반 95% 구간."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def mcnemar_exact(w, l):
    """논문 식 (24): 불일치 쌍 w + l에 대한 양측 정확 검정."""
    nd = w + l
    if nd == 0:
        return 1.0
    k = min(w, l)
    p = 2 * sum(math.comb(nd, j) * 0.5 ** nd for j in range(k + 1))
    return min(1.0, p)


def paired_diff_ci(w, l, n, alpha=0.05):
    """쌍 차이 (w-l)/n의 근사 95% 구간. 불일치 쌍 중 w의 비율에 Clopper-Pearson을 쓰고 d/n을 곱한다."""
    d = w + l
    if d == 0:
        return 0.0, 0.0, 0.0
    lo_q = stats.beta.ppf(alpha / 2, w, d - w + 1) if w > 0 else 0.0
    hi_q = stats.beta.ppf(1 - alpha / 2, w + 1, d - w) if w < d else 1.0
    return (w - l) / n, d / n * (2 * lo_q - 1), d / n * (2 * hi_q - 1)


def fisher(k1, n1, k2, n2):
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(stats.fisher_exact(table, alternative="two-sided")[1])


def chi2_p(k1, n1, k2, n2):
    table = np.array([[k1, n1 - k1], [k2, n2 - k2]])
    return float(stats.chi2_contingency(table, correction=False)[1])


# ---------------------------------------------------------------- 1. LIBERO-10 native (Table 1, 9)
# Table 9 합계: π0.5 10-step 456/500, CauSyncVLA 469/500 (같은 500 starts, seed 7, 한 실행씩)
per_task_base = [46, 50, 47, 50, 48, 43, 45, 49, 29, 49]
per_task_cs = [49, 50, 49, 47, 48, 44, 46, 50, 38, 48]
K_CS, K_BASE, N = 469, 456, 500
assert sum(per_task_base) == K_BASE and sum(per_task_cs) == K_CS
d, lo, hi = newcombe_diff(K_CS, N, K_BASE, N)
RES["libero_native"] = {
    "cauSync": K_CS, "pi05_10step": K_BASE, "n": N,
    "pct_cauSync": round(100 * K_CS / N, 2), "pct_base": round(100 * K_BASE / N, 2),
    "wilson_cauSync_pct": [round(100 * x, 2) for x in wilson(K_CS, N)],
    "wilson_base_pct": [round(100 * x, 2) for x in wilson(K_BASE, N)],
    "diff_pp": round(100 * d, 2), "newcombe_diff_pp": [round(100 * lo, 2), round(100 * hi, 2)],
    "fisher_unpaired_p": round(fisher(K_CS, N, K_BASE, N), 4),
    "chi2_unpaired_p": round(chi2_p(K_CS, N, K_BASE, N), 4),
    # 쌍 검정 불가: 불일치 쌍 w, l이 공개되지 않는다. w - l = 13만 안다.
    "paired_mcnemar_bounds": {
        "note": "w-l=13만 알려져 있어 쌍 검정은 계산할 수 없다. l(상대만 성공한 start 수)에 따라 p가 달라진다.",
        **{f"l={l_}": float(f"{mcnemar_exact(l_ + 13, l_):.3g}") for l_ in [0, 5, 10, 20, 40, 80]},
    },
    "per_task_counts": {"pi05": per_task_base, "cauSync": per_task_cs},
}

# ---------------------------------------------------------------- 2. Matched ablations (Table 10, 5, 12)
# (w, l) = Table 10의 wins/losses. 논문 Table 10의 'Gain' 및 Exact p와 대조한다.
pairs = {
    "memory_prediction (w/ vs w/o latent WM)": (80, 14, 467, 401, "2.12e-12", 13.20),
    "PFWS, full controller (w/ vs w/o PFWS)": (24, 17, 467, 460, "0.3489", 1.40),
    "action distillation (w/ vs w/o action loss)": (54, 18, 464, 428, "2.57e-5", 7.20),
    "PFWS, frozen policy (vs cold)": (26, 15, 465, 454, "0.1173", 2.20),
}
ablation = {}
for name, (w, l_, k_full, k_ref, paper_p, paper_gain) in pairs.items():
    p = mcnemar_exact(w, l_)
    gain, glo, ghi = paired_diff_ci(w, l_, N)
    ablation[name] = {
        "wins": w, "losses": l_, "k_full": k_full, "k_ref": k_ref,
        "check_ref_from_wins_losses": k_full - (w - l_),
        "gain_pp": round(100 * gain, 2), "paper_gain_pp": paper_gain,
        "gain_ci95_pp_clopper_pearson_on_discordant": [round(100 * glo, 2), round(100 * ghi, 2)],
        "exact_mcnemar_p": float(f"{p:.3g}"), "paper_p": paper_p,
        "bonferroni_4_significant": bool(p < 0.05 / 4),
    }
RES["ablation_pairs"] = ablation

# Table 5 하단: w/ vs w/o action loss 464 vs 428 (Table 10 과 같은 run)
RES["consistency_table5_vs_table10"] = {
    "full_native_run_469_vs_table5_full_467": "Table 5 상단 Full 93.40 = 467/500 (별도 실행), native 93.80 = 469/500",
    "action_loss_464_minus_428_equals_54_minus_18": (464 - 428) == (54 - 18),
    "latent_wm_467_minus_401_equals_80_minus_14": (467 - 401) == (80 - 14),
    "pfws_467_minus_460_equals_24_minus_17": (467 - 460) == (24 - 17),
    "frozen_465_minus_454_equals_26_minus_15": (465 - 454) == (26 - 15),
}

# ---------------------------------------------------------------- 3. Table 2 timed 20 Hz (100 starts)
t2 = {"pi05": 86, "RTC": 82, "VLA-Rolling": 87, "CauSyncVLA": 91}
RES["table2_timed"] = {
    "cauSync_vs_pi05_fisher_p": round(fisher(91, 100, 86, 100), 4),
    "cauSync_vs_pi05_diff_pp": 5,
    "cauSync_vs_pi05_newcombe_pp": [round(100 * x, 2) for x in newcombe_diff(91, 100, 86, 100)[1:]],
    "cauSync_vs_rolling_fisher_p": round(fisher(91, 100, 87, 100), 4),
    "response_ratio_p50": round(176.73 / 83.04, 3),
    "replans_ratio": round(3.28 / 2.57, 3),
    "actions_ratio": round(16.60 / 13.02, 3),
}

# ---------------------------------------------------------------- 4. Real robot FR3 (Table 6, App. C.9)
task_base = {"stacking": 17, "placement": 16, "sorting": 17}  # 17/23, 16/23, 17/23 (Table 6 %와 C.9 분모 확인)
task_cs = {"stacking": 19, "placement": 22, "sorting": 19}
n_task = 23
assert sum(task_base.values()) == 50 and sum(task_cs.values()) == 60
K_B, K_C, N_P = 50, 60, 69
d, lo, hi = newcombe_diff(K_C, N_P, K_B, N_P)
# Cochran-Mantel-Haenszel (층 = 과제, 층별 2x2 표) 카이제곱, 연속성 보정 없음
num, var = 0.0, 0.0
for t in task_base:
    a, b = task_cs[t], n_task - task_cs[t]
    c, dd = task_base[t], n_task - task_base[t]
    n_ = a + b + c + dd
    r1, r2 = a + b, c + dd
    c1, c2 = a + c, b + dd
    num += a - r1 * c1 / n_
    var += r1 * r2 * c1 * c2 / (n_ ** 2 * (n_ - 1))
cmh_chi2 = num ** 2 / var
# 민감도: CauSync 성공 중 k개를 실패로 바꾸면(예: 운영자 판정 재검토) 풀링 Fisher p가 0.05를 넘는 최소 k
sens_fisher = next((k for k in range(0, 61) if fisher(K_C - k, N_P, K_B, N_P) > 0.05), None)
sens_chi2 = next((k for k in range(0, 61) if chi2_p(K_C - k, N_P, K_B, N_P) > 0.05), None)
RES["real_robot"] = {
    "pooled_cauSync": f"{K_C}/{N_P}", "pooled_pi05": f"{K_B}/{N_P}",
    "pct_cauSync": round(100 * K_C / N_P, 2), "pct_pi05": round(100 * K_B / N_P, 2),
    "wilson_cauSync_pct": [round(100 * x, 2) for x in wilson(K_C, N_P)],
    "wilson_pi05_pct": [round(100 * x, 2) for x in wilson(K_B, N_P)],
    "paper_wilson_cauSync_pct": [77.03, 92.98], "paper_wilson_pi05_pct": [60.95, 81.61],
    "diff_pp": round(100 * d, 2), "newcombe_diff_pp": [round(100 * lo, 2), round(100 * hi, 2)],
    "fisher_unpaired_p": round(fisher(K_C, N_P, K_B, N_P), 4),
    "chi2_unpaired_p": round(chi2_p(K_C, N_P, K_B, N_P), 4),
    "cmh_chi2_by_task": round(float(cmh_chi2), 3),
    "cmh_p": round(float(stats.chi2.sf(cmh_chi2, 1)), 4),
    "per_task_fisher_p": {t: round(fisher(task_cs[t], n_task, task_base[t], n_task), 4) for t in task_base},
    "per_task_counts": {"pi05": task_base, "cauSync": task_cs, "n_per_task": n_task},
    "fisher_p_already_above_05": fisher(K_C, N_P, K_B, N_P) > 0.05,
    "min_cauSync_relabel_to_lose_p05_chi2": sens_chi2,
    "min_cauSync_relabel_to_lose_p05_fisher": sens_fisher,
    "execution_errors": {"pi05_runs": 15, "cauSync_runs": 5},
}

# ---------------------------------------------------------------- 5. Latency / throughput ratios (Table 1, 2, 16)
RES["latency_ratios"] = {
    "table16_10step_over_shared_warm (5.04 claimed)": round(60.01 / 11.91, 3),
    "table16_10step_over_1step (decomp part 1)": round(60.01 / 36.00, 3),
    "table16_1step_over_shared_warm (decomp part 2)": round(36.00 / 11.91, 3),
    "product_of_parts": round((60.01 / 36.00) * (36.00 / 11.91), 3),
    "table16_10step_over_isolated (12.52 claimed)": round(60.01 / 4.79, 3),
    "table1_plans_per_s_ratio (5.04 claimed)": round(83.96 / 16.66, 3),
    "table1_actions_per_s_check_5x_plans": round(83.96 * 5, 2),
    "table2_response_p50_ratio": round(176.73 / 83.04, 3),
}

# ---------------------------------------------------------------- 6. Table 3, 4, 11, 13 internal checks
RES["table3_ratios"] = {
    "action_mse_stale_over_full (10.5 claimed)": round(0.10355 / 0.00983, 2),
    "action_mse_age_endpoint_over_shuffled (4x claimed)": round(0.04602 / 0.01107, 2),
    "kv_error_full_lowest": 0.31958 == min(0.42430, 0.37217, 0.35669, 0.32037, 0.31958),
}
RES["table4_flips"] = {
    "boundary_flips_rel_change_pct": round(100 * (1.36 / 2.57 - 1), 1),
    "boundary_excess_w_o": round(2.57 - 1.69, 2), "boundary_excess_w": round(1.36 - 1.39, 2),
    "note": "Table 4는 비율만 있어 쌍 검정 불가. 'nearly halving'은 -47%.",
}
RES["checkpoint_selection"] = {
    "weighted_action_mse_by_milestone": {"6k": 0.00807, "12k": 0.00686, "18k": 0.00648},
    "argmin_weighted_val": "18k",
    "reported_checkpoint": "12k",
    "note": "config의 selection-metric은 val/action_weighted(최소)인데 논문은 12k를 보고한다. 12k는 native 500 starts에서 고른 것으로 보인다(Fig. 8b, App. A).",
}
t13 = [(105, 0.1553, 0.1356, 0.0171, 0.0025), (126, 0.3953, 0.3116, 0.0702, 0.0072),
       (40, 0.5856, 0.4400, 0.1125, 0.0121), (21, 0.9364, 0.6357, 0.3208, 0.0284),
       (21, 0.9037, 0.6083, 0.3253, 0.0373), (7, 1.0858, 0.7205, 0.6330, 0.0161)]
RES["table13"] = {
    "pairs_sum (320 expected)": sum(r[0] for r in t13),
    "action_mse_stale_over_updated_by_bin": [round(r[3] / r[4], 1) for r in t13],
}
t12 = {"w/o latent WM": [42, 46, 29, 48, 44, 45, 40, 48, 22, 37], "w/o PFWS": [47, 50, 50, 50, 50, 43, 44, 48, 31, 47],
       "Sync PFWS": [48, 50, 49, 48, 49, 44, 46, 50, 31, 45], "CauSyncVLA": [49, 49, 48, 48, 50, 46, 45, 50, 34, 48]}
RES["table12_totals"] = {k: sum(v) for k, v in t12.items()}
RES["table12_totals_expected"] = {"w/o latent WM": 401, "w/o PFWS": 460, "Sync PFWS": 460, "CauSyncVLA": 467}

# ---------------------------------------------------------------- 7. LIBERO-Plus (Table 15, 2,519 paired variants)
lp_sync = {"cold1": 1974, "cold10": 1997, "pfws1": 2019, "pfws10": 2005}
lp_cs = {"noPFWS": 1924, "PFWS": 1967}
N_LP = 2519
RES["libero_plus"] = {
    "sync_pfws1_pct": round(100 * 2019 / N_LP, 2), "cauSync_pfws_pct": round(100 * 1967 / N_LP, 2),
    "gap_sync_minus_cauSync_pp": round(100 * (2019 - 1967) / N_LP, 2),
    "gap_unpaired_fisher_p (approx, pairs not given)": round(fisher(2019, N_LP, 1967, N_LP), 4),
    "sync_cold1_pct": round(100 * 1974 / N_LP, 2), "cauSync_noPFWS_pct": round(100 * 1924 / N_LP, 2),
    "gap_cold_pp": round(100 * (1974 - 1924) / N_LP, 2),
    "family_episode_sum_ok": sum([289, 419, 383, 274, 312, 393, 449]) == N_LP,
    "family_cold1_sum_ok": sum([254, 234, 345, 256, 270, 287, 328]) == 1974,
    "family_cauSync_pfws_sum_ok": sum([271, 209, 337, 242, 247, 307, 354]) == 1967,
}

# ---------------------------------------------------------------- 8. Table 6 percent 검산
RES["table6_check"] = {
    "stacking_base_17_of_23_pct": round(100 * 17 / 23, 2), "stacking_cs_19_of_23_pct": round(100 * 19 / 23, 2),
    "placement_base_16_of_23_pct": round(100 * 16 / 23, 2), "placement_cs_22_of_23_pct": round(100 * 22 / 23, 2),
    "sorting_base_17_of_23_pct": round(100 * 17 / 23, 2), "sorting_cs_19_of_23_pct": round(100 * 19 / 23, 2),
    "pooled_50_of_69_pct": round(100 * 50 / 69, 2), "pooled_60_of_69_pct": round(100 * 60 / 69, 2),
}

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(RES, ensure_ascii=False, indent=2))
