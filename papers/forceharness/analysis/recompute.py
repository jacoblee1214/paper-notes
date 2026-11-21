"""논문 표 값의 재계산과 일관성 검사 (ForceHarness, arXiv:2611.21099 v1).

실행: python3 analysis/recompute.py  → analysis/recompute_results.json
입력은 논문 본문과 부록 표에서 옮긴 값이다(출처 표 번호를 주석에 적었다). 에피소드 단위 결과는 논문에 없어서
짝 검정(McNemar)은 공개된 집계 값(저장소 docs/forcebench.md의 시드별 22/6, 20/6)으로만 확인한다.
numpy, scipy만 쓴다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
RES = {}


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(100 * p, 2), round(100 * max(0, c - h), 1), round(100 * min(1, c + h), 1)


def newcombe_diff(k1, n1, k2, n2, z=1.96):
    """독립 두 비율 차이 (k1/n1 - k2/n2)의 Newcombe 점수 구간 (Wilson 구간 두 개를 합침)."""
    p1, p2 = k1 / n1, k2 / n2
    _, l1, u1 = (x / 100 for x in wilson(k1, n1))
    _, l2, u2 = (x / 100 for x in wilson(k2, n2))
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return round(100 * d, 2), round(100 * lo, 1), round(100 * hi, 1)


def fisher(k1, n1, k2, n2):
    _, p = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
    return float(p)


def mcnemar_exact(b, c):
    """짝 자료의 정확 McNemar: b = 처리만 성공, c = 대조만 성공 (양측 이항 검정)."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))


def mcnemar_cc(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return float(stats.chi2.sf((abs(b - c) - 1) ** 2 / n, 1))


# ----------------------------------------------------------------------------- V1 표 3 실로봇
TASKS_REAL = ["wipe", "screw", "peel", "usb", "sort", "soft_hard"]
REAL = {  # (no force, ForceHarness) Astra; Sol ForceHarness, 10 episodes each
    "Astra_no": [1, 0, 1, 0, 0, 0],
    "Astra_fh": [8, 9, 9, 8, 10, 10],
    "Sol_fh": [6, 7, 9, 7, 8, 10],
}
RES["V1_real_sums"] = {k: sum(v) for k, v in REAL.items()}
RES["V1_real_totals_paper"] = {"Astra_no": 2, "Astra_fh": 54, "Sol_fh": 47}
RES["V1_real_wilson"] = {
    "Astra_no_2of60": wilson(2, 60), "Astra_fh_54of60": wilson(54, 60), "Sol_fh_47of60": wilson(47, 60)}
RES["V1_real_fisher_per_task"] = {
    t: round(fisher(a, 10, b, 10), 5) for t, a, b in zip(TASKS_REAL, REAL["Astra_no"], REAL["Astra_fh"])}
RES["V1_real_fisher_total_astra"] = fisher(2, 60, 54, 60)
RES["V1_real_contact4_no"] = sum(REAL["Astra_no"][:4])  # 논문: 40회 중 2회
RES["V1_real_diff_newcombe_astra"] = newcombe_diff(54, 60, 2, 60)

# ----------------------------------------------------------------------------- V2 표 4 ForceBench
FB_TASKS = ["weight", "com", "faucet", "faucet_dir", "stove", "stove_press", "drawer", "nut"]
FB = {  # percent; zero-shot rows, 10 episodes per task (Table 4)
    "Astra_no": [80, 50, 100, 50, 100, 80, 0, 100],
    "Astra_fh": [100, 70, 100, 50, 100, 90, 100, 90],
    "Sol_no": [90, 50, 90, 30, 100, 60, 0, 100],
    "Sol_fh": [100, 60, 100, 40, 100, 100, 100, 100],
    "Fable_fh": [100, 80, 100, 20, 100, 50, 100, 50],
}
FB_MEAN_PAPER = {"Astra_no": 70.00, "Astra_fh": 87.50, "Sol_no": 65.00, "Sol_fh": 87.50, "Fable_fh": 75.00}
TRAINED_PAPER = {  # Table 4 trained rows (mean of eight tasks); ForceBench arXiv:2610.19604 Table I 동일 값
    "pi05_no_force": [54.0, 45.5, 9.5, 31.5, 8.5, 1.0, 4.5, 31.5, 23.25],
    "pi05_prompt_fusion": [88.0, 56.5, 21.0, 41.5, 16.0, 2.5, 23.0, 34.0, 35.31],
    "pi05_dual_path": [99.5, 72.0, 22.0, 30.0, 17.5, 19.0, 4.5, 16.5, 35.13],
    "pi0": [61.5, 39.0, 3.5, 27.0, 7.0, 5.0, 7.5, 36.0, 23.31],
    "ForceVLA": [35.5, 45.5, 12.5, 14.0, 10.0, 12.0, 20.0, 26.0, 21.94],
    "forceflow": [51.5, 3.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 6.88],
}
RES["V2_fb_mean_check"] = {
    k: {"recomputed": round(float(np.mean(v)), 2), "paper": FB_MEAN_PAPER[k]} for k, v in FB.items()}
RES["V2_trained_mean_check"] = {
    k: {"recomputed": round(float(np.mean(v[:8])), 2), "paper": v[8]} for k, v in TRAINED_PAPER.items()}
RES["V2_fb_total_successes"] = {k: int(sum(v) / 10) for k, v in FB.items()}
RES["V2_fb_wilson_total"] = {k: wilson(sum(v) // 10, 80) for k, v in FB.items()}
RES["V2_fb_fisher_astra_per_task"] = {
    t: round(fisher(a // 10, 10, b // 10, 10), 5) for t, a, b in zip(FB_TASKS, FB["Astra_no"], FB["Astra_fh"])}
RES["V2_fb_fisher_astra_total"] = fisher(56, 80, 70, 80)  # 비짝, 근사
RES["V2_fb_diff_newcombe_astra"] = newcombe_diff(70, 80, 56, 80)

# 공개 집계(저장소 docs/forcebench.md): 개발 시드 300-309와 테스트 시드 320-329, GPT-6 Astra
DEV = {"Astra_no": [8, 5, 10, 5, 10, 8, 0, 10], "Astra_fh": [10, 7, 10, 5, 10, 9, 10, 9],
       "Sol_no": [9, 5, 9, 3, 10, 6, 0, 10], "Sol_fh": [10, 6, 10, 4, 10, 10, 10, 10],
       "Fable_fh": [10, 8, 10, 2, 10, 5, 10, 5]}
HELD = {"Astra_no": [6, 5, 10, 4, 10, 6, 0, 9], "Astra_fh": [10, 6, 10, 2, 10, 10, 10, 8]}
FB_ONLY_FH = [4, 1, 0, 2, 0, 4, 10, 1]   # 표의 "only ForceHarness" 행
FB_ONLY_NO = [0, 0, 0, 4, 0, 0, 0, 2]    # 표의 "only no force" 행
RES["V2_dev_sums"] = {k: sum(v) for k, v in DEV.items()}
RES["V2_held_sums"] = {k: sum(v) for k, v in HELD.items()}
RES["V2_dev_mcnemar_astra"] = {"fh_only": 20, "no_only": 6,
                               "exact_p": round(mcnemar_exact(20, 6), 4), "cc_p": round(mcnemar_cc(20, 6), 4)}
RES["V2_held_mcnemar_astra"] = {"fh_only": sum(FB_ONLY_FH), "no_only": sum(FB_ONLY_NO),
                                "exact_p": round(mcnemar_exact(sum(FB_ONLY_FH), sum(FB_ONLY_NO)), 4),
                                "cc_p": round(mcnemar_cc(sum(FB_ONLY_FH), sum(FB_ONLY_NO)), 4)}
# 짝 구조 검사: FH 합 = 둘 다 성공 + FH만, 대조 합 = 둘 다 성공 + 대조만 (두 집계가 같은 시드 쌍인지)
RES["V2_pair_consistency_dev"] = {"both": 70 - 20, "no_check": 50 + 6}
RES["V2_pair_consistency_held"] = {"both": 66 - 22, "no_check": 44 + 6}
RES["V2_wilson_held_astra"] = {"fh_66of80": wilson(66, 80), "no_50of80": wilson(50, 80)}
RES["V2_held_gap_pp"] = round(82.5 - 62.5, 1)

# ----------------------------------------------------------------------------- V3 표 5 HiddenPhys
COLS = ["Weight", "CoM", "Wipe", "Stack", "Peg", "Latch", "Lock"]
HP = {  # per-task successes out of 20; total, damage, shift (in-distribution, shift) as printed
    "Expert": ([20, 19, 20, 20, 18, 19, 20], 136, 1, (77, 68)),
    "PositionOnly": ([10, 10, 0, 17, 5, 13, 16], 71, 6, (25, 22)),
    "DP": ([16, 17, 4, 11, 15, 9, 11], 83, 20, (52, 40)),
    "DP_force": ([18, 17, 7, 6, 15, 1, 10], 74, 13, (57, 43)),
    "DP_img": ([9, 17, 4, 12, 14, 0, 9], 65, 25, (44, 42)),
    "DP_img_force": ([16, 17, 5, 10, 14, 3, 9], 74, 16, (52, 43)),
    "pi05": ([9, 9, 14, 12, 12, 2, 11], 69, 24, (44, 35)),
    "pi05_with_force": ([15, 10, 7, 14, 10, 0, 11], 67, 18, (42, 35)),
    "ForceVLA": ([11, 10, 12, 12, 15, 2, 8], 70, 29, (48, 36)),
    "Astra_no": ([17, 10, 16, 20, 1, 13, 10], 87, 17, (44, 37)),
    "Astra_fh": ([20, 20, 20, 20, 15, 20, 20], 135, 0, (75, 69)),
    "Fable_no": ([10, 18, 17, 20, 3, 18, 10], 96, 11, (48, 39)),
    "Fable_fh": ([20, 20, 20, 20, 15, 20, 20], 135, 0, (75, 71)),
}
SHIFT_IDX = [0, 1, 2, 4]  # Weight, CoM, Wipe, Peg: 이 네 과제의 합이 표의 shift 분모(80)와 맞는지 확인
hp_rows = {}
for k, (vals, tot, dmg, (sid, ssh)) in HP.items():
    s = sum(vals)
    s4 = sum(vals[i] for i in SHIFT_IDX)
    hp_rows[k] = {"sum_ok": s == tot, "sum": s, "shift4_in_dist": s4, "shift4_printed": sid,
                  "shift4_ok": s4 == sid, "shift_drop_pct": round(100 * (sid - ssh) / sid, 1)}
RES["V3_hiddenphys_rows"] = hp_rows
RES["V3_astra_fh_wilson"] = wilson(135, 140)
RES["V3_astra_no_wilson"] = wilson(87, 140)
RES["V3_expert_wilson"] = wilson(136, 140)
RES["V3_best_trained_DP_wilson"] = wilson(83, 140)
RES["V3_astra_fh_vs_expert_diff_newcombe"] = newcombe_diff(135, 140, 136, 140)
RES["V3_astra_fh_vs_expert_fisher"] = round(fisher(135, 140, 136, 140), 4)
RES["V3_astra_fh_vs_no_fisher"] = fisher(135, 140, 87, 140)
RES["V3_astra_fh_vs_no_diff_newcombe"] = newcombe_diff(135, 140, 87, 140)
RES["V3_astra_fh_vs_DP_fisher"] = fisher(135, 140, 83, 140)
RES["V3_astra_fh_vs_pi05_with_force_fisher"] = fisher(135, 140, 67, 140)
RES["V3_fable_fh_vs_no_fisher"] = fisher(135, 140, 96, 140)
RES["V3_shift_claim_astra_fable_drop_pct"] = {"Astra_fh": round(100 * (75 - 69) / 75, 1),
                                              "Fable_fh": round(100 * (75 - 71) / 75, 1)}
RES["V3_shift_claim_no_force_drop_pct"] = {
    "Astra_no": round(100 * (44 - 37) / 44, 1), "Fable_no": round(100 * (48 - 39) / 48, 1)}
RES["V3_shift_claim_learned_drop_pct"] = {
    k: round(100 * (HP[k][3][0] - HP[k][3][1]) / HP[k][3][0], 1)
    for k in ["DP", "DP_force", "DP_img", "DP_img_force", "pi05", "pi05_with_force", "ForceVLA"]}

# ----------------------------------------------------------------------------- V4 표 6 형식 비교 (Sol 70, 소형 125)
FORM = {  # success, decision (Sol 70 episodes; small 125 episodes; decision denominator not stated)
    "no_force": ((0.63, 0.73), (0.18, 0.57)),
    "event_words": ((0.77, 0.70), (0.54, 0.64)),
    "force_plot": ((0.90, 1.00), (0.56, 0.71)),
    "raw_numbers": ((0.87, 1.00), (0.63, 0.69)),
    "derived": ((0.87, 0.97), (0.69, 0.85)),
    "L1": ((0.91, 1.00), (0.64, 0.84)),
    "L2": ((0.90, 1.00), (0.70, 0.85)),
}
RES["V4_form_nearest_counts"] = {
    k: {"sol_k_of_70": round(v[0][0] * 70, 1), "small_k_of_125": round(v[1][0] * 125, 1)} for k, v in FORM.items()}
# 판정 분모가 표에 없다. 분모를 125와 75(세 과제 중 판정 대상)로 가정했을 때 raw 대 L2 차이의 비짝 Fisher p
RES["V4_decision_raw_vs_L2_fisher_if_n125"] = round(fisher(round(0.85 * 125), 125, round(0.69 * 125), 125), 4)
RES["V4_decision_raw_vs_L2_fisher_if_n75"] = round(fisher(round(0.85 * 75), 75, round(0.69 * 75), 75), 4)
RES["V4_decision_nof_vs_raw_fisher_if_n125"] = round(fisher(round(0.69 * 125), 125, round(0.57 * 125), 125), 4)
RES["V4_decision_nof_vs_raw_fisher_if_n75"] = round(fisher(round(0.69 * 75), 75, round(0.57 * 75), 75), 4)

# ----------------------------------------------------------------------------- V5 비용과 기타 수치
RES["V5_cost_cut_pct"] = round(100 * (1 - 0.28 / 0.50), 1)  # 구성 파일 주석의 목록가 추정
RES["V5_fig1_gain_checks"] = {"real": round(90.0 - 3.3, 1), "forcebench": round(87.5 - 70.0, 1),
                              "hiddenphys": round(96.4 - 62.1, 1)}
RES["V5_hiddenphys_fig1_best_trained_pct"] = round(100 * 83 / 140, 1)

OUT.write_text(json.dumps(RES, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
               encoding="utf-8")
if __name__ == "__main__":
    print(json.dumps(RES, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))[:6000])
