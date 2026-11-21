"""Ego2X (arXiv:2611.24138 v2) 보고 수치를 다시 계산하고 검정한다.

입력: 논문 본문과 표에서 옮겨 적은 개수와 비율(출처 주석). 외부 데이터는 쓰지 않는다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 성공률 분모: X-VLA 과제 200 paired scenes, 장면 수가 다른 경우(195)는 본문 표기대로 쓴다.
- 비율의 분모는 본문이 적은 시도 수다(31/52, 45/53 등). 개수로 바꿀 수 없는 값(예: 62.9% 슬롯 기준)은 다루지 않는다.
- McNemar 쌍 수 "b:c"는 논문 표 7의 표기를 그대로 b(Ego2X만 성공):c(기준선만 성공)로 가정한다.
  방향이 반대라고 가정하면 결과가 달라지므로 03 문서에서 함께 적는다.
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


def newcombe_diff(k1, n1, k2, n2, z=1.959964):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def mcnemar_exact(b, c):
    """정확 양측 McNemar (이항 검정, 불일치 쌍 b+c)."""
    n = b + c
    if n == 0:
        return 1.0
    return float(stats.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


# 1. 비율 구간 (Wilson) -- 논문이 보고한 개수로부터
counts = {
    "yield_bread_ego2x": (31, 52),      # 표 7, Place bread in basket, 52 matched videos
    "yield_bread_huro_e2r": (1, 52),    # 표 7, HuRo/E2R
    "yield_bell_ego2x": (45, 53),       # 표 7, Place bell on scale, 53 specs
    "yield_bell_baselines": (0, 53),
    "xvla_zero_shot": (81, 200),        # 40.5%
    "xvla_verified_55lw": (134, 200),   # 67.0%
    "xvla_verified_P70": (127, 200),    # 63.5%
    "xvla_unfiltered_bread": (82, 200), # 41.0%
    "xvla_bell_zero_shot": (34, 200),   # 17.0%
    "xvla_bell_zero_retry_prompt": (110, 200),  # 55.0%
    "xvla_bell_unfiltered": (54, 200),  # 27.0%
}
RES["wilson_95"] = {k: {"k": a, "n": n, "p": round(a / n, 4), "ci": [round(x, 4) for x in wilson(a, n)]}
                    for k, (a, n) in counts.items()}

# 2. 독립 비교 (정확 Fisher; 쌍 설계가 아니므로 참고용)
t1 = stats.fisher_exact([[31, 21], [1, 51]])
t2 = stats.fisher_exact([[45, 8], [0, 53]])
RES["fisher_yield"] = {
    "bread_31of52_vs_1of52": {"p_two_sided": float(t1.pvalue), "diff_newcombe": newcombe_diff(31, 52, 1, 52)},
    "bell_45of53_vs_0of53": {"p_two_sided": float(t2.pvalue), "diff_newcombe": newcombe_diff(45, 53, 0, 53)},
}

# 3. 쌍 설계 McNemar: 보고된 불일치 쌍 수와 SR 차이 대조
# (b, c, 보고된 p, 보고된 SR_ego2x, SR_기준, 장면 수)
pairs = {
    "bread_55lw_vs_zero_shot": (70, 21, 5.0e-7, 0.670, 0.405, 200),
    "bread_P70_vs_zero_shot": (67, 24, 7.4e-6, 0.635, 0.405, 200),
    "bell_zero_retry_prompt_vs_zero_shot": (84, 11, 2.7e-14, 0.550, 0.170, 200),
    "bell_fixed_prompt_vs_zero_shot_195": (68, 8, 5.6e-13, 0.480, 0.170, 195),
    "bell_unfiltered_vs_zero_shot": (17, 58, 8.7e-6, 0.270, 0.170, 200),
    "bread_unfiltered_vs_zero_shot": (None, None, None, 0.410, 0.405, 200),
}
pr = {}
for name, (b, c, rep_p, s1, s0, n) in pairs.items():
    if b is None:
        pr[name] = {"note": "쌍 수 미보고"}
        continue
    implied = (s1 - s0) * n
    pr[name] = {
        "b": b, "c": c,
        "exact_mcnemar_p": mcnemar_exact(b, c),
        "reported_p": rep_p,
        "b_minus_c": b - c,
        "implied_b_minus_c_from_SR": round(implied, 1),
        "mismatch_pairs": round((b - c) - implied, 1),
    }
RES["mcnemar_check"] = pr

# 4. 소표본 Wilcoxon (6 clips, HaWoR, 6/6 양수): 정확 p
RES["wilcoxon_6of6"] = {"exact_two_sided_p": float(stats.binomtest(6, 6, 0.5).pvalue) * 1.0,
                        "note": "정확 부호검정 기준, 순위 검정의 최소 가능 양측 p = 2/64"}

# 5. 상대 감소율 재계산 (논문 표 3 값)
RES["relative_changes"] = {
    "HaWoR_ray_error_33.7pct": round((14.35 - 9.51) / 14.35 * 100, 1),
    "EgoHand_ray_error_14.1pct": round((51.71 - 44.41) / 51.71 * 100, 1),
    "EgoHand_MPJPE_reduction_pct": round((54.85 - 48.62) / 54.85 * 100, 1),
    "HaWoR_MPJPE_reduction_pct": round((19.16 - 15.77) / 19.16 * 100, 1),
    "conf_frame_F1_doubling_ratio": round(0.386 / 0.191, 2),
    "in_contact_median_trans_err_cut_pct_p70": round((104.6 - 67.3) / 104.6 * 100, 1),
    "in_contact_median_trans_err_cut_pct_held": round((141.9 - 99.1) / 141.9 * 100, 1),
    "xvla_gain_pp": round(67.0 - 40.5, 1),
    "mean_gain_bread_pp_from_table": round(63.5 - 40.5, 1),
}

# 6. 보고된 성공률을 개수로 바꿀 때의 정수성 확인 (67.0% of 200 = 134 등)
RES["integrality"] = {k: round(p * n, 3) for k, (p, n) in {
    "67.0_of_200": (0.670, 200), "40.5_of_200": (0.405, 200), "63.5_of_200": (0.635, 200),
    "41.0_of_200": (0.410, 200), "44.5_of_200": (0.445, 200), "55.0_of_200": (0.550, 200),
    "17.0_of_200": (0.170, 200), "27.0_of_200": (0.270, 200), "48.0_of_195": (0.480, 195),
    "47.0_of_200": (0.470, 200),
}.items()}

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
if __name__ == "__main__":
    print(json.dumps(RES, ensure_ascii=False, indent=2, default=float))
