"""RT-1 (arXiv:2212.06817v2) 보고 수치 재계산. 표준 라이브러리만 사용한다.

출력: recompute_results.json (같은 폴더). 실행: python3 recompute.py
"""
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "..", "paper", "paper.txt")
OUT = os.path.join(HERE, "recompute_results.json")
res = {}


# ---------- 통계 도구 ----------
def hyp_pmf(k, N, K, n):
    return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)


def fisher_two_sided(a, n1, b, n2):
    """2x2 표 (a/n1 vs b/n2), 양측 Fisher 정확검정 p값."""
    N = n1 + n2
    K = a + b
    p_obs = hyp_pmf(a, N, K, n1)
    lo, hi = max(0, K - n2), min(K, n1)
    p = 0.0
    for x in range(lo, hi + 1):
        px = hyp_pmf(x, N, K, n1)
        if px <= p_obs * (1 + 1e-9):
            p += px
    return min(1.0, p)


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def newcombe_diff(a, n1, b, n2, z=1.959964):
    """RT-1 대 기준선 차이(a/n1 - b/n2)의 Newcombe(1998) 95% CI."""
    p1, p2 = a / n1, b / n2
    l1, u1 = wilson(a, n1, z)
    l2, u2 = wilson(b, n2, z)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def feasible_n(values, lo=5, hi=250):
    """모든 보고 백분율이 같은 분모 n에서 반올림 일치하는 n을 찾는다."""
    out = []
    for n in range(lo, hi + 1):
        ok = True
        for pct in values:
            # 가능한 정수 k 중 반올림이 pct와 같은 것이 하나라도 있어야 한다.
            ks = [k for k in range(0, n + 1) if round(100 * k / n + 1e-9) == pct]
            if not ks:
                ok = False
                break
        if ok:
            out.append(n)
    return out


# ---------- 1. 표 2 · 표 7 · 표 13 백분율의 가능한 분모 ----------
res["V1_feasible_denominators"] = {
    "T2_unseen_RT1_Gato_BCZ_BCZXL_n5to60": feasible_n([76, 52, 19, 43], hi=60),
    "T2_distractor_RT1_Gato_BCZ_BCZXL_n5to60": feasible_n([83, 43, 47, 23], hi=60),
    "T2_background_RT1_Gato_BCZ_BCZXL_n5to60": feasible_n([59, 35, 41, 35], hi=60),
    "note": "n=21(6.2 본문)은 unseen 네 값을 모두 맞춘다. 53(D.1)은 76%를 맞추지 못한다. "
            "배경 n=17이 네 값을 모두 맞추며, 본문의 22 과제·D.1의 6과제×3단계=18과는 다르다.",
}
res["V1_check_53_for_unseen_RT1"] = {
    "k_for_76pct_n53": [k for k in range(54) if round(100 * k / 53) == 76],
}

# ---------- 2. 표 7 "All" 열 = 세 일반화 열의 평균인가 ----------
t7 = {
    "RT-1 (full)": (73, [76, 83, 59]),
    "51% data": (50, [52, 39, 59]),
    "37% data": (46, [57, 35, 47]),
    "22% data": (29, [14, 31, 41]),
    "narrow 75% tasks": (54, [67, 42, 53]),
}
res["V2_table7_all_is_mean_of_three"] = {
    k: {"reported_all": v[0], "mean_of_three": round(sum(v[1]) / 3, 2)} for k, v in t7.items()
}

# ---------- 3. 표 13 distractor "All"과 easy/medium/hard 평균 ----------
t13_dist = {
    "RT-1": (83, [100, 100, 64]),
    "Gato": (43, [71, 44, 29]),
    "BC-Z": (47, [100, 67, 7]),
    "BC-Z XL": (23, [57, 33, 0]),
    "w/o big model": (77, [100, 100, 50]),
    "w/o pre-training": (60, [100, 67, 36]),
    "w/ continuous": (37, [71, 67, 0]),
    "w/ AR actions": (67, [100, 78, 43]),
    "w/o history": (50, [71, 89, 14]),
    "w/o Transformer": (67, [100, 100, 29]),
}
res["V3_table13_distractor_all_vs_mean"] = {
    k: {"reported_all": v[0], "mean_easy_med_hard": round(sum(v[1]) / 3, 1)} for k, v in t13_dist.items()
}

# ---------- 4. "%" 표기: 상대 대 %p ----------
pairs = {
    "seen vs BC-Z (97 vs 72)": (97, 72),
    "seen vs Gato (97 vs 65)": (97, 65),
    "unseen vs Gato (76 vs 52)": (76, 52),
    "distractor vs BC-Z (83 vs 47)": (83, 47),
    "background vs BC-Z (59 vs 41)": (59, 41),
}
res["V4_pct_points_vs_relative"] = {
    k: {"difference_pp": a - b, "relative_pct": round(100 * (a - b) / b, 1)} for k, (a, b) in pairs.items()
}
res["V4_abstract_unseen_claim"] = {
    "abstract_says": 25, "body_6.2_says": 24, "computed_pp_vs_next_best_Gato": 76 - 52,
}

# ---------- 5. 검정: 명목 n을 가정한 Fisher·Newcombe ----------
tests = {}
# 일반화(n=21 가정, 표 2와 일치)
tests["unseen_RT1_vs_BCZ_n21"] = dict(zip(
    ["diff", "ci_lo", "ci_hi"], newcombe_diff(16, 21, 4, 21)), p_fisher=fisher_two_sided(16, 21, 4, 21))
tests["unseen_RT1_vs_Gato_n21"] = dict(zip(
    ["diff", "ci_lo", "ci_hi"], newcombe_diff(16, 21, 11, 21)), p_fisher=fisher_two_sided(16, 21, 11, 21))
# 방해물(n=30 가정, 표 2와 일치)
tests["distractor_RT1_vs_BCZ_n30"] = dict(zip(
    ["diff", "ci_lo", "ci_hi"], newcombe_diff(25, 30, 14, 30)), p_fisher=fisher_two_sided(25, 30, 14, 30))
tests["distractor_RT1_vs_Gato_n30"] = dict(zip(
    ["diff", "ci_lo", "ci_hi"], newcombe_diff(25, 30, 13, 30)), p_fisher=fisher_two_sided(25, 30, 13, 30))
# 표 13 어블레이션 (방해물 n=30, 미수집 표준은 n=30 가정, 일반화 n=21 가정)
tests["abl_distractor_wo_pretrain_vs_full_n30"] = {"p_fisher": fisher_two_sided(18, 30, 25, 30)}
tests["abl_distractor_wo_history_vs_full_n30"] = {"p_fisher": fisher_two_sided(15, 30, 25, 30)}
tests["abl_unseen_wo_pretrain_vs_full_n21"] = {"p_fisher": fisher_two_sided(9, 21, 16, 21)}
tests["abl_unseen_cont_vs_full_n21"] = {"p_fisher": fisher_two_sided(9, 21, 16, 21)}
# 장기 과제 (SayCan, n=15 가정, 표 6과 일치)
tests["saycan_K2_RT1_10of15_vs_BCZ_2of15"] = dict(zip(
    ["diff", "ci_lo", "ci_hi"], newcombe_diff(10, 15, 2, 15)), p_fisher=fisher_two_sided(10, 15, 2, 15))
tests["saycan_K1_RT1_10of15_vs_BCZ_8of15"] = dict(zip(
    ["diff", "ci_lo", "ci_hi"], newcombe_diff(10, 15, 8, 15)), p_fisher=fisher_two_sided(10, 15, 8, 15))
tests["saycan_K2_RT1_10of15_vs_K1_10of15"] = {"p_fisher": fisher_two_sided(10, 15, 10, 15)}
# Kuka 빈 피킹 평가 (72회, 본문 D.2 명시)
tests["kuka_bin_mixed_28of72_vs_EDR_16of72"] = dict(zip(
    ["diff", "ci_lo", "ci_hi"], newcombe_diff(28, 72, 16, 72)), p_fisher=fisher_two_sided(28, 72, 16, 72))
res["V5_tests"] = tests
res["V5_kuka_check"] = {"22pct_of_72": round(100 * 16 / 72, 1), "39pct_of_72": round(100 * 28 / 72, 1)}

# 표 4 sim: 분모 미보고. 분모별 민감도(실세계 seen-skill 92 vs 90, 시뮬 unseen 7 vs 33)
sens = {}
for n in (10, 20, 30, 50):
    k_seen_real = round(0.92 * n); k_seen_mix = round(0.90 * n)
    k_u0 = round(0.07 * n); k_u1 = round(0.33 * n)
    sens[str(n)] = {
        "p_seen_real_vs_mix": round(fisher_two_sided(k_seen_real, n, k_seen_mix, n), 3),
        "p_unseen_skill_0_vs_1": round(fisher_two_sided(k_u0, n, k_u1, n), 3),
    }
res["V6_table4_sensitivity_by_assumed_n"] = sens

# ---------- 6. 표 1 · 부록 Table 8 · 시험 수 ----------
res["V7_table1_sum"] = sum([130, 337, 8, 8, 3, 3, 84, 162, 9])
res["V7_seen_eval_task_sum_6.1"] = 36 + 35 + 35 + 48 + 18 + 36
with open(PAPER, encoding="utf-8", errors="ignore") as f:
    lines = f.read().splitlines()
start = next(i for i, l in enumerate(lines) if l.strip().startswith("Table 8: List of Unseen"))
t8 = [l for l in lines[start - 60:start] if re.match(r"^\s+(pick|move|place|knock|open|close|pull)\b", l)]
res["V7_table8_unseen_row_count"] = len(t8)

# ---------- 7. 파라미터 수: 코드 기본값(layer_size=128, heads=8, d=512, ff=512, 8층) ----------
d, heads, key_dim, ff, layers, vocab = 512, 8, 128, 512, 8, 256
mha = 3 * (d * heads * key_dim + heads * key_dim) + (heads * key_dim * d + d)
ln = 2 * (2 * d)
dense_ff = d * ff + ff
per_layer = mha + ln + dense_ff
trans_core = layers * per_layer
head = d * vocab + vocab
res["V8_transformer_params"] = {
    "per_layer": per_layer,
    "8_layers": trans_core,
    "output_head_256": head,
    "approx_total_M": round((trans_core + head) / 1e6, 2),
    "paper_claim_M": 19,
    "note": "토큰 위치 임베딩 Dense 등 작은 항은 제외했다. 논문의 19M과 맞는다.",
}
res["V8_inference_slowdown"] = {"RT1_ms": 15, "auto_reg_ms": 36, "ratio": round(36 / 15, 2)}
res["V8_param_reduction"] = {
    "D4_first_statement_35_to_21_pct": round(100 * (1 - 21 / 35), 1),
    "D4_second_statement_31_to_25_pct": round(100 * (1 - 25 / 31), 1),
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(json.dumps({k: res[k] for k in list(res)[:1]}, ensure_ascii=False))
print(json.dumps({k: v for k, v in res.items() if not k.startswith("V1_feas")}, ensure_ascii=False, indent=0)[:5000])
