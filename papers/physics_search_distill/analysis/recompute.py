"""논문 보고 수치의 재계산과 일관성 검사 (arXiv:2611.24171 v2).

실행: python3 analysis/recompute.py  → analysis/recompute_results.json
입력은 논문 본문·표·부록에 적힌 값뿐이다. 학습·시뮬레이션·가중치 다운로드는 하지 않는다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}


def check(name, computed, reported, tol):
    ok = abs(computed - reported) <= tol
    res[name] = {"computed": round(computed, 4), "reported": reported, "tol": tol, "ok": bool(ok)}
    return ok


# 1. 수리적 표본 구조 (논문 §4.3, 부록 D·E). 제안 723개 = 241 참조 x 3 시드
N_PROP = 723
acc_base, acc_road = 383, 507
clr_base, clr_road = 331, 206
hd_base, hd_road = 9, 10
res["counts_sum_check"] = {
    "base_total": acc_base + clr_base + hd_base,
    "road_total": acc_road + clr_road + hd_road,
    "expected": N_PROP,
}
check("acc_base_pct", 100 * acc_base / N_PROP, 53.0, 0.05)
check("acc_road_pct", 100 * acc_road / N_PROP, 70.1, 0.05)
gain_pp = 100 * (acc_road - acc_base) / N_PROP
check("acc_gain_pp_vs_17.15", gain_pp, 17.15, 0.01)
check("clr_rej_base_pct", 100 * clr_base / N_PROP, 45.8, 0.05)
check("clr_rej_road_pct", 100 * clr_road / N_PROP, 28.5, 0.05)
check("head_rej_base_pct", 100 * hd_base / N_PROP, 1.2, 0.05)
check("head_rej_road_pct", 100 * hd_road / N_PROP, 1.4, 0.05)

# 1-b. 나이브 두 비율 차이 CI (제안을 독립으로 가정: 과소추정). 논문의 세션 부트스트랩과 비교용
p1, p2 = acc_base / N_PROP, acc_road / N_PROP
se_naive = math.sqrt(p1 * (1 - p1) / N_PROP + p2 * (1 - p2) / N_PROP)
res["naive_two_prop_ci_pp"] = [round(100 * (p2 - p1 - 1.96 * se_naive), 2), round(100 * (p2 - p1 + 1.96 * se_naive), 2)]
res["paper_session_boot_ci_pp"] = [11.88, 21.79]

# 1-c. 정확한 숫자 가능성: 보고된 "62.38%"는 451/723인가
res["repair_fail_62.38_as_count"] = {"count": 451, "pct": round(100 * 451 / N_PROP, 2)}
res["drop_70.1_minus_62.38"] = round(70.1 - 62.38, 2)
res["drop_exact_507_minus_451"] = round(100 * (507 - 451) / N_PROP, 2)

# 2. 참조별 쌍비교 (부록 D). 228 개선, 11 악화, 2 동일
up, down, tie = 228, 11, 2
res["paired_improved_pct"] = {"pct": round(100 * up / 241, 2), "reported": 94.6}
sign = stats.binomtest(up, up + down, 0.5, alternative="two-sided")
res["sign_test_228_vs_11_p"] = float(f"{sign.pvalue:.3g}")
cp = stats.beta.ppf([0.025, 0.975], [up, up + 1], [241 - up + 1, 241 - up])
res["clopper_pearson_228of241_pct"] = [round(100 * float(cp[0]), 2), round(100 * float(cp[1]), 2)]
res["paper_improved_interval_pct"] = [90.1, 97.9]
res["median_clr_diff_cm_naive"] = round(16.45 - 4.32, 2)
res["median_paired_gain_cm_reported"] = 14.20

# 3. 교사 실행 가능 부분집합 (202 참조 x 3 시드 = 606)
n_t = 202 * 3
s_base = round(0.627 * n_t)
s_road = round(0.827 * n_t)
res["teacher_feasible"] = {
    "n_pairs": n_t,
    "base_count_implied": s_base, "base_pct": round(100 * s_base / n_t, 2),
    "road_count_implied": s_road, "road_pct": round(100 * s_road / n_t, 2),
    "gain_pp": round(100 * (s_road - s_base) / n_t, 2),
}

# 4. 표 4 이득과 보유율 (반올림 범위를 포함해 확인)
eas, fixed, prs, fcrs, cast = 90.4, 90.6, 90.7, 91.0, 91.2
d_cast = cast - eas
res["gain_table4"] = {
    "fixed_gain": round(fixed - eas, 2), "prs_gain": round(prs - eas, 2),
    "fcrs_gain_rounded": round(fcrs - eas, 2), "cast_gain_rounded": round(d_cast, 2),
    "fcrs_gain_reported": 0.56, "cast_gain_reported": 0.77,
    "retention_from_reported_gains_pct": round(100 * 0.56 / 0.77, 2),
    "retention_reported_pct": 73.2,
    "fixed_retention_reported": 28.7, "prs_retention_reported": 41.3,
    "prs_retention_from_reported_pct": round(100 * 0.32 / 0.77, 2),
    "fixed_retention_from_reported_pct": round(100 * 0.22 / 0.77, 2),
}
# 90.97 (본문)과 표 4의 EAS 90.4를 쓰면 FCRS 이득 범위
res["fcrs_gain_range_from_rounding"] = [round(90.95 - 90.45, 2), round(90.99 - 90.35, 2)]
res["cast_gain_range_from_rounding"] = [round(91.15 - 90.45, 2), round(91.25 - 90.35, 2)]
res["retention_range_from_rounding_pct"] = [
    round(100 * res["fcrs_gain_range_from_rounding"][0] / res["cast_gain_range_from_rounding"][1], 1),
    round(100 * res["fcrs_gain_range_from_rounding"][1] / res["cast_gain_range_from_rounding"][0], 1),
]
res["fcrs_vs_cast_gap_text_0.21"] = round(91.18 - 90.97, 2)

# 5. 롤아웃 비용 (§4.4)
res["rollouts"] = {
    "search_total": 1040, "prs_total": 106,
    "reduction_pct": round(100 * (1 - 106 / 1040), 2), "reported_pct": 89.8,
    "search_per_trigger": round(1040 / 106, 2), "reported_per_trigger": 9.81,
    "triggers_unfeasible_share_pct": round(100 * 106 / 278, 2), "reported_share": 38.1,
    "latency_ratio_search_over_fcrs": round(239.49 / 25.13, 2),
}

# 6. 파라미터 수 (부록 A.3, Table S1 주변). 정확히 맞아야 한다
def lin(i, o):
    return i * o + o

prs_pointwise = lin(31, 64) + lin(64, 64)
prs_static = lin(10, 64) + lin(64, 64)
prs_trunk = lin(1088, 128) + lin(128, 64)
prs_head = lin(64, 2)
prs_total = prs_pointwise + prs_static + prs_trunk + prs_head
proj = lin(512, 32)  # LayerNorm은 affine이 없다
resid = lin(96, 64) + lin(64, 32) + lin(32, 2)
res["params"] = {
    "prs_computed": prs_total, "prs_reported": 158850,
    "projection_computed": proj, "projection_reported": 16416,
    "residual_computed": resid, "residual_reported": 8354,
    "trainable_sum": proj + resid, "trainable_reported": 24770,
    "student_total": prs_total + proj + resid, "student_reported": 183620,
    "flatten_dim_check": 16 * 64 + 64,
}
res["params"]["all_ok"] = (prs_total == 158850 and proj == 16416 and resid == 8354
                           and proj + resid == 24770 and prs_total + proj + resid == 183620)

# 7. 표 2 (NAVSIM v2 navhard): 이득과 상대 이득
res["epdms"] = {
    "gain_vs_worlddrive": round(38.96 - 27.72, 2), "reported_gain": 11.24,
    "rel_pct": round(100 * (38.96 - 27.72) / 27.72, 2), "reported_rel_pct": 40.5,
    "gain_vs_diffusiondrive": round(38.96 - 27.5, 2), "reported_gain_dd": 11.46,
    "dac_gain_s1": round(93.5 - 80.4, 1), "dac_gain_s2": round(81.0 - 68.0, 1),
    "ep_gain_s1": round(82.6 - 81.8, 1), "ep_gain_s2": round(82.4 - 81.6, 1),
}

# 8. 표 1 (NAVSIM v1): ΔPDMS = PDMS - 89.0 (WorldDrive)
table1 = {"TransFuser": (84.0, -5.0), "Hydra-MDP": (86.5, -2.5), "DiffusionDrive": (88.1, -0.9),
          "WoTE": (88.3, -0.7), "UniAD": (83.4, -5.6), "PARA-Drive": (84.0, -5.0), "LAW": (84.6, -4.4),
          "World4Drive": (85.1, -3.9), "FSDrive": (85.1, -3.9), "DrivingGPT": (82.4, -6.6), "Epona": (86.2, -2.8),
          "ImagiDrive": (87.4, -1.6), "PWM": (88.1, -0.9), "DriveVLA-W0": (88.4, -0.6), "WorldDrive": (89.0, 0.0),
          "CAST": (91.2, 2.2), "FCRS": (91.0, 2.0)}
res["table1_delta_mismatch"] = [k for k, (p, d) in table1.items() if round(p - 89.0, 1) != d]

# 9. 표 1의 PDMS를 하위 점수로 역산 (참고용). 논문 식 (S5)은 장면별 곱의 평균이라 평균끼리의 곱과 다르다
def pdms_of_means(nc, dac, ep, ttc, c):
    return 100 * (nc / 100) * (dac / 100) * ((5 * ep / 100 + 5 * ttc / 100 + 2 * c / 100) / 12)

res["pdms_product_of_means"] = {
    "WorldDrive": {"computed": round(pdms_of_means(98.4, 96.8, 83.3, 95.2, 100.0), 2), "reported": 89.0},
    "FCRS": {"computed": round(pdms_of_means(98.7, 98.9, 84.9, 95.2, 100.0), 2), "reported": 91.0},
    "CAST": {"computed": round(pdms_of_means(98.9, 99.0, 85.0, 95.3, 100.0), 2), "reported": 91.2},
}

# 9-b. 상한: NC, DAC는 장면별 0/1이고 S = (5EP+5TTC+2C)/12는 0–1이므로 PDMS = E[NC·DAC·S] ≤ E[S].
# E[S]는 평균의 선형 결합이라 정확히 계산된다
def e_s(ep, ttc, c):
    return 100 * (5 * ep / 100 + 5 * ttc / 100 + 2 * c / 100) / 12


res["pdms_upper_bound_E_S"] = {
    "WorldDrive": {"E_S": round(e_s(83.3, 95.2, 100.0), 2), "reported": 89.0},
    "FCRS": {"E_S": round(e_s(84.9, 95.2, 100.0), 2), "reported": 91.0},
    "CAST": {"E_S": round(e_s(85.0, 95.3, 100.0), 2), "reported": 91.2},
}
res["pdms_upper_bound_ok"] = all(v["reported"] <= v["E_S"] for v in res["pdms_upper_bound_E_S"].values())

# 10. NAVSIM v1 점수 차이의 필요한 장면별 표준편차 (쌍 비교, 독립 가정, 12,146 장면)
n_scene = 12146
gain = 0.56
sd_needed = gain / 1.96 * math.sqrt(n_scene)
res["navsim_sd_needed_points"] = round(sd_needed, 1)

# 11. 복수 시드 SD로 보고된 값의 검산: 세 시드 표본 SD와 평균의 공차
res["seed_note"] = "세 시드의 개별 값이 없어 평균·표준편차를 재계산할 수 없다. 보고값만 확인"

ok_all = [k for k, v in res.items() if isinstance(v, dict) and v.get("ok") is False]
res["failed_checks"] = ok_all
res["params"]["all_ok"] = bool(res["params"]["all_ok"])

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
for k, v in res.items():
    print(k, v)
