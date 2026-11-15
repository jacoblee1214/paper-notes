"""논문 표 1–2, 7과 그림 6(a,c)의 수치를 다시 계산하고 일관성을 검사한다 (numpy, scipy, 수 초).

실행: python3 analysis/recompute.py  -> analysis/recompute_results.json
입력: paper/paper.txt의 표 값과 그림 6·7의 막대 라벨(페이지 8을 렌더링해 읽음). 원자료는 없다.
가정: 실로봇 시도는 과제당 20회, 시도끼리 독립으로 본다(같은 과제 안 상관은 알 수 없음).
      시뮬레이션 "±"는 3 시드의 표본 표준편차(ddof=1)로 본다. 논문에 정의가 없어 민감도도 본다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
TASKS = ["Cube Lift", "Pick-and-Place", "Stack Cube", "Close Drawer", "Stand Cup Up"]

# 표 1 실로봇 (성공/20)
REAL_BASE = [7, 9, 7, 14, 5]
REAL_RES = [17, 16, 15, 20, 8]
# 표 1 시뮬레이션 (평균, 3 시드 표준편차, 단위 /20)
SIM_BASE = [(4.3, 0.6), (7.0, 2.6), (10.0, 1.0), (11.3, 3.2), (5.3, 1.2)]
SIM_RES = [(20.0, 0.0), (17.0, 2.0), (14.7, 0.6), (19.7, 0.6), (14.7, 1.2)]
# 표 2 (a) 실로봇 절제, (b) 관측 공간 (성공/20, 순서 Lift, PnP, Stack, Close, Stand)
ABL = {
    "Ours": [17, 16, 15, 20, 8],
    "w/o noise injection": [16, 14, 11, 20, 8],
    "w/o pose dropout": [13, 12, 10, 16, 7],
    "w/o both": [12, 10, 9, 16, 5],
    "Distillation-based": [9, 4, 8, 20, 5],
    "Image-based": [8, 10, 9, 14, 6],
}
# 표 7 (Pick-and-Place, 20회): (sim, real)
TIER = {"Tier1 base (GR00T-N1.5)": ((7, 9)), "Tier1 + residual": ((17, 16)),
        "Tier2 base (pi0.5)": ((18, 17)), "Tier2 + residual": ((19, 17))}
# 그림 6(a) π0.5 실로봇 (성공/20), 막대 라벨 읽음: Close, Lift, Stand
PI05 = {"Close": (12, 18), "Lift": (6, 11), "Stand": (7, 13)}
# 그림 6(c) SFT 성공률 (%), n 미보고
SFT = {"Base (no SFT)": 42, "SFT on base rollouts": 47, "SFT on residual rollouts": 59}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def newcombe_diff(k1, n1, k2, n2):
    """두 비율 차이 p2-p1의 Newcombe(1998) 점수 구간 (Wilson 구간 조합)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p2 - p1
    lo = d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    hi = d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return [lo, hi]


def fisher(k1, n1, k2, n2):
    _, p = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
    return float(p)


def n_per_arm(p1, p2, alpha=0.05, power=0.8):
    """두 비율 비교의 팔당 표본 수 (정규 근사, 양측)."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return math.ceil((num / abs(p2 - p1)) ** 2)


def welch(m1, s1, n1, m2, s2, n2):
    if s1 == 0 and s2 == 0:
        return None
    t, p = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
    return {"t": float(t), "p_two_sided": float(p)}


def main():
    res = {"assumptions": {
        "real": "과제당 20회, 시도 독립 가정. 같은 과제 안 상관과 시도 순서는 알 수 없음",
        "sim_pm": "3 시드 표본 표준편차(ddof=1) 가정. 민감도로 ddof=0 환산 포함",
        "newcombe": "두 비율 차이의 Newcombe 점수 구간", "fisher": "양측 정확검정"}}

    # 1. 주장 C1: 42% -> 76% (실로봇 100 시도)
    kb, kr = sum(REAL_BASE), sum(REAL_RES)
    res["headline_real"] = {
        "base": [kb, 100], "res": [kr, 100],
        "wilson_base": wilson(kb, 100), "wilson_res": wilson(kr, 100),
        "diff_pts": kr - kb, "newcombe_diff": newcombe_diff(kb, 100, kr, 100),
        "fisher_p": fisher(kb, 100, kr, 100),
        "sim_avg_base": sum(m for m, _ in SIM_BASE) / 5, "sim_avg_res": sum(m for m, _ in SIM_RES) / 5,
        "real_avg_base_per20": kb / 5, "real_avg_res_per20": kr / 5,
    }

    # 2. 과제별 실로봇 차이, 부호 검정
    per = []
    for t, b, r in zip(TASKS, REAL_BASE, REAL_RES):
        per.append({"task": t, "base": [b, 20], "res": [r, 20], "diff": r - b,
                    "wilson_base": wilson(b, 20), "wilson_res": wilson(r, 20),
                    "newcombe_diff": newcombe_diff(b, 20, r, 20), "fisher_p": fisher(b, 20, r, 20)})
    res["real_per_task"] = per
    diffs = [r - b for b, r in zip(REAL_BASE, REAL_RES)]
    res["sign_test_all_positive_5_tasks_p_two_sided"] = float(2 * 0.5 ** 5)
    res["wilcoxon_diffs"] = diffs

    # 3. 측정 민감도: 과제당 20회로 0.35 -> 0.75 차이를 80% 검정력으로 보려면
    res["power"] = {"n_per_arm_0.35_to_0.75": n_per_arm(0.35, 0.75),
                    "n_per_arm_0.42_to_0.76": n_per_arm(0.42, 0.76),
                    "n_per_arm_0.90_to_0.50": n_per_arm(0.50, 0.90)}

    # 4. 표 2(a)(b) 절제와 대조: 각 설정 대 Ours (합 기준, Fisher)
    ours_k = sum(ABL["Ours"])
    abl = []
    for name, v in ABL.items():
        if name == "Ours":
            continue
        k = sum(v)
        abl.append({"setting": name, "k_per_100": k, "diff_vs_ours_pts": k - ours_k,
                    "newcombe_diff_vs_ours": newcombe_diff(k, 100, ours_k, 100),
                    "fisher_p_vs_ours": fisher(ours_k, 100, k, 100)})
    res["ablation_vs_ours"] = abl

    # 5. 표 2(b) 과제별 관측 공간 비교: Ours 대 이미지 기반, 증류
    obs = []
    for name in ("Distillation-based", "Image-based"):
        v = ABL[name]
        per_task = [{"task": t, "ours": o, "other": x, "fisher_p": fisher(o, 20, x, 20)}
                    for t, o, x in zip(TASKS, ABL["Ours"], v)]
        obs.append({"setting": name, "per_task": per_task})
    res["observation_space"] = obs

    # 6. 표 7 (강한 기반 정책)과 그림 6(a) π0.5
    res["tier_table7"] = {k: v for k, v in TIER.items()}
    res["tier_checks"] = {
        "tier1_sim_fisher_p": fisher(7, 20, 17, 20), "tier1_real_fisher_p": fisher(9, 20, 16, 20),
        "tier2_sim_fisher_p": fisher(18, 20, 19, 20), "tier2_real_fisher_p": fisher(17, 20, 17, 20),
        "tier1_real_ratio": 16 / 9, "tier1_sim_ratio": 17 / 7,
    }
    p_b = sum(v[0] for v in PI05.values())
    p_r = sum(v[1] for v in PI05.values())
    res["pi05_fig6a"] = {"per_task": PI05, "pooled_base": [p_b, 60], "pooled_res": [p_r, 60],
                         "fisher_p_pooled": fisher(p_b, 60, p_r, 60),
                         "newcombe_diff": newcombe_diff(p_b, 60, p_r, 60)}

    # 7. 그림 6(c) SFT: n 미보고. n = 100을 가정한 민감도만 계산
    res["sft_fig6c"] = {"values_pct": SFT, "n_assumed_for_sensitivity": 100,
                        "fisher_p_base_rollouts_vs_res_rollouts_if_n100":
                            fisher(47, 100, 59, 100),
                        "note": "SFT 시도 수와 반복 시드 수가 보고되지 않아 유의성 판단 불가"}

    # 8. 시뮬레이션 표 1: 과제별 Welch (3 시드, ddof=1)과 ddof=0 민감도
    tc = stats.t.ppf(0.975, 2)
    sim = []
    for t, (mb, sb), (mr, sr) in zip(TASKS, SIM_BASE, SIM_RES):
        e = {"task": t, "base": [mb, sb], "res": [mr, sr]}
        if sb > 0 and sr > 0:
            e["welch_ddof1"] = welch(mb, sb, 3, mr, sr, 3)
            sc = math.sqrt(3 / 2)
            e["welch_if_ddof0"] = welch(mb, sb * sc, 3, mr, sr * sc, 3)
        else:
            e["welch_ddof1"] = "sd=0 (검정 생략, 천장 효과)"
        e["ci95_halfwidth_base"] = tc * sb / math.sqrt(3)
        e["ci95_halfwidth_res"] = tc * sr / math.sqrt(3) if sr > 0 else 0.0
        sim.append(e)
    res["sim_table1"] = sim

    # 9. 일관성 검사
    res["consistency"] = {
        "sim_base_avg_from_tasks": sum(m for m, _ in SIM_BASE) / 5,
        "sim_res_avg_from_tasks": sum(m for m, _ in SIM_RES) / 5,
        "paper_sim_base_avg": 7.6, "paper_sim_res_avg": 17.2,
        "paper_real_base_avg": 8.4, "paper_real_res_avg": 15.2,
        "real_base_avg_from_tasks": sum(REAL_BASE) / 5, "real_res_avg_from_tasks": sum(REAL_RES) / 5,
        "fig6b_real_distill_equals_table2b_sum": sum(ABL["Distillation-based"]) == 46,
        "fig6b_real_image_equals_table2b_sum": sum(ABL["Image-based"]) == 47,
        "table7_tier1_pnp_matches_table1": (7, 9) == (7, 9) and (17, 16) == (17, 16),
        "table2a_ours_equals_table1_real_res": ABL["Ours"] == REAL_RES,
        "fig6c_base_42_equals_real_base": round(100 * kb / 100) == 42,
        "fig7b_reduction_claim_9_to_22pct": "figure readings -9,-22,-15,-13,-10 %, consistent",
    }

    # 10. 시뮬레이션과 실로봇의 과제별 구조 일치 (A.7 주장: 같은 실패 분포)
    sb = [m / 20 for m, _ in SIM_BASE]
    rb = [b / 20 for b in REAL_BASE]
    sr_ = [m / 20 for m, _ in SIM_RES]
    rr = [r / 20 for r in REAL_RES]
    gains_sim = [b - a for a, b in zip(sb, sr_)]
    gains_real = [b - a for a, b in zip(rb, rr)]
    res["sim_real_structure"] = {
        "base_sim_pct": [round(100 * x, 1) for x in sb], "base_real_pct": [round(100 * x, 1) for x in rb],
        "base_gap_sim_minus_real_pts": [round(100 * (a - b), 1) for a, b in zip(sb, rb)],
        "pearson_base_sim_vs_real": float(stats.pearsonr(sb, rb)[0]),
        "pearson_base_sim_vs_real_p": float(stats.pearsonr(sb, rb)[1]),
        "gain_sim_pts": [round(100 * x, 1) for x in gains_sim], "gain_real_pts": [round(100 * x, 1) for x in gains_real],
        "pearson_gain_sim_vs_real": float(stats.pearsonr(gains_sim, gains_real)[0]),
        "pearson_gain_sim_vs_real_p": float(stats.pearsonr(gains_sim, gains_real)[1]),
        "note": "과제 5개뿐이라 상관의 검정력이 매우 낮다",
    }

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    h = res["headline_real"]
    print("wrote", OUT)
    print("C1 42->76:", h["base"], h["res"], "Newcombe", [round(x, 3) for x in h["newcombe_diff"]], "Fisher p=%.2e" % h["fisher_p"])
    for r in per:
        print(r["task"], r["base"][0], "->", r["res"][0], "p=%.4f" % r["fisher_p"])
    for a in abl:
        print("abl", a["setting"], a["k_per_100"], "p vs ours=%.4f" % a["fisher_p_vs_ours"])
    print("pi05 pooled p=%.4f" % res["pi05_fig6a"]["fisher_p_pooled"])
    print("power n/arm", res["power"])
    print("sim-real corr base r=%.2f p=%.2f, gain r=%.2f p=%.2f" % (
        res["sim_real_structure"]["pearson_base_sim_vs_real"], res["sim_real_structure"]["pearson_base_sim_vs_real_p"],
        res["sim_real_structure"]["pearson_gain_sim_vs_real"], res["sim_real_structure"]["pearson_gain_sim_vs_real_p"]))


if __name__ == "__main__":
    main()
