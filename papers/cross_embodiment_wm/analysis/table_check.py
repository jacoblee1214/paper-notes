"""Table I(실물, Plasticine Reshaping) 재계산과 유의성 검정. 실행: python3 analysis/table_check.py → analysis/table_check_results.json

입력은 논문 Table I의 값뿐입니다(원시 trial 기록 없음). 다음을 계산합니다.
  1) 성공 횟수: 글자별 5회, 손·방법별 20회. Wilson 95% 구간, 글자별·합계 Fisher 정확검정
  2) Cochran–Mantel–Haenszel(글자를 층으로): 공통 오즈비, 연속성 없는 χ², 층별 조정
  3) CD/EMD: 합계(n=20 runs)의 평균±95% CI에서 표준편차를 역산하고 Welch t 검정. 글자별(n=5)도 같은 방식
  4) Holm 보정(가족별), 표본 크기 계산(두 비율), 글자별 평균 합산 검산, CD+EMD 평균과 성공 기준 0.0125 비교
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "table_check_results.json"

LETTERS = ["X", "R", "T", "A"]
# 논문 Table I (v3). 성공: 글자별 n=5, 손별 합계 n=20. CD, EMD는 ×10^-3, 값은 평균, 95% CI(= 논문 표기)
TABLE1 = {
    "Ability Hand": {
        "co-train": {
            "succ": [5, 4, 5, 4], "succ_total": 18,
            "CD": [(6.85, 0.21), (6.92, 0.10), (6.88, 0.19), (7.14, 0.14)], "CD_agg": (6.95, 0.10),
            "EMD": [(4.76, 0.17), (4.90, 0.35), (4.94, 0.27), (5.10, 0.18)], "EMD_agg": (4.92, 0.13),
        },
        "human-only": {
            "succ": [3, 2, 4, 1], "succ_total": 10,
            "CD": [(7.21, 0.55), (7.20, 0.15), (6.94, 0.14), (7.26, 0.14)], "CD_agg": (7.15, 0.16),
            "EMD": [(5.07, 0.67), (5.38, 0.39), (5.08, 0.25), (5.40, 0.36)], "EMD_agg": (5.23, 0.23),
        },
    },
    "XHand": {
        "co-train": {
            "succ": [5, 5, 4, 3], "succ_total": 17,
            "CD": [(6.65, 0.26), (6.70, 0.11), (6.99, 0.17), (7.05, 0.20)], "CD_agg": (6.85, 0.12),
            "EMD": [(4.52, 0.29), (4.65, 0.17), (4.90, 0.21), (5.07, 0.23)], "EMD_agg": (4.78, 0.14),
        },
        "human-only": {
            "succ": [4, 2, 2, 1], "succ_total": 9,
            "CD": [(6.82, 0.30), (7.22, 0.27), (7.32, 0.31), (7.53, 0.32)], "CD_agg": (7.22, 0.18),
            "EMD": [(4.66, 0.44), (5.26, 0.31), (5.37, 0.29), (5.43, 0.33)], "EMD_agg": (5.18, 0.21),
        },
    },
}
N_LETTER = 5
N_TOTAL = 20
SUCC_THRESHOLD = 12.5  # CD+EMD (×10^-3) 기준 0.0125


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def fisher_two_sided(a, b, c, d):
    """2×2 표 [[a,b],[c,d]] 의 양측 Fisher 정확검정 (scipy)."""
    _, p = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")
    return float(p)


def mantel_haenszel(strata):
    """strata: [(a,b,c,d), ...] 의 공통 오즈비와 χ²(연속성 보정 없음, df=1)와 p."""
    num = den = 0.0
    sA = sE = sV = 0.0
    for a, b, c, d in strata:
        n = a + b + c + d
        r1, r0 = a + b, c + d
        c1, c0 = a + c, b + d
        num += a * d / n
        den += b * c / n
        # 귀무가설(공통 OR=1) 아래 a의 기댓값과 분산
        E = r1 * c1 / n
        V = r1 * r0 * c1 * c0 / (n * n * (n - 1)) if n > 1 else 0.0
        sA += a
        sE += E
        sV += V
    OR = num / den if den > 0 else float("inf")
    chi2 = (sA - sE) ** 2 / sV if sV > 0 else float("nan")
    p = float(stats.chi2.sf(chi2, 1)) if sV > 0 else float("nan")
    # Robins–Breslow–Greenland 분산으로 log(OR) 95% CI
    tot = [sum(st) for st in strata]
    R = sum(a * d / n for (a, b, c, d), n in zip(strata, tot))
    S = sum(b * c / n for (a, b, c, d), n in zip(strata, tot))
    P = sum((a + d) / n * a * d / n for (a, b, c, d), n in zip(strata, tot))
    Q = sum((b + c) / n * b * c / n for (a, b, c, d), n in zip(strata, tot))
    Rr = sum((a + d) / n * a * d / n for (a, b, c, d), n in zip(strata, tot))
    Ss = sum((b + c) / n * b * c / n for (a, b, c, d), n in zip(strata, tot))
    var_log = P / (2 * R * R) + (Rr + Ss) / (2 * R * S) + Q / (2 * S * S) if R > 0 and S > 0 else float("nan")
    lo = math.exp(math.log(OR) - 1.959964 * math.sqrt(var_log)) if var_log == var_log else float("nan")
    hi = math.exp(math.log(OR) + 1.959964 * math.sqrt(var_log)) if var_log == var_log else float("nan")
    return {"OR": OR, "OR_95": [lo, hi], "chi2": chi2, "p": p}


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [None] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj


def sd_from_ci(ci, n):
    """95% t-CI 반폭에서 표준편차 역산: 반폭 = t(0.975, n-1) · s/√n."""
    t = stats.t.ppf(0.975, n - 1)
    return ci * math.sqrt(n) / t


def welch(m1, ci1, n1, m2, ci2, n2):
    s1, s2 = sd_from_ci(ci1, n1), sd_from_ci(ci2, n2)
    v1, v2 = s1 * s1 / n1, s2 * s2 / n2
    t = (m1 - m2) / math.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1 * v1 / (n1 - 1) + v2 * v2 / (n2 - 1))
    p = 2 * stats.t.sf(abs(t), df)
    return {"diff": m1 - m2, "t": t, "df": df, "p": float(p), "sd1": s1, "sd2": s2}


def sample_size_two_prop(p1, p2, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return math.ceil((num / (p1 - p2)) ** 2)


def main():
    res = {"hands": {}, "holm_success": {}, "holm_metric": {}}
    family_success_p = []
    family_success_labels = []
    family_metric_p = []
    family_metric_labels = []
    for hand, arms in TABLE1.items():
        co, ho = arms["co-train"], arms["human-only"]
        out = {"success": {}, "letters": {}, "metrics": {}}
        # 성공 횟수: 합계 Fisher, Wilson
        a, b = co["succ_total"], N_TOTAL - co["succ_total"]
        c, d = ho["succ_total"], N_TOTAL - ho["succ_total"]
        p_fisher = fisher_two_sided(a, b, c, d)
        out["success"] = {
            "co_train": f"{a}/{N_TOTAL}", "human_only": f"{c}/{N_TOTAL}",
            "wilson_co": [round(v, 3) for v in wilson(a, N_TOTAL)],
            "wilson_ho": [round(v, 3) for v in wilson(c, N_TOTAL)],
            "risk_diff": round(a / N_TOTAL - c / N_TOTAL, 3),
            "fisher_p": p_fisher,
        }
        strata = []
        letters = []
        for i, L in enumerate(LETTERS):
            ka, kc = co["succ"][i], ho["succ"][i]
            strata.append((ka, N_LETTER - ka, kc, N_LETTER - kc))
            letters.append({
                "letter": L, "co_train": f"{ka}/5", "human_only": f"{kc}/5",
                "fisher_p": fisher_two_sided(ka, 5 - ka, kc, 5 - kc),
            })
        mh = mantel_haenszel(strata)
        out["mh_letters"] = mh
        out["letters"] = letters
        out["sum_check_succ"] = {"co": sum(co["succ"]) == co["succ_total"], "ho": sum(ho["succ"]) == ho["succ_total"]}
        family_success_labels += [f"{hand}: pooled Fisher", f"{hand}: CMH"]
        family_success_p += [p_fisher, mh["p"]]

        # CD/EMD: 합계 n=20 Welch, 글자별 n=5 Welch
        for metric in ("CD", "EMD"):
            mco, cico = co[f"{metric}_agg"]
            mho, ciho = ho[f"{metric}_agg"]
            w = welch(mco, cico, N_TOTAL, mho, ciho, N_TOTAL)
            means_co = [v[0] for v in co[metric]]
            means_ho = [v[0] for v in ho[metric]]
            out["metrics"][metric] = {
                "agg_co": [mco, cico], "agg_ho": [mho, ciho],
                "agg_welch": w,
                "letter_mean_check_co": round(float(np.mean(means_co)), 3),
                "letter_mean_check_ho": round(float(np.mean(means_ho)), 3),
            }
            family_metric_labels.append(f"{hand} {metric} (합계 Welch)")
            family_metric_p.append(w["p"])
            for i, L in enumerate(LETTERS):
                wl = welch(co[metric][i][0], co[metric][i][1], N_LETTER, ho[metric][i][0], ho[metric][i][1], N_LETTER)
                out["metrics"][f"{metric}_{L}"] = {"welch": wl}
        # CD+EMD 평균과 성공 기준
        out["cd_plus_emd_letter"] = {
            "co": [round(co["CD"][i][0] + co["EMD"][i][0], 2) for i in range(4)],
            "ho": [round(ho["CD"][i][0] + ho["EMD"][i][0], 2) for i in range(4)],
            "threshold_x1e3": SUCC_THRESHOLD,
        }
        res["hands"][hand] = out

    adj_s = holm(family_success_p)
    res["holm_success"] = {lab: {"p": p, "p_holm": q} for lab, p, q in zip(family_success_labels, family_success_p, adj_s)}
    adj_m = holm(family_metric_p)
    res["holm_metric"] = {lab: {"p": p, "p_holm": q} for lab, p, q in zip(family_metric_labels, family_metric_p, adj_m)}

    # 표본 크기: 성공률 차이를 80% 검정력으로 찾으려면 손별 필요 trial 수
    res["power"] = {
        "n_per_arm_for_0.9_vs_0.5": sample_size_two_prop(0.9, 0.5),
        "n_per_arm_for_0.85_vs_0.5": sample_size_two_prop(0.85, 0.5),
        "n_per_arm_for_0.9_vs_0.7": sample_size_two_prop(0.9, 0.7),
    }
    # 20회 대 20회에서 검출 가능한 최소 차이(근사, 검정력 0.8): 직접 계산
    def power_20(p1, p2, n=20, alpha=0.05):
        # 정확 검정 대신 정규근사 검정력
        za = stats.norm.ppf(1 - alpha / 2)
        pbar = (p1 + p2) / 2
        se0 = math.sqrt(2 * pbar * (1 - pbar) / n)
        se1 = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
        return float(stats.norm.cdf((abs(p1 - p2) - za * se0) / se1))
    res["power"]["approx_power_20v20_0.9_vs_0.5"] = power_20(0.9, 0.5)
    res["power"]["approx_power_20v20_0.85_vs_0.5"] = power_20(0.85, 0.5)

    # 학습 실행 수(Fig. 2): 학습 조건 개수 계산 (논문 설명 기반)
    from math import comb
    subsets = {x: comb(5, x) for x in range(1, 6)}
    per_target_per_model = sum(subsets.values())
    res["fig2_runs"] = {
        "subsets_per_x_from_5_remaining": subsets,
        "total_subsets_per_target": per_target_per_model,
        "x5_subsets": subsets[5],
        "note": "x=5에서 부분집합은 1개뿐이라 그 점의 CI는 부분집합 분산이 아니라 학습 시드 분산이어야 합니다(시드 수는 논문에 없음)",
        "models_per_task_both_methods_6_targets": 2 * 6 * (per_target_per_model + 1) + 6,
    }
    # 시뮬 학습 데이터량: 타깃 1손 100 trajectory 기준
    res["fig2_data_budget"] = {"target_only_traj": 100, "x5_total_traj": 500, "x1_total_traj": 100,
                               "note": "Equal data 기준선은 source 5손 합계(500 trajectory)와 같은 샘플 수"}

    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o)))

    # 콘솔 요약
    for hand, out in res["hands"].items():
        s = out["success"]
        print(f"[{hand}] success co {s['co_train']} vs ho {s['human_only']}: Fisher p={s['fisher_p']:.4f}, "
              f"Wilson co {s['wilson_co']} ho {s['wilson_ho']}, diff {s['risk_diff']}")
        print(f"   CMH OR={out['mh_letters']['OR']:.2f} CI {out['mh_letters']['OR_95']} p={out['mh_letters']['p']:.4f}")
        for metric in ("CD", "EMD"):
            w = out["metrics"][metric]["agg_welch"]
            print(f"   {metric} agg diff {w['diff']:.3f} t={w['t']:.2f} df={w['df']:.1f} p={w['p']:.4f}")
    print("Holm success:", {k: round(v["p_holm"], 4) for k, v in res["holm_success"].items()})
    print("Holm metric:", {k: round(v["p_holm"], 4) for k, v in res["holm_metric"].items()})
    print("power:", res["power"])


if __name__ == "__main__":
    main()
