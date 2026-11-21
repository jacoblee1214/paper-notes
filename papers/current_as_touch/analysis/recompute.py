"""논문 표의 원시 횟수로 비율, Wilson 95% 구간, Fisher 정확검정, 위험차 구간(Newcombe)을 다시 계산한다.

입력: 논문 표의 값(paper/paper.pdf, 표 1–4, 7–9). 원시 시도 기록은 공개되어 있지 않으므로
      퍼센트를 k/n 으로 되돌리는 n을 먼저 확인한다(check_denominators).
실행: python3 analysis/recompute.py   → analysis/recompute_results.json (수 초, CPU)
"""
import json
from pathlib import Path

from scipy.stats import fisher_exact, norm

OUT = Path(__file__).resolve().parent / "recompute_results.json"
Z = norm.ppf(0.975)


def wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * ((p * (1 - p) / n + Z * Z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


def newcombe_diff(k1, n1, k2, n2):
    """두 비율 차(p1 - p2)의 Newcombe 구간(Wilson 기반)."""
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    lo = d - ((p1 - l1) ** 2 + (u2 - p2) ** 2) ** 0.5
    hi = d + ((u1 - p1) ** 2 + (p2 - l2) ** 2) ** 0.5
    return d, lo, hi


def fisher(k1, n1, k2, n2):
    _, p = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
    return p


def n_per_arm(p1, p2, alpha=0.05, power=0.8):
    """두 비율 차이를 검출하는 데 필요한 arm당 n(정규근사, 양측)."""
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (za * (2 * pbar * (1 - pbar)) ** 0.5 + zb * (p1 * (1 - p1) + p2 * (1 - p2)) ** 0.5) ** 2
    return int(-(-num // (p1 - p2) ** 2))


def row(label, k, n):
    lo, hi = wilson(k, n)
    return {"label": label, "k": k, "n": n, "pct": round(100 * k / n, 1),
            "wilson95": [round(100 * lo, 1), round(100 * hi, 1)]}


def pair(label, a, b):
    (la, ka, na), (lb, kb, nb) = a, b
    d, lo, hi = newcombe_diff(ka, na, kb, nb)
    return {"comparison": label,
            "a": f"{la} {ka}/{na}", "b": f"{lb} {kb}/{nb}",
            "diff_pp": round(100 * d, 1), "newcombe95_pp": [round(100 * lo, 1), round(100 * hi, 1)],
            "fisher_p": round(fisher(ka, na, kb, nb), 4)}


def check_denominators():
    """표의 퍼센트가 어떤 n에서 정수 k로 설명되는지 확인한다(1자리 반올림 기준)."""
    reported = {
        "T1 novice Deformed (Retargeting)": (43.3, 26, 60), "T1 novice Grasp fail (Retargeting)": (13.3, 8, 60),
        "T1 novice Deformed (w/o)": (15.0, 9, 60), "T1 novice Grasp fail (w/o)": (76.7, 46, 60),
        "T1 novice Grasp fail (w/)": (6.7, 4, 60), "T1 skilled Grasp fail (w/o)": (71.7, 43, 60),
        "T1 skilled Grasp fail (w/)": (3.3, 2, 60), "T2 novice Success (w/o)": (46.7, 7, 15),
        "T2 novice Success (Retargeting)": (40.0, 6, 15), "T4 strict (w/o)": (55.8, 29, 52),
        "T4 strict (w/)": (76.9, 40, 52), "T4 tolerant (w/o)": (65.4, 34, 52), "T4 tolerant (w/)": (90.4, 47, 52),
    }
    out = {}
    for name, (pct, k, n) in reported.items():
        out[name] = {"pct_reported": pct, "k_over_n": f"{k}/{n}", "consistent": round(100 * k / n, 1) == pct}
    out["all_checked_consistent"] = all(v["consistent"] for v in out.values())
    return out


def main():
    res = {}
    res["denominator_check"] = check_denominators()

    # 표 1 (foam-cup): 모든 퍼센트가 분모 60(15 trials x 4 cups)으로 설명되는지 check_denominators에서 확인
    t1 = {
        "novice_grasp_fail": {"Retargeting": (13.3, 8), "w/o Current": (76.7, 46), "w/ Current": (6.7, 4)},
        "skilled_grasp_fail": {"Retargeting": (15.0, 9), "w/o Current": (71.7, 43), "w/ Current": (3.3, 2)},
        "novice_deformed": {"Retargeting": (43.3, 26), "w/o Current": (15.0, 9), "w/ Current": (0.0, 0)},
        "skilled_deformed": {"Retargeting": (25.0, 15), "w/o Current": (16.7, 10), "w/ Current": (0.0, 0)},
    }
    res["T1_per_cup_counts_assumed_n60"] = {
        key: {m: row(m, k, 60) for m, (_, k) in v.items()} for key, v in t1.items()}
    res["T1_fisher_grasp_fail_w_vs_wo"] = {
        "novice": pair("novice grasp fail: w/o - w/ (pp)", ("w/o", 46, 60), ("w/", 4, 60)),
        "skilled": pair("skilled grasp fail: w/o - w/ (pp)", ("w/o", 43, 60), ("w/", 2, 60)),
    }

    res["T1_extra_pairs"] = {
        "novice_deformed_retarget_minus_w": pair("novice deformed: Retargeting - w/", ("R", 26, 60), ("w/", 0, 60)),
        "novice_grasp_fail_retarget_minus_w": pair("novice grasp fail: w/o - Retargeting", ("w/o", 46, 60), ("R", 8, 60)),
        "skilled_grasp_fail_retarget_minus_w": pair("skilled grasp fail: Retargeting - w/", ("R", 9, 60), ("w/", 2, 60)),
        "skilled_grasp_fail_w_minus_wo": pair("skilled grasp fail: w/o - w/", ("w/o", 43, 60), ("w/", 2, 60)),
    }

    # 표 2 (board wiping, n=15)
    res["T2_success"] = {
        "novice_Retargeting": row("Retargeting", 6, 15), "novice_w/o": row("w/o", 7, 15),
        "novice_w/": row("w/", 15, 15), "skilled_Retargeting": row("Retargeting", 15, 15),
        "skilled_w/o": row("w/o", 9, 15), "skilled_w/": row("w/", 15, 15),
    }
    res["T2_pairs"] = [
        pair("novice w/ - Retargeting", ("w/", 15, 15), ("Retargeting", 6, 15)),
        pair("novice w/ - w/o", ("w/", 15, 15), ("w/o", 7, 15)),
        pair("skilled w/ - w/o", ("w/", 15, 15), ("w/o", 9, 15)),
        pair("skilled w/o - Retargeting (ablation harm)", ("w/o", 9, 15), ("Retargeting", 15, 15)),
    ]

    # 표 3 (bottle holding, 로드별 n=12)
    stable = {"0g": (12, 12), "150g": (7, 10), "250g": (2, 12), "350g": (0, 5)}
    res["T3_stable"] = {}
    for load, (wo, w) in stable.items():
        res["T3_stable"][load] = {
            "w/o": row("w/o", wo, 12), "w/": row("w/", w, 12),
            "fisher_p": round(fisher(w, 12, wo, 12), 4),
            "diff_pp_w_minus_wo": pair(f"{load} w/ - w/o", ("w/", w, 12), ("w/o", wo, 12))}
    res["T3_fell_350g_wo"] = row("w/o fell 350g", 12, 12)

    # 표 4 (single-card picking, n=52)
    res["T4_strict"] = pair("strict success w/ - w/o", ("w/", 40, 52), ("w/o", 29, 52))
    res["T4_tolerant"] = pair("tolerant success w/ - w/o", ("w/", 47, 52), ("w/o", 34, 52))
    res["T4_rows"] = {
        "strict_w/o": row("strict w/o", 29, 52), "strict_w/": row("strict w/", 40, 52),
        "tolerant_w/o": row("tolerant w/o", 34, 52), "tolerant_w/": row("tolerant w/", 47, 52),
    }
    # 표 4 실패 유형: 2+ cards 와 missed 는 tolerant 실패 중 유형(합 = 100 - tolerant)
    res["T4_failure_modes_check"] = {
        "w/o": {"sum_pct": round(7.7 + 26.9, 1), "expected_tolerant_fail_pct": round(100 - 65.4, 1)},
        "w/": {"sum_pct": round(0.0 + 9.6, 1), "expected_tolerant_fail_pct": round(100 - 90.4, 1)},
    }
    res["T4_n_needed_per_arm_80pct_power"] = n_per_arm(0.558, 0.769)
    res["T4_n_needed_per_arm_80pct_power_tolerant"] = n_per_arm(0.654, 0.904)

    # 표 7 (M 선택, n=10)
    res["T7_grasp_hold"] = {
        f"M={m}": {"grasp": row("grasp", g, 10), "hold": row("hold", h, 10)}
        for m, (g, h) in {1: (4, 9), 2: (7, 10), 3: (9, 10), 4: (10, 7)}.items()}

    # 표 8, 9 (KL 가중치, n=10)
    res["T8_fist"] = {
        "lambda0_vs_1e-5": pair("fist closing 0 - 1e-5", ("0", 10, 10), ("1e-5", 6, 10)),
        "lambda0_vs_1e-4": pair("fist closing 0 - 1e-4", ("0", 10, 10), ("1e-4", 3, 10)),
        "lambda0_vs_1e-6": pair("fist closing 0 - 1e-6", ("0", 10, 10), ("1e-6", 8, 10)),
    }
    res["T9_bottle_grasp"] = {
        "lambda0_vs_1e-5": pair("bottle grasp 0 - 1e-5", ("0", 9, 10), ("1e-5", 10, 10)),
    }
    res["T8_n_needed_per_arm_80pct_power_0_vs_1e-5"] = n_per_arm(1.0, 0.6)

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
