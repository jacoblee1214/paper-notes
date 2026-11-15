"""논문 본문과 부록 표의 수치를 다시 계산하고 표 사이 일관성을 검사한다.

입력: 논문 PDF에서 옮긴 표 값(아래 상수). 출처: arXiv:2610.30735v1 Table 2, Table 3, Table C1, Table E1, E2, Table 1, Table A1.
출력: analysis/table_check_results.json
실행: python3 analysis/table_check.py   (numpy, scipy, CPU, 수 초)
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "table_check_results.json"
Z90 = stats.norm.ppf(0.95)  # 90% 구간 (TOST와 같은 수준, 양측 5%씩)
DELTA_PP = 3.0  # 논문 동등성 한계 (pp)

# ---- Table 2 (시뮬레이션 최종 값, %). (reference, 1-step). n은 본문 3.2절에서 알 수 있는 셀만 기입 ----
N_LIBERO_SGO = 1500  # 30 tasks x 50 trials (본문 3.2)
N_LIBERO_10 = 500  # 10 tasks x 50 trials
N_CALVIN = 1000  # 1000 chains (SR@5)
N_ROBOTWIN = 5000  # 50 tasks x 100 episodes per regime (본문 3.2)
TABLE2 = {
    "DiT": {"LIBERO SGO": (89.9, 91.9, N_LIBERO_SGO), "LIBERO-10": (68.4, 53.8, N_LIBERO_10)},
    "SmolVLA": {"LIBERO SGO": (72.2, 82.0, N_LIBERO_SGO), "LIBERO-10": (34.6, 44.2, N_LIBERO_10)},
    "X-VLA": {"LIBERO SGO": (97.6, 97.3, N_LIBERO_SGO), "LIBERO-10": (96.2, 96.0, N_LIBERO_10),
              "CALVIN SR@5": (71.0, 70.0, N_CALVIN), "RoboTwin clean": (65.5, 65.1, N_ROBOTWIN),
              "RoboTwin rand.": (24.3, 23.3, N_ROBOTWIN)},
    "FLOWER": {"LIBERO SGO": (97.3, 96.5, N_LIBERO_SGO), "LIBERO-10": (89.6, 83.6, N_LIBERO_10),
               "CALVIN SR@5": (78.5, 75.5, N_CALVIN)},
    "pi0.5": {"LIBERO SGO": (98.5, 98.9, N_LIBERO_SGO), "LIBERO-10": (94.4, 94.4, N_LIBERO_10)},
    "LAP": {"LIBERO SGO": (97.7, 98.8, N_LIBERO_SGO), "LIBERO-10": (90.2, 92.0, N_LIBERO_10)},
    "XR-0": {"LIBERO SGO": (97.3, 97.0, N_LIBERO_SGO), "LIBERO-10": (96.8, 97.4, N_LIBERO_10),
             "CALVIN SR@5": (87.2, 88.1, N_CALVIN)},
}
# Table 2의 LIBERO+ 열은 셀당 n이 본문에 없어 제외한다(10 030 instances 전체만 있음).


def wilson(p, n, z=Z90):
    """Wilson score 구간 (비율 p, 표본 n). 반환: (lo, hi), 비율 단위."""
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def newcombe_independent(p_ref, p_1, n):
    """Newcombe 방법 10(독립 표본 버전)으로 Δ = p_ref − p_1 의 90% 구간. 입력과 출력은 비율."""
    l_r, u_r = wilson(p_ref, n)
    l_1, u_1 = wilson(p_1, n)
    d = p_ref - p_1
    lo = d - math.sqrt((p_ref - l_r) ** 2 + (u_1 - p_1) ** 2)
    hi = d + math.sqrt((u_r - p_ref) ** 2 + (p_1 - l_1) ** 2)
    return lo, hi


def tost_verdict(lo_pp, hi_pp, delta=DELTA_PP):
    """논문 식 (B5). 입력은 pp 단위 구간."""
    if lo_pp > -delta and hi_pp < delta:
        return "equivalent"
    if lo_pp > delta or hi_pp < -delta:
        return "different"
    return "inconclusive"


def check_table2():
    rows, tally = [], {"equivalent": 0, "different": 0, "inconclusive": 0}
    for model, cells in TABLE2.items():
        for metric, (ref, one, n) in cells.items():
            d = ref - one
            lo, hi = newcombe_independent(ref / 100, one / 100, n)
            lo_pp, hi_pp = 100 * lo, 100 * hi
            v = tost_verdict(lo_pp, hi_pp)
            tally[v] += 1
            rows.append({"model": model, "metric": metric, "ref": ref, "one_step": one, "n": n,
                         "delta_pp_ref_minus_1step": round(d, 2),
                         "ci90_pp_independent": [round(lo_pp, 2), round(hi_pp, 2)],
                         "verdict_independent_ci": v})
    return rows, tally


# ---- Table 3 (실로봇, 태스크당 n=100) ----
TABLE3 = {  # task: {model: (ref, 1step, reported CI for Δ=ref−1step or None, starred)}
    "Spoon->Drawer": {"DiT": (33, 32, (-8.2, 10.2), False), "X-VLA": (0, 0, (-2.6, 2.6), False),
                      "FLOWER": (46, 50, (-13.8, 5.9), False), "pi0.5": (88, 90, (-7.8, 3.7), False),
                      "LAP": (75, 73, (-4.8, 8.8), False), "XR-0": (73, 69, (-3.1, 11.1), False)},
    "Open Drawer": {"DiT": (59, 64, (-13.5, 3.6), False), "X-VLA": (68, 72, (-9.5, 1.4), False),
                    "FLOWER": (81, 97, (-22.9, -10.0), True), "pi0.5": (78, 77, (-3.8, 5.8), False),
                    "LAP": (78, 76, (-3.0, 7.1), False), "XR-0": (81, 83, (-6.7, 2.5), False)},
    "Open Oven": {"DiT": (0, 1, (-4.4, 1.7), False), "X-VLA": (0, 0, (-2.6, 2.6), False),
                  "FLOWER": (20, 31, (-20.0, -1.8), False), "pi0.5": (63, 73, (-19.7, -0.1), False),
                  "LAP": (56, 47, (1.1, 16.7), False), "XR-0": (42, 42, (-9.6, 9.6), False)},
    "Close Oven": {"DiT": (22, 23, (-7.2, 5.2), False), "X-VLA": (49, 50, (-7.5, 5.5), False),
                   "FLOWER": (92, 75, (9.4, 24.8), True), "pi0.5": (87, 74, (6.4, 19.9), True),
                   "LAP": (88, 86, (-2.6, 6.8), False), "XR-0": (88, 88, (-5.3, 5.3), False)},
}
TABLE3_POOLED = {  # model: (ref_pooled_reported, 1step_pooled_reported, reported pooled CI)
    "DiT": (28.5, 30.0, (-5.0, 2.0)), "X-VLA": (29.2, 30.5, (-3.4, 0.9)),
    "FLOWER": (59.8, 63.2, (-7.7, 0.7)), "pi0.5": (79.0, 78.5, (-3.0, 4.0)),
    "LAP": (74.2, 70.5, (0.7, 6.8)), "XR-0": (71.0, 70.5, (-2.9, 3.9)),
}


def exact_mcnemar_min_p(net):
    """불일치쌍 수가 |net|뿐이고 모두 한쪽일 때의 최소 양측 정확 p (식 B6). b+c = |net|."""
    k = abs(net)
    if k == 0:
        return 1.0
    return min(1.0, 2 * 0.5 ** k)


def check_table3():
    tasks, reproduced = [], []
    for task, models in TABLE3.items():
        for model, (ref, one, rep_ci, starred) in models.items():
            d = ref - one  # 실로봇 Δ = ref − 1step (pp), 정수 개수 차이 = b−c
            lo, hi = newcombe_independent(ref / 100, one / 100, 100)
            ours = [round(100 * lo, 1), round(100 * hi, 1)]
            tasks.append({"task": task, "model": model, "ref": ref, "one_step": one,
                          "net_count_ref_minus_1step": d,
                          "reported_ci90": list(rep_ci), "independent_newcombe_ci90": ours,
                          "starred_reported": starred,
                          "min_exact_mcnemar_p_if_all_discordant_one_sided": exact_mcnemar_min_p(d)})
            reproduced.append(abs(ours[0] - rep_ci[0]) <= 1.5 and abs(ours[1] - rep_ci[1]) <= 1.5)
    pooled = []
    for model, (ref_rep, one_rep, ci_rep) in TABLE3_POOLED.items():
        ref_mean = np.mean([TABLE3[t][model][0] for t in TABLE3])
        one_mean = np.mean([TABLE3[t][model][1] for t in TABLE3])
        lo, hi = newcombe_independent(ref_mean / 100, one_mean / 100, 400)
        pooled.append({"model": model, "ref_mean_of_tasks": ref_mean, "ref_reported": ref_rep,
                       "one_step_mean_of_tasks": one_mean, "one_step_reported": one_rep,
                       "delta_pp": round(ref_mean - one_mean, 2),
                       "ci90_independent_pp": [round(100 * lo, 1), round(100 * hi, 1)],
                       "ci90_reported_pp": list(ci_rep)})
    return tasks, reproduced, pooled


def check_consistency():
    """표 사이 값 비교. 불일치는 목록으로 남긴다."""
    issues = []
    # (1) π0.5 LIBERO+ : Table 2 와 Table E2 비교
    issues.append({"item": "pi0.5 LIBERO+ SGO", "table2": [86.4, 86.3], "tableE2": [86.3, 86.2]})
    issues.append({"item": "pi0.5 LIBERO+ 10", "table2": [77.6, 76.1], "tableE2": [77.7, 76.3]})
    # (2) FLOWER CALVIN: Table 2 (H=10 기본) 와 Table C1
    issues.append({"item": "FLOWER CALVIN SR@5 (H=10)", "table2": [78.5, 75.5], "tableC1": [79.5, 77.0]})
    issues.append({"item": "FLOWER CALVIN avg len (H=10)", "table2": [4.51, 4.43], "tableC1": [4.55, 4.43]})
    # (3) LAP LIBERO SGO: 본문 "차이가 1 pp 안" 과 표 값
    lap_sgo_diff = round(98.8 - 97.7, 2)
    issues.append({"item": "LAP LIBERO SGO |1-step − ref|", "value_pp": lap_sgo_diff,
                   "text_claim": "within a percentage point (Sec 4.1)", "consistent": lap_sgo_diff <= 1.0})
    # (4) Table 1 속도 향상 비율 재계산
    lat = {"DiT": (30, 15, 2.0), "SmolVLA": (436, 91, 4.8), "X-VLA": (215, 70, 3.1), "FLOWER": (140, 60, 2.3),
           "pi0.5": (80, 50, 1.6), "LAP": (60, 35, 1.7), "XR-0": (170, 90, 1.9)}
    speed = {m: {"recomputed": round(a / b, 2), "reported": r, "ok": abs(a / b - r) < 0.06}
             for m, (a, b, r) in lat.items()}
    # (5) Table A1: 'SmolVLA 3–4× more curved' 의 비율 재계산
    excess = {"DiT": 0.59, "SmolVLA": 2.35, "X-VLA": 0.005, "FLOWER": 0.24, "pi0.5": 0.25, "LAP": 0.27, "XR-0": 0.64}
    others = [v for k, v in excess.items() if k != "SmolVLA" and v > 0.01]
    smol_ratio = {"vs_min": round(2.35 / min(others), 1), "vs_max": round(2.35 / max(others), 1),
                  "text_claim": "3-4x (Sec A.2)"}
    # (6) 평가 규모 검산
    scale = {"LIBERO trials per variant": 40 * 50, "RoboTwin paired episodes per regime": 50 * 100,
             "real-world rollouts": 4 * 100 * 2 * 6, "real-world pooled n": 4 * 100}
    # (7) Table 3 pooled 평균 검산은 check_table3 에서 수행
    return issues, speed, smol_ratio, scale


def main():
    t2_rows, t2_tally = check_table2()
    t3_tasks, t3_repro, t3_pooled = check_table3()
    issues, speed, smol_ratio, scale = check_consistency()
    n_cells = sum(len(v) for v in TABLE2.values())
    result = {
        "source": "arXiv:2610.30735v1, Tables 1, 2, 3, A1, C1, E2",
        "method_note": ("모든 구간은 독립 표본 Newcombe(Wilson 90%)로 계산. 논문은 쌍체 Newcombe(상관 항 phi 포함)을 쓰므로 "
                        "여기 구간은 보수적(넓음)이다. 쌍체 b, c 개수는 논문에 없어 계산할 수 없다."),
        "table2_cells_with_known_n": n_cells,
        "table2_verdict_counts_independent_ci": t2_tally,
        "table2_rows": t2_rows,
        "table2_text_counts": {"equivalent": 23, "1-step better": 2, "multi-step better": 4, "inconclusive": 7, "total": 36},
        "table3_tasks": t3_tasks,
        "table3_reported_ci_reproduced_within_1.5pp": f"{sum(t3_repro)}/{len(t3_repro)}",
        "table3_pooled": t3_pooled,
        "consistency_issues": issues,
        "table1_speedup_check": speed,
        "tableA1_smolvla_ratio": smol_ratio,
        "scale_checks": scale,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT)
    print("table2 tally (independent CI):", t2_tally, "over", n_cells, "cells with known n")
    print("table3 CI reproduction:", f"{sum(t3_repro)}/{len(t3_repro)}")


if __name__ == "__main__":
    main()
