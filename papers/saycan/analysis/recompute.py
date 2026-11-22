"""SayCan (arXiv:2204.01691 v2) 표 재계산과 검정. 논문 표 값만 쓰며 LLM·로봇은 호출하지 않는다.

실행: python3 analysis/recompute.py  → analysis/recompute_results.json (수 초, numpy·scipy)

- 표 2, 3, 6: 정수 계수(k/n)로 역산해 합계·퍼센트를 검사한다(K1–K6).
- 검정: 같은 101개 지시문을 쓰므로 쌍체 정보(지시문별 성공 여부)는 논문에 없다. 모든 비교는 비쌍체 Fisher 정확 검정이며, 쌍체 상관이 양(+)이면 실제보다 보수적이다.
- 구간: Wilson 95%.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
N_TOTAL = 101
FAM = ["NL Single", "NL Nouns", "NL Verbs", "Structured", "Embodiment", "Crowd", "Long-Horizon"]
N_FAM = [15, 15, 15, 15, 11, 15, 15]

# 표 2 (퍼센트, 행 순서는 FAM)
T2 = {
    "mock_plan": [100, 67, 100, 93, 64, 87, 73],
    "mock_exec": [100, 47, 93, 87, 55, 87, 47],
    "kit_plan": [93, 60, 93, 93, 64, 73, 73],
    "kit_exec": [87, 40, 73, 47, 55, 60, 47],
    "novf_plan": [73, 53, 87, 93, 18, 67, 67],
    "gen_plan": [87, 53, 93, 100, 36, 80, 60],
    "bcusE_exec": [60, 0, 0, 0, 0, 0, 0],
    "bcnl_exec": [0, 0, 0, 0, 0, 0, 0],
}
T2_TOT = {"mock_plan": 84, "mock_exec": 74, "kit_plan": 81, "kit_exec": 60, "novf_plan": 67,
          "gen_plan": 74, "bcusE_exec": 9, "bcnl_exec": 0}
# 표 3 (FLAN-SayCan; PaLM 열은 표 2 mock과 같음)
T3_FLAN = {"plan": [67, 60, 80, 100, 64, 73, 47], "exec": [67, 53, 67, 87, 55, 67, 33]}
T3_FLAN_TOT = {"plan": 70, "exec": 61}
# 표 6 (생성 결과만, 가치함수 없음 · USE 투영)
T6 = {"PaLM 540B": [87, 53, 93, 100, 36, 80, 60], "PaLM 62B": [73, 47, 100, 100, 27, 73, 73],
      "PaLM 8B": [20, 20, 60, 67, 27, 47, 20], "FLAN 137B": [40, 40, 87, 73, 0, 47, 0]}
T6_TOT = {"PaLM 540B": 74, "PaLM 62B": 72, "PaLM 8B": 38, "FLAN 137B": 43}


def pct_to_count(p, n):
    """퍼센트 p와 분모 n에서 round(100k/n) == p인 k를 찾는다. 여러 개면 모두 반환."""
    return [k for k in range(n + 1) if round(100 * k / n) == p]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def fisher(k1, n1, k2, n2):
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(stats.fisher_exact(table, alternative="two-sided")[1])


def family_counts(name, pcts):
    """행별 k를 역산하고 합계를 낸다. 해가 유일하지 않으면 None."""
    ks, ambiguous = [], []
    for p, n in zip(pcts, N_FAM):
        cand = pct_to_count(p, n)
        if len(cand) != 1:
            ambiguous.append((p, n, cand))
        ks.append(cand[0] if len(cand) == 1 else None)
    total = sum(k for k in ks if k is not None) if all(k is not None for k in ks) else None
    return {"rows": ks, "sum": total, "ambiguous": ambiguous}


def main():
    res = {"checks": {}, "tests": {}, "wilson": {}, "notes": {}}

    # K1: 표 2 행 합계와 총계 검사
    for key, pcts in T2.items():
        fc = family_counts(key, pcts)
        n_tot = N_TOTAL
        rep = T2_TOT[key]
        if fc["sum"] is None:
            res["checks"][key] = {"status": "ambiguous", "rows": fc}
            continue
        implied = round(100 * fc["sum"] / n_tot)
        res["checks"][key] = {
            "row_sum": fc["sum"], "row_sum_pct": round(100 * fc["sum"] / n_tot, 1),
            "reported_total_pct": rep, "rounded_from_rows": implied,
            "match": implied == rep,
            "reported_total_count_candidates": pct_to_count(rep, n_tot),
        }

    # K2: 표 3 FLAN, K3: 표 6
    for label, pcts, tot in [("T3_FLAN_plan", T3_FLAN["plan"], T3_FLAN_TOT["plan"]),
                             ("T3_FLAN_exec", T3_FLAN["exec"], T3_FLAN_TOT["exec"])] + \
                            [(f"T6_{k}", v, T6_TOT[k]) for k, v in T6.items()]:
        fc = family_counts(label, pcts)
        implied = round(100 * fc["sum"] / N_TOTAL) if fc["sum"] is not None else None
        res["checks"][label] = {"row_sum": fc["sum"], "implied_pct": implied, "reported_total_pct": tot,
                                "match": implied == tot}

    # K4: 표 5 (언어 시뮬, 50 지시문) — 본문 54%와 표 52% 비교
    t5 = {"0 examples (no termination)": 52, "0 examples (termination)": 10,
          "1 example (no termination)": 74, "1 example (termination)": 64,
          "2 examples (no termination)": 76, "2 examples (termination)": 68,
          "4 examples (no termination)": 84, "4 examples (termination)": 82,
          "8 examples (no termination)": 80, "8 examples (termination)": 80,
          "17 examples (no termination)": 88, "17 examples (termination)": 88}
    res["checks"]["T5_sim"] = {k: {"pct": v, "count_of_50": 50 * v / 100} for k, v in t5.items()}
    res["notes"]["T5_text_vs_table"] = "본문 0예시 무종료 54%(27/50 가정) 대 표 5의 52%(26/50)"

    # K5: 표 7 드로어: 3개 지시문, 실행 0.0, 0.83, 0.17. 6회씩이면 18회로 맞는다.
    drawer_exec = [0.0, 0.83, 0.17]
    res["checks"]["T7_drawer"] = {
        "mean_of_three": round(sum(drawer_exec) / 3, 3),
        "trials_if_6_each": 18, "successes_if_6_each": 6,
        "text_claim_queries": 21, "divisible_21_by_3": 21 % 3 == 0,
        "note": "0.83 ≈ 5/6, 0.17 ≈ 1/6. 본문 "21 queries"와 맞지 않는다. 시행 7회씩이면 0.83과 0.17을 만들 수 없다(5/7 = 0.71, 6/7 = 0.86).",
    }

    # K6: 표 8 다국어: 12행 중 실패 1행(9번째: 프랑스어 'J'ai renversé…')
    t8 = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
    res["checks"]["T8_multilingual"] = {
        "english_3": f"{sum(t8[:3])}/3", "non_english_9": f"{sum(t8[3:])}/9",
        "fisher_p_english_vs_nonenglish": round(fisher(3, 3, sum(t8[3:]), 9), 3),
    }

    # K7: 본문 '65% LLM 실패 · 35% affordance 실패'의 분모 검사
    mock_plan_fail = N_TOTAL - 85
    mock_exec_fail = N_TOTAL - 75
    res["checks"]["error_split"] = {
        "plan_failures_mock": mock_plan_fail, "exec_failures_mock": mock_exec_fail,
        "65pct_of_plan_failures": round(0.65 * mock_plan_fail, 2),
        "decomp_17_9_of_exec": {"17/26": round(100 * 17 / 26, 1), "9/26": round(100 * 9 / 26, 1)},
        "note": "계획 실패 16건으로는 65/35가 정수로 안 나온다. 실행 실패 26건 중 17+9면 65.4/34.6으로 맞는다. 원자료 없음",
    }

    # 오류 감소 비율 (본문 '절반 감소', 실행)
    res["checks"]["error_reduction_vs_FLAN"] = {
        "plan_errors_palm_vs_flan": [N_TOTAL - 85, N_TOTAL - 71],
        "plan_error_reduction_pct": round(100 * (1 - (N_TOTAL - 85) / (N_TOTAL - 71)), 1),
        "exec_errors_palm_vs_flan": [N_TOTAL - 75, N_TOTAL - 62],
        "exec_error_reduction_pct": round(100 * (1 - (N_TOTAL - 75) / (N_TOTAL - 62)), 1),
    }

    # 검정 (같은 101 지시문, 비쌍체 Fisher)
    palm_plan, palm_exec = 85, 75
    tests = {
        "mock_plan_PaLM_vs_NoVF": (palm_plan, 68), "mock_plan_PaLM_vs_Gen": (palm_plan, 75),
        "mock_plan_PaLM_vs_FLAN": (palm_plan, 71), "mock_exec_PaLM_vs_FLAN": (palm_exec, 62),
        "mock_vs_kitchen_exec_reported": (75, 61), "mock_vs_kitchen_exec_family_sum": (75, 59),
        "mock_vs_kitchen_plan_reported": (85, 82), "mock_vs_kitchen_plan_family_sum": (85, 80),
        "mock_exec_vs_BC_USE": (75, 9),
    }
    for k, (a, b) in tests.items():
        n_a = N_TOTAL
        res["tests"][k] = {"k1": a, "k2": b, "p_fisher": round(fisher(a, n_a, b, n_a), 4)}

    # 구간
    for lab, k in [("mock_plan", 85), ("mock_exec", 75), ("kit_plan_reported", 82),
                   ("kit_exec_reported", 61), ("novf_plan", 68), ("gen_plan", 75), ("flan_plan", 71),
                   ("flan_exec", 62), ("bc_usE_exec", 9)]:
        res["wilson"][lab] = {"k": k, "pct": round(100 * k / N_TOTAL, 1), "ci95": wilson(k, N_TOTAL)}

    # 가족 단위 검정: 동사 대 명사, 구현(embodiment) 대 나머지 (계획, 모의)
    verb_nouns = {"verbs_15_15_vs_nouns_10_15": (15, 15, 10, 15)}
    k_emb = 7
    rest_k = 85 - k_emb
    res["tests"]["family_verbs_vs_nouns_plan"] = {"p_fisher": round(fisher(15, 15, 10, 15), 4)}
    res["tests"]["family_embodiment_vs_rest_plan"] = {
        "k_emb": f"{k_emb}/11", "k_rest": f"{rest_k}/90", "p_fisher": round(fisher(k_emb, 11, rest_k, 90), 4)}
    res["tests"]["family_structured_vs_verbs_plan"] = {
        "p_fisher": round(fisher(14, 15, 15, 15), 4)}

    # 검출력: 두 독립 비율, n = 101, 양측 0.05, 검정력 0.8 (정규 근사)
    def mde(p, n=101, za=1.96, zb=0.8416):
        return round(100 * (za + zb) * math.sqrt(2 * p * (1 - p) / n), 1)
    res["power_mde_pp"] = {"at_p0.75": mde(0.75), "at_p0.84": mde(0.84), "at_p0.50": mde(0.5)}

    # 명목 비율 주장 확인
    res["notes"]["nearly_doubles_claim"] = {
        "plan_vs_NoVF_ratio": round(85 / 68, 2), "plan_vs_Gen_ratio": round(85 / 75, 2),
        "exec_vs_BCUSE_ratio": round(75 / 9, 2),
        "note": "계획 성공률 기준 상대 차이는 1.13–1.25배다. '거의 두 배'는 실행 대 BC USE(9%)에서만 성립한다.",
    }

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res["checks"], ensure_ascii=False, indent=1)[:4000])
    print("tests:", json.dumps(res["tests"], ensure_ascii=False, indent=1))
    print("wilson:", json.dumps(res["wilson"], ensure_ascii=False))
    print("power:", res["power_mde_pp"])


if __name__ == "__main__":
    main()
