"""논문 표에 적힌 개수와 보고 값을 다시 계산하고 내부 일관성을 검사한다 (CPU, 표준 라이브러리만 사용).

입력: 논문 표의 숫자를 여기에 옮겨 적은 값 (Table 4, 5, 6, S5, S7, S8, S9, S10, 3, 8).
출력: analysis/recompute_results.json
"""
import json
import math
from pathlib import Path

OUT = Path(__file__).with_name("recompute_results.json")
N = 250  # 쌍체 비교의 에피소드 수 (Section 5, Appendix C)


def exact_mcnemar_two_sided(b: int, c: int) -> float:
    """정확 이항 McNemar 양측 p. b, c는 불일치쌍 수."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wald_matched_ci(b: int, c: int, n: int = N, z: float = 1.96):
    """쌍체 Wald 95% 구간. diff = (c - b) / n, 여기서 b = 성공→실패, c = 실패→성공."""
    diff = (c - b) / n
    var = (b + c - (c - b) ** 2 / n) / n ** 2
    half = z * math.sqrt(var)
    return diff, diff - half, diff + half


def check_paired(name, b, c, reported_diff, reported_ci, reported_p, n=N):
    diff, lo, hi = wald_matched_ci(b, c, n)
    p = exact_mcnemar_two_sided(b, c)
    return {
        "name": name,
        "s_to_f": b,
        "f_to_s": c,
        "discordant": b + c,
        "diff_pp_recomputed": round(100 * diff, 2),
        "diff_pp_reported": reported_diff,
        "ci_pp_recomputed": [round(100 * lo, 2), round(100 * hi, 2)],
        "ci_pp_reported": reported_ci,
        "p_recomputed": p,
        "p_reported": reported_p,
    }


results = {"paired_closed_loop": [], "checks": {}}

# Table 4 / Table S5 (성공 50/50 시도, 총 250 에피소드, 기준 113)
results["paired_closed_loop"] = [
    check_paired("endpoint mask (0,49)", 14, 22, 3.2, [-1.5, 7.9], 0.24),
    check_paired("endpoint value zero (0,49)", 21, 28, 2.8, [-2.7, 8.3], 0.39),
    check_paired("endpoint replacement d99", 29, 20, -3.6, [-9.1, 1.9], 0.25),
    check_paired("endpoint replacement d100", 36, 18, -7.2, [-12.9, -1.5], 0.020),
    check_paired("endpoint replacement d101", 30, 22, -3.2, [-8.8, 2.4], 0.33),
    check_paired("interior replacement (2,3) d99", 76, 5, -28.4, [-34.5, -22.3], 2.3e-17),
    check_paired("interior replacement (2,4) d99", 72, 9, -25.2, [-31.5, -18.9], 2.5e-13),
    # Table 6 / Table S7 (BOS, 기준 107)
    check_paired("BOS mask", 17, 26, 3.6, [-1.5, 8.7], 0.22),
    check_paired("BOS value zero (run A, 기준 107)", 17, 50, 13.2, [7.0, 19.4], 6.7e-5),
    check_paired("BOS replacement (run A)", 17, 17, 0.0, [-4.6, 4.6], 1.0),
    # Appendix E: 두 번째 실행, 기준 114 (λ=1). 94 공통 성공, 20 λ=1만, 35 λ=0만
    check_paired("BOS scale λ=0 (run B, 기준 114)", 20, 35, 6.0, [0.2, 11.8], 0.058),
    # Appendix E: 반전 과제 5개 (Table S8), 기준 99
    check_paired("reversed tasks value zero vs unmodified", 6, 25, 7.6, None, 8.8e-4),
]

# 합계와 표 내부 일관성 검사
def total_check(label, per_task, reported_total):
    s = sum(per_task)
    return {"label": label, "sum": s, "reported": reported_total, "ok": s == reported_total}


checks = [
    total_check("Table 4 base per task", [19, 26, 30, 34, 4], 113),
    total_check("Table 4 mask per task", [24, 25, 32, 33, 7], 121),
    total_check("Table 4 value zero per task", [22, 22, 33, 35, 8], 120),
    total_check("Table 4 replacement d99 per task", [16, 17, 37, 26, 8], 104),
    total_check("Table 4 interior (2,3) per task", [9, 4, 28, 1, 0], 42),
    total_check("Table 4 interior (2,4) per task", [10, 7, 32, 1, 0], 50),
    total_check("Table 6 base per task", [28, 23, 19, 32, 5], 107),
    total_check("Table 6 mask per task", [29, 24, 22, 35, 6], 116),
    total_check("Table 6 value zero per task", [35, 30, 33, 32, 10], 140),
    total_check("Table 6 replacement per task", [31, 23, 17, 31, 5], 107),
    total_check("Table S8 value zero (reversed)", [45, 28, 8, 16, 21], 118),
    total_check("Table S8 unmodified (reversed)", [46, 21, 1, 11, 20], 99),
    total_check("Table S9 lambda=0", [32, 26, 31, 32, 8], 129),
    total_check("Table S9 lambda=0.5", [30, 29, 32, 34, 3], 128),
    total_check("Table S9 lambda=1", [22, 26, 29, 34, 3], 114),
    total_check("Table S10 baseline (run A)", [28, 23, 5, 19, 32], 107),
    total_check("Table S10 lambda=1 (run B)", [29, 26, 3, 22, 34], 114),
    total_check("Table S10 baseline only", [1, 3, 2, 2, 2], 10),
    total_check("Table S10 lambda=1 only", [2, 6, 0, 5, 4], 17),
]
results["checks"]["sums"] = checks

# 교차 검사: 두 실행의 기준 차이(107 대 114)가 +13.2와 +6.0의 차이를 설명할 수 있는지
results["checks"]["run_a_vs_run_b"] = {
    "value_zero_run_a_minus_baseline_run_a": 140 - 107,
    "lambda0_run_b_minus_baseline_run_b": 129 - 114,
    "baseline_gap_between_runs": 114 - 107,
    "baseline_discordant_between_runs": {"baseline_only": 10, "lambda1_only": 17},
    "note": "두 실행의 기준 성공 수가 7 차이나고, 불일치 27 에피소드. 쌍체 차이 +13.2와 +6.0은 서로 다른 실행의 기준에 묶여 있다.",
}

# Table 3: 세 단계 평균과 보고 값 (행 순서는 high, mid, last로 가정, 평균은 독립 검사)
t3 = {
    "r0": [1.125, 1.014, 0.965, 1.034],
    "r49": [0.996, 1.027, 1.033, 1.019],
    "c0": [2.122, 1.457, 1.129, 1.569],
    "c49": [1.576, 1.695, 1.197, 1.489],
}
results["checks"]["table3_means"] = {
    k: {"mean_of_three_steps": round(sum(v[:3]) / 3, 4), "reported_mean": v[3]} for k, v in t3.items()
}

# 끝점 질량 식 (2): 균일 기준 2/50 = 0.040, 전체 평균 5.4%와 1.4 pp 초과
results["checks"]["uniform_edge_mass"] = {"2_over_50": 2 / 50, "reported_uniform": 0.040,
                                           "excess_pp_reported": 5.4 - 4.0}

# Table 8 / Appendix F: 3개 과제, 150 에피소드. 기준 58, 공유 라우팅 64. 불일치쌍 수는 보고되지 않음.
# 보고된 p = 0.35를 주는 불일치쌍 수를 역산한다 (순 차이 = +6 고정).
def p_for_discordant(n_disc, net=6):
    c = (n_disc - net) // 2
    b = n_disc - c
    return exact_mcnemar_two_sided(b, c)

candidates = [(n, round(p_for_discordant(n), 4)) for n in range(6, 60, 2)]
closest = min(candidates, key=lambda t: abs(t[1] - 0.35))
results["checks"]["table8_shared_vs_base"] = {
    "episodes": 150, "base": 58, "shared": 64, "net": 6,
    "diff_pp": round(100 * 6 / 150, 2), "reported_ci_pp": [-2.9, 10.9], "reported_p": 0.35,
    "discordant_not_reported": True,
    "min_possible_p_with_net6": exact_mcnemar_two_sided(6, 0),
    "discordant_giving_p_near_0.35": {"n_discordant": closest[0], "p": closest[1]},
}

# Appendix F의 과제별 변화 +8, +2, +2 pp (50 에피소드씩) 합은 +6 에피소드와 일치하는지
results["checks"]["table8_per_task_pp"] = {"per_task_pp": [8, 2, 2],
                                          "episodes_equiv": [8 * 50 / 100, 2 * 50 / 100, 2 * 50 / 100],
                                          "sum_episodes": 4 + 1 + 1, "reported_total_net": 64 - 58}

results["checks"]["donor_disagreement_note"] = "donor 99 대 100 라벨 불일치 9.2%, 99 대 101 10.8%, 100 대 101 8.0% (Appendix B.1)"

OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

for r in results["paired_closed_loop"]:
    print(f'{r["name"]:42s} diff {r["diff_pp_recomputed"]:+6.2f} (rep {r["diff_pp_reported"]:+.1f})  '
          f'CI {r["ci_pp_recomputed"]} (rep {r["ci_pp_reported"]})  p {r["p_recomputed"]:.3g} (rep {r["p_reported"]})')
print()
for c in checks:
    print(c["label"], c["sum"], c["reported"], "OK" if c["ok"] else "MISMATCH")
print()
print(json.dumps({k: results["checks"][k] for k in ["table3_means", "table8_shared_vs_base", "run_a_vs_run_b"]}, ensure_ascii=False, indent=1))
