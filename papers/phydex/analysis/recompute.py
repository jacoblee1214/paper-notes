"""PhyDEX (arXiv:2611.26565v1) 논문 표·그림 수치를 재계산하고 검정한다. 재현(학습·실행)은 하지 않는다.

입력: 논문 Table 1–4, Figure 4, Figure 7의 값을 `paper/paper.txt`와 PDF 렌더링에서 옮겨 적은 값.
출력: analysis/recompute_results.json, 표준 출력 요약.
실행: python3 analysis/recompute.py  (수 초)

가정: 각 셀은 50 rollouts (논문 §4.1, §4.4, Table 3·4 설명). 다른 n은 명시한다.
"""
import json
import math
from pathlib import Path

from scipy.stats import fisher_exact

OUT = Path(__file__).resolve().parent / "recompute_results.json"
N = 50  # 셀당 rollout 수 (논문 §4.1: "evaluate each policy over 50 rollouts")

POLICIES = ["GR00T N1.7", "DynamicVLA", "ForceVLA"]

# Table 1: (object, destination) -> pct per policy (순서: POLICIES)
TABLE1 = {
    ("Uniform", "Single"): [6, 0, 10],
    ("Uniform", "Multiple"): [2, 0, 10],
    ("Uniform", "Dynamic"): [0, 0, 0],
    ("Irregular", "Single"): [0, 0, 0],
    ("Irregular", "Multiple"): [0, 2, 0],
    ("Irregular", "Dynamic"): [0, 0, 0],
    ("Dynamic", "Single"): [0, 0, 0],
    ("Dynamic", "Multiple"): [0, 0, 2],
    ("Dynamic", "Dynamic"): [0, 0, 0],
}
TABLE1_REPORTED_AVG = [0.9, 0.2, 2.4]

# Table 2 (정적 단일 목적지, 3 객체 평균). α -> object -> pct per policy
TABLE2 = {
    1.0: {"Uniform": [6, 0, 10], "Irregular": [0, 0, 0], "Dynamic": [0, 0, 0]},
    0.5: {"Uniform": [6, 2, 16], "Irregular": [6, 4, 10], "Dynamic": [2, 0, 8]},
    0.0: {"Uniform": [18, 6, 44], "Irregular": [10, 4, 40], "Dynamic": [0, 2, 22]},
}
TABLE2_REPORTED_AVG = {1.0: [2, 0, 3.3], 0.5: [4.7, 2, 11.3], 0.0: [9.3, 4, 35.3]}

# Table 3 (휴머노이드, 2,400 에피소드). object -> pct per policy
TABLE3 = {"Uniform": [10, 2, 14], "Irregular": [8, 0, 6], "Dynamic": [0, 0, 6]}
TABLE3_REPORTED_AVG = [6, 0.7, 8.7]

# Table 4 (실물, 3 큐브). cube -> pct per policy
TABLE4 = {"Cube 1": [16, 0, 58], "Cube 2": [12, 0, 56], "Cube 3": [4, 0, 40]}
TABLE4_REPORTED_AVG = [10.7, 0, 51.3]

# Figure 4 (균일 객체, 정적 단일 목적지로 보임: Fig. 1 캡션과 Table 1 값 일치)
FIG4_WITHOUT = {"DynamicVLA": 0, "GR00T N1.7": 6, "ForceVLA": 10}
FIG4_WITH_GT = {"DynamicVLA": 0, "GR00T N1.7": 12.5, "ForceVLA": 24}

# Figure 7 (실패 모드 비율, %). 순서: grasp, approach, throw, reach
FIG7 = {
    "GR00T N1.7": [44.8, 16.4, 14.9, 23.9],
    "DynamicVLA": [32.4, 32.1, 3.4, 32.1],
    "ForceVLA": [38.4, 17.4, 13.7, 30.4],
}
FIG7_PIE = [38.5, 22.0, 10.6, 28.8]  # 전체 파이 (grasp, approach, throw, reach)

# 본문 수치
TEXT = {
    "grasp_success_share_text": 21.24,   # §4.3 (1): "only about 21.24% of trajectories include a successful grasp"
    "throw_reach_share_text": 21.24,     # §4.3 (2): "an average of 21.24% of trajectories reach the throwing stage"
    "overall_sr_text": 1.2,              # §4.3 (2)
    "throws_success_text": 5.6,          # §4.3 (2)
    "dynamicvla_grasp_text": 3.57,       # §4.3 (1)
    "pre_throw_text": 89,                # §4.4 Failure Analysis
}


def wilson(k, n, z=1.959964):
    """Wilson 95% 구간 (비율, 0–1)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def pct_to_count(pct, n):
    """백분율을 개수로. 정수가 아니면 None (셀 n과 맞지 않음)."""
    k = pct * n / 100
    return int(round(k)) if abs(k - round(k)) < 1e-6 else None


def fisher(k1, n1, k2, n2):
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(fisher_exact(table, alternative="two-sided")[1])


def check_integers(name, table, n=N):
    bad = []
    for key, vals in table.items():
        for pol, v in zip(POLICIES, vals if isinstance(vals, list) else [vals]):
            if pct_to_count(v, n) is None:
                bad.append((str(key), pol, v))
    return {"table": name, "n": n, "non_integer_cells": bad}


def main():
    res = {"assumption_n_per_cell": N}

    # V1. Table 1 셀 정수성 (n=50)과 전체 평균 재계산
    res["V1_table1_integrality"] = check_integers("Table 1", TABLE1)
    counts = {p: 0 for p in POLICIES}
    for vals in TABLE1.values():
        for pol, v in zip(POLICIES, vals):
            counts[pol] += pct_to_count(v, N)
    total_n = N * len(TABLE1)
    pooled = {p: 100 * counts[p] / total_n for p in POLICIES}
    res["V1_table1_pooled"] = {
        "successes": counts, "n_total": total_n,
        "pooled_pct": {p: round(pooled[p], 2) for p in POLICIES},
        "reported_avg": dict(zip(POLICIES, TABLE1_REPORTED_AVG)),
        "unweighted_mean_of_cells_pct": {p: round(sum(v[i] for v in TABLE1.values()) / 9, 2) for i, p in enumerate(POLICIES)},
    }
    res["V1_table1_wilson_overall"] = {
        p: [round(100 * x, 2) for x in wilson(counts[p], total_n)] for p in POLICIES
    }
    all_succ = sum(counts.values())
    res["V1_table1_all_policies"] = {"successes": all_succ, "n": 3 * total_n,
                                     "pct": round(100 * all_succ / (3 * total_n), 2),
                                     "note": "본문 1.2%와 비교 (세 정책 pooled)"}

    # V2. Table 2 평균과 개수
    t2 = {}
    for a, objs in TABLE2.items():
        per_pol = {}
        for i, p in enumerate(POLICIES):
            k = sum(pct_to_count(objs[o][i], N) for o in objs)
            per_pol[p] = {"successes": k, "n": 3 * N, "pct": round(100 * k / (3 * N), 2),
                          "reported_avg": TABLE2_REPORTED_AVG[a][i]}
        t2[str(a)] = per_pol
    res["V2_table2"] = t2

    # V3. Table 3, Table 4 평균과 개수
    t3 = {p: sum(pct_to_count(TABLE3[o][i], N) for o in TABLE3) for i, p in enumerate(POLICIES)}
    t4 = {p: sum(pct_to_count(TABLE4[c][i], N) for c in TABLE4) for i, p in enumerate(POLICIES)}
    res["V3_table3"] = {p: {"successes": t3[p], "n": 150, "pct": round(100 * t3[p] / 150, 2),
                            "reported_avg": TABLE3_REPORTED_AVG[i]} for i, p in enumerate(POLICIES)}
    res["V3_table4"] = {p: {"successes": t4[p], "n": 150, "pct": round(100 * t4[p] / 150, 2),
                            "reported_avg": TABLE4_REPORTED_AVG[i]} for i, p in enumerate(POLICIES)}

    # V4. Figure 4: 사전 대 사후 (GT). n 가정 50과 정수성 확인
    f4 = {}
    for p in ["GR00T N1.7", "ForceVLA"]:
        k0 = pct_to_count(FIG4_WITHOUT[p], N)
        k1 = pct_to_count(FIG4_WITH_GT[p], N)
        f4[p] = {"without_gt_k_of_50": k0, "with_gt_pct": FIG4_WITH_GT[p],
                 "with_gt_k_of_50": k1,
                 "gain_pp": round(FIG4_WITH_GT[p] - FIG4_WITHOUT[p], 2),
                 "ratio": round(FIG4_WITH_GT[p] / FIG4_WITHOUT[p], 2)}
        if k1 is not None:
            f4[p]["fisher_p"] = fisher(k0, N, k1, N)
            f4[p]["wilson_with_gt_pct"] = [round(100 * x, 1) for x in wilson(k1, N)]
    # 12.5%는 50 중 정수가 아니므로 n을 48로 두면 6/48 (민감도 분석)
    f4["GR00T N1.7 sensitivity"] = {"with_gt_k_of_48": 6, "n": 48,
                                    "fisher_p_vs_3_of_50": fisher(3, 50, 6, 48),
                                    "note": "n 미보고. 12.5%는 50회에서 가능한 값이 아님"}
    res["V4_fig4_gt"] = f4

    # V5. 주요 비교 (Fisher 양측). 정책 간 pooled와 F/T 대조
    comps = {
        "sim pooled: ForceVLA vs GR00T (11/450 vs 4/450)": (11, 450, 4, 450),
        "sim pooled: ForceVLA vs DynamicVLA (11/450 vs 1/450)": (11, 450, 1, 450),
        "sim pooled: GR00T vs DynamicVLA (4/450 vs 1/450)": (4, 450, 1, 450),
        "sim alpha=1 single-dest: ForceVLA vs GR00T (5/150 vs 3/150)": (5, 150, 3, 150),
        "sim alpha=0 vs alpha=1, ForceVLA (53/150 vs 5/150)": (53, 150, 5, 150),
        "sim alpha=0 vs alpha=1, GR00T (14/150 vs 3/150)": (14, 150, 3, 150),
        "humanoid: ForceVLA vs GR00T (13/150 vs 9/150)": (13, 150, 9, 150),
        "humanoid: ForceVLA vs DynamicVLA (13/150 vs 1/150)": (13, 150, 1, 150),
        "real: ForceVLA vs GR00T (77/150 vs 16/150)": (77, 150, 16, 150),
        "real: ForceVLA vs DynamicVLA (77/150 vs 0/150)": (77, 150, 0, 150),
        "real cube3: ForceVLA vs GR00T (20/50 vs 2/50)": (20, 50, 2, 50),
        "real cube1: GR00T vs cube3 GR00T (8/50 vs 2/50)": (8, 50, 2, 50),
        "sim uniform single: ForceVLA vs GR00T (5/50 vs 3/50)": (5, 50, 3, 50),
        "sim uniform single: ForceVLA GT vs no GT (12/50 vs 5/50)": (12, 50, 5, 50),
    }
    res["V5_comparisons"] = {k: {"p_fisher": fisher(*v), "k1_n1_k2_n2": list(v)} for k, v in comps.items()}

    # V6. 검출력: n=50 두 비율, 80% 검정력 최소 차이 (정규 근사, 양측 0.05)
    z_a, z_b = 1.959964, 0.841621
    power = {}
    for p0 in [0.02, 0.06, 0.10, 0.20]:
        best = None
        for step in range(1, 1000):
            p1 = min(0.999, p0 + step * 0.001)
            pbar = (p0 + p1) / 2
            se = math.sqrt(2 * pbar * (1 - pbar) / N)
            if (p1 - p0) >= (z_a + z_b) * se * 1.0:
                best = p1
                break
        power[f"p0={p0}"] = {"min_detectable_p1": None if best is None else round(best, 3),
                             "min_detectable_diff_pp": None if best is None else round(100 * (best - p0), 1)}
    res["V6_power_n50"] = power

    # V7. 실패 모드 재계산 (Fig. 7 전체 파이는 정책별 값의 무가중 평균과 일치하는지)
    avg = [round(sum(FIG7[p][i] for p in POLICIES) / 3, 2) for i in range(4)]
    sums = {p: round(sum(FIG7[p]), 1) for p in POLICIES}
    res["V7_fig7"] = {"policy_sums": sums, "unweighted_mean": avg, "pie_reported": FIG7_PIE,
                      "pre_throw_share_pie": round(FIG7_PIE[0] + FIG7_PIE[1] + FIG7_PIE[3], 1)}

    # V8. 21.24% 재구성: Fig. 7과 Table 1 개수로 "던지기 단계 도달" 비율을 역산
    fail_n = {p: N * 9 - counts[p] for p in POLICIES}
    throw_err_k = {p: round(FIG7[p][2] / 100 * fail_n[p]) for p in POLICIES}
    throws_k = {p: throw_err_k[p] + counts[p] for p in POLICIES}
    reach_throw_pct = {p: round(100 * throws_k[p] / total_n, 2) for p in POLICIES}
    grasp_k = {p: throws_k[p] for p in POLICIES}  # 던지기 단계 도달 = 파지 성공 (논문 정의상 같은 단계)
    res["V8_reconstruct_21_24"] = {
        "fail_counts_per_policy": fail_n,
        "throw_error_counts_est": throw_err_k,
        "throws_attempted_est": throws_k,
        "reach_throw_pct_per_policy": reach_throw_pct,
        "reach_throw_pct_mean": round(sum(reach_throw_pct.values()) / 3, 2),
        "reported_text_pct": TEXT["throw_reach_share_text"],
        "success_per_throw_pct_mean": round(100 * sum(counts.values()) / sum(throws_k.values()), 2),
        "reported_success_per_throw_pct": TEXT["throws_success_text"],
        "dynamicvla_reach_throw_pct": round(100 * throws_k["DynamicVLA"] / N / 9, 2),
        "dynamicvla_reported_grasp_pct": TEXT["dynamicvla_grasp_text"],
        "note": "throws_attempted = 던지기 오류 + 성공. 파지 단계 도달과 같은 것으로 가정(§3.2.3의 실패 정의 기준).",
    }

    # V9. 논문 본문 비율 검산 (파생 값)
    total_eps = 35357
    hours = 143.2
    res["V9_dataset"] = {
        "mean_episode_seconds_at_10Hz": round(hours * 3600 / total_eps, 2),
        "episode_split_sum_pct": round(31.8 + 39.2 + 29.0, 1),
        "destination_split_sum_pct": round(31.76 + 29.77 + 38.47, 2),
        "asset_sum": 20 + 46 + 37 + 4,
        "dynamic_assets_share_of_episodes_pct": round(29.0 / 4, 2),
        "uniform_irregular_dynamic_episode_pct": [31.8, 39.2, 29.0],
    }

    # V10. 실패 모드 텍스트 검산: 89%, 64.2%
    res["V10_text_checks"] = {
        "pre_throw_text_pct": TEXT["pre_throw_text"],
        "pie_pre_throw_pct": round(FIG7_PIE[0] + FIG7_PIE[1] + FIG7_PIE[3], 1),
        "dynamicvla_approach_plus_reach": round(FIG7["DynamicVLA"][1] + FIG7["DynamicVLA"][3], 1),
    }

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"wrote {OUT}")
    print(json.dumps(res["V1_table1_pooled"], ensure_ascii=False, indent=1))
    print(json.dumps(res["V8_reconstruct_21_24"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
