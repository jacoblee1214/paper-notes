"""논문 표 값의 재계산과 일관성 검사 (FAWAM, arXiv:2606.08555 v3).

입력: 표 1–4의 성공 수(원문 텍스트에서 옮긴 값). 출력: analysis/recompute_results.json
실행: python3 analysis/recompute.py   (numpy, scipy)
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
TASKS = ["erase_board", "peel_cucumber", "pivot_box", "wipe_vase"]
N = 20  # 과제당 평가 시도 수

# 표 1 (v3). 과제 순서는 TASKS와 같다.
TABLE1 = {
    "pi05": [9, 8, 10, 11],
    "ge_act": [9, 10, 9, 11],
    "forcevla": [11, 14, 13, 13],
    "ta_vla": [10, 17, 14, 10],
    "feam": [14, 15, 16, 14],
    "ge_act_res": [11, 10, 10, 12],
    "ta_vla_res": [13, 19, 15, 14],
    "fawam": [15, 19, 17, 17],
}
TABLE1_AVG_REPORTED = {
    "pi05": 47.50, "ge_act": 48.75, "forcevla": 63.75, "ta_vla": 63.75,
    "feam": 73.75, "ge_act_res": 53.75, "ta_vla_res": 76.25, "fawam": 85.00,
}
# 표 2 (Obs, Pre, Res 조합)
TABLE2 = {
    "base_none": ([9, 10, 9, 11], 48.75),
    "obs": ([12, 12, 9, 12], 56.25),
    "obs_pre": ([14, 15, 16, 14], 73.75),
    "res_only": ([12, 15, 12, 15], 67.50),
    "obs_pre_res": ([15, 19, 17, 17], 85.00),
}
# 표 3 (잔차 보정 설계, 두 과제만)
TABLE3 = {"feam_res_no_wrench": [15, 15], "feam_res_no_gate": [16, 16], "fawam": [19, 17]}
# 표 4 (외란 실험, 시도 5회). 열 이름은 원문 그대로 "Slice cucumber"
TABLE4_TASKS = ["erase_board", "peel_cucumber", "slice_cucumber_sic", "wipe_vase"]
TABLE4 = {"no_correction": [0, 2, 0, 1], "with_correction": [3, 5, 4, 4]}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def fisher(a, b, c, d):
    """2x2 표 [[a, b], [c, d]]의 양측 Fisher 정확 검정 p값."""
    return float(stats.fisher_exact([[a, b], [c, d]])[1])


def required_n_per_group(p1, p2, alpha=0.05, power=0.8):
    """두 비율 차이를 잡기 위한 집단당 시도 수(정규 근사)."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return math.ceil((num / (p1 - p2)) ** 2)


results = {"checks": [], "table1": {}, "pairwise": {}, "ablation": {}, "perturbation": {}, "power": {}}

# 1) 표 1: 평균과 보고값 비교, 합계, Wilson 구간
for name, row in TABLE1.items():
    k = sum(row)
    avg = 100 * k / (N * len(row))
    ok = abs(avg - TABLE1_AVG_REPORTED[name]) < 0.005
    results["table1"][name] = {
        "successes": k, "trials": N * len(row), "avg_recomputed": round(avg, 2),
        "avg_reported": TABLE1_AVG_REPORTED[name], "match": ok,
        "wilson95_pct": wilson(k, N * len(row)),
    }
    results["checks"].append({"id": f"avg_{name}", "match": ok})

# 2) 표 2의 평균과 표 1 행과의 일치
for name, (row, rep) in TABLE2.items():
    avg = 100 * sum(row) / (N * 4)
    results["ablation"].setdefault("averages", {})[name] = {"recomputed": round(avg, 2), "reported": rep, "match": abs(avg - rep) < 0.005}
    results["checks"].append({"id": f"table2_{name}", "match": abs(avg - rep) < 0.005})
results["checks"].append({"id": "table2_feam_row_equals_table1", "match": TABLE2["obs_pre"][0] == TABLE1["feam"]})
results["checks"].append({"id": "table2_full_row_equals_table1", "match": TABLE2["obs_pre_res"][0] == TABLE1["fawam"]})
results["checks"].append({"id": "table3_fawam_equals_table1_on_peel_wipe", "match": [TABLE3["fawam"][0], TABLE3["fawam"][1]] == [TABLE1["fawam"][1], TABLE1["fawam"][3]]})

# 3) 본문 수치 대조 (percentage point)
text_claims = {
    "abstract_36.25_vs_best_vision_only": (85.00 - 48.75, 36.25),
    "abstract_21.25_vs_best_force_aware": (85.00 - 63.75, 21.25),
    "matched_31.25_vs_ge_act_res": (85.00 - 53.75, 31.25),
    "matched_8.75_vs_ta_vla_res": (85.00 - 76.25, 8.75),
    "residual_gain_feam_to_fawam": (85.00 - 73.75, 11.25),
    "obs_gain_7.5": (56.25 - 48.75, 7.5),
    "pre_gain_17.5": (73.75 - 56.25, 17.5),
    "res_only_gain_18.75": (67.50 - 48.75, 18.75),
}
for k, (calc, claimed) in text_claims.items():
    results["checks"].append({"id": k, "match": abs(calc - claimed) < 1e-6, "calc": round(calc, 2), "claimed": claimed})
results["relative_gain_vs_best_vision_only_pct"] = round(100 * (85.0 / 48.75 - 1), 1)
results["relative_gain_vs_best_force_aware_pct"] = round(100 * (85.0 / 63.75 - 1), 1)

# 4) 쌍별 Fisher 검정 (풀링한 80회 시도, 독립 가정)
pairs = [
    ("fawam", "ta_vla_res"), ("fawam", "feam"), ("fawam", "ge_act_res"), ("fawam", "forcevla"),
    ("fawam", "ta_vla"), ("feam", "ta_vla"), ("feam", "ge_act"), ("ta_vla_res", "ta_vla"),
    ("ge_act_res", "ge_act"), ("fawam", "ge_act"),
]
# 표 2 단계별 차이 (각 80회)
ABL_PAIRS = [("obs", "base_none"), ("obs_pre", "obs"), ("obs_pre_res", "obs_pre"), ("res_only", "base_none")]
for a, b in ABL_PAIRS:
    ka, kb = sum(TABLE2[a][0]), sum(TABLE2[b][0])
    results["ablation"][f"step_{a}_vs_{b}"] = {"succ_a": ka, "succ_b": kb, "n": 80, "diff_pp": round(100 * (ka - kb) / 80, 2),
                                             "fisher_p": round(fisher(ka, 80 - ka, kb, 80 - kb), 4)}
for a, b in pairs:
    ka, kb = sum(TABLE1[a]), sum(TABLE1[b])
    n = N * 4
    p = fisher(ka, n - ka, kb, n - kb)
    diff = 100 * (ka - kb) / n
    results["pairwise"][f"{a}_vs_{b}"] = {"succ_a": ka, "succ_b": kb, "n": n, "diff_pp": round(diff, 2), "fisher_p": round(p, 4)}

# 5) 잔차 보정 효과의 분해 (Res-only 대 GE-Act+Res: 입력에 wrench가 있는지만 다름이라고 가정)
ka, kb = sum(TABLE2["res_only"][0]), sum(TABLE1["ge_act_res"])
results["pairwise"]["res_only_with_wrench_vs_ge_act_res_without_wrench"] = {
    "succ_a": ka, "succ_b": kb, "n": 80, "diff_pp": round(100 * (ka - kb) / 80, 2), "fisher_p": round(fisher(ka, 80 - ka, kb, 80 - kb), 4),
    "note": "기반 모델과 교정 데이터가 같다고 가정한 비교. 논문은 이 쌍을 직접 비교하지 않음",
}

# 6) 과제별 FAWAM 대 TA-VLA+Res (각 20회)
results["per_task_fawam_vs_ta_vla_res"] = {
    t: {"fawam": TABLE1["fawam"][i], "ta_vla_res": TABLE1["ta_vla_res"][i],
        "fisher_p": round(fisher(TABLE1["fawam"][i], N - TABLE1["fawam"][i], TABLE1["ta_vla_res"][i], N - TABLE1["ta_vla_res"][i]), 3)}
    for i, t in enumerate(TASKS)
}

# 7) 표 4 외란 실험: 풀링(20회) Fisher, 과제별 Fisher(5회)
no, yes = TABLE4["no_correction"], TABLE4["with_correction"]
results["perturbation"]["pooled"] = {
    "no_correction": sum(no), "with_correction": sum(yes), "n_each": 20,
    "fisher_p": round(fisher(sum(no), 20 - sum(no), sum(yes), 20 - sum(yes)), 6),
}
results["perturbation"]["per_task_fisher_p"] = {
    t: round(fisher(no[i], 5 - no[i], yes[i], 5 - yes[i]), 4) for i, t in enumerate(TABLE4_TASKS)
}
results["perturbation"]["per_task_counts"] = {t: [no[i], yes[i]] for i, t in enumerate(TABLE4_TASKS)}

# 8) 표 3: 잔차 보정 설계 (두 과제 40회)
for name in ("feam_res_no_wrench", "feam_res_no_gate"):
    k = sum(TABLE3[name])
    results["ablation"][name] = {"succ": k, "n": 40, "vs_fawam_fisher_p": round(fisher(k, 40 - k, sum(TABLE3["fawam"]), 40 - sum(TABLE3["fawam"])), 4)}
results["ablation"]["fawam_peel_wipe"] = {"succ": sum(TABLE3["fawam"]), "n": 40}
results["ablation"]["gate_removed_vs_fawam_fisher_p"] = round(fisher(sum(TABLE3["feam_res_no_gate"]), 40 - sum(TABLE3["feam_res_no_gate"]), sum(TABLE3["fawam"]), 40 - sum(TABLE3["fawam"])), 4)

# 9) 검출력: 시도 수 20 x 4 = 80 집단당 필요한 시도 수 (두 비율, 양측 5%, 검정력 80%)
for label, p1, p2 in [("fawam_vs_ta_vla_res_85_vs_76", 0.85, 0.7625), ("fawam_vs_feam_85_vs_73.75", 0.85, 0.7375),
                      ("fawam_vs_ge_act_res_85_vs_53.75", 0.85, 0.5375), ("per_task_17_vs_14", 0.85, 0.70)]:
    results["power"][label] = required_n_per_group(p1, p2)

results["checks_all_match"] = all(c["match"] for c in results["checks"])
OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    print("checks all match:", results["checks_all_match"])
    for c in results["checks"]:
        if not c["match"]:
            print("MISMATCH", c)
    for k, v in results["pairwise"].items():
        print(f"{k:60s} {v['succ_a']:>3}/{v['n']} vs {v['succ_b']:>3}/{v['n']}  diff {v['diff_pp']:+.2f}pp  p={v['fisher_p']}")
    print("perturbation pooled", results["perturbation"]["pooled"])
    print("perturbation per task", results["perturbation"]["per_task_fisher_p"])
    print("power", results["power"])
    print("wilson", {k: v["wilson95_pct"] for k, v in results["table1"].items()})
    print("relative", results["relative_gain_vs_best_vision_only_pct"], results["relative_gain_vs_best_force_aware_pct"])
    print("per task", results["per_task_fawam_vs_ta_vla_res"])
    print("ablation", {k: v for k, v in results["ablation"].items() if k.startswith("step_") or k.startswith("gate") or k.startswith("feam_res")})
