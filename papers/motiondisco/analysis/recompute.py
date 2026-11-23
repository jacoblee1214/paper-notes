"""MotionDisco (arXiv:2606.06139) 표 1 재계산. 수 초, CPU, numpy/scipy만 사용.

실행: python3 analysis/recompute.py  → analysis/recompute_results.json

검사 항목 (번호는 03_analysis.md와 같다)
  V1  Valid % 가 어떤 (성공 수 k, 노드 수 n) 으로 만들어질 수 있는지(반올림/내림)와 공식 저장소의 과제별 노드 예산과의 일치
  V2  MD 대 MD w/o TF 비용의 상대 감소, 부호 검정
  V3  SC 성공 수와 Wilson 구간, "모든 시나리오 해결"의 구간
  V4  본문 문장과 표 값의 대조 (더 높은 Valid %, 더 낮은 비용, SC 대 w/o TF)
  V5  Valid % 합산 비교 (시행 수 가정): 과제별 Fisher, 층화 Mantel-Haenszel, 합산 Fisher
  V6  첫 유효해 시간으로부터 노드당 평가 시간의 대략 추정 (i.i.d. 가정)
"""
import json
import math
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"

# 논문 표 1 (7쪽). SC=None 은 표의 "/" (해를 못 찾음)
TASKS = ["Banana", "Box Stacking", "Climb Table w/ Box", "Long-Dist. Pick & Place",
         "Move Through Clutter", "Parkour Pick & Place 1", "Parkour Pick & Place 2",
         "Under-Table Pick & Place"]
SC = [None, 398.05, 16.16, 13.61, None, None, None, 115.64]
WO_COST = [21.17, 398.05, 16.47, 10.70, 11.96, 15.97, 15.90, 18.87]
WO_VALID = [48.14, 45.45, 46.15, 65.00, 73.33, 33.33, 58.82, 64.28]
MD_COST = [20.08, 10.53, 15.97, 10.42, 11.67, 15.47, 15.51, 17.84]
MD_VALID = [70.30, 63.63, 53.84, 70.00, 73.33, 38.88, 67.64, 71.42]
MD_TIME = [7.49, 4.58, 2.15, 2.50, 3.11, 6.05, 4.40, 1.34]

# 공식 저장소 examples/loco_manipulation/run_search.sh 의 SCENE_MAX_NODES (현재 저장소 값, 논문 값이라는 보장 없음)
REPO_NODES = [27, 11, 13, 20, 15, 40, 34, 14]


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def feasible(pct, n_max=200):
    """pct(소수 둘째 자리 %)를 만드는 (k, n). 반올림과 내림(truncation) 두 규칙."""
    out = {"round": [], "trunc": []}
    for n in range(1, n_max + 1):
        for k in range(0, n + 1):
            v = 100.0 * k / n
            if abs(round(v, 2) - pct) < 1e-9:
                out["round"].append((k, n))
            if abs(math.floor(v * 100 + 1e-9) / 100 - pct) < 1e-9:
                out["trunc"].append((k, n))
    return out


def v1():
    rows = []
    for name, wo, md, nrepo in zip(TASKS, WO_VALID, MD_VALID, REPO_NODES):
        row = {"task": name, "repo_nodes": nrepo}
        for col, pct in (("wo", wo), ("md", md)):
            f = feasible(pct)
            kr = [k for k, n in f["round"] if n == nrepo]
            kt = [k for k, n in f["trunc"] if n == nrepo]
            row[col] = {
                "pct": pct,
                "match_repo_n_round": kr, "match_repo_n_trunc": kt,
                "smallest_n_round": min((n for k, n in f["round"]), default=None),
                "smallest_n_trunc": min((n for k, n in f["trunc"]), default=None),
                "n_le_60_round": sorted({n for k, n in f["round"] if n <= 60}),
                "n_le_60_trunc": sorted({n for k, n in f["trunc"] if n <= 60}),
                "nearest_k_at_repo_n": nrepo and round(pct / 100 * nrepo),
                "nearest_pct_at_repo_n": 100 * round(pct / 100 * nrepo) / nrepo,
            }
        # 두 열이 같은 n 을 쓴다는 가정에서 공통 n (<=60)
        common_r = sorted(set(row["wo"]["n_le_60_round"]) & set(row["md"]["n_le_60_round"]))
        common_t = sorted(set(row["wo"]["n_le_60_trunc"]) & set(row["md"]["n_le_60_trunc"]))
        row["common_n_le_60_round"] = common_r
        row["common_n_le_60_trunc"] = common_t
        rows.append(row)
    return rows


def v2():
    rel = [(w - m) / w * 100 for w, m in zip(WO_COST, MD_COST)]
    better = sum(m < w for w, m in zip(WO_COST, MD_COST))
    p_two = min(1.0, 2 * stats.binom.sf(better - 1, 8, 0.5))
    rel_wo_box = [r for t, r in zip(TASKS, rel) if t != "Box Stacking"]
    ratio = [m / w for w, m in zip(WO_COST, MD_COST)]
    return {
        "rel_reduction_pct": dict(zip(TASKS, [round(r, 2) for r in rel])),
        "n_md_lower": better, "sign_test_two_sided_p": p_two,
        "median_rel_reduction_pct": float(np.median(rel)),
        "mean_rel_reduction_excl_box_pct": float(np.mean(rel_wo_box)),
        "range_excl_box_pct": [min(rel_wo_box), max(rel_wo_box)],
        "geo_mean_ratio_md_over_wo": float(np.exp(np.mean(np.log(ratio)))),
        "geo_mean_ratio_excl_box": float(np.exp(np.mean([math.log(r) for t, r in zip(TASKS, ratio) if t != "Box Stacking"]))),
        "box_stacking_ratio": 398.05 / 10.53,
        "wo_cost_equals_sc_box": WO_COST[1] == SC[1],
    }


def v3():
    solved = sum(x is not None for x in SC)
    sc_vs_wo = {}
    for t, s, w, m in zip(TASKS, SC, WO_COST, MD_COST):
        if s is not None:
            sc_vs_wo[t] = {"sc": s, "wo": w, "md": m, "sc_lt_wo": s < w, "sc_lt_md": s < m}
    return {
        "sc_solved": solved, "n_tasks": 8, "sc_wilson95": wilson(solved, 8),
        "sc_p_two_sided_vs_0.5_binom": float(stats.binomtest(solved, 8, 0.5).pvalue),
        "all_8_solved_wilson95": wilson(8, 8),
        "fisher_sc_4of8_vs_all_8of8_p": float(stats.fisher_exact([[8, 0], [4, 4]])[1]),
        "sc_solved_tasks": sc_vs_wo,
    }


def v4():
    higher = sum(m > w for w, m in zip(WO_VALID, MD_VALID))
    tie = sum(m == w for w, m in zip(WO_VALID, MD_VALID))
    lower_cost = sum(m < w for w, m in zip(WO_COST, MD_COST))
    p_valid_two = min(1.0, 2 * stats.binom.sf(higher - 1, higher, 0.5))
    wo_not_better_than_sc = [t for t, s, w in zip(TASKS, SC, WO_COST) if s is not None and s < w]
    return {"md_valid_higher": higher, "md_valid_tie": tie, "md_cost_lower": lower_cost,
            "sign_test_valid_two_sided_p_ties_dropped": p_valid_two,
            "tasks_sc_cheaper_than_wo_tf": wo_not_better_than_sc}


def mh(strata):
    """층화 Mantel-Haenszel 공통 오즈비와 Robins-Breslow-Greenland 95% 구간, 그리고 CMH 검정(연속성 보정 없음)."""
    R = S = 0.0
    PR = PS_QR = QS = 0.0
    sum_a = sum_E = sum_V = 0.0
    for a, n1, c, n0 in strata:  # a: MD 성공, n1: MD 노드, c: w/o TF 성공, n0: w/o TF 노드
        b, d = n1 - a, n0 - c
        n = n1 + n0
        r = a * d / n
        s = b * c / n
        P = (a + d) / n
        Q = (b + c) / n
        R += r
        S += s
        PR += P * r
        PS_QR += P * s + Q * r
        QS += Q * s
        sum_a += a
        E = n1 * (a + c) / n
        V = n1 * n0 * (a + c) * (b + d) / (n * n * (n - 1)) if n > 1 else 0
        sum_E += E
        sum_V += V
    or_mh = R / S
    var = PR / (2 * R * R) + PS_QR / (2 * R * S) + QS / (2 * S * S)
    se = math.sqrt(var)
    lo, hi = math.exp(math.log(or_mh) - 1.959964 * se), math.exp(math.log(or_mh) + 1.959964 * se)
    chi2 = (sum_a - sum_E) ** 2 / sum_V
    return or_mh, [lo, hi], chi2, float(stats.chi2.sf(chi2, 1))


def counts(assumption):
    """assumption: 'repo' = 저장소 예산 n, 단 Parkour 1 은 n=18 (정수 가능한 최소), Banana MD 는 가장 가까운 k."""
    ns = list(REPO_NODES)
    if assumption == "repo":
        ns[5] = 18
    elif assumption == "min_n":
        ns = [27, 11, 13, 20, 15, 18, 34, 14]
    wo = [round(p / 100 * n) for p, n in zip(WO_VALID, ns)]
    md = [round(p / 100 * n) for p, n in zip(MD_VALID, ns)]
    return ns, wo, md


def v5():
    ns, wo, md = counts("repo")
    per = []
    for t, n, w, m in zip(TASKS, ns, wo, md):
        odds, p = stats.fisher_exact([[m, n - m], [w, n - w]])
        per.append({"task": t, "n": n, "wo_k": w, "md_k": m, "wo_pct": 100 * w / n, "md_pct": 100 * m / n,
                    "wo_wilson": wilson(w, n), "md_wilson": wilson(m, n), "fisher_p": float(p)})
    strata = [(m, n, w, n) for n, w, m in zip(ns, wo, md)]
    or_mh, ci, chi2, p_cmh = mh(strata)
    W, M, N = sum(wo), sum(md), sum(ns)
    odds_pool, p_pool = stats.fisher_exact([[M, N - M], [W, N - W]])
    # 노드 독립이 아닌 점을 반영한 감도 분석: 유효 표본을 1/deff 로 줄인다
    sens = {}
    for deff in (1, 2, 4):
        strata_d = [(m / deff, n / deff, w / deff, n / deff) for n, w, m in zip(ns, wo, md)]
        o, c, x, p = mh(strata_d)
        sens[str(deff)] = {"or": o, "ci95": c, "p": p}
    return {"assumed_n": ns, "wo_k": wo, "md_k": md, "total_n": N, "wo_total": W, "md_total": M,
            "wo_pct": 100 * W / N, "md_pct": 100 * M / N, "per_task": per,
            "mh_or": or_mh, "mh_ci95": ci, "cmh_chi2": chi2, "cmh_p": p_cmh,
            "pooled_fisher_or": odds_pool, "pooled_fisher_p": float(p_pool),
            "design_effect_sensitivity": sens,
            "banana_md_k19_pct": 100 * 19 / 27,
            "note": "n 은 저장소 예산을 가정. Parkour 1 은 40 으로는 정수가 안 되어 18, Banana MD 70.30 은 19/27=70.37 로 대체"}


def v6():
    """첫 유효해까지 기대 노드 수 1/p (i.i.d. 가정)에서 노드당 평가 시간 = Time * p. 값은 대략적."""
    rows = []
    ns, _, md = counts("repo")
    for t, p, tm, n in zip(TASKS, MD_VALID, MD_TIME, ns):
        pp = p / 100
        t_node = tm * pp
        rows.append({"task": t, "p": pp, "time_first_min": tm, "t_node_min": t_node,
                     "implied_full_budget_min": t_node * n})
    return rows


def main():
    res = {"V1_integer_feasibility": v1(), "V2_cost_reduction": v2(), "V3_sc": v3(),
           "V4_text_claims": v4(), "V5_valid_pct_pooled": v5(), "V6_time": v6()}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    # 요약 출력
    for r in res["V1_integer_feasibility"]:
        print(r["task"], "repo n", r["repo_nodes"],
              "| wo", r["wo"]["pct"], "k@n(trunc)", r["wo"]["match_repo_n_trunc"], "k@n(round)", r["wo"]["match_repo_n_round"],
              "| md", r["md"]["pct"], "k@n(trunc)", r["md"]["match_repo_n_trunc"], "k@n(round)", r["md"]["match_repo_n_round"],
              "| common n<=60 trunc", r["common_n_le_60_trunc"][:6])
    print(json.dumps(res["V2_cost_reduction"], indent=1, ensure_ascii=False))
    print(json.dumps(res["V3_sc"], indent=1, ensure_ascii=False))
    print(json.dumps(res["V4_text_claims"], indent=1, ensure_ascii=False))
    v = res["V5_valid_pct_pooled"]
    print({k: v[k] for k in ("total_n", "wo_total", "md_total", "wo_pct", "md_pct", "mh_or", "mh_ci95", "cmh_p",
                              "pooled_fisher_or", "pooled_fisher_p", "design_effect_sensitivity")})
    for p in v["per_task"]:
        print(p["task"], p["n"], p["wo_k"], p["md_k"], round(p["fisher_p"], 3))
    for r in res["V6_time"]:
        print(r)


if __name__ == "__main__":
    main()
