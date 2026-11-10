"""블로그 수치 재계산과 통계 검정 (CPU, numpy 표준 라이브러리만 사용).

입력: repo/site/assets/analysis/*.json, repo/site/experience-story.json (git clone, 원자료 공개분)
출력: analysis/recompute_results.json
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "repo" / "site"
DATA = SITE / "assets" / "analysis"
OUT = Path(__file__).resolve().parent / "recompute_results.json"

METHODS = ["hoi_spider", "zero_shot", "codex", "orchestrator", "communicating", "codex120"]
LABEL = {
    "hoi_spider": "HOI-Retarget + SPIDER",
    "zero_shot": "Astra one-shot",
    "codex": "Codex 30분",
    "orchestrator": "독립 스웜 30분",
    "communicating": "협업 스웜 30분",
    "codex120": "Codex 120분",
}
# 블로그 본문·표에 적힌 값 (paper/blog.html에서 추출)
PAGE = {
    "table1_humoto_add": {"hoi_spider": 44.96, "zero_shot": 56.09, "codex120": 9.85},
    "table2_omomo_add": {"hoi_spider": 40.77, "zero_shot": 69.87, "codex120": 15.55},
    "pooled_pass_pct": {"hoi_spider": 17.5, "zero_shot": 2.5, "codex120": 75.0,
                        "codex": 47.5, "orchestrator": 52.5, "communicating": 67.5},
    "pooled_add_cm": {"hoi_spider": 42.87, "zero_shot": 62.98, "codex120": 12.63},
    "best_at_30_120": {"codex120_30": 29.59, "codex120_120": 10.21},
    "shelf_exp": {"without": 23.35, "with": 7.77},
}


def binom_two_sided(k, n):
    """p=0.5 이항검정 양측 p값 (McNemar 정확검정에 사용)."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mcnemar(a, b):
    """같은 과제에서 두 방법의 PASS 여부를 짝지어 비교."""
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n10 + n01
    return {"a_only": n10, "b_only": n01, "p_exact": binom_two_sided(min(n10, n01), n)}


def boot_diff(x, y, n_boot=10000, seed=0):
    """짝지은 차이 x-y 평균의 95% 부트스트랩 구간(과제 단위 재표본)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)], "n": int(len(d))}


def main():
    res = json.load(open(DATA / "retargeting-results.json"))
    recs = res["records"]
    by = defaultdict(dict)  # by[method][(dataset, clip)] = record
    for r in recs:
        by[r["method"]][(r["dataset"], r["clip"])] = r
    keys = sorted(by["hoi_spider"].keys())
    out = {"source": "repo/site/assets/analysis/retargeting-results.json (blog repo commit 84cbe2253)"}

    # 1. 집계 재계산: 과제 동등 가중, 추적 실패(미완료) 과제는 추적 평균에서 제외
    agg = {}
    for m in METHODS:
        row = {}
        for ds in ["humoto", "omomo", "all"]:
            ks = [k for k in keys if ds == "all" or k[0] == ds]
            tracked = [by[m][k]["metrics"]["mean_add_m"] for k in ks if by[m][k]["metrics"]
                       and by[m][k]["metrics"].get("mean_add_m") is not None]
            passes = sum(1 for k in ks if by[m][k]["semantic"] == "PASS")  # 보고 기준: 원래 판정
            row[ds] = {
                "n_tasks": len(ks), "n_tracked": len(tracked),
                "mean_add_cm": round(100 * float(np.mean(tracked)), 2),
                "pass": passes, "pass_pct": round(100 * passes / len(ks), 1),
            }
        agg[m] = row
    out["aggregate_recomputed"] = agg
    out["aggregate_vs_page"] = {
        "table1_humoto_add": {m: [agg[m]["humoto"]["mean_add_cm"], v] for m, v in PAGE["table1_humoto_add"].items()},
        "table2_omomo_add": {m: [agg[m]["omomo"]["mean_add_cm"], v] for m, v in PAGE["table2_omomo_add"].items()},
        "pooled_pass_pct": {m: [agg[m]["all"]["pass_pct"], v] for m, v in PAGE["pooled_pass_pct"].items()},
        "pooled_add_cm": {m: [agg[m]["all"]["mean_add_cm"], v] for m, v in PAGE["pooled_add_cm"].items()},
    }

    # 2. 짝지은 비교 (같은 40과제). 추적 차이는 두 방법 모두 추적 값이 있는 과제만 사용
    pairs = [("codex120", "hoi_spider"), ("codex120", "communicating"), ("communicating", "orchestrator"),
             ("orchestrator", "codex"), ("codex", "hoi_spider"), ("codex120", "codex"), ("communicating", "codex")]
    paired = {}
    for a, b in pairs:
        pa = [by[a][k]["semantic"] == "PASS" for k in keys]
        pb = [by[b][k]["semantic"] == "PASS" for k in keys]
        both = [k for k in keys if by[a][k]["metrics"] and by[b][k]["metrics"]
                and by[a][k]["metrics"].get("mean_add_m") is not None and by[b][k]["metrics"].get("mean_add_m") is not None]
        ta = [by[a][k]["metrics"]["mean_add_m"] * 100 for k in both]
        tb = [by[b][k]["metrics"]["mean_add_m"] * 100 for k in both]
        paired[f"{a} vs {b}"] = {
            "pass_mcnemar": mcnemar(pa, pb),
            "add_cm_boot": boot_diff(ta, tb),
        }
    out["paired_tests_raw_labels"] = paired

    # 3. 재검토 라벨(reviewed_semantic) 민감도: 독립 스웜 21 -> 19
    rev = {m: sum(1 for k in keys if by[m][k]["reviewed_semantic"] == "PASS") for m in METHODS}  # 재검토 후
    out["reviewed_label_pass_counts"] = rev
    pa = [by["orchestrator"][k]["reviewed_semantic"] == "PASS" for k in keys]
    pb = [by["codex"][k]["reviewed_semantic"] == "PASS" for k in keys]
    out["orchestrator_vs_codex30_reviewed"] = mcnemar(pa, pb)
    out["opening_disqualified"] = [f"{m}:{k[1]}" for m in METHODS for k in keys if by[m][k]["opening_disqualified"]]

    # 4. 누락 과제 민감도: codex120 OMOMO sub13_whitechair_069 (미완료, 추적 평균에서 제외됨)
    miss = ("omomo", "sub13_whitechair_069")
    init = json.load(open(DATA / "initial-reference-tracking.json"))["records"]
    init_val = [r for r in init if (r.get("dataset"), r.get("clip")) == miss and r.get("minute") == 0.0]
    others = {m: by[m][miss]["metrics"]["mean_add_m"] * 100 for m in METHODS
              if m != "codex120" and by[m][miss]["metrics"]}
    sens = {"others_on_task_add_cm": {m: round(v, 2) for m, v in others.items()}}
    base39 = [by["codex120"][k]["metrics"]["mean_add_m"] * 100 for k in keys if k != miss]
    sens["codex120_mean39"] = round(float(np.mean(base39)), 2)
    if init_val:
        sens["initial_ref_add_cm_on_task"] = round(init_val[0]["mean_add_cm"], 2)
        sens["codex120_mean40_if_initial_ref"] = round(float(np.mean(base39 + [init_val[0]["mean_add_cm"]])), 2)
    sens["codex120_mean40_if_worst_other"] = round(float(np.mean(base39 + [max(others.values())])), 2)
    out["missing_task_sensitivity"] = sens

    # 5. 최선값 곡선(best-so-far)과 최종 제출 값의 차이 (budget-tracking vs retargeting-results)
    bud = json.load(open(DATA / "budget-tracking.json"))["curves"]

    def val_at(c, t):
        m_ = np.asarray(c["minutes"]); v_ = np.asarray(c["mean_add_cm"])
        i = np.searchsorted(m_, t, side="right") - 1
        return float(v_[i])

    gap = {}
    final_map = {"codex": "codex", "orchestrator": "orchestrator", "communicating": "communicating", "codex120": "codex120"}
    for cur, m in final_map.items():
        c = bud[cur]
        best30 = round(val_at(c, 30.0), 2)
        best120 = round(val_at(c, 120.0), 2) if max(c["minutes"]) >= 120 else None
        gap[m] = {"best_so_far_at_30min_cm": best30, "best_so_far_at_120min_cm": best120,
                  "final_submitted_mean40_cm": agg[m]["all"]["mean_add_cm"]}
    out["best_so_far_vs_final"] = gap

    # 6. 실행 간 변동: 30분 단일 Codex 실행 vs 120분 실행의 앞 30분 최선값 (서로 다른 실행, 접두부 아님)
    c30 = bud["codex"]; c120 = bud["codex120"]
    out["run_to_run_note"] = ("codex(30분)와 codex120은 서로 다른 실행이다. 120분 실행의 30분 시점 값(29.59)이 "
                              "30분 단독 실행의 최종 값(23.74)보다 크다. 실행 간 분산이 블로그가 보고한 시간 효과와 같은 크기일 수 있다.")
    out["run_to_run_note_values"] = {"codex30_best_at_30": round(val_at(c30, 30.0), 2),
                                     "codex120_best_at_30": round(val_at(c120, 30.0), 2)}

    # 7. 추적 기록: 실행당 시도 수 (within-run-tracking.json)
    w = json.load(open(DATA / "within-run-tracking.json"))
    L = [len(x["trials"]) for x in w["records"]]
    out["within_run_trials"] = {"runs": len(L), "mean": round(float(np.mean(L)), 1),
                                "min": int(min(L)), "max": int(max(L)),
                                "summary": w["summary"]}

    # 8. 토큰 사용량 (agent-tokens.json; 가격 정보 없음, 달러 환산 안 함)
    tok = json.load(open(DATA / "agent-tokens.json"))
    tot = {t["method"]: t for t in tok["totals"] if t["dataset"] == "all"}
    out["tokens_per_task_all40"] = {
        m: {"recorded_attempts": v["recorded_attempts"], "partial_records": v["partial_records"],
            "total_tokens_M": round(v["total_tokens"] / 1e6, 1),
            "per_task_M": round(v["total_tokens"] / 1e6 / 40, 2),
            "cached_share_pct": round(100 * v["cached_input_tokens"] / max(1, v["input_tokens"]), 1)}
        for m, v in tot.items()
    }

    # 9. 경험 실험 (experience-story.json): 과제별 without/with 최선값, 그룹 평균, leave-one-out, 부호 검정
    exp = json.load(open(SITE / "experience-story.json"))["tasks"]
    rows = []
    for x in exp:
        wo = x["metrics"]["without_experience"]["mean_add_m"] * 100
        wi = x["metrics"]["with_experience"]["mean_add_m"] * 100
        wo_o = x["metrics"]["without_experience"]["orientation_mean_deg"]
        wi_o = x["metrics"]["with_experience"]["orientation_mean_deg"]
        rows.append({"clip": x["clip"], "group": x["group"], "without_add": wo, "with_add": wi,
                     "without_orient": wo_o, "with_orient": wi_o})
    grp = {}
    for g in ["related", "random"]:
        rr = [r for r in rows if r["group"] == g]
        grp[g] = {"add_without": round(float(np.mean([r["without_add"] for r in rr])), 2),
                  "add_with": round(float(np.mean([r["with_add"] for r in rr])), 2),
                  "orient_without": round(float(np.mean([r["without_orient"] for r in rr])), 2),
                  "orient_with": round(float(np.mean([r["with_orient"] for r in rr])), 2),
                  "tasks_better_with": sum(1 for r in rr if r["with_add"] < r["without_add"]),
                  "tasks_worse_with": sum(1 for r in rr if r["with_add"] > r["without_add"])}
    rel = [r for r in rows if r["group"] == "related"]
    loo = {}
    for r in rel:
        rest = [q for q in rel if q is not r]
        loo[r["clip"]] = round(float(np.mean([q["with_add"] - q["without_add"] for q in rest])), 2)
    diffs = np.array([r["with_add"] - r["without_add"] for r in rows])
    n_better = int((diffs < 0).sum()); n_worse = int((diffs > 0).sum())
    rng = np.random.default_rng(0)
    boot = [float(np.mean(rng.choice(diffs, size=len(diffs)))) for _ in range(10000)]
    out["experience"] = {
        "shelf_without_cm": round(rows[[r["clip"] for r in rows].index("lift_shelf_lie_to_stand-867")]["without_add"], 2),
        "shelf_with_cm": round(rows[[r["clip"] for r in rows].index("lift_shelf_lie_to_stand-867")]["with_add"], 2),
        "group_means": grp,
        "related_leave_one_out_mean_diff_cm_when_dropped": loo,
        "all10_tasks_better_with": n_better, "all10_tasks_worse_with": n_worse,
        "all10_sign_test_p": binom_two_sided(min(n_better, n_worse), n_better + n_worse),
        "all10_mean_diff_cm": round(float(diffs.mean()), 2),
        "all10_boot_ci95_cm": [round(float(np.percentile(boot, 2.5)), 2), round(float(np.percentile(boot, 97.5)), 2)],
        "per_task": rows,
    }

    # 10. 매칭 시간 (matched-time-tracking.json): 데이터셋별 곡선을 20+20 과제 동등 가중으로 평균, 30분 시점 값
    mt = json.load(open(DATA / "matched-time-tracking.json"))["datasets"]
    matched = {}
    for mk in ["codex", "orchestrator", "communicating"]:
        vals = {}
        for ds in ["humoto", "omomo"]:
            mins = np.asarray(mt[ds]["minutes"]); v = np.asarray(mt[ds]["mean_add_cm"][mk])
            vals[ds] = float(v[np.searchsorted(mins, 30.0, side="right") - 1])
        matched[mk] = {"humoto_30min": round(vals["humoto"], 2), "omomo_30min": round(vals["omomo"], 2),
                       "mean_of_two_datasets_30min": round((vals["humoto"] + vals["omomo"]) / 2, 2)}
    out["matched_time_30min_cm"] = matched

    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
