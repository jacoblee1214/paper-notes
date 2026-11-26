"""Diagnosing Compositional Generalization (arXiv:2607.29687 v2) 수치 재계산과 일관성 검사.

입력
  - 논문 본문, 부록, 표 1의 수치(아래 상수, 출처를 주석으로 적음)
  - 저자의 프로젝트 페이지 저장소가 공개한 집계 파일 data/page/*  (analysis/fetch_page_data.py 로 받는다)
출력: analysis/recompute_results.json, 표준출력 요약
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 문장에 표시한다)
  - 표시용 데이터(results_data.json)의 과제별 시드별 성공률이 10의 배수이므로 PPP는 과제·시드당 10회로 읽는다.
  - 파인튜닝 평가의 시도 수는 같은 10회/과제/시드로 가정한다(논문과 내보낸 파일에 명시 없음).
  - 그림에서 읽은 값은 (그림 판독 값)으로 표시한다. 이 스크립트는 그림 값을 쓰지 않고 표 1과 본문 수치, 내보낸 CSV만 쓴다.
"""
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "data" / "page"
OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}


# ---------------------------------------------------------------- 통계 도우미
def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe(k1, n1, k2, n2, z=1.959964):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def welch(a, b):
    r = stats.ttest_ind(a, b, equal_var=False)
    return float(r.statistic), float(r.pvalue)


def welch_stats(m1, s1, n1, m2, s2, n2):
    r = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
    return float(r.statistic), float(r.pvalue)


def pct(x, nd=2):
    return round(100 * x, nd)


def attainable(rate_pct, n, nd=2):
    """rate_pct(소수 nd자리 반올림)를 k/n 으로 낼 수 있는 정수 k 목록."""
    return [k for k in range(n + 1) if round(100 * k / n, nd) == round(rate_pct, nd)]


def read_csv(name):
    with open(PAGE / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- V1: 쌍 식별 (총합 = 가중 평균)
def v1_identity():
    out = {}
    for tag, fname, ntask in (("PPP", "q1_data_success_vs_b_runs.csv", 64), ("PP", "q1_data_l4_1s_success_vs_b_runs.csv", 16)):
        rows = read_csv(fname)
        by = defaultdict(dict)
        for r in rows:
            cond = r.get("condition", "")
            by[(r["b"], cond, r["task_set"], r["seed"])][r["metric"]] = float(r["rate"])
        worst = 0.0
        n = 0
        for (b, cond, ts, seed), m in by.items():
            if {"id", "ood", "total"} <= set(m) and int(b) < ntask:
                b = int(b)
                est = (b * m["id"] + (ntask - b) * m["ood"]) / ntask
                worst = max(worst, abs(est - m["total"]))
                n += 1
        out[tag] = {"rows_checked": n, "max_abs_diff_pp": round(worst, 4)}
    return out


# ---------------------------------------------------------------- V2: 시도 수 정수 검사
def v2_trials():
    out = {}
    # PP: 시드별 비율. B개 ID 과제, 16-B개 OOD 과제. 후보 시도 수/과제.
    rows = read_csv("q1_data_l4_1s_success_vs_b_runs.csv")
    for trials in (10, 20, 25, 30):
        ok = bad = 0
        for r in rows:
            b = int(r["b"])
            if r["metric"] == "total" or b >= 16:
                continue
            nt = b if r["metric"] == "id" else 16 - b
            n = nt * trials
            if attainable(float(r["rate"]), n, 4):
                ok += 1
            else:
                bad += 1
        out[f"PP_trials{trials}"] = {"attainable": ok, "not_attainable": bad}
    rows = read_csv("q1_data_success_vs_b_runs.csv")
    for trials in (10, 30):
        ok = bad = 0
        for r in rows:
            b = int(r["b"])
            if r["metric"] == "total" or b >= 64:
                continue
            nt = b if r["metric"] == "id" else 64 - b
            n = nt * trials
            if attainable(float(r["rate"]), n, 4):
                ok += 1
            else:
                bad += 1
        out[f"PPP_trials{trials}"] = {"attainable": ok, "not_attainable": bad}
    # 표시용 데이터: 과제별 시드별 성공률의 최소 간격
    d = json.load(open(PAGE / "results_data.json", encoding="utf-8"))
    gran = {}
    epp = {}
    for k, v in d["experiments"].items():
        vals = sorted({round(s, 4) for c in v["cells"] for s in c.get("per_seed", [])})
        gran[k] = min(b - a for a, b in zip(vals, vals[1:]))
        ep = (v["id_episodes"] or 0) + (v["ood_episodes"] or 0)
        epp[k] = ep / v["n_eval_config"]
    out["viewer_granularity_pct"] = {
        "min_gap_counts": dict(sorted(((g, list(gran.values()).count(g)) for g in set(gran.values())))),
        "episodes_per_task_counts": dict(sorted(((e, list(epp.values()).count(e)) for e in set(epp.values())))),
        "special": {k: {"gap": gran[k], "ep_per_task": epp[k], "seeds": d["experiments"][k]["n_seeds_used"]}
                    for k in ("orthogonal", "full", "covopt1_b8", "opt_inv_b6")},
    }
    return out


# ---------------------------------------------------------------- V3, V4: 직교 대 전체, 포화
def v3_orth_full():
    d = json.load(open(PAGE / "results_data.json", encoding="utf-8"))["experiments"]
    o, f = d["orthogonal"], d["full"]
    ko = o["pair_overlap"]["3_pair"]["successes"]
    no = o["pair_overlap"]["3_pair"]["episodes"]
    kf = f["pair_overlap"]["train"]["successes"]
    nf = f["pair_overlap"]["train"]["episodes"]
    diff, lo, hi = newcombe(ko, no, kf, nf)
    res = {
        "viewer_orthogonal_ood": {"k": ko, "n": no, "pct": pct(ko / no), "wilson": [pct(x) for x in wilson(ko, no)]},
        "viewer_full_all": {"k": kf, "n": nf, "pct": pct(kf / nf), "wilson": [pct(x) for x in wilson(kf, nf)]},
        "diff_pp": pct(diff), "newcombe95_pp": [pct(lo), pct(hi)], "fisher_p": round(fisher(ko, no, kf, nf), 3),
    }
    # 차트 CSV: 시드 평균 78.1944 (3시드) 대 78.6458
    ortho = {r["metric"]: r for r in read_csv("q1_data_success_vs_b_ortho.csv")}
    full_chart = [r for r in read_csv("q1_data_success_vs_b.csv") if r["b"] == "64" and r["metric"] == "total"][0]
    res["chart_csv"] = {"orth_ood_mean": float(ortho["ood"]["mean"]), "orth_ood_sd": float(ortho["ood"]["std"]),
                        "full_mean": float(full_chart["mean"]), "full_sd": float(full_chart["std"])}
    res["chart_vs_viewer_full_gap_pp"] = round(float(full_chart["mean"]) - 100 * kf / nf, 2)
    t, p = welch_stats(float(ortho["ood"]["mean"]), float(ortho["ood"]["std"]), 3, float(full_chart["mean"]), float(full_chart["std"]), 3)
    res["welch_seed_level_orth_vs_full_chart"] = {"t": round(t, 2), "p": round(p, 3)}
    # 직교 16 대 무작위 16 (시드 단위)
    runs = read_csv("q1_data_success_vs_b_runs.csv")
    rnd = [float(r["rate"]) for r in runs if r["b"] == "16" and r["metric"] == "ood"]
    t, p = stats.ttest_ind_from_stats(float(ortho["ood"]["mean"]), float(ortho["ood"]["std"]), 3, np.mean(rnd), np.std(rnd, ddof=1), len(rnd), equal_var=False)[:2]
    res["orth16_vs_random16_seed_level"] = {"random_mean": round(float(np.mean(rnd)), 2), "random_sd": round(float(np.std(rnd, ddof=1)), 2),
                                            "n_random_runs": len(rnd), "welch_p": round(float(p), 3)}
    sets = defaultdict(list)
    for r in runs:
        if r["b"] == "16" and r["metric"] == "ood":
            sets[r["task_set"]].append(float(r["rate"]))
    set_means = [float(np.mean(v)) for k, v in sorted(sets.items())]
    t, p = stats.ttest_1samp(set_means, float(ortho["ood"]["mean"]))
    res["random16_task_set_means"] = [round(x, 2) for x in set_means]
    res["orth16_vs_random16_set_level_ttest1samp_p"] = round(float(p), 3)
    # 포화(차트 CSV, 3 과제집합 평균): 대 전체
    b = {(r["b"], r["metric"]): r for r in read_csv("q1_data_success_vs_b.csv")}
    sat = {}
    for B in ("12", "16", "20", "24"):
        m, sd = float(b[(B, "ood")]["mean"]), float(b[(B, "ood")]["std"])
        t, p = welch_stats(m, sd, 3, float(full_chart["mean"]), float(full_chart["std"]), 3)
        sat[B] = {"ood_mean": round(m, 2), "gap_to_full_pp": round(m - float(full_chart["mean"]), 2), "welch_p_set_level": round(p, 3)}
    res["PPP_saturation_vs_full"] = sat
    pp = {(r["b"], r["condition"], r["metric"]): r for r in read_csv("q1_data_l4_1s_success_vs_b.csv")}
    full_pp = pp[("16", "full", "total")]
    satpp = {}
    for B in ("7", "8", "10"):
        r = pp[(B, "random", "ood")]
        m, sd = float(r["mean"]), float(r["std"])
        t, p = welch_stats(m, sd, 3, float(full_pp["mean"]), float(full_pp["std"]), 3)
        satpp[B] = {"ood_mean": round(m, 2), "gap_to_full_pp": round(m - float(full_pp["mean"]), 2), "welch_p_set_level": round(p, 3)}
    r = pp[("7", "S", "ood")]
    satpp["7_S_set"] = {"ood_mean": float(r["mean"]), "random7_ood_mean": float(pp[("7", "random", "ood")]["mean"])}
    res["PP_saturation_vs_full"] = {"full_mean": float(full_pp["mean"]), **satpp}
    return res


# ---------------------------------------------------------------- V5: 손실 수치와 성공률의 대응
def v5_loss():
    id_loss = [0.00648, 0.00663, 0.00667]
    ood_loss = [0.03421, 0.01548, 0.00756]
    res = {"ratio_ood_over_id": [round(o / i, 2) for o, i in zip(ood_loss, id_loss)],
           "id_loss_range_rel": round((max(id_loss) - min(id_loss)) / np.mean(id_loss), 3)}
    runs = read_csv("q1_data_success_vs_b_runs.csv")
    cand = {}
    for target, B in ((0.83, "4"), (23.93, "8"), (82.29, "16")):
        hits = [(r["task_set"], r["seed"], r["rate"]) for r in runs
                if r["b"] == B and r["metric"] == "ood" and abs(float(r["rate"]) - target) < 0.006]
        cand[f"B{B}_{target}"] = hits
    res["matches_in_exported_per_seed_ood"] = cand
    res["B8_exported_ood_per_seed_range"] = [min(float(r["rate"]) for r in runs if r["b"] == "8" and r["metric"] == "ood"),
                                             max(float(r["rate"]) for r in runs if r["b"] == "8" and r["metric"] == "ood")]
    # 23.93% 가 정수 성공 수로 가능한 분모
    res["B8_23.93_integer_ok"] = {"56_tasks_x_30": attainable(23.93, 56 * 30), "56_tasks_x_10": attainable(23.93, 56 * 10)}
    return res


# ---------------------------------------------------------------- V6: 본 쌍 수(그림 3a)와 설계 수준 분석
def parse_task(slug):
    lhs, color = slug.rsplit("_press_", 1)
    obj, cont = lhs.split("_into_", 1)
    return obj, cont, color


def seen_pairs(train, task):
    o, c, b = parse_task(task)
    tr = [parse_task(t) for t in train]
    n = 0
    n += any(x[0] == o and x[1] == c for x in tr)
    n += any(x[0] == o and x[2] == b for x in tr)
    n += any(x[1] == c and x[2] == b for x in tr)
    return n


def v6_pairs():
    d = json.load(open(PAGE / "results_data.json", encoding="utf-8"))["experiments"]
    res = {}
    mism = 0
    tot = 0
    designs = []
    for k, v in d.items():
        if k == "full":
            continue
        train = v["train_tasks"]
        ood_cells = [c for c in v["cells"] if not c["is_id"]]
        for c in ood_cells:
            tot += 1
            mism += seen_pairs(train, c["task"]) != c["seen_pairs"]
        sp = [c["seen_pairs"] for c in ood_cells]
        designs.append({"name": k, "B": v["n_train"], "ood_rate": v["ood_rate"], "id_rate": v["id_rate"],
                        "mean_seen": float(np.mean(sp)), "frac3": float(np.mean([s == 3 for s in sp])),
                        "n_seeds": v["n_seeds_used"]})
    res["seen_pair_own_vs_exported_mismatch"] = {"checked": tot, "mismatch": mism}
    # 그림 3a 재계산: 무작위 커버리지 집합(B별 3개 집합)
    table = {}
    for B in (6, 7, 8, 10, 12, 16, 20, 24):
        agg = defaultdict(lambda: [0, 0.0])
        for k, v in d.items():
            if k.startswith("random_coverage") and v["n_train"] == B:
                for c in v["cells"]:
                    if not c["is_id"]:
                        agg[c["seen_pairs"]][0] += 1
                        agg[c["seen_pairs"]][1] += c["mean_success"]
        table[B] = {str(s): {"n_tasks": n, "mean_pct": round(t / n, 1)} for s, (n, t) in sorted(agg.items())}
        # 한 B 안에서 본 쌍 수와 과제별 성공률의 순위상관
        xs, ys = [], []
        for k, v in d.items():
            if k.startswith("random_coverage") and v["n_train"] == B:
                for c in v["cells"]:
                    if not c["is_id"]:
                        xs.append(c["seen_pairs"]); ys.append(c["mean_success"])
        rho, p = stats.spearmanr(xs, ys)
        table[B]["spearman_rho"] = round(float(rho), 2)
        table[B]["spearman_p"] = float(f"{p:.2g}")
    res["fig3a_recomputed_random_sets"] = table
    # 설계 수준: OOD 성공률 대 B, 대 평균 본 쌍 수, 대 3쌍 비율
    ds = [x for x in designs if x["ood_rate"] is not None]
    for key in ("B", "mean_seen", "frac3"):
        rho, p = stats.spearmanr([x[key] for x in ds], [x["ood_rate"] for x in ds])
        res[f"design_level_spearman_ood_vs_{key}"] = {"rho": round(float(rho), 2), "p": float(f"{p:.2g}"), "n_designs": len(ds)}
    # 같은 B 안에서 설계별 변동 (B=8, 10 등 설계가 여러 개인 곳)
    byB = defaultdict(list)
    for x in ds:
        byB[x["B"]].append(x)
    res["designs_per_B"] = {str(B): [{"name": x["name"], "ood": x["ood_rate"], "mean_seen": round(x["mean_seen"], 2)} for x in v]
                            for B, v in sorted(byB.items()) if len(v) >= 4}
    # 같은 B와 같은 고유 쌍 수(세 종류 모두)인 설계끼리의 OOD 성공률 범위, 시드 SD(설계별)와 비교
    grp = defaultdict(list)
    for k, v in d.items():
        if k == "full":
            continue
        tr = [parse_task(t) for t in v["train_tasks"]]
        key = (v["n_train"], len({(t[0], t[1]) for t in tr}), len({(t[0], t[2]) for t in tr}), len({(t[1], t[2]) for t in tr}))
        ood_cells = [c for c in v["cells"] if not c["is_id"]]
        n = min(len(c["per_seed"]) for c in ood_cells)
        ps = [float(np.mean([c["per_seed"][i] for c in ood_cells])) for i in range(n)]
        grp[key].append({"name": k, "ood": v["ood_rate"], "seed_sd": round(float(np.std(ps, ddof=1)), 1) if n > 1 else None, "n_seeds": n})
    res["same_B_same_unique_pairs_groups"] = {
        f"B={k[0]} (oc,ob,cb)=({k[1]},{k[2]},{k[3]})": {"designs": sorted(v, key=lambda x: x["ood"]), "range_pp": round(max(x["ood"] for x in v) - min(x["ood"] for x in v), 1)}
        for k, v in sorted(grp.items()) if len(v) >= 3 and k[1] == k[2] == k[3] == k[0]}
    # 부분 상관: B를 통제한 뒤 평균 본 쌍 수의 기울기 (OLS, 설계 단위)
    X = np.array([[1, x["B"], x["mean_seen"]] for x in ds], float)
    y = np.array([x["ood_rate"] for x in ds], float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / (len(y) - 3)
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    res["design_level_ols_ood_on_B_and_mean_seen"] = {"coef": [round(float(b), 2) for b in beta], "se": [round(float(s), 2) for s in se],
                                                        "n": len(y), "note": "설계 52개, 단순 OLS, 설계 간 독립 가정"}
    return res


# ---------------------------------------------------------------- V7: 파인튜닝 (Q2)
def v7_finetune():
    ft = read_csv("q2_data_ft_success_vs_demos.csv")
    row = {(r["pretrain"], r["group"], r["n_demos"]): r for r in ft}
    res = {}
    # 내보낸 Wilson 구간으로 분모를 역산한다: 사전학습 기준과 처음부터 학습은 60과제 x 30, 파인튜닝은 60과제 x 15
    def implied_n(r, col):
        p, lo, hi = float(r[col + "_rate"]), float(r[col + "_ci_lo"]), float(r[col + "_ci_hi"])
        best = None
        for n in range(100, 7000, 10):
            k = round(p / 100 * n)
            if abs(100 * k / n - p) > 0.01:
                continue
            a, b = wilson(k, n)
            err = abs(100 * a - lo) + abs(100 * b - hi)
            if best is None or err < best[0]:
                best = (err, n)
        return best[1] if best and best[0] < 0.02 else None
    pre = row[("diag_B4", "pretrain (no FT)", "0")]
    after = row[("diag_B4", "full", "64")]
    n0, n1 = implied_n(pre, "ood"), implied_n(after, "ood")
    res["implied_denominators_ood"] = {"pretrain_no_FT": n0, "full_FT_64": n1,
                                       "scratch_64": implied_n([r for r in ft if r["pretrain"] == "scratch_vs_diag_B4" and r["n_demos"] == "64"][0], "ood"),
                                       "all_FT_rows": sorted({implied_n(r, "ood") for r in ft if r["group"] in ("full", "orth") and r["ood_rate"] != "nan"}, key=lambda x: (x is None, x))}
    k0 = round(float(pre["ood_rate"]) / 100 * n0)
    k1 = round(float(after["ood_rate"]) / 100 * n1)
    d, lo, hi = newcombe(k1, n1, k0, n0)
    res["headline"] = {
        "before_pct": float(pre["ood_rate"]), "after_pct": float(after["ood_rate"]),
        "n_before": n0, "n_after": n1, "k_before": k0, "k_after": k1,
        "wilson_before": [pct(x) for x in wilson(k0, n0)], "wilson_after": [pct(x) for x in wilson(k1, n1)],
        "seed_se_after": float(after["ood_se"]),
        "t_interval_after_df2": [round(float(after["ood_rate"]) - stats.t.ppf(0.975, 2) * float(after["ood_se"]), 1),
                                  round(float(after["ood_rate"]) + stats.t.ppf(0.975, 2) * float(after["ood_se"]), 1)],
    }
    # 같은 총 데이터에서 처음부터 학습(scratch)과 비교: 64, 512 시연
    sc = {r["n_demos"]: r for r in ft if r["pretrain"] == "scratch_vs_diag_B4"}
    cmp = {}
    for nd in ("64", "128", "256", "512"):
        a = row[("diag_B4", "full", nd)]
        s = sc[nd]
        sd_a = float(a["ood_se"]) * math.sqrt(3)
        sd_s = float(s["ood_se"]) * math.sqrt(3)
        t, p = welch_stats(float(a["ood_rate"]), sd_a, 3, float(s["ood_rate"]), sd_s, 3)
        cmp[nd] = {"ft_ood": float(a["ood_rate"]), "scratch_ood": float(s["ood_rate"]), "welch_p_seed_level": round(p, 3)}
    res["ft_vs_scratch_ood"] = cmp
    # ID 감소
    res["id_after_ft"] = {nd: float(row[("diag_B4", "full", nd)]["id_rate"]) for nd in ("64", "128", "256", "512")}
    res["id_before"] = float(pre["id_rate"])
    # 직교 FT vs 전체 FT, 같은 시연 수(64): 직교는 과제 16개 × 4
    res["orth_vs_full_ft_ood_64demos"] = {"orth": float(row[("diag_B4", "orth", "64")]["ood_rate"]), "full": float(after["ood_rate"])}
    # 16/32 시연: 직교 대 무작위 5집합 (보류 과제 기준)
    rv = read_csv("q2_data_random_vs_orthogonal_summary.csv")
    cmp2 = {}
    for ckpt in ("diag_B4", "diag_03"):
        for nd in ("16", "32"):
            ort = [float(r["rate"]) for r in rv if r["ckpt"] == ckpt and r["n_demos"] == nd and r["kind"] == "orthogonal" and r["metric"] == "ft_heldout"][0]
            rnd = [float(r["rate"]) for r in rv if r["ckpt"] == ckpt and r["n_demos"] == nd and r["kind"] == "random" and r["metric"] == "ft_heldout"]
            nho = [int(r["n_tasks"]) for r in rv if r["ckpt"] == ckpt and r["n_demos"] == nd and r["kind"] == "orthogonal" and r["metric"] == "ft_heldout"][0]
            nhr = int([r["n_tasks"] for r in rv if r["ckpt"] == ckpt and r["n_demos"] == nd and r["kind"] == "random" and r["metric"] == "ft_heldout"][0])
            rank = 1 + sum(x > ort for x in rnd)
            cmp2[f"{ckpt}_{nd}"] = {"orth_heldout": ort, "random_heldout": rnd, "orth_rank_among_6": rank,
                                    "one_sided_exact_p": round(rank / 6, 3), "n_heldout_tasks_orth": nho, "n_heldout_tasks_random": nhr}
    res["orth_vs_random_ft_heldout"] = cmp2
    return res


# ---------------------------------------------------------------- V8/V9: Q3 (2S-PP)
SANKEY48 = {  # 그림 14 아님, 그림 16 (48과제) 전사값: (start, pick_fail, place_fail, wrong, s2_fail, 그림의 최종 성공 %)
    ("Cov6", "ID"): (800, 36, 229, 46, 147, 42.8), ("Cov6", "OOD"): (1600, 405, 429, 164, 455, 9.2),
    ("Cov9", "ID"): (800, 48, 238, 39, 140, 41.9), ("Cov9", "OOD"): (1600, 182, 534, 184, 398, 18.9),
    ("Cov12", "ID"): (800, 36, 282, 34, 170, 34.8), ("Cov12", "OOD"): (1600, 99, 557, 116, 414, 25.9),
    ("Cov16", "ID"): (800, 34, 221, 55, 184, 38.3), ("Cov16", "OOD"): (1600, 106, 468, 140, 383, 31.4),
    ("Full", "ID"): (2400, 147, 758, 182, 489, 34.3),
}
SANKEY36 = {  # 그림 14 (36과제): 시작 수와 최종 성공 %
    ("Cov4", "ID"): (360, 39.7), ("Cov4", "OOD"): (720, 1.2),
    ("Cov6", "ID"): (360, 33.6), ("Cov6", "OOD"): (720, 7.5),
    ("Cov9", "ID"): (360, 34.4), ("Cov9", "OOD"): (720, 16.1),
    ("Cov12", "ID"): (360, 31.7), ("Cov12", "OOD"): (720, 18.5),
    ("Full", "ID"): (1080, 37.4),
}
COND = {"Cov6": "cov6", "Cov9": "cov9", "Cov12": "cov12", "Cov16": "orthogonal", "Full": "full", "Cov4": "cov4"}


def load_runs(path):
    rows = read_csv(path)
    out = defaultdict(list)
    for r in rows:
        out[(r["condition"], r["split"])].append(r)
    return out


def v8_q3_48():
    runs = load_runs("q3_48tasks_data_l4_2s_no_cont0_runs.csv")
    res = {"sankey_chain": {}, "csv_vs_sankey": {}}
    for (cname, split), (start, pick, place, wrong, s2, shown) in SANKEY48.items():
        succ = start - pick - place - wrong - s2
        res["sankey_chain"][f"{cname}_{split}"] = {"success_count": succ, "success_pct": round(100 * succ / start, 1), "shown_pct": shown,
                                                   "pass": abs(100 * succ / start - shown) < 0.06}
        rs = runs[(COND[cname], split.lower())]
        n = sum(int(r["n_episodes"]) for r in rs)
        pooled = lambda col: sum(float(r[col]) * int(r["n_episodes"]) / 100 for r in rs)
        res["csv_vs_sankey"][f"{cname}_{split}"] = {
            "n": n, "n_shown": start,
            "pick_fail_csv": round(n - pooled("pc_pick_obj0")), "pick_fail_shown": pick,
            "wrong_csv": round(pooled("pc_same_container_violation")), "wrong_shown": wrong,
            "success_csv_pct": round(100 * pooled("pc_success") / n, 1), "success_shown_pct": shown,
        }
    # 조건부 위반율: 그림 15 오른쪽 막대 = 위반 / '두었다'(어디든)
    cond = {}
    for (cname, split) in SANKEY48:
        rs = runs[(COND[cname], split.lower())]
        vals = [float(r["pc_same_container_given_place_obj0"]) for r in rs]
        pooled_den = sum(float(r["pc_place_obj0"]) * int(r["n_episodes"]) / 100 for r in rs)
        pooled_num = sum(float(r["pc_same_container_violation"]) * int(r["n_episodes"]) / 100 for r in rs)
        cond[f"{cname}_{split}"] = {"mean_of_seeds": round(float(np.mean(vals)), 2), "sem": round(float(stats.sem(vals)), 2),
                                    "pooled": round(100 * pooled_num / pooled_den, 2),
                                    "all_rollouts_denominator_pct": round(float(np.mean([float(r["pc_same_container_violation"]) for r in rs])), 2)}
    res["same_container_rates"] = cond
    # 성공률 추세와 검정 (시드 5, OOD)
    ood = {c: [float(r["pc_success"]) for r in runs[(COND[c], "ood")]] for c in ("Cov6", "Cov9", "Cov12", "Cov16")}
    idd = {c: [float(r["pc_success"]) for r in runs[(COND[c], "id")]] for c in ("Cov6", "Cov9", "Cov12", "Cov16", "Full")}
    res["ood_success_by_cov"] = {c: {"mean": round(float(np.mean(v)), 2), "sem": round(float(stats.sem(v)), 2)} for c, v in ood.items()}
    res["id_success_by_cov"] = {c: {"mean": round(float(np.mean(v)), 2), "sem": round(float(stats.sem(v)), 2)} for c, v in idd.items()}
    pairs = {}
    for a, b in (("Cov6", "Cov9"), ("Cov9", "Cov12"), ("Cov12", "Cov16"), ("Cov6", "Cov16")):
        t, p = welch(ood[a], ood[b])
        pairs[f"OOD {a} vs {b}"] = {"diff_pp": round(float(np.mean(ood[b]) - np.mean(ood[a])), 1), "welch_p": round(p, 3)}
    for a, b in (("Cov6", "Cov16"), ("Cov6", "Full")):
        t, p = welch(idd[a], idd[b])
        pairs[f"ID {a} vs {b}"] = {"diff_pp": round(float(np.mean(idd[b]) - np.mean(idd[a])), 1), "welch_p": round(p, 3)}
    # Cov16 OOD 대 Full (같은 정책 종류)
    t, p = welch(ood["Cov16"], idd["Full"])
    pairs["Cov16 OOD vs Full"] = {"diff_pp": round(float(np.mean(idd["Full"]) - np.mean(ood["Cov16"])), 1), "welch_p": round(p, 3)}
    res["pairwise_welch"] = pairs
    # 추세: Cov 수준(6, 9, 12, 16)에 대한 OOD 성공의 선형 기울기, 시드 20개
    xs = [6] * 5 + [9] * 5 + [12] * 5 + [16] * 5
    ys = sum([ood[c] for c in ("Cov6", "Cov9", "Cov12", "Cov16")], [])
    lr = stats.linregress(xs, ys)
    res["ood_linear_trend"] = {"slope_pp_per_pair": round(float(lr.slope), 2), "p": float(f"{lr.pvalue:.2g}"), "r": round(float(lr.rvalue), 2)}
    res["headline_ratio_9.2_to_31.4"] = round(float(np.mean(ood["Cov16"]) / np.mean(ood["Cov6"])), 2)
    # 시드 단위 Cov6 OOD 성공 9.2%, Cov16 31.4% 재현
    return res


def v9_q3_36():
    runs = load_runs("q3_36tasks_data_l4_2s_no_cont0_runs.csv")
    res = {"seeds_per_condition": {}, "n_per_seed": {}, "csv_vs_sankey": {}}
    for (c, s), rs in sorted(runs.items()):
        res["seeds_per_condition"][f"{c}_{s}"] = sorted({r["seed"] for r in rs})
        res["n_per_seed"][f"{c}_{s}"] = sorted({int(r["n_episodes"]) for r in rs})
    for (cname, split), (start, shown) in SANKEY36.items():
        rs = [r for r in runs[(COND[cname], split.lower())] if r["seed"] in ("0", "1", "2")]
        n = sum(int(r["n_episodes"]) for r in rs)
        succ = sum(float(r["pc_success"]) * int(r["n_episodes"]) for r in rs) / n
        res["csv_vs_sankey"][f"{cname}_{split}"] = {"n_csv_seeds012": n, "n_shown": start, "success_csv_pct": round(succ, 1), "success_shown_pct": shown}
    # 36과제: 과제당 시도 수 = 360 / 36 (시드당)
    res["trials_per_task_per_seed_implied"] = 360 / 36
    ood = {c: [float(r["pc_success"]) for r in runs[(COND[c], "ood")] if r["seed"] in ("0", "1", "2")] for c in ("Cov4", "Cov6", "Cov9", "Cov12")}
    res["ood_success"] = {c: {"mean": round(float(np.mean(v)), 2), "sem": round(float(stats.sem(v)), 2)} for c, v in ood.items()}
    t, p = welch(ood["Cov4"], ood["Cov12"])
    res["ood_cov4_vs_cov12_welch_p"] = round(p, 4)
    idd = {c: [float(r["pc_success"]) for r in runs[(COND[c], "id")] if r["seed"] in ("0", "1", "2")] for c in ("Cov4", "Cov6", "Cov9", "Cov12")}
    res["id_success"] = {c: {"mean": round(float(np.mean(v)), 2), "sem": round(float(stats.sem(v)), 2)} for c, v in idd.items()}
    t, p = welch(idd["Cov4"], idd["Cov12"])
    res["id_cov4_vs_cov12_welch_p"] = round(p, 3)
    cond = {}
    for c in ("Cov4", "Cov6", "Cov9", "Cov12"):
        for split in ("ood", "id"):
            v = [float(r["pc_same_container_given_place_obj0"]) for r in runs[(COND[c], split)] if r["seed"] in ("0", "1", "2")]
            cond[f"{c}_{split}"] = round(float(np.mean(v)), 1)
    res["conditional_violation"] = cond
    return res


# ---------------------------------------------------------------- V10: 그림 9b, 모듈러 정책과 우연 수준
def v10_modular():
    rows = read_csv("assets_data_modularity_comparison.csv")
    res = {}
    by = {(r["panel"], r["condition"], r["split"], r["policy"]): r for r in rows}
    mod = by[("b", "Cov16", "ood", "active-instruction-gt-stage")]
    ful = by[("b", "Cov16", "ood", "full-instruction")]
    m1, s1, n1 = float(mod["mean"]), float(mod["err"]), int(mod["n_replicates"])  # SD, n=2
    m2, sem2, n2 = float(ful["mean"]), float(ful["err"]), int(ful["n_replicates"])
    s2 = sem2 * math.sqrt(n2)
    t, p = welch_stats(m1, s1, n1, m2, s2, n2)
    res["cov16_ood_modular_vs_full"] = {"modular": [m1, s1, n1], "full": [m2, round(s2, 2), n2], "welch_p": float(f"{p:.2g}")}
    # 정책이 c2를 모른다면: c1 은 학습 시연에서 c2와 다른 세 컨테이너 중 균등 → 주변분포(c2 평균)는 네 컨테이너에 균등 → 위반 확률 1/4
    res["chance_if_c2_ignored_pct"] = 25.0
    res["modular_observed"] = {"Cov16_ID": float(by[("b", "Cov16", "id", "active-instruction-gt-stage")]["mean"]),
                               "Cov16_OOD": m1, "Full_ID": float(by[("b", "Full", "id", "active-instruction-gt-stage")]["mean"])}
    # 오차 막대 규약이 다름: 모듈러는 SD, 전체 지시는 SEM
    res["err_kinds"] = {k[3]: v["err_kind"] for k, v in by.items() if k[0] == "b"}
    # PPP 패널 (a): 모듈러(오라클 단계) 대 전체 지시(무작위)
    pa = {}
    for B in ("B4", "B8", "B16", "B24"):
        a = by[("a", "PPP", "ood", "active-instruction-gt-stage")] if False else [r for r in rows if r["panel"] == "a" and r["condition"] == B and r["policy"] == "active-instruction-gt-stage"][0]
        f = [r for r in rows if r["panel"] == "a" and r["condition"] == B and r["policy"] == "full-instruction-random"][0]
        pa[B] = {"modular": float(a["mean"]), "full_random": float(f["mean"])}
    res["ppp_modular_vs_full_random"] = pa
    return res


# ---------------------------------------------------------------- V11: 표 1 (시도 수 민감도)
def v11_table1():
    tbl = {5: (56.25, 15.00), 10: (63.75, 15.70), 15: (62.92, 16.30), 30: (61.88, 15.89)}
    res = {"diff_15_to_30": {"id": round(tbl[15][0] - tbl[30][0], 2), "ood": round(tbl[15][1] - tbl[30][1], 2)}}
    # 정수 가능성: 2S-PP 48과제(ID 16, OOD 32 과제) 또는 36과제(ID 12, OOD 24), 시드 1/3/5, 같은 분모 가정(시드 평균 = 합산)
    feas = {}
    for label, (nid, nood) in {"48set": (16, 32), "36set": (12, 24)}.items():
        for seeds in (1, 3, 5):
            ok = []
            for t, (ids, oods) in tbl.items():
                a = bool(attainable(ids, nid * t * seeds, 2))
                b = bool(attainable(oods, nood * t * seeds, 2))
                ok.append((t, a, b))
            feas[f"{label}_seeds{seeds}"] = ok
    res["attainable_(trials,ID,OOD)"] = feas
    # 어떤 (시드 수, 과제 수)에서든 4행 8칸이 모두 가능한 조합 탐색
    combos = []
    for nid in range(4, 25):
        for nood in range(8, 49):
            for seeds in (1, 2, 3, 4, 5):
                if all(attainable(ids, nid * t * seeds, 2) and attainable(oods, nood * t * seeds, 2) for t, (ids, oods) in tbl.items()):
                    combos.append((nid, nood, seeds))
    res["all_cells_attainable_combos_nid_nood_seeds"] = combos[:20]
    res["n_combos_found"] = len(combos)
    # 30회 대 15회 차이의 통계적 크기: 48과제 5시드 가정(ID 800/OOD 1600 롤아웃 @10회; 시도 수에 비례)
    for t, label in ((15, "15"), (30, "30")):
        pass
    k_id = lambda t, nid=16, seeds=5: round(tbl[t][0] / 100 * nid * t * seeds)
    n_id30 = 16 * 30 * 5
    res["wilson_halfwidth_id_30trials_48set_5seeds_pp"] = round(100 * (wilson(round(0.6188 * n_id30), n_id30)[1] - wilson(round(0.6188 * n_id30), n_id30)[0]) / 2, 2)
    n_id30s = 16 * 30
    res["wilson_halfwidth_id_30trials_48set_1seed_pp"] = round(100 * (wilson(round(0.6188 * n_id30s), n_id30s)[1] - wilson(round(0.6188 * n_id30s), n_id30s)[0]) / 2, 2)
    return res


# ---------------------------------------------------------------- V12: 사람 시연 (부록 C)
def v12_human():
    res = {}
    for seeds in (1, 2, 3):
        nid, nood = 9 * 30 * seeds, 18 * 30 * seeds
        res[f"seeds{seeds}"] = {"n_id": nid, "k_id_for_11.67": attainable(11.67, nid), "n_ood": nood, "k_ood_for_12.96": attainable(12.96, nood)}
    n_id, n_ood = 810, 1620
    res["nearest_to_11.67_at_810"] = [round(100 * 94 / 810, 3), round(100 * 95 / 810, 3)]
    k_id, k_ood = round(0.1167 * n_id), 210
    d, lo, hi = newcombe(k_ood, n_ood, k_id, n_id)
    res["ood_minus_id_pp"] = pct(d)
    res["newcombe95_pp"] = [pct(lo), pct(hi)]
    res["wilson_ood_12.96_n1620"] = [pct(x) for x in wilson(210, 1620)]
    # 무작위 정책 대비 우연 수준은 논문에 없다 → 0%와의 차이만 말할 수 있다
    return res


# ---------------------------------------------------------------- V13: PP 부분 입력 (부록 E)
def v13_pp_omit():
    out = {}
    for label, (m1, s1, m2, s2) in {"SD로 읽음": (71.05, 5.00, 49.95, 1.14)}.items():
        t, p = welch_stats(m1, s1, 5, m2, s2, 5)
        out[label] = {"welch_p": float(f"{p:.2g}")}
    t, p = welch_stats(71.05, 5.00 * math.sqrt(5), 5, 49.95, 1.14 * math.sqrt(5), 5)
    out["SEM으로 읽음(SD = SEM·√5)"] = {"welch_p": float(f"{p:.2g}")}
    out["diff_pp"] = round(71.05 - 49.95, 2)
    # PP full 성공률과의 비교: 차트 CSV PP 전체 B=16 평균
    pp = [r for r in read_csv("q1_data_l4_1s_success_vs_b.csv") if r["b"] == "16" and r["condition"] == "full" and r["metric"] == "total"][0]
    out["q1_pp_full_chart_mean"] = float(pp["mean"])
    return out


# ---------------------------------------------------------------- V14: 정리 4.1 수치 검사
def v14_theory():
    rng = np.random.default_rng(0)
    res = {}
    # (a) 임의의 이산 분포에서 정확 분해 Δ = A + B + C 와 부등식 확인
    n_l = (3, 4, 3)
    nz = 5
    worst_gap = 0.0
    viol = 0
    trials = 200
    for _ in range(trials):
        L = list(itertools.product(*[range(n) for n in n_l]))
        # 훈련 지원은 부분 집합
        keep = rng.random(len(L)) < 0.5
        keep[0] = True
        pl = np.where(keep, rng.random(len(L)), 0.0)
        pl /= pl.sum()
        ql = rng.random(len(L))
        ql /= ql.sum()
        pza = rng.dirichlet(np.ones(nz), size=len(L))
        qza = rng.dirichlet(np.ones(nz), size=len(L))
        loss = rng.random((len(L), nz))
        M = loss.max()
        P = pl[:, None] * pza
        Q = ql[:, None] * qza
        delta = float(((Q - P) * loss).sum())
        i = 0
        li = np.array([l[i] for l in L])
        # 하계 항: ||q(l_i) - p(l_i)||_1, E_{p(l_i)} ||q(l_-i|l_i) - p(l_-i|l_i)||_1, E_{p(l)} ||q(z,a|l) - p(z,a|l)||_1
        t1 = t2 = 0.0
        for v in range(n_l[0]):
            idx = li == v
            pv, qv = pl[idx].sum(), ql[idx].sum()
            t1 += abs(qv - pv)
            if pv > 0 and qv > 0:
                t2 += pv * np.abs(ql[idx] / qv - pl[idx] / pv).sum()
            elif pv > 0 and qv == 0:
                t2 += pv * 2
        t3 = float((pl * np.abs(qza - pza).sum(axis=1)).sum())
        bound = M * (t1 + t2 + t3)
        if abs(delta) > bound + 1e-12:
            viol += 1
        worst_gap = max(worst_gap, abs(delta) / bound if bound > 0 else 0)
    res["random_discrete_checks"] = {"trials": trials, "violations_of_bound": viol, "max_|delta|/bound": round(worst_gap, 3)}
    # (b) PPP 설계별 식 5의 둘째 항과 첫째 항 (q = 64과제 균등, p = 훈련 과제 균등)
    d = json.load(open(PAGE / "results_data.json", encoding="utf-8"))
    all_tasks = [tuple(parse_task(t)) for t in d["meta"]["all_tasks"]]

    def terms(train):
        tr = [tuple(parse_task(t)) for t in train]
        out = []
        for i in range(3):
            t1 = 0.0
            t2 = 0.0
            levels = sorted({t[i] for t in all_tasks})
            for v in levels:
                pv = sum(1 for t in tr if t[i] == v) / len(tr)
                qv = 1 / len(levels)
                t1 += abs(qv - pv)
                if pv > 0:
                    # 조건부: 남은 두 인자의 결합 (16칸)
                    pc = defaultdict(float)
                    for t in tr:
                        if t[i] == v:
                            pc[tuple(x for j, x in enumerate(t) if j != i)] += 1 / (pv * len(tr))
                    qc = 1 / 16
                    l1 = sum(abs(pc.get(k, 0) - qc) for k in itertools.product(*[sorted({t[j] for t in all_tasks}) for j in range(3) if j != i]))
                    t2 += pv * l1
            out.append((t1, t2))
        return out

    designs = {}
    exps = d["experiments"]
    pick = ["orthogonal", "diag_b4", "diag_03", "random_coverage_s0_b16", "random_coverage_s1_b16", "random_coverage_s2_b16",
            "random_coverage_s0_b8", "random_coverage_s1_b8", "random_coverage_s0_b24", "full"]
    for k in pick:
        tt = terms(exps[k]["train_tasks"])
        designs[k] = {"B": exps[k]["n_train"], "term1_by_factor": [round(a, 3) for a, _ in tt], "term2_by_factor": [round(b, 3) for _, b in tt],
                      "term2_mean": round(float(np.mean([b for _, b in tt])), 3), "ood_rate": exps[k]["ood_rate"]}
    res["ppp_design_terms"] = designs
    # 모든 희소 설계(52개)에서 식 5의 첫째+둘째 항(인자 평균)과 OOD 성공률의 순위상관, B 단독과 비교
    tot, oods, Bs = [], [], []
    for k, v in exps.items():
        if k == "full" or v["ood_rate"] is None:
            continue
        tt = terms(v["train_tasks"])
        tot.append(float(np.mean([a + b for a, b in tt])))
        oods.append(v["ood_rate"])
        Bs.append(v["n_train"])
    rho, p = stats.spearmanr(tot, oods)
    rhoB, pB = stats.spearmanr(Bs, oods)
    res["bound_terms12_vs_ood_all_designs"] = {"spearman_rho": round(float(rho), 2), "p": float(f"{p:.2g}"), "n": len(tot),
                                               "spearman_rho_B_alone": round(float(rhoB), 2)}
    # 같은 B에서 식 5의 항이 같은 설계 묶음 안의 OOD 범위 (B=8, 균형·모두 고유 쌍 설계)
    same = defaultdict(list)
    for k, v in exps.items():
        if k == "full" or v["ood_rate"] is None:
            continue
        tt = terms(v["train_tasks"])
        same[(v["n_train"], round(float(np.mean([a + b for a, b in tt])), 3))].append((k, v["ood_rate"]))
    res["same_B_same_bound_terms_ood_range"] = {f"B={k[0]}, terms12={k[1]}": {"n_designs": len(v), "ood_min": min(x[1] for x in v), "ood_max": max(x[1] for x in v)}
                                                for k, v in sorted(same.items()) if len(v) >= 4}
    # (c) 균형 설계에서 둘째 항의 닫힌 식: 2(1 - 지원/16)
    res["closed_form_balanced_16_task_design"] = 2 * (1 - 4 / 16)
    return res


# ---------------------------------------------------------------- K: 설계, 구조 검사
FIG4 = [  # 그림 4 표 (object, container, button), 원문 4쪽
    ("cross", "bin", "red"), ("cross", "mug", "green"), ("cross", "plate", "blue"), ("cross", "mug_no_handle", "yellow"),
    ("cube", "bin", "green"), ("cube", "mug", "red"), ("cube", "plate", "yellow"), ("cube", "mug_no_handle", "blue"),
    ("cylinder", "bin", "blue"), ("cylinder", "mug", "yellow"), ("cylinder", "plate", "red"), ("cylinder", "mug_no_handle", "green"),
    ("milk", "bin", "yellow"), ("milk", "mug", "blue"), ("milk", "plate", "green"), ("milk", "mug_no_handle", "red"),
]


def k_design():
    res = {}
    objs = ["cross", "cube", "cylinder", "milk"]
    conts = ["bin", "mug", "plate", "mug_no_handle"]
    cols = ["red", "green", "blue", "yellow"]
    cnt = defaultdict(int)
    for o, c, b in FIG4:
        cnt[("oc", o, c)] += 1
        cnt[("ob", o, b)] += 1
        cnt[("cb", c, b)] += 1
    res["fig4_pairs_all_exactly_once"] = all(v == 1 for v in cnt.values()) and len(cnt) == 48
    allt = list(itertools.product(objs, conts, cols))
    held = [t for t in allt if t not in FIG4]
    sp = [sum([any(x[0] == o and x[1] == c for x in FIG4), any(x[0] == o and x[2] == b for x in FIG4), any(x[1] == c and x[2] == b for x in FIG4)]) for o, c, b in held]
    res["fig4_heldout_seen_pair_counts"] = {str(k): sp.count(k) for k in (0, 1, 2, 3)}
    # 보기: 뷰어 데이터의 훈련 목록과 그림 4의 목록이 같은지
    d = json.load(open(PAGE / "results_data.json", encoding="utf-8"))["experiments"]
    viewer = sorted(parse_task(t) for t in d["orthogonal"]["train_tasks"])
    res["viewer_orthogonal_equals_fig4"] = viewer == sorted(FIG4)
    # XOR 구성(코드의 xor_oa_4x4x4)과 동형인가: 각 쌍 정확히 1번인 16행 집합이 라틴 방진인지
    # 일반 하한: 강도 2 직교배열은 q^2 행이 필요 (인자 수와 무관)
    # 그림 4가 c = o XOR container 구성(코드의 xor_oa_4x4x4와 같은 규칙)인지: 색 색인 red 0, green 1, blue 2, yellow 3 가정
    xor = sorted((objs[o], conts[c], cols[o ^ c]) for o in range(4) for c in range(4))
    res["fig4_equals_xor_construction_(color_order_assumed)"] = xor == sorted(FIG4)
    res["min_rows_strength2_q4"] = 16
    res["fraction_of_full_by_n_factors_strength2"] = {str(n): round(16 / 4 ** n, 4) for n in (3, 4, 5, 6)}
    res["fraction_of_full_by_n_factors_strength3"] = {str(n): round(64 / 4 ** n, 4) for n in (3, 4, 5, 6)}
    # 무작위 B개 훈련 집합(모든 값 1회 이상)에서 보류 과제의 본 쌍 수 분포 (몬테카를로 4000회)
    rng = np.random.default_rng(1)
    dist = {}
    for B in (6, 8, 10, 12, 16, 24):
        tot = np.zeros(4)
        n_ok = 0
        while n_ok < 1500:
            idx = rng.choice(64, B, replace=False)
            tr = [allt[i] for i in idx]
            if not (len({t[0] for t in tr}) == 4 and len({t[1] for t in tr}) == 4 and len({t[2] for t in tr}) == 4):
                continue
            n_ok += 1
            oc = {(t[0], t[1]) for t in tr}
            ob = {(t[0], t[2]) for t in tr}
            cb = {(t[1], t[2]) for t in tr}
            for j, t in enumerate(allt):
                if j in set(idx):
                    continue
                tot[(t[:2] in oc) + ((t[0], t[2]) in ob) + ((t[1], t[2]) in cb)] += 1
        dist[str(B)] = [round(float(x / tot.sum()), 3) for x in tot]
    res["random_B_heldout_seen_pair_count_distribution_0to3"] = dist
    # 논문 그림 3a의 막대 수 일치 확인용: B=4 대각 집합
    diag = [(objs[i], conts[i], cols[i]) for i in range(4)]
    spd = [sum([any(x[0] == o and x[1] == c for x in diag), any(x[0] == o and x[2] == b for x in diag), any(x[1] == c and x[2] == b for x in diag)])
           for o, c, b in allt if (o, c, b) not in diag]
    res["diag4_heldout_seen_pair_counts"] = {str(k): spd.count(k) for k in (0, 1, 2, 3)}
    return res


def k_misc():
    res = {}
    d = json.load(open(PAGE / "results_data.json", encoding="utf-8"))
    ex = d["experiments"]
    res["viewer_experiments"] = {"total": len(ex), "sparse": sum(1 for k in ex if k != "full"), "paper_text_v2": 55, "paper_text_v1_body": 53}
    ch = [r for r in read_csv("q1_data_success_vs_b.csv") if r["b"] == "64" and r["metric"] == "total"][0]
    res["full64_chart_mean_vs_viewer"] = {"chart": float(ch["mean"]), "viewer": ex["full"]["id_rate"], "viewer_episodes": ex["full"]["id_episodes"]}
    # HF 구조 (메타데이터만): 시연 풀 크기 = shard 수 × 5 (가정: shard당 5 에피소드)
    res["hf_shards"] = {"PP": {"hdf5": 1600, "tasks": 16}, "PPP": {"hdf5": 3840, "tasks": 64}, "2S": {"hdf5": 2880, "tasks": 144}}
    res["hf_implied_demos_per_task_at_5_per_shard"] = {k: round(v["hdf5"] * 5 / v["tasks"]) for k, v in res["hf_shards"].items()}
    res["paper_demos_per_task"] = {"PP": 500, "PPP": 300, "2S": 100}
    res["paper_pool_sizes"] = {"PP": 500 * 16, "PPP": 300 * 64, "2S": 100 * 144}
    # 고정 총 시연량: 가장 작은 집합이 허용하는 양 (B.1)
    res["fixed_budget_inferred"] = {"PPP_B4_x_300": 4 * 300, "per_task_at_B64": 4 * 300 / 64, "PP_B5_x_500": 5 * 500, "per_task_at_B16": 5 * 500 / 16}
    return res


def main():
    RES["V1_total_identity"] = v1_identity()
    RES["V2_trial_counts"] = v2_trials()
    RES["V3_V4_orth_full_saturation"] = v3_orth_full()
    RES["V5_loss_numbers"] = v5_loss()
    RES["V6_seen_pairs"] = v6_pairs()
    RES["V7_finetune"] = v7_finetune()
    RES["V8_q3_48"] = v8_q3_48()
    RES["V9_q3_36"] = v9_q3_36()
    RES["V10_modular"] = v10_modular()
    RES["V11_table1"] = v11_table1()
    RES["V12_human"] = v12_human()
    RES["V13_pp_omission"] = v13_pp_omit()
    RES["V14_theory"] = v14_theory()
    RES["K_design"] = k_design()
    RES["K_misc"] = k_misc()
    OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(RES, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
