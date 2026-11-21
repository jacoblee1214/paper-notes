"""논문 표 수치를 공식 저장소의 실행별 점수(repo/scores/*.json.gz)로 재계산하고 검정한다.

입력: paper/paper.txt (pdftotext -layout 결과, Table 6 Atari 파싱), repo/scores/*.json.gz (공식 저장소 점수)
출력: analysis/recompute_results.json, 표준출력 요약
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

공식 저장소는 실행 단위 곡선(xs, ys)을 담고 있다. 각 실행의 "최종 점"은 예산 이하의 마지막 기록 시점의 값이다.
"""
import gzip
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "repo" / "scores"
OUT = ROOT / "analysis" / "recompute_results.json"
PLOT_IN = ROOT / ".scratch" / "plot_inputs.json"
RNG = np.random.default_rng(0)
B = 10000


def load(name):
    return json.loads(gzip.open(SCORES / f"{name}.json.gz", "rt").read())


def final_by_run(name, budget, k=1):
    """task -> {seed: 점수}. 예산 이하의 기록 중 null이 아닌 마지막 k개 점의 평균 (k=1이면 마지막 점).

    공식 저장소 일부 실행에는 null 점이 있어 제외한다. 논문은 최종 점수 정의를 적지 않으므로 k를 바꿔 본다.
    """
    out, last_x = {}, []
    for r in load(name):
        xs = np.array([np.nan if v is None else v for v in r["xs"]], float)
        ys = np.array([np.nan if v is None else v for v in r["ys"]], float)
        ok = (~np.isnan(ys)) & (xs <= budget * 1.0001)
        idx = np.flatnonzero(ok)
        out.setdefault(r["task"], {})[r["seed"]] = float(ys[idx[-k:]].mean())
        last_x.append(xs[idx[-1]] / budget)
    return out, np.asarray(last_x)


def task_means(runs):
    return {t: float(np.mean(list(s.values()))) for t, s in runs.items()}


def boot_ci(x, fn=np.mean, b=B):
    x = np.asarray(x, float)
    idx = RNG.integers(0, len(x), size=(b, len(x)))
    stat = np.array([fn(x[i]) for i in idx])
    return float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))


def norm(n):
    return re.sub(r"[^a-z0-9]", "", n.lower())


# ---------- 1. Atari 57 (논문 Table 6) ----------
def parse_atari_table():
    txt = (ROOT / "paper" / "paper.txt").read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(txt) if l.strip() == "Atari scores")
    end = next(i for i, l in enumerate(txt) if "Gamer median" in l)
    num = r"([–\-]?[\d,]+(?:\.\d+)?)"
    pat = re.compile(r"^\s*([A-Z][A-Za-z' ]+?)\s+" + r"\s+".join([num] * 6) + r"\s*$")
    rows = {}
    for l in txt[start:end]:
        m = pat.match(l)
        if m:
            vals = [float(v.replace("–", "-").replace(",", "")) for v in m.groups()[1:]]
            rows[m.group(1).strip()] = dict(zip(["Random", "Gamer", "Record", "PPO", "MuZero", "Dreamer"], vals))
    summary = {}
    for l in txt[end:end + 6]:
        m = re.match(r"^\s*(Gamer median|Gamer mean|Record mean|Record mean capped) \(%\)\s+(.*)$", l)
        if m:
            summary[m.group(1)] = [float(v.replace("–", "-")) for v in m.group(2).split()]
    return rows, summary


def hns(score, rnd, hum):
    return 100.0 * (score - rnd) / (hum - rnd)


def key_of(rows, t):
    return next(g for g in rows if norm(g) == norm(t.replace("atari_", "")))


def atari_block(res):
    rows, summary = parse_atari_table()
    res["V_atari_table_rows"] = len(rows)
    games = sorted(rows)
    ref = {g: rows[g] for g in games}
    # 논문 표 값으로 HNS 중앙값과 평균을 다시 계산
    for alg in ["PPO", "MuZero", "Dreamer"]:
        h = np.array([hns(ref[g][alg], ref[g]["Random"], ref[g]["Gamer"]) for g in games])
        res[f"V_atari_table_{alg}"] = {
            "median_hns_recomputed": round(float(np.median(h)), 1),
            "mean_hns_recomputed": round(float(np.mean(h)), 1),
            "median_hns_reported": summary["Gamer median"][["Random", "Gamer", "Record", "PPO", "MuZero", "Dreamer"].index(alg)],
            "mean_hns_reported": None,
        }
    # 논문 Table 6 "Gamer mean" 행은 파싱 대상 밖이므로 보고값은 PDF 표에서 직접 옮긴다
    res["V_atari_table_Dreamer"]["mean_hns_reported"] = 3381
    res["V_atari_table_PPO"]["mean_hns_reported"] = 892
    res["V_atari_table_MuZero"]["mean_hns_reported"] = 3054

    # 공식 저장소의 DreamerV3 실행별 점수(게임당 3–6 seed)로 다시 계산
    dr, last = final_by_run("atari57-dreamerv3", 2e8, k=5)
    sens = {}
    for k in (1, 3, 5, 10):
        tk, _ = final_by_run("atari57-dreamerv3", 2e8, k=k)
        tm_k = task_means(tk)
        hk = np.array([hns(tm_k[t], rows[key_of(rows, t)]["Random"], rows[key_of(rows, t)]["Gamer"]) for t in tm_k])
        sens[f"last_{k}"] = {"median_hns": round(float(np.median(hk)), 1), "mean_hns": round(float(hk.mean()), 1)}
    res["atari57_dreamer_k_sensitivity"] = sens
    res["atari57_dreamer_runs_per_game"] = {"min": min(len(v) for v in dr.values()), "max": max(len(v) for v in dr.values())}
    res["atari57_dreamer_last_x_over_budget_min"] = round(float(last.min()), 3)
    means = task_means(dr)
    key = {norm(g): g for g in games}
    per = {}
    for t, m in means.items():
        g = key[norm(t.replace("atari_", ""))]
        per[g] = m
    diffs = [abs(per[g] - ref[g]["Dreamer"]) for g in games]
    res["V_atari_released_vs_table_Dreamer_max_abs_diff_raw"] = round(float(max(diffs)), 2)
    res["V_atari_released_vs_table_Dreamer_n_games_gt_1pct"] = int(sum(d > 0.01 * max(1, abs(ref[g]["Dreamer"])) for d, g in zip(diffs, games)))
    h_rel = np.array([hns(per[g], ref[g]["Random"], ref[g]["Gamer"]) for g in games])
    h_tab = np.array([hns(ref[g]["Dreamer"], ref[g]["Random"], ref[g]["Gamer"]) for g in games])
    res["atari57_dreamer_hns_released"] = {
        "median": round(float(np.median(h_rel)), 1),
        "median_ci95_games": [round(v, 1) for v in boot_ci(h_rel, np.median)],
        "mean": round(float(np.mean(h_rel)), 1),
        "mean_ci95_games": [round(v, 1) for v in boot_ci(h_rel)],
        "median_from_table_values": round(float(np.median(h_tab)), 1),
    }
    # 같은 게임 기준으로 PPO(5 seed)와 MuZero(표 값)와 짝지은 비교
    pp, _ = final_by_run("atari57-ppo_fixhp", 2e8, k=5)
    pm = task_means(pp)
    ppo_h = np.array([hns(pm[f"atari_{norm(g)}"] if f"atari_{norm(g)}" in pm else pm[[t for t in pm if norm(t.replace('atari_', '')) == norm(g)][0]], ref[g]["Random"], ref[g]["Gamer"]) for g in games])
    res["atari57_ppo_hns_released"] = {"median": round(float(np.median(ppo_h)), 1), "mean": round(float(np.mean(ppo_h)), 1)}
    dd = h_rel - ppo_h
    w = stats.wilcoxon(h_rel, ppo_h)
    res["atari57_dreamer_vs_ppo_games"] = {"dreamer_better_games": int((dd > 0).sum()), "ppo_better_games": int((dd < 0).sum()),
                                           "wilcoxon_p": float(w.pvalue)}
    mu_h = np.array([hns(ref[g]["MuZero"], ref[g]["Random"], ref[g]["Gamer"]) for g in games])
    w2 = stats.wilcoxon(h_rel, mu_h)
    res["atari57_dreamer_vs_muzero_games"] = {"dreamer_better_games": int((h_rel > mu_h).sum()),
                                              "muzero_better_games": int((h_rel < mu_h).sum()), "wilcoxon_p": float(w2.pvalue)}
    res["_atari_hns_for_plot"] = {"dreamer": h_rel.tolist(), "ppo": ppo_h.tolist(), "muzero": mu_h.tolist(), "games": games}


# ---------- 2. Atari 100k (논문 Table 9): 기준값은 Table 6의 Random과 Gamer ----------
def atari100k_block(res):
    rows, _ = parse_atari_table()
    dr, last = final_by_run("atari100k-dreamerv3", 4e5, k=3)
    pp, _ = final_by_run("atari100k-ppo_fixhp", 4e5, k=3)
    key = {norm(g): g for g in rows}
    dm, pm = task_means(dr), task_means(pp)
    games, hd, hp = [], [], []
    for t in sorted(dm):
        g = key[norm(t.replace("atari_", ""))]
        games.append(g)
        hd.append(hns(dm[t], rows[g]["Random"], rows[g]["Gamer"]))
        hp.append(hns(pm[t], rows[g]["Random"], rows[g]["Gamer"]))
    hd, hp = np.array(hd), np.array(hp)
    res["atari100k_dreamer_hns"] = {"mean": round(float(hd.mean()), 1), "mean_ci95_games": [round(v, 1) for v in boot_ci(hd)],
                                    "median": round(float(np.median(hd)), 1), "median_ci95_games": [round(v, 1) for v in boot_ci(hd, np.median)],
                                    "n_games": len(games), "reported_mean": 125, "reported_median": 49}
    res["atari100k_ppo_hns"] = {"mean": round(float(hp.mean()), 1), "median": round(float(np.median(hp)), 1),
                                "reported_mean": 11, "reported_median": 2}
    w = stats.wilcoxon(hd, hp)
    res["atari100k_dreamer_vs_ppo"] = {"dreamer_better_games": int((hd > hp).sum()), "wilcoxon_p": float(w.pvalue)}
    res["atari100k_dreamer_runs_per_game"] = sorted({len(v) for v in dr.values()})
    res["_atari100k_hns_for_plot"] = {"dreamer": hd.tolist(), "ppo": hp.tolist()}


# ---------- 3. Minecraft (논문 Table 5, Figure 5) ----------
def minecraft_block(res):
    names = {"dreamerv3": "Dreamer", "impala": "IMPALA", "rainbow": "Rainbow", "ppo_fixhp": "PPO"}
    reported = {"Dreamer": 9.1, "IMPALA": 7.1, "Rainbow": 6.3, "PPO": 5.1}
    vals, mx = {}, {}
    for k, label in names.items():
        runs, last = final_by_run(f"minecraft_diamond-{k}", 1e8, k=1)
        v = np.array(list(runs["minecraft_diamond"].values()))
        vals[label] = v
        res.setdefault("minecraft_last_x_over_100M_min", {})[label] = round(float(last.min()), 3)
        maxes = []
        for r in load(f"minecraft_diamond-{k}"):
            maxes.append(max(r["ys"]))
        mx[label] = np.array(maxes)
    out = {}
    for label, v in vals.items():
        out[label] = {"n_runs": int(len(v)), "mean_final": round(float(v.mean()), 2), "sd": round(float(v.std(ddof=1)), 2),
                      "mean_ci95": [round(c, 2) for c in boot_ci(v)], "reported_table5": reported[label],
                      "runs_reaching_12_items_at_some_point": int((mx[label] >= 11.5).sum())}
    res["minecraft"] = out
    d = vals["Dreamer"]
    res["minecraft_welch_vs"] = {b: {"p": float(stats.ttest_ind(d, vals[b], equal_var=False).pvalue)} for b in ["IMPALA", "Rainbow", "PPO"]}
    # 다이아몬드(반환 12 = 12개 아이템 전부) 도달 실행 수 비교: 반환이 아이템 수라는 가정 하에서
    dia_d = int((mx["Dreamer"] >= 11.5).sum())
    dia_b = int(sum((mx[b] >= 11.5).sum() for b in ["IMPALA", "Rainbow", "PPO"]))
    n_b = int(sum(len(mx[b]) for b in ["IMPALA", "Rainbow", "PPO"]))
    res["minecraft_diamond_fisher"] = {"dreamer": [dia_d, len(mx["Dreamer"])], "baselines": [dia_b, n_b],
                                       "fisher_p": float(stats.fisher_exact([[dia_d, len(mx["Dreamer"]) - dia_d], [dia_b, n_b - dia_b]])[1]),
                                       "assumption": "반환값이 획득 아이템 수(최대 12)라고 가정"}
    res["_minecraft_plot"] = {k: v.tolist() for k, v in vals.items()}


# ---------- 4. DMC (논문 Table 11, 12) ----------
def dmc_block(res):
    spec = {
        "dmc_proprio": ("dmc_proprio-{}", 5e5, {"ppo_fixhp": "PPO", "ddpg": "DDPG", "dmpo": "DMPO", "d4pg": "D4PG", "dreamerv3": "Dreamer", "mpo": "MPO"},
                        {"PPO": 94, "DDPG": 771, "DMPO": 801, "D4PG": 792, "Dreamer": 871}),
        "dmc_vision": ("dmc_vision-{}", 1e6, {"ppo_fixhp": "PPO", "sac": "SAC", "curl": "CURL", "drqv2": "DrQ-v2", "dreamerv3": "Dreamer"},
                       {"PPO": 94, "SAC": 81, "CURL": 479, "DrQ-v2": 770, "Dreamer": 861}),
    }
    for suite, (pat, budget, methods, rep) in spec.items():
        per = {}
        for k, label in methods.items():
            try:
                runs, _ = final_by_run(pat.format(k), budget)
            except FileNotFoundError:
                continue
            per[label] = task_means(runs)
            res.setdefault(f"{suite}_seeds_per_task", {})[label] = [min(len(s) for s in runs.values()), max(len(s) for s in runs.values())]
        tasks = sorted(per["Dreamer"])
        summ = {}
        for label, tm in per.items():
            v = np.array([tm[t] for t in tasks if t in tm])
            summ[label] = {"n_tasks": int(len(v)), "task_mean": round(float(v.mean()), 0), "task_median": round(float(np.median(v)), 0)}
            if label in rep:
                summ[label]["reported_mean"] = rep[label]
        res[f"{suite}_summary"] = summ
        dd = np.array([per["Dreamer"][t] - per["DrQ-v2" if suite == "dmc_vision" else "DMPO"][t] for t in tasks])
        other = "DrQ-v2" if suite == "dmc_vision" else "DMPO"
        w = stats.wilcoxon([per["Dreamer"][t] for t in tasks], [per[other][t] for t in tasks])
        res[f"{suite}_dreamer_vs_{other}"] = {"dreamer_better_tasks": int((dd > 0).sum()), "n_tasks": len(tasks),
                                              "mean_diff": round(float(dd.mean()), 1), "diff_ci95": [round(c, 1) for c in boot_ci(dd)],
                                              "wilcoxon_p": float(w.pvalue)}
        if suite == "dmc_vision":
            res["_dmc_vision_plot"] = {"tasks": tasks, "dreamer": [per["Dreamer"][t] for t in tasks], "drqv2": [per["DrQ-v2"][t] for t in tasks]}
        else:
            res["_dmc_proprio_plot"] = {"tasks": tasks, "dreamer": [per["Dreamer"][t] for t in tasks]}


# ---------- 5. ProcGen (논문 Table 7) 와 DMLab (논문 Table 8) ----------
def parse_rows(start_pat, end_pat, ncols):
    """표 한 줄 = 이름 + 숫자 ncols개. 시작과 끝 패턴 사이에서만 읽는다."""
    txt = (ROOT / "paper" / "paper.txt").read_text(encoding="utf-8").splitlines()
    s0 = next(i for i, l in enumerate(txt) if re.search(start_pat, l))
    s1 = next(i for i, l in enumerate(txt) if i > s0 and re.search(end_pat, l))
    num = r"([–\-]?\d[\d,]*(?:\.\d+)?)"
    pat = re.compile(r"^\s*([A-Z][A-Za-z' ]+?)\s+" + r"\s+".join([num] * ncols) + r"\s*$")
    rows = {}
    for l in txt[s0:s1]:
        m = pat.match(l)
        if m:
            rows[m.group(1).strip()] = [float(v.replace("–", "-").replace(",", "")) for v in m.groups()[1:]]
    return rows


def procgen_block(res):
    dr, _ = final_by_run("procgen-dreamerv3", 5e7, k=3)
    pg, _ = final_by_run("procgen-ppg", 5e7, k=3)
    pp, _ = final_by_run("procgen-ppo_fixhp", 5e7, k=3)
    dm, gm, pm = task_means(dr), task_means(pg), task_means(pp)
    tasks = sorted(dm)
    tab = parse_rows(r"^\s*ProcGen scores", r"^\s*Table 7:", 4)  # Original PPO, PPO, PPG, Dreamer
    res["procgen_table_rows_parsed"] = len(tab)
    mapt = {norm(k): v for k, v in tab.items()}
    diffs = []
    for t in tasks:
        v = mapt[norm(t.replace("procgen_", ""))]
        diffs.append(abs(dm[t] - v[3]))
    res["V_procgen_dreamer_released_vs_table_max_abs_diff"] = round(float(max(diffs)), 2)
    res["V_procgen_dreamer_released_vs_table_n_eq"] = int(sum(d < 0.01 for d in diffs))
    res["procgen_dreamer_seeds"] = sorted({len(v) for v in dr.values()})
    dd = np.array([dm[t] - gm[t] for t in tasks])
    res["procgen_dreamer_vs_ppg"] = {"dreamer_better_tasks": int((dd > 0).sum()), "n_tasks": len(tasks),
                                     "sign_test_p": float(stats.binomtest(int((dd > 0).sum()), len(tasks)).pvalue),
                                     "wilcoxon_p": float(stats.wilcoxon([dm[t] for t in tasks], [gm[t] for t in tasks]).pvalue)}
    res["procgen_reported_normalized_mean"] = {"Original PPO": 41.16, "PPO": 42.80, "PPG": 64.89, "Dreamer": 66.01}
    res["V_procgen_normalized_mean"] = "논문 표 7 정의 없음(기준 점수 미기재), 재계산 불가"


def dmlab_block(res):
    dr, last = final_by_run("dmlab30-dreamerv3", 1e8, k=5)
    dr1, _ = final_by_run("dmlab30-dreamerv3", 1e8, k=1)
    tab = parse_rows(r"^\s*DMLab scores", r"^\s*Human mean capped", 6)  # R2D2+, IMPALA 10B, IMPALA 1B, IMPALA 100M, PPO, Dreamer
    mapt = {norm(k): v for k, v in tab.items()}
    rel5, rel1 = task_means(dr), task_means(dr1)
    d5 = [abs(rel5[t] - mapt[norm(t.replace("dmlab_", ""))][5]) for t in rel5]
    d1 = [abs(rel1[t] - mapt[norm(t.replace("dmlab_", ""))][5]) for t in rel1]
    res["dmlab_table_rows_parsed"] = len(tab)
    res["dmlab_released_vs_table_dreamer_median_abs_diff"] = {"last5": round(float(np.median(d5)), 2), "last1": round(float(np.median(d1)), 2),
                                                             "n_tasks_matched": len(d5)}
    res["dmlab_last_x_over_budget_min"] = round(float(last.min()), 3)
    res["dmlab_reported_human_mean_capped"] = {"Dreamer_100M": 71.4, "IMPALA_1B": 66.3, "PPO_100M": 35.9, "IMPALA_100M": 31.0}
    res["V_dmlab_human_norm_not_recomputable"] = "사람 기준 점수가 논문에 없어 Human mean capped는 재계산 불가"


def window_sensitivity(res):
    """논문이 적지 않은 최종 점수 집계 창(마지막 k개 점 평균)을 k=1,3,5로 바꿔 표 값과 비교한다."""
    out = {}
    tab = parse_rows(r"^\s*ProcGen scores", r"^\s*Table 7:", 4)
    mapt = {norm(k): v for k, v in tab.items()}
    for k in (1, 3, 5):
        dr, _ = final_by_run("procgen-dreamerv3", 5e7, k=k)
        tm = task_means(dr)
        d = [abs(tm[t] - mapt[norm(t.replace("procgen_", ""))][3]) for t in tm]
        out[f"procgen_k{k}_max_abs_diff_vs_table7_dreamer"] = round(float(max(d)), 3)
    tab = parse_rows(r"^\s*DMLab scores", r"^\s*Human mean capped", 6)
    mapt = {norm(k): v for k, v in tab.items()}
    for k in (1, 3, 5):
        tm = task_means(final_by_run("dmlab30-dreamerv3", 1e8, k=k)[0])
        d = [abs(tm[t] - mapt[norm(t.replace("dmlab_", ""))][5]) for t in tm]
        out[f"dmlab_k{k}_median_abs_diff_vs_table8_dreamer"] = round(float(np.median(d)), 3)
    rows, _ = parse_atari_table()
    key = {norm(g): g for g in rows}
    for k in (1, 3, 5):
        tm = task_means(final_by_run("atari100k-dreamerv3", 4e5, k=k)[0])
        h = np.array([hns(tm[t], rows[key[norm(t.replace("atari_", ""))]]["Random"], rows[key[norm(t.replace("atari_", ""))]]["Gamer"]) for t in tm])
        out[f"atari100k_k{k}_mean_median_hns"] = [round(float(h.mean()), 1), round(float(np.median(h)), 1)]
    for k in (1, 3, 5):
        tm = task_means(final_by_run("minecraft_diamond-dreamerv3", 1e8, k=k)[0])
        out[f"minecraft_dreamer_k{k}_mean"] = round(float(np.mean(list(tm.values()))), 2)
    for k in (1, 3, 5):
        tm = task_means(final_by_run("dmc_proprio-dreamerv3", 5e5, k=k)[0])
        v = np.array(list(tm.values()))
        out[f"dmc_proprio_dreamer_k{k}_mean_median"] = [round(float(v.mean()), 1), round(float(np.median(v)), 1)]
    res["window_sensitivity"] = out


def baselines():
    """repo/baselines.yaml의 정규화 기준값 (atari57_gamer, dmlab30, procgen_hard). 형식이 단순해 정규식으로 읽는다."""
    out, sec = {}, None
    for l in (SCORES.parent / "baselines.yaml").read_text(encoding="utf-8").splitlines():
        if re.match(r"^\S", l):
            sec = l.rstrip(":").strip()
            out[sec] = {}
            continue
        m = re.match(r"^\s+(\S+): \[([-\d.]+), ([-\d.]+)\]", l)
        if m and sec:
            out[sec][m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


def reference_checks(res):
    """기준값과 정규화 정의를 검사한다: (1) 표 6의 Random/Gamer가 baselines.yaml과 같은가,
    (2) 표 7의 Normalized mean이 (점수-랜덤)/(최대-랜덤)으로 나오는가, (3) DMLab Human mean capped를 표 8에서 재계산."""
    bl = baselines()
    rows, _ = parse_atari_table()
    diffs = []
    for g, v in rows.items():
        key = "atari_" + re.sub(r"\s+", "_", g.lower()).replace("'", "")
        if key in bl["atari57_gamer"]:
            r, h = bl["atari57_gamer"][key]
            diffs.append(max(abs(r - v["Random"]), abs(h - v["Gamer"])))
    res["V_atari_reference_vs_baselines_yaml"] = {"n": len(diffs), "max_abs_diff": round(float(max(diffs)), 2) if diffs else None}
    tab = parse_rows(r"^\s*ProcGen scores", r"^\s*Table 7:", 4)
    tot = {c: [] for c in range(4)}
    for t, v in tab.items():
        if t.lower() == "normalized mean":
            continue
        key = "procgen_" + norm(t)
        r, mx = bl["procgen_hard"][key]
        for c in range(4):
            tot[c].append(100 * (v[c] - r) / (mx - r))
    res["V_procgen_normalized_mean_check"] = {"definition": "mean_over_16_games of 100*(score-random)/(max-random), baselines.yaml procgen_hard",
                                              "recomputed": [round(float(np.mean(tot[c])), 1) for c in range(4)],
                                              "table_order": ["Original PPO", "PPO", "PPG", "Dreamer"],
                                              "table_reported": [41.16, 42.80, 64.89, 66.01]}
    tab = parse_rows(r"^\s*DMLab scores", r"^\s*Human mean capped", 6)
    cols = ["R2D2+ 10B", "IMPALA 10B", "IMPALA 1B", "IMPALA 100M", "PPO 100M", "Dreamer 100M"]
    capped = {c: [] for c in range(6)}
    for t, v in tab.items():
        key = "dmlab" + norm(t)
        r, h = next((bl["dmlab30"][k] for k in bl["dmlab30"] if norm(k) == key), (None, None))
        if r is None:
            continue
        for c in range(6):
            capped[c].append(min(100.0, 100 * (v[c] - r) / (h - r)))
    rel, _ = final_by_run("dmlab30-dreamerv3", 1e8, k=5)
    rel_capped = []
    for t, sc in task_means(rel).items():
        key = "dmlab" + norm(t.replace("dmlab_", ""))
        r, h = next((bl["dmlab30"][k] for k in bl["dmlab30"] if norm(k) == key), (None, None))
        rel_capped.append(min(100.0, 100 * (sc - r) / (h - r)))
    res["V_dmlab_human_capped_released_dreamer_k5"] = round(float(np.mean(rel_capped)), 1)
    # seed별 사람 기준 평균 (과제 평균을 seed마다 따로 계산한 뒤 seed 사이 표준편차)
    per_seed = {}
    for r in load("dmlab30-dreamerv3"):
        xs = np.array([np.nan if v is None else v for v in r["xs"]], float)
        ys = np.array([np.nan if v is None else v for v in r["ys"]], float)
        ok = (~np.isnan(ys)) & (xs <= 1e8 * 1.0001)
        key = "dmlab" + norm(r["task"].replace("dmlab_", ""))
        rr, hh = next((bl["dmlab30"][k] for k in bl["dmlab30"] if norm(k) == key), (None, None))
        val = float(ys[np.flatnonzero(ok)[-5:]].mean())
        per_seed.setdefault(r["seed"], []).append(min(100.0, 100 * (val - rr) / (hh - rr)))
    seed_means = [np.mean(v) for v in per_seed.values() if len(v) == 30]
    res["V_dmlab_capped_across_seeds"] = {"n_seeds": len(seed_means), "mean": round(float(np.mean(seed_means)), 1),
                                          "sd": round(float(np.std(seed_means, ddof=1)), 1),
                                          "values": [round(float(v), 1) for v in seed_means]}
    res["V_dmlab_human_capped_check"] = {"n_tasks": len(capped[0]),
                                         "recomputed": {cols[c]: round(float(np.mean(capped[c])), 1) for c in range(6)},
                                         "table_reported": {"R2D2+ 10B": 85.4, "IMPALA 10B": 85.1, "IMPALA 1B": 66.3, "IMPALA 100M": 31.0, "PPO 100M": 35.9, "Dreamer 100M": 71.4}}


def main():
    res = {}
    atari_block(res)
    atari100k_block(res)
    minecraft_block(res)
    dmc_block(res)
    procgen_block(res)
    dmlab_block(res)
    window_sensitivity(res)
    reference_checks(res)
    # 본문 수치 간 산술 검사
    res["V_replay_ratio_check"] = {"batch_steps": 16 * 64, "replay_ratio": 32, "action_repeat": 4,
                                   "env_steps_per_grad_step": 16 * 64 / 32 * 4,
                                   "grad_steps_over_200M": round(200e6 / (16 * 64 / 32 * 4) / 1e6, 2), "text_claim_M": 1.5}
    # 36000 스텝 = 30분 이므로 20 스텝/초. 프로젝트 페이지의 "3천만 스텝 = 17일"과 비교한다
    res["V_minecraft_days"] = {"steps_per_second_from_36000_steps_30min": 20,
                               "days_for_100M_steps_at_20_steps_per_s": round(100e6 / 20 / 86400, 1),
                               "steps_per_day_from_project_page_30M_in_17d": round(30e6 / 17, 0),
                               "days_for_100M_from_project_page_rate": round(100e6 / (30e6 / 17), 1),
                               "text_claim_days_at_100M": 100}
    res["V_atari_text_tasks"] = {"atari57": 57, "procgen": 16, "dmlab": 30, "minecraft": 1, "bsuite": 23,
                                 "proprio": 18, "visual": 20, "atari100k": 26}
    res["V_tasks_total_unique_protocols"] = sum(res["V_atari_text_tasks"].values())
    res["V_tasks_excluding_atari100k"] = sum(v for k, v in res["V_atari_text_tasks"].items() if k != "atari100k")

    plot = {k: res.pop(k) for k in list(res) if k.startswith("_")}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    PLOT_IN.parent.mkdir(exist_ok=True)
    PLOT_IN.write_text(json.dumps(plot), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
