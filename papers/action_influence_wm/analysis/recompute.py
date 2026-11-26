"""논문 표 1·그림 3·그림 5 수치 재계산과 검사 (V1–V10). 결과: analysis/recompute_results.json

입력
  paper/paper.txt                       : 원문 v1 텍스트 (pdftotext -layout). 표 1을 여기서 읽는다.
  repo/docs/static/data/*.json          : 공식 저장소(프로젝트 사이트 소스)의 기록 데이터.
      catalog.json, experiments/*.json (모델 16개 x 평가 분포 3개, 시드 1, H=33 창별 EE·Tee 재현율),
      fourrooms_influence_fidelity.json (그림 3), composite_pusht_controllability.json (그림 5)
  repo/ 는 다음과 같이 받은 얕은 복제다 (복제 주소와 커밋):
      git clone --depth 1 https://github.com/princeton-prism/action-influence-maximization repo
      git -C repo checkout 11e0da0a0b43ddb4c02f0140d4804cdc65c69101

실행: python3 analysis/recompute.py   (numpy, scipy; 수 초~30초)
공식 코드는 실행하지 않는다. 사이트의 JSON 데이터만 읽는다.
"""
import json
import re
from math import atanh, tanh, sqrt, log
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "repo" / "docs" / "static" / "data"
OUT = ROOT / "analysis" / "recompute_results.json"
rng = np.random.default_rng(0)
res = {}

# ---------------------------------------------------------------- 표 1 읽기
DISTS = ["BC", "BC-Failure", "Push-Away"]
ROWS = ["PPO-300K", "PPO-3K", "PPO-3K+AIM500", "PPO-3K+Recovery500", "PPO-3K+ICM500", "PPO-3K+DIAYN500",
        "Recovery-3K", "Recovery-3K+AIM500", "AIM w/o re-estimation", "AIM w/o MI reward", "AIM w/o Novelty reward"]
txt = (ROOT / "paper" / "paper.txt").read_text(encoding="utf-8")
table1 = {}
for line in txt.splitlines():
    s = line.strip()
    for name in ROWS:
        if s.startswith(name + " ") and name not in table1:
            toks = re.findall(r"\d*\.\d+±\.\d+|\d+±\d+", s)
            if len(toks) == 12:
                vals = []
                for t in toks:
                    m, h = t.split("±")
                    vals.append((float(m), float(h)))
                table1[name] = {
                    d: {"ee": vals[4 * i], "tee": vals[4 * i + 1], "joint": vals[4 * i + 2], "elo": vals[4 * i + 3]}
                    for i, d in enumerate(DISTS)
                }
assert set(table1) == set(ROWS), set(ROWS) - set(table1)
res["table1_parsed_rows"] = len(table1)

# ---------------------------------------------------------------- V1: 본문 퍼센트 주장 검사
T = lambda r, d, k: table1[r][d][k][0]
claims = {
    "AIM500 BC joint vs PPO-3K (+69%)": (T("PPO-3K+AIM500", "BC", "joint") / T("PPO-3K", "BC", "joint") - 1) * 100,
    "AIM500 BC joint vs PPO-300K (+54%)": (T("PPO-3K+AIM500", "BC", "joint") / T("PPO-300K", "BC", "joint") - 1) * 100,
    "Recovery-3K+AIM500 BC joint vs Recovery-3K (+36%)": (T("Recovery-3K+AIM500", "BC", "joint") / T("Recovery-3K", "BC", "joint") - 1) * 100,
    "w/o re-estimation BC joint vs AIM500 (+10%)": (T("AIM w/o re-estimation", "BC", "joint") / T("PPO-3K+AIM500", "BC", "joint") - 1) * 100,
    "w/o MI(novelty only) Push-Away joint vs AIM500 (+16%)": (T("AIM w/o MI reward", "Push-Away", "joint") / T("PPO-3K+AIM500", "Push-Away", "joint") - 1) * 100,
    "w/o Novelty(MI only) BC Elo vs AIM500 (+16%)": (T("AIM w/o Novelty reward", "BC", "elo") / T("PPO-3K+AIM500", "BC", "elo") - 1) * 100,
    "w/o Novelty(MI only) BC-Failure Elo vs AIM500 (+8%)": (T("AIM w/o Novelty reward", "BC-Failure", "elo") / T("PPO-3K+AIM500", "BC-Failure", "elo") - 1) * 100,
}
res["V1_text_percent_claims"] = {k: round(v, 2) for k, v in claims.items()}
res["V1_85x"] = {"300000/3500": 300000 / 3500, "300000/3505": 300000 / 3505, "300000/500": 300000 / 500}
res["V1_model_counts"] = {"PushT 8 scales x4 + FourRooms 9 scales x4": 8 * 4 + 9 * 4,
                          "4.2 collection models: FourRooms 5 methods x 4 rounds + PushT 8 methods x 5 rounds": 5 * 4 + 8 * 5,
                          "paper says total": 73}
res["V1_arena"] = {"comparisons": 1545, "annotators": 20, "per_annotator": 1545 / 20,
                   "per_distribution_if_even": 1545 / 3}

# H=33 EE: 3K vs 300K (확장은 데이터 100배)
a, b = table1["PPO-300K"]["BC"]["ee"], table1["PPO-3K"]["BC"]["ee"]
dz = (a[0] - b[0]) / sqrt((a[1] / 1.959964) ** 2 + (b[1] / 1.959964) ** 2)
res["V2_H33_expert_scaling_EE_BC"] = {"PPO-300K": a, "PPO-3K": b, "diff": round(a[0] - b[0], 3), "z_eval_noise_only": round(dz, 2),
                                      "p_two_sided": float(2 * stats.norm.sf(abs(dz)))}
for d in DISTS:
    a, b = table1["PPO-300K"][d]["joint"], table1["PPO-3K+AIM500"][d]["joint"]
    c, e = table1["PPO-300K"][d], table1["PPO-3K+AIM500"][d]
    res.setdefault("V2_AIM500_vs_PPO300K_by_dist", {})[d] = {
        k: {"PPO-300K": c[k][0], "AIM500": e[k][0], "ratio": round(e[k][0] / c[k][0], 3)} for k in ("ee", "tee", "joint")}

# ---------------------------------------------------------------- 사이트 데이터: 표 1 재계산 (V3)
cat = json.loads((DATA / "catalog.json").read_text())
ds_ids = [d["id"] for d in cat["datasets"]]
cases = {d["id"]: json.loads((DATA / "experiments" / f"d0{i + 1}.json").read_text())["cases"] for i, d in enumerate(cat["datasets"])}
traj_ids = {d: sorted({c["trajectory_id"] for c in cases[d]}) for d in ds_ids}
res["V3_site_structure"] = {
    "datasets": [{"id": d["id"], "label": d["label"], "windows": d["cases"], "trajectories": len(traj_ids[d["id"]])} for d in cat["datasets"]],
    "models": len(cat["models"]), "seed": cat["seed"], "frames_per_window": 33,
}
site = {}   # site[model_label][dist] = dict(metric -> (per-traj sums, per-traj counts))
for m in cat["models"]:
    site[m["label"]] = {}
    for di, d in enumerate(ds_ids):
        pred = json.loads((ROOT / "repo" / "docs" / m["data"][d].lstrip("./")).read_text())["predictions"]
        tid = {c["id"]: c["trajectory_id"] for c in cases[d]}
        tix = {t: i for i, t in enumerate(traj_ids[d])}
        sums = {k: np.zeros(len(tix)) for k in ("ee", "tee", "joint")}
        cnts = {k: np.zeros(len(tix)) for k in ("ee", "tee", "joint")}
        for cid, p in pred.items():
            i = tix[tid[cid]]
            e, t = p["metrics"]["ee"], p["metrics"]["tee"]
            for x in e:
                if x is not None:
                    sums["ee"][i] += x; cnts["ee"][i] += 1
            for x in t:
                if x is not None:
                    sums["tee"][i] += x; cnts["tee"][i] += 1
            for x, y in zip(e, t):
                if x is not None and y is not None:
                    sums["joint"][i] += min(x, y); cnts["joint"][i] += 1
        site[m["label"]][DISTS[di]] = {k: (sums[k], cnts[k]) for k in sums}

LABEL = {"PPO-300K": "PPO100", "PPO-3K": "PPO1", "PPO-3K+AIM500": "PPO1 + AIM500", "PPO-3K+Recovery500": "PPO1 + Recovery500",
         "PPO-3K+ICM500": "PPO1 + ICM500", "PPO-3K+DIAYN500": "PPO1 + DIAYN500", "Recovery-3K": "Recovery1",
         "Recovery-3K+AIM500": "Recovery1 + transferred AIM500", "AIM w/o re-estimation": "PPO1 + fixed-policy AIM500",
         "AIM w/o MI reward": "PPO1 + AIM500 (novelty only)", "AIM w/o Novelty reward": "PPO1 + AIM500 (MI only)"}


def mean_of(sc, idx=None):
    s, c = sc
    if idx is None:
        return s.sum() / c.sum()
    return s[idx].sum() / c[idx].sum()


def boot_ci(sc, B=2000):
    n = len(sc[0])
    idx = rng.integers(0, n, size=(B, n))
    s, c = sc
    m = s[idx].sum(1) / c[idx].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


cmp_rows = []
diffs = []
hw_ratio = []
for r in ROWS:
    for d in DISTS:
        for k in ("ee", "tee", "joint"):
            mean_site = mean_of(site[LABEL[r]][d][k])
            lo, hi = boot_ci(site[LABEL[r]][d][k], B=1000)
            p_mean, p_hw = table1[r][d][k]
            diffs.append(mean_site - p_mean)
            hw_ratio.append(((hi - lo) / 2) / p_hw)
            cmp_rows.append({"row": r, "dist": d, "metric": k, "paper": p_mean, "paper_hw": p_hw,
                             "site_seed1": round(mean_site, 4), "site_ci": [round(lo, 4), round(hi, 4)],
                             "diff": round(mean_site - p_mean, 4)})
diffs = np.array(diffs)
res["V3_table1_vs_site"] = {
    "n_cells": len(cmp_rows),
    "mean_abs_diff": round(float(np.abs(diffs).mean()), 4),
    "max_abs_diff": round(float(np.abs(diffs).max()), 4),
    "mean_diff": round(float(diffs.mean()), 4),
    "within_0.01": int((np.abs(diffs) <= 0.0105).sum()),
    "within_paper_halfwidth": int(sum(abs(r["diff"]) <= r["paper_hw"] for r in cmp_rows)),
    "site_ci_halfwidth_over_paper_halfwidth_median": round(float(np.median(hw_ratio)), 2),
    "site_ci_halfwidth_over_paper_halfwidth_min_max": [round(float(min(hw_ratio)), 2), round(float(max(hw_ratio)), 2)],
    "rows": cmp_rows,
}
res["V3_extra_models_not_in_table1"] = {
    lab: {d: {k: round(mean_of(site[lab][d][k]), 3) for k in ("ee", "tee", "joint")} for d in DISTS}
    for lab in ["PPO1 + Uniform Random500", "PPO1 + Random Walk500", "PPO1 + DADS500", "PPO1 + METRA500", "Recovery100"]
}

# ---------------------------------------------------------------- V4: 쌍대 궤적 부트스트랩 (시드 1, 평가 잡음만)
def paired(a_lab, b_lab, d, k, B=4000):
    sa, ca = site[a_lab][d][k]
    sb, cb = site[b_lab][d][k]
    n = len(sa)
    idx = rng.integers(0, n, size=(B, n))
    diff = sa[idx].sum(1) / ca[idx].sum(1) - sb[idx].sum(1) / cb[idx].sum(1)
    obs = sa.sum() / ca.sum() - sb.sum() / cb.sum()
    return {"diff": round(float(obs), 4), "ci": [round(float(np.percentile(diff, 2.5)), 4), round(float(np.percentile(diff, 97.5)), 4)],
            "boot_p_two_sided": float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))}


pairs = [("PPO1 + AIM500", "PPO1"), ("PPO1 + AIM500", "PPO100"), ("PPO1 + AIM500", "Recovery1"),
         ("PPO1 + AIM500", "PPO1 + AIM500 (novelty only)"), ("PPO1 + AIM500", "PPO1 + AIM500 (MI only)"),
         ("PPO1 + fixed-policy AIM500", "PPO1 + AIM500"), ("PPO100", "PPO1"), ("Recovery1 + transferred AIM500", "Recovery1"),
         ("PPO1 + AIM500", "PPO1 + Uniform Random500"), ("Recovery100", "PPO1 + AIM500")]
res["V4_paired_bootstrap_seed1"] = {}
for a_lab, b_lab in pairs:
    key = f"{a_lab}  -  {b_lab}"
    res["V4_paired_bootstrap_seed1"][key] = {d: {k: paired(a_lab, b_lab, d, k) for k in ("ee", "tee", "joint")} for d in DISTS}

# ---------------------------------------------------------------- V5: 그림 3 재계산
f3 = json.loads((DATA / "fourrooms_influence_fidelity.json").read_text())["panels"][0]["series"]
X = np.concatenate([s["x"] for s in f3]); Y = np.concatenate([s["y"] for s in f3])
r = float(np.corrcoef(X, Y)[0, 1]); n = len(X)
se = 1 / sqrt(n - 3)
res["V5_fig3"] = {
    "points": n, "series": [s["key"] for s in f3], "pooled_r": round(r, 4),
    "fisher_ci_95": [round(tanh(atanh(r) - 1.959964 * se), 4), round(tanh(atanh(r) + 1.959964 * se), 4)],
    "paper_r": 0.979, "paper_ci": [0.96, 0.99],
    "within_series_r": {s["key"]: round(float(np.corrcoef(s["x"], s["y"])[0, 1]), 4) for s in f3},
    "x_range": [round(float(X.min()), 3), round(float(X.max()), 3)],
    "uniform_random_x_range": [round(min(f3[3]["x"]), 4), round(max(f3[3]["x"]), 4)],
    "uniform_random_y_range": [round(min(f3[3]["y"]), 2), round(max(f3[3]["y"]), 2)],
}
# 시리즈를 하나씩 뺀 r, 시리즈 간 평균만 쓴 r (군집 효과)
loo = {}
for i, s in enumerate(f3):
    xs = np.concatenate([t["x"] for j, t in enumerate(f3) if j != i]); ys = np.concatenate([t["y"] for j, t in enumerate(f3) if j != i])
    loo[s["key"]] = round(float(np.corrcoef(xs, ys)[0, 1]), 4)
res["V5_fig3"]["leave_one_series_out_r"] = loo
# 시리즈 단위 부트스트랩 (4 시리즈 복원추출) 구간
vals = []
for _ in range(5000):
    pick = rng.integers(0, 4, 4)
    xs = np.concatenate([f3[i]["x"] for i in pick]); ys = np.concatenate([f3[i]["y"] for i in pick])
    if xs.std() > 0 and ys.std() > 0:
        vals.append(np.corrcoef(xs, ys)[0, 1])
res["V5_fig3"]["series_cluster_bootstrap_r_2.5_50_97.5"] = [round(float(np.percentile(vals, q)), 3) for q in (2.5, 50, 97.5)]
# 스케일 내 순위: 각 시리즈에서 x와 y가 모두 단조 증가하는지
res["V5_fig3"]["monotone_pairs"] = {s["key"]: bool(np.all(np.diff(s["x"]) <= 1e-9) and np.all(np.diff(s["y"]) <= 1e-9)) for s in f3}

# ---------------------------------------------------------------- V6: 그림 5 (사이트 JSON)
f5 = json.loads((DATA / "composite_pusht_controllability.json").read_text())["panels"]
out5 = {}
for p in f5:
    refs = {r_["label"]: round(r_["y"], 2) for r_ in p["refs"]}
    ser = {s["key"]: {"y0": round(s["y"][0], 2), "y_last": round(s["y"][-1], 2), "min": round(min(s["y"]), 2), "max": round(max(s["y"]), 2),
                      "change": round(s["y"][-1] - s["y"][0], 2)} for s in p["series"]}
    out5[p["title"]] = {"refs": refs, "series": ser}
res["V6_fig5_site_json"] = out5
bc = out5["Push-T BC"]
res["V6_claims"] = {
    "BC: AIM500 vs 300k expert (paper: 'matches')": [bc["series"]["AIM (active)"]["y_last"], bc["refs"]["300k expert"]],
    "PushAway: AIM500 vs 300k expert": [out5["Push-T Push Away"]["series"]["AIM (active)"]["y_last"], out5["Push-T Push Away"]["refs"]["300k expert"]],
    "BC AIM gain over 3k expert (pp)": round(bc["series"]["AIM (active)"]["y_last"] - bc["refs"]["3k expert"], 2),
    "baselines with y_last <= 3k expert (BC)": [k for k, v in bc["series"].items() if k != "AIM (active)" and v["y_last"] <= bc["refs"]["3k expert"]],
    "baselines total (BC)": len(bc["series"]) - 1,
}
# 그림 6 판독 값 (그림 판독, +-0.7p). 원문 19쪽. 표본 크기 격자 ≈ 1.5k, 4.7k, 9.4k, 19k, 38k, 75k, 150k, 300k
res["V7_fig6_read_values_BC_EE"] = {
    "note": "그림 판독 값(논문 19쪽 그림 6, 확대 판독, 오차 약 ±0.7p). 값은 논문 본문에 없음",
    "x_trajectories": [1500, 4700, 9400, 18800, 37500, 75000, 150000, 300000],
    "PPO expert": [20.2, 20.5, 22.3, 27.0, 28.9, 31.0, 32.3, 32.3],
    "Random Walk": [26.7, 37.9, 41.8, 42.1, 43.1, 42.6, 43.1, 43.2],
    "Uniform Random": [20.9, 39.1, 41.7, 42.2, 43.4, 43.1, 43.3, 43.6],
    "Recovery": [19.5, 30.9, 37.2, 40.1, 40.5, 42.5, 42.0, 42.5],
}
res["V7_fig6_read_values_PushAway_EE"] = {
    "note": "그림 판독 값 (그림 6 오른쪽 패널)",
    "x_trajectories": [1500, 4700, 9400, 18800, 37500, 75000, 150000, 300000],
    "PPO expert": [18.0, 18.8, 21.4, 27.7, 36.7, 39.1, 41.1, 41.6],
    "Random Walk": [35.3, 43.7, 45.4, 47.3, 47.8, 47.2, 48.1, 47.6],
    "Uniform Random": [18.0, 38.4, 39.0, 41.2, 40.2, 41.0, 41.1, 40.4],
    "Recovery": [18.0, 38.0, 45.6, 49.0, 50.0, 48.8, 49.8, 50.0],
}
# 읽은 곡선에서 AIM(32.5)을 처음 넘는 데이터 크기
fx = res["V7_fig6_read_values_BC_EE"]
aim_bc = bc["series"]["AIM (active)"]["y_last"]
first = {}
for k in ("PPO expert", "Random Walk", "Uniform Random", "Recovery"):
    ys = fx[k]
    first[k] = next((fx["x_trajectories"][i] for i, v in enumerate(ys) if v >= aim_bc - 0.3), None)
res["V7_first_size_reaching_AIM_BC"] = {"AIM value": aim_bc, "first read size": first,
                                        "ratio to 3500 (AIM total)": {k: (round(v / 3500, 1) if v else None) for k, v in first.items()}}

# ---------------------------------------------------------------- V8: Elo와 재현율, 구간
spearman = {}
for d in DISTS:
    elo = [T(r, d, "elo") for r in ROWS]
    for k in ("ee", "joint"):
        rec = [T(r, d, k) for r in ROWS]
        rho, p = stats.spearmanr(elo, rec)
        spearman[f"{d} Elo vs {k}"] = {"rho": round(float(rho), 3), "p": round(float(p), 4)}
res["V8_spearman_elo_recall"] = spearman


def z_pair(r1, r2, d):
    a, b = table1[r1][d]["elo"], table1[r2][d]["elo"]
    se_ = sqrt((a[1] / 1.959964) ** 2 + (b[1] / 1.959964) ** 2)
    z = (a[0] - b[0]) / se_
    return {"diff": a[0] - b[0], "z": round(z, 2), "p": round(float(2 * stats.norm.sf(abs(z))), 4)}


res["V8_elo_pairs"] = {
    "AIM500 vs PPO-3K, BC": z_pair("PPO-3K+AIM500", "PPO-3K", "BC"),
    "AIM500 vs PPO-300K, BC": z_pair("PPO-3K+AIM500", "PPO-300K", "BC"),
    "AIM500 vs Recovery-3K, BC": z_pair("PPO-3K+AIM500", "Recovery-3K", "BC"),
    "AIM500 vs PPO-3K, BC-Failure": z_pair("PPO-3K+AIM500", "PPO-3K", "BC-Failure"),
    "AIM500 vs Recovery-3K, BC-Failure": z_pair("PPO-3K+AIM500", "Recovery-3K", "BC-Failure"),
    "AIM500 vs PPO-3K, Push-Away": z_pair("PPO-3K+AIM500", "PPO-3K", "Push-Away"),
    "AIM500 vs Recovery-3K, Push-Away": z_pair("PPO-3K+AIM500", "Recovery-3K", "Push-Away"),
    "w/o Novelty(MI only) vs AIM500, BC": z_pair("AIM w/o Novelty reward", "PPO-3K+AIM500", "BC"),
    "w/o Novelty(MI only) vs AIM500, BC-Failure": z_pair("AIM w/o Novelty reward", "PPO-3K+AIM500", "BC-Failure"),
    "w/o re-estimation vs AIM500, BC": z_pair("AIM w/o re-estimation", "PPO-3K+AIM500", "BC"),
}
# 모든 쌍 중 CI 비중첩 비율
nonov = 0; tot = 0
for d in DISTS:
    for i in range(len(ROWS)):
        for j in range(i + 1, len(ROWS)):
            a, b = table1[ROWS[i]][d]["elo"], table1[ROWS[j]][d]["elo"]
            tot += 1
            nonov += abs(a[0] - b[0]) > a[1] + b[1]
res["V8_elo_ci_nonoverlap_pairs"] = {"non_overlapping": nonov, "pairs": tot}

# Elo 구간 시뮬레이션: 비교 수 가정 아래 Bradley-Terry 최대우도 적합의 부트스트랩 반폭
def fit_bt(n_models, gi, gj, wins_i, iters=60):
    th = np.zeros(n_models)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(th[gi] - th[gj])))
        g = np.zeros(n_models); H = np.zeros((n_models, n_models))
        np.add.at(g, gi, wins_i - p); np.add.at(g, gj, -(wins_i - p))
        w = p * (1 - p)
        for a_, b_, w_ in zip(gi, gj, w):
            H[a_, a_] += w_; H[b_, b_] += w_; H[a_, b_] -= w_; H[b_, a_] -= w_
        H += np.eye(n_models) * 1e-3
        th = th + np.linalg.solve(H, g)
        th -= th.mean()
    return th


def sim_bt(true_elo, n_games, reps=150):
    m = len(true_elo); k = np.log(10) / 400
    th_true = (np.array(true_elo) - np.mean(true_elo)) * k
    fits = []
    for _ in range(reps):
        gi = rng.integers(0, m, n_games); gj = rng.integers(0, m, n_games)
        keep = gi != gj
        gi, gj = gi[keep], gj[keep]
        p = 1 / (1 + np.exp(-(th_true[gi] - th_true[gj])))
        wins = (rng.random(len(p)) < p).astype(float)
        fits.append(fit_bt(m, gi, gj, wins) / k)
    fits = np.array(fits)
    return 1.959964 * fits.std(0)


bc_elo = [T(r, "BC", "elo") for r in ROWS]
res["V8_elo_ci_simulation"] = {
    "note": "가정: 비교 쌍이 모델 사이에서 무작위, 무승부 없음, 참 점수는 표 1 BC Elo. 반폭은 적합 점수의 2.5-97.5% 구간 반폭 근사(1.96 SD)",
    "reported_halfwidths_BC": [table1[r]["BC"]["elo"][1] for r in ROWS],
    "sim_11_models_515_games": [round(float(x), 0) for x in sim_bt(bc_elo, 515)],
    "sim_11_models_1545_games": [round(float(x), 0) for x in sim_bt(bc_elo, 1545)],
}
res["V8_elo_ci_simulation"]["median_sim_515"] = float(np.median(res["V8_elo_ci_simulation"]["sim_11_models_515_games"]))
res["V8_elo_ci_simulation"]["median_sim_1545"] = float(np.median(res["V8_elo_ci_simulation"]["sim_11_models_1545_games"]))
res["V8_elo_ci_simulation"]["median_reported_BC"] = float(np.median(res["V8_elo_ci_simulation"]["reported_halfwidths_BC"]))

# ---------------------------------------------------------------- V9: 사이트 Table 1 Elo 전사 대조 (preference-figure.mjs)
mjs = (ROOT / "repo" / "docs" / "preference-figure.mjs").read_text()
site_elo = {}
for m in re.finditer(r"\{model: '([^']+)', scores: \[\[(\d+), (\d+)\], \[(\d+), (\d+)\], \[(\d+), (\d+)\]\]\}", mjs):
    site_elo[m.group(1)] = [(int(m.group(2 * i + 2)), int(m.group(2 * i + 3))) for i in range(3)]
mism = 0
for r in ROWS:
    got = site_elo[LABEL[r]]
    for i, d in enumerate(DISTS):
        want = (int(table1[r][d]["elo"][0]), int(table1[r][d]["elo"][1]))
        mism += tuple(got[i]) != want
res["V9_site_vs_paper_elo_cells"] = {"cells": len(ROWS) * 3, "mismatch": mism}

# ---------------------------------------------------------------- V10: 파라미터 수 (표 3, 표 5)
def layer(d, ratio, cross=False):
    attn = 4 * d * d
    ffn = 3 * d * int(ratio * d)
    return attn + ffn + (4 * d * d if cross else 0)

res["V10_param_estimates"] = {
    "note": "표준 트랜스포머 층 계산 (어텐션 4d^2 + SwiGLU 3*d*(ratio*d)), 임베딩·정규화·MLP 헤드 제외. 표에 없는 모듈이 있으면 차이가 날 수 있음",
    "DF_WM_d512_12layers_ratio3": {"estimate_M": round(12 * layer(512, 3) / 1e6, 1), "paper_M": 44.5},
    "LatentDiffusion_WM_d512_12enc+12dec_ratio3": {"estimate_M": round((12 * layer(512, 3) + 12 * layer(512, 3, True)) / 1e6, 1), "paper_M": 97.8},
    "SoftmaxBC_d256_6layers_ratio4": {"estimate_M": round(6 * layer(256, 4) / 1e6, 1), "paper_M": 19.5},
    "DiffusionBC_d256_12layers_ratio3": {"estimate_M": round(12 * layer(256, 3) / 1e6, 1), "paper_M": 30.6},
}

OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT)
print(json.dumps({k: res[k] for k in ("V1_text_percent_claims", "V2_H33_expert_scaling_EE_BC", "V5_fig3", "V6_claims",
                                       "V7_first_size_reaching_AIM_BC", "V8_elo_pairs", "V8_elo_ci_nonoverlap_pairs",
                                       "V8_elo_ci_simulation", "V9_site_vs_paper_elo_cells", "V10_param_estimates")},
                 indent=1, ensure_ascii=False)[:6000])
print(json.dumps({k: v for k, v in res["V3_table1_vs_site"].items() if k != "rows"}, indent=1))
