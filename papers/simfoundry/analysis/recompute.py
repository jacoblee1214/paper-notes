"""SimFoundry (arXiv:2606.28276 v4) 보고 수치 재계산과 검정.

입력은 논문 부록 표(G.1–G.8, 표 2, L.2, L.4, L.6)에서 옮긴 값입니다. 실행하거나 설치할 것이 없고
numpy, scipy만 씁니다(수 초). 결과는 recompute_results.json에 저장합니다.

실행: python3 analysis/recompute.py
"""
import json
import math
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}

# ---------------------------------------------------------------- 입력: 표 G.1, G.2, G.3
POLICIES = ["pi0", "pi05", "n16", "n17", "dz"]
# 과제: {정책: (real, sim)}. 평가하지 않은 칸은 없음
G1 = {
    "Stack Dishware": {"pi0": (100, 34), "pi05": (100, 64), "n16": (40, 0)},
    "Store Marker": {"pi0": (48, 4), "pi05": (60, 20), "n16": (32, 0)},
    "Throw Away Trash": {"pi0": (20, 0), "pi05": (48, 4), "n16": (0, 0)},
    "Serve Fruits": {"pi0": (0, 4), "pi05": (72, 80), "n16": (4, 20), "n17": (40, 32), "dz": (8, 12)},
    "Cup in Bowl": {"pi0": (88, 56), "pi05": (100, 92), "n16": (68, 40), "n17": (92, 92), "dz": (100, 92)},
    "Marker in Cup": {"pi0": (40, 40), "pi05": (92, 88), "n16": (28, 28), "n17": (88, 88), "dz": (88, 80)},
    "Clear Table": {"pi0": (0, 12), "pi05": (40, 36), "n16": (0, 0), "n17": (8, 28), "dz": (16, 28)},
}
G1_REPORTED = {  # (Pearson r, MMRV)
    "Stack Dishware": (0.883, 0.000), "Store Marker": (0.915, 0.000), "Throw Away Trash": (0.910, 0.067),
    "Serve Fruits": (0.960, 0.016), "Cup in Bowl": (0.907, 0.016), "Marker in Cup": (0.995, 0.008),
    "Clear Table": (0.810, 0.016),
}
G2 = {
    "Stack Dishware": {"pi0": (100, 0), "pi05": (100, 8), "n16": (40, 0)},
    "Store Marker": {"pi0": (48, 0), "pi05": (60, 4), "n16": (32, 0)},
    "Throw Away Trash": {"pi0": (20, 0), "pi05": (48, 0), "n16": (0, 0)},
    "Serve Fruits": {"pi0": (0, 4), "pi05": (72, 28), "n16": (4, 24), "n17": (40, 4), "dz": (8, 4)},
    "Cup in Bowl": {"pi0": (88, 20), "pi05": (100, 36), "n16": (68, 76), "n17": (92, 48), "dz": (100, 68)},
    "Marker in Cup": {"pi0": (40, 0), "pi05": (92, 4), "n16": (28, 4), "n17": (88, 12), "dz": (88, 4)},
    "Clear Table": {"pi0": (0, 0), "pi05": (40, 0), "n16": (0, 4), "n17": (8, 0), "dz": (16, 12)},
}
G2_REPORTED = {
    "Stack Dishware": (0.500, 0.200), "Store Marker": (0.822, 0.053), "Throw Away Trash": (None, 0.253),
    "Serve Fruits": (0.480, 0.288), "Cup in Bowl": (-0.396, 0.280), "Marker in Cup": (0.512, 0.176),
    "Clear Table": (-0.037, 0.352),
}
G3 = {  # 하위 과제 평가 (정책 3개): real, sim
    "Stack Dishware": {"pi0": (100, 64), "pi05": (100, 80), "n16": (40, 24)},
    "Store Marker": {"pi0": (48, 52), "pi05": (60, 76), "n16": (32, 36)},
    "Throw Away Trash": {"pi0": (20, 0), "pi05": (48, 8), "n16": (0, 0)},
}
G3_REPORTED = {"Stack Dishware": (0.961, 0.000), "Store Marker": (0.981, 0.000), "Throw Away Trash": (0.910, 0.067)}


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def mmrv(x, y):
    """식 2: x = 실제 점수, y = 시뮬 점수. 시뮬에서 동률이면 순서 불일치로 센다(PolaRiS Throw Away Trash 0.253으로 확인)."""
    x, y = np.asarray(x, float) / 100, np.asarray(y, float) / 100
    n = len(x)
    tot = 0.0
    for i in range(n):
        m = 0.0
        for j in range(n):
            if (y[i] < y[j]) != (x[i] < x[j]):
                m = max(m, abs(x[i] - x[j]))
        tot += m
    return tot / n


def xy(table_task):
    pol = list(table_task)
    return [table_task[p][0] for p in pol], [table_task[p][1] for p in pol]


def task_metrics(tbl, reported, label):
    rows, rs, ms, rr, rm = [], [], [], [], []
    for t, d in tbl.items():
        x, y = xy(d)
        r, m = pearson(x, y), mmrv(x, y)
        rep_r, rep_m = reported[t]
        rows.append({"task": t, "n_policies": len(x), "r": None if math.isnan(r) else round(r, 3),
                     "r_reported": rep_r, "mmrv": round(m, 3), "mmrv_reported": rep_m,
                     "r_match": (rep_r is None and math.isnan(r)) or (rep_r is not None and abs(r - rep_r) <= 0.0015),
                     "mmrv_match": abs(m - rep_m) <= 0.0015})
        if not math.isnan(r):
            rs.append(r)
            rr.append(rep_r)
        ms.append(m)
        rm.append(rep_m)
    return {"label": label, "rows": rows, "mean_r": round(float(np.mean(rs)), 4), "mean_r_reported": round(float(np.mean(rr)), 4),
            "n_r_defined": len(rs), "mean_mmrv": round(float(np.mean(ms)), 4), "mean_mmrv_reported": round(float(np.mean(rm)), 4)}


RES["G1_simfoundry"] = task_metrics(G1, G1_REPORTED, "표 G.1 SimFoundry")
RES["G2_polaris"] = task_metrics(G2, G2_REPORTED, "표 G.2 PolaRiS")
RES["G3_subtask"] = task_metrics(G3, G3_REPORTED, "표 G.3 하위 과제 평가")
# 표 G.2의 Throw Away Trash는 시뮬이 전부 0이라 r이 정의되지 않음. 평균은 6개 과제만으로 계산한 값과 0으로 넣은 값을 모두 낸다
r_pol = [t["r"] for t in RES["G2_polaris"]["rows"] if t["r"] is not None]
RES["pearson_gap"] = {
    "simfoundry_mean_r_7tasks": RES["G1_simfoundry"]["mean_r"],
    "polaris_mean_r_6tasks_excluding_undefined": round(float(np.mean(r_pol)), 4),
    "gap_excluding_undefined": round(RES["G1_simfoundry"]["mean_r"] - float(np.mean(r_pol)), 4),
    "polaris_mean_r_if_undefined_is_0": round(float(np.sum(r_pol)) / 7, 4),
    "gap_if_undefined_is_0": round(RES["G1_simfoundry"]["mean_r"] - float(np.sum(r_pol)) / 7, 4),
    "paper_text": "over 0.59",
}
# 과제별 짝지은 비교: SimFoundry r > PolaRiS r 인 과제 수, MMRV 낮은 과제 수 (부호 검정, 양측)
wins_r = sum(1 for a, b in zip(RES["G1_simfoundry"]["rows"], RES["G2_polaris"]["rows"]) if b["r"] is not None and a["r"] > b["r"])
wins_m = sum(1 for a, b in zip(RES["G1_simfoundry"]["rows"], RES["G2_polaris"]["rows"]) if a["mmrv"] < b["mmrv"])
RES["paired_sign_tests"] = {
    "r_wins_of_6": wins_r, "r_sign_p_two_sided": float(stats.binomtest(wins_r, 6, 0.5).pvalue),
    "mmrv_wins_of_7": wins_m, "mmrv_sign_p_two_sided": float(stats.binomtest(wins_m, 7, 0.5).pvalue),
}

# ---------------------------------------------------------------- 정책-과제 쌍 수와 시도 수 (25회, 표 J.1)
pairs = [(t, p, rs) for t, d in G1.items() for p, rs in d.items()]
RES["pair_counts"] = {"policy_task_pairs": len(pairs), "claimed_grid_7x5": 35,
                      "real_rollouts_if_25_each": 25 * len(pairs), "sim_rollouts_if_25_each": 25 * len(pairs)}
RES["all_values_multiple_of_4"] = all(v % 4 == 0 for _, _, rs in pairs for v in rs) and \
    all(v % 4 == 0 for d in G2.values() for rs in d.values() for v in rs)

# ---------------------------------------------------------------- 셀 단위 실제 대 시뮬 Fisher (n = 25)
N = 25
sig, cells = 0, []
for t, p, (r_, s_) in pairs:
    a, b = round(r_ * N / 100), round(s_ * N / 100)
    pv = float(stats.fisher_exact([[a, N - a], [b, N - b]])[1])
    cells.append({"task": t, "policy": p, "real": r_, "sim": s_, "fisher_p": round(pv, 4)})
    sig += pv < 0.05
RES["cell_fisher"] = {"n_pairs": len(pairs), "n_significant_p05": int(sig), "cells": cells}
ft_tasks = ["Stack Dishware", "Store Marker", "Throw Away Trash"]
bias = lambda tasks: float(np.mean([s_ - r_ for t, p, (r_, s_) in pairs if t in tasks]))
zs_tasks = [t for t in G1 if t not in ft_tasks]
RES["sim_minus_real_mean_pp"] = {"finetuned_tasks_9_pairs": round(bias(ft_tasks), 2), "zero_shot_tasks_20_pairs": round(bias(zs_tasks), 2)}
sig_ft = sum(c["fisher_p"] < 0.05 for c in cells if c["task"] in ft_tasks)
sig_zs = sum(c["fisher_p"] < 0.05 for c in cells if c["task"] in zs_tasks)
RES["cell_fisher"]["significant_finetuned_of_9"] = int(sig_ft)
RES["cell_fisher"]["significant_zero_shot_of_20"] = int(sig_zs)
mae = float(np.mean([abs(s_ - r_) for _, _, (r_, s_) in pairs]))
RES["mean_abs_gap_pp_all29"] = round(mae, 2)

# 풀링 상관 (29쌍). 과제 난이도가 상관을 키운다는 점을 보기 위해 과제 평균을 뺀 상관도 계산한다
X = np.array([r_ for _, _, (r_, s_) in pairs], float)
Y = np.array([s_ for _, _, (r_, s_) in pairs], float)
tasks_arr = np.array([t for t, _, _ in pairs])
Xc, Yc = X.copy(), Y.copy()
for t in G1:
    m = tasks_arr == t
    Xc[m] -= X[m].mean()
    Yc[m] -= Y[m].mean()
RES["pooled"] = {"pearson_29_pairs": round(pearson(X, Y), 4), "spearman_29_pairs": round(float(stats.spearmanr(X, Y)[0]), 4),
                 "pearson_within_task_demeaned": round(pearson(Xc, Yc), 4)}
Xp, Yp = [], []
for t, d in G2.items():
    for p, (r_, s_) in d.items():
        Xp.append(r_)
        Yp.append(s_)
RES["pooled"]["polaris_pearson_29_pairs"] = round(pearson(Xp, Yp), 4)

# ---------------------------------------------------------------- 정책 수가 3–5개인 과제별 r의 불확실성
def exact_perm_p(x, y):
    """시뮬 점수를 정책 사이에서 섞은 정확 순열 검정(단측: r이 관측 이상일 확률). 동률이 많으면 보수적이지 않을 수 있다."""
    r0 = pearson(x, y)
    if math.isnan(r0):
        return float("nan")
    cnt = tot = 0
    for perm in permutations(range(len(y))):
        yy = [y[i] for i in perm]
        r = pearson(x, yy)
        tot += 1
        if not math.isnan(r) and r >= r0 - 1e-12:
            cnt += 1
    return cnt / tot


perm = []
for t, d in G1.items():
    x, y = xy(d)
    perm.append({"task": t, "n": len(x), "r": round(pearson(x, y), 3), "perm_p_one_sided": round(exact_perm_p(x, y), 4)})
RES["perm_tests_G1"] = perm
# 임계 r: 정책 n개에서 양측 p<0.05가 되는 |r| (정규 가정 피어슨 검정)
crit = {}
for n in (3, 4, 5):
    tcrit = stats.t.ppf(0.975, n - 2)
    crit[n] = round(float(tcrit / math.sqrt(n - 2 + tcrit ** 2)), 4)
RES["critical_abs_r_two_sided_p05"] = crit
# 과제별 r의 정규 가정 p값 (표 값 기준)
RES["pearson_p_values_G1"] = [
    {"task": t, "n": len(d), "r": round(pearson(*xy(d)), 3),
     "p_two_sided": round(float(stats.pearsonr(*xy(d))[1]), 4)} for t, d in G1.items()
]

# ---------------------------------------------------------------- 토이: 25회 시행 잡음 천장 (numpy, 수 초)
rng = np.random.default_rng(0)


def noise_ceiling(tbl, n_trials=25, reps=4000, shrink=0.5):
    """진짜 성공률이 실제=시뮬로 같다고 가정하고 양쪽을 독립적으로 25회씩 뽑았을 때 평균 r과 평균 MMRV의 분포."""
    res_r, res_m = [], []
    for _ in range(reps):
        rs, ms = [], []
        for t, d in tbl.items():
            p_true = np.array([(v[0] / 100 * n_trials + shrink) / (n_trials + 2 * shrink) for v in d.values()])
            a = rng.binomial(n_trials, p_true) * 100 / n_trials
            b = rng.binomial(n_trials, p_true) * 100 / n_trials
            r = pearson(a, b)
            if not math.isnan(r):
                rs.append(r)
            ms.append(mmrv(a, b))
        res_r.append(np.mean(rs))
        res_m.append(np.mean(ms))
    return np.array(res_r), np.array(res_m)


nr, nm = noise_ceiling(G1)
RES["toy_noise_ceiling_25_trials"] = {
    "assumption": "진짜 성공률 = 표의 실제 성공률(0.5회 가산 평활), 실제와 시뮬 모두 독립 25회 시행. 시뮬이 완벽히 같은 환경이라는 가정",
    "mean_of_mean_r": round(float(nr.mean()), 4), "r_5th_pct": round(float(np.percentile(nr, 5)), 4),
    "r_95th_pct": round(float(np.percentile(nr, 95)), 4),
    "mean_of_mean_mmrv": round(float(nm.mean()), 4), "mmrv_5th_pct": round(float(np.percentile(nm, 5)), 4),
    "mmrv_95th_pct": round(float(np.percentile(nm, 95)), 4),
    "observed_mean_r": RES["G1_simfoundry"]["mean_r"], "observed_mean_mmrv": RES["G1_simfoundry"]["mean_mmrv"],
    "frac_draws_with_mean_r_ge_observed": round(float((nr >= RES["G1_simfoundry"]["mean_r"]).mean()), 4),
}
# 귀무: 시뮬 점수를 정책 사이에서 무작위로 섞었을 때 평균 r
null_r = []
for _ in range(4000):
    rs = []
    for t, d in G1.items():
        x, y = xy(d)
        yy = rng.permutation(y)
        r = pearson(x, yy)
        if not math.isnan(r):
            rs.append(r)
    null_r.append(np.mean(rs))
null_r = np.array(null_r)
_bins = np.linspace(-0.4, 1.0, 29)
RES["toy_hist"] = {"bins": [round(float(b), 3) for b in _bins],
                   "noise_ceiling_counts": np.histogram(nr, bins=_bins)[0].tolist(),
                   "null_counts": np.histogram(null_r, bins=_bins)[0].tolist()}
RES["toy_permutation_null_mean_r"] = {"mean": round(float(null_r.mean()), 4), "p95": round(float(np.percentile(null_r, 95)), 4),
                                      "p99": round(float(np.percentile(null_r, 99)), 4),
                                      "frac_ge_observed": round(float((null_r >= RES["G1_simfoundry"]["mean_r"]).mean()), 5)}

# ---------------------------------------------------------------- 표 G.4, G.8, G.5, G.6, G.7 : 사전 학습 다양성
G4 = {  # (과제, 평가 칸): (twin only, +9 cousins). 앞이 YAM, 뒤가 DROID
    ("YAM Stack Dishware", "Sim Twin"): (83, 92), ("YAM Stack Dishware", "Sim Cousins"): (43, 66),
    ("YAM Stack Dishware", "Real Twin"): (39, 43), ("YAM Stack Dishware", "Real Cousins"): (21, 42),
    ("YAM Pot on Stove", "Sim Twin"): (85, 100), ("YAM Pot on Stove", "Sim Cousins"): (17, 93),
    ("YAM Pot on Stove", "Real Twin"): (91, 99), ("YAM Pot on Stove", "Real Cousins"): (14, 64),
    ("YAM Throw Away Trash", "Sim Twin"): (97, 97), ("YAM Throw Away Trash", "Sim Cousins"): (97, 94),
    ("YAM Throw Away Trash", "Real Twin"): (0, 28), ("YAM Throw Away Trash", "Real Cousins"): (2, 8),
    ("DROID Stack Dishware", "Sim Twin"): (80, 88), ("DROID Stack Dishware", "Sim Cousins"): (64, 92),
    ("DROID Stack Dishware", "Real Twin"): (88, 96), ("DROID Stack Dishware", "Real Cousins"): (88, 100),
    ("DROID Store Marker", "Sim Twin"): (20, 60), ("DROID Store Marker", "Sim Cousins"): (8, 28),
    ("DROID Store Marker", "Real Twin"): (4, 20), ("DROID Store Marker", "Real Cousins"): (4, 4),
    ("DROID Throw Away Trash", "Sim Twin"): (8, 48), ("DROID Throw Away Trash", "Sim Cousins"): (4, 48),
    ("DROID Throw Away Trash", "Real Twin"): (0, 20), ("DROID Throw Away Trash", "Real Cousins"): (0, 8),
}
G8_YAM = {  # twin, +1, +3, +9
    ("Stack Dishware", "Sim Twin"): (83, 89, 100, 92), ("Stack Dishware", "Sim Cousins"): (43, 44, 65, 66),
    ("Stack Dishware", "Real Twin"): (39, 41, 37, 43), ("Stack Dishware", "Real Cousins"): (21, 32, 27, 42),
    ("Pot on Stove", "Sim Twin"): (85, 100, 93, 100), ("Pot on Stove", "Sim Cousins"): (17, 27, 35, 93),
    ("Pot on Stove", "Real Twin"): (91, 100, 94, 99), ("Pot on Stove", "Real Cousins"): (14, 38, 16, 64),
    ("Throw Away Trash", "Sim Twin"): (97, 98, 98, 97), ("Throw Away Trash", "Sim Cousins"): (97, 100, 97, 94),
    ("Throw Away Trash", "Real Twin"): (0, 9, 45, 28), ("Throw Away Trash", "Real Cousins"): (2, 17, 14, 8),
}
# G.4(YAM)와 G.8의 twin / +9 열이 같은지
g48_same = all(G4[("YAM " + k[0], k[1])] == (v[0], v[3]) for k, v in G8_YAM.items())
RES["G4_vs_G8_yam_consistent"] = bool(g48_same)

delta = lambda keys: [G4[k][1] - G4[k][0] for k in keys]
real_keys = [k for k in G4 if k[1].startswith("Real")]
sim_keys = [k for k in G4 if k[1].startswith("Sim")]
fig5a_tasks = ["YAM Pot on Stove", "YAM Throw Away Trash", "DROID Store Marker", "DROID Throw Away Trash"]
real_fig5a = [k for k in real_keys if k[0] in fig5a_tasks]
RES["object_cousin_avg_gain_pp"] = {
    "real_all_6task_12cells": round(float(np.mean(delta(real_keys))), 2),
    "real_fig5A_4task_8cells": round(float(np.mean(delta(real_fig5a))), 2),
    "sim_all_12cells": round(float(np.mean(delta(sim_keys))), 2),
    "real_and_sim_24cells": round(float(np.mean(delta(real_keys + sim_keys))), 2),
    "real_twin_only_6": round(float(np.mean(delta([k for k in real_keys if k[1] == "Real Twin"]))), 2),
    "real_cousin_only_6": round(float(np.mean(delta([k for k in real_keys if k[1] == "Real Cousins"]))), 2),
    "paper_claim": 17,
}
RES["zero_shot_real_twin_9cousins"] = {k[0]: G4[k][1] for k in real_keys if k[1] == "Real Twin"}
RES["zero_shot_real_twin_9cousins_mean"] = round(float(np.mean([G4[k][1] for k in real_keys if k[1] == "Real Twin"])), 2)
RES["sim_to_real_gap_9cousins_twin_pp"] = {t: G4[(t, "Sim Twin")][1] - G4[(t, "Real Twin")][1] for t in
                                           sorted({k[0] for k in G4})}

# DROID 셀(모두 4의 배수, 25회) Fisher
def fisher_pct(a_pct, b_pct, n=25):
    a, b = round(a_pct * n / 100), round(b_pct * n / 100)
    return float(stats.fisher_exact([[a, n - a], [b, n - b]])[1])


RES["object_cousin_fisher_droid_n25"] = {
    f"{k[0]} | {k[1]}": {"before": G4[k][0], "after": G4[k][1], "p": round(fisher_pct(*G4[k]), 4)}
    for k in G4 if k[0].startswith("DROID")
}

G5 = {"Stack Dishware": (80, 88), "Stack Dishware-cousin": (28, 64), "Store Marker": (20, 24), "Store Marker-cousin": (0, 16),
      "Throw Away Trash": (8, 36), "Throw Away Trash-cousin": (0, 36)}
G6 = {"Stack Dishware": (80, 88, 100, 100), "Store Marker": (20, 36, 48, 60), "Throw Away Trash": (8, 44, 44, 68)}
RES["scene_cousin"] = {
    "twin_scene_avg_gain": round(float(np.mean([G5[k][1] - G5[k][0] for k in G5 if "cousin" not in k])), 2),
    "cousin_scene_avg_gain": round(float(np.mean([G5[k][1] - G5[k][0] for k in G5 if "cousin" in k])), 2),
    "all_6_avg_gain": round(float(np.mean([v[1] - v[0] for v in G5.values()])), 2), "paper_claim": 21,
    "fisher_n25": {k: {"before": v[0], "after": v[1], "p": round(fisher_pct(*v), 4)} for k, v in G5.items()},
    "evaluation": "시뮬레이션만 (DROID, 표 G.5, 그림 5B)",
}
RES["task_cousin"] = {
    "gain_13_tasks_avg": round(float(np.mean([v[3] - v[0] for v in G6.values()])), 2), "paper_claim": 40,
    "gain_each": {k: v[3] - v[0] for k, v in G6.items()},
    "fisher_n25_twin_vs_13": {k: round(fisher_pct(v[0], v[3]), 5) for k, v in G6.items()},
    "monotone_in_task_count": {k: bool(all(v[i] <= v[i + 1] for i in range(3))) for k, v in G6.items()},
    "evaluation": "시뮬레이션만 (DROID, 표 G.6, 그림 5C). 실제 로봇 평가 없음",
}

G7 = {  # (pi0-S, pi0-R, pi0-co, pi05-S, pi05-R, pi05-co)
    "Stack Dishware Sim": (92, 34, 76, 88, 64, 100), "Stack Dishware Real": (96, 100, 100, 96, 100, 100),
    "Store Marker Sim": (16, 4, 40, 60, 20, 60), "Store Marker Real": (4, 48, 80, 20, 60, 92),
    "Throw Away Trash Sim": (0, 0, 36, 48, 4, 60), "Throw Away Trash Real": (0, 20, 76, 20, 48, 96),
}
cot = []
for k, v in G7.items():
    cot.append({"cell": k, "pi0_delta": v[2] - max(v[0], v[1]), "pi05_delta": v[5] - max(v[3], v[4])})
RES["G7_cotrain_deltas"] = cot
# 그림 G.2D에 찍힌 값
fig_g2d = {"real": {"pi0": [0, 32, 56], "pi05": [0, 32, 48]}, "sim": {"pi0": [-16, 24, 36], "pi05": [12, 0, 16]}}
comp = {}
for side in ("real", "sim"):
    for pol, idx in (("pi0", "pi0_delta"), ("pi05", "pi05_delta")):
        calc = [c[idx] for c in cot if c["cell"].endswith(side.capitalize())]
        comp[f"{side}_{pol}"] = {"table_G7": calc, "figure_G2D": fig_g2d[side][pol], "match": calc == fig_g2d[side][pol]}
RES["G7_vs_figG2D"] = comp
# G.7의 sim-only 열이 G.4 DROID(+9 cousins)와 같은지, real-only가 G.1 실제 값과 같은지
RES["G7_vs_G4_G1"] = {
    "pi05_S_equals_G4_9cousins_DROID": all(
        G7[f"{t} {s}"][3] == G4[(f"DROID {t}", f"{s} Twin")][1]
        for t in ("Stack Dishware", "Store Marker", "Throw Away Trash") for s in ("Sim", "Real")),
    "R_columns_equal_G1": all(
        G7[f"{t} Real"][1] == G1[t]["pi0"][0] and G7[f"{t} Real"][4] == G1[t]["pi05"][0] and
        G7[f"{t} Sim"][1] == G1[t]["pi0"][1] and G7[f"{t} Sim"][4] == G1[t]["pi05"][1]
        for t in ("Stack Dishware", "Store Marker", "Throw Away Trash")),
}
RES["G7_fisher_n25"] = {
    "pi05 Store Marker real: real-only 60 -> co-train 92": round(fisher_pct(60, 92), 4),
    "pi0 Throw Away Trash sim: 0 -> 36": round(fisher_pct(0, 36), 5),
    "pi05 Stack Dishware real: sim-only 96 vs real-only 100": round(fisher_pct(96, 100), 4),
}

# ---------------------------------------------------------------- 정수 가능성: 퍼센트가 n회 시행에서 나올 수 있는가
def feasible(pct, n):
    return any(round(100 * k / n) == pct for k in range(n + 1))


def feasible_ns(values, lo=1, hi=200):
    return [n for n in range(lo, hi + 1) if all(feasible(v, n) for v in values)]


yam_vals = [v for k, vs in G8_YAM.items() for v in vs]
RES["integer_feasibility"] = {
    "yam_G8_values_not_multiple_of_4": sorted({v for v in yam_vals if v % 4 != 0}),
    "yam_G8_n_range_1_to_100_consistent_with_all": feasible_ns(yam_vals, 1, 100),
    "yam_G8_feasible_with_25_trials": all(feasible(v, 25) for v in yam_vals),
    "yam_G8_values_infeasible_at_25": sorted({v for v in yam_vals if not feasible(v, 25)}),
    "yam_G8_feasible_with_50_trials": all(feasible(v, 50) for v in yam_vals),
    "yam_G8_feasible_with_100_trials": all(feasible(v, 100) for v in yam_vals),
}
T2 = {"seen_real": (28, 45, 46), "heldout_real": (26, 29, 26), "seen_sim": (30, 51, 61), "heldout_sim": (37, 45, 33)}
t2 = {}
for k, v in T2.items():
    ns = feasible_ns(v, 1, 300)
    mult = 13 if "seen" in k else 7
    t2[k] = {"values": v, "min_total_trials": ns[0] if ns else None, "n_le_300_consistent_count": len(ns),
             "totals_that_are_multiples_of_task_count": [n for n in ns if n % mult == 0][:8], "task_count": mult}
RES["table2_trial_count_feasibility"] = t2
RES["table2_deltas"] = {"sim_gain_DROIDFT_vs_base": 61 - 30, "real_gain_DROIDFT_vs_base": 46 - 28,
                        "heldout_real_gain_FT_vs_base": 29 - 26, "heldout_real_gain_DROIDFT_vs_base": 26 - 26,
                        "heldout_sim_gain_FT_vs_base": 45 - 37, "heldout_sim_gain_DROIDFT_vs_base": 33 - 37}


# 표 2: 시도 수가 보고되지 않아, 모든 값과 과제 수(13, 7)로 나누어떨어지는 가장 작은 총 시도 수를 가정하고 Fisher 검정을 한다 (가정임)
ASSUMED_N = {"seen_real": 65, "heldout_real": 35, "seen_sim": 104, "heldout_sim": 49}
T2_COLS = ("base", "FT", "DROID-FT")  # pi0.5-DROID, pi0.5-FT, pi0.5-DROID-FT
t2f = {}
for k, n in ASSUMED_N.items():
    ks = [round(v * n / 100) for v in T2[k]]
    assert all(round(100 * kk / n) == v for kk, v in zip(ks, T2[k])), k
    def fp(i, j):
        return round(float(stats.fisher_exact([[ks[i], n - ks[i]], [ks[j], n - ks[j]]])[1]), 4)
    t2f[k] = {"assumed_n": n, "successes_base_FT_DROIDFT": ks, "fisher_p_FT_vs_base": fp(0, 1),
              "fisher_p_DROIDFT_vs_base": fp(0, 2), "fisher_p_DROIDFT_vs_FT": fp(1, 2)}
RES["table2_fisher_under_assumed_n"] = t2f

# ---------------------------------------------------------------- 표 L.2, L.4, L.6
L2 = {  # 난이도: {지표: (SAM3D, SF zero-shot, SF tuned)} 각각 (평균, SD)
    "Easy": {"chamfer": ((0.0081, 0.0024), (0.0042, 0.0013), (0.0026, 0.00026)),
             "f1": ((0.71, 0.15), (0.92, 0.071), (0.99, 0.0069)),
             "pos": ((0.016, 0.0058), (0.0060, 0.0019), (0.0041, 0.00037))},
    "Medium": {"chamfer": ((0.0087, 0.0028), (0.0047, 0.0012), (0.0033, 0.00068)),
               "f1": ((0.66, 0.18), (0.87, 0.089), (0.97, 0.026)),
               "pos": ((0.018, 0.0067), (0.0076, 0.0038), (0.0057, 0.0030))},
    "Hard": {"chamfer": ((0.0088, 0.0022), (0.0091, 0.0076), (0.0039, 0.0013)),
             "f1": ((0.68, 0.14), (0.81, 0.071), (0.93, 0.049)),
             "pos": ((0.022, 0.010), (0.018, 0.018), (0.0073, 0.0022))},
}
l2 = {}
for d, mets in L2.items():
    for m, (a, b, c) in mets.items():
        lower_better = m != "f1"
        better = (b[0] < a[0]) if lower_better else (b[0] > a[0])
        # SD가 장면 간 SD이고 장면이 4개라는 가정 (논문은 SD의 단위를 밝히지 않음)
        t4 = stats.ttest_ind_from_stats(a[0], a[1], 4, b[0], b[1], 4, equal_var=False)
        l2[f"{d}/{m}"] = {"sf_zero_shot_better_than_sam3d": bool(better),
                          "welch_p_if_n4_scenes": round(float(t4.pvalue), 4),
                          "relative_change_pct": round(100 * (b[0] - a[0]) / a[0], 1),
                          "tuned_vs_zero_shot_pct": round(100 * (c[0] - b[0]) / b[0], 1)}
RES["L2_reconstruction"] = l2
RES["L2_claims"] = {
    "f1_zero_shot_range": [0.81, 0.92], "f1_tuned_range": [0.93, 0.99], "f1_sam3d_range": [0.66, 0.71],
    "hard_chamfer_sf_worse_than_sam3d": L2["Hard"]["chamfer"][1][0] > L2["Hard"]["chamfer"][0][0],
}

L4 = [(2, 19.98), (4, 34.07), (9, 41.36), (10, 42.76), (10, 49.02), (10, 51.01), (10, 51.39), (10, 58.71), (11, 67.13)]
n_, t_ = np.array([a for a, _ in L4], float), np.array([b for _, b in L4])
slope, intercept, r_, p_, se_ = stats.linregress(n_, t_)
RES["L4_time"] = {"scenes": len(L4), "objects_total": int(n_.sum()), "minutes_total": round(float(t_.sum()), 2),
                  "minutes_per_object_pooled": round(float(t_.sum() / n_.sum()), 2),
                  "per_scene_min_per_object": [round(float(b / a), 2) for a, b in L4],
                  "linear_fit_slope_min_per_object": round(float(slope), 2), "intercept_min": round(float(intercept), 2),
                  "r2": round(float(r_ ** 2), 3),
                  "scenes_over_60_min": [b for a, b in L4 if b > 60]}
L6 = {"manual": dict(psnr=12.91, ssim=0.497, mae=0.1712, rmse=0.2294, ncc=0.549, dE=20.13, edge=0.0283),
      "auto": dict(psnr=15.29, ssim=0.605, mae=0.1275, rmse=0.1758, ncc=0.749, dE=15.23, edge=0.0248)}
RES["L6_background"] = {
    "auto_better_on_all_7": all([L6["auto"]["psnr"] > L6["manual"]["psnr"], L6["auto"]["ssim"] > L6["manual"]["ssim"],
                                 L6["auto"]["mae"] < L6["manual"]["mae"], L6["auto"]["rmse"] < L6["manual"]["rmse"],
                                 L6["auto"]["ncc"] > L6["manual"]["ncc"], L6["auto"]["dE"] < L6["manual"]["dE"],
                                 L6["auto"]["edge"] < L6["manual"]["edge"]]),
    "psnr_from_mean_rmse_manual": round(-20 * math.log10(L6["manual"]["rmse"]), 2),
    "psnr_from_mean_rmse_auto": round(-20 * math.log10(L6["auto"]["rmse"]), 2),
    "jensen_consistent_manual": L6["manual"]["psnr"] >= -20 * math.log10(L6["manual"]["rmse"]),
    "jensen_consistent_auto": L6["auto"]["psnr"] >= -20 * math.log10(L6["auto"]["rmse"]),
    "rmse_ge_mae": L6["manual"]["rmse"] >= L6["manual"]["mae"] and L6["auto"]["rmse"] >= L6["auto"]["mae"],
}

# ---------------------------------------------------------------- 요약 수치 (초록, 본문 대조)
RES["abstract_checks"] = {
    "mean_pearson_0.911": RES["G1_simfoundry"]["mean_r"], "mean_mmrv_0.018": RES["G1_simfoundry"]["mean_mmrv"],
    "table1_G1_reported_means": [RES["G1_simfoundry"]["mean_r_reported"], RES["G1_simfoundry"]["mean_mmrv_reported"]],
    "subtask_mean_r_G1_3tasks": round(float(np.mean([pearson(*xy(G1[t])) for t in ft_tasks])), 4),
    "subtask_mean_r_G3_3tasks": RES["G3_subtask"]["mean_r"],
    "claim_text": "0.90 -> 0.95 (본문), 0.902 -> 0.951 (부록 G.1.1)",
}
# 본문 문장 점검: "GR00T N1.7이 Marker in Cup에서 다른 정책보다 낫다"
mic = G1["Marker in Cup"]
RES["marker_in_cup_claim"] = {"real": {p: v[0] for p, v in mic.items()}, "sim": {p: v[1] for p, v in mic.items()},
                              "n17_is_top_in_real": mic["n17"][0] >= max(v[0] for v in mic.values()),
                              "fisher_p_pi05_92_vs_n17_88": round(fisher_pct(92, 88), 4)}

# ---------------------------------------------------------------- 부트스트랩: 과제별 r의 정책 재표본 대신, 셀 시행 부트스트랩
def boot_mean_r(tbl, reps=3000):
    out = []
    for _ in range(reps):
        rs = []
        for t, d in tbl.items():
            a = np.array([rng.binomial(25, v[0] / 100) for v in d.values()]) * 4.0
            b = np.array([rng.binomial(25, v[1] / 100) for v in d.values()]) * 4.0
            r = pearson(a, b)
            if not math.isnan(r):
                rs.append(r)
        out.append(np.mean(rs))
    return np.array(out)


bs = boot_mean_r(G1)
RES["bootstrap_mean_r_G1_cell_trials"] = {"mean": round(float(bs.mean()), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4),
                                          round(float(np.percentile(bs, 97.5)), 4)]}
bp = boot_mean_r(G2)
RES["bootstrap_mean_r_G2_cell_trials"] = {"mean": round(float(bp.mean()), 4), "ci95": [round(float(np.percentile(bp, 2.5)), 4),
                                          round(float(np.percentile(bp, 97.5)), 4)]}

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
               encoding="utf-8")
print("wrote", OUT)
