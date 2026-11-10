"""WARP 논문 표 재계산과 검정. 실행: python3 analysis/table_check.py → analysis/table_check_results.json

입력은 논문 값뿐입니다(원시 trial 기록 없음, 시드 수 없음). 다음을 계산합니다.
  1) Table 1 (DexMimicGen): 평균 재계산, 과제별 차이, Fisher 정확검정, Holm 보정.
     표본 크기 n은 논문에 없으므로 값의 격자로 추정합니다: 재생(replay) 값은 0.5% 단위 → n=200 demos,
     정책(policy) 값은 정수 % 단위 → 2%p 격자이면 n=50 (100 단위면 n=100). 두 경우를 모두 보고합니다.
  2) Fig. 7 (실물): 막대 값을 그림에서 읽은 근사치(±1%p)입니다. 10회 시행, 과제 점수가 0.5 단위라
     시행당 비율은 0.05 단위입니다. 근사로 20개 반-시행 표본으로 바꿔 Fisher를 계산합니다.
     intro의 "35%"가 과제 세 개 rollout 차이의 평균(%p)인지 확인합니다.
  3) Table 3/4 (시뮬 재계산): 주장한 "최고" 항목이 실제로 최고인지 항목별로 검사하고, 150배·99.34%·
     30배 속도·처리 시간 "1시간 대 하루"를 표 값으로 검산합니다.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "table_check_results.json"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, run)
    return adj.tolist()


# ---------- 1) Table 1 (DexMimicGen, %) ----------
TASKS = ["can_sort", "pouring", "coffee"]
T1 = {
    "MINK": {"replay": [99.5, 88.5, 50.5], "policy": [94, 74, 8], "avg_replay": 79.5, "avg_policy": 59},
    "WARP": {"replay": [98.5, 90.5, 51.0], "policy": [100, 78, 34], "avg_replay": 80.0, "avg_policy": 71},
}
res1 = {"avg_check": {}, "policy_diff_pts": {}, "fisher": {}}
for meth, d in T1.items():
    res1["avg_check"][meth] = {
        "replay_mean_of_tasks": round(float(np.mean(d["replay"])), 2), "replay_reported": d["avg_replay"],
        "policy_mean_of_tasks": round(float(np.mean(d["policy"])), 2), "policy_reported": d["avg_policy"],
    }
pol_diff = [w - m for w, m in zip(T1["WARP"]["policy"], T1["MINK"]["policy"])]
res1["policy_diff_pts"] = dict(zip(TASKS, pol_diff))
res1["policy_diff_pts"]["mean"] = round(float(np.mean(pol_diff)), 2)
res1["policy_relative_gain_avg_pct"] = round(100 * (71 / 59 - 1), 1)

pvals = []
rows = []
for i, task in enumerate(TASKS):
    for n_pol in (50, 100):
        k_w = round(T1["WARP"]["policy"][i] * n_pol / 100)
        k_m = round(T1["MINK"]["policy"][i] * n_pol / 100)
        p = stats.fisher_exact([[k_w, n_pol - k_w], [k_m, n_pol - k_m]])[1]
        rows.append({"task": task, "n": n_pol, "WARP": f"{k_w}/{n_pol}", "MINK": f"{k_m}/{n_pol}",
                     "fisher_p": round(float(p), 4), "wilson_WARP": [round(x, 3) for x in wilson(k_w, n_pol)],
                     "wilson_MINK": [round(x, 3) for x in wilson(k_m, n_pol)]})
        if n_pol == 50:
            pvals.append(p)
res1["fisher"]["policy_per_task"] = rows
res1["fisher"]["policy_holm_n50"] = holm(pvals)
# 재생(replay): n=200 가정
rep_p = []
rep_rows = []
for i, task in enumerate(TASKS):
    kw = round(T1["WARP"]["replay"][i] * 2)
    km = round(T1["MINK"]["replay"][i] * 2)
    p = stats.fisher_exact([[kw, 200 - kw], [km, 200 - km]])[1]
    rep_p.append(p)
    rep_rows.append({"task": task, "WARP": f"{kw}/200", "MINK": f"{km}/200", "fisher_p": round(float(p), 4)})
res1["fisher"]["replay_per_task_n200"] = rep_rows
res1["fisher"]["replay_holm_n200"] = holm(rep_p)

# 과제 3개를 합친 정책 성공 (n=50씩 가정): 150 대 150
k_w = sum(round(x / 100 * 50) for x in T1["WARP"]["policy"])
k_m = sum(round(x / 100 * 50) for x in T1["MINK"]["policy"])
res1["pooled_policy_n50"] = {"WARP": f"{k_w}/150", "MINK": f"{k_m}/150",
                              "fisher_p": round(float(stats.fisher_exact([[k_w, 150 - k_w], [k_m, 150 - k_m]])[1]), 6)}

# ---------- 2) Fig. 7 (실물, 10회; 그림에서 읽은 근사치) ----------
FIG7 = {
    "Rotate box": {"replay": (100, 95), "rollout": (65, 40)},   # (WARP, MINK)
    "Push cart": {"replay": (95, 25), "rollout": (75, 25)},
    "Laundry": {"replay": (70, 0), "rollout": (37.5, 7.5)},
}
res2 = {"reading_note": "그림 막대 라벨에서 읽음, ±1%p. MINK 변형(EF/TE)은 실물 절에 명시되지 않음"}
roll_diffs = []
rows2 = []
p2 = []
for task, d in FIG7.items():
    dw = d["rollout"][0] - d["rollout"][1]
    roll_diffs.append(dw)
    kw = round(d["rollout"][0] / 100 * 20)
    km = round(d["rollout"][1] / 100 * 20)
    p = stats.fisher_exact([[kw, 20 - kw], [km, 20 - km]])[1]
    p2.append(p)
    rows2.append({"task": task, "rollout_WARP_pct": d["rollout"][0], "rollout_MINK_pct": d["rollout"][1],
                  "diff_pts": round(dw, 2), "fisher_p_(20 half-trials)": round(float(p), 3),
                  "wilson_95_WARP_10trials": [round(100 * x, 1) for x in wilson(d["rollout"][0] / 10, 10)],
                  "wilson_95_MINK_10trials": [round(100 * x, 1) for x in wilson(d["rollout"][1] / 10, 10)]})
res2["rows"] = rows2
res2["rollout_mean_diff_pts_3tasks"] = round(float(np.mean(roll_diffs)), 2)
res2["holm_p"] = [round(x, 3) for x in holm(p2)]
res2["rotate_box_rollout_text_65_vs_40"] = "논문 본문 값과 일치"

# ---------- 3) Table 3 / Table 4 (시뮬, 표 값 그대로) ----------
M3 = {  # 514 motions, lower is better unless noted
    "SEW-M off": dict(palm=178.979, p95=201.056, ori=7.89e-6, torso_out=0.000, limit=0.0126, coll=0.243, nnad_torso=0.488, pca=1.20e-25, rms=6.83e-14, solve_ms=14.83, nnad_R=0.6578, nnad_L=0.6713),
    "MINK-EF off": dict(palm=0.701, p95=1.853, ori=0.0107, torso_out=0.625, limit=0.1610, coll=0.977, nnad_torso=0.368, pca=173.49, rms=3.117, solve_ms=192.1, nnad_R=0.6951, nnad_L=0.7052),
    "MINK-TE off": dict(palm=18.557, p95=73.980, ori=0.157, torso_out=0.027, limit=0.0852, coll=0.640, nnad_torso=1.026, pca=1119.11, rms=6.106, solve_ms=235.8, nnad_R=1.603, nnad_L=1.776),
    "WARP off": dict(palm=0.0046, p95=0.046, ori=8.74e-6, torso_out=0.000, limit=0.0047, coll=0.163, nnad_torso=0.289, pca=1.14e-25, rms=6.66e-14, solve_ms=6.781, nnad_R=0.6825, nnad_L=0.7103),
    "SEW-M on": dict(palm=215.641, p95=272.842, ori=6.304, torso_out=0.162, limit=0.0131, coll=0.084, nnad_torso=0.442, pca=1.09e-25, rms=6.56e-14, solve_ms=4.978, nnad_R=0.6527, nnad_L=0.6567),
    "MINK-EF on": dict(palm=0.751, p95=2.478, ori=0.0112, torso_out=0.611, limit=0.1247, coll=0.222, nnad_torso=0.354, pca=227.29, rms=3.169, solve_ms=192.5, nnad_R=0.7233, nnad_L=0.7321),
    "MINK-TE on": dict(palm=19.492, p95=69.345, ori=0.188, torso_out=0.049, limit=0.0420, coll=0.478, nnad_torso=0.454, pca=613.47, rms=4.964, solve_ms=234.4, nnad_R=0.9570, nnad_L=0.9569),
    "WARP on": dict(palm=24.048, p95=82.036, ori=3.259, torso_out=0.130, limit=0.0060, coll=0.017, nnad_torso=0.266, pca=1.06e-25, rms=6.46e-14, solve_ms=5.430, nnad_R=0.7270, nnad_L=0.7434),
}
M4 = {  # 7,623 motions, off only
    "SEW-M off": dict(palm_raw=200.2, palm_clean=None, p95=252.0, limit=0.01311, coll=0.1028, nnad_torso=1.125, nnad_R=1.061, solve_ms=7.626, margin=-4.891),
    "MINK-EF off": dict(palm_raw=248.0, palm_clean=0.7856, p95=432.0, limit=0.1603, coll=0.9203, nnad_torso=0.7478, nnad_R=1.266, solve_ms=382.7, margin=-61.74),
    "MINK-TE off": dict(palm_raw=310.3, palm_clean=1.586, p95=654.4, limit=0.1890, coll=0.4639, nnad_torso=3.370, nnad_R=5.089, solve_ms=452.1, margin=-270.0),
    "WARP off": dict(palm_raw=11.91, palm_clean=1.51, p95=43.77, limit=0.02343, coll=0.1112, nnad_torso=0.7921, nnad_R=1.298, solve_ms=10.52, margin=-7.736),
}
checks = {}
def best(table, key, subset):
    rows = {k: v[key] for k, v in table.items() if v.get(key) is not None and subset(k)}
    return min(rows, key=lambda k: rows[k]), rows

for key in ["palm", "p95", "ori", "torso_out", "limit", "coll", "nnad_torso", "pca", "rms", "solve_ms", "nnad_R", "nnad_L"]:
    b_off, _ = best(M3, key, lambda k: k.endswith("off"))
    b_on, _ = best(M3, key, lambda k: k.endswith("on"))
    checks[f"514_{key}"] = {"best_off": b_off, "best_on": b_on}
for key in ["palm_raw", "palm_clean", "p95", "limit", "coll", "nnad_torso", "nnad_R", "solve_ms"]:
    b, _ = best(M4, key, lambda k: True)
    checks[f"7.6k_{key}"] = {"best": b}
margin_best = max(M4, key=lambda k: M4[k]["margin"])
checks["7.6k_margin_higher_better"] = {"best": margin_best}

ratios = {
    "palm_raw_reduction_vs_MINK-EF_pct": round(100 * (1 - 0.0046 / 0.701), 2),
    "palm_raw_reduction_vs_MINK-TE_pct": round(100 * (1 - 0.0046 / 18.557), 3),
    "palm_raw_ratio_MINK-EF_over_WARP": round(0.701 / 0.0046, 1),
    "solve_ratio_MINK-EF_over_WARP_514": round(192.1 / 6.781, 1),
    "solve_ratio_SEW-M_over_WARP_514": round(14.83 / 6.781, 2),
    "palm_clean_ratio_WARP_over_MINK-EF_7.6k": round(1.51 / 0.7856, 2),
    "palm_raw_7.6k_ratio_MINK-EF_over_WARP": round(248.0 / 11.91, 1),
}
# 시간 검산: 프레임 수 135,896 (514 subset, 논문 C.2.1)
frames = 135896
ratios["514_wall_time_WARP_min"] = round(frames * 6.781e-3 / 60, 1)
ratios["514_wall_time_MINK-EF_h"] = round(frames * 192.1e-3 / 3600, 2)
ratios["7.6k_frames_est"] = round(frames * 7623 / 514)
ratios["7.6k_wall_time_WARP_h_est"] = round(ratios["7.6k_frames_est"] * 10.52e-3 / 3600, 2)
ratios["7.6k_wall_time_MINK-EF_h_est"] = round(ratios["7.6k_frames_est"] * 382.7e-3 / 3600, 1)
ratios["claim_1_hour_vs_1_day_note"] = "514 subset 기준 WARP 약 15분, MINK-EF 약 7.3시간. 논문 '1시간 대 하루'와 자릿수가 맞지 않음(코어 수, 전처리 포함 여부 미기재)"

# 실측 분포 제한: palm 표의 평균 vs clean
ratios["WARP_on_palm_raw_vs_clean_ratio"] = round(24.048 / 0.154, 1)

out = {"table1": res1, "fig7_reading": res2, "table3_4_best_checks": checks, "ratios": ratios}
OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
print(json.dumps({"avg": res1["avg_check"], "diff": res1["policy_diff_pts"], "holm50": res1["fisher"]["policy_holm_n50"],
                  "pooled": res1["pooled_policy_n50"], "rollout_mean_diff": res2["rollout_mean_diff_pts_3tasks"],
                  "fig7_holm": res2["holm_p"]}, indent=1, ensure_ascii=False))
print({k: v for k, v in checks.items()})
print(ratios)
