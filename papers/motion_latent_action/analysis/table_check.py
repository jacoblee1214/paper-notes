"""논문 Table I(LIBERO), Table II(RoboTwin), Fig. 3c(실물), Fig. 4(CKA)를 재계산하고 유의성을 본다.

실행: python3 analysis/table_check.py  → analysis/table_check_results.json (CPU, 1초 미만)

시행 수는 논문에 명시되지 않은 부분이 있다. 값이 정수 시행 수로 떨어지는지로 추정하고 결과에 가정을 함께 적는다.
  - LIBERO: 과제당 20회 × 10과제 = 과제군당 200회 (본문 IV-A)
  - RoboTwin: 명시 없음. 모든 칸이 정수 %라서 과제당 100회 가정(50회면 홀수 %가 나오지 않음)
  - 실물: 명시 없음. 0.05 단위 값이라 과제당 20회 가정
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "table_check_results.json"


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


def fisher(k1, n1, k2, n2):
    t = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(stats.fisher_exact(t)[1])


def cmh(tables):
    """Mantel-Haenszel 공통 오즈비 + 연속성 보정 카이제곱. tables: [(k1,n1,k2,n2), ...]"""
    num = den = 0.0
    e_sum = v_sum = 0.0
    for k1, n1, k2, n2 in tables:
        a, b, c, d = k1, n1 - k1, k2, n2 - k2
        n = a + b + c + d
        if n == 0:
            continue
        num += a * d / n
        den += b * c / n
        r1, c1 = a + b, a + c
        e_sum += r1 * c1 / n
        v_sum += r1 * (n - r1) * c1 * (n - c1) / (n * n * (n - 1))
    chi = (abs(sum(t[0] for t in tables) - e_sum) - 0.5) ** 2 / v_sum
    return {"or_mh": round(num / den, 3) if den else None, "p": float(stats.chi2.sf(chi, 1))}


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def pct_to_count(p, n):
    return p * n / 100.0


res = {"assumptions": {
    "libero_trials_per_suite": 200, "robotwin_trials_per_task": 100, "real_trials_per_task": 20,
    "note": "시행 수는 논문 본문에 없고 값의 정수성으로 추정함"}}

# 1. 정수 시행 수 점검: 보고된 %가 n회 시행의 정수 성공 수로 떨어지는가
reported = {
    "LIBERO-Spatial": {"Ours": 95.5, "Ours w/o DINO": 91.0, "villa-x": 97.5, "UniVLA-Bridge": 90.0,
                       "pi0-fast": 96.4, "pi0": 88.0, "SpatialVLA": 88.2, "OpenVLA": 84.7, "LAPA": 73.8, "DP": 78.3},
    "LIBERO-Object": {"Ours": 94.0, "Ours w/o DINO": 81.0, "villa-x": 97.0, "UniVLA-Bridge": 86.0,
                      "pi0-fast": 96.8, "pi0": 88.5, "SpatialVLA": 89.9, "OpenVLA": 88.4, "LAPA": 74.6, "DP": 92.5},
    "LIBERO-Goal": {"Ours": 93.5, "Ours w/o DINO": 88.5, "villa-x": 91.5, "UniVLA-Bridge": 90.5,
                    "pi0-fast": 88.6, "pi0": 87.0, "SpatialVLA": 78.6, "OpenVLA": 79.2, "LAPA": 58.8, "DP": 68.3},
    "LIBERO-Long": {"Ours": 84.0, "Ours w/o DINO": 81.0, "villa-x": 74.5, "UniVLA-Bridge": 86.0,
                    "pi0-fast": 60.2, "pi0": 61.0, "SpatialVLA": 55.5, "OpenVLA": 53.7, "LAPA": 55.4, "DP": 50.5},
}
integer_check = {}
for suite, row in reported.items():
    for name, p in row.items():
        ok200 = abs(pct_to_count(p, 200) - round(pct_to_count(p, 200))) < 1e-6
        ok500 = abs(pct_to_count(p, 500) - round(pct_to_count(p, 500))) < 1e-6
        integer_check[f"{suite}/{name}"] = {"n200_integer": ok200, "n500_integer": ok500}
res["libero_integer_check"] = integer_check
res["libero_nonint_200"] = [k for k, v in integer_check.items() if not v["n200_integer"]]

# 2. 평균 재계산 (Table I 평균 열)
avg_reported = {"Ours": 91.8, "Ours w/o DINO": 85.4, "villa-x": 90.1, "UniVLA-Bridge": 88.1,
                "pi0-fast": 85.5, "pi0": 81.1, "SpatialVLA": 78.1, "OpenVLA": 76.5, "LAPA": 65.7, "DP": 72.4}
avg_recomputed = {m: round(np.mean([reported[s][m] for s in reported]), 3) for m in avg_reported}
res["libero_avg"] = {m: {"reported": avg_reported[m], "recomputed": avg_recomputed[m]} for m in avg_reported}

# 3. Ours 대 기준선: 과제군별 Fisher, 과제군 층화 CMH, 합산 Fisher + Holm
N_L = 200
ours = reported
suites = list(reported.keys())
baselines = ["villa-x", "UniVLA-Bridge", "pi0-fast", "pi0", "SpatialVLA", "OpenVLA", "LAPA", "DP"]
libero_cmp = {}
pooled_p = []
for b in baselines:
    strata = []
    per_suite_p = {}
    for s in suites:
        k1 = round(pct_to_count(ours[s]["Ours"] if "Ours" in ours[s] else ours[s]["Ours"], N_L))
        k2 = round(pct_to_count(reported[s][b], N_L))
        strata.append((k1, N_L, k2, N_L))
        per_suite_p[s] = round(fisher(k1, N_L, k2, N_L), 6)
    k1 = sum(t[0] for t in strata)
    k2 = sum(t[2] for t in strata)
    n_tot = N_L * len(suites)
    p_pool = fisher(k1, n_tot, k2, n_tot)
    pooled_p.append(p_pool)
    libero_cmp[b] = {
        "ours_pooled_pct": round(100 * k1 / n_tot, 2), "base_pooled_pct": round(100 * k2 / n_tot, 2),
        "diff_pts": round(100 * (k1 - k2) / n_tot, 2),
        "fisher_pooled_p": p_pool, "per_suite_fisher_p": per_suite_p,
        "cmh_suite_stratified": cmh(strata),
        "ours_wilson_pooled": wilson(k1, n_tot), "base_wilson_pooled": wilson(k2, n_tot),
    }
for b, adj in zip(baselines, holm(pooled_p)):
    libero_cmp[b]["holm_adj_p_over_8_baselines"] = round(adj, 6)
res["libero_ours_vs_baselines"] = libero_cmp

# Ours 대 w/o DINO (같은 과제군, 같은 시행 수로 가정)
strata = [(round(pct_to_count(ours[s]["Ours"], N_L)), N_L,
           round(pct_to_count(reported[s]["Ours w/o DINO"], N_L)), N_L) for s in suites]
res["libero_ours_vs_noDINO"] = {"cmh": cmh(strata),
                                "fisher_pooled_p": fisher(sum(t[0] for t in strata), 800,
                                                          sum(t[2] for t in strata), 800)}

# 4. Wilson 구간 (Ours, 과제군별)
res["libero_ours_wilson_per_suite"] = {s: wilson(round(pct_to_count(ours[s]["Ours"], N_L)), N_L) for s in suites}

# 5. 검출력: 공통 시행 수로 91.8 대 90.1, 91.8 대 88.1 같은 차이를 80%로 잡으려면 한 팔당 몇 회가 필요한가
def n_per_arm(p1, p2, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / (p1 - p2) ** 2))


res["power"] = {
    "n_per_arm_for_1.7pt_at_0.90": n_per_arm(0.918, 0.901),
    "n_per_arm_for_3.7pt_at_0.88": n_per_arm(0.918, 0.881),
    "n_per_arm_for_10pt_at_0.75": n_per_arm(0.84, 0.745),
}

# 6. RoboTwin Table II (과제당 100회 가정, 10과제)
robo_tasks = ["Adjust bottle", "Grab roller", "Place phone stand", "Pick dual bottles", "Place empty cup",
              "Move can pot", "Handover mic", "Open laptop", "Place object basket", "Place burger fries"]
robo = {
    "RDT": [81, 74, 15, 42, 56, 25, 90, 59, 33, 50],
    "pi0": [90, 96, 35, 57, 37, 58, 98, 85, 16, 80],
    "ACT": [97, 94, 2, 31, 61, 22, 85, 56, 15, 49],
    "DP": [97, 98, 13, 24, 37, 39, 53, 49, 15, 72],
    "UniVLA": [87, 80, 40, 58, 46, 60, 82, 81, 18, 84],
    "Ours w/o DINO": [87, 87, 30, 58, 51, 52, 86, 81, 16, 80],
    "Ours (Freeze)": [83, 50, 8, 42, 48, 52, 84, 74, 11, 72],
    "Ours": [97, 90, 28, 52, 63, 65, 92, 87, 25, 78],
}
robo_avg_reported = {"RDT": 52.5, "pi0": 65.2, "ACT": 51.2, "DP": 49.7, "UniVLA": 63.6,
                     "Ours w/o DINO": 62.8, "Ours (Freeze)": 52.4, "Ours": 67.7}
N_R = 100
robo_res = {"avg_check": {m: {"reported": robo_avg_reported[m], "recomputed": round(np.mean(v), 2)}
                          for m, v in robo.items()}}
robo_p, robo_cmp = [], {}
others = [m for m in robo if m != "Ours"]
for m in others:
    strata = [(robo["Ours"][i], N_R, robo[m][i], N_R) for i in range(10)]
    tot = 10 * N_R
    k1, k2 = sum(robo["Ours"]), sum(robo[m])
    p = fisher(k1, tot, k2, tot)
    robo_p.append(p)
    robo_cmp[m] = {"diff_pts": round((k1 - k2) / 10, 2), "fisher_pooled_p": p,
                   "cmh_task_stratified": cmh(strata)}
for m, adj in zip(others, holm(robo_p)):
    robo_cmp[m]["holm_adj_p_over_7"] = round(adj, 6)
# 과제별 부호 검정 (Ours 대 UniVLA, pi0): 동점 제외
for m in ["UniVLA", "pi0"]:
    diffs = [robo["Ours"][i] - robo[m][i] for i in range(10)]
    wins = sum(d > 0 for d in diffs)
    losses = sum(d < 0 for d in diffs)
    robo_cmp[m]["task_sign_test"] = {"wins": wins, "losses": losses, "ties": 10 - wins - losses,
                                     "p_two_sided": float(stats.binomtest(wins, wins + losses, 0.5).pvalue)}
robo_res["vs_ours"] = robo_cmp
res["robotwin"] = robo_res

# 7. 실물 Fig. 3c (과제당 20회 가정; 값 읽기는 그림 확대본으로 함)
real = {"UniVLA": {"Place bottle on plate": 0.30, "Unplug": 0.35, "Fold towel": 0.50},
        "Ours": {"Place bottle on plate": 0.75, "Unplug": 0.80, "Fold towel": 0.75}}
N_RW = 20
real_res = {"per_task": {}}
strata = []
for t in real["Ours"]:
    k1 = round(real["Ours"][t] * N_RW)
    k2 = round(real["UniVLA"][t] * N_RW)
    strata.append((k1, N_RW, k2, N_RW))
    real_res["per_task"][t] = {"ours": f"{k1}/{N_RW}", "univla": f"{k2}/{N_RW}",
                               "fisher_p": round(fisher(k1, N_RW, k2, N_RW), 4)}
k1 = sum(t[0] for t in strata)
k2 = sum(t[2] for t in strata)
real_res["pooled"] = {"ours": f"{k1}/60", "univla": f"{k2}/60",
                      "fisher_p": round(fisher(k1, 60, k2, 60), 6), "cmh_task_stratified": cmh(strata)}
real_res["holm_over_3_tasks"] = [round(x, 4) for x in holm([real_res["per_task"][t]["fisher_p"] for t in real["Ours"]])]
res["real_world"] = real_res

# 8. CKA (본문 IV-C): 부트스트랩 표준편차로 Welch 검정. 부트스트랩 편차는 재표본 변동이지 학습 시드 변동이 아님
cka = {"ours": (0.9139, 0.0099), "univla": (0.8659, 0.0064)}
B = 50
se = np.sqrt(cka["ours"][1] ** 2 / B + cka["univla"][1] ** 2 / B)
t = (cka["ours"][0] - cka["univla"][0]) / se
res["cka"] = {"diff": round(cka["ours"][0] - cka["univla"][0], 4), "welch_t_if_bootstrap_std_is_sampling_sd": round(float(t), 2),
              "caveat": "학습 시드가 1개인지 불명확. 부트스트랩 표준편차를 시드 변동으로 볼 수 없음"}

OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
print("wrote", OUT)
print("non-integer 200-trial entries:", res["libero_nonint_200"])
print("LIBERO avg recomputed:", {m: avg_recomputed[m] for m in ["Ours", "villa-x", "UniVLA-Bridge"]})
for b in ["villa-x", "UniVLA-Bridge", "pi0-fast"]:
    print(b, libero_cmp[b]["diff_pts"], "fisher", round(libero_cmp[b]["fisher_pooled_p"], 4),
          "cmh", libero_cmp[b]["cmh_suite_stratified"], "holm", libero_cmp[b]["holm_adj_p_over_8_baselines"])
print("RoboTwin UniVLA:", robo_cmp["UniVLA"]["diff_pts"], robo_cmp["UniVLA"]["fisher_pooled_p"],
      robo_cmp["UniVLA"]["cmh_task_stratified"], robo_cmp["UniVLA"]["holm_adj_p_over_7"], robo_cmp["UniVLA"]["task_sign_test"])
print("RoboTwin pi0:", robo_cmp["pi0"]["diff_pts"], robo_cmp["pi0"]["fisher_pooled_p"], robo_cmp["pi0"]["holm_adj_p_over_7"])
print("Real pooled:", real_res["pooled"])
