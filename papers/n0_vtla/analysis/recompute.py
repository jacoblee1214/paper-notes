"""N0-VTLA (arXiv:2607.23782 v1) 논문 표·그래프 수치를 다시 계산하고 검정한다.

입력: 논문 Table 1, Table 2 값은 본문에서 옮겼다. Figure 7, 9, 12의 막대 값은 그래프 라벨을 읽은 것이다
(논문 본문과 대조해 일치를 확인했다). 외부 데이터는 쓰지 않는다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 실로봇 시도 수 n은 논문에 없다. 그래프 값이 모두 5%의 배수이므로 n = 20을 기본으로 두고 n = 40을 민감도로 본다.
  (n = 10이면 45%가 나올 수 없어 기각한다.)
- 시뮬레이션 시도 수도 논문에 없다. 시뮬 비교는 과제 단위 쌍비교(20개 과제)로 본다.
- Figure 12의 처리량은 정수로 읽었다(±0.5 가능).
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="two-sided")[1])


# ---------------------------------------------------------------- K1 표 평균 재계산
T1 = {  # UniVTAC, Table 1 (Lift Bottle, Pull-out Key, Lift Can, Put Bottle, Insert Hole, Insert HDMI, Insert Tube, Grasp Classify)
    "ACT (Vision Only)": ([42, 28, 20, 28, 19, 15, 45, 50], 30.9),
    "ACT + UniVTAC": ([71, 46, 29, 31, 24, 28, 56, 99], 48.0),
    "VITaL": ([72, 47, 8, 32, 25, 6, 34, 100], 40.5),
    "pi0.5": ([100, 35, 6, 34, 25, 8, 74, 49], 41.4),
    "StarVLA-alpha": ([32, 51, 65, 88, 52, 24, 69, 68], 56.1),
    "InternVLA-A1": ([58, 66, 90, 37, 93, 12, 95, 86], 67.1),
    "Xiaomi-Robotics-0": ([21, 80, 13, 12, 96, 69, 98, 45], 54.3),
    "GigaWorld-Policy": ([38, 32, 0, 21, 12, 0, 9, 20], 16.5),
    "N0-VTLA": ([99, 99, 88, 60, 95, 25, 99, 100], 83.1),
}
T1_TASKS = ["Lift Bottle", "Pull-out Key", "Lift Can", "Put Bottle", "Insert Hole", "Insert HDMI", "Insert Tube", "Grasp Classify"]
T2 = {  # NeoSim, Table 2: 4 single-arm, 8 dual-arm (order as in paper)
    "pi0.5": ([92, 23, 74, 86], [26, 18, 3, 13, 22, 96, 93, 3], 68.8, 34.3, 45.8),
    "StarVLA-alpha": ([32, 0, 72, 93], [8, 0, 17, 25, 0, 0, 0, 31], 49.3, 10.1, 23.2),
    "InternVLA-A1": ([47, 0, 36, 12], [0, 0, 0, 0, 0, 0, 8, 0], 23.8, 1.0, 8.6),
    "Xiaomi-Robotics-0": ([52, 5, 62, 49], [0, 0, 0, 0, 0, 42, 71, 0][:0] or [0, 0, 0, 0, 0, 0, 42, 71], 42.0, 14.1, 23.4),
    "GigaWorld-Policy": ([0, 3, 34, 71], [0, 0, 0, 0, 12, 0, 0, 9], 27.0, 2.6, 10.8),
    "N0-VTLA": ([96, 30, 81, 88], [24, 29, 17, 22, 23, 94, 96, 10], 73.8, 39.4, 50.8),
}
T2_TASKS = ["Pour Ball", "Unplug and Plug Charger", "Plug USB", "Grasp Chip",
            "Insert Screw", "Place Gears", "Unstack Bowl", "Cup Handover", "Stack Cups", "Stack Plates", "Stack Bowls", "Unstack Cup"]

rows = {}
for m, (v, rep) in T1.items():
    rows[m] = {"univtac": v, "univtac_mean": round(float(np.mean(v)), 3), "univtac_paper": rep}
for m, (s, d, sm, dm, am) in T2.items():
    rows.setdefault(m, {})
    rows[m].update({"neosim_single": s, "neosim_dual": d,
                    "single_mean": round(float(np.mean(s)), 3), "single_paper": sm,
                    "dual_mean": round(float(np.mean(d)), 3), "dual_paper": dm,
                    "neosim_mean": round(float(np.mean(s + d)), 3), "neosim_paper": am})
RES["K1_table_means"] = {m: {k: rows[m][k] for k in rows[m] if k.endswith("mean") or k.endswith("paper")} for m in rows}
K_flags = []
for m in rows:
    for a, b in [("univtac_mean", "univtac_paper"), ("single_mean", "single_paper"),
                 ("dual_mean", "dual_paper"), ("neosim_mean", "neosim_paper")]:
        if a in rows[m] and abs(round(rows[m][a], 1) - rows[m][b]) > 0.051:
            K_flags.append((m, a, rows[m][a], rows[m][b]))
RES["K1_mismatches"] = K_flags

# 20과제 평균 (UniVTAC 8 + NeoSim 12, 과제 수 가중)
avg20 = {}
for m in rows:
    if "univtac" in rows[m] and "neosim_single" in rows[m]:
        allv = rows[m]["univtac"] + rows[m]["neosim_single"] + rows[m]["neosim_dual"]
        avg20[m] = round(float(np.mean(allv)), 3)
RES["K2_avg20_tasks"] = avg20
RES["K2_strongest_external_avg20"] = max((m for m in avg20 if m != "N0-VTLA"), key=lambda m: avg20[m])

# 본문 "best per task" 확인: N0-VTLA가 UniVTAC 8과제 중 최고(동률 포함)인 과제 수
n0 = T1["N0-VTLA"][0]
best_count = []
for i, t in enumerate(T1_TASKS):
    others = {m: v[0][i] for m, v in T1.items() if m != "N0-VTLA"}
    top = max(others.values())
    best_count.append({"task": t, "n0": n0[i], "best_external": max(others, key=others.get), "best_external_value": top,
                       "n0_is_max_or_tied": n0[i] >= top})
RES["K3_univtac_best_per_task"] = best_count
RES["K3_n0_max_or_tied_count"] = sum(b["n0_is_max_or_tied"] for b in best_count)

# 과제별 쌍비교 (N0 - pi0.5), 20과제
n0_all = n0 + T2["N0-VTLA"][0] + T2["N0-VTLA"][1]
pi_all = T1["pi0.5"][0] + T2["pi0.5"][0] + T2["pi0.5"][1]
diff = np.array(n0_all, float) - np.array(pi_all, float)
names20 = T1_TASKS + T2_TASKS
t_res = stats.ttest_1samp(diff, 0.0)
w_res = stats.wilcoxon(diff)
mean_d = float(diff.mean())
half = float(stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / math.sqrt(len(diff)))
RES["C_paired_20tasks_vs_pi05"] = {
    "mean_diff_pp": round(mean_d, 2), "ci95_pp": [round(mean_d - half, 2), round(mean_d + half, 2)],
    "t_p": float(t_res.pvalue), "wilcoxon_p": float(w_res.pvalue),
    "n0_higher_tasks": int((diff > 0).sum()), "n0_lower_tasks": int((diff < 0).sum()),
    "lower_tasks": [names20[i] for i in range(20) if diff[i] < 0],
    "per_task_diff": {names20[i]: float(diff[i]) for i in range(20)},
}
RES["C_paired_20tasks_vs_pi05"]["note"] = "과제를 표본으로 본 검정. 과제 내부 시도 수 변동은 포함되지 않는다."

# ---------------------------------------------------------------- 실로봇 NeoReal (Figure 7 읽기)
NR_TASKS = ["Cup Stacking", "Fruit Collection", "Board Wiping", "Bottle Standing", "Towel Folding",
            "Socket Plugging", "Board Insertion", "Cardboard Box Folding", "Bag Packing"]
NR_SUCC = {"ACT": [0] * 9,
           "pi0.5": [50, 50, 40, 0, 40, 60, 0, 5, 20],
           "N0-VTLA": [75, 60, 45, 30, 50, 85, 25, 20, 35]}
NR_PROG = {"ACT": [18, 21, 12, 15, 9, 7, 5, 0, 5],
           "pi0.5": [66, 67.5, 55.2, 0, 56.8, 73.5, 0, 19, 43],
           "N0-VTLA": [75.75, 69, 52, 45, 69.75, 77, 34, 37.2, 51.25]}
RES["K4_neoreal_means"] = {m: {"success_mean": round(float(np.mean(v)), 2),
                                "progress_mean": round(float(np.mean(NR_PROG[m])), 2)} for m, v in NR_SUCC.items()}
RES["K4_neoreal_paper"] = {"success": {"pi0.5": 29.4, "N0-VTLA": 47.2}, "progress": {"pi0.5": 42.3, "N0-VTLA": 56.8, "ACT": 10.2}}
prog_lead = sum(n > p for n, p in zip(NR_PROG["N0-VTLA"], NR_PROG["pi0.5"]))
RES["K4_progress_lead_count"] = prog_lead  # 논문: 9과제 중 8

for n_trials in (20, 40):
    per = []
    for i, t in enumerate(NR_TASKS):
        k_n = round(NR_SUCC["N0-VTLA"][i] * n_trials / 100)
        k_p = round(NR_SUCC["pi0.5"][i] * n_trials / 100)
        lo_n, hi_n = wilson(k_n, n_trials)
        per.append({"task": t, "n0_k": k_n, "pi_k": k_p,
                    "n0_ci95": [round(100 * lo_n, 1), round(100 * hi_n, 1)],
                    "fisher_p": round(fisher(k_n, n_trials, k_p, n_trials), 4)})
    # 쌍 부호검정: 9과제 중 N0가 모두 높으면 p = 2 * (1/2)^9
    wins = sum(p["n0_k"] > p["pi_k"] for p in per)
    sign_p = float(stats.binomtest(wins, 9, 0.5, alternative="greater").pvalue) if wins else 1.0
    K = sum(NR_SUCC["N0-VTLA"]) * n_trials / 100
    Kp = sum(NR_SUCC["pi0.5"]) * n_trials / 100
    pooled_p = fisher(round(K), 9 * n_trials, round(Kp), 9 * n_trials)
    RES[f"C_neoreal_n{n_trials}"] = {"per_task": per, "wins_of_9": wins,
                                     "sign_test_one_sided_p": round(sign_p, 5),
                                     "pooled_fisher_p_assumes_independence": round(pooled_p, 6)}

# Socket Plugging (85 vs 60), SFT 차이, ALTER 차이 등 개별 비교
RES["C_key_pairs_n20"] = {
    "socket_plugging_85_vs_60": round(fisher(17, 20, 12, 20), 4),
    "board_insertion_25_vs_0": round(fisher(5, 20, 0, 20), 4),
    "bottle_standing_30_vs_0": round(fisher(6, 20, 0, 20), 4),
}

# ---------------------------------------------------------------- Figure 9: SFT과 ALTER (n = 20 가정)
F9 = {"Towel Folding": {"pi0.5-SFT": 40, "N0-SFT": 50, "pi0.5+chi0": 80, "pi0.5+ALTER": 90, "N0+ALTER": 95},
      "Bag Packing": {"pi0.5-SFT": 20, "N0-SFT": 35, "pi0.5+chi0": 65, "pi0.5+ALTER": 75, "N0+ALTER": 80},
      "Cardboard Box Folding": {"pi0.5-SFT": 5, "N0-SFT": 20, "pi0.5+chi0": 55, "pi0.5+ALTER": 60, "N0+ALTER": 75}}
f9 = {}
for t, d in F9.items():
    f9[t] = {
        "SFT_diff_pp": d["N0-SFT"] - d["pi0.5-SFT"],
        "SFT_fisher_p_n20": round(fisher(d["N0-SFT"] * 20 // 100, 20, d["pi0.5-SFT"] * 20 // 100, 20), 4),
        "ALTER_diff_pp": d["N0+ALTER"] - d["pi0.5+ALTER"],
        "ALTER_fisher_p_n20": round(fisher(d["N0+ALTER"] * 20 // 100, 20, d["pi0.5+ALTER"] * 20 // 100, 20), 4),
        "ALTER_pi05_vs_chi0_fisher_p_n20": round(fisher(d["pi0.5+ALTER"] * 20 // 100, 20, d["pi0.5+chi0"] * 20 // 100, 20), 4),
    }
RES["C_figure9_sft_alter"] = f9
# 논문 본문 "10, 15, 15점" 확인
RES["K5_sft_lead_points"] = [f9[t]["SFT_diff_pp"] for t in F9]

# ---------------------------------------------------------------- Figure 12: 처리량과 성공률로 시도 속도 역산
F12 = {"Towel Folding": {"pi0.5-SFT": (18, 40), "N0-SFT": (23, 50), "pi0.5+chi0": (36, 80), "pi0.5+ALTER": (41, 90), "N0+ALTER": (43, 95)},
       "Bag Packing": {"pi0.5-SFT": (4, 20), "N0-SFT": (7, 35), "pi0.5+chi0": (12, 65), "pi0.5+ALTER": (14, 75), "N0+ALTER": (15, 80)},
       "Cardboard Box Folding": {"pi0.5-SFT": (4, 5), "N0-SFT": (8, 20), "pi0.5+chi0": (22, 55), "pi0.5+ALTER": (24, 60), "N0+ALTER": (30, 75)}}
f12 = {}
for t, d in F12.items():
    f12[t] = {m: {"throughput_per_h": th, "success_pct": s, "implied_attempts_per_h": round(th / (s / 100), 1)}
              for m, (th, s) in d.items()}
RES["K6_throughput_implied_attempts_per_h"] = f12
# 반올림 범위(처리량 ±0.5)에서 π0.5-SFT Cardboard의 시도 속도
lo, hi = (4 - 0.5) / 0.05, (4 + 0.5) / 0.05
other = [v["implied_attempts_per_h"] for m, v in f12["Cardboard Box Folding"].items() if m != "pi0.5-SFT"]
RES["K6_cardboard_pi05_sft_range_attempts_per_h"] = [round(lo, 1), round(hi, 1)]
RES["K6_cardboard_other_variants_attempts_per_h"] = other

# ---------------------------------------------------------------- 표현 분석 (5.5)
k_top1, n_q = round(0.923 * 378), 378
RES["C_representation"] = {
    "top1_k": k_top1, "n": n_q, "top1_pct": round(100 * k_top1 / n_q, 2),
    "top1_wilson95_pct": [round(100 * x, 1) for x in wilson(k_top1, n_q)],
    "chance_pool32_binom_p": float(stats.binomtest(k_top1, n_q, 1 / 32, alternative="greater").pvalue),
    "current_touch_top1_vs_pred_fisher_p_assumes_n378": round(fisher(k_top1, n_q, round(0.57 * n_q), n_q), 8),
    "pool128_predictor_81_vs_current_40_note": "시험 수 미보고 (n 가정 불가)",
    "sensitivity_ratio_claimed_4_3_from_0_9_over_0_2": round(0.9 / 0.2, 2),
}

# ---------------------------------------------------------------- 출력
OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print("K1 mismatches (표 평균 재계산 vs 논문, 0.05 초과):", K_flags or "없음")
print("K2 20과제 평균:", {k: avg20[k] for k in ["N0-VTLA", "pi0.5", "InternVLA-A1", "Xiaomi-Robotics-0", "StarVLA-alpha"]},
      "최고 외부:", RES["K2_strongest_external_avg20"])
print("K3 UniVTAC에서 N0 최고(동률 포함) 과제 수:", RES["K3_n0_max_or_tied_count"], "/ 8")
c = RES["C_paired_20tasks_vs_pi05"]
print("C 20과제 쌍비교 평균차 %.2f pp, 95%% CI [%.2f, %.2f], t p=%.2g, Wilcoxon p=%.2g, N0 우위 %d/20, 열세 과제 %s"
      % (c["mean_diff_pp"], c["ci95_pp"][0], c["ci95_pp"][1], c["t_p"], c["wilcoxon_p"], c["n0_higher_tasks"], c["lower_tasks"]))
for nt in (20, 40):
    r = RES[f"C_neoreal_n{nt}"]
    print(f"C NeoReal n={nt}: N0 우위 {r['wins_of_9']}/9, 부호검정 단측 p={r['sign_test_one_sided_p']}, 합산 Fisher p={r['pooled_fisher_p_assumes_independence']}")
print("K4 NeoReal 평균:", RES["K4_neoreal_means"], "진행도 우위 과제:", prog_lead, "/ 9")
print("C 주요 쌍 Fisher (n=20):", RES["C_key_pairs_n20"])
print("C Figure 9 (n=20) SFT 차이 p:", {t: (f9[t]["SFT_diff_pp"], f9[t]["SFT_fisher_p_n20"]) for t in f9})
print("C Figure 9 (n=20) ALTER 차이 p:", {t: (f9[t]["ALTER_diff_pp"], f9[t]["ALTER_fisher_p_n20"]) for t in f9})
print("K6 시도 속도(시도/시간):", {t: {m: v["implied_attempts_per_h"] for m, v in f12[t].items()} for t in f12})
print("K6 Cardboard π0.5-SFT 반올림 범위:", RES["K6_cardboard_pi05_sft_range_attempts_per_h"], "다른 변형:", other)
print("C 표현: top-1", RES["C_representation"]["top1_pct"], "Wilson", RES["C_representation"]["top1_wilson95_pct"],
      "chance p", RES["C_representation"]["chance_pool32_binom_p"])
print("wrote", OUT)
