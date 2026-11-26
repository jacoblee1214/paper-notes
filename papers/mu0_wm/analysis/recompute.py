"""μ0 (arXiv:2606.13769 v3) 보고 수치 재계산과 일관성 검사. 결과는 recompute_results.json에 저장한다.

실행: python3 analysis/recompute.py   (numpy, scipy; 수 초, 단일 프로세스)

입력은 모두 논문 v3의 표와 본문 값을 옮겨 적은 것이다(표·절 번호를 주석에 적음).
Fig. 6의 실로봇 막대 값은 그림에서 읽은 값이며(그림 판독 값), 표 2 이외의 시행 수는 본문 서술을 따른다.
검사 라벨: V1–V19 (재계산 검사), 해석은 03_analysis.md에 있다.
"""
import json
import math
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
R = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * (c - h), 100 * (c + h))


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def cmh(strata):
    """Mantel-Haenszel 층화 검정(연속성 보정 없음). strata: [(k1,n1,k2,n2), ...]"""
    num, var, or_n, or_d = 0.0, 0.0, 0.0, 0.0
    for k1, n1, k2, n2 in strata:
        n = n1 + n2
        a, b, c, d = k1, n1 - k1, k2, n2 - k2
        m1, m0 = a + c, b + d
        e = n1 * m1 / n
        v = n1 * n2 * m1 * m0 / (n * n * (n - 1))
        num += a - e
        var += v
        or_n += a * d / n
        or_d += b * c / n
    chi2 = num * num / var
    return {"chi2": chi2, "p": float(stats.chi2.sf(chi2, 1)), "or_mh": or_n / or_d if or_d else float("inf")}


# ---------------------------------------------------------------- V1 표 2 (RoboCasa365, 과제당 50 롤아웃)
T2_TASKS = ["CloseFridge", "OpenFridge", "CoffeeServeMug", "PickPlaceFridgeShelfToDrawer", "TurnOnMicrowave",
            "SlideToasterOvenRack", "PickPlaceCounterToCabinet", "TurnOnToasterOven"]
T2 = {  # 표 2 (%); 평균 행: DP 22.75, pi0 25.25, pi0.5 42, TraceGen 23, mu0 30.25
    "DP": [34, 28, 28, 28, 0, 48, 6, 10],
    "pi0": [44, 12, 34, 30, 2, 46, 18, 16],
    "pi05": [34, 26, 48, 66, 12, 76, 54, 20],
    "TraceGen": [38, 36, 42, 30, 0, 28, 0, 10],
    "mu0": [54, 18, 36, 40, 4, 56, 12, 22],
}
T2_REPORTED_AVG = {"DP": 22.75, "pi0": 25.25, "pi05": 42.0, "TraceGen": 23.0, "mu0": 30.25}
N_SIM = 50
v1 = {}
for k, v in T2.items():
    v1[k] = {"mean": float(np.mean(v)), "reported": T2_REPORTED_AVG[k],
             "all_even_multiples_of_2": all(x % 2 == 0 for x in v),
             "successes_total": int(sum(x * N_SIM / 100 for x in v))}
R["V1_table2_means"] = v1

# V2 표 2 해석: 과제별 승패, 부호 검정, 쌍체 Wilcoxon, 풀링 Fisher, CMH, 모수 부트스트랩 CI
rng = np.random.default_rng(0)
v2 = {}
succ = {k: np.array([x * N_SIM / 100 for x in v], dtype=int) for k, v in T2.items()}
for other in ["DP", "pi0", "pi05", "TraceGen"]:
    d = succ["mu0"] - succ[other]
    wins, losses, ties = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    sign_p = float(stats.binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0
    try:
        w_p = float(stats.wilcoxon(d, zero_method="wilcox", method="exact").pvalue)
    except ValueError:
        w_p = None
    k1, k2 = int(succ["mu0"].sum()), int(succ[other].sum())
    n = N_SIM * 8
    per_task_fisher = [fisher(int(a), N_SIM, int(b), N_SIM) for a, b in zip(succ["mu0"], succ[other])]
    strata = [(int(a), N_SIM, int(b), N_SIM) for a, b in zip(succ["mu0"], succ[other])]
    # 모수 부트스트랩: 과제별 성공 확률을 관측 비율로 두고 이항 재표집, 평균 차 (퍼센트포인트)
    pm = succ["mu0"] / N_SIM
    po = succ[other] / N_SIM
    diffs = []
    for _ in range(20000):
        a = rng.binomial(N_SIM, pm) / N_SIM
        b = rng.binomial(N_SIM, po) / N_SIM
        diffs.append(100 * (a.mean() - b.mean()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    v2[other] = {
        "mean_diff_pp": float(100 * (pm.mean() - po.mean())),
        "wins_losses_ties": [wins, losses, ties],
        "sign_test_p": sign_p,
        "wilcoxon_exact_p": w_p,
        "pooled_fisher_p_400_vs_400": fisher(k1, n, k2, n),
        "pooled_counts": [k1, k2, n],
        "cmh": cmh(strata),
        "n_tasks_fisher_p_lt_0.05": int(sum(p < 0.05 for p in per_task_fisher)),
        "per_task_fisher_p": [round(p, 4) for p in per_task_fisher],
        "bootstrap_ci95_mean_diff_pp": [float(lo), float(hi)],
    }
R["V2_table2_stats_mu0_vs"] = v2
R["V2_wilson_pooled_400"] = {k: wilson(int(succ[k].sum()), 400) for k in succ}

# ---------------------------------------------------------------- V3 실로봇 (Fig. 6 판독 값, 방법당 과제당 20 롤아웃)
REAL = {  # Pick&Place, Pour, Unfold (%)  -- 그림 판독 값
    "VLM+AE": [75, 75, 70],
    "pi0": [85, 80, 50],
    "pi05": [95, 80, 65],
    "TraceGen+AE": [80, 80, 85],
    "mu0+AE": [90, 90, 95],
}
REAL_AVG_FIG = {"VLM+AE": 73.3, "pi0": 71.7, "pi05": 80.0, "TraceGen+AE": 81.7, "mu0+AE": 91.7}
N_REAL = 20
v3 = {}
for k, v in REAL.items():
    v3[k] = {"mean": float(np.mean(v)), "fig_label": REAL_AVG_FIG[k],
             "multiples_of_5": all(x % 5 == 0 for x in v)}
R["V3_real_means"] = v3
text_gaps = {"vs VLM+AE": 18.4, "vs pi0": 20.0, "vs pi05": 11.7, "vs TraceGen+AE": 10.0}  # 4.2절 본문
ref = {"vs VLM+AE": "VLM+AE", "vs pi0": "pi0", "vs pi05": "pi05", "vs TraceGen+AE": "TraceGen+AE"}
R["V3_real_text_gaps"] = {g: {"text": t, "recomputed": round(float(np.mean(REAL["mu0+AE"]) - np.mean(REAL[ref[g]])), 2)}
                          for g, t in text_gaps.items()}
rs = {k: np.array([x * N_REAL / 100 for x in v], dtype=int) for k, v in REAL.items()}
v4 = {}
for other in ["VLM+AE", "pi0", "pi05", "TraceGen+AE"]:
    k1, k2 = int(rs["mu0+AE"].sum()), int(rs[other].sum())
    strata = [(int(a), N_REAL, int(b), N_REAL) for a, b in zip(rs["mu0+AE"], rs[other])]
    v4[other] = {
        "pooled_counts_60": [k1, k2],
        "pooled_fisher_p": fisher(k1, 60, k2, 60),
        "per_task_fisher_p": [round(fisher(int(a), 20, int(b), 20), 4) for a, b in zip(rs["mu0+AE"], rs[other])],
        "cmh": cmh(strata),
        "wilson95_mu0": wilson(k1, 60),
        "wilson95_other": wilson(k2, 60),
    }
R["V4_real_stats"] = v4
R["V4_real_per_task_unfold_mu0_vs_pi0"] = {"counts": [19, 10], "fisher_p": fisher(19, 20, 10, 20)}
R["V4_real_pp_task_mu0_vs_pi05"] = {"counts": [18, 19], "fisher_p": fisher(18, 20, 19, 20)}

# V5 서론 비율 "120–130% of pi0, 70–115% of pi0.5"
R["V5_intro_ratios"] = {
    "sim_vs_pi0": 30.25 / 25.25, "real_vs_pi0": (275 / 3) / (215 / 3),
    "sim_vs_pi05": 30.25 / 42.0, "real_vs_pi05": (275 / 3) / 80.0,
}

# ---------------------------------------------------------------- V6 표 1/4 (trace prediction)
METHODS = ["Gemini 3.1 Pro", "Gemini 3 Flash", "GPT-5.5", "Track2Act", "Hamster", "mu0_2D",
           "3DFlowAction", "Dream2Flow", "TraceGen", "mu0_3D"]
# 열 순서: top1-ADE(8,16,32) top5-ADE top1-FDE top5-FDE top1-DTW top5-DTW (표 1)
T1 = {
    "Gemini 3.1 Pro": [0.190, 0.274, 0.305, 0.161, 0.232, 0.253, 0.311, 0.425, 0.424, 0.254, 0.321, 0.311, 0.183, 0.258, 0.284, 0.152, 0.208, 0.224],
    "Gemini 3 Flash": [0.187, 0.271, 0.299, 0.158, 0.231, 0.254, 0.312, 0.414, 0.405, 0.252, 0.329, 0.316, 0.183, 0.260, 0.281, 0.150, 0.211, 0.227],
    "GPT-5.5": [0.199, 0.281, 0.307, 0.178, 0.249, 0.272, 0.329, 0.411, 0.404, 0.284, 0.344, 0.329, 0.196, 0.274, 0.299, 0.173, 0.238, 0.259],
    "Track2Act": [0.209, 0.311, 0.369, 0.190, 0.262, 0.293, 0.350, 0.493, 0.555, 0.287, 0.351, 0.346, 0.206, 0.303, 0.358, 0.181, 0.245, 0.270],
    "Hamster": [0.202, 0.276, 0.297, 0.178, 0.239, 0.256, 0.326, 0.400, 0.411, 0.274, 0.320, 0.330, 0.197, 0.261, 0.277, 0.170, 0.220, 0.233],
    "mu0_2D": [0.202, 0.279, 0.315, 0.124, 0.188, 0.227, 0.322, 0.410, 0.447, 0.186, 0.261, 0.284, 0.184, 0.254, 0.296, 0.114, 0.171, 0.211],
    "3DFlowAction": [0.615, 0.692, 0.716, 0.531, 0.605, 0.630, 0.753, 0.819, 0.818, 0.648, 0.714, 0.712, 0.614, 0.688, 0.711, 0.529, 0.600, 0.623],
    "Dream2Flow": [0.354, 0.451, 0.505, 0.201, 0.286, 0.336, 0.497, 0.616, 0.660, 0.287, 0.378, 0.403, 0.352, 0.449, 0.500, 0.198, 0.281, 0.329],
    "TraceGen": [0.327, 0.416, 0.464, 0.208, 0.276, 0.325, 0.478, 0.548, 0.642, 0.267, 0.329, 0.370, 0.298, 0.375, 0.413, 0.204, 0.262, 0.299],
    "mu0_3D": [0.209, 0.288, 0.325, 0.132, 0.199, 0.239, 0.331, 0.425, 0.464, 0.200, 0.278, 0.305, 0.191, 0.263, 0.308, 0.127, 0.187, 0.223],
}
T1_TIME = {"Gemini 3.1 Pro": 78, "Gemini 3 Flash": 62, "GPT-5.5": 38, "Track2Act": 0.85, "Hamster": 14.4, "mu0_2D": 0.29,
           "3DFlowAction": 3.38, "Dream2Flow": 106.8, "TraceGen": 1.20, "mu0_3D": 0.29}
# 표 4 (Fréchet): top1-FD(8,16,32), top5-FD(8,16,32)
T4 = {
    "Gemini 3.1 Pro": [0.324, 0.467, 0.505, 0.269, 0.385, 0.416],
    "Gemini 3 Flash": [0.324, 0.467, 0.504, 0.266, 0.387, 0.417],
    "GPT-5.5": [0.342, 0.476, 0.511, 0.299, 0.415, 0.449],
    "Track2Act": [0.363, 0.543, 0.631, 0.304, 0.420, 0.451],
    "Hamster": [0.339, 0.462, 0.505, 0.291, 0.390, 0.429],
    "mu0_2D": [0.314, 0.446, 0.517, 0.200, 0.306, 0.370],
    "3DFlowAction": [0.765, 0.843, 0.866, 0.664, 0.747, 0.772],
    "Dream2Flow": [0.547, 0.710, 0.787, 0.325, 0.464, 0.530],
    "TraceGen": [0.450, 0.560, 0.642, 0.291, 0.395, 0.457],
    "mu0_3D": [0.329, 0.455, 0.527, 0.210, 0.319, 0.384],
}
GROUPS = {"2D": ["Gemini 3.1 Pro", "Gemini 3 Flash", "GPT-5.5", "Track2Act", "Hamster", "mu0_2D"],
          "3D": ["3DFlowAction", "Dream2Flow", "TraceGen", "mu0_3D"]}
METRICS = ["top1-ADE", "top5-ADE", "top1-FDE", "top5-FDE", "top1-DTW", "top5-DTW"]
HOR = [8, 16, 32]
v6 = {"best_counts": {}, "not_best_cells": {}}
for g, ms in GROUPS.items():
    mu = ms[-1]
    best, not_best = 0, []
    for mi, mname in enumerate(METRICS):
        for hi, h in enumerate(HOR):
            col = mi * 3 + hi
            vals = {m: T1[m][col] for m in ms}
            m_best = min(vals, key=vals.get)
            if m_best == mu:
                best += 1
            else:
                not_best.append({"metric": mname, "T": h, "best": m_best, "best_val": vals[m_best], "mu0": vals[mu]})
    v6["best_counts"][g] = f"{best}/18"
    v6["not_best_cells"][g] = not_best
    top5_cells = [c for c in not_best if c["metric"].startswith("top5")]
    v6.setdefault("top5_not_best", {})[g] = len(top5_cells)
    v6.setdefault("top1_not_best", {})[g] = len([c for c in not_best if c["metric"].startswith("top1")])
R["V6_table1_best_cells"] = v6

# V7 top5/top1 비율(표본 다양성): 방법별 ADE, FDE, DTW
ratio = {}
for m in METHODS:
    row = T1[m]
    r = {}
    for name, c1, c5 in [("ADE", 0, 3), ("FDE", 6, 9), ("DTW", 12, 15)]:
        r[name] = [round(row[c5 + i] / row[c1 + i], 3) for i in range(3)]
    ratio[m] = r
R["V7_top5_over_top1"] = ratio

# V8 표 4 vs 표 1: 불변식(top5 <= top1, DTW <= ADE, Fréchet >= 마지막 스텝 오차 FDE)
inv = {"top5_gt_top1": [], "dtw_gt_ade": [], "frechet_lt_fde": []}
for m in METHODS:
    a, f = T1[m], T4[m]
    for k, (n1, n5) in enumerate([(0, 3), (6, 9), (12, 15)]):
        for i in range(3):
            if a[n5 + i] > a[n1 + i]:
                inv["top5_gt_top1"].append([m, METRICS[2 * k], HOR[i]])
    for i in range(6):
        if a[12 + i] > a[i]:
            inv["dtw_gt_ade"].append([m, i, a[12 + i], a[i]])
    for i in range(6):
        fde = a[6 + i]
        fd = f[i]
        if fd < fde:
            inv["frechet_lt_fde"].append([m, "top1" if i < 3 else "top5", HOR[i % 3], fde, fd])
    for i in range(3):
        if f[3 + i] > f[i]:
            inv["top5_gt_top1"].append([m, "FD", HOR[i]])
R["V8_invariants"] = inv

# V9 표 4 최고 셀과 "strongest overall FD" 서술
v9 = {}
for g, ms in GROUPS.items():
    mu = ms[-1]
    cells = []
    for ci, name in enumerate(["top1-FD", "top5-FD"]):
        for hi, h in enumerate(HOR):
            col = ci * 3 + hi
            vals = {m: T4[m][col] for m in ms}
            b = min(vals, key=vals.get)
            cells.append({"metric": name, "T": h, "best": b, "best_val": vals[b], "mu0": vals[mu]})
    v9[g] = [c for c in cells if c["best"] != mu]
R["V9_table4_not_best"] = v9

# V10 지연 비율, 파라미터 비율
R["V10_latency"] = {"track2act_over_mu0": 0.85 / 0.29, "claimed": 2.9,
                    "mu0_ae_over_pi05_ms": 200.8 / 173.0, "claimed_pct": 16,
                    "mean_in_range": 195.9 <= 200.8 <= 211.2 and 171.1 <= 173.0 <= 178.9}
R["V10_params"] = {"pi0_total": 3.45 + 0.578, "pi05_total": 3.45 + 0.6934, "mu0_total": 2.59 + 0.457,
                   "reported": {"pi0": 4.03, "pi05": 4.14, "mu0": 3.04},
                   "head_sum": 436.4 + 20.6, "pretrain_scale_ratio": 903 / 13.0, "claimed_ratio": 69.5}

# V11 부록 표 3 (사전학습 데이터), A.6, A.7
eps = [51.8, 3.4, 123, 25.4]
frames = [4.1, 2.7, 3.7, 2.5]
store = [4, 2.7, 8, 1]
human_f, robot_f = frames[0] + frames[1], frames[2] + frames[3]
R["V11_table3_sums"] = {
    "episodes_K": sum(eps), "frames_M": sum(frames), "storage_TB": sum(store),
    "human_frac": human_f / sum(frames), "robot_frac": robot_f / sum(frames),
    "days_paper_fps(human5,robot10)": [human_f * 1e6 / 5 / 86400, robot_f * 1e6 / 10 / 86400],
    "days_launcher_preset(egodex10,egoverse10,agibot5,droid5)": [
        (4.1e6 / 10 + 2.7e6 / 10) / 86400, (3.7e6 / 5 + 2.5e6 / 5) / 86400],
    "frames_per_episode": {"EgoDex": 4.1e6 / 51.8e3, "EgoVerse": 2.7e6 / 3.4e3, "Agibot": 3.7e6 / 123e3, "DROID": 2.5e6 / 25.4e3},
    "A6_gpu_hours_range(13M frames, 2.4-3.3 fps)": [13e6 / 3.3 / 3600, 13e6 / 2.4 / 3600],
    "A6_claimed_hours": 1200,
}

# V12 평가 세트 D.2
R["V12_eval_set"] = {"slots": 30 * 4 + 9, "claimed": 129, "bimanual_28pct_slots": round(0.28 * 129, 1),
                     "integer_slots_for_28pct": [k for k in range(30, 42) if abs(k / 129 - 0.28) < 0.005]}

# V13 표 9, 11, 12, 10, E.4
T9 = {"full": [0.127, 0.187, 0.223], "no_bspline": [0.156, 0.222, 0.258], "no_dino": [0.139, 0.193, 0.230],
      "no_rig": [0.138, 0.193, 0.227], "depth+hist": [0.107, 0.160, 0.203], "no_depth": [0.112, 0.168, 0.207],
      "no_hist": [0.126, 0.183, 0.224], "no_depth_no_hist": [0.127, 0.187, 0.223]}
T11 = {"342M": [0.143, 0.205, 0.240], "568M": [0.136, 0.191, 0.227], "2.59B": [0.127, 0.187, 0.223],
       "5%": [0.134, 0.200, 0.235], "20%": [0.138, 0.195, 0.227], "100%": [0.127, 0.187, 0.223]}
v13 = {
    "table9_full_equals_no_depth_no_hist": T9["full"] == T9["no_depth_no_hist"],
    "table9_full_equals_table1_3D_top5DTW": T9["full"] == T1["mu0_3D"][15:18],
    "table9_full_vs_table1_2D_top5DTW": T1["mu0_2D"][15:18],
    "model_scaling_monotone": all(T11["342M"][i] > T11["568M"][i] > T11["2.59B"][i] for i in range(3)),
    "data_scaling_monotone": [T11["5%"][i] >= T11["20%"][i] >= T11["100%"][i] for i in range(3)],
    "data_scaling_5_to_100_gain": [round(T11["5%"][i] - T11["100%"][i], 3) for i in range(3)],
    "model_scaling_342M_to_2.59B_gain": [round(T11["342M"][i] - T11["2.59B"][i], 3) for i in range(3)],
    "depth_gain_with_hist": [round(T9["no_depth"][i] - T9["depth+hist"][i], 3) for i in range(3)],
    "hist_gain_no_depth": [round(T9["no_depth_no_hist"][i] - T9["no_depth"][i], 3) for i in range(3)],
    "bspline_gain": [round(T9["no_bspline"][i] - T9["full"][i], 3) for i in range(3)],
    "rigidity_gain": [round(T9["no_rig"][i] - T9["full"][i], 3) for i in range(3)],
    "dino_gain": [round(T9["no_dino"][i] - T9["full"][i], 3) for i in range(3)],
}
R["V13_ablation_scaling"] = v13

# V14 표 12, 10, E.4 정수 가능성: 8 과제 x 50 = 400회 (한 시드), 두 시드면 800회
def feasible(pct, n):
    k = pct * n / 100
    return abs(k - round(k)) < 1e-9, k
feas = {}
for name, pct in [("10.675 (v1 표 8)", 10.675), ("10.625 (v3 표 12)", 10.625), ("25.625", 25.625), ("28.25", 28.25),
                   ("30.25", 30.25), ("29.5", 29.5), ("26.0", 26.0), ("25.0", 25.0), ("27.5", 27.5)]:
    feas[name] = {"n400": feasible(pct, 400), "n800": feasible(pct, 800), "n1200": feasible(pct, 1200)}
R["V14_integer_feasibility"] = feas
R["V14_table12_table10_stats"] = {
    "200M head: trace 25.625 vs no trace 10.625 (n=800 가정)": fisher(205, 800, 85, 800),
    "400M head: trace 30.25 vs no trace 28.25 (n=800 가정)": fisher(242, 800, 226, 800),
    "400M head: trace 30.25 vs no trace 28.25 (n=400 가정)": fisher(121, 400, 113, 400),
    "learned gate 30.25 vs fixed 0.5 26.0 (n=400)": fisher(121, 400, 104, 400),
    "learned gate 30.25 vs fixed 0.25 25.0 (n=400)": fisher(121, 400, 100, 400),
    "full decoded trace 29.5 vs intermediate 30.25 (n=400)": fisher(118, 400, 121, 400),
    "trace noise 27.5 vs 30.25 (n=400)": fisher(110, 400, 121, 400),
    "mu0+AE 30.25 vs w/o Trace(400M) 28.25 (n=400)": fisher(121, 400, 113, 400),
}
R["V14_gate_note"] = {"sigmoid(0)": 0.5, "learned_reported": [0.483, 0.481, 0.499, 0.500, 0.499],
                      "max_abs_shift_from_init": max(abs(x - 0.5) for x in [0.483, 0.481, 0.499, 0.500, 0.499])}

# V15 표 7 (장기 과제, 50 롤아웃): 짝수성과 본문 문장
T7 = {"pi0": [40, 2, 22, 2, 28, 8, 0], "pi05": [44, 6, 32, 6, 32, 8, 0], "mu0": [44, 4, 26, 4, 22, 6, 0]}
ge_pi0 = sum(m >= p for m, p in zip(T7["mu0"], T7["pi0"]))
lt_pi05 = sum(m < p for m, p in zip(T7["mu0"], T7["pi05"]))
eq_pi05 = sum(m == p for m, p in zip(T7["mu0"], T7["pi05"]))
R["V15_table7"] = {"all_even": all(x % 2 == 0 for v in T7.values() for x in v),
                   "mu0_ge_pi0_subtasks": ge_pi0, "mu0_lt_pi05_subtasks": lt_pi05, "mu0_eq_pi05_subtasks": eq_pi05}

# V16 E.4 상대 증가율, 표 10
R["V16_robustness"] = {"noise_drop_pp": 30.25 - 27.5, "noise_drop_rel_pct": 100 * (30.25 - 27.5) / 30.25}

# V17 검정력: 두 비율 비교에 필요한 방법당 시행 수 (양측 5%, 검정력 80%)
def n_per_arm(p1, p2, alpha=0.05, power=0.8):
    za, zb = stats.norm.isf(alpha / 2), stats.norm.isf(1 - power)
    pb = (p1 + p2) / 2
    num = (za * math.sqrt(2 * pb * (1 - pb)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)
R["V17_power"] = {
    "sim mu0 30.25 vs pi0 25.25": n_per_arm(0.3025, 0.2525),
    "sim mu0 30.25 vs TraceGen 23": n_per_arm(0.3025, 0.23),
    "sim mu0 30.25 vs pi0.5 42": n_per_arm(0.3025, 0.42),
    "real mu0 91.7 vs pi0.5 80": n_per_arm(0.9167, 0.80),
    "real mu0 91.7 vs TraceGen 81.7": n_per_arm(0.9167, 0.8167),
    "real mu0 91.7 vs pi0 71.7": n_per_arm(0.9167, 0.7167),
    "sim_trials_per_arm_reported": 400, "real_trials_per_arm_reported": 60,
    "sim_se_pp_at_30": 100 * math.sqrt(0.3 * 0.7 / 400),
}

# V18 t=1 노이즈 입력 부분 디노이징 설정, 효과적 이력 드롭 확률 (코드 상수와 논문 C.1)
R["V18_history_dropout"] = {"p_full": 0.5, "p_kp": 0.3, "effective_kp_drop": 0.5 + 0.5 * 0.3,
                            "p_depth_omitted": 0.7,
                            "train_share_no_depth_and_no_history_for_a_kp": 0.7 * (0.5 + 0.5 * 0.3)}

# V19 사전학습 예산: B.1 (2e5 스텝, 유효 배치 24) 대 13M 프레임
R["V19_pretraining_budget"] = {"samples_seen": 2e5 * 24, "frames": 13.0e6, "passes_over_frames": 2e5 * 24 / 13.0e6,
                               "passes_over_5pct": 2e5 * 24 / (0.05 * 13.0e6)}

OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print("wrote", OUT)
for key in ["V1_table2_means", "V3_real_means", "V5_intro_ratios", "V6_table1_best_cells", "V8_invariants",
            "V14_table12_table10_stats", "V17_power"]:
    print(key, json.dumps(R[key], ensure_ascii=False, default=float)[:1500])
