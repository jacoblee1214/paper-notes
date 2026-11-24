"""MolmoB0T (arXiv:2603.16861 v3) 보고 수치 재계산.

논문의 표(표 1, 3, 8, 9, 11, 12, 13, 14)와 본문 수치를 옮겨 적고, 평균, 정수 가능성, 구간, 검정,
표 사이 일관성을 다시 계산한다. 그림 5와 8의 값은 그림에서 읽은 근사값이다(결과에 'figure_read' 표시).
시행 수가 논문에 없는 곳은 가정한 시행 수를 결과에 함께 적는다.

실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)  → analysis/recompute_results.json
"""
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}
Z = 1.959964


def wilson(k, n, z=Z):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def clopper(k, n, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def fisher(k1, n1, k2, n2):
    return stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]


def ztest(p1, n1, p2, n2):
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return 2 * (1 - stats.norm.cdf(abs(z)))


# ---------------------------------------------------------------- V1: 표 8 (실로봇 DROID, 40 과제 x 3회)
T8 = {
    "Workroom": {
        "MolmoBot(H=1)": [3, 2, 3, 3, 3, 3, 3, 1, 3, 3], "MolmoBot-Img": [3, 3, 3, 3, 3, 2, 2, 0, 3, 1],
        "MolmoBot-Pi0": [1, 0, 3, 3, 3, 3, 2, 1, 0, 2], "pi0.5-DROID": [2, 2, 2, 0, 1, 1, 0, 0, 0, 0],
        "pi0-DROID": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]},
    "Kitchen": {
        "MolmoBot(H=1)": [1, 2, 3, 3, 3, 3, 3, 3, 0, 0], "MolmoBot-Img": [3, 3, 2, 3, 3, 3, 3, 2, 3, 1],
        "MolmoBot-Pi0": [2, 3, 2, 0, 3, 1, 3, 2, 0, 0], "pi0.5-DROID": [3, 0, 2, 1, 3, 2, 3, 1, 2, 2],
        "pi0-DROID": [0, 1, 0, 0, 1, 0, 3, 1, 0, 0]},
    "Bedroom": {
        "MolmoBot(H=1)": [3, 3, 3, 0, 3, 2, 3, 3, 3, 3], "MolmoBot-Img": [0, 0, 1, 2, 3, 3, 3, 2, 3, 3],
        "MolmoBot-Pi0": [2, 0, 2, 0, 0, 0, 1, 0, 0, 2], "pi0.5-DROID": [0, 0, 0, 0, 1, 0, 0, 0, 2, 0],
        "pi0-DROID": [0] * 10},
    "Office": {
        "MolmoBot(H=1)": [2, 3, 1, 2, 2, 2, 3, 2, 3, 1], "MolmoBot-Img": [2, 1, 0, 1, 3, 1, 3, 2, 3, 2],
        "MolmoBot-Pi0": [1, 3, 0, 0, 0, 1, 3, 2, 3, 2], "pi0.5-DROID": [2, 3, 1, 1, 1, 1, 2, 1, 3, 2],
        "pi0-DROID": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]},
}
T8_PRINTED_AVG = {  # 표 8의 'Avg' 열(정수 %)
    "Workroom": {"MolmoBot(H=1)": 90, "MolmoBot-Img": 77, "MolmoBot-Pi0": 60, "pi0.5-DROID": 27, "pi0-DROID": 3},
    "Kitchen": {"MolmoBot(H=1)": 70, "MolmoBot-Img": 87, "MolmoBot-Pi0": 53, "pi0.5-DROID": 63, "pi0-DROID": 20},
    "Bedroom": {"MolmoBot(H=1)": 87, "MolmoBot-Img": 67, "MolmoBot-Pi0": 23, "pi0.5-DROID": 10, "pi0-DROID": 0},
    "Office": {"MolmoBot(H=1)": 70, "MolmoBot-Img": 60, "MolmoBot-Pi0": 50, "pi0.5-DROID": 57, "pi0-DROID": 13},
}
POL = ["MolmoBot(H=1)", "MolmoBot-Img", "MolmoBot-Pi0", "pi0.5-DROID", "pi0-DROID"]
v1 = {"env_avg_recomputed_pct": {}, "env_avg_mismatch": [], "total": {}}
for env, d in T8.items():
    v1["env_avg_recomputed_pct"][env] = {}
    for pol in POL:
        k = sum(d[pol])
        v1["env_avg_recomputed_pct"][env][pol] = [k, round(100 * k / 30, 2)]
        if round(100 * k / 30) != T8_PRINTED_AVG[env][pol]:
            v1["env_avg_mismatch"].append([env, pol, k, T8_PRINTED_AVG[env][pol]])
for pol in POL:
    k = sum(sum(T8[e][pol]) for e in T8)
    lo, hi = wilson(k, 120)
    v1["total"][pol] = {"k": k, "n": 120, "pct": round(100 * k / 120, 2), "wilson95": [round(100 * lo, 1), round(100 * hi, 1)]}
RES["V1_table8_real_world"] = v1

# V2: 정책 쌍 검정 (총 120회, 비쌍 Fisher) + 과제 단위 쌍체 부호순위(40 과제, 과제당 0-3)
tot = {pol: [x for e in T8 for x in T8[e][pol]] for pol in POL}
pairs = {}
for a, b in [("MolmoBot(H=1)", "pi0.5-DROID"), ("MolmoBot(H=1)", "MolmoBot-Img"), ("MolmoBot-Img", "MolmoBot-Pi0"),
             ("MolmoBot-Pi0", "pi0.5-DROID"), ("MolmoBot-Pi0", "pi0-DROID"), ("MolmoBot-Img", "pi0.5-DROID"),
             ("pi0.5-DROID", "pi0-DROID")]:
    ka, kb = sum(tot[a]), sum(tot[b])
    diff = np.array(tot[a]) - np.array(tot[b])
    try:
        w = stats.wilcoxon(diff, zero_method="wilcox", method="approx").pvalue if np.any(diff != 0) else 1.0
    except ValueError:
        w = float("nan")
    pairs[f"{a} vs {b}"] = {"k": [ka, kb], "fisher_p": fisher(ka, 120, kb, 120), "task_paired_wilcoxon_p": w,
                            "tasks_a_better": int((diff > 0).sum()), "tasks_b_better": int((diff < 0).sum()),
                            "tasks_tie": int((diff == 0).sum())}
RES["V2_table8_pairwise"] = pairs

# V3: 층화 부트스트랩(과제 40개 각각 3회 재표집) 95% 반폭: 그림 4와 표 12의 구간 방법 확인
rng = np.random.default_rng(0)


def strat_boot(cells, n_boot=20000, trials=3):
    cells = np.array(cells)
    p = cells / trials
    draws = rng.binomial(trials, p[None, :], size=(n_boot, len(cells))).sum(1) / (trials * len(cells))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return 100 * (hi - lo) / 2


v3 = {}
for pol in POL:
    v3[pol] = {"total_40_tasks_halfwidth_pct": round(strat_boot(tot[pol]), 1),
               "wilson_120_halfwidth_pct": round(100 * (wilson(sum(tot[pol]), 120)[1] - wilson(sum(tot[pol]), 120)[0]) / 2, 1)}
for pol in ["MolmoBot-Img", "MolmoBot(H=1)"]:
    v3[pol]["workroom_10_tasks_halfwidth_pct"] = round(strat_boot(T8["Workroom"][pol]), 1)
    v3[pol]["workroom_wilson_30_halfwidth_pct"] = round(100 * (wilson(sum(T8["Workroom"][pol]), 30)[1] - wilson(sum(T8["Workroom"][pol]), 30)[0]) / 2, 1)
v3["table12_reported_halfwidth_pct"] = {"MolmoBot-Img": 10.0, "MolmoBot-Img+": 13.3, "MolmoBot(H=1)": 6.7, "MolmoBot(H=2)": 13.3}
RES["V3_stratified_bootstrap"] = v3

# V4: 표 11의 켓 열과 표 8 켓 행 비교, 켓 대 비켓
T11_KITCHEN = {"pi0-DROID": 20.0, "pi0.5-DROID": 63.3, "MolmoBot-Pi0": 53.3, "MolmoBot-SPOC": 36.6, "MolmoBot-Img": 86.6,
               "MolmoBot(H=1)": 70.0, "MolmoBot(H=2)": 73.3}
T11_ALL = {"pi0-DROID": 9.2, "pi0.5-DROID": 39.2, "MolmoBot-Pi0": 46.7, "MolmoBot-Img": 72.5, "MolmoBot(H=1)": 79.2, "MolmoBot(H=2)": 75.0}
v4 = {"kitchen_col_vs_table8_row": {}, "all_scene_vs_table8_total": {}, "kitchen_vs_rest": {}}
for pol in POL:
    kk = sum(T8["Kitchen"][pol])
    v4["kitchen_col_vs_table8_row"][pol] = {"table11": T11_KITCHEN[pol], "table8_k_of_30": kk, "table8_pct": round(100 * kk / 30, 2)}
    v4["all_scene_vs_table8_total"][pol] = {"table11": T11_ALL[pol], "table8_pct": v1["total"][pol]["pct"]}
    rest = sum(sum(T8[e][pol]) for e in T8 if e != "Kitchen")
    v4["kitchen_vs_rest"][pol] = {"kitchen": [kk, 30], "rest": [rest, 90], "rest_pct": round(100 * rest / 90, 1),
                                  "fisher_p": fisher(kk, 30, rest, 90)}
# pi0.5 의 'Real' 31.3 (v1, v2 표 6의 Real 열, v3 본문 C.3)
env_means = {e: round(100 * sum(T8[e]["pi0.5-DROID"]) / 30) for e in T8}
v4["pi0.5_31.3_check"] = {"env_avg_rounded_pct": env_means,
                          "mean_of_three_non_kitchen_env_avgs": round(np.mean([env_means[e] for e in T8 if e != "Kitchen"]), 2),
                          "mean_of_four_env_avgs": round(np.mean(list(env_means.values())), 2)}
RES["V4_table11_vs_table8"] = v4

# ---------------------------------------------------------------- V5: 표 1 평균과 구간
T1_COLS = ["Pick v1", "Pick v2", "Pick v2 (F)", "Pick v2 RC (F)", "PnP Into (F)", "PnP Next-To (F)", "PnP Color"]
# 각 칸: (oracle, ±, final, ±) ; Pick 칸은 final 없음
T1 = {
    "pi0.5": [(18.1, 2.4), (6.4, 1.5), (7.0, 1.6), (8.0, 1.9), (11.7, 2.1, 7.6, 1.7), (8.2, 2.2, 6.2, 1.9), (10.4, 1.9, 6.7, 1.6)],
    "pi0.5-Finetune": [(48.0, 3.1), (28.3, 2.8), (25.8, 2.7), (29.7, 2.9), (43.5, 6.0, 37.4, 5.8), (28.4, 3.2, 14.7, 2.5), (48.3, 3.1, 38.9, 3.0)],
    "StereoVLA": [(6.6, 2.6), (4.3, 1.5), (1.1, 1.0), None, (0.0,), None, (0.0,)],
    "LAP-VLA": [(19.4, 2.4), (2.4, 1.0), (3.1, 1.1), (2.7, 1.0), (3.81, 1.5, 1.59, 1.0), (6.48, 2.8, 3.41, 2.1), (3.1, 1.1, 1.5, 0.8)],
    "X-VLA": [(3.3, 1.0), (0.5, 0.5), (0.7, 0.5), (0.8, 0.5), (0.1, 0.2, 0.1, 0.2), (1.9, 0.9, 1.0, 0.7), (0.9, 0.6, 0.5, 0.5)],
    "MolmoBot-Pi0": [(66.2, 2.9), (35.7, 3.0), (33.3, 2.9), (39.8, 3.1), (44.7, 3.1, 38.2, 3.1), (24.7, 3.2, 13.3, 2.5), (46.2, 3.1, 40.0, 3.1)],
    "MolmoBot-Img": [(92.2, 1.7), (63.5, 3.0), (61.4, 3.0), (62.1, 3.0), (63.0, 3.0, 55.0, 3.1), (21.0, 2.5, 16.4, 2.3), (67.8, 2.9, 60.3, 3.0)],
    "MolmoBot(H=1)": [(93.5, 1.5), (66.8, 2.9), (64.0, 3.0), (63.7, 3.0), (66.4, 2.9, 57.7, 3.0), (26.4, 2.7, 20.2, 2.5), (67.8, 2.9, 60.0, 3.0)],
    "MolmoBot(H=2)": [(91.3, 1.8), (63.8, 3.0), (59.0, 3.0), (62.7, 3.0), (65.4, 2.9, 55.6, 3.1), (28.3, 2.8, 22.6, 2.6), (66.1, 2.9, 57.3, 3.1)],
}
T1_PRINTED_AVG = {"pi0.5": 10.0, "pi0.5-Finetune": 36.0, "StereoVLA": 1.7, "LAP-VLA": 4.8, "X-VLA": 1.2, "MolmoBot-Pi0": 41.5,
                  "MolmoBot-Img": 61.6, "MolmoBot(H=1)": 64.1, "MolmoBot(H=2)": 62.4}
v5 = {"avg": {}, "implied_n": {}}
for m, cells in T1.items():
    orc = [0.0 if c is None else c[0] for c in cells]
    fin = [0.0 if c is None else (c[2] if len(c) > 2 else c[0]) for c in cells]
    a_o, a_f = sum(orc) / 7, sum(fin) / 7
    v5["avg"][m] = {"printed": T1_PRINTED_AVG[m], "oracle_basis": round(a_o, 3), "final_basis": round(a_f, 3),
                    "matches": "oracle" if abs(a_o - T1_PRINTED_AVG[m]) <= 0.06 else ("final" if abs(a_f - T1_PRINTED_AVG[m]) <= 0.1 else "none")}
# 반폭에서 거꾸로 추정한 시행 수 n = z^2 p(1-p)/h^2 (정규 근사)
for m, cells in T1.items():
    v5["implied_n"][m] = {}
    for col, c in zip(T1_COLS, cells):
        if c is None or len(c) == 1:
            continue
        vals = [(c[0], c[1])] + ([(c[2], c[3])] if len(c) > 2 else [])
        out = []
        for i, (pct, h) in enumerate(vals):
            p = pct / 100
            n = Z * Z * p * (1 - p) / (h / 100) ** 2
            out.append(round(n))
        v5["implied_n"][m][col] = out
RES["V5_table1_avg_and_implied_n"] = v5

# V6: 시행 수 후보별 정수 가능성 (소수 둘째 자리 값)
def feasible_n(pct, ns, dec):
    return [n for n in ns if abs(round(100 * round(pct * n / 100) / n, dec) - pct) < 1e-9]


v6 = {}
for pct in (3.81, 1.59, 6.48, 3.41):
    v6[str(pct)] = {"n=1000": feasible_n(pct, [1000], 2), "n=200": feasible_n(pct, [200], 2),
                    "n_in_[50,1000]_count": len(feasible_n(pct, range(50, 1001), 2)),
                    "n_in_[580,660]": feasible_n(pct, range(580, 661), 2), "n_in_[270,320]": feasible_n(pct, range(270, 321), 2)}
# 1자리 값이 n=200에서 가능한지 (표 10의 200 에피소드)
v6["n200_feasible_pick_v2_row_values"] = {m: [c[0] for c in cells[1:4] if c is not None and abs(c[0] * 2 - round(c[0] * 2)) > 1e-9] for m, cells in T1.items()}
# StereoVLA 필터 후 92, 91 에피소드
v6["StereoVLA_filtered"] = {"Pick v2 4.3 at n=92": [k for k in range(0, 10) if round(100 * k / 92, 1) == 4.3],
                            "Pick v2 (F) 1.1 at n=91": [k for k in range(0, 10) if round(100 * k / 91, 1) == 1.1],
                            "reported_halfwidths": {"4.3": 1.5, "1.1": 1.0},
                            "wilson_halfwidth_n92_k4": round(100 * (wilson(4, 92)[1] - wilson(4, 92)[0]) / 2, 1),
                            "wilson_halfwidth_n91_k1": round(100 * (wilson(1, 91)[1] - wilson(1, 91)[0]) / 2, 1)}
def joint_n(vals, lo, hi, dec):
    return [n for n in range(lo, hi + 1) if all(any(round(100 * k / n, dec) == v for k in range(0, n + 1)) for v in vals)]


v6["joint_feasible_n_from_two_values"] = {
    "LAP-VLA PnP Into (3.81, 1.59), n in [550,700]": joint_n([3.81, 1.59], 550, 700, 2),
    "LAP-VLA PnP Next-To (6.48, 3.41), n in [250,340]": joint_n([6.48, 3.41], 250, 340, 2),
    "pi0.5-Finetune PnP Into (43.5, 37.4), n in [240,290]": joint_n([43.5, 37.4], 240, 290, 1)}
RES["V6_integer_feasibility"] = v6

# V7: 표 1 쌍 검정 (n=1000 가정, 비쌍 z)
def cellp(m, col, idx=0):
    c = T1[m][T1_COLS.index(col)]
    return c[idx] / 100


v7 = {}
for col in T1_COLS:
    for a, b in [("MolmoBot(H=1)", "MolmoBot-Img"), ("MolmoBot(H=2)", "MolmoBot-Img"), ("MolmoBot-Pi0", "pi0.5-Finetune"),
                 ("MolmoBot-Img", "MolmoBot-Pi0")]:
        v7[f"{col}: {a} vs {b}"] = {"diff_pp": round(100 * (cellp(a, col) - cellp(b, col)), 1),
                                    "p_assuming_n1000": ztest(cellp(a, col), 1000, cellp(b, col), 1000)}
RES["V7_table1_pairs_n1000_assumed"] = v7

# V8: 본문 수치 대조
RES["V8_text_vs_table"] = {
    "C.3 'pi0.5 zero-shot avg 10.1' vs table 10.0 (oracle-basis recomputed %.3f)" % (sum(c[0] for c in T1["pi0.5"]) / 7): True,
    "C.3 'MolmoBot-Pi0 41.8' vs table 41.5 (recomputed %.3f)" % (sum(c[0] for c in T1["MolmoBot-Pi0"]) / 7): True,
    "C.3 'Pick v2 RC: our models 40-66%' vs table row (min, max over MolmoBot rows)": [min(T1[m][3][0] for m in T1 if m.startswith("MolmoBot")),
                                                                                  max(T1[m][3][0] for m in T1 if m.startswith("MolmoBot"))],
    "C.3 'pi0.5 variants reach only 8-30%' vs rows": [min(T1[m][3][0] for m in ("pi0.5", "pi0.5-Finetune")), max(T1[m][3][0] for m in ("pi0.5", "pi0.5-Finetune"))],
    "C.3 'MolmoBot 63-67% oracle on PnP Into' vs rows (H=1,H=2,Img)": [T1[m][4][0] for m in ("MolmoBot-Img", "MolmoBot(H=1)", "MolmoBot(H=2)")],
    "abstract/intro MolmoBot-Pi0 46.5% vs Table 8 total": v1["total"]["MolmoBot-Pi0"]["pct"],
    "C.3 'pi0.5 31.3% real' vs Table 8 total": v1["total"]["pi0.5-DROID"]["pct"],
}

# ---------------------------------------------------------------- V9: 표 3 데이터 통계
T3 = [  # task, robot, episodes(k), avg_len(s), total_h, frames(M, v2 표 1)
    ("Door-open", "RBY1", 79.0, 19.6, 429, 15.4), ("Open", "RBY1", 46.6, 14.8, 192, 6.9),
    ("Pick", "RBY1", 62.3, 10.7, 184, 7.4), ("Pick", "Franka", 781.8, 4.8, 1042, 56.9),
    ("PnP", "RBY1", 14.8, 14.0, 58, 2.4), ("PnP", "Franka", 554.2, 17.1, 2638, 143.9),
    ("PnP Next-To", "Franka", 182.7, 20.1, 1022, 54.7), ("PnP Color", "Franka", 28.6, 17.4, 138, 7.5)]
v9 = {"rows": []}
for t, r, ep, L, h, fr in T3:
    v9["rows"].append({"task": t, "robot": r, "hours_from_ep_len": round(ep * 1000 * L / 3600, 1), "hours_printed": h,
                       "implied_fps": round(fr * 1e6 / (ep * 1000 * L), 1)})
ep_sum = sum(x[2] for x in T3)
h_sum = sum(x[4] for x in T3)
fr_sum = sum(x[5] for x in T3)
v9["sum_episodes_k"] = round(ep_sum, 1)
v9["sum_hours"] = h_sum
v9["sum_frames_M_v2"] = round(fr_sum, 1)
v9["avg_len_s_from_sums"] = round(h_sum * 3600 / (ep_sum * 1000), 2)
v9["printed_total"] = {"episodes": "1.7M", "hours": 5704, "avg_len_s": 11.7, "frames_M_v2": 295.2}
v9["hf_dataset_storage_TB"] = round(8874478499808 / 1e12, 2)
v9["paper_says_TB_of_robot_experience"] = 12
# 처리량
eps_per_gpu_h_avg, eps_per_gpu_h_peak, gpus, avg_len = 260, 660, 100, 11.7
v9["throughput"] = {
    "robot_hours_per_wall_hour_avg": round(gpus * eps_per_gpu_h_avg * avg_len / 3600, 1),
    "robot_hours_per_wall_hour_peak": round(gpus * eps_per_gpu_h_peak * avg_len / 3600, 1),
    "paper_says": ">88",
    "eps_per_gpu_h_implied_by_88": round(88 * 3600 / (gpus * avg_len), 1),
    "gpu_hours_if_260": round(ep_sum * 1000 / 260),
    "gpu_hours_if_660": round(ep_sum * 1000 / 660),
    "paper_gpu_hours": 6500,
    "episodes_per_gpu_h_implied_by_6500": round(ep_sum * 1000 / 6500, 1),
    "v2_text_660_per_gpu_h_gives_robot_h_per_wall_h": round(gpus * 660 * avg_len / 3600, 1),
    "v1": {"eps_per_gpu_h": 1024, "robot_h_per_wall_h_printed": 130, "gpu_hours": 4500, "episodes_M": 1.8,
           "robot_h_per_wall_h_from_1024": round(gpus * 1024 * 11.8 / 3600, 1), "eps_per_gpu_h_implied_by_130": round(130 * 3600 / (gpus * 11.8), 1),
           "episodes_per_gpu_h_from_1.8M_over_4500": round(1.8e6 / 4500, 1)},
    "speedup_vs_human_realtime_1_over_3": {"v3_88_over_100/3": round(88 / (100 / 3), 2), "paper": 2.6, "v1_130_over_100/3": round(130 / (100 / 3), 2), "v1_paper": 4},
    "cost_per_traj_0.01-0.02_to_per_robot_hour": [round(c * 3600 / avg_len, 2) for c in (0.01, 0.02)],
    "paper_cost_per_robot_hour": [2, 5],
    "gpu_hour_price_implied_by_0.01-0.02_at_260": [round(c * 260, 2) for c in (0.01, 0.02)],
}
v9["v1_vs_v2_table"] = {"v1": {"episodes_k": [99.3, 46.6, 62.7, 781.8, 14.8, 554.2, 182.7, 28.6], "sum_k": round(99.3 + 46.6 + 62.7 + 781.8 + 14.8 + 554.2 + 182.7 + 28.6, 1),
                               "hours_printed_total": 5817},
                        "v3_door_open_drop_pct": round(100 * (79.0 - 99.3) / 99.3, 1)}
RES["V9_table3_dataset_and_throughput"] = v9

# ---------------------------------------------------------------- V10: 표 12, 표 11 절삭, 표 13, 14
t12 = {"MolmoBot-Img": 23, "MolmoBot-Img+": 25, "MolmoBot(H=1)": 27, "MolmoBot(H=2)": 23}
v10 = {"table12_k_of_30": t12, "pct": {m: round(100 * k / 30, 1) for m, k in t12.items()}, "pairs": {}}
for a, b in [("MolmoBot(H=1)", "MolmoBot-Img+"), ("MolmoBot(H=1)", "MolmoBot-Img"), ("MolmoBot-Img+", "MolmoBot-Img"), ("MolmoBot(H=1)", "MolmoBot(H=2)")]:
    v10["pairs"][f"{a} vs {b}"] = fisher(t12[a], 30, t12[b], 30)
v10["table11_truncation"] = {"86.6 vs 26/30": round(100 * 26 / 30, 2), "36.6 vs 11/30": round(100 * 11 / 30, 2)}
# 표 13, 14
t14 = {"MolmoBot-Img": ([0, 0, 20, 0, 0, 0, 0, 0, 10, 50], [16, 24, 40, 42, 42, 52, 2, 26, 78, 44]),
       "MolmoBot-Pi0": ([0, 0, 0, 0, 0, 2, 0, 4, 0, 2], [2, 2, 2, 14, 6, 6, 0, 2, 2, 2]),
       "pi0.5-DROID": ([0, 2, 0, 0, 0, 2, 0, 4, 0, 2], [2, 0, 0, 2, 4, 8, 2, 0, 6, 0]),
       "pi0-DROID": ([0] * 10, [0, 0, 0, 0, 0, 0, 0, 0, 0, 2])}
t14p = {"MolmoBot-Img": (8, 36.6), "MolmoBot-Pi0": (0.8, 3.8), "pi0.5-DROID": (1, 2.4), "pi0-DROID": (0, 0.2)}
v10["table14_avg"] = {m: {"std": np.mean(a), "nodist": np.mean(b), "printed": t14p[m]} for m, (a, b) in t14.items()}
v10["table14_all_even_multiples_of_2"] = all(x % 2 == 0 for a, b in t14.values() for x in a + b)
v10["table14_img_vs_pi0_nodist_n500_assumed"] = fisher(183, 500, 19, 500)
v10["table13_carrot_45.8_is_k_over_24"] = [k for k in range(0, 25) if round(100 * k / 24, 1) == 45.8]
RES["V10_table11_12_13_14"] = v10

# ---------------------------------------------------------------- V11: 그림 5와 8 (그림 판독)
fig5 = {  # 판독한 실제 성공률(%), n=30 가정
    "a_trajectories_real": {"10k": 22, "25k": 53, "50k": 80},
    "b_object_classes_real": {"10": 25, "25": 67, "50": 64, "100": 60},
    "c_environments_real": {"50": 70, "100": 70, "5k": 80, "50k": 63},
}
v11 = {"note": "그림에서 읽은 근사값(±3 pp). 시행 수는 5.3절의 6개 물체 x 5회 = 30 가정, C.6의 3 그룹 x 3 물체 x 5회 = 45도 병기", "tests": {}}
for tag, d in fig5.items():
    ks = list(d.items())
    for (n1, p1), (n2, p2) in combinations(ks, 2):
        out = {}
        for n in (30, 45):
            k1, k2 = round(p1 * n / 100), round(p2 * n / 100)
            out[f"n={n}"] = {"k": [k1, k2], "fisher_p": fisher(k1, n, k2, n)}
        v11["tests"][f"{tag}: {n1} vs {n2}"] = out
# 80% 검정력 최소 검출 차이 (두 비율, n=30, 45)
def min_detectable(n, p0, alpha=0.05, power=0.8):
    for d in np.arange(0.01, 0.8, 0.01):
        p1 = min(p0 + d, 0.999)
        se0 = math.sqrt(2 * ((p0 + p1) / 2) * (1 - (p0 + p1) / 2) / n)
        se1 = math.sqrt(p0 * (1 - p0) / n + p1 * (1 - p1) / n)
        zpow = (d - Z * se0) / se1
        if stats.norm.cdf(zpow) >= power:
            return round(100 * d, 0)
    return None


v11["min_detectable_diff_pp_80power"] = {f"n={n},p0={p}": min_detectable(n, p) for n in (30, 45) for p in (0.5, 0.7)}
fig8 = {"T=1": 67, "T=2": 63, "T=4": 76, "T=8": 63, "delta": 47, "absolute": 76}
v11["fig8_real_n30"] = {"delta_vs_absolute": fisher(round(0.47 * 30), 30, round(0.76 * 30), 30), "T4_vs_T8": fisher(round(0.76 * 30), 30, round(0.63 * 30), 30),
                        "k": {"delta": round(0.47 * 30), "absolute": round(0.76 * 30), "T4": round(0.76 * 30), "T8": round(0.63 * 30)}}
# 같은 설정의 점이 그림 사이에서 일치하는지: 그림 5(c) 50k 환경, 5(b) 100 클래스, 그림 8(a) T=8
v11["same_config_points_real"] = {"fig5b_100classes": 60, "fig5c_50k_env": 63, "fig8a_T8": 63}
v11["same_config_points_sim"] = {"fig5b_100classes": 37, "fig5c_50k_env": 38, "fig8a_T8": 38}
v11["fig5a_50k_vs_fig5c_5k_real"] = {"fig5a_50k": 80, "fig5c_5k_env": 80}
RES["V11_figures_5_8_read"] = v11

# ---------------------------------------------------------------- V12: 모의 대 실제 순위 상관
sim_pick_v1 = {"pi0-DROID": 13.5, "pi0.5-DROID": 18.1, "MolmoBot-Pi0": 66.2, "MolmoBot-Img": 92.2, "MolmoBot(H=1)": 93.5, "MolmoBot(H=2)": 91.3}
sim_avg = {"pi0.5-DROID": 10.0, "MolmoBot-Pi0": 41.5, "MolmoBot-Img": 61.6, "MolmoBot(H=1)": 64.1, "MolmoBot(H=2)": 62.4}
real_all = {"pi0-DROID": 9.2, "pi0.5-DROID": 39.2, "MolmoBot-Pi0": 46.7, "MolmoBot-Img": 72.5, "MolmoBot(H=1)": 79.2, "MolmoBot(H=2)": 75.0}
sim_kit = dict(sim_pick_v1)
sim_kit["MolmoBot-SPOC"] = 70.4
real_kit = dict(T11_KITCHEN)


def corr(sim, real):
    ks = [k for k in sim if k in real]
    x = [sim[k] for k in ks]
    y = [real[k] for k in ks]
    rho, p = stats.spearmanr(x, y)
    r, pr = stats.pearsonr(x, y)
    # 정확 순열 검정(스피어만)
    cnt, tot_ = 0, 0
    from itertools import permutations
    xr = stats.rankdata(x)
    for perm in permutations(range(len(ks))):
        yy = stats.rankdata(np.array(y)[list(perm)])
        tot_ += 1
        if abs(np.corrcoef(xr, yy)[0, 1]) >= abs(rho) - 1e-12:
            cnt += 1
    return {"n": len(ks), "spearman": round(rho, 3), "exact_perm_p": round(cnt / tot_, 4), "pearson": round(r, 3), "pearson_p": round(pr, 4), "models": ks}


RES["V12_sim_real_rank"] = {"pickv1_sim_vs_allscene_real": corr(sim_pick_v1, real_all), "avg_sim_vs_allscene_real": corr(sim_avg, real_all),
                            "pickv1_sim_vs_kitchen_real": corr(sim_kit, real_kit)}

# ---------------------------------------------------------------- V13: RB-Y1 표 2, 표 9
t2 = {"MB Multitask": [44.8, 22.5, 25.2, 70.2], "MB Door": [None, None, None, 77.7], "MB-SPOC Rigid": [10.5, 1.8, None, None],
      "MB-SPOC Artic.": [None, None, 21.8, 58.8]}
n_assumed = [2000, 1000, 2000, 2000]  # v1 본문: pick, open, door-open 2,000; pick-and-place 1,000 (v3에는 시행 수 없음)
v13 = {"assumed_n_v1_text": {"Pick": 2000, "PnP Into": 1000, "Open": 2000, "Open (Door)": 2000},
       "wilson_halfwidth_pp": {}, "pairs": {}}
for m, row in t2.items():
    v13["wilson_halfwidth_pp"][m] = [None if v is None else round(100 * (wilson(round(v * n / 100), n)[1] - wilson(round(v * n / 100), n)[0]) / 2, 1) for v, n in zip(row, n_assumed)]
v13["pairs"]["Door: Door specialist 77.7 vs Multitask 70.2 (n=2000 each)"] = ztest(0.777, 2000, 0.702, 2000)
v13["pairs"]["Open: Multitask 25.2 vs SPOC Artic 21.8 (n=2000 each)"] = ztest(0.252, 2000, 0.218, 2000)
v13["pairs"]["Door: Multitask 70.2 vs SPOC Artic 58.8"] = ztest(0.702, 2000, 0.588, 2000)
t9 = {"Door 1": [(1, 0), (0, 0), (0, 0)], "Door 2": [(1, 0), (1, 1), (1, 1)], "Door 3": [(0, 0), (0, 0), (0, 0)]}
v13["table9"] = {"grasp": 4, "opened": 2, "n": 9, "wilson_grasp": [round(100 * x, 1) for x in wilson(4, 9)], "wilson_open": [round(100 * x, 1) for x in wilson(2, 9)],
                 "door2_vs_others_grasp_fisher": fisher(3, 3, 1, 6), "hw_fault_trials": 6}
RES["V13_rby1"] = v13

# ---------------------------------------------------------------- V14: SPOC 파라미터 수 (정적 계산)
d, L = 512, 24
hid = int(2 * 4 * d / 3)
hid = 256 * ((hid + 255) // 256)
per_block = 4 * d * d + 4 * d * d + 3 * d * hid  # self-attn + cross-attn + SwiGLU
RES["V14_spoc_params"] = {"dim": d, "layers": L, "swiglu_hidden": hid, "per_block_params": per_block, "decoder_total_M": round(L * per_block / 1e6, 1),
                          "paper_trainable_M": 108, "code_docstring_M": 150, "note": "교차 주의가 모든 블록에 있다고 가정한 추정(norm, 임베딩, 병렬 디코더 제외)"}

# V15: 설정 검산 (표 5, 6 합, 코드 README 비율)
RES["V15_mixtures"] = {"table5_sum": 20 + 10 + 35 + 20 + 15, "table6_rows_sum": [20 + 20 + 30 + 30, 100, 50 + 50, 45 + 55],
                       "readme_rates_sum": 0.35 + 0.2 + 0.1 + 0.2 + 0.15, "task_horizon_steps_at_66ms_s": [round(303 * 0.066, 2), round(606 * 0.066, 2)],
                       "task_horizon_steps_at_200ms_s": [round(303 * 0.2, 1), round(606 * 0.2, 1)],
                       "rby1_horizon_400_at_100ms_s": 400 * 0.1}


# V17: 학습 샘플 수 대 Franka 프레임 수 (README 혼합 비율, v2 표 1의 프레임 수; 데이터 일부 사용 가능성은 반영하지 못함)
samples = 200_000 * 1024
rates = {"Pick": 0.2, "PnP": 0.45, "Next-To": 0.2, "Color": 0.15}
frames = {"Pick": 56.9e6, "PnP": 143.9e6, "Next-To": 54.7e6, "Color": 7.5e6}
RES["V17_samples_vs_frames"] = {
    "samples_total": samples, "multiframe_extra_samples": 50_000 * 1024,
    "per_task_samples_over_frames": {k: round(samples * rates[k] / frames[k], 2) for k in rates},
    "franka_frames_total_M": round(sum(frames.values()) / 1e6, 1),
    "overall_samples_over_franka_frames": round(samples / sum(frames.values()), 2),
    "note": "README의 part 선택이 전체의 일부면 비율은 더 커진다. v2 표 1의 프레임 수 기준"}


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return o


OUT.write_text(json.dumps(clean(RES), ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT)
