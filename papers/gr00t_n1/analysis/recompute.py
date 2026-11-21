"""GR00T N1 논문 표 재계산과 일관성 검사 (CPU, 1초 안팎).

실행: python3 analysis/recompute.py  -> analysis/recompute_results.json 과 표준 출력
입력: analysis/tables.py (논문 Table 2–5, 7의 인쇄값을 옮겨 적은 것)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tables as T  # noqa: E402

OUT = Path(__file__).resolve().parent / "recompute_results.json"
rng = np.random.default_rng(0)
res = {}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def boot_mean_ci(x, B=10000):
    x = np.asarray(x, float)
    m = rng.choice(x, size=(B, len(x)), replace=True).mean(axis=1)
    return (round(float(np.percentile(m, 2.5)), 2), round(float(np.percentile(m, 97.5)), 2))


# ---------- V1: Table 2 평균 열 재계산 (과제 수 가중 평균) ----------
w = T.N_TASKS
v1 = {}
for m in T.T2_AVG:
    num = sum(T.T2[b][m] * w[b] for b in w)
    v1[m] = {"weighted": round(num / sum(w.values()), 2), "printed": T.T2_AVG[m]}
res["V1_table2_weighted_average"] = v1

# ---------- V2: Table 4 과제별 행 평균 vs 인쇄된 벤치마크 평균 ----------
v2 = {}
for b, rows in T.TASKS.items():
    arr = np.array([r[1:] for r in rows], float)  # (n_tasks, 6)
    mean = arr.mean(axis=0).round(2)
    v2[b] = {"n_tasks": len(rows), "recomputed": mean.tolist(), "printed": list(T.T4_PRINTED_AVG[b]),
             "max_abs_diff": float(np.max(np.abs(mean - np.array(T.T4_PRINTED_AVG[b])))).__round__(2)}
res["V2_table4_row_means_vs_printed"] = v2

# ---------- V3: Table 2 열 vs Table 4 (100 demos) 벤치마크 평균 ----------
cols100 = {"BC-Transformer": None, "Diffusion Policy": 1, "GR00T-N1-2B": 4}
v3 = {}
for b in ["RoboCasa", "DexMG", "GR-1"]:
    rows = T.TASKS[b]
    dp = float(np.mean([r[2] for r in rows]))
    g = float(np.mean([r[5] for r in rows]))
    v3[b] = {"table2_DP": T.T2[b]["Diffusion Policy"], "table4_DP_100": round(dp, 2),
             "table2_GR00T": T.T2[b]["GR00T-N1-2B"], "table4_GR00T_100": round(g, 2)}
res["V3_table2_vs_table4_100demo"] = v3

# Table 2 평균을 Table 4 기준으로 다시 계산 (BC 제외)
dp_t4 = sum(v3[b]["table4_DP_100"] * w[b] for b in w) / sum(w.values())
g_t4 = sum(v3[b]["table4_GR00T_100"] * w[b] for b in w) / sum(w.values())
res["V3b_average_from_table4"] = {"DP": round(dp_t4, 2), "GR00T": round(g_t4, 2),
                                  "gap_table4": round(g_t4 - dp_t4, 2),
                                  "gap_printed": round(T.T2_AVG["GR00T-N1-2B"] - T.T2_AVG["Diffusion Policy"], 2)}

# 헤드라인 차이 (인쇄값)
res["V4_headline_gaps_printed"] = {
    "GR-1_GR00T_minus_DP_100": round(T.T2["GR-1"]["GR00T-N1-2B"] - T.T2["GR-1"]["Diffusion Policy"], 2),
    "GR-1_text_claim": "more than 17%",
}

# ---------- V5: Table 3 실로봇 평균 재계산 ----------
def real_task_means(col):
    """과제 단위 평균. PnP는 seen/unseen 두 행을 평균(본문 Table 3 정의와 일치 확인)."""
    groups = {}
    for g, name, *vals, n in T.REAL_ROWS:
        key_g = "PnP" if g.startswith("PnP") else g  # seen/unseen 두 행을 한 과제로 묶는다
        groups.setdefault((key_g, name), []).append((vals[col], n))
    out = {}
    for (g, name), lst in groups.items():
        out[(g, name)] = float(np.mean([v for v, _ in lst]))
    return out


names = ["DP 10%", "DP full", "GR00T 10%", "GR00T full"]
v5 = {}
task_means = {}
for col, nm in enumerate(names, start=0):
    tm = real_task_means(col)
    task_means[nm] = tm
    per_group = {}
    for (g, name), v in tm.items():
        per_group.setdefault(g if not g.startswith("PnP") else "PnP", []).append(v)
    grp = {g: round(float(np.mean(vs)), 2) for g, vs in per_group.items()}
    overall = round(float(np.mean(list(tm.values()))), 2)
    v5[nm] = {"group_means": grp, "overall_13_tasks": overall}
res["V5_table3_recomputed"] = v5
res["V5_table3_printed_avg"] = T.T3_PRINTED

# 13개 과제 쌍체 비교 (전체 데이터, 10%)
keys = list(task_means["DP full"].keys())
diff_full = np.array([task_means["GR00T full"][k] - task_means["DP full"][k] for k in keys])
diff_10 = np.array([task_means["GR00T 10%"][k] - task_means["DP 10%"][k] for k in keys])
def paired_summary(d):
    nz = d[d != 0]
    try:
        p_w = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        p_w = None
    k_pos = int((d > 0).sum())
    p_sign = float(stats.binomtest(k_pos, len(d), 0.5).pvalue) if len(d) else None
    return {"mean_diff": round(float(d.mean()), 2), "boot95": boot_mean_ci(d),
            "n_tasks": len(d), "n_positive": k_pos, "wilcoxon_p": round(p_w, 4) if p_w else None,
            "sign_test_p": round(p_sign, 4)}
res["S2_real_paired_13_tasks"] = {"full_data": paired_summary(diff_full), "10pct": paired_summary(diff_10)}

# 풀링한 시행 기반 신뢰구간 (행별 시행 수 가중). 부분 점수는 성공 분수로 취급.
n_total = sum(n for *_, n in T.REAL_ROWS)
pooled = {}
for col, nm in enumerate(names):
    k = sum(r[2 + col] / 100 * r[-1] for r in T.REAL_ROWS)
    pooled[nm] = {"successes_eq": round(k, 1), "n": n_total, "rate": round(100 * k / n_total, 1),
                  "wilson95": wilson(k, n_total)}
res["S2b_pooled_trials"] = {"n_trials_per_setting": n_total, **pooled}
# 2x2 근사 검정 (반올림한 성공 수)
def fisher(a, b):
    ka, kb = round(pooled[a]["successes_eq"]), round(pooled[b]["successes_eq"])
    tab = [[ka, n_total - ka], [kb, n_total - kb]]
    return float("%.3g" % stats.fisher_exact(tab).pvalue)
res["S2c_fisher_pooled_approx"] = {"full: GR00T vs DP": fisher("GR00T full", "DP full"),
                                   "10%: GR00T vs DP": fisher("GR00T 10%", "DP 10%")}

# 사전학습 평가 (15회)
pre = {}
for key, (k, n, printed) in T.PRETRAIN.items():
    pre[key] = {"k_partial_credit": k, "n": n, "rate_recomputed": round(100 * k / n, 2),
                "printed": printed, "wilson95": wilson(k, n)}
res["V6_pretrain"] = pre

# ---------- S1: 시뮬 과제별 쌍체 비교와 시행 수 기반 z 검정 (100 demos) ----------
sim = {}
for b, rows in T.TASKS.items():
    dp100 = np.array([r[2] for r in rows]) / 100
    g100 = np.array([r[5] for r in rows]) / 100
    d = g100 - dp100
    # 두 비율 z 검정 (과제당 시행 100회씩 가정; 체크포인트 최대값 선택은 반영하지 못함)
    p_pool = (dp100 + g100) / 2
    se = np.sqrt(p_pool * (1 - p_pool) * 2 / 100)
    z = np.where(se > 0, d / np.where(se > 0, se, 1), 0.0)
    pz = 2 * (1 - stats.norm.cdf(np.abs(z)))
    sim[b] = {
        "n_tasks": len(rows),
        "gr00t_wins_100": int((d > 0).sum()), "ties_100": int((d == 0).sum()), "dp_wins_100": int((d < 0).sum()),
        "mean_diff_pp": round(float(100 * d.mean()), 2),
        "boot95_mean_diff_pp": tuple(100 * x for x in boot_mean_ci(d)),
        "wilcoxon_p": round(float(stats.wilcoxon(d).pvalue), 6) if np.any(d != 0) else None,
        "n_tasks_p_lt_0.05_ztest": int((pz < 0.05).sum()),
    }
    sim[b]["boot95_mean_diff_pp"] = tuple(round(float(x), 2) for x in sim[b]["boot95_mean_diff_pp"])
res["S1_sim_paired_100demos"] = sim

all_rows = [r for b in T.TASKS for r in T.TASKS[b]]
d_all = np.array([r[5] - r[2] for r in all_rows]) / 100
res["S1b_sim_all_57_tasks_100demos"] = {
    "gr00t_wins": int((d_all > 0).sum()), "n": len(d_all),
    "wilcoxon_p": float("%.3g" % stats.wilcoxon(d_all).pvalue),
    "boot95_pp": tuple(round(float(x) * 100, 2) for x in boot_mean_ci(d_all)),
}
# 30 / 300 demos에서 승패
res["S1c_sim_wins_by_size"] = {
    "30": int(sum(1 for r in all_rows if r[4] > r[1])),
    "100": int(sum(1 for r in all_rows if r[5] > r[2])),
    "300": int(sum(1 for r in all_rows if r[6] > r[3])),
    "n": len(all_rows),
}

# ---------- V7: 부록 Table 7 합계 ----------
def sums(rows):
    return round(sum(r[1] for r in rows), 1), round(sum(r[2] for r in rows), 1)
robot = sums(T.T7_ROBOT)
human = sums(T.T7_HUMAN)
v7 = {
    "robot": {"recomputed": robot, "printed": T.T7_PRINTED["robot"]},
    "human": {"recomputed": human, "printed": T.T7_PRINTED["human"]},
    "sim": {"recomputed": T.T7_SIM, "printed": T.T7_PRINTED["sim"]},
    "neural": {"recomputed": T.T7_NEURAL, "printed": T.T7_PRINTED["neural"]},
}
tot_frames = round(robot[0] + human[0] + T.T7_SIM[0] + T.T7_NEURAL[0], 1)
tot_hours = round(robot[1] + human[1] + T.T7_SIM[1] + T.T7_NEURAL[1], 1)
v7["total"] = {"recomputed": (tot_frames, tot_hours), "printed": T.T7_PRINTED["total"]}
res["V7_table7_sums"] = v7

# ---------- V8: 본문 수치 사이의 산술 일관성 ----------
neural_traj_est = T.T7_NEURAL[0] * 1e6 / 81          # 81 프레임 하나당 궤적 수
neural_hours_from_fps = T.T7_NEURAL[0] * 1e6 / 8 / 3600  # 8 fps 가정
teleop_hours_from_fps = T.T7_ROBOT[0][1] * 1e6 / 20 / 3600  # GR-1 teleop 20 Hz
sim_hours_from_fps = T.T7_SIM[0] * 1e6 / 20 / 3600          # sim 20 fps 가정
gpu_hours_from_rate = 827 * 3600 * 120 / 3600               # 2분/1초 영상, 827시간
res["V8_consistency"] = {
    "neural_trajectories_est_(81_frames)": round(neural_traj_est / 1e3, 1),  # thousand
    "text_claim_neural_trajectories": "~300k",
    "neural_hours_from_8fps": round(neural_hours_from_fps, 1),
    "printed_neural_hours": T.T7_NEURAL[1],
    "teleop_hours_from_20hz": round(teleop_hours_from_fps, 1),
    "printed_teleop_hours": T.T7_ROBOT[0][2],
    "sim_hours_from_20fps": round(sim_hours_from_fps, 1),
    "printed_sim_hours": T.T7_SIM[1],
    "L40_gpu_hours_from_2min_per_s": round(gpu_hours_from_rate, 0),
    "printed_L40_gpu_hours": 105000,
    "sim_pretrain_trajs_540k_avg_s": round(T.T7_SIM[1] * 3600 / 540000, 2),
    "claim_780k_traj_equiv_6500h_s_per_traj": round(6500 * 3600 / 780000, 2),
    "teleop_to_neural_ratio": round(827 / 88, 2),
}

json.dump(res, OUT.open("w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)

# ---------- 출력 ----------
print("V1", res["V1_table2_weighted_average"])
for b in v2:
    print("V2", b, v2[b]["max_abs_diff"], "recomputed", v2[b]["recomputed"])
print("V3", res["V3_table2_vs_table4_100demo"])
print("V3b", res["V3b_average_from_table4"])
print("V5", {k: v["overall_13_tasks"] for k, v in v5.items()}, {k: v["group_means"] for k, v in v5.items()})
print("S2", res["S2_real_paired_13_tasks"])
print("S2b", {k: pooled[k] for k in pooled})
print("S2c", res["S2c_fisher_pooled_approx"])
print("V6", pre)
for b in sim:
    print("S1", b, sim[b])
print("S1b", res["S1b_sim_all_57_tasks_100demos"])
print("S1c", res["S1c_sim_wins_by_size"])
print("V7", v7)
print("V8", res["V8_consistency"])
