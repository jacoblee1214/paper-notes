"""Dex4D(arXiv:2602.15828) 보고 수치 재계산과 일관성 검사. 결과는 recompute_results.json.

실행: python3 analysis/recompute.py   (numpy, scipy, 1초 미만)

입력은 논문 v2(2026-10-13)와 v1(2026-02-17)의 표에서 옮긴 값이다.
검사 ID는 V1–V14. 논문에 없는 시행 수는 가정으로 표시한다.
V12만 공식 코드(clone)를 읽는다. 이 clone은 아래 명령으로 만든 것과 같아야 한다.
    git clone --depth 1 https://github.com/Dex4D/Dex4D-Simulation.git repo/simulation
    git -C repo/simulation checkout 0dfccd82797e929cda6ab665003bc27a6b8cd31a
repo/simulation이 없으면 V12는 건너뛴다.
"""
import itertools
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "recompute_results.json"
R = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


# ---------------------------------------------------------------- 표 값 (v2)
TASKS = ["Apple2Plate", "Pour", "Hammer", "StackCup", "RotateBox", "Sponge2Bowl"]
T1 = {  # SR, TP 쌍, 과제 순서
    "NovaFlow": [(.44, .57), (.66, .76), (.12, .16), (.42, .59), (.25, .32), (.18, .29)],
    "LVP": [(.26, .64), (.39, .42), (.00, .00), (.10, .61), (.00, .16), (.11, .41)],
    "NovaFlow-CL": [(.58, .76), (.73, .85), (.27, .30), (.44, .73), (.33, .60), (.27, .41)],
    "Dex4D": [(.86, .91), (.86, .92), (.28, .31), (.60, .82), (.44, .69), (.56, .62)],
}
T1_AVG = {"NovaFlow": (.345, .448), "LVP": (.143, .373), "NovaFlow-CL": (.437, .608), "Dex4D": (.600, .712)}
T2 = {"MLP Point Encoding": (.057, .172), "Decoupled Point Encoding": (.203, .363), "w/o Self-Attention": (.490, .675),
      "w/o Dynamics Modeling": (.570, .683), "Dex4D": (.600, .712)}
T3 = {  # LiftToy, Broccoli2Plate, Meat2Bowl, Pour, Total (n=20 각)
    "LVP": [9, 6, 7, 0, 22], "NovaFlow-CL": [10, 7, 7, 0, 24], "pi0.5-DROID": [5, 6, 9, 0, 20], "Dex4D": [14, 10, 13, 8, 45]}
REAL_TASKS = ["LiftToy", "Broccoli2Plate", "Meat2Bowl", "Pour"]
APPX_F_CI = {"LiftToy": (14, .70, .48, .86), "Broccoli2Plate": (10, .50, .30, .70), "Meat2Bowl": (13, .65, .43, .82),
             "Pour": (8, .40, .22, .61), "Total": (45, .56, .45, .67)}  # 부록 F (n=20, 총 80)
T3_V1 = {"NovaFlow-CL": [4, 3, 3, 0, 10], "Dex4D": [6, 4, 5, 4, 19]}  # v1 표 III, n=10 각, 총 40

# ---------------------------------------------------------------- V1 표 1 평균
v1 = {}
for m, rows in T1.items():
    sr = np.mean([a for a, _ in rows])
    tp = np.mean([b for _, b in rows])
    v1[m] = {"SR_mean": round(float(sr), 4), "TP_mean": round(float(tp), 4), "reported": T1_AVG[m],
             "SR_match": abs(round(sr, 3) - T1_AVG[m][0]) < 1e-9, "TP_match": abs(round(tp, 3) - T1_AVG[m][1]) < 1e-9}
R["V1_table1_means"] = v1

# V2 TP >= SR, 평균 = 1/600 격자
v2 = {"TP_ge_SR_violations": [(m, TASKS[i]) for m, rows in T1.items() for i, (a, b) in enumerate(rows) if b < a],
      "SR_avg_on_1_over_600_grid": {m: abs(T1_AVG[m][0] * 600 - round(T1_AVG[m][0] * 600)) < 0.5 / 1 for m in T1_AVG}}
v2["SR_avg_times_600"] = {m: round(T1_AVG[m][0] * 600, 1) for m in T1_AVG}
v2["table2_SR_times_600"] = {k: round(v[0] * 600, 1) for k, v in T2.items()}
v2["table2_last_row_equals_table1_dex4d"] = T2["Dex4D"] == T1_AVG["Dex4D"]
R["V2_tp_sr_grid"] = v2

# ---------------------------------------------------------------- V3 이득 표기
sr = {m: T1_AVG[m][0] for m in T1_AVG}
tp = {m: T1_AVG[m][1] for m in T1_AVG}
v3 = {
    "sim_abs_gain_vs_NovaFlowCL_SR_pp": round((sr["Dex4D"] - sr["NovaFlow-CL"]) * 100, 1),
    "sim_abs_gain_vs_NovaFlowCL_TP_pp": round((tp["Dex4D"] - tp["NovaFlow-CL"]) * 100, 1),
    "sim_rel_gain_vs_NovaFlowCL_SR_pct": round((sr["Dex4D"] / sr["NovaFlow-CL"] - 1) * 100, 1),
    "sim_closed_vs_open_NovaFlow_SR_pp": round((sr["NovaFlow-CL"] - sr["NovaFlow"]) * 100, 1),
    "sim_closed_vs_open_NovaFlow_TP_pp": round((tp["NovaFlow-CL"] - tp["NovaFlow"]) * 100, 1),
    "real_rel_gain_vs_best_baseline_pct": round((45 / 24 - 1) * 100, 1),
    "real_abs_gain_vs_best_baseline_pp": round((45 - 24) / 80 * 100, 2),
    "real_rel_gain_vs_LVP_pct": round((45 / 22 - 1) * 100, 1),
    "real_rel_gain_vs_pi05_pct": round((45 / 20 - 1) * 100, 1),
    "v1_text_gain_22p5_is_abs_pp": round((19 - 10) / 40 * 100, 1),
    "v1_rel_gain_pct": round((19 / 10 - 1) * 100, 1),
}
R["V3_gain_definitions"] = v3

# ---------------------------------------------------------------- V4 표 3 합계, Wilson, Fisher
v4 = {"row_sums": {}, "wilson_appendixF": {}, "fisher_total": {}, "fisher_per_task": {}}
for m, row in T3.items():
    v4["row_sums"][m] = {"sum_tasks": sum(row[:4]), "reported_total": row[4], "match": sum(row[:4]) == row[4]}
for t, (k, p, lo, hi) in APPX_F_CI.items():
    n = 80 if t == "Total" else 20
    a, b = wilson(k, n)
    v4["wilson_appendixF"][t] = {"k": k, "n": n, "p": round(k / n, 4), "wilson": [round(a, 3), round(b, 3)],
                                 "reported": [lo, hi], "match": abs(a - lo) < 0.006 and abs(b - hi) < 0.006}
for m in ["LVP", "NovaFlow-CL", "pi0.5-DROID"]:
    v4["fisher_total"][m] = {"baseline": f"{T3[m][4]}/80", "dex4d": "45/80", "p": fisher(45, 80, T3[m][4], 80)}
raw = []
keys = []
for m in ["LVP", "NovaFlow-CL", "pi0.5-DROID"]:
    for i, t in enumerate(REAL_TASKS):
        p = fisher(T3["Dex4D"][i], 20, T3[m][i], 20)
        raw.append(p)
        keys.append(f"{m}|{t}")
adj = holm(raw)
for k, p, a in zip(keys, raw, adj):
    v4["fisher_per_task"][k] = {"p": p, "holm12": a}
v4["n_sig_raw_p05"] = int(sum(p < .05 for p in raw))
v4["n_sig_holm_p05"] = int(sum(a < .05 for a in adj))
# 기준선끼리 (총 80)
v4["baselines_pairwise_fisher"] = {f"{a} vs {b}": fisher(T3[a][4], 80, T3[b][4], 80)
                                   for a, b in itertools.combinations(["LVP", "NovaFlow-CL", "pi0.5-DROID"], 2)}
# 상대 이득의 95% 구간(로그 비율, 근사): 45/80 대 24/80
k1, k2, n = 45, 24, 80
rr = (k1 / n) / (k2 / n)
se = np.sqrt(1 / k1 - 1 / n + 1 / k2 - 1 / n)
v4["rel_gain_ci95_approx_pct"] = [round((rr * np.exp(-1.96 * se) - 1) * 100, 1), round((rr * np.exp(1.96 * se) - 1) * 100, 1)]
R["V4_real_world"] = v4

# ---------------------------------------------------------------- V5 시뮬 검정 (과제당 n=100, 쌍체 정보 없음 -> 독립 가정)
v5 = {"assumption": "과제당 100회(논문 4.1), 시도 독립 가정. 같은 초기 자세 공유 여부 미기재"}
pairs = [("Dex4D", "NovaFlow-CL"), ("Dex4D", "NovaFlow"), ("Dex4D", "LVP"), ("NovaFlow-CL", "NovaFlow")]
v5["per_task_fisher"] = {}
for a, b in pairs:
    row = {}
    for i, t in enumerate(TASKS):
        ka, kb = round(T1[a][i][0] * 100), round(T1[b][i][0] * 100)
        row[t] = {"a": ka, "b": kb, "p": fisher(ka, 100, kb, 100)}
    v5["per_task_fisher"][f"{a} vs {b}"] = row
_pp = [v["p"] for v in v5["per_task_fisher"]["Dex4D vs NovaFlow-CL"].values()]
v5["holm6_dex4d_vs_novaflowcl"] = dict(zip(TASKS, holm(_pp)))
v5["tasks_sig_holm_dex4d_vs_novaflowcl"] = [t for t, a in v5["holm6_dex4d_vs_novaflowcl"].items() if a < .05]
v5["tasks_sig_dex4d_vs_novaflowcl_raw"] = [t for t, d in v5["per_task_fisher"]["Dex4D vs NovaFlow-CL"].items() if d["p"] < .05]
v5["pooled_600"] = {}
for a, b in pairs:
    ka = round(sum(T1[a][i][0] for i in range(6)) * 100)
    kb = round(sum(T1[b][i][0] for i in range(6)) * 100)
    v5["pooled_600"][f"{a} vs {b}"] = {"a": ka, "b": kb, "p": fisher(ka, 600, kb, 600)}
# 절제 (표 2): 합산 600 가정
abl = {}
kd = round(T2["Dex4D"][0] * 600)
for k, (s, _) in T2.items():
    if k == "Dex4D":
        continue
    kk = round(s * 600)
    lo, hi = wilson(kk, 600)
    abl[k] = {"k_600": kk, "dex4d_k_600": kd, "diff_pp": round((kd - kk) / 6, 2), "p_fisher": fisher(kd, 600, kk, 600),
              "wilson_ablation": [round(lo, 3), round(hi, 3)]}
v5["ablation_pooled_600"] = abl
v5["ablation_note"] = "합산 600은 6과제 x 100회 가정. 학생 학습이 1회(시드 1)이면 이 p는 평가 표본 변동만 반영"
# 과제별 Dex4D 시뮬 SR 구간
v5["dex4d_task_wilson"] = {t: [round(x, 3) for x in wilson(round(T1["Dex4D"][i][0] * 100), 100)] for i, t in enumerate(TASKS)}
R["V5_sim_tests"] = v5

# ---------------------------------------------------------------- V6 시뮬 대 실로봇 Pour 등
v6 = {"pour_sim_dex4d": "86/100", "pour_real_dex4d": "8/20", "fisher_p": fisher(86, 100, 8, 20),
      "pour_sim_novaflowcl": "73/100", "pour_real_novaflowcl": "0/20", "fisher_p_nf": fisher(73, 100, 0, 20),
      "note": "시뮬 Pour는 사전 정의 waypoint, 실로봇 Pour는 생성 영상 점 궤적이라 조건이 다름(논문 4.1 대 3.4)"}
R["V6_sim_vs_real"] = v6

# ---------------------------------------------------------------- V7 v1 대 v2 실로봇 정합
v7 = {"v1": T3_V1, "v2_dex4d": T3["Dex4D"], "v2_novaflowcl": T3["NovaFlow-CL"]}
sub = {"Dex4D": [b - a for a, b in zip(T3_V1["Dex4D"][:4], T3["Dex4D"][:4])],
       "NovaFlow-CL": [b - a for a, b in zip(T3_V1["NovaFlow-CL"][:4], T3["NovaFlow-CL"][:4])]}
v7["v2_minus_v1_per_task_if_v1_is_subset"] = sub
v7["v2_minus_v1_total"] = {"Dex4D": 45 - 19, "NovaFlow-CL": 24 - 10}
v7["nonneg_all_cells"] = all(x >= 0 for s in sub.values() for x in s)
v7["added_trials_n_each"] = 10
v7["v1_vs_added_dex4d_fisher"] = {"v1": "19/40", "added": "26/40", "p": fisher(19, 40, 26, 40)}
v7["v1_vs_added_novaflowcl_fisher"] = {"v1": "10/40", "added": "14/40", "p": fisher(10, 40, 14, 40)}
v7["note"] = "v1의 10회가 v2 20회의 부분집합이라는 가정은 논문에 없음. 모든 셀에서 v2 >= v1이라 모순은 없음"
R["V7_v1_v2_real"] = v7

# ---------------------------------------------------------------- V8 실패 흐름 (그림 5, 그림 판독 값)
fl = {"total": 80, "planning_success": 72, "video_failure": 8, "grasp_success": 58, "grasp_failure": 14,
      "task_success": 45, "online_tracking_failure": 10, "execution_failure": 3}
v8 = {"flow": fl,
      "conservation": [fl["planning_success"] + fl["video_failure"] == fl["total"],
                       fl["grasp_success"] + fl["grasp_failure"] == fl["planning_success"],
                       fl["task_success"] + fl["online_tracking_failure"] + fl["execution_failure"] == fl["grasp_success"]],
      "task_success_matches_table3": fl["task_success"] == 45,
      "failure_total": fl["total"] - fl["task_success"],
      "failure_shares_pct": {k: round(fl[k] / (fl["total"] - fl["task_success"]) * 100, 1)
                             for k in ["video_failure", "grasp_failure", "online_tracking_failure", "execution_failure"]},
      "grasp_stage_success_rate": round(fl["grasp_success"] / fl["planning_success"], 3),
      "post_grasp_success_rate": round(fl["task_success"] / fl["grasp_success"], 3),
      "video_failure_wilson": [round(x, 3) for x in wilson(8, 80)],
      "note": "그림 5 숫자는 그림 판독 값. 비디오 실패 8은 80회 중 생성 영상이 계획으로 쓸 수 없었던 횟수(정의는 본문에 없음)"}
R["V8_failure_flow"] = v8

# ---------------------------------------------------------------- V9 그림 4 판독 (그림 판독 값, 근사)
f4 = {"points": [64, 48, 32, 16, 8], "SR": [.60, .58, .52, .365, .21], "TP": [.71, .70, .635, .495, .35]}
v9 = {"fig4_read": f4,
      "sr_ratio_to_64": [round(x / f4["SR"][0], 2) for x in f4["SR"]],
      "drop_pp_to_32": round((f4["SR"][0] - f4["SR"][2]) * 100, 1),
      "drop_pp_to_8": round((f4["SR"][0] - f4["SR"][4]) * 100, 1),
      "fig4_64_matches_table2": abs(f4["SR"][0] - .600) < .01 and abs(f4["TP"][0] - .712) < .01,
      "note": "그림 판독 값(오차 약 ±0.01). 0점으로 이어진 점선은 측정점이 아니라 연결선. 본문 '48과 32에서 한계적 저하'는 SR 0.60 → 0.52(-8 pp)"}
R["V9_point_robustness_figure"] = v9

# ---------------------------------------------------------------- V10 학습 규모와 시간
steps = {"teacher_iters": 15000 + 10000 + 25000, "table4_stage_bounds": [0, 15000, 25000, 50000]}
envs, nsteps = 4096, 8  # 부록 표 7(환경 수), 코드 ppo config nsteps 8(논문 미기재)
teacher_samples = envs * nsteps * steps["teacher_iters"]
st_env, st_nsteps = 3200, 1
student_samples = st_env * st_nsteps * 25000
hours_t = (2 * 24, 3 * 24)
v10 = {"teacher_iters": steps["teacher_iters"],
       "teacher_samples_assuming_nsteps8": teacher_samples,
       "teacher_samples_per_sec_range": [round(teacher_samples / (h * 3600)) for h in hours_t][::-1],
       "student_samples_assuming_nsteps1": student_samples,
       "student_samples_per_sec_20h": round(student_samples / (20 * 3600)),
       "teacher_stage_sum_ok": 15 + 10 + 25 == 50,
       "stage12_iters_text_vs_table4": "본문 15k + 10k + 25k = 50k, 표 4 구간 0-15k, 15-25k, 25-50k 일치",
       "note": "nsteps는 논문에 없고 공식 코드 설정값. 처리량은 2–3일, 20시간 보고 시간에서 역산한 값"}
R["V10_training_scale"] = v10

# ---------------------------------------------------------------- V11 보상식 단위 점검
d = np.array([0.0, 0.05, 0.1, 0.3, 0.4667, 0.5])
v11 = {"r_goal_vs_d": {str(x): round(float(1.4 - 3 * x), 3) for x in d},
       "r_goal_zero_at_d_m": round(1.4 / 3, 4),
       "bonus_at_success_edge_d0.05": round(5.0 / (1 + 10 * 0.05), 3),
       "bonus_at_d0": 5.0,
       "stay_bonus": 10.0,
       "r_action_stage3_weight5_clamp_note": "공식 코드는 -0.01*||a||^2를 -0.5에서 자름(가중치 곱 전)",
       "note": "r_goal은 접촉 시 d_t=0.4667 m 이상에서 음수. 성공 문턱 d_t < 0.05 m에서 1.25"}
R["V11_reward_units"] = v11

# ---------------------------------------------------------------- V12 객체 집합 (공식 코드 clone)
sim = ROOT / "repo" / "simulation" / "dex4d_policy" / "dex4d" / "cfg"


def load_entries(p):
    ents = []
    for line in p.read_text().splitlines():
        m = re.match(r"\s*'([^']+)':\s*\[([^\]]*)\]", line)
        if m:
            ents.append((m.group(1), m.group(2).strip()))
    return ents


if sim.exists():
    import collections
    tr, se, un = (load_entries(sim / f) for f in ["train_set.yaml", "test_set_seen_cat.yaml", "test_set_unseen_cat.yaml"])
    cat = lambda es: collections.Counter(k.split("/")[1].split("-")[0].lower() for k, _ in es)
    ctr = cat(tr)
    tr_meshes = {k for k, _ in tr}
    v12 = {"train_entries": len(tr), "train_unique_meshes": len(tr_meshes), "train_unique_pairs": len(set(tr)),
           "train_categories": len(ctr), "train_top_categories": ctr.most_common(6),
           "bottle_entries": ctr["bottle"], "paper_says": "3,200 objects (3.1, 부록 C)",
           "seen_entries": len(se), "seen_unique_meshes": len({k for k, _ in se}),
           "seen_meshes_also_in_train": len({k for k, _ in se} & tr_meshes),
           "unseen_entries": len(un), "unseen_unique_meshes": len({k for k, _ in un}),
           "unseen_meshes_in_train": len({k for k, _ in un} & tr_meshes),
           "unseen_categories": len(cat(un)), "unseen_categories_in_train": sorted(c for c in cat(un) if c in ctr),
           "scale_values": sorted({v for _, v in tr})}
    R["V12_object_set_from_code"] = v12
else:
    R["V12_object_set_from_code"] = "repo/simulation 없음, 건너뜀"

# ---------------------------------------------------------------- V13 참고문헌 번호 표본 확인 (abs 페이지 제목, 2026-11-26 확인값)
R["V13_reference_spot_check"] = {"[30] 2602.16863": "SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation",
                                "[39] 2609.38172": "Counterfactual Video Generation Enables Scalable Humanoid Loco-Manipulation",
                                "note": "두 번호 모두 abs 페이지가 열리고 제목이 참고문헌과 같음"}

# ---------------------------------------------------------------- V14 하이퍼파라미터 라벨
v14 = {"paper_table7": {"Discount Factor (lambda)": 0.95, "GAE parameter (gamma)": 0.96},
       "code_ppo_config": {"gamma": 0.96, "lam": 0.95},
       "note": "표 7은 할인율을 0.95, GAE를 0.96으로 적었고 코드는 gamma(할인) 0.96, lam(GAE) 0.95. 기호도 관례(λ=GAE, γ=할인)와 반대"}
R["V14_hparam_labels"] = v14

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")

# ---------------------------------------------------------------- 요약 출력
print("V1 means match:", {m: (d["SR_match"], d["TP_match"]) for m, d in v1.items()})
print("V2:", v2["TP_ge_SR_violations"], v2["SR_avg_times_600"], v2["table2_SR_times_600"])
print("V3:", v3)
print("V4 sums:", {m: d["match"] for m, d in v4["row_sums"].items()}, "wilson:", {t: d["match"] for t, d in v4["wilson_appendixF"].items()})
print("V4 fisher total:", {m: round(d["p"], 5) for m, d in v4["fisher_total"].items()})
print("V4 per task raw sig:", v4["n_sig_raw_p05"], "holm sig:", v4["n_sig_holm_p05"])
print("V4 ci:", v4["rel_gain_ci95_approx_pct"], v4["baselines_pairwise_fisher"])
print("V5 pooled:", {k: (d["a"], d["b"], round(d["p"], 5)) for k, d in v5["pooled_600"].items()})
print("V5 abl:", {k: (d["diff_pp"], round(d["p_fisher"], 4)) for k, d in abl.items()})
print("V5 per-task Dex4D vs NF-CL:", {t: (d["a"], d["b"], round(d["p"], 4)) for t, d in v5["per_task_fisher"]["Dex4D vs NovaFlow-CL"].items()})
print("V5 holm6:", v5["holm6_dex4d_vs_novaflowcl"], v5["tasks_sig_holm_dex4d_vs_novaflowcl"])
print("V4 per-task:", {k:(round(d["p"],4),round(d["holm12"],4)) for k,d in v4["fisher_per_task"].items()})
print("V6:", v6)
print("V7:", v7)
print("V8:", v8["conservation"], v8["failure_shares_pct"], v8["grasp_stage_success_rate"], v8["post_grasp_success_rate"])
print("V9:", v9["sr_ratio_to_64"], v9["drop_pp_to_32"])
print("V10:", v10)
print("V12:", R["V12_object_set_from_code"])
