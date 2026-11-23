"""ArtManip (arXiv:2609.12498) 표 수치 재계산과 일관성 검사. numpy, scipy만 사용 (수 초).

입력은 모두 논문 표에서 옮긴 값이다(표 1–5, 부록 A.1 표 4).
  - 표 4(실로봇 시도별 사이클 수)는 5 시도 x 5 grasp x 12 객체이므로 표 3의 모든 값을 원자료에서 다시 계산할 수 있다.
  - 표 1, 2, 5(시뮬레이션)는 요약 값만 있어 논문이 적은 시행 수(grasp x 100 롤아웃)를 가정해 구간을 낸다.
결과는 recompute_results.json 에 저장한다.

실행: python3 analysis/recompute.py
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
rng = np.random.default_rng(0)

# ---------------- 표 4 (부록 A.1): 객체별 grasp 5개 x 시도 5회, 값은 완료한 open-close 사이클 수 ----------------
T4 = {
    "Knife-1": ["0/0/0/0/0", "1/1/1/1/1", "0/0/0/0/0", "1/1/1/1/0", "1/2/1/2/2"],
    "Knife-2": ["1/1/1/1/1", "2/2/2/2/2", "0/0/0/0/0", "3/4/4/4/4", "1/1/1/1/1"],
    "Knife-3": ["3/3/2/3/3", "3/3/3/3/3", "1/1/0/1/0", "0/0/0/0/0", "3/4/3/4/4"],
    "Lighter-1": ["1/1/2/1/2", "1/0/1/1/1", "2/2/2/2/1", "1/0/1/1/0", "0/0/0/0/0"],
    "Lighter-2": ["3/3/3/2/2", "0/0/0/0/0", "3/1/3/2/3", "4/4/3/3/3", "3/3/3/3/3"],
    "Lighter-3": ["1/1/0/0/1", "4/3/4/4/4", "3/3/3/2/2", "1/1/1/1/1", "3/3/3/3/2"],
    "Stapler-1": ["7/7/7/6/7", "7/7/7/7/7", "7/7/7/7/7", "7/7/7/7/7", "7/7/7/7/7"],
    "Stapler-2": ["0/0/0/0/0", "7/7/7/7/7", "7/6/7/6/7", "7/7/7/6/7", "7/7/7/7/7"],
    "Stapler-3": ["4/4/4/4/4", "7/7/7/7/7", "7/7/7/7/7", "4/3/4/4/3", "7/7/7/7/7"],
    "Tong-1": ["4/6/5/6/6", "10/10/10/8/10", "10/8/9/8/10", "3/3/2/3/3", "10/10/10/9/9"],
    "Tong-2": ["3/3/3/3/3", "2/3/2/3/3", "3/1/2/3/2", "4/3/2/4/4", "2/2/2/1/2"],
    "Tong-3": ["4/4/4/3/3", "7/5/6/6/7", "6/6/6/6/5", "2/2/2/2/2", "1/1/1/2/2"],
}
T4_SR_REPORTED = {"Knife-1": 56, "Knife-2": 80, "Knife-3": 72, "Lighter-1": 68, "Lighter-2": 80, "Lighter-3": 92,
                  "Stapler-1": 100, "Stapler-2": 80, "Stapler-3": 100, "Tong-1": 100, "Tong-2": 100, "Tong-3": 100}
# 표 3 (본문 4.4절): IC, GC, CSCmean, CSCmax, SR
T3 = {"Knife": dict(IC=100, GC=73.3, CSCmean=1.9, CSCmax=3.3, SR=69.3),
      "Lighter": dict(IC=100, GC=86.7, CSCmean=2.1, CSCmax=3.3, SR=80.0),
      "Stapler": dict(IC=100, GC=93.3, CSCmean=6.5, CSCmax=7.0, SR=93.3),
      "Tong": dict(IC=100, GC=100.0, CSCmean=4.6, CSCmax=7.0, SR=100.0)}
CATS = ["Knife", "Lighter", "Stapler", "Tong"]


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [100 * (c - h), 100 * (c + h)]


def r1(x):
    return round(float(x), 1)


res = {}

# ---------------- V1–V6: 표 4 -> 표 3 ----------------
trials = {k: np.array([[int(x) for x in cell.split("/")] for cell in cells]) for k, cells in T4.items()}  # (5 grasp, 5 trial)
inst_rows = {}
for name, a in trials.items():
    trial_ok = (a >= 1)
    sr = 100 * trial_ok.mean()
    grasp_ok = trial_ok.any(axis=1)
    cbar = a.mean(axis=1)
    inst_rows[name] = dict(
        SR=r1(sr), SR_reported=T4_SR_REPORTED[name], SR_match=bool(abs(sr - T4_SR_REPORTED[name]) < 0.05),
        n_success_grasps=int(grasp_ok.sum()), IC=bool(grasp_ok.any()), GC=100 * grasp_ok.mean(),
        CSCmean=float(cbar[grasp_ok].mean()), CSCmax=float(a.max()),
        CSCmax_if_mean_over_grasps=float(a.max(axis=1)[grasp_ok].mean()),
    )
res["V1_instance_SR_vs_table4"] = inst_rows

cat_rows = {}
all_trial_ok = 0
all_trials = 0
all_grasp_ok = 0
for c in CATS:
    names = [n for n in trials if n.startswith(c)]
    rows = [inst_rows[n] for n in names]
    k = sum(int((trials[n] >= 1).sum()) for n in names)
    n_tr = 25 * len(names)
    all_trial_ok += k
    all_trials += n_tr
    all_grasp_ok += sum(r["n_success_grasps"] for r in rows)
    mine = dict(IC=100 * np.mean([r["IC"] for r in rows]), GC=np.mean([r["GC"] for r in rows]),
                CSCmean=np.mean([r["CSCmean"] for r in rows]), CSCmax=np.mean([r["CSCmax"] for r in rows]),
                SR_category_mean=np.mean([r["SR"] for r in rows]), SR_pooled=100 * k / n_tr, k=k, n=n_tr,
                CSCmax_if_mean_over_grasps=np.mean([r["CSCmax_if_mean_over_grasps"] for r in rows]))
    rep = T3[c]
    cat_rows[c] = dict(
        recomputed={a: r1(b) for a, b in mine.items()}, reported=rep,
        match=dict(IC=bool(abs(mine["IC"] - rep["IC"]) < .05), GC=bool(abs(mine["GC"] - rep["GC"]) < .06),
                   CSCmean=bool(abs(mine["CSCmean"] - rep["CSCmean"]) < .06),
                   CSCmax=bool(abs(mine["CSCmax"] - rep["CSCmax"]) < .06),
                   SR=bool(abs(mine["SR_category_mean"] - rep["SR"]) < .06)),
        wilson_SR_trial_level=[r1(x) for x in wilson(k, n_tr)],
    )
res["V2_table3_from_table4"] = cat_rows
res["V3_totals"] = dict(
    trials=all_trials, successes=all_trial_ok, SR=r1(100 * all_trial_ok / all_trials),
    SR_mean_of_category_SR=r1(np.mean([cat_rows[c]["recomputed"]["SR_category_mean"] for c in CATS])),
    grasps=60, success_grasps=all_grasp_ok, GC_pooled=r1(100 * all_grasp_ok / 60),
    GC_mean_of_category=r1(np.mean([cat_rows[c]["recomputed"]["GC"] for c in CATS])),
    zero_grasps=[(n, i + 1) for n, a in trials.items() for i in range(5) if (a[i] >= 1).sum() == 0],
    zero_grasps_by_category={c: sum(1 for n, a in trials.items() if n.startswith(c) for i in range(5) if (a[i] >= 1).sum() == 0) for c in CATS},
    text_claims=dict(GC_avg_88_3=r1(np.mean([T3[c]["GC"] for c in CATS])), SR_257_of_300=all_trial_ok == 257),
)

# ---------------- V4: 실로봇 구간 추정 (시도 단위 Wilson vs 군집 부트스트랩) ----------------
grasp_k = np.concatenate([(a >= 1).sum(axis=1) for a in trials.values()])  # 60 grasps, 성공 시도 수(0..5)
inst_of = np.repeat(np.arange(12), 5)
B = 20000


def boot_grasp():
    idx = rng.integers(0, 60, (B, 60))
    return 100 * grasp_k[idx].sum(axis=1) / 300


def boot_inst():
    out = np.empty(B)
    per_inst = grasp_k.reshape(12, 5).sum(axis=1)  # 객체당 성공 시도 (25 중)
    idx = rng.integers(0, 12, (B, 12))
    out[:] = 100 * per_inst[idx].sum(axis=1) / 300
    return out


bg, bi = boot_grasp(), boot_inst()
p = all_trial_ok / all_trials
# 군집 내 상관: grasp 안 시도는 거의 같은 결과 (ANOVA 방식 ICC와 설계 효과)
msb = 5 * np.var(grasp_k / 5, ddof=1)
msw = np.mean([np.var((a >= 1).astype(float)[i], ddof=1) for a in trials.values() for i in range(5)])
icc = (msb - msw) / (msb + 4 * msw)
deff = 1 + 4 * icc
res["V4_real_SR_intervals"] = dict(
    pooled_SR=r1(100 * p), wilson_trial_level_n300=[r1(x) for x in wilson(all_trial_ok, 300)],
    cluster_bootstrap_by_grasp_60=[r1(np.percentile(bg, 2.5)), r1(np.percentile(bg, 97.5))],
    cluster_bootstrap_by_object_12=[r1(np.percentile(bi, 2.5)), r1(np.percentile(bi, 97.5))],
    icc_within_grasp=round(float(icc), 3), design_effect=round(float(deff), 2), effective_n=round(300 / float(deff), 1),
    wilson_effective_n=[r1(x) for x in wilson(round(p * 300 / deff), round(300 / deff))],
    grasp_level_success_53_of_60=[r1(x) for x in wilson(53, 60)],
    IC_12_of_12_wilson=[r1(x) for x in wilson(12, 12)],
    grasp_trial_success_hist={int(k): int((grasp_k == k).sum()) for k in range(6)},
)

# ---------------- V5: 사이클 수의 상한 효과 (실로봇) ----------------
allc = np.concatenate([a.ravel() for a in trials.values()])
stapler = np.concatenate([a.ravel() for n, a in trials.items() if n.startswith("Stapler")])
tong = np.concatenate([a.ravel() for n, a in trials.items() if n.startswith("Tong")])
res["V5_cycle_ceiling"] = dict(
    stapler_trials_with_7=int((stapler == 7).sum()), stapler_trials=int(stapler.size),
    stapler_max=int(stapler.max()), tong_max=int(tong.max()),
    tong_trials_with_10=int((tong == 10).sum()),
    seconds_per_cycle_at_max=dict(stapler=round(40 / 7, 2), tong=round(40 / 10, 2)),
    note="40 s 시도에서 stapler 7회, tong 10회가 관측 최댓값. 상한 값은 정책 한계가 아니라 시도 시간에 의한 절단일 가능성",
)

# ---------------- V6: 시뮬레이션 표 1 (시행 수 = grasp x 100 가정) ----------------
T1 = {"Knife": dict(inst=5, grasps=54, IC=100, GC=93.8, CSCmean=2.6, CSCmax=5.6, SR=85.0),
      "Lighter": dict(inst=5, grasps=101, IC=100, GC=92.8, CSCmean=1.4, CSCmax=7.0, SR=72.1),
      "Stapler": dict(inst=5, grasps=213, IC=100, GC=91.9, CSCmean=1.4, CSCmax=8.6, SR=83.2),
      "Tong": dict(inst=5, grasps=287, IC=100, GC=90.6, CSCmean=2.3, CSCmax=8.6, SR=79.8)}
t1 = {}
for c, d in T1.items():
    n = d["grasps"] * 100
    lo, hi = [wilson(round(d["SR"] / 100 * n), n)[i] for i in (0, 1)]
    k_lo, k_hi = math.ceil((d["SR"] - 0.05) / 100 * n), math.floor((d["SR"] + 0.05) / 100 * n)
    t1[c] = dict(n_assumed=n, k_range_for_rounding=[k_lo, k_hi], integer_feasible=bool(k_lo <= k_hi),
                 SR_wilson_iid=[r1(lo), r1(hi)],
                 CSCmean_ge_SR_check=bool(d["CSCmean"] >= d["SR"] / 100))
t1["knife_text_example"] = dict(n=5400, k=4590, SR=r1(100 * 4590 / 5400))
t1["GC_mean_text_92_3"] = r1(np.mean([d["GC"] for d in T1.values()]))
t1["CSCmax_range_text_5_6_to_8_6"] = [min(d["CSCmax"] for d in T1.values()), max(d["CSCmax"] for d in T1.values())]
t1["sum_total_grasps"] = sum(d["grasps"] for d in T1.values())
t1["grasps_per_instance_mean"] = {c: round(d["grasps"] / 5, 1) for c, d in T1.items()}
res["V6_table1_sim"] = t1

# ---------------- V7: 표 2 (Knife 훈련 인스턴스 수 절제) ----------------
T2 = {1: (99.2, 29.4, 30.8, 20.7), 5: (87.0, 64.6, 32.3, 51.5), 10: (88.8, 70.5, 35.7, 59.6), 30: (93.9, 85.0, 55.5, 74.9)}
cols = ["Anchor", "Unseen", "Geo-OOD", "Dyn-OOD"]
t2 = dict(monotone_increasing={c: bool(all(T2[a][i] < T2[b][i] for a, b in [(1, 5), (5, 10), (10, 30)])) for i, c in enumerate(cols)},
          gain_30_vs_1={c: r1(T2[30][i] - T2[1][i]) for i, c in enumerate(cols)},
          unseen_wilson_n5400={n: [r1(x) for x in wilson(round(T2[n][1] / 100 * 5400), 5400)] for n in T2},
          unseen_30_equals_table1_knife=bool(T2[30][1] == T1["Knife"]["SR"]),
          dyn_ood_vs_unseen={n: r1(T2[n][3] - T2[n][1]) for n in T2},
          anchor_drop_specialist_to_30=r1(T2[30][0] - T2[1][0]))
# 두 비율 차이 검정 (Unseen, n=5400 가정, 독립 가정)
for a in (5, 10):
    k1, k2 = round(T2[a][1] / 100 * 5400), round(T2[30][1] / 100 * 5400)
    t2[f"two_prop_z_30_vs_{a}_unseen"] = dict(z=round(float((k2 / 5400 - k1 / 5400) / math.sqrt(((k1 + k2) / 10800) * (1 - (k1 + k2) / 10800) * (2 / 5400))), 1))
res["V7_table2_ablation"] = t2

# ---------------- V8: 표 5 교사 vs 학생 ----------------
T5 = {  # (train IC, GC, CSCmean, CSCmax), (test IC, GC, CSCmean, CSCmax)
    "Teacher": {"Knife": ((100, 92.1, 3.7, 6.0), (100, 96.9, 3.6, 5.8)), "Lighter": ((100, 98.8, 4.5, 7.0), (100, 98.2, 4.2, 6.8)),
                "Stapler": ((100, 91.7, 2.7, 8.9), (100, 92.3, 2.2, 8.8)), "Tong": ((100, 97.3, 4.5, 8.9), (100, 98.0, 3.9, 9.0))},
    "Student": {"Knife": ((100, 90.7, 3.1, 5.8), (100, 93.8, 2.6, 5.6)), "Lighter": ((96.7, 95.3, 2.2, 6.6), (100, 92.8, 1.4, 7.0)),
                "Stapler": ((100, 95.5, 2.1, 8.5), (100, 91.9, 1.4, 8.6)), "Tong": ((100, 94.7, 2.9, 8.7), (100, 90.6, 2.3, 8.6))},
}
v8 = dict(student_test_equals_table1={c: bool(T5["Student"][c][1][1:] == (T1[c]["GC"], T1[c]["CSCmean"], T1[c]["CSCmax"])) for c in CATS},
          student_lighter_train_IC_29_of_30=r1(100 * 29 / 30))
for split, j in (("train", 0), ("test", 1)):
    d = {}
    for m, name in enumerate(["IC", "GC", "CSCmean", "CSCmax"]):
        if m == 0:
            continue
        diffs = [T5["Student"][c][j][m] - T5["Teacher"][c][j][m] for c in CATS]
        d[name] = dict(mean_student_minus_teacher=round(float(np.mean(diffs)), 2), per_category=[round(x, 2) for x in diffs])
    v8[split] = d
v8["CSCmean_ratio_student_over_teacher_test"] = {c: round(T5["Student"][c][1][2] / T5["Teacher"][c][1][2], 2) for c in CATS}
res["V8_table5_teacher_student"] = v8

# ---------------- V9: 시뮬레이션 대 실로봇 (같은 범주, 다른 프로토콜) ----------------
v9 = {}
for c in CATS:
    ks = cat_rows[c]["recomputed"]["k"]
    n_sim = T1[c]["grasps"] * 100
    k_sim = round(T1[c]["SR"] / 100 * n_sim)
    # 단순 2x2 Fisher (시도 단위 독립 가정, 프로토콜이 다르므로 참고용)
    odds, pval = stats.fisher_exact([[ks, 75 - ks], [k_sim, n_sim - k_sim]])
    v9[c] = dict(real_SR=r1(100 * ks / 75), sim_SR=T1[c]["SR"], fisher_p_trial_level=float(pval),
                 real_CSCmean=T3[c]["CSCmean"], sim_CSCmean=T1[c]["CSCmean"],
                 real_CSCmax=T3[c]["CSCmax"], sim_CSCmax=T1[c]["CSCmax"],
                 real_GC=T3[c]["GC"], sim_GC=T1[c]["GC"])
res["V9_sim_vs_real"] = v9

# ---------------- V10: 학습 예산 산술 (부록 C.6, 표 9) ----------------
steps, envs, horizon, mb = 2.0e9, 20000, 16, 40000
res["V10_training_budget"] = dict(
    env_steps=steps, envs=envs, horizon=horizon,
    updates_if_20000_total=round(steps / (envs * horizon)), updates_if_20000_per_gpu_x2=round(steps / (2 * envs * horizon)),
    minibatches_per_epoch_if_total=envs * horizon // mb, minibatches_per_epoch_if_x2=2 * envs * horizon // mb,
    frames_per_sec_overall=round(steps / (48 * 3600)),
    curriculum_end_epoch_paper=100 + 1000, curriculum_end_epoch_code_config=200 + 2000,
    curriculum_end_fraction_paper_total=round(1100 / (steps / (envs * horizon)), 3),
    curriculum_end_fraction_paper_x2=round(1100 / (steps / (2 * envs * horizon)), 3),
    curriculum_end_fraction_code_total=round(2200 / (steps / (envs * horizon)), 3),
    curriculum_end_fraction_code_x2=round(2200 / (steps / (2 * envs * horizon)), 3),
    action_step_paper_alpha=1 / 40, action_step_code_dt=1 / 120,
    max_joint_speed_rad_s_paper=round(30 / 40, 3), max_joint_speed_rad_s_code=round(30 / 120, 3),
    sim_episode_seconds=1200 / 30,
    max_cycles_with_1p5s_hold_each_stage=round(40 / 3.0, 1), max_cycles_with_1p0s_hold=round(40 / 2.0, 1),
)

# ---------------- V11: 보상 곱셈 (표 7 대 코드 설정) ----------------
res["V11_reward_weights"] = dict(
    table7_progress=dict(Knife=1000, Lighter=50, Stapler=100, Tong=100),
    code_GoalDistance1=dict(Knife=1000, Lighter=50, Stapler=100, Tong=100),
    tau_table7=dict(Knife=0.005, Lighter=0.10, Stapler=0.05, Tong=0.05),
    tau_code=dict(Knife=0.005, Lighter=0.1, Stapler=0.05, Tong=0.05),
    drop_pos_paper=0.05, drop_pos_code=dict(Knife=0.05, Lighter=0.05, Stapler=0.1, Tong=0.1),
    domain_rand_table8_vs_code=dict(
        Knife_damping=[700, 1300], code_knife_damping=[1000 * 0.7, 1000 * 1.3],
        Lighter_damping=[0.49, 0.91], code_lighter_damping=[0.7 * 0.7, 0.7 * 1.3],
        Stapler_stiffness=[8.4, 15.6], code_stapler_stiffness=[12 * 0.7, 12 * 1.3],
        Tong_stiffness=[4.9, 9.1], code_tong_stiffness=[7 * 0.7, 7 * 1.3],
        Knife_link0_mass=[0.04, 0.20], code_knife_link0_mass=[0.04 * 1, 0.04 * 5],
        Lighter_link0_mass=[0.03, 0.15], code_lighter_link0_mass=[0.03 * 1, 0.03 * 5],
        Stapler_link_mass=[0.03, 0.09], code_stapler_link_mass=[0.03 * 1, 0.03 * 3]),
)

# ---------------- V12: 부록 C.1 Geo-OOD 범위가 훈련 범위와 겹치지 않는지, 칼 관절 하한 범위 ----------------
body_z, slider_z = (0.10, 0.16), (0.02, 0.035)
clear_lo, clear_hi, travel = 0.015, 0.030, 0.050
lower_min = clear_lo - 0.5 * (body_z[1] - slider_z[0])
free_min = body_z[0] - slider_z[1]
eff_high = min(clear_hi, min(0.5 * free_min, free_min - travel))
lower_max = eff_high - 0.5 * free_min
train = dict(link0=[(0.015, 0.04), (0.008, 0.02), (0.10, 0.16)], link1=[(0.007, 0.015), (0.002, 0.008), (0.02, 0.035)])
ood = dict(link0=[(0.045, 0.060), (0.025, 0.035), (0.165, 0.180)], link1=[(0.020, 0.025), (0.010, 0.015), (0.040, 0.050)])
disjoint = {k: [bool(o[0] > t[1] or o[1] < t[0]) for o, t in zip(ood[k], train[k])] for k in train}
res["V12_asset_ranges"] = dict(knife_joint_lower_range_from_generator=[round(lower_min, 4), round(lower_max, 4)],
                              table6_knife_joint_lower=[-0.055, -0.0175], geo_ood_disjoint_each_axis=disjoint,
                              dyn_ood_damping_disjoint=bool(1400 > 1300))

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("V3_totals", "V4_real_SR_intervals", "V12_asset_ranges")}, ensure_ascii=False, indent=1, default=float))
for c in CATS:
    print(c, cat_rows[c]["recomputed"], cat_rows[c]["match"])
