"""TacCoRL (arXiv:2606.11743 v1) 표 재계산과 검정. 결과는 recompute_results.json에 저장한다.

실행: python3 analysis/recompute.py   (numpy, scipy 필요, 수 초)
가정: 실로봇 표 2는 셀당 시행 20회(본문 4.2, 모든 값이 0.05의 배수), 시뮬 표 1은 128개 병렬 환경 평가(본문 4.2).
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
TASKS = ["tube", "puzzle", "asm1", "asm2"]

# 표 1 (시뮬레이션), 표 2 (실로봇)
SIM = {
    "V": {"cotrain": [.42, .12, .39, .39], "noise": [.35, .05, .25, .25], "base_rl": [0, 0, 0, 0], "rl": [.50, .54, .77, .61]},
    "VT": {"cotrain": [.41, .16, .47, .58], "noise": [.36, .04, .35, .42], "base_rl": [0, 0, 0, 0], "rl": [.72, .71, .92, .79]},
}
REAL = {
    "V": {"real_only": [.20, .05, .35, .25], "cotrain": [.35, .10, .40, .35], "rl": [.35, .25, .80, .60]},
    "VT": {"real_only": [.45, .15, .35, .40], "cotrain": [.50, .25, .45, .55], "rl": [.70, .45, .95, .80]},
}
N_REAL = 20
N_SIM = 128


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def fisher(a, b, n=N_REAL):
    """a, b = 성공 횟수. 양측 Fisher exact."""
    return float(stats.fisher_exact([[a, n - a], [b, n - b]])[1])


def cmh(pairs, n=N_REAL):
    """층(과제)별 2x2 표 (VT 성공, V 성공)의 Cochran-Mantel-Haenszel(연속성 보정) 검정과 공통 오즈비(MH)."""
    num = 0.0; var = 0.0; r = 0.0; s = 0.0
    for a, b in pairs:
        N = 2 * n; m1 = a + b; m2 = N - m1
        ea = m1 * n / N
        v = m1 * m2 * n * n / (N * N * (N - 1))
        num += a - ea; var += v
        r += a * (n - b) / N; s += b * (n - a) / N
    chi = (abs(num) - 0.5) ** 2 / var
    return {"chi2": round(chi, 3), "p": float(stats.chi2.sf(chi, 1)), "or_mh": round(r / s, 3)}


def k(p, n):
    return int(round(p * n))


res = {}

# 1. 평균 일치 검사
avg = {}
for dom, tab in (("sim", SIM), ("real", REAL)):
    for pol, rows in tab.items():
        for name, v in rows.items():
            avg[f"{dom}_{pol}_{name}"] = round(100 * float(np.mean(v)), 2)
res["averages_pct"] = avg
res["reported_text_averages_pct"] = {"sim_V_cotrain": 33.0, "sim_V_noise": 22.5, "sim_VT_cotrain": 40.5, "sim_VT_noise": 29.3,
                                    "sim_V_rl": 60.5, "sim_VT_rl": 78.5, "real_V_rl": 50.0, "real_VT_rl": 72.5}
res["avg_match"] = {k_: abs(avg[k_] - v) < 0.06 for k_, v in res["reported_text_averages_pct"].items()}

# 2. 정수 가능성
real_ok = all(abs(p * N_REAL - round(p * N_REAL)) < 1e-9 for t in REAL.values() for r in t.values() for p in r)
res["real_integer_feasible_n20"] = real_ok
# 시뮬: 소수 둘째 자리 표 값은 n=128에서 항상 가능(간격 0.0078). 본문 소수 첫째 자리 값(42.9, 70.3, 14.1)은?
txt_vals = [42.9, 70.3, 14.1]
feas = {}
for v in txt_vals:
    feas[str(v)] = [{"k": kk, "pct_exact": round(100 * kk / N_SIM, 3)} for kk in range(N_SIM + 1) if abs(100 * kk / N_SIM - v) < 0.1]
res["sim_n128_text_values"] = feas
ns = [n for n in range(20, 1001) if all(any(round(100 * kk / n, 1) == v for kk in range(n + 1)) for v in txt_vals)]
res["denominators_20_1000_consistent_with_all_three_by_rounding"] = {"count": len(ns), "first": ns[:12], "has_128": 128 in ns}
res["denominators_with_128_fail_which"] = [v for v in txt_vals if not any(round(100 * kk / 128, 1) == v for kk in range(129))]

# 3. 실로봇 Wilson 구간과 Fisher
real = {}
for pol in ("V", "VT"):
    for st, v in REAL[pol].items():
        real[f"{pol}_{st}"] = [{"task": t, "k": k(p, N_REAL), "p": p, "wilson95": wilson(k(p, N_REAL), N_REAL)} for t, p in zip(TASKS, v)]
res["real_wilson"] = real
pool = {}
for st in ("real_only", "cotrain", "rl"):
    kv = sum(k(p, N_REAL) for p in REAL["V"][st]); kt = sum(k(p, N_REAL) for p in REAL["VT"][st])
    pool[st] = {"V": f"{kv}/80", "VT": f"{kt}/80", "V_pct": 100 * kv / 80, "VT_pct": 100 * kt / 80,
                "V_wilson": wilson(kv, 80), "VT_wilson": wilson(kt, 80), "fisher_p_pooled": fisher(kt, kv, 80),
                "cmh": cmh([(k(a, 20), k(b, 20)) for a, b in zip(REAL["VT"][st], REAL["V"][st])])}
res["real_pooled_VT_vs_V"] = pool
per = {}
for st in ("real_only", "cotrain", "rl"):
    ps = [fisher(k(a, 20), k(b, 20)) for a, b in zip(REAL["VT"][st], REAL["V"][st])]
    order = np.argsort(ps); holm = [None] * 4; run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, ps[i] * (4 - rank))); holm[i] = run
    per[st] = [{"task": t, "VT": k(a, 20), "V": k(b, 20), "diff_pp": round(100 * (a - b), 1), "fisher_p": round(p, 4), "holm_p": round(h, 4)}
               for t, a, b, p, h in zip(TASKS, REAL["VT"][st], REAL["V"][st], ps, holm)]
res["real_per_task_VT_vs_V"] = per

# 단계 효과(같은 정책 종류, 80회 합산)
stage = {}
for pol in ("V", "VT"):
    s = {st: sum(k(p, 20) for p in REAL[pol][st]) for st in REAL[pol]}
    stage[pol] = {"counts_of_80": s,
                  "rl_vs_cotrain_p": fisher(s["rl"], s["cotrain"], 80),
                  "cotrain_vs_real_only_p": fisher(s["cotrain"], s["real_only"], 80),
                  "rl_vs_real_only_p": fisher(s["rl"], s["real_only"], 80)}
res["real_stage_effects_pooled80"] = stage
# 교호작용: RL 이득(%p)이 촉각 유무로 다른가 (합산)
gainV = (sum(REAL["V"]["rl"]) - sum(REAL["V"]["cotrain"])) / 4 * 100
gainVT = (sum(REAL["VT"]["rl"]) - sum(REAL["VT"]["cotrain"])) / 4 * 100
res["real_rl_gain_pp"] = {"V": gainV, "VT": gainVT}
# 초록의 '기준선 50.0%'와 가장 약한 기준선
res["baselines_pct"] = {"abstract_baseline_V_rl": 100 * np.mean(REAL["V"]["rl"]), "V_real_only": 100 * np.mean(REAL["V"]["real_only"]),
                        "VT_real_only": 100 * np.mean(REAL["VT"]["real_only"]), "VT_cotrain": 100 * np.mean(REAL["VT"]["cotrain"]),
                        "ours_VT_rl": 100 * np.mean(REAL["VT"]["rl"])}
res["ours_vs_weakest_vision_baseline"] = {"VT_rl_vs_V_real_only": fisher(58, 17, 80)}

# 4. 절제(Assembly #2, 20회 가정)
abl = {"alpha0_b0": 5, "alpha0.5_b0": 9, "alpha0.95_b0": 8, "alpha0.5_b0.1": 16, "alpha0.5_b1.0": 16, "alpha0.5_b5.0": 10}
res["ablation_real_k_of_20"] = abl
res["ablation_wilson"] = {kx: wilson(v, 20) for kx, v in abl.items()}
res["ablation_fisher"] = {"b0.1_vs_b0": fisher(16, 9), "b1.0_vs_b0": fisher(16, 9), "b1.0_vs_b5.0": fisher(16, 10),
                          "a0.5_vs_a0.95": fisher(9, 8), "a0.5_vs_a0": fisher(9, 5), "b1.0_vs_b0.1": fisher(16, 16)}
# 표 2 Assembly #2 VT: 공동학습 0.55 = 11/20, RL(β=1.0) 0.80 = 16/20, 그림 7 β=0은 9/20
res["asm2_VT_cotrain_11_vs_rl_noanchor_9_p"] = fisher(11, 9)
res["asm2_VT_cotrain_11_vs_rl_anchor_16_p"] = fisher(16, 11)

# 5. 시뮬 대 실로봇
sim_all = []; real_all = []; lab = []
for st_s, st_r in (("cotrain", "cotrain"), ("rl", "rl")):
    for pol in ("V", "VT"):
        for t, s_, r_ in zip(TASKS, SIM[pol][st_s], REAL[pol][st_r]):
            sim_all.append(s_); real_all.append(r_); lab.append((st_s, pol, t))
sim_all = np.array(sim_all); real_all = np.array(real_all)
res["sim_vs_real_16cells"] = {"pearson_r": float(np.corrcoef(sim_all, real_all)[0, 1]), "mean_abs_diff_pp": float(100 * np.mean(abs(sim_all - real_all))),
                              "max_abs_diff_pp": float(100 * np.max(abs(sim_all - real_all))),
                              "max_cell": lab[int(np.argmax(abs(sim_all - real_all)))]}
for st in ("cotrain", "rl"):
    m = [i for i, l in enumerate(lab) if l[0] == st]
    res[f"sim_vs_real_{st}"] = {"pearson_r": float(np.corrcoef(sim_all[m], real_all[m])[0, 1]),
                                "mean_abs_diff_pp": float(100 * np.mean(abs(sim_all[m] - real_all[m]))),
                                "spearman": float(stats.spearmanr(sim_all[m], real_all[m])[0])}
res["sim_vs_real_rl_cells_pp"] = [{"cell": f"{l[1]}_{l[2]}", "sim": s_, "real": r_, "diff_pp": round(100 * (s_ - r_), 1)}
                                 for l, s_, r_ in zip(lab, sim_all, real_all) if l[0] == "rl"]
# 시뮬 효과: 촉각 이득, n=128 독립 가정 z 검정
simz = []
for st in ("cotrain", "noise", "rl"):
    for t, a, b in zip(TASKS, SIM["VT"][st], SIM["V"][st]):
        ka, kb = round(a * N_SIM), round(b * N_SIM)
        p_pool = (ka + kb) / (2 * N_SIM)
        se = np.sqrt(p_pool * (1 - p_pool) * 2 / N_SIM) if 0 < p_pool < 1 else float("nan")
        z = (ka - kb) / N_SIM / se if se == se and se > 0 else float("nan")
        simz.append({"stage": st, "task": t, "diff_pp": round(100 * (a - b), 1), "z": round(float(z), 2), "p": round(float(2 * stats.norm.sf(abs(z))), 4) if z == z else None})
res["sim_tactile_gain_z_n128_independent_assumed"] = simz
res["sim_se_pp_at_p0.5"] = 100 * float(np.sqrt(.25 / N_SIM))
res["real_se_pp_at_p0.5"] = 100 * float(np.sqrt(.25 / N_REAL))
res["sim_tactile_gain_rl_pp"] = {t: round(100 * (a - b), 1) for t, a, b in zip(TASKS, SIM["VT"]["rl"], SIM["V"]["rl"])}
res["real_tactile_gain_rl_pp"] = {t: round(100 * (a - b), 1) for t, a, b in zip(TASKS, REAL["VT"]["rl"], REAL["V"]["rl"])}
res["sim_rl_gain_over_cotrain_avg_pp"] = {"V": 100 * (np.mean(SIM["V"]["rl"]) - np.mean(SIM["V"]["cotrain"])), "VT": 100 * (np.mean(SIM["VT"]["rl"]) - np.mean(SIM["VT"]["cotrain"]))}

# 6. 검정력(몬테카를로, 시드 고정): 실로봇 20회 또는 80회에서 50 → 72.5%를 p<0.05로 검출할 확률
rng = np.random.default_rng(0)
def power(p1, p2, n, reps=3000):
    hit = 0
    for _ in range(reps):
        a = rng.binomial(n, p1); b = rng.binomial(n, p2)
        hit += fisher(a, b, n) < 0.05
    return hit / reps
res["power_fisher_alpha05"] = {"50_vs_72.5_n20": power(.5, .725, 20), "50_vs_72.5_n80": power(.5, .725, 80),
                              "35_vs_70_n20": power(.35, .70, 20), "60_vs_80_n20": power(.60, .80, 20)}
# 과제당 필요 시행(80% 검정력, 정규근사 양측 5%)
def n_needed(p1, p2):
    za, zb = 1.96, 0.8416; pb = (p1 + p2) / 2
    return float(np.ceil(((za * np.sqrt(2 * pb * (1 - pb)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p1 - p2) ** 2))
res["n_per_arm_for_80pct_power"] = {"50_vs_72.5": n_needed(.5, .725), "60_vs_80": n_needed(.6, .8), "35_vs_70": n_needed(.35, .70)}

# 7. 부록 수치 산술
sys_id = {"J0": (1813.0, 159.4, .12), "J1": (1392.4, 123.7, -.44), "J2": (974.8, 87.9, .04), "J3": (587.1, 69.8, -.12), "J4": (151.1, 17.1, 1.73), "J5": (151.8, 16.5, .07)}
res["sysid_kd_over_kp"] = {j: round(v[1] / v[0], 4) for j, v in sys_id.items()}
res["sysid_reference_kd_over_kp"] = 50 / 500
res["sysid_within_bounds"] = all(10 <= v[0] <= 1e5 and 1 <= v[1] <= 1e3 and -8 <= v[2] <= 8 for v in sys_id.values())
res["sysid_kp_ratio_to_reference"] = {j: round(v[0] / 500, 2) for j, v in sys_id.items()}
res["taxels"] = {"K_table3": 4 * 32 * 12, "per_pad": 32 * 12, "pads_total_assumed": 4}
res["action_dim_14_is_2x(6+1)"] = 2 * (6 + 1) == 14
res["rollout_arithmetic"] = {"chunk_decisions_per_rollout": 128 * 36, "timesteps_per_rollout_H10": 128 * 36 * 10, "minibatches_per_update_B512": 128 * 36 / 512,
                             "decisions_for_300_steps_if_step_is_one_rollout": 300 * 128 * 36}
res["cotrain_samples"] = {"asm_1000_steps_B192": 1000 * 192, "puzzle_tube_3000_steps_B192": 3000 * 192}
res["data_counts"] = {"real_demos_sec4.2": 50, "real_anchor_episodes_table4": 200, "ratio": 4.0, "sim_teleop": 20, "sim_mimicgen_total": 200}
res["tactile_mean_normalized"] = {"real": 0.616, "sim": 0.658, "diff": round(0.658 - 0.616, 3), "rel_pct": round(100 * (0.658 - 0.616) / 0.616, 1)}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print("wrote", OUT)
