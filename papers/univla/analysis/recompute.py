"""UniVLA 논문 표 재계산과 일관성 검사 (numpy, scipy). 실행: python3 analysis/recompute.py
결과는 analysis/recompute_results.json 에 저장한다. 값은 논문 PDF(arXiv v3)와 공식 README에서 옮긴 것이다.
Fisher·Wilson 검정은 시행 수를 가정한 근사다(아래 N_* 상수 참고). 실로봇 시행 수 15는 백분율의 분모로 역산한 값이다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}


def check(label, computed, reported, tol=0.05):
    ok = abs(computed - reported) <= tol + 1e-9
    return {"label": label, "computed": round(computed, 3), "reported": reported, "ok": ok}


# ---------- V1. 표 I (LIBERO, 논문 Table I): 네 suite 평균 대 보고 평균 ----------
T1 = {  # method: (Spatial, Object, Goal, Long, reported_avg)
    "LAPA*": (73.8, 74.6, 58.8, 55.4, 65.7),
    "Diffusion Policy": (78.3, 92.5, 68.3, 50.5, 72.4),
    "Octo": (78.9, 85.7, 84.6, 51.1, 75.1),
    "MDT": (78.5, 87.5, 73.5, 64.8, 76.1),
    "OpenVLA": (84.7, 88.4, 79.2, 53.7, 76.5),
    "MaIL": (74.3, 90.1, 81.8, 78.6, 83.5),
    "UniVLA Human": (91.2, 94.2, 90.2, 79.4, 88.7),
    "UniVLA Bridge": (95.2, 95.4, 91.9, 87.5, 92.5),
    "UniVLA Full": (96.5, 96.8, 95.6, 92.0, 95.2),
}
v1 = []
for m, (a, b, c, d, rep) in T1.items():
    v1.append(check(f"Table I avg {m}", np.mean([a, b, c, d]), rep, tol=0.06))
res["V1_table1_avg"] = v1

# 본문 주장 (Table I 기준)
full_avg = np.mean(T1["UniVLA Full"][:4])
res["V2_text_claims_libero"] = [
    check("Full - OpenVLA (본문 18.7)", full_avg - T1["OpenVLA"][4], 18.7, tol=0.1),
    check("Full - LAPA (본문 29.5)", full_avg - T1["LAPA*"][4], 29.5, tol=0.1),
    check("Human - OpenVLA (본문 12.2)", T1["UniVLA Human"][4] - T1["OpenVLA"][4], 12.2, tol=0.1),
    check("Full - OpenVLA (서론 18.5)", full_avg - T1["OpenVLA"][4], 18.5, tol=0.1),
    check("MaIL 본문 83.5 대 재계산", np.mean(T1["MaIL"][:4]), 83.5, tol=0.1),
]

# ---------- V3. 표 II (일반화), 표 III (잠재행동), 표 IV (디코더), 표 V (history) ----------
t2 = {  # method: [(succ, score) x3 settings]: lighting, distractor, novel
    "DP": [(20.0, .60), (26.7, .80), (26.7, .67)],
    "OpenVLA": [(13.3, .93), (20.0, .73), (26.7, 1.27)],
    "LAPA": [(26.7, 1.60), (6.7, .6), (53.3, 1.87)],
    "UniVLA": [(66.7, 2.33), (53.3, 2.40), (86.7, 2.73)],
}
v3 = []
for m, rows in t2.items():
    v3.append(check(f"Table II avg succ {m} (68.9/28.9/24.4/20.0 reported)", np.mean([r[0] for r in rows]),
                    {"DP": 24.4, "OpenVLA": 20.0, "LAPA": 28.9, "UniVLA": 68.9}[m], tol=0.06))
    v3.append(check(f"Table II avg score {m}", np.mean([r[1] for r in rows]),
                    {"DP": 0.69, "OpenVLA": 0.98, "LAPA": 1.36, "UniVLA": 2.49}[m], tol=0.006))
res["V3_table2"] = v3

t3 = {  # Table III (LIBERO, Ego4D pretraining): Spatial, Object, Goal, Long
    "Genie": ((89.8, 92.8, 77.2, 69.6), 82.3),
    "Task-irrelevant": ((68.0, 90.4, 67.2, 0.2), 56.5),
    "Task-centric": ((91.2, 94.2, 90.2, 79.4), 88.7),
}
v4 = [check(f"Table III avg {k}", np.mean(v[0]), v[1], tol=0.06) for k, v in t3.items()]
res["V4_table3"] = v4
res["V4_text_claims_table3"] = [
    check("Task-centric - Genie avg (본문 6.4)", np.mean(t3["Task-centric"][0]) - np.mean(t3["Genie"][0]), 6.4, tol=0.1),
    check("Goal diff (본문 13)", t3["Task-centric"][0][2] - t3["Genie"][0][2], 13.0, tol=0.01),
    check("Long diff (본문 9.8)", t3["Task-centric"][0][3] - t3["Genie"][0][3], 9.8, tol=0.01),
]

t4 = {  # Table IV (decoder): Spatial, Object, Goal, Long, avg
    "Auto-regressive": ((85.2, 81.2, 79.0, 49.0), 73.6),
    "Ours w/o visual": ((95.0, 95.4, 93.7, 86.0), 92.5),
    "Ours": ((96.5, 96.8, 95.6, 92.0), 95.2),
}
v5 = [check(f"Table IV avg {k}", np.mean(v[0]), v[1], tol=0.06) for k, v in t4.items()]
v5.append(check("Ours - w/o visual avg (본문 2.2)", np.mean(t4["Ours"][0]) - np.mean(t4["Ours w/o visual"][0]), 2.2, tol=0.05))
v5.append(check("Ours - AR Long (본문 42.1)", t4["Ours"][0][3] - t4["Auto-regressive"][0][3], 42.1, tol=0.05))
res["V5_table4_and_text"] = v5

t5 = {"instruction-only": (95.0, 88.1, 30.6), "w/ history": (95.6, 92.0, 47.1)}  # Goal, Long, R2R
res["V6_table5"] = [
    check("Table V history - instr Long (본문 3.9)", t5["w/ history"][1] - t5["instruction-only"][1], 3.9, tol=0.01),
    check("Table V history - instr R2R (본문 16.5)", t5["w/ history"][2] - t5["instruction-only"][2], 16.5, tol=0.01),
]

# ---------- V7. 표 A-II CALVIN: 평균 길이 = 연속 성공 비율 합 / 100 ----------
calvin = {"UniVLA": ([95.5, 85.8, 75.4, 66.9, 56.5], 3.80), "OpenVLA": ([91.3, 77.8, 62.0, 52.1, 43.5], 3.27),
          "CLOVER": ([96.0, 83.5, 70.8, 57.5, 45.4], 3.53), "RoboDual": ([94.4, 82.7, 72.1, 62.4, 54.4], 3.66),
          "GR-1": ([85.4, 71.2, 59.6, 49.7, 40.1], 3.06), "RoboFlamingo": ([82.4, 61.9, 46.6, 33.1, 23.5], 2.48),
          "SuSIE": ([87.0, 69.0, 49.0, 38.0, 26.0], 2.69), "RT-1": ([53.3, 22.2, 9.4, 3.8, 1.3], 0.90)}
res["V7_calvin"] = [check(f"CALVIN avg len {k}", sum(v[0]) / 100, v[1], tol=0.006) for k, v in calvin.items()]
res["V7_text_calvin"] = [check("UniVLA 5-task - CLOVER (본문 11.1)", 56.5 - 45.4, 11.1, tol=0.05),
                         check("UniVLA 5-task - OpenVLA (본문 13)", 56.5 - 43.5, 13.0, tol=0.05)]

# ---------- V8. 표 A-III SimplerEnv: 전체 평균 = 네 과제 성공률 평균 (README 표와 대조) ----------
simpler_succ = {  # spoon, carrot, stack, eggplant success (%), reported overall
    "RT-1-X": ([0.0, 4.2, 0.0, 0.0], 1.1), "Octo-Base": ([12.5, 8.3, 0.0, 43.1], 16.0),
    "Octo-Small": ([47.2, 9.7, 4.2, 56.9], 30.0), "OpenVLA": ([0.0, 0.0, 0.0, 4.1], 1.0),
    "RoboVLM": ([29.2, 25.0, 12.5, 58.3], 31.3), "UniVLA (Full)": ([52.8, 55.6, 2.8, 80.6], 47.9),
}
res["V8_simplerenv"] = [check(f"SimplerEnv overall {k}", np.mean(v[0]), v[1], tol=0.06) for k, v in simpler_succ.items()]
res["V8_text_simpler"] = [check("UniVLA overall 본문 42.7 대 표 47.9", np.mean(simpler_succ["UniVLA (Full)"][0]), 42.7, tol=0.05),
                          check("UniVLA - OpenVLA (본문 41.7)", np.mean(simpler_succ["UniVLA (Full)"][0]) - 1.0, 41.7, tol=0.1),
                          check("UniVLA - Octo-Base (본문 26.7)", np.mean(simpler_succ["UniVLA (Full)"][0]) - 16.0, 26.7, tol=0.1)]
res["V8_simpler_trials_note"] = ("과제당 시행 수는 분수로 추정: 2.8% = 2/72, 93.0% = 67/72. 72 = 3 seeds x 24 trials (논문 본문 24 trials 언급과 맞음). "
                                 "Octo-Small 평균 29.5 대 보고 30.0은 과제별 시행 수가 같다고 가정한 값이다.")

# ---------- V9. 실로봇 표 A-V: 시행 수 15 가정(백분율 분모 역산) ----------
N_REAL = 15
rw = {  # success % per task: store, clean, fold, hanoi; avg succ reported; avg score reported
    "DP": ([40.0, 33.3, 53.3, 6.7], 33.3, 1.45), "OpenVLA": ([40.0, 53.3, 33.3, 26.7], 38.3, 1.63),
    "LAPA(OXE)": ([60.0, 40.0, 33.3, 46.7], 45.0, 1.95), "UniVLA(Bridge)": ([66.7, 73.3, 33.3, 46.7], 55.0, 2.05),
    "UniVLA(OXE)": ([80.0, 73.3, 33.3, 60.0], 63.3, 2.35), "UniVLA(Full)": ([93.3, 100.0, 46.7, 86.7], 81.7, 2.63),
}
v9 = []
for m, (succ, rep_s, rep_sc) in rw.items():
    v9.append(check(f"Table A-V avg succ {m}", np.mean(succ), rep_s, tol=0.06))
res["V9_realworld_avg"] = v9
res["V9_realworld_note"] = ("UniVLA(OXE) 평균 성공률: 본문 표의 네 과제 평균은 61.65이고 보고값은 63.3이다. 표 A-V 값 자체를 점검하면 어긋난다.")
pool = {m: (round(np.mean(s) / 100 * 4 * N_REAL), 4 * N_REAL) for m, (s, _, _) in rw.items()}
res["V9_pooled_counts"] = {m: f"{k}/{n}" for m, (k, n) in pool.items()}


def fisher_p(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 1), round(100 * (c + h), 1)


res["V9_fisher_pooled"] = {
    "UniVLA(Full) vs LAPA(OXE)": fisher_p(pool["UniVLA(Full)"][0], 60, pool["LAPA(OXE)"][0], 60),
    "UniVLA(Full) vs OpenVLA": fisher_p(pool["UniVLA(Full)"][0], 60, pool["OpenVLA"][0], 60),
    "UniVLA(Full) vs DP": fisher_p(pool["UniVLA(Full)"][0], 60, pool["DP"][0], 60),
}
res["V9_wilson_full_pooled_60"] = wilson(pool["UniVLA(Full)"][0], 60)
res["V9_wilson_LAPA_pooled_60"] = wilson(pool["LAPA(OXE)"][0], 60)
res["V9_towel_fisher_UniVLA_vs_DP"] = fisher_p(7, 15, 8, 15)

# 과제별 Fisher 검정 (과제당 15회 가정): UniVLA(full) 대 LAPA, OpenVLA, DP
per_task_k = {"Store": (14, 9, 6, 6), "Clean": (15, 6, 5, 8), "Fold": (7, 5, 8, 5), "Hanoi": (13, 7, 1, 4)}
# 순서: UniVLA(full), LAPA(OXE), DP, OpenVLA
res["V9_per_task_fisher"] = {
    task: {"vs LAPA": round(fisher_p(u, 15, l, 15), 3), "vs DP": round(fisher_p(u, 15, d, 15), 3),
           "vs OpenVLA": round(fisher_p(u, 15, o, 15), 3)}
    for task, (u, l, d, o) in per_task_k.items()}

# 표 II 일반화 (시행 15 x 3 설정)
gen_pool = {"UniVLA": 31, "LAPA": 13, "DP": 11, "OpenVLA": 9}
res["V9_gen_fisher"] = {"UniVLA vs LAPA (45 trials each)": fisher_p(31, 45, 13, 45),
                        "UniVLA vs DP": fisher_p(31, 45, 11, 45)}

# ---------- V8b. SimplerEnv 전체 평균 검정: 과제 4개 x 72회 = 288회 가정 ----------
res["V8b_simpler_pooled_z"] = {}
for other in ["RoboVLM", "Octo-Small", "OpenVLA"]:
    p_u = np.mean(simpler_succ["UniVLA (Full)"][0]) / 100
    p_o = np.mean(simpler_succ[other][0]) / 100
    pp = (p_u + p_o) / 2
    se = math.sqrt(2 * pp * (1 - pp) / 288)
    z = (p_u - p_o) / se
    res["V8b_simpler_pooled_z"][f"UniVLA vs {other}"] = {"z": round(z, 2), "p": float(2 * (1 - stats.norm.cdf(abs(z))))}

# ---------- V10. LIBERO 시행 수 가정: suite당 500 (README: 50 trials x 10 tasks) ----------
N_LIB = 500
res["V10_wilson_libero_full_suite"] = {s: wilson(round(v / 100 * N_LIB), N_LIB)
                                       for s, v in zip(["Spatial", "Object", "Goal", "Long"], T1["UniVLA Full"][:4])}
pair_tests = {
    "Full vs OpenVLA (Goal)": (95.6, 79.2), "Full vs OpenVLA (Long)": (92.0, 53.7),
    "Ours vs w/o visual (Long)": (92.0, 86.0), "Ours vs w/o visual (Spatial)": (96.5, 95.0),
    "Ours vs AR (Long)": (92.0, 49.0),
    "history vs instr-only (Long)": (92.0, 88.1), "history vs instr-only (Goal)": (95.6, 95.0),
    "task-centric vs Genie (Long)": (79.4, 69.6), "task-centric vs Genie (Goal)": (90.2, 77.2),
}


def z_two_prop(p1, p2, n=N_LIB):
    p1, p2 = p1 / 100, p2 / 100
    pp = (p1 + p2) / 2
    se = math.sqrt(2 * pp * (1 - pp) / n)
    z = (p1 - p2) / se
    return round(z, 2), float(2 * (1 - stats.norm.cdf(abs(z))))


res["V10_pairwise_z_n500_single_run"] = {k: z_two_prop(*v) for k, v in pair_tests.items()}
res["V10_note"] = ("두 비율의 z 검정은 독립 시행과 suite당 500회 한 번 실행을 가정한다. "
                   "논문은 3 seed 평균이라 실제 분산은 이보다 클 수 있다. 비율 차이에 대한 z 검정 p는 참고용이다.")

# seed 기반 t 구간 (README 표의 ± 값은 3 seed 표준편차로 가정). UniVLA full 대 OpenVLA의 Welch
seeds = {"UniVLA Full Spatial": (96.5, 0.5), "UniVLA Full Object": (96.8, 0.5), "UniVLA Full Goal": (95.6, 0.4),
         "UniVLA Full Long": (92.0, 1.0), "OpenVLA Spatial": (84.7, 0.9), "OpenVLA Object": (88.4, 0.8),
         "OpenVLA Goal": (79.2, 1.0), "OpenVLA Long": (53.7, 1.3)}
t_crit = stats.t.ppf(0.975, 2)
res["V10_seed_ci_t_df2"] = {k: (round(m - t_crit * s / math.sqrt(3), 2), round(m + t_crit * s / math.sqrt(3), 2))
                           for k, (m, s) in seeds.items()}
w = stats.ttest_ind_from_stats(96.5, 0.5, 3, 84.7, 0.9, 3, equal_var=False)
res["V10_welch_full_vs_openvla_spatial"] = {"t": round(float(w.statistic), 2), "p": float(w.pvalue)}

# ---------- V11. 데이터 혼합: 논문 표 A-I 비율과 코드 가중치 비율 ----------
# 코드 가중치는 샘플링 가중치이며 balance_weights 여부와 데이터셋 크기에 따라 실제 비율이 달라진다. 직접 비교는 불가능하다.
code_w = {"fractal": 0.5, "kuka": 0.1, "bridge": 1.0, "taco": 2.0, "jaco": 1.0, "cable": 1.0, "roboturk": 2.0,
          "viola": 2.0, "autolab_ur5": 2.0, "toto": 1.0, "language_table": 0.1, "hydra": 2.0, "buds": 1.0,
          "nyu_franka": 3.0, "furniture": 0.1, "ucsd": 2.0, "sailor": 1.0, "sirius": 1.0, "dlr": 1.0, "iamlab": 1.0,
          "mutex": 1.0, "fanuc": 2.0, "cmu_stretch": 1.0, "bc_z": 0.2, "fmb": 1.0, "dobbe": 0.2,
          "gnm_recon": 2.0, "gnm_cory_hall": 2.0, "gnm_sac_son": 2.0, "ego4d_1": 1.0, "ego4d_2": 1.0,
          "ego4d_3": 1.0, "ego4d_4": 1.0}
tot = sum(code_w.values())
res["V11_mix"] = {"entries": len(code_w), "total_weight": tot,
                  "ego4d_share_pct": round(100 * sum(v for k, v in code_w.items() if k.startswith("ego4d")) / tot, 2),
                  "gnm_share_pct": round(100 * sum(v for k, v in code_w.items() if k.startswith("gnm")) / tot, 2),
                  "fractal_share_pct": round(100 * code_w["fractal"] / tot, 2),
                  "paper_table_A1_fractal_pct": 13.9, "paper_table_A1_ego4d_pct": 3.0,
                  "paper_table_A1_gnm_sum_pct": round(8.9 + 2.3 + 3.5, 1)}

# ---------- V12. 디코더 파라미터 수 (정적 계산: prismatic/models/policy/transformer_utils.py MAPBlock 정의와 ActionDecoder 구조) ----------
def lin(i, o, bias=True):
    return i * o + (o if bias else 0)


rms = 512
pool_params = (lin(4096, 512)          # projection (vis_dim 4096 -> 512)
               + 512                   # latents (n_latents=1)
               + rms                   # attn_norm (RMSNorm weight)
               + lin(512, 512, False) + lin(512, 1024, False) + lin(512, 512)  # MAPAttention q, kv, proj
               + rms                   # mlp_norm
               + lin(512, 4096) + lin(2048, 512))  # SwishGLU (512 -> 2*2048) and out projection
decoder = 2 * pool_params + lin(512, 7 * 12)  # latent pool + visual pool + action projection (window 12, 7-DoF)
res["V12_decoder_params"] = {"per_pool": pool_params, "total": decoder, "million": round(decoder / 1e6, 2),
                             "paper_body_and_tableA6_M": 12.6, "paper_intro_M": 10.8}

# ---------- V13. 본문 내부 수치 ----------
res["V13_internal"] = [
    check("Pretraining compute ratio 960/21500 (본문 '1/20 미만')", 960 / 21500, 0.05, tol=0.01),
    check("Fig 9 Bridge->OpenX LIBERO (본문 '2.0%' 추가 상승 대 Bridge->Full 2.7)", 95.2 - 92.5, 2.0, tol=0.05),
    check("Fig 9 real-world Full 2.64 대 표 A-V 2.63", 2.63, 2.64, tol=0.006),
    check("OpenVLA 2567 어휘 (본문) 대 164", 164 / 2567, 0.064, tol=0.01),
]

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else str(x)),
               encoding="utf-8")
bad = []


def walk(obj, path=""):
    if isinstance(obj, dict):
        if "ok" in obj and not obj["ok"]:
            bad.append((path, obj["label"], obj["computed"], obj["reported"]))
        for k, v in obj.items():
            walk(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk(v, f"{path}[{i}]")


walk(res)
print("wrote", OUT)
print("mismatches:")
for b in bad:
    print("  ", b)
