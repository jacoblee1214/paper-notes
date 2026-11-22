"""ADEPT (arXiv:2608.19182 v2) 표 수치 재계산과 일관성 검사.

실행: python3 analysis/recompute.py  → analysis/recompute_results.json (numpy, scipy 필요, 수 초)
모든 입력은 paper/paper.txt 의 표를 손으로 옮긴 것이다. V1–V10 은 03_analysis.md 의 검사 번호와 같다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
R = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


# ---------------------------------------------------------------- V1 표 4 (실로봇 10회 × 5조건)
STAGES = ["Reach", "Grasp", "Lifting", "Reorient", "Align", "Insert"]
T4 = {
    "Kuka vision / FMB star": [10, 9, 8, 8, 7, 5],
    "Kuka vision / FMB square-round": [10, 8, 6, 4, 3, 3],
    "Flexiv vision / FMB square-round": [10, 7, 5, 3, 3, 3],
    "Flexiv visuo-tactile / FMB square-round": [10, 10, 10, 9, 8, 8],
    "Kuka vision / dish": [10, 10, 8, 7, 6, 6],
}
v1 = {}
for name, row in T4.items():
    drops = [row[i] - row[i + 1] for i in range(5)]
    cond = [round(row[i + 1] / row[i], 2) if row[i] else None for i in range(5)]
    v1[name] = {
        "monotone_nonincreasing": all(row[i] >= row[i + 1] for i in range(5)),
        "drops": dict(zip(STAGES[1:], drops)),
        "conditional_pass": dict(zip(STAGES[1:], cond)),
        "insert_wilson95_pct": wilson(row[-1], 10),
        "grasp_wilson95_pct": wilson(row[1], 10),
    }
pool_kuka_fmb = (5 + 3, 20)
pool_kuka_all = (5 + 3 + 6, 30)
v1["pooled"] = {
    "Kuka vision FMB (star+sq/round) insert": [pool_kuka_fmb[0], pool_kuka_fmb[1],
                                              round(100 * pool_kuka_fmb[0] / 20, 1), wilson(*pool_kuka_fmb)],
    "Kuka vision all three tasks insert": [pool_kuka_all[0], pool_kuka_all[1],
                                          round(100 * 14 / 30, 1), wilson(*pool_kuka_all)],
    "all 50 real trials with insert success": [5 + 3 + 3 + 8 + 6, 50, round(100 * 25 / 50, 1), wilson(25, 50)],
    "total_real_trials": 50,
}
R["V1_table4"] = v1


# ---------------------------------------------------------------- V2 표 4 쌍별 검정
def fisher(a, b, n=10):
    t = [[a, n - a], [b, n - b]]
    odds, p = stats.fisher_exact(t, alternative="two-sided")
    return round(p, 4)


def diff_ci(a, na, b, nb, z=1.959964):
    """Newcombe hybrid score 구간 (a/na - b/nb), %p"""
    l1, u1 = [x / 100 for x in wilson(a, na)]
    l2, u2 = [x / 100 for x in wilson(b, nb)]
    p1, p2 = a / na, b / nb
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return [round(100 * lo, 1), round(100 * hi, 1)]


v2 = {}
for label, a, b in [
    ("Flexiv 촉각 대 비전 (Insert 8 대 3)", 8, 3),
    ("Flexiv 촉각 대 비전 (Grasp 10 대 7)", 10, 7),
    ("Flexiv 촉각 대 비전 (Lifting 10 대 5)", 10, 5),
    ("Flexiv 촉각 대 비전 (Reorient 9 대 3)", 9, 3),
    ("Kuka star 대 square/round (Insert 5 대 3)", 5, 3),
    ("Kuka sq/round 대 Flexiv 비전 sq/round (Insert 3 대 3)", 3, 3),
    ("Kuka dish 대 Kuka sq/round (Insert 6 대 3)", 6, 3),
]:
    v2[label] = {"diff_pp": round(100 * (a - b) / 10, 1), "newcombe95_pp": diff_ci(a, 10, b, 10), "fisher_p": fisher(a, b)}


# 검정력: 참 성공률 p1 대 p2, n=10 씩, Fisher 양측 α=0.05 (정확 계산)
def power(p1, p2, n=10, alpha=0.05):
    pw = 0.0
    for a in range(n + 1):
        for b in range(n + 1):
            if fisher(a, b, n) < alpha:
                pw += stats.binom.pmf(a, n, p1) * stats.binom.pmf(b, n, p2)
    return round(pw, 3)


v2["power_n10"] = {"0.8 대 0.3": power(0.8, 0.3), "0.8 대 0.5": power(0.8, 0.5), "0.5 대 0.3": power(0.5, 0.3)}
# 구간 반폭 ±10 %p 를 위해 필요한 시행 수 (p = 0.5 최악)
v2["n_for_halfwidth_10pp_at_p05"] = math.ceil((1.959964 / 0.10) ** 2 * 0.25)
R["V2_pairwise"] = v2

# ---------------------------------------------------------------- V3 시뮬 대 실로봇
sim_vs_real = {}
for label, k, n, psim in [
    ("Kuka vision sq/round: 실 3/10 대 시뮬 학생 46.7%", 3, 10, 0.467),
    ("Kuka vision star: 실 5/10 대 시뮬 학생 65.2%", 5, 10, 0.652),
]:
    bt = stats.binomtest(k, n, psim)
    sim_vs_real[label] = {"exact_binomial_two_sided_p": round(bt.pvalue, 3),
                          "P(X<=k)": round(stats.binom.cdf(k, n, psim), 3)}
sim_vs_real["Flexiv vision/tactile, dish: 시뮬 학생 수치 없음(보고 안 됨)"] = None
R["V3_sim_vs_real"] = sim_vs_real

# ---------------------------------------------------------------- V4 표 2 정수 가능성
ADR = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
T2 = {
    "Kuka": [98.6, 94.1, 84.7, 78.6, 71.9, 67.9, 58.7, 52.5, 39.6, 18.0, 0.00],
    "Flexiv": [94.5, 87.1, 80.3, 76.8, 74.9, 70.4, 66.8, 59.2, 52.3, 22.9, 0.70],
}


def feasible(val, n, dec):
    lo, hi = (val - 0.5 * 10 ** -dec), (val + 0.5 * 10 ** -dec)
    return [k for k in range(n + 1) if lo <= 100 * k / n < hi or (abs(100 * k / n - lo) < 1e-9)]


v4 = {"monotone": {k: all(v[i] >= v[i + 1] for i in range(10)) for k, v in T2.items()}}
bad = {}
for emb, vals in T2.items():
    for lvl, val in zip(ADR, vals):
        dec = 2 if (emb == "Kuka" and lvl == 50) or (emb == "Flexiv" and lvl == 50) else 1
        f1024 = feasible(val, 1024, dec)
        f1000 = feasible(val, 1000, dec)
        if not f1024:
            bad[f"{emb} ADR {lvl}: {val}"] = {"n=1024 가능한 k": f1024, "n=1000 가능한 k": f1000}
v4["n=1024로 표기 정밀도에서 불가능한 칸"] = bad
v4["Flexiv ADR50 0.70 의 후보"] = {"7/1024": round(100 * 7 / 1024, 4), "8/1024": round(100 * 8 / 1024, 4), "7/1000": 0.70}
v4["첫 칸 평균 하락(Kuka)"] = {"ADR0->25": round(T2['Kuka'][0] - T2['Kuka'][5], 1), "ADR35->50": round(T2['Kuka'][7] - T2['Kuka'][10], 1)}
# 시뮬 평가 비율의 정수 가능성 (1024 에피소드, 소수 첫째 자리)
v4["sim_eval_pct_feasible_n1024"] = {str(v): feasible(v, 1024, 1) for v in (85.0, 89.2, 46.7, 65.2)}
R["V4_table2"] = v4

# ---------------------------------------------------------------- V5 표 1 (시드 평균 ± SD, 시드 수 미기재)
T1 = {
    "Kuka": {"primitive": (0.73, 0.003), "fmb": (0.76, 0.003), "visdex": (0.77, 0.011)},
    "Flexiv": {"primitive": (0.64, 0.007), "fmb": (0.58, 0.011), "visdex": (0.61, 0.015)},
}
v5 = {}
for emb, d in T1.items():
    base, bsd = d["primitive"]
    for k in ("fmb", "visdex"):
        m, sd = d[k]
        v5[f"{emb} {k} - primitive"] = {
            "diff": round(m - base, 3),
            "diff_in_pooled_sd_units": round((m - base) / math.sqrt((sd ** 2 + bsd ** 2) / 2), 1),
        }
v5["note"] = "시드 수와 ±의 종류(SD 또는 SEM)가 표에 없어 검정 불가. 단위만 환산."
R["V5_table1"] = v5

# ---------------------------------------------------------------- V6 표 3
rows = {
    "a ADEPT full": dict(adr=5, sr=(46.0, 1.1), t=(19.9, 1.0), nt=5),
    "b No BC": dict(adr=4, sr=(38.5, 1.2), t=(35.2, 4.9), nt=4),
    "c No warm-up": dict(adr=4, sr=(28.4, 3.7), t=(20.2, 1.5), nt=4),
    "f Loose clip": dict(adr=5, sr=(46.8, 2.0), t=(17.6, 1.8), nt=5),
    "g No BC/WU remapped": dict(adr=0, sr=(26.6, 7.9), t=None, nt=0),
    "g+ No BC/WU append": dict(adr=0, sr=(27.0, 10.7), t=None, nt=0),
}
v6 = {"text_arithmetic": {
    "critic warm-up SR gain (a-c)": round(46.0 - 28.4, 1),
    "BC time ratio 19.9/35.2": round(19.9 / 35.2, 3),
    "BC time reduction %": round(100 * (1 - 19.9 / 35.2), 1),
    "loose clip time ratio 17.6/19.9": round(17.6 / 19.9, 3),
    "loose clip SR diff (f-a)": round(46.8 - 46.0, 1),
}}


def welch(m1, s1, n1, m2, s2, n2, sem=False):
    if sem:
        s1, s2 = s1 * math.sqrt(n1), s2 * math.sqrt(n2)
    r = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
    return round(float(r.pvalue), 4)


comps = [
    ("SR: (a) 대 (c) 워밍업 제거", "a ADEPT full", "c No warm-up", "sr"),
    ("SR: (a) 대 (b) BC 제거", "a ADEPT full", "b No BC", "sr"),
    ("SR: (a) 대 (f) 느슨한 클립", "a ADEPT full", "f Loose clip", "sr"),
    ("시간: (a) 대 (b) BC 제거", "a ADEPT full", "b No BC", "t"),
    ("시간: (a) 대 (c) 워밍업 제거", "a ADEPT full", "c No warm-up", "t"),
    ("시간: (a) 대 (f) 느슨한 클립", "a ADEPT full", "f Loose clip", "t"),
]
v6["welch_p"] = {}
for label, ra, rb, key in comps:
    a, b = rows[ra], rows[rb]
    na = 5 if key == "sr" else a["nt"]
    nb = 5 if key == "sr" else b["nt"]
    ma, sa = a[key]
    mb, sb = b[key]
    v6["welch_p"][label] = {"± 가 SD 일 때": welch(ma, sa, na, mb, sb, nb), "± 가 SEM 일 때": welch(ma, sa, na, mb, sb, nb, sem=True)}
v6["fisher_adr50_counts"] = {
    "5/5 대 4/5 (b 또는 c)": round(stats.fisher_exact([[5, 0], [4, 1]])[1], 3),
    "5/5 대 0/5 (d,e,g,h,i)": round(stats.fisher_exact([[5, 0], [0, 5]])[1], 4),
}
v6["ablation_structure"] = {
    "행 수": 10,
    "LR=1e-5 인 행": ["a", "b", "c", "f", "g", "g+"],
    "LR=1e-3 인 행": ["d", "e", "h", "i"],
    "중간 LR(1e-4 등) 행": 0,
    "LR=1e-5 에서 KL β=1 행(본문 0/5 주장)": "표에 없음",
    "표의 LR=1e-3 에서 BC·WU 없이 클립 .05 행": "없음",
}
R["V6_table3"] = v6

# ---------------------------------------------------------------- V7 표 6, 7 차원 합
# 필드 순서(표 6): dof pos, dof vel, hand pos, hand vel, actions, fabric, fingertip contacts, nominal delta,
# obj pos, obj rot, obj state, goal pos, goal quat, onehot, scale, pointcloud, receptacle pose, obj-recept. contact,
# hand ang vel, palm force, palm wrench, joint torque
t6 = {
    "S1 actor": [23, 23, 15, 15, 23, 69, 15, 0, 3, 4, 0, 3, 4, 1, 1, 192, 0, 0, 0, 0, 0, 0],
    "S1 critic": [23, 23, 15, 15, 23, 69, 15, 0, 0, 0, 13, 3, 4, 1, 1, 192, 0, 0, 15, 3, 0, 23],
    "S2 actor": [23, 23, 15, 15, 23, 69, 15, 0, 0, 0, 0, 3, 4, 0, 0, 192, 7, 3, 0, 0, 0, 0],
    "S2 critic": [23, 23, 15, 15, 23, 69, 15, 38, 0, 0, 13, 3, 4, 0, 0, 0, 7, 3, 0, 0, 6, 23],
    "S3 teacher": [23, 23, 15, 15, 23, 69, 15, 0, 0, 0, 0, 3, 4, 0, 0, 192, 7, 3, 0, 0, 0, 0],
    "S3 student": [23, 23, 15, 15, 23, 69, 0, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}
claimed6 = {"S1 actor": 391, "S1 critic": 438, "S2 actor": 392, "S2 critic": 280, "S3 teacher": 392, "S3 student": 206}
# 필드 순서(표 7): proprio, last actions, fabric, contact forces, contact palm, obj pos, obj rot, obj state, goal,
# goal quat, onehot, scale, pc, board pose, target contact, palm wrench, joint torque, tactile anchor
t7 = {
    "S1 actor": [94, 29, 87, 15, 3, 3, 4, 0, 3, 4, 1, 1, 192, 0, 0, 0, 0, 0],
    "S1 critic": [94, 29, 87, 15, 3, 0, 0, 13, 3, 4, 1, 1, 192, 0, 0, 6, 29, 0],
    "S2 actor": [94, 29, 87, 15, 3, 0, 0, 0, 3, 4, 0, 0, 192, 7, 3, 0, 0, 0],
    "S2 critic": [94, 29, 87, 15, 3, 0, 0, 13, 3, 4, 1, 1, 192, 7, 3, 6, 29, 0],
    "S3 teacher": [94, 29, 87, 15, 3, 0, 0, 0, 3, 4, 0, 0, 192, 7, 3, 0, 0, 0],
    "S3 student": [94, 0, 87, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15],
}
claimed7 = {"S1 actor": 436, "S1 critic": 477, "S2 actor": 437, "S2 critic": 487, "S3 teacher": 437, "S3 student": 196}
v7 = {"table6_kuka": {k: {"sum": sum(v), "claimed": claimed6[k], "match": sum(v) == claimed6[k]} for k, v in t6.items()},
      "table7_flexiv": {k: {"sum": sum(v), "claimed": claimed7[k], "match": sum(v) == claimed7[k]} for k, v in t7.items()}}
v7["actor_swap_kuka"] = {"removed": 3 + 4 + 1 + 1, "added": 7 + 3, "391-9+10": 391 - 9 + 10}
v7["actor_swap_flexiv"] = {"removed": 3 + 4 + 1 + 1, "added": 7 + 3, "436-9+10": 436 - 9 + 10}
R["V7_obs_dims"] = v7

# ---------------------------------------------------------------- V8 부록 C 계산, 누적 계산 비용
SEC_YR = 365 * 24 * 3600
v8 = {
    "eq29_flexiv_years": round(100 * 3600 * 75000 * 16 / (60 * SEC_YR), 2),
    "eq29_transitions": 100 * 3600 * 75000 * 16,
    "eq30_kuka_years": round(25 * 3600 * 1.2e6 / (60 * SEC_YR), 2),
    "eq30_kuka_transitions": 25 * 3600 * 1.2e6,
    "75000x16": 75000 * 16,
}
v8["kuka_per_member_B"] = round(v8["eq30_kuka_transitions"] / 16 / 1e9, 2)
v8["flexiv_per_member_B"] = round(v8["eq29_transitions"] / 16 / 1e9, 2)
v8["8B_x16_hours_at_1.2M_per_s"] = round(8e9 * 16 / 1.2e6 / 3600, 1)
v8["ablation_throughput_steps_per_s_4gpu"] = round(3e9 / (19.9 * 3600))
v8["ablation_throughput_per_gpu"] = round(3e9 / (19.9 * 3600) / 4)
v8["fig3_8B_steps_at_ablation_rate_hours"] = round(8e9 / (3e9 / (19.9 * 3600)) / 3600, 1)
R["V8_compute"] = v8

# K1: 손익분기 — 사전학습을 N 개 과제에 나눠 갚으려면
k1 = {}
for label, pre in [("논문 표기 (PBT 구성원 1개 = 8B)", 8e9),
                   ("누적(Kuka 25 h 합산 ≈ 1.08e11)", 1.08e11),
                   ("누적(Flexiv 100 h 합산 ≈ 4.32e11)", 4.32e11)]:
    k1[label] = {}
    for S in (9e9, 20e9, 50e9):
        # N 개 과제: pre + 3B N < S N  → N > pre/(S-3B)
        k1[label][f"scratch {int(S/1e9)}B"] = round(pre / (S - 3e9), 1)
k1["note"] = "scratch 가 ADR 50 에 도달하는 데 필요한 스텝 S(논문은 9B 에서 ADR 6 에 그침). N > 사전학습/(S-3B)."
R["K1_breakeven"] = k1

# ---------------------------------------------------------------- V9 표 5 산술, 학습률 감쇠
v9 = {
    "actor_lr_ratio": 1e-3 / 1e-5,
    "clip_ratio": 0.20 / 0.05,
    "warmup_steps_per_gpu_20iter_4096env_h16": 20 * 4096 * 16,
    "batch_per_gpu_post (4096 env x 16)": 4096 * 16,
    "batch_per_gpu_pre (4096 env x 32)": 4096 * 32,
    "minibatch_post 24576 나누어짐": (4096 * 16) % 24576 == 0,
    "65536/24576": round(65536 / 24576, 3),
    "minibatch_pre 16384 나누어짐": (4096 * 32) % 16384 == 0,
    "24576 = 4096 x": 24576 / 4096,
}
for g in (1, 4):
    ep = 3e9 / (4096 * 16 * g)
    v9[f"3B step 에서 epoch 수 (GPU {g}개)"] = round(ep)
    v9[f"그때까지 선형 감쇠 진행률 % (GPU {g}개, 200k epoch 기준)"] = round(100 * ep / 200000, 1)
    v9[f"그때 LR (최종 0 가정) ×1e-5 (GPU {g}개)"] = round(1 - ep / 200000, 3)
R["V9_hparams"] = v9

# ---------------------------------------------------------------- V10 속도비
v10 = {
    "극단 짝 20/10": 20 / 10,
    "극단 짝 70/5": 70 / 5,
    "교차 짝 20/5": 20 / 5,
    "교차 짝 70/10": 70 / 10,
    "프로젝트 페이지 짝 70/9": round(70 / 9, 1),
    "프로젝트 페이지 짝 20/5": 20 / 5,
}
R["V10_speedup"] = v10

# ---------------------------------------------------------------- V11 학생 BC 손실 (식 31 대 알고리즘 1)
rng = np.random.default_rng(0)
mu_s, mu_t = rng.normal(size=23), rng.normal(size=23)
sig_t = np.full(23, 0.14)
v11 = {}
for sig_s_scale in (1.0, 1.5):
    sig_s = sig_t * sig_s_scale
    eq31 = math.sqrt(np.sum(((mu_s - mu_t) / sig_t) ** 2)) + math.sqrt(np.sum(((sig_s - sig_t) / sig_t) ** 2))
    alg1 = math.sqrt(np.sum(((mu_s - mu_t) / sig_t) ** 2)) + float(np.sum((sig_s - sig_t) ** 2))
    v11[f"sigma_student/teacher={sig_s_scale}"] = {"식 31 (두 항 모두 제곱근)": round(eq31, 3), "알고리즘 1 (둘째 항 제곱 노름)": round(alg1, 3)}
v11["note"] = "부록 H 는 학생의 log-std 가 고정이라고 쓴다. 고정이면 σ 항은 기울기가 0 이라 두 정의의 차이는 학습에 영향이 없다."
R["V11_bc_loss"] = v11

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT)
for k, v in R.items():
    print("==", k)
    print(json.dumps(v, ensure_ascii=False)[:1400])
