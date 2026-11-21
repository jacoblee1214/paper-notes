"""TaF-Foundation (arXiv:2611.20718 v1) 보고 수치 재계산과 검정.

실행: python3 analysis/recompute.py  → analysis/recompute_results.json (수 초, CPU)
입력: 논문 표 값을 아래에 직접 옮겼다 (paper/paper.txt, pdftotext 기준). 학습·추론은 하지 않는다.
표 번호는 논문 본문 표 2–5, 부록 표 S1, S4–S6 기준이다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}

# ---------- 표 2: 핵심 벤치마크 (평균, SD; 3 시드) ----------
# 키: F1 단일프레임 힘 RMSE, T1 단일프레임 토크, F2 시간 힘, T2 시간 토크, RET1 검색 R@1, RET5 검색 R@5
T2 = {
    "DINOv2":    {"F1": (3.032, .053), "T1": (.0821, .0015), "F2": (1.642, .064), "T2": (.0413, .0003), "RET1": (14.00, .27), "RET5": (40.25, 2.56)},
    "Sparsh":    {"F1": (2.886, .155), "T1": (.0788, .0021), "F2": (1.579, .016), "T2": (.0364, .0022), "RET1": (14.21, .59), "RET5": (37.83, 1.01)},
    "T3":        {"F1": (2.820, .131), "T1": (.0780, .0035), "F2": (1.482, .019), "T2": (.0372, .0006), "RET1": (17.38, 1.05), "RET5": (51.52, 1.63)},
    "AnyTouch2": {"F1": (2.799, .203), "T1": (.0775, .0026), "F2": (1.515, .040), "T2": (.0387, .0026), "RET1": (21.27, .80), "RET5": (64.48, 5.04)},
    "HTT":       {"F1": (2.971, .024), "T1": (.0783, .0014), "F2": (1.438, .094), "T2": (.0343, .0008), "RET1": (14.78, .46), "RET5": (45.33, .73)},
    "Ours":      {"F1": (2.321, .084), "T1": (.0657, .0013), "F2": (1.237, .068), "T2": (.0322, .0016), "RET1": (34.38, 1.35), "RET5": (81.41, 2.77)},
}
BASE = [m for m in T2 if m != "Ours"]
LOWER_BETTER = {"F1", "T1", "F2", "T2"}
N_SEEDS = 3

# 논문 주장 값 (본문)
claimed = {
    "F1_rel_reduction_pct": 17.1, "T1_rel_reduction_pct": 15.2,
    "F2_rel_reduction_pct": 14.0, "T2_rel_reduction_pct": 6.1,
    "RET1_gain_pp": 13.11, "RET5_gain_pp": 16.93,
}
best = {}
for k in ["F1", "T1", "F2", "T2", "RET1", "RET5"]:
    vals = {m: T2[m][k][0] for m in BASE}
    if k in LOWER_BETTER:
        bm = min(vals, key=vals.get)
    else:
        bm = max(vals, key=vals.get)
    best[k] = bm
res["table2_strongest_baseline"] = best
rel = {}
for k in ["F1", "T1", "F2", "T2"]:
    b = T2[best[k]][k][0]
    o = T2["Ours"][k][0]
    rel[k] = round(100 * (b - o) / b, 2)
rel["RET1_gain_pp"] = round(T2["Ours"]["RET1"][0] - T2[best["RET1"]]["RET1"][0], 2)
rel["RET5_gain_pp"] = round(T2["Ours"]["RET5"][0] - T2[best["RET5"]]["RET5"][0], 2)
res["table2_recomputed_gain"] = rel
checks = []
pairs = [("F1", "17.1 %"), ("T1", "15.2 %"), ("F2", "14.0 %"), ("T2", "6.1 %")]
for k, s in pairs:
    checks.append({"metric": k, "claimed": s, "recomputed_pct": rel[k]})
checks.append({"metric": "RET1", "claimed": "+13.11 pp", "recomputed_pp": rel["RET1_gain_pp"]})
checks.append({"metric": "RET5", "claimed": "+16.93 pp (그림 없음, 본문)", "recomputed_pp": rel["RET5_gain_pp"]})
res["table2_claim_check"] = checks

# Welch t 검정 (3 시드, 표준편차 ddof=1 가정). ddof=0 민감도도 같이 기록
def welch(m1, s1, m2, s2, n=N_SEEDS, ddof=1):
    # ddof=1: 입력 s는 표본 SD. ddof=0: 입력 s는 모집단 SD이므로 표본 분산 = s^2 * n/(n-1)
    v1 = (s1 ** 2) * n / (n - 1) if ddof == 0 else s1 ** 2
    v2 = (s2 ** 2) * n / (n - 1) if ddof == 0 else s2 ** 2
    se = math.sqrt(v1 / n + v2 / n)
    t = (m1 - m2) / se
    df = (v1 / n + v2 / n) ** 2 / ((v1 / n) ** 2 / (n - 1) + (v2 / n) ** 2 / (n - 1))
    p = 2 * stats.t.sf(abs(t), df)
    tcrit = stats.t.ppf(0.975, df)
    diff = m1 - m2
    return {"diff": round(diff, 4), "t": round(t, 2), "df": round(df, 2), "p": float(f"{p:.3g}"),
            "ci95": [round(diff - tcrit * se, 4), round(diff + tcrit * se, 4)]}

welch_tab = {}
for k in ["F1", "T1", "F2", "T2", "RET1", "RET5"]:
    welch_tab[k] = {}
    for m in BASE:
        o_m, o_s = T2["Ours"][k]
        b_m, b_s = T2[m][k]
        welch_tab[k][m] = welch(o_m, o_s, b_m, b_s)
res["table2_welch_ours_vs_baseline"] = welch_tab
# 민감도: ddof=0 (모집단 SD로 읽었을 때) 대비 가장 강한 기준선 두 개
res["table2_welch_ddof0_sensitivity"] = {
    k: welch(T2["Ours"][k][0], T2["Ours"][k][1], T2[best[k]][k][0], T2[best[k]][k][1], ddof=0)
    for k in ["F1", "RET1"]
}
# 시드 SD 대비 차이 (Ours의 SD 단위 거리)
res["table2_gap_in_ours_sd"] = {
    k: round((T2[best[k]][k][0] - T2["Ours"][k][0]) / T2["Ours"][k][1], 2) for k in ["F1", "T1", "F2", "T2"]
}

# ---------- 표 3: 절제 (단일 시드) ----------
T3 = {
    "Full":            {"F1": 2.415, "T1": .0648, "F2": 1.242, "T2": .0318, "RET1": 33.52, "RET5": 78.90, "D1": 1.581},
    "DirectSup":       {"F1": 2.494, "T1": .0684, "F2": 1.385, "T2": .0379, "RET1": 29.95, "RET5": 73.14, "D1": 1.648},
    "woAlign":         {"F1": 2.481, "T1": .0685, "F2": 1.310, "T2": .0370, "RET1": 28.00, "RET5": 76.10, "D1": 1.652},
    "woSensorMLP":     {"F1": 3.291, "T1": .0884, "F2": 1.490, "T2": .0384, "RET1": 25.90, "RET5": 67.81, "D1": 1.761},
    "woSensorIDToken": {"F1": 2.628, "T1": .0678, "F2": 1.291, "T2": .0323, "RET1": 33.90, "RET5": 78.10, "D1": 1.592},
}
abl = {}
seed_sd = {k: T2["Ours"][k][1] for k in ["F1", "T1", "F2", "T2", "RET1", "RET5"]}
for v in ["DirectSup", "woAlign", "woSensorMLP", "woSensorIDToken"]:
    d = {}
    for k in ["F1", "T1", "F2", "T2", "RET1", "RET5"]:
        delta = T3[v][k] - T3["Full"][k]
        d[k] = {"delta": round(delta, 4), "delta_in_seed_sd": round(delta / seed_sd[k], 2)}
    abl[v] = d
res["table3_ablation_delta_vs_full"] = abl
res["table3_full_vs_table2_ours"] = {
    k: {"table3_full": T3["Full"][k], "table2_ours_mean": T2["Ours"][k][0],
        "diff": round(T3["Full"][k] - T2["Ours"][k][0], 4)}
    for k in ["F1", "T1", "F2", "T2", "RET1", "RET5"]
}
# 표 3 "좋아진다" 모든 지표 확인
dd = {}
for v in ["DirectSup", "woAlign", "woSensorMLP", "woSensorIDToken"]:
    dd[v] = {k: (T3[v][k] > T3["Full"][k]) for k in T3[v]}  # True: 변형 값이 전체 모델보다 큼 (F·T·D1은 나쁨, R은 좋음)
res["table3_worse_than_full_flags"] = dd

# ---------- 표 4: 시뮬 (UniVTAC 3 과제, 과제당 50 에피소드, 단일 실행) ----------
N_SIM = 50
T4 = {
    "UniVTAC": [40, 64, 38], "T3": [30, 62, 32], "AnyTouch2": [32, 68, 28], "Ours": [44, 70, 40],
}
T4_avg_claim = {"UniVTAC": 47.33, "T3": 41.33, "AnyTouch2": 42.67, "Ours": 51.33}
res["table4_avg_recomputed"] = {m: round(float(np.mean(v)), 2) for m, v in T4.items()}
res["table4_avg_claim_match"] = {m: abs(round(float(np.mean(v)), 2) - T4_avg_claim[m]) < 0.01 for m, v in T4.items()}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


def fisher(k1, n1, k2, n2):
    tab = [[k1, n1 - k1], [k2, n2 - k2]]
    _, p = stats.fisher_exact(tab, alternative="two-sided")
    return float(f"{p:.3g}")


# 과제별 Fisher (에피소드 수 = 50)
sim_task = {}
for m in ["UniVTAC", "T3", "AnyTouch2"]:
    sim_task[m] = {}
    for i, task in enumerate(["LiftCan", "PullOutKey", "PutBottleInShelf"]):
        k_o = T4["Ours"][i] * N_SIM // 100
        k_b = T4[m][i] * N_SIM // 100
        sim_task[m][task] = {"ours": f"{k_o}/50", "base": f"{k_b}/50", "fisher_p": fisher(k_o, 50, k_b, 50)}
res["table4_per_task_fisher"] = sim_task
# 합산 (150 에피소드 대 150 에피소드). 과제 군집을 무시한 근사
sim_pool = {}
for m in ["UniVTAC", "T3", "AnyTouch2"]:
    k_o = sum(T4["Ours"]) * N_SIM // 100
    k_b = sum(T4[m]) * N_SIM // 100
    sim_pool[m] = {"ours": f"{k_o}/150", "base": f"{k_b}/150", "fisher_p_pooled_approx": fisher(k_o, 150, k_b, 150),
                   "ours_wilson95": wilson(k_o, 150), "base_wilson95": wilson(k_b, 150)}
res["table4_pooled_approx"] = sim_pool

# ---------- 표 5: 실로봇 (과제당 20 시행) ----------
N_RW = 20
T5 = {
    "UniVTAC":   {"LiftBottle": 15, "RemovePowerBank": 20, "KeyInsert": 10, "BoardClean": None},
    "T3":        {"LiftBottle": 20, "RemovePowerBank": 30, "KeyInsert": 5, "BoardClean": 35},
    "AnyTouch2": {"LiftBottle": 20, "RemovePowerBank": 25, "KeyInsert": 10, "BoardClean": 40},
    "Ours":      {"LiftBottle": 35, "RemovePowerBank": 60, "KeyInsert": 25, "BoardClean": 65},
}
rw = {}
for m, d in T5.items():
    succ = {t: (None if v is None else int(round(v * N_RW / 100))) for t, v in d.items()}
    rw[m] = {"succ_counts": succ,
             "avg3_percent": round(float(np.mean([d["LiftBottle"], d["RemovePowerBank"], d["KeyInsert"]])), 2),
             "avg4_percent": None if d["BoardClean"] is None else round(float(np.mean(list(d.values()))), 2)}
res["table5_avg_recomputed"] = rw
# 3과제 (모든 방법이 평가한 과제) 합산 대 최강 기준선
k3_o = sum(int(round(T5["Ours"][t] * N_RW / 100)) for t in ["LiftBottle", "RemovePowerBank", "KeyInsert"])
for m in ["T3", "AnyTouch2"]:
    k3_b = sum(int(round(T5[m][t] * N_RW / 100)) for t in ["LiftBottle", "RemovePowerBank", "KeyInsert"])
    res[f"table5_3task_fisher_ours_vs_{m}"] = {"ours": f"{k3_o}/60", "base": f"{k3_b}/60",
                                               "fisher_p": fisher(k3_o, 60, k3_b, 60),
                                               "ours_wilson95": wilson(k3_o, 60), "base_wilson95": wilson(k3_b, 60)}
# 4과제 (AnyTouch2만 4과제 전부 평가)
k4_o = sum(int(round(v * N_RW / 100)) for v in T5["Ours"].values())
k4_b = sum(int(round(v * N_RW / 100)) for v in T5["AnyTouch2"].values())
res["table5_4task_fisher_ours_vs_AnyTouch2"] = {"ours": f"{k4_o}/80", "base": f"{k4_b}/80",
                                                "fisher_p": fisher(k4_o, 80, k4_b, 80)}
# 과제별 Fisher (20 시행 대 20 시행)
rw_task = {}
for t in ["LiftBottle", "RemovePowerBank", "KeyInsert", "BoardClean"]:
    rw_task[t] = {}
    for m in ["T3", "AnyTouch2", "UniVTAC"]:
        if T5[m][t] is None or T5["Ours"][t] is None:
            continue
        ko = int(round(T5["Ours"][t] * N_RW / 100))
        kb = int(round(T5[m][t] * N_RW / 100))
        rw_task[t][m] = {"ours": f"{ko}/20", "base": f"{kb}/20", "fisher_p": fisher(ko, 20, kb, 20)}
res["table5_per_task_fisher"] = rw_task
# 검정력: 두 비율 차이 (정규 근사, 양측 0.05, 검정력 0.8)
def n_per_arm(p1, p2, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return math.ceil((num / abs(p1 - p2)) ** 2)
res["power_n_per_arm"] = {
    "diff_22pp_at_0.40_vs_0.18": n_per_arm(0.40, 0.18),
    "diff_22pp_at_0.46_vs_0.24": n_per_arm(0.4625, 0.2375),
    "diff_10pp_at_0.30_vs_0.20": n_per_arm(0.30, 0.20),
    "diff_22pp_at_0.40_vs_0.18_stated_per_task_n": 20,
}

# ---------- 표 S4–S6: few-shot (3 미지 센서, 단일 시드) ----------
S4_sf = {  # 단일프레임 joint, 5/10/25/50/100%
    "DINOv2": [.5453, .5265, .4943, .4486, .4718], "Sparsh": [.5810, .5483, .5156, .4932, .4810],
    "T3": [.5805, .5394, .4841, .4547, .4224], "AnyTouch2": [.5457, .5786, .5026, .4900, .4773],
    "HTT": [.5458, .5717, .5432, .5271, .5180], "Ours": [.5539, .5262, .4992, .4644, .4668],
}
S4_tmp = {
    "DINOv2": [.4485, .4014, .3862, .3862, .3414], "Sparsh": [.4439, .4123, .3901, .3767, .3520],
    "T3": [.4339, .4026, .3818, .3658, .3328], "AnyTouch2": [.4254, .4058, .3957, .3691, .3486],
    "HTT": [.4120, .3859, .3767, .3617, .3381], "Ours": [.4187, .3987, .3736, .3582, .3294],
}
S5_in = {  # 입력 전용 10%: (단일프레임, 시간)
    "DINOv2": (.5103, .4238), "Sparsh": (.5832, .4163), "T3": (.5311, .4079),
    "AnyTouch2": (.5905, .4423), "HTT": (.6414, .4030), "Ours": (.5046, .4203),
}
S6_claim = {"DINOv2": (.4524, .4487), "Sparsh": (.4729, .4661), "T3": (.4497, .4448),
            "AnyTouch2": (.4747, .4643), "HTT": (.4794, .4687), "Ours": (.4468, .4428)}
fs = {}
for m in S6_claim:
    A = np.mean(S4_sf[m]); B = np.mean(S4_tmp[m]); C = (S5_in[m][0] + S5_in[m][1]) / 2
    hier = (A + B + C) / 3
    eq = (sum(S4_sf[m]) + sum(S4_tmp[m]) + S5_in[m][0] + S5_in[m][1]) / 12
    fs[m] = {"hier_recomputed": round(float(hier), 4), "hier_claim": S6_claim[m][0],
             "eq_recomputed": round(float(eq), 4), "eq_claim": S6_claim[m][1],
             "match": (round(float(hier), 4) == S6_claim[m][0]) and (round(float(eq), 4) == S6_claim[m][1])}
res["fewshot_S6_recompute"] = fs
# 시간 결합 최저 여부 (본문 주장: 25/50/100%에서 Ours 최저, 5/10%는 HTT)
res["fewshot_temporal_best_by_budget"] = {
    p: min(S4_tmp, key=lambda m: S4_tmp[m][i]) for i, p in enumerate(["5%", "10%", "25%", "50%", "100%"])
}
res["fewshot_singleframe_best_by_budget"] = {
    p: min(S4_sf, key=lambda m: S4_sf[m][i]) for i, p in enumerate(["5%", "10%", "25%", "50%", "100%"])
}
res["fewshot_hier_margin_ours_vs_T3"] = round(S6_claim["T3"][0] - S6_claim["Ours"][0], 4)


# 부록 C.4: 적응 프레임 수와 단일프레임 스텝 수 (식 S11, 올림). 논문 값: 282, 563, 1407, 2813, 5000
import math as _m
pool = 72000
steps_claim = [282, 563, 1407, 2813, 5000]
steps = [min(5000, _m.ceil(20 * int(pool * p) / 256)) for p in (0.05, 0.10, 0.25, 0.50, 1.00)]
res["fewshot_sf_steps"] = {"recomputed": steps, "claim": steps_claim, "match": steps == steps_claim}

# ---------- 부록 표 S1: 데이터 인벤토리 합계 ----------
S1 = [  # (episodes, frames)
    (2366, 6555482), (2098, 6190543), (849, 3126433), (1640, 4881827), (922, 3755271), (2088, 6236062),
    (1427, 4245576), (858, 1809205), (131, 231435), (343, 798315), (847, 1775553), (886, 2604158),
    (1445, 3020400), (974, 2184516), (8, 360000), (8, 324000), (2871, 16165846), (406, 2324372),
    (804, 4762455), (1031, 6008741), (905, 5270261), (24, 144000), (2972, 8700502), (4600, 9823901),
]
ep = sum(e for e, _ in S1); fr = sum(f for _, f in S1)
res["s1_sum_check"] = {"rows": len(S1), "episodes": ep, "episodes_claim": 30503,
                       "frames": fr, "frames_claim": 101298854}
# 평가 세트 규모 (부록 B.3): 21 센서 × 3 객체 × 3 에피소드 × 3000 프레임
res["eval_set_check"] = {"frames": 21 * 3 * 3 * 3000, "frames_claim": 567000,
                         "episodes": 21 * 3 * 3, "episodes_claim": 189}
# 검색 우연 수준: 후보 20개 → R@1 = 5%, R@5 = 25%
res["retrieval_chance"] = {"RET1_percent": 100 / 20, "RET5_percent": 100 * 5 / 20,
                           "n_trials": 21 * 100, "RET1_wilson95_at_34.38": wilson(round(0.3438 * 2100), 2100)}
# 분할 비율 합
res["split_sum"] = round(89.65 + 2.45 + 7.90, 2)
# 표 S4 단일프레임 joint 100%에서 T3가 Ours보다 낮음 (본문 일관성 점검)
res["singleframe_100pct_T3_minus_Ours"] = round(S4_sf["T3"][4] - S4_sf["Ours"][4], 4)

# 표 1 샘플 수 대비 (10^8 표기 확인): 본문 "over 100 million" vs 프레임 합
res["frames_total_millions"] = round(fr / 1e6, 2)

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print("wrote", OUT)
for k in ["table2_claim_check", "table2_gap_in_ours_sd", "table4_avg_claim_match", "fewshot_S6_recompute",
          "s1_sum_check", "eval_set_check", "power_n_per_arm", "table5_3task_fisher_ours_vs_T3",
          "table5_3task_fisher_ours_vs_AnyTouch2", "table5_4task_fisher_ours_vs_AnyTouch2",
          "table4_pooled_approx", "table3_ablation_delta_vs_full", "retrieval_chance",
          "fewshot_temporal_best_by_budget", "fewshot_singleframe_best_by_budget", "fewshot_hier_margin_ours_vs_T3",
          "singleframe_100pct_T3_minus_Ours", "table3_worse_than_full_flags"]:
    print(k, json.dumps(res[k], ensure_ascii=False, default=str)[:900])
