"""LAPT-Scale (arXiv:2610.13732 v1) 보고 수치의 재계산.

V1 평균과 표 간 일치, V2 정수 가능성, V3 유의성과 검정력, V4 표 사이 교차 검사와 산술(비용, 합계, 배치),
V5 LAM 지표(PSNR을 MSE 비로 환산, 절제의 비가산성)를 계산해 recompute_results.json에 저장한다.

실행: python3 analysis/recompute.py     (numpy, scipy 필요, 약 10초)
"""
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def feasible_k(pct, n, dec=1):
    """pct(소수 dec자리 반올림 결과)가 되는 정수 k 목록."""
    out = []
    for k in range(n + 1):
        if round(100 * k / n + 1e-9, dec) == round(pct, dec):
            out.append(k)
    return out


def two_prop_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return float(2 * (1 - stats.norm.cdf(abs(z))))


def power_fisher(p1, p2, n, sims=4000, seed=0):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(sims):
        a = rng.binomial(n, p1)
        b = rng.binomial(n, p2)
        if fisher(a, n, b, n) < 0.05:
            hits += 1
    return hits / sims


def n_for_power(p1, p2, power=0.8, alpha=0.05):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    n = ((za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p1 - p2) ** 2
    return math.ceil(n)


# ---------------------------------------------------------------- 보고된 값 (논문 표)
T3 = {  # 표 3, 실로봇 3과제 (Mouse, Headphones, Cup), 과제당 20회
    "pi05": [55.0, 40.0, 75.0, 56.7],
    "JoyAI-RA": [80.0, 95.0, 65.0, 80.0],
    "LAPT-Scale": [85.0, 95.0, 75.0, 85.0],
    "DiffusionPolicy": [50.0, 40.0, 45.0, 45.0],
}
T15 = {  # 표 15, (감독 방식, 시간) -> [Mouse, Headphones, Cup, Avg]
    ("hand", 1): [45.0, 80.0, 50.0, 58.3], ("hand", 3): [80.0, 45.0, 100.0, 75.0], ("hand", 5): [75.0, 50.0, 40.0, 55.0],
    ("dreamdojo", 1): [60.0, 55.0, 70.0, 61.7], ("dreamdojo", 3): [80.0, 85.0, 60.0, 75.0], ("dreamdojo", 5): [65.0, 75.0, 90.0, 76.7],
    ("lapt", 1): [50.0, 70.0, 70.0, 63.3], ("lapt", 3): [70.0, 75.0, 95.0, 80.0], ("lapt", 5): [85.0, 95.0, 75.0, 85.0],
}
T14 = {"hand": [55.0, 58.3, 75.0, 55.0], "dreamdojo": [55.0, 61.7, 75.0, 76.7], "lapt": [55.0, 63.3, 80.0, 85.0]}
T6 = {  # 표 6 = 표 19, 4과제 x 20회 = 영역당 80회
    "DiffusionPolicy": [60.0, 20.0, 15.0, 15.0, 27.5],
    "pi05": [85.0, 50.0, 45.0, 40.0, 55.0],
    "LAPT-Scale": [85.0, 75.0, 70.0, 70.0, 75.0],
}
T1 = {"pi0": (65.9, 58.4), "pi05": (82.7, 76.8), "Motus": (88.7, 87.0), "LingBot-VLA": (88.6, 86.7),
      "JoyAI-RA": (88.6, 87.2), "LAPT-Scale": (90.0, 89.3)}
T2 = {"GR00T-N1.6": 47.6, "Qwen3PI": 43.9, "ABot-M0": 58.3, "JoyAI-RA": 63.2, "LAPT-Scale": 64.8}
T5 = {  # (PSNR, LARY, RT-H)
    "MSE-only": (33.9844, 0.675, 74.2), "w/o LPIPS": (34.2105, 0.691, 82.5), "w/o DINO": (34.1500, 0.698, 84.1),
    "w/o flow": (34.3012, 0.685, 77.8), "Full": (34.4217, 0.713, 89.3)}
T7 = {"VQ-LAM": (18.3416, 0.6741, 0.3443, 0.232), "MSE-only": (33.9844, 0.9234, 0.0769, 0.675), "LAPT-Scale-LAM": (34.4217, 0.9326, 0.0546, 0.713)}
T17 = {
    ("TF", "LAPA"): (18.3416, 0.6741, 0.3443), ("TF", "VILLA-X"): (26.9595, 0.7973, 0.1685),
    ("TF", "MSE-only"): (33.9844, 0.9234, 0.0769), ("TF", "LAPT-Scale-LAM"): (34.4217, 0.9326, 0.0546),
    ("AR", "LAPA"): (15.8607, 0.5551, 0.4593), ("AR", "VILLA-X"): (22.5951, 0.7008, 0.3228),
    ("AR", "MSE-only"): (28.3290, 0.8426, 0.1768), ("AR", "LAPT-Scale-LAM"): (28.8291, 0.8510, 0.1347)}

# ---------------------------------------------------------------- V1
v1 = {}
rows = {}
for name, v in T3.items():
    rows[f"T3/{name}"] = {"reported_avg": v[3], "computed": round(sum(v[:3]) / 3, 2)}
for key, v in T15.items():
    rows[f"T15/{key[0]}-{key[1]}K"] = {"reported_avg": v[3], "computed": round(sum(v[:3]) / 3, 2)}
for name, v in T6.items():
    rows[f"T6/{name}"] = {"reported_avg": v[4], "computed": round(sum(v[:4]) / 4, 2)}
v1["avg_checks"] = rows
v1["avg_mismatch_gt_0.06"] = [k for k, r in rows.items() if abs(r["reported_avg"] - r["computed"]) > 0.06]
# 표 14 = 표 15의 평균
t14_vs_t15 = {}
for sup in ("hand", "dreamdojo", "lapt"):
    t14_vs_t15[sup] = {"T14": T14[sup][1:], "T15": [T15[(sup, h)][3] for h in (1, 3, 5)]}
v1["t14_vs_t15_equal"] = all(r["T14"] == r["T15"] for r in t14_vs_t15.values())
# 표 3의 LAPT = 표 15의 LAPT 5K, 표 4 = 표 14 5K, 표 16 None/standard = 표 14 0h
v1["T3_LAPT_equals_T15_lapt_5K"] = T3["LAPT-Scale"] == T15[("lapt", 5)]
v1["T4_equals_T14"] = {"none": 55.0, "T14_0h": T14["lapt"][0], "lapt5K": 85.0, "T14_5K": T14["lapt"][3]}
v1["T6_equals_T19"] = True  # 값이 모두 같음(손으로 대조, 텍스트 비교 완료)
v1["improve_over_JoyAI_RA_per_task"] = {"Mouse": 85 - 80, "Head": 95 - 95, "Cup": 75 - 65, "avg": 85 - 80}
v1["scaling_gain_lapt_1K_to_5K"] = round(85.0 - 63.3, 1)
v1["gain_lapt_5K_vs_0h"] = round(85.0 - 55.0, 1)
v1["gain_lapt_1K_vs_0h"] = round(63.3 - 55.0, 1)
OUT["V1"] = v1

# ---------------------------------------------------------------- V2: 정수 가능성
v2 = {}
# 20회 칸
cells20 = [x for v in T3.values() for x in v[:3]] + [x for v in T15.values() for x in v[:3]]
v2["cells20_all_multiple_of_5"] = all(abs(x / 5 - round(x / 5)) < 1e-9 for x in cells20)
# 평균 셀(n=60): k 집합
avg60 = {}
for name, val in {**{f"T3/{k}": v[3] for k, v in T3.items()}, **{f"T15/{k[0]}-{k[1]}K": v[3] for k, v in T15.items()},
                  **{f"T14/{k}-{i}": x for k, v in T14.items() for i, x in enumerate(v)}, "T16/extended": 53.3}.items():
    ks = feasible_k(val, 60)
    avg60[name] = {"pct": val, "k_of_60": ks}
v2["avg_n60"] = avg60
v2["avg_n60_infeasible"] = [k for k, r in avg60.items() if not r["k_of_60"]]
# 표 6: 영역당 80, 평균은 320
v2["T6_n80_k"] = {f"{m}/{d}": feasible_k(x, 80) for m, v in T6.items() for d, x in zip(["seen", "pos", "ill", "bg"], v[:4])}
v2["T6_avg_n320_k"] = {m: feasible_k(v[4], 320) for m, v in T6.items()}
v2["T6_n80_infeasible"] = [k for k, r in v2["T6_n80_k"].items() if not r]
# RoboCasa-GR1: 24과제 x 50 = 1,200
v2["T2_n1200_k"] = {m: feasible_k(x, 1200) for m, x in T2.items()}
v2["T2_n1200_infeasible"] = [m for m, r in v2["T2_n1200_k"].items() if not r]
# RoboTwin: 과제 수와 과제당 시행 수 미기재. 50과제 x 100회 가정(5,000회)이면 소수 첫째 자리는 모두 가능
rt = {}
for m, (e, h) in T1.items():
    rt[m] = {"easy_k5000": feasible_k(e, 5000)[:1] + feasible_k(e, 5000)[-1:], "hard_k5000": feasible_k(h, 5000)[:1] + feasible_k(h, 5000)[-1:]}
v2["T1_n5000_assumed"] = rt
# 표 5 RT-H의 소수 첫째 자리와 정수 k (5,000회 가정 / 과제 50개 평균 가정)
v2["T5_RTH_vs_T1_hard_same"] = T5["Full"][2] == T1["LAPT-Scale"][1]
# 표 1 JoyAI-RA와 LAPT 차이
v2["T1_diff"] = {"easy_vs_best_baseline(Motus 88.7)": round(90.0 - 88.7, 1), "hard_vs_best_baseline(JoyAI-RA 87.2)": round(89.3 - 87.2, 1),
                 "easy_vs_JoyAI_RA": round(90.0 - 88.6, 1), "hard_vs_JoyAI_RA": round(89.3 - 87.2, 1)}
v2["T2_diff_vs_JoyAI_RA"] = round(64.8 - 63.2, 1)
v2["T1_T2_note"] = "표 1·2의 기준선 값은 인용(Baseline sources는 부록 참조라고 했으나 부록 C에 출처 없음)"
OUT["V2"] = v2

# ---------------------------------------------------------------- V3: 유의성, 검정력
v3 = {}


def k_of(pct, n):
    return int(round(pct * n / 100))


def cmp(label, p1, n1, p2, n2):
    k1, k2 = k_of(p1, n1), k_of(p2, n2)
    return {"label": label, "k1": k1, "n1": n1, "k2": k2, "n2": n2, "fisher_p": round(fisher(k1, n1, k2, n2), 4),
            "wilson1": wilson(k1, n1), "wilson2": wilson(k2, n2), "diff_pts": round(100 * (k1 / n1 - k2 / n2), 1)}


main = {}
main["LAPT_vs_JoyAI"] = cmp("표 3 실로봇 3과제 합계 51/60 대 48/60", 85.0, 60, 80.0, 60)
main["LAPT_vs_pi05"] = cmp("표 3 51/60 대 34/60", 85.0, 60, 56.7, 60)
main["LAPT_vs_DP"] = cmp("표 3 51/60 대 27/60", 85.0, 60, 45.0, 60)
main["JoyAI_vs_pi05"] = cmp("표 3 48/60 대 34/60", 80.0, 60, 56.7, 60)
for i, task in enumerate(["Mouse", "Head", "Cup"]):
    main[f"LAPT_vs_JoyAI_{task}"] = cmp(f"{task} 20회", T3["LAPT-Scale"][i], 20, T3["JoyAI-RA"][i], 20)
v3["main_real_robot"] = main
v3["JoyAI_gap_power"] = {"power_85_vs_80_n60": power_fisher(0.85, 0.80, 60), "n_per_arm_for_80pct_power": n_for_power(0.85, 0.80)}

sc = {}
sc["lapt_1K_vs_5K"] = cmp("LAPT 1K 대 5K (38/60 대 51/60)", 85.0, 60, 63.3, 60)
sc["lapt_0_vs_5K"] = cmp("LAPT 0h 대 5K (33/60 대 51/60)", 85.0, 60, 55.0, 60)
sc["lapt_0_vs_1K"] = cmp("LAPT 1K 대 0h", 63.3, 60, 55.0, 60)
sc["lapt_3K_vs_5K"] = cmp("LAPT 3K 대 5K", 85.0, 60, 80.0, 60)
sc["lapt_1K_vs_3K"] = cmp("LAPT 1K 대 3K", 80.0, 60, 63.3, 60)
sc["lapt_vs_dreamdojo_5K"] = cmp("5K LAPT 대 DreamDojo", 85.0, 60, 76.7, 60)
sc["lapt_vs_dreamdojo_3K"] = cmp("3K", 80.0, 60, 75.0, 60)
sc["lapt_vs_dreamdojo_1K"] = cmp("1K", 63.3, 60, 61.7, 60)
sc["lapt_vs_hand_5K"] = cmp("5K LAPT 대 손 동작", 85.0, 60, 55.0, 60)
sc["hand_3K_vs_5K"] = cmp("손 동작 3K 대 5K (감소)", 55.0, 60, 75.0, 60)
sc["hand_5K_vs_0h"] = cmp("손 동작 5K 대 0h (같은 55.0)", 55.0, 60, 55.0, 60)
sc["dreamdojo_0_vs_5K"] = cmp("DreamDojo 5K 대 0h", 76.7, 60, 55.0, 60)
sc["stage2_extended_vs_standard"] = cmp("표 16: 연장 53.3 대 표준 55.0", 53.3, 60, 55.0, 60)
sc["lapt_5K_vs_compute_matched"] = cmp("표 16: LAPT 5K 대 연장 대조", 85.0, 60, 53.3, 60)
for i, task in enumerate(["Mouse", "Head", "Cup"]):
    sc[f"lapt_vs_dreamdojo_5K_{task}"] = cmp(f"5K {task}", T15[("lapt", 5)][i], 20, T15[("dreamdojo", 5)][i], 20)
    sc[f"lapt_1K_vs_5K_{task}"] = cmp(f"LAPT {task} 1K 대 5K", T15[("lapt", 5)][i], 20, T15[("lapt", 1)][i], 20)
v3["scaling"] = sc
# 추세 검정 (Cochran-Armitage), LAPT 1K, 3K, 5K의 성공 수 38, 48, 51 / 60
def cochran_armitage(ks, ns, scores):
    ks, ns, s = map(np.array, (ks, ns, scores))
    N, R = ns.sum(), ks.sum()
    p = R / N
    T = np.sum(s * (ks - ns * p))
    var = p * (1 - p) * (np.sum(ns * s ** 2) - np.sum(ns * s) ** 2 / N)
    z = T / math.sqrt(var)
    return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


for sup in ("lapt", "dreamdojo", "hand"):
    ks = [k_of(T15[(sup, h)][3], 60) for h in (1, 3, 5)]
    z, p = cochran_armitage(ks, [60, 60, 60], [1, 3, 5])
    v3.setdefault("trend_1_3_5K", {})[sup] = {"k": ks, "z": round(z, 3), "p": round(p, 4)}
# 과제별 단조성
mono = {}
for sup in ("lapt", "dreamdojo", "hand"):
    for i, task in enumerate(["Mouse", "Head", "Cup"]):
        seq = [T15[(sup, h)][i] for h in (1, 3, 5)]
        mono[f"{sup}/{task}"] = {"seq": seq, "monotone_nondecreasing": seq[0] <= seq[1] <= seq[2]}
v3["per_task_monotone"] = mono
v3["monotone_count_of_9"] = sum(r["monotone_nondecreasing"] for r in mono.values())

# 표 6 (영역당 80회, 4과제를 합산한 독립 시행으로 가정)
t6 = {}
for d, name in enumerate(["seen", "position", "illumination", "background"]):
    t6[name] = {
        "LAPT_vs_pi05": cmp(name, T6["LAPT-Scale"][d], 80, T6["pi05"][d], 80),
        "LAPT_vs_DP": cmp(name, T6["LAPT-Scale"][d], 80, T6["DiffusionPolicy"][d], 80),
    }
shift_l = sum(k_of(T6["LAPT-Scale"][d], 80) for d in (1, 2, 3))
shift_p = sum(k_of(T6["pi05"][d], 80) for d in (1, 2, 3))
t6["shifted_pooled"] = {"LAPT": [shift_l, 240], "pi05": [shift_p, 240], "fisher_p": fisher(shift_l, 240, shift_p, 240),
                        "wilson_LAPT": wilson(shift_l, 240), "wilson_pi05": wilson(shift_p, 240)}
# 시프트 대 seen의 하락(LAPT, pi05)
t6["drop_seen_to_shift"] = {"LAPT": round(85 - shift_l / 240 * 100, 1), "pi05": round(85 - shift_p / 240 * 100, 1)}
t6["avg_row_includes_seen"] = {"LAPT_avg_shift_only": round(100 * shift_l / 240, 1), "pi05_avg_shift_only": round(100 * shift_p / 240, 1)}
t6["multiplicity_note"] = "영역 3개 x 비교 2개 = 6개 검정, Bonferroni 기준 p < 0.0083"
v3["table6_generalization"] = t6

# 시뮬레이션 (시행 수 가정)
sim = {}
sim["RoboCasa_n1200"] = {"LAPT_vs_JoyAI_two_prop_p": round(two_prop_z(778, 1200, 758, 1200), 4),
                         "fisher_p": round(fisher(778, 1200, 758, 1200), 4), "wilson_LAPT": wilson(778, 1200),
                         "note": "k = 778, 758은 64.8, 63.2%에 대응하는 정수 하나(후보 777–778, 758). 과제 간 군집을 무시한 낙관적 검정"}
sim["RoboTwin_assumed_n5000"] = {
    "easy_LAPT_vs_Motus_p": round(fisher(4500, 5000, 4435, 5000), 4),
    "hard_LAPT_vs_JoyAI_p": round(fisher(4465, 5000, 4360, 5000), 4),
    "wilson_hard_LAPT": wilson(4465, 5000),
    "note": "50과제 x 100회 가정(논문에 시행 수 없음). 표 1의 기준선 값은 인용이라 같은 시행 구성이라는 보장도 없음"}
sim["RoboTwin_ablation_assumed_n5000"] = {m: wilson(int(round(v[2] * 50)), 5000) for m, v in T5.items()}
v3["simulation_benchmarks"] = sim

# Holm 보정: 표 14 주요 쌍 6개
pair_ps = {
    "LAPT 1K vs 5K": sc["lapt_1K_vs_5K"]["fisher_p"], "LAPT 0h vs 5K": sc["lapt_0_vs_5K"]["fisher_p"],
    "LAPT vs hand 5K": sc["lapt_vs_hand_5K"]["fisher_p"], "LAPT vs DreamDojo 5K": sc["lapt_vs_dreamdojo_5K"]["fisher_p"],
    "LAPT vs DreamDojo 3K": sc["lapt_vs_dreamdojo_3K"]["fisher_p"], "LAPT 3K vs 5K": sc["lapt_3K_vs_5K"]["fisher_p"]}
order = sorted(pair_ps.items(), key=lambda x: x[1])
holm = {}
m = len(order)
prev = 0.0
for i, (k, p) in enumerate(order):
    adj = min(1.0, max(prev, (m - i) * p))
    prev = adj
    holm[k] = {"p": p, "holm_adj": round(adj, 4)}
v3["holm_scaling_pairs"] = holm
OUT["V3"] = v3

# ---------------------------------------------------------------- V4: 산술과 표 사이 일치
v4 = {}
src = {"Egocentric-10K": (10.0, 0.0), "EgoDex": (0.7, 0.5), "Ego4D": (1.8, 0.8), "Xperience": (1.1, 0.4), "EgoVerse": (1.2, 0.5), "Self": (8.5, 2.9)}
raw = round(sum(v[0] for v in src.values()), 1)
cl = round(sum(v[1] for v in src.values()), 1)
v4["T8_sum_raw_hours_K"] = raw
v4["T8_sum_cleaned_hours_K"] = cl
v4["T8_share_cleaned_self_collected"] = round(2.9 / 5.1, 3)
v4["T8_share_raw_egocentric10K"] = round(10.0 / 23.3, 3)
v4["T8_share_raw_self"] = round(8.5 / 23.3, 3)
v4["T8_cleaned_over_raw_by_source"] = {k: (round(v[1] / v[0], 2) if v[0] else None) for k, v in src.items()}
v4["fig2_factory_pct_equals_egocentric10K_share_pct"] = [42.9, round(100 * 10.0 / 23.3, 1)]
v4["policy_pretrain_subset_excludes_largest_LAM_source"] = True
# 비용
lam_hours = 1_000_000 / 1.8 / 3600
v4["LAM_train_hours"] = round(lam_hours, 1)
v4["LAM_gpu_hours"] = round(lam_hours * 32, 0)
v4["LAM_batch_total"] = 16 * 32
v4["LAM_samples_seen_M"] = 16 * 32 * 1_000_000 / 1e6
v4["LAM_epochs_at_10M_samples"] = 16 * 32 * 1_000_000 / 10_000_000
v4["annotation_gpu_hours"] = 5100 / 3
v4["stage1_gpu_hours"] = 43 * 32
v4["stage2_gpu_hours"] = 50 * 32
v4["pipeline_one_run_gpu_hours"] = round(lam_hours * 32 + 5100 / 3 + 43 * 32 + 50 * 32)
v4["stage1_samples_M"] = 1024 * 100_000 / 1e6
v4["batch_products"] = {"stage1": 32 * 32, "posttrain": 8 * 8}
# 논문이 보고한 모든 변형을 같은 레시피로 다시 만들 때의 하한(가정)
one_lam = lam_hours * 32
v4["rebuild_all_lams_lower_bound_gpu_hours"] = {
    "LAMs_trained(Full, MSE-only, w/o LPIPS, w/o DINO, w/o flow, VQ)": 6,
    "LAM_training_only": round(6 * one_lam),
    "note": "VQ-LAM은 표 17에서 LAPA와 값이 같아 별도 학습이 아닐 수 있음. DreamDojo LAM과 VILLA-X는 공개 사전학습 모델 사용 가정"}
# 표 5의 변형마다 정책 파이프라인 전체 재학습(주석 1,700 + Stage1 1,376 + Stage2 1,600) 가정
per_variant_after_lam = 5100 / 3 + 43 * 32 + 50 * 32
v4["per_variant_after_lam_gpu_hours"] = round(per_variant_after_lam)
v4["table5_variants_total_gpu_hours_if_full_retrain"] = round(4 * (one_lam + per_variant_after_lam))
# 암시되는 MFU(대략): LAM ~0.6B 파라미터, 토큰 601/프레임쌍, 6ND
params = 2 * 24 * 12 * 1024 ** 2
tokens = 2 * (320 // 16) * (240 // 16) + 1
flops = 6 * params * tokens * 16 * 32 * 1_000_000
v4["LAM_params_est_B"] = round(params / 1e9, 2)
v4["LAM_tokens_per_pair"] = tokens
v4["LAM_train_flops_e21"] = round(flops / 1e21, 2)
v4["LAM_implied_MFU_vs_990TFLOPs"] = round(flops / (one_lam * 3600 * 990e12), 3)
v4["annotation_frames_per_s_per_gpu_if_30fps"] = round(3 * 3600 * 30 / 3600, 1)
# D.1 시행 수
v4["hard_tasks_trials_total_per_method"] = 4 * 4 * 20
v4["RoboCasa_rollouts"] = 24 * 50
v4["RoboCasa_k_for_64.8"] = feasible_k(64.8, 1200)
# 클러스터링 수치
v4["fig9_windows_per_episode"] = round(2_360_000 / 11_981, 1)
v4["fig9_frames_per_episode_at_stride16"] = round(2_360_000 / 11_981 * 16, 0)
v4["fig9_percentile70_windows"] = round(2_360_000 * 0.3)
# Stage-1: 에폭 추정(가정: 30 fps, 정책 샘플 = 30 전이 청크 = 1초, 비중첩)
for h in (1, 3, 5, 5.1):
    secs = h * 1000 * 3600
    v4.setdefault("stage1_epochs_if_1s_nonoverlapping_chunks", {})[str(h)] = round(1024 * 100_000 / secs, 1)
# 표 7 대 표 17 동일성
v4["T7_VQLAM_equals_T17_LAPA"] = T7["VQ-LAM"][:3] == T17[("TF", "LAPA")]
v4["T7_MSEonly_equals_T17_TF_MSEonly"] = T7["MSE-only"][:3] == T17[("TF", "MSE-only")]
v4["T7_LAPT_equals_T17_TF_LAPT"] = T7["LAPT-Scale-LAM"][:3] == T17[("TF", "LAPT-Scale-LAM")]
v4["T5_MSEonly_Full_equal_T7"] = (T5["MSE-only"][0] == T7["MSE-only"][0] and T5["MSE-only"][1] == T7["MSE-only"][3] and
                                  T5["Full"][0] == T7["LAPT-Scale-LAM"][0] and T5["Full"][1] == T7["LAPT-Scale-LAM"][3])
v4["T5_drop_from_full_RTH"] = {k: round(T5["Full"][2] - v[2], 1) for k, v in T5.items()}
v4["T5_sum_single_drops_vs_all_removed"] = {"sum_single": round(sum(T5["Full"][2] - T5[k][2] for k in ("w/o LPIPS", "w/o DINO", "w/o flow")), 1),
                                            "all_removed": round(T5["Full"][2] - T5["MSE-only"][2], 1)}
# 표 5의 PSNR 순위와 RT-H 순위
names = list(T5)
psnr_rank = sorted(names, key=lambda k: -T5[k][0])
rth_rank = sorted(names, key=lambda k: -T5[k][2])
lary_rank = sorted(names, key=lambda k: -T5[k][1])
v4["T5_rank_psnr"] = psnr_rank
v4["T5_rank_lary"] = lary_rank
v4["T5_rank_rth"] = rth_rank
rho_psnr = stats.spearmanr([T5[k][0] for k in names], [T5[k][2] for k in names])
rho_lary = stats.spearmanr([T5[k][1] for k in names], [T5[k][2] for k in names])
v4["T5_spearman_psnr_vs_rth"] = [round(float(rho_psnr[0]), 3), round(float(rho_psnr[1]), 3)]
v4["T5_spearman_lary_vs_rth"] = [round(float(rho_lary[0]), 3), round(float(rho_lary[1]), 3)]
v4["T5_pearson_psnr_vs_rth"] = round(float(stats.pearsonr([T5[k][0] for k in names], [T5[k][2] for k in names])[0]), 3)
v4["T5_pearson_lary_vs_rth"] = round(float(stats.pearsonr([T5[k][1] for k in names], [T5[k][2] for k in names])[0]), 3)
OUT["V4"] = v4

# ---------------------------------------------------------------- V5: PSNR → MSE 비
v5 = {}
def mse_ratio(db_num, db_den):
    """MSE_num / MSE_den (PSNR이 높을수록 MSE가 작다)."""
    return 10 ** ((db_den - db_num) / 10)


v5["full_vs_mseonly_dB"] = round(34.4217 - 33.9844, 4)
v5["full_vs_mseonly_mse_ratio"] = round(mse_ratio(34.4217, 33.9844), 4)
v5["full_vs_mseonly_mse_reduction_pct"] = round(100 * (1 - mse_ratio(34.4217, 33.9844)), 2)
v5["full_vs_wo_flow_dB"] = round(34.4217 - 34.3012, 4)
v5["wo_flow_vs_mseonly_dB"] = round(34.3012 - 33.9844, 4)
v5["vq_vs_mseonly_dB"] = round(33.9844 - 18.3416, 4)
v5["vq_mse_over_mseonly"] = round(10 ** ((33.9844 - 18.3416) / 10), 1)
v5["ar_drop_dB"] = {k[1]: round(T17[("TF", k[1])][0] - T17[k][0], 3) for k in T17 if k[0] == "AR"}
v5["ar_gap_full_vs_mseonly_dB"] = {"TF": round(34.4217 - 33.9844, 4), "AR": round(28.8291 - 28.3290, 4)}
v5["lary_gain_full_vs_mseonly"] = round(0.713 - 0.675, 3)
v5["lary_chance_if_20_classes_unknown"] = "LARY 클래스 수 논문에 없음 (본문에서 확인 불가)"
v5["lpips_ratio_full_vs_mseonly_TF"] = round(0.0546 / 0.0769, 3)
v5["ssim_gain_TF"] = round(0.9326 - 0.9234, 4)
v5["T18_gt_vs_pred_dB"] = round(30.7705 - 25.5371, 4)
OUT["V5"] = v5

(HERE / "recompute_results.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", HERE / "recompute_results.json")
