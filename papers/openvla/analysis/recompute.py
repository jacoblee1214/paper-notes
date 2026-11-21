"""OpenVLA (arXiv:2406.09246 v3) 표 수치를 다시 계산하고 검정한다.

입력: 논문 부록 표의 값을 아래에 그대로 옮겨 적었다(출처 표 번호 주석). 외부 데이터는 쓰지 않는다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 성공 횟수는 부분 점수(0.5)를 포함한 값이다. 표 4 캡션이 부분 성공 0.5를 허용한다고 적는다.
- 성공률의 ± 값은 표준오차(SE)라고 가정하고, 어떤 공식이 맞는지 여기서 확인한다(§2).
- A/B 평가라 같은 과제와 초기 상태를 쓴다. 과제별 쌍 검정(Wilcoxon)과 시도 단위 비율 검정(z)을 함께 본다.
  시도별 원자료는 없으므로 쌍 검정은 과제 수준에서만 가능하다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}


def two_prop_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return p1 - p2, z, 2 * stats.norm.sf(abs(z))


def newcombe_diff_ci(k1, n1, k2, n2, z=1.959964):
    def wilson(k, n):
        p = k / n
        den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return c - h, c + h
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return lo, hi


# === 1. BridgeData V2 WidowX, 17 과제 x 10회 (표 4) ===
# 열: RT-1-X, Octo, RT-2-X, OpenVLA. 값은 성공 횟수(부분 점수 포함).
BRIDGE = [
    ("Put Eggplant into Pot (Easy)", [1, 5, 7, 10]),
    ("Put Eggplant into Pot", [0, 1, 5, 10]),
    ("Put Cup from Counter into Sink", [1, 1, 0, 7]),
    ("Put Eggplant into Pot (w/ Clutter)", [1, 3.5, 6, 7.5]),
    ("Put Yellow Corn on Pink Plate", [1, 4, 8, 9]),
    ("Lift Eggplant", [3, 0.5, 6.5, 7.5]),
    ("Put Carrot on Plate (w/ Height Change)", [2, 1, 4.5, 4.5]),
    ("Put Carrot on Plate", [1, 0, 1, 8]),
    ("Flip Pot Upright", [2, 6, 5, 8]),
    ("Lift AAA Battery", [0, 0, 2, 7]),
    ("Move Skull into Drying Rack", [1, 0, 5, 5]),
    ("Lift White Tape", [3, 0, 0, 1]),
    ("Take Purple Grapes out of Pot", [6, 0, 5, 4]),
    ("Stack Blue Cup on Pink Cup", [0.5, 0, 5.5, 4.5]),
    ("Put {Eggplant, Red Bottle} into Pot", [2.5, 4, 8.5, 7.5]),
    ("Lift {Cheese, Red Chili Pepper}", [1.5, 2.5, 8.5, 10]),
    ("Put {Blue Cup, Pink Cup} on Plate", [5, 5.5, 8.5, 9.5]),
]
# 범주 (표 4 Category 열)
CATS = ["Visual"] * 5 + ["Motion"] * 2 + ["Physical"] * 3 + ["Semantic"] * 4 + ["Language"] * 3
METHODS = ["RT-1-X", "Octo", "RT-2-X", "OpenVLA"]
B = np.array([r[1] for r in BRIDGE], dtype=float)  # 17 x 4
NB = 10
RES["bridge_mean_pct"] = dict(zip(METHODS, (B.sum(0) / (17 * NB) * 100).round(2).tolist()))
RES["bridge_reported_pct"] = {"RT-1-X": 18.5, "Octo": 20.0, "RT-2-X": 50.6, "OpenVLA": 70.6}

# 표준오차 공식 확인: 보고된 ± 값과 어떤 공식이 맞는지 본다
rep_se = {"RT-1-X": 2.7, "Octo": 2.6, "RT-2-X": 3.5, "OpenVLA": 3.2}
se_check = {}
for j, m in enumerate(METHODS):
    p = B[:, j].sum() / 170
    binom = math.sqrt(p * (1 - p) / 170) * 100
    task_sd = B[:, j].std(ddof=1) / NB  # 과제별 성공률의 SD
    se_task = task_sd / math.sqrt(17) * 100
    se_check[m] = {"reported": rep_se[m], "binomial_n170": round(binom, 2), "task_level_se": round(se_task, 2)}
RES["bridge_se_check"] = se_check

# 범주별 합계
cat_sum = {}
for c in ["Visual", "Motion", "Physical", "Semantic", "Language"]:
    idx = [i for i, cc in enumerate(CATS) if cc == c]
    n = len(idx) * NB
    cat_sum[c] = {m: round(B[idx, j].sum() / n * 100, 1) for j, m in enumerate(METHODS)}
RES["bridge_category_pct"] = cat_sum

# OpenVLA 대 RT-2-X 검정
k_ov, k_rt2 = B[:, 3].sum(), B[:, 2].sum()
d, z, p = two_prop_z(k_ov, 170, k_rt2, 170)
lo, hi = newcombe_diff_ci(k_ov, 170, k_rt2, 170)
diffs = B[:, 3] - B[:, 2]
w = stats.wilcoxon(B[:, 3], B[:, 2])
RES["bridge_ovla_vs_rt2x"] = {
    "diff_pp": round(d * 100, 1), "z": round(z, 2), "p_two_prop": float(f"{p:.2g}"),
    "newcombe95_pp": [round(lo * 100, 1), round(hi * 100, 1)],
    "task_wilcoxon_p": float(f"{w.pvalue:.2g}"), "tasks_ovla_better": int((diffs > 0).sum()),
    "tasks_rt2x_better": int((diffs < 0).sum()), "tasks_tie": int((diffs == 0).sum()),
}

# 방법별 OpenVLA 대 Octo, RT-1-X (참고)
for other in ["Octo", "RT-1-X"]:
    j = METHODS.index(other)
    d, z, p = two_prop_z(k_ov, 170, B[:, j].sum(), 170)
    RES[f"bridge_ovla_vs_{other}"] = {"diff_pp": round(d * 100, 1), "p_two_prop": float(f"{p:.2g}")}

# 17개 과제 최악 과제: OpenVLA가 RT-2-X보다 낮은 과제
RES["bridge_tasks_rt2x_better"] = [BRIDGE[i][0] for i in range(17) if diffs[i] < 0]

# === 2. Google robot, 12 과제 x 5회 (표 6) ===
GOOGLE = [
    ("Pick Coke Can", "ID", [5, 1, 5, 5]),
    ("Move Apple near Green Can", "ID", [3, 3, 3, 5]),
    ("Move Blue Chip Bag near Apple", "ID", [0, 3, 4, 5]),
    ("Place Coke Can Upright", "ID", [0, 0, 4, 4]),
    ("Open Middle Drawer", "ID", [0, 4, 2, 3]),
    ("Move Orange near Brown Chip Bag", "OOD", [1, 2, 5, 5]),
    ("Pick Pepsi Can", "OOD", [3, 0, 5, 4]),
    ("Pick Banana", "OOD", [5, 3, 5, 5]),
    ("Pick Green Cup", "OOD", [1, 0, 5, 5]),
    ("Place Apple on Plate", "OOD", [0, 0, 4, 4]),
    ("Place Banana in Pan", "OOD", [0, 0, 2, 4]),
    ("Move Coke Can near Taylor Swift", "OOD", [2, 0, 3, 2]),
]
G = np.array([g[2] for g in GOOGLE], dtype=float)
RES["google_mean_pct"] = dict(zip(METHODS, (G.sum(0) / 60 * 100).round(1).tolist()))
RES["google_reported_pct"] = {"RT-1-X": 33.3, "Octo": 26.7, "RT-2-X": 78.3, "OpenVLA": 85.0}
d, z, p = two_prop_z(G[:, 3].sum(), 60, G[:, 2].sum(), 60)
lo, hi = newcombe_diff_ci(G[:, 3].sum(), 60, G[:, 2].sum(), 60)
gd = G[:, 3] - G[:, 2]
RES["google_ovla_vs_rt2x"] = {
    "diff_pp": round(d * 100, 1), "z": round(z, 2), "p_two_prop": float(f"{p:.2g}"),
    "newcombe95_pp": [round(lo * 100, 1), round(hi * 100, 1)],
    "task_wilcoxon_p": float(stats.wilcoxon(G[:, 3], G[:, 2]).pvalue) if np.any(gd != 0) else 1.0,
}
ind = slice(0, 5)
ood = slice(5, 12)
RES["google_in_dist_pct"] = dict(zip(METHODS, (G[ind].sum(0) / 25 * 100).round(1).tolist()))
RES["google_ood_pct"] = dict(zip(METHODS, (G[ood].sum(0) / 35 * 100).round(1).tolist()))

# === 3. 29 과제 합산 (초록의 16.5%p) ===
pool_ov = (B[:, 3].sum() + G[:, 3].sum()) / 230
pool_rt2 = (B[:, 2].sum() + G[:, 2].sum()) / 230
RES["pooled29"] = {"OpenVLA_pct": round(pool_ov * 100, 2), "RT-2-X_pct": round(pool_rt2 * 100, 2),
                   "diff_pp": round((pool_ov - pool_rt2) * 100, 2)}

# === 4. Franka (표 7). 값은 % 이며 시도 수는 아래와 같다 ===
FR_METHODS = ["DP", "DP_matched", "Octo", "OpenVLA_scratch", "OpenVLA"]
FR = [  # (과제, 시도 수, [5개 방법 %])
    ("Carrot in Bowl ID", 10, [90, 80, 40, 70, 70]),
    ("Carrot in Bowl OOD", 5, [20, 0, 20, 0, 40]),
    ("Pour Corn ID", 10, [100, 90, 0, 10, 50]),
    ("Pour Corn OOD", 5, [80, 60, 0, 20, 60]),
    ("Flip Pot ID", 10, [100, 85, 40, 85, 100]),
    ("Flip Pot OOD", 5, [50, 20, 0, 40, 80]),
    ("Move onto Plate ID", 12, [25, 25, 41.7, 8.3, 75]),
    ("Move onto Plate OOD", 6, [8.3, 33.3, 8.3, 33.3, 58.3]),
    ("Knock Over ID", 12, [33.3, 25, 83.3, 75, 75]),
    ("Knock Over OOD", 6, [16.7, 16.7, 33.3, 58.3, 83.3]),
    ("Cover with Towel ID", 12, [16.7, 20.8, 91.7, 41.7, 50]),
    ("Cover with Towel OOD", 6, [16.7, 33.3, 91.7, 50, 50]),
]
FR_D = [
    ("Wipe Table ID (DROID)", 18, [50, 27.8, 52.8, 25, 55.6]),
    ("Wipe Table + Distractors OOD (DROID)", 12, [12.5, 25, 16.7, 16.7, 62.5]),
]
tab_n = sum(r[1] for r in FR)
tab_k = np.array([[r[2][j] / 100 * r[1] for j in range(5)] for r in FR]).sum(0)
tab_unw = np.array([r[2] for r in FR]).mean(0)
tab_rep = [48.5, 43.4, 43.4, 43.4, 67.2]
droid_n = sum(r[1] for r in FR_D)
droid_k = np.array([[r[2][j] / 100 * r[1] for j in range(5)] for r in FR_D]).sum(0)
droid_unw = np.array([r[2] for r in FR_D]).mean(0)
droid_rep = [35.0, 26.7, 38.3, 21.7, 58.3]
RES["franka_tabletop"] = {
    "trials": tab_n, "trial_weighted_pct": dict(zip(FR_METHODS, (tab_k / tab_n * 100).round(1).tolist())),
    "task_unweighted_pct": dict(zip(FR_METHODS, tab_unw.round(1).tolist())),
    "reported_avg_pct": dict(zip(FR_METHODS, tab_rep)),
    "min_task_pct": dict(zip(FR_METHODS, [min(r[2][j] for r in FR) for j in range(5)])),
}
RES["franka_droid"] = {
    "trials": droid_n, "trial_weighted_pct": dict(zip(FR_METHODS, (droid_k / droid_n * 100).round(1).tolist())),
    "task_unweighted_pct": dict(zip(FR_METHODS, droid_unw.round(1).tolist())),
    "reported_avg_pct": dict(zip(FR_METHODS, droid_rep)),
}
# 두 환경을 합친 값과 "20.4%p" 주장 후보
both_n = tab_n + droid_n
both_k = tab_k + droid_k
RES["franka_both"] = {
    "trial_pooled_pct": dict(zip(FR_METHODS, (both_k / both_n * 100).round(2).tolist())),
    "env_mean_of_reported": dict(zip(FR_METHODS, [round((a + b) / 2, 2) for a, b in zip(tab_rep, droid_rep)])),
    "ovla_minus_dp_pooled_pp": round((both_k[4] - both_k[0]) / both_n * 100, 2),
    "ovla_minus_dp_env_mean_pp": round(((67.2 + 58.3) / 2) - ((48.5 + 35.0) / 2), 2),
}
# OpenVLA 대 DP (시도 단위 비율 검정, 표준)
k_o, n_o = tab_k[4], tab_n
k_d, n_d = tab_k[0], tab_n
d, z, p = two_prop_z(k_o, n_o, k_d, n_d)
RES["franka_tabletop_ovla_vs_dp"] = {"diff_pp": round(d * 100, 1), "z": round(z, 2), "p_two_prop": float(f"{p:.2g}")}
# OpenVLA 최저 과제 성공률: 논문 "모든 과제에서 50% 이상인 유일한 방법" 주장 검사
RES["franka_min_task_claim_check"] = {
    m: min(r[2][j] for r in FR + FR_D) for j, m in enumerate(FR_METHODS)
}

# === 5. 파라미터 효율 미세조정 (표 8, 표 1) ===
PE = {  # 방법: [(과제, 시도, %)]
    "Full FT": [90, 40, 79.2, 41.7],
    "Last layer only": [40, 0, 33.3, 33.3],
    "Frozen vision": [40, 40, 50, 58.3],
    "Sandwich": [90, 0, 75, 41.7],
    "LoRA r=32": [60, 60, 75, 75],
    "LoRA r=64": [90, 40, 62.5, 66.7],
}
PE_N = [10, 5, 12, 6]
PE_rep = {"Full FT": 69.7, "Last layer only": 30.3, "Frozen vision": 47.0, "Sandwich": 62.1,
          "LoRA r=32": 68.2, "LoRA r=64": 68.2}
pe_calc = {}
for m, vals in PE.items():
    k = sum(v / 100 * n for v, n in zip(vals, PE_N))
    pe_calc[m] = {"calc_pct": round(k / 33 * 100, 1), "reported_pct": PE_rep[m]}
RES["peft_check"] = pe_calc
RES["peft_params_lora_r32_frac_pct"] = round(97.6 / 7188.1 * 100, 2)

# 전체 시도 33회에서 Full FT 대 LoRA r=32의 차이 (작은 표본)
k_full = sum(v / 100 * n for v, n in zip(PE["Full FT"], PE_N))
k_lora = sum(v / 100 * n for v, n in zip(PE["LoRA r=32"], PE_N))
d, z, p = two_prop_z(k_lora, 33, k_full, 33)
RES["peft_lora_vs_full"] = {"diff_pp": round(d * 100, 1), "p_two_prop": float(f"{p:.2g}")}

# === 6. 양자화 (표 5 = 표 2, 8개 과제 x 10회) 와 차단 제어 (표 11) ===
Q_TASKS = ["Put Eggplant (Easy)", "Put Eggplant", "Cup from Counter", "Lift Eggplant",
           "Carrot on Plate", "Lift AAA Battery", "Grapes out of Pot", "Eggplant + Red Bottle"]
Q = {  # 표 5: 열 bf16, int8, int4
    "bf16": [9, 7, 5, 6, 6, 7, 8, 9],
    "int8": [7, 7, 3, 4, 5, 5, 8, 7.5],
    "int4": [9, 7, 7, 7.5, 7, 3, 9, 8],
}
QB = {  # 표 11 (차단 제어): bf16, int8, int4
    "bf16": [10, 9, 5, 8, 10, 3, 2, 9],
    "int8": [10, 10, 5, 7, 10, 6, 2, 9.5],
    "int4": [10, 10, 3, 7.5, 10, 4, 2, 8.5],
}
# 표 4 부분집합 (OpenVLA 열, 같은 8과제) - 표 9의 OpenVLA 열과 같아야 한다
SUB_IDX = [0, 1, 2, 5, 7, 8, 12, 14]  # 표 4에서 같은 8과제 (인덱스는 BRIDGE 순서)
ov_subset = [BRIDGE[i][1][3] for i in SUB_IDX]
RES["quant_means"] = {k: round(sum(v) / 80 * 100, 1) for k, v in Q.items()}
RES["quant_reported"] = {"bf16": 71.3, "int8": 58.1, "int4": 71.9}
RES["blocking_means"] = {k: round(sum(v) / 80 * 100, 1) for k, v in QB.items()}
RES["blocking_reported"] = {"bf16": 70.0, "int8": 74.4, "int4": 68.8}
RES["ovla_subset8_from_table4_pct"] = round(sum(ov_subset) / 80 * 100, 2)
RES["ovla_subset8_task_names"] = [BRIDGE[i][0] for i in SUB_IDX]

k8, k_bf = sum(Q["int8"]), sum(Q["bf16"])
d, z, p = two_prop_z(k_bf, 80, k8, 80)
RES["quant_int8_vs_bf16_unpaired"] = {"diff_pp": round(d * 100, 1), "z": round(z, 2), "p_two_prop": float(f"{p:.2g}")}
RES["quant_int8_vs_bf16_paired_wilcoxon_p"] = float(f"{stats.wilcoxon(Q['bf16'], Q['int8']).pvalue:.2g}")
d, z, p = two_prop_z(sum(Q["int4"]), 80, k_bf, 80)
RES["quant_int4_vs_bf16_unpaired"] = {"diff_pp": round(d * 100, 1), "p_two_prop": float(f"{p:.2g}")}
RES["quant_int4_vs_bf16_paired_wilcoxon_p"] = float(f"{stats.wilcoxon(Q['bf16'], Q['int4']).pvalue:.2g}")

# 같은 정책 bf16의 보고값 불일치: 표 4 부분집합 76.3, 표 2/5 71.3, 표 11 70.0
RES["bf16_reported_across_tables"] = {"table4_subset_OpenVLA": 76.3, "table5_bf16_nonblocking": 71.3,
                                      "table11_bf16_blocking": 70.0}

# === 7. 표 9: OpenVLA-Bridge와 SigLIP 단일 인코더 ablation (8과제 x 10회) ===
AB = {
    "OpenVLA": [10, 10, 7, 7.5, 8, 7, 4, 7.5],
    "OpenVLA-Bridge": [8, 2, 4, 5.5, 4, 2, 3, 8],
    "OpenVLA-Bridge-SigLIP": [8, 3, 2, 6.5, 1, 2, 3, 7],
}
RES["table9_means"] = {k: round(sum(v) / 80 * 100, 1) for k, v in AB.items()}
RES["table9_reported"] = {"OpenVLA": 76.3, "OpenVLA-Bridge": 45.6, "OpenVLA-Bridge-SigLIP": 40.6}
d, z, p = two_prop_z(sum(AB["OpenVLA"]), 80, sum(AB["OpenVLA-Bridge"]), 80)
RES["abl_openx_effect"] = {"diff_pp": round(d * 100, 1), "p_two_prop": float(f"{p:.2g}"),
                           "paired_wilcoxon_p": float(f"{stats.wilcoxon(AB['OpenVLA'], AB['OpenVLA-Bridge']).pvalue:.2g}")}
d, z, p = two_prop_z(sum(AB["OpenVLA-Bridge"]), 80, sum(AB["OpenVLA-Bridge-SigLIP"]), 80)
RES["abl_dino_effect"] = {"diff_pp": round(d * 100, 1), "p_two_prop": float(f"{p:.2g}"),
                          "paired_wilcoxon_p": float(f"{stats.wilcoxon(AB['OpenVLA-Bridge'], AB['OpenVLA-Bridge-SigLIP']).pvalue:.2g}")}

# === 8. 표 10: 비전 인코더 동결 대 미세조정 ===
# 행: (과제, 시도, SigLIP 동결, SigLIP 미세조정, LLaVA 동결, LLaVA 미세조정). None은 "–"
T10 = [
    ("Put Eggplant into Pot", 10, 7, 10, 5, 9),
    ("Put Corn on Plate", 10, 10, 9, 0, 9),
    ("Put {Eggplant, Red Bottle} into Pot", 4, 2, 4, None, 3),
    ("Put {Blue Cup, Pink Cup} on Plate", 4, 0, 0, None, 0),
    ("Lift {Cheese, Red Chili Pepper}", 4, 0, 3, None, 2),
    ("Put {Strawberry, Lime} into Pot", 4, 1, 0, None, 3),
    ("Move {Sushi, Grapes}", 4, 3, 4, None, 3),
]
pair_fr = pair_ft = pair_n = 0
for name, n, sf, sft, lf, lft in T10:
    pair_fr += sf
    pair_ft += sft
    pair_n += n
    if lf is not None:
        pair_fr += lf
        pair_ft += lft
        pair_n += n
RES["table10_paired_frozen_pct"] = round(pair_fr / pair_n * 100, 1)
RES["table10_paired_ft_pct"] = round(pair_ft / pair_n * 100, 1)
RES["table10_paired_n"] = pair_n
RES["table10_siglip_only_frozen_pct"] = round(sum(r[2] for r in T10) / sum(r[1] for r in T10) * 100, 1)
RES["table10_siglip_only_ft_pct"] = round(sum(r[3] for r in T10) / sum(r[1] for r in T10) * 100, 1)
RES["table10_llava_frozen_pct"] = round((5 + 0) / 20 * 100, 1)
RES["table10_llava_ft_pct"] = round((9 + 9) / 20 * 100, 1)
RES["table10_claim_reported"] = {"frozen": 46.7, "ft": 80.0}

# === 9. 표 12 LIBERO: 표준오차 = sqrt(p(1-p)/1500) 확인 (3 시드 x 500회), 평균과 순위 ===
LIB = {
    "DP": [(78.3, 1.1, 3), (92.5, 0.7, 1), (68.3, 1.2, 3), (50.5, 1.3, 3), (72.4, 0.7, 2.5)],
    "Octo": [(78.9, 1.0, 2), (85.7, 0.9, 3), (84.6, 0.9, 1), (51.1, 1.3, 2), (75.1, 0.6, 2)],
    "OpenVLA": [(84.7, 0.9, 1), (88.4, 0.8, 2), (79.2, 1.0, 2), (53.7, 1.3, 1), (76.5, 0.6, 1.5)],
}
lib_se = {}
for m, rows in LIB.items():
    lib_se[m] = []
    for sr, se_rep, _ in rows[:4]:
        se_calc = math.sqrt((sr / 100) * (1 - sr / 100) / 1500) * 100
        lib_se[m].append({"sr": sr, "se_reported": se_rep, "se_binomial_n1500": round(se_calc, 2)})
RES["libero_se_check"] = lib_se
RES["libero_avg_check"] = {m: round(np.mean([r[0] for r in rows[:4]]), 2) for m, rows in LIB.items()}
RES["libero_avg_reported"] = {m: rows[4][0] for m, rows in LIB.items()}
RES["libero_rank_avg_check"] = {m: round(np.mean([r[2] for r in rows[:4]]), 2) for m, rows in LIB.items()}

# === 10. 부록 B.2 Google: 표 6 평균 확인 (위) 과 표 7 DROID 값 ===

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
if __name__ == "__main__":
    print(json.dumps(RES, ensure_ascii=False, indent=1))
