"""EgoEngine 논문 표 수치 재계산과 검정 (CPU, 수 초). 학습·시뮬레이션 실행 없음.

입력: 논문 표의 보고 값(arXiv:2606.12604v1, 표 1–4, 표 E.1)을 아래에 옮겨 적었다.
출력: analysis/recompute_results.json (숫자만), 표준 출력 요약.

시도 수(N)는 논문에 적혀 있지 않다. 그래서 (a) 보고 값이 k/N으로 맞아떨어지는 N을 찾고,
(b) 그 N으로 Wilson 구간과 Fisher 정확 검정을 계산한다. N 추정은 모두 "추정"으로 표시한다.
"""
import json
import math
from pathlib import Path

from scipy.stats import fisher_exact

OUT = Path(__file__).with_name("recompute_results.json")
results = {}


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def denominators(values, lo=1, hi=100, tol=0.005):
    """보고 값(소수 둘째 자리 반올림)을 모두 k/N으로 설명하는 N 목록."""
    ok = []
    for n in range(lo, hi + 1):
        if all(any(round(k / n, 2) == round(v, 2) for k in range(n + 1)) for v in values):
            ok.append(n)
    return ok


# ---- 표 3: 실로봇 과제 성공률 (Aria, 4 tasks) ----
t3 = {
    "Human Video": [0.00, 0.10, 0.00, 0.00],
    "Phantom": [0.00, 0.05, 0.00, 0.00],
    "Real Robot": [0.80, 0.80, 0.70, 0.25],
    "EgoEngine": [0.40, 0.35, 0.70, 0.60],
}
tasks = ["Mustard", "Drawer", "Flower", "Hammer"]
all_t3 = [v for row in t3.values() for v in row]
results["t3_denominator_candidates"] = denominators(all_t3, 1, 40)
N_REAL = 20  # 모든 값이 0.05의 배수이므로 20회 가정이 가장 자연스럽다 (추정)

real = {}
for method in ["EgoEngine", "Real Robot", "Phantom", "Human Video"]:
    real[method] = {}
    for i, t in enumerate(tasks):
        v = t3[method][i]
        k = round(v * N_REAL)
        real[method][t] = {"k": k, "n": N_REAL, "sr": round(k / N_REAL, 3), "wilson95": wilson(k, N_REAL)}
results["t3_real_per_task"] = real

fisher = {}
for i, t in enumerate(tasks):
    a = round(t3["EgoEngine"][i] * N_REAL)
    b = round(t3["Real Robot"][i] * N_REAL)
    _, p = fisher_exact([[a, N_REAL - a], [b, N_REAL - b]])
    fisher[t] = {"EgoEngine_vs_Real_p": round(p, 4), "diff": round((a - b) / N_REAL, 3)}
pooled_e = sum(round(v * N_REAL) for v in t3["EgoEngine"])
pooled_r = sum(round(v * N_REAL) for v in t3["Real Robot"])
n_all = 4 * N_REAL
_, p_pool = fisher_exact([[pooled_e, n_all - pooled_e], [pooled_r, n_all - pooled_r]])
fisher["pooled"] = {"EgoEngine": f"{pooled_e}/{n_all}", "Real": f"{pooled_r}/{n_all}",
                    "p": round(p_pool, 4), "EgoEngine_wilson": wilson(pooled_e, n_all),
                    "Real_wilson": wilson(pooled_r, n_all)}
results["t3_fisher_vs_real"] = fisher

# ---- 표 4: 어블레이션 (4 tasks 평균). 80회 = 4 x 20 (추정) ----
N_ABL = 4 * N_REAL
abl = {"Human Videos": 0.03, "+Visual branch": 0.05, "+Action branch": 0.43, "EgoEngine": 0.51}
abl_k = {"Human Videos": 2, "+Visual branch": 4, "+Action branch": 34, "EgoEngine": 41}
abl_check = {}
for name, reported in abl.items():
    k = abl_k[name]
    abl_check[name] = {"k": k, "n": N_ABL, "k_over_n": round(k / N_ABL, 4), "reported": reported,
                       "consistent_after_rounding": round(k / N_ABL, 2) == reported or
                       math.floor(k / N_ABL * 100 + 0.5) / 100 == reported}
results["t4_ablation_check"] = abl_check
pairs = {"Full vs Visual-only": ("EgoEngine", "+Visual branch"),
         "Full vs Action-only": ("EgoEngine", "+Action branch"),
         "Full vs Human videos": ("EgoEngine", "Human Videos")}
results["t4_fisher"] = {}
for label, (x, y) in pairs.items():
    a, b = abl_k[x], abl_k[y]
    _, p = fisher_exact([[a, N_ABL - a], [b, N_ABL - b]])
    results["t4_fisher"][label] = {"p": float(f"{p:.3g}"), "k": f"{a} vs {b}", "n": N_ABL}

# ---- 표 2: 시뮬레이션 성공률의 denominator 추정 ----
taco_sr = [0.17, 0.25, 0.83, 0.83]  # Mink, Spider, H2S2R, EgoEngine
aria_sr = [0.10, 0.20, 0.90, 0.90]
results["t2_taco_denominators"] = denominators(taco_sr, 1, 100)
results["t2_aria_denominators"] = denominators(aria_sr, 1, 100)
results["t2_taco_denominators_note"] = (
    "본문은 TACO 16쌍이라고 한다. 16으로는 0.17(=2/16 또는 3/16)을 만들 수 없다. "
    "N=12 또는 N=64(16쌍 x 4회 추정)와 맞는다. 실제 시도 수는 보고되지 않았다.")
results["t2_taco_16_possible"] = all(
    any(round(k / 16, 2) == round(v, 2) for k in range(17)) for v in taco_sr)

# 효율 주장 확인
cost = {"TACO": {"H2S2R": 73675, "EgoEngine": 34842}, "Aria": {"H2S2R": 20237, "EgoEngine": 16560}}
results["t2_cost_reduction_pct"] = {
    k: round(100 * (1 - v["EgoEngine"] / v["H2S2R"]), 1) for k, v in cost.items()}
results["t2_spider_vs_egoengine_cost_ratio_taco"] = round(34842 / 7923, 2)
results["t2_throughput_gain_pct"] = round(100 * (2.88 / 2.36 - 1), 2)  # 본문 22.0%
results["t2_taco_over_aria_length_ratio_check"] = {"reported": 2.39, "implied_aria_mean_len": round(327.5 / 2.39, 1)}

# Aria SR (N=20 추정) 에 대한 Fisher: EgoEngine vs H2S2R(RL)
a, b = 18, 18
results["t2_aria_fisher_ego_vs_rl_p"] = round(fisher_exact([[a, 20 - a], [b, 20 - b]])[1], 3)

# ---- 표 E.1: SPARC 평균 확인 ----
sparc = {"Real": {"Mustard": -8.68, "Drawer": -10.40, "Hammer": -3.21, "Flower": -4.66},
         "EgoEngine": {"Mustard": -4.88, "Drawer": -7.49, "Hammer": -3.25, "Flower": -3.88}}
reported_all = {"Real": -6.60, "EgoEngine": -4.81}
results["tE1_sparc_check"] = {}
for m in sparc:
    mean_tasks = sum(sparc[m].values()) / 4
    results["tE1_sparc_check"][m] = {"mean_of_task_values": round(mean_tasks, 3),
                                     "reported_all": reported_all[m],
                                     "gap": round(reported_all[m] - mean_tasks, 3)}
results["tE1_hammer_note"] = "Hammer에서는 Real(-3.21)이 EgoEngine(-3.25)보다 약간 매끄럽다."

# ---- 표 1: FD 상대 차이 (시료 수 없음, 검정 불가) ----
fd = {"ResNet18": {"Phantom": 620.0, "EgoEngine": 614.7},
      "VGG16": {"Phantom": 650.8, "EgoEngine": 644.2},
      "DINOv2": {"Phantom": 470.6, "EgoEngine": 473.1}}
results["t1_fd_rel_diff_pct_vs_phantom"] = {
    k: round(100 * (v["EgoEngine"] - v["Phantom"]) / v["Phantom"], 2) for k, v in fd.items()}

OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(results, ensure_ascii=False, indent=1))
