"""Targeting World Models to Compromise Robot Learning Pipelines (arXiv:2606.09499) 수치 재계산.

실행: python3 analysis/recompute.py  ->  analysis/recompute_results.json

입력은 논문 v2의 표 1, 표 4, 표 5와 v1의 표 1, 그리고 그림 4·5에서 읽은 근사값이다.
논문에 없는 값은 가정으로 표시한다(시행 수 n 가정, 그림 판독 값).
numpy와 scipy만 쓴다. CPU로 1분 안에 끝난다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
R = {}

OBJ = ["Bomb", "Dynamite", "Gun", "Oil", "Cardboard", "CerealBox", "Fork", "Knife", "Grater"]
SCENE = ["GiftBox"] * 3 + ["Stove"] * 3 + ["Microwave"] * 3

# 표 1 (v2). 값은 %.
T1_V2 = {
    "ood_vague":    [20, 50, 100, 5, 10, 90, 25, 20, 55],
    "ood_specific": [10, 25, 95, 5, 0, 65, 5, 5, 25],
    "ood_none":     [0, 0, 10, 0, 0, 0, 0, 0, 0],
    "id_vague":     [5, 25, 50, 0, 0, 5, 20, 0, 30],
    "id_specific":  [5, 0, 10, 0, 0, 0, 0, 0, 25],
    "id_none":      [0, 0, 5, 0, 0, 0, 0, 0, 0],
}
# 표 1 (v1)
T1_V1 = {
    "ood_vague":    [20, 0, 100, 5, 10, 90, 0, 20, 55],
    "ood_specific": [10, 0, 95, 5, 0, 65, 0, 5, 25],
    "ood_none":     [0, 0, 10, 0, 0, 0, 0, 0, 0],
    "id_vague":     [5, 0, 50, 0, 0, 5, 0, 0, 30],
    "id_specific":  [5, 0, 10, 0, 0, 0, 0, 0, 25],
    "id_none":      [0, 0, 5, 0, 0, 0, 0, 0, 0],
}
N = 20  # 5 perturbed images x 4 seeds (Sec. 6.1). 무공격 행의 n은 논문에 없어 20으로 가정.


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def counts(row):
    return [round(v * N / 100) for v in row]


# ---- V1: 표 1 정수 가능성 -------------------------------------------------
v1 = {}
bad = []
for key, row in T1_V2.items():
    for o, v in zip(OBJ, row):
        k = v * N / 100
        if abs(k - round(k)) > 1e-9:
            bad.append((key, o, v))
v1["n_assumed"] = N
v1["non_integer_cells_v2"] = bad
v1["all_multiples_of_5pct"] = all(v % 5 == 0 for r in T1_V2.values() for v in r)
v1["min_step_pct"] = 100 / N
R["V1_table1_integrality"] = v1

# ---- V2: 셀별 Wilson 95% CI (n=20 가정) ------------------------------------
ci = {}
for key, row in T1_V2.items():
    ci[key] = [
        {"obj": o, "pct": v, "k": round(v * N / 100), "wilson95": [round(100 * x, 1) for x in wilson(round(v * N / 100), N)]}
        for o, v in zip(OBJ, row)
    ]
R["V2_cell_wilson_ci"] = ci
# 이미지 5개를 독립 단위로 본 보수적 폭(시드 4개는 같은 섭동 이미지를 공유한다는 가정)
cons = {}
for pct in (0, 5, 10, 20, 50, 90, 100):
    k = round(pct * 5 / 100) if pct % 20 == 0 else None
    cons[str(pct)] = None if k is None else [round(100 * x, 1) for x in wilson(k, 5)]
R["V2_conservative_n5_wilson"] = cons

# ---- V3: 행·열 집계와 쌍체 검정 --------------------------------------------
agg = {}
for key, row in T1_V2.items():
    agg[key] = {
        "mean_of_9": round(float(np.mean(row)), 2),
        "pooled_k": int(sum(counts(row))),
        "pooled_n": 9 * N,
    }
R["V3_row_means"] = agg


def pair_test(a, b, label):
    a, b = np.array(a, float), np.array(b, float)
    d = a - b
    nz = d[d != 0]
    npos = int((nz > 0).sum())
    nneg = int((nz < 0).sum())
    sign_p = stats.binomtest(npos, npos + nneg, 0.5).pvalue if (npos + nneg) else 1.0
    try:
        w = stats.wilcoxon(a, b, zero_method="wilcox", alternative="greater")
        wp = float(w.pvalue)
    except ValueError:
        wp = None
    ka, kb = sum(counts(a)), sum(counts(b))
    fisher = stats.fisher_exact([[ka, 9 * N - ka], [kb, 9 * N - kb]], alternative="greater")[1]
    return {
        "label": label,
        "mean_diff_pp": round(float(d.mean()), 2),
        "objects_up": npos, "objects_down": nneg, "objects_tie": int((d == 0).sum()),
        "sign_test_two_sided_p": round(sign_p, 4),
        "wilcoxon_greater_p": None if wp is None else round(wp, 4),
        "pooled_fisher_greater_p_independence_assumed": float(f"{fisher:.3g}"),
    }


R["V3_paired_tests"] = [
    pair_test(T1_V2["ood_vague"], T1_V2["ood_specific"], "OOD: vague > specific"),
    pair_test(T1_V2["id_vague"], T1_V2["id_specific"], "ID: vague > specific"),
    pair_test(T1_V2["ood_vague"], T1_V2["id_vague"], "vague: OOD > ID"),
    pair_test(T1_V2["ood_specific"], T1_V2["id_specific"], "specific: OOD > ID"),
]

# 장면별(Gift Box/Stove/Microwave) 평균, 4개 조건의 합계
scene_means = {}
for s in ("GiftBox", "Stove", "Microwave"):
    idx = [i for i, x in enumerate(SCENE) if x == s]
    scene_means[s] = {k: round(float(np.mean([T1_V2[k][i] for i in idx])), 2) for k in T1_V2}
R["V3_scene_means"] = scene_means

# ---- V4: 공격 대 무공격(같은 카메라) 셀별 Fisher (무공격 n=20 가정) ----------
base_tests = []
for cam in ("ood", "id"):
    for pr in ("vague", "specific"):
        for i, o in enumerate(OBJ):
            ka = round(T1_V2[f"{cam}_{pr}"][i] * N / 100)
            kb = round(T1_V2[f"{cam}_none"][i] * N / 100)
            p = stats.fisher_exact([[ka, N - ka], [kb, N - kb]], alternative="greater")[1]
            base_tests.append({"cam": cam, "prompt": pr, "obj": o, "attack_k": ka, "base_k": kb, "p": float(f"{p:.3g}")})
m = len(base_tests)
ps = np.array([t["p"] for t in base_tests])
order = np.argsort(ps)
holm_sig = np.zeros(m, bool)
for rank, j in enumerate(order):
    if ps[j] <= 0.05 / (m - rank):
        holm_sig[j] = True
    else:
        break
R["V4_attack_vs_baseline"] = {
    "n_cells": m,
    "p_lt_0.05_unadjusted": int((ps < 0.05).sum()),
    "p_lt_0.05_holm": int(holm_sig.sum()),
    "cells_not_significant_unadjusted": [f"{t['cam']}/{t['prompt']}/{t['obj']}({t['attack_k']}/20 vs {t['base_k']}/20)" for t in base_tests if t["p"] >= 0.05 and t["attack_k"] > 0],
    "zero_cells": sum(1 for t in base_tests if t["attack_k"] == 0),
    "holm_significant_cells": [f"{t['cam']}/{t['prompt']}/{t['obj']}({t['attack_k']}/20 vs {t['base_k']}/20, p={t['p']})" for t, h in zip(base_tests, holm_sig) if h],
    "unadjusted_significant_cells": [f"{t['cam']}/{t['prompt']}/{t['obj']}({t['attack_k']}/20 vs {t['base_k']}/20, p={t['p']})" for t in base_tests if t["p"] < 0.05],
    "bomb_ood_vague_p": next(t["p"] for t in base_tests if t["cam"] == "ood" and t["prompt"] == "vague" and t["obj"] == "Bomb"),
}

# ---- V5: v1 -> v2 표 1 변경 -------------------------------------------------
chg = []
for key in T1_V1:
    for o, a, b in zip(OBJ, T1_V1[key], T1_V2[key]):
        if a != b:
            chg.append({"cond": key, "obj": o, "v1": a, "v2": b})
R["V5_v1_vs_v2_table1"] = {
    "changed_cells": chg,
    "n_changed": len(chg),
    "objects_changed": sorted({c["obj"] for c in chg}),
    "mean_ood_vague_v1_v2": [round(float(np.mean(T1_V1["ood_vague"])), 2), round(float(np.mean(T1_V2["ood_vague"])), 2)],
    "mean_id_vague_v1_v2": [round(float(np.mean(T1_V1["id_vague"])), 2), round(float(np.mean(T1_V2["id_vague"])), 2)],
    "sum_pct_all_cells_v1_v2": [int(sum(sum(r) for r in T1_V1.values())), int(sum(sum(r) for r in T1_V2.values()))],
    "v1_text_zero_objects": "dynamite, fork (all settings 0%)",
    "v2_dynamite_fork_cells_nonzero": sum(1 for key in T1_V2 for o, v in zip(OBJ, T1_V2[key]) if o in ("Dynamite", "Fork") and v > 0),
}

# ---- V6: 표 4 (Qwen3-VL 라벨 정확도, n=10) -----------------------------------
T4 = [60, 80, 70, 100, 100, 90, 10, 40, 50]
t4 = []
for o, v in zip(OBJ, T4):
    k = round(v / 10)
    t4.append({"obj": o, "pct": v, "k": k, "wilson95": [round(100 * x, 1) for x in wilson(k, 10)]})
k_all = sum(round(v / 10) for v in T4)
R["V6_table4_vlm_accuracy"] = {
    "cells": t4,
    "all_multiples_of_10": all(v % 10 == 0 for v in T4),
    "mean_pct": round(float(np.mean(T4)), 2),
    "pooled_k_n": [k_all, 90],
    "pooled_wilson95": [round(100 * x, 1) for x in wilson(k_all, 90)],
    "objects_at_100pct": [o for o, v in zip(OBJ, T4) if v == 100],
    "objects_at_or_below_50pct": [o for o, v in zip(OBJ, T4) if v <= 50],
    "note": "10개 표본이 모두 대상 물체를 집는 영상이므로(표 4 캡션) 정확도는 재현율과 같다. 오탐지율은 측정되지 않았다.",
}

# ---- V7: 표 5 (r2Dreamer, 5회 평균, 95% 구간) -------------------------------
T5 = {
    "cartpole_triggered_action_error": {"clean": (0.49, 0.10), "poisoned": (0.08, 0.5)},
    "cartpole_triggered_action_error_if_0.05": {"clean": (0.49, 0.10), "poisoned": (0.08, 0.05)},
    "cartpole_real_return": {"clean": (166, 86), "poisoned": (256, 37)},
    "reacher_triggered_action_error": {"clean": (1.41, 0.22), "poisoned": (0.51, 0.23)},
    "reacher_real_return": {"clean": (626, 12), "poisoned": (629, 28)},
}
tcrit = stats.t.ppf(0.975, 4)
t5 = {}
for name, d in T5.items():
    (m1, h1), (m2, h2) = d["clean"], d["poisoned"]
    s1, s2 = h1 * math.sqrt(5) / tcrit, h2 * math.sqrt(5) / tcrit
    se = math.sqrt(s1 ** 2 / 5 + s2 ** 2 / 5)
    t = (m2 - m1) / se
    df = (s1 ** 2 / 5 + s2 ** 2 / 5) ** 2 / ((s1 ** 2 / 5) ** 2 / 4 + (s2 ** 2 / 5) ** 2 / 4)
    p = 2 * stats.t.sf(abs(t), df)
    t5[name] = {
        "clean": [m1, h1], "poisoned": [m2, h2],
        "implied_sd": [round(s1, 3), round(s2, 3)],
        "diff": round(m2 - m1, 3), "welch_t": round(t, 2), "welch_df": round(df, 1), "p_two_sided": float(f"{p:.3g}"),
        "ci_overlap": (m1 + h1 >= m2 - h2) and (m2 + h2 >= m1 - h1),
    }
R["V7_table5_r2dreamer"] = {
    "assumption": "'±'를 5회 평균의 95% 신뢰구간 반폭(t 분포, 자유도 4)으로 읽음(표 5 캡션). 표준편차로 읽으면 SE가 작아져 p가 더 작아진다.",
    "rows": t5,
    "cartpole_halfwidth_exceeds_mean": 0.5 > 0.08,
}

# ---- V8: 그림 4, 5 판독 근사값 -------------------------------------------------
# 그림 4(Dino WM, 시드 5개, 선=평균, 띠=표준편차). 마지막 구간의 근사값(그림 판독 값).
fig4 = {
    "attack_success_vth_end": 0.33, "attack_success_vth_peak": 0.40, "attack_success_none_end": 0.03,
    "attack_cos_vth_end": 0.88, "attack_cos_none_end": -0.45, "attack_cos_vth_start": 0.60, "attack_cos_none_start": 0.05,
    "attack_success_start_vth": 0.14, "attack_success_start_none": 0.08,
    "real_env_success_both_end": 0.92,
}
R["V8_fig4_readings_approx"] = fig4
# 공격 성공의 정의: 모든 행동 차원이 목표 1의 0.2 이내 -> 행동이 2차원이면 [0.8,1]^2 상자.
# 균일 무작위 행동([-1,1]^2)이 상자에 들어갈 확률:
box = (0.2 / 2) ** 2
R["V8_attack_success_chance_level"] = {
    "assumption": "Single Wall 행동 공간을 [-1,1]^2 균일로 가정(논문은 2차원 행동만 명시, 범위는 미명시)",
    "uniform_random_hit_prob": box,
    "cos_sim_of_random_2d_with_(1,1)": 0.0,
}
fig5 = {
    "x": [-1.0, -0.5, 0.0, 0.5, 1.0],
    "embedding_l2_mean": [44.8, 46.7, 42.0, 36.0, 34.6],
    "embedding_sd_at_ends": [9.2, 9.2],
    "visual_l2_mean": [13800, 16900, 13400, 11300, 10900],
}
n_att = 20
sd = 9.2
d_end = fig5["embedding_l2_mean"][0] - fig5["embedding_l2_mean"][-1]
se_d = math.sqrt(2 * sd ** 2 / n_att)
R["V8_fig5_readings_approx"] = {
    **fig5,
    "rel_drop_embedding_minus1_to_1_pct": round(100 * d_end / fig5["embedding_l2_mean"][0], 1),
    "rel_drop_embedding_peak_to_1_pct": round(100 * (fig5["embedding_l2_mean"][1] - fig5["embedding_l2_mean"][-1]) / fig5["embedding_l2_mean"][1], 1),
    "rel_drop_visual_minus1_to_1_pct": round(100 * (fig5["visual_l2_mean"][0] - fig5["visual_l2_mean"][-1]) / fig5["visual_l2_mean"][0], 1),
    "unpaired_z_embedding_minus1_vs_1": round(d_end / se_d, 2),
    "note": "n=20, 독립 가정, 띠 반폭=표준편차로 읽음. 같은 20개 공격에서 나온 값이라 짝지은 비교가 맞고 그 상관은 알 수 없다. 완전한 원복이라면 거리가 0에 가까워야 하지만 목표 행동에서도 거리가 약 35(축은 25에서 시작).",
}

# ---- V9: LAB 예산을 RGB로 환산 (논문 표 2: beta_L=5, beta_AB=20, alpha=0.025) ----
M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
WP = np.array([0.95047, 1.0, 1.08883])


def srgb2lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin2srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def rgb2lab(rgb):
    xyz = M @ srgb2lin(rgb) / WP
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def lab2rgb(lab):
    fy = (lab[0] + 16) / 116
    fx = fy + lab[1] / 500
    fz = fy - lab[2] / 200
    inv = lambda t: np.where(t > 6 / 29, t ** 3, 3 * (6 / 29) ** 2 * (t - 4 / 29))
    xyz = np.array([inv(fx), inv(fy), inv(fz)]) * WP
    return lin2srgb(np.linalg.solve(M, xyz))


samples = {"gray128": [128, 128, 128], "table_wood": [190, 150, 110], "white": [235, 235, 235], "dark": [40, 40, 45], "red_box": [180, 40, 40]}
rgb_eq = {}
for name, rgb in samples.items():
    base = np.array(rgb) / 255
    lab0 = rgb2lab(base)
    best = []
    for sl in (-5, 5):
        for sa in (-20, 20):
            for sb in (-20, 20):
                new = lab2rgb(lab0 + np.array([sl, sa, sb]))
                best.append(float(np.max(np.abs(new - base)) * 255))
    rgb_eq[name] = {"max_abs_rgb_change_0_255": [round(min(best), 1), round(max(best), 1)]}
R["V9_lab_budget_in_rgb"] = {
    "per_pixel_max_deltaE76": round(math.sqrt(5 ** 2 + 20 ** 2 + 20 ** 2), 1),
    "jnd_deltaE76_rule_of_thumb": 2.3,
    "ratio_to_jnd": round(math.sqrt(5 ** 2 + 20 ** 2 + 20 ** 2) / 2.3, 1),
    "step_ab_per_iter": 0.025 * 20, "step_L_per_iter": 0.025 * 5,
    "min_steps_to_reach_ab_budget_no_momentum": math.ceil(20 / (0.025 * 20)),
    "steps_total": 400,
    "rgb_equiv_by_sample_color": rgb_eq,
    "common_rgb_linf_budgets_for_comparison": ["8/255", "16/255"],
    "note": "RGB 환산은 sRGB-D65 변환을 직접 구현한 근사이며 색마다 부호 조합 8개의 최솟값과 최댓값을 적었다.",
}

# ---- V10: Table 2의 예산과 식 6의 부호 ----------------------------------------
# 식 6 첫 항 min(0, ||z_best - z_g|| - ||z - z_g||): z_best가 지금까지 최선 거리라면 항상 <= 0
rng = np.random.default_rng(0)
viol = 0
for _ in range(20000):
    d = rng.random(30)               # 에피소드 안의 ||z_t - z_g||
    best = np.minimum.accumulate(d)  # z_best가 현재 상태를 포함해 갱신되는 경우
    r1 = np.minimum(0, best - d)
    r_prev = np.minimum(0, np.concatenate([[d[0]], best[:-1]]) - d)  # 갱신 전 최선과 비교
    viol += int((r1 > 0).any() or (r_prev > 0).any())
R["V10_eq6_first_term_sign"] = {
    "violations_of_nonpositive_in_20000_random_episodes": viol,
    "reading": "첫 항은 항상 0 이하다. '가까워지면 보너스'라는 서술과 달리 보너스가 되는 구간이 없다(표기가 부호를 뒤집었거나 서술이 느슨할 가능성).",
}

OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False))
print(json.dumps({k: R[k] for k in ("V1_table1_integrality", "V3_row_means", "V3_paired_tests", "V4_attack_vs_baseline")}, indent=1, ensure_ascii=False))
print(json.dumps(R["V5_v1_vs_v2_table1"], indent=1, ensure_ascii=False))
print(json.dumps(R["V6_table4_vlm_accuracy"], indent=1, ensure_ascii=False)[:1500])
print(json.dumps(R["V7_table5_r2dreamer"], indent=1, ensure_ascii=False))
print(json.dumps({k: R[k] for k in ("V8_attack_success_chance_level", "V8_fig5_readings_approx", "V9_lab_budget_in_rgb", "V10_eq6_first_term_sign", "V3_scene_means")}, indent=1, ensure_ascii=False))
