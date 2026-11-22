"""Being-H0.5 (arXiv:2601.12993 v1) 본문 수치 재계산과 일관성 검사.

입력은 모두 논문 본문 표와 그림 막대 라벨을 손으로 옮긴 값이다(표 1, 4, 5, 6, 7, 8, 그림 4, 10, 13).
그림 10·13의 막대 값은 PDF에서 읽은 라벨이다. 재계산 결과는 recompute_results.json에 저장한다.

실행: python3 analysis/recompute.py   (numpy, scipy만 필요, 수 초)
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
R = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def fisher(k1, n1, k2, n2):
    return stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]


def mde_two_prop(p, n, alpha=0.05, power=0.8):
    """두 집단 각 n회, 기준 비율 p 근처에서 검출 가능한 최소 차이(정규 근사)."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return (za + zb) * math.sqrt(2 * p * (1 - p) / n)


# ---------------------------------------------------------------- V1 표 1 합계
t1 = {
    "Franka": 2196.4, "Google Robot": 1195.2, "Kuka-iiwa": 338.2, "WidowX": 305.2, "Xarm7": 594.4,
    "UR5E": 208.3, "Hello Stretch": 270.3, "Kinova Jaco": 194.6, "Kinova Gen3": 197.4, "Sawyer": 196.1,
    "Cobotta": 10.6, "DLR Sara": 12.1, "PR2": 11.9, "RMC RM65": 41.3,
    "RMC Aida L": 325.7, "Galaxea R1 Lite": 630.5, "Agilex Split ALOHA": 1099.1, "Agilex Cobot Magic": 399.2,
    "Piper": 904.5, "BeingBeyond D1": 100.0, "SO101": 194.4,
    "PND AdamU": 200.0, "Agibot-G1": 2391.7, "Leju kuavo 4 LB": 1198.2, "AlphaBot 2": 13.8,
    "Airbot MMK2": 23.1, "Galbot G1": 195.8, "Tianqing A2": 19.4,
    "Unitree G1edu-u3": 135.7, "Tiankung": 214.3,
}
groups = {"single_arm": 14, "dual_arm": 5, "portable": 2, "half_humanoid": 7, "humanoid": 2}
s1 = sum(t1.values())
R["V1_table1"] = {
    "n_rows": len(t1), "sum_hours": round(s1, 1), "reported_total": 13817.4, "diff": round(s1 - 13817.4, 2),
    "min_hours": min(t1.values()), "rows_ge_10h": all(v > 10 for v in t1.values()),
    "group_counts_sum": sum(groups.values()),
}

# ---------------------------------------------------------------- V2 그림 4와 본문 규모
sim, real = 3592.0, 10225.0
oxe, agibot, other_real = 424.0, 420.0, 9381.0
tok = {"human": 25.6, "robot": 45.7, "vlm": 50.2}  # 그림 4 캡션 (B tokens)
hrs = {"human": 16.0, "robot": 13.8, "vlm": 5.0}  # 그림 4 (k hours)
R["V2_scale"] = {
    "real_plus_sim_hours": sim + real, "robot_table1_hours": 13817.4,
    "real_parts_sum": oxe + agibot + other_real,
    "sim_share_of_robot_hours_pct": round(100 * sim / (sim + real), 2),
    "sim_share_of_total_34p8k_hours_pct": round(100 * sim / 34800, 2),
    "oxe_share_pct": round(100 * oxe / 13817, 2), "agibot_share_pct": round(100 * agibot / 13817, 2),
    "total_hours_k": round(sum(hrs.values()), 1), "total_tokens_B": round(sum(tok.values()), 1),
    "text_total_hours": 35000, "text_total_tokens_B": 120, "fig4_total_tokens_B": 122,
    "robot_text_to_vision_ratio_text": "approximately 1,000:1 (45.7B visual / 30M text)",
    "robot_vision_to_text_ratio_actual": round(45.7e9 / 30e6, 1),
    "robot_tokens_per_frame_if_1p5B_frames": round(45.7e9 / 1.5e9, 1),
    "robot_frames_per_second_if_1p5B_frames_over_13817h": round(1.5e9 / (13817.4 * 3600), 1),
    "total_samples_M": 400, "human_samples_M": 134,
    "samples_left_for_robot_and_vlm_M": 400 - 134,
    "vs_unihand1_hours_ratio": round(16000 / 1155, 1), "vs_unihand1_total_hours_ratio": round(35000 / 1155, 1),
    "vs_unihand1_samples_ratio_human_134M_over_166p4M": round(134 / 166.4, 2),
    "claimed_human_ratio": "100x (3절)", "claimed_total_ratio": "200x (1절)",
}

# ---------------------------------------------------------------- V3 표 4 LIBERO
t4 = {
    "Diffusion Policy": (78.5, 87.5, 73.5, 64.8, 76.1), "OpenVLA": (84.7, 88.4, 79.2, 53.7, 76.5),
    "SpatialVLA": (88.2, 89.9, 78.6, 55.5, 78.1), "CoT-VLA": (87.5, 91.6, 87.6, 69.0, 83.9),
    "pi0-Fast": (96.4, 96.8, 88.6, 60.2, 85.5), "GR00T-N1": (94.4, 97.6, 93.0, 90.6, 93.9),
    "pi0": (98.0, 96.8, 94.4, 88.4, 94.4), "F1": (98.2, 97.8, 95.4, 91.3, 95.7),
    "InternVLA-M1": (98.0, 99.0, 93.8, 92.6, 95.9), "Discrete Diffusion VLA": (97.2, 98.6, 97.4, 92.0, 96.3),
    "pi0.5": (98.8, 98.2, 98.0, 92.4, 96.9), "OpenVLA-OFT": (97.6, 98.4, 97.9, 94.5, 97.1),
    "X-VLA": (98.2, 98.6, 97.8, 97.6, 98.1), "EO1": (99.7, 99.8, 99.2, 94.8, 98.2),
    "Being-H0.5 (generalist)": (97.0, 98.2, 99.0, 96.2, 97.6), "Being-H0.5 (specialist)": (99.2, 99.6, 99.4, 97.4, 98.9),
}
avg_check = {}
for k, v in t4.items():
    m = sum(v[:4]) / 4
    avg_check[k] = {"mean4": round(m, 3), "reported": v[4], "abs_diff": round(abs(m - v[4]), 3),
                    "ok_at_0p1": abs(m - v[4]) <= 0.05 + 1e-9}
cols = ["Spatial", "Object", "Goal", "Long", "Average"]
best = {}
for j, c in enumerate(cols):
    mx = max(v[j] for v in t4.values())
    best[c] = [k for k, v in t4.items() if abs(v[j] - mx) < 1e-9]
spec = t4["Being-H0.5 (specialist)"]
gen = t4["Being-H0.5 (generalist)"]
n_suite = 500
grid_ok = all(abs(x * n_suite / 100 - round(x * n_suite / 100)) < 1e-6 for v in (spec, gen) for x in v[:4])
k_spec = round(spec[4] * 2000 / 100)
ci_spec = wilson(k_spec, 2000)
k_eo1 = round(98.2 * 2000 / 100)
R["V3_libero"] = {
    "avg_check": avg_check, "best_per_column": best,
    "n_failed_avg_checks": sum(1 for v in avg_check.values() if not v["ok_at_0p1"]),
    "being_h05_grid_multiple_of_0p2_ok": grid_ok,
    "specialist_avg_k_of_2000": k_spec, "specialist_wilson95": [round(100 * x, 2) for x in ci_spec],
    "specialist_vs_EO1_fisher_p_n2000_each": round(fisher(k_spec, 2000, k_eo1, 2000), 4),
    "specialist_vs_Xvla_long_diff_pp": round(spec[3] - t4["X-VLA"][3], 1),
    "specialist_vs_EO1_avg_diff_pp": round(spec[4] - t4["EO1"][4], 1),
    "generalist_vs_EO1_avg_diff_pp": round(gen[4] - t4["EO1"][4], 1),
    "specialist_long_vs_pi0p5_long_diff_pp": round(spec[3] - t4["pi0.5"][3], 1),
    "generalist_rank_by_average": 1 + sum(1 for k, v in t4.items() if v[4] > gen[4]),
    "specialist_columns_best": [c for c in cols if "Being-H0.5 (specialist)" in best[c]],
    "mde_pp_at_98_n2000": round(100 * mde_two_prop(0.98, 2000), 2),
    "mde_pp_at_97_n500": round(100 * mde_two_prop(0.97, 500), 2),
}

# ---------------------------------------------------------------- V4 표 5 RoboCasa
t5 = {
    "3DA": (0.0, 2.3, 13.1, 5.5), "DP3": (1.5, 41.7, 32.0, 22.8), "GWM": (14.8, 54.3, 49.8, 39.3),
    "BC": (4.3, 47.0, 42.2, 28.9), "GR00T-N1": (18.6, 50.2, 39.1, 36.0), "pi0.5": (21.5, 57.8, 44.9, 41.4),
    "pi0": (14.0, 53.1, 58.5, 42.4), "Being-H0.5 (generalist)": (40, 73, 52, 53.3),
    "Being-H0.5 (specialist)": (36, 71.7, 57.6, 53.9),
}
w = np.array([8, 6, 10]) / 24  # 공식 코드 YAML의 24개 과제 분류 (PnP 8, 문/서랍 6, 기타 10)
rc = {}
for k, v in t5.items():
    wt = float(np.dot(w, v[:3]))
    un = float(np.mean(v[:3]))
    rc[k] = {"weighted_8_6_10": round(wt, 2), "unweighted": round(un, 2), "reported": v[3],
             "match_weighted": abs(wt - v[3]) <= 0.1, "match_unweighted": abs(un - v[3]) <= 0.1}
trial_grid = {"pnp_400": [round(100 * i / 400, 2) for i in (144, 160)],
              "door_300": [round(100 * 215 / 300, 2)], "other_500": [round(100 * 288 / 500, 2)]}
k_rs = round(0.539 * 1200)
k_pi0 = round(0.424 * 1200)
k_pi05 = round(0.414 * 1200)
R["V4_robocasa"] = {
    "rows": rc,
    "n_match_weighted": sum(v["match_weighted"] for v in rc.values()),
    "n_match_unweighted": sum(v["match_unweighted"] for v in rc.values()),
    "being_h05_unweighted_if_same_convention": {k: rc[k]["unweighted"] for k in rc if k.startswith("Being")},
    "being_h05_specialist_wilson95_n1200": [round(100 * x, 2) for x in wilson(k_rs, 1200)],
    "spec_vs_pi0_diff_pp": round(53.9 - 42.4, 1),
    "spec_vs_pi0_fisher_p_n1200_each": float(f"{fisher(k_rs, 1200, k_pi0, 1200):.2e}"),
    "spec_vs_gen_diff_pp": round(53.9 - 53.3, 1),
    "spec_vs_gen_fisher_p": round(fisher(k_rs, 1200, round(0.533 * 1200), 1200), 3),
    "gen_pnp_40_check_grid_400": 40 * 400 / 100,
    "pnp_text_range": "36-40% (표 5 일치)",
}

# ---------------------------------------------------------------- V5 표 6, 7 (LIBERO 5-shot)
T6 = {
    "Und+Proj+ViT": {"native": (67.8, 76.8, 57.2, 29.8, 57.9), "hc": (78.6, 78.9, 72.6, 46.2, 69.0),
                     "gain": (10.8, 2.1, 15.4, 16.4, 11.1)},
    "Und+ViT": {"native": (70.6, 50.0, 67.0, 17.4, 51.3), "hc": (85.0, 84.2, 80.0, 59.0, 77.1),
                "gain": (14.4, 34.4, 13.0, 41.6, 25.8)},
    "Und only": {"native": (78.4, 84.8, 62.0, 49.2, 68.6), "hc": (83.2, 79.6, 74.0, 63.6, 75.1),
                 "gain": (4.8, -5.2, 12.0, 14.4, 6.5)},
    "Proj+ViT": {"native": (87.2, 86.6, 80.8, 58.0, 78.2), "hc": (86.2, 86.2, 84.6, 66.8, 81.0),
                 "gain": (-1.0, -0.4, 3.8, 8.8, 2.8)},
    "None (Full FT)": {"native": (84.2, 83.6, 75.0, 65.8, 77.2), "hc": (87.2, 91.4, 81.6, 66.8, 81.8),
                       "gain": (3.0, 7.8, 6.6, 1.0, 4.6)},
}
T7 = {
    "Und+Proj+ViT": {"native": (61.6, 77.0, 68.0, 36.0, 60.7), "hc": (80.8, 81.8, 74.4, 52.4, 72.4),
                     "gain": (19.2, 4.8, 6.4, 16.4, 11.7)},
    "Und+ViT": {"native": (78.4, 84.8, 71.4, 38.6, 68.3), "hc": (83.6, 82.6, 80.8, 59.2, 76.6),
                "gain": (5.2, -2.2, 9.4, 20.6, 8.3)},
    "Und only": {"native": (75.8, 84.2, 71.2, 59.8, 72.8), "hc": (85.2, 90.6, 84.2, 68.4, 82.1),
                 "gain": (9.4, 6.4, 13.0, 8.6, 9.3)},
    "Proj+ViT": {"native": (87.6, 91.0, 82.2, 62.6, 80.9), "hc": (89.6, 88.8, 89.6, 65.8, 83.5),
                 "gain": (2.0, -2.2, 7.4, 3.2, 2.6)},
    "None (Full FT)": {"native": (87.8, 94.6, 82.6, 71.4, 84.1), "hc": (88.8, 90.2, 85.4, 76.0, 85.1),
                       "gain": (1.0, -4.4, 2.8, 4.6, 1.0)},
}


def check_table(T, tag):
    out = {}
    bad = []
    grid_bad = []
    pvals = []
    for cfg, d in T.items():
        for key in ("native", "hc"):
            v = d[key]
            m = sum(v[:4]) / 4
            if abs(m - v[4]) > 0.051:
                bad.append((cfg, key, round(m, 2), v[4]))
            for x in v[:4]:
                if abs(x * 5 - round(x * 5)) > 1e-6:  # 500회 -> 0.2 간격
                    grid_bad.append((cfg, key, x))
        for j in range(5):
            g = round(d["hc"][j] - d["native"][j], 1)
            if abs(g - d["gain"][j]) > 0.051:
                bad.append((cfg, "gain", j, g, d["gain"][j]))
        for j in range(4):
            n = 500
            k1, k2 = round(d["hc"][j] * n / 100), round(d["native"][j] * n / 100)
            pvals.append((cfg, cols[j], d["gain"][j], fisher(k1, n, k2, n)))
        k1, k2 = round(d["hc"][4] * 2000 / 100), round(d["native"][4] * 2000 / 100)
        out[cfg + "|avg_p"] = float(f"{fisher(k1, 2000, k2, 2000):.3g}")
    ps = np.array([p[3] for p in pvals])
    # Holm
    order = np.argsort(ps)
    m = len(ps)
    holm = np.zeros(m, dtype=bool)
    for r, i in enumerate(order):
        if ps[i] * (m - r) < 0.05:
            holm[i] = True
        else:
            break
    out["arithmetic_mismatches"] = bad
    out["grid_mismatches_0p2"] = grid_bad
    out["n_cells"] = m
    out["n_p_lt_0p05_uncorrected"] = int((ps < 0.05).sum())
    out["n_holm_sig"] = int(holm.sum())
    out["nonpositive_gain_cells"] = [(p[0], p[1], p[2]) for p in pvals if p[2] <= 0]
    out["mde_pp_at_75_n500"] = round(100 * mde_two_prop(0.75, 500), 1)
    out["mde_pp_at_80_n2000"] = round(100 * mde_two_prop(0.8, 2000), 1)
    return out


R["V5_table6"] = check_table(T6, "6")
R["V6_table7"] = check_table(T7, "7")
nat6 = {k: v["native"][4] for k, v in T6.items()}
R["V5_table6"]["native_avg_by_frozen_set"] = nat6
R["V5_table6"]["native_spread_pp"] = round(max(nat6.values()) - min(nat6.values()), 1)
R["V5_table6"]["native_L_object_range"] = [min(v["native"][1] for v in T6.values()), max(v["native"][1] for v in T6.values())]
R["V5_table6"]["gain_avg_by_row"] = {k: v["gain"][4] for k, v in T6.items()}
R["V6_table7"]["gain_avg_by_row"] = {k: v["gain"][4] for k, v in T7.items()}
R["V6_table7"]["native_avg_by_frozen_set"] = {k: v["native"][4] for k, v in T7.items()}
# 본문 인용: '+25.8', '+41.6', 'L-Object -4.4'
R["V5_table6"]["text_quotes_ok"] = {"+25.8": T6["Und+ViT"]["gain"][4] == 25.8, "+41.6": T6["Und+ViT"]["gain"][3] == 41.6,
                                      "-4.4 (Table 7 full FT L-Object)": T7["None (Full FT)"]["gain"][1] == -4.4}
R["V5_table6"]["max_frozen_row_gain"] = T6["Und+Proj+ViT"]["gain"][4]

# ---------------------------------------------------------------- V7 표 8
R["V7_table8"] = {
    "hybrid_lab_wild": (0.33, 0.20), "wo_mask_lab_wild": (0.35, 0.28),
    "delta_wo_minus_hybrid": (round(0.35 - 0.33, 2), round(0.28 - 0.20, 2)),
    "text_claim": "removing masked prediction leads to a clear drop in MWDS on both splits",
    "direction_consistent_with_text": False,
    "wild_relative_increase_pct_when_removed": round(100 * (0.28 - 0.20) / 0.20, 1),
}

# ---------------------------------------------------------------- V8 그림 10, 13 (막대 라벨)
F10 = {  # specialist, generalist, pi0.5, scratch-spec, scratch-gen
    "Spatial": (75, 70, 55, 50, 35), "Long Horizon": (60, 60, 45, 35, 25),
    "Bimanual": (55, 45, 40, 25, 30), "Generalization": (75, 80, 65, 60, 50),
}
F13 = {  # generalist, w/o MPG, w/o UAC, w/o both
    "Spatial": (70, 55, 65, 50), "Long Horizon": (60, 55, 40, 35),
    "Bimanual": (45, 40, 20, 20), "Generalization": (80, 60, 80, 55),
}
names10 = ["specialist", "generalist", "pi0.5", "scratch_spec", "scratch_gen"]
cats = list(F10)
f10 = {}
for c in cats:
    s, g, p, ss, sg = F10[c]
    f10[c] = {"spec_minus_pi05": s - p, "gen_minus_pi05": g - p, "spec_minus_gen": s - g,
              "spec_minus_scratch_spec": s - ss, "gen_minus_scratch_gen": g - sg,
              "scratch_gen_minus_scratch_spec": sg - ss}
mean10 = {n: round(float(np.mean([F10[c][i] for c in cats])), 2) for i, n in enumerate(names10)}
f13 = {}
for c in cats:
    g, a, b, ab = F13[c]
    f13[c] = {"drop_wo_MPG": g - a, "drop_wo_UAC": g - b, "drop_wo_both": g - ab,
              "additivity_gap(both - (MPG+UAC))": (g - ab) - ((g - a) + (g - b))}
mean13 = {n: round(float(np.mean([F13[c][i] for c in cats])), 2) for i, n in enumerate(["generalist", "wo_MPG", "wo_UAC", "wo_both"])}
rank_gap = {k: sorted(((c, f10[c][k]) for c in cats), key=lambda x: -x[1]) for k in ("spec_minus_pi05", "gen_minus_pi05")}
# 정수 시도 수 가능성: 모든 값이 5의 배수 -> 시도 수 N은 20의 배수여야 함
all_vals = [x for c in cats for x in F10[c]] + [x for c in cats for x in F13[c]]
R["V8_real_robot"] = {
    "all_values_multiple_of_5": all(x % 5 == 0 for x in all_vals),
    "min_N_for_integer_counts": 20,
    "fig13_generalist_equals_fig10_generalist": all(F13[c][0] == F10[c][1] for c in cats),
    "fig10": f10, "mean_over_categories": mean10, "fig13": f13, "fig13_mean": mean13,
    "gap_ranking": rank_gap,
    "task_rows_in_table3": 11, "tasks_in_text": 10,
    "tasks_per_category_table3": {"Spatial": 4, "Long Horizon": 2, "Bimanual": 2, "Generalization": 3},
}
# 시도 수 시나리오별 검정 (가정: 범주당 N회, 서로 독립)
scen = {}
for N in (20, 40, 80, 160):
    d = {}
    for c in cats:
        s, g, p, ss, sg = F10[c]
        d[c] = {
            "spec_vs_pi05_p": round(fisher(round(s * N / 100), N, round(p * N / 100), N), 3),
            "gen_vs_pi05_p": round(fisher(round(g * N / 100), N, round(p * N / 100), N), 3),
            "spec_vs_gen_p": round(fisher(round(s * N / 100), N, round(g * N / 100), N), 3),
            "gen_vs_scratch_gen_p": round(fisher(round(g * N / 100), N, round(sg * N / 100), N), 3),
        }
    # 네 범주 합산 (독립 가정)
    tot = lambda i: sum(round(F10[c][i] * N / 100) for c in cats)
    d["pooled_4cats"] = {
        "N_total": 4 * N,
        "spec_vs_pi05_p": float(f"{fisher(tot(0), 4 * N, tot(2), 4 * N):.3g}"),
        "gen_vs_pi05_p": float(f"{fisher(tot(1), 4 * N, tot(2), 4 * N):.3g}"),
        "spec_vs_gen_p": round(fisher(tot(0), 4 * N, tot(1), 4 * N), 3),
        "gen_vs_scratch_gen_p": float(f"{fisher(tot(1), 4 * N, tot(4), 4 * N):.3g}"),
        "spec_vs_scratch_spec_p": float(f"{fisher(tot(0), 4 * N, tot(3), 4 * N):.3g}"),
    }
    d["mde_pp_at_50"] = round(100 * mde_two_prop(0.5, N), 1)
    d["wilson_for_60pct"] = [round(100 * x, 1) for x in wilson(round(0.6 * N), N)]
    d["fig13_pooled_gen_vs_both_removed_p"] = float(
        f"{fisher(sum(round(F13[c][0] * N / 100) for c in cats), 4 * N, sum(round(F13[c][3] * N / 100) for c in cats), 4 * N):.3g}")
    scen[str(N)] = d
R["V9_real_robot_tests"] = scen
# '비영' 영샷의 증거 가치: N회 중 1회 이상 성공이 관측될 확률
R["V10_zero_shot_nonzero"] = {
    str(N): {f"true_rate_{int(p*100)}pct": round(1 - (1 - p) ** N, 3) for p in (0.01, 0.05, 0.10)}
    for N in (10, 20, 40)
}
R["V10_zero_shot_nonzero"]["one_success_in_20_wilson95_pct"] = [round(100 * x, 1) for x in wilson(1, 20)]
R["V10_zero_shot_nonzero"]["zero_success_in_20_upper95_pct"] = round(100 * (1 - 0.05 ** (1 / 20)), 1)

# ---------------------------------------------------------------- V11 표 2 자유도 합
R["V11_table2_dof"] = {
    "G1_LinkerBot_O6": {"2x7 arms + 2x6 hands": 2 * 7 + 2 * 6, "reported": 26},
    "FR3_Inspire": {"7 + 6": 7 + 6, "reported": 13},
    "AdamU": {"2x7 arms + head 2 + waist 3 + 2x6 hands": 2 * 7 + 2 + 3 + 2 * 6, "reported": 31,
              "note": "공식 문서의 슬롯(머리 2, 허리 3, 팔 7)과 일치"},
    "SO101": {"5 arm + 1 gripper": 6, "reported": 6},
}

# ---------------------------------------------------------------- V12 스텝 수와 저장 격자 (코드 대조용)
saved = [s for s in range(1, 60001) if s > 25000 and s % 10000 == 0]
R["V12_steps"] = {
    "paper_steps_libero": 45000, "script_max_steps": 60000, "script_save_steps_grid": saved,
    "paper_45k_in_save_grid": 45000 in saved,
    "paper_batch": 128, "paper_tokens_per_gpu": 7680, "script_expected_tokens": 8192, "script_max_tokens": 8704,
    "paper_chunk": 8, "script_train_chunk": 16, "checkpoint_config_libero_chunk": 15,
    "paper_cross_steps_2x": 90000, "script_cross_steps": 120000,
}

# ---------------------------------------------------------------- V13 MPG 게이트 장난감 (CPU, 수 초)
rng = np.random.default_rng(0)


def ln(x):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)


def swd_code(x, y, M, rng):
    """공식 코드(layers.py SlicedWassersteinDistance)와 같은 계산: 배치·토큰을 펴서 정렬 후 제곱차 평균."""
    d = x.shape[-1]
    tot = 0.0
    for _ in range(M):
        th = rng.standard_normal(d)
        th /= np.linalg.norm(th)
        a = np.sort((x @ th).reshape(-1))
        b = np.sort((y @ th).reshape(-1))
        tot += np.mean((a - b) ** 2)
    return tot / M


d, N = 1024, 80  # 은닉 크기와 토큰 수(상태 + 청크)는 장난감 값
toy = {}
z = ln(rng.standard_normal((16, d))).mean(0)  # 정규화된 행동 임베딩의 시퀀스 평균
h = ln(rng.standard_normal((N, d)))
closed = float(np.mean(np.sum((h - z) ** 2, axis=1)) / d)
sw = [swd_code(h, np.broadcast_to(z, h.shape), 32, np.random.default_rng(1000 + s)) for s in range(20)]
toy["closed_form_mean_sq_dist_over_d"] = round(closed, 4)
toy["swd_code_mean_over_20_seeds"] = round(float(np.mean(sw)), 4)
toy["swd_code_std_over_20_seeds"] = round(float(np.std(sw)), 4)
# 크기 변화는 LayerNorm이 지운다
D_scale = {}
for s in (0.1, 1.0, 10.0):
    D_scale[str(s)] = round(float(np.mean(np.sum((ln(s * h) - z) ** 2, axis=1)) / d), 4)
toy["D_after_rescaling_features"] = D_scale
# 앵커와 정렬된 정도에 따른 D와 게이트
gates = {}
for rho in (0.0, 0.2, 0.5, 0.8):
    base = ln(rng.standard_normal((N, d)))
    zz = ln(rng.standard_normal((1, d)))[0]
    hh = ln(rho * zz + math.sqrt(1 - rho ** 2) * base)
    D = float(np.mean(np.sum((hh - zz) ** 2, axis=1)) / d)
    gates[str(rho)] = {"D": round(D, 3), "g_tau1": round(math.exp(-D / 1.0), 3), "g_tau2": round(math.exp(-D / 2.0), 3)}
toy["gate_vs_alignment"] = gates
toy["gate_vs_alignment_note"] = "앵커 z가 단위 분산 한 벡터(c=1)일 때 D = 1 + 1 - 2*rho. 16개 독립 토큰 평균이면 c≈0.06, 무관한 특징의 D≈1.06"
toy["unrelated_features_averaged_anchor"] = {"D": round(closed, 3), "g_tau1": round(math.exp(-closed), 3), "g_tau2": round(math.exp(-closed / 2), 3)}
# 이론 범위: D = (1/N) sum ||h_i - z||^2 / d, h_i는 LN이라 ||h_i||^2 = d
toy["D_range_when_z_unit_norm_per_dim"] = [0.0, 4.0]
toy["gate_range_tau1"] = [round(math.exp(-4), 3), 1.0]
toy["gate_range_tau2"] = [round(math.exp(-2), 3), 1.0]
R["V13_mpg_toy"] = toy

OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                encoding="utf-8")
print("wrote", OUT)
for k, v in R.items():
    print(k, json.dumps(v, ensure_ascii=False, default=str)[:400])
