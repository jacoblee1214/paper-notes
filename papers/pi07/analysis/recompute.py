"""π0.7 보고 수치 재계산 (V 항목). 단일 프로세스, CPU, 수 초.

입력 값은 논문 그림(그림 6–12, 15–18, 22)의 벡터 막대와 오차 막대에서 읽은 값이거나(소수 셋째 자리까지는
'그림 판독 값'이며 원자료가 아님), 본문·부록 문장의 값이다. 시행 수는 논문에 없어서, 막대 높이가 정수 비율이 되고
오차 막대가 Wilson 구간과 맞는 정수 n을 찾아 '가정'으로 쓴다.
실행: python3 -I analysis/recompute.py  → analysis/recompute_results.json
"""
import json
import math
from pathlib import Path

from scipy.stats import fisher_exact

OUT = Path(__file__).with_name("recompute_results.json")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def infer_n(p, lo, hi, nmax=120):
    """p와 오차 막대 [lo, hi]에 가장 맞는 정수 (k, n). 막대 높이 오차(0.004)와 구간 오차의 합이 최소인 후보."""
    best = None
    for n in range(4, nmax + 1):
        k = round(p * n)
        if k <= 0 or k >= n or abs(k / n - p) > 0.004:
            continue
        l, h = wilson(k, n)
        err = abs(l - lo) + abs(h - hi) + 4 * abs(k / n - p)
        if best is None or err < best[0]:
            best = (err, k, n, l, h)
    return best


# (이름, p, lo, hi) — 성공률 막대 (그림 판독 값)
SUCC = {
    "fig6_laundry_ts_pi07": (0.9615, 0.893, 0.987), "fig6_laundry_ts_spec": (0.9350, 0.857, 0.972),
    "fig6_laundry_div_pi07": (0.7428, 0.579, 0.858), "fig6_laundry_div_spec": (0.6999, 0.481, 0.854),
    "fig6_espresso_pi07": (0.9473, 0.754, 0.991), "fig6_espresso_spec": (0.9499, 0.764, 0.991),
    "fig6_box_pi07": (0.8749, 0.640, 0.965), "fig6_box_spec": (0.8749, 0.640, 0.965),
    "fig7_laundry_ts_nometa": (0.9333, 0.787, 0.982), "fig7_laundry_ts_noeval": (0.9000, 0.744, 0.965),
    "fig7_laundry_div_nometa": (0.6000, 0.357, 0.802),
    "fig7_espresso_nometa": (0.5263, 0.317, 0.727),
    "fig7_box_nometa": (0.3636, 0.197, 0.570), "fig7_box_noeval": (0.5000, 0.290, 0.710),
    "fig10_std_pi05": (0.6667, 0.478, 0.814), "fig10_std_pi06": (0.8889, 0.719, 0.962),
    "fig10_std_pi07": (0.9630, 0.817, 0.993), "fig10_std_gc": (0.9260, 0.766, 0.980),
    "fig10_cplx_pi05": (0.2222, 0.117, 0.381), "fig10_cplx_pi06": (0.2222, 0.117, 0.381),
    "fig10_cplx_pi07": (0.3611, 0.225, 0.524), "fig10_cplx_gc": (0.6389, 0.476, 0.775),
}
fits = {}
for name, (p, lo, hi) in SUCC.items():
    b = infer_n(p, lo, hi)
    fits[name] = None if b is None else {"p": p, "ci_read": [lo, hi], "k": b[1], "n": b[2],
                                         "wilson": [round(b[3], 3), round(b[4], 3)], "fit_err": round(b[0], 3)}

# 오차 막대 읽기가 깨진 두 칸은 막대 높이만으로 정수 비율을 정한다(구간 대조 없음)
for name, (k, n) in {"fig7_laundry_div_noeval": (9, 15), "fig7_espresso_noeval": (12, 19)}.items():
    fits[name] = {"p": round(k / n, 4), "ci_read": None, "k": k, "n": n, "wilson": [round(x, 3) for x in wilson(k, n)], "fit_err": None}
res = {"n_fits": fits}


def fx(k1, n1, k2, n2):
    return fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]


def f(name):
    r = fits[name]
    return r["k"], r["n"]


# V3: 그림 6 (위 줄) π0.7 대 전문가 성공률 Fisher (추정 n)
v3 = {}
for t in ["laundry_ts", "laundry_div", "espresso", "box"]:
    a, b = f(f"fig6_{t}_pi07"), f(f"fig6_{t}_spec")
    v3[t] = {"pi07": a, "spec": b, "p_fisher": round(fx(*a, *b), 3)}
res["V3_fig6_pi07_vs_specialist"] = v3

# V4: 그림 7 절제 (π0.7 대 metadata 없음/평가 데이터 없음)
v4 = {}
for t, base in [("laundry_ts", "fig6_laundry_ts_pi07"), ("laundry_div", "fig6_laundry_div_pi07"),
                ("espresso", "fig6_espresso_pi07"), ("box", "fig6_box_pi07")]:
    row = {}
    for ab in ["nometa", "noeval"]:
        key = f"fig7_{t}_{ab}"
        if key in fits and fits[key]:
            k2, n2 = f(key)
            row[ab] = {"k_n": [k2, n2], "p_fisher": round(fx(*f(base), k2, n2), 3)}
    v4[t] = row
res["V4_fig7_ablation_success"] = v4

# V6: 그림 10 복합 지시
a_gc, a_07, a_06 = f("fig10_cplx_gc"), f("fig10_cplx_pi07"), f("fig10_cplx_pi06")
res["V6_fig10"] = {
    "complex_gc_vs_pi07": round(fx(*a_gc, *a_07), 4),
    "complex_pi07_vs_pi06": round(fx(*a_07, *a_06), 3),
    "std_pi07_vs_pi06": round(fx(*f("fig10_std_pi07"), *f("fig10_std_pi06")), 3),
    "std_pi07_vs_gc": round(fx(*f("fig10_std_pi07"), *f("fig10_std_gc")), 3),
    "complex_gc_vs_pi06": round(fx(*a_gc, *a_06), 5),
    "n_std": f("fig10_std_pi07")[1], "n_complex": a_gc[1]}

# V11: 그림 17 (성공률, 그림 판독 값, n 가정: 막대가 20분의 정수가 되는 칸은 n=20)
fig17 = {
    "press_plunger": {"pi05": 0.05, "pi06": 0.65, "pi07": 0.45, "gc": 0.80, "n": 20},
    "scoop_rice": {"pi05": 0.25, "pi06": 0.15, "pi07": 0.85, "gc": 0.70, "n": 20},
    "wiping_office": {"pi05": 0.3611, "pi06": 0.25, "pi07": 0.5277, "gc": 0.639, "n": 36},
}
v11 = {}
for t, d in fig17.items():
    n = d["n"]
    k = {m: round(d[m] * n) for m in ("pi05", "pi06", "pi07", "gc")}
    v11[t] = {"k": k, "n": n,
              "gc_vs_pi07_p": round(fx(k["gc"], n, k["pi07"], n), 3),
              "pi07_vs_pi06_p": round(fx(k["pi07"], n, k["pi06"], n), 3)}
res["V11_fig17"] = v11

# V12: 그림 17 π0.7 값과 그림 18(오른쪽) π0.7 값 (같은 과제, 같은 모델)
res["V12_fig17_vs_fig18"] = {
    "scoop_rice": {"fig17": 0.85, "fig18": 0.556}, "spinning": {"fig17": 0.392, "fig18": 0.481},
    "wiping": {"fig17": 0.5277, "fig18": 0.467},
    "note": "그림 18은 판독 값(래스터 아님, 그림 판독), n은 Wilson 구간 폭에서 27로 추정(55.6% = 15/27)"}
res["V12_fisher_scoop"] = {"fig17_17of20_vs_fig18_15of27": round(fx(17, 20, 15, 27), 4)}

# V13: 그림 18 왼쪽 처리량 (판독 값, 성공/시간)
meta = {"30": 12.7, "50": 16.2, "80": 15.8, "100": 22.7}
nometa = {"30": 9.1, "50": 11.4, "80": 11.9, "100": 9.3}
res["V13_fig18_left"] = {
    "with_meta": meta, "without_meta": nometa,
    "ratio_with_100_over_30": round(meta["100"] / meta["30"], 2),
    "ratio_without_100_over_30": round(nometa["100"] / nometa["30"], 2),
    "with_meta_diff_80_to_100": round(meta["100"] - meta["80"], 1),
    "with_meta_diff_50_to_80": round(meta["80"] - meta["50"], 1),
    "gap_at_30": round(meta["30"] - nometa["30"], 1), "gap_at_100": round(meta["100"] - nometa["100"], 1)}

# V14: 사람 연구 (부록 F, 그림 22)
human = {"progress_pct": 90.9, "success_pct": 80.6, "stated_trials": 30}
cand = {}
for n in (30, 31):
    cand[n] = {"success_k_near": [k for k in range(n + 1) if abs(100 * k / n - 80.6) < 0.05],
               "progress_points_near": [s for s in range(6 * n + 1) if abs(100 * s / (6 * n) - 90.9) < 0.05],
               "k_30_pct": [round(100 * k / n, 2) for k in (24, 25)]}
pi07 = {}
for n in (10, 15, 20, 25, 30, 45):
    pi07[n] = {"success_80_ok": (0.8 * n).is_integer(),
               "progress_points_85.6": [s for s in range(6 * n + 1) if abs(100 * s / (6 * n) - 85.6) < 0.05]}
res["V14_human_study"] = {"human": human, "candidates_by_n": cand, "pi07_gc_n_candidates": pi07,
                          "fisher_25of31_vs_12of15": round(fx(25, 31, 12, 15), 3),
                          "wilson_25of31": [round(x, 3) for x in wilson(25, 31)],
                          "wilson_12of15": [round(x, 3) for x in wilson(12, 15)]}

# V10: 그림 16 코칭 대 자율 (작업 진행도 %, 벡터 판독)
c = {"scoop_rice": (70.0, 70.0), "reverse_fridge": (73.7, 65.0), "air_fryer_load": (97.2, 89.7),
     "bagel": (79.5, 59.8), "air_fryer_unload": (79.6, 69.1)}
res["V10_fig16"] = {k: {"coaching": a, "autonomous": b, "diff": round(a - b, 1)} for k, (a, b) in c.items()}
res["V10_fig16"]["mean_diff"] = round(sum(a - b for a, b in c.values()) / 5, 1)

# V9: 그림 15 (작업 진행도 %, 벡터 판독; 해칭 막대는 GC)
res["V9_fig15"] = {"air_fryer_load": {"pi05": 13.2, "pi06": 25.0, "pi07": 97.5, "gc": 97.2},
                   "air_fryer_unload": {"pi05": 14.0, "pi06": 23.3, "pi07": 76.7, "gc": 79.6},
                   "bagel": {"pi05": 28.6, "pi06": 47.6, "pi07": 47.4, "gc": 79.5}}

# V8: 그림 12 (작업 진행도 %, 그림 판독 값)
res["V8_fig12"] = {"table_setting": [63.1, 79.1, 81], "bag_backpack": [38.7, 51.1, 60], "tupperware": [46.7, 77.2, 86],
                   "shirt_bagging": [57.1, 66.1, 93], "towel_folding": [6.7, 44.2, 59.1, 77], "shirt_folding": [2.2, 43.3, 68, 85.6]}
# 그림 12와 부록 그림 20의 관절 제어 막대 (π0.5, π0.6)가 같은 값인지
res["V8b_fig20_joint_equals_fig12"] = {
    "fig20": {"table": [63.1, 79.1], "bag": [38.7, 51.1], "tupper": [46.7, 77.2], "shirtbag": [57.1, 66.1],
              "towel": [6.7, 44.2], "shirt": [2.2, 43.3]}, "match": True}

# V15: 산술 점검
res["V15_arithmetic"] = {
    "rtc_delay_steps_max_ms_at_50Hz": 12 * 1000 / 50, "rtc_delay_steps_max_ms_at_20Hz": 12 * 1000 / 20,
    "chunk_seconds_50Hz": 50 / 50, "chunk_seconds_20Hz": 50 / 20,
    "exec_seconds_H15_H25_at_50Hz": [15 / 50, 25 / 50],
    "worldmodel_ms_per_denoise_step": 1250 / 25,
    "dropout_subgoal_and_no_subtask_frac": 0.25 * 0.30,
    "p_speed_label_missing": round(1 - 0.85 * 0.95, 4),
    "p_all_metadata_missing": round(0.15 + 0.85 * 0.05 ** 3, 4),
    "cfg_effective_scale_w": [1 + b for b in (1.3, 1.7, 2.2)],
    "param_sum_B": round(4 + 0.86, 2),
    "wm_tokens_single_branch_est": {"vit_3x768": 3 * 32 * 24, "vae_3_cur_3_tgt_6x768": 6 * 32 * 24,
                                    "total": 3 * 32 * 24 + 6 * 32 * 24},
    "speed_bin_check": "1750–2250 → 2000, 폭 500 (일관)",
    "speed_8000_steps_seconds_at_50Hz": 8000 / 50,
}

# V5: 그림 9 (지시 따르기 성공률, 래스터 그림의 픽셀 판독 ±2%p). Wilson 구간 폭에서 유효 시행 수를 추정
def wilson_n(p, lo, hi):
    best = None
    for n in range(10, 400):
        l, h = wilson(round(p * n), n)
        e = abs(l - lo) + abs(h - hi)
        if best is None or e < best[0]:
            best = (e, n)
    return best[1]


fig9 = {"kitchen": {"pi05": (0.28, 0.17, 0.40), "pi06": (0.537, 0.424, 0.646), "pi07": (0.863, 0.769, 0.923)},
        "bedroom": {"pi05": (0.253, 0.169, 0.364), "pi06": (0.545, 0.439, 0.653), "pi07": (0.728, 0.622, 0.812)}}
v5 = {}
for env, d in fig9.items():
    ns = {m: wilson_n(*v) for m, v in d.items()}
    n = int(round(sum(ns.values()) / 3))
    k = {m: round(d[m][0] * n) for m in d}
    v5[env] = {"n_eff_by_arm": ns, "n_used": n, "k": k,
               "pi07_vs_pi06_p": fx(k["pi07"], n, k["pi06"], n), "pi06_vs_pi05_p": fx(k["pi06"], n, k["pi05"], n)}
res["V5_fig9"] = v5


def min_detectable(n1, n2, p0):
    """기준 p0(n2회) 대비 n1회에서 Fisher p<0.05가 되는 가장 작은 '하락' 폭(성공률 차)."""
    k2 = round(p0 * n2)
    for k1 in range(min(n1, round(p0 * n1)), -1, -1):
        if fx(k1, n1, k2, n2) < 0.05:
            return round(k2 / n2 - k1 / n1, 3)
    return None


res["V3b_min_detectable_diff"] = {"laundry_ts_78v77_p0.93": min_detectable(78, 77, 0.935),
                                  "laundry_div_35v20_p0.70": min_detectable(35, 20, 0.70),
                                  "espresso_19v20_p0.95": min_detectable(19, 20, 0.95),
                                  "box_16v16_p0.875": min_detectable(16, 16, 0.875),
                                  "n20v20_p0.5": min_detectable(20, 20, 0.5)}
res["V8c_fig20_ee_read"] = {"note": "해칭 막대(EE)는 래스터 판독 ±3%p", "pi05_joint_vs_ee": {"table": [63, 71], "bag": [39, 57], "tupper": [47, 45], "shirtbag": [57, 65], "towel": [7, 3], "shirt": [2, 1]},
                           "pi06_joint_vs_ee": {"table": [79, 79], "bag": [51, 58], "tupper": [77, 82], "shirtbag": [66, 62], "towel": [44, 54], "shirt": [43, 37]}}

# V16: 그림 18 오른쪽 (래스터 픽셀 판독, n=27 가정: 쌀 뜨기 15/27 = 55.6%에서 Wilson 폭으로 추정)
v16 = {}
for name, (a, b, c) in {"scoop_rice": (15, 14, 9), "spinning": (13, 9, 2), "wiping": (13, 13, 2)}.items():
    n = 27
    v16[name] = {"k_full_random_nodiverse": [a, b, c], "n": n,
                 "full_vs_nodiverse_p": round(fx(a, n, c, n), 4), "random_vs_nodiverse_p": round(fx(b, n, c, n), 4),
                 "full_vs_random_p": round(fx(a, n, b, n), 3)}
res["V16_fig18_right"] = v16

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT)
for k in ("V5_fig9", "V3b_min_detectable_diff", "V3_fig6_pi07_vs_specialist", "V4_fig7_ablation_success", "V6_fig10", "V11_fig17", "V12_fisher_scoop", "V13_fig18_left", "V14_human_study", "V15_arithmetic"):
    print(k, json.dumps(res[k], ensure_ascii=False))
print({k: (v and (v['k'], v['n'], v['fit_err'])) for k, v in fits.items()})
