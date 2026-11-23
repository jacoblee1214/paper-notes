"""MaskWAM (arXiv:2606.13515 v1) 표 수치 재계산과 검정.

입력은 논문 표 1, 2, 3, 4, 5와 그림 4, 8에서 옮긴 값이다. 시행 수는 논문이 적은 값을 쓰고,
적지 않은 곳은 가정을 밝힌다(LIBERO 스위트당 500회, RoboTwin 과제당 100회).
검사 항목 V1-V14는 analysis/03_analysis.md 의 번호와 같다.
결과는 analysis/recompute_results.json 에 저장한다.

실행: python3 analysis/recompute.py   (numpy, scipy; 수 초)
"""
import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
R = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def feasible(pct, n, decimals=1):
    """보고된 백분율(소수 decimals자리 반올림)이 k/n 로 나올 수 있는 정수 k 목록."""
    half = 0.5 * 10 ** (-decimals)
    lo, hi = (pct - half) * n / 100, (pct + half) * n / 100
    return [k for k in range(int(np.floor(lo)) - 1, int(np.ceil(hi)) + 2) if lo - 1e-9 <= k <= hi + 1e-9]


def mde(n1, n2, p0, power=0.8, alpha=0.05):
    """두 비율 비교에서 80% 검정력으로 잡을 수 있는 최소 차이(정규 근사, 단위 pp)."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    d = 0.0
    for d in np.arange(0.005, 0.6, 0.0005):
        p1 = min(p0 + d, 0.999)
        se = np.sqrt(p0 * (1 - p0) / n1 + p1 * (1 - p1) / n2)
        if d / se >= za + zb:
            return round(100 * d, 1)
    return None


# ---------------------------------------------------------------- V1: 표 1 LIBERO
lib = {
    "WorldVLA": [87.6, 96.2, 83.4, 60.0, 81.8],
    "GR00T-N1": [94.4, 97.6, 93.0, 90.6, 93.9],
    "pi0": [96.8, 98.8, 95.8, 85.2, 94.1],
    "pi0.5": [98.6, 98.2, 98.0, 92.4, 96.8],
    "Motus": [96.8, 99.8, 96.6, 97.6, 97.7],
    "FastWAM": [98.2, 100.0, 97.0, 95.2, 97.6],
    "Ours-RGB": [96.8, 99.6, 97.0, 95.8, 97.3],
    "Ours-Mask": [97.2, 99.8, 97.4, 96.0, 97.6],
    "Ours": [98.8, 100.0, 98.2, 96.4, 98.4],
}
N_LIB = 500  # 가정: 스위트당 10과제 x 50회 (공식 eval 기본 --num_trials 50)
v1 = {}
for m, row in lib.items():
    mean4 = float(np.mean(row[:4]))
    ks = [feasible(x, N_LIB) for x in row[:4]]
    v1[m] = {"stated_avg": row[4], "mean_of_4": round(mean4, 3),
             "avg_match": abs(np.floor(mean4 * 10 + 0.5 + 1e-9) / 10 - row[4]) < 1e-9 or abs(mean4 - row[4]) < 0.051,
             "mean_ends_in_half": abs(mean4 * 100 - np.floor(mean4 * 100)) < 1e-9 and int(round(mean4 * 100)) % 10 == 5,
             "integer_ok_n500": all(len(k) > 0 for k in ks),
             "counts_n500": [k[0] if k else None for k in ks],
             "total_2000": int(sum(k[0] for k in ks)) if all(ks) else None}
R["V1_libero_rows"] = v1
tot = {m: v1[m]["total_2000"] for m in lib}
tests = {}
for other in ["Ours-RGB", "Ours-Mask", "FastWAM", "Motus", "pi0.5"]:
    tests["Ours vs " + other] = {
        "diff_pp": round(100 * (tot["Ours"] - tot[other]) / 2000, 2),
        "fisher_p_pooled2000": fisher(tot["Ours"], 2000, tot[other], 2000)}
R["V2_libero_tests"] = tests
R["V2_libero_wilson_ours"] = wilson(tot["Ours"], 2000)
suite_names = ["Spatial", "Object", "Goal", "Long"]
R["V2_per_suite_rank"] = {s: sorted(lib, key=lambda m: -lib[m][i])[:3] for i, s in enumerate(suite_names)}
R["V2_ours_minus_motus_per_suite"] = [round(lib["Ours"][i] - lib["Motus"][i], 1) for i in range(4)]
R["V2_ours_minus_rgb_per_suite"] = [round(lib["Ours"][i] - lib["Ours-RGB"][i], 1) for i in range(4)]
R["V2_ours_minus_mask_per_suite"] = [round(lib["Ours"][i] - lib["Ours-Mask"][i], 1) for i in range(4)]
R["V2_pi05_spatial_note"] = {"maskwam_table1": 98.6, "fastwam_paper_table2_and_openpi": 98.8,
                           "avg_with_98.8": round(np.mean([98.8, 98.2, 98.0, 92.4]), 3)}
R["V2_mde_pp_n2000_base97"] = mde(2000, 2000, 0.97)

# ---------------------------------------------------------------- V3: 표 2 RoboTwin
rt_tasks = ["Hammer", "Bell", "Card", "Burger", "Stand", "Shoe"]
rt = {
    "pi0": [68, 72, 81, 79, 63, 74, 72.8],
    "FastWAM": [83, 87, 92, 94, 80, 90, 87.7],
    "Ours-RGB": [82, 87, 91, 93, 79, 92, 87.3],
    "Ours-Mask": [85, 90, 93, 93, 81, 91, 88.8],
    "Ours": [88, 93, 95, 97, 85, 95, 92.2],
}
N_RT = 100  # 가정: 과제당 100회 (정수 값이 모두 100회로 설명됨)
v3 = {m: {"stated": r[6], "mean_of_6": round(float(np.mean(r[:6])), 3),
          "match": abs(round(float(np.mean(r[:6])) + 1e-9, 1) - r[6]) < 0.051} for m, r in rt.items()}
R["V3_robotwin_rows"] = v3
rt_tests = {}
for other in ["pi0", "FastWAM", "Ours-RGB", "Ours-Mask"]:
    a, b = np.array(rt["Ours"][:6]), np.array(rt[other][:6])
    wins, losses = int((a > b).sum()), int((a < b).sum())
    rt_tests["Ours vs " + other] = {
        "diff_pp": round(float((a - b).mean()), 2),
        "wins_losses_ties": [wins, losses, int((a == b).sum())],
        "sign_test_p": float(stats.binomtest(wins, wins + losses, 0.5).pvalue),
        "fisher_p_pooled600_n100": fisher(int(a.sum()), 600, int(b.sum()), 600),
        "paired_t_p": float(stats.ttest_rel(a, b).pvalue)}
R["V4_robotwin_tests"] = rt_tests
a, b = np.array(rt["Ours-Mask"][:6]), np.array(rt["Ours-RGB"][:6])
R["V4_mask_only_vs_rgb_only"] = {"diff_pp": round(float((a - b).mean()), 2),
                                 "wins_losses_ties": [int((a > b).sum()), int((a < b).sum()), int((a == b).sum())],
                                 "fisher_p_pooled600": fisher(int(a.sum()), 600, int(b.sum()), 600)}
a, b = np.array(rt["FastWAM"][:6]), np.array(rt["Ours-RGB"][:6])
R["V4_fastwam_vs_rgb_only"] = {"diff_pp": round(float((a - b).mean()), 2),
                               "fisher_p_pooled600": fisher(int(a.sum()), 600, int(b.sum()), 600)}
R["V4_fastwam_paper_randomized_same_names"] = {
    "note": "Fast-WAM 논문 표 3 Rand. 열에서 이름이 같은 과제(과제 대응은 이름 기준 추정)",
    "Beat Block Hammer": 97, "Click Bell": 100, "Move Playingcard Away": 100, "Place Burger Fries": 99,
    "maskwam_table2_fastwam": {"Hammer": 83, "Bell": 87, "Card": 92, "Burger": 94}}
R["V4_mde_pp_n600_base88"] = mde(600, 600, 0.88)

# ---------------------------------------------------------------- V5: 표 3 실로봇 language-clear
rc = {
    "pi0": [57, 54, 54, 58, 55.8],
    "pi0.5": [83, 55, 74, 77, 72.3],
    "FastWAM": [88, 76, 77, 75, 79.0],
    "Ours-RGB": [86, 77, 76, 78, 79.3],
    "Ours": [91, 82, 81, 83, 84.3],
}
v5 = {m: {"stated": r[4], "mean_of_4": float(np.mean(r[:4])),
          "match_round_half_up": abs(np.floor(np.mean(r[:4]) * 10 + 0.5) / 10 - r[4]) < 1e-9} for m, r in rc.items()}
R["V5_real_clear_rows"] = v5
rct = {}
for other in ["pi0", "pi0.5", "FastWAM", "Ours-RGB"]:
    a, b = np.array(rc["Ours"][:4]), np.array(rc[other][:4])
    rct["Ours vs " + other] = {
        "diff_pp": round(float((a - b).mean()), 2),
        "fisher_p_pooled400": fisher(int(a.sum()), 400, int(b.sum()), 400),
        "per_task_fisher_p": [round(fisher(int(x), 100, int(y), 100), 3) for x, y in zip(a, b)],
        "wins_losses": [int((a > b).sum()), int((a < b).sum())]}
R["V6_real_clear_tests"] = rct
R["V6_wilson_ours_337_of_400"] = wilson(337, 400)
R["V6_mde_pp_n400_base80"] = mde(400, 400, 0.80)
R["V6_mde_pp_n100_base80"] = mde(100, 100, 0.80)

# ---------------------------------------------------------------- V7: 표 5, 그림 4, 표 4 (language-ambiguous)
settings = ["ID", "Distractors", "Novel", "Lighting"]
amb = {
    "pi0-mask": {"ID": [96.7, 93.3, 38.3, 23.3, 62.9], "Distractors": [86.7, 75.0, 31.7, 18.3, 52.9],
                 "Novel": [71.7, 65.0, 26.7, 15.0, 44.6], "Lighting": [75.0, 68.3, 28.3, 13.3, 46.3]},
    "pi0-coord": {"ID": [53.3, 88.3, 20.0, 5.0, 41.7], "Distractors": [46.7, 61.7, 11.7, 0.0, 30.0],
                  "Novel": [41.7, 51.7, 8.3, 0.0, 25.4], "Lighting": [48.3, 66.7, 15.0, 3.3, 33.3]},
    "FastWAM-coord": {"ID": [35.0, 55.0, 13.3, 1.7, 26.3], "Distractors": [15.0, 23.3, 5.0, 0.0, 10.8],
                      "Novel": [18.3, 28.3, 6.7, 0.0, 13.3], "Lighting": [31.7, 36.7, 8.3, 0.0, 19.2]},
    "Ours": {"ID": [100.0, 100.0, 88.3, 83.3, 92.9], "Distractors": [100.0, 98.3, 85.0, 78.3, 90.4],
             "Novel": [86.7, 76.7, 70.0, 65.0, 74.6], "Lighting": [93.3, 91.7, 73.3, 68.3, 81.7]},
}
N_AMB = 60
counts = {}
v7 = {}
for m, d in amb.items():
    counts[m] = {}
    for s in settings:
        ks = [feasible(x, N_AMB) for x in d[s][:4]]
        counts[m][s] = [k[0] if k else None for k in ks]
        mean_k = sum(counts[m][s]) / 4
        v7[f"{m}/{s}"] = {"integer_ok_n60": all(len(k) > 0 for k in ks), "counts": counts[m][s],
                          "avg_from_counts": round(100 * mean_k / N_AMB, 2), "stated_avg": d[s][4],
                          "avg_match": abs(round(100 * mean_k / N_AMB + 1e-9, 1) - d[s][4]) < 0.051}
R["V7_table5_integer_and_avg"] = v7
overall = {}
for m in amb:
    tot_k = sum(sum(counts[m][s]) for s in settings)
    overall[m] = {"k": tot_k, "n": 960, "pct": round(100 * tot_k / 960, 2),
                  "mean_of_stated_setting_avgs": round(float(np.mean([amb[m][s][4] for s in settings])), 2),
                  "wilson95": wilson(tot_k, 960)}
R["V8_overall"] = overall
R["V8_fig4_stated"] = {"pi0-mask": 51.7, "pi0-coord": 32.6, "FastWAM-coord": 17.4, "Ours": 84.9}
R["V8_gap_ours_minus_best_baseline_pp"] = round(overall["Ours"]["mean_of_stated_setting_avgs"]
                                                 - overall["pi0-mask"]["mean_of_stated_setting_avgs"], 2)
R["V8_fisher_ours_vs_pi0mask_pooled960"] = fisher(overall["Ours"]["k"], 960, overall["pi0-mask"]["k"], 960)
# 표 4: 사용 가능한 정수 복원
t4 = {}
for name, val in [("Ours-no-pred", 21.6), ("Ours-coord", 18.2), ("Ours", 84.9)]:
    t4[name] = {"stated": val, "k_of_960": feasible(val, 960)}
R["V9_table4_integer"] = t4
k_coord = t4["Ours-coord"]["k_of_960"][0]
k_nopred = t4["Ours-no-pred"]["k_of_960"][0]
R["V9_ours_coord_vs_fastwam_coord"] = {
    "diff_pp": round(100 * k_coord / 960 - overall["FastWAM-coord"]["pct"], 2),
    "fisher_p_pooled960": fisher(k_coord, 960, overall["FastWAM-coord"]["k"], 960)}
R["V9_ours_coord_vs_pi0coord"] = {
    "diff_pp": round(100 * k_coord / 960 - overall["pi0-coord"]["pct"], 2),
    "fisher_p_pooled960": fisher(k_coord, 960, overall["pi0-coord"]["k"], 960)}
R["V9_nopred_vs_pi0mask"] = {
    "diff_pp": round(100 * k_nopred / 960 - overall["pi0-mask"]["pct"], 2),
    "fisher_p_pooled960": fisher(k_nopred, 960, overall["pi0-mask"]["k"], 960)}
R["V9_ours_vs_nopred_diff_pp"] = round(overall["Ours"]["pct"] - 100 * k_nopred / 960, 2)
# 군집 민감도: 16칸(과제 4 x 설정 4)의 성공률을 단위로 한 Ours 평균의 95% 구간
cells = np.array([x for s in settings for x in amb["Ours"][s][:4]])
se = cells.std(ddof=1) / np.sqrt(len(cells))
R["V10_ours_cell_level_ci95_pm"] = round(float(1.96 * se), 2)
R["V10_ours_cell_values_sd"] = round(float(cells.std(ddof=1)), 2)
# 설정별 하락
R["V10_ours_drop_from_ID_pp"] = {s: round(amb["Ours"]["ID"][4] - amb["Ours"][s][4], 1) for s in settings[1:]}
R["V10_ours_task78_vs_task56_ID"] = {"task5_6_mean": float(np.mean(amb["Ours"]["ID"][:2])),
                                     "task7_8_mean": float(np.mean(amb["Ours"]["ID"][2:4]))}

# ---------------------------------------------------------------- V11: 그림 8 (그림 판독 값)
fig8 = {
    "Task5": {"online": 100.0, "erosion": [(0.40, 100), (0.70, 100), (0.82, 100)],
              "dilation": [(0.35, 98.3), (0.55, 100), (0.84, 100)], "shift": [(0.10, 85.0), (0.32, 91.7), (0.72, 95.0)],
              "dropout": [(0.23, 96.7), (0.50, 100), (0.70, 100)]},
    "Task8": {"online": 83.3, "erosion": [(0.40, 80.0), (0.68, 83.3), (0.82, 83.3)],
              "dilation": [(0.35, 75.0), (0.55, 78.3), (0.84, 83.3)], "shift": [(0.10, 51.7), (0.32, 68.3), (0.72, 75.0)],
              "dropout": [(0.23, 75.0), (0.50, 80.0), (0.70, 83.3)]},
}
f8 = {}
for task, d in fig8.items():
    vals = [d["online"]] + [v for k, pts in d.items() if k != "online" for _, v in pts]
    f8[task] = {
        "n_values": len(vals),
        "values_possible_with_20_trials_multiples_of_5": sum(1 for v in vals if abs(v / 5 - round(v / 5)) < 0.02),
        "feasible_if_60_trials": sum(1 for v in vals if feasible(v, 60)),
        "non_multiples_of_5": sorted({v for v in vals if abs(v / 5 - round(v / 5)) > 0.02}),
        "k_over_60": {str(v): feasible(v, 60)[0] for v in sorted(set(vals))},
    }
R["V11_fig8_reading"] = f8
R["V11_fig8_online_equals_table5_ID"] = {"Task5": [100.0, 100.0], "Task8": [83.3, 83.3]}
R["V11_fig8_shift_lowest_task8_wilson60"] = wilson(31, 60)
R["V11_fig8_task8_online_wilson60"] = wilson(50, 60)
R["V11_fig8_task8_shift_vs_online_fisher_n60"] = fisher(31, 60, 50, 60)
R["V11_fig8_task8_dropout23_vs_online_fisher_n60"] = fisher(45, 60, 50, 60)

# ---------------------------------------------------------------- V12: 서술·산수 점검
R["V12_text_checks"] = {
    "abstract_33.2": round(overall["Ours"]["mean_of_stated_setting_avgs"] - 51.7, 2),
    "q1_spatial_plus2.0": round(98.8 - 96.8, 1), "q1_goal_plus1.2": round(98.2 - 97.0, 1),
    "libero_rgb_to_full": round(98.4 - 97.3, 1),
    "robotwin_vs_pi0": round(92.2 - 72.8, 1), "robotwin_vs_fastwam": round(92.2 - 87.7, 1),
    "task1_86_to_91": 91 - 86, "task3_76_to_81": 81 - 76,
    "ann_50_episodes_3min_sec_per_episode": 180 / 50,
    "ann_91pct_no_correction_of_100": 91,
}

# ---------------------------------------------------------------- V13: 토큰·파라미터 산술 (코드 설정 기준)
H, W = 384, 320
lat_T, lat_H, lat_W = 1 + 8 // 4, H // 32, W // 32  # VAE 공간 16배 x 패치 2
video_tokens = lat_T * lat_H * lat_W
d, ffn, L = 3072, 14336, 30
dit_params = L * (4 * d * d + 4 * d * d + 2 * d * ffn)  # 자기어텐션 + 교차어텐션 + FFN (근사)
h_a, ffn_a = 1024, 4096
act_params = L * (3 * h_a * d + d * h_a + 2 * h_a * ffn_a)
flops_video_pass = 2 * dit_params * video_tokens
flops_action_step = 2 * act_params * 48
R["V13_budget"] = {
    "latent_grid": [lat_T, lat_H, lat_W], "video_tokens": video_tokens, "action_tokens": 48,
    "video_dit_params_B_approx": round(dit_params / 1e9, 2), "action_expert_params_B_approx": round(act_params / 1e9, 2),
    "bf16_weights_GB": round((dit_params + act_params) * 2 / 2**30, 1),
    "adamw_fp32_states_GB_if_full_finetune": round((dit_params + act_params) * 16 / 2**30, 0),
    "tflops_per_video_pass": round(flops_video_pass / 1e12, 2),
    "tflops_per_chunk_inference_two_video_passes_plus_10_action_steps": round(
        (2 * flops_video_pass + 10 * flops_action_step) / 1e12, 2),
    "note": "파라미터와 FLOPs는 config 상수로 계산한 근사(교차어텐션 키·값 입력 차원, 어텐션 점수 항 무시). 측정값이 아니다.",
}

def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


OUT.write_text(json.dumps(_clean(R), ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT)
for k in ["V2_libero_tests", "V4_robotwin_tests", "V6_real_clear_tests", "V8_overall", "V9_ours_coord_vs_fastwam_coord",
          "V9_nopred_vs_pi0mask", "V11_fig8_reading", "V13_budget"]:
    print(k, json.dumps(R[k], ensure_ascii=False)[:900])
