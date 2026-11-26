"""NativeMEM (arXiv:2607.06678 v2) 표 수치 재계산과 일관성 검사.

입력: 논문 v2 표 1–8과 그림 4(b)에 인쇄된 값(아래 딕셔너리에 옮김). 시행 수는 논문 본문(§4.1, 부록 C)의 값을 쓴다:
시뮬레이션 100회, 실로봇 50회. LIBERO(표 3)와 그림 4(b)의 시행 수는 논문에 없어 가정을 문장에 밝힌다.
출력: analysis/recompute_results.json 과 표준 출력. 항목 이름 V1, V2, ... 는 03_analysis.md 의 번호와 같다.
실행: python3 analysis/recompute.py   (numpy, scipy; 수 초)
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
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(100 * (c - h), 1), round(100 * (c + h), 1)


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def mean(xs):
    return sum(xs) / len(xs)


# ---------------- 표 1 (시뮬레이션, 100회) ----------------
T1_COLS = ["click_buttons", "click_buttons_hard", "swap_blocks", "put_back_block", "observe_and_pickup"]
T1 = {
    "pi0.5": ([0, 0, 24, 11, 9], 8.8),
    "X-VLA": ([7, 12, 16, 18, 9], 12.4),
    "MemER": ([12, 8, 18, 12, 2], 10.4),
    "Mem-0": ([0, 1, 67, 90, 4], 32.4),
    "HAMLET*": ([4, 17, 11, 3, 10], 9.0),
    "MEM-short*": ([0, 39, 4, 15, 6], 12.8),
    "Ours": ([94, 88, 94, 100, 44], 84.0),
}
v1 = {}
for name, (vals, avg) in T1.items():
    v1[name] = {"computed": round(mean(vals), 2), "reported": avg, "match": abs(mean(vals) - avg) < 0.05}
R["V1_table1_row_means"] = v1
# 열별 최고가 Ours인지
best = {}
for j, c in enumerate(T1_COLS):
    others = {n: v[0][j] for n, v in T1.items() if n != "Ours"}
    top = max(others, key=others.get)
    k_o, k_b = T1["Ours"][0][j], others[top]
    best[c] = {"ours": k_o, "best_baseline": top, "baseline": k_b,
               "fisher_p": fisher(k_o, 100, k_b, 100), "ours_wilson95": wilson(k_o, 100),
               "baseline_wilson95": wilson(k_b, 100)}
R["V2_table1_ours_vs_best_baseline"] = best
pooled = {n: sum(v[0]) for n, v in T1.items()}  # 500회 합
R["V3_table1_pooled_500"] = {
    n: {"k": k, "pct": k / 5, "wilson95": wilson(k, 500)} for n, k in pooled.items()
}
R["V3_table1_pooled_500"]["Ours_vs_Mem-0_fisher_p"] = fisher(pooled["Ours"], 500, pooled["Mem-0"], 500)
# 세 RMBench 과제만 (swap_blocks, put_back_block, observe_and_pickup)
rm = {n: sum(v[0][2:]) for n, v in T1.items()}
R["V3_table1_rmbench3_pooled_300"] = {
    "Ours": {"k": rm["Ours"], "pct": rm["Ours"] / 3},
    "Mem-0": {"k": rm["Mem-0"], "pct": rm["Mem-0"] / 3},
    "pi0.5": {"k": rm["pi0.5"], "pct": rm["pi0.5"] / 3},
}

# ---------------- 표 2 (실로봇, 50회) ----------------
T2 = {  # click S1-S3, putback S1-S2, grocery S1-S3, avg
    "pi0.5": ([16, 8, 2], [72, 14], [90, 58, 28], 14.7),
    "pi0.5+RTC": ([26, 18, 16], [56, 24], [74, 66, 64], 34.7),
    "Mem-0": ([36, 4, 0], [0, 0], [6, 0, 0], 0.0),
    "MEM-short*": ([72, 52, 40], [6, 0], [72, 56, 52], 30.7),
    "Ours": ([100, 98, 96], [100, 100], [100, 100, 100], 98.7),
}
v4 = {}
for name, (c, p, g, avg) in T2.items():
    finals = [c[-1], p[-1], g[-1]]
    allv = c + p + g
    mono = all(a >= b for a, b in zip(c, c[1:])) and all(a >= b for a, b in zip(p, p[1:])) and all(a >= b for a, b in zip(g, g[1:]))
    even = all(x % 2 == 0 for x in allv)
    ks = [x // 2 for x in finals]
    v4[name] = {"mean_of_finals": round(mean(finals), 2), "reported": avg,
                "match": abs(mean(finals) - avg) < 0.05, "monotone_cumulative": mono,
                "all_values_multiples_of_2pct": even, "pooled_k_of_150": sum(ks),
                "pooled_pct": round(sum(ks) / 150 * 100, 2), "pooled_wilson95": wilson(sum(ks), 150)}
R["V4_table2_means_and_integrality"] = v4
v5 = {}
for j, task in enumerate(["click_buttons", "put_back_block", "grocery_unseen"]):
    for base in ["pi0.5+RTC", "MEM-short*", "pi0.5", "Mem-0"]:
        a = [T2["Ours"][0], T2["Ours"][1], T2["Ours"][2]][j][-1] // 2
        b = [T2[base][0], T2[base][1], T2[base][2]][j][-1] // 2
        v5[f"{task}_Ours_vs_{base}"] = {"k_ours": a, "k_base": b, "fisher_p": fisher(a, 50, b, 50)}
R["V5_table2_final_stage_fisher_n50"] = v5
R["V5_table2_pooled"] = {
    "Ours_vs_RTC_fisher_p": fisher(148, 150, 52, 150),
    "Ours_vs_MEM-short_fisher_p": fisher(148, 150, 46, 150),
}


# 단계 간 조건부 진행률 (표 2): S(k+1)/S(k). 앞 단계 성공 수가 0이면 None
cond = {}
for name, (c, p_, g, avg) in T2.items():
    def ratios(v):
        return [round(b / a, 3) if a else None for a, b in zip(v, v[1:])]
    cond[name] = {"click_buttons": ratios(c), "put_back_block": ratios(p_), "grocery": ratios(g),
                  "first_stage_S1": {"click": c[0], "put_back": p_[0], "grocery": g[0]}}
R["V5_table2_stage_continuation"] = cond

# ---------------- 표 3 LIBERO (시행 수 가정: 스위트당 500회) ----------------
T3 = {"pi0.5": ([97.8, 98.6, 92.8, 84.4], 93.4),
      "NativeMEM128": ([94.4, 98.2, 94.4, 88.6], 93.9),
      "NativeMEM512": ([95.2, 98.2, 95.6, 87.0], 94.0)}
v6 = {}
for n, (vals, avg) in T3.items():
    ks = [round(v * 5) for v in vals]   # n=500 가정
    v6[n] = {"mean": round(mean(vals), 2), "reported": avg, "match": abs(mean(vals) - avg) < 0.05,
             "k_of_500": ks, "integer_feasible_n500": all(abs(k / 5 - v) < 1e-9 for k, v in zip(ks, vals))}
R["V6_table3_libero"] = v6
suites = ["Spatial", "Object", "Goal", "LIBERO-10"]
cmp = {}
for n in ["NativeMEM128", "NativeMEM512"]:
    for j, s in enumerate(suites):
        k1 = round(T3[n][0][j] * 5)
        k0 = round(T3["pi0.5"][0][j] * 5)
        cmp[f"{n}_vs_pi0.5_{s}"] = {"diff_pp": round(T3[n][0][j] - T3["pi0.5"][0][j], 1), "fisher_p_n500": fisher(k1, 500, k0, 500)}
    k1 = round(T3[n][1] * 20)
    k0 = round(T3["pi0.5"][1] * 20)
    cmp[f"{n}_vs_pi0.5_avg"] = {"diff_pp": round(T3[n][1] - T3["pi0.5"][1], 1), "fisher_p_n2000": fisher(k1, 2000, k0, 2000)}
R["V7_libero_vs_pi05_assumed_n500_per_suite"] = cmp

# ---------------- 표 4 절제 (100회) ----------------
T4 = {
    "Unfrozen VLA": ([38, 24, 45, 0, 9], 23.2),
    "w/o Stage1": ([94, 80, 92, 17, 7], 58.0),
    "Sparse update 0.5Hz": ([83, 63, 91, 24, 9], 53.8),
    "Short horizon 2s": ([26, 87, 0, 18, 39], 34.0),
    "Short horizon 4s": ([69, 87, 0, 18, 37], 43.0),
    "Short horizon 6s": ([80, 88, 0, 18, 44], 46.0),
    "Ours": ([94, 88, 94, 100, 44], 84.0),
}
v8 = {}
for n, (vals, avg) in T4.items():
    s = sum(vals)
    v8[n] = {"sum": s, "computed_mean": round(s / 5, 2), "reported": avg,
             "match": abs(s / 5 - avg) < 0.05, "sum_implied_by_reported": round(avg * 5, 1),
             "diff_in_sum": round(s - avg * 5, 1)}
R["V8_table4_row_means"] = v8
abl = {}
for n in T4:
    if n == "Ours":
        continue
    row = {}
    for j, c in enumerate(T1_COLS):
        k_a, k_o = T4[n][0][j], T4["Ours"][0][j]
        row[c] = {"abl": k_a, "ours": k_o, "fisher_p": fisher(k_a, 100, k_o, 100)}
    abl[n] = row
R["V9_table4_ablation_vs_ours_fisher_n100"] = abl
# Table 4 Ours 행과 Table 1 Ours 행 동일성
R["V8_table4_ours_equals_table1_ours"] = T4["Ours"][0] == T1["Ours"][0]

# ---------------- 표 5, 6, 7, 8과 그림 4(b) ----------------
T5 = {"Clean": [94, 100, 44], "Hard": [85, 86, 32]}
R["V10_table5_means"] = {k: {"mean": round(mean(v), 2), "reported": {"Clean": 79.3, "Hard": 67.7}[k]} for k, v in T5.items()}
R["V10_table5_clean_equals_table1_ours_rmbench3"] = T5["Clean"] == T1["Ours"][0][2:]
R["V10_table5_drop_fisher_n100"] = {c: fisher(T5["Hard"][j], 100, T5["Clean"][j], 100)
                                    for j, c in enumerate(["swap_blocks", "put_back_block", "observe_and_pickup"])}
R["V10_table5_pooled_n300_fisher"] = fisher(sum(T5["Hard"]), 300, sum(T5["Clean"]), 300)
T6 = {"pi0.5": [24, 11, 9], "NativeHistory_K4": [9, 18, 20], "Ours": [94, 100, 44]}
R["V11_table6_matches_table1"] = {"pi0.5": T6["pi0.5"] == T1["pi0.5"][0][2:], "Ours": T6["Ours"] == T1["Ours"][0][2:]}
R["V11_native_history_vs_pi05_fisher_n100"] = {
    c: fisher(T6["NativeHistory_K4"][j], 100, T6["pi0.5"][j], 100)
    for j, c in enumerate(["swap_blocks", "put_back_block", "observe_and_pickup"])}
R["V11_native_history_vs_ours_fisher_n100"] = {
    c: fisher(T6["NativeHistory_K4"][j], 100, T6["Ours"][j], 100)
    for j, c in enumerate(["swap_blocks", "put_back_block", "observe_and_pickup"])}
R["V11_token_arithmetic"] = {"K4": 5 * 3 * 256, "K8": 9 * 3 * 256}
T7 = {"3": (100, 100), "5": (88, 90), "7": (68, 74)}   # (Once, Retry), 50회
R["V12_table7_integrality_n50"] = {k: {"once_k": a // 2, "retry_k": b // 2, "once_le_retry": a <= b,
                                         "even": a % 2 == 0 and b % 2 == 0,
                                         "once_wilson95": wilson(a // 2, 50), "retry_wilson95": wilson(b // 2, 50)}
                                   for k, (a, b) in T7.items()}
R["V12_table7_3items_equals_table2_grocery_S3"] = (T7["3"][0] == T2["Ours"][2][-1])
fig4b = {"Ours": [60, 71, 79], "Mem-0": [17, 31, 54]}   # 데모 10, 25, 50, 그림에 인쇄된 막대 값
R["V13_fig4b"] = {
    "ratio_10demo": round(fig4b["Ours"][0] / fig4b["Mem-0"][0], 2),
    "ours_50demo_equals_table5_clean_mean": round(mean(T5["Clean"])),
    "mem0_50demo_equals_table1_rmbench3_mean": round(mean(T1["Mem-0"][0][2:])),
    "ours10_vs_mem0_50_fisher_assumed_n300": fisher(180, 300, 162, 300),
    "ours25_vs_mem0_25_fisher_assumed_n300": fisher(213, 300, 93, 300),
    "ours10_vs_mem0_10_fisher_assumed_n300": fisher(180, 300, 51, 300),
    "ours50_vs_mem0_50_fisher_assumed_n300": fisher(237, 300, 162, 300),
    "note": "막대 값은 그림 4(b)에 인쇄된 정수 값. 막대당 시행 수는 논문에 없어 3과제 x 100회 = 300을 가정",
}
T8 = {"Rearrange Block": (89, 98), "Swap T": (14, 69), "Press Button": (0, 5), "Press Button (easy)": (None, 32)}
R["V14_table8_fisher_n100"] = {k: (fisher(b, 100, a, 100) if a is not None else None) for k, (a, b) in T8.items()}

# ---------------- 토큰과 시간 산술 ----------------
ctrl_hz, upd_every = 30, 10
upd_hz = ctrl_hz / upd_every
R["V15_fig1_ratios"] = {"160_over_18": round(160 / 18, 2), "160_over_4": round(160 / 4, 2)}
R["V15_horizon_seconds"] = {
    "update_Hz": upd_hz,
    "160_frames_s": 160 / upd_hz, "200_frames_s": 200 / upd_hz, "5000_frames_min": 5000 / upd_hz / 60,
    "short_2s_frames": 2 * upd_hz, "short_4s_frames": 4 * upd_hz, "short_6s_frames": 6 * upd_hz,
    "sparse_0.5Hz_interval_steps": ctrl_hz / 0.5,
}
R["V15_memory_tokens_V3"] = {"160_frames_tokens": 160 * 3 + 1, "LIBERO_128_frames_V3": round(128 / 3, 1),
                              "LIBERO_512_frames_V3": round(512 / 3, 1), "LIBERO_128_frames_V2": 64, "LIBERO_512_frames_V2": 256}
R["V15_native_history_vs_memory_tokens"] = {"K4_tokens": 3840, "memory_160_frames_tokens": 480, "ratio": round(3840 / 480, 1)}
R["V15_stage1_samples"] = {"steps": 50000, "batch_16_samples": 50000 * 16, "batch_64_samples": 50000 * 64,
                           "trajectories": 4000, "samples_per_traj_batch16": 50000 * 16 / 4000,
                           "stage2_samples_batch32": 20000 * 32, "stage2_epochs_over_50_demos_samples_unknown": None}

# ---------------- 코드에서 읽은 단계 수와 메모리 상한(04 문서의 근거 산술) ----------------
step_limits = {"observe_and_pickup": 250, "put_back_block": 500, "click_button": 600, "swap_T": 600,
               "rearrange_blocks": 700, "swap_blocks": 1000, "cover_blocks": 1500}
caps = {"observe_and_pickup": 256, "put_back_block": 256, "click_button": 256, "swap_T": 256,
        "rearrange_blocks": 256, "swap_blocks": 256, "cover_blocks": 512}
interval = 12   # deploy_policy.yml memory_interval (NativeMEM-RMBench)
cov = {}
for t, lim in step_limits.items():
    frames = lim // interval + 1   # 현재 프레임에서 12 간격으로 0까지 + 0 (항상 0 포함)
    cov[t] = {"step_limit": lim, "max_memory_frames": frames, "tokens_V3": frames * 3, "cap": caps[t],
              "within_cap": frames * 3 <= caps[t]}
R["V16_sim_memory_length_vs_cap_interval12"] = cov
R["V16_real_cap"] = {"arx_scan_code_cap_512_frames_V3": 512 // 3, "seconds_at_interval10_30Hz": (512 // 3) * 10 / 30}

OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False))
for k, v in R.items():
    print(k, json.dumps(v, ensure_ascii=False)[:600])
