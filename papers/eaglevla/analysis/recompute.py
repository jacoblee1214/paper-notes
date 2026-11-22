"""EagleVLA (arXiv:2607.12659 v7) 표 재계산과 일관성 검사 (numpy, scipy). 실행: python3 analysis/recompute.py
결과: analysis/recompute_results.json

- 표 값은 paper/paper.txt(pdftotext -layout)에서 옮겼고, 공식 README의 Table 3 복사본과 대조한다(V1).
- 원자료(에피소드 단위 결과)가 없으므로 검정은 '시행 수 가정'을 명시한 근사다.
  공식 README가 "50 trials/task"라고 적으므로 LIBERO 한 군집(10과제)은 n = 500으로 둔다(K1).
- 버전 비교용 값은 arXiv v1–v7 PDF 텍스트에서 옮겼다(H1).
"""
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "recompute_results.json"
README = ROOT / "repo" / "EagleVLA" / "README.md"
N_TRIALS = 500  # K1: 50 trials/task x 10 tasks (공식 README "50 trials/task")
res = {"assumption_K1_trials_per_suite": N_TRIALS}

# ------------------------------------------------------------------ Table 3 (논문 §5.1)
SUITES = ["SPATIAL", "OBJECT", "GOAL", "LIBERO10"]
METHODS = ["VLASH", "RTC", "Ours", "Sched"]
SYNC = {"SPATIAL": 97.3, "OBJECT": 99.6, "GOAL": 96.7, "LIBERO10": 93.5}
T3 = {
    "SPATIAL": {"VLASH": [98.8, 97.5, 94.4, 92.5, 84.3, 74.4, 51.3, 46.7, 30.1],
                "RTC": [97.1, 95.4, 94.5, 92.5, 91.2, 91.7, 91.7, 90.9, 88.8],
                "Ours": [97.7, 97.1, 97.1, 96.7, 95.9, 97.0, 97.0, 97.2, 97.0],
                "Sched": [98.6, 97.6, 97.2, 96.9, 96.7, 97.5, 97.3, 97.5, 97.3]},
    "OBJECT": {"VLASH": [99.2, 99.2, 98.8, 96.9, 94.3, 88.5, 81.4, 65.1, 51.7],
               "RTC": [98.5, 98.5, 96.4, 97.7, 96.9, 97.4, 97.1, 93.8, 93.1],
               "Ours": [98.7, 98.5, 98.2, 97.8, 96.9, 98.3, 98.0, 98.3, 97.5],
               "Sched": [98.8, 99.3, 98.4, 98.9, 98.6, 98.4, 98.4, 98.8, 97.6]},
    "GOAL": {"VLASH": [96.7, 97.0, 93.3, 93.3, 89.9, 81.3, 76.1, 70.8, 59.7],
             "RTC": [96.5, 96.0, 94.3, 93.3, 93.9, 94.0, 92.6, 93.4, 92.5],
             "Ours": [97.0, 96.2, 96.7, 95.9, 96.0, 96.7, 97.1, 96.8, 96.3],
             "Sched": [97.5, 97.3, 96.4, 96.9, 96.1, 96.9, 97.2, 96.8, 96.5]},
    "LIBERO10": {"VLASH": [94.4, 94.6, 91.9, 89.6, 80.3, 78.5, 77.0, 65.5, 59.6],
                 "RTC": [92.3, 92.1, 89.6, 85.8, 83.6, 84.6, 85.3, 83.2, 81.0],
                 "Ours": [93.4, 92.7, 91.3, 91.8, 91.5, 92.2, 92.6, 92.3, 92.0],
                 "Sched": [93.1, 92.9, 91.9, 92.3, 92.0, 92.9, 92.6, 92.7, 92.2]},
}
AVG_REP = {"SPATIAL": {"VLASH": 74.4, "RTC": 92.6, "Ours": 97.0, "Sched": 97.4},
           "OBJECT": {"VLASH": 86.1, "RTC": 96.6, "Ours": 98.0, "Sched": 98.6},
           "GOAL": {"VLASH": 84.2, "RTC": 94.1, "Ours": 96.5, "Sched": 96.8},
           "LIBERO10": {"VLASH": 81.3, "RTC": 86.4, "Ours": 92.2, "Sched": 92.5}}

# V1: 논문 표 대 공식 README 표 (같은 저자 저장소)
mismatch = []
rows = []
if README.exists():  # 저장소를 클론하지 않은 사본에서는 V1 검사를 건너뜀
    readme_txt = README.read_text(encoding="utf-8")
    rows = re.findall(r"<tr><td>(\d)</td>(.*?)</tr>", readme_txt)
    readme_vals = {}
    for d, cells in rows:
        vals = [float(x) for x in re.findall(r"<td>([\d.]+)</td>", cells)]
        readme_vals[int(d)] = vals
    for i, d in enumerate(range(1, 10)):
        v = readme_vals[d]
        k = 0
        for s in SUITES:
            for m in METHODS:
                if abs(v[k] - T3[s][m][i]) > 1e-9:
                    mismatch.append((d, s, m, T3[s][m][i], v[k]))
                k += 1
    res["V1_table3_paper_vs_readme_mismatches"] = mismatch
    res["V1_readme_rows_found"] = len(rows)
else:
    res["V1_table3_paper_vs_readme_mismatches"] = "skipped: repo/EagleVLA/README.md not found"
    res["V1_readme_rows_found"] = 0

# 평균 행 재계산(∆=1..9 평균) 대 보고값
avg_check = []
for s in SUITES:
    for m in METHODS:
        mean9 = float(np.mean(T3[s][m]))
        avg_check.append({"suite": s, "method": m, "mean9": round(mean9, 3), "reported": AVG_REP[s][m],
                          "ok": abs(round(mean9, 1) - AVG_REP[s][m]) < 0.05})
res["T3_avg_row_check"] = avg_check

# 본문 주장: 4개 군집과 ∆=1..9 평균의 차이 (percentage points)
def suite_mean(method):
    return np.mean([np.mean(T3[s][method]) for s in SUITES])

claim = {}
for base in ["VLASH", "RTC"]:
    for m in ["Ours", "Sched"]:
        claim[f"{m}_minus_{base}_pp"] = round(suite_mean(m) - suite_mean(base), 2)
claim["Sched_mean4"] = round(suite_mean("Sched"), 2)
claim["VLASH_mean4"] = round(suite_mean("VLASH"), 2)
claim["RTC_mean4"] = round(suite_mean("RTC"), 2)
claim["Ours_mean4"] = round(suite_mean("Ours"), 2)
# ∆=9 (본문: VLASH 대비 45.6, RTC 대비 7.0)
d9 = {}
for m in ["Ours", "Sched"]:
    avg9 = np.mean([T3[s][m][8] for s in SUITES])
    d9[m] = {"mean4": round(avg9, 2),
             "minus_VLASH": round(avg9 - np.mean([T3[s]["VLASH"][8] for s in SUITES]), 2),
             "minus_RTC": round(avg9 - np.mean([T3[s]["RTC"][8] for s in SUITES]), 2)}
claim["delta9"] = d9
res["T3_claims"] = claim
res["T3_text_claims_reported"] = {"outperforms_VLASH_pp": 14.8, "outperforms_RTC_pp": 3.9,
                                  "delta9_VLASH_pp": 45.6, "delta9_RTC_pp": 7.0}

# 셀 단위: Sched 대 VLASH, Sched 대 RTC, Sched 대 Sync; VLASH가 Sched보다 높거나 같은 칸
cells_vlash_ge = []
for s in SUITES:
    for i in range(9):
        if T3[s]["VLASH"][i] >= T3[s]["Sched"][i]:
            cells_vlash_ge.append((s, i + 1, T3[s]["VLASH"][i], T3[s]["Sched"][i]))
res["T3_cells_VLASH_ge_Sched"] = cells_vlash_ge
res["T3_count_cells"] = 36

# Sched 대 Ours (스케줄링의 추가 효과), ∆별 평균 차이
res["T3_Sched_minus_Ours_by_delta"] = [
    round(float(np.mean([T3[s]["Sched"][i] - T3[s]["Ours"][i] for s in SUITES])), 2) for i in range(9)]
res["T3_Sched_minus_Sync_by_suite"] = {s: round(float(np.mean(T3[s]["Sched"]) - SYNC[s]), 2) for s in SUITES}
# 그림용: ∆별 네 군집 평균 (그림 스크립트가 읽음)
res["T3_mean_by_delta"] = {m: [round(float(np.mean([T3[s][m][i] for s in SUITES])), 2) for i in range(9)]
                           for m in METHODS}
res["T3_sync_mean4"] = round(float(np.mean(list(SYNC.values()))), 2)

# K1 검사: n = 500이면 비율은 0.2 단위의 배수여야 한다. 아닌 값의 개수
def on_grid(v, n):
    k = round(v / 100 * n)
    return abs(100 * k / n - v) < 0.051

off = []
for s in SUITES:
    for m in METHODS:
        for i, v in enumerate(T3[s][m]):
            if not on_grid(v, N_TRIALS):
                off.append((s, m, i + 1, v))
    if not on_grid(SYNC[s], N_TRIALS):
        off.append((s, "Sync", 0, SYNC[s]))
res["K1_values_not_on_n500_grid"] = {"count": len(off), "total": 4 * 4 * 9 + 4, "examples": off[:8]}

# 셀별 Wilson 95% 구간 (n=500) 과 Sched 대 VLASH 두 비율 검정 (비쌍, 독립 가정)
def wilson(p, n, z=1.96):
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)

sig = {"Sched_gt_VLASH": 0, "Sched_lt_VLASH": 0, "ns": 0}
sig_cells = []
for s in SUITES:
    for i in range(9):
        a = T3[s]["Sched"][i] / 100
        b = T3[s]["VLASH"][i] / 100
        k1, k2 = round(a * N_TRIALS), round(b * N_TRIALS)
        tab = [[k1, N_TRIALS - k1], [k2, N_TRIALS - k2]]
        p = stats.fisher_exact(tab)[1]
        if p < 0.05:
            key = "Sched_gt_VLASH" if a > b else "Sched_lt_VLASH"
            sig[key] += 1
            sig_cells.append((s, i + 1, key, round(p, 6)))
        else:
            sig["ns"] += 1
res["K1_fisher_Sched_vs_VLASH_cells"] = {"counts": sig, "significant_cells": sig_cells}

# 셀별 Sched 대 Ours (비쌍, n=500)
sig2 = {"Sched_gt_Ours": 0, "Sched_lt_Ours": 0, "ns": 0}
for s in SUITES:
    for i in range(9):
        a, b = T3[s]["Sched"][i] / 100, T3[s]["Ours"][i] / 100
        k1, k2 = round(a * N_TRIALS), round(b * N_TRIALS)
        p = stats.fisher_exact([[k1, N_TRIALS - k1], [k2, N_TRIALS - k2]])[1]
        if p < 0.05:
            sig2["Sched_gt_Ours" if a > b else "Sched_lt_Ours"] += 1
        else:
            sig2["ns"] += 1
res["K1_fisher_Sched_vs_Ours_cells"] = sig2

# 군집 평균 차이의 근사 표준오차: 셀 독립, 각 셀 이항분산 평균 (쌍체 정보 없음, 과소·과대 가능)
def mean_diff_se(m1, m2):
    # Var(mean over 36 cells of (x-y)) approx = sum(var_x+var_y)/36^2 under independence
    vars_ = []
    for s in SUITES:
        for i in range(9):
            p1, p2 = T3[s][m1][i] / 100, T3[s][m2][i] / 100
            vars_.append(p1 * (1 - p1) / N_TRIALS + p2 * (1 - p2) / N_TRIALS)
    return math.sqrt(sum(vars_)) / 36

for base in ["VLASH", "RTC"]:
    diff = claim[f"Sched_minus_{base}_pp"] / 100
    se = mean_diff_se("Sched", base)
    claim[f"Sched_minus_{base}_approx95CI_pp"] = [round(100 * (diff - 1.96 * se), 2), round(100 * (diff + 1.96 * se), 2)]
res["T3_claims"] = claim

# ------------------------------------------------------------------ Table 1 (지연 분해, §3)
T1 = {  # (ViT, LLM, AE, Total)
    ("pi0", "Orin-30W"): (322.4, 856.4, 1090.0, 2268.8), ("pi0", "Orin-50W"): (152.5, 632.3, 518.2, 1403.0),
    ("pi0", "Thor"): (63.7, 185.0, 199.2, 447.9), ("pi0", "A6000"): (23.4, 43.5, 46.9, 116.1),
    ("pi0", "4090"): (13.1, 32.2, 22.4, 69.5),
    ("pi05", "Orin-30W"): (329.0, 858.2, 1102.2, 2289.4), ("pi05", "Orin-50W"): (152.3, 631.0, 536.8, 1420.8),
    ("pi05", "Thor"): (61.7, 187.0, 209.2, 457.9), ("pi05", "A6000"): (23.8, 51.2, 50.2, 128.2),
    ("pi05", "4090"): (13.2, 35.9, 25.2, 76.4),
}
t1 = []
for k, (v, l, a, t) in T1.items():
    gap = round(t - (v + l + a), 2)
    t1.append({"model": k[0], "device": k[1], "sum_components": round(v + l + a, 1), "total": t, "gap_ms": gap})
res["T1_sum_check"] = t1

# ------------------------------------------------------------------ Table 4 (지연과 제어 주기, §5.2)
# (ViT, LLM, AE, Total, RT, Hz) — v7 값
T4_V7 = {
    "Orin": [("Naive PI05", 152.3, 631.0, 536.8, 1420.8, 1420.8, 0.70),
             ("+Schedule opt.", 152.3, 631.0, 536.8, 1420.8, 674.9, 1.48),
             ("+Graph reuse", 71.9, 196.9, 143.9, 412.9, 184.0, 5.43),
             ("+Intermediate Buffer & Unroll", 66.6, 137.0, 67.6, 271.1, 97.1, 10.30)],
    "Thor": [("Naive PI05", 61.7, 187.0, 209.2, 457.9, 457.9, 2.18),
             ("+Schedule opt.", 61.7, 187.0, 209.2, 457.9, 244.7, 4.08),
             ("+Graph reuse", 32.0, 92.3, 115.7, 240.0, 134.6, 7.43),
             ("+Intermediate Buffer & Unroll", 29.0, 54.9, 48.9, 132.8, 61.1, 16.37)],
}
t4 = {}
for dev, lst in T4_V7.items():
    entries = []
    for name, v, l, a, t, rt, hz in lst:
        entries.append({"row": name, "sum_components": round(v + l + a, 1), "total": t,
                        "gap_ms": round(t - (v + l + a), 2),
                        "Hz_from_RT": round(1000 / rt, 2), "Hz_reported": hz,
                        "hz_ok": abs(1000 / rt - hz) < 0.01})
    t4[dev] = entries
res["T4_v7_rows"] = t4

naive = {"Orin": 1420.8, "Thor": 457.9}
final_rt = {"Orin": 97.1, "Thor": 61.1}
sched_rt = {"Orin": 674.9, "Thor": 244.7}
graph_rt = {"Orin": 184.0, "Thor": 134.6}
sp = {}
for dev in ["Orin", "Thor"]:
    sp[dev] = {
        "sched_vs_naive": round(naive[dev] / sched_rt[dev], 6),
        "graph_vs_sched": round(sched_rt[dev] / graph_rt[dev], 6),
        "buffer_unroll_vs_graph": round(graph_rt[dev] / final_rt[dev], 6),
        "overall_RT_ratio": round(naive[dev] / final_rt[dev], 6),
    }
hz_naive = {"Orin": 0.70, "Thor": 2.18}
hz_final = {"Orin": 10.30, "Thor": 16.37}
sp["Orin"]["headline_from_rounded_Hz"] = round(hz_final["Orin"] / hz_naive["Orin"], 6)
sp["Thor"]["headline_from_rounded_Hz"] = round(hz_final["Thor"] / hz_naive["Thor"], 6)
sp["Orin"]["headline_from_rounded_Hz_vs_ms"] = round(1000 / final_rt["Orin"] / (1000 / naive["Orin"]), 3)
sp["Thor"]["headline_from_rounded_Hz_vs_ms"] = round(1000 / final_rt["Thor"] / (1000 / naive["Thor"]), 3)
VLA_CPP_MS = 893.0
sp["Orin"]["vs_vla_cpp_ms"] = round(VLA_CPP_MS / final_rt["Orin"], 6)
sp["Orin"]["vs_vla_cpp_Hz"] = round(10.30 / (1000 / VLA_CPP_MS), 3)
res["T4_speedups_v7"] = sp
res["T4_text_claims_v7"] = {"sched_orin": 2.11, "sched_thor": 1.87, "graph_orin": 3.67, "graph_thor": 1.82,
                            "buf_orin": 1.89, "buf_thor": 2.20, "headline_orin": 14.71,
                            "headline_thor": 7.51, "vs_vla_cpp": 9.20}

# ------------------------------------------------------------------ 버전 이력 (H1)
# v1–v5: 같은 표 값(Orin 최종 행 412.9 ms, 6.06 Hz; Thor 최종 309.5 ms, 7.59 Hz), 헤드라인 8.66×/5.41×
# v6: 그래프 재측정(Orin 374.5 ms, 6.93 Hz; Thor 202.8 ms, 10.25 Hz), 헤드라인 9.85×/6.19×
# v7: Orin 271.1 ms, 10.30 Hz; Thor 132.8 ms, 16.37 Hz; 헤드라인 14.71×/9.20×
VERS = {
    "v1-v5": {"orin_final_total": 412.9, "orin_final_RT": 165.1, "orin_final_Hz": 6.06,
              "thor_final_total": 309.5, "thor_final_RT": 131.8, "thor_final_Hz": 7.59,
              "headline_orin": 8.66, "headline_vla_cpp": 5.41,
              "orin_graph_total": 476.1, "orin_graph_RT": 227.0, "thor_graph_total": 372.1, "thor_graph_RT": 191.4},
    "v6": {"orin_final_total": 374.5, "orin_final_RT": 144.3, "orin_final_Hz": 6.93,
           "thor_final_total": 202.8, "thor_final_RT": 97.6, "thor_final_Hz": 10.25,
           "headline_orin": 9.85, "headline_vla_cpp": 6.19,
           "orin_graph_total": 412.9, "orin_graph_RT": 184.0, "thor_graph_total": 240.0, "thor_graph_RT": 134.6,
           "text_1_33_1_46": {"orin_RT_ratio_184_over_144.3": round(184.0 / 144.3, 3),
                              "thor_RT_ratio_134.6_over_97.6": round(134.6 / 97.6, 3)}},
    "v7": {"orin_final_total": 271.1, "orin_final_RT": 97.1, "orin_final_Hz": 10.30,
           "thor_final_total": 132.8, "thor_final_RT": 61.1, "thor_final_Hz": 16.37,
           "headline_orin": 14.71, "headline_vla_cpp": 9.20,
           "orin_graph_total": 412.9, "orin_graph_RT": 184.0, "thor_graph_total": 240.0, "thor_graph_RT": 134.6},
}
res["H1_version_headlines"] = VERS
res["H1_ratio_v7_vs_v1_orin"] = round(14.71 / 8.66, 3)
res["H1_orin_final_Hz_change_v6_to_v7_pct"] = round(100 * (10.30 / 6.93 - 1), 1)

# 같은 방법(Schedule 0.70 Hz 기준)의 최종 행 주파수 변화: 기준값이 변하지 않았는데 최종 행만 바뀜
# 역산: v7 최종 행의 ViT+LLM+AE 합 대 v6 합 차이
res["H1_orin_final_component_change_v6_to_v7_ms"] = {
    "ViT": round(66.6 - 71.9, 1), "LLM": round(137.0 - 194.8, 1), "AE": round(67.6 - 107.8, 1), "total": round(271.1 - 374.5, 1)}

# ------------------------------------------------------------------ 가정 기반 역산 (표 4 Sched 행)
# 가정: RT = AE + f*(ViT+LLM) (VLM을 비율 f로만 호출). 논문에 정의가 없으므로 값의 크기만 본다.
ftrans = {}
for dev, (v, l, a) in {"Orin": (152.3, 631.0, 536.8), "Thor": (61.7, 187.0, 209.2)}.items():
    rt = sched_rt[dev]
    ftrans[dev] = round((rt - a) / (v + l), 3)
res["T4_sched_implied_VLM_fraction_ASSUMED_MODEL"] = ftrans

# ------------------------------------------------------------------ Table 2 (계수 2)
T2 = {"RTX4090": (309, 1024, 618), "A6000": (309, 768, 804), "Orin32": (200, 204.8, 1953),
      "Orin64": (275, 204.8, 2685), "Thor": (510, 273, 3736)}
t2 = {}
for k, (tops, bw, reported) in T2.items():
    raw = tops / bw * 1000
    t2[k] = {"TOPS_over_BW_x1000": round(raw, 1), "reported": reported, "ratio_reported_to_raw": round(reported / raw, 3)}
res["T2_balance_factor"] = t2

# 거친 roofline 점검(가정): 단일 행렬곱 M x K x N, K=N=2048, FP16(2바이트), M=800 토큰(VLM), M=10(행동 전문가)
# 시간 = max(2*M*K*N / 피크 FLOPS, (M*K + K*N) * 2바이트 / 대역폭). 표 2의 TOPS를 FLOPS로 본다.
def roof(M, K, N, tops, bw_gbs):
    tc = 2 * M * K * N / (tops * 1e12)
    tm = (M * K + K * N) * 2 / (bw_gbs * 1e9)
    return ("compute" if tc > tm else "bandwidth"), round(tc * 1e3, 4), round(tm * 1e3, 4)
roof_rows = {}
for dev, (tops, bw) in {"RTX4090": (309, 1024), "Orin32": (200, 204.8), "Thor": (510, 273)}.items():
    roof_rows[dev] = {"VLM_M800": roof(800, 2048, 2048, tops, bw), "AE_M10": roof(10, 1024, 1024, tops, bw)}
res["T2_roofline_check_ASSUMED"] = roof_rows

# ------------------------------------------------------------------ Figure 7 (실로봇, §5.3)
# 과제당 10회. 4090 기준, Orin naive, EagleVLA
FIG7 = {"RTX4090": [10, 7, 10], "Orin_naive": [6, 0, 5], "EagleVLA": [10, 8, 9]}
n7 = 10
fig7 = {}
for k, v in FIG7.items():
    s = sum(v)
    fig7[k] = {"success": s, "of": 3 * n7, "rate": round(s / (3 * n7), 3),
               "wilson95": [round(x, 3) for x in wilson(s / (3 * n7), 3 * n7)]}
pairs = {"EagleVLA_vs_Orin_naive": (27, 30, 11, 30), "EagleVLA_vs_4090": (27, 30, 27, 30),
         "Orin_naive_vs_4090": (11, 30, 27, 30)}
fisher7 = {}
for name, (a, na, b, nb) in pairs.items():
    fisher7[name] = {"fisher_p_two_sided": round(float(stats.fisher_exact([[a, na - a], [b, nb - b]])[1]), 6)}
res["F7_real_world"] = {"counts": fig7, "fisher": fisher7,
                        "note": "과제 3개를 독립 시행 30회로 합산. 과제 군집과 시행 순서는 무시."}

# ------------------------------------------------------------------ 저장
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print("wrote", OUT)
print("V1 mismatches:", len(mismatch), "| readme rows:", len(rows), "| readme found:", README.exists())
print("T3 claims:", json.dumps(claim, ensure_ascii=False))
print("K1 off-grid:", res["K1_values_not_on_n500_grid"]["count"], "of", res["K1_values_not_on_n500_grid"]["total"])
print("K1 fisher Sched vs VLASH:", sig, "| Sched vs Ours:", sig2)
print("cells VLASH>=Sched:", len(cells_vlash_ge), cells_vlash_ge[:6])
print("T4 speedups:", json.dumps(sp))
print("T4 ft:", ftrans)
print("T2 roof:", roof_rows)
print("F7:", json.dumps(fig7), json.dumps(fisher7))
