"""논문 표 수치를 다시 계산하고 검정한다 (수 초, numpy·scipy).

입력: 논문 v4 (arXiv:2603.18532) 표 1, 표 2, 표 B.1, 표 C.1의 값을 옮겨 적은 것.
출력: analysis/recompute_results.json (검사 항목 V1–Vn의 결과)

실행: python3 analysis/recompute.py
표준편차는 표본 SD(ddof=1)로 가정한다. 논문이 SD의 정의를 밝히지 않았다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
R = {}

# ---------- 표 2 (실로봇, 12 장면 x 10회, 정책당 120회) ----------
scenes = list(range(12))
IL = dict(
    PSR=[0.8, 0.4, 0.3, 0.2, 0.4, 0.6, 0.4, 0.7, 0.3, 0.6, 0.2, 0.5],
    SR=[0.6, 0.2, 0.1, 0.1, 0.3, 0.3, 0.2, 0.2, 0.1, 0.3, 0.0, 0.2],
    DFR=[0.4, 0.8, 0.9, 0.7, 0.7, 0.7, 0.8, 0.2, 0.9, 0.6, 0.8, 0.5],
    SFR=[0.0, 0.0, 0.0, 0.5, 0.1, 0.0, 0.0, 0.6, 0.0, 0.1, 0.3, 0.6],
    TF=[10.7, 13.3, 11.5, 6.1, 11.0, 10.5, 7.8, 14.6, 29.6, 9.2, None, 11.6],
)
RL = dict(
    PSR=[0.9, 1.0, 0.9, 0.8, 1.0, 1.0, 1.0, 0.8, 0.9, 0.7, 0.6, 1.0],
    SR=[0.9, 0.7, 0.9, 0.7, 1.0, 1.0, 0.7, 0.8, 0.7, 0.6, 0.5, 0.5],
    # 표의 "−"(장면 4, 5: SR 1.0이라 실패가 없음)은 0으로 둔다
    DFR=[0.1, 0.3, 0.1, 0.1, 0.0, 0.0, 0.3, 0.0, 0.2, 0.4, 0.2, 0.5],
    SFR=[0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2, 0.1, 0.0, 0.3, 0.0],
    TF=[7.4, 10.4, 8.4, 12.2, 10.2, 9.5, 11.1, 12.2, 12.2, 8.8, 10.4, 11.0],
)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 4), round(c + h, 4)


# V1: 표 2 평균 재계산 (논문 값과 비교)
rep = {"IL": {"PSR": 0.45, "SR": 0.217, "DFR": 0.667, "SFR": 0.183, "TF": 11.5},
       "RL": {"PSR": 0.883, "SR": 0.75, "DFR": 0.183, "SFR": 0.067, "TF": 10.2}}
V1 = {}
for name, pol in (("IL", IL), ("RL", RL)):
    V1[name] = {}
    for k in ("PSR", "SR", "DFR", "SFR"):
        m = float(np.mean(pol[k]))
        V1[name][k] = {"recomputed": round(m, 4), "reported": rep[name][k], "diff": round(m - rep[name][k], 4)}
    tf = [t for t in pol["TF"] if t is not None]
    V1[name]["TF"] = {"recomputed_mean_of_scene_means": round(float(np.mean(tf)), 2),
                      "reported": rep[name]["TF"], "n_scenes": len(tf)}
R["V1_table2_means"] = V1

# V2: 실제 시도 수 (정책당 120회)와 Wilson 95% 구간
N_TRIALS = 120
V2 = {}
for name, pol in (("IL", IL), ("RL", RL)):
    k = round(float(np.mean(pol["SR"])) * N_TRIALS)
    V2[name] = {"successes": k, "trials": N_TRIALS, "SR": round(k / N_TRIALS, 4), "wilson95": wilson(k, N_TRIALS)}
    k2 = round(float(np.mean(pol["PSR"])) * N_TRIALS)
    V2[name]["PSR_successes"] = k2
R["V2_trial_counts_wilson"] = V2

# V3: 두 정책 성공률 차이 (Fisher 정확 검정, 비대응 2x2)
tab = [[V2["RL"]["successes"], N_TRIALS - V2["RL"]["successes"]],
       [V2["IL"]["successes"], N_TRIALS - V2["IL"]["successes"]]]
odds, p_f = stats.fisher_exact(tab)
diff = V2["RL"]["SR"] - V2["IL"]["SR"]
R["V3_fisher_real_SR"] = {"table": tab, "diff_pp": round(100 * diff, 1), "p_fisher": float(p_f)}

# V4: 장면 쌍 비교 (같은 장면, 10회씩). 부호 검정과 Wilcoxon
il = np.array(IL["SR"]); rl = np.array(RL["SR"])
n_pos = int(np.sum(rl > il)); n_tie = int(np.sum(rl == il)); n_neg = int(np.sum(rl < il))
sign_p = stats.binomtest(n_pos, n_pos + n_neg, 0.5, alternative="two-sided").pvalue
w = stats.wilcoxon(rl, il, zero_method="wilcox", alternative="two-sided")
R["V4_paired_scenes_SR"] = {"RL_gt_IL": n_pos, "tie": n_tie, "RL_lt_IL": n_neg,
                           "sign_test_p": float(sign_p), "wilcoxon_p": float(w.pvalue)}

# V5: 실패 분해 일관성. 실패 = 1 - SR. DFR + SFR >= 1 - SR 이어야 하고, 같으면 겹치지 않음
V5 = {}
for name, pol in (("IL", IL), ("RL", RL)):
    fail = 1 - np.array(pol["SR"])
    both = np.array(pol["DFR"]) + np.array(pol["SFR"])
    V5[name] = {"violations_DFR_SFR_lt_fail": int(np.sum(both < fail - 1e-9)),
                "scenes_with_overlap": int(np.sum(both > fail + 1e-9))}
    V5[name]["mean_overlap_pp"] = round(100 * float(np.mean(np.clip(both - fail, 0, None))), 1)
V5["check"] = "SR <= PSR 위반 개수"
V5["PSR_lt_SR_violations"] = {"IL": int(np.sum(np.array(IL["PSR"]) < np.array(IL["SR"]) - 1e-9)),
                              "RL": int(np.sum(np.array(RL["PSR"]) < np.array(RL["SR"]) - 1e-9))}
R["V5_failure_consistency"] = V5

# V6: 시뮬 SR 0.798 (초록 헤드라인)과 표 1 N=100 평균, 표 B.1 K=1 단일 실행
R["V6_headline_79_8"] = {"abstract_claim": 79.8, "table1_N100_W_mean": 78.9, "table1_N100_W_sd": 0.9,
                         "tableB1_K1_single_run": 79.75,
                         "note": "초록 79.8은 표 1 평균(78.9 ± 0.9, 3 시드)과 다르고 표 B.1의 단일 실행 79.75와 일치"}

# ---------- 표 1 (시뮬레이션) ----------
# (평균, SD 또는 None). N=0은 사전학습 정책, "SE"는 SimplerEnv 3장면 학습 정책
ID = {1: (94.3, 2.6), 3: (88.7, 4.7), 10: (87.0, 3.3), 25: (84.3, 2.4), 50: (80.5, 0.4)}
OOD = {0: (9.6, None), "SE": (36.5, None), 1: (53.2, 5.5), 3: (61.3, 1.8), 10: (72.4, 0.4),
       25: (77.6, 0.8), 50: (77.9, 0.9)}
OOD_TF = {1: (9.0, 0.5), 3: (8.7, 0.5), 10: (8.2, 0.2), 25: (8.1, 0.2), 50: (8.0, 0.2)}
OOD_dG = {"SE": 60.2, 1: 41.1, 3: 27.4, 10: 14.6, 25: 6.7, 50: 2.6}
W = {0: (9.7, None), "SE": (36.0, None), 1: (51.6, 5.3), 3: (60.0, 4.6), 10: (72.1, 0.9),
     25: (78.3, 0.5), 50: (79.2, 0.3), 100: (78.9, 0.9)}
BENCH = {  # SR (평균, SD), ∆G 보고값 (ID 또는 N=100은 W 평균을 기준으로 함)
    "SimplerEnv(3)": {"SR": {0: 23.7, "SE": 96.7, 1: (36.1, 10.4), 3: (47.4, 2.3), 10: (54.3, 9.2),
                             25: (70.1, 5.4), 50: (68.4, 5.5), 100: (74.3, 4.1)},
                      "dG": {1: 58.2, 3: 41.3, 10: 32.7, 25: 14.2, 50: 12.1, 100: 4.6}},
    "GoogleScanned(18)": {"SR": {0: 12.1, "SE": 35.2, 1: (42.3, 4.8), 3: (48.8, 7.3), 10: (59.3, 2.6),
                                 25: (62.7, 0.6), 50: (62.5, 0.4), 100: (60.7, 0.9)},
                          "dG": {"SE": 61.5, 1: 52.0, 3: 39.9, 10: 27.7, 25: 21.6, 50: 18.0, 100: 18.2}},
    "Hunyuan3D(25)": {"SR": {0: 10.2, "SE": 38.4, 1: (42.7, 3.8), 3: (50.0, 5.0), 10: (58.2, 2.9),
                             25: (61.8, 0.3), 50: (63.3, 0.4), 100: (62.5, 1.4)},
                      "dG": {"SE": 58.3, 1: 51.6, 3: 38.7, 10: 28.8, 25: 22.5, 50: 17.2, 100: 16.4}},
}


def val(x):
    return x[0] if isinstance(x, tuple) else x


# V7: ∆G 재계산. OOD 행: ID(N) - OOD(N). 벤치마크 행: ID(N) - SR_bench(N) (N=100은 W 평균 사용)
mism = []
for n in (1, 3, 10, 25, 50):
    d = round(ID[n][0] - OOD[n][0], 1)
    if abs(d - OOD_dG[n]) > 0.05:
        mism.append(("OOD", n, d, OOD_dG[n]))
d_se = round(96.7 - OOD["SE"][0], 1)  # SE 정책의 OOD 행은 EG OOD 36.5 기준
if abs(d_se - OOD_dG["SE"]) > 0.05:
    mism.append(("OOD", "SE", d_se, OOD_dG["SE"]))
bench_checks = []
for bname, b in BENCH.items():
    for n, dg in b["dG"].items():
        # "SE" 행의 기준은 SimplerEnv 3장면 정책의 자기 장면 성공률 96.7 (표 1 SimplerEnv 블록)
        ref = 96.7 if n == "SE" else (W[100][0] if n == 100 else ID[n][0])
        sr = b["SR"][n][0] if isinstance(b["SR"][n], tuple) else b["SR"][n]
        d = round(ref - sr, 1)
        bench_checks.append({"bench": bname, "N": n, "recomputed": d, "reported": dg, "ok": abs(d - dg) < 0.05})
        if abs(d - dg) > 0.05:
            mism.append((bname, n, d, dg))
R["V7_dG_recompute"] = {"mismatches": mism, "bench_checks_n": len(bench_checks),
                        "bench_checks_ok": sum(c["ok"] for c in bench_checks)}

# V8: 혼합 평가 W와 ID, OOD의 관계. N=50은 W가 ID와 OOD의 평균이어야 함 (장면 50개씩)
R["V8_W_is_mean_of_ID_OOD_at_N50"] = {
    "recomputed": round((ID[50][0] + OOD[50][0]) / 2, 2), "reported_W": W[50][0],
    "note": "N=50에서만 W=(ID+OOD)/2가 성립 (ID와 OOD가 서로소인 50장면씩). 다른 N은 ID 장면 수가 달라 불가"}

# V9: 같은 N에서 OOD 차이의 Welch 검정 (3 실행, 표본 SD 가정)
def welch(a, b, na=3, nb=3):
    t, p = stats.ttest_ind_from_stats(a[0], a[1], na, b[0], b[1], nb, equal_var=False)
    return {"diff": round(a[0] - b[0], 2), "t": round(float(t), 2), "p": round(float(p), 4)}


R["V9_welch_OOD_and_W"] = {
    "OOD_N1_vs_N3": welch(OOD[1], OOD[3]),
    "OOD_N3_vs_N10": welch(OOD[3], OOD[10]),
    "OOD_N10_vs_N25": welch(OOD[10], OOD[25]),
    "OOD_N25_vs_N50": welch(OOD[25], OOD[50]),
    "OOD_N1_vs_N50": welch(OOD[50], OOD[1]),
    "W_N25_vs_N50": welch(W[25], W[50]),
    "W_N50_vs_N100": welch(W[50], W[100]),
    "note": "포화 구간(N=25 대 50)은 차이가 작고 p가 커야 함. 3 실행이라 검정력이 낮음",
}

# V10: 24.7 pp (본문) = OOD(50) - OOD(1); 초록의 시뮬 향상 70.1 pp = 79.8 - 9.7
R["V10_claims_arith"] = {
    "OOD_50_minus_1": round(OOD[50][0] - OOD[1][0], 1), "text_claim": 24.7,
    "sim_gain_pp_79_8_minus_9_7": round(79.8 - 9.7, 1), "text_claim_sim_gain": 70.1,
    "real_gain_pp": round(75.0 - 21.7, 1), "text_claim_real_gain": 53.3,
}

# V11: 속도 향상 비율 (시간 단위 초)
R["V11_speedups"] = {
    "sim_W_TF_pre_over_N100": round(10.0 / 8.1, 3),
    "sim_W_TF_pre_over_N50": round(10.0 / 7.9, 3),
    "sim_W_TF_pre_over_K1_single_run": round(10.0 / 7.94, 3),
    "text_claim": 1.25,
    "real_TF_IL_over_RL": round(11.5 / 10.2, 3), "text_claim_real": 1.13,
    "real_TF_recomputed_RL_mean_of_scene_means": round(float(np.mean(RL["TF"])), 2),
    "real_TF_recomputed_IL_mean_of_scene_means": round(float(np.mean([t for t in IL["TF"] if t])), 2),
}

# V12: 표 B.1 (K에 따른 추론 지연과 학습 시간)
lat_reg = {"K10": 0.267, "K4": 0.153, "K2": 0.120, "K1": 0.098}
lat_cmp = {"K10": 0.172, "K4": 0.107, "K2": 0.086, "K1": 0.073}
bwd = {"K4": 108.80, "K2": 87.55, "K1": 74.74}
R["V12_tableB1_speedups"] = {
    "reg_K10_over_K1": round(lat_reg["K10"] / lat_reg["K1"], 2), "text_claim_reg": 2.72,
    "compile_K10_over_K1": round(lat_cmp["K10"] / lat_cmp["K1"], 2), "text_claim_compile": 2.36,
    "backward_K2_over_K1": round(bwd["K2"] / bwd["K1"], 2), "text_claim_bwd_K2": 1.17,
    "backward_K4_over_K1": round(bwd["K4"] / bwd["K1"], 2), "text_claim_bwd_K4": 1.45,
    "SR_K1_minus_K2_pp": round(79.75 - 77.23, 2), "SR_K1_minus_K4_pp": round(79.75 - 77.29, 2),
    "note": "SR 차이는 실행 1회. 불확실성 없음",
}

# V13: 표 C.1 (생성 시간과 QA 통과율)
per_scene = 46.8
parts = 25.0 + 3.9 * 5.16
R["V13_tableC1"] = {
    "scenes_x_objects": round(516 / 100, 2), "reported_avg_objects": 5.16,
    "sum_of_parts_min": round(parts, 2), "reported_total_min": per_scene,
    "unexplained_gap_min": round(per_scene - parts, 2),
    "QA_product_of_stage_pass_rates": round(0.833 * 0.752 * 0.919, 3),
    "total_sequential_hours_100_scenes": round(100 * per_scene / 60, 1),
}

# V14: 학습 설정의 시간 환산 (표 A.1: 제어 5 Hz, 청크 C=4). 에피소드 길이 25를 청크 단위로 가정할 때
chunk_s = 4 / 5
R["V14_episode_time"] = {"chunk_seconds": chunk_s, "max_episode_s_if_25_chunks": 25 * chunk_s,
                         "TF_RL_s": 10.2, "chunks_at_TF": round(10.2 / chunk_s, 1),
                         "note": "에피소드 길이 단위(청크인지 환경 스텝인지) 논문에 명시 없음"}

# V15: 로그-σ 범위 (표 A.1)
R["V15_sigma_range"] = {"sigma_min": round(math.exp(-2.5), 4), "sigma_max": round(math.exp(-2.0), 4)}

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: R[k] for k in ("V2_trial_counts_wilson", "V3_fisher_real_SR", "V4_paired_scenes_SR",
                                    "V7_dG_recompute", "V8_W_is_mean_of_ID_OOD_at_N50")}, ensure_ascii=False, indent=1))
print("wrote", OUT)
