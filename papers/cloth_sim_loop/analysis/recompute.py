"""SILR-Cloth (arXiv:2606.24552 v2) 논문 표 수치를 다시 계산하고 통계 검정을 한다.

입력: 논문 표와 본문의 값을 아래에 그대로 옮겨 적었다(출처는 주석). 외부 데이터는 쓰지 않는다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 표 1의 "±"는 20개 시드의 표준편차(SD)로 읽는다(부록 B.1이 시드 20개라고만 쓴다).
- 표 3, 4, D.1의 비율 분모는 본문의 "ten trials"(10회)다.
- 표 2의 ±는 검증 분할 프레임 간 SD로 읽고, 분할 크기는 논문에 없어 t 검정은 하지 않는다.
검사 번호는 V1, V2, ... 이다(분석 문서 03의 번호와 같다).
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def welch(m1, s1, n1, m2, s2, n2):
    r = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
    return float(r.pvalue)


# ---------------------------------------------------------------- V1: 표 1 정수 가능성과 비율
# (simulator, K): (step_ms, mppi_s, kp_mean, kp_sd, mse_mean, mse_sd, sr_pct)   [표 1, 4.1절]
T1 = {
    ("Newton", 64): (41.0, 0.85, 8.40, 8.72, 56.8, 73.1, 40),
    ("Newton", 128): (55.1, 1.14, 10.28, 11.08, 68.9, 93.7, 40),
    ("Newton", 256): (104.3, 2.15, 8.85, 10.71, 58.1, 89.6, 60),
    ("Newton", 512): (232.8, 4.80, 13.02, 13.06, 92.3, 105.3, 35),
    ("Isaac", 64): (11.7, 0.26, 9.78, 10.74, 60.5, 104.4, 55),
    ("Isaac", 128): (22.2, 0.50, 10.88, 11.13, 67.4, 108.0, 45),
    ("Isaac", 256): (41.1, 0.92, 7.09, 5.60, 32.4, 47.6, 55),
    ("Isaac", 512): (80.7, 1.79, 5.84, 4.73, 24.6, 32.8, 60),
    ("FLASH", 64): (19.4, 0.40, 1.61, 1.18, 3.7, 4.3, 100),
    ("FLASH", 128): (36.0, 0.75, 1.54, 0.88, 3.1, 1.9, 100),
    ("FLASH", 256): (68.9, 1.42, 1.95, 1.82, 5.5, 11.0, 95),
    ("FLASH", 512): (137.3, 2.84, 2.08, 1.93, 3.9, 3.2, 95),
}
N_SEED = 20


def v1_table1():
    rows = []
    for (sim, k), (st, mp, kpm, kps, msm, mss, sr) in T1.items():
        succ = sr * N_SEED / 100
        rows.append({
            "sim": sim, "K": k, "sr_pct": sr, "successes_of_20": succ,
            "integer_ok": abs(succ - round(succ)) < 1e-9,
            "wilson95_pct": [round(100 * x, 1) for x in wilson(round(succ), N_SEED)],
            "mppi_over_step": round(mp * 1000 / st, 2),   # MPPI 한 번 = 시뮬 몇 스텝에 해당하는가
        })
    RES["V1_table1_rows"] = rows
    # FLASH 대 Isaac: 같은 K에서 SR Fisher와 KP err Welch
    cmp_ = []
    for k in (64, 128, 256, 512):
        f = T1[("FLASH", k)]
        i = T1[("Isaac", k)]
        n_ = T1[("Newton", k)]
        cmp_.append({
            "K": k,
            "SR_FLASH_vs_Isaac_fisher_p": round(fisher(round(f[6] * .2), 20, round(i[6] * .2), 20), 5),
            "SR_FLASH_vs_Newton_fisher_p": round(fisher(round(f[6] * .2), 20, round(n_[6] * .2), 20), 5),
            "KPerr_FLASH_vs_Isaac_welch_p": welch(f[2], f[3], 20, i[2], i[3], 20),
            "step_ratio_FLASH_over_Isaac": round(f[0] / i[0], 2),
            "step_ratio_Newton_over_Isaac": round(n_[0] / i[0], 2),
            "MSE_ratio_Isaac_over_FLASH": round(i[4] / f[4], 1),
            "MSE_ratio_Newton_over_FLASH": round(n_[4] / f[4], 1),
            "RMS_ratio_Isaac_over_FLASH": round(math.sqrt(i[4] / f[4]), 2),
            "KP_ratio_Isaac_over_FLASH": round(i[2] / f[2], 1),
        })
    RES["V1_flash_vs_isaac"] = cmp_
    # 본문 "Isaac Sim is marginally faster per step (1.5-1.9x)"와 부록 "constant ~1.7x", "Newton 2.5-3.5x slower than Isaac"
    r = [c["step_ratio_FLASH_over_Isaac"] for c in cmp_]
    RES["V1_step_ratio_range_FLASH_over_Isaac"] = [min(r), max(r)]
    r2 = [c["step_ratio_Newton_over_Isaac"] for c in cmp_]
    RES["V1_step_ratio_range_Newton_over_Isaac"] = [min(r2), max(r2)]
    # 평균 SR (FLASH 대 나머지) 풀링
    RES["V1_pooled_SR"] = {
        s: sum(round(T1[(s, k)][6] * .2) for k in (64, 128, 256, 512)) for s in ("Newton", "Isaac", "FLASH")
    }


# ---------------------------------------------------------------- V2: 부록 B.1 서술 대 표 1 (마르코프 상한)
def v2_median_bound():
    """음이 아닌 변수의 중앙값은 평균의 2배를 넘을 수 없다(마르코프). 표 1 평균 MSE로 중앙값 상한을 낸다."""
    out = {}
    for s in ("Newton", "Isaac", "FLASH"):
        means = [T1[(s, k)][4] for k in (64, 128, 256, 512)]
        out[s] = {"mean_MSE_range_cm2": [min(means), max(means)], "median_upper_bound_cm2": round(2 * max(means), 1)}
    out["text_claim_median_MSE_Newton_Isaac_cm2"] = [30, 300]   # 부록 B.1 (iii)
    out["text_claim_EMD_Newton_Isaac_cm"] = [5, 10]
    out["text_claim_FLASH_median_MSE_cm2"] = [3, 4]
    out["text_claim_FLASH_EMD_cm"] = [1.5, 2.5]
    out["note"] = ("프로젝트 페이지의 표 1 확장판은 EMD(cm) 평균을 Newton 1.52-1.97, Isaac 1.16-1.56, FLASH 0.66-1.02로 적는다. "
                   "그림 B.1 판독 중앙값은 EMD 약 0.5-1.3 cm, MSE 중앙값은 Newton 약 25-35, Isaac 약 9-21, FLASH 약 2-3 cm2(그림 판독 값).")
    RES["V2_appendix_text_vs_table1"] = out
    RES["V2_projectpage_EMD_cm_mean"] = {
        "Newton": [1.77, 1.52, 1.56, 1.97], "Isaac": [1.47, 1.56, 1.16, 1.18], "FLASH": [0.92, 0.66, 1.02, 1.02],
    }
    ratios = []
    for k, (n_, i_, f_) in enumerate(zip([1.77, 1.52, 1.56, 1.97], [1.47, 1.56, 1.16, 1.18], [0.92, 0.66, 1.02, 1.02])):
        ratios.append(round(i_ / f_, 2))
    RES["V2_EMD_ratio_Isaac_over_FLASH_projectpage"] = ratios


# ---------------------------------------------------------------- V3: 표 2 변환
T2 = {  # (asset, method): (mse_mean, mse_sd, cd, emd, lat)
    ("towel", "Ours"): (2.33, 1.21, 1.94, 1.98, 7.4),
    ("towel", "DPM"): (7.17, 2.22, 3.89, 4.01, 318.5),
    ("towel", "DeFM"): (43.36, 34.66, 5.51, 6.53, 12.6),
    ("shirt", "Ours"): (24.23, 11.49, 3.65, 4.67, 12.4),
    ("shirt", "DPM"): (45.74, 14.19, 5.93, 8.21, 512.9),
    ("shirt", "DeFM"): (43.52, 22.02, 4.05, 5.93, 18.0),
}


def v3_table2():
    rows = []
    for (a, m), (mse, sd, cd, emd, lat) in T2.items():
        rms = math.sqrt(mse)
        rows.append({"asset": a, "method": m, "rms_mm": round(rms, 2), "CD_mm": cd, "CD_over_RMS": round(cd / rms, 2),
                     "CD_gt_RMS": cd > rms, "EMD_over_RMS": round(emd / rms, 2)})
    RES["V3_table2_rms"] = rows
    # 이긴 방법 확인
    best = {}
    for a in ("towel", "shirt"):
        for j, name in enumerate(["MSE", "CD", "EMD", "Lat"]):
            vals = {m: T2[(a, m)][[0, 2, 3, 4][j]] for m in ("Ours", "DPM", "DeFM")}
            best[f"{a}_{name}"] = min(vals, key=vals.get)
    RES["V3_best_method_per_metric"] = best
    # 개선 배율
    RES["V3_ratios"] = {
        "towel_MSE_DPM_over_Ours": round(7.17 / 2.33, 2), "towel_MSE_DeFM_over_Ours": round(43.36 / 2.33, 1),
        "shirt_MSE_DPM_over_Ours": round(45.74 / 24.23, 2), "shirt_MSE_DeFM_over_Ours": round(43.52 / 24.23, 2),
        "towel_latency_DPM_over_Ours": round(318.5 / 7.4, 1), "shirt_latency_DPM_over_Ours": round(512.9 / 12.4, 1),
    }
    # 변동계수
    RES["V3_cv"] = {f"{a}_{m}": round(T2[(a, m)][1] / T2[(a, m)][0], 2) for (a, m) in T2}
    # 표 2 대 부록 표 D.1 (온 설정)
    RES["V3_D1_vs_T2_towel_ours_equal"] = (2.33, 1.21, 1.94, 1.98) == T2[("towel", "Ours")][:4]


# ---------------------------------------------------------------- V4-V7: 실로봇 비율
T3 = {  # (task, variant): successes of 10   [표 3]
    ("single", "full"): 9, ("single", "DPM"): 6, ("single", "Isaac"): 5, ("single", "base"): 3,
    ("dual", "full"): 8, ("dual", "DPM"): 3, ("dual", "Isaac"): 4, ("dual", "base"): 1,
}
T4 = {"long-sleeve": 8, "shorts": 7, "towel reverse diag.": 8, "towel mid-fold disturb.": 8}
D1 = {("single", "on"): 9, ("single", "off"): 6, ("dual", "on"): 8, ("dual", "off"): 5}


def cmh_exact(strata):
    """층별 2x2 (k1,n1,k2,n2)의 정확 조건부 검정(양측, 확률 합산). 층마다 초기하 분포를 합성곱한다."""
    pmf = np.array([1.0])
    obs = 0
    for k1, n1, k2, n2 in strata:
        K = k1 + k2
        lo, hi = max(0, K - n2), min(K, n1)
        sp = np.array([stats.hypergeom.pmf(x, n1 + n2, n1, K) for x in range(lo, hi + 1)])
        # 합성곱을 위해 0..hi 로 오프셋 맞춤
        full = np.zeros(hi + 1)
        full[lo:hi + 1] = sp
        pmf = np.convolve(pmf, full)
        obs += k1
    p_obs = pmf[obs]
    return float(pmf[pmf <= p_obs + 1e-12].sum())


def v4_table3():
    out = []
    for task in ("single", "dual"):
        full = T3[(task, "full")]
        for v in ("DPM", "Isaac", "base"):
            k = T3[(task, v)]
            lo, hi = wilson(k, 10)
            out.append({"task": task, "full": f"{full}/10", "variant": v, "variant_sr": f"{k}/10",
                        "fisher_p": round(fisher(full, 10, k, 10), 4),
                        "wilson_variant_pct": [round(100 * lo), round(100 * hi)]})
    RES["V4_table3_fisher"] = out
    pooled = []
    for v in ("DPM", "Isaac", "base"):
        strata = [(T3[(t, "full")], 10, T3[(t, v)], 10) for t in ("single", "dual")]
        kk = sum(T3[(t, v)] for t in ("single", "dual"))
        pooled.append({"variant": v, "full": "17/20", "variant_sr": f"{kk}/20",
                       "fisher_p_pooled_20v20": round(fisher(17, 20, kk, 20), 5),
                       "cmh_exact_p": round(cmh_exact(strata), 5)})
    RES["V4_table3_pooled"] = pooled
    RES["V4_full_wilson"] = {"single": [round(100 * x) for x in wilson(9, 10)], "dual": [round(100 * x) for x in wilson(8, 10)],
                             "pooled_17_20": [round(100 * x) for x in wilson(17, 20)]}


def v5_table4():
    out = []
    for t, k in T4.items():
        lo, hi = wilson(k, 10)
        out.append({"task": t, "sr": f"{k}/10", "wilson95_pct": [round(100 * lo), round(100 * hi)]})
    RES["V5_table4_wilson"] = out
    tot = sum(T4.values())
    RES["V5_table4_pooled"] = {"k": tot, "n": 40, "wilson95_pct": [round(100 * x) for x in wilson(tot, 40)]}
    # 표 4 가장 낮은 값 7/10 대 표 3 전체 가동 9/10, 8/10
    RES["V5_table4_vs_table3_full"] = {
        "towel_single_9of10_vs_long_sleeve_8of10_fisher_p": round(fisher(9, 10, 8, 10), 3),
        "shorts_7of10_vs_single_9of10_fisher_p": round(fisher(7, 10, 9, 10), 3),
    }
    # 정규 성공 대 기준선 하한 가능성: "변형 대 가동" 에서 개별 비율이 3/10인 기준선의 상한(Wilson)
    RES["V5_note"] = "표 4에는 기준선 행이 없어 교란 회복에서 MPPI 이득은 비교할 수 없다."


def v6_ablation():
    out = []
    for t in ("single", "dual"):
        out.append({"task": t, "on": D1[(t, "on")], "off": D1[(t, "off")],
                    "fisher_p": round(fisher(D1[(t, "on")], 10, D1[(t, "off")], 10), 4)})
    RES["V6_D1_fisher"] = out
    RES["V6_D1_pooled"] = {"on": "17/20", "off": "11/20", "fisher_p": round(fisher(17, 20, 11, 20), 4),
                           "cmh_exact_p": round(cmh_exact([(9, 10, 6, 10), (8, 10, 5, 10)]), 4)}
    # 표 3 DPM 행과 D.1 증강 off 행은 서로 다른 소거(상태 추정기 교체 vs 증강 제거)지만 같은 열 구조를 가진다
    RES["V6_D1_recon_off_better_on_single"] = {"MSE_on": 2.33, "MSE_off": 1.85, "CD_on": 1.94, "CD_off": 1.72, "EMD_on": 1.98, "EMD_off": 1.74}
    # 표 3의 DPM 행(실 SR 6/10)과 D.1 off(6/10), 둘 다 6/10
    RES["V6_note"] = "단일 팔 off는 합성 검증 오차가 더 낮다(MSE 1.85 대 2.33), 그런데 실 SR은 6/10 대 9/10."


def v7_power():
    """n=10 대 10 Fisher 검정의 검정력과 필요한 표본 수(정확 계산)."""
    cache = {}

    def fcached(a, b, n):
        if (a, b, n) not in cache:
            cache[(a, b, n)] = fisher(a, n, b, n)
        return cache[(a, b, n)]

    def power(p1, p2, n, alpha=0.05):
        pw = 0.0
        pm1 = [stats.binom.pmf(k, n, p1) for k in range(n + 1)]
        pm2 = [stats.binom.pmf(k, n, p2) for k in range(n + 1)]
        for a in range(n + 1):
            for b in range(n + 1):
                if fcached(a, b, n) <= alpha:
                    pw += pm1[a] * pm2[b]
        return pw
    out = {}
    for p1, p2 in [(0.9, 0.6), (0.9, 0.3), (0.8, 0.5), (0.8, 0.3)]:
        out[f"{p1}_vs_{p2}"] = {"power_n10": round(power(p1, p2, 10), 3)}
        for n in (15, 20, 25, 30, 40):
            pw = power(p1, p2, n)
            if pw >= 0.8:
                out[f"{p1}_vs_{p2}"]["n_for_80pct_power"] = n
                out[f"{p1}_vs_{p2}"]["power_at_n"] = round(pw, 3)
                break
    RES["V7_power"] = out


# ---------------------------------------------------------------- V8: 부록 B.3 분해
def v8_grasp():
    RES["V8_grasp_decomposition"] = {
        "single_diag": {"grasp": "10/10", "end_to_end": "9/10", "conditional": "9/10"},
        "dual_sym": {"grasp": "9/10", "end_to_end": "8/10", "conditional": "8/9 = 0.889"},
        "disturb": {"grasp": "10/10", "end_to_end": "8/10", "conditional": "8/10"},
        "note": "정렬 실패 건수: 단일 1, 이중 1, 교란 2. 전체 기준선 행(DPM, Isaac, base)의 파지 성공은 보고되지 않음.",
    }


# ---------------------------------------------------------------- V9: 시뮬 폐루프 수치(부록 C.3)
def v9_closed_loop():
    diag = 30 * math.sqrt(2)
    RES["V9_closed_loop"] = {
        "diagonal_of_30cm_square_cm": round(diag, 2),
        "reported_start_error_cm": 42.2,
        "min_steps_at_2cm_per_step_clamp": math.ceil(42.2 / 2.0),   # 부록 B.1의 행동 클램프 20 mm/스텝
        "reported_steps_range": [22, 31],
        "success_18_of_20_wilson95_pct": [round(100 * x, 1) for x in wilson(18, 20)],
        "real_single_arm_9_of_10_wilson95_pct": [round(100 * x, 1) for x in wilson(9, 10)],
        "sim_vs_real_fisher_p": round(fisher(18, 20, 9, 10), 3),
    }


def compositions(n, k):
    """n을 k개의 0 이상 정수로 나누는 모든 방법(재귀)."""
    if k == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for rest in compositions(n - i, k - 1):
            yield (i,) + rest


def feasible_int_stats(n, mean_r, sd_r, support, ddof=1):
    """정수 지원 집합 위의 n개 값이 보고된 평균·SD(소수 첫째 자리 반올림)와 맞는 분포가 있는지 센다."""
    sup = np.array(support, dtype=float)
    sols = []
    for counts in compositions(n, len(support)):
        c = np.array(counts)
        m = (c * sup).sum() / n
        var = (c * (sup - m) ** 2).sum() / (n - ddof)
        if round(m, 1) == mean_r and round(math.sqrt(var), 1) == sd_r:
            sols.append(dict(zip(map(int, support), map(int, counts))))
    return sols


def v10_table_c1():
    out = {}
    for push, mean_r, sd_r in [(6, 4.0, 0.0), (9, 4.1, 0.7), (12, 6.0, 1.0)]:
        sup = list(range(2, 9))  # 2~8 스텝
        s1 = feasible_int_stats(20, mean_r, sd_r, sup, ddof=1)
        s0 = feasible_int_stats(20, mean_r, sd_r, sup, ddof=0)
        out[f"push_{push}cm"] = {"n_distributions_ddof1": len(s1), "n_distributions_ddof0": len(s0),
                                  "example_ddof1": s1[0] if s1 else None}
    RES["V10_tableC1_integer_feasibility"] = out
    # 20 mm/스텝 클램프에서 밀림 d cm를 되돌리는 데 필요한 최소 스텝(밀림이 오차 방향과 정확히 반대일 때의 최악)
    RES["V10_worst_case_min_steps_at_20mm"] = {d: math.ceil(d / 2.0) for d in (6, 9, 12)}


# ---------------------------------------------------------------- V11: 지연 산술(부록 B.3)
def v11_latency():
    towel = 95.8 + 8.5 + 0.5 + 970
    shorts_total = 95.8 + 11.6 + 0.5 + 2550
    RES["V11_latency"] = {
        "towel_step_ms": round(towel, 1), "towel_hz": round(1000 / towel, 2),
        "shorts_step_ms_assuming_same_seg_and_write": round(shorts_total, 1), "shorts_hz": round(1000 / shorts_total, 2),
        "mppi_share_towel_pct": round(100 * 970 / towel, 1),
        "mesh_ratio_1156_over_464": round(1156 / 464, 2), "mppi_ratio_2550_over_970": round(2550 / 970, 2),
        "fold_compute_time_s_22_to_31_steps_towel": [round(22 * towel / 1000, 1), round(31 * towel / 1000, 1)],
        "table1_FLASH_K256_mppi_s_H6": 1.42, "scaled_to_H5_s": round(1.42 * 5 / 6, 2), "reported_H5_towel_s": 0.97,
        "implied_sim_steps_in_970ms_at_68.9ms": round(970 / 68.9, 1), "expected_H5x4_substeps": 20,
        "r2s_latency_towel_table2_ms": 7.4, "r2s_latency_towel_B3_ms": 8.5,
    }


# ---------------------------------------------------------------- V12: 매개변수와 토큰 수 어림
def v12_params():
    d, dvit, h, L, ff = 256, 384, 8, 4, 1024     # ff = 4d 가정(논문은 "standard", 폭 미기재)
    proj = dvit * d + d
    attn = 4 * d * d + 4 * d
    ffn = 2 * d * ff + ff + d
    norms = 2 * 2 * d
    block = attn + ffn + norms
    dec = 2 * d + (d * 512 + 512) + (512 * 512 + 512) + (512 * 3 + 3) + 2 * 512 * 0
    base = proj + L * block + dec
    out = {}
    for name, N in (("towel", 464), ("shorts", 1156), ("shirt", 1379)):
        tok = N * d
        out[name] = {"N": N, "canonical_token_params": tok, "total_trainable_est_M": round((base + tok) / 1e6, 2),
                     "seq_len": N + 784, "attn_matrix_entries_per_head_M": round((N + 784) ** 2 / 1e6, 2)}
    out["assumption"] = "FFN 은닉폭 4d=1024 가정(논문은 'standard'라고만 씀). 판독용 어림이며 보고값이 아니다."
    RES["V12_params"] = out
    RES["V12_patches"] = {"392/14": 392 / 14, "Mv": (392 // 14) ** 2}


# ---------------------------------------------------------------- V13: 판본 간 표 동일성
def v13_versions():
    import re
    p = Path(__file__).resolve().parents[1] / "paper"
    if not (p / "paper_v1.txt").exists():
        # v1 텍스트는 저장소에 넣지 않는다: curl -L -o paper/paper_v1.pdf https://arxiv.org/pdf/2606.24552v1 후
        # pdftotext -layout paper/paper_v1.pdf paper/paper_v1.txt 로 만든다. 없으면 이 검사를 건너뛴다.
        RES["V13_skipped"] = "paper/paper_v1.txt 없음"
        return
    a = re.sub(r"\s+", " ", (p / "paper_v1.txt").read_text())
    b = re.sub(r"\s+", " ", (p / "paper.txt").read_text())
    probes = ["9/10", "6/10", "5/10", "3/10", "8/10", "1/10", "7/10", "41.0 0.85 8.40 ± 8.72 56.8 ± 73.1 40%",
              "137.3 2.84 2.08 ± 1.93 3.9 ± 3.2 95%", "2.33 ± 1.21 1.94 ± 0.41 1.98 ± 0.45 7.4 ± 0.6",
              "24.23 ± 11.49 3.65 ± 0.78 4.67 ± 1.24", "2.33±1.21 1.94±0.41 1.98±0.45 9/10", "3.59±2.08 2.42±0.57 2.51±0.68 8/10"]
    RES["V13_table_strings_in_v1_and_v2"] = {s: [s in a, s in b] for s in probes}
    kw = ["no trial was discarded or re-run", "closed-loop correction", "stronger robustness", "real-time",
          "quasi-static", "5.2 mm", "13.6 mm", "10.2", "RTX 4090", "~12 h", "Appendix C"]
    RES["V13_keywords_v1_v2"] = {s: [s in a, s in b] for s in kw}
    RES["V13_pdf_pages"] = {"v1": 23, "v2": 27}


def main():
    v1_table1()
    v2_median_bound()
    v3_table2()
    v4_table3()
    v5_table4()
    v6_ablation()
    v7_power()
    v8_grasp()
    v9_closed_loop()
    v10_table_c1()
    v11_latency()
    v12_params()
    v13_versions()
    OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1))
    for k, v in RES.items():
        print("==", k)
        print(json.dumps(v, ensure_ascii=False)[:900])


if __name__ == "__main__":
    main()
