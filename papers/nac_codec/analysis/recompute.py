"""NAC (arXiv:2606.21372 v2) 표 수치 재계산과 검정. 단일 프로세스, 수 초.

입력은 논문 v2의 표 1, 2, 3, 6 값을 옮긴 것이다. 새 실험은 하지 않는다.
  V1  표 2 시뮬 평균의 정수 시도 수 가능성 (LIBERO 8 x 500 = 4000, RoboMimic 8 x 200 = 1600 가정)
  V2  ±의 정의 후보(표준편차 / 표준오차)별 이항 분포 하한과의 비교
  V3  표 2 시뮬 쌍별 검정 (±를 표준오차로 읽을 때와 시도 합산 이항 검정)
  V4  표 6 실로봇 평균 재계산, Fisher 정확 검정, Wilson 구간, 과제별 부호 검정, 검출 가능 차이
  V5  표 3 비트 수, 토큰 비율, 재구성 시간 합
  V6  표 1 ablation: 재구성 MSE와 정책 성공률의 순위 상관, 같은 설정이 여러 값으로 나오는 정도
  V7  본문 문장 수치 (11.71, 5.56, 19x, 3x, 725/13,757 등)
실행: python3 analysis/recompute.py   → analysis/recompute_results.json
"""
import json
import math
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}

# ---------------- 논문 값 (v2) ----------------
T2 = {  # 표 2: 평균, ± (정의 없음)
    "LIBERO-10": {"Bin": (3.95, 0.8), "Diffusion Policy": (25.48, 1.3), "FAST": (38.02, 1.3),
                  "VQ-VLA": (10.85, 1.85), "OAT": (44.17, 1.2), "NAC": (49.73, 1.0)},
    "RoboMimic": {"Bin": (7.56, 1.05), "Diffusion Policy": (27.25, 1.87), "FAST": (28.38, 2.37),
                  "VQ-VLA": (21.44, 1.45), "OAT": (31.94, 2.15), "NAC": (33.94, 1.86)},
}
N_TOTAL = {"LIBERO-10": 8 * 500, "RoboMimic": 8 * 200}   # 시드 8, 시드당 LIBERO 10과제 x 50, RoboMimic 4과제 x 50 (코드 n_test 500, 200)
N_SEED = {"LIBERO-10": 500, "RoboMimic": 200}

T6_TASKS = ["Weighing", "Grapes", "Marker", "Two Blocks", "Three Blocks", "Chess", "Place Stone", "Fold Towel"]
T6 = {  # 표 6: 과제별 성공률 (%), 과제당 10회
    "Bin": [50, 0, 0, 0, 0, 0, 0, 0],
    "Diffusion Policy": [40, 30, 30, 0, 0, 0, 0, 80],
    "FAST": [80, 80, 60, 0, 0, 10, 10, 80],
    "VQ-VLA": [90, 30, 30, 0, 0, 0, 50, 50],
    "OAT": [40, 80, 50, 0, 10, 40, 30, 70],
    "NAC": [90, 100, 50, 30, 0, 10, 40, 80],
}
T6_AVG = {"Bin": 6.25, "Diffusion Policy": 22.5, "FAST": 40.0, "VQ-VLA": 31.25, "OAT": 40.0, "NAC": 50.0}

T3 = {  # 표 3: 파라미터(M), 토큰, |K|, 비트, enc ms, dec ms, recon ms
    "Bin": (0.002, 224, 1024, 2240, 0.045, 0.039, 0.079),
    "OAT": (65.207, 12, 1024, 120, 0.931, 2.392, 3.347),
    "VQ-VLA": (65.556, 12, 1024, 120, 7.086, 4.045, 11.049),
    "FAST": (0.000, 36, 1024, 360, 0.170, 0.110, 0.290),
    "NAC": (63.006, 12, 1024, 120, 1.270, 2.183, 3.536),
}

T1A = {  # 표 1a: 성공률, ±, MSE, MSE ±
    "L1": (44.78, 2.48, 0.002, 0.005), "MSE": (49.2, 1.54, 0.0008, 0.0007), "DCT": (47.85, 1.18, 0.0007, 0.0008),
    "Log-Mel": (0.0, 0.11, 0.038, 0.026), "Mel(no log)": (44.6, 2.0, 0.0006, None),
    "Spectrogram": (48.3, 2.92, 0.0002, 0.001),
}
T1B = {
    "DAC": (49.45, 2.02, 0.0005, 0.0018), "MPD": (46.28, 1.48, 0.0005, 0.0007), "MRD": (45.68, 1.72, 0.0004, 0.0006),
    "None(Spec)": (0.0, None, 0.35, 0.12), "None(MSE)": (50.1, 1.7, 0.0010, None),
}
T1C = {"ISTFT": (48.3, 2.92, 0.0002, 0.001), "Linear": (42.1, 1.54, 0.0006, 0.001)}


def r2(x, mode):
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=mode))


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h) * 100, (c + h) * 100


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


# ---------------- V1: 정수 시도 수 가능성 ----------------
v1 = {}
for env, rows in T2.items():
    n = N_TOTAL[env]
    v1[env] = {"n_trials_assumed": n, "rows": {}}
    for m, (mean, _) in rows.items():
        k_exact = mean / 100 * n
        k = round(k_exact)
        rate = k / n * 100
        exact = Decimal(k * 100) / Decimal(n)              # 정확한 십진수 (4000, 1600 시도는 소수 셋째 자리에서 끝난다)
        v1[env]["rows"][m] = {
            "reported": mean, "k_nearest": k, "rate_of_k": rate,
            "abs_gap": abs(rate - mean),
            "k_exact_float": k_exact,
            "exact_percent": str(exact),
            "rounds_to_reported_half_up": r2(exact, ROUND_HALF_UP) == mean,
            "rounds_to_reported_half_even": r2(exact, ROUND_HALF_EVEN) == mean,
        }
res["V1_integer_feasibility"] = v1

# ---------------- V2: ± 정의 후보와 이항 하한 ----------------
v2 = {}
for env, rows in T2.items():
    ns = N_SEED[env]
    v2[env] = {}
    for m, (mean, pm) in rows.items():
        p = mean / 100
        binom_sd = math.sqrt(p * (1 - p) / ns) * 100        # 시드 하나(ns회)의 성공률 표준편차 하한(독립 시행 가정)
        v2[env][m] = {
            "pm": pm, "binom_sd_per_seed": binom_sd,
            "ratio_if_pm_is_sd": pm / binom_sd,              # < 1 이면 이항 하한보다 작음
            "implied_sd_if_pm_is_sem": pm * math.sqrt(8),
            "ratio_if_pm_is_sem": pm * math.sqrt(8) / binom_sd,
        }
res["V2_pm_definition"] = v2

# ---------------- V3: 표 2 쌍별 비교 ----------------
v3 = {}
for env, rows in T2.items():
    n = N_TOTAL[env]
    out = {}
    nac_m, nac_pm = rows["NAC"]
    for m, (mean, pm) in rows.items():
        if m == "NAC":
            continue
        d = nac_m - mean
        se_sem = math.hypot(nac_pm, pm)                     # ±가 표준오차일 때
        se_sd = math.hypot(nac_pm, pm) / math.sqrt(8)       # ±가 시드 간 표준편차일 때
        z_sem, z_sd = d / se_sem, d / se_sd
        p1, p2 = nac_m / 100, mean / 100
        se_pool = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n) * 100
        z_bin = d / se_pool
        out[m] = {
            "diff_pt": round(d, 2),
            "z_if_pm_sem": z_sem, "p_if_pm_sem": float(2 * stats.norm.sf(abs(z_sem))),
            "z_if_pm_sd": z_sd, "p_if_pm_sd": float(2 * stats.norm.sf(abs(z_sd))),
            "z_pooled_binomial": z_bin, "p_pooled_binomial": float(2 * stats.norm.sf(abs(z_bin))),
        }
    v3[env] = out
res["V3_sim_pairwise"] = v3

# ---------------- V4: 실로봇 ----------------
v4 = {"avg_recomputed": {}, "avg_matches": {}, "counts_of_80": {}, "fisher_nac_vs": {}, "wilson_80": {},
      "per_task_diff_vs_nac": {}, "sign_tests": {}, "per_task_fisher_nac_vs_oat_fast": {}}
for m, row in T6.items():
    avg = sum(row) / 8
    v4["avg_recomputed"][m] = avg
    v4["avg_matches"][m] = abs(avg - T6_AVG[m]) < 1e-9
    k = sum(row) // 10
    v4["counts_of_80"][m] = k
    v4["wilson_80"][m] = [round(x, 1) for x in wilson(k, 80)]
nk = v4["counts_of_80"]["NAC"]
for m in T6:
    if m == "NAC":
        continue
    v4["fisher_nac_vs"][m] = {"k": v4["counts_of_80"][m], "p_two_sided": fisher(nk, 80, v4["counts_of_80"][m], 80),
                              "p_one_sided": float(stats.fisher_exact(
                                  [[nk, 80 - nk], [v4["counts_of_80"][m], 80 - v4["counts_of_80"][m]]],
                                  alternative="greater")[1])}
    diffs = [a - b for a, b in zip(T6["NAC"], T6[m])]
    v4["per_task_diff_vs_nac"][m] = dict(zip(T6_TASKS, diffs))
    wins, losses, ties = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs), sum(d == 0 for d in diffs)
    nn = wins + losses
    sign_p = float(stats.binomtest(wins, nn, 0.5).pvalue) if nn else 1.0
    try:
        wil = stats.wilcoxon(diffs, zero_method="wilcox", method="auto")
        wp = float(wil.pvalue)
    except ValueError:
        wp = None
    v4["sign_tests"][m] = {"wins": wins, "losses": losses, "ties": ties, "sign_p": sign_p, "wilcoxon_p": wp,
                           "paired_t_p": float(stats.ttest_1samp(diffs, 0).pvalue)}
for m in ("OAT", "FAST"):
    v4["per_task_fisher_nac_vs_oat_fast"][m] = {
        t: fisher(a // 10, 10, b // 10, 10) for t, a, b in zip(T6_TASKS, T6["NAC"], T6[m])}
# 과제별 최고(동률 포함)
best = {}
for i, t in enumerate(T6_TASKS):
    top = max(T6[m][i] for m in T6)
    best[t] = {"best_value": top, "methods": [m for m in T6 if T6[m][i] == top]}
v4["best_per_task"] = best
v4["nac_best_or_tied_count"] = sum("NAC" in b["methods"] for b in best.values())
v4["nac_strict_best_count"] = sum(b["methods"] == ["NAC"] for b in best.values())
# 검출 가능한 차이: 방법당 80회, 양측 alpha 0.05, 검정력 0.8, 기준 성공률 0.40
def mde(n, p0=0.4, alpha=0.05, power=0.8):
    za, zb = stats.norm.isf(alpha / 2), stats.norm.isf(1 - power)
    for d in np.arange(0.01, 0.6, 0.005):
        p1 = p0 + d
        pb = (p0 + p1) / 2
        need = ((za * math.sqrt(2 * pb * (1 - pb)) + zb * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) / d) ** 2
        if need <= n:
            return float(d)
    return None
v4["mde_pt_n80_base40"] = mde(80) * 100
v4["mde_pt_n10_base40"] = mde(10) * 100
# NAC 대 OAT 격차 +10pt를 80% 검정력으로 보려면 방법당 시도 수
def need_n(p0, p1, alpha=0.05, power=0.8):
    za, zb = stats.norm.isf(alpha / 2), stats.norm.isf(1 - power)
    pb = (p0 + p1) / 2
    return float(((za * math.sqrt(2 * pb * (1 - pb)) + zb * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) / (p1 - p0)) ** 2)
v4["n_per_arm_for_50_vs_40"] = need_n(0.4, 0.5)
res["V4_real_world"] = v4

# ---------------- V5: 표 3 ----------------
v5 = {}
for m, (par, tok, K, bits, enc, dec, rec) in T3.items():
    v5[m] = {"bits_recomputed": tok * math.log2(K), "bits_match": abs(tok * math.log2(K) - bits) < 1e-9,
             "enc_plus_dec": enc + dec, "recon_reported": rec, "recon_minus_sum": rec - (enc + dec),
             "rel_gap_pct": (rec - (enc + dec)) / (enc + dec) * 100}
v5["ratios"] = {
    "tokens_bin_over_nac": 224 / 12, "tokens_fast_over_nac": 36 / 12,
    "recon_vqvla_over_nac": 11.049 / 3.536, "recon_nac_over_oat": 3.536 / 3.347,
    "recon_nac_over_fast": 3.536 / 0.290, "params_nac_over_oat": 63.006 / 65.207,
    "raw_float32_bits_per_chunk": 32 * 32 * 7, "compression_vs_float32": 32 * 32 * 7 / 120,
    "bits_per_chunk_per_action_dim_step": 120 / (32 * 7),
}
v5["layout_from_config"] = {
    "horizon": 32, "action_dim": 7, "L": 224, "ratios": [7, 4], "hop": 28, "vq_scales": [2, 1],
    "latent_frames": 224 // 28, "tokens_scale0": (224 // 28) // 2, "tokens_scale1": (224 // 28),
    "tokens_total": (224 // 28) // 2 + (224 // 28), "divisor_ratios_x_scales": 28 * 2,
    "divisible": 224 % 56 == 0, "bits_per_token_fsq_8_8_4_4": math.log2(8 * 8 * 4 * 4),
}
res["V5_table3"] = v5

# ---------------- V6: 표 1 ----------------
v6 = {}
a_rows = [k for k in T1A if k != "Log-Mel"]
succ = [T1A[k][0] for k in a_rows]
mse = [T1A[k][2] for k in a_rows]
rho = stats.spearmanr(mse, succ)
v6["spearman_mse_vs_success_1a_excl_logmel"] = {"rows": a_rows, "rho": float(rho.statistic), "p": float(rho.pvalue)}
succ_all = [v[0] for v in T1A.values()]
mse_all = [v[2] for v in T1A.values()]
rho2 = stats.spearmanr(mse_all, succ_all)
v6["spearman_mse_vs_success_1a_all"] = {"rho": float(rho2.statistic), "p": float(rho2.pvalue)}
# 1a + 1b + 1c 의 비실패 행 전체
pool = {}
for src, tab in (("1a", T1A), ("1b", T1B), ("1c", T1C)):
    for k, v in tab.items():
        pool[f"{src}:{k}"] = v
pool_ok = {k: v for k, v in pool.items() if v[0] > 10}
rho3 = stats.spearmanr([v[2] for v in pool_ok.values()], [v[0] for v in pool_ok.values()])
v6["spearman_all_nonfailed_rows"] = {"n": len(pool_ok), "rho": float(rho3.statistic), "p": float(rho3.pvalue)}
v6["duplicate_rows"] = {"ISTFT_equals_Spectrogram": T1C["ISTFT"] == T1A["Spectrogram"]}
v6["nominal_default_success_values"] = {
    "Table1a MSE": 49.2, "Table1b DAC": 49.45, "Table2 NAC LIBERO-10": 49.73,
    "range_pt": 49.73 - 49.2,
    "z_1a_vs_T2_if_pm_sem": (49.73 - 49.2) / math.hypot(1.54, 1.0)}
v6["nominal_default_recon_mse"] = {"Table1a MSE": 0.0008, "Table1b DAC": 0.0005}
def zpair(a, b):  # (평균, ±) 두 개를 ±=표준오차로 읽어 비교
    return (a[0] - b[0]) / math.hypot(a[1], b[1])
v6["pairs_pm_as_sem"] = {
    "DAC vs None(MSE)": {"diff": 49.45 - 50.1, "z": zpair((49.45, 2.02), (50.1, 1.7))},
    "MSE vs Spectrogram": {"diff": 49.2 - 48.3, "z": zpair((49.2, 1.54), (48.3, 2.92))},
    "MSE vs L1": {"diff": 49.2 - 44.78, "z": zpair((49.2, 1.54), (44.78, 2.48))},
    "MSE vs Mel(no log)": {"diff": 49.2 - 44.6, "z": zpair((49.2, 1.54), (44.6, 2.0))},
    "DAC vs MPD": {"diff": 49.45 - 46.28, "z": zpair((49.45, 2.02), (46.28, 1.48))},
    "DAC vs MRD": {"diff": 49.45 - 45.68, "z": zpair((49.45, 2.02), (45.68, 1.72))},
    "ISTFT vs Linear (Spec. loss)": {"diff": 48.3 - 42.1, "z": zpair((48.3, 2.92), (42.1, 1.54))},
    "None(MSE) vs NAC Table2": {"diff": 50.1 - 49.73, "z": zpair((50.1, 1.7), (49.73, 1.0))},
}
for d in v6["pairs_pm_as_sem"].values():
    d["p"] = float(2 * stats.norm.sf(abs(d["z"])))
res["V6_table1"] = v6

# ---------------- V7: 본문 수치 ----------------
res["V7_text_numbers"] = {
    "NAC-FAST libero": round(49.73 - 38.02, 2), "NAC-OAT libero": round(49.73 - 44.17, 2),
    "NAC-FAST robomimic": round(33.94 - 28.38, 2), "NAC-OAT robomimic": round(33.94 - 31.94, 2),
    "fig5_coarse_tokens_total": 13757 * 4, "fig5_usage_pct": {"NAC": 725 / 1024 * 100, "no-log": 745 / 1024 * 100,
                                                            "spec-no-disc": 907 / 1024 * 100, "log-mel": 108 / 1024 * 100},
    "fig5_sample_vs_table1": {"fig5_chunks": 13757, "table1_chunks": 14000, "diff": 14000 - 13757},
    "table2_real_total_trials_per_method": 80,
}

OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf-8")


# ---------------- 요약 출력 ----------------
def main():
    print("V1 정수 시도 수 (가정 n):")
    for env, d in v1.items():
        for m, r in d["rows"].items():
            print(f"  {env:10s} {m:17s} {r['reported']:6.2f}  k={r['k_nearest']:5d} -> {r['rate_of_k']:.3f}  "
                  f"exact={r['exact_percent']} half_up={r['rounds_to_reported_half_up']} "
                  f"half_even={r['rounds_to_reported_half_even']}")
    print("V2 ± 비교 (이항 하한 대비):")
    for env, d in v2.items():
        for m, r in d.items():
            print(f"  {env:10s} {m:17s} ±{r['pm']:5.2f} binomSD={r['binom_sd_per_seed']:.2f} "
                  f"SD읽기 x{r['ratio_if_pm_is_sd']:.2f}  SEM읽기 x{r['ratio_if_pm_is_sem']:.2f}")
    print("V3 NAC 대 각 방법:")
    for env, d in v3.items():
        for m, r in d.items():
            print(f"  {env:10s} vs {m:17s} diff={r['diff_pt']:6.2f} p(SEM)={r['p_if_pm_sem']:.4f} "
                  f"p(SD)={r['p_if_pm_sd']:.2g} p(binom)={r['p_pooled_binomial']:.2g}")
    print("V4 실로봇:")
    print("  평균 일치:", v4["avg_matches"])
    for m, r in v4["fisher_nac_vs"].items():
        print(f"  NAC 40/80 vs {m} {r['k']}/80: Fisher p={r['p_two_sided']:.4f} (단측 {r['p_one_sided']:.4f})")
    for m, r in v4["sign_tests"].items():
        print(f"  과제별 {m}: 승{r['wins']} 패{r['losses']} 무{r['ties']} sign p={r['sign_p']:.3f} "
              f"Wilcoxon p={r['wilcoxon_p']} t p={r['paired_t_p']:.3f}")
    print("  NAC 최고(동률 포함) 과제 수:", v4["nac_best_or_tied_count"], " 단독 최고:", v4["nac_strict_best_count"])
    print("  MDE(방법당 80회, 기준 40%):", round(v4["mde_pt_n80_base40"], 1), "pt; 50 대 40에 필요한 방법당 시도:",
          round(v4["n_per_arm_for_50_vs_40"]))
    print("V5:", {m: (v5[m]["bits_match"], round(v5[m]["recon_minus_sum"], 3)) for m in T3})
    print("V6 Spearman:", v6["spearman_mse_vs_success_1a_excl_logmel"], v6["spearman_all_nonfailed_rows"])
    for k, d in v6["pairs_pm_as_sem"].items():
        print(f"  {k:30s} diff={d['diff']:6.2f} z={d['z']:.2f} p={d['p']:.3f}")


if __name__ == "__main__":
    main()
