"""논문 표 1·2와 본문 수치를 다시 계산한다 (arXiv:2611.12804 v1). 수 초.

실행: python3 analysis/recompute.py   → analysis/recompute_results.json
의존: numpy, scipy

원자료(시드별 성공률)는 논문에 없다. 그래서 보고된 값에서 거꾸로 풀 수 있는 것만 푼다.
  V1  시도 수 합계와 시드 평균 vs 시도 합산 비율
  V2  표 2 차이값과 z 합(2.49) 검산
  V3  검출 가능 한계 → 표준오차 → Welch 신뢰구간(본문)과 맞는지
  V4  Welch 자유도 상한 K1+K2-2 검사, (표준오차, 자유도)에서 시드 간 분산 역산, 같은 분산 가정 V
  V5  관측 차이의 Welch p값, z 기준과 t 기준 검출 한계
  V6  p = 0.5 혼합 설계의 한계에서 시도 잡음 비율 역산, 이항 잡음 추정과 비교
  V7  시도 단위 합산 검정(군집 무시)과 시드 단위 검정의 p값 비교
  V8  표 1 부분집합 순위 뒤집힘 쌍 세기
  V9  필요 시드 수 계획표(역산한 분산 사용)
  V10 시드 6-8개로 추정한 분산의 카이제곱 구간, 설정 공간 크기
"""
import json
import math
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")

# 논문 5.1, 표 2
N_TRIALS = {"door": {"spot": 62, "stretch": 130}, "drawer": {"spot": 88, "stretch": 288}}
N_SEEDS = {"door": {"spot": 6, "stretch": 7}, "drawer": {"spot": 6, "stretch": 8}}
RATE = {"door": {"spot": 0.7109, "stretch": 0.1802}, "drawer": {"spot": 0.8792, "stretch": 0.6352}}
DIFF = {"door": 0.5307, "drawer": 0.2439}
MARGIN = {"door": (0.333, 0.256), "drawer": (0.232, 0.183)}  # (독립 시드, p=0.5)
WELCH_CI = {"door": (23.4, 82.8, 10.4), "drawer": (4.2, 44.6, 12.9)}  # 점, 점, 자유도
# 표 1 (%)
T1 = {
    "drawer": {"spot": {"same": 92.7, "s1": 65.0, "s2": 100.0, "s3": 87.5},
               "stretch": {"same": 58.1, "s1": 61.9, "s2": 51.3, "s3": 70.7}},
    "door": {"spot": {"same": 86.3, "s1": 66.7, "s2": 75.0, "s3": 100.0},
             "stretch": {"same": 12.8, "s1": 16.8, "s2": 20.2, "s3": 30.0}},
}
ZSUM_EXACT = stats.norm.ppf(0.95) + stats.norm.ppf(0.80)


def welch_df(a, b, k1, k2):
    return (a + b) ** 2 / (a ** 2 / (k1 - 1) + b ** 2 / (k2 - 1))


def solve_split(se2, nu, k1, k2):
    """a + b = se2, welch_df(a, b) = nu 를 만족하는 a = V1/K1 의 두 근."""
    # r = a/(a+b):  nu * (r^2/(k1-1) + (1-r)^2/(k2-1)) = 1
    A = 1 / (k1 - 1) + 1 / (k2 - 1)
    B = -2 / (k2 - 1)
    C = 1 / (k2 - 1) - 1 / nu
    disc = B * B - 4 * A * C
    if disc < 0:
        return []
    roots = sorted({(-B - math.sqrt(disc)) / (2 * A), (-B + math.sqrt(disc)) / (2 * A)})
    return [r * se2 for r in roots if 0 < r < 1]


def main():
    res = {"zsum_exact": ZSUM_EXACT}

    # V1
    tot = sum(N_TRIALS[t][r] for t in N_TRIALS for r in N_TRIALS[t])
    v1 = {"total_trials": tot, "ok_568": tot == 568, "spot_trials": 62 + 88, "spot_150_ok": 62 + 88 == 150}
    pooled = {}
    for t in N_TRIALS:
        for r in N_TRIALS[t]:
            n, rate = N_TRIALS[t][r], RATE[t][r]
            ks = [k for k in range(n + 1) if abs(k / n - rate) < 0.00005 * 1.0001]
            near = min(range(n + 1), key=lambda k: abs(k / n - rate))
            pooled[f"{t}_{r}"] = {"n": n, "rate": rate, "integer_k_matches_4dp": ks,
                                  "nearest_k": near, "nearest_rate": round(near / n, 4),
                                  "trials_per_seed": round(n / N_SEEDS[t][r], 2)}
    v1["pooled"] = pooled
    res["V1"] = v1

    # V2
    res["V2"] = {t: {"gap_from_rates": round(RATE[t]["spot"] - RATE[t]["stretch"], 4), "reported": DIFF[t]}
                 for t in DIFF}
    res["V2"]["zsum"] = {"exact": round(ZSUM_EXACT, 4), "paper": 2.49}

    # V3, V4, V5
    v3, v4, v5 = {}, {}, {}
    for t in ("door", "drawer"):
        k1, k2 = N_SEEDS[t]["spot"], N_SEEDS[t]["stretch"]
        gap = RATE[t]["spot"] - RATE[t]["stretch"]
        m_ind, m_mix = MARGIN[t]
        lo, hi, nu = WELCH_CI[t]
        se_margin = m_ind / ZSUM_EXACT
        se_margin_249 = m_ind / 2.49
        half = (hi - lo) / 2 / 100
        tq = stats.t.ppf(0.975, nu)
        se_ci = half / tq
        v3[t] = {"se_from_margin": round(se_margin, 5), "se_from_margin_249": round(se_margin_249, 5),
                 "se_from_welch_ci": round(se_ci, 5), "rel_diff": round(se_ci / se_margin - 1, 4),
                 "ci_center_pts": round((lo + hi) / 2, 2), "gap_pts": round(100 * gap, 2),
                 "t975": round(tq, 4),
                 "ci_recomputed_pts": [round(100 * (gap - tq * se_margin), 1), round(100 * (gap + tq * se_margin), 1)]}
        # V4: 분산 분해
        se2 = se_margin ** 2
        roots = solve_split(se2, nu, k1, k2)
        sols = []
        for a in roots:
            b = se2 - a
            v_spot, v_str = a * k1, b * k2
            cap_spot = RATE[t]["spot"] * (1 - RATE[t]["spot"]) * k1 / (k1 - 1)
            cap_str = RATE[t]["stretch"] * (1 - RATE[t]["stretch"]) * k2 / (k2 - 1)
            sols.append({"var_seed_spot": round(v_spot, 4), "sd_seed_spot": round(math.sqrt(v_spot), 3),
                         "var_seed_stretch": round(v_str, 4), "sd_seed_stretch": round(math.sqrt(v_str), 3),
                         "max_var_spot_given_mean": round(cap_spot, 4), "max_var_stretch_given_mean": round(cap_str, 4),
                         "feasible": bool(v_spot <= cap_spot and v_str <= cap_str)})
        v4[t] = {"se2": round(se2, 6), "welch_df_reported": nu, "welch_df_max_given_K": k1 + k2 - 2,
                 "df_reported_feasible": bool(nu <= k1 + k2 - 2 + 1e-9), "solutions": sols,
                 "V_equal_var_approx": round(se2 / (1 / k1 + 1 / k2), 4)}
        # V5
        se = se_margin
        w_t = gap / se
        p_welch = 2 * stats.t.sf(abs(w_t), nu)
        p_norm = 2 * stats.norm.sf(abs(w_t))
        t_margin_central = (stats.t.ppf(0.95, nu) + stats.t.ppf(0.80, nu)) * se
        # 비중심 t 근사: 검정력 80%가 되는 비중심 모수
        v5[t] = {"z": round(w_t, 3), "p_welch_t": round(p_welch, 4), "p_normal": round(p_norm, 4),
                 "margin_z_exact": round(ZSUM_EXACT * se, 4), "margin_paper": m_ind,
                 "margin_t_central": round(t_margin_central, 4),
                 "gap": round(gap, 4), "gap_exceeds_margin_z": bool(gap > ZSUM_EXACT * se),
                 "gap_exceeds_margin_t": bool(gap > t_margin_central),
                 "p_at_z249_two_sided": round(2 * stats.norm.sf(2.49), 4),
                 "welch_ci_95_recomputed_pts": v3[t]["ci_recomputed_pts"],
                 "observed_power_alpha10": round(float(stats.norm.sf(stats.norm.ppf(0.95) - w_t)), 3)}
    res["V3"], res["V4"], res["V5"] = v3, v4, v5

    # V6: p = 0.5 혼합 한계에서 b = B/J 와 sigma^2 의 비를 역산 (식 2, rho = 1 가정)
    v6 = {}
    for t in ("door", "drawer"):
        m0, m5 = MARGIN[t]
        q = (m5 / m0) ** 2          # ((1-p) s + b) / (s + b), p = 0.5
        # 0.5 s + b = q (s + b)  → b (1 - q) = s (q - 0.5)
        ratio_b_over_s = (q - 0.5) / (1 - q)
        share_trial = ratio_b_over_s / (1 + ratio_b_over_s)   # b / (s + b)
        margin_p1 = m0 * math.sqrt(share_trial)
        # 이항 잡음 추정(과제를 시도마다 한 번씩 본다고 가정): p(1-p)/n_seed
        bn = []
        for r in ("spot", "stretch"):
            n_s = N_TRIALS[t][r] / N_SEEDS[t][r]
            bn.append(RATE[t][r] * (1 - RATE[t][r]) / n_s)
        v6[t] = {"margin_ratio_sq": round(q, 4), "b_over_sigma2": round(ratio_b_over_s, 3),
                 "trial_noise_share_of_V": round(share_trial, 3), "implied_margin_p1_rho1": round(margin_p1, 4),
                 "binomial_noise_per_seed_spot_stretch": [round(x, 4) for x in bn],
                 "binomial_noise_mean": round(sum(bn) / 2, 4)}
        # 같은 지표를 V4 해에서 비교
        v6[t]["binomial_share_of_V_equal_var"] = round(v6[t]["binomial_noise_mean"] / v4[t]["V_equal_var_approx"], 3)
        # rho < 1 이면 p 공유 이득이 줄어든다: factor = (1 - p rho) s + b
        sig_share = 1 - share_trial
        rho_tab = {}
        for rho in (1.0, 0.8, 0.5, 0.0):
            for p in (0.5, 1.0):
                f = (1 - p * rho) * sig_share + share_trial
                rho_tab[f"rho{rho}_p{p}"] = round(m0 * math.sqrt(f), 3)
        v6[t]["margin_vs_rho"] = rho_tab
    res["V6"] = v6

    # V7: 시도 합산(군집 무시) 검정 vs 시드 단위 Welch
    v7 = {}
    for t in ("door", "drawer"):
        ks = {r: round(RATE[t][r] * N_TRIALS[t][r]) for r in ("spot", "stretch")}
        tab = [[ks["spot"], N_TRIALS[t]["spot"] - ks["spot"]], [ks["stretch"], N_TRIALS[t]["stretch"] - ks["stretch"]]]
        _, p_f = stats.fisher_exact(tab)
        v7[t] = {"successes_rounded": ks, "fisher_p_naive": float(f"{p_f:.3g}"),
                 "welch_p_seed_level": v5[t]["p_welch_t"],
                 "ratio_log10": round(math.log10(v5[t]["p_welch_t"] / p_f), 1)}
    res["V7"] = v7

    # V8: 표 1 순위 뒤집힘
    v8 = {}
    for t in ("door", "drawer"):
        sp = [T1[t]["spot"][k] for k in ("s1", "s2", "s3")]
        st = [T1[t]["stretch"][k] for k in ("s1", "s2", "s3")]
        rev = [(i + 1, j + 1) for i, j in product(range(3), range(3)) if sp[i] < st[j]]
        v8[t] = {"spot_subsets": sp, "stretch_subsets": st, "reversed_pairs_spot_i_stretch_j": rev,
                 "n_pairs": 9, "spot_range": round(max(sp) - min(sp), 1), "stretch_range": round(max(st) - min(st), 1),
                 "mean_subsets_spot": round(float(np.mean(sp)), 2), "mean_subsets_stretch": round(float(np.mean(st)), 2),
                 "table2_spot_pct": round(100 * RATE[t]["spot"], 2), "table2_stretch_pct": round(100 * RATE[t]["stretch"], 2),
                 "same_seed_spot": T1[t]["spot"]["same"], "same_seed_stretch": T1[t]["stretch"]["same"]}
    res["V8"] = v8

    # V9: 필요 시드 수 (식 1, 3). V는 V4 해의 두 로봇 평균, 두 근 평균. rho = 1
    v9 = {}
    for t in ("door", "drawer"):
        vbar = v4[t]["V_equal_var_approx"]
        share_trial = v6[t]["trial_noise_share_of_V"]
        s2, b = vbar * (1 - share_trial), vbar * share_trial
        rows = {}
        for alpha, pw in ((0.10, 0.80), (0.05, 0.80)):
            zs = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(pw)
            for delta in (0.30, 0.20, 0.10, 0.05):
                for p in (0.0, 0.5):
                    k = 2 * zs ** 2 * ((1 - p) * s2 + b) / delta ** 2
                    rows[f"alpha{alpha}_delta{delta}_p{p}"] = math.ceil(k)
        v9[t] = {"V_mean_per_seed": round(vbar, 4), "sigma2_seed": round(s2, 4), "B_over_J": round(b, 4), "K_per_robot": rows}
    res["V9"] = v9

    # V10: 시드 6-8개에서 추정한 분산의 불확실성 (카이제곱 95% 구간 배수). 필요 시드 수 K 는 V 에 비례한다.
    v10 = {}
    for K in (6, 7, 8, 30):
        nu = K - 1
        lo = nu / stats.chi2.ppf(0.975, nu)
        hi = nu / stats.chi2.ppf(0.025, nu)
        v10[f"K{K}"] = {"df": nu, "variance_multiplier_95": [round(lo, 2), round(hi, 2)],
                        "sd_multiplier_95": [round(math.sqrt(lo), 2), round(math.sqrt(hi), 2)]}
    res["V10"] = v10

    # 설정 공간 크기 (3 x 4 격자, 빈칸 6): 빈칸 위치 C(12, 6), 칸마다 문/서랍 둘 중 하나라면 2^6
    res["config_space"] = {"empty_positions_C12_6": math.comb(12, 6), "door_or_drawer_labelings_2pow6": 2 ** 6,
                           "functional_layouts": math.comb(12, 6) * 2 ** 6,
                           "bernoulli_empty_count_sd": round(math.sqrt(12 * 0.25), 3),
                           "bernoulli_p_exactly_6_empty": round(float(stats.binom.pmf(6, 12, 0.5)), 4)}

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
