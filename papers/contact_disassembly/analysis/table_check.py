"""CONTACT 표 재계산 (CPU, numpy·scipy, 1초 미만). 실행: python3 analysis/table_check.py

- T1: Table I(sim) 평균·Fig. 1 평균(60.5/61.2/65.0, 52.0/37.0/68.0) 재계산, 정수 성공 횟수 복원(1500회 → k = 15p)
- T2: 칸별 Wilson 95% CI (n = 1500 가정 = 논문의 i.i.d. 가정; n = 150 = 3 seed × 50 초기조건; 실물 n = 20)
- T3: 시뮬 과제별 Vision vs TacFF / TacRGB 대조 z 검정(n 가정별) + Holm, CMH(과제 층화) MH 오즈비
- T4: 실물(Table II) 과제별 Fisher 정확검정 + Holm, 100회 합산 Fisher, CMH(R1–R5 층화)
- T5: Table IV(조명)·Table V(결합) Fisher, 교란 하락폭의 Newcombe 차이 CI
- T6: seed 수준 한계: 3 seed에서 t검정이 유의해지는 seed 표준편차 상한(독립 / 짝짓기)
- T7: 검정력: 실물 칸당 필요한 시행 수(두 비율, α=0.05 양측, 검정력 0.8)
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "table_check_results.json"

SIM_TASKS = ["S1", "S2", "S3", "S4", "S5"]
REAL_TASKS = ["R1", "R2", "R3", "R4", "R5"]
# Table I (sim, % 값; 1500 rollouts)
SIM = {
    "Vision": [81.7, 36.9, 50.0, 74.6, 59.5],
    "TacRGB": [84.4, 37.2, 51.4, 75.1, 57.8],
    "TacFF": [86.5, 44.3, 53.5, 78.8, 61.9],
}
# Table II (real, 20회): 성공 횟수 = % / 5
REAL = {
    "Vision": [16, 11, 8, 14, 3],
    "TacRGB": [18, 6, 3, 1, 9],
    "TacFF": [19, 14, 9, 15, 11],
}
# Table IV (조명): R1 normal = Table II, R5 normal = Table II
DIM = {
    "R1": {"Vision": (5, 16), "TacRGB": (2, 18), "TacFF": (5, 19)},  # (dim, normal)
    "R5": {"Vision": (0, 3), "TacRGB": (8, 9), "TacFF": (11, 11)},
}
# Table V (결합): R3, R5 (20회)
COMBO = {"R3": {"Vision": 8, "TacRGB": 3, "TacFF": 9, "Combo": 0},
         "R5": {"Vision": 3, "TacRGB": 9, "TacFF": 11, "Combo": 4}}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def newcombe_diff(k1, n1, k2, n2):
    """차이 p1 - p2의 Newcombe(1998) 점수 구간."""
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def two_prop_z(k1, n1, k2, n2):
    """두 비율의 풀링 z 검정(양측)."""
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return z, 2 * stats.norm.sf(abs(z))


def fisher(k1, n1, k2, n2):
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(stats.fisher_exact(table, alternative="two-sided")[1])


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def cmh(strata):
    """strata: list of (a, b, c, d) with rows = [exposed success a, failure b] / [control c, d]."""
    num = den = 0.0
    sa = ea = va = 0.0
    for a, b, c, d in strata:
        n = a + b + c + d
        if n == 0:
            continue
        r1, r0 = a + b, c + d
        c1, c0 = a + c, b + d
        num += a * d / n
        den += b * c / n
        ea += r1 * c1 / n
        va += r1 * r0 * c1 * c0 / (n * n * (n - 1))
        sa += a
    chi2 = (abs(sa - ea)) ** 2 / va
    return {"MH_OR": num / den, "chi2": chi2, "p": float(stats.chi2.sf(chi2, 1))}


def seed_breakeven(gap_pp, n_seeds=3):
    """3 seed에서 t검정이 p<0.05가 되는 seed 표준편차 상한(pp). 독립 두 표본 / 짝지은 표본."""
    t_ind = stats.t.ppf(0.975, 2 * n_seeds - 2)
    t_pair = stats.t.ppf(0.975, n_seeds - 1)
    sd_ind = gap_pp / (t_ind * math.sqrt(2 / n_seeds))
    sd_pair = gap_pp / (t_pair / math.sqrt(n_seeds))
    return {"t_crit_indep": t_ind, "sd_max_indep_pp": sd_ind,
            "t_crit_paired": t_pair, "sd_max_paired_pp": sd_pair}


def n_per_arm(p1, p2, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return math.ceil(num ** 2 / (p1 - p2) ** 2)


def counts_from_pct(p, n):
    """Table I의 % 값 p를 n 시행에서 가능한 정수 횟수 범위로 바꾼다(반올림 한 칸 폭)."""
    lo = math.ceil((p / 100 - 0.0005) * n)
    hi = math.floor((p / 100 + 0.0005) * n)
    return lo, hi


def main():
    R = {}

    # ---- T1: 평균 재계산 (Fig. 1 헤드라인과 비교)
    R["T1_means"] = {m: {"sim_mean": round(float(np.mean(v)), 2),
                         "real_mean_from_table_II": round(float(np.mean(REAL[m]) * 5), 2)}
                     for m, v in SIM.items()}
    R["T1_fig1_check"] = {
        "sim_vision_60.5": round(float(np.mean(SIM["Vision"])), 1) == 60.5,
        "sim_tacrgb_61.2": round(float(np.mean(SIM["TacRGB"])), 1) == 61.2,
        "sim_tacff_65.0": round(float(np.mean(SIM["TacFF"])), 1) == 65.0,
        "real_vision_52.0": round(float(np.mean(REAL["Vision"]) * 5), 1) == 52.0,
        "real_tacrgb_37.0": round(float(np.mean(REAL["TacRGB"]) * 5), 1) == 37.0,
        "real_tacff_68.0": round(float(np.mean(REAL["TacFF"]) * 5), 1) == 68.0,
    }
    # TacFF가 10개 과제 모두에서 칸 최댓값인지(동점 포함)
    best = []
    for i, t in enumerate(SIM_TASKS + REAL_TASKS):
        if i < 5:
            vals = {m: SIM[m][i] for m in SIM}
        else:
            vals = {m: REAL[m][i - 5] for m in REAL}
        best.append({"task": t, "TacFF_is_max": vals["TacFF"] == max(vals.values()),
                     "ranking": sorted(vals, key=lambda k: -vals[k])})
    R["T1_best_per_task"] = best

    # ---- T1b: 시뮬 성공 횟수 범위 (1500회)
    R["T1b_sim_count_ranges"] = {
        m: [list(counts_from_pct(p, 1500)) for p in v] for m, v in SIM.items()}

    # ---- T2: Wilson CI, n = 1500 / 150 (3 seed × 50 초기조건, checkpoint 중복 제거 가정) / 50
    wil = {}
    for m, v in SIM.items():
        wil[m] = {}
        for i, p in enumerate(v):
            k = round(p * 15)  # 1500회 기준 정수
            lo, hi = wilson(k, 1500)
            k150 = round(p * 1.5)
            lo150, hi150 = wilson(k150, 150)
            wil[m][SIM_TASKS[i]] = {"p": p, "wilson1500": [round(100 * lo, 1), round(100 * hi, 1)],
                                    "wilson150": [round(100 * lo150, 1), round(100 * hi150, 1)]}
    R["T2_sim_wilson"] = wil
    real_wil = {}
    for m, v in REAL.items():
        real_wil[m] = {REAL_TASKS[i]: {"k": k, "n": 20, "p": 100 * k / 20,
                                       "wilson": [round(100 * wilson(k, 20)[0], 1),
                                                  round(100 * wilson(k, 20)[1], 1)]}
                       for i, k in enumerate(v)}
    R["T2_real_wilson"] = real_wil

    # ---- T3: 시뮬 과제별 대조. n 가정 3가지(1500, 150). Holm은 과제 5개 기준.
    sim_tests = {}
    for comp, (a, b) in {"TacFF_vs_Vision": ("TacFF", "Vision"),
                         "TacRGB_vs_Vision": ("TacRGB", "Vision"),
                         "TacFF_vs_TacRGB": ("TacFF", "TacRGB")}.items():
        for n_lbl, n in (("n1500", 1500), ("n150", 150)):
            pvals, rows = [], []
            for i in range(5):
                ka, kb = round(SIM[a][i] * n / 100), round(SIM[b][i] * n / 100)
                z, p = two_prop_z(ka, n, kb, n)
                pvals.append(p)
                d, lo, hi = newcombe_diff(ka, n, kb, n)
                rows.append({"task": SIM_TASKS[i], "diff_pp": round(100 * d, 1),
                             "newcombe_pp": [round(100 * lo, 1), round(100 * hi, 1)],
                             "z": round(z, 2), "p": float(p)})
            adj = holm(pvals)
            for r, h in zip(rows, adj):
                r["p_holm"] = float(h)
            strata = []
            for i in range(5):
                ka, kb = round(SIM[a][i] * n / 100), round(SIM[b][i] * n / 100)
                strata.append((ka, n - ka, kb, n - kb))
            sim_tests[f"{comp}_{n_lbl}"] = {"rows": rows, "CMH": cmh(strata)}
    R["T3_sim_tests"] = sim_tests

    # 시드 수준 한계 (T6 성격): 3 seed에서 대조가 유의하려면 seed 표준편차가 얼마 이하여야 하나
    gaps = {}
    for comp, (a, b) in {"TacFF_vs_Vision": ("TacFF", "Vision"),
                         "TacRGB_vs_Vision": ("TacRGB", "Vision"),
                         "TacFF_vs_TacRGB": ("TacFF", "TacRGB")}.items():
        gaps[comp] = {SIM_TASKS[i]: {"gap_pp": round(SIM[a][i] - SIM[b][i], 1),
                                     **{k: round(v, 2) for k, v in seed_breakeven(abs(SIM[a][i] - SIM[b][i])).items()}}
                      for i in range(5)}
    R["T6_seed_breakeven_sim"] = gaps

    # ---- T4: 실물 과제별 Fisher + Holm, 100회 합산, CMH
    real_tests = {}
    for comp, (a, b) in {"TacFF_vs_Vision": ("TacFF", "Vision"),
                         "TacRGB_vs_Vision": ("TacRGB", "Vision"),
                         "TacFF_vs_TacRGB": ("TacFF", "TacRGB")}.items():
        pv = [fisher(REAL[a][i], 20, REAL[b][i], 20) for i in range(5)]
        adj = holm(pv)
        rows = []
        for i in range(5):
            d, lo, hi = newcombe_diff(REAL[a][i], 20, REAL[b][i], 20)
            rows.append({"task": REAL_TASKS[i], "diff_pp": round(100 * d, 1),
                         "newcombe_pp": [round(100 * lo, 1), round(100 * hi, 1)],
                         "p_fisher": pv[i], "p_holm": adj[i]})
        strata = [(REAL[a][i], 20 - REAL[a][i], REAL[b][i], 20 - REAL[b][i]) for i in range(5)]
        ka, kb = sum(REAL[a]), sum(REAL[b])
        pooled_p = fisher(ka, 100, kb, 100)
        real_tests[comp] = {"rows": rows, "CMH": cmh(strata),
                            "pooled_100": {"k_a": ka, "k_b": kb, "p_fisher": pooled_p,
                                           "diff_pp": ka - kb,
                                           "newcombe_pp": [round(100 * x, 1) for x in newcombe_diff(ka, 100, kb, 100)[1:]]}}
    R["T4_real_tests"] = real_tests

    # ---- T5: Table IV 조명, Table V 결합
    lighting = {}
    for task, d in DIM.items():
        lighting[task] = {}
        for m, (kd, kn) in d.items():
            dd, lo, hi = newcombe_diff(kn, 20, kd, 20)
            lighting[task][m] = {"normal": kn, "dim": kd, "drop_pp": round(100 * dd, 1),
                                 "drop_newcombe_pp": [round(100 * lo, 1), round(100 * hi, 1)],
                                 "p_fisher_normal_vs_dim": fisher(kn, 20, kd, 20)}
    R["T5_lighting"] = lighting
    combo = {}
    for task, c in COMBO.items():
        combo[task] = {}
        for other in ("Vision", "TacRGB", "TacFF"):
            combo[task][f"Combo_vs_{other}"] = {
                "diff_pp": round(100 * (c["Combo"] - c[other]) / 20, 1),
                "p_fisher": fisher(c["Combo"], 20, c[other], 20)}
    R["T5_combo"] = combo

    # ---- T7: 검정력 (실물 칸당 필요한 시행 수)
    R["T7_power"] = {
        "real_pooled_V_vs_F_52_68": n_per_arm(0.52, 0.68),
        "real_task_15pp_at_0.55_0.70": n_per_arm(0.55, 0.70),
        "real_task_30pp_at_0.20_0.50": n_per_arm(0.20, 0.50),
        "sim_S2_7.4pp_at_0.369_0.443": n_per_arm(0.369, 0.443),
        "sim_S5_2.4pp_at_0.595_0.619": n_per_arm(0.595, 0.619),
    }

    OUT.write_text(json.dumps(R, indent=2, ensure_ascii=False, default=float))
    print("wrote", OUT)
    print("means", R["T1_means"])
    print("fig1", R["T1_fig1_check"])
    for c in ("TacFF_vs_Vision", "TacRGB_vs_Vision", "TacFF_vs_TacRGB"):
        print("real", c, "pooled p=%.4f" % R["T4_real_tests"][c]["pooled_100"]["p_fisher"],
              "CMH p=%.4g" % R["T4_real_tests"][c]["CMH"]["p"])
        print("   per-task p_fisher", [round(r["p_fisher"], 4) for r in R["T4_real_tests"][c]["rows"]])
    for k, v in R["T3_sim_tests"].items():
        print("sim", k, "CMH p=%.3g" % v["CMH"]["p"], "OR=%.2f" % v["CMH"]["MH_OR"],
              "per-task p=", [f"{r['p']:.3g}" for r in v["rows"]])
    print("power", R["T7_power"])


if __name__ == "__main__":
    main()
