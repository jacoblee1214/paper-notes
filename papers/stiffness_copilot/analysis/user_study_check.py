"""Stiffness Copilot 사용자 연구(n=18) 통계 재계산 (CPU, numpy/scipy 1스레드, 수 초).

입력은 논문 Table I(평균·SD·검정 통계량)과 Fig. 3 성공률 막대의 참가자별 성공 횟수 분포뿐이다.
원자료(참가자 단위 시행 기록)는 공개되지 않았다. 웹사이트의 static/data는 조건·과제당 대표 기록 1개뿐이라 통계에 쓰지 않았다.

실행: python3 analysis/user_study_check.py  → analysis/user_study_check_results.json
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "user_study_check_results.json"
N = 18  # 참가자 수 (Sec. IV-F)

# ---- Table I: (mean, SD) ----
T1 = {
    "force": {"Vase": {"Low": (60.58, 29.23), "SC": (59.28, 27.04), "High": (117.46, 42.87)},
              "Peg": {"Low": (49.00, 34.25), "SC": (33.41, 11.67), "High": (80.60, 54.44)},
              "Door": {"Low": (40.96, 21.97), "SC": (63.63, 29.82), "High": (101.42, 58.96)}},
    "time": {"Vase": {"Low": (30.96, 8.12), "SC": (25.31, 7.54), "High": (33.62, 11.57)},
             "Peg": {"Low": (35.35, 13.59), "SC": (24.22, 9.58), "High": (21.95, 6.31)},
             "Door": {"Low": (24.89, 11.68), "SC": (15.09, 6.67), "High": (18.63, 9.67)}},
    "success": {"Vase": {"Low": (0.76, 0.36), "SC": (0.87, 0.17), "High": (0.24, 0.28)},
                "Peg": {"Low": (0.67, 0.40), "SC": (0.81, 0.31), "High": (0.70, 0.30)},
                "Door": {"Low": (0.83, 0.26), "SC": (0.70, 0.28), "High": (0.31, 0.33)}},
}
# 과제별 완료 시간 분석에 포함된 참가자 수 (Fig. 3 "n=")
N_TIME = {"Vase": 9, "Peg": 14, "Door": 10}

# ---- Fig. 3 성공률 막대: 참가자 수 by 성공 횟수 k=0..3 (읽은 값, 각 막대 합 18) ----
COUNTS = {
    "Vase": {"Low": [2, 2, 3, 11], "SC": [0, 0, 7, 11], "High": [8, 8, 1, 1]},
    "Peg": {"Low": [3, 3, 3, 9], "SC": [1, 2, 3, 12], "High": [1, 3, 7, 7]},
    "Door": {"Low": [0, 3, 3, 12], "SC": [0, 5, 6, 7], "High": [8, 4, 5, 1]},
}

# ---- Table I 검정 통계량 (F, df1, df2, reported p-level, ηp²) ----
F_TESTS = [
    ("Vase force", 17.15, 34, 0.502), ("Peg force", 6.99, 34, 0.291), ("Door force", 11.89, 34, 0.412),
    ("Vase time", 4.29, 16, 0.349), ("Peg time", 12.01, 26, 0.480), ("Door time", 4.99, 18, 0.357),
    ("Control", 10.16, 34, 0.374), ("Trust", 7.22, 34, 0.298), ("Predictability", 11.31, 34, 0.400),
    ("Fluency", 8.44, 34, 0.332), ("NASA-TLX", 11.36, 34, 0.401),
]
# Friedman (성공률): (χ², 보고된 W)
FRIEDMAN = {"Vase": (20.13, 0.559), "Peg": (3.21, 0.089), "Door": (19.77, 0.549)}

# ---- 그림 3 괄호 표기 (읽은 값; 괄호 위치 판독은 해상도 한계로 확실하지 않음) ----
FIG_LABELS = {
    ("Vase", "force", "Low", "High"): ("***", 1.06), ("Vase", "force", "SC", "High"): ("***", 1.13),
    ("Peg", "force", "SC", "High"): ("**", 0.86),
    ("Peg", "time", "Low", "High"): ("**", 1.15), ("Peg", "time", "Low", "SC"): ("*", 0.88),
    ("Door", "force", "Low", "High"): ("**", 0.96), ("Door", "force", "Low", "SC"): ("**", 0.84),
    ("Door", "force", "SC", "High"): ("*", 0.61),
    ("Door", "time", "Low", "SC"): ("*", 1.04),
    ("Vase", "success", "Low", "High"): ("**", 0.94), ("Vase", "success", "SC", "High"): ("**", 0.95),  # r (Wilcoxon)
    ("Door", "success", "Low", "High"): ("**", 0.97), ("Door", "success", "SC", "High"): ("**", 0.82),  # r
}
# 본문 보고 (Sec. IV-G): Vase time Low–SC p=.091, d=.72
TEXT_LABELS = {("Vase", "time", "Low", "SC"): ("p=.091", 0.72)}


def label(table, *key):
    """괄호 표기는 쌍의 순서와 상관없이 찾는다"""
    task, metric, a, b = key
    return table.get((task, metric, a, b)) or table.get((task, metric, b, a))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def rates_from_counts(counts):
    """참가자별 성공률 분포(0, 1/3, 2/3, 1) 펼치기"""
    vals = []
    for k, c in enumerate(counts):
        vals += [k / 3] * c
    return np.array(vals)


def main():
    res = {"n_participants": N}

    # 1. F 검정: p와 ηp² 재계산
    rows = []
    for name, F, df2, eta_rep in F_TESTS:
        df1 = 2
        p = stats.f.sf(F, df1, df2)
        eta = F * df1 / (F * df1 + df2)
        rows.append({"test": name, "F": F, "df": [df1, df2], "p_recomputed": float(p),
                     "eta_p2_recomputed": round(eta, 3), "eta_p2_reported": eta_rep,
                     "df2_implies_n": df2 // 2 + 1})
    res["anova_check"] = rows

    # 2. Friedman: W = χ²/(n(k-1))
    fr = {}
    for task, (chi2, w_rep) in FRIEDMAN.items():
        fr[task] = {"chi2": chi2, "p_recomputed": float(stats.chi2.sf(chi2, 2)),
                    "W_recomputed": round(chi2 / (N * 2), 3), "W_reported": w_rep}
    res["friedman_check"] = fr

    # 3. 성공률: 막대 횟수로 평균·SD 재현 (Table I과 대조)
    succ = {}
    for task, conds in COUNTS.items():
        succ[task] = {}
        for cond, counts in conds.items():
            assert sum(counts) == N, (task, cond, sum(counts))
            r = rates_from_counts(counts)
            mean, sd = r.mean(), r.std(ddof=1)
            rep_mean, rep_sd = T1["success"][task][cond]
            k_succ = int(round(mean * N * 3))
            lo, hi = wilson(k_succ, N * 3)
            succ[task][cond] = {
                "counts_k0_to_3": counts, "successes_of_54": k_succ,
                "mean_recomputed": round(mean, 3), "mean_reported": rep_mean,
                "sd_recomputed": round(sd, 3), "sd_reported": rep_sd,
                "wilson95_pooled_trials": [round(lo, 3), round(hi, 3)],
            }
        # 조건 간 비교 (독립 근사): Mann-Whitney on 참가자 성공률, Holm within task
        pairs = [("SC", "High"), ("SC", "Low"), ("Low", "High")]
        ps = []
        out = []
        for a, b in pairs:
            ra, rb = rates_from_counts(conds[a]), rates_from_counts(conds[b])
            u = stats.mannwhitneyu(ra, rb, alternative="two-sided")
            ps.append(u.pvalue)
            out.append({"pair": f"{a}-{b}", "MWU_p_unpaired": float(u.pvalue),
                        "fig_label": label(FIG_LABELS, task, "success", a, b)})
        for o, adj in zip(out, holm(ps)):
            o["holm_p"] = adj
        succ[task]["pairwise"] = out
    res["success"] = succ

    # 4. 연속 측정: Welch(독립 근사)와 pooled d, Holm within metric·task
    cont = {}
    for metric in ("force", "time"):
        cont[metric] = {}
        for task in ("Vase", "Peg", "Door"):
            n_here = N if metric == "force" else N_TIME[task]
            conds = T1[metric][task]
            pairs = [("SC", "High"), ("SC", "Low"), ("Low", "High")]
            ps, rows = [], []
            for a, b in pairs:
                ma, sa = conds[a]
                mb, sb = conds[b]
                t, p = stats.ttest_ind_from_stats(ma, sa, n_here, mb, sb, n_here, equal_var=False)
                sp = np.sqrt((sa ** 2 + sb ** 2) / 2)
                d = (ma - mb) / sp
                se = np.sqrt(sa ** 2 / n_here + sb ** 2 / n_here)
                df = (sa ** 2 / n_here + sb ** 2 / n_here) ** 2 / (
                    (sa ** 2 / n_here) ** 2 / (n_here - 1) + (sb ** 2 / n_here) ** 2 / (n_here - 1))
                tcrit = stats.t.ppf(0.975, df)
                ps.append(p)
                rows.append({"pair": f"{a}-{b}", "diff": round(ma - mb, 2),
                             "diff_pct_of_b": round(100 * (ma - mb) / mb, 1),
                             "ci95_diff_welch": [round(ma - mb - tcrit * se, 2), round(ma - mb + tcrit * se, 2)],
                             "p_welch_unpaired": float(p), "d_pooled_recomputed": round(d, 2),
                             "fig_label": label(FIG_LABELS, task, metric, a, b),
                             "text_label": label(TEXT_LABELS, task, metric, a, b)})
            for r_, adj in zip(rows, holm(ps)):
                r_["holm_p_welch"] = adj
            cont[metric][task] = {"n": n_here, "pairs": rows}
    res["continuous"] = cont

    # 5. 주관 지표 (n=18): d 재계산 vs 그림 3 표기
    subj = {}
    subj_means = {"Control": {"Low": (4.64, 1.10), "SC": (5.67, 0.89), "High": (4.29, 1.34)},
                  "Trust": {"Low": (4.03, 1.14), "SC": (5.01, 1.27), "High": (4.01, 1.48)},
                  "Predictability": {"Low": (4.51, 0.94), "SC": (5.51, 0.79), "High": (4.21, 1.38)},
                  "Fluency": {"Low": (4.21, 1.34), "SC": (5.28, 1.23), "High": (3.61, 1.59)},
                  "NASA-TLX": {"Low": (49.33, 15.89), "SC": (38.73, 14.15), "High": (52.43, 16.00)}}
    subj_fig = {("Control", "Low", "SC"): ("**", 0.87), ("Control", "SC", "High"): ("***", 1.20),
                ("Trust", "Low", "SC"): ("**", 0.89), ("Trust", "SC", "High"): ("**", 0.85),
                ("Predictability", "Low", "SC"): ("***", 1.07), ("Predictability", "SC", "High"): ("***", 1.10),
                ("Fluency", "Low", "SC"): ("*", 0.63), ("Fluency", "SC", "High"): ("**", 1.15),
                ("NASA-TLX", "Low", "SC"): ("**", 0.77), ("NASA-TLX", "SC", "High"): ("***", 1.38)}
    for name, conds in subj_means.items():
        rows = []
        for a, b in [("Low", "SC"), ("SC", "High"), ("Low", "High")]:
            ma, sa = conds[a]
            mb, sb = conds[b]
            t, p = stats.ttest_ind_from_stats(ma, sa, N, mb, sb, N, equal_var=False)
            d = (ma - mb) / np.sqrt((sa ** 2 + sb ** 2) / 2)
            rows.append({"pair": f"{a}-{b}", "d_pooled_recomputed": round(d, 2),
                         "p_welch_unpaired": float(p), "fig_label": subj_fig.get((name, a, b)) or subj_fig.get((name, b, a))})
        subj[name] = rows
    res["subjective"] = subj

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("F-tests (p, ηp²):")
    for r in res["anova_check"]:
        print(f"  {r['test']:<16} p={r['p_recomputed']:.2g} ηp²={r['eta_p2_recomputed']} (rep {r['eta_p2_reported']}) n={r['df2_implies_n']}")
    for task, v in fr.items():
        print(f"Friedman {task}: W={v['W_recomputed']} (rep {v['W_reported']}), p={v['p_recomputed']:.2g}")
    for task in COUNTS:
        for cond in ("Low", "SC", "High"):
            s = succ[task][cond]
            print(f"success {task:<5}{cond:<5} k={s['successes_of_54']}/54 mean={s['mean_recomputed']} (rep {s['mean_reported']}) sd={s['sd_recomputed']} (rep {s['sd_reported']})")
    for metric in ("force", "time"):
        for task in ("Vase", "Peg", "Door"):
            for r_ in cont[metric][task]["pairs"]:
                print(f"{metric:<6}{task:<5}{r_['pair']:<8} diff={r_['diff']:>7} p_w={r_['p_welch_unpaired']:.3g} holm={r_['holm_p_welch']:.3g} d={r_['d_pooled_recomputed']} fig={r_['fig_label']}")


if __name__ == "__main__":
    main()
