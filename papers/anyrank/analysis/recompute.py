"""AnyRank 보고 수치 재계산 (CPU, 수 초). 실행: python3 analysis/recompute.py
입력: repo/data/ 아래 공개 CSV (공식 저장소에 포함된 per-trial 데이터). 공식 코드는 실행하지 않는다.
출력: analysis/recompute_results.json
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "repo" / "data" / "roboarena"
OUT = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def holm(ps, alpha):
    order = np.argsort(ps)
    rej = np.zeros(len(ps), bool)
    for r, i in enumerate(order):
        if ps[i] <= alpha / (len(ps) - r):
            rej[i] = True
        else:
            break
    return rej


# V1 RoboArena per-policy 평균과 표 3, 표 1 대조
d = pd.read_csv(DATA / "data2" / "per_trial_progress_data.csv")
tbl3 = {"pi0_fast_droid": .455, "paligemma_fast_droid": .432, "paligemma_fast_specialist_droid": .429,
        "paligemma_vq_droid": .418, "paligemma_diffusion_droid": .404, "pi0_droid": .351,
        "paligemma_binning_droid": .047, "dreaming_zebra": .606, "floppy_bagel_195k_temp1p0": .479,
        "floppy_bagel_525k_temp1p0": .380, "dam": .337, "floppy_bagel_420k": .326,
        "floppy_bagel_285k_temp1p0": .442, "floppy_bagel_90k_temp1p0": .291}
tbl1 = {"pi0_fast_droid": .459, "paligemma_fast_droid": .430, "paligemma_fast_specialist_droid": .428,
        "paligemma_vq_droid": .419, "paligemma_diffusion_droid": .403, "pi0_droid": .350,
        "paligemma_binning_droid": .051}
v1 = {}
for k in tbl3:
    x = d[k].dropna().values
    v1[k] = dict(n=len(x), mean=round(x.mean(), 4), sd=round(x.std(ddof=1), 4),
                 table3=tbl3[k], diff3=round(round(x.mean(), 3) - tbl3[k], 4),
                 table1=tbl1.get(k), diff1=None if k not in tbl1 else round(x.mean() - tbl1[k], 4))
OUT["V1_policy_means"] = v1
OUT["V1_total_trials_14_policies"] = int(d.count().sum())

# V2 사전(world model) 상관
wm = pd.read_csv(DATA / "wm_evals" / "evaluations.csv").groupby("wm_id").score.mean()
AF = ["pi0_fast_droid", "paligemma_fast_droid", "paligemma_fast_specialist_droid",
      "paligemma_vq_droid", "paligemma_diffusion_droid", "pi0_droid"]
real = {k: d[k].mean() for k in AF + ["paligemma_binning_droid"]}
OUT["V2_wm_real_r"] = {
    "A-F": round(float(np.corrcoef([wm[k] for k in AF], [real[k] for k in AF])[0, 1]), 3),
    "A-G": round(float(np.corrcoef([wm[k] for k in real], [real[k] for k in real])[0, 1]), 3),
    "spearman_A-F": round(float(stats.spearmanr([wm[k] for k in AF], [real[k] for k in AF])[0]), 3),
    "spearman_A-G": round(float(stats.spearmanr([wm[k] for k in real], [real[k] for k in real])[0]), 3),
    "paper": {"A-F": 0.05}}

# V3 식 (5) 위험 배분: 7정책 WM prior, gamma별
pols = AF + ["paligemma_binning_droid"]
pairs = [(i, j) for i in range(7) for j in range(i + 1, 7)]
dd = np.array([abs(wm[pols[i]] - wm[pols[j]]) for i, j in pairs])
v3 = {}
for g in [0, 1, 10]:
    a = 0.1 * np.exp(g * dd) / np.exp(g * dd).sum()
    v3[str(g)] = dict(alpha_max=round(a.max(), 4), alpha_min=round(a.min(), 4),
                      ratio=round(a.max() / a.min(), 2))
OUT["V3_alpha_allocation_7pol_wm"] = v3
OUT["V3_K"] = len(pairs)
# 식 (4) 대 코드의 표본 선택 가중치 비 (alpha 0.1 대 0.01)
OUT["V3_sampling_weight_ratio_alpha_0.1_vs_0.01"] = dict(
    paper_exp_alpha=round(math.exp(0.1) / math.exp(0.01), 3),
    code_exp2alpha_minus1=round((math.exp(0.2) - 1) / (math.exp(0.02) - 1), 2))

# V4 고정 표본 Welch + Holm (자체 계산, AnyRank 아님)
def fixed_batch(names, n_each, alpha=0.1):
    xs = {k: d[k].dropna().values[:n_each] for k in names}
    pr = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    ps = np.array([stats.ttest_ind(xs[a], xs[b], equal_var=False).pvalue for a, b in pr])
    rej = holm(ps, alpha)
    return dict(n_each=n_each, K=len(pr), decided=int(rej.sum()),
                undecided=[f"{a[:10]}|{b[:10]}" for (a, b), r in zip(pr, rej) if not r])

four = ["pi0_fast_droid", "paligemma_diffusion_droid", "pi0_droid", "paligemma_binning_droid"]
seven = AF + ["paligemma_binning_droid"]
OUT["V4_fixed_welch_holm"] = {
    "4pol_300each": fixed_batch(four, 300), "4pol_all": fixed_batch(four, 10**6),
    "7pol_730each": fixed_batch(seven, 730), "7pol_all": fixed_batch(seven, 10**6)}

# V5 표 2 산술
t2 = {5: (1102, 9, 1170, 9, 10), 10: (2480, 44, 2855, 43, 45), 15: (6705, 98, 6742, 93, 105), 20: (9075, 182, 9220, 175, 190)}
OUT["V5_table2_N500"] = {M: dict(K=comb, K_check=M * (M - 1) // 2, reduction_pct=round(100 * (1 - a / b), 1),
                                 rej_any=ra, rej_bon=rb)
                         for M, (a, ra, b, rb, comb) in t2.items()}
t2b = {5: (1554, 1683), 10: (3259, 5191), 15: (38000, 42000), 20: (57000, 63000)}
OUT["V5_table2_N5000_reduction_pct(Bonferroni 값은 하한 '>' 표기: 최소 감소율)"] = {
    M: round(100 * (1 - a / b), 1) for M, (a, b) in t2b.items()}

# V6 부록 F 수치
OUT["V6_binom_5of6_vs_0.05"] = float(stats.binom.sf(4, 6, 0.05))
OUT["V6_HO_I_error_9of200_wilson"] = wilson(9, 200)
# 반례: Ber(0.3), Ber(0.7), {0.8,0.9} 균등
rng = np.random.default_rng(0)
X1 = rng.binomial(1, .3, 10**6); X2 = rng.binomial(1, .7, 10**6); X3 = rng.choice([.8, .9], 10**6)
OUT["V6_counterexample_win_prob"] = dict(
    p2_beats_p3=round(float((X2 > X3).mean()), 3), p2_beats_p1=round(float((X2 > X1).mean()), 3),
    p2_ties_p1=round(float((X2 == X1).mean()), 3), p3_beats_p1=round(float((X3 > X1).mean()), 3))

# V7 연구 질문 3 수치
OUT["V7_R3"] = dict(anyrank_204_vs_195_rel_pct=round(100 * (204 - 195) / 204, 1), vs_188=round(100 * (204 - 188) / 204, 1))

# V8 표 3 비교 수 합
OUT["V8_table3_comparisons_sum"] = sum([58, 564, 739, 8, 33, 762, 751, 953, 951, 60, 29, 1009, 12, 10, 629])
pref = pd.read_csv(DATA / "data2" / "policy_preferences.csv")
OUT["V8_pref_rows_per_policy_data2"] = {k: int(((pref.A == k) | (pref.B == k)).sum()) for k in AF}

# V9 LBM 실로봇 (공식 저장소 per-trial CSV)
lbm = ROOT / "repo" / "lbm_real_data"
v9 = {}
for ed in sorted(lbm.iterdir()):
    if not ed.is_dir():
        continue
    arms = {}
    for f in sorted(ed.glob("*.csv")):
        s = pd.read_csv(f)["success"].astype(float).values
        arms[f.stem.replace("policy_", "").replace("_data", "")] = s
    names = list(arms)
    pr = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    ps = []
    for a, b in pr:
        ka, kb, n = int(arms[a].sum()), int(arms[b].sum()), len(arms[a])
        ps.append(stats.fisher_exact([[ka, n - ka], [kb, n - kb]])[1])
    ps = np.array(ps)
    v9[ed.name] = dict(
        n=[len(arms[a]) for a in names], rates={a: round(float(arms[a].mean()), 3) for a in names},
        wilson={a: wilson(int(arms[a].sum()), len(arms[a])) for a in names},
        fisher_p={f"{a}|{b}": round(float(p), 4) for (a, b), p in zip(pr, ps)},
        holm_decided=int(holm(ps, 0.1).sum()), K=len(pr))
OUT["V9_lbm_fixed_batch"] = v9

# V10 DROID-task 사전 상관 (공식 저장소 CSV)
irom = ROOT / "repo" / "data" / "irom_data" / "scores_for_ranking"
v10 = {}
for task in ["towel", "carrot"]:
    real_m, wm_cols = {}, {}
    for f in sorted((irom / task / "real").glob("*.csv")):
        r = pd.read_csv(f)
        real_m[f.stem] = float(r["success"].astype(float).mean())
    for f in sorted((irom / task / "wm").glob("*.csv")):
        w = pd.read_csv(f)
        wm_cols[f.stem] = {c: float(w[c].dropna().astype(float).mean()) for c in w.columns[1:5]}
    v10[task] = dict(real_mean=real_m, wm_means=wm_cols, n_real={f.stem: int(len(pd.read_csv(f))) for f in (irom / task / "real").glob("*.csv")})
    for c in list(next(iter(wm_cols.values()))):
        ks = [k for k in real_m if k in wm_cols]
        v10[task].setdefault("r_by_wm_column", {})[c] = round(float(np.corrcoef([real_m[k] for k in ks], [wm_cols[k][c] for k in ks])[0, 1]), 3)
OUT["V10_droid_task_prior_r(paper: towel 0.89, carrot 0.18)"] = v10


# V10b 'all' 초기조건 평균(저장소 avgs 열 4,5행 = base, specialist)으로 상관 재계산
v10b = {}
for task in ["towel", "carrot"]:
    pol = sorted(f.stem for f in (irom / task / "real").glob("*.csv"))
    rl = [float(pd.read_csv(irom / task / "real" / f"{q}.csv")["success"].astype(float).mean()) for q in pol]
    for i, nm in [(4, "base_all"), (5, "specialist_all")]:
        wv = [float(pd.read_csv(irom / task / "wm" / f"{q}.csv").iloc[i, 7]) for q in pol]
        v10b.setdefault(task, {})[nm] = round(float(np.corrcoef(rl, wv)[0, 1]), 3)
OUT["V10b_all_init_r (논문 towel 0.89, carrot 0.18)"] = v10b
# V11 그림 4 막대(그림 판독 값)
OUT["V11_fig4_read"] = dict(values={"AnyRank": 1275, "AnyRank0": 1276, "AnyRank_inf": 1251, "Bonferroni": 2487, "W.Bonferroni(5/6)": 2920},
                            reduction_vs_bonf_pct=round(100 * (1 - 1275 / 2487), 1), best_vs_bonf_pct=round(100 * (1 - 1251 / 2487), 1))

(ROOT / "analysis" / "recompute_results.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1))
print(json.dumps(OUT, ensure_ascii=False, indent=1)[:6000])
