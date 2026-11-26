"""FACT (arXiv:2608.10232 v2) 보고 수치 재계산.

입력: paper/paper.txt (표 6은 이 파일에서 파싱), 본문 표 1-4, 7, 8은 아래에 옮겨 적은 값.
출력: analysis/recompute_results.json (수 초, numpy + scipy)
실행: python3 analysis/recompute.py

가정(논문에 시행 수가 명시된 곳은 그대로, 아닌 곳은 표시):
- RoboTwin: 과제 x 분할(clean, rand.)마다 100회 (부록 D, 표 6 설명)
- 실로봇: 표 2, 3의 칸마다 20회 (4.2절)
- 그림 판독 값(그림 6, 7)은 픽셀에서 읽은 근사 값
"""
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [100 * (c - h), 100 * (c + h)]


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


# ---------------------------------------------------------------- 표 6 (RoboTwin 50과제)
def parse_table6():
    rows = {}
    for line in (ROOT / "paper" / "paper.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([A-Z][A-Za-z0-9 ]+?)\s{2,}((?:\d+(?:\.\d+)?\s+){9}\d+(?:\.\d+)?)\s*$", line)
        if m:
            rows[m.group(1).strip()] = [float(x) for x in m.group(2).split()]
    return rows


rows = parse_table6()
avg = rows.pop("Average")
assert len(rows) == 50, len(rows)
cols = ["X-VLA", "Motus", "FACT w/o video", "FACT", "FACT w/ failure"]
M = np.array(list(rows.values()))  # 50 x 10
names = list(rows.keys())
res6 = {"n_tasks": len(rows)}
res6["col_means"] = {}
for i, c in enumerate(cols):
    cl, rd = M[:, 2 * i].mean(), M[:, 2 * i + 1].mean()
    res6["col_means"][c] = {"clean": round(cl, 3), "rand": round(rd, 3), "avg": round((cl + rd) / 2, 3),
                            "reported_clean": avg[2 * i], "reported_rand": avg[2 * i + 1]}
OUT["table6_means"] = res6
t1 = {"X-VLA": (72.9, 72.8, 72.9), "Motus": (88.7, 87.0, 87.8), "FACT w/o video": (82.5, 81.0, 81.8),
      "FACT": (86.3, 84.9, 85.6), "FACT w/ failure": (88.4, 86.6, 87.5)}
OUT["table1_vs_table6"] = {c: {"table1": t1[c], "from_table6": [round(res6["col_means"][c][k], 2) for k in ("clean", "rand", "avg")]}
                           for c in cols}

# 과제 단위 쌍체 검정 (과제별 clean/rand 평균)
T = {c: (M[:, 2 * i] + M[:, 2 * i + 1]) / 2 for i, c in enumerate(cols)}
Tc = {c: M[:, 2 * i] for i, c in enumerate(cols)}
Tr = {c: M[:, 2 * i + 1] for i, c in enumerate(cols)}


def paired(a, b):
    d = a - b
    nz = d[d != 0]
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    sign_p = float(stats.binomtest(wins, wins + losses, 0.5).pvalue)
    t = stats.ttest_rel(a, b)
    try:
        w = stats.wilcoxon(a, b)
        wp = float(w.pvalue)
    except ValueError:
        wp = None
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"mean_diff": round(float(d.mean()), 3), "ci95": [round(float(d.mean() - 2.0096 * se), 3), round(float(d.mean() + 2.0096 * se), 3)],
            "wins": wins, "losses": losses, "ties": int((d == 0).sum()), "sign_p": round(sign_p, 4),
            "paired_t_p": round(float(t.pvalue), 5), "wilcoxon_p": None if wp is None else round(wp, 5)}


pairs = [("FACT", "FACT w/o video"), ("FACT w/ failure", "FACT"), ("FACT w/ failure", "Motus"),
         ("FACT", "Motus"), ("FACT w/ failure", "X-VLA")]
OUT["table6_paired_tasks"] = {f"{a} vs {b}": paired(T[a], T[b]) for a, b in pairs}
OUT["table6_paired_clean"] = {f"{a} vs {b}": paired(Tc[a], Tc[b]) for a, b in pairs[:3]}
OUT["table6_paired_rand"] = {f"{a} vs {b}": paired(Tr[a], Tr[b]) for a, b in pairs[:3]}

# 시행 단위(과제 군집 무시) 비교: 두 분할 합 10,000회 (하한 성격의 p)
def trial_level(a, b):
    ka, kb = int(round(T[a].sum() * 2)), int(round(T[b].sum() * 2))
    n = 10000
    return {"k_a": ka, "k_b": kb, "n": n, "pct_a": 100 * ka / n, "pct_b": 100 * kb / n, "fisher_p": fisher(ka, n, kb, n)}


OUT["table6_trial_level"] = {f"{a} vs {b}": trial_level(a, b) for a, b in pairs}

# 과제 군집 부트스트랩 (과제 50개 재표집)
rng = np.random.default_rng(0)
boot = {}
for a, b in pairs[:3]:
    d = T[a] - T[b]
    bs = [d[rng.integers(0, 50, 50)].mean() for _ in range(20000)]
    boot[f"{a} vs {b}"] = [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
OUT["table6_bootstrap_task_cluster_ci"] = boot

# 과제별 대비: FACT w/ failure가 FACT보다 낮은 과제와 폭
d = T["FACT w/ failure"] - T["FACT"]
order = np.argsort(d)
OUT["failure_effect_per_task"] = {
    "most_negative": [(names[i], round(float(d[i]), 1)) for i in order[:5]],
    "most_positive": [(names[i], round(float(d[i]), 1)) for i in order[::-1][:5]],
    "share_ge_10pp": int((d >= 10).sum()), "share_le_minus5": int((d <= -5).sum()),
    "top5_gain_sum_share": round(float(np.sort(d)[::-1][:5].sum() / d.sum()), 3),
}
# 영향이 큰 5개 과제를 빼고 평균
keep = np.sort(d)[:-5]
OUT["failure_effect_per_task"]["mean_excluding_top5"] = round(float(keep.mean()), 3)

# 본문 3과제(그림 6) 표 6 값
f6 = ["Handover Block", "Open Microwave", "Put Bottles Dustbin"]
idx = [names.index(x) for x in f6]
OUT["fig6_tasks_in_table6_clean"] = {
    "FACT": [M[i, 6] for i in idx], "FACT_mean": float(np.mean([M[i, 6] for i in idx])),
    "FACT w/ failure": [M[i, 8] for i in idx], "FACT w/ failure_mean": float(np.mean([M[i, 8] for i in idx])),
}

# ---------------------------------------------------------------- 실로봇 표 2, 3
tab2_cols = ["Stack Cubes", "Pick Cubes", "Handover", "Stack Bowls", "Pour"]
tab2 = {
    "Cosmos": ([5, 45, 25, 35, 15], 25), "pi0": ([35, 70, 40, 50, 45], 48), "pi0.5": ([75, 100, 85, 80, 100], 88),
    "Motus": ([50, 70, 55, 85, 60], 64), "Ours": ([70, 85, 90, 80, 85], 82),
    "Ours w/ failure": ([75, 95, 85, 95, 95], 89), "Ours w/ failure + scoring": ([85, 100, 85, 100, 90], 92),
    "Ours + scoring": ([80, 80, 70, 90, 75], 79), "Ours w/o causal mask": ([50, 75, 95, 85, 80], 77),
    "Ours w/ failed-action loss": ([45, 55, 75, 65, 75], 63), "Ours w/o video co-train": ([60, 55, 35, 85, 55], 58),
}
tab3_cols = ["Stack Cubes", "Pick Cubes", "Stack Bowls"]
tab3 = {
    "Cosmos": ([15, 10, 0], 8), "pi0": ([30, 65, 75], 57), "pi0.5": ([65, 90, 100], 85), "Motus": ([55, 60, 70], 62),
    "Ours": ([45, 75, 80], 67), "Ours w/ failure": ([60, 85, 85], 77), "Ours w/ failure + scoring": ([65, 95, 85], 82),
}
rw = {"table2_avg_check": {}, "table3_avg_check": {}}
for nm, (v, rep) in tab2.items():
    rw["table2_avg_check"][nm] = {"computed": round(float(np.mean(v)), 2), "reported": rep, "ok": int(round(np.mean(v))) == rep}
for nm, (v, rep) in tab3.items():
    rw["table3_avg_check"][nm] = {"computed": round(float(np.mean(v)), 2), "reported": rep, "ok": int(round(np.mean(v))) == rep}


def counts(v):
    return int(round(sum(v) / 100 * 20))  # 칸당 20회 -> 성공 횟수 합


cmp2 = [("Ours", "Ours w/ failure"), ("Ours w/ failure", "Ours w/ failure + scoring"), ("Ours", "Motus"),
        ("Ours w/ failure", "pi0.5"), ("Ours", "Ours w/o video co-train"), ("Ours", "Ours w/o causal mask"),
        ("Ours w/ failure", "Ours w/ failed-action loss"), ("Ours", "Ours + scoring"), ("Ours w/ failure", "Motus")]
rw["table2_pooled_fisher_n100"] = {}
for a, b in cmp2:
    ka, kb = counts(tab2[a][0]), counts(tab2[b][0])
    rw["table2_pooled_fisher_n100"][f"{a} vs {b}"] = {"k_a": ka, "k_b": kb, "fisher_p": round(fisher(ka, 100, kb, 100), 4)}
rw["table2_wilson_n100"] = {nm: [round(x, 1) for x in wilson(counts(v), 100)] for nm, (v, _) in tab2.items()}
cmp3 = [("Ours", "Ours w/ failure"), ("Ours w/ failure", "Ours w/ failure + scoring"), ("Ours w/ failure", "pi0.5"),
        ("Ours", "Motus"), ("Ours w/ failure + scoring", "pi0.5")]
rw["table3_pooled_fisher_n60"] = {}
for a, b in cmp3:
    ka, kb = counts(tab3[a][0]), counts(tab3[b][0])
    rw["table3_pooled_fisher_n60"][f"{a} vs {b}"] = {"k_a": ka, "k_b": kb, "fisher_p": round(fisher(ka, 60, kb, 60), 4)}
rw["table3_wilson_n60"] = {nm: [round(x, 1) for x in wilson(counts(v), 60)] for nm, (v, _) in tab3.items()}

# 과제 하나(20회) 칸 비교 중 p < 0.05가 되는 최소 차이
def min_diff(n, base):
    for k in range(0, n + 1):
        kb = int(round(base * n))
        if kb + k <= n and fisher(kb + k, n, kb, n) < 0.05:
            return 100 * k / n
    return None


rw["min_detectable_diff_pp"] = {"n20_base80": min_diff(20, 0.8), "n20_base50": min_diff(20, 0.5), "n60_base70": min_diff(60, 0.7),
                                "n100_base80": min_diff(100, 0.8), "n10000_base85": min_diff(10000, 0.85)}
# 단계별 변화: failure 추가(칸별 차이) 과제 수
dv = np.array(tab2["Ours w/ failure"][0]) - np.array(tab2["Ours"][0])
rw["failure_effect_cells_table2"] = {"diffs": dv.tolist(), "pos": int((dv > 0).sum()), "neg": int((dv < 0).sum()),
                                     "cube_tasks_sum": int(dv[:2].sum()), "non_cube_sum": int(dv[2:].sum())}
ds = np.array(tab2["Ours w/ failure + scoring"][0]) - np.array(tab2["Ours w/ failure"][0])
dn = np.array(tab2["Ours + scoring"][0]) - np.array(tab2["Ours"][0])
rw["scoring_effect_cells"] = {"with_failure": ds.tolist(), "without_failure": dn.tolist(),
                              "mean_with": float(ds.mean()), "mean_without": float(dn.mean())}
# 점수 효과의 차이의 차이: 성공 횟수(n=100 가정) 이항 검정 (Wald z)
ks = [counts(tab2[x][0]) for x in ("Ours w/ failure + scoring", "Ours w/ failure", "Ours + scoring", "Ours")]
dd = (ks[0] - ks[1]) - (ks[2] - ks[3])
se = np.sqrt(sum(k / 100 * (1 - k / 100) / 100 for k in ks))
rw["scoring_diff_in_diff"] = {"dd_pp": float(dd), "z": float(dd / 100 / se), "p_two_sided": float(2 * (1 - stats.norm.cdf(abs(dd / 100 / se))))}
OUT["real_world"] = rw

# ---------------------------------------------------------------- 표 4, 8 (PSNR, SSIM)
fut = {"PSNR": {"ours": (26.12, 19.51, 22.82), "fail": (26.08, 25.92, 26.00)},
       "SSIM": {"ours": (0.8285, 0.7461, 0.7873), "fail": (0.8286, 0.8290, 0.8288)}}
OUT["future_metrics"] = {}
for m, d_ in fut.items():
    OUT["future_metrics"][m] = {k: {"all_computed": round((v[0] + v[1]) / 2, 4), "all_reported": v[2]} for k, v in d_.items()}
OUT["future_metrics"]["psnr_gain_failure_windows_db"] = round(25.92 - 19.51, 2)
OUT["future_metrics"]["mse_ratio_failure_windows"] = round(10 ** ((25.92 - 19.51) / 10), 2)
OUT["future_metrics"]["psnr_change_success_windows_db"] = round(26.08 - 26.12, 2)
OUT["future_metrics"]["samples_per_subset_assumed"] = 256

# ---------------------------------------------------------------- 표 7, 그림 6, 7
t7 = {"pi0.5": (47, 79.8, 88, 85), "pi0": (45, 62.2, 48, 57), "Motus": (1220, 87.8, 64, 62), "Cosmos": (620, None, 25, 8),
      "Ours w/ failure": (380, 87.5, 89, 77)}
OUT["table7"] = {
    "speedup_vs_motus": round(1220 / 380, 2), "speedup_vs_cosmos": round(620 / 380, 2), "slower_than_pi05": round(380 / 47, 2),
    "consistency_with_tables": {"sim": [79.8, 62.2, 87.8, 87.5], "real_seen": [88, 48, 64, 25, 89], "real_unseen": [85, 57, 62, 8, 77]},
}
fig6 = {"Handover Block": [14, 38, 46], "Open Microwave": [62, 80, 90], "Put Bottles Dustbin": [22, 16, 36]}  # 그림 판독 값 (%)
avg6 = np.mean(list(fig6.values()), axis=0)
OUT["fig6_read"] = {"values_pct_read": fig6, "avg_from_read": [round(float(x), 1) for x in avg6], "text_avg": [32.7, 57.3],
                    "implied_total_successes_p0": 32.7 * 3, "implied_total_successes_p100": 57.3 * 3,
                    "non_monotone_tasks": [k for k, v in fig6.items() if not (v[0] <= v[1] <= v[2])]}
fig7 = {"N": [1, 2, 4, 8], "rel_completion_read": [0.80, 0.89, 1.00, 0.945], "latency_ms_read": [370, 815, 925, 1270]}
OUT["fig7_read"] = fig7

# ---------------------------------------------------------------- 식 (6)과 코드 값 목표
T_ep = 200
t = np.arange(0, T_ep + 1)
H = 48
p = np.minimum((t + H) / T_ep, 1.0)
lam = 1.0
v_fail_eq6 = np.clip(p - lam, 0, 1)
OUT["eq6"] = {
    "lambda": lam, "T": T_ep, "H": H,
    "distinct_values_failed_windows": sorted(set(np.round(v_fail_eq6, 6).tolist())),
    "n_windows_with_value_0": int((v_fail_eq6 == 0).sum()), "n_windows": int(len(t)),
}
# 코드: 성공 raw = (L - fidx - 1)/(L-1), 실패 활성 raw = raw + 1, 정규화 (min=-1, max=2) -> [-1,1]
L = T_ep
fidx = np.minimum(t + H, L - 1)
raw = (L - fidx - 1) / (L - 1)
norm_ok = ((raw - (-1)) / 3) * 2 - 1
norm_fail = np.clip(((raw + 1 - (-1)) / 3) * 2 - 1, -1, 1)
OUT["code_value_target"] = {"success_norm_range": [round(float(norm_ok.min()), 4), round(float(norm_ok.max()), 4)],
                            "failure_norm_range": [round(float(norm_fail.min()), 4), round(float(norm_fail.max()), 4)],
                            "success_never_exceeds_failure": bool(norm_ok.max() <= norm_fail.min() + 1e-9),
                            "corr_success_vs_1minus_p": round(float(np.corrcoef(raw, 1 - p)[0, 1]), 4)}
# 창 시작 t에서 실패 표시 시점: 논문 1_fail(t+H) vs 코드 local_frame_index >= onset
onset = 100
paper_flag = (t + H) >= onset
code_flag = t >= onset
OUT["failure_flag_mismatch"] = {"onset": onset, "H": H, "windows_paper_only": int((paper_flag & ~code_flag).sum())}

# ---------------------------------------------------------------- 가중치/설정 비교
OUT["loss_weights"] = {"paper": {"wa": 20, "wv": 1, "wI": 1}, "code": {"action": 10.0, "value": 0.4, "visual": 1.0, "future_state": 0.4},
                       "ratio_action_to_visual_paper": 20.0, "ratio_action_to_visual_code": 10.0}

# 1.3K 실패 vs 데모 수
OUT["sim_data_share"] = {"demos": 2500 + 25000, "failures": 1300, "failure_share_pct": round(100 * 1300 / (27500 + 1300), 2)}

(ROOT / "analysis" / "recompute_results.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(json.dumps({k: OUT[k] for k in ("table6_means", "table6_paired_tasks", "table6_trial_level", "eq6", "code_value_target", "failure_flag_mismatch")}, ensure_ascii=False, indent=1, default=float))
