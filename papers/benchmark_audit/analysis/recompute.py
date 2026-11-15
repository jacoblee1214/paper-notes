"""논문 수치와 공개 결과 파일의 재계산 (CPU, numpy/scipy).

입력: repo/ (공식 저장소를 얕게 clone한 것, 정적으로 읽기만 함). 공식 스크립트(analysis/run_analysis.py 등)는 실행하지 않는다.
출력: analysis/recompute_results.json (표 1–9의 재계산, 검정, 시각화 입력)

실행: python3 -I analysis/recompute.py
"""
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm, kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "repo"
OUT = ROOT / "analysis" / "recompute_results.json"
Z = norm.ppf(0.975)


def wilson(x, n, z=Z):
    p = x / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def newcombe(xc, nc, xa, na):
    """Newcombe–Wilson CI for pc - pa (논문 부록 A.2 식)."""
    pc, pa = xc / nc, xa / na
    Lc, Uc = wilson(xc, nc)
    La, Ua = wilson(xa, na)
    d = pc - pa
    lo = d - math.sqrt((pc - Lc) ** 2 + (Ua - pa) ** 2)
    hi = d + math.sqrt((Uc - pc) ** 2 + (pa - La) ** 2)
    return 100 * d, 100 * lo, 100 * hi


def read_csv(rel):
    with open(REPO / rel, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


res = {}

# ---------------------------------------------------------------- 표 1: 보고 점수와 차이
t1 = {
    "LIBERO Spatial": (99.0, 84.7, 99.8), "LIBERO Object": (100.0, 88.4, 100.0),
    "LIBERO Goal": (98.8, 79.2, 99.5), "LIBERO Long": (92.4, 53.7, 98.8),
}
res["table1_libero_delta"] = {k: {"probe": p, "foundation_delta": round(f - p, 1), "best_delta": round(b - p, 1)}
                              for k, (p, f, b) in t1.items()}
res["table1_calvin_delta"] = {
    "D->D": {"probe": 3.123, "foundation_delta": round(2.83 - 3.123, 2), "best_delta": round(4.52 - 3.123, 2)},
    "ABC->D": {"probe": 3.242, "foundation_delta": round(2.48 - 3.242, 2), "best_delta": round(4.78 - 3.242, 2)},
    "ABCD->D": {"probe": 3.872, "foundation_delta": round(4.09 - 3.872, 2), "best_delta": round(4.80 - 3.872, 2)},
}
res["table1_other_delta"] = {
    "RoboTwin clean": {"probe": 60.4, "x_vla_delta": round(72.8 - 60.4, 1), "best_delta": round(95.8 - 60.4, 1)},
    "RoboTwin rand": {"probe": 59.4, "x_vla_delta": round(72.8 - 59.4, 1), "best_delta": round(96.1 - 59.4, 1)},
    "RoboCasa RSS24": {"probe": 18.8, "gr00t_delta": round(49.6 - 18.8, 1), "best_delta": round(79.2 - 18.8, 1)},
    "SimplerEnv WidowX": {"probe": 0.0, "x_vla_delta": 95.8, "best_delta": 97.9},
}

# ---------------------------------------------------------------- 표 2: 최적 체크포인트 대 최종 체크포인트
best = [99.0, 100.0, 98.8, 92.4]
final = [98.2, 99.4, 94.8, 88.0]
res["table2_libero"] = {
    "best_mean": round(float(np.mean(best)), 2), "final_mean": round(float(np.mean(final)), 2),
    "gap_pp_range": [round(min(b - f for b, f in zip(best, final)), 1), round(max(b - f for b, f in zip(best, final)), 1)],
}

# ---------------------------------------------------------------- 표 3: RoboTwin 2.0 probe 과제별 (논문 표 3 전사)
clean = [95, 47, 44, 36, 64, 83, 48, 100, 67, 86, 8, 99, 69, 63, 83, 6, 96, 14, 83, 97, 30, 25, 83, 80, 91,
         50, 97, 62, 24, 90, 42, 15, 56, 50, 79, 58, 69, 99, 9, 50, 42, 82, 100, 100, 28, 62, 20, 60, 33, 46]
rand = [90, 42, 55, 40, 68, 81, 50, 100, 67, 80, 16, 100, 61, 59, 86, 5, 95, 21, 68, 93, 24, 32, 80, 69, 94,
        57, 97, 51, 27, 87, 43, 20, 62, 47, 76, 58, 67, 92, 7, 36, 44, 77, 100, 100, 31, 58, 25, 58, 29, 45]
assert len(clean) == 50 and len(rand) == 50
res["table3_robotwin_probe_mean"] = {"clean": round(float(np.mean(clean)), 2), "rand": round(float(np.mean(rand)), 2),
                                     "reported": [60.4, 59.4]}

# ---------------------------------------------------------------- 표 4–5: SimplerEnv stack (보정 288, 변경 3종 288)
calib_rows = read_csv("creeping_overfitting/results/simplerenv/fixed_grid_calibration/per_episode_results_all.csv")
alt_rows = read_csv("creeping_overfitting/results/simplerenv/distribution_overfitting/per_episode_results_all.csv")
pol_map = {"cogact": "CogACT-Base", "internvla_m1": "InternVLA-M1", "xvla": "X-VLA-WidowX", "dexbotic": "Dexbotic (DB-MemVLA)"}
cond_map = {"reverse_language": "Reverse language", "stacked_support": "Stacked support", "random_pose_arm": "Random pose + arm"}
paper_t5 = {  # 논문 표 5: (calib, altered, drop, lo, hi)
    ("cogact", "reverse_language"): (60, 28, 11.11, 5.26, 16.95), ("cogact", "stacked_support"): (60, 64, -1.39, -8.09, 5.33),
    ("cogact", "random_pose_arm"): (60, 31, 10.07, 4.13, 15.99), ("internvla_m1", "reverse_language"): (73, 46, 9.38, 2.76, 15.91),
    ("internvla_m1", "stacked_support"): (73, 60, 4.51, -2.38, 11.35), ("internvla_m1", "random_pose_arm"): (73, 46, 9.38, 2.76, 15.91),
    ("xvla", "reverse_language"): (172, 153, 6.60, -1.49, 14.57), ("xvla", "stacked_support"): (172, 90, 28.47, 20.46, 35.96),
    ("xvla", "random_pose_arm"): (172, 160, 4.17, -3.88, 12.14), ("dexbotic", "reverse_language"): (129, 137, -2.78, -10.84, 5.33),
    ("dexbotic", "stacked_support"): (129, 79, 17.36, 9.54, 24.89), ("dexbotic", "random_pose_arm"): (129, 112, 5.90, -2.14, 13.84),
}
calib_stack = defaultdict(lambda: [0, 0])
for r in calib_rows:
    if r["task"] == "stack" and r["policy"] in pol_map:
        calib_stack[r["policy"]][0] += int(r["success"])
        calib_stack[r["policy"]][1] += 1
alt = defaultdict(lambda: [0, 0])
for r in alt_rows:
    if r["policy"] in pol_map and r["condition"] in cond_map:
        alt[(r["policy"], r["condition"])][0] += int(r["success"])
        alt[(r["policy"], r["condition"])][1] += 1
t5 = []
max_abs_diff_t5 = 0.0
for (p, c), (pc_cnt, pa_cnt, pdrop, plo, phi) in paper_t5.items():
    xc, nc = calib_stack[p] if calib_stack[p][1] == 288 else (None, None)
    # 보정 값은 논문 표 5의 보정 열과 같아야 한다 (stack 288)
    xa, na = alt[(p, c)]
    d, lo, hi = newcombe(xc, nc, xa, na)
    max_abs_diff_t5 = max(max_abs_diff_t5, abs(d - pdrop), abs(lo - plo), abs(hi - phi))
    t5.append({"policy": pol_map[p], "change": cond_map[c], "calib": f"{xc}/{nc}", "altered": f"{xa}/{na}",
               "drop_pp": round(d, 2), "ci": [round(lo, 2), round(hi, 2)],
               "paper": [pc_cnt, pa_cnt, pdrop, plo, phi], "match": (xc == pc_cnt and xa == pa_cnt)})
res["table5_simplerenv_newcombe"] = t5
res["table5_max_abs_diff_pp"] = round(max_abs_diff_t5, 3)
res["table5_all_counts_match"] = all(x["match"] for x in t5)

# 그림 1·표 4 보정 값 (stack): 논문 표 4 보정 열 20.8, 25.3, 59.7, 44.8
res["table4_calibration_rates"] = {pol_map[p]: round(100 * calib_stack[p][0] / calib_stack[p][1], 1) for p in pol_map}

# ---------------------------------------------------------------- 표 6, 8: CALVIN 짝지은 CI와 독립 CI
rs = read_csv("creeping_overfitting/results/calvin/resampled_pose_per_sequence.csv")
fs = read_csv("creeping_overfitting/results/calvin/fresh_sequence_per_sequence.csv")
conds = sorted({r["condition"] for r in rs})
res["calvin_resampled_conditions"] = conds
cal_name = [c for c in conds if "calib" in c][0]
rsm_name = [c for c in conds if "calib" not in c][0]
paper_t6 = {"X-VLA": (4.165, 3.138, 1.027, 0.890, 1.164), "GR-1": (3.244, 2.495, 0.749, 0.624, 0.874),
            "RoboFlamingo": (2.367, 1.869, 0.498, 0.372, 0.624)}
t6 = []
for pol, (pcal, prs, pdrop, plo, phi) in paper_t6.items():
    cal = {int(r["global_index"]): r for r in rs if r["policy"] == pol and r["condition"] == cal_name}
    rsd = {int(r["global_index"]): r for r in rs if r["policy"] == pol and r["condition"] == rsm_name}
    idx = sorted(set(cal) & set(rsd))
    dif = np.array([int(cal[i]["tasks_completed"]) - int(rsd[i]["tasks_completed"]) for i in idx], float)
    m = dif.mean()
    se = dif.std(ddof=1) / math.sqrt(len(dif))
    ch_c = np.array([int(cal[i]["chain_success"]) for i in idx]).sum()
    ch_r = np.array([int(rsd[i]["chain_success"]) for i in idx]).sum()
    cal_atc = np.mean([int(cal[i]["tasks_completed"]) for i in idx])
    rs_atc = np.mean([int(rsd[i]["tasks_completed"]) for i in idx])
    t6.append({"policy": pol, "n": len(idx), "calib_atc": round(cal_atc, 3), "resampled_atc": round(rs_atc, 3),
               "drop": round(m, 3), "ci": [round(m - Z * se, 3), round(m + Z * se, 3)],
               "chain_counts": [int(ch_c), int(ch_r)], "paper": [pcal, prs, pdrop, plo, phi]})
res["table6_calvin_resampled"] = t6

paper_t8 = {"X-VLA": (-0.0145, -0.1287, 0.0997), "GR-1": (0.1070, -0.0373, 0.2513), "RoboFlamingo": (0.0690, -0.0677, 0.2057)}
t8 = []
for pol, (pdrop, plo, phi) in paper_t8.items():
    cal = [int(r["tasks_completed"]) for r in rs if r["policy"] == pol and r["condition"] == cal_name]
    fresh = [int(r["tasks_completed"]) for r in fs if r["policy"] == pol]
    cal, fresh = np.array(cal, float), np.array(fresh, float)
    drop = cal.mean() - fresh.mean()
    se = math.sqrt(cal.var(ddof=1) / len(cal) + fresh.var(ddof=1) / len(fresh))
    t8.append({"policy": pol, "n_calib": len(cal), "n_fresh": len(fresh), "fresh_seeds": 2,
               "calib_atc": round(cal.mean(), 4), "fresh_atc": round(fresh.mean(), 4),
               "drop": round(drop, 4), "ci": [round(drop - Z * se, 4), round(drop + Z * se, 4)],
               "paper": [pdrop, plo, phi]})
res["table8_calvin_fresh"] = t8

# 순위 보존 (CALVIN 세 정책): 보정 대 재표본 대 새 시퀀스
cal_rank = [t["calib_atc"] for t in t6]
rs_rank = [t["resampled_atc"] for t in t6]
tau_cr = kendalltau(cal_rank, rs_rank)[0]
res["calvin_rank_preservation"] = {"kendall_tau_calib_vs_resampled": round(float(tau_cr), 3),
                                   "n_policies": 3, "note": "세 정책 모두 순위가 같습니다. 수준(level)만 내려갑니다."}

# ---------------------------------------------------------------- 표 7: LIBERO 새 init-state (2000 대 10000)
lib = read_csv("creeping_overfitting/results/libero/sample_overfitting_summary.csv")
paper_t7 = {"spatial_forcing": (0.62, -0.18, 1.27), "simvla": (-0.30, -1.15, 0.40), "pi05_lerobot": (0.14, -0.69, 0.82)}
t7 = []
for r in lib:
    d, lo, hi = newcombe(int(r["official_successes"]), int(r["official_rows"]),
                         int(r["fresh_successes"]), int(r["fresh_rows"]))
    pdrop, plo, phi = paper_t7[r["policy"]]
    t7.append({"policy": r["policy_label"], "calib": f'{r["official_successes"]}/{r["official_rows"]}',
               "fresh": f'{r["fresh_successes"]}/{r["fresh_rows"]}', "drop_pp": round(d, 2),
               "ci": [round(lo, 2), round(hi, 2)], "paper": [pdrop, plo, phi]})
res["table7_libero_newcombe"] = t7

# ---------------------------------------------------------------- 표 9: SimplerEnv 데이터 출처 (24개 과제 격자)
dsd_trials = read_csv("data_source_dependency/results/scripted_widowx/trials.csv")
dsd_count = defaultdict(lambda: [0, 0])
for r in dsd_trials:
    dsd_count[r["task"]][0] += int(r["success"] == "true")
    dsd_count[r["task"]][1] += 1
res["table9_recount_from_trials"] = {t: f"{v[0]}/{v[1]}" for t, v in dsd_count.items()}
res["table9_recount_all"] = f"{sum(v[0] for v in dsd_count.values())}/{sum(v[1] for v in dsd_count.values())}"
res["table9_paper"] = {"stack": "24/24", "carrot": "23/24", "spoon": "21/24", "eggplant": "23/24", "all": "91/96 (94.8%)"}

# ---------------------------------------------------------------- 그림 3: 공개 집계 기준 유의성 분류 (재집계)
ac = read_csv("statistical_significance/significance_categories/all_comparisons.csv")
cats = Counter((r["benchmark"], r["track"], r["category"]) for r in ac)
bench_tot = Counter((r["benchmark"], r["track"]) for r in ac)
fig3 = {}
for (b, tr), n in sorted(bench_tot.items()):
    fig3[f"{b}/{tr}"] = {c: cats.get((b, tr, c), 0) for c in
                         ["no_improvement", "provably_not_significant", "indeterminate", "provably_significant"]}
    fig3[f"{b}/{tr}"]["n"] = n
    fig3[f"{b}/{tr}"]["sig_share_pct"] = round(100 * cats.get((b, tr, "provably_significant"), 0) / n, 1)
libero_pool = {c: sum(cats.get(("LIBERO", t, c), 0) for t in ["spatial", "object", "goal"]) for c in
               ["no_improvement", "provably_not_significant", "indeterminate", "provably_significant"]}
libero_n = sum(bench_tot[("LIBERO", t)] for t in ["spatial", "object", "goal"])
fig3["LIBERO spatial+object+goal (pooled)"] = dict(libero_pool, n=libero_n,
                                                   sig_share_pct=round(100 * libero_pool["provably_significant"] / libero_n, 1))
res["figure3_recount_current_release"] = fig3
res["figure3_paper_reported_share_pct"] = {"LIBERO": 19.8, "SimplerEnv": 19.7, "RoboCasa": 53.3, "RoboTwin2": 73.7,
                                           "CALVIN ABC-D (fig3 label)": 43.9}

# 필요 기준 검산: 논문 부록 C의 하한 (paired Wald). L = qS + r, Qlo = r - r^2/S, 기각 조건 L > z sqrt(S/(S-1) Q)
def necessary_count(T, S, alpha=0.05):
    z = norm.ppf(1 - alpha)
    c = z * math.sqrt(S / (S - 1))
    for L in range(1, T * S + 1):
        r = L % S
        Qlo = r - r * r / S
        if L > c * math.sqrt(max(Qlo, 0.0)):
            return L
    return None

nec_check = []
for r in ac:
    T, S = int(r["T"]), int(r["S"])
    nec = necessary_count(T, S)
    if r["necessary_count_delta"].lstrip("-").isdigit():
        nec_check.append((int(r["necessary_count_delta"]) == nec))
res["figure3_necessary_cutoff_match_rows"] = [sum(nec_check), len(nec_check)]

# 충분 기준 검산 (T=1인 행만): 부록 C.4 Qhi 닫힌 형식(R=1) = min(a+b, 2S-a-b) - (b-a)^2/S
def qhi_binary_T1(A, B, S):
    return min(A + B, 2 * S - A - B) - (B - A) ** 2 / S

suff_checked, suff_ok = 0, 0
for r in ac:
    T, S, R = int(r["T"]), int(r["S"]), int(r["R"])
    if T != 1 or R != 1 or r["category"] == "no_improvement":
        continue
    A = int(r["previous_sota_count"])
    Lr = int(r["count_delta"])
    z = norm.ppf(1 - 0.05)
    c = z * math.sqrt(S / (S - 1))
    # 충분 기준 = 모든 L' >= 기준에서 L' > c sqrt(Qhi(A, A+L'))인 최소 L'
    n_la = S - A
    ok = [Lq > c * math.sqrt(max(qhi_binary_T1(A, A + Lq, S), 0.0)) for Lq in range(1, n_la + 1)]
    if all(ok):
        sufficient = 1
    else:
        last_false = max(i for i, v in enumerate(ok) if not v)
        sufficient = last_false + 2 if last_false + 2 <= n_la else None
    if not r["sufficient_count_delta"].lstrip("-").isdigit():
        continue
    suff_checked += 1
    suff_ok += int(sufficient == int(r["sufficient_count_delta"]))
res["figure3_sufficient_check_T1_binary"] = [suff_ok, suff_checked]

# 범주 규칙 자체의 내부 일관성 (CSV의 count_delta와 기준 열 사용)
inconsistent = 0
for r in ac:
    d = int(r["count_delta"]) if r["count_delta"] not in ("",) else None
    rep = float(r["reported_delta"])
    if not (r["necessary_count_delta"].lstrip("-").isdigit() and r["sufficient_count_delta"].lstrip("-").isdigit()):
        continue
    nec, suf = int(r["necessary_count_delta"]), int(r["sufficient_count_delta"])
    if rep <= 0:
        exp = "no_improvement"
    elif d < nec:
        exp = "provably_not_significant"
    elif d >= suf:
        exp = "provably_significant"
    else:
        exp = "indeterminate"
    inconsistent += int(exp != r["category"])
res["figure3_category_rule_mismatch_rows"] = [inconsistent, len(ac)]

# ---------------------------------------------------------------- 그림 3 대 paper_inputs 대 release_current
pie_cur = read_csv("analysis/release_current_values/statistical_significance_pie_counts.csv")
res["pie_counts_release_current"] = [{k: r[k] for k in r} for r in pie_cur]

# ---------------------------------------------------------------- 논문 v1 대 공개 릴리스: 행 단위 분류 차이 (paper_name으로 맞춤)
MAP = {"negative": "no_improvement", "not_significant": "provably_not_significant",
       "possibly_significant": "indeterminate", "certainly_significant": "provably_significant"}
PI_FILES = {("LIBERO", "spatial"): "stat_sig_libero_spatial_cutoff_data.csv", ("LIBERO", "object"): "stat_sig_libero_object_cutoff_data.csv",
            ("LIBERO", "goal"): "stat_sig_libero_goal_cutoff_data.csv", ("CALVIN", "abc_d_atc"): "stat_sig_calvin_abc_d_cutoff_data.csv",
            ("SimplerEnv", "widowx_bridge"): "stat_sig_simplerenv_widowx_bridge_cutoff_data.csv",
            ("RoboCasa", "rss24"): "stat_sig_robocasa_rss24_cutoff_data.csv",
            ("RoboTwin2", "hard_randomized"): "stat_sig_robotwin2_hard_randomized_cutoff_data.csv"}
cur_by = {}
for r in ac:
    cur_by.setdefault((r["benchmark"], r["track"]), []).append(r)
changed = Counter()
rows_changed = 0
rows_total = 0
unmatched = 0
for key, fname in PI_FILES.items():
    pi_rows = read_csv(f"analysis/paper_inputs/{fname}")
    cur = cur_by.get(key, [])
    cur_map = Counter()
    for r in cur:
        cur_map[(r["paper_name"], r["category"])] += 1
    pi_names = Counter(r["paper_name"] for r in pi_rows)
    for r in pi_rows:
        rows_total += 1
        cand = [c for c in cur if c["paper_name"] == r["paper_name"]]
        if not cand:
            unmatched += 1
            continue
        if MAP[r["delta_bin"]] != cand[0]["category"]:
            rows_changed += 1
            changed[(key[0], MAP[r["delta_bin"]], cand[0]["category"])] += 1
res["paper_v1_vs_release_rowlevel"] = {"paper_input_rows": rows_total, "rows_with_changed_category": rows_changed,
                                       "unmatched_by_paper_name": unmatched,
                                       "by_benchmark_from_to": {f"{k[0]}: {k[1]} -> {k[2]}": v for k, v in changed.items()}}

# ---------------------------------------------------------------- 표 9 검정: 91/96 대 92/96 (X-VLA 보고값)
from scipy.stats import fisher_exact
_or, _p = fisher_exact([[91, 5], [92, 4]])
res["table9_fisher_22M_vs_xvla"] = {"22M_scripted": "91/96", "x_vla_reported": "92/96", "fisher_two_sided_p": round(float(_p), 3)}

# ---------------------------------------------------------------- LIBERO Goal 5개 정책 쌍 불일치 D (공개 에피소드 파일)
pol_dirs = sorted((REPO / "statistical_significance/libero_goal_5x5k/policies").glob("*/episodes_combined.csv"))
outcomes = {}
for f in pol_dirs:
    name = f.parent.name
    with open(f, encoding="utf-8", newline="") as fh:
        outcomes[name] = {r["instance_id"]: int(r["success"]) for r in csv.DictReader(fh)}
pairs = []
names = sorted(outcomes)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = outcomes[names[i]], outcomes[names[j]]
        common = set(a) & set(b)
        dis = sum(1 for k in common if a[k] != b[k])
        pairs.append(dis / len(common))
pair_csv = {(r["policy_a"], r["policy_b"]): float(r["D"]) for r in read_csv("statistical_significance/libero_goal_5x5k/pairwise_disagreement.csv")}
res["libero_goal_pairwise_D"] = {
    "policies": names, "n_instances_per_policy": len(next(iter(outcomes.values()))),
    "mean_D_recomputed": round(float(np.mean(pairs)), 5), "median_D_recomputed": round(float(np.median(pairs)), 5),
    "mean_D_reported_in_release": 0.03528, "median_D_reported_in_release": 0.0354,
}
ok_pairs = 0
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        key = (names[i], names[j]) if (names[i], names[j]) in pair_csv else (names[j], names[i])
        if key in pair_csv:
            a, b = outcomes[names[i]], outcomes[names[j]]
            common = set(a) & set(b)
            d = sum(1 for k in common if a[k] != b[k]) / len(common)
            ok_pairs += int(abs(d - pair_csv[key]) < 5e-5)
res["libero_goal_pairwise_D"]["pairs_matching_csv_to_1e-4"] = [ok_pairs, len(pairs)]

# ---------------------------------------------------------------- 리더보드 집계 (LIBERO 2026-03 79편)
lt = read_csv("leaderboards/libero/libero_citation_tracker.csv")
rr = [r for r in lt if r["LIBERO usage"] == "reports-results"]
mar = [r for r in rr if r["First arXiv date"].startswith("2026-03")]
res["libero_tracker"] = {"rows": len(lt), "reports_results": len(rr), "march2026_reports_results": len(mar),
                         "march2026_all_rows": sum(1 for r in lt if r["First arXiv date"].startswith("2026-03")),
                         "reports_results_no_arxiv_date": sum(1 for r in rr if r["First arXiv date"] in ("", "n/a"))}

# ---------------------------------------------------------------- 스위트 간 순위 (LIBERO, 설명용)
probe_suite = [99.0, 100.0, 98.8, 92.4]
openvla_suite = [84.7, 88.4, 79.2, 53.7]
res["libero_suite_rank_tau_probe_vs_openvla"] = round(float(kendalltau(probe_suite, openvla_suite)[0]), 3)
res["libero_suite_rank_rho_probe_vs_openvla"] = round(float(spearmanr(probe_suite, openvla_suite)[0]), 3)

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", OUT)
print(json.dumps({k: res[k] for k in [
    "table2_libero", "table3_robotwin_probe_mean", "table5_max_abs_diff_pp", "table5_all_counts_match",
    "table4_calibration_rates", "calvin_rank_preservation", "figure3_necessary_cutoff_match_rows",
    "figure3_sufficient_check_T1_binary", "figure3_category_rule_mismatch_rows", "libero_tracker",
    "libero_suite_rank_tau_probe_vs_openvla"]}, ensure_ascii=False, indent=1))
for t in t5:
    print(t)
for t in t6 + t8:
    print(t)
for t in t7:
    print(t)
for k, v in fig3.items():
    print(k, v)
