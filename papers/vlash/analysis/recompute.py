"""VLASH(arXiv:2512.01031 v2) 표 수치 재계산과 일관성 검사. numpy, scipy만 사용. 1초 미만.

실행: python3 analysis/recompute.py  → analysis/recompute_results.json
검사 라벨: V1 표 I 평균, V2 백분율의 정수 가능성, V3 표 II 지연 산술, V4 표 III 검정,
V5 표 IV 효율, V6 그림 7·8 디지타이즈와 시도 수, V7 v1 산술, V8 Kinetix, V9 공식 평가 스크립트의 에피소드 수.
"""
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
R = {}


def h1(v):
    return float(Decimal(repr(v)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# V1 표 I (v2)
T1 = {
    "pi05": {"Sync0": [97.3, 99.6, 96.7, 93.5, 96.8], "D1": [98.8, 99.2, 96.7, 94.4, 97.2],
             "D2": [97.5, 99.2, 97.3, 94.6, 97.1], "D3": [94.4, 98.8, 93.3, 91.9, 94.6]},
    "smolvla": {"Sync0": [81.3, 92.9, 85.8, 55.8, 79.0], "D1": [80.0, 92.9, 82.3, 53.1, 77.1],
                "D2": [78.5, 92.1, 86.9, 55.0, 78.1], "D3": [79.8, 94.2, 87.5, 54.8, 79.1]},
}
v1 = {}
for m, rows in T1.items():
    for k, r in rows.items():
        mean = sum(r[:4]) / 4
        v1[f"{m}_{k}"] = {"reported": r[4], "mean_of_4": round(mean, 3), "diff": round(mean - r[4], 3),
                          "rounds_to_reported": round(mean + 1e-9, 1) == r[4]}
R["V1_table1_avg"] = v1
# 스위트당 480 에피소드(V9)로 가정하고, 표시값과 맞는 정수 개수에서 만들 수 있는 평균 범위
def counts(x, N=480):
    return [k for k in range(N + 1) if h1(100 * k / N) == x]


rng = {}
for m, rows_ in T1.items():
    for k_, r in rows_.items():
        lo_ = sum(min(counts(x)) for x in r[:4]) / (4 * 480) * 100
        hi_ = sum(max(counts(x)) for x in r[:4]) / (4 * 480) * 100
        rng[f"{m}_{k_}"] = {"reported": r[4], "mean_range_N480": [round(lo_, 2), round(hi_, 2)],
                            "consistent": h1(lo_) <= r[4] <= h1(hi_) or lo_ - 0.05 <= r[4] <= hi_ + 0.05}
R["V1_table1_avg_with_N480_counts"] = rng

# V2 정수 가능성: 스위트당 N 에피소드면 모든 값이 round(100k/N,1)로 나와야 한다
allvals = [x for rows in T1.values() for r in rows.values() for x in r[:4]]


def feasible(N):
    bad = []
    for x in allvals:
        ok = any(h1(100 * k / N) == x for k in range(N + 1))
        if not ok:
            bad.append(x)
    return bad


R["V2_integer_feasibility"] = {str(N): {"n_infeasible_of_%d" % len(allvals): len(feasible(N)),
                                         "infeasible": feasible(N)[:12]} for N in (50, 100, 480, 500, 1000)}
R["V2_min_N_all_feasible_up_to_1000"] = [N for N in range(10, 1001) if not feasible(N)][:10]

# LIBERO 유의성: 스위트 N=480 또는 500, 4스위트 합 2000, 한 시드, 독립 가정
sig = {}
for N in (480, 500):
    n = 4 * N
    for label, a, b in (("pi05 Sync 96.8 vs D3 94.6", 96.8, 94.6), ("pi05 Sync 96.8 vs D1 97.2", 96.8, 97.2),
                        ("smol Sync 79.0 vs D3 79.1", 79.0, 79.1)):
        ka, kb = round(a / 100 * n), round(b / 100 * n)
        _, p = stats.fisher_exact([[ka, n - ka], [kb, n - kb]])
        sig[f"N={n} {label}"] = round(p, 4)
R["V2_libero_fisher_total"] = sig

# V3 표 II
D = 500.0
L = {"H100": (23.2, 48.4), "RTX5090": (29.4, 49.6), "RTX4090": (32.3, 54.2)}
rep = {"H100": (546.4, 96.8, 46.4, 11.8, 2.1), "RTX5090": (558.8, 99.2, 58.8, 9.5, 1.7),
       "RTX4090": (564.6, 108.4, 64.6, 8.7, 1.7)}
t2 = {}
for g, (Lv, Lr) in L.items():
    sync, rtc, vl = D + 2 * Lv, 2 * Lr, 2 * Lv
    t2[g] = {"sync": sync, "rtc": rtc, "vlash": vl, "sp_sync": round(sync / vl, 3), "sp_rtc": round(rtc / vl, 3),
             "matches_reported": [sync, rtc, vl, round(sync / vl, 1), round(rtc / vl, 1)] == list(rep[g])}
R["V3_table2"] = t2
# 동기 청크 길이 K에 대한 민감도 (50 Hz, RTX 5090 L=29.4)
sens = {}
for K in (5, 10, 25, 50):
    Dk = K / 50 * 1000
    sens[f"K={K}"] = round((Dk + 2 * 29.4) / (2 * 29.4), 2)
R["V3_speedup_vs_sync_chunk_K_rtx5090"] = sens
R["V3_L_in_control_steps_50Hz"] = {g: round(v[0] / 20, 2) for g, v in L.items()}
R["V3_realworld_L_from_15Hz_ms"] = round(1000 / 15, 1)

# V4 표 III
k = 11
lo, hi = wilson(k, 20)
R["V4_pingpong"] = {"vlash_wilson95": [round(lo, 3), round(hi, 3)],
                    "fisher_p_11of20_vs_0of20": float(stats.fisher_exact([[11, 9], [0, 20]])[1]),
                    "zero_of_20_wilson_upper": round(wilson(0, 20)[1], 3)}
R["V4_whack_totals_5_rounds"] = {m: round(v * 5, 2) for m, v in
                                 {"Sync": 3.2, "Naive": 8.6, "RTC": 11.0, "VLASH": 28.8}.items()}

# V5 표 IV
tok_orig = 16 * 650
tok_vl = 4 * (600 + 4 * 50)
R["V5_table4"] = {
    "speedup_420.99_129.29": round(420.99 / 129.29, 3),
    "tokens_per_gpu_step_original_16x650": tok_orig, "tokens_per_gpu_step_vlash_4x(600+4x50)": tok_vl,
    "token_ratio": round(tok_orig / tok_vl, 3),
    "unique_observations_per_step_original_vs_vlash": [16 * 4, 4 * 4],
    "wallclock_s": {"orig_10K": round(10000 * .42099), "orig_20K": round(20000 * .42099), "orig_30K": round(30000 * .42099),
                    "vlash_10K": round(10000 * .12929), "vlash_20K": round(20000 * .12929), "vlash_30K": round(30000 * .12929)},
    "acc": {"orig": [94.1, 97.1, 96.8], "vlash": [87.1, 94.4, 96.6]},
    "token_ratio_Nδ5_example": round(850 / 650, 4),
}
w = R["V5_table4"]["wallclock_s"]
R["V5_table4"]["vlash30K_vs_orig20K_time_ratio"] = round(w["orig_20K"] / w["vlash_30K"], 2)
R["V5_table4"]["vlash30K_vs_orig30K_time_ratio"] = round(w["orig_30K"] / w["vlash_30K"], 2)

# V6 그림 7, 8
dig = json.load(open(HERE / "fig_digitized.json"))


def rows(key):
    return dig[key]


f7 = rows("fig7_pi05")
out6 = []
for r in f7:
    p = r["errmid"] / 100
    hw = r["hw"] / 100
    r["N_from_SE"] = round(p * (1 - p) / hw ** 2, 1)
    out6.append({k: r[k] for k in ("task", "q", "method", "bar", "hw", "N_from_SE")})
R["V6_fig7_attempt_denominators"] = out6
Ns = [o["N_from_SE"] for o in out6]
R["V6_fig7_N_range"] = [min(Ns), max(Ns)]
R["V6_fig7_N_at_most_60_20rollouts_x3"] = all(n <= 60.5 for n in Ns)

# VLASH vs Sync 같은 (과제, q) z 검정 (시도 단위, 군집 무시: 반보수)
cmp = []
wins = 0
for task in ("Pick&Place", "Sorting", "Stacking"):
    for q in (1, 1.5, 2):
        a = next(r for r in f7 if r["task"] == task and r["q"] == q and r["method"] == "VLASH")
        b = next(r for r in f7 if r["task"] == task and r["q"] == q and r["method"] == "Sync")
        d = a["errmid"] - b["errmid"]
        se = math.sqrt(a["hw"] ** 2 + b["hw"] ** 2)
        z = d / se
        cmp.append({"task": task, "q": q, "vlash": a["errmid"], "sync": b["errmid"], "diff_pp": round(d, 1),
                    "z": round(z, 2), "p_two_sided": round(2 * (1 - stats.norm.cdf(abs(z))), 3)})
        wins += d > 0
R["V6_pi05_vlash_vs_sync_z"] = cmp
R["V6_pi05_vlash_gt_sync_settings"] = f"{wins}/9"
R["V6_sign_test_p_9of9"] = 2 * 0.5 ** 9
# 최대 경쟁 기준선(Sync, Naive, RTC 중 최고) 대비
best = []
for task in ("Pick&Place", "Sorting", "Stacking"):
    for q in (1, 1.5, 2):
        grp = [r for r in f7 if r["task"] == task and r["q"] == q]
        v = next(r for r in grp if r["method"] == "VLASH")
        oth = max((r for r in grp if r["method"] != "VLASH"), key=lambda r: r["errmid"])
        d = v["errmid"] - oth["errmid"]
        z = d / math.sqrt(v["hw"] ** 2 + oth["hw"] ** 2)
        best.append({"task": task, "q": q, "best_baseline": oth["method"], "diff_pp": round(d, 1), "z": round(z, 2)})
R["V6_pi05_vlash_vs_best_baseline"] = best

# 시도 단위 성공률 vs 롤아웃 단위(20 롤아웃, 최대 3회) 가능한 S 범위
R["V6_rollout_bounds"] = {
    "note": "20 롤아웃, 성공하면 종료, 실패 롤아웃은 시도 3회. N은 60-2S 이상 60-S 이하",
    "example_sync_PP_q1": "19/24(79.2%)만 후보. 롤아웃 단위 성공 19/20 = 95%"}

# 그림 8
g8 = rows("fig8_groot")
R["V6_fig8"] = [{k: r[k] for k in ("task", "method", "bar", "hw")} for r in g8]
R["V6_fig8_stacking_three_equal"] = [r["bar"] for r in g8 if r["task"] == "Stacking"]

# V7 v1 산술
D1 = 500.0
v1t2 = {"RTX5090": 30.4, "RTX4090": 36.1, "RTX5070": 64.1}
R["V7_v1_table2"] = {g: {"sync": D1 + L_, "speedup": round((D1 + L_) / L_, 2)} for g, L_ in v1t2.items()}
steps = {"sync": 156.0, "d1": 153.9, "d2": 157.6, "d3": 167.3, "d4": 176.7}
rep_time = {"sync": 8.4, "d1": 7.2, "d2": 6.4, "d3": 5.7, "d4": 5.8}
exe, inf, per = 5 / 30 * 1000, 103.0, 1000 / 30
calc = {"sync": steps["sync"] / 5 * (exe + inf) / 1000}
for d in (1, 2, 3, 4):
    calc[f"d{d}"] = steps[f"d{d}"] / 5 * (exe + max(0.0, inf - per * d)) / 1000
R["V7_v1_time_formula_vs_reported_s"] = {k: [round(calc[k], 2), rep_time[k]] for k in calc}
R["V7_v1_speedups"] = {k: round(rep_time["sync"] / rep_time[k], 2) for k in ("d1", "d2", "d3", "d4")}

# V8 Kinetix
R["V8_kinetix_delay4"] = {"abs_pp": round(81.7 - 51.2, 1), "relative_pct": round((81.7 / 51.2 - 1) * 100, 1)}

# V9 공식 평가 스크립트 run.sh의 에피소드 배분(정적 계산, 코드 실행 아님)
def sched(N, G):
    per_, rem = divmod(N, G)
    out = []
    for g in range(G):
        gn, st = (per_ + 1, g * (per_ + 1)) if g < rem else (per_, g * per_ + rem)
        t = max(1, gn // 10)
        out.append([(st % 50 + i) % 50 for i in range(t)])
    return out


res = {}
for N, G in ((500, 8), (500, 4), (500, 5), (500, 10)):
    s = sched(N, G)
    allidx = [i for x in s for i in x]
    res[f"N={N},GPUs={G}"] = {"episodes_per_suite": 10 * sum(len(x) for x in s), "distinct_init_states_per_task": len(set(allidx))}
R["V9_official_eval_episode_split"] = res
R["V9_table1_values_feasible_with_480"] = len(feasible(480)) == 0

json.dump(R, open(HERE / "recompute_results.json", "w"), indent=1, ensure_ascii=False)
for k, v in R.items():
    print(k, json.dumps(v, ensure_ascii=False)[:600])
