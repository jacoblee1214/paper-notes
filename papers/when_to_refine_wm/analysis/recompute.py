"""논문 표 3–8의 값을 다시 계산하고 표 사이 일관성을 검사한다 (numpy, scipy, CPU, 1초 미만).

입력은 모두 논문 본문·표에서 옮긴 값이다(원문 v1, arXiv:2611.24470). 결과는 recompute_results.json.
실행: python3 analysis/recompute.py
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}
checks = []


def check(name, paper, mine, tol):
    ok = abs(paper - mine) <= tol
    checks.append({"name": name, "paper": paper, "recomputed": round(float(mine), 4), "ok": bool(ok)})


# ---- 표 4: 5개 컨트롤러 (PushT, 32 paired trials) ----
T4 = {  # successes, N5, N15, R(=CEM iterations), W(=candidates), time mean, time sd
    "Fixed-5": dict(s=17, n5=56, n15=0, R=280, W=84000, t=2.567, sd=0.128),
    "Fixed-15": dict(s=28, n5=0, n15=42, R=630, W=189000, t=4.286, sd=0.689),
    "IG": dict(s=26, n5=13, n15=33, R=560, W=168000, t=4.043, sd=0.567),
    "IG+P": dict(s=27, n5=12, n15=34, R=570, W=171000, t=4.010, sd=0.450),
    "SCR": dict(s=25, n5=21, n15=25, R=480, W=144000, t=3.548, sd=0.286),
}
n = 32
for k, v in T4.items():
    check(f"{k} R = 5*N5 + 15*N15", v["R"], 5 * v["n5"] + 15 * v["n15"], 0)
    check(f"{k} W = 300*R", v["W"], 300 * v["R"], 0)
    v["succ_rate"] = v["s"] / n * 100
    v["solves"] = v["n5"] + v["n15"]

# 표 5 (단계별): t=0 과 t=25의 (5회 solve, 15회 solve) 수와 W0, W25
T5 = {  # (t0_5, t0_15, t25_5, t25_15)
    "Fixed-5": (32, 0, 24, 0), "Fixed-15": (0, 32, 0, 10), "IG": (13, 19, 0, 14),
    "IG+P": (12, 20, 0, 14), "SCR": (12, 20, 9, 5),
}
for k, (a, b, c, d) in T5.items():
    W0 = 300 * (5 * a + 15 * b)
    W25 = 300 * (5 * c + 15 * d)
    check(f"{k} W0+W25 = W (표 4)", T4[k]["W"], W0 + W25, 0)
    check(f"{k} solves at t=25 + t=0 = N (표 4)", T4[k]["solves"], a + b + c + d, 0)

# SCR 조기 종료 절약: 9 * (15-5) * 300
check("SCR early-exit saving 9*10*300", 27000, 9 * 10 * 300, 0)
check("SCR 45,000 fewer than Fixed-15 (W)", 45000, 189000 - 144000, 0)
check("SCR % fewer candidates vs Fixed-15", 23.81, (189000 - 144000) / 189000 * 100, 0.01)
check("SCR % less controller time vs Fixed-15", 17.22, (4.286 - 3.548) / 4.286 * 100, 0.01)
check("Replan work 63000 -> 36000 (%)", 42.86, (63000 - 36000) / 63000 * 100, 0.01)
check("Total 171000 -> 144000 (%)", 15.79, (171000 - 144000) / 171000 * 100, 0.01)
check("Five-controller total W", 756000, sum(v["W"] for v in T4.values()), 0)

# 표 4 Fixed-15 공식: 630 - 21*10 + 4*15 = 480
check("Eq.12 R_SCR from Fixed-15", 480, 630 - 21 * 10 + (46 - 42) * 15, 0)

# ---- 종점 혼합(endpoint mixture) 기준선, 식 (11) ----
def mixture(C, C5, C15, S5, S15):
    q = (C - C5) / (C15 - C5)
    return q, (1 - q) * S5 + q * S15

S5, S15 = T4["Fixed-5"]["succ_rate"], T4["Fixed-15"]["succ_rate"]
mix = {}
for k in ["IG", "IG+P", "SCR"]:
    q, sm = mixture(T4[k]["W"], T4["Fixed-5"]["W"], T4["Fixed-15"]["W"], S5, S15)
    mix[f"{k} workload"] = dict(q=round(q, 6), mixture=round(sm, 3), diff_pp=round(T4[k]["succ_rate"] - sm, 3))
for k in ["IG", "IG+P", "SCR"]:
    q, sm = mixture(T4[k]["t"], T4["Fixed-5"]["t"], T4["Fixed-15"]["t"], S5, S15)
    mix[f"{k} time"] = dict(q=round(q, 6), mixture=round(sm, 3), diff_pp=round(T4[k]["succ_rate"] - sm, 3))
res["mixture"] = mix
check("SCR workload-matched diff (pp)", 5.357, mix["SCR workload"]["diff_pp"], 0.002)
check("SCR time-matched diff (pp)", 5.379, mix["SCR time"]["diff_pp"], 0.01)
check("IG workload-matched diff (pp)", 0.625, mix["IG workload"]["diff_pp"], 0.002)
check("IG+P workload-matched diff (pp)", 2.768, mix["IG+P workload"]["diff_pp"], 0.002)
check("IG time-matched diff (pp)", -1.394, mix["IG time"]["diff_pp"], 0.02)

# ---- 짝지은 결과(표 7): 궤적 단위 집계 ----
# (Y5, Y15, SCR, IG+P) 그룹과 인원. 표 7의 열 합으로 복원.
groups = [  # (Y5, Y15, SCR, count)
    (1, 1, 1, 16), (0, 1, 1, 9), (0, 1, 0, 3), (1, 0, 0, 1), (0, 0, 0, 3),
]
tot = sum(g[3] for g in groups)
check("paired groups total trials", 32, tot, 0)
check("F5 successes from groups", 17, sum(g[0] * g[3] for g in groups), 0)
check("F15 successes from groups", 28, sum(g[1] * g[3] for g in groups), 0)
check("SCR successes from groups", 25, sum(g[2] * g[3] for g in groups), 0)
scr_on_f15 = sum(g[2] * g[3] for g in groups if g[1] == 1)
check("SCR successes shared with F15 (retention numerator)", 25, scr_on_f15, 0)
res["retention_SCR_vs_F15"] = dict(num=scr_on_f15, den=28, pct=round(scr_on_f15 / 28 * 100, 2))
check("paired retention 25/28 (%)", 89.29, scr_on_f15 / 28 * 100, 0.01)

# 짝지은 McNemar 정확 검정 (불일치 쌍만 사용)
def mcnemar_exact(b, c):
    k = min(b, c)
    p = min(1.0, 2 * stats.binom.cdf(k, b + c, 0.5)) if b + c > 0 else 1.0
    return float(p)

f15_only = sum(g[3] for g in groups if g[1] == 1 and g[2] == 0)   # F15 성공, SCR 실패
scr_only_vs_f15 = sum(g[3] for g in groups if g[1] == 0 and g[2] == 1)  # 없음
f5_only = sum(g[3] for g in groups if g[0] == 1 and g[2] == 0)     # F5 성공, SCR 실패
scr_only_vs_f5 = sum(g[3] for g in groups if g[0] == 0 and g[2] == 1)  # F5 실패, SCR 성공
res["mcnemar"] = {
    "SCR vs Fixed-15": dict(b_F15only=f15_only, c_SCRonly=scr_only_vs_f15, p_exact=round(mcnemar_exact(f15_only, scr_only_vs_f15), 4)),
    "SCR vs Fixed-5": dict(b_F5only=f5_only, c_SCRonly=scr_only_vs_f5, p_exact=round(mcnemar_exact(f5_only, scr_only_vs_f5), 4)),
}

# 궤적 단위 차이 (q 고정 = 4/7) 의 iid 부트스트랩: 논문의 장면 군집 부트스트랩과 다름
q_w = 4 / 7
d = []
for y5, y15, ys, c in groups:
    di = ys - ((1 - q_w) * y5 + q_w * y15)
    d += [di] * c
d = np.array(d)
rng = np.random.default_rng(0)
boot = np.array([rng.choice(d, size=d.size, replace=True).mean() for _ in range(20000)]) * 100
res["trial_bootstrap_SCR_workload_pp"] = dict(mean=round(d.mean() * 100, 3),
                                              ci95=[round(float(np.percentile(boot, 2.5)), 2), round(float(np.percentile(boot, 97.5)), 2)],
                                              note="iid 궤적 부트스트랩, q=4/7 고정. 논문 장면 군집 구간 [-6.70, +16.07]과 비교")
check("trial-level mean diff equals 12/7 over 32 (pp)", 5.357, d.mean() * 100, 0.002)

# Wilson 95% 구간 (성공률)
def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]
res["wilson95"] = {k: wilson(v["s"], n) for k, v in T4.items()}

# ---- 표 4 시간: 배치 4개의 평균과 표준편차로 Welch 검정 ----
def welch(m1, s1, m2, s2, n1=4, n2=4):
    t, p = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
    return dict(t=round(float(t), 3), p=round(float(p), 4))
res["welch_time"] = {
    "SCR vs Fixed-15": welch(3.548, 0.286, 4.286, 0.689),
    "SCR vs IG+P": welch(3.548, 0.286, 4.010, 0.450),
    "IG+P vs Fixed-15": welch(4.010, 0.450, 4.286, 0.689),
}

# ---- 표 8: 시드별 ----
# seed 42: n=16, F5 8, F15 15, SCR 14. seed 99: n=16, F5 9, F15 13, SCR 11.
# 논문은 시드별 q를 쓰지만 시드별 W는 보고하지 않는다 -> 보고된 차이에서 역산
seed_rows = {"42": (8, 15, 14, 11.486), "99": (9, 13, 11, -1.136)}
res["table8_implied_q"] = {}
for s, (a, b, c, diff) in seed_rows.items():
    S5s, S15s, Ss = a / 16 * 100, b / 16 * 100, c / 16 * 100
    mix_s = Ss - diff
    q_imp = (mix_s - S5s) / (S15s - S5s)
    res["table8_implied_q"][s] = dict(S5=round(S5s, 2), S15=round(S15s, 2), SCR=round(Ss, 2),
                                      implied_q=round(q_imp, 4), note="보고된 시드별 q 없이 차이를 맞추는 q")
check("seed 42 + seed 99 F5 = pooled 17", 17, 8 + 9, 0)
check("seed 42 + seed 99 F15 = pooled 28", 28, 15 + 13, 0)
check("seed 42 + seed 99 SCR = pooled 25", 25, 14 + 11, 0)

# ---- 표 3: 개발 장면 깊이, Fisher 정확 검정 (비짝지음) ----
p_dev_5_15 = stats.fisher_exact([[15, 17], [30, 2]])[1]
p_dev_15_30 = stats.fisher_exact([[30, 2], [29, 3]])[1]
res["dev_fisher"] = dict(p_5_vs_15=float(round(p_dev_5_15, 6)), p_15_vs_30=float(round(p_dev_15_30, 4)))
check("Cube time ratio 30/5 (%)", 30.0, 2.196 / 7.313 * 100, 0.1)
check("Table 3 PushT 30-15 time difference", 2.820, 6.595 - 3.775, 0.001)

# ---- Cube 짝지은 결과 ----
cube = dict(both=22, only5=3, only30=3, neither=4)
check("Cube total trials", 32, sum(cube.values()), 0)
check("Cube 5 successes", 25, cube["both"] + cube["only5"], 0)
res["cube_mcnemar_p"] = float(round(mcnemar_exact(cube["only5"], cube["only30"]), 4))

# ---- 표 1 관련: 로짓 임계값 ----
tau = 0.3966121994
res["gate_threshold_logit"] = round(float(np.log(tau / (1 - tau))), 4)

res["checks"] = checks
res["n_checks"] = len(checks)
res["n_fail"] = sum(1 for c in checks if not c["ok"])
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

for c in checks:
    print(("OK  " if c["ok"] else "FAIL"), c["name"], "paper", c["paper"], "mine", c["recomputed"])
print(json.dumps({k: res[k] for k in ["mixture", "mcnemar", "trial_bootstrap_SCR_workload_pp", "welch_time",
                                      "table8_implied_q", "dev_fisher", "cube_mcnemar_p", "wilson95"]}, ensure_ascii=False, indent=1))
print("checks", res["n_checks"], "fail", res["n_fail"])
