"""ZETA(arXiv:2609.02546 v3) 보고 수치 재계산. numpy, scipy만 쓴다. 실행 약 1초. 결과는 recompute_results.json.

실행: python3 -I analysis/recompute.py   (analysis/ 안에서도, 밖에서도 된다)
부록 표 8-23(논문 paper/paper.txt)을 읽어 본문 표 2-5, 그림 4-6 라벨, 초록 수치를 다시 만든다.
집계 규칙(부록 H.4): 실행(run)마다 로봇별로 과제 평균 -> 범주 안 로봇 평균 -> 네 범주(APP, GRP, ARM, FULL) 평균.
 ± 는 세 실행 집계값의 표준편차.
"""
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_tables import parse  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
T = parse()
OUT = {}

ROBOTS = ["Franka", "Franka+logo", "Franka+green", "Franka+UMI", "UR5e+Franka", "Google+Franka", "UR5e+UMI", "GoogleRobot"]
CAT = {"APP": ["Franka+logo", "Franka+green"], "GRP": ["Franka+UMI"], "ARM": ["UR5e+Franka", "Google+Franka"],
       "FULL": ["UR5e+UMI", "GoogleRobot"]}
CATS = list(CAT)


def sim_runs(tables, col):
    """tables: 과제별 표 번호 3개. 반환 {로봇: np.array(3 runs)} (과제 평균)."""
    return {r: np.mean([[T[t][r][col][k] for k in range(3)] for t in tables], axis=0) for r in ROBOTS}


def agg(per_robot):
    """로봇별 실행값 -> es, 범주별, 평균 (각 길이 3 배열)."""
    out = {"es": per_robot["Franka"]}
    for c, rs in CAT.items():
        out[c] = np.mean([per_robot[r] for r in rs], axis=0)
    out["Average"] = np.mean([out[c] for c in CATS], axis=0)
    return out


def fmt(a, ddof=0):
    return f"{a.mean():.1f} ± {a.std(ddof=ddof):.1f}"


def welch(a, b):
    r = stats.ttest_ind(a, b, equal_var=False)
    return float(r.statistic), float(r.pvalue)


def sep(a, b):
    """모든 a 실행이 모든 b 실행보다 큰가 (3 대 3 완전 분리)."""
    return bool(min(a) > max(b))


# ---------------------------------------------------------------- V1 표 2 (RQ1)
RQ1_COLS = ["Abs/World", "Abs/EEF", "EEF/World", "EEF/EEF"]
REPORTED_T2 = {  # 논문 표 2 (평균 ± 표준편차)
    "Abs/World": dict(es=(87.7, 2.1), APP=(86.3, 1.3), GRP=(82.6, 1.7), ARM=(38.5, 0.8), FULL=(33.9, 2.3), Average=(60.3, 1.1)),
    "Abs/EEF": dict(es=(92.6, 1.0), APP=(91.8, 1.1), GRP=(88.2, 2.6), ARM=(39.9, 0.5), FULL=(38.6, 0.8), Average=(64.6, 0.4)),
    "EEF/World": dict(es=(87.8, 1.4), APP=(88.0, 0.9), GRP=(77.2, 1.0), ARM=(69.8, 1.9), FULL=(58.4, 1.3), Average=(73.4, 0.7)),
    "EEF/EEF": dict(es=(91.5, 0.6), APP=(90.0, 1.1), GRP=(78.3, 2.4), ARM=(74.3, 0.9), FULL=(60.4, 0.9), Average=(75.7, 1.3)),
}
v1 = {}
rq1_runs = {}
for j, name in enumerate(RQ1_COLS):
    a = agg(sim_runs([8, 9, 10], j))
    rq1_runs[name] = a
    v1[name] = {}
    for k in ["es"] + CATS + ["Average"]:
        rm, rs = REPORTED_T2[name][k]
        v1[name][k] = dict(mean=round(float(a[k].mean()), 2), sd0=round(float(a[k].std(ddof=0)), 2),
                           sd1=round(float(a[k].std(ddof=1)), 2), rep_mean=rm, rep_sd=rs,
                           mean_ok=abs(a[k].mean() - rm) <= 0.06,
                           sd_ok0=abs(a[k].std(ddof=0) - rs) <= 0.06, sd_ok1=abs(a[k].std(ddof=1) - rs) <= 0.06)
OUT["V1_table2"] = v1
bad_mean = [(n, k) for n in v1 for k, d in v1[n].items() if not d["mean_ok"]]
n_sd0 = int(sum(d["sd_ok0"] for n in v1 for d in v1[n].values()))
n_sd1 = int(sum(d["sd_ok1"] for n in v1 for d in v1[n].values()))
OUT["V1_summary"] = dict(n_cells=24, mean_mismatch=bad_mean, sd_match_ddof0=n_sd0, sd_match_ddof1=n_sd1)

# 로봇별 이득 (RQ1): 절대 상태 -> EEF-Delta 상태 (행동 EEF-Delta 고정)
rq1_robot = {}
for j, name in enumerate(RQ1_COLS):
    pr = sim_runs([8, 9, 10], j)
    rq1_robot[name] = {r: round(float(pr[r].mean()), 1) for r in ROBOTS}
OUT["V2_rq1_per_robot"] = rq1_robot
OUT["V2_state_gain_per_robot"] = {r: round(rq1_robot["EEF/EEF"][r] - rq1_robot["Abs/EEF"][r], 1) for r in ROBOTS}
# 과제별 구글 로봇
OUT["V2_google_abs_state_by_task"] = {
    f"table{t}": {r: [round(float(np.mean(T[t][r][0])), 1), round(float(np.mean(T[t][r][1])), 1)] for r in ("Google+Franka", "GoogleRobot")}
    for t in (8, 9, 10)}

# 구글 로봇 두 변형을 뺀 5개 대상의 평균과 주효과
NONG = ["Franka+logo", "Franka+green", "Franka+UMI", "UR5e+Franka", "UR5e+UMI"]
OUT["V2_mean_over_non_google_targets"] = {n: round(float(np.mean([rq1_robot[n][r] for r in NONG])), 1) for n in RQ1_COLS}
OUT["V2_mean_over_google_targets"] = {n: round(float(np.mean([rq1_robot[n][r] for r in ("Google+Franka", "GoogleRobot")])), 1) for n in RQ1_COLS}
_avg = {n: float(rq1_runs[n]["Average"].mean()) for n in RQ1_COLS}
OUT["V2_main_effects_average"] = dict(
    state_EEFDelta_minus_Abs=round(((_avg["EEF/World"] + _avg["EEF/EEF"]) - (_avg["Abs/World"] + _avg["Abs/EEF"])) / 2, 1),
    action_EEFDelta_minus_World=round(((_avg["Abs/EEF"] + _avg["EEF/EEF"]) - (_avg["Abs/World"] + _avg["EEF/World"])) / 2, 1))
_ng = OUT["V2_mean_over_non_google_targets"]
OUT["V2_main_effects_non_google"] = dict(
    state=round(((_ng["EEF/World"] + _ng["EEF/EEF"]) - (_ng["Abs/World"] + _ng["Abs/EEF"])) / 2, 1),
    action=round(((_ng["Abs/EEF"] + _ng["EEF/EEF"]) - (_ng["Abs/World"] + _ng["EEF/World"])) / 2, 1))

# 유의성 (3 대 3 Welch, 완전 분리)
best = rq1_runs["EEF/EEF"]
sig1 = {}
for name in RQ1_COLS[:3]:
    for k in CATS + ["Average"]:
        t_, p_ = welch(best[k], rq1_runs[name][k])
        sig1[f"EEF/EEF - {name} {k}"] = dict(diff=round(float(best[k].mean() - rq1_runs[name][k].mean()), 1), welch_t=round(t_, 2),
                                            p=float(f"{p_:.2g}"), separated=sep(best[k], rq1_runs[name][k]) or sep(rq1_runs[name][k], best[k]))
OUT["V3_rq1_welch"] = sig1
OUT["V3_note"] = "3 대 3 순열검정의 최소 양측 p는 0.1이다. Welch p는 정규성·등분산 가정에 의존한다."

# ---------------------------------------------------------------- V4 RQ2 (고정 총량, 1/8/512)
rq2 = {}
for j, name in enumerate(["1", "8", "512"]):
    rq2[name] = agg(sim_runs([11, 12, 13], j))
OUT["V4_rq2_sim"] = {n: {k: round(float(rq2[n][k].mean()), 1) for k in ["es"] + CATS + ["Average"]} for n in rq2}
OUT["V4_rq2_gain_512_vs_1"] = {k: round(float(rq2["512"][k].mean() - rq2["1"][k].mean()), 1) for k in ["es"] + CATS + ["Average"]}
OUT["V4_rq2_gain_512_vs_8"] = {k: round(float(rq2["512"][k].mean() - rq2["8"][k].mean()), 1) for k in ["es"] + CATS + ["Average"]}
OUT["V4_rq2_sd"] = {n: round(float(rq2[n]["Average"].std(ddof=0)), 2) for n in rq2}
t_, p_ = welch(rq2["512"]["Average"], rq2["1"]["Average"])
OUT["V4_rq2_welch_avg_512_vs_1"] = dict(t=round(t_, 1), p=float(f"{p_:.2g}"), separated=sep(rq2["512"]["Average"], rq2["1"]["Average"]))
# 표 11-13 의 512 열 == 표 8-10 의 EEF/EEF 열 (같은 모델 재사용)
same = all(T[t2][r][2] == T[t1][r][3] for t1, t2 in ((8, 11), (9, 12), (10, 13)) for r in ROBOTS)
OUT["K_rq2_512_equals_rq1_best"] = same
# 로봇별 (1 -> 512)
OUT["V4_rq2_per_robot_1_vs_512"] = {r: [round(float(np.mean([T[t][r][0][k] for t in (11, 12, 13) for k in range(3)])), 1),
                                         round(float(np.mean([T[t][r][2][k] for t in (11, 12, 13) for k in range(3)])), 1)] for r in ROBOTS}

# ---------------------------------------------------------------- V5 RQ3 (표 4)
RQ3_COLS = ["No co-training", "LAP-EEF", "LAP-both", "Subgoal", "BBox"]
REPORTED_T4 = {
    "No co-training": dict(es=(91.5, 0.6), APP=(90.0, 1.1), GRP=(78.3, 2.4), ARM=(74.3, 0.9), FULL=(60.4, 0.9), Average=(75.7, 1.3)),
    "LAP-EEF": dict(es=(90.5, 1.1), APP=(88.6, 0.2), GRP=(86.7, 1.4), ARM=(79.3, 0.9), FULL=(70.5, 0.5), Average=(81.3, 0.5)),
    "LAP-both": dict(es=(89.5, 1.5), APP=(88.6, 0.4), GRP=(79.8, 0.8), ARM=(73.8, 1.6), FULL=(68.4, 1.5), Average=(77.7, 0.3)),
    "Subgoal": dict(es=(89.1, 1.8), APP=(90.3, 0.8), GRP=(87.4, 2.2), ARM=(77.8, 0.5), FULL=(71.5, 1.5), Average=(81.8, 0.8)),
    "BBox": dict(es=(90.4, 1.4), APP=(90.2, 0.4), GRP=(87.0, 0.9), ARM=(80.1, 0.8), FULL=(71.9, 1.4), Average=(82.3, 0.2)),
}
rq3 = {}
v5 = {}
for j, name in enumerate(RQ3_COLS):
    a = agg(sim_runs([14, 15, 16], j))
    rq3[name] = a
    v5[name] = {k: dict(mean=round(float(a[k].mean()), 2), sd0=round(float(a[k].std(ddof=0)), 2), sd1=round(float(a[k].std(ddof=1)), 2),
                        rep=REPORTED_T4[name][k],
                        mean_ok=abs(a[k].mean() - REPORTED_T4[name][k][0]) <= 0.06,
                        sd_ok0=abs(a[k].std(ddof=0) - REPORTED_T4[name][k][1]) <= 0.06,
                        sd_ok1=abs(a[k].std(ddof=1) - REPORTED_T4[name][k][1]) <= 0.06) for k in ["es"] + CATS + ["Average"]}
OUT["V5_table4"] = v5
OUT["V5_summary"] = dict(
    mean_mismatch=[(n, k) for n in v5 for k, d in v5[n].items() if not d["mean_ok"]],
    sd_match_ddof0=int(sum(d["sd_ok0"] for n in v5 for d in v5[n].values())),
    sd_match_ddof1=int(sum(d["sd_ok1"] for n in v5 for d in v5[n].values())), n_cells=30)
OUT["V5_baseline_equals_rq1_best"] = all(T[t3][r][0] == T[t1][r][3] for t1, t3 in ((8, 14), (9, 15), (10, 16)) for r in ROBOTS)
sig3 = {}
for name in RQ3_COLS[1:]:
    for k in ["es", "ARM", "Average"]:
        t_, p_ = welch(rq3[name][k], rq3["No co-training"][k])
        sig3[f"{name} - base {k}"] = dict(diff=round(float(rq3[name][k].mean() - rq3["No co-training"][k].mean()), 1),
                                          welch_t=round(t_, 2), p=float(f"{p_:.2g}"),
                                          separated=sep(rq3[name][k], rq3["No co-training"][k]) or sep(rq3["No co-training"][k], rq3[name][k]))
OUT["V5_welch"] = sig3
OUT["V5_runs_baseline_avg"] = [round(float(x), 2) for x in rq3["No co-training"]["Average"]]
OUT["V5_runs_avg"] = {n: [round(float(x), 2) for x in rq3[n]["Average"]] for n in RQ3_COLS}
# 범주별 이득
OUT["V5_gain_by_category"] = {n: {k: round(float(rq3[n][k].mean() - rq3["No co-training"][k].mean()), 1) for k in ["es"] + CATS + ["Average"]}
                              for n in RQ3_COLS[1:]}

# ---------------------------------------------------------------- V6 RQ4 (표 17-19)
RQ4_ROWS = ["Strict zero-shot (0%)", "Pretrain-exposed (5%)", "Pretrain-exposed (30%)", "Pretrain-exposed (100%)", "Target post-training oracle"]
rq4 = {}
for i, row in enumerate(RQ4_ROWS):
    rq4[row] = {}
    for col, tgt in enumerate(["UR5e+UMI", "GoogleRobot"]):
        runs = np.mean([[T[t][row][col][k] for k in range(3)] for t in (17, 18, 19)], axis=0)
        per_task = [round(float(np.mean(T[t][row][col])), 1) for t in (17, 18, 19)]
        rq4[row][tgt] = dict(mean=round(float(runs.mean()), 1), sd0=round(float(runs.std(ddof=0)), 1), per_task=per_task, runs=[round(float(x), 2) for x in runs])
    rq4[row]["avg"] = round((rq4[row]["UR5e+UMI"]["mean"] + rq4[row]["GoogleRobot"]["mean"]) / 2, 2)
OUT["V6_rq4"] = rq4
gain = {tgt: round(rq4[RQ4_ROWS[1]][tgt]["mean"] - rq4[RQ4_ROWS[0]][tgt]["mean"], 1) for tgt in ("UR5e+UMI", "GoogleRobot")}
OUT["V6_5pct_gain"] = dict(per_target=gain, mean=round(sum(gain.values()) / 2, 2), claimed=13.4,
                           from_fig6_text=dict(ur5eumi=(69.3, 78.6), google=(51.4, 68.9)))
# 오라클 대비 (과제별)
OUT["V6_vs_oracle_by_task"] = {tgt: {RQ4_ROWS[i][:25]: [round(rq4[RQ4_ROWS[i]][tgt]["per_task"][k] - rq4[RQ4_ROWS[4]][tgt]["per_task"][k], 1) for k in range(3)]
                                      for i in (1, 2, 3)} for tgt in ("UR5e+UMI", "GoogleRobot")}
OUT["V6_0pct_equals_rq1_best_FULL"] = all(T[t4][RQ4_ROWS[0]][c] == T[t1][r][3] for t1, t4 in ((8, 17), (9, 18), (10, 19))
                                           for c, r in enumerate(("UR5e+UMI", "GoogleRobot")))
# 표 5 일관성
T5 = {"Abs/World": (33.9, 66.9), "Abs/EEF": (38.6, 69.2), "EEF/World": (58.4, 67.5), "EEF/EEF": (60.4, 71.1)}
OUT["K_table5"] = dict(
    reported=T5,
    zero_pct_equals_table2_FULL=all(abs(T5[n][0] - REPORTED_T2[n]["FULL"][0]) < 0.05 for n in T5),
    best_5pct_reported=71.1, best_5pct_from_tables17_19=rq4[RQ4_ROWS[1]]["avg"],
    gap_zero=round(max(v[0] for v in T5.values()) - min(v[0] for v in T5.values()), 1),
    gap_five=round(max(v[1] for v in T5.values()) - min(v[1] for v in T5.values()), 1),
    gains={n: round(v[1] - v[0], 1) for n, v in T5.items()})

# ---------------------------------------------------------------- V7 실세계 (표 3, 20-23)
RR = ["Franka+Robotiq", "Franka+Robotiq (CoRL logo)", "Franka+Umi", "UR", "Humanoid robot", "UR+Umi", "Humanoid robot+Umi", "Piper-on-Quadruped"]
RCAT = {"es": ["Franka+Robotiq"], "APP": ["Franka+Robotiq (CoRL logo)"], "GRP": ["Franka+Umi"], "ARM": ["UR"],
        "FULL": ["Humanoid robot", "UR+Umi", "Humanoid robot+Umi", "Piper-on-Quadruped"]}
T3 = {"Abs/World": [87.5, 87.5, 67.5, 47.5, 21.3, 56.0], "Abs/EEF": [90.0, 90.0, 80.0, 60.0, 16.3, 61.6],
      "EEF/World": [85.0, 77.5, 60.0, 62.5, 43.1, 60.8], "EEF/EEF": [100.0, 97.5, 92.5, 87.5, 81.9, 89.9]}


def real_agg(vals_fryer, vals_water):
    """vals_*: {로봇: 값}. 반환 {es, APP, GRP, ARM, FULL, Average} (과제 평균 -> 범주 평균)."""
    o = {}
    for c, rs in RCAT.items():
        o[c] = float(np.mean([(vals_fryer[r] + vals_water[r]) / 2 for r in rs]))
    o["Average"] = float(np.mean([o[c] for c in CATS]))
    return o


v7 = {}
for j, name in enumerate(RQ1_COLS):
    f = {r: T[20][r][j] for r in RR}
    w = {r: T[21][r][j] for r in RR}
    a = real_agg(f, w)
    rep = dict(zip(["es", "APP", "GRP", "ARM", "FULL", "Average"], T3[name]))
    v7[name] = dict(computed={k: round(v, 2) for k, v in a.items()}, reported=rep,
                    max_abs_diff=round(max(abs(a[k] - rep[k]) for k in rep), 2))
OUT["V7_table3"] = v7
# 미시 평균(대상 14 셀 단순 평균) 대비
OUT["V7_micro_vs_macro"] = {name: dict(micro_targets=round(float(np.mean([T[t][r][j] for t in (20, 21) for r in RR[1:]])), 1),
                                       macro=T3[name][5]) for j, name in enumerate(RQ1_COLS)}
# 부호검정: EEF/EEF 대 다른 표현, 14개 (대상 x 과제) 셀
sign = {}
for j, name in enumerate(RQ1_COLS[:3]):
    d = [T[t][r][3] - T[t][r][j] for t in (20, 21) for r in RR[1:]]
    w_, l_, t0 = sum(x > 0 for x in d), sum(x < 0 for x in d), sum(x == 0 for x in d)
    p = stats.binomtest(w_, w_ + l_, 0.5).pvalue if w_ + l_ else 1.0
    sign[f"EEF/EEF vs {name}"] = dict(wins=w_, losses=l_, ties=t0, p_sign=float(f"{p:.2g}"))
OUT["V7_rq1_real_signtest"] = sign
# 점수 함수: 시도당 {0, .5, 1} -> 10회 평균은 5의 배수
OUT["V7_real_multiples_of_5"] = all(abs(v / 5 - round(v / 5)) < 1e-9 for t in (20, 21, 22, 23) for r in RR for v in T[t][r])

# RQ2 실세계 (표 22, 23)
rq2r = {}
for j, name in enumerate(["1", "8", "512"]):
    f = {r: T[22][r][j] for r in RR}
    w = {r: T[23][r][j] for r in RR}
    rq2r[name] = real_agg(f, w)
OUT["V7_rq2_real"] = {n: {k: round(v, 1) for k, v in a.items()} for n, a in rq2r.items()}
OUT["V7_rq2_real_nonmonotone_ARM_FULL"] = dict(ARM_1_8_512=[round(rq2r[n]["ARM"], 1) for n in ("1", "8", "512")],
                                                 FULL_1_8_512=[round(rq2r[n]["FULL"], 1) for n in ("1", "8", "512")],
                                                 Average_1_8_512=[round(rq2r[n]["Average"], 1) for n in ("1", "8", "512")])
OUT["K_rq2_real_512_equals_rq1_best"] = all(T[22][r][2] == T[20][r][3] and T[23][r][2] == T[21][r][3] for r in RR)
# 8 소스가 1 소스보다 나쁜 셀
bad = [(("fryer" if t == 22 else "water"), r, T[t][r][0], T[t][r][1]) for t in (22, 23) for r in RR if T[t][r][1] < T[t][r][0]]
OUT["V7_rq2_real_8_below_1_cells"] = bad
d = [T[t][r][2] - T[t][r][0] for t in (22, 23) for r in RR[1:]]
w_, l_ = sum(x > 0 for x in d), sum(x < 0 for x in d)
OUT["V7_rq2_real_512_vs_1_sign"] = dict(wins=w_, losses=l_, ties=sum(x == 0 for x in d), p_sign=float(f"{stats.binomtest(w_, w_ + l_, 0.5).pvalue:.2g}"))
d = [T[t][r][1] - T[t][r][0] for t in (22, 23) for r in RR[1:]]
w_, l_ = sum(x > 0 for x in d), sum(x < 0 for x in d)
OUT["V7_rq2_real_8_vs_1_sign"] = dict(wins=w_, losses=l_, ties=sum(x == 0 for x in d), p_sign=float(f"{stats.binomtest(w_, w_ + l_, 0.5).pvalue:.2g}"))


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# 합산 근사: 점수 0/.5/1 을 성공 수로 환산(0.5 = 반 성공)하면 이항 분산보다 작다 -> 보수적 근사
N = 140
pairs = [("EEF/EEF", 89.9, "Abs/EEF", 61.6), ("EEF/EEF", 89.9, "EEF/World", 60.8), ("EEF/EEF", 89.9, "Abs/World", 56.0)]
approx = {}
for a_, pa, b_, pb in pairs:
    ka, kb = round(pa / 100 * N), round(pb / 100 * N)
    approx[f"{a_} vs {b_}"] = dict(fisher_p=float(f"{stats.fisher_exact([[ka, N-ka], [kb, N-kb]])[1]:.2g}"),
                                    wilson_a=[round(x, 3) for x in wilson(ka, N)], wilson_b=[round(x, 3) for x in wilson(kb, N)])
OUT["V7_rq1_real_pooled_approx_N140"] = approx

# ---------------------------------------------------------------- V8 분모와 정수 가능성
def feas(v, n, tol=0.06):
    """v(%) 가 n 번 시도, 점수 {0,.5,1} 평균(격자 k/(2n)*100)에 소수 첫째 자리 반올림 오차(0.06) 안에서 맞는가."""
    return any(abs(k / (2 * n) * 100 - v) <= tol for k in range(2 * n + 1))


den = {"table8(sausage)": 8, "table9(bowls)": 9, "table10(cubes)": 10}
res8 = {}
for t, nm in ((8, "sausage"), (9, "bowls"), (10, "cubes")):
    vals = [x for r in T[t].values() for tr in r for x in tr]
    res8[nm] = dict(n_values=len(vals), ok_100=sum(feas(v, 100) for v in vals), ok_99=sum(feas(v, 99) for v in vals),
                    not_100=sorted({v for v in vals if not feas(v, 100)}))
# 모든 sim 표(8-19)에 대해
allv = {t: [x for r in T[t].values() for tr in r for x in tr] for t in range(8, 20)}
res8["all_tables"] = {f"table{t}": dict(n=len(v), not_feasible_100=[x for x in v if not feas(x, 100)][:12], n_not_100=sum(not feas(x, 100) for x in v),
                                        n_not_99=sum(not feas(x, 99) for x in v), n_neither=sum((not feas(x, 100)) and (not feas(x, 99)) for x in v))
                      for t, v in allv.items()}
OUT["V8_denominators"] = res8
# 100 시도 가정 시 과제별 소수 둘째 자리: 99 분모 후보
OUT["V8_note"] = "점수가 {0,.5,1}이고 시도가 100이면 값은 0.5의 배수여야 한다. 소시지 표의 80.8, 83.8, 89.9 등은 k/99에 맞는다(분모 99)."
OUT["V8_rollout_counts"] = dict(paper_sim_per_model=3 * 7 * 100 * 3, with_source_embodiment=3 * 8 * 100 * 3, paper_real_per_model=2 * 7 * 10,
                                real_with_source=2 * 8 * 10, per_run_trials_per_task_robot=100)

# ---------------------------------------------------------------- V9 초록 수치
v9 = {
    "표현 +15 (최고 대 최저 평균)": round(75.7 - 60.3, 1),
    "표현 (EEF/EEF 대 Abs/EEF, 행동 고정)": round(75.7 - 64.6, 1),
    "표현 (EEF/EEF 대 EEF/World, 상태 고정)": round(75.7 - 73.4, 1),
    "다양성 +18 (512 대 1, 표 11-13)": OUT["V4_rq2_gain_512_vs_1"]["Average"],
    "공동학습 +7 (최고, 표 4)": round(82.3 - 75.7, 1),
    "공동학습 (LAP-EEF, Subgoal, BBox)": [round(81.3 - 75.7, 1), round(81.8 - 75.7, 1), round(82.3 - 75.7, 1)],
    "노출 5% (표 17-19 재계산)": OUT["V6_5pct_gain"]["mean"],
}
OUT["V9_abstract"] = v9

# ---------------------------------------------------------------- 저장
(ROOT / "analysis" / "recompute_results.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")

if __name__ == "__main__":
    print("V1 summary", OUT["V1_summary"])
    print("V2 per robot", json.dumps(OUT["V2_rq1_per_robot"]))
    print("V2 state gain", OUT["V2_state_gain_per_robot"])
    print("V2 nonG", OUT["V2_mean_over_non_google_targets"], OUT["V2_mean_over_google_targets"], OUT["V2_main_effects_average"], OUT["V2_main_effects_non_google"])
    print("V3 rq1 welch avg", {k: v for k, v in OUT["V3_rq1_welch"].items() if "Average" in k})
    print("V4 rq2", OUT["V4_rq2_sim"], OUT["V4_rq2_gain_512_vs_1"], OUT["V4_rq2_welch_avg_512_vs_1"], OUT["K_rq2_512_equals_rq1_best"])
    print("V5 summary", OUT["V5_summary"], OUT["V5_baseline_equals_rq1_best"])
    print("V5 welch", OUT["V5_welch"])
    print("V5 runs", OUT["V5_runs_avg"])
    print("V5 gain", OUT["V5_gain_by_category"])
    print("V6", json.dumps(OUT["V6_rq4"]), OUT["V6_5pct_gain"], OUT["V6_vs_oracle_by_task"], OUT["V6_0pct_equals_rq1_best_FULL"])
    print("K5", OUT["K_table5"])
    print("V7", json.dumps(OUT["V7_table3"]), OUT["V7_micro_vs_macro"])
    print("V7 sign", OUT["V7_rq1_real_signtest"], OUT["V7_rq1_real_pooled_approx_N140"])
    print("V7 rq2", OUT["V7_rq2_real"], OUT["V7_rq2_real_nonmonotone_ARM_FULL"], OUT["K_rq2_real_512_equals_rq1_best"], OUT["V7_rq2_real_8_below_1_cells"],
          OUT["V7_rq2_real_512_vs_1_sign"], OUT["V7_rq2_real_8_vs_1_sign"], OUT["V7_real_multiples_of_5"])
    print("V8", json.dumps({k: v for k, v in OUT["V8_denominators"].items() if k != "all_tables"}))
    print("V8 all", {k: (v["n"], v["n_not_100"], v["n_not_99"], v["n_neither"], v["not_feasible_100"][:4]) for k, v in OUT["V8_denominators"]["all_tables"].items()})
    print("V9", OUT["V9_abstract"])
