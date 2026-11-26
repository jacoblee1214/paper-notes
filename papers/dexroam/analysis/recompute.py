"""DexRoam(arXiv:2609.35761 v1) 보고 수치 재계산과 일관성 검사. 결과는 recompute_results.json.

실행: python3 analysis/recompute.py   (numpy, scipy, 1초 미만)

입력은 논문 v1의 표와 그림 라벨에서 옮긴 값이다.
- 표 I–V, 그림 6, 그림 7의 막대 라벨, 그림 9의 라벨, 표 III은 본문 텍스트에서 옮겼다.
- 그림 8(로봇 시연 수 곡선)에는 숫자 라벨이 없어서 PDF의 벡터 경로 좌표를 5% 격자로 환산한 값을 아래 FIG8에 적었다(그림 판독 값).
검사 ID는 V1–V15. 논문에 없는 시행 수는 가정으로 표시한다(정책 평가는 본문 IV-A2의 "과제당 20회"를 쓴다).
V15만 공식 코드 clone을 읽는다. 이 clone은 아래 명령으로 만든 것과 같아야 한다.
    git clone --depth 1 https://github.com/zhourui9813/DexRoam-Policy-Training.git repo_policy
    git -C repo_policy checkout a82d2c98dc0dff00c17e1aa4fb2bb93d1c5655fe
repo_policy가 없으면 V15는 건너뛴다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "recompute_results.json"
R = {}
N_TRIALS = 20  # 본문 IV-A2: 과제당 20회


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def cmh(tables):
    """층화 2x2 표의 Cochran-Mantel-Haenszel 검정(연속성 보정 없음). tables: [(a,b,c,d), ...]"""
    num, var = 0.0, 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        r1, c1 = a + b, a + c
        e = r1 * c1 / n
        v = r1 * (c + d) * c1 * (b + d) / (n * n * (n - 1))
        num += a - e
        var += v
    chi2 = num * num / var
    return float(chi2), float(stats.chi2.sf(chi2, 1))


# ---------------------------------------------------------------- 표 값
TASKS = ["Pick Chips Can", "Throw Trash", "Deliver Fruit", "Push Chair & Close Laptop", "Pour Water"]
STRATS = ["Robot-only", "Cotrain", "Pretrain+RobotFT", "Pretrain+CotrainFT"]
# 그림 7 막대 라벨. 순서: 위 과제 순서 x 전략 순서. (SR, TCS)
SR = {
    "GR00T": [[.50, .70, .65, .90], [.35, .60, .55, .65], [.40, .55, .60, .75], [.10, .20, .15, .25], [.10, .25, .15, .25]],
    "pi05": [[.40, .50, .75, .75], [.35, .65, .55, .65], [.60, .70, .70, .75], [.20, .30, .35, .40], [.05, .30, .10, .30]],
}
SR_AVG = {"GR00T": [.29, .46, .42, .56], "pi05": [.32, .49, .49, .57]}
TCS = {
    "GR00T": [[.63, .86, .74, .96], [.67, .80, .80, .86], [.80, .83, .83, .92], [.49, .64, .64, .71], [.19, .40, .33, .42]],
    "pi05": [[.60, .66, .85, .84], [.59, .81, .79, .84], [.74, .86, .90, .89], [.48, .61, .59, .62], [.19, .42, .34, .44]],
}
TCS_AVG = {"GR00T": [.56, .70, .67, .77], "pi05": [.52, .67, .69, .73]}
# 부록 표 X: 과제별 단계 가중치(합 1.0)
TCS_W = {
    "Pick Chips Can": [.2, .2, .2, .2, .2],
    "Pour Water": [.1, .2, .2, .2, .3],
    "Throw Trash": [.2, .2, .2, .3, .1],
    "Deliver Fruit": [.2, .2, .2, .2, .2],
    "Push Chair & Close Laptop": [.2, .2, .2, .2, .2],
}
TEXT_GAINS = {"GR00T": [17, 13, 27], "pi05": [17, 17, 25]}  # IV-C 본문 %p
# 표 III (GR00T SR): [w/o Temporal, Absolute, DexRoam], 전략 순서 Cotrain, Pretrain+FT, Pretrain+Cotrain FT
T3 = {"Pick Chips Can": [[.65, .00, .70], [.30, .00, .65], [.20, .00, .90]],
      "Throw Trash": [[.40, .00, .65], [.50, .00, .55], [.60, .00, .65]]}
# 그림 6
MMD = {"Pick Chips Can": (.598, .313), "Pour Water": (.631, .384), "Throw Trash": (.628, .359), "Deliver Fruit": (.585, .413),
       "Push Chair & Close Laptop": (.553, .274)}
SWD = {"Pick Chips Can": (.926, .563), "Pour Water": (1.277, .860), "Throw Trash": (1.004, .729), "Deliver Fruit": (1.008, .684),
       "Push Chair & Close Laptop": (.940, .719)}
# 표 I (Pick, Pour, Throw, 평균), 표 IV (PICO), 표 II
T1 = {"Torso pos": ([13.49, 11.82, 12.31], 12.54), "Torso rot": ([2.32, 1.64, 1.57], 1.84),
      "Left pos": ([3.75, 4.27, 4.02], 4.01), "Left rot": ([.51, .58, .57], .55),
      "Right pos": ([5.30, 5.93, 7.67], 6.30), "Right rot": ([.70, .84, 1.22], .92), "Head rot": ([1.22, 1.36, 1.53], 1.37)}
T4 = {"Torso pos": ([14.76, 11.50, 13.22], 13.16), "Torso rot": ([1.71, 2.74, 1.74], 2.06),
      "Left pos": ([3.85, 4.11, 4.41], 4.12), "Left rot": ([.54, .58, .60], .57),
      "Right pos": ([5.44, 5.45, 6.02], 5.63), "Right rot": ([.79, .79, .85], .81), "Head rot": ([1.27, 1.32, 1.09], 1.23)}
T2_FREE = {"Torso pos": 12.54, "Torso rot": 1.84, "Left pos": 4.01, "Left rot": .55, "Right pos": 6.30, "Right rot": .92, "Head rot": 1.37}
T2_PICO = {"Torso pos": 13.16, "Torso rot": 2.06, "Left pos": 4.12, "Left rot": .57, "Right pos": 5.63, "Right rot": .81, "Head rot": 1.23}
# 표 V (retargeted human, teleop robot) x (LDLJ, SPARC)
T5 = {"Pick Chips Can": ((-20.2802, -5.8049), (-20.4726, -7.0548)), "Pour Water": ((-20.5962, -5.9222), (-20.4183, -6.5718)),
      "Throw Trash": ((-20.8349, -6.6556), (-20.3022, -6.9229)), "Deliver Fruit": ((-20.3453, -5.4803), (-19.8355, -6.2193)),
      "Push Chair & Close Laptop": ((-20.5130, -7.2351), (-20.5575, -8.4923))}
T5_AVG = ((-20.5139, -6.2196), (-20.3172, -7.0522))
# 그림 9 (Pick Chips Can, %): Origin, Novel Scene, Novel Instruction; 전략 4개. 라벨의 괄호는 원점 대비 상대 하락(%)
FIG9 = {"Robot-only": (50, 20, 35, 60, 30), "Cotrain": (85, 35, 65, 59, 24), "Pretrain+RobotFT": (60, 30, 50, 50, 17),
        "Pretrain+CotrainFT": (90, 55, 85, 39, 6)}
# 그림 8: 벡터 경로에서 읽은 값(%). 로봇 시연 수 {0, 10, 25, 50}. Robot-only는 50에서만.
FIG8 = {
    "GR00T": {"Pick Chips Can": {"Cotrain": [0, 40, 50, 70], "Pretrain+RobotFT": [0, 35, 50, 65], "Robot-only50": 50},
              "Throw Trash": {"Cotrain": [0, 40, 50, 60], "Pretrain+RobotFT": [0, 50, 70, 55], "Robot-only50": 35},
              "Deliver Fruit": {"Cotrain": [0, 20, 50, 55], "Pretrain+RobotFT": [0, 25, 65, 60], "Robot-only50": 40}},
    "pi05": {"Pick Chips Can": {"Cotrain": [0, 20, 40, 50], "Pretrain+RobotFT": [0, 50, 35, 75], "Robot-only50": 40},
             "Throw Trash": {"Cotrain": [0, 50, 45, 60], "Pretrain+RobotFT": [0, 20, 30, 50], "Robot-only50": 35},
             "Deliver Fruit": {"Cotrain": [0, 50, 55, 70], "Pretrain+RobotFT": [0, 35, 60, 70], "Robot-only50": 60}},
}


def r2(x, nd=3):
    return round(float(x), nd)


# ---------------------------------------------------------------- V1: 표 I 평균, 표 IV 평균, 표 II 대응
v1 = {}
for name, (vals, rep) in T1.items():
    m = float(np.mean(vals))
    v1[f"TableI {name}"] = {"values": vals, "reported_mean": rep, "recomputed": r2(m, 4), "match_2dp": abs(round(m, 2) - rep) < 0.0051}
for name, (vals, rep) in T4.items():
    m = float(np.mean(vals))
    v1[f"TableIV {name}"] = {"values": vals, "reported_mean": rep, "recomputed": r2(m, 4), "match_2dp": abs(round(m, 2) - rep) < 0.0051}
v1["TableII_equals_TableI_avg"] = all(T2_FREE[k] == T1[k][1] for k in T1)
v1["TableII_pico_equals_TableIV_avg"] = all(T2_PICO[k] == T4[k][1] for k in T4)
v1["mismatch_cells"] = [k for k, v in v1.items() if isinstance(v, dict) and not v["match_2dp"]]
v1["tracker_diff"] = {k: r2(T2_FREE[k] - T2_PICO[k], 2) for k in T2_FREE}
R["V1"] = v1

# ---------------------------------------------------------------- V2: 그림 6 감소율
v2 = {"MMD": {}, "SWD": {}}
for label, D in (("MMD", MMD), ("SWD", SWD)):
    red = {t: 1 - a / r for t, (r, a) in D.items()}
    mean_raw = np.mean([r for r, a in D.values()])
    mean_al = np.mean([a for r, a in D.values()])
    v2[label] = {"per_task_reduction": {t: r2(v, 4) for t, v in red.items()},
                 "mean_of_per_task_reductions": r2(np.mean(list(red.values())), 4),
                 "reduction_of_means": r2(1 - mean_al / mean_raw, 4),
                 "mean_raw": r2(mean_raw, 4), "mean_aligned": r2(mean_al, 4),
                 "min_reduction": r2(min(red.values()), 4), "max_reduction": r2(max(red.values()), 4)}
v2["reported"] = {"MMD": .417, "SWD": .311}
# 라벨이 소수 셋째 자리로 반올림된 것을 고려한 평균 감소율의 범위
for label, D in (("MMD", MMD), ("SWD", SWD)):
    lo = np.mean([1 - (a + .0005) / (r - .0005) for r, a in D.values()])
    hi = np.mean([1 - (a - .0005) / (r + .0005) for r, a in D.values()])
    v2[label]["mean_reduction_rounding_range"] = [r2(lo, 4), r2(hi, 4)]
R["V2"] = v2

# ---------------------------------------------------------------- V3: 그림 7 평균, 본문 %p, 격자
v3 = {}
for bb in SR:
    avg = np.mean(np.array(SR[bb]), axis=0)
    tavg = np.mean(np.array(TCS[bb]), axis=0)
    v3[bb] = {
        "SR_avg_recomputed": [r2(x, 4) for x in avg], "SR_avg_reported": SR_AVG[bb],
        "SR_avg_match": [abs(round(a, 2) - b) < 0.0051 for a, b in zip(avg, SR_AVG[bb])],
        "TCS_avg_recomputed": [r2(x, 4) for x in tavg], "TCS_avg_reported": TCS_AVG[bb],
        "TCS_avg_match": [abs(round(a, 2) - b) < 0.0051 for a, b in zip(tavg, TCS_AVG[bb])],
        "gain_pp_recomputed": [r2((avg[i] - avg[0]) * 100, 1) for i in (1, 2, 3)], "gain_pp_text": TEXT_GAINS[bb],
        "SR_on_0.05_grid": all(abs(x * 20 - round(x * 20)) < 1e-9 for row in SR[bb] for x in row),
    }
v3["best_push_gain_pp"] = {bb: max(SR[bb][3][i] for i in (1, 2, 3)) * 100 - SR[bb][3][0] * 100 for bb in SR}
v3["best_push_gain_text"] = {"GR00T": 15, "pi05": 20}
R["V3"] = v3

# ---------------------------------------------------------------- V4: 그림 7 문장 검사(개선, 최선/동률)
v4 = {"all_three_improve_SR": {}, "pre_cot_best_or_tied_SR": {}, "pre_cot_best_or_tied_TCS": {}, "exceptions_TCS": []}
for bb in SR:
    imp = [all(row[i] > row[0] for i in (1, 2, 3)) for row in SR[bb]]
    v4["all_three_improve_SR"][bb] = imp
    v4["pre_cot_best_or_tied_SR"][bb] = [row[3] >= max(row) for row in SR[bb]]
    v4["pre_cot_best_or_tied_TCS"][bb] = [row[3] >= max(row) for row in TCS[bb]]
    for t, row in zip(TASKS, TCS[bb]):
        if row[3] < max(row):
            v4["exceptions_TCS"].append({"backbone": bb, "task": t, "values": row})
    imp_t = [all(row[i] > row[0] for i in (1, 2, 3)) for row in TCS[bb]]
    v4.setdefault("all_three_improve_TCS", {})[bb] = imp_t
R["V4"] = v4

# ---------------------------------------------------------------- V5: TCS와 SR의 정수·경계 일관성
v5 = {"cells": [], "violations_lower": 0, "violations_upper": 0}
for bb in SR:
    for t, srow, trow in zip(TASKS, SR[bb], TCS[bb]):
        w_last = TCS_W[t][-1]
        for s_name, s, c in zip(STRATS, srow, trow):
            lower = s  # 성공 시행 TCS=1, 실패 >= 0
            upper = s + (1 - s) * (1 - w_last)  # 실패 시행은 마지막 단계 미달(>= 마지막 가중치만큼 부족)
            ok_l = c >= lower - 1e-9
            ok_u = c <= upper + 1e-9
            v5["cells"].append({"backbone": bb, "task": t, "strategy": s_name, "SR": s, "TCS": c, "lower": r2(lower, 3), "upper": r2(upper, 3),
                                "ok_lower": ok_l, "ok_upper": ok_u})
            v5["violations_lower"] += (not ok_l)
            v5["violations_upper"] += (not ok_u)
v5["weights_sum"] = {t: r2(sum(w), 3) for t, w in TCS_W.items()}
v5["note"] = "성공=모든 단계 완료로 가정. Throw Trash는 5단계(0.1)가 '쓰레기통을 건드리지 않음'이라 성공 정의가 불명확해 상한 검사는 보수적(0.9)"
R["V5"] = v5

# ---------------------------------------------------------------- V6: 그림 7 유의성 (과제당 20회 가정)
v6 = {"per_task": [], "pooled": {}, "pairwise_pooled": {}}
all_p = []
rows = []
for bb in SR:
    for t, srow in zip(TASKS, SR[bb]):
        k0 = round(srow[0] * N_TRIALS)
        for i in (1, 2, 3):
            k = round(srow[i] * N_TRIALS)
            p = fisher(k, N_TRIALS, k0, N_TRIALS)
            rows.append({"backbone": bb, "task": t, "strategy": STRATS[i], "k": k, "k_robot": k0, "p": p})
            all_p.append(p)
adj = holm(all_p)
for r_, a in zip(rows, adj):
    r_["p"] = float(f"{r_['p']:.4g}")
    r_["p_holm_30"] = float(f"{a:.4g}")
v6["per_task"] = rows
v6["n_per_task_sig_raw05"] = sum(1 for r_ in rows if r_["p"] < 0.05)
v6["n_per_task_sig_holm05"] = sum(1 for r_ in rows if r_["p_holm_30"] < 0.05)
for bb in SR:
    ks = [sum(round(SR[bb][t][i] * N_TRIALS) for t in range(5)) for i in range(4)]
    n = 5 * N_TRIALS
    v6["pooled"][bb] = {"k": ks, "n": n, "rate": [r2(k / n, 3) for k in ks],
                        "wilson95": [[r2(x, 3) for x in wilson(k, n)] for k in ks],
                        "fisher_vs_robot": [None] + [float(f"{fisher(ks[i], n, ks[0], n):.3g}") for i in (1, 2, 3)]}
    # 과제 층화(CMH), 전략 i 대 Robot-only
    cm = []
    for i in (1, 2, 3):
        tabs = []
        for t in range(5):
            a = round(SR[bb][t][i] * N_TRIALS)
            c = round(SR[bb][t][0] * N_TRIALS)
            tabs.append((a, N_TRIALS - a, c, N_TRIALS - c))
        chi2, p = cmh(tabs)
        cm.append({"strategy": STRATS[i], "chi2": r2(chi2, 2), "p": float(f"{p:.3g}")})
    v6["pooled"][bb]["cmh_vs_robot"] = cm
    # 전략 쌍 비교(풀링)
    pw = {}
    for i, j in ((3, 1), (3, 2), (1, 2)):
        pw[f"{STRATS[i]} vs {STRATS[j]}"] = float(f"{fisher(ks[i], n, ks[j], n):.3g}")
    v6["pairwise_pooled"][bb] = pw
# 두 백본 합산(200 대 200)
ks_all = [sum(sum(round(SR[bb][t][i] * N_TRIALS) for t in range(5)) for bb in SR) for i in range(4)]
v6["both_backbones"] = {"k": ks_all, "n": 200, "fisher_precot_vs_robot": float(f"{fisher(ks_all[3], 200, ks_all[0], 200):.3g}"),
                        "wilson95": [[r2(x, 3) for x in wilson(k, 200)] for k in ks_all]}
R["V6"] = v6

# ---------------------------------------------------------------- V7: 표 III 평균, 칸별 검정
v7 = {"cells": []}
flat_dex, flat_wo = [], []
for t, rows3 in T3.items():
    for strat, (wo, ab, dx) in zip(["Cotrain", "Pretrain+RobotFT", "Pretrain+CotrainFT"], rows3):
        k_wo, k_ab, k_dx = (round(x * N_TRIALS) for x in (wo, ab, dx))
        i = STRATS.index(strat)
        ref = round(SR["GR00T"][TASKS.index(t)][0] * N_TRIALS)
        flat_dex.append(k_dx)
        flat_wo.append(k_wo)
        v7["cells"].append({
            "task": t, "strategy": strat, "k_wo_temporal": k_wo, "k_dexroam": k_dx, "k_absolute": k_ab, "k_robot_only_fig7": ref,
            "diff_pp": (k_dx - k_wo) * 5, "p_dexroam_vs_wo": float(f"{fisher(k_dx, 20, k_wo, 20):.3g}"),
            "p_wo_vs_robot_only": float(f"{fisher(k_wo, 20, ref, 20):.3g}"),
            "wo_below_robot_only": k_wo < ref,
            "dexroam_equals_fig7": abs(dx - SR["GR00T"][TASKS.index(t)][i]) < 1e-9,
        })
v7["mean_dexroam"] = r2(np.mean([c["k_dexroam"] for c in v7["cells"]]) / 20, 4)
v7["mean_wo"] = r2(np.mean([c["k_wo_temporal"] for c in v7["cells"]]) / 20, 4)
v7["drop_exact_pp"] = r2((v7["mean_dexroam"] - v7["mean_wo"]) * 100, 2)
v7["reported"] = {"dexroam": 68.3, "wo": 44.2, "drop": 24.1}
v7["pooled_k"] = {"dexroam": sum(flat_dex), "wo": sum(flat_wo), "n": 120}
v7["pooled_fisher"] = float(f"{fisher(sum(flat_dex), 120, sum(flat_wo), 120):.3g}")
v7["pooled_wilson_wo"] = [r2(x, 3) for x in wilson(sum(flat_wo), 120)]
v7["pooled_wilson_dexroam"] = [r2(x, 3) for x in wilson(sum(flat_dex), 120)]
v7["absolute_zero_wilson_upper_n20"] = r2(wilson(0, 20)[1], 3)
v7["absolute_zero_wilson_upper_n120"] = r2(wilson(0, 120)[1], 3)
# 과제별 층화(전략 3개씩 풀링)
tabs = []
for t in T3:
    a = sum(round(x[2] * 20) for x in T3[t]); b = sum(round(x[0] * 20) for x in T3[t])
    tabs.append((a, 60 - a, b, 60 - b))
v7["cmh_by_task"] = dict(zip(["chi2", "p"], [r2(x, 3) for x in cmh(tabs)]))
v7["n_cells_diff_le_5pp"] = sum(1 for c in v7["cells"] if abs(c["diff_pp"]) <= 5)
R["V7"] = v7

# ---------------------------------------------------------------- V8: 그림 8과 그림 7의 50개 시연 칸 비교, 25개 시연 주장
v8 = {"mismatch_with_fig7": [], "claim_25_vs_robot50": []}
for bb in FIG8:
    for t, d in FIG8[bb].items():
        ti = TASKS.index(t)
        for strat in ("Cotrain", "Pretrain+RobotFT"):
            si = STRATS.index(strat)
            v8_ = d[strat][3]
            f7 = SR[bb][ti][si] * 100
            if abs(v8_ - f7) > 1e-6:
                v8["mismatch_with_fig7"].append({"backbone": bb, "task": t, "strategy": strat, "fig8_at50": v8_, "fig7": f7})
        if abs(d["Robot-only50"] - SR[bb][ti][0] * 100) > 1e-6:
            v8["mismatch_with_fig7"].append({"backbone": bb, "task": t, "strategy": "Robot-only", "fig8": d["Robot-only50"], "fig7": SR[bb][ti][0] * 100})
        for strat in ("Cotrain", "Pretrain+RobotFT"):
            v25 = d[strat][2]
            ref = d["Robot-only50"]
            k25, k50 = round(v25 / 5), round(ref / 5)
            v8["claim_25_vs_robot50"].append({"backbone": bb, "task": t, "strategy": strat, "at25": v25, "robot50": ref,
                                              "diff_pp": v25 - ref, "fisher_p": float(f"{fisher(k25, 20, k50, 20):.3g}")})
v8["greater_or_equal_GR00T"] = all(c["diff_pp"] >= 0 for c in v8["claim_25_vs_robot50"] if c["backbone"] == "GR00T")
v8["worst_diff_pi05"] = min(c["diff_pp"] for c in v8["claim_25_vs_robot50"] if c["backbone"] == "pi05")
v8["zero_robot_all_zero"] = all(d[s][0] == 0 for bb in FIG8 for d in FIG8[bb].values() for s in ("Cotrain", "Pretrain+RobotFT"))
# 총 시연 수: 25 로봇 + 50 사람 = 75 대 로봇 50
v8["demo_counts"] = {"human_assisted_25": {"robot": 25, "human": 50, "total": 75}, "robot_only_50": {"robot": 50, "human": 0, "total": 50}}
R["V8"] = v8

# ---------------------------------------------------------------- V9: 그림 9 상대 하락과 그림 7 비교
v9 = {"relative_drop": {}, "vs_fig7": {}}
for s, (o, sc, ins, d_sc, d_ins) in FIG9.items():
    v9["relative_drop"][s] = {"scene_recomputed": r2((o - sc) / o * 100, 1), "scene_label": d_sc,
                              "instr_recomputed": r2((o - ins) / o * 100, 1), "instr_label": d_ins,
                              "scene_ok": abs(round((o - sc) / o * 100) - d_sc) <= 1, "instr_ok": abs(round((o - ins) / o * 100) - d_ins) <= 1}
for bb in SR:
    v9["vs_fig7"][bb] = {s: {"fig9_origin": FIG9[s][0], "fig7": SR[bb][0][STRATS.index(s)] * 100,
                             "equal": abs(FIG9[s][0] - SR[bb][0][STRATS.index(s)] * 100) < 1e-6} for s in FIG9}
v9["scene_fisher_robot_vs_precot"] = float(f"{fisher(11, 20, 4, 20):.3g}")
v9["instr_fisher_robot_vs_precot"] = float(f"{fisher(17, 20, 7, 20):.3g}")
v9["scene_fisher_cotrain_vs_robot"] = float(f"{fisher(7, 20, 4, 20):.3g}")
v9["ratio_of_changes"] = "신분 확인용: 원점 90 대 장면 55는 18/20 대 11/20"
R["V9"] = v9

# ---------------------------------------------------------------- V10: 표 V 평균과 본문
v10 = {}
for j, name in enumerate(("retargeted_human", "teleop_robot")):
    ld = np.mean([v[j][0] for v in T5.values()])
    sp = np.mean([v[j][1] for v in T5.values()])
    v10[name] = {"LDLJ_mean": r2(ld, 4), "SPARC_mean": r2(sp, 4), "reported": T5_AVG[j]}
v10["sparc_human_higher_all_five"] = all(v[0][1] > v[1][1] for v in T5.values())
v10["ldlj_human_higher_count"] = sum(1 for v in T5.values() if v[0][0] > v[1][0])
v10["ldlj_diff"] = r2(T5_AVG[0][0] - T5_AVG[1][0], 4)
R["V10"] = v10

# ---------------------------------------------------------------- V11: 표 VI, VII 차원 합과 표 VIII, IX 일관성
v11 = {
    "state_dims": [9, 9, 9, 12, 12, 2], "state_sum": 9 + 9 + 9 + 12 + 12 + 2,
    "action_dims": [12, 12, 9, 9, 9, 2, 3], "action_sum": 12 + 12 + 9 + 9 + 9 + 2 + 3,
    "pi05": {"gpus": 2, "per_gpu": 2, "accum": 1, "global_batch_reported": 4, "global_batch_computed": 2 * 2 * 1,
             "warmup_steps": 1000, "max_steps": 30000, "warmup_ratio_reported": .033, "warmup_ratio_computed": r2(1000 / 30000, 4)},
    "gr00t": {"gpus": 4, "per_gpu": 6, "accum": 1, "global_batch_reported": 24, "global_batch_computed": 4 * 6 * 1,
              "warmup_steps": 2000, "max_steps": 40000, "warmup_ratio_reported": .05, "warmup_ratio_computed": r2(2000 / 40000, 4)},
}
R["V11"] = v11

# ---------------------------------------------------------------- V12: 학습량 비교(에폭 환산) - 시연 수와 평균 길이 가정
# 논문에 시연당 길이가 없다. 데이터셋 카드: Push Chair 로봇 에피소드 50개. 길이는 미상이므로 스텝 수 대 표본 수의 관계만 기록
v12 = {"pi05_samples_seen": 30000 * 4, "gr00t_samples_seen": 40000 * 24,
       "ratio_gr00t_over_pi05": r2(40000 * 24 / (30000 * 4), 2),
       "note": "같은 시연 수에서 GR00T는 π0.5보다 8배 많은 표본을 본다(스텝 x 전역 배치). 두 백본의 절대 성공률을 비교할 때 학습량이 같지 않다"}
R["V12"] = v12

# ---------------------------------------------------------------- V13: 시행 수 20으로 본 최소 구분 차이(검정력)
v13 = {}
for p0 in (.10, .30, .50):
    grid = []
    for d in range(1, 11):
        k0 = round(p0 * 20)
        k1 = k0 + d
        if k1 > 20:
            break
        grid.append({"k_base": k0, "k_new": k1, "diff_pp": d * 5, "fisher_p": float(f"{fisher(k1, 20, k0, 20):.3g}")})
    v13[f"base_{p0}"] = grid
# 정확 검정으로 p<0.05가 되는 최소 차이(기준 성공 횟수별)
min_diff = {}
for k0 in range(0, 20):
    for k1 in range(k0 + 1, 21):
        if fisher(k1, 20, k0, 20) < 0.05:
            min_diff[k0] = (k1 - k0) * 5
            break
v13["min_detectable_pp_by_base_k"] = min_diff
R["V13"] = v13

# ---------------------------------------------------------------- V14: 실제 사용된 시연 수의 부재
R["V14"] = {"stated": {"Q3_human_demos_per_task": 50, "Q3_robot_demos": [0, 10, 25, 50], "Q2_counts": "본문 IV-C에 없음",
                       "replay_trajectories": "3개 과제 x 5 = 15", "policy_trials": "과제당 20", "dataset_card_robot_episodes_push_chair": 50},
            "note": "Q2의 로봇·사람 시연 수는 본문에 명시되지 않았고, Q3 설정(50 사람)과 Fig 8의 50 로봇 값이 Fig 7과 일치하는 칸이 많아 50/50으로 추정"}

# ---------------------------------------------------------------- V15: 공식 코드 상수와 논문 표 대조
repo = ROOT / "repo_policy"
v15 = {"available": repo.exists()}
if repo.exists():
    import re
    t = (repo / "dexroam_gr00t/scripts/run_train_gr00t.sh").read_text(encoding="utf-8")
    g = lambda k: re.search(k + r"=(\d+)", t).group(1)
    v15["gr00t_script"] = {"MAX_STEPS": int(g("MAX_STEPS")), "GLOBAL_BATCH_SIZE": int(g("GLOBAL_BATCH_SIZE")),
                           "CUDA_VISIBLE_DEVICES_count": 1 if 'CUDA_VISIBLE_DEVICES=0"' in t else None}
    v15["paper_table_IX"] = {"max_steps": 40000, "global_batch": 24, "gpus": 4}
    c = (repo / "dexroam_pi05/src/openpi/training/dexroam_train_config.py").read_text(encoding="utf-8")
    v15["pi05_config"] = {"action_horizon": int(re.search(r"action_horizon=(\d+)", c).group(1)), "batch_size": int(re.search(r"batch_size=(\d+)", c).group(1)),
                          "num_train_steps_default": 30000}
R["V15"] = v15

OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
print("wrote", OUT)
print("V1 mismatch:", R["V1"]["mismatch_cells"])
print("V2:", {k: (v["mean_of_per_task_reductions"], v["reduction_of_means"]) for k, v in R["V2"].items() if k != "reported"})
print("V3 SR avg match:", {bb: R["V3"][bb]["SR_avg_match"] for bb in SR})
print("V4 exceptions TCS:", R["V4"]["exceptions_TCS"])
print("V5 violations:", R["V5"]["violations_lower"], R["V5"]["violations_upper"])
print("V6 per-task sig raw/holm:", R["V6"]["n_per_task_sig_raw05"], R["V6"]["n_per_task_sig_holm05"])
print("V7:", R["V7"]["mean_dexroam"], R["V7"]["mean_wo"], R["V7"]["drop_exact_pp"], R["V7"]["pooled_fisher"])
print("V8 mismatch:", R["V8"]["mismatch_with_fig7"])
print("V9 vs fig7:", {bb: {s: v["equal"] for s, v in d.items()} for bb, d in R["V9"]["vs_fig7"].items()})
