"""HumanoidNDM (arXiv:2610.34126) 표 수치 재계산 V1-V14.

실행: python3 recompute.py  -> recompute_results.json (수 초, numpy/scipy만 사용)
시행 수: 표 2, 3, 7은 5회(본문 표 2 캡션), 표 4는 3회(5.2절)로 읽고, 표 5는 미기재라 3회로 가정한다.
"""
import json
import math
import os
import sys
from itertools import product

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table_data import (DIST, METRICS, MOTIONS, T2, T3, T4, T5, T7)

OUT = {}


def welch(m1, s1, n1, m2, s2, n2):
    """요약 통계 Welch t 검정 (양측). SD가 둘 다 0이면 None."""
    se2 = s1 ** 2 / n1 + s2 ** 2 / n2
    if se2 == 0:
        return None
    t = (m1 - m2) / math.sqrt(se2)
    df = se2 ** 2 / ((s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1))
    return {"t": t, "df": df, "p": 2 * stats.t.sf(abs(t), df)}


def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i])
    m = len(ps)
    adj = [0.0] * m
    run = 0.0
    for r, i in enumerate(idx):
        run = max(run, min(1.0, (m - r) * ps[i]))
        adj[i] = run
    return adj


def cells(T, motions):
    return [(mo, d, me) for mo in motions for d in DIST for me in METRICS]


# ---------------------------------------------------------------- V1: "60개 중 40개 최고"
def count_best(T, motions, include_ppo, tol):
    """tol=0: 표시된 값 기준 최솟값(동률 포함), 반환 NDM이 최솟값(동률 포함)인 칸 수"""
    methods = ["PPO", "Direct", "ASAP", "RWM", "NDM"] if include_ppo else ["Direct", "ASAP", "RWM", "NDM"]
    n_best = 0
    n_strict = 0
    per_motion = {}
    for mo in motions:
        k = 0
        for d, me in product(DIST, METRICS):
            vals = {m: T[(mo, m, d, me)][0] for m in methods}
            mn = min(vals.values())
            if vals["NDM"] <= mn + tol:
                n_best += 1
                k += 1
            others = [v for m, v in vals.items() if m != "NDM"]
            if vals["NDM"] < min(others):
                n_strict += 1
        per_motion[mo] = k
    return n_best, n_strict, per_motion


v1 = {}
for inc in (False, True):
    for tol in (0.0, 0.005):
        b, s, pm = count_best(T2, MOTIONS, inc, tol)
        v1[f"include_ppo={inc},tol={tol}"] = {"ndm_best_incl_ties": b, "ndm_strictly_best": s, "per_motion": pm}
# 5개 동작 x 3 교란 = 15 설정, 4지표 = 60칸
OUT["V1_best_count"] = v1

# ---------------------------------------------------------------- V2: 정규화 개선율 (30/47, 23/27, 9/35)
def reductions(T, motions):
    res = {}
    for meth in ("ASAP", "RWM", "NDM", "PPO"):
        out = {}
        for me in ("link_pos", "link_rot"):
            ratios = []
            sm, sd = 0.0, 0.0
            for mo in motions:
                for d in DIST:
                    a = T[(mo, meth, d, me)][0]
                    b = T[(mo, "Direct", d, me)][0]
                    ratios.append(a / b)
                    sm += a
                    sd += b
            out[me] = {"mean_of_ratios_reduction_pct": 100 * (1 - float(np.mean(ratios))),
                       "ratio_of_means_reduction_pct": 100 * (1 - sm / sd),
                       "median_ratio_reduction_pct": 100 * (1 - float(np.median(ratios)))}
        res[meth] = out
    return res


OUT["V2_reduction_vs_direct"] = reductions(T2, MOTIONS)
OUT["V2_reduction_vs_direct_with_jump"] = reductions({**T2, **T7}, MOTIONS + ["jump"])
OUT["V2_paper_claim"] = {"NDM": [30, 47], "ASAP": [23, 27], "RWM": [9, 35]}

# ---------------------------------------------------------------- V3: NDM 대 PPO w/ Disturb. ("on par")
ratios = {me: [] for me in METRICS}
nd_better_or_within10 = 0
nd_better = 0
tot = 0
for mo in MOTIONS:
    for d in DIST:
        for me in METRICS:
            a = T2[(mo, "NDM", d, me)][0]
            p = T2[(mo, "PPO", d, me)][0]
            ratios[me].append(a / p)
            tot += 1
            nd_better += a < p
            nd_better_or_within10 += a <= 1.10 * p
OUT["V3_ndm_vs_ppo"] = {
    "cells": tot, "ndm_lower": nd_better, "ndm_lower_or_within_10pct": nd_better_or_within10,
    "mean_ratio_by_metric": {me: float(np.mean(v)) for me, v in ratios.items()},
    "median_ratio_by_metric": {me: float(np.median(v)) for me, v in ratios.items()},
}
# 기준(PPO w/ Disturb.)이 Direct Transfer보다 나쁜 칸
ppo_worse = [(mo, d, me, T2[(mo, 'PPO', d, me)][0], T2[(mo, 'Direct', d, me)][0])
             for mo in MOTIONS for d in DIST for me in METRICS
             if T2[(mo, 'PPO', d, me)][0] > T2[(mo, 'Direct', d, me)][0]]
OUT["V3_ppo_worse_than_direct"] = ppo_worse

# ---------------------------------------------------------------- V4: 칸별 Welch (NDM 대 최선 기준선), n=5
rows = []
for mo in MOTIONS:
    for d in DIST:
        for me in METRICS:
            nd = T2[(mo, "NDM", d, me)]
            base = {m: T2[(mo, m, d, me)] for m in ("ASAP", "RWM")}
            bm = min(base, key=lambda m: base[m][0])
            b = base[bm]
            w = welch(nd[0], nd[1], 5, b[0], b[1], 5)
            rows.append({"motion": mo, "dist": d, "metric": me, "ndm": nd[0], "best_base": bm, "base": b[0],
                         "diff": nd[0] - b[0], "p": None if w is None else w["p"]})
valid = [r for r in rows if r["p"] is not None]
adj = holm([r["p"] for r in valid])
for r, a in zip(valid, adj):
    r["p_holm"] = a
sig_better = [r for r in valid if r["diff"] < 0 and r["p_holm"] < 0.05]
sig_worse = [r for r in valid if r["diff"] > 0 and r["p_holm"] < 0.05]
OUT["V4_ndm_vs_best_baseline"] = {
    "cells": len(rows), "testable": len(valid),
    "ndm_lower_mean": sum(r["diff"] < 0 for r in rows),
    "ndm_lower_p05_raw": sum(1 for r in valid if r["diff"] < 0 and r["p"] < 0.05),
    "ndm_lower_holm05": len(sig_better), "ndm_higher_holm05": len(sig_worse),
    "per_motion_holm_better": {mo: sum(1 for r in sig_better if r["motion"] == mo) for mo in MOTIONS},
    "per_motion_holm_worse": {mo: sum(1 for r in sig_worse if r["motion"] == mo) for mo in MOTIONS},
    "rows": rows,
}

# ---------------------------------------------------------------- V5: NDM 대 Direct (평균 개선, 표준편차 0 칸 처리)
nd_vs_dir = {"lower": 0, "higher": 0, "equal": 0}
for mo in MOTIONS:
    for d in DIST:
        for me in METRICS:
            a = T2[(mo, "NDM", d, me)][0]
            b = T2[(mo, "Direct", d, me)][0]
            nd_vs_dir["lower" if a < b else "higher" if a > b else "equal"] += 1
OUT["V5_ndm_vs_direct_cells"] = nd_vs_dir
worse_than_direct = [(mo, d, me, T2[(mo, 'NDM', d, me)][0], T2[(mo, 'Direct', d, me)][0])
                     for mo in MOTIONS for d in DIST for me in METRICS
                     if T2[(mo, 'NDM', d, me)][0] >= T2[(mo, 'Direct', d, me)][0]]
OUT["V5_ndm_not_better_than_direct"] = worse_than_direct

# ---------------------------------------------------------------- V6: 동작별 "분명한 우위"(walk, squat, swing) 대 "비슷"(balance, stride)
pm = {}
for mo in MOTIONS:
    cnt = 0
    tot = 0
    for d in DIST:
        for me in METRICS:
            vals = {m: T2[(mo, m, d, me)][0] for m in ("Direct", "ASAP", "RWM", "NDM")}
            cnt += vals["NDM"] <= min(vals.values())
            tot += 1
    pm[mo] = {"ndm_best_among_4": cnt, "of": tot}
OUT["V6_per_motion_best"] = pm

# ---------------------------------------------------------------- V7: 표 7 (점프) 같은 검사
b, s, pmj = count_best(T7, ["jump"], False, 0.0)
OUT["V7_jump_best_count"] = {"ndm_best_incl_ties": b, "strict": s, "of": 12, "per_dist": {}}
for d in DIST:
    OUT["V7_jump_best_count"]["per_dist"][d] = {
        me: {m: T7[("jump", m, d, me)][0] for m in ("Direct", "ASAP", "RWM", "NDM")} for me in METRICS}
jump_rows = []
for d in DIST:
    for me in METRICS:
        nd = T7[("jump", "NDM", d, me)]
        for base in ("ASAP", "RWM"):
            bb = T7[("jump", base, d, me)]
            w = welch(nd[0], nd[1], 5, bb[0], bb[1], 5)
            jump_rows.append({"dist": d, "metric": me, "vs": base, "ndm": nd[0], "base": bb[0], "p": None if w is None else w["p"]})
OUT["V7_jump_welch"] = jump_rows

# ---------------------------------------------------------------- V8: 표 3 FENCE 적용
t3 = {}
for base in ("ASAP", "RWM"):
    a, b = T3[base], T3[base + " w/ FENCE"]
    r = {}
    for k in ("link_pos", "qpos", "sqpos", "sroll"):
        w = welch(b[k][0], b[k][1], 5, a[k][0], a[k][1], 5)
        r[k] = {"base": a[k], "fence": b[k], "ratio_fence_over_base": b[k][0] / a[k][0],
                "welch_p_n5": None if w is None else w["p"]}
    t3[base] = r
OUT["V8_table3"] = t3
# 표 3의 ASAP/RWM 평균이 표 2 어느 행과 같은지 (설정 미표기)
match = []
for (mo, meth, d, me), v in T2.items():
    if meth == "ASAP" and me == "link_pos" and abs(v[0] - 3.68) < 1e-9:
        match.append((mo, d, me))
    if meth == "RWM" and me == "link_pos" and abs(v[0] - 4.42) < 1e-9:
        match.append((mo, d, me))
OUT["V8_table3_matches_in_table2"] = match

# ---------------------------------------------------------------- V9: 실로봇 표 4 (3회) Welch
t4 = {}
for task, d in T4.items():
    for ref in ("Direct", "PPO"):
        if ref not in d:
            continue
        for k in ("link_pos", "qpos", "task"):
            nd, rf = d["NDM"][k], d[ref][k]
            w = welch(nd[0], nd[1], 3, rf[0], rf[1], 3)
            t4[f"{task}|NDM vs {ref}|{k}"] = {
                "ndm": nd, "ref": rf, "rel_change_pct": 100 * (nd[0] - rf[0]) / rf[0] if rf[0] else None,
                "p": None if w is None else w["p"], "df": None if w is None else w["df"]}
OUT["V9_table4"] = t4
# 개선율: 추종 지표 상대 감소 (NDM vs Direct)
imp = {}
for task, d in T4.items():
    imp[task] = {k: 100 * (d["Direct"][k][0] - d["NDM"][k][0]) / d["Direct"][k][0] for k in ("link_pos", "qpos")}
OUT["V9_table4_tracking_reduction_pct"] = imp

# ---------------------------------------------------------------- V10: 표 5 (시행 수 3 가정)
t5 = {}
for meth, v in T5.items():
    if meth == "FENCE":
        continue
    r = {}
    for k in ("sqpos", "sroll"):
        w = welch(v[k][0], v[k][1], 3, T5["FENCE"][k][0], T5["FENCE"][k][1], 3)
        r[k] = {"ratio_to_fence": v[k][0] / T5["FENCE"][k][0], "p_n3": None if w is None else w["p"]}
    t5[meth] = r
fn = {k: T5["FENCE"][k][0] / T5["No Noise"][k][0] for k in ("sqpos", "sroll")}
fw = {k: welch(T5["FENCE"][k][0], T5["FENCE"][k][1], 3, T5["No Noise"][k][0], T5["No Noise"][k][1], 3) for k in ("sqpos", "sroll")}
OUT["V10_table5"] = {"vs_fence": t5, "fence_over_nonoise": fn,
                     "fence_vs_nonoise_p_n3": {k: v["p"] for k, v in fw.items()},
                     "ou_over_fence": {k: T5["Action OU"][k][0] / T5["FENCE"][k][0] for k in ("sqpos", "sroll")}}

# ---------------------------------------------------------------- V11: 설정 산술 (부록 A.4, A.5, D)
V11 = {}
state_dim = 3 + 2 + 29 + 29 + 1  # omega, rho(roll-pitch), q, qdot, h  (29 DoF 가정)
V11["state_dim_if_29dof"] = state_dim
V11["state_dim_if_23dof"] = 3 + 2 + 23 + 23 + 1
V11["paper_state_dim"] = 64
V11["paper_action_dim"] = 23
V11["input_dim_projection"] = 64 + 23  # 87
V11["decoder_input_dim"] = 64 + 64 + 23  # z + s + a = 151
V11["iteration_seconds"] = 3 + 10
V11["iterations_in_one_hour"] = 3600 / 13
V11["minutes_for_200_iterations"] = 200 * 13 / 60
V11["bins_total"] = 2 * 16
V11["samples_per_traj_at_50Hz_8s"] = 8 * 50
# 모델 파라미터 수 (본문 A.5 구조 그대로, 정수 계산)
d, dff, nl = 64, 16, 2
lin_proj = 87 * d + d
pos = (10 - 1) * d
enc_layer = (4 * d * d + 4 * d) + (d * dff + dff + dff * d + d) + 2 * (2 * d)  # MHA(qkv+out), FFN, 2 LayerNorm
dec = (151 * 64 + 64) + (64 * 512 + 512) + (512 * 512 + 512)
heads = 5 * (512 * 64 + 64)
V11["params"] = {"proj": lin_proj, "pos": pos, "encoder_layers": nl * enc_layer, "decoder": dec, "heads5": heads,
                 "total": lin_proj + pos + nl * enc_layer + dec + heads}
# 부록 A.5의 FK 입력(자유도)
OUT["V11_config_arithmetic"] = V11

# ---------------------------------------------------------------- V12: 주파수 섭동 크기 (식 1-2의 비정규화 DFT 기준)
V12 = {}
for T_sec in (4.0, 8.0, 12.0):
    Tn = int(T_sec * 50)  # dt = 0.02
    kc = int(math.floor(1.0 * T_sec))  # f_k = k/(T dt) <= 1 Hz
    for sigma in (1, 3, 6):
        std_unnorm = sigma * math.sqrt(2 * kc) / Tn  # 식 (1) 비정규화, 식 (2) 역변환이 1/T
        std_ortho = sigma * math.sqrt(2 * kc) / math.sqrt(Tn)  # 정규화(ortho) 규약일 때
        V12[f"T={T_sec}s,sigma={sigma}"] = {"T_steps": Tn, "kc_index": kc,
                                            "rms_rad_unnormalized_DFT": std_unnorm,
                                            "rms_deg_unnormalized_DFT": math.degrees(std_unnorm),
                                            "rms_rad_ortho_DFT": std_ortho,
                                            "rms_deg_ortho_DFT": math.degrees(std_ortho)}
OUT["V12_perturbation_amplitude"] = V12

# ---------------------------------------------------------------- V13: UCB 보너스 크기 (식 9, beta=0.05)
V13 = {}
for N in (10, 50, 200):
    for n in (1, 5, 20):
        if n <= N:
            V13[f"N={N},n_i={n}"] = 0.05 * math.sqrt(math.log(N) / n)
OUT["V13_ucb_bonus"] = V13

# ---------------------------------------------------------------- V14: 시간 예산
OUT["V14_time_budget"] = {"per_iter_s": 13, "200_iter_min": 200 * 13 / 60,
                          "pen_capping_50_iter_min": 50 * 13 / 60,
                          "claim_one_hour": "200 iterations within one hour (App. D)"}

# ---------------------------------------------------------------- V15: 격차 폐쇄율 (Direct - 방법) / (Direct - PPO w/ Disturb.)
gc = {}
for me in METRICS:
    per = {}
    for meth in ("ASAP", "RWM", "NDM"):
        vals = []
        for mo in MOTIONS:
            for d in DIST:
                direct = T2[(mo, "Direct", d, me)][0]
                ppo = T2[(mo, "PPO", d, me)][0]
                if direct - ppo > 0.03 * direct:  # 격차가 직접 전이 값의 3% 넘는 칸만
                    vals.append((direct - T2[(mo, meth, d, me)][0]) / (direct - ppo))
        per[meth] = {"n": len(vals), "mean": float(np.mean(vals)), "median": float(np.median(vals))}
    gc[me] = per
OUT["V15_gap_closure"] = gc
# 표 2에서 직접 전이와 PPO 사이 격차가 작은 칸(<= 10%)의 수
small = [(mo, d, me) for mo in MOTIONS for d in DIST for me in METRICS
         if abs(T2[(mo, "Direct", d, me)][0] - T2[(mo, "PPO", d, me)][0]) <= 0.1 * T2[(mo, "Direct", d, me)][0]]
OUT["V15_cells_with_small_gap"] = len(small)

# ---------------------------------------------------------------- V7 보강: 점프에서 NDM이 최선이 아닌 칸, 직접 전이 대비
jb = []
for d in DIST:
    for me in METRICS:
        vals = {m: T7[("jump", m, d, me)][0] for m in ("Direct", "ASAP", "RWM", "NDM")}
        if vals["NDM"] > min(vals.values()):
            jb.append({"dist": d, "metric": me, **vals})
OUT["V7_jump_not_best_cells"] = jb
OUT["V7_jump_ndm_lower_than_direct"] = sum(T7[("jump", "NDM", d, me)][0] < T7[("jump", "Direct", d, me)][0]
                                           for d in DIST for me in METRICS)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "recompute_results.json"), "w") as f:
    json.dump(OUT, f, indent=1, ensure_ascii=False, default=str)

# 요약 출력
print("V1", json.dumps({k: (v["ndm_best_incl_ties"], v["ndm_strictly_best"]) for k, v in v1.items()}))
print("V1 per motion", v1["include_ppo=False,tol=0.0"]["per_motion"])
print("V2", json.dumps(OUT["V2_reduction_vs_direct"], indent=0)[:1500])
print("V3", OUT["V3_ndm_vs_ppo"])
print("V3 ppo worse than direct", len(ppo_worse))
print("V4", {k: v for k, v in OUT["V4_ndm_vs_best_baseline"].items() if k != "rows"})
print("V5", nd_vs_dir, len(worse_than_direct))
print("V6", pm)
print("V7", OUT["V7_jump_best_count"]["ndm_best_incl_ties"], OUT["V7_jump_best_count"]["strict"])
print("V8", json.dumps(t3, indent=0)[:1800], match)
print("V9", {k: (round(v["p"], 4) if v["p"] is not None else None, v["rel_change_pct"]) for k, v in t4.items()})
print("V9 imp", imp)
print("V10", OUT["V10_table5"])
print("V11", V11)
print("V12", {k: (round(v["rms_deg_unnormalized_DFT"], 2), round(v["rms_deg_ortho_DFT"], 1)) for k, v in V12.items()})
print("V13", V13)
