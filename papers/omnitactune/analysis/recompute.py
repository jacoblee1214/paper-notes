"""OmniTacTune(arXiv:2607.03723 v1) 보고 수치 재계산과 일관성 검사 (numpy, scipy, 수 초).

실행: python3 analysis/recompute.py  -> analysis/recompute_results.json
입력: 논문 표 1, 표 2, 표 3-5와 그림 4, 5, 6의 라벨. 원자료(시행별 결과)는 공개되어 있지 않다.
가정(결과 해석에 항상 붙는다):
  - 최종 평가는 논문 4.1절 그대로 20회(표 1의 모든 값이 5%의 배수라 20회와 맞는다).
  - 표 2, 그림 5, 그림 6의 시행 수는 논문에 적혀 있지 않다. 값이 5%의 배수라 20회로 가정한다.
  - 시행끼리는 독립이고 학습 실행은 방법당 과제당 한 번이다(논문에 시드 반복 언급 없음).
  - 그림 4와 6의 막대·끝점 값은 그림에서 읽은 값이다(눈금 대비 약 ±5점).
"""
import json
import math
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
N = 20
Z = 1.959963984540054

TASKS = ["Peg-in-Hole", "Charger Insertion", "Cap Opening", "Box Opening"]
# 표 1 최종 성공률(%), 20회
T1 = {
    "PLD*": [65, 60, 50, 35],
    "PLD (Visual Only)": [60, 30, 40, 20],
    "ViTAL": [50, 50, 45, 30],
    "Ours": [100, 100, 90, 85],
}
T1_AVG_REPORTED = {"PLD*": 52.5, "PLD (Visual Only)": 37.5, "ViTAL": 43.75, "Ours": 93.75}
BASE = [40, 10, 5, 5]  # 4.1절: 인간 flow 기반 정책 20회
# 그림 4 끝점(그림 판독 값, 10회 체크포인트)
FIG4_END = {
    "Ours": [100, 100, 90, 80],
    "PLD*": [60, 60, 50, 40],
    "PLD (Visual Only)": [60, 30, 40, 20],
    "ViTAL": [50, 50, 45, 30],
}
# 표 2 (Peg-in-Hole)
T2 = {"ACT": 25, "ACT + Tactile": 60, "DP": 30, "RDP": 65, "pi0.5": 15, "pi0.5 + Tactile": 45,
      "Ours (ACT)": 80, "Ours (DP)": 90, "Ours (pi0.5)": 75, "Ours": 100}
# 그림 5 (전/후)
F5 = {"pi0.5 (TeleOp)": (15, 75, 60), "DP (TeleOp)": (30, 90, 60), "ACT (TeleOp)": (25, 80, 55),
      "Flow (TeleOp)": (50, 90, 40), "Flow (Human)": (40, 100, 60)}
# 4.2절 본문의 초기 성공률
TEXT42 = {"Flow (Human)": 40, "Flow (TeleOp)": 50, "ACT (TeleOp)": 20, "DP (TeleOp)": 30, "pi0.5 (TeleOp)": 20}
# 그림 6 (그림 판독 값, 막대 라벨)
F6 = {"Peg-in-Hole": {"AnyTouch2": 100, "Sparsh": 100, "T3": 90, "Marker": 100, "baseline": 40},
      "Charger Insertion": {"AnyTouch2": 100, "Sparsh": 80, "T3": 70, "Marker": 100, "baseline": 10}}
# 표 5 (부록 C.1)
T5 = {"Human Demos": (-2.889, -18.03), "Teleop Demos": (-7.320, -19.70),
      "Human Policy": (-2.664, -17.29), "TeleOp Policy": (-9.521, -20.89)}


def k_of(pct, n=N):
    return pct * n / 100.0


def wilson(k, n, z=Z):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def newcombe(k1, n1, k2, n2):
    """p1 - p2의 Newcombe(Wilson) 95% 구간."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    return [p1 - p2 - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            p1 - p2 + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)]


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = [0.0] * m
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[idx]))
        adj[idx] = run
    return adj


def mh_or(tables):
    """Mantel-Haenszel 공통 오즈비와 CMH 카이제곱(연속성 보정 없음)."""
    num = den = 0.0
    sa = ea = va = 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        num += a * d / n
        den += b * c / n
        r1, c1 = a + b, a + c
        sa += a
        ea += r1 * c1 / n
        va += r1 * (c + d) * c1 * (b + d) / (n * n * (n - 1))
    chi = (sa - ea) ** 2 / va
    return {"or": num / den if den else float("inf"), "cmh_chi2": chi,
            "cmh_p": float(stats.chi2.sf(chi, 1))}


def fisher_power(p1, p2, n=N, alpha=0.05):
    """양측 Fisher 검정의 정확 검정력(두 이항을 전수 합산)."""
    pm1 = stats.binom.pmf(range(n + 1), n, p1)
    pm2 = stats.binom.pmf(range(n + 1), n, p2)
    cache = {}
    pw = 0.0
    for a, b in product(range(n + 1), repeat=2):
        if (a, b) not in cache:
            cache[(a, b)] = fisher(a, n, b, n)
        if cache[(a, b)] < alpha:
            pw += pm1[a] * pm2[b]
    return float(pw)


def main():
    R = {"assumptions": {"n_final": N, "t2_t5_n_assumed": N, "wilson_z": Z}}

    # V1 표 1 평균 재계산
    avg = {m: float(np.mean(v)) for m, v in T1.items()}
    R["V1_averages"] = {m: {"recomputed": avg[m], "reported": T1_AVG_REPORTED[m],
                            "match": abs(avg[m] - T1_AVG_REPORTED[m]) < 1e-9} for m in T1}

    # V2 정수 가능성: 표 1의 모든 값이 N=20에서 정수 성공 횟수인가
    feas = {}
    for m, v in T1.items():
        feas[m] = [k_of(x) for x in v]
    for tag, d in (("BASE", {"base": BASE}), ("T2", {"t2": list(T2.values())}),
                   ("F5", {"f5": [x for t in F5.values() for x in t[:2]]}),
                   ("F6", {"f6": [x for t in F6.values() for x in t.values()]})):
        for key, v in d.items():
            feas[f"{tag}:{key}"] = [k_of(x) for x in v]
    R["V2_integer_counts_n20"] = {k: v for k, v in feas.items()}
    R["V2_all_integer"] = all(abs(x - round(x)) < 1e-9 for v in feas.values() for x in v)
    R["V2_n10_feasible"] = {"T1_cells_multiple_of_10": {m: [x % 10 == 0 for x in v] for m, v in T1.items()}}

    # V3 Wilson 구간 (표 1, 기반)
    R["V3_wilson"] = {m: [[round(x, 4) for x in wilson(k_of(p), N)] for p in v] for m, v in T1.items()}
    R["V3_wilson"]["base"] = [[round(x, 4) for x in wilson(k_of(p), N)] for p in BASE]

    # V4 Ours 대 각 기준선: 과제별 Fisher, Newcombe, Holm
    comps = []
    for m in ("PLD*", "PLD (Visual Only)", "ViTAL"):
        for j, t in enumerate(TASKS):
            k1, k2 = round(k_of(T1["Ours"][j])), round(k_of(T1[m][j]))
            comps.append({"baseline": m, "task": t, "ours": k1, "base": k2, "diff_pp": (k1 - k2) / N * 100,
                          "newcombe_95": [round(x * 100, 1) for x in newcombe(k1, N, k2, N)],
                          "fisher_p": fisher(k1, N, k2, N)})
    adj = holm([c["fisher_p"] for c in comps])
    for c, a in zip(comps, adj):
        c["holm_p"] = a
    R["V4_ours_vs_baselines"] = comps
    R["V4_n_holm_sig"] = int(sum(c["holm_p"] < 0.05 for c in comps))
    R["V4_n_raw_sig"] = int(sum(c["fisher_p"] < 0.05 for c in comps))

    # V5 4과제 합산(각 80회)과 층화
    pooled = {}
    for m in ("PLD*", "PLD (Visual Only)", "ViTAL"):
        k1 = sum(round(k_of(x)) for x in T1["Ours"])
        k2 = sum(round(k_of(x)) for x in T1[m])
        tabs = [(round(k_of(T1["Ours"][j])), N - round(k_of(T1["Ours"][j])),
                 round(k_of(T1[m][j])), N - round(k_of(T1[m][j]))) for j in range(4)]
        pooled[m] = {"ours": k1, "base": k2, "n": 4 * N, "fisher_p": fisher(k1, 4 * N, k2, 4 * N),
                     "newcombe_95": [round(x * 100, 1) for x in newcombe(k1, 4 * N, k2, 4 * N)],
                     "mh": mh_or(tabs)}
    R["V5_pooled_vs_baselines"] = pooled
    R["V5_ours_pooled"] = {"k": 75, "n": 80, "wilson": [round(x, 4) for x in wilson(75, 80)]}

    # V6 Ours 대 기반 정책(과제별과 합산)
    vs_base = []
    for j, t in enumerate(TASKS):
        k1, k2 = round(k_of(T1["Ours"][j])), round(k_of(BASE[j]))
        vs_base.append({"task": t, "base": k2, "ours": k1, "fisher_p": fisher(k1, N, k2, N),
                        "newcombe_95": [round(x * 100, 1) for x in newcombe(k1, N, k2, N)]})
    R["V6_ours_vs_base"] = vs_base
    kb, ko = sum(round(k_of(x)) for x in BASE), sum(round(k_of(x)) for x in T1["Ours"])
    R["V6_pooled"] = {"base": kb, "ours": ko, "n": 80, "fisher_p": fisher(ko, 80, kb, 80),
                      "newcombe_95": [round(x * 100, 1) for x in newcombe(ko, 80, kb, 80)]}
    # 기준선 대 기반 (효과 없음 검사)
    bvb = {}
    for m in ("PLD*", "PLD (Visual Only)", "ViTAL"):
        bvb[m] = [{"task": t, "base_pol": round(k_of(BASE[j])), "method": round(k_of(T1[m][j])),
                   "fisher_p": fisher(round(k_of(T1[m][j])), N, round(k_of(BASE[j])), N)}
                  for j, t in enumerate(TASKS)]
        k1 = sum(round(k_of(x)) for x in T1[m])
        bvb[m].append({"task": "합산", "base_pol": kb, "method": k1, "fisher_p": fisher(k1, 80, kb, 80)})
    R["V6_baselines_vs_base_policy"] = bvb

    # V7 본문 "outperforming all baselines with more than 40% success rate" 해석
    gaps = {}
    for j, t in enumerate(TASKS):
        best = max(T1[m][j] for m in ("PLD*", "PLD (Visual Only)", "ViTAL"))
        gaps[t] = {"ours": T1["Ours"][j], "best_baseline": best, "gap_pp": T1["Ours"][j] - best}
    R["V7_gap_vs_best_baseline"] = gaps
    R["V7_mean_gap_vs_best"] = float(np.mean([g["gap_pp"] for g in gaps.values()]))
    R["V7_min_gap"] = min(g["gap_pp"] for g in gaps.values())
    R["V7_mean_gap_vs_each_baseline_avg"] = {m: T1_AVG_REPORTED["Ours"] - T1_AVG_REPORTED[m]
                                             for m in ("PLD*", "PLD (Visual Only)", "ViTAL")}

    # V8 그림 4 끝점(판독) 대 표 1
    R["V8_fig4_end_vs_table1"] = {m: [{"task": t, "fig4_read": FIG4_END[m][j], "table1": T1[m][j],
                                       "diff": FIG4_END[m][j] - T1[m][j]} for j, t in enumerate(TASKS)]
                                  for m in T1}
    R["V8_n_cells_differ"] = int(sum(FIG4_END[m][j] != T1[m][j] for m in T1 for j in range(4)))

    # V9 그림 5 대 4.2절 본문 대 표 2
    v9 = {}
    for name, (bef, aft, delta) in F5.items():
        v9[name] = {"fig5_before": bef, "fig5_after": aft, "fig5_delta_label": delta,
                    "delta_recomputed": aft - bef, "delta_match": aft - bef == delta,
                    "text42_initial": TEXT42[name], "text_equals_fig": TEXT42[name] == bef,
                    "fisher_p_before_vs_after": fisher(round(k_of(aft)), N, round(k_of(bef)), N)}
    R["V9_fig5"] = v9
    R["V9_table2_vs_fig5"] = {
        "ACT": [T2["ACT"], F5["ACT (TeleOp)"][0]],
        "DP": [T2["DP"], F5["DP (TeleOp)"][0]],
        "pi0.5": [T2["pi0.5"], F5["pi0.5 (TeleOp)"][0]],
        "Ours(ACT)": [T2["Ours (ACT)"], F5["ACT (TeleOp)"][1]],
        "Ours(DP)": [T2["Ours (DP)"], F5["DP (TeleOp)"][1]],
        "Ours(pi0.5)": [T2["Ours (pi0.5)"], F5["pi0.5 (TeleOp)"][1]],
        "Ours": [T2["Ours"], F5["Flow (Human)"][1]],
    }
    R["V9_range_claim"] = {"text": "15-50% -> 75-100%",
                           "before_min_max": [min(v[0] for v in F5.values()), max(v[0] for v in F5.values())],
                           "after_min_max": [min(v[1] for v in F5.values()), max(v[1] for v in F5.values())]}

    # V10 표 2: 같은 백본 모방 학습 대 Ours (시행 20회 가정)
    pairs = [("Ours (ACT)", "ACT + Tactile"), ("Ours (ACT)", "ACT"), ("Ours (DP)", "RDP"), ("Ours (DP)", "DP"),
             ("Ours (pi0.5)", "pi0.5 + Tactile"), ("Ours (pi0.5)", "pi0.5"),
             ("Ours", "RDP"), ("Ours", "ACT + Tactile"), ("Ours", "pi0.5 + Tactile")]
    v10 = []
    for a, b in pairs:
        k1, k2 = round(k_of(T2[a])), round(k_of(T2[b]))
        v10.append({"ours": a, "baseline": b, "k_ours": k1, "k_base": k2, "diff_pp": T2[a] - T2[b],
                    "fisher_p": fisher(k1, N, k2, N),
                    "newcombe_95": [round(x * 100, 1) for x in newcombe(k1, N, k2, N)]})
    R["V10_table2_pairs"] = v10
    best_il = {"ACT backbone": (T2["Ours (ACT)"], T2["ACT + Tactile"]), "DP backbone": (T2["Ours (DP)"], T2["RDP"]),
               "pi0.5 backbone": (T2["Ours (pi0.5)"], T2["pi0.5 + Tactile"])}
    R["V10_claim_20_30"] = {k: v[0] - v[1] for k, v in best_il.items()}
    # 세 백본 합산(같은 백본끼리)
    k1 = round(k_of(80)) + round(k_of(90)) + round(k_of(75))
    k2 = round(k_of(60)) + round(k_of(65)) + round(k_of(45))
    R["V10_pooled_3backbones"] = {"ours": k1, "best_imitation": k2, "n": 60, "fisher_p": fisher(k1, 60, k2, 60),
                                  "newcombe_95": [round(x * 100, 1) for x in newcombe(k1, 60, k2, 60)]}

    # V11 그림 6 (판독 값, 20회 가정)
    v11 = {}
    for task, d in F6.items():
        base = round(k_of(d["baseline"]))
        row = {}
        for rep in ("AnyTouch2", "Sparsh", "T3", "Marker"):
            k = round(k_of(d[rep]))
            row[rep] = {"pct": d[rep], "k": k, "wilson": [round(x, 3) for x in wilson(k, N)],
                        "fisher_vs_baseline": fisher(k, N, base, N)}
        row["AnyTouch2_vs_Sparsh_p"] = fisher(round(k_of(d["AnyTouch2"])), N, round(k_of(d["Sparsh"])), N)
        row["AnyTouch2_vs_T3_p"] = fisher(round(k_of(d["AnyTouch2"])), N, round(k_of(d["T3"])), N)
        row["AnyTouch2_vs_Marker_p"] = fisher(round(k_of(d["AnyTouch2"])), N, round(k_of(d["Marker"])), N)
        v11[task] = row
    R["V11_fig6"] = v11

    # V12 시간·예산 산술 (10 Hz, 에피소드 최대 300 스텝, 부록 A.8, 4.1절)
    hz, max_ep = 10, 300
    total = {"Peg-in-Hole": 50, "Charger Insertion": 40, "Cap Opening": 60, "Box Opening": 80}
    v12 = {}
    for t, mins in total.items():
        online = mins - 12
        v12[t] = {"total_min": mins, "warmstart_min": 12, "online_min": online,
                  "max_steps_warmstart": 12 * 60 * hz, "max_steps_online": online * 60 * hz,
                  "min_episodes_if_full_length_warmstart": 12 * 60 * hz / max_ep,
                  "min_episodes_if_full_length_online": online * 60 * hz / max_ep}
    R["V12_budget"] = v12
    R["V12_replay_buffer_vs_steps"] = {"buffer_max": 10000, "steps_warmstart_upper": 12 * 60 * hz,
                                       "steps_online_upper_peg": (50 - 12) * 60 * hz,
                                       "note": "ControlTac이 궤적당 2개를 더 만들므로 웜스타트 전이는 최대 3배(상한)"}
    R["V12_scale_schedule"] = {"schedule": [0.05, 0.10, 0.15], "interval_online_steps": 500,
                               "steps_to_reach_max": 1000, "seconds_at_10hz": 1000 / hz,
                               "fraction_of_peg_online_budget": 1000 / ((50 - 12) * 60 * hz)}
    R["V12_total_eval_trials_peg_min"] = {
        "checkpoints_10_trials": "0(기반 20회), 12, 15, 20, 30, 40 -> 5개 체크포인트 x 10회 + 최종 20회",
        "trials": 5 * 10 + 20,
        "note": "평가 시행의 시간이 '훈련 시간'에 들어가는지 논문에 적혀 있지 않다."}

    # V13 보상 산술 (A.6, 표 4)
    wr, wg, wf, ws = 1.0, 0.5, 2.0, 1.0
    R["V13_reward"] = {
        "normalizer": wr + wg + wf,
        "max_positive_all_terms": (wr + wg + wf) / (wr + wg + wf),
        "max_if_reach_term_zero_after_grasp": (wg + wf) / (wr + wg + wf),
        "max_if_grasp_term_zero": (wr + wf) / (wr + wg + wf),
        "flow_reward_when_l_equals_L_main_text": 1 + 1 / 30,
        "flow_reward_max_appendix_l_le_L_minus_1": (29 / 30) + 1 / 30,
        "safety_penalty_range_table4": [-1, 0],
    }

    # V14 표 5 부드러움
    v14 = {k: {"M-SPARC": v[0], "M-LDLJ": v[1]} for k, v in T5.items()}
    R["V14_smoothness"] = v14
    R["V14_ldlj_ratio"] = {"demos_teleop_over_human_exp": math.exp(T5["Human Demos"][1] - T5["Teleop Demos"][1]),
                           "policy_teleop_over_human_exp": math.exp(T5["Human Policy"][1] - T5["TeleOp Policy"][1]),
                           "note": "LDLJ는 -log(무차원 저크)라 차이 1은 저크 비용 e배"}
    R["V14_policy_minus_demos"] = {"human": {"SPARC": T5["Human Policy"][0] - T5["Human Demos"][0],
                                              "LDLJ": T5["Human Policy"][1] - T5["Human Demos"][1]},
                                   "teleop": {"SPARC": T5["TeleOp Policy"][0] - T5["Teleop Demos"][0],
                                              "LDLJ": T5["TeleOp Policy"][1] - T5["Teleop Demos"][1]}}

    # V15 검정력: n=20에서 검출 가능한 차이(Fisher 정확 검정, 양측 0.05)
    pw = {}
    for p1, p2 in ((0.5, 0.5), (0.85, 0.35), (0.65, 0.35), (0.80, 0.60), (0.90, 0.65), (0.75, 0.45),
                   (0.85, 0.50), (0.95, 0.75)):
        pw[f"{p1:.2f}_vs_{p2:.2f}"] = fisher_power(p1, p2)
    R["V15_power_n20"] = pw
    need = {}
    for p1, p2 in ((0.80, 0.60), (0.90, 0.65), (0.75, 0.45)):
        for n in range(20, 121, 5):
            if fisher_power(p1, p2, n=n) >= 0.8:
                need[f"{p1:.2f}_vs_{p2:.2f}"] = n
                break
    R["V15_n_for_80pct_power"] = need

    # V16 10회 체크포인트의 이항 표준오차
    R["V16_checkpoint_se_pp"] = {f"p={p}": 100 * math.sqrt(p * (1 - p) / 10) for p in (0.1, 0.3, 0.5, 0.7, 0.9)}
    R["V16_peg_ours_curve_read"] = {"minutes": [0, 12, 15, 20, 30, 40, 50], "pct": [40, 30, 30, 50, 90, 90, 100],
                                    "note": "그림 판독. 12분(웜스타트 끝)과 15분 사이 평가는 각 10회, 0분은 20회"}

    # V17 절제 끝점(그림 15-18 판독 값, 50분, 체크포인트 10회 가정)과 Ours 대비 Fisher
    abl = {"보상(그림 15)": {"No Reaching Reward": 80, "No dense reward": 70, "No Flow Reward": 70,
                            "No tactile reward": 90},
           "잔차 설계(그림 16)": {"ACT + Ours": 50, "Keypoints + per-step": 90, "ACT + per-step": 30,
                              "No gating": 80, "Visual-Tactile": 40},
           "웜스타트(그림 17)": {"w/o optimization (PLD*)": 60, "w/o ControlTac": 80, "w/o critic loss": 90,
                             "w/o reconstruction loss": 90},
           "행동 공간(그림 18)": {"No Scheduler": 90, "No Action Scale": 50, "Action Scale = 0.5": 70}}
    v17 = {}
    for grp, d in abl.items():
        v17[grp] = {k: {"pct": v, "k_of_10": v // 10, "fisher_p_vs_ours_10_of_10": fisher(10, 10, v // 10, 10)}
                    for k, v in d.items()}
    R["V17_ablation_end_n10"] = v17
    R["V17_n_sig_005"] = int(sum(x["fisher_p_vs_ours_10_of_10"] < 0.05 for d in v17.values() for x in d.values()))
    R["V17_n_total"] = int(sum(len(d) for d in v17.values()))
    R["V17_cell_unit_note"] = "10회 가정이면 값은 10%p 단위. Fig. 16 'ACT + Ours' 50%는 Fig. 5/표 2의 Ours (ACT) 80%와 다른 값"
    print("V17", R["V17_n_sig_005"], "/", R["V17_n_total"],
          {g: {k: round(x["fisher_p_vs_ours_10_of_10"], 3) for k, x in d.items()} for g, d in v17.items()})

    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")

    # 요약 출력
    print("V1", {m: v["match"] for m, v in R["V1_averages"].items()})
    print("V2 all integer at n=20:", R["V2_all_integer"])
    print("V4 raw sig:", R["V4_n_raw_sig"], "/12, Holm sig:", R["V4_n_holm_sig"])
    for c in comps:
        print(f'  {c["baseline"]:18s} {c["task"]:18s} {c["ours"]}/20 vs {c["base"]}/20 '
              f'p={c["fisher_p"]:.4f} holm={c["holm_p"]:.4f} diff={c["diff_pp"]:.0f} {c["newcombe_95"]}')
    for m, v in pooled.items():
        print("V5", m, v["ours"], v["base"], f'p={v["fisher_p"]:.2e}', v["newcombe_95"], v["mh"])
    print("V6 pooled", R["V6_pooled"])
    for r in vs_base:
        print("V6", r)
    print("V7", R["V7_gap_vs_best_baseline"], R["V7_mean_gap_vs_best"])
    print("V8 differing cells", R["V8_n_cells_differ"])
    for k, v in v9.items():
        print("V9", k, v)
    for r in v10:
        print("V10", r)
    print("V10 pooled", R["V10_pooled_3backbones"])
    print("V11", json.dumps(v11, ensure_ascii=False))
    print("V12", R["V12_scale_schedule"], R["V12_budget"]["Peg-in-Hole"])
    print("V13", R["V13_reward"])
    print("V14", R["V14_ldlj_ratio"], R["V14_policy_minus_demos"])
    print("V15", pw, need)
    print("V16", R["V16_checkpoint_se_pp"])


if __name__ == "__main__":
    main()
