#!/usr/bin/env python3
"""HumanoidToolBench(arXiv:2610.02089 v1) 보고 수치 재계산.

표준 라이브러리만 사용합니다. 실행: python3 recompute.py  -> recompute_results.json
입력 값은 논문 v1의 Table II-IV, Fig. 5-7, 프로젝트 페이지(2026-11-23 확인)에서 옮겼습니다.
검사 이름은 V1, V2, ... 로 붙입니다.
"""
import json, math, itertools, os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- 통계 도구
def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def fisher_two_sided(a, n1, b, n2):
    """2x2 Fisher exact (양측, 확률이 관측 이하인 표의 합)."""
    K = a + b
    N = n1 + n2
    def pmf(x):
        return math.comb(n1, x) * math.comb(n2, K - x) / math.comb(N, K)
    lo, hi = max(0, K - n2), min(n1, K)
    p0 = pmf(a)
    return min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p0 * (1 + 1e-9)))

def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

# ---------------------------------------------------------------- 논문 Table III
# 값 순서: S: L0(Move,Retrieve,Break) L1(...) L2(...) | D: L0 L1 L2
T3 = {
 "ACT":    [5,0,0, 0,0,0, 0,0,0,   3,4,1, 0,0,0, 0,0,0],
 "DP":     [14,5,17, 0,0,2, 0,0,0, 11,9,13, 0,0,1, 0,0,0],
 "pi0.5":  [29,24,31, 4,0,12, 1,0,0, 26,28,21, 5,3,1, 1,1,0],
 "Psi0":   [19,14,22, 9,0,13, 1,0,2, 14,15,18, 7,3,8, 3,1,0],
 "GR00T":  [77,73,88, 64,36,66, 10,9,31, 67,59,75, 58,30,63, 30,5,48],
 "Cosmos": [12,0,2, 0,0,1, 0,0,0, 14,0,0, 2,0,0, 1,0,0],
 "FastWAM":[64,62,98, 76,57,86, 9,20,69, 74,45,93, 75,45,82, 8,11,61],
}
SC = ["Move", "Retrieve", "Break"]

def cell(pol, mode, level, sc):
    base = 0 if mode == "S" else 9
    return T3[pol][base + 3 * level + SC.index(sc)]

# ---------------------------------------------------------------- 논문 Fig. 5 (S, D 순)
EV_NAMES = ["candidate", "correct_first", "correct_lift", "tool_target", "success"]
FIG5 = {
 "ACT":    {"S": [46.0,46.0,5.3,2.5,0.0],   "D": [57.2,44.8,5.5,2.3,0.0]},
 "DP":     {"S": [89.8,70.0,15.0,7.7,0.3],  "D": [97.0,49.2,12.2,5.7,0.2]},
 "pi0.5":  {"S": [98.8,72.7,44.2,22.3,2.8], "D": [99.5,44.3,29.7,15.7,1.8]},
 "Psi0":   {"S": [98.1,74.4,38.3,26.6,4.3], "D": [99.7,52.0,25.3,18.8,3.7]},
 "GR00T":  {"S": [99.7,87.3,90.0,79.8,36.0],"D": [100.0,77.2,83.7,74.3,39.0]},
 "Cosmos": {"S": [39.8,33.2,6.3,2.3,0.2],   "D": [51.3,26.3,7.8,2.3,0.5]},
 "FastWAM":{"S": [99.8,87.7,89.2,88.2,52.8],"D": [100.0,74.7,84.5,81.0,47.0]},
}

# ---------------------------------------------------------------- 프로젝트 페이지 표 (2026-11-23)
PAGE_NAMES = {"ACT":"ACT IL","DP":"DP IL","pi0.5":"π 0.5 VLA","Psi0":"Ψ 0 VLA","GR00T":"GR00T N1.7 VLA",
              "Cosmos":"Cosmos 3 Edge WAM","FastWAM":"FastWAM WAM"}
PAGE_T3 = json.load(open(os.path.join(HERE, "page_table.json")))
PAGE_EV = json.load(open(os.path.join(HERE, "page_events.json")))

R = {}

# V1: 논문 Table III 와 프로젝트 페이지 표의 일치
mism = []
for pol, row in T3.items():
    pg = PAGE_T3["S"][PAGE_NAMES[pol]] + PAGE_T3["D"][PAGE_NAMES[pol]]
    if pg != row:
        mism.append(pol)
R["V1_table3_vs_page"] = {"policies_mismatch": mism, "cells_compared": 7 * 18}

# V2: Fig.5 vs 프로젝트 페이지
mism = []
for pol in FIG5:
    for m in "SD":
        if FIG5[pol][m] != PAGE_EV[m][PAGE_NAMES[pol]]:
            mism.append((pol, m))
R["V2_fig5_vs_page"] = {"mismatch": mism}

# V3: Fig.5 success 열 = Table III L1, L2 6칸 평균 (S/D 각각)
rows = {}
for pol in T3:
    for m in "SD":
        vals = [cell(pol, m, lv, sc) for lv in (1, 2) for sc in SC]
        exact = sum(vals) / 6
        rep = FIG5[pol][m][4]
        rows[f"{pol}-{m}"] = {"table3_sum": sum(vals), "mean": round(exact, 3), "fig5": rep,
                              "match_at_0.1": abs(round(exact, 1) - rep) < 1e-9}
R["V3_fig5_success_from_table3"] = rows
R["V3_mismatches"] = [k for k, v in rows.items() if not v["match_at_0.1"]]

# V4: Fig.5 값이 '정수 % 6칸 평균'으로 가능한가 (n/6 반올림)
infeas = []
for pol in FIG5:
    for m in "SD":
        for name, v in zip(EV_NAMES, FIG5[pol][m]):
            n = round(v * 6)
            if abs(round(n / 6, 1) - v) > 1e-9:
                infeas.append((pol, m, name, v))
R["V4_fig5_integer_feasibility_infeasible"] = infeas

# V5: 본문 서술의 방향성 확인
claims = {}
# (a) L1->L2: L1>0이면 항상 감소
viol = []
for pol in T3:
    for m in "SD":
        for sc in SC:
            a, b = cell(pol, m, 1, sc), cell(pol, m, 2, sc)
            if a > 0 and not (b < a):
                viol.append((pol, m, sc, a, b))
claims["L1_to_L2_decrease_whenever_L1_positive_violations"] = viol
# (b) GR00T/FastWAM: D가 S보다 L1 세 시나리오 모두 낮은가
for pol in ("GR00T", "FastWAM"):
    claims[f"{pol}_L1_D_lower_all3"] = all(cell(pol, "D", 1, s) < cell(pol, "S", 1, s) for s in SC)
# (c) FastWAM이 L1 최고 (모든 6칸)
best = {}
for m in "SD":
    for s in SC:
        vals = {p: cell(p, m, 1, s) for p in T3}
        best[f"{m}-{s}"] = max(vals, key=vals.get)
claims["L1_best_policy_per_cell"] = best
# (d) 'Retrieve가 push/strike보다 낮다' (GR00T, FastWAM) 레벨·모드별
exc = []
for pol in ("GR00T", "FastWAM"):
    for m in "SD":
        for lv in (0, 1, 2):
            r = cell(pol, m, lv, "Retrieve")
            if not (r < cell(pol, m, lv, "Move") and r < cell(pol, m, lv, "Break")):
                exc.append((pol, m, f"L{lv}", cell(pol, m, lv, "Move"), r, cell(pol, m, lv, "Break")))
claims["Retrieve_not_lowest_exceptions(pol,mode,level,Move,Retrieve,Break)"] = exc
# (e) L2에서 Break가 두 공 과제보다 높은가
exc = []
for pol in ("GR00T", "FastWAM"):
    for m in "SD":
        b = cell(pol, m, 2, "Break")
        if not (b > cell(pol, m, 2, "Move") and b > cell(pol, m, 2, "Retrieve")):
            exc.append((pol, m))
claims["L2_Break_highest_exceptions"] = exc
# (f) FastWAM L2: D가 S보다 세 시나리오 모두 낮은가
claims["FastWAM_L2_D_lower_all3"] = all(cell("FastWAM", "D", 2, s) < cell("FastWAM", "S", 2, s) for s in SC)
claims["GR00T_L2_Move_S_to_D"] = (cell("GR00T", "S", 2, "Move"), cell("GR00T", "D", 2, "Move"))
R["V5_text_claims"] = claims

# V6: L0 성공률(3 시나리오 평균) vs L1/L2의 'correct_lift' 이벤트
rows = {}
for pol in T3:
    for m in "SD":
        l0 = sum(cell(pol, m, 0, s) for s in SC) / 3
        rows[f"{pol}-{m}"] = {"L0_mean": round(l0, 1), "L1L2_correct_lift_event": FIG5[pol][m][2],
                              "event_minus_L0": round(FIG5[pol][m][2] - l0, 1)}
R["V6_L0_vs_lift_event"] = rows

# V7: 표 III 셀 Wilson 95% (n=100) 폭 및 근접 순위
w = {}
for k in (5, 36, 50, 64, 76, 98):
    lo, hi = wilson(k, 100); w[str(k)] = (round(lo * 100, 1), round(hi * 100, 1))
R["V7_wilson_n100"] = w

# V8: 정책 쌍 비교 (L1, 6칸 풀링; 비독립 가능성 있음)
def pooled(pol, m, lv):
    k = sum(cell(pol, m, lv, s) for s in SC); return k, 300
pairs = {}
for m in "SD":
    for lv in (1, 2):
        a, n1 = pooled("GR00T", m, lv); b, n2 = pooled("FastWAM", m, lv)
        pairs[f"GR00T_vs_FastWAM_{m}_L{lv}"] = {"GR00T": f"{a}/{n1}", "FastWAM": f"{b}/{n2}",
                                               "fisher_p": fisher_two_sided(a, n1, b, n2)}
# 셀 단위 L1
cellp = {}
for m in "SD":
    for s in SC:
        a, b = cell("GR00T", m, 1, s), cell("FastWAM", m, 1, s)
        cellp[f"{m}-{s}-L1"] = {"GR00T": a, "FastWAM": b, "fisher_p": round(fisher_two_sided(a, 100, b, 100), 4)}
R["V8_gr00t_vs_fastwam_pooled"] = pairs
R["V8_gr00t_vs_fastwam_cells_L1"] = cellp

# V9: S vs D (같은 정책, L1 풀링, 비쌍 Fisher 근사)
sd = {}
for pol in ("GR00T", "FastWAM", "pi0.5", "Psi0"):
    for lv in (1, 2):
        a, n = pooled(pol, "S", lv); b, _ = pooled(pol, "D", lv)
        sd[f"{pol}_L{lv}"] = {"S": f"{a}/{n}", "D": f"{b}/{n}", "fisher_p": round(fisher_two_sided(a, n, b, n), 4)}
R["V9_decoy_effect_pooled_unpaired"] = sd
cells = {}
for pol in ("GR00T", "FastWAM"):
    for s in SC:
        a, b = cell(pol, "S", 1, s), cell(pol, "D", 1, s)
        cells[f"{pol}-{s}-L1"] = {"S": a, "D": b, "fisher_p": round(fisher_two_sided(a, 100, b, 100), 4)}
R["V9_decoy_effect_cells_L1"] = cells

# V10: L1 -> L2 낙폭 (FastWAM BallMove S 76->9 등)
R["V10_L1_to_L2"] = {
    "FastWAM-Move-S": {"L1": 76, "L2": 9, "fisher_p": fisher_two_sided(76, 100, 9, 100)},
    "GR00T-Move-S": {"L1": 64, "L2": 10, "fisher_p": fisher_two_sided(64, 100, 10, 100)},
}

# V11: 실로봇 Table IV
T4 = {"ACT": [1, 0, 2, 1], "GR00T": [6, 4, 4, 4], "FastWAM": [5, 4, 4, 5]}   # MoveS MoveD RetS RetD
real = {"cells": {}, "pooled": {}}
for pol, v in T4.items():
    real["pooled"][pol] = {"k": sum(v), "n": 40, "wilson95": [round(x * 100, 1) for x in wilson(sum(v), 40)]}
    for name, k in zip(["Move-S", "Move-D", "Ret-S", "Ret-D"], v):
        real["cells"][f"{pol}-{name}"] = {"k": k, "n": 10, "wilson95": [round(x * 100, 1) for x in wilson(k, 10)]}
cmp_ = {}
for a, b in itertools.combinations(T4, 2):
    cmp_[f"{a}_vs_{b}_pooled40"] = round(fisher_two_sided(sum(T4[a]), 40, sum(T4[b]), 40), 4)
    cmp_[f"{a}_vs_{b}_Move-S"] = round(fisher_two_sided(T4[a][0], 10, T4[b][0], 10), 4)
real["fisher"] = cmp_
# 실로봇 S vs D (GR00T, FastWAM, 풀링)
real["decoy_pooled"] = {p: {"S": T4[p][0] + T4[p][2], "D": T4[p][1] + T4[p][3], "n_each": 20,
                            "fisher_p": round(fisher_two_sided(T4[p][0] + T4[p][2], 20, T4[p][1] + T4[p][3], 20), 4)}
                        for p in T4}
R["V11_real_robot"] = real

# V12: sim vs real (S, L1: 시뮬 BallMove/BallRetrieve와 실로봇)
simreal = {}
for pol, key in (("ACT", "ACT"), ("GR00T", "GR00T"), ("FastWAM", "FastWAM")):
    sim = [cell(pol, m, 1, s) for m in "SD" for s in ("Move", "Retrieve")]
    simreal[pol] = {"sim_L1_ball_cells_mean_pct": round(sum(sim) / 4, 1),
                    "real_mean_pct": round(sum(T4[key]) / 4 * 10, 1)}
R["V12_sim_vs_real_L1_ball"] = simreal

# V13: 도구 선택 probe (Fig.6, Fig.7)
R["V13_probes"] = {
 "fig6_Move_first_contact_85_vs_74": round(fisher_two_sided(85, 100, 74, 100), 4),
 "fig6_Retrieve_first_contact_59_vs_36": round(fisher_two_sided(59, 100, 36, 100), 6),
 "fig7_original77_vs_unrelated84": round(fisher_two_sided(77, 100, 84, 100), 4),
 "fig7_original77_vs_paraphrase70": round(fisher_two_sided(77, 100, 70, 100), 4),
 "fig7_paraphrase70_vs_unrelated84": round(fisher_two_sided(70, 100, 84, 100), 4),
 "fig7_wilson_77": [round(x * 100, 1) for x in wilson(77, 100)],
 "fig7_wilson_84": [round(x * 100, 1) for x in wilson(84, 100)],
 "fig6_note_paired_scenes_Move_91_vs_episodes_100": True,
}

# V14: ToolBook 산술 (Table II, III-D)
l0 = {"Move": (289, 244), "Retrieve": (388, 313), "Break": (316, 253)}
sum_l0 = sum(a + b for a, b in l0.values())
std_l0 = sum(a for a, _ in l0.values())
R["V14_toolbook"] = {
 "L0_total": sum_l0, "L0_text": 1803, "L1L2": 12 * 100, "real": 29 + 21 + 20 + 21,
 "sim_total": sum_l0 + 1200, "total": sum_l0 + 1200 + 91,
 "L0_from_success": 1189, "L0_from_fail": 614, "L0_split_sum": 1189 + 614,
 "successes_not_reaching_lift": 1200 - 1189,
 "teleop_attempts": 1959, "teleop_success_rate": round(1200 / 1959, 4),
 "failed_attempts": 1959 - 1200, "failed_with_lift": 614,
 "std_mode_demos_for_probe_policy": std_l0 + 6 * 100, "text_1593": 1593,
}

# V15: 학습 규모 (본문 IV-A 및 HF 모델 카드)
R["V15_training_scale"] = {
 "samples_seen": 40000 * 128, "per_demo_epochs_over_3003": round(40000 * 128 / 3003, 1),
 "real_finetune_samples": 5000 * 128, "per_real_demo_epochs_over_91": round(5000 * 128 / 91, 1),
 "ACT_params_from_fp32_bytes_M": round(208520792 / 4 / 1e6, 2),
 "DP_params_from_fp32_bytes_M": round(338813432 / 4 / 1e6, 2),
 "ACT_anchors_per_demo": round(2205328 / 3003, 1),
}

# V16: 필요 에피소드 수 (두 비율 구분, 검정력 0.8, alpha 0.05, 양측 정규근사)
def n_per_group(p1, p2, za=1.96, zb=0.8416):
    pb = (p1 + p2) / 2
    return math.ceil(((za * math.sqrt(2 * pb * (1 - pb)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p1 - p2) ** 2)
R["V16_required_n_per_group"] = {
 "57_vs_45": n_per_group(0.57, 0.45), "64_vs_58": n_per_group(0.64, 0.58),
 "36_vs_30": n_per_group(0.36, 0.30), "85_vs_74": n_per_group(0.85, 0.74),
 "0.6_vs_0.1": n_per_group(0.6, 0.1),
}
# 시행 10회 실로봇에서 6/10 vs 1/10를 가를 수 있는 최소 차이
R["V16_real_10_trials_min_detectable"] = {
 "k_ACT=1_vs_k": {str(k): round(fisher_two_sided(1, 10, k, 10), 4) for k in range(3, 9)}
}

# V17: 프로젝트 페이지 LLM agent 표 (논문 v1에는 없음, 페이지만)
AG = {  # (L0 성공/18, L1+L2 성공/36, 시간(분), 중앙 비용, 성공당, 총)
 "CAP Opus 5.5": (14, 0, 1.35, 4.24, 59), "CAP GPT-6.1 Sol": (2, 0, 0.29, 7.90, 16),
 "CAP GPT-6 Astra": (2, 0, 1.51, 40.12, 80), "RUA Opus 5.5": (16, 1, 1.40, 4.44, 76),
 "RUA GPT-6.1 Sol": (18, 0, 0.46, 1.31, 24), "RUA GPT-6 Astra": (18, 0, 3.86, 11.00, 198),
}
ag = {}
for name, (k0, k12, med, per, tot) in AG.items():
    s = k0 + k12
    ag[name] = {"successes": s, "total_over_successes": round(tot / s, 2), "listed_per_success": per,
                "implied_total_from_per": round(per * s, 1), "listed_total": tot,
                "mean_cost_per_episode_54": round(tot / 54, 2), "listed_median": med}
R["V17_agent_costs"] = ag
# 페이지 L0 열 퍼센트(3 에피소드)로 L0 성공 수 재계산
pagerows = {"CAP Opus 5.5": ([67,67,100],[100,67,67]), "CAP GPT-6.1 Sol": ([33,33,0],[0,0,0]),
            "CAP GPT-6 Astra": ([33,0,0],[33,0,0]), "RUA Opus 5.5": ([67,100,100],[100,100,67]),
            "RUA GPT-6.1 Sol": ([100,100,100],[100,100,100]), "RUA GPT-6 Astra": ([100,100,100],[100,100,100])}
chk = {}
for name, (s, d) in pagerows.items():
    chk[name] = sum(round(x / 100 * 3) for x in s + d)
R["V17_agent_L0_counts_from_percent"] = chk
# 에이전트 이벤트 퍼센트 = k/18 가능성
infe = []
for m in "SD":
    for name, v in PAGE_EV[m].items():
        if "CAP" in name or "RUA" in name:
            for x in v:
                if abs(round(round(x / 100 * 18) / 18 * 100, 1) - x) > 1e-9:
                    infe.append((m, name, x))
R["V17_agent_event_integer_infeasible"] = infe
# L1/L2 에이전트 총 에피소드 수
R["V17_agent_episode_counts"] = {"L1L2_per_setup": 3 * 2 * 2 * 3, "all_setups": 6 * 36, "success_total": 1}

# V18: 사건 깔때기 조건부 비율 (Fig.5 값 기준, 근사)
fun = {}
for pol in FIG5:
    for m in "SD":
        c, f, l, t, s_ = FIG5[pol][m]
        fun[f"{pol}-{m}"] = {"lift_per_candidate": round(l / c, 3), "target_per_lift": round(t / l, 3),
                             "success_per_target": round(s_ / t, 3), "success_per_lift": round(s_ / l, 3)}
R["V18_event_funnel_ratios"] = fun

json.dump(R, open(os.path.join(HERE, "recompute_results.json"), "w"), indent=1, ensure_ascii=False, default=str)
print(json.dumps({k: R[k] for k in ("V1_table3_vs_page", "V2_fig5_vs_page", "V3_mismatches", "V4_fig5_integer_feasibility_infeasible")}, ensure_ascii=False, indent=1))
