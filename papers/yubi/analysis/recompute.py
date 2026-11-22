"""YUBI (arXiv:2606.10244 v1) 보고 수치 재계산과 검정.

실행: python3 analysis/recompute.py  -> analysis/recompute_results.json (수 초, numpy/scipy)
입력은 논문에서 옮긴 값이다(그림 5, 6은 막대/점 위 숫자를 읽은 것).
검사 번호는 V1..Vn (분석 문서 03과 같은 번호).
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
R = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def fisher(a, n1, b, n2):
    return stats.fisher_exact([[a, n1 - a], [b, n2 - b]])[1]


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


# ---------------- V1, V2: 그림 5 (손 크기별 육각너트 집기, n = 50) ----------------
sizes = ["M3", "M4", "M5", "M6", "M8", "M10"]
umi = [14, 46, 74, 70, 100, 98]
yubi = [44, 36, 84, 90, 94, 100]
N = 50
feas = {s: [(p * N / 100).is_integer() for p in (u, y)] for s, u, y in zip(sizes, umi, yubi)}
ku = [round(p * N / 100) for p in umi]
ky = [round(p * N / 100) for p in yubi]
p_cell = [fisher(y, N, u, N) for y, u in zip(ky, ku)]
p_holm = holm(p_cell)
R["V1_fig5"] = {
    "sizes": sizes, "umi_pct": umi, "yubi_pct": yubi, "n_per_cell": N,
    "integer_feasible_all": all(all(v) for v in feas.values()),
    "diff_pp": [y - u for y, u in zip(yubi, umi)],
    "ratio_M3": round(yubi[0] / umi[0], 3),
    "text_claims": {"M6_plus20": yubi[3] - umi[3] == 20, "M5_plus10": yubi[2] - umi[2] == 10,
                    "ceiling_ge94_M8_M10": min(umi[4], umi[5], yubi[4], yubi[5]) >= 94,
                    "umi_M6_lt_M5": umi[3] < umi[2], "yubi_M4_lt_M3": yubi[1] < yubi[0],
                    "umi_beats_yubi_sizes": [s for s, u, y in zip(sizes, umi, yubi) if u > y]},
}
R["V2_fig5_tests"] = {
    "yubi_k": ky, "umi_k": ku,
    "wilson_yubi": [wilson(k, N) for k in ky], "wilson_umi": [wilson(k, N) for k in ku],
    "fisher_p": [round(p, 4) for p in p_cell], "holm_p": [round(p, 4) for p in p_holm],
}
# 합산과 Cochran-Mantel-Haenszel (연속성 보정 없음, 크기별 층)
ty, tu = sum(ky), sum(ku)
R["V2_fig5_tests"]["pooled"] = {"yubi": f"{ty}/{6*N}", "umi": f"{tu}/{6*N}",
                                "yubi_pct": round(100 * ty / (6 * N), 1), "umi_pct": round(100 * tu / (6 * N), 1),
                                "fisher_p": round(fisher(ty, 6 * N, tu, 6 * N), 4)}
num = den = 0.0
for a, b in zip(ky, ku):
    n1 = n2 = N
    m1 = a + b
    t = n1 + n2
    num += a - n1 * m1 / t
    den += n1 * n2 * m1 * (t - m1) / (t * t * (t - 1))
chi = num * num / den
R["V2_fig5_tests"]["cmh_chi2"] = round(chi, 3)
R["V2_fig5_tests"]["cmh_p"] = round(float(stats.chi2.sf(chi, 1)), 4)
# 작업자 군집 민감도: 설계 효과 DE = 1 + (m-1) rho, m = 5 (작업자당 시도)
sens = {}
for rho in (0.0, 0.2, 0.5):
    de = 1 + 4 * rho
    a, b = ty / de, tu / de
    n = 6 * N / de
    z = (a / n - b / n) / math.sqrt((a + b) / (2 * n) * (1 - (a + b) / (2 * n)) * 2 / n)
    sens[str(rho)] = {"de": de, "n_eff_per_arm": round(n, 1), "z": round(z, 2), "p_two_sided": round(2 * float(stats.norm.sf(abs(z))), 4)}
R["V2_fig5_tests"]["pooled_cluster_sensitivity"] = sens
# M3와 M6 한 칸 검사에 대한 군집 민감도 (rho = 0.2)
de = 1.8
R["V2_fig5_tests"]["M3_cluster_rho0.2_p"] = round(fisher(round(ky[0] / de), round(N / de), round(ku[0] / de), round(N / de)), 4)

# ---------------- V3: 그림 6 (작업 완료 시간, 초) ----------------
tasks = ["domino", "pencil_case", "tissue_box", "whiteboard_ABCDE", "phone_charge"]
hand = [13.7, 20.1, 18.1, 12.1, 7.3]
umi_t = [36.4, 65.4, 105.6, 70.0, 99.8]
yubi_t = [26.5, 36.6, 70.65, 41.5, 23.8]
sp = [u / y for u, y in zip(umi_t, yubi_t)]
gap_closed = [(u - y) / (u - h) for u, y, h in zip(umi_t, yubi_t, hand)]
R["V3_fig6"] = {
    "tasks": tasks, "speedup_umi_over_yubi": [round(x, 3) for x in sp],
    "min_max_match_text": [round(min(sp), 2) == 1.37, round(max(sp), 2) == 4.19],
    "geomean_speedup": round(float(np.exp(np.mean(np.log(sp)))), 3),
    "pooled_time_ratio": round(sum(umi_t) / sum(yubi_t), 3),
    "yubi_over_hand": [round(y / h, 2) for y, h in zip(yubi_t, hand)],
    "umi_over_hand": [round(u / h, 2) for u, h in zip(umi_t, hand)],
    "gap_closed_fraction": [round(g, 3) for g in gap_closed],
    "design": {"operators": 10, "tasks_per_operator": 2, "operators_per_task": 10 * 2 / 5, "trials_per_cell": 5,
               "cell_n_trials": 4 * 5, "sd_reported": False, "test_reported": False},
}

# ---------------- V4: 표 3, 표 5 (로봇 배치, 20회) ----------------
t3 = {"ball_basket_UR": 20, "cup_pyramid_UR": 13, "unfold_glasses_UR": 9, "socks_Franka": 18, "tape_box_Franka": 18, "cup_place_ELEY": 11}
R["V4_table3"] = {"k_of_20": t3, "wilson": {k: wilson(v, 20) for k, v in t3.items()},
                  "total": f"{sum(t3.values())}/120", "mean_pct": round(100 * sum(t3.values()) / 120, 1),
                  "wilson_total": wilson(sum(t3.values()), 120),
                  "by_robot": {"UR": f"{20+13+9}/60", "Franka": "36/40", "ELEY": "11/20"}}
dp = {"ball": 20, "cup_pyramid": 9, "glasses": 0}
pi = {"ball": 20, "cup_pyramid": 13, "glasses": 9}
pp = [fisher(pi[k], 20, dp[k], 20) for k in dp]
R["V4_table5"] = {"fisher_p_pi05_vs_dp": {k: round(p, 4) for k, p in zip(dp, pp)},
                  "holm_p": [round(x, 4) for x in holm(pp)],
                  "pi05_equals_table3_UR_cells": pi["cup_pyramid"] == 13 and pi["glasses"] == 9 and pi["ball"] == 20,
                  "pooled_pi05": "42/60", "pooled_dp": "29/60", "pooled_fisher_p": round(fisher(42, 60, 29, 60), 4)}
# 표 3 캡션은 다중 과제 정책, 표 5 캡션은 과제별 미세 조정인데 UR 세 칸이 같다
demos = [408, 275, 194, 1903, 1691, 3985]
succ = [20, 13, 9, 18, 18, 11]
rho_s, p_s = stats.spearmanr(demos, succ)
R["V7_post_training"] = {"demos": demos, "total": sum(demos), "fraction_of_1.20M_episodes_pct": round(100 * sum(demos) / 1.20e6, 3),
                         "spearman_demos_vs_success": [round(float(rho_s), 3), round(float(p_s), 3)],
                         "note": "n=6, 과제와 로봇이 섞여 있어 인과 해석 불가"}

# ---------------- V5: 표 2 일관성 ----------------
eps, hrs, avg_s, demos_m, sub_task = 1.20e6, 8434, 39.8, 6.80e6, 7.99
R["V5_table2"] = {
    "hours_per_episode_s": round(hrs * 3600 / eps, 2),
    "implied_hours_from_avg_39.8s": round(eps * avg_s / 3600, 0),
    "ratio_implied_over_reported": round(eps * avg_s / 3600 / hrs, 3),
    "demos_per_episode": round(demos_m / eps, 3),
    "episodes_x_7.99": round(eps * sub_task / 1e6, 3),
    "seconds_per_subaction": round(hrs * 3600 / demos_m, 2),
    "avg_episode_duration_from_subaction_model_s": round(hrs * 3600 / eps, 2),
}
# 수집 처리량
desks, days, ops = 22, 61, 179
R["V10_throughput"] = {
    "hours_per_desk": round(hrs / desks, 1), "hours_per_operator": round(hrs / ops, 1),
    "max_desk_hours_24x7_61d": desks * 24 * days, "utilization_pct": round(100 * hrs / (desks * 24 * days), 1),
    "episodes_per_desk_per_day": round(eps / desks / days, 0),
    "arena_page_rigs": 40, "paper_desks": desks,
}

# ---------------- V6: 표 4와 본문 비교 ----------------
R["V6_table4"] = {
    "tasks_vs_fastumi100k": round(119 / 32, 2), "demos_vs_fastumi100k": round(6800 / 92.8, 1),
    "hours_vs_fastumi100k": round(8434 / 600, 2), "demos_vs_dexwild": round(6800 / 9.5, 0),
    "demos_vs_genrobot": round(6800 / 790, 2), "hours_vs_genrobot_13k": round(8434 / 13000, 2),
    "episodes_vs_genrobot_demos": round(1.2e6 / 790e3, 2),
    "note": "YUBI의 demo는 하위 행동(video-language-action 쌍) 단위, 다른 열은 기록 단위여서 단위가 다름",
}

# ---------------- V8, V9, V12: 무게, 표 1 대 표 4, 비용 ----------------
R["V8_weight"] = {"gripper_plus_controller_g": 200 + 119, "paper_total_g": 319, "ratio_umi_780": round(780 / 319, 2),
                  "ratio_activeumi_905": round(905 / 319, 2)}
R["V9_table1_vs_table4"] = {"UMI_tasks": {"table1_and_sec4_text": 4, "table4": 5},
                            "exUMI_tasks": {"table1": 9, "table4": 8}, "ActiveUMI_tasks": {"table1": 6, "table4": 6}}
R["V12_cost"] = {"paper_gripper_usd_lt": 200, "arena_page_full_robot_setup_usd_lt": 600,
                 "note": "BOM에 가격 열 없음. 두 문장은 다른 대상(그리퍼 하나 대 로봇 전체)이라 직접 비교 불가"}
OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(R, indent=1, ensure_ascii=False))
