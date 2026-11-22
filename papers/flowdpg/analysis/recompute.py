"""FlowDPG (arXiv:2606.22303 v2) 보고 수치의 재계산과 일관성 검사. numpy, scipy만 쓰며 1초 안에 끝난다.

실행: python3 analysis/recompute.py  -> analysis/recompute_results.json

검사 번호(V1...)는 analysis/03_analysis.md의 절과 같다.
  V1 표 1 Avg = 여덟 단계 비율의 평균인가
  V2 비율이 정수 개수로 가능한가 (표 1 단계별, 그림 4, 그림 5d, 그림 6, 표 2)
  V3 유의성 (Fisher 정확 검정, Wilson 구간)
  V4 그림 4 대 표 1 교차 (네 쌍이 모두 정확히 2pp 차이)
  V5 표 2, 표 3 산술, 보상 척도
"""
import json
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
res = {}

STAGES = ["grasp_case", "open_case", "grasp_R", "insert_R", "grasp_L", "insert_L", "close_case", "place_case"]
T1 = {  # 논문 표 1 (v2, v1과 동일): 단계 8개, Overall, Avg
    "BC": [100, 78.32, 72.44, 68.54, 78.56, 66.23, 92, 96, 64, 81.51],
    "RA-BC": [100, 86.15, 80.52, 64.52, 82.49, 68.46, 96, 96, 76, 84.27],
    "AWR": [100, 80.65, 78.43, 72.45, 78.77, 70.65, 100, 100, 76, 85.12],
    "RECAP": [100, 89.29, 80.62, 69.69, 76.54, 64.52, 96, 100, 72, 84.58],
    "DSRL": [100, 79.85, 75.47, 70.59, 78.13, 67.94, 92, 92, 68, 82.00],
    "PLD": [100, 85.27, 80.13, 79.82, 80.43, 79.59, 96, 96, 76, 87.16],
    "RLT": [100, 86.93, 82.41, 77.92, 84.14, 80.18, 96, 100, 80, 88.45],
    "QAM": [100, 88, 83.33, 70.58, 73.52, 80, 100, 100, 80, 86.93],
    "FlowDPG_off": [100, 89.29, 90.63, 84.38, 96.43, 83.33, 100, 100, 88, 93.01],
    "FlowDPG_on": [100, 89.29, 92.31, 86.21, 96.67, 88.89, 100, 100, 92, 94.17],
}

# ---- V1: Avg 재계산
v1 = {}
for k, row in T1.items():
    avg = float(np.mean(row[:8]))
    v1[k] = {"reported": row[9], "recomputed": round(avg, 3), "diff": round(avg - row[9], 3)}
res["V1_avg"] = v1
res["V1_max_abs_diff"] = max(abs(v["diff"]) for v in v1.values())

# ---- V2a: 표 1 Overall은 25 에피소드의 정수 개수인가
v2a = {}
for k, row in T1.items():
    x = row[8] * 25 / 100
    v2a[k] = {"overall": row[8], "count_of_25": x, "integer": abs(x - round(x)) < 1e-9}
res["V2a_overall_integer"] = v2a


# ---- V2b: 단계별 비율 = k/n (n은 재시도 포함 시도 수) 로 가능한가
def feasible(value, n_max=75, k_cap=None):
    out = []
    for n in range(1, n_max + 1):
        for k in range(0, n + 1):
            if k_cap is not None and k > k_cap:
                break
            if abs(100 * k / n - value) < 0.005 + 1e-9:
                out.append((k, n))
    return out


v2b = {}
n_tested = 0
for name, row in T1.items():
    cells = {}
    for st, val in zip(STAGES, row[:8]):
        if abs(val * 25 / 100 - round(val * 25 / 100)) < 1e-9:  # 25 에피소드의 정수 비율(4의 배수 %)은 건너뛴다
            continue
        n_tested += 1
        f_any = feasible(val, 75)
        f_cap25 = feasible(val, 75, k_cap=25)
        f_cap50 = feasible(val, 75, k_cap=50)
        big = None
        for n in range(76, 1000):
            if any(abs(100 * k / n - val) < 0.005 + 1e-9 for k in range(n + 1)):
                big = n
                break
        cells[st] = {
            "value": val,
            "smallest_k_n_le75": f_any[0] if f_any else None,
            "smallest_n_any": big if not f_any else f_any[0][1],
            "feasible_k_le_25": f_cap25[:3],
            "feasible_k_le_50": f_cap50[:3],
        }
    v2b[name] = cells
res["V2b_cells_tested_non_multiple_of_4pct"] = n_tested
res["V2b_stage_fractions"] = v2b
infeasible25 = [(n, s, c["value"]) for n, cs in v2b.items() for s, c in cs.items() if not c["feasible_k_le_25"]]
infeasible50 = [(n, s, c["value"]) for n, cs in v2b.items() for s, c in cs.items() if not c["feasible_k_le_50"]]
res["V2b_infeasible_if_k_le_25"] = infeasible25
res["V2b_infeasible_if_k_le_50"] = infeasible50
res["V2b_infeasible_any_n_le_75"] = [(n, s, c["value"], c["smallest_n_any"]) for n, cs in v2b.items() for s, c in cs.items() if c["smallest_k_n_le75"] is None]
res["V2b_by_method_infeasible_n_le_75"] = {n: sum(1 for c in cs.values() if c["smallest_k_n_le75"] is None) for n, cs in v2b.items()}
res["V2b_by_method_tested"] = {n: len(cs) for n, cs in v2b.items()}

# ---- V2c: 그림 4, 그림 5d, 그림 6, 표 2 정수 가능성
F4_succ = {"BC": 62, "QAM_off": 78, "QAM_on": 80, "FlowDPG_off": 86, "FlowDPG_on": 90}  # n=50
res["V2c_fig4_success_counts_of_50"] = {k: v * 50 / 100 for k, v in F4_succ.items()}
F4_rub_air = {"BC": 66.6, "QAM_off": 79.4, "QAM_on": 81.4, "FlowDPG_off": 88.2, "FlowDPG_on": 92.3}
F4_rub_egg = {"BC": 64.2, "QAM_off": 76.9, "FlowDPG_off": 85.8}


def rubric_points(pct, per_rollout_points, n=50):
    total = per_rollout_points * n
    lo, hi = (pct - 0.05) * total / 100, (pct + 0.05) * total / 100
    ints = [i for i in range(int(np.ceil(lo)), int(np.floor(hi)) + 1)]
    return ints


res["V2c_rubric_integer_points"] = {
    "airpods_24pt_x50": {k: rubric_points(v, 24) for k, v in F4_rub_air.items()},
    "egg_42pt_x50": {k: rubric_points(v, 42) for k, v in F4_rub_egg.items()},
}
F5d = {"w/o both": 72, "w/o shift": 76, "w/o cons": 80, "Full": 92}  # n=25
F6 = {
    "Full": [100, 96, 96, 96, 96, 92, 92, 92],
    "w/o progress": [100, 96, 92, 88, 84, 84, 80, 80],
    "w/o stage transition": [100, 96, 92, 88, 84, 80, 76, 76],
    "Only terminal": [100, 80, 76, 72, 68, 64, 60, 60],
}
res["V2c_fig5d_counts_of_25"] = {k: v * 25 / 100 for k, v in F5d.items()}
res["V2c_fig6_counts_of_25"] = {k: [x * 25 / 100 for x in row] for k, row in F6.items()}
res["V2c_fig6_all_integer"] = all(abs(x * 25 / 100 - round(x * 25 / 100)) < 1e-9 for row in F6.values() for x in row)
res["V2c_fig6_monotone"] = all(all(a >= b for a, b in zip(row, row[1:])) for row in F6.values())
res["V2c_fig6_final_vs_fig5d_and_table1"] = {
    "Full_final": F6["Full"][-1], "table1_online_overall": T1["FlowDPG_on"][8],
    "w/o progress_final": F6["w/o progress"][-1], "w/o stage_final": F6["w/o stage transition"][-1],
    "only_terminal_final": F6["Only terminal"][-1], "BC_overall": T1["BC"][8],
}
T2 = {"re-grasp": (78, 92, 14), "re-open/close": (82, 94, 12), "re-insert": (86, 92, 6), "adapt grasp": (70, 88, 18)}  # n=50
res["V2c_table2"] = {k: {"offline_count": a * 0.5, "online_count": b * 0.5, "delta_ok": (b - a) == d} for k, (a, b, d) in T2.items()}

# ---- V3: 유의성
def fisher(k1, n1, k2, n2):
    odds, p = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
    return round(p, 4)


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


v3 = {}
# 표 1 Overall (n = 25)
c25 = {k: round(T1[k][8] * 25 / 100) for k in T1}
v3["table1_overall_wilson"] = {k: {"k": c25[k], "ci95": wilson(c25[k], 25)} for k in ("BC", "QAM", "RLT", "FlowDPG_off", "FlowDPG_on")}
v3["table1_fisher_p"] = {
    "FlowDPG_on_vs_BC": fisher(c25["FlowDPG_on"], 25, c25["BC"], 25),
    "FlowDPG_off_vs_BC": fisher(c25["FlowDPG_off"], 25, c25["BC"], 25),
    "FlowDPG_off_vs_QAM": fisher(c25["FlowDPG_off"], 25, c25["QAM"], 25),
    "FlowDPG_off_vs_RLT": fisher(c25["FlowDPG_off"], 25, c25["RLT"], 25),
    "FlowDPG_on_vs_QAM_offline": fisher(c25["FlowDPG_on"], 25, c25["QAM"], 25),
    "FlowDPG_on_vs_off": fisher(c25["FlowDPG_on"], 25, c25["FlowDPG_off"], 25),
    "QAM_vs_BC": fisher(c25["QAM"], 25, c25["BC"], 25),
}
# 그림 4 (n = 50)
c50 = {k: round(v * 50 / 100) for k, v in F4_succ.items()}
v3["fig4_success_wilson"] = {k: {"k": c50[k], "ci95": wilson(c50[k], 50)} for k in c50}
v3["fig4_fisher_p"] = {
    "FlowDPG_on_vs_QAM_on": fisher(c50["FlowDPG_on"], 50, c50["QAM_on"], 50),
    "FlowDPG_off_vs_QAM_off": fisher(c50["FlowDPG_off"], 50, c50["QAM_off"], 50),
    "FlowDPG_off_vs_BC": fisher(c50["FlowDPG_off"], 50, c50["BC"], 50),
    "FlowDPG_on_vs_BC": fisher(c50["FlowDPG_on"], 50, c50["BC"], 50),
    "FlowDPG_on_vs_off": fisher(c50["FlowDPG_on"], 50, c50["FlowDPG_off"], 50),
    "QAM_on_vs_off": fisher(c50["QAM_on"], 50, c50["QAM_off"], 50),
    "QAM_off_vs_BC": fisher(c50["QAM_off"], 50, c50["BC"], 50),
}
# 그림 5d (n = 25)
c5 = {k: round(v * 25 / 100) for k, v in F5d.items()}
v3["fig5d_fisher_p_vs_full"] = {k: fisher(c5["Full"], 25, c5[k], 25) for k in c5 if k != "Full"}
v3["fig5d_wilson"] = {k: {"k": c5[k], "ci95": wilson(c5[k], 25)} for k in c5}
v3["fig5d_cons_vs_shift_p"] = fisher(c5["w/o cons"], 25, c5["w/o shift"], 25)
v3["fig5d_shift_vs_both_p"] = fisher(c5["w/o shift"], 25, c5["w/o both"], 25)
# 그림 6 최종 (n = 25)
v3["fig6_only_terminal_vs_BC"] = {"only_terminal_k": 15, "BC_k": 16, "p": fisher(15, 25, 16, 25)}
v3["fig6_full_vs_only_terminal_p"] = fisher(23, 25, 15, 25)
# 표 2 (n = 50, 같은 에피소드 쌍이 아니므로 독립 검정으로 근사)
v3["table2_fisher_p"] = {k: fisher(round(b * 0.5), 50, round(a * 0.5), 50) for k, (a, b, d) in T2.items()}
# 보고된 이항 표준오차 대비 그림 4의 오차막대 해석: 체크포인트 5개의 평균에 대한 s.e.
v3["fig4_note"] = "오차막대는 한 학습 실행의 체크포인트 5개 평균의 s.e.이므로 시드 변동을 포함하지 않는다"
res["V3"] = v3

# ---- V3b: 검정력. n=50, p1=0.9 대 p2=0.8, 양측 5% 로 차이를 잡을 확률 (모의)
rng = np.random.default_rng(0)


def power(p1, p2, n, trials=4000):
    hit = 0
    for _ in range(trials):
        a = rng.binomial(n, p1)
        b = rng.binomial(n, p2)
        if fisher(a, n, b, n) < 0.05:
            hit += 1
    return hit / trials


res["V3b_power"] = {
    "n50_0.90_vs_0.80": power(0.90, 0.80, 50),
    "n25_0.92_vs_0.80": power(0.92, 0.80, 25),
    "n25_0.92_vs_0.72": power(0.92, 0.72, 25),
}
# 필요한 n (검정력 0.8)
need = {}
for n in range(25, 400, 25):
    if power(0.90, 0.80, n, trials=1000) >= 0.8:
        need = {"n_per_arm_for_80pct_power_0.90_vs_0.80": n}
        break
res["V3b_required_n"] = need

# ---- V4: 그림 4 대 표 1 교차. 네 쌍의 차이가 모두 정확히 -2pp
pairs = {"BC": (64, 62), "QAM_off": (80, 78), "FlowDPG_off": (88, 86), "FlowDPG_on": (92, 90)}
res["V4_pairs_table1_minus_fig4"] = {k: a - b for k, (a, b) in pairs.items()}
pm = 1.0
detail = {}
for k, (a, b) in pairs.items():
    p = (a + b) / 200  # 공통 성공률 추정
    # 표 1: Bin(25,p)/25, 그림 4: Bin(50,p)/50. 그림 4 - 표 1 = -0.02 일 확률
    pr = 0.0
    for x in range(26):
        for y in range(51):
            if abs(y / 50 - x / 25 + 0.02) < 1e-9:
                pr += stats.binom.pmf(x, 25, p) * stats.binom.pmf(y, 50, p)
    detail[k] = round(float(pr), 4)
    pm *= pr
res["V4_prob_each_diff_exact_minus2"] = detail
res["V4_prob_all_four_independent"] = float(pm)
res["V4_sd_of_difference_pp"] = round(float(np.sqrt(0.85 * 0.15 * (1 / 25 + 1 / 50)) * 100), 2)

# ---- V5: 표 2, 3 산술, 일관성 문장
t3 = {
    "air_spread": (1.61, 1.51, -0.10, 0.03), "egg_spread": (0.87, 0.85, -0.02, 0.01),
    "air_wass": (9.93, 10.02, 0.09, 0.04), "egg_wass": (7.35, 7.48, 0.13, 0.05),
    "air_Q": (32.79, 37.66, 4.87, 1.24), "egg_Q": (53.84, 59.66, 5.82, 1.54),
}
res["V5_table3"] = {k: {"after_minus_before": round(b - a, 3), "reported_delta": d, "arith_ok": abs((b - a) - d) < 0.0051,
                        "delta_over_se": round(d / se, 2)} for k, (a, b, d, se) in t3.items()}
res["V5_table3_relative"] = {"air_wass_pct": round(0.09 / 9.93 * 100, 2), "egg_wass_pct": round(0.13 / 7.35 * 100, 2),
                              "air_spread_pct": round(-0.10 / 1.61 * 100, 2), "egg_spread_pct": round(-0.02 / 0.87 * 100, 2),
                              "air_Q_above_demo": round(37.66 - 33.58, 2), "air_Q_before_vs_demo": round(32.79 - 33.58, 2),
                              "egg_Q_above_demo": round(59.66 - 54.24, 2), "egg_Q_before_vs_demo": round(53.84 - 54.24, 2)}
# Spearman rho 의 s.e. (50 롤아웃): 0.36 +- 0.05 는 z = 7.2, 0.28 +- 0.03 은 z = 9.3
res["V5_spearman_z"] = {"air": round(0.36 / 0.05, 1), "egg": round(0.28 / 0.03, 1)}

# 보상 척도: v1 부록 표 3의 보상 값 (beta_stage = 1.0, 성공 +5.0, 실패 -1.0), 8개 단계, 진행도는 0에서 1
beta, succ, n_stage = 1.0, 5.0, 8
max_return_air = 1.0 + beta * n_stage + succ  # 진행도 전체 합 <= 1 (작은 증가분의 합), 단계 보너스 8, 종료 +5
res["V5_reward_scale_v1_hparams"] = {
    "max_undiscounted_return_airpods": max_return_air,
    "reported_Q_demo_airpods": 33.58, "reported_Q_demo_egg": 54.24,
    "ratio_Q_over_max_air": round(33.58 / max_return_air, 2),
    "egg_14_stage_max": 1.0 + beta * 14 + succ,
    "ratio_Q_over_max_egg": round(54.24 / (1.0 + beta * 14 + succ), 2),
    "fig7_V_range": [30, 42.5],
}
# 청크 할인 gamma^H
res["V5_chunk_discount"] = {"gamma": 0.99, "H": 30, "gamma_pow_H": round(0.99 ** 30, 4)}

# 식 (4)의 항등식 a_hat - x1 = (1-t)(v - u)
rng = np.random.default_rng(1)
x0, x1 = rng.normal(size=(1000, 6)), rng.normal(size=(1000, 6))
t = rng.uniform(size=(1000, 1))
xt = (1 - t) * x0 + t * x1
u = x1 - x0
v = u + rng.normal(size=u.shape) * 0.3
ahat = xt + (1 - t) * v
res["V5_identity_ahat_minus_x1_max_err"] = float(np.abs((ahat - x1) - (1 - t) * (v - u)).max())

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print("V1 max |diff|:", res["V1_max_abs_diff"])
print("V2a all integer:", all(v["integer"] for v in v2a.values()))
print("V2b tested:", n_tested, "infeasible k<=25:", len(infeasible25), "k<=50:", len(infeasible50), "any n<=75:", len(res["V2b_infeasible_any_n_le_75"]))
print("V2b by method:", res["V2b_by_method_infeasible_n_le_75"], res["V2b_by_method_tested"])
print("V3 table1:", v3["table1_fisher_p"])
print("V3 fig4:", v3["fig4_fisher_p"])
print("V3 fig5d:", v3["fig5d_fisher_p_vs_full"])
print("V3b power:", res["V3b_power"], res["V3b_required_n"])
print("V4:", res["V4_pairs_table1_minus_fig4"], res["V4_prob_each_diff_exact_minus2"], res["V4_prob_all_four_independent"])
print("V5 reward:", res["V5_reward_scale_v1_hparams"])
print("wrote", OUT)
