"""논문 표·본문 수치의 재계산과 일관성 검사 (numpy, scipy, 1초 미만).

대상: RARM (arXiv:2606.22027) v5 표 1(실로봇), 표 2(LIBERO-10 회복), 그림 5 비율, 표 4–7, 부록 F의 정수·단위 관계.
v1과 v5의 실로봇 표 비교(10회 평가 대 30회 평가)도 포함한다.
검사 번호는 V1, V2, ... 이고 analysis/03_analysis.md 의 절과 같다.

실행: python3 analysis/recompute.py   → analysis/recompute_results.json
"""
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
res = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


# ---------------------------------------------------------------- V1. 표 1 (v5): 실로봇 30회 평가
BUDGETS = [30, 60, 90]
T1 = {  # 과제: (w/o DSRL, {예산: (BinR, RoboM, Ours)})  분모는 모두 30
    "Drawer open": (10, {30: (16, 17, 19), 60: (19, 18, 23), 90: (23, 20, 29)}),
    "Pick eraser": (5, {30: (9, 13, 17), 60: (15, 17, 21), 90: (20, 20, 25)}),
    "Hand over": (2, {30: (5, 3, 14), 60: (12, 4, 22), 90: (14, 5, 23)}),
    "Clothes folding": (2, {30: (2, 4, 4), 60: (3, 9, 12), 90: (7, 9, 25)}),
}
N = 30
v1 = {"cells": [], "claims": {}}
for task, (base, d) in T1.items():
    for b in BUDGETS:
        binr, rob, ours = d[b]
        cell = {
            "task": task, "budget": b, "BinR": binr, "RoboM": rob, "Ours": ours,
            "ours_ci": [round(x, 3) for x in wilson(ours, N)],
            "binr_ci": [round(x, 3) for x in wilson(binr, N)],
            "robom_ci": [round(x, 3) for x in wilson(rob, N)],
            "p_ours_vs_binr": round(fisher(ours, N, binr, N), 4),
            "p_ours_vs_robom": round(fisher(ours, N, rob, N), 4),
            "p_ours_vs_base": round(fisher(ours, N, base, N), 4),
            "ours_gt_both": ours > binr and ours > rob,
        }
        v1["cells"].append(cell)
cells = v1["cells"]
v1["n_cells"] = len(cells)
v1["sig_vs_binr_p05"] = sum(c["p_ours_vs_binr"] < 0.05 for c in cells)
v1["sig_vs_robom_p05"] = sum(c["p_ours_vs_robom"] < 0.05 for c in cells)
v1["sig_vs_binr_bonf"] = sum(c["p_ours_vs_binr"] < 0.05 / 24 for c in cells)
v1["sig_vs_robom_bonf"] = sum(c["p_ours_vs_robom"] < 0.05 / 24 for c in cells)
v1["nonsig_cells_vs_binr"] = [(c["task"], c["budget"], c["p_ours_vs_binr"]) for c in cells if c["p_ours_vs_binr"] >= 0.05]
v1["nonsig_cells_vs_robom"] = [(c["task"], c["budget"], c["p_ours_vs_robom"]) for c in cells if c["p_ours_vs_robom"] >= 0.05]
# 본문 주장 검사
three = ["Drawer open", "Pick eraser", "Hand over"]
gains = [T1[t][1][b][2] - T1[t][1][b][0] for t in three for b in BUDGETS]
v1["claims"]["gain_over_binr_range_three_tasks"] = [min(gains), max(gains)]  # 본문: 3–10
v1["claims"]["dominates_both_all_budgets_three_tasks"] = all(
    T1[t][1][b][2] > max(T1[t][1][b][0], T1[t][1][b][1]) for t in three for b in BUDGETS)
v1["claims"]["smallest_budget_pairs"] = {t: [T1[t][1][30][2], T1[t][1][30][0]] for t in three}  # 19 vs 16, 17 vs 9, 14 vs 5
v1["claims"]["clothes_ratio_vs_binr_90"] = round(25 / 7, 2)   # 본문 "3배 이상"
v1["claims"]["clothes_ratio_vs_robom_90"] = round(25 / 9, 2)  # 본문 "2.8배"
v1["claims"]["base_max_share"] = round(max(v[0] for v in T1.values()) / N, 3)  # 본문 "최대 1/3"
v1["claims"]["base_long_horizon_share"] = [T1["Hand over"][0] / N, T1["Clothes folding"][0] / N]  # "10분의 1 미만"
v1["claims"]["robom_below_binr_handover_all_budgets"] = all(T1["Hand over"][1][b][1] < T1["Hand over"][1][b][0] for b in BUDGETS)
v1["claims"]["robom_vs_binr_cells"] = {  # 본문: "일부 단기 설정에서 BinR보다 좋음"
    t: [T1[t][1][b][1] - T1[t][1][b][0] for b in BUDGETS] for t in T1}
# 예산별 4과제 합산 (과제를 교환 가능하다고 가정한 거친 합산)
pooled = {}
for b in BUDGETS:
    o = sum(T1[t][1][b][2] for t in T1); bi = sum(T1[t][1][b][0] for t in T1); r = sum(T1[t][1][b][1] for t in T1)
    pooled[b] = {"ours": o, "binr": bi, "robom": r, "n": 4 * N,
                 "p_vs_binr": float(f"{fisher(o, 4 * N, bi, 4 * N):.2e}"), "p_vs_robom": float(f"{fisher(o, 4 * N, r, 4 * N):.2e}")}
v1["pooled_four_tasks"] = pooled
# 최대 격차(Ours-RoboM)와 학습 구간: 30회 예산의 앞 10회는 무작위 노이즈(부록 F.3)
v1["learning_episodes_at_budget"] = {b: b - 10 for b in BUDGETS}
res["V1_table1_realworld"] = v1

# ---------------------------------------------------------------- V2. v1(10회 평가)과 v5(30회 평가)의 실로봇 표
T1_v1 = {  # 과제: (w/o, {예산: (BinR, Ours)})  분모 10
    "Drawer open": (2, {30: (4, 6), 60: (6, 7), 90: (8, 10)}),
    "Pick eraser": (2, {30: (3, 6), 60: (5, 8), 90: (8, 9)}),
    "Hand over": (2, {30: (2, 5), 60: (4, 6), 90: (5, 6)}),
    "Clothes folding": (1, {30: (0, 1), 60: (1, 3), 90: (2, 8)}),
}
vv = {"cells": []}
for task in T1_v1:
    b0 = T1_v1[task][0]
    vv["cells"].append({"task": task, "budget": "base", "v1": f"{b0}/10", "v5": f"{T1[task][0]}/30",
                        "p_fisher": round(fisher(b0, 10, T1[task][0], 30), 3)})
    for b in BUDGETS:
        for who, i1, i5 in (("BinR", 0, 0), ("Ours", 1, 2)):
            k1 = T1_v1[task][1][b][i1]; k5 = T1[task][1][b][i5]
            vv["cells"].append({"task": task, "budget": b, "who": who, "v1": f"{k1}/10", "v5": f"{k5}/30",
                                "p_fisher": round(fisher(k1, 10, k5, 30), 3)})
vv["min_p"] = min(c["p_fisher"] for c in vv["cells"])
vv["n_p_below_0.1"] = sum(c["p_fisher"] < 0.1 for c in vv["cells"])
vv["v1_gain_over_binr"] = {t: [T1_v1[t][1][b][1] - T1_v1[t][1][b][0] for b in BUDGETS] for t in T1_v1}
vv["v1_ours_vs_binr_p_at_90"] = {t: round(fisher(T1_v1[t][1][90][1], 10, T1_v1[t][1][90][0], 10), 3) for t in T1_v1}

# v1의 10회와 v5의 30회가 같은 시행을 포함하는지 거친 검정: d = 3*k1 - k5.
# 독립 재수집이면 Var(d) = 120 p(1-p), v1의 10회가 30회에 포함되면 Var(d) = 60 p(1-p) (p는 칸별 합동 추정).
dsq_ind, dsq_nest, ds = 0.0, 0.0, []
cnt = 0
for task in T1_v1:
    pairs = [(T1_v1[task][0], T1[task][0])]
    for b in BUDGETS:
        pairs += [(T1_v1[task][1][b][0], T1[task][1][b][0]), (T1_v1[task][1][b][1], T1[task][1][b][2])]
    for k1, k5 in pairs:
        ph = (k1 + k5) / 40
        v = ph * (1 - ph)
        if v == 0:
            continue
        d = 3 * k1 - k5
        ds.append(d)
        dsq_ind += d * d / (120 * v)
        dsq_nest += d * d / (60 * v)
        cnt += 1
vv["nested_test"] = {"cells": cnt, "chi2_if_independent": round(dsq_ind, 1), "chi2_if_nested": round(dsq_nest, 1),
                     "sd_of_3k1_minus_k5": round(float(np.std(ds)), 2),
                     "p_lower_tail_independent": float(f"{stats.chi2.cdf(dsq_ind, cnt):.3f}"),
                     "p_lower_tail_nested": float(f"{stats.chi2.cdf(dsq_nest, cnt):.3f}")}
res["V2_v1_vs_v5_realworld"] = vv

# ---------------------------------------------------------------- V3. 표 2: LIBERO-10 회복 (4 시드, ± 정의 없음)
T2 = {
    "GVL": ([45.1, 38.7, 26.3, 17.5], [8.8, 8.4, 6.0, 4.5]),
    "RoboDopamine": ([36.7, 31.9, 23.2, 15.1], [8.1, 7.4, 5.4, 3.4]),
    "RoboMeter": ([48.8, 40.5, 34.7, 23.7], [6.2, 4.9, 4.0, 3.7]),
    "TemporalOT": ([22.6, 18.6, 15.8, 14.1], [5.6, 4.9, 4.1, 4.0]),
    "RARM": ([95.6, 90.9, 78.9, 48.2], [1.0, 1.5, 2.2, 4.8]),
}
v2 = {"retention_20cm_over_none": {}, "monotone": {}, "welch": {}}
for k, (m, s) in T2.items():
    v2["retention_20cm_over_none"][k] = round(m[3] / m[0], 3)
    v2["monotone"][k] = all(m[i] > m[i + 1] for i in range(3))
n_seed = 4
for who in ("RoboMeter", "GVL"):
    for col, name in enumerate(["none", "5cm", "10cm", "20cm"]):
        m1, s1 = T2["RARM"][0][col], T2["RARM"][1][col]
        m2, s2 = T2[who][0][col], T2[who][1][col]
        out = {}
        for label, sd1, sd2 in (("pm=SD", s1, s2), ("pm=SEM", s1 * math.sqrt(n_seed), s2 * math.sqrt(n_seed))):
            t, p = stats.ttest_ind_from_stats(m1, sd1, n_seed, m2, sd2, n_seed, equal_var=False)
            out[label] = {"t": round(float(t), 2), "p": float(f"{p:.2e}")}
        v2["welch"][f"RARM_vs_{who}_{name}"] = out
# 정수 성공 수 일관성: 모든 값이 k/N (소수 한 자리 반올림)이 되는 가장 작은 N (N <= 2000)
vals = [x for m, _ in T2.values() for x in m]
feas = [n for n in range(1, 2001) if all(abs(round(100 * round(x * n / 100) / n, 1) - x) < 1e-9 for x in vals)]
v2["common_N_up_to_2000"] = feas[:10]
single = {}
for x in T2["RARM"][0]:
    single[str(x)] = [n for n in (40, 50, 200, 400) if abs(round(100 * round(x * n / 100) / n, 1) - x) < 1e-9]
v2["rarm_cells_feasible_for_N"] = single  # N=40(10회x4시드), 50(5과제x10회), 200(5과제x10회x4시드), 400(20회x...)
# 시드 하나가 10회 평가의 평균이면 이항 분산만으로 나오는 시드 간 SD 하한
p = 0.956
v2["binomial_sd_per_seed_10eps_pct"] = round(100 * math.sqrt(p * (1 - p) / 10), 1)  # 약 6.5%
v2["reported_pm_rarm_none"] = 1.0
res["V3_table2_recovery"] = v2

# ---------------------------------------------------------------- V4. 그림 5: 실패/성공 보상 비율의 정수 이벤트 일관성
RATIOS = {"RARM": 0.50, "GVL": 0.97, "RoboMeter": 0.88, "RoboDopamine": 0.84, "VIP": 0.82, "RARM_no_gate": 1.89}
v4 = {}
for k, r in RATIOS.items():
    # 성공 이벤트 수 s에서 실패 이벤트 수 f = round(r*s) 가 r을 소수 둘째 자리로 재현하는 최소 s
    s_min = next(s for s in range(1, 400) if abs(round(r * s) / s - r) < 0.005 + 1e-9)
    v4[k] = {"ratio": r, "min_success_events": s_min, "fail_events": round(r * s_min)}
res["V4_fig5_ratios"] = v4

# ---------------------------------------------------------------- V5. 부록 표 4–5 신경망 크기, 부록 F.3 시간 관계
conv = []
size = 84
for i, stride in enumerate([2, 1, 1, 1]):
    size = (size - 3) // stride + 1
    conv.append(size)
v5 = {"conv_out": conv, "flatten": 32 * conv[-1] ** 2}  # 표 5: 41, 39, 37, 35 / 39,200
v5["table6_channels"] = {"MetaWorld": 3 * 3, "LIBERO": 6 * 3}  # 9, 18
tasks = {"fold clothes": (1000, 40, 50), "open drawer": (175, 7, 8.75), "pick eraser": (200, 8, 10), "handover": (300, 12, 15)}
v5["F3_timing"] = {k: {"max_steps": a, "query_steps_from_25": a / 25, "query_steps_table": b,
                       "seconds_at_20Hz": a / 20, "seconds_table": c,
                       "ok": (a / 25 == b) and (a / 20 == c)} for k, (a, b, c) in tasks.items()}
v5["reward_rate_Hz"] = 20 / 4  # 부록 B: 4 환경 스텝마다 → 5 Hz
v5["chunk_seconds_at_50Hz"] = 50 / 50          # F.2: 50스텝 청크, 50 Hz 격자
v5["executed_25_steps_seconds_at_20Hz"] = 25 / 20
v5["slowdown_factor_if_50Hz_actions_played_at_20Hz"] = 50 / 20
res["V5_appendix_consistency"] = v5

# ---------------------------------------------------------------- V6. 표 7 단위 시간과 처리량, 실로봇 DSRL 예산
T7 = {"GVL": (50, 13200), "Robometer": (100, 13200), "RoboDopamine": (300, 19000), "VIP": (10, 4200),
      "RoboCLIP": (10, 5400), "TemporalOT": (10, 6600), "RARM": (10, 5100)}
res["V6_table7"] = {
    "upper_bounds_ms_per_frame": {k: v[0] for k, v in T7.items()},
    "min_fps_implied_by_upper_bound": {k: round(1000 / v[0], 1) for k, v in T7.items()},
    "rarm_mem_vs_gvl": round(5100 / 13200, 3),
    "note": "표 7의 시간은 모두 부등호 상한이라 방법 사이의 속도 순위를 정하지 못한다",
}
res["V6_table7"]["tasks_with_v5_realworld_rollouts"] = {"total_eval_rollouts": 4 * 3 * 3 * N,  # 과제 4 x 예산 3 x (BinR, RoboM, Ours) x 30
                                                         "base_rollouts": 4 * N}


# ---------------------------------------------------------------- V7. 비용 추정 (구조 값은 공식 코드를 정적으로 읽은 것, 부동소수 연산은 우리가 센 것)
def vit_flops(n_tok=576, d=768, layers=12, ff=4):
    macs = layers * (n_tok * (4 * d * d + 2 * ff * d * d) + 2 * n_tok * n_tok * d)
    return 2 * macs  # FLOPs


def pair_flops(n=320, d=768, h=512, layers=2, ff=512):
    ca = 2 * (4 * n * d * d + 2 * n * n * d + n * d * h)
    head = 2 * layers * (4 * n * h * h + 2 * n * n * h + 2 * n * h * ff)
    return ca + head


F_vit, F_pair = vit_flops(), pair_flops()
v7 = {"vit_b_384_per_frame_GFLOP": round(F_vit / 1e9, 1), "pair_compare_GFLOP": round(F_pair / 1e9, 2)}
for M in (70, 140):
    q5 = 5 * F_vit + M * F_pair   # 클립 5프레임을 매번 다시 인코딩 (코드의 step())
    q1 = 1 * F_vit + M * F_pair   # 프레임 인코딩을 캐시한다고 가정
    v7[f"query_M{M}_GFLOP_recode5"] = round(q5 / 1e9)
    v7[f"query_M{M}_GFLOP_cached1"] = round(q1 / 1e9)
    v7[f"sustained_TFLOPs_for_10ms_M{M}_recode5"] = round(q5 / 1e-2 / 1e12, 1)
    v7[f"sustained_TFLOPs_for_10ms_M{M}_cached1"] = round(q1 / 1e-2 / 1e12, 1)
bench = Path(__file__).with_name("bench_mps_results.json")
if bench.exists():
    b = json.loads(bench.read_text(encoding="utf-8"))
    mac_tflops = 5 * F_vit / b["encode_5_frames_s"] / 1e12
    v7["mac_measured_encoder_TFLOPs"] = round(mac_tflops, 1)
    v7["mac_one_query_M70_s"] = b["one_query_M70_s"]
    v7["x_slower_than_10ms_M70"] = round(b["one_query_M70_s"] / 0.010, 1)
    # 학습 한 스텝: 배치 64, 긴 영상 설정(앵커 1 + 양성 5 + 음성 영역 25 + 역방향 5 = 36클립 x 5프레임), 인코더 전/역전파 2–3배
    clips = 1 + 5 + 25 + 5
    for mult in (2, 3):
        flop_step = 64 * clips * 5 * F_vit * mult
        v7[f"train_step_PFLOP_x{mult}"] = round(flop_step / 1e15, 2)
        sec = flop_step / (mac_tflops * 1e12)
        v7[f"train_step_s_on_mac_x{mult}"] = round(sec)
        # 영상 수 1500(코드의 기본 경로 이름 dataset_1500에서 짐작, 미확인), 배치 64 -> 에폭당 23.4 스텝, 45 에폭
        v7[f"train_45epochs_days_on_mac_x{mult}"] = round(sec * (1500 / 64) * 45 / 86400, 1)
res["V7_cost_estimates"] = v7

# ---------------------------------------------------------------- 저장
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

# 요약 출력
print("V1 표1: 칸", v1["n_cells"], "| Ours vs BinR p<0.05:", v1["sig_vs_binr_p05"], "| vs RoboM p<0.05:", v1["sig_vs_robom_p05"],
      "| Bonferroni(24):", v1["sig_vs_binr_bonf"], v1["sig_vs_robom_bonf"])
print("   본문 주장:", v1["claims"])
print("   합산:", v1["pooled_four_tasks"])
print("V2 v1 대 v5: p<0.1 칸", vv["n_p_below_0.1"], "최소 p", vv["min_p"], vv["nested_test"])
print("V3 표2 유지율(20cm/None):", v2["retention_20cm_over_none"], "공통 N:", v2["common_N_up_to_2000"])
print("   RARM 칸이 가능한 N:", v2["rarm_cells_feasible_for_N"], "| 시드당 이항 SD(10회) %:", v2["binomial_sd_per_seed_10eps_pct"])
print("V4 그림 5 최소 이벤트 수:", {k: v["min_success_events"] for k, v in v4.items()})
print("V5", v5["conv_out"], v5["flatten"], {k: x["ok"] for k, x in v5["F3_timing"].items()})
print("V7", v7)
print("저장:", OUT.name)
