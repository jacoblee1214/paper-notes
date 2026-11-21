"""FADA (arXiv:2606.28476 v2) 논문 표 수치 재계산과 검정.

실행: python3 analysis/recompute.py   → analysis/recompute_results.json (수 초, CPU)

가정 (논문에 명시되지 않은 것은 여기서 정한 것이며 03 §0에 적음):
- 표의 "±"는 표준편차(SD)로 읽는다. 시도 수 n은 본문과 표 설명에서 가져온다.
  실로봇 표 1: 5회 (본문 §5, 부록 A). 시뮬 표 2: 20회 (§5.6 "20 randomized-command trials"). 표 5: 5 시드.
  표 8: 20회. 표 9: 1024 병렬 환경. 그림 8: n = 10.
- 성공률은 이항 시도로 보고 Fisher 정확검정을 쓴다.
- 평균 비교는 Welch t 검정(분산이 같다고 가정하지 않음).
"""
import json
import math
from pathlib import Path

from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}


def welch(m1, s1, n1, m2, s2, n2):
    t, p = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
    return {"t": round(float(t), 3), "p": round(float(p), 4)}


def fisher(a, n1, b, n2):
    table = [[a, n1 - a], [b, n2 - b]]
    _, p = stats.fisher_exact(table)
    return {"success_a": f"{a}/{n1}", "success_b": f"{b}/{n2}", "p": round(float(p), 4)}


def pct(x):
    return round(100 * x, 1)


# ---- 표 1: 실로봇 (n = 5) ----
# 성공률 (관측 횟수)
res["T1_success_slope"] = {
    "FADA_vs_FADAzs": fisher(4, 5, 1, 5),
    "FADA_vs_TFDAgger": fisher(4, 5, 0, 5),
}
res["T1_success_basket"] = {
    "FADA_vs_FADAzs": fisher(5, 5, 1, 5),
    "FADA_vs_TFDAgger": fisher(5, 5, 0, 5),
}
# 두 과제 합산 (FADA 9/10, FADA-zs 2/10)
res["T1_success_pooled"] = {"FADA_vs_FADAzs": fisher(9, 10, 2, 10)}
# 평균 성공률 (표 1 본문 주장: 20% -> 90%)
res["T1_mean_success_pct"] = {"FADAzs": pct((20 + 20) / 2 / 100), "FADA": pct((80 + 100) / 2 / 100)}

# 정규화 오차 (TF-DAgger = 1.0). 행: (FADA-zs mean, sd), (FADA mean, sd)
t1_err = {
    "G1_loco_payload_Ev": ((1.289, 0.711), (0.797, 0.018)),
    "G1_kungfu_Empjpe": ((1.18, 0.103), (0.86, 0.084)),
    "T1_loco_payload_Ev": ((1.043, 0.025), (0.866, 0.040)),
}
rel = {}
for k, (zs, ft) in t1_err.items():
    rel[k] = {
        "reduction_vs_zs_pct": pct(1 - ft[0] / zs[0]),
        "reduction_vs_TFDAgger_pct": pct(1 - ft[0] / 1.0),
        "welch_vs_zs": welch(ft[0], ft[1], 5, zs[0], zs[1], 5),
    }
mean_red_zs = sum(v["reduction_vs_zs_pct"] for v in rel.values()) / len(rel)
mean_red_tfd = sum(v["reduction_vs_TFDAgger_pct"] for v in rel.values()) / len(rel)
rel["mean_of_three_reductions_vs_zs_pct"] = round(mean_red_zs, 1)
rel["mean_of_three_reductions_vs_TFDAgger_pct"] = round(mean_red_tfd, 1)
res["T1_normalized_error"] = rel

# ---- 표 2: 시뮬 (MuJoCo 열, n = 20 가정) ----
# (FADA-zs mean, sd), (FADA mean, sd), (TF-DAgger mean, sd)
t2 = {
    "G1_slope": ((0.946, 0.358), (0.800, 0.236), (1.000, 0.439)),
    "G1_kungfu": ((0.961, 0.055), (0.714, 0.048), (1.000, 0.052)),
    "T1_loco_payload": ((1.042, 0.147), (0.885, 0.135), (1.000, 0.143)),
    "T1_slope": ((1.034, 0.334), (0.914, 0.193), (1.000, 0.252)),
    "T1_falcon": ((0.880, 0.094), (0.347, 0.043), (1.000, 0.086)),
}
t2_res = {}
for k, (zs, ft, tf) in t2.items():
    t2_res[k] = {
        "FADA_vs_FADAzs_welch": welch(ft[0], ft[1], 20, zs[0], zs[1], 20),
        "FADA_vs_TFDAgger_welch": welch(ft[0], ft[1], 20, tf[0], tf[1], 20),
        "reduction_vs_zs_pct": pct(1 - ft[0] / zs[0]),
        "reduction_vs_TFDAgger_pct": pct(1 - ft[0] / tf[0]),
    }
avg_zs_ratio = sum(1 - ft[0] / zs[0] for zs, ft, _ in t2.values()) / 5
avg_zs_means = 1 - sum(ft[0] for _, ft, _ in t2.values()) / sum(zs[0] for zs, _, _ in t2.values())
avg_tf = sum(1 - ft[0] / tf[0] for _, ft, tf in t2.values()) / 5
t2_res["avg_over_5_tasks"] = {
    "mean_of_reductions_vs_zs_pct": pct(avg_zs_ratio),
    "reduction_of_means_vs_zs_pct": pct(avg_zs_means),
    "mean_of_reductions_vs_TFDAgger_pct": pct(avg_tf),
    "paper_claims": {"vs_zs": 24.7, "vs_TFDAgger": 26.8},
}
# TF-CoPred-ft 가 TF-CoPred-zs 보다 나빠졌는지 (MuJoCo 열)
copred = {
    "G1_slope": (0.973, 0.387, 1.611, 0.488),
    "G1_kungfu": (1.031, 0.068, 1.421, 0.198),
    "T1_loco_payload": (1.029, 0.142, 1.737, 0.387),
    "T1_slope": (1.080, 0.224, 1.193, 0.246),
    "T1_falcon": (0.943, 0.133, 1.054, 0.162),
}
t2_res["TF_CoPred_ft_vs_zs_welch"] = {
    k: welch(v[2], v[3], 20, v[0], v[1], 20) for k, v in copred.items()
}
res["T2_sim_to_sim"] = t2_res

# ---- 표 3: 페이로드 스윕 (단일 실행, 분산 없음). 유지율 ----
# 주장: FADA@10kg 는 5 kg와 15 kg에서 최적 개선의 약 90%를 유지
imp = {5: (1.20 - 0.87, 1.20 - 0.90), 15: (1.98 - 0.99, 1.98 - 1.09)}
res["T3_FADA_at10_retention"] = {
    str(k): round(100 * v[1] / v[0], 1) for k, v in imp.items()
}

# ---- 표 4: K 절제 (1 -> 6) ----
res["T4_K1_to_K6_reduction_pct"] = {
    "G1_Empjpe": pct(1 - 0.830 / 1.000),
    "T1_Ev": pct(1 - 0.813 / 1.000),
}

# ---- 표 5: LoRA 대 전체 IDM 미세조정 (n = 5 시드) ----
t5 = {
    "G1_slope": {"zs": (1.0, 0.3788), "full": (1.1, 0.4484), "lora": (0.8463, 0.2493)},
    "T1_loco": {"zs": (1.0, 0.1413), "full": (1.1163, 0.3342), "lora": (0.8489, 0.1293)},
    "T1_slope": {"zs": (1.0, 0.3230), "full": (1.2918, 0.2819), "lora": (0.8846, 0.1870)},
}
res["T5_lora_vs_full"] = {
    k: {
        "lora_vs_full_welch": welch(v["lora"][0], v["lora"][1], 5, v["full"][0], v["full"][1], 5),
        "lora_vs_zs_welch": welch(v["lora"][0], v["lora"][1], 5, v["zs"][0], v["zs"][1], 5),
        "full_vs_zs_welch": welch(v["full"][0], v["full"][1], 5, v["zs"][0], v["zs"][1], 5),
    }
    for k, v in t5.items()
}

# ---- 그림 8: 제로샷 성공률 (n = 10) ----
res["F8_zero_shot_success"] = {
    "FADA_vs_wo_A": fisher(10, 10, 7, 10),
    "FADA_vs_wo_B": fisher(10, 10, 4, 10),
    "FADA_vs_wo_both": fisher(10, 10, 1, 10),
    "wo_B_vs_wo_A": fisher(4, 10, 7, 10),
}

# ---- 표 8: 구조 절제 (n = 20 가정). MMM 은 열 정규화 기준 ----
t8 = {
    "MMM": (1.0, 0.1235),
    "MTM": (0.8345, 0.1129),
    "TTM": (0.7664, 0.1210),
    "TTT": (0.7056, 0.1150),
}
res["T8_architecture_ablation_welch"] = {
    "TTT_vs_MMM": welch(t8["TTT"][0], t8["TTT"][1], 20, t8["MMM"][0], t8["MMM"][1], 20),
    "TTT_vs_MTM": welch(t8["TTT"][0], t8["TTT"][1], 20, t8["MTM"][0], t8["MTM"][1], 20),
    "TTT_vs_TTM": welch(t8["TTT"][0], t8["TTT"][1], 20, t8["TTM"][0], t8["TTM"][1], 20),
}

# ---- 표 9: IsaacSim 대 IsaacSim (n = 1024 환경) ----
t9 = {
    "G1_loco_payload": ((1.0501, 0.0234), (0.8514, 0.0282)),
    "T1_slope": ((0.9906, 0.0351), (0.9461, 0.0243)),
    "T1_loco_payload": ((1.0177, 0.0189), (0.9380, 0.0215)),
}
res["T9_same_sim_welch_FADA_vs_zs"] = {
    k: {
        "reduction_pct": pct(1 - v[1][0] / v[0][0]),
        "welch": welch(v[1][0], v[1][1], 1024, v[0][0], v[0][1], 1024),
    }
    for k, v in t9.items()
}

# ---- §5.7 팔 실험 (그림 값을 본문 수치로 대조) ----
res["S5_7_arm"] = {
    "consistency_gap_reduction_pct": pct(1 - 0.037 / 0.079),
    "paper_claims_consistency_gap_pct": 54,
    "planner_rmse_change_m0_to_m5_pct": round(100 * (0.052 / 0.049 - 1), 1),
    "paper_claims_planner_rmse_change_pct": 7,
}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
print(json.dumps(res, ensure_ascii=False, indent=2))
