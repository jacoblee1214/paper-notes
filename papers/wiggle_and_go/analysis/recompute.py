"""Wiggle and Go! (arXiv:2604.22102 v2) 표 재계산과 일관성 검사.

입력: 논문 표 값을 여기에 옮겨 적음 (v2 PDF의 Table 2, 3, 4, 5, 6, 9, v1의 Table II·VI에서 소수 둘째 자리 값).
출력: analysis/recompute_results.json
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"

# ---------- 표 3 (v1 Table VI와 같은 값, 소수 둘째 자리). 행: 로프, 열: 설정 6개 ----------
CONFIGS = ["45cm 5g", "45cm 10g", "45cm 20g", "45cm 30g", "55cm 5g", "65cm 5g"]
ROPES = ["Brown", "Yellow", "Red", "Orange"]  # 학습 분포 안 4개
NN = np.array([[0.14, 2.03, 5.28, 0.82, 0.00, 2.07],
               [8.11, 6.27, 0.78, 4.71, 3.47, 0.00],
               [0.00, 15.96, 0.91, 1.37, 0.33, 4.27],
               [10.08, 2.16, 3.71, 0.00, 11.45, 1.33]])
CMA = np.array([[0.00, 0.61, 8.86, 5.24, 0.20, 4.43],
                [1.47, 3.76, 0.76, 3.25, 7.69, 8.07],
                [0.00, 3.19, 2.34, 7.18, 1.39, 0.00],
                [0.00, 8.65, 1.50, 0.00, 12.04, 19.06]])
RND = np.array([[15.88, 17.34, 18.28, 17.70, 15.47, 22.14],
                [14.65, 15.85, 16.50, 16.97, 14.98, 19.25],
                [17.63, 15.23, 16.95, 16.59, 14.05, 18.17],
                [10.90, 9.99, 11.82, 13.51, 8.00, 9.06]])
CHAIN_NN = [51.37, 29.60, 12.33, 3.31, 51.94, 0.69]  # OOD
CHAIN_CMA = [3.86, 0.63, 2.65, 1.55, 4.46, 6.68]

# 논문이 보고한 값 (대조용)
REPORTED = {"NN_avg_in": 3.55, "CMA_avg_in": 4.16, "RND_avg_in": 15.29,
            "NN_rope_avg": {"Brown": 1.72, "Yellow": 3.89, "Red": 3.81, "Orange": 4.79},
            "ratio_claim": [3.3, 13.1], "rollouts_claim": 600}

# ---------- 표 2: 학습 범위, v2 MAE, v1 상대 오차(%) ----------
PARAMS = [  # 이름, 최소, 최대, 단위, v2 MAE, v1 상대오차(%)
    ("Number of links", 20, 26, "-", 0.098, 0.44),
    ("Rope length", 0.45, 0.65, "m", 0.006, 1.13),
    ("Ball damping", 0.001, 0.05, "N·s/m", 0.010, 33.73),
    ("Ball stiffness", 0.05, 1.0, "N/m", 0.111, 33.82),
    ("Rope radius", 0.003, 0.015, "m", 0.002, 36.37),
    ("Mass per unit length", 0.02, 0.12, "kg/m", 0.007, 62.56),
    ("Lead mass", 0.02, 0.12, "kg", 0.005, 45.20),
    ("Lead radius", 0.015, 0.045, "m", 0.008, 28.81),
    ("Link extra scale", 50, 120, "%", 0.152, 19.85),
]
# 표 1 (테스트 로프): 질량(kg/m), 반지름(mm)과 시험 설정의 리드 질량(kg)
ROPE_MASS = {"Brown": 0.013, "Yellow": 0.011, "Red": 0.033, "Orange": 0.082}
ROPE_RAD_MM = {"Brown": 6, "Yellow": 5, "Red": 9.5, "Orange": 12.7}
LEAD_KG = [0.005, 0.010, 0.020, 0.030, 0.005, 0.005]  # 설정 6개 순서

# ---------- 표 4 (시뮬레이션 100 로프): 평균 ± SD, 중앙값 ----------
T4 = {"Ground truth": (1.2, 1.4, 2.2), "Predicted (Φ-NN)": (2.1, 5.7, 11.5), "Random in-dist.": (12.8, 16.3, 12.1)}

# ---------- 표 6 (lobbing, draping; 로프별 6 설정 평균 %, 거리 cm) ----------
T6 = {"Brown": dict(lob_t=(67, 83), lob_s=(50, 33), dra_s=(67, 83), dra_d=(4.1, 3.8)),
      "Yellow": dict(lob_t=(67, 67), lob_s=(17, 50), dra_s=(67, 67), dra_d=(7.4, 4.1)),
      "Red": dict(lob_t=(67, 83), lob_s=(17, 50), dra_s=(67, 67), dra_d=(3.6, 4.8)),
      "Orange": dict(lob_t=(17, 17), lob_s=(0, 17), dra_s=(50, 50), dra_d=(3.4, 3.5))}

# ---------- 표 9 (시뮬 충실도; 로프별, 리드 10 g/30 g 순서) ----------
T9 = {  # 주파수 상관: NN, CMA, Rnd
    "Brown": dict(f=([0.98, 0.97], [0.94, 0.91], [0.76, 0.85]), d=([4.8, 4.7], [6.9, 8.4], [13.4, 14.7])),
    "Yellow": dict(f=([0.98, 0.97], [0.96, 0.94], [0.80, 0.77]), d=([4.0, 4.9], [5.6, 7.4], [13.7, 15.2])),
    "Red": dict(f=([0.97, 0.97], [0.95, 0.93], [0.81, 0.78]), d=([7.7, 6.3], [5.8, 8.3], [14.2, 15.9])),
    "Orange": dict(f=([0.97, 0.99], [0.89, 0.99], [0.91, 0.90]), d=([4.5, 3.2], [4.3, 2.8], [12.4, 10.7])),
    "Chain*": dict(f=([0.77, 0.98], [0.99, 0.99], [0.76, 0.78]), d=([8.4, 6.0], [3.2, 5.2], [14.2, 12.8])),
}

# ---------- 표 5 (파라미터 중요도, 시뮬, cm) ----------
T5 = {"Full Φ-NN": 0.9, "Random, keep length+lead mass": 10.9, "Random, keep length only": 13.3,
      "Random, keep lead mass only": 12.3, "Fully random": 14.2}

# ---------- 표 8 (wiggle ablation MAE). 행 = 파라미터 9개, 열 = Abl1..8, Main ----------
WIGGLES = ["Abl1 20°0.5Hz", "Abl2 20°0.75Hz", "Abl3 20°1.0Hz", "Abl4 30°0.5Hz", "Abl5 30°0.75Hz",
           "Abl6 30°1.0Hz", "Abl7 random1", "Abl8 random2", "Main joint6"]
T8 = np.array([
    [0.122, 0.127, 0.225, 0.205, 0.206, 0.360, 2.098, 1.453, 0.098],
    [0.007, 0.013, 0.017, 0.015, 0.018, 0.022, 0.057, 0.028, 0.006],
    [0.010, 0.033, 0.038, 0.018, 0.031, 0.096, 0.036, 0.040, 0.010],
    [0.002, 0.003, 0.003, 0.003, 0.004, 0.006, 0.003, 0.004, 0.002],
    [0.006, 0.010, 0.010, 0.013, 0.012, 0.014, 0.019, 0.014, 0.007],
    [0.153, 0.160, 0.160, 0.179, 0.169, 0.241, 0.205, 0.208, 0.152],
    [0.005, 0.011, 0.014, 0.008, 0.011, 0.018, 0.021, 0.016, 0.005],
    [0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.007, 0.008, 0.008],
    [0.118, 0.257, 0.258, 0.224, 0.244, 0.336, 0.388, 0.322, 0.111]])


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def main():
    res = {}
    # 1. 평균과 재계산 (표 3)
    nn_in, cma_in, rnd_in = NN.ravel(), CMA.ravel(), RND.ravel()
    res["table3"] = {
        "NN_avg_in_domain": round(nn_in.mean(), 3), "CMA_avg_in_domain": round(cma_in.mean(), 3),
        "RND_avg_in_domain": round(rnd_in.mean(), 3), "reported": REPORTED["NN_avg_in"],
        "NN_rope_avg_recomputed": {r: round(NN[i].mean(), 2) for i, r in enumerate(ROPES)},
        "CMA_rope_avg_recomputed": {r: round(CMA[i].mean(), 2) for i, r in enumerate(ROPES)},
        "RND_rope_avg_recomputed": {r: round(RND[i].mean(), 2) for i, r in enumerate(ROPES)},
        "headline_ratio_RND_over_NN": round(rnd_in.mean() / nn_in.mean(), 2),
        "headline_ratio_RND_over_CMA": round(rnd_in.mean() / cma_in.mean(), 2),
    }
    # 셀 단위 통계 (24 셀 = 4 로프 x 6 설정, 각 셀은 5회 평균이라고 보고됨)
    sd = nn_in.std(ddof=1)
    tcrit = stats.t.ppf(0.975, len(nn_in) - 1)
    res["table3"]["NN_cell_sd"] = round(sd, 2)
    res["table3"]["NN_mean_t95_CI_cells"] = [round(nn_in.mean() - tcrit * sd / np.sqrt(len(nn_in)), 2),
                                             round(nn_in.mean() + tcrit * sd / np.sqrt(len(nn_in)), 2)]
    w = stats.wilcoxon(nn_in, cma_in)
    t = stats.ttest_rel(nn_in, cma_in)
    res["table3"]["NN_vs_CMA"] = {
        "NN_lower_cells": int((nn_in < cma_in).sum()), "CMA_lower_cells": int((cma_in < nn_in).sum()),
        "ties": int((nn_in == cma_in).sum()),
        "wilcoxon_p": round(float(w.pvalue), 4), "paired_t_p": round(float(t.pvalue), 4),
        "mean_diff_CMA_minus_NN": round(float(cma_in.mean() - nn_in.mean()), 2),
    }
    # 설정별 NN vs CMA (어느 설정에서 이기는지)
    res["table3"]["NN_vs_CMA_by_config"] = {
        c: {"NN": round(NN[:, j].mean(), 2), "CMA": round(CMA[:, j].mean(), 2), "RND": round(RND[:, j].mean(), 2)}
        for j, c in enumerate(CONFIGS)}
    # 리드 질량이 학습 범위(20 g) 안인 설정과 밖인 설정을 나눠 비교
    in_lead = [j for j, m in enumerate(LEAD_KG) if m >= 0.02]
    out_lead = [j for j, m in enumerate(LEAD_KG) if m < 0.02]
    res["table3"]["NN_vs_CMA_in_lead_range_cols"] = {
        "cols": [CONFIGS[j] for j in in_lead],
        "NN": round(NN[:, in_lead].mean(), 2), "CMA": round(CMA[:, in_lead].mean(), 2)}
    res["table3"]["NN_vs_CMA_out_lead_range_cols"] = {
        "cols": [CONFIGS[j] for j in out_lead],
        "NN": round(NN[:, out_lead].mean(), 2), "CMA": round(CMA[:, out_lead].mean(), 2)}
    # 배율 주장 (Rnd / NN)
    per_cfg = [RND[:, j].mean() / NN[:, j].mean() for j in range(6)]
    per_rope = [RND[i].mean() / NN[i].mean() for i in range(4)]
    res["table3"]["ratio_RND_over_NN_per_config"] = [round(x, 1) for x in per_cfg]
    res["table3"]["ratio_RND_over_NN_per_rope"] = [round(x, 1) for x in per_rope]
    res["table3"]["ratio_claim_check"] = "논문 3.3–13.1배: 설정별 %.1f–%.1f, 로프별 %.1f–%.1f" % (
        min(per_cfg), max(per_cfg), min(per_rope), max(per_rope))
    # 로프 7.5 cm 범위 주장 (같은 길이·리드, 다른 로프의 Φ-Random 범위)
    res["table3"]["RND_range_same_config_max"] = round(max(RND[:, j].max() - RND[:, j].min() for j in range(6)), 2)
    res["table3"]["RND_range_same_config_argmax"] = CONFIGS[int(np.argmax([RND[:, j].max() - RND[:, j].min() for j in range(6)]))]
    cfg_ranges = [float(RND[:, j].max() - RND[:, j].min()) for j in range(6)]
    res["table3"]["RND_range_across_4_ropes_per_config"] = {c: round(x, 2) for c, x in zip(CONFIGS, cfg_ranges)}
    # 리드 질량이 범위 안인 8개 셀(20 g, 30 g)만으로 NN 대 CMA
    w8 = stats.wilcoxon(NN[:, in_lead].ravel(), CMA[:, in_lead].ravel())
    res["table3"]["NN_vs_CMA_in_lead_range_cells_wilcoxon_p"] = round(float(w8.pvalue), 4)
    # 롤아웃 수 주장
    cells_in = 4 * 6 * 3
    cells_chain = 1 * 6 * 2
    res["table3"]["rollout_count"] = {"cells": cells_in + cells_chain, "per_cell": 5,
                                      "implied_rollouts": (cells_in + cells_chain) * 5,
                                      "claimed": REPORTED["rollouts_claim"]}

    # 2. 표 2: 학습 범위 대비 오차, 테스트 로프가 범위 밖인지
    rows = []
    for name, lo, hi, unit, mae, rel in PARAMS:
        rows.append({"param": name, "min": lo, "max": hi, "unit": unit, "MAE_v2": mae,
                     "rel_err_v1_pct": rel, "MAE_over_range_pct": round(100 * mae / (hi - lo), 1)})
    res["table2"] = rows
    res["table2_note"] = ("v2는 상대 오차 열을 뺐고, v1의 상대 오차는 질량 단위 길이 62.6%, 리드 질량 45.2%, "
                          "반지름 36.4%, 감쇠 33.7%, 강성 33.8%로 가장 컸다. 'sub-millimeter MAE'는 v2 본문에서 "
                          "길이 6 mm, 반지름 2 mm 값과 함께 쓰여 단위 표현이 맞지 않는다.")
    res["range_check"] = {
        "rope_mass_below_train_min": {r: ROPE_MASS[r] for r in ROPES if ROPE_MASS[r] < 0.02},
        "rope_mass_train_min": 0.02,
        "lead_mass_below_train_min_configs": [CONFIGS[j] for j in out_lead],
        "lead_mass_train_min": 0.02,
        "clamped_bias_lower_bound_kg_per_m": {r: round(0.02 - ROPE_MASS[r], 3) for r in ROPES if ROPE_MASS[r] < 0.02},
        "relative_bias_lower_bound_pct": {r: round(100 * (0.02 - ROPE_MASS[r]) / ROPE_MASS[r], 0) for r in ROPES if ROPE_MASS[r] < 0.02},
        "radius_in_range": {r: (0.003 <= ROPE_RAD_MM[r] / 1000 <= 0.015) for r in ROPES},
    }

    # 3. 표 4 (시뮬 100개): 표준편차 ±를 SD로 가정하고 평균의 SE, 95% 구간
    t4 = {}
    for k, (med, mean, sdv) in T4.items():
        se = sdv / np.sqrt(100)
        t4[k] = {"median": med, "mean": mean, "SD_assumed": sdv, "SE": round(se, 2),
                 "CI95_mean": [round(mean - 1.984 * se, 2), round(mean + 1.984 * se, 2)]}
    t4["predicted_minus_GT_median"] = round(2.1 - 1.2, 1)
    t4["random_over_predicted_median_ratio"] = round(12.8 / 2.1, 1)
    res["table4"] = t4

    # 4. 표 5 차이 계산
    base = T5["Full Φ-NN"]
    res["table5"] = {k: {"mean": v, "gap_closed_by_NN_pct": round(100 * (T5["Fully random"] - v) / (T5["Fully random"] - base), 1)}
                     for k, v in T5.items()}
    res["table5"]["text_claim_check"] = {
        "length_and_leadmass_closes_cm": round(T5["Fully random"] - T5["Random, keep length+lead mass"], 1),
        "remaining_gap_driven_by_7_params_cm": round(T5["Random, keep length+lead mass"] - base, 1)}

    # 5. 표 6 (lobbing/draping): 24회(4로프 x 6설정) 가정, 백분율 x 6 = 성공 수
    cnt = lambda pcts: [int(round(p * 6 / 100)) for p in pcts]
    lob_t_nn = sum(cnt([T6[r]["lob_t"][0] for r in ROPES])); lob_t_cma = sum(cnt([T6[r]["lob_t"][1] for r in ROPES]))
    lob_s_nn = sum(cnt([T6[r]["lob_s"][0] for r in ROPES])); lob_s_cma = sum(cnt([T6[r]["lob_s"][1] for r in ROPES]))
    dra_nn = sum(cnt([T6[r]["dra_s"][0] for r in ROPES])); dra_cma = sum(cnt([T6[r]["dra_s"][1] for r in ROPES]))
    lob = {"target_NN": (lob_t_nn, 24), "target_CMA": (lob_t_cma, 24), "stay_NN": (lob_s_nn, 24),
           "stay_CMA": (lob_s_cma, 24), "draping_NN": (dra_nn, 24), "draping_CMA": (dra_cma, 24)}
    t6 = {}
    for k, (x, n) in lob.items():
        lo, hi = wilson(x, n)
        t6[k] = {"k": x, "n": n, "pct": round(100 * x / n, 1), "wilson95": [round(100 * lo, 1), round(100 * hi, 1)]}
    t6["fisher_p_target_NN_vs_CMA"] = round(float(stats.fisher_exact([[lob_t_nn, 24 - lob_t_nn], [lob_t_cma, 24 - lob_t_cma]])[1]), 3)
    t6["fisher_p_stay_NN_vs_CMA"] = round(float(stats.fisher_exact([[lob_s_nn, 24 - lob_s_nn], [lob_s_cma, 24 - lob_s_cma]])[1]), 3)
    t6["fisher_p_draping_NN_vs_CMA"] = round(float(stats.fisher_exact([[dra_nn, 24 - dra_nn], [dra_cma, 24 - dra_cma]])[1]), 3)
    t6["full_success_NN_note"] = "전체 성공(목표 도달과 체류의 교집합)은 표에 없어 계산하지 못함. 초록의 50%는 목표 도달 기준으로만 맞다."
    t6["draping_orange_exactly_50pct"] = True
    res["table6"] = t6

    # 6. 표 9 (시뮬 충실도): 평균 재계산과 "10 in-distribution pairs" 점검
    f_nn = [x for r in T9 for x in T9[r]["f"][0]]
    f_cma = [x for r in T9 for x in T9[r]["f"][1]]
    f_rnd = [x for r in T9 for x in T9[r]["f"][2]]
    d_nn = [x for r in T9 for x in T9[r]["d"][0]]
    d_cma = [x for r in T9 for x in T9[r]["d"][1]]
    d_rnd = [x for r in T9 for x in T9[r]["d"][2]]
    f_nn_in = [x for r in ROPES for x in T9[r]["f"][0]]
    res["table9"] = {
        "freq_mean_all10": {"NN": round(np.mean(f_nn), 3), "CMA": round(np.mean(f_cma), 3), "RND": round(np.mean(f_rnd), 3)},
        "freq_mean_in_domain8": {"NN": round(np.mean(f_nn_in), 3),
                                 "CMA": round(np.mean([x for r in ROPES for x in T9[r]["f"][1]]), 3),
                                 "RND": round(np.mean([x for r in ROPES for x in T9[r]["f"][2]]), 3)},
        "dist_mean_all10": {"NN": round(np.mean(d_nn), 2), "CMA": round(np.mean(d_cma), 2), "RND": round(np.mean(d_rnd), 2)},
        "n_rows": len(T9), "n_values_per_method": len(f_nn),
        "chain_NN_freq_mean_two_leads": round(np.mean(T9["Chain*"]["f"][0]), 3),
        "text_claim": "본문은 '10 in-distribution pairs'라고 쓰지만 표의 10개 행에는 chain(OOD) 2개가 들어 있다. 평균 0.95는 chain 포함 값.",
    }

    # 7. 시뮬 Table 4 Φ-NN 평균 0.9 vs 실기 3.55 (단위 cm) -> 비율
    res["sim_vs_real"] = {"sim_full_pipeline_mean_cm_table5": 0.9, "real_NN_avg_cm": 3.55,
                          "note": "시뮬 0.9는 표 5의 30 설정 시뮬, 실기는 표 3. 두 집합이 같은지 논문이 밝히지 않음."}

    # 8. 표 8: 파라미터별 Main과 나머지 비교
    main_idx = 8
    res["table8"] = {
        "mean_abs_err_over_9_params": {w_: round(float(T8[:, i].mean()), 3) for i, w_ in enumerate(WIGGLES)},
        "main_is_best_per_param": [PARAMS[r][0] for r in range(9) if T8[r, main_idx] <= T8[r].min() + 1e-12],
        "main_worse_than": {w_: [PARAMS[r][0] for r in range(9) if T8[r, main_idx] > T8[r, i] + 1e-12] for i, w_ in enumerate(WIGGLES[:-1])},
        "ratio_random1_over_main_links": round(T8[0, 6] / T8[0, 8], 1),
        "ratio_abl2_over_main_stiffness": round(T8[8, 1] / T8[8, 8], 2),
    }
    res["table8"]["note"] = ("'wiggle는 중요하지 않다'는 주장은 평면 wiggle 사이에서 맞고, 무작위 궤적(Abl7–8)에서는 "
                             "링크 수 MAE가 주 궤적의 약 15–21배다.")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ["table3", "table8", "table9"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
