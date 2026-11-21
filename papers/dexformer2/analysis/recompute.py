"""논문 표 수치를 다시 계산하고 검정한다 (arXiv:2611.20273 v2, 10쪽 본문 표 1–6).

실행: python3 analysis/recompute.py  → analysis/recompute_results.json (수 초, numpy·scipy만 사용)

입력은 논문 표에 적힌 평균과 표준편차뿐이다(에피소드별 원자료는 공개되지 않았다).
- 평균 ± 표준편차는 시드 5개의 표본 표준편차로 보고 있다고 논문이 적었다(표 1 캡션).
- 95% 구간: 평균 ± t(0.975, 4) · SD/√5.
- 두 조건 비교: Welch t 검정(SD 기준, n = 5 대 5).
- 실로봇(표 6): Wilson 구간, Fisher 정확 검정.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
N_SEEDS = 5
T975 = stats.t.ppf(0.975, N_SEEDS - 1)
EPS_PER_EMB = 1000  # 표 1 프로토콜: 임베디먼트·조건당 1,000 에피소드

# ---- 표 1: 객체 × 조합 분할 (mean, sd) ----
T1 = {
    "GRU": {"seen_obj_seen": (64.82, 0.41), "seen_obj_held": (31.28, 0.94),
            "unseen_obj_seen": (65.61, 0.93), "unseen_obj_held": (33.18, 1.89)},
    "DexFormer1.0": {"seen_obj_seen": (66.02, 0.91), "seen_obj_held": (14.15, 0.58),
                     "unseen_obj_seen": (70.65, 0.49), "unseen_obj_held": (11.53, 0.60)},
    "Ours": {"seen_obj_seen": (89.01, 0.56), "seen_obj_held": (51.18, 1.72),
             "unseen_obj_seen": (89.01, 0.31), "unseen_obj_held": (48.85, 1.94)},
}
# ---- 표 2: 구조 절제 (seen 18, held-out 6), 객체는 학습 프리미티브 ----
T2 = {
    "Full (Ours)": ((89.01, 0.56), (51.18, 1.72)),
    "w/o Memory": ((79.14, 1.07), (24.45, 0.41)),
    "w/o Tree Bias": ((78.43, 0.39), (24.15, 1.31)),
    "MLP + TXL": ((73.69, 0.54), (18.90, 0.72)),
}
# ---- 표 3: 학습 조합 수 6/12/18 (common held-out = 고정된 6쌍, all = 24쌍) ----
T3 = {
    6: {"n_seen": 6, "n_held": 18, "seen": (56.58, 0.58), "held": (0.08, 0.05),
        "common": (0.07, 0.07), "all": (14.21, 0.18)},
    12: {"n_seen": 12, "n_held": 12, "seen": (61.07, 0.31), "held": (3.37, 0.22),
         "common": (5.65, 0.41), "all": (32.22, 0.24)},
    18: {"n_seen": 18, "n_held": 6, "seen": (89.01, 0.56), "held": (51.18, 1.72),
         "common": (51.18, 1.72), "all": (79.55, 0.73)},
}
# ---- 표 4: 미학습 손(열) × 학습 팔(행), 평균 ± SD (각 1,000 에피소드, 5 시드) ----
HANDS = ["Revo2", "Wuji 1", "Tesollo", "XHAND1"]
GRIPPERS = ["Franka Hand", "Robotiq 2F-85", "xArm G1", "UMI WSG-50"]
ARMS = ["Flexiv", "Franka", "KUKA", "xArm7"]
T4_MEAN = np.array([
    [26.60, 41.20, 12.70, 26.50, 14.70, 7.00, 5.90, 3.70],
    [4.60, 10.00, 7.00, 6.10, 18.70, 16.40, 20.20, 15.10],
    [14.40, 24.10, 6.70, 6.80, 14.30, 8.10, 6.10, 3.60],
    [10.70, 33.40, 20.70, 27.70, 18.80, 15.00, 13.10, 7.80],
])
T4_SD = np.array([
    [2.22, 2.80, 3.88, 4.47, 0.91, 0.61, 0.82, 1.35],
    [1.08, 2.03, 1.27, 1.29, 1.30, 3.23, 1.78, 1.78],
    [2.22, 1.24, 1.35, 1.44, 1.72, 1.19, 0.74, 1.56],
    [1.04, 2.48, 2.08, 2.84, 3.21, 1.41, 1.85, 2.11],
])
T4_COL_MEAN_REPORTED = [14.07, 27.18, 11.78, 16.78, 16.62, 11.62, 11.32, 7.55]
T4_COL_SD_REPORTED = [0.91, 1.32, 1.44, 1.87, 0.83, 1.00, 0.58, 0.90]
T4_HAND_MEAN_OVERALL_REPORTED = 17.45
# ---- 표 5: 손 관절 고정 결함 (표준편차 없음) ----
T5 = {
    "GRU": {"seen_normal": 64.82, "seen_initial": 59.51, "seen_mid": 62.95,
            "held_normal": 31.28, "held_initial": 28.53, "held_mid": 30.77},
    "DexFormer1.0": {"seen_normal": 66.02, "seen_initial": 56.61, "seen_mid": 62.43,
                     "held_normal": 14.15, "held_initial": 11.70, "held_mid": 12.25},
    "Ours": {"seen_normal": 89.01, "seen_initial": 81.74, "seen_mid": 86.40,
             "held_normal": 51.18, "held_initial": 45.53, "held_mid": 49.63},
}
# ---- 표 6: 실로봇 성공 (5회 × 5 물체) ----
T6 = {
    "Flexiv+Revo3 (held-out)": [4, 2, 5, 3, 3],
    "Flexiv+LEAP (training)": [5, 4, 5, 5, 4],
}


def ci95(mean, sd):
    se = sd / math.sqrt(N_SEEDS)
    return (mean - T975 * se, mean + T975 * se)


def welch(m1, s1, m2, s2):
    t, p = stats.ttest_ind_from_stats(m1, s1, N_SEEDS, m2, s2, N_SEEDS, equal_var=False)
    return float(t), float(p)


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * (c - h), 100 * (c + h))


def main():
    res = {"n_seeds": N_SEEDS, "t975_df4": round(float(T975), 4)}

    # 1) 표 1: 95% 구간과 Ours 대 기준선 Welch 검정
    t1 = {}
    for pol, cells in T1.items():
        t1[pol] = {k: {"mean": m, "sd": s, "ci95": [round(x, 2) for x in ci95(m, s)]}
                   for k, (m, s) in cells.items()}
    res["t1_ci"] = t1
    t1_tests = {}
    for base in ["GRU", "DexFormer1.0"]:
        for key in ["seen_obj_seen", "seen_obj_held", "unseen_obj_seen", "unseen_obj_held"]:
            m1, s1 = T1["Ours"][key]
            m2, s2 = T1[base][key]
            t, p = welch(m1, s1, m2, s2)
            t1_tests[f"Ours_vs_{base}_{key}"] = {"diff_pp": round(m1 - m2, 2), "t": round(t, 2), "p": p}
    # 객체 효과: 같은 조합에서 seen 대 unseen 객체 (Ours held-out)
    m1, s1 = T1["Ours"]["seen_obj_held"]
    m2, s2 = T1["Ours"]["unseen_obj_held"]
    t, p = welch(m1, s1, m2, s2)
    t1_tests["Ours_held_seenobj_vs_unseenobj"] = {"diff_pp": round(m1 - m2, 2), "t": round(t, 2), "p": p}
    m1, s1 = T1["Ours"]["unseen_obj_seen"]
    m2, s2 = T1["Ours"]["seen_obj_seen"]
    t1_tests["Ours_seen_unseenobj_vs_seenobj_diff"] = {"diff_pp": round(m1 - m2, 2)}
    res["t1_tests"] = t1_tests

    # 2) 표 2: 절제 대 Full, held-out과 seen 모두
    t2 = {}
    (f_seen, f_held) = T2["Full (Ours)"]
    for name, (s_, h_) in T2.items():
        if name == "Full (Ours)":
            continue
        ts, ps = welch(f_seen[0], f_seen[1], s_[0], s_[1])
        th, ph = welch(f_held[0], f_held[1], h_[0], h_[1])
        t2[name] = {"held_drop_pp": round(f_held[0] - h_[0], 2),
                    "held_drop_ratio": round(h_[0] / f_held[0], 3),
                    "held_t": round(th, 2), "held_p": ph,
                    "seen_drop_pp": round(f_seen[0] - s_[0], 2),
                    "seen_t": round(ts, 2), "seen_p": ps}
    res["t2_ablation"] = t2

    # 3) 표 3: 학습 조합 수 효과와 All(24) 가중 평균 검사
    t3_check = {}
    for n, row in T3.items():
        w = (row["n_seen"] * row["seen"][0] + row["n_held"] * row["held"][0]) / 24
        t3_check[str(n)] = {"all24_recomputed": round(w, 2), "all24_reported": row["all"][0],
                            "consistent_within_0.01": abs(round(w, 2) - row["all"][0]) <= 0.011}
    res["t3_all24_check"] = t3_check
    t3_tests = {}
    for a, b in [(6, 12), (12, 18)]:
        for key in ["common", "all"]:
            t, p = welch(T3[b][key][0], T3[b][key][1], T3[a][key][0], T3[a][key][1])
            t3_tests[f"{a}_to_{b}_{key}"] = {"diff_pp": round(T3[b][key][0] - T3[a][key][0], 2),
                                              "t": round(t, 2), "p": p}
    res["t3_tests"] = t3_tests

    # 4) 표 4: 행·열 평균과 전체 평균을 다시 계산
    row_mean = T4_MEAN.mean(axis=0)
    hand_mean = T4_MEAN[:, :4].mean(axis=0)
    grip_mean = T4_MEAN[:, 4:].mean(axis=0)
    res["t4_col_mean_recomputed"] = [round(float(x), 2) for x in row_mean]
    res["t4_col_mean_reported"] = T4_COL_MEAN_REPORTED
    res["t4_hand_overall_recomputed"] = round(float(T4_MEAN[:, :4].mean()), 2)
    res["t4_hand_overall_reported"] = T4_HAND_MEAN_OVERALL_REPORTED
    res["t4_gripper_overall_recomputed"] = round(float(T4_MEAN[:, 4:].mean()), 2)
    # 보고된 열 SD가 무엇인지 확인: 네 팔에 걸친 SD 대 시드 SD 근사(독립 가정)
    sd_indep = np.sqrt((T4_SD ** 2).sum(axis=0)) / 4
    res["t4_col_sd_reported"] = T4_COL_SD_REPORTED
    res["t4_col_sd_if_independent_seeds"] = [round(float(x), 2) for x in sd_indep]
    res["t4_col_sd_across_arms"] = [round(float(x), 2) for x in T4_MEAN.std(axis=0, ddof=1)]
    # Wuji 1이 네 팔 모두에서 손 중 최고인지
    res["t4_wuji_best_on_all_arms"] = bool(all(T4_MEAN[i, 1] == max(T4_MEAN[i, :4]) for i in range(4)))
    # 손 중 최고(Wuji) 대 XHAND1 열 차이 검정 (팔 네 개를 짝으로 본 대응 t)
    d = T4_MEAN[:, 1] - T4_MEAN[:, 3]
    res["t4_wuji_minus_xhand_paired"] = {"mean_diff": round(float(d.mean()), 2),
                                         "p_paired_4_arms": float(stats.ttest_rel(T4_MEAN[:, 1], T4_MEAN[:, 3]).pvalue)}

    # 5) 표 5: 결함 하 낙폭
    t5 = {}
    for pol, v in T5.items():
        t5[pol] = {"seen_initial_drop": round(v["seen_normal"] - v["seen_initial"], 2),
                   "seen_mid_drop": round(v["seen_normal"] - v["seen_mid"], 2),
                   "held_initial_drop": round(v["held_normal"] - v["held_initial"], 2),
                   "held_mid_drop": round(v["held_normal"] - v["held_mid"], 2)}
    res["t5_drops"] = t5
    res["t5_ours_above_baselines_all_conditions"] = all(
        T5["Ours"][k] > max(T5["GRU"][k], T5["DexFormer1.0"][k]) for k in T5["Ours"])

    # 6) 표 6: 실로봇 집계, Wilson 구간, Fisher 검정
    t6 = {}
    for name, obj in T6.items():
        k = sum(obj)
        t6[name] = {"successes": k, "trials": 5 * len(obj), "rate": round(100 * k / (5 * len(obj)), 2),
                    "wilson95": [round(x, 2) for x in wilson(k, 5 * len(obj))],
                    "per_object": obj}
    res["t6_real"] = t6
    a = t6["Flexiv+Revo3 (held-out)"]
    b = t6["Flexiv+LEAP (training)"]
    oddsr, pf = stats.fisher_exact([[a["successes"], a["trials"] - a["successes"]],
                                    [b["successes"], b["trials"] - b["successes"]]])
    res["t6_fisher_revo3_vs_leap"] = {"p_two_sided": float(pf), "odds_ratio": float(oddsr)}
    # v1에만 있던 시뮬 값 37.90%와의 이항 검정 (v1 전용 주장, 시뮬 시도 수는 미상이라 25회 기준 참고값)
    res["t6_v1_sim_37.9_binom_p_if_n25"] = float(stats.binomtest(17, 25, 0.379).pvalue)

    # 7) 검정력: 시드 5개, Welch, α = 0.05 양측, 검정력 0.8에서 최소 검출 차이
    power = {}
    for label, sd in [("held_out_SD1.72", 1.72), ("seen_obj_SD0.56", 0.56)]:
        df = 2 * N_SEEDS - 2
        t_a = stats.t.ppf(0.975, df)
        t_b = stats.t.ppf(0.8, df)
        mdd = (t_a + t_b) * math.sqrt(2 * sd * sd / N_SEEDS)
        power[label] = round(float(mdd), 2)
    res["min_detectable_diff_pp_80pct_power"] = power

    # 8) 이항 샘플링 잡음: 임베디먼트당 1,000 에피소드, 6쌍 평균
    p_h = 0.5118
    se_binom_split = math.sqrt(6 * p_h * (1 - p_h) / EPS_PER_EMB) / 6
    res["binomial_se_heldout_split_pp"] = round(100 * se_binom_split, 2)
    res["seed_sd_heldout_pp"] = 1.72
    res["ratio_seed_sd_to_binomial_se"] = round(1.72 / (100 * se_binom_split), 2)

    # 9) 표 1의 "Ours" 두 열이 같은 평균(89.01)인지: 우연 가능성을 정규근사로 대략 계산
    diff_sd = math.sqrt(0.56 ** 2 + 0.31 ** 2)
    p_round_equal = 0.01 / (diff_sd * math.sqrt(2 * math.pi)) * 2
    res["same_mean_89.01_rough_chance_of_match_to_0.01"] = round(float(min(1, p_round_equal)), 4)

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT)
    print(json.dumps({k: res[k] for k in ["t3_all24_check", "t4_col_mean_recomputed", "t6_real",
                                           "t6_fisher_revo3_vs_leap", "min_detectable_diff_pp_80pct_power",
                                           "binomial_se_heldout_split_pp", "ratio_seed_sd_to_binomial_se"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
