"""DexMimicGen (arXiv:2410.24185 v2) 보고 수치 재계산과 일치 검사.

논문 본문의 표와 문장 수치를 옮겨 적고(paper/paper.txt 와 대조함), 다음을 계산한다.
  K: 표 사이, 표와 본문, 버전 사이의 일치 검사
  C: 본문 주장의 수치 대조
  S: 3 시드 요약 통계로 한 검정 (± 를 SD로 볼 때와 SEM으로 볼 때 둘 다)
  B: 이항 표준오차 참고값 (롤아웃 수는 논문에 없으므로 N=50을 가정한 참고값)
  W: 실로봇 20회 시도의 Wilson 구간과 Fisher 검정 (10회씩 두 컵, 논문 VI-D 가정)

실행: python3 analysis/recompute.py   → analysis/recompute_results.json
numpy, scipy 만 쓴다. 1초 안에 끝난다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
N_ROLLOUT_ASSUMED = 50  # 논문에 롤아웃 수가 없어 MimicGen 관행(50)을 가정한 참고값
T975_DF2 = stats.t.ppf(0.975, df=2)  # 3 시드, df = 2

TASKS = ["Piece Assembly", "Threading", "Transport", "Box Cleanup", "Drawer Cleanup",
         "Tray Lift", "Pouring", "Coffee", "Can Sorting"]
DEXTEROUS = {"Box Cleanup", "Drawer Cleanup", "Tray Lift", "Pouring", "Coffee", "Can Sorting"}

# Table I (v2). 열: 원천 시연, BC-RNN-GMM, BC-RNN, DP. 값은 (평균, ±값)
T1_SRC = [(3.3, .9), (1.3, .9), (52.7, 7.7), (62.0, 1.6), (0.7, .9), (3.3, .9), (0.7, .9), (14.7, .9), (0.7, .9)]
T1_GMM = [(74.0, 2.8), (54.0, 4.3), (64.0, 3.3), (64.0, 10.0), (30.7, 5.0), (66.0, 8.2), (74.0, 8.6), (12.0, 1.6), (75.3, 1.9)]
T1_BCRNN_V2 = [(66.0, 2.0), (68.0, 1.6), (57.3, 4.1), (94.7, .9), (80.0, 0.0), (78.0, 2.0), (62.0, 7.5), (84.7, 4.9), (96.0, 4.3)]
T1_DP = [(80.7, .9), (69.3, 1.9), (83.3, .9), (92.0, 4.3), (76.0, 0.0), (88.7, .9), (79.3, .9), (77.3, .9), (97.3, .9)]
# v1 Table I 의 BC-RNN 열 (arXiv v1, 2024-10-31)
T1_BCRNN_V1 = [(74.7, 2.5), (55.3, 5.0), (60.0, 0.0), (96.7, 2.5), (56.7, 4.1), (75.3, 7.5), (86.0, 3.3), (73.3, 12.3), (96.0, 1.6)]

# Table II (D0, D1, D2): (task, policy, [D0, D1, D2])
T2 = [("Piece Assembly", "BC-RNN-GMM", [(74.0, 2.8), (67.3, .9), (44.0, 3.3)]),
      ("Box Cleanup", "BC-RNN", [(96.7, 2.5), (88.0, 5.9), (78.7, 2.5)]),
      ("Pouring", "DP", [(79.3, .9), (82.0, 2.0), (71.3, 2.5)])]

# Table III: (task, policy, DemoNoise, DexMimicGen)
T3 = [("Piece Assembly", "BC-RNN-GMM", (12.7, 3.4), (74.0, 2.8)),
      ("Tray Lift", "BC-RNN", (16.7, 2.5), (75.3, 7.5)),
      ("Pouring", "DP", (26.7, 2.5), (79.3, .9))]

# 본문 문장 수치 (v2)
TEXT = {
    "drawer_bcrnn_ordering": 50.7, "drawer_bcrnn_no_ordering": 48.0,
    "pouring_dp_ordering": 88.7, "pouring_dp_no_ordering": 76.7,
    "drawer_same_source": 56.7, "pouring_same_source": 79.3,
    "transport_replay_gmm": 63.3, "transport_transform_gmm": 46.0,
    "cansort_dp_replay": 97.3, "cansort_dp_transform": 98.6,
    "bigym_gen": {"FlipCup": 29.1, "DishwasherLoadPlates": 43.6, "CupBoardsCloseAll": 76.4},
    "realworld_dexmg": 90.0, "realworld_source": 0.0, "realworld_trials_per_cup": 10,
}
COUNTS = {"source_demos_parallel_tasks": 3, "source_per_parallel": 10,
          "source_demos_dex_tasks": 6, "source_per_dex": 5, "stated_total_source": 60,
          "dexmg_per_task_default": 1000, "default_tasks": 9, "variant_tasks": 3, "variant_per": 2,
          "bigym_tasks": 3, "stated_total_demos": 21000}


def tstat_welch(m1, s1, n1, m2, s2, n2):
    res = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
    return float(res.statistic), float(res.pvalue)


def ci_half_sd(sd):
    # 3 시드의 평균 95% 구간 반폭 (± 를 SD로 볼 때)
    return T975_DF2 * sd / math.sqrt(3)


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def main():
    res = {}

    # K1: 표 I 과 표 II/III 의 같은 셀 비교 (같은 과제, 같은 정책)
    k1 = []
    idx = {t: i for i, t in enumerate(TASKS)}
    col = {"BC-RNN-GMM": T1_GMM, "BC-RNN": T1_BCRNN_V2, "DP": T1_DP}
    for task, pol, d in T2:
        t1 = col[pol][idx[task]]
        k1.append({"table": "II D0", "task": task, "policy": pol, "table2": d[0], "table1_v2": t1,
                   "match": abs(d[0][0] - t1[0]) < 1e-9 and abs(d[0][1] - t1[1]) < 1e-9})
    for task, pol, _dn, dx in T3:
        t1 = col[pol][idx[task]]
        k1.append({"table": "III DexMG", "task": task, "policy": pol, "table3": dx, "table1_v2": t1,
                   "match": abs(dx[0] - t1[0]) < 1e-9 and abs(dx[1] - t1[1]) < 1e-9})
    res["K1_table_cell_matches_v2"] = k1

    # K2: v1 과 v2 의 BC-RNN 열 차이 (v1 은 표 II/III 값과 맞는 쪽)
    k2 = []
    for i, t in enumerate(TASKS):
        k2.append({"task": t, "v1": T1_BCRNN_V1[i], "v2": T1_BCRNN_V2[i],
                   "diff_pp": round(T1_BCRNN_V2[i][0] - T1_BCRNN_V1[i][0], 1),
                   "changed": T1_BCRNN_V1[i] != T1_BCRNN_V2[i]})
    res["K2_bcrnn_column_v1_vs_v2"] = k2
    res["K2_summary"] = {"changed_cells": sum(r["changed"] for r in k2), "of": len(k2),
                         "mean_abs_change_pp": round(float(np.mean([abs(r["diff_pp"]) for r in k2])), 1)}

    # K3: 본문과 표 비교
    k3 = {
        "drawer_same_source_text_vs_table_bcrnn_v2": {"text": TEXT["drawer_same_source"],
                                                       "table_v2": T1_BCRNN_V2[4][0],
                                                       "table_v1": T1_BCRNN_V1[4][0]},
        "pouring_same_source_text_vs_table_dp": {"text": TEXT["pouring_same_source"], "table_dp": T1_DP[6][0]},
        "transport_replay_text_vs_table_gmm_default": {"text": TEXT["transport_replay_gmm"],
                                                       "table_gmm": T1_GMM[2][0]},
        "cansort_replay_text_vs_table_dp": {"text": TEXT["cansort_dp_replay"], "table_dp": T1_DP[8][0]},
        "pouring_ordering_text_vs_table_dp": {"text": TEXT["pouring_dp_ordering"], "table_dp": T1_DP[6][0]},
    }
    res["K3_text_vs_table"] = k3

    # K4: 데이터 수 합산
    c = COUNTS
    src_total = c["source_demos_parallel_tasks"] * c["source_per_parallel"] + c["source_demos_dex_tasks"] * c["source_per_dex"]
    default_total = c["default_tasks"] * c["dexmg_per_task_default"]
    variant_total = c["variant_tasks"] * c["variant_per"] * c["dexmg_per_task_default"]
    bigym_total = c["bigym_tasks"] * c["dexmg_per_task_default"]
    res["K4_counts"] = {"source_total": src_total, "stated_source_total": c["stated_total_source"],
                        "default_1000_each": default_total, "variant_1000_each": variant_total,
                        "bigym_1000_each": bigym_total,
                        "sum_of_stated_1000_blocks": default_total + variant_total + bigym_total,
                        "stated_total": c["stated_total_demos"],
                        "unexplained_remainder": c["stated_total_demos"] - (default_total + variant_total + bigym_total)}

    # K5: BC-RNN-GMM 과 BC-RNN, DP 의 비교 (보고 평균 기준)
    gmm_lt_bc = [TASKS[i] for i in range(9) if T1_GMM[i][0] < T1_BCRNN_V2[i][0]]
    gmm_gt_bc = [TASKS[i] for i in range(9) if T1_GMM[i][0] > T1_BCRNN_V2[i][0]]
    dp_gt_gmm = sum(T1_DP[i][0] > T1_GMM[i][0] for i in range(9))
    dp_gt_bc = [TASKS[i] for i in range(9) if T1_DP[i][0] > T1_BCRNN_V2[i][0]]
    dp_lt_bc = [TASKS[i] for i in range(9) if T1_DP[i][0] < T1_BCRNN_V2[i][0]]
    gmm_dex = [TASKS[i] for i in range(9) if TASKS[i] in DEXTEROUS and T1_GMM[i][0] < T1_BCRNN_V2[i][0]]
    res["K5_architecture_counts"] = {
        "gmm_below_bcrnn": gmm_lt_bc, "n_gmm_below_bcrnn": len(gmm_lt_bc),
        "gmm_above_bcrnn": gmm_gt_bc, "n_gmm_above_bcrnn": len(gmm_gt_bc),
        "gmm_below_bcrnn_among_dex_tasks": gmm_dex,
        "n_dex_tasks": sum(t in DEXTEROUS for t in TASKS),
        "dp_above_gmm_n": int(dp_gt_gmm), "dp_above_bcrnn": dp_gt_bc, "n_dp_above_bcrnn": len(dp_gt_bc),
        "dp_below_bcrnn": dp_lt_bc,
    }

    # C1: "원천 시연 대비 DexMimicGen 향상(모든 과제)" 의 대조
    c1 = []
    for i, t in enumerate(TASKS):
        s = T1_SRC[i]
        for name, col_ in (("BC-RNN-GMM", T1_GMM), ("BC-RNN", T1_BCRNN_V2), ("DP", T1_DP)):
            v = col_[i]
            c1.append({"task": t, "policy": name, "source": s[0], "policy_value": v[0],
                       "diff_pp": round(v[0] - s[0], 1)})
    res["C1_vs_source"] = c1
    res["C1_summary"] = {"policies_below_source": [f"{r['task']}/{r['policy']}" for r in c1 if r["diff_pp"] < 0]}

    # C2: 잡음 생성 대비 ">58%" (Table III, 포인트 차이 기준)
    c2 = []
    for task, pol, dn, dx in T3:
        c2.append({"task": task, "policy": pol, "demonoise": dn[0], "dexmg": dx[0],
                   "diff_pp": round(dx[0] - dn[0], 1), "ratio": round(dx[0] / dn[0], 2)})
    res["C2_demonoise_gap"] = c2

    # S1: 3 시드 검정 (Welch, 요약 통계). ± 를 SD로 볼 때(S1a)와 SEM으로 볼 때(S1b)
    def pair(a, b, label):
        out = {"label": label, "a": a, "b": b}
        if a[1] == 0 or b[1] == 0:
            # ±0.0 은 3 시드 모두 같은 값이라는 뜻이라 검정이 정의되지 않는다(표기 확인 필요)
            out["p_sd"] = out["p_sem"] = None
            out["note"] = "±0.0 셀 포함: 검정 정의 안 됨"
            return out
        for tag, sd_fn in (("sd", lambda x: x), ("sem", lambda x: x * math.sqrt(3))):
            t, p = tstat_welch(a[0], sd_fn(a[1]), 3, b[0], sd_fn(b[1]), 3)
            out[f"p_{tag}"] = round(p, 6)
        return out

    s1 = [
        pair(T1_DP[1], T1_BCRNN_V2[1], "Threading DP vs BC-RNN"),
        pair(T1_GMM[7], T1_BCRNN_V2[7], "Coffee BC-RNN-GMM vs BC-RNN"),
        pair(T1_GMM[4], T1_BCRNN_V2[4], "Drawer BC-RNN-GMM vs BC-RNN"),
        pair(T1_GMM[3], T1_BCRNN_V2[3], "Box Cleanup BC-RNN-GMM vs BC-RNN"),
        pair(T1_DP[4], T1_BCRNN_V2[4], "Drawer DP vs BC-RNN"),
        pair(T1_BCRNN_V1[6], T1_BCRNN_V2[6], "Pouring BC-RNN v1 vs v2 (same policy, two versions)"),
        pair(T1_DP[6], T1_GMM[6], "Pouring DP vs BC-RNN-GMM"),
        pair(T1_DP[2], T1_BCRNN_V2[2], "Transport DP vs BC-RNN"),
        pair(T1_BCRNN_V2[2], T1_SRC[2], "Transport BC-RNN vs source demos"),
        pair(T1_GMM[2], T1_SRC[2], "Transport BC-RNN-GMM vs source demos"),
        pair(T1_GMM[7], T1_SRC[7], "Coffee BC-RNN-GMM vs source demos"),
    ]
    res["S1_welch_summary_stats"] = s1

    # S2: 평균 95% 구간 반폭 (3 시드, ± 를 SD로 볼 때). 평균 ±
    res["S2_ci_half_width_if_sd_dp"] = {TASKS[i]: (round(ci_half_sd(T1_DP[i][1]), 2) if T1_DP[i][1] else "SD=0")
                                        for i in range(9)}

    # B1: 보고된 ± 와 이항 표준오차(N=50 가정) 비교. ± 는 SD로 가정
    b1 = []
    for i, t in enumerate(TASKS):
        for n, col_ in (("BC-RNN-GMM", T1_GMM), ("BC-RNN", T1_BCRNN_V2), ("DP", T1_DP)):
            p = col_[i][0] / 100
            se = 100 * math.sqrt(p * (1 - p) / N_ROLLOUT_ASSUMED)
            b1.append({"task": t, "policy": n, "mean": col_[i][0], "reported_pm": col_[i][1],
                       "binom_se_n50": round(se, 2), "pm_below_binom": col_[i][1] < se})
    res["B1_pm_vs_binomial_n50_assumed"] = b1
    res["B1_summary"] = {"cells": len(b1), "pm_below_binom_if_sd": sum(r["pm_below_binom"] for r in b1),
                         "pm_below_binom_if_sem": sum(r["reported_pm"] * math.sqrt(3) < r["binom_se_n50"] for r in b1)}

    # W1: 실로봇. 10회씩 두 컵 = 20회로 가정 (18/20 과 0/20)
    n = 2 * TEXT["realworld_trials_per_cup"]
    k_dex = round(TEXT["realworld_dexmg"] / 100 * n)
    lo, hi = wilson(k_dex, n)
    lo0, hi0 = wilson(0, n)
    _or, p_f = stats.fisher_exact([[k_dex, n - k_dex], [0, n]])
    res["W1_realworld"] = {"n_per_policy_assumed": n, "dexmg_successes": k_dex,
                           "dexmg_wilson95": [round(lo, 3), round(hi, 3)],
                           "source_wilson95": [round(lo0, 3), round(hi0, 3)],
                           "fisher_two_sided_p": float(p_f),
                           "note": "논문은 '90%', '0%' 만 보고. 20회 가정은 'two cups x 10 trials' 문장에서 온 것"}

    # V1: 데모 수 합산이 21,000 과 맞는지는 K4 에서 확인.
    res["version"] = "arXiv 2410.24185v2 (2025-03-06); v1 은 BC-RNN 열만 다름 (K2)"
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print("wrote", OUT)
    print("K2 summary", res["K2_summary"])
    print("K4", res["K4_counts"])
    print("K5 gmm<bc", res["K5_architecture_counts"]["n_gmm_below_bcrnn"],
          "dp>gmm", res["K5_architecture_counts"]["dp_above_gmm_n"],
          "dp>bc", res["K5_architecture_counts"]["n_dp_above_bcrnn"])
    print("C1 below source", res["C1_summary"])
    print("B1", res["B1_summary"])
    print("W1", res["W1_realworld"])


if __name__ == "__main__":
    main()
