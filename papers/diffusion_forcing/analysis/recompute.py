"""논문 표 값을 옮겨 적고 평균, 신뢰구간, 검정, 순위, 본문 주장과의 대조를 다시 계산한다.

실행: python3 analysis/recompute.py  →  analysis/recompute_results.json (numpy, scipy만 사용, 수 초)
값의 출처: paper/paper.txt (arXiv v4 = NeurIPS 2024 본문). 표 1 = Table 1, 표 2 = Table 2.
v1 값은 paper/paper_v1.txt의 같은 표에서 옮겼다.

주의: 표 1은 시행 수(N)를 논문이 적지 않았다. 표 2는 우리 방법만 n = 5 시드(논문 명시)다.
기준선의 ± 가 표준편차인지 표준오차인지, 기준선의 n이 몇인지도 적혀 있지 않다.
그래서 검정은 n을 가정한 민감도 분석으로만 보고한다.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
results = {}

# ---------------------------------------------------------------- 표 1 (planning, v4 기준)
# (mean, sd). 'Ours'와 'Ours wo/ MCG'는 시행 수 N이 적히지 않았다.
T1_COLS = ["MPPI", "CQL", "IQL", "Diffuser*", "Diffuser_diffused_action", "Ours_wo_MCG", "Ours"]
T1 = {
    "Maze2D U-Maze": [33.2, 5.7, 47.4, (113.9, 3.1), (6.3, 2.1), (110.1, 3.9), (116.7, 2.0)],
    "Maze2D Medium": [10.2, 5.0, 34.9, (121.5, 2.7), (13.5, 2.3), (136.1, 10.2), (149.4, 7.5)],
    "Maze2D Large": [5.1, 12.5, 58.6, (123.0, 6.4), (6.3, 2.1), (142.8, 5.6), (159.0, 2.7)],
    "Multi2D U-Maze": [41.2, None, 24.8, (128.9, 1.8), (32.8, 1.7), (107.7, 4.9), (119.1, 4.0)],
    "Multi2D Medium": [15.4, None, 12.1, (127.2, 3.4), (22.0, 2.7), (145.6, 6.5), (152.3, 9.9)],
    "Multi2D Large": [8.0, None, 13.9, (132.1, 5.8), (6.9, 1.7), (129.8, 1.5), (167.1, 2.7)],
}
# 논문 표에 적힌 평균 행(Single-task, Multi-task). 재계산 값과 비교한다.
T1_PRINTED_AVG = {
    "single": {"MPPI": 16.2, "CQL": 7.7, "IQL": 47.0, "Diffuser*": 119.5, "Diffuser_diffused_action": 8.7,
               "Ours_wo_MCG": 129.67, "Ours": 141.7},
    "multi": {"MPPI": 21.5, "IQL": 16.9, "Diffuser*": 129.4, "Diffuser_diffused_action": 20.6,
              "Ours_wo_MCG": 127.7, "Ours": 146.2},
}


def t1_mean(env_keys, col_idx):
    vals = []
    for k in env_keys:
        v = T1[k][col_idx]
        if v is None:
            continue
        vals.append(v[0] if isinstance(v, tuple) else v)
    return float(np.mean(vals))


single_keys = ["Maze2D U-Maze", "Maze2D Medium", "Maze2D Large"]
multi_keys = ["Multi2D U-Maze", "Multi2D Medium", "Multi2D Large"]
avg_check = {}
for col_name, col_idx in [("Diffuser*", 3), ("Diffuser_diffused_action", 4), ("Ours_wo_MCG", 5), ("Ours", 6),
                          ("MPPI", 0), ("IQL", 2)]:
    s = round(t1_mean(single_keys, col_idx), 2)
    m = round(t1_mean(multi_keys, col_idx), 2)
    avg_check[col_name] = {
        "single_recomputed": s,
        "single_printed": T1_PRINTED_AVG["single"].get(col_name),
        "multi_recomputed": m,
        "multi_printed": T1_PRINTED_AVG["multi"].get(col_name),
    }
results["V1_table1_average_recheck"] = avg_check


def welch(m1, s1, n1, m2, s2, n2):
    """Welch t와 양측 p, 두 집단의 표준오차 기준."""
    se = np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    t = (m1 - m2) / se
    df = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 2 / ((s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1))
    p = 2 * stats.t.sf(abs(t), df)
    return float(t), float(p)


# V2: MCG 효과(Ours 대 Ours wo/ MCG)와 Diffuser* 대 Ours. 시행 수 N을 5와 10으로 가정한 민감도 분석.
comparisons = []
for env in T1:
    ours = T1[env][6]
    woM = T1[env][5]
    dif = T1[env][3]
    for label, other in [("Ours_vs_Ours_wo_MCG", woM), ("Ours_vs_Diffuser*", dif)]:
        row = {"env": env, "comparison": label, "diff_mean": round(ours[0] - other[0], 2)}
        for n in (5, 10):
            t, p = welch(ours[0], ours[1], n, other[0], other[1], n)
            row[f"welch_n{n}_t"] = round(t, 2)
            row[f"welch_n{n}_p"] = float(f"{p:.2g}")
        comparisons.append(row)
results["V2_table1_pairwise_sensitivity_assumed_n"] = comparisons

# K1: 본문 주장 "DF가 모든 6개 환경에서 Diffuser와 모든 기준선보다 낫다" 대조.
claim_violations = []
for env in T1:
    ours = T1[env][6][0]
    for i, col in enumerate(T1_COLS[:-1]):
        v = T1[env][i]
        if v is None:
            continue
        val = v[0] if isinstance(v, tuple) else v
        if val >= ours:
            claim_violations.append({"env": env, "baseline": col, "baseline_mean": val, "ours": ours})
results["K1_claim_outperform_all_baselines_all_envs"] = {
    "text_claim": "DF outperforms Diffuser and all baselines across all 6 environments (Sec. 4.2)",
    "violations": claim_violations,
}

# ---------------------------------------------------------------- 표 2 (시계열 CRPSsum, v4)
T2_COLS = ["Exchange", "Solar", "Electricity", "Traffic", "Taxi", "Wikipedia"]
T2 = {
    "VES": [0.005, 0.900, 0.880, 0.350, None, None],
    "VAR": [0.005, 0.830, 0.039, 0.290, None, None],
    "VAR-Lasso": [0.012, 0.510, 0.025, 0.150, None, 3.100],
    "GARCH": [0.023, 0.880, 0.190, 0.370, None, None],
    "DeepAR": [None, 0.336, 0.023, 0.055, None, 0.127],
    "LSTM-Copula": [0.007, 0.319, 0.064, 0.103, 0.326, 0.241],
    "GP-Copula": [0.007, 0.337, 0.025, 0.078, 0.208, 0.086],
    "KVAE": [0.014, 0.340, 0.051, 0.100, None, 0.095],
    "NKF": [None, 0.320, 0.016, 0.100, None, 0.071],
    "Transformer-MAF": [0.005, 0.301, 0.021, 0.056, 0.179, 0.063],
    "TimeGrad": [0.006, 0.287, 0.021, 0.044, 0.114, 0.049],
    "ScoreGrad": [0.006, 0.256, 0.019, 0.041, 0.101, 0.043],
    "Ours": [0.003, 0.289, 0.023, 0.040, 0.075, 0.085],
}
T2_SD = {
    "VES": [0.000, 0.003, 0.004, 0.002, None, None],
    "VAR": [0.000, 0.006, 0.001, 0.001, None, None],
    "VAR-Lasso": [0.000, 0.006, 0.000, 0.002, None, 0.004],
    "GARCH": [0.000, 0.002, 0.001, 0.001, None, None],
    "DeepAR": [None, 0.014, 0.001, 0.003, None, 0.042],
    "LSTM-Copula": [0.000, 0.011, 0.008, 0.006, 0.007, 0.033],
    "GP-Copula": [0.000, 0.024, 0.002, 0.002, 0.183, 0.004],
    "KVAE": [0.002, 0.025, 0.019, 0.005, None, 0.012],
    "NKF": [None, 0.020, 0.001, 0.002, None, 0.002],
    "Transformer-MAF": [0.003, 0.014, 0.000, 0.001, 0.002, 0.003],
    "TimeGrad": [0.001, 0.020, 0.001, 0.006, 0.020, 0.002],
    "ScoreGrad": [0.001, 0.015, 0.001, 0.004, 0.004, 0.002],
    "Ours": [0.001, 0.002, 0.001, 0.004, 0.002, 0.007],
}

# V3: 우리 방법의 95% 신뢰구간 (n = 5, t_{0.975,4} = 2.776). 표준편차를 가정.
T_CRIT = stats.t.ppf(0.975, df=4)
ci_rows = []
for j, ds in enumerate(T2_COLS):
    m, s = T2["Ours"][j], T2_SD["Ours"][j]
    half = T_CRIT * s / np.sqrt(5)
    ci_rows.append({"dataset": ds, "mean": m, "sd": s, "ci95_low": round(m - half, 4), "ci95_high": round(m + half, 4)})
results["V3_ours_ci95_n5"] = {"t_crit": round(float(T_CRIT), 4), "rows": ci_rows}

# V4: 데이터셋별 순위. 값이 낮을수록 좋다. 같은 값은 같은 순위로 센다.
rank_rows = {}
for j, ds in enumerate(T2_COLS):
    vals = {k: v[j] for k, v in T2.items() if v[j] is not None}
    ours = vals["Ours"]
    better = sorted([k for k, v in vals.items() if v < ours and k != "Ours"], key=lambda k: vals[k])
    rank = 1 + len(better)
    rank_rows[ds] = {"ours_rank": rank, "better_than_ours": [f"{k} {vals[k]}" for k in better],
                     "n_methods": len(vals)}
results["V4_table2_ranks"] = rank_rows

# K2: 본문 "Wikipedia에서 4위" 와 "대부분 ScoreGrad와 동률" 대조.
results["K2_text_claim_ranks"] = {
    "text_claim": "outperforms all prior methods except [68] (ScoreGrad) overall tied; Wikipedia fourth place",
    "wikipedia_rank_recomputed": rank_rows["Wikipedia"]["ours_rank"],
    "wins_or_ties_vs_scoregrad_per_dataset": {
        ds: ("Ours better" if T2["Ours"][j] < T2["ScoreGrad"][j] else
             "ScoreGrad better" if T2["Ours"][j] > T2["ScoreGrad"][j] else "tie")
        for j, ds in enumerate(T2_COLS)
    },
}

# V5: 우리 방법과 ScoreGrad의 Welch 검정. 기준선의 n과 ±의 의미를 몰라 n=5 가정, 민감도 n=10.
sg_rows = []
for j, ds in enumerate(T2_COLS):
    row = {"dataset": ds, "ours": T2["Ours"][j], "scoregrad": T2["ScoreGrad"][j]}
    for n in (5, 10):
        t, p = welch(T2["Ours"][j], T2_SD["Ours"][j], 5, T2["ScoreGrad"][j], T2_SD["ScoreGrad"][j], n)
        row[f"welch_n{n}_p"] = float(f"{p:.2g}")
    sg_rows.append(row)
results["V5_ours_vs_scoregrad_assumed_n"] = sg_rows

# K3: 표 2에서 서로 다른 ±의 크기와 반올림 한계. 소수 셋째 자리에서 0.001 차이는 반올림 범위와 같다.
results["K3_rounding_note"] = (
    "Table 2 values are printed to 3 decimals; differences of 0.001 are within rounding (+/-0.0005) "
    "and the Exchange 0.003 vs 0.005/0.006 comparisons inherit that limit."
)

# ---------------------------------------------------------------- 표 1 v1 대 v4
results["K4_v1_v4_table1"] = {
    "note": "v1 (arXiv 2024-07-01) and v4 (2024-12-10) Table 1 rows are numerically identical; "
            "v1 labels the ablation 'Ours wo/ MCTG' and v4 'Ours wo/ MCG'. Multi-task averages differ only by rounding "
            "(v1 146.17 / 20.57 vs v4 146.2 / 20.6).",
    "v1_multi_task_avg_ours": 146.17,
    "v4_multi_task_avg_ours": 146.2,
}

# V6: 로봇 성공률(본문 4.4). 시행 수가 적혀 있지 않다. 시행 수를 가정한 Wilson 95% 구간의 민감도 분석.
def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return float(c - h), float(c + h)


robot_rows = []
for label, rate in [("DF 성공률 80%", 0.80), ("DF 방해 조건 성공률 76%", 0.76), ("다음 프레임 확산 기준선 48%", 0.48)]:
    row = {"label": label, "rate": rate, "wilson95_by_assumed_n": {}}
    for n in (10, 20, 50):
        k = round(rate * n)
        lo, hi = wilson(k, n)
        row["wilson95_by_assumed_n"][str(n)] = {"k": k, "low": round(lo, 3), "high": round(hi, 3)}
    robot_rows.append(row)
results["V6_robot_success_wilson_assumed_n"] = {
    "note": "Trial counts are not reported in Sec. 4.4; intervals are for assumed n only.",
    "rows": robot_rows,
}

OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", OUT)
print("K1 violations:", claim_violations)
print("K2 wikipedia rank:", rank_rows["Wikipedia"])
print("V3 ours CI:", [(r["dataset"], r["ci95_low"], r["ci95_high"]) for r in ci_rows])
