"""FoAR (arXiv:2411.15753 v2) 표 I–IV와 본문 수치를 다시 계산하고 검정한다.

실행: python3 analysis/recompute.py   → analysis/recompute_results.json (수 초, 인터넷 불필요)

입력은 논문 표에서 옮긴 값이다(표 번호와 셀 위치를 주석으로 적었다).
- 시행 수: 표 I, III, IV는 20회(ASR이 5%의 배수), 표 II(몰기)는 10회(ASR이 10%의 배수)로 확인했다.
- 점수(Score)는 개별 시행 점수가 공개되지 않아 평균만 있다. 점수 검정은 Bhatia–Davis 상한
  (분산 <= m(1-m), 0–1 범위 변수)을 쓰는 보수적 Welch 검정이다. 분산이 실제로 더 작으면 p는 과대 추정된다.
- 표 II의 ± 값은 SD라고 가정한다(논문이 SD인지 SE인지 밝히지 않음).
"""
import json
import math
from pathlib import Path

from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
Z = stats.norm.ppf


def wilson(k, n, conf=0.95):
    z = Z(1 - (1 - conf) / 2)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(centre - half, 3), round(centre + half, 3)]


def fisher(k1, n1, k2, n2):
    _, p = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
    return p


def welch_bounded(m1, n1, m2, n2, lo=0.0, hi=1.0):
    """Bhatia–Davis 상한 분산을 쓰는 Welch 검정 (보수적)."""
    v1 = (m1 - lo) * (hi - m1) * n1 / (n1 - 1)
    v2 = (m2 - lo) * (hi - m2) * n2 / (n2 - 1)
    se = math.sqrt(v1 / n1 + v2 / n2)
    t = (m1 - m2) / se
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    return {"diff": round(m1 - m2, 3), "t": round(t, 2), "df": round(df, 1), "p": round(2 * stats.t.sf(abs(t), df), 4)}


def welch_sd(m1, s1, n1, m2, s2, n2):
    """보고된 SD로 Welch 검정 (표 II)."""
    res = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
    return {"diff": round(m1 - m2, 3), "t": round(float(res.statistic), 2), "p": float(f"{res.pvalue:.2e}")}


def mde_two_sample(sd, n, alpha=0.05, power=0.8):
    """두 독립 평균, 군당 n, 양측 검정에서 80% 검출력의 최소 차이."""
    return round((Z(1 - alpha / 2) + Z(power)) * sd * math.sqrt(2 / n), 3)


res = {}

# ---------- 표 I (n = 20): (점수, grasp 성공, 과제 성공) ----------
T1 = {
    "Wiping": {
        "ACT": (0.275, 13, 10), "Diffusion Policy": (0.400, 15, 12), "RISE": (0.500, 20, 15),
        "RISE (force-token)": (0.575, 17, 16), "RISE (force-concat)": (0.475, 20, 13),
        "FoAR (3D-cls)": (0.175, 8, 7), "FoAR": (0.875, 20, 20),
    },
    "Wiping (General)": {
        "ACT": (0.250, 13, 10), "Diffusion Policy": (0.350, 15, 10), "RISE": (0.500, 18, 16),
        "RISE (force-token)": (0.600, 18, 16), "RISE (force-concat)": (0.675, 20, 19),
        "FoAR (3D-cls)": (0.200, 8, 8), "FoAR": (0.850, 20, 20),
    },
    "Peeling": {  # 마지막 열은 peel 성공
        "ACT": (0.120, 19, 5), "Diffusion Policy": (0.386, 17, 14), "RISE": (0.377, 20, 10),
        "RISE (force-token)": (0.487, 19, 15), "RISE (force-concat)": (0.524, 20, 16),
        "FoAR (3D-cls)": (0.270, 19, 8), "FoAR": (0.756, 20, 20),
    },
}
n1 = 20
res["table1_ci_asr"] = {
    task: {m: {"grasp_ci": wilson(g, n1), "task_ci": wilson(t, n1)} for m, (_, g, t) in rows.items()}
    for task, rows in T1.items()
}
# FoAR 대 최강 기준선(점수 기준)과 대 RISE
best_base = {"Wiping": "RISE (force-token)", "Wiping (General)": "RISE (force-concat)", "Peeling": "RISE (force-concat)"}
res["table1_tests"] = {}
for task, rows in T1.items():
    f_s, f_g, f_t = rows["FoAR"]
    b = best_base[task]
    r_s, r_g, r_t = rows["RISE"]
    res["table1_tests"][task] = {
        "score_vs_best_baseline": {"baseline": b, **welch_bounded(f_s, n1, rows[b][0], n1)},
        "score_vs_RISE": {"baseline": "RISE", **welch_bounded(f_s, n1, r_s, n1)},
        "task_ASR_fisher_vs_RISE": round(fisher(f_t, n1, r_t, n1), 4),
        "task_ASR_fisher_vs_best_baseline": round(fisher(f_t, n1, rows[b][2], n1), 4),
        "grasp_ASR_fisher_vs_RISE": round(fisher(f_g, n1, r_g, n1), 4),
    }
# 기준선 전체 대비 FoAR 점수: 다중 비교 보정 없이 보고한다(Holm 보정은 아래에서 계산)
pvals = []
for task in T1:
    for m in T1[task]:
        if m == "FoAR":
            continue
        pvals.append(("T1 score " + task + " FoAR vs " + m,
                      welch_bounded(T1[task]["FoAR"][0], n1, T1[task][m][0], n1)["p"]))
pvals.sort(key=lambda x: x[1])
m_tot = len(pvals)
holm = []
running = 0.0
for i, (name, p) in enumerate(pvals):
    adj = min(1.0, (m_tot - i) * p)
    running = max(running, adj)
    holm.append({"test": name, "p": p, "holm": round(running, 4)})
res["table1_holm_scores"] = holm

# ---------- 표 II (chopping, n = 10) ----------
n2 = 10
res["table2"] = {
    "segments_welch_sd": welch_sd(3.9, 0.9, n2, 1.8, 0.6, n2),
    "avg_norm_length_welch_sd": welch_sd(0.353, 0.094, n2, 0.727, 0.411, n2),
    "place_asr_fisher": round(fisher(7, n2, 3, n2), 4),
    "place_ci": {"FoAR": wilson(7, n2), "RISE": wilson(3, n2)},
    "oracle_segments": 5.0,
    "ratio_segments": round(3.9 / 1.8, 2),
    "ratio_avg_len": round(0.353 / 0.727, 3),
}

# ---------- 표 III (wiping, n = 20): (설정, 점수, grasp, wipe) ----------
T3 = {
    "100Hz, reactive only": (0.650, 20, 17),
    "100Hz, predictor only": (0.650, 20, 16),
    "2Hz, predictor + reactive": (0.625, 20, 17),
    "10Hz, predictor + reactive": (0.800, 20, 20),
    "100Hz, predictor + reactive (FoAR)": (0.875, 20, 20),
}
full = T3["100Hz, predictor + reactive (FoAR)"]
res["table3_tests"] = {
    k: {**welch_bounded(full[0], 20, v[0], 20), "wipe_fisher": round(fisher(full[2], 20, v[2], 20), 4)}
    for k, v in T3.items() if k != "100Hz, predictor + reactive (FoAR)"
}
res["table3_ci_wipe"] = {k: wilson(v[2], 20) for k, v in T3.items()}

# ---------- 표 IV (Wiping General 외란, n = 20 추정): (점수, grasp, wipe) ----------
T4 = {
    "RISE": {"Original": (0.500, 18, 16), "Rewrite": (0.500, 18, 14), "Move": (0.600, 20, 20), "Rewrite+Move": (0.500, 20, 14)},
    "RISE (force-token)": {"Original": (0.600, 18, 16), "Rewrite": (0.450, 16, 18), "Move": (0.500, 18, 16), "Rewrite+Move": (0.600, 20, 20)},
    "FoAR": {"Original": (0.850, 20, 20), "Rewrite": (0.800, 20, 20), "Move": (0.850, 20, 20), "Rewrite+Move": (0.800, 20, 20)},
}
res["table4_tests"] = {
    "RISE_RM_vs_original_score": welch_bounded(0.500, 20, 0.500, 20),
    "forcetoken_RM_vs_original_score": welch_bounded(0.600, 20, 0.600, 20),
    "forcetoken_RM_vs_RISE_RM_fisher_wipe": round(fisher(20, 20, 14, 20), 4),
    "FoAR_RM_vs_RISE_RM_score": welch_bounded(0.800, 20, 0.500, 20),
    "FoAR_RM_vs_RISE_RM_wipe_fisher": round(fisher(20, 20, 14, 20), 4),
    "RISE_Move_vs_Original_score": welch_bounded(0.600, 20, 0.500, 20),
}

# ---------- 일관성 검사 (본문과 표) ----------
checks = {
    "K_text_0.875_0.850_0.756_match_table1": (0.875, 0.850, 0.756) == (T1["Wiping"]["FoAR"][0], T1["Wiping (General)"]["FoAR"][0], T1["Peeling"]["FoAR"][0]),
    "K_tableIII_FoAR_equals_tableI_wiping": T3["100Hz, predictor + reactive (FoAR)"] == (T1["Wiping"]["FoAR"][0], 20, 20),
    "K_tableIV_original_equals_tableI_WG": all(
        T4[m]["Original"][0] == T1["Wiping (General)"][m][0] for m in ["RISE", "RISE (force-token)", "FoAR"]),
    "K_tableIV_original_grasp_wipe_equal_tableI": all(
        (T4[m]["Original"][1], T4[m]["Original"][2]) == (T1["Wiping (General)"][m][1], T1["Wiping (General)"][m][2])
        for m in ["RISE", "RISE (force-token)", "FoAR"]),
    "K_forcetoken_RM_not_worse_than_original": T4["RISE (force-token)"]["Rewrite+Move"] >= T4["RISE (force-token)"]["Original"],
    "K_force_concat_wiping_below_RISE": T1["Wiping"]["RISE (force-concat)"][0] < T1["Wiping"]["RISE"][0],
    "K_text_segments_ratio_over_2": 3.9 / 1.8 > 2,
    "K_score_granularity_n20_multiple_of_0.025": all(
        abs(round(v[0] / 0.025) * 0.025 - v[0]) < 1e-9
        for task in ["Wiping", "Wiping (General)"] for v in T1[task].values()),
    # 본문 주장: "RISE (force-token)은 복합 외란에서 성능이 떨어진다"
    "K_claim_forcetoken_struggles_in_RM_supported": T4["RISE (force-token)"]["Rewrite+Move"][0] < T4["RISE (force-token)"]["Original"][0],
    # 본문 주장: "외란에서 RISE는 전반적으로 약하다" (Move 시나리오에서 RISE는 원래 조건보다 높다)
    "K_claim_RISE_Move_worse_supported": T4["RISE"]["Move"][0] < T4["RISE"]["Original"][0],
}
res["consistency"] = checks

# ---------- 검출력 ----------
res["mde_scores_n20_sd0.33"] = mde_two_sample(0.33, 20)
res["mde_scores_n20_sd0.40"] = mde_two_sample(0.40, 20)
res["mde_scores_n10_sd0.40"] = mde_two_sample(0.40, 10)
res["mde_rate_n20_p0.5"] = round((Z(0.975) + Z(0.8)) * math.sqrt(2 * 0.25 / 20), 3)
res["mde_rate_n20_p0.9"] = round((Z(0.975) + Z(0.8)) * math.sqrt(2 * 0.09 / 20), 3)

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1)[:6000])
