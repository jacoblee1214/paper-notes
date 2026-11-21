"""SHAPE (arXiv:2611.24585 v1)의 표 수치를 다시 계산하고 일관성을 검사한다.

입력: 논문 Table 2, 3, 4, S4와 본문 수치를 아래 상수에 그대로 옮겼다(출처 주석).
외부 데이터, 가중치, 재현 실험은 쓰지 않는다.
출력: analysis/recompute_results.json (표준출력에도 요약)
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초. perm 검정 포함 약 10초)

가정(논문에 없는 것은 표시한다)
- 6과제 점수의 분산 상한은 논문 Appendix C 식을 그대로 쓴다: Var(S) <= sum_t mu_t(1-mu_t) / (36 n), n = 20.
- USB와 Type-C 셀은 평균 부분 점수다. 이진 과제는 성공 수 / 20이다.
- Table 2 행 배치는 pdftotext 결과에서 맞췄다. 행별 USB와 Type-C 평균이 Table S4 단계 표에서 역산한 값과 맞는지 검사한다(섹션 K3).
- H2R 계층 검정: 논문은 정확 CMH를 쓴다. 여기서는 Mantel-Haenszel 점근 검정과 층화 순열 검정(근사)을 쓴다.
- 1,200 쌍의 RoboCasa 개별 결과는 논문에 없어 p값은 다시 계산하지 못한다. 집계 평균만 검사한다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
N = 20
TASKS = ["Insert Battery", "Reinsert USB", "Reinsert Type-C", "Wipe Bowl", "Open Cabinet", "Close Bottle Cap"]
RES = {}

# 논문 Table 2 (보고된 셀). 이진 과제는 성공 수, USB와 Type-C는 평균 부분 점수.
T2 = {
    "GR00T N1.7 joint": [5, 0.18, 0.52, 12, 18, 12],
    "pi0.5 joint":      [13, 0.25, 0.62, 13, 17, 15],
    "SHAPE EEF":        [6, 0.15, 0.54, 14, 17, 12],
    "SHAPE joint":      [16, 0.55, 0.64, 18, 18, 20],
    "w/o human EEF":    [7, 0.15, 0.42, 10, 15, 13],
    "w/o human joint":  [9, 0.25, 0.55, 16, 18, 18],
    "w/o FAST EEF":     [6, 0.00, 0.20, 1, 13, 2],
    "w/o FAST joint":   [11, 0.08, 0.51, 9, 11, 14],
    "w/o both EEF":     [3, 0.03, 0.11, 5, 11, 2],
    "w/o both joint":   [6, 0.33, 0.35, 10, 12, 9],
}
T2_SCORE = {  # 보고된 six-task score
    "GR00T N1.7 joint": 0.51, "pi0.5 joint": 0.63, "SHAPE EEF": 0.52, "SHAPE joint": 0.80,
    "w/o human EEF": 0.47, "w/o human joint": 0.64, "w/o FAST EEF": 0.22, "w/o FAST joint": 0.47,
    "w/o both EEF": 0.20, "w/o both joint": 0.42,
}
# 논문 Table S4 (20회 중 단계 달성 수)
S4 = {  # (USB unplug, USB insert, TypeC unplug, TypeC grasp, TypeC insert)
    "GR00T N1.7 joint": (5, 2, 16, 16, 2), "pi0.5 joint": (8, 2, 20, 20, 1),
    "SHAPE EEF": (6, 0, 20, 16, 0), "SHAPE joint": (20, 2, 20, 20, 2),
    "w/o human EEF": (6, 0, 16, 12, 0), "w/o human joint": (10, 0, 20, 14, 2),
    "w/o FAST EEF": (0, 0, 6, 6, 1), "w/o FAST joint": (3, 0, 15, 15, 3),
    "w/o both EEF": (1, 0, 5, 2, 0), "w/o both joint": (12, 1, 12, 10, 1),
}
# 논문 Table 3 (H2R, EEF 스트림, 과제별 성공 수: Eggplant, Cup, Stack)
H2R = {"w/o both": (0, 0, 0), "w/o human": (0, 3, 1), "w/o FAST": (3, 2, 2), "SHAPE": (10, 8, 3)}
# 논문 Table 4 (RoboCasa GR-1, 성공률 %: Close 6과제, PnP 18과제, All)
T4 = {"GR00T N1.7": (69.2, 60.4, 62.6), "pi0.5": (51.4, 69.2, 64.8), "w/o both": (61.4, 62.2, 62.0),
      "w/o human": (62.6, 66.1, 65.2), "w/o FAST": (57.1, 73.8, 69.6), "SHAPE": (66.6, 74.1, 72.2)}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def frac(x):
    return x / N if isinstance(x, int) else float(x)


# K1. 6과제 점수, 분산 상한, 보고값 비교
SCORE = {}
for name, row in T2.items():
    mu = np.array([frac(v) for v in row])
    S = float(mu.mean())
    V = float(np.sum(mu * (1 - mu)) / (36 * N))
    h = 1.96 * math.sqrt(V)
    SCORE[name] = {"score": round(S, 4), "reported": T2_SCORE[name],
                   "matches_reported": round(S, 2) == T2_SCORE[name],
                   "ci95": [round(S - h, 3), round(S + h, 3)], "V": V}
RES["K1_six_task_scores"] = {k: {kk: vv for kk, vv in v.items() if kk != "V"} for k, v in SCORE.items()}


def z_test(a, b):
    z = (SCORE[a]["score"] - SCORE[b]["score"]) / math.sqrt(SCORE[a]["V"] + SCORE[b]["V"])
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"comparison": f"{a} - {b}", "diff": round(SCORE[a]["score"] - SCORE[b]["score"], 4),
            "z": round(z, 2), "p_two_sided": float(f"{p:.2g}")}


RES["K1_z_tests"] = [
    z_test("SHAPE joint", "pi0.5 joint"),
    z_test("SHAPE joint", "GR00T N1.7 joint"),
    z_test("SHAPE joint", "w/o human joint"),
    z_test("w/o human joint", "pi0.5 joint"),
    z_test("SHAPE joint", "SHAPE EEF"),
    z_test("w/o FAST joint", "w/o both joint"),
    z_test("w/o human joint", "w/o both joint"),
]

# K2. 본문 주장의 셀 단위 확인
pairs = []
for m in ["SHAPE", "w/o human", "w/o FAST", "w/o both"]:
    for i, t in enumerate(TASKS):
        pairs.append((m, t, frac(T2[f"{m} joint"][i]), frac(T2[f"{m} EEF"][i])))
RES["K2_joint_vs_eef"] = {
    "pairs": len(pairs),
    "joint_better": sum(p[2] > p[3] for p in pairs),
    "exceptions": [(p[0], p[1], round(p[2], 2), round(p[3], 2)) for p in pairs if p[2] <= p[3]],
    "score_gap_joint_minus_eef": {m: round(T2_SCORE[f"{m} joint"] - T2_SCORE[f"{m} EEF"], 2)
                                  for m in ["SHAPE", "w/o human", "w/o FAST", "w/o both"]},
}
rem = []
for stream in ["EEF", "joint"]:
    for i, t in enumerate(TASKS):
        shape = T2[f"SHAPE {stream}"][i]
        wofast = T2[f"w/o FAST {stream}"][i]
        wh = T2[f"w/o human {stream}"][i]
        wb = T2[f"w/o both {stream}"][i]
        rem.append((stream, t, "SHAPE vs w/o FAST", frac(shape) - frac(wofast)))
        rem.append((stream, t, "w/o human vs w/o both", frac(wh) - frac(wb)))
RES["K2_removing_LFAST"] = {
    "comparisons": len(rem),
    "pretrained_with_LFAST_higher": sum(r[3] > 0 for r in rem),
    "not_higher": [(r[0], r[1], r[2], round(r[3], 2)) for r in rem if r[3] <= 0],
}
bin_idx = [0, 3, 4, 5]  # 이진 과제: Insert Battery, Wipe Bowl, Open Cabinet, Close Bottle Cap
for name in ["w/o human joint", "SHAPE joint"]:
    k = sum(T2[name][i] for i in bin_idx)
    RES.setdefault("K2_binary_tasks", {})[name] = f"{k}/80 = {100 * k / 80:.2f}%"
strict = {
    "SHAPE joint": sum(T2["SHAPE joint"][i] for i in bin_idx) + S4["SHAPE joint"][1] + S4["SHAPE joint"][4],
    "pi0.5 joint": sum(T2["pi0.5 joint"][i] for i in bin_idx) + S4["pi0.5 joint"][1] + S4["pi0.5 joint"][4],
    "GR00T N1.7 joint": sum(T2["GR00T N1.7 joint"][i] for i in bin_idx) + S4["GR00T N1.7 joint"][1] + S4["GR00T N1.7 joint"][4],
}
RES["K2_strict_full_completions"] = {k: f"{v}/120 = {100 * v / 120:.1f}%" for k, v in strict.items()}
RES["K2_three_task_score_joint_insert_usb_wipe"] = {
    "SHAPE joint": round((T2["SHAPE joint"][0] / N + T2["SHAPE joint"][1] + T2["SHAPE joint"][3] / N) / 3, 3),
    "w/o FAST joint": round((T2["w/o FAST joint"][0] / N + T2["w/o FAST joint"][1] + T2["w/o FAST joint"][3] / N) / 3, 3),
}


# K3. Table S4 단계 표에서 USB와 Type-C 평균을 역산하여 Table 2 행 배치를 검사한다
def usb_mean(unplug, insert):
    return (0.5 * (unplug - insert) + 1.0 * insert) / N


def tc_mean(unplug, grasp, insert):
    return (0.3 * (unplug - grasp) + 0.6 * (grasp - insert) + 1.0 * insert) / N


chk = {}
for name, (uu, ui, tu, tg, ti) in S4.items():
    a, b = round(usb_mean(uu, ui), 3), round(tc_mean(tu, tg, ti), 3)
    ra, rb = T2[name][1], T2[name][2]
    chk[name] = {"USB_from_S4": a, "USB_reported": ra, "TypeC_from_S4": b, "TypeC_reported": rb,
                 "match": abs(a - ra) <= 0.006 and abs(b - rb) <= 0.006}
RES["K3_stage_table_consistency"] = chk
RES["K3_all_match"] = all(v["match"] for v in chk.values())

# K4. 시도 수 (1,680회)
RES["K4_trial_count"] = {
    "pretrained_4_models_x_2_streams_x_6_tasks": 4 * 2 * 6 * N,
    "baselines_2_models_x_6_tasks": 2 * 6 * N,
    "H2R_4_models_x_2_streams_x_3_tasks": 4 * 2 * 3 * N,
    "total": 4 * 2 * 6 * N + 2 * 6 * N + 4 * 2 * 3 * N,
    "paper_claim": 1680,
}

# K5. H2R (Table 3)
RES["K5_h2r_totals"] = {k: f"{sum(v)}/60" for k, v in H2R.items()}
RES["K5_h2r_joint_stream"] = {"successes": 0, "trials": 4 * 3 * N}
RES["K5_h2r_eef_stream"] = {"successes": sum(sum(v) for v in H2R.values()), "trials": 4 * 3 * N}
RES["K5_h2r_wilson95"] = {k: [round(100 * x, 1) for x in wilson(sum(v), 60)] for k, v in H2R.items()}


def fisher_p(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


pooled = {
    "SHAPE vs w/o FAST (LFAST, human pretraining on)": (21, 60, 7, 60),
    "SHAPE vs w/o human (human data, LFAST on)": (21, 60, 4, 60),
    "w/o FAST vs w/o both (human data, LFAST off)": (7, 60, 0, 60),
    "w/o human vs w/o both (LFAST, robot-only pretraining)": (4, 60, 0, 60),
}
RES["K5_pooled_fisher"] = {k: {"diff_pp": round(100 * (v[0] / v[1] - v[2] / v[3]), 1),
                               "fisher_p": round(fisher_p(*v), 4)} for k, v in pooled.items()}


def strata(x, y, n=N):
    return [(x[i], n - x[i], y[i], n - y[i]) for i in range(3)]


def mh_chi2(st):
    num = var = 0.0
    for a, b, c, d in st:
        n1, n2 = a + b, c + d
        m1, m2 = a + c, b + d
        n = n1 + n2
        if n <= 1:
            continue
        num += a - n1 * m1 / n
        var += n1 * n2 * m1 * m2 / (n * n * (n - 1))
    chi2 = num * num / var
    return round(chi2, 2), round(float(stats.chi2.sf(chi2, 1)), 5)


def perm_p(x, y, n_perm=20000, seed=0):
    """과제 안에서 두 집단 라벨을 섞는 층화 순열 검정 (정확 CMH의 근사)."""
    rng = np.random.default_rng(seed)
    obs = sum(x) - sum(y)
    pools = [np.array([1] * xi + [0] * (N - xi) + [1] * yi + [0] * (N - yi)) for xi, yi in zip(x, y)]
    cnt = 0
    for _ in range(n_perm):
        d = 0
        for arr in pools:
            perm = rng.permutation(arr)
            d += perm[:N].sum() - perm[N:].sum()
        cnt += abs(d) >= abs(obs)
    return round((cnt + 1) / (n_perm + 1), 5)


for label, other, paper in [("SHAPE vs w/o FAST", "w/o FAST", "p = 0.004"),
                            ("SHAPE vs w/o human", "w/o human", "p < 0.001")]:
    st = strata(H2R["SHAPE"], H2R[other])
    chi2, p_mh = mh_chi2(st)
    RES.setdefault("K5_stratified", {})[label] = {"MH_chi2": chi2, "MH_p": p_mh,
                                                  "perm_p_20k": perm_p(H2R["SHAPE"], H2R[other]),
                                                  "paper": paper}


# K7. 연속 헤드 대 SHAPE (Table 5, S9의 H2R, 과제별 Pick Eggplant, Pick Cup, Stack Cups)
CONT = (5, 5, 1)  # 논문 Sec. 6.4, 과제별 성공 수
st = strata(H2R["SHAPE"], CONT)
chi2, p_mh = mh_chi2(st)
RES["K7_cont_head_H2R"] = {"SHAPE_total": 21, "cont_total": sum(CONT), "diff_pts": round(100 * (21 - sum(CONT)) / 60, 1),
                           "MH_chi2": chi2, "MH_p": p_mh, "perm_p_20k": perm_p(H2R["SHAPE"], CONT),
                           "paper_CMH_p": 0.055}

# K6. RoboCasa 집계 (Close 6과제, PnP 18과제 가중 평균 = All)
RES["K6_robocasa_weighted"] = {k: {"calc_All": round((6 * c + 18 * p) / 24, 2), "reported_All": a,
                                   "match": abs((6 * c + 18 * p) / 24 - a) < 0.06}
                               for k, (c, p, a) in T4.items()}
RES["K6_robocasa_differences_pp"] = {
    "human data with LFAST (72.2 - 65.2)": round(72.2 - 65.2, 1),
    "human data without LFAST (69.6 - 62.0)": round(69.6 - 62.0, 1),
    "LFAST with human data (72.2 - 69.6)": round(72.2 - 69.6, 1),
    "LFAST without human data (65.2 - 62.0)": round(65.2 - 62.0, 1),
    "SHAPE minus w/o both (72.2 - 62.0)": round(72.2 - 62.0, 1),
    "SHAPE minus pi0.5 (72.2 - 64.8)": round(72.2 - 64.8, 1),
}

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
for k, v in RES.items():
    print(k, json.dumps(v, ensure_ascii=False)[:400])
