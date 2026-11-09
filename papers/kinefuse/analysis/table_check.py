"""KineFuse 표 I·II·III와 본문 수치 재계산 (CPU, 단일 스레드). 실행: python3 analysis/table_check.py

입력은 논문 PDF의 표 값(paper/paper.txt로 확인, 눈으로 대조)이다. 이 스크립트는 논문 PDF를 읽지 않는다.
- C1: Table I 평균과 본문 수치(4.04 cm, 8.11 cm, 2×, 47.6°) 재계산, 'lowest at every level', '격차 확대' 주장 검사
- C2: Table II 평균, 본문 '15×'의 정의, 'GT 대비 9%' 주장 후보 탐색
- C3: 3-seed 평균±표준편차(ddof=1 가정, ddof=0 민감도)로 Welch t 검정, 쌍별 Holm 보정
- C4: 16-token의 오차·성공이 폐색 수준에 거의 무관한지(변동계수), KineFuse와 비교
- C5: 각도 오차 순위와 성공 순위의 상관(Spearman, 5개 설계)
- C6: Table III 실환경 한 궤적에서 KineFuse가 회전 오차에서 뒤집히는 구간
- C7: Fig. 7 점 위치가 Table I의 어떤 부분집합 평균과 맞는지(눈 판독 값과 비교)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OCC = [0, 10, 30, 50, 70, 90]
N_SEEDS = 3

# Table I: open-loop tracking, mean (3 seeds), std
T1 = {
    "V-only":    {"pos": [8.30, 6.80, 9.12, 7.59, 9.68, 7.14], "pos_sd": [.51, .25, .54, .45, .73, .44],
                  "ang": [59.9, 64.2, 57.7, 60.6, 54.9, 79.4], "ang_sd": [3.2, 3.4, 3.4, 3.5, 1.3, 3.7]},
    "Naive V+H": {"pos": [8.30, 7.76, 9.13, 10.1, 7.01, 5.57], "pos_sd": [.04, .18, .21, .17, .20, .19],
                  "ang": [48.0, 47.3, 47.9, 47.7, 50.8, 53.7], "ang_sd": [1.1, 1.1, 1.5, 1.3, 1.2, 1.4]},
    "FingerMLP": {"pos": [5.01, 5.46, 5.35, 6.07, 4.84, 6.50], "pos_sd": [.23, .32, .28, .34, .24, .62],
                  "ang": [49.2, 52.8, 49.7, 50.9, 47.0, 67.6], "ang_sd": [4.2, 5.1, 3.8, 4.3, 1.8, 4.2]},
    "16-token":  {"pos": [5.84, 5.84, 5.81, 5.79, 5.80, 5.79], "pos_sd": [.08] * 6,
                  "ang": [69.2, 69.3, 69.3, 69.3, 69.3, 69.2], "ang_sd": [1.3] * 6},
    "KineFuse":  {"pos": [3.68, 3.49, 3.47, 4.17, 4.33, 5.11], "pos_sd": [.14, .10, .12, .06, .06, .07],
                  "ang": [47.7, 47.7, 47.6, 47.7, 47.6, 47.6], "ang_sd": [2.7] * 6},
}

# Table II: downstream success per episode, mean, std (3 seeds). GT = 21.25 (no std)
GT = 21.25
T2 = {
    "V-only":    {"m": [0.46, 0.63, 0.83, 1.58, 1.97, 1.52], "sd": [.16, .06, .24, .34, 1.0, .18]},
    "Naive V+H": {"m": [1.54, 1.40, 1.86, 1.53, 1.75, 2.00], "sd": [.47, .55, .58, .53, .40, .29]},
    "FingerMLP": {"m": [1.63, 0.61, 0.79, 1.48, 5.04, 2.45], "sd": [.06, .14, .67, .19, 2.1, .38]},
    "16-token":  {"m": [0.27, 0.23, 0.33, 0.30, 0.29, 0.30], "sd": [.04, .03, .05, .04, .05, .10]},
    "KineFuse":  {"m": [4.61, 4.47, 4.72, 4.47, 5.10, 3.81], "sd": [1.3, 1.4, .42, .83, .28, .85]},
}

# Table III: real-world, one trajectory, AprilTag reference (no seeds)
T3_STEPS = [0, 67, 135, 202, 269]
T3 = {"trans_V": [11.9, 78.2, 108.2, 95.3, 96.6], "trans_K": [1.4, 17.4, 35.9, 58.1, 85.4],
      "rot_V": [6.7, 57.7, 130.3, 177.4, 94.3], "rot_K": [2.5, 30.8, 63.5, 86.6, 131.5]}

# Fig. 7 점 (눈 판독, 대략적): (위치 cm, 각도 deg)
FIG7 = {"V-only": (8.3, 60), "Naive V+H": (8.3, 48), "FingerMLP": (5.0, 49.2), "KineFuse": (3.6, 47.6), "16-token": (5.8, 69.3)}

ARCH = ["V-only", "Naive V+H", "FingerMLP", "16-token", "KineFuse"]


def welch(m1, s1, m2, s2, n=N_SEEDS, ddof=1):
    """표준편차를 ddof=1 표본 표준편차로 본 Welch t 검정 (ddof=0이면 s*sqrt(n/(n-1))로 보정)."""
    if ddof == 0:
        s1 = s1 * math.sqrt(n / (n - 1))
        s2 = s2 * math.sqrt(n / (n - 1))
    v1, v2 = s1 ** 2 / n, s2 ** 2 / n
    if v1 + v2 == 0:
        return float("nan")
    t = (m1 - m2) / math.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n - 1) + v2 ** 2 / (n - 1))
    return float(2 * stats.t.sf(abs(t), df))


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def main():
    out = {}
    K1, K2 = T1["KineFuse"], T2["KineFuse"]

    # C1 Table I 평균과 본문 수치
    mean_pos = {a: float(np.mean(T1[a]["pos"])) for a in T1}
    mean_ang = {a: float(np.mean(T1[a]["ang"])) for a in T1}
    out["C1_table1_means"] = {"pos_cm": {a: round(v, 3) for a, v in mean_pos.items()},
                              "ang_deg": {a: round(v, 2) for a, v in mean_ang.items()}}
    out["C1_text_4.04_checked"] = round(mean_pos["KineFuse"], 2)
    out["C1_text_8.11_checked"] = round(mean_pos["V-only"], 2)
    out["C1_ratio_Vonly_over_KineFuse_pos"] = round(mean_pos["V-only"] / mean_pos["KineFuse"], 2)
    out["C1_ang_range_KineFuse"] = [min(K1["ang"]), max(K1["ang"])]
    out["C1_ang_range_Vonly"] = [min(T1["V-only"]["ang"]), max(T1["V-only"]["ang"])]
    out["C1_claim_only_constant_ang_model"] = {
        "16-token ang range": [min(T1["16-token"]["ang"]), max(T1["16-token"]["ang"])],
        "Naive ang range": [min(T1["Naive V+H"]["ang"]), max(T1["Naive V+H"]["ang"])],
        "note": "본문 'KineFuse만 각도 오차가 일정'은 16-token(69.2–69.3°)과 Naive(47.3–53.7°)에서도 거의 일정하다"}
    # 'lowest at every occlusion level'
    lowest = []
    for j, o in enumerate(OCC):
        others = {a: T1[a]["pos"][j] for a in T1 if a != "KineFuse"}
        lowest.append({"occ": o, "KineFuse": T1["KineFuse"]["pos"][j], "best_other": min(others, key=others.get),
                       "best_other_val": min(others.values()), "KineFuse_lowest": T1["KineFuse"]["pos"][j] < min(others.values())})
    out["C1_lowest_every_level"] = lowest
    # '격차 확대' 주장: 0% 대비 90%의 차이 변화
    gap = {}
    for a in ["V-only", "Naive V+H", "16-token", "FingerMLP"]:
        d = [round(T1[a]["pos"][j] - T1["KineFuse"]["pos"][j], 2) for j in range(6)]
        gap[a] = {"gap_by_occ": d, "gap_0": d[0], "gap_90": d[-1], "widens_0_to_90": d[-1] > d[0]}
    out["C1_gap_widening_check_pos"] = gap
    # 1-3 occlusion 평균 (Fig. 7 비교용)
    sub = {a: round(float(np.mean(T1[a]["pos"][:3])), 2) for a in T1}
    out["C7_fig7_subset_0_30_pos_mean"] = sub
    out["C7_fig7_all_levels_pos_mean"] = {a: round(v, 2) for a, v in mean_pos.items()}
    out["C7_fig7_reading"] = FIG7
    # 2× 비율 (본문 '2 times')과 Naive 대비
    out["C1_ratio_Naive_over_KineFuse_pos"] = round(mean_pos["Naive V+H"] / mean_pos["KineFuse"], 2)

    # C2 Table II 평균과 15× 정의
    mean_s = {a: float(np.mean(T2[a]["m"])) for a in ARCH}
    out["C2_table2_means"] = {a: round(v, 3) for a, v in mean_s.items()}
    out["C2_ratio_KineFuse_over_16token_avg"] = round(mean_s["KineFuse"] / mean_s["16-token"], 2)
    out["C2_ratio_KineFuse_over_16token_cells"] = [round(K2["m"][j] / T2["16-token"]["m"][j], 2) for j in range(6)]
    out["C2_ratio_KineFuse_over_Vonly_avg"] = round(mean_s["KineFuse"] / mean_s["V-only"], 2)
    out["C2_ratio_best_over_worst_pos_per_cell"] = [
        round(max(T2[a]["m"][j] for a in ARCH) / min(T2[a]["m"][j] for a in ARCH), 1) for j in range(6)]
    out["C2_pct_of_GT_avg"] = {a: round(100 * mean_s[a] / GT, 1) for a in ARCH}
    out["C2_pct_of_GT_cells_KineFuse"] = [round(100 * v / GT, 1) for v in K2["m"]]
    out["C2_9pct_candidates"] = {a: [round(100 * v / GT, 1) for v in T2[a]["m"]] for a in ARCH}
    # 15× 의 로그비 95% 구간 (3 seeds, delta method)
    lr = []
    for j in range(6):
        m1, s1 = K2["m"][j], K2["sd"][j]
        m2, s2 = T2["16-token"]["m"][j], T2["16-token"]["sd"][j]
        se = math.sqrt((s1 / m1) ** 2 / N_SEEDS + (s2 / m2) ** 2 / N_SEEDS)
        lr.append([round(math.exp(math.log(m1 / m2) - 1.96 * se), 1), round(math.exp(math.log(m1 / m2) + 1.96 * se), 1)])
    out["C2_ratio_KineFuse_over_16token_95CI_by_cell"] = lr

    # C3 Welch + Holm: KineFuse vs 각 설계, 셀별 (표 I 위치, 표 II 성공)
    fam = []
    for metric, T, key_m, key_s in [("T1_pos", T1, "pos", "pos_sd"), ("T2_success", T2, "m", "sd")]:
        for a in ["V-only", "Naive V+H", "FingerMLP", "16-token"]:
            for j, o in enumerate(OCC):
                mK = T["KineFuse"][key_m][j]; sK = T["KineFuse"][key_s][j]
                mA = T[a][key_m][j]; sA = T[a][key_s][j]
                fam.append({"metric": metric, "vs": a, "occ": o, "diff_KF_minus_other": round(mK - mA, 3),
                            "p_ddof1": welch(mK, sK, mA, sA, ddof=1), "p_ddof0": welch(mK, sK, mA, sA, ddof=0)})
    p1 = [f["p_ddof1"] for f in fam]
    p0 = [f["p_ddof0"] for f in fam]
    h1 = holm(p1); h0 = holm(p0)
    for f, a1, a0 in zip(fam, h1, h0):
        f["holm_ddof1"] = round(a1, 4); f["holm_ddof0"] = round(a0, 4)
        f["p_ddof1"] = round(f["p_ddof1"], 4); f["p_ddof0"] = round(f["p_ddof0"], 4)
    out["C3_welch_holm_family"] = fam
    out["C3_n_sig_holm_0.05_ddof1"] = {m: sum(1 for f in fam if f["metric"] == m and f["holm_ddof1"] < 0.05) for m in ["T1_pos", "T2_success"]}
    out["C3_n_sig_holm_0.05_ddof0"] = {m: sum(1 for f in fam if f["metric"] == m and f["holm_ddof0"] < 0.05) for m in ["T1_pos", "T2_success"]}
    out["C3_n_tests_per_metric"] = 24

    # C4 16-token의 오차 변동계수: 폐색 수준에 따른 변화가 측정 잡음 수준인가
    def cv(x):
        x = np.asarray(x, float)
        return round(float(x.std(ddof=1) / x.mean()), 4)
    out["C4_cv_across_occlusion"] = {a: {"pos_cv": cv(T1[a]["pos"]), "ang_cv": cv(T1[a]["ang"]),
                                         "success_cv": cv(T2[a]["m"])} for a in ARCH}
    out["C4_16token_pos_range"] = [min(T1["16-token"]["pos"]), max(T1["16-token"]["pos"])]
    out["C4_16token_success_range"] = [min(T2["16-token"]["m"]), max(T2["16-token"]["m"])]

    # C5 각도 오차 순위 vs 성공 순위 (5개 설계, 평균)
    ang_m = [mean_ang["V-only"], mean_ang["Naive V+H"], mean_ang["FingerMLP"], mean_ang["16-token"], mean_ang["KineFuse"]]
    suc_m = [mean_s[a] for a in ARCH]
    rho, p = stats.spearmanr(ang_m, suc_m)
    out["C5_spearman_angle_vs_success"] = {"rho": round(float(rho), 3), "p": round(float(p), 3), "archs": ARCH,
                                           "ang_means": [round(v, 1) for v in ang_m],
                                           "success_means": [round(v, 2) for v in suc_m]}
    pos_m = [mean_pos["V-only"], mean_pos["Naive V+H"], mean_pos["FingerMLP"], mean_pos["16-token"], mean_pos["KineFuse"]]
    rho2, p2 = stats.spearmanr(pos_m, suc_m)
    out["C5_spearman_pos_vs_success"] = {"rho": round(float(rho2), 3), "p": round(float(p2), 3)}
    # 성공과 오차의 관계가 V-only의 폐색 추세와 같은 방향인가 (V-only 성공은 폐색 증가와 함께 상승)
    rho3, p3 = stats.spearmanr(OCC, T2["V-only"]["m"])
    out["C5_Vonly_success_vs_occlusion_spearman"] = {"rho": round(float(rho3), 3), "p": round(float(p3), 3)}

    # C6 Table III: 한 궤적, 회전 오차에서 KineFuse가 뒤집히는 구간
    out["C6_table3_rot_KineFuse_minus_Vonly"] = [round(a - b, 1) for a, b in zip(T3["rot_K"], T3["rot_V"])]
    out["C6_table3_trans_KineFuse_minus_Vonly"] = [round(a - b, 1) for a, b in zip(T3["trans_K"], T3["trans_V"])]
    out["C6_table3_note"] = "단일 궤적, 시드 없음, 기준(AprilTag) 시점 불명확. Fig.10 주석은 AprilTag를 초기 GT 자세로만 쓴다고 함"

    # 2.3× 주장 (영점화 실험, 표 없음): 0–30% 부분집합 비와 대조
    out["C6_2p3x_candidate_subset_ratio_0_30"] = round(sub["V-only"] / sub["KineFuse"], 2)

    path = ROOT / "analysis" / "table_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in list(out)[:12]}, indent=1, ensure_ascii=False)[:3500])


if __name__ == "__main__":
    main()
