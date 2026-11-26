"""ContactGuard(arXiv:2608.13438 v2, v1) 표 값의 재계산과 일관성 검사.

입력: 논문 표 1, 2, 3, 4, 5, 6, 7(v2)과 v1의 표 1, 8을 아래 상수로 옮겨 둠(출처: paper/paper.txt, paper/paper_v1.txt).
출력: analysis/recompute_results.json, 표준 출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 1초 미만)

시행 수는 표 7의 N+/N-(Cup 25/25, Box 22/28, Pencil 22/28, Towel 25/25)와 혼동행렬을 그대로 썼다.
AUC의 표준오차는 점수 원자료가 없어 Hanley-McNeil 근사를 썼다(정규 가정). 점수를 알면 DeLong 검정을 써야 한다.
"""
import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["Cup", "Box", "Pencil", "Towel"]

# 표 7 (v2): task -> method -> (Npos, Nneg, tau, TP, FP, TN, FN, recall, FAR, bacc, prec, auc)
T7 = {
    "Cup": {"Ours": (25, 25, 0.487, 22, 2, 23, 3, 0.880, 0.080, 0.900, 0.917, 0.960),
            "Single view": (25, 25, 0.194, 21, 3, 22, 4, 0.840, 0.120, 0.860, 0.875, 0.930),
            "Direct-linear": (25, 25, 0.301, 17, 4, 21, 8, 0.680, 0.160, 0.760, 0.810, 0.875)},
    "Box": {"Ours": (22, 28, 0.022, 22, 7, 21, 0, 1.000, 0.250, 0.875, 0.759, 0.966),
            "Single view": (22, 28, 0.087, 21, 7, 21, 1, 0.955, 0.250, 0.852, 0.750, 0.925),
            "Direct-linear": (22, 28, 0.018, 21, 11, 17, 1, 0.955, 0.393, 0.781, 0.656, 0.963)},
    "Pencil": {"Ours": (22, 28, 0.519, 19, 3, 25, 3, 0.864, 0.107, 0.878, 0.864, 0.930),
               "Single view": (22, 28, 0.859, 15, 1, 27, 7, 0.682, 0.036, 0.823, 0.938, 0.907),
               "Direct-linear": (22, 28, 0.195, 15, 4, 24, 7, 0.682, 0.143, 0.769, 0.789, 0.833)},
    "Towel": {"Ours": (25, 25, 0.613, 19, 1, 24, 6, 0.760, 0.040, 0.860, 0.950, 0.904),
              "Single view": (25, 25, 0.193, 16, 1, 24, 9, 0.640, 0.040, 0.800, 0.941, 0.890),
              "Direct-linear": (25, 25, 0.051, 13, 1, 24, 12, 0.520, 0.040, 0.740, 0.929, 0.898)},
}
# 표 1 (v2)에서 읽은 값: task -> method -> (prec, rec, far, auc, bacc)
T1 = {
    "Cup": {"Direct-linear": (0.810, 0.680, 0.160, 0.875, 0.760), "Single view": (0.875, 0.840, 0.120, 0.930, 0.860),
            "Ours": (0.917, 0.880, 0.080, 0.960, 0.900)},
    "Box": {"Direct-linear": (0.656, 0.955, 0.393, 0.963, 0.781), "Single view": (0.750, 0.955, 0.250, 0.925, 0.852),
            "Ours": (0.759, 1.000, 0.250, 0.966, 0.875)},
    "Pencil": {"Direct-linear": (0.789, 0.682, 0.143, 0.833, 0.769), "Single view": (0.938, 0.682, 0.036, 0.907, 0.823),
               "Ours": (0.864, 0.864, 0.107, 0.930, 0.878)},
    "Towel": {"Direct-linear": (0.929, 0.520, 0.040, 0.898, 0.740), "Single view": (0.941, 0.640, 0.040, 0.890, 0.800),
              "Ours": (0.950, 0.760, 0.040, 0.904, 0.860)},
}
# 표 2 (v2): detector -> [Cup, Box, Pencil, Towel] AUC
T2 = {
    "FAIL-Detect": [0.840, 0.284, 0.557, 0.794], "RND": [0.875, 0.326, 0.657, 0.763],
    "PCA-kmeans": [0.710, 0.282, 0.744, 0.651], "SAFE": [0.821, 0.899, 0.763, 0.811],
    "Current latent": [0.829, 0.843, 0.833, 0.862], "Direct-linear": [0.875, 0.963, 0.833, 0.898],
    "Single view": [0.930, 0.925, 0.907, 0.890], "ContactGuard": [0.960, 0.966, 0.930, 0.904],
}
# 표 3 (v2): task -> (transplant dp, transplant frac>0, shift dp, hold dp)
T3 = {"Cup": (0.29, 0.84, -0.01, 0.12), "Box": (0.54, 0.79, 0.02, 0.87),
      "Pencil": (0.38, 0.75, 0.00, 0.02), "Towel": (0.20, 0.64, -0.05, 0.11)}
# 표 4 (v2): task -> (w/ state, img only)
T4 = {"Cup": (0.893, 0.960), "Box": (0.906, 0.966), "Pencil": (0.821, 0.930), "Towel": (0.669, 0.904)}
# 표 5 (v2): K -> (horizon s, rollout ms, full ms); v1 표 4
T5 = {5: (0.17, 3.08, 9.02), 15: (0.50, 9.26, 13.24), 25: (0.83, 14.31, 18.71), 30: (1.00, 17.20, 21.48)}
T5_V1 = {5: (0.17, 2.73, 6.53), 15: (0.50, 7.90, 11.56), 25: (0.83, 13.01, 16.63), 30: (1.00, 15.39, 19.18)}
# 표 6 (v2): task -> model -> (auc, bacc)
T6 = {
    "Cup": {"Ours": (0.960, 0.900), "Single view": (0.930, 0.860), "Direct-linear": (0.875, 0.760),
            "Direct-MLP small": (0.866, 0.671), "Direct-MLP large": (0.870, 0.540)},
    "Box": {"Ours": (0.966, 0.875), "Single view": (0.925, 0.852), "Direct-linear": (0.963, 0.781),
            "Direct-MLP small": (0.943, 0.602), "Direct-MLP large": (0.815, 0.626)},
    "Pencil": {"Ours": (0.930, 0.878), "Single view": (0.907, 0.823), "Direct-linear": (0.833, 0.769),
               "Direct-MLP small": (0.844, 0.591), "Direct-MLP large": (0.799, 0.483)},
    "Towel": {"Ours": (0.904, 0.860), "Single view": (0.890, 0.800), "Direct-linear": (0.898, 0.740),
              "Direct-MLP small": (0.787, 0.605), "Direct-MLP large": (0.798, 0.758)},
}
# v1 표 1: task -> method -> (prec, rec, far, auc, bacc); v1 표 8 혼동행렬 (TP, FP, TN, FN)
V1_T1 = {
    "Cup": {"Direct-linear": (0.545, 0.960, 0.800, 0.661, 0.580), "LeWM": (0.706, 0.960, 0.400, 0.928, 0.780),
            "Ours": (0.893, 1.000, 0.120, 0.992, 0.940)},
    "Box": {"Direct-linear": (0.647, 0.500, 0.214, 0.619, 0.643), "LeWM": (0.857, 0.818, 0.107, 0.933, 0.856),
            "Ours": (0.864, 0.864, 0.107, 0.946, 0.878)},
    "Pencil": {"Direct-linear": (0.468, 1.000, 0.893, 0.732, 0.554), "LeWM": (0.679, 0.864, 0.321, 0.872, 0.771),
               "Ours": (0.889, 0.727, 0.071, 0.898, 0.828)},
    "Towel": {"Direct-linear": (0.759, 0.880, 0.280, 0.841, 0.800), "LeWM": (0.750, 0.600, 0.200, 0.794, 0.700),
              "Ours": (0.786, 0.880, 0.240, 0.917, 0.820)},
}
V1_CM = {
    "Cup": {"Ours": (25, 3, 22, 0), "LeWM": (24, 10, 15, 1), "Direct-linear": (24, 20, 5, 1)},
    "Box": {"Ours": (19, 3, 25, 3), "LeWM": (18, 3, 25, 4), "Direct-linear": (11, 6, 22, 11)},
    "Pencil": {"Ours": (16, 2, 26, 6), "LeWM": (19, 9, 19, 3), "Direct-linear": (22, 25, 3, 0)},
    "Towel": {"Ours": (22, 6, 19, 3), "LeWM": (15, 5, 20, 10), "Direct-linear": (22, 7, 18, 3)},
}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def hm_se(auc, npos, nneg):
    """Hanley-McNeil(1982) AUC 표준오차."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    v = (auc * (1 - auc) + (npos - 1) * (q1 - auc * auc) + (nneg - 1) * (q2 - auc * auc)) / (npos * nneg)
    return float(np.sqrt(v))


def r3(x):
    return round(float(x), 3)


def main():
    out = {}

    # V1. 표 7 혼동행렬 -> 표 1/7의 비율 재계산
    rows, bad = [], []
    for t in TASKS:
        for m, (npos, nneg, tau, tp, fp, tn, fn, rec, far, bacc, prec, auc) in T7[t].items():
            calc = dict(rec=tp / (tp + fn), far=fp / (fp + tn), prec=tp / (tp + fp),
                        bacc=0.5 * (tp / (tp + fn) + tn / (fp + tn)))
            ok_counts = (tp + fn == npos) and (fp + tn == nneg)
            diffs = {k: abs(calc[k] - v) for k, v in zip(["rec", "far", "bacc", "prec"], [rec, far, bacc, prec])}
            worst = max(diffs.values())
            rows.append(dict(task=t, method=m, counts_ok=ok_counts, max_abs_diff=r3(worst),
                             recomputed={k: r3(v) for k, v in calc.items()}))
            if (not ok_counts) or worst > 0.0006:
                bad.append((t, m, diffs))
    out["V1_confusion_consistency"] = dict(rows=rows, mismatches=[str(b) for b in bad])
    print("V1 혼동행렬 -> 비율: 불일치", len(bad), "/", len(rows))

    # V2. 표 1(v2)의 값이 표 7과 같은지
    mism = []
    for t in TASKS:
        for m in T1[t]:
            a = T1[t][m]
            b = T7[t][m]
            if (a[0], a[1], a[2], a[3], a[4]) != (b[10], b[7], b[8], b[11], b[9]):
                mism.append((t, m))
    out["V2_table1_vs_table7_mismatch"] = [str(x) for x in mism]
    print("V2 표 1 대 표 7 불일치:", mism)

    # V3. 표 2, 6의 Ours/Single/Direct 값이 표 1과 같은지, 표 4의 img-only가 표 1 Ours와 같은지
    chk = []
    for j, t in enumerate(TASKS):
        chk.append(("T2 ContactGuard", t, T2["ContactGuard"][j] == T1[t]["Ours"][3]))
        chk.append(("T2 Single view", t, T2["Single view"][j] == T1[t]["Single view"][3]))
        chk.append(("T2 Direct-linear", t, T2["Direct-linear"][j] == T1[t]["Direct-linear"][3]))
        chk.append(("T4 img only", t, T4[t][1] == T1[t]["Ours"][3]))
        for m in ["Ours", "Single view", "Direct-linear"]:
            chk.append(("T6 " + m, t, T6[t][m] == (T1[t][m][3], T1[t][m][4])))
    out["V3_cross_table_equalities"] = [dict(item=a, task=b, equal=c) for a, b, c in chk]
    print("V3 표 간 같은 값 불일치:", [c for c in chk if not c[2]])

    # V4. AUC 정수 가능성(U/(n+ n-)), 작동점으로부터의 AUC 하한/상한
    auc_rows = []
    for t in TASKS:
        for m, (npos, nneg, tau, tp, fp, tn, fn, rec, far, bacc, prec, auc) in T7[t].items():
            npairs = npos * nneg
            cand = [u / 2 for u in range(0, 2 * npairs + 1) if round(u / 2 / npairs, 3) == auc]
            lo = (1 + rec - far) / 2  # (0,0)-(far,rec)-(1,1) 꺾은선 아래 넓이 = 균형 정확도
            hi = far * rec + (1 - far)
            auc_rows.append(dict(task=t, method=m, pairs=npairs, auc=auc, u_candidates=cand[:4],
                                 integer_feasible=bool(cand), lower_bound=r3(lo), upper_bound=r3(hi),
                                 within_bounds=bool(lo - 1e-9 <= auc <= hi + 1e-9),
                                 se_hm=r3(hm_se(auc, npos, nneg))))
    out["V4_auc_feasibility"] = auc_rows
    print("V4 AUC 불가능:", [(r["task"], r["method"]) for r in auc_rows if not r["integer_feasible"]],
          " 하한/상한 밖:", [(r["task"], r["method"]) for r in auc_rows if not r["within_bounds"]])

    # V5. Wilson 95% 구간 (재현율, FAR), 균형 정확도는 두 비율의 평균이므로 Wilson 구간의 평균(근사)
    ci = []
    for t in TASKS:
        for m, (npos, nneg, tau, tp, fp, tn, fn, *_) in T7[t].items():
            rl, rh = wilson(tp, npos)
            fl, fh = wilson(fp, nneg)
            ci.append(dict(task=t, method=m, recall=[r3(tp / npos), r3(rl), r3(rh)], far=[r3(fp / nneg), r3(fl), r3(fh)]))
    out["V5_wilson"] = ci

    # V6. 방법 간 AUC 차이: 쌍 환산과 Hanley-McNeil z(독립 가정, 상관을 알 수 없어 보수 쪽)
    diff_rows = []
    for j, t in enumerate(TASKS):
        npos, nneg = T7[t]["Ours"][0], T7[t]["Ours"][1]
        for other in ["Single view", "Direct-linear", "Current latent", "SAFE"]:
            a_ours = T2["ContactGuard"][j]
            a_o = T2[other][j]
            d = a_ours - a_o
            se = np.hypot(hm_se(a_ours, npos, nneg), hm_se(a_o, npos, nneg))
            diff_rows.append(dict(task=t, vs=other, diff=r3(d), pairs_equiv=r3(d * npos * nneg),
                                  z_independent=r3(d / se), p_independent=r3(2 * (1 - stats.norm.cdf(abs(d / se))))))
    out["V6_auc_differences"] = diff_rows
    print("V6 (독립 가정 z):")
    for r in diff_rows:
        print("   ", r["task"], r["vs"], r["diff"], "쌍", r["pairs_equiv"], "z", r["z_independent"])

    # V7. 합산(네 과제 풀링) 재현율/FAR과 Fisher 검정 (방법별 임계값이 다르므로 같은 작동점 비교가 아님)
    pooled = {}
    for m in ["Ours", "Single view", "Direct-linear"]:
        tp = sum(T7[t][m][3] for t in TASKS); fn = sum(T7[t][m][6] for t in TASKS)
        fp = sum(T7[t][m][4] for t in TASKS); tn = sum(T7[t][m][5] for t in TASKS)
        pooled[m] = dict(TP=tp, FN=fn, FP=fp, TN=tn, recall=r3(tp / (tp + fn)), far=r3(fp / (fp + tn)),
                         recall_ci=[r3(x) for x in wilson(tp, tp + fn)], far_ci=[r3(x) for x in wilson(fp, fp + tn)])
    fish = {}
    for m in ["Single view", "Direct-linear"]:
        a, b = pooled["Ours"], pooled[m]
        fish[m] = dict(
            recall_p=r3(stats.fisher_exact([[a["TP"], a["FN"]], [b["TP"], b["FN"]]])[1]),
            far_p=r3(stats.fisher_exact([[a["FP"], a["TN"]], [b["FP"], b["TN"]]])[1]))
    out["V7_pooled"] = dict(pooled=pooled, fisher_two_sided_unpaired=fish)
    print("V7 풀링:", pooled, fish)

    # V8. 평균 AUC와 방향 일치(4/4) 부호 검정
    means = {k: r3(np.mean(v)) for k, v in T2.items()}
    wins = {k: sum(T2["ContactGuard"][j] > T2[k][j] for j in range(4)) for k in T2 if k != "ContactGuard"}
    out["V8_mean_auc"] = dict(mean=means, ours_wins_of_4=wins, sign_test_one_sided_4of4=r3(0.5 ** 4))
    print("V8 평균 AUC", means)

    # V9. 역전 AUC(0.5 미만): Box의 FAIL-Detect, RND, PCA-kmeans
    flip = {k: [r3(1 - x) for x in v] for k, v in T2.items() if min(v) < 0.5}
    out["V9_below_chance_flipped"] = flip
    print("V9 0.5 미만 AUC 뒤집은 값:", flip)

    # V10. 표 3: 분율의 정수 가능성(성공 시행 수 = N-로 가정), 부호 검정
    t3 = []
    for t in TASKS:
        nneg = T7[t]["Ours"][1]
        frac = T3[t][1]
        ks = [k for k in range(nneg + 1) if round(k / nneg, 2) == frac]
        p_one = [float(stats.binomtest(k, nneg, 0.5, alternative="greater").pvalue) for k in ks]
        t3.append(dict(task=t, n_assumed=nneg, frac=frac, k_candidates=ks, feasible=bool(ks),
                       sign_test_p_one_sided=[r3(p) for p in p_one]))
    out["V10_table3_fractions"] = t3
    print("V10 표 3:", t3)

    # V11. 표 4: 상태 포함 대 이미지만 AUC 차이와 쌍 환산
    t4 = []
    for t in TASKS:
        npos, nneg = T7[t]["Ours"][0], T7[t]["Ours"][1]
        d = T4[t][1] - T4[t][0]
        t4.append(dict(task=t, w_state=T4[t][0], img_only=T4[t][1], diff=r3(d), pairs_equiv=r3(d * npos * nneg)))
    out["V11_table4"] = t4
    print("V11 표 4:", t4)

    # V12. 표 5 지연: 전체 - 롤아웃(인코딩+분류기), 스텝당 롤아웃 비용, 선형 적합, 제어 주기 대비
    t5 = {}
    for name, T in [("v2", T5), ("v1", T5_V1)]:
        ks = np.array(sorted(T))
        roll = np.array([T[k][1] for k in ks]); full = np.array([T[k][2] for k in ks])
        slope, icpt = np.polyfit(ks, roll, 1)
        t5[name] = dict(K=ks.tolist(), full_minus_rollout=[r3(x) for x in full - roll],
                        rollout_ms_per_step=[r3(x) for x in roll / ks], fit_slope_ms_per_step=r3(slope),
                        fit_intercept_ms=r3(icpt), horizon_s_check=[r3(k / 30) for k in ks],
                        full_over_period=r3(T[30][2] / (1000 / 30)))
    out["V12_latency"] = t5
    print("V12 지연:", t5)

    # V13. v1 대 v2: 실시간 표 1의 변화
    v = []
    for t in TASKS:
        npos, nneg = T7[t]["Ours"][0], T7[t]["Ours"][1]
        a1, a2 = V1_T1[t]["Ours"][3], T1[t]["Ours"][3]
        d1, d2 = V1_T1[t]["Direct-linear"][3], T1[t]["Direct-linear"][3]
        l1, s2 = V1_T1[t]["LeWM"][3], T1[t]["Single view"][3]
        v.append(dict(task=t, ours_v1=a1, ours_v2=a2, ours_change=r3(a2 - a1),
                      direct_v1=d1, direct_v2=d2, direct_change=r3(d2 - d1),
                      gap_ours_minus_direct_v1=r3(a1 - d1), gap_ours_minus_direct_v2=r3(a2 - d2),
                      single_v1_lewm=l1, single_v2=s2, gap_ours_minus_single_v1=r3(a1 - l1), gap_ours_minus_single_v2=r3(a2 - s2)))
    gaps1 = np.mean([r["gap_ours_minus_direct_v1"] for r in v]); gaps2 = np.mean([r["gap_ours_minus_direct_v2"] for r in v])
    gs1 = np.mean([r["gap_ours_minus_single_v1"] for r in v]); gs2 = np.mean([r["gap_ours_minus_single_v2"] for r in v])
    out["V13_v1_vs_v2"] = dict(rows=v, mean_gap_direct_v1=r3(gaps1), mean_gap_direct_v2=r3(gaps2),
                               mean_gap_single_v1=r3(gs1), mean_gap_single_v2=r3(gs2))
    print("V13 평균 Ours-Direct AUC 간격 v1 %.3f -> v2 %.3f; Ours-Single v1 %.3f -> v2 %.3f" % (gaps1, gaps2, gs1, gs2))
    # v1 표 8 혼동행렬 -> 비율, 클래스 수 동일 여부
    v1chk = []
    for t in TASKS:
        for m, (tp, fp, tn, fn) in V1_CM[t].items():
            p, r_, f, a, b = V1_T1[t][m]
            ok = (abs(tp / (tp + fp) - p) < 6e-4 and abs(tp / (tp + fn) - r_) < 6e-4 and abs(fp / (fp + tn) - f) < 6e-4
                  and abs(0.5 * (tp / (tp + fn) + tn / (fp + tn)) - b) < 6e-4)
            v1chk.append(dict(task=t, method=m, ok=bool(ok), n_pos=tp + fn, n_neg=fp + tn))
    out["V13_v1_confusion_check"] = v1chk
    print("V13 v1 혼동행렬 불일치:", [(x["task"], x["method"]) for x in v1chk if not x["ok"]])
    print("   v1 클래스 수 == v2 클래스 수:", all(
        (x["n_pos"], x["n_neg"]) == (T7[x["task"]]["Ours"][0], T7[x["task"]]["Ours"][1]) for x in v1chk))

    # V14. 같은 라벨 조건의 무정보 기준선: 항상 중단 / 항상 계속
    base = []
    for t in TASKS:
        npos, nneg = T7[t]["Ours"][0], T7[t]["Ours"][1]
        base.append(dict(task=t, failure_rate=r3(npos / (npos + nneg)), always_abort_precision=r3(npos / (npos + nneg)),
                         ours_precision=T7[t]["Ours"][10]))
    out["V14_baseline_rates"] = base

    # V14b. 유병률이 낮아질 때의 정밀도(작동점 재현율/FAR는 그대로라는 가정). 표 7의 실패율은 44-50%
    prev = {}
    for t in TASKS:
        rec, far = T7[t]["Ours"][7], T7[t]["Ours"][8]
        prev[t] = {str(p): r3(p * rec / (p * rec + (1 - p) * far)) for p in [0.5, 0.3, 0.1, 0.05]}
    out["V14b_precision_vs_prevalence"] = prev
    print("V14b 유병률별 정밀도(Ours):", prev)

    # V16. 라이브 N=50에서 검출 가능한 AUC 차이(검정력 0.8, 양측 5%). 상관 r은 같은 시행을 채점하는 두 점수의 상관(미지)
    mde = {}
    for t in TASKS:
        npos, nneg = T7[t]["Ours"][0], T7[t]["Ours"][1]
        se = hm_se(0.93, npos, nneg)
        mde[t] = {"se_auc_at_0.93": r3(se), **{f"mde_r={r}": r3((1.959964 + 0.841621) * se * np.sqrt(2 * (1 - r))) for r in [0.0, 0.5, 0.8]}}
    out["V16_min_detectable_auc_diff"] = mde
    print("V16 최소 검출 AUC 차이:", mde)

    # V15. 임계값 τ 요약(Youden-J, 검증 분할): Box의 Ours와 Direct는 0.02 안팎
    out["V15_thresholds"] = {t: {m: T7[t][m][2] for m in T7[t]} for t in TASKS}

    (ROOT / "analysis" / "recompute_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote analysis/recompute_results.json")


if __name__ == "__main__":
    main()
