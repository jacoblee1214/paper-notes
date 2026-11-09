"""NoContactNoWorries 표 재계산 (CPU, numpy·scipy, 수 초). 실행: python3 analysis/table_check.py

출처: paper/paper.pdf (arXiv:2606.24450v2)
- Table I  (p.5): 시뮬·실물 F1, 7개 물체(정해진 5개 seen + 2개 novel), 설정 9행, mean ± sd, 10 splits
- Table II (p.6): 시각 가림 조건별 F1 (Hexagonal Prism, 시뮬레이션)
- Table III(p.7): CRR(시뮬)·CRA(실물) 평균 ± sd, 10 trials
값은 렌더링한 PDF 페이지 이미지에서 옮겼고, pdftotext 결과와 대조했다.

가정: ± 는 표준편차(sd), n = 10. 민감도 점검으로 ± 를 표준오차(se)로 읽으면 sd = se·√10 으로 바꿔 다시 계산한다.
검정: 두 평균의 Welch t (분산 같다고 가정하지 않음), 여러 칸은 Holm 보정, 칸 방향 일치는 부호 검정(binomial).
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
N = 10
COLS = ["Cuboid", "Pentagonal", "Dodecahedron", "Star", "Stairs", "Hexagonal(novel)", "LetterR(novel)"]
SEEN = [0, 1, 2, 3, 4]
NOVEL = [5, 6]
BASE = ["a", "b", "c", "d", "e", "f", "g"]  # 외부 기준선 (mism은 Full의 교란 변형이라 제외)
NAMES = {
    "a": "(a) Pose-only query-asym", "b": "(b) Pose-only symmetric", "c": "(c) Pose-only MLP",
    "d": "(d) Pose-delta only", "e": "(e) Vision-only", "f": "(f) No temporal", "g": "(g) Kinematic depth",
    "mism": "Full (mismatched command)", "full": "Full (Proposed)",
}

# Table I, 열 순서: Cuboid, Pentagonal, Dodecahedron, Star, Stairs | Hexagonal, Letter R
T1 = {
    "sim": {
        "a": ([.81, .80, .79, .74, .75, .78, .73], [.026, .024, .017, .025, .013, .018, .022]),
        "b": ([.78, .75, .77, .72, .72, .74, .69], [.014, .022, .014, .020, .012, .033, .031]),
        "c": ([.74, .71, .74, .65, .65, .69, .60], [.023, .012, .013, .020, .025, .024, .031]),
        "d": ([.72, .71, .71, .61, .62, .67, .57], [.027, .024, .033, .031, .020, .055, .023]),
        "e": ([.66, .64, .65, .57, .58, .62, .55], [.030, .018, .031, .046, .037, .042, .045]),
        "f": ([.78, .81, .75, .78, .77, .78, .68], [.015, .023, .025, .020, .026, .024, .031]),
        "g": ([.69, .67, .68, .61, .63, .65, .59], [.020, .018, .021, .024, .022, .019, .021]),
        "mism": ([.91, .87, .86, .83, .86, .86, .82], [.012, .018, .020, .015, .028, .021, .028]),
        "full": ([.93, .90, .91, .88, .88, .89, .87], [.010, .013, .015, .012, .025, .026, .026]),
    },
    "real": {
        "a": ([.72, .71, .70, .63, .68, .69, .63], [.029, .034, .017, .024, .023, .027, .016]),
        "b": ([.70, .69, .68, .61, .66, .67, .61], [.027, .023, .015, .031, .020, .027, .028]),
        "c": ([.66, .65, .64, .57, .61, .63, .58], [.026, .031, .032, .017, .030, .034, .028]),
        "d": ([.52, .50, .51, .43, .49, .50, .41], [.023, .026, .031, .028, .024, .029, .019]),
        "e": ([.48, .45, .46, .34, .44, .45, .36], [.028, .026, .032, .018, .026, .022, .018]),
        "f": ([.69, .68, .67, .65, .65, .66, .62], [.024, .034, .035, .034, .015, .024, .021]),
        "g": ([.55, .52, .53, .42, .50, .51, .45], [.024, .021, .027, .019, .023, .022, .018]),
        "mism": ([.83, .80, .80, .68, .77, .74, .69], [.027, .035, .020, .018, .025, .036, .032]),
        "full": ([.84, .83, .82, .71, .79, .80, .74], [.026, .033, .016, .031, .025, .027, .029]),
    },
}

# Table II: F1 조건부 가림 (Hexagonal Prism, 시뮬)  (visible, occluded)
T2 = {"kinematic": (.81, .54), "vision": (.79, .51), "pose_best": (.75, .81), "full": (.93, .85)}
OCC_FRAC = 0.61  # 본문 Sec. V-B(a): 접촉 부위가 가려진 프레임 비율 약 61%

# Table III: 열 순서 CRR, CRA를 물체별로 (Cub, Pen, Dod, Star, Sta, Hex, R)
OBJ3 = ["Cuboid", "Pentagonal", "Dodecahedron", "Star", "Stairs", "Hexagonal(novel)", "LetterR(novel)"]
T3 = {
    "a": [(101.97, 11.02), (15.83, .75), (106.47, 5.85), (17.71, 3.26), (89.36, 16.03), (15.95, .71), (80.64, 7.11), (11.32, 2.46), (80.05, 18.67), (12.12, 2.53), (86.62, 16.47), (11.73, 1.15), (76.67, 3.19), (12.01, 1.39)],
    "b": [(94.49, 7.28), (14.51, 1.82), (113.31, 16.92), (20.03, 2.12), (100.64, 16.19), (15.90, 3.07), (81.16, 8.77), (13.45, 2.57), (83.46, 7.39), (14.48, 3.14), (106.47, 8.19), (18.50, 1.52), (68.20, 5.07), (13.54, .55)],
    "c": [(108.37, 9.71), (17.22, 1.75), (101.39, 11.43), (16.11, 2.68), (99.42, 11.27), (17.42, 1.56), (89.60, 19.96), (15.61, 3.23), (76.49, 11.06), (11.95, .59), (111.50, 14.34), (15.32, 3.08), (68.46, 3.74), (9.49, .86)],
    "d": [(103.32, 8.49), (17.02, 1.57), (84.95, 5.15), (15.09, .43), (88.54, 9.08), (13.97, 2.73), (86.98, 8.28), (14.34, 1.70), (73.47, 6.51), (13.09, 2.90), (94.29, 10.17), (13.50, 1.36), (83.24, 12.95), (13.06, 1.87)],
    "e": [(85.88, 8.64), (15.70, 1.73), (89.70, 9.10), (14.73, 1.85), (85.16, 9.37), (11.62, .64), (88.14, 7.07), (13.28, 1.03), (94.88, 5.49), (15.01, 2.50), (106.12, 11.18), (17.68, 2.61), (76.10, 7.15), (14.96, .76)],
    "f": [(84.34, 9.89), (13.03, .74), (103.79, 9.41), (17.09, 1.32), (109.38, 10.71), (16.50, 1.37), (71.14, 8.90), (12.95, 2.51), (87.31, 4.02), (14.65, .64), (109.51, 11.88), (15.73, 3.31), (86.32, 15.68), (12.76, 1.24)],
    "proprio": [(113.09, 12.24), (19.46, 2.81), (111.84, 15.58), (19.36, 2.46), (113.38, 17.45), (18.95, 4.51), (90.01, 18.10), (13.25, 3.34), (98.02, 15.48), (17.57, 3.92), (117.03, 19.00), (19.68, 3.27), (89.72, 15.98), (15.29, 3.21)],
    "oracle": [(124.43, 5.89), (19.10, 1.06), (124.65, 13.70), (19.56, .92), (121.34, 6.81), (18.43, .62), (97.98, 8.19), (14.95, 1.19), (107.19, 7.95), (16.88, 1.57), (124.18, 7.34), (18.34, 1.28), (99.11, 19.88), (15.39, 1.37)],
    "ours": [(122.99, 11.93), (20.24, 1.46), (116.67, 16.35), (20.65, 2.48), (118.77, 8.49), (19.40, 2.05), (92.77, 11.79), (14.37, 1.83), (107.00, 10.22), (16.83, 2.08), (118.92, 13.31), (17.64, 2.12), (94.80, 18.72), (15.12, 1.76)],
}


def welch_p(m1, s1, m2, s2, n=N):
    return float(stats.ttest_ind_from_stats(m1, s1, n, m2, s2, n, equal_var=False).pvalue)


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    m, adj, run = len(ps), np.zeros(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj.tolist()


def mde(sd, n=N, alpha=0.05, power=0.8):
    """두 독립 평균(n 대 n, 같은 sd)의 Welch 검정에서 80% 검출력의 최소 차이 (df = 2n − 2 근사)"""
    df = 2 * n - 2
    return float((stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)) * np.sqrt(2 * sd ** 2 / n))


def scale(sd, mode):
    # ± 가 표준오차였다면 표준편차는 √n 배
    return sd if mode == "sd" else [s * np.sqrt(N) for s in sd]


def table1(mode):
    out = {}
    for setting in ("sim", "real"):
        M = {k: np.array(v[0]) for k, v in T1[setting].items()}
        S = {k: np.array(scale(v[1], mode)) for k, v in T1[setting].items()}
        full, fs = M["full"], S["full"]
        best_idx = [max(BASE, key=lambda k: M[k][j]) for j in range(7)]
        p_best = [welch_p(full[j], fs[j], M[best_idx[j]][j], S[best_idx[j]][j]) for j in range(7)]
        p_mism = [welch_p(full[j], fs[j], M["mism"][j], S["mism"][j]) for j in range(7)]
        out[setting] = {
            "full_beats_all_baselines_cols": int(sum(full[j] > max(M[k][j] for k in BASE) for j in range(7))),
            "full_beats_mism_cols": int(np.sum(full > M["mism"])),
            "best_baseline_per_col": dict(zip(COLS, best_idx)),
            "full_minus_best_pp": dict(zip(COLS, np.round((full - np.array([M[best_idx[j]][j] for j in range(7)])) * 100, 1).tolist())),
            "p_full_vs_best_welch": dict(zip(COLS, [float(f"{p:.2g}") for p in p_best])),
            "p_full_vs_best_holm": dict(zip(COLS, [float(f"{p:.2g}") for p in holm(p_best)])),
            "full_minus_mism_pp": dict(zip(COLS, np.round((full - M["mism"]) * 100, 1).tolist())),
            "p_full_vs_mism_holm": dict(zip(COLS, [float(f"{p:.2g}") for p in holm(p_mism)])),
            "mism_drop_sign_test_p": float(stats.binomtest(int(np.sum(full > M["mism"])), 7, 0.5).pvalue),
            "mism_drop_pp": dict(zip(COLS, np.round((full - M["mism"]) * 100, 1).tolist())),
            "pose_order_a_gt_b_cols": int(np.sum(M["a"] > M["b"])),
            "pose_order_b_gt_c_cols": int(np.sum(M["b"] > M["c"])),
            "pose_a_gt_c_cols": int(np.sum(M["a"] > M["c"])),
            "seen_mean_full": round(float(full[SEEN].mean()), 3),
            "novel_mean_full": round(float(full[NOVEL].mean()), 3),
        }
        if setting == "real":
            pd_ = [welch_p(M["d"][j], S["d"][j], M["e"][j], S["e"][j]) for j in range(7)]
            out[setting]["pose_delta_minus_vision_pp"] = dict(zip(COLS, np.round((M["d"] - M["e"]) * 100, 1).tolist()))
            out[setting]["p_pose_delta_vs_vision_holm"] = dict(zip(COLS, [float(f"{p:.2g}") for p in holm(pd_)]))
    sim_full, real_full = np.array(T1["sim"]["full"][0]), np.array(T1["real"]["full"][0])
    out["sim_to_real_drop_full_pp"] = dict(zip(COLS, np.round((sim_full - real_full) * 100, 1).tolist()))
    out["claim_exceeds_0.9_convex_sim"] = {"values": T1["sim"]["full"][0][:3], "all_strictly_above_0.9": bool(all(v > .9 for v in T1["sim"]["full"][0][:3]))}
    out["claim_real_convex_0.82_0.84_star_0.71_R_0.74"] = {
        "convex": T1["real"]["full"][0][:3], "star": T1["real"]["full"][0][3], "R": T1["real"]["full"][0][6]}
    return out


def table2():
    # 시각 가림 비율 w로 전체 F1을 근사: w·visible + (1−w)·occluded. 표 I의 Hexagonal Prism(시뮬) 칸과 비교
    hex_col = {"kinematic": T1["sim"]["g"][0][5], "vision": T1["sim"]["e"][0][5], "pose_best": T1["sim"]["a"][0][5], "full": T1["sim"]["full"][0][5]}
    res = {}
    for k, (v, o) in T2.items():
        approx = (1 - OCC_FRAC) * v + OCC_FRAC * o
        res[k] = {"visible": v, "occluded": o, "implied_overall_at_61pct_occluded": round(approx, 3),
                  "table1_hex_sim": hex_col[k], "gap": round(approx - hex_col[k], 3)}
    # 가림 비율이 얼마일 때 Table I 값과 맞는지 (full, vision 두 칸)
    fit = {}
    for k in ("vision", "full"):
        v, o = T2[k]
        w = (v - hex_col[k]) / (v - o) if v != o else float("nan")
        fit[k] = round(float(w), 3)
    res["occluded_fraction_that_matches_table1"] = fit
    return res


def table3(mode):
    def get(k, idx):
        m, s = T3[k][idx]
        return m, (s if mode == "sd" else s * np.sqrt(N))

    out = {}
    # Ours vs Proprio-only, Ours vs Oracle, 각 물체 CRA(실물)·CRR(시뮬)
    for metric, off in (("CRR", 0), ("CRA", 1)):
        rows = {}
        p_vp, p_vo = [], []
        for j in range(7):
            mo, so = get("ours", 2 * j + off)
            mp, sp = get("proprio", 2 * j + off)
            mr, sr = get("oracle", 2 * j + off)
            p_vp.append(welch_p(mo, so, mp, sp))
            p_vo.append(welch_p(mo, so, mr, sr))
        rows["ours_minus_proprio"] = dict(zip(OBJ3, np.round([get("ours", 2 * j + off)[0] - get("proprio", 2 * j + off)[0] for j in range(7)], 2).tolist()))
        rows["p_ours_vs_proprio_holm"] = dict(zip(OBJ3, [float(f"{p:.2g}") for p in holm(p_vp)]))
        rows["ours_minus_oracle"] = dict(zip(OBJ3, np.round([get("ours", 2 * j + off)[0] - get("oracle", 2 * j + off)[0] for j in range(7)], 2).tolist()))
        rows["p_ours_vs_oracle_welch"] = dict(zip(OBJ3, [float(f"{p:.2g}") for p in p_vo]))
        rows["ours_over_oracle_mean_ratio"] = round(float(np.mean([get("ours", 2 * j + off)[0] / get("oracle", 2 * j + off)[0] for j in range(7)])), 3)
        # 논문 주장 "볼록 물체에서 실물 Oracle보다 높다" (CRA, 3개 물체) 점검
        rows["convex_CRA_ours_gt_oracle_p"] = dict(zip(OBJ3[:3], [float(f"{p:.2g}") for p in p_vo[:3]]))  # 칸별 Welch
        # 자세만(query-asym, a)과 MLP(c) 비교
        rows["a_gt_c_cols"] = int(sum(get("a", 2 * j + off)[0] > get("c", 2 * j + off)[0] for j in range(7)))
        rows["a_gt_c_welch_p"] = dict(zip(OBJ3, [float(f"{welch_p(*get('a', 2*j+off), *get('c', 2*j+off)):.2g}") for j in range(7)]))
        # 논문 Table III에서 굵게 표시된 칸(실물 CRA)과 Ours를 넘는 비오라클 최댓값
        best_nonoracle = {}
        for j in range(7):
            cand = {k: get(k, 2 * j + off)[0] for k in ("a", "b", "c", "d", "e", "f", "proprio", "ours")}
            best_nonoracle[OBJ3[j]] = max(cand, key=cand.get)
        rows["best_nonoracle_per_col"] = best_nonoracle
        out[metric] = rows
    # 변형별 평균 (7물체). "시각 전용이 가장 불안정하다"(본문 Sec. V-C) 점검용
    out["variant_means_over_7_objects"] = {
        k: {"CRA": round(float(np.mean([get(k, 2 * j + 1)[0] for j in range(7)])), 2),
            "CRR": round(float(np.mean([get(k, 2 * j)[0] for j in range(7)])), 2)}
        for k in ("a", "b", "c", "d", "e", "f", "proprio", "oracle", "ours")}
    out["note"] = ("CRR(시뮬)와 CRA(실물)는 단위가 다르고 같은 칸에서 두 설정을 함께 재지 않았다. "
                   "따라서 '시뮬이 실물보다 uniformly 높다'는 주장은 이 표만으로 검증할 수 없다.")
    return out


def sign_summary(t3):
    # 방향 일치 칸 수 검정: 자세 query-asym > MLP (CRA, CRR 합쳐 14칸)
    k = t3["CRR"]["a_gt_c_cols"] + t3["CRA"]["a_gt_c_cols"]
    return {"a_gt_c_cols_of_14": k, "binomial_p_vs_half": float(stats.binomtest(k, 14, 0.5).pvalue)}


def main():
    res = {}
    for mode in ("sd", "se"):
        res[f"assume_pm_is_{mode}"] = {"table1": table1(mode), "table3": table3(mode)}
    res["table2"] = table2()
    # 검출력: 칸별 중앙값 sd에서 80% 검출력의 최소 차이
    f1_sd = np.median(np.concatenate([np.array(T1[s][k][1]) for s in ("sim", "real") for k in T1[s]]))
    cra_sd = np.median([T3[k][2 * j + 1][1] for k in T3 for j in range(7)])
    crr_sd = np.median([T3[k][2 * j][1] for k in T3 for j in range(7)])
    res["mde_80pct_power_n10"] = {"F1_median_sd": round(float(f1_sd), 4), "F1_mde": round(mde(f1_sd), 4),
                                  "CRA_median_sd": round(float(cra_sd), 2), "CRA_mde": round(mde(cra_sd), 2),
                                  "CRR_median_sd": round(float(crr_sd), 2), "CRR_mde": round(mde(crr_sd), 2)}
    res["table3_sign_a_vs_c"] = sign_summary(res["assume_pm_is_sd"]["table3"])
    # 민감도: 기본(SD) 가정에서 p<0.05였던 칸이 SE 가정에서 어떻게 바뀌는지
    sd_t3 = res["assume_pm_is_sd"]["table3"]["CRA"]["p_ours_vs_proprio_holm"]
    se_t3 = res["assume_pm_is_se"]["table3"]["CRA"]["p_ours_vs_proprio_holm"]
    res["sensitivity_CRA_ours_vs_proprio_holm"] = {"sd": sd_t3, "se": se_t3}
    (ROOT / "analysis" / "table_check_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({"sd_table1_real": res["assume_pm_is_sd"]["table1"]["real"], "table2": res["table2"]},
                     ensure_ascii=False, indent=1, default=float)[:3000])
    for m in ("CRR", "CRA"):
        print(m, json.dumps(res["assume_pm_is_sd"]["table3"][m], ensure_ascii=False, default=float))
    print("sign a>c", res["table3_sign_a_vs_c"])
    print("sensitivity", json.dumps(res["sensitivity_CRA_ours_vs_proprio_holm"]))
    print("MDE", res["mde_80pct_power_n10"])
    print("T1 sim", json.dumps({k: v for k, v in res["assume_pm_is_sd"]["table1"]["sim"].items() if k in ("full_beats_all_baselines_cols","best_baseline_per_col","p_full_vs_best_holm","full_beats_mism_cols","mism_drop_pp","seen_mean_full","novel_mean_full")}, ensure_ascii=False, default=float))
    print("T1 se", json.dumps({k: res["assume_pm_is_se"]["table1"]["real"][k] for k in ("p_full_vs_best_holm",)}, default=float))


if __name__ == "__main__":
    main()
