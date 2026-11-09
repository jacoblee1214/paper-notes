"""BA-T(arXiv:2606.03287v2) 표 수치를 다시 계산해 본문 주장과 대조한다. 실행: python3 analysis/table_check.py

숫자는 paper/paper.txt(v2)의 표에서 옮겼다. v1과 v2의 표 1·2·4·5·6 숫자는 같다(01 §0).
- A: 표 1 — "두 시점 모델 중 포즈 최고", "5% 크기", ViSTA† 기준선 세기
- B: 표 2 — 반복별 이득, "Corr. 27% vs 44%"
- C: 표 3 — v2에 새로 넣은 사다리 비율(25/15/27%, 76/54/75%, 68/46%)
- D: 표 4·5·6 — 5배·45%·39%, 반복당 시간, 표 사이 불일치, 최대 메모리 ≈ 가중치 크기
- E: 식 11 가중치(α=0.8, K=5)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

T1 = {  # 7Scenes AUC@5/10/20, Chamf, Acc, Comp, δ1.25 | BundleFusion 같은 순서 | decoder M
    "VGGT":   ([0.19, 0.39, 0.60, 0.10, 0.10, 0.07, 0.97], [0.18, 0.36, 0.56, 0.06, 0.07, 0.04, 0.99], 605),
    "MA":     ([0.09, 0.23, 0.44, 0.10, 0.12, 0.07, 0.98], [0.03, 0.11, 0.27, 0.15, 0.19, 0.09, 0.98], 171),
    "DA3":    ([0.22, 0.43, 0.64, 0.08, 0.09, 0.06, 0.98], [0.20, 0.37, 0.57, 0.06, 0.07, 0.04, 0.99], 765),
    "DUSt3R": ([0.09, 0.22, 0.42, 0.13, 0.13, 0.12, 0.81], [0.16, 0.33, 0.52, 0.25, 0.26, 0.22, 0.58], 227),
    "MASt3R": ([0.15, 0.32, 0.52, 0.16, 0.16, 0.11, 0.79], [0.16, 0.32, 0.50, 0.24, 0.26, 0.22, 0.58], 227),
    "ViSTA":  ([0.13, 0.33, 0.56, 0.10, 0.10, 0.10, 0.98], [0.15, 0.31, 0.51, 0.06, 0.06, 0.05, 0.99], 113),
    "ViSTA†": ([0.01, 0.06, 0.17, 0.15, 0.12, 0.15, 0.94], [0.02, 0.07, 0.17, 0.09, 0.09, 0.09, 0.98], 38),
    "BA-T":   ([0.16, 0.37, 0.59, 0.11, 0.10, 0.11, 0.98], [0.16, 0.34, 0.54, 0.07, 0.07, 0.06, 0.99], 38),
}
TWO_VIEW = ["DUSt3R", "MASt3R", "ViSTA", "ViSTA†", "BA-T"]
MULTI = ["VGGT", "MA", "DA3"]
COLS = ["AUC@5", "AUC@10", "AUC@20", "Chamf", "Acc", "Comp", "d1.25"]
HIGHER = [True, True, True, False, False, False, True]

# 표 2·3: Rot AUC @5/10/20, Trans AUC @0.05/0.10/0.20, Corr, Rel, δ1.05, δ1.25
T2_VISTA_ITER = {1: [0.442, 0.676, 0.830, 0.101, 0.270, 0.505, 0.062, 0.061, 0.497, 0.967],
                 2: [0.472, 0.699, 0.844, 0.133, 0.322, 0.555, 0.048, 0.060, 0.513, 0.968],
                 3: [0.485, 0.708, 0.848, 0.146, 0.341, 0.575, 0.045, 0.060, 0.515, 0.968],
                 4: [0.491, 0.712, 0.851, 0.151, 0.346, 0.580, 0.045, 0.060, 0.516, 0.968]}
T2_BAT = {1: [0.650, 0.819, 0.908, 0.322, 0.555, 0.749, 0.046, 0.051, 0.569, 0.972],
          2: [0.671, 0.830, 0.914, 0.401, 0.624, 0.794, 0.032, 0.050, 0.585, 0.975],
          3: [0.679, 0.834, 0.916, 0.421, 0.642, 0.805, 0.028, 0.049, 0.594, 0.976],
          4: [0.683, 0.836, 0.917, 0.422, 0.643, 0.805, 0.026, 0.048, 0.601, 0.976]}
T3 = {"a": [0.426, 0.662, 0.823, 0.104, 0.272, 0.510, 0.052, 0.079, 0.380, 0.940],
      "b": [0.491, 0.712, 0.851, 0.151, 0.346, 0.580, 0.045, 0.060, 0.516, 0.968],
      "h": [0.637, 0.810, 0.904, 0.296, 0.534, 0.735, 0.039, 0.051, 0.565, 0.971],
      "g": [0.683, 0.836, 0.917, 0.422, 0.643, 0.805, 0.026, 0.048, 0.601, 0.976],
      "c": [0.665, 0.827, 0.912, 0.385, 0.613, 0.787, 0.027, 0.050, 0.599, 0.974],
      "d": [0.667, 0.827, 0.912, 0.404, 0.625, 0.793, 0.036, 0.049, 0.620, 0.974],
      "e": [0.677, 0.833, 0.916, 0.412, 0.635, 0.802, 0.026, 0.048, 0.591, 0.976]}
I_ROT, I_TRA, I_CORR, I_REL, I_D105 = 0, 3, 6, 7, 8

T4 = {"Droid-SLAM": {"rot5": 0.53, "tra05": 0.08, "rel": 1.19, "d125": 0.36, "ms": 160.93, "gb": 2.43},
      "BA-T": {"rot5": 0.68, "tra05": 0.42, "rel": 0.20, "d125": 0.98, "ms": 30.92, "gb": 1.35}}
T5 = {"ms": {"Droid": 160.93, "VGGT": 128.71, "DA3": 78.14, "MapAny": 41.13, "MASt3R": 93.25,
             "DUSt3R": 66.47, "ViSTA": 37.96, "BA-T k1": 21.40, "BA-T k2": 24.65, "BA-T k3": 28.32,
             "BA-T k4": 30.92},
      "gb": {"Droid": 2.43, "VGGT": 4.93, "DA3": 6.52, "MapAny": 3.23, "MASt3R": 2.72, "DUSt3R": 2.02,
             "ViSTA": 1.76, "BA-T": 1.32}}
T6 = {"MapAny": {"trans": 0.14, "rot": 29.65, "rel": 0.36, "d125": 0.30, "ms": 62.95, "gb": 3.42},
      "VGGT": {"trans": 0.06, "rot": 10.72, "rel": 0.15, "d125": 0.79, "ms": 230.13, "gb": 5.11},
      "BA-T": {"trans": 0.09, "rot": 16.39, "rel": 0.20, "d125": 0.70, "ms": 50.62, "gb": 1.35}}


def r(x, n=3):
    return round(float(x), n)


def frac(a, b, g, i):
    """(a→b) 이득이 (a→g) 전체 이득에서 차지하는 비율."""
    return r((b[i] - a[i]) / (g[i] - a[i]), 3)


def part_a():
    out = {}
    for ds, k in (("7Scenes", 0), ("BundleFusion", 1)):
        best = {}
        for c, (name, hi) in enumerate(zip(COLS, HIGHER)):
            vals = {m: T1[m][k][c] for m in TWO_VIEW}
            top = max(vals.values()) if hi else min(vals.values())
            best[name] = [m for m, v in vals.items() if v == top]
        multi_better = {name: [m for m in MULTI if (T1[m][k][c] > T1["BA-T"][k][c]) == hi
                               and T1[m][k][c] != T1["BA-T"][k][c]]
                        for c, (name, hi) in enumerate(zip(COLS, HIGHER))}
        out[ds] = {"two_view_best_per_metric": best, "multi_view_models_better_than_BA-T": multi_better}
    out["size_ratio_BA-T_over"] = {m: r(38 / T1[m][2]) for m in MULTI + ["DUSt3R", "ViSTA"]}
    out["ViSTA†_vs_ViSTA_AUC@5_7Scenes"] = {"ViSTA†": 0.01, "ViSTA": 0.13, "ratio": r(0.13 / 0.01, 1)}
    out["ViSTA†_rot_only_AUC@5_table3a"] = 0.426
    out["note"] = ("표 1의 포즈 AUC는 max(회전, 이동 방향 각) 오차 기준, 표 2·3은 회전·이동(m) 따로. "
                   "같은 ViSTA†가 표 1 AUC@5=0.01, 표 3 (a) 회전 AUC@5=0.426 — 이동 방향 오차가 지배한다는 뜻")
    return out


def part_b():
    out = {}
    for name, d in (("ViSTA†_iter", T2_VISTA_ITER), ("BA-T", T2_BAT)):
        out[name] = {"rot5_gain_1to4": r(d[4][I_ROT] - d[1][I_ROT]),
                     "tra05_gain_1to4": r(d[4][I_TRA] - d[1][I_TRA]),
                     "corr_reduction_1to4_pct": r(100 * (1 - d[4][I_CORR] / d[1][I_CORR]), 1),
                     "rel_reduction_1to4_pct": r(100 * (1 - d[4][I_REL] / d[1][I_REL]), 1)}
    out["claim"] = "본문 4.2: ViSTA† Corr. 27% 감소, BA-T 44% 감소"
    out["BA-T_iter1_vs_ViSTA†_iter4"] = {"rot5": [T2_BAT[1][I_ROT], T2_VISTA_ITER[4][I_ROT]],
                                       "tra05": [T2_BAT[1][I_TRA], T2_VISTA_ITER[4][I_TRA]]}
    gains = [r(T2_BAT[k + 1][I_TRA] - T2_BAT[k][I_TRA]) for k in (1, 2, 3)]
    out["BA-T_tra05_gain_per_step"] = gains
    out["BA-T_share_of_iter4_gap_over_ViSTA†iter4_reached_at_iter1"] = {
        "rot5": r((T2_BAT[1][I_ROT] - T2_VISTA_ITER[4][I_ROT]) / (T2_BAT[4][I_ROT] - T2_VISTA_ITER[4][I_ROT])),
        "tra05": r((T2_BAT[1][I_TRA] - T2_VISTA_ITER[4][I_TRA]) / (T2_BAT[4][I_TRA] - T2_VISTA_ITER[4][I_TRA]))}
    out["table3_a_b_equal_table2"] = {"a==ViSTA†(no iter)": True,
                                      "b==ViSTA†_iter4": T3["b"] == T2_VISTA_ITER[4],
                                      "g==BA-T_iter4": T3["g"] == T2_BAT[4]}
    return out


def part_c():
    a, b, h, g = T3["a"], T3["b"], T3["h"], T3["g"]
    out = {
        "(a→b)/(a→g)": {"rot5": frac(a, b, g, I_ROT), "tra05": frac(a, b, g, I_TRA), "corr": frac(a, b, g, I_CORR),
                        "claim": "25% / 15% / 27%"},
        "(b→h)/(b→g)": {"rot5": frac(b, h, g, I_ROT), "tra05": frac(b, h, g, I_TRA), "rel": frac(b, h, g, I_REL),
                        "claim": "76% / 54% / 75%"},
        "(h→g)/(b→g)": {"corr": r(1 - frac(b, h, g, I_CORR)), "tra05": r(1 - frac(b, h, g, I_TRA)),
                        "claim": "Corr 68%, 이동 46%"},
    }
    # 어떤 지표를 고르느냐에 따라 비율이 크게 달라지는지
    names = ["rot5", "rot10", "rot20", "tra05", "tra10", "tra20", "corr", "rel", "d105", "d125"]
    out["(b→h)/(b→g)_all_metrics"] = {n: frac(b, h, g, i) for i, n in enumerate(names)}
    out["leave_one_out_drop_rot5_tra05"] = {k: [r(g[I_ROT] - T3[k][I_ROT]), r(g[I_TRA] - T3[k][I_TRA])]
                                           for k in ("c", "d", "e")}
    out["(d)_d1.05_higher_than_full"] = [T3["d"][I_D105], g[I_D105]]
    return out


def part_d():
    out = {}
    out["T4_speedup"] = r(T4["Droid-SLAM"]["ms"] / T4["BA-T"]["ms"], 2)
    out["T4_mem_reduction_pct"] = r(100 * (1 - T4["BA-T"]["gb"] / T4["Droid-SLAM"]["gb"]), 1)
    out["T4_vs_T2_consistency(BA-T K=4, 7Scenes)"] = {
        "rot5": [T4["BA-T"]["rot5"], T2_BAT[4][I_ROT]], "tra05": [T4["BA-T"]["tra05"], T2_BAT[4][I_TRA]],
        "d1.25": [T4["BA-T"]["d125"], T2_BAT[4][9]], "rel": [T4["BA-T"]["rel"], T2_BAT[4][I_REL]],
        "rel_ratio": r(T4["BA-T"]["rel"] / T2_BAT[4][I_REL], 2)}
    ks = [1, 2, 3, 4]
    ts = [T5["ms"][f"BA-T k{k}"] for k in ks]
    n = len(ks); mx = sum(ks) / n; my = sum(ts) / n
    slope = sum((k - mx) * (t - my) for k, t in zip(ks, ts)) / sum((k - mx) ** 2 for k in ks)
    out["T5_time_fit_ms"] = {"per_iter": r(slope, 2), "intercept(encoder+heads)": r(my - slope * mx, 2),
                             "diffs": [r(ts[i + 1] - ts[i], 2) for i in range(3)],
                             "K8_extrapolated": r(my - slope * mx + 8 * slope, 1),
                             "README_RTX4070_K8": "약 50 ms (GPU가 다름)"}
    out["T5_BA-T_k4_vs_fastest_other"] = {"BA-T k4": 30.92, "ViSTA": 37.96, "MapAny": 41.13}
    out["mem_values_BA-T"] = {"T4": 1.35, "T5": 1.32, "T6(4view)": 1.35}
    params = 344.4e6                              # analysis/param_check.py 전체 파라미터
    w_gib = params * 4 / 2 ** 30
    out["weights_fp32"] = {"GB": r(params * 4 / 1e9, 3), "GiB": r(w_gib, 3),
                           "T5_peak_minus_weights_if_GiB": r(1.32 - w_gib, 3),
                           "note": "최대 메모리 1.32가 GiB라면 활성값은 약 0.04 GiB — 메모리 비교는 거의 가중치 크기 비교"}
    out["T6_better_count"] = {
        "VGGT_better_than_BA-T": [k for k in ("trans", "rot", "rel") if T6["VGGT"][k] < T6["BA-T"][k]]
        + (["d125"] if T6["VGGT"]["d125"] > T6["BA-T"]["d125"] else []),
        "BA-T_better_than_MapAny": [k for k in ("trans", "rot", "rel") if T6["BA-T"][k] < T6["MapAny"][k]]
        + (["d125"] if T6["BA-T"]["d125"] > T6["MapAny"]["d125"] else [])}
    out["T6_mem_ratio_vs_MapAny"] = r(T6["BA-T"]["gb"] / T6["MapAny"]["gb"], 3)
    out["T6_time_ratio_vs_VGGT"] = r(T6["VGGT"]["ms"] / T6["BA-T"]["ms"], 2)
    return out


def part_e():
    a, K = 0.8, 5
    w = [a ** (K - k) for k in range(1, K + 1)]
    s = sum(w)
    return {"weights_k1..k5": [r(x, 4) for x in w], "sum(=ViSTA† loss 배수)": r(s, 4),
            "share_k1..k5": [r(x / s, 3) for x in w]}


def main():
    out = {"A_table1": part_a(), "B_table2": part_b(), "C_table3": part_c(), "D_tables456": part_d(),
           "E_eq11_weights": part_e()}
    path = ROOT / "analysis" / "table_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
