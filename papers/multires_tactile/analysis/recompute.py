"""MiTaS(arXiv:2606.06281 v2)의 표와 본문 수치를 다시 계산하고 일관성을 확인한다.

입력은 논문 표 8, 9, 10, 5와 그림 4의 막대 라벨을 옮긴 값이다(코드 안의 딕셔너리). 외부 파일은 읽지 않는다.
출력: analysis/recompute_results.json (수 초, numpy와 scipy만 사용)

점검 항목
  V1 표 8(성공 수)과 표 9, 10(퍼센트)의 일치, 표 평균, 표 1–3이 어느 데이터셋 값인지
  V2 그림 4 막대 라벨과 본문 평균(80, 55.5, 35.5, 34.5), 초록 수치
  V3 Wilson 95% 구간
  V4 주 비교 검정(MiTaS 대 Sparsh-X, Q-Former, PolyTouch, 시각 전용): Fisher, 층화(CMH), 정책 단위 쌍 검정
  V5 센서 절제(표 2, 데이터셋 1)
  V6 멀티 촉각 드롭아웃(표 3, 데이터셋 1)과 데이터셋 2의 추가 값
  V7 두 데이터셋 사이의 같은 모델·과제 값 차이(데이터셋 이질성)
  V8 표 5 산술(총합, 토큰 수, 가산성, Hz 대 ms)
  K1 토큰 수 640 대 644, K2 Evetac 프레임 수 대 샘플링 창, K3 DiT 머리 파라미터 추정
  K4 시행 수 3가지 가정(20, 40, 정책 단위)에서의 검출력과 훈련 실행 변동 토이
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
TASKS = ["Gear", "Board", "Lamp", "Key", "Lightbulb"]

# 표 9 (첫 번째 데이터셋, 시행 20, %). 표 2, 표 3, 그림 4의 일부가 여기서 나옴
D1 = {
    "ViT-CNN V": [40, 70, 20, 0, 0],
    "ViT V": [40, 60, 55, 0, 0],
    "MiTaS V+E": [0, 80, 65, 0, 40],
    "Sparsh V+G": [25, 50, 55, 20, 15],
    "MiTaS V+G": [45, 70, 65, 55, 20],
    "Sparsh V+G MTD": [15, 85, 70, 5, 10],
    "MiTaS V+G MTD": [55, 85, 60, 20, 45],
    "Sparsh V+G+E": [50, 85, 70, 10, 55],
    "MiTaS V+G+E": [90, 90, 80, 75, 65],
}
# 표 10 (두 번째 데이터셋, 시행 20, %). 표 1이 여기서 나옴
D2 = {
    "ViT-CNN V": [35, 70, 45, 35, 30],
    "ViT V": [35, 60, 40, 45, 20],
    "MiTaS V+G MTD": [55, 80, 55, 65, 45],
    "MiTaS V+G+E MTD": [60, 85, 65, 55, 50],
    "Q-Former V+G+E": [50, 75, 75, 60, 55],
    "Poly V+G+E": [55, 70, 55, 55, 55],
    "Sparsh V+G+E": [40, 75, 65, 60, 45],
    "MiTaS V+G+E": [80, 100, 85, 75, 60],
}
# 표 8 (성공 수 / 시행, †는 40). 키는 (모델, 두 데이터셋 풀링 여부)
T8 = {
    "ViT-CNN V": ([15, 28, 13, 7, 6], 40, 34.5),
    "ViT V": ([15, 24, 19, 9, 4], 40, 35.5),
    "MiTaS V+E": ([0, 16, 13, 0, 8], 20, 37.0),
    "Sparsh V+G": ([5, 10, 11, 4, 3], 20, 33.0),
    "MiTaS V+G": ([9, 14, 13, 11, 4], 20, 51.0),
    "Sparsh V+G MTD": ([3, 17, 14, 1, 2], 20, 37.0),
    "MiTaS V+G MTD": ([22, 33, 23, 17, 18], 40, 56.5),
    "MiTaS V+G+E MTD": ([12, 17, 13, 11, 10], 20, 63.0),
    "Q-Former V+G+E": ([10, 15, 15, 12, 11], 20, 63.0),
    "Poly V+G+E": ([11, 14, 11, 11, 11], 20, 58.0),
    "Sparsh V+G+E": ([18, 32, 27, 14, 20], 40, 55.5),
    "MiTaS V+G+E": ([34, 38, 33, 30, 25], 40, 80.0),
}
T8_TOTALS = {"ViT-CNN V": 69, "ViT V": 71, "MiTaS V+E": 37, "Sparsh V+G": 33, "MiTaS V+G": 51, "Sparsh V+G MTD": 37,
             "MiTaS V+G MTD": 113, "MiTaS V+G+E MTD": 63, "Q-Former V+G+E": 63, "Poly V+G+E": 58,
             "Sparsh V+G+E": 111, "MiTaS V+G+E": 160}
# 그림 4 막대 라벨(그림 판독, 반올림된 값)
FIG4 = {
    "MiTaS V+G+E": [85, 95, 83, 75, 63],
    "Sparsh V+G+E": [45, 80, 68, 35, 50],
    "ViT-CNN V": [38, 70, 33, 18, 15],
    "ViT V": [38, 60, 48, 23, 10],
}
# 표 5
T5 = [  # 모델, 센서, 인코더 파라미터, 총 파라미터, 토큰/관측, ms, Hz
    ("ViT-CNN", "V", 3079232, 16780740, 260, 35.5, 227.4),
    ("ViT", "V", 4051328, 17752836, 260, 34.4, 232.2),
    ("Sparsh", "V+G", 9398744, 23100252, 448, 39.2, 203.5),
    ("Sparsh MTD", "V+G", 15500056, 29201564, 448, 38.9, 208.4),
    ("Sparsh", "V+G+E", 15500056, 29201564, 640, 54.6, 153.4),
    ("MiTaS", "V+E", 8769280, 22470788, 452, 35.6, 223.7),
    ("MiTaS", "V+G", 5074776, 18776284, 452, 35.3, 224.1),
    ("MiTaS MTD", "V+G", 10764824, 24466332, 452, 36.1, 230.1),
    ("MiTaS", "V+G+E", 10764824, 24466332, 644, 43.5, 180.5),
]
HEAD = 13701508


def half_up(x):
    return int(math.floor(x + 0.5))


def k_of(pct, n=20):
    return [round(p * n / 100) for p in pct]


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def cmh(tables):
    """층화 2x2 표의 Mantel-Haenszel 카이제곱(연속성 보정 없음)과 공통 오즈비. tables: [(a,b,c,d)]"""
    num, var, or_n, or_d = 0.0, 0.0, 0.0, 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        r1, c1 = a + b, a + c
        e = r1 * c1 / n
        v = r1 * (c + d) * c1 * (b + d) / (n * n * (n - 1))
        num += a - e
        var += v
        or_n += a * d / n
        or_d += b * c / n
    chi = num * num / var
    return float(chi), float(stats.chi2.sf(chi, 1)), float(or_n / or_d)


def cells(model, ds):
    return k_of(ds[model])


def main():
    R = {}

    # ---------- V1 표 8 대 표 9, 10
    v1 = {}
    mism = []
    for name, (ks, n, avg) in T8.items():
        k1 = cells(name, D1) if name in D1 else None
        k2 = cells(name, D2) if name in D2 else None
        if n == 40:
            calc = [a + b for a, b in zip(k1, k2)]
        elif k1 is not None and k2 is None:
            calc = k1
        else:
            calc = k2
        ok = calc == ks
        tot = sum(ks)
        total_ok = tot == T8_TOTALS[name]
        denom = n * 5
        avg_calc = 100 * tot / denom
        v1[name] = {"table8": ks, "recomputed": calc, "match": ok, "total": tot, "total_matches": total_ok,
                    "avg_pct": round(avg_calc, 2), "avg_matches": abs(avg_calc - avg) < 0.05}
        if not (ok and total_ok and abs(avg_calc - avg) < 0.05):
            mism.append(name)
    v1["trials_total"] = sum(n * 5 for (_, n, _) in T8.values())  # 1700
    v1["mismatches"] = mism
    # 표 9, 10 행 평균
    avgs = {}
    for tag, ds in (("D1", D1), ("D2", D2)):
        for m, row in ds.items():
            avgs[f"{tag} {m}"] = {"mean": sum(row) / 5, "integer_k": all(p % 5 == 0 for p in row)}
    v1["row_means"] = avgs
    R["V1"] = v1

    # ---------- V2 그림 4 라벨과 본문 평균
    v2 = {}
    for name, labels in FIG4.items():
        k = T8[name][0]
        exact = [100 * x / 40 for x in k]
        rounded = [half_up(x) for x in exact]
        v2[name] = {"exact": exact, "half_up": rounded, "figure_labels": labels, "match": rounded == labels,
                    "avg_exact": sum(exact) / 5}
    v2["text"] = {"MiTaS 80": T8_TOTALS["MiTaS V+G+E"] / 200 * 100, "Sparsh-X 55.5": T8_TOTALS["Sparsh V+G+E"] / 200 * 100,
                  "ViT 35.5": T8_TOTALS["ViT V"] / 200 * 100, "ViT-CNN 34.5": T8_TOTALS["ViT-CNN V"] / 200 * 100}
    R["V2"] = v2

    # ---------- V3 Wilson
    v3 = {"pooled_n40": {}, "single_n20": {}, "totals": {}}
    for name in ("MiTaS V+G+E", "Sparsh V+G+E", "ViT-CNN V", "ViT V"):
        v3["pooled_n40"][name] = {t: [round(100 * x, 1) for x in wilson(k, 40)] for t, k in zip(TASKS, T8[name][0])}
    for name in ("MiTaS V+G+E", "Sparsh V+G+E", "Q-Former V+G+E", "Poly V+G+E"):
        ds = D2 if name in D2 else D1
        v3["single_n20"][name] = {t: [round(100 * x, 1) for x in wilson(k, 20)] for t, k in zip(TASKS, cells(name, ds))}
    for name, (ks, n, _) in T8.items():
        tot, nn = sum(ks), n * 5
        v3["totals"][name] = {"k": tot, "n": nn, "pct": round(100 * tot / nn, 1),
                              "wilson95": [round(100 * x, 1) for x in wilson(tot, nn)]}
    R["V3"] = v3

    # ---------- V4 주 비교
    v4 = {}
    m, s = T8["MiTaS V+G+E"][0], T8["Sparsh V+G+E"][0]
    v4["MiTaS_vs_SparshX_pooled_per_task_p"] = {t: fisher(a, 40, b, 40) for t, a, b in zip(TASKS, m, s)}
    v4["MiTaS_vs_SparshX_pooled_total"] = {"k": [sum(m), sum(s)], "n": 200, "p": fisher(sum(m), 200, sum(s), 200),
                                           "diff_pts": 100 * (sum(m) - sum(s)) / 200}
    tabs = []
    for ds in (D1, D2):
        a, b = cells("MiTaS V+G+E", ds), cells("Sparsh V+G+E", ds)
        tabs += [(x, 20 - x, y, 20 - y) for x, y in zip(a, b)]
    chi, p, orr = cmh(tabs)
    v4["MiTaS_vs_SparshX_CMH_10strata"] = {"chi2": chi, "p": p, "common_odds_ratio": orr}
    for tag, ds in (("D1", D1), ("D2", D2)):
        a, b = sum(cells("MiTaS V+G+E", ds)), sum(cells("Sparsh V+G+E", ds))
        v4[f"MiTaS_vs_SparshX_{tag}_total"] = {"k": [a, b], "n": 100, "p": fisher(a, 100, b, 100)}
        a_, b_ = cells("MiTaS V+G+E", ds), cells("Sparsh V+G+E", ds)
        v4[f"MiTaS_vs_SparshX_{tag}_per_task_p"] = {t: fisher(x, 20, y, 20) for t, x, y in zip(TASKS, a_, b_)}
    # 셀 단위(정책 단위): 10개 (과제, 데이터셋) 쌍
    diffs = []
    for ds in (D1, D2):
        diffs += [x - y for x, y in zip(ds["MiTaS V+G+E"], ds["Sparsh V+G+E"])]
    v4["cell_level_10"] = {"diffs_pts": diffs, "mean": float(np.mean(diffs)), "sd": float(np.std(diffs, ddof=1)),
                           "all_positive": all(d > 0 for d in diffs),
                           "paired_t_p": float(stats.ttest_1samp(diffs, 0).pvalue),
                           "wilcoxon_p": float(stats.wilcoxon(diffs).pvalue),
                           "sign_test_p": 2 * 0.5 ** 10}
    # 데이터셋 2에서 같은 데이터로 학습된 융합 구조 비교
    mi = cells("MiTaS V+G+E", D2)
    v4["D2_MiTaS_vs_others_total_p"] = {}
    for other in ("Q-Former V+G+E", "Poly V+G+E", "Sparsh V+G+E", "MiTaS V+G+E MTD"):
        o = cells(other, D2)
        v4["D2_MiTaS_vs_others_total_p"][other] = {"k": [sum(mi), sum(o)], "p": fisher(sum(mi), 100, sum(o), 100)}
        v4["D2_MiTaS_vs_others_total_p"][other]["per_task_p"] = {t: fisher(x, 20, y, 20) for t, x, y in zip(TASKS, mi, o)}
    # 융합 구조 4개의 평균 사이 (표 1): 시각촉각 기준선끼리
    qf, po, sp = (sum(cells(n, D2)) for n in ("Q-Former V+G+E", "Poly V+G+E", "Sparsh V+G+E"))
    v4["D2_baselines_pairwise_p"] = {"QF_vs_Sparsh": fisher(qf, 100, sp, 100), "Poly_vs_Sparsh": fisher(po, 100, sp, 100)}
    # 시각 전용 대 MiTaS
    for name in ("ViT-CNN V", "ViT V"):
        v4[f"MiTaS_vs_{name}_pooled"] = {"k": [sum(m), sum(T8[name][0])], "p": fisher(sum(m), 200, sum(T8[name][0]), 200)}
    # 초록의 "시각촉각 기준선 55.5%, 63%"는 모집단이 다르다
    v4["abstract_visuotactile_baselines"] = {"Sparsh-X pooled n=200": 100 * sum(s) / 200, "Q-Former D2 n=100": float(sum(cells("Q-Former V+G+E", D2))),
                                            "Sparsh-X on D2 n=100": float(sum(cells("Sparsh V+G+E", D2))),
                                            "MiTaS on D2": float(sum(mi))}
    R["V4"] = v4

    # ---------- V5 센서 절제 (D1)
    v5 = {}
    full, vg, ve = (cells(n, D1) for n in ("MiTaS V+G+E", "MiTaS V+G", "MiTaS V+E"))
    vis = cells("ViT-CNN V", D1)
    v5["full_vs_VG"] = {"per_task_p": {t: fisher(a, 20, b, 20) for t, a, b in zip(TASKS, full, vg)},
                        "total": {"k": [sum(full), sum(vg)], "p": fisher(sum(full), 100, sum(vg), 100)}}
    v5["full_vs_VE"] = {"per_task_p": {t: fisher(a, 20, b, 20) for t, a, b in zip(TASKS, full, ve)},
                        "total": {"k": [sum(full), sum(ve)], "p": fisher(sum(full), 100, sum(ve), 100)}}
    v5["VG_vs_vision_only_ViTCNN"] = {"per_task_p": {t: fisher(a, 20, b, 20) for t, a, b in zip(TASKS, vg, vis)},
                                      "total": {"k": [sum(vg), sum(vis)], "p": fisher(sum(vg), 100, sum(vis), 100)}}
    v5["VE_vs_vision_only_ViTCNN"] = {"per_task_p": {t: fisher(a, 20, b, 20) for t, a, b in zip(TASKS, ve, vis)},
                                      "total": {"k": [sum(ve), sum(vis)], "p": fisher(sum(ve), 100, sum(vis), 100)}}
    v5["VG_vs_VE"] = {"total": {"k": [sum(vg), sum(ve)], "p": fisher(sum(vg), 100, sum(ve), 100)}}
    v5["tasks_VG_at_least_50pct"] = sum(1 for x in D1["MiTaS V+G"] if x >= 50)
    # Sparsh-X: V+G+E 대 V+G
    sf, sg = cells("Sparsh V+G+E", D1), cells("Sparsh V+G", D1)
    v5["Sparsh_full_vs_VG"] = {"total": {"k": [sum(sf), sum(sg)], "p": fisher(sum(sf), 100, sum(sg), 100)},
                               "per_task_p": {t: fisher(a, 20, b, 20) for t, a, b in zip(TASKS, sf, sg)}}
    R["V5"] = v5

    # ---------- V6 MTD
    v6 = {}
    for fam, base, mtd in (("MiTaS", "MiTaS V+G", "MiTaS V+G MTD"), ("Sparsh", "Sparsh V+G", "Sparsh V+G MTD")):
        b, m_ = cells(base, D1), cells(mtd, D1)
        v6[fam] = {"delta_pts": [100 * (y - x) / 20 for x, y in zip(b, m_)],
                   "per_task_p": {t: fisher(x, 20, y, 20) for t, x, y in zip(TASKS, b, m_)},
                   "total": {"k": [sum(b), sum(m_)], "p": fisher(sum(b), 100, sum(m_), 100)},
                   "n_tasks_up": int(sum(y > x for x, y in zip(b, m_))), "n_tasks_down": int(sum(y < x for x, y in zip(b, m_))),
                   "n_tasks_significant_p05": int(sum(fisher(x, 20, y, 20) < 0.05 for x, y in zip(b, m_)))}
    # 데이터셋 2의 MiTaS V+G MTD와 V+G+E MTD (추론에 Evetac를 쓴 경우)
    a, b = cells("MiTaS V+G MTD", D2), cells("MiTaS V+G+E MTD", D2)
    v6["D2_MTD_with_vs_without_evetac_inference"] = {"k": [sum(b), sum(a)], "p": fisher(sum(b), 100, sum(a), 100),
                                                    "per_task_delta_pts": [100 * (y - x) / 20 for x, y in zip(a, b)]}
    # MTD V+G(데이터셋 1) 대 MTD V+G(데이터셋 2): 같은 학습 방식, 다른 데이터셋
    a1, a2 = cells("MiTaS V+G MTD", D1), cells("MiTaS V+G MTD", D2)
    v6["MiTaS_MTD_D1_vs_D2_per_task_p"] = {t: fisher(x, 20, y, 20) for t, x, y in zip(TASKS, a1, a2)}
    # 표 3의 Key −35는 데이터셋 2의 MTD(65%)와 비교하면 반대 부호
    v6["Key_MiTaS"] = {"V+G D1": D1["MiTaS V+G"][3], "MTD D1": D1["MiTaS V+G MTD"][3], "MTD D2": D2["MiTaS V+G MTD"][3],
                      "p_D1_VG_vs_D1_MTD": fisher(11, 20, 4, 20), "p_D1_MTD_vs_D2_MTD": fisher(4, 20, 13, 20)}
    R["V6"] = v6

    # ---------- V7 데이터셋 이질성
    shared = ["ViT-CNN V", "ViT V", "Sparsh V+G+E", "MiTaS V+G+E", "MiTaS V+G MTD"]
    v7 = {"cells": {}, "n_cells": 0}
    ds_diffs, p_cells = [], []
    for name in shared:
        a, b = cells(name, D1), cells(name, D2)
        row = {}
        for t, x, y in zip(TASKS, a, b):
            p = fisher(x, 20, y, 20)
            row[t] = {"D1": x, "D2": y, "diff_pts": 5 * (y - x), "p": p}
            ds_diffs.append((y - x) / 20)
            p_cells.append((x + y) / 40)
        v7["cells"][name] = row
    allp = [c["p"] for r in v7["cells"].values() for c in r.values()]
    v7["n_cells"] = len(allp)
    v7["n_p_lt_005"] = int(sum(p < 0.05 for p in allp))
    v7["expected_by_chance_if_equal"] = 0.05 * len(allp)
    # 이항 분산만으로 설명되는 분산과 비교
    d = np.array(ds_diffs)
    pc = np.array(p_cells)
    var_obs = float(np.mean(d ** 2) - np.mean(d) ** 2)   # 차이의 분산(모집단 형태)
    var_binom = float(np.mean(2 * pc * (1 - pc) / 20))
    excess = max(var_obs - var_binom, 0.0)
    v7["variance"] = {"mean_diff_pts": float(100 * d.mean()), "sd_obs_pts": 100 * math.sqrt(var_obs),
                      "sd_binomial_only_pts": 100 * math.sqrt(var_binom), "excess_var": excess,
                      "sigma_run_pts_if_two_independent_runs": 100 * math.sqrt(excess / 2)}
    # 두 데이터셋의 평균(5과제) 차이
    v7["row_avg"] = {n: [float(np.mean(D1[n])), float(np.mean(D2[n]))] for n in shared}
    v7["row_avg_diff_pts"] = {n: float(np.mean(D2[n]) - np.mean(D1[n])) for n in shared}
    R["V7"] = v7

    # ---------- V8 표 5
    v8 = {"rows": []}
    for model, sens, perc, total, tok, ms, hz in T5:
        v8["rows"].append({"model": model, "sensors": sens, "perc_plus_head": perc + HEAD, "total_reported": total,
                           "sum_ok": perc + HEAD == total, "ms": ms, "hz_reported": hz,
                           "hz_from_ms": round(1000 / ms, 1), "hz_ratio": round(hz * ms / 1000, 3)})
    ratios = [r["hz_ratio"] for r in v8["rows"]]
    v8["hz_ratio_range"] = [min(ratios), max(ratios)]
    # 가산성: 비전 + (V+G − V) + (V+E − V) = V+G+E ?
    v_, vg_, ve_, vge = 3079232, 5074776, 8769280, 10764824
    v8["additivity_MiTaS"] = {"V": v_, "gelsight_stem": vg_ - v_, "evetac_stem": ve_ - v_, "predicted_VGE": v_ + (vg_ - v_) + (ve_ - v_),
                              "reported_VGE": vge, "exact": v_ + (vg_ - v_) + (ve_ - v_) == vge,
                              "evetac_stem_over_gelsight_stem": (ve_ - v_) / (vg_ - v_)}
    v8["sparsh_increments"] = {"G": 9398744 - 4051328, "E": 15500056 - 9398744}
    v8["inference_ms_overhead"] = {"MiTaS V+G vs V": round(35.3 - 35.5, 1), "MiTaS V+G+E vs V+G": round(43.5 - 35.3, 1),
                                   "Sparsh V+G+E vs V+G": round(54.6 - 39.2, 1)}
    R["V8"] = v8

    # ---------- K1 토큰 수
    k1 = {"vision": 16 * 16, "gelsight": 12 * 16, "evetac": 12 * 16}
    k1["sum"] = sum(k1.values())
    k1["reported_Ncond"] = 644
    k1["unexplained_extra"] = 644 - k1["sum"]
    k1["table5"] = {"MiTaS V+G+E": 644 - 640, "MiTaS V+G": 452 - 448, "MiTaS V+E": 452 - (256 + 192), "ViT": 260 - 256,
                    "Sparsh V+G+E": 640 - 640, "Sparsh V+G": 448 - 448}
    k1["attention_share_if_uniform"] = {"vision": 256 / 644, "gelsight": 192 / 644, "evetac": 192 / 644, "extra": 4 / 644}
    R["K1"] = k1

    # ---------- K2 Evetac 프레임과 창
    R["K2"] = {"frames": 16, "rate_hz": 200, "window_s": 0.06, "frames_in_window_at_200hz": 0.06 * 200,
               "rate_needed_hz": 16 / 0.06, "span_of_16_frames_at_200hz_ms": 16 / 200 * 1000,
               "vision_frames_2_at_25hz_span_ms": 1 / 25 * 1000}

    # ---------- K3 DiT 머리 파라미터 추정 (D=256, L=10, MLP 4배, 헤드 8, 시간 임베딩 128)
    D, L, dt, Hh, da = 256, 10, 128, 16, 4
    attn = 4 * (D * D + D)
    mlp = D * 4 * D + 4 * D + 4 * D * D + D
    ada_block = dt * 9 * D + 9 * D
    block = 2 * attn + mlp + ada_block
    time_mlp = dt * D + D + D * D + D
    inp = da * D + D + Hh * D
    final_ada = dt * 2 * D + 2 * D
    outp = D * da + da
    est = L * block + time_mlp + inp + final_ada + outp
    R["K3"] = {"per_block": block, "ten_blocks": L * block, "estimate_total": est, "reported_head": HEAD,
               "relative_gap": (est - HEAD) / HEAD,
               "note": "시간 임베딩 128 -> AdaLN 9D(셀프, 크로스, MLP 각 β·γ·g)를 가정한 근사. 논문이 층 구성을 다 적지 않아 정확히 맞추지는 못함"}

    # ---------- K4 검출력과 훈련 실행 변동 토이
    rng = np.random.default_rng(0)

    def power(p1, p2, n, reps=3000):
        hit = 0
        for _ in range(reps):
            a, b = rng.binomial(n, p1), rng.binomial(n, p2)
            hit += fisher(int(a), n, int(b), n) < 0.05
        return hit / reps

    k4 = {"power": {}}
    for n in (20, 40, 100, 200):
        k4["power"][f"n={n}"] = {f"{p1:.2f} vs {p2:.2f}": power(p1, p2, n, 1500 if n >= 100 else 3000)
                                 for p1, p2 in ((0.5, 0.65), (0.5, 0.75), (0.5, 0.85), (0.3, 0.55))}
    # 같은 참 성공률 0.6인 두 모델을 과제 5개 평균으로 비교. 훈련 실행 변동 sigma_run(셀당)이 있을 때
    sig_hat = v7["variance"]["sigma_run_pts_if_two_independent_runs"] / 100

    def avg_gap_sd(sigma, n=20, p=0.6, reps=20000):
        g = np.empty(reps)
        for i in range(reps):
            pa = np.clip(p + rng.normal(0, sigma, 5), 0.02, 0.98)
            pb = np.clip(p + rng.normal(0, sigma, 5), 0.02, 0.98)
            g[i] = (rng.binomial(n, pa) - rng.binomial(n, pb)).mean() / n
        return g

    k4["null_gap"] = {}
    for sig in (0.0, sig_hat, 0.15):
        g = avg_gap_sd(sig)
        k4["null_gap"][f"sigma_run={sig:.3f}"] = {"sd_pts": float(100 * g.std()), "p_gap_ge_24.5_pts_one_dataset": float(np.mean(g >= 0.245)),
                                                  "p_gap_ge_24.5_abs": float(np.mean(np.abs(g) >= 0.245))}
    # 두 데이터셋을 평균(10셀)
    def avg_gap_two(sigma, reps=20000):
        g = np.empty(reps)
        for i in range(reps):
            pa = np.clip(0.6 + rng.normal(0, sigma, 10), 0.02, 0.98)
            pb = np.clip(0.6 + rng.normal(0, sigma, 10), 0.02, 0.98)
            g[i] = (rng.binomial(20, pa) - rng.binomial(20, pb)).mean() / 20
        return g

    k4["null_gap_two_datasets"] = {}
    for sig in (0.0, sig_hat, 0.15):
        g = avg_gap_two(sig)
        k4["null_gap_two_datasets"][f"sigma_run={sig:.3f}"] = {"sd_pts": float(100 * g.std()),
                                                               "p_gap_ge_24.5_pts_abs": float(np.mean(np.abs(g) >= 0.245))}
    R["K4"] = k4

    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    # 요약 출력
    print("V1 불일치:", R["V1"]["mismatches"], "총 시행", R["V1"]["trials_total"])
    print("V2 그림 4 라벨 일치:", {k: v["match"] for k, v in R["V2"].items() if k != "text"})
    print("V4 MiTaS vs Sparsh-X 총:", R["V4"]["MiTaS_vs_SparshX_pooled_total"])
    print("V4 CMH:", R["V4"]["MiTaS_vs_SparshX_CMH_10strata"])
    print("V4 셀 단위:", R["V4"]["cell_level_10"])
    print("V4 D2 vs 기준선:", {k: v["p"] for k, v in R["V4"]["D2_MiTaS_vs_others_total_p"].items()})
    print("V5 full vs VG:", R["V5"]["full_vs_VG"]["total"], R["V5"]["full_vs_VG"]["per_task_p"])
    print("V6:", R["V6"]["MiTaS"]["per_task_p"], R["V6"]["Sparsh"]["per_task_p"], R["V6"]["Key_MiTaS"])
    print("V7:", R["V7"]["n_p_lt_005"], "/", R["V7"]["n_cells"], R["V7"]["variance"])
    print("V8 Hz ratio", R["V8"]["hz_ratio_range"], R["V8"]["additivity_MiTaS"])
    print("K3:", R["K3"]["estimate_total"], R["K3"]["relative_gap"])
    print("K4:", R["K4"]["null_gap"], R["K4"]["null_gap_two_datasets"])


if __name__ == "__main__":
    main()
