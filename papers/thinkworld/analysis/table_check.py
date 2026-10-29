"""ThinkWorld 표·그림·본문 수치 재계산 (CPU). 실행: python3 analysis/table_check.py

- C1: Table 6 실물 로봇 Macro/Micro, Fig. 1·3 막대값과의 관계
- C2: 실물 로봇 비교의 통계(Wilson 95% 구간, Fisher 정확검정): 전체·ID·OOD·과제별
- C3: "기준선이 OOD에서 크게 떨어진다"(Fig. 3 캡션) 점검
- C4: Table 1 LIBERO·LIBERO-PRO 평균 재계산, 축별 순위, 공식 LIBERO-PRO 리더보드와 대조
- C5: Table 3 구성 요소 기여 재계산
- C6: Table 2 반올림 구간 정합성
- C7: Table 4 데이터 규모 합계·주파수 정합성, 사람 데이터 비율
- C8: Table 8 LIBERO-Plus 평균: 단순 평균과 다름 → 공식 리더보드로 차원 가중치 추정 후 재현
- C9: Table 9 지연시간·wall time 정합성
- C10: 학습 규모(SFT·RL 반복, 표본 수)
"""
import json
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
from pathlib import Path

import numpy as np
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def fisher(k1, n1, k2, n2):
    return float(f"{stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]:.2e}")


def main() -> None:
    out: dict = {}

    # ── C1/C2/C3: Table 6 (successes/trials) ─────────────────────────────────
    # 열: Color Matching ID, OOD, Cup Hanging ID, OOD, Sorting ID, OOD
    T6 = {
        "Diffusion Policy": [(18, 30), (11, 30), (9, 15), (1, 15), (4, 15), (1, 15)],
        "MolmoAct2": [(24, 30), (8, 30), (11, 15), (14, 15), (12, 15), (3, 15)],
        "LaWAM": [(16, 30), (8, 30), (9, 15), (12, 15), (4, 15), (4, 15)],
        "ThinkWorld": [(30, 30), (24, 30), (15, 15), (15, 15), (15, 15), (13, 15)],
    }
    REPORTED = {"Diffusion Policy": (32.8, 36.7), "MolmoAct2": (62.2, 60.0), "LaWAM": (45.6, 44.2),
                "ThinkWorld": (94.4, 93.3)}
    FIG1 = {"Diffusion Policy": (48.9, 16.7), "MolmoAct2": (77.8, 46.7), "LaWAM": (46.7, 44.4),
            "ThinkWorld": (100.0, 88.9)}
    c1 = {}
    for m, cells in T6.items():
        rates = [100 * k / n for k, n in cells]
        macro = sum(rates) / 6
        micro = 100 * sum(k for k, _ in cells) / sum(n for _, n in cells)
        id_task_macro = (rates[0] + rates[2] + rates[4]) / 3
        ood_task_macro = (rates[1] + rates[3] + rates[5]) / 3
        c1[m] = {
            "macro": round(macro, 2), "micro": round(micro, 2), "reported": REPORTED[m],
            "fig1_ID_OOD_task_mean": [round(id_task_macro, 1), round(ood_task_macro, 1)], "fig1_reported": FIG1[m],
            "ID_pooled": [sum(cells[i][0] for i in (0, 2, 4)), 60],
            "OOD_pooled": [sum(cells[i][0] for i in (1, 3, 5)), 60],
        }
    out["C1_table6_recompute"] = c1

    tw = T6["ThinkWorld"]
    c2 = {}
    for m in ("MolmoAct2", "LaWAM", "Diffusion Policy"):
        b = T6[m]
        k_tw, k_b = sum(k for k, _ in tw), sum(k for k, _ in b)
        idt, idb = sum(tw[i][0] for i in (0, 2, 4)), sum(b[i][0] for i in (0, 2, 4))
        odt, odb = sum(tw[i][0] for i in (1, 3, 5)), sum(b[i][0] for i in (1, 3, 5))
        c2[m] = {
            "all_120": {"thinkworld": [k_tw, wilson(k_tw, 120)], "baseline": [k_b, wilson(k_b, 120)],
                        "fisher_p": fisher(k_tw, 120, k_b, 120)},
            "ID_60": {"thinkworld": idt, "baseline": idb, "fisher_p": fisher(idt, 60, idb, 60)},
            "OOD_60": {"thinkworld": odt, "baseline": odb, "fisher_p": fisher(odt, 60, odb, 60)},
            "per_setting_p": [fisher(tw[i][0], tw[i][1], b[i][0], b[i][1]) for i in range(6)],
        }
    c2["note"] = "시행은 독립으로 가정(설정당 5회 묶음, 체크포인트 1개). Appendix D: 학습 seed 1개"
    out["C2_real_robot_tests"] = c2

    c3 = {}
    for m, cells in T6.items():
        r = [100 * k / n for k, n in cells]
        c3[m] = {"color_ID_to_OOD": round(r[1] - r[0], 1), "cup_ID_to_OOD": round(r[3] - r[2], 1),
                 "sort_ID_to_OOD": round(r[5] - r[4], 1)}
    c3["claim"] = "Fig. 3 캡션: 'despite substantial drops for the baselines' / Sec. 4.2: 'perform similarly in-domain'"
    out["C3_ood_drop_check"] = c3

    # ── C4: Table 1 ──────────────────────────────────────────────────────────
    LIB = {  # Spatial Object Goal Long Avg
        "OpenVLA": ([84.7, 88.4, 79.2, 53.7], 76.5), "pi0": ([96.8, 98.8, 95.8, 85.2], 94.2),
        "pi0.5": ([98.8, 98.2, 98.0, 92.4], 96.9), "MolmoAct": ([87.0, 95.4, 87.6, 77.2], 86.6),
        "VLA-JEPA": ([96.2, 99.6, 97.2, 95.8], 97.2), "ThinkWorld": ([98.2, 98.8, 97.8, 94.2], 97.3),
    }
    PRO = {  # Object Position Semantic Task Environment Avg
        "OpenVLA": ([93.0, 0.0, 97.3, 0.0, 68.0], 51.7), "pi0": ([90.5, 0.0, 90.5, 0.0, 38.8], 44.0),
        "pi0.5": ([96.0, 20.8, 95.8, 0.8, 52.8], 53.2), "MolmoAct": ([76.0, 1.5, 85.8, 1.5, None], 41.2),
        "VLA-JEPA": ([80.4, 3.5, 97.2, 20.8, 41.0], 48.5), "ThinkWorld": ([73.3, 18.2, 93.9, 31.9, 60.9], 55.6),
    }
    c4 = {"LIBERO_avg": {m: {"recomputed": round(sum(v) / 4, 3), "reported": a} for m, (v, a) in LIB.items()}}
    c4["LIBERO_PRO_avg"] = {m: {"recomputed": round(np.mean([x for x in v if x is not None]), 3),
                                "n_axes": sum(x is not None for x in v), "reported": a} for m, (v, a) in PRO.items()}
    axes = ["Object", "Position", "Semantic", "Task", "Environment"]
    rank = {}
    for j, ax in enumerate(axes):
        vals = sorted(((v[j], m) for m, (v, _) in PRO.items() if v[j] is not None), reverse=True)
        rank[ax] = {"best": vals[0], "thinkworld_rank": [m for _, m in vals].index("ThinkWorld") + 1, "n": len(vals)}
    c4["LIBERO_PRO_axis_rank"] = rank
    # 공식 LIBERO-PRO 리더보드(Zxy-MLlab/LIBERO-PRO README @ eafdb80): suite별 Obj Pos Sem Task Env
    OFFICIAL = {
        "OpenVLA": [[0.96, 0.00, 0.98, 0.00, 0.98], [0.97, 0.00, 0.97, 0.00, 0.89], [0.81, 0.00, 0.96, 0.00, 0.85], [0.98, 0.00, 0.98, 0.00, 0.00]],
        "pi0": [[0.94, 0.00, 0.93, 0.00, 0.39], [0.95, 0.00, 0.97, 0.00, 0.60], [0.79, 0.00, 0.82, 0.00, 0.27], [0.94, 0.00, 0.90, 0.00, 0.29]],
        "pi0.5": [[0.97, 0.38, 0.97, 0.00, 0.46], [0.97, 0.20, 0.97, 0.01, 0.46], [0.92, 0.08, 0.93, 0.01, 0.46], [0.98, 0.17, 0.96, 0.01, 0.73]],
        "MolmoAct": [[0.68, 0.00, 0.85, 0.00, None], [0.90, 0.00, 0.88, 0.00, None], [0.54, 0.00, 0.74, 0.06, None], [0.92, 0.06, 0.96, 0.00, None]],
    }
    match = {}
    for m, suites in OFFICIAL.items():
        means = []
        for j in range(5):
            col = [s[j] for s in suites]
            means.append(None if col[0] is None else round(100 * sum(col) / 4, 2))
        match[m] = {"official_4suite_mean": means, "table1": PRO[m][0],
                    "max_abs_diff": round(max(abs(a - b) for a, b in zip(means, PRO[m][0]) if a is not None), 2)}
    c4["official_leaderboard_match"] = match
    c4["margin_vs_VLA_JEPA_LIBERO"] = round(sum(LIB["ThinkWorld"][0]) / 4 - sum(LIB["VLA-JEPA"][0]) / 4, 3)
    c4["margin_vs_pi05_LIBERO_PRO"] = round(np.mean(PRO["ThinkWorld"][0]) - np.mean(PRO["pi0.5"][0]), 2)
    out["C4_table1"] = c4

    # ── C5: Table 3 ──────────────────────────────────────────────────────────
    T3 = [("Qwen baseline", [59.5, 1.9, 84.4, 17.2, 37.8], 40.1), ("+ History", [61.3, 4.9, 84.6, 22.8, 29.8], 40.7),
          ("+ Future prediction", [70.0, 11.9, 89.4, 26.3, 54.4], 50.4), ("+ CoD", [74.5, 16.9, 94.4, 29.7, 56.4], 54.4),
          ("+ RL", [73.3, 18.2, 93.9, 31.9, 60.9], 55.6)]
    c5 = {"avg": {n: {"recomputed": round(np.mean(v), 2), "reported": a} for n, v, a in T3}}
    c5["step_gain"] = {T3[i][0]: {"avg": round(np.mean(T3[i][1]) - np.mean(T3[i - 1][1]), 2),
                                  "per_axis": [round(a - b, 1) for a, b in zip(T3[i][1], T3[i - 1][1])]}
                       for i in range(1, len(T3))}
    total = np.mean(T3[-1][1]) - np.mean(T3[0][1])
    c5["share_of_total_gain"] = {T3[i][0]: round((np.mean(T3[i][1]) - np.mean(T3[i - 1][1])) / total, 3)
                                 for i in range(1, len(T3))}
    out["C5_table3"] = c5

    # ── C6: Table 2 반올림 구간 ─────────────────────────────────────────────
    def interval(on_shown, rows):
        lo, hi = on_shown - 0.005, on_shown + 0.005
        for shown, delta in rows:  # 행 값 = on + delta, 반올림해 shown
            lo, hi = max(lo, shown - 0.005 - delta), min(hi, shown + 0.005 - delta)
        return [round(lo, 4), round(hi, 4), lo < hi]
    out["C6_table2_rounding"] = {
        "all_patches_CoD_on_interval": interval(0.74, [(0.71, -0.024), (0.72, -0.020), (0.72, -0.018)]),
        "top25_CoD_on_interval": interval(0.62, [(0.55, -0.071), (0.54, -0.079), (0.55, -0.071)]),
    }

    # ── C7: Table 4 ──────────────────────────────────────────────────────────
    T4 = {  # episodes K, frames M, hours, Hz
        "YAM": (32.2, 76.0, 704.1, 30), "DROID": (74.6, 17.8, 328.9, 15), "RoboCOIN": (112.0, 82.3, 755.2, 30),
        "Bridge": (53.2, 1.9, 105.2, 5), "RT-1": (87.2, 3.8, 350.6, 3), "BC-Z": (39.4, 5.5, 152.0, 10),
        "EgoVerse": (55.6, 124.8, 1155.4, 30), "EgoDex": (314.8, 78.3, 724.8, 30),
    }
    sums = [round(sum(v[i] for v in T4.values()), 1) for i in range(3)]
    hz = {k: round(v[1] * 1e6 / (v[2] * 3600), 2) for k, v in T4.items()}
    human = ("EgoVerse", "EgoDex")
    share = {name: round(sum(T4[h][i] for h in human) / sums[i], 3) for i, name in enumerate(("episodes", "frames", "hours"))}
    out["C7_table4"] = {"sums_EP_K_frames_M_hours": sums, "reported_totals": [769.1, 390.3, 4276.1],
                        "implied_Hz": hz, "human_share": share,
                        "note": "사람 데이터도 'One-second 6D/12D motion tokens'(손목 자세 주석)를 정답으로 씀 → 'no action labels'가 아님"}

    # ── C8: Table 8 LIBERO-Plus ─────────────────────────────────────────────
    # 공식 리더보드(sylvestf/LIBERO-plus README @ 4976dc3): Camera Robot Language Light Background Noise Layout Total
    LP = {
        "OpenVLA": [0.8, 3.5, 23.0, 8.1, 34.8, 15.2, 28.5, 15.6], "OpenVLA-OFT": [56.4, 31.9, 79.5, 88.7, 93.3, 75.8, 74.2, 69.6],
        "OpenVLA-OFT_w": [10.4, 38.7, 70.5, 76.8, 93.6, 49.9, 69.9, 55.8], "NORA": [2.2, 37.0, 65.1, 45.7, 58.6, 12.8, 62.1, 39.0],
        "WorldVLA": [0.1, 27.9, 41.6, 43.7, 17.1, 10.9, 38.0, 25.0], "UniVLA": [1.8, 46.2, 69.6, 69.0, 81.0, 21.2, 31.9, 43.9],
        "pi0": [13.8, 6.0, 58.8, 85.0, 81.4, 79.0, 68.9, 53.6], "pi0-Fast": [65.1, 21.6, 61.0, 73.2, 73.2, 74.4, 68.8, 61.6],
        "RIPT-VLA": [55.2, 31.2, 77.6, 88.4, 91.6, 73.5, 74.2, 68.4], "OpenVLA-OFT_m": [55.6, 21.7, 81.0, 92.7, 91.0, 78.6, 68.7, 67.9],
        "OpenVLA-OFT+": [92.8, 30.3, 85.8, 94.9, 93.9, 89.3, 77.6, 79.6],
    }
    X = np.array([v[:7] for v in LP.values()])
    y = np.array([v[7] for v in LP.values()])
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
    res = optimize.minimize(lambda w: ((X @ w - y) ** 2).sum(), np.full(7, 1 / 7), bounds=[(0, 1)] * 7,
                            constraints=cons, method="SLSQP")
    w = res.x
    T8 = {"OpenVLA-OFT": [56.4, 31.9, 79.5, 88.7, 93.3, 75.8, 74.2, 69.6], "pi0-FAST": [65.1, 21.6, 61.0, 73.2, 73.2, 74.4, 68.8, 61.6],
          "DreamVLA": [65.0, 40.8, 63.5, 85.7, 82.6, 84.9, 74.0, 69.9], "ThinkWorld": [46.9, 70.5, 73.5, 90.7, 90.0, 62.7, 79.6, 71.7]}
    out["C8_libero_plus"] = {
        "estimated_dim_weights": [round(x, 4) for x in w],
        "fit_rmse_on_leaderboard": round(float(np.sqrt(((X @ w - y) ** 2).mean())), 3),
        "table8": {m: {"uniform_mean": round(np.mean(v[:7]), 2), "weighted_by_estimate": round(float(np.dot(w, v[:7])), 2),
                       "reported": v[7]} for m, v in T8.items()},
        "leaderboard_rows_not_in_table8": [k for k in LP if k not in ("OpenVLA-OFT", "pi0-Fast")],
        "thinkworld_worst_dims": "Camera 46.9, Noise 62.7 (표의 4개 방법 중 최하)",
    }

    # ── C9: Table 9 ──────────────────────────────────────────────────────────
    T9 = {"MolmoAct2": (335.07, 467.96, 18.44), "MolmoAct2-Think": (879.37, 1014.47, 46.76), "ThinkWorld": (324.84, 333.09, 16.31)}
    out["C9_latency"] = {m: {"wall_per_call_ms": round(w_ * 1000 / 50, 1), "p50": p50, "p95": p95} for m, (p50, p95, w_) in T9.items()}

    # ── C10: 학습 규모 ──────────────────────────────────────────────────────
    out["C10_training_scale"] = {
        "VLM_SFT_samples": 150_000 * 128, "action_SFT_samples": 50_000 * 128,
        "RL_prompts": 2_000 * 128, "RL_CoD_samples(K=4)": 2_000 * 128 * 4,
        "RL_iters_over_SFT_iters": round(2_000 / 150_000, 4),
        "RL_avg_gain_pp(Table3)": round(np.mean(T3[-1][1]) - np.mean(T3[-2][1]), 2),
        "frames_per_SFT_sample": round(390.3e6 / (150_000 * 128), 1),
    }

    path = ROOT / "analysis" / "table_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in out.items():
        print(k, json.dumps(v, ensure_ascii=False)[:400])
    print("wrote", path)


if __name__ == "__main__":
    main()
