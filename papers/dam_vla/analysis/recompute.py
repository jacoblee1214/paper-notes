"""DAM-VLA (arXiv:2606.12105 v1) 보고 수치 재계산과 검정. 수 초.

실행: python3 analysis/recompute.py  → analysis/recompute_results.json (표준 출력에 요약)

입력은 모두 논문에서 옮긴 값입니다.
- 표 2 (과제별 성공률, 과제당 15회), 그림 4 (네 과제의 clean/partial 분해, 프로젝트 페이지 막대 값과 같음)
- 그림 6, 7, 8 (SPARC, 추종 지연, 에피소드 시간)은 PDF에서 막대와 오차 막대 길이를 재서 얻은 판독 값입니다.
시행 수는 논문 본문이 적은 과제당 15회를 씁니다(§4.1). 공식 코드는 없으므로 저장소 읽기는 하지 않습니다.

번호 V1–V12는 이 문서군 안의 확인 항목입니다.
"""
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
N = 15
TASKS = ["Scarf", "Whiteboard", "Button", "Handwash", "Lego", "Socket", "Sweep"]
CONFIGS = ["X-VLA25", "X-VLA100", "X-VLA_AFM", "DAM/F/M", "DAM/F", "DAM/M", "DAM-VLA"]
# 표 2 (%) 와 보고된 평균 열
T2 = {
    "X-VLA25": [80.0, 86.7, 13.3, 0.0, 0.0, 6.7, 100.0],
    "X-VLA100": [80.0, 13.3, 6.7, 0.0, 0.0, 0.0, 53.3],
    "X-VLA_AFM": [100.0, 73.3, 13.3, 86.7, 0.0, 6.7, 100.0],
    "DAM/F/M": [80.0, 66.7, 40.0, 20.0, 0.0, 6.7, 66.7],
    "DAM/F": [100.0, 73.3, 86.7, 40.0, 0.0, 6.7, 100.0],
    "DAM/M": [100.0, 86.7, 86.7, 80.0, 13.3, 13.3, 86.7],
    "DAM-VLA": [100.0, 100.0, 93.3, 100.0, 93.3, 80.0, 100.0],
}
REPORTED_AVG = {"X-VLA25": 40.95, "X-VLA100": 21.9, "X-VLA_AFM": 54.3, "DAM/F/M": 40.0,
                "DAM/F": 58.1, "DAM/M": 66.7, "DAM-VLA": 95.2}
# 그림 4: (clean %, partial %) — Button, Handwash, Lego, Socket
FIG4 = {
    "Button": {"X-VLA25": (0, 13.33), "X-VLA100": (6.67, 0), "X-VLA_AFM": (0, 13.33), "DAM/F/M": (13.33, 26.67),
               "DAM/F": (80, 6.67), "DAM/M": (86.67, 0), "DAM-VLA": (86.67, 6.67)},
    "Handwash": {"X-VLA25": (0, 0), "X-VLA100": (0, 0), "X-VLA_AFM": (73.33, 13.33), "DAM/F/M": (13.33, 6.67),
                 "DAM/F": (20, 20), "DAM/M": (33.33, 46.67), "DAM-VLA": (86.67, 13.33)},
    "Lego": {"X-VLA25": (0, 0), "X-VLA100": (0, 0), "X-VLA_AFM": (0, 0), "DAM/F/M": (0, 0),
             "DAM/F": (0, 0), "DAM/M": (13.33, 0), "DAM-VLA": (66.67, 26.67)},
    "Socket": {"X-VLA25": (0, 6.67), "X-VLA100": (0, 0), "X-VLA_AFM": (6.67, 0), "DAM/F/M": (6.67, 0),
               "DAM/F": (6.67, 0), "DAM/M": (6.67, 6.67), "DAM-VLA": (73.33, 6.67)},
}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def fisher(a, na, b, nb):
    return stats.fisher_exact([[a, na - a], [b, nb - b]])[1]


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def cmh(tables):
    """tables: [(a, n1, b, n2)] 과제별 (성공1, 시행1, 성공2, 시행2). Mantel-Haenszel 공통 오즈비와 연속성 보정 카이제곱."""
    num = den = 0.0
    sa = se = sv = 0.0
    for a, n1, b, n2 in tables:
        n = n1 + n2
        c, d = n1 - a, n2 - b
        num += a * d / n
        den += b * c / n
        m1, m0 = a + b, c + d
        sa += a
        se += n1 * m1 / n
        sv += n1 * n2 * m1 * m0 / (n * n * (n - 1))
    chi = (abs(sa - se) - 0.5) ** 2 / sv
    return num / den if den > 0 else float("inf"), chi, float(stats.chi2.sf(chi, 1))


def main():
    res = {}
    # V1 정수 격자: 표 2의 모든 칸이 k/15 의 반올림인가
    counts = {}
    bad = []
    for c in CONFIGS:
        counts[c] = []
        for t, v in zip(TASKS, T2[c]):
            ks = [k for k in range(N + 1) if abs(round(100 * k / N, 1) - v) < 1e-9 or abs(100 * k / N - v) < 0.051]
            if len(ks) != 1:
                bad.append((c, t, v, ks))
            counts[c].append(ks[0] if ks else None)
    res["V1_counts"] = counts
    res["V1_not_on_grid"] = bad
    # V2 평균: 7개 과제 단순 평균 = 총 성공 수 / 105
    avg = {}
    for c in CONFIGS:
        tot = sum(counts[c])
        avg[c] = {"k": tot, "pooled_pct": 100 * tot / (N * 7), "mean_of_cells": float(np.mean(T2[c])),
                  "reported": REPORTED_AVG[c]}
    res["V2_avg"] = avg
    # V3 초록·서론 비율
    full, x25, afm = avg["DAM-VLA"]["pooled_pct"], avg["X-VLA25"]["pooled_pct"], avg["X-VLA_AFM"]["pooled_pct"]
    res["V3_headline"] = {"full_pct": full, "x25_pct": x25, "ratio_vs_x25": full / x25, "diff_pp_vs_x25": full - x25,
                          "diff_pp_from_rounded_text": 95.2 - 40.95, "ratio_vs_afm": full / afm,
                          "diff_pp_vs_afm": full - afm}
    # V4 Wilson 구간 (합산 105회, 칸 15회)
    res["V4_wilson_pooled"] = {c: [round(100 * x, 1) for x in wilson(avg[c]["k"], 105)] for c in CONFIGS}
    res["V4_wilson_full_cells"] = {t: [round(100 * x, 1) for x in wilson(counts["DAM-VLA"][i], N)]
                                   for i, t in enumerate(TASKS)}
    # V5 과제별 Fisher: DAM-VLA 대 각 구성 (42개), Holm
    rows, ps = [], []
    for c in CONFIGS[:-1]:
        for i, t in enumerate(TASKS):
            p = fisher(counts["DAM-VLA"][i], N, counts[c][i], N)
            rows.append((c, t, counts["DAM-VLA"][i], counts[c][i], p))
            ps.append(p)
    adj = holm(np.array(ps))
    res["V5_fisher_cells"] = [{"vs": r[0], "task": r[1], "k_full": r[2], "k_other": r[3], "p": r[4], "p_holm42": float(a)}
                              for r, a in zip(rows, adj)]
    res["V5_summary"] = {
        "n_tests": len(rows), "p_lt_0.05": int(sum(p < 0.05 for p in ps)),
        "holm_lt_0.05": int(sum(a < 0.05 for a in adj)),
        "ties_or_full_not_higher": int(sum(r[2] <= r[3] for r in rows)),
    }
    # V6 CMH: 과제로 층화한 DAM-VLA 대 각 구성
    cm = {}
    for c in CONFIGS[:-1]:
        tabs = [(counts["DAM-VLA"][i], N, counts[c][i], N) for i in range(7)]
        orr, chi, p = cmh(tabs)
        cm[c] = {"or_mh": orr, "chi2": chi, "p": p}
    res["V6_cmh_vs_full"] = cm
    # V7 부호 검정: DAM-VLA가 칸마다 같거나 높은가
    sign = {}
    for c in CONFIGS[:-1]:
        win = sum(counts["DAM-VLA"][i] > counts[c][i] for i in range(7))
        tie = sum(counts["DAM-VLA"][i] == counts[c][i] for i in range(7))
        lose = 7 - win - tie
        sign[c] = {"win": win, "tie": tie, "lose": lose,
                   "exact_p_one_sided": float(stats.binom.sf(win - 1, win + lose, 0.5)) if win + lose else 1.0}
    res["V7_sign"] = sign
    # V8 절제의 상호작용: 메모리와 힘의 효과가 더해지는가 (본문 "each contributing independently")
    pooled = lambda c: avg[c]["pooled_pct"]
    base = pooled("DAM/F/M")
    mem_only, force_only, both = pooled("DAM/F"), pooled("DAM/M"), pooled("DAM-VLA")
    inter = both - mem_only - force_only + base
    rng = np.random.default_rng(0)
    draws = []
    cells = lambda c: [counts[c][i] / N for i in range(7)]
    pb, pm, pf, pa = cells("DAM/F/M"), cells("DAM/F"), cells("DAM/M"), cells("DAM-VLA")
    for _ in range(20000):
        s = [rng.binomial(N, p, size=7).sum() for p in (pb, pm, pf, pa)]
        draws.append(100 * (s[3] - s[1] - s[2] + s[0]) / 105)
    per_task = {t: round(100 * (pa[i] - pm[i] - pf[i] + pb[i]), 1) for i, t in enumerate(TASKS)}
    res["V8_interaction"] = {
        "base_FM": base, "memory_effect": mem_only - base, "force_effect": force_only - base,
        "both_effect": both - base, "additive_prediction": base + (mem_only - base) + (force_only - base),
        "interaction_pp": inter, "bootstrap_95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "per_task_interaction_pp": per_task,
    }
    # V9 clean 대 partial (그림 4)
    f4 = {}
    for t, d in FIG4.items():
        i = TASKS.index(t)
        f4[t] = {}
        for c, (cl, pa_) in d.items():
            k_cl, k_pa = round(cl * N / 100), round(pa_ * N / 100)
            f4[t][c] = {"clean": k_cl, "partial": k_pa, "total_from_fig4": k_cl + k_pa, "table2": counts[c][i],
                        "match": k_cl + k_pa == counts[c][i]}
    res["V9_fig4"] = f4
    res["V9_all_totals_match_table2"] = all(v["match"] for d in f4.values() for v in d.values())
    clean_upper = {}
    for c in CONFIGS:
        k = 0
        for i, t in enumerate(TASKS):
            k += f4[t][c]["clean"] if t in f4 else counts[c][i]
        clean_upper[c] = {"k": k, "pct_upper_bound": 100 * k / 105}
    res["V9_clean_only_upper_bound"] = clean_upper
    res["V9_note"] = ("Scarf, Whiteboard, Sweep은 clean/partial 분해가 없어 전부 clean이라고 둔 상한입니다. "
                      "네 과제의 clean은 그림 4 값입니다.")
    # V10 핸드워시 반복 누름 서술: /M 에서 46.67% 는 그림 4의 partial 7/15
    res["V10_multipress_text"] = {"text_pct": 46.67, "fig4_partial_DAM/M_handwash": 100 * 7 / 15,
                                  "AFM_handwash_partial_k": f4["Handwash"]["X-VLA_AFM"]["partial"]}
    # V11 재계획 주파수 산술 (식: 상한 = 제어 주파수 / 실행 지평 s)
    hz = []
    for ctrl, s, reported, where in [(100, 22, 5.5, "본문 §4.1 (100 Hz 제어, s=22)"),
                                     (200, 22, 8.0, "본문 §4.1·부록 F 표 4 (200 Hz, s=22)"),
                                     (200, 6, 17.0, "본문 §4.1·부록 F 표 4 (200 Hz, s=6)")]:
        bound = ctrl / s
        implied_cycle_ms = 1000 / reported
        exec_ms = 1000 * s / ctrl
        hz.append({"where": where, "ctrl_hz": ctrl, "s": s, "reported_hz": reported, "upper_bound_hz_if_full_chunk": bound,
                   "reported_over_bound": reported / bound, "chunk_exec_ms": exec_ms, "reported_cycle_ms": implied_cycle_ms})
    res["V11_replan_arithmetic"] = hz
    res["V11_window_arithmetic"] = {
        "vision_16_frames_at_25Hz_s": 16 / 25, "vision_16_frames_stride8_at_100Hz_s": 16 * 8 / 100,
        "vision_16_frames_stride4_at_100Hz_s": 16 * 4 / 100, "stride8_at_100Hz_effective_rate_hz": 100 / 8,
        "stride8_at_200Hz_effective_rate_hz": 200 / 8, "force_96_at_100Hz_s": 96 / 100, "force_96_at_200Hz_s": 96 / 200,
        "vlm_refresh_every4_replans_at_5.5Hz_hz": 5.5 / 4, "vlm_refresh_every4_control_steps_at_100Hz_hz": 100 / 4,
    }
    # V12 그림 6–8 판독 값
    sparc = {"X-VLA100": (25.04, 7.03), "X-VLA_AFM": (10.47, 2.93), "DAM/F/M": (16.83, 3.54), "DAM/F": (10.98, 2.40),
             "DAM/M": (10.6, 2.96), "DAM-VLA": (8.1, 1.90)}
    lag = {"X-VLA25": (0.189, 0.010), "X-VLA100": (0.135, 0.043), "X-VLA_AFM": (0.124, 0.015), "DAM/F/M": (0.139, 0.018),
           "DAM/F": (0.128, 0.026), "DAM/M": (0.127, 0.023), "DAM-VLA": (0.118, 0.008)}
    res["V12_fig6_sparc_read"] = {k: {"value": v[0], "err_halfwidth_read": v[1]} for k, v in sparc.items()}
    res["V12_fig7_lag_read"] = {k: {"value": v[0], "err_halfwidth_read": v[1]} for k, v in lag.items()}
    res["V12_lag_text_vs_fig"] = {"text_s": 0.116, "fig_label_s": 0.118, "fig_bar_height_read_s": 0.1179}
    ov = {}
    for c in ["X-VLA_AFM", "DAM/F", "DAM/M", "DAM/F/M", "X-VLA100"]:
        d = lag[c][0] - lag["DAM-VLA"][0]
        ov[c] = {"diff_s": round(d, 3), "sum_halfwidths": round(lag[c][1] + lag["DAM-VLA"][1], 3),
                 "bars_overlap": d < lag[c][1] + lag["DAM-VLA"][1]}
    res["V12_lag_overlap_vs_full"] = ov
    ep = {  # 그림 8 판독 (s). 해칭 막대는 색 검출이 안 돼 본문 수치만 둔다.
        "DAM-VLA": {"Sweep": 22.9, "Handwash": 12.9, "Lego": 28.4, "Scarf": 24.8, "Whiteboard": 24.0, "Button": 10.2, "Socket": 21.9},
        "X-VLA25": {"Sweep": 24.6, "Scarf": 27.7, "Whiteboard": 24.8, "Button": 15.6, "Socket": 79.9},
        "X-VLA100": {"Sweep": 49.8, "Scarf": 36.7, "Whiteboard": 83.0, "Button": 19.6},
        "demo": {"Sweep": 13.8, "Handwash": 6.7, "Lego": 20.6, "Scarf": 13.1, "Whiteboard": 21.9, "Button": 6.3, "Socket": 17.1},
    }
    ratio = {t: round(ep["DAM-VLA"][t] / ep["demo"][t], 2) for t in ep["DAM-VLA"]}
    res["V12_fig8_read_s"] = ep
    res["V12_dam_over_demo"] = {"per_task": ratio, "mean": round(float(np.mean(list(ratio.values()))), 2)}
    res["V12_text_checks"] = {"whiteboard_x100_over_dam": 83 / 24.4, "button_text": [10.8, 16, 20],
                              "button_fig_read": [ep["DAM-VLA"]["Button"], ep["X-VLA25"]["Button"], ep["X-VLA100"]["Button"]]}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    print("V1 격자 밖 칸:", bad)
    for c in CONFIGS:
        a = avg[c]
        print(f"V2 {c:10s} k={a['k']:3d}/105 pooled={a['pooled_pct']:.2f} mean={a['mean_of_cells']:.2f} reported={a['reported']}")
    print("V3", {k: round(v, 3) for k, v in res["V3_headline"].items()})
    print("V4", res["V4_wilson_pooled"])
    print("V5", res["V5_summary"])
    print("V6", {k: (round(v["or_mh"], 1), f"{v['p']:.2g}") for k, v in cm.items()})
    print("V7", sign)
    print("V8", {k: (round(v, 2) if not isinstance(v, (list, dict)) else v) for k, v in res["V8_interaction"].items()})
    print("V9 totals match:", res["V9_all_totals_match_table2"], {c: round(v["pct_upper_bound"], 1) for c, v in clean_upper.items()})
    print("V11", hz)
    print("V12", res["V12_lag_overlap_vs_full"], res["V12_dam_over_demo"])


if __name__ == "__main__":
    main()
