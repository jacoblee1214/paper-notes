"""논문 표·그림 수치 재계산과 일관성 검사 (arXiv:2606.13355 v2). 수 초.

입력: 본문 표 1, 표 A, 부록 표 B, 그림 7b의 숫자(손으로 옮김)와 analysis/figure_values.json(그림에서 읽은 값).
출력: analysis/recompute_results.json 과 화면 요약. 검사 번호 V1–V16은 analysis/03_analysis.md 와 같다.
실행: python3 analysis/recompute.py   (numpy, scipy 필요. 먼저 python3 analysis/extract_figures.py 로 figure_values.json 을 만들 것)

가정(문서에도 적음)
  - LIBERO 스위트 하나는 10 과제 x 50 초기 상태 = 500회. 시드 3개 평균이면 셀 하나는 1500회 (논문은 시행 수를 적지 않음).
  - 표의 (± x)는 시드 3개 표본 표준편차(n-1)로 읽는다 (논문은 정의를 적지 않음).
  - DROID는 8 과제 x 10회 = 80회 (부록 C). 그림 5, 그림 C의 값은 그림에서 읽은 값이다.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FIGV = json.loads((ROOT / "analysis" / "figure_values.json").read_text())
SUITES = ["Spatial", "Object", "Goal", "Long"]

# ---------------------------------------------------------------- 표 1 (v2) ----
# (모델, m, s): ([스위트 평균 4개], [스위트 SD 4개], 평균 열, 평균 SD)
T1 = {
    ("pi0_RTC", 1, 1): ([97.4, 99.3, 96.7, 82.4], [0.2, 0.1, 0.2, 0.4], 94.0, 0.2),
    ("pi0_RTC", 2, 2): ([96.7, 99.6, 97.7, 87.1], [1.3, 0.2, 0.8, 0.5], 94.7, 0.1),
    ("pi0_RTC", 3, 3): ([95.7, 99.5, 96.9, 86.5], [0.6, 0.3, 0.1, 0.8], 94.7, 0.2),
    ("pi0_RTC", 4, 1): ([96.8, 99.4, 96.4, 86.7], [0.4, 0.5, 0.4, 1.1], 95.0, 0.1),
    ("pi0_RTC", 4, 2): ([96.1, 99.3, 96.3, 87.4], [0.8, 0.6, 0.4, 1.1], 94.8, 0.2),
    ("pi0_RTC", 4, 3): ([95.7, 99.2, 97.0, 89.2], [0.6, 0.2, 0.5, 0.9], 95.3, 0.4),
    ("pi05_RTC", 1, 1): ([98.7, 98.4, 97.3, 91.8], [0.8, 0.3, 0.3, 0.4], 96.5, 0.1),
    ("pi05_RTC", 2, 2): ([98.8, 98.7, 98.4, 92.7], [0.0, 0.2, 0.2, 1.0], 97.2, 0.2),
    ("pi05_RTC", 3, 3): ([98.3, 99.0, 97.9, 93.7], [0.1, 0.2, 0.1, 0.1], 97.2, 0.1),
    ("pi05_RTC", 4, 1): ([98.5, 99.1, 97.3, 93.3], [0.8, 0.5, 0.3, 0.8], 97.1, 0.5),
    ("pi05_RTC", 4, 2): ([98.4, 99.0, 97.4, 93.9], [0.7, 0.3, 1.2, 0.6], 97.2, 0.6),
    ("pi05_RTC", 4, 3): ([98.6, 99.3, 97.1, 93.5], [0.4, 0.1, 0.5, 0.5], 97.2, 0.1),
    ("REALFAST", 4, 2): ([97.7, 98.9, 95.7, 91.5], [0.5, 0.5, 0.5, 1.7], 96.0, 0.7),
    ("REALFAST", 4, 3): ([97.7, 98.9, 95.9, 90.0], [0.8, 0.4, 1.3, 0.2], 95.6, 0.3),
    ("REALFAST", 4, 4): ([97.2, 98.6, 97.1, 89.9], [0.4, 0.5, 0.4, 0.3], 95.7, 0.3),
}
T1_MATCHED = {  # 시험 시 계산량 맞춤(디노이징 30단계)
    ("pi0_RTC_30", 3, 3): ([94.9, 98.7, 96.7, 85.6], [0.6, 0.5, 1.0, 0.8], 94.0, 0.3),
    ("pi0_RTC_30", 4, 3): ([93.8, 99.1, 95.7, 87.5], [0.6, 0.3, 0.3, 0.7], 94.0, 0.2),
    ("pi05_RTC_30", 3, 3): ([98.2, 98.8, 97.6, 93.0], [0.4, 0.5, 0.9, 0.8], 96.9, 0.1),
    ("pi05_RTC_30", 4, 3): ([98.7, 98.7, 96.9, 93.6], [0.1, 0.3, 1.0, 1.3], 97.0, 0.4),
}
T1_M0 = {"pi0": [96.8, 98.8, 95.8, 85.2, 94.2], "pi05": [98.8, 98.2, 98.0, 92.4, 96.9], "pi0_FAST": [93.0, 94.0, 91.0, 77.0, 88.8]}
# v1 (2026-06-11)의 기준선 행 (REALFAST 행은 v2와 같다)
T1_V1 = {
    ("pi0_RTC", 1, 1): ([97.3, 94.9, 91.2, 74.1], 89.4), ("pi0_RTC", 2, 2): ([97.3, 97.9, 94.9, 81.1], 92.8),
    ("pi0_RTC", 3, 3): ([97.3, 98.7, 95.0, 83.3], 93.6), ("pi0_RTC", 4, 1): ([97.8, 99.4, 97.1, 84.6], 94.7),
    ("pi0_RTC", 4, 2): ([97.2, 98.7, 96.1, 85.3], 94.3), ("pi0_RTC", 4, 3): ([97.3, 98.5, 96.2, 87.3], 94.8),
    ("pi05_RTC", 1, 1): ([98.4, 94.6, 95.5, 90.4], 94.7), ("pi05_RTC", 2, 2): ([98.5, 97.9, 97.7, 92.2], 96.6),
    ("pi05_RTC", 3, 3): ([98.5, 98.9, 97.9, 93.5], 97.2), ("pi05_RTC", 4, 1): ([98.2, 98.5, 98.2, 92.8], 96.9),
    ("pi05_RTC", 4, 2): ([98.7, 98.3, 97.4, 91.5], 96.5), ("pi05_RTC", 4, 3): ([98.8, 99.1, 97.3, 92.9], 97.0),
}

R = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def welch(m1, s1, m2, s2, n=3):
    t, p = stats.ttest_ind_from_stats(m1, s1, n, m2, s2, n, equal_var=False)
    return float(t), float(p)


def pooled_z(p1, p2, n):
    """두 비율(%)이 각각 n회 평균일 때 두 비율 z 검정(양측)."""
    a, b = p1 / 100, p2 / 100
    pp = (a + b) / 2
    se = math.sqrt(2 * pp * (1 - pp) / n)
    z = (a - b) / se
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


# ---------------------------------------------------------------- V1: 평균 열 ----
def v1_averages():
    rows = []
    for k, (means, sds, avg, _) in {**T1, **T1_MATCHED}.items():
        calc = sum(means) / 4
        rows.append({"row": list(k), "calc": round(calc, 3), "reported": avg, "diff": round(calc - avg, 3), "ok_within_rounding": abs(calc - avg) <= 0.1 + 1e-9})
    for name, v in T1_M0.items():
        calc = sum(v[:4]) / 4
        rows.append({"row": [name, 0, 5], "calc": round(calc, 3), "reported": v[4], "diff": round(calc - v[4], 3), "ok_within_rounding": abs(calc - v[4]) <= 0.1 + 1e-9})
    R["V1_average_column"] = {"rows": rows, "n_rows": len(rows), "n_off": sum(not r["ok_within_rounding"] for r in rows)}


# ---------------------------------------------------------------- V2: 정수 격자 ----
def v2_grid():
    out = {}
    # m=0 행: 시드 하나, 스위트당 500회라면 값은 0.2 %p 격자 위에 있어야 한다
    bad = [(n, v) for n, vs in T1_M0.items() for v in vs[:4] if abs(round(v / 0.2) * 0.2 - v) > 1e-6]
    out["m0_values_off_0.2_grid"] = bad
    # 시드 평균 셀: 1500회 격자(0.0667 %p)는 0.1 %p 표기 간격보다 촘촘해 모든 표기 값이 가능
    out["seed_mean_cells_n1500_step_pct"] = round(100 / 1500, 4)
    # 그림 4a, 4b의 점은 2000회(스위트 4개 x 500) 격자(0.05 %p)에 있어야 한다
    vals = []
    for panel in ("a", "b"):
        for series, d in FIGV["fig4"][panel].items():
            if series.startswith("no_c"):
                continue
            vals += list(d.values())
    off = [v for v in vals if abs(v * 2000 - round(v * 2000)) > 0.35]  # 그림 판독 오차(±0.00017)를 허용
    out["fig4_points_checked"] = len(vals)
    out["fig4_points_off_1/2000_grid(>0.35 count)"] = len(off)
    R["V2_integer_grid"] = out


# ---------------------------------------------------------------- V3: 평균 열 SD ----
def v3_sd_bound():
    rows = []
    for k, (means, sds, avg, avg_sd) in {**T1, **T1_MATCHED}.items():
        upper = sum(sds) / 4            # 스위트 오차가 완전히 같은 방향이면 평균 SD의 상한
        indep = math.sqrt(sum(s * s for s in sds)) / 4  # 서로 독립이면
        rows.append({"row": list(k), "avg_sd": avg_sd, "indep": round(indep, 3), "upper": round(upper, 3),
                     "above_upper": avg_sd > upper + 0.05 + 1e-9})
    R["V3_avg_sd_vs_suite_sd"] = {"rows": rows, "n_above_upper": sum(r["above_upper"] for r in rows)}
    # 스위트 SD가 이항 변동보다 작은 정도 (스위트당 500회, 시드 3개)
    binom = []
    for k, (means, sds, avg, _) in T1.items():
        for m, s in zip(means, sds):
            p = m / 100
            binom.append(s / (100 * math.sqrt(p * (1 - p) / 500)))
    R["V3_sd_ratio_to_binomial_single_seed"] = {"median": round(float(np.median(binom)), 2), "share_below_1": round(float(np.mean(np.array(binom) < 1)), 2),
                                                 "n_cells": len(binom)}


# ---------------------------------------------------------------- V4: REALFAST 대 RTC ----
def v4_compare():
    out = []
    def cmp(a, b, label):
        ma, sa, avga, avgsda = T1[a][0], T1[a][1], T1[a][2], T1[a][3]
        mb, sb, avgb, avgsdb = T1[b][0], T1[b][1], T1[b][2], T1[b][3]
        diff = avga - avgb
        t, p = welch(avga, avgsda, avgb, avgsdb)
        z, pz = pooled_z(avga, avgb, 1500 * 4)
        out.append({"pair": label, "a": list(a), "b": list(b), "diff_pp": round(diff, 2), "welch_t": round(t, 2), "welch_p": p,
                    "pooled_z_n6000": round(z, 2), "pooled_p_n6000": pz})
    for s in (2, 3):
        cmp(("REALFAST", 4, s), ("pi0_RTC", 4, s), f"같은 s={s}: REALFAST 대 π0+RTC")
        cmp(("REALFAST", 4, s), ("pi05_RTC", 4, s), f"같은 s={s}: REALFAST 대 π0.5+RTC")
    # 같은 외부 지연 ε (REALFAST의 s = RTC의 s + 1): ε=1Δt, 2Δt, 3Δt
    for s in (1, 2, 3):
        cmp(("REALFAST", 4, s + 1), ("pi0_RTC", 4, s), f"같은 ε={s}Δt: REALFAST(s={s+1}) 대 π0+RTC(s={s})")
        cmp(("REALFAST", 4, s + 1), ("pi05_RTC", 4, s), f"같은 ε={s}Δt: REALFAST(s={s+1}) 대 π0.5+RTC(s={s})")
    R["V4_realfast_vs_rtc_average"] = out
    # 스위트별 (m=4, s=3): REALFAST 대 π0+RTC
    per = []
    for i, nm in enumerate(SUITES):
        a, b = T1[("REALFAST", 4, 3)], T1[("pi0_RTC", 4, 3)]
        t, p = welch(a[0][i], a[1][i], b[0][i], b[1][i])
        per.append({"suite": nm, "realfast": a[0][i], "pi0_rtc": b[0][i], "diff": round(a[0][i] - b[0][i], 2), "welch_p": p})
    R["V4_per_suite_s3"] = per
    # "consistently surpasses π0": 모든 (REALFAST 행, π0+RTC 행) 쌍에서 평균 열 우위 비율
    wins = sum(T1[("REALFAST", 4, s)][2] > T1[("pi0_RTC", m, ss)][2] for s in (2, 3, 4) for (_, m, ss) in [k for k in T1 if k[0] == "pi0_RTC"])
    tot = 3 * 6
    R["V4_realfast_above_pi0_rtc_pairs"] = {"wins": wins, "total": tot}
    # 같은 방식으로 π0.5
    wins5 = sum(T1[("REALFAST", 4, s)][2] > T1[("pi05_RTC", m, ss)][2] for s in (2, 3, 4) for (_, m, ss) in [k for k in T1 if k[0] == "pi05_RTC"])
    R["V4_realfast_above_pi05_rtc_pairs"] = {"wins": wins5, "total": tot}


# ---------------------------------------------------------------- V5: dm=m 성능 저하 ----
def v5_degeneration():
    out = []
    for model in ("pi0_RTC", "pi05_RTC"):
        for ver in ("v2", "v1"):
            if ver == "v2":
                a = T1[(model, 1, 1)][2]; sa = T1[(model, 1, 1)][3]; b = T1[(model, 4, 1)][2]; sb = T1[(model, 4, 1)][3]
                t, p = welch(b, sb, a, sa)
                out.append({"model": model, "version": ver, "m1s1": a, "m4s1": b, "gap_pp": round(b - a, 2), "welch_p": p})
            else:
                a = T1_V1[(model, 1, 1)][1]; b = T1_V1[(model, 4, 1)][1]
                out.append({"model": model, "version": ver, "m1s1": a, "m4s1": b, "gap_pp": round(b - a, 2), "welch_p": None})
    # v2: dm=m 대 dm<=m 전체 6행 쌍 (같은 s=1,2,3)
    gaps = {}
    for model in ("pi0_RTC", "pi05_RTC"):
        gaps[model] = {s: round(T1[(model, 4, s)][2] - T1[(model, s, s)][2], 2) for s in (1, 2, 3)}
    R["V5_dm_eq_m_degeneration_v2_gap_pp(m4 - m=s)"] = gaps
    R["V5_dm_eq_m_degeneration"] = out


# ---------------------------------------------------------------- V6: 계산량 맞춤 ----
def v6_matched():
    out = []
    for model, base in (("pi0", "pi0_RTC"), ("pi05", "pi05_RTC")):
        for m, s in ((3, 3), (4, 3)):
            a = T1[(base, m, s)]; b = T1_MATCHED[(base + "_30", m, s)]
            t, p = welch(a[2], a[3], b[2], b[3])
            out.append({"model": model, "m": m, "s": s, "avg_5step": a[2], "avg_30step": b[2], "diff": round(b[2] - a[2], 2), "welch_p": p})
    # 같은 계산량 비교의 짝: REALFAST(s=2)와 비교된 행은 s=3 (지연 설정이 다르다)
    R["V6_compute_matched"] = out
    r = T1[("REALFAST", 4, 2)]
    R["V6_realfast_vs_matched"] = [{"vs": k, "diff_pp": round(r[2] - v[2], 2), "welch_p": round(welch(r[2], r[3], v[2], v[3])[1], 3)} for k, v in T1_MATCHED.items()]


# ---------------------------------------------------------------- V7: 표 A, 지연 산술 ----
def v7_latency():
    out = {}
    # 지연 선: 그림 3a 점선 (토큰 수, ms)
    tl = FIGV["fig3"]["token_len_at_latency_lines"]
    pts_lib = [(round(tl["LIBERO_129ms"]), 129), (round(tl["LIBERO_183ms"]), 183)]
    pts_dro = [(round(tl["DROID_183ms"]), 183), (round(tl["DROID_302ms"]), 302)]
    slope_l = (pts_lib[1][1] - pts_lib[0][1]) / (pts_lib[1][0] - pts_lib[0][0])
    slope_d = (pts_dro[1][1] - pts_dro[0][1]) / (pts_dro[1][0] - pts_dro[0][0])
    out["lib_slope_ms_per_token"] = round(slope_l, 2); out["lib_intercept_ms"] = round(pts_lib[0][1] - slope_l * pts_lib[0][0], 1)
    out["droid_slope_ms_per_token"] = round(slope_d, 2); out["droid_intercept_ms"] = round(pts_dro[0][1] - slope_d * pts_dro[0][0], 1)
    out["token_counts_at_lines_read"] = {"LIBERO": [p[0] for p in pts_lib], "DROID": [p[0] for p in pts_dro]}
    # 표 A = 본문 상한 + N=4 추가 지연 (각주 3: 11토큰 +7 ms, 20토큰 +13 ms)
    out["tableA_realfast_libero"] = {"text_183_plus_N4_7": 183 + 7, "tableA": 190}
    out["tableA_realfast_droid"] = {"text_302_plus_N4_13": 302 + 13, "tableA": 315}
    out["delta_tilde_droid_N4"] = {"315_plus_eps9": 315 + 9, "text_324": 324}
    # d_m 계산과 s의 대응
    dm = {}
    for eps in (100, 200, 300):
        dm[f"eps={eps}ms"] = {"pi0_RTC(70)": math.floor((70 + eps) / 100), "pi05_RTC(74)": math.floor((74 + eps) / 100), "REALFAST(190)": math.floor((190 + eps) / 100)}
    out["libero_dm_by_eps"] = dm
    out["droid_dm_realfast_N4"] = math.floor((315 + 9) / 66.7)
    out["droid_dm_realfast_N1"] = math.floor((302 + 9) / 66.7)
    out["droid_dm_pi0_rtc"] = math.floor((70 + 9) / 66.7); out["droid_dm_pi05_rtc"] = math.floor((74 + 9) / 66.7)
    out["droid_trial_timesteps_40s_at_66.7ms"] = round(40 / 0.0667)
    # Jetson Orin: δ(1) ≈ 1035 ms (그림 7a 오른쪽 막대 판독), Δt = 66.7 ms
    out["jetson_dm_if_delta1_1035ms"] = math.floor((1035 + 9) / 66.7)
    # 그림 7a의 지연 증가율로 N=2, 4 추가 지연 예측 (RTX 4090 선: N=64에서 0.592, N=256에서 2.491)
    r4090 = {p["N"]: p["rel_overhead"] for p in FIGV["fig7a"]["RTX 4090"]}
    per_traj = (r4090[256] - r4090[64]) / (256 - 64)
    out["rtx4090_overhead_per_extra_trajectory_frac"] = round(per_traj, 5)
    out["predicted_extra_ms_N2_N4_for_183ms"] = [round(183 * per_traj * (n - 1), 1) for n in (2, 4)]
    out["predicted_extra_ms_N2_N4_for_302ms"] = [round(302 * per_traj * (n - 1), 1) for n in (2, 4)]
    out["reported_extra_ms"] = {"11_tokens_N2_N4": [4, 7], "20_tokens_N2_N4": [8, 13]}
    R["V7_latency_arithmetic"] = out


# ---------------------------------------------------------------- V8: DROID 그림 5, 그림 C ----
def v8_droid():
    out = {}
    C = FIGV["figC_success"]
    sync = {k: [t["sync"] for t in v] for k, v in C.items()}
    rt = {k: [t["realtime"] for t in v] for k, v in C.items()}
    k80 = {
        "pi0_sync": round(sum(sync["pi0"]) * 10), "pi0_rtc": round(sum(rt["pi0"]) * 10),
        "pi05_sync": round(sum(sync["pi05"]) * 10), "pi05_rtc_translucent": round(sum(rt["pi05"]) * 10),
        "pi0fast_sync": round(sum(sync["pi0fast"]) * 10), "realfast": round(sum(rt["pi0fast"]) * 10),
    }
    out["counts_of_80_from_figC"] = k80
    f5 = FIGV["fig5"]
    left = {x: v for x, v, o in f5["bars_left_x_value_opacity"]}
    out["fig5_bar_heights_x80"] = {"pi0_sync": round(left[135.4] * 80, 2), "pi05_sync": round(left[158.8] * 80, 2), "pi0fast_sync": round(left[182.2] * 80, 2),
                                    "pi0_rt": round(left[223.2] * 80, 2), "pi05_rt_opaque": round(left[246.7] * 80, 2), "realfast": round(f5["hatched_left"][0][1] * 80, 2)}
    out["pi05_rt_translucent_top_x80"] = 55  # 그림 5 반투명 막대 높이 0.688 (아래 문서 참조)
    out["pi05_rt_opaque_equals_pi05_sync"] = abs(left[246.7] - left[158.8]) < 1e-3
    # Wilson 95% (과제 구분 없이 80회 풀링)
    out["wilson95"] = {k: [round(x, 3) for x in wilson(v, 80)] for k, v in k80.items()}
    # Fisher 정확 검정 (80회 풀링, 시행 독립 가정)
    def fisher(a, b):
        return float(stats.fisher_exact([[a, 80 - a], [b, 80 - b]])[1])
    out["fisher_pooled_80"] = {
        "realfast_vs_pi0fast_sync": fisher(k80["realfast"], k80["pi0fast_sync"]),
        "realfast_vs_pi0_rtc": fisher(k80["realfast"], k80["pi0_rtc"]),
        "realfast_vs_pi05_rtc_translucent": fisher(k80["realfast"], k80["pi05_rtc_translucent"]),
        "realfast_vs_pi05_rtc_opaque": fisher(k80["realfast"], 45),
        "pi05_rtc_translucent_vs_pi05_sync": fisher(k80["pi05_rtc_translucent"], k80["pi05_sync"]),
        "pi0fast_sync_vs_pi0_sync": fisher(k80["pi0fast_sync"], k80["pi0_sync"]),
        "pi0fast_sync_vs_pi05_sync": fisher(k80["pi0fast_sync"], k80["pi05_sync"]),
        "pi0_rtc_vs_pi0_sync": fisher(k80["pi0_rtc"], k80["pi0_sync"]),
    }
    # 과제 단위 쌍 비교 (같은 8과제): 부호 검정과 Wilcoxon, 과제 층별 Cochran-Mantel-Haenszel(10회 가정)
    def paired(a, b):
        d = np.array(a) - np.array(b)
        w, l = int((d > 1e-9).sum()), int((d < -1e-9).sum())
        n = w + l
        p_sign = float(stats.binomtest(min(w, l), n, 0.5).pvalue) if n else 1.0
        try:
            pw = float(stats.wilcoxon(d[np.abs(d) > 1e-9]).pvalue) if n else 1.0
        except ValueError:
            pw = 1.0
        # CMH: 2x2 표 8개 (성공 k/10)
        num = 0.0; var = 0.0; sw = 0.0; sl = 0.0
        for x, y in zip(a, b):
            ka, kb = round(x * 10), round(y * 10)
            n1 = n2 = 10; tot = n1 + n2; kk = ka + kb
            num += ka - n1 * kk / tot
            var += n1 * n2 * kk * (tot - kk) / (tot ** 2 * (tot - 1))
            sw += ka * (n2 - kb) / tot; sl += kb * (n1 - ka) / tot
        chi = (abs(num) - 0.5) ** 2 / var if var else 0.0
        return {"wins": w, "losses": l, "ties": 8 - n, "sign_p": round(p_sign, 4), "wilcoxon_p": round(pw, 4), "cmh_chi2": round(chi, 2),
                "cmh_p": float(stats.chi2.sf(chi, 1)), "mh_odds_ratio": round(sw / sl, 2) if sl else None}
    out["paired_tasks"] = {
        "realfast_vs_pi0fast_sync": paired(rt["pi0fast"], sync["pi0fast"]),
        "realfast_vs_pi0_rtc": paired(rt["pi0fast"], rt["pi0"]),
        "realfast_vs_pi05_rtc_translucent": paired(rt["pi0fast"], rt["pi05"]),
        "pi05_rtc_vs_pi05_sync": paired(rt["pi05"], sync["pi05"]),
        "pi0_rtc_vs_pi0_sync": paired(rt["pi0"], sync["pi0"]),
        "pi0fast_sync_vs_pi0_sync": paired(sync["pi0fast"], sync["pi0"]),
    }
    # 과제 단위 부트스트랩 90% 구간 (그림 5의 오차 막대와 비교)
    rng = np.random.default_rng(0)
    def boot(v):
        v = np.array(v)
        m = rng.choice(v, size=(20000, 8)).mean(axis=1)
        return [round(float(np.percentile(m, 5)), 3), round(float(np.percentile(m, 95)), 3)]
    out["task_bootstrap90"] = {"realfast": boot(rt["pi0fast"]), "pi0fast_sync": boot(sync["pi0fast"]), "pi05_rtc_translucent": boot(rt["pi05"]),
                               "pi0_rtc": boot(rt["pi0"]), "pi05_sync": boot(sync["pi05"]), "pi0_sync": boot(sync["pi0"])}
    # 그림 5 오차 막대(90%)를 y좌표에서 읽은 값 (쪽 8, 막대 선 길이)
    cal = lambda y: (167.3 - y) / 88.0
    bars_px = {"pi0_sync": (120.5, 159.0), "pi05_sync": (96.2, 139.3), "pi0fast_sync": (119.0, 142.9), "pi0_rt": (114.9, 158.0), "pi05_rt_translucent": (85.8, 127.8), "realfast": (95.6, 117.9)}
    out["fig5_errorbar_90_read"] = {k: [round(cal(b), 3), round(cal(a), 3)] for k, (a, b) in bars_px.items()}
    out["task_means_vectors"] = {"sync": sync, "realtime": rt}
    # 처리율(그림 5 오른쪽): 막대 높이에서 읽은 평균 처리율과 성공 평균으로 만든 "성공 시 평균 소요 시간"
    thr = {"pi0_sync": 0.1509, "pi05_sync": 0.3357, "pi0fast_sync": 0.2108, "pi0_rt": 0.2262, "pi05_rt_translucent": 0.461, "realfast": 0.4337}
    succ = {"pi0_sync": 25 / 80, "pi05_sync": 45 / 80, "pi0fast_sync": 33 / 80, "pi0_rt": 28 / 80, "pi05_rt_translucent": 55 / 80, "realfast": 55 / 80}
    out["mean_time_to_success_s_given_success(40s*(1-thr/succ))"] = {k: round(40 * (1 - thr[k] / succ[k]), 1) for k in thr}
    R["V8_droid"] = out


# ---------------------------------------------------------------- V9: 그림 7b ----
def v9_dynamic():
    ks = {"pi0": 11, "pi05": 14, "pi0_FAST": 6, "pi0_RTC": 15, "pi05_RTC": 16, "REALFAST": 16}
    out = {"pct_check": {k: v / 20 * 100 for k, v in ks.items()}}
    out["wilson95"] = {k: [round(x, 3) for x in wilson(v, 20)] for k, v in ks.items()}
    f = lambda a, b: float(stats.fisher_exact([[a, 20 - a], [b, 20 - b]])[1])
    out["fisher"] = {"REALFAST_vs_pi0_FAST": f(16, 6), "REALFAST_vs_pi0_RTC": f(16, 15), "REALFAST_vs_pi05_RTC": f(16, 16), "pi0_RTC_vs_pi0": f(15, 11),
                     "pi05_RTC_vs_pi05": f(16, 14), "pi0_FAST_vs_pi0": f(6, 11)}
    # 20회로 80% 검정력에 필요한 차이 (p1 = 0.8 기준, Fisher 대신 정규 근사)
    out["min_detectable_diff_n20_80power_at_p0.75_normal_approx"] = round((1.96 + 0.8416) * math.sqrt(2 * 0.75 * 0.25 / 20), 3)
    R["V9_dynamic_task_fig7b"] = out


# ---------------------------------------------------------------- V10: 그림 4 ----
def v10_ablation():
    a, b = FIGV["fig4"]["a"], FIGV["fig4"]["b"]
    out = {}
    out["fig4a_no_constrained_gap_pp"] = {k: round((a["N=4"][k] - a["no_constrained_N=4"][k]) * 100, 2) for k in a["N=4"]}
    out["fig4a_N4_vs_N1_pp"] = {k: round((a["N=4"][k] - a["N=1"][k]) * 100, 2) for k in a["N=1"]}
    taus = sorted(b["N=1"].keys(), key=float)
    n1 = np.array([b["N=1"][t] for t in taus]); n2 = np.array([b["N=2"][t] for t in taus]); n4 = np.array([b["N=4"][t] for t in taus])
    out["fig4b_mean_over_tau"] = {"N=1": round(float(n1.mean()), 4), "N=2": round(float(n2.mean()), 4), "N=4": round(float(n4.mean()), 4)}
    out["fig4b_range_over_tau_pp"] = {"N=1": round(float(np.ptp(n1) * 100), 2), "N=2": round(float(np.ptp(n2) * 100), 2), "N=4": round(float(np.ptp(n4) * 100), 2)}
    d = (n4 - n1) * 100
    out["fig4b_N4_minus_N1_pp_by_tau"] = [round(float(x), 2) for x in d]
    out["fig4b_N4_above_N1_count"] = int((d > 0).sum())
    out["fig4b_sign_test_p"] = round(float(stats.binomtest(int((d > 0).sum()), int((d != 0).sum()), 0.5).pvalue), 4)
    out["fig4b_paired_t_p"] = round(float(stats.ttest_rel(n4, n1).pvalue), 4)
    out["fig4b_N4_peak"] = {"tau": taus[int(n4.argmax())], "value": float(n4.max())}
    # 단일 평가(2000회)의 이항 표준오차
    p = 0.955
    out["binomial_se_single_eval_n2000_pp"] = round(100 * math.sqrt(p * (1 - p) / 2000), 2)
    # 선택 편향: 평균이 같은 11개 독립 평가의 최댓값이 평균보다 얼마나 높은가 (모의)
    rng = np.random.default_rng(1)
    sims = rng.binomial(2000, 0.9577, size=(20000, 11)) / 2000
    out["expected_max_minus_mean_of_11_iid_pp"] = round(float((sims.max(axis=1) - 0.9577).mean() * 100), 2)
    out["table1_realfast_avg_vs_fig4b_peak_pp"] = {"table1_s2_s3_s4": [T1[("REALFAST", 4, s)][2] for s in (2, 3, 4)], "fig4b_N4_tau0.3": round(float(b["N=4"]["0.3"]) * 100, 2)}
    R["V10_ablation_fig4"] = out


# ---------------------------------------------------------------- V11: 그림 3b Kinetix ----
def v11_kinetix():
    k = FIGV["fig3"]["kinetix"]
    out = {"means": {kk: v["mean"] for kk, v in k.items()}}
    out["dm_effect_at_m4_pp"] = {f"dm={d}": round(k[f"dm={d},m=4"]["mean"] * 100, 1) for d in (1, 2, 3, 4)}
    out["m_effect_at_dm1_pp"] = round((k["dm=1,m=1"]["mean"] - k["dm=1,m=4"]["mean"]) * 100, 1)
    out["m_effect_at_dm2_pp"] = round((k["dm=2,m=2"]["mean"] - k["dm=2,m=4"]["mean"]) * 100, 1)
    out["dm_effect_dm1_to_dm4_at_m4_pp"] = round((k["dm=1,m=4"]["mean"] - k["dm=4,m=4"]["mean"]) * 100, 1)
    out["diag_dm_eq_m"] = {f"m=dm={d}": k[f"dm={d},m={d}"]["mean"] for d in (1, 2, 3, 4)}
    out["ci90_halfwidth_dm4_pp"] = round((k["dm=4,m=4"]["hi90"] - k["dm=4,m=4"]["lo90"]) / 2 * 100, 1)
    # 구간이 겹치는지: m=1..4 @ dm=1
    out["dm1_m1_vs_m4_ci_overlap"] = k["dm=1,m=1"]["lo90"] < k["dm=1,m=4"]["hi90"]
    R["V11_kinetix_fig3b"] = out


# ---------------------------------------------------------------- V12: v1 대 v2 ----
def v12_versions():
    rows = []
    for k, (means, avg) in T1_V1.items():
        m2, _, a2, _ = T1[k]
        rows.append({"row": list(k), "avg_v1": avg, "avg_v2": a2, "diff": round(a2 - avg, 1), "long_v1": means[3], "long_v2": m2[3], "object_v1": means[1], "object_v2": m2[1]})
    d = np.array([r["diff"] for r in rows])
    out = {"rows": rows, "mean_change_pp": round(float(d.mean()), 2), "max_change_pp": float(d.max()), "min_change_pp": float(d.min()), "n_rows_changed": int((d != 0).sum()), "n_rows": len(rows)}
    # REALFAST 행 동일성 (v1에서 읽은 값과 같음; 문서에 적은 대로)
    out["realfast_rows_identical"] = True
    # v1 기준 REALFAST - π0 격차 (같은 s)
    out["gap_realfast_minus_pi0_same_s_v1"] = {s: round(T1[("REALFAST", 4, s)][2] - T1_V1[("pi0_RTC", 4, s)][1], 1) for s in (2, 3)}
    out["gap_realfast_minus_pi0_same_s_v2"] = {s: round(T1[("REALFAST", 4, s)][2] - T1[("pi0_RTC", 4, s)][2], 1) for s in (2, 3)}
    R["V12_v1_vs_v2_table1"] = out


# ---------------------------------------------------------------- V13: 계산량, 단계 산술 ----
def v13_misc():
    out = {}
    out["libero_steps_x_batch"] = 6000 * 256
    out["libero_samples_per_epoch_if_6_epochs"] = round(6000 * 256 / 6)
    out["droid_steps_x_batch"] = 300000 * 256
    out["droid_samples_per_epoch_if_3_epochs"] = round(300000 * 256 / 3)
    # 39 GFLOPs 대략 추정 (가정: 언어 모델 비임베딩 매개변수 2.0e9, π0 행동 전문가 0.3e9, 행동 길이 H는 미기재)
    P_lm, P_ae = 2.0e9, 0.3e9
    out["realfast_decode_gflops_N4_11tok(2*P*tokens)"] = round(2 * P_lm * 11 * 4 / 1e9, 1)
    out["realfast_decode_gflops_N1_11tok"] = round(2 * P_lm * 11 / 1e9, 1)
    out["pi0_30steps_gflops_for_H"] = {f"H={h}": round(2 * P_ae * h * 30 / 1e9, 1) for h in (2, 4, 8, 10, 50)}
    out["pi0_H_that_gives_39GFLOPs_30steps"] = round(39e9 / (2 * P_ae * 30), 2)
    # REALFAST 학습에서 접두 조건 입력 길이: 11 토큰 상한과 m=4 접두 토큰 평균 약 7
    out["token_budget_B_vs_mean"] = {"LIBERO": [11, 7], "DROID": [20, 11]}
    # m * Δt = 400 ms 확인
    out["m_dt_ms"] = {"LIBERO": 4 * 100, "DROID": round(6 * 66.7, 1)}
    R["V13_misc"] = out


# ---------------------------------------------------------------- V14: 토큰 수 분포와 B ----
def v14_coverage():
    out = {}
    for name, key, B in (("LIBERO", "hist_LIBERO", 11), ("DROID", "hist_DROID", 20)):
        h = {int(k): v for k, v in FIGV["fig3"][key].items()}
        tot = sum(h.values())
        mean = sum(t * v for t, v in h.items()) / tot
        cov = sum(v for t, v in h.items() if t <= B) / tot
        out[name] = {"mean_tokens_from_histogram": round(mean, 2), "coverage_at_B": round(cov, 4), "share_over_B": round(1 - cov, 4), "B": B,
                     "p_at_least_one_chunk_over_B_in_100_inferences": round(1 - cov ** 100, 3)}
    R["V14_token_budget_coverage"] = out


def sig(o):
    """작은 실수는 유효숫자 3자리로 남긴다 (p 값이 0.0으로 잘리지 않게)."""
    if isinstance(o, float):
        return float(f"{o:.3g}") if abs(o) < 0.01 and o != 0 else o
    if isinstance(o, dict):
        return {k: sig(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sig(v) for v in o]
    return o


def main():
    v1_averages(); v2_grid(); v3_sd_bound(); v4_compare(); v5_degeneration(); v6_matched(); v7_latency(); v8_droid(); v9_dynamic()
    v10_ablation(); v11_kinetix(); v12_versions(); v13_misc(); v14_coverage()
    global R
    R = sig(R)
    (ROOT / "analysis" / "recompute_results.json").write_text(json.dumps(R, ensure_ascii=False, indent=1))
    print("V1 평균 열: 불일치 행", R["V1_average_column"]["n_off"], "/", R["V1_average_column"]["n_rows"])
    print("V3 평균 SD가 상한 초과:", R["V3_avg_sd_vs_suite_sd"]["n_above_upper"], " SD/이항 SD 중앙값", R["V3_sd_ratio_to_binomial_single_seed"])
    for r in R["V4_realfast_vs_rtc_average"]:
        print("V4", r["pair"], r["diff_pp"], "Welch p", r["welch_p"], "pooled p", r["pooled_p_n6000"])
    print("V4 per suite s3", R["V4_per_suite_s3"])
    print("V4 wins", R["V4_realfast_above_pi0_rtc_pairs"], R["V4_realfast_above_pi05_rtc_pairs"])
    print("V5", R["V5_dm_eq_m_degeneration"], R["V5_dm_eq_m_degeneration_v2_gap_pp(m4 - m=s)"])
    print("V6", R["V6_compute_matched"], R["V6_realfast_vs_matched"])
    print("V7", json.dumps(R["V7_latency_arithmetic"], ensure_ascii=False))
    v8 = dict(R["V8_droid"]); v8.pop("task_means_vectors")
    print("V8", json.dumps(v8, ensure_ascii=False))
    print("V9", R["V9_dynamic_task_fig7b"])
    print("V10", R["V10_ablation_fig4"])
    print("V11", R["V11_kinetix_fig3b"])
    print("V12", {k: v for k, v in R["V12_v1_vs_v2_table1"].items() if k != "rows"})
    print("V13", R["V13_misc"])
    print("V14", R["V14_token_budget_coverage"])


if __name__ == "__main__":
    main()
