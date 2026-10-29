"""Table 1–6·본문·Fig. 1(b) 수치 재계산 (CPU). 실행: python3 analysis/table_check.py

- C1: Table 5(24개 과제 평균) = Table 1 PnP 8개 · Others 16개의 가중 평균인지 (과제 구성은 GR00T 저장소 RoboCasa 표)
- C2: ES를 SR·TPR·t_max만으로 다시 계산 (식 7·8, 성공 궤적 길이는 조화평균으로 근사)
- C3: ES 분해 — 실패 항이 차지하는 몫, 성공률만 바뀐 경우와 속도만 바뀐 경우
- C4: 본문의 실물 수치 주장(37–45%, 33–39%, 28.2·21.9%p, 1.75배, 최대 1.8배)
- C5: Fig. 1(b) 다섯 점 = 6개 실물 과제의 단순 평균인지 (그림 판독값과 비교)
- C6: 시행 수 격자 — 표의 성공률이 k/n 꼴로 나오는 n
- C7: Wilson 95% 구간과 Fisher 정확검정 (비대응 근사)
- C8: "best/second-best" 주장 — 가속 방법 중 PaceVLA가 TPR·ES 1위인 칸의 수
- C9: 실물 시간비로 본 양자화 모드 사용 비율의 상한 추정, 고정 양자화와의 비교
- C10: 추론 지연 1.05배, LIBERO 양자화 비율
"""
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
BASES = ("RLDX-1-IMG", "GR00T N1.5")
METHODS = ("base", "Fixed Quantization", "DemoSpeedUp", "SAIL", "PaceVLA")

# Table 1: 열 = (PnP, Others, LIBERO Short, LIBERO Long), 값 = (SR %, TPR ×1e-3, ES)
T1 = {
    "RLDX-1-IMG": {
        "base": [(48.5, 1.10, 1.00), (76.3, 2.73, 1.00), (96.7, 7.93, 1.00), (88.6, 3.15, 1.00)],
        "Fixed Quantization": [(43.0, 1.55, 0.87), (72.0, 4.00, 0.97), (89.4, 11.80, 1.23), (79.2, 4.44, 1.10)],
        "DemoSpeedUp": [(28.7, 0.80, 0.49), (62.7, 3.37, 0.68), (93.5, 14.03, 1.54), (75.2, 4.95, 1.06)],
        "SAIL": [(38.5, 1.03, 0.72), (66.6, 2.80, 0.75), (96.5, 10.34, 1.27), (76.6, 3.38, 0.92)],
        "PaceVLA": [(45.0, 1.54, 0.93), (76.9, 3.21, 1.07), (97.7, 10.36, 1.29), (90.2, 4.44, 1.31)],
    },
    "GR00T N1.5": {
        "base": [(52.3, 1.05, 1.00), (72.4, 2.20, 1.00), (97.8, 8.36, 1.00), (89.2, 3.12, 1.00)],
        "Fixed Quantization": [(49.0, 1.56, 0.96), (63.1, 2.92, 0.81), (91.2, 12.41, 1.25), (82.0, 4.78, 1.21)],
        "DemoSpeedUp": [(40.3, 1.15, 0.71), (57.3, 2.77, 0.67), (87.7, 12.09, 1.14), (71.6, 3.91, 0.90)],
        "SAIL": [(32.5, 0.77, 0.52), (58.5, 2.31, 0.68), (95.5, 10.85, 1.23), (79.0, 3.89, 1.03)],
        "PaceVLA": [(55.0, 1.64, 1.16), (72.9, 3.30, 1.14), (97.1, 10.95, 1.27), (89.2, 4.27, 1.26)],
    },
}
COLS = ("RoboCasa PnP", "RoboCasa Others", "LIBERO Short", "LIBERO Long")
# 부록 B: t_max. LIBERO Short는 Spatial/Object/Goal = 220/280/300 세 suite 평균이라 근사만 가능
TMAX = {"RoboCasa PnP": 1500, "RoboCasa Others": 1500, "LIBERO Short": (220, 280, 300), "LIBERO Long": 520}
# Table 3 DexJoCo (GR00T N1.5, t_max 1500)
T3 = {"base": (76.3, 1.83, 1.00), "Fixed Quantization": (63.0, 1.65, 0.70), "DemoSpeedUp": (65.7, 1.75, 0.76),
      "SAIL": (60.3, 1.58, 0.65), "PaceVLA": (77.3, 2.22, 1.11)}
# Table 5 (GR00T N1.5, RoboCasa 24개 평균)
T5 = {"w/o Mode-specific Decoders": (63.3, 2.44, 1.00), "w/o Quantization Judge": (65.1, 2.01, 1.01),
      "w/o KL": (63.7, 2.04, 0.98), "w/o Balance": (64.8, 2.38, 1.04), "Attn. Pool": (64.6, 2.38, 1.04),
      "Last Token": (64.2, 2.43, 1.03), "PaceVLA": (66.9, 2.75, 1.14)}
# Table 2 (SR %, Time s): Pick-and-Place, Air Fryer, Stack Cup, Table Sweeping, Pump and Scrub
REAL_TASKS = ("Pick-and-Place", "Air Fryer", "Stack Cup", "Table Sweeping", "Pump and Scrub")
T2 = {
    "base": [(81.3, 12.1), (81.3, 24.6), (84.4, 35.3), (62.5, 21.9), (64.1, 30.0)],
    "Fixed Quantization": [(62.5, 7.7), (53.1, 14.2), (62.5, 18.7), (48.4, 13.3), (42.2, 19.9)],
    "DemoSpeedUp": [(68.8, 8.1), (71.9, 18.9), (53.1, 24.2), (56.3, 17.9), (53.1, 20.2)],
    "SAIL": [(65.6, 7.9), (56.3, 16.8), (45.3, 20.0), (46.9, 15.4), (45.3, 15.3)],
    "PaceVLA": [(84.4, 7.6), (81.3, 14.6), (81.3, 19.4), (59.4, 14.6), (65.6, 18.2)],
}
PARTIAL = {"Stack Cup", "Table Sweeping", "Pump and Scrub"}  # 0.5점 부분 점수 (Table 2 캡션)
# Table 4 ALLEX Conveyor (SR %, packages/min), 5 trial × 10 packages
T4 = {"base": (86.0, 3.6), "Fixed Quantization": (72.0, 4.2), "DemoSpeedUp": (86.0, 4.7), "SAIL": (78.0, 4.3),
      "PaceVLA": (94.0, 6.3)}
# Fig. 1(b) 판독값 (400 dpi 판독, x: 1.0×=160 px · 0.1× ≈ 82.75 px, y: 85%=42 px · 1% ≈ 16.66 px)
FIG1B_PX = {"base": (160, 181), "PaceVLA": (729, 163), "DemoSpeedUp": (482, 377),
            "Fixed Quantization": (659, 512), "SAIL": (633, 521)}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def harmonic_len(sr, tpr, tmax):
    """식 7: TPR = SR·E[1/t | 성공] − (1−SR)/t_max → 성공 궤적 길이의 조화평균."""
    s = sr / 100
    return s / (tpr * 1e-3 + (1 - s) / tmax)


def T_per_success(sr, L, tmax):
    s = sr / 100
    return L + (1 - s) / s * tmax


def es_hat(row, base, tmax):
    if isinstance(tmax, tuple):
        tmax = sum(tmax) / len(tmax)
    Lb, L = harmonic_len(base[0], base[1], tmax), harmonic_len(row[0], row[1], tmax)
    return T_per_success(base[0], Lb, tmax) / T_per_success(row[0], L, tmax), L, Lb, tmax


def main() -> None:
    out: dict = {}

    # C1 가중 평균 (PnP 8 · Others 16)
    c1 = {}
    for name, vals in (("GR00T N1.5", T1["GR00T N1.5"]),):
        for m in METHODS:
            pnp, oth = vals[m][0], vals[m][1]
            c1[f"{name}/{m}"] = {"SR_24": round((8 * pnp[0] + 16 * oth[0]) / 24, 2),
                                 "TPR_24": round((8 * pnp[1] + 16 * oth[1]) / 24, 3)}
    c1["Table5_PaceVLA"] = {"SR": T5["PaceVLA"][0], "TPR": T5["PaceVLA"][1]}
    c1["match"] = (abs(c1["GR00T N1.5/PaceVLA"]["SR_24"] - 66.9) <= 0.05
                   and abs(c1["GR00T N1.5/PaceVLA"]["TPR_24"] - 2.75) <= 0.005)
    # 24개 풀링 ES(조화평균 근사): PnP·Others를 과제 수로 가중한 성공당 기대 step
    def pooled_T(m):
        tot_s, tot_steps = 0.0, 0.0
        for idx, ntask in ((0, 8), (1, 16)):
            sr, tpr, _ = T1["GR00T N1.5"][m][idx]
            L = harmonic_len(sr, tpr, 1500)
            s = sr / 100
            tot_s += ntask * s
            tot_steps += ntask * (s * L + (1 - s) * 1500)
        return tot_steps / tot_s
    c1["pooled_ES_hat_PaceVLA_vs_base"] = round(pooled_T("base") / pooled_T("PaceVLA"), 3)
    c1["Table5_ES"] = 1.14
    out["C1_table5_weighted_average"] = c1

    # C2 ES 재계산
    c2 = {}
    for b in BASES:
        for m in METHODS[1:]:
            for j, col in enumerate(COLS):
                e, L, Lb, tm = es_hat(T1[b][m][j], T1[b]["base"][j], TMAX[col])
                c2[f"{b}/{m}/{col}"] = {"reported": T1[b][m][j][2], "recomputed": round(e, 3),
                                         "diff": round(e - T1[b][m][j][2], 3)}
    for m in METHODS[1:]:
        e, *_ = es_hat(T3[m], T3["base"], 1500)
        c2[f"GR00T N1.5/{m}/DexJoCo"] = {"reported": T3[m][2], "recomputed": round(e, 3), "diff": round(e - T3[m][2], 3)}
    diffs = [abs(v["diff"]) for k, v in c2.items() if "LIBERO Short" not in k]
    out["C2_es_recomputed"] = {"rows": c2, "max_abs_diff_excl_LIBERO_Short": round(max(diffs), 3),
                               "n_within_0.03": sum(d <= 0.03 for d in diffs), "n": len(diffs),
                               "note": "LIBERO Short는 t_max가 suite마다 달라(220/280/300) 평균 t_max로 근사"}

    # C3 ES 분해 (GR00T N1.5 + PaceVLA)
    c3 = {}
    for j, col in enumerate(COLS[:2] + (COLS[3],)):
        jj = COLS.index(col)
        base, row, tm = T1["GR00T N1.5"]["base"][jj], T1["GR00T N1.5"]["PaceVLA"][jj], TMAX[col]
        Lb, L = harmonic_len(base[0], base[1], tm), harmonic_len(row[0], row[1], tm)
        Tb = T_per_success(base[0], Lb, tm)
        c3[col] = {
            "base_L_steps": round(Lb), "PaceVLA_L_steps": round(L),
            "failure_share_of_base_T": round(1 - Lb / Tb, 3),
            "ES_SR_change_only": round(Tb / T_per_success(row[0], Lb, tm), 3),
            "ES_speed_change_only": round(Tb / T_per_success(base[0], L, tm), 3),
            "ES_both": round(Tb / T_per_success(row[0], L, tm), 3),
            "step_ratio_success_only": round(Lb / L, 3),
        }
    base, row = T3["base"], T3["PaceVLA"]
    Lb, L = harmonic_len(base[0], base[1], 1500), harmonic_len(row[0], row[1], 1500)
    Tb = T_per_success(base[0], Lb, 1500)
    c3["DexJoCo"] = {"base_L_steps": round(Lb), "PaceVLA_L_steps": round(L),
                     "failure_share_of_base_T": round(1 - Lb / Tb, 3),
                     "ES_SR_change_only": round(Tb / T_per_success(row[0], Lb, 1500), 3),
                     "ES_speed_change_only": round(Tb / T_per_success(base[0], L, 1500), 3),
                     "step_ratio_success_only": round(Lb / L, 3)}
    out["C3_es_decomposition"] = c3

    # C4 실물 수치 주장
    b, p, fq = T2["base"], T2["PaceVLA"], T2["Fixed Quantization"]
    red = {t: round(100 * (1 - p[i][1] / b[i][1]), 1) for i, t in enumerate(REAL_TASKS)}
    out["C4_real_claims"] = {
        "time_reduction_pct": red,
        "claim_franka_37_45": [red["Pick-and-Place"], red["Air Fryer"], red["Stack Cup"]],
        "claim_openarm_33_39": [red["Table Sweeping"], red["Pump and Scrub"]],
        "fixedQ_SR_drop_airfryer_stackcup": [round(b[1][0] - fq[1][0], 1), round(b[2][0] - fq[2][0], 1)],
        "conveyor_throughput_ratio": round(T4["PaceVLA"][1] / T4["base"][1], 3),
        "max_time_ratio": max(round(b[i][1] / p[i][1], 3) for i in range(5)),
        "time_ratios": {t: round(b[i][1] / p[i][1], 3) for i, t in enumerate(REAL_TASKS)},
    }

    # C5 Fig. 1(b)
    c5 = {}
    for m in METHODS:
        srs = [T2[m][i][0] for i in range(5)] + [T4[m][0]]
        sps = [T2["base"][i][1] / T2[m][i][1] for i in range(5)] + [T4[m][1] / T4["base"][1]]
        px, py = FIG1B_PX[m]
        c5[m] = {"mean_SR": round(sum(srs) / 6, 2), "mean_speedup": round(sum(sps) / 6, 3),
                 "fig_SR": round(85 - (py - 42) / 16.66, 2), "fig_speedup": round(1 + (px - 160) / 827.5, 3)}
    out["C5_fig1b_means"] = c5

    # C6 시행 수 격자
    def fits(v, n):
        k = round(v / 100 * n)
        return abs(round(100 * k / n, 1) - v) < 0.051 or abs(100 * k / n - v) < 0.051
    def min_n(values, mult=1, upto=200):
        return next((n for n in range(1, upto + 1) if all(fits(v, n * mult) for v in values)), None)
    pnp = [T1[b][m][0][0] for b in BASES for m in METHODS]
    oth = [T1[b][m][1][0] for b in BASES for m in METHODS]
    dex = [T3[m][0] for m in METHODS]
    out["C6_trial_lattice"] = {
        "RoboCasa_PnP_min_trials_per_task(8 tasks)": min_n(pnp, 8),
        "RoboCasa_Others_min_trials_per_task(16 tasks)": min_n(oth, 16),
        "DexJoCo_min_trials_per_task(6 tasks)": min_n(dex, 6),
        "RoboCasa_PnP_fits_50": all(fits(v, 400) for v in pnp),
        "RoboCasa_Others_fits_50": all(fits(v, 800) for v in oth),
        "DexJoCo_fits_50": all(fits(v, 300) for v in dex),
        "real_binary_tasks_fit_32": all(fits(T2[m][i][0], 32) for m in METHODS for i in (0, 1)),
        "real_partial_tasks_fit_64": all(fits(T2[m][i][0], 64) for m in METHODS for i in (2, 3, 4)),
        "conveyor_fits_50": all(fits(T4[m][0], 50) for m in METHODS),
    }

    # C7 신뢰구간과 검정 (RoboCasa 과제당 50회, DexJoCo 과제당 50회 가정 — C6에서 정합)
    def cmp(sr_a, sr_b, n):
        ka, kb = round(sr_a / 100 * n), round(sr_b / 100 * n)
        _, pv = stats.fisher_exact([[ka, n - ka], [kb, n - kb]])
        return {"a": [ka, n, wilson(ka, n)], "b": [kb, n, wilson(kb, n)], "fisher_p": float(f"{pv:.3g}")}
    g = T1["GR00T N1.5"]
    out["C7_tests"] = {
        "GR00T RoboCasa PnP PaceVLA vs base (n=400)": cmp(g["PaceVLA"][0][0], g["base"][0][0], 400),
        "GR00T RoboCasa Others PaceVLA vs base (n=800)": cmp(g["PaceVLA"][1][0], g["base"][1][0], 800),
        "GR00T RoboCasa Others PaceVLA vs Fixed (n=800)": cmp(g["PaceVLA"][1][0], g["Fixed Quantization"][1][0], 800),
        "DexJoCo PaceVLA vs base (n=300)": cmp(T3["PaceVLA"][0], T3["base"][0], 300),
        "Air Fryer PaceVLA vs Fixed (n=32)": cmp(T2["PaceVLA"][1][0], T2["Fixed Quantization"][1][0], 32),
        "Pick-and-Place PaceVLA vs base (n=32)": cmp(T2["PaceVLA"][0][0], T2["base"][0][0], 32),
        "Conveyor PaceVLA vs base (n=50 packages)": cmp(T4["PaceVLA"][0], T4["base"][0], 50),
        "Table5 w/o decoders vs full (n=1200, 근사)": cmp(66.9, 63.3, 1200),
    }

    # C8 best 칸 세기 (가속 방법 4개 중)
    c8 = {"TPR_best": 0, "ES_best": 0, "SR_best": 0, "cells": 0, "TPR_lost_to": []}
    for b in BASES:
        for j, col in enumerate(COLS):
            c8["cells"] += 1
            accel = {m: T1[b][m][j] for m in METHODS[1:]}
            for key, idx in (("SR_best", 0), ("TPR_best", 1), ("ES_best", 2)):
                best = max(v[idx] for v in accel.values())
                if accel["PaceVLA"][idx] >= best:
                    c8[key] += 1
            tb = max(accel.items(), key=lambda kv: kv[1][1])
            if tb[0] != "PaceVLA":
                c8["TPR_lost_to"].append(f"{b}/{col}: {tb[0]} {tb[1][1]} > {accel['PaceVLA'][1]}")
    out["C8_best_counts_table1"] = c8

    # C9 양자화 비율 추정 (q = 2, 시간이 명령 수에 비례한다고 가정한 상한)
    c9 = {}
    for i, t in enumerate(REAL_TASKS):
        r_p = T2["base"][i][1] / T2["PaceVLA"][i][1]
        r_f = T2["base"][i][1] / T2["Fixed Quantization"][i][1]
        c9[t] = {"PaceVLA_speedup": round(r_p, 3), "FixedQ_speedup": round(r_f, 3),
                 "PaceVLA_over_FixedQ_time": round(T2["PaceVLA"][i][1] / T2["Fixed Quantization"][i][1], 3),
                 "implied_quantized_frac_if_ideal": round(2 * (1 - 1 / r_p), 2)}
    out["C9_quantized_fraction"] = c9

    # C10
    out["C10_misc"] = {"latency_ratio": round(42.15 / 40.07, 3), "latency_ms_added": round(42.15 - 40.07, 2),
                       "LIBERO_q": [round(16 / 9, 3), round(8 / 5, 3)], "RoboCasa_q": [2.0, 2.0],
                       "airfryer_time_saved_s": round(24.6 - 14.6, 1)}

    path = ROOT / "analysis" / "table_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in out.items():
        print(k, json.dumps(v, ensure_ascii=False)[:400])
    print("wrote", path)


if __name__ == "__main__":
    main()
