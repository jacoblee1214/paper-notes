"""DeCAL (arXiv:2609.09119 v1) 논문 표 수치를 다시 계산하고 통계 검정을 한다.

입력: 논문의 표 1, 표 2, 표 3, 표 5, 표 6과 그림 1, 그림 9의 값을 아래에 그대로 옮겨 적었다(출처는 주석).
      외부 데이터는 쓰지 않는다. 그림 1과 그림 9의 값은 막대 위 숫자를 눈으로 읽어 옮겼다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 표 1의 분모는 과제·방법당 20회다(§5.1 "evaluated with 20 trials by default", 표 1 캡션).
  OOD(그림 9, 표 6)의 시도 수는 쓰여 있지 않다. 값이 모두 5%의 배수라서 20회로 가정한다.
- 표 5의 단계별 값은 "그 단계까지 성공한 시도의 비율"(누적)로 읽는다. 마지막 단계 값이 표 1의 SR과 같기 때문이다.
- PSR은 표 5의 단계별 값의 평균과 같은지 확인한다(논문은 "average completion ratio across all task stages"라고만 쓴다).
- 검정 라벨은 V1, V2, ... 로 붙인다(03 문서의 검증 항목 번호).
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}

METHODS = ["GR00T N1.6", "InternVLA-A1", "ViTacFormer", "DECO", "InternVLA-A1t", "DeCAL"]
TASKS = ["Wipe Vase", "Erase Whiteboard", "Assemble Parts", "Twist Cap", "Pipetting", "Screw Light Bulb"]
NSTAGE = [4, 3, 3, 3, 4, 4]  # 표 4
N = 20

# 표 1 (SR %, PSR %) 과제 순서 = TASKS
T1 = {
    "GR00T N1.6":    [(30.0, 78.8), (50.0, 55.0), (30.0, 60.0), (10.0, 46.7), (60.0, 85.0), (10.0, 45.0)],
    "InternVLA-A1":  [(65.0, 85.0), (50.0, 58.3), (15.0, 43.3), (5.0, 36.7), (35.0, 78.8), (30.0, 53.8)],
    "ViTacFormer":   [(80.0, 88.8), (45.0, 50.0), (15.0, 71.7), (65.0, 81.7), (55.0, 86.3), (25.0, 46.3)],
    "DECO":          [(90.0, 95.0), (60.0, 70.0), (35.0, 61.7), (70.0, 83.3), (45.0, 71.3), (35.0, 41.3)],
    "InternVLA-A1t": [(75.0, 87.5), (45.0, 51.7), (45.0, 65.0), (25.0, 55.0), (20.0, 58.8), (25.0, 31.3)],
    "DeCAL":         [(100.0, 100.0), (80.0, 86.7), (65.0, 85.0), (80.0, 93.3), (60.0, 86.3), (40.0, 48.8)],
}

# 표 5 (ID 단계별 성공률). 과제별 단계 수대로 이어 붙인 21개 값 + 논문 Avg
T5_FLAT = {
    "GR00T N1.6":    ([1.00, .95, .90, .30, .60, .55, .50, .80, .70, .30, .85, .45, .10, 1.00, 1.00, .80, .60, .90, .60, .20, .10], 0.69),
    "InternVLA-A1":  ([1.00, 1.00, .75, .65, .65, .60, .50, .65, .50, .15, .95, .10, .05, 1.00, 1.00, .80, .35, .75, .70, .40, .30], 0.68),
    "ViTacFormer":   ([1.00, .95, .80, .80, .55, .50, .45, 1.00, 1.00, .15, 1.00, .80, .65, 1.00, .95, .95, .55, .70, .65, .25, .25], 0.79),
    "DECO":          ([1.00, 1.00, .90, .90, .75, .75, .60, .80, .70, .35, 1.00, .80, .70, 1.00, .75, .65, .45, .45, .45, .40, .35], 0.78),
    "InternVLA-A1t": ([.95, .95, .85, .75, .60, .50, .45, .80, .70, .45, .90, .50, .25, .95, .90, .30, .20, .35, .35, .30, .25], 0.64),
    "DeCAL":         ([1.00, 1.00, 1.00, 1.00, .95, .85, .80, 1.00, .90, .65, 1.00, 1.00, .80, 1.00, .95, .90, .60, .65, .50, .40, .40], 0.91),
}

# 표 6 (Twist Cap OOD 단계별, S1 S2 S3 x 4 시나리오) 과 Avg
T6_SCEN = ["Unseen Background", "Cluttered", "Unseen Lighting", "Unseen Object"]
T6 = {
    "ViTacFormer": ([.80, .60, .25, 1.00, 1.00, .40, .90, .85, .25, .90, .90, .45], 0.69),
    "DECO":        ([.90, .75, .50, 1.00, .85, .65, .95, .60, .55, .70, .60, .35], 0.70),
    "DeCAL":       ([.75, .75, .60, 1.00, .80, .70, .95, .90, .65, .95, .90, .75], 0.81),
}
# 그림 9 (OOD 성공률 %) 시나리오 순서 = T6_SCEN
F9 = {"ViTacFormer": [25, 40, 25, 45], "DECO": [50, 65, 55, 35], "DeCAL": [60, 70, 70, 75]}
# 그림 1 막대 위 숫자
F1 = {"SR": {"GR00T N1.6": 32, "DECO": 56, "DeCAL": 71},
      "PSR": {"GR00T N1.6": 69, "DECO": 78, "DeCAL": 91},
      "OOD": {"ViTacFormer": 34, "DECO": 51, "DeCAL": 69}}

# 표 2 (절제) : (Factorized FM, Tac Gating, Vis Gen, Tac Gen), Assemble %, Twist %
T2 = [
    ((0, 1, 1, 1), 25.0, 35.0),
    ((1, 1, 0, 0), 20.0, 30.0),
    ((1, 1, 0, 1), 35.0, 45.0),
    ((1, 1, 1, 0), 55.0, 60.0),
    ((1, 0, 1, 1), 50.0, 70.0),
    ((1, 1, 1, 1), 65.0, 80.0),
]
# 표 3 (시각 생성)
T3 = {"InternVLA-A1": {"Assemble Parts": (0.913, 0.243), "Twist Cap": (0.908, 0.257)},
      "DeCAL": {"Assemble Parts": (0.946, 0.222), "Twist Cap": (0.927, 0.245)}}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe_diff(k1, n1, k2, n2, z=1.959964):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="two-sided")[1])


def cmh(tables):
    """Mantel-Haenszel 공통 승산비와 카이제곱(연속성 보정 없음). tables: [(a,b,c,d)] a,b=성공,실패(방법1) c,d=(방법2)."""
    num = den = 0.0
    sa = se = sv = 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        num += a * d / n
        den += b * c / n
        r1, r2, c1, c2 = a + b, c + d, a + c, b + d
        sa += a
        se += r1 * c1 / n
        sv += r1 * r2 * c1 * c2 / (n * n * (n - 1))
    chi2 = (sa - se) ** 2 / sv
    return num / den, chi2, float(stats.chi2.sf(chi2, 1))


def k_of(pct, n=N):
    return round(pct / 100 * n)


def main():
    # V1 정수 가능성: 표 1 SR이 20회에서 가능한 값인가
    bad = [(m, TASKS[i], sr) for m, rows in T1.items() for i, (sr, _) in enumerate(rows) if abs(sr / 5 - round(sr / 5)) > 1e-9]
    RES["V1_SR_integer_feasible"] = {"violations": bad, "ok": not bad}

    # 표 5를 과제별로 나눈다
    T5 = {}
    for m, (flat, avg) in T5_FLAT.items():
        out, i = [], 0
        for ns in NSTAGE:
            out.append(flat[i:i + ns])
            i += ns
        assert i == 21
        T5[m] = out

    # V2 표 1 PSR = 표 5 단계 평균 인가 / V3 SR = 마지막 단계 인가 / V4 단계 값이 단조 감소인가 / V5 PSR 정수 가능성
    v2, v3, v4, v5 = [], [], [], []
    for m in METHODS:
        for ti, t in enumerate(TASKS):
            st = T5[m][ti]
            sr, psr = T1[m][ti]
            mean_pct = 100 * sum(st) / len(st)
            v2.append({"method": m, "task": t, "table1_psr": psr, "stage_mean": round(mean_pct, 2), "abs_diff": round(abs(mean_pct - psr), 2)})
            v3.append({"method": m, "task": t, "table1_sr": sr, "last_stage": 100 * st[-1], "match": abs(100 * st[-1] - sr) < 1e-9})
            v4.append({"method": m, "task": t, "monotone": all(st[j] >= st[j + 1] - 1e-12 for j in range(len(st) - 1))})
            # PSR*nstage 는 5의 배수여야 한다(단계값이 5%의 배수)
            x = psr * len(st) / 5
            v5.append({"method": m, "task": t, "psr_x_nstage_over5": round(x, 2), "feasible_after_rounding": abs(x - round(x)) < 0.05 * len(st)})
    RES["V2_psr_vs_stage_mean"] = {"max_abs_diff": max(r["abs_diff"] for r in v2), "rows": v2}
    RES["V3_sr_vs_last_stage"] = {"all_match": all(r["match"] for r in v3), "rows": [r for r in v3 if not r["match"]]}
    RES["V4_stage_monotone"] = {"all_monotone": all(r["monotone"] for r in v4), "rows": [r for r in v4 if not r["monotone"]]}
    RES["V5_psr_feasible"] = {"all_feasible": all(r["feasible_after_rounding"] for r in v5), "rows": [r for r in v5 if not r["feasible_after_rounding"]]}
    RES["V5b_psr_ge_sr"] = {"all": all(T1[m][i][1] >= T1[m][i][0] for m in METHODS for i in range(6))}

    # V6 평균 (표 1) 과 초록·그림 1·표 5 Avg
    avg = {}
    for m in METHODS:
        sr_vals = [T1[m][i][0] for i in range(6)]
        psr_rounded = [T1[m][i][1] for i in range(6)]
        psr_exact = [100 * sum(T5[m][i]) / NSTAGE[i] for i in range(6)]
        flat, paper_avg = T5_FLAT[m]
        avg[m] = {
            "sr_mean": round(float(np.mean(sr_vals)), 2),
            "psr_mean_from_table1_rounded": round(float(np.mean(psr_rounded)), 2),
            "psr_mean_from_stages_exact": round(float(np.mean(psr_exact)), 2),
            "table5_avg_reported": paper_avg,
            "table5_avg_all21_mean": round(float(np.mean(flat)), 3),
            "table5_avg_task_mean_of_stage_means": round(float(np.mean([np.mean(x) for x in T5[m]])), 3),
            "table5_avg_ratio_reported_over_recomputed": round(paper_avg / float(np.mean(flat)), 3),
        }
    RES["V6_averages"] = avg
    best_base_sr = max((m for m in METHODS if m != "DeCAL"), key=lambda m: avg[m]["sr_mean"])
    RES["V7_claim_gt_15pp"] = {
        "best_baseline": best_base_sr,
        "best_baseline_sr": avg[best_base_sr]["sr_mean"],
        "decal_sr": avg["DeCAL"]["sr_mean"],
        "gap_pp": round(avg["DeCAL"]["sr_mean"] - avg[best_base_sr]["sr_mean"], 3),
        "claim": "outperforming the strongest baseline by more than 15% average success rate (§1)",
    }
    RES["V7b_psr_83p4"] = {"exact_mean": avg["DeCAL"]["psr_mean_from_stages_exact"], "from_rounded_table1": avg["DeCAL"]["psr_mean_from_table1_rounded"], "reported": 83.4}

    # V8 그림 1: SR 32/56/71, PSR 69/78/91, OOD 34/51/69 가 어느 표의 값인가
    fig1 = {}
    for m, v in F1["SR"].items():
        fig1.setdefault("SR", {})[m] = {"fig1": v, "table1_mean": avg[m]["sr_mean"]}
    for m, v in F1["PSR"].items():
        fig1.setdefault("PSR", {})[m] = {"fig1": v, "table1_psr_mean": avg[m]["psr_mean_from_table1_rounded"], "table5_avg_x100": round(100 * T5_FLAT[m][1])}
    t6_last = {m: [100 * v[0][3 * s + 2] for s in range(4)] for m, v in T6.items()}
    for m, v in F1["OOD"].items():
        fig1.setdefault("OOD", {})[m] = {"fig1": v, "fig9_mean": round(float(np.mean(F9[m])), 2), "table6_last_stage_mean": round(float(np.mean(t6_last[m])), 2)}
    RES["V8_fig1_sources"] = fig1

    # V9 표 6 Avg 재계산, 표 6 마지막 단계 vs 그림 9
    v9 = {}
    for m, (flat, a) in T6.items():
        v9[m] = {"avg_reported": a, "avg_all12_mean": round(float(np.mean(flat)), 3), "last_stage_by_scenario": t6_last[m], "fig9": F9[m],
                 "mismatch_scenarios": [T6_SCEN[i] for i in range(4) if abs(t6_last[m][i] - F9[m][i]) > 1e-9]}
    RES["V9_table6_vs_fig9"] = v9
    # 표 6 단계 단조성
    mono6 = {m: all(flat[3 * s + j] >= flat[3 * s + j + 1] - 1e-12 for s in range(4) for j in range(2)) for m, (flat, _) in T6.items()}
    RES["V9b_table6_monotone"] = mono6
    # 표 1 ID Twist Cap 과 표 6 OOD 의 하락폭
    twist_id = {m: T1[m][3][0] for m in ["ViTacFormer", "DECO", "DeCAL"]}
    RES["V9c_twist_id_vs_ood_drop_pp"] = {m: {"id_sr": twist_id[m], "ood_fig9": F9[m], "drop_pp": [twist_id[m] - x for x in F9[m]]} for m in F9}

    # V10 과제별 DeCAL 대 최고 기준선, 대 DECO: Fisher, Wilson
    per_task = []
    for ti, t in enumerate(TASKS):
        ksr = {m: k_of(T1[m][ti][0]) for m in METHODS}
        base = max((m for m in METHODS if m != "DeCAL"), key=lambda m: ksr[m])
        d, lo, hi = newcombe_diff(ksr["DeCAL"], N, ksr[base], N)
        d2, lo2, hi2 = newcombe_diff(ksr["DeCAL"], N, ksr["DECO"], N)
        per_task.append({
            "task": t, "decal_k": ksr["DeCAL"], "decal_wilson95": [round(100 * x, 1) for x in wilson(ksr["DeCAL"], N)],
            "best_baseline": base, "best_k": ksr[base],
            "diff_pp": round(100 * d, 1), "newcombe95_pp": [round(100 * lo, 1), round(100 * hi, 1)], "fisher_p": round(fisher(ksr["DeCAL"], N, ksr[base], N), 4),
            "vs_DECO_diff_pp": round(100 * d2, 1), "vs_DECO_fisher_p": round(fisher(ksr["DeCAL"], N, ksr["DECO"], N), 4),
            "decal_strictly_best": all(ksr["DeCAL"] > ksr[m] for m in METHODS if m != "DeCAL"),
            "n_baselines_tied_or_above": sum(1 for m in METHODS if m != "DeCAL" and ksr[m] >= ksr["DeCAL"]),
        })
    RES["V10_per_task_sr"] = per_task
    RES["V10b_strictly_best_tasks_sr"] = sum(r["decal_strictly_best"] for r in per_task)
    # PSR 과제별 최고
    psr_best = []
    for ti, t in enumerate(TASKS):
        vals = {m: T1[m][ti][1] for m in METHODS}
        top = max(vals.values())
        psr_best.append({"task": t, "decal": vals["DeCAL"], "best_other": max(v for m, v in vals.items() if m != "DeCAL"), "decal_best_or_tied": vals["DeCAL"] >= top,
                         "decal_strictly_best": vals["DeCAL"] > max(v for m, v in vals.items() if m != "DeCAL")})
    RES["V10c_psr_best"] = {"rows": psr_best, "strict_count": sum(r["decal_strictly_best"] for r in psr_best), "best_or_tied_count": sum(r["decal_best_or_tied"] for r in psr_best)}

    # V11 합산 (6과제 x 20 = 120회)
    pooled = {}
    for m in METHODS:
        k = sum(k_of(T1[m][i][0]) for i in range(6))
        lo, hi = wilson(k, 6 * N)
        pooled[m] = {"k": k, "n": 6 * N, "rate_pct": round(100 * k / (6 * N), 1), "wilson95_pct": [round(100 * lo, 1), round(100 * hi, 1)]}
    RES["V11_pooled_sr"] = pooled
    comps = []
    for b in ["GR00T N1.6", "InternVLA-A1", "ViTacFormer", "DECO", "InternVLA-A1t"]:
        kd = pooled["DeCAL"]["k"]
        kb = pooled[b]["k"]
        d, lo, hi = newcombe_diff(kd, 120, kb, 120)
        tabs = [(k_of(T1["DeCAL"][i][0]), N - k_of(T1["DeCAL"][i][0]), k_of(T1[b][i][0]), N - k_of(T1[b][i][0])) for i in range(6)]
        or_mh, chi2, p_mh = cmh(tabs)
        comps.append({"baseline": b, "diff_pp": round(100 * d, 1), "newcombe95_pp": [round(100 * lo, 1), round(100 * hi, 1)],
                      "fisher_pooled_p": round(fisher(kd, 120, kb, 120), 5), "mh_or": round(or_mh, 2), "mh_chi2": round(chi2, 2), "mh_p": round(p_mh, 5)})
    RES["V11b_pooled_vs_baselines"] = comps
    # 과제를 6개 쌍으로 본 부호 검정 (DeCAL SR >= DECO SR 인 과제 수)
    wins = sum(1 for i in range(6) if T1["DeCAL"][i][0] > T1["DECO"][i][0])
    RES["V11c_sign_vs_DECO"] = {"decal_wins_of_6": wins, "binom_two_sided_p": float(stats.binomtest(wins, 6, 0.5).pvalue)}

    # V12 절제 (표 2) : 각 행 대 전체 모델, Fisher
    full = T2[-1]
    ab = []
    for combo, a, t in T2[:-1]:
        row = {"combo(FM,Gate,VisGen,TacGen)": combo}
        for name, v, fv in (("assemble", a, full[1]), ("twist", t, full[2])):
            k, kf = k_of(v), k_of(fv)
            row[name] = {"k": k, "full_k": kf, "diff_pp": round(v - fv, 1), "fisher_p": round(fisher(kf, N, k, N), 4)}
        # 두 과제 합산
        kk = k_of(a) + k_of(t)
        kf = k_of(full[1]) + k_of(full[2])
        row["pooled_two_tasks"] = {"k": kk, "full_k": kf, "fisher_p": round(fisher(kf, 40, kk, 40), 4)}
        ab.append(row)
    RES["V12_ablation_vs_full"] = ab
    # 가산성: 시각 생성만 뺀 것, 촉각 생성만 뺀 것, 둘 다 뺀 것
    f_a, f_t = full[1], full[2]
    rows = {tuple(c): (a, t) for c, a, t in T2}
    no_vis, no_tac, no_both = rows[(1, 1, 0, 1)], rows[(1, 1, 1, 0)], rows[(1, 1, 0, 0)]
    RES["V12b_additivity"] = {
        "assemble": {"drop_vis": f_a - no_vis[0], "drop_tac": f_a - no_tac[0], "drop_both": f_a - no_both[0], "sum_of_single_drops": (f_a - no_vis[0]) + (f_a - no_tac[0])},
        "twist": {"drop_vis": f_t - no_vis[1], "drop_tac": f_t - no_tac[1], "drop_both": f_t - no_both[1], "sum_of_single_drops": (f_t - no_vis[1]) + (f_t - no_tac[1])},
    }
    # 절제 완전 모델 = 표 1 과 일치
    RES["V12c_full_equals_table1"] = {"assemble": full[1] == T1["DeCAL"][2][0], "twist": full[2] == T1["DeCAL"][3][0]}
    # 절제 중 DeCAL 구성보다 나은 기준선과 비교: 두 과제 모두 w/o Tac Gate(50/70) 이 DECO(35/70)와 비교
    RES["V12d_ablation_vs_best_baseline"] = {
        "Assemble: full 65 vs InternVLA-A1t 45": round(fisher(13, 20, 9, 20), 4),
        "Assemble: w/o FM (25) vs InternVLA-A1t (45)": round(fisher(5, 20, 9, 20), 4),
        "Twist: full 80 vs DECO 70": round(fisher(16, 20, 14, 20), 4),
        "Twist: w/o Tac Gate 70 vs DECO 70": round(fisher(14, 20, 14, 20), 4),
    }

    # V13 표 3
    t3 = {}
    for task in ("Assemble Parts", "Twist Cap"):
        a, d = T3["InternVLA-A1"][task], T3["DeCAL"][task]
        t3[task] = {"cos_gain": round(d[0] - a[0], 3), "lpips_gain": round(a[1] - d[1], 3), "cos_rel_gain_pct": round(100 * (d[0] - a[0]) / a[0], 2), "lpips_rel_gain_pct": round(100 * (a[1] - d[1]) / a[1], 2)}
    RES["V13_table3_gain"] = t3

    # V14 학습·평가 예산 (부록 B, F 와 데이터셋 README)
    frames_per_traj = 1989891 / 1279
    RES["V14_budget"] = {
        "dataset_trajectories": 1279, "dataset_frames": 1989891, "dataset_tasks": 13, "mean_frames_per_traj": round(frames_per_traj, 1), "mean_seconds_per_traj_30hz": round(frames_per_traj / 30, 1),
        "paper_demos": 6 * 100, "paper_frames_estimate_if_mean": round(600 * frames_per_traj),
        "decal_samples_seen": 100000 * 4 * 8, "decal_epochs_est": round(100000 * 4 * 8 / (600 * frames_per_traj), 1),
        "groot_samples_seen": 50000 * 32, "groot_epochs_est": round(50000 * 32 / (600 * frames_per_traj), 1),
        "internvla_a1_lr_at_100k_if_linear_to_200k": 5e-5 + (5e-6 - 5e-5) * (100000 - 2000) / (200000 - 2000),
        "chunk_seconds_at_30hz": 50 / 30, "latency_s": 0.27,
        "note": "시나리오 값은 추정이다. 과제당 trajectory 평균은 6과제 600개를 13과제 평균 길이로 환산한 것.",
    }

    # V15 선택 효과: 6개 과제 중 DeCAL이 DECO보다 유의하게(p<0.05) 높은 과제 수
    sig = sum(1 for r in per_task if r["vs_DECO_fisher_p"] < 0.05)
    RES["V15_sig_vs_DECO_tasks"] = {"count_p_lt_0.05": sig, "p_values": {r["task"]: r["vs_DECO_fisher_p"] for r in per_task}}
    # 기준선 5개 x 과제 6개 = 30 비교 중 p<0.05 개수 (Bonferroni 0.05/30)
    allp = []
    for ti, t in enumerate(TASKS):
        for b in METHODS[:-1]:
            p = fisher(k_of(T1["DeCAL"][ti][0]), N, k_of(T1[b][ti][0]), N)
            allp.append({"task": t, "baseline": b, "p": round(p, 4)})
    RES["V15b_all30"] = {"n_p_lt_0.05": sum(r["p"] < 0.05 for r in allp), "n_p_lt_bonferroni_0.00167": sum(r["p"] < 0.05 / 30 for r in allp), "rows": allp}

    # V16 OOD (그림 9) 20회 가정 Fisher
    ood = []
    for si, s in enumerate(T6_SCEN):
        kd = k_of(F9["DeCAL"][si])
        for b in ("ViTacFormer", "DECO"):
            kb = k_of(F9[b][si])
            ood.append({"scenario": s, "baseline": b, "decal": F9["DeCAL"][si], "base": F9[b][si], "fisher_p": round(fisher(kd, N, kb, N), 4)})
    RES["V16_ood_fisher"] = ood
    kd = sum(k_of(x) for x in F9["DeCAL"])
    RES["V16b_ood_pooled"] = {b: {"decal_k": kd, "base_k": sum(k_of(x) for x in F9[b]), "n": 80, "fisher_p": round(fisher(kd, 80, sum(k_of(x) for x in F9[b]), 80), 5)} for b in ("ViTacFormer", "DECO")}
    # 같은 과제(Twist Cap) ID 80 대 OOD 평균: 하락폭
    RES["V16c_decal_id_vs_ood"] = {"id": 80, "ood_mean_fig9": float(np.mean(F9["DeCAL"])), "ood_mean_table6": float(np.mean(t6_last["DeCAL"]))}

    OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")

    # 요약 출력
    print("V1 SR 정수 가능:", RES["V1_SR_integer_feasible"]["ok"])
    print("V2 PSR vs 단계 평균 최대 차이(%p):", RES["V2_psr_vs_stage_mean"]["max_abs_diff"])
    print("V3 SR=마지막 단계:", RES["V3_sr_vs_last_stage"]["all_match"], "| V4 단조:", RES["V4_stage_monotone"]["all_monotone"], "| V5 PSR 가능:", RES["V5_psr_feasible"]["all_feasible"])
    for m in METHODS:
        print(" ", m, avg[m])
    print("V7", RES["V7_claim_gt_15pp"])
    print("V7b", RES["V7b_psr_83p4"])
    print("V8", json.dumps(fig1, ensure_ascii=False))
    print("V9", json.dumps(v9, ensure_ascii=False))
    print("V10 과제별:")
    for r in per_task:
        print(" ", r)
    print("V10b 엄밀히 최고인 과제 수(SR):", RES["V10b_strictly_best_tasks_sr"], "| PSR:", RES["V10c_psr_best"]["strict_count"], "(동률 포함", RES["V10c_psr_best"]["best_or_tied_count"], ")")
    print("V11", json.dumps(pooled, ensure_ascii=False))
    for c in comps:
        print(" ", c)
    print("V11c", RES["V11c_sign_vs_DECO"])
    for r in ab:
        print("V12", r)
    print("V12b", RES["V12b_additivity"])
    print("V12d", RES["V12d_ablation_vs_best_baseline"])
    print("V13", t3)
    print("V14", RES["V14_budget"])
    print("V15", RES["V15_sig_vs_DECO_tasks"], RES["V15b_all30"]["n_p_lt_0.05"], RES["V15b_all30"]["n_p_lt_bonferroni_0.00167"])
    for r in ood:
        print("V16", r)
    print("V16b", RES["V16b_ood_pooled"])


if __name__ == "__main__":
    main()
