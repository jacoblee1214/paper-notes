"""NestDex 표·공개 데이터 재계산 (CPU). 실행: python3 analysis/table_check.py

- C1: Table II 수집 성공률(20회) Fisher 정확검정 + Holm
- C2: Table II "성공 1건당 시간"을 시행 1회당 시간으로 분해 (시간 = 시행당 시간 / 성공률)
- C3: Table III 외부 정책 성공률(20회) — H-VAE 효과, 수집 방식 효과, 과제 합산(Mantel–Haenszel)
- C4: Q4 실행 조건 3개(10회씩) Fisher + Holm, 본문 p = 0.0198 대조
- C5: 공개 데이터 data/inner-policy-effort.json(프로젝트 사이트) 재분석 — holdMedian 재계산, 정확 Mann–Whitney p 대조
- C6: 학습 규모 산술 — 내부 정책 epoch 수, 외부 정책 데이터 크기(제어 주기 가정)
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
N = 20  # Q1·Q2 시행 수 (Sec. V-B Shared protocol)

TASKS = ["Tongs Transfer", "Bottle Disposal", "Dual-Object Transfer", "Ingredient and Pot Transfer",
         "Toast Preparation", "Binder Filing"]
# Table II: (시간 s, 성공률 %); 시간이 N/A면 None
T2 = {
    "Copilot": [(44.33, 100), (41.37, 100), (36.19, 100), (43.26, 100), (327.46, 100), (221.80, 100)],
    "AnyTeleop": [(None, 0), (88.88, 50), (121.63, 30), (55.29, 75), (None, 0), (None, 0)],
}
# Table III: 단일 팔 4과제, 20 rollout, %
T3_TASKS = TASKS[:4]
T3 = {
    "Copilot, no H-VAE": [65, 60, 80, 85],
    "Copilot, H-VAE": [100, 75, 90, 100],
    "AnyTeleop, no H-VAE": [None, 40, 20, 75],
}
# Q4 (Sec. V-C): 10회씩
Q4 = {"Fixed-command replay": 3, "Closed-loop, no ensemble": 7, "Closed-loop, temporal ensemble": 9}
Q4_N = 10


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return adj


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 3), round(c + h, 3)]


def mh_test(tables):
    """Cochran–Mantel–Haenszel (연속성 보정 없음). tables: [(k1, n1, k2, n2)]"""
    num, var, or_num, or_den = 0.0, 0.0, 0.0, 0.0
    for k1, n1, k2, n2 in tables:
        a, b, c, d = k1, n1 - k1, k2, n2 - k2
        n = a + b + c + d
        m1, m0, r1 = a + c, b + d, a + b
        e = r1 * m1 / n
        v = r1 * (n - r1) * m1 * m0 / (n * n * (n - 1)) if n > 1 else 0.0
        num += a - e
        var += v
        or_num += a * d / n
        or_den += b * c / n
    chi2 = num * num / var if var > 0 else float("nan")
    return {"chi2": round(chi2, 3), "p": float(f"{stats.chi2.sf(chi2, 1):.3g}"),
            "MH_odds_ratio": (round(or_num / or_den, 2) if or_den > 0 else "inf")}


def exact_mwu_p(x, y):
    """두 표본 정확 Mann–Whitney (양측, 동점 없음 가정). 모든 조합 열거."""
    x, y = list(x), list(y)
    pooled = x + y
    nx = len(x)
    ranks = stats.rankdata(pooled)
    obs = sum(ranks[:nx])
    mean = nx * (len(pooled) + 1) / 2
    dev_obs = abs(obs - mean)
    cnt = tot = 0
    for idx in itertools.combinations(range(len(pooled)), nx):
        s = sum(ranks[i] for i in idx)
        tot += 1
        if abs(s - mean) >= dev_obs - 1e-9:
            cnt += 1
    return cnt / tot


def main() -> None:
    out: dict = {}

    # C1 Table II 성공률
    c1 = {}
    ps = []
    for i, t in enumerate(TASKS):
        kc = round(T2["Copilot"][i][1] / 100 * N)
        ka = round(T2["AnyTeleop"][i][1] / 100 * N)
        p = fisher(kc, N, ka, N)
        ps.append(p)
        c1[t] = {"copilot": f"{kc}/{N}", "anyteleop": f"{ka}/{N}", "anyteleop_wilson95": wilson(ka, N), "fisher_p": float(f"{p:.3g}")}
    for t, a in zip(TASKS, holm(ps)):
        c1[t]["holm_p"] = float(f"{a:.3g}")
    out["C1_table2_success"] = c1

    # C2 시간 분해: 성공 1건당 시간 × 성공률 = 시행 1회당 평균 시간
    c2 = {}
    for i, t in enumerate(TASKS):
        tc, sc = T2["Copilot"][i]
        ta, sa = T2["AnyTeleop"][i]
        row = {"copilot_per_attempt_s_upper": round(tc * sc / 100, 2)}
        if ta is not None:
            row["anyteleop_per_attempt_s"] = round(ta * sa / 100, 2)
            row["ratio_anyteleop_over_copilot_per_attempt"] = round(ta * sa / 100 / (tc * sc / 100), 3)
            row["ratio_time_per_success"] = round(ta / tc, 3)
        c2[t] = row
    out["C2_time_decomposition"] = {
        "rows": c2,
        "note": "Copilot 시간에는 내부 정책 시연 10개 수집의 분할 상각분이 포함되어 있어, 텔레옵 시행 자체는 이 값보다 짧다(상한). "
                "실패 시행과 성공 시행의 길이가 같다고 가정할 때 AnyTeleop 값은 시행당 평균 길이다.",
    }

    # C3 Table III
    c3: dict = {"hvae_effect": {}, "source_effect_no_hvae": {}}
    hv_tables, src_tables, ps_h = [], [], []
    for i, t in enumerate(T3_TASKS):
        a = round(T3["Copilot, H-VAE"][i] / 100 * N)
        b = round(T3["Copilot, no H-VAE"][i] / 100 * N)
        p = fisher(a, N, b, N)
        ps_h.append(p)
        hv_tables.append((a, N, b, N))
        c3["hvae_effect"][t] = {"with": f"{a}/{N}", "without": f"{b}/{N}", "fisher_p": float(f"{p:.3g}")}
        if T3["AnyTeleop, no H-VAE"][i] is not None:
            c_ = round(T3["AnyTeleop, no H-VAE"][i] / 100 * N)
            p2 = fisher(b, N, c_, N)
            src_tables.append((b, N, c_, N))
            c3["source_effect_no_hvae"][t] = {"copilot": f"{b}/{N}", "anyteleop": f"{c_}/{N}", "fisher_p": float(f"{p2:.3g}")}
    for t, a_ in zip(T3_TASKS, holm(ps_h)):
        c3["hvae_effect"][t]["holm_p"] = float(f"{a_:.3g}")
    c3["hvae_pooled_MH"] = mh_test(hv_tables)
    c3["hvae_pooled_counts"] = f"{sum(x[0] for x in hv_tables)}/80 vs {sum(x[2] for x in hv_tables)}/80"
    c3["source_pooled_MH_3tasks"] = mh_test(src_tables)
    c3["missing_cells"] = ["AnyTeleop + H-VAE 조건 없음", "양팔 2과제(Toast, Binder)의 외부 정책 결과 없음"]
    out["C3_table3_outer_policy"] = c3

    # C4 Q4
    names = list(Q4)
    pairs = list(itertools.combinations(names, 2))
    pq = [fisher(Q4[a], Q4_N, Q4[b], Q4_N) for a, b in pairs]
    out["C4_q4_execution"] = {
        f"{a} vs {b}": {"counts": f"{Q4[a]}/10 vs {Q4[b]}/10", "fisher_two_sided_p": float(f"{p:.4g}"), "holm_p": float(f"{h:.3g}")}
        for (a, b), p, h in zip(pairs, pq, holm(pq))
    }
    out["C4_q4_execution"]["paper_reported_fixed_vs_ensemble_p"] = 0.0198
    # 단측 Fisher도 함께 (본문이 어느 쪽을 썼는지 확인용)
    out["C4_q4_execution"]["fixed_vs_ensemble_one_sided_p"] = float(f"{stats.fisher_exact([[9, 1], [3, 7]], alternative='greater')[1]:.4g}")

    # C5 공개 데이터 재분석
    d = json.loads((ROOT / "paper" / "inner-policy-effort.json").read_text())
    t = np.array(d["time"])
    lo, hi = d["holdWindowSec"]
    win = (t >= lo - 1e-9) & (t <= hi + 1e-9)
    c5: dict = {"source": d["source"], "alignment": d["alignment"], "n_time": len(t), "dt_s": round(float(t[1] - t[0]), 3)}
    recomputed_ok, maxdiff = True, 0.0
    groups = {}
    for c in d["conditions"]:
        sv, fv = [], []
        for tr in c["trials"]:
            e = np.array([np.nan if v is None else v for v in tr["effort"]], dtype=float)
            hm = float(np.nanmedian(e[win]))
            maxdiff = max(maxdiff, abs(hm - tr["holdMedian"]))
            (sv if tr["outcome"] == "success" else fv).append(tr["holdMedian"])
        groups[c["id"]] = {"success": sv, "failure": fv, "label": c["label"]}
        cmp_ = c["comparison"]
        p_re = exact_mwu_p(sv, fv) if sv and fv else None
        c5[c["id"]] = {
            "label": c["label"], "success_n": len(sv), "failure_n": len(fv),
            "episodes": [tr["episode"] for tr in c["trials"]],
            "success_median": round(float(np.median(sv)), 4) if sv else None,
            "failure_median": round(float(np.median(fv)), 4) if fv else None,
            "mwu_p_recomputed": round(p_re, 4) if p_re is not None else None,
            "mwu_p_file": round(cmp_["exact_mann_whitney_two_sided_p"], 4),
            "min_attainable_p": round(2 / math.comb(len(sv) + len(fv), min(len(sv), len(fv))), 4) if sv and fv else None,
        }
    c5["holdMedian_recomputed_max_abs_diff"] = float(f"{maxdiff:.3g}")
    between = {}
    for s in d["successfulBetweenConditions"]:
        a, b = s["first_condition"], s["second_condition"]
        between[f"{a} vs {b}"] = {
            "recomputed_p": round(exact_mwu_p(groups[a]["success"], groups[b]["success"]), 4),
            "file_p": round(s["exact_mann_whitney_two_sided_p"], 4),
        }
    c5["successful_between_conditions"] = between
    # 모든 30회를 합쳐 hold 노력값으로 성공 판별: AUC
    allv = [(v, 1) for g in groups.values() for v in g["success"]] + [(v, 0) for g in groups.values() for v in g["failure"]]
    pos = [v for v, y in allv if y == 1]
    neg = [v for v, y in allv if y == 0]
    auc = sum((p_ > n_) + 0.5 * (p_ == n_) for p_ in pos for n_ in neg) / (len(pos) * len(neg))
    thr_best = max(sorted(set(v for v, _ in allv)), key=lambda th: sum((v >= th) == bool(y) for v, y in allv))
    acc = sum((v >= thr_best) == bool(y) for v, y in allv) / len(allv)
    c5["pooled_effort_predicts_success"] = {"n_success": len(pos), "n_failure": len(neg), "AUC": round(auc, 3),
                                             "best_threshold": round(thr_best, 3), "accuracy_at_threshold": round(acc, 3)}
    c5["condition_ids_note"] = "temporal ensemble 조건의 파일 id가 'online_rtc', 무앙상블이 'online_no_rtc' — 본문은 RTC(Real-Time Chunking)가 아니라 Eq. 2(ACT식 지수 가중 평균)라고 서술"
    out["C5_public_effort_data"] = c5

    # C6 학습 규모 산술
    h, H_in, steps_in, batch_in, n_traj = 30, 30, 20_000, 256, 10
    rows = {}
    for T in (200, 300, 500, 1000):  # 내부 기술 시연 길이(100 Hz 스텝) 가정
        windows = n_traj * max(0, T - h - H_in + 1)
        rows[f"T={T} ({T/100:.0f} s)"] = {"windows": windows, "epochs": round(steps_in * batch_in / windows, 0) if windows else None}
    steps_out, batch_out, H_out, n_demo = 50_000, 8, 100, 20
    out_rows = {}
    for task, (tsec, _) in zip(TASKS[:4], T2["Copilot"][:4]):
        for hz in (10, 30, 100):
            T = int(tsec * hz)  # 시연 길이 상한(성공 1건당 시간 기준)
            windows = n_demo * max(0, T - H_out)
            out_rows[f"{task} @ {hz} Hz"] = {"demo_steps_upper": T, "windows": windows,
                                            "epochs": round(steps_out * batch_out / windows, 2) if windows else None,
                                            "chunk_seconds": round(H_out / hz, 2)}
    out["C6_training_scale"] = {
        "inner_policy": {"history": h, "chunk": H_in, "steps": steps_in, "batch": batch_in, "trajectories": n_traj,
                         "samples_seen": steps_in * batch_in, "by_demo_length": rows},
        "outer_policy": {"chunk": H_out, "steps": steps_out, "batch": batch_out, "demos": n_demo,
                         "samples_seen": steps_out * batch_out, "by_task_and_rate": out_rows,
                         "note": "외부 정책 제어 주기는 논문에 없다(내부 정책만 100 Hz 명시). 100 Hz면 chunk 100 = 1 s"},
        "clutch_speed_bound": "Eq. 8: 실행 index가 제어 주기마다 최대 1칸 → 기술 하나를 끝까지 실행하는 데 최소 T_max/100 s (100 Hz). 클러치로는 시연 속도보다 빠르게 할 수 없다",
    }

    path = ROOT / "analysis" / "table_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False)[:6000])
    print("wrote", path)


if __name__ == "__main__":
    main()
