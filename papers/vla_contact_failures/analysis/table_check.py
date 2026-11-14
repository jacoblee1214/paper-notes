"""논문 표(arXiv v2, tables/*.tex)의 SR 값으로 성공 수를 복원하고 신뢰구간과 검정을 다시 계산한다.

실행: python3 analysis/table_check.py  → analysis/table_check_results.json (CPU, 수 초)
입력: 표 값은 논문 v2 TeX 소스에서 옮겼다(출처 주석 참조). 원문 파일을 읽지 않는다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

N = 40  # 논문 §5: 과제·조건당 40 rollouts
OUT = Path(__file__).resolve().parent / "table_check_results.json"
TASKS = ["plug", "usb", "button", "board", "key"]

# tab_results.tex: SR(%) 과 보고된 p (π0.5 대비, 단측 Fisher, 보정 없음)
MAIN = {
    "pi05":     [30.0, 37.5, 12.5, 100.0, 15.0],
    "pi05_LN":  [50.0, 47.5, 57.5, 87.5, 37.5],
    "FACT":     [57.5, 47.5, 75.0, 90.0, 60.0],
    "ForceVLA": [32.5, 37.5, 12.5, 77.5, 42.5],
    "TA-VLA":   [30.0, 25.0, 20.0, 97.5, 15.0],
}
MAIN_P = {
    "pi05_LN":  [0.055, 0.249, 0.001, 1.00, 0.020],
    "FACT":     [0.012, 0.249, 0.001, 1.00, 0.001],
    "ForceVLA": [0.500, 0.591, 0.631, 1.00, 0.006],
    "TA-VLA":   [0.596, 0.926, 0.273, 1.00, 0.622],
}  # "<.001"은 0.001로 적었다
MAIN_ALL_REPORTED = {"pi05": 39.0, "pi05_LN": 56.0, "FACT": 66.0, "ForceVLA": 40.5, "TA-VLA": 37.5}

# tab_baselines_pi05 (tab_ablations_combined.tex 좌측 표). 과제: plug, key, button
BASE_LN = {
    "TA-VLA": [30.0, 15.0, 20.0], "TA-VLA+LN": [47.5, 32.5, 42.5],
    "ForceVLA": [32.5, 42.5, 12.5], "ForceVLA+LN": [57.5, 30.0, 30.0],
}
# tab_ablations_combined.tex 가운데 표. 과제: plug, key, button
ABL = {"FACT": [57.5, 60.0, 75.0], "w/o grad th": [42.5, 27.5, 57.5],
       "w/o curr. read": [60.0, 35.0, 50.0], "w/o history": [30.0, 20.0, 12.5]}
# tab_ablations_combined.tex 오른쪽 표(π0). 과제: plug, key, button
PI0 = {"pi0": [32.5, 45.0, 47.5], "FACT_pi0": [70.0, 50.0, 60.0],
       "ForceVLA_pi0": [55.0, 42.5, 42.5], "TA-VLA_pi0": [30.0, 27.5, 27.5]}
# tab_force_noise.tex: (measured F/T SR, 보고된 p, noise SR). 과제: plug, key, button
NOISE = {
    "TA-VLA": ([30.0, 15.0, 20.0], [0.306, 0.919, 0.273], [22.5, 25.0, 12.5]),
    "ForceVLA": ([32.5, 42.5, 12.5], [0.824, 0.006, 0.826], [40.0, 15.0, 17.5]),
    "FACT": ([57.5, 60.0, 75.0], [0.090, 0.001, 0.001], [40.0, 5.0, 17.5]),
}


def counts(pcts):
    c = [p * N / 100 for p in pcts]
    assert all(abs(x - round(x)) < 1e-9 for x in c), pcts
    return [int(round(x)) for x in c]


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fisher_greater(k1, k0):
    """H1: 방법 1의 성공률이 더 높다 (단측 Fisher, 2x2 표 [[k1, N-k1], [k0, N-k0]])."""
    return stats.fisher_exact([[k1, N - k1], [k0, N - k0]], alternative="greater")[1]


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adj[idx] = running
    return adj.tolist()


def main():
    res = {"n_per_cell": N}

    # 1. 성공 수 복원과 평균 재계산
    main_k = {m: counts(v) for m, v in MAIN.items()}
    res["main_counts"] = main_k
    res["main_avg_recomputed"] = {m: round(float(np.mean(v)), 2) for m, v in MAIN.items()}
    res["main_avg_reported"] = MAIN_ALL_REPORTED
    res["main_avg_mismatch"] = {m: round(res["main_avg_recomputed"][m] - MAIN_ALL_REPORTED[m], 2)
                                for m in MAIN}

    # 2. 셀별 Wilson 95%와 보고 p 대 재계산 p (π0.5 대비 단측)
    rows, pvals, labels = [], [], []
    for m in ["pi05_LN", "FACT", "ForceVLA", "TA-VLA"]:
        for j, t in enumerate(TASKS):
            k1, k0 = main_k[m][j], main_k["pi05"][j]
            p_re = fisher_greater(k1, k0)
            rows.append({"method": m, "task": t, "k": k1, "k_base": k0,
                         "wilson95": [round(x * 100, 1) for x in wilson(k1, N)],
                         "p_reported": MAIN_P[m][j], "p_recomputed": round(p_re, 4)})
            pvals.append(p_re)
            labels.append((m, t))
    adj = holm(pvals)
    for r, a in zip(rows, adj):
        r["p_holm_20"] = round(a, 4)
    res["main_tests"] = rows
    res["main_pi05_wilson"] = {t: [round(x * 100, 1) for x in wilson(main_k["pi05"][j], N)]
                               for j, t in enumerate(TASKS)}

    # 3. 풀링 (200 rollouts, 과제 불균형을 무시한 단순 집계)
    pooled = {}
    for m in MAIN:
        k = sum(main_k[m])
        pooled[m] = {"k": k, "n": 5 * N, "rate": round(k / (5 * N) * 100, 1),
                     "wilson95": [round(x * 100, 1) for x in wilson(k, 5 * N)]}
    res["pooled_200"] = pooled

    # 4. 차이 신뢰구간 (Newcombe, 셀 단위): FACT - π0.5, FACT - ForceVLA 풀링 기준
    def newcombe_diff(k1, n1, k2, n2):
        l1, u1 = wilson(k1, n1)
        l2, u2 = wilson(k2, n2)
        p1, p2 = k1 / n1, k2 / n2
        d = p1 - p2
        lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
        hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
        return [round(d * 100, 1), round(lo * 100, 1), round(hi * 100, 1)]

    res["diff_FACT_minus_ForceVLA_pooled_pp"] = newcombe_diff(sum(main_k["FACT"]), 200, sum(main_k["ForceVLA"]), 200)
    res["diff_FACT_minus_pi05_pooled_pp"] = newcombe_diff(sum(main_k["FACT"]), 200, sum(main_k["pi05"]), 200)
    res["diff_FACT_minus_ForceVLA_per_task_pp"] = {
        t: newcombe_diff(main_k["FACT"][j], N, main_k["ForceVLA"][j], N) for j, t in enumerate(TASKS)}

    # 5. LN 대 Beta의 과제별 차이 (논문 본문 수치 확인)
    res["text_claims"] = {
        "LN_plug_gain_pp": round(MAIN["pi05_LN"][0] - MAIN["pi05"][0], 1),
        "LN_usb_gain_pp": round(MAIN["pi05_LN"][1] - MAIN["pi05"][1], 1),
        "LN_button_gain_pp": round(MAIN["pi05_LN"][2] - MAIN["pi05"][2], 1),
        "LN_key_gain_pp": round(MAIN["pi05_LN"][4] - MAIN["pi05"][4], 1),
        "FACT_over_LN_button_pp": round(MAIN["FACT"][2] - MAIN["pi05_LN"][2], 1),
        "FACT_over_LN_key_pp": round(MAIN["FACT"][4] - MAIN["pi05_LN"][4], 1),
        "FACT_minus_ForceVLA_avg_pp": round(MAIN_ALL_REPORTED["FACT"] - MAIN_ALL_REPORTED["ForceVLA"], 1),
        "FACT_minus_ForceVLA_per_task_pp": [round(a - b, 1) for a, b in zip(MAIN["FACT"], MAIN["ForceVLA"])],
        "FACT_minus_ForceVLA_min_task_pp": round(min(a - b for a, b in zip(MAIN["FACT"], MAIN["ForceVLA"])), 1),
    }

    # 6. π0.5 + LN 기준선 (tab_baselines_pi05): LN 추가 효과의 부호
    ln_rows = []
    for base, lnm, tk in [("TA-VLA", "TA-VLA+LN", ["plug", "key", "button"]),
                          ("ForceVLA", "ForceVLA+LN", ["plug", "key", "button"])]:
        kb, kl = counts(BASE_LN[base]), counts(BASE_LN[lnm])
        for j, t in enumerate(tk):
            ln_rows.append({"base": base, "task": t, "k_base": kb[j], "k_ln": kl[j],
                            "delta_pp": round(BASE_LN[lnm][j] - BASE_LN[base][j], 1),
                            "p_ln_greater": round(fisher_greater(kl[j], kb[j]), 4)})
    res["ln_on_baselines"] = ln_rows
    res["ln_on_baselines_note"] = ("ForceVLA의 key insertion은 42.5 -> 30.0 (-12.5 pp)으로 떨어진다. "
                                   "본문 '일관된 이득(consistent gains)'과 맞지 않는다.")

    # 7. 노이즈 대체 검정: 보고 p 대 재계산 (측정 F/T 대 노이즈, 같은 방법 내 단측)
    noise_rows = []
    for m, (meas, prep, noi) in NOISE.items():
        km, kn = counts(meas), counts(noi)
        for j, t in enumerate(["plug", "key", "button"]):
            noise_rows.append({"method": m, "task": t, "k_measured": km[j], "k_noise": kn[j],
                               "p_reported": prep[j],
                               "p_recomputed_one_sided": round(fisher_greater(km[j], kn[j]), 4)})
    res["noise_tests"] = noise_rows

    # 8. 구성요소 제거 (논문에 검정 없음): FACT 대비 단측 p, 과제별
    abl_rows = []
    for name in ["w/o grad th", "w/o curr. read", "w/o history"]:
        ka = counts(ABL[name])
        for j, t in enumerate(["plug", "key", "button"]):
            kf = counts(ABL["FACT"])[j]
            abl_rows.append({"ablation": name, "task": t, "k_FACT": kf, "k_ablation": ka[j],
                             "delta_pp": round(ABL[name][j] - ABL["FACT"][j], 1),
                             "p_FACT_greater": round(fisher_greater(kf, ka[j]), 4)})
    # Holm은 9개 비교 전체에서
    ap = holm([r["p_FACT_greater"] for r in abl_rows])
    for r, a in zip(abl_rows, ap):
        r["p_holm_9"] = round(a, 4)
    res["ablation_tests_not_in_paper"] = abl_rows

    # 9. π0 백본: FACT_pi0 대 π0 (논문에 검정 없음), 과제별
    pi0_rows = []
    for j, t in enumerate(["plug", "key", "button"]):
        kf = counts(PI0["FACT_pi0"])[j]
        kb = counts(PI0["pi0"])[j]
        pi0_rows.append({"task": t, "k_FACT_pi0": kf, "k_pi0": kb,
                         "delta_pp": round(PI0["FACT_pi0"][j] - PI0["pi0"][j], 1),
                         "p_greater": round(fisher_greater(kf, kb), 4)})
    res["pi0_tests_not_in_paper"] = pi0_rows

    # 10. 롤아웃 수 세기: 표에 나오는 실행(run)을 중복 없이 센다.
    #     같은 방법·과제가 여러 표에 나오면 같은 실행으로 본다(main의 FACT 등). 백본(pi0)과 노이즈 대체는 별도 실행이다.
    runs = set()
    for m in MAIN:
        for t in TASKS:
            runs.add(("main", m, t))
    for m in BASE_LN:
        for t in ["plug", "key", "button"]:
            runs.add(("main" if m in MAIN else "ln_base", m if m in MAIN else m, t))
    for name in ABL:
        for t in ["plug", "key", "button"]:
            runs.add(("main", "FACT", t) if name == "FACT" else ("abl", name, t))
    for m in PI0:
        for t in ["plug", "key", "button"]:
            runs.add(("pi0", m, t))
    for m in NOISE:
        for t in ["plug", "key", "button"]:
            runs.add(("noise", m, t))
    res["table_runs_unique"] = len(runs)
    res["rollouts_from_tables"] = len(runs) * N
    res["paper_claim_rollouts"] = "nearly 2,500 (논문 초록·§5)"

    # 11. 검정 파워: n=40 대 40, 단측 Fisher 0.05에서 기준 성공률별 최소 검출 차이(시뮬레이션 2000회)
    rng = np.random.default_rng(0)
    power_rows = []
    for p0 in [0.125, 0.30, 0.50]:
        found = None
        for d in np.arange(0.0, 0.7, 0.025):
            p1 = min(0.999, p0 + d)
            k1 = rng.binomial(N, p1, size=2000)
            k0 = rng.binomial(N, p0, size=2000)  # 기준 쪽 표본은 같은 분포에서 추가 샘플링
            hits = 0
            for a, b in zip(k1, k0):
                if fisher_greater(int(a), int(b)) <= 0.05:
                    hits += 1
            pw = hits / 2000
            if pw >= 0.8:
                found = (round(float(d) * 100, 1), round(pw, 3))
                break
        power_rows.append({"baseline_rate_pct": p0 * 100, "min_detectable_pp_power80": found})
    res["power_n40_fisher_one_sided"] = power_rows

    # 12. 본문 Q2 주장: 힘 주입의 추가 효과 = FACT 대 π0.5+LN (논문은 이 검정을 보고하지 않음)
    fl = []
    for j, t in enumerate(TASKS):
        fl.append({"task": t, "k_FACT": main_k["FACT"][j], "k_LN": main_k["pi05_LN"][j],
                   "delta_pp": round(MAIN["FACT"][j] - MAIN["pi05_LN"][j], 1),
                   "p_FACT_greater_than_LN": round(fisher_greater(main_k["FACT"][j], main_k["pi05_LN"][j]), 4)})
    res["FACT_vs_LN_not_in_paper"] = fl

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("wrote", OUT)
    print("main avg recomputed:", res["main_avg_recomputed"])
    print("main avg reported  :", res["main_avg_reported"])
    for r in rows:
        print(f'{r["method"]:9s} {r["task"]:7s} k={r["k"]:2d}/{N} base={r["k_base"]:2d} p_rep={r["p_reported"]:.3f} p_re={r["p_recomputed"]:.4f} holm={r["p_holm_20"]:.4f} CI={r["wilson95"]}')
    print("ln on baselines:", [(r["base"], r["task"], r["delta_pp"], r["p_ln_greater"]) for r in ln_rows])
    print("noise:", [(r["method"], r["task"], r["p_reported"], r["p_recomputed_one_sided"]) for r in noise_rows])
    print("ablation:", [(r["ablation"], r["task"], r["delta_pp"], r["p_FACT_greater"], r["p_holm_9"]) for r in abl_rows])
    print("pi0:", [(r["task"], r["delta_pp"], r["p_greater"]) for r in pi0_rows])
    print("runs:", res["table_runs_unique"], "rollouts:", res["rollouts_from_tables"])
    print("power:", power_rows)
    print("FACT vs LN:", [(r["task"], r["delta_pp"], r["p_FACT_greater_than_LN"]) for r in fl])
    print("pooled:", pooled)
    print("diffs:", res["diff_FACT_minus_ForceVLA_pooled_pp"], res["diff_FACT_minus_pi05_pooled_pp"])
    print("text claims:", res["text_claims"])


if __name__ == "__main__":
    main()
