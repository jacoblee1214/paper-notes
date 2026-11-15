"""논문 표 1–3 수치를 다시 계산하고 일관성을 검사한다 (numpy, scipy, 수 초).

실행: python3 analysis/recompute.py  → analysis/recompute_results.json
입력은 논문 PDF 본문(paper/paper.txt)의 표 값을 옮긴 것이다. 원자료는 없다.
가정: 시뮬레이션 "±"는 3 시드의 표본 표준편차(ddof=1)로 본다(SPIDER는 5 시드). 논문에 정의가 없어 민감도도 본다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"

# 표 2 (실로봇, 10 시도). (성공 수, 시도 수)
REAL_T2 = {
    "LEAP-Scissors": {"REGRIND": (9, 10), "DexMachina": (0, 10), "Mink IK + RL": None},
    "LEAP-Screwdriver": {"REGRIND": (10, 10), "DexMachina": (2, 10), "Mink IK + RL": None},
    "WUJI-Scissors": {"REGRIND": (0, 10), "DexMachina": None, "Mink IK + RL": None},
    "WUJI-Screwdriver": {"REGRIND": (9, 10), "DexMachina": (5, 10), "Mink IK + RL": (0, 10)},
}
# 표 3 (REGRIND, 무작위 초기 물체 배치, 10 시도)
REAL_T3 = {"LEAP-Scissors": (8, 10), "LEAP-Screwdriver": (10, 10), "WUJI-Screwdriver": (9, 10)}

# 표 1 (시뮬레이션). (평균, ±값) 형식. SR은 %.
SIM_T1 = {
    "LEAP-Scissors": {"REGRIND": {"err": (5.6, 6.4), "sr": (99.8, 0.3)},
                      "DexMachina": {"err": (10.1, 9.5), "sr": (22.3, 17.7)}},
    "LEAP-Screwdriver": {"REGRIND": {"err": (5.4, 5.8), "sr": (99.7, 0.0)},
                         "DexMachina": {"err": (8.3, 6.7), "sr": (99.7, 0.1)}},
    "WUJI-Scissors": {"REGRIND": {"err": (5.3, 13.1), "sr": (98.7, 1.3)},
                      "DexMachina": {"err": (67.2, 31.8), "sr": (0.0, 0.0)}},
    "WUJI-Screwdriver": {"REGRIND": {"err": (6.5, 15.5), "sr": (98.8, 1.3)},
                         "DexMachina": {"err": (5.9, 9.7), "sr": (99.3, 0.1)}},
}
N_SEEDS = 3


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def clopper(k, n, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return [float(lo), float(hi)]


def welch(m1, s1, n1, m2, s2, n2):
    """두 평균의 Welch t 검정. 표준편차가 0이면 판정하지 않는다."""
    if s1 == 0 and s2 == 0:
        return None
    t, p = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
    return {"t": float(t), "p_two_sided": float(p)}


def main():
    res = {"assumptions": {"sim_pm": "3 seeds sample sd (ddof=1); SPIDER 5 seeds not compared",
                           "fisher": "two-sided exact test on 10-trial counts",
                           "ci": "Wilson and Clopper-Pearson 95%"}}

    # 1. 실로봇 표 2
    rows = []
    for task, d in REAL_T2.items():
        for meth, v in d.items():
            if v is None:
                continue
            k, n = v
            rows.append({"task": task, "method": meth, "k": k, "n": n, "sr": k / n,
                         "wilson95": wilson(k, n), "cp95": clopper(k, n)})
    res["real_table2"] = rows

    fis = []
    for task, d in REAL_T2.items():
        if d["REGRIND"] is None or d["DexMachina"] is None:
            continue
        a, b = d["REGRIND"], d["DexMachina"]
        odds, p = stats.fisher_exact([[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]])
        fis.append({"comparison": f"REGRIND vs DexMachina ({task})", "p_fisher": float(p)})
    if REAL_T2["WUJI-Screwdriver"]["Mink IK + RL"]:
        a, b = REAL_T2["WUJI-Screwdriver"]["REGRIND"], REAL_T2["WUJI-Screwdriver"]["Mink IK + RL"]
        _, p = stats.fisher_exact([[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]])
        fis.append({"comparison": "REGRIND vs Mink IK + RL (WUJI-Screwdriver)", "p_fisher": float(p)})
    # 같은 세 과제에서 풀링 (DexMachina가 WUJI-Scissors를 보고하지 않음)
    pool_r = [REAL_T2[t]["REGRIND"] for t in ("LEAP-Scissors", "LEAP-Screwdriver", "WUJI-Screwdriver")]
    pool_d = [REAL_T2[t]["DexMachina"] for t in ("LEAP-Scissors", "LEAP-Screwdriver", "WUJI-Screwdriver")]
    kr, nr = sum(x[0] for x in pool_r), sum(x[1] for x in pool_r)
    kd, nd = sum(x[0] for x in pool_d), sum(x[1] for x in pool_d)
    _, p = stats.fisher_exact([[kr, nr - kr], [kd, nd - kd]])
    fis.append({"comparison": "REGRIND vs DexMachina pooled (3 tasks)", "k_n_regrind": [kr, nr],
                "k_n_dexmachina": [kd, nd], "p_fisher": float(p), "wilson_regrind": wilson(kr, nr),
                "wilson_dexmachina": wilson(kd, nd)})
    res["real_tests"] = fis

    # 2. 표 3 vs 표 2 (같은 과제에서 무작위 초기 배치와 고정 배치)
    gen = []
    for task, (k3, n3) in REAL_T3.items():
        k2, n2 = REAL_T2[task]["REGRIND"]
        _, p = stats.fisher_exact([[k2, n2 - k2], [k3, n3 - k3]])
        gen.append({"task": task, "demo_init": [k2, n2], "random_init": [k3, n3], "p_fisher": float(p)})
    res["table3_vs_table2"] = gen

    # 3. 시뮬레이션 표 1: 3 시드 평균의 95% 신뢰구간과 Welch 검정
    tc = stats.t.ppf(0.975, N_SEEDS - 1)
    sim = []
    for task, d in SIM_T1.items():
        for metric in ("sr", "err"):
            r = d["REGRIND"][metric]
            m = d["DexMachina"][metric]
            entry = {"task": task, "metric": metric,
                     "REGRIND": {"mean": r[0], "sd": r[1], "ci95_halfwidth": tc * r[1] / math.sqrt(N_SEEDS)},
                     "DexMachina": {"mean": m[0], "sd": m[1], "ci95_halfwidth": tc * m[1] / math.sqrt(N_SEEDS)}}
            if r[1] > 0 and m[1] > 0:
                entry["welch_ddof1"] = welch(r[0], r[1], N_SEEDS, m[0], m[1], N_SEEDS)
                # 민감도: 논문이 모집단 표준편차(ddof=0)를 썼다면
                sc = math.sqrt(N_SEEDS / (N_SEEDS - 1))
                entry["welch_if_ddof0"] = welch(r[0], r[1] * sc, N_SEEDS, m[0], m[1] * sc, N_SEEDS)
            else:
                entry["welch_ddof1"] = "sd=0 또는 DexMachina 0% (검정 생략)"
            sim.append(entry)
    res["sim_table1"] = sim

    # 4. 시뮬레이션 ± 해석: 1024 롤아웃의 이항 표준오차와 비교
    binom = []
    for task, d in SIM_T1.items():
        p = d["REGRIND"]["sr"][0] / 100
        se = math.sqrt(p * (1 - p) / 1024) * 100
        binom.append({"task": task, "binomial_se_pct_1024": se, "reported_sd_pct": d["REGRIND"]["sr"][1]})
    res["sim_pm_vs_binomial"] = binom

    # 5. 일관성 검사 (산술과 본문 대조)
    checks = {
        "ppo_batch_per_iter": 4096 * 24,
        "ppo_minibatch": 4096 * 24 // 4,
        "ppo_optimizer_passes_per_iter": 5 * 4,
        "episode_control_steps": round(9.0 * 30),
        "control_dt_s": 4 * (1 / 120),
        "lag_1_2_steps_ms": [round(1 * 1000 / 30), round(2 * 1000 / 30)],
        "aug_yaw_30deg_rad": round(math.radians(30), 4),
        "max_iterations_total_transitions_billion": 20000 * 4096 * 24 / 1e9,
        "gravity_full_at_vector_steps_iterations": 130_000 / 24,
        "keypoints_in_code_per_object": 100,
        "keypoints_in_paper_per_object": 50,
    }
    res["consistency"] = checks

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT)
    for r in rows:
        print(f"{r['task']:18s} {r['method']:12s} {r['k']}/{r['n']} SR={r['sr']:.2f} Wilson={[round(x,3) for x in r['wilson95']]}")
    for f in fis:
        print(f["comparison"], "p=", round(f["p_fisher"], 5))
    for g in gen:
        print("T3 vs T2", g["task"], "p=", round(g["p_fisher"], 3))
    for e in sim:
        w = e["welch_ddof1"]
        print(e["task"], e["metric"], "Welch:", None if isinstance(w, str) else round(w["p_two_sided"], 4))


if __name__ == "__main__":
    main()
