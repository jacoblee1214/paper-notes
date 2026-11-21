"""RealDexUMI (arXiv:2606.06033 v2) 표 수치 재계산과 검정.

입력: 논문 본문과 부록의 표 값 (paper/paper.txt, pdftotext -layout 기준 v2). 원시 시도 기록은 공개되지 않았다.
      표의 비율은 n = 20 시도(표 2, 3, 7, 실로봇 초기 자세)와 n = 10(그림 8, 그림에서 읽은 값)에서 나온 것으로 본다.
출력: analysis/recompute_results.json, 표준 출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, statsmodels; 수 초)
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "recompute_results.json"
N = 20  # 표 2, 3, 7의 시도 수 (논문 5.1, 부록 C)

TASKS = ["Cube", "Multi-obj", "Plug", "Cap", "Tea", "Drawer", "Egg", "Biman"]
# 표 2 (v2). 값은 [0, 1] 비율 → 20회 기준 정수 횟수
T2 = {
    "RealDexUMI": [1.00, 1.00, 0.85, 1.00, 0.80, 0.85, 0.70, 0.90],
    "w/o tactile": [0.90, 1.00, 0.45, 0.80, 0.55, 0.70, 0.60, 0.60],
    "State-as-action": [0.65, 0.85, 0.30, 0.35, 0.20, 0.45, 0.60, 0.70],
}
T2_AVG_REPORTED = {"RealDexUMI": 88.75, "w/o tactile": 70.00, "State-as-action": 51.25}
# 표 7 Diffusion Policy (부록 D.1), 같은 8개 과제
T7 = [0.85, 0.75, 0.25, 0.75, 0.40, 0.60, 0.80, 0.70]
T7_AVG_REPORTED = 63.75
# 표 3 교차 로봇 (FR3, RM65, Adam-U), 각 20회
T3 = {"Cube": [1.00, 1.00, 1.00], "Drawer": [0.85, 0.75, 0.80]}
# 표 8 누적 하위 목표 (부록 E.1), 각 20회
T8 = {
    "Drawer": [20, 18, 17],
    "Tea": [17, 16, 16],
    "Egg": [15, 14, 14],
    "Multi-obj": [20, 20],
}
# 그림 6: 초기 자세 4개 × 5회 (cube pick-and-place)
INIT_POSE = {"left": 5, "right": 5, "center": 5, "raised-center": 5}
# 그림 8 (5.3): 막대 위 라벨에서 읽은 성공 횟수(n = 10), 성공 시도 평균 시간(초)
F8_SUCC = {"Cap": {"Human": 10, "AVP": 2, "Manus": 10, "RealDexUMI": 10},
           "Tea": {"Human": 10, "AVP": 1, "Manus": 4, "RealDexUMI": 10}}
F8_TIME = {"Cap": {"Human": 6, "AVP": 273, "Manus": 94, "RealDexUMI": 40},
           "Tea": {"Human": 15, "AVP": 214, "Manus": 86, "RealDexUMI": 59}}
# 표 9 (부록 G) 투표 수 (20명)
T9 = {"UMI": (18, 2, 0, "Low"), "DexViTac": (15, 5, 0, "Low"), "DEXOP": (0, 0, 20, "High"),
      "DexExo": (1, 6, 13, "High"), "DEX-Mouse": (1, 5, 14, "High"), "Exo-ViHa": (0, 1, 19, "High"),
      "DexUMI": (6, 10, 4, "Medium"), "RealDexUMI": (16, 4, 0, "Low")}
LEVELS = ["Low", "Medium", "High"]


def to_counts(fracs, n=N):
    out = []
    for f in fracs:
        k = round(f * n)
        assert abs(k / n - f) < 1e-9, (f, n)
        out.append(int(k))
    return out


def wilson(k, n, alpha=0.05):
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return float(lo), float(hi)


def clopper(k, n, alpha=0.05):
    lo, hi = proportion_confint(k, n, alpha=alpha, method="beta")
    return float(lo), float(hi)


def newcombe_diff(k1, n1, k2, n2, alpha=0.05):
    """두 비율 차이 p1 - p2의 Newcombe(1998) 점수 구간."""
    l1, u1 = wilson(k1, n1, alpha)
    l2, u2 = wilson(k2, n2, alpha)
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    lo = d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return float(d), float(lo), float(hi)


def fisher(k1, n1, k2, n2):
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(stats.fisher_exact(table, alternative="two-sided")[1])


def main():
    res = {"source": "arXiv:2606.06033 v2 (2026-06-06), 표 2·3·7·8·9, 그림 6·8 값", "checks": {}}

    # --- 표 2: 셀별 횟수, Wilson 95% 구간 ---
    counts = {m: to_counts(v) for m, v in T2.items()}
    res["table2_counts"] = counts
    res["table2_wilson"] = {m: [wilson(k, N) for k in v] for m, v in counts.items()}

    # V1: 평균 재계산 (과제 수 동일하므로 비가중 평균 = 합 / 160)
    for m, v in T2.items():
        avg = 100 * np.mean(v)
        pooled = 100 * sum(counts[m]) / (N * len(TASKS))
        res["checks"][f"V1_avg_{m}"] = {"recomputed": round(avg, 2), "pooled": round(pooled, 2),
                                         "reported": T2_AVG_REPORTED[m]}
    res["checks"]["V1_avg_DP"] = {"recomputed": round(100 * np.mean(T7), 2), "reported": T7_AVG_REPORTED}

    # 합계 기준 풀링 (8개 과제 × 20회)
    pooled = {m: (sum(c), N * len(TASKS)) for m, c in counts.items()}
    dp_counts = to_counts(T7)
    pooled["Diffusion Policy"] = (sum(dp_counts), N * len(TASKS))
    res["pooled_counts"] = {m: list(v) for m, v in pooled.items()}
    res["pooled_wilson"] = {m: wilson(*v) for m, v in pooled.items()}

    # --- 풀링 비교 (독립 표본 Fisher, Newcombe 차이 구간) ---
    comps = {}
    for other in ["w/o tactile", "State-as-action"]:
        k1, n1 = pooled["RealDexUMI"]
        k2, n2 = pooled[other]
        d, lo, hi = newcombe_diff(k1, n1, k2, n2)
        comps[f"RealDexUMI vs {other}"] = {"diff_pts": round(100 * d, 2), "newcombe95_pts": [round(100 * lo, 2), round(100 * hi, 2)],
                                          "fisher_p": fisher(k1, n1, k2, n2)}
    k1, n1 = pooled["RealDexUMI"]
    k2, n2 = pooled["Diffusion Policy"]
    d, lo, hi = newcombe_diff(k1, n1, k2, n2)
    comps["RealDexUMI vs Diffusion Policy"] = {"diff_pts": round(100 * d, 2), "newcombe95_pts": [round(100 * lo, 2), round(100 * hi, 2)],
                                              "fisher_p": fisher(k1, n1, k2, n2)}
    res["pooled_comparisons"] = comps

    # --- 과제별 Fisher (Holm 보정, 과제 8개 묶음) ---
    per_task = {}
    for other in ["w/o tactile", "State-as-action"]:
        ps = [fisher(counts["RealDexUMI"][i], N, counts[other][i], N) for i in range(len(TASKS))]
        rej, p_holm, _, _ = multipletests(ps, method="holm")
        per_task[f"RealDexUMI vs {other}"] = {t: {"p_raw": round(p, 4), "p_holm": round(q, 4)}
                                             for t, p, q in zip(TASKS, ps, p_holm)}
    ps = [fisher(counts["RealDexUMI"][i], N, dp_counts[i], N) for i in range(len(TASKS))]
    _, p_holm, _, _ = multipletests(ps, method="holm")
    per_task["RealDexUMI vs Diffusion Policy"] = {t: {"p_raw": round(p, 4), "p_holm": round(q, 4)}
                                                 for t, p, q in zip(TASKS, ps, p_holm)}
    res["per_task_fisher"] = per_task

    # 과제를 층으로 둔 Cochran-Mantel-Haenszel 검정 (풀링 Fisher의 과제 구조 보정)
    from statsmodels.stats.contingency_tables import StratifiedTable
    cmh = {}
    for other, arr in [("w/o tactile", counts["w/o tactile"]), ("State-as-action", counts["State-as-action"]),
                       ("Diffusion Policy", dp_counts)]:
        tabs = [np.array([[counts["RealDexUMI"][i], N - counts["RealDexUMI"][i]],
                          [arr[i], N - arr[i]]]) for i in range(len(TASKS))]
        st = StratifiedTable(tabs)
        cmh[f"RealDexUMI vs {other}"] = {"mh_or": round(float(st.oddsratio_pooled), 2),
                                        "cmh_p": float(st.test_null_odds().pvalue)}
    res["cmh_stratified_by_task"] = cmh

    # 검정력: n = 20 대 20, 양측 0.05, 검정력 0.8에서 찾을 수 있는 최소 차이 (정규 근사, Cohen h)
    power = NormalIndPower()
    mde = {}
    for base in [0.5, 0.7, 0.85]:
        for p2 in np.arange(base, 1.0001, 0.005):
            h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(base)))
            if abs(h) < 1e-9:
                continue
            if power.power(effect_size=abs(h), nobs1=N, alpha=0.05, ratio=1.0) >= 0.8:
                mde[str(base)] = round(float(p2 - base), 3) if p2 > base else round(float(base - p2), 3)
                break
    res["mde_n20_power80"] = mde

    # --- 표 3 교차 로봇 ---
    xemb = {}
    for task, vals in T3.items():
        k = to_counts(vals)
        xemb[task] = {"counts": k, "wilson": [wilson(x, N) for x in k]}
    tot_k = sum(sum(to_counts(v)) for v in T3.values())
    xemb["total"] = {"counts": tot_k, "of": 120, "pct": round(100 * tot_k / 120, 1), "wilson": wilson(tot_k, 120)}
    d_cmp = {}
    dr = to_counts(T3["Drawer"])
    d_cmp["Drawer FR3 vs RM65"] = fisher(dr[0], N, dr[1], N)
    d_cmp["Drawer FR3 vs Adam-U"] = fisher(dr[0], N, dr[2], N)
    d_cmp["Drawer RM65 vs Adam-U"] = fisher(dr[1], N, dr[2], N)
    xemb["fisher_drawer"] = d_cmp
    xemb["fr3_equals_table2"] = {"Cube": T3["Cube"][0] == T2["RealDexUMI"][0], "Drawer": T3["Drawer"][0] == T2["RealDexUMI"][5]}
    res["table3"] = xemb

    # --- 초기 자세 (그림 6): 20/20 ---
    k = sum(INIT_POSE.values())
    res["init_pose"] = {"counts": k, "of": 20, "wilson": wilson(k, 20), "clopper_pearson": clopper(k, 20),
                        "pose_counts": INIT_POSE}

    # --- 표 8 단조성과 표 2 최종값 일치 ---
    col = {"Drawer": 5, "Tea": 4, "Egg": 6, "Multi-obj": 1}
    mono = {}
    for task, seq in T8.items():
        final_table2 = counts["RealDexUMI"][col[task]]
        mono[task] = {"nonincreasing": all(a >= b for a, b in zip(seq, seq[1:])),
                      "final_subgoal": seq[-1], "table2_count": final_table2,
                      "match": seq[-1] == final_table2}
    res["checks"]["V4_table8"] = mono

    # --- 그림 8 (n = 10 추정치, 시간 SD 없음) ---
    f8 = {}
    for task in F8_SUCC:
        f8[task] = {m: {"succ": k, "of": 10, "wilson": wilson(k, 10)} for m, k in F8_SUCC[task].items()}
        f8[task]["fisher_RDU_vs_Manus_p"] = fisher(F8_SUCC[task]["RealDexUMI"], 10, F8_SUCC[task]["Manus"], 10)
        f8[task]["fisher_RDU_vs_AVP_p"] = fisher(F8_SUCC[task]["RealDexUMI"], 10, F8_SUCC[task]["AVP"], 10)
    f8["time_s"] = F8_TIME
    res["fig8"] = f8

    # --- 표 9 투표 ---
    survey = {}
    for sysname, (lo_v, me_v, hi_v, final) in T9.items():
        votes = [lo_v, me_v, hi_v]
        plural = LEVELS[int(np.argmax(votes))]
        survey[sysname] = {"votes": votes, "sum": sum(votes), "plurality": plural,
                           "reported_final": final, "consistent": plural == final}
    res["checks"]["V5_survey"] = survey
    res["survey_RDU_low_wilson"] = wilson(16, 20)

    # --- 데이터 규모 산술 ---
    episodes_min = 8 * 200
    res["checks"]["V6_data_scale"] = {"hours": 100, "episodes_min": episodes_min,
                                     "mean_episode_s_upper": round(100 * 3600 / episodes_min, 1)}

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- 요약 출력 ---
    print("표 2 풀링 (k/n):", {m: f"{v[0]}/{v[1]}" for m, v in pooled.items()})
    for m, v in pooled.items():
        lo, hi = res["pooled_wilson"][m]
        print(f"  {m}: {100*v[0]/v[1]:.2f}%  Wilson95 [{100*lo:.1f}, {100*hi:.1f}]")
    for name, c in comps.items():
        print(f"  {name}: diff {c['diff_pts']} pts, Newcombe95 {c['newcombe95_pts']}, Fisher p = {c['fisher_p']:.2e}")
    print("과제별 Holm 보정 p (RDU vs w/o tactile):")
    for t, v in per_task["RealDexUMI vs w/o tactile"].items():
        print(f"  {t}: raw {v['p_raw']}, holm {v['p_holm']}")
    print("과제별 Holm 보정 p (RDU vs Diffusion Policy):")
    for t, v in per_task["RealDexUMI vs Diffusion Policy"].items():
        print(f"  {t}: raw {v['p_raw']}, holm {v['p_holm']}")
    print("과제 층화 CMH:", {k: f"OR {v['mh_or']}, p {v['cmh_p']:.2e}" for k, v in cmh.items()})
    print("검정력 0.8 MDE (n=20 대 20):", mde)
    print("표 3 총계:", xemb["total"]["counts"], "/ 120 =", xemb["total"]["pct"], "%")
    print("표 3 드로어 Fisher:", {k: round(v, 3) for k, v in d_cmp.items()})
    print("초기 자세 20/20 Wilson:", [round(x, 3) for x in res["init_pose"]["wilson"]],
          "Clopper-Pearson:", [round(x, 3) for x in res["init_pose"]["clopper_pearson"]])
    print("표 8 검사:", {k: (v["nonincreasing"], v["match"]) for k, v in mono.items()})
    print("그림 8 Fisher (cap RDU vs Manus p, tea RDU vs Manus p, tea RDU vs AVP p):",
          round(f8["Cap"]["fisher_RDU_vs_Manus_p"], 3), round(f8["Tea"]["fisher_RDU_vs_Manus_p"], 4),
          round(f8["Tea"]["fisher_RDU_vs_AVP_p"], 5))
    print("표 9 일관성:", {k: v["consistent"] for k, v in survey.items()})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
