"""HiFun (arXiv:2610.19518 v1) 논문 표 수치를 다시 계산하고 통계 검정을 한다.

입력: 논문 본문과 표의 값을 아래에 그대로 옮겨 적었다(출처는 주석). 외부 데이터는 쓰지 않는다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 표 1의 분모는 부록 B.3의 "각 과제 50회"다. 반 점(0.5 크레딧) 규칙 때문에 성공 "수"는 정수가 아닐 수 있다.
  Fisher·Wilson은 정수 성공 수가 필요하므로 반 점이 있으면 내림과 올림 두 값을 모두 계산해 p는 큰 쪽(보수적)을 쓴다.
- 표 2(절제, 데이터 원천)의 시행 수는 논문에 없다. 50회(본 평가)와 20회(교란 평가) 두 경우를 모두 계산한다.
- 표 3은 20회, 표 6은 본 평가 프로토콜(50회, 본문 캡션)이다.
- 표 1 Skill-HIL 평균 46%의 과제별 값은 논문에 없으므로 합산(과제당 50회 가정, 92/200)으로만 비교한다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="two-sided")[1])


def fisher_cons(p1, n1, p2, n2):
    """백분율 두 개(0–100)를 받아 반 점이면 내림·올림 네 조합 중 가장 큰 p를 쓴다."""
    def ks(p, n):
        x = p * n / 100.0
        return sorted({math.floor(x), math.ceil(x)})
    ps = [fisher(a, n1, b, n2) for a in ks(p1, n1) for b in ks(p2, n2)]
    return max(ps), min(ps)


def newcombe_diff(k1, n1, k2, n2, z=1.959964):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def n_per_arm(p1, p2, alpha=0.05, power=0.8):
    """두 비율의 정규 근사(arcsine 변환) 표본 크기, 팔당."""
    h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return math.ceil(2 * ((za + zb) / h) ** 2)


TASKS = ["Power-Drill Actuation", "Angle-Spreader Retrieval", "Thin-Handle Retrieval",
         "Power-Drill Bit Removal", "Pipette Dispensing", "Thin-Shaft Tool Retrieval"]

# ---- 표 1 (본문 6쪽). None = "–"
T1 = {
    "il_min":     [150, 60, 80, 80, 100, 100],
    "rl_data":    [34, 13, 17, 21, 14, 14],
    "rl_online":  [60, 55, 60, 22, 60, 20],
    "glove":      [15, None, None, None, 12, None],
    "interv_app": [62, 40, 28, 60, 60, 30],    # HIL-SERL (Ours Interv.) 팔 접근 성공(a)
    "interv_fin": [30, 0, 0, 20, 32, 0],       # 같은 열의 최종 성공(b)
    "hifun":      [100, 100, 100, 100, 90, 100],
    "bc":         [15, 30, 35, 30, 12, 20],
    "dp":         [30, 60, 40, 36, 26, 58],
}
T1_AVG = {"il_min": 95.0, "rl_data": 18.8, "rl_online": 46.2, "glove": 4.5, "interv_fin": 13.7,
          "hifun": 98.3, "bc": 23.7, "dp": 41.7}
# ---- 표 5 (부록 C.1)
T5 = {"classifier": [4, 13, 11, 7, 10, 7.5], "dmp": [34, 13, 17, 21, 14, 14], "coord": [60, 55, 60, 22, 60, 20]}
# ---- 표 2 (본문 8쪽)
T2_DATA = {"Angle-Spreader Retrieval": {"dmp": 100, "human": 85, "t_dmp": 13, "t_human": 60},
           "Power-Drill Bit Removal": {"dmp": 100, "human": 100, "t_dmp": 21, "t_human": 80}}
T2_ABL = {"Angle-Spreader Retrieval": {"full": 100, "no_multihead": 85, "uniform_bc": 60, "no_interv_bc": 40},
          "Power-Drill Bit Removal": {"full": 100, "no_multihead": 75, "uniform_bc": 60, "no_interv_bc": 40}}
# ---- 표 3 (본문 9쪽, 20회)
T3 = {"Power-Drill Actuation": (20, 0, 18, 19), "Angle-Spreader Retrieval": (20, 0, 7, 5),
      "Thin-Handle Retrieval": (20, 0, 0, 0), "Power-Drill Bit Removal": (20, 5, 3, 3),
      "Pipette Dispensing": (20, 0, 20, 20), "Thin-Shaft Tool Retrieval": (20, 0, 0, 0)}  # ours, replay, dexpilot, anyteleop
# ---- 표 6 (부록 D.2)
T6 = {"bc_human": [15, 30, 35, 30], "bc_hifun": [12, 32, 32, 28], "dp_human": [30, 60, 40, 36], "dp_hifun": [26, 64, 40, 40]}
T6_AVG = {"bc_human": 27.5, "bc_hifun": 26.0, "dp_human": 41.5, "dp_hifun": 42.5}


def mean(xs):
    xs = [0 if x is None else x for x in xs]
    return sum(xs) / len(xs)


def main():
    # V1. 표 1 열 평균
    v1 = {}
    for k, rep in T1_AVG.items():
        calc = mean(T1[k])
        v1[k] = {"reported": rep, "recomputed": round(calc, 2), "match": abs(calc - rep) < 0.06}
    v1["interv_app_mean"] = round(mean(T1["interv_app"]), 2)
    v1["glove_mean_over_reported_cells"] = round((15 + 12) / 2, 2)
    RES["V1_table1_means"] = v1

    # V2. 표 1 vs 표 5 (같은 열), 표 5 합계와 범위
    RES["V2_table5"] = {
        "dmp_equals_table1": T5["dmp"] == T1["rl_data"],
        "coord_equals_table1": T5["coord"] == T1["rl_online"],
        "classifier_sum_min": sum(T5["classifier"]), "classifier_mean_min": round(mean(T5["classifier"]), 2),
        "classifier_range": [min(T5["classifier"]), max(T5["classifier"])],
        "per_task_robot_plus_attended": [round(c + d + o, 1) for c, d, o in zip(T5["classifier"], T5["dmp"], T5["coord"])],
        "mean_total_reported_components": round(mean([c + d + o for c, d, o in zip(T5["classifier"], T5["dmp"], T5["coord"])]), 2),
        "mean_attended_online_plus_classifier": round(mean([c + o for c, o in zip(T5["classifier"], T5["coord"])]), 2),
    }

    # V3. 정수·반 점 가능성
    odd = {}
    for k in ["glove", "interv_app", "interv_fin", "hifun", "bc", "dp"]:
        odd[k] = [v for v in T1[k] if v is not None and v % 2 == 1]
    odd["table6"] = {k: [v for v in vs if v % 2 == 1] for k, vs in T6.items()}
    odd["table2_ablation"] = {t: {k: v for k, v in d.items() if v % 2 == 1} for t, d in T2_ABL.items()}
    odd["table2_data"] = {t: d["human"] for t, d in T2_DATA.items() if d["human"] % 2 == 1}
    feas20 = {"table2_all_multiples_of_5": all(v % 5 == 0 for d in T2_ABL.values() for v in d.values())
              and all(d["human"] % 5 == 0 for d in T2_DATA.values())}
    RES["V3_half_credit_needed_if_n50"] = {"odd_percent_values": odd, "table2_integer_at_n20": feas20,
                                           "note": "n=50에서 홀수 %는 성공 수가 xx.5라 0.5 크레딧 시행이 1회 이상 필요"}

    # V4. HiFun Wilson
    k = [round(v / 2) for v in T1["hifun"]]  # n=50
    w = {t: [round(100 * x, 1) for x in wilson(kk, 50)] for t, kk in zip(TASKS, k)}
    pooled = sum(k)
    RES["V4_hifun_wilson"] = {"counts_n50": dict(zip(TASKS, k)), "wilson95_pct": w, "pooled": [pooled, 300],
                              "pooled_pct": round(100 * pooled / 300, 2),
                              "pooled_wilson95_pct": [round(100 * x, 1) for x in wilson(pooled, 300)],
                              "all100_wilson_lower_n50": round(100 * wilson(50, 50)[0], 1)}

    # V5. 과제별 Fisher: HiFun 대 각 기준선 (n=50), Holm
    comps = {}
    for base in ["interv_fin", "bc", "dp"]:
        rows, ps = [], []
        for t, h, b in zip(TASKS, T1["hifun"], T1[base]):
            pmax, pmin = fisher_cons(h, 50, b, 50)
            rows.append({"task": t, "hifun": h, "base": b, "p_cons": pmax, "p_min": pmin})
            ps.append(pmax)
        adj = holm(ps)
        for r, a in zip(rows, adj):
            r["holm_p"] = a
        comps[base] = rows
    # 최강 기준선 대비(과제별 BC/DP/Interv 중 최고)
    best = [max(T1["interv_fin"][i], T1["bc"][i], T1["dp"][i]) for i in range(6)]
    rows = []
    for t, h, b in zip(TASKS, T1["hifun"], best):
        pmax, pmin = fisher_cons(h, 50, b, 50)
        rows.append({"task": t, "hifun": h, "best_baseline": b, "p_cons": pmax})
    comps["vs_best_per_task"] = rows
    # 합산
    for base in ["interv_fin", "bc", "dp"]:
        kb = sum(T1[base]) / 2  # 반 점 가능
        kb_i = int(round(kb))
        comps[base + "_pooled"] = {"hifun": [pooled, 300], "base_k": kb, "p": fisher(pooled, 300, kb_i, 300),
                                   "newcombe_diff_pp": [round(100 * x, 1) for x in newcombe_diff(pooled, 300, kb_i, 300)]}
    RES["V5_fisher_table1"] = comps

    # V6. 표 2 (n 가정 두 가지)
    v6 = {}
    for n in (20, 50):
        rows = []
        for t, d in T2_ABL.items():
            for var in ["no_multihead", "uniform_bc", "no_interv_bc"]:
                pmax, pmin = fisher_cons(d["full"], n, d[var], n)
                rows.append({"task": t, "variant": var, "full": d["full"], "var": d[var], "p_cons": pmax})
        adj = holm([r["p_cons"] for r in rows])
        for r, a in zip(rows, adj):
            r["holm_p"] = a
        v6[f"ablation_n{n}"] = rows
        rows = []
        for t, d in T2_DATA.items():
            pmax, pmin = fisher_cons(d["dmp"], n, d["human"], n)
            rows.append({"task": t, "dmp": d["dmp"], "human": d["human"], "p_cons": pmax})
        v6[f"data_source_n{n}"] = rows
    # 반 점 없이 n=50 불가능 확인
    v6["n_needed_100_vs_85_80pct_power"] = n_per_arm(0.985, 0.85)  # 100%는 0.985로 보정
    v6["n_needed_100_vs_75_80pct_power"] = n_per_arm(0.985, 0.75)
    RES["V6_table2"] = v6

    # V7. 데이터 수집 시간 비
    RES["V7_data_time"] = {
        "ratio_total": {t: round(d["t_human"] / d["t_dmp"], 2) for t, d in T2_DATA.items()},
        "per_trajectory_min_dmp_n100": {t: round(d["t_dmp"] / 100, 3) for t, d in T2_DATA.items()},
        "per_trajectory_min_human_n200": {t: round(d["t_human"] / 200, 3) for t, d in T2_DATA.items()},
        "ratio_per_trajectory_if_human200_dmp100": {t: round((d["t_human"] / 200) / (d["t_dmp"] / 100), 2) for t, d in T2_DATA.items()},
        "table1_il_equals_table2_human_time": [T1["il_min"][1], T1["il_min"][3]] == [60, 80],
    }

    # V8. 표 3
    v8 = []
    for t, (o, r, d, a) in T3.items():
        for name, kk in (("replay", r), ("dexpilot", d), ("anyteleop", a)):
            v8.append({"task": t, "alt": name, "ours": o, "alt_k": kk, "p": fisher(o, 20, kk, 20)})
    adj = holm([x["p"] for x in v8])
    for x, a in zip(v8, adj):
        x["holm_p"] = a
    ours_total = sum(v[0] for v in T3.values())
    RES["V8_table3"] = {"rows": v8, "ours_total": [ours_total, 120],
                        "ours_wilson95_pct": [round(100 * x, 1) for x in wilson(ours_total, 120)],
                        "zero_of_20_wilson_upper_pct": round(100 * wilson(0, 20)[1], 1),
                        "column_totals_of_120": {"replay": sum(v[1] for v in T3.values()), "dexpilot": sum(v[2] for v in T3.values()),
                                                  "anyteleop": sum(v[3] for v in T3.values())}}

    # V9. 표 6
    v9 = {}
    for kind in ("bc", "dp"):
        rows = []
        for t, a, b in zip(TASKS[:4], T6[f"{kind}_human"], T6[f"{kind}_hifun"]):
            pmax, _ = fisher_cons(a, 50, b, 50)
            rows.append({"task": t, "human": a, "hifun_data": b, "p": pmax})
        ka, kb = round(sum(T6[f"{kind}_human"]) / 2), round(sum(T6[f"{kind}_hifun"]) / 2)
        d, lo, hi = newcombe_diff(kb, 200, ka, 200)
        v9[kind] = {"rows": rows, "avg_human": mean(T6[f"{kind}_human"]), "avg_hifun_data": mean(T6[f"{kind}_hifun"]),
                    "avg_reported_match": [abs(mean(T6[f"{kind}_human"]) - T6_AVG[f"{kind}_human"]) < 0.01,
                                           abs(mean(T6[f"{kind}_hifun"]) - T6_AVG[f"{kind}_hifun"]) < 0.01],
                    "pooled_diff_pp_hifun_data_minus_human": round(100 * d, 1),
                    "newcombe95_pp": [round(100 * lo, 1), round(100 * hi, 1)],
                    "pooled_fisher_p": fisher(kb, 200, ka, 200)}
    v9["table6_matches_table1_human_columns"] = (T6["bc_human"] == T1["bc"][:4]) and (T6["dp_human"] == T1["dp"][:4])
    RES["V9_table6"] = v9

    # V10. Skill-HIL, 강건성
    skill_pooled = round(0.46 * 200)
    RES["V10_skill_hil"] = {
        "doubling_35_of_50_pct": 70.0, "angle_35_vs_50_p": fisher(35, 50, 50, 50),
        "pooled_92_of_200_vs_200_of_200_p": fisher(skill_pooled, 200, 200, 200),
        "skill_avg46_wilson_n200_pct": [round(100 * x, 1) for x in wilson(skill_pooled, 200)],
    }
    rob = [20, 20, 20, 20, 18, 18]
    RES["V11_robustness"] = {"sum": [sum(rob), 120], "pct": round(100 * sum(rob) / 120, 2),
                             "wilson95_pct": [round(100 * x, 1) for x in wilson(sum(rob), 120)],
                             "angle_19_of_20_wilson": [round(100 * x, 1) for x in wilson(19, 20)],
                             "bit_18_of_20_wilson": [round(100 * x, 1) for x in wilson(18, 20)],
                             "table1_vs_robust_note": "표 1 Power-Drill Actuation 50/50 대 교란 18/20, Pipette 45/50 대 18/20 (서로 다른 조건)"}

    # V12. Fig. 8A 일관성: p95 오차와 2 cm 밖 활성화 비율
    RES["V12_fig8A"] = {
        "claim": "multi-head: 95% of activations within 2.00 cm (본문 4.3), p95=2.00 cm, outside-2cm=3.7% (부록 D.4)",
        "consistent_ours": "밖 3.7% < 5%이면 p95 ≤ 2.00 cm → 일치",
        "consistent_single": "밖 6.0% > 5%이면 p95 > 2.00 cm → 4.58 cm와 모순 없음",
    }

    # V13. IAW: 가중치 0.175, 0.884가 말하는 온도
    rng = np.random.default_rng(0)
    z = rng.standard_normal(200000)
    z = (z - z.mean()) / z.std()
    out = []
    for tau in (0.1, 0.2, 0.3, 0.5, 1.0):
        w_all = 1 / (1 + np.exp(-z / tau))
        lo_q, hi_q = np.quantile(z, [0.25, 0.75])
        out.append({"tau": tau, "mean_w_all": round(float(w_all.mean()), 3),
                    "mean_w_bottom_quartile": round(float(w_all[z <= lo_q].mean()), 3),
                    "mean_w_top_quartile": round(float(w_all[z >= hi_q].mean()), 3),
                    "mean_w_bottom_half": round(float(w_all[z <= 0].mean()), 3),
                    "mean_w_top_half": round(float(w_all[z > 0].mean()), 3)})
    RES["V13_iaw_temperature"] = {"note": "Norm(A)를 z-점수, 정규 분포로 가정. 논문은 τ_I와 Norm의 정의를 쓰지 않는다. 보고된 두 평균은 0.175와 0.884.",
                                  "table": out}

    # V14. 필요 시행 수 (HiFun 100% 대 BC 최고 35%는 작은 n으로 충분, 대 85%는 크다)
    RES["V14_power"] = {"n_per_arm_100_vs_90": n_per_arm(0.985, 0.90), "n_per_arm_100_vs_85": n_per_arm(0.985, 0.85),
                        "n_per_arm_90_vs_60": n_per_arm(0.90, 0.60)}

    # V15. 시간 회계: 표 1이 담는 것과 담지 않는 것
    mean_il, mean_rl = mean(T1["il_min"]), mean(T1["rl_data"])
    RES["V15_cost_accounting"] = {
        "mean_il_demo_minutes": mean_il,
        "mean_hifun_reported": round(mean(T5["classifier"]) + mean(T5["dmp"]) + mean(T5["coord"]), 2),
        "mean_hifun_human_attended_reported": round(mean(T5["classifier"]) + mean(T5["coord"]), 2),
        "unreported": ["손 기술 잔차 RL 시간", "기준 시연 1회 (Gello)", "손 기술 교시 5분 미만(기준 시연당, 부록 C.1)",
                       "리셋 인력", "분류기 정확도 95% 검증"],
    }

    # V16. HIL-SERL (Glove+Gello) 대 (Ours Interv.): 값이 있는 두 과제만
    v16 = []
    for i in (0, 4):
        pmax, pmin = fisher_cons(T1["interv_fin"][i], 50, T1["glove"][i], 50)
        v16.append({"task": TASKS[i], "interv": T1["interv_fin"][i], "glove": T1["glove"][i], "p_cons": pmax, "p_min": pmin})
    pm = max(fisher(31, 100, 13, 100), fisher(31, 100, 14, 100))
    RES["V16_glove_vs_interv"] = {"rows": v16, "pooled_2tasks": {"interv": [31, 100], "glove": ["13 또는 14 (13.5)", 100], "p_cons": pm}}

    OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
    # 요약 출력
    print("V1", {k: (v["recomputed"], v["match"]) for k, v in RES["V1_table1_means"].items() if isinstance(v, dict)})
    print("V2", RES["V2_table5"])
    print("V3", RES["V3_half_credit_needed_if_n50"])
    print("V4", RES["V4_hifun_wilson"]["pooled_wilson95_pct"], RES["V4_hifun_wilson"]["all100_wilson_lower_n50"])
    for base in ["interv_fin", "bc", "dp"]:
        print("V5", base, [(r["task"][:8], round(r["p_cons"], 4), round(r["holm_p"], 4)) for r in RES["V5_fisher_table1"][base]])
        print("V5pool", base, RES["V5_fisher_table1"][base + "_pooled"])
    print("V5best", [(r["task"][:8], r["best_baseline"], round(r["p_cons"], 4)) for r in RES["V5_fisher_table1"]["vs_best_per_task"]])
    for n in (20, 50):
        print("V6 n", n, [(r["task"][:5], r["variant"], round(r["p_cons"], 4), round(r["holm_p"], 3)) for r in RES["V6_table2"][f"ablation_n{n}"]])
        print("V6data n", n, RES["V6_table2"][f"data_source_n{n}"])
    print("V6 n needed", RES["V6_table2"]["n_needed_100_vs_85_80pct_power"], RES["V6_table2"]["n_needed_100_vs_75_80pct_power"])
    print("V7", RES["V7_data_time"])
    print("V8", [(r["task"][:6], r["alt"], round(r["p"], 4), round(r["holm_p"], 4)) for r in RES["V8_table3"]["rows"] if r["p"] < 0.2])
    print("V8 tot", {k: RES["V8_table3"][k] for k in ("ours_total", "ours_wilson95_pct", "zero_of_20_wilson_upper_pct", "column_totals_of_120")})
    print("V9", {k: (v["pooled_diff_pp_hifun_data_minus_human"], v["newcombe95_pp"], round(v["pooled_fisher_p"], 3)) for k, v in RES["V9_table6"].items() if isinstance(v, dict)})
    print("V10", RES["V10_skill_hil"])
    print("V11", RES["V11_robustness"])
    print("V13", out)
    print("V14", RES["V14_power"])
    print("V16", RES["V16_glove_vs_interv"])
    print("V15", RES["V15_cost_accounting"])


if __name__ == "__main__":
    main()
