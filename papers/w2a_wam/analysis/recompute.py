"""W2A (arXiv:2611.02600 v1) 논문 표 수치를 다시 계산하고 통계 검정을 한다.

입력: 표 값은 아래에 옮겨 적었다(출처는 주석, 원문은 paper/paper.txt). 그림에서 읽은 값은 "그림 판독"으로 표시한다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- LIBERO: 스위트당 10과제 x 50회 = 500회(표준 프로토콜). W2A의 표 1 값은 3개 시드 평균이라 스위트당 1500회로 둔다.
  기준선(표 1의 다른 행)은 인용 값이라 스위트당 500회로 가정한다. 시행 수가 다르면 p 값은 달라진다.
- CALVIN ABC->D: 시드당 1000개 연쇄(표준 프로토콜)를 가정한다. W2A는 3시드 평균이라 3000, VPP는 1000으로 둔다.
- 실로봇: 소과제당 30회(논문 본문, 표 16). 시행이 독립이라고 가정한 값이다(같은 장면 배치를 반복하면 독립이 아닐 수 있다).
- 표의 "±" 값(표 15)은 표본 표준편차(n-1)로 가정한다. 시드 3개라 표준편차 추정이 매우 불안정하다.
- 절제 표(표 4-6)는 시드 수가 적혀 있지 않다. 스위트당 500회, 시드 1개로 가정한 값과 W2A 3시드 값을 함께 보인다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "recompute_results.json"
RES = {}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def z_two_prop(p1, n1, p2, n2):
    """두 비율의 z 검정(합동 분산). 정수 성공 수를 모르는 값(평균 비율)에 쓴다."""
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return z, float(2 * stats.norm.sf(abs(z)))


def prop_comp(name, k1, n1, k2, n2):
    d, lo, hi = newcombe(k1, n1, k2, n2)
    r = {"comparison": name, "k1": k1, "n1": n1, "k2": k2, "n2": n2,
         "diff_pp": round(100 * d, 2), "newcombe95_pp": [round(100 * lo, 2), round(100 * hi, 2)],
         "fisher_p": float(f"{fisher(k1, n1, k2, n2):.3g}")}
    print(f"[{name}] {k1}/{n1} vs {k2}/{n2}: diff {100*d:.2f}pp, 95% [{100*lo:.2f}, {100*hi:.2f}], Fisher p={r['fisher_p']}")
    return r


# ---------------------------------------------------------------- 표 값 (논문 원문)
# 표 1 (LIBERO): Spatial, Object, Goal, Long, 표의 Average
T1 = {
    "Diffusion Policy": ([78.3, 92.5, 68.3, 50.5], 72.4),
    "Octo": ([78.9, 85.7, 84.6, 51.1], 75.1),
    "OpenVLA": ([84.7, 88.4, 79.2, 53.7], 76.5),
    "pi0.5": ([98.8, 98.2, 98.0, 92.4], 96.9),
    "mimic-video": ([94.2, 96.8, 90.6, 84.9], 91.6),
    "DreamZero-5B": ([90.4, 94.8, 87.2, 82.2], 88.7),
    "W2A": ([96.3, 98.8, 95.0, 93.6], 95.9),
}
# 표 2 (CALVIN ABC->D): 1/5..5/5, Avg. Len
T2 = {
    "Diffusion Policy": ([40.2, 12.3, 2.6, 0.8, 0.0], 0.56),
    "UniPi": ([56.0, 16.0, 8.0, 8.0, 4.0], 0.92),
    "OpenVLA": ([91.3, 77.8, 62.0, 52.1, 43.5], 3.27),
    "GR-1": ([85.4, 71.2, 59.6, 49.7, 40.1], 3.06),
    "VPP": ([96.5, 90.9, 86.6, 82.0, 76.9], 4.33),
    "W2A": ([95.3, 91.7, 85.5, 80.2, 77.1], 4.30),
}
# 표 3 (실로봇, %): PnP, Cube, Long, Towel, Table, USB
T3 = {
    "Diffusion Policy": [29.2, 10.0, 2.5, 23.3, 6.7, 0.0],
    "OpenVLA": [50.8, 35.0, 18.3, 50.0, 26.7, 20.0],
    "W2A": [78.3, 65.0, 54.2, 76.7, 63.3, 50.0],
}
# 표 16 (소과제별 성공 횟수 / 30): DP, OpenVLA, W2A
SUB = [
    ("PnP", "당근 -> 바구니", 12, 15, 25),
    ("PnP", "바나나 -> 바구니", 6, 15, 28),
    ("PnP", "흰 그릇 -> 접시", 9, 22, 19),
    ("PnP", "파란 컵 -> 접시", 8, 9, 22),
    ("Cube", "초록 -> 빨강", 0, 10, 21),
    ("Cube", "보라 -> 빨강", 4, 10, 15),
    ("Cube", "초록 -> 보라", 8, 6, 27),
    ("Cube", "빨강 -> 파랑", 0, 16, 15),
    ("Long", "서랍 + 당근", 1, 6, 21),
    ("Long", "서랍 + 바나나", 1, 3, 14),
    ("Long", "서랍 + 흰 그릇", 1, 0, 10),
    ("Long", "당근과 바나나 -> 바구니", 0, 13, 20),
    ("Practical", "Towel folding", 7, 15, 23),
    ("Practical", "Table clearing", 2, 8, 19),
    ("Practical", "USB insertion", 0, 6, 15),
]
T16_OVERALL = {"Diffusion Policy": 13.1, "OpenVLA": 34.2, "W2A": 65.3}
# 표 4, 5, 6 (LIBERO 절제): 스위트별 성공률(Spatial, Object, Goal, Long)과 표의 Avg., 괄호 안 하락 폭
ABL = {
    "w/o distill": ([85.9, 87.7, 83.6, 80.5], 84.4, [10.4, 11.1, 11.4, 13.1], 11.5),
    "w/o PCA": ([87.0, 92.3, 90.2, 84.8], 88.6, [9.3, 6.5, 4.8, 8.8], 7.3),
    "Shuffled Teacher": ([80.5, 81.6, 77.7, 73.1], 78.2, [15.8, 17.2, 17.3, 20.5], 17.7),
    "Random Teacher": ([72.5, 79.8, 65.2, 59.3], 69.2, [23.8, 19.0, 29.8, 34.3], 26.7),
    "Token0 Teacher": ([88.9, 91.0, 84.4, 82.3], 86.7, [7.4, 7.8, 10.6, 11.3], 9.2),
    "Current Feature": ([83.3, 86.6, 81.7, 77.1], 82.2, [13.0, 12.2, 13.3, 16.5], 13.7),
    "Future Feature": ([89.8, 92.5, 86.4, 82.7], 87.8, [6.5, 6.3, 8.6, 10.9], 8.1),
    "Global MLP": ([83.7, 85.1, 82.2, 79.8], 82.7, [12.6, 13.7, 12.8, 13.8], 13.2),
    "Grouped MLP": ([89.5, 90.4, 87.2, 83.8], 87.7, [6.8, 8.4, 7.8, 9.8], 8.2),
}
# 표 15 (시드별): 시드 0, 5, 42 / Spatial, Object, Goal, Long, CALVIN Avg.Len
T15 = {
    "Spatial": ([95.4, 98.6, 95.0], 96.3, 1.97),
    "Object": ([99.0, 98.2, 99.2], 98.8, 0.53),
    "Goal": ([91.6, 97.4, 96.0], 95.0, 3.03),
    "Long": ([91.0, 94.2, 95.6], 93.6, 2.36),
    "CALVIN Avg.Len": ([4.34, 4.26, 4.30], 4.30, 0.04),
}
# 표 14 (선형 프로브 위치 오차, px): t, t+5, t+10
T14 = {"w/o distill": [6.4, 13.7, 23.8], "W2A": [5.9, 8.2, 13.1]}
# 그림 6 지연 (ms)
LAT = {"W2A": 55, "pi0": 198, "pi0.5": 200, "OpenVLA": 290, "mimic-video": 1162}
# 그림 9 (K 훑기)와 그림 10 (깊이): 그림 판독 값(축 눈금 기준, +-0.3 정도)
FIG9 = {32: 64.3, 64: 71.1, 94: 93.7, 482: 95.8, 2048: 88.6}
FIG10 = {"10%": 92.5, "30%": 93.9, "50%": 95.9, "70%": 90.1, "90%": 87.9}


def check_means():
    bad = []
    out = {}
    for k, (vals, avg) in T1.items():
        calc = sum(vals) / 4
        out[k] = round(calc, 3)
        if abs(round(calc, 1) - avg) > 0.051:
            bad.append(("T1", k, calc, avg))
    for k, (vals, avg) in T2.items():
        calc = sum(vals) / 100
        out["calvin_" + k] = round(calc, 3)
        if abs(round(calc, 2) - avg) > 0.0051:
            bad.append(("T2", k, calc, avg))
        mono = all(vals[i] >= vals[i + 1] for i in range(4))
        if not mono:
            bad.append(("T2 단조", k, vals, None))
    print("V1/V2 표 1·2 평균:", "불일치 없음" if not bad else bad)
    return {"values": out, "mismatch": bad}


def check_real():
    res = {}
    tot = {"Diffusion Policy": 0, "OpenVLA": 0, "W2A": 0}
    types = {}
    for typ, name, dp, ov, w in SUB:
        for m, k in zip(["Diffusion Policy", "OpenVLA", "W2A"], [dp, ov, w]):
            tot[m] += k
            types.setdefault(typ, {}).setdefault(m, 0)
            types[typ][m] += k
    res["totals_450"] = tot
    res["totals_pct"] = {m: round(100 * k / 450, 2) for m, k in tot.items()}
    res["by_type"] = {}
    n_by = {"PnP": 120, "Cube": 120, "Long": 120, "Practical": 90}
    for typ, d in types.items():
        res["by_type"][typ] = {m: round(100 * k / n_by[typ], 2) for m, k in d.items()}
    print("V3 표 3·16 유형별 합산 %:", res["by_type"])
    print("V4 표 16 Overall:", res["totals_pct"], "표:", T16_OVERALL)
    # 표 3의 소수점 값이 표 16 횟수와 일치하는지
    ok = True
    for m in T3:
        calc = [res["by_type"][t][m] for t in ["PnP", "Cube", "Long"]] + [100 * k / 30 for k in
              [r[{"Diffusion Policy": 2, "OpenVLA": 3, "W2A": 4}[m] + 0] for r in SUB[12:]]]
        for a, b in zip(calc, T3[m]):
            if abs(a - b) > 0.06:
                ok = False
                print("  표 3 불일치", m, a, b)
    res["table3_matches_table16"] = ok
    print("   표 3 = 표 16 합산:", ok)
    # 유형별 W2A 대 OpenVLA, W2A 대 DP
    res["w2a_vs_openvla_by_type"] = []
    for typ in ["PnP", "Cube", "Long", "Practical"]:
        n = n_by[typ]
        res["w2a_vs_openvla_by_type"].append(prop_comp(f"{typ} W2A 대 OpenVLA", types[typ]["W2A"], n, types[typ]["OpenVLA"], n))
    res["w2a_vs_openvla_all"] = prop_comp("전체 W2A 대 OpenVLA", tot["W2A"], 450, tot["OpenVLA"], 450)
    res["w2a_vs_dp_all"] = prop_comp("전체 W2A 대 DP", tot["W2A"], 450, tot["Diffusion Policy"], 450)
    # 소과제별
    res["subtask"] = []
    wins = losses = 0
    pvals = []
    for typ, name, dp, ov, w in SUB:
        p = fisher(w, 30, ov, 30)
        pvals.append(p)
        wins += w > ov
        losses += w < ov
        res["subtask"].append({"type": typ, "task": name, "dp": dp, "openvla": ov, "w2a": w, "fisher_p_vs_openvla": float(f"{p:.3g}")})
    m = len(pvals)
    order = np.argsort(pvals)
    holm = [None] * m
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[idx]))
        holm[idx] = run
    for r, h in zip(res["subtask"], holm):
        r["holm_p"] = float(f"{h:.3g}")
    sig_raw = sum(p < 0.05 for p in pvals)
    sig_holm = sum(h < 0.05 for h in holm)
    res["subtask_summary"] = {"w2a_wins": int(wins), "w2a_losses": int(losses), "raw_p_lt_0.05": int(sig_raw),
                              "holm_p_lt_0.05": int(sig_holm)}
    kmin = min(wins, losses)
    res["sign_test_p_vs_openvla"] = float(f"{min(1.0, 2 * sum(math.comb(15, i) for i in range(kmin + 1)) / 2 ** 15):.3g}")
    print("   소과제 W2A 대 OpenVLA 승/패", wins, losses, "보정 전 p<0.05:", sig_raw, "Holm 후:", sig_holm,
          "부호 검정 p:", res["sign_test_p_vs_openvla"])
    # 무작위 평균 5.5점 차이 등 요약
    ci = {m_: [round(100 * x, 1) for x in wilson(k, 450)] for m_, k in tot.items()}
    res["wilson95_overall"] = ci
    print("   전체 Wilson 95%:", ci)
    return res


def check_libero():
    res = {"w2a_vs_baselines": []}
    suites = ["Spatial", "Object", "Goal", "Long"]
    w = T1["W2A"][0]
    for base in ["pi0.5", "mimic-video", "DreamZero-5B"]:
        b = T1[base][0]
        row = {"baseline": base, "suites": {}}
        for s, wv, bv in zip(suites, w, b):
            z, p = z_two_prop(wv / 100, 1500, bv / 100, 500)
            row["suites"][s] = {"diff_pp": round(wv - bv, 1), "z": round(z, 2), "p_z": float(f"{p:.3g}")}
        wa, ba = sum(w) / 4, sum(b) / 4
        z, p = z_two_prop(wa / 100, 6000, ba / 100, 2000)
        row["average"] = {"diff_pp": round(wa - ba, 2), "z": round(z, 2), "p_z": float(f"{p:.3g}")}
        res["w2a_vs_baselines"].append(row)
        print(f"V5 W2A 대 {base}: " + ", ".join(f"{s} {row['suites'][s]['diff_pp']:+.1f} (p={row['suites'][s]['p_z']})" for s in suites)
              + f" | 평균 {row['average']['diff_pp']:+.2f} (p={row['average']['p_z']})")
    # 정수 복원 가능성 (스위트당 500회)
    feas = {}
    for k, (vals, _) in T1.items():
        feas[k] = [abs(v * 5 - round(v * 5)) < 0.02 for v in vals]
    res["integer_feasible_N500"] = feas
    print("   N=500에서 정수 성공 수가 되는 값(스위트별):", {k: v for k, v in feas.items()})
    # W2A 시드별
    seed_avg = [np.mean([T15[s][0][i] for s in suites]) for i in range(3)]
    res["seed_avg_libero"] = [round(float(x), 3) for x in seed_avg]
    res["seed_avg_sd"] = round(float(np.std(seed_avg, ddof=1)), 3)
    res["seed_avg_mean"] = round(float(np.mean(seed_avg)), 3)
    feas15 = {s: [abs(v * 5 - round(v * 5)) < 0.02 for v in T15[s][0]] for s in suites}
    res["seed_values_integer_N500"] = feas15
    print("V6 표 15 시드별 LIBERO 평균:", res["seed_avg_libero"], "평균", res["seed_avg_mean"], "SD", res["seed_avg_sd"])
    return res


def check_table15():
    res = {}
    for k, (vals, mean, sd) in T15.items():
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1))
        res[k] = {"mean_calc": round(m, 3), "sd_calc_ddof1": round(s, 3), "sd_calc_ddof0": round(float(np.std(vals)), 3),
                  "table_mean": mean, "table_sd": sd,
                  "match": abs(round(m, 2 if k == "CALVIN Avg.Len" else 1) - mean) < 0.011 and abs(round(s, 2) - sd) < 0.011}
    print("V6 표 15 평균·SD:", {k: v["match"] for k, v in res.items()})
    return res


def check_calvin():
    res = {}
    w, v = T2["W2A"][0], T2["VPP"][0]
    for i, label in [(0, "1/5"), (1, "2/5"), (2, "3/5"), (3, "4/5"), (4, "5/5")]:
        z, p = z_two_prop(w[i] / 100, 3000, v[i] / 100, 1000)
        res[label] = {"w2a": w[i], "vpp": v[i], "diff_pp": round(w[i] - v[i], 1), "z": round(z, 2), "p": float(f"{p:.3g}")}
    seeds = T15["CALVIN Avg.Len"][0]
    sd = float(np.std(seeds, ddof=1))
    t = (np.mean(seeds) - 4.33) / (sd / math.sqrt(3))
    p = float(2 * stats.t.sf(abs(t), 2))
    res["avg_len_vs_VPP"] = {"w2a_mean": round(float(np.mean(seeds)), 3), "seed_sd": round(sd, 3), "vpp": 4.33,
                             "t": round(float(t), 2), "p": float(f"{p:.3g}")}
    print("V7 CALVIN W2A 대 VPP:", {k: (x["diff_pp"], x["p"]) for k, x in res.items() if k != "avg_len_vs_VPP"}, res["avg_len_vs_VPP"])
    return res


def check_ablations():
    res = {}
    w = T1["W2A"][0]
    wavg = sum(w) / 4
    for name, (vals, avg, drops, davg) in ABL.items():
        calc = sum(vals) / 4
        recon = [round(a + b, 1) for a, b in zip(vals, drops)]
        ok_drop = all(abs(r - x) < 0.051 for r, x in zip(recon, w))
        ok_avg = abs(round(calc, 1) - avg) < 0.051 or abs(calc - avg) < 0.06
        ok_davg = abs(round(wavg - avg, 1) - davg) < 0.051
        z, p = z_two_prop(wavg / 100, 6000, avg / 100, 2000)
        res[name] = {"avg_calc": round(calc, 3), "avg_table": avg, "drops_consistent_with_w2a_row": ok_drop,
                     "avg_ok": ok_avg, "avg_drop_ok": ok_davg, "drop_pp": round(wavg - avg, 2),
                     "z_N6000_vs_N2000": round(z, 1), "p_z": float(f"{p:.2g}")}
    print("V8 절제 표 4-6 검산:", {k: (v["drops_consistent_with_w2a_row"], v["avg_ok"], v["avg_drop_ok"]) for k, v in res.items()})
    # 시드 변동으로 본 검출 한계: 4개 스위트 평균의 시드 SD
    sd_avg = np.std([np.mean([T15[s][0][i] for s in ["Spatial", "Object", "Goal", "Long"]]) for i in range(3)], ddof=1)
    res["detect_limit_note"] = {"seed_sd_of_suite_average": round(float(sd_avg), 2),
                                "two_single_seed_runs_diff_sd": round(float(sd_avg * math.sqrt(2)), 2),
                                "approx_2sd_diff": round(float(2 * sd_avg * math.sqrt(2)), 2)}
    print("   시드 SD(스위트 평균) =", round(float(sd_avg), 2), "-> 단일 시드 두 개의 차이 SD =", round(float(sd_avg * math.sqrt(2)), 2))
    # 절제 간 차이 (단일 시드 가정)
    pairs = {
        "Future Feature 대 w/o distill": (87.8, 84.4),
        "Token0 Teacher 대 w/o distill": (86.7, 84.4),
        "Current Feature 대 w/o distill": (82.2, 84.4),
        "Global MLP 대 w/o distill": (82.7, 84.4),
        "Grouped MLP 대 w/o distill": (87.7, 84.4),
        "Grouped MLP 대 Global MLP": (87.7, 82.7),
        "w/o PCA(K=2048) 대 K=94(그림 판독)": (88.6, 93.7),
        "K=482 대 K=94(그림 판독)": (95.8, 93.7),
        "깊이 50% 대 30%(그림 판독)": (95.9, 93.9),
    }
    res["pairs"] = {}
    for k, (a, b) in pairs.items():
        z, p = z_two_prop(a / 100, 2000, b / 100, 2000)
        res["pairs"][k] = {"diff_pp": round(a - b, 1), "z_N2000each": round(z, 2), "p_z": float(f"{p:.2g}"),
                           "diff_over_seed_diff_sd": round((a - b) / float(sd_avg * math.sqrt(2)), 2)}
    print("   절제 간 차이:", {k: (v["diff_pp"], v["p_z"], v["diff_over_seed_diff_sd"]) for k, v in res["pairs"].items()})
    return res


def check_misc():
    res = {}
    res["latency_ratio"] = {k: round(v / LAT["W2A"], 2) for k, v in LAT.items()}
    print("V9 지연 배율(W2A=1):", res["latency_ratio"])
    res["probe_reduction_pct"] = [round(100 * (a - b) / a, 1) for a, b in zip(T14["w/o distill"], T14["W2A"])]
    res["probe_diff_px"] = [round(a - b, 1) for a, b in zip(T14["w/o distill"], T14["W2A"])]
    print("V10 표 14 상대 감소 %:", res["probe_reduction_pct"], "차이 px:", res["probe_diff_px"])
    # 구성 수치 산술
    shape = {"teacher_T": 24, "G_groups": 4, "H": 16, "r_t": 4, "Hv": 14, "Wv": 14, "Dv": 2048, "K": 482}
    shape["M_aligned_positions"] = shape["G_groups"] * shape["Hv"] * shape["Wv"]
    shape["tokens_per_group"] = shape["H"] // shape["r_t"]
    shape["pca_ratio_pct"] = round(100 * shape["K"] / shape["Dv"], 1)
    # 표 8의 24는 93프레임/4+1 = 24, 본문 그림 7의 5는 17프레임/4+1 = 5
    shape["latent_T_for_93_frames"] = (93 - 1) // 4 + 1
    shape["latent_T_for_17_frames"] = (17 - 1) // 4 + 1
    shape["teacher_numel_per_sample_MB_fp16"] = round(24 * 14 * 14 * 2048 * 2 / 1e6, 1)
    shape["selected_teacher_numel_after_pca"] = shape["G_groups"] * 14 * 14 * 482
    res["shape_arithmetic"] = shape
    print("V11 형상 산술:", shape)
    # PCA 설명 분산: 65/94/482 PC (그림 8)
    res["pca_thresholds"] = {"90%": 65, "95%": 94, "99%": 482, "ratio_482_over_94": round(482 / 94, 2)}
    res["fig9_fig10_read"] = {"K": FIG9, "depth": FIG10}
    # 행동 DiT 파라미터 어림 (표 7: 768, 10블록, 8헤드, MLP 4.0, 교차 주의, AdaLN)
    d = 768
    blk = 4 * d * d + 4 * d * d + 8 * d * d + 6 * d * d  # 자기주의 + 교차주의(차원 같다고 가정) + MLP + AdaLN
    res["action_dit_param_estimate_M"] = round(10 * blk / 1e6, 1)
    res["resnet18_x2_M"] = round(2 * 11.2, 1)
    res["total_estimate_M"] = round(10 * blk / 1e6 + 2 * 11.2, 1)
    print("V12 W2A 학생 파라미터 어림(M, 구조 가정):", res["total_estimate_M"])
    # 1 NFE FLOPs 어림: 2 * params * tokens
    tokens = 16
    nfe = 20
    res["student_flops_estimate_GFLOP"] = round(2 * 10 * blk * tokens * nfe / 1e9, 1)
    print("   20스텝 토큰 16개 GFLOP 어림:", res["student_flops_estimate_GFLOP"])
    # 유효 제어 빈도: 55 ms/예측, 8 동작 실행, 실로봇 15 Hz
    res["control_arithmetic"] = {"actions_per_prediction": 8, "latency_ms": 55,
                                 "actions_per_second_if_back_to_back": round(8 / 0.055, 1), "real_robot_hz": 15,
                                 "prediction_period_at_15hz_ms": round(1000 / 15 * 8, 1)}
    # 교사 쪽 (Cosmos-Predict2 2B, 28블록, 14번째 = 50%)
    res["teacher_depth"] = {"block": 14, "of": 28, "pct": 50.0, "student_block": 5, "student_of": 10, "student_pct": 50.0}
    return res


def main():
    RES["V1_V2_means"] = check_means()
    RES["real_world"] = check_real()
    RES["libero"] = check_libero()
    RES["table15"] = check_table15()
    RES["calvin"] = check_calvin()
    RES["ablations"] = check_ablations()
    RES["misc"] = check_misc()
    OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
