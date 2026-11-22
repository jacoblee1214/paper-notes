"""HumanX (arXiv:2602.02473 v2) 보고 수치 재계산.

Table 1(시뮬레이션), Table 2(실제 로봇), Table 3(다중 패턴)의 평균·배율·신뢰구간·검정을 다시 계산한다.
논문에는 시뮬레이션 평가 횟수 n과 ± 의 정의가 없으므로, 시뮬레이션 열은 평균/배율만 다루고
실제 로봇 열(시도 수가 명시됨: 10회, 50회)만 구간 추정과 검정을 한다.
실행: python3 recompute.py   (표준 라이브러리만 사용, 1초 미만)
"""
import json
import math
from math import comb, erfc, exp, sqrt, pi

# ---------------- Table 1 (v2, p.5) ----------------
methods = ["SkillMimic", "OmniRetarget", "HDMI", "XMimic(Base)", "+IP", "+IVT", "+DataAug", "+TeaStu"]
T1 = {
    "basketball": {
        "SR": [0.0, 0.0, 53.1, 93.4, 86.8, 93.4, 77.6, 86.8],
        "GSR": [0.0, 0.0, 2.4, 4.9, 13.5, 10.9, 60.6, 64.7],
        "Eo": [1.25, 1.18, 0.70, 0.69, 0.70, 0.83, 0.76, 0.85],        # m
        "Eh": [10.3, 7.83, 10.8, 6.36, 7.46, 6.97, 8.34, 7.83],         # cm
    },
    "badminton": {
        "SR": [68.2, 75.8, 90.0, 97.1, 94.0, 100.0, 100.0, 100.0],
        "GSR": [30.9, 20.4, 25.3, 41.6, 60.5, 67.2, 84.4, 90.6],
        "Eo": [2.94, 1.25, 1.64, 1.32, 1.03, 1.33, 0.98, 0.90],        # m
        "Eh": [0.48, 0.15, 0.07, 0.12, 0.10, 0.10, 0.10, 0.09],         # m (헤더 표기)
    },
    "cargo": {
        "SR": [0.4, 0.0, 95.8, 100.0, 100.0, 99.6, None, 99.3],
        "GSR": [0.0, 0.0, 1.8, 50.9, 94.5, 94.3, None, 96.3],
        "Eo": [25.1, 10.8, 1.79, 1.56, 1.62, 1.51, None, 1.38],        # cm
        "Eh": [28.3, 13.3, 13.1, 6.58, 10.5, 9.62, None, 8.67],         # cm
    },
}
res = {}

def idx(m):
    return methods.index(m)

# V1: GSR 평균, HDMI 대비 배율, 작업별 최고 baseline 대비 배율
tasks = list(T1)
gsr_final = [T1[t]["GSR"][idx("+TeaStu")] for t in tasks]
gsr_hdmi = [T1[t]["GSR"][idx("HDMI")] for t in tasks]
gsr_best_base = [max(T1[t]["GSR"][:3]) for t in tasks]
avg = lambda xs: sum(xs) / len(xs)
res["V1"] = {
    "final_GSR_per_task": dict(zip(tasks, gsr_final)),
    "final_GSR_mean": avg(gsr_final),
    "HDMI_GSR_mean": avg(gsr_hdmi),
    "ratio_of_means_vs_HDMI": avg(gsr_final) / avg(gsr_hdmi),
    "best_baseline_GSR_per_task": dict(zip(tasks, gsr_best_base)),
    "best_baseline_GSR_mean": avg(gsr_best_base),
    "ratio_of_means_vs_best_baseline": avg(gsr_final) / avg(gsr_best_base),
    "per_task_ratio_vs_HDMI": {t: T1[t]["GSR"][idx("+TeaStu")] / T1[t]["GSR"][idx("HDMI")] for t in tasks},
    "per_task_ratio_vs_best_baseline": {t: T1[t]["GSR"][idx("+TeaStu")] / max(T1[t]["GSR"][:3]) for t in tasks},
    "mean_of_per_task_ratios_vs_HDMI": avg([T1[t]["GSR"][idx("+TeaStu")] / T1[t]["GSR"][idx("HDMI")] for t in tasks]),
}

# V2: 열별 최고 방법 (낮을수록 좋은 오차, 높을수록 좋은 SR/GSR)
best = {}
for t in tasks:
    for k in ("SR", "GSR", "Eo", "Eh"):
        vals = T1[t][k]
        cand = [(v, methods[i]) for i, v in enumerate(vals) if v is not None]
        better = max if k in ("SR", "GSR") else min
        b = better(cand)[0]
        winners = [m for v, m in cand if v == b]
        final = vals[idx("+TeaStu")]
        hdmi = vals[idx("HDMI")]
        best[f"{t}.{k}"] = {"best": b, "winners": winners, "final": final, "final_is_best": final == b,
                            "HDMI": hdmi}
res["V2_best_per_column"] = best

# V3: 절제 연구 증분 (인접 행)
abl = {}
order = ["XMimic(Base)", "+IP", "+IVT", "+DataAug", "+TeaStu"]
for t in tasks:
    for k in ("SR", "GSR"):
        seq = [(m, T1[t][k][idx(m)]) for m in order if T1[t][k][idx(m)] is not None]
        abl[f"{t}.{k}"] = [(seq[i + 1][0], round(seq[i + 1][1] - seq[i][1], 1)) for i in range(len(seq) - 1)]
res["V3_ablation_deltas"] = abl

# V3b: 인접 행 GSR 차이가 표본 오차만으로 유의해지려면 몇 회 평가가 필요한가 (두 비율 z 검정, 독립 가정, 시드 변동 제외)
def two_prop_p(p1, p2, n):
    p1, p2 = p1 / 100, p2 / 100
    pp = (p1 + p2) / 2
    se = sqrt(2 * pp * (1 - pp) / n) if 0 < pp < 1 else 0.0
    if se == 0:
        return 1.0
    z = abs(p1 - p2) / se
    return erfc(z / sqrt(2))

v3b = {}
for t in tasks:
    seq = [(m, T1[t]["GSR"][idx(m)]) for m in order if T1[t]["GSR"][idx(m)] is not None]
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        v3b[f"{t}: {a[0]} -> {b[0]}"] = {"delta_pts": round(b[1] - a[1], 1),
                                       **{f"p_n{n}": float("%.3g" % two_prop_p(a[1], b[1], n)) for n in (100, 500, 4096)}}
res["V3b_ablation_pvalues_if_n"] = v3b

# V4: 중복 셀
dups = []
seen = {}
for t in tasks:
    for k in ("Eo", "Eh"):
        for i, v in enumerate(T1[t][k]):
            if v is None:
                continue
            seen.setdefault((t, k, v), []).append(methods[i])
for key, ms in seen.items():
    if len(ms) > 1:
        dups.append({"cell": key, "methods": ms})
res["V4_duplicate_cells_same_column"] = dups
res["V4_note"] = "basketball Eh: OmniRetarget 7.83 (±0.95) = +TeaStu 7.83 (±0.95), 평균과 표준편차가 둘 다 같다 (v1 Table I도 동일)"

# ---------------- Table 2 (실제 로봇) ----------------
T2 = {
    "Basketball Catch-Pass (MoCap)": (41, 50), "Cargo Pickup (MoCap)": (43, 50),
    "Football Kicking (MoCap)": (42, 50), "Reactive Fighting (MoCap)": (37, 50),
    "Jumpshot (NEP)": (8, 10), "Dribble (NEP)": (8, 10), "Pump-fake (NEP)": (9, 10),
    "Layup (NEP)": (7, 10), "Basketball Pickup (NEP)": (10, 10),
}
T2_v1_extra = {"Spin Move (NEP)": (9, 10)}  # v1 에만 있고 v2 에서 삭제

def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

def fisher_two_sided(a, b, c, d):
    # 표 [[a,b],[c,d]] (a=성공1, b=실패1, c=성공2, d=실패2)
    n1, n2, s = a + b, c + d, a + c
    tot = n1 + n2
    denom = comb(tot, s)
    def pr(x):
        return comb(n1, x) * comb(n2, s - x) / denom
    p0 = pr(a)
    lo, hi = max(0, s - n2), min(n1, s)
    return sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p0 + 1e-12)

def chi2_sf_df3(x):
    return erfc(sqrt(x / 2)) + sqrt(2 * x / pi) * exp(-x / 2)

t2 = {}
for name, (k, n) in {**T2, **T2_v1_extra}.items():
    lo, hi = wilson(k, n)
    t2[name] = {"k": k, "n": n, "rate": k / n, "wilson95": [round(lo, 3), round(hi, 3)], "in_v2": name in T2}
res["V5_table2_wilson"] = t2

moc = [(k, n) for name, (k, n) in T2.items() if "MoCap" in name]
nep = [(k, n) for name, (k, n) in T2.items() if "NEP" in name]
sm, nm = sum(k for k, _ in moc), sum(n for _, n in moc)
sn, nn = sum(k for k, _ in nep), sum(n for _, n in nep)
res["V6_pooled"] = {
    "MoCap": {"k": sm, "n": nm, "rate": sm / nm, "wilson95": [round(x, 3) for x in wilson(sm, nm)]},
    "NEP_v2": {"k": sn, "n": nn, "rate": sn / nn, "wilson95": [round(x, 3) for x in wilson(sn, nn)]},
    "NEP_v1_with_spin": {"k": sn + 9, "n": nn + 10, "rate": (sn + 9) / (nn + 10), "wilson95": [round(x, 3) for x in wilson(sn + 9, nn + 10)]},
    "all_v2": {"k": sm + sn, "n": nm + nn, "rate": (sm + sn) / (nm + nn), "wilson95": [round(x, 3) for x in wilson(sm + sn, nm + nn)]},
    "mean_of_skill_rates_unweighted_v2": avg([k / n for k, n in moc + nep]),
    "fisher_MoCap_vs_NEP_p": fisher_two_sided(sm, nm - sm, sn, nn - sn),
    "claim_NEP_avg_over_80pct": sn / nn > 0.8,
    "claim_MoCap_over_80pct": sm / nm > 0.8,
}
# MoCap 네 작업 동질성 (카이제곱, df=3)
tot_s, tot_n = sm, nm
p = tot_s / tot_n
chi = sum((k - n * p) ** 2 / (n * p) + ((n - k) - n * (1 - p)) ** 2 / (n * (1 - p)) for k, n in moc)
res["V7_mocap_homogeneity"] = {"chi2": chi, "df": 3, "p": chi2_sf_df3(chi)}
# 최저 작업의 Wilson 하한이 80%를 넘는가
res["V8_lower_bounds_vs_80"] = {n: (v["wilson95"][0] > 0.8) for n, v in t2.items()}

# 쌍 비교: 7/10 vs 10/10 (NEP 내부), 37/50 vs 43/50 (MoCap 내부)
res["V9_pairwise_fisher"] = {
    "Layup 7/10 vs Pickup 10/10": fisher_two_sided(7, 3, 10, 0),
    "Fighting 37/50 vs Cargo 43/50": fisher_two_sided(37, 13, 43, 7),
}

# ---------------- Table 3 vs Table 1 ----------------
T3 = {"football": (74.2, 93.1), "badminton": (52.4, 84.3)}
res["V10_table3_gains_pts"] = {k: round(b - a, 1) for k, (a, b) in T3.items()}
res["V10_single_pattern_gains_pts"] = {
    "badminton +DataAug->+TeaStu": round(90.6 - 84.4, 1),
    "basketball +DataAug->+TeaStu": round(64.7 - 60.6, 1),
}
res["V10_note"] = ("다중 패턴 w/o Tea-Stu(52.4)는 단일 패턴 +Data Aug(84.4)보다 32.0점 낮고, "
                   "w/ Tea-Stu(84.3)는 단일 패턴 +Tea-Stu(90.6)보다 6.3점 낮다. 다중 패턴의 학습 데이터 양(3편×증강 수)은 명시되지 않았다.")

# ---------------- 기타 문서 내부 수 ----------------
res["V11_skill_and_domain_counts"] = {
    "v2_skills_in_Table2": len(T2), "v2_domains_in_Table2": 4,
    "domains_in_Table2": ["basketball(6)", "football(1)", "cargo(1)", "fighting(1)"],
    "abstract_claim": "5 domains, 9 skills transferred zero-shot to physical G1 (v1: 10 skills)",
    "badminton_real_robot_rows": 0,
}
json.dump(res, open("recompute_results.json", "w"), ensure_ascii=False, indent=1, default=str)

# ----- 출력 -----
print("V1  최종 GSR 평균 %.1f%%, HDMI 평균 %.2f%% -> %.2f배; 작업별 최고 baseline 평균 %.2f%% -> %.2f배"
      % (res["V1"]["final_GSR_mean"], res["V1"]["HDMI_GSR_mean"], res["V1"]["ratio_of_means_vs_HDMI"],
         res["V1"]["best_baseline_GSR_mean"], res["V1"]["ratio_of_means_vs_best_baseline"]))
print("   작업별 배율(vs HDMI):", {k: round(v, 1) for k, v in res["V1"]["per_task_ratio_vs_HDMI"].items()},
      " (vs 최고 baseline):", {k: round(v, 1) for k, v in res["V1"]["per_task_ratio_vs_best_baseline"].items()})
for k, v in best.items():
    flag = "최종 모델이 최고" if v["final_is_best"] else "최고=%s(%s)" % (v["winners"], v["best"])
    print("V2 ", k, "final=%s HDMI=%s ->" % (v["final"], v["HDMI"]), flag)
print("V3 ", json.dumps(abl, ensure_ascii=False))
print("V3b", json.dumps(v3b, ensure_ascii=False))
print("V4 ", dups)
for n, v in t2.items():
    print("V5 ", n, "%d/%d=%.0f%%" % (v["k"], v["n"], 100 * v["rate"]), "Wilson95", v["wilson95"])
print("V6 ", json.dumps(res["V6_pooled"], ensure_ascii=False))
print("V7 ", res["V7_mocap_homogeneity"])
print("V8 ", res["V8_lower_bounds_vs_80"])
print("V9 ", res["V9_pairwise_fisher"])
print("V10", res["V10_table3_gains_pts"], res["V10_single_pattern_gains_pts"])
