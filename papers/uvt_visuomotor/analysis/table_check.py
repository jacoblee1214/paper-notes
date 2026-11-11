"""UVT 논문 표 재계산과 통계 검정 (CPU, 수 초). 실행: python3 analysis/table_check.py
결과: analysis/table_check_results.json

- Table I·II의 평균 열을 다시 계산하고 논문 값과 맞춘다.
- LIBERO: 논문은 suite별 시행 횟수와 시드 수를 쓰지 않았다. 그래서 공식 코드의 기본값(과제당 50회 × 10과제 = suite당 500회)을
  가정하고, README 예시(과제당 20회 = suite당 200회)를 민감도로 함께 본다. 시드 분산은 반영되지 않는다.
- 실물: Lift Pot(전체 성공만), Close Marker와 Plate Handover(전체 성공과 절반 성공, 과제당 50회).
  이진 결과는 Fisher·Wilson·CMH·Holm, 절반 성공을 포함한 점수는 과제별 시행 점수(1, 0.5, 0)의 Mann–Whitney로 검정한다.
- 검출력: 과제당 50회 두 팔의 최소 검출 차이(80% 검출력, 양측 5%).
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "table_check_results.json"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def cmh(tables):
    """tables: 층별 [[a, b], [c, d]] (행 = 방법 A/B, 열 = 성공/실패). Mantel–Haenszel 공통 오즈비와 연속성 보정 카이제곱."""
    num = den = 0.0
    e_sum = v_sum = 0.0
    for (a, b), (c, d) in tables:
        n = a + b + c + d
        r1, r2 = a + b, c + d
        c1, c2 = a + c, b + d
        if n == 0:
            continue
        num += a * d / n
        den += b * c / n
        e_sum += r1 * c1 / n
        v_sum += r1 * r2 * c1 * c2 / (n * n * (n - 1))
    orr = num / den if den > 0 else float("inf")
    a_obs = sum(t[0][0] for t in tables)
    chi2 = (abs(a_obs - e_sum) - 0.5) ** 2 / v_sum
    return orr, chi2, 1 - stats.chi2.cdf(chi2, 1)


def fisher(k1, n1, k2, n2):
    return stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]


def mde_two_prop(p, n, z_a=1.959964, z_b=0.841621):
    """두 비율, 각 팔 n회, 기준 성공률 p에서의 최소 검출 차이(근사식, 양측 5%, 검출력 80%)."""
    return (z_a + z_b) * math.sqrt(2 * p * (1 - p) / n)


R = {}

# ---- 1. 표 산술 ----
T1 = {  # Table I: suite별 (10k, 100k)
    "VLA-Adapter LIBERO": {"Object": (93.4, 97.6), "Spatial": (78.8, 96.2), "Goal": (90.4, 95.6), "Long": (56.2, 89.0), "Avg": (79.7, 94.6)},
    "UVT LIBERO": {"Object": (94.0, 99.6), "Spatial": (95.0, 98.6), "Goal": (95.0, 97.2), "Long": (68.8, 91.2), "Avg": (88.2, 96.7)},
    "VLA-Adapter LIBERO-Plus": {"Object": (52.7, 58.3), "Spatial": (34.0, 86.8), "Goal": (42.3, 72.4), "Long": (41.7, 64.3), "Avg": (42.7, 70.5)},
    "UVT LIBERO-Plus": {"Object": (52.4, 60.4), "Spatial": (81.5, 87.2), "Goal": (67.8, 74.7), "Long": (63.0, 72.1), "Avg": (66.2, 73.6)},
}
T2 = {  # Table II: suite별 최종 (100k)
    "VLA-Adapter": {"Object": 97.6, "Spatial": 96.2, "Goal": 95.6, "Long": 89.0, "Avg": 94.6},
    "pi0.5": {"Object": 97.8, "Spatial": 98.4, "Goal": 98.0, "Long": 90.6, "Avg": 96.2},
    "UVT (VLA-Adapter)": {"Object": 99.6, "Spatial": 98.6, "Goal": 97.2, "Long": 91.2, "Avg": 96.7},
    "UVT (pi0.5)": {"Object": 99.0, "Spatial": 98.4, "Goal": 97.8, "Long": 92.4, "Avg": 96.9},
}
arith = {}
for name, d in T1.items():
    suites = ["Object", "Spatial", "Goal", "Long"]
    for idx, step in enumerate(["10k", "100k"]):
        mean = np.mean([d[s][idx] for s in suites])
        arith[f"{name} {step}"] = {"recomputed": round(float(mean), 3), "reported": d["Avg"][idx],
                                   "match": abs(mean - d["Avg"][idx]) < 0.06}
for name, d in T2.items():
    suites = ["Object", "Spatial", "Goal", "Long"]
    mean = np.mean([d[s] for s in suites])
    arith[f"Table II {name}"] = {"recomputed": round(float(mean), 3), "reported": d["Avg"],
                                 "match": abs(mean - d["Avg"]) < 0.06}
R["table_arithmetic"] = arith

# ---- 2. LIBERO 검정 (가정 n) ----
def lib_tests(n):
    out = {}
    pvals10, rows = [], []
    for step, idx in (("10k", 0), ("100k", 1)):
        for s in ["Object", "Spatial", "Goal", "Long"]:
            a = T1["VLA-Adapter LIBERO"][s][idx]
            b = T1["UVT LIBERO"][s][idx]
            ka, kb = round(a / 100 * n), round(b / 100 * n)
            p = fisher(ka, n, kb, n)
            rows.append({"step": step, "suite": s, "base": a, "uvt": b, "diff_pp": round(b - a, 1),
                         "k_base": ka, "k_uvt": kb, "fisher_p": p,
                         "wilson_base": [round(v * 100, 1) for v in wilson(ka, n)],
                         "wilson_uvt": [round(v * 100, 1) for v in wilson(kb, n)]})
            pvals10.append(p)
    adj = holm(pvals10)
    for r, q in zip(rows, adj):
        r["holm_p"] = q
    out["suites"] = rows
    # 네 suite 평균의 차이: 이항 분산을 suite 평균으로 모은 근사 (suite 간 독립 가정, 시드 분산 미포함)
    for step, idx in (("10k", 0), ("100k", 1)):
        diffs, var = [], 0.0
        for s in ["Object", "Spatial", "Goal", "Long"]:
            a = T1["VLA-Adapter LIBERO"][s][idx] / 100
            b = T1["UVT LIBERO"][s][idx] / 100
            var += (a * (1 - a) + b * (1 - b)) / n
            diffs.append(b - a)
        d = float(np.mean(diffs))
        se = math.sqrt(var) / 4
        out[f"avg_diff_{step}_pp"] = round(d * 100, 2)
        out[f"avg_diff_{step}_se_pp"] = round(se * 100, 2)
        out[f"avg_diff_{step}_p_normal"] = float(2 * (1 - stats.norm.cdf(abs(d / se))))
    return out


R["libero_n500_assumed"] = lib_tests(500)
R["libero_n200_assumed"] = lib_tests(200)

# pi0.5 대 UVT(VLA-Adapter) 최종 평균 (100k) 차이, 가정 n=500/suite
d_pi = (96.9 - 96.2) / 100
suites_pi = [(97.8, 99.0), (98.4, 98.4), (98.0, 97.8), (90.6, 92.4)]  # (pi0.5, UVT) suite별 100k 최종
var_pi = sum((a / 100 * (1 - a / 100) + b / 100 * (1 - b / 100)) / 500 for a, b in suites_pi)
se_pi = math.sqrt(var_pi) / 4
R["pi05_vs_uvt_final_avg"] = {"diff_pp": round(d_pi * 100, 2), "se_pp_n500": round(se_pi * 100, 2),
                               "p_normal_n500": float(2 * (1 - stats.norm.cdf(abs(d_pi / se_pi))))}
R["libero_mde_per_suite_pp_n500"] = {"p=0.95": round(mde_two_prop(0.95, 500) * 100, 2),
                                     "p=0.90": round(mde_two_prop(0.90, 500) * 100, 2)}

# ---- 3. 실물 (과제당 50회) ----
REAL = {
    "Lift Pot": {"base_full": 19, "uvt_full": 27, "base_half": 0, "uvt_half": 0, "scored": False},
    "Close Marker": {"base_full": 0, "uvt_full": 5, "base_half": 0, "uvt_half": 28, "scored": True},
    "Plate Handover": {"base_full": 7, "uvt_full": 12, "base_half": 4, "uvt_half": 16, "scored": True},
}
N = 50
real = {}
pv_full = {}
tables = []
for task, v in REAL.items():
    kb, ku = v["base_full"], v["uvt_full"]
    p = fisher(kb, N, ku, N)
    pv_full[task] = p
    tables.append(((kb, N - kb), (ku, N - ku)))
    entry = {
        "base_full": kb, "uvt_full": ku,
        "base_full_wilson": [round(x * 100, 1) for x in wilson(kb, N)],
        "uvt_full_wilson": [round(x * 100, 1) for x in wilson(ku, N)],
        "diff_full_pp": round((ku - kb) * 2, 1),
        "fisher_p_full": p,
        "base_score": (kb + 0.5 * v["base_half"]) / N,
        "uvt_score": (ku + 0.5 * v["uvt_half"]) / N,
    }
    if v["scored"]:
        # 과제별 시행 점수 분포로 Mann–Whitney (동점 보정, 점근)
        xb = np.array([1.0] * kb + [0.5] * v["base_half"] + [0.0] * (N - kb - v["base_half"]))
        xu = np.array([1.0] * ku + [0.5] * v["uvt_half"] + [0.0] * (N - ku - v["uvt_half"]))
        mw = stats.mannwhitneyu(xu, xb, alternative="two-sided", method="asymptotic")
        entry["mannwhitney_p_score"] = float(mw.pvalue)
    real[task] = entry
adj_full = holm([pv_full[t] for t in REAL])
for t, q in zip(REAL, adj_full):
    real[t]["holm_p_full"] = q
R["real_tasks"] = real
orr, chi2, p_cmh = cmh(tables)
R["real_cmh_full"] = {"common_or": round(orr, 3), "chi2_cc": round(chi2, 3), "p": p_cmh}
pool_b = sum(v["base_full"] for v in REAL.values())
pool_u = sum(v["uvt_full"] for v in REAL.values())
R["real_pooled_full"] = {"base": f"{pool_b}/150", "uvt": f"{pool_u}/150",
                         "fisher_p_pooled_naive": fisher(pool_b, 150, pool_u, 150),
                         "note": "과제가 다르므로 풀링은 참고용. 공식 검정은 CMH를 쓴다."}
R["real_mde_n50_pp"] = {"p=0.2": round(mde_two_prop(0.2, 50) * 100, 1),
                        "p=0.4": round(mde_two_prop(0.4, 50) * 100, 1),
                        "p=0.5": round(mde_two_prop(0.5, 50) * 100, 1)}

# 논문 본문 값과 대조
R["paper_text_checks"] = {
    "Lift Pot 19/50=38%, 27/50=54%": [round(19 / 50 * 100), round(27 / 50 * 100)],
    "Close Marker 5 full + 0.5*28 half = 38%": round((5 + 0.5 * 28) / 50 * 100, 1),
    "Plate baseline 7 + 0.5*4 = 18%": round((7 + 0.5 * 4) / 50 * 100, 1),
    "Plate UVT 12 + 0.5*16 = 40%": round((12 + 0.5 * 16) / 50 * 100, 1),
}

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2, default=float))

if __name__ == "__main__":
    print("산술 일치:", {k: v["match"] for k, v in arith.items()})
    for key in ("libero_n500_assumed",):
        for r in R[key]["suites"]:
            print(r["step"], r["suite"], r["base"], "->", r["uvt"], f"p={r['fisher_p']:.3g}", f"holm={r['holm_p']:.3g}")
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in R[key].items() if k != "suites"})
    print("n=200 avg 10k:", {k: v for k, v in R["libero_n200_assumed"].items() if k.startswith("avg_diff_10k")})
    print("pi0.5 vs UVT:", R["pi05_vs_uvt_final_avg"])
    for t, v in real.items():
        print(t, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items() if "wilson" not in k})
    print("CMH:", R["real_cmh_full"])
    print("pooled:", R["real_pooled_full"])
    print("MDE:", R["real_mde_n50_pp"], R["libero_mde_per_suite_pp_n500"])
