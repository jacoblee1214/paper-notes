"""APM (arXiv:2611.22592 v1) 논문 표 수치를 다시 계산하고 통계 검정을 한다.

입력: 논문 Table 1–3과 본문 수치를 아래에 그대로 옮겼다(출처 주석). 외부 데이터는 쓰지 않는다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 로컬 평가의 분모는 조건당 2,500 에피소드(50 과제 × 50 시드, 본문 4.1)다.
  퍼센트는 0.1 포인트 단위로 반올림되어 있어 성공 수 k는 정수가 아닐 수 있다. 따라서 k = p·n 을 그대로 쓴다.
- 조건(Clean, Random)은 같은 시드를 쓰지만 에피소드별 결과가 없어 쌍체 검정을 할 수 없다.
  모든 비교는 독립 두 비율 검정으로 한다. 쌍체 설계는 분산을 줄이므로 이 검정은 보수적이다.
  다만 과제 간 이질성(클러스터)은 반영되지 않는다. 과제 단위 데이터가 없어서다. 따라서 실제 불확실성은 여기보다 클 수 있다.
- Avg.의 표준오차는 두 조건의 평균으로 보고 0.5·sqrt(se_C² + se_R²)로 근사한다(독립 가정).
- 실로봇 분모는 Table 3 캡션의 과제당 24 시도다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}
Z = 1.959964


def wilson(p, n, z=Z):
    """Wilson 95% 구간. p는 비율(0–1), n은 분모."""
    k = p * n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def two_prop_z(p1, n1, p2, n2):
    """독립 두 비율의 차(p1 - p2), 표준오차, 양측 p값(정규 근사)."""
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p1 - p2
    zval = d / se
    return d, se, 2 * (1 - stats.norm.cdf(abs(zval)))


def ci_diff(d, se):
    return d - Z * se, d + Z * se


# ---------------------------------------------------------------- Table 1
# 출처: 논문 Table 1 (로컬 평가 2,500 에피소드/조건). 값은 %.
T1 = {
    "1/1": {
        "LingBot-VA": (57.0, 50.7, 53.9),
        "Flash-WAM(DirectWAM 보고 1/1)": (82.1, 74.2, 78.2),
        "Continued": (82.3, 76.1, 79.2),
        "APM": (84.3, 77.5, 80.9),
        "delta_reported": 1.7,
    },
    "2/2": {
        "LingBot-VA": (88.3, 86.0, 87.2),
        "Flash-WAM": (90.5, 88.6, 89.6),
        "Continued": (89.6, 87.7, 88.6),
        "APM": (90.4, 87.1, 88.8),
        "delta_reported": 0.1,
    },
    "4/4": {
        "LingBot-VA": (90.4, 90.0, 90.2),
        "Flash-WAM": (92.0, 90.4, 91.2),
        "Continued": (90.5, 89.1, 89.8),
        "APM": (92.5, 90.3, 91.4),
        "delta_reported": 1.6,
    },
}
N_LOCAL = 2500
T1_RES = {}
for budget, rows in T1.items():
    out = {"checks": {}, "cells": {}, "tests": {}}
    for name, vals in rows.items():
        if name == "delta_reported":
            continue
        c, r, a = vals
        avg_calc = (c + r) / 2
        out["checks"][name] = {"avg_reported": a, "avg_recomputed": round(avg_calc, 3),
                               "ok": abs(avg_calc - a) <= 0.05 + 1e-9}
        out["cells"][name] = {
            "clean_ci95": [round(100 * x, 2) for x in wilson(c / 100, N_LOCAL)],
            "random_ci95": [round(100 * x, 2) for x in wilson(r / 100, N_LOCAL)],
        }
    # APM 대 Continued: 조건별 독립 두 비율 검정, 평균은 근사
    ca, ra = rows["APM"][:2]
    cc, rc = rows["Continued"][:2]
    for label, (p1, p2) in {"clean": (ca, cc), "random": (ra, rc)}.items():
        d, se, p = two_prop_z(p1 / 100, N_LOCAL, p2 / 100, N_LOCAL)
        lo, hi = ci_diff(d, se)
        out["tests"][f"APM_minus_Continued_{label}"] = {
            "diff_pp": round(100 * d, 2), "ci95_pp": [round(100 * lo, 2), round(100 * hi, 2)],
            "p_two_sided": round(p, 4)}
    se_c = two_prop_z(ca / 100, N_LOCAL, cc / 100, N_LOCAL)[1]
    se_r = two_prop_z(ra / 100, N_LOCAL, rc / 100, N_LOCAL)[1]
    avg_d = (rows["APM"][2] - rows["Continued"][2]) / 100
    se_avg = 0.5 * math.sqrt(se_c ** 2 + se_r ** 2)
    lo, hi = ci_diff(avg_d, se_avg)
    p_avg = 2 * (1 - stats.norm.cdf(abs(avg_d / se_avg)))
    delta_calc = rows["APM"][2] - rows["Continued"][2]
    out["tests"]["APM_minus_Continued_avg"] = {
        "delta_recomputed_from_rounded_avg_pp": round(delta_calc, 2),
        "delta_reported_pp": rows["delta_reported"],
        "ci95_pp_approx": [round(100 * lo, 2), round(100 * hi, 2)],
        "p_two_sided_approx": round(p_avg, 4)}
    T1_RES[budget] = out

# 검정력: 독립 두 비율, 기준 p=0.80, 차이 d(1.7 pp)를 80% 검정력으로 찾는 조건당 n
def n_per_arm(p, d, alpha=0.05, power=0.80):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return (za + zb) ** 2 * 2 * p * (1 - p) / d ** 2

RES["table1"] = T1_RES
RES["power"] = {
    "n_per_arm_for_1.7pp_p0.80": round(n_per_arm(0.80, 0.017)),
    "n_per_arm_for_2.0pp_p0.83": round(n_per_arm(0.83, 0.020)),
    "episodes_available_per_arm": N_LOCAL,
}

# ---------------------------------------------------------------- Table 2 (ablation)
# 출처: 논문 Table 2, Clean/Random %, 열 1/1, 2/2, 4/4
T2 = {
    "Shared": {"1/1": (82.6, 75.5), "2/2": (89.5, 87.3), "4/4": (90.6, 90.0)},
    "Continued": {"1/1": (82.3, 76.1), "2/2": (89.6, 87.7), "4/4": (90.5, 89.1)},
    "Original w/o F": {"1/1": (84.2, 76.2), "2/2": (90.4, 87.9), "4/4": (90.5, 89.8)},
    "Original + F + Proj": {"1/1": (82.5, 74.7), "2/2": (89.6, 87.7), "4/4": (90.7, 89.6)},
    "Restart w/o F": {"1/1": (84.5, 77.4), "2/2": (89.6, 88.3), "4/4": (91.0, 89.8)},
    "Restart + F, no Proj": {"1/1": (83.2, 76.7), "2/2": (88.9, 87.4), "4/4": (90.6, 90.5)},
    "APM (Restart + F + Proj)": {"1/1": (84.3, 77.5), "2/2": (90.4, 87.1), "4/4": (92.5, 90.3)},
}
# Table 1의 APM 및 Continued 열과 일치해야 한다
assert T2["APM (Restart + F + Proj)"]["1/1"] == T1["1/1"]["APM"][:2]
assert T2["APM (Restart + F + Proj)"]["2/2"] == T1["2/2"]["APM"][:2]
assert T2["APM (Restart + F + Proj)"]["4/4"] == T1["4/4"]["APM"][:2]
assert T2["Continued"]["1/1"] == T1["1/1"]["Continued"][:2]
assert T2["Continued"]["2/2"] == T1["2/2"]["Continued"][:2]
assert T2["Continued"]["4/4"] == T1["4/4"]["Continued"][:2]

def avg(x):
    return (x[0] + x[1]) / 2

def pair(a, b, budget):
    """두 Table 2 행의 평균 차와 독립 근사 검정(조건별 평균의 SE)."""
    ca, ra = T2[a][budget]
    cb, rb = T2[b][budget]
    se_c = two_prop_z(ca / 100, N_LOCAL, cb / 100, N_LOCAL)[1]
    se_r = two_prop_z(ra / 100, N_LOCAL, rb / 100, N_LOCAL)[1]
    d = (avg(T2[a][budget]) - avg(T2[b][budget])) / 100
    se = 0.5 * math.sqrt(se_c ** 2 + se_r ** 2)
    lo, hi = ci_diff(d, se)
    return {"avg_diff_pp": round(100 * d, 2), "ci95_pp_approx": [round(100 * lo, 2), round(100 * hi, 2)],
            "p_two_sided_approx": round(2 * (1 - stats.norm.cdf(abs(d / se))), 3),
            "clean_diff_pp": round(ca - cb, 1), "random_diff_pp": round(ra - rb, 1)}

T2_RES = {}
comparisons = [
    ("Restart w/o F", "Original w/o F", "anchor 효과(F 없음)"),
    ("Restart + F, no Proj", "Restart w/o F", "F 효과(투영 없음)"),
    ("APM (Restart + F + Proj)", "Restart + F, no Proj", "투영 효과"),
    ("APM (Restart + F + Proj)", "Continued", "APM 대 Continued"),
    ("APM (Restart + F + Proj)", "Shared", "APM 대 Shared"),
    ("Continued", "Shared", "Continued 대 Shared (Stage II 기준선 효과)"),
    ("Original + F + Proj", "Original w/o F", "Original에서 F+Proj 효과"),
]
for a, b, label in comparisons:
    T2_RES[f"{a} minus {b} [{label}]"] = {bud: pair(a, b, bud) for bud in ("1/1", "2/2", "4/4")}

# 본문 4.3의 문장과 표의 차이를 대조한다
text_claims = {
    "Restart vs Original Random (1/2/4)": ([T2["Restart w/o F"][b][1] - T2["Original w/o F"][b][1] for b in ("1/1", "2/2", "4/4")], [1.2, 0.4, 0.0]),
    "Restart vs Original Clean (1/2/4)": ([T2["Restart w/o F"][b][0] - T2["Original w/o F"][b][0] for b in ("1/1", "2/2", "4/4")], [0.3, -0.8, 0.5]),
    "F without Proj, Clean (1/2/4)": ([T2["Restart + F, no Proj"][b][0] - T2["Restart w/o F"][b][0] for b in ("1/1", "2/2", "4/4")], [-1.3, -0.7, 0.4]),
    "F without Proj, Random (1/2/4)": ([T2["Restart + F, no Proj"][b][1] - T2["Restart w/o F"][b][1] for b in ("1/1", "2/2", "4/4")], [-0.7, -0.9, 0.7]),
    "Projection avg gain (1/2/4)": ([avg(T2["APM (Restart + F + Proj)"][b]) - avg(T2["Restart + F, no Proj"][b]) for b in ("1/1", "2/2", "4/4")], [1.0, 0.6, 0.9]),
}
TEXT_CHECK = {}
for k, (calc, claimed) in text_claims.items():
    TEXT_CHECK[k] = [{"table_value": round(c, 2), "text_value": t, "match": abs(c - t) < 0.051} for c, t in zip(calc, claimed)]
RES["table2_pairs"] = T2_RES
RES["table2_text_check"] = TEXT_CHECK

# ---------------------------------------------------------------- Table 3 (real robot)
# 출처: 논문 Table 3, 과제당 24 시도
N_REAL = 24
T3 = {
    "PenHolder": {"Teacher (8/8)": (83.3, 4323.6), "Coarse teacher (1/1)": (29.2, 187.5), "APM (1/1)": (62.5, 203.1)},
    "FoldTowel": {"Teacher (8/8)": (70.8, 4323.6), "Coarse teacher (1/1)": (25.0, 187.5), "APM (1/1)": (75.0, 203.1)},
}
REAL = {}
for task, rows in T3.items():
    REAL[task] = {}
    for name, (s, _lat) in rows.items():
        k = round(s / 100 * N_REAL)
        REAL[task][name] = {"successes": k, "of": N_REAL, "rounding_ok": abs(k / N_REAL * 100 - s) < 0.05,
                            "wilson95_pct": [round(100 * x, 1) for x in wilson(k / N_REAL, N_REAL)]}
    for a, b in [("APM (1/1)", "Coarse teacher (1/1)"), ("APM (1/1)", "Teacher (8/8)")]:
        ka, kb = REAL[task][a]["successes"], REAL[task][b]["successes"]
        _, p_f = stats.fisher_exact([[ka, N_REAL - ka], [kb, N_REAL - kb]])
        REAL[task][f"Fisher {a} vs {b}"] = {"p_two_sided": round(float(p_f), 4)}
RES["real_robot"] = REAL
lat = {"teacher_ms": 4323.6, "coarse_ms": 187.5, "apm_ms": 203.1}
RES["latency"] = {
    "speedup_vs_teacher": round(lat["teacher_ms"] / lat["apm_ms"], 2),
    "reported_speedup": 21.3,
    "extra_vs_coarse_ms": round(lat["apm_ms"] - lat["coarse_ms"], 1),
    "reported_extra_ms": 15.6,
    "teacher_over_coarse": round(lat["teacher_ms"] / lat["coarse_ms"], 1),
}

# ---------------------------------------------------------------- 본문·초록 정합성
RES["consistency"] = {
    "stage_steps_total": 4000 + 5000,
    "project_page_eval_label": "step10000 EMA (project page 그림 캡션)",
    "psnr_gain_db": round(32.317 - 30.169, 3),
    "psnr_gain_reported_fig4_db": 2.15,
    "stage1_ktrain": [2, 4],
    "stage2_ktrain": [1, 2, 4],
    "teacher_full_budget_avg_reported_on_25_of_50_tasks": round((92.9 + 91.6) / 2, 2),
    "apm_4_4_minus_full_teacher_avg": round(91.4 - (92.9 + 91.6) / 2, 2),
    "shared_1_1_avg": round(avg((82.6, 75.5)), 2),
    "apm_1_1_minus_shared_pp": round(80.9 - avg((82.6, 75.5)), 2),
    "eval_episodes_per_condition": 50 * 50,
}

# ---------------------------------------------------------------- 보정 상태 인덱스 분포 (식 11 아래 Beta(5,2))
# 4단계 분기의 상태 인덱스 i = min(1 + floor(3b), 3), b ~ Beta(5, 2). i=1 → r=0.75, i=2 → r=0.5, i=3 → r=0.25
_b = stats.beta(5, 2)
RES["beta52_index"] = {
    "P_i1_r0.75": round(float(_b.cdf(1 / 3)), 4),
    "P_i2_r0.50": round(float(_b.cdf(2 / 3) - _b.cdf(1 / 3)), 4),
    "P_i3_r0.25": round(float(1 - _b.cdf(2 / 3)), 4),
    "mean_b": round(float(_b.mean()), 4),
}

with OUT.open("w", encoding="utf-8") as f:
    json.dump(RES, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------- 요약 출력
print("== Table 1 평균 재계산 일치")
for b, out in T1_RES.items():
    print(b, {k: v["ok"] for k, v in out["checks"].items()})
    for k, v in out["tests"].items():
        print("   ", k, v)
print("== 검정력", RES["power"])
print("== Table 2 쌍 비교 (평균 차, 근사 95% CI, p)")
for k, v in T2_RES.items():
    print(k, {b: (x["avg_diff_pp"], x["ci95_pp_approx"], x["p_two_sided_approx"]) for b, x in v.items()})
print("== 본문 대조")
for k, v in TEXT_CHECK.items():
    print(k, v)
print("== 실로봇", json.dumps(REAL, ensure_ascii=False))
print("== 지연", RES["latency"])
print("== 정합성", RES["consistency"])
print("== Beta(5,2) 인덱스", RES["beta52_index"])
print("wrote", OUT)
