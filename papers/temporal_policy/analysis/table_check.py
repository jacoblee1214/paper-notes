"""Temporal Policy 표 재계산 (CPU, numpy·scipy, 수 초). 실행: python3 analysis/table_check.py

- T1: Table I (Robomimic, 9셀 = 5과제 × ph/mh, Tool Hang은 ph만). 평균(Average)을 성공 횟수로 바꿔 Fisher·CMH·Holm.
      n = 150(3 시드 × 50 평가)은 근사다. Average는 체크포인트 10개 평균이라 이항 표본이 아니다.
      따라서 p값은 "유효 표본 150" 가정 아래의 근사다.
- T2: 박사 논문(thesis) Table 5.3 단일 단계(n = 100), Table 5.4 전달 비용(평균 ± SD, n = 100): Fisher, Welch.
- T3: Table II (논문, Square ph, 50 에피소드): 비율과 논문 식의 해석.
- T4: Table I 지연 분해: 같은 NFE(10)에서 모델 크기 차이(255M vs 17M)와 NFE 차이를 분리.
- T5: 실물 실험(10/20, 19/20): Wilson 구간. 비교 기준선이 없음을 기록.
- T6: 결합 비용 항등식: 단위 분산 가정에서 J_TP / J_indep = 1 - ρ(d), 정규화 분산 m이면 2m(1-ρ)/(1+m).

주의: Table I의 Max는 학습 전체의 최댓값이라 선택 편향이 있다. 여기서는 Average만 검정한다.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "table_check_results.json"
N_EFF = 150  # 3 시드 × 50 평가 (근사)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def cmh(tables):
    """Cochran–Mantel–Haenszel: tables = [[a, b, c, d], ...], a=성공(처치), b=실패(처치), c=성공(대조), d=실패(대조)."""
    num = 0.0   # Σ(a - E a) = Σ(ad - bc)/n
    R = 0.0     # Σ ad/n
    S = 0.0     # Σ bc/n
    var = 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        r1, r2 = a + b, c + d
        c1, c2 = a + c, b + d
        num += (a * d - b * c) / n
        R += a * d / n
        S += b * c / n
        var += r1 * r2 * c1 * c2 / (n * n * (n - 1)) if n > 1 else 0
    or_mh = R / S if S > 0 else float("nan")
    chi2 = (abs(num) - 0.5) ** 2 / var if var > 0 else float("nan")
    p = stats.chi2.sf(chi2, 1)
    return or_mh, chi2, p


# ---------- T1: Table I ----------
TABLE_I = [  # (과제, 분할, DP(max, avg), CFM(max, avg), TP(max, avg))
    ("Lift", "ph", (1.00, 1.00), (1.00, 1.00), (1.00, 1.00)),
    ("Lift", "mh", (1.00, 1.00), (1.00, 0.99), (1.00, 0.99)),
    ("Can", "ph", (1.00, 0.97), (1.00, 0.98), (1.00, 0.98)),
    ("Can", "mh", (0.99, 0.96), (0.99, 0.95), (1.00, 0.98)),
    ("Square", "ph", (0.98, 0.91), (0.97, 0.91), (0.99, 0.96)),
    ("Square", "mh", (0.93, 0.87), (0.85, 0.78), (0.91, 0.83)),
    ("Transport", "ph", (0.97, 0.89), (0.98, 0.91), (0.97, 0.91)),
    ("Transport", "mh", (0.82, 0.73), (0.67, 0.54), (0.85, 0.76)),
    ("Tool Hang", "ph", (0.87, 0.71), (0.79, 0.70), (0.81, 0.69)),
]


def t1():
    cells = []
    p_tp_dp, p_tp_cfm = [], []
    strata_dp, strata_cfm = [], []
    for task, split, dp, cfm, tp in TABLE_I:
        row = {"task": task, "split": split}
        k = {name: int(round(v[1] * N_EFF)) for name, v in (("DP", dp), ("CFM", cfm), ("TP", tp))}
        for name, v in (("DP", dp), ("CFM", cfm), ("TP", tp)):
            lo, hi = wilson(k[name], N_EFF)
            row[f"{name}_avg"] = v[1]
            row[f"{name}_avg_wilson95"] = [round(lo, 3), round(hi, 3)]
        _, p1 = stats.fisher_exact([[k["TP"], N_EFF - k["TP"]], [k["DP"], N_EFF - k["DP"]]])
        _, p2 = stats.fisher_exact([[k["TP"], N_EFF - k["TP"]], [k["CFM"], N_EFF - k["CFM"]]])
        row["TP_vs_DP_fisher_p"] = float(p1)
        row["TP_vs_CFM_fisher_p"] = float(p2)
        row["diff_TP_minus_DP_pp"] = round(100 * (tp[1] - dp[1]), 1)
        row["diff_TP_minus_CFM_pp"] = round(100 * (tp[1] - cfm[1]), 1)
        p_tp_dp.append(p1)
        p_tp_cfm.append(p2)
        strata_dp.append([k["TP"], N_EFF - k["TP"], k["DP"], N_EFF - k["DP"]])
        strata_cfm.append([k["TP"], N_EFF - k["TP"], k["CFM"], N_EFF - k["CFM"]])
        cells.append(row)
    adj_dp = holm(np.array(p_tp_dp))
    adj_cfm = holm(np.array(p_tp_cfm))
    for i, row in enumerate(cells):
        row["TP_vs_DP_holm18"] = float(min(1.0, max(adj_dp[i], 0)))
        row["TP_vs_CFM_holm18"] = float(min(1.0, adj_cfm[i]))
    # Holm은 18개(9셀 × 2비교)를 한꺼번에 보정
    allp = np.concatenate([p_tp_dp, p_tp_cfm])
    adj_all = holm(allp)
    for i, row in enumerate(cells):
        row["TP_vs_DP_holm18"] = float(adj_all[i])
        row["TP_vs_CFM_holm18"] = float(adj_all[len(cells) + i])
    or_dp, chi_dp, p_cmh_dp = cmh(strata_dp)
    or_cfm, chi_cfm, p_cmh_cfm = cmh(strata_cfm)
    return {
        "n_eff_assumed": N_EFF,
        "cells": cells,
        "CMH_TP_vs_DP": {"OR_MH": or_dp, "chi2": chi_dp, "p": p_cmh_dp},
        "CMH_TP_vs_CFM": {"OR_MH": or_cfm, "chi2": chi_cfm, "p": p_cmh_cfm},
        "note": "Average 기준. Max는 선택 편향이 있어 검정하지 않았다. 9셀 CMH는 Tool Hang ph를 포함한다.",
    }


# ---------- T2: thesis Table 5.3 (단일 단계, n=100) ----------
SINGLE_STEP = [  # 과제, 분할, CFM, TP  (성공률, n=100)
    ("Lift", "ph", 1.00, 1.00), ("Lift", "mh", 1.00, 1.00),
    ("Can", "ph", 0.98, 0.98), ("Can", "mh", 0.94, 0.97),
    ("Square", "ph", 0.84, 0.90), ("Square", "mh", 0.70, 0.88),
    ("Transport", "ph", 0.91, 0.92), ("Transport", "mh", 0.63, 0.67),
    ("Tool Hang", "ph", 0.56, 0.61),
]


def t2():
    rows, ps = [], []
    for task, split, cfm, tp in SINGLE_STEP:
        kc, kt = int(round(cfm * 100)), int(round(tp * 100))
        _, p = stats.fisher_exact([[kt, 100 - kt], [kc, 100 - kc]])
        ps.append(p)
        rows.append({"task": task, "split": split, "CFM": cfm, "TP": tp, "fisher_p": float(p)})
    adj = holm(np.array(ps))
    for r, a in zip(rows, adj):
        r["holm9"] = float(a)
    return rows


# ---------- T2b: thesis Table 5.4 (전달 비용, 평균 ± SD, n=100) ----------
def t2b():
    data = {"DP": (190.02, 3.94, 20.61, 1.67), "CFM": (9.99, 0.82, 1.09, 0.03), "TP": (1.21, 0.90, 1.02, 0.03)}
    out = {}
    for name in ("CFM", "DP"):
        m1, s1 = data["TP"][0], data["TP"][1]
        m2, s2 = data[name][0], data[name][1]
        se = math.sqrt(s1 * s1 / 100 + s2 * s2 / 100)
        t = (m2 - m1) / se
        out[f"transport_{name}_vs_TP"] = {"ratio": m2 / m1, "welch_t": t, "p": float(2 * stats.t.sf(abs(t), 190))}
        m1, s1 = data["TP"][2], data["TP"][3]
        m2, s2 = data[name][2], data[name][3]
        se = math.sqrt(s1 * s1 / 100 + s2 * s2 / 100)
        t = (m2 - m1) / se
        out[f"straightness_{name}_vs_TP"] = {"diff": m2 - m1, "welch_t": t, "p": float(2 * stats.t.sf(abs(t), 190))}
    return out


# ---------- T3: Table II (논문) ----------
def t3():
    tr = {"DP": 189.99, "CFM": 10.00, "TP": 1.21}
    st = {"DP": 20.61, "CFM": 1.08, "TP": 1.02}
    return {
        "transport_ratio_CFM_over_TP": tr["CFM"] / tr["TP"],
        "transport_ratio_DP_over_TP": tr["DP"] / tr["TP"],
        "transport_ratio_DP_over_CFM": tr["DP"] / tr["CFM"],
        "straightness_ratio_eq8_for_10_straight_steps_expected": 1 / 10,
        "note": "Eq.7(제곱 합)·Eq.8(제곱 거리)로 10단계 직선 경로를 계산하면 1/10이다. 논문의 1.02는 코드의 길이 기준(compute_transport)과 맞는다.",
        "straightness_reported": st,
    }


# ---------- T4: 지연 분해 ----------
def t4():
    rows = {"DP": (615.6, 100, 255), "CFM": (63.5, 10, 255), "TP": (19.1, 10, 17)}
    per_nfe = {k: v[0] / v[1] for k, v in rows.items()}
    return {
        "ms_per_nfe": per_nfe,
        "CFM_over_TP_same_NFE10": 63.5 / 19.1,
        "params_ratio_CFM_over_TP": 255 / 17,
        "thesis_breakdown_TP_RTX4080": {"visual_encoder_ms": 1.86, "ode_10_steps_ms": 17.78,
                                        "per_step_ms": 1.78, "total_ms": 19.62},
        "note": "NFE가 같은 10일 때 3.3배 차이는 모델 크기(255M → 17M)와 함께 바뀌었다. 이 비교로 이력 초기화의 기여를 분리할 수 없다.",
    }


# ---------- T5: 실물 (10/20, 19/20) ----------
def t5():
    return {
        "overall_10_of_20": {"rate": 0.5, "wilson95": wilson(10, 20)},
        "grasp_19_of_20": {"rate": 0.95, "wilson95": wilson(19, 20)},
        "note": "기준선(다른 정책)이 없다. 20회는 단일 과제 한 번의 시행이다.",
    }


# ---------- T6: 결합 비용 항등식 ----------
def t6():
    target = 1.21 / 10.00
    res = {}
    for m in (1.0, 1 / 3):
        rho_needed = 1 - target * (1 + m) / (2 * m)
        res[f"m={m:.3f}"] = {
            "rho_lag14_needed_for_ratio_0.121": float(rho_needed),
            "ratio_at_rho_0.9": 2 * m * (1 - 0.9) / (1 + m),
            "ratio_at_rho_0.8": 2 * m * (1 - 0.8) / (1 + m),
        }
    res["derivation"] = "J_TP = 2HDm(1-ρ_d), J_indep = HD(1+m); m는 정규화 후 E[x²] (논문은 m=1을 가정, 코드는 [-1,1] 정규화이므로 m<1)"
    return res


def main():
    out = {
        "T1_table_I": t1(),
        "T2_thesis_single_step_n100": t2(),
        "T2b_thesis_transport_n100": t2b(),
        "T3_table_II": t3(),
        "T4_latency": t4(),
        "T5_real_world": t5(),
        "T6_coupling_identity": t6(),
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    t = out["T1_table_I"]
    print("T1 CMH TP vs DP OR=%.2f p=%.3g | TP vs CFM OR=%.2f p=%.3g" % (
        t["CMH_TP_vs_DP"]["OR_MH"], t["CMH_TP_vs_DP"]["p"], t["CMH_TP_vs_CFM"]["OR_MH"], t["CMH_TP_vs_CFM"]["p"]))
    for c in t["cells"]:
        print(f"  {c['task']:10s} {c['split']}  TP-DP {c['diff_TP_minus_DP_pp']:+.1f}pp p={c['TP_vs_DP_fisher_p']:.3f}"
              f"  TP-CFM {c['diff_TP_minus_CFM_pp']:+.1f}pp p={c['TP_vs_CFM_fisher_p']:.3f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
