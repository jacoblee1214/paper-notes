"""DexUMI (arXiv:2505.21864 v3) Table 1 재계산과 검정.

입력: 논문 Table 1(본문 5절, paper/paper.txt 기준 v3). 시도 기록은 공개되지 않았다.
      각 셀은 과제당 20회 평가 에피소드의 단계별 누적 성공률이다(0.05 단위 = 1회).
      이 스크립트는 scipy만 쓴다. 출력: analysis/recompute_results.json
실행: python3 analysis/recompute.py
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "recompute_results.json"
N = 20

# 열: IH(Inspire) Cube, Carton, Tea-tool, Tea-leaf | XH(XHand) Tea-tool, Tea-leaf, Knob, Pan, Salt
COLS = ["IH_cube", "IH_carton", "IH_tea_tool", "IH_tea_leaf",
        "XH_tea_tool", "XH_tea_leaf", "XH_knob", "XH_pan", "XH_salt"]
# 단계별 누적 성공 이후 최종 단계 열 (IH: cube, carton, tea_leaf / XH: tea_leaf, salt)
FINAL = ["IH_cube", "IH_carton", "IH_tea_leaf", "XH_tea_leaf", "XH_salt"]
T1 = {
    "Rel_Yes_Inpaint": [1.00, 0.85, 1.00, 0.85, 1.00, 0.85, 0.95, 0.95, 0.75],
    "Abs_Yes_Inpaint": [0.10, 0.35, 0.80, 0.00, 1.00, 0.25, 0.50, 0.45, 0.00],
    "Rel_No_Inpaint":  [0.95, 0.90, 1.00, 0.90, 0.95, 0.80, 0.95, 0.95, 0.15],
    "Abs_No_Inpaint":  [0.90, 0.85, 0.90, 0.60, 1.00, 0.75, 0.60, 0.60, 0.00],
    "Rel_No_Mask":     [0.60, 0.10, 0.90, 0.50, None, None, None, None, None],
    "Rel_No_Raw":      [0.20, 0.05, 0.85, 0.05, None, None, None, None, None],
}
REPORTED_AVG = 0.86  # 초록: average task success rate 86%
CT = {"DexUMI": 36, "bare_hand": 51, "teleop": 11}  # 그림 7 막대 값 (그림 판독)
REPORTED_RATIO = 3.2


def cnt(v):
    return int(round(v * N))


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def fisher(a, b):
    """a, b: (성공, 시도) 쌍. 양측 Fisher 정확 검정."""
    table = [[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]]
    return float(stats.fisher_exact(table)[1])


def h_power(p1, p2, alpha=0.05, power=0.8):
    """Cohen's h로 두 비율을 구분하는 1군당 필요 표본 수(근사)."""
    h = abs(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return h, math.ceil(((za + zb) / h) ** 2)


def main():
    res = {}
    # 1. 초록 86% 재현: 최종 단계 5셀의 평균
    res["avg_final_rel_yes"] = float(np.mean([T1["Rel_Yes_Inpaint"][COLS.index(c)] for c in FINAL]))
    res["avg_final_by_row"] = {}
    for name, row in T1.items():
        vals = [row[COLS.index(c)] for c in FINAL if row[COLS.index(c)] is not None]
        res["avg_final_by_row"][name] = {"mean": float(np.mean(vals)) if vals else None, "n_cells": len(vals)}
    res["avg_all9_rel_yes"] = float(np.mean(T1["Rel_Yes_Inpaint"]))

    # 2. 셀별 Wilson 95% CI (최종 단계 5셀, 주 설정)
    res["wilson_final_rel_yes"] = {}
    for c in FINAL:
        k = cnt(T1["Rel_Yes_Inpaint"][COLS.index(c)])
        lo, hi = wilson(k, N)
        res["wilson_final_rel_yes"][c] = {"k": k, "n": N, "ci95": [round(lo, 3), round(hi, 3)]}

    # 3. 시각 표현 절제: Mask/Raw 대 Inpaint (촉각 없음, IH cube·carton)
    abl = {}
    for variant in ["Rel_No_Mask", "Rel_No_Raw"]:
        ps = []
        names = []
        for c in ["IH_cube", "IH_carton", "IH_tea_tool", "IH_tea_leaf"]:
            i = COLS.index(c)
            if T1[variant][i] is None:
                continue
            a = (cnt(T1["Rel_No_Inpaint"][i]), N)
            b = (cnt(T1[variant][i]), N)
            ps.append(fisher(a, b))
            names.append(c)
        adj = holm(ps)
        abl[variant] = {c: {"p_raw": round(p, 6), "p_holm": round(q, 6)} for c, p, q in zip(names, ps, adj)}
    # Cube + Carton 합산(독립 과제 2개, 각 40회) Inpaint 대 Mask/Raw
    pooled = {}
    for variant in ["Rel_No_Mask", "Rel_No_Raw"]:
        ki = cnt(T1["Rel_No_Inpaint"][0]) + cnt(T1["Rel_No_Inpaint"][1])
        kv = cnt(T1[variant][0]) + cnt(T1[variant][1])
        pooled[variant] = {"inpaint": [ki, 2 * N], "variant": [kv, 2 * N],
                           "p": float(stats.fisher_exact([[ki, 2 * N - ki], [kv, 2 * N - kv]])[1])}
    res["ablation_inpaint_vs_variant"] = {"per_task": abl, "pooled_cube_carton": pooled}

    # 4. 행동 표현: 상대 대 절대 (과제 열별 Fisher, Holm 보정). 촉각 있음/없음 각각
    for tac in ["Yes", "No"]:
        R = T1[f"Rel_{tac}_Inpaint"]
        A = T1[f"Abs_{tac}_Inpaint"]
        ps, rows = [], []
        for i, c in enumerate(COLS):
            a = (cnt(R[i]), N)
            b = (cnt(A[i]), N)
            ps.append(fisher(a, b))
            rows.append((c, cnt(R[i]), cnt(A[i])))
        adj = holm(ps)
        res[f"rel_vs_abs_tactile_{tac}"] = [
            {"col": c, "rel_k": rk, "abs_k": ak, "p_raw": round(p, 6), "p_holm": round(q, 6)}
            for (c, rk, ak), p, q in zip(rows, ps, adj)
        ]
        res[f"rel_ge_abs_count_tactile_{tac}"] = {
            "rel_better": int(sum(rk > ak for _, rk, ak in rows)),
            "tie": int(sum(rk == ak for _, rk, ak in rows)),
            "abs_better": int(sum(rk < ak for _, rk, ak in rows)),
        }

    # 5. 촉각 효과 (같은 행동 표현 안에서 Yes 대 No, XHand 9열 중 XH 열)
    tac = {}
    for act in ["Rel", "Abs"]:
        Y = T1[f"{act}_Yes_Inpaint"]
        Nn = T1[f"{act}_No_Inpaint"]
        better = worse = same = 0
        for i in range(4, 9):  # XH 열
            if Y[i] > Nn[i]:
                better += 1
            elif Y[i] < Nn[i]:
                worse += 1
            else:
                same += 1
        tac[act] = {"xh_tactile_better": better, "xh_tactile_worse": worse, "xh_same": same}
    tac["xh_salt_rel_yes_vs_no_p"] = fisher((cnt(T1["Rel_Yes_Inpaint"][8]), N), (cnt(T1["Rel_No_Inpaint"][8]), N))
    tac["ih_rel_yes_vs_no"] = {
        "better": int(sum(T1["Rel_Yes_Inpaint"][i] > T1["Rel_No_Inpaint"][i] for i in range(4))),
        "worse": int(sum(T1["Rel_Yes_Inpaint"][i] < T1["Rel_No_Inpaint"][i] for i in range(4))),
    }
    res["tactile_effects"] = tac

    # 6. 검정력: 20회 두 군에서 구분 가능한 최소 차이 (기저 0.85, 0.60)
    res["power"] = {}
    for base in [0.85, 0.60, 0.95]:
        for diff in [0.15, 0.20, 0.30]:
            p2 = base - diff
            if p2 < 0:
                continue
            h, need = h_power(base, p2)
            res["power"][f"base{base}_diff{diff}"] = {"cohen_h": round(h, 3), "n_per_arm_needed": need}

    # 7. 수집 효율 (그림 7 막대 판독, n = 세션 1회로 추정)
    ratio = CT["DexUMI"] / CT["teleop"]
    res["collection_efficiency"] = {"CT": CT, "ratio_dexumi_over_teleop": round(ratio, 3),
                                    "reported_ratio": REPORTED_RATIO,
                                    "note": "그림 판독값. 세션 반복 수와 분산이 없어 구간을 못 만든다."}
    res["avg_reported_abstract"] = REPORTED_AVG
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))

    # 요약 출력
    print(f"최종 단계 5셀 평균(주 설정): {res['avg_final_rel_yes']:.3f} (초록 {REPORTED_AVG})")
    print(f"9열 평균(주 설정): {res['avg_all9_rel_yes']:.3f}")
    for k, v in res["avg_final_by_row"].items():
        print(f"  {k}: 최종 5셀 평균 {v['mean'] if v['mean'] is None else round(v['mean'],3)} (셀 {v['n_cells']})")
    print("Wilson 95% (주 설정 최종 단계):", {c: v["ci95"] for c, v in res["wilson_final_rel_yes"].items()})
    print("Mask/Raw 대 Inpaint, 큐브+카턴 합산:", {k: round(v["p"], 8) for k, v in pooled.items()})
    print("상대>=절대 (촉각 Yes):", res["rel_ge_abs_count_tactile_Yes"], " (촉각 No):", res["rel_ge_abs_count_tactile_No"])
    print("촉각 효과 XH 열:", tac)
    print("검정력 예:", {k: v for k, v in res["power"].items() if k.startswith("base0.85")})
    print(f"수집 효율 비율 {ratio:.2f} (논문 {REPORTED_RATIO})")


if __name__ == "__main__":
    main()
