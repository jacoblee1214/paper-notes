"""MoRE (arXiv:2611.25336 v1) 논문 표 수치를 다시 계산하고 일관성·통계 검사를 한다.

입력: 논문 본문과 표의 값을 아래에 그대로 옮겨 적었다(출처 주석). 외부 데이터는 쓰지 않는다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 표 2·3·4의 값에는 표준편차, 구간, 시드 수가 없다. 그래서 평균 차이의 검정은 하지 않고
  상대 변화와 표시 해상도(소수 셋째 자리 반올림)만 계산한다.
- 실로봇 MPC(그림 4) 시도 수는 본문에 없다. 곡선의 계단 간격 0.1에서 n = 10으로 읽었다(추정).
- 시뮬 MPC 10회 (본문 4.3)는 본문에 명시되어 있다.
- 표 5 기체 정확도의 분모는 "50 random interactions per task category"(부록 A.4)로 되어 있다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}

# ---------- 표 2: 동역학 비교 (RMSE, CD, Rigidity|Laplacian). 출처 논문 Table 2 ----------
T2 = {
    "rigid_np": {
        "AdaptiGraph": (0.028, 0.084, 0.0001),
        "ParticleFormer": (0.054, 0.082, 0.0005),
        "Privileged Expert": (0.021, 0.048, 0.0),
        "Ours (MoE)": (0.022, 0.048, 0.0),
    },
    "deform_np": {
        "AdaptiGraph": (0.031, 0.052, 0.0068),
        "ParticleFormer": (0.061, 0.079, 0.0072),
        "Privileged Expert": (0.018, 0.032, 0.0027),
        "Ours (MoE)": (0.024, 0.027, 0.0020),
    },
    "deform_p": {
        "AdaptiGraph": (0.047, 0.143, 0.0092),
        "ParticleFormer": (0.058, 0.075, 0.0058),
        "Privileged Expert": (0.029, 0.038, 0.0068),
        "Ours (MoE)": (0.026, 0.039, 0.0050),
    },
}

# ---------- 표 3: 절제. 출처 논문 Table 3 ----------
T3 = {
    "rigid_np": {
        "Single Expert": (0.030, 0.054, 0.0001),
        "Implicit Router": (0.026, 0.051, 0.0),
        "Implicit MoE": (0.045, 0.070, 0.0002),
        "w/o Self-Rollout": (0.031, 0.057, 0.0),
        "Ours (Full)": (0.022, 0.048, 0.0),
    },
    "deform_np": {
        "Single Expert": (0.028, 0.038, 0.0038),
        "Implicit Router": (0.032, 0.033, 0.0032),
        "Implicit MoE": (0.026, 0.031, 0.0032),
        "w/o Self-Rollout": (0.035, 0.039, 0.0056),
        "Ours (Full)": (0.024, 0.027, 0.0020),
    },
    "deform_p": {
        "Single Expert": (0.037, 0.047, 0.0061),
        "Implicit Router": (0.028, 0.040, 0.0054),
        "Implicit MoE": (0.032, 0.039, 0.0071),
        "w/o Self-Rollout": (0.047, 0.060, 0.0058),
        "Ours (Full)": (0.026, 0.039, 0.0050),
    },
}

# ---------- 표 4: DROID 비교. 출처 논문 Table 4 ----------
T4 = {
    "rigid_np": {"PointWorld": (0.031, 0.054, 0.0003), "Ours": (0.023, 0.046, 0.0)},
    "deform_np": {"PointWorld": (0.028, 0.031, 0.0041), "Ours": (0.014, 0.026, 0.0022)},
    "deform_p": {"PointWorld": (0.029, 0.035, 0.0063), "Ours": (0.021, 0.031, 0.0051)},
}

MET = ["RMSE", "CD", "Lap/Rig"]


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    # 1) 표 4: 상대 변화 (ours 대 PointWorld). 논문 본문 "50% RMSE" 주장 확인
    rel4 = {}
    for reg, d in T4.items():
        pw, ou = d["PointWorld"], d["Ours"]
        rel4[reg] = {
            "RMSE": round(100 * (ou[0] - pw[0]) / pw[0], 1),
            "CD": round(100 * (ou[1] - pw[1]) / pw[1], 1),
            "Lap_or_Rig_abs": [pw[2], ou[2]],
        }
        if pw[2] > 0:
            rel4[reg]["Lap_or_Rig_rel"] = round(100 * (ou[2] - pw[2]) / pw[2], 1)
    RES["table4_relative_change_pct"] = rel4
    print("표 4 상대 변화(%):", json.dumps(rel4, ensure_ascii=False))

    # 2) 표 3: 절제 - 전체 모델 대비 각 변형이 더 나쁜(>) 지표 수, 같은(=) 지표 수
    abl = {}
    for reg, d in T3.items():
        full = d["Ours (Full)"]
        for name, v in d.items():
            if name == "Ours (Full)":
                continue
            worse = sum(1 for a, b in zip(v, full) if a > b)
            tie = sum(1 for a, b in zip(v, full) if a == b)
            abl.setdefault(name, {})[reg] = {"worse": worse, "tie": tie, "better": 3 - worse - tie}
    RES["table3_vs_full"] = abl
    print("표 3 전체 모델 대비 (worse/tie/better, 지표 3개 중):")
    for name, regs in abl.items():
        print(f"  {name:18s}", {r: (v["worse"], v["tie"], v["better"]) for r, v in regs.items()})
    # 본문 "consistently underperforms": Implicit MoE의 촉각 CD는 동률(0.039 = 0.039)
    RES["implicit_moe_tie_example"] = {"deform_p_CD": [T3["deform_p"]["Implicit MoE"][1], T3["deform_p"]["Ours (Full)"][1]]}

    # 3) 표 2: 우리 방법과 privileged expert 차이 (부호: 양수 = 우리가 나쁨)
    priv = {}
    for reg, d in T2.items():
        a, b = d["Ours (MoE)"], d["Privileged Expert"]
        priv[reg] = {m: round(x - y, 4) for m, x, y in zip(MET, a, b)}
    RES["table2_ours_minus_privileged"] = priv
    print("표 2 우리-privileged (양수 = 우리가 나쁨):", priv)

    # 4) 표시 해상도: 소수 셋째 자리 반올림 => 최대 0.0005. 상대 해상도
    res = {}
    for reg, d in T3.items():
        full = d["Ours (Full)"]
        res[reg] = {"rmse_half_step_rel_pct": round(100 * 0.0005 / full[0], 1)}
    RES["rounding_resolution_rmse_pct"] = res
    print("RMSE 반올림 해상도(%):", res)

    # 5) 시뮬 MPC 10회: 5/10 대 2/10
    k1, n1, k2, n2 = 5, 10, 2, 10
    fe = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
    w1, w2 = wilson(k1, n1), wilson(k2, n2)
    # Newcombe 차이 구간
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    lo = d - math.sqrt((p1 - w1[0]) ** 2 + (w2[1] - p2) ** 2)
    hi = d + math.sqrt((w1[1] - p1) ** 2 + (p2 - w2[0]) ** 2)
    # 검정력: 50% 대 20%를 80% 검정력으로 찾을 1군당 n (정규근사, 양측 0.05)
    za, zb = 1.959964, 0.841621
    pbar = (0.5 + 0.2) / 2
    n_need = ((za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(0.5 * 0.5 + 0.2 * 0.8)) / 0.3) ** 2
    RES["mpc_sim_10trials"] = {
        "fisher_two_sided_p": round(fe.pvalue, 3),
        "wilson95_moe": [round(100 * w1[0], 1), round(100 * w1[1], 1)],
        "wilson95_dinowm": [round(100 * w2[0], 1), round(100 * w2[1], 1)],
        "newcombe_diff_pp": [round(100 * lo, 1), round(100 * hi, 1)],
        "n_per_arm_for_80pct_power_50_vs_20": math.ceil(n_need),
    }
    print("시뮬 MPC:", RES["mpc_sim_10trials"])

    # 6) 표 5 기체 정확도: 3개 범주, 범주당 50회
    cats = {"Push-X": 95.4, "Manipulate Toy": 94.9, "Soft Grasp": 99.8}
    n = 50
    gate = {}
    for name, pct in cats.items():
        k_exact = pct / 100 * n
        nearest = round(k_exact)
        gate[name] = {
            "reported_pct": pct,
            "implied_k_of_50": round(k_exact, 2),
            "nearest_k": nearest,
            "nearest_pct": round(100 * nearest / n, 1),
            "wilson95_nearest_k": [round(100 * x, 1) for x in wilson(nearest, n)],
            "k_over_50_matches_reported": abs(100 * nearest / n - pct) < 0.05,
        }
    overall_mean = round(np.mean(list(cats.values())), 2)
    k_tot = 145  # 150회 전체에서 145/150 = 96.67%
    gate["overall"] = {
        "unweighted_mean_of_three": overall_mean,
        "k_of_150_145_pct": round(100 * k_tot / 150, 2),
        "wilson95_145_of_150": [round(100 * x, 1) for x in wilson(k_tot, 150)],
    }
    RES["table5_gating"] = gate
    print("표 5:", json.dumps(gate, ensure_ascii=False, indent=1))

    # 7) 데이터 규모: 본문 20K + 7K vs 프로젝트 페이지 38.7K
    RES["dataset_episodes"] = {"paper_sim": 20000, "paper_real": 7000, "paper_sum": 27000,
                               "project_page_claim": 38700, "gap": 38700 - 27000}

    OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
