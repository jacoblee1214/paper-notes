"""FR-VLA (arXiv:2610.25981 v1) 논문 표·그림 수치를 다시 계산하고 검정한다.

입력: 논문 본문, 표 2–4, 그림 5(막대 위 라벨)의 값을 아래에 그대로 옮겨 적었다(출처는 주석).
      외부 데이터는 쓰지 않는다. 학습이나 재현은 하지 않는다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 성공률의 분모는 캡션의 시도 수다(그림 5와 표 2: 단계당 20회, 표 3: 범주당 20회, 표 4: 그룹당 10회).
- 여러 단계를 하나의 풀로 합치는 검정은 시도가 독립이라고 가정한다. 같은 로봇·같은 정책의 시도가 묶여 있을 수 있어 p값은 낙관적이다.
- 그림 5의 막대 값은 막대 위에 적힌 라벨을 그대로 쓴다.
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


def newcombe_diff(k1, n1, k2, n2, z=1.959964):
    """Newcombe (1998) 방법 10: 두 비율 차이의 Wilson 기반 구간."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def fisher(k1, n1, k2, n2):
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(stats.fisher_exact(table, alternative="two-sided")[1])


def comp(name, k1, n1, k2, n2):
    d, lo, hi = newcombe_diff(k1, n1, k2, n2)
    return {
        "comparison": name,
        "k1": k1, "n1": n1, "k2": k2, "n2": n2,
        "diff_pp": round(100 * d, 1),
        "newcombe95_pp": [round(100 * lo, 1), round(100 * hi, 1)],
        "fisher_p": round(fisher(k1, n1, k2, n2), 4),
    }


# ---------- 1. 그림 5: 단계별 성공률 (막대 라벨, 시도 20회) ----------
STAGES = ["Gear", "SwitchTop", "SwitchSide", "Cleaning", "Screw", "Buckle"]
FIG5 = {  # 방법: 단계 순서대로
    "VLA-Adapter":  [55, 40, 0, 40, 10, 20],
    "VLA-Adapter w/F": [60, 35, 5, 45, 15, 20],
    "ForceVLA":     [70, 45, 10, 45, 25, 30],
    "TA-VLA":       [60, 30, 15, 50, 20, 40],
    "FR-VLA":       [90, 60, 35, 70, 40, 55],
}
N_STAGE = 20
means = {m: round(sum(v) / len(v), 2) for m, v in FIG5.items()}
RES["fig5_means_pct"] = means
best_base = max((m for m in FIG5 if m != "FR-VLA"), key=lambda m: means[m])
RES["fig5_strongest_baseline"] = {"method": best_base, "mean_pct": means[best_base],
                                  "gap_pp": round(means["FR-VLA"] - means[best_base], 2)}
# 단계별: FR-VLA가 모든 단계에서 최고인가, 그리고 가장 가까운 기준선과의 차이
per_stage = []
for i, s in enumerate(STAGES):
    fr = FIG5["FR-VLA"][i]
    others = {m: FIG5[m][i] for m in FIG5 if m != "FR-VLA"}
    top = max(others.values())
    top_m = [m for m, v in others.items() if v == top]
    k_fr, k_top = round(fr * N_STAGE / 100), round(top * N_STAGE / 100)
    per_stage.append({
        "stage": s, "fr_pct": fr, "best_baseline_pct": top, "best_baseline": top_m,
        "is_highest": fr > top,
        "fisher_p_vs_best_baseline": round(fisher(k_fr, N_STAGE, k_top, N_STAGE), 4),
    })
RES["fig5_per_stage"] = per_stage
RES["fig5_all_six_highest"] = all(p["is_highest"] for p in per_stage)
# 가장 큰 격차와 작은 격차(단계별)
RES["fig5_min_margin_pp"] = min(fr - b for fr, b in
                               [(p["fr_pct"], p["best_baseline_pct"]) for p in per_stage])

# ---------- 2. 표 2: 어블레이션 평균 ----------
T2 = {
    "FR-VLA": [90, 60, 35, 70, 40, 55],
    "w/o force admission": [60, 30, 15, 50, 25, 30],
    "w/o force tokenization": [70, 40, 20, 55, 20, 35],
}
RES["table2_avg_pct"] = {k: round(sum(v) / len(v), 2) for k, v in T2.items()}
RES["table2_all_stages_lower_than_full"] = {
    k: all(a < b for a, b in zip(v, T2["FR-VLA"])) for k, v in T2.items() if k != "FR-VLA"
}
# 풀링: 6단계 x 20회 = 120회
N_ALL = 6 * N_STAGE
k_fr = sum(T2["FR-VLA"]) * N_STAGE // 100
RES["table2_pooled"] = [
    comp("FR-VLA vs w/o admission (pooled 120)", k_fr, N_ALL,
         sum(T2["w/o force admission"]) * N_STAGE // 100, N_ALL),
    comp("FR-VLA vs w/o tokenization (pooled 120)", k_fr, N_ALL,
         sum(T2["w/o force tokenization"]) * N_STAGE // 100, N_ALL),
    comp("FR-VLA vs ForceVLA (pooled 120)", k_fr, N_ALL,
         sum(FIG5["ForceVLA"]) * N_STAGE // 100, N_ALL),
    comp("FR-VLA vs TA-VLA (pooled 120)", k_fr, N_ALL,
         sum(FIG5["TA-VLA"]) * N_STAGE // 100, N_ALL),
    comp("FR-VLA vs VLA-Adapter w/F (pooled 120)", k_fr, N_ALL,
         sum(FIG5["VLA-Adapter w/F"]) * N_STAGE // 100, N_ALL),
]

# ---------- 3. 본문 소규모 비교 (시도 20회) ----------
RES["gear_magnitude_gating"] = comp("Gear: FR-VLA 90% vs magnitude gating 70%", 18, 20, 14, 20)
RES["gear_vs_vla_adapter_wf"] = comp("Gear: FR-VLA 90% vs VLA-Adapter w/F 60% (fig5)", 18, 20, 12, 20)

# 표 3: 섭동 (시도 20회). 표는 VLA-Adapter w/F 대비만 있다.
RES["table3_gear_unseen"] = comp("Unseen gear: FR-VLA 85% vs VLA-Adapter w/F 60%", 17, 20, 12, 20)
RES["table3_target_pos"] = comp("Unseen target position: FR-VLA 75% vs VLA-Adapter w/F 50%", 15, 20, 10, 20)

# 표 4: 케이블 초기 위치 4그룹, 그룹당 10회. 방법명이 표에 없다(본문상 FR-VLA로 추정).
t4 = [4, 6, 5, 5]
k4 = sum(t4)
RES["table4_pooled_k_n"] = [k4, 40]
RES["table4_groups_wilson95_pct"] = [[round(100 * x, 1) for x in wilson(k, 10)] for k in t4]
RES["table4_pooled_wilson95_pct"] = [round(100 * x, 1) for x in wilson(k4, 40)]
RES["table4_vs_in_distribution_top"] = comp(
    "Cable unseen (20/40) vs in-distribution Switch-Top (12/20, fig5 60%)", k4, 40, 12, 20)

RES["fr_stage_wilson95_pct"] = {s: [round(100 * x, 1) for x in wilson(round(FIG5["FR-VLA"][i] * N_STAGE / 100), N_STAGE)] for i, s in enumerate(STAGES)}

# ---------- 4. 검정력: 90% 대 70%를 80% 검정력으로 찾으려면 단계당 몇 회인가 ----------
def n_needed(p1, p2, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)

RES["n_per_arm_for_80pct_power"] = {
    "90_vs_70": n_needed(0.90, 0.70),
    "58.3_vs_37.5": n_needed(0.583, 0.375),
    "85_vs_60": n_needed(0.85, 0.60),
}

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    print("평균(%):", RES["fig5_means_pct"])
    print("최고 기준선:", RES["fig5_strongest_baseline"])
    print("6단계 모두 최고:", RES["fig5_all_six_highest"], "최소 격차(pp):", RES["fig5_min_margin_pp"])
    for p in per_stage:
        print(" ", p["stage"], p["fr_pct"], "vs", p["best_baseline"], p["best_baseline_pct"],
              "Fisher p=", p["fisher_p_vs_best_baseline"])
    print("표2 평균:", RES["table2_avg_pct"], "모든 단계 낮음:", RES["table2_all_stages_lower_than_full"])
    for c in RES["table2_pooled"]:
        print(" ", c["comparison"], c["diff_pp"], c["newcombe95_pp"], "p=", c["fisher_p"])
    for key in ["gear_magnitude_gating", "gear_vs_vla_adapter_wf", "table3_gear_unseen",
                "table3_target_pos", "table4_vs_in_distribution_top"]:
        c = RES[key]
        print(" ", c["comparison"], c["diff_pp"], c["newcombe95_pp"], "p=", c["fisher_p"])
    print("표4 그룹 Wilson 95%:", RES["table4_groups_wilson95_pct"], "합:", RES["table4_pooled_k_n"])
    print("표4 합계 Wilson 95%:", RES["table4_pooled_wilson95_pct"])
    print("FR-VLA 단계별 Wilson 95%:", RES["fr_stage_wilson95_pct"])
    print("80% 검정력 필요 n/팔:", RES["n_per_arm_for_80pct_power"])
    print("->", OUT)
