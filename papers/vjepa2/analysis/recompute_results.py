"""V-JEPA 2 논문 보고 수치 재계산 (CPU, 표준 라이브러리만 사용).

입력: 논문 표의 수치를 코드에 직접 적음 (paper/paper.txt의 Table 2, 3, 4, 5, 8).
출력: analysis/recompute_results.json
"""
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

def fisher_two_sided(a, b, c, d):
    """2x2 표 [[a,b],[c,d]] 의 양측 Fisher 정확검정 p값 (초기하분포 합)."""
    n1, n2, m1 = a + b, c + d, a + c
    N = n1 + n2
    def p_of(x):
        return math.comb(m1, x) * math.comb(N - m1, n1 - x) / math.comb(N, n1)
    lo, hi = max(0, m1 - n2), min(n1, m1)
    p_obs = p_of(a)
    return min(1.0, sum(p_of(x) for x in range(lo, hi + 1) if p_of(x) <= p_obs * (1 + 1e-9)))

out = {}

# ---- Table 2: 두 로봇 랩, 각 셀 10회 시도 ----
t2 = {
    "Octo": {"Reach": (100, 100), "Grasp_Cup": (20, 10), "Grasp_Box": (0, 0),
             "ReachObj_Cup": (20, 10), "ReachObj_Box": (70, 70),
             "PnP_Cup": (20, 10), "PnP_Box": (10, 10)},
    "VJEPA2AC": {"Reach": (100, 100), "Grasp_Cup": (70, 60), "Grasp_Box": (30, 20),
                 "ReachObj_Cup": (90, 60), "ReachObj_Box": (80, 70),
                 "PnP_Cup": (80, 80), "PnP_Box": (80, 50)},
}
reported_avg_t2 = {
    "Octo": {"Reach": 100, "Grasp_Cup": 15, "Grasp_Box": 0, "ReachObj_Cup": 15, "ReachObj_Box": 70, "PnP_Cup": 15, "PnP_Box": 10},
    "VJEPA2AC": {"Reach": 100, "Grasp_Cup": 65, "Grasp_Box": 25, "ReachObj_Cup": 75, "ReachObj_Box": 75, "PnP_Cup": 80, "PnP_Box": 65},
}
t2_res = {}
for m in t2:
    t2_res[m] = {}
    for task, (lab1, lab2) in t2[m].items():
        avg = (lab1 + lab2) / 2
        k = round(lab1 / 10 + lab2 / 10)  # 두 랩 합계 성공 수 (n=20)
        lo, hi = wilson(k, 20)
        t2_res[m][task] = {"avg_recomputed": avg, "avg_reported": reported_avg_t2[m][task],
                           "successes_of_20": k, "wilson95": [round(lo, 3), round(hi, 3)]}
out["table2_avg_check"] = t2_res

# Octo vs V-JEPA 2-AC, 풀링 n=20 per cell, Fisher 양측
cmp = {}
for task in t2["Octo"]:
    kO = round(sum(t2["Octo"][task]) / 10)
    kV = round(sum(t2["VJEPA2AC"][task]) / 10)
    p = fisher_two_sided(kV, 20 - kV, kO, 20 - kO)
    cmp[task] = {"VJEPA2AC_of_20": kV, "Octo_of_20": kO, "fisher_p_two_sided": round(p, 5)}
out["table2_vjepa_vs_octo_fisher"] = cmp

# ---- Table 3: Lab 2 only, 10회 ----
t3 = {"Cosmos": {"Reach": 8, "Grasp_Cup": 0, "Grasp_Box": 2, "PnP_Cup": 0, "PnP_Box": 0},
      "VJEPA2AC": {"Reach": 10, "Grasp_Cup": 6, "Grasp_Box": 2, "PnP_Cup": 8, "PnP_Box": 5}}
t3_cmp = {}
for task in t3["Cosmos"]:
    kC, kV = t3["Cosmos"][task], t3["VJEPA2AC"][task]
    p = fisher_two_sided(kV, 10 - kV, kC, 10 - kC)
    t3_cmp[task] = {"VJEPA2AC_of_10": kV, "Cosmos_of_10": kC,
                    "VJEPA2AC_wilson95": [round(x, 3) for x in wilson(kV, 10)],
                    "Cosmos_wilson95": [round(x, 3) for x in wilson(kC, 10)],
                    "fisher_p_two_sided": round(p, 4)}
out["table3_lab2_vjepa_vs_cosmos"] = t3_cmp

# 평균 시간 비교: 16 s vs 4 min per action (텍스트 수치)
out["planning_time_ratio_cosmos_over_vjepa"] = round(240 / 16, 1)

# ---- Table 4: 6개 과제 평균 (보고된 Avg와 비교) ----
t4 = {
    "V-JEPA 2 ViT-L": ([73.7, 89.0, 97.6, 85.1, 86.8, 83.5], 86.0),
    "V-JEPA 2 ViT-H": ([74.0, 89.8, 97.7, 85.3, 87.9, 83.8], 86.4),
    "V-JEPA 2 ViT-g": ([75.3, 90.1, 97.7, 86.6, 90.7, 84.6], 87.5),
    "V-JEPA 2 ViT-g384": ([77.3, 90.2, 97.8, 87.3, 91.1, 85.1], 88.2),
    "DINOv2": ([50.7, 82.5, 93.4, 83.6, 90.7, 86.1], 81.1),
    "PEcore G": ([55.4, 76.9, 90.0, 88.5, 95.3, 87.6], 82.3),
}
out["table4_avg_check"] = {k: {"mean_of_6": round(sum(v[0]) / 6, 2), "reported_avg": v[1]} for k, v in t4.items()}

# ---- Table 5: EK100 action anticipation R@5, 상대 향상 ----
prev_best_action = 27.6  # PlausiVL 8B
vj = 39.7
out["table5_rel_improvement_pct"] = round((vj - prev_best_action) / prev_best_action * 100, 1)

# ---- Table 8: 7개 벤치마크 평균 ----
t8 = {
    "V-JEPA 2 ViT-g384 + Llama3.1 8B": ([84.0, 44.5, 76.9, 36.7, 40.3, 60.6, 73.5], 59.5),
    "PLM 8B": ([82.7, 39.7, 72.7, 28.3, 33.2, 63.5, 77.1], 56.7),
    "InternVL-2.5": ([68.9, 39.9, 68.3, 24.3, 29.4, 61.6, 72.6], 52.1),
}
out["table8_avg_check"] = {k: {"mean_of_7": round(sum(v[0]) / 7, 2), "reported_avg": v[1]} for k, v in t8.items()}

with open(os.path.join(HERE, "recompute_results.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=1))
