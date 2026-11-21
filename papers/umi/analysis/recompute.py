"""UMI (arXiv:2402.10329 v3, RSS 2024) 보고 수치 재계산과 일관성 검사 (numpy, scipy).
실행: python3 analysis/recompute.py   결과: analysis/recompute_results.json
원자료(시행별 기록)는 없으므로 모든 검정은 '보고된 성공 횟수'에서 다시 계산한다.
시행은 독립 이항으로 가정하고 같은 장면을 반복하는 것의 상관(시드, 장면 고정)은 반영하지 않는다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fisher(k1, n1, k2, n2):
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(stats.fisher_exact(table, alternative="two-sided")[1])


# ---------------------------------------------------------------- 1. 과제별 성공률 (본문 V-A..V-D, 그림 8)
# (이름, 성공, 시도, 논문 보고 위치)
ROWS = [
    ("Cup 전체 방법 (UMI)", 20, 20, "§V-A"),
    ("Cup, 절대 행동", 5, 20, "§V-A, 그림 8c"),
    ("Cup, 상대 델타 행동", 16, 20, "§V-A, 그림 8c"),
    ("Cup, 어안 렌즈 없음", 11, 20, "§V-A, 그림 8c"),
    ("Cup, 거울 없음", 18, 20, "§V-A"),
    ("Cup, 거울 있으나 반사 없음", 17, 20, "§V-A"),
    ("Cup, Franka FR2 교차 로봇", 18, 20, "§V-A (그림 1, 8)"),
    ("Tossing, UMI (지연 맞춤)", 105, 120, "§V-B"),
    ("Tossing, 지연 맞춤 없음", 69, 120, "§V-B, 그림 8"),
    ("Cloth, UMI", 14, 20, "§V-C"),
    ("Cloth, 상대 그리퍼 자세 없음", 6, 20, "§V-C"),
    ("Dish, UMI (ViT-B CLIP)", 14, 20, "§V-D"),
    ("Dish, ResNet-34 (본문은 0/10)", 0, 20, "§V-D 본문 '0/10', 그림 8은 20회"),
    ("Cup in-the-wild, 학습 컵", 28, 40, "§VI"),
    ("Cup in-the-wild, 미지 컵", 15, 20, "§VI"),
    ("Cup in-the-wild, 전체 (카페 22/35 + 분수 21/25)", 43, 60, "§VI, 그림 9c"),
    ("Cup 협소 데이터만 학습, 미지 환경", 0, 20, "§VI 그림 9b (카페 0/10 + 분수 0/10)"),
]
res["rates"] = []
for name, k, n, where in ROWS:
    lo, hi = wilson(k, n)
    res["rates"].append({"name": name, "k": k, "n": n, "rate": round(k / n, 4),
                         "wilson95": [round(lo, 3), round(hi, 3)], "where": where})

# ---------------------------------------------------------------- 2. 대비 검정 (Fisher 양측)
PAIRS = [
    ("Cup: UMI vs 어안 없음", (20, 20), (11, 20)),
    ("Cup: UMI vs 절대 행동", (20, 20), (5, 20)),
    ("Cup: UMI vs 델타 행동", (20, 20), (16, 20)),
    ("Cup: UMI vs 거울 없음", (20, 20), (18, 20)),
    ("Cup: 거울 반사 없음 vs 거울 없음", (17, 20), (18, 20)),
    ("Cup: UMI vs 거울 반사 없음", (20, 20), (17, 20)),
    ("Cup: 델타 vs 절대 행동", (16, 20), (5, 20)),
    ("Tossing: 지연 맞춤 vs 없음", (105, 120), (69, 120)),
    ("Cloth: UMI vs 상대 자세 없음", (14, 20), (6, 20)),
    ("Dish: UMI vs ResNet-34 (n=20 가정)", (14, 20), (0, 20)),
    ("Dish: UMI vs ResNet-34 (n=10 본문 가정)", (14, 20), (0, 10)),
    ("In-the-wild: 학습 컵 vs 미지 컵", (28, 40), (15, 20)),
]
res["fisher"] = []
for name, a, b in PAIRS:
    res["fisher"].append({"name": name, "a": a, "b": b,
                          "p_two_sided": float(f"{fisher(a[0], a[1], b[0], b[1]):.3g}")})

# ---------------------------------------------------------------- 3. 일관성 검사
checks = {}
# 3a. 객체별 값 평균이 전체와 같은지 (그림 8 토싱, 객체당 20회 가정)
ours_obj = np.array([0.80, 0.85, 0.85, 1.00, 0.90, 0.85])
noL_obj = np.array([0.70, 0.50, 0.70, 0.30, 0.65, 0.60])
checks["tossing_per_object_mean_ours"] = round(float(ours_obj.mean()), 4)
checks["tossing_per_object_mean_noLM"] = round(float(noL_obj.mean()), 4)
checks["tossing_object_counts_ours_x20"] = [round(x * 20) for x in ours_obj]
checks["tossing_object_counts_ours_sum"] = int(np.round(ours_obj * 20).sum())
checks["tossing_object_counts_noLM_x20"] = [round(x * 20) for x in noL_obj]
checks["tossing_object_counts_noLM_sum"] = int(np.round(noL_obj * 20).sum())
# 3b. 천 개 단위 합: 카페/분수 합
checks["inwild_cafe_sum"] = [16 + 6, 25 + 10]    # 22/35
checks["inwild_fountain_sum"] = [12 + 9, 15 + 10]  # 21/25
checks["inwild_total"] = [22 + 21, 35 + 25]        # 43/60
checks["inwild_train_total"] = [16 + 12, 25 + 15]  # 28/40
checks["inwild_unseen_total"] = [6 + 9, 10 + 10]   # 15/20
# 3c. 교차 로봇 및 본문 수치의 반올림
checks["intro_OOD_70pct_vs_pooled"] = round(43 / 60, 4)  # 소개문 70% vs 풀링 71.7%
checks["train_cup_rate_vs_intro"] = round(28 / 40, 4)
# 3d. 처리량 비율 (그림 11 텍스트 추출값 기준)
checks["throughput_cup_UMI_over_human"] = round(111 / 231, 4)   # 본문 48%
checks["throughput_cup_UMI_over_spacemouse"] = round(111 / 35, 3)  # 본문 >3배
checks["throughput_toss_UMI_over_human"] = round(149 / 237, 4)  # 본문 64%
# 3e. SLAM 오차 (그림 12): 개별 그리퍼 < 1 cm, 4°; 그리퍼 간 RPE 10.1 mm > 1 cm
checks["slam_per_gripper_ATE_mm"] = 6.1
checks["slam_inter_gripper_RPE_mm"] = 10.1
res["consistency"] = checks

# ---------------------------------------------------------------- 4. 검정력 참고 (V-A 대비, 20회)
# 20회에서 20/20 대 16/20(델타)을 보려면 필요한 표본 근사 (두 비율, 양측 0.05, 검정력 0.8)
def n_needed(p1, p2, alpha=0.05, power=0.8):
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (z_a * math.sqrt(2 * pbar * (1 - pbar)) + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)

res["power"] = {
    "n_per_arm_for_80_vs_100": n_needed(0.80, 1.00),
    "n_per_arm_for_0.70_vs_0.30": n_needed(0.70, 0.30),
    "n_per_arm_for_0.875_vs_0.575": n_needed(0.875, 0.575),
}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
for r in res["rates"]:
    print(f"{r['name']:<46} {r['k']:>3}/{r['n']:<3} {r['rate']:.3f}  Wilson95 {r['wilson95']}")
print()
for f in res["fisher"]:
    print(f"{f['name']:<42} p={f['p_two_sided']:.4g}")
print()
print(json.dumps(res["consistency"], ensure_ascii=False, indent=1))
print(json.dumps(res["power"], indent=1))
