"""논문 표 1–5의 수치를 다시 계산하고 일관성과 검정을 본다 (CPU, 수 초).

실행: python3 analysis/recompute.py  → analysis/recompute_results.json
입력: 논문 표 값(arXiv:2601.16163 v1)을 아래에 옮겨 적었다. 원자료는 공개되지 않았다.
"""
import json
import math
from pathlib import Path
from statistics import mean

OUT = Path(__file__).resolve().parent / "recompute_results.json"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def two_prop_z(p1, n1, p2, n2):
    """두 비율 차이의 양측 z 검정 (풀링 분산)."""
    pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p


def achievable(pct, n):
    """표의 백분율(소수 첫째 자리)을 만들 수 있는 정수 성공 수 k가 n개 시도에 있는지."""
    ks = [k for k in range(n + 1) if round(100 * k / n, 1) == round(pct, 1)]
    return ks[0] if ks else None


res = {}

# ---- 1. 표 1 LIBERO: 평균 재계산과 시도 수 정합성 ----
libero = {"Spatial": 98.1, "Object": 100.0, "Goal": 98.2, "Long": 97.6}
res["libero_avg_recomputed"] = round(mean(libero.values()), 3)  # 보고 98.5
# 표 캡션: 스위트당 500 trial x 3 seed = 1500 시도 (6000 총합)
N_LIBERO = 1500
res["libero_achievable_k"] = {s: achievable(v, N_LIBERO) for s, v in libero.items()}
res["libero_wilson95"] = {
    s: [round(100 * x, 2) for x in wilson(res["libero_achievable_k"][s], N_LIBERO)] for s in libero
}

# ---- 2. 표 2 RoboCasa: 67.1 (3600 시도) 과 근접 기준선 차이 검정 ----
N_ROBO = 3600
robo_base = {"FLARE (Zheng 2025)": 66.4, "GR00T-N1.5+HAMLET (Koo 2025)": 66.4,
             "Video Policy (Liang 2025)": 66.0, "GR00T-N1.5 (Bjorck 2025)": 64.1}
res["robocasa_cosmos_wilson95"] = [round(100 * x, 2) for x in wilson(round(0.671 * N_ROBO), N_ROBO)]
res["robocasa_vs_baselines"] = {}
for name, v in robo_base.items():
    # 기준선의 시도 수는 논문에 없다. 같은 3600 시도로 가정한 근사다.
    z, p = two_prop_z(0.671, N_ROBO, v / 100, N_ROBO)
    res["robocasa_vs_baselines"][name] = {"diff_pts": round(67.1 - v, 2), "z": round(z, 2), "p_two_sided": round(p, 4)}

# 표 2 평균 재계산 (24 과제 평균)
res["robocasa_mean_check"] = "논문 본문 67.1은 24 과제 평균 (표 2에 과제별 값 없음, 재계산 불가)"

# ---- 3. 표 3 ALOHA: 점수 평균과 시도 가중 평균 재계산 ----
# 과제별 시도 수: plate 30(ID20+OOD10), fold 20(12+8), candies 25(15+10), ziploc 26(20+6)
trials = {"plate": 30, "fold": 20, "candies": 25, "ziploc": 26}
aloha_full = {
    "Diffusion Policy": [63.3, 23.5, 32.8, 14.6],
    "OpenVLA-OFT+": [68.3, 99.5, 21.6, 58.5],
    "pi0": [85.0, 98.5, 71.2, 56.9],
    "pi0.5": [98.3, 99.5, 95.2, 61.5],
    "Cosmos Policy": [100.0, 99.5, 89.6, 85.4],
}
aloha_reported_avg_full = {"Diffusion Policy": 33.6, "OpenVLA-OFT+": 62.0, "pi0": 77.9, "pi0.5": 88.6, "Cosmos Policy": 93.6}
res["aloha_avg_check"] = {}
for m, v in aloha_full.items():
    simple = mean(v)
    weighted = sum(x * n for x, n in zip(v, trials.values())) / sum(trials.values())
    res["aloha_avg_check"][m] = {"simple_mean": round(simple, 2), "trial_weighted": round(weighted, 2),
                                 "reported": aloha_reported_avg_full[m]}

# 표 3 ID/OOD 값 일관성 (ID 67, OOD 34 시도; 과제별 ID+OOD 합산이 full과 맞는지)
id_n = {"plate": 20, "fold": 12, "candies": 15, "ziploc": 20}
ood_n = {"plate": 10, "fold": 8, "candies": 10, "ziploc": 6}
cosmos_id = {"plate": 100.0, "fold": 99.2, "candies": 100.0, "ziploc": 86.0}
cosmos_ood = {"plate": 100.0, "fold": 100.0, "candies": 74.0, "ziploc": 83.3}
full_check = {k: round((cosmos_id[k] * id_n[k] + cosmos_ood[k] * ood_n[k]) / (id_n[k] + ood_n[k]), 2)
              for k in id_n}
res["aloha_cosmos_full_from_id_ood"] = full_check  # 보고 100, 99.5, 89.6, 85.4
res["aloha_total_trials"] = sum(trials.values())  # 101

# 점수는 이진이 아니다. 과제 점수의 시도별 분산 없이 이항 검정은 할 수 없다.
res["aloha_note"] = "점수(0–100) 평균만 보고되어 이항 검정, 신뢰구간을 계산할 수 없다. 차이는 점 추정만 적는다."
res["aloha_pi05_vs_cosmos_avg_diff"] = round(93.6 - 88.6, 2)
res["aloha_ood_pi05_minus_cosmos"] = round(92.5 - 89.3, 2)

# ---- 4. 표 4·5 절제 ----
abl4 = {"full": 98.5, "w/o aux": 97.0, "scratch": 94.6}
res["table4_drops"] = {k: round(98.5 - v, 2) for k, v in abl4.items() if k != "full"}  # 1.5, 3.9
res["table4_avg_check"] = {"w/o aux": round(mean([97.6, 99.8, 96.7, 94.0]), 3),
                           "scratch": round(mean([94.7, 98.9, 96.3, 88.6]), 3)}
res["table5_1step_drop_text_vs_table"] = {"text_says": 0.5, "table_gives": round(67.1 - 66.4, 2)}
res["table5_latency_ratio"] = round(0.61 / 0.16, 2)  # 본문 "nearly 4x"

# ---- 5. 부록 A.2 배치와 GPU 수 ----
res["batch_check"] = {"libero": 64 * 30, "robocasa": 32 * 25, "aloha": 8 * 25}  # 1920, 800, 200 (보고와 일치)
res["aloha_demo_count"] = 80 + 15 + 45 + 45  # 185
res["rollout_count"] = 505 + 143  # 648

# ---- 6. 계획(best-of-N) 연산 수 ----
res["planning_value_queries_per_action"] = 3 * 5  # WM 3회 x 가치 5회 = 15 (본문과 일치)

OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=2, ensure_ascii=False))
