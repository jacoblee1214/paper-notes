"""DAPPT 보고 수치 재계산과 표 사이 일관성 검사 (numpy, scipy). 결과는 recompute_results.json.

입력은 논문 표 값 그대로다(출처: paper/paper_raw.txt, 표 I–IV, §V-B, 부록 A.1).
실행: python3 analysis/recompute.py
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
res = {}

# 표 IV: 시뮬레이션 평균(Genesis Avg)과 최선(Genesis Best), 작업 지표
# 지표 방향: Scrape는 남은 비율(낮을수록 좋음), 나머지는 높을수록 좋음
T4 = {
    "Scoop":        {"avg": 3.64, "best": 13.9, "dir": "higher"},
    "Rod-Roll":     {"avg": 1.55, "best": 2.85, "dir": "higher"},
    "Scrape":       {"avg": 33.0, "best": 17.0, "dir": "lower"},
    "Mouse-Wheel":  {"avg": 0.23, "best": 0.37, "dir": "higher"},
    "Dispenser":    {"avg": 0.97e3, "best": 2.26e3, "dir": "higher"},
    "Spray":        {"avg": 5.64e3, "best": 9.12e3, "dir": "higher"},
}
ratios = {}
for k, v in T4.items():
    # 개선 배율: 높을수록 좋은 지표는 best/avg, 낮을수록 좋은 지표는 avg/best
    r = v["best"] / v["avg"] if v["dir"] == "higher" else v["avg"] / v["best"]
    pct = (v["best"] - v["avg"]) / v["avg"] * 100  # 그림 3의 "차이 %"(부호 그대로)
    ratios[k] = {"factor": round(r, 3), "pct_diff_best_vs_avg": round(pct, 1)}
res["table4_improvement_factor"] = ratios
fac = [v["factor"] for v in ratios.values()]
res["claim_1_5_to_3_8_check"] = {
    "reported_range": [1.5, 3.8],
    "computed_min": min(fac), "computed_max": max(fac),
    "min_task": min(ratios, key=lambda k: ratios[k]["factor"]),
    "note": "상한 3.8은 Scoop(3.82)과 일치. 하한 1.5는 어떤 과제도 만들지 않음(최솟값 Mouse-Wheel 1.61).",
}

# 표 I: Genesis Best 대 SimToolReal (sim-to-sim 기준선)
T1 = {"Scrape (남은 %, 낮을수록 좋음)": (17, 95), "Rod-Roll (cm)": (2.85, 0),
      "Scoop (g)": (13.9, 9.2), "Mouse-Wheel (rad)": (0.37, 0.19),
      "Dispenser (#)": (2.26e3, 0), "Spray (#)": (9.12e3, 0)}
res["table1_vs_simtoolreal"] = {
    k: {"ours_best": a, "simtoolreal": b,
        "ratio_ours_over_base": (round(a / b, 3) if b else "기준선 0이라 비율 없음")}
    for k, (a, b) in T1.items()}
# 표 I과 표 IV의 Best 열 일치 여부
res["table1_vs_table4_best_match"] = {
    "Scoop": (13.9, T4["Scoop"]["best"]), "Rod-Roll": (2.85, T4["Rod-Roll"]["best"]),
    "Scrape": (17, T4["Scrape"]["best"]), "Mouse-Wheel": (0.37, T4["Mouse-Wheel"]["best"]),
    "Dispenser": (2.26e3, T4["Dispenser"]["best"]), "Spray": (9.12e3, T4["Spray"]["best"])}

# 표 II: 도메인 조건 대 고정 도메인(Genesis Best 파라미터 하나로 학습)
T2 = {"Mouse-Wheel": (0.37, 0.19), "Dispenser": (2.26e3, 0.96e3), "Scoop": (13.9, 0.32)}
res["table2_drop_pct"] = {k: {"computed": round((a - b) / a * 100, 2)} for k, (a, b) in T2.items()}
res["table2_reported_drop_pct"] = {"Mouse-Wheel": 48.6, "Dispenser": 57.5, "Scoop": 97.7}
res["table2_note"] = ("Mouse-Wheel 고정 도메인 값 0.19 rad가 표 I의 SimToolReal 값 0.19 rad와 같습니다. "
                      "우연일 수도 있고 복사일 수도 있어 저자 확인이 필요합니다.")

# 표 III: 실로봇 20회 시행, 성공 횟수. Wilson 95% 구간
def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

T3 = {"Scooping": (20, 14, 7.7), "Rod-Rolling": (16, 11, 2.7), "Scraping": (19, 19, 29),
      "Mouse-Wheel": (19, 19, 0.33), "Dispenser": (18, 14, None), "Spray": (16, 10, None)}
real = {}
for k, (mot, task, metric) in T3.items():
    n = 20
    lo, hi = wilson(task, n)
    mlo, mhi = wilson(mot, n)
    # 정확 이항 검정: 성공률이 0.5보다 큰가(기준 0.5는 임의 값, 비교 대상 없음)
    p_half = stats.binomtest(task, n, 0.5, alternative="greater").pvalue
    real[k] = {"motion": f"{mot}/20 = {mot/n*100:.0f}%", "motion_wilson95": [round(mlo*100, 1), round(mhi*100, 1)],
               "task": f"{task}/20 = {task/n*100:.0f}%", "task_wilson95": [round(lo*100, 1), round(hi*100, 1)],
               "task_binom_p_vs_0.5": round(p_half, 4), "metric": metric}
res["table3_real_world"] = real
res["table3_note"] = ("기준선(실로봇 SimToolReal 등)이 없어 두 집단 검정은 불가합니다. 작업 지표 평균의 SD와 "
                      "시행별 값이 없어 평균의 신뢰구간도 못 만듭니다. 0.5 기준은 임의입니다.")
res["table3_metric_ci"] = "불가: SD 또는 시행별 값 없음"

# 부록 A.1 및 본문: 표 III의 "시도 횟수가 20회 고정" 가정. 표 III 전체가 20회라는 근거는 본문 §V-B.
# 시뮬레이션: 에피소드 수 미기재, 따라서 신뢰구간 불가.
res["sim_ci"] = "불가: 에피소드 수, 시드 수, 분산 모두 미기재"

# 실로봇 Welch 등 비교는 기준선 없음으로 건너뜀
# 그림 3: 차이 %의 부호(높을수록 좋은 지표는 양수가 좋음)

json.dump(res, OUT.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps({"factors": {k: v["factor"] for k, v in ratios.items()},
                  "claim": res["claim_1_5_to_3_8_check"]["computed_min"],
                  "table2": res["table2_drop_pct"]}, ensure_ascii=False, indent=1))
print("wrote", OUT)
