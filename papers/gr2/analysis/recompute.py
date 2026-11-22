"""GR-2 (arXiv:2410.06158 v1) 논문의 수치를 다시 계산하고 일관성을 검사한다.

입력: 논문 본문에 적힌 값(출처는 주석의 절·그림 번호)과, 그림에서 눈으로 읽은 값(FIG 표시, ±2 pp 정도).
      논문에는 표가 없고, 그림 6·9·10·11의 막대와 점에 숫자가 적혀 있지 않은 것은 픽셀 높이로 읽었다.
      그림 10의 1–5 과제 성공률만 막대 위에 숫자로 적혀 있다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 1초 미만)

가정(논문에 없는 것은 표시한다)
- 시도 횟수(trial 수)는 논문에 없다. 모든 비율 검정은 설정별 시도 수 n을 20, 50, 100으로 바꿔 본다.
- CALVIN은 1000 사슬로 평가했다고 적혀 있다(3.3절). 성공률이 0.1% 단위로 적히고 모두 정수 k/1000으로 떨어진다.
  이것은 n = 1000과 맞는다는 뜻이지 n = 1000이라는 확인은 아니다.
- 그림에서 읽은 값은 본문의 같은 값과 비교해 보정 오차를 추정한다(§A).
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
RES = {}
Z = 1.959964


def wilson(k, n, z=Z):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def two_prop_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else float("nan")
    return z, float(2 * stats.norm.sf(abs(z)))


def newcombe(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def pct(x):
    return round(100 * x, 1)


# ---------------------------------------------------------------- A. 본문 값의 내부 일관성
print("A. 내부 일관성")
# K1. CALVIN (3.3절, 그림 10): 평균 길이 = 연속 1..5 과제 성공률의 합 (정의상)
calvin_gr2 = [98.6, 96.1, 93.1, 90.1, 85.9]          # 그림 10 막대 숫자 (본문 1·5 과제 값과 일치)
avg_len_gr2_reported = 4.64                           # 본문 3.3절
sum_gr2 = sum(calvin_gr2) / 100
RES["K1_calvin_avg_len"] = {
    "sum_of_rates": round(sum_gr2, 3),
    "reported": avg_len_gr2_reported,
    "match_within_0.005": abs(sum_gr2 - avg_len_gr2_reported) <= 0.005,
}
print("  K1 평균 길이 합 =", round(sum_gr2, 3), "보고값 4.64 ->", RES["K1_calvin_avg_len"]["match_within_0.005"])
# GR-1: 본문 4.21, 1과제 94.9, 5과제 73.1. 2–4과제 값은 그림 10에 숫자가 없다(§B에서 눈으로 읽은 값으로 본다).
gr1_p2_4_needed = 4.21 - 0.949 - 0.731
RES["K1_gr1_implied_mean_p2_4"] = round(100 * gr1_p2_4_needed / 3, 1)
print("  K1 GR-1 2–4과제 평균이 가져야 할 값 =", RES["K1_gr1_implied_mean_p2_4"], "%")

# K2. 다중 과제(3.1절, 그림 6): 'Unseen Average'는 세 일반화 설정의 단순 평균으로 보인다
unseen_gr2_400 = [71.4, 71.7, 55.8]                   # 본문 BG, Env, Manip
RES["K2_unseen_avg_gr2_400"] = round(sum(unseen_gr2_400) / 3, 2)
print("  K2 GR-2(400) Unseen 평균 =", RES["K2_unseen_avg_gr2_400"], "(그림 6 막대 읽기 65.6 근처)")
# GR-2 w/ DA: 본문 평균 74.7, Env 87.0 -> BG와 Manip의 합
da_sum_bg_manip = 3 * 74.7 - 87.0
RES["K2_da_bg_plus_manip"] = round(da_sum_bg_manip, 1)
print("  K2 DA에서 BG+Manip 합 =", RES["K2_da_bg_plus_manip"], "(그림 6 읽기 83.2 + 52.8 = 136.0)")

# K3. 데이터 산술 (1·2·3.1·3.2절)
RES["K3_data"] = {
    "multitask_trajs_per_task_from_40k": round(40000 / 105, 1),   # 본문 '평균 400'과 비교
    "multitask_trajs_per_task_from_105x400": 105 * 400,
    "one_eighth_of_40k_per_task": round(40000 / 8 / 105, 1),      # 본문 '약 50'
    "bin_pick_trajs_per_object": round(94000 / 55, 0),
    "tokens_per_clip_50B_over_38M": round(50e9 / 38e6, 0),
    "pretrain_scale_ratio_38M_over_0.8M": round(38 / 0.8, 1),    # GR-1 0.8M 대비 (본문 4.5절)
}
print("  K3", RES["K3_data"])

# K4. 파라미터 (2.1·3.5절): 기본 모델 총 230M, 학습 가능 95M. 3.5절 GR-2-B도 95M
RES["K4_params_note"] = ("기본 모델 230M 중 95M 학습 가능(3절). 3.5절은 같은 95M을 사전학습의 학습 가능 파라미터로 쓰고, "
                         "파인튜닝 때 사전학습 파라미터는 동결한다고 적었다. 파인튜닝의 학습 가능 수는 적히지 않았다.")

# ---------------------------------------------------------------- B. 그림에서 읽은 값과 본문 값 대조 (보정)
print("B. 그림 읽기 보정")
# 픽셀 읽기(render 110 dpi, bars2.py). 같은 막대의 본문 값과 비교해 편향을 본다.
fig6_read = {  # (GR-1, GR-2 50, GR-2 400, GR-2 w/DA)
    "Simple": (82.4, 72.8, 96.8, 90.4),
    "Distractor": (56.8, 57.6, 79.2, 80.0),
    "Unseen BG": (32.8, 43.2, 70.4, 83.2),
    "Unseen Env": (33.6, 57.6, 71.2, 86.4),
    "Unseen Manip": (48.0, 55.2, 55.2, 52.8),
    "Unseen Avg": (38.4, 52.0, 65.6, 73.6),
}
pairs = [("Simple GR-2 50 vs 본문 73.9", fig6_read["Simple"][1], 73.9),
         ("Simple GR-2 400 vs 본문 97.7", fig6_read["Simple"][2], 97.7),
         ("Unseen Env GR-2 400 vs 본문 71.7", fig6_read["Unseen Env"][2], 71.7),
         ("Unseen Env DA vs 본문 87.0", fig6_read["Unseen Env"][3], 87.0),
         ("Unseen Avg DA vs 본문 74.7", fig6_read["Unseen Avg"][3], 74.7)]
bias = [r - t for _, r, t in pairs]
RES["B1_fig6_reading_check"] = [{"item": n, "read": r, "text": t, "diff": round(r - t, 1)} for n, r, t in pairs]
RES["B1_fig6_mean_bias"] = round(float(np.mean(bias)), 2)
print("  B1 그림 6 읽기 편향(평균) =", RES["B1_fig6_mean_bias"], "pp")

fig9_read = {  # 막대 상단 픽셀 기준 (GR-1, GR-2)
    "Seen": (48.4, 80.4), "Unseen": (36.2, 76.3), "Cluttered Seen": (22.4, 74.8), "Cluttered Unseen": (24.0, 77.9)}
g1 = np.mean([v[0] for v in fig9_read.values()])
g2 = np.mean([v[1] for v in fig9_read.values()])
RES["B2_fig9_read_mean"] = {"GR-1": round(g1, 1), "GR-2": round(g2, 1), "text_GR-1": 33.3, "text_GR-2": 79.0}
print("  B2 그림 9 읽기 평균:", RES["B2_fig9_read_mean"])

# ---------------------------------------------------------------- C. 시도 수 n에 따른 불확실성
print("C. 시도 수에 따른 구간 (논문에 n 없음)")
ns = [20, 50, 100]
cmp_rows = []
def add(name, p1, p2, n):
    k1, k2 = round(p1 / 100 * n), round(p2 / 100 * n)
    d, lo, hi = newcombe(k1, n, k2, n)
    z, p = two_prop_z(k1, n, k2, n)
    cmp_rows.append({"comparison": name, "n_per_arm": n, "p1": p1, "p2": p2,
                     "diff_pp": pct(d), "newcombe95_pp": [pct(lo), pct(hi)], "z": round(z, 2), "p": round(p, 4)})

for n in ns:
    add("Unseen BG: GR-2(400) 71.4 vs GR-1 32.8(읽기)", 71.4, 32.8, n)
    add("Unseen Env: GR-2(400) 71.7 vs GR-1 33.6(읽기)", 71.7, 33.6, n)
    add("Unseen Manip: GR-2(400) 55.8 vs GR-1 48.0(읽기)", 55.8, 48.0, n)
    add("Unseen Manip: GR-2(50) 55.2 vs GR-2(400) 55.8(읽기)", 55.2, 55.8, n)
    add("Simple: GR-2(400) 97.7 vs DA 90.4(읽기)", 97.7, 90.4, n)
    add("Bin picking 전체 Seen: GR-2 80.4 vs GR-1 48.4(읽기)", 80.4, 48.4, n)
    add("Bin picking Seen 80 vs Unseen 76 (읽기, 본문 '비슷하다')", 80.4, 76.3, n)
RES["C1_multitask_bin_sensitivity"] = cmp_rows
for r in cmp_rows:
    print("  ", r["comparison"], "n=", r["n_per_arm"], "diff", r["diff_pp"], "CI", r["newcombe95_pp"], "p", r["p"])

# 단일 비율의 구간 (n = 20, 50, 100)
single = []
for n in ns:
    for name, p in [("GR-2(400) Unseen Manip 55.8", 55.8), ("GR-2(400) Unseen BG 71.4", 71.4), ("Bin GR-2 Avg 79.0", 79.0)]:
        k = round(p / 100 * n)
        lo, hi = wilson(k, n)
        single.append({"item": name, "n": n, "wilson95_pp": [pct(lo), pct(hi)]})
RES["C2_single_wilson"] = single

# CALVIN: n = 1000 가정 (0.1% 단위와 정수 일치)
calvin_rows = []
for k, (gr2, gr1) in enumerate([(98.6, 94.9), (96.1, None), (93.1, None), (90.1, None), (85.9, 73.1)], start=1):
    lo, hi = wilson(round(gr2 * 10), 1000)
    row = {"tasks_in_row": k, "GR-2_pct": gr2, "wilson95_pp": [pct(lo), pct(hi)]}
    if gr1 is not None:
        d, nlo, nhi = newcombe(round(gr2 * 10), 1000, round(gr1 * 10), 1000)
        z, p = two_prop_z(round(gr2 * 10), 1000, round(gr1 * 10), 1000)
        row.update({"GR-1_pct": gr1, "diff_pp": pct(d), "newcombe95_pp": [pct(nlo), pct(nhi)], "z": round(z, 2), "p": float(f"{p:.2e}")})
    calvin_rows.append(row)
RES["C3_calvin"] = calvin_rows
for r in calvin_rows:
    print("  CALVIN", r)

# ---------------------------------------------------------------- D. 규모 실험 (3.5절, 그림 11d)
print("D. 규모")
params_M = np.array([30, 95, 312, 719], dtype=float)
succ = np.array([38.0, 42.0, 57.0, 60.0])   # 그림 11(d) 막대 눈 읽기 ±3 pp, 설정 수와 오차막대 없음
slope, intercept = np.polyfit(np.log2(params_M), succ, 1)
RES["D1_scaling"] = {
    "params_M": params_M.tolist(), "succ_read_pct": succ.tolist(),
    "slope_pp_per_doubling": round(float(slope), 2),
    "doublings_S_to_XL": round(float(np.log2(719 / 30)), 2),
    "note": "네 점, 오차막대 없음, 읽기 오차 ±3 pp. 기울기는 참고용",
}
print("  D1", RES["D1_scaling"])

# ---------------------------------------------------------------- E. 정답 대비 시간이 든 항목 (video)
RES["E1_video_metrics"] = "본문에 FVD, PSNR, FID 등 정량 지표가 없음. 3.4절은 정성 시각화만 제시"

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", OUT)
