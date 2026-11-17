"""Pixel2Catch (arXiv:2602.22733 v1) 표 수치를 다시 계산하고 검정한다.

입력: 논문 Table II(시뮬), Table III(실로봇)의 값을 아래에 그대로 옮겨 적었다(출처 표시).
      프로젝트 페이지(seongdrgn.github.io/pixel2catch) 수치는 참고용으로만 따로 적었다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 실로봇 비율의 분모는 물체당 30회다(Table III 캡션). 퍼센트를 k/30으로 바꿀 때 가장 가까운 정수 k를 쓴다.
- 시뮬 "±"는 3 seed의 표준편차(ddof=1)로 가정한다. 논문은 SD인지 SE인지 밝히지 않는다.
- 두 정책 비교의 검정은 독립 표본으로 본다(같은 물체와 같은 던지기 조건을 공유하므로 실제로는 짝지은 설계일 수 있다).
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
    """Newcombe(1998) 방법 10: 두 비율 차이의 Wilson 기반 구간."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


# ---------- 1. 실로봇 Table III (arXiv v1), 물체당 30회 ----------
N_REAL = 30
REAL = {  # 방법: {지표: [Cube, L-block, Triangle]} (%)
    "only-WH":     {"TR": [0, 0, 0],     "SR": [0, 0, 0]},
    "only-Center": {"TR": [57, 54, 54],  "SR": [13, 3, 23]},
    "S-A RL":      {"TR": [50, 43, 47],  "SR": [33, 20, 20]},
    "Proposed":    {"TR": [73, 73, 70],  "SR": [63, 43, 43]},
}

# 1-1. 퍼센트가 30회의 정수 비율과 맞는지 검사
mismatch = []
counts = {}
for m, d in REAL.items():
    for metric, vals in d.items():
        ks = []
        for obj, v in zip(["Cube", "L-block", "Triangle"], vals):
            k = round(v * N_REAL / 100)
            back = 100 * k / N_REAL
            ks.append(k)
            if abs(back - v) >= 0.5:  # 반올림 범위를 벗어남
                mismatch.append({"method": m, "metric": metric, "object": obj,
                                 "reported": v, "nearest_k_over_30": round(back, 1), "k": k})
        counts[(m, metric)] = ks
RES["real_percent_not_k_over_30"] = mismatch

# 1-2. 물체 합산(n=90) 비율과 Wilson 구간, 두 방법의 차이 검정
pooled = {}
for m in REAL:
    for metric in ["TR", "SR"]:
        k = sum(counts[(m, metric)])
        pooled[(m, metric)] = k
        lo, hi = wilson(k, 3 * N_REAL)
        pooled.setdefault("ci", {})[f"{m}|{metric}"] = [round(100 * k / 90, 1), round(100 * lo, 1), round(100 * hi, 1), k]

tests = {}
for other in ["S-A RL", "only-Center"]:
    for metric in ["TR", "SR"]:
        k1, n1 = pooled[("Proposed", metric)], 90
        k2, n2 = pooled[(other, metric)], 90
        table = np.array([[k1, n1 - k1], [k2, n2 - k2]])
        _, p_chi = stats.chi2_contingency(table, correction=False)[:2]
        _, p_fis = stats.fisher_exact(table)
        d, lo, hi = newcombe_diff(k1, n1, k2, n2)
        tests[f"Proposed vs {other} | {metric}"] = {
            "diff_pp": round(100 * d, 1), "newcombe95_pp": [round(100 * lo, 1), round(100 * hi, 1)],
            "p_fisher": float(f"{p_fis:.2g}"), "p_chi2": float(f"{p_chi:.2g}")}

# 1-3. 물체별 Proposed 대 S-A RL의 S.R. Fisher 검정 (30 vs 30)
per_obj = {}
for i, obj in enumerate(["Cube", "L-block", "Triangle"]):
    k1 = counts[("Proposed", "SR")][i]
    k2 = counts[("S-A RL", "SR")][i]
    _, p = stats.fisher_exact([[k1, N_REAL - k1], [k2, N_REAL - k2]])
    per_obj[obj] = {"proposed_k": k1, "sa_rl_k": k2, "p_fisher": float(f"{p:.2g}")}
RES["real_pooled_ci_pct_k"] = pooled.get("ci")
RES["real_tests_pooled"] = tests
RES["real_per_object_SR_proposed_vs_SA"] = per_obj

# 1-4. 본문 문장과 표 평균의 일치
RES["text_check"] = {
    "proposed_TR_mean_vs_text_approx70": round(float(np.mean(REAL["Proposed"]["TR"])), 1),
    "proposed_SR_mean_vs_text_approx50": round(float(np.mean(REAL["Proposed"]["SR"])), 1),
    "sa_rl_SR_mean_vs_text_approx24": round(float(np.mean(REAL["S-A RL"]["SR"])), 1),
    "only_center_TR_mean": round(float(np.mean(REAL["only-Center"]["TR"])), 1),
}

# ---------- 2. 시뮬 Table II (3 seed 평균 ± SD) ----------
SIM = {  # {객체 집합: {방법: (mean, sd)}}
    "seen": {
        "TR": {"w/o-PF": (12.13, 1.27), "only-WH": (8.03, 0.6), "only-Center": (87.07, 1.62),
               "S-A RL": (78.2, 1.32), "Proposed": (89.97, 0.21)},
        "SR": {"w/o-PF": (8.93, 1.04), "only-WH": (5.53, 0.51), "only-Center": (81.27, 0.76),
               "S-A RL": (63.5, 0.85), "Proposed": (84.13, 0.50)},
    },
    "unseen": {
        "TR": {"w/o-PF": (12.11, 1.9), "only-WH": (6.11, 0.35), "only-Center": (87.5, 1.53),
               "S-A RL": (75.83, 2.0), "Proposed": (89.28, 0.79)},
        "SR": {"w/o-PF": (7.89, 1.46), "only-WH": (4.0, 0.17), "only-Center": (80.72, 0.38),
               "S-A RL": (65.44, 2.25), "Proposed": (84.83, 1.17)},
    },
}
N_SEED = 3
sim_tests = {}
for split, d in SIM.items():
    for metric, dd in d.items():
        for other in ["S-A RL", "only-Center", "w/o-PF"]:
            m1, s1 = dd["Proposed"]
            m2, s2 = dd[other]
            t, p = stats.ttest_ind_from_stats(m1, s1, N_SEED, m2, s2, N_SEED, equal_var=False)
            # Welch 자유도와 95% 구간
            se = math.sqrt(s1 ** 2 / N_SEED + s2 ** 2 / N_SEED)
            v = se ** 4 / ((s1 ** 2 / N_SEED) ** 2 / (N_SEED - 1) + (s2 ** 2 / N_SEED) ** 2 / (N_SEED - 1))
            tc = stats.t.ppf(0.975, v)
            diff = m1 - m2
            sim_tests[f"{split}|{metric}|Proposed vs {other}"] = {
                "diff_pp": round(diff, 2), "ci95": [round(diff - tc * se, 2), round(diff + tc * se, 2)],
                "welch_p": float(f"{p:.2g}")}
RES["sim_tests"] = sim_tests

# 2-1. 시뮬 3 seed 검정력: 차이가 0.5pp일 때 필요한 seed 수(SD 0.5 가정)
RES["sim_note"] = ("seed 3개, SD 0.2~2.3 pp. 같은 평가 설정이면 차이 1 pp 이상은 SD가 0.5일 때 검정이 가능하지만 "
                   "seed 간 변동이 SD로 충분히 잡혔는지는 3개로 알 수 없다.")

# ---------- 3. 프로젝트 페이지 수치 (논문 버전 확인용, 별도 표시) ----------
PAGE_REAL = {  # 페이지: 360회/정책 (3 물체 x 2 던지는 사람 x 2 배경 x 30회), 평균 ± SD
    "Proposed": {"TR": [77.93, 76.60, 74.67], "SR": [60.60, 51.27, 46.60]},
    "S-A RL": {"TR": [64.67, 57.27, 61.40], "SR": [31.27, 34.00, 30.00]},
    "Only-Center": {"TR": [44.73, 44.80, 47.47], "SR": [15.27, 10.60, 17.27]},
}
RES["page_real_means"] = {m: {k: round(float(np.mean(v)), 1) for k, v in d.items()} for m, d in PAGE_REAL.items()}
RES["page_vs_arxiv_v1_SR_mean_diff"] = round(float(np.mean(PAGE_REAL["Proposed"]["SR"]) - np.mean(REAL["Proposed"]["SR"])), 1)
# 360 x 0.606 = 218.2 : 정수 개수가 아니므로 페이지 수치는 하위 집단 평균으로 보인다
RES["page_360_nonint_check"] = {"proposed_SR_cube_x360": round(360 * 0.606, 2)}

# ---------- 2-2. 보고된 ± 값과 이항 표준오차 비교 (seed당 1000회 가정) ----------
# 한 seed의 성공률 추정치에는 평가 횟수만큼의 이항 잡음이 있다. 보고된 SD가 이보다 작으면
# "SD가 seed 간 변동인지", "평가 횟수가 다른지"를 확인해야 한다.
binom = {}
for split, d in SIM.items():
    for metric, dd in d.items():
        for m, (mu, sd) in dd.items():
            p_ = mu / 100
            se_pp = 100 * math.sqrt(p_ * (1 - p_) / 1000)
            binom[f"{split}|{metric}|{m}"] = {"reported_sd": sd, "binomial_se_1000": round(se_pp, 2),
                                              "sd_below_binomial": bool(sd < se_pp)}
RES["sim_sd_vs_binomial"] = binom
RES["sim_sd_below_binomial_count"] = sum(v["sd_below_binomial"] for v in binom.values())
RES["sim_sd_total"] = len(binom)

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
if __name__ == "__main__":
    print(json.dumps(RES, ensure_ascii=False, indent=1, default=str))
