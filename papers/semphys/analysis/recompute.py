"""SemPhys (arXiv:2611.24074 v1) 보고 수치 재계산과 일관성 검사.

실행: python3 analysis/recompute.py  → analysis/recompute_results.json
의존성: numpy, scipy (CPU, 1초 미만). 학습이나 재현은 하지 않는다.
입력은 논문 본문 표·문장에 적힌 값뿐이다. 원자료(클립별 점수)는 공개되지 않았다.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")
res = {}

# ---- 표 1 (BONES-SEED, 3개 생성 시드 평균 ± SD) ----
T1 = {  # method: (MM, r1, r2, r3, FID, Fail, motion, semantics, sd_r1)
    "TextOp":     dict(MM=1.279, r1=17.72, r2=26.73, r3=33.37, FID=0.3404, Fail=9.53,  motion=1.68, sem=1.40, sd_r1=0.19),
    "TeXedo":     dict(MM=1.250, r1=22.58, r2=30.27, r3=36.03, FID=0.3037, Fail=0.46,  motion=1.58, sem=1.40, sd_r1=0.13),
    "MoMask":     dict(MM=1.180, r1=36.61, r2=44.46, r3=48.99, FID=0.3521, Fail=1.81,  motion=1.84, sem=1.86, sd_r1=0.11),
    "ECHO":       dict(MM=1.177, r1=26.36, r2=39.36, r3=47.72, FID=0.2242, Fail=2.09,  motion=1.64, sem=1.60, sd_r1=0.23),
    "OMG":        dict(MM=1.060, r1=50.78, r2=64.67, r3=71.48, FID=0.1441, Fail=2.24,  motion=1.98, sem=1.82, sd_r1=0.21),
    "MotionGPT3": dict(MM=1.017, r1=58.35, r2=70.55, r3=75.88, FID=0.1640, Fail=19.65, motion=1.56, sem=1.72, sd_r1=0.14),
    "MDM":        dict(MM=1.010, r1=60.10, r2=72.24, r3=78.15, FID=0.1228, Fail=5.71,  motion=1.40, sem=1.82, sd_r1=0.29),
    "SemPhys":    dict(MM=0.843, r1=79.62, r2=89.27, r3=92.93, FID=0.0838, Fail=0.25,  motion=2.44, sem=2.68, sd_r1=0.06),
}
res["table1_as_read"] = T1

# 1) 본문 정량 주장과 표의 대조
checks = {}
def chk(name, claimed, computed, tol=0.01):
    checks[name] = dict(claimed=claimed, computed=round(computed, 4), ok=abs(claimed - computed) <= tol)

chk("SemPhys - MDM R@1 (본문 19.52 pp)", 19.52, T1["SemPhys"]["r1"] - T1["MDM"]["r1"])
chk("MotionGPT3 ref->exec drop, 본문 74.08->58.18 (15.90)", 15.90, 74.08 - 58.18)
chk("MotionGPT3 exec R@1 표 1 값 58.35 대 본문 58.18 (차이)", 0.17, T1["MotionGPT3"]["r1"] - 58.18, tol=0.005)
chk("SemPhys ref->exec drop (seed 42, 80.63->79.59)", 1.04, 80.63 - 79.59)
chk("Paired drop difference (15.90 - 1.04)", 14.86, 15.90 - 1.04)
chk("Anchoring gap start 74.61 -> 79.30 (4.69)", 4.69, 79.30 - 74.61)
chk("Shared start 76.71 -> 79.30 (2.59, 본문 약 +2.6)", 2.59, 79.30 - 76.71)
chk("Exec-ref-only gap 79.30 - 74.21 (5.09)", 5.09, 79.30 - 74.21)
chk("Matched control 79.31 - 73.31 (6.00)", 6.00, 79.31 - 73.31)
chk("Preference control 79.31 - 48.54 (30.77)", 30.77, 79.31 - 48.54)
chk("Window context 79.31 - 68.07 (11.24)", 11.24, 79.31 - 68.07)
chk("GT-trajectory relative change 56.93/79.30 - 1 (%) (-28.2)", -28.2, (56.93 / 79.30 - 1) * 100)
chk("MM H=32 - H=96 (0.0133)", 0.0133, 0.8453 - 0.8320, tol=0.00005)
chk("Replay sign agreement 102/112 (91.1%, 반올림 0.1)", 91.1, 100 * 102 / 112, tol=0.06)
chk("Replay improvement 107/127 (84.3%, 반올림 0.1)", 84.3, 100 * 107 / 127, tol=0.06)
chk("Perturbation mean MDM = 78.89 - 20.43", 58.46, 78.89 - 20.43)  # 부록 미포함, 역산값
res["consistency_checks"] = checks

# 2) 표 1 R@1의 불확실성: 시드 SD로 만든 Welch와 본문 구간 [11.99, 27.05] 비교
s, m = T1["SemPhys"], T1["MDM"]
n = 3
diff = s["r1"] - m["r1"]
se_seed = math.sqrt(s["sd_r1"] ** 2 / n + m["sd_r1"] ** 2 / n)
welch_df = (s["sd_r1"] ** 2 / n + m["sd_r1"] ** 2 / n) ** 2 / (
    (s["sd_r1"] ** 2 / n) ** 2 / (n - 1) + (m["sd_r1"] ** 2 / n) ** 2 / (n - 1))
t_crit = stats.t.ppf(0.975, welch_df)
res["seed_welch_R1_SemPhys_vs_MDM"] = dict(
    diff_pp=round(diff, 2), se_from_seed_sd=round(se_seed, 4), df=round(welch_df, 2),
    ci95_from_seed_sd=[round(diff - t_crit * se_seed, 2), round(diff + t_crit * se_seed, 2)],
    reported_ci=[11.99, 27.05],
    reported_half_width=round((27.05 - 11.99) / 2, 2),
    note="보고 구간은 시드 SD로 재현되지 않는다(폭이 약 11배). 구간의 출처(프롬프트 부트스트랩 등)는 본문에 없다.",
)
# 같은 구간 폭을 독립 이항 가정으로 만들려면 쿼리 수 n이 몇 개여야 하는가
p1, p2 = s["r1"] / 100, m["r1"] / 100
half = (27.05 - 11.99) / 2
z = 1.96
se_needed = (half / 100) / z
n_needed = (p1 * (1 - p1) + p2 * (1 - p2)) / se_needed ** 2
res["implied_queries_for_reported_ci_binomial"] = dict(
    se_needed_pp=round(se_needed * 100, 3), queries_per_system_needed=round(n_needed, 0),
    note="R@k의 쿼리 수는 본문에 없다(20개 풀 x 32 후보). 이 값은 독립 이항 가정의 대략적 역산이다.")

# 3) 실패율의 사건 수: 0.25‰ x 13,082 upright tasks
upright = 13082
def poisson_ci(k, alpha=0.05):
    lo = 0.0 if k == 0 else stats.chi2.ppf(alpha / 2, 2 * k) / 2
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2
    return lo, hi
fail_rows = {}
for name in ["SemPhys", "TeXedo", "MDM", "MotionGPT3", "MoMask"]:
    rate = T1[name]["Fail"] / 1000
    k = rate * upright
    lo, hi = poisson_ci(round(k))
    fail_rows[name] = dict(fail_per_thousand=T1[name]["Fail"], expected_events=round(k, 2),
                           exact_poisson95_events=[round(lo, 2), round(hi, 2)],
                           exact_poisson95_per_thousand=[round(lo / upright * 1000, 3), round(hi / upright * 1000, 3)])
res["fail_event_counts_mean_over_seeds"] = fail_rows
res["fail_note"] = ("SemPhys 0.25‰는 약 3건 사건입니다(13,082 upright tasks). 95% 구간은 약 0.05–0.67‰. "
                    "차이 검정은 사건 수가 적어 제한적입니다. 표 1의 Fail은 3개 시드 평균이라 정수가 아닐 수 있습니다.")

# 4) 세 번의 SemPhys 수치가 다른 프로토콜에서 나온다는 점
res["semphys_R1_by_protocol"] = {
    "표 1, 3시드 평균 (생성)": 79.62,
    "그림 3, 생성 시드 42 (실행)": 79.59,
    "그림 7, 학습 시드 3개 평균 (생성 시드 42)": 79.30,
    "표 2 / 그림 8, H=32 (학습 시드 3개)": 79.31,
    "표 2, H=64": 80.11,
}

# 5) 제거 실험 SD 기반 Welch (SD가 주어진 것만)
def welch(a, sa, b, sb, n=3):
    se = math.sqrt(sa ** 2 / n + sb ** 2 / n)
    df = (sa ** 2 / n + sb ** 2 / n) ** 2 / ((sa ** 2 / n) ** 2 / (n - 1) + (sb ** 2 / n) ** 2 / (n - 1))
    t = (a - b) / se
    p = 2 * stats.t.sf(abs(t), df)
    return dict(diff=round(a - b, 3), se=round(se, 3), t=round(t, 2), df=round(df, 2), p_welch=float(f"{p:.3g}"))
res["welch_ablations"] = {
    "anchoring: 79.30±0.16 vs no-anchor 74.61 (SD 6.35)": welch(79.30, 0.16, 74.61, 6.35),
    "exec vs generated-only 79.30±0.16 vs 74.21±0.40": welch(79.30, 0.16, 74.21, 0.40),
    "window 32-step ordered 79.31±0.16 vs endpoint padding 68.07±1.47": welch(79.31, 0.16, 68.07, 1.47),
    "H=32 vs H=64 R@1 79.31±0.16 vs 80.11±0.32": welch(79.31, 0.16, 80.11, 0.32),
    "H=32 vs H=96 R@1 79.31±0.16 vs 79.81±0.20": welch(79.31, 0.16, 79.81, 0.20),
    "H=32 vs H=96 MM 0.8453±0.0015 vs 0.8320±0.0002": welch(0.8453, 0.0015, 0.8320, 0.0002),
}


# 5b) 앵커링의 분산 감소 주장: F 검정 (n=3, df 2,2)
F = (6.35 / 0.16) ** 2
p_f = 2 * min(stats.f.cdf(F, 2, 2), stats.f.sf(F, 2, 2))
res["anchoring_variance_F"] = dict(F=round(F, 1), p_two_sided=float(f"{p_f:.3g}"),
                                   note="SD 6.35 대 0.16, 시드 3개. 분산 비교는 가능하지만 시드 3개라 검정력이 낮고, 평균 차이(4.69 pp)는 p≈0.33입니다.")


# 5c) 실패 건수 비교: 같은 분모(13,082)에서 두 방법의 사건 수 비율 검정 (이항 조건부, 양측)
from scipy.stats import binomtest
def fail_pair(a, b):
    ka = round(T1[a]["Fail"] / 1000 * upright)
    kb = round(T1[b]["Fail"] / 1000 * upright)
    r = binomtest(ka, ka + kb, 0.5, alternative="two-sided")
    return dict(k_a=ka, k_b=kb, p_binom=float(f"{r.pvalue:.3g}"))
res["fail_pairwise_binomial"] = {"SemPhys vs TeXedo": fail_pair("SemPhys", "TeXedo"),
                                  "SemPhys vs MDM": fail_pair("SemPhys", "MDM"),
                                  "SemPhys vs MoMask": fail_pair("SemPhys", "MoMask")}
# 5d) 쿼리 640개(20풀 x 32)로 가정했을 때 독립 이항 95% 반폭 (R@1, 참고용)
q = 640
se_q = math.sqrt(p1 * (1 - p1) / q + p2 * (1 - p2) / q)
res["pointwise_halfwidth_if_640_queries_pp"] = round(100 * 1.96 * se_q, 2)

# 6) 재현 검사 Wilson 구간 (그림 9)
def wilson(k, n_, z=1.96):
    p = k / n_
    den = 1 + z * z / n_
    c = (p + z * z / (2 * n_)) / den
    h = z * math.sqrt(p * (1 - p) / n_ + z * z / (4 * n_ * n_)) / den
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]
res["fig9_wilson95"] = dict(sign_agreement=dict(k=102, n=112, wilson=wilson(102, 112)),
                            mm_improve=dict(k=107, n=127, wilson=wilson(107, 127)))
res["fig9_note"] = "무작위 섭동 기준 48.8%는 표본 수가 없어 구간을 못 만듭니다. 그림 9 막대 구간은 프롬프트 부트스트랩입니다."

# 7) 채택 수준: 무작위 추측 기준선
res["chance_R1_32_candidates_pct"] = round(100 / 32, 3)

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: res[k] for k in ["seed_welch_R1_SemPhys_vs_MDM", "implied_queries_for_reported_ci_binomial"]},
                 ensure_ascii=False, indent=2))
print("consistency:")
for k, v in checks.items():
    print(("OK  " if v["ok"] else "DIFF"), k, v)
print("fail:", json.dumps(fail_rows, ensure_ascii=False))
print("welch:", json.dumps(res["welch_ablations"], ensure_ascii=False, indent=1))
print("fig9:", res["fig9_wilson95"])
