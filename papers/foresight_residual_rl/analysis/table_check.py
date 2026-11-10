"""Table I·II·III 재계산과 유의성 (CPU, 수 초).

논문 수치는 arXiv:2607.16506v1 Table I(예측기 정확도), Table II(성공률 mean ± sd, 4 seeds), Table III(예측기 점수)에서 옮겼다.
실행: python3 analysis/table_check.py  → analysis/table_check_results.json
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "table_check_results.json"
SEEDS = 4

# Table II: (mean, sd) 순서 = Grasp, Insert, Rotate, Grasp+Insert, Insert+Rotate, All
COLS = ["Grasp", "Insert", "Rotate", "Grasp+Insert", "Insert+Rotate", "All"]
T2 = {
    "pi0_e2e": [None, None, None, None, None, (54.9, 5.3)],
    "pi0_chained": [(87.1, 1.4), (45.7, 6.1), (93.4, 3.0), (37.1, 1.5), (61.5, 4.6), (41.4, 3.3)],
    "residual": [(98.4, 0.6), (92.2, 1.1), (99.2, 0.6), (55.5, 3.4), (83.4, 3.2), (54.5, 3.6)],
    "foresight": [(95.7, 3.3), (91.4, 0.6), (99.8, 0.4), (87.3, 1.3), (91.8, 3.3), (85.6, 3.9)],
}
# Table I: (정확도 %, 다수 클래스 기준 %)
T1 = {"Grasp->Insert": (86.2, 73.9), "Insert->Rotate": (93.9, 90.4)}
# Table III: 평균 예측기 점수 (Grasp terminal, Insert terminal)
T3 = {"pi0_base": (0.64, 0.97), "residual": (0.32, 0.86), "foresight": (0.73, 0.91)}


def welch(m1, s1, m2, s2, n=SEEDS):
    """두 평균의 Welch t 검정 (summary stats, 평균 기준). 반환: diff, t, df, p(양측), 95% CI"""
    r = stats.ttest_ind_from_stats(m1, s1, n, m2, s2, n, equal_var=False)
    v1, v2 = s1 ** 2 / n, s2 ** 2 / n
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n - 1) + v2 ** 2 / (n - 1))
    se = math.sqrt(v1 + v2)
    tc = stats.t.ppf(0.975, df)
    d = m1 - m2
    return {"diff_pp": round(d, 2), "t": round(float(r.statistic), 2), "df": round(df, 2),
            "p": float(f"{r.pvalue:.3g}"), "ci95": [round(d - tc * se, 2), round(d + tc * se, 2)]}


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(100 * (c - h), 2), round(100 * (c + h), 2)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


R = {}

# 1. 시드 간 sd 해석에 따른 Welch 검정 (foresight 대 residual, 열별)
rows = []
for j, col in enumerate(COLS):
    f, r = T2["foresight"][j], T2["residual"][j]
    # sd가 시드 표준편차(ddof=1)라고 볼 때
    a = welch(f[0], f[1], r[0], r[1])
    # ±가 표준오차(SEM)라고 볼 때: sd = SEM * sqrt(4)
    b = welch(f[0], f[1] * 2, r[0], r[1] * 2)
    rows.append({"column": col, "foresight": f, "residual": r, "as_sd": a, "as_sem": b})
pv = [x["as_sd"]["p"] for x in rows]
adj = holm(pv)
for x, pa in zip(rows, adj):
    x["holm_p_as_sd"] = float(f"{pa:.3g}")
R["T2_foresight_vs_residual"] = rows

# 2. 전체 과제: 시드 기준 Welch (foresight vs residual, foresight vs chained, residual vs e2e)
R["T2_all_task_comparisons"] = {
    "foresight_vs_residual_sd": welch(85.6, 3.9, 54.5, 3.6),
    "foresight_vs_residual_sem": welch(85.6, 3.9 * 2, 54.5, 3.6 * 2),
    "residual_vs_e2e_sd": welch(54.5, 3.6, 54.9, 5.3),
    "foresight_vs_chained_sd": welch(85.6, 3.9, 41.4, 3.3),
    "residual_vs_chained_sd": welch(54.5, 3.6, 41.4, 3.3),
}

# 3. 시드 수준 표준편차의 기대치: 시도 수에 따른 이항 sd와 관측 sd 비교
expl = {}
for name, p in [("foresight", 0.856), ("residual", 0.545), ("e2e", 0.549)]:
    expl[name] = {}
    for label, n in [("128_per_seed(512 total)", 128), ("512_per_seed(2048 total)", 512)]:
        expl[name][label] = round(100 * math.sqrt(p * (1 - p) / n), 2)
R["binomial_sd_expectations_pp"] = expl
R["observed_sd_pp"] = {"foresight": 3.9, "residual": 3.6, "e2e": 5.3}

# 4. Wilson 구간과 (의사복제) Fisher: 512회 대 2048회
wil = {}
for label, nn in [("n512", 512), ("n2048", 2048)]:
    wil[label] = {}
    for name, p in [("foresight", 85.6), ("residual", 54.5)]:
        k = round(p / 100 * nn)
        wil[label][name] = {"k": k, "wilson95": wilson(k, nn)}
    kf, kr = round(0.856 * (512 if label == "n512" else 2048)), round(0.545 * (512 if label == "n512" else 2048))
    nn_ = 512 if label == "n512" else 2048
    tab = [[kf, nn_ - kf], [kr, nn_ - kr]]
    wil[label]["fisher_pooled_pseudo_replicated_p"] = float(f"{stats.fisher_exact(tab)[1]:.3g}")
R["wilson_and_pooled_fisher"] = wil

# 5. 부분 과제의 곱 검사: 독립 가정이면 연속 성공 ≈ 개별 성공의 곱 (oracle 초기화 값 사용, 근사 확인)
prod = {}
for name in ["pi0_chained", "residual", "foresight"]:
    v = [x[0] / 100 for x in T2[name][:3]]
    prod[name] = {
        "G*I_product_pct": round(100 * v[0] * v[1], 1),
        "G+I_reported": T2[name][3][0],
        "I*R_product_pct": round(100 * v[1] * v[2], 1),
        "I+R_reported": T2[name][4][0],
        "G*I*R_product_pct": round(100 * v[0] * v[1] * v[2], 1),
        "All_reported": T2[name][5][0],
    }
R["subtask_product_check"] = prod

# 6. Table I: 검정 세트 크기를 가정(2560의 10% = 256, 민감도로 512)해 Wilson과 두 비율 차이 근사
t1 = {}
for key, (acc, base) in T1.items():
    t1[key] = {}
    for n in (256, 512):
        k = round(acc / 100 * n)
        kb = round(base / 100 * n)
        se = math.sqrt((acc / 100) * (1 - acc / 100) / n + (base / 100) * (1 - base / 100) / n)
        z = (acc - base) / 100 / se
        t1[key][f"n{n}"] = {"acc_wilson95": wilson(k, n), "base_wilson95": wilson(kb, n),
                            "diff_pp": round(acc - base, 2), "z_indep_approx": round(z, 2),
                            "p_indep_approx": float(f"{2 * stats.norm.sf(abs(z)):.3g}")}
R["table1_predictor_accuracy"] = t1
R["table1_assumption"] = "검정 세트 크기는 논문에 없음. N=2,560의 80/10/10 분할에서 10% = 256을 가정, 512는 민감도 점검. 두 비율은 같은 샘플이므로 독립 근사는 과대 추정 가능."

# 7. 기준 대비 상대 개선과 단위 확인
R["unit_check"] = {
    "all_task_diff_pp": round(85.6 - 54.5, 1),
    "all_task_rel_pct": round(100 * (85.6 / 54.5 - 1), 1),
    "grasp_insert_diff_pp": round(87.3 - 55.5, 1),
    "grasp_insert_rel_pct": round(100 * (87.3 / 55.5 - 1), 1),
}

# 8. Table III: 같은 예측기로 평가한 점수 (학습 목적함수와 동일 함수)
R["table3_scores"] = {k: {"grasp": v[0], "insert": v[1]} for k, v in T3.items()}
R["table3_residual_drop"] = {"grasp_drop_base_to_residual": round(T3["residual"][0] - T3["pi0_base"][0], 2),
                             "grasp_gain_foresight_over_residual": round(T3["foresight"][0] - T3["residual"][0], 2),
                             "insert_base_minus_foresight": round(T3["pi0_base"][1] - T3["foresight"][1], 2)}

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1))
if __name__ == "__main__":
    print(json.dumps(R, ensure_ascii=False, indent=1))
