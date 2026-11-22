"""OmniH2O (arXiv:2406.08858 v1 = CoRL 2024 PMLR 270) 논문 표 수치를 다시 계산하고 통계 검정을 한다.

입력: 논문 본문과 부록 표의 값을 아래에 그대로 옮겨 적었다(출처 표기: 표 번호). 외부 데이터는 쓰지 않는다.
출력: analysis/recompute_results.json, 표준출력 요약.
실행: python3 analysis/recompute.py   (numpy, scipy, 수 초)

가정(논문에 없는 것은 표시한다)
- 표 1의 시퀀스 수는 "14k"(본문 §4.1)로만 적혀 있다. 정확한 수를 모르므로 N = 14000을 기준으로 하고 13000, 15000으로 민감도를 본다.
- 표 2는 실로봇 standing 시퀀스 20개다. 표준편차를 보고하지 않으므로 SD = CV × 평균으로 두고 CV별 결과를 낸다.
- 표 3은 표 17(작업별 10회)의 평균으로 다시 계산한다. 작업 네 개 × 10회 = 40회를 독립으로 보는 것은 근사다.
  작업별 층화 검정(Cochran–Mantel–Haenszel)도 같이 낸다.
- 시드 수는 어느 표에도 없다. 모든 비교는 학습 한 번(단일 시드)일 수 있다는 전제를 깐다.
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
    """Newcombe (1998) 방법 10: 두 비율 차이의 Wilson 기반 95% 구간."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="two-sided")[1])


def cmh(strata):
    """strata: [(a, n1, b, n2), ...] 층별 2x2. 연속성 보정 없는 Mantel–Haenszel 카이제곱 검정(자유도 1)."""
    num, var = 0.0, 0.0
    for a, n1, b, n2 in strata:
        m = a + b
        N = n1 + n2
        num += a - n1 * m / N
        var += n1 * n2 * m * (N - m) / (N * N * (N - 1))
    chi2 = num * num / var
    return float(chi2), float(stats.chi2.sf(chi2, 1))


def welch_from_cv(m1, m2, n1, n2, cv):
    """평균 m1, m2, 두 그룹 SD = cv × 평균(가정). Welch t와 양측 p."""
    s1, s2 = cv * m1, cv * m2
    r = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
    return float(r.statistic), float(r.pvalue)


# ---------------- 표 1: 시뮬레이션 성공률 (Succ) ----------------
T1 = {  # 이름: (Succ %, 평가식이 가능한 All-seq Eg-mpjpe, 성공 시퀀스 Eg-mpjpe)
    "Privileged policy": (94.77, 126.51, 122.71),
    "H2O [3]": (87.52, 148.13, 133.28),
    "OmniH2O": (94.10, 141.11, 135.49),
    "OmniH2O-w/o-DAgger-History0": (90.62, 163.44, 153.31),
    "OmniH2O-w/o-DAgger": (47.11, 223.27, 182.13),
    "OmniH2O-History0": (93.80, 141.21, 134.90),
    "OmniH2O-History50": (93.56, 141.51, 135.04),
    "OmniH2O-History5": (93.60, 139.23, 132.67),
    "OmniH2O-GRU": (92.85, 147.67, 142.75),
    "OmniH2O-LSTM": (91.03, 147.36, 142.64),
    "OmniH2O-22points": (94.72, 127.71, 123.87),
    "OmniH2O-8points": (94.31, 129.30, 125.14),
    "OmniH2O-w-linvel": (93.80, 138.18, 132.44),
}
N_SEQ = {"base": 14000, "lo": 13000, "hi": 15000}

t1 = {}
for name, (succ, _, _) in T1.items():
    t1[name] = succ
res_t1 = []
ref = "OmniH2O"
for other, n_label in [("H2O [3]", "vs H2O"), ("Privileged policy", "vs 교사(privileged)"),
                       ("OmniH2O-w/o-DAgger", "vs RL만(w/o DAgger)"), ("OmniH2O-History0", "vs History0"),
                       ("OmniH2O-GRU", "vs GRU"), ("OmniH2O-LSTM", "vs LSTM"),
                       ("OmniH2O-22points", "vs 22점"), ("OmniH2O-w-linvel", "vs 선속도 입력")]:
    row = {"comparison": f"OmniH2O {n_label}", "diff_pp_reported": round(T1[ref][0] - T1[other][0], 2)}
    for key, n in N_SEQ.items():
        k1 = round(T1[ref][0] / 100 * n)
        k2 = round(T1[other][0] / 100 * n)
        d, lo, hi = newcombe_diff(k1, n, k2, n)
        row[f"newcombe95_pp_N{key}"] = [round(100 * lo, 2), round(100 * hi, 2)]
        row[f"fisher_p_N{key}"] = float(f"{fisher(k1, n, k2, n):.3g}")
    res_t1.append(row)
RES["T1_sim_success"] = {"assumed_N": N_SEQ, "rows": res_t1}

# 실패 시퀀스 평균의 역산 (표 1 All = s·Succ평균 + (1-s)·실패평균). 값이 양수이고 성공 평균보다 크면 정합.
implied = {}
for name in ["Privileged policy", "H2O [3]", "OmniH2O", "OmniH2O-History0", "OmniH2O-22points"]:
    s = T1[name][0] / 100
    all_, succ_ = T1[name][1], T1[name][2]
    fail = (all_ - s * succ_) / (1 - s)
    implied[name] = round(fail, 1)
RES["T1_implied_failed_mean_eg_mpjpe"] = implied

# ---------------- 표 2: 실로봇 standing 20 시퀀스 ----------------
T2 = {  # Eg-mpjpe, Empjpe, Eacc, Evel
    "H2O": 87.33, "OmniH2O": 47.94, "MLP+linvel": 50.93, "GRU+linvel": 49.75,
    "History0": 83.26, "History5": 62.18, "History50": 50.24, "LSTM": 87.00,
}
N_REAL = 20
cvs = [0.1, 0.2, 0.3, 0.5]
real = []
for other in ["H2O", "MLP+linvel", "GRU+linvel", "History0", "History50", "History5", "LSTM"]:
    row = {"comparison": f"OmniH2O vs {other} (Eg-mpjpe)", "diff_mm": round(T2["OmniH2O"] - T2[other], 2)}
    for cv in cvs:
        t, p = welch_from_cv(T2["OmniH2O"], T2[other], N_REAL, N_REAL, cv)
        row[f"welch_p_cv{int(cv*100)}"] = float(f"{p:.3g}")
    real.append(row)
# p = 0.05가 되는 CV (이진 탐색)
def cv_at_p05(m1, m2, n):
    lo, hi = 0.001, 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        _, p = welch_from_cv(m1, m2, n, n, mid)
        if p < 0.05:
            lo = mid
        else:
            hi = mid
    return round(lo, 3)
RES["T2_real_sensitivity"] = {"assumed_n": N_REAL, "rows": real,
                              "cv_at_p05_vs_MLP_linvel": cv_at_p05(T2["OmniH2O"], T2["MLP+linvel"], N_REAL),
                              "cv_at_p05_vs_H2O": cv_at_p05(T2["OmniH2O"], T2["H2O"], N_REAL)}

# ---------------- 표 17 (작업별 10회) -> 표 3 재계산 ----------------
TASKS = ["Catch-Release", "Squat", "Hammer-Catch", "RPS"]
# 성공 횟수 (10회 중), 작업 순서는 TASKS. MSE는 표 17의 값
T17 = {
    "data": {  # (a) 25%, 50%, 100%
        "25%": {"succ": [1, 9, 3, 3], "mse": [3.01e-3, 1.25e-4, 2.18e-2, 2.72e-2]},
        "50%": {"succ": [3, 10, 6, 9], "mse": [3.04e-4, 1.10e-4, 1.56e-2, 1.39e-2]},
        "100%": {"succ": [6, 10, 6, 10], "mse": [9.89e-5, 7.07e-5, 3.29e-4, 1.60e-3]},
    },
    "obs_act": {  # (b) Si-O-Si-A, Se-O-Se-A, Si-O-Se-A
        "Si-O-Si-A": {"succ": [3, 10, 5, 10], "mse": [2.52e-4, 5.18e-5, 2.22e-4, 1.43e-3]},
        "Se-O-Se-A": {"succ": [7, 10, 9, 9], "mse": [1.47e-4, 9.60e-5, 3.62e-4, 3.36e-3]},
        "Si-O-Se-A": {"succ": [6, 10, 6, 10], "mse": [9.89e-5, 7.07e-5, 3.29e-4, 1.60e-3]},
    },
    "arch": {  # (c) BC, DP-DDIM, DP-DDPM
        "BC": {"succ": [0, 3, 0, 1], "mse": [1.39e-3, 6.24e-4, 4.50e-3, 1.46e-2]},
        "DP-DDIM": {"succ": [6, 10, 5, 10], "mse": [4.79e-5, 6.42e-5, 3.41e-4, 2.42e-3]},
        "DP-DDPM": {"succ": [6, 10, 6, 10], "mse": [9.89e-5, 7.07e-5, 3.29e-4, 1.60e-3]},
    },
}
# 표 3 보고값
T3_REPORTED = {
    "data": {"25%": (4.0, 1.30e-2), "50%": (6.5, 7.48e-3), "100%": (8.0, 5.25e-4)},
    "obs_act": {"Si-O-Si-A": (6.5, 4.89e-4), "Se-O-Se-A": (8.75, 9.91e-4), "Si-O-Se-A": (8.0, 5.25e-4)},
    "arch": {"BC": (1.0, 5.63e-3), "DP-DDIM": (7.75, 1.9e-3), "DP-DDPM": (8.0, 5.25e-4)},
}
recon = {}
for group, d in T17.items():
    for name, v in d.items():
        s = sum(v["succ"])
        mean_s = s / 4
        mean_mse = float(np.mean(v["mse"]))
        rep_s, rep_mse = T3_REPORTED[group][name]
        recon[f"{group}/{name}"] = {
            "succ_sum_of_40": s,
            "succ_mean_per_10": mean_s,
            "reported_succ_per_10": rep_s,
            "succ_diff": round(mean_s - rep_s, 3),
            "mse_mean_recomputed": float(f"{mean_mse:.3g}"),
            "mse_reported": rep_mse,
            "mse_rel_diff_pct": round(100 * (mean_mse - rep_mse) / rep_mse, 1),
        }
RES["T3_recompute_from_T17"] = recon

# 검정 (40회 = 4작업 × 10회). Fisher는 작업을 무시한 풀링, CMH는 작업별 층화.
def pool(group, name):
    v = T17[group][name]
    return sum(v["succ"]), 40

tests = []
def add_test(label, g, a, b):
    ka, na = pool(g, a)
    kb, nb = pool(g, b)
    strata = [(T17[g][a]["succ"][i], 10, T17[g][b]["succ"][i], 10) for i in range(4)]
    chi2, p_cmh = cmh(strata)
    d, lo, hi = newcombe_diff(ka, na, kb, nb)
    tests.append({"comparison": label, "succ_a": f"{ka}/40", "succ_b": f"{kb}/40",
                  "diff_pp": round(100 * d, 1), "newcombe95_pp": [round(100 * lo, 1), round(100 * hi, 1)],
                  "fisher_p": float(f"{fisher(ka, na, kb, nb):.3g}"),
                  "cmh_chi2": round(chi2, 2), "cmh_p": float(f"{p_cmh:.3g}")})

add_test("DP-DDPM 대 BC", "arch", "DP-DDPM", "BC")
add_test("DP-DDIM 대 BC", "arch", "DP-DDIM", "BC")
add_test("DP-DDPM 대 DP-DDIM", "arch", "DP-DDPM", "DP-DDIM")
add_test("Se-O-Se-A 대 Si-O-Si-A", "obs_act", "Se-O-Se-A", "Si-O-Si-A")
add_test("Si-O-Se-A 대 Si-O-Si-A", "obs_act", "Si-O-Se-A", "Si-O-Si-A")
add_test("100%데이터 대 25%데이터", "data", "100%", "25%")
add_test("100%데이터 대 50%데이터", "data", "100%", "50%")
RES["T3_tests"] = tests

# ---------------- 데이터 규모 (부록 J) ----------------
frames = {"Catch-Release": 13234, "Squat": 8535, "Hammer-Catch": 12759, "RPS": 9380,
          "Boxing": 11118, "Basket-Pick-Place": 18436}
total = sum(frames.values())
RES["J_dataset"] = {
    "frames_total": total,
    "minutes_at_30hz": round(total / 30 / 60, 1),
    "minutes_per_task_at_30hz": {k: round(v / 30 / 60, 1) for k, v in frames.items()},
    "text_claim_minutes": {"Catch-Release": 5, "Squat": 5, "RPS": 5, "Hammer-Catch": 10,
                           "Basket-Pick-Place": 10, "Boxing": None},
}

# ---------------- 입력 차원 (표 4–14) ----------------
dims = {
    "표 4 privileged 합": [66, 138, 69, 69, 69, 138, 69, 69, 69, 138, 19],
    "표 5 H2O 합": [19, 19, 3, 3, 3, 72, 19],
    "표 6 OmniH2O 단일 스텝(3점)": [19, 19, 3, 3, 27, 19],
    "표 6 히스토리 1스텝": [19, 19, 3, 3, 19],
    "표 12 22점 단일 스텝": [19, 19, 3, 3, 198, 19],
    "표 13 8점 단일 스텝": [19, 19, 3, 3, 72, 19],
    "표 14 선속도 단일 스텝": [19, 19, 3, 3, 3, 27, 19],
    "표 14 선속도 히스토리 1스텝": [19, 19, 3, 3, 3, 19],
}
sums = {k: sum(v) for k, v in dims.items()}
sums["OmniH2O 총 (63×25 + 90)"] = 63 * 25 + 90
sums["22점 총 (63×25 + 261)"] = 63 * 25 + 261
sums["8점 총 (63×25 + 135)"] = 63 * 25 + 135
sums["선속도 총 (66×25 + 93)"] = 66 * 25 + 93
sums["Fig.3 표기 26프레임 (63×26 + 27)"] = 63 * 26 + 27
RES["dims"] = sums

# ---------------- 저장과 요약 ----------------
OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")

print("[표 1] N=14000 기준 차이와 Newcombe 구간")
for r in res_t1:
    print(f"  {r['comparison']:<30} diff {r['diff_pp_reported']:+6.2f} pp  CI {r['newcombe95_pp_Nbase']}  p(Fisher) {r['fisher_p_Nbase']}")
print("[표 1] 실패 시퀀스 평균 역산 (Eg-mpjpe):", implied)
print("[표 2] CV별 Welch p, 그리고 p=0.05 경계 CV")
for r in real:
    print("  ", r["comparison"], r["diff_mm"], {k: v for k, v in r.items() if k.startswith("welch")})
print("   CV at p=0.05 vs MLP+linvel:", RES["T2_real_sensitivity"]["cv_at_p05_vs_MLP_linvel"],
      " vs H2O:", RES["T2_real_sensitivity"]["cv_at_p05_vs_H2O"])
print("[표 3] 표 17에서 재계산 대 보고값")
for k, v in recon.items():
    print("  ", k, v["succ_mean_per_10"], "vs", v["reported_succ_per_10"], "| MSE", v["mse_mean_recomputed"], "vs", v["mse_reported"], f"({v['mse_rel_diff_pct']}%)")
print("[표 3] 검정")
for t in tests:
    print("  ", t)
print("[부록 J] 총 프레임", total, "=", RES["J_dataset"]["minutes_at_30hz"], "분 at 30 Hz")
print("[차원]", sums)
