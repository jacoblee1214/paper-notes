"""논문 표의 보고값으로 성공률을 다시 계산하고 일관성을 검사한다 (재현 아님).

실행: python3 analysis/recompute.py  → analysis/recompute_results.json
입력: arXiv v1 (2401.02117v1)의 Table 1·2·3·4 값. 페이지 이미지(pdftoppm)로 직접 읽었다.
      시도 수는 논문 6절 본문대로 과제당 20회, 쿡 쉬림프만 5회다.
      하위 과제 시도 수 = 직전 하위 과제 성공 수 (논문 6절 정의). 전체 성공률 = 하위 과제 성공률의 곱.

검사
  블록 A  whole-task 셀의 Wilson 95% CI, co-train 대 no-co-train Fisher 정확 검정 (과제별, Holm 보정)
  블록 B  본문 "평균 34% 절대 향상"과 표 1에서 다시 계산한 값
  블록 C  표 1 사슬이 정수 성공 횟수로 가능한지 (0.5%p 이내 = 정확, 1%p 이내 = 반올림 차이)
  블록 D  표 2 사슬 (VINN·DP·ACT, 3 과제)
  블록 E  표 3·4 (20회 기준 정수 여부)
"""
import json
import math
from pathlib import Path

from scipy.stats import fisher_exact

OUT = Path(__file__).resolve().parent / "recompute_results.json"
Z = 1.959963984540054


def wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def feasible(N, subs, whole, tol):
    """정수 성공 횟수 사슬이 존재하는지. subs의 각 값은 직전 단계 성공 수를 분모로 한 백분율."""
    def dfs(i, trials):
        if i == len(subs):
            return abs(100.0 * trials / N - whole) <= tol + 1e-9
        if trials == 0:
            return False
        for k in range(trials + 1):
            if abs(100.0 * k / trials - subs[i]) <= tol + 1e-9 and dfs(i + 1, k):
                return True
        return False
    return dfs(0, N)


def verdict(N, subs, whole):
    if feasible(N, subs, whole, 0.5):
        return "정확 일치"
    if feasible(N, subs, whole, 1.0):
        return "반올림 차이 (1%p 이내)"
    return "불일치 (1%p 초과)"


# ---- 표 1: ACT. (하위 과제 %, 전체 %), 전체 성공 횟수 ----
TABLE1 = {
    "Wipe Wine":     {"N": 20, "co": ([100, 95, 100], 95), "no": ([95, 58, 90], 50), "k": (19, 10)},
    "Cook Shrimp":   {"N": 5,  "co": ([100, 100, 60, 67], 40), "no": ([100, 100, 40, 50], 20), "k": (2, 1)},
    "Rinse Pan":     {"N": 20, "co": ([100, 80, 100], 80), "no": ([100, 0, 100], 0), "k": (16, 0)},
    "Use Cabinet":   {"N": 20, "co": ([95, 100, 95, 95], 85), "no": ([95, 95, 100, 95], 85), "k": (17, 17)},
    "Call Elevator": {"N": 20, "co": ([100, 100, 95], 95), "no": ([100, 5, 0], 0), "k": (19, 0)},
    # Push Chairs: 1-3번째 의자 100%는 20/20으로 두고 4번째·5번째 (OOD)를 검사
    "Push Chairs (5 chairs)": {"N": 20, "co": ([100, 85, 89], 80), "no": ([100, 70, 0], 0), "k": (16, 0)},
    # High Five: 전체 85만 확인 (attire·human은 별도 조건이라 사슬이 아님)
    "High Five": {"N": 20, "co": ([100], 85), "no": ([100], 85), "k": (17, 17)},
}

# ---- 표 2: 3 과제 (N=20) ----
TABLE2 = {
    "VINN Wipe Wine co": ([85, 18, 100], 15),
    "VINN Wipe Wine no": ([50, 40, 100], 20),
    "DP Wipe Wine co": ([90, 72, 100], 65),
    "DP Wipe Wine no": ([75, 47, 100], 35),
    "ACT Wipe Wine co": ([100, 95, 100], 95),
    "ACT Wipe Wine no": ([95, 58, 90], 50),
    "VINN Push co (1-3 chair)": ([100, 70, 86], 60),
    "VINN Push no (1-3 chair)": ([90, 72, 62], 40),
    "DP Push co (1-3 chair)": ([100, 100, 100], 100),
    "DP Push no (1-3 chair)": ([100, 80, 100], 80),
    "ACT Push co (1-3 chair)": ([100, 100, 100], 100),
    "ACT Push no (1-3 chair)": ([100, 100, 100], 100),
}

# ---- 표 3, 4 (Wipe Wine, ACT, 20회) ----
TABLE3 = {"30%": 95, "50% (기본)": 95, "70%": 90}
TABLE4 = {"Co-train": 95, "Pre-train": 40, "No pre-train": 50}


def main():
    res = {"A_whole_task": {}, "B_mean_gain": {}, "C_chain": {}, "C_push_chairs_5th": {},
           "D_table2_chain": {}, "T3_T4": {}}

    names, pvals = [], []
    for task, d in TABLE1.items():
        N = d["N"]
        kc, kn = d["k"]
        _, p = fisher_exact([[kc, N - kc], [kn, N - kn]])
        names.append(task)
        pvals.append(p)
        res["A_whole_task"][task] = {
            "N": N,
            "co": f"{kc}/{N}", "co_pct": round(100 * kc / N, 1), "co_wilson95": wilson(kc, N),
            "no": f"{kn}/{N}", "no_pct": round(100 * kn / N, 1), "no_wilson95": wilson(kn, N),
            "fisher_p_raw": round(p, 6),
        }
    for task, a in zip(names, holm(pvals)):
        res["A_whole_task"][task]["fisher_p_holm"] = round(a, 6)

    co_vals = [95, 40, 80, 85, 95, 80, 85]
    no_vals = [50, 20, 0, 85, 0, 0, 85]
    gains = [c - n for c, n in zip(co_vals, no_vals)]
    res["B_mean_gain"] = {
        "tasks": ["Wipe Wine", "Cook Shrimp", "Rinse Pan", "Use Cabinet", "Call Elevator", "Push Chairs", "High Five"],
        "co_mean": round(sum(co_vals) / 7, 2),
        "no_mean": round(sum(no_vals) / 7, 2),
        "mean_gain_pp": round(sum(gains) / 7, 2),
        "gains_pp": gains,
        "max_gain_pp": max(gains),
        "paper_intro_claim_pct": 34,
        "paper_abstract_up_to_pct": 90,
    }

    for task, d in TABLE1.items():
        if task == "High Five":
            res["C_chain"]["High Five | co / no"] = "전체 85 = 17/20 (정수 가능), 하위 조건은 사슬 아님"
            continue
        for arm in ("co", "no"):
            subs, whole = d[arm]
            if task.startswith("Push Chairs"):
                # 4번째 의자 85% -> 17/20 이고, 5번째는 17회 중 시도
                chain = [100, 85, 89] if arm == "co" else [100, 70, 0]
                res["C_chain"][f"{task} | {arm}"] = verdict(20, chain, whole)
                continue
            if task == "Rinse Pan" and arm == "no":
                res["C_chain"][f"{task} | {arm}"] = "0회 이후 Place Pan의 분모가 0이라 정의되지 않음 (표는 100 표시)"
                continue
            res["C_chain"][f"{task} | {arm}"] = verdict(d["N"], subs, whole)
    res["C_chain"]["Cook Shrimp | co (N=5)"] = verdict(5, TABLE1["Cook Shrimp"]["co"][0], TABLE1["Cook Shrimp"]["co"][1])
    res["C_chain"]["Cook Shrimp | no (N=5)"] = verdict(5, TABLE1["Cook Shrimp"]["no"][0], TABLE1["Cook Shrimp"]["no"][1])

    # 5번째 의자: 4번째 성공 17회에서 89%에 가장 가까운 정수 해
    best = min(range(0, 18), key=lambda k: abs(100 * k / 17 - 89))
    res["C_push_chairs_5th"] = {
        "4th_success_of_20": 17,
        "5th_closest_k_of_17": best,
        "5th_closest_pct": round(100 * best / 17, 1),
        "whole_implied_pct": round(100 * best / 20, 1),
        "reported_whole_pct": 80,
    }

    for name, (subs, whole) in TABLE2.items():
        res["D_table2_chain"][name] = verdict(20, subs, whole)

    for label, v in {**TABLE3, **TABLE4}.items():
        k = round(v * 20 / 100)
        res["T3_T4"][label] = {"reported_pct": v, "k_of_20": k, "exact_integer": abs(100 * k / 20 - v) < 1e-9}

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
