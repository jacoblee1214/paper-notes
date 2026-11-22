"""GELLO (arXiv:2309.13037 v2) 사용자 연구 표 재계산과 검정. 수 초, 입력은 논문 표 값뿐이다.

실행: python3 analysis/recompute.py   → analysis/recompute_results.json
입력: 논문 Table II(성공률, 12명 x 과제 5개)와 Table III(실패 기호 개수).
원자료(사용자별 결과)는 공개되지 않아 쌍체(사용자 단위) 검정은 할 수 없다. 그래서 비쌍체 Fisher 검정과
쌍체 McNemar의 가능한 범위를 함께 낸다.
"""
import itertools
import json
import math
from pathlib import Path

from scipy.stats import binomtest, fisher_exact, norm

OUT = Path(__file__).resolve().parent / "recompute_results.json"
N_USERS = 12
TASKS = ["Hat", "Mask", "Banana", "Towel", "USB"]
# Table II (논문 값, 소수 둘째 자리 반올림)
TABLE2 = {
    "GELLO": [0.92, 0.92, 1.0, 0.92, 0.83],
    "3D Mice": [0.75, 0.58, 0.67, 0.58, 0.58],
    "VR": [0.92, 0.83, 0.75, 0.58, 0.5],
}
TABLE2_AVG = {"GELLO": 0.92, "3D Mice": 0.63, "VR": 0.72}
# Table III (실패 기호 개수, 논문 본문의 ✗ 개수)
TABLE3 = {
    "GELLO": {"Timeout": 1, "Self Collision": 1, "Env Collision": 3, "Other": 0},
    "3D Mice": {"Timeout": 9, "Self Collision": 5, "Env Collision": 6, "Other": 2},
    "VR": {"Timeout": 5, "Self Collision": 5, "Env Collision": 6, "Other": 1},
}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe_diff(k1, n1, k2, n2, z=1.959964):
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def mcnemar_exact_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue


def mcnemar_range(diff, max_discordant=60):
    """쌍체일 때 McNemar 정확 p의 범위. b - c = diff (성공 수 차이), b + c <= max_discordant."""
    ps = []
    for c in itertools.count(0):
        b = c + diff
        if b + c > max_discordant:
            break
        ps.append((mcnemar_exact_p(b, c), b, c))
    lo = min(ps)
    hi = max(ps)
    return {"min_p": lo[0], "min_p_bc": [lo[1], lo[2]], "max_p": hi[0], "max_p_bc": [hi[1], hi[2]]}


def main():
    res = {}
    # 1. Table II 재계산: 성공 수 = 반올림 값 x 12
    counts = {}
    for dev, vals in TABLE2.items():
        ks = [round(v * N_USERS) for v in vals]
        counts[dev] = ks
        avg_recalc = sum(ks) / (N_USERS * len(TASKS))
        res.setdefault("table2_check", {})[dev] = {
            "counts_out_of_12": dict(zip(TASKS, ks)),
            "pooled_success": sum(ks),
            "pooled_n": N_USERS * len(TASKS),
            "avg_recalc": round(avg_recalc, 4),
            "avg_reported": TABLE2_AVG[dev],
            "avg_matches": round(avg_recalc, 2) == TABLE2_AVG[dev],
            "max_rounding_gap": round(max(abs(k / N_USERS - v) for k, v in zip(ks, vals)), 4),
        }

    # 2. Table III 재계산: 실패 수 = 60 - 성공 수
    res["table3_check"] = {}
    for dev, fails in TABLE3.items():
        total_fail = sum(fails.values())
        expected_fail = N_USERS * len(TASKS) - sum(counts[dev])
        res["table3_check"][dev] = {"sum_of_marks": total_fail, "expected_failures": expected_fail,
                                    "consistent": total_fail == expected_fail}

    # 3. 장치별 Wilson 95% 구간 (풀링, 과제별)
    res["wilson_pooled"] = {}
    for dev in TABLE2:
        k, n = sum(counts[dev]), N_USERS * len(TASKS)
        lo, hi = wilson(k, n)
        res["wilson_pooled"][dev] = {"k": k, "n": n, "p": round(k / n, 4), "ci95": [round(lo, 3), round(hi, 3)]}

    # 4. 풀링 Fisher 검정 (비쌍체 가정, 60 대 60)
    res["fisher_pooled"] = {}
    for other in ("3D Mice", "VR"):
        k1, n1 = sum(counts["GELLO"]), 60
        k2, n2 = sum(counts[other]), 60
        _, p = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="two-sided")
        d, lo, hi = newcombe_diff(k1, n1, k2, n2)
        res["fisher_pooled"][f"GELLO vs {other}"] = {
            "diff": round(d, 4), "newcombe95": [round(lo, 3), round(hi, 3)], "fisher_p": round(p, 5),
            "mcnemar_if_paired": {k: (round(v, 5) if isinstance(v, float) else v)
                                  for k, v in mcnemar_range(k1 - k2).items()},
        }

    # 5. 과제별 Fisher (12 대 12), Holm 보정
    res["fisher_by_task"] = {}
    for other in ("3D Mice", "VR"):
        rows = []
        for i, t in enumerate(TASKS):
            a, b = counts["GELLO"][i], counts[other][i]
            _, p = fisher_exact([[a, 12 - a], [b, 12 - b]], alternative="two-sided")
            rows.append((t, a, b, p))
        # Holm
        order = sorted(range(len(rows)), key=lambda j: rows[j][3])
        adj = [0.0] * len(rows)
        running = 0.0
        for rank, j in enumerate(order):
            running = max(running, min(1.0, (len(rows) - rank) * rows[j][3]))
            adj[j] = running
        res["fisher_by_task"][f"GELLO vs {other}"] = [
            {"task": t, "gello": a, "other": b, "p_raw": round(p, 4), "p_holm": round(adj[j], 4)}
            for j, (t, a, b, p) in enumerate(rows)]

    # 6. 검정력: 60 시도 대 60 시도에서 GELLO 0.917 대 VR 0.717을 80%로 가르는가 (정규 근사, 양측 0.05)
    def power(p1, p2, n):
        pbar = (p1 + p2) / 2
        se0 = math.sqrt(2 * pbar * (1 - pbar) / n)
        se1 = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
        z = norm.ppf(0.975)
        return norm.cdf((abs(p1 - p2) - z * se0) / se1)

    p_g = sum(counts["GELLO"]) / 60
    res["power_normal_approx"] = {
        "vs_VR_n60_power": round(power(p_g, sum(counts["VR"]) / 60, 60), 3),
        "vs_3D_n60_power": round(power(p_g, sum(counts["3D Mice"]) / 60, 60), 3),
        "n_per_group_for_80pct_vs_VR": next(n for n in range(10, 5000)
                                            if power(p_g, sum(counts["VR"]) / 60, n) >= 0.8),
        "n_per_group_for_80pct_vs_3D": next(n for n in range(10, 5000)
                                            if power(p_g, sum(counts["3D Mice"]) / 60, n) >= 0.8),
    }


    # 6b. 사용자 안 상관(군집) 민감도: 시도가 사용자 12명에 묶여 있다. 설계 효과 DE = 1 + (m-1)*rho, m = 5 (과제 수)
    res["cluster_sensitivity"] = {}
    for other in ("3D Mice", "VR"):
        k1, n1 = sum(counts["GELLO"]), 60
        k2, n2 = sum(counts[other]), 60
        pbar = (k1 + k2) / (n1 + n2)
        z = (k1 / n1 - k2 / n2) / math.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
        res["cluster_sensitivity"][f"GELLO vs {other}"] = {
            str(rho): round(2 * norm.sf(abs(z) / math.sqrt(1 + 4 * rho)), 5) for rho in (0.0, 0.1, 0.2, 0.3, 0.5)}

    # 7. 비용 비율 (논문 Table I)
    cost = {"3D Mouse": 150, "GELLO": 300, "VR (Quest 2)": 300, "UR5 teleop": 30000, "Omega7 haptic": 40000}
    res["cost_ratio_vs_gello"] = {k: round(v / 300, 1) for k, v in cost.items()}

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
