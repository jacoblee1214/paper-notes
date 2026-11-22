"""논문 표 수치 재계산: 성공률 Wilson 95% 구간, Fisher 정확 검정, 본문 주장 대조.

외부 패키지 없이 표준 라이브러리만 씀. 실행: python3 analysis/recompute.py
입력: 논문 Table II, III, IV와 본문 수치 (paper/paper.txt의 위치 주석 참조)
출력: analysis/recompute_results.json
"""
import json
import math
from pathlib import Path

OUT = Path(__file__).with_name("recompute_results.json")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fisher_two_sided(a, n1, b, n2):
    """2x2 표 [[a, n1-a], [b, n2-b]]의 양측 Fisher 정확 p값 (초기하분포 합)."""
    total_pos = a + b
    N = n1 + n2

    def pmf(x):
        return math.comb(total_pos, x) * math.comb(N - total_pos, n1 - x) / math.comb(N, n1)

    lo, hi = max(0, n1 - (N - total_pos)), min(n1, total_pos)
    p_obs = pmf(a)
    return min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs * (1 + 1e-9)))


# 논문 Table III 성공률 (SR, 시행 수). 라운딩을 보정해 정수 횟수로 옮긴 값
SR = {
    "Laundry (n=40)": {"n": 40, "ACT": 22, "MimicPlay": 20, "EgoMimic w/o human": 29, "EgoMimic": 35},
    "Groceries SR (n=50)": {"n": 50, "ACT": 11, "MimicPlay": 4, "EgoMimic w/o human": 14, "EgoMimic": 15},
    "Groceries Open Bag (n=50)": {"n": 50, "ACT": 27, "MimicPlay": 20, "EgoMimic w/o human": 30, "EgoMimic": 35},
}
# 표 III와 IV의 점수 (pts)
PTS = {
    "Bowl (Table III)": {"ACT": 39, "MimicPlay": 71, "EgoMimic w/o human": 68, "EgoMimic": 128},
    "Laundry (Table III)": {"ACT": 82, "MimicPlay": 78, "EgoMimic w/o human": 104, "EgoMimic": 114},
    "Groceries (Table III)": {"ACT": 82, "MimicPlay": 53, "EgoMimic w/o human": 92, "EgoMimic": 110},
}
ABL = {"EgoMimic": 128, "w/o line": 112, "w/o line and mask": 95, "w/o action norm": 79, "w/o hand data": 68}


def main():
    res = {"sr_wilson_fisher": {}, "score_checks": {}, "claim_checks": {}}
    for task, d in SR.items():
        n = d["n"]
        rows = {}
        for m in ["ACT", "MimicPlay", "EgoMimic w/o human", "EgoMimic"]:
            k = d[m]
            lo, hi = wilson(k, n)
            rows[m] = {"k": k, "n": n, "sr": round(k / n * 100, 1), "wilson95": [round(lo * 100, 1), round(hi * 100, 1)]}
        ego = d["EgoMimic"]
        tests = {}
        for m in ["ACT", "MimicPlay", "EgoMimic w/o human"]:
            tests[f"EgoMimic vs {m}"] = round(fisher_two_sided(ego, n, d[m], n), 4)
        res["sr_wilson_fisher"][task] = {"rows": rows, "fisher_p_two_sided": tests}

    # 점수 비율 (상대 향상)
    for task, d in PTS.items():
        base = d["ACT"]
        res["score_checks"][task] = {
            "EgoMimic_vs_ACT_rel_pct": round((d["EgoMimic"] / base - 1) * 100, 1),
            "EgoMimic_vs_w_o_human_rel_pct": round((d["EgoMimic"] / d["EgoMimic w/o human"] - 1) * 100, 1),
        }
    res["score_checks"]["ablation_drop_pct"] = {k: round((ABL["EgoMimic"] - v) / ABL["EgoMimic"] * 100, 1) for k, v in ABL.items() if k != "EgoMimic"}

    # 본문 주장 대조
    rel_bowl = PTS["Bowl (Table III)"]["EgoMimic"] / PTS["Bowl (Table III)"]["ACT"] - 1
    res["claim_checks"] = {
        "본문 228%: 대 ACT 점수 상대향상 (Bowl)": round(rel_bowl * 100, 1),
        "본문 34–228% 범위의 하한 (Groceries 상대향상)": res["score_checks"]["Groceries (Table III)"]["EgoMimic_vs_ACT_rel_pct"],
        "본문 8–33%p: Laundry SR 절대차": round(35 / 40 * 100 - 22 / 40 * 100, 1),
        "본문 8–33%p: Groceries SR 절대차": round(15 / 50 * 100 - 11 / 50 * 100, 1),
        "본문 10–88%: Bowl 대 w/o human 상대향상": round((128 / 68 - 1) * 100, 1),
        "본문 2–15%p: Groceries SR 절대차 (대 w/o human)": round(15 / 50 * 100 - 14 / 50 * 100, 1),
        "본문 2–15%p: Laundry SR 절대차 (대 w/o human)": round(35 / 40 * 100 - 29 / 40 * 100, 1),
        "본문 scaling 128 대 74 (ACT 3h): 상대차 %": round((128 / 74 - 1) * 100, 1),
        "표 II 인간 1h 데모 수 (1400/60min)*60": round(1400 / 60 * 60),
        "표 II 로봇 1h 데모 수 (270/120min)*60": round(270 / 120 * 60),
        "Fig 7 unseen color: 대 ACT 25% vs 85% (n 미보고)": "n 미보고 → 검정 불가",
        "Fig 8 scaling: 128 vs 74 (n 미보고)": "n 미보고 → 검정 불가",
        "Bowl Pts (45 rollouts) 평균 pts/rollout EgoMimic": round(128 / 45, 2),
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
