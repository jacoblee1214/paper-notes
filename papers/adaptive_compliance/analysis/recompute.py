"""논문 표 I, II의 성공 횟수를 원시 시도 수로 되돌려 재계산한다(표준 라이브러리만 사용).

실행: python3 analysis/recompute.py  → analysis/recompute_results.json
표 값은 arXiv:2410.09309v2 Table I, II(% 와 시도 수 20/16 기준)에서 옮겼다.
열 순서는 본문 시나리오 순서이며, 행 합계가 열 "All"과 맞는지 검사한다.
"""
import json
import math
from pathlib import Path

OUT = Path(__file__).with_name("recompute_results.json")

# Table I: 시나리오별 시도 20회. 성공 횟수 = 표의 % / 5
FLIP_SCEN = ["Train", "Unseen", "Push&Flip", "FixturePose", "Unstable"]
FLIP_N = 20
FLIP_PCT_ALL = {"ACP": 96, "ACP_woFFT": 95, "Compliant": 23, "Stiff": 14}
FLIP_PCT = {
    "ACP":        [90, 95, 95, 100, 100],
    "ACP_woFFT":  [90, 100, 100, 95, 90],
    "Compliant":  [80, 15, 15, 5, 0],
    "Stiff":      [20, 5, 0, 35, 10],
}
# Table II: 시나리오 Small(5), Large(5), PbC(4), PaC(2). 행 = 방법, 열 = 시나리오
WIPE_SCEN = ["Small", "Large", "PbC", "PaC"]
WIPE_N = [5, 5, 4, 2]
WIPE_PCT_ALL = {"ACP": 93.75, "ACP_woFFT": 81.25, "Compliant": 43.75}
WIPE_PCT = {
    "ACP":       [100, 80, 100, 100],
    "ACP_woFFT": [100, 60, 75, 100],
    "Compliant": [60, 20, 25, 100],
}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def fisher_two_sided(a, b, c, d):
    """2x2 표 [[a,b],[c,d]]의 양측 Fisher 정확 검정 p값 (초기하분포 합)."""
    n1, n2, k = a + b, c + d, a + c
    N = n1 + n2

    def prob(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(N, k)

    p_obs = prob(a)
    lo, hi = max(0, k - n2), min(n1, k)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs * (1 + 1e-9)))


def counts(pct, n):
    return [round(p / 100 * m) for p, m in zip(pct, n)]


def main():
    res = {"checks": {}, "flip": {}, "wipe": {}, "tests": {}}
    # 1. 성공 횟수 복원과 합계 검사
    flip_k = {m: [round(p / 100 * FLIP_N) for p in v] for m, v in FLIP_PCT.items()}
    flip_tot = {m: sum(v) for m, v in flip_k.items()}
    wipe_k = {m: counts(v, WIPE_N) for m, v in WIPE_PCT.items()}
    wipe_tot = {m: sum(v) for m, v in wipe_k.items()}
    res["checks"]["flip_total_matches_all"] = {m: (flip_tot[m], FLIP_PCT_ALL[m] * 100 // 100) for m in FLIP_PCT}
    res["checks"]["flip_total_ok"] = all(flip_tot[m] == FLIP_PCT_ALL[m] for m in FLIP_PCT)
    res["checks"]["wipe_total_ok"] = {m: wipe_tot[m] for m in WIPE_PCT}
    res["checks"]["wipe_total_expected"] = {m: round(v / 100 * 16, 2) for m, v in WIPE_PCT_ALL.items()}
    res["checks"]["wipe_total_ok_bool"] = all(abs(wipe_tot[m] - WIPE_PCT_ALL[m] / 100 * 16) < 1e-9 for m in WIPE_PCT)
    # 2. 구간과 검정
    total_n = 100
    for m in FLIP_PCT:
        k = FLIP_PCT_ALL[m]
        lo, hi = wilson(k, total_n)
        res["flip"][m] = {"success": k, "n": total_n, "pct": k, "wilson95": [round(lo * 100, 1), round(hi * 100, 1)],
                          "scenarios": {s: f"{v}/20" for s, v in zip(FLIP_SCEN, flip_k[m])}}
    for m in WIPE_PCT:
        k = round(WIPE_PCT_ALL[m] / 100 * 16)
        lo, hi = wilson(k, 16)
        res["wipe"][m] = {"success": k, "n": 16, "pct": WIPE_PCT_ALL[m], "wilson95": [round(lo * 100, 1), round(hi * 100, 1)],
                          "scenarios": {s: f"{v}/{n}" for s, v, n in zip(WIPE_SCEN, wipe_k[m], WIPE_N)}}
    pairs = [("flip", "ACP", "Compliant"), ("flip", "ACP", "Stiff"), ("flip", "ACP", "ACP_woFFT"),
             ("wipe", "ACP", "Compliant"), ("wipe", "ACP", "ACP_woFFT")]
    for task, a, b in pairs:
        d = res[task]
        ka, na, kb, nb = d[a]["success"], d[a]["n"], d[b]["success"], d[b]["n"]
        p = fisher_two_sided(ka, na - ka, kb, nb - kb)
        res["tests"][f"{task}:{a}_vs_{b}"] = {"diff_pts": round(100 * (ka / na - kb / nb), 2),
                                              "rel_gain_pct": round(100 * (ka / na / (kb / nb) - 1), 1) if kb else None,
                                              "fisher_p_two_sided": round(p, 6)}
    # 3. "50% 이상" 주장 점검
    res["claim_50pct"] = {
        "wipe_ACP_minus_Compliant_pts": round(93.75 - 43.75, 2),
        "wipe_relative_gain_pct": round(100 * (93.75 / 43.75 - 1), 1),
        "flip_ACP_minus_Compliant_pts": 96 - 23,
        "flip_relative_gain_pct": round(100 * (96 / 23 - 1), 1),
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
