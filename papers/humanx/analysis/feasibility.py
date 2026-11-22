"""Table 1 퍼센트 값의 정수 분모 가능성 점검 (HumanX arXiv:2602.02473 v2).

각 열(같은 평가 횟수 n을 공유한다고 가정)의 모든 값이 k/n의 소수 첫째 자리 반올림으로
나올 수 있는 최소 n을 찾는다. 실행: python3 feasibility.py
"""
import json
from fractions import Fraction

def feasible(n, val, dec):
    # val: 퍼센트, dec: 표기 소수 자릿수. 100k/n 을 dec 자리로 반올림해 val 이 되는 정수 k(0..n)가 있는지
    import math
    half = 0.5 * 10 ** (-dec)
    k_lo = max(0, math.ceil((val - half) * n / 100 - 1e-9))
    k_hi = min(n, math.floor((val + half) * n / 100 + 1e-9))
    return any(round(100.0 * k / n + 1e-12, dec) == round(val, dec) for k in range(k_lo, k_hi + 1))

def decimals(s):
    return len(s.split('.')[1]) if '.' in s else 0

cols = {
    "basketball_SR": ["0.0", "0.0", "53.1", "93.4", "86.8", "93.4", "77.6", "86.8"],
    "basketball_GSR": ["0.0", "0.0", "2.4", "4.9", "13.5", "10.9", "60.6", "64.7"],
    "badminton_SR": ["68.2", "75.8", "90.0", "97.1", "94.0", "100", "100", "100"],
    "badminton_GSR": ["30.9", "20.4", "25.3", "41.6", "60.5", "67.2", "84.4", "90.6"],
    "cargo_SR": ["0.4", "0.0", "95.8", "100", "100", "99.6", "99.3"],
    "cargo_GSR": ["0.0", "0.0", "1.8", "50.9", "94.5", "94.3", "96.3"],
    "table3_GSR": ["74.2", "93.1", "52.4", "84.3"],
}
NMAX = 3000
out = {}
for name, vals in cols.items():
    ok = []
    for n in range(1, NMAX + 1):
        if all(feasible(n, float(v), decimals(v)) for v in vals):
            ok.append(n)
    # 각 값의 개별 최소 n
    per = {v: next((n for n in range(1, NMAX + 1) if feasible(n, float(v), decimals(v))), None) for v in sorted(set(vals), key=float)}
    out[name] = {"joint_min_n": ok[0] if ok else None, "joint_n_count_le_%d" % NMAX: len(ok), "joint_first10": ok[:10], "per_value_min_n": per}
    print(name, "공통 최소 n =", ok[0] if ok else "없음(<=%d)" % NMAX, "| 값별 최소 n:", per)
print()
for n in (1000, 2048, 4096, 16384):
    print("n=%d에서 모든 열 가능?" % n, {name: all(feasible(n, float(v), decimals(v)) for v in vals) for name, vals in cols.items()})
    out.setdefault("check_n", {})[str(n)] = {name: all(feasible(n, float(v), decimals(v)) for v in vals) for name, vals in cols.items()}
json.dump(out, open("feasibility_results.json", "w"), ensure_ascii=False, indent=1)
