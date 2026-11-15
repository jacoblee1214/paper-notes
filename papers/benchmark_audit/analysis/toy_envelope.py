"""토이 2: 논문 부록 C의 짝지은 상한·하한 봉투(envelope)를 완전 열거로 검산 (numpy, 약 10초).

논문 주장: 공개된 집계 점수(µA, µB)만으로 짝지은 Wald 검정의 기각 여부가 정해지는 구간이 있고,
그 경계는 '가능한 모든 짝지은 결과표'에 대한 Q의 최소·최대로 정해진다 (C.3–C.5).

검산: 이진 결과, T=2 과제, 과제당 S개 짝지은 시도 (S=4, 5). 가능한 모든 결과표를 열거해서
  (1) 각 (A, B) 쌍에서 Q의 최소값이 Qlo(L; T, S) = r - r²/S (L = qS + r)와 같은지,
  (2) Q의 최대값이 DP 기반 Qhi(A, L)와 같은지,
  (3) 기각 분류 (모든 표 기각 / 어떤 표는 기각 / 기각 없음)가 논문의 경계와 같은지 확인한다.
또한 독립 표본 Qind가 [Qlo, Qhi] 안에 있는지 확인한다 (C.6).

실행: python3 -I analysis/toy_envelope.py  → analysis/toy_envelope_results.json
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "toy_envelope_results.json"
ALPHA = 0.05
Z = norm.ppf(1 - ALPHA)


def qlo(L, T, S):
    r = L % S
    return r - r * r / S


def qhi_dp(T, S):
    """DP: 과제별 최대 기여 qmax(a,b) = min(a+b, 2S-a-b) - (b-a)²/S, 과제 합으로 (A,B)별 최대 Q."""
    N = T * S
    dp = np.full((N + 1, N + 1), -np.inf)
    dp[0, 0] = 0.0
    for _ in range(T):
        new = np.full_like(dp, -np.inf)
        for a in range(S + 1):
            for b in range(S + 1):
                q = min(a + b, 2 * S - a - b) - (b - a) ** 2 / S
                shifted = np.full_like(dp, -np.inf)
                shifted[a:, b:] = dp[: N + 1 - a, : N + 1 - b] + q
                new = np.maximum(new, shifted)
        dp = new
    return dp


def exact_q(T, S):
    """과제별 (a, b, s, d)를 모두 조합해 (A, B) -> (min Q, max Q)."""
    types = [(0, 0), (0, 1), (1, 0), (1, 1)]
    seqs = list(itertools.product(range(4), repeat=S))
    rows = []
    for seq in seqs:
        x = np.array([types[k][0] for k in seq])
        y = np.array([types[k][1] for k in seq])
        rows.append((int(x.sum()), int(y.sum()), float(((y - x) ** 2).sum()), float((y - x).sum())))
    rows = np.array(rows)
    # 과제 하나의 기여 q_t = s_t - d_t²/S (Q는 과제 기여의 합)
    qt = rows[:, 2] - rows[:, 3] ** 2 / S
    base = rows[:, :2]
    # T개 과제 조합: 과제 기여 합 Q와 총합 (A, B)
    tot = base.copy()
    qsum = qt.copy()
    for _ in range(T - 1):
        tot = (tot[:, None, :] + base[None, :, :]).reshape(-1, 2)
        qsum = (qsum[:, None] + qt[None, :]).reshape(-1)
    A = tot[:, 0].astype(int)
    B = tot[:, 1].astype(int)
    N = T * S
    qmin = np.full((N + 1, N + 1), np.inf)
    qmax = np.full((N + 1, N + 1), -np.inf)
    np.minimum.at(qmin, (A, B), qsum)
    np.maximum.at(qmax, (A, B), qsum)
    return qmin, qmax


def check(T, S):
    qmin, qmax = exact_q(T, S)
    dp = qhi_dp(T, S)
    N = T * S
    worst_lo, worst_hi, n_pairs = 0.0, 0.0, 0
    mism_class = []
    for A in range(N + 1):
        for B in range(A + 1, N + 1):
            if not np.isfinite(qmin[A, B]):
                continue
            L = B - A
            lo_theory = qlo(L, T, S)
            hi_theory = dp[A, B]
            worst_lo = max(worst_lo, abs(qmin[A, B] - lo_theory))
            worst_hi = max(worst_hi, abs(qmax[A, B] - hi_theory))
            n_pairs += 1
            # 분류: 기각 조건 L > cα sqrt(S/(S-1) Q)
            c = Z * math.sqrt(S / (S - 1))
            all_reject_exact = L > c * math.sqrt(max(qmax[A, B], 0.0))
            none_reject_exact = L <= c * math.sqrt(max(qmin[A, B], 0.0))
            all_reject_theory = L > c * math.sqrt(max(hi_theory, 0.0))
            none_reject_theory = L <= c * math.sqrt(max(lo_theory, 0.0))
            if (all_reject_exact, none_reject_exact) != (all_reject_theory, none_reject_theory):
                mism_class.append((A, B))
    return {"T": T, "S": S, "feasible_pairs": n_pairs, "max_abs_err_Qlo": float(worst_lo),
            "max_abs_err_Qhi": float(worst_hi), "classification_mismatches": len(mism_class)}


results = {"alpha": ALPHA, "z": float(Z), "checks": []}
for T, S in [(2, 4), (2, 5)]:
    r = check(T, S)
    results["checks"].append(r)
    print(r)

# 독립 표본 Qind가 봉투 안에 있는지 (C.6): 무작위 독립 쌍 샘플
rng = np.random.default_rng(7)
T, S = 2, 5
dp = qhi_dp(T, S)
viol = 0
trials = 20000
for _ in range(trials):
    xa = rng.integers(0, 2, size=(T, S))
    xb = rng.integers(0, 2, size=(T, S))
    a = xa.sum(axis=1)
    b = xb.sum(axis=1)
    qind = sum(((xa[t] ** 2).sum() - a[t] ** 2 / S) + ((xb[t] ** 2).sum() - b[t] ** 2 / S) for t in range(T))
    A, B = int(a.sum()), int(b.sum())
    if B <= A:
        continue
    L = B - A
    if not (qlo(L, T, S) - 1e-9 <= qind <= dp[A, B] + 1e-9):
        viol += 1
results["independent_sample_envelope_check"] = {"T": T, "S": S, "draws": trials, "violations": viol}
print("independent-sample envelope violations:", viol, "of", trials)

OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", OUT)
