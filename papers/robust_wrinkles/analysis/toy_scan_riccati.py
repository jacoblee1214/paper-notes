"""토이 T2: 논문 식 (13)의 칼만 공분산 재귀가 식 (16)-(17)의 결합 규칙으로 정확히 재현되는지 확인한다.

논문 §4.2는 Π_{k,j}를 순차 재귀 (13)으로 구하고, 이를 병렬 연관 스캔 (16)-(17)로 바꾼다고 쓴다.
(16)-(17)의 원소 (Ã, D̃, P̃)와 결합 ⊗을 그대로 구현해 가능한 순서 규약 네 가지를 시험하고, (13)의 순차 재귀와
최대 오차를 비교한다. 임의 선형 시변 계통(n=4, ny=2, 길이 16), numpy, 1초 미만.

실행: python3 analysis/toy_scan_riccati.py → analysis/toy_scan_riccati_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_scan_riccati_results.json"
rng = np.random.default_rng(1)
n, ny, T = 4, 2, 16
A = [rng.standard_normal((n, n)) * 0.5 for _ in range(T)]
C = [rng.standard_normal((ny, n)) for _ in range(T)]
E = [rng.standard_normal((n, n)) * 0.3 for _ in range(T)]
F = [np.eye(ny) * 0.4 + 0.1 * rng.standard_normal((ny, ny)) for _ in range(T)]
Xi = rng.standard_normal((n, n)) * 0.5
I = np.eye(n)

# (13) 순차 재귀
Pi = [Xi @ Xi.T]
for j in range(T):
    Pj = Pi[-1]
    S = F[j] @ F[j].T + C[j] @ Pj @ C[j].T
    L = -np.linalg.solve(S, C[j] @ Pj @ A[j].T)          # ny x n
    Pi.append(E[j] @ E[j].T + A[j] @ Pj @ A[j].T + A[j] @ Pj @ C[j].T @ L)


def comb(left, right):
    """식 (16): (A~, D~, P~)_{i,m} ⊗ (A~, D~, P~)_{m,l}."""
    Al, Dl, Pl = left
    Ar, Dr, Pr = right
    M1 = np.linalg.inv(I + Pr @ Dl)
    M2 = np.linalg.inv(I + Dl @ Pr)
    P = Al.T @ M1 @ Pr @ Al + Pl
    Aout = Ar @ M2 @ Al
    D = Ar @ M2 @ Dl @ Ar.T + Dr
    return (Aout, D, P)


def elem(j, transpose_A):
    Aj = A[j].T if transpose_A else A[j]
    return (Aj, C[j].T @ np.linalg.inv(F[j] @ F[j].T) @ C[j], E[j] @ E[j].T)


res = {}
for name, transpose_A in (("A_T (식 17 그대로)", True), ("A (전치 없음)", False)):
    for order in ("left_to_right", "right_to_left"):
        first = (np.zeros((n, n)), np.zeros((n, n)), Xi @ Xi.T)       # P~_{1,0}, A~_{1,0} = D~_{1,0} = 0
        els = [elem(j, transpose_A) for j in range(T)]
        errs = []
        acc = first
        for j in range(T):
            if order == "left_to_right":
                acc = comb(acc, els[j])
            else:
                acc = comb(els[j], acc)
            errs.append(float(np.max(np.abs(acc[2] - Pi[j + 1]))))
        res[f"{name} / {order}"] = {"max_abs_err_vs_eq13": max(errs), "first_step_err": errs[0]}

# 참고: 표준 병렬 칼만 필터(Sarkka) 결합 규칙으로 같은 원소를 결합하면 일치하는가
def comb_std(e1, e2):
    Aa, Ca, Ja = e1
    Ab, Cb, Jb = e2
    M = np.linalg.inv(I + Ca @ Jb)
    return (Ab @ M @ Aa, Ab @ M @ Ca @ Ab.T + Cb, Aa.T @ np.linalg.inv(I + Jb @ Ca) @ Jb @ Aa + Ja)


acc = (np.zeros((n, n)), Xi @ Xi.T, np.zeros((n, n)))
errs = []
for j in range(T):
    ej = (A[j], E[j] @ E[j].T, C[j].T @ np.linalg.inv(F[j] @ F[j].T) @ C[j])
    acc = comb_std(acc, ej)
    errs.append(float(np.max(np.abs(acc[1] - Pi[j + 1]))))
res["표준 병렬 칼만(Sarkka 식) / left_to_right"] = {"max_abs_err_vs_eq13": max(errs), "first_step_err": errs[0]}

# 결합법칙 검사: 균형 이진 트리로 묶은 결과가 순차 결과와 같은가 (병렬 스캔의 전제)
def tree(els):
    if len(els) == 1:
        return els[0]
    mid = len(els) // 2
    # 식 (17)의 인덱스(내려가는 표기)에 맞춰 나중 단계가 왼쪽
    return comb(tree(els[mid:]), tree(els[:mid]))


els = [elem(j, True) for j in range(T)]
first = (np.zeros((n, n)), np.zeros((n, n)), Xi @ Xi.T)
full = comb(tree(els), first)
res["이진 트리 결합 (식 17 원소, 나중 단계가 왼쪽) vs (13)"] = {"abs_err_final": float(np.max(np.abs(full[2] - Pi[T])))}

OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False))
