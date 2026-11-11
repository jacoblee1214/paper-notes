"""추가 검증 2: 고정소수점 정수 리덕션(식 15)의 셔플 재현성과 기울기 수준 정확성 (CPU numpy 1스레드, 1 분 이내).

논문 식 15: 각 기여 c_j를 2^F로 곱해 반올림 → 정수로 더함 → 한 번 복원. 정수 합은 순서에 무관하다.
검증 두 가지:
  (1) 셔플 테스트: 기여 N = 2000개(크기가 10^-6 … 10^3에 걸친 무작위 값)를 1000번 무작위 순서로 더한다.
      f32 순차 합, f64 순차 합, 고정소수점(F = 20, 40) 정수 합의 서로 다른 결과 수(비트 기준)와
      정확한 합(math.fsum, 정확 반올림) 대비 오차를 본다.
  (2) 기울기 수준: 혼돈 사상 x_{k+1} = a S_k (1 − S_k), S_k = Σ_j w_j x_k, 접선 d_{k+1} = a(1 − 2S_k) Σ_j w_j d_k
      (a = 3.9, T = 40). 같은 리덕션 방식으로 접선을 끝까지 돌리고, 80자리 Decimal로 같은 정의를 정확히 계산한 값과 비교한다.
      고정소수점은 실행 순서와 무관하게 같은 값을 준다. 그 값이 참값에 얼마나 가까운지가 질문이다.
가정: 기여 값의 범위, F = 20, 40(식 20의 비트 예산 안에서 두 점), w_j는 pde 대신 무작위 가중치(합 1 정규화).
실행: python3 analysis/extra_fixedpoint_reduce.py -> analysis/extra_fixedpoint_reduce_results.json
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import json
import math
import time
from decimal import Decimal, getcontext
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N = 2000
SHUF = 1000
A0, X0 = 3.9, 0.3
T_CHAIN = 40


def fixed_point_sum(c, F, perm):
    """c: (N,) float64, perm: (R, N) 순서. 각 행을 같은 순서로 정수 합산(식 15). 반환 (R,) float64."""
    q = np.rint(c * (2.0**F)).astype(np.int64)            # 기여마다 한 번 반올림
    tot = np.zeros(perm.shape[0], dtype=np.int64)
    rows = np.arange(perm.shape[0])
    for j in range(N):
        tot += q[perm[:, j]]                              # 정수 덧셈은 결합법칙이 정확히 성립
    return tot.astype(np.float64) * (2.0**-F)


def seq_sum(vals, perm, dtype):
    acc = np.zeros(perm.shape[0], dtype=dtype)
    rows = np.arange(perm.shape[0])
    for j in range(N):
        acc += vals[rows, perm[:, j]].astype(dtype)
    return acc


def part1():
    rng = np.random.default_rng(11)
    sgn = rng.choice([-1.0, 1.0], N)
    mag = 10.0 ** rng.uniform(-6, 3, N)
    c = sgn * mag
    exact = math.fsum(c)
    perm = np.argsort(rng.random((SHUF, N)), axis=1)
    res = {"N": N, "shuffles": SHUF, "exact_fsum": exact, "max_abs_contrib": float(np.max(np.abs(c)))}
    cm = np.broadcast_to(c, (SHUF, N))
    # f32, f64 순차 합
    for name, dt in (("f32_seq", np.float32), ("f64_seq", np.float64)):
        vals = np.broadcast_to(c.astype(dt), (SHUF, N))
        out = seq_sum(np.ascontiguousarray(vals), perm, dt).astype(np.float64)
        res[name] = dict(n_distinct_bitpatterns=int(len(np.unique(out.view(np.int64) if dt == np.float64 else out.view(np.int32)))),
                         abs_err_max=float(np.max(np.abs(out - exact))),
                         abs_err_median=float(np.median(np.abs(out - exact))),
                         rel_err_median=float(np.median(np.abs(out - exact)) / abs(exact)))
    # 고정소수점 F = 20, 40 (식 20의 비트 예산보다 작은 값과 큰 값)
    for F in (20, 40):
        out = fixed_point_sum(c, F, perm)
        res[f"fixed_F{F}"] = dict(n_distinct_bitpatterns=int(len(np.unique(out))),
                                  abs_err_max=float(np.max(np.abs(out - exact))),
                                  abs_err_median=float(np.median(np.abs(out - exact))),
                                  quantum_2_to_minus_F=2.0**-F,
                                  int64_headroom_bits=float(63 - math.log2(max(np.max(np.abs(c)) * 2**F * N, 1))))
    return res


def decimal_chain_tangent(W):
    """80자리 Decimal로 같은 리덕션 정의(exact 합)를 계산한 d_T."""
    getcontext().prec = 80
    Wd = [Decimal(float(w)) for w in W]
    a = Decimal(A0)
    x = Decimal(X0)
    d = Decimal(1)
    out = {}
    for k in range(1, T_CHAIN + 1):
        S = sum(wj * x for wj in Wd)
        dS = sum(wj * d for wj in Wd)
        d = a * (1 - 2 * S) * dS
        x = a * S * (1 - S)
        out[k] = float(d)
    return out


overflow = {}


def chain(method, W, F=None):
    """같은 정의를 float 방식으로 돌린다. method: 'f32', 'f64', 'fixed'. 실행 순서는 고정(0..N-1)."""
    x = X0
    d = 1.0
    out = {}
    if method == "f32":
        Wf = W.astype(np.float32)
    for k in range(1, T_CHAIN + 1):
        if method == "f64":
            S = 0.0                              # 고정 순서 순차 합(pairwise 합 아님)
            for wj in W:
                S += wj * x
            dS = 0.0
            for wj in W:
                dS += wj * d
        elif method == "f32":
            Sf, dSf = np.float32(0.0), np.float32(0.0)
            xf, df = np.float32(x), np.float32(d)
            for wj in Wf:
                Sf += wj * xf
                dSf += wj * df
            S, dS = float(Sf), float(dSf)
        else:  # fixed point: 기여를 정수로 반올림, 정수 합(Python int로 정확히 계산), 한 번 복원
            cS = [int(v) for v in np.rint((W * x) * 2.0**F)]
            cD = [int(v) for v in np.rint((W * d) * 2.0**F)]
            tS, tD = sum(cS), sum(cD)
            if abs(tS) >= 2**63 or abs(tD) >= 2**63:
                overflow.setdefault(method + str(F), k)   # int64였다면 여기서 넘친다
            S = tS * 2.0**-F
            dS = tD * 2.0**-F
        d = A0 * (1 - 2 * S) * dS
        x = A0 * S * (1 - S)
        out[k] = d
    return out


def part2():
    rng = np.random.default_rng(3)
    W = rng.uniform(0.5, 1.5, N)
    W = W / W.sum()
    ref = decimal_chain_tangent(W)
    res = {"T": T_CHAIN, "reference": "Decimal 80자리, 같은 리덕션 정의"}
    rows = {}
    overflow.clear()
    for name, method, F in (("f64_seq", "f64", None), ("f32_seq", "f32", None), ("fixed_F20", "fixed", 20),
                            ("fixed_F30", "fixed", 30), ("fixed_F40", "fixed", 40), ("fixed_F50", "fixed", 50)):
        ch = chain(method, W, F)
        rows[name] = {str(T): dict(d=ch[T], rel_err=abs(ch[T] - ref[T]) / abs(ref[T])) for T in (5, 10, 20, 30, 40)}
        if F is not None:
            rows[name]["first_step_int64_overflow"] = overflow.get(method + str(F))
    res["ref_dT"] = {str(T): ref[T] for T in (5, 10, 20, 30, 40)}
    res["methods"] = rows
    # 고정소수점 F=40은 실행 순서와 무관한가: 같은 기여를 무작위 순서로 더해도 정수 합은 같다
    rng2 = np.random.default_rng(5)
    perm = np.argsort(rng2.random((50, N)), axis=1)
    q = np.rint((W * 0.37) * 2.0**40).astype(np.int64)
    sums = np.array([int(q[p].sum()) for p in perm])
    res["fixed_F40_shuffle_distinct_T1_contrib"] = int(len(np.unique(sums)))
    return res


def main():
    t_all = time.perf_counter()
    out = {"part1_shuffle": part1(), "part2_tangent_chain": part2()}
    out["total_wall_s"] = round(time.perf_counter() - t_all, 2)
    (HERE / "extra_fixedpoint_reduce_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    p1 = out["part1_shuffle"]
    for k in ("f32_seq", "f64_seq", "fixed_F20", "fixed_F40"):
        v = p1[k]
        print(f"{k:10s} distinct={v['n_distinct_bitpatterns']:4d} abs_err_max={v['abs_err_max']:.3e} "
              f"abs_err_median={v['abs_err_median']:.3e}")
    p2 = out["part2_tangent_chain"]
    for name, rows in p2["methods"].items():
        print(name, {T: f"{rows[T]['rel_err']:.2e}" for T in rows if T.isdigit()}, "overflow_at", rows.get("first_step_int64_overflow"))
    print("wall", out["total_wall_s"], "s")


if __name__ == "__main__":
    main()
