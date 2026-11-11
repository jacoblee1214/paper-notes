"""병렬 누적 순서 toy (CPU numpy, 1스레드, 수십 초 이내).

논문 §III-A의 주장("GPU 원자 덧셈 순서가 긴 롤아웃의 기울기 부호를 바꾼다")을 작은 모형에서 재현하고,
논문에 빠진 것(앙상블 분산)을 같이 보고한다.

모형: 매 스텝 N = 2000개 입자가 기여 c_j = x_k · w_j (Σ w_j = 1)를 한 격자 항목에 더한다.
  - 정방향: 합 S_k를 float32로, 실행마다 다른 무작위 순서(원자 덧셈의 도착 순서 모사)로 순차 합산.
  - 역방향(접선): dS_k를 또 다른 무작위 순서로 합산 (역전파의 누적 순서는 정방향과 독립이라고 가정).
  - 상태 갱신: x_{k+1} = a · S_k (1 - S_k), 접선 d_{k+1} = S_k(1-S_k) + a(1-2S_k)·dS_k.
  - 손실 두 가지:
      (A) L = x_T                      : 기울기가 크다(부호가 안정).
      (B) L = (x_T - c)^2, c = x_T^ref + δ : 기준 실행에서 기울기가 0 근처(최적화 중 흔한 상황)
  - 기준: float64, 고정 순서(인덱스 순) 합산. 논문의 "결정론 CPU 기준"에 대응.
  - 각 T에서 R = 400번 실행의 기울기 평균, 표준편차, 기준과 부호가 다른 비율을 본다.

이 모형의 a, x0, N, w, δ는 가정이다. 논문의 값과 직접 비교하지 않고, 메커니즘(누적 순서 → 기울기 분산 → 부호 뒤집힘)을 보는 용도다.
실행: python3 analysis/reduction_order_toy.py → analysis/reduction_order_results.json
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A0, X0 = 3.9, 0.3
N, R = 2000, 400
TS = (2, 5, 10, 20, 30, 40)
DELTA = 1e-3


def weights(seed=3):
    w = np.random.default_rng(seed).uniform(0.5, 1.5, N)
    return w / w.sum()


W64 = weights()
W32 = W64.astype(np.float32)


def reduce_orders(vals, perm):
    """vals: (R,) 또는 (R,N) 기여 행렬, perm: (R,N) 실행별 순서. float32 순차 합산."""
    acc = np.zeros(perm.shape[0], dtype=np.float32)
    for j in range(N):
        acc += vals[np.arange(perm.shape[0]), perm[:, j]]
    return acc


def run_ref(T, loss):
    x, d = X0, 0.0
    for _ in range(T):
        S = float(np.sum(x * W64))  # 고정 순서 f64
        dS = float(np.sum(d * W64))
        d = S * (1 - S) + A0 * (1 - 2 * S) * dS
        x = A0 * S * (1 - S)
    if loss == "A":
        return d
    return 2.0 * (x - CTARGET[0]) * d


CTARGET = [None]  # 손실 B의 목표값 c = x_T^ref + δ (main에서 설정)


def run_random(T, rng, loss):
    x = np.full(R, X0, dtype=np.float32)
    d = np.zeros(R, dtype=np.float32)
    a = np.float32(A0)
    for _ in range(T):
        perm1 = np.argsort(rng.random((R, N)), axis=1)  # 정방향 도착 순서
        perm2 = np.argsort(rng.random((R, N)), axis=1)  # 역방향 도착 순서
        C = x[:, None] * W32[perm1]                    # 기여값(실행마다 다름)
        S = reduce_orders(C, perm1)
        dC = d[:, None] * W32[perm2]
        dS = reduce_orders(dC, perm2)
        d = (S * (np.float32(1) - S) + a * (np.float32(1) - np.float32(2) * S) * dS).astype(np.float32)
        x = (a * S * (np.float32(1) - S)).astype(np.float32)
    if loss == "A":
        return d
    return 2.0 * (x.astype(np.float64) - CTARGET[0]) * d.astype(np.float64)


def main():
    out = {"N": N, "R": R, "a0": A0, "x0": X0, "delta_B": DELTA, "rows": []}
    rng = np.random.default_rng(11)
    for T in TS:
        for loss in ("A", "B"):
            if loss == "B":
                # 기준 x_T를 구하고 c = x_T^ref + δ 로 둔다
                xr = X0
                for _ in range(T):
                    S = float(np.sum(xr * W64))
                    xr = A0 * S * (1 - S)
                CTARGET[0] = xr + DELTA
            g_ref = run_ref(T, loss)
            g = run_random(T, rng, loss).astype(np.float64)
            mean, sd = g.mean(), g.std(ddof=1)
            flip = float(np.mean(np.sign(g) != np.sign(g_ref)))
            row = {
                "T": T, "loss": loss,
                "ref_f64_fixed_order": g_ref,
                "f32_random_mean": mean, "f32_random_sd": sd,
                "f32_min": float(g.min()), "f32_max": float(g.max()),
                "sign_flip_fraction_vs_ref": flip,
                "mean_over_sd": mean / sd if sd > 0 else None,
            }
            out["rows"].append(row)
    (HERE / "reduction_order_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    print(f"N={N} contributions per reduction, R={R} runs, a0={A0}, x0={X0}")
    print(f"{'T':>3} {'loss':>4} {'ref(f64)':>12} {'f32 mean':>12} {'f32 sd':>11} {'mean/sd':>8} {'sign flip':>10}")
    for r in out["rows"]:
        mo = r["mean_over_sd"]
        mo_s = f"{mo:8.2f}" if mo is not None else "     n/a"
        print(f"{r['T']:>3} {r['loss']:>4} {r['ref_f64_fixed_order']:>12.4e} {r['f32_random_mean']:>12.4e} "
              f"{r['f32_random_sd']:>11.3e} {mo_s} {r['sign_flip_fraction_vs_ref']:>10.3f}")


if __name__ == "__main__":
    main()
