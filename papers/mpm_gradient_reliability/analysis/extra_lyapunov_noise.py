"""추가 검증 1: 혼돈 롤아웃의 리아푸노프 지수와 f32·f64 기울기 잡음의 임계 스텝 (CPU numpy 1스레드, 수십 초).

질문: (a) 접선 ∂x_T/∂x_0의 로그 기울기로 잰 λ가 궤도 평균 리아푸노프 지수와 같은가?
      (b) 누적 순서 잡음이 만드는 기울기 상대 산포가 1이 되는 T*가 모형 ln(1/σ0)/λ와 맞는가?
모형 (03 문서 §2.3과 같은 설정, 단 접선은 초기값 x_0에 대한 것):
  - 사상 x_{k+1} = a·S_k(1 − S_k), a = 3.9, x_0 = 0.3. S_k는 N = 2000개 기여 w_j x_k의 합이다(Σw_j = 1).
  - 접선: d_{k+1} = a(1 − 2S_k)·dS_k, dS_k = Σ_j w_j d_k (∂x_T/∂x_0 = d_T).
  - 기준: f64, 인덱스 순서 고정 합. 잡음: 매 스텝 기여를 실행마다 다른 무작위 순서로 더한다(f32 또는 f64).
  - R = 400 실행. 상대 잡음 r_T = sd_R(d_T) / |d_T^ref|.
  - 모형 예측: σ0 = ε·√N (ε = 기계 엡실론: f32 1.19e-7, f64 2.2e-16), r_T ≈ σ0·e^{λT}, T* = ln(1/σ0)/λ.
    λ는 궤도 평균 리아푸노프 지수(0.4946)를 쓴다. σ0는 맞춰서 얻지 않고 ε·√N으로 정한다.
    기울기 적합은 잡음 바닥 이후(T ≥ 10)부터 T*의 0.8배까지.
  - 한계: 기여 N = 2000과 가중치 w는 가정이다. 실제 MPM의 누적 구조와 1:1 대응이 아니다.
실행: python3 analysis/extra_lyapunov_noise.py -> analysis/extra_lyapunov_noise_results.json
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A0, X0 = 3.9, 0.3
N, R, TMAX = 2000, 400, 80
W64 = np.random.default_rng(3).uniform(0.5, 1.5, N)
W64 = W64 / W64.sum()


def lam_from_tangent_and_orbit():
    """(a) f64 고정 순서 접선 d_T = ∂x_T/∂x_0 의 로그 기울기 λ_tan, 궤도 평균 λ_orb."""
    x, d = X0, 1.0
    logd, lyap = [], 0.0
    for T in range(1, 81):
        S = float(np.sum(x * W64))
        dS = float(np.sum(d * W64))
        d = A0 * (1 - 2 * S) * dS
        lyap += np.log(abs(A0 * (1 - 2 * S)))
        x = A0 * S * (1 - S)
        logd.append(np.log(abs(d)))
    Ts = np.arange(1, 81)
    sel = (Ts >= 5) & (Ts <= 60)
    slope = np.polyfit(Ts[sel], np.array(logd)[sel], 1)[0]
    # 궤도 평균 리아푸노프 지수 (같은 궤도, 긴 구간)
    x = 0.1234
    s = 0.0
    n = 20000
    for _ in range(n):
        s += np.log(abs(A0 * (1 - 2 * x)) + 1e-300)
        x = A0 * x * (1 - x)
    return float(slope), float(s / n)


def ref_tangent(Tmax):
    x, d = X0, 1.0
    dref, xref = [], []
    for _ in range(Tmax):
        S = float(np.sum(x * W64))
        dS = float(np.sum(d * W64))
        d = A0 * (1 - 2 * S) * dS
        x = A0 * S * (1 - S)
        dref.append(d)
        xref.append(x)
    return np.array(dref)


def reduce_orders(vals, perm, dtype):
    acc = np.zeros(perm.shape[0], dtype=dtype)
    rows = np.arange(perm.shape[0])
    for j in range(N):
        acc += vals[rows, perm[:, j]]
    return acc


def noisy_tangent(dtype, rng, Tmax):
    """R 실행을 동시에 돌린다. 반환: (Tmax, R) 접선 d_T."""
    Wd = W64.astype(dtype)
    x = np.full(R, X0, dtype=dtype)
    d = np.ones(R, dtype=dtype)
    a = dtype(A0)
    out = np.empty((Tmax, R))
    one = dtype(1)
    two = dtype(2)
    for k in range(Tmax):
        perm1 = np.argsort(rng.random((R, N)), axis=1)
        S = reduce_orders(x[:, None] * Wd[perm1], perm1, dtype)
        perm2 = np.argsort(rng.random((R, N)), axis=1)
        dS = reduce_orders(d[:, None] * Wd[perm2], perm2, dtype)
        d = (a * (one - two * S) * dS).astype(dtype)
        x = (a * S * (one - S)).astype(dtype)
        out[k] = d.astype(np.float64)
    return out


def threshold_T(r, Ts):
    """r_T가 1을 처음 넘는 T를 로그 선형 보간으로 구한다. 없으면 None."""
    idx = np.where(r >= 1.0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    if i == 0:
        return float(Ts[0])
    l0, l1 = np.log(r[i - 1]), np.log(r[i])
    frac = (0 - l0) / (l1 - l0)
    return float(Ts[i - 1] + frac * (Ts[i] - Ts[i - 1]))


def main():
    t_all = time.perf_counter()
    lam_tan, lam_orb = lam_from_tangent_and_orbit()
    dref = ref_tangent(TMAX)
    Ts = np.arange(1, TMAX + 1)
    out = {"setup": dict(a=A0, x0=X0, N_contrib=N, R=R, Tmax=TMAX, sigma0_model="eps*sqrt(N)"),
           "lyapunov": {"lambda_from_tangent_fit_T5_60": lam_tan, "lambda_orbit_mean": lam_orb,
                        "ratio_tangent_over_orbit": lam_tan / lam_orb}}
    rng = np.random.default_rng(20261111)
    for name, dtype, eps in (("f32", np.float32, float(np.finfo(np.float32).eps)),
                             ("f64", np.float64, float(np.finfo(np.float64).eps))):
        nt = noisy_tangent(dtype, rng, TMAX)
        mean = nt.mean(axis=1)
        sd = nt.std(axis=1, ddof=1)
        rel = sd / np.abs(dref)
        T_meas = threshold_T(rel, Ts)
        # 선형(지수) 구간: 상대 산포가 0.2 미만이면서 1e-12 이상인 점으로 기울기 적합
        # 기울기 적합 구간: 잡음 바닥(T < 10) 이후부터 T*의 0.8배까지 (지수 구간)
        hi = 0.8 * T_meas if T_meas is not None else TMAX
        sel = (Ts >= 10) & (Ts <= hi)
        slope = float(np.polyfit(Ts[sel], np.log(rel[sel]), 1)[0]) if sel.sum() >= 3 else None
        sigma0_model = eps * np.sqrt(N)
        T_model = float(np.log(1.0 / sigma0_model) / lam_orb)
        out[name] = {
            "eps": eps,
            "sigma0_model": sigma0_model,
            "T_star_measured_rel_sd_eq_1": T_meas,
            "T_star_model_ln(1/sigma0)/lambda": T_model,
            "growth_rate_fit_log_rel_sd_per_step": slope,
            "growth_rate_model_lambda_orbit": lam_orb,
            "fit_window_T": [10, float(hi)],
            "table": [
                {"T": int(T), "rel_sd": float(rel[T - 1]), "ref_dT": float(dref[T - 1]),
                 "mean_dT_noisy": float(mean[T - 1]), "bias_rel": float((mean[T - 1] - dref[T - 1]) / abs(dref[T - 1]))}
                for T in (1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 50, 60, 65, 70, 80)
            ],
        }
    out["total_wall_s"] = round(time.perf_counter() - t_all, 2)
    (HERE / "extra_lyapunov_noise_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"lambda tangent fit (T 5-60) = {lam_tan:.4f}, orbit mean = {lam_orb:.4f} (model uses orbit mean)")
    for name in ("f32", "f64"):
        o = out[name]
        print(f"{name}: T* measured = {o['T_star_measured_rel_sd_eq_1']}, model = {o['T_star_model_ln(1/sigma0)/lambda']:.1f}, "
              f"fitted growth rate = {o['growth_rate_fit_log_rel_sd_per_step']}")
        for row in o["table"]:
            print(f"   T={row['T']:>3} rel_sd={row['rel_sd']:.3e}")
    print("wall", out["total_wall_s"], "s")


if __name__ == "__main__":
    main()
