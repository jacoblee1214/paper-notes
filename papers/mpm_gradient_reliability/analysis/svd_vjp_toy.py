"""SVD 역전파(VJP)의 고유값 간격 민감도 toy (CPU numpy, 1스레드, 수 초 이내).

논문 Appendix B는 Taichi SVD에 생성된 adjoint가 없어서 손으로 쓴 SVD VJP를 쓴다고 한다. 이 VJP에는 F_ij = 1/(s_j² - s_i²)가 들어간다.
특이값이 가까워지면(간격 g → 0) F가 1/g로 커진다. 손실이 게이지 불변(U, V를 같이 회전해도 값이 같음)이면
해석적으로 antisymmetric 항 (UᵀŪ - ŪᵀU)는 0이어야 하는데, 유한 정밀도에서는 eps 크기의 잔차가 남고 F가 이를 1/g배 키운다.

이 toy는 다음을 본다.
  1) 손실 L(A) = Σ W ∘ (U Vᵀ)(극분해 인자, 게이지 불변)에 대해 해석적 VJP 식을 쓰고, f64 중심 차분(방향 미분)과 맞는지 확인한다.
  2) 간격 g = 1e-1 ... 1e-8에서 f32 SVD + 같은 VJP의 상대오차를 f64 FD와 비교한다. 오차가 eps/g로 자라는지 본다.
  3) 게이지 불변 손실에 대해 antisymmetric 항을 빼면(해석적으로 0이라는 사실을 사용) 1/g 증폭이 사라지는지 본다.

실행: python3 analysis/svd_vjp_toy.py → analysis/svd_vjp_results.json
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(5)
N_TRIALS = 20
GAPS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def loss(A, W):
    """극분해 인자 U Vᵀ 에 대한 선형 손실 (게이지 불변)."""
    U, S, Vt = np.linalg.svd(A)
    return float(np.sum(W * (U @ Vt)))


def vjp_svd(A, W, safe=False):
    """L = Σ W ∘ (U Vᵀ) 의 A에 대한 기울기. 정사각 A(gS = 0)에서 Townsend 식을 쓴다.

    safe=False: 일반 식 gA = U[(F∘(UᵀḠU-ḠᵀU))Σ + Σ(F∘(VᵀV̄-V̄ᵀV))]Vᵀ, F_ij = 1/(s_j²-s_i²).
    safe=True : 게이지 불변 손실(L이 UVᵀ만의 함수)에서는 두 antisymmetric 블록이 B = -A를 만족한다(수치로 확인).
                그러면 F가 상쇄되어 gA = U (A ∘ G) Vᵀ, G_ij = 1/(s_i + s_j). 간격에 대한 1/g 특이성이 없다.
    """
    U, S, Vt = np.linalg.svd(A)
    V = Vt.T
    gU = W @ V
    gV = W.T @ U
    Am = U.T @ gU - gU.T @ U
    if safe:
        G = 1.0 / (S[:, None] + S[None, :])
        np.fill_diagonal(G, 0.0)
        return U @ (Am * G) @ Vt
    Bm = V.T @ gV - gV.T @ V
    s2 = S ** 2
    F = 1.0 / (s2[None, :] - s2[:, None])
    np.fill_diagonal(F, 0.0)
    J = F * Am
    K = F * Bm
    return U @ (J @ np.diag(S) + np.diag(S) @ K) @ Vt


def fd_direction(A, W, E, h=1e-6):
    return (loss(A + h * E, W) - loss(A - h * E, W)) / (2 * h)


def make_A(gap, rs):
    s = np.array([1.0, 1.0 - gap])
    return rot(rs[0]) @ np.diag(s) @ rot(rs[1]).T


def check_formula():
    errs = []
    for _ in range(N_TRIALS):
        A = make_A(0.5, rng.uniform(0, 2 * np.pi, 2))
        W = rng.normal(size=(2, 2))
        E = rng.normal(size=(2, 2))
        ad = float(np.sum(vjp_svd(A, W) * E))
        fd = fd_direction(A, W, E)
        errs.append(abs(ad - fd) / abs(fd))
    return float(np.max(errs))


def sweep(dtype, safe):
    rows = []
    np.seterr(divide="ignore", invalid="ignore")  # 간격 1e-8, f32 naive에서 F가 무한대가 된다(NaN으로 보고)
    for gap in GAPS:
        errs = []
        for _ in range(N_TRIALS):
            A64 = make_A(gap, rng.uniform(0, 2 * np.pi, 2))
            W = rng.normal(size=(2, 2))
            E = rng.normal(size=(2, 2))
            ref = fd_direction(A64, W, E)  # 기준: f64 중심 차분
            gA = vjp_svd(A64.astype(dtype), W.astype(dtype), safe=safe)
            ad = float(np.sum(gA.astype(np.float64) * E))
            errs.append(abs(ad - ref) / abs(ref))
        rows.append({"gap": gap, "median_relerr": float(np.median(errs)), "max_relerr": float(np.max(errs))})
    return rows


def main():
    out = {"formula_check_max_relerr_f64_gap0.5": check_formula(), "sweeps": {}}
    for label, dtype, safe in (("f64_naive", np.float64, False),
                               ("f32_naive", np.float32, False),
                               ("f64_safe", np.float64, True),
                               ("f32_safe", np.float32, True)):
        out["sweeps"][label] = sweep(dtype, safe)
    (HERE / "svd_vjp_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"formula check (f64, gap 0.5, max rel err over {N_TRIALS} trials): {out['formula_check_max_relerr_f64_gap0.5']:.2e}")
    print(f"{'gap':>8} | {'f64 naive':>10} | {'f32 naive':>10} | {'f64 safe':>10} | {'f32 safe':>10}   (median rel. err vs f64 FD)")
    for i, gap in enumerate(GAPS):
        v = [out["sweeps"][k][i]["median_relerr"] for k in ("f64_naive", "f32_naive", "f64_safe", "f32_safe")]
        print(f"{gap:>8.0e} | " + " | ".join(f"{x:>10.2e}" for x in v))


if __name__ == "__main__":
    main()
