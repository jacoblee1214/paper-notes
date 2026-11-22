"""토이: 쌓인 상자의 충돌 임펄스 LCP를 PGS와 PADMM(논문 Alg. 1 형태)으로 풀어 질량비에 대한 반복 횟수를 비교한다.

검증하려는 주장 (v2 §3.1, 부록 A.4 "Stacked Boxes"): 국소 PGS 계열 반복은 질량비가 크면 반복이 많이 필요하고,
전역 선형 풀이를 쓰는 PADMM은 질량비에 덜 민감하다.

설정: 상자 n = 5개를 수직으로 쌓고 전부 아래로 속도 -1로 떨어지는 순간(비탄성 충돌 한 스텝).
  접촉 k의 법선 속도 v_k = u_k - u_{k-1} (u_{-1} = 0, 지면). 비침투: v+ >= 0 ⊥ λ >= 0.
  D = J M^-1 J^T, v_f = J u^-. 마찰 없음(원뿔은 반직선) → de Saxcé 항은 0이다. (마찰 원뿔 효과는 이 토이에 없음)
  질량 분포: 위가 무거움(top), 아래가 무거움(bottom). 무거운 상자 질량 = R, 나머지 1.
PADMM의 벌칙 ρ는 논문이 값을 적지 않아(코드 기본 rho_0 = 1.0) ρ = 1, 1e-2, 1e-4를 모두 돌린다.
수렴 기준: 상대 잔차 max(r_primal, r_dual, r_cmpl)/(1+|λ|_inf) < 1e-6 (논문 Alg. 1 9행과 같은 세 잔차).
numpy 단일 프로세스, 약 10초.

실행: python3 analysis/toy_padmm_stack.py → analysis/toy_padmm_stack_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_padmm_stack_results.json"
N = 5
TOL = 1e-6
MAX_IT = 20000


def build(masses, dtype=np.float64):
    n = len(masses)
    J = np.zeros((n, n))
    for k in range(n):
        J[k, k] = 1.0
        if k > 0:
            J[k, k - 1] = -1.0
    Minv = np.diag(1.0 / np.asarray(masses, dtype=float))
    D = J @ Minv @ J.T
    vf = J @ (-np.ones(n))
    return D.astype(dtype), vf.astype(dtype)


def residual(D, vf, lam):
    v = D @ lam + vf
    rp = np.max(np.maximum(0.0, -lam))
    rd = np.max(np.maximum(0.0, -v))
    rc = np.max(np.abs(lam * v))
    return max(rp, rd, rc) / (1.0 + np.max(np.abs(lam)))


def pgs(D, vf, max_it=MAX_IT, order="up"):
    n = len(vf)
    lam = np.zeros(n)
    idx = range(n) if order == "up" else range(n - 1, -1, -1)
    for it in range(1, max_it + 1):
        for i in idx:
            r = D[i] @ lam + vf[i]
            lam[i] = max(0.0, lam[i] - r / D[i, i])
        if residual(D, vf, lam) < TOL:
            return it, lam
    return None, lam


def padmm(D, vf, eta=1e-5, rho=1.0, accel=True, max_it=MAX_IT, dtype=np.float64):
    """논문 Alg. 1: Jacobi 전처리 P = diag(D)^-1/2 후 x-갱신(선형 풀이) → y-사영 → z-갱신. 한 번 분해해서 재사용한다."""
    n = len(vf)
    P = 1.0 / np.sqrt(np.diag(D))
    Dt = (P[:, None] * D * P[None, :]).astype(dtype)
    vt = (P * vf).astype(dtype)
    A = Dt + (eta + rho) * np.eye(n, dtype=dtype)
    Ainv = np.linalg.inv(A).astype(dtype)  # 분해 한 번 (토이이므로 역행렬)
    x = np.zeros(n, dtype); y = np.zeros(n, dtype); z = np.zeros(n, dtype)
    yh, zh = y.copy(), z.copy()
    a, r_prev = 1.0, np.inf
    rho_ = dtype(rho); eta_ = dtype(eta)
    for it in range(1, max_it + 1):
        rhs = -vt + eta_ * x + rho_ * yh + zh  # Alg. 1의 5행 (부호는 코드/유도와 같은 형태)
        xn = Ainv @ rhs
        yn = np.maximum(0.0, xn - zh / rho_)  # 6행: 반직선(R+) 사영
        zn = zh - rho_ * (xn - yn)  # 7행
        if accel:
            r_a = rho_ * np.sum((yn - yh) ** 2) + np.sum((zn - zh) ** 2) / rho_
            if r_a < 0.999 * r_prev:
                an = (1 + np.sqrt(1 + 4 * a * a)) / 2
                f = (a - 1) / an
                yh_n = yn + f * (yn - y)
                zh_n = zn + f * (zn - z)
                a = an
            else:  # 재시작
                a = 1.0
                yh_n, zh_n = yn.copy(), zn.copy()
                r_a = r_prev / 0.999
            r_prev = r_a
            yh, zh = yh_n, zh_n
        else:
            yh, zh = yn, zn
        x, y, z = xn, yn, zn
        lam = P * y.astype(np.float64)
        if residual(D, vf, lam) < TOL:
            return it, lam
    return None, P * y.astype(np.float64)


def main():
    out = {"settings": dict(n_boxes=N, tol=TOL, max_it=MAX_IT, eta=1e-5, rho=1.0), "cases": []}
    for dist in ("heavy_top", "heavy_bottom"):
        for R in (1.0, 10.0, 100.0, 1e4):
            m = np.ones(N)
            if dist == "heavy_top":
                m[-1] = R
            else:
                m[0] = R
            D, vf = build(m)
            lam_exact = np.array([m[k:].sum() for k in range(N)])  # 해석해: 위쪽 질량의 합
            row = {"dist": dist, "R": R, "lambda_exact": lam_exact.tolist()}
            for name, fn in (
                ("PGS_up", lambda: pgs(D, vf, order="up")),
                ("PGS_down", lambda: pgs(D, vf, order="down")),
                ("PADMM_plain_rho1_f64", lambda: padmm(D, vf, accel=False)),
                ("PADMM_nesterov_rho1_f64", lambda: padmm(D, vf, accel=True)),
                ("PADMM_nesterov_rho1e-2_f64", lambda: padmm(D, vf, rho=1e-2, accel=True)),
                ("PADMM_nesterov_rho1e-4_f64", lambda: padmm(D, vf, rho=1e-4, accel=True)),
                ("PADMM_nesterov_rho1e-2_f32", lambda: padmm(D, vf, rho=1e-2, accel=True, dtype=np.float32)),
            ):
                it, lam = fn()
                err = float(np.max(np.abs(lam - lam_exact)) / np.max(lam_exact))
                row[name] = {"iterations": it, "rel_err_vs_analytic": err}
            out["cases"].append(row)
            print(dist, R, {k: v["iterations"] for k, v in row.items() if isinstance(v, dict)})
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
