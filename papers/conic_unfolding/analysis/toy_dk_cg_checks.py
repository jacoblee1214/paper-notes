"""토이 T1: 논문 2.3절과 2.4절의 두 역전파 규칙을 작은 크기에서 직접 확인한다 (torch CPU, float64, 수 초).

공식 코드는 쓰지 않고, 논문 식만 보고 새로 짰다.
  (a) 식 9–10 (Daleckii–Krein, DK) 대 eigh를 통한 autograd 대 중심 차분: 고유값 간격 sweep (논문 그림 4 왼쪽과 같은 설정, n=40, float64)
  (a32) 같은 행렬을 float32로 뒤로 계산한 기울기와 float64 DK 기울기의 차이
  (b) S = c I (모든 고유값 같음)에서 autograd와 DK
  (c) 식 5–6 (행렬 없는 암묵 미분) 대 조밀 행렬 autograd: 상대 오차, CG 허용오차에 따른 오차
  (d) 식 11 (분모 정규화)의 편향: 간격 sweep

실행: python3 analysis/toy_dk_cg_checks.py   → analysis/toy_dk_cg_checks_results.json
"""
import json
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
OUT = Path(__file__).resolve().parent / "toy_dk_cg_checks_results.json"
R = {}


# ------------------------------------------------------------------ PSD 투영과 역전파 규칙
class PsdDK(torch.autograd.Function):
    """식 7을 앞으로 계산하고, 뒤로는 식 9–10 (Loewner 행렬)을 쓴다. f(x)=max(x,0), f'(0)=0."""

    @staticmethod
    def forward(ctx, S):
        lam, U = torch.linalg.eigh(S)
        ctx.save_for_backward(lam, U)
        return U @ torch.diag_embed(torch.clamp(lam, min=0)) @ U.mT

    @staticmethod
    def backward(ctx, G):
        lam, U = ctx.saved_tensors
        G = 0.5 * (G + G.mT)
        f = torch.clamp(lam, min=0)
        d = lam.unsqueeze(-2) - lam.unsqueeze(-1)  # d_ij = lam_j - lam_i
        same = d == 0
        L = torch.where(same, (lam.unsqueeze(-1) > 0).to(d.dtype), (f.unsqueeze(-2) - f.unsqueeze(-1)) / torch.where(same, torch.ones_like(d), d))
        return U @ (L * (U.mT @ G @ U)) @ U.mT


def proj_autograd(S):
    lam, U = torch.linalg.eigh(S)
    return U @ torch.diag_embed(torch.clamp(lam, min=0)) @ U.mT


class PsdDenomReg(torch.autograd.Function):
    """식 8에서 F_ij = 1/(lam_j - lam_i)를 식 11로 바꾼 규칙 (eps)."""

    @staticmethod
    def forward(ctx, S, eps):
        lam, U = torch.linalg.eigh(S)
        ctx.save_for_backward(lam, U)
        ctx.eps = eps
        return U @ torch.diag_embed(torch.clamp(lam, min=0)) @ U.mT

    @staticmethod
    def backward(ctx, G):
        lam, U = ctx.saved_tensors
        G = 0.5 * (G + G.mT)
        d = lam.unsqueeze(-2) - lam.unsqueeze(-1)
        F = d / (d * d + ctx.eps ** 2)
        F = F * (1 - torch.eye(lam.shape[-1]))
        # P = U f(L) U^T, Lbar = diag(f'(lam) * (U^T G U)_ii)
        inner = U.mT @ G @ U
        fp = (lam > 0).to(lam.dtype)
        Lbar = torch.diag_embed(fp * torch.diagonal(inner, dim1=-2, dim2=-1))
        # 아래 항은 식 8과 같은 구조: S_bar = U (Lbar + F * (U^T Ubar)) U^T, 여기서 Ubar = 2 G U f(L)
        Ubar = 2 * G @ U @ torch.diag_embed(torch.clamp(lam, min=0))
        return U @ (Lbar + F * (U.mT @ Ubar)) @ U.mT, None


def clustered(n, gap, seed):
    """식 14: 중심 -gap/2, +gap/2 근처에 gap*1e-3 폭으로 모은 고유값."""
    g = np.random.default_rng(seed)
    sig = gap * 1e-3
    lam = np.concatenate([-gap / 2 + g.uniform(-sig, sig, n // 2), gap / 2 + g.uniform(-sig, sig, n // 2)])
    Q, _ = np.linalg.qr(g.standard_normal((n, n)))
    S = Q @ np.diag(lam) @ Q.T
    return torch.tensor(0.5 * (S + S.T))


def sym_rand(n, g):
    X = torch.tensor(g.standard_normal((n, n)))
    return 0.5 * (X + X.T)


def vjp(fn, S, G):
    S = S.clone().requires_grad_(True)
    (fn(S) * G).sum().backward()
    return S.grad


def fd_ref(S, G, V, h):
    """방향 V로 중심 차분한 <G, dPi(S)[V]>."""
    f = lambda T: (proj_autograd(T) * G).sum()
    return (f(S + h * V) - f(S - h * V)) / (2 * h)


def check_a():
    n, reps = 40, 20
    out = {}
    for gap in (1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
        errs = {"dk": [], "autograd": [], "denom_1e-3": [], "denom_1e-6": []}
        nan = {"dk": 0, "autograd": 0}
        for r in range(reps):
            g = np.random.default_rng(1000 + r)
            S = clustered(n, gap, seed=r)
            G, V = sym_rand(n, g), sym_rand(n, g)
            V = V / V.norm()
            h = min(1e-6, gap * 1e-3) if gap < 1e-3 else 1e-6
            ref = fd_ref(S, G, V, h)
            for name, fn in (("dk", PsdDK.apply), ("autograd", proj_autograd),
                             ("denom_1e-3", lambda T: PsdDenomReg.apply(T, 1e-3)),
                             ("denom_1e-6", lambda T: PsdDenomReg.apply(T, 1e-6))):
                grad = vjp(fn, S, G)
                if not torch.isfinite(grad).all():
                    nan[name] = nan.get(name, 0) + 1
                    continue
                errs[name].append(float(abs((grad * V).sum() - ref) / (abs(ref) + 1e-30)))
        out[f"{gap:.0e}"] = {k: {"median_rel_err": float(np.median(v)) if v else None, "n": len(v)} for k, v in errs.items()}
        out[f"{gap:.0e}"]["nonfinite"] = nan
    R["a_gap_sweep"] = out


def check_a32():
    """같은 클러스터 행렬에서 float32로 뒤로 계산한 기울기를 float64 DK 기울기와 비교한다 (정밀도 의존성).
    논문은 그림 4의 정밀도를 적지 않았다."""
    n, reps = 40, 10
    out = {}
    for gap in (1e0, 1e-2, 1e-4, 1e-6, 1e-8):
        res = {}
        for name, fn in (("dk", PsdDK.apply), ("autograd", proj_autograd)):
            e = []
            for r in range(reps):
                g = np.random.default_rng(r)
                S = clustered(n, gap, r)
                G = sym_rand(n, g)
                ref = vjp(PsdDK.apply, S, G)
                ref = 0.5 * (ref + ref.T)
                S32 = S.float().clone().requires_grad_(True)
                torch.set_default_dtype(torch.float32)
                (fn(S32) * G.float()).sum().backward()
                torch.set_default_dtype(torch.float64)
                gr = S32.grad.double()
                gr = 0.5 * (gr + gr.T)
                e.append(float((gr - ref).norm() / ref.norm()) if torch.isfinite(gr).all() else float("nan"))
            res[name] = float(np.nanmedian(e))
        out[f"{gap:.0e}"] = res
    R["a32_float32_vs_float64_reference"] = out


def check_b():
    n = 8
    g = np.random.default_rng(5)
    Q, _ = np.linalg.qr(g.standard_normal((n, n)))
    S = torch.tensor(2.0 * Q @ Q.T)
    G = sym_rand(n, g)
    gdk = vjp(PsdDK.apply, S, G)
    gau = vjp(proj_autograd, S, G)
    R["b_repeated_eigs"] = {"dk_finite": bool(torch.isfinite(gdk).all()),
                            "dk_equals_sym_G_maxabs_diff": float((gdk - 0.5 * (G + G.T)).abs().max()),
                            "autograd_finite": bool(torch.isfinite(gau).all()),
                            "autograd_max_abs": float(gau.abs().max()) if torch.isfinite(gau).all() else None}


# ------------------------------------------------------------------ 행렬 없는 암묵 미분 (식 5–6)
def cg(matvec, b, precond, tol_rel, max_iter=5000):
    x = torch.zeros_like(b)
    r = b - matvec(x)
    z = precond(r)
    p = z.clone()
    rz = r @ z
    rz0 = rz
    for it in range(max_iter):
        if rz <= (tol_rel ** 2) * rz0:
            break
        Ap = matvec(p)
        a = rz / (p @ Ap)
        x = x + a * p
        r = r - a * Ap
        z = precond(r)
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, it


class MatrixFreeSolve(torch.autograd.Function):
    """M(rho) x = xi, M = P + sigma I + A^T diag(rho) A. rho에 대한 기울기는 식 6, xi에 대한 기울기는 -lambda."""

    @staticmethod
    def forward(ctx, rho, xi, P, A, sigma, tol_f, tol_b):
        mv = lambda v: P @ v + sigma * v + A.T @ (rho * (A @ v))
        diag = torch.diagonal(P) + sigma + (A ** 2).T @ rho
        x, it = cg(mv, xi, lambda r: r / diag, tol_f)
        ctx.save_for_backward(rho, x, P, A, diag)
        ctx.sigma, ctx.tol_b = sigma, tol_b
        return x

    @staticmethod
    def backward(ctx, gx):
        rho, x, P, A, diag = ctx.saved_tensors
        mv = lambda v: P @ v + ctx.sigma * v + A.T @ (rho * (A @ v))
        lam, _ = cg(mv, -gx, lambda r: r / diag, ctx.tol_b)
        return (A @ lam) * (A @ x), -lam, None, None, None, None, None


def check_c():
    g = np.random.default_rng(11)
    n, m = 300, 150
    A = torch.tensor(g.standard_normal((m, n)) * (g.random((m, n)) < 0.05))
    P = torch.diag(torch.tensor(g.uniform(0, 10, n)))
    sigma = 1e-6
    rho0 = torch.tensor(g.uniform(0.1, 1, m))
    q = torch.tensor(g.standard_normal(n))
    b = torch.tensor(g.standard_normal(m))
    s = torch.tensor(g.standard_normal(m))
    y = torch.tensor(g.standard_normal(m))
    xs = torch.tensor(g.standard_normal(n))
    target = torch.tensor(g.standard_normal(n))

    def xi_fn(rho):  # xi(rho) = sigma x - q + A^T (y + rho (b - s))
        return sigma * xs - q + A.T @ (y + rho * (b - s))

    def dense(rho):
        rho = rho.clone().requires_grad_(True)
        M = P + sigma * torch.eye(n) + A.T @ torch.diag(rho) @ A
        x = torch.linalg.solve(M, xi_fn(rho))
        loss = 0.5 * ((x - target) ** 2).sum()
        loss.backward()
        return rho.grad, float(loss.detach())

    gd, _ = dense(rho0)
    out = {}
    for tol_f, tol_b in ((1e-4, 1e-4), (1e-8, 1e-6), (1e-10, 1e-10)):
        rho = rho0.clone().requires_grad_(True)
        x = MatrixFreeSolve.apply(rho, xi_fn(rho), P, A, sigma, tol_f, tol_b)
        loss = 0.5 * ((x - target) ** 2).sum()
        loss.backward()
        out[f"tol_f={tol_f:.0e},tol_b={tol_b:.0e}"] = float((rho.grad - gd).norm() / gd.norm())
    R["c_matrix_free_vs_dense_rel_err"] = {"n": n, "m": m, "density": 0.05, "errors": out}
    # 메모리: 조밀 M은 n^2, 행렬 없는 규칙은 A와 벡터만
    R["c_storage_floats"] = {"dense_M": n * n, "matrix_free_A_nnz": int((A != 0).sum()) + n + m}


def check_d():
    """식 11의 편향: 간격이 eps보다 작을 때."""
    n = 40
    out = {}
    for gap in (1e-1, 1e-3, 1e-5):
        errs = {}
        for eps in (1e-3, 1e-6):
            e = []
            for r in range(10):
                g = np.random.default_rng(500 + r)
                S = clustered(n, gap, seed=r)
                G, V = sym_rand(n, g), sym_rand(n, g)
                V = V / V.norm()
                ref = fd_ref(S, G, V, min(1e-6, gap * 1e-3))
                grad = vjp(lambda T: PsdDenomReg.apply(T, eps), S, G)
                e.append(float(abs((grad * V).sum() - ref) / (abs(ref) + 1e-30)))
            errs[f"eps={eps:.0e}"] = float(np.median(e))
        out[f"gap={gap:.0e}"] = errs
    R["d_denominator_reg_bias"] = out


def main():
    check_a()
    check_a32()
    check_b()
    check_c()
    check_d()
    OUT.write_text(json.dumps(R, indent=1), encoding="utf-8")
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
