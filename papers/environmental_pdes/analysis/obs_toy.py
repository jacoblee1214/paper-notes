"""관측가능성 toy (CPU, numpy 1스레드): pde_toy.py와 같은 장(참값 80×80, κ = 0.2)과 같은 경로를 쓴다.

(1) 자세(x0, y0, θ0) 피셔 정보행렬: 편향(상수 b)을 알 때와 모를 때, 장을 t = 0에서 고정했을 때의 고유값과
    최소 고유벡터가 방사 대칭의 null 방향 (rot90(x0 − c), 1)과 맞는지 확인한다.
(2) 대칭 시험: 경로를 중심(10, 10) 둘레로 θ만큼 회전했을 때 예측 측정값의 차이
    D(θ) = Σ_t (φ(R_θ p_t, t) − φ(p_t, t))² / σ²  (잡음 없는 판별력). 방사 대칭 장에서는 D ≈ 0이어야 한다.
(3) 평행이동 가설: D(Δ) 의 95% 등고선(χ²₂ = 5.99) 안의 면적. 편향을 미지수로 두면 평균 잔차를 빼고 D를 다시 계산한다.
실행: python3 -I analysis/obs_toy.py → analysis/obs_toy_results.json
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pde_toy as pt  # noqa: E402

SIG = 0.05
CENTER = np.array([pt.L / 2, pt.L / 2])
H = 0.25  # 중심차분 간격 [m]


def grad_at(field_t, p):
    gx = (pt.sample(field_t, p + [H, 0]) - pt.sample(field_t, p - [H, 0])) / (2 * H)
    gy = (pt.sample(field_t, p + [0, H]) - pt.sample(field_t, p - [0, H])) / (2 * H)
    return np.stack([gx, gy], axis=1)


def fim_summary(J):
    w = np.linalg.eigvalsh(J)
    return dict(eig=[float(x) for x in w], cond=float(w[-1] / max(w[0], 1e-30)))


def rotate_path(p, th):
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    return (p - CENTER) @ R.T + CENTER


def numerical_diffusion(v=0.25, t_end=20.0, cfl=0.5):
    """(4) 수치 확산: κ = 0인 순수 이류(풍상, 1차)에서 가우시안 분산 증가율로 κ_eff를 잰다. 이론값 v·dx/2·(1 − C)."""
    out = {}
    for n in (40, 80, 160):
        L = pt.L
        dx = L / n
        xs = (np.arange(n) + 0.5) * dx
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        f = np.exp(-((X - 6.0) ** 2 + (Y - 10.0) ** 2) / (2 * pt.W0**2))
        dt = cfl * dx / v
        steps = int(round(t_end / dt))
        dt = t_end / steps
        def var_x(a):
            m = a.sum()
            mx = (a * X).sum() / m
            return float((a * (X - mx) ** 2).sum() / m)
        v0 = var_x(f)
        for _ in range(steps):
            f = f - dt * pt.upwind(f, v, 0.0, dx)
        v1 = var_x(f)
        k_eff = (v1 - v0) / (2 * t_end)
        out[f"n{n}"] = dict(dx=dx, kappa_eff_measured=round(k_eff, 5),
                            kappa_eff_theory=round(v * dx / 2 * (1 - v * dt / dx), 5),
                            courant=round(v * dt / dx, 3))
    return out


def main():
    p_path, _ = pt.path_disp()
    res = {}
    fields = {}
    for scen in ("asym", "sym"):
        fields[scen], _ = pt.solve(pt.N_TRUE, pt.KAPPA, scen)

    # (1) 자세 피셔 정보: ξ = (x0, y0, θ0), 상대 경로는 주행기록계로 알고 있다고 가정
    #     p_t(ξ) = x0 + R(θ0)(p_t − p_0), ∂y_t/∂ξ = [∇φ, ∇φ·rot90(p_t − p_0), (1 if 편향)]
    for scen in ("asym", "sym"):
        F = fields[scen]
        x0 = p_path[0]
        rel = p_path[1:] - x0
        rot = np.array([[0.0, -1.0], [1.0, 0.0]])
        for bias in (False, True):
            J = np.zeros((4 if bias else 3,) * 2)
            gr2 = []
            for t in range(1, pt.T + 1):
                g = grad_at(F[t], p_path[t:t + 1])[0]
                u = rot @ rel[t - 1]
                h = [g[0], g[1], g @ u] + ([1.0] if bias else [])
                h = np.array(h)
                J += np.outer(h, h) / SIG**2
                gr2.append(float(g @ g))
            w, V = np.linalg.eigh(J)
            v_min = V[:, 0]
            # 방사 대칭의 정확한 null 방향: 전체 경로를 중심 c 둘레로 회전 → (rot90(x0 − c), 1)
            v_sym = np.append(rot @ (x0 - CENTER), [1.0] + ([0.0] if bias else []))
            v_sym = v_sym / np.linalg.norm(v_sym)
            key = f"pose_fim_{scen}_{'bias_unknown' if bias else 'bias_known'}"
            res[key] = dict(eig=[float(x) for x in w], cond=float(w[-1] / max(w[0], 1e-30)),
                            cos_min_eigvec_vs_rotation_null=float(abs(v_min @ v_sym)),
                            grad_rms=float(np.sqrt(np.mean(gr2))))

    # (1b) 장을 t = 0 으로 고정했을 때(시간 변화 없음)의 자세 정보
    for scen in ("asym", "sym"):
        F = fields[scen]
        x0 = p_path[0]
        rel = p_path[1:] - x0
        rot = np.array([[0.0, -1.0], [1.0, 0.0]])
        J = np.zeros((3, 3))
        for t in range(1, pt.T + 1):
            g = grad_at(F[0], p_path[t:t + 1])[0]
            h = np.array([g[0], g[1], g @ (rot @ rel[t - 1])])
            J += np.outer(h, h) / SIG**2
        w = np.linalg.eigvalsh(J)
        res[f"pose_fim_{scen}_frozen_t0_bias_known"] = dict(eig=[float(x) for x in w],
                                                            cond=float(w[-1] / max(w[0], 1e-30)))

    # (2) 회전 대칭 시험
    thetas = np.linspace(-np.pi / 2, np.pi / 2, 81)
    for scen in ("asym", "sym"):
        F = fields[scen]
        D = []
        for th in thetas:
            pr = rotate_path(p_path, th)
            diff = np.array([pt.sample(F[t], pr[t:t + 1])[0] - pt.sample(F[t], p_path[t:t + 1])[0]
                             for t in range(1, pt.T + 1)])
            D.append(float(np.sum(diff**2) / SIG**2))
        D = np.array(D)
        inside = thetas[D < 5.99]
        res[f"rotation_{scen}"] = dict(
            D_max=float(D.max()), D_at_pm30deg=float(D[np.argmin(np.abs(thetas - np.deg2rad(30)))]),
            plausible_arc_deg=float(np.rad2deg(inside.max() - inside.min())) if inside.size else 0.0,
            D_curve=[round(x, 4) for x in D.tolist()],
        )

    # (3) 평행이동 가설과 편향 (미지수일 때 평균 잔차 제거)
    grid = np.arange(-4.0, 4.0 + 1e-9, 0.25)
    for scen in ("asym", "sym"):
        F = fields[scen]
        y_true = np.array([pt.sample(F[t], p_path[t:t + 1])[0] for t in range(1, pt.T + 1)])
        for bias_mode in ("known", "unknown"):
            Dg = np.zeros((grid.size, grid.size))
            for i, dx_ in enumerate(grid):
                for j, dy_ in enumerate(grid):
                    sh = p_path + [dx_, dy_]
                    pred = np.array([pt.sample(F[t], sh[t:t + 1])[0] for t in range(1, pt.T + 1)])
                    r = y_true - pred
                    if bias_mode == "unknown":
                        r = r - r.mean()
                    Dg[i, j] = np.sum(r**2) / SIG**2
            plaus = Dg < 5.99
            res[f"translation_{scen}_bias_{bias_mode}"] = dict(
                plausible_area_m2=float(plaus.sum() * 0.25**2),
                plausible_cells=int(plaus.sum()),
                min_D=float(Dg.min()),
                argmin_shift_m=[float(grid[np.unravel_index(Dg.argmin(), Dg.shape)[0]]),
                                float(grid[np.unravel_index(Dg.argmin(), Dg.shape)[1]])],
            )

    res["numerical_diffusion"] = numerical_diffusion()

    out = dict(setup=dict(sigma=SIG, grid_true=pt.N_TRUE, kappa=pt.KAPPA, H_fd=H,
                          center=CENTER.tolist(), note="D(theta) 와 D(shift)는 잡음 없는 판별력, 단위 σ²"),
               results=res)
    (HERE / "obs_toy_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for k, v in res.items():
        if k.startswith("pose_fim"):
            print(k, {kk: (np.round(vv, 4).tolist() if isinstance(vv, list) else round(float(vv), 5)) for kk, vv in v.items()})
        elif k == "numerical_diffusion":
            print(k, v)
        elif k.startswith("rotation"):
            print(k, {kk: vv for kk, vv in v.items() if kk != "D_curve"})
        else:
            print(k, v)


if __name__ == "__main__":
    main()
