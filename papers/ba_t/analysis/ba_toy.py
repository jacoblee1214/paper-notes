"""두 시점 bundle adjustment toy (CPU, numpy, 단일 스레드). 실행: python3 analysis/ba_toy.py

논문 3.1절(식 1–3)과 3.3절(카메라 먼저 → geometry), 공개 코드의 순서(geometry 먼저 → 카메라),
부록 F.2의 "포즈는 N개 잔차를 모아 쓰므로 geometry보다 잘 좋아진다"를 고전 BA로 확인한다.

문제: 시점 a의 광선 방향 f_i(고정)와 역깊이 ρ_i(= local geometry), 상대 포즈 (R, t)(= camera).
잔차: 시점 b 관측 u_i와 π(R·f_i/ρ_i + t)의 차(2N개) + 단안 깊이 사전 w·(ρ_i − ρ̂_i)(N개).
깊이 사전이 스케일을 고정한다(인코더의 단안 geometry 사전에 해당, 부록 F.2).

- T1: Schur 보완(식 3)으로 푼 Δ가 전체 정규방정식 해와 같은지
- T2: 갱신 방식 비교 — joint(Schur) / 카메라 먼저(논문 순서) / geometry 먼저(코드 순서), 기선·초기 오차별
- T2b: 세 방식을 200회 돌렸을 때 같은 해(고정점)로 가는지
- T3: 대응점 수 N에 따른 최종 포즈 오차와 깊이 오차(차원 논증)
주의: BA-T는 정확한 블록 풀이가 아니라 학습된 갱신이다. 이 toy는 고전 BA 쪽 사실만 확인한다.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def expm_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3) + hat(w)
    K = hat(w / th)
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def rot_angle_deg(R):
    return math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def make_scene(rng, n, baseline, depth=(1.5, 4.0), fov=0.6):
    xy = rng.uniform(-fov, fov, size=(n, 2))
    f = np.c_[xy, np.ones(n)]
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    z = rng.uniform(*depth, size=n)
    X = f * (z / f[:, 2])[:, None]                       # 시점 a 좌표계 3D 점
    rho = 1.0 / np.linalg.norm(X, axis=1)                # 광선 방향 기준 역깊이
    R = expm_so3(rng.normal(size=3) * 0.15)
    t = np.array([baseline, 0.1 * baseline, 0.05 * baseline]) * np.sign(rng.normal())
    return f, rho, R, t


def project(Xc):
    return Xc[:, :2] / Xc[:, 2:3]


def residual(params, f, u, rho_prior, w_prior):
    R, t, rho = params
    Xb = (R @ (f / rho[:, None]).T).T + t
    r_rep = (u - project(Xb)).ravel()
    r_pri = w_prior * (rho - rho_prior)
    return np.r_[r_rep, r_pri]


def jacobians(params, f, u, rho_prior, w_prior, eps=1e-6):
    R, t, rho = params
    r0 = residual(params, f, u, rho_prior, w_prior)
    n = len(rho)
    Jc = np.zeros((len(r0), 6))
    for k in range(3):
        d = np.zeros(3); d[k] = eps
        Jc[:, k] = (residual((expm_so3(d) @ R, t, rho), f, u, rho_prior, w_prior) - r0) / eps
        Jc[:, 3 + k] = (residual((R, t + d, rho), f, u, rho_prior, w_prior) - r0) / eps
    Jg = np.zeros((len(r0), n))
    for i in range(n):       # 점 i는 자기 잔차 2개 + 사전 1개에만 들어간다(희소)
        rho2 = rho.copy(); rho2[i] += eps
        Jg[:, i] = (residual((R, t, rho2), f, u, rho_prior, w_prior) - r0) / eps
    return r0, Jc, Jg


def apply(params, dc, dg):
    R, t, rho = params
    return expm_so3(dc[:3]) @ R, t + dc[3:], np.maximum(rho + dg, 1e-3)


def step_joint(params, data, damp=1e-6):
    r, Jc, Jg = jacobians(params, *data)
    B, E, C = Jc.T @ Jc, Jc.T @ Jg, Jg.T @ Jg
    v, w = -Jc.T @ r, -Jg.T @ r
    Ci = np.linalg.inv(C + damp * np.eye(len(C)))
    S = B + damp * np.eye(6) - E @ Ci @ E.T                # Schur 보완(식 3)
    dc = np.linalg.solve(S, v - E @ Ci @ w)
    dg = Ci @ (w - E.T @ dc)
    return apply(params, dc, dg)


def step_camera_first(params, data, damp=1e-6):
    """논문 3.3절 순서: 현재 geometry로 카메라만 갱신 → 새 카메라로 geometry 갱신(재선형화)."""
    r, Jc, _ = jacobians(params, *data)
    dc = np.linalg.solve(Jc.T @ Jc + damp * np.eye(6), -Jc.T @ r)
    params = apply(params, dc, np.zeros(len(params[2])))
    r, _, Jg = jacobians(params, *data)
    dg = np.linalg.solve(Jg.T @ Jg + damp * np.eye(Jg.shape[1]), -Jg.T @ r)
    return apply(params, np.zeros(6), dg)


def step_geometry_first(params, data, damp=1e-6):
    """공개 코드 순서: 이전 카메라로 geometry 갱신 → 새 geometry의 잔차로 카메라 갱신."""
    r, _, Jg = jacobians(params, *data)
    dg = np.linalg.solve(Jg.T @ Jg + damp * np.eye(Jg.shape[1]), -Jg.T @ r)
    params = apply(params, np.zeros(6), dg)
    r, Jc, _ = jacobians(params, *data)
    dc = np.linalg.solve(Jc.T @ Jc + damp * np.eye(6), -Jc.T @ r)
    return apply(params, dc, np.zeros(len(params[2])))


def errors(params, gt):
    R, t, rho = params
    Rg, tg, rhog = gt
    rot = rot_angle_deg(R @ Rg.T)
    tdir = math.degrees(math.acos(np.clip(t @ tg / (np.linalg.norm(t) * np.linalg.norm(tg)), -1, 1)))
    depth = float(np.median(np.abs(1 / rho - 1 / rhog) / (1 / rhog)))
    return rot, tdir, depth


def setup(rng, n, baseline, sigma_px=1.0, focal=500.0, prior_rel=0.10, w_prior=0.05,
          init_rot_deg=8.0, init_t_err=0.3):
    f, rho, R, t = make_scene(rng, n, baseline)
    Xb = (R @ (f / rho[:, None]).T).T + t
    u = project(Xb) + rng.normal(scale=sigma_px / focal, size=(n, 2))
    rho_prior = rho * (1 + prior_rel * rng.normal(size=n))             # 단안 깊이 사전(10% 잡음)
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
    R0 = expm_so3(ax * math.radians(init_rot_deg)) @ R
    t0 = t + init_t_err * np.linalg.norm(t) * rng.normal(size=3) / math.sqrt(3)
    init = (R0, t0, rho_prior.copy())
    data = (f, u, rho_prior, w_prior)
    return init, data, (R, t, rho)


def run(step, init, data, gt, iters):
    p = init
    hist = [errors(p, gt)]
    for _ in range(iters):
        p = step(p, data)
        hist.append(errors(p, gt))
    return hist


def main():
    out = {}
    rng = np.random.default_rng(0)

    # T1: Schur vs 전체 해
    init, data, gt = setup(rng, 60, 0.3)
    r, Jc, Jg = jacobians(init, *data)
    J = np.c_[Jc, Jg]
    full = np.linalg.solve(J.T @ J + 1e-6 * np.eye(J.shape[1]), -J.T @ r)
    B, E, C = Jc.T @ Jc, Jc.T @ Jg, Jg.T @ Jg
    v, w = -Jc.T @ r, -Jg.T @ r
    Ci = np.linalg.inv(C + 1e-6 * np.eye(len(C)))
    dc = np.linalg.solve(B + 1e-6 * np.eye(6) - E @ Ci @ E.T, v - E @ Ci @ w)
    dg = Ci @ (w - E.T @ dc)
    offdiag = np.abs(C - np.diag(np.diag(C))).max() / np.abs(np.diag(C)).max()
    out["T1_schur_equals_full"] = {
        "max_abs_diff": float(np.abs(np.r_[dc, dg] - full).max()),
        "C_offdiag_over_diag": float(offdiag),
        "note": "C(점끼리 블록)가 대각이라 역행렬이 O(N), Schur 보완 S는 6×6",
    }

    # T2: 갱신 방식 비교
    t2 = {}
    tol_rot = 0.2
    for baseline in (0.05, 0.3):
        for init_rot in (3.0, 10.0):
            key = f"baseline{baseline}_init{init_rot}deg"
            res = {"joint": [], "camera_first(paper)": [], "geometry_first(code)": []}
            for s in range(10):
                rr = np.random.default_rng(100 + s)
                init, data, gt = setup(rr, 40, baseline, init_rot_deg=init_rot)
                for name, st in (("joint", step_joint), ("camera_first(paper)", step_camera_first),
                                 ("geometry_first(code)", step_geometry_first)):
                    res[name].append(run(st, init, data, gt, 8))
            summ = {}
            for name, hs in res.items():
                arr = np.array(hs)                         # (seed, iter+1, 3)
                rot = arr[:, :, 0]
                k_reach = [int(np.argmax(r_ <= tol_rot)) if (r_ <= tol_rot).any() else None for r_ in rot]
                summ[name] = {
                    "rot_deg_median_by_iter": [round(float(x), 3) for x in np.median(rot, axis=0)],
                    "tdir_deg_median_iter4": round(float(np.median(arr[:, 4, 1])), 3),
                    "depth_rel_median_iter4": round(float(np.median(arr[:, 4, 2])), 4),
                    "iters_to_rot<0.2deg_median": (float(np.median([k for k in k_reach if k is not None]))
                                                  if any(k is not None for k in k_reach) else None),
                    "reached_frac": round(sum(k is not None for k in k_reach) / len(k_reach), 2),
                }
            t2[key] = summ
    out["T2_update_order"] = t2

    # T2b: 고정점 비교 (기선 0.3, 초기 회전 10°, 200회)
    t2b = []
    for s in range(3):
        rr = np.random.default_rng(100 + s)
        init, data, gt = setup(rr, 40, 0.3, init_rot_deg=10.0)
        finals = {}
        for name, st in (("joint", step_joint), ("camera_first", step_camera_first),
                         ("geometry_first", step_geometry_first)):
            p = init
            for _ in range(200):
                p = st(p, data)
            finals[name] = p
        ref = finals["joint"]
        row = {"seed": 100 + s}
        for name in ("camera_first", "geometry_first"):
            R, t, rho = finals[name]
            row[name] = {"rot_diff_deg_vs_joint": float(rot_angle_deg(R @ ref[0].T)),
                         "t_diff_vs_joint": float(np.linalg.norm(t - ref[1])),
                         "rho_maxdiff_vs_joint": float(np.abs(rho - ref[2]).max())}
        row["joint_cost"] = float(np.sum(residual(ref, *data) ** 2))
        t2b.append(row)
    out["T2b_same_fixed_point_200iters"] = t2b

    # T3: N에 따른 최종 오차 (joint 8회)
    t3 = {}
    for n in (10, 40, 160):
        rots, depths = [], []
        for s in range(10):
            rr = np.random.default_rng(500 + s)
            init, data, gt = setup(rr, n, 0.3, init_rot_deg=3.0)
            h = run(step_joint, init, data, gt, 8)
            rots.append(h[-1][0]); depths.append(h[-1][2])
            init_depth = h[0][2]
        t3[f"N{n}"] = {"rot_deg_median": round(float(np.median(rots)), 4),
                       "depth_rel_median": round(float(np.median(depths)), 4),
                       "depth_prior_rel_median": round(float(init_depth), 4)}
    r10, r160 = t3["N10"]["rot_deg_median"], t3["N160"]["rot_deg_median"]
    d10, d160 = t3["N10"]["depth_rel_median"], t3["N160"]["depth_rel_median"]
    t3["ratio_N10_over_N160"] = {"rot": round(r10 / r160, 2), "depth": round(d10 / d160, 2),
                                 "sqrt(160/10)": round(math.sqrt(16), 2)}
    out["T3_dimensionality"] = t3

    path = ROOT / "analysis" / "ba_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
