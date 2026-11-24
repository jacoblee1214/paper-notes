"""Toy 검증: 논문 Algorithm 1을 numpy로 직접 구현해 공식 저장소가 공개한 예시 앙상블 4개에 적용한다.

- 입력: 공식 저장소의 examples/data/<object>_ensemble.npz (chunks: 128x24x10, tcp_pose, object_pose).
  numpy 배열만 읽고 공식 코드는 import하지 않는다.
- 확인 항목 (K1-K5):
  K1 차원 m과 SE: 논문 Algorithm 1 -> 기준 부분공간(해석)과의 최대 주각
  K2 앙상블 크기 N을 줄여 부분표본으로 재추정한 SE (표 11과 같은 추세인지)
  K3 비중심 PCA 대 중심 PCA (부록 D.3의 주장)
  K4 특성 길이 alpha 민감도 (부록 H.1의 "최대 1.7도" 주장)
  K5 게인 대수: 사영 행렬의 멱등성, Lambda-직교성, K/D의 대칭·PSD, 고유값
실행: python3 toy_ensemble_estimator.py <ensemble 폴더> (CPU, 수 초)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

DT = 0.1
V_ABS, KAPPA, A_MIN, A_MAX = 0.005, 4.0, 0.03, 0.5
K0, R_RED, ZETA = 250.0, 0.1, 0.7655
MASS, INERTIA = 0.586, 0.00622

# 기준 부분공간 정의(논문 부록 H.3, 그림 3): 물체 좌표계에서의 나사 생성자
GENERATORS = {
    "revolute": [("rot", (0, 0, 1), (-0.215, -0.15, 0.0))],
    "cylindrical": [("rot", (0, 0, 1), (0, 0, 0.10)), ("trans", (0, 0, 1), None)],
    "planar": [("rot", (1, 0, 0), (-0.025, 0, 0)), ("trans", (0, 1, 0), None), ("trans", (0, 0, 1), None)],
    "universal": [("rot", (1, 0, 0), (0, 0, 0)), ("rot", (0, 1, 0), (0, 0, 0)), ("rot", (0, 0, 1), (0, 0, 0))],
}
TRUE_DIM = {"revolute": 1, "cylindrical": 2, "planar": 3, "universal": 3}
ALPHA_STAR = {
    "revolute": math.hypot(-0.035, 0.24),
    "cylindrical": math.sqrt(0.12 ** 2 + (0.30 / math.pi) ** 2),
    "planar": math.sqrt(0.30 ** 2 + 0.30 ** 2) / math.pi,
    "universal": 0.08 * math.sqrt(2.0 / 3.0),
}


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0.0]])


def rot6d_to_R(x):
    a1, a2 = x[:3], x[3:6]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - b1 @ a2 * b1
    b2 /= np.linalg.norm(b2)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1)


def quat_to_R(q):  # (w, x, y, z)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def so3_log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return th * w


def se3_log(T):
    w = so3_log(T[:3, :3])
    th = np.linalg.norm(w)
    W = skew(w)
    if th < 1e-9:
        Jinv = np.eye(3)
    else:
        Jinv = np.eye(3) - 0.5 * W + (1 / th ** 2 - (1 + math.cos(th)) / (2 * th * math.sin(th))) * W @ W
    return np.concatenate([w, Jinv @ T[:3, 3]])


def inv(T):
    R, p = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out


def poses_from_chunks(chunks):
    N, H, _ = chunks.shape
    T = np.tile(np.eye(4), (N, H, 1, 1))
    for i in range(N):
        for t in range(H):
            T[i, t, :3, :3] = rot6d_to_R(chunks[i, t, 3:9])
            T[i, t, :3, 3] = chunks[i, t, :3]
    return T


def body_twists(T):  # Algorithm 1 line 3 (중앙차분, 양 끝은 한쪽 차분)
    N, H = T.shape[:2]
    V = np.zeros((N, H, 6))
    for i in range(N):
        for t in range(H):
            if t == 0:
                V[i, t] = se3_log(inv(T[i, 0]) @ T[i, 1]) / DT
            elif t == H - 1:
                V[i, t] = se3_log(inv(T[i, H - 2]) @ T[i, H - 1]) / DT
            else:
                V[i, t] = se3_log(inv(T[i, t - 1]) @ T[i, t + 1]) / (2 * DT)
    return V


def estimate(W, alpha=None, centered=False, v_abs=V_ABS, kappa=KAPPA):
    """W: 6xM. 반환 dict(m, U(6x6), rms(6), alpha)."""
    if centered:
        W = W - W.mean(axis=1, keepdims=True)
    if alpha is None:
        u = np.linalg.svd(W, full_matrices=False)[0][:, 0]
        alpha = float(np.clip((np.linalg.norm(u[3:]) + 1e-6) / (np.linalg.norm(u[:3]) + 1e-6), A_MIN, A_MAX))
    L = np.diag([alpha] * 3 + [1.0] * 3)
    Ut, S, _ = np.linalg.svd(L @ W, full_matrices=False)
    U = np.linalg.solve(L, Ut)
    rms = S / math.sqrt(W.shape[1])
    thr = max(kappa * rms[5], v_abs)
    m = int((rms > thr).sum())
    return dict(m=m, U=U, rms=rms, alpha=alpha)


def ref_basis(name, tcp, obj):
    p_ee, R_ee = tcp[:3], quat_to_R(tcp[3:7])
    p_ob, R_ob = obj[:3], quat_to_R(obj[3:7])
    cols = []
    for kind, axis, point in GENERATORS[name]:
        a = R_ob @ np.array(axis, float)
        if kind == "trans":
            om, v = np.zeros(3), a
        else:
            om, v = a, np.cross(a, p_ee - (p_ob + R_ob @ np.array(point, float)))
        cols.append(np.concatenate([R_ee.T @ om, R_ee.T @ v]))
    return np.stack(cols, axis=1)


def max_angle(Ua, Ub, alpha):
    if Ua.shape[1] != Ub.shape[1]:
        return 90.0
    L = np.diag([alpha] * 3 + [1.0] * 3)
    Qa = np.linalg.qr(L @ Ua)[0]
    Qb = np.linalg.qr(L @ Ub)[0]
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(s.min(), 0, 1))))


def main(folder):
    rng = np.random.default_rng(0)
    out = {"K1": {}, "K2": {}, "K3": {}, "K4": {}, "K5": {}}
    metric = np.diag([INERTIA / MASS] * 3 + [1.0] * 3)  # 정규화된 Lambda-bar
    for name in ["revolute", "cylindrical", "planar", "universal"]:
        d = np.load(Path(folder) / f"{name}_ensemble.npz", allow_pickle=False)
        chunks, tcp, obj = d["chunks"].astype(float), d["tcp_pose"], d["object_pose"]
        T = poses_from_chunks(chunks)
        V = body_twists(T)  # N,H,6
        W = V.reshape(-1, 6).T
        Uref = ref_basis(name, tcp, obj)
        est = estimate(W)
        m = est["m"]
        se = max_angle(est["U"][:, :m], Uref, ALPHA_STAR[name])
        out["K1"][name] = dict(m=m, true_dim=TRUE_DIM[name], alpha=est["alpha"], rms_speed_mm_s=[round(1000 * x, 2) for x in est["rms"]],
                                threshold_mm_s=round(1000 * max(KAPPA * est["rms"][5], V_ABS), 2), SE_deg=round(se, 3))
        # K2: 부분표본
        k2 = {}
        N = chunks.shape[0]
        for n in [1, 2, 4, 8, 16, 32, 64]:
            ses, ok = [], 0
            for _ in range(200):
                idx = rng.choice(N, n, replace=False)
                Wn = V[idx].reshape(-1, 6).T
                e = estimate(Wn, v_abs=V_ABS)
                ok += int(e["m"] == TRUE_DIM[name])
                # 차원을 참값으로 고정한 SE (논문 5.3절 방식)
                ses.append(max_angle(e["U"][:, :TRUE_DIM[name]], Uref, ALPHA_STAR[name]))
            k2[n] = dict(se_p50=round(float(np.percentile(ses, 50)), 2), se_p95=round(float(np.percentile(ses, 95)), 2),
                         correct_dim_frac=ok / 200)
        out["K2"][name] = k2
        # K3: 중심 PCA
        ec = estimate(W, centered=True)
        out["K3"][name] = dict(m_centered=ec["m"], SE_centered_deg=round(max_angle(ec["U"][:, :TRUE_DIM[name]], Uref, ALPHA_STAR[name]), 3),
                               SE_uncentered_trueDim_deg=round(max_angle(est["U"][:, :TRUE_DIM[name]], Uref, ALPHA_STAR[name]), 3))
        # K4: alpha 민감도 (차원 고정, Lambda-bar 메트릭 주각)
        a0 = est["alpha"]
        base = estimate(W, alpha=a0)["U"][:, :m]
        l_lam = math.sqrt(INERTIA / MASS)
        worst = 0.0
        for f in [0.25, 0.5, 2, 4]:
            Ua = estimate(W, alpha=a0 * f)["U"][:, :m]
            worst = max(worst, max_angle(Ua, base, l_lam))
        out["K4"][name] = dict(alpha0=a0, max_angle_deg=round(worst, 3))
        # K5: 게인 대수
        Ue = est["U"][:, :m]
        P = Ue @ np.linalg.solve(Ue.T @ metric @ Ue, Ue.T @ metric)
        Pp = np.eye(6) - P
        k_par, k_perp = K0, R_RED * K0
        d_par, d_perp = 2 * ZETA * math.sqrt(k_par), 2 * ZETA * math.sqrt(k_perp)
        K = metric @ (k_par * P + k_perp * Pp)
        D = metric @ (d_par * P + d_perp * Pp)
        ev = np.sort(np.linalg.eigvals(np.linalg.solve(metric, K)).real)
        out["K5"][name] = dict(idempotent_err=float(np.abs(P @ P - P).max()), sym_err_K=float(np.abs(K - K.T).max()),
                                min_eig_K=float(np.linalg.eigvalsh((K + K.T) / 2).min()),
                                modal_eigs=[round(float(x), 3) for x in ev], d_perp_over_d_par=round(d_perp / d_par, 4),
                                sqrt_r=round(math.sqrt(R_RED), 4), K_diag=[round(float(x), 3) for x in np.diag(K)])
    # m=0 경우 (알고리즘 문면): U 비면 P_par=0 -> K = k_perp * Lambda (전 방향 compliant)
    out["m0_note"] = "알고리즘 10-11행을 그대로 따르면 m=0일 때 Kv = k_perp*Lambda(전 방향 저강성); 공식 코드는 등방(Iso) 게인을 유지."
    Path(__file__).with_name("toy_ensemble_estimator_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
