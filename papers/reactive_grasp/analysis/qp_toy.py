"""힘 할당 QP(Eq. 3–9) 구조 toy (CPU, numpy/scipy). 실행: python3 analysis/qp_toy.py

같은 연구실 공개 코드 `utils/grasping/force_gen.py: generate_contact_forces`의 lifted 형식
(z = [f; w], w = G f, 비용 ½‖w‖²)을 그대로 쓰고, OSQP와 같은 ADMM 반복(ρ=0.1, σ=1e-6, α=1.6,
등식 행 ρ×1e3, 0에서 시작, 1000회, polish 없음)을 numpy로 구현해 푼다.

손 모델: 반지름 4 cm 원통을 감싸는 5손가락 × 4관절(20 DoF) 직렬 사슬 + 손바닥(비구동).
손가락 링크 3개(근위·중위·원위)와 손바닥에 링크당 K개 접점. 관절축은 원통축(z)과 평행한 굽힘 3개 + 벌림 1개.
실제 Robotis 5F 기구학이 아니라 Jacobian의 희소 구조(링크 j의 점은 그 손가락의 j번째 관절까지만 의존)를 살린 모델이다.

- E1 해의 비유일성: 목적함수 0인 해가 원뿔 전체(스케일까지)라서, 시작점에 따라 해가 다르다
- E2 해의 연속성: 접점 위치를 1 mm 흔들 때 해가 얼마나 튀는가 (정규화 λ‖f‖² 유무)
- E3 actuation-consistency: null(J_actᵀ) 차원, 상자 제약 ε=0.5 N이 허용하는 null 성분 크기 ε√d,
     "실현 힘 = range(J_act) 사영"(접점 강성이 같고 물체가 고정된 준정적 가정)에서 남는 잔차
- E4 힘(N)과 토크(N·m)를 같은 가중치로 더할 때 회전 잔차의 실효 가중치
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
R_OBJ = 0.04
MU, FMIN, TAU_MAX, EPS = 0.7, 10.0, 0.2, 0.5   # Sec. IV (공개 sim 설정과 같음)


# ── 손·물체 기하 ─────────────────────────────────────────────────────────────
def cyl(theta_deg: float, z: float, r: float = R_OBJ) -> np.ndarray:
    t = np.radians(theta_deg)
    return np.array([r * np.cos(t), r * np.sin(t), z])


def hand_model(K: int, rng: np.random.Generator, jitter_m: float = 0.0, subset=None, palm: bool = True):
    """접점(위치, 안쪽 법선, 링크 id, 구동 여부)과 점별 3×20 Jacobian. subset: 접점을 둘 (손가락, 링크) 집합."""
    fingers = []
    for i, z in enumerate([-0.03, -0.01, 0.01, 0.03]):            # 네 손가락: +x 쪽으로 감쌈
        fingers.append(dict(z=z, links=[40, -10, -60], joints=[65, 65, 15, -35]))
    fingers.append(dict(z=0.0, links=[140, 190, 240], joints=[115, 115, 165, 215]))  # 엄지: −x 쪽
    pts, nrm, link, J = [], [], [], []
    dz = np.linspace(-0.004, 0.004, K) if K > 1 else [0.0]
    dth = np.linspace(-4, 4, K) if K > 1 else [0.0]
    for f, spec in enumerate(fingers):
        o = [cyl(a, spec["z"], R_OBJ + 0.012) for a in spec["joints"]]
        radial0 = o[0] / np.linalg.norm(o[0][:2] + 1e-12)
        axes = [np.array([radial0[0], radial0[1], 0.0]) / np.linalg.norm(radial0[:2]),
                np.array([0, 0, 1.0]), np.array([0, 0, 1.0]), np.array([0, 0, 1.0])]
        for l, ang in enumerate(spec["links"]):
            if subset is not None and (f, l) not in subset:
                continue
            for k in range(K):
                p = cyl(ang + dth[k], spec["z"] + dz[k])
                p = p + jitter_m * rng.normal(size=3)
                n_in = -np.array([p[0], p[1], 0.0]) / np.hypot(p[0], p[1])
                Jp = np.zeros((3, 20))
                for jj in range(l + 2):                         # 링크 l은 관절 0..l+1에 의존
                    Jp[:, 4 * f + jj] = np.cross(axes[jj], p - o[jj])
                pts.append(p); nrm.append(n_in); link.append(4 * f + l); J.append(Jp)
    for k in range(max(K, 1) if palm else 0):                     # 손바닥: 비구동
        p = cyl(90 + (dth[k] if K > 1 else 0) * 2, (dz[k] if K > 1 else 0) * 4)
        p = p + jitter_m * rng.normal(size=3)
        pts.append(p); nrm.append(-np.array([p[0], p[1], 0.0]) / np.hypot(p[0], p[1]))
        link.append(-1); J.append(np.zeros((3, 20)))
    return np.array(pts), np.array(nrm), np.array(link), np.array(J)


def frames(nrm: np.ndarray):
    t1 = np.tile(np.array([0, 0, 1.0]), (len(nrm), 1))
    t2 = np.cross(nrm, t1)
    return t1, t2


def grasp_map(pts, nrm):
    t1, t2 = frames(nrm)
    G = np.zeros((6, 3 * len(pts)))
    for i, (p, n, a, b) in enumerate(zip(pts, nrm, t1, t2)):
        Ri = np.stack([n, a, b], axis=1)
        G[:3, 3 * i:3 * i + 3] = Ri
        G[3:, 3 * i:3 * i + 3] = np.cross(p, Ri.T).T
    return G


def contact_jac(J, nrm):
    """점별 world Jacobian을 접점 좌표계 [n, t1, t2]로 (code: compute_contact_frame)."""
    t1, t2 = frames(nrm)
    out = np.zeros((3 * len(J), J.shape[2]))
    for i in range(len(J)):
        Ri = np.stack([nrm[i], t1[i], t2[i]], axis=1)
        out[3 * i:3 * i + 3] = Ri.T @ J[i]
    return out


# ── OSQP식 ADMM ──────────────────────────────────────────────────────────────
def admm(P, q, A, l, u, x0=None, iters=1000, rho=0.1, sigma=1e-6, alpha=1.6):
    n, m = P.shape[0], A.shape[0]
    rv = np.where(np.isclose(l, u), rho * 1e3, rho)
    M = P + sigma * np.eye(n) + A.T @ (rv[:, None] * A)
    cf = cho_factor(M)
    x = np.zeros(n) if x0 is None else x0.copy()
    z = A @ x
    y = np.zeros(m)
    for _ in range(iters):
        xt = cho_solve(cf, sigma * x - q + A.T @ (rv * z - y))
        zt = A @ xt
        x = alpha * xt + (1 - alpha) * x
        zr = alpha * zt + (1 - alpha) * z
        z = np.clip(zr + y / rv, l, u)
        y = y + rv * (zr - z)
    viol = float(np.max(np.maximum(A @ x - u, 0) + np.maximum(l - A @ x, 0)))
    return x, viol


def build_qp(G, Jc, act, ac=True, ridge=0.0, eps=EPS):
    """code의 generate_contact_forces와 같은 제약 묶음. act: 접점별 구동 여부."""
    P_ = len(act)
    nf, nw = 3 * P_, 6
    Pm = np.zeros((nf + nw, nf + nw))
    Pm[nf:, nf:] = np.eye(nw)
    Pm[:nf, :nf] += ridge * np.eye(nf)
    rows, lo, hi = [], [], []

    def add(Af, l_, u_):
        rows.append(np.hstack([Af, np.zeros((Af.shape[0], nw))])); lo.append(l_); hi.append(u_)

    rows.append(np.hstack([G, -np.eye(nw)])); lo.append(np.zeros(nw)); hi.append(np.zeros(nw))
    act_rows = np.concatenate([[3 * i, 3 * i + 1, 3 * i + 2] for i in np.where(act)[0]])
    Ja = Jc[act_rows]
    A_tau = np.zeros((Ja.shape[1], nf)); A_tau[:, act_rows] = Ja.T
    add(A_tau, -TAU_MAX * np.ones(Ja.shape[1]), TAU_MAX * np.ones(Ja.shape[1]))
    A_sum = np.zeros((1, nf)); A_sum[0, 0::3] = 1
    add(A_sum, np.array([FMIN]), np.array([np.inf]))
    A_pos = np.zeros((P_, nf)); A_pos[np.arange(P_), 3 * np.arange(P_)] = 1
    add(A_pos, np.zeros(P_), np.full(P_, np.inf))
    A_fr = np.zeros((4 * P_, nf))
    for i in range(P_):
        for r, (s1, s2) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
            A_fr[4 * i + r, 3 * i:3 * i + 3] = [-MU, s1, s2]
    add(A_fr, np.full(4 * P_, -np.inf), np.zeros(4 * P_))
    U, s, _ = np.linalg.svd(Ja, full_matrices=True)
    rank = int(np.sum(s > 1e-6))
    Nu = U[:, rank:]
    if ac and Nu.shape[1] > 0:
        A_ac = np.zeros((Nu.shape[1], nf)); A_ac[:, act_rows] = Nu.T
        add(A_ac, -eps * np.ones(Nu.shape[1]), eps * np.ones(Nu.shape[1]))
    A = np.vstack(rows)
    return Pm, np.zeros(nf + nw), A, np.concatenate(lo), np.concatenate(hi), act_rows, Ja, rank, Nu.shape[1]


def summarize(G, f, act_rows, Ja):
    nf = len(f)
    Pr = Ja @ np.linalg.pinv(Ja)                        # range(J_act) 사영
    fa = f[act_rows]
    f_null = fa - Pr @ fa
    G_act = G[:, act_rows]
    res = G_act @ f_null
    return {
        "obj_residual_N": round(float(np.linalg.norm(G[:3] @ f)), 4),
        "sum_fn_N": round(float(f[0::3].sum()), 3),
        "max_fn_N": round(float(f[0::3].max()), 3),
        "invisible_fraction_of_actuated_force": round(float(np.linalg.norm(f_null) / max(np.linalg.norm(fa), 1e-9)), 3),
        "realized_residual_force_N": round(float(np.linalg.norm(res[:3])), 3),
        "realized_residual_torque_Nm": round(float(np.linalg.norm(res[3:])), 4),
    }


def main() -> None:
    rng = np.random.default_rng(0)
    out: dict = {}

    # E3 먼저: 차원 셈과 제약 유무
    e3 = {}
    for K in (1, 3, 5):
        pts, nrm, link, J = hand_model(K, rng)
        act = link >= 0
        G, Jc = grasp_map(pts, nrm), contact_jac(J, nrm)
        row = {"contacts": int(len(pts)), "actuated_contacts": int(act.sum())}
        for ac in (False, True):
            Pm, q, A, l, u, act_rows, Ja, rank, d = build_qp(G, Jc, act, ac=ac)
            x, viol = admm(Pm, q, A, l, u)
            f = x[:3 * len(pts)]
            row.update({"rank_J_act": rank, "null_dim": d, "eps_sqrt_d_N": round(EPS * np.sqrt(d), 2)})
            row["with_AC" if ac else "without_AC"] = {**summarize(G, f, act_rows, Ja), "max_violation": round(viol, 5)}
        e3[f"K={K}"] = row
    out["E3_actuation_consistency"] = e3

    # E3b 세 손가락 파지(검지·중지 원위 + 엄지 근위, 손바닥 없음)와 ε 크기
    e3b = {}
    for name, kw in (("power_K5", dict()), ("precision_3finger_K5", dict(subset={(0, 2), (1, 2), (4, 0)}, palm=False))):
        pts, nrm, link, J = hand_model(5, rng, **kw)
        act = link >= 0
        G, Jc = grasp_map(pts, nrm), contact_jac(J, nrm)
        row = {"contacts": int(len(pts))}
        for eps in (None, 0.5, 0.1, 0.02):
            Pm, q, A, l, u, act_rows, Ja, rank, d = build_qp(G, Jc, act, ac=eps is not None, eps=eps or EPS)
            x, viol = admm(Pm, q, A, l, u)
            f = x[:3 * len(pts)]
            row["rank_J_act"], row["null_dim"] = rank, d
            row["no_AC" if eps is None else f"eps={eps}"] = {**summarize(G, f, act_rows, Ja), "max_violation": round(viol, 4)}
        e3b[name] = row
    out["E3b_precision_and_eps"] = e3b

    # E1 비유일성: K=3, 0에서 시작 vs 무작위 시작
    pts, nrm, link, J = hand_model(3, rng)
    act = link >= 0
    G, Jc = grasp_map(pts, nrm), contact_jac(J, nrm)
    Pm, q, A, l, u, act_rows, Ja, _, _ = build_qp(G, Jc, act, ac=True)
    nf = 3 * len(pts)
    x0, _ = admm(Pm, q, A, l, u)
    sols = []
    for s in range(8):
        r = np.random.default_rng(100 + s)
        start = np.concatenate([np.abs(r.normal(0, 1.0, nf)) * np.tile([1, 0, 0], len(pts)), np.zeros(6)])
        xs, viol = admm(Pm, q, A, l, u, x0=start)
        sols.append((xs[:nf], viol))
    f0 = x0[:nf]
    out["E1_non_uniqueness"] = {
        "cold_start": summarize(G, f0, act_rows, Ja),
        "random_starts": [{**summarize(G, fs, act_rows, Ja), "rel_diff_from_cold": round(float(np.linalg.norm(fs - f0) / np.linalg.norm(f0)), 3)} for fs, _ in sols],
        "note": "목적함수(물체 잔차)는 모두 0 근처인데 Σf_n·최대 접점 힘·분포가 시작점마다 다르다",
    }

    # E2 연속성: 1 mm 흔들림, 0에서 시작(코드는 매 주기 OSQP 객체를 새로 만들어 0에서 시작)
    e2 = {}
    for ridge in (0.0, 1e-3):
        jumps = []
        for s in range(10):
            r = np.random.default_rng(200 + s)
            p2, n2, l2, J2 = hand_model(3, r, jitter_m=0.001)
            G2, Jc2 = grasp_map(p2, n2), contact_jac(J2, n2)
            Pa, qa, Aa, la, ua, *_ = build_qp(G, Jc, act, ac=True, ridge=ridge)
            Pb, qb, Ab, lb, ub, *_ = build_qp(G2, Jc2, l2 >= 0, ac=True, ridge=ridge)
            xa, _ = admm(Pa, qa, Aa, la, ua)
            xb, _ = admm(Pb, qb, Ab, lb, ub)
            fa, fb = xa[:nf], xb[:nf]
            jumps.append(float(np.linalg.norm(fa - fb) / np.linalg.norm(fa)))
        e2[f"ridge={ridge}"] = {"median_rel_change": round(float(np.median(jumps)), 3), "max_rel_change": round(float(np.max(jumps)), 3)}
    e2["geometry_rel_change"] = round(0.001 / R_OBJ, 3)
    out["E2_continuity_1mm_jitter"] = e2

    # E4 힘·토크 같은 가중치
    out["E4_unit_weights"] = {
        "torque_residual_effective_weight_vs_force_at_r": {f"{int(r*100)}cm": round(r * r, 4) for r in (0.03, 0.05, 0.1)},
        "note": "같은 크기의 접선 힘 불균형 1 N이 반지름 r에서 만드는 토크 r N·m는 비용에서 r²배로만 벌점을 받는다",
    }

    path = ROOT / "analysis" / "qp_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
