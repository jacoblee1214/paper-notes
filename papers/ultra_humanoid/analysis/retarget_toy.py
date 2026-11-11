"""리타게팅 토이: 운동학 리타게팅(IK)의 궤적이 토크 한계 안에서 실행 가능한지 검사한다.

이것은 ULTRA가 아니라 토이다. 평면 2링크 팔(어깨 고정, 끝에 물체 질량 mp)이 손 경로를 따라가는 상황을 본다.
- 운동학 리타게팅: 손 경로를 damped least squares IK로 따라가 관절 궤적 q(t)를 만든다(동역학 무시).
- 동역학 검사: 역동역학 τ = M(q) q̈ + C(q,q̇) q̇ + G(q)로 필요한 토크를 계산하고, 한계 τmax와 비교한다.
- 물리 필터: 시간을 s배 늦춰 가속도를 줄였을 때 처음으로 실행 가능해지는 s를 이분 탐색한다.
- 증강: 손 경로의 x 방향을 1.15배 늘린 변형(재학습 없는 기하 증강)도 같은 검사를 받는다.

실행: python3 analysis/retarget_toy.py  (numpy 단일 스레드, 수 초). 결과: analysis/retarget_toy_results.json
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

OUT = Path(__file__).with_name("retarget_toy_results.json")
L1 = L2 = 0.35            # 링크 길이 (m)
M1 = M2 = 1.5             # 링크 질량 (kg)
MP = 1.0                  # 끝 물체 질량 (kg)
G = 9.81
TAU_MAX = np.array([30.0, 15.0])   # 관절 토크 한계 (N·m), 토이 값
DT = 1.0 / 60.0
RNG = np.random.default_rng(0)


def fk(q):
    x = L1 * np.cos(q[0]) + L2 * np.cos(q[0] + q[1])
    y = L1 * np.sin(q[0]) + L2 * np.sin(q[0] + q[1])
    return np.array([x, y])


def jac(q):
    s1, c1 = np.sin(q[0]), np.cos(q[0])
    s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12],
                     [L1 * c1 + L2 * c12, L2 * c12]])


def ik(target, q0, iters=60, lam=1e-3):
    q = q0.copy()
    for _ in range(iters):
        e = target - fk(q)
        if np.linalg.norm(e) < 1e-7:
            break
        J = jac(q)
        q = q + J.T @ np.linalg.solve(J @ J.T + lam * np.eye(2), e)
    return q, np.linalg.norm(target - fk(q))


def inv_dyn(q, qd, qdd):
    """평면 2링크 역동역학(어깨 고정). 끝 물체는 끝 질점으로 링크 2에 더한다."""
    m2 = M2 + MP
    lc1, lc2 = L1 / 2, L2                 # 링크 2의 질량중심을 끝으로 근사(물체 포함)
    I1 = M1 * L1 ** 2 / 12
    I2 = M2 * L2 ** 2 / 12 + MP * L2 ** 2
    c2 = np.cos(q[1])
    M11 = M1 * lc1 ** 2 + m2 * (L1 ** 2 + lc2 ** 2 + 2 * L1 * lc2 * c2) + I1 + I2
    M12 = m2 * (lc2 ** 2 + L1 * lc2 * c2) + I2
    M22 = m2 * lc2 ** 2 + I2
    M = np.array([[M11, M12], [M12, M22]])
    h = -m2 * L1 * lc2 * np.sin(q[1])
    Cqd = np.array([h * qd[1] * qd[0] + h * (qd[0] + qd[1]) * qd[1],
                    -h * qd[0] ** 2])
    g1 = (M1 * lc1 + m2 * L1) * G * np.cos(q[0]) + m2 * lc2 * G * np.cos(q[0] + q[1])
    g2 = m2 * lc2 * G * np.cos(q[0] + q[1])
    return M @ qdd + Cqd + np.array([g1, g2])


def hand_path(T, scale_x=1.0):
    """시작 (0.45, 0.0) → 끝 (0.25, 0.45)으로 가는 5차 시간 법칙 경로. scale_x로 x를 늘린다."""
    n = int(round(T / DT)) + 1
    s = np.linspace(0, 1, n)
    tau = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    p0, p1 = np.array([0.45, 0.0]), np.array([0.25, 0.45])
    pts = p0[None] + tau[:, None] * (p1 - p0)[None]
    pts[:, 0] *= scale_x
    return pts


def run(T, scale_x=1.0):
    pts = hand_path(T, scale_x)
    q = np.array([0.3, 1.2])
    qs = []
    errs = []
    for p in pts:
        q, e = ik(p, q)
        qs.append(q.copy())
        errs.append(e)
    qs = np.array(qs)
    qd = np.gradient(qs, DT, axis=0)
    qdd = np.gradient(qd, DT, axis=0)
    taus = np.array([inv_dyn(qs[i], qd[i], qdd[i]) for i in range(len(qs))])
    ratio = np.abs(taus) / TAU_MAX
    infeasible_frames = float(np.mean(ratio.max(axis=1) > 1.0))
    return {"T": T, "scale_x": scale_x,
            "max_ik_error_m": float(np.max(errs)),
            "peak_torque_Nm": [float(np.abs(taus[:, 0]).max()), float(np.abs(taus[:, 1]).max())],
            "peak_ratio": float(ratio.max()),
            "infeasible_frame_frac": round(infeasible_frames, 4)}


def min_feasible_T(scale_x, lo=0.1, hi=8.0, iters=22):
    """가능한 가장 빠른 T를 이분 탐색(단조 가정). 전 구간 피크 토크비가 1 이하인 최소 T."""
    if run(hi, scale_x)["peak_ratio"] > 1:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if run(mid, scale_x)["peak_ratio"] <= 1:
            hi = mid
        else:
            lo = mid
    return round(hi, 3)


if __name__ == "__main__":
    res = {"params": {"L1=L2": L1, "M1=M2": M1, "payload_kg": MP, "TAU_MAX": TAU_MAX.tolist(), "dt": DT}}
    # 1) 같은 경로를 세 속도로 실행: 빠름(0.3 s), 중간(0.45 s), 보통(1.5 s)
    res["speed_sweep"] = [run(T) for T in (0.3, 0.45, 1.5)]
    # 2) 기하 증강(x 1.15배) 후 같은 속도
    res["augmented_x115"] = [run(T, 1.15) for T in (0.3, 0.45, 1.5)]
    # 3) 물리 필터: 실행 가능한 최소 T (시간 늘리기)
    res["min_feasible_T"] = {"original_x1.00": min_feasible_T(1.0),
                             "augmented_x1.15": min_feasible_T(1.15)}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=1))
