"""토이 T1: 논문 식 (2)의 거리 게이트 메트릭이 만드는 coarse-to-fine 접근을 평면 로봇에서 본다.

논문 3.1절과 표 4의 주장: 가중 IK는 먼 목표로 팔을 쭉 뻗어 조작성이 낮아지고,
기준 어트랙터(BA, mmin = 0)와 말단 어트랙터(EEA, mfar = 0)의 게이트는 먼 곳은 기저가, 가까운 곳은 팔이 맡게 한다.

로봇: 전방향 기저 (x, y) + 팔 2링크(링크 0.4 m 둘, 평면). 목표: 기저에서 2~3 m 떨어진 점의 위치(자세 제외).
비교: (a) 가중 감쇠 IK (기저 가중 w_b 를 1, 3, 10 으로 변화), (b) 게이트 어트랙터 + 자세 정칙화 (논문 부록 C의 EEA, BA, PR 형태를
2차 동역학으로 조합. 에너지화(energization)와 충돌 회피 항은 넣지 않음).
지표: 팔 조작성 w = l1 l2 |sin q2| 의 시간 평균과 최솟값, 팔 신장비(|ee - base| / (l1 + l2))의 최댓값,
수렴 시간(오차 1 cm 이하), 기저 이동 거리.

한계: 평면 2링크이고 충돌, 시야, 관절 한계가 없다. 가중치와 게이트 반경은 이 토이에서 정한 값이며 논문 값이 아니다.
결과는 표 4를 재현하는 것이 아니라 "게이트가 있으면 팔 신장을 늦춘다"는 메커니즘의 확인이다.
실행: python3 analysis/toy_gate_reaching.py   (numpy, 수 초)
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_gate_reaching_results.json"
L1 = L2 = 0.4
REACH = L1 + L2
DT = 1 / 60
STEPS = 60 * 20


def fk(q):
    x, y, q1, q2 = q
    ee = np.array([x + L1 * np.cos(q1) + L2 * np.cos(q1 + q2), y + L1 * np.sin(q1) + L2 * np.sin(q1 + q2)])
    return ee


def jac(q):
    x, y, q1, q2 = q
    J = np.zeros((2, 4))
    J[0, 0] = J[1, 1] = 1.0
    J[0, 2] = -L1 * np.sin(q1) - L2 * np.sin(q1 + q2)
    J[1, 2] = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    J[0, 3] = -L2 * np.sin(q1 + q2)
    J[1, 3] = L2 * np.cos(q1 + q2)
    return J


def metrics(traj, target):
    traj = np.array(traj)
    ee = np.array([fk(q) for q in traj])
    err = np.linalg.norm(ee - target, axis=1)
    arm_ext = np.linalg.norm(ee - traj[:, :2], axis=1) / REACH
    manip = L1 * L2 * np.abs(np.sin(traj[:, 3]))
    conv = np.where(err < 0.01)[0]
    base_dist = float(np.sum(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1)))
    return {
        "converge_s": float(conv[0] * DT) if len(conv) else None,
        "final_err_m": float(err[-1]),
        "arm_extension_max": float(arm_ext.max()),
        "manip_mean": float(manip.mean()),
        "manip_min": float(manip.min()),
        "base_travel_m": base_dist,
    }


def run_ik(q0, target, w_base, lam=0.05, gain=3.0, vmax=1.0):
    """가중 감쇠 IK: min |J dq - gain e|^2 + lam |W dq|^2. 기저 가중 w_base > 1 이면 기저 이동을 벌한다."""
    q = q0.copy()
    W2 = np.diag([w_base ** 2, w_base ** 2, 1.0, 1.0])
    traj = [q.copy()]
    for _ in range(STEPS):
        e = target - fk(q)
        J = jac(q)
        dq = np.linalg.solve(J.T @ J + lam * W2, J.T @ (gain * e))
        dq = np.clip(dq, -vmax, vmax)
        q = q + dq * DT
        q[3] = np.clip(q[3], -2.8, 2.8)
        traj.append(q.copy())
    return traj


def gate(dist, r, alpha=6.0):
    """식 (2)의 0.5 tanh(-alpha (dist - r)) + 0.5. 멀면 0, 가까우면 1."""
    return 0.5 * np.tanh(-alpha * (dist - r)) + 0.5


def run_fabric(q0, target, r_base=1.0, r_ee=0.9, k=6.0, d_gain=4.0, eta=1.0):
    """게이트 어트랙터 (BA: m_min = 0, EEA: m_far = 0) + 약한 자세 정칙화(PR) + 관절 감쇠. 에너지화 없음."""
    q = q0.copy()
    qd = np.zeros(4)
    q_nom = np.array([0, 0, 0.6, 1.4])   # 접힌 자세 (PR 목표)
    traj = [q.copy()]
    for _ in range(STEPS):
        ee = fk(q)
        J = jac(q)
        Jb = np.zeros((2, 4)); Jb[0, 0] = Jb[1, 1] = 1.0   # 기저 평면 위치 작업 공간
        # BA: 기저가 (목표 방향으로) 목표 점 가까이 가도록. 게이트는 EE 목표까지의 거리 기준 (본문 3.1절)
        dist_t = np.linalg.norm(target - ee)
        m_ba = eta * (1.0 * (1 - gate(dist_t, r_base)) + 0.0 * gate(dist_t, r_base))   # m_far = 1, m_min = 0
        m_ee = eta * (0.0 * (1 - gate(dist_t, r_ee)) + 1.0 * gate(dist_t, r_ee))        # m_far = 0, m_min = 1
        e_b = q[:2] - target
        rho_b = np.linalg.norm(e_b) + 1e-9
        a_b = -k * np.tanh(2.0 * rho_b) * e_b / rho_b - d_gain * (Jb @ qd)
        e_e = ee - target
        rho_e = np.linalg.norm(e_e) + 1e-9
        a_e = -k * np.tanh(2.0 * rho_e) * e_e / rho_e - d_gain * (J @ qd)
        M = Jb.T @ (m_ba * np.eye(2)) @ Jb + J.T @ (m_ee * np.eye(2)) @ J
        f = Jb.T @ (-(m_ba * np.eye(2)) @ a_b) + J.T @ (-(m_ee * np.eye(2)) @ a_e)
        # PR: 팔 관절만, 작은 등방 메트릭
        eta_pr = 0.02
        Mpr = np.diag([0, 0, eta_pr, eta_pr])
        a_pr = np.zeros(4); a_pr[2:] = -2.0 * (q[2:] - q_nom[2:]) - 2.0 * qd[2:]
        M = M + Mpr + 1e-3 * np.eye(4)
        f = f + (-Mpr @ a_pr)
        qdd = -np.linalg.solve(M, f)
        qd = np.clip(qd + qdd * DT, -1.5, 1.5)
        q = q + qd * DT
        q[3] = np.clip(q[3], -2.8, 2.8)
        traj.append(q.copy())
    return traj


def main():
    rng = np.random.default_rng(0)
    results = {}
    dists = [2.0, 2.5, 3.0]
    q0_base = np.array([0.0, 0.0, 0.6, 1.4])
    configs = {"weighted_IK_wb1": ("ik", 1.0), "weighted_IK_wb3": ("ik", 3.0), "weighted_IK_wb10": ("ik", 10.0), "gated_fabric": ("fab", None)}
    for name, (kind, wb) in configs.items():
        per = []
        for trial in range(30):
            d = rng.choice(dists) + rng.uniform(-0.2, 0.2)
            ang = rng.uniform(-np.pi / 3, np.pi / 3)
            target = d * np.array([np.cos(ang), np.sin(ang)])
            q0 = q0_base.copy()
            q0[2] += ang   # 팔이 대략 목표 쪽을 향한 접힌 자세로 시작
            traj = run_ik(q0, target, wb) if kind == "ik" else run_fabric(q0, target)
            per.append(metrics(traj, target))
        agg = {}
        for key in per[0]:
            vals = [p[key] for p in per if p[key] is not None]
            agg[key] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals)), "n": len(vals)}
        results[name] = agg
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'config':20s} {'converge_s':>11s} {'arm_ext_max':>12s} {'manip_mean':>11s} {'manip_min':>10s} {'base_m':>8s}")
    for name, agg in results.items():
        print(f"{name:20s} {agg['converge_s']['mean']:11.2f} {agg['arm_extension_max']['mean']:12.3f} "
              f"{agg['manip_mean']['mean']:11.3f} {agg['manip_min']['mean']:10.3f} {agg['base_travel_m']['mean']:8.2f}"
              f"  (수렴 {agg['converge_s']['n']}/30)")


if __name__ == "__main__":
    main()
