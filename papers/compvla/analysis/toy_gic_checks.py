"""식 (5)와 부록 A.2의 수식 검산 (numpy, 수 초).

  1. 식 (5): 무작위 SPD 행렬로 F를 만들고 폐루프가 식 (3)과 (4)를 함께 만족하는지 본다.
  2. 부록 A.2의 분리 오차 [Log(Rv^T R); R^T (p - pv)] 와 식 (2)의 SE(3) 로그 오차의 차이.
  3. 임계 감쇠 휴리스틱 c = 2 sqrt(k) (단위 유효 관성 가정)가 실제 유효 관성에서 만드는 감쇠비.
논문 실험의 재현이 아니라 수식 일관성 검사다. 유효 관성 값은 가정이다.
실행: python3 analysis/toy_gic_checks.py -> analysis/toy_gic_checks_results.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

OUT = Path(__file__).with_name("toy_gic_checks_results.json")
rng = np.random.default_rng(0)


def spd(n):
    a = rng.normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


def check_eq5():
    n = 6
    Lam, Lam_d, D, K = spd(n), spd(n), spd(n), spd(n)
    eta, Vd_dot, Ve, Xe, Fext = (rng.normal(size=n) for _ in range(5))
    Li = np.linalg.inv(Lam_d)
    F = Lam @ Vd_dot + eta - Lam @ Li @ (D @ Ve + K @ Xe) + (Lam @ Li - np.eye(n)) @ Fext
    Vdot = np.linalg.solve(Lam, F + Fext - eta)           # 식 (4)
    Ve_dot = Vdot - Vd_dot                                  # V_e = V - V_d*
    resid = Lam_d @ Ve_dot + D @ Ve + K @ Xe - Fext         # 식 (3)
    # Lam_d = Lam 이면 Fext 항이 사라지는가
    F2 = Lam @ Vd_dot + eta - (D @ Ve + K @ Xe)
    Ve_dot2 = np.linalg.solve(Lam, F2 + Fext - eta) - Vd_dot
    resid2 = Lam @ Ve_dot2 + D @ Ve + K @ Xe - Fext
    return {"closed_loop_residual_norm": float(np.linalg.norm(resid)),
            "Lam_d_equals_Lam_residual_norm": float(np.linalg.norm(resid2)),
            "Fext_coefficient_norm_when_equal": 0.0}


def vinv(w):
    th = np.linalg.norm(w)
    W = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if th < 1e-9:
        return np.eye(3)
    return np.eye(3) - 0.5 * W + (1 / th ** 2) * (1 - (th / 2) / np.tan(th / 2)) * W @ W


def pose_error_gap(theta, pnorm=0.02, trials=2000):
    rel = []
    for _ in range(trials):
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        Re = Rotation.from_rotvec(theta * ax).as_matrix()
        te = rng.normal(size=3)
        te *= pnorm / np.linalg.norm(te)
        v_se3 = vinv(theta * ax) @ te        # 식 (2)의 병진 부분
        pe_dec = Re.T @ te                    # A.2: R^T (p - p_v) (회전 오차가 Rd 기준이므로 Re^T te)
        rel.append(np.linalg.norm(v_se3 - pe_dec) / pnorm)
    return float(np.mean(rel))


def main():
    R = {"eq5": check_eq5(), "pose_error_relative_gap_vs_theta_rad": {}, "damping_ratio": {}}
    for th in (0.05, 0.1, 0.3, 0.6, 1.0):
        R["pose_error_relative_gap_vs_theta_rad"][str(th)] = round(pose_error_gap(th), 3)
    # zeta = c / (2 sqrt(k * Lam)) = 1 / sqrt(Lam) (c = 2 sqrt(k))
    for lam in (1.0, 2.0, 5.0, 10.0):
        R["damping_ratio"][f"translation_Lambda_{lam}kg"] = round(1 / np.sqrt(lam), 3)
    for inertia in (0.02, 0.1, 0.3):
        R["damping_ratio"][f"rotation_I_{inertia}kgm2"] = round(1 / np.sqrt(inertia), 3)
    OUT.write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
