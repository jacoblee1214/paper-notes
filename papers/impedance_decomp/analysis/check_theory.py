"""논문 부록 B, C의 수식 주장을 수치로 확인한다 (V6-V8).

V6 표 3: 단일 관절 원시 요소의 생성자 집합이 Lie 부분대수인지 (괄호 닫힘)
V7 표 4: 2R 연쇄(U-joint, 평행 오프셋 crank, 문 경첩+레버)의 전역 포락선 차원
V8 식 (54): d_perp = sqrt(r) d_par 와 표 6의 감쇠 값
실행: python3 check_theory.py  (CPU, 1초 미만)
"""
import json
import math
from pathlib import Path

import numpy as np


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0.0]])


def bracket(a, b):  # 식 (23)
    w1, v1, w2, v2 = a[:3], a[3:], b[:3], b[3:]
    return np.concatenate([np.cross(w1, w2), np.cross(w1, v2) - np.cross(w2, v1)])


def rank(M, tol=1e-9):
    return int(np.linalg.matrix_rank(M, tol=tol)) if M.size else 0


def closed(gens):
    B = np.stack(gens, axis=1)
    r0 = rank(B)
    for i in range(len(gens)):
        for j in range(len(gens)):
            if rank(np.concatenate([B, bracket(gens[i], gens[j])[:, None]], axis=1)) != r0:
                return False
    return True


def exp_rot(axis, th):
    K = skew(axis)
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def adj_exp(screw, th):  # Ad_{exp(screw * th)} (screw = (omega, v), 단위 회전축)
    w, v = screw[:3], screw[3:]
    R = exp_rot(w, th)
    p = (np.eye(3) * th + (1 - math.cos(th)) * skew(w) + (th - math.sin(th)) * skew(w) @ skew(w)) @ v if np.linalg.norm(w) > 0 else th * v
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, :3] = skew(p) @ R
    Ad[3:, 3:] = R
    return Ad


def main():
    out = {}
    n = np.array([0, 0, 1.0]); p = np.array([0.3, 0.1, 0.0]); h = 0.2
    e = np.eye(3)
    prims = {
        "revolute": [np.concatenate([n, -np.cross(n, p)])],
        "prismatic": [np.concatenate([np.zeros(3), n])],
        "helical": [np.concatenate([n, h * n - np.cross(n, p)])],
        "cylindrical": [np.concatenate([n, -np.cross(n, p)]), np.concatenate([np.zeros(3), n])],
        "spherical": [np.concatenate([e[i], -np.cross(e[i], p)]) for i in range(3)],
        "planar": [np.concatenate([n, np.zeros(3)]), np.concatenate([np.zeros(3), e[0]]), np.concatenate([np.zeros(3), e[1]])],
    }
    out["V6_lie_closed"] = {k: closed(g) for k, g in prims.items()}
    # 반례: 평행 오프셋 2R (crank)과 U-joint는 닫히지 않아야 한다
    L = 0.1
    B2 = np.array([0, 0, 1, 0, 0, 0.0]); B1 = np.array([0, 0, 1, 0, -L, 0.0])
    out["V6_lie_closed"]["crank_2R_home_screws"] = closed([B1, B2])
    out["V6_lie_closed"]["ujoint_2R_home_screws"] = closed([np.array([0, 1, 0, 0, 0, 0.0]), np.array([1, 0, 0, 0, 0, 0.0])])

    # V7: 포락선 차원
    qs = np.linspace(0, 2 * math.pi, 40, endpoint=False)
    def env_dim(B1_home, B2_screw):
        cols = [B2_screw]
        for q in qs:
            cols.append(adj_exp(-B2_screw, q) @ B1_home)  # B1(q2) = Ad_{exp(-B2 q2)} B1
        return rank(np.stack(cols, axis=1))
    ex = np.array([1, 0, 0.0]); ey = np.array([0, 1, 0.0]); ez = np.array([0, 0, 1.0])
    cases = {
        "U-joint": (np.concatenate([ey, np.zeros(3)]), np.concatenate([ex, np.zeros(3)]), 3),
        "parallel-offset crank": (np.concatenate([ez, [0, -L, 0]]), np.concatenate([ez, np.zeros(3)]), 3),
        "door hinge + lever": (np.concatenate([ez, [-L, 0, 0]]), np.concatenate([ex, np.zeros(3)]), 4),
    }
    out["V7_envelope_dim"] = {k: {"numeric": env_dim(a, b), "paper_table4": d, "local_dim": 2} for k, (a, b, d) in cases.items()}
    # 크랭크 본문 식: B1(q) = (0,0,1,-l sin q2, -l cos q2, 0)
    l = 0.1
    chk = max(np.abs(adj_exp(-B2, q) @ np.concatenate([ez, [0, -l, 0]]) - np.array([0, 0, 1, -l * math.sin(q), -l * math.cos(q), 0])).max() for q in qs)
    out["V7_crank_eq27_vs_sec32_maxdiff"] = float(chk)

    # V8: 감쇠 규칙
    r = 0.1
    rows = {}
    for name, (m_bar, k0, zeta) in {"sim": (0.586, 250.0, math.sqrt(0.586)), "real": (7.70, 400.0, math.sqrt(7.70))}.items():
        d0 = 2 * zeta * math.sqrt(k0)
        dperp = 2 * zeta * math.sqrt(r * k0)
        rows[name] = dict(zeta_sqrt_mbar=round(zeta, 4), d_par=round(d0, 2), d_perp=round(dperp, 2), ratio=round(dperp / d0, 4), sqrt_r=round(math.sqrt(r), 4),
                          ell_Lambda=round(math.sqrt({"sim": 0.00622, "real": 0.724}[name] / m_bar), 3),
                          wn_phys_rad_s=round(math.sqrt(k0 / m_bar), 2))
    out["V8_damping"] = rows
    Path(__file__).with_name("check_theory_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
