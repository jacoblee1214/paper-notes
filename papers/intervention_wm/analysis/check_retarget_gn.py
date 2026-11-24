"""부록 A 식 (10)의 가우스-뉴턴 법선방정식을 수치로 확인한다 (numpy, 1초 미만).

4자유도 엄지 사슬은 임의의 3D 직렬 사슬(축 z, y, y, y, 링크 길이 임의)로 대신한다. LEAP의 실제 기구학이 아니다.
식 (9)의 잔차 r_x, r_o, r_s를 쌓아 유한차분 야코비안으로 댐핑 가우스-뉴턴 한 걸음을 계산하고,
부록의 닫힌 식(식 10: J^T J + (lam_o/|v|^2) J^T P J + (lam_s + mu^2) I 와 우변)에서 나온 한 걸음과 비교한다.
실행: python3 analysis/check_retarget_gn.py → analysis/check_retarget_gn_results.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(1)
AX = [np.array([0, 0, 1.0]), np.array([0, 1.0, 0]), np.array([0, 1.0, 0]), np.array([0, 1.0, 0])]
LINK = [np.array([0.03, 0, 0]), np.array([0.04, 0, 0]), np.array([0.035, 0, 0]), np.array([0.03, 0, 0])]


def rot(ax, th):
    k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * k @ k


def fk(q):
    R, p = np.eye(3), np.zeros(3)
    for a, l, th in zip(AX, LINK, q):
        R = R @ rot(a, th)
        p = p + R @ l
    return p


def jac(f, q, h=1e-6):
    f0 = np.atleast_1d(f(q))
    J = np.zeros((len(f0), len(q)))
    for i in range(len(q)):
        dq = np.zeros(len(q))
        dq[i] = h
        J[:, i] = (np.atleast_1d(f(q + dq)) - np.atleast_1d(f(q - dq))) / (2 * h)
    return J


def run(trial):
    lam_o, lam_s, mu = 0.2, 0.05, 0.02
    q_prev = rng.uniform(-0.3, 0.6, 4)
    qk = q_prev + rng.uniform(-0.05, 0.05, 4)
    x_idx = np.array([0.09, 0.03, 0.02]) + 0.01 * rng.standard_normal(3)
    target = fk(rng.uniform(-0.3, 0.6, 4))      # T_GL x_th^G
    m = rng.standard_normal(3)
    m /= np.linalg.norm(m)                       # R_GL n^G
    nhat = lambda q: (x_idx - fk(q)) / np.linalg.norm(x_idx - fk(q))

    def resid(q):
        return np.concatenate([fk(q) - target, np.sqrt(lam_o) * (nhat(q) - m), np.sqrt(lam_s) * (q - q_prev)])

    res0 = resid(qk)
    JR = jac(resid, qk)
    dq_num = np.linalg.solve(JR.T @ JR + mu ** 2 * np.eye(4), -JR.T @ res0)

    J = jac(fk, qk)
    v = x_idx - fk(qk)
    nv = np.linalg.norm(v)
    n = v / nv
    P = np.eye(3) - np.outer(n, n)
    e = target - fk(qk)
    A = J.T @ J + (lam_o / nv ** 2) * J.T @ P @ J + (lam_s + mu ** 2) * np.eye(4)
    b = J.T @ e - (lam_o / nv) * J.T @ P @ m + lam_s * (q_prev - qk)
    dq_paper = np.linalg.solve(A, b)
    # 부호를 뒤집은 변형(우변 opposition 항의 부호가 +일 때)도 같이 본다
    b_wrong = J.T @ e + (lam_o / nv) * J.T @ P @ m + lam_s * (q_prev - qk)
    dq_wrong = np.linalg.solve(A, b_wrong)
    return float(np.linalg.norm(dq_num - dq_paper)), float(np.linalg.norm(dq_num - dq_wrong)), float(np.linalg.norm(dq_num))


res = np.array([run(i) for i in range(200)])
out = {"trials": 200, "max_abs_step_difference_paper_vs_numeric": float(res[:, 0].max()),
       "median_step_norm": float(np.median(res[:, 2])),
       "median_difference_if_opposition_sign_flipped": float(np.median(res[:, 1])),
       "note": "식 (10)의 닫힌 식은 임의 사슬에서 수치 GN 걸음과 일치한다. LEAP의 실제 기구학과 칼리브레이션은 확인하지 않았다."}
Path(ROOT / "analysis" / "check_retarget_gn_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
