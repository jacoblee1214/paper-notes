"""토이 T2: 논문 부록 B.1과 C.1의 해석적 역변환이 맞는지 수치로 확인한다 (numpy, 수 초).

확인 1. 불감대(dead zone) 식 B.4와 역식 B.7의 왕복 오차 (양/음 방향 δ가 다를 때 포함).
확인 2. 액추에이터 전진 모델 B.5와 역변환(B.6, B.7, B.8)의 왕복 오차 (스케일, 바이어스, 감쇠, 시그모이드 마찰, 불감대).
확인 3. 교사 토크 식 1과 C.1의 Φξ: 목표 팔에 τ* = Φξ(q, qd, q̈0)을 주면 목표 팔의 가속도가 이상 모델의 q̈0과 같은가.
        (토이 T1의 팔 모델에서 확인. 불감대는 이 팔에 없으므로 확인 3은 스케일·바이어스·감쇠·마찰 부분집합이다.)

실행: python3 analysis/toy_inverse_roundtrip.py
출력: analysis/toy_inverse_roundtrip_results.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import toy_tam_2link as T1  # noqa: E402

OUT = HERE / "toy_inverse_roundtrip_results.json"
EPS = 1e-2   # 논문 식 B.4의 ε
K_SIG = 20.0


def dz_fwd(x, dp, dm):
    d = np.where(x >= 0, dp, dm)
    return np.where(np.abs(x) <= d, EPS * x, np.sign(x) * (np.abs(x) - d + EPS * d))


def dz_inv(y, dp, dm):
    d = np.where(y >= 0, dp, dm)
    return np.where(np.abs(y) <= EPS * d, y / EPS, np.sign(y) * (np.abs(y) - EPS * d + d))


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def act_fwd(tau0, qd, p):
    """B.5: tau_a = dz(scale(tau0)) + bias(qd) - damp(qd) - fric(qd)."""
    s = np.where(tau0 >= 0, p["ap"] * tau0, p["am"] * tau0)
    x = dz_fwd(s, p["dp"], p["dm"])
    bias = np.where(qd >= 0, p["bp"], p["bm"])
    damp = np.where(qd >= 0, p["dampp"] * qd, p["dampm"] * qd)
    fric = np.where(qd >= 0,
                    p["Ap"] * (sig(p["kp"] * (qd + p["sp"])) - sig(p["kp"] * p["sp"])),
                    p["Am"] * (sig(p["km"] * (qd + p["sm_"])) - sig(p["km"] * p["sm_"])))
    return x + bias - damp - fric


def act_inv(tau_star, qd, p):
    """B.6-B.8: y = tau* + damp + fric - bias, z = dz^-1(y), tau0 = scale^-1(z)."""
    bias = np.where(qd >= 0, p["bp"], p["bm"])
    damp = np.where(qd >= 0, p["dampp"] * qd, p["dampm"] * qd)
    fric = np.where(qd >= 0,
                    p["Ap"] * (sig(p["kp"] * (qd + p["sp"])) - sig(p["kp"] * p["sp"])),
                    p["Am"] * (sig(p["km"] * (qd + p["sm_"])) - sig(p["km"] * p["sm_"])))
    y = tau_star + damp + fric - bias
    z = dz_inv(y, p["dp"], p["dm"])
    return np.where(z >= 0, z / p["ap"], z / p["am"])


def random_actuator(rng, n):
    p = dict(
        ap=rng.uniform(0.97, 1.03, n), am=rng.uniform(0.97, 1.03, n),
        dp=rng.uniform(0, 1, n), dm=rng.uniform(0, 1, n),
        bp=rng.uniform(-1, 1, n), bm=None,
        dampp=rng.uniform(0, 2, n), dampm=rng.uniform(0, 2, n),
        Ap=rng.uniform(0.005, 3.0, n), Am=rng.uniform(0.005, 3.0, n),
        kp=rng.uniform(1 / 0.2, 1 / 0.02, n), km=rng.uniform(1 / 0.2, 1 / 0.02, n),
        sp=rng.uniform(-0.02, 0.02, n), sm_=rng.uniform(-0.02, 0.02, n),
    )
    p["bm"] = p["bp"] + rng.uniform(-0.2, 0.2, n)   # B.1: tau_b- = tau_b+ + U[-0.2, 0.2]
    return p


def main():
    rng = np.random.default_rng(7)
    res = {}
    # 확인 1: 불감대 왕복
    x = rng.uniform(-3, 3, 200000)
    dp, dm = rng.uniform(0, 1, x.size), rng.uniform(0, 1, x.size)
    back = dz_inv(dz_fwd(x, dp, dm), dp, dm)
    res["deadzone_roundtrip_max_abs_err"] = float(np.max(np.abs(back - x)))
    res["deadzone_samples"] = int(x.size)
    # 확인 2: 액추에이터 왕복 (순방향 후 역방향으로 τ*를 되찾는다)
    n = 200000
    p = random_actuator(rng, n)
    qd = rng.uniform(-2, 2, n)
    tau0 = rng.uniform(-10, 10, n)
    tau_a = act_fwd(tau0, qd, p)
    tau0_back = act_inv(tau_a, qd, p)
    # 불감대 안쪽(|x| <= δ)에서는 역변환이 ε로 나눗셈하므로 정보가 손실되지 않는다. 기대 오차는 0이다.
    res["actuator_roundtrip_tau0_max_abs_err"] = float(np.max(np.abs(tau0_back - tau0)))
    res["actuator_roundtrip_tau0_median_abs_err"] = float(np.median(np.abs(tau0_back - tau0)))
    res["actuator_samples"] = n
    # 확인 3: 교사 토크 (식 1, C.1)
    m = 2000
    P = dict(sm=rng.uniform(0.9, 1.1, m), b=rng.uniform(0, 0.5, (m, 2)), A=rng.uniform(0, 1.0, (m, 2)),
             alpha=rng.uniform(0.97, 1.03, (m, 2)), beta=rng.uniform(-0.1, 0.1, (m, 2)))
    q = rng.uniform(-1.5, 1.5, (m, 2))
    qd = rng.uniform(-2, 2, (m, 2))
    tau0 = rng.uniform(-5, 5, (m, 2))
    qdd0 = T1.forward_acc(q, qd, tau0, T1.IDEAL)           # 이상 모델 F0
    tau_star = T1.inverse_torque(q, qd, qdd0, P)            # Φξ
    qdd_tgt = T1.forward_acc(q, qd, tau_star, P)            # 목표 팔 Fξ
    res["teacher_match_max_abs_qdd_err"] = float(np.max(np.abs(qdd_tgt - qdd0)))
    res["teacher_match_median_abs_qdd_err"] = float(np.median(np.abs(qdd_tgt - qdd0)))
    res["teacher_samples"] = m
    # 참고: 교사 보정을 주지 않은 직접 제어의 가속도 오차 (규모 비교용)
    qdd_direct = T1.forward_acc(q, qd, tau0, P)
    res["direct_max_abs_qdd_err_for_scale"] = float(np.max(np.abs(qdd_direct - qdd0)))
    res["direct_median_abs_qdd_err_for_scale"] = float(np.median(np.abs(qdd_direct - qdd0)))
    res["pass"] = bool(res["deadzone_roundtrip_max_abs_err"] < 1e-9 and
                       res["actuator_roundtrip_tau0_max_abs_err"] < 1e-9 and
                       res["teacher_match_max_abs_qdd_err"] < 1e-8)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
