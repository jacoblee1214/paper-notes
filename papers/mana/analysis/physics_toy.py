"""물리 toy 세 가지 (CPU, numpy 단일 스레드, 수 초). 실행: python3 analysis/physics_toy.py

P1 손가락 힘 용량: 관절 토크 상한 0.7 Nm(Sec. 6)으로 평면 3링크 손가락이 손끝에서 낼 수 있는 최대 힘.
   "10 N이 넘는 도구는 못 다룬다"(Sec. 6)와 "3–7 N이 필요하다"(Sec. 5.1)를 자세별로 본다.
P2 적분기 안정성: 스프링 구동 도구 관절(부록 A.6)을 dt = 1/200 s로 적분할 때 명시적 Euler, 반암시적(symplectic) Euler,
   암시적 Euler의 안정성. "Implicit Euler가 결정적"(A.6)이라는 주장의 크기를 본다.
P3 랜덤화 크기: 부록 A.3의 범위를 필요한 힘(3–7 N)과 같은 단위로 환산한다.

가정(논문에 없는 값)은 PARAMS에 모았다. 링크 길이는 Allegro 손가락 치수에 대한 사전 지식에 맞춘 근사값이다.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARAMS = {
    "tau_max_Nm": 0.7,                      # Sec. 6
    "links_m": [0.054, 0.0384, 0.040],      # 가정: 근위·중위·원위(+손끝 패드), Allegro 근사
    "longest_link_paper_m": [0.06, 0.07],   # Sec. 6
    "force_needed_N": [3.0, 7.0],           # Sec. 5.1
    "dt_s": 1 / 200,                        # A.6
    "stroke_m": [0.01, 0.02],               # 가정: 도구 관절 행정
    "eff_mass_kg": [0.005, 0.05],           # 가정: 손잡이·플런저 유효 질량
    "tool_mass_kg": [0.03, 0.1, 0.2],       # 가정: 도구 질량
}


# ── P1 ───────────────────────────────────────────────────────────────────────
def planar_jacobian(q, L):
    """평면 3링크 손끝 Jacobian (2×3)."""
    c = np.cumsum(q)
    J = np.zeros((2, 3))
    for i in range(3):
        J[0, i] = -sum(L[j] * math.sin(c[j]) for j in range(i, 3))
        J[1, i] = sum(L[j] * math.cos(c[j]) for j in range(i, 3))
    return J


def max_force(J, d, tau):
    """방향 d로 낼 수 있는 최대 힘: 모든 관절에서 |J_iᵀ d| f ≤ τ."""
    lever = np.abs(J.T @ d).max()
    return float("inf") if lever < 1e-9 else float(tau / lever)


def p1():
    L, tau = PARAMS["links_m"], PARAMS["tau_max_Nm"]
    grid = np.radians(np.linspace(0, 90, 31))
    f_pad, f_best = [], []
    for q1 in grid:
        for q2 in grid:
            for q3 in grid:
                q = np.array([q1, q2, q3])
                J = planar_jacobian(q, L)
                c = q.sum()
                n = np.array([-math.sin(c), math.cos(c)])  # 원위 링크에 수직, 굽힘 쪽(패드가 누르는 방향)
                f_pad.append(max_force(J, n, tau))
                # 방향을 자유롭게 고를 때의 최대
                dirs = [np.array([math.cos(a), math.sin(a)]) for a in np.linspace(0, 2 * math.pi, 72, endpoint=False)]
                f_best.append(min(50.0, max(max_force(J, d, tau) for d in dirs)))
    f_pad, f_best = np.array(f_pad), np.array(f_best)
    single = [round(tau / l, 1) for l in PARAMS["longest_link_paper_m"]]
    return {
        "single_joint_bound_tau_over_L_N": single,
        "pad_normal_force_N": {"median": round(float(np.median(f_pad)), 2),
                               "p90": round(float(np.quantile(f_pad, 0.9)), 2),
                               "max": round(float(f_pad.max()), 2)},
        "share_postures_pad_force_ge": {f"{t}N": round(float((f_pad >= t).mean()), 3) for t in (3, 5, 7, 10)},
        "share_postures_any_dir_ge_7N": round(float((f_best >= 7).mean()), 3),
        "extended_finger_pad_force_N": round(max_force(planar_jacobian(np.zeros(3), L), np.array([0.0, 1.0]), tau), 2),
        "note": "한 관절의 레버암 τ/L는 상한일 뿐이다. 근위 관절은 손끝까지 거리가 최대 0.13 m라 펴진 자세에서는 패드 방향 힘이 5 N 남짓이다",
    }


# ── P2 ───────────────────────────────────────────────────────────────────────
def amplification(omega, zeta, dt, method):
    """선형 감쇠 진동자 x'' + 2ζω x' + ω² x = 0 한 스텝 행렬의 스펙트럼 반경."""
    w2, c = omega ** 2, 2 * zeta * omega
    if method == "explicit":
        A = np.array([[1, dt], [-w2 * dt, 1 - c * dt]])
    elif method == "semi_implicit":   # v를 먼저 갱신하고 새 v로 x 갱신
        A = np.array([[1 - w2 * dt * dt, dt * (1 - c * dt)], [-w2 * dt, 1 - c * dt]])
    else:                             # 암시적(backward) Euler
        M = np.array([[1, -dt], [w2 * dt, 1 + c * dt]])
        A = np.linalg.inv(M)
    return float(max(abs(np.linalg.eigvals(A))))


def p2():
    dt = PARAMS["dt_s"]
    rows = {}
    for F in PARAMS["force_needed_N"]:
        for s in PARAMS["stroke_m"]:
            for m in PARAMS["eff_mass_kg"]:
                k = F / s
                om = math.sqrt(k / m)
                key = f"F{F:g}N_s{s * 100:g}cm_m{m * 1000:g}g"
                rows[key] = {
                    "k_N_per_m": round(k), "omega_rad_s": round(om, 1), "omega_dt": round(om * dt, 3),
                    **{f"rho_{meth}_zeta{z}": round(amplification(om, z, dt, meth), 4)
                       for meth in ("explicit", "semi_implicit", "implicit") for z in (0.0, 0.2)},
                    "explicit_stable_needs_dt_below_ms_zeta0.2": round(1000 * 2 * 0.2 / om, 2),
                }
    # 대표 경우: 1초 동안 진폭 변화
    k, m = 7.0 / 0.01, 0.01
    om = math.sqrt(k / m)
    growth = {meth: float(f"{amplification(om, 0.2, dt, meth) ** 200:.3g}")
              for meth in ("explicit", "semi_implicit", "implicit")}
    return {"cases": rows, "one_second_amplitude_factor_F7N_s1cm_m10g_zeta0.2": growth,
            "omega_for_that_case": round(om, 1)}


# ── P3 ───────────────────────────────────────────────────────────────────────
def p3():
    g = 9.81
    perturb = {f"m{int(m * 1000)}g": [round(a * m * g, 3) for a in (0.05, 0.2)] for m in PARAMS["tool_mass_kg"]}
    lo, hi = PARAMS["force_needed_N"]
    return {
        "object_force_sigma_N_alpha_0.05_to_0.2": perturb,
        "sigma_over_actuation_force_max": round(0.2 * max(PARAMS["tool_mass_kg"]) * g / lo, 3),
        "tool_gain_x0.8_1.5_actuation_range_N": [round(lo * 0.8, 1), round(hi * 1.5, 1)],
        "exceeds_10N_hardware_limit": hi * 1.5 > 10,
        "robot_gain_x0.8_1.2_fingertip_force_scale": [0.8, 1.2],
        "friction_x0.5_1.5_cone_half_angle_deg_if_mu0=1": [round(math.degrees(math.atan(0.5)), 1),
                                                           round(math.degrees(math.atan(1.5)), 1)],
        "action_noise_U(-0.1,0.1)_share_of_range": 0.1,
        "note": "물체 섭동(σ = αmg)은 필요한 구동력의 수 % 수준이다. 힘 쪽 견고성은 주로 이득·마찰 랜덤화에서 온다(해석)",
    }


def main():
    out = {"params": PARAMS, "P1_force_capacity": p1(), "P2_integrator": p2(), "P3_randomization": p3()}
    path = ROOT / "analysis" / "physics_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "P2_integrator"}, indent=1, ensure_ascii=False)[:2500])
    print("P2 1s factor:", out["P2_integrator"]["one_second_amplitude_factor_F7N_s1cm_m10g_zeta0.2"])
    print("wrote", path)


if __name__ == "__main__":
    main()
