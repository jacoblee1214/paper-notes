"""추가 검증 (stiffness_copilot): MuJoCo 접촉 모델로 벽을 만든 평면 2링크 팔의 임피던스 제어 비교.

실행: .venv/bin/python analysis/extra_copilot_contact.py
결과: analysis/extra_copilot_contact_results.json

앞선 추가 검증 4(extra_copilot_mujoco.py)는 벽을 스프링-마찰 외력으로 계산했다. 이 스크립트는 같은 종류의
조건(고강성, 저강성, 법선·접선 방향별 강성, 법선 오차)을 MuJoCo 접촉 솔버로 된 벽에서 다시 돌려, 힘과
추적의 분리가 접촉 모델을 바꿔도 남는지 본다.

- 팔: 수평면 2링크(링크 0.3 m), 관절 두 개(수직축), 중력 없음. 팁은 질량 1 kg 구(반지름 5 mm).
  관절 토크는 Jᵀ F를 qfrc_applied로 넣는다. 제어 주기 1 kHz(MuJoCo 적분 dt = 1 ms).
- 벽: 정적 상자. 법선 n(θ) = (sin θ, cos θ), 벽 표면은 n·p = C (C = 0.45 m). 접촉 파라미터는
  solref(시간 상수 0.0149 s, 감쇠비 1)로 맞췄다. 이는 유효 질량 약 1 kg에서 법선 강성이 약 5000 N/m 근처가
  되도록 한 값이며, 측정된 법선 힘과 스프링 모형의 값을 비교해 확인한다. 마찰 μ = 0.4(쿨롱 마찰, condim 3).
- 임피던스: F = K(x_d − x) + D(ẋ_d − ẋ), K = k_n n̂n̂ᵀ + k_t t̂t̂ᵀ, D = 1.4√k를 축별로 쓴다(1 kg).
- 조작자: 0–1 s에 표면 2 cm 위에서 표면 δ 아래까지 내려가고, 1–2 s에 접선으로 0.05 m/s로 이동한다.
  측정: 법선 힘 최댓값, 2–3 s 법선 힘 평균(F_ss), 2–3 s 접선 추적 오차 RMS.
- 조건: (A) θ = 0, δ = 0.5, 1, 2 cm. (B) δ = 1 cm, θ ∈ {−30°, −15°, 0°, 15°, 30°}, 법선 정확 또는 +20° 오차.
  결정론적 시뮬레이션이라 시드 반복은 없다.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"

import json
from pathlib import Path

import mujoco
import numpy as np

RESULTS = Path(__file__).resolve().parent / "extra_copilot_contact_results.json"

L1 = L2 = 0.30
TIP_MASS = 1.0
TIP_R = 0.005
C = 0.45                       # wall plane n·p = C
MU = 0.4
SOLREF = "0.0149 1"            # contact time constant, damping ratio: ~5000 N/m with m_eff ~ 1 kg
K_ENV_SPRING = 5000.0          # spring model stiffness used by the earlier check (for comparison)
DT = 0.001
T_END = 3.0
N = int(T_END / DT)
V_TAN = 0.05
DEG = np.pi / 180.0


def build_model(theta):
    phi = -theta                              # rotation about z that maps +y onto n(θ)
    qw, qz = np.cos(phi / 2), np.sin(phi / 2)
    cx, cy = (C + 0.01) * np.sin(theta), (C + 0.01) * np.cos(theta)
    xml = f"""
<mujoco>
  <compiler angle="radian"/>
  <option gravity="0 0 0" timestep="{DT}" integrator="implicitfast"/>
  <default><joint type="hinge" axis="0 0 1" damping="0.5" armature="1e-3"/></default>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="j1" range="-3.14 3.14"/>
      <geom type="capsule" fromto="0 0 0 {L1} 0 0" size="0.012" mass="0.05" contype="0" conaffinity="0"/>
      <body name="link2" pos="{L1} 0 0">
        <joint name="j2" range="-3.14 3.14"/>
        <geom type="capsule" fromto="0 0 0 {L2} 0 0" size="0.012" mass="0.05" contype="0" conaffinity="0"/>
        <body name="tip" pos="{L2} 0 0">
          <geom name="tip_g" type="sphere" size="{TIP_R}" mass="{TIP_MASS}" friction="{MU} 0.005 0.0001"
                condim="3" solref="{SOLREF}" contype="1" conaffinity="1"/>
          <site name="tip_s" pos="0 0 0" size="0.001"/>
        </body>
      </body>
    </body>
    <body name="wall" pos="{cx:.6f} {cy:.6f} 0" quat="{qw:.6f} 0 0 {qz:.6f}">
      <geom name="wall_g" type="box" size="0.25 0.01 0.1" friction="{MU} 0.005 0.0001"
            condim="3" solref="{SOLREF}" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def fk_tip(d):
    return d.site_xpos[0][:2].copy()


def run_trial(theta, delta, k_n, k_t, normal_err=0.0):
    """k_n, k_t: stiffness along the true normal and tangent (possibly rotated by normal_err)."""
    m = build_model(theta)
    d = mujoco.MjData(m)
    site = m.site("tip_s").id
    wall = m.geom("wall_g").id
    # initial posture: put the tip at C - 0.02 along n, tangent coordinate 0
    n = np.array([np.sin(theta), np.cos(theta)])
    t = np.array([np.cos(theta), -np.sin(theta)])
    x_start = (C - 0.02) * n
    # IK for the 2-link arm in the plane (elbow positive)
    r = x_start
    c2 = (r @ r - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    q2 = np.arccos(np.clip(c2, -1, 1))
    q1 = np.arctan2(r[1], r[0]) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    d.qpos[:] = [q1, q2]
    mujoco.mj_forward(m, d)

    # controller directions (normal error rotates the assumed normal inside the plane)
    ang = normal_err * DEG
    na = np.array([np.cos(ang) * n[0] - np.sin(ang) * n[1], np.sin(ang) * n[0] + np.cos(ang) * n[1]])
    ta = np.array([-na[1], na[0]])
    K = k_n * np.outer(na, na) + k_t * np.outer(ta, ta)
    Dm = 1.4 * (np.sqrt(k_n) * np.outer(na, na) + np.sqrt(k_t) * np.outer(ta, ta))   # D = 1.4 sqrt(k) per axis

    jacp = np.zeros((3, m.nv))
    f_norm, track_err = [], []
    x0 = x_start
    for i in range(N):
        tt = i * DT
        # along n: from 2 cm short of the surface (n·p = C) to δ beyond it, in 1 s; then slide tangentially
        depth = (C - 0.02) + (0.02 + delta) * min(tt, 1.0)
        xd = depth * n + V_TAN * max(min(tt - 1.0, 1.0), 0.0) * t
        # x_d projected: n-part and t-part
        x = fk_tip(d)
        mujoco.mj_jacSite(m, d, jacp, None, site)
        J = jacp[:2, :m.nv]
        v = J @ d.qvel[:m.nv]
        vd = np.zeros(2)
        F = K @ (xd - x) + Dm @ (vd - v)
        d.qfrc_applied[:m.nv] = J.T @ F
        mujoco.mj_step(m, d)
        if tt >= 2.0:
            # normal force from the tip–wall contact
            fn = 0.0
            f6 = np.zeros(6)
            for c in range(d.ncon):
                ct = d.contact[c]
                if wall in (ct.geom1, ct.geom2):
                    mujoco.mj_contactForce(m, d, c, f6)
                    fn += f6[0]
            f_norm.append(fn)
            xtr = fk_tip(d)
            # tangential tracking error of the commanded tangent position
            track_err.append(float((xd - xtr) @ t))
    f_norm = np.array(f_norm)
    track_err = np.array(track_err)
    return dict(F_peak=float(max(f_norm.max(), 0.0)), F_ss=float(f_norm[: len(f_norm) // 2].mean()),
                track_rms_mm=float(np.sqrt(np.mean(track_err ** 2)) * 1e3))


def main():
    out = {"setup": dict(L1=L1, L2=L2, tip_mass=TIP_MASS, wall_plane_C=C, mu=MU, solref=SOLREF,
                         spring_reference_k_env=K_ENV_SPRING, dt=DT, T_end=T_END)}
    k_hi, k_lo = 3000.0, 300.0
    rows = {}
    for delta in (0.005, 0.01, 0.02):
        rows[f"A_delta={delta}"] = {
            "fixed_high_3000": run_trial(0.0, delta, k_hi, k_hi),
            "fixed_low_300": run_trial(0.0, delta, k_lo, k_lo),
            "dir_normal_300_tangent_3000": run_trial(0.0, delta, k_lo, k_hi),
        }
    for theta_deg in (-30, -15, 0, 15, 30):
        th = theta_deg * DEG
        rows[f"B_theta={theta_deg}"] = {
            "fixed_high_3000": run_trial(th, 0.01, k_hi, k_hi),
            "fixed_low_300": run_trial(th, 0.01, k_lo, k_lo),
            "dir_normal_300_tangent_3000": run_trial(th, 0.01, k_lo, k_hi),
            "dir_normal_err_plus20": run_trial(th, 0.01, k_lo, k_hi, normal_err=20.0),
        }
    out["results"] = rows
    RESULTS.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in rows.items():
        print(k)
        for a, b in v.items():
            print("   ", a, {x: round(y, 3) for x, y in b.items()})


if __name__ == "__main__":
    main()
