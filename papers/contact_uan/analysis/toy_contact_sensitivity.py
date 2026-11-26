#!/usr/bin/env python3
"""토이 T1: MuJoCo 소프트 접촉에서 리셋 높이·기울기 오차가 접촉력에 주는 영향.

논문 주장(§2.1, 그림 3): "+2 mm 루트 높이 오프셋은 수직항력을 약 50% 줄이고, -2도 피치 오프셋은
지지 모멘트를 약 45% 바꾼다." 이를 G1이 아닌 단순 강체(발 하나가 받는 하중 17.5 kg, 발바닥 상자)로
같은 접촉 설정(MuJoCo 기본 solref [0.02, 1], solimp [0.9, 0.95, 0.001, 0.5, 2], condim 3, 마찰 0.6, 5 ms)에서
확인한다. 논문 표 4의 접촉 매개변수는 MuJoCo 기본값과 같다. 로봇의 유효 질량과 발 모양이 다르므로 크기는 참고용이다.

방법: (1) 상자를 위에서 떨어뜨려 평형 높이 z_eq를 찾는다. (2) z_eq에서 dz만큼 옮기고(속도 0) 한 번
mj_forward 한 뒤 접촉 법선력 합을 읽는다. (3) 평형 높이에서 피치를 주고 압력중심(CoP)의 이동을 읽는다.
실행: python3 analysis/toy_contact_sensitivity.py   (mujoco, numpy, 약 1초)
"""
import json
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "toy_contact_sensitivity_results.json")

MASS = 17.5  # kg, 발 하나가 지지하는 질량(G1 약 35 kg의 절반, 양발 지지 가정)
HALF = (0.11, 0.05, 0.01)  # 발바닥 상자 반치수 (m)

XML = f"""
<mujoco>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" friction="1 0.005 0.0001"
          solref="0.02 1" solimp="0.9 0.95 0.001 0.5 2" condim="3"/>
    <body name="foot" pos="0 0 0.05">
      <freejoint/>
      <geom name="sole" type="box" size="{HALF[0]} {HALF[1]} {HALF[2]}" mass="{MASS}"
            friction="0.6 0.005 0.0001" solref="0.02 1" solimp="0.9 0.95 0.001 0.5 2" condim="3"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)


def contact_summary():
    """접촉 법선력 합(N)과 압력중심 x(m, 상자 중심 기준)."""
    fn_sum, mx = 0.0, 0.0
    f6 = np.zeros(6)
    for i in range(data.ncon):
        mujoco.mj_contactForce(model, data, i, f6)
        fn = f6[0]
        # 접촉 프레임의 법선이 월드 z 근사(평면 위 상자). 접촉 위치의 x로 가중
        fn_sum += fn
        mx += fn * (data.contact[i].pos[0] - data.qpos[0])
    cop = mx / fn_sum if fn_sum > 1e-9 else float("nan")
    return fn_sum, cop


def reset(z, pitch_deg=0.0):
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = [0.0, 0.0, z]
    th = np.deg2rad(pitch_deg)
    data.qpos[3:7] = [np.cos(th / 2), 0.0, np.sin(th / 2), 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


# (1) 평형 높이
reset(0.03)
for _ in range(2000):
    mujoco.mj_step(model, data)
z_eq = float(data.qpos[2])
fn_eq, _ = contact_summary()
weight = MASS * 9.81
penetration_eq = HALF[2] - z_eq  # 상자 아래면이 바닥(z=0)보다 아래로 들어간 깊이(m)

res = {
    "model": {"mass_kg": MASS, "weight_N": weight, "box_half_m": HALF},
    "equilibrium": {
        "z_eq_m": z_eq,
        "penetration_mm": penetration_eq * 1000,
        "normal_force_N": fn_eq,
        "force_over_weight": fn_eq / weight,
    },
}

# (2) 높이 오프셋 스윕: 첫 mj_forward의 법선력 (논문 그림 3 왼쪽에 해당)
sweep = {}
for dz_mm in (-2, -1, 0, 1, 2):
    reset(z_eq + dz_mm * 1e-3)
    fn, _ = contact_summary()
    sweep[str(dz_mm)] = {"normal_force_N": fn, "change_pct": 100 * (fn / fn_eq - 1)}
res["height_offset_sweep"] = sweep
res["paper_fig3_height_read_pct"] = {"-2": 16, "-1": 10, "0": 0, "1": -28, "2": -50}  # 그림 판독 값

# (3) 피치 스윕: 평형 높이에서 피치만 줌. 압력중심 이동과 법선력 변화
pitch = {}
for deg in (-2, -1, 0, 1, 2):
    reset(z_eq, pitch_deg=deg)
    fn, cop = contact_summary()
    pitch[str(deg)] = {
        "normal_force_N": fn,
        "force_change_pct": 100 * (fn / fn_eq - 1),
        "cop_x_mm": cop * 1000,
        "cop_over_half_length_pct": 100 * cop / HALF[0],
    }
res["pitch_sweep"] = pitch
res["paper_fig3_moment_read_pct"] = {"-2": -46, "-1": -27, "0": 0, "1": 2, "2": 3}  # 그림 판독 값

# (4) 해석용 숫자: 평형 침투 깊이 대비 오프셋 비율
res["note"] = (
    "평형 침투 깊이 대비 2 mm는 비율 %.0f%%. 소프트 접촉은 침투 깊이에 거의 비례하므로 "
    "오프셋이 평형 침투 깊이에 가까울수록 힘이 크게 바뀐다." % (200.0 / (penetration_eq * 1000))
)

with open(OUT, "w") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
