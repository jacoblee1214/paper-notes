#!/usr/bin/env python3
"""이 맥(CPU 한 프로세스)에서 G1 MJCF의 MuJoCo 스텝 속도를 재어 UAN 재생 학습의 CPU 비용을 어림한다.

입력: 공식 저장소의 G1 MJCF (src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml). 공식 코드는 실행하지 않고
      MJCF 파일만 mujoco로 읽는다. repo/ 는 https://github.com/nolie-rolie/contact-uan 의 커밋
      55de136a33f44b7f22190c5f9de882266268b5ab 이다:
      git clone --depth 1 https://github.com/nolie-rolie/contact-uan repo && git -C repo checkout 55de136a33f44b7f22190c5f9de882266268b5ab
측정: 바닥을 더하고 5 ms 스텝 수만 잰다. MJCF에는 구동기가 없어(nu = 0) 로봇이 힘없이 무너지는 상태이고
      접촉 수(평균 약 19개)는 실제 보행과 다르다. 정확한 값이 아니라 자릿수 확인용이다.
실행: python3 analysis/bench_g1_cpu.py   (mujoco 필요, 약 10초) → bench_g1_cpu_results.json
"""
import json
import os
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "..", "repo", "src", "mjlab", "asset_zoo", "robots", "unitree_g1", "xmls", "g1.xml")
OUT = os.path.join(HERE, "bench_g1_cpu_results.json")

spec = mujoco.MjSpec.from_file(XML)
spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1], friction=[1, 0.005, 0.0001])
spec.option.timestep = 0.005
model = spec.compile()
data = mujoco.MjData(model)
print("nq", model.nq, "nv", model.nv, "nu", model.nu, "ngeom", model.ngeom)

# 서 있는 높이(0.8 m)에서 시작, 관절은 0 자세. 구동기가 없으므로 무너진다
data.qpos[2] = 0.8
mujoco.mj_forward(model, data)

n_warm, n_meas = 200, 4000
for _ in range(n_warm):
    mujoco.mj_step(model, data)
t0 = time.perf_counter()
ncon = 0
for _ in range(n_meas):
    mujoco.mj_step(model, data)
    ncon += data.ncon
dt = time.perf_counter() - t0
steps_per_s = n_meas / dt

# UAN 재생 PPO 한 반복: 환경 4096 x 24 정책 스텝 x 감속비 4 물리 스텝
phys_per_iter = 4096 * 24 * 4
res = {
    "model": {"nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu), "ngeom": int(model.ngeom)},
    "mujoco_version": mujoco.__version__,
    "physics_steps_per_s_one_core": steps_per_s,
    "mean_contacts": ncon / n_meas,
    "physics_steps_per_ppo_iteration": phys_per_iter,
    "seconds_per_iteration_one_core": phys_per_iter / steps_per_s,
    "hours_for_2000_iterations_one_core": 2000 * phys_per_iter / steps_per_s / 3600,
    "hours_for_2000_iterations_10_cores_ideal": 2000 * phys_per_iter / steps_per_s / 3600 / 10,
    "note": "물리 스텝만 센 값이다. 신경망 추론, PPO 갱신, 재설정 비용은 포함하지 않았고, 구동기 없이 무너지는 상태의 접촉 수라서 자릿수 확인용이다.",
}
with open(OUT, "w") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
