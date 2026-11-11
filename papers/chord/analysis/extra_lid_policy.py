"""Lid opening with a learned 2-link planar finger in MuJoCo: position reward vs wrench reward
(CPU, 1 thread, a few minutes).

A box with a hinged lid (front lip) is opened by a 2-link planar finger. The fingertip moves in
two straight Cartesian segments through two waypoints (elbow-up IK); the four waypoint
coordinates are found with the cross-entropy method (CEM). Two rewards are compared, each with
three CEM seeds:
  position: exp(-d^2 / (2 sigma^2)), d = fingertip to the lip contact point at the end of the motion.
  wrench:   the mean opening torque about the hinge (N m) produced by the finger contacts during
            the push segment, with no clipping.
Success: the lid angle exceeds 0.3 rad at the end of the motion.

Run: .venv/bin/python chord/analysis/extra_lid_policy.py
"""
import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import json
import math
import platform
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "extra_lid_policy_results.json"

SIM_DT = 0.002
T_SEG = 0.5                                   # s per segment (two segments)
N_STEPS = int(round(2 * T_SEG / SIM_DT))
LIP_TARGET = np.array([0.108, 0.025])        # fingertip contact on the lip front face (x, z)
SIGMA = 0.01                                  # m
OPEN_SUCCESS = 0.3                            # rad
POP, ELITE, ITERS = 40, 8, 20
SEEDS = [0, 1, 2]
TIP_START = np.array([0.15, 0.0])            # (x, z): in front of the lid, below the lip line
BASE_XZ = np.array([0.2, -0.02])
L_LINK = 0.07
LO = np.array([0.09, -0.03, 0.09, -0.03])    # waypoint bounds (x, z, x, z)
HI = np.array([0.17, 0.08, 0.17, 0.08])

XML = """
<mujoco>
  <compiler angle="radian"/>
  <option timestep="%g" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="base" type="box" size="0.09 0.075 0.025" pos="0 0 0.025"/>
    <body name="lid" pos="-0.1 0 0.05">
      <joint name="hinge" type="hinge" axis="0 1 0" range="-1.8 0.05" damping="0.02" armature="0.001"/>
      <geom name="lid_plate_g" type="box" size="0.1 0.075 0.005" pos="0.1 0 0.005" mass="0.05"/>
      <geom name="lid_lip_g" type="box" size="0.005 0.075 0.025" pos="0.195 0 -0.025" mass="0.01"/>
    </body>
    <body name="finger_base" pos="0.2 0 -0.02">
      <body name="f_l1">
        <joint name="f_j1" type="hinge" axis="0 1 0" range="-1.5 1.5" armature="0.01" damping="0.5"/>
        <geom name="f_link1" type="capsule" fromto="0 0 0 -0.07 0 0" size="0.006" mass="0.02"/>
        <body name="f_l2" pos="-0.07 0 0">
          <joint name="f_j2" type="hinge" axis="0 1 0" range="-1.5 1.5" armature="0.01" damping="0.5"/>
          <geom name="f_link2" type="capsule" fromto="0 0 0 -0.07 0 0" size="0.006" mass="0.02"/>
          <geom name="f_tip" type="sphere" pos="-0.07 0 0" size="0.008" mass="0.005"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="f_j1" kp="50" ctrlrange="-1.5 1.5"/>
    <position name="a2" joint="f_j2" kp="50" ctrlrange="-1.5 1.5"/>
  </actuator>
</mujoco>
""" % SIM_DT

MODEL = mujoco.MjModel.from_xml_string(XML)
TIP_GEOM = MODEL.geom("f_tip").id
LID_GEOMS = {MODEL.geom("lid_plate_g").id, MODEL.geom("lid_lip_g").id}
HINGE_POS = np.array([-0.1, 0.0, 0.05])
HINGE_AXIS = np.array([0.0, 1.0, 0.0])
QADR = [MODEL.joint("f_j1").qposadr[0], MODEL.joint("f_j2").qposadr[0]]
LID_ADR = MODEL.joint("hinge").qposadr[0]


def ik_elbow_up(target):
    """Joint angles (q1, q2) that put the fingertip at target (x, z), elbow up (equal links)."""
    v = np.asarray(target, float) - BASE_XZ
    dist = float(np.clip(np.linalg.norm(v), 0.01, 2 * L_LINK - 1e-3))
    beta = math.atan2(v[1], v[0])
    gamma = math.acos(dist / (2 * L_LINK))
    psi1 = beta - gamma
    return np.array([math.pi - psi1, -2 * gamma])


def fk_tip(q):
    psi1 = math.pi - q[0]
    psi2 = math.pi - (q[0] + q[1])
    return BASE_XZ + L_LINK * np.array([math.cos(psi1), math.sin(psi1)]) \
        + L_LINK * np.array([math.cos(psi2), math.sin(psi2)])


def contact_open_torque(data):
    """Opening torque (N m, positive = opening) about the hinge from finger-lid contacts."""
    total = 0.0
    f6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if not ((g1 == TIP_GEOM and g2 in LID_GEOMS) or (g2 == TIP_GEOM and g1 in LID_GEOMS)):
            continue
        mujoco.mj_contactForce(MODEL, data, i, f6)
        f_world = c.frame.reshape(3, 3).T @ f6[:3]     # force on geom2
        if g2 == TIP_GEOM:
            f_world = -f_world                          # force on the lid is the reaction
        tau = np.cross(c.pos - HINGE_POS, f_world)
        total += -float(tau @ HINGE_AXIS)               # hinge angle decreases when the lid opens
    return total


def rollout(params):
    """params = [xg, zg, xp, zp]. Straight segments: start -> waypoint g (0.5 s) -> waypoint p (0.5 s).
    Returns the fingertip position, the lid opening angle (rad, positive = open) and the mean
    opening torque over the push segment (N m)."""
    data = mujoco.MjData(MODEL)
    q0 = ik_elbow_up(TIP_START)
    data.qpos[QADR[0]], data.qpos[QADR[1]] = q0
    mujoco.mj_forward(MODEL, data)
    wg, wp = np.array(params[:2], float), np.array(params[2:], float)
    tau_push = []
    for k in range(N_STEPS):
        t = k * SIM_DT
        if t < T_SEG:
            tip_t = TIP_START + (wg - TIP_START) * (t / T_SEG)
        else:
            tip_t = wg + (wp - wg) * min(1.0, (t - T_SEG) / T_SEG)
        data.ctrl[:] = ik_elbow_up(tip_t)
        mujoco.mj_step(MODEL, data)
        if t >= T_SEG:
            tau_push.append(contact_open_torque(data))
    tip = fk_tip(np.array([data.qpos[QADR[0]], data.qpos[QADR[1]]]))
    angle = -float(data.qpos[LID_ADR])          # positive = opened
    return tip, angle, float(np.mean(tau_push))


def reward_of(params, kind):
    tip, angle, tau_push = rollout(params)
    if kind == "position":
        d2 = float(np.sum((np.asarray(params[2:]) - LIP_TARGET) ** 2))   # fingertip at end of motion
        return float(np.exp(-d2 / (2 * SIGMA ** 2))), angle
    return float(tau_push), angle


def cem(kind, seed):
    rng = np.random.default_rng(seed)
    mean = np.array([0.14, 0.0, 0.12, 0.03])
    std = np.full(4, 0.03)
    best_r, best_p = -1.0, None
    for _ in range(ITERS):
        samples = np.clip(rng.normal(mean, std, size=(POP, 4)), LO, HI)
        scores = np.array([reward_of(s, kind)[0] for s in samples])
        order = np.argsort(-scores)
        if scores[order[0]] > best_r:
            best_r, best_p = float(scores[order[0]]), samples[order[0]].copy()
        elite = samples[order[:ELITE]]
        mean = elite.mean(0)
        std = elite.std(0) + 0.005
    r, angle = reward_of(best_p, kind)
    tip, _, _ = rollout(best_p)
    return {
        "seed": seed,
        "best_reward_final_eval": round(r, 4),
        "best_waypoints_xz": [round(float(x), 4) for x in best_p],
        "lid_angle_rad_at_end": round(angle, 3),
        "success": bool(angle > OPEN_SUCCESS),
        "tip_xz_at_end": [round(float(tip[0]), 4), round(float(tip[1]), 4)],
        "tip_distance_to_lip_m": round(float(np.linalg.norm(tip - LIP_TARGET)), 4),
    }


def main():
    t0 = time.time()
    _, closed_angle, _ = rollout(np.array([0.15, 0.0, 0.15, 0.0]))
    out = {
        "script": "chord/analysis/extra_lid_policy.py",
        "device": "cpu",
        "threads": 1,
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "params": {"sim_dt_s": SIM_DT, "segment_s": T_SEG, "lip_target_xz_m": LIP_TARGET.tolist(),
                   "sigma_m": SIGMA, "open_success_rad": OPEN_SUCCESS,
                   "cem": {"pop": POP, "elite": ELITE, "iters": ITERS, "init_std": 0.03,
                           "waypoint_bounds_xz": [LO.tolist(), HI.tolist()]},
                   "seeds": SEEDS},
        "reference_nonpushing_lid_angle_rad": round(closed_angle, 4),
        "runs": {},
    }
    for kind in ("position", "wrench"):
        runs = []
        for seed in SEEDS:
            r = cem(kind, seed)
            runs.append(r)
            print(kind, r, flush=True)
        out["runs"][kind] = {"per_seed": runs,
                             "success_rate": round(float(np.mean([r["success"] for r in runs])), 3)}
    out["wall_time_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wall", out["wall_time_s"])


if __name__ == "__main__":
    main()
