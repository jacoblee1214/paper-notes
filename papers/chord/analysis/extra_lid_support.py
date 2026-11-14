"""Lid opening with a learned 2-link planar finger in MuJoCo: position reward vs a friction-cone
support-function reward computed from the live contacts (CPU, 1 thread, a few minutes).

Same box, hinged lid, finger, waypoint policy (four Cartesian waypoint coordinates, elbow-up IK)
and cross-entropy search as the earlier lid-policy check. The only change is the reward:

  position: exp(-d^2 / (2 sigma^2)), d = second waypoint to the lip contact point.
  support:  for each finger-lid contact, the friction cone (per-contact mu from MuJoCo) is
            discretised into 8 edge wrenches [f ; (p - hinge) x f] with unit normal force, plus the
            origin. The support in the lid-opening torque direction is max(0, max_j b.w_j), summed
            over contacts (Minkowski sum of the per-contact sets). Two variants are scored:
            support_end, the value at the end of the motion (same instant as the position reward),
            and support_max, the maximum over the push segment. The torque is divided by the box
            boundary radius r_c.

Success: the lid angle exceeds 0.3 rad at the end of the 1.0 s motion.

Run: .venv/bin/python chord/analysis/extra_lid_support.py
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
OUT = ROOT / "analysis" / "extra_lid_support_results.json"

SIM_DT = 0.002
T_SEG = 0.5
N_STEPS = int(round(2 * T_SEG / SIM_DT))
MU = 0.5                                      # per-geom friction (MuJoCo sliding term)
N_EDGES = 8
R_C = 0.10                                    # m, box half-length used as boundary radius
LIP_TARGET = np.array([0.108, 0.025])        # fingertip contact on the lip front face (x, z)
SIGMA = 0.01
OPEN_SUCCESS = 0.3
POP, ELITE, ITERS = 40, 8, 20
SEEDS = [0, 1, 2, 3, 4]
TIP_START = np.array([0.15, 0.0])
BASE_XZ = np.array([0.2, -0.02])
L_LINK = 0.07
LO = np.array([0.09, -0.03, 0.09, -0.03])
HI = np.array([0.17, 0.08, 0.17, 0.08])

XML = """
<mujoco>
  <compiler angle="radian"/>
  <option timestep="%g" gravity="0 0 -9.81"/>
  <default>
    <geom friction="%g 0.005 0.0001"/>
  </default>
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
""" % (SIM_DT, MU)

MODEL = mujoco.MjModel.from_xml_string(XML)
TIP_GEOM = MODEL.geom("f_tip").id
LID_GEOMS = {MODEL.geom("lid_plate_g").id, MODEL.geom("lid_lip_g").id}
HINGE_POS = np.array([-0.1, 0.0, 0.05])
HINGE_AXIS = np.array([0.0, 1.0, 0.0])
QADR = [MODEL.joint("f_j1").qposadr[0], MODEL.joint("f_j2").qposadr[0]]
LID_ADR = MODEL.joint("hinge").qposadr[0]
CONE_DIRS = [(math.cos(2 * math.pi * k / N_EDGES), math.sin(2 * math.pi * k / N_EDGES))
             for k in range(N_EDGES)]


def ik_elbow_up(target):
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


def support_open(data):
    """Sum over finger-lid contacts of max(0, max_j b.w_j), b = lid-opening torque about the hinge.
    Each contact's set is conv({0} + 8 edge wrenches) with unit normal force, pushing into the lid."""
    total = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if g1 == TIP_GEOM and g2 in LID_GEOMS:
            sgn = 1.0                                  # contact normal (geom1 -> geom2) points into lid
        elif g2 == TIP_GEOM and g1 in LID_GEOMS:
            sgn = -1.0
        else:
            continue
        rot = c.frame.reshape(3, 3)                    # rows: normal, tangent 1, tangent 2
        n_push = sgn * rot[0]
        t1, t2 = rot[1], rot[2]
        mu = float(c.friction[0])
        arm = c.pos - HINGE_POS
        best = 0.0
        for ct, st in CONE_DIRS:
            g = n_push + mu * (ct * t1 + st * t2)     # force on lid for one edge (unit normal force)
            open_torque = -float(np.cross(arm, g) @ HINGE_AXIS) / R_C
            best = max(best, open_torque)
        total += best
    return total


def rollout(params):
    """params = [xg, zg, xp, zp]. Straight segments start -> g (0.5 s) -> p (0.5 s).
    Returns the lid opening angle (rad, positive = open), the support at the end of the motion and
    the maximum support over the push segment."""
    data = mujoco.MjData(MODEL)
    q0 = ik_elbow_up(TIP_START)
    data.qpos[QADR[0]], data.qpos[QADR[1]] = q0
    mujoco.mj_forward(MODEL, data)
    wg, wp = np.array(params[:2], float), np.array(params[2:], float)
    sup_max, sup_end = 0.0, 0.0
    for k in range(N_STEPS):
        t = k * SIM_DT
        if t < T_SEG:
            tip_t = TIP_START + (wg - TIP_START) * (t / T_SEG)
        else:
            tip_t = wg + (wp - wg) * min(1.0, (t - T_SEG) / T_SEG)
        data.ctrl[:] = ik_elbow_up(tip_t)
        mujoco.mj_step(MODEL, data)
        if t >= T_SEG:
            sup_max = max(sup_max, support_open(data))
    sup_end = support_open(data)
    tip = fk_tip(np.array([data.qpos[QADR[0]], data.qpos[QADR[1]]]))
    angle = -float(data.qpos[LID_ADR])
    return tip, angle, sup_end, sup_max


def reward_of(params, kind):
    tip, angle, sup_end, sup_max = rollout(params)
    if kind == "position":
        d2 = float(np.sum((np.asarray(params[2:]) - LIP_TARGET) ** 2))
        return float(np.exp(-d2 / (2 * SIGMA ** 2))), angle
    if kind == "support_end":
        return sup_end, angle
    return sup_max, angle


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
    tip, _, _, _ = rollout(best_p)
    return {
        "seed": seed,
        "best_reward_final_eval": round(r, 4),
        "lid_angle_rad_at_end": round(angle, 3),
        "success": bool(angle > OPEN_SUCCESS),
        "tip_distance_to_lip_m": round(float(np.linalg.norm(tip - LIP_TARGET)), 4),
    }


def main():
    t0 = time.time()
    _, closed_angle, _, _ = rollout(np.array([0.15, 0.0, 0.15, 0.0]))
    out = {
        "script": "chord/analysis/extra_lid_support.py",
        "device": "cpu",
        "threads": 1,
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "params": {"sim_dt_s": SIM_DT, "segment_s": T_SEG, "geom_friction_mu": MU,
                   "cone_edges": N_EDGES, "boundary_radius_m": R_C,
                   "lip_target_xz_m": LIP_TARGET.tolist(), "sigma_m": SIGMA,
                   "open_success_rad": OPEN_SUCCESS,
                   "cem": {"pop": POP, "elite": ELITE, "iters": ITERS, "init_std": 0.03,
                           "waypoint_bounds_xz": [LO.tolist(), HI.tolist()]},
                   "seeds": SEEDS},
        "reference_nonpushing_lid_angle_rad": round(closed_angle, 4),
        "runs": {},
    }
    for kind in ("position", "support_end", "support_max"):
        runs = []
        for seed in SEEDS:
            r = cem(kind, seed)
            runs.append(r)
            print(kind, r, flush=True)
        out["runs"][kind] = {"per_seed": runs,
                             "success_rate": round(float(np.mean([r["success"] for r in runs])), 3),
                             "success_count": int(sum(r["success"] for r in runs)),
                             "n_seeds": len(runs),
                             "median_lid_angle_rad": round(float(np.median(
                                 [r["lid_angle_rad_at_end"] for r in runs])), 3)}
    out["wall_time_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wall", out["wall_time_s"])


if __name__ == "__main__":
    main()
