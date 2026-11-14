"""Integrator and time-step stability of spring-driven tongs closing on a block in MuJoCo (CPU).

Two hinged jaws (mass 10 g each, arm 12 cm, damping 0.002 N m s/rad) are driven by torsional springs
toward a closed reference of -0.3 rad (jaw 2 mirrored) and close on a fixed block. The block sits
2 cm from the hinges. Gravity is off. Two contact settings are compared: MuJoCo default soft contact,
and a stiffer contact (solref 0.005 s time constant) on all tongs and block geoms.
Each run is 1 s long. For the last 0.2 s the script records:
  - whether the state stays finite,
  - the mean jaw angle and its peak-to-peak range (chatter),
  - the mean sum of contact normal force magnitudes on the block (counts each contact point),
  - the torque-balance residual of jaw 1: contact torque about the hinge plus the spring torque
    (zero in equilibrium),
  - the deepest penetration (most negative contact distance).
Errors are measured against a reference run with the same spring and contact setting:
implicitfast at dt = 1/4000 s. A run is flagged as tunnelled when the mean jaw angle is below the
angle at which the arm tip crosses the block mid-plane (about -0.197 rad); those states are not
valid grips, so errors against a tunnelled reference are not reported.

Two spring stiffnesses are used (0.5 and 2.0 N m/rad). Integrators: Euler, implicit, implicitfast,
RK4. Time steps: 1/200, 1/500, 1/1000, 1/2000 s.

Run: .venv/bin/python mana/analysis/extra_tongs_contact.py
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
OUT = ROOT / "analysis" / "extra_tongs_contact_results.json"

T_END = 1.0
TAIL = 0.2
K_SPRINGS = [0.5, 2.0]               # N m / rad
DAMPING = 0.002                      # N m s / rad
SPRINGREF = -0.3                     # rad (closed reference, beyond the block)
INTEGRATORS = ["Euler", "implicit", "implicitfast", "RK4"]
DTS = {"1/200": 1 / 200, "1/500": 1 / 500, "1/1000": 1 / 1000, "1/2000": 1 / 2000}
TUNNEL_ANGLE = -0.197                # rad, arm tip at the block mid-plane (x = 0.10 m, y = 0)
CONTACTS = {"default": "", "stiff_solref": 'solref="0.005 1"'}
REF = ("implicitfast", 1 / 4000)
REF_CHECK = ("RK4", 1 / 4000)

XML = """
<mujoco>
  <compiler angle="radian"/>
  <option timestep="%(dt)r" gravity="0 0 0" integrator="%(integ)s"/>
  <worldbody>
    <body name="jaw1" pos="0 0.02 0">
      <joint name="j1" type="hinge" axis="0 0 1" range="-0.6 0.6" limited="true"
             stiffness="%(k)r" springref="%(ref)r" damping="%(c)r"/>
      <geom name="g1" type="capsule" fromto="0 0 0 0.12 0 0" size="0.004" mass="0.01" %(geomattr)s/>
    </body>
    <body name="jaw2" pos="0 -0.02 0">
      <joint name="j2" type="hinge" axis="0 0 -1" range="-0.6 0.6" limited="true"
             stiffness="%(k)r" springref="%(ref)r" damping="%(c)r"/>
      <geom name="g2" type="capsule" fromto="0 0 0 0.12 0 0" size="0.004" mass="0.01" %(geomattr)s/>
    </body>
    <body name="block" pos="0.10 0 0">
      <geom name="blk" type="box" size="0.02 0.012 0.02" %(geomattr)s/>
    </body>
  </worldbody>
</mujoco>
"""


def build(k, integ, dt, contact):
    return mujoco.MjModel.from_xml_string(
        XML % {"dt": dt, "integ": integ, "k": k, "ref": SPRINGREF, "c": DAMPING,
               "geomattr": CONTACTS[contact]})


def run(k, integ, dt, contact):
    model = build(k, integ, dt, contact)
    data = mujoco.MjData(model)
    n = int(round(T_END / dt))
    tail = int(round(TAIL / dt))
    block = model.geom("blk").id
    q_tail, f_tail, pen_tail, res_tail = [], [], [], []
    f6 = np.zeros(6)
    jaw1 = model.geom("g1").id
    hinge1 = np.array([0.0, 0.02, 0.0])
    k_spring = float(model.jnt_stiffness[model.joint("j1").id])
    for i in range(n):
        mujoco.mj_step(model, data)
        q = float(np.mean(data.qpos[:2]))
        if not (math.isfinite(q) and abs(q) < 1e3 and np.all(np.isfinite(data.qpos))):
            return {"finite": False}
        if i >= n - tail:
            total_fn = 0.0
            deepest = 0.0
            tau_c = 0.0
            for ci in range(data.ncon):
                c = data.contact[ci]
                if block not in (c.geom1, c.geom2):
                    continue
                mujoco.mj_contactForce(model, data, ci, f6)
                total_fn += abs(float(f6[0]))
                deepest = min(deepest, float(c.dist))
                if jaw1 in (c.geom1, c.geom2):
                    f_world = c.frame.reshape(3, 3).T @ f6[:3]     # force on geom2
                    if c.geom1 == jaw1:
                        f_world = -f_world                          # force on jaw 1
                    tau_c += float(np.cross(c.pos - hinge1, f_world)[2])
            spring_tau = -k_spring * (float(data.qpos[model.joint("j1").qposadr[0]]) - SPRINGREF)
            q_tail.append(q)
            f_tail.append(total_fn)
            pen_tail.append(deepest)
            res_tail.append(abs(tau_c + spring_tau))
    q_tail = np.array(q_tail)
    q_mean = float(np.mean(q_tail))
    return {
        "finite": True,
        "tunnelled": bool(q_mean < TUNNEL_ANGLE),
        "q_mean_last_0p2s_rad": round(q_mean, 5),
        "chatter_p2p_last_0p2s_rad": round(float(np.ptp(q_tail)), 5),
        "contact_normal_sum_mean_N": round(float(np.mean(f_tail)), 4),
        "torque_balance_residual_mean_N_m": round(float(np.mean(res_tail)), 6),
        "spring_torque_N_m": round(-k_spring * (float(np.mean(q_tail)) - SPRINGREF), 5),
        "deepest_penetration_mm": round(-1000 * float(np.min(pen_tail)), 3),
    }


def main():
    t0 = time.time()
    refs = {}
    ref_check = {}
    for ct in CONTACTS:
        for k in K_SPRINGS:
            refs[(ct, k)] = run(k, REF[0], REF[1], ct)
            ref_check[(ct, k)] = run(k, REF_CHECK[0], REF_CHECK[1], ct)
    rows = []
    for ct in CONTACTS:
        for k in K_SPRINGS:
            ref = refs[(ct, k)]
            for dt_name, dt in DTS.items():
                for integ in INTEGRATORS:
                    r = run(k, integ, dt, ct)
                    entry = {"contact": ct, "k_N_m_per_rad": k, "dt": dt_name, "integrator": integ, **r}
                    if r["finite"] and not ref["tunnelled"]:
                        entry["dq_vs_ref_rad"] = round(abs(r["q_mean_last_0p2s_rad"] - ref["q_mean_last_0p2s_rad"]), 5)
                        if ref["contact_normal_sum_mean_N"] > 0:
                            entry["contact_sum_rel_err_vs_ref"] = round(
                                abs(r["contact_normal_sum_mean_N"] - ref["contact_normal_sum_mean_N"]) / ref["contact_normal_sum_mean_N"], 3)
                    rows.append(entry)
                    print(ct, k, dt_name, integ, {kk: v for kk, v in entry.items() if kk not in ("contact", "k_N_m_per_rad", "dt", "integrator")}, flush=True)

    res = {
        "script": "mana/analysis/extra_tongs_contact.py",
        "device": "cpu",
        "threads": 1,
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "params": {"T_end_s": T_END, "tail_s": TAIL, "damping_N_m_s_per_rad": DAMPING,
                   "springref_rad": SPRINGREF, "k_springs": K_SPRINGS, "dts": DTS,
                   "integrators": INTEGRATORS, "reference": f"{REF[0]} dt=1/4000",
                   "reference_check": f"{REF_CHECK[0]} dt=1/4000",
                   "jaw_mass_kg": 0.01, "arm_m": 0.12, "block_half_size_m": [0.02, 0.012, 0.02],
                   "contacts": CONTACTS, "tunnel_angle_rad": TUNNEL_ANGLE},
        "reference": {f"{ct}_k{k}": refs[(ct, k)] for ct in CONTACTS for k in K_SPRINGS},
        "reference_check_rk4": {f"{ct}_k{k}": ref_check[(ct, k)] for ct in CONTACTS for k in K_SPRINGS},
        "rows": rows,
        "wall_time_s": None,
    }
    res["wall_time_s"] = round(time.time() - t0, 2)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wall", res["wall_time_s"], "s")


if __name__ == "__main__":
    main()
