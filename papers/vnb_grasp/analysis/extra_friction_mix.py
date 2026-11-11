"""MuJoCo contact friction combination check (CPU, single thread, a few seconds).

A flat slab rests on a horizontal plane. Gravity is tilted by an angle theta, so the
slab slides when tan(theta) exceeds the effective contact friction. Bisection on theta
gives the effective friction for each pair of geom friction coefficients. The result is
compared with the candidate combination rules (max, geometric mean, min, arithmetic mean).
The contact friction value stored by MuJoCo is read directly as a cross-check.

Run: .venv/bin/python analysis/extra_friction_mix.py
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
OUT = ROOT / "analysis" / "extra_friction_mix_results.json"

G = 9.81
T_SIM = 1.0          # s of sliding time per trial
SPEED_TH = 0.05      # m/s at the end of the window; slip means a steady drift, creep stays below ~0.01
DT = 0.001           # s


def model_xml(mu_floor, mu_slab):
    return f"""
<mujoco>
  <option timestep="{DT}" gravity="0 0 -{G}"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" friction="{mu_floor} 0.0 0.0"/>
    <body name="slab" pos="0 0 0.0099">
      <freejoint name="slab_free"/>
      <geom name="slab_g" type="box" size="0.05 0.05 0.01" mass="0.1"
            friction="{mu_slab} 0.0 0.0"/>
    </body>
  </worldbody>
</mujoco>
"""


def stored_friction(mu_floor, mu_slab):
    model = mujoco.MjModel.from_xml_string(model_xml(mu_floor, mu_slab))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon >= 1, "no contact at start"
    return float(data.contact[0].friction[0])


def slides(mu_floor, mu_slab, theta):
    model = mujoco.MjModel.from_xml_string(model_xml(mu_floor, mu_slab))
    data = mujoco.MjData(model)
    model.opt.gravity[:] = [G * math.sin(theta), 0.0, -G * math.cos(theta)]
    for _ in range(int(round(T_SIM / DT))):
        mujoco.mj_step(model, data)
    speed = float(np.linalg.norm(data.qvel[0:2]))
    tipped = abs(float(data.qpos[2]) - 0.0099) > 2e-2
    return speed > SPEED_TH and not tipped


def effective_mu(mu_floor, mu_slab):
    """Bisection on the tilt angle: returns tan(theta) at the slip threshold."""
    lo, hi = 0.0, math.atan(1.2)
    assert not slides(mu_floor, mu_slab, lo)
    assert slides(mu_floor, mu_slab, hi)
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if slides(mu_floor, mu_slab, mid):
            hi = mid
        else:
            lo = mid
    return math.tan(0.5 * (lo + hi))


RULES = {
    "max": lambda a, b: max(a, b),
    "geometric_mean": lambda a, b: math.sqrt(a * b),
    "min": lambda a, b: min(a, b),
    "arithmetic_mean": lambda a, b: 0.5 * (a + b),
}


def main():
    t0 = time.time()
    pairs = [(1.0, 0.4), (0.4, 1.0), (1.0, 0.9), (0.9, 1.0), (0.4, 0.4), (0.9, 0.9), (1.0, 1.0), (0.4, 0.9)]
    rows = []
    for f_floor, f_slab in pairs:
        mu_eff = effective_mu(f_floor, f_slab)
        stored = stored_friction(f_floor, f_slab)
        rules = {name: round(fn(f_floor, f_slab), 4) for name, fn in RULES.items()}
        match = min(RULES, key=lambda n: abs(rules[n] - mu_eff))
        rows.append({
            "geom_friction_floor": f_floor,
            "geom_friction_slab": f_slab,
            "effective_mu_sliding": round(mu_eff, 4),
            "contact_friction_stored_by_mujoco": round(stored, 4),
            "closest_rule": match,
            "rule_values": rules,
        })
        print(f"floor {f_floor} slab {f_slab}: eff {mu_eff:.4f}, stored {stored:.4f}, closest {match}")

    # Direct test of the case in question: mu_f = 1.0 with mu_o 0.4 vs 0.9 (slab = 1.0)
    by = {(r["geom_friction_floor"], r["geom_friction_slab"]): r for r in rows}
    a = by[(0.4, 1.0)]["effective_mu_sliding"]
    b = by[(0.9, 1.0)]["effective_mu_sliding"]
    res = {
        "script": "analysis/extra_friction_mix.py",
        "device": "cpu",
        "threads": 1,
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "params": {"dt_s": DT, "sim_time_s": T_SIM, "speed_threshold_m_s": SPEED_TH,
                   "bisection_iters": 22, "gravity": G},
        "pairs": rows,
        "case_mu_f_1p0": {
            "mu_o_0p4_effective": a,
            "mu_o_0p9_effective": b,
            "difference": round(b - a, 4),
            "note": "geom friction of the slab is 1.0 in both cases; the floor geom changes",
        },
        "wall_time_s": None,
    }
    res["wall_time_s"] = round(time.time() - t0, 2)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wall", res["wall_time_s"], "s")


if __name__ == "__main__":
    main()
