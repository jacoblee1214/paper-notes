"""Integrator stability of a spring-driven tool joint in MuJoCo (CPU, single thread, seconds).

A prismatic joint with mass m, stiffness k = F / s, and viscous damping c = 2 zeta sqrt(k m)
is displaced by s and released. The model is integrated with MuJoCo's Euler, implicit,
implicitfast and RK4 integrators at several time steps. For each case the file records
the peak displacement in the last 0.2 s relative to the initial displacement and whether
the trajectory stays finite. The same parameter grid as the 1-DOF spectral-radius table
is used, so the two can be compared directly.

Run: .venv/bin/python analysis/extra_joint_integrator.py
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
OUT = ROOT / "analysis" / "extra_joint_integrator_results.json"

T_END = 1.0
INTEGRATORS = ["Euler", "implicit", "implicitfast", "RK4"]
DTS = {"1/200": 1 / 200, "1/500": 1 / 500, "1/1000": 1 / 1000}
CASES = [  # (F [N], s [m], m [kg])
    (3.0, 0.01, 0.005),
    (3.0, 0.02, 0.050),
    (7.0, 0.01, 0.005),
    (7.0, 0.01, 0.050),
]
ZETAS = [0.0, 0.2]


def xml(k, c, m, integrator, dt):
    return f"""
<mujoco>
  <option timestep="{dt}" gravity="0 0 0" integrator="{integrator}"/>
  <worldbody>
    <body name="tool" pos="0 0 0">
      <joint name="slide" type="slide" axis="1 0 0" stiffness="{k}" springref="0"
             damping="{c}" limited="false"/>
      <geom type="sphere" size="0.005" mass="{m}"/>
    </body>
  </worldbody>
</mujoco>
"""


def run(F, s, m, zeta, integrator, dt):
    k = F / s
    c = 2 * zeta * math.sqrt(k * m)
    model = mujoco.MjModel.from_xml_string(xml(k, c, m, integrator, dt))
    data = mujoco.MjData(model)
    data.qpos[0] = s
    n = int(round(T_END / dt))
    tail = int(round(0.2 / dt))
    peak = 0.0
    finite = True
    for i in range(n):
        mujoco.mj_step(model, data)
        x = float(data.qpos[0])
        if not math.isfinite(x) or abs(x) > 1e6:
            finite = False
            break
        if i >= n - tail:
            peak = max(peak, abs(x))
    if not finite:
        return {"peak_ratio_last_0p2s": None, "finite": False}
    return {"peak_ratio_last_0p2s": float(f"{peak / s:.4g}"), "finite": True}


def main():
    t0 = time.time()
    rows = []
    for (F, s, m) in CASES:
        k = F / s
        om = math.sqrt(k / m)
        for zeta in ZETAS:
            for dt_name, dt in DTS.items():
                entry = {
                    "F_N": F, "s_m": s, "m_kg": m, "k_N_per_m": round(k, 1),
                    "omega_rad_s": round(om, 1), "omega_dt": round(om * dt, 3),
                    "zeta": zeta, "dt": dt_name,
                }
                for integ in INTEGRATORS:
                    entry[integ] = run(F, s, m, zeta, integ, dt)
                rows.append(entry)
                print(F, s, m, zeta, dt_name, {i: entry[i]["peak_ratio_last_0p2s"] for i in INTEGRATORS})

    # Stability at the reference step dt = 1/200 s and zeta = 0.2 (the case set of the 1-DOF table)
    ref = [r for r in rows if r["dt"] == "1/200" and r["zeta"] == 0.2]
    summary = {i: {"diverged_or_growing_cases": sum(1 for r in ref if (not r[i]["finite"]) or r[i]["peak_ratio_last_0p2s"] > 1.0),
                   "of": len(ref)} for i in INTEGRATORS}

    res = {
        "script": "analysis/extra_joint_integrator.py",
        "device": "cpu",
        "threads": 1,
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "params": {"T_end_s": T_END, "tail_window_s": 0.2, "integrators": INTEGRATORS,
                   "dts": DTS, "cases_F_s_m": CASES, "zetas": ZETAS,
                   "growth_criterion": "peak |x| in last 0.2 s divided by s; >1 counts as growing"},
        "rows": rows,
        "summary_at_dt_1_200_zeta_0p2": summary,
        "wall_time_s": None,
    }
    res["wall_time_s"] = round(time.time() - t0, 2)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("summary", summary)
    print("wall", res["wall_time_s"], "s")


if __name__ == "__main__":
    main()
