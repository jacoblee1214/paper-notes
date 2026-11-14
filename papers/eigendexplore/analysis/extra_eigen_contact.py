"""CEM object-push task in MuJoCo with EigenDEXplore-style exploration covariances (CPU).

Hand: public Allegro right hand (MuJoCo Menagerie, wonik_allegro), 16 position actuators, base
fixed. Object: a free sphere (radius 1.5 cm, mass 0.5 kg) placed in front of the fingertips.
Gravity is off in the scene (no table) and the hand is not moved, so the fingers must push the
sphere. Each seed draws a goal 3 cm from the sphere start in a random direction.

Samplers compared per goal (paired seeds, same covariances as the posture-tracking check):
  iid           Sigma = I
  eigen_total   Sigma = 0.5 I + Sigma_e  (PCA-shaped eigen part, same trace as iid)
  eigen_2x      2 x eigen_total          (twice the total exploration)
  random_dir    0.5 I + Sigma_r          (same eigen spectrum, random orthonormal directions)
Basis: public Allegro PCA asset (safetensors, read with the standard library).
Optimizer: cross-entropy method over 4 control steps x 16 joints, 10 iterations, 32 samples,
3 elites, two step scales (normalized joint units). Cost: squared distance of the sphere center
to the goal, divided by the goal distance squared (1.0 = no motion). Success: final error < 1 cm.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import json
import platform
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import scipy
import yaml

T0 = time.perf_counter()
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MJCF = ROOT / "data" / "menagerie" / "wonik_allegro" / "right_hand.xml"
PCA = ROOT / "repo" / "src" / "action_bench" / "assets" / "allegro" / "pca.safetensors"
MANIFEST = ROOT / "repo" / "src" / "action_bench" / "assets" / "allegro" / "manifest.yaml"
OUT = HERE / "extra_eigen_contact_results.json"

SEEDS = 40
H = 4
T_STEP = 0.1
N_POP = 32
N_ELITE = 3
ITERS = 10
TRACE_TARGET = 0.5
SCALES = [0.25, 0.10]
K_VAR = 0.90
SAMPLERS = ["iid", "eigen_total", "eigen_2x", "random_dir"]
OBJ_START = np.array([0.085, 0.0, 0.075])     # m, hand (palm) frame, sphere center
OBJ_RADIUS = 0.015
OBJ_MASS = 0.5
GOAL_RADIUS = 0.03
SUCCESS_ERR = 0.01

JOINTS = ["ffj0", "ffj1", "ffj2", "ffj3", "mfj0", "mfj1", "mfj2", "mfj3",
          "rfj0", "rfj1", "rfj2", "rfj3", "thj0", "thj1", "thj2", "thj3"]


def load_safetensors(path):
    b = path.read_bytes()
    n = int(np.frombuffer(b[:8], dtype="<u8")[0])
    head = json.loads(b[8:8 + n])
    out = {}
    for k, v in head.items():
        if k == "__metadata__":
            continue
        dt = {"F32": np.float32, "F64": np.float64}[v["dtype"]]
        s, e = v["data_offsets"]
        out[k] = np.frombuffer(b[8 + n + s:8 + n + e], dtype=dt).reshape(v["shape"]).astype(np.float64)
    return out


def scale_trace(S, target_mean_diag):
    d = S.shape[0]
    return S * (target_mean_diag * d / np.trace(S))


def build_model():
    """Hand XML with an absolute mesh directory, gravity off, and a free sphere added to the world."""
    xml = MJCF.read_text()
    xml = xml.replace('meshdir="assets"', 'meshdir="%s"' % (MJCF.parent / "assets"))
    xml = xml.replace('<option cone="elliptic" impratio="10"/>',
                      '<option cone="elliptic" impratio="10" gravity="0 0 0"/>')
    body = ('<body name="obj" pos="%g %g %g"><freejoint/>'
            '<geom name="ball" type="sphere" size="%g" mass="%g" friction="1 0.005 0.0001"/></body>'
            % (*OBJ_START, OBJ_RADIUS, OBJ_MASS))
    xml = xml.replace("</worldbody>", body + "</worldbody>")
    return mujoco.MjModel.from_xml_string(xml)


def main():
    rng_master = np.random.default_rng(0)
    t = load_safetensors(PCA)
    man = yaml.safe_load(MANIFEST.read_text())
    lo, hi = np.array(man["joint_lower"], float), np.array(man["joint_upper"], float)
    c, half = (lo + hi) / 2, (hi - lo) / 2
    d = len(lo)
    ev = t["explained_variance"]
    k = int(np.searchsorted(np.cumsum(ev) / ev.sum(), K_VAR) + 1)
    B = t["components"][:k] / half
    sig = t["component_std"][:k]
    mean_n = (t["mean"] - c) / half

    Sig_e = scale_trace((B * (sig ** 2)[:, None]).T @ B, TRACE_TARGET)
    Qr = np.linalg.qr(rng_master.standard_normal((d, k)))[0].T
    Sig_r = scale_trace((Qr * (sig ** 2)[:, None]).T @ Qr, TRACE_TARGET)
    covs = {
        "iid": np.eye(d),
        "eigen_total": TRACE_TARGET * np.eye(d) + Sig_e,
        "eigen_2x": 2 * (TRACE_TARGET * np.eye(d) + Sig_e),
        "random_dir": TRACE_TARGET * np.eye(d) + Sig_r,
    }
    chol = {name: np.linalg.cholesky(S + 1e-9 * np.eye(d)) for name, S in covs.items()}
    trace_per_dim = {name: round(float(np.trace(S) / d), 4) for name, S in covs.items()}

    model = build_model()
    data = mujoco.MjData(model)
    jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS]
    qadr = np.array([model.jnt_qposadr[j] for j in jid])
    act_for_joint = np.array([[i for i in range(model.nu) if model.actuator_trnid[i, 0] == j][0]
                              for j in jid])
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj")
    nsub = int(round(T_STEP / model.opt.timestep))
    q0 = np.zeros(d)

    def run_episode(seq, goal):
        mujoco.mj_resetData(model, data)
        data.qpos[qadr] = c + q0 * half
        mujoco.mj_forward(model, data)
        for h in range(seq.shape[0]):
            data.ctrl[act_for_joint] = c + seq[h] * half
            for _ in range(nsub):
                mujoco.mj_step(model, data)
        return float(np.sum((data.xpos[obj_body] - goal) ** 2))

    def make_goal(rng):
        v = rng.standard_normal(3)
        return OBJ_START + GOAL_RADIUS * v / np.linalg.norm(v)

    def cost(seq, goal):
        return run_episode(seq, goal) / GOAL_RADIUS ** 2

    def run_cem(goal, sampler, scale, rng):
        Lc = chol[sampler]
        mu = np.tile(q0, (H, 1))
        best = np.inf
        for _ in range(ITERS):
            eps = rng.standard_normal((N_POP, H, d)) @ Lc.T
            U = np.clip(mu[None] + scale * eps, -0.95, 0.95)
            cs = np.array([cost(U[i], goal) for i in range(N_POP)])
            el = np.argsort(cs)[:N_ELITE]
            mu = U[el].mean(axis=0)
            best = min(best, cs[el[0]])
        final_cost = cost(mu, goal)
        err = float(np.sqrt(final_cost) * GOAL_RADIUS)
        return final_cost, best, err

    cells = []
    t1 = time.perf_counter()
    for scale_i, scale in enumerate(SCALES):
        for s in range(SEEDS):
            grng = np.random.default_rng(30_000 + s)
            goal = make_goal(grng)
            row = {"scale": scale, "seed": s}
            for si, sampler in enumerate(SAMPLERS):
                rng = np.random.default_rng(40_000 + 100 * scale_i + 10 * s + si)
                final, _, err = run_cem(goal, sampler, scale, rng)
                row[sampler] = round(final, 6)
                row[sampler + "_err_m"] = round(err, 6)
            cells.append(row)
        print("scale", scale, "done", round(time.perf_counter() - t1, 1), "s", flush=True)

    summary = {}
    for scale in SCALES:
        rows = [r for r in cells if r["scale"] == scale]
        base = np.array([r["iid"] for r in rows])
        entry = {}
        for sampler in SAMPLERS:
            v = np.array([r[sampler] for r in rows])
            err = np.array([r[sampler + "_err_m"] for r in rows])
            entry[sampler] = dict(
                median_cost=round(float(np.median(v)), 5),
                median_ratio_to_iid=round(float(np.median(v / base)), 3),
                win_rate_vs_iid=round(float(np.mean(v < base)), 2) if sampler != "iid" else None,
                success_rate_err_lt_1cm=round(float(np.mean(err < SUCCESS_ERR)), 3),
                median_final_err_mm=round(float(np.median(err)) * 1000, 2),
            )
        summary[f"scale_{scale}"] = entry

    result = {
        "source": {
            "hand": "MuJoCo Menagerie wonik_allegro right_hand.xml and right-hand STL, google-deepmind/mujoco_menagerie @ 370c439 (BSD-3 per LICENSE file)",
            "pca": "hgupt3/eigendexplore repo (MIT), src/action_bench/assets/allegro/pca.safetensors, read from the existing clone",
        },
        "config": {
            "seeds": SEEDS, "control_steps": H, "step_s": T_STEP, "substeps": nsub,
            "pop": N_POP, "elites": N_ELITE, "iterations": ITERS, "scales": SCALES,
            "k_components_90pct": k, "trace_per_dim": trace_per_dim,
            "object": {"start_palm_frame_m": OBJ_START.tolist(), "radius_m": OBJ_RADIUS,
                       "mass_kg": OBJ_MASS, "goal_distance_m": GOAL_RADIUS, "gravity": "off, no table, hand base fixed"},
            "start": "joint mid-range", "cost": "squared sphere-goal distance / goal distance^2",
            "success": f"final sphere error < {SUCCESS_ERR} m",
        },
        "summary": summary,
        "cells": cells,
        "versions": {"python": sys.version.split()[0], "mujoco": mujoco.__version__, "numpy": np.__version__,
                     "scipy": scipy.__version__, "platform": platform.platform()},
        "wall_time_s": {"cem": round(time.perf_counter() - t1, 1), "total": round(time.perf_counter() - T0, 1)},
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT.name, result["wall_time_s"], flush=True)


if __name__ == "__main__":
    main()
