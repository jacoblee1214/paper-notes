"""CEM posture tracking in MuJoCo with EigenDEXplore-style exploration covariances (CPU).

Hand: public Allegro right hand (MuJoCo Menagerie, wonik_allegro), 16 position actuators.
Basis: public Allegro PCA asset (safetensors, read with the standard library).
Samplers compared per target (paired seeds):
  iid           Sigma = I
  eigen_total   Sigma = 0.5 I + Sigma_e  (PCA-shaped eigen part, same trace as iid)
  eigen_2x      2 x eigen_total          (twice the total exploration)
  random_dir    0.5 I + Sigma_r          (same eigen spectrum, random orthonormal directions)
Targets: in-subspace (PCA prior), off-subspace (complement of the PCA span), isotropic.
Optimizer: cross-entropy method over 4 control steps x 16 joints, 10 iterations, 32 samples,
3 elites. Cost: normalized squared error of the final joint posture to the target.
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
OUT = HERE / "extra_eigen_cem_results.json"

SEEDS = 40
H = 4                 # control steps
T_STEP = 0.1          # seconds per control step
N_POP = 32
N_ELITE = 3
ITERS = 10
TRACE_TARGET = 0.5    # mean variance of each noise part (total = 1 = iid)
SCALES = [0.25, 0.10] # exploration step size in normalized joint units (two regimes)
TARGETS = ["in_subspace", "off_subspace", "isotropic"]
SAMPLERS = ["iid", "eigen_total", "eigen_2x", "random_dir"]
K_VAR = 0.90          # components kept for 90% explained variance


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


def main():
    rng_master = np.random.default_rng(0)
    t = load_safetensors(PCA)
    man = yaml.safe_load(MANIFEST.read_text())
    lo, hi = np.array(man["joint_lower"], float), np.array(man["joint_upper"], float)
    c, half = (lo + hi) / 2, (hi - lo) / 2
    d = len(lo)
    ev = t["explained_variance"]
    k = int(np.searchsorted(np.cumsum(ev) / ev.sum(), K_VAR) + 1)
    B = t["components"][:k] / half                # (k, d) in normalized coordinates
    sig = t["component_std"][:k]
    mean_n = (t["mean"] - c) / half

    Sig_e = (B * (sig ** 2)[:, None]).T @ B        # eigen part, unnormalized
    Sig_e = scale_trace(Sig_e, TRACE_TARGET)
    Qr = np.linalg.qr(rng_master.standard_normal((d, k)))[0].T   # random orthonormal rows
    Sig_r = scale_trace((Qr * (sig ** 2)[:, None]).T @ Qr, TRACE_TARGET)
    covs = {
        "iid": np.eye(d),
        "eigen_total": TRACE_TARGET * np.eye(d) + Sig_e,
        "eigen_2x": 2 * (TRACE_TARGET * np.eye(d) + Sig_e),
        "random_dir": TRACE_TARGET * np.eye(d) + Sig_r,
    }
    chol = {name: np.linalg.cholesky(S + 1e-9 * np.eye(d)) for name, S in covs.items()}
    # Allegro right: trace per dimension is 1 for iid, eigen_total, random_dir; 2 for eigen_2x
    trace_per_dim = {name: round(float(np.trace(S) / d), 4) for name, S in covs.items()}

    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    jn = ["ffj0", "ffj1", "ffj2", "ffj3", "mfj0", "mfj1", "mfj2", "mfj3",
          "rfj0", "rfj1", "rfj2", "rfj3", "thj0", "thj1", "thj2", "thj3"]
    jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in jn]
    qadr = np.array([model.jnt_qposadr[j] for j in jid])
    act_for_joint = []
    for j in jid:
        a = [i for i in range(model.nu) if model.actuator_trnid[i, 0] == j]
        act_for_joint.append(a[0])
    act_for_joint = np.array(act_for_joint)
    nsub = int(round(T_STEP / model.opt.timestep))
    q0 = np.zeros(d)                               # normalized start: joint mid-range

    def to_joint(un):
        return c + un * half

    def cost(seq, target_n):
        mujoco.mj_resetData(model, data)
        data.qpos[qadr] = to_joint(q0)
        mujoco.mj_forward(model, data)
        for h in range(seq.shape[0]):
            data.ctrl[act_for_joint] = to_joint(seq[h])
            for _ in range(nsub):
                mujoco.mj_step(model, data)
        qf = (data.qpos[qadr] - c) / half
        return float(np.mean((qf - target_n) ** 2))

    Pb = np.linalg.qr(B.T)[0]                      # orthonormal basis of the PCA span (d x k)
    r_in = float(np.mean([np.linalg.norm((np.random.default_rng(i).standard_normal(k) * sig) @ B)
                          for i in range(200)]))   # typical in-subspace deviation norm

    def make_target(kind, rng):
        if kind == "in_subspace":
            z = rng.standard_normal(k)
            x = mean_n + (z * sig) @ B
        elif kind == "off_subspace":
            z = rng.standard_normal(d)
            z = z - Pb @ (Pb.T @ z)                       # complement of the PCA span
            x = mean_n + z / np.linalg.norm(z) * r_in
        else:
            x = rng.uniform(-0.9, 0.9, size=d)
        return np.clip(x, -0.95, 0.95)

    def run_cem(target_n, sampler, scale, rng):
        Lc = chol[sampler]
        mu = np.tile(q0, (H, 1))
        best = np.inf
        for _ in range(ITERS):
            eps = rng.standard_normal((N_POP, H, d)) @ Lc.T
            U = np.clip(mu[None] + scale * eps, -0.95, 0.95)
            cs = np.array([cost(U[i], target_n) for i in range(N_POP)])
            el = np.argsort(cs)[:N_ELITE]
            mu = U[el].mean(axis=0)
            best = min(best, cs[el[0]])
        return cost(mu, target_n), best

    cells = []
    t1 = time.perf_counter()
    for kind_i, kind in enumerate(TARGETS):
        for scale_i, scale in enumerate(SCALES):
            for s in range(SEEDS):
                trng = np.random.default_rng(10_000 + 100 * kind_i + s)
                target_n = make_target(kind, trng)
                row = {"target": kind, "scale": scale, "seed": s}
                for si, sampler in enumerate(SAMPLERS):
                    rng = np.random.default_rng(20_000 + 1000 * kind_i + 100 * scale_i + 10 * s + si)
                    final, best = run_cem(target_n, sampler, scale, rng)
                    row[sampler] = round(final, 6)
                cells.append(row)
            print(kind, scale, "done", round(time.perf_counter() - t1, 1), "s", flush=True)

    summary = {}
    for kind in TARGETS:
        for scale in SCALES:
            rows = [r for r in cells if r["target"] == kind and r["scale"] == scale]
            base = np.array([r["iid"] for r in rows])
            entry = {}
            for sampler in SAMPLERS:
                v = np.array([r[sampler] for r in rows])
                entry[sampler] = dict(
                    median_cost=round(float(np.median(v)), 5),
                    median_ratio_to_iid=round(float(np.median(v / base)), 3),
                    win_rate_vs_iid=round(float(np.mean(v < base)), 2) if sampler != "iid" else None,
                )
            summary[f"{kind}_scale_{scale}"] = entry

    result = {
        "source": {
            "hand": "MuJoCo Menagerie wonik_allegro right_hand.xml and 12 right-hand STL, google-deepmind/mujoco_menagerie @ 370c439 (BSD-3 per LICENSE file)",
            "hand_bytes": {p.name: p.stat().st_size for p in sorted((ROOT / "data" / "menagerie" / "wonik_allegro").rglob("*")) if p.is_file()},
            "pca": "hgupt3/eigendexplore repo (MIT), src/action_bench/assets/allegro/pca.safetensors, read from the existing clone (not re-downloaded)",
            "pca_bytes": PCA.stat().st_size,
            "pca_manifest_joint_limits": "manifest.yaml in the same folder",
        },
        "config": {
            "seeds": SEEDS, "control_steps": H, "step_s": T_STEP, "substeps": nsub,
            "pop": N_POP, "elites": N_ELITE, "iterations": ITERS, "scales": SCALES,
            "k_components_90pct": k, "trace_per_dim": trace_per_dim,
            "joint_order": jn, "start": "joint mid-range", "cost": "mean normalized squared error of final posture",
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
