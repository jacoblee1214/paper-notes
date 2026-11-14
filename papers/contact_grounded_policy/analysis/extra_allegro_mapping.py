"""Contact-consistency mapping on a 16-DoF public hand model in MuJoCo (CPU, torch MLP, fixed seeds).

Question: in a dexterous hand with per-link contact sensing, how well can a learned mapping
from (joint state, tactile) recover the commanded target, compared with the analytic
equilibrium relation and a zero-residual baseline? Does sensing only the fingertip links
keep the information when contacts happen on proximal or medial links?

Model: the Allegro right hand (MJCF from google-deepmind/mujoco_menagerie, folder
wonik_allegro, BSD-3-Clause), loaded from a local copy. Gravity is off, joints are driven by
position actuators (gain KP, zero velocity gain), joint damping is raised for settling,
hand-to-hand self-collision is disabled, and a fixed sphere sits in the grasp volume.
Quasi-static equilibrium gives KP (a - q) = J^T f_contact, so a = q - J^T f / KP.
Tactile proxy: per-link sum of contact forces (world frame) for the links that carry a joint.

Run: python3 contact_grounded_policy/analysis/extra_allegro_mapping.py
     (set ALLEGRO_N to override the number of samples for a quick run)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import json
import re
import time
from pathlib import Path

import numpy as np
import mujoco
import torch
from torch import nn

torch.set_num_threads(4)
T0 = time.time()
ROOT = Path(__file__).resolve().parents[1]
MODEL_XML = ROOT / "data" / "menagerie" / "wonik_allegro" / "right_hand.xml"
OUT = ROOT / "analysis" / "extra_allegro_mapping_results.json"

N_SAMPLES = int(os.environ.get("ALLEGRO_N", "6000"))
SETTLE_STEPS = 900          # 1.8 s at dt = 0.002
DRIFT_WINDOW = 200          # 0.4 s; rest means no joint drift over this window
DRIFT_TOL = 2e-5            # rad
KP = 5.0                    # N m / rad, every joint
JOINT_DAMP = 0.5           # N m s / rad (menagerie default is 0.1; raised to settle faster)
SEEDS = [0, 1, 2]
EPOCHS = 60
FINGERS = ["ff", "mf", "rf", "th"]
LINK_NAMES = ["base", "proximal", "medial", "distal"]   # link k carries joint k of each finger
MARGIN = 0.03               # target kept this far inside the joint range (rad)
OBJ_R = (0.025, 0.04)       # sphere radius range (m)
OBJ_JIT = 0.012             # object centre jitter around the grasp centre (m)


def load_model():
    spec = mujoco.MjSpec.from_file(str(MODEL_XML))
    spec.option.gravity = [0.0, 0.0, 0.0]
    obj = spec.worldbody.add_body(name="obj", pos=[0.0, 0.0, 0.0])
    obj.add_geom(name="obj_g", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.03, 0.0, 0.0],
                 contype=2, conaffinity=1, mass=0.1)
    m = spec.compile()
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    m.dof_damping[:] = JOINT_DAMP
    # position actuators with a common gain KP and zero velocity gain
    m.actuator_gainprm[:, 0] = KP
    m.actuator_biasprm[:, 1] = -KP
    for i in range(m.nu):
        jid = m.actuator_trnid[i, 0]
        lo, hi = m.jnt_range[jid]
        m.actuator_ctrlrange[i] = [lo + MARGIN, hi - MARGIN]
    # frictionless contacts: with friction, the fingers stick and slip on the fixed sphere and the
    # rest criterion never settles (checked in a short run); the mapping is normal-force only
    m.geom_friction[:, 0] = 0.0
    # hand collision geoms collide with the object only (no hand self-contact)
    obj_gid = m.geom("obj_g").id
    for g in range(m.ngeom):
        if g == obj_gid:
            continue
        if m.geom_contype[g] == 1 and m.geom_conaffinity[g] == 1:
            m.geom_conaffinity[g] = 2
    return m


def main():
    m = load_model()
    d = mujoco.MjData(m)
    jnames = [f"{f}j{k}" for f in FINGERS for k in range(4)]
    nj = len(jnames)
    qadr = [m.joint(j).qposadr[0] for j in jnames]
    dof_of = [m.joint(j).dofadr[0] for j in jnames]
    lo = np.array([m.jnt_range[m.joint(j).id][0] + MARGIN for j in jnames])
    hi = np.array([m.jnt_range[m.joint(j).id][1] - MARGIN for j in jnames])

    obj_gid = m.geom("obj_g").id
    obj_bid = m.body("obj").id

    # map each hand body to (finger, link); tip bodies belong to the distal link
    body_link = {}
    for fi, f in enumerate(FINGERS):
        for k, ln in enumerate(LINK_NAMES):
            body_link[m.body(f"{f}_{ln}").id] = (fi, k)
        body_link[m.body(f"{f}_tip").id] = (fi, 3)
    n_links = len(FINGERS) * 4

    # grasp centre: fingertip mean at a closed-hand pose
    d.qpos[:] = 0.0
    d.ctrl[:] = np.where(np.arange(nj) % 4 == 0, 0.0, 0.5 * (lo + hi))
    for _ in range(1500):
        mujoco.mj_step(m, d)
    tip_b = [m.body(f"{f}_tip").id for f in FINGERS]
    centre = np.mean([d.xpos[b] for b in tip_b], axis=0)

    rng = np.random.default_rng(0)
    jacp = np.zeros((3, m.nv))
    Q, TAC, ANA, A, FREE = [], [], [], [], []
    n_discard = 0
    sign_err = 0.0
    n_nontip_contact = 0
    q_home = np.zeros(m.nq)

    n_episodes = 0
    while len(Q) < N_SAMPLES:
        n_episodes += 1
        if n_episodes > 50 * N_SAMPLES:
            raise RuntimeError("too many discarded episodes")
        # new episode: reset the hand open, place the object near the grasp centre
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q_home
        m.geom_size[obj_gid][0] = rng.uniform(*OBJ_R)
        m.body_pos[obj_bid] = centre + rng.uniform(-OBJ_JIT, OBJ_JIT, 3)
        mujoco.mj_forward(m, d)
        if d.ncon > 0:
            n_discard += 1
            continue
        for _ in range(4):   # up to 4 targets per episode, chained from the previous rest state
            if rng.uniform() < 0.6:   # grasp-like targets
                frac = rng.uniform(0.4, 1.0, nj)
                a = lo + frac * (hi - lo) + rng.normal(0.0, 0.08, nj)
            else:                     # free targets
                a = rng.uniform(lo, hi)
            a = np.clip(a, lo, hi)
            d.ctrl[:] = a
            for _ in range(SETTLE_STEPS - DRIFT_WINDOW):
                mujoco.mj_step(m, d)
            q_ref = d.qpos[qadr].copy()
            for _ in range(DRIFT_WINDOW):
                mujoco.mj_step(m, d)
            if np.abs(d.qpos[qadr] - q_ref).max() > DRIFT_TOL:
                n_discard += 1
                break
            q = d.qpos[qadr].copy()
            tac = np.zeros((n_links, 3))
            tau = np.zeros(m.nv)
            nontip = False
            for i in range(d.ncon):
                c = d.contact[i]
                if c.geom1 == obj_gid:
                    hg, sgn_hand = c.geom2, 1.0
                elif c.geom2 == obj_gid:
                    hg, sgn_hand = c.geom1, -1.0
                else:
                    continue
                b = m.geom_bodyid[hg]
                if b not in body_link:
                    continue    # palm: fixed, no joint to load
                fi, k = body_link[b]
                f6 = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, f6)
                R = c.frame.reshape(3, 3)
                F = sgn_hand * (R.T @ f6[:3])
                mujoco.mj_jac(m, d, jacp, None, c.pos, b)
                tau += jacp.T @ F
                tac[fi * 4 + k] += F
                if k < 3:
                    nontip = True
            tau_j = np.array([tau[dd] for dd in dof_of])
            # joint-limit rows of the solver are not part of the contact sum; compare only without them
            n_lim = int(np.sum(d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))
            if n_lim == 0:
                sign_err = max(sign_err, float(np.max(np.abs(tau - d.qfrc_constraint[:m.nv]))))
            touching = np.abs(tac).sum() > 0
            Q.append(q)
            TAC.append(tac.reshape(-1))
            ANA.append(q - tau_j / KP)
            A.append(a)
            FREE.append(not touching)
            n_nontip_contact += int(nontip)
            if len(Q) >= N_SAMPLES:
                break

    Q, TAC, ANA, A = np.array(Q), np.array(TAC), np.array(ANA), np.array(A)
    FREE = np.array(FREE)
    TIP_IDX = [fi * 4 + 3 for fi in range(len(FINGERS))]
    TAC3 = TAC.reshape(len(Q), n_links, 3)
    TAC_TIP = TAC3[:, TIP_IDX, :].reshape(len(Q), -1)
    has_nontip = np.abs(TAC3[:, [i for i in range(n_links) if i not in TIP_IDX], :]).sum(axis=(1, 2)) > 0
    touching = ~FREE
    idx = np.random.default_rng(1).permutation(len(Q))
    tr, va = idx[: len(Q) // 2], idx[len(Q) // 2:]

    def mae(pred, tgt, mask=None):
        e = np.abs(pred - tgt)
        if mask is not None:
            e = e[mask]
        return float(np.mean(e) * 1e3)  # x 1e-3 rad

    va_mask_c = np.zeros(len(Q), bool)
    va_mask_c[va] = True
    contact_va = va_mask_c & touching
    nontip_va = va_mask_c & has_nontip
    tip_only_va = va_mask_c & touching & ~has_nontip
    free_va = va_mask_c & FREE

    results = {}
    results["zero_residual_a_eq_q"] = {"all": mae(Q, A, va_mask_c), "contact": mae(Q, A, contact_va),
                                       "nontip_contact": mae(Q, A, nontip_va)}
    results["analytic_all_links"] = {"all": mae(ANA, A, va_mask_c), "contact": mae(ANA, A, contact_va),
                                     "nontip_contact": mae(ANA, A, nontip_va)}
    results["analytic_floor_free_space"] = mae(ANA, A, free_va)

    variants = {
        "state+tactile, residual": (np.hstack([Q, TAC]), "res"),
        "state+tactile, absolute": (np.hstack([Q, TAC]), "abs"),
        "state only, residual": (Q, "res"),
        "tactile only, residual": (TAC, "res"),
        "fingertip-only state+tactile, residual": (np.hstack([Q, TAC_TIP]), "res"),
    }
    trained = {}
    for name, (X, kind) in variants.items():
        preds = []
        for seed in SEEDS:
            pred = fit_predict(X, Q, A, kind, tr, seed)
            preds.append(pred)
        scores = {"all": [], "contact": [], "nontip_contact": [], "tip_only_contact": []}
        for pred in preds:
            scores["all"].append(mae(pred, A, va_mask_c))
            scores["contact"].append(mae(pred, A, contact_va))
            scores["nontip_contact"].append(mae(pred, A, nontip_va))
            scores["tip_only_contact"].append(mae(pred, A, tip_only_va))
        results[name] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in scores.items()}
        trained[name] = preds

    report = {
        "device": "cpu",
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "hand_model": {"source": "google-deepmind/mujoco_menagerie", "folder": "wonik_allegro",
                       "commit": "370c43965323c0944b433b202aac5058b5f907a9", "license": "BSD-3-Clause",
                       "local_copy": "contact_grounded_policy/data/menagerie (not in git)"},
        "n_joints": nj,
        "n_samples": int(len(Q)),
        "discarded_episodes_or_not_at_rest": n_discard,
        "contact_fraction": float(np.mean(touching)),
        "nontip_contact_fraction_among_contact": float(np.mean(has_nontip[touching])) if touching.any() else 0.0,
        "tip_only_contact_fraction_among_contact": float(np.mean(~has_nontip[touching])) if touching.any() else 0.0,
        "n_validation": int(len(va)),
        "n_contact_validation": int(contact_va.sum()),
        "n_nontip_contact_validation": int(nontip_va.sum()),
        "max_contact_jacobian_vs_solver_force_error": sign_err,
        "kp": KP,
        "joint_damping": JOINT_DAMP,
        "settle_steps": SETTLE_STEPS,
        "drift_tol_rad": DRIFT_TOL,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "units": "MAE x 1e-3 rad on validation split",
        "results_mae_1e3rad": results,
        "wall_time_s": round(time.time() - T0, 1),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in report if k != "results_mae_1e3rad"}, indent=1))
    for k, v in results.items():
        print(k, v)
    print("wall time", report["wall_time_s"], "s")


def fit_predict(X, Q, A, kind, tr, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tgt = (A - Q) if kind == "res" else A
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    ymu, ysd = tgt[tr].mean(0), tgt[tr].std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    Yt = torch.tensor((tgt - ymu) / ysd, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                        nn.Linear(128, Yt.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    tr_t = torch.tensor(tr)
    for _ in range(EPOCHS):
        perm = tr_t[torch.randperm(len(tr_t))]
        for b in range(0, len(perm), 128):
            bi = perm[b:b + 128]
            loss = nn.functional.l1_loss(net(Xt[bi]), Yt[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = net(Xt).numpy() * ysd + ymu
    return pred + Q if kind == "res" else pred


if __name__ == "__main__":
    main()
