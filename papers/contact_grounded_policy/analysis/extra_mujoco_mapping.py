"""MuJoCo multi-finger check of the contact-consistency mapping (CPU, torch MLP, fixed seeds).

Question: in a simulated multi-finger hand with per-link contact sensing, does the
mapping from (state, tactile) to the commanded target behave as in the paper's mapping
ablation? Compared: residual vs absolute targets, state vs tactile inputs, a zero-residual
baseline, the analytic mapping a = q - Kp^-1 J^T f (all links, and fingertip links only).

Model: 3 planar fingers, 3 hinge joints each (vertical axes), position actuators with
zero velocity gain, joint damping for settling, gravity off, fixed spherical object.
Quasi-static equilibrium gives kp (a - q) + J^T f = 0, so a = q - Kp^-1 J^T f exactly.
Tactile proxy: per-link sum of contact forces in the world frame (3 values per link).
Fingertip-only sensing keeps the distal link of each finger.

Run: python3 contact_grounded_policy/analysis/extra_mujoco_mapping.py
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import json
import time
from pathlib import Path

import numpy as np
import mujoco
import torch
from torch import nn

torch.set_num_threads(4)
T0 = time.time()
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "extra_mujoco_mapping_results.json"

N_SAMPLES = 10000
SETTLE_STEPS = 2500         # 5 s at dt = 0.002 (time constant Kd/Kp = 0.25 s)
REST_WINDOW = 500           # drift window used as the rest criterion
KP = 2.0                    # N m / rad, same for all joints
N_FING, N_LINK = 3, 3
BASE_R = 0.05               # palm radius of the finger bases (m)
LINKS = [0.05, 0.045, 0.04]   # link lengths (m)
JOINT_LO, JOINT_HI = -0.2, 1.4        # target range
RANGE_HI = 1.7                       # joint limit, kept above the target range to avoid limit chatter
SEEDS = [0, 1, 2]
EPOCHS = 100


def fingers_xml():
    lines = []
    lengths = LINKS
    for f in range(N_FING):
        ang = 2 * np.pi * f / N_FING
        x, y = BASE_R * np.cos(ang), BASE_R * np.sin(ang)
        # local x axis points radially outward at zero joint angle (open hand); flexion curls the finger
        yaw = ang
        body = f'<body name="f{f}_base" pos="{x:.5f} {y:.5f} 0" euler="0 0 {yaw:.5f}">'
        for k in range(N_LINK):
            L = lengths[k]
            body += (f'<body name="f{f}_l{k}" pos="{0 if k == 0 else lengths[k-1]} 0 0">'
                     f'<joint name="f{f}_j{k}" type="hinge" axis="0 0 1" range="{JOINT_LO} {RANGE_HI}" damping="0.5" armature="1e-4"/>'
                     f'<geom name="f{f}_g{k}" type="capsule" fromto="0 0 0 {L} 0 0" size="0.008" '
                     f'contype="1" conaffinity="2" mass="0.01" friction="0 0.005 0.0001"/>')
        body += "</body>" * N_LINK + "</body>"
        lines.append(body)
    return "\n".join(lines)


def build_model():
    xml = f"""
<mujoco model="hand3">
  <compiler angle="radian"/>
  <option gravity="0 0 0" timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    {fingers_xml()}
    <body name="obj" pos="0 0 0">
      <geom name="obj_g" type="sphere" size="0.03" contype="2" conaffinity="1" mass="0.1" friction="0 0.005 0.0001"/>
    </body>
  </worldbody>
  <actuator>
    {"".join(f'<position name="a{f}_{k}" joint="f{f}_j{k}" kp="{KP}" kv="0" ctrlrange="{JOINT_LO} {RANGE_HI}"/>' for f in range(N_FING) for k in range(N_LINK))}
  </actuator>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def main():
    m = build_model()
    d = mujoco.MjData(m)
    nj = N_FING * N_LINK
    obj_gid = m.geom("obj_g").id
    # link bodies and geom ids for finger geoms
    link_body = {}
    finger_geom_to_link = {}
    for f in range(N_FING):
        for k in range(N_LINK):
            bid = m.body(f"f{f}_l{k}").id
            link_body[(f, k)] = bid
            finger_geom_to_link[m.geom(f"f{f}_g{k}").id] = (f, k)
    body_to_link = {v: k for k, v in link_body.items()}
    dof_of = [m.joint(f"f{f}_j{k}").dofadr[0] for f in range(N_FING) for k in range(N_LINK)]

    rng = np.random.default_rng(0)
    Q, TAC, TAC_TIP, ANA, ANA_TIP, A = [], [], [], [], [], []
    sign_ok = []
    n_contact = 0
    n_tip_contact = 0
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    n_discard = 0
    while len(Q) < N_SAMPLES:
        m.geom_size[obj_gid][0] = rng.uniform(0.03, 0.05)
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        if d.ncon > 0:  # object overlaps the open hand at reset: not a valid start
            n_discard += 1
            continue
        if rng.uniform() < 0.5:  # grasp-like targets (closing toward the object)
            a = np.clip(rng.uniform(0.9, JOINT_HI) + rng.normal(0.0, 0.1, nj), JOINT_LO, JOINT_HI)
        else:                    # free-motion targets
            a = rng.uniform(JOINT_LO, 1.2, nj)
        d.ctrl[:] = a
        for _ in range(SETTLE_STEPS - REST_WINDOW):
            mujoco.mj_step(m, d)
        q_ref = d.qpos[:nj].copy()
        for _ in range(REST_WINDOW):
            mujoco.mj_step(m, d)
        # at rest: no drift over the last REST_WINDOW steps (contact jitter of ~1e-4 rad is allowed)
        if np.abs(d.qpos[:nj] - q_ref).max() > 2e-4:
            n_discard += 1
            continue
        q = d.qpos[:nj].copy()

        tac = np.zeros((N_FING, N_LINK, 3))
        tau_all = np.zeros(m.nv)
        tau_tip = np.zeros(m.nv)
        for i in range(d.ncon):
            c = d.contact[i]
            fg = c.geom1 if c.geom1 in finger_geom_to_link else c.geom2
            if fg not in finger_geom_to_link:
                continue
            f, k = finger_geom_to_link[fg]
            f6 = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f6)
            R = c.frame.reshape(3, 3)
            f_world = R.T @ f6[:3]
            sgn = 1.0 if c.geom2 == fg else -1.0   # force on the finger link
            F = sgn * f_world
            mujoco.mj_jac(m, d, jacp, jacr, c.pos, link_body[(f, k)])
            tau_all += jacp.T @ F
            if k == N_LINK - 1:
                tau_tip += jacp.T @ F
                n_tip_contact += 1
            tac[f, k] += F
        if d.ncon > 0:
            n_contact += 1
        # sign check: J^T f over all finger contacts must equal the solver's constraint force
        sign_ok.append(float(np.max(np.abs(tau_all - d.qfrc_constraint[:m.nv]))))  # nonzero only at joint limits

        tau_j = np.array([d.qfrc_constraint[dof] for dof in dof_of])  # contact + joint-limit forces
        tau_tip_j = np.array([tau_tip[dof] for dof in dof_of])
        tac_tip = np.stack([tac[f, N_LINK - 1] for f in range(N_FING)])
        Q.append(q)
        TAC.append(tac.reshape(-1))
        TAC_TIP.append(tac_tip.reshape(-1))
        ANA.append(q - tau_j / KP)
        ANA_TIP.append(q - tau_tip_j / KP)
        A.append(a)

    Q, TAC, TAC_TIP = np.array(Q), np.array(TAC), np.array(TAC_TIP)
    ANA, ANA_TIP, A = np.array(ANA), np.array(ANA_TIP), np.array(A)
    sign_err = float(np.max(sign_ok))
    # mismatch between Jacobian-based torque and solver constraint force (should be ~1e-6)
    print(f"samples {N_SAMPLES}, discarded {n_discard}, contact fraction {n_contact / N_SAMPLES:.3f}, "
          f"max |J^T f - qfrc_constraint| {sign_err:.2e}")

    # Table-III style comparison on a 1:1 train/validation split
    idx = rng.permutation(N_SAMPLES)
    tr, va = idx[: N_SAMPLES // 2], idx[N_SAMPLES // 2:]
    def mae(pred, tgt):
        return float(np.mean(np.abs(pred - tgt)) * 1e3)  # x10^-3 rad

    results = {}
    results["zero_residual_a_eq_q"] = mae(Q[va], A[va])
    results["analytic_all_links"] = mae(ANA[va], A[va])
    results["analytic_fingertip_only"] = mae(ANA_TIP[va], A[va])

    variants = {
        "state+tactile, residual": (np.hstack([Q, TAC]), "res"),
        "state+tactile, absolute": (np.hstack([Q, TAC]), "abs"),
        "state only, residual": (Q, "res"),
        "tactile only, residual": (TAC, "res"),
        "tactile only, absolute": (TAC, "abs"),
        "fingertip-only state+tactile, residual": (np.hstack([Q, TAC_TIP]), "res"),
    }
    for name, (X, kind) in variants.items():
        vals = []
        for seed in SEEDS:
            vals.append(fit_and_eval(X, Q, A, kind, tr, va, seed))
        results[name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "per_seed": vals}
    report = {
        "device": "cpu",
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "threads": 4,
        "n_samples": N_SAMPLES,
        "discarded_not_at_rest": n_discard,
        "train_val_split": "1:1",
        "settle_steps": SETTLE_STEPS,
        "kp": KP,
        "contact_fraction": n_contact / N_SAMPLES,
        "fingertip_contact_events_per_sample": n_tip_contact / N_SAMPLES,
        "max_jacobian_vs_solver_force_error": sign_err,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "units": "MAE x 1e-3 rad on validation split",
        "results_mae_1e3rad": results,
        "wall_time_s": round(time.time() - T0, 1),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for k, v in results.items():
        print(k, v if not isinstance(v, dict) else f"{v['mean']:.2f} +- {v['std']:.2f}")
    print("wall time", report["wall_time_s"], "s")


def fit_and_eval(X, Q, A, kind, tr, va, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tgt = (A - Q) if kind == "res" else A
    tgt = tgt * 1e3  # x10^-3 rad, keeps the output scale near 1
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    Yt = torch.tensor(tgt, dtype=torch.float32)
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
        pred = net(Xt[va]).numpy() / 1e3
    pred_abs = pred + Q[va] if kind == "res" else pred
    return float(np.mean(np.abs(pred_abs - A[va])) * 1e3)


if __name__ == "__main__":
    main()
