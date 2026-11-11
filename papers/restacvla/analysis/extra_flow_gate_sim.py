"""Closed-loop check of tactile gating in a small flow-matching policy (CPU, torch, fixed seeds).

Self-written MuJoCo scene: a 2-DoF point finger (position-controlled slide joints, frictionless
contacts, gravity off) must press a visible pad at x = 0.15. A wall segment that the wrist
view does not show lies on the straight path in half of the episodes. Contact force on the
finger is the tactile signal (2-D). Vision gives finger position and the pad offset with noise.

Compared policy inputs (same flow-matching action-chunk policy, 8-step chunks, 4 executed):
  vision only, raw tactile token, sigma gate (gate from log sigma), sigma + prediction-error gate.
sigma comes from a heteroscedastic CMP (vision -> tactile mean and log variance) fit on
episodes without the wall, i.e. on expected contacts only. Gates are trained end-to-end with
the action loss only.

Run: python3 restacvla/analysis/extra_flow_gate_sim.py
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
OUT = ROOT / "analysis" / "extra_flow_gate_sim_results.json"

SEEDS = [0, 1, 2]
N_DEMO = 240            # expert episodes (half with wall)
N_CMP_EP = 120          # episodes without wall used to fit the CMP
N_EVAL = 60             # closed-loop test episodes per variant and seed (half with wall)
T_MAX = 160             # control steps per episode
CHUNK, EXEC = 8, 4
EXEC_SET = [4, 1]        # executed steps per chunk: 4 (main), 1 (reactive sensitivity)
STEP = 0.02             # max commanded displacement per control step (m)
LEAD = 0.10             # max lead of the position target over the finger (m); actuator force <= 50 * LEAD = 5 N
F_LIM = 3.0             # N, unexpected-contact force above this counts as damage
PAD_FACE_X = 0.15       # finger centre position at which the pad is pressed
FRAME_SKIP = 10
TRAIN_ITERS = 3000
EP_TRAIN_STEPS = 1500   # CMP iterations

MJCF = """
<mujoco model="point_finger">
  <option timestep="0.002" gravity="0 0 0" integrator="implicitfast"/>
  <default><geom friction="0 0.005 0.0001"/></default>
  <worldbody>
    <body name="hand" pos="0 0 0.02">
      <joint name="hx" type="slide" axis="1 0 0" damping="5" armature="0.001"/>
      <joint name="hy" type="slide" axis="0 1 0" damping="5" armature="0.001"/>
      <geom name="finger" type="sphere" size="0.02" mass="0.005" contype="1" conaffinity="2"/>
    </body>
    <body name="pad" pos="0.2 0 0.02">
      <geom name="padg" type="box" size="0.03 0.08 0.02" contype="2" conaffinity="1"/>
    </body>
    <body name="wall" pos="5 0 0.02">
      <geom name="wallg" type="box" size="0.01 0.03 0.02" contype="2" conaffinity="1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="ax" joint="hx" kp="50" kv="0" ctrlrange="-0.3 0.3"/>
    <position name="ay" joint="hy" kp="50" kv="0" ctrlrange="-0.15 0.15"/>
  </actuator>
</mujoco>
"""


class Scene:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_string(MJCF)
        self.d = mujoco.MjData(self.m)
        self.fid = self.m.geom("finger").id
        self.wall_body = self.m.body("wall").id
        self.jac = np.zeros((3, self.m.nv))
        self.f6 = np.zeros(6)

    def reset(self, wall_xy=None):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[:2] = [-0.2, 0.0]
        if wall_xy is None:
            m.body_pos[self.wall_body] = [5.0, 0.0, 0.02]
        else:
            m.body_pos[self.wall_body] = [wall_xy[0], wall_xy[1], 0.02]
        mujoco.mj_forward(m, d)
        self.tgt = np.array([-0.2, 0.0])
        d.ctrl[:] = self.tgt

    def pos(self):
        return self.d.qpos[:2].copy()

    def tactile(self):
        """Sum of contact forces on the finger, world frame, x-y components."""
        d, m = self.d, self.m
        F = np.zeros(2)
        for i in range(d.ncon):
            c = d.contact[i]
            if self.fid not in (c.geom1, c.geom2):
                continue
            mujoco.mj_contactForce(m, d, i, self.f6)
            R = c.frame.reshape(3, 3)
            sgn = 1.0 if c.geom2 == self.fid else -1.0
            F += (sgn * (R.T @ self.f6[:3]))[:2]
        return F

    def step(self, a):
        # target accumulates the commanded displacement but may lead the finger by at most LEAD (bounded push force)
        p = self.pos()
        self.tgt = np.clip(self.tgt + a, np.maximum(p - LEAD, [-0.3, -0.15]), np.minimum(p + LEAD, [0.3, 0.15]))
        self.d.ctrl[:] = self.tgt
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.m, self.d)


def vision(scene, rng):
    p = scene.pos()
    v = np.array([p[0], p[1], PAD_FACE_X - p[0], -p[1]])
    return v + rng.normal(0, 0.003, 4)


def expert_action(scene, u, state):
    """Scripted demonstrator: approach the pad, back off and sidestep on unexpected contact (tactile-triggered)."""
    p = scene.pos()
    unexpected = np.linalg.norm(u) > 0.3 and p[0] < 0.12
    if unexpected:
        state["retreat"] = 5
    if state.get("retreat", 0) > 0:
        state["retreat"] -= 1
        return np.array([-STEP, STEP])
    goal = np.array([PAD_FACE_X + 0.01, 0.0])
    return np.clip(0.5 * (goal - scene.tgt), -STEP, STEP)


def run_episode(scene, rng, wall, policy=None, cmp_fn=None, noise_seed=None):
    """Runs one episode. policy is None -> expert. Returns trajectory dict and metrics."""
    wall_xy = None
    if wall:
        wall_xy = (rng.uniform(-0.05, 0.1), rng.uniform(-0.02, 0.02))
    scene.reset(wall_xy)
    state = {}
    rec = {"v": [], "u": [], "a": []}
    peak_unexp = 0.0
    reached = False
    plan, plan_i = None, 0
    for t in range(T_MAX):
        v = vision(scene, rng)
        u = scene.tactile()
        rec["v"].append(v)
        rec["u"].append(u)
        p = scene.pos()
        if p[0] < 0.12:
            peak_unexp = max(peak_unexp, float(np.linalg.norm(u)))
        if p[0] >= PAD_FACE_X - 0.01:
            reached = True
        if policy is None:
            a = expert_action(scene, u, state)
        else:
            if plan is None or plan_i >= EXEC:
                plan = policy(v, u, cmp_fn)
                plan_i = 0
            a = plan[plan_i]
            plan_i += 1
        rec["a"].append(a)
        scene.step(a)
    success = reached and peak_unexp < F_LIM
    return rec, {"success": bool(success), "reached": bool(reached), "peak_unexpected_force": peak_unexp,
                 "damage": bool(peak_unexp >= F_LIM)}


def collect(rng_seed, n, wall_every=2):
    scene = Scene()
    rng = np.random.default_rng(rng_seed)
    eps, metrics = [], []
    for i in range(n):
        wall = (i % wall_every == 0)
        rec, met = run_episode(scene, rng, wall, policy=None)
        met["wall"] = bool(wall)
        eps.append(rec)
        metrics.append(met)
    return eps, metrics


class CMP(nn.Module):
    """Heteroscedastic Gaussian map from vision to tactile (mean, log variance)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, v):
        out = self.net(v)
        return out[:, :2], out[:, 2:].clamp(-8.0, 4.0)


class SigmaGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 1)

    def forward(self, feats):
        return torch.sigmoid(self.lin(feats[:, :1]))


class SigmaErrGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))

    def forward(self, feats):
        return torch.sigmoid(self.mlp(feats[:, :2]))


class FlowPolicy(nn.Module):
    def __init__(self, cond_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cond_dim + CHUNK * 2 + 1, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, CHUNK * 2))

    def forward(self, cond, x, tau):
        return self.net(torch.cat([cond, x, tau], -1))


VARIANTS = {"vision only": None, "raw tactile": None, "sigma gate": SigmaGate, "sigma+error gate": SigmaErrGate}


def feats_from(cmp, v_t, u_t, stats):
    mu, logvar = cmp(v_t)
    log_sig = 0.5 * logvar.mean(-1, keepdim=True)
    err = ((u_t - mu) ** 2).sum(-1, keepdim=True)
    f = torch.cat([log_sig, torch.log(err + 1e-4)], -1)
    return (f - stats[0]) / stats[1]


def build_cond(variant, v_n, u_t, gate, feats):
    if variant == "vision only":
        return v_n
    un = u_t / 2.0
    if variant == "raw tactile":
        return torch.cat([v_n, un], -1)
    g = gate(feats)
    return torch.cat([v_n, g * un, g], -1)


def make_chunks(eps):
    V, U, C, F = [], [], [], []
    for rec in eps:
        a = np.array(rec["a"]) / STEP
        T = len(a)
        for t in range(T):
            idx = np.minimum(np.arange(t, t + CHUNK), T - 1)
            C.append(a[idx].reshape(-1))
        V.append(np.array(rec["v"]))
        U.append(np.array(rec["u"]))
    return np.concatenate(V), np.concatenate(U), np.array(C, dtype=np.float32)


def train_cmp(V, U, seed):
    torch.manual_seed(seed)
    cmp = CMP()
    opt = torch.optim.Adam(cmp.parameters(), lr=3e-3)
    vt = torch.tensor(V, dtype=torch.float32)
    ut = torch.tensor(U, dtype=torch.float32)
    for _ in range(EP_TRAIN_STEPS):
        mu, logvar = cmp(vt)
        loss = (0.5 * (logvar + (ut - mu) ** 2 / logvar.exp())).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    cmp.eval()
    return cmp


def train_policy(variant, V, U, C, cmp, seed):
    torch.manual_seed(seed)
    vmu, vsd = V.mean(0), V.std(0) + 1e-6
    vt = torch.tensor((V - vmu) / vsd, dtype=torch.float32)
    ut = torch.tensor(U, dtype=torch.float32)
    ct = torch.tensor(C)
    with torch.no_grad():
        f_raw = feats_from(cmp, torch.tensor(V, dtype=torch.float32), ut, (torch.zeros(1, 2), torch.ones(1, 2)))
    fmu, fsd = f_raw.mean(0, keepdim=True), f_raw.std(0, keepdim=True) + 1e-6
    stats = (torch.zeros(1, 2), torch.ones(1, 2))
    feats = (f_raw - fmu) / fsd
    gate_cls = VARIANTS[variant]
    gate = gate_cls() if gate_cls is not None else None
    cond_dim = {"vision only": 4, "raw tactile": 6}.get(variant, 7)
    policy = FlowPolicy(cond_dim)
    params = list(policy.parameters()) + (list(gate.parameters()) if gate is not None else [])
    opt = torch.optim.Adam(params, lr=1e-3)
    N = len(vt)
    for _ in range(TRAIN_ITERS):
        idx = torch.randint(0, N, (256,))
        cond = build_cond(variant, vt[idx], ut[idx], gate, feats[idx])
        x1 = ct[idx]
        x0 = torch.randn_like(x1)
        tau = torch.rand(len(idx), 1)
        xt = (1 - tau) * x0 + tau * x1
        loss = ((policy(cond, xt, tau) - (x1 - x0)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return policy, gate, (vmu, vsd), (fmu, fsd), cmp


def make_fn(variant, model, rng_seed, diag):
    policy, gate, (vmu, vsd), (fmu, fsd), cmp = model
    gen = torch.Generator().manual_seed(rng_seed)

    def fn(v, u, _):
        with torch.no_grad():
            vt = torch.tensor(v, dtype=torch.float32)[None]
            ut = torch.tensor(u, dtype=torch.float32)[None]
            vn = (vt - torch.tensor(vmu, dtype=torch.float32)) / torch.tensor(vsd, dtype=torch.float32)
            feats = None
            if variant in ("sigma gate", "sigma+error gate"):
                feats = (feats_from(cmp, vt, ut, (torch.zeros(1, 2), torch.ones(1, 2))) - fmu) / fsd
            cond = build_cond(variant, vn, ut, gate, feats)
            if gate is not None and variant != "vision only":
                g = float(cond[0, -1])
                unexp = (np.linalg.norm(u) > 0.3) and (v[0] < 0.12)
                diag.append((g, bool(unexp), bool(v[0] >= 0.12 and np.linalg.norm(u) > 0.3)))
            x = torch.randn(1, CHUNK * 2, generator=gen)
            for k in range(5):
                tau = torch.full((1, 1), k / 5.0)
                x = x + policy(cond, x, tau) / 5.0
        return (x.reshape(CHUNK, 2).numpy() * STEP)

    return fn


def run_all(V, U, C, cmp, demo_met):
    results = {}
    for variant in VARIANTS:
        per_seed = []
        for seed in SEEDS:
            model = train_policy(variant, V, U, C, cmp, seed)
            diag = []
            fn = make_fn(variant, model, 1000 + seed, diag)
            scene = Scene()
            rng = np.random.default_rng(2000 + seed)
            mets = []
            for i in range(N_EVAL):
                wall = (i % 2 == 0)
                _, met = run_episode(scene, rng, wall, policy=fn)
                met["wall"] = wall
                mets.append(met)
            g_unexp = [d[0] for d in diag if d[1]]
            g_exp = [d[0] for d in diag if d[2]]
            per_seed.append({
                "success": float(np.mean([m["success"] for m in mets])),
                "success_wall": float(np.mean([m["success"] for m in mets if m["wall"]])),
                "success_no_wall": float(np.mean([m["success"] for m in mets if not m["wall"]])),
                "damage_rate_wall": float(np.mean([m["damage"] for m in mets if m["wall"]])),
                "gate_unexpected_contact_mean": float(np.mean(g_unexp)) if g_unexp else None,
                "gate_expected_contact_mean": float(np.mean(g_exp)) if g_exp else None,
            })
        keys = ["success", "success_wall", "success_no_wall", "damage_rate_wall"]
        results[variant] = {"per_seed": per_seed,
                            "mean": {k: float(np.mean([x[k] for x in per_seed])) for k in keys},
                            "std": {k: float(np.std([x[k] for x in per_seed])) for k in keys}}
        print(variant, {k: round(v, 3) for k, v in results[variant]["mean"].items()}, flush=True)
    return results


def main():
    global EXEC
    demo_eps, demo_met = collect(100, N_DEMO)
    expert_succ = float(np.mean([m["success"] for m in demo_met]))
    V, U, C = make_chunks(demo_eps)
    cmp_eps = [e for e, m in zip(demo_eps, demo_met) if not m["wall"]]
    Vc, Uc, _ = make_chunks(cmp_eps)
    cmp = train_cmp(Vc, Uc, 0)

    results = {}
    for ex in EXEC_SET:
        EXEC = ex
        print("executed steps per chunk:", ex, flush=True)
        results[f"exec{ex}_of_{CHUNK}"] = run_all(V, U, C, cmp, demo_met)
    report = {
        "device": "cpu",
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "threads": 4,
        "seeds": SEEDS,
        "settings": {"n_demo_episodes": N_DEMO, "cmp_episodes_without_wall": len(cmp_eps),
                     "n_eval_episodes_per_variant_seed": N_EVAL, "t_max_steps": T_MAX,
                     "chunk": CHUNK, "executed_sensitivity": EXEC_SET, "flow_steps_inference": 5,
                     "policy_train_iters": TRAIN_ITERS, "cmp_iters": EP_TRAIN_STEPS, "batch": 256,
                     "force_limit_N": F_LIM, "target_lead_limit_m": LEAD, "finger_mass_kg": 0.005,
                     "wall_fraction": 0.5, "step_m": STEP, "expert_demo_success": expert_succ},
        "results": results,
        "wall_time_s": round(time.time() - T0, 1),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("wall time", report["wall_time_s"], "s; expert demo success", round(expert_succ, 3))


if __name__ == "__main__":
    main()
