"""Small proprio-only chunked policy for a 3-finger planar grasp in MuJoCo (CPU, 4 threads).

Scripted demonstrations (10 episodes, two object radii) train a tiny Transformer that maps the
last four joint readings to a chunk of future joint targets. Five execution schemes are
compared on four object radii, two of which were not in the training set: open-loop replay of
one demonstration, closed-loop re-planning with chunk lengths 1, 4 and 8, and temporal
ensembling of chunks of length 8. Three training seeds. The object radius is never an input.

Run: .venv/bin/python nestdex/analysis/extra_chunk_policy.py
"""
import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "4"

import json
import math
import platform
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "extra_chunk_policy_results.json"

SIM_DT = 0.002
CTRL_DT = 0.05                       # 20 Hz control
SIM_PER_CTRL = int(round(CTRL_DT / SIM_DT))
N_CTRL = 40                          # 2.0 s episodes
BASE_R = 0.12                            # finger base radius (m)
L1, L2 = 0.05, 0.04                  # link lengths (m)
FINGER_R = 0.008
OPEN_ALPHA = 1.0
Z_OBJ = 0.04                         # object half height 0.04, rests on table
PALM_Z0 = 0.04
PALM_LIFT = 0.09
TRAIN_RADII = [0.03, 0.045]
TEST_RADII = [0.025, 0.035, 0.045, 0.055]
N_DEMO = 10
N_TEST_OFFSETS = 10
SEEDS = [0, 1, 2]
HIST = 4
ENS_K = 0.01
TRAIN_STEPS = 1500
SUCCESS_LIFT = 0.02                  # object must rise 2 cm at the end

JOINTS = [f"f{i}_j{k}" for i in range(3) for k in (1, 2)] + ["lift"]


def mjcf(radius, ox, oy):
    fingers = []
    for i in range(3):
        phi = 2 * math.pi * i / 3
        fingers.append(f"""
      <body name="f{i}_base" pos="{BASE_R * math.cos(phi):.5f} {BASE_R * math.sin(phi):.5f} 0"
            euler="0 0 {phi:.5f}">
        <body name="f{i}_l1">
          <joint name="f{i}_j1" type="hinge" axis="0 0 1" range="-0.3 1.3" damping="0.5" armature="0.01"/>
          <geom type="capsule" fromto="0 0 0 -{L1} 0 0" size="{FINGER_R}" friction="1.0 0.005 0.0001"/>
          <body name="f{i}_l2" pos="-{L1} 0 0">
            <joint name="f{i}_j2" type="hinge" axis="0 0 1" range="-0.3 1.3" damping="0.5" armature="0.01"/>
            <geom type="capsule" fromto="0 0 0 -{L2} 0 0" size="{FINGER_R}" friction="1.0 0.005 0.0001"/>
          </body>
        </body>
      </body>""")
    acts = "".join(f'<position name="{n}" joint="{n}" kp="100" ctrlrange="-0.3 1.3"/>' for n in JOINTS[:6])
    return f"""
<mujoco>
  <compiler angle="radian"/>
  <option timestep="{SIM_DT}" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="table" type="plane" size="1 1 0.1" friction="0.8 0.005 0.0001"/>
    <body name="obj" pos="{ox:.5f} {oy:.5f} {Z_OBJ}">
      <freejoint name="obj_free"/>
      <geom name="obj_g" type="cylinder" size="{radius} {Z_OBJ}" mass="0.1" friction="1.0 0.005 0.0001"/>
    </body>
    <body name="palm" pos="0 0 {PALM_Z0}">
      <joint name="lift" type="slide" axis="0 0 1" range="-0.02 0.2" damping="20"/>
      <geom type="cylinder" size="0.03 0.005" mass="0.3" contype="0" conaffinity="0"/>
      {''.join(fingers)}
    </body>
  </worldbody>
  <actuator>
    {acts}
    <position name="lift" joint="lift" kp="800" ctrlrange="-0.02 0.2"/>
  </actuator>
</mujoco>
"""


def tip_radius(alpha):
    """Distance of the fingertip from the object axis when both finger joints are at alpha."""
    x = BASE_R - L1 * math.cos(alpha) - L2 * math.cos(2 * alpha)
    y = -(L1 * math.sin(alpha) + L2 * math.sin(2 * alpha))
    return math.hypot(x, y)


def squeeze_alpha(radius):
    """Joint angle at which the fingertip reaches the object surface (bisection)."""
    target = radius + FINGER_R
    lo, hi = 0.0, OPEN_ALPHA            # tip_radius increases on [0, OPEN_ALPHA]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if tip_radius(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class Sim:
    def __init__(self, radius, ox, oy):
        self.model = mujoco.MjModel.from_xml_string(mjcf(radius, ox, oy))
        self.data = mujoco.MjData(self.model)
        self.qadr = [self.model.joint(n).qposadr[0] for n in JOINTS]
        for a in self.qadr[:6]:
            self.data.qpos[a] = OPEN_ALPHA   # start open, so fingers do not overlap the object
        mujoco.mj_forward(self.model, self.data)

    def proprio(self):
        return np.array([self.data.qpos[a] for a in self.qadr])

    def apply(self, action):
        self.data.ctrl[:] = action
        for _ in range(SIM_PER_CTRL):
            mujoco.mj_step(self.model, self.data)

    def lifted(self):
        return float(self.data.body("obj").xpos[2]) > Z_OBJ + SUCCESS_LIFT


def demo_episode(radius, ox, oy):
    """Scripted demonstrator with the object radius as privileged input.
    Closes the fingers with a 0.4 s ramp, then lifts the palm once the finger joints stop."""
    sim = Sim(radius, ox, oy)
    a_target = squeeze_alpha(radius) - 0.1
    acts, hist = [], []
    prev, still, lift_t = None, 0, None
    for k in range(N_CTRL):
        t = k * CTRL_DT
        q = sim.proprio()
        hist.append(q)
        if prev is not None and t > 0.5:
            still = still + 1 if np.max(np.abs(q[:6] - prev[:6])) < 0.002 else 0
        prev = q
        if lift_t is None and still >= 2:
            lift_t = t
        alpha = OPEN_ALPHA + min(1.0, t / 0.4) * (a_target - OPEN_ALPHA)
        s = 0.0 if lift_t is None else min(1.0, (t - lift_t) / 0.4)
        a = np.array([alpha] * 6 + [s * (PALM_LIFT - PALM_Z0)])
        acts.append(a)
        sim.apply(a)
    return np.array(hist), np.array(acts), sim.lifted()


def run_policy(sim, method, model, stats, m):
    """Runs one closed-loop episode. method: 'chunk' (execute m steps, then re-plan) or 'ens'."""
    qmu, qsd, amu, asd = stats
    hist = []
    buf = []  # for ensembling: (start_step, predicted chunk)
    action_seq = []
    for k in range(N_CTRL):
        q = sim.proprio()
        hist.append(q)
        window = hist[-HIST:]
        window = [window[0]] * (HIST - len(window)) + window
        x = torch.tensor((np.array(window) - qmu) / qsd, dtype=torch.float32)[None]
        if method == "chunk":
            if k % m == 0:
                with torch.no_grad():
                    chunk = model(x).numpy()[0] * asd + amu
                chunk = chunk.reshape(m, -1)
                plan_t0 = k
            a = chunk[k - plan_t0]
        else:  # temporal ensembling of the newest chunk predictions
            with torch.no_grad():
                pred = (model(x).numpy()[0] * asd + amu).reshape(m, -1)
            buf.append((k, pred))
            buf = [(s, p) for (s, p) in buf if k - s < m]
            num, den = 0.0, 0.0
            for (s, p) in buf:
                w = math.exp(-ENS_K * (k - s))
                num = num + w * p[k - s]
                den += w
            a = num / den
        sim.apply(a)
        action_seq.append(a)
    return sim.lifted()


class Policy(nn.Module):
    def __init__(self, m, d=64, nlayers=2):
        super().__init__()
        self.m = m
        self.inp = nn.Linear(7, d)
        self.pos = nn.Parameter(torch.zeros(HIST, d))
        layer = nn.TransformerEncoderLayer(d, 4, 128, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Linear(d, m * 7)

    def forward(self, x):
        h = self.enc(self.inp(x) + self.pos)
        return self.head(h[:, -1])


def build_dataset(demos, m):
    X, Y = [], []
    for hist, acts, *_ in demos:
        for k in range(len(acts)):
            idx = [max(0, j) for j in range(k - HIST + 1, k + 1)]
            X.append(hist[idx])
            fut = [acts[min(k + j, len(acts) - 1)] for j in range(m)]
            Y.append(np.concatenate(fut))
    return np.array(X), np.array(Y)


def train(demos, m, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X, Y = build_dataset(demos, m)
    qmu, qsd = X.reshape(-1, 7).mean(0), X.reshape(-1, 7).std(0) + 1e-3
    amu, asd = Y.mean(0), Y.std(0) + 1e-3
    xt = torch.tensor((X - qmu) / qsd, dtype=torch.float32)
    yt = torch.tensor((Y - amu) / asd, dtype=torch.float32)
    net = Policy(m)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for it in range(TRAIN_STEPS):
        opt.zero_grad()
        loss = ((net(xt) - yt) ** 2).mean()
        loss.backward()
        opt.step()
    net.eval()
    return net, (qmu, qsd, amu, asd), float(loss.item())


def main():
    t0 = time.time()
    rng = np.random.default_rng(123)
    demos = []
    for i in range(N_DEMO):
        r = TRAIN_RADII[i % 2]
        ox, oy = rng.uniform(-0.01, 0.01, size=2)
        demos.append(demo_episode(r, ox, oy) + (r, float(ox), float(oy)))
    demo_success = [bool(d[2]) for d in demos]
    print("demo success", sum(demo_success), "/", N_DEMO)

    rng_t = np.random.default_rng(7)
    tests = []
    for r in TEST_RADII:
        for _ in range(N_TEST_OFFSETS):
            ox, oy = rng_t.uniform(-0.01, 0.01, size=2)
            tests.append((r, float(ox), float(oy)))

    # privileged scripted demonstrator on the test set (feasibility reference)
    oracle = [demo_episode(r, ox, oy)[2] for (r, ox, oy) in tests]
    oracle_by_r = {str(r): float(np.mean([s for (rr, _, _), s in zip(tests, oracle) if rr == r]))
                   for r in TEST_RADII}

    # open-loop replay of demonstration 0 (radius 0.03), no feedback
    replay_acts = demos[0][1]
    replay_s, replay_by_r = [], {}
    for (r, ox, oy) in tests:
        sim = Sim(r, ox, oy)
        for a in replay_acts:
            sim.apply(a)
        replay_s.append(sim.lifted())
    replay_by_r = {str(r): float(np.mean([s for (rr, _, _), s in zip(tests, replay_s) if rr == r]))
                   for r in TEST_RADII}

    methods = {"chunk_m1": ("chunk", 1), "chunk_m4": ("chunk", 4), "chunk_m8": ("chunk", 8), "ens_m8": ("ens", 8)}
    results = {"open_loop_replay_demo0": {"success": float(np.mean(replay_s)), "by_radius": replay_by_r}}
    train_loss = {}
    for seed in SEEDS:
        for name, (method, m) in methods.items():
            model, stats, loss = train(demos, m, seed)
            train_loss[f"{name}_seed{seed}"] = round(loss, 5)
            succ = []
            for (r, ox, oy) in tests:
                sim = Sim(r, ox, oy)
                succ.append(run_policy(sim, method, model, stats, m))
            results.setdefault(name, {"success_per_seed": [], "by_radius_per_seed": {}})
            results[name]["success_per_seed"].append(round(float(np.mean(succ)), 4))
            for r in TEST_RADII:
                rv = float(np.mean([s for (rr, _, _), s in zip(tests, succ) if rr == r]))
                results[name]["by_radius_per_seed"].setdefault(str(r), []).append(round(rv, 4))
            print(seed, name, round(float(np.mean(succ)), 3), flush=True)

    for name in methods:
        vals = results[name]["success_per_seed"]
        results[name]["success_mean"] = round(float(np.mean(vals)), 4)
        results[name]["success_std_over_seeds"] = round(float(np.std(vals)), 4)
        results[name]["by_radius_mean_over_seeds"] = {
            rk: round(float(np.mean(v)), 4) for rk, v in results[name]["by_radius_per_seed"].items()}

    out = {
        "script": "nestdex/analysis/extra_chunk_policy.py",
        "device": "cpu",
        "threads": 4,
        "torch_version": torch.__version__,
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "setup": {
            "hand": "3 planar fingers, 2 hinge joints each, palm lift (7 actuators), position servo kp 100 (fingers) / 800 (palm)",
            "object": "upright cylinder, 0.08 m tall, 0.1 kg; radius not given to policy",
            "train_radii_m": TRAIN_RADII, "test_radii_m": TEST_RADII,
            "demos": N_DEMO, "test_offsets_per_radius": N_TEST_OFFSETS,
            "control_hz": 20, "episode_s": N_CTRL * CTRL_DT, "sim_dt_s": SIM_DT,
            "policy": "Transformer encoder, d=64, 2 layers, 4 heads, input = last 4 joint readings",
            "train": f"Adam lr 1e-3, wd 1e-4, {TRAIN_STEPS} full-batch steps",
            "seeds": SEEDS,
            "ensemble_weight": f"exp(-{ENS_K} * age)",
        },
        "demo_success": f"{sum(demo_success)}/{N_DEMO}",
        "oracle_scripted_demonstrator_by_radius": oracle_by_r,
        "oracle_success": round(float(np.mean(oracle)), 4),
        "train_final_loss": train_loss,
        "results": results,
        "wall_time_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k in ("demo_success", "oracle_success", "wall_time_s")}))
    print({k: (v["success_mean"] if isinstance(v, dict) and "success_mean" in v else None) for k, v in results.items()})


if __name__ == "__main__":
    main()
