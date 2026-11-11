"""Low-dimensional comparison: contact-grounded diffusion vs visuomotor and visuotactile diffusion policies (CPU, torch).

Data: the MuJoCo 3-finger hand of extra_mujoco_mapping.py (same model), tracked position targets
over 40 control steps (dt 0.02 s), half grasp-like and half free-motion targets, object at the origin
with a random radius. Observations are low-dimensional: hand joints, object radius, and (tactile
variants only) per-link contact forces. No images.

Models (all: small diffusion MLP, DDPM with 20 steps, 8-step chunks, 3 seeds):
  DP visuomotor   : predicts the action chunk from (joints, object radius)
  DP visuotactile : predicts the action chunk from (joints, object radius, tactile)
  CGP             : diffuses the future chunk of (joints, tactile), then maps each predicted
                    (joints, tactile) pair to an action with a residual mapping MLP trained on the
                    same training episodes
Metric: held-out mean absolute action error over the 8-step chunk (x 1e-3 rad). Open loop only.

Run: python3 contact_grounded_policy/analysis/extra_cgp_vs_dp.py
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import json
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_mujoco_mapping as hand  # noqa: E402

torch.set_num_threads(4)
T0 = time.time()
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "extra_cgp_vs_dp_results.json"

SEEDS = [0, 1, 2]
N_TRAIN_EP, N_TEST_EP, T_EP = 400, 100, 48
H = 8
RAMP_STEPS = 25  # the hand reaches the target by step 25 so that contacts fall inside the windows
K_DIFF = 20
DIFF_STEPS, MAP_STEPS, BATCH = 3000, 3000, 256
NJ = hand.N_FING * hand.N_LINK


KAPPA = 2000.0  # N/m, penalty stiffness of the tactile proxy


def tactile(m, d, geom_map, link_body, jacp):
    """Per-link tactile proxy: penalty force KAPPA * penetration along the contact normal (touch-sensor style).

    The solver's normal force is not used: on penetrating grasp contacts it diverged in about half the episodes.
    """
    tac = np.zeros((hand.N_FING, hand.N_LINK, 3))
    for i in range(d.ncon):
        c = d.contact[i]
        fg = c.geom1 if c.geom1 in geom_map else c.geom2
        if fg not in geom_map or c.dist >= 0:
            continue
        f, k = geom_map[fg]
        n = c.frame[:3]
        mag = KAPPA * (-c.dist)
        sgn = 1.0 if c.geom2 == fg else -1.0  # the normal points from geom1 to geom2
        tac[f, k] += sgn * mag * n
    return tac.reshape(-1)


def rollouts(rng, n):
    m = hand.build_model()
    d = mujoco.MjData(m)
    gid = m.geom("obj_g").id
    geom_map = {m.geom(f"f{f}_g{k}").id: (f, k) for f in range(hand.N_FING) for k in range(hand.N_LINK)}
    link_body = {}
    jacp = np.zeros((3, m.nv))
    eps = []
    for _ in range(n):
        r = rng.uniform(0.04, 0.06)
        m.geom_size[gid][0] = r
        mujoco.mj_resetData(m, d)
        if rng.uniform() < 0.5:  # closing toward the object
            target = rng.uniform(1.2, hand.JOINT_HI, NJ)
        else:                    # free motion
            target = rng.uniform(hand.JOINT_LO, 0.8, NJ)
        Q, U, A = [], [], []
        for t in range(T_EP):
            Q.append(d.qpos[:NJ].copy())
            U.append(tactile(m, d, geom_map, link_body, jacp))
            s = min((t + 1) / RAMP_STEPS, 1.0)
            a = target * (3 * s ** 2 - 2 * s ** 3)  # smooth ramp from open hand, then hold the target
            A.append(a)
            d.ctrl[:] = a
            for _ in range(20):  # 0.04 s per control step
                mujoco.mj_step(m, d)
        Q.append(d.qpos[:NJ].copy())
        U.append(tactile(m, d, geom_map, link_body, jacp))
        eps.append({"q": np.array(Q), "u": np.array(U), "a": np.array(A), "r": r})
    return eps


def windows(eps):
    """Per time step: joints, radius, tactile, action chunk, future joints and tactile."""
    out = {k: [] for k in ("q", "r", "u", "act", "fut")}
    for e in eps:
        for t in range(T_EP - H + 1):
            out["q"].append(e["q"][t])
            out["r"].append([e["r"]])
            out["u"].append(e["u"][t])
            out["act"].append(e["a"][t:t + H].reshape(-1))
            out["fut"].append(np.concatenate([e["q"][t + 1:t + H + 1], e["u"][t + 1:t + H + 1]], 1).reshape(-1))
    return {k: np.array(v, dtype=np.float32) for k, v in out.items()}


def sinusoid(t, dim=16):
    freqs = torch.exp(torch.linspace(0, 4, dim // 2))
    ang = t * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)], -1)


class Denoiser(nn.Module):
    def __init__(self, dim_y, dim_c):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_y + dim_c + 16, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, dim_y))

    def forward(self, y, c, k):
        t = sinusoid(k.float().unsqueeze(-1) / K_DIFF)
        return self.net(torch.cat([y, c, t], -1))


class DDPM:
    def __init__(self, dim_y, dim_c, seed):
        torch.manual_seed(seed)
        self.net = Denoiser(dim_y, dim_c)
        betas = torch.linspace(1e-3, 0.2, K_DIFF)
        self.beta = betas
        self.alpha = 1 - betas
        self.ab = torch.cumprod(self.alpha, 0)

    def fit(self, y0, c, steps):
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        N = len(y0)
        for _ in range(steps):
            idx = torch.randint(0, N, (BATCH,))
            k = torch.randint(0, K_DIFF, (BATCH,))
            eps = torch.randn_like(y0[idx])
            ab = self.ab[k].unsqueeze(-1)
            yk = ab.sqrt() * y0[idx] + (1 - ab).sqrt() * eps
            loss = ((self.net(yk, c[idx], k) - eps) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    @torch.no_grad()
    def sample(self, c, dim_y):
        y = torch.randn(len(c), dim_y)
        for k in reversed(range(K_DIFF)):
            kk = torch.full((len(c),), k)
            eps = self.net(y, c, kk)
            y = (y - self.beta[k] / (1 - self.ab[k]).sqrt() * eps) / self.alpha[k].sqrt()
            if k > 0:
                y = y + self.beta[k].sqrt() * torch.randn_like(y)
        return y


class Mapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(NJ + NJ * 3, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, NJ))


def fit_mapper(q, u, a, seed):
    torch.manual_seed(seed)
    X = np.hstack([q, u])
    mu, sd = X.mean(0), np.maximum(X.std(0), 1e-2)  # floor: tactile channels are mostly zero
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    Yt = torch.tensor((a - q) * 1e3, dtype=torch.float32)
    net = Mapper()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(MAP_STEPS):
        idx = torch.randint(0, len(Xt), (BATCH,))
        loss = (net.net((Xt[idx])) - Yt[idx]).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def apply(qq, uu):
        Z = torch.tensor((np.hstack([qq, uu]) - mu) / sd, dtype=torch.float32)
        with torch.no_grad():
            return qq + net.net(Z).numpy() / 1e3
    return apply


def main():
    rng = np.random.default_rng(0)
    tr_eps = rollouts(rng, N_TRAIN_EP)
    te_eps = rollouts(np.random.default_rng(1), N_TEST_EP)
    tr, te = windows(tr_eps), windows(te_eps)
    dim_a = H * NJ
    dim_f = H * (NJ + NJ * 3)
    q_all = np.concatenate([e["q"][:-1] for e in tr_eps])
    u_all = np.concatenate([e["u"][:-1] for e in tr_eps])
    a_all = np.concatenate([e["a"] for e in tr_eps])

    res = {"DP visuomotor": [], "DP visuotactile": [], "CGP": []}
    for seed in SEEDS:
        torch.manual_seed(seed)
        # normalisers from the training split
        def norm_stats(X):
            return X.mean(0), np.maximum(X.std(0), 1e-2)  # floor: tactile channels are mostly zero
        c_vm = np.hstack([tr["q"], tr["r"]])
        c_vt = np.hstack([tr["q"], tr["r"], tr["u"]])
        cm_vm, cs_vm = norm_stats(c_vm)
        cm_vt, cs_vt = norm_stats(c_vt)
        am, as_ = norm_stats(tr["act"])
        fm, fs = norm_stats(tr["fut"])

        def T(x):
            return torch.tensor(x, dtype=torch.float32)

        # DP visuomotor and visuotactile on action chunks
        for name, c, cm, cs, c_te in (
                ("DP visuomotor", c_vm, cm_vm, cs_vm, np.hstack([te["q"], te["r"]])),
                ("DP visuotactile", c_vt, cm_vt, cs_vt, np.hstack([te["q"], te["r"], te["u"]]))):
            dm = DDPM(dim_a, c.shape[1], seed)
            dm.fit(T((tr["act"] - am) / as_), T((c - cm) / cs), DIFF_STEPS)
            pred = dm.sample(T((c_te - cm) / cs), dim_a).numpy() * as_ + am
            err = np.abs(pred - te["act"]).mean() * 1e3
            res[name].append(float(err))

        # CGP: diffuse the future (joint, tactile) chunk conditioned on the current joints and radius,
        # then map each predicted (joint, tactile) pair to an action with the residual mapping.
        dm = DDPM(dim_f, c_vm.shape[1], seed)
        dm.fit(T((tr["fut"] - fm) / fs), T((c_vm - cm_vm) / cs_vm), DIFF_STEPS)
        pred_f = dm.sample(T((np.hstack([te["q"], te["r"]]) - cm_vm) / cs_vm), dim_f).numpy() * fs + fm
        mapper = fit_mapper(q_all, u_all, a_all, seed)
        pf = pred_f.reshape(-1, H, NJ + NJ * 3)
        acts = np.stack([mapper(pf[:, k, :NJ], pf[:, k, NJ:]) for k in range(H)], 1)
        err_cgp = np.abs(acts - te["act"].reshape(-1, H, NJ)).mean() * 1e3
        res["CGP"].append(float(err_cgp))
        print("seed", seed, {k: round(v[-1], 2) for k, v in res.items()}, flush=True)

    summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "per_seed": v} for k, v in res.items()}
    # reference: residual mapping applied to true future states and tactile (upper bound of the mapping alone)
    tf = te["fut"].reshape(-1, H, NJ + NJ * 3)
    ref = []
    for seed in SEEDS:
        mapper = fit_mapper(q_all, u_all, a_all, seed)
        acts_t = np.stack([mapper(tf[:, k, :NJ], tf[:, k, NJ:]) for k in range(H)], 1)
        ref.append(float(np.abs(acts_t - te["act"].reshape(-1, H, NJ)).mean() * 1e3))
    summary["mapper_on_true_future"] = {"mean": float(np.mean(ref)), "std": float(np.std(ref)), "per_seed": ref}
    report = {
        "device": "cpu",
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "threads": 4,
        "seeds": SEEDS,
        "settings": {"train_episodes": N_TRAIN_EP, "test_episodes": N_TEST_EP, "steps_per_episode": T_EP,
                     "chunk": H, "ddpm_steps": K_DIFF, "diffusion_train_steps": DIFF_STEPS,
                     "mapper_train_steps": MAP_STEPS, "batch": BATCH, "units": "MAE x 1e-3 rad over chunk",
                     "n_train_windows": int(len(tr["act"])), "n_test_windows": int(len(te["act"])),
                     "contact_fraction_of_windows_train": float(np.mean(np.any(np.abs(tr["u"]) > 0, 1))),
                     "contact_fraction_of_windows_test": float(np.mean(np.any(np.abs(te["u"]) > 0, 1))),
                     "tactile_proxy": "penalty KAPPA x penetration (N/m = 2000)"},
        "results_mae_1e3rad": summary,
        "wall_time_s": round(time.time() - T0, 1),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for k, v in summary.items():
        print(k, round(v["mean"], 2), "+-", round(v["std"], 2))
    print("wall time", report["wall_time_s"], "s")


if __name__ == "__main__":
    main()
