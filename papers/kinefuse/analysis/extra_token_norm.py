"""Token norm and attention mass during training: flat tokens vs per-finger tokens (CPU, torch, fixed seeds).

Synthetic in-hand data (self-written generator, no external data): an object pose y in R^3
drives 64 visual tokens (weak, noisy features of y) and 16 joint readings (4 fingers x 4 joints,
each reading 4 values: angle, sin, cos, torque). Joint readings are strongly informative.
The joint features can be multiplied by a unit scale s (1 or 10) to mimic raw units.

Architectures (same width, 1 attention layer, 4 heads, residual FFN):
  flat   : tokens = [64 visual, 16 joint tokens]
  finger : tokens = [64 visual, 4 finger tokens], each pooled from its 4 joint tokens by a learned query
Normalisation: none (raw token norms enter the attention logits) or pre-LayerNorm.
Tracked during training: mean L2 norm of visual and joint-type tokens after the input embedding,
and the attention mass received by joint-type tokens (averaged over query, head and batch).
Reported with the count-based share for reference.

Run: python3 kinefuse/analysis/extra_token_norm.py
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

torch.set_num_threads(4)
T0 = time.time()
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "extra_token_norm_results.json"

SEEDS = [0, 1, 2]
N_TRAIN, N_TEST = 6000, 1000
N_VIS, D_VIS, D_EMB, N_HEAD = 64, 16, 32, 4
N_FING, N_JOINT_PER, C_JOINT = 4, 4, 4
STEPS, BATCH, LR, LOG_EVERY = 1500, 128, 1e-3, 100
SCALES = [1.0, 10.0]
ARCHS = ["flat", "finger"]
NORMS = ["none", "layernorm"]


def make_data(rng, n):
    y = rng.uniform(-1, 1, (n, 3))
    phi = np.concatenate([y, np.sin(2 * y), y ** 2], 1)                   # (n, 9)
    A = np.random.default_rng(123).normal(0, 0.5, (N_VIS, 9, D_VIS))      # fixed token projections
    vis = np.einsum("nk,ikd->nid", phi, A) + 0.5 * rng.normal(size=(n, N_VIS, D_VIS))
    W = np.random.default_rng(321).normal(0, 0.6, (N_FING * N_JOINT_PER, 3))
    ang = np.tanh(y @ W.T) + 0.02 * rng.normal(size=(n, N_FING * N_JOINT_PER))
    torque = 0.3 * ang + 0.1 * rng.normal(size=ang.shape)
    joints = np.stack([ang, np.sin(ang), np.cos(ang), torque], -1)        # (n, 16, 4)
    return vis.astype(np.float32), joints.astype(np.float32), y.astype(np.float32)


class Block(nn.Module):
    def __init__(self, ln):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_EMB) if ln else nn.Identity()
        self.ln2 = nn.LayerNorm(D_EMB) if ln else nn.Identity()
        self.qkv = nn.Linear(D_EMB, 3 * D_EMB)
        self.o = nn.Linear(D_EMB, D_EMB)
        self.ff = nn.Sequential(nn.Linear(D_EMB, 2 * D_EMB), nn.GELU(), nn.Linear(2 * D_EMB, D_EMB))

    def forward(self, x):
        B, T, _ = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, -1)
        dh = D_EMB // N_HEAD
        q = q.view(B, T, N_HEAD, dh).transpose(1, 2)
        k = k.view(B, T, N_HEAD, dh).transpose(1, 2)
        v = v.view(B, T, N_HEAD, dh).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-1, -2) / np.sqrt(dh), -1)     # (B, H, T, T)
        out = (att @ v).transpose(1, 2).reshape(B, T, D_EMB)
        x = x + self.o(out)
        x = x + self.ff(self.ln2(x))
        return x, att


class Fuser(nn.Module):
    def __init__(self, arch, ln):
        super().__init__()
        self.arch = arch
        self.vis_proj = nn.Linear(D_VIS, D_EMB)
        self.joint_proj = nn.Linear(C_JOINT, D_EMB)
        self.joint_id = nn.Parameter(torch.zeros(N_FING * N_JOINT_PER, D_EMB))
        if arch == "finger":
            self.finger_query = nn.Parameter(torch.randn(N_FING, D_EMB) * 0.1)
            self.finger_id = nn.Parameter(torch.zeros(N_FING, D_EMB))
        self.blocks = nn.ModuleList([Block(ln), Block(ln)])
        self.head = nn.Linear(D_EMB, 3)

    def embed(self, vis, joints):
        ve = self.vis_proj(vis)                                          # (B, 64, D)
        je = self.joint_proj(joints) + self.joint_id                     # (B, 16, D)
        if self.arch == "flat":
            jt = je
        else:
            je_f = je.view(-1, N_FING, N_JOINT_PER, D_EMB)
            logits = torch.einsum("bfjd,fd->bfj", je_f, self.finger_query) / np.sqrt(D_EMB)
            w = torch.softmax(logits, -1)
            jt = (w.unsqueeze(-1) * je_f).sum(2) + self.finger_id       # (B, 4, D)
        return ve, jt

    def forward(self, vis, joints, track=False):
        ve, jt = self.embed(vis, joints)
        x = torch.cat([ve, jt], 1)
        n_vis = ve.shape[1]
        masses = []
        for blk in self.blocks:
            x, att = blk(x)
            if track:
                # attention mass received by joint-type keys, averaged over queries and heads
                masses.append(att[..., n_vis:].sum(-1).mean().item())
        pooled = x.mean(1)
        out = self.head(pooled)
        if track:
            return out, ve, jt, masses
        return out


def run(arch, scale, norm, seed, data_tr, data_te):
    torch.manual_seed(seed)
    vis_tr, jnt_tr, y_tr = data_tr
    vis_te, jnt_te, y_te = data_te
    jnt_tr = jnt_tr * scale
    jnt_te = jnt_te * scale
    model = Fuser(arch, norm == "layernorm")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    vt, jt_t, yt = map(torch.tensor, (vis_tr, jnt_tr, y_tr))
    ve_te, jt_te_t, y_te_t = map(torch.tensor, (vis_te[:512], jnt_te[:512], y_te[:512]))
    traj = []
    rng = np.random.default_rng(seed)

    def snapshot(step):
        model.eval()
        with torch.no_grad():
            out, ve, jt, masses = model(ve_te, jt_te_t, track=True)
            mse = float(((out - y_te_t) ** 2).mean())
            rec = {"step": step, "test_mse": mse,
                   "vis_token_norm": float(ve.norm(dim=-1).mean()),
                   "joint_token_norm": float(jt.norm(dim=-1).mean()),
                   "attn_mass_joint_tokens": float(np.mean(masses))}
        model.train()
        traj.append(rec)

    snapshot(0)
    for step in range(1, STEPS + 1):
        idx = rng.integers(0, len(vt), BATCH)
        loss = ((model(vt[idx], jt_t[idx]) - yt[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % LOG_EVERY == 0:
            snapshot(step)
    return traj


def main():
    rng = np.random.default_rng(0)
    vis_tr, jnt_tr, y_tr = make_data(rng, N_TRAIN)
    vis_te, jnt_te, y_te = make_data(np.random.default_rng(1), N_TEST)
    results = {}
    for arch in ARCHS:
        for scale in SCALES:
            for norm in NORMS:
                key = f"{arch} | scale {scale:g} | {norm}"
                runs = [run(arch, scale, norm, s, (vis_tr, jnt_tr, y_tr), (vis_te, jnt_te, y_te)) for s in SEEDS]
                # mean over seeds of each tracked quantity at each checkpoint
                steps = [r["step"] for r in runs[0]]
                mean_traj = []
                for i, st in enumerate(steps):
                    entry = {"step": st}
                    for k in ("test_mse", "vis_token_norm", "joint_token_norm", "attn_mass_joint_tokens"):
                        vals = [r[i][k] for r in runs]
                        entry[k] = float(np.mean(vals))
                        entry[k + "_std"] = float(np.std(vals))
                    mean_traj.append(entry)
                n_joint = 16 if arch == "flat" else 4
                count_share = n_joint / (N_VIS + n_joint)
                results[key] = {"trajectory_mean_over_seeds": mean_traj,
                                "count_based_share_joint_tokens": count_share,
                                "final_norm_ratio_joint_over_vis": mean_traj[-1]["joint_token_norm"] / mean_traj[-1]["vis_token_norm"],
                                "initial_norm_ratio_joint_over_vis": mean_traj[0]["joint_token_norm"] / mean_traj[0]["vis_token_norm"]}
                print(key, "final mass", round(mean_traj[-1]["attn_mass_joint_tokens"], 3),
                      "final mse", round(mean_traj[-1]["test_mse"], 4),
                      "norm ratio", round(results[key]["final_norm_ratio_joint_over_vis"], 2), flush=True)
    report = {
        "device": "cpu",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "threads": 4,
        "seeds": SEEDS,
        "settings": {"n_train": N_TRAIN, "n_test": N_TEST, "n_visual_tokens": N_VIS, "d_visual_input": D_VIS,
                     "d_model": D_EMB, "heads": N_HEAD, "layers": 2, "steps": STEPS, "batch": BATCH,
                     "lr": LR, "scales": SCALES, "joint_feature_dim": C_JOINT,
                     "joints": N_FING * N_JOINT_PER, "fingers": N_FING},
        "results": results,
        "wall_time_s": round(time.time() - T0, 1),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("wall time", report["wall_time_s"], "s")


if __name__ == "__main__":
    main()
