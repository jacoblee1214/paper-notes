"""Pair-count curve for a small two-modality alignment trained only on simulated pairs (CPU, torch).

Self-written simulator (not the official one): a 12 x 12 taxel grid (1 mm pitch) is indented by a
sphere, flat punch or cylinder. Depth map = penetration per taxel (penalty-style). Stress map =
blurred depth with a nonlinearity (normal channel) plus two shear channels from its gradient,
5 % noise. Force label = summed normal stress. The stress is a deterministic function of depth
plus noise, so this is a sanity-scale test of the pair-count question, not a model of real stress.

Alignment: a small ViT for depth and a small ViT for stress, symmetric InfoNCE on the simulated pairs.
Downstream: a shallow MLP trained on 500 labelled simulated depth latents (encoder frozen) predicts the
force and classifies the indenter type. Held-out simulated test set: 2000 samples.
Pair counts: 0 (random encoder, no alignment), 100, 300, 1000, 3000, 10000; 3 seeds each.

Run: python3 tactspace/analysis/extra_pair_count.py
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import json
import time
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as Fn
from torch import nn

torch.set_num_threads(4)
T0 = time.time()
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "extra_pair_count_results.json"

G = 12
N_POOL, N_TEST, N_LABEL = 10000, 2000, 500
PAIR_COUNTS = [0, 100, 300, 1000, 3000, 10000]
SEEDS = [0, 1, 2]
ALIGN_STEPS, BATCH, LR, TAU = 1000, 128, 1e-3, 0.1
HEAD_STEPS = 600


def make_pairs(rng, n):
    xs = (np.arange(G) - (G - 1) / 2) * 1.0
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    depth = np.zeros((n, G, G), np.float32)
    kind = rng.integers(0, 3, n)
    force = np.zeros(n, np.float32)
    for i in range(n):
        cx, cy = rng.uniform(-3, 3, 2)
        d = rng.uniform(0.2, 1.5)
        if kind[i] == 0:      # sphere
            R = rng.choice([2.0, 4.0, 6.0])
            r2 = (X - cx) ** 2 + (Y - cy) ** 2
            pen = np.where(r2 < R * R, d - (R - np.sqrt(np.maximum(R * R - r2, 0))), 0.0)
        elif kind[i] == 1:    # flat punch
            a = rng.choice([2.0, 3.0, 4.0])
            pen = np.where((X - cx) ** 2 + (Y - cy) ** 2 < a * a, d, 0.0)
        else:                 # cylinder along y
            R = rng.choice([2.0, 4.0, 6.0])
            dx = X - cx
            pen = np.where(np.abs(dx) < R, d - (R - np.sqrt(np.maximum(R * R - dx * dx, 0))), 0.0)
        depth[i] = np.maximum(pen, 0.0)
    stress_n = np.zeros_like(depth)
    stress = np.zeros((n, 3, G, G), np.float32)
    for i in range(n):
        s = ndi.gaussian_filter(depth[i], sigma=1.2) * 2.0
        sn = np.tanh(s)
        gy, gx = np.gradient(s)
        stress[i, 0] = sn + 0.02 * rng.normal(size=(G, G))
        stress[i, 1] = gx + 0.02 * rng.normal(size=(G, G))
        stress[i, 2] = gy + 0.02 * rng.normal(size=(G, G))
        stress_n[i] = sn
        force[i] = float(sn.sum())
    return depth[:, None].astype(np.float32), stress, force, kind


class ViT(nn.Module):
    def __init__(self, in_ch, out_dim=32, dim=32, depth=2, heads=2, patch=3):
        super().__init__()
        self.patch = nn.Conv2d(in_ch, dim, patch, stride=patch)
        n_tok = (G // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n_tok + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.out = nn.Linear(dim, out_dim)

    def forward(self, x):
        t = self.patch(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1) + self.pos
        return self.out(self.enc(t)[:, 0])


def info_nce(a, b):
    a, b = Fn.normalize(a, dim=-1), Fn.normalize(b, dim=-1)
    logits = a @ b.T / TAU
    lab = torch.arange(len(a))
    return 0.5 * (Fn.cross_entropy(logits, lab) + Fn.cross_entropy(logits.T, lab))


def align(depth, stress, n_pairs, seed, recon):
    """Train depth and stress encoders on the first n_pairs simulated pairs. recon adds cross-reconstruction."""
    torch.manual_seed(seed)
    enc_d, enc_s = ViT(1), ViT(3)
    params = list(enc_d.parameters()) + list(enc_s.parameters())
    if recon:
        dec_d = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, G * G))  # stress latent -> depth
        dec_s = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 3 * G * G))  # depth latent -> stress
        params += list(dec_d.parameters()) + list(dec_s.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    if n_pairs == 0:
        return enc_d.eval(), enc_s.eval()
    rng = np.random.default_rng(seed)
    dt = torch.tensor(depth[:n_pairs])
    st = torch.tensor(stress[:n_pairs])
    bsz = min(BATCH, n_pairs)
    for _ in range(ALIGN_STEPS):
        idx = torch.tensor(rng.choice(n_pairs, bsz, replace=False))
        zd, zs = enc_d(dt[idx]), enc_s(st[idx])
        loss = info_nce(zd, zs)
        if recon:
            rd = ((dec_d(zs) - dt[idx].flatten(1)) ** 2).mean()
            rs = ((dec_s(zd) - st[idx].flatten(1)) ** 2).mean()
            loss = loss + rd + rs
        opt.zero_grad()
        loss.backward()
        opt.step()
    return enc_d.eval(), enc_s.eval()


@torch.no_grad()
def embed(enc, x):
    return torch.cat([enc(x[i:i + 512]) for i in range(0, len(x), 512)])


def downstream(enc_d, depth, force, kind, lab_idx, test_idx, seed):
    """Shallow MLP on frozen depth latents, trained on labelled simulated samples only."""
    torch.manual_seed(seed)
    z = embed(enc_d, torch.tensor(depth))
    mu, sd = z[lab_idx].mean(0), z[lab_idx].std(0) + 1e-6
    z = (z - mu) / sd
    y = torch.tensor(force)
    ym, ys = y[lab_idx].mean(), y[lab_idx].std()
    yn = (y - ym) / ys
    reg = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1))
    cls = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 3))
    kt = torch.tensor(kind)
    opt = torch.optim.Adam(list(reg.parameters()) + list(cls.parameters()), lr=3e-3)
    for _ in range(HEAD_STEPS):
        loss = Fn.mse_loss(reg(z[lab_idx]).squeeze(-1), yn[lab_idx]) + \
            Fn.cross_entropy(cls(z[lab_idx]), kt[lab_idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = reg(z[test_idx]).squeeze(-1) * ys + ym
        yt = y[test_idx]
        r2 = 1 - float(((pred - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
        acc = float((cls(z[test_idx]).argmax(-1) == kt[test_idx]).float().mean())
    return r2, acc


def main():
    rng = np.random.default_rng(0)
    depth, stress, force, kind = make_pairs(rng, N_POOL + N_TEST)
    tr_d, tr_s, tr_f, tr_k = depth[:N_POOL], stress[:N_POOL], force[:N_POOL], kind[:N_POOL]
    te_d, te_f, te_k = depth[N_POOL:], force[N_POOL:], kind[N_POOL:]
    depth_all = np.concatenate([tr_d, te_d])
    force_all = np.concatenate([tr_f, te_f])
    kind_all = np.concatenate([tr_k, te_k])
    test_idx = torch.arange(N_POOL, N_POOL + N_TEST)
    rs = np.random.default_rng(7)

    curve = {}
    for n in PAIR_COUNTS:
        r2s, accs = [], []
        for seed in SEEDS:
            enc_d, _ = align(tr_d, tr_s, n, seed, recon=False)
            lab_idx = torch.tensor(rs.choice(N_POOL, N_LABEL, replace=False))
            r2, acc = downstream(enc_d, depth_all, force_all, kind_all, lab_idx, test_idx, seed)
            r2s.append(r2)
            accs.append(acc)
        curve[str(n)] = {"force_r2_mean": float(np.mean(r2s)), "force_r2_std": float(np.std(r2s)),
                         "shape_acc_mean": float(np.mean(accs)), "shape_acc_std": float(np.std(accs)),
                         "force_r2_per_seed": r2s, "shape_acc_per_seed": accs}
        print("pairs", n, "force R^2", round(np.mean(r2s), 3), "+-", round(np.std(r2s), 3),
              "acc", round(np.mean(accs), 3), flush=True)

    # Pair-count reference: how often is the stress latent informative? (not reported as a claim)
    lab_fixed = torch.tensor(np.random.default_rng(7).choice(N_POOL, N_LABEL, replace=False))

    report = {
        "device": "cpu",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "threads": 4,
        "seeds": SEEDS,
        "settings": {"grid": G, "pitch_mm": 1.0, "n_pool_pairs": N_POOL, "n_test": N_TEST,
                     "n_labelled_downstream": N_LABEL, "encoder": "ViT dim 32, depth 2, heads 2, patch 3, out 32",
                     "align_steps": ALIGN_STEPS, "batch": BATCH, "lr": LR, "temperature": TAU,
                     "head_steps": HEAD_STEPS, "pair_counts": PAIR_COUNTS},
        "pair_count_curve": curve,
        "wall_time_s": round(time.time() - T0, 1),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("wall time", report["wall_time_s"], "s")


if __name__ == "__main__":
    main()
