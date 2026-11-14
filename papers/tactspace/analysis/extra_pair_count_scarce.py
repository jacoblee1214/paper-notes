"""Pair-count curve with scarce downstream labels: same simulated pairs and ViT alignment as extra_pair_count.py.

Reuses the simulator, the two-ViT symmetric InfoNCE alignment, and the frozen-depth-latent downstream head from
extra_pair_count.py (import, not copy). The only change is the number of labelled simulated samples for the
downstream head: 20 and 50 instead of 500, so that the downstream task is no longer at the ceiling
(random-encoder force R^2 was 0.995 with 500 labels).

Run: python3 tactspace/analysis/extra_pair_count_scarce.py
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extra_pair_count as base  # noqa: E402

torch.set_num_threads(4)
T0 = time.time()
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "extra_pair_count_scarce_results.json"

LABEL_COUNTS = [20, 50]
PAIR_COUNTS = base.PAIR_COUNTS
SEEDS = base.SEEDS


def main():
    rng = np.random.default_rng(0)
    depth, stress, force, kind = base.make_pairs(rng, base.N_POOL + base.N_TEST)
    n_pool, n_test = base.N_POOL, base.N_TEST
    tr_d, tr_s = depth[:n_pool], stress[:n_pool]
    test_idx = torch.arange(n_pool, n_pool + n_test)

    curve = {str(L): {} for L in LABEL_COUNTS}
    for n in PAIR_COUNTS:
        per = {L: {"r2": [], "acc": []} for L in LABEL_COUNTS}
        for seed in SEEDS:
            enc_d, _ = base.align(tr_d, tr_s, n, seed, recon=False)
            for L in LABEL_COUNTS:
                lab_rng = np.random.default_rng(100 * seed + L)
                lab_idx = torch.tensor(lab_rng.choice(n_pool, L, replace=False))
                r2, acc = base.downstream(enc_d, depth, force, kind, lab_idx, test_idx, seed)
                per[L]["r2"].append(r2)
                per[L]["acc"].append(acc)
        for L in LABEL_COUNTS:
            r2s, accs = per[L]["r2"], per[L]["acc"]
            curve[str(L)][str(n)] = {"force_r2_mean": float(np.mean(r2s)), "force_r2_std": float(np.std(r2s)),
                                     "shape_acc_mean": float(np.mean(accs)), "shape_acc_std": float(np.std(accs)),
                                     "force_r2_per_seed": r2s, "shape_acc_per_seed": accs}
            print("labels", L, "pairs", n, "force R^2", round(np.mean(r2s), 3), "+-", round(np.std(r2s), 3),
                  "acc", round(np.mean(accs), 3), "+-", round(np.std(accs), 3), flush=True)

    report = {
        "device": "cpu",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "threads": 4,
        "seeds": SEEDS,
        "settings": {"grid": base.G, "n_pool_pairs": n_pool, "n_test": n_test,
                     "label_counts": LABEL_COUNTS, "encoder": "same as base (ViT dim 32, depth 2, heads 2, patch 3)",
                     "align_steps": base.ALIGN_STEPS, "batch": base.BATCH, "lr": base.LR,
                     "temperature": base.TAU, "head_steps": base.HEAD_STEPS, "pair_counts": PAIR_COUNTS},
        "pair_count_curve": curve,
        "wall_time_s": round(time.time() - T0, 1),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("wall time", report["wall_time_s"], "s")


if __name__ == "__main__":
    main()
