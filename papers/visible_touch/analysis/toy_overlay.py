"""Two CPU toys (numpy only).
Toy A: how much of a 3-axis contact force survives projection into a 2D image overlay
       (arrow tail/tip with pinhole projection, quantized to pixels), vs. a 3-value vector channel.
Toy B: mechanical correlation between Delta = ACRO - plain and headroom = 1 - plain when
       ACRO is independent of plain (shared term), i.e. the r=0.924 pattern can arise from noise.
"""
import json, os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "toy_overlay_results.json")
rng = np.random.default_rng(0)
R = {}

# ---------- Toy A ----------
def project(X, f=300.0, c=64.0):
    # X: (n,3) camera-frame points, Z>0 in front of camera
    return np.stack([f * X[:, 0] / X[:, 2] + c, f * X[:, 1] / X[:, 2] + c], axis=1)

def arrow_features(F, cam_dir_deg, lam=0.02, sigma_pos=None, depth=0.25, quant=True):
    """Arrow from a fixed sensor anchor (0,0,depth) in camera frame along force F (N, m-scaled by lam).
    Camera looks along +Z rotated by cam_dir_deg about the X axis (oblique viewing)."""
    th = np.deg2rad(cam_dir_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    anchor = np.array([0.0, 0.0, depth]) @ Rx.T
    tip_3d = anchor + (lam * F) @ Rx.T
    tail = project(anchor[None, :])[0]
    tip = project(tip_3d[None, :])[0]
    v = tip - tail
    if quant:
        v = np.round(v)  # 1-pixel quantization on a 128px image
    return v  # 2D arrow in pixels

def fit_r2(X, y):
    Xb = np.c_[X, np.ones(len(X))]
    coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    pred = Xb @ coef
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()

n = 4000
F = rng.normal(size=(n, 3)) * np.array([1.0, 1.0, 1.0])  # normal force along Z of sensor in general
results_A = {}
for cam in [0, 30, 60, 80]:
    V = np.array([arrow_features(f, cam) for f in F])
    # For each force component: how well is it linearly recoverable from the 2D overlay arrow vector?
    r2 = [float(fit_r2(V, F[:, k:k+1])) for k in range(3)]
    # Vector channel: raw 3-value readout is exact (R2 = 1) by construction
    results_A[f"cam_tilt_{cam}deg"] = {
        "overlay_2d_R2_per_axis_Fx_Fy_Fz": [round(x, 3) for x in r2],
        "overlay_pixel_arrow_len_median": round(float(np.median(np.linalg.norm(V, axis=1))), 2),
        "frac_arrows_under_2px": round(float(np.mean(np.linalg.norm(V, axis=1) < 2)), 3),
    }
R["toy_A_projection_info"] = results_A
R["toy_A_note"] = ("Linear readout from the 2D arrow vector (2 numbers per element) cannot recover all 3 force "
                   "components; the depth-direction (normal) component is only partly visible. A separate 3-value "
                   "vector channel recovers all 3 exactly. Pixel quantization and arrow length lam=0.02 m/N are assumed.")

# ---------- Toy B ----------
def toy_b(sd_acro, n_tasks=40, trials=2000):
    rs = []
    for _ in range(trials):
        plain = rng.uniform(0.05, 0.95, size=n_tasks)           # vision-only success
        acro = np.clip(0.92 + sd_acro * rng.normal(size=n_tasks), 0, 1)  # ACRO independent of plain
        delta = acro - plain
        head = 1 - plain
        rs.append(np.corrcoef(delta, head)[0, 1])
    rs = np.array(rs)
    return float(np.median(rs)), float(np.quantile(rs, 0.05)), float(np.quantile(rs, 0.95))

sweep = {}
for sd in [0.02, 0.05, 0.10, 0.15]:
    med, lo, hi = toy_b(sd)
    sweep[f"acro_sd_{sd}"] = {"median_r": round(med, 3), "q05": round(lo, 3), "q95": round(hi, 3)}
R["toy_B_mechanical_headroom_corr"] = {
    "setting": "40 tasks, plain ~ U(0.05,0.95), ACRO ~ N(0.92, sd) independent of plain (no true effect of plain on ACRO); 2000 reps",
    "sweep": sweep,
    "paper_reported_r": 0.924,
    "note": "Delta and headroom share the plain-success term, so r is inflated by construction. The paper's r=0.924 is in the range this artifact produces, so it cannot be read as evidence that overlays help most where vision fails.",
}
np.random.seed(0)
with open(OUT, "w") as f:
    json.dump(R, f, indent=2, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))
