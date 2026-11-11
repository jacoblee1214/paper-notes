"""Allegro right hand contact field on CPU: box occupancy and object-surface reach.

Uses the public Allegro URDF and mesh files from the lightning-grasp v1.0.0 asset
release (data/assets/). URDF and OBJ/STL are parsed with the standard library and
numpy only. The hand is not executed and no third-party code is imported.

Stage 1: fingertip positions over random joint configurations, voxelized into
         1 cm boxes in the palm frame (box occupancy statistics).
Stage 2: object surface (YCB apple) queried against sampled fingertip surface points
         with a KD-tree; a point counts as reachable when a fingertip point lies within
         r and its normal faces the object within 36 degrees.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import json
import math
import platform
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import scipy
from scipy.spatial import cKDTree

T0 = time.perf_counter()
SEED = 0
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "assets")
HAND_DIR = os.path.join(DATA, "hand", "allegro_right")
URDF = os.path.join(HAND_DIR, "allegro_hand_right.urdf")
APPLE = os.path.join(DATA, "object", "ycb", "013_apple", "nontextured.stl")
OUT = os.path.join(HERE, "extra_contact_field_results.json")

N_SURF = 3000          # surface samples per fingertip link (as in the public code)
N_CONF_BOX = 80000     # configurations for the occupancy stage (paper: 80,000)
M_BOX = 6              # vectors per configuration per tip (80,000 x 6 = 480,000 per tip)
BOX = 0.01             # 1 cm box
CAP = 256              # leaf cap used in the public code
KQ = 4096              # neighbours fetched per object point (saturation is reported)
N_CONF_REACH = 4000
M_REACH = 30
N_OBJ = 8000
COS36 = math.cos(math.radians(36.0))


# ---------------------------------------------------------------- URDF / mesh parsing
def rpy_to_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def origin_T(el):
    T = np.eye(4)
    if el is None:
        return T
    xyz = [float(v) for v in el.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in el.get("rpy", "0 0 0").split()]
    T[:3, :3] = rpy_to_mat(*rpy)
    T[:3, 3] = xyz
    return T


def load_obj(path):
    verts, faces = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                for k in range(1, len(idx) - 1):  # fan triangulation
                    faces.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(verts, float), np.asarray(faces, int)


def orient_outward(V, F):
    """Flip faces if the signed volume is negative (outward normals)."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0
    return F if vol >= 0 else F[:, ::-1]


def sample_surface(V, F, n, rng):
    """Area-weighted surface samples with outward unit normals (mesh frame)."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    area2 = np.linalg.norm(cr, axis=1)
    keep = area2 > 0
    cr, area2, a, b, c = cr[keep], area2[keep], a[keep], b[keep], c[keep]
    p = area2 / area2.sum()
    tri = rng.choice(len(p), size=n, p=p)
    u = rng.random(n)
    v = rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = a[tri] + u[:, None] * (b[tri] - a[tri]) + v[:, None] * (c[tri] - a[tri])
    nrm = cr[tri] / area2[tri][:, None]
    return pts, nrm


def load_stl_binary(path):
    raw = open(path, "rb").read()
    count = int(np.frombuffer(raw[80:84], dtype="<u4")[0])
    if 84 + 50 * count != len(raw):
        raise ValueError("expected binary STL")
    dt = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec = np.frombuffer(raw, dtype=dt, count=count, offset=84)
    V = rec["v"].reshape(-1, 3).astype(float)
    F = np.arange(len(V)).reshape(-1, 3)
    return V, F


# ---------------------------------------------------------------- kinematics
def rot_batch(axis, th):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    s = np.sin(th)[:, None, None]
    c = np.cos(th)[:, None, None]
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    T = np.zeros((len(th), 4, 4))
    T[:, :3, :3] = R
    T[:, 3, 3] = 1
    return T


def build_chain(root_el):
    links = {l.get("name"): l for l in root_el.findall("link")}
    children = {}
    joints = {}
    for j in root_el.findall("joint"):
        parent = j.find("parent").get("link")
        children.setdefault(parent, []).append(j)
        joints[j.get("name")] = j
    return links, children, joints


def fk(K, Q, children, joints, root="base_link"):
    """Batch forward kinematics. Q maps revolute joint name -> (K,) angles."""
    T = {root: np.broadcast_to(np.eye(4), (K, 4, 4))}
    stack = [root]
    while stack:
        lname = stack.pop()
        for j in children.get(lname, []):
            child = j.find("child").get("link")
            To = origin_T(j.find("origin"))
            if j.get("type") == "revolute":
                ax = [float(v) for v in j.find("axis").get("xyz").split()]
                Tj = np.matmul(To, rot_batch(ax, Q[j.get("name")]))
            else:
                Tj = To
            T[child] = np.matmul(T[lname], Tj)
            stack.append(child)
    return T


def tip_surface(links, name, rng):
    """Collision mesh of a tip link, sampled, in that link's frame."""
    col = links[name].find("collision")
    fn = col.find("geometry/mesh").get("filename")
    V, F = load_obj(os.path.join(HAND_DIR, fn))
    F = orient_outward(V, F)
    p, n = sample_surface(V, F, N_SURF, rng)
    To = origin_T(col.find("origin"))
    p = p @ To[:3, :3].T + To[:3, 3]
    n = n @ To[:3, :3].T
    return p, n, {"file": os.path.relpath(os.path.join(HAND_DIR, fn), ROOT), "bytes": os.path.getsize(os.path.join(HAND_DIR, fn)), "verts": len(V), "faces": len(F)}


def sample_q(joints_rev, K, rng):
    Q = {}
    for name, j in joints_rev.items():
        lim = j.find("limit")
        lo, hi = float(lim.get("lower")), float(lim.get("upper"))
        Q[name] = rng.uniform(lo, hi, size=K)
    return Q


def mid_q(joints_rev):
    Q = {}
    for name, j in joints_rev.items():
        lim = j.find("limit")
        Q[name] = np.array([(float(lim.get("lower")) + float(lim.get("upper"))) / 2])
    return Q


# ---------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(SEED)
    root_el = ET.parse(URDF).getroot()
    links, children, joints = build_chain(root_el)
    rev = {n: j for n, j in joints.items() if j.get("type") == "revolute"}
    tips = [n for n in links if n.endswith("_tip") and links[n].find("collision") is not None]
    tips.sort()
    print("revolute joints:", len(rev), "tips:", tips, flush=True)

    # Palm frame (fixed in base frame); all fingertips are expressed in it.
    T_base_palm = fk(1, mid_q(rev), children, joints)["palm"][0]
    T_palm_inv = np.linalg.inv(T_base_palm)

    surf, meshinfo = {}, {}
    for t in tips:
        p, n, info = tip_surface(links, t, rng)
        surf[t] = (p, n)
        meshinfo[t] = info

    def rel_positions(Tl, pts, nrm):
        """Transform (K,4,4) link poses and (K,m,3) local points into the palm frame."""
        Trel = np.matmul(T_palm_inv[None], Tl)
        P = np.einsum("kij,kmj->kmi", Trel[:, :3, :3], pts) + Trel[:, None, :3, 3]
        N = np.einsum("kij,kmj->kmi", Trel[:, :3, :3], nrm) if nrm is not None else None
        return P, N

    # ---- Stage 1: box occupancy
    t1 = time.perf_counter()
    box_stats = {}
    chunk = 10000
    for t in tips:
        p_loc, _ = surf[t]
        counts_parts = []
        for start in range(0, N_CONF_BOX, chunk):
            K = min(chunk, N_CONF_BOX - start)
            Q = sample_q(rev, K, rng)
            T = fk(K, Q, children, joints)
            sel = rng.integers(0, N_SURF, size=(K, M_BOX))
            P, _ = rel_positions(T[t], p_loc[sel], None)
            idx = np.floor(P / BOX).astype(np.int64).reshape(-1, 3)
            counts_parts.append(idx)
        idx_all = np.concatenate(counts_parts, axis=0)
        _, cnt = np.unique(idx_all, axis=0, return_counts=True)
        total = int(cnt.sum())
        over256 = int(np.maximum(cnt - CAP, 0).sum())
        box_stats[t] = {
            "vectors": total,
            "occupied_boxes": int(len(cnt)),
            "median_per_box": float(np.median(cnt)),
            "frac_boxes_gt50": float((cnt > 50).mean()),
            "frac_boxes_gt256": float((cnt > CAP).mean()),
            "frac_vectors_cut_by_256": over256 / total,
        }
        print(t, box_stats[t], flush=True)
    t1_wall = time.perf_counter() - t1

    # Fingertip workspace centroid in palm frame (used as object placement anchor)
    Qm = sample_q(rev, 2000, rng)
    Tm = fk(2000, Qm, children, joints)
    cen = []
    for t in tips:
        sel1 = rng.integers(0, N_SURF, size=(2000, 1))
        P, _ = rel_positions(Tm[t], surf[t][0][sel1], None)
        cen.append(P[:, 0].mean(axis=0))
    anchor = np.mean(cen, axis=0)

    # ---- Stage 2: object reach
    t2 = time.perf_counter()
    OV, OF = load_stl_binary(APPLE)
    ext = OV.max(axis=0) - OV.min(axis=0)
    obj_scale = 1.0 if ext.max() < 1.0 else 0.001
    OV = OV * obj_scale
    OF = orient_outward(OV, OF)
    op, on = sample_surface(OV, OF, N_OBJ, rng)
    ocen = op.mean(axis=0)
    op = op - ocen
    ext_m = (OV.max(axis=0) - OV.min(axis=0)).tolist()

    H_pts, H_nrm, H_lab = [], [], []
    Qr = sample_q(rev, N_CONF_REACH, rng)
    Tr = fk(N_CONF_REACH, Qr, children, joints)
    for fi, t in enumerate(tips):
        p_loc, n_loc = surf[t]
        sel = rng.integers(0, N_SURF, size=(N_CONF_REACH, M_REACH))
        P, N = rel_positions(Tr[t], p_loc[sel], n_loc[sel])
        H_pts.append(P.reshape(-1, 3))
        H_nrm.append(N.reshape(-1, 3))
        H_lab.append(np.full(P.shape[0] * P.shape[1], fi))
    H_pts = np.concatenate(H_pts)
    H_nrm = np.concatenate(H_nrm)
    H_lab = np.concatenate(H_lab)
    tree = cKDTree(H_pts)

    reach = {}
    for off in [-0.02, 0.0, 0.02]:
        center = anchor + np.array([0, 0, off])
        O = op + center
        for r in [0.005, 0.01, 0.02]:
            ok_all, sat_all = [], []
            for s in range(0, len(O), 500):
                dist, idx = tree.query(O[s:s + 500], k=KQ, distance_upper_bound=r)
                valid = np.isfinite(dist)
                safe = np.where(valid, idx, 0)
                ok = valid & (np.einsum("mkj,mj->mk", H_nrm[safe], -on[s:s + 500]) >= COS36)
                fing = np.stack([(ok & (H_lab[safe] == fi)).any(axis=1) for fi in range(len(tips))], axis=1)
                ok_all.append(fing)
                sat_all.append(valid.sum(axis=1) == KQ)
            fing = np.concatenate(ok_all)
            per_finger = {t: float(fing[:, fi].mean()) for fi, t in enumerate(tips)}
            reach[f"offset_{off:+.2f}m_r_{r*100:.1f}cm"] = {
                "reachable_any_finger": float(fing.any(axis=1).mean()),
                "reachable_per_finger": per_finger,
                "saturated_at_k": float(np.concatenate(sat_all).mean()),
            }
            print("offset", off, "r", r, reach[f"offset_{off:+.2f}m_r_{r*100:.1f}cm"]["reachable_any_finger"], flush=True)
    t2_wall = time.perf_counter() - t2

    result = {
        "source": {
            "hand_and_objects": "zhaohengyin/lightning-grasp release v1.0.0, assets.zip (GitHub release, public, no login)",
            "url": "https://github.com/zhaohengyin/lightning-grasp/releases/download/v1.0.0/assets.zip",
            "zip_bytes": 44807401,
            "zip_sha256": "42d1cad23398678b0f64274c3ead57f5f1d9203f9df0cfaa30ee34f3f9d092a6",
            "used_files": [
                "hand/allegro_right/allegro_hand_right.urdf",
                "hand/allegro_right/meshes/collision/link_tip.obj",
                "object/ycb/013_apple/nontextured.stl",
            ],
            "license_note": "repo license NOASSERTION (gh API); assets include YCB objects; check original licenses before redistribution",
        },
        "mesh_info": meshinfo,
        "apple_extent_m": ext_m,
        "apple_scale_applied": obj_scale,
        "config": {
            "seed": SEED,
            "tips": tips,
            "surface_samples_per_tip": N_SURF,
            "configs_box_stage": N_CONF_BOX,
            "vectors_per_config_per_tip": M_BOX,
            "box_m": BOX,
            "leaf_cap": CAP,
            "configs_reach_stage": N_CONF_REACH,
            "tip_points_per_config_reach": M_REACH,
            "object_surface_samples": N_OBJ,
            "normal_threshold_deg": 36,
            "palm_anchor_m": anchor.tolist(),
            "joint_sampling": "uniform within URDF limits; fixed joints ignored",
        },
        "box_occupancy": box_stats,
        "object_reach": reach,
        "hand_points_in_tree": int(len(H_pts)),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "wall_time_s": {
            "box_stage": round(t1_wall, 2),
            "reach_stage": round(t2_wall, 2),
            "total": round(time.perf_counter() - T0, 2),
        },
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print("wrote", OUT, "total", result["wall_time_s"]["total"], "s", flush=True)


if __name__ == "__main__":
    main()
