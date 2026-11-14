"""Force-distribution QP on a real five-finger hand model with YCB objects (CPU).

Hand: public Robotis HX5-D20 right-hand description (URDF.xacro expanded by hand, 20 revolute
joints, visual STL meshes), read with the standard library. Objects: YCB meshes (STL).
Contact search: random sampling + hill climbing over joint configurations, geometric contacts
from signed distance to the object surface, facing test on normals.
QP: the same lifted formulation as qp_toy.py (imported from this folder), solved with the
same ADMM settings.

Outputs: null-space dimension d, eps*sqrt(d), realized residual of the projected force,
and recall / false-positive / normal-error of geometric contact estimates under object pose error.

Default search size: 100 random configurations per case, 3 hill-climbing starts.
With --large: 400 random configurations, 6 hill-climbing starts, pose-error tests only for
grasps with an actuated-finger contact, written to extra_real_hand_qp_large_results.json
(the default output is unchanged).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import json
import math
import platform
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial import cKDTree

T0 = time.perf_counter()
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
sys.path.insert(0, str(HERE))
import qp_toy as qt  # noqa: E402  (same folder; grasp map, Jacobian, ADMM, constraints)

URDF = DATA / "robotis" / "hx5_d20_right.urdf.xacro"
MESH_DIR = DATA / "robotis" / "meshes" / "hx5_d20_rev2" / "hx5_d20_right"
OBJECTS = {
    "adjustable_wrench": DATA / "ycb" / "042_adjustable_wrench.stl",
    "phillips_screwdriver": DATA / "ycb" / "043_phillips_screwdriver.stl",
}
LARGE = "--large" in sys.argv
OUT = HERE / ("extra_real_hand_qp_large_results.json" if LARGE else "extra_real_hand_qp_results.json")
SEED = 0
N_PER_LINK = 800
N_OBJ = 20000
N_INIT = 400 if LARGE else 100          # random joint configurations per case
N_HILL = 6 if LARGE else 3              # hill-climbing starts
STEPS = 60
D_CONTACT = 0.003      # physical contact band (m)
D_THRESH = 0.01        # estimate rule: signed distance < d_thresh (paper-style)
K_PER_LINK = 6         # contacts kept per link for the QP
COS_FACE = 0.5
EPS = qt.EPS


# ---------------------------------------------------------------- meshes
def load_stl_binary(path):
    raw = path.read_bytes()
    count = int(np.frombuffer(raw[80:84], dtype="<u4")[0])
    if 84 + 50 * count != len(raw):
        raise ValueError("expected binary STL")
    dt = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec = np.frombuffer(raw, dtype=dt, count=count, offset=84)
    V = rec["v"].reshape(-1, 3).astype(float)
    F = np.arange(len(V)).reshape(-1, 3)
    return V, F


def orient_outward(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0
    return F if vol >= 0 else F[:, ::-1]


def sample_surface(V, F, n, rng):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    area2 = np.linalg.norm(cr, axis=1)
    keep = area2 > 0
    cr, area2, a, b, c = cr[keep], area2[keep], a[keep], b[keep], c[keep]
    p = area2 / area2.sum()
    tri = rng.choice(len(p), size=n, p=p)
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = a[tri] + u[:, None] * (b[tri] - a[tri]) + v[:, None] * (c[tri] - a[tri])
    nrm = cr[tri] / area2[tri][:, None]
    return pts, nrm


# ---------------------------------------------------------------- URDF
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
    T[:3, :3] = rpy_to_mat(*[float(v) for v in el.get("rpy", "0 0 0").split()])
    T[:3, 3] = [float(v) for v in el.get("xyz", "0 0 0").split()]
    return T


def rot4(ax, th):
    a = np.asarray(ax, float) / np.linalg.norm(ax)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)
    return T


def load_robot():
    txt = URDF.read_text(encoding="utf-8")
    txt = re.sub(r"<\?xml[^>]*\?>", "", txt)
    txt = re.sub(r"<xacro:property[^>]*/>", "", txt)
    txt = re.sub(r"<xacro:macro[^>]*>", "", txt).replace("</xacro:macro>", "")
    txt = txt.replace("${meshes_file_direction}", "MESHDIR").replace("${prefix}", "")
    root = ET.fromstring(txt)

    joints, children, rev_names = {}, {}, []
    for j in root.findall("joint"):
        name = j.get("name")
        jt = j.get("type")
        rec = dict(
            type=jt,
            parent=j.find("parent").get("link"),
            child=j.find("child").get("link"),
            origin=origin_T(j.find("origin")),
        )
        if jt == "revolute":
            rec["axis"] = np.array([float(v) for v in j.find("axis").get("xyz").split()])
            lim = j.find("limit")
            rec["lo"], rec["hi"] = float(lim.get("lower")), float(lim.get("upper"))
            rec["idx"] = len(rev_names)
            rev_names.append(name)
        joints[name] = rec
        children.setdefault(rec["parent"], []).append(name)

    base = None
    childset = {r["child"] for r in joints.values()}
    for r in joints.values():
        if r["parent"] not in childset:
            base = r["parent"]
            break

    links = {}
    for l in root.findall("link"):
        vis = l.find("visual")
        if vis is None or vis.find("geometry/mesh") is None:
            continue
        m = vis.find("geometry/mesh")
        fn = m.get("filename").split("/")[-1]
        scale = float(m.get("scale", "1 1 1").split()[0])
        links[l.get("name")] = dict(file=MESH_DIR / fn, scale=scale, origin=origin_T(vis.find("origin")))
    return joints, children, rev_names, base, links


def fk(q, joints, children, base):
    T = {base: np.eye(4)}
    anc = {base: []}
    stack = [base]
    while stack:
        l = stack.pop()
        for jn in children.get(l, []):
            j = joints[jn]
            Tpre = T[l] @ j["origin"]
            if j["type"] == "revolute":
                i = j["idx"]
                anc[j["child"]] = anc[l] + [(i, Tpre[:3, 3].copy(), Tpre[:3, :3] @ j["axis"])]
                Tc = Tpre @ rot4(j["axis"], q[i])
            else:
                anc[j["child"]] = anc[l]
                Tc = Tpre
            T[j["child"]] = Tc
            stack.append(j["child"])
    return T, anc


# ---------------------------------------------------------------- scene
def hand_points(T, LINKS, LOCAL):
    P, N, L = [], [], []
    for li, name in enumerate(LINKS):
        p, n = LOCAL[name]
        Tl = T[name]
        P.append(p @ Tl[:3, :3].T + Tl[:3, 3])
        N.append(n @ Tl[:3, :3].T)
        L.append(np.full(len(p), li))
    return np.concatenate(P), np.concatenate(N), np.concatenate(L)


def contacts(Hp, Hn, tree, Op, On):
    dist, idx = tree.query(Hp)
    sgn = np.sign(np.einsum("ij,ij->i", Hp - Op[idx], On[idx]))
    sgn[sgn == 0] = 1
    sd = sgn * dist
    facing = np.einsum("ij,ij->i", Hn, On[idx]) <= -COS_FACE
    return sd, facing, idx


def main():
    rng = np.random.default_rng(SEED)
    joints, children, rev_names, base, links_raw = load_robot()
    LINKS = sorted(links_raw)
    LOCAL, ACT, FINGER = {}, {}, {}
    for name in LINKS:
        rec = links_raw[name]
        V, F = load_stl_binary(rec["file"])
        V = V * rec["scale"]
        F = orient_outward(V, F)
        p, n = sample_surface(V, F, N_PER_LINK, rng)
        R, t = rec["origin"][:3, :3], rec["origin"][:3, 3]
        LOCAL[name] = (p @ R.T + t, n @ R.T)
    q_mid = np.array([(joints[n]["lo"] + joints[n]["hi"]) / 2 for n in rev_names])
    lo = np.array([joints[n]["lo"] for n in rev_names])
    hi = np.array([joints[n]["hi"] for n in rev_names])
    T_mid, anc_mid = fk(q_mid, joints, children, base)
    for name in LINKS:
        a = anc_mid.get(name, [])
        ACT[name] = len(a) > 0
        FINGER[name] = a[0][0] if a else -1
    # group finger links by their first revolute joint (palm-attached links are not actuated)
    print("links:", len(LINKS), "revolute:", len(rev_names), "base:", base, flush=True)
    print("finger groups:", sorted({FINGER[n] for n in LINKS}), flush=True)
    ACT_L = np.array([ACT[n] for n in LINKS])
    FING_L = np.array([FINGER[n] for n in LINKS])

    # anchor: mean of the distal links (tips) at the mid configuration
    tip_names = [n for n in LINKS if "tip" in links_raw[n]["file"].name]
    anchor = np.mean(np.concatenate([LOCAL[n][0] for n in tip_names]), axis=0)

    cases, all_pose, all_qp = [], [], []
    for oname, opath in OBJECTS.items():
        OV, OF = load_stl_binary(opath)
        OF = orient_outward(OV, OF)
        op, on = sample_surface(OV, OF, N_OBJ, rng)
        cen = op.mean(axis=0)
        op = op - cen
        for oriname, Rrot in [("identity", np.eye(3)), ("rot_x90", rot4([1, 0, 0], math.pi / 2)[:3, :3])]:
            Op = op @ Rrot.T + anchor
            On = on @ Rrot.T
            tree = cKDTree(Op)
            case = dict(object=oname, orientation=oriname, Op=Op, On=On, tree=tree, Rrot=Rrot, cen=cen)
            cases.append(case)

    def score(q):
        T, _ = fk(q, joints, children, base)
        Hp, Hn, Hl = hand_points(T, LINKS, LOCAL)
        sd, fac, _ = contacts(Hp, Hn, case["tree"], case["Op"], case["On"])
        cont = fac & (np.abs(sd) <= D_CONTACT) & ACT_L[Hl]
        groups = len(set(FING_L[Hl[cont]].tolist()))
        pen = float(np.mean(sd < -D_CONTACT))
        # closeness of each actuated finger to the object (continuous guidance for the search)
        close = 0.0
        for gid in set(FING_L[ACT_L].tolist()):
            m = ACT_L[Hl] & (FING_L[Hl] == gid)
            close += max(0.0, 0.03 - float(np.min(np.abs(sd[m]))))
        return groups + close * 100.0, int(cont.sum()), pen

    summaries = []
    for case in cases:
        t1 = time.perf_counter()
        valid = []
        init = rng.uniform(lo, hi, size=(N_INIT, len(rev_names)))
        for q in init:
            key, nc, pen = score(q)
            if pen < 0.01:
                valid.append((key, q, key, nc))
        valid.sort(key=lambda x: -x[0])
        starts = valid[:N_HILL]
        found = []
        for key, q0, _, _ in starts:
            q, k = q0.copy(), key
            for _ in range(STEPS):
                cq = np.clip(q + rng.normal(0, 0.08, size=q.shape), lo, hi)
                ck, nc, pen = score(cq)
                if pen < 0.01 and ck >= k:
                    q, k = cq, ck
            key, nc, pen = score(q)
            found.append((k, q, key, nc, pen))
        found.sort(key=lambda x: -x[0])
        grasps, used = [], []
        for k, q, g, nc, pen in found:
            if any(np.linalg.norm(q - u) < 0.15 for u in used):
                continue
            used.append(q)
            grasps.append((k, q, g, nc, pen))
        case["grasps"] = grasps
        summaries.append(dict(
            object=case["object"], orientation=case["orientation"],
            valid_random_configs=len(valid), search_wall_s=round(time.perf_counter() - t1, 2),
            grasps_found=len(grasps),
            grasp_search_key=[round(float(g), 3) for _, _, g, _, _ in grasps], grasp_contacts=[nc for _, _, _, nc, _ in grasps],
        ))
        print(summaries[-1], flush=True)

    # QP and pose-error tests on each grasp found
    def grasp_qp(q, case):
        T, anc = fk(q, joints, children, base)
        Hp, Hn, Hl = hand_points(T, LINKS, LOCAL)
        sd, fac, idx = contacts(Hp, Hn, case["tree"], case["Op"], case["On"])
        cont = np.where(fac & (np.abs(sd) <= D_CONTACT))[0]
        keep = []
        for li in np.unique(Hl[cont]):
            c = cont[Hl[cont] == li]
            keep.extend(rng.choice(c, size=min(K_PER_LINK, len(c)), replace=False).tolist())
        keep = np.array(sorted(keep), dtype=int)
        if len(keep) == 0 or not any(ACT[LINKS[Hl[k]]] for k in keep):
            return None
        pts = Hp[keep]
        nrm = -case["On"][idx[keep]]
        J = np.zeros((len(keep), 3, len(rev_names)))
        act = np.zeros(len(keep), dtype=bool)
        for r, k in enumerate(keep):
            name = LINKS[Hl[k]]
            for (i, o, a) in anc.get(name, []):
                J[r, :, i] = np.cross(a, pts[r] - o)
            act[r] = ACT[name]
        G = qt.grasp_map(pts, nrm)
        Jc = qt.contact_jac(J, nrm)
        P, qv, A, l, u, act_rows, Ja, rank, nd = qt.build_qp(G, Jc, act)
        x, viol = qt.admm(P, qv, A, l, u)
        f = x[: 3 * len(pts)]
        s = qt.summarize(G, f, act_rows, Ja)
        s.update(
            n_contacts=int(len(pts)), n_actuated=int(act.sum()), rank_Jact=int(rank),
            null_dim_d=int(nd), eps_sqrt_d_N=round(EPS * math.sqrt(nd), 4), admm_violation=round(viol, 6),
            fingers_with_contact=int(len(set(FING_L[Hl[keep]].tolist()) - {-1})),
        )
        return s

    def pose_error(q, case):
        T, _ = fk(q, joints, children, base)
        Hp, Hn, Hl = hand_points(T, LINKS, LOCAL)
        sd0, fac0, idx0 = contacts(Hp, Hn, case["tree"], case["Op"], case["On"])
        true_set = fac0 & (np.abs(sd0) <= D_CONTACT)
        far = sd0 > 0.005
        levels = [(0, 0.0), (10, 0.001), (20, 0.003), (30, 0.005)]
        out = {}
        for deg, mm in levels:
            rec, fp, nerr = [], [], []
            for _ in range(30):
                ax = rng.normal(size=3)
                ax /= np.linalg.norm(ax)
                R = rot4(ax, math.radians(deg))[:3, :3]
                dvec = rng.normal(size=3)
                dvec = dvec / np.linalg.norm(dvec) * mm
                Op2 = case["Op"] @ R.T + dvec
                On2 = case["On"] @ R.T
                tree2 = cKDTree(Op2)
                sd, _, idx2 = contacts(Hp, Hn, tree2, Op2, On2)
                est = sd < D_THRESH
                if true_set.sum() > 0:
                    rec.append(float(est[true_set].mean()))
                fp.append(float(est[far].mean()))
                both = true_set & est
                if both.any():
                    cosang = np.clip(np.einsum("ij,ij->i", case["On"][idx0[both]], On2[idx2[both]]), -1, 1)
                    nerr.append(np.degrees(np.arccos(cosang)))
            nerr = np.concatenate(nerr) if nerr else np.array([np.nan])
            out[f"{deg}deg_{mm}mm"] = dict(
                recall_mean=round(float(np.mean(rec)), 3), recall_min=round(float(np.min(rec)), 3),
                false_positive_rate=round(float(np.mean(fp)), 3),
                normal_err_median_deg=round(float(np.nanmedian(nerr)), 2),
                normal_err_p90_deg=round(float(np.nanpercentile(nerr, 90)), 2),
            )
        return dict(true_contacts=int(true_set.sum()), levels=out)

    for case in cases:
        for k, q, g, nc, pen in case["grasps"]:
            qp_res = grasp_qp(q, case)
            if LARGE and qp_res is None:
                # no actuated-finger contact: the pose-error contact set would only hold non-actuated links
                continue
            pe = pose_error(q, case)
            all_qp.append(dict(object=case["object"], orientation=case["orientation"],
                               search_key=round(float(k), 3), search_fingers_term=round(float(g), 3), q=[round(float(v), 4) for v in q],
                               penetrating_fraction=round(pen, 5), qp=qp_res))
            all_pose.append(dict(object=case["object"], orientation=case["orientation"], pose_error=pe))
            print("QP", case["object"], case["orientation"], qp_res and {kk: qp_res[kk] for kk in ("n_contacts", "null_dim_d", "eps_sqrt_d_N", "realized_residual_force_N")}, flush=True)

    qp_ok = [r["qp"] for r in all_qp if r["qp"] is not None]
    agg = {}
    if qp_ok:
        for key in ("n_contacts", "n_actuated", "null_dim_d", "eps_sqrt_d_N", "realized_residual_force_N",
                    "realized_residual_torque_Nm", "invisible_fraction_of_actuated_force", "sum_fn_N", "max_fn_N"):
            vals = [float(r[key]) for r in qp_ok]
            agg[key] = dict(min=round(min(vals), 4), median=round(float(np.median(vals)), 4), max=round(max(vals), 4))
    pe_levels = {}
    for lv in all_pose[0]["pose_error"]["levels"] if all_pose else []:
        rows = [p["pose_error"]["levels"][lv] for p in all_pose]
        pe_levels[lv] = {
            "recall_mean_over_grasps_median": round(float(np.median([r["recall_mean"] for r in rows])), 3),
            "false_positive_rate_median": round(float(np.median([r["false_positive_rate"] for r in rows])), 3),
            "normal_err_median_deg_median": round(float(np.median([r["normal_err_median_deg"] for r in rows])), 2),
        }

    result = {
        "source": {
            "hand": "ROBOTIS-GIT/robotis_hand @ 7730b76 robotis_hand_description (hx5_d20_rev2 right): URDF.xacro expanded by hand (25 joints, 20 revolute), visual STL",
            "hand_url_base": "https://raw.githubusercontent.com/ROBOTIS-GIT/robotis_hand/7730b76bbac51786b122023a7ab2afc17afc7d12/robotis_hand_description/",
            "hand_license": "Apache-2.0 (repository)",
            "hand_files": {str(p.relative_to(HERE.parent)): p.stat().st_size for p in sorted(MESH_DIR.glob("*.stl"))},
            "hand_urdf_bytes": URDF.stat().st_size,
            "objects": {k: {"file": str(v.relative_to(HERE.parent)), "bytes": v.stat().st_size} for k, v in OBJECTS.items()},
            "objects_source": "YCB nontextured.stl, copied from the lightning-grasp v1.0.0 release assets.zip (already downloaded in the sibling folder)",
        },
        "config": {
            "seed": SEED, "n_init_random": N_INIT, "hill_starts": N_HILL, "hill_steps": STEPS,
            "contact_band_m": D_CONTACT, "estimate_threshold_m": D_THRESH, "contacts_per_link": K_PER_LINK,
            "facing_cos": COS_FACE, "points_per_link": N_PER_LINK, "object_surface_samples": N_OBJ,
            "mu": qt.MU, "fmin_N": qt.FMIN, "tau_max": qt.TAU_MAX, "eps_N": EPS,
            "anchor_m": [round(float(v), 4) for v in anchor],
        },
        "search": summaries,
        "qp_per_grasp": all_qp,
        "qp_aggregate": agg,
        "pose_error_per_grasp": all_pose,
        "pose_error_aggregate": pe_levels,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "scipy": scipy.__version__, "platform": platform.platform()},
        "wall_time_s": round(time.perf_counter() - T0, 2),
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT.name, result["wall_time_s"], "s", flush=True)


if __name__ == "__main__":
    main()
