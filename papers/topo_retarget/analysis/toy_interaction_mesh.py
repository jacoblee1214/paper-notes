"""상호작용 메시(Delaunay + 거리 가중 라플라시안)의 성질을 CPU로 확인하는 두 가지 작은 실험.

입력: 공개 저장소 wuji-technology/toporetarget 의 예제 클립 `data/pen_spin/motions/pen_spin_029.npz`(손 21점, 펜 포즈 231프레임)와
      펜 메시 `data/pen_spin/assets/pen250/mesh.stl`. 저장소 코드는 실행하지도 가져오지도 않고, 데이터 파일만 읽는다.
      이 폴더의 repo/ 는 `git clone https://github.com/wuji-technology/toporetarget repo && git -C repo checkout 039c54022016a882ef8d3a6d34b4578aee7cb51f` 로 받은 것이다.
      repo/ 가 없으면 --clip, --stl 로 같은 두 파일의 경로를 준다.

실험 1 (그래프 통계): 71개 점(손 21 + 펜 표면 50)의 Delaunay 그래프에서 간선 종류별 개수와, 행 정규화 가중치(식 5)의
        손-물체 간선 몫을 κ = 30, 10, 0(균일)에서 센다.
실험 2 (축소 리타게팅): 뼈 길이가 사람 손과 다른 26자유도 골격에 대해, 손 중심 IK(21점 위치 일치)와 상호작용 메시(식 7)를
        같은 최적화기로 풀어 접촉 점의 변위(E_prec 정의)와 뼈 방향 오차(E_align 정의)를 비교한다.
        논문의 식 8 전체(관통 제약, 슬랙)와 Wuji 손 모델은 쓰지 않는다. 결과는 장난감 수준의 경향이다.

실행: python3 analysis/toy_interaction_mesh.py   (numpy, scipy, 약 1분)
쓰는 것: analysis/toy_interaction_mesh_results.json
"""
import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--clip", default=str(ROOT / "repo/data/pen_spin/motions/pen_spin_029.npz"))
ap.add_argument("--stl", default=str(ROOT / "repo/data/pen_spin/assets/pen250/mesh.stl"))
args = ap.parse_args()

CHAINS = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
BONES = [(c[k], c[k + 1]) for c in CHAINS for k in range(4)]
rng = np.random.default_rng(0)


def read_stl(path):
    raw = Path(path).read_bytes()
    n = struct.unpack("<I", raw[80:84])[0]
    tri = np.frombuffer(raw[84:84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return tri["v"].reshape(-1, 3, 3).astype(np.float64)


def sample_surface(tris, count, seed=0):
    r = np.random.default_rng(seed)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    idx = r.choice(len(tris), size=count, p=area / area.sum())
    u, v = r.random(count), r.random(count)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return a[idx] + u[:, None] * (b[idx] - a[idx]) + v[:, None] * (c[idx] - a[idx])


d = np.load(args.clip, allow_pickle=False)
L = d["mediapipe_landmarks_w"][:, 0].astype(np.float64)   # (T,21,3)
P = d["object_pose_w"][:, 0].astype(np.float64)           # (T,7) xyz + wxyz
T = len(L)
tris = read_stl(args.stl)
loc = sample_surface(tris, 50)
ext = (tris.reshape(-1, 3).max(0) - tris.reshape(-1, 3).min(0))


def obj_world(t):
    q = P[t, 3:7]
    rot = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    return loc @ rot.T + P[t, :3]


def lap_matrix(V, kappa):
    tri = Delaunay(V)
    n = len(V)
    adj = [set() for _ in range(n)]
    for s in tri.simplices:
        for i in range(4):
            for j in range(i + 1, 4):
                adj[s[i]].add(s[j]); adj[s[j]].add(s[i])
    W = np.zeros((n, n))
    for i in range(n):
        for j in adj[i]:
            W[i, j] = np.exp(-kappa * np.linalg.norm(V[i] - V[j]))
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    Lm = np.eye(n) - W
    return Lm, adj


# ------------------------------------------------------------------ 실험 1
out = {"clip_frames": T, "pen_extent_m": ext.tolist()}
g = {}
for kappa in (30.0, 10.0, 0.0):
    ee = {"hh": [], "ho": [], "oo": [], "deg_hand": [], "ho_share": [], "hand_with_obj_nbr": [], "far_share": [], "neff": []}
    for t in range(0, T, 5):
        V = np.vstack([L[t], obj_world(t)])
        Lm, adj = lap_matrix(V, kappa)
        W = -(Lm - np.eye(len(V)))
        hh = ho = oo = 0
        for i in range(len(V)):
            for j in adj[i]:
                if i < j:
                    hh += (i < 21 and j < 21); ho += ((i < 21) != (j < 21)); oo += (i >= 21 and j >= 21)
        ee["hh"].append(hh); ee["ho"].append(ho); ee["oo"].append(oo)
        ee["deg_hand"].append(np.mean([len(adj[i]) for i in range(21)]))
        ee["ho_share"].append(float(np.mean(W[:21, 21:].sum(1))))
        ee["hand_with_obj_nbr"].append(float(np.mean(W[:21, 21:].sum(1) > 0)))
        dist = np.linalg.norm(V[:, None] - V[None], axis=2)
        far = ((dist > 0.05) * W).sum() / W.sum()
        ee["far_share"].append(float(far))
        ee["neff"].append(float(np.mean(1.0 / np.maximum((W[:21] ** 2).sum(1), 1e-12))))
    g[f"kappa_{kappa:g}"] = {k: float(np.mean(v)) for k, v in ee.items()}
out["graph_stats_every5th_frame"] = g
# 손 21점이 펜에서 떨어진 거리(손가락 끝-펜 축)
def axis_dist(t):
    q = P[t, 3:7]
    rot = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    ax = rot[:, 2]
    rel = L[t][[4, 8, 12, 16, 20]] - P[t, :3]
    return np.linalg.norm(rel - np.outer(rel @ ax, ax), axis=1)
ad = np.array([axis_dist(t) for t in range(T)])
out["fingertip_to_pen_axis_m"] = {"min": float(ad.min()), "median": float(np.median(ad)), "frac_within_2cm": float((ad < 0.02).mean())}


# ------------------------------------------------------------------ 실험 2
base = L[0]
blen = {b: np.linalg.norm(base[b[1]] - base[b[0]]) for b in BONES}
rest_dir = {b: (base[b[1]] - base[b[0]]) / blen[b] for b in BONES}
mcp_off = {c[1]: base[c[1]] - base[0] for c in CHAINS}


def fk(params, scale, ratios):
    wrist_t = params[:3]
    wrist_R = R.from_rotvec(params[3:6]).as_matrix()
    th = params[6:].reshape(5, 4, 3)
    pts = np.zeros((21, 3))
    pts[0] = wrist_t
    for f, c in enumerate(CHAINS):
        Rc = np.eye(3)
        pts[c[1]] = wrist_t + wrist_R @ (scale * mcp_off[c[1]])
        for k in range(1, 4):
            Rc = Rc @ R.from_rotvec(th[f, k - 1]).as_matrix()
            b = (c[k], c[k + 1])
            pts[c[k + 1]] = pts[c[k]] + wrist_R @ (Rc @ rest_dir[b]) * scale * ratios[b] * blen[b]
        # 첫 뼈(손목-MCP)는 강체
    return pts


LO = np.concatenate([[-2.0] * 3, [-3.1] * 3, np.tile([-1.2, -0.5, -0.5], 20)])
HI = np.concatenate([[2.0] * 3, [3.1] * 3, np.tile([1.2, 0.5, 0.5], 20)])


def solve(res_fn, x0):
    r = least_squares(res_fn, x0, bounds=(LO, HI), method="trf", max_nfev=300, xtol=1e-9, ftol=1e-9)
    return r.x


def bone_dir_prior(pts, src):
    out = []
    for b in BONES:
        dr = pts[b[1]] - pts[b[0]]; ds = src[b[1]] - src[b[0]]
        out.append(dr / np.linalg.norm(dr) - ds / np.linalg.norm(ds))
    return np.concatenate(out)


def run(scale, kappa_im, frames, lam_im, nv_norm):
    r2 = np.random.default_rng(1)
    ratios = {b: float(r2.uniform(0.85, 1.15)) for b in BONES}
    rows = {"keypoint_ik": [], "im": []}
    for t in frames:
        src = L[t]
        V_s = np.vstack([src, obj_world(t)])
        Lm, adj = lap_matrix(V_s, kappa_im)
        target = Lm @ V_s
        nv = len(V_s)
        # 접촉 점: 펜 표면에서 1.5 cm 이내인 손 키포인트
        objpts = V_s[21:]
        dmin = np.min(np.linalg.norm(src[:, None] - objpts[None], axis=2), axis=1)
        contact = np.where(dmin < 0.015)[0]
        contact = contact[contact > 0]
        x0 = np.zeros(6 + 60)
        x0[:3] = src[0]
        # 시작점: 손목 위치를 맞춘 중립 자세
        def r_ik(x):
            pts = fk(x, scale, ratios)
            return np.concatenate([(pts - src).ravel(), 0.01 * x[6:]])
        def r_im(x):
            pts = fk(x, scale, ratios)
            Vr = np.vstack([pts, V_s[21:]])
            e = (Lm @ Vr - target)
            w = lam_im / (nv if nv_norm else 1.0)
            return np.concatenate([np.sqrt(w) * e.ravel(), 0.1 * bone_dir_prior(pts, src) / np.sqrt(20), 0.1 * x[6:], 10.0 * x[:3] - 10.0 * src[0]])
        x_ik = solve(r_ik, x0)
        x_im = solve(r_im, x_ik.copy())
        for name, x in (("keypoint_ik", x_ik), ("im", x_im)):
            pts = fk(x, scale, ratios)
            prec = np.linalg.norm(pts[contact] - src[contact], axis=1)
            ang = []
            for b in BONES:
                if b[1] in contact or b[0] in contact:
                    u = pts[b[1]] - pts[b[0]]; v = src[b[1]] - src[b[0]]
                    ang.append(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))
            tip_err = np.linalg.norm(pts[[4, 8, 12, 16, 20]] - src[[4, 8, 12, 16, 20]], axis=1).mean()
            rows[name].append({"prec_mm": 1000 * float(prec.mean()), "align_deg": float(np.mean(ang)) if ang else float("nan"),
                               "tip_err_mm": 1000 * float(tip_err), "n_contact": int(len(contact))})
    return rows


frames = list(range(10, T, 22))[:10]
t0 = time.time()
sweep = {}
for scale in (1.0, 1.25):
    for kappa in (30.0, 0.0):
        for nv_norm in (True, False):
            key = f"scale{scale:g}_kappa{kappa:g}_nvnorm{int(nv_norm)}"
            rows = run(scale, kappa, frames, 500.0, nv_norm)
            sweep[key] = {m: {k: float(np.nanmean([r[k] for r in rs])) for k in ("prec_mm", "align_deg", "tip_err_mm")} for m, rs in rows.items()}
            sweep[key]["n_frames"] = len(frames)
            sweep[key]["mean_contact_points"] = float(np.mean([r["n_contact"] for r in rows["im"]]))
            print(key, sweep[key], f"{time.time() - t0:.0f}s", flush=True)
out["retarget_toy"] = sweep
out["retarget_toy_runtime_s"] = time.time() - t0
(ROOT / "analysis" / "toy_interaction_mesh_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out["graph_stats_every5th_frame"], indent=1), out["fingertip_to_pen_axis_m"])
