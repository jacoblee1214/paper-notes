"""PHUMA 공개 로봇 모션(G1, H1-2)의 물리 지표 재계산과 로봇 공간 질량중심 필터 검사.

실행 (분석 폴더 밖에서, 데이터 폴더 밖의 스크립트만 실행):
  .venv/bin/python -I phuma/analysis/extra_public_data_metrics.py \
      phuma/data/zip/data.zip phuma/analysis/extra_public_data_metrics_results.json [--max-clips N]

- 입력: data.zip의 data/<robot>/<source>/*.npy. 압축을 풀지 않고 zipfile로 한 개씩 읽는다.
- npy 안에는 numpy 배열로 감싼 dict가 피클로 들어 있다. np.load(allow_pickle=False)로는 열리지 않는다.
  그래서 헤더만 직접 읽고, 피클 본문은 numpy 재구성 함수 몇 개만 허용하는 Unpickler로 읽는다.
- 출력: 집계값만 쓴다. 모션 원자료는 저장하지 않는다.
- 로봇별로 나눠 돌린 뒤 --merge로 합칠 수 있다 (전체 실행은 로봇당 약 15–18분, 맥 CPU 기준).
- 로봇 모델: phuma/repo/asset/humanoid_model/<robot>/ 의 정적 MJCF와 config.yaml (속도 한계).
"""
import argparse
import hashlib
import io
import json
import os
import pickle
import sys
import time
import zipfile

import mujoco
import numpy as np
import yaml
from scipy.spatial import ConvexHull, Delaunay

ANALYSIS = os.path.dirname(os.path.abspath(__file__))
ASSET = os.path.join(os.path.dirname(ANALYSIS), "repo", "asset", "humanoid_model")
SEED = 20261112

# 지표 명세 값 (분석 문서 03의 계산 명세와 같음). 접지 높이 g = 0 (데이터의 바닥 높이).
CONTACT_BANDS = [0.025, 0.05, 0.10]  # 접촉 판정 대역 (주값 0.025 = 식 (11)의 δ)
MAIN_BAND = 0.025
TOL = 0.01  # 뜸·관통 허용 1 cm
SKATE_V = 0.10  # 수평 발 속도 10 cm/s
JLIM = 0.98  # 관절 한계의 98%
GROUND_DELTA = 0.025  # 접지 높이 다수결 창 폭 (식 (11))

# 2차 로봇 공간 필터 (G1), 리포지토리 기본값과 같은 값
FILTER_CHUNK = 120
FILTER_OVERLAP = 15
FILTER_MIN = 30
# 로봇별 2차 필터 임계 (질량중심 최소 높이 m, 질량중심–BoS 평균 거리 m). 저장소 preprocess_<robot>_folder.py 기본값
FILTER_THRESH = {"g1": (0.40, 0.16), "h1_2": (0.60, 0.24)}

# npy 안의 피클을 읽을 때 허용하는 전역 이름. 이 밖의 모든 전역 참조는 거부한다.
ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.numeric", "_frombuffer"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}
EXPECTED_KEYS = {"root_trans", "root_ori", "dof_pos", "fps"}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global: {module}.{name}")


def load_clip(zf, info):
    raw = zf.read(info)
    f = io.BytesIO(raw)
    version = np.lib.format.read_magic(f)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
    else:
        raise ValueError(f"unsupported npy version {version}")
    if dtype.hasobject is False:
        raise ValueError("expected pickled object array")
    arr = RestrictedUnpickler(f).load()
    if f.read(1):
        raise ValueError("trailing bytes after pickle")
    if not (isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ()):
        raise ValueError("payload is not a 0-d object array")
    d = arr.item()
    if not isinstance(d, dict) or set(d.keys()) != EXPECTED_KEYS:
        raise ValueError(f"unexpected keys {sorted(d.keys()) if isinstance(d, dict) else type(d)}")
    return d


class Robot:
    def __init__(self, name):
        self.name = name
        d = os.path.join(ASSET, name)
        self.model = mujoco.MjModel.from_xml_path(os.path.join(d, "scene.xml"))
        self.data = mujoco.MjData(self.model)
        m = self.model
        with open(os.path.join(d, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        hinge = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        assert all(m.jnt_qposadr[j] == 7 + k for k, j in enumerate(hinge)), "hinge qpos order"
        self.nd = len(hinge)
        self.nq = m.nq
        # 공개 G1 열 순서(29열)에서 손목 6열의 위치. 저장소 MJCF(23 자유도)에는 이 관절이 없다.
        self.zero_cols = [19, 20, 21, 26, 27, 28] if name == "g1" else []
        self.joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinge]
        self.qmin = np.array([m.jnt_range[j][0] for j in hinge])
        self.qmax = np.array([m.jnt_range[j][1] for j in hinge])
        self.limited = np.array([bool(m.jnt_limited[j]) for j in hinge])
        names = [n for n in cfg["joint_names"] if n != "floating_base_joint"]
        vlim = cfg["joint_velocity_limits"]
        assert len(names) == len(vlim), "config velocity list length"
        vmap = dict(zip(names, vlim))
        self.vmax = np.array([float(vmap.get(n, np.nan)) for n in self.joint_names])
        self.vel_mask = ~np.isnan(self.vmax)
        self.foot = {}
        for side in ("left", "right"):
            self.foot[side] = self._foot_points(side)
        self.kp_bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_{p}_keypoint")
                          for s in ("left", "right") for p in ("heel", "toe")]
        assert all(b >= 0 for b in self.kp_bodies), "keypoint bodies"

    def _foot_points(self, side):
        """발 충돌 기하의 표면 점. 구는 중심과 반지름, 메시는 꼭짓점(반지름 0).
        뒤꿈치/발끝 구분은 키포인트 두 개의 x 중점으로 자른다 (몸체 좌표계 기준, 자세와 무관)."""
        m = self.model
        body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body and m.geom_contype[g] > 0]
        heel_x = m.body(f"{side}_heel_keypoint").pos[0]
        toe_x = m.body(f"{side}_toe_keypoint").pos[0]
        x_split = 0.5 * (heel_x + toe_x)
        out = []
        for g in geoms:
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                verts = np.zeros((1, 3))
                rad = float(m.geom_size[g][0])
            elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                adr, num = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                verts = m.mesh_vert[adr:adr + num].copy()
                rad = 0.0
            else:
                raise ValueError(f"unsupported foot geom type {m.geom_type[g]}")
            # 몸체 좌표계 기준으로 뒤꿈치(0) 또는 발끝(1)을 나눈다
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, m.geom_quat[g])
            body_pts = m.geom_pos[g] + verts @ rot.reshape(3, 3).T
            lab = (body_pts[:, 0] >= x_split).astype(int)
            out.append((g, verts, rad, lab))
        return out

    def fk_clip(self, rt, ro, dof, need_com=False):
        """프레임별 발 4영역 최저 높이 H (N,4), 수평 위치 XY (N,4,2).
        영역 순서: 왼발 뒤꿈치, 왼발 발끝, 오른발 뒤꿈치, 오른발 발끝.
        need_com이면 질량중심 (N,3)과 키포인트 몸체 수평 위치 (N,4,2)도 준다."""
        m, d = self.model, self.data
        N = rt.shape[0]
        H = np.empty((N, 4))
        XY = np.empty((N, 4, 2))
        if need_com:
            COM = np.empty((N, 3))
            KP = np.empty((N, 4, 2))
        for t in range(N):
            d.qpos[0:3] = rt[t]
            d.qpos[3:7] = ro[t, [3, 0, 1, 2]]
            d.qpos[7:] = dof[t]
            mujoco.mj_kinematics(m, d)
            k = 0
            for si, side in enumerate(("left", "right")):
                pts_z = []
                pts_xy = []
                labs = []
                for g, verts, rad, lab in self.foot[side]:
                    w = d.geom_xpos[g] + verts @ d.geom_xmat[g].reshape(3, 3).T
                    pts_z.append(w[:, 2] - rad)
                    pts_xy.append(w[:, :2])
                    labs.append(lab)
                z = np.concatenate(pts_z)
                xy = np.concatenate(pts_xy)
                lab = np.concatenate(labs)
                for r in (0, 1):
                    sel = lab == r
                    H[t, k] = z[sel].min()
                    XY[t, k] = xy[sel].mean(axis=0)
                    k += 1
            if need_com:
                mujoco.mj_comPos(m, d)
                COM[t] = d.subtree_com[0]
                for i, b in enumerate(self.kp_bodies):
                    KP[t, i] = d.xpos[b][:2]
        if need_com:
            return H, XY, COM, KP
        return H, XY


def ground_vote(H):
    """식 (11)과 같은 다수결 높이 (영역 최저 값 4N개 기준, 진단용)."""
    v = np.sort(H.ravel())
    lo = np.searchsorted(v, v - GROUND_DELTA, side="left")
    hi = np.searchsorted(v, v + GROUND_DELTA, side="right")
    cnt = hi - lo
    return float(v[int(np.argmax(cnt))])


def metrics_clip(robot, rt, ro, dof, fps, H, XY):
    dt = 1.0 / float(fps)
    out = {"n_frames": int(H.shape[0])}
    V = np.gradient(XY, dt, axis=0)
    speed = np.linalg.norm(V, axis=-1)
    for band in CONTACT_BANDS:
        key = f"{band:.3f}"
        C = H <= band  # 접촉 판정 (g = 0)
        nC = int(C.sum())
        nf = int((C & (H > TOL)).sum())
        npen = int((C & (H < -TOL)).sum())
        nsk = int((C & (speed >= SKATE_V)).sum())
        out[key] = {"n_contact": nC, "float_fail": nf, "pen_fail": npen, "skate_fail": nsk}
    # 관절 한계 (위치는 98% 범위, 속도는 config의 98%)
    lo = JLIM * robot.qmin
    hi = JLIM * robot.qmax
    pos_bad = (robot.limited[None, :] & ((dof < lo) | (dof > hi)))
    Qd = np.gradient(dof, dt, axis=0)
    vel_bad = robot.vel_mask[None, :] & (np.abs(Qd) > JLIM * np.nan_to_num(robot.vmax, nan=np.inf)[None, :])
    pos_ok = ~pos_bad.any(axis=1)
    vel_ok = ~vel_bad.any(axis=1)
    out["joint"] = {"n_frames": int(dof.shape[0]),
                    "pos_ok": int(pos_ok.sum()), "vel_ok": int(vel_ok.sum()),
                    "both_ok": int((pos_ok & vel_ok).sum()),
                    "joints_pos_bad": int(pos_bad.any(axis=0).sum()),
                    "joints_vel_bad": int(vel_bad.any(axis=0).sum())}
    out["ground_vote"] = ground_vote(H)
    # 대안: 클립마다 다수결 바닥 높이를 뺀 값으로 같은 지표를 계산
    Ha = H - out["ground_vote"]
    Ca = Ha <= MAIN_BAND
    out["alt"] = {"n_contact": int(Ca.sum()),
                  "float_fail": int((Ca & (Ha > TOL)).sum()),
                  "pen_fail": int((Ca & (Ha < -TOL)).sum()),
                  "skate_fail": int((Ca & (speed >= SKATE_V)).sum())}
    return out


def point_to_segment(p, a, b):
    ab = b - a
    ap = p - a
    dd = float(np.dot(ab, ab))
    if dd == 0:
        return float(np.linalg.norm(ap))
    t = float(np.dot(ap, ab)) / dd
    if 0.0 <= t <= 1.0:
        return float(np.linalg.norm(p - (a + t * ab)))
    return min(float(np.linalg.norm(p - a)), float(np.linalg.norm(p - b)))


def bos_distance(kp_xy, com_xy):
    """질량중심 수평 투영과 발 키포인트 볼록 껍질의 거리 (프레임별 값의 평균). 리포지토리 정의와 같은 식."""
    dists = []
    errors = 0
    for i in range(kp_xy.shape[0]):
        pts = np.unique(kp_xy[i], axis=0)
        try:
            hull = ConvexHull(pts)
        except Exception:
            errors += 1
            dists.append(np.nan)
            continue
        hv = hull.points[hull.vertices]
        if Delaunay(hv).find_simplex(com_xy[i]) >= 0:
            dists.append(0.0)
        else:
            dists.append(min(point_to_segment(com_xy[i], hv[j], hv[(j + 1) % len(hv)])
                             for j in range(len(hv))))
    dists = np.array(dists)
    return float(np.nanmean(dists)) if np.isfinite(dists).any() else 0.0, errors


def filter_chunks(N):
    """리포지토리 큐레이션 루프와 같은 4초 창(겹침 0.5초, 최소 1초)."""
    starts = []
    start = 0
    while start < N:
        end = start + FILTER_CHUNK
        if end + (FILTER_MIN - FILTER_OVERLAP) > N:
            end = N
        else:
            end = min(end, N)
        if end - start < FILTER_MIN:
            break
        starts.append((start, end))
        start += FILTER_CHUNK - FILTER_OVERLAP
    return starts


def filter_clip(com, kp, fps, com_min_h, bos_max):
    res = []
    for s, e in filter_chunks(com.shape[0]):
        cmin = float(com[s:e, 2].min())
        bos, err = bos_distance(kp[s:e], com[s:e, :2])
        res.append({"n": e - s, "com_min_h": cmin, "bos": bos, "bos_err": err,
                    "fail_h": cmin < com_min_h, "fail_bos": bos > bos_max})
    return res


def pct(x, qs=(5, 25, 50, 75, 95)):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return {f"p{q}": round(float(np.percentile(x, q)), 4) for q in qs} | {"n": int(x.size)}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path", nargs="?", default="")
    ap.add_argument("out_json")
    ap.add_argument("--merge", nargs="+", default=None, help="로봇별로 따로 돌린 결과 json들을 하나로 합친다")
    ap.add_argument("--max-clips", type=int, default=0, help="0이면 전체. 양수면 고정 시드 무작위 표본")
    ap.add_argument("--robots", default="g1,h1_2")
    ap.add_argument("--probe", type=int, default=0, help="앞쪽 N개 클립의 구조만 출력하고 끝냄")
    args = ap.parse_args()
    if args.merge:
        parts = [json.load(open(p, encoding="utf-8")) for p in args.merge]
        res = parts[0]
        for p in parts[1:]:
            assert p["zip_sha256"] == res["zip_sha256"]
            res["robots"].update(p["robots"])
        res["elapsed_s"] = round(sum(p["elapsed_s"] for p in parts), 1)
        res["merged_from_parts"] = len(parts)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print("merged", sorted(res["robots"]))
        return

    t0 = time.time()
    zip_sha = sha256_file(args.zip_path)
    results = {"zip_sha256": zip_sha, "seed": SEED, "max_clips_arg": args.max_clips,
               "contact_bands": CONTACT_BANDS, "main_band": MAIN_BAND, "tol_m": TOL,
               "skate_speed_mps": SKATE_V, "joint_fraction": JLIM, "robots": {}}
    zf = zipfile.ZipFile(args.zip_path)
    members = {}
    for info in zf.infolist():
        if info.is_dir() or not info.filename.endswith(".npy"):
            continue
        parts = info.filename.split("/")
        if len(parts) < 4 or parts[0] != "data":
            continue
        members.setdefault(parts[1], []).append(info)

    if args.probe:
        for robot in ("g1", "h1_2"):
            infos = sorted(members.get(robot, []), key=lambda i: i.filename)[: args.probe]
            for info in infos:
                d = load_clip(zf, info)
                print(info.filename, {k: (tuple(np.shape(v)), str(np.asarray(v).dtype)) if k != "fps" else v
                                      for k, v in d.items()})
        return

    robot_objs = {r: Robot(r) for r in args.robots.split(",")}
    for robot_name, rob in robot_objs.items():
        infos = sorted(members.get(robot_name, []), key=lambda i: i.filename)
        results["robots"][robot_name] = {
            "n_zip_members": len(infos),
            "model_nq": rob.nq, "model_hinge_joints": rob.nd,
            "joint_names": rob.joint_names,
            "joints_without_velocity_limit": [n for n, v in zip(rob.joint_names, rob.vmax) if np.isnan(v)],
        }
        if args.max_clips and args.max_clips < len(infos):
            rng = np.random.default_rng(SEED)
            idx = np.sort(rng.choice(len(infos), size=args.max_clips, replace=False))
            infos = [infos[i] for i in idx]
            results["robots"][robot_name]["sampled"] = True
        results["robots"][robot_name]["n_clips_processed"] = len(infos)

        agg = {"clips": 0, "frames": 0, "errors": {}, "dof_len_hist": {}, "fps_hist": {},
               "clip_frames": []}
        per_clip = {"nf": {}, "np": {}, "sk": {}, "feas": {}, "pos": {}, "vel": {}, "contact_frac": {}}
        for band in CONTACT_BANDS:
            key = f"{band:.3f}"
            agg[key] = {"n_contact": 0, "float_fail": 0, "pen_fail": 0, "skate_fail": 0}
            per_clip["nf_" + key] = []
            per_clip["np_" + key] = []
            per_clip["sk_" + key] = []
        agg["joint"] = {"n_frames": 0, "pos_ok": 0, "vel_ok": 0, "both_ok": 0}
        agg["alt"] = {"n_contact": 0, "float_fail": 0, "pen_fail": 0, "skate_fail": 0}
        agg["joints_pos_bad_clips"] = 0
        agg["joints_vel_bad_clips"] = 0
        agg["ground_vote"] = []
        agg["ground_zero_clip_share"] = 0
        do_filter = robot_name in FILTER_THRESH
        if do_filter:
            filt = {"clips": 0, "chunks": 0, "chunk_frames": 0, "fail_h_chunks": 0, "fail_bos_chunks": 0,
                    "fail_any_chunks": 0, "fail_chunk_frames": 0, "clips_fail_any": 0,
                    "bos_hull_errors": 0, "com_min_h": [], "bos_mean": []}
        ts = time.time()
        groups = {}  # 원본 시퀀스별 (최대 창 번호, 저장된 창 수)
        overlap = {"pairs": 0, "exact": 0}
        prev = None
        for n_done, info in enumerate(infos):
            try:
                d = load_clip(zf, info)
                rt = np.asarray(d["root_trans"], dtype=float)
                ro = np.asarray(d["root_ori"], dtype=float)
                dof = np.asarray(d["dof_pos"], dtype=float)
                fps = int(d["fps"])
                if rt.shape[0] != dof.shape[0] or dof.shape[0] < 3:
                    raise ValueError(f"shape mismatch {rt.shape} {dof.shape}")
                if dof.shape[1] == rob.nd + len(rob.zero_cols):
                    # 공개 G1 자료는 손목 6개 열이 0으로 채워진 29열이다. 저장소 MJCF에는 손목 관절이 없으므로
                    # 그 열이 모두 0인지 확인한 뒤 지운다 (0 관절은 발과 질량중심 위치에 영향이 없다).
                    if np.any(dof[:, rob.zero_cols] != 0.0):
                        agg["nonzero_zero_cols_clips"] = agg.get("nonzero_zero_cols_clips", 0) + 1
                    dof = np.delete(dof, rob.zero_cols, axis=1)
                if dof.shape[1] != rob.nd:
                    raise ValueError(f"dof columns {dof.shape[1]} vs model {rob.nd}")
                agg["dof_len_hist"][str(dof.shape[1])] = agg["dof_len_hist"].get(str(dof.shape[1]), 0) + 1
                agg["fps_hist"][str(fps)] = agg["fps_hist"].get(str(fps), 0) + 1
                if do_filter:
                    H, XY, COM, KP = rob.fk_clip(rt, ro, dof, need_com=True)
                else:
                    H, XY = rob.fk_clip(rt, ro, dof)
                m = metrics_clip(rob, rt, ro, dof, fps, H, XY)
            except Exception as e:  # 손상 또는 형식 불일치 클립은 세고 넘긴다
                key = type(e).__name__
                agg["errors"][key] = agg["errors"].get(key, 0) + 1
                if agg["errors"][key] <= 3:
                    print(f"[{robot_name}] skipped {info.filename}: {key}: {e}", file=sys.stderr)
                continue
            # 파일명 규칙: <시퀀스>_chunk_<번호>.npy. 번호는 4초 창 순서이고, 창은 15프레임 겹친다.
            stem = info.filename.rsplit(".", 1)[0]
            if "_chunk_" in stem:
                base, cid = stem.rsplit("_chunk_", 1)
                cid = int(cid)
            else:
                base, cid = stem, 0
            g = groups.setdefault(base, {"max_id": -1, "n": 0})
            g["max_id"] = max(g["max_id"], cid)
            g["n"] += 1
            if prev is not None and prev[0] == base and cid == prev[1] + 1 and prev[2].shape[0] == FILTER_CHUNK:
                overlap["pairs"] += 1
                ok = np.array_equal(prev[2][-FILTER_OVERLAP:], rt[:FILTER_OVERLAP]) and \
                    np.array_equal(prev[3][-FILTER_OVERLAP:], dof[:FILTER_OVERLAP])
                overlap["exact"] += int(ok)
            prev = (base, cid, rt, dof)
            agg["clips"] += 1
            agg["frames"] += m["n_frames"]
            agg["clip_frames"].append(m["n_frames"])
            agg["ground_vote"].append(m["ground_vote"])
            if abs(m["ground_vote"]) < 0.01:
                agg["ground_zero_clip_share"] += 1
            for k2 in ("n_contact", "float_fail", "pen_fail", "skate_fail"):
                agg["alt"][k2] += m["alt"][k2]
            for band in CONTACT_BANDS:
                key = f"{band:.3f}"
                c = m[key]
                for k2 in ("n_contact", "float_fail", "pen_fail", "skate_fail"):
                    agg[key][k2] += c[k2]
                if c["n_contact"] > 0:
                    per_clip["nf_" + key].append(1 - c["float_fail"] / c["n_contact"])
                    per_clip["np_" + key].append(1 - c["pen_fail"] / c["n_contact"])
                    per_clip["sk_" + key].append(1 - c["skate_fail"] / c["n_contact"])
                if band == MAIN_BAND:
                    per_clip["contact_frac"].setdefault("v", []).append(c["n_contact"] / (4 * m["n_frames"]))
            j = m["joint"]
            for k2 in ("n_frames", "pos_ok", "vel_ok", "both_ok"):
                agg["joint"][k2] += j[k2]
            per_clip["feas"].setdefault("v", []).append(j["both_ok"] / j["n_frames"])
            if j["joints_pos_bad"]:
                agg["joints_pos_bad_clips"] += 1
            if j["joints_vel_bad"]:
                agg["joints_vel_bad_clips"] += 1
            if do_filter:
                filt["clips"] += 1
                chunks = filter_clip(COM, KP, fps, *FILTER_THRESH[robot_name])
                any_fail_clip = False
                for ch in chunks:
                    filt["chunks"] += 1
                    filt["chunk_frames"] += ch["n"]
                    filt["com_min_h"].append(ch["com_min_h"])
                    filt["bos_mean"].append(ch["bos"])
                    filt["bos_hull_errors"] += ch["bos_err"]
                    fh, fb = ch["fail_h"], ch["fail_bos"]
                    filt["fail_h_chunks"] += int(fh)
                    filt["fail_bos_chunks"] += int(fb)
                    if fh or fb:
                        filt["fail_any_chunks"] += 1
                        filt["fail_chunk_frames"] += ch["n"]
                        any_fail_clip = True
                filt["clips_fail_any"] += int(any_fail_clip)
            if (n_done + 1) % 5000 == 0:
                print(f"[{robot_name}] {n_done + 1}/{len(infos)} clips, {time.time() - ts:.0f}s",
                      file=sys.stderr, flush=True)

        clip_frames = agg.pop("clip_frames")
        cf = np.array(clip_frames)
        main_key = f"{MAIN_BAND:.3f}"
        mc = agg[main_key]
        rates = {}
        if mc["n_contact"]:
            rates["non_floating_pct"] = round(100 * (1 - mc["float_fail"] / mc["n_contact"]), 2)
            rates["non_penetration_pct"] = round(100 * (1 - mc["pen_fail"] / mc["n_contact"]), 2)
            rates["non_skating_pct"] = round(100 * (1 - mc["skate_fail"] / mc["n_contact"]), 2)
        ac = agg["alt"]
        rates_alt = {}
        if ac["n_contact"]:
            rates_alt["non_floating_pct"] = round(100 * (1 - ac["float_fail"] / ac["n_contact"]), 2)
            rates_alt["non_penetration_pct"] = round(100 * (1 - ac["pen_fail"] / ac["n_contact"]), 2)
            rates_alt["non_skating_pct"] = round(100 * (1 - ac["skate_fail"] / ac["n_contact"]), 2)
        rates["joint_feasibility_pct"] = round(100 * agg["joint"]["both_ok"] / agg["joint"]["n_frames"], 2)
        rates["joint_pos_only_pct"] = round(100 * agg["joint"]["pos_ok"] / agg["joint"]["n_frames"], 2)
        rates["joint_vel_only_pct"] = round(100 * agg["joint"]["vel_ok"] / agg["joint"]["n_frames"], 2)
        sens = {}
        for band in CONTACT_BANDS:
            key = f"{band:.3f}"
            c = agg[key]
            if c["n_contact"]:
                sens[key] = {
                    "contact_region_frames": c["n_contact"],
                    "non_floating_pct": round(100 * (1 - c["float_fail"] / c["n_contact"]), 2),
                    "non_penetration_pct": round(100 * (1 - c["pen_fail"] / c["n_contact"]), 2),
                    "non_skating_pct": round(100 * (1 - c["skate_fail"] / c["n_contact"]), 2),
                }
        gv = np.array(agg["ground_vote"])
        res_r = results["robots"][robot_name]
        res_r.update({
            "clips_ok": agg["clips"],
            "errors": agg["errors"],
            "fps_hist": agg["fps_hist"],
            "dof_len_hist": agg["dof_len_hist"],
            "frames_total": agg["frames"],
            "clip_frames_pct": pct(cf),
            "pass_rates_main_band": rates,
            "pass_rates_main_band_per_clip_ground": rates_alt,
            "contact_region_frames_per_clip_ground": ac["n_contact"],
            "main_band_m": MAIN_BAND,
            "contact_band_sensitivity": sens,
            "contact_region_frames_main": mc["n_contact"],
            "per_clip_rate_pct_main": {
                "non_floating": pct(100 * np.array(per_clip["nf_" + main_key])),
                "non_penetration": pct(100 * np.array(per_clip["np_" + main_key])),
                "non_skating": pct(100 * np.array(per_clip["sk_" + main_key])),
                "joint_feasibility": pct(100 * np.array(per_clip["feas"]["v"])),
                "contact_share_pct": pct(100 * np.array(per_clip["contact_frac"]["v"])),
            },
            "ground_vote_m": pct(gv, qs=(5, 25, 50, 75, 95)),
            "ground_vote_abs_lt_1cm_clip_share": round(float(np.mean(np.abs(gv) < 0.01)), 4) if gv.size else None,
            "nonzero_wrist_clips": agg.get("nonzero_zero_cols_clips", 0),
            "joints_pos_bad_clips": agg["joints_pos_bad_clips"],
            "joints_vel_bad_clips": agg["joints_vel_bad_clips"],
            "motion_fidelity": "not computed (no human reference in released data)",
            "window_structure": {
                "source_sequences": len(groups),
                "saved_windows": sum(g["n"] for g in groups.values()),
                "window_ids_absent_below_max": sum(g["max_id"] + 1 - g["n"] for g in groups.values()),
                "sequences_with_absent_window": sum(1 for g in groups.values() if g["max_id"] + 1 > g["n"]),
                "consecutive_pairs_checked": overlap["pairs"],
                "consecutive_pairs_overlap_exact": overlap["exact"],
            },
        })
        if do_filter:
            res_r[f"robot_filter_{robot_name}"] = {
                "thresholds_com_min_h_m_bos_m": list(FILTER_THRESH[robot_name]),
                "clips": filt["clips"],
                "chunks_total": filt["chunks"],
                "chunk_frames_total": filt["chunk_frames"],
                "chunks_fail_com_height": filt["fail_h_chunks"],
                "chunks_fail_bos": filt["fail_bos_chunks"],
                "chunks_fail_any": filt["fail_any_chunks"],
                "chunk_frames_in_failing_chunks": filt["fail_chunk_frames"],
                "chunk_frames_in_failing_share_pct": round(100 * filt["fail_chunk_frames"] / max(1, filt["chunk_frames"]), 3),
                "clips_with_any_failing_chunk": filt["clips_fail_any"],
                "clips_with_any_failing_chunk_pct": round(100 * filt["clips_fail_any"] / max(1, filt["clips"]), 3),
                "bos_hull_errors": filt["bos_hull_errors"],
                "chunk_min_com_height_m": pct(filt["com_min_h"]),
                "chunk_mean_bos_distance_m": pct(filt["bos_mean"]),
            }
        print(f"[{robot_name}] done: {agg['clips']} clips, {agg['frames']} frames, "
              f"{time.time() - ts:.0f}s", file=sys.stderr, flush=True)

    results["elapsed_s"] = round(time.time() - t0, 1)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in results.items() if k != "robots"}, ensure_ascii=False)[:800])


if __name__ == "__main__":
    main()
