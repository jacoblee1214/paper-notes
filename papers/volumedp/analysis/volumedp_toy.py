"""관측 표현 비교 토이 (numpy 단일 스레드, CPU).

가정(단순화):
- 목표 물체: 한 변 0.1의 정육면체. 위치 (x, y, z) ~ U[0,1]^3. 정책이 맞혀야 할 값은 z(깊이, 높이).
- 카메라: 위(+z)에서 내려다보는 직교 카메라. 같은 xy 위에 있는 물체의 윗면만 보입니다.
- 가림(occlusion): 확률 p_occ 로 목표 위에 가림판(xy 가 목표와 겹치는 판)을 둡니다. 가림판 높이는 무작위입니다.
- 관측 표현 3종:
  (A) 2D 투영: xy 8x8 점유 격자 (깊이 정보 없음, 가림판도 같은 격자에 찍힘)
  (B) 복셀 점유: 8x8x8, 카메라에서 처음 보이는 표면만 기록 (깊이 있음)
  (C) 점 집합: 보이는 표면에서 32점 샘플 후 평균·최솟값·최댓값 요약 (깊이 있음, 순서 불변)
- 정책 근사: 선형 최소제곱(bias 포함)과 kNN(k=5)으로 z 예측. 기준선은 평균 예측.
- 측정: 테스트 RMSE. 가림이 없는 경우와 있는 경우를 나눠 보고합니다.

이 토이는 "깊이가 관측에 들어가면 z 예측이 좋아진다"는 방향성만 보여 줍니다.
실제 로봇 정책 성능이나 VolumeDP 의 우위를 증명하지 않습니다.
"""
import json
import os
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(0)
N_TRAIN, N_TEST = 1500, 500
SIZE = 0.1
GRID = 8
N_POINTS = 32


def sample_scene(n, p_occ):
    tgt = RNG.uniform(0, 1, size=(n, 3))
    occ = RNG.uniform(size=n) < p_occ
    occ_top = RNG.uniform(0.4, 1.0, size=n)  # 가림판의 높이(위에서 본 판의 z)
    return tgt, occ, occ_top


def visible_surface_points(tgt, occ, occ_top, n_pts=N_POINTS):
    """목표 윗면 위의 점들. 가림판이 있으면 가림판 윗면이 먼저 보이므로 목표 점은 가려집니다."""
    n = tgt.shape[0]
    u = RNG.uniform(-SIZE / 2, SIZE / 2, size=(n, n_pts, 2))
    pts = np.empty((n, n_pts, 3))
    pts[..., 0] = tgt[:, None, 0] + u[..., 0]
    pts[..., 1] = tgt[:, None, 1] + u[..., 1]
    pts[..., 2] = tgt[:, None, 2] + SIZE / 2
    hidden = occ[:, None] & (occ_top[:, None] > tgt[:, None, 2] + SIZE / 2)
    return pts, hidden


def feat_2d(tgt, occ, occ_top):
    n = tgt.shape[0]
    grid = np.zeros((n, GRID, GRID))
    ix = np.clip((tgt[:, 0] * GRID).astype(int), 0, GRID - 1)
    iy = np.clip((tgt[:, 1] * GRID).astype(int), 0, GRID - 1)
    grid[np.arange(n), ix, iy] = 1.0
    # 가림판도 xy 에 찍힙니다(같은 격자, 깊이 없음)
    grid[np.arange(n)[occ], ix[occ], iy[occ]] = 1.0
    return grid.reshape(n, -1)


def feat_voxel(tgt, occ, occ_top):
    """카메라에서 처음 보이는 표면의 복셀만 1. 가려진 목표 윗면은 0."""
    n = tgt.shape[0]
    vox = np.zeros((n, GRID, GRID, GRID))
    pts, hidden = visible_surface_points(tgt, occ, occ_top)
    vis = ~hidden.any(axis=1)  # 점이 하나라도 가려지면 목표 기여는 0 (단순화)
    ix = np.clip((tgt[:, 0] * GRID).astype(int), 0, GRID - 1)
    iy = np.clip((tgt[:, 1] * GRID).astype(int), 0, GRID - 1)
    iz = np.clip((tgt[:, 2] * GRID).astype(int), 0, GRID - 1)
    idx = np.arange(n)
    vox[idx[vis], ix[vis], iy[vis], iz[vis]] = 1.0
    # 가림판의 윗면(처음 보이는 면)을 복셀로 기록
    o = idx[occ]
    oz = np.clip((occ_top[occ] * GRID).astype(int), 0, GRID - 1)
    vox[o, ix[occ], iy[occ], oz] = 1.0
    return vox.reshape(n, -1)


def feat_points(tgt, occ, occ_top):
    pts, hidden = visible_surface_points(tgt, occ, occ_top)
    keep = ~hidden  # (n, 32)
    n = tgt.shape[0]
    z = pts[..., 2]
    big = 10.0
    zmin = np.where(keep, z, big).min(axis=1)
    zmax = np.where(keep, z, -big).max(axis=1)
    cnt = keep.sum(axis=1).clip(min=1)
    zmean = np.where(keep, z, 0).sum(axis=1) / cnt
    xy = np.where(keep[..., None], pts[..., :2], 0).sum(axis=1) / cnt[:, None]
    has = keep.any(axis=1)
    zmin = np.where(has, zmin, 0.0)
    zmax = np.where(has, zmax, 0.0)
    zmean = np.where(has, zmean, 0.0)
    occ_z = np.where(occ, occ_top, 0.0)  # 가림판 존재 신호
    return np.column_stack([xy, zmin, zmax, zmean, has.astype(float), occ.astype(float), occ_z]).reshape(n, -1)


def fit_linear(X, y, Xt):
    A = np.column_stack([X, np.ones(len(X))])
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return np.column_stack([Xt, np.ones(len(Xt))]) @ w


def knn(X, y, Xt, k=5):
    d = ((Xt[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    idx = np.argpartition(d, k, axis=1)[:, :k]
    return y[idx].mean(axis=1)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run(p_occ):
    tr = sample_scene(N_TRAIN, p_occ)
    te = sample_scene(N_TEST, p_occ)
    res = {}
    ytr, yte = tr[0][:, 2], te[0][:, 2]
    res["mean_baseline"] = rmse(np.full_like(yte, ytr.mean()), yte)
    for name, f in [("2d_projection", feat_2d), ("voxel_occupancy", feat_voxel), ("point_set_summary", feat_points)]:
        Xtr, Xte = f(*tr), f(*te)
        res[f"{name}_linear"] = rmse(fit_linear(Xtr, ytr, Xte), yte)
        res[f"{name}_knn5"] = rmse(knn(Xtr, ytr, Xte), yte)
    return {k: round(v, 4) for k, v in res.items()}


t0 = time.time()
results = {"no_occlusion_p0": run(0.0), "occlusion_p0.5": run(0.5), "occlusion_p1.0": run(1.0)}
results["config"] = {"n_train": N_TRAIN, "n_test": N_TEST, "grid": GRID, "points": N_POINTS, "seed": 0}
results["runtime_s"] = round(time.time() - t0, 2)
with open(os.path.join(HERE, "volumedp_toy_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(json.dumps(results, ensure_ascii=False, indent=2))
