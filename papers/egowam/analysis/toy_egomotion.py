"""토이: 머리 운동이 있는 일인칭 영상에서 3D 흐름 목표가 얼마나 오염되는가 (numpy, 약 1초).

논문 §4.2 / 부록 A.5, B.2의 주장을 작은 기하 모델로 확인한다.
  - 정지한 배경 점은 카메라 안정화 없이는 머리 회전 때문에 큰 "흐름"을 갖는다.
  - 안정화에 쓰는 카메라 포즈가 정확(VIO)하면 배경 흐름은 거의 0이다.
  - RGB만으로 추정한 포즈(A.5: 1초 상대 회전 오차 1.9도, 상대 이동 오차 1.8 cm)를 쓰면
    사람 이동 임계값(10 mm)을 넘는 배경 앵커가 얼마나 남는가.

장면 가정(논문에 없음, 이 토이의 가정): 앵커 28x40 격자(1120점), 평면 장면 깊이 0.5 m(±0.1 m 잡음),
수평 시야각 110도(Aria RGB 카메라의 공칭 값으로 가정), 앵커의 8%가 실제로 5 cm 움직임(손과 물체),
머리는 1초 동안 4.7도 회전, 2.5 cm 이동(A.5의 평균). 임계값 10 mm (B.2, 사람).
실행: python3 analysis/toy_egomotion.py  → analysis/toy_egomotion_results.json
"""
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("toy_egomotion_results.json")
rng = np.random.default_rng(0)

H, W = 28, 40
FOV_H = math.radians(110.0)
DEPTH, DEPTH_SD = 0.5, 0.1
MOVE_FRAC, MOVE_DIST = 0.08, 0.05
HEAD_ROT_DEG, HEAD_TRANS = 4.7, 0.025
ROT_ERR_DEG, TRANS_ERR = 1.9, 0.018
THRESH = 0.010
N_TRIALS = 300


def rot(axis, deg):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    a = math.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


def rand_unit():
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def make_scene():
    f = (W / 2) / math.tan(FOV_H / 2)               # 가상 픽셀 격자 기준 초점거리 (격자 단위)
    u, v = np.meshgrid(np.arange(W) - W / 2 + 0.5, np.arange(H) - H / 2 + 0.5)
    rays = np.stack([u / f, v / f, np.ones_like(u)], -1).reshape(-1, 3)
    depth = np.clip(rng.normal(DEPTH, DEPTH_SD, size=rays.shape[0]), 0.25, None)
    X = rays * depth[:, None]                       # 시각 t의 카메라 좌표 점 (세계 = 시각 t의 카메라)
    moving = rng.random(rays.shape[0]) < MOVE_FRAC
    disp = np.zeros_like(X)
    d = rng.normal(size=(moving.sum(), 3))
    disp[moving] = MOVE_DIST * d / np.linalg.norm(d, axis=1, keepdims=True)
    return X, moving, disp


def trial():
    X, moving, disp = make_scene()
    X_world_future = X + disp                       # 시각 t+T의 세계 좌표(=시각 t 카메라 좌표계) 점
    R = rot(rand_unit(), HEAD_ROT_DEG)
    t = HEAD_TRANS * rand_unit()
    # 머리(카메라)가 시각 t+T에 세계에서 (R, t) 자세. 카메라 좌표계의 점 = R^T (X_w - t)
    X_cam_future = (X_world_future - t) @ R          # 행벡터 형태의 R^T (X_w - t)
    out = {}
    # (a) 안정화 없음: 시각 t+T의 카메라 좌표 점에서 시각 t 카메라 좌표 점을 뺀다
    raw = X_cam_future - X
    # (b) 정확한 포즈로 안정화: 시각 t 카메라 좌표로 되돌린다 (논문 식: T_t^-1 T_{t+T} X_{t+T})
    stab = X_cam_future @ R.T + t - X
    # (c) RGB만 추정한 포즈: 회전 1.9도, 이동 1.8 cm 오차가 임의 방향으로 섞임
    R_hat = R @ rot(rand_unit(), ROT_ERR_DEG)
    t_hat = t + TRANS_ERR * rand_unit()
    stab_rgb = X_cam_future @ R_hat.T + t_hat - X
    for name, flow in (("no_stabilization", raw), ("vio_exact", stab), ("rgb_only_pose", stab_rgb)):
        mag = np.linalg.norm(flow, axis=1)
        keep = mag >= THRESH
        static = ~moving
        out[name] = {
            "static_median_mm": 1000 * float(np.median(mag[static])),
            "static_pass_threshold_frac": float(keep[static].mean()),
            "moving_pass_threshold_frac": float(keep[moving].mean()) if moving.any() else float("nan"),
            "precision_of_kept": float(moving[keep].mean()) if keep.any() else float("nan"),
            "moving_flow_err_mm": 1000 * float(np.mean(np.linalg.norm((flow - disp)[moving], axis=1))) if moving.any() else float("nan"),
        }
    return out


def main():
    acc = {}
    for _ in range(N_TRIALS):
        r = trial()
        for k, v in r.items():
            for m, x in v.items():
                acc.setdefault(k, {}).setdefault(m, []).append(x)
    res = {k: {m: round(float(np.nanmean(xs)), 4) for m, xs in v.items()} for k, v in acc.items()}
    # DINO 패치 격자에서 머리 회전이 만드는 이미지 이동 (224 px 입력, 패치 14 px)
    px_per_deg = 224 / 110.0
    res["dino_patch_shift_for_head_rotation"] = {
        "assumed_hfov_deg": 110, "input_px": 224, "patch_px": 14,
        "shift_px_4.7deg": round(HEAD_ROT_DEG * px_per_deg, 1),
        "shift_in_patches": round(HEAD_ROT_DEG * px_per_deg / 14, 2),
    }
    res["settings"] = {"anchors": H * W, "depth_m": DEPTH, "move_frac": MOVE_FRAC, "move_dist_m": MOVE_DIST,
                       "head_rot_deg": HEAD_ROT_DEG, "head_trans_m": HEAD_TRANS, "rot_err_deg": ROT_ERR_DEG,
                       "trans_err_m": TRANS_ERR, "threshold_m": THRESH, "trials": N_TRIALS, "seed": 0}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
