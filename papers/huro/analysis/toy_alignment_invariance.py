"""식 (2), (7)의 정렬 변환 T^{B<-W}(이동 3 + yaw)가 정책 행동 표현에 미치는 영향을 수치로 확인하는 toy.

확인할 것
  (1) 손목 목표를 "현재 손목 자세에 대한, 현재 카메라 프레임 C_obs 기준 상대 자세"로 쓰면(논문 3.2절),
      정렬 변환을 어떻게 고르든 행동 값은 변하지 않는다. 같은 강체 변환이 카메라와 손목에 함께 걸리기 때문이다.
  (2) yaw만 있는 정렬은 중력 방향을 보존한다. pitch/roll이 섞이면 보존하지 않는다.
  (3) 따라서 정렬 변환 ξ는 IK의 도달 가능성(관절 한계, 작업 공간)에만 영향을 주고, 행동 표현의 값 자체에는 들어가지 않는다.
실행: python3 analysis/toy_alignment_invariance.py  → analysis/toy_alignment_invariance_results.json (1초 미만)
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_alignment_invariance_results.json"
rng = np.random.default_rng(0)


def rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def se3(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def random_pose(scale=1.0):
    R = rot("z", rng.uniform(-3, 3)) @ rot("y", rng.uniform(-.5, .5)) @ rot("x", rng.uniform(-.5, .5))
    return se3(R, rng.normal(size=3) * scale)


T_ALIGN_YAW = se3(rot("z", 0.9), [0.3, -0.2, 0.1])               # 논문의 정렬: 이동 3 + yaw
T_ALIGN_FULL = se3(rot("z", 0.9) @ rot("y", 0.4), [0.3, -0.2, 0.1])  # 비교용: pitch 포함

n_traj, H = 200, 40
max_dev, g_dev_yaw, g_dev_full = 0.0, 0.0, 0.0
g_world = np.array([0.0, 0.0, -1.0])
for _ in range(n_traj):
    cam = [random_pose() for _ in range(H + 1)]       # T^{W<-Csrc}_t
    wrist = [random_pose() for _ in range(H + 1)]     # 세계 좌표계 손목 자세
    for T_align in (T_ALIGN_YAW, T_ALIGN_FULL):
        # 정렬 전 상대 행동: t+k의 손목을 시각 t의 카메라 프레임에서 표현
        rel0 = [np.linalg.inv(cam[0]) @ wrist[k] for k in range(1, H + 1)]
        # 정렬 후: 카메라와 손목에 같은 T를 곱한다
        cam_b = [T_align @ c for c in cam]
        wrist_b = [T_align @ w for w in wrist]
        rel1 = [np.linalg.inv(cam_b[0]) @ wrist_b[k] for k in range(1, H + 1)]
        max_dev = max(max_dev, max(np.abs(a - b).max() for a, b in zip(rel0, rel1)))
    # 중력 방향이 정렬 뒤에도 같은가 (세계 z축이 아래라고 둔 벡터를 회전시켜 비교)
    g_dev_yaw = max(g_dev_yaw, np.linalg.norm(T_ALIGN_YAW[:3, :3] @ g_world - g_world))
    g_dev_full = max(g_dev_full, np.linalg.norm(T_ALIGN_FULL[:3, :3] @ g_world - g_world))

res = {
    "n_trajectories": n_traj, "horizon_H": H,
    "max_abs_deviation_relative_wrist_action_after_alignment": float(max_dev),
    "gravity_vector_change_yaw_only": float(g_dev_yaw),
    "gravity_vector_change_with_pitch": float(g_dev_full),
    "interpretation": "상대 손목 행동은 정렬 변환과 무관(편차 ~1e-15). yaw 정렬은 중력을 보존, pitch 포함 정렬은 보존하지 않음.",
}
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
