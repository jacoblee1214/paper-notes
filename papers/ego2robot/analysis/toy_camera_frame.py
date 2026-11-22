"""토이: 부록 C.2 식 (10)의 카메라 프레임 상대 EEF 행동이 무엇을 통일하는지 수치로 확인한다.

논문은 "base-frame deltas를 R_ce = R_wc^T R_we 로 변환"한다고 쓴다(식 10). 이 토이는 세 가지를 잰다.
  K1. Δ가 월드(베이스) 프레임이면 R_ce를 곱하는 것은 올바른 변환 R_wc^T 와 다르다(EEF 자세에 따라 틀어짐).
      Δ가 EEF(몸체) 프레임이면 R_ce가 올바르다. 즉 식 10은 Δ_ee가 몸체 프레임일 때만 맞다.
  K2. 올바른 카메라 프레임 행동은 월드 좌표계 선택에 불변이다(캘리브레이션이 필요 없다는 주장의 근거).
      반면 카메라 자세에는 의존한다(같은 동작이 카메라마다 다르게 보인다).
  K3. 카메라(머리)가 움직이면, "시각 t의 카메라 프레임으로 표현한 변위"와 "T_ce(t+1) - T_ce(t) 상대 변화"가 달라진다.
      머리 회전이 손 움직임에 섞이는 크기를 어림한다. 속도 값은 우리가 가정한 것이다.
CPU numpy, 1초 미만. 결과: analysis/toy_camera_frame_results.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)
N = 20000


def rand_rot(n):
    return R.random(n, random_state=int(rng.integers(1 << 30))).as_matrix()


# ---- K1
R_wc = rand_rot(N)           # 카메라 -> 월드
R_we = rand_rot(N)           # EEF -> 월드
dp_w = rng.normal(size=(N, 3)) * 0.01   # 월드 프레임 변위 (1 cm 규모)
dp_e = np.einsum("nji,nj->ni", R_we, dp_w)  # 같은 변위를 EEF 프레임으로 (R_we^T dp_w)
R_ce = np.einsum("nji,njk->nik", R_wc, R_we)  # R_wc^T R_we
correct = np.einsum("nji,nj->ni", R_wc, dp_w)  # R_wc^T dp_w  (올바른 카메라 프레임 변위)
read_world = np.einsum("nij,nj->ni", R_ce, dp_w)  # 식 10을 월드 프레임 Δ에 적용
read_body = np.einsum("nij,nj->ni", R_ce, dp_e)   # 식 10을 EEF 프레임 Δ에 적용


def ang(a, b):
    c = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


k1 = {
    "angle_error_world_delta_reading_deg_mean": round(float(ang(read_world, correct).mean()), 1),
    "angle_error_world_delta_reading_deg_p90": round(float(np.percentile(ang(read_world, correct), 90)), 1),
    "max_abs_error_body_delta_reading": float(np.abs(read_body - correct).max()),
    "norm_preserved_world_reading": bool(np.allclose(np.linalg.norm(read_world, axis=1), np.linalg.norm(correct, axis=1))),
}

# ---- K2: 월드 좌표계 불변성과 카메라 의존성
G = rand_rot(N)                               # 임의의 월드 좌표 회전(좌표계 선택)
dp_w2 = np.einsum("nij,nj->ni", G, dp_w)       # 같은 동작을 다른 월드 좌표로
R_wc2 = np.einsum("nij,njk->nik", G, R_wc)     # 카메라 자세도 같은 좌표계로
correct2 = np.einsum("nji,nj->ni", R_wc2, dp_w2)
k2 = {"world_frame_choice_invariance_max_err": float(np.abs(correct2 - correct).max())}
# 카메라 요(yaw)만 바꿔 같은 월드 동작을 보면
dp = np.tile(np.array([0.01, 0.0, 0.0]), (N, 1))
yaw = rng.uniform(-np.pi / 2, np.pi / 2, N)
Rz = R.from_euler("z", yaw).as_matrix()
seen = np.einsum("nji,nj->ni", Rz, dp)
k2["same_motion_two_cameras_angle_deg_mean"] = round(float(np.degrees(np.abs(yaw)).mean()), 1)
k2["same_motion_camera_dependent"] = bool(np.abs(seen - dp).max() > 1e-3)

# ---- K3: 머리 움직임 섞임. 가정: 손은 카메라 앞 0.4 m, 손 속도 0.3 m/s, 15 Hz, 머리 각속도 w deg/s
fps = 15.0
hand_speed = 0.3
dist = 0.4
true_step = hand_speed / fps
k3 = {"assumed": {"fps": fps, "hand_speed_m_s": hand_speed, "hand_distance_m": dist, "true_step_m": round(true_step, 4)}}
for w in (10, 30, 60):
    dth = np.radians(w / fps)
    apparent = dist * dth  # 머리 회전에 의한 카메라 프레임 겉보기 이동(카메라 정면 직교 방향)
    k3[f"head_{w}deg_s"] = {"apparent_shift_m_per_step": round(apparent, 4), "ratio_to_true_step": round(apparent / true_step, 2)}
# 같은 계산을 직접 회전으로
Rw = R.from_euler("y", 30 / fps, degrees=True).as_matrix()
p_c0 = np.array([0.0, 0.0, dist])
p_c1_if_static_hand = Rw.T @ p_c0
k3["check_30deg_s_direct_m"] = round(float(np.linalg.norm(p_c1_if_static_hand - p_c0)), 4)

out = {"K1_eq10_frame_reading": k1, "K2_invariance": k2, "K3_head_motion_mixing": k3, "n_samples": N}
(ROOT / "analysis" / "toy_camera_frame_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=False))
