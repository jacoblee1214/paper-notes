"""RealDexUMI 식 (2)와 4.2절 목표 식의 대수 검사 (numpy, 1–2초).

검사하는 것 (논문 2절 식 (2), 4.2절 목표 식 T̂_target = T̂_t ΔT̂):
  V-a. 로봇 기저 좌표계가 고정 변환 B만큼 바뀌어도 상대 행동 ΔT_{t,k} = T_t^{-1} T_{t+k}는 같다 (base-frame 불변).
  V-b. 역순 정의 ΔT' = T_{t+k} T_t^{-1}를 쓰면 같은 목표 식이 맞지 않는다 (식 (2)의 순서가 필요).
  V-c. 손 기준 좌표계의 고정 보정 오차 E (추적기-손 변환 또는 로봇 FK-손 변환)가 목표 자세에 그대로 옮겨 간다.
        목표 오차는 T_{t+k}(E - I)이므로 평행 이동 오차 크기는 대략 |E의 이동 성분|이다.

이것은 정책 학습이나 로봇 실험의 검증이 아니다. 식이 논문 본문과 맞는지를 확인할 뿐이다.
실행: python3 analysis/toy_relative_action.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "toy_relative_action_results.json"
RNG = np.random.default_rng(0)


def rand_se3(scale_t=0.3, rot_deg=60.0):
    T = np.eye(4)
    T[:3, :3] = Rot.from_rotvec(RNG.normal(size=3) * np.deg2rad(rot_deg) / np.sqrt(3)).as_matrix()
    T[:3, 3] = RNG.uniform(-scale_t, scale_t, size=3)
    return T


def inv(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def trans_err(A, Bm):
    return float(np.linalg.norm(A[:3, 3] - Bm[:3, 3]))


def rot_err_deg(A, Bm):
    Rrel = A[:3, :3].T @ Bm[:3, :3]
    return float(np.rad2deg(np.linalg.norm(Rot.from_matrix(Rrel).as_rotvec())))


def main():
    n = 2000
    res = {"n_trials": n}
    a_err, b_err_t, b_err_r, c_t, c_r = [], [], [], [], []
    for _ in range(n):
        Tt = rand_se3()          # 수집 시점 손 자세 (수집 기준 좌표계)
        Tk = rand_se3()          # t+k 시점 손 자세
        B = rand_se3()           # 로봇 기저 좌표계의 고정 변환 (초기 자세 차이)
        # 식 (2)
        dT = inv(Tt) @ Tk
        # V-a: 기저 변환 뒤에도 같은 상대 행동
        dT_base = inv(B @ Tt) @ (B @ Tk)
        a_err.append(max(trans_err(dT, dT_base), rot_err_deg(dT, dT_base) * 1e-3))
        # 배포 목표 식 T̂_target = T̂_t ΔT̂ (T̂_t = B T_t 로 가정)
        target = (B @ Tt) @ dT_base
        ideal = B @ Tk
        # V-b: 역순 정의 (T_{t+k} T_t^{-1})를 같은 식에 넣으면
        dT_rev = Tk @ inv(Tt)
        target_rev = (B @ Tt) @ dT_rev
        b_err_t.append(trans_err(target_rev, ideal))
        b_err_r.append(rot_err_deg(target_rev, ideal))
        # V-c: 손 기준 고정 보정 오차 E (이동 2 mm, 회전 1도)
        E = np.eye(4)
        E[:3, :3] = Rot.from_rotvec(np.deg2rad(1.0) * RNG.normal(size=3) / np.linalg.norm(RNG.normal(size=3))).as_matrix()
        E[:3, 3] = RNG.normal(size=3)
        E[:3, 3] = E[:3, 3] / np.linalg.norm(E[:3, 3]) * 0.002
        T_hat_t = B @ Tt @ E     # 로봇 FK가 보고하는 손 자세 (보정 오차 E 포함)
        dT_hat = inv(T_hat_t) @ (B @ Tk @ E)  # 같은 수집 행동을 로봇 쪽 손 좌표계에서 계산
        target_E = T_hat_t @ dT_hat
        c_t.append(trans_err(target_E, B @ Tk))
        c_r.append(rot_err_deg(target_E, B @ Tk))

    res["V_a_max_base_invariance_error"] = float(np.max(a_err))
    res["V_b_reversed_definition_trans_err_mm_median"] = float(1000 * np.median(b_err_t))
    res["V_b_reversed_definition_rot_err_deg_median"] = float(np.median(b_err_r))
    res["V_c_fixed_offset_trans_err_mm_median"] = float(1000 * np.median(c_t))
    res["V_c_fixed_offset_trans_err_mm_max"] = float(1000 * np.max(c_t))
    res["V_c_fixed_offset_rot_err_deg_median"] = float(np.median(c_r))
    res["note"] = ("V-c에서 목표 오차는 기저 변환 B와 무관하며, 보정 오차 E 자체에 따라 정해진다. "
                   "V-b는 '역순 정의' 쪽 목표가 배포 기준 목표와 어긋난다는 것을 보인다.")
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
