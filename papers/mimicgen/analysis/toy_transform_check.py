"""토이 V1: 세그먼트 변환 식(논문 식 4.2, 부록 M)과 공식 코드의 변환 줄이 같은 결과를 내는지 수치로 검사한다.

검사하는 것:
  (a) 부록 M의 식  C'_t = O'_0 (O_0)^{-1} C_t  이 객체 좌표계 기준 상대 자세를 보존하는가 (무작위 SE(3) 1000개 사례).
  (b) 공식 코드 pose_utils.transform_source_data_segment_using_object_pose 의 두 단계 계산
      (rel = inv(O) C 를 pose_in_A_to_pose_in_B로 계산, 그다음 O' rel) 이 (a)와 같은가.
  (c) 음성 대조: 곱하는 순서를 바꾸면(O^{-1} O' C) 상대 자세가 깨지는가.

코드는 저장소를 import하지 않고, 저장소 함수를 읽고 같은 연산을 numpy로 옮겨 쓴 것이다(실행하지 않음).
입력: 없음 (무작위 시드 0). 출력: analysis/toy_transform_check_results.json
실행: python3 analysis/toy_transform_check.py   (1초 미만)
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "toy_transform_check_results.json"
rng = np.random.default_rng(0)


def rand_rot(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rand_pose(rng, t_scale=0.3):
    T = np.eye(4)
    T[:3, :3] = rand_rot(rng)
    T[:3, 3] = rng.uniform(-t_scale, t_scale, size=3)
    return T


def inv(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# 저장소 pose_utils.py 의 규약을 그대로 옮긴 것: pose_in_B = pose_A_in_B @ pose_in_A
def pose_in_A_to_pose_in_B(pose_in_A, pose_A_in_B):
    return np.einsum("ij,njk->nik", pose_A_in_B, pose_in_A)


def code_transform(obj_pose, src_eef_poses, src_obj_pose):
    """저장소 transform_source_data_segment_using_object_pose 의 두 단계."""
    rel = pose_in_A_to_pose_in_B(pose_in_A=src_eef_poses, pose_A_in_B=inv(src_obj_pose)[None][0])
    return pose_in_A_to_pose_in_B(pose_in_A=rel, pose_A_in_B=obj_pose)


def paper_transform(O_new, O_src, C):
    """부록 M: C'_t = O'_0 (O_0)^{-1} C_t."""
    return np.einsum("ij,njk->nik", O_new @ inv(O_src), C)


def rel_err(O_new, O_src, C, Cp):
    """객체 좌표계에서 본 자세 차이: inv(O')C' 대 inv(O)C 의 최대 원소 차이."""
    a = np.einsum("ij,njk->nik", inv(O_new), Cp)
    b = np.einsum("ij,njk->nik", inv(O_src), C)
    return float(np.abs(a - b).max())


N_TRIALS, K = 1000, 20
err_paper, err_code, err_wrong = [], [], []
max_diff_code_vs_paper = 0.0
for _ in range(N_TRIALS):
    O_src, O_new = rand_pose(rng), rand_pose(rng)
    # 소스 세그먼트: 객체 근처의 엔드이펙터 목표 자세 K+1개 (무작위 걸음)
    C = [O_src @ rand_pose(rng, t_scale=0.1) for _ in range(K + 1)]
    C = np.array(C)
    Cp_paper = paper_transform(O_new, O_src, C)
    Cp_code = code_transform(O_new, C, O_src)
    err_paper.append(rel_err(O_new, O_src, C, Cp_paper))
    err_code.append(rel_err(O_new, O_src, C, Cp_code))
    max_diff_code_vs_paper = max(max_diff_code_vs_paper, float(np.abs(Cp_code - Cp_paper).max()))
    # 음성 대조: 곱하는 순서를 바꿈
    Cp_wrong = np.einsum("ij,njk->nik", inv(O_src) @ O_new, C)
    err_wrong.append(rel_err(O_new, O_src, C, Cp_wrong))

res = {
    "n_trials": N_TRIALS,
    "segment_len_plus1": K + 1,
    "paper_formula_max_rel_err": float(max(err_paper)),
    "code_two_step_max_rel_err": float(max(err_code)),
    "code_vs_paper_max_abs_diff": max_diff_code_vs_paper,
    "wrong_order_max_rel_err_median": float(np.median(err_wrong)),
    "wrong_order_max_rel_err_max": float(max(err_wrong)),
    "wrong_order_share_above_1e-6": float(np.mean(np.array(err_wrong) > 1e-6)),
    "note": "오차는 부동소수점 수준(1e-12 안팎)이면 식이 맞다. 음성 대조는 순서가 바뀌면 깨져야 한다.",
}
OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=2, ensure_ascii=False))
