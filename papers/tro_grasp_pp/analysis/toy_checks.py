"""T(R,O) Grasp++ 설계 선택 두 가지를 numpy 토이로 점검한다(약 20초, CPU).

T1. 회전을 축-각 벡터(rotvec)로 두고 가우시안 잡음을 더하는 선택(공식 코드 model/troe_graph.py)의 성질.
    - 무작위 회전 중 각도가 pi에 가까운 비율, rotvec이 불연속인 지점(cut locus) 근처 비율
    - 초기 잡음 N(0, I3)의 노름이 pi를 넘는 비율(그 rotvec은 다른 표현과 같은 회전을 가리킴)
T2. 평가의 환경 충돌 검사(validation/collision_check.py 기본값): 손 표면 점 중 환경 점군에서 3 mm 이내인 점이
    48개 이상이고 손 점 수의 0.6 % 이상이면 충돌. 환경 점군이 성기면 얕은 관통을 놓친다.
    - 구(반지름 4 cm, 표면 점 4,608개 = 링크 18개 x 256)가 평면 장애물에 깊이 d만큼 관통
    - 환경 점 밀도 rho(점/cm^2)별 검출률

한계: 실제 손 메시와 장면이 아니라 구와 평면이다. 임계값 규칙이 어떻게 반응하는지만 본다.
실행: python3 analysis/toy_checks.py   (numpy, scipy)
출력: analysis/toy_checks_results.json
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

OUT = Path(__file__).resolve().parent / "toy_checks_results.json"
rng = np.random.default_rng(0)
RES = {}

# ---------------------------------------------------------------- T1
n = 400_000
rot = R.random(n, random_state=1)
rv = rot.as_rotvec()
theta = np.linalg.norm(rv, axis=1)
t1 = {"n_random_rotations": n}
for d in (0.3, 0.1, 0.05):
    t1[f"frac_angle_within_{d}_rad_of_pi"] = float(np.mean(theta > np.pi - d))
    t1[f"analytic_frac_{d}"] = float(((np.pi - (np.pi - d)) - (np.sin(np.pi) - np.sin(np.pi - d))) / np.pi)
# 작은 섭동에 대한 rotvec 점프
eps = 0.01
axis = rng.normal(size=(n, 3))
axis /= np.linalg.norm(axis, axis=1, keepdims=True)
rot2 = rot * R.from_rotvec(axis * eps)
jump = np.linalg.norm(rot2.as_rotvec() - rv, axis=1)
t1["perturbation_rad"] = eps
t1["frac_rotvec_jump_gt_1rad"] = float(np.mean(jump > 1.0))
near = theta > np.pi - 0.05
t1["among_near_pi_frac_jump_gt_1rad"] = float(np.mean(jump[near] > 1.0))
t1["max_jump_rad"] = float(jump.max())
# 초기 잡음 N(0, I3)
t1["P(|N(0,I3)|>pi)"] = float(stats.chi.sf(np.pi, 3))
t1["note"] = ("rotvec은 |r| <= pi에서 유일하고 pi에서 불연속이다. 학습 목표와 상대 회전 간선 특징(so3_log_map)이 모두 이 표현을 쓰므로, "
              "각도가 pi 근처인 링크 자세(손바닥이 아래·뒤를 향하는 틀 등)에서 입력이 튄다. 실제 손 틀 분포는 데이터가 없어 보지 못했다.")
RES["T1_rotvec"] = t1

# ---------------------------------------------------------------- T2
H = 18 * 256
r_cm = 4.0
prox = 0.3  # cm = 3 mm
min_close, ratio = 48, 0.006


def sphere_pts(n_pts):
    v = rng.normal(size=(n_pts, 3))
    return r_cm * v / np.linalg.norm(v, axis=1, keepdims=True)


def detect(rho, depth_mm, trials=150):
    hits, counts = 0, []
    for _ in range(trials):
        pts = sphere_pts(H)
        pts[:, 2] += r_cm - depth_mm / 10.0  # 구 최하점이 평면 아래 depth만큼
        side = 30.0
        m = rng.poisson(rho * side * side)
        env = np.column_stack([rng.uniform(-side / 2, side / 2, m), rng.uniform(-side / 2, side / 2, m), np.zeros(m)])
        if m == 0:
            counts.append(0)
            continue
        c = int((cKDTree(env).query(pts, distance_upper_bound=prox)[0] <= prox).sum())
        counts.append(c)
        hits += int(c >= min_close and c / H >= ratio)
    return hits / trials, float(np.mean(counts))


depths = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0]
t2 = {"hand_points": H, "sphere_radius_cm": r_cm, "rule": {"within_mm": 3, "min_close_points": min_close, "min_ratio": ratio},
      "depths_mm": depths, "by_density": {}}
for rho in (0.3, 1.0, 3.0, 10.0):
    rates, cnts = zip(*(detect(rho, d) for d in depths))
    t2["by_density"][f"rho_{rho}_per_cm2"] = {"mean_spacing_mm": float(10 / np.sqrt(rho)),
                                              "detect_rate": [round(x, 2) for x in rates],
                                              "mean_close_points": [round(x, 1) for x in cnts]}
t2["note"] = ("검출률은 관통 깊이가 아니라 환경 점 밀도가 정한다. 깊이 0(닿기만 함)에서도 밀도 3점/cm^2이면 95 %가 충돌로 판정되고, "
              "밀도 0.3점/cm^2(간격 18 mm)이면 깊이 30 mm 관통도 2 %만 검출한다. 즉 규칙은 접촉과 관통을 구분하지 못하고, 성긴 점군에서는 관통을 놓친다. "
              "공식 코드의 환경 점은 목표 중심에서 가까운 2,048개이고 점군 밀도는 장면마다 다르다(실제 밀도는 데이터가 없어 보지 못했다).")
RES["T2_collision_rule"] = t2

OUT.write_text(json.dumps(RES, ensure_ascii=False, indent=1))
print(json.dumps(RES, ensure_ascii=False, indent=1)[:3500])
