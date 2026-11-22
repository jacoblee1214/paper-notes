"""토이: 다중 뷰 손끝 삼각측량이 단일 뷰의 깊이 오차를 줄이는가 (논문 §3.2.1, 부록 A.1의 주장).

논문은 "각 뷰는 단안이라 깊이가 부정확하고, 삼각측량이 잔차를 다른 카메라가 가로 방향으로 구속하는 방향으로
밀어낸다"고 주장하지만 융합된 손끝 위치의 오차 수치는 보고하지 않는다. 여기서는 카메라 배치와 오차 크기를 가정해
오차 타원체(공분산)를 계산하고 몬테카를로로 확인한다. 가정값이므로 실제 시스템의 오차가 아니다.

가정 (모두 논문에 없음, 조정 가능):
  - 작업 공간 중심 (0.5, 0, 0.15) m, 장면 카메라 둘은 좌우 +-0.7 m, 높이 0.6 m, 손목 카메라는 위쪽 0.8 m
  - 뷰별 손끝 2D 오차: 3 px (f = 600 px) -> 방위각 오차 0.005 rad, 광선에 수직인 방향
  - 단안 MANO 깊이 오차(광선 방향): 30 mm (HaPTIC가 센서 깊이에 맞춰 이동되기 전의 값을 가정)
  - 가중치는 모두 1 (논문은 실루엣 IoU 가중)
실행: python3 analysis/toy_triangulation.py   -> analysis/toy_triangulation_results.json  (numpy, 1초 미만)
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_triangulation_results.json"
rng = np.random.default_rng(0)
X = np.array([0.5, 0.0, 0.15])
SIG_LAT = 0.005      # rad
SIG_DEPTH = 0.030    # m, 단안 깊이(광선 방향)

CAMS = {
    "left": np.array([0.5, -0.7, 0.6]),
    "right": np.array([0.5, 0.7, 0.6]),
    "wrist_top": np.array([0.5, 0.0, 0.8]),
}


def unit(v):
    return v / np.linalg.norm(v)


def ray_basis(o):
    d = unit(X - o)
    a = unit(np.cross(d, [0, 0, 1.0]) if abs(d[2]) < 0.99 else np.cross(d, [1.0, 0, 0]))
    b = np.cross(d, a)
    return d, a, b, np.linalg.norm(X - o)


def single_view_cov(o):
    d, a, b, r = ray_basis(o)
    lat = (r * SIG_LAT) ** 2
    return SIG_DEPTH ** 2 * np.outer(d, d) + lat * (np.outer(a, a) + np.outer(b, b))


def fused_cov(names):
    """광선(깊이는 무시, 방위각만 사용) 최소제곱 삼각측량의 공분산. 가중치 1."""
    A = np.zeros((3, 3))
    M = np.zeros((3, 3))
    for n in names:
        d, a, b, r = ray_basis(CAMS[n])
        P = np.eye(3) - np.outer(d, d)
        A += P
        M += (r * SIG_LAT) ** 2 * P
    Ai = np.linalg.inv(A)
    return Ai @ M @ Ai


def solve_mc(names, n=20000):
    errs = np.zeros((n, 3))
    for k in range(n):
        A = np.zeros((3, 3))
        rhs = np.zeros(3)
        for nme in names:
            o = CAMS[nme]
            d, a, b, r = ray_basis(o)
            nd = unit(d + SIG_LAT * (rng.normal() * a + rng.normal() * b))
            P = np.eye(3) - np.outer(nd, nd)
            A += P
            rhs += P @ o
        errs[k] = np.linalg.solve(A, rhs) - X
    return errs


def summarize(cov):
    w = np.linalg.eigvalsh(cov)
    return {"rms_mm": float(np.sqrt(np.trace(cov)) * 1000),
            "sigma_axes_mm": [float(np.sqrt(x) * 1000) for x in w[::-1]]}


res = {"assumptions": {"sigma_lateral_rad": SIG_LAT, "sigma_depth_m": SIG_DEPTH, "target": X.tolist(),
                       "cameras": {k: v.tolist() for k, v in CAMS.items()}}}
res["single_view"] = {n: summarize(single_view_cov(CAMS[n])) for n in CAMS}
for names in (["left", "right"], ["left", "right", "wrist_top"]):
    key = "+".join(names)
    res["fused_" + key] = summarize(fused_cov(names))
    e = solve_mc(names, 4000)
    res["fused_" + key]["mc_rms_mm"] = float(np.sqrt((e ** 2).sum(1).mean()) * 1000)

# 두 장면 카메라 사이 각도에 따른 오차 (대칭 배치, 높이 0.6 m 고정)
sweep = []
for deg in (10, 20, 30, 45, 60, 90, 120, 160):
    half = np.deg2rad(deg / 2)
    dist = 0.9
    old = dict(CAMS)
    CAMS["left"] = X + dist * np.array([0, -np.sin(half), np.cos(half)])
    CAMS["right"] = X + dist * np.array([0, np.sin(half), np.cos(half)])
    c = fused_cov(["left", "right"])
    s = summarize(c)
    sweep.append({"angle_deg": deg, "rms_mm": s["rms_mm"], "worst_axis_mm": s["sigma_axes_mm"][0]})
    CAMS.update(old)
res["two_camera_angle_sweep"] = sweep
res["note"] = ("가정 하에서 두 카메라가 넓게 벌어지면 단안 깊이 오차 30 mm가 수 mm로 줄고, 각도가 작으면 줄지 않는다. "
               "이 토이는 손끝 정점의 오검출(MANO 정점과 실제 접촉점의 차이), 가림, 캘리브레이션 오차를 넣지 않았다.")
OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "assumptions"}, indent=1, ensure_ascii=False))
