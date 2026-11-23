"""토이: 태그 배열 PnP의 이론적 자세 정밀도(Cramér–Rao 하한)를 직접 계산한다.

논문 부록 A.2.2는 모서리 위치 정밀도 0.25 px를 가정하고, 태그 하나의 폭(37 px = 2 mm)만으로
병진 감도 0.0135 mm를 세 축(x, y, z)에 똑같이 적용한다. 여기서는 35개 태그의 실제 배치
(analysis/tag_layout_mm.json, 공식 설정 파일의 태그 위치)와 공식 설정의 카메라 내부 파라미터(fx = 346 px)로
핀홀 모델을 세우고, 보이는 태그 수별로 6자유도 자세의 표준편차 하한을 계산한다.
모서리 잡음은 논문과 같은 0.25 px(독립, 등방)로 가정한다. 렌즈 왜곡, 태그 검출 편향, 가림은 모델에 없다.

실행: python3 analysis/toy_tag_crlb.py  (numpy, 1초 미만)  → analysis/toy_tag_crlb_results.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
rng = np.random.default_rng(0)
tags = {int(k): np.array(v) for k, v in json.loads((HERE / "tag_layout_mm.json").read_text()).items()}
FX, FY, CX, CY = 346.0, 345.4, 323.5, 222.2
SIGMA = 0.25
HALF = 1.0  # 태그 폭 2 mm


def corners(ids):
    pts = []
    for i in ids:
        c = tags[i]
        for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            pts.append([c[0] + dx * HALF, c[1] + dy * HALF, 0.0])
    return np.array(pts)


def rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def project(X, pose):
    R = rodrigues(pose[3:])
    Xc = X @ R.T + pose[:3]
    return np.stack([FX * Xc[:, 0] / Xc[:, 2] + CX, FY * Xc[:, 1] / Xc[:, 2] + CY], axis=1)


def jac(X, pose, eps=1e-6):
    base = project(X, pose).ravel()
    J = np.zeros((base.size, 6))
    for k in range(6):
        p = pose.copy()
        p[k] += eps
        J[:, k] = (project(X, p).ravel() - base) / eps
    return J


ids_all = sorted(tags)
out = {"sigma_px": SIGMA, "fx_px": FX, "array_extent_mm": [float(np.ptp([t[0] for t in tags.values()]) + 2), float(np.ptp([t[1] for t in tags.values()]) + 2)], "cases": {}}
for Z in (12.0, 18.7):
    px_per_mm = FX / Z
    row = {"tag_width_px": round(2 * px_per_mm, 1)}
    for n in (35, 20, 5, 1):
        stds = []
        for rep in range(40 if n < 35 else 1):
            sub = ids_all if n == 35 else (list(rng.choice(ids_all, n, replace=False)) if n > 1 else [0])
            X = corners(sub)
            pose = np.array([0.0, 0.0, Z, 0.0, 0.0, 0.0])
            J = jac(X, pose)
            cov = SIGMA ** 2 * np.linalg.inv(J.T @ J)
            stds.append(np.sqrt(np.diag(cov)))
        s = np.mean(stds, axis=0)
        row[f"{n}_tags"] = {"std_xyz_mm": [round(float(v), 5) for v in s[:3]], "std_rot_rad": [round(float(v), 5) for v in s[3:]]}
    out["cases"][f"Z={Z}mm"] = row
out["paper_assumption"] = {"dl_min_mm": 0.0135, "dtheta_min_rad": 0.0136}
out["note"] = "CRLB(모서리 독립 등방 잡음 0.25 px)이므로 달성 가능한 정밀도의 하한. 실제 지터(표 5: 35태그 0.012 mm)는 검출 편향과 조명이 더해진 값이라 직접 비교는 참고용."
(HERE / "toy_tag_crlb_results.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
