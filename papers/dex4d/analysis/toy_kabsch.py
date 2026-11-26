"""토이 2: 대응이 알려진 점쌍에서 Kabsch(최소제곱 강체 정합)가 점 잡음, 가시 점 수, 이상치에 얼마나 민감한가.

논문 부록 F와 4.1절은 NovaFlow-CL의 실패 원인으로 "가시 점이 적고 잡음이 많을 때 Kabsch가 회전을 풀지 못한다"를 든다.
이 토이는 그 서술을 두 요인으로 나눠 본다: (a) 가우시안 점 잡음 + 가시 점 수, (b) 일부 점의 추적 이상치(드리프트, ID 혼동).
기준선 구현(NovaFlow-CL)이 논문에 자세히 없으므로, 여기 결과는 "Kabsch 일반"의 성질이지 그 기준선의 실제 오차가 아니다.

설정(가정): 물체 점군은 표준편차 (4, 3, 2) cm인 비등방 가우시안, 카메라 쪽 반쪽만 보임(가시 점 수 n),
참 회전은 30–90도 임의 축. 반복 2000회. 이상치는 비율 q의 점에 반지름 5 cm의 임의 방향 변위를 더함.
RANSAC 변형(3점 표본 200회, 임계 1.5 cm)과 비교한다.

실행: python3 analysis/toy_kabsch.py  (numpy, 약 10초) → analysis/toy_kabsch_results.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "toy_kabsch_results.json"
rng = np.random.default_rng(0)
TRIALS = 2000


def rot(axis, ang):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def ransac_kabsch(P, Q, iters=200, thr=0.015):
    n = len(P)
    best, best_in = None, -1
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        R, t = kabsch(P[idx], Q[idx])
        inl = np.linalg.norm(P @ R.T + t - Q, axis=1) < thr
        if inl.sum() > best_in:
            best, best_in = inl, inl.sum()
    if best_in < 3:
        return kabsch(P, Q)
    return kabsch(P[best], Q[best])


def angle_err(R, Rt):
    c = (np.trace(R.T @ Rt) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def trial(n, sigma, q, method):
    pts = rng.normal(size=(400, 3)) * np.array([0.04, 0.03, 0.02])
    vis = pts[pts[:, 0] < np.quantile(pts[:, 0], 0.5)]  # 카메라 쪽 반
    P = vis[rng.choice(len(vis), n, replace=False)]
    Rt = rot(rng.normal(size=3), np.radians(rng.uniform(30, 90)))
    tt = rng.normal(size=3) * 0.05
    Q = P @ Rt.T + tt + rng.normal(size=P.shape) * sigma
    k = int(round(q * n))
    if k:
        out = rng.choice(n, k, replace=False)
        d = rng.normal(size=(k, 3))
        Q[out] += d / np.linalg.norm(d, axis=1, keepdims=True) * 0.05
    R, _ = (kabsch if method == "kabsch" else ransac_kabsch)(P, Q)
    return angle_err(R, Rt)


def main():
    res = {"setup": {"trials": TRIALS, "object_std_cm": [4, 3, 2], "outlier_radius_cm": 5}, "gauss": {}, "outlier": {}}
    for sigma in [0.002, 0.005, 0.01]:
        for n in [64, 32, 16, 8]:
            e = [trial(n, sigma, 0.0, "kabsch") for _ in range(TRIALS)]
            res["gauss"][f"sigma{sigma * 1000:.0f}mm_n{n}"] = {"median_deg": round(float(np.median(e)), 2), "p90_deg": round(float(np.percentile(e, 90)), 2)}
    for q in [0.0, 0.1, 0.2, 0.3]:
        for n in [32, 16]:
            ek = [trial(n, 0.005, q, "kabsch") for _ in range(TRIALS)]
            er = [trial(n, 0.005, q, "ransac") for _ in range(TRIALS // 4)]
            res["outlier"][f"q{q}_n{n}"] = {"kabsch_median_deg": round(float(np.median(ek)), 2), "kabsch_p90_deg": round(float(np.percentile(ek, 90)), 2),
                                            "ransac_median_deg": round(float(np.median(er)), 2), "ransac_p90_deg": round(float(np.percentile(er, 90)), 2)}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in res["gauss"].items():
        print("gauss", k, v)
    for k, v in res["outlier"].items():
        print("outlier", k, v)


if __name__ == "__main__":
    main()
