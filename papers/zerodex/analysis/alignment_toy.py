"""Eq. 7 원자 동작 정렬 toy (CPU, numpy). 실행: python3 analysis/alignment_toy.py

저장된 도구 궤적 T의 시작·끝 앵커 (X^s, X^e)를 장면의 (X^app, X^term)에 맞추는 강체 정렬을,
논문 v3 설명대로 회전을 미리 정한 축(여기서는 연직축 z)으로 제한해 풉니다.

- A1: 대응점이 두 개뿐이라 3-DoF 회전은 앵커를 잇는 축 둘레로 1자유도가 남는다는 것 확인
- A2: yaw + 병진(4-DoF) 최소제곱 해의 앵커 잔차 = 두 앵커 간 벡터 불일치의 절반
- A3: 저장 궤적과 장면의 앵커 간격·높이차가 다를 때 끝점 오차 표
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def rot(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def align_yaw(Xs, Xe, Ya, Yb):
    """min Σ||R_z(θ) X + t − Y||² over θ, t. 두 점의 중심화 벡터는 ±(차이)/2."""
    u, w = (Xe - Xs) / 2, (Yb - Ya) / 2
    th = np.arctan2(u[0] * w[1] - u[1] * w[0], u[0] * w[0] + u[1] * w[1])
    R = rot([0, 0, 1], th)
    t = (Ya + Yb) / 2 - R @ (Xs + Xe) / 2
    return R, t


def main() -> None:
    out: dict = {}
    Xs, Xe = np.array([0.0, 0.0, 0.10]), np.array([0.25, 0.0, 0.05])  # 예: 주둥이 시작점 → 기울인 끝점

    # A1 두 점 대응은 앵커 축 둘레 회전을 결정하지 못함
    axis = Xe - Xs
    mid = (Xs + Xe) / 2
    res = []
    for ang in (0.0, 0.5, 1.0, 2.0):
        R = rot(axis, ang)
        res.append(float(np.linalg.norm(R @ (Xs - mid) + mid - Xs) + np.linalg.norm(R @ (Xe - mid) + mid - Xe)))
    out["A1_free_roll_about_anchor_axis_residual_m"] = [round(r, 12) for r in res]

    # A2, A3 yaw 제한 정렬의 앵커 잔차
    rows = {}
    for dl in (0.0, 0.02, 0.05, 0.10):        # 장면 앵커 간 수평 거리 − 저장 궤적의 수평 거리
        for dh in (0.0, 0.03):                # 높이차 불일치
            Ya = np.array([0.40, 0.20, 0.12])
            d_h = np.linalg.norm((Xe - Xs)[:2]) + dl
            ang = 0.7
            Yb = Ya + np.array([d_h * np.cos(ang), d_h * np.sin(ang), (Xe - Xs)[2] + dh])
            R, t = align_yaw(Xs, Xe, Ya, Yb)
            ea = np.linalg.norm(R @ Xs + t - Ya)
            eb = np.linalg.norm(R @ Xe + t - Yb)
            pred = 0.5 * np.hypot(dl, dh)
            rows[f"dl_{100 * dl:g}cm_dh_{100 * dh:g}cm"] = {
                "anchor_err_cm": [round(100 * ea, 2), round(100 * eb, 2)], "half_mismatch_cm": round(100 * pred, 2)}
    out["A2_A3_yaw_alignment_residuals"] = rows

    path = ROOT / "analysis" / "alignment_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("wrote", path)


if __name__ == "__main__":
    main()
