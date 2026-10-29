"""물체 자세 오차가 기하 접점 추정을 얼마나 흔드는지 (Fig. 6 조건, CPU numpy). 실행: python3 analysis/pose_error_toy.py

물체: 6×6×12 cm 상자(해석적 SDF). 실제 자세에서 손 표면 점을 두 부류로 둔다.
- 실제 접점 16개: +x 면(엄지 3), −x 면(손가락 9), +y 면(손바닥 4), 표면 위(거리 0)
- 비접촉 점 48개: 실제 표면에서 5–20 mm 떨어진 손가락 옆·등 쪽 점
제어기가 믿는 자세 = 실제 자세 ∘ (임의 축으로 θ 회전 + 임의 방향으로 δ 이동). 회전 중심은 물체 원점.
논문 규칙대로 믿는 자세의 부호 거리 d < d_thresh인 점을 접점으로 본다(침투한 점도 포함, Sec. III-B).
조건마다 오프셋 400개(고정 시드)로 실제 접점 재현율, 비접촉 점 오검출률, 법선 오차를 잰다.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B = np.array([0.03, 0.03, 0.06])          # 반 크기 (m)
OFFSETS = [(0, 0.0), (10, 0.001), (20, 0.003), (30, 0.005)]   # Fig. 6
THRESH = [0.005, 0.01, 0.02]


def sdf_box(p: np.ndarray) -> np.ndarray:
    q = np.abs(p) - B
    outside = np.linalg.norm(np.maximum(q, 0), axis=-1)
    inside = np.minimum(q.max(axis=-1), 0)
    return outside + inside


def grad(p: np.ndarray, h: float = 1e-5) -> np.ndarray:
    g = np.zeros_like(p)
    for i in range(3):
        e = np.zeros(3); e[i] = h
        g[:, i] = (sdf_box(p + e) - sdf_box(p - e)) / (2 * h)
    return g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-12)


def rot(axis: np.ndarray, ang: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def hand_points(rng):
    con, n_true = [], []
    for z in (-0.02, 0.0, 0.02):                       # 엄지 +x
        con.append([B[0], 0.0, z]); n_true.append([1, 0, 0])
    for y in (-0.02, 0.0, 0.02):                       # 손가락 −x
        for z in (-0.04, 0.0, 0.04):
            con.append([-B[0], y, z]); n_true.append([-1, 0, 0])
    for x in (-0.015, 0.015):                          # 손바닥 +y
        for z in (-0.03, 0.03):
            con.append([x, B[1], z]); n_true.append([0, 1, 0])
    con, n_true = np.array(con, float), np.array(n_true, float)
    # 비접촉 점: 실제 접점에서 바깥 법선 방향으로 5–20 mm, 옆으로 조금 흩뜨림
    idx = rng.integers(0, len(con), 48)
    off = rng.uniform(0.005, 0.020, 48)
    side = rng.normal(0, 0.01, (48, 3))
    side -= (side * n_true[idx]).sum(1, keepdims=True) * n_true[idx]
    far = con[idx] + off[:, None] * n_true[idx] + side
    return con, n_true, far


def main() -> None:
    rng = np.random.default_rng(0)
    con, n_true, far = hand_points(rng)
    assert np.allclose(sdf_box(con), 0, atol=1e-9) and (sdf_box(far) > 0.004).all()
    r_max = float(np.linalg.norm(B))
    out = {"object_half_extent_m": B.tolist(), "n_true_contacts": len(con), "n_noncontact_points": len(far),
           "max_surface_shift_by_rotation_cm": {f"{a}deg": round(100 * 2 * r_max * math.sin(math.radians(a) / 2), 2) for a, _ in OFFSETS},
           "results": {}}
    for ang, dpos in OFFSETS:
        draws = []
        for s in range(400):
            r = np.random.default_rng(1000 + s)
            Ro = rot(r.normal(size=3), math.radians(ang))
            to = dpos * (lambda v: v / np.linalg.norm(v))(r.normal(size=3))
            # 믿는 자세의 물체 좌표로: p_b = Roᵀ (p − to)
            pc, pf = (con - to) @ Ro, (far - to) @ Ro
            dc, df = sdf_box(pc), sdf_box(pf)
            nb = grad(pc) @ Ro.T                                   # 믿는 법선을 world로
            nerr = np.degrees(np.arccos(np.clip((nb * n_true).sum(1), -1, 1)))
            draws.append((dc, df, nerr))
        row = {}
        for th in THRESH:
            rec = np.mean([(dc < th).mean() for dc, _, _ in draws])
            fp = np.mean([(df < th).mean() for _, df, _ in draws])
            ne = np.concatenate([ne[dc < th] for dc, _, ne in draws]) if any((dc < th).any() for dc, _, _ in draws) else np.array([np.nan])
            row[f"thresh_{int(th*1000)}mm"] = {
                "true_contact_recall": round(float(rec), 3),
                "noncontact_false_detect": round(float(fp), 3),
                "normal_err_deg_median": round(float(np.median(ne)), 1),
                "normal_err_deg_p90": round(float(np.quantile(ne, 0.9)), 1),
                "share_detected_penetrating": round(float(np.mean([((dc < th) & (dc < 0)).sum() / max((dc < th).sum(), 1) for dc, _, _ in draws])), 3),
            }
        out["results"][f"{ang}deg_{int(dpos*1000)}mm"] = row
    path = ROOT / "analysis" / "pose_error_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
