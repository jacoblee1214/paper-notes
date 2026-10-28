"""보상 설계 toy 검증 (CPU, numpy). 실행: python3 analysis/reward_toy.py

공식 코드 src/envs/rewards/motion_tracking.py의 두 보상을 numpy로 옮겨 다음을 확인합니다.
- V1: rho = 0.058 m에서 30도 회전이 약 3 cm가 된다는 주장 (Sec. III-B a)
- V2: 물체 모양과 회전축에 따라 worst-3 오차가 얼마나 달라지는지 (길쭉한 물체의 장축 회전)
- V3: 보상값과 종료 기준의 환산 (sigma_obj = 0.1, early reset 0.6)
- V4: 레퍼런스 손끝 거리를 "가장 가까운 메쉬 꼭짓점"으로 재는 방식의 여유 (sigma_int = 0.01)
- V5: SDF 격자 해상도 (128^3, +-0.30 m)
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)
RHO, SIGMA_OBJ, SIGMA_INT, TOPK = 0.058, 0.1, 0.01, 3  # config/envs/rewards/flashdexretarget.yaml


def rotmat(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def fps(v: np.ndarray, n: int) -> np.ndarray:
    """object_points.farthest_point_sample와 같은 역할 (첫 점은 0번)."""
    idx = [0]
    d = np.linalg.norm(v - v[0], axis=1)
    for _ in range(n - 1):
        i = int(d.argmax())
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(v - v[i], axis=1))
    return v[idx]


def box_surface(size, m: int = 2048) -> np.ndarray:
    """상자 표면 균일 샘플 (면적 비례)."""
    sx, sy, sz = (s / 2 for s in size)
    faces = [(sy * sz * 4, 0, sx), (sx * sz * 4, 1, sy), (sx * sy * 4, 2, sz)]
    area = np.array([f[0] for f in faces] * 2)
    pick = rng.choice(6, size=m, p=area / area.sum())
    p = rng.uniform(-1, 1, size=(m, 3)) * np.array([sx, sy, sz])
    for k in range(6):
        ax, half = faces[k % 3][1], faces[k % 3][2]
        p[pick == k, ax] = half if k < 3 else -half
    return p


def sphere_surface(r: float, m: int = 2048) -> np.ndarray:
    v = rng.normal(size=(m, 3))
    return r * v / np.linalg.norm(v, axis=1, keepdims=True)


def pcd_error(sp: np.ndarray, R_ref, t_ref, R_sim, t_sim, norm_radius=RHO, topk=TOPK) -> float:
    """_obj_pcd_match_error_exp의 오차 부분: 반지름 정규화 → (R_ref - R_sim) sp + dt → worst-k 평균."""
    if norm_radius > 0:
        sp = sp * (norm_radius / np.linalg.norm(sp, axis=1).max())
    disp = (t_ref - t_sim)[None] + sp @ (R_ref - R_sim).T
    d = np.linalg.norm(disp, axis=1)
    return float(np.sort(d)[-topk:].mean()) if topk > 0 else float(d.mean())


def main() -> None:
    out: dict = {}

    # V1 단일 점: 반지름 rho에 있는 점이 회전축에 수직일 때 30도 회전의 현 길이
    out["V1_chord_at_rho_30deg_cm"] = round(100 * 2 * RHO * math.sin(math.radians(15)), 3)

    # V2 모양별 30도 회전 worst-3 오차 분포 (무작위 축 2,000개)와 특정 축
    shapes = {
        "sphere_r4cm": sphere_surface(0.04),
        "cube_8cm": box_surface((0.08, 0.08, 0.08)),
        "stick_20x3x3cm": box_surface((0.20, 0.03, 0.03)),
        "plate_16x16x1cm": box_surface((0.16, 0.16, 0.01)),
        "offset_origin_cube": box_surface((0.06, 0.06, 0.06)) + np.array([0.05, 0.0, 0.0]),
    }
    I3, z3 = np.eye(3), np.zeros(3)
    axes = rng.normal(size=(2000, 3))
    v2 = {}
    for name, verts in shapes.items():
        sp = fps(verts, 128)
        errs = np.array([pcd_error(sp, rotmat(a, math.radians(30)), z3, I3, z3) for a in axes])
        errs_mean = np.array([pcd_error(sp, rotmat(a, math.radians(30)), z3, I3, z3, topk=0) for a in axes[:500]])
        row = {
            "worst3_cm": {"min": round(100 * errs.min(), 2), "median": round(100 * np.median(errs), 2),
                          "max": round(100 * errs.max(), 2)},
            "share_below_2cm": round(float((errs < 0.02).mean()), 3),
            "mean_all_points_cm_median": round(100 * float(np.median(errs_mean)), 2),
            "reward_at_median": round(math.exp(-np.median(errs) / SIGMA_OBJ), 3),
        }
        if name == "stick_20x3x3cm":
            row["long_axis_roll_30deg_cm"] = round(100 * pcd_error(sp, rotmat(np.array([1.0, 0, 0]), math.radians(30)), z3, I3, z3), 2)
            row["long_axis_roll_90deg_cm"] = round(100 * pcd_error(sp, rotmat(np.array([1.0, 0, 0]), math.radians(90)), z3, I3, z3), 2)
            row["angle_for_3cm_on_long_axis_deg"] = next(
                d for d in range(1, 181)
                if pcd_error(sp, rotmat(np.array([1.0, 0, 0]), math.radians(d)), z3, I3, z3) >= 0.03
            ) if pcd_error(sp, rotmat(np.array([1.0, 0, 0]), math.radians(180)), z3, I3, z3) >= 0.03 else None
        if name == "plate_16x16x1cm":
            row["normal_axis_spin_30deg_cm"] = round(100 * pcd_error(sp, rotmat(np.array([0, 0, 1.0]), math.radians(30)), z3, I3, z3), 2)
        v2[name] = row
    # 정규화를 끈 경우: 물체 크기에 따라 같은 30도가 다른 보상
    v2["no_normalization_30deg_random_axis_median_cm"] = {
        n: round(100 * float(np.median([pcd_error(fps(v, 128), rotmat(a, math.radians(30)), z3, I3, z3, norm_radius=0)
                                        for a in axes[:300]])), 2)
        for n, v in shapes.items()
    }
    out["V2_shape_and_axis_dependence"] = v2

    # V3 보상 ↔ 거리 환산
    out["V3_reward_conversions"] = {
        "r_obj_at_3cm": round(math.exp(-0.03 / SIGMA_OBJ), 3),
        "r_obj_at_10cm": round(math.exp(-0.10 / SIGMA_OBJ), 3),
        "early_reset_reward_0.6_means_error_cm": round(100 * SIGMA_OBJ * math.log(1 / 0.6), 2),
        "translation_equivalent_of_1rad_on_rho_cm": round(100 * 2 * RHO * math.sin(0.5), 2),
        "r_int_per_tip_at_excess_5mm": round(math.exp(-0.005 / SIGMA_INT), 3),
        "r_int_per_tip_at_excess_2cm": round(math.exp(-0.02 / SIGMA_INT), 3),
    }

    # V4 레퍼런스 거리: build_motion_pt.tip_contacts는 KD-tree로 가장 가까운 "꼭짓점" 거리를 씀.
    #     표면 위의 손끝(진짜 거리 0)이 얼마로 기록되는지 = 로봇이 벌칙 없이 떨어져 있을 수 있는 여유
    v4 = {}
    for n_verts in (500, 2000, 8000, 32000):
        verts = sphere_surface(0.04, n_verts)
        q = sphere_surface(0.04, 4000)  # 표면에 정확히 닿은 손끝
        d, _ = cKDTree(verts).query(q)
        v4[f"sphere_r4cm_{n_verts}_verts"] = {
            "slack_mm_mean": round(1000 * float(d.mean()), 2),
            "slack_mm_p95": round(1000 * float(np.quantile(d, 0.95)), 2),
            "slack_over_sigma_int_p95": round(float(np.quantile(d, 0.95)) / SIGMA_INT, 2),
        }
    out["V4_ref_distance_vertex_slack"] = v4
    # 레퍼런스 손끝이 물체에서 멀면(비접촉) 그 거리 안쪽은 전부 만점 → 스스로 꺼지는(gate) 항
    out["V4b_self_gating_example"] = {
        "ref_tip_far_5cm_robot_at_3cm_reward": round(math.exp(-max(0.0, 0.03 - 0.05) / SIGMA_INT), 3),
        "ref_tip_touch_0mm_robot_at_1cm_reward": round(math.exp(-max(0.0, 0.01 - 0.0) / SIGMA_INT), 3),
    }

    # V5 SDF 격자: 128^3, +-0.30 m (config/retarget/kinematic_retargeting.yaml sdf, commands.object.sdf)
    h = 0.60 / 127
    # 구(반지름 4 cm) SDF를 격자에 굽고 삼선형 보간 오차를 표면 근처(+-2 cm)에서 측정
    g = np.linspace(-0.30, 0.30, 128)


    def sdf_sphere(p, r=0.04):
        return np.linalg.norm(p, axis=-1) - r


    def trilinear(p):
        f = (p + 0.30) / h
        i0 = np.clip(np.floor(f).astype(int), 0, 126)
        t = f - i0
        acc = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = ((t[:, 0] if dx else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1]) * (t[:, 2] if dz else 1 - t[:, 2]))
                    corner = np.stack([g[i0[:, 0] + dx], g[i0[:, 1] + dy], g[i0[:, 2] + dz]], axis=1)
                    acc = acc + w * sdf_sphere(corner)
        return acc


    q = sphere_surface(1.0, 20000) * rng.uniform(0.02, 0.06, size=(20000, 1))
    err = np.abs(trilinear(q) - sdf_sphere(q))
    out["V5_sdf_grid"] = {
        "voxel_mm": round(1000 * h, 2),
        "trilinear_error_mm_p99_sphere_r4cm": round(1000 * float(np.quantile(err, 0.99)), 3),
        "note": "매끈한 형상에서는 보간 오차가 sigma_int(10 mm)보다 훨씬 작음. 얇은 벽(<1 voxel)은 별도 문제",
    }

    path = ROOT / "analysis" / "reward_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("wrote", path)


if __name__ == "__main__":
    main()
