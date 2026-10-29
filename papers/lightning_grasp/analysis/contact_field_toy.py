"""Contact field 표본 밀도 toy (CPU, numpy, 단일 스레드). 실행: python3 analysis/contact_field_toy.py

공식 코드의 contact field 생성 설정을 따릅니다.
- 관절 배치 80,000개를 관절 한계 안에서 균일 추출 (contact_field.generate_contact_field n_iter=80000)
- 패치마다 keyvector 6개 (mesh_analyzer.generate_mesh_group n_keyvector=6)
- 1 cm 정육면체 box로 덮고, leaf 하나에 최대 256개 (LBVHS2ContactFieldAccelStructure box_w=0.01, max_leaf_size=256)

손가락은 Allegro와 비슷한 치수와 관절 범위로 둔 가정입니다(실제 URDF가 아님).
논문 Sec. 4.4의 "대부분의 box는 벡터 50개 이하"와, 256개 상한에 걸리는 box의 비율을 봅니다.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)
N_CFG, N_KEY, BOX, LEAF = 80000, 6, 0.01, 256

# 가정: Allegro 비슷한 손가락 (m, rad). 0번 관절 = 벌림(z축), 1–3번 = 굽힘(y축)
LIMITS = np.array([[-0.47, 0.47], [-0.196, 1.61], [-0.174, 1.709], [-0.227, 1.618]])
L1, L2, L3 = 0.054, 0.0384, 0.0424  # 근위, 중위, 원위(손끝 중심까지)
TIP_R = 0.012


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3)); R[..., 0, 0] = c; R[..., 0, 1] = -s; R[..., 1, 0] = s; R[..., 1, 1] = c; R[..., 2, 2] = 1
    return R


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3)); R[..., 0, 0] = c; R[..., 0, 2] = s; R[..., 2, 0] = -s; R[..., 2, 2] = c; R[..., 1, 1] = 1
    return R


def fk_tip(q):
    """q (N, 4) → 원위 링크 좌표계 회전 R (N,3,3)과 손끝 중심 c (N,3). 손가락은 +z 방향으로 뻗음."""
    R0 = rot_z(q[:, 0])
    R1 = R0 @ rot_y(q[:, 1])
    p = np.einsum("nij,j->ni", R1, [0, 0, L1])
    R2 = R1 @ rot_y(q[:, 2])
    p = p + np.einsum("nij,j->ni", R2, [0, 0, L2])
    R3 = R2 @ rot_y(q[:, 3])
    c = p + np.einsum("nij,j->ni", R3, [0, 0, L3])
    return R3, c


def patch_keyvectors(axis, n=N_KEY, cone=math.radians(36), radius=TIP_R):
    """손끝 구면 위 한 패치: 중심 방향 axis에서 36° 안, 6개 (위치는 손끝 중심 기준, 법선 = 바깥)."""
    axis = axis / np.linalg.norm(axis)
    out = []
    while len(out) < n:
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        if math.acos(np.clip(v @ axis, -1, 1)) < cone / 2:
            out.append(v)
    u = np.array(out)
    return radius * u, u


def occupancy(points):
    cells = np.floor((points - points.min(0)) / BOX).astype(np.int64)
    key = cells[:, 0] * 1_000_000 + cells[:, 1] * 1_000 + cells[:, 2]
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    return uniq, inv, counts


def main() -> None:
    t0 = time.time()
    q = rng.uniform(LIMITS[:, 0], LIMITS[:, 1], size=(N_CFG, 4))
    R, c = fk_tip(q)
    out: dict = {"assumption": "Allegro-like finger (not the real URDF); fingertip patches on a 12 mm sphere"}
    patches = {"palmar": np.array([1.0, 0, 0.3]), "side": np.array([0, 1.0, 0.3]), "tip": np.array([0.2, 0, 1.0])}
    for name, ax in patches.items():
        pos_l, nrm_l = patch_keyvectors(ax)
        pos = (np.einsum("nij,kj->nki", R, pos_l) + c[:, None, :]).reshape(-1, 3)
        nrm = np.einsum("nij,kj->nki", R, nrm_l).reshape(-1, 3)
        uniq, inv, counts = occupancy(pos)
        kept = np.minimum(counts, LEAF).sum()
        # 상한에 걸린 box 안에서 무작위 256개만 남길 때 법선 덮임 (0.2 rad 안에 남은 벡터가 있는가)
        sat = np.where(counts > LEAF)[0]
        cov = []
        for b in sat[:200]:
            idx = np.where(inv == b)[0]
            keep = rng.choice(idx, LEAF, replace=False)
            dots = nrm[idx] @ nrm[keep].T
            cov.append(float((dots.max(1) > math.cos(0.2)).mean()))
        out[name] = {
            "vectors": int(len(pos)),
            "occupied_boxes": int(len(uniq)),
            "mean_per_box": round(float(counts.mean()), 1),
            "median_per_box": int(np.median(counts)),
            "p90_per_box": int(np.quantile(counts, 0.9)),
            "max_per_box": int(counts.max()),
            "share_boxes_over_50": round(float((counts > 50).mean()), 3),
            "share_boxes_over_256": round(float((counts > LEAF).mean()), 3),
            "share_vectors_dropped_if_capped": round(float(1 - kept / len(pos)), 3),
            "normal_coverage_in_capped_boxes_0p2rad": round(float(np.mean(cov)), 4) if cov else None,
            "workspace_volume_cm3": int(len(uniq)),
        }
    # 패치 하나에 필요한 관절 배치 수: box당 평균 50개가 되려면
    pal = out["palmar"]
    out["configs_for_mean_50_per_box"] = int(50 * pal["occupied_boxes"] / N_KEY)
    out["runtime_s"] = round(time.time() - t0, 1)

    path = ROOT / "analysis" / "contact_field_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("wrote", path)


if __name__ == "__main__":
    main()
