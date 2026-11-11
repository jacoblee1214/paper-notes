"""추가 검증 (GraspSense R1): 두 단계 광선 두께 추정이 알려진 컵 벽 두께를 맞히는가, 테두리 보정이 필요한 조건은 무엇인가.

논문 Sec. III-C의 힘 지도 알고리즘 중 두께 추정 부분(1, 2, 3단계)을 numpy로 구현한다 (CPU, 단일 스레드, 수 초).
  - 1단계 바깥 광선: 각 셀(높이 층 l, 각도 a)에서 바깥 표면 점 p1과 법선 n1.
  - 2단계 안쪽 광선: p1에서 -n1(평면 성분) 방향으로 쏘아 안쪽 벽 점 p2. tau = |p2 - p1|.
  - 3단계 테두리 보정: 층 경계 근처에서 하위 층을 더 샘플링한 평균 (논문 설명을 따른 가정 구현).

모델: 절차적 컵 (논문에 치수 없음, 모두 가정).
  - 외경 R_o = 40 mm, 높이 H = 100 mm. 바깥 표면과 안쪽 벽은 n-gon (메쉬 해상도 n).
  - 벽 두께 t_w = 0.4 mm, 바닥 두께 t_b = 1.5 mm (높이 0–3 mm), 테두리 두께 t_r = 1.2 mm (높이 H - h_r 이상).
  - 정답 두께 프로파일 t(z)는 정해져 있어 두께 추정 오차를 바로 잴 수 있다.
  - 광선-다각형 교차는 해석적으로 계산한다 (선분과 광선의 교차, 가장 가까운 양의 해).

실험:
  E1. 메쉬 해상도 n in {8, 16, 32, 64, 128}에서 벽 두께 상대 오차 (각도별 분포). 방향 평균(±5°, ±10°)이 바이어스를 줄이는지.
      해석 예상: 평행한 면 사이 거리가 t cos(pi/n)이므로 상대 오차 ≈ cos(pi/n) - 1.
  E2. 높이 층 수 L in {10, 20, 50, 100}, 테두리 높이 h_r in {2, 6} mm에서 테두리 두께 적분 오차.
      층 중심만 보는 기본 방식과, 경계 층에서 하위 샘플 8개를 평균하는 보정 방식을 비교한다.
  E3. 바이어스 보정 cos(pi/n)으로 나눈 값 (해석적 보정)이 E1의 바이어스를 없애는지.

실행: python3 analysis/extra_ray_thickness.py -> analysis/extra_ray_thickness_results.json
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "extra_ray_thickness_results.json"
R_O = 40.0
H = 100.0
T_W, T_B, T_R = 0.4, 1.5, 1.2
H_B = 3.0
N_LIST = [8, 16, 32, 64, 128]
A_BINS = 36
JITTERS_DEG = [0.0, 5.0, 10.0]


def t_profile(z, h_r):
    t = np.full_like(z, T_W)
    t[z < H_B] = T_B
    t[z >= H - h_r] = T_R
    return t


def trap(f, z):
    return float(np.sum((f[1:] + f[:-1]) * 0.5 * np.diff(z)))


def polygon(radius, n):
    th = 2 * np.pi * np.arange(n) / n
    return np.stack([radius * np.cos(th), radius * np.sin(th)], axis=1)


def ray_polygon_hit(o, d, V):
    """o, d: (m, 2). V: (n, 2) 닫힌 다각형 꼭짓점. 각 광선의 가장 가까운 양의 교차 거리."""
    V2 = np.roll(V, -1, axis=0)
    e = V2 - V  # (n, 2)
    # 광선 o + s d 와 선분 V + u e 의 교차
    denom = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]  # (m, n)
    w = V[None, :, :] - o[:, None, :]  # (m, n, 2)
    s = (w[..., 0] * e[None, :, 1] - w[..., 1] * e[None, :, 0]) / np.where(np.abs(denom) > 1e-15, denom, np.nan)
    u = (w[..., 0] * d[:, None, 1] - w[..., 1] * d[:, None, 0]) / np.where(np.abs(denom) > 1e-15, denom, np.nan)
    ok = (s > 1e-9) & (u >= -1e-9) & (u <= 1 + 1e-9)
    s = np.where(ok, s, np.inf)
    return s.min(axis=1)


def thickness_at(t_z, n, angles_deg_jitter=(0.0,)):
    """한 높이의 두께 추정. 바깥 다각형은 반지름 R_o, 안쪽은 R_o - t 의 같은 n-gon."""
    outer = polygon(R_O, n)
    inner = polygon(R_O - t_z, n)
    phi_c = np.pi / n  # 면 중심 각 (면 법선 방향 = 꼭짓점 사이 중간)
    a_mid = 2 * np.pi * (np.arange(A_BINS) + 0.5) / A_BINS
    est = []
    for j in angles_deg_jitter:
        a = a_mid + np.deg2rad(j)
        # 바깥 점: 중심에서 각 a로 쏜 광선과 바깥 다각형의 교점
        dvec = np.stack([np.cos(a), np.sin(a)], axis=1)
        r = ray_polygon_hit(np.zeros((A_BINS, 2)), dvec, outer)
        p1 = dvec * r[:, None]
        # 면 법선: p1이 속한 면의 중심 법선 (가장 가까운 면 중심 각)
        face_idx = np.floor(np.mod(a, 2 * np.pi) / (2 * np.pi / n) + 0.0).astype(int) % n
        face_ang = (face_idx + 0.5) * 2 * np.pi / n
        n1 = np.stack([np.cos(face_ang), np.sin(face_ang)], axis=1)
        # 안쪽 광선: -n1 방향
        t_hat = ray_polygon_hit(p1, -n1, inner)
        est.append(t_hat)
    return np.mean(est, axis=0)


def e1_mesh_resolution():
    out = {}
    for n in N_LIST:
        row = {}
        for jit_set, name in [((0.0,), "single_ray"), ((0.0, 5.0, -5.0, 10.0, -10.0), "avg_pm10deg")]:
            est = thickness_at(T_W, n, jit_set)
            rel = est / T_W - 1.0
            row[name] = {"mean_rel_err": round(float(rel.mean()), 5),
                         "min_rel_err": round(float(rel.min()), 5),
                         "max_rel_err": round(float(rel.max()), 5),
                         "max_abs_rel_err": round(float(np.abs(rel).max()), 5)}
        row["cos_pi_over_n_minus_1"] = round(float(np.cos(np.pi / n) - 1.0), 5)
        # E3: 해석 보정 (cos(pi/n) 으로 나눔)
        est = thickness_at(T_W, n, (0.0,))
        corr = est / np.cos(np.pi / n)
        rel_c = corr / T_W - 1.0
        row["corrected_by_cos"] = {"mean_rel_err": round(float(rel_c.mean()), 5),
                                   "max_abs_rel_err": round(float(np.abs(rel_c).max()), 5)}
        out[f"n={n}"] = row
    return out


def e2_layers_rim():
    out = {}
    for h_r in (2.0, 5.0, 6.0):
        truth_int = None
        for L in (10, 20, 50, 100):
            dz = H / L
            zc = (np.arange(L) + 0.5) * dz
            # 기본: 층 중심 한 점
            t_plain = t_profile(zc, h_r)
            # 보정: 경계 층(테두리 경계를 포함)은 하위 샘플 8개 평균
            t_ref = t_plain.copy()
            for i in range(L):
                lo, hi = i * dz, (i + 1) * dz
                if (lo < H - h_r < hi) or (lo < H_B < hi):
                    sub = lo + (np.arange(8) + 0.5) * dz / 8
                    t_ref[i] = t_profile(sub, h_r).mean()
            # 적분 정답 (테두리 구간 z in [H-h_r, H])
            zf = np.linspace(0, H, 200001)
            tf = t_profile(zf, h_r)
            rim_mask = zf >= H - h_r
            truth_rim = trap(tf[rim_mask], zf[rim_mask])
            # 테두리 구간과 겹치는 길이만큼 층 두께를 곱해 더한다 (층 중심이 테두리 밖이어도 겹치면 포함)
            lo = np.arange(L) * dz
            hi = lo + dz
            overlap = np.clip(np.minimum(hi, H) - np.maximum(lo, H - h_r), 0.0, None)
            rim_plain = float(np.sum(t_plain * overlap))
            rim_ref = float(np.sum(t_ref * overlap))
            # 두께 적분 전체 오차 (상대)
            total_truth = trap(tf, zf)
            total_plain = np.sum(t_plain) * dz
            total_ref = np.sum(t_ref) * dz
            out[f"h_r={h_r:g}mm,L={L}"] = {
                "layer_height_mm": round(dz, 2),
                "rim_integral_rel_err_plain": round(float((rim_plain - truth_rim) / truth_rim), 4),
                "rim_integral_rel_err_refined": round(float((rim_ref - truth_rim) / truth_rim), 4),
                "total_integral_rel_err_plain": round(float((total_plain - total_truth) / total_truth), 5),
                "total_integral_rel_err_refined": round(float((total_ref - total_truth) / total_truth), 5),
            }
    return out


def main():
    out = {"settings": {"outer_radius_mm": R_O, "height_mm": H, "wall_mm": T_W, "base_mm": T_B,
                        "rim_mm": T_R, "base_height_mm": H_B, "angle_bins": A_BINS,
                        "note": "절차적 컵, 치수는 가정. 논문 치수 없음. 두께 추정 알고리즘만 재구현"},
           "E1_mesh_resolution_wall_thickness": e1_mesh_resolution(),
           "E2_layer_count_and_rim": e2_layers_rim()}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for k, v in out["E1_mesh_resolution_wall_thickness"].items():
        print(k, v["single_ray"]["mean_rel_err"], v["single_ray"]["max_abs_rel_err"], "avg", v["avg_pm10deg"]["mean_rel_err"],
              "cos-theory", v["cos_pi_over_n_minus_1"], "corrected", v["corrected_by_cos"]["max_abs_rel_err"])
    for k, v in out["E2_layer_count_and_rim"].items():
        print(k, v)


if __name__ == "__main__":
    main()
