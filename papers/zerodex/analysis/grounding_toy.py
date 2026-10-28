"""다시점 VLM 그라운딩 toy (CPU, numpy). 실행: python3 analysis/grounding_toy.py

논문 Sec. III-B(Eq. 5–6)의 "시점 쌍 삼각측량 + 재투영 합의"를, 같은 VLM 2D 출력을 단일 RGB-D 시점에서
깊이로 역투영하는 MOKA식 방법과 비교합니다. 실제 VLM 대신 다음 2D 출력 모델을 씁니다.

- 장면마다 고정된 시점별 의미 편향(같은 이미지면 같은 곳을 가리킴): N(0, 4 px)
- 실행마다 달라지는 흔들림(Gemini 샘플링): N(0, 3 px)
- 가끔 엉뚱한 곳을 가리키는 이상치: 질의당 5%, 40–100 px
- 목표점은 높이가 있는 용기의 윗면: 절반은 테두리에서 2 cm 이상 안쪽, 절반은 테두리 0–6 mm 안쪽
  (단일 시점 깊이 역투영은 픽셀이 테두리를 넘어가면 탁자 깊이로 튐)
- 카메라 배치(스테레오 쌍 기선 12 cm 포함)는 논문에 없어서 가정한 값입니다

합의 방법은 두 가지로 구현합니다.
- literal: Eq. 6 그대로. 지지 수가 같으면 먼저 나온 쌍을 씀(동률 규칙은 본문에 없음)
- refit: 표준 RANSAC처럼 최다 지지 후보의 inlier 시점 전체로 다시 삼각측량하고, 동률은 재투영 오차 합으로 가름

G1: Table IV처럼 10 장면 × 8회 반복의 σ(중심 기준 RMS), 최댓값, 중앙값
G2: 4개 시점 가운데 이상치 시점 수 k에 따른 오차 (합의 vs 전체 최소제곱)
G3: 합의 임계값 ε_tri(본문에 값 없음)에 대한 민감도
"""
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
F, CX, CY, W, H = 900.0, 640.0, 360.0, 1280, 720
RUNS, SCENES = 8, 10


def look_at(pos, target=np.array([0.0, 0.0, 0.05])):
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])  # world → cam
    return R, -R @ pos


# 4개 보정 카메라 (스테레오 쌍 포함, Sec. IV-A)
CAMS = [look_at(np.array(p)) for p in ([0.85, -0.05, 0.65], [0.85, 0.07, 0.65], [-0.2, 0.9, 0.7], [-0.6, -0.7, 0.75])]


def project(X, cam):
    R, t = cam
    c = R @ X + t
    return np.array([F * c[0] / c[2] + CX, F * c[1] / c[2] + CY])


def ray(p, cam):
    R, t = cam
    d = R.T @ np.array([(p[0] - CX) / F, (p[1] - CY) / F, 1.0])
    o = -R.T @ t
    return o, d / np.linalg.norm(d)


def triangulate(ps, cams):
    """DLT 최소제곱 (2개 이상 시점)."""
    K = np.array([[F, 0, CX], [0, F, CY], [0, 0, 1.0]])
    A = []
    for p, (R, t) in zip(ps, cams):
        P = K @ np.hstack([R, t[:, None]])
        A.append(p[0] * P[2] - P[0])
        A.append(p[1] * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3]


def consensus(ps, eps, refit=False):
    """Eq. 5–6: 시점 쌍마다 삼각측량, 재투영 오차 ≤ eps인 시점 수가 가장 많은 후보."""
    best, best_key, best_in = None, None, None
    for a, b in itertools.combinations(range(len(CAMS)), 2):
        X = triangulate([ps[a], ps[b]], [CAMS[a], CAMS[b]])
        err = np.array([np.linalg.norm(project(X, c) - p) for p, c in zip(ps, CAMS)])
        inl = err <= eps
        # literal: 지지 수만 비교(동률이면 먼저 나온 쌍). refit: 동률을 inlier 재투영 오차 합으로 가름
        key = (int(inl.sum()), -float(err[inl].sum()) if refit else 0.0)
        if best_key is None or key > best_key:
            best, best_key, best_in = X, key, inl
    if refit and best_in.sum() >= 2:
        idx = np.flatnonzero(best_in)
        best = triangulate([ps[i] for i in idx], [CAMS[i] for i in idx])
    return best


def ray_box(o, d, lo, hi):
    """광선과 축정렬 상자(용기 외곽)의 첫 교점 t, 없으면 inf."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - o) / d, (hi - o) / d
    tmin, tmax = np.max(np.minimum(t1, t2)), np.min(np.maximum(t1, t2))
    return tmin if tmax >= max(tmin, 0) else np.inf


def rgbd_deproject(p, cam, box, rng, sigma_z=0.003):
    """단일 시점 깊이 역투영: 픽셀 광선이 처음 맞는 표면(용기 또는 탁자)의 깊이 + 깊이 잡음."""
    o, d = ray(p, cam)
    t_box = ray_box(o, d, *box)
    t_tab = -o[2] / d[2] if d[2] < 0 else np.inf
    t = min(t_box, t_tab)
    return o + d * (t + rng.normal(0, sigma_z))


def vlm(X, bias, rng, jitter=3.0, p_out=0.05):
    ps = []
    for v, c in enumerate(CAMS):
        p = project(X, c) + bias[v] + rng.normal(0, jitter, 2)
        if rng.random() < p_out:
            ang = rng.uniform(0, 2 * np.pi)
            p = p + rng.uniform(40, 100) * np.array([np.cos(ang), np.sin(ang)])
        ps.append(p)
    return ps


def scene(rng):
    h = rng.uniform(0.08, 0.15)
    c = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)])
    lo, hi = np.array([c[0] - 0.12, c[1] - 0.09, 0.0]), np.array([c[0] + 0.12, c[1] + 0.09, h])
    # 목표: 절반은 윗면 안쪽(테두리에서 2 cm 이상), 절반은 테두리 0–6 mm 안쪽
    edge = rng.random() < 0.5
    side = rng.integers(4)
    u = rng.uniform(0.2, 0.8)
    inset = rng.uniform(0.0, 0.006) if edge else rng.uniform(0.02, 0.05)
    x = lo[0] + u * (hi[0] - lo[0]) if side < 2 else (lo[0] + inset if side == 2 else hi[0] - inset)
    y = lo[1] + u * (hi[1] - lo[1]) if side >= 2 else (lo[1] + inset if side == 0 else hi[1] - inset)
    return np.array([x, y, h]), (lo, hi)


def sigma(points):
    P = np.array(points)
    return float(np.sqrt(np.mean(np.sum((P - P.mean(0)) ** 2, axis=1))))


def main() -> None:
    rng = np.random.default_rng(0)
    out: dict = {}

    # G1 반복성 (10 장면 × 8회, Table IV와 같은 설계), 여러 seed로 반복해 분포를 봄
    rows = {"consensus_literal": [], "consensus_refit": [], "all_view_lsq": [], "single_rgbd": []}
    acc = {k: [] for k in rows}
    for rep in range(20):
        for _ in range(SCENES):
            X, box = scene(rng)
            bias = rng.normal(0, 4.0, (len(CAMS), 2))
            ref = int(rng.integers(len(CAMS)))
            res = {k: [] for k in rows}
            for _ in range(RUNS):
                ps = vlm(X, bias, rng)
                res["consensus_literal"].append(consensus(ps, 10.0))
                res["consensus_refit"].append(consensus(ps, 10.0, refit=True))
                res["all_view_lsq"].append(triangulate(ps, CAMS))
                res["single_rgbd"].append(rgbd_deproject(ps[ref], CAMS[ref], box, rng))
            for k in rows:
                rows[k].append(1000 * sigma(res[k]))
                acc[k].append(1000 * float(np.linalg.norm(np.mean(res[k], 0) - X)))
    g1 = {}
    for k, s in rows.items():
        s = np.array(s).reshape(20, SCENES)
        g1[k] = {
            "sigma_mean_mm (Table IV σ)": round(float(s.mean()), 2),
            "sigma_worst_scene_mm (δ_max, 해석 A)": round(float(s.max(1).mean()), 1),
            "sigma_median_mm": round(float(np.median(s)), 2),
            "share_scenes_sigma_gt_10mm": round(float((s > 10).mean()), 3),
            "accuracy_error_median_mm": round(float(np.median(acc[k])), 1),
        }
    out["G1_repeatability_10x8"] = g1
    out["G1_sigma_samples_mm"] = {k: [round(v, 2) for v in s] for k, s in rows.items()}  # 그래프용

    # G2 이상치 시점 수 (편향·흔들림은 그대로, 이상치만 고정 개수로 주입)
    g2 = {}
    for k_out in range(4):
        errs = {"consensus_literal": [], "consensus_refit": [], "all_view_lsq": []}
        for _ in range(400):
            X, _ = scene(rng)
            ps = [project(X, c) + rng.normal(0, 4.0, 2) + rng.normal(0, 3.0, 2) for c in CAMS]
            for v in rng.choice(len(CAMS), k_out, replace=False):
                ang = rng.uniform(0, 2 * np.pi)
                ps[v] = ps[v] + rng.uniform(40, 100) * np.array([np.cos(ang), np.sin(ang)])
            errs["consensus_literal"].append(1000 * np.linalg.norm(consensus(ps, 10.0) - X))
            errs["consensus_refit"].append(1000 * np.linalg.norm(consensus(ps, 10.0, refit=True) - X))
            errs["all_view_lsq"].append(1000 * np.linalg.norm(triangulate(ps, CAMS) - X))
        g2[f"{k_out}_bad_views"] = {k: {"median_mm": round(float(np.median(v)), 1),
                                        "p90_mm": round(float(np.quantile(v, 0.9)), 1)} for k, v in errs.items()}
    out["G2_outlier_views"] = g2

    # G3 ε_tri 민감도 (이상치 시점 1개)
    g3 = {}
    for eps in (3.0, 5.0, 10.0, 20.0, 40.0):
        e, er = [], []
        for _ in range(300):
            X, _ = scene(rng)
            ps = [project(X, c) + rng.normal(0, 4.0, 2) + rng.normal(0, 3.0, 2) for c in CAMS]
            v = int(rng.integers(len(CAMS)))
            ps[v] = ps[v] + 70 * np.array([1.0, 0.0])
            e.append(1000 * np.linalg.norm(consensus(ps, eps) - X))
            er.append(1000 * np.linalg.norm(consensus(ps, eps, refit=True) - X))
        g3[f"eps_{eps:g}px"] = {"literal": {"median_mm": round(float(np.median(e)), 1), "p90_mm": round(float(np.quantile(e, 0.9)), 1)},
                                "refit": {"median_mm": round(float(np.median(er)), 1), "p90_mm": round(float(np.quantile(er, 0.9)), 1)}}
    out["G3_eps_tri_sensitivity_1_bad_view"] = g3

    # 참고: 1 m 거리에서 1 px ≈ 1.1 mm (f = 900 px)
    # v1 부록 D.1: ε_tri = 20 px @ 640×480. 같은 수평 화각(이 toy: 2·atan(640/900) ≈ 70°)이면 toy 해상도로 약 39 px
    f_v1 = 320 / np.tan(np.arctan(CX / F))
    out["note"] = {"mm_per_px_at_1m": round(1000 / F, 2), "cameras": len(CAMS), "pairs": len(CAMS) * (len(CAMS) - 1) // 2,
                   "v1_eps_tri_20px_at_640x480_in_toy_px": round(20 * F / f_v1, 1)}

    path = ROOT / "analysis" / "grounding_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("wrote", path)


if __name__ == "__main__":
    main()
