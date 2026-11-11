"""추가 검증 1: FSWO 점수식에 마찰 원뿔·중력·크기 정규화 토크를 더해 판정 변화를 측정.

변형 (모두 같은 무작위 3접점 배치에서 비교)
- V_code : 공식 코드식 그대로. 토크 가중 α = 10 m⁻¹, 법선력만, 중력 없음 (fswo_toy.py의 F1과 같은 정의)
- V_norm : 토크 가중을 α = 1/R (R = 외접 반지름)으로 바꿈 (크기 정규화), 법선력만
- V_fric : V_norm + 마찰 원뿔(μ = 0.5, 4면 근사)
- V_grav : V_fric + 중력 바이어스 (무게 G = 0.5 단위, 물체 무게중심에 작용, 세계 -z 방향)
- V_code_grav : 토크 가중은 V_code(α = 10), 마찰·중력은 V_grav와 같음 (크기 의존이 남는지 확인용)

점수 = min_j min_{λ≥0} ‖A_j λ − b_j‖  (pivot j의 법선력을 1로 고정, 나머지 접점은 마찰 생성자 λ ≥ 0)
  A_j의 열: 나머지 접점의 생성자 힘 [f; α (p_i − p_j) × f]
  b_j = −[n_j; α (p_j − c) × ... ]  대신 −(pivot 법선 wrench + 중력 wrench)
  중력 wrench = [F_g; α (c − p_j) × F_g],  F_g = (0, 0, −G) (세계 좌표; 물체 자세가 무작위이므로 방향도 무작위)
- 통과: 점수 < 0.15 (공식 코드 기준)
- 가정: μ = 0.5, G = 0.5 (단위 없음, 법선력 1 대비), pivot 접점의 접선력은 쓰지 않음(보수적), 직육면체 3접점.
- 크기 불변성 검사: V_norm과 V_fric에서 같은 배치를 2배·7.5배 키웠을 때 판정이 바뀌는 비율.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr

OUT_JSON = Path(__file__).with_name("extra_fswo_variants_results.json")
THRESH = 0.15
MU = 0.5
G_BIAS = 0.5
N_PER = 800
SHAPES = {"cube": (1.0, 1.0, 1.0), "box": (1.0, 0.6, 0.4), "tool": (1.0, 0.15, 0.15)}
SIZES = [0.04, 0.08, 0.16, 0.30]  # 가장 긴 변 (m)
VARIANTS = {
    # name: (alpha_mode, friction, gravity)
    "V_code": ("code", False, False),
    "V_norm": ("norm", False, False),
    "V_fric": ("norm", True, False),
    "V_grav": ("norm", True, True),
    "V_code_grav": ("code", True, True),
}


def box_contacts(rng, dims, k=3):
    half = np.asarray(dims) / 2
    area = np.array([dims[1] * dims[2], dims[0] * dims[2], dims[0] * dims[1]] * 2)
    faces = rng.choice(6, size=k, p=area / area.sum())
    pos, nrm = np.empty((k, 3)), np.zeros((k, 3))
    for i, f in enumerate(faces):
        ax, sgn = f % 3, (1 if f < 3 else -1)
        p = rng.uniform(-half, half)
        p[ax] = sgn * half[ax]
        pos[i] = p
        nrm[i, ax] = -sgn  # 안쪽 법선
    return pos, nrm


def tangents(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(n, a); t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return t1, t2


def generators(n, friction):
    if not friction:
        return [n]
    t1, t2 = tangents(n)
    return [n + MU * t1, n - MU * t1, n + MU * t2, n - MU * t2]


def score(pos, nrm, dims_radius, alpha_mode, friction, gravity, rng_unused=None):
    alpha = 10.0 if alpha_mode == "code" else 1.0 / dims_radius
    k = len(pos)
    com = np.zeros(3)
    Fg = np.array([0.0, 0.0, -G_BIAS]) if gravity else np.zeros(3)
    best = np.inf
    for j in range(k):
        others = [i for i in range(k) if i != j]
        cols = []
        for i in others:
            for g in generators(nrm[i], friction):
                cols.append(np.concatenate([g, alpha * np.cross(pos[i] - pos[j], g)]))
        A = np.array(cols).T
        w_piv = np.concatenate([nrm[j], np.zeros(3)])
        w_g = np.concatenate([Fg, alpha * np.cross(com - pos[j], Fg)])
        b = -(w_piv + w_g)
        sol = lsq_linear(A, b, bounds=(0.0, np.inf), method="bvls")
        best = min(best, float(np.linalg.norm(A @ sol.x - b)))
    return best


def pass_rate(scores):
    return float(np.mean(scores < THRESH))


def main():
    t0 = time.time()
    rng = np.random.default_rng(20261111)
    out = {"setup": {"threshold": THRESH, "mu": MU, "G_bias": G_BIAS, "n_per_shape_size": N_PER,
                     "sizes_m": SIZES, "shapes": {k: list(v) for k, v in SHAPES.items()}, "seed": 20261111},
           "pass_rate_by_size": {}, "invariance": {}, "rank_and_flip_8cm": {}}

    # 같은 배치 집합을 모든 변형에 쓴다 (짝 비교). 형상·크기별로 배치를 한 번 뽑고 키운다.
    base_sets = {}
    for sname, rel in SHAPES.items():
        base_sets[sname] = [box_contacts(rng, tuple(0.04 * r for r in rel)) for _ in range(N_PER)]

    scores = {}  # (variant, shape, size) -> array
    for sname, rel in SHAPES.items():
        for s in SIZES:
            dims = np.array([s * r for r in rel])
            R = 0.5 * float(np.linalg.norm(dims))  # 외접 반지름 (상자 대각선의 절반)
            scale = s / 0.04
            for vname, (am, fr, gr) in VARIANTS.items():
                vals = np.empty(N_PER)
                for idx, (p, n) in enumerate(base_sets[sname]):
                    vals[idx] = score(p * scale, n, R, am, fr, gr)
                scores[(vname, sname, s)] = vals

    # 크기별 통과율 (형상 세 개를 합침)
    for vname in VARIANTS:
        row = {}
        for s in SIZES:
            allv = np.concatenate([scores[(vname, sn, s)] for sn in SHAPES])
            row[f"{int(round(s*100))}cm"] = round(pass_rate(allv), 3)
        out["pass_rate_by_size"][vname] = row

    # 크기 불변성: 정규화 변형은 크기가 바뀌어도 판정이 같아야 한다 (4 cm 기준 대비)
    for vname in ("V_code", "V_norm", "V_fric", "V_grav", "V_code_grav"):
        flips = {}
        for s in SIZES[1:]:
            a = np.concatenate([scores[(vname, sn, SIZES[0])] for sn in SHAPES]) < THRESH
            b = np.concatenate([scores[(vname, sn, s)] for sn in SHAPES]) < THRESH
            flips[f"{int(round(s*100))}cm_vs_4cm"] = round(float(np.mean(a != b)), 3)
        out["invariance"][vname] = {"decision_flip_rate": flips,
                                    "median_score_ratio_30cm_over_4cm": round(float(np.median(
                                        np.concatenate([scores[(vname, sn, SIZES[-1])] for sn in SHAPES]) /
                                        np.maximum(np.concatenate([scores[(vname, sn, SIZES[0])] for sn in SHAPES]), 1e-9))), 3)}

    # 8 cm에서 변형 간 순위 상관과 판정 뒤집힘 (V_code 기준)
    ref = np.concatenate([scores[("V_code", sn, 0.08)] for sn in SHAPES])
    for vname in VARIANTS:
        if vname == "V_code":
            continue
        v = np.concatenate([scores[(vname, sn, 0.08)] for sn in SHAPES])
        out["rank_and_flip_8cm"][vname] = {
            "spearman_vs_V_code": round(float(spearmanr(ref, v).correlation), 3),
            "decision_flip_vs_V_code": round(float(np.mean((ref < THRESH) != (v < THRESH))), 3),
        }

    # 임계값 보정: α = 1/R 변형은 점수 척도가 달라 0.15가 같은 뜻이 아니다.
    # 8 cm에서 V_code의 통과율과 같아지도록 변형별 임계값을 정하고(분위수), 그 임계값으로 크기별 통과율을 다시 잰다.
    ref_rate = pass_rate(ref)
    out["rate_matched"] = {}
    for vname in VARIANTS:
        v8 = np.concatenate([scores[(vname, sn, 0.08)] for sn in SHAPES])
        thr = float(np.quantile(v8, ref_rate))
        row = {"threshold": round(thr, 4)}
        for s in SIZES:
            allv = np.concatenate([scores[(vname, sn, s)] for sn in SHAPES])
            row[f"pass_{int(round(s*100))}cm"] = round(float(np.mean(allv < thr)), 3)
        v8f = v8 < thr
        out["rate_matched"][vname] = row
        if vname != "V_code":
            out["rate_matched"][vname]["decision_flip_vs_V_code_8cm"] = round(float(np.mean((ref < THRESH) != v8f)), 3)
    # 같은 임계값 기준으로 V_code가 뒤집는 판정 (8 cm 기준 0.15 대비 V_code의 임계값 자체는 그대로)

    out["runtime_s"] = round(time.time() - t0, 1)
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
