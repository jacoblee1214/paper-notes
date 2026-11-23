"""토이: 물체 자세 추적 오차가 지역 근접도 γ와 단계 라벨에 주는 영향 (논문 부록 A.3 식 11–16, 부록 D.6).

논문은 가우시안 폭 σ와 단계 임계값 τ_palm을 적지 않았다(`04` C7). 여기서는 σ ∈ {5, 10, 20} mm, τ_palm = 0.5로 가정한다.
설정(가정): 반지름 40 mm 구 위에 손바닥 평면 조각(60×60 mm, 가장 가까운 표면에서 간격 g)을 둔다.
  - 손 점 400개, 물체 점 4000개 표면 표본. 최근접 거리 d_i (식 11), η_i = exp(−½ (d_i/σ)²) (식 13), γ = 평균 η (식 14).
  - 물체 자세 오차는 병진 오차만 넣는다: 벡터 크기의 제곱평균제곱근이 s (논문 그림 18의 ADD 8 mm, 50 mm와 비교).
  - 무시한 것: 회전 오차, SDF 관통 처리(식 12), 손가락 영역, 점군 표본 밀도.
측정: 간격 g를 0–60 mm에서 훑으며 (a) γ의 평균 절대 오차, (b) 단계 라벨 I(γ > τ)이 오차 없는 값과 달라지는 비율.
실행: python3 analysis/toy_proximity_noise.py   (numpy, scipy, 수십 초 이내) → analysis/toy_proximity_noise_results.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "toy_proximity_noise_results.json"
rng = np.random.default_rng(0)

R_OBJ = 40.0  # mm
N_OBJ, N_HAND = 4000, 400
TAU = 0.5


def sphere_points(n):
    v = rng.normal(size=(n, 3))
    return R_OBJ * v / np.linalg.norm(v, axis=1, keepdims=True)


def hand_patch(g):
    """z축 위쪽 구 표면(0, 0, R) 위로 간격 g만큼 뜬 60×60 mm 평면 조각."""
    xy = rng.uniform(-30, 30, size=(N_HAND, 2))
    z = np.full((N_HAND, 1), R_OBJ + g)
    return np.hstack([xy, z])


def gamma(hand, obj, sigma):
    d, _ = cKDTree(obj).query(hand)
    return float(np.exp(-0.5 * (d / sigma) ** 2).mean())


def main():
    obj = sphere_points(N_OBJ)
    gaps = np.arange(0, 61, 4.0)
    sig_list = [5.0, 10.0, 20.0]
    s_list = [0.0, 3.0, 6.0, 8.0, 15.0, 30.0, 50.0]
    n_rep = 30
    res = {"assumptions": {"sphere_radius_mm": R_OBJ, "tau_palm": TAU, "n_obj_pts": N_OBJ, "n_hand_pts": N_HAND, "n_rep": n_rep,
                           "pose_error": "translation only, RMS magnitude s"}, "gaps_mm": gaps.tolist(), "by_sigma": {}}
    for sigma in sig_list:
        base = np.array([gamma(hand_patch(g), obj, sigma) for g in gaps])
        row = {"gamma_clean": [round(float(x), 3) for x in base], "by_s": {}}
        for s in s_list:
            errs, flips = [], []
            for gi, g in enumerate(gaps):
                e, f = [], []
                for _ in range(n_rep):
                    t = rng.normal(size=3) * s / np.sqrt(3)
                    hand = hand_patch(g)
                    gm = gamma(hand, obj + t, sigma)
                    gc = gamma(hand, obj, sigma)
                    e.append(abs(gm - gc))
                    f.append((gm > TAU) != (gc > TAU))
                errs.append(np.mean(e))
                flips.append(np.mean(f))
            row["by_s"][str(s)] = {"mean_abs_err_over_gaps": round(float(np.mean(errs)), 3),
                                   "max_abs_err_over_gaps": round(float(np.max(errs)), 3),
                                   "phase_flip_rate_over_gaps": round(float(np.mean(flips)), 3),
                                   "phase_flip_rate_gap_le_12mm": round(float(np.mean([f for f, g in zip(flips, gaps) if g <= 12])), 3)}
        res["by_sigma"][str(sigma)] = row
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    for sigma, row in res["by_sigma"].items():
        print(f"sigma={sigma} mm")
        for s, v in row["by_s"].items():
            print(f"  s={s:>5} mm  mean|dγ|={v['mean_abs_err_over_gaps']:.3f}  max={v['max_abs_err_over_gaps']:.3f}  flip={v['phase_flip_rate_over_gaps']:.3f}  flip(g<=12)={v['phase_flip_rate_gap_le_12mm']:.3f}")


if __name__ == "__main__":
    main()
