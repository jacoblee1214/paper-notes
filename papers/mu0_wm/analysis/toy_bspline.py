"""토이: μ0의 B-spline 궤적 표현(D = 10개 제어점, H = 32 스텝)이 얼마나 잘 맞고, 무엇을 잃는지 합성 궤적으로 잰다.

검증하려는 주장: 논문 3.2절, 부록 B.2의 "cubic B-spline, D=10, 능형 λ_bsp=0.2, 간결성·매끄러움·쉬운 디노이징",
부록 D.2의 "B-spline 적합과 원본의 재구성 오차는 약 0.02 DTW". 공식 데이터가 없어 합성 궤적(정규화 단위, 전체 변위 약 1)에서 본다.
실제 학습 데이터의 값은 아니므로 해석에는 "표현이 허용하는 오차의 크기"로만 쓴다.

비교하는 기저 두 가지:
  A) 'bezier3': 내부 매듭 중복도 3 (0, 0,0,0, 1/3 ×3, 2/3 ×3, 1,1,1,1) = 세 구간 3차 Bézier (C0 연결). 공개 코드의 매듭 벡터가 이 형태다.
  B) 'cubic_c2': 내부 매듭 중복도 1인 균일 3차 B-spline, 제어점 10개 (C2 연속).
적합은 행 가중 능형 최소제곱: min ||B P - Y||² + ||λ Γ P||² (Γ는 이웃 제어점 1차 차분, 논문 식 4의 형태; 앵커 행 t=0 포함).
출력: toy_bspline_results.json. 실행: python3 analysis/toy_bspline.py (numpy, 수 초)
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_bspline_results.json"
H, D, DEG = 32, 10, 3


def basis_matrix(knots, n_ctrl, t):
    """Cox-de Boor. t: (T,), 반환 (T, n_ctrl). 마지막 구간은 닫힌 구간으로 처리."""
    knots = np.asarray(knots, float)
    N = np.zeros((len(t), n_ctrl, DEG + 1))
    for i in range(n_ctrl):
        last = i == n_ctrl - 1
        N[:, i, 0] = np.where((knots[i] <= t) & ((t < knots[i + 1]) | (last & (t <= knots[i + 1]))), 1.0, 0.0)
    for k in range(1, DEG + 1):
        for i in range(n_ctrl):
            d1 = knots[i + k] - knots[i]
            d2 = knots[i + k + 1] - knots[i + 1]
            a = (t - knots[i]) / d1 * N[:, i, k - 1] if d1 > 0 else 0.0
            b = (knots[i + k + 1] - t) / d2 * N[:, i + 1, k - 1] if (d2 > 0 and i + 1 < n_ctrl) else 0.0
            N[:, i, k] = a + b
    return N[:, :, DEG]


KNOTS = {
    "bezier3": [0, 0, 0, 0, 1 / 3, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 2 / 3, 1, 1, 1, 1],
    "cubic_c2": [0, 0, 0, 0] + [k / 7 for k in range(1, 7)] + [1, 1, 1, 1],
}
T_GRID = np.arange(H + 1) / H            # 앵커 t=0 + 미래 32스텝
BASIS = {k: basis_matrix(v, D, T_GRID) for k, v in KNOTS.items()}
GAMMA = np.zeros((D - 1, D))
for i in range(D - 1):
    GAMMA[i, i], GAMMA[i, i + 1] = -1.0, 1.0


def fit(B, Y, lam):
    """Y: (H+1, 3). 반환 제어점 (D, 3)."""
    A = np.vstack([B, lam * GAMMA])
    rhs = np.vstack([Y, np.zeros((D - 1, 3))])
    return np.linalg.lstsq(A, rhs, rcond=None)[0]


def min_jerk(t):
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def make_traj(kind, rng):
    t = T_GRID
    a = rng.normal(size=3)
    a = a / np.linalg.norm(a)
    if kind == "reach":                 # 최소 저크 한 번
        s = min_jerk(t)
    elif kind == "pick_place":          # 도달-정지-이동(두 번 최소 저크, 중간 정지)
        s = np.where(t < 0.45, 0.5 * min_jerk(np.clip(t / 0.45, 0, 1)),
                     np.where(t < 0.55, 0.5, 0.5 + 0.5 * min_jerk(np.clip((t - 0.55) / 0.45, 0, 1))))
    elif kind == "oscillate":           # 붓기·흔들기 같은 진동(1.5 주기)
        s = np.sin(2 * np.pi * 1.5 * t) * 0.5 + 0.5 * min_jerk(t)
    else:
        raise ValueError(kind)
    return s[:, None] * a[None, :] * rng.uniform(0.6, 1.4)


def metrics(B, Y_clean, Y_obs, lam):
    P = fit(B, Y_obs, lam)
    Yh = B @ P
    return P, Yh


def tangent_jump(P, knots_name):
    """내부 매듭에서 좌우 속도 차이(제어점 기준). Bézier 이어붙임은 (P3-P2) 와 (P4-P3)의 차이로 본다."""
    if knots_name != "bezier3":
        return 0.0
    d = np.diff(P, axis=0)                   # (9, 3), 구간 속도 ∝ 제어점 차
    return float(np.mean([np.linalg.norm(d[2] - d[3]), np.linalg.norm(d[5] - d[6])]))


rng = np.random.default_rng(0)
N_TRAJ = 300
res = {}
for kind in ["reach", "pick_place", "oscillate"]:
    for noise in [0.0, 0.02, 0.05]:
        for basis_name in ["bezier3", "cubic_c2"]:
            for lam in [0.0, 0.2]:
                e_clean, e_obs, jumps, rel_jump = [], [], [], []
                local = np.random.default_rng(1)        # 같은 궤적과 잡음을 모든 설정에 쓴다
                for _ in range(N_TRAJ):
                    Y = make_traj(kind, local)
                    obs = Y + noise * local.normal(size=Y.shape)
                    obs[0] = 0.0 * obs[0]               # 앵커는 0 (anchor-relative)
                    Y = Y - Y[0]
                    P, Yh = metrics(BASIS[basis_name], Y, obs, lam)
                    e_clean.append(np.sqrt(np.mean((Yh[1:] - Y[1:]) ** 2)))
                    e_obs.append(np.sqrt(np.mean((Yh[1:] - obs[1:]) ** 2)))
                    jumps.append(tangent_jump(P, basis_name))
                    scale = np.mean(np.linalg.norm(np.diff(P, axis=0), axis=1)) + 1e-9
                    rel_jump.append(tangent_jump(P, basis_name) / scale)
                key = f"{kind}|noise{noise}|{basis_name}|lam{lam}"
                res[key] = {"rmse_to_clean": float(np.mean(e_clean)), "rmse_to_observed": float(np.mean(e_obs)),
                            "tangent_jump_rel": float(np.mean(rel_jump))}

summary = {
    "setting": {"H": H, "D": D, "N_traj_per_cell": N_TRAJ, "noise_unit": "정규화 단위(전체 변위 약 1)",
                "output_dim_raw": H * 3, "output_dim_spline": D * 3, "compression": H / D},
    "results": res,
    "note": "합성 궤적. 실제 TraceExtract 궤적의 재구성 오차(논문 D.2의 약 0.02 DTW)와 직접 비교하지 않는다.",
}
OUT.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")

print("kind | noise | basis | lam -> rmse_to_clean, rmse_to_observed, rel tangent jump")
for k, v in res.items():
    print(k, {a: round(b, 4) for a, b in v.items()})
