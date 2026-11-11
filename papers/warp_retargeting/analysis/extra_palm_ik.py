"""추가 검증 1: 손바닥 등식과 SP3 팔꿈치 해의 존재·유일성을 numpy 닫힌 형식으로 검산.

모형 (가정: 3D 팔 한 쪽, 어깨 s 고정, 팔꿈치 e, 손목 w, 손바닥 중심 = w + L·u, u = 전완 단위방향)
- 로봇 링크: ℓ_SE = 0.30 m (상완), ℓ_EW = 0.27 m (전완), 손 길이 L = 0.10 m (가정).
- 손바닥 등식: 손바닥 목표 P*에 대해 e + K·u = P*, K = ℓ_EW + L (w = e + ℓ_EW u, 손바닥 = w + L u).
  상완 길이 |e − s| = ℓ_SE 와 |u| = 1 로부터 (q = P* − s):
     (i)  u·n = (q·n)/K                 (팔꿈치는 사람 회전면 n 위에 있다)
     (ii) q·u = (|q|² + K² − ℓ_SE²)/(2K) =: β
  평면 성분 x = (β − α q·n)/|q_⊥| 로 두면 u = α n + x q̂ ± y (n × q̂), y² = 1 − α² − x².  최대 두 해.
- 사람 쪽 반평면 규칙: 두 해는 직선 s→P* 에 대해 회전면 안에서 거울상이다. 팔꿈치가 사람 팔꿈치와 같은
  쪽(부호 ((e−s) × q)·n)에 있는 해를 고른다.
- 사람 입력: 상완·전완 0.33·0.30 m, 무작위 팔꿈치 자세 (n, ê 정함). 목표는 로봇 도달 구 안 무작위 20,000개.

검산 항목
 (a) 선택된 해가 세 원래 조건을 만족: 손바닥 오차, 상완·전완 길이 오차, |u| 오차, 회전면 이탈
 (b) 실근 개수 0/1/2 비율, 실근이 있으면 삼각 부등식 [|ℓ_SE−ℓ_EW|, ℓ_SE+ℓ_EW]를 만족하는지
 (c) 유일성: 실근이 있는 목표 중 사람 쪽 조건을 만족하는 해가 정확히 하나인 비율, 두 해가 반대 쪽일 때의 비율,
     선택 해의 부호 값이 0에 가까운(|값| < 1 cm) 경우의 비율
 (d) 링크 비율 민감도: 로봇 상완을 0.9·1.0·1.1배로 바꿀 때 실근 비율, 손바닥 오차, 사람 팔꿈치 방향과의 각도
결과: analysis/extra_palm_ik_results.json
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import json
import time
from pathlib import Path

import numpy as np

OUT_JSON = Path(__file__).with_name("extra_palm_ik_results.json")
L_SE, L_EW, L_HAND = 0.30, 0.27, 0.10
K = L_EW + L_HAND
N_TARGETS = 20000
SEED = 20261111


def solve_palm(P, s, n, e_h, l_se):
    """P (N,3) 손바닥 목표, n 회전면 법선, e_h 사람 팔꿈치 위치(회전면 안).
    반환: E (N,2,3) 팔꿈치 후보, U (N,2,3) 전완 방향, V (N,2) 실근 여부, S (N,2) 직선 s→P* 쪽 부호,
          n_ok (N,) 사람 쪽 조건을 만족하는 실근 수, pick (N,) 선택 인덱스"""
    q = P - s
    qn = q @ n
    qq = np.sum(q * q, 1)
    beta = (qq + K * K - l_se * l_se) / (2 * K)
    alpha = qn / K
    q_p = q - qn[:, None] * n
    Dp = np.maximum(np.linalg.norm(q_p, axis=1), 1e-12)
    qhat = q_p / Dp[:, None]
    m_hat = np.cross(np.broadcast_to(n, qhat.shape), qhat)
    x = (beta - alpha * qn) / Dp
    y2 = 1 - alpha * alpha - x * x
    y = np.sqrt(np.clip(y2, 0, None))
    V = np.stack([y2 >= 0, y2 >= 0], 1)
    U = np.stack([alpha[:, None] * n + x[:, None] * qhat + sg * y[:, None] * m_hat for sg in (1.0, -1.0)], 1)
    E = P[:, None, :] - K * U
    # 직선 s→P* 쪽 부호 (회전면 안에서 ((e−s) × q)·n)
    S = np.sign(np.sum(np.cross(E - s, q[:, None, :]) * n, axis=-1))
    ref = np.sign(np.sum(np.cross(np.broadcast_to(e_h - s, q.shape), q) * n, axis=-1))
    cand = V & (S == ref[:, None])
    n_ok = cand.sum(1)
    pick = np.argmax(cand, axis=1)  # 조건을 만족하는 첫 해 (n_ok == 1이면 유일)
    return E, U, V, S, ref, n_ok, pick


def random_unit(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    s = np.zeros(3)
    n_h = random_unit(rng, 1)[0]
    e_dir = random_unit(rng, 1)[0]
    e_dir -= (e_dir @ n_h) * n_h
    e_dir /= np.linalg.norm(e_dir)
    e_h = s + 0.33 * e_dir                                   # 사람 팔꿈치 (회전면 안)
    dirs = random_unit(rng, N_TARGETS)
    dist = rng.uniform(0.02, 0.70, N_TARGETS)
    P = dirs * dist[:, None]
    ii = np.arange(N_TARGETS)

    E, U, V, S, ref, n_ok, pick = solve_palm(P, s, n_h, e_h, L_SE)
    e_p, u_p = E[ii, pick], U[ii, pick]
    has = V.any(1)
    w_p = e_p + L_EW * u_p
    palm_err = np.linalg.norm(w_p + L_HAND * u_p - P, axis=1)
    upper_err = np.abs(np.linalg.norm(e_p - s, axis=1) - L_SE)
    fore_err = np.abs(np.linalg.norm(w_p - e_p, axis=1) - L_EW)
    u_err = np.abs(np.linalg.norm(u_p, axis=1) - 1)
    plane_err = np.abs((e_p - s) @ n_h)
    out = {"setup": {"L_SE": L_SE, "L_EW": L_EW, "L_hand": L_HAND, "n_targets": N_TARGETS, "seed": SEED,
                     "target_dist_range_m": [0.02, 0.70], "human_elbow_dist_m": 0.33}}
    out["a_residuals_selected_root"] = {
        "n_with_real_root": int(has.sum()),
        "palm_err_max_m": float(palm_err[has].max()), "palm_err_median_m": float(np.median(palm_err[has])),
        "upper_arm_err_max_m": float(upper_err[has].max()), "forearm_err_max_m": float(fore_err[has].max()),
        "u_norm_err_max": float(u_err[has].max()), "elbow_plane_err_max_m": float(plane_err[has].max()),
    }

    nroots = V.sum(1)
    d_ws = np.linalg.norm(w_p - s, axis=1)
    tri = (d_ws >= abs(L_SE - L_EW) - 1e-9) & (d_ws <= L_SE + L_EW + 1e-9)
    out["b_root_count"] = {
        "frac_0_real_roots": round(float(np.mean(nroots == 0)), 4),
        "frac_1_real_root": round(float(np.mean(nroots == 1)), 4),
        "frac_2_real_roots": round(float(np.mean(nroots == 2)), 4),
        "triangle_holds_for_selected_root_given_real_root": round(float(np.mean(tri[has])), 4),
    }

    two = has & (nroots == 2)
    opp = np.mean(S[two, 0] != S[two, 1]) if two.any() else float("nan")
    out["c_uniqueness"] = {
        "frac_exactly_one_satisfies_human_side_given_real_root": round(float(np.mean(n_ok[has] == 1)), 4),
        "frac_zero_satisfy_human_side_given_real_root": round(float(np.mean(n_ok[has] == 0)), 4),
        "frac_two_satisfy_human_side_given_real_root": round(float(np.mean(n_ok[has] == 2)), 4),
        "frac_two_root_cases_with_roots_on_opposite_sides": round(float(opp), 4),
        "n_two_root_cases": int(two.sum()),
    }
    # 사람 쪽 조건을 만족하는 해의 "여유": 팔꿈치가 직선 s→P*에서 떨어진 거리 (cm). 0에 가까우면 규칙이 불안정
    # 팔꿈치에서 직선 s→P*까지의 거리 (회전면 안): |((e−s) × q)·n| / |q_⊥|
    qv = P - s
    dist_line = np.abs(np.sum(np.cross(E - s, qv[:, None, :]) * n_h, axis=-1))
    dist_line = dist_line / np.maximum(np.linalg.norm(qv, axis=1), 1e-9)[:, None]
    sel_d = dist_line[ii, pick]
    out["c_uniqueness"]["frac_selected_elbow_within_1cm_of_palm_axis"] = round(float(np.mean(sel_d[has] < 0.01)), 4)

    rows = {}
    for scale in (0.9, 1.0, 1.1):
        E2, U2, V2, S2, ref2, n2, pick2 = solve_palm(P, s, n_h, e_h, L_SE * scale)
        e2, u2 = E2[ii, pick2], U2[ii, pick2]
        ok = n2 >= 1
        pe = np.linalg.norm(e2 + (L_EW + L_HAND) * u2 - P, axis=1)
        cosang = np.clip(((e2 - s) / np.linalg.norm(e2 - s, axis=1, keepdims=True)) @ e_dir, -1, 1)
        ang = np.degrees(np.arccos(cosang))
        rows[f"upper_arm_x{scale}"] = {
            "frac_with_selected_solution": round(float(ok.mean()), 4),
            "palm_err_max_m": float(pe[ok].max()) if ok.any() else None,
            "elbow_angle_from_human_median_deg": round(float(np.median(ang[ok])), 2) if ok.any() else None,
        }
    out["d_link_ratio_sensitivity"] = rows
    out["runtime_s"] = round(time.time() - t0, 2)
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
