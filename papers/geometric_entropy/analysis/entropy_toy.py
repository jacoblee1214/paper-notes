"""Geometric Entropy H_G (arXiv:2606.20871 식 1–3)를 합성 전이 궤적에 그대로 적용하는 CPU toy.

논문 파이프라인을 따른다: 호 길이 재표집(T=50) → 끝점 고정 → 목표 좌표계 회전 → 시작–끝 거리로 정규화 → 150차원 벡터 →
공분산 고유값 → d_eff, H_G = ln d_eff + ln Σλ.
목적: (1) 표 I의 순서(r↑, k=1이 k=0보다 낮음)가 이 정의에서 나오는지, (2) 샘플 수 N에 따른 편향,
(3) 노이즈·재표집 점 수 민감도, (4) kNN 엔트로피(논문 비교 기준선)의 k 민감도를 본다.
논문 코드가 없으므로 수치는 표 I와 같은 값이 아니다. 순서와 기전만 본다.

실행: python3 analysis/entropy_toy.py  → analysis/entropy_toy_results.json (1스레드, 약 1분 이내)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T_DEFAULT = 50
DZ = 0.2  # 경유점 높이 (논문 식 4의 Δz에 해당, 값은 임의로 정함)


def make_demos(M, r, k, rng, noise=0.0, T_pts=T_DEFAULT, yaw_frame=True, start_radius=1.0, fixed_start=False):
    """논문 식 4 형태의 경유점 w = g + [r cosθ, r sinθ, Δz]. 시작점은 목표 주변 원형 위치에서 뽑는다(외생 변동).
    k=0: θ ~ U(0, 2π), k>0: k개 등간격 θ 중 하나. 궤적은 시작 → 경유점 → 목표의 두 직선을 이어 붙인 것."""
    g = np.zeros(3)
    phi = rng.uniform(0, 2 * np.pi, M)           # 시작점 방위 (외생 변동)
    if fixed_start:
        phi = np.zeros(M)                        # 시작점 고정: 경유점 기하만 남긴다(격리 검사용)
    start = start_radius * np.stack([np.cos(phi), np.sin(phi), 0.3 * np.ones(M)], axis=1)  # 시작–목표 거리 ≈ start_radius
    if k == 0:
        theta = rng.uniform(0, 2 * np.pi, M)
    else:
        theta = 2 * np.pi * rng.integers(0, k, M) / k
    w = g + np.stack([r * np.cos(theta), r * np.sin(theta), DZ * np.ones(M)], axis=1)
    # 목표 좌표계 회전 (외생): 같은 회전을 궤적과 목표 프레임에 적용한 뒤 되돌리는 정렬을 검사하려고 넣는다
    if yaw_frame:
        yaw = rng.uniform(0, 2 * np.pi, M)
    else:
        yaw = np.zeros(M)
    trajs = np.zeros((M, T_pts, 3))
    for i in range(M):
        pts = np.stack([start[i], w[i], g])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        u = np.linspace(0, s[-1], T_pts)
        xyz = np.stack([np.interp(u, s, pts[:, d]) for d in range(3)], axis=1)
        if noise > 0:
            xyz = xyz + rng.normal(0, noise, xyz.shape)
        trajs[i] = xyz
    # 세계 좌표 회전: 목표 프레임 R_T (z축 yaw)
    c, s_ = np.cos(yaw), np.sin(yaw)
    R = np.zeros((M, 3, 3))
    R[:, 0, 0] = c; R[:, 0, 1] = -s_; R[:, 1, 0] = s_; R[:, 1, 1] = c; R[:, 2, 2] = 1
    world = np.einsum("mij,mtj->mti", R, trajs)
    return world, R, g


def align(world, R, g):
    """논문 III-C: 끝점 고정 → 목표 프레임으로 회전(R^T) → 시작점까지 거리로 나눔 → 평탄화.
    끝점을 원점으로 옮기면 ∥p̄1 − p̄T∥ = ∥p'_1∥ 이다."""
    anch = world - world[:, -1:, :]                       # p' = p − p_T
    local = np.einsum("mji,mtj->mti", R, anch)            # R^T p'
    disp = np.linalg.norm(local[:, 0, :], axis=1) + 1e-9  # 시작–끝 거리
    norm = local / disp[:, None, None]
    return norm.reshape(norm.shape[0], -1)


def h_g(X, unbiased=True, return_parts=False):
    """식 1–2. X: (M, D). λ는 표본 공분산 고유값(÷(M−1)). unbiased=False면 ÷M 대신 SVD 특이값²을 그대로 쓴다(코드 미공개라 비교용)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    M = X.shape[0]
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2 / ((M - 1) if unbiased else 1.0)
    lam = lam[lam > 1e-12 * lam.max()]
    p = lam / lam.sum()
    d_eff = np.exp(-np.sum(p * np.log(p)))
    H = np.log(d_eff) + np.log(lam.sum())
    if return_parts:
        return H, d_eff, lam.sum()
    return H


def digamma_int(n):
    """정수 n의 디감마 함수 ψ(n) = H_{n−1} − γ."""
    return sum(1.0 / j for j in range(1, n)) - 0.5772156649015329


def knn_kl_entropy(X, k=5):
    """Kozachenko–Leonenko 추정량(논문 Sec. V-D 기준선). H = ψ(M) − ψ(k) + ln V_D + (D/M) Σ ln ρ_i.
    k가 사실상의 대역폭 역할을 한다. ρ_i는 i에서 k번째 최근접 이웃 거리."""
    from math import lgamma, log, pi
    M, D = X.shape
    sq = np.sum(X ** 2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
    np.fill_diagonal(d2, np.inf)
    rho = np.sqrt(np.partition(d2, k - 1, axis=1)[:, k - 1].clip(min=1e-18))
    logVD = (D / 2) * log(pi) - lgamma(D / 2 + 1)
    return digamma_int(M) - digamma_int(k) + logVD + (D / M) * np.sum(np.log(rho))


def main():
    rng = np.random.default_rng(20260621)
    out = {"definition": "H_G = ln d_eff + ln sum(lambda), lambda = eig(cov) (paper Eq.1-2), T=50, D=150"}

    # T1: 표 I 순서 재현 여부 (N=1000, 3 시드)
    rs = [0.0, 0.01, 0.03, 0.05, 0.10]
    t1 = {}
    for k in [0, 1, 2, 3, 4]:
        row = {}
        for r in rs:
            vals = []
            for seed in range(3):
                rng = np.random.default_rng(1000 * seed + 17 * k + int(r * 1000))
                W, R, g = make_demos(1000, r, k, rng, noise=0.002)
                vals.append(h_g(align(W, R, g)))
            row[str(r)] = [round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)]
        t1[f"k={k}"] = row
    out["T1_H_vs_r_and_k_N1000"] = t1
    out["T1_note"] = ("시작점 외생 변동이 남아 있으면 r의 효과가 작다(시작–목표 거리 1). 해석은 T1c(격리) 참고.")

    # T1b: 시작–목표 거리(논문 미보고)를 0.3으로 줄이면 r의 효과가 커진다 (정규화 분모가 바뀜)
    t1b = {}
    for k in [0, 1]:
        row = {}
        for r in rs:
            rng = np.random.default_rng(4242 + 17 * k + int(r * 1000))
            W, R, g = make_demos(1000, r, k, rng, noise=0.002, start_radius=0.3)
            row[str(r)] = round(float(h_g(align(W, R, g))), 4)
        t1b[f"k={k}"] = row
    out["T1b_H_vs_r_startdist0.3"] = t1b

    # T1c: 시작점 고정(외생 변동 없음) → 경유점 기하만 남긴 격리 검사. k=2는 k=0보다 약 ln2 낮아야 하는가?
    t1c = {}
    for k in [0, 1, 2, 3]:
        rng = np.random.default_rng(777 + k)
        W, R, g = make_demos(3000, 0.10, k, rng, noise=0.0, start_radius=0.3, fixed_start=True, yaw_frame=False)
        H, d_eff, tr = h_g(align(W, R, g), return_parts=True)
        t1c[f"k={k}"] = {"H": round(float(H), 4), "d_eff": round(float(d_eff), 3), "ln_trace": round(float(np.log(tr)), 4)}
    out["T1c_isolated_waypoint_modes_r0.10"] = t1c

    # 논문 표 I 값과 비교할 자리: 같은 정의의 절대 값은 비교 불가(파이프라인 세부가 다름). 순서만 본다.
    # T2: N에 따른 편향 (같은 모집단, r=0.10 k=0, 노이즈 0.002)
    t2 = {}
    for normalize in [True, False]:
        rows = {}
        for M in [100, 200, 500, 1000, 2000]:
            vals = []
            for seed in range(5):
                rng = np.random.default_rng(7000 + seed)
                W, R, g = make_demos(M, 0.10, 0, rng, noise=0.002)
                vals.append(h_g(align(W, R, g), unbiased=normalize))
            rows[str(M)] = [round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)]
        t2["÷(M−1) (논문 식 그대로)" if normalize else "÷1 (특이값² 그대로)"] = rows
    out["T2_sample_count_bias_r0.10_k0"] = t2

    # 큰 N(모집단 근사) 대비 편향. 같은 모집단 자료로 계산
    rng = np.random.default_rng(99)
    Wbig, Rbig, gbig = make_demos(6000, 0.10, 0, rng, noise=0.002)
    Xbig = align(Wbig, Rbig, gbig)
    H_pop = h_g(Xbig)
    out["T2_population_H_M6000"] = round(float(H_pop), 4)

    # 무잡음 귀무: 형태 차이가 없는 자료(r=0)에서도 N에 따라 H가 움직이는지
    t2b = {}
    for M in [100, 200, 500, 1000]:
        rng = np.random.default_rng(31 + M)
        W, R, g = make_demos(M, 0.0, 0, rng, noise=0.0)
        t2b[str(M)] = round(float(h_g(align(W, R, g))), 4)
    out["T2b_null_r0_noise0_by_N"] = t2b

    # T2c: 노이즈가 큰 자료(스펙트럼이 평평함)에서 N에 따른 편향, 그리고 등방 가우시안 귀무(참값 알려짐)
    t2c = {}
    for M in [100, 200, 500, 1000, 3000]:
        rng = np.random.default_rng(900 + M)
        W, R, g = make_demos(M, 0.10, 0, rng, noise=0.03, start_radius=0.3)
        t2c[str(M)] = round(float(h_g(align(W, R, g))), 4)
    out["T2c_noise0.03_bias_by_N"] = t2c
    rng = np.random.default_rng(5)
    iso = {}
    for M in [100, 200, 500, 1000, 6000]:
        X = rng.normal(size=(M, 150))
        iso[str(M)] = round(float(h_g(X)), 4)
    out["T2d_isotropic_gaussian_D150_by_N"] = {"values": iso,
        "note": "모집단 공분산이 I이면 ln d_eff = ln 150, ln Σλ = ln 150이므로 참값은 약 10.02. M이 작을수록 낮게 편향된다."}

    # T3: 노이즈, 재표집 점 수 민감도 (N=500, r=0.10, k=0)
    t3 = {}
    for noise in [0.0, 0.002, 0.01, 0.03]:
        rng = np.random.default_rng(55)
        W, R, g = make_demos(500, 0.10, 0, rng, noise=noise)
        t3[f"noise={noise}"] = round(float(h_g(align(W, R, g))), 4)
    out["T3_noise_sensitivity"] = t3
    t3b = {}
    for T_pts in [25, 50, 100]:
        rng = np.random.default_rng(56)
        W, R, g = make_demos(500, 0.10, 0, rng, noise=0.002, T_pts=T_pts)
        t3b[f"T={T_pts}"] = round(float(h_g(align(W, R, g))), 4)
    out["T3_resample_points_sensitivity"] = t3b

    # T4: 프레임 회전 불변성 검사 (같은 자료, 목표 프레임 yaw만 바꿔도 H가 같아야 한다)
    rng = np.random.default_rng(88)
    W, R, g = make_demos(500, 0.10, 0, rng, noise=0.0, yaw_frame=True)
    X1 = align(W, R, g)
    W2 = W  # 같은 세계 좌표 궤적, 목표 프레임만 다른 yaw로 가정
    ang = 1.234
    Rz = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    R2 = np.einsum("ij,mjk->mik", Rz, R)  # 같은 목표 프레임에 전역 회전이 들어간 경우
    X2 = align(np.einsum("ij,mtj->mti", Rz, W), R2, g)
    out["T4_frame_rotation_invariance"] = {
        "H_original": round(float(h_g(X1)), 6),
        "H_after_global_rotation": round(float(h_g(X2)), 6),
        "note": "전역 회전을 궤적과 목표 프레임에 함께 적용하면 H가 같아야 한다(차이는 부동소수점 수준).",
    }

    # T5: kNN 엔트로피의 k 민감도 (기준선, 같은 디스크립터)
    t5 = {}
    for M in [100, 300, 1000]:
        rng = np.random.default_rng(404 + M)
        W, R, g = make_demos(M, 0.10, 0, rng, noise=0.002)
        X = align(W, R, g)
        row = {}
        for kk in [3, 5, 10, 20]:
            if kk < M:
                row[f"k={kk}"] = round(float(knn_kl_entropy(X, kk)), 3)
        row["H_G"] = round(float(h_g(X)), 3)
        t5[f"M={M}"] = row
    out["T5_knn_entropy_k_sensitivity_baseline"] = t5

    path = HERE / "entropy_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
