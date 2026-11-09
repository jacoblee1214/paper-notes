"""환경 강성 라벨 추정 toy (CPU numpy, 1스레드, 수 초).

(A) 식 (1)–(2) 검증: 섹터별 f̃f̃ᵀ+εI의 로그 평균 → exp. 닫힌 해 S = log ε·I + (1/m)Σ w_i u_i u_iᵀ 와 일치하는지,
    공분산 방식(Duan 2018식 E[ff^T])과 섹터 방식의 방향 편향, ε에 따른 이방성 비율을 본다.
(B) 식립성(identifiability) 점검: 라벨이 접촉 상태 c에 의존할 때, 영상이 θ만 보여 주면 c는 복원되지 않는다.
    kNN 회귀로 영상에서 라벨을 예측했을 때의 오차 바닥(irreducible error)을 재고, (θ, c)를 직접 줄 때와 비교한다.

실행: python3 analysis/label_toy.py → analysis/label_toy_results.json
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "label_toy_results.json"
rng = np.random.default_rng(0)
NORMAL = np.array([0.0, 0.0, 1.0])


def fib_sphere(m):
    i = np.arange(m) + 0.5
    phi = np.arccos(1 - 2 * i / m)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def sector_forces(F, centers):
    """힘 벡터를 가장 가까운 섹터 중심에 배정하고, 섹터별 (평균 방향, 95퍼센타일 크기)를 낸다."""
    mags = np.linalg.norm(F, axis=1)
    dirs = F / mags[:, None]
    idx = np.argmax(dirs @ centers.T, axis=1)
    reps = []
    for j in range(len(centers)):
        sel = idx == j
        if not sel.any():
            continue
        mbar = dirs[sel].mean(0)
        u = mbar / np.linalg.norm(mbar)
        p95 = np.percentile(mags[sel], 95)
        reps.append((u, p95))
    return reps


def paper_label(reps, eps):
    """식 (1)–(2): S = mean_i log(f̃f̃ᵀ + εI), K ∝ exp(S/2). 행렬 로그를 직접 계산한다."""
    logs = []
    for u, p95 in reps:
        A = p95 ** 2 * np.outer(u, u) + eps * np.eye(3)
        w, V = np.linalg.eigh(A)
        logs.append(V @ np.diag(np.log(w)) @ V.T)
    S = np.mean(logs, 0)
    w, V = np.linalg.eigh(S)
    K = V @ np.diag(np.exp(w / 2)) @ V.T
    return K, S


def closed_form_S(reps, eps):
    """닫힌 해: S = log ε I + (1/m) Σ (log(|f̃|²+ε) − log ε) u uᵀ"""
    S = np.log(eps) * np.eye(3)
    for u, p95 in reps:
        S = S + (np.log(p95 ** 2 + eps) - np.log(eps)) * np.outer(u, u) / len(reps)
    return S


def euclid_label(reps, eps):
    M = np.mean([p95 ** 2 * np.outer(u, u) + eps * np.eye(3) for u, p95 in reps], 0)
    return M


def cov_label(F):
    """Duan et al. (2018)식 공분산 방식: K² ∝ E[f fᵀ]"""
    C = F.T @ F / len(F)
    w, V = np.linalg.eigh(C)
    return V @ np.diag(np.sqrt(np.maximum(w, 0))) @ V.T


def sample_forces(n, skew):
    """접촉력 샘플. 크기는 법선(z축)에 가까울수록 크다: |f| = 20 N · cos⁴α + 잡음 (α = 법선과의 각).
    skew=0: 방향을 구면에 균등하게 샘플. skew=1: 접선 쪽 방향을 80% 샘플(닦기처럼 접선 접촉이 잦은 수집)."""
    if skew:
        n_tan = int(n * 0.8)
        tang = rng.normal(size=(n_tan, 3)) * np.array([1, 1, 0.1])
        dirs_tan = tang / np.linalg.norm(tang, axis=1)[:, None]
        rest = rng.normal(size=(n - n_tan, 3))
        dirs = np.vstack([dirs_tan, rest / np.linalg.norm(rest, axis=1)[:, None]])
    else:
        d = rng.normal(size=(n, 3))
        dirs = d / np.linalg.norm(d, axis=1)[:, None]
    cosa = np.clip(dirs @ NORMAL, 0, 1)
    mag = 20.0 * cosa ** 4 + rng.uniform(0.05, 0.3, n)
    return dirs * mag[:, None]


def eig_summary(K):
    w, V = np.linalg.eigh((K + K.T) / 2)
    top = V[:, np.argmax(w)]
    ang = np.degrees(np.arccos(min(1.0, abs(top @ NORMAL))))
    return {"angle_top_to_normal_deg": round(float(ang), 2), "aniso_ratio": round(float(w.max() / w.min()), 2),
            "det": float(np.linalg.det(K))}


def part_a():
    centers = fib_sphere(200)
    out = {}
    for skew in (0, 1):
        F = sample_forces(4000, skew)
        reps = sector_forces(F, centers)
        eps = 0.1
        K_paper, S = paper_label(reps, eps)
        S_cf = closed_form_S(reps, eps)
        out[f"skew{skew}"] = {
            "m_valid": len(reps),
            "closed_form_max_abs_err": float(np.abs(S - S_cf).max()),
            "paper_log_label": eig_summary(K_paper),
            "covariance_duan_label": eig_summary(cov_label(F)),
        }
    # ε에 따른 이방성 비율 (논문은 ε 값을 밝히지 않음)
    F = sample_forces(4000, 1)
    reps = sector_forces(F, centers)
    out["eps_sweep_skew1"] = {}
    for eps in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        K, _ = paper_label(reps, eps)
        out["eps_sweep_skew1"][f"{eps:g}"] = eig_summary(K)["aniso_ratio"]
    # 행렬 로그 평균과 유클리드 평균의 부피 차이 (팽창 효과 점검)
    # 섹터별 K_i = A_i^{1/2}: 유클리드 평균 대 로그 평균(=exp(mean log K_i), 논문 식 (1)–(2)의 형태)
    eps = 0.1
    Ks = []
    for u, p95 in reps:
        A = p95 ** 2 * np.outer(u, u) + eps * np.eye(3)
        w, V = np.linalg.eigh(A)
        Ks.append(V @ np.diag(np.sqrt(w)) @ V.T)
    K_euc = np.mean(Ks, 0)
    logs = []
    for K in Ks:
        w, V = np.linalg.eigh(K)
        logs.append(V @ np.diag(np.log(w)) @ V.T)
    lw, lV = np.linalg.eigh(np.mean(logs, 0))
    K_log = lV @ np.diag(np.exp(lw)) @ lV.T
    out["swelling_check"] = {
        "det_euclid_mean_K": float(np.linalg.det(K_euc)),
        "det_logeuclid_mean_K": float(np.linalg.det(K_log)),
        "det_ratio_euclid_over_log": float(np.linalg.det(K_euc) / np.linalg.det(K_log)),
    }
    return out


def part_b(n_train=3000, n_test=1000, k=30):
    """상태 s = (θ, d). 접촉 c = [d < 0.3]. 라벨 K = 접촉이면 R(θ)diag(300, 3000)R(θ)ᵀ, 아니면 1000·I.
    영상 특징: 벽 각도 2θ는 항상 보인다. 거리 d는 (a) 보이지 않거나 (b) σ=0.02/0.1 잡음으로 보인다고 가정.
    kNN(k=30) 회귀로 영상에서 라벨을 예측했을 때의 평균 오차(‖K‖_F=1 정규화)를 잰다."""
    def sample(n):
        th = rng.uniform(-np.pi / 6, np.pi / 6, n)
        d = rng.uniform(0, 1, n)
        c = (d < 0.3).astype(int)
        R = np.stack([np.stack([np.cos(th), -np.sin(th)], 1), np.stack([np.sin(th), np.cos(th)], 1)], 1)
        Ka = R @ np.diag([300.0, 3000.0]) @ np.transpose(R, (0, 2, 1))
        Kiso = np.tile(np.eye(2) * 1000.0, (n, 1, 1))
        K = np.where(c[:, None, None] == 1, Ka, Kiso)
        K = K / np.linalg.norm(K, axis=(1, 2), keepdims=True)
        ang = np.column_stack([np.cos(2 * th), np.sin(2 * th)])
        return th, d, c, K, ang

    th_tr, d_tr, c_tr, K_tr, ang_tr = sample(n_train)
    th_te, d_te, c_te, K_te, ang_te = sample(n_test)
    Ytr, Yte = K_tr.reshape(n_train, -1), K_te.reshape(n_test, -1)

    def knn_err(Xtr, Xte):
        d2 = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
        idx = np.argpartition(d2, k, axis=1)[:, :k]
        pred = Ytr[idx].mean(1)
        return float(np.sqrt(((pred - Yte) ** 2).sum(1)).mean())

    res = {}
    res["angle_only_no_distance"] = round(knn_err(ang_tr, ang_te), 4)
    for sd in (0.02, 0.1):
        Xtr = np.column_stack([ang_tr, d_tr + rng.normal(0, sd, n_train)])
        Xte = np.column_stack([ang_te, d_te + rng.normal(0, sd, n_test)])
        res[f"angle_plus_distance_noise{sd}"] = round(knn_err(Xtr, Xte), 4)
    res["oracle_theta_and_contact_c"] = round(knn_err(np.column_stack([ang_tr, c_tr * 2.0]),
                                                      np.column_stack([ang_te, c_te * 2.0])), 4)
    res["note"] = ("kNN k=30, 학습 3000, 시험 1000, 시드 0. 접촉 c = [d<0.3]. "
                   "거리 d가 영상에 없으면 접촉 상태가 라벨에서 평균으로 흐려진다(식별 불가).")
    return res


def main():
    res = {"partA_label_estimators": part_a(), "partB_identifiability": part_b()}
    OUT.write_text(json.dumps(res, indent=1))
    a = res["partA_label_estimators"]
    for skew in ("skew0", "skew1"):
        r = a[skew]
        print(f"A {skew} m_valid={r['m_valid']} closed-form err={r['closed_form_max_abs_err']:.1e}")
        for key in ("paper_log_label", "covariance_duan_label"):
            print(f"   {key:<24} {r[key]}")
    print("A eps sweep (aniso ratio):", a["eps_sweep_skew1"])
    print("A swelling:", a["swelling_check"])
    print("B:", res["partB_identifiability"])


if __name__ == "__main__":
    main()
