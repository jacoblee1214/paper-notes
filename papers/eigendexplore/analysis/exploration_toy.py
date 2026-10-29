"""탐색 분포 toy (CPU, numpy, 단일 스레드). 실행: python3 analysis/exploration_toy.py

공개 Sharpa PCA(정규화 관절 좌표, d=22, k=9)를 사람 자세 공분산으로 쓰고, 세 가지를 본다.
- T1 SPIDER식 표본 최적화(식 3): N개 섭동 → 비용 하위 10% 가중 평균으로 목표 자세 추적.
      목표 방향이 사람 분포를 따르는지(human), 등방인지(isotropic), 일부가 부분공간 밖인지(mixed)에 따라
      IID / EigenDEXplore(총 분산 맞춤, 논문 서술) / EigenDEXplore 2배(SimToolReal preset) /
      Eigen만 / 무작위 상관(같은 스펙트럼, 무작위 방향) / 가파른 스펙트럼(SPIDER σ_e ∝ λ)을 비교
- T2 희소 성공: 한 번의 표본이 목표 방향 u로 a(IID σ 단위) 이상 나아갈 꼬리 확률 P(<ε,u> ≥ a)
- T3 정책 기울기(REINFORCE)의 평균 기울기는 이차 보상에서 Σ와 무관함을 수치로 확인하고,
      배치 추정치가 참 기울기와 이루는 cos 유사도를 비교
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import numpy as np
from scipy.special import ndtr

import pca_check as pc

ROOT = Path(__file__).resolve().parents[1]
H = pc.hand("sharpa")
D, K = len(H["half"]), 9
Bf = H["t"]["components"] / H["half"]
C_H = Bf.T @ np.diag(H["t"]["component_std"] ** 2) @ Bf          # 사람 자세 공분산 (정규화 좌표)


def dist(name: str, rng_dirs: np.random.Generator) -> np.ndarray:
    """관절당 평균 분산이 1인 공분산 (2배 설정만 2). Joint-Space σ=1과 같은 눈금."""
    I = np.eye(D)
    if name == "iid":
        return I
    if name.startswith("eigen"):
        power = {"eigen+iid": 1.0, "eigen+iid x2": 0.5, "eigen only": 1.0, "eigen+iid steep(λ)": 2.0}[name]
        Se, _ = pc.eigen_cov(H, K, 1.0 if name in ("eigen+iid x2", "eigen only") else np.sqrt(0.5), power)
        if name == "eigen only":
            return Se + 1e-6 * I
        return (1.0 if name == "eigen+iid x2" else 0.5) * I + Se
    if name == "random-corr+iid":
        Q, _ = np.linalg.qr(rng_dirs.standard_normal((D, D)))
        R = Q[:, :K].T                                              # 무작위 정규직교 방향
        sig = H["t"]["component_std"][:K]
        Se = R.T @ np.diag(sig ** 2) @ R
        Se *= 0.5 * D / np.trace(Se)
        return 0.5 * I + Se
    raise ValueError(name)


METHODS = ["iid", "eigen+iid", "eigen+iid x2", "eigen only", "random-corr+iid", "eigen+iid steep(λ)"]


def target(kind: str, rng: np.random.Generator, norm: float = 1.5) -> np.ndarray:
    if kind == "human":
        v = rng.multivariate_normal(np.zeros(D), C_H)
    elif kind == "isotropic":
        v = rng.standard_normal(D)
    else:   # mixed: 사람 방향 70% + 부분공간 밖 30% (에너지 기준)
        Q, _ = np.linalg.qr(Bf[:K].T)
        h = rng.multivariate_normal(np.zeros(D), C_H)
        o = rng.standard_normal(D)
        o = o - Q @ (Q.T @ o)
        v = np.sqrt(0.7) * h / np.linalg.norm(h) + np.sqrt(0.3) * o / np.linalg.norm(o)
    return norm * v / np.linalg.norm(v)


def spider_like(S: np.ndarray, qstar: np.ndarray, rng, N=128, T=16, s0=0.35, decay=0.85) -> float:
    """식 3: 상위 10% 섭동의 가중 평균만큼 이동. 노이즈 크기는 고정 스케줄(학습하지 않음)."""
    L = np.linalg.cholesky(S)
    q = np.zeros(D)
    c0 = float(((q - qstar) ** 2).sum())
    top = max(1, N // 10)
    w = np.exp(-np.arange(top) / 3.0)
    w /= w.sum()
    for t in range(T):
        eps = (rng.standard_normal((N, D)) @ L.T) * s0 * decay ** t
        cost = ((q + eps - qstar) ** 2).sum(1)
        idx = np.argsort(cost)[:top]
        q = q + (w[:, None] * eps[idx]).sum(0)
    return float(((q - qstar) ** 2).sum() / c0)


def reinforce_cos(S: np.ndarray, qstar: np.ndarray, rng, batch=64, reps=200) -> dict:
    """보상 R(a) = -||a - q*||², μ = 0. 참 기울기(평균 보상의 μ 기울기) = 2 q*."""
    L = np.linalg.cholesky(S)
    Sinv = np.linalg.inv(S)
    true = 2 * qstar
    cos, means = [], []
    for _ in range(reps):
        eps = rng.standard_normal((batch, D)) @ L.T
        R = -((eps - qstar) ** 2).sum(1)
        A = R - R.mean()
        g = (A[:, None] * (eps @ Sinv)).mean(0)
        means.append(g)
        cos.append(float(g @ true / (np.linalg.norm(g) * np.linalg.norm(true))))
    gbar = np.mean(means, 0)
    return {"cos_batch_median": round(float(np.median(cos)), 3),
            "mean_grad_cos_to_true": round(float(gbar @ true / (np.linalg.norm(gbar) * np.linalg.norm(true))), 3),
            "mean_grad_norm_ratio": round(float(np.linalg.norm(gbar) / np.linalg.norm(true)), 3)}


def main():
    rng = np.random.default_rng(20261029)
    rng_dirs = np.random.default_rng(7)
    covs = {m: dist(m, rng_dirs) for m in METHODS}
    out = {"setup": {"hand": "sharpa", "d": D, "k": K, "top": "10%", "trials": 300, "N_T_s0": "영역 이름에 표기", "note": "정규화 관절 좌표. 'eigen+iid'가 논문 서술(총 분산 맞춤), 'x2'가 공개 SimToolReal preset"},
           "T1_spider_like_final_cost_ratio_median": {}, "T1_vs_iid_win_rate": {}, "T2_hit_prob": {}, "T3_reinforce": {}}
    regimes = {"noise_ok (N32,T4,s0=0.3)": dict(N=32, T=4, s0=0.3, norm=1.5),
               "noise_too_small (N32,T8,s0=0.15)": dict(N=32, T=8, s0=0.15, norm=2.0)}
    for rname, cfg in regimes.items():
        out["T1_spider_like_final_cost_ratio_median"][rname] = {}
        out["T1_vs_iid_win_rate"][rname] = {}
        for kind in ("human", "mixed", "isotropic"):
            res = {m: [] for m in METHODS}
            for trial in range(300):
                qs = target(kind, rng, cfg["norm"])
                seed = int(rng.integers(1 << 31))
                for m in METHODS:   # 같은 난수 쌍으로 비교
                    res[m].append(spider_like(covs[m], qs, np.random.default_rng(seed), N=cfg["N"], T=cfg["T"], s0=cfg["s0"]))
            base = np.median(res["iid"])
            out["T1_spider_like_final_cost_ratio_median"][rname][kind] = {m: round(float(np.median(v) / base), 3) for m, v in res.items()}
            b = np.array(res["iid"])
            out["T1_vs_iid_win_rate"][rname][kind] = {m: round(float((np.array(v) < b).mean()), 3) for m, v in res.items() if m != "iid"}
    for kind in ("human", "isotropic"):
        U = np.array([target(kind, rng, norm=1.0) for _ in range(2000)])
        row = {}
        for m in ("iid", "eigen+iid", "eigen+iid x2", "random-corr+iid"):
            sd = np.sqrt(np.einsum("ni,ij,nj->n", U, covs[m], U))
            row[m] = {f"P(<eps,u> >= {a})": float(f"{np.mean(ndtr(-a / sd)):.3g}") for a in (2.0, 3.0)}
        # 몬테카를로 확인 (a = 3, 방향 20개)
        mc = []
        for u in U[:20]:
            e = rng.standard_normal((20000, D)) @ np.linalg.cholesky(covs["eigen+iid"]).T
            mc.append(float((e @ u >= 3.0).mean()))
        row["eigen+iid_MC_check_a3"] = float(f"{np.mean(mc):.3g}")
        out["T2_hit_prob"][kind] = row
    for kind in ("human", "isotropic"):
        qs = target(kind, rng, norm=1.0)
        out["T3_reinforce"][kind] = {m: reinforce_cos(covs[m], qs, rng) for m in ("iid", "eigen+iid")}
    path = ROOT / "analysis" / "exploration_toy_results.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
