"""토이 3: LLMD 의 "정규화 상수 λ = 1e-6 은 튜닝이 필요 없다"(부록 F)를 합성 특징으로 본다.

설정(가정): 마지막 층 특징 차원 d = 1024(표 13), 학습 특징은 저차원 신호(k = 32 방향) + 작은 등방 잡음.
같은 롤아웃의 연속 프레임은 상관되어 있으므로 학습 표본 수는 "유효 표본 수 n"으로만 둔다(실제 값은 논문에 없음).
- ID 평가: 학습과 같은 분포에서 새로 뽑은 특징. OOD: 신호 방향이 아닌 임의의 방향으로 이동한 특징.
- 점수 = 제곱 마할라노비스 거리 (식 3의 한 토큰 버전). λ 는 공분산의 평균 고유값에 대한 비율로 둔다.
출력: n 과 λ 별로 ID 점수의 중앙값(이상적 값은 d 근처), OOD 대 ID AUROC.
실행: python3 analysis/toy_llmd_cov.py  (numpy 필요, 약 1분)
"""
import json
from pathlib import Path

import numpy as np

A = Path(__file__).resolve().parent
D, K = 1024, 32
SIGMA = 0.05
rng = np.random.default_rng(0)
W = rng.normal(size=(D, K)) / np.sqrt(K)


def feats(n, rng, shift=None):
    z = rng.normal(size=(n, K))
    f = z @ W.T + SIGMA * rng.normal(size=(n, D))
    if shift is not None:
        f = f + shift
    return f


def auroc(neg, pos):
    allv = np.concatenate([neg, pos])
    ranks = allv.argsort().argsort() + 1
    return (ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos))


def main():
    res = []
    # OOD: 신호 방향에 수직인 임의 방향으로 크기 c * SIGMA * sqrt(D)(잡음 한 번의 표준 크기의 c 배)
    base = rng.normal(size=D)
    base -= W @ np.linalg.lstsq(W, base, rcond=None)[0]
    base /= np.linalg.norm(base)
    for c in (0.5, 1.0):
        shift = base * c * SIGMA * np.sqrt(D)
        for n in (300, 1000, 5000, 50000):
            tr = feats(n, np.random.default_rng(10 + n))
            mu = tr.mean(0)
            C = (tr - mu).T @ (tr - mu) / n
            evals, evecs = np.linalg.eigh(C)
            mean_eig = evals.mean()
            id_te = feats(400, np.random.default_rng(99)) - mu
            ood_te = feats(400, np.random.default_rng(98), shift) - mu
            proj_id, proj_ood = id_te @ evecs, ood_te @ evecs
            for lam_rel in (1e-6, 1e-3, 1e-1, 1.0):
                lam = lam_rel * mean_eig
                inv = 1.0 / (np.maximum(evals, 0) + lam)
                s_id = (proj_id**2 * inv).sum(1)
                s_ood = (proj_ood**2 * inv).sum(1)
                res.append({"shift_c": c, "n_eff": n, "lambda_rel_to_mean_eig": lam_rel, "n_lt_d": n < D,
                            "ID_score_median": round(float(np.median(s_id)), 1), "ID_over_d": round(float(np.median(s_id)) / D, 2),
                            "AUROC_OOD_vs_ID": round(float(auroc(s_id, s_ood)), 3)})
    json.dump(res, open(A / "toy_llmd_cov_results.json", "w"), ensure_ascii=False, indent=1)
    for r in res:
        print(r)


if __name__ == "__main__":
    main()
