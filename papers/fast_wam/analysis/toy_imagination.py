"""토이: 테스트 시점의 미래 상상이 정보를 더하는가, 학습 중 미래 예측이 표현을 바꾸는가.

합성 설정 (논문과 무관, 시드 5개):
  잠재 상태 s in R^2, 관측 x = [W s + 0.3 노이즈 (30차원), 교란 변수 20차원 (sd 2)]
  행동 a = f(s) + 0.05 노이즈 (2차원), 다음 잠재 s' = s + 0.5 a + 0.1 노이즈,
  미래 관측 x' = [W s' + 0.3 노이즈, 교란 새로 표본]
  모든 회귀는 같은 특징 수(랜덤 푸리에 40 주파수, 입력당 80 특징 + 원본)와 같은 릿지 람다 1.0을 쓴다.

E1 (데이터 처리 부등식): 직접 정책 a = h(x) 대 상상 정책 a = h(x̂'), x̂' = G(x) + sigma * 잔차 척도 * 노이즈.
     상상은 x의 함수이므로 상상만으로 얻는 정보는 0이다. sigma는 테스트 때만 더한다.
E2 (표현 효과): 미래 예측 계수 행렬의 상위 2개 특이 방향으로 x를 줄인 z = P^T x 에서 행동을 맞춘다.
     학습 표본 수 N에 따른 테스트 행동 MSE를 raw x와 비교한다.

한계: 선형 축소와 랜덤 특징만 쓴다. 신경망 WAM이나 논문 설정을 재현하지 않으며, 결과는 메커니즘 방향만 본다.
실행: OMP_NUM_THREADS=2 python3 fast_wam/analysis/toy_imagination.py
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("toy_imagination_results.json")
LAM = 1.0
NFREQ = 40
D_S, D_X_SIG, D_NUIS = 2, 30, 20
SEEDS = [0, 1, 2, 3, 4]
W = np.random.default_rng(12345).normal(size=(D_X_SIG, D_S))


def make(n, rng):
    s = rng.normal(size=(n, D_S))
    x = np.concatenate([s @ W.T + 0.3 * rng.normal(size=(n, D_X_SIG)),
                        2.0 * rng.normal(size=(n, D_NUIS))], axis=1)
    a = np.stack([np.sin(2 * s[:, 0]) + s[:, 1], np.cos(s[:, 1]) - 0.5 * s[:, 0]], axis=1)
    a = a + 0.05 * rng.normal(size=a.shape)
    s2 = s + 0.5 * a + 0.1 * rng.normal(size=s.shape)
    x2 = np.concatenate([s2 @ W.T + 0.3 * rng.normal(size=(n, D_X_SIG)),
                         2.0 * rng.normal(size=(n, D_NUIS))], axis=1)
    return x, a, x2


def feats(z):
    omega = np.random.default_rng(z.shape[1]).normal(size=(z.shape[1], NFREQ)) * 0.5
    proj = z @ omega
    return np.concatenate([z, np.sin(proj), np.cos(proj)], axis=1)


def ridge_fit(X, Y, lam=LAM):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)


def mse(p, y):
    return float(np.mean((p - y) ** 2))


def run_seed(seed):
    rng = np.random.default_rng(100 + seed)
    x_tr, a_tr, x2_tr = make(2000, rng)
    x_te, a_te, _ = make(2000, rng)
    out = {"E1": {}, "E2": {}}

    # E1
    Phi_tr, Phi_te = feats(x_tr), feats(x_te)
    B_direct = ridge_fit(Phi_tr, a_tr)
    out["E1"]["direct"] = mse(Phi_te @ B_direct, a_te)
    G = ridge_fit(Phi_tr, x2_tr)              # 상상 예측기 x -> x'
    resid_sd = float(np.std(x2_tr - Phi_tr @ G))
    H = ridge_fit(feats(Phi_tr @ G), a_tr)    # IDM형: 상상된 미래만으로 행동 (학습은 깨끗한 상상)
    for sigma in [0.0, 0.5, 1.0, 2.0]:
        noisy = Phi_te @ G + sigma * resid_sd * rng.normal(size=(len(x_te), x2_tr.shape[1]))
        out["E1"][f"imagine_sigma_{sigma}"] = mse(feats(noisy) @ H, a_te)

    # E2: 미래 예측 계수의 상위 2개 특이 방향으로 랭크 축소
    for n in [50, 200, 1000, 4000]:
        xtr, atr, x2tr = make(n, np.random.default_rng(1000 * seed + n))
        Bm = ridge_fit(feats(xtr), x2tr)     # 미래 예측 (특징 공간) -> 입력 공간 투영을 위해 원 입력에 대해 다시 맞춘다
        Bx = ridge_fit(xtr, x2tr)
        _, _, Vt = np.linalg.svd(Bx, full_matrices=False)
        P = Vt[:2].T
        raw_B = ridge_fit(feats(xtr), atr)
        out["E2"][f"raw_N{n}"] = mse(feats(x_te) @ raw_B, a_te)
        z_B = ridge_fit(feats(xtr @ P), atr)
        out["E2"][f"future_subspace_N{n}"] = mse(feats(x_te @ P) @ z_B, a_te)
    return out


def summarize(runs):
    keys = runs[0]
    res = {}
    for part in ["E1", "E2"]:
        res[part] = {}
        for k in runs[0][part]:
            vals = np.array([r[part][k] for r in runs])
            res[part][k] = {"mean": round(float(vals.mean()), 5), "sd": round(float(vals.std(ddof=1)), 5)}
    return res


if __name__ == "__main__":
    runs = [run_seed(s) for s in SEEDS]
    res = summarize(runs)
    res["settings"] = {"seeds": SEEDS, "ridge_lambda": LAM, "fourier_features": NFREQ,
                       "caveat": "합성 데이터, 선형 축소, 신경망 아님. 메커니즘의 방향만 본다."}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    for part in ["E1", "E2"]:
        for k, v in res[part].items():
            print(part, f"{k:28s}", v)
