"""토이: 공유 잠재(InfoNCE 정렬, CCA)가 직접 회귀·도메인 무작위화보다 나은 조건과, 쌍 데이터가 얼마나 필요한가.

모델: 물리 자극 z ∈ R^3 (깊이, 위치, 형상 코드). 시뮬 관측 x_s = tanh(A z) + 잡음,
실측 관측 x_r = tanh(A_r z) + γ·Q(z) + 잡음 (A_r은 A와 δ만큼 다르고, Q는 이차항으로 시뮬 가족 밖의 비선형 왜곡).
downstream 라벨 y = w·z + 0.5 z0 z1 (힘 대리). 시뮬 라벨은 모두 있고, 실측 라벨은 평가에만 쓴다.
방법:
 M0 시뮬 회귀를 실측에 그대로 적용 (정렬 없음)
 M1 도메인 무작위화: 시뮬 가족(tanh 선형 혼합)을 무작위로 바꿔 학습 (실측 데이터 없음)
 M2 실측 라벨 N개로 직접 회귀 (라벨 기반 상한선)
 M3 CCA: 쌍 N개로 공유 잠재, 그다음 시뮬 라벨로 회귀
 M4 InfoNCE 정렬: 쌍 N개로 선형 인코더(고정 특징 위) 학습, 그다음 시뮬 라벨로 회귀
 M5 M4와 같은데 쌍을 섞은 비율 f를 둠 (쌍 오류 대응)

numpy 1스레드, 전체 수 분 이내. 실행: python3 analysis/align_toy.py -> analysis/align_toy_results.json
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("align_toy_results.json")
P = 12          # 관측 차원
K = 4           # 공유 잠재 차원
W_Y = np.array([1.0, -0.5, 0.8])
T_TAU = 0.1


def feats(x):
    # 고정 특징: 원 입력 + 이차항 (인코더는 이 위의 선형)
    iu = np.triu_indices(x.shape[1])
    quad = (x[:, :, None] * x[:, None, :])[:, iu[0], iu[1]]
    return np.hstack([x, quad, np.ones((x.shape[0], 1))])


def world(seed=0):
    r = np.random.default_rng(seed)
    A = r.normal(size=(P, 3)) * 0.9
    E = r.normal(size=(P, 3))
    Q = r.normal(size=(P, 9)) * 0.5
    return A, E, Q


A, E, Q = world(0)


def sim_obs(z, rng, A_=A, noise=0.05):
    return np.tanh(z @ A_.T) + noise * rng.normal(size=(z.shape[0], P))


def real_obs(z, rng, gamma, delta, noise=0.1):
    zz = (z[:, :, None] * z[:, None, :]).reshape(z.shape[0], 9)
    return np.tanh(z @ (A + delta * E).T) + gamma * zz @ Q.T + noise * rng.normal(size=(z.shape[0], P))


def labels(z):
    return z @ W_Y + 0.5 * z[:, 0] * z[:, 1]


def ridge(X, y, lam=1e-2):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ y)


def pred(w, X):
    return np.hstack([X, np.ones((X.shape[0], 1))]) @ w


def r2(y, yh):
    return 1 - np.mean((y - yh) ** 2) / np.var(y)


def cca(Fx, Fy, k=K, lam=1e-3):
    mx, my = Fx.mean(0), Fy.mean(0)
    X, Y = Fx - mx, Fy - my
    n = X.shape[0]
    Cxx = X.T @ X / n + lam * np.eye(X.shape[1])
    Cyy = Y.T @ Y / n + lam * np.eye(Y.shape[1])
    Cxy = X.T @ Y / n

    def inv_sqrt(C):
        w, V = np.linalg.eigh(C)
        return V @ np.diag(w ** -0.5) @ V.T
    Mx, My = inv_sqrt(Cxx), inv_sqrt(Cyy)
    U, s, Vt = np.linalg.svd(Mx @ Cxy @ My)
    return (mx, Mx @ U[:, :k], my, My @ Vt[:k].T)


def info_nce_train(Fa, Fb, k=K, steps=400, lr=0.02, seed=0, batch=256, wd=1e-3):
    r = np.random.default_rng(seed)
    d = Fa.shape[1]
    Wa = r.normal(size=(d, k)) * 0.1
    Wb = r.normal(size=(d, k)) * 0.1
    ma, va = np.zeros_like(Wa), np.zeros_like(Wa)
    mb, vb = np.zeros_like(Wb), np.zeros_like(Wb)
    n = Fa.shape[0]
    for t in range(1, steps + 1):
        idx = r.choice(n, size=min(batch, n), replace=False)
        a, b = Fa[idx] @ Wa, Fb[idx] @ Wb
        na, nb = np.linalg.norm(a, axis=1, keepdims=True), np.linalg.norm(b, axis=1, keepdims=True)
        U, V = a / na, b / nb
        B = U.shape[0]
        S = U @ V.T / T_TAU
        Pr = np.exp(S - S.max(1, keepdims=True)); Pr /= Pr.sum(1, keepdims=True)
        Pc = np.exp(S - S.max(0, keepdims=True)); Pc /= Pc.sum(0, keepdims=True)
        I = np.eye(B)
        dS = 0.5 * (Pr - I) / B + 0.5 * (Pc - I) / B
        dU = dS @ V / T_TAU
        dV = dS.T @ U / T_TAU
        da = (dU - U * np.sum(U * dU, 1, keepdims=True)) / na
        db = (dV - V * np.sum(V * dV, 1, keepdims=True)) / nb
        gA = Fa[idx].T @ da + wd * Wa
        gB = Fb[idx].T @ db + wd * Wb
        for W, g, m, v in ((Wa, gA, ma, va), (Wb, gB, mb, vb)):
            m *= 0.9; m += 0.1 * g
            v *= 0.999; v += 0.001 * g * g
            W -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return Wa, Wb


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def run(N_pairs, gamma, delta, seed, mismatch=0.0, n_sim=4000, n_test=2000, dr_fam=True):
    rs = np.random.default_rng(1000 + seed)
    rr = np.random.default_rng(2000 + seed)
    # 시뮬 학습 데이터
    zs = rs.normal(size=(n_sim, 3))
    xs = sim_obs(zs, rs)
    ys = labels(zs)
    # 실측 쌍 (같은 자극 z, 실측 관측)
    zp = rr.normal(size=(N_pairs, 3))
    xp_r = real_obs(zp, rr, gamma, delta)
    xp_s = sim_obs(zp, rr)
    # 쌍 오류: 일부를 무작위 다른 자극과 짝짓기
    if mismatch > 0:
        bad = rr.random(N_pairs) < mismatch
        xp_r[bad] = xp_r[rr.permutation(N_pairs)][bad]
    # 평가: 실측 관측 (자극은 새로 뽑음)
    zt = rs.normal(size=(n_test, 3))
    xt_r = real_obs(zt, rs, gamma, delta)
    yt = labels(zt)
    out = {}
    # M0 정렬 없음
    w0 = ridge(xs, ys)
    out["M0 시뮬 회귀 그대로"] = r2(yt, pred(w0, xt_r))
    # M1 도메인 무작위화: 시뮬 특징을 무작위 tanh 혼합(가족 안)으로 여러 번 바꿔 학습
    if dr_fam:
        Xd, Yd = [], []
        for _ in range(20):
            Ak = A + rs.uniform(0.0, 1.0) * rs.normal(size=(P, 3)) * 0.5  # 시뮬 매핑 근처의 무작위 섭동
            Xd.append(sim_obs(zs[:400], rs, A_=Ak)); Yd.append(ys[:400])
        w1 = ridge(np.vstack(Xd), np.concatenate(Yd))
        out["M1 도메인 무작위화(시뮬 매핑 근처 섭동)"] = r2(yt, pred(w1, xt_r))
    # M2 실측 라벨 N개 직접 회귀 (라벨은 쌍의 자극 y)
    if N_pairs >= 5:
        w2 = ridge(xp_r, labels(zp), lam=1e-1)
        out["M2 실측 라벨 N개 직접 회귀"] = r2(yt, pred(w2, xt_r))
    # M3 CCA (특징: 시뮬 특징, 실측 특징)
    Fs, Fr = feats(xs), feats(xp_r)
    if N_pairs >= 3 * Fs.shape[1] // 2 or True:
        mxs, Ms, mys, Mr = cca(feats(xp_s), Fr)
        zs_lat = (feats(xs) - feats(xp_s).mean(0)) @ Ms
        w3 = ridge(zs_lat, ys, lam=1e-2)
        zt_lat = (feats(xt_r) - Fr.mean(0)) @ Mr
        out["M3 CCA 공유 잠재"] = r2(yt, pred(w3, zt_lat))
    # M4 InfoNCE 정렬 (특징: 쌍에서 학습, 시뮬 전체 특징에 적용)
    Fps = feats(xp_s)
    Wa, Wb = info_nce_train(Fps, Fr, seed=seed)
    ua_s = unit(feats(xs) @ Wa)
    w4 = ridge(ua_s, ys, lam=1e-2)
    ub_t = unit(feats(xt_r) @ Wb)
    out["M4 InfoNCE 정렬"] = r2(yt, pred(w4, ub_t))
    # M4b: 같은 인코더, 정규화 없이 선형 출력(노름 정보 보존)으로 회귀
    w4b = ridge(feats(xs) @ Wa, ys, lam=1e-2)
    out["M4b InfoNCE 정렬(정규화 없음)"] = r2(yt, pred(w4b, feats(xt_r) @ Wb))
    return out


def main():
    t0 = time.time()
    res = {"config": {"P": P, "K": K, "tau": T_TAU, "note": "지표는 실측 테스트 R² (1=완벽)"}, "runs": []}
    Ns = [10, 30, 100, 300, 1000, 3000]
    for gamma, delta in [(0.0, 0.3), (0.5, 0.3), (1.0, 0.3), (0.5, 1.0)]:
        for N in Ns:
            seeds = [0, 1, 2] if N <= 300 else [0, 1]
            agg = {}
            for s in seeds:
                o = run(N, gamma, delta, s)
                for k, v in o.items():
                    agg.setdefault(k, []).append(v)
            row = {"gamma": gamma, "delta": delta, "N_pairs": N,
                   **{k: [round(float(np.mean(v)), 3), round(float(np.std(v)), 3)] for k, v in agg.items()}}
            res["runs"].append(row)
            print(gamma, delta, N, {k: v[0] for k, v in row.items() if k.startswith("M")}, flush=True)
    # 쌍 오류 실험 (γ=0.5, δ=0.3, N=1000)
    res["mismatch"] = []
    for f in [0.0, 0.1, 0.3, 0.6]:
        vals = [run(1000, 0.5, 0.3, s, mismatch=f)["M4 InfoNCE 정렬"] for s in [0, 1]]
        res["mismatch"].append({"mismatch_fraction": f, "M4_R2": round(float(np.mean(vals)), 3)})
    res["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("done", res["seconds"], "s")


if __name__ == "__main__":
    main()
