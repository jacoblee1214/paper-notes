"""논문 Figure 2의 해석을 점검하는 numpy 장난감 실험 (V-JEPA 2 검증이 아님).

질문: "예측적(시간 구조를 쓰는) 표현은 과제 상태를 더 잘 담고, 조명·배경 같은 과제 무관 요인은 덜 담는다"는
Figure 2 해석이, 인코더의 목적함수가 '시간 예측 가능성'이기만 하면 성립하는가?

설정
- 상태 s_t (4차원, 평균회귀 AR(1) + 잡음), 교란 n (3차원). 교란은 '상수'(한 궤적 안에서 고정, 조명·배경 유형)와
  '변동'(AR(1), 매 스텝 바뀜, 예측 불가 유형) 두 가지로 바꾼다.
- 관측 x_t = A s_t + B n + 잡음 (32차원). 교란 B의 크기(2.0)를 키워 교란의 픽셀 분산이 과제 신호보다 크게 만든다.
- 인코더는 두 프레임 [x_t, x_{t-1}]을 입력받는 선형 투영(4차원 출력)으로 통일한다.
  * PCA: 분산이 큰 방향 (재구성형, 이미지 기반 SSL의 단순 대리)
  * SFA: 시간 차분의 분산이 작은 방향 (느린 특징, '예측 가능한 것을 남기는' 대리)
  * RANDOM: 무작위 투영 (대조군)
- 프로브: 인코더 출력에서 리지 회귀로 (a) s_t, (b) 10스텝 뒤 상태 변화 Δs, (c) 교란 n을 예측.
  점수는 테스트 MSE를 타깃 분산으로 나눈 값(낮을수록 많이 담음).

실행: python3 analysis/03_toy_encoder_probe.py  → analysis/03_toy_encoder_probe_results.json (CPU, 수 초)
"""
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "03_toy_encoder_probe_results.json"
rng = np.random.default_rng(0)

N_TRAJ, T = 300, 20          # 궤적 300개 × 20스텝
D_OBS, K = 32, 4             # 관측 차원, 인코더 출력 차원 (K가 잠재 차원 7보다 작아 병목이 걸리게 함)
S_DIM, N_DIM = 4, 3
H = 10                        # Δs 예측 지평


def simulate(b_scale, seed, nuis_kind):
    """nuis_kind: 'const' = 궤적 안에서 상수(조명 등), 'fluct' = 매 스텝 바뀌는 AR(1) 교란(예측 불가)."""
    r = np.random.default_rng(seed)
    A = r.normal(size=(D_OBS, S_DIM))
    B = r.normal(size=(D_OBS, N_DIM)) * b_scale
    S, Nn, X = [], [], []
    for _ in range(N_TRAJ):
        s = np.zeros(S_DIM)
        n = r.normal(size=N_DIM)
        traj_s, traj_n, traj_x = [], [], []
        for t in range(T):
            s = 0.9 * s + 0.4 * r.normal(size=S_DIM)          # 평균회귀 + 잡음 (목표 없음)
            if nuis_kind == "fluct":
                n = 0.3 * n + 0.95 * r.normal(size=N_DIM)      # 빠르게 바뀌는 교란
            traj_s.append(s.copy()); traj_n.append(n.copy())
            traj_x.append(A @ s + B @ n + 0.1 * r.normal(size=D_OBS))
        S.append(np.array(traj_s)); Nn.append(np.array(traj_n)); X.append(np.array(traj_x))
    return np.array(S), np.array(Nn), np.array(X)


def build_pairs(S, Nn, X):
    """프레임 쌍 특징 [x_t, x_{t-1}], 타깃 s_t, Δs_t, n. t>=H와 t>=1만 사용."""
    feats, s_t, ds, nn = [], [], [], []
    for tr in range(N_TRAJ):
        for t in range(1, T - H):
            feats.append(np.concatenate([X[tr, t], X[tr, t - 1]]))
            s_t.append(S[tr, t]); ds.append(S[tr, t + H] - S[tr, t]); nn.append(Nn[tr, t])
    return np.array(feats), np.array(s_t), np.array(ds), np.array(nn)


def fit_encoder(name, F_train, S_train, Nn_train, Xs_train):
    """F_train: (n, 2*D_OBS). 반환: (2*D_OBS, K) 선형 투영 행렬."""
    if name == "RANDOM":
        return rng.normal(size=(F_train.shape[1], K)) / np.sqrt(F_train.shape[1])
    if name == "PCA":
        C = np.cov(F_train[:, :D_OBS], rowvar=False)
        w, V = eigh(C)
        P = V[:, ::-1][:, :K]
        return np.vstack([P, np.zeros((D_OBS, K))])   # 현재 프레임만 사용
    if name == "SFA":
        # 같은 궤적 안의 차분 x_t - x_{t-1}의 분산 대비 x_t의 분산이 작은 방향을 고른다.
        dX = F_train[:, :D_OBS] - F_train[:, D_OBS:]
        C_x = np.cov(F_train[:, :D_OBS], rowvar=False) + 1e-6 * np.eye(D_OBS)
        C_d = np.cov(dX, rowvar=False) + 1e-6 * np.eye(D_OBS)
        w, V = eigh(C_d, C_x)                       # 작은 고유값 = 느린 방향
        P = V[:, :K]
        return np.vstack([P, np.zeros((D_OBS, K))])
    raise ValueError(name)


def ridge_fit(Z, Y, lam=1e-3):
    Zb = np.hstack([Z, np.ones((Z.shape[0], 1))])
    W = np.linalg.solve(Zb.T @ Zb + lam * np.eye(Zb.shape[1]), Zb.T @ Y)
    return W


def ridge_mse(W, Z, Y):
    Zb = np.hstack([Z, np.ones((Z.shape[0], 1))])
    return float(np.mean((Zb @ W - Y) ** 2, axis=0).mean())


def run(b_scale, seed, nuis_kind):
    S, Nn, X = simulate(b_scale, seed, nuis_kind)
    F, s_t, ds, nn = build_pairs(S, Nn, X)
    n_tr = int(0.7 * N_TRAJ) * (T - H - 1)
    tr, te = slice(0, n_tr), slice(n_tr, None)
    out = {}
    for enc in ("PCA", "SFA", "RANDOM"):
        P = fit_encoder(enc, F[tr], s_t[tr], nn[tr], None)
        Z = F @ P
        scores = {}
        for tgt_name, Y in (("relevant_state", s_t), ("delta_s_10", ds), ("nuisance", nn)):
            var = Y[te].var(axis=0).mean()
            W = ridge_fit(Z[tr], Y[tr])
            scores[tgt_name] = round(ridge_mse(W, Z[te], Y[te]) / var, 4)
        out[enc] = scores
    # 상태를 그대로 쓴 상한선 (관측 차원 비교용)
    Zs = s_t
    out["ORACLE_STATE"] = {}
    for tgt_name, Y in (("relevant_state", s_t), ("delta_s_10", ds), ("nuisance", nn)):
        var = Y[te].var(axis=0).mean()
        W = ridge_fit(Zs[tr], Y[tr])
        out["ORACLE_STATE"][tgt_name] = round(ridge_mse(W, Zs[te], Y[te]) / var, 4)
    return out


results = {"description": "정규화 테스트 MSE (타깃 분산 대비). 낮을수록 해당 정보를 많이 담음.",
           "caveat": "선형 인코더 3종과 합성 데이터뿐이다. V-JEPA 2나 DINOv2, SigLIP에 대한 직접 증거가 아니다.",
           "settings": {}}
for nuis_kind in ("const", "fluct"):
    res = {}
    for seed in range(3):
        res[f"seed{seed}"] = run(2.0, seed, nuis_kind)
    keys = res["seed0"].keys()
    avg = {k: {t: round(float(np.mean([res[s][k][t] for s in res])), 4) for t in res["seed0"][k]} for k in keys}
    results["settings"][f"nuisance_{nuis_kind}_scale2"] = {"per_seed": res, "mean_over_seeds": avg}

OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
for k, v in results["settings"].items():
    print(k)
    for enc, sc in v["mean_over_seeds"].items():
        print(f"  {enc:12s}", sc)
print("wrote", OUT)
