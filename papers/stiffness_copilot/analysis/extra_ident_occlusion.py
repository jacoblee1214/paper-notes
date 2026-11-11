"""추가 검증 3 (stiffness_copilot): 가림·프레임 지연이 식별 오차를 얼마나 키우는가. 실행: python3 analysis/extra_ident_occlusion.py

label_toy.py 식별 toy의 설정을 그대로 쓴다: 상태 (θ, d), 접촉 c = [d < 0.3], 라벨 K = 접촉이면 R(θ)diag(300, 3000)R(θ)ᵀ, 아니면 1000·I, ‖K‖_F = 1.
영상 특징: 벽 각도 2θ는 항상 보인다. 접촉 거리 d는 다음 두 가지로 흐려진다.
  가림: 확률 p로 d가 가려지면 특징은 사전 평균(0.5)으로 채우고 가림 표시를 1로 둔다.
  프레임 지연: 영상은 L 프레임(1/30 s) 전 거리를 보여 준다. 접근 속도 s(무작위, 학습기는 모름)에 따라 d_obs = d + s·L/30 이다.
학습기: kNN(k=30, label_toy와 같음)과 numpy 작은 MLP(은닉 32, tanh, Adam 2000 스텝, 시드 3개 평균).
기준선: 각도만 / (θ, 참 d) / (θ, 참 c). 시험 1000, 학습 3000, 시드 5개.
질문: 가림·지연이 식별 오차를 얼마나 키우는가. 학습기를 MLP로 바꾸면 그 손해가 줄어드는가.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N_TR, N_TE, K_NN = 3000, 1000, 30
SEEDS = range(5)
P_OCC = [0.0, 0.3, 0.6]
LAGS = [0, 2, 5]
DT = 1.0 / 30.0
MLP_STEPS, MLP_H, MLP_LR = 2000, 32, 0.01


def sample(rng, n):
    th = rng.uniform(-np.pi / 6, np.pi / 6, n)
    d = rng.uniform(0, 1, n)
    s = rng.uniform(-0.3, 0.3, n)                       # 접근 속도 (m/s 가정, 학습기는 모름)
    c = (d < 0.3).astype(int)
    R = np.stack([np.stack([np.cos(th), -np.sin(th)], 1), np.stack([np.sin(th), np.cos(th)], 1)], 1)
    Ka = R @ np.diag([300.0, 3000.0]) @ np.transpose(R, (0, 2, 1))
    Kiso = np.tile(np.eye(2) * 1000.0, (n, 1, 1))
    K = np.where(c[:, None, None] == 1, Ka, Kiso)
    K = K / np.linalg.norm(K, axis=(1, 2), keepdims=True)
    ang = np.column_stack([np.cos(2 * th), np.sin(2 * th)])
    return {"d": d, "c": c, "K": K.reshape(n, 4), "ang": ang, "s": s}


def observe(rng, S, p_occ, lag):
    n = len(S["d"])
    d_obs = S["d"] + S["s"] * lag * DT                  # 지연: 과거 거리 (접근 중이면 d가 더 컸다)
    occ = rng.random(n) < p_occ
    d_feat = np.where(occ, 0.5, d_obs) + rng.normal(0, 0.01, n)
    return np.column_stack([S["ang"], d_feat, occ.astype(float)])


def knn_pred(Xtr, Ytr, Xte, k=K_NN):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    d2 = ((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    idx = np.argpartition(d2, k, axis=1)[:, :k]
    return Ytr[idx].mean(1)


def mlp_pred(Xtr, Ytr, Xte, seed):
    rng = np.random.default_rng(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    W1 = rng.normal(0, 1 / np.sqrt(A.shape[1]), (A.shape[1], MLP_H)); b1 = np.zeros(MLP_H)
    W2 = rng.normal(0, 1 / np.sqrt(MLP_H), (MLP_H, Ytr.shape[1])); b2 = np.zeros(Ytr.shape[1])
    P = [W1, b1, W2, b2]
    M = [np.zeros_like(p) for p in P]; V = [np.zeros_like(p) for p in P]
    n = len(A)
    for it in range(1, MLP_STEPS + 1):
        W1, b1, W2, b2 = P
        h = np.tanh(A @ W1 + b1)
        out = h @ W2 + b2
        g_out = 2 * (out - Ytr) / n
        gW2 = h.T @ g_out; gb2 = g_out.sum(0)
        g_h = (g_out @ W2.T) * (1 - h ** 2)
        gW1 = A.T @ g_h; gb1 = g_h.sum(0)
        G = [gW1, gb1, gW2, gb2]
        for i in range(4):
            M[i] = 0.9 * M[i] + 0.1 * G[i]
            V[i] = 0.999 * V[i] + 0.001 * G[i] ** 2
            P[i] = P[i] - MLP_LR * (M[i] / (1 - 0.9 ** it)) / (np.sqrt(V[i] / (1 - 0.999 ** it)) + 1e-8)
    W1, b1, W2, b2 = P
    return np.tanh(B @ W1 + b1) @ W2 + b2


def err(pred, Y):
    return float(np.sqrt(((pred - Y) ** 2).sum(1)).mean())


def run_cell(seed, p_occ, lag):
    rng = np.random.default_rng(1000 + seed)
    tr, te = sample(rng, N_TR), sample(rng, N_TE)
    Xtr, Xte = observe(rng, tr, p_occ, lag), observe(rng, te, p_occ, lag)
    res = {"kNN": err(knn_pred(Xtr, tr["K"], Xte), te["K"]),
           "MLP": err(mlp_pred(Xtr, tr["K"], Xte, seed), te["K"])}
    return res


def baselines(seed):
    rng = np.random.default_rng(1000 + seed)
    tr, te = sample(rng, N_TR), sample(rng, N_TE)
    out = {}
    out["angle_only_kNN"] = err(knn_pred(tr["ang"], tr["K"], te["ang"]), te["K"])
    Xtr = np.column_stack([tr["ang"], tr["d"]]); Xte = np.column_stack([te["ang"], te["d"]])
    out["theta_and_true_d_kNN"] = err(knn_pred(Xtr, tr["K"], Xte), te["K"])
    out["theta_and_true_d_MLP"] = err(mlp_pred(Xtr, tr["K"], Xte, seed), te["K"])
    Xtr = np.column_stack([tr["ang"], tr["c"] * 2.0]); Xte = np.column_stack([te["ang"], te["c"] * 2.0])
    out["theta_and_true_c_kNN"] = err(knn_pred(Xtr, tr["K"], Xte), te["K"])
    return out


def mean_sd(vals):
    a = np.array(vals)
    return round(float(a.mean()), 4), round(float(a.std(ddof=1)), 4)


def main():
    base = {k: [] for k in ["angle_only_kNN", "theta_and_true_d_kNN", "theta_and_true_d_MLP", "theta_and_true_c_kNN"]}
    for seed in SEEDS:
        b = baselines(seed)
        for k in base:
            base[k].append(b[k])
    res = {"baselines_mean_sd": {k: mean_sd(v) for k, v in base.items()}, "grid": {}}
    for p in P_OCC:
        for L in LAGS:
            kn, ml = [], []
            for seed in SEEDS:
                c = run_cell(seed, p, L)
                kn.append(c["kNN"]); ml.append(c["MLP"])
            key = f"occ={p}|lag={L}"
            res["grid"][key] = {"kNN": mean_sd(kn), "MLP": mean_sd(ml)}
            print(key, res["grid"][key], flush=True)
    print("baselines", res["baselines_mean_sd"])
    out = {"setup": {"N_train": N_TR, "N_test": N_TE, "k_nn": K_NN, "seeds": list(SEEDS), "p_occ": P_OCC,
                     "lag_frames": LAGS, "frame_dt_s": DT, "speed_range_m_s": [-0.3, 0.3],
                     "mlp": {"hidden": MLP_H, "steps": MLP_STEPS, "lr": MLP_LR, "act": "tanh", "opt": "Adam"},
                     "label": "label_toy part B와 같음 (K/‖K‖_F, 오차 = 행별 ‖·‖ 평균)"},
           **res}
    path = ROOT / "analysis" / "extra_ident_occlusion_results.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", path)


if __name__ == "__main__":
    main()
