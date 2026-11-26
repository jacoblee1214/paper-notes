"""토이: "미래 잠재에 선형 프로브"가 "현재 잠재 + 행동에 선형 프로브"보다 소량 라벨에서 나아지는 조건.

논문 4.2절은 라벨이 적을 때 롤아웃한 미래 잠재 위의 선형 프로브가 (z_t, 행동 묶음)에 대한 직접 선형 프로브보다 AUC가 높다고 주장한다.
여기서는 합성 환경에서 그 조건을 따져 본다. 모두 우리 가정이며 논문 수치의 재현이 아니다.

  - 잠재 z in R^16, 행동 a in R^4, 청크 K = 6. 참 동역학 z' = z + 0.4 tanh(W z + U a) (비선형) + 작은 잡음.
  - 결과 y(실패) = 마지막 잠재 z_K에 대한 규칙. 두 가지:
      linear_zK    : y = 1[v . z_K + 라벨잡음 > 0]  (논문의 가정: 실패 정보가 미래 잠재에서 선형으로 읽힘)
      nonlinear_zK : y = 1[z_K[0] * z_K[1] > 0]    (선형으로 읽히지 않는 경우)
  - 세계모델: 라벨 없는 전이 (z, a, z')로 MLP 동역학을 학습(n_unlab개), 자기회귀로 K 스텝 롤아웃. 논문처럼 인코더는 고정(여기서는 z가 관측값).
  - 프로브: 표준화 + L2 로지스틱(클래스 균형), 라벨 n개. 입력만 다르게:
      current  : z_0
      direct   : [z_0, a_1..a_K] (논문의 Direct-linear)
      rollout  : 롤아웃한 z_K (ContactGuard)
      oracle   : 참 z_K (세계모델이 완벽할 때의 상한)
실행: python3 analysis/toy_rollout_probe.py   (torch CPU, 약 70초)  → analysis/toy_rollout_probe_results.json
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
D, A, K = 16, 4, 6
SEEDS = 5          # 동역학 시드
LABEL_SETS = 8     # 시드당 라벨 집합 수
N_TEST = 3000


def make_env(rng):
    W = rng.normal(0, 1 / np.sqrt(D), (D, D))
    U = rng.normal(0, 1.0, (D, A)) / np.sqrt(A)
    v = rng.normal(0, 1, D)
    v /= np.linalg.norm(v)
    return W, U, v


def step(z, a, W, U, rng=None, noise=0.02):
    zn = z + 0.4 * np.tanh(z @ W.T + a @ U.T)
    if rng is not None:
        zn = zn + noise * rng.normal(size=zn.shape)
    return zn


def rollout_true(z0, acts, W, U, rng=None):
    z = z0
    for k in range(K):
        z = step(z, acts[:, k], W, U, rng)
    return z


def outcome(zK, v, kind, rng, label_noise=0.05):
    if kind == "linear_zK":
        y = (zK @ v > 0).astype(int)
    else:
        y = (zK[:, 0] * zK[:, 1] > 0).astype(int)
    flip = rng.random(len(y)) < label_noise
    return np.where(flip, 1 - y, y)


def auc(score, y):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_probe(X, y, l2=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    n1, n0 = yt.sum().item(), len(yt) - yt.sum().item()
    sw = torch.where(yt == 1, len(yt) / (2 * n1), len(yt) / (2 * n0))
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Xs @ w + b
        loss = 0.5 * (w ** 2).sum() + l2 * (sw * torch.nn.functional.softplus(-(2 * yt - 1) * z)).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return lambda Xn: ((torch.tensor((Xn - mu) / sd, dtype=torch.float32) @ w + b).detach().numpy())


def train_dynamics(rng, n_unlab, W, U, epochs=35):
    z = rng.normal(0, 1.0, (n_unlab, D))
    a = rng.normal(0, 1.0, (n_unlab, A))
    zn = step(z, a, W, U, rng)
    X = torch.tensor(np.concatenate([z, a], 1), dtype=torch.float32)
    Y = torch.tensor(zn - z, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(D + A, 128), torch.nn.GELU(), torch.nn.Linear(128, 128),
                              torch.nn.GELU(), torch.nn.Linear(128, D))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    bs = 256
    for _ in range(epochs):
        perm = torch.randperm(n_unlab)
        for i in range(0, n_unlab, bs):
            idx = perm[i:i + bs]
            loss = ((net(X[idx]) - Y[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return net


def rollout_model(net, z0, acts):
    z = torch.tensor(z0, dtype=torch.float32)
    with torch.no_grad():
        for k in range(K):
            a = torch.tensor(acts[:, k], dtype=torch.float32)
            z = z + net(torch.cat([z, a], 1))
    return z.numpy()


def main():
    t0 = time.time()
    res = {}
    for kind in ["linear_zK", "nonlinear_zK"]:
        for n_unlab in [500, 20000]:
            for n_lab in [50, 100, 250]:
                acc = {m: [] for m in ["current", "direct", "rollout", "oracle"]}
                dyn_err = []
                for s in range(SEEDS):
                    rng = np.random.default_rng(1000 + s)
                    W, U, v = make_env(rng)
                    net = train_dynamics(rng, n_unlab, W, U)
                    # 테스트
                    zt = rng.normal(0, 1.0, (N_TEST, D)); at = rng.normal(0, 1.0, (N_TEST, K, A))
                    zK_true = rollout_true(zt, at, W, U, None)
                    yt = outcome(zK_true, v, kind, np.random.default_rng(5), label_noise=0.0)
                    zK_hat_t = rollout_model(net, zt, at)
                    dyn_err.append(float(np.mean((zK_hat_t - zK_true) ** 2) / np.mean(zK_true ** 2)))
                    for L in range(LABEL_SETS):
                        rl = np.random.default_rng(100 * s + L)
                        zl = rl.normal(0, 1.0, (n_lab, D)); al = rl.normal(0, 1.0, (n_lab, K, A))
                        zK_l = rollout_true(zl, al, W, U, rl)
                        yl = outcome(zK_l, v, kind, rl)
                        if yl.min() == yl.max():
                            continue
                        feats_l = {"current": zl, "direct": np.concatenate([zl, al.reshape(n_lab, -1)], 1),
                                   "rollout": rollout_model(net, zl, al), "oracle": rollout_true(zl, al, W, U, None)}
                        feats_t = {"current": zt, "direct": np.concatenate([zt, at.reshape(N_TEST, -1)], 1),
                                   "rollout": zK_hat_t, "oracle": zK_true}
                        for m in acc:
                            f = fit_probe(feats_l[m], yl)
                            acc[m].append(auc(f(feats_t[m]), yt))
                res[f"{kind}|unlab={n_unlab}|lab={n_lab}"] = dict(
                    auc_mean={m: round(float(np.mean(v)), 3) for m, v in acc.items()},
                    auc_se={m: round(float(np.std(v, ddof=1) / np.sqrt(len(v))), 3) for m, v in acc.items()},
                    n_runs=len(acc["rollout"]), dynamics_rel_mse=round(float(np.mean(dyn_err)), 4))
                print(kind, n_unlab, n_lab, res[f"{kind}|unlab={n_unlab}|lab={n_lab}"]["auc_mean"],
                      "dyn_rel_mse", res[f"{kind}|unlab={n_unlab}|lab={n_lab}"]["dynamics_rel_mse"], f"{time.time()-t0:.0f}s")
    out = dict(config=dict(D=D, A=A, K=K, seeds=SEEDS, label_sets=LABEL_SETS, n_test=N_TEST, label_noise=0.05), results=res)
    (ROOT / "analysis" / "toy_rollout_probe_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote analysis/toy_rollout_probe_results.json", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
