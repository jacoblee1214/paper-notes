"""추가 검증 2 (restacvla): 잔차 대 원시 촉각을 비선형 CMP와 VQ 병목에서 다시 본다. 실행: python3 analysis/extra_residual_vq.py

gate_toy.py의 part B(선형 CMP, ridge)를 확장한다.
  세계: 시각 v ∈ R^256, 촉각 z ∈ R^8 = √ρ²·p(v) + √(1−ρ²)·u (p는 tanh 비선형 함수, 표준화), 행동 y = w_v·v + w_u·u + 잡음.
  CMP: 작은 MLP(256→32→8, tanh)로 ẑ = f(v)를 학습. 학습 데이터는 (a) 별도 5000개, (b) 정책 데이터와 같은 600개(in-sample).
  정책 입력 변형:
    연속값: [v, z] (원시) / [v, r] (잔차, r = z − ẑ)
    VQ(K개 코드, k-means 코드북, 정책 학습 데이터의 비지도 특징으로 적합): [v, onehot(q(z))] / [v, onehot(q(r))] / [v, onehot(q(z)), onehot(q(r))]
  정책: 특징을 표준화한 ridge, λ는 검증 세트(300)로 고름. 학습 600, 시험 3000, 시드 5개.
질문: 선형 toy의 결론(잔차가 원시보다 나은 것은 CMP가 정확하고 시각 차원이 클 때)이 비선형 CMP와 VQ 병목에서도 유지되는가.
K에 따라 힘 해상도(정보) 손실이 어떻게 커지는가.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DV, DZ = 256, 8
N_POL, N_VAL, N_TEST, N_CMP = 600, 300, 3000, 5000
K_GRID = [8, 32, 128, 512]
RHO2 = [0.5, 0.9]
SEEDS = range(5)
CMP_STEPS, CMP_H, CMP_LR, CMP_L2 = 800, 32, 0.01, 0.1
LAMS = np.logspace(-3, 3, 13)


def make_world(rng, rho2):
    A = rng.normal(size=(DZ, DV))
    w_v = rng.normal(size=DV) / np.sqrt(DV)
    w_u = rng.normal(size=DZ) / np.sqrt(DZ)
    ref_v = rng.normal(size=(20000, DV))
    ref_p = np.tanh(ref_v @ A.T / np.sqrt(DV))
    mu, sd = ref_p.mean(0), ref_p.std(0)
    return {"A": A, "w_v": w_v, "w_u": w_u, "rho2": rho2, "mu": mu, "sd": sd}


def sample(rng, W, n):
    v = rng.normal(size=(n, DV))
    u = rng.normal(size=(n, DZ))
    p = (np.tanh(v @ W["A"].T / np.sqrt(DV)) - W["mu"]) / W["sd"]
    z = np.sqrt(W["rho2"]) * p + np.sqrt(1 - W["rho2"]) * u
    y = v @ W["w_v"] + u @ W["w_u"] + 0.3 * rng.normal(size=n)
    return v, z, y


def fit_cmp(rng, v, z):
    """작은 MLP로 ẑ = f(v)를 맞춘다 (Adam, 전체 배치)."""
    mu, sd = v.mean(0), v.std(0) + 1e-9
    X = (v - mu) / sd
    W1 = rng.normal(0, 1 / np.sqrt(DV), (DV, CMP_H)); b1 = np.zeros(CMP_H)
    W2 = rng.normal(0, 1 / np.sqrt(CMP_H), (CMP_H, DZ)); b2 = np.zeros(DZ)
    P = [W1, b1, W2, b2]
    M = [np.zeros_like(p) for p in P]; V = [np.zeros_like(p) for p in P]
    n = len(X)
    for it in range(1, CMP_STEPS + 1):
        W1, b1, W2, b2 = P
        h = np.tanh(X @ W1 + b1)
        out = h @ W2 + b2
        g_out = 2 * (out - z) / n
        gW2, gb2 = h.T @ g_out + CMP_L2 * W2, g_out.sum(0)
        g_h = (g_out @ W2.T) * (1 - h ** 2)
        gW1, gb1 = X.T @ g_h + CMP_L2 * W1, g_h.sum(0)
        G = [gW1, gb1, gW2, gb2]
        for i in range(4):
            M[i] = 0.9 * M[i] + 0.1 * G[i]
            V[i] = 0.999 * V[i] + 0.001 * G[i] ** 2
            P[i] = P[i] - CMP_LR * (M[i] / (1 - 0.9 ** it)) / (np.sqrt(V[i] / (1 - 0.999 ** it)) + 1e-8)
    W1, b1, W2, b2 = P

    def predict(vv):
        return np.tanh(((vv - mu) / sd) @ W1 + b1) @ W2 + b2
    return predict


def kmeans(rng, X, K, iters=15):
    idx = rng.choice(len(X), size=min(K, len(X)), replace=False)
    C = X[idx].copy()
    if len(C) < K:
        C = np.vstack([C, X[rng.integers(0, len(X), K - len(C))]])
    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d2.argmin(1)
        for k in range(K):
            m = lab == k
            if m.any():
                C[k] = X[m].mean(0)
    return C


def quantize(C, X):
    d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    return d2.argmin(1)


def onehot(idx, K):
    return np.eye(K)[idx]


def ridge_fit_eval(Xtr, ytr, Xva, yva, Xte, yte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A, B, C = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean()
    G, b = A.T @ A, A.T @ (ytr - ym)
    best = (np.inf, None)
    for lam in LAMS:
        w = np.linalg.solve(G + lam * len(A) * np.eye(A.shape[1]), b)
        mse = float(np.mean((B @ w + ym - yva) ** 2))
        if mse < best[0]:
            best = (mse, w)
    w = best[1]
    return float(np.mean((C @ w + ym - yte) ** 2))


def run(seed, rho2, cmp_mode):
    rng = np.random.default_rng(5000 + seed)
    W = make_world(rng, rho2)
    v_tr, z_tr, y_tr = sample(rng, W, N_POL)
    v_va, z_va, y_va = sample(rng, W, N_VAL)
    v_te, z_te, y_te = sample(rng, W, N_TEST)
    if cmp_mode == "separate":
        v_c, z_c, _ = sample(rng, W, N_CMP)
        f = fit_cmp(rng, v_c, z_c)
    else:
        f = fit_cmp(rng, v_tr, z_tr)
    r_tr, r_va, r_te = z_tr - f(v_tr), z_va - f(v_va), z_te - f(v_te)
    out = {}
    out["raw_continuous"] = ridge_fit_eval(np.hstack([v_tr, z_tr]), y_tr, np.hstack([v_va, z_va]), y_va,
                                           np.hstack([v_te, z_te]), y_te)
    out["resid_continuous"] = ridge_fit_eval(np.hstack([v_tr, r_tr]), y_tr, np.hstack([v_va, r_va]), y_va,
                                             np.hstack([v_te, r_te]), y_te)
    for K in K_GRID:
        Cz = kmeans(rng, z_tr, K)
        Cr = kmeans(rng, r_tr, K)
        qz_tr, qz_va, qz_te = (onehot(quantize(Cz, x), K) for x in (z_tr, z_va, z_te))
        qr_tr, qr_va, qr_te = (onehot(quantize(Cr, x), K) for x in (r_tr, r_va, r_te))
        out[f"raw_VQ{K}"] = ridge_fit_eval(np.hstack([v_tr, qz_tr]), y_tr, np.hstack([v_va, qz_va]), y_va,
                                           np.hstack([v_te, qz_te]), y_te)
        out[f"resid_VQ{K}"] = ridge_fit_eval(np.hstack([v_tr, qr_tr]), y_tr, np.hstack([v_va, qr_va]), y_va,
                                             np.hstack([v_te, qr_te]), y_te)
        out[f"both_VQ{K}"] = ridge_fit_eval(np.hstack([v_tr, qz_tr, qr_tr]), y_tr,
                                            np.hstack([v_va, qz_va, qr_va]), y_va,
                                            np.hstack([v_te, qz_te, qr_te]), y_te)
    return out


def main():
    res = {}
    for rho2 in RHO2:
        for cmp_mode in ("separate", "in-sample"):
            key = f"rho2={rho2}|CMP={cmp_mode}"
            runs = [run(s, rho2, cmp_mode) for s in SEEDS]
            agg = {}
            for k in runs[0]:
                vals = np.array([r[k] for r in runs])
                agg[k] = {"mse_mean": round(float(vals.mean()), 4), "mse_sd": round(float(vals.std(ddof=1)), 4)}
            # 짝을 이룬 차이 (같은 시드): 잔차 − 원시
            pairs = {}
            for K in K_GRID:
                for a, b in (("resid_VQ", "raw_VQ"), ("both_VQ", "raw_VQ")):
                    d = np.array([r[f"{a}{K}"] - r[f"{b}{K}"] for r in runs])
                    pairs[f"{a}{K} - {b}{K}"] = {"mean": round(float(d.mean()), 4),
                                                 "se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 4),
                                                 "resid_better_in_seeds": int((d < 0).sum())}
            d = np.array([r["resid_continuous"] - r["raw_continuous"] for r in runs])
            pairs["resid_continuous - raw_continuous"] = {"mean": round(float(d.mean()), 4),
                                                          "se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 4),
                                                          "resid_better_in_seeds": int((d < 0).sum())}
            res[key] = {"mse": agg, "paired_diff": pairs}
            print(key, {k: agg[k]["mse_mean"] for k in ("raw_continuous", "resid_continuous",
                                                         "raw_VQ8", "resid_VQ8", "raw_VQ128", "resid_VQ128",
                                                         "raw_VQ512", "resid_VQ512")}, flush=True)
    out = {"setup": {"d_v": DV, "d_z": DZ, "n_policy": N_POL, "n_val": N_VAL, "n_test": N_TEST, "n_cmp_separate": N_CMP,
                     "cmp": f"MLP {DV}→{CMP_H}→{DZ}, tanh, Adam {CMP_STEPS} steps", "K_grid": K_GRID, "rho2": RHO2,
                     "seeds": list(SEEDS), "irreducible_mse": 0.09,
                     "policy": "ridge (표준화, λ는 검증 세트), VQ는 k-means 코드북 + one-hot"},
           "results": res}
    path = ROOT / "analysis" / "extra_residual_vq_results.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    plot(res)
    print("wrote", path)


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                              "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
    if font:
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 8.5, "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    styles = {"raw_VQ": ("#9a9a9f", "o"), "resid_VQ": ("#2a78d6", "s"), "both_VQ": ("#1baf7a", "^")}
    for ax, key in zip(axes, ["rho2=0.9|CMP=separate", "rho2=0.9|CMP=in-sample"]):
        mse = res[key]["mse"]
        for pre, (c, mk) in styles.items():
            ys = [mse[f"{pre}{K}"]["mse_mean"] for K in K_GRID]
            ax.plot(K_GRID, ys, marker=mk, color=c, ms=4, lw=1.4, label=pre.replace("_VQ", " 코드"))
        ax.axhline(mse["raw_continuous"]["mse_mean"], color="#9a9a9f", ls=":", lw=0.9, label="원시 연속값")
        ax.axhline(mse["resid_continuous"]["mse_mean"], color="#2a78d6", ls=":", lw=0.9, label="잔차 연속값")
        ax.axhline(0.09, color="#0b0b0b", ls="--", lw=0.8, label="잡음 바닥 0.09")
        ax.set_xscale("log", base=2)
        ax.set_xticks(K_GRID)
        ax.set_xticklabels([str(k) for k in K_GRID])
        ax.set_xlabel("코드 수 K")
        ax.set_title("CMP 별도 데이터" if "separate" in key else "CMP 정책 데이터와 같음 (in-sample)", fontsize=8.5)
        ax.grid(color="#e4e4e7", lw=0.6)
    axes[0].set_ylabel("시험 MSE (ρ² = 0.9)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.8, frameon=False, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.savefig(ROOT / "report" / "figures" / "extra_residual_vq.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
