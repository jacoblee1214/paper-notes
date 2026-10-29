"""DUST 추정기 numpy toy (CPU, 1분 안팎).

모델: 토큰 T개, h_t = W1 x_t → u_t = tanh(h_t) → m_t = u_t + β·mean_{s<t} u_s (인과 혼합, 파라미터 없음)
      → z_t = W2 m_t → ℓ_t = CE(z_t, y_t).  β가 attention처럼 토큰 사이 영향을 만든다.
Dust: 층 출력에 토큰별 독립 잡음 σa_t를 더하고, 토큰별 손실 변화로 보상, ĝ_t = (1/Kσ) Σ_i r_t^(i) a_t^(i), Ĝ = Σ_t ĝ_t x_tᵀ.

T1: cosine(Ĝ, backprop) vs K가 cos(K) = c_max/√(1 + c/K)를 따르는지, c가 이론값
    c = (d_out + 1)·Σ_t‖x_t‖²‖g_t‖² / ‖Σ_t g_t x_tᵀ‖² (σ→0, 선형 근사)와 맞는지
T2: 토큰 사이 영향(β)이 있을 때 자기 토큰 손실만 보상(γ=0)하면 c_max < 1, 미래 토큰까지(γ=1) 보상하면 c_max ≈ 1
T3: 두 층을 한 forward에서 함께 흔들면(간섭) c가 커지는가
T4: 같은 forward 수의 weight-space ES(antithetic)와 비교
T5: draw 묶음(n개) 안에서 보상을 중심화하면 추정 크기가 (1 − 1/n)배

실행: python3 analysis/dust_toy.py → analysis/dust_toy_results.json, analysis/dust_toy_curves.npz
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
T, DIN, DH, C = 256, 16, 64, 32
KS = [4, 16, 64, 256, 1024, 4096]
REPS = 4
CHUNK = 128


def make_problem(seed=0, beta=0.0, train_steps=150):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, DIN))
    Wt = rng.normal(size=(C, DIN))
    y = np.argmax(X @ Wt.T + 0.5 * rng.normal(size=(T, C)), 1)
    W1 = rng.normal(size=(DH, DIN)) / np.sqrt(DIN)
    W2 = rng.normal(size=(C, DH)) / np.sqrt(DH)
    P = dict(X=X, y=y, W1=W1, W2=W2, beta=beta)
    for _ in range(train_steps):             # 논문처럼 backprop으로 학습한 체크포인트에서 잰다
        g1, g2, _ = backprop(P)
        P["W1"] = P["W1"] - 0.5 * g1 / T
        P["W2"] = P["W2"] - 0.5 * g2 / T
    return P


def mix(u, beta):
    """m_t = u_t + β·(1/t)Σ_{s<t} u_s (t ≥ 1). u: [..., T, d]."""
    cs = np.cumsum(u, axis=-2)
    prev = np.concatenate([np.zeros_like(u[..., :1, :]), cs[..., :-1, :]], axis=-2)
    cnt = np.arange(T).reshape(-1, 1).astype(float)
    cnt[0] = 1
    return u + beta * prev / cnt


def forward(P, jit_h=None, jit_z=None):
    h = P["X"] @ P["W1"].T
    if jit_h is not None:
        h = h + jit_h
    u = np.tanh(h)
    m = mix(u, P["beta"])
    z = m @ P["W2"].T
    if jit_z is not None:
        z = z + jit_z
    z = z - z.max(-1, keepdims=True)
    lse = np.log(np.exp(z).sum(-1))
    ell = lse - np.take_along_axis(z, P["y"].reshape((1,) * (z.ndim - 2) + (T, 1)).repeat(z.shape[0], 0) if z.ndim == 3
                                    else P["y"][:, None], -1)[..., 0]
    return ell, (h, u, m, z)


def backprop(P):
    ell, (h, u, m, z) = forward(P)
    p = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
    dz = p.copy()
    dz[np.arange(T), P["y"]] -= 1
    g2 = dz.T @ m
    dm = dz @ P["W2"]
    # m_s = u_s + β/s Σ_{r<s} u_r  →  du_t = dm_t + β Σ_{s>t} dm_s / s
    cnt = np.arange(T).astype(float)
    cnt[0] = 1
    w = dm / cnt[:, None]
    suffix = np.cumsum(w[::-1], 0)[::-1]
    future = np.concatenate([suffix[1:], np.zeros((1, DH))], 0)
    du = dm + P["beta"] * future
    dh = du * (1 - u ** 2)
    g1 = dh.T @ P["X"]
    return g1, g2, dict(dh=dh, dz=dz, m=m, dm_local=dm * (1 - u ** 2))


def cos(a, b):
    return float((a * b).sum() / np.linalg.norm(a) / np.linalg.norm(b))


def dust_estimate(P, K, rng, site="h", sigma=0.05, gamma=0.0, n=1, joint=False):
    """K draws, n개씩 묶어 한 번에. n > 1이면 묶음 안에서 보상을 중심화(공식 코드와 같음)."""
    clean, _ = forward(P)
    eh = np.zeros((T, DH))
    ez = np.zeros((T, C))
    done = 0
    while done < K:
        k = min(CHUNK, K - done)
        k = max(n, k - k % n)
        ah = rng.normal(size=(k, T, DH)) if (site == "h" or joint) else None
        az = rng.normal(size=(k, T, C)) if (site == "z" or joint) else None
        ell, _ = forward(P, None if ah is None else sigma * ah, None if az is None else sigma * az)
        r = ell - clean                                         # 손실 증가 (= −보상)
        if n > 1:
            rr = r.reshape(k // n, n, T)
            r = (rr - rr.mean(1, keepdims=True)).reshape(k, T)
        if gamma > 0:                                           # r_t = Σ_{s≥t} γ^{s−t} c_s
            out = np.zeros_like(r)
            acc = np.zeros(k)
            for t in range(T - 1, -1, -1):
                acc = r[:, t] + gamma * acc
                out[:, t] = acc
            r = out
        if ah is not None:
            eh += np.einsum("kt,ktd->td", r, ah) / sigma
        if az is not None:
            ez += np.einsum("kt,ktd->td", r, az) / sigma
        done += k
    G1 = (eh / K).T @ P["X"]
    G2 = (ez / K).T @ forward(P)[1][2]
    return G1, G2


def es_estimate(P, F, rng, sigma=0.01):
    """weight-space ES (antithetic): forward F회 = F/2 쌍, 보상은 전체 손실 하나."""
    clean_W1 = P["W1"].copy()
    G = np.zeros_like(clean_W1)
    pairs = max(1, F // 2)
    for _ in range(pairs):
        E = rng.normal(size=clean_W1.shape)
        P["W1"] = clean_W1 + sigma * E
        lp = forward(P)[0].sum()
        P["W1"] = clean_W1 - sigma * E
        lm = forward(P)[0].sum()
        G += (lp - lm) / (2 * sigma) * E
    P["W1"] = clean_W1
    return G / pairs


def law(K, cm, c):
    return cm / np.sqrt(1 + c / K)


def fit_law(ks, cs):
    p, _ = curve_fit(law, np.array(ks, float), np.array(cs), p0=(0.9, 100.0), bounds=([0, 0], [1, 1e9]))
    pred = law(np.array(ks, float), *p)
    return {"c_max": round(float(p[0]), 3), "c": round(float(p[1]), 1), "rmse": round(float(np.sqrt(np.mean((pred - cs) ** 2))), 4)}


def ladder(P, target, fn, seed):
    rng = np.random.default_rng(seed)
    out = []
    for K in KS:
        out.append(float(np.mean([cos(fn(K, rng), target) for _ in range(REPS)])))
    return out


def theory_c(G_true, g_tok, x_tok, d_out):
    num = (d_out + 1) * np.sum((g_tok ** 2).sum(1) * (x_tok ** 2).sum(1))
    return float(num / (G_true ** 2).sum())


if __name__ == "__main__":
    res, curves = {}, {"K": np.array(KS)}

    # T1: β = 0, 두 층 각각, σ 작게, 중심화 없음
    P = make_problem(beta=0.0)
    g1, g2, aux = backprop(P)
    m = aux["m"]
    l1 = ladder(P, g1, lambda K, r: dust_estimate(P, K, r, "h", sigma=0.01)[0], 1)
    l2 = ladder(P, g2, lambda K, r: dust_estimate(P, K, r, "z", sigma=0.01)[1], 2)
    res["T1"] = {"W1 (hidden, d_out=64)": {**fit_law(KS, l1), "c_theory": round(theory_c(g1, aux["dh"], P["X"], DH), 1), "cos": [round(x, 3) for x in l1]},
                 "W2 (head, d_out=32)": {**fit_law(KS, l2), "c_theory": round(theory_c(g2, aux["dz"], m, C), 1), "cos": [round(x, 3) for x in l2]}}
    curves["T1_W1"], curves["T1_W2"] = np.array(l1), np.array(l2)

    # T2: 토큰 사이 영향 β와 credit decay γ
    res["T2"] = {}
    for beta in (0.0, 1.0, 3.0):
        P = make_problem(beta=beta)
        g1, _, aux = backprop(P)
        local_target = aux["dm_local"].T @ P["X"]
        row = {"cos(local grad, full grad)": round(cos(local_target, g1), 3)}
        for gamma in (0.0, 1.0):
            l = ladder(P, g1, lambda K, r: dust_estimate(P, K, r, "h", sigma=0.01, gamma=gamma)[0], 3)
            row[f"gamma={gamma:g}"] = {**fit_law(KS, l), "cos": [round(x, 3) for x in l]}
            curves[f"T2_b{beta:g}_g{gamma:g}"] = np.array(l)
        res["T2"][f"beta={beta:g}"] = row

    # T3: 간섭 (β = 1, γ = 0)
    P = make_problem(beta=1.0)
    g1, g2, _ = backprop(P)
    sep = ladder(P, g1, lambda K, r: dust_estimate(P, K, r, "h", sigma=0.05)[0], 4)
    jnt = ladder(P, g1, lambda K, r: dust_estimate(P, K, r, "h", sigma=0.05, joint=True)[0], 4)
    res["T3"] = {"W1 separate pass": {**fit_law(KS, sep), "cos": [round(x, 3) for x in sep]},
                 "W1 jointly with head": {**fit_law(KS, jnt), "cos": [round(x, 3) for x in jnt]}}
    curves["T3_sep"], curves["T3_joint"] = np.array(sep), np.array(jnt)

    # T4: weight-space ES, 같은 forward 수
    es = ladder(P, g1, lambda K, r: es_estimate(P, K, r), 5)
    res["T4"] = {"W1 weight-space ES": {"cos": [round(x, 3) for x in es]},
                 "W1 Dust (같은 forward 수)": {"cos": [round(x, 3) for x in sep]},
                 "W1 parameters": DH * DIN, "tokens_per_forward": T}
    curves["T4_es"] = np.array(es)

    # T5: 묶음 중심화의 크기 효과 (선형 영역, β = 0)
    P = make_problem(beta=0.0)
    g1, _, _ = backprop(P)
    rng = np.random.default_rng(6)
    t5 = {}
    for n in (1, 2, 8, 16):
        G = np.mean([dust_estimate(P, 4096, rng, "h", sigma=0.01, n=n)[0] for _ in range(2)], 0)
        t5[f"n={n}"] = {"norm_ratio(proj on true)": round(float((G * g1).sum() / (g1 ** 2).sum()), 3),
                        "expected": round(1 - 1 / n, 3) if n > 1 else 1.0, "cos": round(cos(G, g1), 3)}
    res["T5"] = t5

    np.savez(ROOT / "analysis" / "dust_toy_curves.npz", **curves)
    (ROOT / "analysis" / "dust_toy_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))
