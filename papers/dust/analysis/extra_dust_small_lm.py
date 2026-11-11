"""Dust(토큰별 잡음 + 토큰별 손실 변화 보상)와 backprop, backprop + 가우스 잡음의 극초기 학습 비교 (torch CPU).

모델: 문자 단위 작은 causal Transformer (2층, 폭 64, 헤드 4, 문맥 64, 파라미터 가중치 행렬만, 정규화는 파라미터 없는 RMSNorm).
잡음 자리(site): 임베딩 출력, qkv, attention 출력 투영, MLP 입력(fc1), MLP 출력(fc2), 출력 head.
- Dust: 모든 자리에 토큰별 독립 잡음 σ·a_t를 한 forward에서 동시에 더한다. 각 토큰의 손실 변화 Δℓ를
  미래 토큰까지 합한 보상 r_t = Σ_{s≥t} Δℓ_s (γ = 1 가정)로 ĝ_t = (1/Kσ) Σ_k r_t^(k) a_t^(k)를 만들고,
  이 ĝ를 각 가중치의 gradient 추정 Ĝ_W = Σ_t ĝ_t x_tᵀ로 바꾼다. 가중치 업데이트는 Adam을 쓴다.
- backprop: 같은 모델, 같은 Adam.
- backprop + 가우스 잡음: backprop 중 같은 자리에 같은 σ의 잡음을 더한다(평활화 대조군).
데이터: 로컬 Python 표준 라이브러리 소스 텍스트(다운로드 없음), 앞 2%를 검증에 씀.

가정(논문에 없는 설정): σ = 0.1, K = 32 draw, 배치 8 × 문맥 64, γ = 1, 시드 2개, 학습률 3e-3.
실행: .venv/bin/python analysis/extra_dust_small_lm.py → analysis/extra_dust_small_lm_results.json
"""
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "4")

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cpu")
D, HEADS, LAYERS, CTX, VAL_FRAC = 64, 4, 2, 64, 0.02
B, K, SIGMA, LR = 8, 32, 0.1, 3e-3
STEPS = 1000
CHECK = [6, 10, 30, 61, 100, 300, 610, 1000]
SEEDS = [0, 1]


def load_corpus(max_chars=2_000_000):
    """표준 라이브러리 소스 파일을 읽어 문자열을 만든다(실행하지 않고 텍스트로만 읽음)."""
    base = Path(os.path.dirname(os.__file__))
    texts, total = [], 0
    for p in sorted(base.glob("*.py")):
        t = p.read_text(encoding="utf-8", errors="ignore")
        texts.append(t)
        total += len(t)
        if total >= max_chars:
            break
    s = "".join(texts)[:max_chars]
    vocab = sorted(set(s))
    stoi = {c: i for i, c in enumerate(vocab)}
    return torch.tensor([stoi[c] for c in s], dtype=torch.long), len(vocab)


def rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def init_params(V, seed):
    g = torch.Generator().manual_seed(seed)
    n = lambda *s, fan: torch.randn(*s, generator=g) / math.sqrt(fan)
    P = {"emb": torch.randn(V, D, generator=g) * 0.5}
    for i in range(LAYERS):
        P[f"qkv{i}"] = n(3 * D, D, fan=D)
        P[f"proj{i}"] = n(D, D, fan=D)
        P[f"fc1{i}"] = n(4 * D, D, fan=D)
        P[f"fc2{i}"] = n(D, 4 * D, fan=4 * D)
    P["head"] = n(V, D, fan=D)
    return {k: v.requires_grad_(True) for k, v in P.items()}


def site_dims(V):
    d = {"emb": D}
    for i in range(LAYERS):
        d[f"qkv{i}"] = 3 * D
        d[f"proj{i}"] = D
        d[f"fc1{i}"] = 4 * D
        d[f"fc2{i}"] = D
    d["head"] = V
    return d


def forward(P, idx, noise=None, cache=None):
    """idx: (N, T). noise: dict site -> (N, T, dim) 또는 None. cache: 입력 저장용 dict."""
    def add(site, y):
        if noise is not None:
            y = y + noise[site]
        return y

    x = add("emb", P["emb"][idx])
    if cache is not None:
        cache["emb_idx"] = idx
    for i in range(LAYERS):
        h = rms(x)
        if cache is not None:
            cache[f"qkv{i}"] = h
        qkv = add(f"qkv{i}", h @ P[f"qkv{i}"].T)
        N, T, _ = qkv.shape
        q, k, v = qkv.view(N, T, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(N, T, D)
        if cache is not None:
            cache[f"proj{i}"] = a
        x = x + add(f"proj{i}", a @ P[f"proj{i}"].T)
        h = rms(x)
        if cache is not None:
            cache[f"fc1{i}"] = h
        u = F.gelu(add(f"fc1{i}", h @ P[f"fc1{i}"].T))
        if cache is not None:
            cache[f"fc2{i}"] = u
        x = x + add(f"fc2{i}", u @ P[f"fc2{i}"].T)
    h = rms(x)
    if cache is not None:
        cache["head"] = h
    return add("head", h @ P["head"].T)


def per_token_loss(logits, tgt):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view(tgt.shape)


def sample_batch(corpus, g, n):
    starts = torch.randint(0, len(corpus) - CTX - 1, (n,), generator=g)
    idx = torch.stack([corpus[s:s + CTX + 1] for s in starts])
    return idx[:, :-1], idx[:, 1:]


@torch.no_grad()
def val_loss(P, val, V):
    g = torch.Generator().manual_seed(12345)
    x, y = sample_batch(val, g, 256)
    return float(per_token_loss(forward(P, x), y).mean())


def dust_step(P, x, y, dims, g, opt):
    cache = {}
    with torch.no_grad():
        logits0 = forward(P, x, cache=cache)
        l0 = per_token_loss(logits0, y)                     # (B, T)
        N, T = x.shape
        a = {s: torch.randn(K, N, T, d, generator=g) for s, d in dims.items()}
        noise = {s: (SIGMA * a[s]).reshape(K * N, T, dims[s]) for s in dims}
        xk = x.repeat(K, 1)
        yk = y.repeat(K, 1)
        lk = per_token_loss(forward(P, xk, noise=noise), yk).view(K, N, T)
        delta = lk - l0[None]
        r = delta.flip(-1).cumsum(-1).flip(-1)              # γ = 1: 미래 토큰까지 합산
        ghat = {s: torch.einsum("knt,kntd->ntd", r, a[s]) / (K * SIGMA) for s in dims}
    grads = {}
    scale = 1.0 / (N * T)
    for s in dims:
        if s == "emb":
            G = torch.zeros_like(P["emb"]).index_add_(0, cache["emb_idx"].reshape(-1),
                                                       ghat[s].reshape(-1, D))
        else:
            xin = cache[s].reshape(-1, cache[s].shape[-1])
            G = ghat[s].reshape(-1, dims[s]).T @ xin
        grads[s] = G * scale
    for s in dims:
        P[s].grad = grads[s]
    opt.step()
    opt.zero_grad(set_to_none=True)


def backprop_step(P, x, y, opt, noisy, dims, g):
    noise = None
    if noisy:
        N, T = x.shape
        noise = {s: SIGMA * torch.randn(N, T, dims[s], generator=g) for s in dims}
    loss = per_token_loss(forward(P, x, noise=noise), y).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


def run(method, seed, corpus, val, V, dims):
    P = init_params(V, seed)
    opt = torch.optim.Adam(P.values(), lr=LR)
    g = torch.Generator().manual_seed(seed + 7)
    curve = {}
    t0 = time.time()
    for step in range(1, STEPS + 1):
        x, y = sample_batch(corpus, g, B)
        if method == "dust":
            dust_step(P, x, y, dims, g, opt)
        else:
            backprop_step(P, x, y, opt, method == "backprop_noise", dims, g)
        if step in CHECK:
            curve[step] = round(val_loss(P, val, V), 4)
    return curve, round(time.time() - t0, 1)


def main():
    t_all = time.time()
    corpus_all, V = load_corpus()
    cut = int(len(corpus_all) * (1 - VAL_FRAC))
    corpus, val = corpus_all[:cut], corpus_all[cut:]
    dims = site_dims(V)
    res = {"device": str(DEVICE), "torch": torch.__version__, "numpy": np.__version__,
           "python": platform.python_version(), "threads": torch.get_num_threads(),
           "config": dict(d_model=D, heads=HEADS, layers=LAYERS, context=CTX, batch=B,
                          population_K=K, sigma=SIGMA, gamma=1.0, lr=LR, optimizer="Adam",
                          steps=STEPS, seeds=SEEDS, vocab=V, corpus_chars=len(corpus_all),
                          corpus="python stdlib sources (local text, no download)"),
           "val_nll_per_char": {}, "wall_time_s": {}}
    for method in ["backprop", "backprop_noise", "dust"]:
        per_seed, times = [], []
        for seed in SEEDS:
            curve, wt = run(method, seed, corpus, val, V, dims)
            per_seed.append(curve)
            times.append(wt)
            print(method, seed, curve, wt, "s", flush=True)
        res["val_nll_per_char"][method] = {
            str(s): {"mean": round(float(np.mean([c[s] for c in per_seed])), 4),
                     "per_seed": [c[s] for c in per_seed]} for s in CHECK}
        res["wall_time_s"][method] = times
    res["wall_time_total_s"] = round(time.time() - t_all, 1)
    (ROOT / "analysis" / "extra_dust_small_lm_results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1))
    print("total", res["wall_time_total_s"], "s")


if __name__ == "__main__":
    main()
