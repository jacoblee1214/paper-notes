"""추가 검증 1: 학습된 작은 MLP velocity 모델로 Fig. 9 경로 17개를 평가 (CPU numpy 1스레드, 시드 3개).

noise_toy.py의 세계(다봉 Gaussian 혼합, 영상 8차원·행동 4차원, 봉우리 4개, base 폭 0.15)와 경로 식(schedule),
Fig. 9 값, σ = f5(1 − u)의 학습 분포를 그대로 옮긴다(numpy 함수는 noise_toy.py에서 복사: 그 파일은 import 시 바로 실행됨).
- 학습: 두 스트림의 σ를 각자 독립으로 균등하게 뽑고(u ~ U(0,1), σ = f5(1 − u)), x_σ = (1 − σ)x0 + σε,
  목표 ε − x0 (코드 부호). 입력 [x_t, σ_v, σ_a], 2층 tanh MLP(폭 128), Adam 1e-3, 배치 512, 15000회, 손실은 가중치 없는 MSE.
  (코드의 손실 가중치 w(σ)는 넣지 않았다: 어느 스트림의 σ에 곱하는지 확인하지 못함)
- 평가: 경로 17개 각각 N=10 step Euler, 초기 잡음 M=20000 (noise_toy와 같은 규모). 지표는 action_valid(행동이 봉우리 99% 영역 안)와
  영상–행동 같은 봉우리 비율. 이상적 사후 평균 결과는 noise_toy_results.json의 base 항목을 그대로 읽는다.
- 분석: 학습 모델 대 Fig. 9 Spearman, 경로 통계(행동 최대 step, 행동 갱신 횟수)와의 상관, 이상적 denoiser와 순서가 갈리는 경로.
실행: python3 analysis/extra_learned_denoiser.py  (수 분 이내, 1스레드)
출력: analysis/extra_learned_denoiser_results.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "extra_learned_denoiser_results.json"
IDEAL = HERE / "noise_toy_results.json"
SHIFT = 5.0
N_STEPS = 10
K, DA, DV = 4, 4, 8
HID = 128
ITERS = 15000
BATCH = 512
SEEDS = (0, 1, 2)
M_EVAL = 20000
S_BASE = 0.15
t0 = time.time()

FIG9 = {("sync", None, 1.0, 0.0): 93.0}
for a, v in zip((4, 8, 16, 32), (91.0, 90.7, 89.7, 88.5)):
    FIG9[("VS", "action", float(a), 0.0)] = v
for a, v in zip((4, 8, 16, 32), (91.8, 91.1, 91.9, 91.5)):
    FIG9[("VS", "video", float(a), 0.0)] = v
for o, v in zip((0.2, 0.4, 0.6, 0.8), (92.2, 92.0, 92.3, 92.3)):
    FIG9[("LO", "action", 1.0, o)] = v
for o, v in zip((0.2, 0.4, 0.6, 0.8), (91.7, 91.1, 91.5, 91.2)):
    FIG9[("LO", "video", 1.0, o)] = v


def alpha_shift(u, shift):
    return shift * u / (1.0 + (shift - 1.0) * u)


def schedule(lead, alpha, offset, n=N_STEPS):
    """noise_toy.schedule과 같음. σ 쌍 (n+1, 2), 마지막은 (0, 0)."""
    s = np.linspace(1.0, 0.0, n + 1, dtype=np.float32)[:-1]
    lead_sigma = alpha_shift(s, np.float32(1.0 / alpha))
    lag_sigma = np.minimum(s / (1.0 - offset), 1.0) if offset > 0 else s
    if lead == "video" or lead is None:
        v, a = lead_sigma, lag_sigma
    else:
        v, a = lag_sigma, lead_sigma
    sv, sa = alpha_shift(v, SHIFT), alpha_shift(a, SHIFT)
    return np.stack([np.append(sv, 0.0), np.append(sa, 0.0)], 1).astype(np.float64)


def name_of(key):
    fam, lead, a, o = key
    if fam == "sync":
        return "sync"
    return f"{fam} {lead}-lead " + (f"α={int(a)}" if fam == "VS" else f"o={o}")


# noise_toy.py와 같은 난수 순서로 세계를 만든다
rng0 = np.random.default_rng(0)
mu_a = rng0.normal(size=(K, DA))
mu_a = 1.5 * mu_a / np.linalg.norm(mu_a, axis=1, keepdims=True)
G = np.linalg.qr(rng0.normal(size=(DV, DA)))[0] * 1.5


def make_world(s_a, s_v):
    mu = np.concatenate([mu_a @ G.T, mu_a], 1)
    Sa = s_a ** 2 * np.eye(DA)
    Sig = np.block([[G @ Sa @ G.T + s_v ** 2 * np.eye(DV), G @ Sa], [Sa @ G.T, Sa]])
    return mu, Sig


def init(sizes, rng):
    P = []
    for i, o in zip(sizes[:-1], sizes[1:]):
        P += [rng.normal(0, np.sqrt(1.0 / i), (i, o)), np.zeros(o)]
    return P


def fwd(P, X):
    hs = [X]
    h = X
    L = len(P) // 2
    for l in range(L):
        z = h @ P[2 * l] + P[2 * l + 1]
        h = np.tanh(z) if l < L - 1 else z
        hs.append(h)
    return h, hs


def bwd(P, hs, dout):
    g = [None] * len(P)
    L = len(P) // 2
    d = dout
    for l in reversed(range(L)):
        if l < L - 1:
            d = d * (1 - hs[l + 1] ** 2)
        g[2 * l] = hs[l].T @ d
        g[2 * l + 1] = d.sum(0)
        d = d @ P[2 * l].T
    return g


class Adam:
    def __init__(self, P, lr=1e-3):
        self.m = [np.zeros_like(p) for p in P]
        self.v = [np.zeros_like(p) for p in P]
        self.t = 0
        self.lr = lr

    def step(self, P, g):
        self.t += 1
        for i in range(len(P)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g[i]
            self.v[i] = 0.999 * self.v[i] + 0.001 * g[i] ** 2
            mh = self.m[i] / (1 - 0.9 ** self.t)
            vh = self.v[i] / (1 - 0.999 ** self.t)
            P[i] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def train(seed):
    rng = np.random.default_rng(100 + seed)
    mu, Sig = make_world(S_BASE, S_BASE)
    L = np.linalg.cholesky(Sig)
    P = init([DV + DA + 2, HID, HID, DV + DA], rng)
    opt = Adam(P)
    dvec = np.arange(DV + DA)
    for _ in range(ITERS):
        k = rng.integers(0, K, BATCH)
        x0 = mu[k] + rng.normal(size=(BATCH, DV + DA)) @ L.T
        eps = rng.normal(size=(BATCH, DV + DA))
        sv = alpha_shift(1 - rng.uniform(size=BATCH), SHIFT)     # 영상 스트림 독립 σ
        sa = alpha_shift(1 - rng.uniform(size=BATCH), SHIFT)     # 행동 스트림 독립 σ
        sig = np.concatenate([np.repeat(sv[:, None], DV, 1), np.repeat(sa[:, None], DA, 1)], 1)
        xt = (1 - sig) * x0 + sig * eps
        inp = np.concatenate([xt, sv[:, None], sa[:, None]], 1)
        pred, hs = fwd(P, inp)
        dout = 2 * (pred - (eps - x0)) / (BATCH * (DV + DA))
        opt.step(P, bwd(P, hs, dout))
    return P, float(np.mean((pred - (eps - x0)) ** 2))


def sample_learned(P, path, rng, M):
    x = rng.normal(size=(M, DV + DA))
    for k in range(len(path) - 1):
        (sv, sa), (nv, na) = path[k], path[k + 1]
        if sv == nv and sa == na:
            continue
        inp = np.concatenate([x, np.full((M, 1), sv), np.full((M, 1), sa)], 1)
        v, _ = fwd(P, inp)
        step = np.concatenate([np.full(DV, nv - sv), np.full(DA, na - sa)])
        x = x + step * v
    return x


def metrics(xs):
    v, a = xs[:, :DV], xs[:, DV:]
    da2 = ((a[:, None, :] - mu_a[None]) ** 2).sum(-1) / S_BASE ** 2
    ka = da2.argmin(1)
    valid = da2.min(1) < stats.chi2.ppf(0.99, DA)
    dv2 = ((v[:, None, :] - (mu_a @ G.T)[None]) ** 2).sum(-1)
    kv = dv2.argmin(1)
    return {"action_valid": float(valid.mean()), "same_mode": float((ka == kv).mean())}


def main():
    keys = list(FIG9)
    paths = {name_of(k): schedule(k[1], k[2], k[3]) for k in keys}
    ideal = json.loads(IDEAL.read_text())
    ideal_base = ideal["B_gmm_ideal_denoiser"]["base (s=0.15)"]["per_schedule"]
    pstats = ideal["path_stats"]
    fig9 = np.array([FIG9[k] for k in keys])
    names = [name_of(k) for k in keys]

    per_seed = {nm: [] for nm in names}
    train_mse = []
    for sd in SEEDS:
        P, mse = train(sd)
        train_mse.append(mse)
        rng = np.random.default_rng(1000 + sd)
        for nm in names:
            per_seed[nm].append(metrics(sample_learned(P, paths[nm], rng, M_EVAL)))
        print("seed", sd, "train mse", round(mse, 4), round(time.time() - t0, 1), "s", flush=True)

    learned = {nm: {"action_valid_mean": round(float(np.mean([r["action_valid"] for r in per_seed[nm]])), 4),
                    "action_valid_sd": round(float(np.std([r["action_valid"] for r in per_seed[nm]], ddof=1)), 4),
                    "same_mode_mean": round(float(np.mean([r["same_mode"] for r in per_seed[nm]])), 4)}
               for nm in names}
    lv = np.array([learned[nm]["action_valid_mean"] for nm in names])
    iv = np.array([ideal_base[nm]["action_valid"] for nm in names])
    out = {"setup": {"hidden": HID, "iters": ITERS, "batch": BATCH, "seeds": list(SEEDS), "M_eval": M_EVAL,
                     "loss": "unweighted MSE to eps - x0", "train_mse_per_seed": [round(m, 4) for m in train_mse]},
           "per_path": {nm: {"fig9": FIG9[k], "learned": learned[nm], "ideal_valid": round(float(ideal_base[nm]["action_valid"]), 4),
                             "max_action_du": pstats[nm]["max_action_du"], "action_updates": pstats[nm]["action_updates"]}
                        for k, nm in zip(keys, names)}}
    rho_l, p_l = stats.spearmanr(lv, fig9)
    rho_i, p_i = stats.spearmanr(iv, fig9)
    out["spearman_vs_fig9"] = {"learned": {"rho": round(float(rho_l), 3), "p": float(f"{p_l:.3g}")},
                               "ideal": {"rho": round(float(rho_i), 3), "p": float(f"{p_i:.3g}")}}
    stat_rows = {}
    for f in ("max_action_du", "action_updates"):
        x = np.array([pstats[nm][f] for nm in names])
        r1, p1 = stats.spearmanr(x, fig9)
        r2, p2 = stats.spearmanr(x, lv)
        stat_rows[f] = {"vs_fig9": {"rho": round(float(r1), 3), "p": float(f"{p1:.3g}")},
                        "vs_learned_valid": {"rho": round(float(r2), 3), "p": float(f"{p2:.3g}")}}
    out["path_stat_correlations"] = stat_rows
    # 행동 선행 α 계열 안에서만 본 상관 (α=4,8,16,32), 선형 지연 계열 (o=0.2–0.8)
    fam_a = [nm for nm in names if nm.startswith("VS action-lead")]
    fam_o = [nm for nm in names if nm.startswith("LO action-lead")]
    for tag, fam in (("action_lead_alpha", fam_a), ("action_lead_linear_delay", fam_o)):
        f9 = [FIG9[keys[names.index(nm)]] for nm in fam]
        lvv = [learned[nm]["action_valid_mean"] for nm in fam]
        out.setdefault("within_family", {})[tag] = {
            "n": len(fam), "learned_vs_fig9": round(float(stats.spearmanr(lvv, f9)[0]), 3),
            "learned_valid": {nm: learned[nm]["action_valid_mean"] for nm in fam}}
    out["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", OUT.name, out["runtime_s"], "s")


if __name__ == "__main__":
    main()
