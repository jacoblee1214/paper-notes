"""탐색 잡음 스케줄 toy (CPU, numpy). 실행: python3 analysis/exploration_toy.py

논문의 질문("학습되는 σ 대신 σ를 학습 step의 함수로 정하면 최종 결정론적 성공이 오르는가")을
한 단계짜리 문맥 밴딧으로 줄여서 봅니다. 실제 dexterous 과제와 수치를 비교하면 안 되고, 경향만 봅니다.

- 관측 o ∈ [-1, 1]², 행동 a ∈ R⁴, 목표 a*(o) = M o + b
- 보상 r = exp(−‖a − a*‖² / (2w²)): 목표 근처에서만 신호가 있는 "접촉 발견"형 보상
- 성공(Target과 같은 역할): 결정론적 평균 행동이 ‖μ(o) − a*(o)‖ < tol
- 두 regime: near(|b| 0.3, w 0.08, tol 0.03, 초기 위치에서도 신호가 있음)와
  far(|b| 0.7, w 0.05, tol 0.02, 넓게 탐색해야 신호를 찾음). 스케줄을 유리하게 맞추려고 고른 값이 아니라,
  "탐색이 필요 없는 경우"와 "필요한 경우"를 나눠 보려는 두 점입니다
- 정책: 선형 평균 μ = W o + c, Gaussian. PPO-clip(ε 0.2, 5 epoch, minibatch 128), 배치 내 advantage 정규화,
  Adam 하나로 모든 매개변수(lr 3e-4, SB3처럼 log σ도 같은 lr), 엔트로피 계수 1e-3
- lr × Adam step 수 ≈ 0.9로 두어, 논문 설정(1e-5 × 97,656 ≈ 0.98, `04` C6)과 log σ의 최대 이동 폭을 맞춤
조건: learned(차원별 log σ 학습, 초기 0.20) · scheduled(Eq. 4 모양) · linear(0.20→0.04) · fixed 0.20 / 0.10 / 0.04
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DA, DO = 4, 2
REGIMES = {"near": (0.3, 0.08, 0.03), "far": (0.7, 0.05, 0.02)}  # (|b|, w, tol)
ITERS, BATCH, MB, EPOCHS = 150, 512, 128, 5
LR, CLIP, ENT = 3e-4, 0.2, 1e-3
SEEDS = 10


def sigma_of(cond: str, frac: float) -> float | None:
    """학습 진행률 frac ∈ [0,1]에서의 σ. None이면 학습되는 σ."""
    if cond == "learned":
        return None
    if cond == "scheduled":  # Eq. 4를 5M → 1로 줄인 모양: 0.20→0.10 (0–0.4), 0.10→0.05 (0.4–0.8), 0.04
        if frac < 0.4:
            return 0.20 - 0.10 * frac / 0.4
        if frac < 0.8:
            return 0.10 - 0.05 * (frac - 0.4) / 0.4
        return 0.04
    if cond == "linear":
        return 0.20 - 0.16 * frac
    return float(cond.split("_")[1])


class Adam:
    def __init__(self, shapes, lr):
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]
        self.t, self.lr = 0, lr

    def step(self, params, grads):
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            mh, vh = m / (1 - 0.9 ** self.t), v / (1 - 0.999 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + 1e-8)  # 손실 최소화 방향


def task(rng, dist):
    M = rng.normal(0, 0.15, size=(DA, DO))
    b = rng.normal(0, 1, size=DA)
    b *= dist / np.linalg.norm(b)
    return M, b


def run(cond: str, seed: int, regime: str) -> dict:
    dist, W_BUMP, TOL = REGIMES[regime]
    rng = np.random.default_rng(seed)
    M, b = task(np.random.default_rng(1000 + seed), dist)  # 과제는 seed마다 다르지만 조건 사이에는 같다
    W, c = np.zeros((DA, DO)), np.zeros(DA)
    log_std = np.full(DA, np.log(0.20))
    opt = Adam([W.shape, c.shape, log_std.shape], LR)
    hist = []
    for it in range(ITERS):
        sig = sigma_of(cond, it / ITERS)
        if sig is not None:
            log_std[:] = np.log(sig)
        o = rng.uniform(-1, 1, size=(BATCH, DO))
        mu = o @ W.T + c
        std = np.exp(log_std)
        a = mu + std * rng.normal(size=(BATCH, DA))
        target = o @ M.T + b
        r = np.exp(-((a - target) ** 2).sum(1) / (2 * W_BUMP ** 2))
        adv = (r - r.mean()) / (r.std() + 1e-8)
        old_lp = (-((a - mu) ** 2) / (2 * std ** 2) - log_std).sum(1)
        for _ in range(EPOCHS):
            perm = rng.permutation(BATCH)
            for s in range(0, BATCH, MB):
                idx = perm[s:s + MB]
                oo, aa, A, olp = o[idx], a[idx], adv[idx], old_lp[idx]
                mu_ = oo @ W.T + c
                std_ = np.exp(log_std)
                z = (aa - mu_) / std_
                lp = (-0.5 * z ** 2 - log_std).sum(1)
                ratio = np.exp(lp - olp)
                active = ~(((A > 0) & (ratio > 1 + CLIP)) | ((A < 0) & (ratio < 1 - CLIP)))
                coef = (A * ratio * active)[:, None] / len(idx)  # d(목적함수)/d(log π)
                g_mu = coef * (z / std_)                        # d log π / d μ
                gW, gc = -(g_mu.T @ oo), -g_mu.sum(0)           # 손실 = −목적함수
                g_ls = -(coef * (z ** 2 - 1)).sum(0) - ENT      # 엔트로피 보너스(차원당 +1)
                if sig is not None:
                    g_ls = np.zeros_like(g_ls)
                opt.step([W, c, log_std], [gW, gc, g_ls])
        if it % 10 == 9 or it == ITERS - 1:
            hist.append(round(float(r.mean()), 4))
    # 최종 평가: 결정론적, 자기 σ로 확률적, 공통 σ 0.03/0.10
    te = np.random.default_rng(seed + 77)
    o = te.uniform(-1, 1, size=(2000, DO))
    mu, target = o @ W.T + c, o @ M.T + b
    err = np.linalg.norm(mu - target, axis=1)
    out = {"det_success": float((err < TOL).mean()), "final_std": round(float(np.exp(log_std).mean()), 4),
           "mean_err": round(float(err.mean()), 4), "train_reward_curve": hist}
    own = np.exp(log_std)
    for name, s in (("own_noise", own), ("common_0.03", 0.03), ("common_0.10", 0.10)):
        a = mu + s * te.normal(size=mu.shape)
        out[f"success_{name}"] = float((np.linalg.norm(a - target, axis=1) < TOL).mean())
    out["train_reward_auc"] = round(float(np.mean(hist)), 4)
    return out


CONDS = ["learned", "scheduled", "linear", "fixed_0.20", "fixed_0.10", "fixed_0.04"]


def regime_summary(regime: str) -> dict:
    dist, W_BUMP, TOL = REGIMES[regime]
    res = {c: [run(c, s, regime) for s in range(SEEDS)] for c in CONDS}
    summary = {}
    for c, runs in res.items():
        det = np.array([r["det_success"] for r in runs])
        summary[c] = {
            "det_success_mean": round(float(det.mean()), 3), "det_success_sd": round(float(det.std(ddof=1)), 3),
            "seeds_ge_0.9": int((det >= 0.9).sum()),
            "final_std_mean": round(float(np.mean([r["final_std"] for r in runs])), 4),
            "mean_err": round(float(np.mean([r["mean_err"] for r in runs])), 4),
            "success_own_noise": round(float(np.mean([r["success_own_noise"] for r in runs])), 3),
            "success_common_0.03": round(float(np.mean([r["success_common_0.03"] for r in runs])), 3),
            "success_common_0.10": round(float(np.mean([r["success_common_0.10"] for r in runs])), 3),
            "train_reward_auc": round(float(np.mean([r["train_reward_auc"] for r in runs])), 4),
        }
    # 쌍(같은 seed = 같은 과제) 비교: scheduled vs learned
    pairs = [(a["det_success"], b["det_success"]) for a, b in zip(res["learned"], res["scheduled"])]
    summary["paired_scheduled_vs_learned"] = {
        "wins": int(sum(s > l for l, s in pairs)), "ties": int(sum(s == l for l, s in pairs)),
        "losses": int(sum(s < l for l, s in pairs))}
    # 이론값: 평균이 목표에서 Δ만큼 떨어져 있을 때 기대 보상 (가우시안 컨볼루션)
    th = {}
    for s in (0.04, 0.10, 0.20):
        for d in (0.3, 0.1, 0.0):
            v = W_BUMP ** 2 + s ** 2
            th[f"sigma{s}_delta{d}"] = float(f"{(W_BUMP ** 2 / v) ** (DA / 2) * np.exp(-d ** 2 / (2 * v)):.3g}")
    return {"regime": {"dist": dist, "w": W_BUMP, "tol": TOL}, "summary": summary, "expected_reward_theory": th,
           "per_seed_det_success": {c: [r["det_success"] for r in runs] for c, runs in res.items()},
           "train_reward_curves_mean": {c: [round(float(np.mean([r["train_reward_curve"][i] for r in runs])), 4)
                                            for i in range(len(runs[0]["train_reward_curve"]))] for c, runs in res.items()}}


def main() -> None:
    out = {"setup": {"DA": DA, "ITERS": ITERS, "BATCH": BATCH, "MB": MB, "EPOCHS": EPOCHS, "LR": LR,
                     "adam_steps": ITERS * EPOCHS * BATCH // MB, "lr_x_steps": LR * (ITERS * EPOCHS * BATCH // MB),
                     "SEEDS": SEEDS}}
    for regime in REGIMES:
        out[regime] = regime_summary(regime)
        print(regime, json.dumps({c: (v["det_success_mean"], v["seeds_ge_0.9"], v["final_std_mean"], v["success_own_noise"])
                                  for c, v in out[regime]["summary"].items() if c in CONDS}, ensure_ascii=False))
        print("  paired", out[regime]["summary"]["paired_scheduled_vs_learned"])
    path = ROOT / "analysis" / "exploration_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("wrote", path)


if __name__ == "__main__":
    main()
