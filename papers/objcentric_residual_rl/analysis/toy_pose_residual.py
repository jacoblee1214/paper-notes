"""토이 C: 객체 포즈 잔차가 편향된 기반 정책을 고치는가, 포즈 오차가 어디서 깨뜨리는가 (numpy, 단일 프로세스, 수 분 이내).

모델: 2차원 점 그리퍼 x, 물체 o (m 단위). 한 스텝 이동은 ±2 cm로 자른다. T = 60 스텝, 성공은 마지막 스텝에서 |o - x| < 1 cm.
기반 정책 (VLA 역할): 물체 위치를 자기 방식으로 본다. o_v = o + e_v, e_v ~ N(0, 2 cm²)는 에피소드마다 고정된 기반 정책의 오차다.
  a_base = clip(0.5 (o_v - x)). 고정점에서 x = o_v, 즉 e_v만큼 빗나간다. 포즈 입력과 독립인 오차라서 잔차가 포즈로 고쳐야 한다.
  (처음 설계는 a_base를 참 물체 위치로 계산해 잔차가 a_base에서 물체 위치를 새는 문제가 있었다. 결과 1.0이 그 증거라 버렸다.)
잔차 정책: 2층 MLP(은닉 16, tanh). 입력 = [o_hat (2), x (2), a_base (2)], 출력 = 0.02 tanh(·).
  o_hat는 o에 포즈 오차를 더한 값이다. 드롭아웃이면 o_hat = (0, 0)으로 둔다(논문 식 5의 "포즈 벡터 0").
  x는 두 영역에서 같은 값(η ≈ 0)이라고 가정한다. 이는 논문도 같은 가정을 쓴다.
  주의: a_base는 o_v를 통해 o와 상관되어 있다. 그래서 잔차는 o_hat와 a_base를 함께 써서 o를 일부 추정할 수 있다. 이 누설 경로는 남겨 둔다.
학습: ES(antithetic, 인구 64, 세대 300, 에피소드 32). 적합도 = -최종 거리.
  - none: 학습 중 포즈 오차 없음
  - aug: 논문식 증강. 스텝마다 iid N(0, 0.005²) + 드롭아웃 0.1
  - aug+bias: aug에 에피소드마다 상수 편향 N(0, 0.01²)을 더함
평가(고정 시드, 1000 에피소드, 한 번에 한 요인만 바꿈): iid 오차 σ, 에피소드 편향 σ_β, 드롭아웃 p.

실행: python3 analysis/toy_pose_residual.py  -> analysis/toy_pose_residual_results.json
"""
import json
import math
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_pose_residual_results.json"
T, STEP, TOL = 60, 0.02, 0.01
H, RMAX = 16, 0.02
D_IN = 6
SD_BASE_ERR = 0.02
N_EVAL = 1000


def unpack(theta, P):
    i = 0
    W1 = theta[:, i:i + D_IN * H].reshape(P, D_IN, H); i += D_IN * H
    b1 = theta[:, i:i + H]; i += H
    W2 = theta[:, i:i + H * 2].reshape(P, H, 2); i += H * 2
    b2 = theta[:, i:i + 2]
    return W1, b1, W2, b2


def nparam():
    return D_IN * H + H + H * 2 + 2


def rollout(theta, rng, n_ep, iid=0.0, bias_sd=0.0, drop=0.0, residual=True, analytic=False):
    """theta: (P, nparam) 또는 None(기반 정책만). 반환: 최종 거리 (P, n_ep)."""
    P = 1 if theta is None else theta.shape[0]
    o = rng.uniform(-0.1, 0.1, size=(n_ep, 2))
    x = np.zeros((P, n_ep, 2))
    beta = rng.normal(0, bias_sd, size=(n_ep, 2)) if bias_sd > 0 else np.zeros((n_ep, 2))
    e_v = rng.normal(0, SD_BASE_ERR, size=(n_ep, 2))  # 기반 정책의 고정 오차
    if residual and theta is not None:
        W1, b1, W2, b2 = unpack(theta, P)
    for _ in range(T):
        a_base = np.clip(0.5 * ((o + e_v)[None] - x), -STEP, STEP)
        if analytic:
            # 상한 참조: 잔차가 오차 없는 포즈로 정확히 0.5(o - x)를 내는 해석해
            o_hat = np.broadcast_to(o[None] + beta[None], x.shape)
            a = np.clip(0.5 * (o_hat - x), -STEP, STEP)
        elif residual and theta is not None:
            o_hat = np.broadcast_to(o[None] + beta[None], x.shape).copy()
            if iid > 0:
                o_hat = o_hat + rng.normal(0, iid, size=o_hat.shape)
            if drop > 0:
                keep = rng.random((P, n_ep, 1)) >= drop
                o_hat = np.where(keep, o_hat, 0.0)
            feat = np.concatenate([o_hat, x, a_base], axis=-1)  # (P, n, 6)
            h = np.tanh(np.einsum("pnd,pdh->pnh", feat, W1) + b1[:, None, :])
            r = RMAX * np.tanh(np.einsum("pnh,phk->pnk", h, W2) + b2[:, None, :])
            a = np.clip(a_base + r, -STEP, STEP)
        else:
            a = a_base
        x = x + a
    return np.linalg.norm(o[None] - x, axis=-1)


def train(mode, rng, gens=800, pop=128, n_ep=32, sigma=0.1, lr=0.05):
    theta = rng.normal(0, 0.1, size=(1, nparam()))[0]
    iid, bias_sd, drop = {"none": (0, 0, 0), "aug": (0.005, 0, 0.1), "aug+bias": (0.005, 0.01, 0.1)}[mode]
    m = np.zeros_like(theta); v = np.zeros_like(theta)
    for g in range(1, gens + 1):
        eps = rng.normal(size=(pop // 2, nparam()))
        eps = np.concatenate([eps, -eps])
        cand = theta[None] + sigma * eps
        dist = rollout(cand, rng, n_ep, iid, bias_sd, drop)  # (pop, n_ep)
        fit = -dist.mean(axis=1)
        adv = (fit - fit.mean()) / (fit.std() + 1e-8)
        grad = (adv[:, None] * eps).mean(axis=0) / sigma
        m = 0.9 * m + 0.1 * grad; v = 0.999 * v + 0.001 * grad ** 2
        mh = m / (1 - 0.9 ** g); vh = v / (1 - 0.999 ** g)
        theta = theta + lr * mh / (np.sqrt(vh) + 1e-8)
    return theta


def success(dist):
    return float((dist < TOL).mean())


def wilson(k, n, z=1.959964):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def evaluate(theta, residual, iid=0.0, bias_sd=0.0, drop=0.0, seed=2024, analytic=False):
    rng = np.random.default_rng(seed)
    dist = rollout(theta[None] if theta is not None else None, rng, N_EVAL, iid, bias_sd, drop, residual=residual, analytic=analytic)
    d = dist.reshape(-1)
    k = int((d < TOL).sum())
    return {"success": k / N_EVAL, "wilson95": wilson(k, N_EVAL), "median_final_err_m": float(np.median(d))}


def main():
    t0 = time.time()
    res = {"setup": {"T": T, "step_max_m": STEP, "tol_m": TOL, "base_error_sd_m": SD_BASE_ERR,
                     "residual_max_m": RMAX, "hidden": H, "eval_episodes": N_EVAL,
                     "ES": {"gens": 800, "pop": 128, "episodes_per_candidate": 32, "sigma": 0.1, "lr": 0.05}}}
    base = evaluate(None, residual=False)
    res["base_only"] = base
    res["oracle_analytic_clean_pose"] = evaluate(None, residual=True, analytic=True)
    res["oracle_analytic_bias_0.01"] = evaluate(None, residual=True, analytic=True, bias_sd=0.01)
    thetas = {}
    for mode in ["none", "aug", "aug+bias"]:
        rng = np.random.default_rng({"none": 1, "aug": 2, "aug+bias": 3}[mode])
        thetas[mode] = train(mode, rng)
    res["train_time_s"] = round(time.time() - t0, 1)

    grids = {
        "iid_sd_m": [0.0, 0.005, 0.01, 0.02],
        "episode_bias_sd_m": [0.0, 0.005, 0.01, 0.015, 0.02],
        "dropout_p": [0.0, 0.1, 0.3],
    }
    res["sweep"] = {}
    for mode, th in thetas.items():
        out = {}
        for axis, vals in grids.items():
            rows = []
            for v in vals:
                kw = {"iid_sd_m": {"iid": v}, "episode_bias_sd_m": {"bias_sd": v}, "dropout_p": {"drop": v}}[axis]
                e = evaluate(th, residual=True, **kw)
                rows.append({"value": v, **e})
            out[axis] = rows
        res["sweep"][mode] = out
    # 기반 정책 (편향만): 같은 조건에서 잔차가 없을 때
    res["base_sweep_iid_bias"] = {
        "episode_bias_sd_m": [{"value": 0.0, **evaluate(None, residual=False)}],
    }
    res["elapsed_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT, "elapsed", res["elapsed_s"], "s")
    print("base only success:", base["success"], "median err:", round(base["median_final_err_m"], 4))
    for mode in thetas:
        for axis in grids:
            print(mode, axis, [(r["value"], round(r["success"], 3)) for r in res["sweep"][mode][axis]])


if __name__ == "__main__":
    main()
