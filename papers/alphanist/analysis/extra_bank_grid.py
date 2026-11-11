"""추가 검증: 시작 상태 bank 샘플러(식 4, SGS T=2, 균등)와 보상 구조(이진, dense+종료, OmniReset식)의 교차 비교 (numpy, CPU).

실행: python3 analysis/extra_bank_grid.py  → analysis/extra_bank_grid_results.json

모형
- 진행도 K = 20 칸짜리 사슬. 행동은 "시도"(확률 σ(θ_k))이고, 시도가 성공하면 확률 0.9로 한 칸 전진한다.
  정책은 칸마다 하나의 로짓 θ_k(표 형태)이고, 과제마다 하나의 시작 칸을 가진다.
- bank: 과제 M = 200개, 시작 칸은 0..19에 10개씩 고르게 배정. 시험 분포는 bank 위의 균등(= 시작 칸 균등)이다.
- 학습: 매 iteration마다 샘플러가 과제 16개를 고르고(복원 추출), 그 시작 칸에서 벡터화된 에피소드를 굴린다.
  T = 60 step, 할인 γ = 0.99, REINFORCE + 배치 평균 기준선. lr = 1.0, 1,500 iteration.
- 샘플러: 균등, 식 4(q ∝ ε_q + √(p̂(1−p̂)), ε_q ∈ {1e-4, 1e-2, 1e-1}), SGS(T = 2, q ∝ exp(ℓ/2), 논문 식의 조작 설정과 같은 w).
  p̂는 과제별 최근 50개 결과의 평균이고, 결과가 없으면 0.5다.
- 보상 구조
  - binary: 완료(칸 K 도달) 시 1을 주고 종료.
  - dense+종료: 매 step 0.02·k/K, 완료 시 1을 주고 종료 (식 7의 c2+c3 = 0.02와 같은 크기).
  - OmniReset식: 매 step 0.02·k/K, 완료 상태에 머무는 동안 매 step 0.1, 종료하지 않음.
- 측정(시험 분포 기준, 결정적): 현재 정책의 시작 칸별 horizon 안 완료 확률을 후진 DP로 정확히 계산하고 평균한다.
  학습 곡선은 100 iteration마다 기록한다. 보조로 완료 직전 칸(K−1)의 시도 확률 σ(θ_{K−1})을 본다(완료를 미루는지).
- 시드 5개.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
K, T, ADV, GAMMA, B_EP, ITERS, LR, WIN = 20, 60, 0.9, 0.99, 16, 1500, 1.0, 50
THETA0 = -2.0  # 초기 시도 확률 0.12: 초기 정책의 시험 완료율은 약 0.32(시작 칸 0은 0, 칸 19는 1)
M = 200
SEEDS = (0, 1, 2, 3, 4)
SAMPLERS = ("uniform", "eq4_1e-4", "eq4_1e-2", "eq4_1e-1", "sgs_T2")
REWARDS = ("binary", "dense_term", "omni")
EVAL_EVERY = 25
STARTS = np.repeat(np.arange(K), M // K)  # 과제별 시작 칸


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def exact_success(theta):
    """시작 칸별 horizon 안 완료 확률 (후진 DP). 완료는 칸 K 도달, 완료 후에는 머무는 것으로 본다."""
    p = sigmoid(theta)
    V = np.zeros(K + 1)
    V[K] = 1.0
    for _ in range(T):
        Vn = np.empty(K + 1)
        Vn[K] = 1.0
        Vn[:K] = p * (ADV * V[1:] + (1 - ADV) * V[:K]) + (1 - p) * V[:K]
        V = Vn
    return V[:K]


def sampler_q(name, phat):
    if name == "uniform":
        q = np.ones_like(phat)
    elif name.startswith("eq4_"):
        eps = float(name.split("_")[1])
        q = eps + np.sqrt(np.clip(phat * (1 - phat), 0, None))
    else:  # SGS T=2 (κ = 1, t = 0.5, ε = 1e-4)
        e = 1e-4
        w = (phat + e) ** 0.5 * (1 - phat + e) ** 0.5
        ell = np.log(np.maximum(w, e) + e)
        q = np.exp(ell / 2.0)
    return q / q.sum()


def rollout(starts, theta, reward, rng):
    """벡터화된 에피소드 B개. 반환: 완료 여부(B,), 반환 G(B,), 기울기 (sum_t (a−p)·adv 에 쓰일 자료)."""
    B = len(starts)
    k = starts.copy()
    done = np.zeros(B, bool)
    completed = np.zeros(B, bool)
    G = np.zeros(B)
    disc = 1.0
    ks, diffs, alive_hist = [], [], []
    terminate = reward in ("binary", "dense_term")
    for _ in range(T):
        alive = ~done
        p = sigmoid(theta[np.minimum(k, K - 1)])
        try_ = (rng.random(B) < p) & (k < K)
        ks.append(k.copy()); diffs.append(np.where(k < K, try_ - p, 0.0)); alive_hist.append(alive & (k < K))
        moved = try_ & (rng.random(B) < ADV)
        k = k + moved
        newly = (k == K) & ~completed
        completed |= newly
        r = np.zeros(B)
        if reward in ("dense_term", "omni"):
            r += np.where(alive, 0.02 * k / K, 0.0)
        if reward in ("binary", "dense_term"):
            r += np.where(alive & newly, 1.0, 0.0)
            done |= newly
        else:  # omni: 완료 상태에서 매 step 0.1, 종료 없음
            r += np.where(alive & (k == K), 0.1, 0.0)
        G += disc * r
        disc *= GAMMA
    grad_items = (np.array(ks), np.array(diffs), np.array(alive_hist))
    return completed, G, grad_items


def train(sampler, reward, seed):
    rng = np.random.default_rng(seed)
    theta = np.full(K, THETA0)
    hist = np.full((M, WIN), np.nan)
    ptr = np.zeros(M, int)
    curve = []
    for it in range(ITERS + 1):
        if it % EVAL_EVERY == 0:
            curve.append([it, round(float(exact_success(theta).mean()), 4)])
        if it == ITERS:
            break
        cnt = np.sum(~np.isnan(hist), 1)
        phat = np.where(cnt > 0, np.nansum(hist, 1) / np.maximum(cnt, 1), 0.5)
        q = sampler_q(sampler, phat)
        items = rng.choice(M, size=B_EP, p=q)
        completed, G, (ks, diffs, alive) = rollout(STARTS[items], theta, reward, rng)
        # 과제별 결과 기록 (완료 여부 = 성공)
        for j, i in enumerate(items):
            hist[i, ptr[i] % WIN] = float(completed[j]); ptr[i] += 1
        adv = G - G.mean()
        g = np.zeros(K)
        w = diffs * alive * adv[None, :]
        np.add.at(g, np.minimum(ks, K - 1).ravel(), w.ravel())
        theta = theta + LR * g / B_EP
    p_end = sigmoid(theta)
    return {"sampler": sampler, "reward": reward, "seed": seed, "curve": curve,
            "final_success_mean": curve[-1][1], "p_try_at_K-1": round(float(p_end[K - 1]), 4),
            "auc_mean_success": round(float(np.mean([c[1] for c in curve])), 4),
            "first_iter_0.9": next((c[0] for c in curve if c[1] >= 0.9), None)}


def job(args):
    s, r, seed = args
    return train(s, r, 1000 + seed)


def main():
    tasks = [(s, r, seed) for r in REWARDS for s in SAMPLERS for seed in SEEDS]
    with Pool(min(10, os.cpu_count() or 4)) as pool:
        runs = pool.map(job, tasks)
    summary = {}
    for r in REWARDS:
        for s in SAMPLERS:
            rs = [x for x in runs if x["reward"] == r and x["sampler"] == s]
            fin = [x["final_success_mean"] for x in rs]
            reach = [x["first_iter_0.9"] for x in rs if x["first_iter_0.9"] is not None]
            summary[f"{r} | {s}"] = {
                "final_success_mean": round(float(np.mean(fin)), 4), "final_success_sd": round(float(np.std(fin)), 4),
                "reached_0.9": f"{len(reach)}/{len(rs)}",
                "median_iter_to_0.9": float(np.median(reach)) if reach else None,
                "auc_mean_success": round(float(np.mean([x["auc_mean_success"] for x in rs])), 4),
                "p_try_at_K-1_mean": round(float(np.mean([x["p_try_at_K-1"] for x in rs])), 4),
            }
    # 학습 곡선 평균 (시드 평균)
    curves = {}
    for r in REWARDS:
        for s in SAMPLERS:
            rs = [x for x in runs if x["reward"] == r and x["sampler"] == s]
            arr = np.array([[c[1] for c in x["curve"]] for x in rs])
            curves[f"{r} | {s}"] = np.round(arr.mean(0), 4).tolist()
    out = {"settings": {"K": K, "T": T, "adv_prob": ADV, "gamma": GAMMA, "batch": B_EP, "iters": ITERS, "lr": LR,
                        "window": WIN, "M": M, "seeds": list(SEEDS), "eval_every": EVAL_EVERY},
           "summary": summary, "curves_mean_over_seeds": curves, "runs": runs}
    path = ROOT / "analysis" / "extra_bank_grid_results.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for k_, v in summary.items():
        print(k_, v)
    print("wrote", path)


if __name__ == "__main__":
    main()
