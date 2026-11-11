"""RPD 토이: 1차원 연속 행동 밴딧에서 정책 경사 + 교사 MSE 당김 항.

목적
- 교사 행동을 MSE로 당기면 (1) 학습이 빨라지는지, (2) 고정점에 편향이 남는지 확인한다.
- 이론 고정점 μ* = (a* + λ·t̄)/(1 + λ) (dense, 2차 보상, 상수 λ)을 수치로 대조한다.
- 교사 종류: good(t=a*), mediocre(a*+0.4, 같은 방향), wrong(-2, 반대 방향),
  noisy_unbiased(평균 a*, 잡음 1), irrelevant(평균 0, 잡음 1: 쓸모없는 교사 대조군).
- 가중치 스케줄: const, linear(T/2에서 0), cosine(0.5cos(πt/T)+0.5, 코드의 max_ppd_anneal_steps 형태).

모델
- 정책 a ~ N(μ, σ²), σ = 0.3 고정 (평균만 학습). 배치 B = 32, 반복 T = 800, 시드 30.
- 보상 dense: r = -(a - a*)², sparse: r = 1{|a - a*| < 0.1}.
- 이득은 평균만 빼는 기준선 사용 (정규화 안 함), 정책 경사 g = mean[(r - r̄)(a - μ)/σ²].
- 교사 항: 표본마다 t_i를 뽑고 λ_t·mean((μ - t_i)²)을 더한다. 갱신 μ += η(g - 2λ_t(μ - t̄)), η = 0.1.
- 교사 질의 수는 반복당 B (코드는 미니배치마다 질의하므로 실제는 더 많음, 04 D-항목 참고).

실행: OMP_NUM_THREADS=1 python3 analysis/toy_teacher_pull.py  (약 수 초, 결과는 toy_teacher_pull_results.json)
"""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A_STAR = 1.0
SIGMA = 0.3
B = 32
T = 800
S = 30
ETA = 0.1
EPS = 0.1
MU0 = -1.0
TEACHERS = {
    "good": ("const", 1.0, 0.0),
    "mediocre": ("const", 1.4, 0.0),
    "wrong": ("const", -2.0, 0.0),
    "noisy_unbiased": ("noise", 1.0, 1.0),
    "irrelevant": ("noise", 0.0, 1.0),
}
LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0]
SCHEDULES = ["const", "linear_half", "cosine"]


def lam_t(lam, sched, t):
    if sched == "const":
        return lam
    if sched == "linear_half":
        return lam * max(0.0, 1.0 - t / (T / 2))
    if sched == "cosine":
        return lam * (0.5 * math.cos(math.pi * t / T) + 0.5)
    raise ValueError(sched)


def teacher_sample(kind, rng):
    mode, mean, sd = TEACHERS[kind]
    if mode == "const":
        return np.full((S, B), mean)
    return mean + sd * rng.standard_normal((S, B))


def phi(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def run(kind, lam, sched, reward, seed=0):
    rng = np.random.default_rng(seed)
    mu = np.full(S, MU0)
    first_hit = np.full(S, np.nan)
    tail = []
    for t in range(T):
        a = mu[:, None] + SIGMA * rng.standard_normal((S, B))
        if reward == "dense":
            r = -(a - A_STAR) ** 2
        else:
            r = (np.abs(a - A_STAR) < EPS).astype(float)
        adv = r - r.mean(axis=1, keepdims=True)
        g_rl = (adv * (a - mu[:, None]) / SIGMA**2).mean(axis=1)
        tt = teacher_sample(kind, rng)
        pull = 2.0 * lam_t(lam, sched, t) * (mu[:, None] - tt).mean(axis=1)
        mu = mu + ETA * (g_rl - pull)
        hit = np.isnan(first_hit) & (np.abs(mu - A_STAR) < 0.1)
        first_hit[hit] = t
        if t >= T - 100:
            tail.append(mu.copy())
    mean_tail = np.mean(tail, axis=0)
    final_mu = mean_tail
    # 성공 확률 (sparse 해석): P(|a - a*| < EPS) under N(μ, σ²)
    p_succ = phi((A_STAR + EPS - final_mu) / SIGMA) - phi((A_STAR - EPS - final_mu) / SIGMA)
    hit_t = np.where(np.isnan(first_hit), T, first_hit)
    return {
        "reach_iter_median": float(np.median(hit_t)),
        "reach_fraction": float(np.mean(~np.isnan(first_hit))),
        "final_bias_mean": float(np.mean(final_mu - A_STAR)),
        "final_abs_err_mean": float(np.mean(np.abs(final_mu - A_STAR))),
        "final_abs_err_sd": float(np.std(np.abs(final_mu - A_STAR), ddof=1)),
        "success_prob_mean": float(np.mean(p_succ)),
    }


def theory_bias(kind, lam):
    """const 스케줄, dense에서 이론 고정점 편향 λ/(1+λ)·(t̄ - a*)."""
    mode, mean, _ = TEACHERS[kind]
    return lam / (1.0 + lam) * (mean - A_STAR)


def main():
    out = {"config": dict(A_STAR=A_STAR, SIGMA=SIGMA, B=B, T=T, S=S, ETA=ETA, EPS=EPS, MU0=MU0),
           "dense": {}, "sparse": {}, "theory_check": []}
    for reward in ["dense", "sparse"]:
        for kind in TEACHERS:
            for sched in SCHEDULES:
                for lam in LAMBDAS:
                    if lam == 0.0 and sched != "const":
                        continue
                    key = f"{kind}|{sched}|lam={lam}"
                    out[reward][key] = run(kind, lam, sched, reward)
    for kind in TEACHERS:
        for lam in LAMBDAS:
            if lam == 0.0:
                continue
            sim = out["dense"][f"{kind}|const|lam={lam}"]["final_bias_mean"]
            out["theory_check"].append({
                "teacher": kind, "lam": lam,
                "theory_bias": theory_bias(kind, lam),
                "sim_bias": sim,
                "diff": sim - theory_bias(kind, lam),
            })
    (HERE / "toy_teacher_pull_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("dense/const/λ=1 bias by teacher:")
    for kind in TEACHERS:
        r = out["dense"][f"{kind}|const|lam=1.0"]
        print(f"  {kind:15s} sim_bias={r['final_bias_mean']:+.3f} theory={theory_bias(kind,1.0):+.3f} reach={r['reach_iter_median']:.0f}")
    print("sparse/const/λ=1 success prob by teacher:")
    for kind in TEACHERS:
        r = out["sparse"][f"{kind}|const|lam=1.0"]
        print(f"  {kind:15s} p_succ={r['success_prob_mean']:.3f} reach_frac={r['reach_fraction']:.2f}")


if __name__ == "__main__":
    main()
