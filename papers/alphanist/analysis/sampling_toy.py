"""시작 상태 샘플링 커널 toy (CPU, numpy). 실행: python3 analysis/sampling_toy.py

AlphaNIST 식 4와 SGS 원 논문(arXiv:2610.12465 식 1–2, 조작 설정)을 그대로 구현해 비교한다.
- AlphaNIST Eq. 4:  q_i ∝ ε_q + sqrt(p̂_i (1 − p̂_i))          (ε_q 미기재 → 1e-4(독립 구현 기본값), 1e-2, 1e-1)
- SGS (κ = 1, t = 0.5, ε = 1e-4, T = 2): w = (p+ε)^{κt} (1−p+ε)^{κ(1−t)},  ℓ = log(max(w, ε) + ε),  q ∝ exp(ℓ / T)
  → 대략 (p(1−p))^{1/4}. AlphaNIST는 "SGS의 규칙을 유지한다"고 쓰지만 지수가 1/2 대 1/4로 다르다.
- 균등: q_i = 1/M

S1: 학습 단계별 bank 구성에서 프런티어(0 < p < 1) 항목에 가는 표본 비율과 표본당 이항 분산 E_q[p(1−p)]
S2: p̂를 최근 L개 결과의 평균으로 쓸 때(L = 50) 실제 p와 다른 '굶는 항목'이 얼마나 생기는가
S3: 논문 규모 숫자로 본 재방문 간격(다 익힌 항목이 몇 iteration마다 다시 뽑히는가)
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)


def q_alphanist(p, eps_q):
    return eps_q + np.sqrt(np.clip(p * (1 - p), 0, None))


def q_sgs(p, kappa=1.0, t=0.5, eps=1e-4, T=2.0):
    w = (p + eps) ** (kappa * t) * (1 - p + eps) ** (kappa * (1 - t))
    ell = np.log(np.maximum(w, eps) + eps)
    return np.exp(ell / max(T, 1.0))


def q_uniform(p):
    return np.ones_like(p)


KERNELS = {
    "uniform": q_uniform,
    "alphanist_eps1e-4": lambda p: q_alphanist(p, 1e-4),
    "alphanist_eps1e-2": lambda p: q_alphanist(p, 1e-2),
    "alphanist_eps1e-1": lambda p: q_alphanist(p, 1e-1),
    "sgs_T2": q_sgs,
    "sgs_T1": lambda p: q_sgs(p, T=1.0),
}


def bank(M, mastered, unsolved):
    """학습 단계를 흉내 낸 bank: 다 익힌(p=1)·못 푸는(p=0)·프런티어(0.05–0.95 균등) 항목."""
    n_m, n_u = int(M * mastered), int(M * unsolved)
    n_f = M - n_m - n_u
    return np.concatenate([np.ones(n_m), np.zeros(n_u), rng.uniform(0.05, 0.95, n_f)])


def stats(p_true, q):
    q = q / q.sum()
    frontier = (p_true > 0) & (p_true < 1)
    return {"frontier_share": round(float(q[frontier].sum()), 3),
            "signal_E_p1mp": round(float((q * p_true * (1 - p_true)).sum()), 4)}


def main():
    out: dict = {}
    M = 32_768  # SGS 조작 설정의 N (AlphaNIST의 M은 미기재)

    # S1 정확한 p를 안다고 할 때의 배분
    stages = {"early(80% unsolved)": (0.0, 0.8), "mid(30/30)": (0.3, 0.3), "late(95% mastered)": (0.95, 0.0)}
    s1 = {}
    for name, (m, u) in stages.items():
        p = bank(M, m, u)
        s1[name] = {k: stats(p, f(p)) for k, f in KERNELS.items()}
    out["S1_allocation_exact_p"] = s1

    # S2 p̂ = 최근 L개 결과 평균 (L = 50). 실제 p가 0.9–0.99인 항목은 p̂ = 1이 자주 나와 ε만 받는다
    L = 50
    p_true = rng.uniform(0.0, 1.0, M)
    p_hat = rng.binomial(L, p_true) / L
    s2 = {}
    for k, f in KERNELS.items():
        q = f(p_hat)
        q = q / q.sum()
        near1 = (p_true > 0.9) & (p_true < 1)
        starved = near1 & (p_hat == 1.0)
        s2[k] = {"share_on_true_frontier_0.05_0.95": round(float(q[(p_true > 0.05) & (p_true < 0.95)].sum()), 3),
                 "starved_entries_frac_of_p_0.9_1": round(float(starved.sum() / near1.sum()), 3),
                 "weight_ratio_starved_vs_p0.5": round(float(f(np.array([1.0]))[0] / f(np.array([0.5]))[0]), 4)}
    out["S2_estimation_L50"] = s2
    out["S2_note"] = ("실제 p = 0.95인 항목은 L = 50에서 p̂ = 1이 될 확률이 0.95^50 = %.3f다. 이때 ε만 받으므로 "
                      "ε_q가 작으면 오래 다시 뽑히지 않는다. 다 익혔다고 본 상태를 잊어도 늦게 알아챈다." % (0.95 ** 50))

    # S3 논문 규모의 재방문 간격 (late 단계, 다 익힌 항목 하나)
    envs, steps_iter = 64_000, 32
    s3 = {}
    for ep_len in (100, 375):
        episodes_per_iter = envs * steps_iter / ep_len
        row = {}
        for k in ("alphanist_eps1e-4", "alphanist_eps1e-2", "alphanist_eps1e-1", "sgs_T2", "uniform"):
            p = bank(M, 0.95, 0.0)
            q = KERNELS[k](p)
            q = q / q.sum()
            q_mastered = q[0]
            visits_per_iter = episodes_per_iter * q_mastered
            row[k] = {"iters_between_visits": round(1 / visits_per_iter, 1),
                      "visits_in_25K_iters": round(25_000 * visits_per_iter, 1)}
        s3[f"mean_episode_{ep_len}_steps"] = row
    out["S3_revisit_interval_late"] = s3
    out["S3_assumptions"] = "M = 32,768(SGS 조작 설정), 64K env, iteration당 32 step, 성공 시 조기 종료라 평균 episode 100–375 step으로 가정"

    path = ROOT / "analysis" / "sampling_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False)[:4000])
    print("wrote", path)


if __name__ == "__main__":
    main()
