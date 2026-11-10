"""ContactBench 1.0 벤치마크 검정력 토이 세 개 (CPU, numpy 단일 스레드 + scipy의 Fisher 정확 검정).

토이 P1: 방법당 시행 수 n과 성공률 차이 Δ에 따른 검정력, 순위 신뢰도.
토이 P2: 초기조건을 공유할 때와 매번 다시 뽑을 때의 겉보기 유의성(거짓 양성률), 학습 실행 분산의 영향.
토이 P3: 집계 점수의 과제 가중치 민감도(Dirichlet 가중치와 이항 잡음).

설계값(σ, δ, 가중치 분포)은 모두 가정이다. 방향만 주장으로 쓴다.
실행: python3 analysis/power_toy.py → analysis/power_toy_results.json (약 30초)
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "power_toy_results.json"
ALPHA = 0.05
ZA = norm.ppf(1 - ALPHA / 2)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def n_per_arm(p1, p2, power=0.8):
    """정규 근사 필요 표본 수(방법당). 두 비율의 양측 검정."""
    pbar = (p1 + p2) / 2
    za, zb = ZA, norm.ppf(power)
    num = za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return float(np.ceil((num / (p2 - p1)) ** 2))


def fisher_two_sided(ka, kb, n):
    return float(fisher_exact([[ka, n - ka], [kb, n - kb]], alternative="two-sided")[1])


def toy_p1(rng):
    out = {}
    # (a) 해석식: 80% 검정력에 필요한 방법당 n
    grid = {}
    for p1 in (0.2, 0.5, 0.8):
        for d in (0.1, 0.2, 0.3):
            p2 = p1 + d if p1 + d <= 1 else p1 - d
            grid[f"p1={p1},|d|={d}"] = n_per_arm(p1, p2)
    out["n_for_80pct_power_normal_approx"] = grid

    # (b) 시뮬레이션: p1 = 0.5, Fisher 정확 검정, 양측 0.05
    reps = 2000
    sim = {}
    for d in (0.2, 0.3):
        row = {}
        for n in (10, 20, 30, 50, 80, 120):
            hits = 0
            pa = rng.binomial(n, 0.5, size=reps)
            pb = rng.binomial(n, 0.5 + d, size=reps)
            for a, b in zip(pa, pb):
                if fisher_two_sided(int(a), int(b), n) < ALPHA:
                    hits += 1
            row[str(n)] = round(hits / reps, 3)
        sim[f"d={d}"] = row
    out["simulated_power_p1_0.5_fisher_exact"] = sim

    # (c) 순위 신뢰도: 참 성공률이 p_a > p_b일 때 관측 순서가 맞을 확률(동률은 0.5점)
    rank = {}
    for pa, pb in ((0.5, 0.4), (0.5, 0.3), (0.7, 0.5), (0.3, 0.1), (0.9, 0.8)):
        row = {}
        for n in (10, 30, 100, 300):
            ka = rng.binomial(n, pa, size=20000)
            kb = rng.binomial(n, pb, size=20000)
            score = (ka > kb).mean() + 0.5 * (ka == kb).mean()
            row[str(n)] = round(float(score), 3)
        rank[f"pa={pa},pb={pb}"] = row
    out["rank_correct_probability"] = rank
    return out


def toy_p2(rng):
    """초기조건(IC) 공유 여부와 학습 실행 분산이 겉보기 유의성에 주는 영향.

    로짓 모형: P(성공) = σ(μ_k + s_k + b_i). b_i는 초기조건 난이도(σ_b = 1), s_k는 정책 k의 학습 실행 효과.
    R(재추출): 시행마다 새 b를 뽑는다. S(공유): 두 정책이 같은 30개 b_i를 쓴다.
    학습 실행이 정책당 1회이면 s_k는 한 번 뽑힌 값이고, 시행을 늘려도 평균되지 않는다.
    """
    reps = 2000
    sigma_b = 1.0
    res = {}
    for design in ("R", "S"):
        for sigma_s in (0.0, 0.25, 0.5):
            for n in (30, 150):
                for delta in (0.0, 0.85):
                    key = f"design={design},sigma_s={sigma_s},n={n},delta={delta}"
                    fisher_hits, mcn_hits = 0, 0
                    for _ in range(reps):
                        sa, sb = rng.normal(0, sigma_s, 2)
                        if design == "S":
                            b = rng.normal(0, sigma_b, n)
                            ba, bb = b, b
                        else:
                            ba = rng.normal(0, sigma_b, n)
                            bb = rng.normal(0, sigma_b, n)
                        ya = rng.uniform(size=n) < sigmoid(sa + ba)
                        yb = rng.uniform(size=n) < sigmoid(delta + sb + bb)
                        ka, kb = int(ya.sum()), int(yb.sum())
                        if fisher_two_sided(ka, kb, n) < ALPHA:
                            fisher_hits += 1
                        if design == "S":
                            # 쌍 검정(McNemar, 연속성 보정): 에피소드별 결과가 있을 때만 가능하다
                            b01 = int(np.sum(ya & ~yb))
                            b10 = int(np.sum(~ya & yb))
                            if b01 + b10 > 0:
                                stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
                                if stat > 3.841458820694124:
                                    mcn_hits += 1
                    res[key] = {"fisher_rate": round(fisher_hits / reps, 3)}
                    if design == "S":
                        res[key]["mcnemar_rate"] = round(mcn_hits / reps, 3)
    return res


def dirichlet_weights(rng, n_draws, k):
    return rng.dirichlet(np.ones(k), size=n_draws)


def toy_p3(rng):
    """집계 점수의 과제 가중치 민감도. Table III의 k/30을 쓴다."""
    tasks6 = ["thread", "tighten", "plug", "door", "utensil", "card"]
    tasks4 = ["thread", "tighten", "plug", "door"]
    k = {
        "DP": dict(zip(tasks6, [8, 0, 2, 30, 0, 14])),
        "PI05": dict(zip(tasks6, [14, 17, 6, 30, 16, 2])),
        "RDP": dict(zip(tasks4, [0, 0, 4, 29])),
        "FTP1": dict(zip(tasks4, [14, 3, 4, 29])),
    }
    skills = {"thread": 2, "tighten": 2, "plug": 1, "door": 1, "utensil": 2, "card": 1}

    def arr(pol, tasks):
        return np.array([k[pol][t] for t in tasks], float) / 30.0

    out = {"unweighted_mean": {}, "skill_count_weighted": {}}
    for pol, tasks in (("DP", tasks6), ("PI05", tasks6), ("DP", tasks4), ("RDP", tasks4),
                       ("PI05", tasks4), ("FTP1", tasks4)):
        key = f"{pol}_on_{len(tasks)}tasks"
        out["unweighted_mean"][key] = round(float(arr(pol, tasks).mean()), 4)
        w = np.array([skills[t] for t in tasks], float)
        out["skill_count_weighted"][key] = round(float((arr(pol, tasks) * w).sum() / w.sum()), 4)

    comparisons = {
        "PI05>DP (6 tasks)": ("PI05", "DP", tasks6),
        "PI05>DP (4 tasks)": ("PI05", "DP", tasks4),
        "DP>RDP (4 tasks)": ("DP", "RDP", tasks4),
        "PI05>FTP1 (4 tasks)": ("PI05", "FTP1", tasks4),
    }
    draws = 20000
    dirichlet = {}
    binomial_plus_dirichlet = {}
    for name, (a, b, tasks) in comparisons.items():
        pa, pb = arr(a, tasks), arr(b, tasks)
        w = dirichlet_weights(rng, draws, len(tasks))
        dirichlet[name] = round(float(((w @ pa) > (w @ pb)).mean()), 4)
        # 이항 잡음까지 넣기: 각 칸을 Binomial(30, p̂)로 다시 뽑는다
        ka = rng.binomial(30, np.broadcast_to(pa, (draws, len(tasks))))
        kb = rng.binomial(30, np.broadcast_to(pb, (draws, len(tasks))))
        ra, rb = ka / 30.0, kb / 30.0
        binomial_plus_dirichlet[name] = round(float(((w * ra).sum(1) > (w * rb).sum(1)).mean()), 4)
    out["prob_ordering_holds_dirichlet_weights"] = dirichlet
    out["prob_ordering_holds_dirichlet_plus_binomial_noise"] = binomial_plus_dirichlet

    # 상한·하한 효과: 과제 하나를 뺄 때 차이(pp)
    drop = {}
    for t in tasks6:
        keep = [x for x in tasks6 if x != t]
        drop[t] = round(100 * float((arr("PI05", keep) - arr("DP", keep)).mean()), 2)
    out["mean_diff_pp_pi05_minus_dp_when_dropping_task"] = drop
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(20261110)
    res = {"meta": {"seed": 20261110, "alpha": ALPHA, "note": "가정된 설계값에서의 시뮬레이션. 논문 원자료가 아님."}}
    res["P1_power"] = toy_p1(rng)
    res["P2_initial_conditions"] = toy_p2(rng)
    res["P3_weights"] = toy_p3(rng)
    res["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("wrote", OUT, "in", res["runtime_s"], "s")
    print("P1 n80:", res["P1_power"]["n_for_80pct_power_normal_approx"])
    print("P1 sim:", res["P1_power"]["simulated_power_p1_0.5_fisher_exact"])
    print("P1 rank:", res["P1_power"]["rank_correct_probability"])
    for k_, v in res["P2_initial_conditions"].items():
        print("P2", k_, v)
    print("P3", json.dumps(res["P3_weights"], ensure_ascii=False)[:900])


if __name__ == "__main__":
    main()
