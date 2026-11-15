"""토이 B: 논문의 평가 설계(체크포인트 1개 x 세션 2개)가 얼마나 믿을 만한지 시뮬레이션한다 (numpy, scipy, 수십 초).

논문 App. D.1의 "±"는 같은 체크포인트를 두 세션 평가한 값의 모집단 표준편차다. 학습 시드 간 변동은 재지 않는다.
이 토이는 (1) 학습 시드 변동이 있을 때 "±"가 얼마나 작게 나오는지, (2) 같은 설계에서 Fisher 검정(시도 단위 이항 가정)이
거짓 양성을 얼마나 내는지, (3) 실제 차이 Δ를 얼마나 자주 잡는지를 본다.

모형:
- 방법 k의 체크포인트 참 성공률 p_k(seed) = clip(μ_k + N(0, s²), 0, 1). 시드 변동 s ∈ {0, 0.03, 0.05, 0.10}.
- 한 체크포인트를 세션 2개 평가. 세션당 n_s 시도(이항). 세션 두 개를 합쳐 2 n_s 시도로 Fisher 검정(양측 0.05).
- (1) 논문식 ±: 두 세션 성공률의 모집단 SD. 시드 간 참 SD(s)와 비교.
- (2) 거짓 양성률: μ_A = μ_B = 0.5, 차이 0.
- (3) 검정력: μ_A = 0.65, μ_B = 0.45(Δ = 0.20, 논문 OpenVLA·π0.5 평균 차이 규모).

가정과 한계: 시도 독립, 초기 조건 독립(논문은 같은 초기 조건을 공유하므로 실제로는 짝지은 설계). 시드 간 변동 s는 가정값이다.

실행: python3 analysis/toy_eval_design.py  → analysis/toy_eval_results.json
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "toy_eval_results.json"
SEEDS_SD = [0.0, 0.03, 0.05, 0.10]
N_SESS = [10, 20, 32]
REPS = 3000
ALPHA = 0.05
RNG = np.random.default_rng(1)


def sample_checkpoint(mu, s, size):
    return np.clip(mu + RNG.standard_normal(size) * s, 0.0, 1.0)


def fisher_p(k1, n1, k2, n2):
    _, p = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="two-sided")
    return p


def one_arm(mu, s, n_s):
    """체크포인트 1개를 세션 2개 평가한다. 반환: (세션 성공 수 두 개, 세션 SD(모집단), 실제 참 p)."""
    p = float(sample_checkpoint(mu, s, 1)[0])
    k = RNG.binomial(n_s, p, size=2)
    rates = k / n_s
    return k, float(np.std(rates)), p


def main():
    out = {"setup": {"reps": REPS, "sessions_per_checkpoint": 2, "alpha": ALPHA,
                     "seed_sd_grid": SEEDS_SD, "session_sizes": N_SESS,
                     "note": "가정 시드 SD. 시도 독립 가정. 실제 논문은 초기 조건을 공유하는 짝지은 설계"},
           "paper_pm_vs_seed": [], "type1": [], "power": []}

    # (1) 논문식 ±와 시드 간 참 SD
    for n_s in N_SESS:
        for s in SEEDS_SD:
            pms, true_ps = [], []
            for _ in range(REPS):
                _, pm, p = one_arm(0.6, s, n_s)
                pms.append(pm)
                true_ps.append(p)
            out["paper_pm_vs_seed"].append({
                "n_per_session": n_s, "seed_sd": s,
                "mean_paper_pm": round(float(np.mean(pms)) * 100, 2),
                "true_seed_sd": round(float(np.std(true_ps)) * 100, 2),
            })

    # (2) 거짓 양성률 (Δ = 0), (3) 검정력 (Δ = 0.20)
    for n_s in N_SESS:
        for s in SEEDS_SD:
            fp = pw = 0
            for _ in range(REPS):
                k1, _, _ = one_arm(0.5, s, n_s)
                k2, _, _ = one_arm(0.5, s, n_s)
                if fisher_p(int(k1.sum()), 2 * n_s, int(k2.sum()), 2 * n_s) < ALPHA:
                    fp += 1
                k1, _, _ = one_arm(0.65, s, n_s)
                k2, _, _ = one_arm(0.45, s, n_s)
                if fisher_p(int(k1.sum()), 2 * n_s, int(k2.sum()), 2 * n_s) < ALPHA:
                    pw += 1
            out["type1"].append({"n_per_session": n_s, "seed_sd": s, "false_positive_rate": round(fp / REPS, 3)})
            out["power"].append({"n_per_session": n_s, "seed_sd": s, "power_delta_020": round(pw / REPS, 3)})
            print(f"n_s={n_s:2d} seed_sd={s:.2f}  FPR(delta=0)={fp / REPS:.3f}  power(delta=0.20)={pw / REPS:.3f}")

    for row in out["paper_pm_vs_seed"]:
        print(f"n_s={row['n_per_session']:2d} seed_sd={row['seed_sd']:.2f}  paper±(mean)={row['mean_paper_pm']:6.2f}  true seed SD={row['true_seed_sd']:6.2f}")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
