"""토이 B: (1) 리셋 실패 확률과 처리량·사람 개입, (2) 라벨 가능 조건으로 걸러진 데이터베이스의 선택 편향 (numpy, 1스레드).

실행: python3 analysis/collection_toy.py -> analysis/collection_toy_results.json (수 초)
가정(설계값, 논문에 없음): 루프 시간 평균 48.2 s(논문 4.1절), 표준편차 8 s. 루프 밖 시간은 논문 값에서 역산한 26 s/회.
리셋은 자세 전환마다 일어나며 r회 시도마다 한 번(r=1,3,10). 리셋 실패 시 사람이 개입하며 벽시계 600 s, 사람 시간 120 s를 쓴다고 가정.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("collection_toy_results.json")
N_TRIALS = 500
LOOP_MU, LOOP_SD = 48.2, 8.0
OVERHEAD_S = 26.0          # 논문 10.3 h - 루프 6.69 h = 3.6 h, 500회당 약 26 s
INT_WALL_S, INT_HUMAN_S = 600.0, 120.0
REPS = 400


def sim_throughput(q, every, rng):
    loop = np.clip(rng.normal(LOOP_MU, LOOP_SD, N_TRIALS), 20, None)
    wall = loop.sum() + OVERHEAD_S * N_TRIALS
    n_resets = N_TRIALS // every
    fails = rng.random(n_resets) < q
    k = int(fails.sum())
    wall += k * INT_WALL_S
    human = k * INT_HUMAN_S + 0.0
    return wall / 3600, k, human / 3600


def throughput_grid():
    out = {}
    for every in (1, 3, 10):
        for q in (0.0, 0.02, 0.05, 0.10, 0.20):
            rng = np.random.default_rng(7)
            rs = np.array([sim_throughput(q, every, rng) for _ in range(REPS)])
            out[f"reset_every={every}, q={q}"] = {
                "wall_h_mean": round(float(rs[:, 0].mean()), 2),
                "wall_h_sd": round(float(rs[:, 0].std(ddof=1)), 2),
                "interventions_mean": round(float(rs[:, 1].mean()), 1),
                "human_h_mean": round(float(rs[:, 2].mean()), 2),
                "trials_per_wall_h": round(N_TRIALS / float(rs[:, 0].mean()), 1),
            }
    # 역산: 관측 개입 31회/500회를 설명하는 리셋 실패율
    implied = {str(every): round(31 / (N_TRIALS // every), 3) for every in (1, 3, 10)}
    return out, implied


def selection_bias(gamma, rng, C=20000):
    """후보 C개. 잠재 난이도 z. 참 성공 확률은 z가 클수록 낮다.
    라벨 가능 여부(물체 검출·추적이 되어 라벨이 붙는 확률)도 z가 클수록 낮다(gamma가 세기).
    라벨이 없는 시도는 버려진다(run_auto.py의 label_unjudgeable 처리와 같은 방식)."""
    z = rng.normal(size=C)
    p = 1 / (1 + np.exp(-(np.log(0.45 / 0.55) - 1.0 * z + rng.normal(0, 0.3, C))))
    labelable = rng.random(C) < 1 / (1 + np.exp(-(0.8 - gamma * z)))
    y = rng.random(C) < p
    return {
        "population_success_pct": 100 * p.mean(),
        "labelable_frac_pct": 100 * labelable.mean(),
        "db_success_pct": 100 * y[labelable].mean(),
        "bias_pp_db_minus_population": 100 * (y[labelable].mean() - p.mean()),
        "corr_labelable_vs_p": float(np.corrcoef(labelable.astype(float), p)[0, 1]),
    }


def selection_grid():
    out = {}
    for g in (0.0, 0.5, 1.0, 2.0):
        rs = [selection_bias(g, np.random.default_rng(100 + r)) for r in range(40)]
        out[f"gamma={g}"] = {k: round(float(np.mean([r[k] for r in rs])), 3) for k in rs[0]}
        out[f"gamma={g}"]["bias_pp_sd"] = round(float(np.std([r["bias_pp_db_minus_population"] for r in rs], ddof=1)), 2)
    return out


if __name__ == "__main__":
    thr, implied = throughput_grid()
    sel = selection_grid()
    res = {"meta": {"N_TRIALS": N_TRIALS, "loop_mu_s": LOOP_MU, "loop_sd_s": LOOP_SD, "overhead_s": OVERHEAD_S,
                    "intervention_wall_s": INT_WALL_S, "intervention_human_s": INT_HUMAN_S, "reps": REPS,
                    "note": "설계값 토이. 개입 비용과 리셋 빈도는 논문에 없다."},
           "throughput": thr, "implied_reset_fail_rate_from_31_interventions": implied,
           "selection_bias": sel}
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    for k in list(thr)[:6]:
        print(k, thr[k])
    print("implied", implied)
    for k, v in sel.items():
        print(k, v)
    print("written", OUT)
