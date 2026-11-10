"""두 개의 numpy toy (CPU, 단일 스레드, 수 초 안에 끝남).

Toy A. 피드백 기반 제안기(proposer) vs 고정 가중치 샘플링 최적화.
  - 목적함수: 여러 접촉 전략(기저)을 가진 다봉 함수. 초기값은 잘못된 기저(국소 최적)에 있다.
  - 고정 가중치: 샘플 평균을 고정 온도·고정 잡음으로 갱신하는 MPPI형 탐색 (SPIDER의 샘플링 최적화와 같은 종류).
  - 제안기: 정체가 몇 라운드 이어지면 다른 앵커(접촉 전략)로 옮기고 다시 지역 탐색. LLM 대신 규칙으로 흉내 낸 것이다.
  - 예산은 시뮬레이터 호출 수로 맞춘다. 관측값에는 잡음이 들어간다.
  - 단봉(기저 하나) 함수도 같이 돌려서 제안기의 간접비가 손해가 되는 조건을 본다.

Toy B. 같은 지표로 선택하고 같은 지표로 보고할 때의 낙관 편향(evaluator 과적합).
  - 후보 K개를 잡음 있는 시뮬레이션으로 평가하고 관측 최솟값을 보고한다.
  - 선택된 후보를 새 잡음으로 다시 평가해 held-out 값을 얻는다. 보고값 - held-out이 편향이다.

결과: analysis/refinement_toy_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "refinement_toy_results.json"
D = 6            # 결정 변수 차원 (motion 파라미터 축약)
KAPPA = 1.0


def make_landscape(multimodal=True, rng=None):
    """기저 중심 c_m과 기저 최솟값 a_m. f(x) = min_m (a_m + kappa*||x-c_m||^2)."""
    if multimodal:
        centers = np.stack([np.full(D, 0.5), np.full(D, -1.5), rng.normal(0, 1.0, D)])
        floors = np.array([18.0, 2.0, 6.0])        # 기저 0이 초기값 근처의 잘못된 접촉 전략 (국소 최적, 20 cm대)
    else:
        centers = np.stack([np.full(D, -1.5)])
        floors = np.array([2.0])
    return centers, floors


def f_true(x, centers, floors):
    return float(np.min(floors + KAPPA * np.sum((x[None, :] - centers) ** 2, axis=1)))


def run_fixed_weight(budget, x0, centers, floors, sigma_obs, rng, n_per_iter=32, step=0.35, lam=1.0):
    """고정 온도·고정 샘플 잡음 MPPI형 탐색. 최선 관측값을 보고."""
    mu = x0.copy(); best_obs = np.inf; best_x = mu.copy(); used = 0
    while used + n_per_iter <= budget:
        X = mu[None, :] + step * rng.standard_normal((n_per_iter, D))
        y = np.array([f_true(x, centers, floors) for x in X]) + sigma_obs * rng.standard_normal(n_per_iter)
        used += n_per_iter
        i = int(np.argmin(y))
        if y[i] < best_obs:
            best_obs, best_x = y[i], X[i].copy()
        w = np.exp(-(y - y.min()) / lam); w /= w.sum()
        mu = w @ X
    return best_x, best_obs


def run_proposer(budget, x0, centers, floors, sigma_obs, rng, n_per_iter=32, step=0.35, patience=4):
    """정체가 patience 라운드 이어지면 새 앵커(다른 접촉 전략)를 제안하고 같은 지역 탐색을 한다.
    새 앵커는 이전 시도와 떨어진 곳에서 고른다(실패 보고를 반영하는 흉내)."""
    anchors = [x0.copy()]; mu = x0.copy(); best_obs = np.inf; best_x = mu.copy(); used = 0
    stall = 0; last_best = np.inf; tried = []
    while used + n_per_iter <= budget:
        X = mu[None, :] + step * rng.standard_normal((n_per_iter, D))
        y = np.array([f_true(x, centers, floors) for x in X]) + sigma_obs * rng.standard_normal(n_per_iter)
        used += n_per_iter
        i = int(np.argmin(y))
        if y[i] < best_obs:
            best_obs, best_x = y[i], X[i].copy()
        if best_obs < last_best - 1e-9:
            last_best = best_obs; stall = 0
        else:
            stall += 1
        w = np.exp(-(y - y.min())); w /= w.sum()
        mu = w @ X
        if stall >= patience:
            # 새 앵커: 이미 시도한 앵커에서 멀리 떨어진 후보 중 임의로 고른다
            cand = rng.uniform(-2.5, 2.5, size=(64, D))
            dist = np.min(np.linalg.norm(cand[:, None, :] - np.array(tried + [mu])[None], axis=2), axis=1) if tried else np.ones(64)
            mu = cand[int(np.argmax(dist))]
            tried.append(mu.copy()); stall = 0; last_best = np.inf
            anchors.append(mu.copy())
    return best_x, best_obs


def toy_a(seeds=100):
    budgets = [256, 512, 1024, 2048]
    res = {}
    for scen, multimodal in [("multimodal_wrong_init", True), ("unimodal", False)]:
        res[scen] = {}
        for sigma in [0.0, 2.0]:
            rows = {"fixed": {b: [] for b in budgets}, "proposer": {b: [] for b in budgets}}
            for s in range(seeds):
                rng = np.random.default_rng(1000 + s)
                centers, floors = make_landscape(multimodal, rng)
                x0 = centers[0] + 0.15 * rng.standard_normal(D) if multimodal else centers[0] + 1.2 * rng.standard_normal(D)
                for b in budgets:
                    xf, _ = run_fixed_weight(b, x0, centers, floors, sigma, np.random.default_rng(5000 + s))
                    xp, _ = run_proposer(b, x0, centers, floors, sigma, np.random.default_rng(9000 + s))
                    rows["fixed"][b].append(f_true(xf, centers, floors))
                    rows["proposer"][b].append(f_true(xp, centers, floors))
            res[scen][f"sigma_{sigma}"] = {
                arm: {str(b): {"median_true_cm": round(float(np.median(v)), 2),
                               "q25": round(float(np.percentile(v, 25)), 2),
                               "q75": round(float(np.percentile(v, 75)), 2)}
                      for b, v in rows[arm].items()}
                for arm in rows}
            # 쌍 비교: 같은 시드에서 제안기가 더 낮은 진짜 값을 얻은 비율
            res[scen][f"sigma_{sigma}"]["proposer_better_frac_at_2048"] = round(float(np.mean(
                np.array(rows["proposer"][2048]) < np.array(rows["fixed"][2048]))), 2)
    return res


def toy_b(seeds=4000):
    """K개 후보를 잡음 시뮬레이션으로 평가하고 최솟값을 보고. held-out은 새 잡음으로 재평가."""
    Ks = [1, 2, 5, 10, 20, 40, 80]
    out = {}
    for sigma in [1.0, 2.0, 4.0]:
        rows = {}
        for K in Ks:
            rng = np.random.default_rng(42 + K)
            true_q = 20.0 + 5.0 * rng.standard_normal((seeds, K))   # 후보들의 진짜 오차 (cm)
            true_q = np.clip(true_q, 3.0, None)
            obs = true_q + sigma * rng.standard_normal((seeds, K))
            j = np.argmin(obs, axis=1)
            reported = obs[np.arange(seeds), j]
            heldout = true_q[np.arange(seeds), j] + sigma * rng.standard_normal(seeds)
            rows[K] = {"reported_mean_cm": round(float(reported.mean()), 2),
                       "heldout_mean_cm": round(float(heldout.mean()), 2),
                       "optimism_bias_cm": round(float((reported - heldout).mean()), 2),
                       "true_of_selected_cm": round(float(true_q[np.arange(seeds), j].mean()), 2)}
        out[f"sigma_{sigma}"] = rows
    return out


def main():
    a = toy_a()
    b = toy_b()
    OUT.write_text(json.dumps({"toy_a_feedback_proposer_vs_fixed": a,
                               "toy_b_selection_optimism": b,
                               "limits": ["제안기는 LLM이 아니라 규칙 기반 흉내다",
                                          "목적함수는 2차 기저의 최솟값이며 실제 HOI 물리와 다르다",
                                          "Toy B의 진짜 품질 분포(평균 20, 표준편차 5 cm)는 가정이다"]},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
