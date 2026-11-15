"""토이 A: 정확한 속도장에서 한 스텝 Euler와 다스텝 Euler의 차이가 목표 분포의 다중모드에 따라 어떻게 달라지는가.

설정 (직접 작성, numpy만 사용):
  x0 ~ N(0, 1), x1 ~ 두 성분 혼합 0.5 N(-d/2, s^2) + 0.5 N(+d/2, s^2)
  학습된 정확한 속도장: rectified flow의 목표 v(x,t) = E[x1 - x0 | x_t = x] 를 닫힌 식으로 계산
  (성분별 가우시안 결합 분포에서 사후 확률 r_k와 조건부 기댓값을 씀).

측정:
  - 한 스텝 출력 (N=1), 다스텝 출력 (N=10, 1000)의 분포를 목표와 1-Wasserstein으로 비교
  - 기하: 경로 길이 대 현 길이 (논문 A.2의 Excess), 끝점 오차 (논문 A.2의 Error)
  - 모드 간격 d/s 를 바꿔 가며 sweep

이 토이는 논문 주장의 기계적 원인을 시험하는 장치다. 학습 오차, 용량, 데이터 분포는 다루지 않는다.
결과: analysis/toy_one_step_gmm_results.json   실행: python3 analysis/toy_one_step_gmm.py (수 초)
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_one_step_gmm_results.json"
RNG = np.random.default_rng(0)
S = 0.2  # 성분 표준편차
N_SAMPLES = 20000
SEPS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]  # d/s 가 아니라 d (모드 중심 간격), s=0.2 고정 -> d/s = 0, 2.5, 5, 10, 15, 20


def velocity(x, t, d, s=S):
    """목표 혼합 N(±d/2, s^2), 0.5/0.5. x: (n,), t: 스칼라. 반환 E[x1 - x0 | x_t = x]."""
    means = np.array([-d / 2, d / 2])
    w = np.array([0.5, 0.5])
    V = (1 - t) ** 2 + (t * s) ** 2  # Var(x_t | k), 두 성분 같음
    mk = t * means  # E[x_t | k]
    diff = x[:, None] - mk[None, :]  # (n, 2)
    logp = np.log(w)[None, :] - 0.5 * diff ** 2 / V - 0.5 * np.log(2 * np.pi * V)
    logp -= logp.max(axis=1, keepdims=True)
    r = np.exp(logp)
    r /= r.sum(axis=1, keepdims=True)  # 사후 r_k(x)
    # E[x1 | x_t, k] - E[x0 | x_t, k]
    coef = (t * s ** 2 - (1 - t)) / V
    per_k = means[None, :] + coef * diff  # (n,2)
    return (r * per_k).sum(axis=1)


def euler(x0, d, n_steps, s=S):
    x = x0.copy()
    h = 1.0 / n_steps
    for i in range(n_steps):
        t = i * h
        x = x + h * velocity(x, t, d, s)
    return x


def sample_target(n, d, s=S):
    k = RNG.integers(0, 2, size=n)
    centre = np.where(k == 0, -d / 2, d / 2)
    return centre + s * RNG.standard_normal(n)


def w1(a, b):
    """1차원 Wasserstein-1 (정렬된 분위수 평균 차이)."""
    a = np.sort(a)
    b = np.sort(b)
    q = np.linspace(0, 1, min(len(a), len(b)) + 2)[1:-1]
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def mode_coverage(x, d, s=S):
    """출력 중 어느 모드 근처(3s 이내)에 있는 비율과, 두 모드 모두 채워졌는지(최소 비율)."""
    centres = np.array([-d / 2, d / 2])
    near = np.abs(x[:, None] - centres[None, :]) < 3 * s
    frac = near.any(axis=1).mean()
    per_mode = near.mean(axis=0)
    return float(frac), float(per_mode.min())


def geometry(x0, d, n_steps=1000, s=S):
    """논문 A.2 식. Excess: 경로 길이/현 길이 - 1 (%). 현 = |x1 - x0|. 샘플 수 적게."""
    h = 1.0 / n_steps
    x = x0.copy()
    path = np.zeros_like(x0)
    for i in range(n_steps):
        t = i * h
        v = velocity(x, t, d, s)
        path += np.abs(h * v)
        x = x + h * v
    chord = np.abs(x - x0)
    ok = chord > 1e-6
    ratio = path[ok] / chord[ok]
    # 현 길이가 0에 가까운 표본이 평균을 망가뜨리므로 중앙값과 합의 비율을 같이 쓴다
    excess = 100 * (np.median(ratio) - 1)
    excess_sum = 100 * (path[ok].sum() / chord[ok].sum() - 1)
    return float(excess), float(excess_sum), x


def main():
    x0 = RNG.standard_normal(N_SAMPLES)
    rows = []
    for d in SEPS:
        target = sample_target(N_SAMPLES, d)
        out1 = euler(x0, d, 1)
        out10 = euler(x0, d, 10)
        out1000 = euler(x0, d, 1000)
        cov1 = mode_coverage(out1, d)
        cov_ref = mode_coverage(out1000, d)
        excess, excess_sum, x_ref = geometry(x0[:2000], d)
        out1_small = out1[:2000]
        endpoint_err = float(np.mean(np.abs(out1_small - x_ref)) / np.mean(np.abs(x_ref - x0[:2000])))
        rows.append({
            "d": d, "d_over_s": round(d / S, 2),
            "W1_one_step_vs_target": round(w1(out1, target), 4),
            "W1_10step_vs_target": round(w1(out10, target), 4),
            "W1_1000step_vs_target": round(w1(out1000, target), 4),
            "one_step_output_std": round(float(out1.std()), 5),
            "one_step_output_mean": round(float(out1.mean()), 5),
            "target_std": round(float(target.std()), 4),
            "mode_coverage_one_step": {"any_mode": cov1[0], "min_mode": cov1[1]},
            "mode_coverage_1000step": {"any_mode": cov_ref[0], "min_mode": cov_ref[1]},
            "excess_percent_1000step_path_median": round(excess, 3),
            "excess_percent_1000step_path_sumratio": round(excess_sum, 3),
            "endpoint_error_1step_over_chord": round(endpoint_err, 4),
        })
    # 1스텝 출력이 x0 에 무관하게 목표 평균 (0) 으로 붕괴하는지 확인 (d 가 클 때)
    check = euler(x0[:5], 4.0, 1)
    result = {
        "setup": {"s": S, "n_samples": N_SAMPLES, "seed": 0, "target": "0.5 N(-d/2,s^2)+0.5 N(d/2,s^2)",
                  "field": "exact E[x1-x0 | x_t] for rectified flow (closed form, no training)"},
        "one_step_collapse_check_d4": {"x0": [round(float(v), 4) for v in x0[:5]],
                                       "one_step_output": [round(float(v), 6) for v in check]},
        "rows": rows,
        "note": "Euler N=1 은 x0 + v(x0, 0) 이며 v(x0,0)=E[x1]-x0 이므로 출력은 x0 에 무관하게 E[x1]=0 이다. 표 값은 이 식과 맞는지 확인한다.",
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT)
    for r in rows:
        print(f"d={r['d']:>4} d/s={r['d_over_s']:>6}  W1(1)={r['W1_one_step_vs_target']:.4f}  "
              f"W1(10)={r['W1_10step_vs_target']:.4f}  W1(1000)={r['W1_1000step_vs_target']:.4f}  "
              f"std1={r['one_step_output_std']:.4f} cov1={r['mode_coverage_one_step']} excess_med%={r['excess_percent_1000step_path_median']} excess_sum%={r['excess_percent_1000step_path_sumratio']}")


if __name__ == "__main__":
    main()
