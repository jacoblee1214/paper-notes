"""토이: 식 (1)의 짝지은 재생 미세조정이 접촉에서 남기는 잔차 (1축 손가락, 준정적, numpy).

검사하려는 것
  논문 식 (1): L = ||g(f(e_h)) - e_h||^2 + beta ||f(e_h) - q_r||^2 + lambda ||df||^2
  두 번째 항은 명령을 "재생에 쓴 명령 q_r"(= 직전 매핑의 출력) 쪽으로 당긴다. 접촉에서 로봇 엔코더가 명령에
  둔감하면(순응 계수 a가 작으면) 첫 항의 기울기가 a^2에 비례해 작아져서 이 당김이 보정을 누른다.
  1축 선형 접촉에서는 닫힌 식으로 보정 후 남는 잔차 비율이 beta / (a^2 + beta)이다(재생 1회마다 곱해진다).
  (i) 이 식을 최소제곱 직접 풀이로 확인하고, (ii) 재생을 한 번 더 한 경우(논문 "한 번 반복")의 잔차를 보고,
  (iii) 순방향 모델 g의 접촉 기울기를 엔코더 잡음이 있는 재생 데이터에서 적합할 때 보정이 얼마나 어긋나는지 몬테카를로로 본다.

모형(논문에 없는 가정)
  로봇: 명령 q -> 엔코더 e_r = q (q <= c), e_r = c + a (q - c) (접촉, q > c). a = kp / (kp + k_obj), c = 접촉 위치.
  사람: 닫는 동작 후 물체를 눌러 e_h가 c + 0.05까지 가고 유지한다.
  초기 매핑 f0(자유 공간 babbling으로 학습)은 항등으로 둔다. 접촉 구간의 매핑은 q = e + k (e - c)로 매개화하고 k를 맞춘다.
  g는 접촉 기울기 a_hat를 갖는 선형 모델(정답 또는 재생 데이터 적합). beta, lambda, 이력 길이는 논문이 값을 밝히지 않았다.
한계: 1축, 준정적, 히스테리시스 없음, 신경망 아님. 닫힌 식의 대수를 확인하는 수준이고 실로봇 결과가 아니다.
실행: python3 analysis/toy_paired_replay.py   (numpy, 수 초)
출력: analysis/toy_paired_replay_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_paired_replay_results.json"
PRESS = 0.05
BETAS = [0.0, 0.01, 0.1, 1.0]
A_LIST = [0.2, 0.4, 0.6, 0.9]


def contact_samples(c, n_ramp=10, n_hold=40):
    """닫기 구간 중 접촉 이후(c -> c+PRESS)와 유지 구간의 사람 엔코더 값."""
    return np.concatenate([np.linspace(c, c + PRESS, n_ramp), np.full(n_hold, c + PRESS)])


def residual(c, a, k):
    """사람 e_h를 q = e + k (e - c)로 재생했을 때 유지 구간의 |e_r - e_h|."""
    e = c + PRESS
    q = e + k * (e - c)
    return abs((c + a * (q - c)) - e)


def refine(c, a_true, a_hat, beta, k_prev):
    """식 (1)을 접촉 샘플에서 k에 대해 최소화한다(닫힌 최소제곱, lambda 항은 뺌).
    x = e - c.  q - c = x (1 + k).  term1 = sum (a_hat x (1 + k) - x)^2,  term2 = beta sum x^2 (k - k_prev)^2.
    d/dk = 0  ->  k = (a_hat (1 - a_hat) + beta k_prev) / (a_hat^2 + beta).   (beta = 0 이면 k = (1 - a_hat) / a_hat)
    """
    den = a_hat * a_hat + beta
    return float((a_hat * (1 - a_hat) + beta * k_prev) / den) if den > 0 else k_prev


def main():
    res = {"setup": {"press": PRESS, "betas": BETAS, "a_list": A_LIST,
                     "note": "f0 = 항등(k = 0). 초기 잔차 = (1 - a) * 0.05"}}
    c = 0.5
    closed = {}
    for a in A_LIST:
        init = residual(c, a, 0.0)
        row = {"initial": round(init, 5)}
        for beta in BETAS:
            k = 0.0
            out = []
            for rd in (1, 2):
                k = refine(c, a, a, beta, k)
                out.append(round(residual(c, a, k), 5))
            frac_pred = beta / (a * a + beta) if a * a + beta else 0.0
            row[f"beta={beta}"] = {"after_round1": out[0], "after_round2": out[1],
                                   "closed_form_fraction_per_round": round(frac_pred, 4),
                                   "closed_form_r1": round(init * frac_pred, 5), "closed_form_r2": round(init * frac_pred ** 2, 5)}
        closed[f"a={a}"] = row
    res["exact_g"] = closed

    # 순방향 모델 g의 접촉 기울기 a_hat를 재생 데이터(닫기 구간의 q in (c, c+0.05], 유지 구간)에서 적합.
    # 엔코더 잡음 sigma. 1회차 재생 명령은 q = e (항등)라 q의 변화 폭이 PRESS = 0.05 이하이다.
    rng = np.random.default_rng(0)
    noisy = {}
    for sigma in (0.0, 0.002, 0.005, 0.01):
        row = {}
        for a in (0.2, 0.4, 0.9):
            errs = []
            slopes = []
            for _ in range(300):
                e = contact_samples(c)
                q = e.copy()                      # 항등 재생
                e_r = c + a * (q - c) + rng.normal(0, sigma, q.shape)
                x = q - c
                a_hat = float(np.sum(x * (e_r - c)) / np.sum(x * x))   # 원점(접촉점) 통과 최소제곱
                a_hat = max(a_hat, 1e-3)
                k = refine(c, a, a_hat, 0.0, 0.0)
                errs.append(residual(c, a, k))
                slopes.append(a_hat)
            row[f"a={a}"] = {"a_hat_mean": round(float(np.mean(slopes)), 3), "a_hat_sd": round(float(np.std(slopes)), 3),
                             "residual_after_round1_mean": round(float(np.mean(errs)), 5),
                             "residual_after_round1_p95": round(float(np.percentile(errs, 95)), 5),
                             "initial": round(residual(c, a, 0.0), 5)}
        noisy[f"sigma={sigma}"] = row
    res["learned_g_slope_noise"] = noisy
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
