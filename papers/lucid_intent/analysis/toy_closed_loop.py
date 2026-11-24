"""토이: 폐루프 재질의 대 일회 질의 (LUCID §4.1의 "폐루프가 견고성을 만든다" 주장의 구조만 확인).

목적
- 논문의 개루프 기준선은 (a) 의도 출처가 다르고(Veo 3.1 영상 대 학습된 f_theta) (b) 한 번만 질의한다.
  이 토이는 "같은 의도 모델을 한 번만 질의하는" 기준선이 있으면 폐루프의 이득이 어디서 오는지 분리된다는 것을
  가장 단순한 2차원 점 물체로 보인다. LUCID의 수치를 재현하거나 예측하지 않는다.

설정 (모두 가정이며 논문 값이 아니다)
- 물체 위치 x in R^2, 목표 g, 초기 거리 1.0. 의도 모델은 "다음 1초 동안 목표 쪽으로 가야 할 변위"를 예측한다
  (최대 0.25/s, 예측 잡음 sigma_pred). 실행기는 요청 변위의 kappa 배(평균 1.0, SD 0.1)를 실현한다.
- 교란: 매초 확률 p_slip로 물체가 크기 d의 임의 방향으로 튄다(잘못 잡음, 미끄러짐).
- 세 가지 방식
    closed      : 매초 현재 위치로 f_theta를 다시 질의 (LUCID)
    open_rel    : t=0에 한 번 질의해 변위열 계획을 얻고, 현재 위치와 상관없이 그 변위열을 실행
    open_abs    : t=0에 계획 경로(절대 위치)를 얻고, 매초 계획의 다음 점으로 가도록 실행 (계획을 추종)
- 12초 후 목표 반경 0.1 안이면 성공. 시행 500회, 시드 고정.

실행: python3 analysis/toy_closed_loop.py   (numpy, 수 초)
출력: analysis/toy_closed_loop_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_closed_loop_results.json"
T_SEC, V_MAX, KAPPA, KAPPA_SD, D_SLIP, TOL = 12, 0.25, 1.0, 0.1, 0.3, 0.1


def predict(x, g, rng, sigma):
    delta = g - x
    n = np.linalg.norm(delta)
    step = delta if n <= V_MAX else delta / n * V_MAX
    return step + rng.normal(0, sigma, 2)


def run(mode, p_slip, sigma, rng):
    g = np.array([1.0, 0.0])
    x = np.zeros(2)
    plan_steps, plan_pts = [], []
    if mode != "closed":
        xp = x.copy()
        for _ in range(T_SEC):
            s = predict(xp, g, rng, sigma)
            xp = xp + s
            plan_steps.append(s)
            plan_pts.append(xp.copy())
    for t in range(T_SEC):
        if mode == "closed":
            req = predict(x, g, rng, sigma)
        elif mode == "open_rel":
            req = plan_steps[t]
        else:
            req = plan_pts[t] - x
            n = np.linalg.norm(req)
            if n > V_MAX:
                req = req / n * V_MAX
        x = x + req * max(0.0, rng.normal(KAPPA, KAPPA_SD))
        if rng.random() < p_slip:
            ang = rng.uniform(0, 2 * np.pi)
            x = x + D_SLIP * np.array([np.cos(ang), np.sin(ang)])
    return np.linalg.norm(x - g) < TOL


def main():
    rng = np.random.default_rng(0)
    res = {}
    for sigma in (0.01, 0.04):
        for p_slip in (0.0, 0.05, 0.15):
            key = f"sigma{sigma}_slip{p_slip}"
            res[key] = {}
            for mode in ("closed", "open_rel", "open_abs"):
                k = sum(run(mode, p_slip, sigma, rng) for _ in range(500))
                res[key][mode] = k / 500
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"{'조건':<22}{'closed':>8}{'open_rel':>10}{'open_abs':>10}")
    for k, v in res.items():
        print(f"{k:<22}{v['closed']:>8.2f}{v['open_rel']:>10.2f}{v['open_abs']:>10.2f}")


if __name__ == "__main__":
    main()
