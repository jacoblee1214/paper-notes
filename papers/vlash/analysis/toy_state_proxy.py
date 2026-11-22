"""토이 2: '절대 위치 행동의 마지막 값을 미래 상태로 쓴다'(논문 IV-A, 공식 코드 run.py)의 오차.
로봇 상태가 명령을 1차 지연으로 따라간다고 가정: s[t+1] = s[t] + alpha (a[t] - s[t]) (alpha=1이면 즉시 추종).
기준 궤적은 임의 주파수의 사인 합. 지연 Δ 스텝 뒤의 실제 상태 s[t+Δ]에 대해
  naive : 현재 상태 s[t]를 그대로 씀(오래된 상태)
  proxy : 마지막 명령 a[t+Δ-1]을 씀(VLASH, 코드 경로)
  model : 명령열 a[t..t+Δ-1]를 1차 모델로 굴림(추종 모형을 알 때)
오차는 궤적 진폭으로 정규화. 속도 배율 v(행동 양자화 q 흉내: 시간축을 v배 압축)를 키우면 어떻게 되는지 본다.
가정: 1차 지연, 지연 Δ는 정수 스텝, 상태 관측 잡음 없음. numpy, 1초 미만.
실행: python3 analysis/toy_state_proxy.py → analysis/toy_state_proxy_results.json
"""
import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(1)
HERE = Path(__file__).resolve().parent


def sim(alpha, delta, v, T=4000, runs=40):
    errs = {"naive": [], "proxy": [], "model": []}
    for _ in range(runs):
        t = np.arange(T)
        f = rng.uniform(0.002, 0.02, size=4) * v
        ph = rng.uniform(0, 2 * np.pi, 4)
        a = sum(np.sin(2 * np.pi * fi * t + p) for fi, p in zip(f, ph)) / 4
        s = np.zeros(T)
        for k in range(T - 1):
            s[k + 1] = s[k] + alpha * (a[k] - s[k])
        idx = np.arange(0, T - delta - 1)
        truth = s[idx + delta]
        naive = s[idx]
        proxy = a[idx + delta - 1]
        model = s[idx].copy()
        for j in range(delta):
            model = model + alpha * (a[idx + j] - model)
        sd = a.std()
        for n, e in (("naive", naive), ("proxy", proxy), ("model", model)):
            errs[n].append(np.sqrt(np.mean((truth - e) ** 2)) / sd)
    return {k: float(np.mean(v_)) for k, v_ in errs.items()}


out = []
for alpha in (1.0, 0.5, 0.2, 0.1):
    for v in (1, 2, 4):
        r = sim(alpha, delta=4, v=v)
        r.update(alpha=alpha, speed=v, delta=4,
                 proxy_removes_pct_of_naive_error=round(100 * (1 - r["proxy"] / r["naive"]), 1))
        out.append(r)
json.dump(out, open(HERE / "toy_state_proxy_results.json", "w"), indent=1)
for o in out:
    print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in o.items()})
