#!/usr/bin/env python3
"""토이 T1: 논문의 갱신 채택 규칙(새 task instance 100개에서 성공률이 떨어지면 거부, §4.2·그림 3)이
잡음 때문에 얼마나 틀리게 채택하는지, 반복하면 보고 성공률이 실제 성공률보다 얼마나 높아지는지.

실행: python3 analysis/toy_accept_rule.py   (numpy만, 수 초)
결과: analysis/toy_accept_rule_results.json

가정(논문에 없음): 갱신 하나의 실제 효과 d ~ N(mu, sd), 성공 여부는 instance마다 독립 베르누이.
- 'paired': 같은 100개 instance(같은 난이도 난수)에서 후보와 현재 그래프를 비교. 후보 성공 = U < p+d, 현재 = U < p.
- 'unpaired': 두 그래프를 서로 다른 100개에서 따로 측정.
채택 규칙: 측정 성공률(후보) >= 측정 성공률(현재)이면 채택(동률은 채택). 논문은 '떨어지면 거부'라고만 씀.
"""
import json
import os

import numpy as np

rng = np.random.default_rng(0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "toy_accept_rule_results.json")
N = 100


def accept_prob(p, d, paired, trials=20000):
    pc = np.clip(p + d, 0, 1)
    if paired:
        u = rng.random((trials, N))
        kc = (u < pc).sum(1)
        ki = (u < p).sum(1)
    else:
        kc = rng.binomial(N, pc, trials)
        ki = rng.binomial(N, p, trials)
    return float((kc >= ki).mean())


res = {"N_fresh_instances": N, "accept_prob": {}}
for p in (0.5, 0.9):
    for paired in (True, False):
        key = f"p={p},{'paired' if paired else 'unpaired'}"
        res["accept_prob"][key] = {f"d={d:+.2f}": round(accept_prob(p, d, paired), 3)
                                   for d in (-0.10, -0.05, -0.03, -0.02, -0.01, 0.0, 0.02, 0.05)}

# 반복 루프: 갱신 K번, 실제 효과 d ~ N(-0.01, 0.06) (대체로 해롭거나 중립인 편집이 섞인다고 가정)
def loop(p0, K, mu, sd, paired, runs=3000):
    true_final, reported_final, accepted_n = [], [], []
    for _ in range(runs):
        p = p0
        rep = None
        acc = 0
        for _k in range(K):
            d = rng.normal(mu, sd)
            pc = float(np.clip(p + d, 0, 1))
            if paired:
                u = rng.random(N)
                kc, ki = int((u < pc).sum()), int((u < p).sum())
            else:
                kc, ki = int(rng.binomial(N, pc)), int(rng.binomial(N, p))
            if kc >= ki:
                p, rep, acc = pc, kc / N, acc + 1
        true_final.append(p)
        reported_final.append(rep if rep is not None else float(np.nan))
        accepted_n.append(acc)
    return {"true_final_mean": round(float(np.mean(true_final)), 4),
            "reported_last_accepted_measured_mean": round(float(np.nanmean(reported_final)), 4),
            "accepted_updates_mean": round(float(np.mean(accepted_n)), 2),
            "true_start": p0}


res["loop_K10"] = {}
for p0 in (0.3, 0.9):
    for mu, sd in ((-0.01, 0.06), (0.0, 0.03)):
        for paired in (True, False):
            res["loop_K10"][f"p0={p0},mu={mu},sd={sd},{'paired' if paired else 'unpaired'}"] = loop(p0, 10, mu, sd, paired)

with open(OUT, "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
