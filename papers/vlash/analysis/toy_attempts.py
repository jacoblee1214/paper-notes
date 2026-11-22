"""토이 1: '20 롤아웃, 롤아웃당 최대 3회 시도, 성공 수/전체 시도 수' 지표(논문 V-B 1절)의 성질.
성공하면 그 롤아웃은 끝나고, 실패하면 다시 시도한다(최대 3회). 보고 SR = 성공/전체 시도.
확인: (a) 이 SR이 무엇을 재는가 (b) 이항 표준오차 sqrt(p(1-p)/N)를 에러바로 쓰면 구간이 맞는가.
가정(우리 모형): 롤아웃 난이도 q_i ~ Beta(a,b)로 이질적, 같은 롤아웃의 시도는 q_i로 독립. numpy, 약 1초.
실행: python3 analysis/toy_attempts.py → analysis/toy_attempts_results.json
"""
import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)
HERE = Path(__file__).resolve().parent


def run(mean_q, conc, n_roll=20, max_att=3, reps=20000):
    a, b = mean_q * conc, (1 - mean_q) * conc
    q = rng.beta(a, b, size=(reps, n_roll))
    succ = np.zeros((reps, n_roll), bool)
    att = np.zeros((reps, n_roll), int)
    alive = np.ones((reps, n_roll), bool)
    for _ in range(max_att):
        s = rng.random((reps, n_roll)) < q
        att += alive
        newly = alive & s
        succ |= newly
        alive &= ~s
    S = succ.sum(1)
    N = att.sum(1)
    sr = S / N
    se = np.sqrt(sr * (1 - sr) / N)
    # 모집단의 참값: 이 지표의 기대값 근사(대량 반복의 ΣS/ΣN)
    true_sr = S.sum() / N.sum()
    cover = np.mean(np.abs(sr - true_sr) <= se)
    cover2 = np.mean(np.abs(sr - true_sr) <= 1.96 * se)
    return {"mean_q": mean_q, "conc": conc, "mean_attempts": float(N.mean()), "sr_mean": float(sr.mean()),
            "attempt_level_true": float(true_sr), "mean_of_q": float(q.mean()),
            "rollout_level_pass_at_3": float(succ.mean()),
            "sd_of_sr": float(sr.std()), "mean_reported_se": float(se.mean()),
            "coverage_1se": float(cover), "coverage_1.96se": float(cover2)}


out = [run(m, c) for m in (0.3, 0.5, 0.8) for c in (1.0, 4.0, 1000.0)]
json.dump(out, open(HERE / "toy_attempts_results.json", "w"), indent=1)
for o in out:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items()})
