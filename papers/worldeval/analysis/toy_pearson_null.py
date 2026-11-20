"""toy: 정책 4개일 때 순위·상관 지표가 우연으로도 얼마나 높게 나오는지 (numpy, 단일 프로세스).
- 실제 성공률은 가정값 4개(0.9, 0.6, 0.4, 0.1)로 두고, 프록시는 실제와 무관한 난수로 만든다.
- 귀무(무관) 분포에서 Pearson r >= 0.942(논문 표 1 WorldEval 평균)가 나오는 비율을 본다.
- 프록시가 실제와 약하게 관련된 경우(잡음 σ)도 함께 본다. 실행: python3 analysis/toy_pearson_null.py
"""
import json
from pathlib import Path
import numpy as np

rng = np.random.default_rng(0)
OUT = Path(__file__).resolve().parent / "toy_pearson_null_results.json"
TRUE = np.array([0.9, 0.6, 0.4, 0.1])
M = 200_000

def pearson_rows(a, b):
    a = a - a.mean(axis=1, keepdims=True); b = b - b.mean(axis=1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))

out = {"true_rates_assumed": TRUE.tolist(), "trials": M}
# 1) 귀무: 프록시가 완전히 무작위
proxy = rng.random((M, 4))
r0 = pearson_rows(np.tile(TRUE, (M, 1)), proxy)
out["null_p_r_ge_0942"] = float(np.mean(r0 >= 0.942))
out["null_p_r_ge_0958"] = float(np.mean(r0 >= 0.958))
out["null_p_abs_r_ge_095"] = float(np.mean(np.abs(r0) >= 0.95))

# 2) 프록시 = 실제 + N(0, σ): σ별로 r>=0.942가 나오는 비율
sig_res = {}
for sig in [0.02, 0.05, 0.1, 0.2]:
    p = TRUE + rng.normal(0, sig, (M, 4))
    r = pearson_rows(np.tile(TRUE, (M, 1)), p)
    sig_res[str(sig)] = {"median_r": float(np.median(r)), "p_r_ge_0942": float(np.mean(r >= 0.942))}
out["signal_plus_noise"] = sig_res

# 3) 실제 성공률이 40회 시도에서 추정됐다면 (이항 잡음) 추정 자체가 흔들림
real_hat = rng.binomial(40, TRUE, (M, 4)) / 40
out["real_estimate_sd_40_trials"] = float(np.std(real_hat[:, 0]))

OUT.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
