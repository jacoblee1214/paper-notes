"""두 가지 작은 toy (numpy, 1스레드, 수 초).

1) 시도 수별 검출력: 백서가 쓰는 실물 시도 수(과제당 20, Dexter 213, 쌍 216)에서
   두 성공률 차이를 얼마나 자주 잡아내는지 몬테카를로로 본다. 과제 군집 효과는 넣지 않는다.
2) 메모리 적합성: 2B 파라미터 DiT 가중치만의 크기를 정밀도별로 계산해 24 GiB 맥과 비교한다.
   VAE·VLM(동결) 크기는 백서에 없으므로 제외한다.

실행: OMP_NUM_THREADS=1 python3 analysis/toy_power_fit.py  →  analysis/toy_power_fit_results.json
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "toy_power_fit_results.json"
rng = np.random.default_rng(0)
SIMS = 20000
ALPHA_Z = 1.959964


def power(n, p1, p2):
    a = rng.binomial(n, p1, SIMS)
    b = rng.binomial(n, p2, SIMS)
    pa, pb = a / n, b / n
    pp = (a + b) / (2 * n)
    den = np.sqrt(np.maximum(pp * (1 - pp) * 2 / n, 1e-12))
    z = (pa - pb) / den
    return float(np.mean(np.abs(z) > ALPHA_Z))


def wilson_cover(n, p, z=ALPHA_Z):
    k = rng.binomial(n, p, SIMS)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return float(np.mean((c - h <= p) & (p <= c + h)))


res = {"power_two_prop_alpha05": {}, "wilson_coverage_n20": {}}
# 백서 설정: 과제당 20 시도(Tab. 1), Dexter 213 (표 3), 쌍 216 (Fig. 10 실물)
for n in (20, 120, 213, 216):
    for diff in (0.10, 0.20, 0.30):
        res["power_two_prop_alpha05"][f"n={n},p1=0.5,diff={diff}"] = round(power(n, 0.5, 0.5 + diff), 3)
# 곡선용: 기준 성공률 0.5에서 차이를 0.05–0.40으로 바꿈
res["power_curve_p1_0.5"] = {}
for n in (20, 120, 213):
    res["power_curve_p1_0.5"][f"n={n}"] = {
        str(round(d, 2)): round(power(n, 0.5, 0.5 + d), 3) for d in np.arange(0.05, 0.401, 0.05)
    }
# Turbo 대 GR00T 2 (Dexter 평균 차 34pt를 n=20×7과제 ≈ 140 시도로 본 근사)
res["power_two_prop_alpha05"]["n=140,p1=0.686,diff=0.343(Dexter 평균)"] = round(power(140, 0.686, 0.343), 3)
for p in (0.5, 0.85, 0.95):
    res["wilson_coverage_n20"][f"p={p}"] = round(wilson_cover(20, p), 3)

# 메모리 적합성 (2B 파라미터, 가중치만)
N = 2.0e9
GiB = 2**30
mem = {}
for name, bytes_per in [("BF16", 2.0), ("FP8", 1.0), ("FP4(NVFP4 근사, 블록 scale 제외)", 0.5)]:
    mem[name] = round(N * bytes_per / GiB, 2)
res["weights_gib_2B_dit"] = mem
res["mac_budget_note"] = "24 GiB 통합 메모리. macOS가 GPU에 허용하는 작업 집합은 보통 전체의 3/4 안팎이라 약 18 GiB를 기준으로 본다(추정)"
res["simplified_total_check"] = {
    "bf16_dit_plus_unknown_frozen_vae_vlm_note": "VLM·VAE 크기는 백서 미기재. 2B 파라미터 DiT의 BF16 가중치만 4 GiB 안팎",
}

OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
print(json.dumps(res, indent=1, ensure_ascii=False))
