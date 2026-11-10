"""토이 B: 촉각 출력의 히스테리시스·드리프트가 힘 추정에 주는 오차 (numpy, 1스레드, 수 초).

논문은 "Force calibration across curved surfaces remains future work"(§8)라서 힘 보정 자체가 없습니다.
이 토이는 보정 없이 법선 힘 크기만 읽을 때 생기는 오차의 규모를 봅니다. 모든 계수는 가정입니다.
- 힘: 0 -> 5 N -> 0 삼각파 30회(약 10분), 한 주기 20 s, 120 Hz (dt = 1/120)
- 출력 모델: s = F^b (b = 0.7, 단조 증가), 로딩 구간은 그대로
  언로딩 구간은 히스테리시스 h를 곱해 s = s0 * (1 - h)  (h in {0, 0.05, 0.10, 0.20})
  드리프트: s *= (1 + k * t/60), k는 분당 비율 (k in {0, 0.005, 0.01, 0.02})
- 보정: 같은 센서의 깨끗한 로딩 곡선(히스테리시스·드리프트 없음, 1회)으로 log s = a + b log F를 맞춘 뒤 역변환
- 지표: 힘 오차 RMSE(N), 최대 절댓값 오차(N), Fmax 대비 비율(%)
실행: python3 analysis/hysteresis_toy.py -> analysis/hysteresis_results.json
"""
import json
import os
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FMAX, B = 5.0, 0.7
DT, PERIOD, CYCLES = 1 / 120, 20.0, 30
H_LIST = [0.0, 0.05, 0.10, 0.20]
K_LIST = [0.0, 0.005, 0.01, 0.02]  # 분당 비율

t = np.arange(0, PERIOD * CYCLES, DT)
phase = (t % PERIOD) / PERIOD
F = FMAX * (1 - np.abs(2 * phase - 1))           # 삼각파
unload = np.gradient(F) < 0


def sensor(h, k):
    s = F ** B
    s = np.where(unload, s * (1 - h), s)
    return s * (1 + k * t / 60.0)


# 보정: 깨끗한 로딩 곡선 1회 (0~Fmax 로딩 램프)
Fc = np.linspace(0.05, FMAX, 200)
sc = Fc ** B
A = np.stack([np.ones_like(sc), np.log(Fc)], 1)
coef, *_ = np.linalg.lstsq(A, np.log(sc), rcond=None)
a0, b0 = coef


def invert(s):
    return np.exp((np.log(np.clip(s, 1e-9, None)) - a0) / b0)


res = {"config": {"Fmax_N": FMAX, "b": B, "dt_s": DT, "period_s": PERIOD, "cycles": CYCLES,
                  "note": "모든 계수는 가정. 논문에는 힘 보정 데이터가 없음"},
       "fit": {"a": round(float(a0), 4), "b": round(float(b0), 4)},
       "rmse_N": {}, "max_err_N": {}, "rmse_pct_Fmax": {}, "unload_bias_at_2p5N_N": {}}
for h in H_LIST:
    for k in K_LIST:
        Fhat = invert(sensor(h, k))
        e = Fhat - F
        key = f"h={h:.2f},k={k:.3f}"
        res["rmse_N"][key] = round(float(np.sqrt(np.mean(e ** 2))), 4)
        res["max_err_N"][key] = round(float(np.max(np.abs(e))), 4)
        res["rmse_pct_Fmax"][key] = round(100 * float(np.sqrt(np.mean(e ** 2))) / FMAX, 2)
        # 2.5 N 근처 언로딩 구간의 평균 편향
        m = unload & (np.abs(F - 2.5) < 0.1)
        res["unload_bias_at_2p5N_N"][key] = round(float(np.mean(e[m])), 4) if m.any() else None
(ROOT / "analysis" / "hysteresis_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
print(json.dumps(res, indent=2, ensure_ascii=False))
