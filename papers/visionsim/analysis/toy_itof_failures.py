"""토이 T2: iToF 식 2로 위상 모호, 포화, 주변광, 잡음이 만드는 깊이 오차를 재현한다.

검증하는 주장: 논문 Sec. 4.2와 S.5.3(그림 2). (i) 변조 주파수 35 MHz에서 약 4.28 m를 넘는 깊이가 되접힌다.
(ii) 완전 용량(50k e-)을 넘으면 포화로 깊이가 틀어진다. (iii) 주변광이 활성 광원 세기와 같아지면 먼 점일수록 오차가 커진다.
(v) 샷/읽기 잡음(읽기 10 e-)과 12비트 ADC가 고주파 깊이 잡음을 만든다.
공식 저장소의 iToF 방출기는 잡음 없이 상관 샘플만 낸다(docs/sections/sensors/itof.rst). 이 토이는 식 2를 직접 구현했다.

모델: 3탭 정현파, B_i = Q(beta * F_i(d) + A + noise), F_i = 0.5 (1 + cos(4 pi f d / c - 2 pi i / 3)),
beta = beta1 / d^2 (d=1 m에서 신호 진폭 beta1 전자), 탭 복호 phi = atan2(sqrt(3)(B1 - B2), 2 B0 - B1 - B2).
장면은 0.5-7 m의 깊이 램프, 알베도 일정. 40 MHz가 아니라 논문 그림 2의 35 MHz와 20 MHz를 쓴다.
실행: python3 analysis/toy_itof_failures.py (numpy, 1초 미만, 결과 toy_itof_failures_results.json)
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_itof_failures_results.json"
C = 299_792_458.0
rng = np.random.default_rng(1)


def measure(d, f, beta1=40_000.0, ambient=0.0, fwc=50_000.0, read=0.0, bits=None, shot=False, saturate=False):
    phase = 4 * np.pi * f * d / C
    taps = []
    for i in range(3):
        sig = (beta1 / d**2) * 0.5 * (1 + np.cos(phase - 2 * np.pi * i / 3))
        b = sig + ambient
        if shot:
            b = rng.poisson(np.maximum(b, 0)).astype(float)
        if read > 0:
            b = b + rng.normal(0, read, b.shape)
        if saturate:
            b = np.minimum(b, fwc)
        if bits:
            q = 2**bits - 1
            b = np.round(np.clip(b, 0, fwc) / fwc * q) / q * fwc
        taps.append(b)
    return taps


def decode(taps, f):
    b0, b1, b2 = taps
    phi = np.arctan2(np.sqrt(3) * (b1 - b2), 2 * b0 - b1 - b2) % (2 * np.pi)
    return phi * C / (4 * np.pi * f)


def rmse_in(depth, est, lo, hi, wrap_m=None):
    m = (depth >= lo) & (depth < hi)
    e = est[m] - depth[m]
    return float(np.sqrt(np.mean(e**2))), float(np.median(np.abs(e)))


def main():
    d = np.tile(np.linspace(0.5, 7.0, 6500), 50)  # 50회 반복으로 잡음 평균
    res = {"setup": "3탭 정현파, beta1=40k e- (d=1 m), fwc=50k e-, 읽기 10 e-, 12비트", "cases": {}}
    f35, f20 = 35e6, 20e6
    cases = {
        "ideal_35MHz": dict(f=f35),
        "ideal_20MHz": dict(f=f20),
        "noise_35MHz": dict(f=f35, shot=True, read=10.0, bits=12),
        "ambient_equal_active_35MHz": dict(f=f35, ambient=40_000.0 / 2, shot=True, read=10.0, bits=12),
        "saturation_35MHz": dict(f=f35, beta1=400_000.0, saturate=True, bits=12),
    }
    for name, kw in cases.items():
        f = kw.pop("f")
        est = decode(measure(d, f, **kw), f)
        wrap = C / (2 * f)
        near = rmse_in(d, est, 0.5, 2.0)
        mid = rmse_in(d, est, 2.0, min(4.0, wrap))
        far_in = rmse_in(d, est, min(4.0, wrap), wrap) if wrap > 4.0 else (None, None)
        beyond = rmse_in(d, est, wrap, 7.0) if wrap < 7.0 else (None, None)
        res["cases"][name] = {"wrap_range_m": wrap, "rmse_median_0.5_2m": near, "rmse_median_2_4m": mid,
                              "rmse_median_4m_to_wrap": far_in, "rmse_median_beyond_wrap": beyond}
        # 되접힘 검사: 4.28 m를 넘는 깊이가 d - wrap 로 읽히는지
        if wrap < 7.0:
            m = d > wrap + 0.2
            fold = est[m] - (d[m] - wrap)
            res["cases"][name]["fold_error_rmse_m_beyond_wrap"] = float(np.sqrt(np.mean(fold**2)))
    # 먼 점일수록 잡음 오차가 커지는지: 구간별 중앙값 오차
    est_n = decode(measure(d, f20, shot=True, read=10.0, bits=12), f20)
    est_a = decode(measure(d, f20, ambient=20_000.0, shot=True, read=10.0, bits=12), f20)
    edges = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 7.0]
    prof = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        prof.append({"range_m": [lo, hi],
                     "noise_only_rmse_m": rmse_in(d, est_n, lo, hi)[0],
                     "ambient_eq_active_rmse_m": rmse_in(d, est_a, lo, hi)[0]})
    res["distance_profile_20MHz"] = prof
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
