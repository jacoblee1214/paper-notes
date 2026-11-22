"""토이 T1: 단일 광자 LiDAR의 비마비형 데드타임이 만드는 파일업과 거리 걸음(range walk).

검증하는 주장: 논문 Sec. 4.1, S.5.2.1. "데드타임 때문에 늦게 도착한 광자가 체계적으로 사라지고, 히스토그램이
이른 구간으로 치우쳐 얕은 깊이를 낸다." 강한 반사(레트로리플렉터)일수록 이 편향이 커지는지 본다.

설정(가정, 논문 값은 ADS6311 사양 S.6.1에서만 가져옴):
  시간 구간 754 ps, 데드타임 11 ns(비마비형), 레이저 주기 256구간(약 193 ns).
  IRF는 가우시안 시그마 2구간(논문에 IRF 값이 없어 가정), 배경 0.05광자/주기(균일).
  신호 평균 광자 수 phi_sig를 0.01-10으로 바꾼다. 10만 주기. 자유 주행(free-running) 방식:
  검출 뒤 데드타임 동안 도착한 광자는 버린다. 주기 경계를 넘는 데드타임도 이어서 적용한다.
추정: (a) 히스토그램을 5구간 박스 필터로 평활한 뒤 최대 구간(분해능 1구간), (b) 참 구간 주변 +-15구간의 무게중심.
신호 중심 시간은 100.0이고 구간 k는 [k, k+1)이므로 구간 인덱스로 본 기준값은 99.5(REF)다.
실행: python3 analysis/toy_dtof_pileup.py  (numpy, 수 초, 결과 toy_dtof_pileup_results.json)
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_dtof_pileup_results.json"
BIN_PS, DEAD_NS, N_BINS, TRUE_BIN, SIGMA = 754.0, 11.0, 256, 100, 2.0
DEAD_BINS = DEAD_NS * 1000.0 / BIN_PS
BG, N_CYCLES = 0.05, 100_000
REF = TRUE_BIN - 0.5
CM_PER_BIN = 100 * 299_792_458.0 * BIN_PS * 1e-12 / 2


def simulate(phi_sig, rng):
    n_sig = rng.poisson(phi_sig, N_CYCLES)
    n_bg = rng.poisson(BG, N_CYCLES)
    cyc_sig = np.repeat(np.arange(N_CYCLES), n_sig)
    cyc_bg = np.repeat(np.arange(N_CYCLES), n_bg)
    t_sig = rng.normal(TRUE_BIN, SIGMA, cyc_sig.size).clip(0, N_BINS - 1e-6)
    t_bg = rng.uniform(0, N_BINS, cyc_bg.size)
    t = np.concatenate([cyc_sig * N_BINS + t_sig, cyc_bg * N_BINS + t_bg])
    is_sig = np.concatenate([np.ones(t_sig.size, bool), np.zeros(t_bg.size, bool)])
    order = np.argsort(t)
    t, is_sig = t[order], is_sig[order]
    keep = np.zeros(t.size, bool)
    free_at = -1e18
    for i in range(t.size):
        if t[i] >= free_at:
            keep[i] = True
            free_at = t[i] + DEAD_BINS
    det = t[keep]
    hist = np.bincount((det % N_BINS).astype(int), minlength=N_BINS).astype(float)
    return hist, int(is_sig.sum()), int((keep & is_sig).sum())


def main():
    rng = np.random.default_rng(0)
    rows = []
    for phi in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        hist, n_sig_photons, n_sig_detected = simulate(phi, rng)
        smooth = np.convolve(hist, np.ones(5) / 5, mode="same")
        peak = int(np.argmax(smooth))
        win = slice(TRUE_BIN - 15, TRUE_BIN + 16)
        centroid = float((np.arange(N_BINS)[win] * hist[win]).sum() / max(hist[win].sum(), 1))
        rows.append({
            "phi_sig_per_cycle": phi,
            "peak_bin": peak,
            "peak_bias_bins": peak - REF,
            "peak_bias_cm": (peak - REF) * CM_PER_BIN,
            "centroid_bias_bins": centroid - REF,
            "centroid_bias_cm": (centroid - REF) * CM_PER_BIN,
            "signal_photon_detection_fraction": n_sig_detected / max(n_sig_photons, 1),
            "total_counts_in_window": float(hist[win].sum()),
        })
    res = {"setup": {"bin_ps": BIN_PS, "dead_ns": DEAD_NS, "dead_bins": DEAD_BINS, "n_bins": N_BINS, "true_bin": TRUE_BIN,
                     "irf_sigma_bins_assumed": SIGMA, "background_per_cycle": BG, "n_cycles": N_CYCLES,
                     "cm_per_bin": CM_PER_BIN}, "rows": rows}
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
