"""토이 2: 25 Hz 명령을 100 Hz로 0차 유지(ZOH)하면 추종 지연과 SPARC가 얼마나 달라지는가 (논문 부록 D, 그림 6·7).

명령은 2 Hz 이하 사인 합, 로봇은 시정수 tau_r의 1차 지연 추종. 지연 추정은 부록 D 서술을 따른다
(유한 차분 속도, 정규화 상호상관, tau in [0, 0.5] s). SPARC는 Balasubramanian 등의 표준 정의(fc=10 Hz, 영 패딩 4, 진폭 문턱 0.05).
두 가지 관행(측정 신호가 ZOH 명령에 대한 응답인 경우와 공통 측정 신호인 경우)을 모두 잰다. 논문의 실제 구현은 공개되지 않았으므로 정의의 세부(예: 속도 노름, 문턱)는 가정이다.
실행: python3 analysis/toy_zoh_lag_sparc.py (numpy, 수 초)
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_zoh_lag_sparc_results.json"
FS = 100.0


def sparc(speed, fs=FS, pad=4, fc=10.0, th=0.05):
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + pad))
    f = np.arange(nfft) * fs / nfft
    mag = np.abs(np.fft.fft(speed, nfft))
    mag = mag / mag.max()
    sel = f <= fc
    f, mag = f[sel], mag[sel]
    idx = np.where(mag >= th)[0]
    f, mag = f[idx[0]:idx[-1] + 1], mag[idx[0]:idx[-1] + 1]
    return -np.sum(np.sqrt(np.diff(f / f[-1]) ** 2 + np.diff(mag) ** 2))


def lag(cmd, meas, fs=FS, maxlag=0.5):
    vc, vm = np.diff(cmd) * fs, np.diff(meas) * fs
    best, bl = -2, 0
    for k in range(int(maxlag * fs) + 1):
        a, b = vc[:len(vc) - k], vm[k:]
        c = np.dot(a - a.mean(), b - b.mean()) / (np.linalg.norm(a - a.mean()) * np.linalg.norm(b - b.mean()))
        if c > best:
            best, bl = c, k
    return bl / fs


def follow(cmd, tau_r, fs=FS):
    out = np.zeros_like(cmd)
    a = 1 - np.exp(-1 / (fs * tau_r))
    for i in range(1, len(cmd)):
        out[i] = out[i - 1] + a * (cmd[i - 1] - out[i - 1])
    return out


def main():
    rng = np.random.default_rng(0)
    res = {"tau_r": {}}
    for tau_r in (0.03, 0.08):
        L100, L25, L25c, S100, S25 = [], [], [], [], []
        for _ in range(40):
            t = np.arange(0, 20, 1 / FS)
            cmd = sum(rng.uniform(0.2, 1) * np.sin(2 * np.pi * rng.uniform(0.2, 2.0) * t + rng.uniform(0, 6.28)) for _ in range(4))
            zoh = np.repeat(cmd[::4], 4)[:len(cmd)]
            L100.append(lag(cmd, follow(cmd, tau_r)))
            L25.append(lag(zoh, follow(zoh, tau_r)))      # 측정 신호가 ZOH 명령에 대한 응답
            L25c.append(lag(zoh, follow(cmd, tau_r)))     # 측정 신호가 100 Hz 원 명령에 대한 응답(공통 측정 신호 관행)
            S100.append(sparc(np.abs(np.diff(cmd)) * FS))
            S25.append(sparc(np.abs(np.diff(zoh)) * FS))
        res["tau_r"][str(tau_r)] = {"lag_100Hz_ms": 1000 * np.mean(L100), "lag_zoh25_ms": 1000 * np.mean(L25),
                                    "lag_increase_ms": 1000 * (np.mean(L25) - np.mean(L100)),
                                    "lag_zoh25_common_meas_ms": 1000 * np.mean(L25c),
                                    "sparc_100Hz": float(np.mean(S100)), "sparc_zoh25": float(np.mean(S25))}
        print(tau_r, {k: round(v, 2) for k, v in res["tau_r"][str(tau_r)].items()})
    res["paper_lag_gap_ms"] = 1000 * (0.189 - 0.118)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
