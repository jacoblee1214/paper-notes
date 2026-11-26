"""부록 I의 방향 불변성 주장을 수치로 확인하는 작은 CPU 토이 (numpy, 1초 안팎).

논문 식 (5), (13)–(14): 축별 log-mel을 log-sum-exp로 합치면 슬립 방향 d와 무관하다 (가정 1–4).
확인하는 것
  T1  가정 1–4 아래 : 방향 200개에서 합친 log-mel의 최대 편차 (기대 ~1e-13)와 축별 log-mel의 방향 의존성
  T2  가정 4(단일 진동원) 위반: 슬립 진동원 방향 d만 돌리고 방해 진동원 방향 e를 고정하면 편차가 생기는가
  T3  센서 좌표계 전체를 돌림(진동 벡터장을 강체 회전): 두 진동원이어도 불변인가
  T4  분석 창: 16 ms(128 샘플, 8 kHz) 창, 프레임 이동 1/240 s (33.3 샘플)에서 인접 프레임 겹침
합성 신호이며 TacV5 데이터가 아니다. 논문의 성능 주장(정확도, 지연)을 검증하는 것이 아니라 식 변형과 가정의 범위를 확인한다.

실행: python3 analysis/toy_lse_invariance.py  → analysis/toy_lse_invariance_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_lse_invariance_results.json"
FS, WIN, F = 8000, 128, 16
rng = np.random.default_rng(0)


def mel_fb(n_fft=WIN, n_mel=F, fs=FS):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(10), hz2mel(fs / 2), n_mel + 2))
    bins = np.fft.rfftfreq(n_fft, 1 / fs)
    fb = np.zeros((n_mel, len(bins)))
    for m in range(n_mel):
        lo, c, hi = pts[m], pts[m + 1], pts[m + 2]
        fb[m] = np.clip(np.minimum((bins - lo) / (c - lo), (hi - bins) / (hi - c)), 0, None)
    return fb


FB = mel_fb()
HANN = np.hanning(WIN)


def log_mel_frames(x, hop=33):
    """x: (3, N) → 축별 로그 멜 (3, frames, F). 창 128 샘플(16 ms), 이동 약 33 샘플(1/240 s)."""
    out = []
    for k in range(x.shape[0]):
        fr = []
        for s in range(0, x.shape[1] - WIN, hop):
            seg = x[k, s:s + WIN] * HANN
            p = np.abs(np.fft.rfft(seg)) ** 2
            fr.append(np.log(FB @ p + 1e-12))
        out.append(fr)
    return np.array(out)


def lse(M):
    m = M.max(axis=0)
    return m + np.log(np.exp(M - m).sum(axis=0))


def unit(v):
    return v / np.linalg.norm(v)


N = 2400  # 0.3 s
t = np.arange(N) / FS
s = rng.normal(size=N) * (1 + 0.5 * np.sin(2 * np.pi * 7 * t))          # 슬립 마찰 진동(광대역)
ev = rng.normal(size=N) * 0.5                                          # 방해 진동(다른 원천)
RES = {}

# T1
devs, ax0 = [], []
ref = None
for _ in range(200):
    d = unit(rng.normal(size=3))
    M = log_mel_frames(np.outer(d, s))
    L = lse(M)
    ref = L if ref is None else ref
    devs.append(np.abs(L - ref).max())
    ax0.append(M[0])
RES["T1_single_source"] = {"max_abs_dev_of_LSE_across_200_directions": float(f"{max(devs):.2e}"),
                         "std_across_directions_of_x_axis_logmel_ln_mean_over_frames_bins": round(float(np.std(np.array(ax0), axis=0).mean()), 3),
                         "note": "축별 log-mel은 방향에 따라 변하고(ln 단위 표준편차), 합치면 편차는 반올림 수준"}

# T2: 슬립원 방향 d만 회전, 방해원 방향 e 고정
e = unit(rng.normal(size=3))
base = unit(np.array([1.0, 0, 0]))
ref = lse(log_mel_frames(np.outer(base, s) + np.outer(e, ev)))
devs2 = []
for ang in np.linspace(0, np.pi, 19):
    d = np.array([np.cos(ang), np.sin(ang), 0.0])
    L = lse(log_mel_frames(np.outer(d, s) + np.outer(e, ev)))
    devs2.append(float(np.abs(L - ref).max()))
RES["T2_two_sources_rotate_one"] = {"max_abs_dev_ln": round(max(devs2), 3),
                                  "note": "두 원천이 독립이면 파워 합의 교차항 기대값은 0이지만 짧은 창(16 ms)의 표본 교차항이 남아 편차가 생김"}

# T3: 센서 좌표계 전체(두 원천 모두)를 같은 직교행렬로 회전
x0 = np.outer(base, s) + np.outer(e, ev)
ref = lse(log_mel_frames(x0))
devs3 = []
for _ in range(50):
    Q, _r = np.linalg.qr(rng.normal(size=(3, 3)))
    devs3.append(float(np.abs(lse(log_mel_frames(Q @ x0)) - ref).max()))
RES["T3_rigid_rotation_two_sources"] = {"max_abs_dev_ln": float(f"{max(devs3):.2e}"),
                                      "note": "축별 파워의 합 = 3-벡터 노름의 제곱이라 직교 회전에 불변. 단일 원천 가정(가정 4)이 없어도 성립"}

RES["T4_window"] = {"win_samples": WIN, "hop_samples": FS / 240, "overlap_pct_between_consecutive_frames": round(100 * (1 - (FS / 240) / WIN), 1),
                  "frames_spanned_by_one_window": round(WIN / (FS / 240), 2)}

OUT.write_text(json.dumps(RES, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(RES, indent=1, ensure_ascii=False))
