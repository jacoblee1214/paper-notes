"""KineFuse 'frame-level은 같은데 chain에서 벌어진다'를 보는 drift toy (CPU, numpy, 단일 스레드). 실행: python3 analysis/drift_toy.py

논문: Table I(open-loop, 프레임 t의 추정이 t+1의 초기값), Fig. 6(149 스텝 drift), 본문 'per-frame 1.12배 → tracking 2배 → 조작 15배'.
이 toy는 논문의 메커니즘을 재현하지 않는다. 같은 per-frame 격차가 chain에서 커지는 조건을 보는 가정 모델이다.
모델: 1차원 회전 오차 e(도). 한 프레임 정제: e_out = (1 - kappa(|d|)) d + eps,  d = 가설 오차 (= 이전 e - 물체 회전 u_t).
- kappa(|d|) = kappa_in (|d| <= R, 학습 교란 범위 U[1,10]도), kappa_out (|d| > R)
- A: kappa_in = kappa_out = 0.55
- B: kappa_in = 0.496 → per-frame 오차 비 (1-0.496)/(1-0.55)≈1.12. kappa_out 은 0.55(같음) 또는 0(교정 없음)
A. frame-level: 학습과 같은 교란 분포에서 RMS 오차 (두 모델 비 ≈ 1.12)
B. open-loop chain: T=149, 초기 오차 U[1,10]도, u_t ~ N(0, sigma_u) 도/프레임. RMS(e_T)의 비
C. 해석용 해석해: 선형 chain e_t = rho e_{t-1} + w, 정상 분산 = var_w/(1-rho²). rho = 1-kappa.
   per-frame 이득 비 1.12를 고정하면 rho가 1에 가까울수록 chain 증폭이 커지고, rho>1이면 발산한다.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = 10.0          # 학습 교란 상한 (도)
SIG_EPS = 0.2     # 정제 출력 잡음 (도)
T_STEPS = 149     # Fig. 6 길이
N_SEQ = 4000
rng = np.random.default_rng(0)


def kappa_of(d, kin, kout):
    return np.where(np.abs(d) <= R, kin, kout)


def refine(d, kin, kout, rng):
    return (1.0 - kappa_of(d, kin, kout)) * d + SIG_EPS * rng.standard_normal(d.shape)


def part_a():
    n = 200_000
    d = rng.uniform(1.0, R, n) * rng.choice([-1, 1], n)
    out = {}
    rA = refine(d, 0.55, 0.55, rng)
    rB = refine(d, 0.496, 0.496, rng)
    out["frame_rms_A"] = round(float(np.sqrt(np.mean(rA ** 2))), 3)
    out["frame_rms_B"] = round(float(np.sqrt(np.mean(rB ** 2))), 3)
    out["frame_ratio_B_over_A"] = round(out["frame_rms_B"] / out["frame_rms_A"], 3)
    out["frame_rms_no_refine_baseline"] = round(float(np.sqrt(np.mean(d ** 2))), 3)
    return out


def chain(kin_a, kout_a, kin_b, kout_b, sig_u, rng):
    e0 = rng.uniform(1.0, R, N_SEQ) * rng.choice([-1, 1], N_SEQ)
    ea = e0.copy(); eb = e0.copy()
    for _ in range(T_STEPS):
        u = sig_u * rng.standard_normal(N_SEQ)
        ea = refine(ea - u, kin_a, kout_a, rng)
        eb = refine(eb - u, kin_b, kout_b, rng)
    rms_a = float(np.sqrt(np.mean(ea ** 2)))
    rms_b = float(np.sqrt(np.mean(eb ** 2)))
    return rms_a, rms_b, float(np.mean(np.abs(eb) > R)), float(np.mean(np.abs(ea) > R))


def part_b():
    out = {}
    for kout_b in (0.55, 0.0):
        rows = []
        for sig_u in (0.1, 0.5, 1.0, 2.0, 3.0):
            ra, rb, fb, fa = chain(0.55, 0.55, 0.496, kout_b, sig_u, rng)
            rows.append({"sigma_u_deg_per_frame": sig_u, "rms_A": round(ra, 3), "rms_B": round(rb, 3),
                         "ratio_B_over_A": round(rb / ra, 2), "frac_B_outside_R": round(fb, 3), "frac_A_outside_R": round(fa, 3)})
        out[f"kappa_out_B_{kout_b}"] = rows
    # 교정 없음(kappa=0) 기준: 누적 드리프트 sqrt(T)*sigma
    out["no_correction_rms_at_T"] = {str(s): round(float(s * np.sqrt(T_STEPS)), 2) for s in (0.1, 0.5, 1.0)}
    return out


def part_c():
    """per-frame 이득 비 (1-kB)/(1-kA)=1.12 를 고정하고 kA를 바꿔 chain 정상 표준편차 비 std_B/std_A를 본다.
    선형 chain e_t = rho e_{t-1} + w (var_w 같음): var = var_w/(1-rho^2), 따라서 std_B/std_A = sqrt((1-rhoA^2)/(1-rhoB^2)).
    """
    rows = []
    for kA in (0.9, 0.7, 0.55, 0.3, 0.2, 0.1, 0.05):
        rhoA = 1.0 - kA
        rhoB = 1.12 * rhoA
        if rhoB >= 1.0:
            rows.append({"kappa_A": kA, "rho_A": round(rhoA, 3), "rho_B": round(rhoB, 3), "chain_std_ratio_B_over_A": "발산 (rho_B>=1)"})
            continue
        ratio = np.sqrt((1 - rhoA ** 2) / (1 - rhoB ** 2))
        rows.append({"kappa_A": kA, "rho_A": round(rhoA, 3), "rho_B": round(rhoB, 3), "chain_std_ratio_B_over_A": round(float(ratio), 3)})
    return rows


def main():
    out = {"A_frame_level": part_a(), "B_open_loop_chain": part_b(), "C_analytic_gain_amplification": part_c()}
    path = ROOT / "analysis" / "drift_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
