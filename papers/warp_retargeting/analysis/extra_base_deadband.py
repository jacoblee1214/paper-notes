"""추가 검증 2: 몸통 여유가 기지 데드밴드 5 cm를 흡수할 수 있는 조건 (1차원 toy, 가정).

모형
- 손 목표 h(t): 사람 손목 궤적을 흉내 낸 부드러운 2차원 경로 (정지·진동, 진폭 A, 주파수 f, 20 s, dt = 5 ms).
  x: 전방, y: 측방. 두 축을 같은 모형으로 처리하고 결과는 합쳐 본다.
- 기지(lazy base filter, 03 §4.4와 같은 구조): 데드밴드 δ = 0.05 m. 편차 e = h − b 가 δ 안이면 스프링 힘 0,
  밖이면 δ를 뺀 나머지만 따라감: b'' = ω² · (e − sat(e, δ)) − 2ζω b',  ζ = 1, ω = 2π·1.5 rad/s (03 §4.4 값)
  해석 참고: 등속 목표(속도 v)를 따를 때 정상 편차는 δ + 2ζv/ω, 힘 한계 때문에 v > ωδ/(2ζ) ≈ 0.24 m/s 에서는 편차가 수렴하지 않는다.
- 몸통: 상체 기울기 θ_t 로 손을 L_t sin θ_t 만큼 옮긴다 (L_t = 0.45 m, 가정). 기울기 한계 |θ_t| ≤ θ_max.
  몸통 명령 = 남은 편차 h − b 를 한계 안에서 따름: 몸통 변위 d_t = clip(h − b, ±L_t sin θ_max)
- 손 오차 = h − b − d_t.  성공 = 시간의 비율 중 |손 오차| ≤ 2 cm (허용 오차 가정).
변수: θ_max ∈ {0, 2, 4, 6, 7, 8, 10, 15}°, 손 진폭 A ∈ {0.1, 0.2, 0.3} m, 주파수 f ∈ {0.2, 0.5, 0.8} Hz,
      무작위 경로 30개씩 (시드 고정). 정적 한계: asin(δ/L_t) = 6.4°.
기대: 정지 상태에서는 θ_max ≥ 6.4°면 흡수. 동적 상황에서 기지 응답이 데드밴드 밖으로 나가는 최대 편차를 본다.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import json
import time
from pathlib import Path

import numpy as np

OUT_JSON = Path(__file__).with_name("extra_base_deadband_results.json")
DELTA, OMEGA, ZETA, L_T = 0.05, 2 * np.pi * 1.5, 1.0, 0.45
TOL = 0.02
DT, T_END = 0.005, 20.0
THETAS_DEG = [0, 2, 4, 6, 7, 8, 10, 15]
AMPS = [0.1, 0.2, 0.3]
FREQS = [0.2, 0.5, 0.8]
N_PATHS = 30


def sat(x, d):
    return np.clip(x, -d, d)


def simulate_axis(h, theta_max_deg):
    """h (T,) 손 목표 1축. 반환: 기지 위치 b, 손 오차 e_hand (T,), 기지 편차 max."""
    n = len(h)
    lim = L_T * np.sin(np.radians(theta_max_deg))
    b = np.zeros(n)
    v = np.zeros(n)
    b[0] = h[0]
    for k in range(n - 1):
        e = h[k] - b[k]
        # 진짜 데드밴드: 편차가 δ 안이면 힘 0, 밖이면 δ를 뺀 나머지만 따라감
        acc = OMEGA ** 2 * (e - sat(e, DELTA)) - 2 * ZETA * OMEGA * v[k]
        v[k + 1] = v[k] + acc * DT
        b[k + 1] = b[k] + v[k + 1] * DT
    dev = h - b
    d_t = np.clip(dev, -lim, lim) if lim > 0 else np.zeros(n)
    e_hand = dev - d_t
    return b, e_hand, float(np.max(np.abs(dev)))


def path(rng, A, f):
    t = np.arange(0, T_END, DT)
    ph = rng.uniform(0, 2 * np.pi, 2)
    x = A * np.sin(2 * np.pi * f * t + ph[0]) + 0.5 * A * np.sin(2 * np.pi * 1.7 * f * t + ph[1])
    return x


def main():
    t0 = time.time()
    rng = np.random.default_rng(20261111)
    out = {"setup": {"deadband_m": DELTA, "omega_rad_s": round(OMEGA, 3), "zeta": ZETA, "L_t_m": L_T, "tol_m": TOL,
                     "dt_s": DT, "T_s": T_END, "paths_per_setting": N_PATHS, "seed": 20261111,
                     "static_threshold_deg": round(float(np.degrees(np.arcsin(DELTA / L_T))), 2),
                     "speed_limit_for_convergence_m_s": round(OMEGA * DELTA / (2 * ZETA), 3)}}
    # 등속 목표의 정상 편차 검산: δ + 2ζv/ω (v = 0.1, 0.2 m/s), 힘 한계 안(v < 0.24)
    ramp = {}
    for v in (0.05, 0.1, 0.2):
        t = np.arange(0, 6.0, DT)
        h = v * t
        b, _, _ = simulate_axis(h, 0)
        dev = h - b
        ramp[f"v={v}_m_s"] = {"steady_dev_cm_sim": round(100 * float(np.median(dev[t > 4])), 2),
                              "steady_dev_cm_theory_delta_plus_2zv_over_w": round(100 * (DELTA + 2 * ZETA * v / OMEGA), 2)}
    out["ramp_check"] = ramp
    grid = {}
    base_dev_max = {}
    for A in AMPS:
        for f in FREQS:
            paths = [path(rng, A, f) for _ in range(N_PATHS)]
            key = f"A={A}_f={f}"
            # 기지 편차 최댓값 (θ와 무관)
            devs = [simulate_axis(p, 0)[2] for p in paths]
            base_dev_max[key] = {"max_base_dev_cm_median": round(100 * float(np.median(devs)), 2),
                                 "max_base_dev_cm_p90": round(100 * float(np.percentile(devs, 90)), 2),
                                 "max_base_dev_cm_max": round(100 * float(np.max(devs)), 2)}
            row = {}
            for th in THETAS_DEG:
                fr = []
                for p in paths:
                    _, eh, _ = simulate_axis(p, th)
                    fr.append(float(np.mean(np.abs(eh) <= TOL)))
                row[f"theta_{th}deg_success_frac"] = round(float(np.mean(fr)), 4)
            grid[key] = row
    out["base_deviation"] = base_dev_max
    out["success_fraction_by_theta"] = grid
    # 요약: 모든 조건 평균 성공률과 최소 θ_max에서 평균이 0.99 이상인지
    summary = {}
    for th in THETAS_DEG:
        vals = [grid[k][f"theta_{th}deg_success_frac"] for k in grid]
        summary[f"theta_{th}deg"] = {"mean_over_settings": round(float(np.mean(vals)), 4),
                                     "min_over_settings": round(float(np.min(vals)), 4)}
    out["summary_by_theta"] = summary
    out["runtime_s"] = round(time.time() - t0, 2)
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
