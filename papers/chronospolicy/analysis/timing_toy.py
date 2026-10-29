"""time-aware RoPE와 시간 무작위화의 기하 검증 (numpy, CPU 수 초).

1. RoPE 성질: 회전 후 내적이 시간 차이에만 의존(Eq. 6), 시각 원점 이동 불변.
2. 주기 0.1–2 s 로그 간격 16쌍(부록 A.2)의 시간 커널 s(Δt) = mean_k cos(ω_k Δt): 해상도와 되풀이.
3. 학습(부록 B.2)과 평가 조건(Table 2)에서 토큰 상대 시각 분포를 시뮬레이션해,
   평가 때 나오는 시간 차이가 학습 범위를 얼마나 벗어나는지 계산.
4. 시간 정보 없는 정책이 두 관측 차이로 속도를 추정할 때의 배율 오차(Conveyor 직관).

가정(논문에 없는 부분): 스트림마다 위상이 무작위인 균일 주기 샘플링, 요청 시각에 도착한(capture + delay ≤ t)
측정 중 최신 n개 사용, nominal·관측 조건의 행동 시작 지연 dt = 0, 추론 조건 dt ~ U(0, 상한).
실행: python3 analysis/timing_toy.py → analysis/timing_toy_results.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)
res = {}

# ---------------------------------------------------------------- 1. RoPE 성질
PERIODS = np.geomspace(0.1, 2.0, 16)  # s, 부록 A.2
OMEGA = 2 * np.pi / PERIODS


def rope(x, t):
    """x: (..., 32) = 16쌍, t: (...,) 초"""
    x = x.reshape(*x.shape[:-1], 16, 2)
    c, s = np.cos(OMEGA * t[..., None]), np.sin(OMEGA * t[..., None])
    return np.stack([x[..., 0] * c - x[..., 1] * s, x[..., 0] * s + x[..., 1] * c], -1).reshape(*x.shape[:-2], 32)


q, k = rng.normal(size=(2, 1000, 32))
ti, tj = rng.uniform(-2, 1, size=(2, 1000))
lhs = np.sum(rope(q, ti) * rope(k, tj), -1)
rhs = np.sum(q * rope(k, tj - ti), -1)
shift = rng.uniform(-100, 100, 1000)
lhs2 = np.sum(rope(q, ti + shift) * rope(k, tj + shift), -1)
res["rope_relative_max_err"] = float(np.abs(lhs - rhs).max())
res["rope_shift_invariance_max_err"] = float(np.abs(lhs - lhs2).max())

# ---------------------------------------------------------------- 2. 시간 커널
dt_grid = np.linspace(-4, 4, 8001)
kern = np.cos(OMEGA[None] * dt_grid[:, None]).mean(1)
pos = dt_grid >= 0
first_half = float(dt_grid[pos][np.argmax(kern[pos] < 0.5)])
far = (np.abs(dt_grid) >= 0.5)
res["kernel"] = {"periods_s": PERIODS.round(4).tolist(),
                 "halfwidth_s": first_half,                       # s(Δt)가 0.5로 떨어지는 Δt
                 "max_abs_beyond_0.5s": float(np.abs(kern[far]).max()),
                 "slowest_component_wraps_at_s": float(PERIODS.max())}
np.savez(ROOT / "analysis" / "timing_kernel.npz", dt=dt_grid, kern=kern)

# ---------------------------------------------------------------- 3. 학습 vs 평가 시간 차이
ACT_DT, H = 0.05, 16            # 20 Hz, 16개
N_HIST = {"image": 2, "pose": 2, "ft": 32}  # Conveyor·표준 과제는 image·pose 2개, Pivoting은 pose 3 + F/T 32


def stream_times(rate_lo, rate_hi, delay_lo, delay_hi, n, size):
    """요청 시각 0 기준 최신 n개 측정의 상대 캡처 시각(음수)."""
    rate = rng.uniform(rate_lo, rate_hi, size)
    delay = rng.uniform(delay_lo, delay_hi, size)
    period = 1 / rate
    # 도착 가능한 최신 캡처 시각: ≤ -delay 인 격자점, 격자 위상은 무작위
    phase = rng.uniform(0, 1, size) * period
    latest = -delay - phase
    return latest[:, None] - period[:, None] * np.arange(n)[None]


def action_times(dt_mode, size):
    if dt_mode == "train":
        dt = np.where(rng.uniform(size=size) < 0.5, 0.0, rng.uniform(0, 0.1, size))
    elif dt_mode == 0:
        dt = np.zeros(size)
    else:
        dt = rng.uniform(0, dt_mode, size)
    return dt[:, None] + ACT_DT * np.arange(H)[None]


# (image/pose 주기 Hz, 지연 s, F/T Hz, F/T 지연 s, 행동 dt)
CONDITIONS = {
    "train":    ((12.5, 33.33), (0, 0.05), (1, 100), (0, 0.1), "train"),
    "nominal":  ((20, 20), (0, 0), (100, 100), (0, 0), 0),
    "small":    ((12.5, 33.33), (0, 0.05), (75, 100), (0, 0.05), 0),
    "large":    ((1, 50), (0, 0.1), (1, 100), (0, 0.1), 0),
    "delay200": ((20, 20), (0, 0), (100, 100), (0, 0), 0.2),
    "combined": ((1, 50), (0, 0.1), (1, 100), (0, 0.1), 0.2),
}
N = 200_000
summ, samples = {}, {}
for name, (rr, dd, fr, fd, adt) in CONDITIONS.items():
    img = stream_times(*rr, *dd, N_HIST["image"], N)
    pose = stream_times(*rr, *dd, N_HIST["pose"], N)
    ft = stream_times(*fr, *fd, N_HIST["ft"], N)
    act = action_times(adt, N)
    oldest_img = -img.min(1)
    span_img_act = act.max(1) - img.min(1)            # 가장 오래된 이미지 ~ 마지막 행동
    summ[name] = {
        "oldest_image_age_s": np.percentile(oldest_img, [50, 99, 100]).round(3).tolist(),
        "image_gap_s": np.percentile(img[:, 0] - img[:, 1], [1, 50, 99]).round(3).tolist(),
        "span_image_to_last_action_s": np.percentile(span_img_act, [50, 99, 100]).round(3).tolist(),
        "frac_span_over_2s": float(np.mean(span_img_act > PERIODS.max())),
        "ft_history_span_s": np.percentile(ft[:, 0] - ft[:, -1], [1, 50, 99]).round(2).tolist(),
    }
    samples[name] = {"oldest_img": oldest_img, "gap": img[:, 0] - img[:, 1], "span": span_img_act}
tr_gap_max = np.percentile(samples["train"]["gap"], 100)
tr_age_max = np.percentile(samples["train"]["oldest_img"], 100)
for name in CONDITIONS:
    s = samples[name]
    summ[name]["frac_image_gap_outside_train"] = float(np.mean(s["gap"] > tr_gap_max))
    summ[name]["frac_oldest_image_older_than_train_max"] = float(np.mean(s["oldest_img"] > tr_age_max))
res["coverage"] = summ
res["train_max"] = {"image_gap_s": float(tr_gap_max), "oldest_image_age_s": float(tr_age_max)}
np.savez(ROOT / "analysis" / "timing_samples.npz",
         **{f"{n}_{k}": v[:20000] for n, d in samples.items() for k, v in d.items()})

# ---------------------------------------------------------------- 4. 시간 없는 정책의 속도 추정 배율
# 학습 간격 50 ms를 가정한 정책이 실제 간격 g로 찍힌 두 관측을 보면 속도를 (g / 0.05)배로 착각한다.
ratio = {name: np.percentile(samples[name]["gap"] / 0.05, [1, 50, 99]).round(2).tolist() for name in ("small", "large")}
res["index_policy_velocity_scale"] = ratio
# 추론 지연 dt를 모르는 정책의 목표 위치 오차 = v·dt (벨트 속도는 논문에 없음 → 0.1 m/s 예시)
res["delay_position_error_cm_at_0.1mps"] = {"0-100ms_mean": 0.5, "0-200ms_mean": 1.0, "0-200ms_max": 2.0}

(ROOT / "analysis" / "timing_toy_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print("RoPE 상대성 오차 %.1e, 원점 이동 불변 오차 %.1e" % (res["rope_relative_max_err"], res["rope_shift_invariance_max_err"]))
print("커널 반값 폭 %.3f s, |Δt|≥0.5 s에서 |s| 최대 %.3f" % (first_half, res["kernel"]["max_abs_beyond_0.5s"]))
for n, s in summ.items():
    print(f"{n:9s} 최고령 이미지 p50/p99/max {s['oldest_image_age_s']}  이미지 간격 p1/50/99 {s['image_gap_s']}  "
          f"이미지→마지막 행동 {s['span_image_to_last_action_s']}  >2s {s['frac_span_over_2s']:.3f}  "
          f"학습 밖 간격 {s['frac_image_gap_outside_train']:.3f}  F/T 이력 {s['ft_history_span_s']}")
print("학습 최대: 이미지 간격 %.3f s, 최고령 %.3f s" % (tr_gap_max, tr_age_max))
print("시간 없는 정책의 속도 배율 p1/50/99:", ratio)
