"""토이 A: 노이즈 분위수 임계값 대 표본 내 튜닝 임계값 (합성 데이터, 논문 값 아님).

설계 가정(논문에서 가져온 것이 아니라 가정이다):
  - 정지 상태 Sy 잡음 = s0 * |AR(1) 가우시안|, AR 계수 0.5 (프레임 간 상관). s0은 논문 soft 행의 P95 ~ 0.10이 되게 맞춤.
  - 슬립 피크 = 중앙값 0.45, 로그 표준편차 0.5인 로그정규. 논문 Table III의 최솟값 0.20 근처가 되도록 고른 분포.
  - 슬립 사건 하나는 3프레임(12 Hz에서 약 0.25 s) 동안 지속. 사건을 잡았다는 것은 그 구간 안에서 임계값을 넘었다는 뜻.
  - 드리프트 d: 시험 때 잡음 척도가 보정 때보다 d배 커짐(조명, 예압, 손가락 상태 변화).

비교하는 임계값 규칙 (모두 보정 데이터만 사용):
  R1 잡음 P99.9
  R2 "논문형": 보정 슬립 최솟값 - 0.0006 (잡음 상한이 이보다 작을 때만 유효). 표본 내 경계에 붙는 값
  R3 표본 내 최적화: 보정 데이터에서 TPR - FPR이 최대인 임계값 (격자 탐색)
  R4 저장소 중점: (P99.9 + 보정 슬립 최솟값) / 2

출력: analysis/toy_threshold_results.json
실행: OMP_NUM_THREADS=1 python3 analysis/toy_threshold.py   (numpy 단일 스레드, 수 초)
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("toy_threshold_results.json")
RNG = np.random.default_rng(20260523)
HZ = 12.0
N_CAL_STATIC = 6477       # 논문 Exp. 1 정지 보유 프레임 수
N_CAL_SLIP = 18           # 논문 Exp. 2 슬립 이벤트 수
N_TEST_STATIC = 20000     # 시험용 정지 프레임(연속 시계열, 오경보율 계산용)
N_TEST_SLIP = 2000        # 시험용 슬립 사건
REPS = 200                # 보정 데이터 반복 (규칙의 변동성)
AR = 0.5
S0 = 0.10 / 1.96          # half-normal P95 = 1.96 s0 -> P95 ~ 0.10
SLIP_MEDIAN, SLIP_LOGSD = 0.45, 0.5
EVENT_LEN = 3
DRIFTS = [1.0, 1.25, 1.5]


def noise_series(n, scale, rng):
    """AR(1) 가우시안을 절댓값으로 바꾼 정지 잡음 시계열 (길이 n)."""
    e = rng.standard_normal(n)
    z = np.empty(n)
    z[0] = e[0]
    c = np.sqrt(1 - AR * AR)
    for t in range(1, n):
        z[t] = AR * z[t - 1] + c * e[t]
    return scale * np.abs(z)


def slip_events(n_events, rng):
    """슬립 사건 n개의 피크 값 (각 사건 = EVENT_LEN 프레임의 최댓값으로 판정)."""
    return SLIP_MEDIAN * np.exp(SLIP_LOGSD * rng.standard_normal(n_events))


def event_frames(peaks, rng):
    """피크마다 EVENT_LEN 프레임. 첫 프레임만 피크, 나머지는 피크의 0.8배 (감쇠)."""
    frames = np.empty((len(peaks), EVENT_LEN))
    frames[:, 0] = peaks
    frames[:, 1:] = 0.8 * peaks[:, None]
    return frames


def rules(noise_cal, slip_cal_peaks):
    p999 = float(np.quantile(noise_cal, 0.999))
    slip_min = float(slip_cal_peaks.min())
    r2 = slip_min - 0.0006 if slip_min - 0.0006 > p999 else p999
    r4 = 0.5 * (p999 + slip_min)
    # R3: 보정 데이터에서 Youden 지수 최대화
    grid = np.linspace(0.02, 0.6, 581)
    fpr_g = (noise_cal[None, :] > grid[:, None]).mean(axis=1)
    tpr_g = (slip_cal_peaks[None, :] > grid[:, None]).mean(axis=1)
    r3 = float(grid[np.argmax(tpr_g - fpr_g)])
    return {"R1_noise_P99.9": p999, "R2_paper_style_min_slip": r2,
            "R3_in_sample_Youden": r3, "R4_midpoint": r4}


def evaluate(theta, static_scaled, slip_peaks):
    """정지 시계열의 오경보율(상승 에지/분)과 슬립 사건 검출률."""
    alarms = static_scaled > theta
    fpr_frame = float(alarms.mean())
    rising = int(np.sum(alarms[1:] & ~alarms[:-1]) + alarms[0])
    minutes = len(static_scaled) / HZ / 60.0
    frames = event_frames(slip_peaks, None)
    tpr_event = float((frames > theta).any(axis=1).mean())
    return {"theta": theta, "fpr_frame": fpr_frame, "false_alarm_per_min": rising / minutes,
            "tpr_event": tpr_event}


def main():
    names = ["R1_noise_P99.9", "R2_paper_style_min_slip", "R3_in_sample_Youden", "R4_midpoint"]
    summary = {d: {n: {"theta": [], "tpr": [], "fpr": [], "fa_per_min": []} for n in names} for d in DRIFTS}
    example = None

    for rep in range(REPS):
        noise_cal = noise_series(N_CAL_STATIC, S0, RNG)
        slip_cal = slip_events(N_CAL_SLIP, RNG)
        th = rules(noise_cal, slip_cal)
        for d in DRIFTS:
            static_test = noise_series(N_TEST_STATIC, S0 * d, RNG)
            slip_test = slip_events(N_TEST_SLIP, RNG)
            for n in names:
                ev = evaluate(th[n], static_test, slip_test)
                summary[d][n]["theta"].append(ev["theta"])
                summary[d][n]["tpr"].append(ev["tpr_event"])
                summary[d][n]["fpr"].append(ev["fpr_frame"])
                summary[d][n]["fa_per_min"].append(ev["false_alarm_per_min"])
        if rep == 0:
            example = th

    report = {}
    for d in DRIFTS:
        report[f"drift_{d}"] = {}
        for n in names:
            s = summary[d][n]
            report[f"drift_{d}"][n] = {
                "theta_median": float(np.median(s["theta"])),
                "theta_p05_p95": [float(np.quantile(s["theta"], 0.05)), float(np.quantile(s["theta"], 0.95))],
                "tpr_event_median": float(np.median(s["tpr"])),
                "tpr_event_p05": float(np.quantile(s["tpr"], 0.05)),
                "fpr_frame_median": float(np.median(s["fpr"])),
                "false_alarm_per_min_median": float(np.median(s["fa_per_min"])),
                "false_alarm_per_min_p95": float(np.quantile(s["fa_per_min"], 0.95)),
            }

    # 12 Hz 한 사이클 = 1/12 s. 오경보 한 번이 effort를 +300 올리는 것과 연결해 1분당 누적 증가량을 계산
    for d in DRIFTS:
        for n in names:
            fa = report[f"drift_{d}"][n]["false_alarm_per_min_median"]
            report[f"drift_{d}"][n]["effort_added_per_min_by_false_alarms"] = fa * 300.0

    out = {
        "assumptions": {"AR": AR, "s0": S0, "slip_median": SLIP_MEDIAN, "slip_logsd": SLIP_LOGSD,
                        "event_len_frames": EVENT_LEN, "hz": HZ, "n_cal_static": N_CAL_STATIC,
                        "n_cal_slip": N_CAL_SLIP, "n_test_static": N_TEST_STATIC,
                        "n_test_slip": N_TEST_SLIP, "reps": REPS,
                        "effort_step_per_false_alarm": 300,
                        "note": "합성 데이터. 논문의 값을 맞추지 않았다. 방향성(규칙 간 차이, 드리프트 효과)만 읽는다."},
        "example_rep0_thresholds": example,
        "results": report,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"example rep0 thresholds: {({k: round(v, 3) for k, v in example.items()})}")
    for d in DRIFTS:
        print(f"-- drift x{d}")
        for n in names:
            r = report[f"drift_{d}"][n]
            print(f"  {n:26s} theta={r['theta_median']:.3f} [{r['theta_p05_p95'][0]:.3f},{r['theta_p05_p95'][1]:.3f}]  "
                  f"TPR_ev={r['tpr_event_median']:.3f} (p05 {r['tpr_event_p05']:.3f})  "
                  f"FPR_frame={r['fpr_frame_median']:.5f}  FA/min={r['false_alarm_per_min_median']:.2f}")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
