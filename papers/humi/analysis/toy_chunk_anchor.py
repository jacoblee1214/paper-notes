"""토이 V1: 행동 덩어리(chunk)의 기준을 "이전 예정 목표"로 잡을 때와 "실행된 자세"로 잡을 때의 차이.

논문 §3.3과 그림 4, §4.1.3(던지기 75% 대 40%)의 메커니즘만 따로 검사한다. 로봇도 학습도 없다.

- 저수준 제어기: 1차 지연 xdot = (u - x) / tau (tau는 추종 지연). 고수준 정책은 5 Hz로 2초짜리 덩어리를 낸다.
- 정책은 시연을 완벽히 모방한다고 가정한다. 덩어리는 anchor + (d(t+s) - d(t)) 꼴의 상대 궤적이다.
- A(예정 목표 기준): anchor = 지금 시점의 이전 목표 u(t).   B(실행 자세 기준): anchor = 측정된 x(t).
- 측정값: 던지기 최고 속도 보존율, 마지막 위치 부족분, 덩어리 경계에서 목표가 뒤로 튀는 크기.

해석해: B에서 덩어리 하나가 등속 v로 진행할 때 실제 진행량은 v*T*(1 - (tau/T)(1 - exp(-T/tau))).

실행: python3 analysis/toy_chunk_anchor.py   (numpy만, 1초 미만)
"""
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_chunk_anchor_results.json"
DT = 0.002          # 시뮬레이션 간격(s)
T_CHUNK = 0.2       # 고수준 5 Hz
HORIZON = 2.0       # 덩어리 길이(2 s, 10 waypoint)


def min_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def demo(t):
    """던지기 시연의 앞뒤 위치 d(t) [m]: 0.8초에 걸쳐 -0.15, 0.4초에 걸쳐 +0.75, 이후 유지."""
    d = -0.15 * min_jerk(t / 0.8)
    d = d + 0.75 * min_jerk((t - 0.8) / 0.4)
    return d


def simulate(tau, mode, t_end=3.0):
    n = int(t_end / DT)
    t = np.arange(n) * DT
    x = np.zeros(n)         # 실행 위치
    u = np.zeros(n)         # 명령 위치
    anchor = 0.0
    t_k = 0.0
    chunk_start_d = demo(0.0)
    jumps = []
    next_switch = 0.0
    u_prev = 0.0
    for i in range(n):
        if t[i] >= next_switch - 1e-12:
            # 새 덩어리 시작
            prev_u = u[i - 1] if i > 0 else 0.0
            anchor = prev_u if mode == "scheduled" else x[i - 1] if i > 0 else 0.0
            t_k = t[i]
            chunk_start_d = demo(t_k)
            jumps.append(anchor - prev_u)
            next_switch += T_CHUNK
        u[i] = anchor + demo(t[i]) - chunk_start_d
        xi = x[i - 1] if i > 0 else 0.0
        x[i] = xi + DT * (u[i] - xi) / tau
    return t, x, u, np.array(jumps[1:])


def analyze(tau):
    out = {}
    d = demo(np.arange(int(3.0 / DT)) * DT)
    v_demo = np.gradient(d, DT).max()
    for mode in ("scheduled", "executed"):
        t, x, u, jumps = simulate(tau, mode)
        v = np.gradient(x, DT)
        out[mode] = dict(
            peak_speed_ratio=float(v.max() / v_demo),
            final_position_deficit_m=float(d[-1] - x[-1]),
            mean_backward_jump_mm=float(1000 * -jumps[jumps < -1e-6].mean()) if (jumps < -1e-6).any() else 0.0,
            max_backward_jump_mm=float(1000 * -jumps.min()),
        )
    out["closed_form_progress_gain_B"] = 1 - (tau / T_CHUNK) * (1 - math.exp(-T_CHUNK / tau))
    return out


def main():
    res = {"setup": dict(dt=DT, chunk_period_s=T_CHUNK, demo_peak_speed_mps=float(np.gradient(demo(np.arange(1500) * DT), DT).max()),
                         note="1차 지연 추종기, 완벽한 고수준 모방, 1축 위치. 정책 오차·시각 피드백·동역학 제약 없음")}
    for tau in (0.02, 0.05, 0.1, 0.2):
        res[f"tau_{tau}"] = analyze(tau)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("wrote", OUT)
    for tau in (0.02, 0.05, 0.1, 0.2):
        r = res[f"tau_{tau}"]
        print(f"tau={tau:<5} A 최고속도 {r['scheduled']['peak_speed_ratio']:.2f}, B {r['executed']['peak_speed_ratio']:.2f}, "
              f"B 마지막 부족 {r['executed']['final_position_deficit_m']*100:.1f} cm, B 최대 뒤로 튐 {r['executed']['max_backward_jump_mm']:.0f} mm, "
              f"해석 이득 {r['closed_form_progress_gain_B']:.2f}")


if __name__ == "__main__":
    main()
