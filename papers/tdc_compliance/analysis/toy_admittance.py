"""TDC 4.2절 관리 제어기(식 2, 3)의 1축 toy 세 가지. numpy만 쓰고 수 초 안에 끝난다.

가정(논문 밖): 어드미턴스 참조가 이상적으로 추종됨(로봇 위치 루프 지연 없음), 제어 주기 1 kHz(v1 Fig. 2 표기, v2 본문에는 없음),
환경은 선형 스프링(강성 ke, 위치 z = 0)이다. 질량 m = 1, ξ = 2, k_low = 50, k_high = 4 k_low는 논문 4.1절 값이다.

T1  힘 제어 축으로 벽에 접근할 때의 최대 힘과 정착 시간을 벽 강성 ke에 따라 본다 (논문: 크기는 접촉 동역학에 민감하나 수동 조정 가능).
T2  등방성 어드미턴스(L/M/H-A, F_cmd = 0)가 벽에 걸어 주는 정상 힘이 자세 오프셋 e에 얼마나 민감한지, TDC 힘 축과 비교한다.
   (v1 부록 G의 법선 동역학 m x'' + 2d x' = f_ext - f_H는 감쇠가 v2의 D보다 2배 크므로 그 변형도 함께 돌린다.)
T3  예측 과제 프레임이 어긋났을 때(각도 θ) 쌓인 오프셋 d가 강성 축으로 새는 힘과, 벽 법선 방향 힘의 정상값.

실행: python3 analysis/toy_admittance.py   → analysis/toy_admittance_results.json
"""
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("toy_admittance_results.json")
M, XI, K_LOW = 1.0, 2.0, 50.0
K_HIGH = 4 * K_LOW
D_FORCE = 2 * XI * math.sqrt(M * K_LOW)  # 힘 제어 축 감쇠 (논문 3.4절)
DT = 1e-3


def contact_run(ke, f_target=4.0, v0=None, z0=0.02, t_end=3.0, dt=DT, D=None):
    """힘 제어 축: M z'' + D z' = F_ext - F_cmd. 벽은 z < 0에서 F_ext = -ke z (위쪽 반력). F_cmd = +f_target.
    v0 None이면 자유 공간에서 정상 접근 속도(-f_target/D)로 시작한다."""
    D_FORCE_ = D_FORCE if D is None else D
    z = z0
    v = -f_target / D_FORCE_ if v0 is None else v0
    fs, t_contact = [], None
    for i in range(int(t_end / dt)):
        f_ext = -ke * z if z < 0 else 0.0
        a = (f_ext - f_target - D_FORCE_ * v) / M
        v += a * dt
        z += v * dt
        if f_ext > 0 and t_contact is None:
            t_contact = i * dt
        fs.append(f_ext)
    fs = np.array(fs)
    # 접촉 이후 정착: 목표의 ±10 % 안에 들어와 끝까지 머무는 시각
    i0 = int(t_contact / dt) if t_contact is not None else 0
    after = fs[i0:]
    out = np.where(np.abs(after - f_target) > 0.1 * f_target)[0]
    settle = (out[-1] + 1) * dt if len(out) else 0.0
    zeta = D_FORCE_ / (2 * math.sqrt(M * ke))
    return {"ke": ke, "peak_force": float(fs.max()), "peak_over_target": float(fs.max() / f_target),
            "settle_s": float(settle), "final_force": float(fs[-1]), "zeta": zeta}


def static_iso(k, ke, e):
    """등방성 어드미턴스, F_cmd = 0, 지령 위치가 표면 아래 e(m). 직렬 스프링: F = e / (1/k + 1/ke)."""
    return e / (1 / k + 1 / ke)


def main():
    R = {"params": {"m": M, "xi": XI, "k_low": K_LOW, "k_high": K_HIGH, "D_force_axis": D_FORCE,
                    "approach_speed_per_N": 1 / D_FORCE, "dt": DT}}

    # T1
    kes = [1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]
    R["T1_contact"] = {
        "target_4N_approach": [contact_run(ke, 4.0) for ke in kes],
        "target_2N_approach": [contact_run(ke, 2.0) for ke in kes],
        "target_4N_at_rest": [contact_run(ke, 4.0, v0=0.0, z0=0.0) for ke in kes],
        "target_4N_v1_damping_2x": [contact_run(ke, 4.0, D=2 * D_FORCE) for ke in kes],
        "target_4N_dt_0.5ms": [contact_run(ke, 4.0, dt=5e-4) for ke in (1e4, 1e5, 1e6)],
    }

    # T2
    es = [0.002, 0.005, 0.01, 0.02]
    ke = 1e4
    R["T2_static"] = {"ke": ke, "e_m": es,
                      "L-A_50": [static_iso(50, ke, e) for e in es],
                      "M-A_200": [static_iso(200, ke, e) for e in es],
                      "H-A_800": [static_iso(800, ke, e) for e in es],
                      "TDC_force_axis_4N": [4.0] * len(es),
                      "offset_needed_for_4N_m": {"L-A_50": 4 * (1 / 50 + 1 / ke), "M-A_200": 4 * (1 / 200 + 1 / ke),
                                                 "H-A_800": 4 * (1 / 800 + 1 / ke)},
                      "slope_N_per_cm": {"L-A_50": static_iso(50, ke, 0.01) / 1.0, "M-A_200": static_iso(200, ke, 0.01),
                                         "H-A_800": static_iso(800, ke, 0.01)}}
    R["T2_static"]["ke_sweep_H-A_800_at_e_5mm"] = {str(k): static_iso(800, k, 0.005) for k in (1e3, 1e4, 1e5)}
    R["T2_static"]["ke_sweep_TDC_4N"] = {str(k): 4.0 for k in (1e3, 1e4, 1e5)}

    # T3
    ds = [0.01, 0.02, 0.05, 0.10]
    ths = [5, 10, 20, 30]
    leak = {f"d={d*100:.0f}cm": {f"th={t}": K_HIGH * d * math.sin(math.radians(t)) for t in ths} for d in ds}
    R["T3_frame_leak_N"] = leak
    R["T3_normal_force_factor"] = {f"th={t}": 1 / math.cos(math.radians(t)) for t in ths}
    R["T3_sliding_angle_deg"] = {f"mu={mu}": math.degrees(math.atan(mu)) for mu in (0.2, 0.3, 0.5)}
    # 모드 전환 시 쌓인 오프셋이 스프링 힘으로 돌아오는 크기 (힘 제어 -> 위치 제어)
    R["T3_mode_switch_spike_N"] = {f"d={d*100:.0f}cm": {"k_low_50": K_LOW * d, "k_high_200": K_HIGH * d} for d in ds}

    OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False))
    print("wrote", OUT)
    print("D_force_axis=%.2f  approach speed per N = %.4f m/s" % (D_FORCE, 1 / D_FORCE))
    for key in ("target_4N_approach", "target_2N_approach", "target_4N_at_rest", "target_4N_v1_damping_2x", "target_4N_dt_0.5ms"):
        print(key)
        for r in R["T1_contact"][key]:
            print("  ke=%.0e peak=%.2f N (x%.2f) settle=%.3fs final=%.2f zeta=%.2f" %
                  (r["ke"], r["peak_force"], r["peak_over_target"], r["settle_s"], r["final_force"], r["zeta"]))
    print("T2", json.dumps(R["T2_static"], ensure_ascii=False)[:900])
    print("T3", json.dumps(R["T3_frame_leak_N"])[:500])


if __name__ == "__main__":
    main()
