"""토이 B: 검출 지연과 채널 절제 (1차원 파지 모형, 합성 데이터, 논문 값 아님).

설계 가정(논문에 없는 것은 모두 가정이다):
  - 파지력 f: 목표 g = e/1000 + 1e-4*(p0 - p) (effort와 위치가 모두 힘에 기여). f는 시정수 0.08 s의 1차 지연으로 g를 따른다.
  - 필요한 힘: f_req(t) = L(t)/mu, mu = 0.8. L(t) = 0.85 + 틸트 진동(2 s에 걸쳐 시작) - 중량 감소 계단(물 빠짐). 10 s.
  - 미끄럼 낙하: f_req - f > 0.05가 0.3 s 이상 누적되면 실패(drop). 0.05는 슬립 시작 판정(onset)과 같은 허용 편차.
  - 변형: f > F_c가 0.1 s 이상이면 실패(crush). F_c = max(f_req) + W. W(여유 폭)는 안티슬립 1회 증분(0.3)의 배수 k = 0.5, 1, 2, 3으로 스윕한다. 논문은 여유 폭을 보고하지 않는다.
  - 촉각 Sy(관측) = 2 * max(0, f_req - f) + |N(0, 0.04)|. 즉 θs = 0.20은 힘 부족 약 0.1에서 걸린다(가정).
  - CoP 근사: 기준 L의 지수 평균(alpha 0.02). 릴리스는 기준 대비 중량 감소 > 0.1 이고 EMA Sy < θq = 0.10일 때.
  - 보호: EMA(0.3) 힘 > F_c - 0.1 이면 힘을 낮춘다(논문 Flim 대응은 미상이라 가정).
  - 증분: 저장소 코드 값(슬립 effort +300 / 위치 -80 µm, 릴리스 -200 / +60, 보호 -400 / +100)을 쓴다. effort 1000 = 힘 1 단위로 본다.
  - 감지 지연: 센서값은 한 제어 주기만큼 늦다(카메라·처리 지연 가정).

구성: A 슬립만, B 슬립+보호, C 슬립+릴리스, D 전체, F0 고정 e=1200, Fh 고정 e=3000 (피드백 없음).
검출 지연: D, W=0.2, 제어 12·25·50 Hz.

실행: OMP_NUM_THREADS=1 python3 analysis/toy_control.py  → analysis/toy_control_results.json (수 초)
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("toy_control_results.json")
SIM_HZ = 600.0            # 시뮬 적분 1/600 s. 12, 25, 50 Hz 제어 주기가 모두 정수 배
T_END = 10.0
N_EP = 400
MU = 0.8
TAU = 0.08
THETA_S, THETA_Q, THETA_C = 0.20, 0.10, 0.10
SY_NOISE = 0.04
P0 = 70000.0
PMIN = P0 - 3000.0
E_INIT, E_MAX = 1200.0, 5000.0
D_E_SLIP, D_P_SLIP = 300.0, 80.0
D_E_REL, D_P_REL = 200.0, 60.0
D_E_PROT, D_P_PROT = 400.0, 100.0
CRUSH_T, DROP_T = 0.1, 0.3
# 여유 폭 W는 "안티슬립 1회 증분(e +300 = 힘 0.3)"의 배수 k로 정한다. 논문은 여유 폭을 보고하지 않으므로 k를 스윕한다.
SLIP_STEP_F = D_E_SLIP / 1000.0   # 힘 단위 0.3
WIDTH_K = [0.5, 1.0, 2.0, 3.0]
WIDTHS = [round(k * SLIP_STEP_F, 3) for k in WIDTH_K]

CONFIGS = {
    "A_slip_only": dict(slip=True, rel=False, prot=False),
    "B_slip_protect": dict(slip=True, rel=False, prot=True),
    "C_slip_release": dict(slip=True, rel=True, prot=False),
    "D_full": dict(slip=True, rel=True, prot=True),
}


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def sample_episodes(rng, n):
    return dict(
        A=rng.uniform(0.10, 0.35, n), fr=rng.uniform(0.25, 0.60, n),
        phi=rng.uniform(0, 2 * np.pi, n), B=rng.uniform(0.10, 0.30, n),
        t_drop=rng.uniform(3.0, 7.0, n),
    )


def load_at(ep, t):
    ramp = np.minimum(1.0, t / 2.0)   # 틸트는 2 s에 걸쳐 시작(초기 파지가 충분하도록)
    base = 0.85 + ramp * ep["A"] * np.sin(2 * np.pi * ep["fr"] * t + ep["phi"])
    return base - ep["B"] * (t >= ep["t_drop"])


def run(cfg, W, hz, ep, rng, fixed_e=None):
    """모든 에피소드를 벡터로 한꺼번에 돌린다. cfg가 None이면 고정 effort."""
    n = len(ep["A"])
    dt = 1.0 / SIM_HZ
    steps = int(round(T_END * SIM_HZ))
    tick_every = int(round(SIM_HZ / hz))

    F_c = None
    ts = np.arange(steps + 1) * dt
    L_all = load_at(ep, ts[:, None])
    F_c = (L_all / MU).max(axis=0) + W
    F_lim = F_c - 0.1

    e = np.full(n, E_INIT if fixed_e is None else fixed_e, dtype=float)
    p = np.full(n, P0)
    f = e / 1000.0
    f_ema = f.copy()
    L_ref = L_all[0].copy()
    f_prev, L_prev = f.copy(), L_all[0].copy()      # 한 제어 주기 전 상태(센서 지연)
    slip_clock = np.zeros(n)
    crush_clock = np.zeros(n)
    drop = np.zeros(n, bool)
    crush = np.zeros(n, bool)
    onset = np.full(n, np.nan)
    detect = np.full(n, np.nan)
    deficit_at_detect = np.full(n, np.nan)
    e_max = e.copy()
    n_rel = np.zeros(n)
    rel_last = np.full(n, -1e9)
    drop_after_rel = np.zeros(n, bool)
    n_rel = np.zeros(n)
    decay = math.exp(-dt / TAU)

    for k in range(steps):
        t = ts[k]
        L = L_all[k]
        f_req = L / MU
        delta = np.maximum(0.0, f_req - f)
        onset = np.where(np.isnan(onset) & (delta > 0.05), t, onset)
        slip_clock = np.where(f_req - f > 0.05, slip_clock + dt, 0.0)   # 0.05 이상 부족이 누적되면 낙하
        drop |= slip_clock >= DROP_T
        crush_clock = np.where(f > F_c, crush_clock + dt, 0.0)
        crush |= crush_clock >= CRUSH_T

        if fixed_e is None and k % tick_every == 0:
            # 센서: 한 주기 전 상태로 측정 (지연 1 주기)
            sy_obs = 2.0 * np.maximum(0.0, L_prev / MU - f_prev) + np.abs(rng.normal(0, SY_NOISE, n))
            f_obs = f_prev + rng.normal(0, 0.02, n)
            L_obs = L_prev + rng.normal(0, 0.02, n)
            f_ema = 0.3 * f_obs + 0.7 * f_ema
            L_ref = 0.98 * L_ref + 0.02 * L_obs
            dcop = L_ref - L_obs
            slip_on = sy_obs > THETA_S
            quiet = sy_obs < THETA_Q
            if cfg["slip"]:
                newly = slip_on & ~np.isnan(onset) & np.isnan(detect)
                detect[newly] = t
                deficit_at_detect[newly] = delta[newly]
                e[slip_on] = np.minimum(e[slip_on] + D_E_SLIP, E_MAX)
                p[slip_on] = np.maximum(p[slip_on] - D_P_SLIP, PMIN)
            if cfg["rel"]:
                rel = (~slip_on) & (dcop > THETA_C) & quiet
                n_rel += rel
                rel_last = np.where(rel, t, rel_last)
                e[rel] = np.maximum(e[rel] - D_E_REL, E_INIT)
                p[rel] = np.minimum(p[rel] + D_P_REL, P0)
            if cfg["prot"]:
                pr = f_ema > F_lim
                e[pr] = np.maximum(e[pr] - D_E_PROT, E_INIT)
                p[pr] = np.minimum(p[pr] + D_P_PROT, P0)
            e_max = np.maximum(e_max, e)
            f_prev, L_prev = f.copy(), L.copy()

        if fixed_e is None:
            target = e / 1000.0 + 1e-4 * (P0 - p)
        else:
            target = np.full(n, fixed_e / 1000.0)
        f = target + (f - target) * decay

    return {"success": (~drop) & (~crush), "drop": drop, "crush": crush, "n_rel": n_rel,
            "e_max": e_max, "delay": detect - onset, "deficit": deficit_at_detect,
            "onset": onset}


def main():
    results = {
        "assumptions": {"SIM_HZ": SIM_HZ, "T_END": T_END, "N_EP": N_EP, "MU": MU, "TAU": TAU,
                        "THETA_S": THETA_S, "THETA_Q": THETA_Q, "THETA_C": THETA_C, "SY_NOISE": SY_NOISE,
                        "E_INIT": E_INIT, "E_MAX": E_MAX, "step_slip": [D_E_SLIP, D_P_SLIP],
                        "step_release": [D_E_REL, D_P_REL], "step_protect": [D_E_PROT, D_P_PROT],
                        "CRUSH_T": CRUSH_T, "DROP_T": DROP_T, "W_sweep": WIDTHS, "W_in_slip_steps": WIDTH_K,
                        "note": "합성 데이터. 논문 값을 맞추지 않았다. 구성 간 순서와 민감도만 읽는다."},
        "ablation": {}, "fixed": {}, "delay": {},
    }
    for k, W in zip(WIDTH_K, WIDTHS):
        ep = sample_episodes(np.random.default_rng(1000 + int(k * 10)), N_EP)
        results["ablation"][f"W={W}"], results["fixed"][f"W={W}"] = {}, {}
        for name, cfg in CONFIGS.items():
            out = run(cfg, W, 12.0, ep, np.random.default_rng(2000 + int(k * 10)))
            s = int(out["success"].sum())
            results["ablation"][f"W={W}"][name] = {
                "success": s, "n": N_EP, "rate": s / N_EP, "wilson95": wilson(s, N_EP),
                "drop_rate": float(out["drop"].mean()), "crush_rate": float(out["crush"].mean()),
                "e_max_median": float(np.median(out["e_max"])),
                "release_events_median": float(np.median(out["n_rel"])),
                "release_episode_share": float(np.mean(out["n_rel"] > 0)),
            }
        for label, fe in {"F0_fixed_1200": 1200.0, "Fh_fixed_3000": 3000.0}.items():
            out = run(None, W, 12.0, ep, np.random.default_rng(2000 + int(k * 10)), fixed_e=fe)
            s = int(out["success"].sum())
            results["fixed"][f"W={W}"][label] = {
                "success": s, "n": N_EP, "rate": s / N_EP, "wilson95": wilson(s, N_EP),
                "drop_rate": float(out["drop"].mean()), "crush_rate": float(out["crush"].mean()),
            }

    ep = sample_episodes(np.random.default_rng(1020), N_EP)
    for hz in (12.0, 25.0, 50.0):
        out = run(CONFIGS["D_full"], WIDTHS[2], hz, ep, np.random.default_rng(3000))
        det = out["delay"][~np.isnan(out["delay"])]
        n_onset = int(np.sum(~np.isnan(out["onset"])))
        results["delay"][f"{int(hz)}Hz"] = {
            "episodes_with_slip_onset": n_onset,
            "detected_after_onset": int(det.size),
            "delay_median_ms": float(np.median(det) * 1000) if det.size else None,
            "delay_p90_ms": float(np.quantile(det, 0.9) * 1000) if det.size else None,
            "deficit_at_detect_median": float(np.nanmedian(out["deficit"])) if det.size else None,
        }

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float))

    print("ablation success/N (W: A B C D | F0 Fh)")
    for W in WIDTHS:
        row = results["ablation"][f"W={W}"]
        fx = results["fixed"][f"W={W}"]
        print(f"  W={W}: " + " ".join(f"{v['success']}" for v in row.values()) + " | " +
              " ".join(f"{v['success']}" for v in fx.values()))
    print("delay:", json.dumps(results["delay"], ensure_ascii=False))
    print("saved:", OUT)


if __name__ == "__main__":
    main()
