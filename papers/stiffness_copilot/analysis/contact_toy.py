"""2D 임피던스 접촉 toy: 방향별(비등방) 강성 대 고정 저·고 강성 (CPU numpy, 1스레드, 수 초).

모델: 질량 1 kg 점 로봇, 평면 벽(법선 n), 벽은 k_env 스프링 + 쿨롱 마찰 μ.
조작자 목표 x_d(t) = (x0 + v t, −δ): 벽 안으로 δ만큼 과하게 누르고 접선 방향으로 v로 이동(조작 과오버슛).
제어: F = K (x_d − x) + D (ẋ_d − ẋ), D = 1.4 √K (논문 Sec. IV-C와 같은 감쇠 규칙, 축별).
K = R(θ̂) diag(k_n, k_t) R(θ̂)ᵀ, θ̂ = 벽 법선 각 + 추정 오차 e (법선 추정 오차 실험).

조작자는 1 s 동안 2 cm 위에서 −δ까지 천천히 내려온다(접근 충격 최소화). 지표:
- F_peak: 전 구간 최대 접촉력 (접근 과도 포함)
- F_ss: 2–3 s 구간 평균 접촉력 (누름 정상상태; 해석식 k_env·K_n·δ/(K_n+k_env))
- track: 2–3 s 접선 추적 오차 RMS (정상상태에서는 마찰 지연 μF/K_t가 지배)
출력: analysis/contact_toy_results.json
실행: python3 analysis/contact_toy.py
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "contact_toy_results.json"

M = 1.0          # 질량 (kg), 감쇠 d = 1.4√k 와 함께 쓰면 ζ = 0.7
K_ENV = 5000.0   # 벽 강성 (N/m)
MU = 0.4         # 쿨롱 마찰 계수
V = 0.05         # 접선 목표 속도 (m/s)
T_END = 3.0
DT = 1e-3
F_LIMIT = 20.0   # 가정한 안전 임계 (N). 논문에는 없는 값이며 피크 초과 비율 해석용


def rot(e):
    c, s = np.cos(e), np.sin(e)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)  # (B,2,2)


def simulate(kn, kt, delta, theta_err=0.0, seeds=5, noise=0.5e-3, mu=MU):
    """kn, kt: (C,) 강성 배열. 설정 C개 × 시드 seeds개를 한꺼번에 적분한다.
    반환: 설정별 지표 dict (시드 평균·표준편차)."""
    kn = np.repeat(np.asarray(kn, float), seeds)
    kt = np.repeat(np.asarray(kt, float), seeds)
    B = kn.size
    rng = np.random.default_rng(0)
    d_noise = rng.normal(0, noise, size=(B, int(T_END / DT) + 1))  # 조작 잡음 (목표 y)

    n_world = np.array([0.0, 1.0])
    # 벽 법선 기준 K: 법선 성분 kn, 접선 kt. 추정 법선이 e만큼 틀리면 축이 회전한다.
    R = rot(np.full(B, theta_err))
    Kd = np.zeros((B, 2, 2)); Kd[:, 0, 0] = kt; Kd[:, 1, 1] = kn   # 벽 좌표계: x=접선, y=법선
    Kmat = R @ Kd @ np.transpose(R, (0, 2, 1))
    Dd = np.zeros((B, 2, 2)); Dd[:, 0, 0] = 1.4 * np.sqrt(kt); Dd[:, 1, 1] = 1.4 * np.sqrt(kn)
    Dmat = R @ Dd @ np.transpose(R, (0, 2, 1))

    x = np.zeros((B, 2)); x[:, 1] = 0.02          # 벽 위 2 cm에서 시작
    vel = np.zeros((B, 2))
    fpeak = np.zeros(B)
    fss_acc = np.zeros(B); fss_n = 0
    err_t = []
    steps = int(T_END / DT)
    for i in range(steps):
        t = i * DT
        ramp = min(t / 1.0, 1.0)
        yd = 0.02 + (-delta - 0.02) * ramp
        xd = np.stack([np.full(B, t * V), np.full(B, yd) + d_noise[:, i]], -1)
        vd = np.stack([np.full(B, V), np.zeros(B)], -1)
        e = xd - x
        ed = vd - vel
        F = np.einsum("bij,bj->bi", Kmat, e) + np.einsum("bij,bj->bi", Dmat, ed)
        pen = np.minimum(0.0, x[:, 1])
        Fn = -K_ENV * pen                                  # 벽의 수직 반력 (≥0)
        Ff = -mu * Fn * np.tanh(vel[:, 0] / 0.01)          # 접선 마찰 (속도 반대)
        acc = np.zeros((B, 2))
        acc[:, 0] = (F[:, 0] + Ff) / M
        acc[:, 1] = (F[:, 1] + Fn) / M
        vel = vel + acc * DT
        x = x + vel * DT
        fpeak = np.maximum(fpeak, Fn)
        if t >= 2.0:
            fss_acc += Fn
            fss_n += 1
            err_t.append(x[:, 0] - xd[:, 0])
    fss = fss_acc / fss_n
    rms_t = np.sqrt((np.array(err_t) ** 2).mean(0))
    fpeak = fpeak.reshape(-1, seeds)
    fss = fss.reshape(-1, seeds)
    rms_t = rms_t.reshape(-1, seeds)
    return {
        "Fpeak_N_mean": fpeak.mean(1), "Fpeak_N_sd": fpeak.std(1, ddof=1),
        "Fss_N_mean": fss.mean(1), "Fss_N_sd": fss.std(1, ddof=1),
        "track_rms_mm_mean": 1000 * rms_t.mean(1), "track_rms_mm_sd": 1000 * rms_t.std(1, ddof=1),
        "p_peak_over_limit": (fpeak > F_LIMIT).mean(1),
    }


def main():
    seeds = 5
    res = {"params": {"m": M, "k_env": K_ENV, "mu": MU, "v": V, "T": T_END, "dt": DT,
                      "F_limit_N": F_LIMIT, "damping_rule": "d=1.4*sqrt(k)", "seeds": seeds}}

    # (A) 세 가지 대표 설정, 조작 과오버슛 δ = 1 cm
    named = {"fixed low (300/300)": (300, 300), "fixed high (3000/3000)": (3000, 3000),
             "directional (300 normal / 3000 tangent)": (300, 3000)}
    cfg = list(named.values())
    out = simulate([c[0] for c in cfg], [c[1] for c in cfg], delta=0.01, seeds=seeds)
    res["A_named_delta1cm"] = {
        name: {k: float(v[i]) for k, v in out.items()} for i, name in enumerate(named)
    }

    # (B) 과오버슛 δ 세 수준에서 같은 세 설정
    res["B_delta_sweep"] = {}
    for delta in (0.005, 0.01, 0.02):
        out = simulate([c[0] for c in cfg], [c[1] for c in cfg], delta=delta, seeds=seeds)
        res["B_delta_sweep"][f"{delta*100:.1f}cm"] = {
            name: {k: float(v[i]) for k, v in out.items() if k.endswith("mean")} for i, name in enumerate(named)
        }

    # (C) 파레토: 등방 K 스윕 vs 방향별(k_t=3000) k_n 스윕
    grid = np.array([300, 500, 800, 1200, 1800, 2400, 3000], float)
    iso = simulate(grid, grid, delta=0.01, seeds=seeds)
    dirn = simulate(grid, np.full_like(grid, 3000.0), delta=0.01, seeds=seeds)
    res["C_pareto"] = {
        "k_grid": grid.tolist(),
        "isotropic": {k: v.tolist() for k, v in iso.items()},
        "directional_kt3000": {k: v.tolist() for k, v in dirn.items()},
    }

    # (D) 법선 추정 오차: 방향별(300/3000) 설정의 θ̂ 오차 0–30°
    res["D_normal_error"] = {}
    for deg in (0, 10, 20, 30):
        out = simulate([300.0], [3000.0], delta=0.01, theta_err=np.deg2rad(deg), seeds=seeds)
        res["D_normal_error"][f"{deg}deg"] = {k: float(v[0]) for k, v in out.items() if k.endswith("mean")}

    # (E) 마찰 없음(μ=0)에서 법선 오차의 정상상태 해석식 검증
    # 접선 자유 → 법선 유효 강성 K_eff = det K / K_xx (Schur 보수), 힘 F = δ K_eff k/(K_eff + k),
    # 접선 오차 e_x = K_xy F / (K_xx K_eff)   (K = R diag(kn,kt) Rᵀ, 벽 좌표계에서 x=접선, y=법선)
    res["E_normal_error_analytic_mu0"] = {}
    for deg in (0, 10, 20, 30):
        e = np.deg2rad(deg)
        kn_, kt_ = 300.0, 3000.0
        Kxx = kn_ * np.sin(e) ** 2 + kt_ * np.cos(e) ** 2
        Kxy = (kt_ - kn_) * np.sin(e) * np.cos(e)
        Kyy = kn_ * np.cos(e) ** 2 + kt_ * np.sin(e) ** 2
        detK = kn_ * kt_
        Keff = detK / Kxx
        F_pred = 0.01 * Keff * K_ENV / (Keff + K_ENV)
        ex_pred = 1000 * Kxy * F_pred / (Kxx * Keff)
        out = simulate([kn_], [kt_], delta=0.01, theta_err=e, seeds=seeds, mu=0.0)
        res["E_normal_error_analytic_mu0"][f"{deg}deg"] = {
            "Kxx": Kxx, "Kxy": Kxy, "Kyy": Kyy, "Keff": Keff,
            "F_pred_N": F_pred, "F_sim_N": float(out["Fss_N_mean"][0]),
            "ex_pred_mm": ex_pred, "track_sim_mm": float(out["track_rms_mm_mean"][0]),
        }

    OUT.write_text(json.dumps(res, indent=1))
    print("A (δ=1 cm):")
    for name, v in res["A_named_delta1cm"].items():
        print(f"  {name:<42} Fpeak={v['Fpeak_N_mean']:6.2f} N  Fss={v['Fss_N_mean']:6.2f}±{v['Fss_N_sd']:.2f} N  track={v['track_rms_mm_mean']:5.2f}±{v['track_rms_mm_sd']:.2f} mm")
    print("B (δ sweep, Fmax / track):")
    for dk, d in res["B_delta_sweep"].items():
        print("  δ=", dk, {n[:12]: (round(v["Fss_N_mean"], 1), round(v["track_rms_mm_mean"], 1)) for n, v in d.items()})
    print("E (μ=0, analytic vs sim):")
    for k, v in res["E_normal_error_analytic_mu0"].items():
        print(f"  {k:<6} Keff={v['Keff']:.0f} F pred/sim={v['F_pred_N']:.2f}/{v['F_sim_N']:.2f} N  e_x pred/sim={v['ex_pred_mm']:.2f}/{v['track_sim_mm']:.2f} mm")
    print("D (normal error):")
    for k, v in res["D_normal_error"].items():
        print(f"  {k:<6} Fss={v['Fss_N_mean']:.2f} N track={v['track_rms_mm_mean']:.2f} mm")


if __name__ == "__main__":
    main()
