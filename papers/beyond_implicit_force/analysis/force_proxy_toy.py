"""leader-follower 불일치와 전류 토크 대리값의 물리 toy (CPU, numpy, 단일 스레드). 실행: python3 analysis/force_proxy_toy.py

ALOHA follower는 위치 제어(Position P gain 800, I gain 0; aloha/aloha_scripts/robot_utils.py:174-175)로 명령을 따른다.
I gain이 0이면 하중이 있을 때 정상상태 오차가 남고, 접촉 중에는 그 오차가 곧 누르는 힘이 된다.
아래 수치(강성, 질량, 마찰)는 가정이다. 결론은 비(ratio)와 경향으로 읽는다.

T1 구동: leader 목표(Base ACT) vs follower 목표(ACT-o)를 위치 서보로 재생할 때 나오는 접촉력, 예측 오차에 대한 여유
T2 감지: 같은 위치에서 단단한 병과 무른 병을 한 프레임의 qpos로 가를 수 있는가, 토크로는?
T3 대리값 품질: tau = k_t * I에 기어 마찰(정지 마찰 띠)과 중력이 섞일 때 감지할 수 있는 최소 접촉력, 자세 특징으로 중력을 학습해 빼기
T4 불일치 = 힘: 준정적 접촉에서 delta = q_leader - q_follower가 접촉력에 비례하는지, 빠른 자유 운동이 섞이면?
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)

# ── 공통 가정 (말단 좌표, 아래 방향 +) ─────────────────────────────────────────
KP = 800.0        # 서보의 말단 등가 강성 [N/m] (가정. P gain 800 레지스터를 말단 강성으로 옮긴 값이 아님)
MG = 1.5          # 말단에 걸리는 유효 중력 하중 [N] (가정, 서보 I=0이라 보상되지 않음)
F_DEMO = 6.0      # 시연에서 사람이 낸 접촉력 [N]
F_MIN = 2.0       # 닦기 '줄무늬'가 남는 최소 힘 [N] (가정)


def contact_force(z_cmd, k_e, kp=KP, mg=MG):
    """위치 서보(P만) + 중력 + 표면(z=0, 강성 k_e)의 준정적 평형 접촉력. z_cmd>0이면 표면 아래를 명령."""
    z_free = z_cmd + mg / kp                       # 접촉이 없을 때의 평형(처짐 포함)
    z = np.where(z_free > 0, (kp * z_cmd + mg) / (kp + k_e), z_free)
    return np.where(z > 0, k_e * z, 0.0)


def t1_actuation():
    out = {}
    for k_e in (2e3, 2e4, 2e5):  # 무른 패드 / 지우개+보드 / 거의 강체
        # 시연: 사람이 F_DEMO를 내도록 leader를 표면 아래로 민다
        z_follow = F_DEMO / k_e                                   # 기록되는 follower 위치
        z_leader = (F_DEMO * (KP + k_e) / k_e - MG) / KP          # 그 힘을 내는 leader 명령
        f_leader = float(contact_force(z_leader, k_e))
        f_follow = float(contact_force(z_follow, k_e))
        margin_mm = (z_leader - z_follow) * 1e3
        # 정책 예측 오차 sigma로 재생할 때 F_MIN 이상을 유지할 확률
        sig = np.array([0.5, 1, 2, 4, 8]) * 1e-3
        succ, touch = {}, {}
        for name, z0 in (("leader_target", z_leader), ("follower_target", z_follow)):
            p, c = [], []
            for s in sig:
                f = contact_force(z0 + rng.normal(0, s, 20000), k_e)
                p.append(float((f >= F_MIN).mean()))
                c.append(float((f > 0.05).mean()))
            succ[name] = [round(x, 3) for x in p]
            touch[name] = [round(x, 3) for x in c]
        out[f"k_e={k_e:g}"] = {
            "z_leader_mm": round(z_leader * 1e3, 2), "z_follower_mm": round(z_follow * 1e3, 3),
            "margin_mm": round(margin_mm, 2), "force_replay_leader_N": round(f_leader, 2),
            "force_replay_follower_N": round(f_follow, 2), "follower_over_demo": round(f_follow / F_DEMO, 3),
            "sigma_mm": (sig * 1e3).tolist(), "P(F>=F_min)": succ, "P(contact)": touch,
            "gravity_sag_mm": round(MG / KP * 1e3, 2),
        }
    return out


def t2_sensing():
    """병 누르기: 같은 속도로 내려가며 무른(vented) 병은 계속 눌리고 단단한(locked) 병은 멈춘다."""
    dt, v = 0.02, 0.02                       # 50 Hz, 20 mm/s
    k_soft, k_stiff = 150.0, 3000.0          # [N/m] 가정
    t = np.arange(0, 1.5, dt)
    rows = []
    for label, k in (("soft", k_soft), ("stiff", k_stiff)):
        for _ in range(40):
            z_cmd = v * t + rng.normal(0, 1e-3)              # 시작 오프셋 잡음
            z = (KP * z_cmd) / (KP + k)                      # 준정적(접촉 시작 z=0)
            z = np.maximum(z, 0)
            tau = k * z + rng.normal(0, 0.15, z.shape)        # 접촉력 대리값 [N], 측정 잡음
            zq = z + rng.normal(0, 2e-4, z.shape)             # qpos 측정 잡음(0.2 mm)
            for i in range(len(t)):
                if z[i] > 1e-3:                               # 접촉 뒤 프레임만
                    rows.append((label == "stiff", zq[i], tau[i], zq[i] - zq[max(0, i - 5)]))
    R = np.array(rows, dtype=float)
    y, zq, tau, dz5 = R[:, 0].astype(bool), R[:, 1], R[:, 2], R[:, 3]

    def best_threshold_acc(x, yy):
        xs = np.unique(np.quantile(x, np.linspace(0, 1, 201)))
        best = 0.5
        for th in xs:
            for sgn in (1, -1):
                pred = (sgn * (x - th)) > 0
                acc = 0.5 * ((pred & yy).sum() / yy.sum() + (~pred & ~yy).sum() / (~yy).sum())
                best = max(best, acc)
        return round(float(best), 3)

    # 같은 위치 구간에서만 비교(위치 겹침 구간): 단단한 병이 멈춘 높이 근처
    z_stiff_max = float(np.quantile(zq[y], 0.95))
    m = zq <= z_stiff_max
    return {
        "frames_after_contact": int(len(R)), "overlap_frames": int(m.sum()),
        "overlap_z_max_mm": round(z_stiff_max * 1e3, 2),
        "balanced_acc_single_frame_qpos": best_threshold_acc(zq[m], y[m]),
        "balanced_acc_single_frame_tau": best_threshold_acc(tau[m], y[m]),
        "balanced_acc_qpos_history_5frames": best_threshold_acc(dz5[m], y[m]),
        "note": "ACT 입력은 현재 프레임 qpos와 이미지뿐(속도·이력 없음). 위치가 겹치는 구간에서 qpos 한 프레임은 강성을 못 가른다",
    }


def t3_proxy_quality():
    """tau_hat = k_t I = tau_ext + tau_g(q) + tau_fric + 잡음. 정지 상태에서 마찰 방향은 모른다(정지 마찰 띠 ±tau_c)."""
    L = 0.30                     # 접촉점까지 지렛대 [m] (가정)
    tau_c = 0.25                 # 기어 정지 마찰 [N m] (가정)
    sig_i = 0.05                 # 전류 잡음을 토크로 환산 [N m] (가정)
    res = {}
    for F in (0.5, 1, 2, 4, 8):
        n = 20000
        tau_ext = F * L
        fric = rng.uniform(-tau_c, tau_c, n)
        noise = rng.normal(0, sig_i, n)
        on = tau_ext + fric + noise
        off = fric + rng.normal(0, sig_i, n)
        th = np.quantile(off, 0.95)                    # 오경보 5% 문턱
        res[f"F={F}N"] = round(float((on > th).mean()), 3)
    # 중력: 자세 q에 따라 tau_g = A cos q. 자세 특징으로 선형 회귀하면 빼낼 수 있는가
    q = rng.uniform(-1.0, 1.0, 4000)
    tau_g = 2.0 * np.cos(q) + 0.6 * np.sin(2 * q)
    contact = rng.random(4000) < 0.5
    tau_ext = np.where(contact, rng.uniform(0.6, 2.4, 4000), 0.0)
    tau_hat = tau_ext + tau_g + rng.uniform(-tau_c, tau_c, 4000) + rng.normal(0, sig_i, 4000)
    X = np.stack([np.ones_like(q), np.cos(q), np.sin(q), np.cos(2 * q), np.sin(2 * q)], 1)
    w, *_ = np.linalg.lstsq(X[~contact], tau_hat[~contact], rcond=None)   # 비접촉 구간으로 중력 학습
    resid = tau_hat - X @ w
    tpr_raw = float((tau_hat[contact] > np.quantile(tau_hat[~contact], 0.95)).mean())  # 중력 보상 없이 같은 문턱 규칙
    th = np.quantile(resid[~contact], 0.95)
    tpr = float((resid[contact] > th).mean())
    return {"assumptions": {"lever_m": L, "static_friction_Nm": tau_c, "current_noise_Nm": sig_i},
            "detect_rate_at_5pct_false_alarm": res,
            "min_force_band_N": round(tau_c / L, 2),
            "gravity": {"raw_tau_TPR_at_5pct_FA": round(tpr_raw, 3),
                        "after_pose_regression_TPR_at_5pct_FA": round(tpr, 3)}}


def t4_series():
    """시연 궤적: 자유 이동(빠름) → 접촉 누르기(준정적). 서보는 1차 지연(대역폭 wc)으로 따라간다고 가정."""
    dt, wc, k_e = 0.02, 12.0, 2e4
    t = np.arange(0, 6, dt)
    # 0–2 s 공중 왕복(-60 ~ -20 mm), 2–3 s 표면으로 접근, 3–6 s 누르며 닦기(목표 10±4 mm 아래)
    z_L = np.where(t < 2, -0.04 - 0.02 * np.cos(np.pi * t),
                   np.where(t < 3, -0.06 + 0.07 * (t - 2), 0.01 + 0.004 * np.sin(2 * np.pi * (t - 3))))
    z_f = np.zeros_like(t)
    z_f[0] = z_L[0] + MG / KP
    F = np.zeros_like(t)
    for i in range(1, len(t)):
        target = float(z_L[i] + MG / KP)
        if target > 0:                                           # 접촉: 준정적 평형으로 바로 수렴(강성 커서 빠름)
            z_eq = (KP * z_L[i] + MG) / (KP + k_e)
            z_f[i] = z_f[i - 1] + (z_eq - z_f[i - 1]) * min(1.0, wc * dt * 3)
        else:
            z_f[i] = z_f[i - 1] + (target - z_f[i - 1]) * wc * dt
        F[i] = k_e * max(z_f[i], 0.0)
    return t, z_L, z_f, F


def t4_discrepancy_vs_force():
    t, z_L, z_f, F = t4_series()
    delta = z_L - z_f
    cont = F > 0.1

    def r2(x, y):
        if len(x) < 3:
            return float("nan")
        A = np.stack([x, np.ones_like(x)], 1)
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        e = y - A @ w
        return round(float(1 - e.var() / y.var()), 3)

    return {"R2_delta_vs_force_contact_only": r2(delta[cont], F[cont]),
            "R2_delta_vs_force_whole_episode": r2(delta, F),
            "delta_free_motion_max_mm": round(float(np.abs(delta[~cont]).max() * 1e3), 2),
            "delta_contact_median_mm": round(float(np.median(delta[cont]) * 1e3), 2),
            "slope_N_per_mm_contact": round(float(np.polyfit(delta[cont] * 1e3, F[cont], 1)[0]), 3),
            "note": "준정적 접촉에서는 delta ∝ F(기울기 ≈ K_p). 빠른 자유 운동의 추종 지연도 delta를 만들어 전체 구간에서는 힘과 섞인다"}


def main():
    out = {"assumptions": {"Kp_eef_N_per_m": KP, "mg_N": MG, "F_demo_N": F_DEMO, "F_min_N": F_MIN},
           "T1_actuation": t1_actuation(), "T2_sensing": t2_sensing(),
           "T3_proxy_quality": t3_proxy_quality(), "T4_discrepancy_vs_force": t4_discrepancy_vs_force()}
    path = ROOT / "analysis" / "force_proxy_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", path)


if __name__ == "__main__":
    main()
