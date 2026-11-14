"""토이 1: 마찰(LuGre, 이력 의존) 1자유도 관절에서 외부 토크 추정의 이력 길이 효과.

목적: 논문의 핵심 가정(자유 공간 토크를 학습으로 빼면 외부 토크가 남는다)이
이력 의존 마찰 아래에서 어떻게 깨지는지, 그리고 이력 길이 H가 얼마나 도움이 되는지 본다.
모델: J q'' = u - F(q', z) - tau_ext,  u = Kp(q_cmd - q) - Kd q'  (측정 토크 tau_m = u)
     F: LuGre 마찰 (z의 이력 의존). 백래시는 넣지 않았다.
추정기:
  (M) 명목 모델: tau_f = J q''_hat + 명목 점성·쿨롱 마찰 (명목 쿨롱 마찰을 +20% 틀리게 둠)
  (L_H) 선형 리지 회귀: 최근 H 스텝의 [q, q', dq, tanh(q'/0.02), tanh(q'/0.1)]을 입력, 자유 운동만으로 학습
주의: 파라미터와 외부 토크 모양은 가정이다. 논문 값에 맞추지 않았다.
실행 시간: 약 10–20초 (numpy).
"""
import json
from pathlib import Path

import numpy as np

import sys
W_AMP = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0  # 관측되지 않는 피드포워드 토크 진폭 (Nm)
rng = np.random.default_rng(0)

dt = 0.01
T = 600.0
N = int(T / dt)
J = 0.05
Kp, Kd = 12.0, 0.6
sig0, sig1, sig2 = 50.0, 1.0, 0.2  # LuGre
Fc, Fs, vs = 0.3, 0.5, 0.1
enc_noise = 2e-4  # rad
tau_noise = 0.02  # Nm


def g_of(v):
    return Fc + (Fs - Fc) * np.exp(-(v / vs) ** 2)


# 관측되지 않는 피드포워드 w(t): 입력 특징에 들어가지 않는 부드러운 신호 (W_AMP=0이면 순수 PD)
w_ff = W_AMP * np.sin(2 * np.pi * 0.23 * np.arange(N) * 0.01 + 0.7) * (1 + 0.5 * np.sin(2 * np.pi * 0.05 * np.arange(N) * 0.01))
# 명령 궤적: 부드러운 사인 합
t = np.arange(N) * dt
q_cmd = sum(a * np.sin(2 * np.pi * f * t + p) for a, f, p in
            [(0.6, 0.11, 0.0), (0.3, 0.37, 1.0), (0.2, 0.83, 2.0)])
dq_cmd = np.gradient(q_cmd, dt)

# 외부 토크: 반사인 펄스, 무작위 시작
tau_ext = np.zeros(N)
contact_mask = np.zeros(N, dtype=bool)
starts = np.sort(rng.uniform(0, T - 2, size=int(0.05 * T)))
for s in starts:
    d = rng.uniform(0.3, 1.0)
    a = rng.uniform(0.5, 2.0)
    i0 = int(s / dt)
    i1 = min(N, int((s + d) / dt))
    if i1 > i0:
        tau_ext[i0:i1] += a * np.sin(np.pi * np.linspace(0, 1, i1 - i0))
        contact_mask[i0:i1] = True

# 시뮬레이션 (오일러 적분)
q = np.zeros(N)
dq = np.zeros(N)
ddq = np.zeros(N)
z = 0.0
tau_m = np.zeros(N)
F_true = np.zeros(N)
q[0] = q_cmd[0]
for k in range(N - 1):
    u = Kp * (q_cmd[k] - q[k]) - Kd * dq[k] + w_ff[k]
    g = g_of(dq[k])
    zdot = dq[k] - abs(dq[k]) / g * z if g > 0 else 0.0
    z = z + dt * zdot
    fric = sig0 * z + sig1 * (dq[k] - abs(dq[k]) / g * z) + sig2 * dq[k]
    ddq[k] = (u - fric - tau_ext[k]) / J
    tau_m[k] = u
    F_true[k] = fric
    dq[k + 1] = dq[k] + dt * ddq[k]
    q[k + 1] = q[k] + dt * dq[k]
ddq[-1] = ddq[-2]
tau_m[-1] = Kp * (q_cmd[-1] - q[-1]) - Kd * dq[-1] + w_ff[-1]
F_true[-1] = F_true[-2]

# 센서 잡음
q_obs = q + rng.normal(0, enc_noise, N)
dq_obs = np.gradient(q_obs, dt)
ddq_obs = np.gradient(dq_obs, dt)
tau_obs = tau_m + rng.normal(0, tau_noise, N)
dq_cmd_k = q_cmd - q_obs

# 명목 모델 (쿨롱 마찰 20% 틀림, 정지 마찰·이력 없음)
Fc_nom = Fc * 1.2
tau_hat_M = J * ddq_obs + sig2 * dq_obs + Fc_nom * np.tanh(dq_obs / 0.02)
ext_M = tau_obs - tau_hat_M


def feats(H):
    """시점 k의 특징: 최근 H 스텝의 [q, dq, dq_cmd_k, tanh 기저]를 이어 붙인다."""
    base = np.stack([q_obs, dq_obs, dq_cmd_k,
                     np.tanh(dq_obs / 0.02), np.tanh(dq_obs / 0.1)], axis=1)
    idx = np.arange(H - 1, N)
    X = np.concatenate([base[idx - j] for j in range(H)], axis=1)
    return X, idx


def fit_eval(H, lam=1e-3):
    X, idx = feats(H)
    free_tr = idx < int(300 / dt)
    free_te = (idx >= int(300 / dt)) & (idx < int(450 / dt)) & (~contact_mask[idx])
    cont_te = (idx >= int(450 / dt)) & contact_mask[idx]
    y = tau_obs[idx]
    mu, sd = X[free_tr].mean(0), X[free_tr].std(0) + 1e-9
    Xn = (X - mu) / sd
    Xn = np.concatenate([Xn, np.ones((len(Xn), 1))], axis=1)
    A = Xn[free_tr]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[free_tr])
    pred = Xn @ w
    ext_hat = y - pred  # tau_m - tau_free_hat
    ext_true = tau_ext[idx]
    res_extra = {"mean_abs_ext_hat_contact": float(np.mean(np.abs(ext_hat[cont_te])))}
    return {**res_extra, "free_rms_nm": float(np.sqrt(np.mean((ext_hat[free_te]) ** 2))),
        "contact_l1_nm": float(np.mean(np.abs(ext_hat[cont_te] - ext_true[cont_te]))),
        "contact_recall_at_0p25": float(np.mean(np.abs(ext_hat[cont_te]) > 0.25)),
        "false_alarm_rate_free": float(np.mean(np.abs(ext_hat[free_te]) > 0.25)),
    }


res = {"W_AMP": W_AMP, "params": dict(J=J, Kp=Kp, Kd=Kd, lugre=[sig0, sig1, sig2, Fc, Fs, vs],
                      enc_noise=enc_noise, tau_noise=tau_noise, T=T, dt=dt),
       "true_friction_rms_nm": float(np.sqrt(np.mean(F_true ** 2)))}

# (M) 명목 모델
m_idx = np.arange(int(450 / dt), N)
m_free = m_idx[~contact_mask[m_idx]]
m_cont = m_idx[contact_mask[m_idx]]
res["M_nominal"] = {
    "free_rms_nm": float(np.sqrt(np.mean(ext_M[m_free] ** 2))),
    "contact_l1_nm": float(np.mean(np.abs(ext_M[m_cont] - tau_ext[m_cont]))),
    "false_alarm_rate_free": float(np.mean(np.abs(ext_M[m_free]) > 0.25)),
}
# 명목 모델이 마찰을 정확히 알 때 (oracle friction) 비교
tau_hat_oracle = J * ddq_obs + F_true
res["M_oracle_friction_free_rms_nm"] = float(np.sqrt(np.mean((tau_obs - tau_hat_oracle)[m_free] ** 2)))

# (L_H) 이력 길이 비교
res["L_H"] = {}
for H in (1, 10, 50):
    res["L_H"][f"H{H}"] = fit_eval(H)
# 접촉 구간의 참값 크기와 H=50 추정치 크기 (식별 가능성 진단)
c_idx = np.where(contact_mask)[0]
c_idx = c_idx[c_idx >= int(450 / dt) + 50]
res["contact_abs_tau_ext_mean_nm"] = float(np.mean(np.abs(tau_ext[c_idx])))

OUT = Path(__file__).with_name(f"toy_residual_history_w{W_AMP:g}_results.json")
OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
print(json.dumps(res, indent=1, ensure_ascii=False))
