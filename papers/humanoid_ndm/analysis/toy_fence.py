"""토이 T1: 1자유도 링크에서 탐색 방식별 흔들림과 파라미터 식별 비교.

검증하려는 것: (1) 같은 크기의 관절 편차(RMS)를 만드는 조건에서 기준 궤적 저주파 섭동(FENCE 식 1-2)이
행동 잡음보다 흔들림이 작은가(표 5의 방향), (2) 저주파 섭동만으로 부하 파라미터(중력 토크 G = m g l)와
관성 I가 식별되는가(저주파 탐색이 주는 정보의 한계).
가정: 1자유도 링크, 부하 1.5 kg, 팔 길이 0.3 m, PD 추종 (Kp = 40, Kd = 1.5), 제어 50 Hz,
엔코더 잡음 0.002 rad, 모든 값은 이 토이 설정이며 논문 값이 아니다.
"No Noise"는 기준 궤적을 그대로 추종하는 롤아웃. 시행마다 시드가 다르다.
실행: python3 toy_fence.py  (수 초)  -> toy_fence_results.json
"""
import json
import os

import numpy as np

DT = 0.02
SUB = 10
I0, B0, M, L, G0 = 0.05, 0.05, 1.5, 0.3, 9.81
I_TOT = I0 + M * L ** 2
G_TRUE = M * G0 * L
KP, KD, TAU_MAX = 40.0, 1.5, 10.0
ENC = 0.002
T_SEC, N_TRAJ = 10.0, 20
T = int(T_SEC / DT)
t_axis = np.arange(T) * DT
REF = 0.5 * np.sin(2 * np.pi * 0.25 * t_axis)  # 기준 궤적 (0.25 Hz)


def fence_ref(rng, sigma, kc_hz=1.0, ortho=False):
    """식 (1)-(2): 비정규화 DFT 기준. Z0=0, 0<k<=kc에 복소 가우시안, 켤레 대칭, 나머지 0."""
    kc = int(np.floor(kc_hz * T * DT))
    Z = np.zeros(T, dtype=complex)
    z = (rng.normal(size=kc) + 1j * rng.normal(size=kc)) * sigma / np.sqrt(2)
    Z[1:kc + 1] = z
    Z[T - kc:] = np.conj(z[::-1])
    Qt = np.fft.fft(REF)
    out = np.fft.ifft(Qt + Z).real
    return out


def rollout(ref, act_noise, rng, obs_noise=None):
    q, qd = ref[0], 0.0
    qs, taus = np.zeros(T), np.zeros(T)
    for t in range(T):
        qm = q + (obs_noise[t] if obs_noise is not None else 0.0)
        qdes = ref[t] + act_noise[t]
        tau = np.clip(KP * (qdes - qm) - KD * qd, -TAU_MAX, TAU_MAX)
        for _ in range(SUB):
            h = DT / SUB
            acc = (tau - B0 * qd - G_TRUE * np.sin(q)) / I_TOT
            qd += acc * h
            q += qd * h
        qs[t], taus[t] = q, tau
    return qs, taus


def metrics(qs, taus):
    s_q = float(np.mean(np.abs(np.diff(qs, 2))))  # 관절 흔들림: 2차 차분 평균
    s_tau = float(np.mean(np.abs(np.diff(taus))))  # 토크 변화율
    return s_q, s_tau


def identify(data):
    """tau = I qdd + b qd + G sin q 최소제곱. q는 엔코더 잡음 포함, 미분은 중앙차분."""
    X, y = [], []
    for qs, taus, rng in data:
        qn = qs + rng.normal(0, ENC, qs.shape)
        qd = (qn[2:] - qn[:-2]) / (2 * DT)
        qdd = (qn[2:] - 2 * qn[1:-1] + qn[:-2]) / DT ** 2
        X.append(np.stack([qdd, qd, np.sin(qn[1:-1])], 1))
        y.append(taus[1:-1])
    X, y = np.concatenate(X), np.concatenate(y)
    th, *_ = np.linalg.lstsq(X, y, rcond=None)
    return th


def run_method(name, gen, seed0):
    data, qdev, sq, st = [], [], [], []
    for i in range(N_TRAJ):
        rng = np.random.default_rng(seed0 + i)
        ref, an, on = gen(rng)
        qs, taus = rollout(ref, an, rng, on)
        data.append((qs, taus, np.random.default_rng(10_000 + seed0 + i)))
        a, b = metrics(qs, taus)
        sq.append(a)
        st.append(b)
    base_q, _ = rollout(REF, np.zeros(T), np.random.default_rng(0))
    for qs, _, _ in data:
        qdev.append(float(np.sqrt(np.mean((qs - base_q) ** 2))))
    th = identify(data)
    return {"rms_dev_rad": float(np.mean(qdev)), "s_q": float(np.mean(sq)), "s_tau": float(np.mean(st)),
            "I_hat": float(th[0]), "b_hat": float(th[1]), "G_hat": float(th[2]),
            "I_err_pct": float(100 * (th[0] - I_TOT) / I_TOT), "G_err_pct": float(100 * (th[2] - G_TRUE) / G_TRUE)}


def ou(rng, theta, sigma):
    x, out = 0.0, np.zeros(T)
    for t in range(T):
        x += -theta * x * DT + sigma * np.sqrt(DT) * rng.normal()
        out[t] = x
    return out


def tune(gen_factory, target, lo=1e-4, hi=10.0):
    """RMS 편차가 target이 되도록 이분 탐색 (시드 고정 5궤적)"""
    for _ in range(18):
        mid = np.sqrt(lo * hi)
        devs = []
        base_q, _ = rollout(REF, np.zeros(T), np.random.default_rng(0))
        for i in range(5):
            rng = np.random.default_rng(900 + i)
            ref, an, on = gen_factory(mid)(rng)
            qs, _ = rollout(ref, an, rng, on)
            devs.append(np.sqrt(np.mean((qs - base_q) ** 2)))
        if np.mean(devs) < target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


if __name__ == "__main__":
    target = 0.05  # 목표 RMS 편차 (rad, 약 2.9도)
    zero = np.zeros(T)
    facts = {
        "FENCE": lambda s: (lambda rng: (fence_ref(rng, s), zero, None)),
        "Action Gaussian": lambda s: (lambda rng: (REF, s * rng.normal(size=T), None)),
        "Action OU": lambda s: (lambda rng: (REF, ou(rng, 1.0, s), None)),
        "Obs Gaussian": lambda s: (lambda rng: (REF, zero, s * rng.normal(size=T))),
    }
    res = {"setting": {"target_rms_dev_rad": target, "n_traj": N_TRAJ, "T_sec": T_SEC,
                       "true_I": I_TOT, "true_G": G_TRUE, "enc_noise": ENC}}
    nn = run_method("No Noise", lambda rng: (REF, zero, None), 100)
    res["No Noise"] = nn
    for k, f in facts.items():
        s = tune(f, target)
        r = run_method(k, f(s), 100)
        r["noise_scale"] = s
        res[k] = r
    # 같은 FENCE에서 식(1)-(2) 비정규화 DFT에 σ를 1,3,6으로 줄 때의 RMS 편차 (T=10 s)
    sig = {}
    for s in (1, 3, 6):
        devs = []
        base_q, _ = rollout(REF, zero, np.random.default_rng(0))
        for i in range(5):
            rng = np.random.default_rng(i)
            qs, _ = rollout(fence_ref(rng, s), zero, rng)
            devs.append(float(np.sqrt(np.mean((qs - base_q) ** 2))))
        sig[str(s)] = float(np.mean(devs))
    res["fence_sigma_to_rms_dev_rad"] = sig
    for k, v in res.items():
        if isinstance(v, dict) and "s_q" in v:
            print(f"{k:16s} dev={v['rms_dev_rad']:.3f} s_q={v['s_q']:.2e} s_tau={v['s_tau']:.2e} I_err={v['I_err_pct']:+.1f}% G_err={v['G_err_pct']:+.1f}%")
    print(sig)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "toy_fence_results.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
