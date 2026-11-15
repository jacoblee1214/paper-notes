"""토이 T1: 두 2-링크 팔 사이의 토크 잔차 보정 (TAM 개념의 CPU 축소판, numpy 단독).

질문: 이상 모델(ideal)로 설계한 명령 토크 tau0를 질량·마찰·토크 이득이 다른 목표 팔에 그대로 주면
      관절 추종이 얼마나 벌어지나? 시뮬레이션에서 학습한 토크 잔차 모델(히스토리 입력)이 이를 얼마나 줄이나?
      보정 모델이 학습 분포 밖의 상태(속도, 위치)나 마찰에서 어떻게 깨지나?

설정 (논문 대응, 전부 단순화)
- 이상 팔: 평면 2-링크, m=1 kg, l=0.4 m, 막대 관성, 중력 있음. 제어기는 이상 모델 기반 계산토크 PD.
- 목표 팔: 질량 배율 s_m, 점성 b, 시그모이드 마찰 A(sigma(20 qd)-0.5), 토크 이득 alpha, 바이어스 beta.
- 교사 토크 (논문 식 1, 상수 4): 이상 모델의 응답 q0''을 목표 팔에서 내는 토크 tau*를 역동역학으로 계산.
  학습 목표는 dtau* = tau* - tau0 (잔차). 이 라벨은 시뮬레이션에서만 계산한다.
- 학습 입력: 현재 (q, qd, tau0) + 200 ms 히스토리 (10 ms 간격 20점의 q, qd, 적용 토크)
  + 선택적 e_tau = tau_a - Phi0(q, qd, qdd_hat, 0) (논문 식 e_tau의 이상 모델 역동역학 잔차).
- 모델: 2층 tanh MLP (numpy, Adam). 학습은 랜덤 ξ와 랜덤 기준 궤적의 자유 운동 롤아웃에서만 한다.
- 평가: 기준은 이상 팔이 같은 제어기로 따라간 궤적. 지표는 관절 위치 RMSE(deg), 콜드 스타트 0.2 s 뒤부터.
  학습 분포 밖 조건 5개: 속도 2배, 위치 이동, 마찰 2배·3배, 질량 1.3배.

실행: python3 analysis/toy_tam_2link.py   (단일 프로세스, 수 분 이내)
출력: analysis/toy_tam_2link_results.json
"""
import json
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_tam_2link_results.json"
DT = 1e-3
T_TRAIN = 1.2          # s, 학습 롤아웃 길이
T_TEST = 2.0           # s, 평가 롤아웃 길이
COLD = 0.2             # s, 히스토리가 찰 때까지 공칭 토크 통과(논문 0.5 s의 축소)
LAG_STEP = 10          # 10 ms 간격
LAGS = 20              # 200 ms 히스토리
L1 = L2 = 0.4
LC1 = LC2 = 0.2
M1 = M2 = 1.0
I1 = I2 = M1 * L1 ** 2 / 12
G = 9.81
KP, KD = 100.0, 20.0
TAU_CLIP = 30.0
K_SIG = 20.0
N_TRAIN_ROLL, N_VAL_ROLL, N_TEST_ROLL, N_DAGGER_ROLL = 300, 100, 120, 150
EPOCHS = 15

# 이상 팔 파라미터 (질량 배율 1.0, 마찰 없음)
IDEAL = dict(sm=1.0, b=np.zeros(2), A=np.zeros(2), alpha=np.ones(2), beta=np.zeros(2))


def inertia(q, sm):
    """M(q), C(q,qd) 행렬 성분, 중력 벡터. sm: 질량 배율(R,)."""
    sm = np.broadcast_to(np.asarray(sm, dtype=float), q.shape[:1])
    m1, m2 = M1 * sm, M2 * sm
    i1, i2 = I1 * sm, I2 * sm
    c2, s2 = np.cos(q[:, 1]), np.sin(q[:, 1])
    c1, c12 = np.cos(q[:, 0]), np.cos(q[:, 0] + q[:, 1])
    M11 = m1 * LC1 ** 2 + i1 + m2 * (L1 ** 2 + LC2 ** 2 + 2 * L1 * LC2 * c2) + i2
    M12 = m2 * (LC2 ** 2 + L1 * LC2 * c2) + i2
    M22 = m2 * LC2 ** 2 + i2
    M = np.stack([np.stack([M11, M12], -1), np.stack([M12, M22], -1)], -2)
    h = -m2 * L1 * LC2 * s2
    return M, h, (m1 * LC1 + m2 * L1) * G * c1 + m2 * LC2 * G * c12, m2 * LC2 * G * c12


def coriolis(q, qd, sm):
    """C(q, qd) qd. 표준 2-링크 형태 C = [[h qd2, h (qd1+qd2)], [-h qd1, 0]], h = -m2 l1 lc2 sin(q2)."""
    m2 = M2 * sm
    h = -m2 * L1 * LC2 * np.sin(q[:, 1])
    c = np.zeros_like(qd)
    c[:, 0] = h * (2 * qd[:, 0] * qd[:, 1] + qd[:, 1] ** 2)
    c[:, 1] = -h * qd[:, 0] ** 2
    return c


def friction(qd, P):
    sig = 1.0 / (1.0 + np.exp(-K_SIG * qd))
    return P["b"] * qd + P["A"] * (sig - 0.5)


def forward_acc(q, qd, tau, P):
    """목표(또는 이상) 팔의 순동역학: qdd = M^-1 (u - C qd - G), u = alpha tau + beta - fric."""
    M, _, g1, g2 = inertia(q, P["sm"])
    G_ = np.stack([g1, g2], -1)
    u = P["alpha"] * tau + P["beta"] - friction(qd, P)
    rhs = u - coriolis(q, qd, P["sm"]) - G_
    return np.linalg.solve(M, rhs[..., None])[..., 0]


def inverse_torque(q, qd, qdd_des, P):
    """Phi_xi: 원하는 가속도를 내는 명령 토크. tau = (M qdd + C qd + G + fric - beta) / alpha."""
    M, _, g1, g2 = inertia(q, P["sm"])
    G_ = np.stack([g1, g2], -1)
    rhs = np.einsum("rij,rj->ri", M, qdd_des) + coriolis(q, qd, P["sm"]) + G_ + friction(qd, P) - P["beta"]
    return rhs / P["alpha"]


def phi0(q, qd, qdd, ext=None):
    """이상 모델의 역동역학 (e_tau 계산용)."""
    return inverse_torque(q, qd, qdd, IDEAL)


def nominal_torque(q, qd, qref, qdref, qddref):
    """이상 모델 계산토크 PD. 측정 상태를 쓴다(논문 C(a, q_t, qd_t))."""
    e, ed = qref - q, qdref - qd
    qdd = qddref + KD * ed + KP * e
    return inverse_torque(q, qd, qdd, IDEAL)


def sample_params(rng, R, A_max=1.0, sm_rng=(0.9, 1.1), A_fixed=None, sm_fixed=None):
    sm = rng.uniform(*sm_rng, size=R) if sm_fixed is None else np.full(R, sm_fixed)
    A = rng.uniform(0.0, A_max, size=(R, 2)) if A_fixed is None else np.full((R, 2), A_fixed)
    return dict(sm=sm, b=rng.uniform(0.0, 0.5, size=(R, 2)), A=A,
                alpha=rng.uniform(0.97, 1.03, size=(R, 2)), beta=rng.uniform(-0.1, 0.1, size=(R, 2)))


def sample_refs(rng, R, center_rng=0.5, center_shift=(0.0, 0.0), amp_scale=1.0, freq_scale=1.0):
    """기준 궤적: 관절별 사인 합 (0.5–1.5 Hz). 반환: 함수 값, 속도, 가속도 (R,2) 배열을 시간 격자에 대해."""
    n = int(round((T_TRAIN + T_TEST) / DT)) + 2
    t = np.arange(n) * DT
    center = rng.uniform(-center_rng, center_rng, size=(R, 2)) + np.asarray(center_shift)[None, :]
    amp = rng.uniform(0.15, 0.3, size=(R, 2, 2)) * amp_scale
    freq = rng.uniform(0.5, 1.5, size=(R, 2, 2)) * freq_scale
    ph = rng.uniform(0, 2 * np.pi, size=(R, 2, 2))
    q = np.zeros((R, n, 2))
    qd = np.zeros((R, n, 2))
    qdd = np.zeros((R, n, 2))
    for k in range(2):
        w = 2 * np.pi * freq[:, :, k]
        arg = w[:, :, None] * t[None, None, :] + ph[:, :, k][:, :, None]
        q += (amp[:, :, k][:, :, None] * np.sin(arg)).transpose(0, 2, 1)
        qd += (amp[:, :, k][:, :, None] * w[:, :, None] * np.cos(arg)).transpose(0, 2, 1)
        qdd += (-amp[:, :, k][:, :, None] * (w ** 2)[:, :, None] * np.sin(arg)).transpose(0, 2, 1)
    q += center[:, None, :]
    return q, qd, qdd


def rollout(P, refs, T, tau_mode="direct", model=None, feat_flags=None, scale=1.0):
    """폐루프 롤아웃(벡터화). tau_mode: 'direct' | 'oracle' | 'learned'.
    반환: q (R,T,2), qd (R,T,2), tau0 (R,T,2), tau_applied (R,T,2)."""
    qref, qdref, qddref = refs
    R = qref.shape[0]
    n = T
    q = np.zeros((R, n + 1, 2))
    qd = np.zeros((R, n + 1, 2))
    tau_a = np.zeros((R, n, 2))
    tau0_all = np.zeros((R, n, 2))
    q[:, 0] = qref[:, 0]
    qd[:, 0] = qdref[:, 0]
    for t in range(n):
        qt, qdt = q[:, t], qd[:, t]
        tau0 = nominal_torque(qt, qdt, qref[:, t], qdref[:, t], qddref[:, t])
        tau0_all[:, t] = tau0
        dtau = np.zeros_like(tau0)
        if tau_mode == "oracle":  # 알고 있는 ξ로 매 시점 교사 보정 (히스토리 불필요, 콜드 스타트 없음)
            qdd0 = forward_acc(qt, qdt, tau0, IDEAL)
            dtau = inverse_torque(qt, qdt, qdd0, P) - tau0
        elif tau_mode == "learned" and t * DT >= COLD:
            x = features(t, q, qd, tau_a, tau0, feat_flags)
            dtau = scale * model.predict(x)
        tau = np.clip(tau0 + dtau, -TAU_CLIP, TAU_CLIP)
        tau_a[:, t] = tau
        qdd = forward_acc(qt, qdt, tau, P)
        qd[:, t + 1] = qdt + DT * qdd
        q[:, t + 1] = qt + DT * qd[:, t + 1]
    return q[:, :n], qd[:, :n], tau0_all, tau_a


def features(t, q, qd, tau_a, tau0, flags):
    """시점 t의 입력 (R, D). 적용 토크 히스토리는 t-1부터 본다(현재 토크는 아직 없음).
    flags: use_hist, use_etau."""
    feats = [q[:, t], qd[:, t], tau0]
    if flags["use_hist"]:
        idx = np.clip(t - LAG_STEP * np.arange(LAGS), 0, None)
        idx_tau = np.clip(t - 1 - LAG_STEP * np.arange(LAGS), 0, None)
        qh, qdh, tah = q[:, idx], qd[:, idx], tau_a[:, idx_tau]
        feats += [qh.reshape(len(q), -1), qdh.reshape(len(q), -1), tah.reshape(len(q), -1)]
        if flags["use_etau"]:
            idx2 = np.clip(idx - LAG_STEP, 0, None)
            qdd_hat = (qdh - qd[:, idx2]) / (LAG_STEP * DT)
            etau = tah - phi0(qh.reshape(-1, 2), qdh.reshape(-1, 2), qdd_hat.reshape(-1, 2)).reshape(len(q), LAGS, 2)
            feats += [etau.reshape(len(q), -1)]
    return np.concatenate(feats, axis=1)


class MLP:
    def __init__(self, din, dh, rng):
        self.W1 = rng.normal(0, np.sqrt(1 / din), (din, dh)); self.b1 = np.zeros(dh)
        self.W2 = rng.normal(0, np.sqrt(1 / dh), (dh, dh)); self.b2 = np.zeros(dh)
        self.W3 = np.zeros((dh, 2)); self.b3 = np.zeros(2)
        self.mu = None; self.sd = None; self.ysd = 1.0

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, x):
        h1 = np.tanh(x @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        return h1, h2, h2 @ self.W3 + self.b3

    def predict(self, x):
        xn = (x - self.mu) / self.sd
        return self.forward(xn)[2] * self.ysd

    def fit(self, X, Y, rng, epochs=20, lr=2e-3, bs=512):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.ysd = float(Y.std() + 1e-9)
        Xn = (X - self.mu) / self.sd
        Yn = Y / self.ysd
        ps = self.params()
        m = [np.zeros_like(p) for p in ps]
        v = [np.zeros_like(p) for p in ps]
        step = 0
        N = len(Xn)
        losses = []
        for ep in range(epochs):
            perm = rng.permutation(N)
            tot = 0.0
            for s in range(0, N, bs):
                idx = perm[s:s + bs]
                x, y = Xn[idx], Yn[idx]
                h1, h2, out = self.forward(x)
                err = out - y
                tot += float((err ** 2).sum())
                g_out = 2 * err / len(idx)
                gW3 = h2.T @ g_out; gb3 = g_out.sum(0)
                g_h2 = (g_out @ self.W3.T) * (1 - h2 ** 2)
                gW2 = h1.T @ g_h2; gb2 = g_h2.sum(0)
                g_h1 = (g_h2 @ self.W2.T) * (1 - h1 ** 2)
                gW1 = x.T @ g_h1; gb1 = g_h1.sum(0)
                grads = [gW1, gb1, gW2, gb2, gW3, gb3]
                step += 1
                for i, (p, g) in enumerate(zip(ps, grads)):
                    m[i] = 0.9 * m[i] + 0.1 * g
                    v[i] = 0.999 * v[i] + 0.001 * g * g
                    mh = m[i] / (1 - 0.9 ** step); vh = v[i] / (1 - 0.999 ** step)
                    p -= lr * mh / (np.sqrt(vh) + 1e-8)
            losses.append(tot / (N * 2))
        return losses


def build_dataset(rng, R, flags, n_stride=8, mode="direct", model=None):
    """학습 데이터: 랜덤 ξ, 랜덤 기준. mode='direct'면 보정 없이 롤아웃(논문 3.1), 'learned'면 현재 모델로 폐루프(DAgger, 3.3).
    질의 토크에 0.5 Nm 노이즈를 더해 증강한다."""
    nT = int(T_TRAIN / DT)
    P = sample_params(rng, R)
    refs = tuple(a[:, :nT] for a in sample_refs(rng, R))
    q, qd, tau0, tau_a = rollout(P, refs, nT, tau_mode=mode, model=model, feat_flags=flags)
    Xs, Ys = [], []
    for aug in (0.0, 0.5):  # 질의 토크 노이즈 0.5 Nm (논문 B.9 축소)
        tau0_q = tau0 + aug * rng.standard_normal(tau0.shape)
        for t in range(int(COLD / DT) + 1, nT, n_stride):
            qt, qdt = q[:, t], qd[:, t]
            qdd0 = forward_acc(qt, qdt, tau0_q[:, t], IDEAL)
            ystar = inverse_torque(qt, qdt, qdd0, P) - tau0_q[:, t]
            Xs.append(features(t, q, qd, tau_a, tau0_q[:, t], flags))
            Ys.append(ystar)
    return np.concatenate(Xs).astype(np.float64), np.concatenate(Ys).astype(np.float64)


def bootstrap_ci(x, rng, B=500):
    means = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(B)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_condition(name, rng, R, P_fn, ref_kw, methods):
    P = P_fn(rng, R)
    refs = sample_refs(rng, R, **ref_kw)
    refs = tuple(a[:, :int(T_TEST / DT)] for a in refs)
    q_ideal = rollout(IDEAL_PARAMS(R), refs, int(T_TEST / DT), tau_mode="direct")[0]
    out = {}
    tau0_direct = rollout(P, refs, int(T_TEST / DT), tau_mode="direct")[2]
    clip_frac = float(np.mean(np.abs(tau0_direct[:, int(COLD / DT):]) > TAU_CLIP))
    out["_nominal_torque_over_limit_fraction"] = round(clip_frac, 4)
    for mname, spec in methods.items():
        mode, model, flags = spec[0], spec[1], spec[2]
        scale = spec[3] if len(spec) > 3 else 1.0
        q = rollout(P, refs, int(T_TEST / DT), tau_mode=mode, model=model, feat_flags=flags, scale=scale)[0]
        err = np.degrees(q - q_ideal)[:, int(COLD / DT):]
        rmse = np.sqrt((err ** 2).mean(axis=(1, 2)))
        lo, hi = bootstrap_ci(rmse, rng)
        out[mname] = {"rmse_deg_mean": round(float(rmse.mean()), 3), "ci95": [round(lo, 3), round(hi, 3)]}
    return out


def IDEAL_PARAMS(R):
    return dict(sm=np.ones(R), b=np.zeros((R, 2)), A=np.zeros((R, 2)), alpha=np.ones((R, 2)), beta=np.zeros((R, 2)))


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    flags_full = dict(use_hist=True, use_etau=True)
    flags_noetau = dict(use_hist=True, use_etau=False)
    flags_nohist = dict(use_hist=False, use_etau=False)
    models = {}
    train_log = {}
    for key, flags in [("full", flags_full), ("noetau", flags_noetau), ("nohist", flags_nohist)]:
        X, Y = build_dataset(np.random.default_rng(1), N_TRAIN_ROLL, flags)
        if key == "full":
            X_full, Y_full = X, Y
        Xv, Yv = build_dataset(np.random.default_rng(11), N_VAL_ROLL, flags)   # 검증: 학습과 다른 ξ와 기준
        mdl = MLP(X.shape[1], 128, np.random.default_rng(2))
        losses = mdl.fit(X, Y, np.random.default_rng(3), epochs=EPOCHS)
        val_mse = float(np.mean((mdl.predict(Xv) - Yv) ** 2))
        train_mse = float(np.mean((mdl.predict(X[::5]) - Y[::5]) ** 2))
        var_y = float(np.var(Yv))
        models[key] = (mdl, flags)
        train_log[key] = {"n_samples": int(len(X)), "n_val_samples": int(len(Xv)), "n_features": int(X.shape[1]),
                          "train_loss_first": round(losses[0], 5), "train_loss_last": round(losses[-1], 5),
                          "train_mse_Nm2": float(f"{train_mse:.4g}"), "val_mse_Nm2": float(f"{val_mse:.4g}"),
                          "val_R2_vs_zero": round(1 - val_mse / var_y, 3)}
        print(f"trained {key}: N={len(X)} D={X.shape[1]} loss {losses[0]:.4f}->{losses[-1]:.4f} val_mse {val_mse:.4g} ({time.time()-t0:.0f}s)")

    # DAgger 1회 (논문 3.3 축소): 학습된 full 모델로 폐루프 히스토리를 모아 라벨을 붙여 데이터에 더하고 다시 학습한다.
    Xd, Yd = build_dataset(np.random.default_rng(21), N_DAGGER_ROLL, flags_full, mode="learned", model=models["full"][0])
    Xa, Ya = np.concatenate([X_full, Xd]), np.concatenate([Y_full, Yd])
    mdl_d = MLP(Xa.shape[1], 128, np.random.default_rng(2))
    losses_d = mdl_d.fit(Xa, Ya, np.random.default_rng(3), epochs=EPOCHS)
    Xv, Yv = build_dataset(np.random.default_rng(11), N_VAL_ROLL, flags_full)
    val_d = float(np.mean((mdl_d.predict(Xv) - Yv) ** 2))
    models["dagger"] = (mdl_d, flags_full)
    train_log["dagger"] = {"n_samples": int(len(Xa)), "n_dagger_samples": int(len(Xd)), "n_features": int(Xa.shape[1]),
                           "train_loss_first": round(losses_d[0], 5), "train_loss_last": round(losses_d[-1], 5),
                           "val_mse_Nm2_on_direct_val_set": float(f"{val_d:.4g}")}
    print(f"trained dagger: N={len(Xa)} val_mse {val_d:.4g} ({time.time()-t0:.0f}s)")

    def methods():
        return {
            "Direct": ("direct", None, None),
            "Oracle (known xi)": ("oracle", None, None),
            "Learned full (hist+e_tau)": ("learned", models["full"][0], models["full"][1]),
            "Learned no e_tau": ("learned", models["noetau"][0], models["noetau"][1]),
            "Learned no history": ("learned", models["nohist"][0], models["nohist"][1]),
            "Learned full + DAgger x1": ("learned", models["dagger"][0], models["dagger"][1]),
            "Diag: Learned full x0.5": ("learned", models["full"][0], models["full"][1], 0.5),
            "Diag: Learned no e_tau x0.5": ("learned", models["noetau"][0], models["noetau"][1], 0.5),
        }

    R = N_TEST_ROLL
    conditions = {
        "in_distribution": (lambda r, n: sample_params(r, n), dict()),
        "velocity_2x": (lambda r, n: sample_params(r, n), dict(freq_scale=2.0, amp_scale=1.5)),
        "position_shift": (lambda r, n: sample_params(r, n), dict(center_shift=(0.0, 1.1), center_rng=0.3)),
        "friction_2x_train_max": (lambda r, n: sample_params(r, n, A_fixed=2.0), dict()),
        "friction_3x_train_max": (lambda r, n: sample_params(r, n, A_fixed=3.0), dict()),
        "mass_1p3x": (lambda r, n: sample_params(r, n, sm_fixed=1.3), dict()),
    }
    results = {}
    for cname, (pfn, rkw) in conditions.items():
        results[cname] = run_condition(cname, np.random.default_rng(100 + len(results)), R, pfn, rkw, methods())
        line = ", ".join(f"{k}: {v['rmse_deg_mean']}" for k, v in results[cname].items() if isinstance(v, dict))
        print(f"{cname}: {line} ({time.time()-t0:.0f}s)")

    out = {
        "setup": dict(dt_s=DT, T_test_s=T_TEST, cold_start_s=COLD, history_lags=LAGS, lag_step_ms=LAG_STEP,
                      train_rollouts=N_TRAIN_ROLL, val_rollouts=N_VAL_ROLL, dagger_rollouts=N_DAGGER_ROLL, train_T_s=T_TRAIN, test_rollouts=R, hidden=128, epochs=EPOCHS,
                      ideal_mass_kg=M1, link_m_l=[L1, L2], kp=KP, kd=KD,
                      train_ranges=dict(sm=[0.9, 1.1], b=[0, 0.5], A=[0, 1.0], alpha=[0.97, 1.03], beta=[-0.1, 0.1]),
                      clip_note="_nominal_torque_over_limit_fraction: 직접 제어의 공칭 토크가 +-30 Nm를 넘는 시간 비율(콜드 스타트 이후)",
                      rmse_definition="joint position RMSE vs ideal-model rollout, deg, after cold start, per rollout mean; bootstrap 95% CI over rollouts"),
        "train": train_log,
        "results_rmse_deg": results,
        "runtime_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT, f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
