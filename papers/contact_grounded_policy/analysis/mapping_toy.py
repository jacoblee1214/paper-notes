"""Contact-consistency mapping toy (CPU, numpy). 실행: python3 analysis/mapping_toy.py

논문 Fig. 2의 3-DoF 평면 손가락을 관절 PD로 제어하고, 원형 물체를 누르는 준정적 평형에서
(실제 상태 q, 촉각 u, 목표 상태 a) 세 쌍을 만든다. 그 위에서 다음을 확인한다.
- M1: 해석식 a = q − Kp⁻¹ (Jᵀf + J_gᵀF_g)가 모든 접촉을 감지하면 정확히 성립 (Eq. 1의 물리적 의미)
- M2: Table III과 같은 비교 (State+Tactile / State only / Tactile only × 잔차 / 절대), 3 seed, MLP
- M3: 손끝만 감지할 때 (Digit360처럼 손가락 끝에만 센서) 생기는 식별 불가능 오차
- M4: 학습 데이터에 손가락 속도 항(Kd q̇)이 섞일 때의 오차 하한 (x에 속도가 없음)
- M5: 촉각 잡음 강건성 (Fig. 13과 같은 방식: 데이터셋 전체 촉각 표준편차의 0–50%)
"""
import json
import math
import os
from pathlib import Path

# 작은 행렬에서는 BLAS 멀티스레드가 오히려 수십 배 느려서 단일 스레드로 고정한다
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
L = np.array([0.050, 0.035, 0.025])  # 링크 길이 (m)
MASS = np.array([0.030, 0.020, 0.012])  # 링크 질량 (kg)
KP = np.array([1.5, 1.0, 0.6])  # 관절 강성 (N·m/rad)
KD = np.array([0.03, 0.02, 0.012])  # 관절 감쇠 (N·m·s/rad)
N_TAX, R_PAD, K_C = 6, 0.006, 3000.0  # 링크당 taxel 수, 손바닥 쪽 패드 두께 (m), 접촉 강성 (N/m)
G = 9.81


def fk(q):
    """관절 위치 O_i, 링크 방향 d_i, 손바닥 법선 p_i. 굴곡(q > 0)이 아래(−y)로 말림."""
    th = -np.cumsum(q)
    d = np.stack([np.cos(th), np.sin(th)], 1)
    p = np.stack([np.sin(th), -np.cos(th)], 1)
    O = np.vstack([[0.0, 0.0], np.cumsum(L[:, None] * d, 0)[:-1]])
    return O, d, p


def points(q):
    """taxel 위치 (3·N_TAX, 2), 각 taxel의 링크 번호, 링크 질량중심 위치."""
    O, d, p = fk(q)
    s = (np.arange(N_TAX) + 0.5) / N_TAX
    P = (O[:, None, :] + (L[:, None] * s[None, :])[..., None] * d[:, None, :] + R_PAD * p[:, None, :]).reshape(-1, 2)
    link = np.repeat(np.arange(3), N_TAX)
    com = O + 0.5 * L[:, None] * d
    return P, link, com, O


def jac(P, i, O):
    """링크 i 위의 점 P의 2×3 Jacobian. θ = −Σq 이므로 열 j는 −ẑ × (P − O_j)."""
    J = np.zeros((2, 3))
    for j in range(i + 1):
        r = P - O[j]
        J[:, j] = [r[1], -r[0]]
    return J


def external_torque(q, c, R):
    """원형 물체(c, R)와의 penalty 접촉력과 중력에 의한 관절 토크, taxel 법선력."""
    P, link, com, O = points(q)
    tau = np.zeros(3)
    u = np.zeros(len(P))
    for k, (Pk, i) in enumerate(zip(P, link)):
        r = Pk - c
        dist = np.linalg.norm(r)
        pen = R - dist
        if pen > 0:
            f = K_C * pen * r / dist  # 물체가 로봇을 바깥으로 미는 힘
            u[k] = K_C * pen
            tau += jac(Pk, i, O).T @ f
    for i in range(3):
        tau += jac(com[i], i, O).T @ np.array([0.0, -MASS[i] * G])
    return tau, u


def equilibrium(a, c, R):
    """에너지 ½(q−a)ᵀKp(q−a) + 접촉 penalty + 중력 위치에너지를 최소화하는 q."""
    def energy(q):
        P, link, com, O = points(q)
        pen = np.clip(R - np.linalg.norm(P - c, axis=1), 0, None)
        e = 0.5 * np.sum(KP * (q - a) ** 2) + 0.5 * K_C * np.sum(pen ** 2) + G * np.sum(MASS * com[:, 1])
        tau, _ = external_torque(q, c, R)
        return e, KP * (q - a) - tau

    res = minimize(energy, a, jac=True, method="L-BFGS-B", bounds=[(-0.3, 1.9)] * 3)
    return res.x


def make_dataset(n, rng):
    Q, U, A, QD, C_R = [], [], [], [], []
    while len(Q) < n:
        a = rng.uniform([0.0, 0.0, 0.0], [1.6, 1.6, 1.4])
        R = rng.uniform(0.012, 0.035)
        c = np.array([rng.uniform(0.02, 0.10), rng.uniform(-0.07, -0.01)])
        q = equilibrium(a, c, R)
        tau, u = external_torque(q, c, R)
        # 평형 확인: Kp(a − q) = −τ_ext (해석식과 같은 식)
        if np.max(np.abs(KP * (a - q) + tau)) > 1e-3:
            continue
        if u.sum() == 0 and rng.random() > 0.2:  # 파지 데이터처럼 대부분 접촉 중인 표본을 남긴다
            continue
        Q.append(q), U.append(u), A.append(a), QD.append(rng.normal(0, 0.4, 3)), C_R.append((c, R))
    return np.array(Q), np.array(U), np.array(A), np.array(QD), C_R


def analytic(Q, sensed, C_R):
    """감지된 taxel 힘만으로 계산한 해석식 목표. sensed: 감지 taxel 마스크."""
    out = []
    for q, (c, R) in zip(Q, C_R):
        P, link, com, O = points(q)
        tau = np.zeros(3)
        for k, (Pk, i) in enumerate(zip(P, link)):
            r = Pk - c
            pen = R - np.linalg.norm(r)
            if pen > 0 and sensed[k]:
                tau += jac(Pk, i, O).T @ (K_C * pen * r / np.linalg.norm(r))
        for i in range(3):
            tau += jac(com[i], i, O).T @ np.array([0.0, -MASS[i] * G])
        out.append(q - tau / KP)
    return np.array(out)


class MLP:
    """2층 ReLU MLP, Adam. 입력·출력 표준화는 밖에서."""

    def __init__(self, din, dout, h, rng):
        self.W = [rng.normal(0, math.sqrt(2 / din), (din, h)), rng.normal(0, math.sqrt(2 / h), (h, h)),
                  rng.normal(0, math.sqrt(1 / h), (h, dout))]
        self.b = [np.zeros(h), np.zeros(h), np.zeros(dout)]
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    def forward(self, X):
        h1 = np.maximum(X @ self.W[0] + self.b[0], 0)
        h2 = np.maximum(h1 @ self.W[1] + self.b[1], 0)
        return h1, h2, h2 @ self.W[2] + self.b[2]

    def step(self, X, Y, lr):
        h1, h2, out = self.forward(X)
        g = 2 * (out - Y) / len(X)
        grads_W, grads_b = [None] * 3, [None] * 3
        grads_W[2], grads_b[2] = h2.T @ g, g.sum(0)
        g2 = (g @ self.W[2].T) * (h2 > 0)
        grads_W[1], grads_b[1] = h1.T @ g2, g2.sum(0)
        g1 = (g2 @ self.W[1].T) * (h1 > 0)
        grads_W[0], grads_b[0] = X.T @ g1, g1.sum(0)
        self.t += 1
        for idx, (p, gr) in enumerate(zip(self.W + self.b, grads_W + grads_b)):
            self.m[idx] = 0.9 * self.m[idx] + 0.1 * gr
            self.v[idx] = 0.999 * self.v[idx] + 0.001 * gr ** 2
            mh = self.m[idx] / (1 - 0.9 ** self.t)
            vh = self.v[idx] / (1 - 0.999 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)


def fit_predict(Xtr, Ytr, Xte, seed, steps=4000, h=128):
    rng = np.random.default_rng(seed)
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8
    ym, ys = Ytr.mean(0), Ytr.std(0) + 1e-8
    net = MLP(Xtr.shape[1], Ytr.shape[1], h, rng)
    Xn, Yn = (Xtr - xm) / xs, (Ytr - ym) / ys
    for s in range(steps):
        idx = rng.integers(0, len(Xn), 256)
        net.step(Xn[idx], Yn[idx], 2e-3 * (0.1 ** (s / steps)))
    return lambda X: net.forward((X - xm) / xs)[2] * ys + ym


def main() -> None:
    rng = np.random.default_rng(0)
    n = 6000
    Q, U, A, QD, C_R = make_dataset(n, rng)

    tr, te = slice(0, n // 2), slice(n // 2, n)  # 1:1 분할 (논문과 같은 비율)
    mae = lambda P, T: float(1e3 * np.mean(np.abs(P - T)))  # ×1e-3 rad
    out: dict = {"n_samples": n, "contact_rate": round(float((U.sum(1) > 0).mean()), 3),
                 "mean_abs_gap_a_minus_q_mrad": mae(Q[te], A[te])}

    # M1 해석식
    full = np.ones(3 * N_TAX, bool)
    tip = np.zeros(3 * N_TAX, bool)
    tip[2 * N_TAX:] = True
    out["M1_analytic_full_sensing_mae"] = mae(analytic(Q[te], full, C_R[n // 2:]), A[te])
    out["M3_analytic_tip_only_sensing_mae"] = mae(analytic(Q[te], tip, C_R[n // 2:]), A[te])

    # M2 Table III식 비교
    feats = {"S+T": np.hstack([Q, U]), "S": Q, "T": U}
    m2 = {"zero_residual_a=q": mae(Q[te], A[te])}
    preds = {}
    for name, X in feats.items():
        for mode in ("residual", "absolute"):
            Y = A - Q if mode == "residual" else A
            errs = []
            for seed in range(3):
                f = fit_predict(X[tr], Y[tr], X[te], seed)
                P = f(X[te]) + (Q[te] if mode == "residual" else 0)
                errs.append(mae(P, A[te]))
                if seed == 0:
                    preds[(name, mode)] = f
            m2[f"{name}/{mode}"] = [round(float(np.mean(errs)), 2), round(float(np.std(errs)), 2)]
    out["M2_table3_style"] = m2

    # M3 손끝만 감지
    Xtip = np.hstack([Q, U[:, 2 * N_TAX:]])
    errs = [mae(fit_predict(Xtip[tr], (A - Q)[tr], Xtip[te], s)(Xtip[te]) + Q[te], A[te]) for s in range(3)]
    out["M3_mlp_tip_only_S+T_residual"] = [round(float(np.mean(errs)), 2), round(float(np.std(errs)), 2)]
    out["M3_fraction_samples_with_unsensed_contact"] = round(float(((U[:, :2 * N_TAX].sum(1) > 0)).mean()), 3)

    # M4 속도 항: 명령 목표 = 평형 목표 + Kp⁻¹ Kd q̇ (q̇는 관측에 없음)
    A_dyn = A + (KD / KP) * QD
    errs = [mae(fit_predict(feats["S+T"][tr], (A_dyn - Q)[tr], feats["S+T"][te], s)(feats["S+T"][te]) + Q[te], A_dyn[te])
            for s in range(3)]
    out["M4_with_unobserved_velocity_S+T_residual"] = [round(float(np.mean(errs)), 2), round(float(np.std(errs)), 2)]
    out["M4_irreducible_floor_mae"] = mae((KD / KP) * QD[te], 0)

    # M5 촉각 잡음 강건성 (데이터셋 전체 촉각 표준편차 기준)
    u_std = float(U[tr].std())
    noise_rng = np.random.default_rng(1)
    m5 = {}
    for pct in (0, 10, 20, 30, 40, 50, 75, 100):
        Un = U[te] + noise_rng.normal(0, u_std * pct / 100, U[te].shape)
        m5[pct] = {
            "S+T/residual": mae(preds[("S+T", "residual")](np.hstack([Q[te], Un])) + Q[te], A[te]),
            "T/residual": mae(preds[("T", "residual")](Un) + Q[te], A[te]),
        }
    out["M5_noise_robustness"] = m5
    out["M5_state_only_residual_reference"] = m2["S/residual"][0]
    out["tactile_dataset_std_N"] = round(u_std, 3)

    path = ROOT / "analysis" / "mapping_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("wrote", path)


if __name__ == "__main__":
    main()
