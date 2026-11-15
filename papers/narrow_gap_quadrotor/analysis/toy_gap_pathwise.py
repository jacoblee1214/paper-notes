"""토이 G: 움직이는 틈 통과에서 경로 미분(PW) 기울기의 품질 (numpy, 단일 프로세스, CPU 수 분).

논문 주장 (2610.37557 §3.2, §4, 부록 A.1–A.4)을 우리 방식으로 검사한다.
  주장 1: 부드러운 틈 충돌 비용의 경로 미분은 점수 함수(SF) 추정량보다 샘플당 분산이 작다.
  주장 2: 학습 신호는 틈 충돌 항을 부드럽게 해야 얻어진다 (딱딱한 지시함수는 기울기 0).
  주장 3: 부드러움의 정도(온도 β)가 기울기 크기와 학습 결과에 영향을 준다.

모델 (우리가 직접 쓴 단순화, 논문 코드 아님)
  - 틈 평면을 지나는 드론의 횡방향 2차원 (y, z) 이중 적분기, dt = 1/60 s.
  - 틈은 y 방향으로 진자처럼 움직인다: c(t) = A sin(ωt + φ), A = 0.3 m, f = 0.5 Hz.
  - 틈 크기 W × H = 0.80 × 0.25 m, 드론 반축 a = 0.19 m(y), c = 0.05 m(z) (논문 §3.1, 코드 기본값과 같음).
  - 정책 (y축): μ = −k_p (y − c) − k_d (v_y − ċ) + k_f c̈, z축은 고정 PD. 학습 파라미터 θ = (k_p, k_d, k_f).
  - 행동 노이즈 a = μ + σ ε (σ = 0.5 m/s²). PW와 SF가 같은 노이즈를 쓰므로 같은 목적 E[J]를 추정한다.
  - 비용 J = Σ_k 0.5 (y_k − c_k)² + 18 · Σ_k g(d_k) · 충돌_k, g(d) = exp(−6 d²) (논문 코드의 λ_coll = 6), d_k = 틈 평면까지 거리.
    충돌_k: 부드러운 버전 softplus_β(pen_y) + softplus_β(pen_z), 지시함수 버전 1{pen > 0}.
    pen_y = |y − c| + a − W/2, pen_z = |z| + c − H/2 (드론 반축을 틈 반폭에서 뺀 여유 대비 침범).
  - 성공: 틈 평면 통과 시점(k = T)에 pen ≤ 0 (노이즈 없는 평균 정책).

실험
  A. 기울기 품질 vs 지평 T (10, 20, 40, 80, 120 스텝): PW 샘플당 분산, SF(REINFORCE, 한 롤아웃) 분산, 지시함수 기울기의 0 아닌 비율.
     PW와 SF 평균의 일치(편향 없음)도 확인한다.
  B. 온도 β (1, 4, 16, 64 /m): PW 기울기 크기, 부드러운 비용과 지시함수 비용의 차이, 짧은 학습(Adam) 후 성공률. 지시함수 학습은 대조군.
  C. 의미 점검 (정적 독해 공식의 손계산 재현): 논문 식 (25)와 코드의 충돌 벡터를 드론 정렬 상태에서 틈 기준 대 축소 기준으로 비교.

실행: python3 analysis/toy_gap_pathwise.py   (약 1–2분, 결과: analysis/toy_gap_pathwise_results.json)
"""
import json
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_gap_pathwise_results.json"

DT = 1.0 / 60.0
A_GAP, F_GAP = 0.30, 0.5
OMEGA = 2 * np.pi * F_GAP
W_GAP, H_GAP = 0.80, 0.25
A_DRONE, C_DRONE = 0.19, 0.05
LAMBDA_COLL = 6.0
W_COLL = 18.0
W_TRACK = 0.5
SIGMA = 0.5
Z_KP, Z_KD = 4.0, 3.0
THETA0 = np.array([1.5, 1.0, 0.0])
H_FD = 1e-6


def gap_state(t, phi):
    c = A_GAP * np.sin(OMEGA * t + phi)
    cd = A_GAP * OMEGA * np.cos(OMEGA * t + phi)
    cdd = -A_GAP * OMEGA ** 2 * np.sin(OMEGA * t + phi)
    return c, cd, cdd


def softplus(x):
    return np.logaddexp(0.0, x)


def rollout(theta, x0, phi, eps, T, beta, track_w=W_TRACK, sigma=SIGMA, want_score=False, coll_on=True):
    """x0: dict of (N,) arrays y, vy, z, vz. eps: (T+1, 2, N) 표준 정규 노이즈.

    beta: None이면 지시함수 충돌(비미분), 아니면 softplus 온도(1/m).
    track_w: 중심 추적 항 가중치 (0이면 충돌 항만 남음).
    반환: J (N,), 성공(N,) bool, 점수 함수 항 score (N, 3) (want_score일 때), J_coll (N,) 충돌 항만
    """
    y, vy, z, vz = x0["y"].copy(), x0["vy"].copy(), x0["z"].copy(), x0["vz"].copy()
    N = y.shape[0]
    J = np.zeros(N)
    J_coll = np.zeros(N)
    score = np.zeros((N, 3))
    kp, kd, kf = theta
    for k in range(T + 1):
        t = k * DT
        c, cd, cdd = gap_state(t, phi)
        d = (T - k) * DT * 1.0  # 전진 속도 1 m/s 가정, 틈까지 남은 거리
        g = np.exp(-LAMBDA_COLL * d * d)
        pen_y = np.abs(y - c) + A_DRONE - W_GAP / 2
        pen_z = np.abs(z) + C_DRONE - H_GAP / 2
        if beta is None:
            coll = ((pen_y > 0) | (pen_z > 0)).astype(float)
        elif beta == "hinge":
            coll = np.maximum(pen_y, 0.0) + np.maximum(pen_z, 0.0)
        else:
            coll = softplus(beta * pen_y) / beta + softplus(beta * pen_z) / beta
        if not coll_on:
            coll = np.zeros_like(coll)
        J += track_w * (y - c) ** 2 * DT + W_COLL * g * coll * DT
        J_coll += W_COLL * g * coll * DT
        if k == T:
            break
        mu_y = -kp * (y - c) - kd * (vy - cd) + kf * cdd
        mu_z = -Z_KP * z - Z_KD * vz
        a_y = mu_y + sigma * eps[k, 0]
        a_z = mu_z + sigma * eps[k, 1]
        if want_score:
            # d mu / d theta 는 (-(y-c), -(vy-cd), cdd) 이고 score = (a - mu)/sigma^2 * d mu/d theta
            s = (a_y - mu_y) / sigma ** 2
            score += s[:, None] * np.stack([-(y - c), -(vy - cd), cdd], axis=1)
        vy = vy + DT * a_y
        y = y + DT * vy
        vz = vz + DT * a_z
        z = z + DT * vz
    # 최종 상태의 성공 판정 (노이즈 없는 정책이 아니라 같은 롤아웃의 끝 상태)
    pen_y = np.abs(y - gap_state(T * DT, phi)[0]) + A_DRONE - W_GAP / 2
    pen_z = np.abs(z) + C_DRONE - H_GAP / 2
    success = (pen_y <= 0) & (pen_z <= 0)
    return J, success, score, J_coll


def sample_inits(rng, N):
    x0 = {
        "y": rng.uniform(-0.3, 0.3, N),
        "vy": rng.uniform(-0.2, 0.2, N),
        "z": rng.uniform(-0.05, 0.05, N),
        "vz": rng.uniform(-0.05, 0.05, N),
    }
    phi = rng.uniform(0, 2 * np.pi, N)
    return x0, phi


def fd_grad(theta, x0, phi, eps, T, beta, track_w=W_TRACK, coll_only=False, coll_on=True):
    """중심차분 기울기 (N, 3). 같은 노이즈(공통 난수)로 계산한다. coll_only면 충돌 항만 미분.
    coll_on=False 이면 충돌 항을 빼고 계산한다 (지시함수의 a.e. 기울기 0을 명시적으로 쓸 때)."""
    N = phi.shape[0]
    g = np.zeros((N, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = H_FD
        Jp = rollout(theta + e, x0, phi, eps, T, beta, track_w=track_w, coll_on=coll_on)
        Jm = rollout(theta - e, x0, phi, eps, T, beta, track_w=track_w, coll_on=coll_on)
        if coll_only:
            g[:, j] = (Jp[3] - Jm[3]) / (2 * H_FD)
        else:
            g[:, j] = (Jp[0] - Jm[0]) / (2 * H_FD)
    return g


def sf_grad(theta, x0, phi, eps, T, beta, baseline, track_w=W_TRACK):
    J, _, score, _ = rollout(theta, x0, phi, eps, T, beta, track_w=track_w, want_score=True)
    return (J - baseline)[:, None] * score


def trace_cov(G):
    return float(np.sum(np.var(G, axis=0, ddof=1)))


def loglog_slope(xs, ys):
    xs, ys = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
    return float(np.polyfit(xs, ys, 1)[0])


def experiment_A(rng, N=2048, beta=16.0):
    res = {"config": {"N": N, "beta_per_m": beta, "theta0": THETA0.tolist(), "sigma": SIGMA, "h_fd": H_FD,
                      "objective": "중심 추적(0.5) + 충돌(18·g·softplus_β) — PW와 SF가 같은 J를 쓴다"},
           "rows": []}
    horizons = [10, 20, 40, 80, 120]
    for T in horizons:
        x0, phi = sample_inits(rng, N)
        eps = rng.standard_normal((T + 1, 2, N))
        # 기준선: 독립 배치의 평균 비용 (SF 분산 감소용). 같은 정책, 같은 T.
        x0b, phib = sample_inits(rng, 4096)
        epsb = rng.standard_normal((T + 1, 2, 4096))
        Jb, _, _, _ = rollout(THETA0, x0b, phib, epsb, T, beta)
        baseline = float(Jb.mean())
        gpw = fd_grad(THETA0, x0, phi, eps, T, beta)
        gsf = sf_grad(THETA0, x0, phi, eps, T, beta, baseline)
        # 충돌 항만 미분: 부드러운 버전 대 지시함수 버전
        g_coll_smooth = fd_grad(THETA0, x0, phi, eps, T, beta, track_w=0.0, coll_only=True)
        g_coll_hard = fd_grad(THETA0, x0, phi, eps, T, None, track_w=0.0, coll_only=True)
        J_total, succ_s, _, J_coll_s = rollout(THETA0, x0, phi, eps, T, beta)
        _, _, _, J_coll_h = rollout(THETA0, x0, phi, eps, T, None)
        # 편향 점검: 큰 표본 평균 (PW와 SF가 같은 기대값을 추정하는지)
        bias_N = 8192
        x0m, phim = sample_inits(rng, bias_N)
        epsm = rng.standard_normal((T + 1, 2, bias_N))
        gpw_m = fd_grad(THETA0, x0m, phim, epsm, T, beta)
        gsf_m = sf_grad(THETA0, x0m, phim, epsm, T, beta, baseline)
        diff = gpw_m.mean(0) - gsf_m.mean(0)
        se = np.sqrt(gpw_m.var(0, ddof=1) / bias_N + gsf_m.var(0, ddof=1) / bias_N)
        z = diff / np.maximum(se, 1e-12)
        _, succ_det, _, _ = rollout(THETA0, x0, phi, np.zeros_like(eps), T, beta)
        res["rows"].append({
            "T": T,
            "horizon_s": round(T * DT, 3),
            "pw_grad_norm_mean": float(np.linalg.norm(gpw, axis=1).mean()),
            "pw_grad_norm_max": float(np.linalg.norm(gpw, axis=1).max()),
            "pw_var_trace": trace_cov(gpw),
            "sf_var_trace_single_rollout": trace_cov(gsf),
            "sf_over_pw_var_ratio": trace_cov(gsf) / max(trace_cov(gpw), 1e-300),
            "coll_only_smooth_nonzero_fraction": float(np.mean(np.any(np.abs(g_coll_smooth) > 1e-12, axis=1))),
            "coll_only_smooth_norm_mean": float(np.linalg.norm(g_coll_smooth, axis=1).mean()),
            "coll_only_hard_nonzero_fraction": float(np.mean(np.any(np.abs(g_coll_hard) > 1e-9, axis=1))),
            "coll_only_hard_norm_mean": float(np.linalg.norm(g_coll_hard, axis=1).mean()),
            "smooth_coll_cost_mean": float(np.mean(J_coll_s)),
            "violation_fraction_at_crossing_mean_policy_pct": float(100 * (1 - succ_det.mean())),
            "success_rate_mean_policy_pct": float(100 * succ_det.mean()),
            "bias_z_scores_PW_minus_SF": [round(float(v), 2) for v in z],
        })
    rows = res["rows"]
    res["slopes_loglog"] = {
        "pw_var_vs_T": loglog_slope([r["T"] for r in rows], [r["pw_var_trace"] for r in rows]),
        "sf_var_vs_T": loglog_slope([r["T"] for r in rows], [r["sf_var_trace_single_rollout"] for r in rows]),
        "pw_gradnorm_vs_T": loglog_slope([r["T"] for r in rows], [r["pw_grad_norm_mean"] for r in rows]),
    }
    # 같은 스윕을 학습된 정책 (T=60, β=16, 중심 추적 + 충돌, 시드 0)에서 반복: θ0 실패 정책의 긴 지평 효과를 분리
    theta_tr = train(16.0, 0, "full", T=60)
    res["trained_policy"] = {"theta": [round(float(x), 3) for x in theta_tr], "rows": []}
    for T in horizons:
        x0, phi = sample_inits(rng, 1024)
        eps = rng.standard_normal((T + 1, 2, 1024))
        x0b, phib = sample_inits(rng, 4096)
        epsb = rng.standard_normal((T + 1, 2, 4096))
        Jb, _, _, _ = rollout(theta_tr, x0b, phib, epsb, T, beta)
        baseline = float(Jb.mean())
        gpw = fd_grad(theta_tr, x0, phi, eps, T, beta)
        gsf = sf_grad(theta_tr, x0, phi, eps, T, beta, baseline)
        _, succ_det, _, _ = rollout(theta_tr, x0, phi, np.zeros_like(eps), T, beta)
        res["trained_policy"]["rows"].append({
            "T": T, "pw_var_trace": trace_cov(gpw), "sf_var_trace_single_rollout": trace_cov(gsf),
            "sf_over_pw_var_ratio": trace_cov(gsf) / max(trace_cov(gpw), 1e-300),
            "success_rate_mean_policy_pct": float(100 * succ_det.mean()),
        })
    trows = res["trained_policy"]["rows"]
    res["trained_policy"]["slopes_loglog"] = {
        "pw_var_vs_T": loglog_slope([r["T"] for r in trows], [r["pw_var_trace"] for r in trows]),
        "sf_var_vs_T": loglog_slope([r["T"] for r in trows], [r["sf_var_trace_single_rollout"] for r in trows]),
    }
    return res


def train(beta, seed, mode, iters=150, lr=0.05, batch=256, T=60, hard_grad="fd"):
    """mode='full': 중심 추적 + 충돌. mode='coll_only': 충돌 항만. beta=None이면 지시함수 충돌.
    hard_grad='fd': 지시함수를 중심차분으로 미분 (불연속점에서 큰 값).
    hard_grad='zero': 지시함수 충돌의 a.e. 기울기 0을 그대로 씀 (충돌 항 제외)."""
    rng = np.random.default_rng(1000 + seed)
    track_w = W_TRACK if mode == "full" else 0.0
    coll_on = not (beta is None and hard_grad == "zero")
    theta = THETA0.copy()
    m = np.zeros(3)
    v = np.zeros(3)
    b1, b2, eps_adam = 0.9, 0.999, 1e-8
    for it in range(1, iters + 1):
        x0, phi = sample_inits(rng, batch)
        eps = rng.standard_normal((T + 1, 2, batch))
        g = fd_grad(theta, x0, phi, eps, T, beta, track_w=track_w, coll_on=coll_on).mean(0)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** it)
        vh = v / (1 - b2 ** it)
        theta = theta - lr * mh / (np.sqrt(vh) + eps_adam)
    return theta


def eval_success(theta, seed, N=4096, T=60):
    rng = np.random.default_rng(777 + seed)
    x0, phi = sample_inits(rng, N)
    _, succ, _, _ = rollout(theta, x0, phi, np.zeros((T + 1, 2, N)), T, 16.0)
    return float(100 * succ.mean())


def experiment_B(rng, N=2048, T=60):
    res = {"config": {"T": T, "train_iters": 150, "lr": 0.05, "batch": 256, "seeds": [0, 1, 2],
                      "theta0": THETA0.tolist(),
                      "modes": {"full": "중심 추적 0.5 + 충돌", "coll_only": "충돌 항만"}},
           "beta_rows": []}
    x0, phi = sample_inits(rng, N)
    eps = rng.standard_normal((T + 1, 2, N))
    _, _, _, J_coll_h = rollout(THETA0, x0, phi, eps, T, None, track_w=0.0)
    _, _, _, J_coll_hinge = rollout(THETA0, x0, phi, eps, T, "hinge", track_w=0.0)
    rows_spec = [(b, f"beta={b}") for b in [1.0, 4.0, 16.0, 64.0]] + [(None, "hard")]
    for beta, label in rows_spec:
        row = {"label": label, "beta_per_m": beta}
        if beta is not None:
            g_full = fd_grad(THETA0, x0, phi, eps, T, beta)
            _, _, _, J_coll_s = rollout(THETA0, x0, phi, eps, T, beta, track_w=0.0)
            row["pw_grad_norm_full_mean"] = float(np.linalg.norm(g_full, axis=1).mean())
            # 온도 β의 근사 오차: 힌지(β→∞ 극한, max(pen,0)) 비용과의 차이 (단위가 같은 비교)
            row["coll_cost_mean_smooth"] = float(np.mean(J_coll_s))
            row["coll_cost_mean_hinge_limit"] = float(np.mean(J_coll_hinge))
            row["coll_cost_abs_dev_from_hinge_mean"] = float(np.mean(np.abs(J_coll_s - J_coll_hinge)))
        else:
            g_full = fd_grad(THETA0, x0, phi, eps, T, None)
            row["pw_grad_norm_full_mean"] = float(np.linalg.norm(g_full, axis=1).mean())
            row["coll_cost_mean_smooth"] = float(np.mean(J_coll_h))
            row["coll_cost_mean_hinge_limit"] = float(np.mean(J_coll_hinge))
            row["coll_cost_abs_dev_from_hinge_mean"] = float(np.mean(np.abs(J_coll_h - J_coll_hinge)))
        combos = [("full", "fd"), ("coll_only", "fd")]
        if beta is None:
            combos += [("full", "zero"), ("coll_only", "zero")]
        for mode, hg in combos:
            key = mode if hg == "fd" else f"{mode}_zero_grad"
            thetas = [train(beta, s, mode, T=T, hard_grad=hg) for s in (0, 1, 2)]
            succ = [eval_success(th, s, T=T) for th, s in zip(thetas, (0, 1, 2))]
            row[key] = {
                "theta_per_seed": [[round(float(x), 3) for x in th] for th in thetas],
                "success_pct_per_seed": [round(x, 2) for x in succ],
                "success_pct_mean": float(np.mean(succ)),
                "success_pct_sd": float(np.std(succ, ddof=1)),
            }
        res["beta_rows"].append(row)
    res["initial_theta0_success_pct"] = eval_success(THETA0, 0, T=T)
    res["beta_rows_note"] = ("coll_only는 충돌 항만 학습하는 조건이다. 지시함수 행은 기울기가 0이라 "
                             "θ가 바뀌지 않아야 한다(full에서는 추적 항만 학습).")
    return res


def code_style_collision(p, R, c, gap_half, drone_Q, offset_is_margin):
    """논문 식 (25) 및 코드 _get_collision_vector 의 손계산 재현 (학습 항 대 평가 항 비교용).

    gap_half: (반폭, 반높이). offset_is_margin=True 이면 코드의 축소 기준(w_half = gap/2 - margin = 드론 반축)을 쓴다.
    """
    u, v = np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    A = np.stack([u, -u, v, -v])
    w_half_phys, h_half_phys = gap_half
    if offset_is_margin:
        w_half, h_half = DRONE_A, DRONE_C          # 코드: (gap_w - 2*margin_h)/2 = drone_radius
    else:
        w_half, h_half = w_half_phys, h_half_phys  # 물리적 틈 반폭 (평가 함수와 같은 기준)
    b = np.array([w_half + u @ c, w_half - u @ c, h_half + v @ c, h_half - v @ c])
    ARQ = A @ R @ drone_Q
    norms = np.linalg.norm(ARQ, axis=1)
    return np.maximum(norms + A @ p - b, 0.0)


DRONE_A, DRONE_C = A_DRONE, C_DRONE


def experiment_C():
    Q = np.diag([A_DRONE, A_DRONE, C_DRONE])
    R = np.eye(3)
    c = np.zeros(3)
    out = []
    for off in [0.0, 0.05, 0.10, 0.20, 0.30]:
        p = np.array([0.0, off, 0.0])  # 틈 폭 방향(y)으로 밀기
        train_like = code_style_collision(p, R, c, (W_GAP / 2, H_GAP / 2), Q, True)
        phys = code_style_collision(p, R, c, (W_GAP / 2, H_GAP / 2), Q, False)
        out.append({"offset_y_m": off, "code_style_margin_sum": float(train_like.sum()),
                    "physical_half_sizes_sum": float(phys.sum())})
    return {"aligned_drone_R_eye": True, "rows": out,
            "note": "정렬된 드론(R=I)에서 축소 기준 합은 |offset|과 같고, 물리 기준은 offset ≤ 0.21 m에서 0. "
                    "이것은 손계산 재현이며 논문 저자의 의도는 확인하지 못함"}


def main():
    t0 = time.time()
    rng = np.random.default_rng(20261027)
    resA = experiment_A(rng)
    print("A done", round(time.time() - t0, 1), "s")
    resB = experiment_B(np.random.default_rng(20261028))
    print("B done", round(time.time() - t0, 1), "s")
    resC = experiment_C()
    out = {"model": "2-D 횡방향 이중 적분기 + 움직이는 틈, 우리가 쓴 numpy 토이 (논문 코드 아님)",
           "dt_s": DT, "gap_W_H_m": [W_GAP, H_GAP], "drone_half_axes_m": [A_DRONE, C_DRONE],
           "lambda_coll_per_m2": LAMBDA_COLL, "w_coll": W_COLL, "w_track": W_TRACK,
           "A_horizon_sweep": resA, "B_temperature": resB, "C_semantics_check": resC,
           "runtime_s": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT, "runtime", out["runtime_s"], "s")
    for r in resA["rows"]:
        print(f"T={r['T']:4d} pwVar={r['pw_var_trace']:.3g} sfVar={r['sf_var_trace_single_rollout']:.3g} "
              f"ratio={r['sf_over_pw_var_ratio']:.3g} hardNZ={r['coll_only_hard_nonzero_fraction']:.3f} "
              f"smoothNZ={r['coll_only_smooth_nonzero_fraction']:.3f} succ={r['success_rate_mean_policy_pct']:.1f} "
              f"z={r['bias_z_scores_PW_minus_SF']}")
    print("slopes", resA["slopes_loglog"])
    for r in resB["beta_rows"]:
        print(r["label"], f"|g|={r['pw_grad_norm_full_mean']:.3g}", f"dev_hinge={r['coll_cost_abs_dev_from_hinge_mean']:.3g}",
              "full", round(r["full"]["success_pct_mean"], 1), round(r["full"]["success_pct_sd"], 1),
              "coll_only", round(r["coll_only"]["success_pct_mean"], 1), round(r["coll_only"]["success_pct_sd"], 1))
        if "coll_only_zero_grad" in r:
            print("   zero-grad hard:", r["full_zero_grad"]["success_pct_mean"], r["coll_only_zero_grad"]["success_pct_mean"])
    print("theta0 success", resB["initial_theta0_success_pct"])


if __name__ == "__main__":
    main()
