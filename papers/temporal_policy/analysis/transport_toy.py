"""Temporal Policy toy (CPU, numpy 1스레드). 실행: python3 analysis/transport_toy.py  (약 1–2분)

설정 (논문 식 1–3, 코드 config와 같은 인덱스 구조: H=16, 이력 x0 = s[t-H+1:t], 목표 x1 = s[t-H+1+d:t+d], d=14)
- 상태 과정: 단위분산 AR(1) s_τ = φ s_{τ-1} + ..., 상관 r(k) = φ^|k|. 1차원(D=1), 청크 길이 H=16.
  (논문은 상태와 행동을 같은 표현으로 쓰고 데이터를 정규화한다. 여기서는 분산 1로 맞춘다.)
- 이력 초기화(TP): 원천 x0 = c (이력 그 자체), 보간 x_t = (1-t)x0 + t x1 + ε(1-t)√t z, ε = 1 (코드 기본값).
- 잡음 초기화(CFM): 원천 x0 ~ N(0, I), x_t = (1-t)x0 + t x1 (ε = 0).
- 목표 조건부 p(x1 | c): 가우시안(파트 B) 또는 두 봉우리 혼합(파트 C). 두 방법 모두 같은 조건부 목표를 공유한다.

드리프트는 닫힌 식으로 계산한다(학습 오차 없음 = 오라클). 조건부 기대값
  E[x1 | x_t, k] = μ_k + t Σ S_k^{-1} (x_t - a_k),  a_k = (1-t) x0 + t μ_k,  S_k = t² Σ + ε²(1-t)² t I   (TP)
  b = (E[x1 | x_t] - x_t)/(1-t)   (두 원천 모두 이 형태; 논문 식 2의 드리프트)
  점수 ∇log p(x_t|x0) = Σ_k w_k (-S^{-1}(x_t - a_k))
  논문 식 6(g = 0): b̃ = b - ½ ε²(1-t)² ∇log p  (코드 compute_updated_drift)
  식 5 검산: ∇log p = (t b - (x_t - x0)) / (ε² t (1-t))  → 파트 B에서 수치 비교한다.

출력: analysis/transport_toy_results.json
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "transport_toy_results.json"
SEED = 0
H = 16
LAG = 14          # d (논문 d = H − 2)
EPS = 1.0         # 학습 잡음 계수 (TP)
N = 2000
N_REF = 400       # 기준 해: 같은 방법의 400단계 Euler
NS = [1, 2, 3, 4, 5, 8, 10, 16, 20]


# ---------- 공분산 ----------
def joint_blocks(phi):
    """x0 = s[-15..0] (i=0..15 → 시각 -15+i), x1 = s[-1..14] (j=0..15 → 시각 -1+j). 단위분산 AR(1)."""
    t0 = np.arange(H) - (H - 1)
    t1 = np.arange(H) - (H - 1) + LAG
    S00 = phi ** np.abs(t0[:, None] - t0[None, :])
    S11 = phi ** np.abs(t1[:, None] - t1[None, :])
    S01 = phi ** np.abs(t0[:, None] - t1[None, :])
    return S00, S01, S11


def gauss_conditional(phi):
    S00, S01, S11 = joint_blocks(phi)
    A = S01.T @ np.linalg.inv(S00)            # x1 | x0 평균 = A x0
    Sig = S11 - A @ S01                        # 조건부 공분산
    return A, Sig


# ---------- 드리프트 (오라클) ----------
def _inv(M):
    return np.linalg.inv(M + 1e-12 * np.eye(M.shape[0]))


def tp_drift(x, c, t, mus, logpi, Sig, eps=EPS):
    """이력 원천(x0 = c). mus: (N,K,d) 성분 평균, logpi: (N,K)."""
    d = x.shape[1]
    S = t * t * Sig + (eps ** 2) * (1 - t) ** 2 * t * np.eye(d)
    Si = _inv(S)
    a = (1 - t) * c[:, None, :] + t * mus                  # (N,K,d)
    diff = x[:, None, :] - a
    q = np.einsum("nkd,de,nke->nk", diff, Si, diff)
    logw = logpi - 0.5 * q
    logw = logw - logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    w = w / w.sum(axis=1, keepdims=True)
    G = t * Sig @ Si
    Ex1 = np.einsum("nk,nkd->nd", w, mus + diff @ G.T)
    b = (Ex1 - x) / (1 - t)
    score = -np.einsum("nk,nkd->nd", w, diff @ Si)
    return b, score


def cfm_velocity(x, t, mus, logpi, Sig):
    """잡음 원천 x0 ~ N(0,I), ε = 0. 속도 v = E[x1 - x0 | x_t]."""
    d = x.shape[1]
    V = (1 - t) ** 2 * np.eye(d) + t * t * Sig
    Vi = _inv(V)
    diff = x[:, None, :] - t * mus
    q = np.einsum("nkd,de,nke->nk", diff, Vi, diff)
    logw = logpi - 0.5 * q
    logw = logw - logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    w = w / w.sum(axis=1, keepdims=True)
    M = t * Sig @ Vi - (1 - t) * Vi
    v = np.einsum("nk,nkd->nd", w, mus + diff @ M.T)
    return v


# ---------- 적분 ----------
def run(method, x_init, c, mus, logpi, Sig, n, correction=True):
    """method: 'tp' (x_init = c, ε=1) 또는 'cfm' (x_init = 잡음, ε=0). 반환: 끝점, 경로(n+1, N, d)."""
    x = x_init.copy()
    path = [x.copy()]
    dt = 1.0 / n
    for i in range(n):
        t = i * dt
        if method == "tp":
            b, score = tp_drift(x, c, t, mus, logpi, Sig)
            if correction:
                b = b - 0.5 * (EPS * (1 - t)) ** 2 * score
        else:
            b = cfm_velocity(x, t, mus, logpi, Sig)
        x = x + dt * b
        path.append(x.copy())
    return x, np.stack(path)


def run_sde(c, mus, logpi, Sig, n, g, rng):
    """이력 원천의 SDE 샘플러 (논문 식 6, 코드 compute_updated_drift). g: 추론 잡음 계수."""
    x = c.copy()
    dt = 1.0 / n
    for i in range(n):
        t = i * dt
        b, score = tp_drift(x, c, t, mus, logpi, Sig)
        b = b + 0.5 * (g ** 2 - (EPS * (1 - t)) ** 2) * score
        x = x + dt * b + g * np.sqrt(dt) * rng.standard_normal(x.shape)
    return x


def path_metrics(path):
    seg = np.diff(path, axis=0)                        # (n, N, d)
    sq = (np.linalg.norm(seg, axis=2) ** 2).sum(axis=0)   # Eq.7 (제곱 합), 샘플별
    ln = np.linalg.norm(seg, axis=2).sum(axis=0)          # 코드 compute_transport (길이 합)
    direct = np.linalg.norm(path[-1] - path[0], axis=1)
    ok = direct > 1e-6
    return {
        "eq7_squared_sum_mean": float(sq.mean()),
        "eq8_ratio_squared": float(np.mean(sq[ok] / direct[ok] ** 2)),
        "code_length_ratio": float(np.mean(ln[ok] / direct[ok])),
    }


# ---------- 파트 A: 결합 비용 항등식 ----------
def part_a():
    rng = np.random.default_rng(SEED)
    rows = []
    for phi in [0.5, 0.8, 0.9, 0.95, 0.98, 0.991]:
        S00, S01, S11 = joint_blocks(phi)
        Cj = np.block([[S00, S01], [S01.T, S11]])
        L = np.linalg.cholesky(Cj + 1e-10 * np.eye(2 * H))
        z = rng.standard_normal((20000, 2 * H)) @ L.T
        x0, x1 = z[:, :H], z[:, H:]
        J_tp = np.mean(np.sum((x1 - x0) ** 2, axis=1))
        J_ind = np.mean(np.sum((x1 - rng.standard_normal((20000, H))) ** 2, axis=1))
        rows.append({
            "phi": phi,
            "rho_lag14": float(phi ** LAG),
            "J_temporal_MC": float(J_tp),
            "J_indep_MC": float(J_ind),
            "ratio_MC": float(J_tp / J_ind),
            "ratio_formula_1_minus_rho": float(1 - phi ** LAG),
        })
    # 논문 비율(표 II: CFM 10.00 / TP 1.21 = 8.3)을 맞추는 φ
    target = 1.21 / 10.00
    phi_needed = (1 - target) ** (1 / LAG)
    # 분산 m (정규화 후 E x² = m)일 때 J_TP/J_ind = 2m(1-ρ)/(1+m)
    m_list = [1.0, 1 / 3]
    implied = {}
    for m in m_list:
        rho = 1 - target * (1 + m) / (2 * m)
        implied[f"m={m:.3f}"] = {"rho_lag14_needed_for_ratio_1_over_8.3": float(rho)}
    return {"table": rows, "phi_for_ratio_0.121": float(phi_needed), "implied_rho": implied}


# ---------- 파트 B: 단봉 가우시안 조건부 (오라클 드리프트) ----------
def part_b(phi=0.991, seed=SEED):
    rng = np.random.default_rng(seed)
    A, Sig = gauss_conditional(phi)
    S00, _, _ = joint_blocks(phi)
    Lc = np.linalg.cholesky(S00 + 1e-10 * np.eye(H))
    c = rng.standard_normal((N, H)) @ Lc.T                 # 이력 샘플 (주변분산 1)
    mu = c @ A.T
    mus = mu[:, None, :]
    logpi = np.zeros((N, 1))
    noise0 = rng.standard_normal((N, H))

    # 식 5 검산: 점수 = (t b - (x - c)) / (ε² t (1-t)), 임의 x_t
    checks = []
    for t in [0.2, 0.5, 0.8]:
        x = c + 0.3 * rng.standard_normal((N, H))
        b, sc = tp_drift(x, c, t, mus, logpi, Sig)
        eq5 = (t * b - (x - c)) / (EPS ** 2 * t * (1 - t))
        checks.append({"t": t, "max_abs_diff_score_vs_eq5": float(np.max(np.abs(sc - eq5))),
                       "score_scale": float(np.max(np.abs(sc)))})

    # 기준 해 (각 방법 자기 자신의 400단계)
    x_ref_tp, _ = run("tp", c, c, mus, logpi, Sig, N_REF, correction=True)
    x_ref_cfm, _ = run("cfm", noise0, c, mus, logpi, Sig, N_REF)
    # 무보정(b만) 기준 해
    x_ref_tp_nc, _ = run("tp", c, c, mus, logpi, Sig, N_REF, correction=False)

    res = {"phi": phi, "ratio_formula": float(1 - phi ** LAG), "eq5_check": checks, "per_n": []}
    # 1단계 TP = 조건부 평균(회귀)인지
    x1s, _ = run("tp", c, c, mus, logpi, Sig, 1, correction=True)
    res["tp_1step_minus_conditional_mean_max_abs"] = float(np.max(np.abs(x1s - mu)))
    # 목표 분포 검산: 끝점의 조건부 분산 (tr/d)
    res["target_cond_var_per_dim"] = float(np.trace(Sig) / H)

    for n in NS:
        row = {"n": n}
        for name, (meth, init, corr) in {
            "CFM_noise": ("cfm", noise0, False),
            "TP_history_corrected": ("tp", c, True),
            "TP_history_uncorrected": ("tp", c, False),
        }.items():
            x_end, path = run(meth, init, c, mus, logpi, Sig, n, correction=corr)
            ref = {"CFM_noise": x_ref_cfm, "TP_history_corrected": x_ref_tp,
                   "TP_history_uncorrected": x_ref_tp_nc}[name]
            row[name] = {
                "err_vs_own_400step_per_dim_mse": float(np.mean((x_end - ref) ** 2)),
                "mean_cond_var_err": float(np.mean(np.var(x_end - mu, axis=0)) - np.trace(Sig) / H),
                **path_metrics(path),
            }
        # 복사 기준선: 움직이지 않음(출력 = 이력)
        row["copy_history_baseline"] = {
            "err_vs_own_400step_per_dim_mse": float(np.mean((c - x_ref_tp) ** 2)),
            "eq7_squared_sum_mean": 0.0, "code_length_ratio": None,
        }
        res["per_n"].append(row)

    # 논문 식 5/6 대 무보정: 기준 해끼리 차이 (보정이 필요한가)
    res["ref_corrected_vs_uncorrected_mse"] = float(np.mean((x_ref_tp - x_ref_tp_nc) ** 2))
    res["ref_corrected_endpoint_var_minus_target"] = float(np.mean(np.var(x_ref_tp - mu, axis=0)) - np.trace(Sig) / H)
    res["ref_uncorrected_endpoint_var_minus_target"] = float(np.mean(np.var(x_ref_tp_nc - mu, axis=0)) - np.trace(Sig) / H)
    return res


# ---------- 파트 C: 두 봉우리 혼합, 이력이 가리키는 모드 ----------
def part_c(h_list=(0.02, 0.2, 0.5), sigma=0.2, M=0.5, seed=SEED):
    rng = np.random.default_rng(seed)
    phi = 0.99
    A, _ = gauss_conditional(phi)
    S00, _, _ = joint_blocks(phi)
    Lc = np.linalg.cholesky(S00 + 1e-10 * np.eye(H))
    c = rng.standard_normal((N, H)) @ Lc.T
    mu = c @ A.T
    delta = M * np.linspace(-1, 1, H)                # 모드 방향: + 는 상승 ramp, - 는 하강 ramp
    dhat = delta / np.linalg.norm(delta)
    slope_sign = np.sign(c[:, -1] - c[:, 0])         # 이력이 가리키는 모드 (옛 모드)
    Sig = sigma ** 2 * np.eye(H)
    out = {"sigma": sigma, "M": M, "delta_norm": float(np.linalg.norm(delta)), "rows": []}
    for h in h_list:
        # 조건부 목표: 이력이 가리키는 모드가 확률 1-h, 반대 모드가 h
        p_plus = np.where(slope_sign > 0, 1 - h, h)
        mus = np.stack([mu + delta, mu - delta], axis=1)     # (N,2,d)
        logpi = np.log(np.stack([p_plus, 1 - p_plus], axis=1) + 1e-300)
        # 기준: 실제 샘플 (조건부에서 직접 추출)
        k = (rng.random(N) > p_plus).astype(int)              # 0 = +, 1 = -
        x_true = mus[np.arange(N), k] + sigma * rng.standard_normal((N, H))
        noise0 = rng.standard_normal((N, H))
        x_ref_tp, _ = run("tp", c, c, mus, logpi, Sig, N_REF, correction=True)
        x_ref_cfm, _ = run("cfm", noise0, c, mus, logpi, Sig, N_REF)

        def label(x):
            proj = ((x - mu) @ dhat)
            lab_plus = proj > np.linalg.norm(delta) / 2
            lab_minus = proj < -np.linalg.norm(delta) / 2
            between = ~(lab_plus | lab_minus)
            return lab_plus.mean(), lab_minus.mean(), between.mean()

        truth_plus_frac = float(np.mean(p_plus))
        for n in [1, 2, 4, 8, 16]:
            row = {"h": h, "n": n}
            for name, (meth, init, corr) in {
                "CFM_noise": ("cfm", noise0, False),
                "TP_history_corrected": ("tp", c, True),
                "TP_history_uncorrected": ("tp", c, False),
            }.items():
                x_end, _ = run(meth, init, c, mus, logpi, Sig, n, correction=corr)
                ref = x_ref_cfm if name == "CFM_noise" else x_ref_tp
                fp, fm, fb = label(x_end)
                row[name] = {
                    "frac_plus": float(fp), "frac_minus": float(fm), "frac_between": float(fb),
                    "err_vs_own_400step_per_dim_mse": float(np.mean((x_end - ref) ** 2)),
                    "mse_vs_sampled_truth_per_dim": float(np.mean((x_end - x_true) ** 2)),
                }
            for g in (1.0, 0.1):
                x_s = run_sde(c, mus, logpi, Sig, n, g, rng)
                fp, fm, fb = label(x_s)
                row[f"TP_history_SDE_g{g}"] = {
                    "frac_plus": float(fp), "frac_minus": float(fm), "frac_between": float(fb),
                    "err_vs_own_400step_per_dim_mse": None,
                    "mse_vs_sampled_truth_per_dim": float(np.mean((x_s - x_true) ** 2)),
                }
            row["truth_frac_plus_given_mode_rule"] = truth_plus_frac
            out["rows"].append(row)
    return out


# ---------- 파트 D: 초기 근처 오차의 증폭 (보정항 1/t 효과) ----------
def part_d(phi=0.991, n=200, sizes=1e-2, seed=SEED):
    rng = np.random.default_rng(seed)
    A, Sig = gauss_conditional(phi)
    S00, _, _ = joint_blocks(phi)
    c = rng.standard_normal((N, H)) @ np.linalg.cholesky(S00 + 1e-10 * np.eye(H)).T
    mu = c @ A.T
    mus = mu[:, None, :]
    logpi = np.zeros((N, 1))
    rows = []
    for tj in [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]:
        j = int(round(tj * n))
        out = {"t_inject": tj}
        for corr in (True, False):
            x = c.copy()
            dt = 1.0 / n
            xr = c.copy()
            for i in range(n):
                t = i * dt
                if i == j:
                    x = x + sizes * rng.standard_normal(x.shape)
                b, score = tp_drift(x, c, t, mus, logpi, Sig)
                if corr:
                    b = b - 0.5 * (EPS * (1 - t)) ** 2 * score
                x = x + dt * b
                bR, scR = tp_drift(xr, c, t, mus, logpi, Sig)
                if corr:
                    bR = bR - 0.5 * (EPS * (1 - t)) ** 2 * scR
                xr = xr + dt * bR
            amp = float(np.sqrt(np.mean(np.sum((x - xr) ** 2, axis=1))) / sizes / np.sqrt(H))
            out["corrected" if corr else "uncorrected"] = amp
        out["predicted_sqrt_1_over_t"] = float(np.sqrt(1.0 / tj))
        rows.append(out)
    return {"sizes": sizes, "n": n, "note": "amp = RMS(끝점 편차 per dim)/주입 크기: 주입 시점 t에서 오차 1단위가 끝점에 남기는 배수", "rows": rows}


def main():
    res = {"settings": {"H": H, "LAG": LAG, "EPS_TP": EPS, "N": N, "N_REF": N_REF, "seed": SEED}}
    res["A_coupling_identity"] = part_a()
    res["B_gaussian_oracle_phi0.991"] = part_b(0.991)
    res["B_gaussian_oracle_phi0.9"] = part_b(0.9)
    res["C_mode_switch"] = part_c()
    res["D_perturbation_amplification"] = part_d()
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
