"""토이: 비백색(AR(1)) 잔차에서 κ 보정 후 검출 AUROC가 식 (8)을 따르는지 확인 (CPU, 수십 초).

모형: 한 블록당 K=4 방향의 점수 s_t = δ + ε_t, ε_t는 AR(1)(ρ), Σ=I, F_t=I.
  윈도 n = 20, S = Σ s_t, F = n I, 정상(δ=0) 대 이동(δ) 윈도의 Λ = S⊤F⁻¹S/κ.
  κ는 정상 윈도 1만 개의 Var(S)/F 평균으로 추정 (논문의 nominal 보정과 같은 절차).
  비교: (a) 참 κ = (1+ρ)/(1-ρ) 근사, (b) 추정 κ, (c) 식 (8) 근사 AUROC(λ = n‖δ‖²/κ).
실행: python3 analysis/auroc_mc_check.py → analysis/auroc_mc_check_results.json
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "auroc_mc_check_results.json"
rng = np.random.default_rng(0)
K, n, rho, N = 4, 20, 0.5, 4000

def windows(delta_norm, N):
    # 방향 하나의 경로 (δ는 첫 축 방향)
    mu = np.zeros(K); mu[0] = delta_norm
    e = np.zeros((N, n, K))
    e[:, 0] = rng.standard_normal((N, K))
    for t in range(1, n):
        e[:, t] = rho * e[:, t - 1] + math.sqrt(1 - rho * rho) * rng.standard_normal((N, K))
    s = mu + e
    return s.sum(axis=1)  # S, F = nI

def lam_stat(S, kappa):
    return (S * S).sum(axis=1) / (n * kappa)

# κ 추정: 정상 윈도 Var(S)/F의 평균 (K 방향 평균)
S0 = windows(0.0, 20000)
kappa_hat = float((S0.var(axis=0) / n).mean())
kappa_true = (1 + rho) / (1 - rho)  # 긴 윈도 극한 (AR(1) 누적 분산 / 1)

def emp_auc(shift, kappa):
    a = lam_stat(windows(shift, N), kappa)
    b = lam_stat(windows(0.0, N), kappa)
    return float((a[:, None] > b[None, :]).mean())

def eq8(lam, K=K):
    return float(stats.norm.cdf(lam / (2 * math.sqrt(K + lam))))

rows = []
for d in (0.3, 0.5, 0.7, 0.9):
    lam = n * d * d / kappa_true
    rows.append({
        "shift_per_dir": d,
        "lambda_true_kappa": round(lam, 2),
        "auc_emp_true_kappa": round(emp_auc(d, kappa_true), 3),
        "auc_emp_est_kappa": round(emp_auc(d, kappa_hat), 3),
        "auc_eq8_true_kappa": round(eq8(lam), 3),
    })
res = {"K": K, "n": n, "rho": rho, "kappa_true_approx": round(kappa_true, 3),
       "kappa_estimated_from_nominal": round(kappa_hat, 3),
       "note": "식 (8)은 λ = n‖δ‖²/κ로 쓴다. 사용한 δ는 한 방향 크기이고 F̄ = I이므로 ‖δ‖²_F = δ². 짧은 윈도 n = 20은 κ 극한과 다르다.",
       "rows": rows}
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
