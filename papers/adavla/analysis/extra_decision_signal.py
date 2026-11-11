"""추가 검증 3: 다봉 목표에서 결정 시점 탐지 신호, 곡률 대 분산 변화율 (extra_flow_nfe.py와 같은 toy·시드 재사용).

질문: 낮은 곡률로 분기해 한 번에 점프(1단계 Euler)했을 때 틀리는 샘플(03 §6.3)을,
곡률 대신 '분산 변화율'로 더 잘 골라낼 수 있는가?

정의
- 대상 점: 기준 궤적(RK4 2000단계) 위의 τ_d ∈ {0, 0.5, 0.75, 0.9} 에서의 상태 x_τd.
  τ_d = 0 에서는 거의 모든 점프가 틀려(03 §6.1) 신호를 비교할 수 없어, 후반 τ를 함께 본다.
- 틀린 점프: x_τd + (1−τd)·v(x_τd, τd) 의 끝점 오차 > TOL(0.1).
- 곡률 신호 C: flow_toy와 같은 C = ‖v_τ − v_mid‖/‖v_mid‖ (h = (1−τ)/2). 낮을수록 점프 허용.
- 분산 신호 S: 주어진 상태 x에서 τ를 ±δ 움직였을 때 GMM 사후 분산 V(x,τ)의 변화율 |dV/dτ|.
  V(x,τ) = Σ_k w_k [d(σ_k² − τ²σ_k⁴/D_k) + ‖E_k − Ē‖²],  E_k = μ_k + τσ_k²/D_k (x − τμ_k)
  (x1 | x_τ, k 의 사후 평균·분산, GMM 해석해). 값이 크면 아직 결정되지 않았다고 본다.
- 공정 비교: 두 신호 모두 점프 허용 비율을 50%로 맞춘다(각 신호의 중앙값으로 분할).
  그 안에서 틀린 점프 비율과 AUROC를 비교한다.

실행: python3 analysis/extra_decision_signal.py  (약 1–2분)
출력: analysis/extra_decision_signal_results.json
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import flow_toy as ft  # noqa: E402
import extra_flow_nfe as r1  # noqa: E402  (같은 목표 설정과 시드 재사용)

OUT = HERE / "extra_decision_signal_results.json"
N = 2000
TAUS = [0.0, 0.5, 0.75, 0.9]
DELTA = 1e-3


def posterior_var(g, x, tau):
    tau = np.asarray(tau, float)
    mu = g.mu[None]
    s2 = (g.sig ** 2)[None, :, None]
    xx = x[:, None, :]
    D = tau ** 2 * s2 + (1 - tau) ** 2
    logw = np.log(g.pi)[None, :, None] - 0.5 * ((xx - tau * mu) ** 2).sum(-1, keepdims=True) / D \
        - 0.5 * g.d * np.log(D)
    logw = logw - logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    w = w / w.sum(axis=1, keepdims=True)                       # (N,K,1)
    E = mu + tau * s2 / D * (xx - tau * mu)                     # (N,K,d) 사후 평균
    Ebar = (w * E).sum(1, keepdims=True)
    within = (w * (g.d * (s2 - tau ** 2 * s2 ** 2 / D))).sum(1)[:, 0]
    between = (w * ((E - Ebar) ** 2).sum(-1, keepdims=True)).sum(1)[:, 0]
    return within + between


def var_rate(g, x, tau):
    lo = max(tau - DELTA, 0.0)
    hi = min(tau + DELTA, 1.0)
    return np.abs(posterior_var(g, x, hi) - posterior_var(g, x, lo)) / (hi - lo)


def curvature(g, x, tau):
    h = (1 - tau) / 2
    va = g.v(x, tau)
    vm = g.v(x + va * h, tau + h)
    return np.linalg.norm(va - vm, axis=1) / (np.linalg.norm(vm, axis=1) + ft.EPS)


def auroc(score, label):
    """label=1 이 양성. score가 클수록 양성이라고 본다."""
    pos = label.sum()
    neg = len(label) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def partial_rank_corr(a, b, z):
    """z(상태 노름)의 순위를 통제한 부분 Spearman 상관."""
    ra, rb, rz = rankdata(a), rankdata(b), rankdata(z)
    def resid(y):
        A = np.column_stack([np.ones_like(rz), rz])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return y - A @ coef
    return float(np.corrcoef(resid(ra), resid(rb))[0, 1])


def trajectory_states(g, x0, taus):
    """RK4 2000단계 기준 궤적에서 taus 시점의 상태를 돌려준다."""
    steps = 2000
    h = 1.0 / steps
    want = {int(round(t * steps)): t for t in taus}
    states = {}
    x = x0.copy()
    for i in range(steps + 1):
        if i in want:
            states[want[i]] = x.copy()
        if i == steps:
            break
        t = i * h
        k1 = g.v(x, t)
        k2 = g.v(x + 0.5 * h * k1, t + 0.5 * h)
        k3 = g.v(x + 0.5 * h * k2, t + 0.5 * h)
        k4 = g.v(x + h * k3, t + h)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return states, x


def main():
    t0 = time.time()
    res = {"N": N, "taus": TAUS, "tol": ft.TOL, "seeds": r1.SEEDS, "configs": {}}
    for name, g in r1.configs().items():
        rows = {tau: {"C": [], "S": [], "wrong": [], "e1": [], "norm": []} for tau in TAUS}
        for s in r1.SEEDS:
            x0 = np.random.default_rng(s).normal(size=(N, 2))
            states, ref = trajectory_states(g, x0, TAUS)
            ref_end = ref
            for tau in TAUS:
                x = states[tau]
                jump = x + (1 - tau) * g.v(x, tau)
                err = np.linalg.norm(jump - ref_end, axis=1)
                rows[tau]["C"].append(curvature(g, x, tau))
                rows[tau]["S"].append(var_rate(g, x, tau))
                rows[tau]["wrong"].append(err > ft.TOL)
                rows[tau]["e1"].append(err)
                rows[tau]["norm"].append(np.linalg.norm(x, axis=1))
        rec = {}
        for tau in TAUS:
            C = np.concatenate(rows[tau]["C"])
            S = np.concatenate(rows[tau]["S"])
            err = np.concatenate(rows[tau]["e1"])
            norm = np.concatenate(rows[tau]["norm"])
            wrong = np.concatenate(rows[tau]["wrong"]).astype(int)
            # 가설: C가 클수록, S가 클수록 점프가 위험하다. 허용 비율 50%로 맞춘 비교.
            acc_C = C <= np.median(C)
            acc_S = S <= np.median(S)
            rec[f"tau_{tau}"] = {
                "wrong_rate_all": round(float(wrong.mean()), 4),
                "spearman_C_vs_jump_err": round(ft.spearman(C, err), 4),
                "spearman_S_vs_jump_err": round(ft.spearman(S, err), 4),
                "auroc_wrong_risk_C": round(auroc(C, wrong), 4),
                "auroc_wrong_risk_S": round(auroc(S, wrong), 4),
                "wrong_rate_among_accepted_C": round(float(wrong[acc_C].mean()), 4),
                "wrong_rate_among_accepted_S": round(float(wrong[acc_S].mean()), 4),
                "auroc_within_C_accepted_S": round(auroc(S[acc_C], wrong[acc_C]), 4),
                "partial_spearman_C_vs_err_given_norm": round(partial_rank_corr(C, err, norm), 4),
                "partial_spearman_S_vs_err_given_norm": round(partial_rank_corr(S, err, norm), 4),
                "wrong_missed_by_C_count": int((acc_C & (wrong == 1)).sum()),
                "of_which_rejected_by_S": int((acc_C & (wrong == 1) & ~acc_S).sum()),
            }
        res["configs"][name] = rec
        print("done", name, f"{time.time() - t0:.0f}s", flush=True)
    res["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("wrote", OUT.name, res["runtime_s"], "s")


if __name__ == "__main__":
    main()
