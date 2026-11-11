"""추가 검증 (VNB): 식 (8) 샘플이 두 봉우리 혼합을 따르는가, 온도 tau와 어닐링에 따라 간격 질량이 어떻게 줄어드는가.

03 §7(d)와 cvar_toy.py part4를 확장한다 (numpy, 단일 스레드, 수 초). 가정 모델:
  - 실제 믿음: 0.5 N(0.18, 0.03^2) + 0.5 N(1.0, 0.05^2)를 기본으로 하고, 낮은 봉우리 질량 w를 0.1–0.9로 바꾼다.
  - 식 (8): theta = sum_k zeta_k (mu_k + sigma_k eps_k), zeta = softmax((logit_k + g_k)/tau), g_k ~ Gumbel(0,1).
    logit은 w에서 정한다 (log(w/(1-w)), 기본 w = 0.5면 0). 코드의 rsample은 tau = 0.5 고정이라 본다.
  - 간격 질량: P(0.4 < theta < 0.8). 실제 혼합의 간격 질량은 거의 0이다.
  - W1 거리: 식 (8) 표본과 실제 혼합 표본의 분위수 차이 평균 (1차원 Wasserstein 근사).
  - 평균과 분산: 식 (8)은 평균은 맞지만 분산이 줄어든다(볼록 결합). 이를 같이 본다.
  - 표본 200,000개, 시드 7 (기존 part4와 같음). 어닐링은 tau를 1.0에서 0.1까지 지수적으로 10단계 낮추는 것으로 가정.

실행: python3 analysis/extra_eq8_tau.py -> analysis/extra_eq8_tau_results.json
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "extra_eq8_tau_results.json"
MU = np.array([0.18, 1.0])
SD = np.array([0.03, 0.05])
N = 200_000
GAP = (0.4, 0.8)
TAUS = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
WS = [0.1, 0.3, 0.5, 0.7, 0.9]
QS = np.linspace(0.005, 0.995, 199)


def true_mixture(w, rng):
    comp = rng.random(N) < w
    return np.where(comp, rng.normal(MU[0], SD[0], N), rng.normal(MU[1], SD[1], N))


def eq8(w, tau, rng):
    logits = np.array([np.log(w), np.log(1 - w)])
    g = -np.log(-np.log(rng.random((N, 2))))
    a = (logits + g) / tau
    a -= a.max(axis=1, keepdims=True)  # 수치 안정화 (작은 tau에서 overflow 방지)
    z = np.exp(a)
    zeta = z / z.sum(axis=1, keepdims=True)
    eps = rng.standard_normal((N, 2))
    return (zeta * (MU + SD * eps)).sum(axis=1)


def stats(x, ref):
    gap = float(np.mean((x > GAP[0]) & (x < GAP[1])))
    w1 = float(np.mean(np.abs(np.quantile(x, QS) - np.quantile(ref, QS))))
    return {"gap_mass": round(gap, 4), "mean": round(float(x.mean()), 4), "sd": round(float(x.std()), 4),
            "W1_to_true_mixture": round(w1, 4)}


def main():
    out = {"settings": {"mu": MU.tolist(), "sd": SD.tolist(), "samples": N, "gap": list(GAP),
                        "taus": TAUS, "w_low_grid": WS, "seed": 7,
                        "note": "가정 모델. 코드 rsample의 tau=0.5는 03 §7(d)에서 확인한 값"}}

    rng = np.random.default_rng(7)
    ref = true_mixture(0.5, rng)
    out["true_mixture_w0.5"] = {"gap_mass": round(float(np.mean((ref > GAP[0]) & (ref < GAP[1]))), 4),
                                "mean": round(float(ref.mean()), 4), "sd": round(float(ref.std()), 4)}

    # (1) tau 스윕 (w = 0.5)
    tau_rows = []
    for tau in TAUS:
        rng = np.random.default_rng(7)
        x = eq8(0.5, tau, rng)
        row = {"tau": tau}
        row.update(stats(x, ref))
        tau_rows.append(row)
    out["tau_sweep_w0.5"] = tau_rows

    # (2) 낮은 봉우리 질량 w 스윕 (tau = 0.5, 코드 값), 실제 혼합 간격 질량과 비교
    w_rows = []
    for w in WS:
        rng = np.random.default_rng(11)
        r = true_mixture(w, rng)
        rng = np.random.default_rng(11)
        x = eq8(w, 0.5, rng)
        row = {"w_low": w, "true_gap_mass": round(float(np.mean((r > GAP[0]) & (r < GAP[1]))), 4),
               "eq8_tau0.5_gap_mass": round(float(np.mean((x > GAP[0]) & (x < GAP[1]))), 4),
               "eq8_tau0.5_mean": round(float(x.mean()), 4), "true_mean": round(float(r.mean()), 4),
               "eq8_tau0.5_W1": round(float(np.mean(np.abs(np.quantile(x, QS) - np.quantile(r, QS)))), 4),
               # 표본 평균에서 역산한 유효 낮은 봉우리 질량 (평균 = w*0.18 + (1-w)*1.0)
               "eq8_tau0.5_effective_w_from_mean": round(float((1.0 - x.mean()) / (1.0 - MU[0])), 4)}
        w_rows.append(row)
    out["w_sweep_tau0.5"] = w_rows

    # (3) 어닐링: tau를 1.0 -> 0.1 지수 감소 10단계, 각 단계의 간격 질량과 W1 (w = 0.5)
    ann = np.geomspace(1.0, 0.1, 10)
    ann_rows = []
    for tau in ann:
        rng = np.random.default_rng(7)
        x = eq8(0.5, float(tau), rng)
        s = stats(x, ref)
        ann_rows.append({"tau": round(float(tau), 4), "gap_mass": s["gap_mass"], "W1": s["W1_to_true_mixture"]})
    out["annealing_1.0_to_0.1_10steps"] = ann_rows
    out["annealing_mean_gap_mass"] = round(float(np.mean([r["gap_mass"] for r in ann_rows])), 4)
    out["annealing_final_gap_mass_tau0.1"] = ann_rows[-1]["gap_mass"]

    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("true mixture w=0.5:", out["true_mixture_w0.5"])
    for r in tau_rows:
        print("tau", r["tau"], "gap", r["gap_mass"], "mean", r["mean"], "sd", r["sd"], "W1", r["W1_to_true_mixture"])
    for r in w_rows:
        print("w", r)
    print("annealing mean gap", out["annealing_mean_gap_mass"], "final", out["annealing_final_gap_mass_tau0.1"])


if __name__ == "__main__":
    main()
