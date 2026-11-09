"""토이 A: 중복되고 잡음 낀 모달리티를 이어 붙이면 유한한 데이터에서 왜 손해인가 (numpy, 1스레드, 약 1분).

가정(논문에 없음, 설계자가 정함):
- 잠재 z ∈ R², 라벨 y = 1[z1 + 0.5 z2 > 0] (무잡음). 힘장(TacFF)은 z를 낮은 차원으로 담는다고 본다.
- 시각 V: 64차원, z를 σ=2 잡음으로 본다(차원마다 신호가 약함).
- 촉각 RGB T: 512차원. z는 약하게(계수 0.1) 담기고, 과제와 무관한 잠재 50개가 큰 분산으로 섞인다(시각 덮기 대신 분산 증가).
- 힘장 F: 20차원, z를 σ=0.2 잡음으로 담는다. E1의 "V+F_raw"는 실제 단위(약 1e-3)를 가정해 F×1e-3을 그대로 넣는다.
- 학습기: 최소제곱 릿지(λ=1, 절편 있음), 라벨 ±1. 평가는 무잡음 라벨의 정확도.

실험:
- E1: 표본 수 n에 따른 V, V+T, V+F, V+T+F, 잠재 z 오라클 (표준화 포함/미포함)
- E2: F의 척도 α ∈ {1, 0.1, 0.01, 0.001}에 따른 V+F(원단위) 대 V+F(표준화), n = 30과 500

실행: python3 analysis/concat_toy.py → analysis/concat_toy_results.json
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "concat_toy_results.json"

D_Z = 2
W_TRUE = np.array([1.0, 0.5])
N_TEST = 4000
SEEDS = 40
LAM = 1.0
NS = [30, 60, 120, 250, 500, 1000]

# 측정 행렬은 고정(장치 설계), 잡음만 시행마다 바뀐다
_g = np.random.default_rng(12345)
A_V = _g.standard_normal((64, D_Z))
A_T = 0.1 * _g.standard_normal((512, D_Z))
U_T = 2.0 * _g.standard_normal((512, 50))  # 과제와 무관한 잠재 50개
A_F = _g.standard_normal((20, D_Z))
SIG_V, SIG_T, SIG_F = 2.0, 0.5, 0.2


def sample(n, rng, alpha=1.0):
    z = rng.standard_normal((n, D_Z))
    y_clean = np.sign(z @ W_TRUE)
    y_clean[y_clean == 0] = 1
    V = z @ A_V.T + SIG_V * rng.standard_normal((n, 64))
    T = z @ A_T.T + (rng.standard_normal((n, 50)) @ U_T.T) + SIG_T * rng.standard_normal((n, 512))
    F = alpha * (z @ A_F.T + SIG_F * rng.standard_normal((n, 20)))
    # 원단위 힘장: 실제 힘 단위(약 1e-3 수준)를 가정한 척도 불일치 버전
    return {"z": z, "V": V, "T": T, "F": F, "Fmm": 1e-3 * F}, y_clean


def fit_predict(Xtr, ytr, Xte, standardize, lam=LAM):
    if standardize:
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    m_y = ytr.mean()
    Xc = Xtr - Xtr.mean(0)
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ (ytr - m_y))
    return np.sign(Xte @ w + m_y - Xtr.mean(0) @ w + 1e-12)


def acc(pred, y):
    return float(np.mean(pred == y))


def run_E1():
    conds = {
        "oracle_z": (["z"], True),
        "V": (["V"], True),
        "V+T_std": (["V", "T"], True),
        "V+F_raw": (["V", "Fmm"], False),
        "V+F_std": (["V", "F"], True),
        "V+T+F_std": (["V", "T", "F"], True),
    }
    res = {c: {str(n): [] for n in NS} for c in conds}
    for seed in range(SEEDS):
        rng = np.random.default_rng(1000 + seed)
        test, y_te = sample(N_TEST, rng)
        for n in NS:
            train, y_tr = sample(n, rng)
            for c, (keys, std) in conds.items():
                Xtr = np.hstack([train[k] for k in keys])
                Xte = np.hstack([test[k] for k in keys])
                pred = fit_predict(Xtr, y_tr, Xte, std)
                res[c][str(n)].append(acc(pred, y_te))
    out = {}
    for c in conds:
        out[c] = {n: {"mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v)), 4)}
                  for n, v in res[c].items()}
    return out


def run_E2(ns=(30, 500)):
    alphas = [1.0, 0.1, 0.01, 0.001]
    out = {"alphas": alphas}
    for n in ns:
        rows = {"V": [], "V+F_raw": [], "V+F_std": []}
        for a in alphas:
            r = {k: [] for k in rows}
            for seed in range(SEEDS):
                rng = np.random.default_rng(5000 + seed)
                test, y_te = sample(N_TEST, rng, alpha=a)
                train, y_tr = sample(n, rng, alpha=a)
                Xtr = np.hstack([train["V"], train["F"]])
                Xte = np.hstack([test["V"], test["F"]])
                r["V+F_raw"].append(acc(fit_predict(Xtr, y_tr, Xte, False), y_te))
                r["V+F_std"].append(acc(fit_predict(Xtr, y_tr, Xte, True), y_te))
                r["V"].append(acc(fit_predict(train["V"], y_tr, test["V"], True), y_te))
            for k in rows:
                rows[k].append(round(float(np.mean(r[k])), 4))
        out[f"n={n}"] = rows
    return out


def main():
    t0 = time.time()
    e1 = run_E1()
    e2 = run_E2()
    res = {"config": {"seeds": SEEDS, "n_test": N_TEST, "lambda": LAM, "ns": NS,
                      "dims": {"V": 64, "T": 512, "F": 20, "z": 2}, "sig": [SIG_V, SIG_T, SIG_F]},
           "E1_sample_size": e1, "E2_force_scale": e2,
           "runtime_s": None}
    res["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("wrote", OUT, "in", res["runtime_s"], "s")
    for c in e1:
        print(f"{c:12s}", " ".join(f"n={n}:{e1[c][str(n)]['mean']:.3f}" for n in NS))
    for n in (30, 500):
        print("E2 n=%d alphas" % n, e2["alphas"])
        for k in ("V", "V+F_raw", "V+F_std"):
            print(f"  {k:8s}", e2[f"n={n}"][k])


if __name__ == "__main__":
    main()
