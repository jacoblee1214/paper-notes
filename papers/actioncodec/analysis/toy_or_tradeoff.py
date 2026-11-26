"""토이 T1: 겹침률(OR)과 재구성 오차의 절충을 합성 행동 데이터로 확인한다.

목적: 논문 §3.1과 부록 B.2.1은 OR(식 7: 인접 청크 토큰의 위치별 일치 비율)을 높이면 VLA 학습이 빨라진다고 보고하고,
      표 5는 OR 26% -> 70%에서 L1 재구성 오차가 0.020 -> 0.028로 늘었다고 적는다. 이 토이는 학습 없이도
      양자화 간격만으로 OR이 오르고 재구성 오차도 같이 오른다는 것, 그리고 간격에서 OR을 예측할 수 있다는 것을 보인다.
      즉 OR은 토큰 안정성의 대리 지표이고, 재구성 오차와 함께 읽어야 한다(논문 부록 E.1도 같은 주의를 적음).
      이 토이는 논문 수치를 재현하지 않는다. 학습된 VQ가 아니라 PCA 잠재 + 스칼라 양자화이다.

설정(모두 합성, 외부 데이터 없음): 7차원 매끄러운 신호(가우시안 필터 폭 sigma_t 스텝), 청크 길이 T = 20, 보폭 1.
  - Binning: 원소별 B개 구간 (논문 표 2의 Binning과 같은 방식, 토큰 140개)
  - PCA-양자화: 140차원 청크를 PCA로 n개 성분만 남기고, 각 성분을 표준편차로 나눠 [-4, 4]를 S단계로 양자화
출력: analysis/toy_or_tradeoff_results.json
실행: python3 analysis/toy_or_tradeoff.py   (numpy만, 10초 안팎)
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_or_tradeoff_results.json"
T, D = 20, 7
rng = np.random.default_rng(0)


def make_signal(length, sigma_t, seed):
    r = np.random.default_rng(seed)
    w = r.standard_normal((length + 12 * sigma_t, D))
    k = np.exp(-0.5 * (np.arange(-4 * sigma_t, 4 * sigma_t + 1) / sigma_t) ** 2)
    k /= k.sum()
    x = np.stack([np.convolve(w[:, d], k, mode="same") for d in range(D)], 1)[6 * sigma_t: 6 * sigma_t + length]
    x = x / x.std(0, keepdims=True) * 0.35
    return np.clip(x, -1, 1)


def chunks(x):
    n = len(x) - T + 1
    idx = np.arange(T)[None, :] + np.arange(n)[:, None]
    return x[idx]  # (n, T, D)


def overlap_rate(tok):
    """식 7: 인접 청크(보폭 1)의 같은 위치 토큰이 같은 비율의 평균."""
    return float((tok[1:] == tok[:-1]).mean())


def run(sigma_t, seed):
    x = make_signal(20000, sigma_t, seed)
    C = chunks(x)
    n_train = len(C) // 2
    flat = C.reshape(len(C), -1)
    res = {"sigma_t": sigma_t}

    # Binning (B = 1000), 표 2의 Binning 방식
    B = 1000
    tok = np.floor((C + 1) / 2 * B).clip(0, B - 1).astype(int).reshape(len(C), -1)
    rec = ((tok + 0.5) / B * 2 - 1)
    res["binning_B1000"] = {"tokens": int(tok.shape[1]), "OR": round(overlap_rate(tok), 3),
                            "L1": round(float(np.abs(rec - flat).mean()), 5)}

    # PCA-양자화
    mu = flat[:n_train].mean(0)
    U, s, Vt = np.linalg.svd(flat[:n_train] - mu, full_matrices=False)
    rows = []
    for ncomp in (8, 16, 32):
        V = Vt[:ncomp].T
        z = (flat - mu) @ V
        sd = z[:n_train].std(0)
        zn = z / sd
        for S in (16, 32, 64, 128, 256, 512, 1024, 2048):
            step = 8.0 / S
            idx = np.floor((np.clip(zn, -4, 4 - 1e-9) + 4) / step).astype(int)
            zq = (idx + 0.5) * step - 4
            recon = (zq * sd) @ V.T + mu
            l1 = float(np.abs(recon - flat).mean())
            ORm = overlap_rate(idx)
            dz = np.abs(zn[1:] - zn[:-1])
            or_pred = float(np.clip(1 - dz / step, 0, 1).mean())
            rows.append({"n_tokens": ncomp, "S": S, "OR": round(ORm, 3), "OR_pred_from_step": round(or_pred, 3),
                         "L1": round(l1, 5)})
    res["pca_quant"] = rows
    return res


results = [run(6, 1), run(12, 2)]
OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

for r in results:
    print("sigma_t =", r["sigma_t"], "Binning B=1000:", r["binning_B1000"])
    print("  n  S     OR   OR_pred  L1")
    for row in r["pca_quant"]:
        if row["n_tokens"] == 16:
            print(f"  {row['n_tokens']:2d} {row['S']:5d} {row['OR']:.3f} {row['OR_pred_from_step']:.3f}  {row['L1']:.5f}")
    ors = [row["OR"] for row in r["pca_quant"] if row["n_tokens"] == 16]
    l1s = [row["L1"] for row in r["pca_quant"] if row["n_tokens"] == 16]
    mono_or = all(a >= b for a, b in zip(ors, ors[1:]))
    mono_l1 = all(a >= b for a, b in zip(l1s, l1s[1:]))
    print("  S 증가에 따라 OR 단조 감소:", mono_or, ", L1 단조 감소:", mono_l1)
