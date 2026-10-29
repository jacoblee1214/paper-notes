"""AW-0 video noise 재표본화(Eq. 3, 5, 6) toy (numpy, CPU 수 초).

N1: 사전학습 분포 u = 0.999·Beta(7,1), σ = 5u/(1+4u)에서 σ > 0.9 비율 (코드 주석의 "~95%")
N2: 혼합 q = 0.5·p_pre + 0.5·U(0,1)의 중요도 가중치 w_v = p_pre/q, w_g = 1/q의 범위 → clip 4.0이 작동하는가
N3: 가중 손실이 목표 분포의 기댓값을 재현하는가, GPU당 배치 6(층화 3+3)에서 "배치 평균 1로 정규화"가 만드는 편향
N4: 추론 joint denoising 단계 수(논문 4, 공개 설정 10/20/40)별로 action expert가 읽는 video σ의 범위
N5: action이 추론에서 읽는 σ 구간이 각 학습 분포에서 차지하는 비율

실행: python3 analysis/noise_toy.py  → analysis/noise_toy_results.json, analysis/noise_samples.npz
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(0)
A, B, S, SHIFT, ALPHA, CLIP = 7.0, 1.0, 0.999, 5.0, 0.5, 4.0
N = 2_000_000


def shift(u):
    return SHIFT * u / (1 + (SHIFT - 1) * u)


def p_pre(u):
    """u = S·b, b ~ Beta(A, B)의 밀도."""
    y = np.clip(u / S, 1e-12, 1 - 1e-12)
    return np.where(u <= S, stats.beta.pdf(y, A, B) / S, 0.0)


def q(u):
    return ALPHA * p_pre(u) + (1 - ALPHA)


def sample_pre(n):
    return S * RNG.beta(A, B, n)


def sample_q(n, stratified_batch=None):
    if stratified_batch is None:
        m = RNG.random(n) < ALPHA
        return np.where(m, sample_pre(n), RNG.random(n))
    # 공식 코드: 배치 B에서 round(αB)개는 p_pre, 나머지는 U(0,1)
    b = stratified_batch
    k = int(round(ALPHA * b))
    nb = n // b
    u = np.concatenate([sample_pre(nb * k).reshape(nb, k), RNG.random((nb, b - k))], axis=1)
    return u


def inference_sigmas(steps):
    t = np.linspace(1.0, 0.0, steps + 1)
    return shift(t)                       # _video_inference_sigmas: [steps+1], 마지막은 0


if __name__ == "__main__":
    res = {}
    u_pre, u_uni = sample_pre(N), RNG.random(N)
    s_pre, s_uni = shift(u_pre), shift(u_uni)
    u_q = sample_q(N)
    s_q = shift(u_q)
    res["N1"] = {"P_pre(sigma>0.9)": round(float((s_pre > 0.9).mean()), 4),
                 "P_pre(sigma>0.95)": round(float((s_pre > 0.95).mean()), 4),
                 "P_uniform(sigma>0.9)": round(float((s_uni > 0.9).mean()), 4),
                 "P_q(sigma>0.9)": round(float((s_q > 0.9).mean()), 4),
                 "median_sigma": {"pre": round(float(np.median(s_pre)), 4), "uniform": round(float(np.median(s_uni)), 4),
                                  "q": round(float(np.median(s_q)), 4)}}

    grid = np.linspace(1e-6, 1 - 1e-6, 200_001)
    wv, wg = p_pre(grid) / q(grid), 1.0 / q(grid)
    res["N2"] = {"max_w_v": round(float(wv.max()), 4), "max_w_g": round(float(wg.max()), 4),
                 "min_w_g": round(float(wg.min()), 4), "bound_1/alpha": 1 / ALPHA, "bound_1/(1-alpha)": 1 / (1 - ALPHA),
                 "clip": CLIP, "clip_ever_active": bool(max(wv.max(), wg.max()) > CLIP),
                 "u_range_where_w_g_eq_2 (p_pre=0)": [S, 1.0]}

    # N3: 손실을 σ의 함수로 가정 (ℓ = σ, ℓ = (1−σ)², ℓ = 1[σ<0.7]) 하고 가중 기댓값 비교
    losses = {"sigma": lambda s: s, "(1-sigma)^2": lambda s: (1 - s) ** 2, "1[sigma<0.7]": lambda s: (s < 0.7).astype(float)}
    n3 = {}
    for name, f in losses.items():
        target_pre, target_uni = float(f(s_pre).mean()), float(f(s_uni).mean())
        lq = f(s_q)
        wvq, wgq = np.minimum(p_pre(u_q) / q(u_q), CLIP), np.minimum(1 / q(u_q), CLIP)
        ub = sample_q(600_000, stratified_batch=6)                 # [100000, 6]
        lb = f(shift(ub))
        wvb, wgb = np.minimum(p_pre(ub) / q(ub), CLIP), np.minimum(1 / q(ub), CLIP)
        wvb, wgb = wvb / wvb.mean(1, keepdims=True), wgb / wgb.mean(1, keepdims=True)
        n3[name] = {"video_target(pre)": round(target_pre, 4), "video_weighted_unnorm": round(float((wvq * lq).mean()), 4),
                    "video_weighted_norm_B6": round(float((wvb * lb).mean()), 4),
                    "guidance_target(uniform)": round(target_uni, 4), "guidance_weighted_unnorm": round(float((wgq * lq).mean()), 4),
                    "guidance_weighted_norm_B6": round(float((wgb * lb).mean()), 4),
                    "unweighted_q": round(float(lq.mean()), 4)}
    res["N3"] = n3
    # 배치 정규화 가중치의 실효 표본 수(ESS/B)
    ub = sample_q(600_000, stratified_batch=6)
    wvb, wgb = p_pre(ub) / q(ub), 1 / q(ub)
    ess = lambda w: float(((w.sum(1) ** 2) / (w ** 2).sum(1)).mean() / w.shape[1])
    res["N3"]["ESS_fraction_B6"] = {"video": round(ess(wvb), 3), "guidance": round(ess(wgb), 3)}

    n4 = {}
    for steps in (4, 10, 20, 40):
        sig = inference_sigmas(steps)[:-1]                   # action이 각 단계에서 읽는 video σ
        n4[steps] = {"sigmas_read_by_action": [round(float(x), 3) for x in sig] if steps <= 10 else
                     [round(float(sig[0]), 3), "...", round(float(sig[-1]), 3)],
                     "min_sigma_read": round(float(sig[-1]), 3),
                     "share_of_steps_sigma>0.9": round(float((sig > 0.9).mean()), 3)}
    res["N4"] = n4

    n5 = {}
    for steps in (4, 10, 20):
        lo = inference_sigmas(steps)[-2]
        n5[steps] = {"inference_sigma_range": [round(float(lo), 3), 1.0],
                     "P_pre(sigma in range)": round(float((s_pre >= lo).mean()), 4),
                     "P_q(sigma in range)": round(float((s_q >= lo).mean()), 4),
                     "P_uniform(sigma in range)": round(float((s_uni >= lo).mean()), 4),
                     "P_pre(sigma<0.7)": round(float((s_pre < 0.7).mean()), 4),
                     "P_uniform(sigma<0.7)": round(float((s_uni < 0.7).mean()), 4)}
    res["N5"] = n5

    np.savez_compressed(ROOT / "analysis" / "noise_samples.npz", s_pre=s_pre[:200_000], s_uni=s_uni[:200_000],
                        s_q=s_q[:200_000], grid=grid[::100], wv=wv[::100], wg=wg[::100])
    (ROOT / "analysis" / "noise_toy_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))
