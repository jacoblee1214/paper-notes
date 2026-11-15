"""RL token 핵심 주장 검증용 toy (numpy, 단일 프로세스, CPU).

질문: 고정된 VLA 임베딩 위에서 (a) 저차원 readout(PCA 토큰)에 잔차 정책을 얹고 기준 행동에 묶는 방식이
(b) 임베딩 전체(256차원)에 잔차 헤드를 얹고 묶지 않는 방식(전체 헤드 미세조정)보다
정책 개선이 빠르고 기울기 추정의 분산이 작은가?

환경: 1스텝 문맥 밴딧(연속 행동 D=4). 문맥 s ~ N(0, I_16). 임베딩 z = G [3s; n]의 256차원, n은 과제와 무관한 잡음 64차원(분산 1).
목표 행동 a* = T s. 기준(VLA) 행동 ã = a* + (U s + b): 체계적 오차가 있는 선형 편향.
보상 r = exp(-||a - a*||² / (2 τ²)), τ = 0.25. 기준 정책의 보상은 낮고(탐색 필요), 최적 보상은 1.

비교 방법 (모두 평균 μ = ã + Θ φ, 행동 a ~ N(μ, σ² I), σ = 0.1):
- full:        φ = z (256차원), 묶음 없음 (전체 헤드 미세조정)
- tokM:        φ = PCA_m(z) (m차원 readout), 묶음 없음
- tokM+anchor: φ = PCA_m(z), 기준 행동에 묶음 β ||μ - ã||² 추가 (논문 식 (5)의 근사)

주의: 이 toy는 RL token을 재구성 손실로 학습하지 않습니다. PCA를 readout 대리로 씁니다.
또 한 스텝 밴딧이라 논문의 청크 크레딧 할당 문제는 검증하지 않습니다.

측정:
- (변경) 업데이트는 Q 대신 실제 보상의 행동 기울기(오라클)를 씀. 논문 식 (5)의 "Q의 행동 기울기" 꼴과 대응.
- 학습 곡선: 에피소드 수에 따른 결정적 평균 행동의 보상(평가는 별도 256문맥, 샘플 수에 포함하지 않음)
- 샘플 수 기준 90% 개선 도달 에피소드 수 (기준에서 최적까지 거리의 90%)
- 초기 정책에서 기울기 추정(배치 16) 분산의 대각 합(trace)
- 튜닝: 학습률 {0.05, 0.2, 1.0}을 튜닝 시드 10개에서 골라 고정하고, 평가 시드 40개에서 재측정
"""
import json
import time
from pathlib import Path

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent
K, NUI, E, D = 16, 64, 256, 4
SIG, TAU = 0.1, 0.4
BATCH, UPDATES, EVAL_EVERY, EVAL_N = 16, 1200, 40, 256
BETA = 0.5
LRS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
TUNE_SEEDS = list(range(1000, 1010))
EVAL_SEEDS = list(range(40))


def make_world(seed):
    rng = np.random.default_rng(seed)
    G = rng.normal(0, 1 / np.sqrt(K + NUI), size=(E, K + NUI))
    scale = np.concatenate([3.0 * np.ones(K), np.ones(NUI)])
    T = rng.normal(0, 1 / np.sqrt(K), size=(D, K))
    U = rng.normal(0, 0.3 / np.sqrt(K), size=(D, K))
    b = np.array([0.4, -0.3, 0.2, 0.0])
    return dict(G=G, scale=scale, T=T, U=U, b=b, seed=seed)


def sample_contexts(w, n, rng):
    eps = rng.normal(size=(n, K + NUI))
    s = eps[:, :K]
    z = (eps * w["scale"]) @ w["G"].T
    a_star = s @ w["T"].T
    a_ref = a_star + s @ w["U"].T + w["b"]
    return s, z, a_star, a_ref


def reward(a, a_star):
    d2 = np.sum((a - a_star) ** 2, axis=1)
    return np.exp(-np.minimum(d2, 1e6) / (2 * TAU ** 2))


_FEAT_CACHE = {}


def make_features(method, w, rng_pca):
    """방법별 특징 변환: (z - mu) @ W.T. 모든 특징을 단위 분산으로 맞춘다(학습률 비교를 위해).
    PCA는 라벨 없는 임베딩 2000개로 한 번 계산한다."""
    kind, m = method[0], method[1]
    key = (method[:2], w["seed"])
    if key in _FEAT_CACHE:
        return _FEAT_CACHE[key]
    _, zcal, _, _ = sample_contexts(w, 2000, rng_pca)
    mu = zcal.mean(0)
    if kind == "full":
        sd = zcal.std(0) + 1e-8
        W = np.diag(1.0 / sd)
        dim = E + 1
    else:
        _, S, Vt = np.linalg.svd(zcal - mu, full_matrices=False)
        W = Vt[:m] * (np.sqrt(len(zcal)) / S[:m])[:, None]
        dim = m + 1
    _FEAT_CACHE[key] = ((mu, W), dim)
    return (mu, W), dim


def phi_of(feat, z):
    """특징에 상수항 1을 붙인다(기준 행동의 상수 편향 b를 표현하려면 필요)."""
    mu, W = feat
    return np.hstack([(z - mu) @ W.T, np.ones((len(z), 1))])


def run(method, lr, seed, record=True):
    """한 시드에서 학습. 반환: 곡선, 샘플 수 기준 90% 도달 에피소드, 초기 기울기 분산."""
    kind, m = method[0], method[1]
    anchor = kind == "tokA"
    beta_m = method[2] if len(method) > 2 else BETA
    w = make_world(seed)
    rng = np.random.default_rng(seed + 77)
    rng_pca = np.random.default_rng(seed + 991)
    feat, dim = make_features(method, w, rng_pca)
    Theta = np.zeros((D, dim))

    s_ev, z_ev, a_star_ev, a_ref_ev = sample_contexts(w, EVAL_N, np.random.default_rng(seed + 5))
    phi_ev = phi_of(feat, z_ev)
    ref_ret = reward(a_ref_ev, a_star_ev).mean()
    opt_ret = 1.0

    def det_return():
        mu = a_ref_ev + phi_ev @ Theta.T
        return reward(mu, a_star_ev).mean()

    def grad_once(gen):
        """재매개변수화 기울기: a = μ + σε, ∇_μ E[r(a)] ≈ mean ∇_a r(a). 논문 식 (5)가 Q의 행동 기울기를 쓰는 것과 같은 꼴.
        여기서 Q 대신 실제 보상의 행동 기울기를 씀(오라클). 따라서 critic 학습 문제는 검증하지 않는다."""
        _, z, a_star, a_ref = sample_contexts(w, BATCH, gen)
        phi = phi_of(feat, z)
        mu = a_ref + phi @ Theta.T
        a = mu + SIG * gen.normal(size=mu.shape)
        r = reward(a, a_star)
        grad_a = -(r[:, None]) * (a - a_star) / TAU ** 2
        g = grad_a.T @ phi / BATCH
        if anchor:
            g = g - beta_m * 2 * ((mu - a_ref).T @ phi) / BATCH
        return g

    # 초기 기울기 분산 (Theta = 0)
    gen0 = np.random.default_rng(seed + 123)
    gs = np.stack([grad_once(gen0) for _ in range(300)])
    grad_trace = float(gs.reshape(len(gs), -1).var(axis=0).sum())

    curve_x, curve_y = [0], [det_return()]
    episodes = 0
    for u in range(1, UPDATES + 1):
        g = grad_once(rng)
        Theta += lr * g
        episodes += BATCH
        if u % EVAL_EVERY == 0:
            curve_x.append(episodes)
            curve_y.append(det_return())
    curve_y = np.array(curve_y)
    target = ref_ret + 0.9 * (opt_ret - ref_ret)
    hit = np.where(curve_y >= target)[0]
    samples90 = float(curve_x[hit[0]]) if len(hit) else float("nan")
    return dict(curve_x=curve_x, curve_y=curve_y.tolist(), samples90=samples90,
                final=float(curve_y[-1]), ref=float(ref_ret), grad_trace=grad_trace)


def sweep_lr(method):
    """튜닝 시드에서 학습률별 최종 보상(선택 기준)과 90% 도달 에피소드(참고용)를 기록한다."""
    best = None
    table, s90_table = {}, {}
    for lr in LRS:
        runs = [run(method, lr, s) for s in TUNE_SEEDS]
        table[str(lr)] = float(np.mean([r["final"] for r in runs]))
        ok = [r["samples90"] for r in runs if not np.isnan(r["samples90"])]
        s90_table[str(lr)] = dict(median=float(np.median(ok)) if ok else None,
                                  reached=len(ok) / len(runs))
        if best is None or table[str(lr)] > table[str(best)]:
            best = lr
    return best, table, s90_table


def summarize(vals):
    v = np.asarray(vals, float)
    n = len(v)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                ci95=float(1.96 * v.std(ddof=1) / np.sqrt(n)), n=n)


def main():
    t0 = time.time()
    methods = {
        "full": ("full", E),
        "tok16": ("tok", 16),
        "tok16+anchor": ("tokA", 16),
        "tok4+anchor": ("tokA", 4),
        "tok8+anchor": ("tokA", 8),
        "tok32+anchor": ("tokA", 32),
        "tok16+anchor(b=0.05)": ("tokA", 16, 0.05),
    }
    results = {"config": dict(K=K, NUI=NUI, E=E, D=D, SIG=SIG, TAU=TAU, BATCH=BATCH,
                              UPDATES=UPDATES, BETA=BETA, LRS=LRS,
                              TUNE_SEEDS=len(TUNE_SEEDS), EVAL_SEEDS=len(EVAL_SEEDS)),
               "methods": {}}
    chosen = {}
    for name, meth in methods.items():
        best, table, s90_table = sweep_lr(meth)
        chosen[name] = best
        runs = [run(meth, best, s) for s in EVAL_SEEDS]
        finals = [r["final"] for r in runs]
        s90 = [r["samples90"] for r in runs]
        reached = np.mean([not np.isnan(x) for x in s90])
        s90_ok = [x for x in s90 if not np.isnan(x)]
        results["methods"][name] = dict(
            lr=best, lr_tuning_mean_final=table, lr_tuning_samples90=s90_table,
            final=summarize(finals),
            ref_return=float(np.mean([r["ref"] for r in runs])),
            samples90=summarize(s90_ok) if s90_ok else None,
            frac_reached_90=float(reached),
            grad_trace_init=summarize([r["grad_trace"] for r in runs]),
            curve_x=runs[0]["curve_x"],
            curve_mean=np.mean([r["curve_y"] for r in runs], axis=0).tolist(),
            curve_sd=np.std([r["curve_y"] for r in runs], axis=0, ddof=1).tolist(),
            finals=finals, samples90_all=s90,
        )

    # 대비 검정: 전체 헤드 vs tok16+anchor, tok16 (무묶음) vs tok16+anchor, 최종 보상
    def welch(a, b):
        t, p = stats.ttest_ind(a, b, equal_var=False)
        return dict(t=float(t), p=float(p))

    M = results["methods"]
    results["tests"] = {
        "final_full_vs_tok16anchor": welch(M["full"]["finals"], M["tok16+anchor"]["finals"]),
        "final_tok16_vs_tok16anchor": welch(M["tok16"]["finals"], M["tok16+anchor"]["finals"]),
        "grad_trace_full_vs_tok16anchor": welch(
            [r for r in [run(methods["full"], chosen["full"], s)["grad_trace"] for s in EVAL_SEEDS[:20]]],
            [r for r in [run(methods["tok16+anchor"], chosen["tok16+anchor"], s)["grad_trace"] for s in EVAL_SEEDS[:20]]]),
    }
    results["runtime_sec"] = round(time.time() - t0, 1)
    (OUT / "toy_readout_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    for name in methods:
        r = M[name]
        s90 = r["samples90"]
        s90txt = f"{s90['mean']:.0f}" if s90 else "-"
        print(f"{name:14s} lr={r['lr']:<5} final={r['final']['mean']:.3f}±{r['final']['ci95']:.3f} "
              f"ref={r['ref_return']:.3f} s90={s90txt} reached={r['frac_reached_90']:.2f} "
              f"gradtr={r['grad_trace_init']['mean']:.3g}")
    print(results["tests"])
    print("runtime", results["runtime_sec"], "s")


if __name__ == "__main__":
    main()
