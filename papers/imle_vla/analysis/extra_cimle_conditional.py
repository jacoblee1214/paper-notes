"""추가 검증 2: 관측 조건 c에 따라 봉우리 위치가 바뀌는 목표에서 m 효과 (CPU numpy 1스레드, 시드 3개).

목표: c=0이면 반지름 1 원 위의 4봉우리(0°, 90°, 180°, 270°), c=1이면 같은 4봉우리를 45° 돌린 위치. σ=0.05.
비교 (생성기 2층 tanh MLP, Adam 2e-3, 배치 256, 3000회, m ∈ {1, 2, 5}):
- 조건부 cIMLE: 입력 [z, c], c는 0/1 중 균등 추출 (논문 식 3–4의 관측 조건과 같은 구조)
- 무조건 cIMLE: 입력 z만, 두 조건의 목표를 섞은 8봉우리 혼합에서 학습 (관측을 쓰지 않을 때)
평가: 조건별로 4000 샘플. 조건부는 c를 고정해 생성, 무조건은 같은 샘플을 두 조건 모두와 비교.
- 덮음률(해당 조건 4봉우리 기준, 반경 0.25, 질량 ≥ 5%)
- 무조건의 경우 8봉우리 전체 덮음률도 함께 보고
- 봉우리 사이 비율(가장 가까운 해당 봉우리까지 거리 > 0.3)
실행: python3 analysis/extra_cimle_conditional.py  (수 분, 1스레드)
출력: analysis/extra_cimle_conditional_results.json
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extra_cimle_geometry import init, fwd, bwd, Adam, centers, agg, ITERS, BATCH, SEEDS, NS  # noqa: E402

OUT = HERE / "extra_cimle_conditional_results.json"
MS = (1, 2, 5)
SIG = 0.05
t0 = time.time()


def modes_for(c):
    base = centers(4, 1.0)
    if c == 0:
        return base
    th = np.pi / 4
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return base @ rot.T


MU = {0: modes_for(0), 1: modes_for(1)}
MU_ALL = np.concatenate([MU[0], MU[1]], 0)


def sample_y(rng, c):
    k = rng.integers(0, 4, c.shape[0])
    centers_ = np.where(c[:, None] == 0, MU[0][k], MU[1][k])
    return centers_ + SIG * rng.normal(size=(c.shape[0], 2))


def train(rng, m, conditional):
    zin = 2 + (1 if conditional else 0)
    P = init([zin, 64, 64, 2], rng)
    opt = Adam(P)
    for _ in range(ITERS):
        c = rng.integers(0, 2, BATCH).astype(float)
        if conditional:
            y = sample_y(rng, c)
        else:
            k = rng.integers(0, 8, BATCH)
            y = MU_ALL[k] + SIG * rng.normal(size=(BATCH, 2))
        z = rng.normal(size=(BATCH * m, 2))
        if conditional:
            cz = np.repeat(c, m)[:, None]
            zz = np.concatenate([z, cz], 1)
        else:
            zz = z
        out, _ = fwd(P, zz)
        cand = out.reshape(BATCH, m, 2)
        d = ((cand - y[:, None, :]) ** 2).sum(-1)
        j = d.argmin(1)
        zb = z.reshape(BATCH, m, 2)[np.arange(BATCH), j]
        if conditional:
            zbb = np.concatenate([zb, c[:, None]], 1)
        else:
            zbb = zb
        pred, hs = fwd(P, zbb)
        dout = 2 * (pred - y) / (BATCH * 2)
        opt.step(P, bwd(P, hs, dout))
    return P


def gen(P, rng, n, c=None):
    z = rng.normal(size=(n, 2))
    if c is None:
        zz = z
    else:
        zz = np.concatenate([z, np.full((n, 1), float(c))], 1)
    return fwd(P, zz)[0]


def cov_between(s, mus):
    d_all = np.sqrt(((s[:, None, :] - mus[None]) ** 2).sum(-1))
    near = d_all.argmin(1)
    dmin = d_all.min(1)
    K = mus.shape[0]
    cov = sum(np.mean((near == k) & (dmin < 0.25)) >= 0.05 for k in range(K)) / K
    return {"coverage": float(cov), "between_frac": float(np.mean(dmin > 0.3))}


def main():
    out = {"setup": {"sigma": SIG, "iters": ITERS, "batch": BATCH, "seeds": list(SEEDS), "samples": NS,
                     "ms": list(MS)}, "methods": {}}
    for m in MS:
        cond_runs = {0: [], 1: []}
        unc_runs = {0: [], 1: []}
        unc_all = []
        for sd in SEEDS:
            rng = np.random.default_rng(100 + sd)
            Pc = train(rng, m, True)
            Pu = train(rng, m, False)
            for c in (0, 1):
                cond_runs[c].append(cov_between(gen(Pc, rng, NS, c), MU[c]))
                s = gen(Pu, rng, NS)
                unc_runs[c].append(cov_between(s, MU[c]))
                if c == 0:
                    unc_all.append(cov_between(s, MU_ALL))
        out["methods"][f"m={m}"] = {
            "conditional_c0": agg(cond_runs[0]), "conditional_c1": agg(cond_runs[1]),
            "uncond_mix_on_c0_modes": agg(unc_runs[0]), "uncond_mix_on_c1_modes": agg(unc_runs[1]),
            "uncond_mix_on_all8_modes": agg(unc_all),
        }
        print("m", m, round(time.time() - t0, 1), "s", flush=True)
    out["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", OUT.name, out["runtime_s"], "s")


if __name__ == "__main__":
    main()
