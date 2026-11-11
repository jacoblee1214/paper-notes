"""추가 검증: 체인 toy를 3–5단계로 넓히고, 핸드오프 품질 격차 Δ와 예측기 표본 수 n을 바꿔 곱셈 foresight의 이득을 잰다 (numpy, CPU).

실행: python3 analysis/extra_chain_sweep.py  → analysis/extra_chain_sweep_results.json

shaping_toy.py와 같은 모형을 단계 수 L로 일반화했다(정책은 단계별 softmax 표, 상태 = 직전 행동, 실패하면 종료).
- 단계 k 성공 확률: P_k(s, a) = SIG[k mod 3][a] · Q_{k−1}(s)   (k ≥ 1; 첫 단계는 SIG만). 원 toy의 SIG 세 행을 순환해 쓴다.
- 핸드오프 품질: Q_k(a) = mean + (Q_base[k mod 2](a) − mean) · Δ / 0.6. Δ = 0.6이 원 toy와 같고, Δ = 0이면 모든 행동의 핸드오프가 같다.
- 경로는 모두 열거한다(L = 5일 때 약 1,400개). 단계별 기대 기울기를 정확히 계산하고, 하류는 고정한다(Alg. 1 형식, 뒤에서부터).
- 비교 방식: 상수 기준선(단계 성공 지표), 전역 희소 보상, foresight 곱셈(V 정확), foresight 곱셈(표본 n개로 추정한 V̂).
  고정형(fixed): 예측 오차를 학습 시작에 한 번 뽑아 끝까지 유지한다(sd = 0.5/√n). 학습된 예측기의 고정 오차에 대응한다.
  재표본형(resample): 매 반복 Binomial(n, V)/n을 새로 뽑는다. 반복마다 잡음이 평균으로 사라져 이득 차이가 작다.
- 초기값 8개, 단계당 400 반복. 참 최적 성공률은 역방향 DP로 구한다.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A = 4
SIG_ROWS = np.array([[0.98, 0.92, 0.85, 0.75], [0.95, 0.90, 0.85, 0.70], [0.99, 0.97, 0.90, 0.80]])
Q_ROWS = np.array([[0.35, 0.60, 0.80, 0.95], [0.40, 0.65, 0.85, 0.97]])
LAYERS = (3, 4, 5)
GAPS = (0.0, 0.15, 0.3, 0.6)
NS = (5, 20, 100, 1000)
LR, ITERS, N_INIT = 5.0, 400, 8


def make_model(L, delta):
    SIG = np.array([SIG_ROWS[k % 3] for k in range(L)])
    Q = []
    for k in range(L - 1):
        base = Q_ROWS[k % 2]
        Q.append(base.mean() + (base - base.mean()) * delta / 0.6)
    Q = np.array(Q)
    P = np.zeros((L, A, A))
    P[0] = SIG[0][None, :]
    for k in range(1, L):
        P[k] = SIG[k][None, :] * Q[k - 1][:, None]
    return P


def leaves(L):
    """전체 경로 열거. 반환: states, actions, x (성공 지표, 실행 안 된 단계는 0), executed (bool), R (전역 성공)."""
    rows = []

    def rec(k, prev, st, ac, xs):
        for a in range(A):
            s_k = prev
            for x in (0, 1):
                if x == 0:
                    rows.append((st + [s_k], ac + [a], xs + [0]))
                elif k == L - 1:
                    rows.append((st + [s_k], ac + [a], xs + [1]))
                else:
                    rec(k + 1, a, st + [s_k], ac + [a], xs + [1])

    rec(0, 0, [], [], [])
    n = len(rows)
    ST = np.zeros((n, L), int); AC = np.zeros((n, L), int); XS = np.zeros((n, L)); TK = np.zeros((n, L), bool)
    for i, (st, ac, xs) in enumerate(rows):
        m = len(st)
        ST[i, :m] = st; AC[i, :m] = ac; XS[i, :m] = xs; TK[i, :m] = True
    R = (TK.all(1) & (XS == 1).all(1)).astype(float)
    return ST, AC, XS, TK, R


class Chain:
    def __init__(self, L, delta):
        self.L = L
        self.P = make_model(L, delta)
        self.ST, self.AC, self.XS, self.TK, self.R = leaves(L)
        self.OH = np.eye(A)[self.AC]
        self.PSUCC = np.stack([self.P[k][self.ST[:, k], self.AC[:, k]] for k in range(L)], 1)
        self.gap = float(delta)

    def softmax_all(self, th):
        return [np.exp(t - t.max(1, keepdims=True)) / np.exp(t - t.max(1, keepdims=True)).sum(1, keepdims=True) for t in th]

    def leaf_prob(self, pi):
        pol = np.stack([pi[k][self.ST[:, k], self.AC[:, k]] for k in range(self.L)], 1)
        out = np.where(self.XS == 1, self.PSUCC, 1 - self.PSUCC)
        return np.prod(np.where(self.TK, pol * out, 1.0), 1)

    def success(self, pi):
        return float(np.sum(self.leaf_prob(pi) * self.R))

    def values_downstream(self, pi):
        """V[k][h]: 단계 k 시작, 직전 행동 h, 단계 k 이후 전부 성공 확률 (하류 정책 pi 고정)."""
        V = [None] * (self.L + 1)
        V[self.L] = np.ones(A)
        for k in range(self.L - 1, -1, -1):
            V[k] = (pi[k] * self.P[k] * V[k + 1][None, :]).sum(1)
        return V

    def grad_stage(self, pi, G_k, k):
        w = self.leaf_prob(pi)
        m = self.TK[:, k]
        g = np.zeros((A, A))
        contrib = (w * G_k)[m, None] * (self.OH[m, k, :] - pi[k][self.ST[m, k]])
        np.add.at(g, self.ST[m, k], contrib)
        return g

    def optimum(self):
        V = np.ones(A)
        for k in range(self.L - 1, -1, -1):
            V = np.max(self.P[k] * V[None, :], axis=1)
        return float(V[0])

    def train(self, scheme, n, init, rng, fixed=False):
        L = self.L
        theta = [t.copy() for t in init]
        X = self.XS
        # 고정 예측기: 예측 오차를 학습 시작 때 한 번 뽑아 끝까지 유지한다. sd = 0.5/√n (p = 0.5 최악값, 03 §2.5의 K_rep 잡음과 같은 식)
        fixed_err = [rng.normal(0, 0.5 / np.sqrt(n), A) if (fixed and n is not None) else None for _ in range(L)]
        for k in range(L - 1, -1, -1):
            for _ in range(ITERS):
                pi = self.softmax_all(theta)
                if scheme == "constant":
                    G_k = X[:, k]
                elif scheme == "sparse":
                    G_k = self.R
                else:
                    V = self.values_downstream(pi)
                    Vn = V[k + 1]
                    if n is None:
                        Vh = Vn
                    elif fixed:
                        Vh = np.clip(Vn + fixed_err[k], 0, 1)
                    else:
                        Vh = rng.binomial(n, np.clip(Vn, 0, 1)) / n
                    G_k = X[:, k] * Vh[self.AC[:, k]]
                theta[k] = theta[k] + LR * self.grad_stage(pi, G_k, k)
        return self.success(self.softmax_all(theta))


SCHEMES = [("constant", "constant", None), ("sparse_global", "sparse", None), ("foresight_exact", "foresight", None)] + \
          [(f"foresight_fixed_n{n}", "foresight_fixed", n) for n in NS] + [(f"foresight_resample_n{n}", "foresight", n) for n in NS]


def job(args):
    L, delta, name, scheme, n, i = args
    fixed = scheme == "foresight_fixed"
    scheme = "foresight" if scheme.startswith("foresight") else scheme
    chain = Chain(L, delta)
    rng0 = np.random.default_rng(1000 + i)
    init = [rng0.normal(0, 0.1, (A, A)) for _ in range(L)]
    val = chain.train(scheme, n, init, np.random.default_rng(5000 + i), fixed=fixed)
    return L, delta, name, i, val, chain.optimum(), chain.gap


def main():
    tasks = [(L, d, name, scheme, n, i) for L in LAYERS for d in GAPS for (name, scheme, n) in SCHEMES for i in range(N_INIT)]
    with Pool(min(10, os.cpu_count() or 4)) as pool:
        res = pool.map(job, tasks)
    table = {}
    opt = {}
    gaps = {}
    for L, d, name, i, val, popt, gap in res:
        table.setdefault(f"L={L}", {}).setdefault(f"gap={d}", {}).setdefault(name, []).append(val)
        opt[f"L={L}, gap={d}"] = popt
        gaps[f"L={L}, gap={d}"] = gap
    summary = {}
    for Lk, dd in table.items():
        summary[Lk] = {}
        for dk, names in dd.items():
            key = f"{Lk}, {dk}"
            c = np.array(names["constant"]); f = np.array(names["foresight_exact"])
            row = {"optimum": round(opt[key], 4), "handoff_delta": round(gaps[key], 3)}
            for nm, vals in names.items():
                row[nm] = round(float(np.mean(vals)), 4)
            row["foresight_exact_minus_constant"] = round(float(np.mean(f - c)), 4)
            row["foresight_exact_minus_constant_sd"] = round(float(np.std(f - c, ddof=1)), 4)
            for n in NS:
                for mode in ("fixed", "resample"):
                    row[f"foresight_{mode}_n{n}_minus_constant"] = round(float(np.mean(np.array(names[f"foresight_{mode}_n{n}"]) - c)), 4)
            row["sparse_minus_constant"] = round(float(np.mean(np.array(names["sparse_global"]) - c)), 4)
            summary[Lk][dk] = row
    out = {"settings": {"A": A, "LAYERS": list(LAYERS), "GAPS": list(GAPS), "NS": list(NS), "LR": LR,
                        "ITERS_per_stage": ITERS, "N_INIT": N_INIT,
                        "note": "SIG rows cycle the 3 rows of shaping_toy.py; Q deviations scaled by gap/0.6"},
           "summary": summary}
    path = ROOT / "analysis" / "extra_chain_sweep_results.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", path)


if __name__ == "__main__":
    main()
