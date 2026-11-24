"""AnyRank 핵심 주장 소형 검증 (자체 e-process 구현, N-SCORE 아님). 실행: python3 analysis/toy_anyrank.py (약 1분)
쌍별 차이 d=x_a-x_b in [-1,1]에 대해 고정 lambda 격자 혼합 베팅 e-process로 H0: mu_a<=mu_b 검정(양방향),
그래프 기반 위험 재배분(Holm형, gamma=0)과 Bonferroni, 표본 선택 규칙(균등/논문 식4/코드)을 비교한다.
출력: analysis/toy_anyrank_results.json
"""
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LAM = np.linspace(0.05, 0.95, 19)
ALPHA = 0.1


class Pair:
    def __init__(self):
        self.logE = np.zeros((2, len(LAM)))  # 방향별 lambda당 log e
        self.ctr = 0
        self.maxE = 1.0

    def update(self, d):
        for s in (0, 1):
            self.logE[s] += np.log1p(LAM * (d if s == 0 else -d))
        e = max(np.logaddexp.reduce(self.logE[0]) - math.log(len(LAM)), np.logaddexp.reduce(self.logE[1]) - math.log(len(LAM)))
        self.maxE = max(self.maxE, math.exp(e))


def run(data, mode, sampling, rng, nmax, alpha=ALPHA):
    M = len(data)
    hyp = list(combinations(range(M), 2))
    K = len(hyp)
    pairs = [Pair() for _ in hyp]
    cnt = [0] * M
    alpha_i = np.full(K, alpha / K)
    W = np.ones((K, K)) / (K - 1) - np.eye(K) / (K - 1)
    done = np.zeros(K, bool)
    total = 0
    while not done.all() and max(cnt) < nmax:
        act = ~done
        if sampling == "uniform":
            cand = sorted({j for i in np.where(act)[0] for j in hyp[i]})
            j = min(cand, key=lambda x: cnt[x])
        else:
            if sampling == "paper":
                ph = np.zeros(M)
                for i in np.where(act)[0]:
                    a, b = hyp[i]
                    lo = a if cnt[a] <= cnt[b] else b
                    ph[lo] = max(ph[lo], alpha_i[i])
                w = np.exp(ph)
                j = rng.choice(M, p=w / w.sum())
            else:  # code
                w = np.where(act, np.exp(2 * alpha_i) - 1, 0.0)
                i = rng.choice(K, p=w / w.sum())
                a, b = hyp[i]
                j = a if cnt[a] <= cnt[b] else b
        cnt[j] += 1
        total += 1
        for i in np.where(act)[0]:
            a, b = hyp[i]
            if j in (a, b):
                m = min(cnt[a], cnt[b])
                while pairs[i].ctr < m:
                    k = pairs[i].ctr
                    pairs[i].update(data[a][k] - data[b][k])
                    pairs[i].ctr += 1
        # 기각 판정과 위험 재배분
        while True:
            c = [i for i in range(K) if not done[i] and pairs[i].maxE >= 1 / alpha_i[i]]
            if not c:
                break
            i = c[0]
            done[i] = True
            if mode == "graphical":
                for jj in range(K):
                    if not done[jj]:
                        alpha_i[jj] += alpha_i[i] * W[i, jj]
                Wn = np.zeros_like(W)
                for k in range(K):
                    for jj in range(K):
                        if k != jj and not done[k] and not done[jj]:
                            den = 1 - W[k, i] * W[i, k]
                            Wn[k, jj] = (W[k, jj] + W[k, i] * W[i, jj]) / den if den > 1e-9 else 0
                W = Wn
            alpha_i[i] = 0 if mode == "graphical" else alpha_i[i]
    return int(done.sum()), K, total


def main():
    d = pd.read_csv(ROOT / "repo/data/roboarena/data2/per_trial_progress_data.csv")
    names = ["pi0_fast_droid", "paligemma_diffusion_droid", "pi0_droid", "paligemma_binning_droid"]
    cols = {k: d[k].dropna().values for k in names}
    nmax = 1000
    rng = np.random.default_rng(0)
    res = {}
    for label, mode, sampling in [("bonferroni_uniform", "bonferroni", "uniform"), ("graphical_uniform", "graphical", "uniform"),
                                  ("graphical_paper_eq4", "graphical", "paper"), ("graphical_code_rule", "graphical", "code")]:
        out = []
        for r in range(20):
            data = [rng.permutation(cols[k])[:nmax] for k in names]
            out.append(run(data, mode, sampling, rng, nmax))
        dec = [o[0] for o in out]
        tot = [o[2] for o in out]
        res[label] = dict(runs=20, K=out[0][1], mean_decided=float(np.mean(dec)), mean_total_samples=float(np.mean(tot)),
                          full_ranking_runs=int(sum(x == out[0][1] for x in dec)))
        print(label, res[label])
    # 귀무 점검: 모든 정책이 같은 Bernoulli(0.5) 분포. 어느 쌍이든 판정하면 오류.
    fw = {}
    for mode in ["bonferroni", "graphical"]:
        bad = 0
        R = 150
        for r in range(R):
            data = [rng.binomial(1, 0.5, 300).astype(float) for _ in range(3)]
            dn, K, _ = run(data, mode, "uniform", rng, 300)
            bad += dn > 0
        fw[mode] = dict(runs=R, any_decision_rate=bad / R, note="동률이라 어떤 판정도 오류. ranking_no_ties 모드는 동률에서 2*alpha=0.2까지 허용")
    res["null_check"] = fw
    print(fw)
    (ROOT / "analysis/toy_anyrank_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
