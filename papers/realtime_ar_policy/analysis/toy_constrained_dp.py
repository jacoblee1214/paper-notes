"""토이: 논문 §3.3의 DP 기반 제약 디코딩을 합성 어휘에서 독립 구현해 세 가지를 검사한다. CPU, numpy, 수 초.

  K1. 논문 부록 A.1의 벡터화 DP 표(그림 A)가 완전탐색 도달 가능성과 같은 표를 만드는가
  K2. 표로 로짓을 가리면 항상 "B 토큰 이하, 용량 C를 정확히 채우는" 시퀀스가 나오는가 (제약 없는 디코딩의 유효율과 비교)
  K3. 단계마다 허용 토큰으로 다시 정규화한 분포(국소 정규화, 논문 §3.3 "제약된 지지 집합 위의 정확한 가능도")가
      "유효 시퀀스 조건부 확률"(전역 정규화)과 얼마나 다른가 (총변동 거리). best-of-N을 가능도로 고르는 근거와 관련 있다.

이것은 학습된 정책이 아니라 무작위 언어 모델(문맥별 무작위 로짓)이므로 논문의 성공률에 대한 증거가 아니다.
표 구성의 정당성과 국소/전역 정규화 차이의 크기만 본다.
실행: python3 analysis/toy_constrained_dp.py   → analysis/toy_constrained_dp_results.json
"""
import itertools
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def build_table_paper(B, C, C_v_values):
    """논문 그림 A의 construct_DP_table. 반환 T[i, c] = 용량 c를 i단계 이하로 채울 수 있는가 (누적 OR 이후)."""
    cv = sorted(set(C_v_values))
    T = np.zeros((B + 1, C + 1), dtype=bool)
    T[0, 0] = True
    for i in range(1, B + 1):
        for c in cv:
            if c <= C:
                if c == 0:
                    T[i, :] |= T[i - 1, :]
                else:
                    T[i, c:] |= T[i - 1, :-c]
    return np.logical_or.accumulate(T, axis=0)


def build_table_brute(B, C, C_v_values):
    """완전탐색: i단계 이하에 합이 정확히 c가 되는 토큰 열이 있는가 (집합 도달 가능성)."""
    cv = list(C_v_values)
    reach = [set([0])]
    for _ in range(B):
        nxt = set(reach[-1])
        for c in reach[-1]:
            for v in cv:
                if c + v <= C:
                    nxt.add(c + v)
        reach.append(nxt)
    T = np.zeros((B + 1, C + 1), dtype=bool)
    for i, s in enumerate(reach):
        for c in s:
            T[i, c] = True
    return T


def k1_table_equivalence(rng, trials=300):
    ok = 0
    for _ in range(trials):
        V = int(rng.integers(3, 9))
        cvs = rng.integers(0, 6, size=V)          # C_v = 0 인 토큰(연속 바이트만 있는 토큰)도 포함
        B = int(rng.integers(2, 9)); C = int(rng.integers(3, 20))
        ok += bool(np.array_equal(build_table_paper(B, C, cvs), build_table_brute(B, C, cvs)))
    return {"trials": trials, "equal": ok}


class RandomLM:
    """문맥(지금까지의 토큰 열)마다 고정된 무작위 로짓을 주는 가짜 언어 모델."""
    def __init__(self, V, seed, temp=1.0):
        self.V = V; self.seed = seed; self.temp = temp

    def logits(self, prefix):
        h = hash((self.seed,) + tuple(prefix)) & 0xFFFFFFFF
        g = np.random.default_rng(h)
        return g.normal(0, 1.5, size=self.V) / self.temp


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def sample_constrained(lm, cvs, T, B, C, rng):
    """표로 가린 디코딩. 단계 j에서 'T[B-j, C - used - C_v] 가 참'인 토큰만 허용 (논문 §3.3 식)."""
    seq, used = [], 0
    for j in range(1, B + 1):
        if used == C:
            break
        lg = lm.logits(seq)
        allow = np.array([(C - used - cvs[v] >= 0) and T[B - j, C - used - cvs[v]] for v in range(lm.V)])
        if not allow.any():
            return None
        p = softmax(np.where(allow, lg, -1e9)); p = p * allow; p /= p.sum()
        v = int(rng.choice(lm.V, p=p))
        seq.append(v); used += int(cvs[v])
    return seq if used == C else None


def sample_unconstrained(lm, cvs, B, C, rng):
    """제약 없이 B 토큰까지 디코딩. 유효 = 용량을 정확히 채움 (초과하면 무효)."""
    seq, used = [], 0
    for j in range(B):
        if used >= C:
            break
        v = int(rng.choice(lm.V, p=softmax(lm.logits(seq))))
        seq.append(v); used += int(cvs[v])
    return seq if used == C else None


def k2_validity(rng, n=3000):
    out = {}
    for name, (B, C, V) in {"LIBERO형(B=11, C=28)": (11, 28, 40), "DROID형(B=20, C=48)": (20, 48, 40)}.items():
        # BPE처럼 짧은 토큰이 많고 긴 병합 토큰이 드문 C_v 분포 (합성)
        cvs = rng.choice([1, 2, 3, 4, 5, 6], size=V, p=[0.35, 0.25, 0.18, 0.12, 0.06, 0.04])
        T = build_table_paper(B, C, cvs)
        lm = RandomLM(V, seed=int(rng.integers(1 << 30)))
        t0 = time.perf_counter(); build_table_paper(B, C, cvs); t_table = time.perf_counter() - t0
        valid_c = valid_u = 0
        for _ in range(n):
            s = sample_constrained(lm, cvs, T, B, C, rng); valid_c += s is not None
            s = sample_unconstrained(lm, cvs, B, C, rng); valid_u += s is not None
        out[name] = {"constrained_valid_rate": valid_c / n, "unconstrained_valid_rate": valid_u / n, "table_shape": list(T.shape),
                     "table_build_ms": round(t_table * 1000, 3), "feasible_at_all": bool(T[B, C])}
    return out


def k3_local_vs_global(rng, trials=60):
    """작은 문제(B, C, V)에서 모든 유효 시퀀스를 열거해 두 분포를 비교한다."""
    tvs, kls = [], []
    for _ in range(trials):
        V = 4; B = 4; C = 6
        cvs = rng.integers(1, 4, size=V)
        T = build_table_paper(B, C, cvs)
        if not T[B, C]:
            continue
        lm = RandomLM(V, seed=int(rng.integers(1 << 30)))
        # 유효 시퀀스 열거: 합이 정확히 C 이고 길이 <= B, 중간에 C를 넘지 않음
        valid = []
        def rec(seq, used):
            if used == C:
                valid.append(tuple(seq)); return
            if len(seq) == B:
                return
            for v in range(V):
                if used + cvs[v] <= C:
                    rec(seq + [v], used + int(cvs[v]))
        rec([], 0)
        # 전역: p(seq | 유효) ∝ ∏ softmax(전체 어휘)
        glob = []
        for s in valid:
            lp = 0.0
            for j, v in enumerate(s):
                lp += np.log(softmax(lm.logits(s[:j]))[v])
            glob.append(lp)
        glob = np.exp(np.array(glob) - max(glob)); glob /= glob.sum()
        # 국소: 단계마다 도달 가능한 토큰만 남겨 다시 정규화
        loc = []
        for s in valid:
            lp, used = 0.0, 0
            for j, v in enumerate(s):
                allow = np.array([(C - used - cvs[u] >= 0) and T[B - (j + 1), C - used - cvs[u]] for u in range(V)])
                p = softmax(np.where(allow, lm.logits(s[:j]), -1e9)) * allow; p /= p.sum()
                lp += np.log(p[v]); used += int(cvs[v])
            loc.append(lp)
        loc = np.exp(np.array(loc));
        mass = loc.sum()
        loc = loc / loc.sum()
        tvs.append(0.5 * np.abs(loc - glob).sum())
        kls.append(float(np.sum(glob * (np.log(glob + 1e-300) - np.log(loc + 1e-300)))))
        # 국소 분포의 총 질량은 1이어야 한다(표가 막다른 길을 없애므로)
        assert abs(mass - 1) < 1e-6, mass
        # 최대 가능도 시퀀스가 같은가
    return {"n_problems": len(tvs), "tv_mean": round(float(np.mean(tvs)), 4), "tv_median": round(float(np.median(tvs)), 4), "tv_max": round(float(np.max(tvs)), 4),
            "kl_global_to_local_mean": round(float(np.mean(kls)), 4)}


def k3b_argmax_agreement(rng, trials=200):
    """가능도 최대 시퀀스(best-of-N의 N→∞ 극한)가 국소와 전역에서 같은 비율."""
    same = tot = 0
    for _ in range(trials):
        V = 5; B = 4; C = 6
        cvs = rng.integers(1, 4, size=V)
        T = build_table_paper(B, C, cvs)
        if not T[B, C]:
            continue
        lm = RandomLM(V, seed=int(rng.integers(1 << 30)))
        valid = []
        def rec(seq, used):
            if used == C:
                valid.append(tuple(seq)); return
            if len(seq) == B:
                return
            for v in range(V):
                if used + cvs[v] <= C:
                    rec(seq + [v], used + int(cvs[v]))
        rec([], 0)
        g_scores, l_scores = [], []
        for s in valid:
            lg, ll, used = 0.0, 0.0, 0
            for j, v in enumerate(s):
                lg += np.log(softmax(lm.logits(s[:j]))[v])
                allow = np.array([(C - used - cvs[u] >= 0) and T[B - (j + 1), C - used - cvs[u]] for u in range(V)])
                p = softmax(np.where(allow, lm.logits(s[:j]), -1e9)) * allow; p /= p.sum()
                ll += np.log(p[v]); used += int(cvs[v])
            g_scores.append(lg); l_scores.append(ll)
        tot += 1; same += int(np.argmax(g_scores) == np.argmax(l_scores))
    return {"n_problems": tot, "same_argmax": same}


def main():
    rng = np.random.default_rng(0)
    res = {"K1_table_equals_bruteforce": k1_table_equivalence(rng), "K2_validity": k2_validity(rng),
           "K3_local_vs_global_normalization": k3_local_vs_global(rng), "K3b_argmax": k3b_argmax_agreement(rng)}
    (ROOT / "analysis" / "toy_constrained_dp_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
