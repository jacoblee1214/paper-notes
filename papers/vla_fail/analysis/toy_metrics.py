"""토이 1·2: 논문의 AUCPR/AUCPDT 정의(4.1절, 부록 A)를 합성 에피소드에 적용해 두 가지를 본다.

T1  롤아웃 표본 크기에서 오는 AUCPR·AUCPDT의 부트스트랩 표준편차.
    논문 표 1의 95% 구간은 "같은 롤아웃에 시드 3개"만 반영하므로, 롤아웃을 다시 뽑을 때의 변동은 따로 봐야 한다.
    실패·성공 롤아웃 수는 논문 표 15·16의 ID+OOD 수와 성공률에서 복원한 값이다(시행 수 가정).
T2  신호가 없는 검출기(점수가 에피소드와 무관)의 AUCPR과 AUCPDT.
    논문 부록 A는 이때 PDT가 구간 전체에서 1이라고 쓴다. 정의대로 구현하면 AUCPDT = 1 - 실패율이어야 한다.

가정(합성): 에피소드 길이 T = 100(정규화 시간 [0, 1]), 성공 에피소드 점수는 N(0,1) 잡음의 AR(1)(계수 0.7),
실패 에피소드는 시작 시각 U(0.2, 0.8)T 이후 평균이 delta 만큼 올라간다. 점수는 시각별 원점수이고 임계값을 한 번 넘으면 탐지한다.
실행: python3 analysis/toy_metrics.py  (numpy 필요, 약 1분)
"""
import json
from pathlib import Path

import numpy as np

A = Path(__file__).resolve().parent
rng = np.random.default_rng(0)
T = 100


def make_episodes(P, N, delta, rho=0.7, rng=rng):
    n = P + N
    y = np.zeros(n, bool)
    y[:P] = True
    onset = rng.uniform(0.2, 0.8, size=n) * T
    eps = rng.normal(size=(n, T))
    s = np.zeros((n, T))
    s[:, 0] = eps[:, 0]
    for t in range(1, T):
        s[:, t] = rho * s[:, t - 1] + np.sqrt(1 - rho**2) * eps[:, t]
    shift = (np.arange(T)[None, :] >= onset[:, None]) * delta
    s = s + shift * y[:, None]
    return y, s


def curves(y, s):
    """임계값마다 (정밀도, 재현율, PDT)를 계산한다. 에피소드 점수는 시간 최댓값으로 첫 탐지를 정의한다."""
    n, Tn = s.shape
    ep_max = s.max(axis=1)
    # 후보 임계값: 모든 시각의 점수에서 서로 다른 값(큰 값부터)
    taus = np.unique(np.quantile(s, np.linspace(0.0, 1.0, 400)))
    P = y.sum()
    prec, rec, pdt = [], [], []
    for tau in taus:
        hit = s >= tau
        det = hit.any(axis=1)
        first = np.where(det, hit.argmax(axis=1) / Tn, 1.0)
        tp = (det & y).sum()
        fp = (det & ~y).sum()
        prec.append(tp / (tp + fp) if tp + fp else 1.0)
        rec.append(tp / P)
        pdt.append(first[y].mean())
    return np.array(taus), np.array(prec), np.array(rec), np.array(pdt)


def aucpr(prec, rec):
    """재현율 기준 계단 적분(평균 정밀도). 재현율 0에서 시작."""
    order = np.argsort(rec)
    r, p = rec[order], prec[order]
    # 재현율마다 그 이상에서 얻을 수 있는 최대 정밀도(보간 PR 곡선의 계단 형태)
    pmax = np.maximum.accumulate(p[::-1])[::-1]
    r_full = np.concatenate([[0.0], r])
    return float(np.sum(np.diff(r_full) * pmax))


def aucpdt(prec, pdt, pi):
    """부록 A: 정밀도 구간 [pi, 1]에서 파레토 임계값만 남기고 오른쪽 리만 합. P_K = 1, 마지막 PDT = 1."""
    pts = sorted(zip(prec, pdt))
    # 파레토: 같은 정밀도 이상에서 더 작은 PDT만 남긴다(정밀도 내림차순 훑기)
    front, best = [], 1e9
    for p, d in sorted(pts, key=lambda x: -x[0]):
        if d < best - 1e-12:
            front.append((p, d))
            best = d
    front = sorted(front)
    ps = [pi] + [p for p, _ in front if p > pi] + [1.0]
    ds = [None] + [d for p, d in front if p > pi] + [1.0]
    area = 0.0
    for k in range(1, len(ps)):
        area += (ps[k] - ps[k - 1]) * ds[k]
    return float(area)


def metrics(y, s):
    taus, prec, rec, pdt = curves(y, s)
    pi = y.mean()
    return aucpr(prec, rec), aucpdt(prec, pdt, pi)


def calibrate_delta(P, N, target):
    lo, hi = 0.0, 6.0
    for _ in range(14):
        mid = (lo + hi) / 2
        y, s = make_episodes(P, N, mid, rng=np.random.default_rng(1))
        v = metrics(y, s)[0]
        if v < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def boot_sd(y, s, B=60, seed=2):
    r = np.random.default_rng(seed)
    n = len(y)
    out = []
    for _ in range(B):
        idx = r.integers(0, n, n)
        yy, ss = y[idx], s[idx]
        if yy.sum() in (0, n):
            continue
        out.append(metrics(yy, ss))
    out = np.array(out)
    return out.std(axis=0, ddof=1), out.mean(axis=0)


def main():
    res = {"T1": [], "T2": []}
    # T1: (이름, 실패 수 P, 성공 수 N) 논문 표 15·16 의 ID+OOD 수와 성공률에서 복원(성공 수 = 성공률 x 롤아웃, 보정 20개 포함 가정 H_b 와 H_a 평균 크기)
    cases = [
        ("X-VLA Kitchen (n=62)", 28, 34),
        ("pi0.5 Blocks (n=75)", 33, 42),
        ("pi0.5 Mixer (n=80)", 29, 51),
        ("X-VLA Blocks (n=84)", 45, 39),
        ("X-VLA Cups (n=105)", 49, 56),
        ("pi0.5 Cups (n=235)", 74, 161),
    ]
    for name, P, N in cases:
        for target in (0.85,):
            d = calibrate_delta(P, N, target)
            y, s = make_episodes(P, N, d, rng=np.random.default_rng(3))
            (sd_pr, sd_pdt), (m_pr, m_pdt) = boot_sd(y, s)
            res["T1"].append({"case": name, "P": P, "N": N, "delta": round(d, 2), "AUCPR_mean": round(float(m_pr), 3),
                              "AUCPR_boot_sd": round(float(sd_pr), 3), "AUCPDT_mean": round(float(m_pdt), 3),
                              "AUCPDT_boot_sd": round(float(sd_pdt), 3), "approx_95_halfwidth_AUCPR": round(1.96 * float(sd_pr), 3)})
    # T2: 신호 없는 검출기
    for pi in (0.29, 0.54, 0.79):
        n = 400
        P = int(round(pi * n))
        r = np.random.default_rng(5)
        y = np.zeros(n, bool)
        y[:P] = True
        eps = r.normal(size=(n, T))
        s = np.zeros((n, T))
        s[:, 0] = eps[:, 0]
        for t in range(1, T):
            s[:, t] = 0.7 * s[:, t - 1] + np.sqrt(1 - 0.49) * eps[:, t]
        pr, pdt = metrics(y, s)
        res["T2"].append({"failure_rate": pi, "AUCPR_no_signal": round(pr, 3), "AUCPDT_no_signal": round(pdt, 3), "one_minus_pi": round(1 - pi, 3)})
    json.dump(res, open(A / "toy_metrics_results.json", "w"), ensure_ascii=False, indent=1)
    for r_ in res["T1"]:
        print(r_)
    for r_ in res["T2"]:
        print(r_)


if __name__ == "__main__":
    main()
