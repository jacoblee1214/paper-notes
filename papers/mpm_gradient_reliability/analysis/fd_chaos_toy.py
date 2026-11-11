"""FD 검사 toy (CPU numpy, 1스레드, 수십 초 이내).

목적: 논문 Table II의 "20 스텝에서 AD/FD 불일치 0.34"가 FD(절단 오차, 반올림 오차) 탓인지, AD(손으로 쓴 VJP) 탓인지
구분할 수 있는 실험 절차를 작은 모형에서 보인다. 두 모형을 쓴다.

(a) 매끄러운 혼돈 사상: x_{k+1} = a·x_k(1 - x_k), 손실 L = x_T, 대상 매개변수 a (a0 = 3.9).
    - 참값: Decimal 80자리 전진 접선(forward tangent). 입력 값은 float의 정확한 이진 값을 그대로 쓴다.
    - AD(f64): 같은 접선 recurrence를 float64로 계산.
    - AD(고장, buggy): 체인 항 a(1-2x)·d에서 계수 a를 빼먹은 손 작성 미분(가정한 버그). 불일치가 h에 무관하게 남는 '고원'이 나타난다.
    - FD(f64): 중심 차분 (L(a+h) - L(a-h)) / (실제 차이), h = 1e-2 … 1e-12.
    → 오차 대 h 곡선: h가 클 때는 절단(h²), 작을 때는 반올림(1/h)이 지배한다. 두 구간이 모두 보이면 FD가 주 원인이다.
    → 어떤 h에서도 줄지 않는 고원이 있으면 AD(또는 참조)가 틀린 것이다.

(b) 구간별 선형(클램프) 탄소성 스프링: 준정적 반복 하중 g_k, 항복력 Y(매개변수), 손실 L = Σ_k p_k (소성 변위의 누적합).
    - 항복 임계를 지날 때 ∂L/∂Y가 불연속이다(매끄럽지 않은 롤아웃). AD는 현재 분기의 미분을 준다.
      (L = Σ p_k: 소성 사건마다 기울기가 바뀐다)
    - 중심 차분은 임계가 [Y-h, Y+h] 안에 들어오는 h에서만 어긋난다. 오차가 h에 대해 계단처럼 나타나고,
      사건(임계 통과) 수가 늘면(롤아웃이 길면) 그 h 창에 임계가 들어올 가능성이 커진다.

실행: python3 analysis/fd_chaos_toy.py → analysis/fd_chaos_results.json
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import json
from decimal import Decimal, getcontext
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A0 = 3.9
X0 = 0.3
HS = [10.0 ** -e for e in range(2, 13)]
EPS64 = np.finfo(np.float64).eps


# ---------- (a) 혼돈 사상 ----------
def roll64(a, T):
    x = X0
    for _ in range(T):
        x = a * x * (1.0 - x)
    return x


def tangent64(a, T, buggy=False):
    x, d = X0, 0.0
    for _ in range(T):
        if buggy:  # 가정한 버그: 체인 항의 계수 a를 빼먹음 (대수 실수)
            d_new = x * (1.0 - x) + (1.0 - 2.0 * x) * d
        else:
            d_new = x * (1.0 - x) + a * (1.0 - 2.0 * x) * d
        x, d = a * x * (1.0 - x), d_new
    return d


def truth(T):
    getcontext().prec = 80
    a, x, d = Decimal(A0), Decimal(X0), Decimal(0)
    for _ in range(T):
        d = x * (1 - x) + a * (1 - 2 * x) * d
        x = a * x * (1 - x)
    return float(d)


def lyapunov(a, n=20000):
    x, s = 0.1234, 0.0
    for _ in range(n):
        x = a * x * (1 - x)
        s += np.log(abs(a * (1 - 2 * x)) + 1e-300)
    return s / n


def fd64(T, h):
    ap, am = A0 + h, A0 - h
    return (roll64(ap, T) - roll64(am, T)) / (ap - am)


def relerr(v, ref):
    return abs(v - ref) / abs(ref)


def chaos_part():
    lam = lyapunov(A0)
    out = {"lambda_per_step": lam, "eps64": EPS64, "rows": []}
    for T in (2, 5, 10, 20, 40):
        ref = truth(T)
        ad = tangent64(A0, T)
        bad = tangent64(A0, T, buggy=True)
        curve = {f"{h:.0e}": relerr(fd64(T, h), ref) for h in HS}
        best_h = min(curve, key=curve.get)
        out["rows"].append({
            "T": T,
            "truth_dL_da": ref,
            "ad64_relerr": relerr(ad, ref),
            "buggy_ad_relerr": relerr(bad, ref),
            "fd_relerr_by_h": curve,
            "fd_best_h": best_h,
            "fd_best_relerr": curve[best_h],
            # 모형 예측(03 문서 §1): 최소 FD 상대오차 ~ eps^(2/3) · exp(2λT/3)
            "model_min_relerr": EPS64 ** (2 / 3) * np.exp(2 * lam * T / 3),
        })
    return out


# ---------- (b) 클램프 탄소성 스프링 ----------
def clamp_path(n_steps, seed=7):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0.0, 0.35, n_steps))


def clamp_run(Y, g, K=1.0, want_ad=False):
    """L = sum_k p_k (소성 변위의 누적합). 항복 분기가 바뀔 때마다 dL/dY가 불연속으로 바뀐다."""
    p, dp, acc, dacc = 0.0, 0.0, 0.0, 0.0  # 소성 변위, dp/dY, 손실, dL/dY
    switches = []
    for k, gk in enumerate(g):
        s = K * (gk - p)
        if abs(s) > Y:
            sg = np.sign(s)
            p_new = gk - sg * Y / K
            dp_new = -sg / K
            switches.append((k, abs(s) - Y))
        else:
            p_new, dp_new = p, dp
        p, dp = p_new, dp_new
        acc += p
        dacc += dp
    return (acc, dacc, switches) if want_ad else (acc, switches)


def clamp_part(Y=1.0):
    out = {"Y": Y, "rows": []}
    for n_steps in (20, 40, 80, 160, 320):
        g = clamp_path(n_steps)
        p, ad, sw = clamp_run(Y, g, want_ad=True)
        rows = []
        for h in np.logspace(-1, -9, 9):
            pp, _ = clamp_run(Y + h, g)
            pm, _ = clamp_run(Y - h, g)
            fd = (pp - pm) / (2 * h)
            rows.append({"h": float(h), "fd": fd, "relerr_vs_ad": abs(fd - ad) / max(abs(ad), 1e-12)})
        # 임계 근접: 각 단계에서 |s| - Y 의 최소 거리(이 값보다 작은 h에서는 임계가 중심 차분 창에 들지 않는다)
        gaps = np.array([abs(abs(gk - clamp_run(Y, g[:k])[0]) - Y) for k, gk in enumerate(g)])
        big = [r["h"] for r in rows if r["relerr_vs_ad"] > 1e-3]
        out["rows"].append({
            "n_steps": n_steps,
            "first_h_with_kink_error": max(big) if big else None,
            "n_plastic_events": len(sw),
            "dL_dY_AD": ad,
            "fd_by_h": rows,
            "min_gap_to_threshold": float(gaps.min()),
        })
    return out


def main():
    res = {"chaos": chaos_part(), "clamp": clamp_part()}
    (HERE / "fd_chaos_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    c = res["chaos"]
    print(f"logistic a0={A0}: Lyapunov lambda = {c['lambda_per_step']:.4f} /step")
    print(f"{'T':>3} {'truth':>12} {'AD64 err':>10} {'buggy err':>10} {'FD best h':>10} {'FD best err':>12} {'model':>10}")
    for r in c["rows"]:
        print(f"{r['T']:>3} {r['truth_dL_da']:>12.4e} {r['ad64_relerr']:>10.2e} {r['buggy_ad_relerr']:>10.2e} "
              f"{r['fd_best_h']:>10} {r['fd_best_relerr']:>12.2e} {r['model_min_relerr']:>10.2e}")
    print("\nclamp spring (Y=1): FD vs AD relerr for h = 1e-1 ... 1e-9 (9 values)")
    for r in res["clamp"]["rows"]:
        e = " ".join(f"{x['relerr_vs_ad']:.0e}" for x in r["fd_by_h"])
        print(f"  n={r['n_steps']:>4} events={r['n_plastic_events']:>3} min_gap={r['min_gap_to_threshold']:.2e} "
              f"AD={r['dL_dY_AD']:+.1f} first_bad_h={r['first_h_with_kink_error']}  [{e}]")


if __name__ == "__main__":
    main()
