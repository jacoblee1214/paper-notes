"""식 (5)의 법선 힘 항을 스프링 접촉에 놓은 토이: 정상상태 오차가 3.5% 이내가 되려면 무엇이 필요한가.

논문 부록 8.8절은 "비례 admittance 법칙"이 1 N에서 4 N 계단 지령을 정상상태 오차 3.5% 이내로 추종한다고 보고한다(그림 8).
식 (5)는 dp_z = K_F (f_ref - f_meas) 이다. 접촉을 스프링 f = k * (x0 + dp_z) 로 두고 두 가지로 읽는다.
  A. 기억 없는 비례항: 매 틱 dp_z = K_F (f_ref - f)  (T_cmd = T_tilde * dT_t 를 그대로 읽은 것)
  B. 누적(적분)형:    dp_z <- dp_z + K_F (f_ref - f)
루프 이득 g = k * K_F 로 무차원화하고, 한 틱 지연(측정 힘이 한 틱 늦게 들어옴)을 둔다.
초기 1 N 접촉(k * x0 = 1 N), 지령 4 N 단일 계단(논문 그림 8은 1, 2, 3, 4 N의 네 단계). 로봇 관절 추종, 센서 잡음, 겔 패드의 비선형은 넣지 않는다(논문에 값이 없음).
실행: python3 analysis/toy_admittance.py   -> analysis/toy_admittance_results.json (numpy, 1초 미만)
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_admittance_results.json"


def run(g, mode, steps=4000, f0=1.0, fref=4.0):
    """g = k*K_F. 힘 f_t = k*(x0 + dp_t), 지연된 측정 f_{t-1}로 dp 갱신."""
    k = 1.0
    x0 = f0 / k
    KF = g / k
    dp = 0.0
    f_prev = f0
    fs = []
    for _ in range(steps):
        if mode == "A":
            dp = KF * (fref - f_prev)
        else:
            dp = dp + KF * (fref - f_prev)
        f = k * (x0 + dp)
        if not np.isfinite(f) or abs(f) > 1e6:
            return None, None
        fs.append(f)
        f_prev = f
    f_end = float(np.mean(fs[-200:]))
    return f_end, float(np.std(fs[-200:]))


def main():
    res = {}
    for mode in ("A", "B"):
        rows = []
        for g in (0.05, 0.1, 0.3, 0.5, 0.9, 1.0, 1.5, 3.0, 10.0, 30.0):
            f_end, sd = run(g, mode)
            if f_end is None:
                rows.append(dict(g=g, stable=False))
            else:
                err = abs(4.0 - f_end) / 4.0
                rows.append(dict(g=g, stable=bool(sd < 1e-6), f_end=round(f_end, 3), err_pct=round(100 * err, 2), tail_sd=round(sd, 4)))
        res[mode] = rows
    # 이론값: A는 f_ss = (1 + 4g)/(1+g), 오차 = 3/(4(1+g)). 안정 조건 g < 1 (한 틱 지연)
    res["A_theory_gain_needed_for_3p5pct"] = round(3 / (4 * 0.035) - 1, 1)
    res["A_stable_gain_limit"] = 1.0
    res["A_min_error_pct_in_stable_range"] = round(100 * 3 / (4 * (1 + 1.0)), 1)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for mode in ("A", "B"):
        print(mode)
        for r in res[mode]:
            print(" ", r)
    print({k: v for k, v in res.items() if k.startswith("A_")})


if __name__ == "__main__":
    main()
