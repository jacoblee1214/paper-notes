"""추가 검증 3: 편향을 증강한 표준 PF 대 RBPF (CPU, numpy 1스레드, 수 분 이내).

질문: 같은 입자 수·같은 시드·같은 장에서, RBPF의 이득은 Rao-Blackwell 구조(편향을 칼만으로 주변화)에서 오는가,
        아니면 편향 모형을 넣은 것 자체에서 오는가?
설정 (pde_toy.py의 장·경로·운동 잡음·측정 잡음 그대로; pde_toy를 import해 재사용):
  - 참 장: pde_toy.solve(80, 0.2, "asym"), 필터 지도: pde_toy.solve(40, 0.2, "asym") (일치 κ).
  - 측정: y_t = φ(p_t, t) + b + ε_t, ε ~ N(0, 0.05²). 참 편향 b ~ N(0, P_true)를 실행마다 뽑는다.
    P_true ∈ {0.01 (sd 0.1), 0.25 (sd 0.5)}.
  - 필터 세 종 (모두 N 입자, 체계적 재표본추출, N_eff < N/2일 때만 재표본추출):
      PF-nobias : 편향을 무시(pde_toy의 PF와 같은 식). 참 편향이 있으면 잘못된 모형.
      PF-bias   : 표준 PF. 입자 = (p, b). b는 사전 분포 N(0, P_0)에서 한 번 뽑아 고정. 가중치 N(y; φ+b, σ²).
      RBPF      : 입자 = p, 각 입자에 b의 칼만 상태(b̂, P)를 둔다. 가중치 = 예측 우도 N(e; 0, P + σ²),
                  e = y − φ(p) − b̂, 이후 칼만 갱신. 재표본추출 때 상태를 함께 복사한다.
    P_0 (필터의 편향 사전 분포) = P_true로 일치시킨다(사전 분포 오지정은 하지 않음).
  - 공통 난수: 실행마다 초기 입자 잡음, 운동 잡음, 재표본추출 난수, 참 b를 미리 뽑아 세 필터가 공유한다.
    N = 100, 500 두 가지. 실행 30회.
  - 지표: 시간 평균 위치 RMSE (추정치는 가중 평균). 쌍 비교 (같은 시드) 대응 t-검정과 Welch를 같이 본다.
가정·한계: 편향은 시간에 따라 변하지 않는다고 가정(모형과 필터가 같음). 장과 경로는 pde_toy와 같고, 장 κ는 참값.
실행: python3 analysis/extra_rbpf_bias.py -> analysis/extra_rbpf_bias_results.json
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pde_toy as P  # noqa: E402

T = P.T
L = P.L
SIG_S = P.SIG_S
SIG_O = P.SIG_O
NMAX = 500
N_RUNS = 30
P_TRUE_LIST = {"sd0.1": 0.01, "sd0.5": 0.25}
N_LIST = (100, 500)


def run_filter(kind, N, y, p_path_d, maps, pre, P0):
    """kind ∈ {'nobias', 'pfbias', 'rbpf'}. pre: 공통 난수. 반환 (T+1, 2) 추정 궤적."""
    d = p_path_d
    init_noise = pre["init"][:N]
    mot = pre["mot"][:, :N, :]
    u_res = pre["ures"]
    p = np.clip(pre["p0"] + init_noise, 0, L)
    w = np.full(N, 1.0 / N)
    if kind == "pfbias":
        b = pre["bnorm"][:N] * np.sqrt(P0)
    elif kind == "rbpf":
        bhat = np.zeros(N)
        Pb = np.full(N, P0)
    est = np.empty((T + 1, 2))
    est[0] = p.mean(axis=0)
    for t in range(1, T + 1):
        p = np.clip(p + d[t - 1] + mot[t - 1], 0, L)
        pred = P.sample(maps[t], p)
        if kind == "nobias":
            e = y[t] - pred
            S = SIG_S**2 * np.ones(N)
            logw = np.log(w + 1e-300) - 0.5 * e**2 / S
        elif kind == "pfbias":
            e = y[t] - pred - b
            logw = np.log(w + 1e-300) - 0.5 * e**2 / SIG_S**2
        else:  # rbpf
            e = y[t] - pred - bhat
            S = Pb + SIG_S**2
            logw = np.log(w + 1e-300) - 0.5 * (e**2 / S + np.log(2 * np.pi * S))
        logw -= logw.max()
        w = np.exp(logw)
        w /= w.sum()
        if kind == "rbpf":
            # 칼만 갱신 (b의 사후 평균과 분산). 예측 우도 계산에 쓴 e, S를 그대로 쓴다.
            K = Pb / S
            bhat = bhat + K * e
            Pb = Pb - K * K * S
        est[t] = w @ p
        if 1.0 / np.sum(w**2) < N / 2:
            uu = (u_res[t] + np.arange(N)) / N
            idx = np.minimum(np.searchsorted(np.cumsum(w), uu), N - 1)
            p = p[idx]
            w = np.full(N, 1.0 / N)
            if kind == "pfbias":
                b = b[idx]
            elif kind == "rbpf":
                bhat = bhat[idx]
                Pb = Pb[idx]
    return est


def main():
    t_all = time.perf_counter()
    p_path, d = P.path_disp()
    truth = P.solve(P.N_TRUE, P.KAPPA, "asym")[0]
    maps = P.solve(P.N_MAP, P.KAPPA, "asym")[0]
    out = {"setup": dict(N_list=list(N_LIST), N_runs=N_RUNS, sigma_sensor=SIG_S, sigma_odo=SIG_O,
                         bias_prior_var_list=P_TRUE_LIST, P0_equals_P_true=True,
                         note="PF-nobias는 편향을 0으로 둠; PF-bias와 RBPF는 P_0 = P_true"),
           "results": {}}
    for label, Pt in P_TRUE_LIST.items():
        for N in N_LIST:
            rows = {"nobias": [], "pfbias": [], "rbpf": []}
            for s in range(N_RUNS):
                rng = np.random.default_rng(7000 + s)
                # 참 궤적과 측정
                p_true = np.empty((T + 1, 2))
                p_true[0] = p_path[0]
                b_true = rng.normal(0, np.sqrt(Pt))
                mot_true = rng.normal(0, SIG_O, (T, 2))
                for t in range(1, T + 1):
                    p_true[t] = np.clip(p_true[t - 1] + d[t - 1] + mot_true[t - 1], 0, L)
                y = np.array([P.sample(truth[t], p_true[t:t + 1])[0] for t in range(T + 1)])
                y = y + b_true + rng.normal(0, SIG_S, T + 1)
                # 공통 난수 (필터 공유)
                pre = dict(init=rng.normal(0, P.SIG_INIT, (NMAX, 2)),
                           mot=rng.normal(0, SIG_O, (T, NMAX, 2)),
                           ures=rng.random(T + 1),
                           bnorm=rng.standard_normal(NMAX),
                           p0=p_true[0])
                for kind in rows:
                    est = run_filter(kind, N, y, d, maps, pre, Pt)
                    e = np.linalg.norm(est[1:] - p_true[1:], axis=1)
                    rows[kind].append(float(np.sqrt(np.mean(e**2))))
            r = {k: np.array(v) for k, v in rows.items()}
            key = f"{label}_N{N}"
            summ = {}
            for k, v in r.items():
                summ[k] = dict(rmse_mean=float(v.mean()), rmse_sd=float(v.std(ddof=1)), rmse_median=float(np.median(v)))
            summ["rbpf_minus_pfbias_paired_p"] = float(stats.ttest_rel(r["rbpf"], r["pfbias"]).pvalue)
            summ["rbpf_minus_pfbias_mean_diff"] = float((r["rbpf"] - r["pfbias"]).mean())
            summ["pfbias_minus_nobias_paired_p"] = float(stats.ttest_rel(r["pfbias"], r["nobias"]).pvalue)
            summ["rbpf_minus_nobias_paired_p"] = float(stats.ttest_rel(r["rbpf"], r["nobias"]).pvalue)
            summ["raw"] = {k: v.round(4).tolist() for k, v in r.items()}
            out["results"][key] = summ
            print(key, {k: (round(summ[k]["rmse_mean"], 3), round(summ[k]["rmse_sd"], 3)) for k in ("nobias", "pfbias", "rbpf")},
                  "RBPF-PF diff", round(summ["rbpf_minus_pfbias_mean_diff"], 3),
                  "paired p", f"{summ['rbpf_minus_pfbias_paired_p']:.3g}")
    out["total_wall_s"] = round(time.perf_counter() - t_all, 2)
    (HERE / "extra_rbpf_bias_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("wall", out["total_wall_s"], "s")


if __name__ == "__main__":
    main()
