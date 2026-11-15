"""토이 W: 짧은 wiggle에서 로프 강성(k)·감쇠(c)가 얼마나 식별되는가 (Fisher 정보와 CRLB, 잡음·설계 스윕).

모형 (논문의 ball-joint 시뮬레이터가 아니라 가장 단순한 선형 대응물):
- 평면 횡방향 변위 u_i(t), i=1..N. 노드 0은 로봇 끝단으로 u_0(t) = a sin(2 pi f t)로 강제.
- 굽힘 (k/l^2) B^T B u, 감쇠 (c/l^2) B^T B u̇, 질량 diag(rho*l) + 끝단 리드 질량 m_L.
  기저 자세 복원 스프링(강성 k)을 하나 더 둬서 힌지 기저의 0 Hz 강체 회전 모드를 막음(가정).
- 중력: 횡방향(wiggle 평면이 수직이라고 가정) 상수 힘 m_i g. 없으면 (k, c, rho, lead)를 같은 배율로
  바꿔도 운동이 똑같아서 절대 척도가 식별되지 않는다(실제로 확인함, 아래 README/03 참조).
- 선형 ODE: 기저 강제와 중력 상수를 augmented 상태 [u, u̇, sin, cos, 1]로 넣고 expm으로 프레임(60 Hz) 단위 정확 전파.
- 관측: 프레임마다 노드 1..N의 u_i + 가우시안 잡음 sigma.
- 파라미터(로그): k, c, rho, m_L. Fisher F = sum J^T J / sigma^2, CRLB = F^{-1}.
  m_L을 알려진 값이 아니라 추정 대상으로 두는 이유: 논문 Table 2도 리드 질량과 반지름을 추정 대상에 넣음.
- 검증: 두 설계에서 잡음 실현 200개로 비선형 최소제곱을 돌려 경험적 SD와 CRLB를 비교.

한계: 소진폭 선형, 중력·장력·접촉·3D 없음. 논문 파라미터의 정량 예측이 아니라
"짧은 wiggle이 어떤 파라미터 조합을 보는가"의 구조와 잡음 민감도를 본다.
실행: python3 analysis/toy_wiggle_fisher.py  (numpy, scipy, 단일 프로세스, 약 30–60초)
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

OUT = Path(__file__).resolve().parent / "toy_wiggle_fisher_results.json"
N = 20          # 노드 수 (논문: 20–26 링크)
L = 0.55        # 로프 길이 m (논문 학습 범위 0.45–0.65)
ELL = L / N
FPS = 60
NOM = dict(k=0.30, c=0.010, rho=0.050, lead=0.010)   # 굽힘 강성, 감쇠, kg/m, 리드 kg (toy 단위)
PNAMES = ["k", "c", "rho", "lead"]


def operators(k, c, rho, lead):
    """K, C, M과 기저 결합 항. 자유 노드 1..N, 기저 노드 0."""
    B = np.zeros((N - 1, N + 1))
    for r in range(N - 1):
        i = r + 1
        B[r, i - 1], B[r, i], B[r, i + 1] = 1.0, -2.0, 1.0
    G = B.T @ B / ELL**2
    H = np.zeros((N + 1, N + 1))             # 기저 자세 복원 항
    H[0, 0] = H[1, 1] = 1.0 / ELL**2
    H[0, 1] = H[1, 0] = -1.0 / ELL**2
    Kfull, Cfull = k * (G + H), c * (G + H)
    M = np.diag(np.full(N, rho * ELL))
    M[-1, -1] += lead
    return M, Kfull[1:, 1:], Kfull[1:, 0], Cfull[1:, 1:], Cfull[1:, 0]


G_ACC = 9.81


def system(th, a, f, gravity=True):
    k, c, rho, lead = th
    M, Kff, Kf0, Cff, Cf0 = operators(k, c, rho, lead)
    Minv = np.linalg.inv(M)
    w = 2 * np.pi * f
    n = N
    A = np.zeros((2 * n + 3, 2 * n + 3))
    if gravity:
        mvec = np.diag(M).copy()
        A[n:2 * n, 2 * n + 2] = Minv @ (mvec * G_ACC)   # 상수 상태 1에 곱함
    else:
        A[n:2 * n, 2 * n + 2] = 0.0
    A[:n, n:2 * n] = np.eye(n)
    A[n:2 * n, :n] = -Minv @ Kff
    A[n:2 * n, n:2 * n] = -Minv @ Cff
    A[n:2 * n, 2 * n] = -Minv @ (Kf0 * a)           # u0 = a sin
    A[n:2 * n, 2 * n + 1] = -Minv @ (Cf0 * a * w)   # u0dot = a w cos
    A[2 * n, 2 * n + 1] = w
    A[2 * n + 1, 2 * n] = -w
    return A


def simulate(th, a, f, T, gravity=True):
    """(K, N) 프레임별 노드 횡변위. th = (k, c, rho, lead)."""
    Ad = expm(system(th, a, f, gravity) / FPS)
    x = np.zeros(2 * N + 3)
    x[2 * N + 1] = 1.0
    x[2 * N + 2] = 1.0
    K = int(round(T * FPS))
    ys = np.zeros((K, N))
    for j in range(K):
        x = Ad @ x
        ys[j] = x[:N]
    return ys


def jac_log(th_log, a, f, T, h=1e-4):
    J = []
    for j in range(len(th_log)):
        tp, tm = th_log.copy(), th_log.copy()
        tp[j] += h
        tm[j] -= h
        J.append((simulate(np.exp(tp), a, f, T) - simulate(np.exp(tm), a, f, T)) / (2 * h))
    return np.stack(J, axis=-1)


ALPHA_AR = 0.8   # 추적 잡음의 시간 상관 (논문 Appendix J.3의 값을 가져옴)


def ar_inv(K, alpha):
    """AR(1) 상관 행렬 R_{st} = alpha^{|s-t|}의 역행렬. alpha=0이면 단위행렬."""
    if alpha == 0.0:
        return np.eye(K)
    idx = np.arange(K)
    R = alpha ** np.abs(idx[:, None] - idx[None, :])
    return np.linalg.inv(R)


def fisher_general(free, a, f, T, sig, alpha=0.0):
    """free: 추정할 파라미터 이름 목록 (k, c, rho, lead, gain). 나머지는 공칭(gain=1)으로 고정.
    관측 y = gain * u + 잡음(AR(1) alpha, 표준편차 sig). 로그 척도 Fisher와 CRLB 반환."""
    base = dict(k=np.log(NOM["k"]), c=np.log(NOM["c"]), rho=np.log(NOM["rho"]), lead=np.log(NOM["lead"]), gain=0.0)
    h = 1e-4

    def model(vals):
        th = np.exp(np.array([vals["k"], vals["c"], vals["rho"], vals["lead"]]))
        return np.exp(vals["gain"]) * simulate(th, a, f, T)
    y0 = model(base)
    J = []
    for name in free:
        vp, vm = dict(base), dict(base)
        vp[name] += h
        vm[name] -= h
        J.append((model(vp) - model(vm)) / (2 * h))
    J = np.stack(J, axis=-1)                 # (K, N, P)
    Rinv = ar_inv(J.shape[0], alpha)
    F = np.einsum("kni,kl,lnj->ij", J, Rinv, J) / sig**2
    ev = np.linalg.eigvalsh(F)
    if ev.min() / ev.max() < 1e-10:          # 특이 행렬: 척도 불변 방향이 있어 식별 불가
        return dict(zip(free, [np.inf] * len(free))), np.full((len(free), len(free)), np.nan), F
    cov = np.linalg.inv(F)
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    return dict(zip(free, d)), corr, F


def theta_from(vals):
    """vals: 자유 파라미터 로그값 (k, c) 순서. rho와 lead는 저울로 잰 값으로 고정."""
    return np.array([np.exp(vals[0]), np.exp(vals[1]), NOM["rho"], NOM["lead"]])


def crlb(a, f, T, sig, free=("k", "c"), alpha=0.0):
    """기본: rho와 lead는 측정으로 고정, k와 c만 추정. 로그 척도 CRLB (sd, 상관)."""
    d, corr, _ = fisher_general(list(free), a, f, T, sig, alpha)
    return np.array([d[n] for n in free]), corr


def scale_null_check(a=0.04, f=0.75, T=6.7, sig=0.005):
    """4개 파라미터(k, c, rho, lead)를 모두 추정할 때 Fisher 행렬의 최소 고유값과 그 방향.
    스케일 불변이면 (1,1,1,1) 방향 고유값이 0에 가깝다."""
    th = np.log(np.array([NOM["k"], NOM["c"], NOM["rho"], NOM["lead"]]))
    J = []
    h = 1e-4
    for j in range(4):
        tp, tm = th.copy(), th.copy()
        tp[j] += h
        tm[j] -= h
        J.append((simulate(np.exp(tp), a, f, T) - simulate(np.exp(tm), a, f, T)) / (2 * h))
    J = np.stack(J, axis=-1)
    F = np.einsum("kni,knj->ij", J, J) / sig**2
    ev, V = np.linalg.eigh(F)
    v0 = V[:, 0]
    return dict(eigvals_rel=(ev / ev.max()).tolist(), smallest_eigvec=(v0 * np.sign(v0[0])).round(3).tolist(),
                design=dict(T=T, a=a, f=f, sigma_mm=sig * 1000))


def mismatch_test(n_true=26, a=0.04, f=0.75, T=6.7):
    """데이터는 N=26 노드 모형(진짜), 추정은 N=20 모형으로 한다. 노드 위치를 선형 보간해 맞춘다.
    잡음 없이 k, c를 최소제곱으로 맞춘 편향(모형 오차만의 효과)을 본다."""
    global N, ELL
    out = {}
    for label, sig in [("noiseless", 1e-9), ("sig5mm", 0.005)]:
        N_keep, ELL_keep = N, ELL
        N, ELL = n_true, L / n_true
        y_true = simulate(np.exp(np.log([NOM["k"], NOM["c"], NOM["rho"], NOM["lead"]])), a, f, T)
        N, ELL = N_keep, ELL_keep
        x_true = np.linspace(0, 1, n_true)
        x_est = np.linspace(0, 1, N)
        y_obs_base = np.array([np.interp(x_est, x_true, row) for row in y_true])
        rng = np.random.default_rng(2)
        y_obs = y_obs_base + (rng.normal(0, sig, y_obs_base.shape) if sig > 1e-6 else 0.0)

        def resid(p):
            return (simulate(theta_from(p), a, f, T) - y_obs).ravel() / max(sig, 1e-3)
        r = least_squares(resid, np.log([NOM["k"], NOM["c"]]) + 0.2, bounds=(np.log([0.01, 1e-4]), np.log([10, 1])),
                          xtol=1e-10, ftol=1e-10, max_nfev=300)
        out[label] = dict(k_est=round(float(np.exp(r.x[0])), 4), c_est=round(float(np.exp(r.x[1])), 5),
                          k_bias_pct=round(100 * (np.exp(r.x[0]) / NOM["k"] - 1), 1),
                          c_bias_pct=round(100 * (np.exp(r.x[1]) / NOM["c"] - 1), 1))
    out["note"] = "진짜 모형 N=26 대 추정 모형 N=20, 노드 위치는 선형 보간. 이 편향이 잡음 SD보다 크면 모형 오차가 식별 오차를 지배한다."
    return out


def mc_check(a, f, T, sig, draws=200, seed=1):
    rng = np.random.default_rng(seed)
    th_true = np.log(np.array([NOM["k"], NOM["c"]]))
    y0 = simulate(theta_from(th_true), a, f, T)
    sd, _ = crlb(a, f, T, sig)
    est = []
    for _ in range(draws):
        yobs = y0 + rng.normal(0, sig, y0.shape)

        def resid(p):
            return (simulate(theta_from(p), a, f, T) - yobs).ravel() / sig
        start = th_true + rng.normal(0, 0.3, 2)
        r = least_squares(resid, start, bounds=(th_true - 3, th_true + 3), xtol=1e-10, ftol=1e-10, max_nfev=200)
        est.append(r.x)
    est = np.array(est)
    emp = est.std(axis=0, ddof=1)
    return dict(empirical_sd_log=emp.round(4).tolist(), crlb_sd_log=sd.round(4).tolist(),
                ratio_emp_over_crlb=(emp / sd).round(3).tolist(),
                bias_log=(est.mean(axis=0) - th_true).round(4).tolist(), draws=draws, sigma_mm=sig * 1000,
                design=dict(T=T, a=a, f=f))


def lowest_modes():
    M, Kff, _, _, _ = operators(NOM["k"], 0.0, NOM["rho"], NOM["lead"])
    ev = np.sort(np.linalg.eigvals(np.linalg.solve(M, Kff)).real)
    return (np.sqrt(np.abs(ev[:3])) / (2 * np.pi)).round(3).tolist()


def main():
    t0 = time.time()
    res = {"model": f"linear planar chain, base-driven, gravity, N={N}, L={L} m, 60 fps; unknown log(k,c); rho, lead measured",
           "nominal": NOM, "lowest_undamped_modes_Hz": lowest_modes(),
           "note": "k, c, rho, lead는 toy 단위. 논문 범위의 비율(k 20배, c 50배)만 참고."}
    amps = [0.01, 0.02, 0.04, 0.08]                  # m, 기저 변위 진폭 (20 cm 연장대 기준 약 3°–23°)
    freqs = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]    # Hz (논문 wiggle 0.5–1.0 Hz 포함)
    durs = [2.0, 4.0, 6.7]                           # s (6.7 s = 논문 400 프레임)
    sigs = [0.001, 0.005, 0.010, 0.020, 0.050]       # 관측 잡음 1, 5, 10, 20, 50 mm
    res["grid"] = {}
    for sig in sigs:
        rows = []
        for T in durs:
            for a in amps:
                for f in freqs:
                    sd, corr = crlb(a, f, T, sig)
                    rows.append(dict(T=T, a=a, f=f, relsd_k=round(float(sd[0]), 4), relsd_c=round(float(sd[1]), 4),
                                     corr_kc=round(float(corr[0, 1]), 3)))
        res["grid"][str(int(round(sig * 1000))) + "mm"] = rows
    # 요약 1: 설계 하나의 k, c 상대 SD (이상적으로는 작아야 함)
    summ = {}
    for key, rows in res["grid"].items():
        best = min(rows, key=lambda g: g["relsd_k"] + g["relsd_c"])
        worst_kc = max(rows, key=lambda g: g["relsd_k"] + g["relsd_c"])
        summ[key] = dict(best=best, worst=worst_kc,
                         median_relsd_c_T2=float(np.median([g["relsd_c"] for g in rows if g["T"] == 2.0])),
                         median_relsd_c_T67=float(np.median([g["relsd_c"] for g in rows if g["T"] == 6.7])),
                         median_relsd_k_T2=float(np.median([g["relsd_k"] for g in rows if g["T"] == 2.0])),
                         median_relsd_k_T67=float(np.median([g["relsd_k"] for g in rows if g["T"] == 6.7])))
    res["summary_by_sigma"] = summ
    # 요약 2: 척도 불변 확인 (4개 추정 시 최소 고유값)
    res["scale_null_check_4params"] = scale_null_check()
    # 요약 3: 가정을 바꾼 변형 (잡음 상관, 미지 질량, 이득 교란)
    var = {}
    dsg = dict(a=0.04, f=0.75, T=6.7, sig=0.005)
    for label, free, alpha in [
        ("A 기준: rho·lead 측정, k·c 추정, 백색잡음", ["k", "c"], 0.0),
        ("B AR(1) 잡음 0.8, rho·lead 측정", ["k", "c"], ALPHA_AR),
        ("C rho 미지(추정), lead 측정", ["k", "c", "rho"], 0.0),
        ("D rho·lead 미지", ["k", "c", "rho", "lead"], 0.0),
    ]:
        d, corr, _ = fisher_general(free, dsg["a"], dsg["f"], dsg["T"], dsg["sig"], alpha)
        var[label] = dict(relsd={k_: round(float(v), 4) for k_, v in d.items()}, corr_kc=round(float(corr[0, 1]), 3))
    res["variants_T6.7_a0.04_f0.75_sig5mm"] = var
    # 요약 2: 검증 (경험적 SD 대 CRLB)
    res["mc_check"] = [mc_check(0.04, 0.75, 6.7, 0.001), mc_check(0.04, 0.75, 6.7, 0.005)]
    res["mc_check_short"] = mc_check(0.08, 1.0, 2.0, 0.005)
    res["mismatch_test"] = mismatch_test()
    res["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("modes", res["lowest_undamped_modes_Hz"])
    for key in summ:
        print(key, {k: v for k, v in summ[key].items() if k.startswith("median")}, "best", summ[key]["best"])
    print("mc", [(m["sigma_mm"], m["ratio_emp_over_crlb"]) for m in res["mc_check"]])
    print("null", res["scale_null_check_4params"]["eigvals_rel"][:2])
    for kk, vv in res["variants_T6.7_a0.04_f0.75_sig5mm"].items():
        print(kk, vv)
    print("mismatch", res["mismatch_test"])
    print("time", res["runtime_s"])


if __name__ == "__main__":
    main()
