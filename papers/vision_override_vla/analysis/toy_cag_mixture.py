"""토이: CAG의 행동 공간 외삽이 "언어 우도를 ω제곱으로 키운 분포"와 얼마나 같은가.

논문 v1·v2 부록 A는 a_CAG = a_u + ω(a_c − a_u)를 P(a|o) P(l|a,o)^ω 로 읽을 수 있다고 쓰고,
v3 §5.1은 이를 "CFG에서 영감을 얻은 행동 공간 외삽 휴리스틱"이라고 쓴다. 이 토이는 닫힌 형식이 있는 1차원
가우시안 혼합 행동 분포에서 네 가지를 비교한다.

  prior   : 시각만 보는 정책 p_u(a) = Σ w_u,k N(μ_k, s²), μ = (−1 [훈련 과제], +1 [지시된 과제])
  cond    : 언어를 받은 정책 p_c(a)  = Σ w_c,k N(μ_k, s²)
  exact   : 분포 수준의 정확한 기울이기  p ∝ p_u^(1−ω) · p_c^ω  (모드 가중치만 w_u^(1−ω) w_c^ω로 바뀜)
  vel     : 흐름 정합 속도장에서 CFG: v = v_u + ω (v_c − v_u), 같은 노이즈에서 10단계 오일러
  act_sh  : 두 가지가 각각 끝점을 만든 뒤 행동에서 외삽, 같은 노이즈 공유 (공식 구현의 방식)
  act_ind : 위와 같으나 노이즈를 따로 뽑음

속도장은 선형 경로 x_t = (1−t) x0 + t x1 의 주변 속도 v(x,t) = (E[x1|x_t=x] − x)/(1−t) 를 닫힌 형식으로 쓴다.
신경망 학습은 없다. 이 토이는 한 모드가 +1, 한 모드가 −1인 1차원 행동이라는 단순화이며 로봇 정책의 증거가 아니다.

실행: python3 analysis/toy_cag_mixture.py   (numpy, 수 초)
결과: analysis/toy_cag_mixture_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_cag_mixture_results.json"
MU = np.array([-1.0, 1.0])
S = 0.1
N = 40000
STEPS = 10


def velocity(x, t, w):
    """가우시안 혼합 데이터 + 표준정규 노이즈, 선형 경로의 주변 속도."""
    var = (1 - t) ** 2 + (t * S) ** 2
    logr = np.log(w)[None, :] - 0.5 * (x[:, None] - t * MU[None, :]) ** 2 / var
    logr -= logr.max(axis=1, keepdims=True)
    r = np.exp(logr)
    r /= r.sum(axis=1, keepdims=True)
    ex1_k = MU[None, :] + (t * S * S / var) * (x[:, None] - t * MU[None, :])
    ex1 = (r * ex1_k).sum(axis=1)
    return (ex1 - x) / (1 - t)


def integrate(x0, vfun):
    x = x0.copy()
    for k in range(STEPS):
        t = k / STEPS
        x = x + vfun(x, t) / STEPS
    return x


def classify(a):
    """모드 A(훈련 과제, −1 근처), 모드 B(지시된 과제, +1 근처), 그 밖."""
    near_a = np.abs(a + 1) < 0.5
    near_b = np.abs(a - 1) < 0.5
    return near_a, near_b


def summarize(a):
    na, nb = classify(a)
    return {"P_B": float(nb.mean()), "P_A": float(na.mean()), "P_off_mode": float(1 - na.mean() - nb.mean()),
            "P_positive_side": float((a > 0).mean()),
            "P_beyond_1.25": float((np.abs(a) > 1.25).mean()), "P_wrong_sign_overshoot": float((a < -1.25).mean()),
            "mean_a": float(a.mean())}


def run(w_u, w_c, omegas, seed=0):
    rng = np.random.default_rng(seed)
    w_u, w_c = np.array(w_u, float), np.array(w_c, float)
    out = {}
    for om in omegas:
        x0 = rng.standard_normal(N)
        x0b = rng.standard_normal(N)
        # exact tilt
        wt = w_u ** (1 - om) * w_c ** om
        wt /= wt.sum()
        k = (rng.random(N) < wt[1]).astype(int)
        a_exact = MU[k] + S * rng.standard_normal(N)
        # velocity CFG
        a_vel = integrate(x0, lambda x, t: velocity(x, t, w_u) + om * (velocity(x, t, w_c) - velocity(x, t, w_u)))
        # action-space extrapolation
        a_u = integrate(x0, lambda x, t: velocity(x, t, w_u))
        a_c = integrate(x0, lambda x, t: velocity(x, t, w_c))
        a_sh = a_u + om * (a_c - a_u)
        a_ub = integrate(x0b, lambda x, t: velocity(x, t, w_u))
        a_ind = a_ub + om * (a_c - a_ub)
        out[f"{om:g}"] = {"exact_tilt": summarize(a_exact), "velocity_cfg": summarize(a_vel),
                          "action_shared_noise": summarize(a_sh), "action_independent_noise": summarize(a_ind),
                          "exact_weight_B": float(wt[1])}
    return out


def closed_loop(w_u, w_c, om, mode, n=4000, step=0.1, max_steps=3000, seed=1):
    """1차원 위치 x=0에서 시작해 매 스텝 행동 a를 뽑아 x += step*a. |x| >= 1에 닿으면 끝.
    x >= 1이면 지시된 과제(B) 도달, x <= −1이면 훈련 과제(A) 도달. 모드 선택 확률은 위치와 무관한 상수(최소 가정).
    mode: 'exact_tilt' | 'action_shared' | 'action_independent'
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    alive = np.ones(n, bool)
    steps_used = np.zeros(n)
    wt = np.array(w_u) ** (1 - om) * np.array(w_c) ** om
    wt /= wt.sum()
    for t in range(max_steps):
        idx = np.where(alive)[0]
        if len(idx) == 0:
            break
        m = len(idx)
        u1, u2 = rng.random(m), rng.random(m)
        z1, z2 = rng.standard_normal(m), rng.standard_normal(m)
        if mode == "exact_tilt":
            a = MU[(u1 < wt[1]).astype(int)] + S * z1
        else:
            u_c = u1
            u_u = u1 if mode == "action_shared" else u2
            z_u = z1 if mode == "action_shared" else z2
            a_c = MU[(u_c < w_c[1]).astype(int)] + S * z1
            a_u = MU[(u_u < w_u[1]).astype(int)] + S * z_u
            a = a_u + om * (a_c - a_u)
        x[idx] += step * a
        steps_used[idx] += 1
        done = np.abs(x[idx]) >= 1.0
        alive[idx[done]] = False
    return {"P_reach_B": float((x >= 1).mean()), "P_reach_A": float((x <= -1).mean()), "P_timeout": float(alive.mean()),
            "mean_steps": float(steps_used.mean())}


def main():
    omegas = [1.0, 1.5, 2.0, 3.0]
    res = {"setup": {"modes": "A=-1 (훈련 과제), B=+1 (지시된 과제)", "s": S, "samples": N, "euler_steps": STEPS,
                     "note": "ω=1이면 네 방법 모두 조건부 정책 p_c를 준다(행동 외삽은 a_c). 로봇 증거가 아니라 1차원 혼합의 닫힌 형식 점검."}}
    # 케이스 1: 시각 선행이 훈련 과제로 강하게 쏠림(B 10%), 언어 조건부는 B 40%
    res["case1_w_u=(0.9,0.1)_w_c=(0.6,0.4)"] = run((0.9, 0.1), (0.6, 0.4), omegas)
    # 케이스 2: 언어가 한 모드를 거의 정하는 경우 (B 90%)
    res["case2_w_u=(0.9,0.1)_w_c=(0.1,0.9)"] = run((0.9, 0.1), (0.1, 0.9), omegas)
    # 케이스 3: 시각 선행이 B를 5%만, 조건부도 B 25%
    res["case3_w_u=(0.95,0.05)_w_c=(0.75,0.25)"] = run((0.95, 0.05), (0.75, 0.25), omegas)
    # 닫힌 루프 토이: 한 번의 행동이 아니라 매 스텝 다시 뽑는 상황에서 B 도달 확률
    cl = {}
    for name, (wu, wc) in {"w_u=(0.9,0.1)_w_c=(0.6,0.4)": ((0.9, 0.1), (0.6, 0.4)),
                           "w_u=(0.95,0.05)_w_c=(0.75,0.25)": ((0.95, 0.05), (0.75, 0.25))}.items():
        cl[name] = {f"{om:g}": {m: closed_loop(wu, wc, om, m) for m in ("exact_tilt", "action_shared", "action_independent")}
                    for om in omegas}
    res["closed_loop_random_walk"] = {"note": "행동 외삽은 평균 행동을 (1−ω)E[a_u] + ωE[a_c]로 바꾸므로 닫힌 루프에서 표류(drift)가 ω에 따라 커진다. "
                                              "모드 선택 확률이 위치와 무관한 최소 가정이다.", "cases": cl}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    for case, d in res.items():
        if not case.startswith("case"):
            continue
        print(case)
        for om, r in d.items():
            print(f"  ω={om}: exact B={r['exact_weight_B']:.3f} | vel B={r['velocity_cfg']['P_B']:.3f} | "
                  f"act(shared) B={r['action_shared_noise']['P_B']:.3f} pos={r['action_shared_noise']['P_positive_side']:.3f} off={r['action_shared_noise']['P_off_mode']:.3f} "
                  f"wrong-sign={r['action_shared_noise']['P_wrong_sign_overshoot']:.3f} | "
                  f"act(indep) B={r['action_independent_noise']['P_B']:.3f} off={r['action_independent_noise']['P_off_mode']:.3f}")
    for name, d in cl.items():
        print("closed loop", name)
        for om, r in d.items():
            print(f"  ω={om}: " + " | ".join(f"{m} B={v['P_reach_B']:.3f} A={v['P_reach_A']:.3f}" for m, v in r.items()))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
