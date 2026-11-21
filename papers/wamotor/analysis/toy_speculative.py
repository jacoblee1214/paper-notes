"""CPU 토이: 속도 재사용 투기 샘플러에서 TSAV(시간 간격 가중 검증)와 균일 MSE 검증을 비교한다.

논문의 핵심 주장(Lemma 4.4, 식 (6))을 작은 흐름 모델에서 점검한다.
- 모델: 고정 난수 가중치를 가진 작은 MLP 속도장 v(x,t) (학습 없음, CPU, numpy).
- 스케줄: 논문 Shift-5 (phi(u) = s*u / (1 + (s-1)*u), s=5), N=10 스텝.
- 기준: 같은 초기 잡음에서 정확한 Euler 적분 결과.
- 비교: (a) 검증 없이 K=1 이후 앵커 속도 재사용, (b) 균일 임계 MSE < eps, (c) TSAV MSE*(h_k/h_mid)^2 < eps.
- 지표: 최종 상태 오차, 모델 호출 수(NFE). 코드의 action.py 로직을 따른다.

실행: python3 analysis/toy_speculative.py  → analysis/toy_speculative_results.json (수 초)
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_speculative_results.json"
D, H_HID, N, W = 16, 64, 10, 10
SHIFT = 5.0


def make_model(seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1.0 / np.sqrt(D), (H_HID, D + 1))
    W2 = rng.normal(0, 1.0 / np.sqrt(H_HID), (D, H_HID))

    def v(x, t):
        inp = np.concatenate([x, [t]])
        return W2 @ np.tanh(W1 @ inp)
    return v


def schedule(n, shift=SHIFT):
    u = np.linspace(0, 1, n + 1)
    sig = shift * u / (1 + (shift - 1) * u)
    t = 1 - sig  # t: 1 -> 0 (노이즈 -> 데이터). 델타는 음수이므로 절댓값을 쓴다.
    return t[:-1], np.diff(t)  # 시작 시각 t_k, 적분 간격 Δt_k (음수)


def euler_ref(v, x0, ts, dts):
    x = x0.copy()
    calls = 0
    for t, dt in zip(ts, dts):
        x = x + dt * v(x, t)
        calls += 1
    return x, calls


def speculative(v, x0, ts, dts, eps, mode):
    """mode: 'uniform' 또는 'tsav'. action.py 의 speculative_denoise_action 과 같은 흐름."""
    abs_d = np.abs(dts)
    h_mid = np.median(abs_d)
    thr = eps * (h_mid / np.maximum(abs_d, 1e-12)) ** 2
    calls = 0

    def predict(x, t):
        nonlocal calls
        calls += 1
        return v(x, t)

    x = x0.copy()
    vel = predict(x, ts[0])
    x = x + dts[0] * vel  # 첫 Euler 스텝 (K=1)
    m = 1
    vel = None
    num = len(ts)
    while m < num:
        if vel is None:
            vel = predict(x, ts[m])
        if m == num - 1:
            return x + dts[m] * vel, calls
        end = min(m + W, num - 1)
        cand_idx = list(range(m + 1, end + 1))
        count = len(cand_idx)
        drafts = []
        acc_state = x.copy()
        for j in range(count):
            acc_state = acc_state + dts[m + j] * vel  # 앵커 속도 재사용
            drafts.append(acc_state.copy())
        # 검증: 배치 호출을 한 번으로 셈
        calls += 1
        vals = [v(d, ts[ci]) for d, ci in zip(drafts, cand_idx)]
        mses = [np.mean((vel - vv) ** 2) for vv in vals]
        if mode == "uniform":
            ok = [mse < eps for mse in mses]
        else:
            ok = [mses[j] < thr[cand_idx[j]] for j in range(count)]
        accepted = 0
        for flag in ok:
            if flag:
                accepted += 1
            else:
                break
        if accepted == count:
            x = drafts[-1]
            vel = vals[-1]
            m = end
            continue
        if accepted:
            x = drafts[accepted - 1]
            fb = vals[accepted - 1]
        else:
            fb = vel
        step = m + accepted
        x = x + dts[step] * fb
        m = step + 1
        vel = vals[0] if accepted == 0 else None
        if accepted == 0:
            calls += 0  # 검증 결과 재사용(추가 호출 없음), 코드와 동일
    return x, calls


def main():
    res = {"config": {"D": D, "hidden": H_HID, "N": N, "W": W, "shift": SHIFT}, "runs": []}
    eps_list = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1]
    seeds = range(20)
    v = make_model(0)
    ts, dts = schedule(N)
    for mode in ["uniform", "tsav"]:
        for eps in eps_list:
            errs, nfes = [], []
            for s in seeds:
                x0 = np.random.default_rng(100 + s).normal(size=D)
                ref, _ = euler_ref(v, x0, ts, dts)
                out, nfe = speculative(v, x0, ts, dts, eps, mode)
                errs.append(float(np.linalg.norm(out - ref)))
                nfes.append(nfe)
            res["runs"].append({"mode": mode, "eps": eps,
                                "final_err_mean": round(float(np.mean(errs)), 6),
                                "final_err_max": round(float(np.max(errs)), 6),
                                "nfe_mean": round(float(np.mean(nfes)), 2)})
    # 기준: 정확한 Euler 호출 수
    res["euler_nfe"] = N
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    for r in res["runs"]:
        print(r)


if __name__ == "__main__":
    main()
