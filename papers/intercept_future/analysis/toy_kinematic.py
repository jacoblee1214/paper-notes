"""토이: 정속 외삽 대 등가속 외삽(식 3)의 위치 오차, 광류 잡음 포함 (1차원, numpy, 수 초).

논문 §4.2 식 (3) V_k = V_0 + A k dt 의 핵심 가정을 시험한다. A 는 두 광류 속도의 차분(식 10)이므로
속도 잡음 sigma 가 있으면 A 의 잡음은 sqrt(2) sigma / dt 이다. 등가속 외삽이 정속 외삽보다 나아지려면
실제 가속이 얼마나 커야 하는지, 그리고 비선형 충돌(속도가 갑자기 바뀜)에서는 어떻게 되는지 본다.

가정(논문에 없는 값, 우리가 정함):
- 프레임 간격 dt = 1 프레임, 속도 단위 px/프레임, 가속 단위 px/프레임^2. 1 스텝 = 1 프레임(30 Hz).
- 관측된 속도 V_t, V_{t-1} 에 서로 독립인 가우시안 잡음 sigma (패치 평균 풀링 뒤 RAFT 오차).
- 예측 위치 변화 = sum_{k=1..K} V_k dt 로 두고 V_k = V0 + A k dt 를 그대로 적분한다(세계모델이 적분을 잘 한다고 가정한 낙관적 상한).
- 참값은 등가속(가속 a) 또는 K 스텝 안의 무작위 시각에 속도가 N(0, jump^2) 만큼 튀는 충돌.
이 토이는 우리가 만든 모형이며 논문의 표 수치로 옮기면 안 된다.

실행: python3 analysis/toy_kinematic.py   → analysis/toy_kinematic_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_kinematic_results.json"
rng = np.random.default_rng(0)
N = 40000
V0_TRUE = 2.0          # px/프레임 (약 25 cm/s 를 224 px 시야 1 m 로 가정한 값)
KS = np.arange(1, 11)


def rms_errors(a, sigma, jump=0.0):
    """K=1..10 에서 정속/등가속 외삽의 위치 RMS 오차(px)."""
    # 참 속도 V_t = V0 (현재), V_{t-1} = V0 - a (등가속이면)
    e1 = rng.normal(0, sigma, N)
    e2 = rng.normal(0, sigma, N)
    v_now = V0_TRUE + e1
    v_prev = V0_TRUE - a + e2
    A_hat = v_now - v_prev                      # dt = 1
    res = {"const": [], "accel": []}
    for K in KS:
        true_disp = (V0_TRUE * K + a * K * (K + 1) / 2) * np.ones(N)   # sum_{j<=K} (V0 + a j)
        if jump > 0:
            # 충돌: 시각 J 이후 속도가 d 만큼 바뀜 (가속은 0 으로 둠)
            J = rng.integers(1, K + 1, N)
            d = rng.normal(0, jump, N)
            true_disp = V0_TRUE * K + d * np.maximum(K - J + 1, 0)
        const = v_now * K
        accel = v_now * K + A_hat * (K * (K + 1) / 2)
        res["const"].append(float(np.sqrt(np.mean((const - true_disp) ** 2))))
        res["accel"].append(float(np.sqrt(np.mean((accel - true_disp) ** 2))))
    return res


results = {"setup": dict(V0=V0_TRUE, N=N, dt="1 frame", K="1..10")}

# 1) 임계 가속: K=5 에서 두 방법의 RMS 오차가 같아지는 |a| (sigma 별)
crit = {}
for sigma in (0.03, 0.1, 0.3):
    grid = np.linspace(0.0, 1.0, 101)
    diff = []
    for a in grid:
        r = rms_errors(a, sigma)
        diff.append(r["accel"][4] - r["const"][4])
    diff = np.array(diff)
    idx = int(np.argmax(diff < 0)) if (diff < 0).any() else None
    crit[str(sigma)] = dict(a_star=float(grid[idx]) if idx is not None else None,
                            ratio_a_star_over_sigma=float(grid[idx] / sigma) if idx is not None else None)
results["critical_acceleration_K5"] = crit

# 2) 대표 설정의 오차표
cases = {}
for name, a, sigma, jump in [
    ("완만한 가속 a=0.02", 0.02, 0.1, 0.0),
    ("중간 가속 a=0.10", 0.10, 0.1, 0.0),
    ("큰 가속 a=0.30", 0.30, 0.1, 0.0),
    ("충돌 (속도 점프 SD 1.0)", 0.0, 0.1, 1.0),
    ("잡음 없음, a=0.10", 0.10, 0.0, 0.0),
]:
    r = rms_errors(a, sigma, jump)
    cases[name] = dict(a=a, sigma=sigma, jump=jump,
                       const_K1_5_10=[round(r["const"][i], 2) for i in (0, 4, 9)],
                       accel_K1_5_10=[round(r["accel"][i], 2) for i in (0, 4, 9)])
results["cases"] = cases

# 3) 같은 속도 변화를 5 스텝(약 167 ms) 안에 몇 % 로 환산하는가
results["speed_change_in_5_steps_pct"] = {str(a): round(100 * a * 5 / V0_TRUE, 1) for a in (0.02, 0.10, 0.30)}

OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(results, indent=1, ensure_ascii=False))
