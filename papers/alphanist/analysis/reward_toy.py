"""보상 구조 toy: 성공 시 종료 + 매 step 양의 보조 보상이 만드는 '끝내지 않을 유인' (CPU, numpy). 실행: python3 analysis/reward_toy.py

AlphaNIST의 dense 비교군(식 7)은 OmniReset의 보조 계수(c2 = c3 = 0.01)를 가져오지만 성공 구조는 다르다.
- AlphaNIST(식 5–6): 성공하면 보상 1을 한 번 주고 episode를 끝낸다.
- OmniReset 구현(UW-Lab/UWLab @ 2b40afb, rl_state_cfg.py): 보조 항 가중치 0.1, 성공 항 1.0을 매 step 주고(정렬된 동안),
  Isaac Lab RewardManager가 dt = 0.1 s를 곱한다 → 실효 보조 0.01/step, 성공 0.1/step. 성공으로 끝나지 않는다(시간 초과·비정상만 종료).

1차원 진행도 사슬 MDP(진행도 0..K, K = 조립 완료)를 유한 horizon backward induction으로 풀어 최적 정책을 본다.
행동: advance(확률 a로 한 칸 전진) / hold(제자리). 보조 보상 s(k) = (c2 + c3) · k/K (손이 부품에 가까울수록, 정렬될수록 커지는 항을 단순화).
H1: 보상 종류별 최적 정책의 완료 시점
H2: 완료 직전 '대기' 중 미끄러짐 위험 h가 있으면 성공률이 얼마나 떨어지는가
H3: γ, 보조 크기, 성공 보상 크기에 따른 '지금 끝내기' vs '끝까지 대기' 손익분기 잔여 step
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
K, T, ADV = 60, 375, 0.9          # 진행도 칸 수, horizon(15 s × 25 Hz), 전진 성공 확률
SLIP_BACK = 15                     # 미끄러지면 K - 1에서 이만큼 뒤로


def solve(gamma, shaping, success, terminate_on_success, success_per_step=0.0, hazard=0.0):
    """V[t, k]와 정책 pi[t, k] (1 = advance, 0 = hold). 칸 K는 완료 상태."""
    V = np.zeros((T + 1, K + 1))
    pi = np.zeros((T, K + 1), dtype=int)
    s = shaping * np.arange(K + 1) / K
    for t in range(T - 1, -1, -1):
        for k in range(K + 1):
            if k == K:
                if terminate_on_success:
                    V[t, k] = 0.0  # 흡수(이미 종료)
                else:
                    V[t, k] = success_per_step + s[k] + gamma * V[t + 1, k]
                continue
            nxt = k + 1
            r_hit = success if nxt == K else 0.0
            # advance: 확률 ADV로 전진(전진 직후 보상 포함), 아니면 제자리
            adv = ADV * (r_hit + s[nxt] + gamma * (0.0 if (nxt == K and terminate_on_success) else V[t + 1, nxt])) \
                + (1 - ADV) * (s[k] + gamma * V[t + 1, k])
            # hold: 제자리. 완료 직전 대기 중에는 hazard 확률로 미끄러짐
            back = max(k - SLIP_BACK, 0)
            haz = hazard if k >= K - 3 else 0.0
            hold = (1 - haz) * (s[k] + gamma * V[t + 1, k]) + haz * (s[back] + gamma * V[t + 1, back])
            if adv >= hold:
                V[t, k], pi[t, k] = adv, 1
            else:
                V[t, k], pi[t, k] = hold, 0
    return V, pi


def rollout(pi, hazard, n=4000, seed=0):
    """최적 정책을 확률적으로 굴려 완료율과 완료 시점."""
    rng = np.random.default_rng(seed)
    done_t = []
    for _ in range(n):
        k = 0
        for t in range(T):
            if pi[t, k] == 1:
                if rng.random() < ADV:
                    k += 1
            elif k >= K - 3 and rng.random() < hazard:
                k = max(k - SLIP_BACK, 0)
            if k == K:
                done_t.append(t + 1)
                break
    return {"success_within_T": round(len(done_t) / n, 3),
            "median_completion_step": int(np.median(done_t)) if done_t else None}


def first_finish_step(pi):
    """k = K-1에서 advance를 처음 고르는 시각(그 전에는 대기)."""
    col = pi[:, K - 1]
    idx = np.where(col == 1)[0]
    return int(idx[0]) if len(idx) else None


def main():
    out: dict = {"setup": {"K": K, "T": T, "advance_prob": ADV, "min_steps_to_finish": int(np.ceil(K / ADV))}}
    cases = {
        "binary_only(AlphaNIST Eq.6, R=1)": dict(shaping=0.0, success=1.0, terminate_on_success=True),
        "dense+terminate(AlphaNIST Eq.7, c2+c3=0.02, R=1)": dict(shaping=0.02, success=1.0, terminate_on_success=True),
        "dense+terminate, R=0.04 (if R also scaled by dt=0.04)": dict(shaping=0.02, success=0.04, terminate_on_success=True),
        "OmniReset-style (shaping 0.02/step, success 0.1/step, no termination)": dict(shaping=0.02, success=0.0,
                                                                                     terminate_on_success=False, success_per_step=0.1),
    }
    h1 = {}
    for g in (0.99, 0.995):
        for name, kw in cases.items():
            for hz in (0.0, 0.01):
                V, pi = solve(g, hazard=hz, **kw)
                key = f"gamma={g} | {name} | hazard={hz}"
                h1[key] = {"first_step_choosing_finish_at_K-1": first_finish_step(pi), **rollout(pi, hz)}
    out["H1_H2_optimal_policies"] = h1

    # H3 해석식: K-1에서 '지금 끝내기' R 대 'n step 대기 후 끝내기' s(1-γ^n)/(1-γ) + γ^n R.
    #   차이 = (1-γ^n) [s/(1-γ) - R] → s/(1-γ) > R이면 남은 step 수와 상관없이 끝까지 미루는 편이 낫다.
    h3 = {}
    for g in (0.99, 0.995, 0.999):
        for s in (0.005, 0.01, 0.02):
            for R in (1.0, 0.04):
                cap = s / (1 - g)
                h3[f"gamma={g},s={s},R={R}"] = {"s_over_1mg": round(cap, 3), "delay_to_horizon_end": bool(cap > R)}
    out["H3_delay_condition"] = h3
    out["H3_note"] = ("완료 직전 칸의 보조 보상 s가 s/(1-γ) > R을 만족하면, 최적 정책은 완료를 horizon 끝까지 미룬다. "
                      "OmniReset처럼 성공 상태에 머무는 동안 매 step 성공 보상을 주고 종료하지 않으면 이 유인이 없다. "
                      "보조 항의 실제 꼴, γ, 성공 보상 스케일이 논문에 없어 이 결과는 조건부다.")

    path = ROOT / "analysis" / "reward_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", path)


if __name__ == "__main__":
    main()
