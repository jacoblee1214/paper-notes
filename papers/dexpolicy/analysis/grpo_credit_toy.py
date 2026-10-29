"""GRPO credit 비교 toy (CPU, numpy). 실행: python3 analysis/grpo_credit_toy.py

논문 Eq. 5와 공개 코드 `algos/grpo.py: trajectory_group_advantages`가 같은 advantage를 만드는지 봅니다.

- 논문 Eq. 5: 완료된 궤적 i의 단계 t마다 할인 reward-to-go G_{i,t} = Σ_{u≥t} γ^{u−t} r_{i,u} (γ = 0.95)를 구하고,
  같은 에피소드 나이 t에 도달한 완료 궤적끼리 표준화(모집단 표준편차)
- 공개 코드: 완료된 궤적마다 할인 없는 총 보상 Σ_u r_{i,u}를 rollout 전체 완료 궤적끼리 표준화하고,
  그 궤적의 모든 단계에 같은 값을 줌
- 둘 다 경계에서 잘린 조각은 advantage 0

가상 rollout: 환경 16개 × 512 step(= 8,192, Sec. V-C). 에피소드 길이 120–200 step(가정, ViViDex 클립 길이는 확인 못함).
보상: 접근 구간은 작은 추적 보상, 성공 궤적은 후반에 lift·target 보너스, 실패 궤적은 후반 보상이 낮음.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N_ENV, T_ROLL, GAMMA = 16, 512, 0.95


def make_rollout(rng):
    rewards = np.zeros((T_ROLL, N_ENV))
    starts = np.zeros((T_ROLL, N_ENV), dtype=bool)
    meta = []  # (env, start, end, success, complete)
    for e in range(N_ENV):
        t = -int(rng.integers(0, 150))  # 첫 에피소드는 rollout 이전에 시작했을 수 있음(앞쪽 잘림)
        while t < T_ROLL:
            L = int(rng.integers(120, 201))
            success = rng.random() < 0.4
            s, end = t, t + L
            for u in range(max(s, 0), min(end, T_ROLL)):
                age = u - s
                base = 0.3 + 0.05 * rng.normal()
                if age > 0.6 * L:  # 후반: 성공 궤적만 보너스
                    base += (1.0 if success else -0.1) + 0.1 * rng.normal()
                rewards[u, e] = base
            if s >= 0:
                starts[s, e] = True
            meta.append((e, s, end, success, s >= 0 and end <= T_ROLL))
            t = end
    final_starts = np.array([any(m[0] == e and m[2] == T_ROLL for m in meta) for e in range(N_ENV)])
    return rewards, starts, final_starts, meta


def code_advantages(rewards, episode_starts, final_episode_starts):
    """공개 코드 trajectory_group_advantages와 같은 로직(torch 의존 없이 옮김)."""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    segments = []
    for env_index in range(rewards.shape[1]):
        starts = np.flatnonzero(episode_starts[:, env_index]).tolist()
        if final_episode_starts[env_index]:
            starts.append(rewards.shape[0])
        for start, end in zip(starts[:-1], starts[1:]):
            segments.append((start, end, env_index, rewards[start:end, env_index].sum()))
    returns = np.asarray([s[3] for s in segments], dtype=np.float32)
    if len(returns) >= 2:
        normalized = (returns - returns.mean()) / (returns.std() + 1e-8)
        for (start, end, env_index, _), adv in zip(segments, normalized):
            advantages[start:end, env_index] = adv
    return advantages, len(segments)


def eq5_advantages(rewards, meta):
    adv = np.zeros_like(rewards)
    comp = [(e, s, end) for e, s, end, _, ok in meta if ok]
    G = {}
    for e, s, end in comp:
        r = rewards[s:end, e]
        g = np.zeros(len(r))
        acc = 0.0
        for k in range(len(r) - 1, -1, -1):
            acc = r[k] + GAMMA * acc
            g[k] = acc
        G[(e, s)] = g
    max_len = max((end - s for _, s, end in comp), default=0)
    for t in range(max_len):
        members = [(e, s) for e, s, end in comp if end - s > t]
        if len(members) < 2:
            continue
        vals = np.array([G[m][t] for m in members])
        z = (vals - vals.mean()) / (vals.std() + 1e-8)
        for (e, s), v in zip(members, z):
            adv[s + t, e] = v
    return adv, len(comp)


def main() -> None:
    rng = np.random.default_rng(0)
    stats = []
    for rep in range(20):
        rewards, starts, final_starts, meta = make_rollout(rng)
        a_code, n_code = code_advantages(rewards, starts, final_starts)
        a_eq5, n_eq5 = eq5_advantages(rewards, meta)
        mask = (a_code != 0) | (a_eq5 != 0)
        both = (a_code != 0) & (a_eq5 != 0)
        x, y = a_code[both], a_eq5[both]
        # 에피소드 나이 구간별로, 그 구간 advantage 평균의 부호가 궤적 성공 여부를 따르는지(성공=양수)
        bins = {"age_0-20%": (0.0, 0.2), "age_20-40%": (0.2, 0.4), "age_40-60%": (0.4, 0.6), "age_60-100%": (0.6, 1.0)}
        match = {f"{k}_{w}": [] for k in bins for w in ("code", "eq5")}
        for e, s, end, success, ok in meta:
            if not ok:
                continue
            L = end - s
            for k, (lo, hi) in bins.items():
                sl = slice(s + int(lo * L), s + max(int(hi * L), int(lo * L) + 1))
                match[f"{k}_code"].append((a_code[sl, e].mean() > 0) == success)
                match[f"{k}_eq5"].append((a_eq5[sl, e].mean() > 0) == success)
        stats.append({
            "complete_episodes": n_code, "same_count": n_code == n_eq5,
            "nonzero_steps_frac": round(float(mask.mean()), 3),
            "pearson_r": round(float(np.corrcoef(x, y)[0, 1]), 3),
            "sign_disagree_frac": round(float((np.sign(x) != np.sign(y)).mean()), 3),
            **{f"outcome_match_{k}": round(float(np.mean(v)), 3) for k, v in match.items()},
        })
    keys = ["complete_episodes", "nonzero_steps_frac", "pearson_r", "sign_disagree_frac"] + \
        [k for k in stats[0] if k.startswith("outcome_match_")]
    summary = {k: [round(float(np.mean([s[k] for s in stats])), 3), round(float(np.min([s[k] for s in stats])), 3),
                   round(float(np.max([s[k] for s in stats])), 3)] for k in keys}
    summary["format"] = "[평균, 최소, 최대] over 20 rollouts"
    summary["discount_horizon"] = {"1/(1-gamma)": round(1 / (1 - GAMMA), 1),
                                   "gamma^60": round(GAMMA ** 60, 4), "gamma^120": round(GAMMA ** 120, 5)}
    out = {"setup": {"N_ENV": N_ENV, "T_ROLL": T_ROLL, "GAMMA": GAMMA, "episode_len": [120, 200],
                     "success_rate": 0.4, "reps": 20},
           "summary": summary, "per_rollout": stats}
    path = ROOT / "analysis" / "grpo_credit_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print("wrote", path)


if __name__ == "__main__":
    main()
