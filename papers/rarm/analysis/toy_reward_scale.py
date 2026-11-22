"""토이 1: 공식 정책 학습 코드가 만드는 단계별 보상의 크기 (numpy, 1초 미만).

논문 식 (3)의 진행 보상은 1{p_t > p_(t-1)}, 즉 진행이 갱신될 때 +1이다.
공식 정책 학습 코드(정적 읽기)는 이 +1을 (a) 클립 중앙 프레임으로 옮기고, (b) 앞의 두 샘플에도 1을 채우고,
(c) 렌더 간격(4 환경 스텝) 동안 값을 유지해 모든 스텝에 뿌린 뒤, (d) 성공 보상 5와 스텝당 비용 0.5를 더한다.
이 스크립트는 그 단계 논리만 다시 구현한다(코드를 실행하거나 가져오지 않는다). 세부 값은 analysis/04 의 표를 본다.

질문: 진행 이벤트 하나가 학습 신호에서 차지하는 크기는 얼마이고, 성공 보상 5와 어떻게 비교되는가?

실행: python3 analysis/toy_reward_scale.py   → analysis/toy_reward_scale_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("toy_reward_scale_results.json")
RENDER_INTERVAL = 4   # 코드 상수: 4 환경 스텝마다 이미지 한 장
HALF_SPAN = 2         # (num_frames - 1) // 2, num_frames = 5
SUCCESS_SCALER = 5.0
STEP_COST = 0.5


def episode_rewards(num_steps, event_idx, success_steps=()):
    """event_idx: 이미지 번호(0부터) 중 진행 이벤트(+1)가 발생한 클립 끝 위치."""
    n_img = (num_steps + RENDER_INTERVAL - 1) // RENDER_INTERVAL
    img_steps = [i * RENDER_INTERVAL for i in range(n_img)]
    chunk = [0.0] * n_img
    for e in event_idx:
        chunk[e] = 1.0
    # (a) 클립 중앙 프레임으로 귀속
    mp = {}
    for i, v in enumerate(chunk):
        mp[img_steps[max(0, i - HALF_SPAN)]] = v
    steps_sorted = sorted(mp)
    sr = [mp[s] for s in steps_sorted]
    # (b) 이벤트마다 앞의 두 샘플도 1
    for i, r in enumerate(sr):
        if r == 1:
            sr[max(0, i - 2):i] = [1.0] * min(2, i)
    # (c) 각 스텝은 그 시점 이전 마지막 샘플의 값을 유지
    arr = np.array(steps_sorted)
    out = []
    for t in range(num_steps):
        idx = max(0, int(np.searchsorted(arr, t, side="right")) - 1)
        rm = sr[idx]
        succ = SUCCESS_SCALER if t in success_steps else 0.0
        out.append((rm, rm + succ - STEP_COST))
    rm_sum = sum(x[0] for x in out)
    total = sum(x[1] for x in out)
    return rm_sum, total


res = {"assumptions": {"render_interval": RENDER_INTERVAL, "half_span": HALF_SPAN,
                       "success_scaler": SUCCESS_SCALER, "step_cost": STEP_COST}}

# 1) 격리된 이벤트 하나, 연속 k개
T = 300
iso = {}
base_rm, base_total = episode_rewards(T, [])
for k in (1, 2, 3, 5, 10, 20):
    ev = [40 + i for i in range(k)]
    rm, tot = episode_rewards(T, ev)
    iso[k] = {"rm_sum": rm, "rm_per_event": round(rm / k, 2)}
res["isolated_and_runs"] = iso

# 2) 이벤트 밀도에 따른 한 에피소드 보상 합 (T = 300 스텝, 이미지 75장, 이벤트 위치는 무작위)
rng = np.random.default_rng(0)
n_img = T // RENDER_INTERVAL
dens = {}
for lam in (0.05, 0.1, 0.3, 0.6, 0.9):
    rm_s, tot_s = [], []
    for _ in range(300):
        ev = [i for i in range(4, n_img) if rng.random() < lam]
        rm, tot = episode_rewards(T, ev, success_steps=(T - 1,))
        rm_s.append(rm); tot_s.append(tot)
    dens[str(lam)] = {"rm_sum_mean": round(float(np.mean(rm_s)), 1), "total_return_mean": round(float(np.mean(tot_s)), 1),
                      "events_mean": round(lam * (n_img - 4), 1)}
res["density"] = dens
res["step_cost_total_300_steps"] = -STEP_COST * T
res["success_reward_one_step"] = SUCCESS_SCALER
res["ratio_rm_sum_to_success_at_lambda_0.3"] = round(dens["0.3"]["rm_sum_mean"] / SUCCESS_SCALER, 1)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("격리 이벤트 한 개의 보상 합:", iso[1], "| 연속 20개:", iso[20])
print("밀도별(T=300):", dens)
print("스텝 비용 합(300 스텝):", res["step_cost_total_300_steps"], "| 성공 보상(한 스텝):", SUCCESS_SCALER)
