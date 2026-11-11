"""2D 다단계 도달 toy: 계획 스트림의 형식(없음 / one-hot 부분목표 / 인자 있는 코드)과 시각 입력 유무 교차 (torch CPU).

장면: [0.1, 0.9]^2 안의 부분목표 3개. 고정 슬롯 순서로 0 → 1 → 2 를 차례로 도달해야 한다.
행동 헤드는 현재 위치와 슬롯 좌표를 보지만, 현재 위상(어느 슬롯을 향하는지)은 관측에 없다.
계획 스트림이 위상을 알려 준다.
- 계획 형식: none(없음), onehot(현재 슬롯 one-hot), code(현재 슬롯 one-hot + 목표 좌표 인자 2개)
- 계획 시각 입력: on이면 계획 모듈이 현재 슬롯이 강조된 장면을 본다. off이면 위치와 슬롯 좌표만 본다.
학습 2단계: (1) 계획 모듈을 전문가 데이터로 분류·회귀 학습, (2) 계획 모듈을 고정하고 그 예측을 입력으로
행동 헤드를 전문가 행동에 맞춘다(학습과 평가에서 같은 입력 경로).
평가: 닫힌 루프 200 스텝, 성공 = 세 슬롯을 순서대로 도달.

가정(toy 설정, 논문 값 아님): 한 스텝 최대 이동 0.05, 도달 반경 0.03, 전문가는 노이즈 없는 비례 이동.
실행: .venv/bin/python analysis/extra_plan_stream.py → analysis/extra_plan_stream_results.json
"""
import json
import os
import platform
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "4")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cpu")
NSLOT, STEP, RADIUS, T_LIMIT = 3, 0.05, 0.03, 200
EP_TRAIN, EP_EVAL, SEEDS, EP_PLAN, EP_ACT, BATCH = 1000, 500, [0, 1, 2], 10, 70, 1024
CASES = [("none", False), ("onehot", False), ("onehot", True), ("code", False), ("code", True)]


def make_scenes(n, g):
    slots = 0.1 + 0.8 * torch.rand(n, NSLOT, 2, generator=g)
    start = 0.1 + 0.8 * torch.rand(n, 2, generator=g)
    return slots, start


def expert_step(pos, slots, k):
    """전문가 이동, 다음 위상. k == NSLOT 이면 완료 상태(이동 0)."""
    done = k >= NSLOT
    tgt = slots[torch.arange(len(k)), k.clamp(max=NSLOT - 1)]
    d = tgt - pos
    dist = d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    act = d / dist * torch.clamp(dist, max=STEP)
    act = torch.where(done[:, None], torch.zeros_like(act), act)
    new_pos = pos + act
    reached = (~done) & ((tgt - new_pos).norm(dim=-1) < RADIUS)
    return act, new_pos, k + reached.long()


def expert_data(slots, start):
    """전문가 궤적 전체를 (스텝, 에피소드) 순서로 펼친 행 데이터."""
    pos, k = start.clone(), torch.zeros(len(start), dtype=torch.long)
    P, K, A = [], [], []
    for _ in range(T_LIMIT):
        act, new_pos, k_new = expert_step(pos, slots, k)
        P.append(pos), K.append(k), A.append(act)
        pos, k = new_pos, k_new
    return torch.cat(P), torch.cat(K), torch.cat(A)


class Planner(nn.Module):
    def __init__(self, vision):
        super().__init__()
        self.vision = vision
        inp = 2 + 2 * NSLOT + ((NSLOT + 1) if vision else 0)
        self.net = nn.Sequential(nn.Linear(inp, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.cls = nn.Linear(128, NSLOT + 1)
        self.arg = nn.Linear(128, 2)

    def forward(self, pos, slots, hl):
        x = [pos, slots.flatten(1)] + ([hl] if self.vision else [])
        h = self.net(torch.cat(x, -1))
        return self.cls(h), self.arg(h)


class Actor(nn.Module):
    def __init__(self, plan):
        super().__init__()
        self.plan = plan
        extra = {"none": 0, "onehot": NSLOT + 1, "code": NSLOT + 1 + 2}[plan]
        self.net = nn.Sequential(nn.Linear(2 + 2 * NSLOT + extra, 256), nn.GELU(),
                                 nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 2))

    def forward(self, pos, slots, plan_feat):
        rel = (slots - pos[:, None]).flatten(1)  # 슬롯을 현재 위치 기준 상대 좌표로 입력
        x = [pos, rel] + ([plan_feat] if plan_feat is not None else [])
        return self.net(torch.cat(x, -1))


def plan_feature(plan, logits, arg):
    if plan == "none":
        return None
    oh = F.one_hot(logits.argmax(-1), NSLOT + 1).float()
    return oh if plan == "onehot" else torch.cat([oh, arg], -1)


def train_pair(plan, vision, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(100 + seed)
    slots_e, start_e = make_scenes(EP_TRAIN, g)
    P, K, A = expert_data(slots_e, start_e)
    S = slots_e.repeat(T_LIMIT, 1, 1)
    HL = F.one_hot(K, NSLOT + 1).float()
    n = len(P)
    planner = Planner(vision)
    opt = torch.optim.Adam(planner.parameters(), lr=1e-3)
    tgt_arg_all = S[torch.arange(n), K.clamp(max=NSLOT - 1)]
    for _ in range(EP_PLAN):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            logits, arg = planner(P[b], S[b], HL[b])
            loss = F.cross_entropy(logits, K[b])
            if plan == "code":
                loss = loss + F.mse_loss(arg, tgt_arg_all[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    planner.eval()
    # 행동 헤드: 고정된 계획 모듈의 예측을 입력으로 전문가 행동 모방
    with torch.no_grad():
        logits, arg = planner(P, S, HL)
        feat = plan_feature(plan, logits, arg)
    actor = Actor(plan)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    for _ in range(EP_ACT):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            pred = actor(P[b], S[b], None if feat is None else feat[b])
            loss = F.mse_loss(pred * 20, A[b] * 20)
            opt.zero_grad()
            loss.backward()
            opt.step()
    actor.eval()
    return planner, actor


@torch.no_grad()
def rollout(planner, actor, plan, vision, seed):
    g = torch.Generator().manual_seed(7000 + seed)
    slots, cur = make_scenes(EP_EVAL, g)
    k = torch.zeros(EP_EVAL, dtype=torch.long)
    success = torch.zeros(EP_EVAL, dtype=torch.bool)
    for _ in range(T_LIMIT):
        hl = F.one_hot(k, NSLOT + 1).float()
        logits, arg = planner(cur, slots, hl)
        act = actor(cur, slots, plan_feature(plan, logits, arg))
        step_len = act.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        nxt = cur + act / step_len * torch.clamp(step_len, max=STEP)
        done = k >= NSLOT
        tgt = slots[torch.arange(EP_EVAL), k.clamp(max=NSLOT - 1)]
        reached = (~done) & ((tgt - nxt).norm(dim=-1) < RADIUS)
        k = k + reached.long()
        cur = nxt
        success |= k >= NSLOT
    return float(success.float().mean())


def main():
    t0 = time.time()
    res = {"device": str(DEVICE), "torch": torch.__version__, "numpy": np.__version__,
           "python": platform.python_version(), "threads": torch.get_num_threads(),
           "config": dict(slots=NSLOT, step=STEP, radius=RADIUS, horizon=T_LIMIT,
                          train_episodes=EP_TRAIN, eval_episodes=EP_EVAL, planner_epochs=EP_PLAN,
                          actor_epochs=EP_ACT, batch=BATCH, seeds=SEEDS),
           "success_rate": {}}
    for plan, vision in CASES:
        key = f"{plan}_vision_{'on' if vision else 'off'}"
        vals = []
        for seed in SEEDS:
            planner, actor = train_pair(plan, vision, seed)
            vals.append(rollout(planner, actor, plan, vision, seed))
        res["success_rate"][key] = {"mean": round(float(np.mean(vals)), 3),
                                    "std": round(float(np.std(vals)), 3),
                                    "per_seed": [round(v, 3) for v in vals]}
        print(key, res["success_rate"][key], flush=True)
    res["wall_time_s"] = round(time.time() - t0, 1)
    (ROOT / "analysis" / "extra_plan_stream_results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1))
    print("wall", res["wall_time_s"], "s")


if __name__ == "__main__":
    main()
