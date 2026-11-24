"""토이: 자기지도 대응의 비유일성(대칭 뒤집힘)과 부분 Chamfer, 앵커 정렬의 역할.

논문 주장 두 가지를 아주 작은 곡선 문제로 확인합니다.
  (a) 부분 Chamfer + 거리 보존만으로는 해가 유일하지 않다 (좌우 뒤집힌 해도 같은 손실). 앵커 몇 점이 이를 푼다 (3.3절, Fig. 3(e)).
  (b) 로봇 쪽에 사람 영역에 대응이 없는 여분 영역이 있으면 양방향 Chamfer는 사람 쪽을 늘려 왜곡하고,
      부분 Chamfer는 그렇지 않다 (3.2절, Fig. 3(b)).
논문 방법 전체의 재현이 아닙니다. 지역 운동 손실(Lmotion)과 접촉 분류기는 넣지 않았습니다.
사람 지문 공간은 3차원의 원호(곡선), 로봇 공간은 같은 원호(+ 선택적으로 z로 떨어진 여분 원호)입니다.
매퍼는 (3,128,128,3) MLP, Adam 학습률 1e-3(논문은 1e-4), 손실 가중치 모두 1, 600 스텝, 시드 5개, CPU.

실행: python3 analysis/toy_anchor_symmetry.py   (torch, numpy; 약 1–2분)
결과: analysis/toy_anchor_symmetry_results.json
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

OUT = Path(__file__).resolve().parent / "toy_anchor_symmetry_results.json"
torch.set_num_threads(2)
STEPS, BATCH, SEEDS = 600, 512, 5
N_H, N_R, N_ANCHOR = 1024, 2048, 8
Z_OFF = 1.2  # 여분 원호의 z 간격
LO, HI = np.pi / 6, 5 * np.pi / 6


def arc(s, z=0.0):
    th = LO + (HI - LO) * s
    return np.stack([np.cos(th), np.sin(th), np.full_like(th, z)], axis=-1)


def make_data(redundant):
    rng = np.random.default_rng(0)
    s_h = np.sort(rng.uniform(0, 1, N_H))
    human = arc(s_h) + rng.normal(0, 0.01, (N_H, 3))
    s_r = np.linspace(0, 1, N_R)
    robot = arc(s_r)
    if redundant:  # 사람 영역에 대응이 없는 여분 영역: 같은 원호를 z로 띄운 것, 점 수는 같음
        robot = np.concatenate([robot, arc(s_r, Z_OFF)])
    s_a = np.linspace(0, 1, N_ANCHOR)
    return (s_h, torch.tensor(human, dtype=torch.float32), torch.tensor(robot, dtype=torch.float32),
            torch.tensor(arc(s_a), dtype=torch.float32), torch.tensor(arc(s_a), dtype=torch.float32))


def mlp():
    return nn.Sequential(nn.Linear(3, 128), nn.LeakyReLU(), nn.Linear(128, 128), nn.LeakyReLU(), nn.Linear(128, 3))


def pdist(a):
    return torch.cdist(a, a)


def train(seed, human, robot, anchors_h, anchors_r, full_chamfer, use_anchor):
    torch.manual_seed(seed)
    net = mlp()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(STEPS):
        idx = torch.randint(0, len(human), (BATCH,))
        x = human[idx]
        y = net(x)
        # 부분 Chamfer: 사람 -> 로봇 한 방향
        l_p = torch.cdist(y, robot).min(dim=1).values.mean()
        loss = l_p
        if full_chamfer:  # 양방향: 로봇 -> 매핑된 사람 점
            y_all = net(human)
            loss = loss + torch.cdist(robot, y_all).min(dim=1).values.mean()
        # 거리 보존 (식 2, 제곱 오차; 코드는 L1)
        loss = loss + ((pdist(y) - pdist(x)) ** 2).mean()
        if use_anchor:
            loss = loss + (net(anchors_h) - anchors_r).norm(dim=-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return net(human).numpy()


def metrics(y, s_h, human, redundant):
    on_redundant = float(np.mean(y[:, 2] > Z_OFF / 2)) if redundant else 0.0
    # 매핑된 점의 원호 매개변수(각도)와 사람 매개변수의 상관 (부호 -1이면 뒤집힘)
    th = np.arctan2(y[:, 1], y[:, 0])
    corr = float(np.corrcoef(s_h, th)[0, 1])
    # 거리 보존 상대 오차 (무작위 쌍)
    rng = np.random.default_rng(1)
    i, j = rng.integers(0, len(y), 4000), rng.integers(0, len(y), 4000)
    dh = np.linalg.norm(human[i] - human[j], axis=1)
    dy = np.linalg.norm(y[i] - y[j], axis=1)
    rel = float(np.abs(dy - dh).mean() / dh.mean())
    # 매핑된 점의 각도 범위(1~99 백분위) / 사람 원호의 각도 범위: 1이면 스케일 보존
    lo, hi = np.percentile(th, [1, 99])
    span = float((hi - lo) / (HI - LO))
    return {"on_redundant_frac": on_redundant, "corr_param": corr, "dist_rel_err": rel, "span_ratio": span}


def run(name, redundant, full_chamfer, use_anchor):
    s_h, human, robot, ah, ar = make_data(redundant)
    ys, ms = [], []
    for seed in range(SEEDS):
        y = train(seed, human, robot, ah, ar, full_chamfer, use_anchor)
        ys.append(y)
        ms.append(metrics(y, s_h, human.numpy(), redundant))
    ys = np.stack(ys)
    std = float(ys.std(axis=0).mean())  # Fig. 6과 같은 종류: 시드 간 매핑 위치 표준편차의 평균
    flipped = int(sum(m["corr_param"] < 0 for m in ms))
    out = {
        "config": {"redundant_branch": redundant, "full_chamfer": full_chamfer, "anchors": use_anchor},
        "seeds": SEEDS,
        "flipped_seeds": flipped,
        "corr_param_per_seed": [round(m["corr_param"], 3) for m in ms],
        "seed_std_of_mapped_positions": round(std, 4),
        "on_redundant_frac_mean": round(float(np.mean([m["on_redundant_frac"] for m in ms])), 3),
        "dist_rel_err_mean": round(float(np.mean([m["dist_rel_err"] for m in ms])), 3),
        "span_ratio_mean": round(float(np.mean([m["span_ratio"] for m in ms])), 3),
    }
    print(name, json.dumps(out, ensure_ascii=False))
    return out


def main():
    t0 = time.time()
    res = {
        "partial_noanchor": run("partial_noanchor", False, False, False),
        "partial_anchor": run("partial_anchor", False, False, True),
        "full_chamfer_anchor_redundant": run("full_chamfer_anchor_redundant", True, True, True),
        "partial_anchor_redundant": run("partial_anchor_redundant", True, False, True),
    }
    res["meta"] = {"steps": STEPS, "batch": BATCH, "seeds": SEEDS, "anchors": N_ANCHOR, "z_offset": Z_OFF,
                   "elapsed_sec": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("elapsed", res["meta"]["elapsed_sec"], "s; wrote", OUT)


if __name__ == "__main__":
    main()
