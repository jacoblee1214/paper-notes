"""토이: 상태 별칭(state aliasing)과 과거 행동 요약(TEMPO_ACT 방식)의 효과를 1차원 과제에서 본다.

논문 3절, 부록 A의 주장("같은 관측에서 반대 행동이 필요하면 단일 프레임 정책은 머뭇거리고, 최근 행동의 평균 요약이 이를 푼다")을
가장 작은 형태로 시험한다. 실로봇 결과를 재현하는 것이 아니다. 아래 가정이 결과를 정한다.

과제(1차원 팔)
- 위치 x는 0에서 시작한다. 잠깐 기다린 뒤(3-10 스텝) +속도로 x = 1(물체)까지 가고, 거기서 5-40 스텝 멈춘 뒤(잡기)
  -속도로 x = 0까지 돌아와 멈춘다. 정책은 x만 본다(그리퍼 상태, 시간은 보이지 않는다).
  그러면 x = 0(시작 대기 대 종료), x = 0.5(나가는 중 대 돌아오는 중), x = 1(도착 직후 대 잡는 중)이 별칭이다.
- 행동은 {-1, 0, +1} 세 값(속도 0.04)이고, 정책은 소프트맥스에서 표본을 뽑는다(생성 정책의 다봉 표본 추출을 흉내).
- 성공: 150 스텝 안에 x >= 0.95에 닿고, 마지막 스텝에 x <= 0.1이다.

TEMPO_ACT 대응
- 최근 H = K*L 스텝의 행동을 K개 버킷으로 나눠 평균하고(식 2), 시작 이전은 0으로 채우며, 버킷이 전부 시작 이전이면 pad 플래그를 준다.
- 학습 이력은 시연 행동(교사 강제), 평가 이력은 정책이 방금 뽑은 행동(폐루프)이다. 이 차이가 이 토이가 보는 또 하나의 점이다.

변형: 이력 없음 / K=1(L=30, L=1) / K=10 이고 L=1, 3, 10, 30 / K=2 이고 L=15. 시드 3개, 시드당 롤아웃 300개.
실행: python3 analysis/toy_aliasing.py   (torch CPU, 약 1-2분). 출력: analysis/toy_aliasing_results.json
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OUT = Path(__file__).resolve().parent / "toy_aliasing_results.json"
V = 0.04
T = 150
N_DEMO = 1500
EPOCHS = 6
N_ROLL = 300
SEEDS = (0, 1, 2)
OBS_NOISE = 0.01
ACT_NOISE = 0.003
torch.set_num_threads(1)


def make_demos(rng, n):
    """시연: 행동 {-1,0,1}, 위치. 반환 acts (n, T), xs (n, T) (각 스텝 이전의 위치)."""
    acts = np.zeros((n, T), dtype=np.int64)
    xs = np.zeros((n, T))
    for i in range(n):
        x, phase, wait, dwell = 0.0, 0, int(rng.integers(3, 11)), int(rng.integers(5, 41))
        for t in range(T):
            xs[i, t] = x
            if phase == 0:
                a = 0
                wait -= 1
                if wait <= 0:
                    phase = 1
            elif phase == 1:
                a = 1
                if x + V >= 1.0:
                    phase = 2
            elif phase == 2:
                a = 0
                dwell -= 1
                if dwell <= 0:
                    phase = 3
            elif phase == 3:
                a = -1
                if x - V <= 0.0:
                    phase = 4
            else:
                a = 0
            acts[i, t] = a
            x = x + a * V + rng.normal(0, ACT_NOISE)
    return acts, xs


def bucket_features(acts_so_far, K, L):
    """acts_so_far: (B, t) 지금까지의 행동. 반환 (B, 2K): 버킷 평균 K개와 pad 플래그 K개."""
    B, t = acts_so_far.shape
    H = K * L
    pad = np.zeros((B, H))
    a = np.concatenate([pad, acts_so_far.astype(float)], axis=1)  # 시작 이전은 0
    win = a[:, t:t + H]  # 마지막 H개 (길이 H)
    means = win.reshape(B, K, L).mean(axis=2)
    # 버킷 k는 윈도 안에서 [k*L, (k+1)*L) 구간. 윈도 시작 위치의 에피소드 시각 = t - H + k*L
    starts = t - H + np.arange(K) * L
    ends = starts + L
    is_pad = (ends <= 0).astype(float)[None, :].repeat(B, axis=0)
    return np.concatenate([means, is_pad], axis=1)


def build_xy(acts, xs, K, L, rng):
    n = acts.shape[0]
    X, Y = [], []
    for t in range(T):
        obs = xs[:, t:t + 1] + rng.normal(0, OBS_NOISE, size=(n, 1))
        if K > 0:
            f = bucket_features(acts[:, :t], K, L)
            X.append(np.concatenate([obs, f], axis=1))
        else:
            X.append(obs)
        Y.append(acts[:, t] + 1)
    return np.concatenate(X).astype(np.float32), np.concatenate(Y)


def train(K, L, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    acts, xs = make_demos(rng, N_DEMO)
    X, Y = build_xy(acts, xs, K, L, rng)
    in_dim = X.shape[1]
    net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    n = len(Yt)
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            loss = nn.functional.cross_entropy(net(Xt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


@torch.no_grad()
def rollout(net, K, L, seed, n=N_ROLL):
    rng = np.random.default_rng(10_000 + seed)
    g = torch.Generator().manual_seed(10_000 + seed)
    x = np.zeros(n)
    acts = np.zeros((n, 0), dtype=np.int64)
    reached = np.zeros(n, dtype=bool)
    left_start = np.zeros(n, dtype=bool)
    for t in range(T):
        obs = (x + rng.normal(0, OBS_NOISE, size=n))[:, None]
        feat = np.concatenate([obs, bucket_features(acts, K, L)], axis=1) if K > 0 else obs
        logits = net(torch.from_numpy(feat.astype(np.float32)))
        a = torch.multinomial(torch.softmax(logits, dim=1), 1, generator=g).squeeze(1).numpy() - 1
        x = x + a * V + rng.normal(0, ACT_NOISE, size=n)
        acts = np.concatenate([acts, a[:, None]], axis=1)
        reached |= x >= 0.95
        left_start |= x > 0.15
    success = reached & (x <= 0.1)
    return success, reached, left_start


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


VARIANTS = [("이력 없음 (단일 관측)", 0, 1), ("K=1, L=30 (평균 하나)", 1, 30), ("K=10, L=1 (H=10)", 10, 1),
            ("K=10, L=3 (H=30)", 10, 3), ("K=10, L=10 (H=100)", 10, 10), ("K=10, L=30 (H=300)", 10, 30), ("K=2, L=15 (H=30)", 2, 15),
            ("K=1, L=1 (직전 행동 하나)", 1, 1)]


def main():
    t0 = time.time()
    results = []
    for name, K, L in VARIANTS:
        succ, reach, left = [], [], []
        for s in SEEDS:
            net = train(K, L, s)
            su, re_, le = rollout(net, K, L, s)
            succ.append(su.mean())
            reach.append(re_.mean())
            left.append(le.mean())
        k = int(round(np.mean(succ) * N_ROLL * len(SEEDS)))
        lo, hi = wilson(k, N_ROLL * len(SEEDS))
        row = {"변형": name, "K": K, "L": L, "H": K * L,
               "성공률_시드별": [round(float(x), 3) for x in succ], "성공률_평균": round(float(np.mean(succ)), 3),
               "wilson95_(시드 합산 900회, 독립 근사)": [round(lo, 3), round(hi, 3)],
               "x>=0.95 도달률": round(float(np.mean(reach)), 3), "시작 대기 이탈률": round(float(np.mean(left)), 3)}
        results.append(row)
        print(f"{name:28s} 성공 {row['성공률_평균']:.3f} {row['성공률_시드별']} 도달 {row['x>=0.95 도달률']:.3f} "
              f"이탈 {row['시작 대기 이탈률']:.3f}  ({time.time() - t0:.0f}s)")
    OUT.write_text(json.dumps({"설정": {"V": V, "T": T, "N_DEMO": N_DEMO, "EPOCHS": EPOCHS, "N_ROLL": N_ROLL,
                                      "SEEDS": list(SEEDS)}, "결과": results}, ensure_ascii=False, indent=1),
                   encoding="utf-8")


if __name__ == "__main__":
    main()
