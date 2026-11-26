"""토이: 해석적 점수를 가진 2차원 혼합 사전분포에서 네 가지 안내를 비교한다.

질문: "DDIM 섭동 ω를 최적화 변수 δ로 바꾸면(식 3-4) 같은 실행 가능률에서 사전분포를 더 잘 보존하는가?"
(논문 5.1, 표 1의 주장 구조를 2차원에서 보는 것이다. 로봇 수치와는 무관하다.)

설정(우리가 정한 값, 논문 값이 아님):
  - 데이터: 반지름 2 원 위 6개 가우시안(표준편차 0.15, 균등 가중). 점수는 해석적으로 계산한다(학습 없음).
  - 역방향: DDPM형 DDIM(σ_k는 식 11, η = 1), K = 30단계, ᾱ는 코사인 스케줄.
  - 실행 가능 집합 F = {x1 >= c}, 비용 J(x) = max(0, c - x1), 허용오차 0.01. c = 1.0(두 모드의 중앙을 가름), c = 1.5(한 모드만 안쪽).
  - 방법: 무안내(U), 투영(P, 매 단계 x1 >= sqrt(ᾱ_{k-1}) c), 기울기 안내(G, 코사인 창, g를 훑음),
          σ-안내(S, ω 대신 -g ∇J), 최적화 안내(O1: 식 4의 x_K 자유, 사전항 없음 / O2: x_K에 ½|x_K|² 추가).
          O는 Adam 500회로 직접 슈팅(논문 부록 8.5.2의 L-BFGS 변형처럼 샘플러를 펼쳐 미분)이고 IPOPT가 아니다. 비용은 k <= 5 상태와 종단 벌점.
지표: 실행 가능률, 모드 위 비율(가장 가까운 모드 중심에서 3σ 이내), 유효율(둘 다), 모드 분포와 조건부 사전분포의 총변동거리(TV),\n      모드 안 퍼짐 비율(유효 표본의 중심 대비 RMS 거리 / (σ√2), 사전분포 표본은 약 1).
실행: python3 analysis/toy_guidance.py  ->  analysis/toy_guidance_results.json (CPU, 약 1분)
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)
OUT = Path(__file__).with_name("toy_guidance_results.json")
K, N, M, RAD, STD = 30, 1000, 6, 2.0, 0.15
EPS_TOL = 0.01
MODES = torch.tensor([[RAD * math.cos(2 * math.pi * j / M), RAD * math.sin(2 * math.pi * j / M)] for j in range(M)])

s_ = 0.008
t = torch.arange(K + 1, dtype=torch.float64) / K
ab = torch.cos((t + s_) / (1 + s_) * math.pi / 2) ** 2
ab = (ab / ab[0]).clamp(max=0.9999)  # k = 0 이 깨끗한 쪽
ab[K] = ab[K].clamp(min=1e-4)
ab = ab.float()
sig = torch.zeros(K + 1)
for k in range(1, K + 1):
    sig[k] = torch.sqrt((1 - ab[k - 1]) / (1 - ab[k]) * (1 - ab[k] / ab[k - 1]))


def score(x, k):
    a = ab[k]
    var = a * STD ** 2 + 1 - a
    d = x[:, None, :] - torch.sqrt(a) * MODES[None]
    logw = -(d ** 2).sum(-1) / (2 * var)
    w = torch.softmax(logw, dim=1)
    return -(w[..., None] * d).sum(1) / var


def eps_hat(x, k):
    return -torch.sqrt(1 - ab[k]) * score(x, k)


def mean_step(x, k):
    e = eps_hat(x, k)
    x0 = (x - torch.sqrt(1 - ab[k]) * e) / torch.sqrt(ab[k])
    return torch.sqrt(ab[k - 1]) * x0 + torch.sqrt((1 - ab[k - 1] - sig[k] ** 2).clamp(min=0)) * e


def J(x, c):
    return torch.relu(c - x[..., 0])


def sample(method, c, seed, g=0.0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(N, 2, generator=gen)
    for k in range(K, 0, -1):
        mu = mean_step(x, k)
        om = torch.randn(N, 2, generator=gen)
        if method == "S":
            push = torch.zeros_like(mu)
            push[:, 0] = (J(mu, c) > 0).float()
            x = mu + sig[k] * g * push
            continue
        x = mu + sig[k] * om
        if method == "P":
            x = torch.cat([torch.maximum(x[:, :1], torch.sqrt(ab[k - 1]) * torch.tensor(c)), x[:, 1:]], dim=1)
        if method == "G":
            p = 1 - k / K
            w = math.cos(math.pi * p) ** 2
            push = torch.zeros_like(x)
            push[:, 0] = (J(x, c) > 0).float()
            x = x + g * w * push
    return x


def optimize(c, seed, prior_xK, beta=20.0, mu_term=1000.0, iters=500, lr=0.03):
    gen = torch.Generator().manual_seed(seed)
    xK = torch.randn(N, 2, generator=gen)
    om = torch.stack([torch.randn(N, 2, generator=gen) for _ in range(K)], dim=1)  # 같은 난수열(U와 동일)
    xK = xK.clone().requires_grad_(True)
    delta = om.clone().requires_grad_(True)

    def unroll():
        x = xK
        path = [x]
        for k in range(K, 0, -1):
            x = mean_step(x, k) + sig[k] * delta[:, K - k]
            path.append(x)
        return path

    opt = torch.optim.Adam([xK, delta], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        path = unroll()
        loss = 0.5 * (delta ** 2).sum()
        if prior_xK:
            loss = loss + 0.5 * (xK ** 2).sum()
        for i, x in enumerate(path):  # i = 0 이 k = K, 마지막이 x_0
            if i >= K - 5:  # 마지막 6개 상태(k <= 5)에서만 비용(논문 4.1: 데이터 다양체 근처에서만)
                loss = loss + beta * (J(x, c) ** 2).sum()
        loss = loss + mu_term * (torch.relu(J(path[-1], c) - EPS_TOL) ** 2).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return unroll()[-1].clone(), xK.detach().abs().max().item(), delta.detach().abs().mean().item()


def metrics(x, c):
    x = x.detach()
    feas = (J(x, c) <= EPS_TOL).float()
    dist = torch.cdist(x, MODES)
    near, idx = dist.min(1)
    on = (near <= 3 * STD).float()
    valid = feas * on
    # 조건부 사전분포(모드별 F 안의 질량)
    zs = (c - MODES[:, 0]) / STD
    mass = 0.5 * torch.erfc(zs / math.sqrt(2)) / M
    target = mass / mass.sum()
    cnt = torch.zeros(M)
    for j in range(M):
        cnt[j] = ((idx == j) * (valid > 0)).sum()
    emp = cnt / cnt.sum().clamp(min=1)
    tv = 0.5 * (emp - target).abs().sum().item() if cnt.sum() > 0 else float("nan")
    off = (x - MODES[idx])[valid > 0]
    spread = (off.pow(2).sum(1).mean().sqrt() / (STD * math.sqrt(2))).item() if off.shape[0] > 1 else float("nan")
    return {"spread_ratio": round(spread, 3), "feasible": round(feas.mean().item(), 3), "on_manifold": round(on.mean().item(), 3),
            "valid": round(valid.mean().item(), 3), "tv_to_conditional_prior": round(tv, 3),
            "mean_J": round(J(x, c).mean().item(), 4), "median_dist_to_mode": round(near.median().item(), 3)}


def main():
    t0 = time.time()
    SEEDS = (0, 1, 2)
    res = {"config": {"K": K, "N": N, "modes": M, "radius": RAD, "mode_std": STD, "eps_tol": EPS_TOL, "seeds": list(SEEDS)},
           "mean_over_seeds": {}, "per_seed": {}}
    for c in (1.0, 1.5):
        per = {}
        for seed in SEEDS:
            rows = {"U": metrics(sample("U", c, seed), c), "P": metrics(sample("P", c, seed), c)}
            for g in (0.2, 0.8, 1.6):
                rows[f"G(g={g})"] = metrics(sample("G", c, seed, g), c)
            for g in (1.0, 4.0):
                rows[f"S(g={g})"] = metrics(sample("S", c, seed, g), c)
            for name, pri in (("O1(x_K 자유)", False), ("O2(x_K에 사전항)", True)):
                x0, maxxK, dmean = optimize(c, seed, pri)
                m = metrics(x0, c)
                m["mean_abs_delta"] = round(dmean, 3)
                rows[name] = m
            per[seed] = rows
        res["per_seed"][f"c={c}"] = per
        mean = {}
        for name in per[SEEDS[0]]:
            mean[name] = {k: round(float(np.nanmean([per[sd][name][k] for sd in SEEDS])), 3) for k in per[SEEDS[0]][name]}
            mean[name]["valid_seed_range"] = [min(per[sd][name]["valid"] for sd in SEEDS), max(per[sd][name]["valid"] for sd in SEEDS)]
        res["mean_over_seeds"][f"c={c}"] = mean
        print(f"c = {c}  (3 seeds 평균)")
        for k, v in mean.items():
            print(f"  {k:18s}", v)
    res["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("seconds", res["seconds"])


if __name__ == "__main__":
    main()
