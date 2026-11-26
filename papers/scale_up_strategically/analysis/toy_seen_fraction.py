"""장난감 검증: 전체 격자 평가 성공률에서 '훈련에서 본 칸' 비율이 얼마나 차지하는가.

논문의 쌍별 격자 실험(그림 4)은 공개 코드 기준으로 f와 상관없이 6x6 전체 격자를 평가하고(평가 코드 grid 전체 순회),
훈련 지지집합은 V(stair) 5/10/15칸, L(Lrandom) 4/9/14칸, Random 3/6/9칸이다(f = 6/12/18).
이 스크립트는 정책 대신 작은 MLP가 (값1, 값2) -> 라벨 (mod 합) 을 학습하는 합성 과제에서
  - 전체 격자 성공률
  - 훈련에서 보지 않은 칸만의 성공률
을 같은 지지집합으로 비교한다. 논문 정책(VLA)의 행동을 재현하는 것이 아니라, 전체 격자 평균에 seen 비율이
섞인다는 점과 그 크기를 보이는 용도다. 지지집합 리스트는 저장소
https://github.com/yqi19/CoRL2026-Scale-Up-Strategically 커밋 b5b32418bd070b2ce55f3fcde21d742306331ce0
의 data_collection/collection_strategy/lib/pairwise_factor_patterns.py 에서 옮겨 적었다(repo/ 를 import하지 않는다).

실행: python3 analysis/toy_seen_fraction.py   (torch CPU, 수십 초)  → analysis/toy_seen_fraction_results.json
"""
import json
import random
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
K = 6
STAIR_6 = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)]
STAIR_12 = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3)]
LR_6 = [(0, 0), (0, 1), (1, 0), (1, 1)]
LR_12 = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (3, 0), (1, 1), (1, 2)]


def stair18():
    s = [(0, k) for k in range(6)] + [(i, i) for i in range(6)] + [(i, i + 1) for i in range(5)]
    return list(dict.fromkeys(s))


def lr18(rng):
    s = [(0, j) for j in range(6)] + [(i, 0) for i in range(1, 6)]
    return s + rng.sample([(i, j) for i in range(1, 6) for j in range(1, 6)], 3)


def rnd(f, rng):
    side, n = {6: (2, 3), 12: (4, 6), 18: (6, 9)}[f]
    return rng.sample([(i, j) for i in range(side) for j in range(side)], n)


def support(strategy, f, rng):
    if strategy == "V":
        return {6: STAIR_6, 12: STAIR_12, 18: stair18()}[f]
    if strategy == "L":
        return {6: LR_6, 12: LR_12}.get(f) or lr18(rng)
    return rnd(f, rng)


def onehot(i, j):
    x = torch.zeros(2 * K)
    x[i] = 1
    x[K + j] = 1
    return x


def run(strategy, f, seed):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    sup = support(strategy, f, rng)
    tr = [(i, j, (i + 2 * j) % K) for i, j in sup]
    X = torch.stack([onehot(i, j) for i, j, _ in tr] * 20)
    Y = torch.tensor([y for _, _, y in tr] * 20)
    net = nn.Sequential(nn.Linear(2 * K, 64), nn.ReLU(), nn.Linear(64, K))
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    for _ in range(300):
        opt.zero_grad()
        nn.functional.cross_entropy(net(X), Y).backward()
        opt.step()
    allc = [(i, j) for i in range(K) for j in range(K)]
    with torch.no_grad():
        pred = net(torch.stack([onehot(i, j) for i, j in allc])).argmax(1)
    ok = {c: int(p) == (c[0] + 2 * c[1]) % K for c, p in zip(allc, pred)}
    seen = set(sup)
    unseen = [c for c in allc if c not in seen]
    return (100 * sum(ok.values()) / 36, 100 * sum(ok[c] for c in unseen) / len(unseen), len(seen) / 36)


def main():
    res = {}
    for f in (6, 12, 18):
        for s in ("V", "L", "Random"):
            r = [run(s, f, seed) for seed in range(10)]
            res[f"f{f}/{s}"] = {"overall_pct": round(sum(x[0] for x in r) / 10, 1),
                                "unseen_only_pct": round(sum(x[1] for x in r) / 10, 1),
                                "seen_fraction": round(sum(x[2] for x in r) / 10, 3)}
    (ROOT / "analysis" / "toy_seen_fraction_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in res.items():
        print(k, v)


if __name__ == "__main__":
    main()
