"""평면 3관절 팔 접촉 toy: 잔차 flow matching, 절대 flow matching, 최근접 검색을 pass@S로 비교 (torch CPU).

과제: 관절각 a ∈ R³가 끝점 f(a) ∈ R²를 목표 c에 ε = 2 cm 안으로 맞춰야 성공한다(시뮬레이터 판정 = 정확한 f).
해가 여럿이다(관절 3개, 제약 2개 → 해의 곡선). 기준 행동 a0(c)는 명목 자세에서 한 번의 Jacobian 역변환이다.
- residual: 조건 (c, a0)에서 잔차 Δ = a − a0를 flow matching으로 학습, 샘플 a = a0 + Δ
- absolute: 조건 c에서 a를 직접 flow matching으로 학습
- search: 학습 데이터에서 c에 가장 가까운 S개를 찾아 그 잔차를 a0에 더함(검색 기준선)
평가: 시험 목표 500개, 각 목표마다 S개 표본을 뽑아 하나라도 시뮬레이터 판정을 통과하면 성공(pass@S).
a 추정: 실패율 1 − pass@S ≈ S^(−a)를 S = 1..256에서 로그-로그 적합.

가정(toy 설정, 논문 값 아님): 학습 쌍 40,000개(관절각 균일 샘플), 1500 step, MLP 256폭 3층
(논문의 1D U-Net 대신 MLP로 단순화: 3차원 행동에는 시간 해상도 구조가 없다고 보았다), Euler 16 step, 시드 2개.
실행: .venv/bin/python analysis/extra_residual_flow.py → analysis/extra_residual_flow_results.json
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

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cpu")
LINKS = torch.tensor([0.5, 0.35, 0.25])
EPS, N_TRAIN, N_TEST, SS, SEEDS = 0.02, 40_000, 500, [1, 2, 4, 8, 16, 32, 64, 128, 256], [0, 1]
STEPS, BATCH, EULER = 1500, 256, 16
Q_NOM = torch.tensor([0.3, 0.3, 0.3])


def fk(q):
    ang = torch.cumsum(q, -1)
    x = (LINKS * torch.cos(ang)).sum(-1)
    y = (LINKS * torch.sin(ang)).sum(-1)
    return torch.stack([x, y], -1)


def jac(q, h=1e-4):
    cols = []
    for i in range(3):
        e = torch.zeros(3)
        e[i] = h
        cols.append((fk(q + e) - fk(q - e)) / (2 * h))
    return torch.stack(cols, -1)  # (2, 3)


def a0_of(c):
    """명목 자세에서 한 번의 Jacobian 최소 노름 역변환: 기준 행동."""
    f0 = fk(Q_NOM)
    J = jac(Q_NOM)
    JJt = J @ J.T + 1e-6 * torch.eye(2)
    step = J.T @ torch.linalg.solve(JJt, (c - f0).T)  # (3, N)
    return Q_NOM + step.T


class FM(nn.Module):
    def __init__(self, cdim, adim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim + adim + 1, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                                 nn.Linear(256, 256), nn.GELU(), nn.Linear(256, adim))

    def v(self, cond, tau, y):
        return self.net(torch.cat([cond, tau[:, None], y], -1))

    @torch.no_grad()
    def sample(self, cond):
        y = torch.randn(cond.shape[0], 3)
        for i in range(EULER):
            tau = torch.full((cond.shape[0],), i / EULER)
            y = y + self.v(cond, tau, y) / EULER
        return y


def fit(cond, target, seed):
    torch.manual_seed(seed)
    m = FM(cond.shape[1], target.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(cond)
    for _ in range(STEPS):
        b = torch.randint(0, n, (BATCH,))
        z = torch.randn(BATCH, target.shape[1])
        tau = torch.rand(BATCH)
        yt = (1 - tau[:, None]) * z + tau[:, None] * target[b]
        loss = ((m.v(cond[b], tau, yt) - (target[b] - z)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m.eval()


def pass_at(samples_fn, c_test):
    out = {}
    for S in SS:
        a = samples_fn(S)
        ok = (fk(a) - c_test.repeat_interleave(S, 0)).norm(dim=-1) < EPS
        out[S] = float(ok.view(-1, S).any(1).float().mean())
    return out


def fit_exponent(curve):
    S = np.array(SS, dtype=float)
    fail = np.array([max(1 - curve[s], 1e-3) for s in SS])
    mask = fail < 0.999
    if mask.sum() < 2:
        return None
    slope = np.polyfit(np.log(S[mask]), np.log(fail[mask]), 1)[0]
    return float(-slope)


def main():
    t0 = time.time()
    res = {"device": str(DEVICE), "torch": torch.__version__, "numpy": np.__version__,
           "python": platform.python_version(), "threads": torch.get_num_threads(),
           "config": dict(links=LINKS.tolist(), eps_m=EPS, train_pairs=N_TRAIN, test_targets=N_TEST,
                          S_values=SS, steps=STEPS, batch=BATCH, euler_steps=EULER, seeds=SEEDS,
                          q_nominal=Q_NOM.tolist(), model="MLP 256x3 (U-Net 대신)"),
           "pass_at_S": {}, "exponent_a": {}}
    curves = {"residual": [], "absolute": [], "search": []}
    exps = {"residual": [], "absolute": [], "search": []}
    for seed in SEEDS:
        g = torch.Generator().manual_seed(seed)
        a_train = -2 + 4 * torch.rand(N_TRAIN, 3, generator=g)
        c_train = fk(a_train)
        a0_train = a0_of(c_train)
        delta_train = a_train - a0_train
        a_test = -2 + 4 * torch.rand(N_TEST, 3, generator=g)
        c_test = fk(a_test)
        a0_test = a0_of(c_test)

        res_model = fit(torch.cat([c_train, a0_train], -1), delta_train, seed)
        abs_model = fit(c_train, a_train, seed + 50)

        def res_samples(S):
            cond = torch.cat([c_test, a0_test], -1).repeat_interleave(S, 0)
            return a0_test.repeat_interleave(S, 0) + res_model.sample(cond)

        def abs_samples(S):
            return abs_model.sample(c_test.repeat_interleave(S, 0))

        dist = torch.cdist(c_test, c_train)
        nn_idx = torch.topk(dist, max(SS), largest=False).indices

        def search_samples(S):
            idx = nn_idx[:, :S].reshape(-1)
            return a0_test.repeat_interleave(S, 0) + delta_train[idx]

        for name, fn in [("residual", res_samples), ("absolute", abs_samples), ("search", search_samples)]:
            curve = pass_at(fn, c_test)
            curves[name].append(curve)
            exps[name].append(fit_exponent(curve))
        print("seed", seed, {n: {S: round(curves[n][-1][S], 3) for S in (1, 16, 256)} for n in curves}, flush=True)

    for name in curves:
        res["pass_at_S"][name] = {str(S): {"mean": round(float(np.mean([c[S] for c in curves[name]])), 3),
                                           "per_seed": [round(c[S], 3) for c in curves[name]]} for S in SS}
        vals = [e for e in exps[name] if e is not None]
        res["exponent_a"][name] = {"per_seed": [None if e is None else round(e, 3) for e in exps[name]],
                                   "mean": None if not vals else round(float(np.mean(vals)), 3)}
    res["wall_time_s"] = round(time.time() - t0, 1)
    (ROOT / "analysis" / "extra_residual_flow_results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1))
    print("wall", res["wall_time_s"], "s")


if __name__ == "__main__":
    main()
