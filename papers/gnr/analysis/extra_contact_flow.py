"""평면 3관절 팔의 접촉(테이블 충돌) 제약 toy: 잔차 flow matching, 절대 flow matching, 최근접 검색의 pass@S (torch CPU).

과제: 관절각 a ∈ R³로 끝점 f(a) ∈ R²를 목표 c에 ε = 2 cm 안으로 맞추고, 모든 링크 관절점의 y가 테이블 높이
Y_TABLE 이상이어야 성공한다. 테이블 제약을 어기는 해(링크가 테이블을 뚫는 자세)는 실패로 판정한다.
기준 행동 a0(c)는 명목 자세에서 한 번의 Jacobian 최소 노름 역변환이라 테이블을 뚫을 수 있다.
- residual: 조건 (c, a0)에서 잔차 Δ = a − a0를 flow matching으로 학습, 샘플 a = a0 + Δ
- absolute: 조건 c에서 a를 직접 flow matching으로 학습
- search: 학습 데이터에서 c에 가장 가까운 S개를 찾아 그 잔차를 a0에 더함
학습 쌍은 성공 조건(테이블 위)을 만족하는 시연만 쓴다(거부 표집). 시험 목표는 성공 자세가 존재하는 자세에서 뽑는다.

평가: 시험 목표 500개, 각 목표마다 S개 표본을 뽑아 하나라도 성공하면 성공(pass@S).
a 추정: 실패율 1 − pass@S ≈ S^(−a)를 S = 1..256에서 로그-로그 적합(시드별 값의 평균 ± 표준편차).

가정(toy 설정): 학습 쌍 N_train ∈ {2000, 40000}, MLP 256폭 3층 flow matching, 3000 step, Euler 16 step,
시드 3개. 링크와 기준 행동은 추가 검증 3과 같다. 테이블은 Y_TABLE = −0.15 m, 접촉력은 다루지 않는다.
실행: .venv/bin/python analysis/extra_contact_flow.py → analysis/extra_contact_flow_results.json
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
LINKS = torch.tensor([0.5, 0.35, 0.25])
Y_TABLE = -0.15
EPS = 0.02
N_TRAIN_LIST, N_TEST, SS, SEEDS = [2_000, 40_000], 500, [1, 2, 4, 8, 16, 32, 64, 128, 256], [0, 1, 2]
STEPS, BATCH, EULER = 3000, 256, 16
Q_NOM = torch.tensor([0.3, 0.3, 0.3])


def joint_points(q):
    """관절 점 세 개(elbow, wrist, EE)의 좌표. base는 원점."""
    ang = torch.cumsum(q, -1)
    pts = []
    x = torch.zeros(q.shape[:-1])
    y = torch.zeros(q.shape[:-1])
    for i in range(3):
        x = x + LINKS[i] * torch.cos(ang[..., i])
        y = y + LINKS[i] * torch.sin(ang[..., i])
        pts.append(torch.stack([x, y], -1))
    return pts


def fk(q):
    return joint_points(q)[-1]


def feasible(q):
    """모든 관절 점(base 제외)이 테이블 위에 있는가."""
    pts = joint_points(q)
    ok = torch.ones(q.shape[:-1], dtype=torch.bool)
    for p in pts:
        ok &= p[..., 1] >= Y_TABLE
    return ok


def jac(q, h=1e-4):
    cols = []
    for i in range(3):
        e = torch.zeros(3)
        e[i] = h
        cols.append((fk(q + e) - fk(q - e)) / (2 * h))
    return torch.stack(cols, -1)


def a0_of(c):
    """명목 자세에서 한 번의 Jacobian 최소 노름 역변환: 기준 행동(테이블 제약은 고려하지 않음)."""
    f0 = fk(Q_NOM)
    J = jac(Q_NOM)
    JJt = J @ J.T + 1e-6 * torch.eye(2)
    step = J.T @ torch.linalg.solve(JJt, (c - f0).T)
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


def sample_feasible(n, g):
    out = []
    got = 0
    while got < n:
        a = -2 + 4 * torch.rand(4 * n, 3, generator=g)
        a = a[feasible(a)][: n - got]
        out.append(a)
        got += len(a)
    return torch.cat(out)


def pass_at(samples_fn, c_test):
    out = {}
    for S in SS:
        a = samples_fn(S)
        ok = ((fk(a) - c_test.repeat_interleave(S, 0)).norm(dim=-1) < EPS) & feasible(a)
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


def run_once(n_train, seed):
    g = torch.Generator().manual_seed(seed)
    a_train = sample_feasible(n_train, g)
    c_train = fk(a_train)
    a0_train = a0_of(c_train)
    delta_train = a_train - a0_train
    a_test = sample_feasible(N_TEST, g)
    c_test = fk(a_test)
    a0_test = a0_of(c_test)
    stats = {"a0_feasible_frac_test": float(feasible(a0_test).float().mean())}

    res_model = fit(torch.cat([c_train, a0_train], -1), delta_train, seed)
    abs_model = fit(c_train, a_train, seed + 50)

    def res_samples(S):
        cond = torch.cat([c_test, a0_test], -1).repeat_interleave(S, 0)
        return a0_test.repeat_interleave(S, 0) + res_model.sample(cond)

    def abs_samples(S):
        return abs_model.sample(c_test.repeat_interleave(S, 0))

    nn_idx = torch.topk(torch.cdist(c_test, c_train), max(SS), largest=False).indices

    def search_samples(S):
        idx = nn_idx[:, :S].reshape(-1)
        return a0_test.repeat_interleave(S, 0) + delta_train[idx]

    curves = {}
    for name, fn in [("residual", res_samples), ("absolute", abs_samples), ("search", search_samples)]:
        curves[name] = pass_at(fn, c_test)
    stats["pass_at_1_nominal_ik_only"] = float(
        ((fk(a0_test) - c_test).norm(dim=-1) < EPS).float().mul(feasible(a0_test).float()).mean())
    return curves, stats


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"mean": round(float(np.mean(xs)), 3), "sd": round(float(np.std(xs, ddof=1)), 3) if len(xs) > 1 else 0.0,
            "n": len(xs)}


def main():
    t0 = time.time()
    res = {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
           "threads": torch.get_num_threads(),
           "config": dict(links=[round(float(x), 3) for x in LINKS], table_y_m=Y_TABLE, eps_m=EPS,
                          n_train_list=N_TRAIN_LIST, test_targets=N_TEST, S_values=SS, steps=STEPS, batch=BATCH,
                          euler_steps=EULER, seeds=SEEDS, q_nominal=Q_NOM.tolist(), model="MLP 256x3"),
           "by_n_train": {}}
    for n_train in N_TRAIN_LIST:
        per = {"residual": [], "absolute": [], "search": []}
        stats = []
        for seed in SEEDS:
            curves, st = run_once(n_train, seed)
            stats.append(st)
            for k in per:
                per[k].append(curves[k])
            print("N", n_train, "seed", seed, {k: round(curves[k][1], 3) for k in curves},
                  {k: round(curves[k][16], 3) for k in curves}, flush=True)
        block = {"pass_at_S": {}, "exponent_a": {}, "nominal_ik_stats": {
            "a0_feasible_frac_test": mean_sd([s["a0_feasible_frac_test"] for s in stats]),
            "pass_at_1_nominal_ik_only": mean_sd([s["pass_at_1_nominal_ik_only"] for s in stats])}}
        for k in per:
            block["pass_at_S"][k] = {str(S): mean_sd([c[S] for c in per[k]]) for S in SS}
            block["exponent_a"][k] = mean_sd([fit_exponent(c) for c in per[k]])
        res["by_n_train"][str(n_train)] = block
    res["wall_time_s"] = round(time.time() - t0, 1)
    (ROOT / "analysis" / "extra_contact_flow_results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1))
    print("wall", res["wall_time_s"], "s")


if __name__ == "__main__":
    main()
