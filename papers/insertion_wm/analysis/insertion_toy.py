"""2차원 peg-in-hole toy: 학습된 모델(오차 있음)로 계획하는 방식과 미지 기하에서 모델 없는 정책을 비교한다.
CPU, numpy 1스레드, 약 1–2분. 실행: python3 analysis/insertion_toy.py  → analysis/insertion_toy_results.json

설계 (논문과 다른 점은 모두 toy의 가정이다. 논문의 TD-MPC2 잠재 모델이나 깊이 영상을 흉내 내지 않는다):
- 환경: 반지름 r=0.5의 점 같은 peg가 y=0 표면의 슬롯(깊이 D=2)에 들어간다. 슬롯 중심 h는 에피소드마다 무작위(±0.3),
  슬롯 폭 W = 2r + c (c = 간극). 간극이 곧 기하(geometry)다. 벽에 막히면 그 스텝은 움직이지 않는다(접촉 = 정지).
- 관측: 슬롯 중심의 잡음 섞인 추정 ĥ (σ=0.01). 간극 c는 관측하지 않는다(깊이 영상이 이를 드러내지 못한다는 가정).
- 계획자 (MPPI, TD-MPC2처럼 샘플 기반): 모델은 실제 동역학과 같지만 벽 위치에 오차 δ가 있다(한 에피소드에서 부호 무작위).
  δ = 0이면 참 모델 오라클이다. N 샘플, 호라이즌 H=8, 반복 2회, 온도 λ=0.1, 매 스텝 재계획(MPC).
- 모델 없는 정책: 학습 기하(c ∈ {0.2, 0.4, 0.6})에서 참 모델 MPPI의 시연을 모은 뒤, 관측 특징 φ=(ĥ−x, −D−y)에 대한
  k-최근접 이웃(k=10) 평균 행동을 정책으로 쓴다(비선형이어야 정렬 후 하강 같은 전환을 표현할 수 있다. 처음에는 선형 정책을 썼으나
  학습 기하에서도 0%였다). 미지 기하에서는 같은 정책을 그대로 쓴다. 학습 기하 성공률을 함께 보고해 기준선이 제대로 학습됐는지 확인한다.
- 평가: 셀당 60 에피소드, 성공률과 Wilson 95% 구간. 미지 기하 c_test ∈ {0.05, 0.1, 0.15}.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "insertion_toy_results.json"

R = 0.5          # peg 반지름
D = 2.0          # 슬롯 깊이
STEP = 0.05      # 스텝당 최대 이동
MAX_T = 200      # 에피소드 최대 스텝
SIG_H = 0.01     # 슬롯 중심 관측 잡음
H_MPC = 8        # 계획 호라이즌
IT = 2           # MPPI 반복
LAM = 0.1        # MPPI 온도
SIG_A = 0.5      # 샘플 잡음 표준편차
EP = 150         # 셀당 에피소드
H_SEEN = 0.3     # 학습 시연의 슬롯 중심 범위 ±H_SEEN
SEEN_C = (0.2, 0.4, 0.6)


def blocked(p, h, hw):
    """p: (..., 2). 벽 영역(y<0, |x-h| > hw - r)이면 True."""
    x, y = p[..., 0], p[..., 1]
    return (y < 0) & (np.abs(x - h) > hw - R)


def mppi_action(p0, h_hat, hw_model, N, rng, mu):
    goal = np.array([h_hat, -D])
    for _ in range(IT):
        eps = rng.standard_normal((N, H_MPC, 2)) * SIG_A
        U = np.clip(mu[None] + eps, -1, 1)
        p = np.repeat(p0[None], N, axis=0)
        cost = np.zeros(N)
        for t in range(H_MPC):
            pn = p + STEP * U[:, t]
            bl = blocked(pn, h_hat, hw_model)
            p = np.where(bl[:, None], p, pn)
            cost += np.linalg.norm(p - goal, axis=1) + 0.2 * bl
        w = np.exp(-(cost - cost.min()) / LAM)
        w /= w.sum()
        mu = (w[:, None, None] * U).sum(0)
    return mu


LAST_SIGNS = []  # 마지막 run_mppi 호출의 (모델 부호, 성공) 목록


def run_mppi(c, delta, N, rng, n_ep, record=False, hr=H_SEEN):
    """delta: 모델의 벽 반폭 오차(부호는 에피소드마다 무작위). 반환: 성공률 배열, 성공 시 스텝 수."""
    succ = []
    LAST_SIGNS.clear()
    steps = []
    demos = []
    for _ in range(n_ep):
        h = rng.uniform(-hr, hr)
        x0 = h + rng.uniform(-0.5, 0.5)
        p = np.array([x0, 3.0])
        W = 2 * R + c
        sgn = rng.choice([-1.0, 1.0])
        hw_model = W / 2 + sgn * delta
        mu = np.zeros((H_MPC, 2))
        ok = False
        for t in range(MAX_T):
            h_hat = h + rng.standard_normal() * SIG_H
            mu = mppi_action(p, h_hat, hw_model - 0.0, N, rng, mu)
            u = np.clip(mu[0], -1, 1)
            if record:
                demos.append((np.array([h_hat - p[0], -D - p[1]]), u.copy()))
            pn = p + STEP * u
            if not blocked(pn, h, W / 2):
                p = pn
            if p[1] <= -D:
                ok = True
                steps.append(t + 1)
                break
            mu = np.concatenate([mu[1:], np.zeros((1, 2))])
        succ.append(ok)
        LAST_SIGNS.append((float(sgn), bool(ok)))
    return np.array(succ), steps, demos


def run_bc(tree, Y, c, n_ep, rng, k=10, hr=H_SEEN):
    succ = []
    for _ in range(n_ep):
        h = rng.uniform(-hr, hr)
        p = np.array([h + rng.uniform(-0.5, 0.5), 3.0])
        W = 2 * R + c
        ok = False
        for t in range(MAX_T):
            h_hat = h + rng.standard_normal() * SIG_H
            phi = np.array([h_hat - p[0], -D - p[1]])
            _, idx = tree.query(phi, k=k)
            u = np.clip(Y[idx].mean(axis=0), -1, 1)
            pn = p + STEP * u
            if not blocked(pn, h, W / 2):
                p = pn
            if p[1] <= -D:
                ok = True
                break
        succ.append(ok)
    return np.array(succ)


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    cc = (p + z * z / (2 * n)) / den
    hh = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(cc - hh, 3), round(cc + hh, 3)]


def cell(succ):
    k = int(succ.sum())
    n = len(succ)
    return {"k": k, "n": n, "rate": round(k / n, 3), "wilson95": wilson(k, n)}


def main():
    t0 = time.time()
    seeds = iter(range(1000, 2000))
    nr = lambda: np.random.default_rng(next(seeds))
    res = {"settings": {"R": R, "D": D, "STEP": STEP, "H_MPC": H_MPC, "IT": IT, "LAM": LAM, "EP": EP,
                        "SEEN_C": SEEN_C, "H_SEEN": H_SEEN, "SIG_H": SIG_H, "k_nn": 10,
                        "seeds": "셀마다 고정 시드(1000부터 증가)"}}

    # 1. 모델 없는 정책: 학습 기하에서 참 모델(δ=0, N=128) 시연
    demo_X, demo_Y = [], []
    rng = nr()
    for c in SEEN_C:
        _, _, demos = run_mppi(c, 0.0, 128, rng, 20, record=True)
        for phi, u in demos:
            demo_X.append(phi)
            demo_Y.append(u)
    X = np.array(demo_X)
    Y = np.array(demo_Y)
    tree = cKDTree(X)
    res["bc_fit"] = {"n_samples": int(len(X)), "policy": "kNN k=10 on (h_hat-x, -D-y)"}

    # 2. BC: 학습 기하(sanity)와 미지 간극, 미지 오프셋(|h| ≤ 0.6, 학습은 0.3)
    bc = {}
    for c in (0.2, 0.4):
        bc[f"clearance_{c}_seen"] = cell(run_bc(tree, Y, c, EP, nr()))
    for c in (0.05, 0.1, 0.15):
        bc[f"clearance_{c}_unseen"] = cell(run_bc(tree, Y, c, EP, nr()))
    bc["offset_0.1_hr0.6_unseen"] = cell(run_bc(tree, Y, 0.1, EP, nr(), hr=0.6))
    res["bc"] = bc

    # 3. 계획자: 모델 오차 δ 대 성공률 (c=0.1, N=128)
    res["planner_vs_model_error_c0.1_N128"] = {}
    for d in (0.0, 0.02, 0.05, 0.1):
        s_, st, _ = run_mppi(0.1, d, 128, nr(), EP)
        res["planner_vs_model_error_c0.1_N128"][str(d)] = {**cell(s_), "median_steps_success": float(np.median(st)) if st else None}

    # 4. 계획자: 간극 c 대 성공률 (δ=0.05, N=128)
    # 3b. 부호별 집계: δ > 0이면 모델이 간극을 과대평가(벽이 실제보다 멀다)하는 에피소드
    sign_split = {}
    for d in (0.05, 0.1):
        run_mppi(0.1, d, 128, nr(), EP)
        arr = np.array(LAST_SIGNS)
        pos = arr[arr[:, 0] > 0]
        neg = arr[arr[:, 0] < 0]
        sign_split[str(d)] = {"over_estimate_clearance_half": cell(pos[:, 1].astype(bool)),
                              "under_estimate_clearance_half": cell(neg[:, 1].astype(bool))}
    res["planner_sign_split_c0.1_N128"] = sign_split

    # 4. 계획자: 간극 c 대 성공률 (δ=0.05, N=128)
    res["planner_vs_clearance_d0.05_N128"] = {}
    for c in (0.05, 0.1, 0.2, 0.4):
        s_, _, _ = run_mppi(c, 0.05, 128, nr(), EP)
        res["planner_vs_clearance_d0.05_N128"][str(c)] = cell(s_)

    # 5. 계획자: 샘플 수 N 대 성공률 (δ=0.05, c=0.05)
    res["planner_vs_samples_d0.05_c0.05"] = {}
    for N in (16, 64, 256):
        s_, _, _ = run_mppi(0.05, 0.05, N, nr(), EP)
        res["planner_vs_samples_d0.05_c0.05"][str(N)] = cell(s_)

    # 6. 미지 기하에서 계획자(δ=0.05, N=128) 대 BC: Fisher (간극 축소, 오프셋 확대)
    comp = {}
    for name, c, hr in (("clearance_0.05", 0.05, H_SEEN), ("clearance_0.1", 0.1, H_SEEN), ("offset_0.1_hr0.6", 0.1, 0.6)):
        sp, _, _ = run_mppi(c, 0.05, 128, nr(), EP, hr=hr)
        sb = run_bc(tree, Y, c, EP, nr(), hr=hr)
        k1, k2 = int(sp.sum()), int(sb.sum())
        p_ = stats.fisher_exact([[k1, EP - k1], [k2, EP - k2]])[1]
        comp[name] = {"planner": cell(sp), "bc": cell(sb), "fisher_p": float(p_)}
    res["planner_vs_bc_unseen"] = comp

    res["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps({k: res[k] for k in ("bc", "planner_vs_model_error_c0.1_N128", "planner_vs_clearance_d0.05_N128",
                                           "planner_vs_samples_d0.05_c0.05", "planner_vs_bc_unseen", "runtime_s")},
                     ensure_ascii=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
