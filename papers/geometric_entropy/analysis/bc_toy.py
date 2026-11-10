"""논문의 두 기전을 2차원 토이에서 본다 (CPU numpy, 1스레드). 논문 실험이 아니다.

과제: 시작점 (-1, y0)에서 목표 (1, 0)까지 이동. 원점에 반지름 0.2 장애물이 있다.
시연 경로: 시작 → 경유점 (0, h) → 목표. h>0은 장애물 위쪽, h<0은 아래쪽 전략.
정책 두 가지: (a) kNN(K=10) BC, 상태 (x, y)에서 가까운 시연들의 평균 속도. (b) 2차 다항식 최소제곱(단봉 MSE 회귀의 대리).

두 실험:
  E1 (coverage, 도움): 위쪽 한 전략만 쓰고, 학습 시작점 퍼짐 s_tr을 늘린다. 시험 시작점은 y0 ∈ [-0.6, 0.6].
  E2 (mode averaging, 해): 학습 시작점 퍼짐 0.2. 아래쪽 전략 비율 q를 늘린다.
      q가 작으면 이웃이 대부분 위쪽이라 괜찮고, q가 커지면 이웃의 평균이 장애물 쪽으로 향한다.
  H_G: 같은 시연 집합의 H_G(2차원 버전, 논문 식 1–2와 같은 정의, T=50)를 함께 기록해 성공률과 방향을 비교한다.

실행: python3 analysis/bc_toy.py → analysis/bc_toy_results.json
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from entropy_toy import h_g  # 같은 정의의 H_G (식 1–2)

HERE = Path(__file__).resolve().parent
DT = 0.05
RHO = 0.2          # 장애물 반지름
K_NN = 10
T_RES = 50         # 호 길이 재표집 점 수(논문과 같은 값)
N_DEMO = 60        # 시연 수
N_TEST = 400       # 시험 시작점 수(고정 격자)


def demo_path(y0, h):
    """시작 → 경유점 → 목표를 단위 속도로 따라간 위치 열."""
    pts = np.array([[-1.0, y0], [0.0, h], [1.0, 0.0]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = seg.sum()
    s = np.concatenate([[0], np.cumsum(seg)])
    u = np.arange(0, L + 1e-9, DT)
    xy = np.stack([np.interp(u, s, pts[:, 0]), np.interp(u, s, pts[:, 1])], axis=1)
    return xy


def build_demos(rng, s_tr, q, mode="two"):
    """mode='two': 아래쪽 비율 q (h = +0.4 또는 −0.4).  mode='single': 위쪽만.  mode='continuum': 양쪽 연속 분포."""
    paths, hs = [], []
    for _ in range(N_DEMO):
        y0 = rng.uniform(-s_tr, s_tr)
        if mode == "single":
            h = 0.4
        elif mode == "two":
            h = -0.4 if rng.uniform() < q else 0.4
        else:  # continuum: |h| ∈ [0.25, 0.45], 부호는 반반
            h = rng.uniform(0.25, 0.45) * (1 if rng.uniform() < 0.5 else -1)
        paths.append(demo_path(y0, h))
        hs.append(h)
    S, A = [], []
    for xy in paths:
        S.append(xy[:-1])
        A.append((xy[1:] - xy[:-1]) / DT)
    return np.concatenate(S), np.concatenate(A), paths


def feats(Q):
    x, y = Q[:, 0], Q[:, 1]
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], 1)


def make_policy(S, A, kind):
    """kind='knn': 이웃 평균(비모수). kind='poly2': 2차 다항식 최소제곱(단봉 MSE 회귀 대리). 두 경우 모두 속도 크기를 1로 자른다."""
    if kind == "knn":
        def pol(Q):
            d2 = np.sum(Q ** 2, 1)[:, None] + np.sum(S ** 2, 1)[None, :] - 2 * Q @ S.T
            idx = np.argpartition(d2, K_NN, axis=1)[:, :K_NN]
            return A[idx].mean(axis=1)
    else:
        W = np.linalg.lstsq(feats(S), A, rcond=None)[0]

        def pol(Q):
            return feats(Q) @ W
    def clipped(Q):
        V = pol(Q)
        n = np.linalg.norm(V, axis=1, keepdims=True)
        return V / np.maximum(n, 1.0)
    return clipped


def rollout(policy, y0s, max_steps=200):
    P = np.stack([np.full_like(y0s, -1.0), y0s], axis=1)
    alive = np.ones(len(y0s), bool)
    success = np.zeros(len(y0s), bool)
    collided = np.zeros(len(y0s), bool)
    for _ in range(max_steps):
        idx = np.where(alive)[0]
        if idx.size == 0:
            break
        V = policy(P[idx])
        P[idx] += DT * V
        hit = np.linalg.norm(P[idx], axis=1) < RHO
        collided[idx[hit]] = True
        alive[idx[hit]] = False
        done = (P[idx, 0] >= 1.0) & (~hit)
        success[idx[done]] = True
        alive[idx[done]] = False
    return success, collided


def h_of_demos(paths):
    """2차원 디스크립터: 호 길이 재표집(T=50) → 끝점 고정 → 시작–끝 거리로 정규화 → 식 1–2."""
    X = []
    for xy in paths:
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        u = np.linspace(0, s[-1], T_RES)
        r = np.stack([np.interp(u, s, xy[:, 0]), np.interp(u, s, xy[:, 1])], axis=1)
        r = r - r[-1]
        r = r / (np.linalg.norm(r[0]) + 1e-9)
        X.append(r.reshape(-1))
    return h_g(np.array(X))


def evaluate(s_tr, q, mode, seed, kind):
    rng = np.random.default_rng(seed)
    S, A, paths = build_demos(rng, s_tr, q, mode)
    y0s = np.linspace(-0.6, 0.6, N_TEST)
    succ, col = rollout(make_policy(S, A, kind), y0s)
    return float(succ.mean()), float(col.mean()), float(h_of_demos(paths))


def summarize(vals):
    v = np.array(vals)
    return {"success_mean": round(float(v[:, 0].mean()), 3), "success_sd": round(float(v[:, 0].std()), 3),
            "collision_mean": round(float(v[:, 1].mean()), 3), "H_G_mean": round(float(v[:, 2].mean()), 3)}


def main():
    out = {"setup": {"dt": DT, "obstacle_radius": RHO, "knn_k": K_NN, "T_resample": T_RES,
                     "demos_per_config": N_DEMO, "test_starts": N_TEST, "test_start_range": [-0.6, 0.6],
                     "seeds": 5, "policies": ["knn (K=10 이웃 평균)", "poly2 (2차 다항식 최소제곱, 단봉 MSE 대리)"],
                     "note": "toy; 논문 실험 아님. 실패는 장애물 충돌(collision)과 시간 초과(timeout)를 합친 값 1 − success에서 충돌을 뺀 것."}}
    for kind in ["poly2", "knn"]:
        e1 = {}
        for s_tr in [0.0, 0.1, 0.2, 0.3, 0.5]:
            e1[str(s_tr)] = summarize([evaluate(s_tr, 0.0, "single", 100 + sd, kind) for sd in range(5)])
        out[f"E1_coverage_single_mode_{kind}"] = e1
        e2 = {}
        for q in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            e2[str(q)] = summarize([evaluate(0.2, q, "two", 200 + sd, kind) for sd in range(5)])
        out[f"E2_two_mode_fraction_q_{kind}"] = e2
    out["E2b_continuum_both_sides_poly2"] = summarize([evaluate(0.2, 0.0, "continuum", 300 + sd, "poly2") for sd in range(5)])
    out["E2b_continuum_both_sides_knn"] = summarize([evaluate(0.2, 0.0, "continuum", 300 + sd, "knn") for sd in range(5)])
    (HERE / "bc_toy_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
