"""추가 검증 1: 상자 뚜껑에서 접촉점·법선을 CEM으로 최적화하면 위치 보상·코드 보상·Eq. 3이 어떤 접촉을 고르고,
그 접촉이 뚜껑을 실제로 여는지(wrench 방향 검사) 본다.

설정 (wrench_toy.py와 같은 상자·기저·μ = 0.1·보상 식, 가정은 그대로)
- 상자 반치수 (0.10, 0.075, 0.05) m, 뚜껑 경첩: 뒤쪽 윗모서리 (x = −0.10, z = +0.05), 축 y.
- 사람 기준 접촉: 앞면 윗모서리 아래 (0.10, 0, 0.04), 법선 +z (엄지가 위로 밂).
- 로봇 접촉 탐색 공간: 여섯 면 전체의 표면점 + 법선(면 안쪽 방향에서 최대 90° 기울임). 법선은 힘 방향(물체 안쪽)이어야 함.
- 보상 세 가지 (각각 CEM 최대화): 위치 보상 exp(−‖Δp‖²/σ²), σ = 2 cm (가정); 코드 보상; Eq. 3 (v = 0.1, β = 0.1).
- CEM: 표본 1500개 × 25 반복, 엘리트 10%, 시드 8개.
- 뚜껑 검사: 접촉 (p, n)이 뚜껑 영역(z ≥ 0, 가정)에 있고, 마찰원뿔(μ = 0.1, 8모서리) 안 힘 중 경첩 둘레 여는 방향 토크의 최댓값 > 0 이면 "열림".
  μ = 0.5 민감도도 같이 본다. 정적 토크 부호만 보는 단순 검사이며 뚜껑 무게·경첩 마찰·상자 고정은 무시한다.
- 탐색 두 가지: free(여섯 면 전체), lid_only(아랫면 제외, 옆면은 뚜껑 영역 z ≥ 0만).
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import json
import time
from pathlib import Path

import numpy as np

OUT_JSON = Path(__file__).with_name("extra_lid_cem_results.json")
HALF = np.array([0.10, 0.075, 0.05])
RC = float(np.linalg.norm(HALF))
MU, K_EDGES, NB, TOL, VAR, ACTIVE = 0.1, 8, 512, 0.1, 0.1, 1e-3
SIGMA_LOC = 0.02
HINGE = np.array([-HALF[0], 0.0, HALF[2]])
P_HUMAN = np.array([HALF[0], 0.0, HALF[2] - 0.01])
N_HUMAN = np.array([0.0, 0.0, 1.0])
N_SAMP, N_ITER, ELITE, N_SEEDS = 1500, 25, 0.10, 8


def tangent(n):
    nx, ny, nz = n
    s = 1.0 if nz >= 0 else -1.0
    a = -1.0 / (s + nz)
    b = nx * ny * a
    t1 = np.array([1 + s * nx * nx * a, s * b, -s * nx])
    t2 = np.array([b, s + ny * ny * a, -ny])
    return t1 / np.linalg.norm(t1), t2 / np.linalg.norm(t2)


def cone_edges(n, mu, k=K_EDGES):
    n = n / np.linalg.norm(n)
    t1, t2 = tangent(n)
    th = np.linspace(0, 2 * np.pi, k + 1)[:-1]
    e = n[None] + mu * (np.cos(th)[:, None] * t1 + np.sin(th)[:, None] * t2)
    return np.vstack([e / np.linalg.norm(e, axis=1, keepdims=True), n])


def wrench_matrix(points, normals, mu=MU):
    cols = []
    for p, n in zip(points, normals):
        e = cone_edges(n, mu)
        cols.append(np.hstack([e, np.cross(p, e) / RC]))
    return np.vstack(cols).T


def support(W, B):
    return np.clip((B @ W).max(axis=1), 0.0, None)


def code_reward(sh, sr):
    cmd, cur = sh > ACTIVE, sr > ACTIVE
    loss = np.clip((1 - TOL) * sh - sr, 0, None) ** 2 + np.clip(sr - (1 + TOL) * sh, 0, None) ** 2
    return float(((cmd & cur) * np.exp(-loss / VAR)).sum() / max(cmd.sum(), 1e-6))


def eq3_reward(sh, sr):
    lo = np.clip((1 - TOL) * sh - sr, 0, None)
    hi = np.clip(sr - (1 + TOL) * sh, 0, None)
    return float(np.exp(-(lo @ lo + hi @ hi) / VAR))


def loc_reward(p):
    return float(np.exp(-np.sum((p - P_HUMAN) ** 2) / SIGMA_LOC ** 2))


def opening_torque(p, n, mu):
    """접촉 (p, n)의 마찰원뿔 안에서 경첩 둘레 여는 방향 토크의 최댓값(단위 법선력당).
    뚜껑 영역(z >= 0, 상자 윗절반, 가정)에 없는 접촉은 뚜껑이 아니라 몸체를 움직이므로 열림으로 치지 않는다(-inf)."""
    if p[2] < 0:
        return float("-inf")
    F = cone_edges(n, mu, k=16)
    r = p - HINGE
    tau_y = np.cross(r, F)[:, 1]
    return float(np.max(-tau_y))  # 여는 방향 = -y (앞면이 위로 올라가는 회전)


# 여섯 면: 중심, 면 안쪽 법선(힘 방향 = 물체 안쪽), 바깥 법선
FACES = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]


def decode(z, lid_only=False):
    """z (N, 5) in [0,1]^5 -> 표면점 p (N,3), 힘 방향 n (N,3). 면 선택, 면 안 좌표 2, 법선 기울임 2.
    lid_only: 아랫면(−z)을 빼고, 옆면은 z ∈ [0, HALF_z] (뚜껑 영역)만 쓴다."""
    N = len(z)
    nf = 5 if lid_only else 6
    face = np.minimum((z[:, 0] * nf).astype(int), nf - 1)
    u = 2 * z[:, 1] - 1
    v = 2 * z[:, 2] - 1
    P = np.zeros((N, 3))
    Nout = np.zeros((N, 3))
    for i in range(N):
        ax, sg = FACES[face[i]]
        other = [a for a in range(3) if a != ax]
        p = np.zeros(3)
        p[ax] = sg * HALF[ax]
        p[other[0]] = u[i] * HALF[other[0]]
        if lid_only and ax != 2:
            p[2] = z[i, 2] * HALF[2]
        else:
            p[other[1]] = v[i] * HALF[other[1]]
        P[i] = p
        Nout[i, ax] = sg
    # 힘 방향: 면 안쪽 법선에서 최대 90° 기울임 (방위각 z3, 기울기 z4)
    phi = 2 * np.pi * z[:, 3]
    tilt = 0.5 * np.pi * z[:, 4]
    ref = -Nout
    # 면과 직교하는 두 접선
    t_a = np.zeros((N, 3)); t_b = np.zeros((N, 3))
    for i in range(N):
        ax = FACES[face[i]][0]
        o = [a for a in range(3) if a != ax]
        t_a[i, o[0]] = 1.0
        t_b[i, o[1]] = 1.0
    Nn = (np.cos(tilt)[:, None] * ref + np.sin(tilt)[:, None] * (np.cos(phi)[:, None] * t_a + np.sin(phi)[:, None] * t_b))
    return P, Nn


def cem(reward_fn, seed, lid_only=False):
    rng = np.random.default_rng(seed)
    mu_, sig = np.full(5, 0.5), np.full(5, 0.35)
    best = (-np.inf, None)
    n_el = max(1, int(ELITE * N_SAMP))
    for _ in range(N_ITER):
        z = np.clip(mu_ + sig * rng.standard_normal((N_SAMP, 5)), 0, 1 - 1e-9)
        P, Nn = decode(z, lid_only)
        r = np.array([reward_fn(P[i], Nn[i]) for i in range(N_SAMP)])
        idx = np.argsort(-r)[:n_el]
        mu_ = z[idx].mean(0)
        sig = np.maximum(z[idx].std(0), 0.02)
        if r[idx[0]] > best[0]:
            best = (float(r[idx[0]]), (P[idx[0]], Nn[idx[0]]))
    return best


def main():
    t0 = time.time()
    B = np.random.default_rng(0).standard_normal((NB, 6))
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    sh = support(wrench_matrix([P_HUMAN], [N_HUMAN]), B)
    print("human support active dirs:", int((sh > ACTIVE).sum()), "of", NB, flush=True)

    def r_loc(p, n):
        return loc_reward(p)

    def r_code(p, n):
        return code_reward(sh, support(wrench_matrix([p], [n]), B))

    def r_eq3(p, n):
        return eq3_reward(sh, support(wrench_matrix([p], [n]), B))

    rewards = {"location": r_loc, "code": r_code, "eq3": r_eq3}
    out = {"setup": {"box_half_m": HALF.tolist(), "hinge": HINGE.tolist(), "human_contact": P_HUMAN.tolist(),
                     "human_normal": N_HUMAN.tolist(), "mu_reward": MU, "n_samples": N_SAMP, "n_iter": N_ITER,
                     "n_seeds": N_SEEDS, "sigma_loc_m": SIGMA_LOC, "basis": NB},
           "human_opening_torque_mu0.1": round(opening_torque(P_HUMAN, N_HUMAN, 0.1), 4),
           "results": {}}
    for name, fn, lid in [(f"{k}|{m}", f, m == "lid_only") for k, f in rewards.items() for m in ("free", "lid_only")]:
        rows = []
        for s in range(N_SEEDS):
            val, (p, n) = cem(fn, seed=1000 + s, lid_only=lid)
            ot01 = opening_torque(p, n, 0.1)
            ot05 = opening_torque(p, n, 0.5)
            rows.append({
                "reward": round(val, 4),
                "p": np.round(p, 4).tolist(), "n": np.round(n, 3).tolist(),
                "dist_to_human_contact_cm": round(100 * float(np.linalg.norm(p - P_HUMAN)), 2),
                "normal_angle_to_human_deg": round(float(np.degrees(np.arccos(np.clip(n @ N_HUMAN, -1, 1)))), 1),
                "code_reward_of_pick": round(code_reward(sh, support(wrench_matrix([p], [n]), B)), 4),
                "eq3_of_pick": round(eq3_reward(sh, support(wrench_matrix([p], [n]), B)), 4),
                "loc_reward_of_pick": round(loc_reward(p), 4),
                "opening_torque_mu0.1": round(ot01, 4), "opens_mu0.1": bool(ot01 > 0),
                "opening_torque_mu0.5": round(ot05, 4), "opens_mu0.5": bool(ot05 > 0),
            })
            print(name, s, rows[-1]["reward"], rows[-1]["opens_mu0.1"], rows[-1]["dist_to_human_contact_cm"], flush=True)
        summ = {
            "opens_rate_mu0.1": round(float(np.mean([r["opens_mu0.1"] for r in rows])), 3),
            "opens_rate_mu0.5": round(float(np.mean([r["opens_mu0.5"] for r in rows])), 3),
            "median_dist_to_human_cm": round(float(np.median([r["dist_to_human_contact_cm"] for r in rows])), 2),
            "median_normal_angle_deg": round(float(np.median([r["normal_angle_to_human_deg"] for r in rows])), 1),
            "median_reward": round(float(np.median([r["reward"] for r in rows])), 4),
        }
        out["results"][name] = {"summary": summ, "seeds": rows}
        print(name, summ, flush=True)
    out["runtime_s"] = round(time.time() - t0, 1)
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("runtime", out["runtime_s"])


if __name__ == "__main__":
    main()
