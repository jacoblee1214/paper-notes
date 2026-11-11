"""추가 검증 1: Algorithm 1(침투 해소 LM IK)을 평면 toy로 구현해 일반 접촉 IK와 비교.

설정 (모두 가정, 논문 수치 아님)
- 평면(x, y), 지면 y = 0, 1 cm 높이 상자 (폭 4 cm), 손가락 2개(왼쪽·오른쪽)
- 손가락: 평면 3링크(0.030, 0.030, 0.025 m), 반지름 r = 3 mm, 밑동 (±0.06, 0.07 m), 관절 6개
- 접촉 목표: 각 손가락 끝을 상자 옆면(x = ±(0.02 + r))의 y ∈ [0.2H, 0.8H] 한 점으로 보냄 (NearestProjection은 옆면 위 최근접점)
- Algorithm 1: n = 60 외부 반복. 매 반복마다 접촉 투영과 침투 점(q에서의 sd < r) 계산 후 LM 스텝 1회.
  잔차 = [손끝 - 접촉 목표, λ (침투 점 - (점 + (r - sd)·n))]. λ = 0이면 일반 접촉 IK(침투 항 없음).
- 성공(valid): 두 손끝 접촉 오차 < 2 mm 이고, 모든 링크 점의 침투 깊이 max(r - sd) ≤ 0.5 mm.
- 쌍비교: 같은 q0와 같은 접촉 높이로 500회 시작, λ만 바꿈. 지면 켬/끔 두 조건.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = Path(__file__).with_name("extra_lmik_results.json")

W, H, R = 0.04, 0.01, 0.003
LINKS = np.array([0.030, 0.030, 0.025])
BASES = np.array([[-0.06, 0.07], [0.06, 0.07]])
SIDE_X = np.array([-(W / 2 + R), (W / 2 + R)])  # 왼쪽, 오른쪽 손가락의 접촉 목표 x
LIM = 2.6
N_TRIALS = 500
N_OUTER = 60
TOL_CONTACT = 2e-3
TOL_PEN = 0.5e-3
LAMBDAS = [0.0, 0.3, 1.0, 3.0]
SAMPLE_T = np.array([0.25, 0.5, 0.75, 1.0])  # 링크 위 샘플 점 (손끝 포함)
LO, HI = np.array([-W / 2, 0.0]), np.array([W / 2, H])
NORMALS_FACE = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], float)


def finger_points(th, base):
    """관절 각 th (3,) -> 샘플 점 (4*3, 2) 전부와 손끝 (2,)."""
    a = np.cumsum(th)
    joints = [base]
    for k in range(3):
        joints.append(joints[-1] + LINKS[k] * np.array([np.cos(a[k]), np.sin(a[k])]))
    pts = []
    for k in range(3):
        p0, p1 = joints[k], joints[k + 1]
        for t in SAMPLE_T:
            pts.append(p0 + t * (p1 - p0))
    return np.array(pts), joints[-1]


def all_points(q):
    P1, T1 = finger_points(q[:3], BASES[0])
    P2, T2 = finger_points(q[3:], BASES[1])
    return np.vstack([P1, P2]), np.vstack([T1, T2])


def sd_normal(P, ground):
    """점 P (M,2)의 장애물까지 부호 거리(바깥 양수)와 바깥 법선."""
    C = np.clip(P, LO, HI)
    d = P - C
    dist = np.linalg.norm(d, axis=1)
    sd_b = dist.copy()
    n_b = np.zeros_like(P)
    out = dist > 0
    n_b[out] = d[out] / dist[out, None]
    ins = ~out
    if ins.any():
        Q = P[ins]
        dd = np.stack([Q[:, 0] - LO[0], HI[0] - Q[:, 0], Q[:, 1] - LO[1], HI[1] - Q[:, 1]], 1)
        k = np.argmin(dd, 1)
        sd_b[ins] = -dd[np.arange(len(Q)), k]
        n_b[ins] = NORMALS_FACE[k]
    if not ground:
        return sd_b, n_b
    use_g = P[:, 1] < sd_b
    sd = np.where(use_g, P[:, 1], sd_b)
    n = np.where(use_g[:, None], np.array([0.0, 1.0]), n_b)
    return sd, n


def contact_targets(q, yc):
    """NearestProjection: 손끝에서 가장 가까운 옆면 접촉 목표 (y는 [0.2H, 0.8H]로 제한)."""
    _, tips = all_points(q)
    x = np.zeros((2, 2))
    x[0] = [SIDE_X[0], np.clip(tips[0, 1], yc[0], yc[1])]
    x[1] = [SIDE_X[1], np.clip(tips[1, 1], yc[0], yc[1])]
    return x


def run_alg1(q0, yc, lam, ground):
    q = q0.copy()
    mu = 1e-2
    for _ in range(N_OUTER):
        x = contact_targets(q, yc)
        P, _ = all_points(q)
        sd, n = sd_normal(P, ground)
        pen = (sd < R) if lam > 0 else np.zeros(len(P), bool)
        T_pen = P[pen] + (R - sd[pen])[:, None] * n[pen]

        def resid(qq):
            PP, tt = all_points(qq)
            r = [(tt - x).ravel()]
            if lam > 0:
                r.append(lam * (PP[pen] - T_pen).ravel())
            return np.concatenate(r)

        r0 = resid(q)
        cost0 = r0 @ r0
        J = np.zeros((len(r0), 6))
        eps = 1e-7
        for j in range(6):
            dq = np.zeros(6)
            dq[j] = eps
            J[:, j] = (resid(q + dq) - r0) / eps
        A = J.T @ J
        g = J.T @ r0
        for _ in range(8):
            dq = -np.linalg.solve(A + mu * np.eye(6), g)
            qn = np.clip(q + dq, -LIM, LIM)
            rn = resid(qn)
            if rn @ rn < cost0:
                q = qn
                mu = max(mu / 3, 1e-9)
                break
            mu *= 4
    P, tips = all_points(q)
    x = contact_targets(q, yc)
    contact_err = np.linalg.norm(tips - x, axis=1)
    sd, _ = sd_normal(P, ground)
    depth = float(np.max(np.maximum(R - sd, 0.0)))
    return contact_err, depth


def run_config(q0s, ycs, lam, ground):
    ok_c, ok_p, valid, errs, depths = [], [], [], [], []
    for q0, yc in zip(q0s, ycs):
        ce, dp = run_alg1(q0, yc, lam, ground)
        c_ok = bool(np.all(ce < TOL_CONTACT))
        p_ok = dp <= TOL_PEN
        ok_c.append(c_ok)
        ok_p.append(p_ok)
        valid.append(c_ok and p_ok)
        errs.append(float(np.max(ce)))
        depths.append(dp)
    return {
        "contact_ok": np.array(ok_c), "pen_free": np.array(ok_p), "valid": np.array(valid),
        "max_contact_err_mm_median": 1e3 * float(np.median(errs)),
        "pen_depth_mm_median": 1e3 * float(np.median(depths)),
        "pen_depth_mm_p90": 1e3 * float(np.percentile(depths, 90)),
    }


def summarize(res):
    v, c, p = res["valid"], res["contact_ok"], res["pen_free"]
    n = len(v)
    k = int(v.sum())
    lo_hi = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    return {
        "n": n,
        "valid_rate": k / n,
        "valid_ci95": [float(lo_hi.low), float(lo_hi.high)],
        "contact_ok_rate": float(c.mean()),
        "pen_free_rate": float(p.mean()),
        "contact_ok_but_penetrating_rate": float((c & ~p).mean()),
        "max_contact_err_mm_median": res["max_contact_err_mm_median"],
        "pen_depth_mm_median": res["pen_depth_mm_median"],
        "pen_depth_mm_p90": res["pen_depth_mm_p90"],
    }


def main():
    t0 = time.time()
    rng = np.random.default_rng(20261111)
    q0s = rng.uniform(-1.5, 1.5, size=(N_TRIALS, 6))
    ycs = [np.array([0.2 * H, 0.8 * H]) for _ in range(N_TRIALS)]
    # 접촉 높이는 시도마다 [0.2H, 0.8H] 안에서 무작위로 고른다
    yc_lo = rng.uniform(0.2 * H, 0.5 * H, size=N_TRIALS)
    yc_hi = rng.uniform(0.5 * H, 0.8 * H, size=N_TRIALS)
    ycs = [np.array([a, b]) for a, b in zip(yc_lo, yc_hi)]

    out = {"setup": {"box_W_m": W, "box_H_m": H, "finger_radius_m": R, "links_m": LINKS.tolist(),
                     "bases_m": BASES.tolist(), "joint_limit_rad": LIM, "n_trials": N_TRIALS,
                     "n_outer_steps": N_OUTER, "tol_contact_m": TOL_CONTACT, "tol_pen_m": TOL_PEN,
                     "lambdas": LAMBDAS, "seed": 20261111},
           "results": {}}
    per = {}
    for ground in (True, False):
        for lam in LAMBDAS:
            key = f"ground={'on' if ground else 'off'}_lambda={lam}"
            res = run_config(q0s, ycs, lam, ground)
            per[key] = res
            out["results"][key] = summarize(res)
            print(key, {k: round(v, 3) if isinstance(v, float) else v for k, v in out["results"][key].items()
                        if k in ("valid_rate", "contact_ok_rate", "pen_free_rate")}, flush=True)
    # 쌍비교: 기준(λ = 0)과 λ = 1
    paired = {}
    for ground in (True, False):
        g = "on" if ground else "off"
        base = per[f"ground={g}_lambda=0.0"]["valid"]
        alg = per[f"ground={g}_lambda=1.0"]["valid"]
        b = int(np.sum(alg & ~base))
        c = int(np.sum(base & ~alg))
        p = binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0
        paired[f"ground={g}"] = {"alg_only_valid": b, "base_only_valid": c, "p_sign_test": float(p)}
    out["paired_lambda1_vs_lambda0"] = paired
    out["runtime_s"] = round(time.time() - t0, 2)
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(paired, indent=2))
    print("runtime", out["runtime_s"], "s")


if __name__ == "__main__":
    main()
