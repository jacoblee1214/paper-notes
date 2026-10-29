"""안정성 점수(FSWO) 구현의 toy 검증 (CPU, numpy, 단일 스레드). 실행: python3 analysis/fswo_toy.py

공식 코드 lygra/batch_grasp_optimizer.py의 batched_nnls(투영 경사하강, lr 0.15, 15회 / warm start 4회)와
compute_wrench_score_vectorized(wrench_alpha = 10, 점수 = 잔차 노름, pivot 최소)를 numpy로 옮겨
scipy의 정확한 NNLS와 비교합니다.

- F1: 15회 PGD 점수 vs 정확한 FSWO 최적값 (과대추정, 임계값 0.15 판정 뒤집힘)
- F2: 물체 크기 의존성 (토크 가중 10 m⁻¹ 고정 → 조건수와 lr·L이 크기에 따라 변함)
- F3: warm start 4회의 오차 (접점 하나를 σ만큼 움직였을 때)
- F4: 손바닥 접점의 좌표계 혼용 (손 좌표계 값을 물체 좌표계 접점과 이어 붙였을 때 점수 변화)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)
ALPHA, LR, ITERS, WARM_ITERS, THRESH = 10.0, 0.15, 15, 4, 0.15  # 공식 코드 기본값


def build_systems(pos, nrm, alpha=ALPHA):
    """pivot j마다 A_j (6, k−1), b_j (6,)  — compute_wrench_score_vectorized와 같은 구성."""
    k = len(pos)
    out = []
    for j in range(k):
        others = [i for i in range(k) if i != j]
        A_f = nrm[others].T
        tau = np.cross(pos[others] - pos[j], nrm[others]).T * alpha
        out.append((np.vstack([A_f, tau]), np.concatenate([-nrm[j], np.zeros(3)])))
    return out


def pgd(A, b, iters=ITERS, lr=LR, x0=None):
    x = np.zeros(A.shape[1]) if x0 is None else x0.copy()
    AtA, Atb = A.T @ A, A.T @ b
    for _ in range(iters):
        x = np.maximum(x - lr * (AtA @ x - Atb), 0.0)
    return x, float(np.linalg.norm(A @ x - b))


def score_code(pos, nrm, iters=ITERS, x0s=None, alpha=ALPHA):
    vals, xs = [], []
    for j, (A, b) in enumerate(build_systems(pos, nrm, alpha)):
        x, v = pgd(A, b, iters, x0=None if x0s is None else x0s[j])
        vals.append(v); xs.append(x)
    return min(vals), xs


def score_exact(pos, nrm, alpha=ALPHA):
    return min(nnls(A, b)[1] for A, b in build_systems(pos, nrm, alpha))


def lipschitz(pos, nrm, alpha=ALPHA):
    return max(np.linalg.eigvalsh(A.T @ A)[-1] for A, _ in build_systems(pos, nrm, alpha))


def box_contacts(dims, k=3):
    """직육면체 표면의 무작위 점과 안쪽 법선(접촉력 방향) k개. dims = 변 길이(m)."""
    half = np.asarray(dims) / 2
    area = np.array([dims[1] * dims[2], dims[0] * dims[2], dims[0] * dims[1]] * 2)
    faces = rng.choice(6, size=k, p=area / area.sum())
    pos, nrm = np.empty((k, 3)), np.zeros((k, 3))
    for i, f in enumerate(faces):
        ax, sgn = f % 3, (1 if f < 3 else -1)
        p = rng.uniform(-half, half)
        p[ax] = sgn * half[ax]
        pos[i] = p
        nrm[i, ax] = -sgn  # 안쪽
    return pos, nrm


SHAPES = {"cube": (1.0, 1.0, 1.0), "box": (1.0, 0.6, 0.4), "tool": (1.0, 0.15, 0.15)}
SIZES = [0.04, 0.08, 0.15, 0.30]  # 가장 긴 변 (m)


def main() -> None:
    t0 = time.time()
    out: dict = {}

    # F1·F2: 크기·형상별 PGD vs 정확
    f12 = {}
    N = 1500
    for sname, rel in SHAPES.items():
        for s in SIZES:
            dims = tuple(s * r for r in rel)
            ex, cd, lrL = [], [], []
            for _ in range(N):
                p, n = box_contacts(dims)
                ex.append(score_exact(p, n)); cd.append(score_code(p, n)[0]); lrL.append(LR * lipschitz(p, n))
            ex, cd, lrL = map(np.array, (ex, cd, lrL))
            acc_ex, acc_cd = ex < THRESH, cd < THRESH
            f12[f"{sname}_{int(s*100)}cm"] = {
                "exact_accept_rate": round(float(acc_ex.mean()), 3),
                "code_accept_rate": round(float(acc_cd.mean()), 3),
                "false_reject_among_exact_accept": round(float((acc_ex & ~acc_cd).sum() / max(acc_ex.sum(), 1)), 3),
                "false_accept_rate": round(float((~acc_ex & acc_cd).mean()), 4),
                "median_overestimate_when_exact_accept": round(float(np.median((cd - ex)[acc_ex])) if acc_ex.any() else float("nan"), 4),
                "share_lr_times_L_over_2": round(float((lrL > 2).mean()), 3),
                "median_lr_times_L": round(float(np.median(lrL)), 2),
            }
    out["F1_F2_pgd_vs_exact_by_size"] = f12

    # F2b: 4 cm에서 통과한 배치를 기하적으로 그대로 키웠을 때 (힘 항은 불변, 토크 항만 크기에 비례)
    base = []
    while len(base) < 300:
        p, n = box_contacts((0.04, 0.024, 0.016))
        if score_exact(p, n) < THRESH:
            base.append((p, n))
    f2b = {}
    for s in (1, 2, 4, 7.5):
        sc = np.array([score_exact(p * s, n) for p, n in base])
        f2b[f"x{s}_longest_{round(4 * s, 1)}cm"] = {
            "still_accepted": round(float((sc < THRESH).mean()), 3),
            "median_score": round(float(np.median(sc)), 4),
        }
    f2b["torque_weight_per_m"] = ALPHA
    out["F2b_scaled_same_configuration"] = f2b

    # F3: warm start 4회 — 정확히 안정한(exact < 0.05) 구성에서 접점 하나를 σ만큼 움직임
    f3 = {}
    for s, sigma in ((0.08, 0.0157), (0.15, 0.0157)):
        dims = (s, 0.6 * s, 0.4 * s)
        errs_w, errs_c = [], []
        tries = 0
        while len(errs_w) < 400 and tries < 50000:
            tries += 1
            p, n = box_contacts(dims)
            if score_exact(p, n) > 0.05:
                continue
            _, xs = score_code(p, n)
            q = p.copy()
            q[0] += rng.normal(0, sigma, 3) * np.where(n[0] != 0, 0, 1)  # 같은 면 위 이동
            q[0] = np.clip(q[0], -np.asarray(dims) / 2, np.asarray(dims) / 2)
            exact = score_exact(q, n)
            warm = score_code(q, n, iters=WARM_ITERS, x0s=xs)[0]
            cold = score_code(q, n)[0]
            errs_w.append(warm - exact); errs_c.append(cold - exact)
        errs_w, errs_c = np.array(errs_w), np.array(errs_c)
        f3[f"box_{int(s*100)}cm"] = {
            "n": len(errs_w),
            "warm4_overestimate_median": round(float(np.median(errs_w)), 4),
            "warm4_overestimate_p90": round(float(np.quantile(errs_w, 0.9)), 4),
            "cold15_overestimate_median": round(float(np.median(errs_c)), 4),
        }
    out["F3_warm_start"] = f3

    # F4: 손바닥 접점 좌표계 혼용
    # 손바닥 접점(손 좌표계) p_h = (0.1, 0, 0), 법선(물체를 미는 방향) n_h = (1, 0, 0) 근처.
    # get_align_transform처럼 물체 표면점 q, 바깥 법선 m을 p_h, −n_h에 맞추는 회전·평행이동을 만들고
    # (손바닥 법선 축 회전은 무작위), 손가락 접점 2개는 물체 좌표계에서 무작위로 둔다.
    def rot_from_to(a, b):
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        v, c = np.cross(a, b), float(a @ b)
        if np.linalg.norm(v) < 1e-9:
            return np.eye(3) if c > 0 else -np.eye(3) + 2 * np.outer([0, 0, 1], [0, 0, 1])
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + K + K @ K * ((1 - c) / (v @ v))

    diffs, flips, agree_rank = [], 0, []
    M = 600
    for _ in range(M):
        dims = (0.08, 0.06, 0.05)
        p_all, n_all = box_contacts(dims, k=3)
        q_obj, m_in = p_all[0], n_all[0]           # 손바닥이 닿는 물체 점, 안쪽 법선
        n_h = np.array([1.0, 0, 0]); p_h = np.array([0.10, 0.0, 0.0])
        R = rot_from_to(-m_in, -n_h)               # 바깥 법선(−m_in) → −n_h
        ang = rng.uniform(0, 2 * math.pi)
        c, s_ = math.cos(ang), math.sin(ang)
        Rx = np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])
        R = Rx @ R
        t = p_h - R @ q_obj
        finger_p, finger_n = p_all[1:], n_all[1:]
        # 올바른 방식: 손바닥 접점을 물체 좌표계로 = (q_obj, m_in)
        correct = score_exact(np.vstack([finger_p, q_obj]), np.vstack([finger_n, m_in]))
        # 코드 방식: 손 좌표계 값을 그대로 이어 붙임
        mixed = score_exact(np.vstack([finger_p, p_h]), np.vstack([finger_n, n_h]))
        diffs.append(abs(mixed - correct))
        flips += int((mixed < THRESH) != (correct < THRESH))
        agree_rank.append((correct, mixed))
    cr = np.array(agree_rank)
    from scipy.stats import spearmanr
    out["F4_palm_frame_mismatch"] = {
        "n": M,
        "median_abs_score_diff": round(float(np.median(diffs)), 4),
        "accept_decision_flip_rate": round(flips / M, 3),
        "spearman_correct_vs_mixed": round(float(spearmanr(cr[:, 0], cr[:, 1]).correlation), 3),
        "note": "최종 수용 판정은 손바닥 접점을 빼고 다시 계산하므로, 영향은 최적화 방향에 한정",
    }
    out["runtime_s"] = round(time.time() - t0, 1)

    path = ROOT / "analysis" / "fswo_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("wrote", path)


if __name__ == "__main__":
    main()
