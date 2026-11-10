"""WARP 아이디어를 보는 numpy 장난감 (논문 모델 아님, CPU 1스레드, 약 1분).

실험 A: 평면 다리(2링크, 골반 높이 자유) 리타게팅. 사람과 로봇의 다리 길이 비율이 다를 때
        접촉 제약(발이 바닥에 있음) 가중치를 바꾸면 발 뜸(float)·미끄럼(skate)·관통(penetration)과
        추종 오차가 어떻게 교환되는지, 그리고 스케일 정규화 방식(N0 없음, N1 전체 높이 비, N2 링크별 비)에
        따라 결과가 얼마나 달라지는지 본다.
실험 B: 평면 3관절 팔에서 손바닥 위치만 맞추는 가중 IK(MINK 류)의 자세 가중치와 초기값(seed)에 따른
        해의 분산, 팔꿈치 자세 오차를 본다. 그리고 링크 길이가 다를 때 방향만 복사하면 손바닥 오차가
        얼마나 생기는지 해석 식으로 확인한다. SEW 해법 자체는 구현하지 않았다.

실행: python3 analysis/retarget_toy.py → analysis/retarget_toy_results.json
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "retarget_toy_results.json"
rng = np.random.default_rng(0)

# ---------------- 실험 A: 평면 다리 ----------------
H1, H2 = 0.44, 0.44      # 사람 허벅지·정강이 (m)
R1, R2 = 0.34, 0.40      # 로봇 허벅지·정강이 (정강이가 더 김 → 비율 불일치)
H_PELVIS = 0.80          # 사람 골반 높이 (m)
T = 120


def ik_knee_forward(rel, L1, L2):
    """골반 기준 발목 위치 rel에 대해 무릎을 앞(+x)으로 꺾은 2링크 IK."""
    d = np.linalg.norm(rel)
    d = np.clip(d, abs(L1 - L2) + 1e-9, L1 + L2 - 1e-9)
    t_hat = rel / np.linalg.norm(rel)
    cg = np.clip((L1**2 + d**2 - L2**2) / (2 * L1 * d), -1, 1)
    g = np.arccos(cg)
    # t_hat을 +g 또는 -g만큼 회전 중 앞(+x)쪽을 택함
    rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    cands = [L1 * rot(s * g) @ t_hat for s in (+1, -1)]
    return max(cands, key=lambda k: k[0])


def human_motion():
    """사람: 0–59 정지(발 x=0), 60–89 발 들어 x=0.3으로 이동, 90–119 정지(발 x=0.3). 골반 x = 발 x / 2."""
    foot = np.zeros((T, 2))
    stance = np.zeros(T, bool)
    for t in range(T):
        if t < 60:
            foot[t] = (0.0, 0.0); stance[t] = True
        elif t < 90:
            u = (t - 60) / 29
            foot[t] = (0.3 * u, 0.10 * np.sin(np.pi * u))
        else:
            foot[t] = (0.3, 0.0); stance[t] = True
    pelvis = np.stack([0.5 * foot[:, 0], np.full(T, H_PELVIS)], 1)
    rel_ankle = foot - pelvis
    rel_knee = np.array([ik_knee_forward(r, H1, H2) for r in rel_ankle])
    return pelvis, rel_knee, rel_ankle, foot, stance


def robot_targets(rel_knee, rel_ankle, mode, s_glob):
    if mode == "N0_raw":
        return rel_knee, rel_ankle
    if mode == "N1_global":
        return s_glob * rel_knee, s_glob * rel_ankle
    if mode == "N2_segment":
        k = (R1 / H1) * rel_knee
        a = k + (R2 / H2) * (rel_ankle - rel_knee)
        return k, a
    raise ValueError(mode)


def leg_fk(al, be):
    u1 = np.array([np.sin(al), -np.cos(al)])
    u2 = np.array([np.sin(al + be), -np.cos(al + be)])
    K = R1 * u1
    A = K + R2 * u2
    return K, A


def solve_leg(relK, relA, x_p, z_h, anchor_x, stance, w_c, x0):
    def res(v):
        z_p, al, be = v
        K, A = leg_fk(al, be)
        out = [K - relK, A - relA, [z_p - z_h]]
        if stance and w_c > 0:
            out.append([w_c * (z_p + A[1]), w_c * (x_p + A[0] - anchor_x)])
        return np.concatenate([np.ravel(o) for o in out])
    sol = least_squares(res, x0, xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=200)
    return sol.x


def run_leg(mode, w_c):
    pelvis, rel_knee, rel_ankle, foot, stance = HUMAN
    s_glob = (R1 + R2) / (H1 + H2)
    relK, relA = robot_targets(rel_knee, rel_ankle, mode, s_glob)
    z_h = s_glob * H_PELVIS if mode != "N0_raw" else H_PELVIS
    x0 = np.array([z_h - 0.7, 0.2, 0.5])
    knee_e, ank_e, float_mm, skate_mm, pen_mm, pelv_dev = [], [], [], [], [], []
    zs = []
    for t in range(T):
        z_p, al, be = solve_leg(relK[t], relA[t], pelvis[t, 0], z_h, foot[t, 0], stance[t], w_c, x0)
        x0 = np.array([z_p, al, be])
        K, A = leg_fk(al, be)
        knee_e.append(np.linalg.norm(K - relK[t]))
        ank_e.append(np.linalg.norm(A - relA[t]))
        wz = z_p + A[1]
        wx = pelvis[t, 0] + A[0]
        zs.append(wz)
        if stance[t]:
            float_mm.append(max(wz, 0.0))
            skate_mm.append(abs(wx - foot[t, 0]))
        pelv_dev.append(abs(z_p - z_h))
    zs = np.array(zs)
    return {
        "knee_err_mm": 1e3 * float(np.mean(knee_e)),
        "ankle_err_mm": 1e3 * float(np.mean(ank_e)),
        "stance_float_mm_mean": 1e3 * float(np.mean(float_mm)),
        "stance_skate_mm_mean": 1e3 * float(np.mean(skate_mm)),
        "penetration_mm_max": 1e3 * float(max(0.0, -zs.min())),
        "pelvis_dev_from_target_mm_mean": 1e3 * float(np.mean(pelv_dev)),
    }


HUMAN = human_motion()
WC = [0.0, 0.1, 1.0, 10.0, 100.0]
MODES = ["N0_raw", "N1_global", "N2_segment"]
exp_a = {}
for m in MODES:
    exp_a[m] = {}
    for w in WC:
        exp_a[m][str(w)] = run_leg(m, w)

# ---------------- 실험 B: 평면 3관절 팔 ----------------
LR = np.array([0.30, 0.27, 0.10])   # 로봇 상완·전완·손 (m)
LH = np.array([0.33, 0.30, 0.10])   # 사람 (사람 팔이 더 김)


def fk(q, L):
    a = np.cumsum(q)
    return np.array([np.sum(L * np.cos(a)), np.sum(L * np.sin(a))])


def make_poses(n):
    poses = []
    while len(poses) < n:
        qh = rng.uniform(-1.2, 1.2, 3) + np.array([0.0, 0.0, 0.0])
        qh[1] = rng.uniform(0.3, 2.2)          # 팔꿈치 굽힘 (양수)
        p_h = fk(qh, LH)
        reach = LR.sum()
        if 0.15 < np.linalg.norm(p_h) < reach * 0.95 and np.linalg.norm(p_h) > 0.02:
            poses.append((qh, p_h))
    return poses


def ik_weighted(p_t, q_post, w_p, seed_q):
    def res(q):
        r = [fk(q, LR) - p_t]
        r.append(np.sqrt(w_p) * (q - q_post))
        return np.concatenate(r)
    sol = least_squares(res, seed_q, xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=300)
    return sol.x


POSES = make_poses(120)
SEEDS = 12
exp_b = {}
for w_p in [0.0, 0.01, 0.03, 0.1, 0.3]:
    palm_e, elbow_e, seed_sd = [], [], []
    for qh, p_h in POSES:
        p_t = p_h
        sols = []
        for _ in range(SEEDS):
            seed_q = rng.normal(0, 0.8, 3) + np.array([0.0, 1.0, 0.0])
            q = ik_weighted(p_t, qh, w_p, seed_q)
            sols.append(q)
        sols = np.array(sols)
        mean_q = sols.mean(0)
        palm_e.append(np.mean([np.linalg.norm(fk(q, LR) - p_t) for q in sols]))
        elbow_e.append(np.mean(np.abs(sols[:, 1] - qh[1])))
        seed_sd.append(np.mean(sols.std(0)))
    exp_b[str(w_p)] = {
        "palm_err_mm_mean": 1e3 * float(np.mean(palm_e)),
        "elbow_err_deg_mean": float(np.rad2deg(np.mean(elbow_e))),
        "seed_std_deg_mean": float(np.rad2deg(np.mean(seed_sd))),
    }

# 길이 불일치에서 방향만 복사할 때 손바닥 오차 (해석 식 검산)
# 사람의 단위 방향 u1,u2를 로봇 링크 길이로 그대로 쓰면 손 위치 차는 (LR1-LH1)u1 + (LR2-LH2)u2 이다.
copy_err = []
for ratio in [0.7, 0.8, 0.9, 1.0, 1.1]:
    lr = LH[:2] * ratio
    qh, _ = POSES[0]
    u_h = np.array([np.cos(np.cumsum(qh)[0]), np.sin(np.cumsum(qh)[0])])
    u_h2 = np.array([np.cos(np.cumsum(qh)[1]), np.sin(np.cumsum(qh)[1])])
    palm_h = LH[0] * u_h + LH[1] * u_h2
    palm_r = lr[0] * u_h + lr[1] * u_h2
    analytic = np.linalg.norm((lr[0] - LH[0]) * u_h + (lr[1] - LH[1]) * u_h2)
    copy_err.append({"length_ratio": ratio,
                     "palm_err_mm_numeric": 1e3 * float(np.linalg.norm(palm_r - palm_h)),
                     "palm_err_mm_formula": 1e3 * float(analytic)})

results = {
    "setup": {"human_leg_m": [H1, H2], "robot_leg_m": [R1, R2], "T_frames": T,
              "human_arm_m": LH.tolist(), "robot_arm_m": LR.tolist(), "poses": len(POSES), "seeds": SEEDS},
    "exp_A_leg": exp_a,
    "exp_B_arm_posture_sweep": exp_b,
    "exp_B_direction_copy_check": copy_err,
}
OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
for m in MODES:
    print(m, {k: round(v["stance_skate_mm_mean"], 2) for k, v in exp_a[m].items()})
print(json.dumps(exp_b, indent=1)[:800])
print(copy_err)
print("wrote", OUT)
