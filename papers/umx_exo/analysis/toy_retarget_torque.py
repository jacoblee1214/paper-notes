"""토이 T1: UME 토크 피드백 경로를 숫자로 따라가 본다 (CPU, numpy만, 수 초).

두 부분으로 나눈다.

A. 되돌림 이득 사슬. 공식 코드에서 읽은 상수(SCALE 0.5, 추종 오차 문턱 1도, tanh 기울기 10, 클립 4 Nm와 1 Nm,
   팔로워 kp)로, 팔로워 접촉 토크가 조작자 손에 어떤 크기로 전달되는지 정적 가정(추종 오차 = 토크/kp)으로 계산한다.
   논문 §2.2는 "transparent torque feedback"이라 하고 F = J^{-T} tau, tau_exo = J^T F 만 적는다.
   정적 가정과 상수는 코드를 읽은 값이며 실기에서 측정한 것이 아니다.

B. 어깨 구면 관절의 야코비안 사상. 외골격(UME)과 OpenArm의 어깨 3관절 축 배치를 MJCF에서 옮겨 적은 상수로 놓고,
   M = J_exo^T J_robot^{-T} (팔로워 관절 토크 -> 외골격 관절 토크)가 얼마나 증폭하는지 본다.
   (1) 가상일 보존: tau_exo . dq_exo = tau_robot . dq_robot 이 수치로 맞는지.
   (2) 외골격 관절각을 표본 추출하고, 팔로워 해는 코드와 같은 방식(의사역행렬 1스텝 x 0.5)으로 연속 추종해 구한다.
   (3) M의 최대 특이값(최악 방향 증폭)과 팔로워 어깨 특이점 근접도의 분포.

가정: 코드는 외골격 자세 행렬을 그대로 팔로워 목표로 쓰지만, 외골격 0 자세가 팔로워의 어떤 자세에 대응하는지는
코드와 논문에 없다(두 모델의 0 자세 회전이 90도 다르다). 그래서 팔로워 가동 범위 안의 중립 자세를 무작위로 12개 잡아 민감도로 본다.
외골격 관절각 표본은 [-1.0, 1.0] rad 상자와 [-1.5, 1.5] rad 상자 두 가지다. 외골격의 실제 가동 범위는 코드에 없어 임의로 잡은 값이다.

실행: python3 analysis/toy_retarget_torque.py   → analysis/toy_retarget_torque_results.json
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_retarget_torque_results.json"


# ---------- A. 되돌림 이득 사슬 ----------
SCALE = 0.5            # 외골격 컨트롤러 SCALE
TOL = np.deg2rad(1.0)  # torque_feedback_tolerance
SHARP = 10.0           # tanh_sharpness
CLIP = {"shoulder_elbow": 4.0, "wrist_gripper": 1.0}   # MAX_TORQUE_4340, MAX_TORQUE_4310 (Nm)
KP = {"shoulder_elbow": 240.0, "wrist": 25.0}          # 팔로워 kp (어깨 240, 손목 24/31/25의 중간값)


def gate_gain(err):
    return SCALE * (np.tanh(SHARP * (np.abs(err) - TOL)) + 1.0) / 2.0


def felt_torque(tau_follower, kp, clip):
    """정적 가정: 추종 오차 err = tau/kp (rad). 한 관절의 오차로 단순화(코드는 3축 벡터 노름)."""
    err = np.abs(tau_follower) / kp
    return np.minimum(clip, gate_gain(err) * np.abs(tau_follower))


def part_a():
    taus = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0])
    res = {"gain_at_err": {}, "shoulder": [], "wrist": []}
    for e_deg in [0, 0.5, 1, 2, 5, 10, 20]:
        res["gain_at_err"][str(e_deg)] = float(gate_gain(np.deg2rad(e_deg)))
    for t in taus:
        f = float(felt_torque(t, KP["shoulder_elbow"], CLIP["shoulder_elbow"]))
        res["shoulder"].append({"tau_follower": float(t), "felt": f, "ratio": f / float(t)})
    wt = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0])
    for t in wt:
        f = float(felt_torque(t, KP["wrist"], CLIP["wrist_gripper"]))
        res["wrist"].append({"tau_follower": float(t), "felt": f, "ratio": f / float(t)})
    # 클립에 처음 닿는 팔로워 토크
    grid = np.linspace(0.01, 60, 60000)
    fs = felt_torque(grid, KP["shoulder_elbow"], CLIP["shoulder_elbow"])
    hit = grid[np.argmax(fs >= CLIP["shoulder_elbow"] - 1e-9)] if np.any(fs >= CLIP["shoulder_elbow"] - 1e-9) else None
    res["shoulder_clip_first_hit_tau"] = None if hit is None else float(hit)
    gw = np.linspace(0.01, 30, 30000)
    fw = felt_torque(gw, KP["wrist"], CLIP["wrist_gripper"])
    hitw = gw[np.argmax(fw >= CLIP["wrist_gripper"] - 1e-9)] if np.any(fw >= CLIP["wrist_gripper"] - 1e-9) else None
    res["wrist_clip_first_hit_tau"] = None if hitw is None else float(hitw)
    res["assumption"] = "정적: 추종 오차 = tau/kp, 한 관절. kp 어깨 240, 손목 25."
    return res


# ---------- B. 어깨 구면 관절 야코비안 ----------
def qmat(w, x, y, z):
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rot(axis, t):
    a = np.array(axis, float)
    a /= np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


# 외골격 오른팔 RJ1..RJ3: (몸체 quat(w,x,y,z), 관절 축). MJCF v6_bimanual/models/robot.xml 에서 옮김
UME = [((0.183013, 0.683013, -0.183013, -0.683013), (0, 0, 1)),
       ((0.5, 0.5, 0.5, 0.5), (0, 0, 1)),
       ((0.353553, 0.612372, -0.612372, -0.353553), (0, 0, 1))]
UME_SITE = (0.5, 0.5, 0.5, 0.5)
# OpenArm 오른팔 joint1..3. MJCF openarm1/models/openarm.xml 에서 옮김
OA = [((1, 0, 0, 0), (0, 0, 1)),
      ((0.7071067811882787, 0.7071067811848163, 0, 0), (-1, 0, 0)),
      ((1, 0, 0, 0), (0, 0, 1))]
OA_SITE = (-4.32978028e-17, 7.07106781e-01, 4.32978028e-17, 7.07106781e-01)
OA_LO = np.array([-1.396263, -0.17453267, -1.570796])
OA_HI = np.array([3.490659, 3.3161253, 1.570796])


def fk(chain, site, q):
    """어깨 사이트의 회전행렬과 월드 정렬 각속도 야코비안(3x3)."""
    r = np.eye(3)
    cols = []
    for (quat, ax), qi in zip(chain, q):
        r = r @ qmat(*quat)
        cols.append(r @ np.array(ax, float))
        r = r @ rot(ax, qi)
    return r @ qmat(*site), np.array(cols).T


def log3(r):
    c = np.clip((np.trace(r) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (2 * np.sin(th))
    return th * w


def ik_track(r_target, q0, iters=30, step=0.5):
    """코드의 ik_ball_joint와 같은 갱신(오차 = R log(R^T R*), pinv, 스텝 0.5)을 여러 번 되풀이한다."""
    q = q0.copy()
    for _ in range(iters):
        r, j = fk(OA, OA_SITE, q)
        err = r @ log3(r.T @ r_target)
        q = np.clip(q + step * np.linalg.pinv(j) @ err, OA_LO, OA_HI)
    r, j = fk(OA, OA_SITE, q)
    return q, float(np.linalg.norm(log3(r.T @ r_target))), j


def part_b(seed=0, n=150, box=1.0, n_neutral=12):
    """외골격 0 자세가 팔로워의 어떤 자세(q_neutral)에 대응하는지는 코드에 없다(프레임 정렬이 보정 값에 의존).
    그래서 팔로워 가동 범위 안에서 q_neutral 을 무작위로 12개 잡고, 각 경우에 대해 상수 회전 R_off 를
    R_robot(q_neutral) = R_off R_exo(0) 로 정한 뒤 통계를 낸다. 결과는 q_neutral 에 대한 민감도이기도 하다."""
    rng = np.random.default_rng(seed)
    per = []
    work_err = []
    for _ in range(n_neutral):
        q_n = rng.uniform(OA_LO + 0.3, OA_HI - 0.3)
        q_n[2] = rng.uniform(-0.8, 0.8)
        r_n, _ = fk(OA, OA_SITE, q_n)
        r_e0, _ = fk(UME, UME_SITE, [0, 0, 0])
        r_off = r_n @ r_e0.T
        amps, errs, sig_r = [], [], []
        for _ in range(n):
            q_e = rng.uniform(-box, box, 3)
            r_e, j_e = fk(UME, UME_SITE, q_e)
            q_r = q_n.copy()
            for s_ in np.linspace(0.05, 1.0, 20):          # 0 -> 목표를 20 스텝으로 추종
                r_s, _ = fk(UME, UME_SITE, s_ * q_e)
                q_r, _, _ = ik_track(r_off @ r_s, q_r, iters=5)
            q_r, e, j_r = ik_track(r_off @ r_e, q_r, iters=30)
            errs.append(e)
            sv = np.linalg.svd(j_r, compute_uv=False)
            sig_r.append(float(sv[-1]))
            if sv[-1] < 1e-6:
                amps.append(np.inf)
                continue
            m = j_e.T @ np.linalg.inv(j_r.T)
            amps.append(float(np.linalg.svd(m, compute_uv=False)[0]))
            tau_r = rng.normal(size=3)
            w = rng.normal(size=3)
            dq_e = np.linalg.solve(j_e, w)
            dq_r = np.linalg.solve(j_r, w)
            work_err.append(abs(float((m @ tau_r) @ dq_e - tau_r @ dq_r)))
        a = np.array(amps)
        ok = np.array(errs) < 1e-3
        fin = a[np.isfinite(a) & ok]
        per.append({"q_neutral": np.round(q_n, 2).tolist(),
                    "converged_frac": float(ok.mean()),
                    "amp_median": float(np.median(fin)) if fin.size else None,
                    "amp_p99": float(np.percentile(fin, 99)) if fin.size else None,
                    "frac_amp_gt_2": float(np.mean(a[ok] > 2)) if ok.any() else None,
                    "frac_sigma_min_lt_0.2": float(np.mean(np.array(sig_r) < 0.2))})
    conv = np.array([p["converged_frac"] for p in per])
    return {"box_rad": box, "n_per_neutral": n, "n_neutral": n_neutral,
            "converged_frac_median": float(np.median(conv)), "converged_frac_min": float(conv.min()),
            "converged_frac_max": float(conv.max()),
            "amp_median_over_neutrals": float(np.median([p["amp_median"] for p in per if p["amp_median"]])),
            "amp_p99_max_over_neutrals": float(max(p["amp_p99"] for p in per if p["amp_p99"])),
            "frac_amp_gt_2_median": float(np.median([p["frac_amp_gt_2"] for p in per if p["frac_amp_gt_2"] is not None])),
            "frac_sigma_min_lt_0.2_median": float(np.median([p["frac_sigma_min_lt_0.2"] for p in per])),
            "virtual_work_max_abs_err": float(max(work_err)),
            "per_neutral": per}


def zero_pose_frames():
    r_e, j_e = fk(UME, UME_SITE, [0, 0, 0])
    r_r, j_r = fk(OA, OA_SITE, [0, 0, 0])
    return {"exo_R0": np.round(r_e, 3).tolist(), "robot_R0": np.round(r_r, 3).tolist(),
            "angle_between_deg": float(np.rad2deg(np.linalg.norm(log3(r_e.T @ r_r)))),
            "exo_axes_angle_deg": [float(np.rad2deg(np.arccos(np.clip(j_e[:, i] @ j_e[:, k], -1, 1))))
                                   for i, k in [(0, 1), (1, 2), (0, 2)]]}


def robot_singularity_scan():
    """OpenArm 어깨 특이점 위치: q2 를 훑으며 최소 특이값을 본다 (q1=0, q3=0)."""
    q2 = np.linspace(OA_LO[1], OA_HI[1], 721)
    smin = []
    for v in q2:
        _, j = fk(OA, OA_SITE, [0.0, v, 0.0])
        smin.append(float(np.linalg.svd(j, compute_uv=False)[-1]))
    smin = np.array(smin)
    i = int(np.argmin(smin))
    return {"q2_at_min_sigma_deg": float(np.rad2deg(q2[i])), "min_sigma": float(smin[i]),
            "q2_range_deg": [float(np.rad2deg(OA_LO[1])), float(np.rad2deg(OA_HI[1]))],
            "frac_of_q2_range_with_sigma_lt_0.2": float(np.mean(smin < 0.2))}


def main():
    res = {"A_feedback_chain": part_a(), "B_zero_pose": zero_pose_frames(),
           "B_robot_singularity": robot_singularity_scan(),
           "B_map_box1.0": part_b(seed=0, n=150, box=1.0),
           "B_map_box1.5": part_b(seed=1, n=150, box=1.5)}
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    a = res["A_feedback_chain"]
    print("gate gain at err(deg):", {k: round(v, 3) for k, v in a["gain_at_err"].items()})
    print("shoulder felt/follower:", [(r["tau_follower"], round(r["felt"], 2)) for r in a["shoulder"]])
    print("wrist felt/follower:", [(r["tau_follower"], round(r["felt"], 2)) for r in a["wrist"]])
    print("clip first hit (sh, wr):", a["shoulder_clip_first_hit_tau"], a["wrist_clip_first_hit_tau"])
    print("zero pose:", res["B_zero_pose"]["angle_between_deg"], res["B_zero_pose"]["exo_axes_angle_deg"])
    print("robot singularity:", res["B_robot_singularity"])
    for k in ("B_map_box1.0", "B_map_box1.5"):
        print(k, {x: (round(v, 3) if isinstance(v, float) else v) for x, v in res[k].items() if x != "per_neutral"})


if __name__ == "__main__":
    main()
