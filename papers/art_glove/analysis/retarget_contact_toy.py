"""토이 C: 장갑의 접촉 라벨을 관절 공간 리타게팅 목적함수에 넣었을 때 (numpy, 1스레드, 1초 미만).

설정 (모든 값은 가정)
- 평면 3관절 손가락(x-z 평면). 사람 링크 L_h = (4.5, 2.8, 2.0) cm, 로봇 링크 = L_h * (1.08, 0.95, 1.05) (embodiment 차이).
- 사람 손끝 목표 P_t: 책상(z=0) 위 2D 곡선. 접촉 프레임(z<0.3 cm)은 z=0으로 고정. 사람 관절 q_h는 사람 링크로 IK(자세만 맞춤).
  (3관절 평면 손가락은 2D 자세를 항상 맞출 수 있으므로, 자세 기반 리타게팅으로는 접촉이 자동으로 맞는다. 그래서 관절 공간 리타게팅을 쓴다.)
- 로봇 목적함수 (프레임마다 Gauss-Newton, 수치 Jacobian):
    min_q ||q - q_h||^2 + lam * c~_t * z_r(q)^2
  c~_t: 장갑 접촉 라벨 (오류율 eps로 뒤집음). lam = 0이면 관절 복사만 함.
- 측정: 실제 접촉 프레임의 로봇 손끝 높이 |z| (mm, 0이 목표)와 관통(z<0) 비율, 접촉 프레임의 관절 편차(도, 중앙값, 리타게팅 비용),
  비접촉 프레임의 관절 편차 90분위수 (잘못된 접촉 라벨이 만드는 비용).
- 시드 0, 400 프레임.
실행: python3 analysis/retarget_contact_toy.py -> analysis/retarget_results.json
"""
import json
import os
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LH = np.array([4.5, 2.8, 2.0])
LR = LH * np.array([1.08, 0.95, 1.05])
N = 400
rng = np.random.default_rng(0)


def fk(q, L):
    a = np.cumsum(q)
    return np.array([np.sum(L * np.cos(a)), np.sum(L * np.sin(a))])


def gn(residual_fn, q, iters=15):
    for _ in range(iters):
        R = residual_fn(q)
        J = np.zeros((R.size, 3))
        for j in range(3):
            dq = np.zeros(3)
            dq[j] = 1e-6
            J[:, j] = (residual_fn(q + dq) - R) / 1e-6
        step = np.linalg.solve(J.T @ J + 1e-9 * np.eye(3), -J.T @ R)
        q = q + step
        if np.linalg.norm(step) < 1e-9:
            break
    return q


# 1) 사람 손끝 목표와 사람 관절 (사람 링크로 자세 IK)
tt = np.linspace(0, 2 * np.pi, N)
P = np.stack([6.0 + 2.5 * np.sin(tt), 0.9 + 1.2 * np.sin(2 * tt + 0.5)], 1)
P = P + rng.normal(0, 0.02, P.shape)              # 장갑 자세 잡음 0.2 mm (가정)
contact_true = P[:, 1] < 0.3
P[contact_true, 1] = 0.0
qh = np.zeros((N, 3))
q = np.array([0.5, 0.5, 0.5])
for t_ in range(N):
    q = gn(lambda qq: fk(qq, LH) - P[t_], q)
    qh[t_] = q
tip_h = np.array([fk(qq, LH) for qq in qh])
human_err_mm = np.linalg.norm(tip_h - P, axis=1).max() * 10


def run(lam, eps):
    flips = rng.uniform(size=N) < eps
    label = np.where(flips, ~contact_true, contact_true)
    q = qh[0].copy()
    zs, dev_c, dev_f, pen = [], [], [], 0
    for t_ in range(N):
        qt = qh[t_]
        if lam > 0 and label[t_]:
            fn = lambda qq, qt=qt: np.concatenate([qq - qt, [np.sqrt(lam) * fk(qq, LR)[1]]])
        else:
            fn = lambda qq, qt=qt: qq - qt
        q = gn(fn, q)
        tip = fk(q, LR)
        d = np.degrees(np.linalg.norm(q - qt))
        if contact_true[t_]:
            zs.append(abs(tip[1]) * 10)
            dev_c.append(d)
            if tip[1] < -1e-3:
                pen += 1
        else:
            dev_f.append(d)
    nc = contact_true.sum()
    return {"contact_height_mm_median": round(float(np.median(zs)), 3),
            "contact_height_mm_p90": round(float(np.percentile(zs, 90)), 3),
            "contact_penetration_frac": round(pen / nc, 3),
            "joint_dev_contact_deg_median": round(float(np.median(dev_c)), 3),
            "joint_dev_free_deg_p90": round(float(np.percentile(dev_f, 90)), 3)}


res = {"config": {"link_human_cm": LH.tolist(), "link_robot_cm": LR.round(3).tolist(), "frames": N,
                  "contact_frac": round(float(contact_true.mean()), 3),
                  "human_ik_max_tip_err_mm": round(float(human_err_mm), 4),
                  "note": "링크 비율, 접촉 기준, 잡음, 라벨 오류율은 모두 가정"},
       "runs": {}}
for eps in (0.0, 0.1):
    for lam in (0.0, 1.0, 10.0, 100.0):
        res["runs"][f"lam={lam:g},eps={eps:g}"] = run(lam, eps)
# 3) 선형화 검산: 접촉 프레임에서 z_after = |z_h| / (1 + lam |J|^2), 관절 이동 한계 |z_h| / |J| (03 §6.4)
def dz_dq(q, eps=1e-6):
    base = fk(q, LR)[1]
    out = np.zeros(3)
    for j in range(3):
        dq = np.zeros(3)
        dq[j] = eps
        out[j] = (fk(q + dq, LR)[1] - base) / eps
    return out


idx = np.where(contact_true)[0]
zh = np.array([fk(qh[i], LR)[1] for i in idx])
J2 = np.array([np.sum(dz_dq(qh[i]) ** 2) for i in idx])
res["linear_check"] = {
    "J_sq_median_cm2_per_rad2": round(float(np.median(J2)), 3),
    "pred_contact_height_mm_median": {f"lam={l:g}": round(float(np.median(np.abs(zh) / (1 + l * J2)) * 10), 4) for l in (1.0, 10.0, 100.0)},
    "pred_contact_height_mm_p90": {f"lam={l:g}": round(float(np.percentile(np.abs(zh) / (1 + l * J2) * 10, 90)), 4) for l in (1.0, 10.0, 100.0)},
    "joint_limit_deg_median": round(float(np.degrees(np.median(np.abs(zh) / np.sqrt(J2)))), 3),
}
(ROOT / "analysis" / "retarget_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
print(json.dumps(res, indent=2, ensure_ascii=False))
