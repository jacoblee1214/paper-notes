"""챌린지 키트 채점 지표의 numpy 포트와 SP-SR 정의 비교 (numpy, scipy, matplotlib만, CPU).

키트 metric_code/ 의 수식을 읽고 손으로 옮겨 쓴 독자 구현이다. 키트 파일은 import하거나 실행하지 않았다.
포함 지표: PEN, SP-SR, MP-SR, ADD-AUC (1–9 cm), RPE, MPPE.

검증 구성
  1. 합성 궤적에 대한 알려진 답 테스트: 항등, 상수 오프셋, 회전 오프셋, 선형 드리프트, 과도 오프셋, 떨림 σ.
  2. SP-SR 정의 세 가지를 같은 합성 궤적에 적용한다.
     (a) SPIDER 양손 모드: 쪽별 시간 평균 오프셋을 뺀 뒤 위치 노름 (flashdexretarget tracking_performance.py _spider_errors)
     (b) CHORD spider_success: 원시 시간 평균 오차, 중심화 없음 (chord metrics.py)
     (c) 키트 SP-SR: 첫 프레임 강체 정렬 후 원시 시간 평균 오차. 키트 트랙 3 설정 (정렬 켜짐)을 쓰고,
         트랙 2 설정 (정렬 꺼짐)은 (b)와 같은 계산이 된다.
  3. 떨림 σ에 따른 AUC와 SP-SR 곡선. 떨림만 있는 경우에는 해석 값이 있어 함께 그린다.

임계값: SP-SR 0.10 m, 0.5 rad / MP-SR 0.03 m, 30도.

실행: python3 -I analysis/extra_kit_metrics.py
  → analysis/extra_kit_metrics_results.json, report/figures/extra_kit_jitter.png
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "analysis" / "extra_kit_metrics_results.json"
OUT_FIG = ROOT / "report" / "figures" / "extra_kit_jitter.png"

SEED = 20261112
FPS = 30.0
T = 300                                   # 10 s
HALF = np.array([0.03, 0.03, 0.05])       # 6 x 6 x 10 cm 상자 반치수 (m)
ADD_THR = np.arange(1, 10) * 0.01         # 1–9 cm (키트는 np.arange(0.01, 0.10, 0.01), 9개)
SP_POS, SP_ROT = 0.10, 0.5
MP_POS, MP_DEG = 0.03, 30.0
MPPE_R = 0.05
MPPE_DIRS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                      [-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
STATIC_DRIFT = 1e-3                       # (a) 정적 쪽 판정 기준 (m)
N_VERT = 200

rng0 = np.random.default_rng(SEED)
RESULTS = {"config": {}, "unit_tests": [], "perturbations": {}, "populations": {}, "jitter": {}}


# ---------------------------------------------------------------- 쿼터니언 기초 (xyzw 순서)

def qmul(a, b):
    ax, ay, az, aw = (a[..., i] for i in range(4))
    bx, by, bz, bw = (b[..., i] for i in range(4))
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def qrot(q):
    x, y, z, w = (q[..., i] for i in range(4))
    m = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                  2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                  2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1)
    return m.reshape(*q.shape[:-1], 3, 3)


def qangle(a, b):
    d = np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arctan2(np.sqrt(np.maximum(0.0, 1.0 - d * d)), d)


def qnorm(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_z(theta):
    theta = np.asarray(theta, dtype=float)
    return np.stack([np.zeros_like(theta), np.zeros_like(theta),
                     np.sin(theta / 2), np.cos(theta / 2)], axis=-1)


def trap9(frac):
    """9점 사다리꼴, x = linspace(0, 1, 9). 키트 _add_auc 와 같은 정규화."""
    dx = 1.0 / (frac.size - 1)
    return float(np.sum((frac[:-1] + frac[1:]) * 0.5) * dx)


# ---------------------------------------------------------------- 지표 포트

def refb(ach, ref):
    """(T,B,7) 참조를 (T,W,B,7)로 펼친다."""
    return np.broadcast_to(ref[:, None], ach.shape)


def pose_errors(ach, ref):
    """(W,B) 시간 평균 위치 오차 (m)와 회전 오차 (rad)."""
    rb = refb(ach, ref)
    pos = np.linalg.norm(ach[..., :3] - rb[..., :3], axis=-1).mean(axis=0)
    ang = qangle(ach[..., 3:], rb[..., 3:]).mean(axis=0)
    return pos, ang


def warp_start(ach, ref):
    """키트 _warp_to_reference_start 와 같은 식. 바디 0의 첫 프레임 포즈로 강체 변환 하나를 만들어 모든 바디에 적용."""
    r0 = ref[0, 0]                                   # (7,)
    a0 = ach[0, :, 0]                                # (W,7)
    conj = a0[:, 3:] * np.array([-1.0, -1.0, -1.0, 1.0])
    q_al = qnorm(qmul(r0[None, 3:], conj))           # (W,4)
    rot = qrot(q_al)                                 # (W,3,3)
    off = r0[None, :3] - np.einsum("wij,wj->wi", rot, a0[:, :3])
    out = np.empty_like(ach)
    out[..., :3] = np.einsum("wij,twbj->twbi", rot, ach[..., :3]) + off[None, :, None, :]
    out[..., 3:] = qmul(q_al[None, :, None, :], ach[..., 3:])
    return out


def sp_raw(ach, ref):
    """(b) CHORD spider_success 와 키트 SP-SR (트랙 2 설정, 정렬 꺼짐): 원시 시간 평균 오차, 바디 평균."""
    pos, ang = pose_errors(ach, ref)
    p, a = pos.mean(-1), ang.mean(-1)
    return (p <= SP_POS) & (a <= SP_ROT), p, a


def sp_warp(ach, ref):
    """(c) 키트 SP-SR, 트랙 3 설정: 첫 프레임 정렬 후 원시 시간 평균 오차."""
    return sp_raw(warp_start(ach, ref), ref)


def sp_centered(ach, ref, lag=False):
    """(a) SPIDER 양손 모드: 쪽별 시간 평균 오프셋을 뺀 위치 노름. lag=True 는 코드의 직전 스텝 참조 지연을 흉내 낸다 (진단용)."""
    rb = refb(ach, ref)
    if lag:
        prev = np.concatenate([rb[:1], rb[:-1]], axis=0)
        e = ach[..., :3] - prev[..., :3]
    else:
        e = ach[..., :3] - rb[..., :3]
    ec = e - e.mean(axis=0, keepdims=True)
    pos = np.linalg.norm(ec, axis=-1).mean(axis=0)              # (W,B)
    ang = qangle(ach[..., 3:], rb[..., 3:]).mean(axis=0)        # (W,B), 회전은 중심화하지 않음
    drift = np.linalg.norm(ref[:, :, :3] - ref[0:1, :, :3], axis=-1).mean(axis=0)   # (B,)
    static = drift < STATIC_DRIFT
    if pos.shape[-1] == 1:
        p, a = pos[:, 0], ang[:, 0]
    else:
        # 코드와 같은 규칙: 쪽 1이 정적이면 쪽 0만, 쪽 0이 정적이면 쪽 1만, 아니면 평균
        p = np.where(static[1], pos[:, 0], np.where(static[0], pos[:, 1], pos.mean(-1)))
        a = np.where(static[1], ang[:, 0], np.where(static[0], ang[:, 1], ang.mean(-1)))
    return (p <= SP_POS) & (a <= SP_ROT), p, a


def add_values(ach, ref, verts):
    """ADD 프레임 값 (T,W,B). 키트 _object_add 의 식: (R_ach - R_ref) v + (t_ach - t_ref) 의 노름, 꼭짓점 평균."""
    d_rot = qrot(ach[..., 3:]) - qrot(refb(ach, ref)[..., 3:])
    d_pos = ach[..., :3] - refb(ach, ref)[..., :3]
    out = np.empty(ach.shape[:3])
    for b in range(ach.shape[2]):
        off = np.einsum("twij,vj->twvi", d_rot[:, :, b], verts[b]) + d_pos[:, :, b, None, :]
        out[:, :, b] = np.linalg.norm(off, axis=-1).mean(axis=-1)
    return out


def add_auc(ach, ref, verts):
    """키트 _add_auc: 바디마다 (T,W) 전체에서 ADD < t 비율을 9개 임계로 사다리꼴 적분, 바디 평균."""
    add = add_values(ach, ref, verts)
    per_body = [trap9(np.array([np.mean(add[..., b] < t) for t in ADD_THR]))
                for b in range(add.shape[2])]
    return float(np.mean(per_body))


def mp_sr(ach, ref, object_ids):
    """키트 _maniptrans_success: 물체별로 바디 평균 위치 < 0.03 m 이고 회전 < 30도, 모든 물체가 만족해야 성공."""
    pos, ang = pose_errors(ach, ref)
    deg = np.degrees(ang)
    ok = np.ones(pos.shape[0], dtype=bool)
    for oid in np.unique(object_ids):
        m = object_ids == oid
        ok &= (pos[:, m].mean(-1) < MP_POS) & (deg[:, m].mean(-1) < MP_DEG)
    return ok


def mppe_cm(ach, ref):
    """키트 _mean_per_point_error_cm: 물체 원점에서 5 cm 떨어진 6점의 평균 위치 오차 (cm)."""
    rb = refb(ach, ref)
    ka = ach[..., None, :3] + np.einsum("...ij,kj->...ki", qrot(ach[..., 3:]), MPPE_DIRS * MPPE_R)
    kr = rb[..., None, :3] + np.einsum("...ij,kj->...ki", qrot(rb[..., 3:]), MPPE_DIRS * MPPE_R)
    return float(np.linalg.norm(ka - kr, axis=-1).mean() * 100.0)


def rpe_cm(ach, ref):
    """키트 track_3 RPE (_relative_position_error_cm): 바디 쌍마다 첫 바디 좌표계의 상대 위치 오차, 쌍 평균 (cm)."""
    rb = refb(ach, ref)
    bodies = ach.shape[2]
    if bodies < 2:
        return 0.0
    errs = []
    ra, rr = qrot(ach[..., 3:]), qrot(rb[..., 3:])
    for i in range(bodies):
        for j in range(i + 1, bodies):
            da = ach[:, :, j, :3] - ach[:, :, i, :3]
            dr = rb[:, :, j, :3] - rb[:, :, i, :3]
            la = np.einsum("twji,twj->twi", ra[:, :, i], da)
            lr = np.einsum("twji,twj->twi", rr[:, :, i], dr)
            errs.append(np.linalg.norm(la - lr, axis=-1))
    return float(np.mean(np.stack(errs)) * 100.0)


def box_sdf(p, half=HALF):
    """상자의 부호 거리 (물체 좌표계). 내부에서 음수."""
    q = np.abs(p) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


def pen_cm(hands, obj_rot, obj_trans, sdf=box_sdf):
    """키트 PEN: 손 점을 물체 좌표계로 옮긴 뒤 relu(-sdf) 의 평균 (cm).
    키트는 sdf 자리에 -depth(외부에서 0)를 넣고 같은 relu(-x)를 쓴다. 참 부호 거리를 쓰면 내부 점에서 값이 같다."""
    local = np.einsum("tpi,tij->tpj", hands - obj_trans[:, None], obj_rot)
    return float(np.mean(np.maximum(-sdf(local), 0.0)) * 100.0)


# ---------------------------------------------------------------- 합성 궤적

def minjerk(n):
    tau = np.linspace(0.0, 1.0, n)
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5


def ref_traj(n=T):
    """두 바디(양손 모드). 바디 0은 +x로 15 cm, 바디 1은 -x로 15 cm 최소 저크로 이동하며 들었다 내린다. 같은 z축 회전을 한다."""
    s = minjerk(n)
    ref = np.zeros((n, 2, 7))
    starts = np.array([[-0.10, 0.0, 0.10], [0.10, 0.0, 0.10]])
    dx = np.array([0.15, -0.15])
    yaw = np.array([1.0, -0.6])
    for b in range(2):
        ref[:, b, 0] = starts[b, 0] + dx[b] * s
        ref[:, b, 1] = starts[b, 1]
        ref[:, b, 2] = starts[b, 2] + 0.05 * np.sin(np.pi * s)
        ref[:, b, 5] = np.sin(yaw[b] * s / 2)
        ref[:, b, 6] = np.cos(yaw[b] * s / 2)
    return ref


def box_surface(n, rng):
    axis = rng.integers(0, 3, n)
    sign = rng.choice([-1.0, 1.0], n)
    pts = rng.uniform(-1.0, 1.0, (n, 3)) * HALF
    pts[np.arange(n), axis] = sign * HALF[axis]
    return pts


def compose(ref, w, off=None, drift=None, trans_amp=None, trans_tau=None,
            quat_delta=None, jitter=0.0, rng=None):
    """ach (T,W,B,7). off/drift/trans_amp (W,B,3), quat_delta (W,B,4, 세계 좌표계 회전), jitter: 위치 가우시안 σ (m)."""
    n, bodies, _ = ref.shape
    ach = np.broadcast_to(ref[:, None], (n, w, bodies, 7)).copy()
    pos = ach[..., :3]
    if off is not None:
        pos += off[None]
    if drift is not None:
        s = np.linspace(0.0, 1.0, n)
        pos += s[:, None, None, None] * drift[None]
    if trans_amp is not None:
        decay = np.exp(-np.arange(n) / trans_tau)[:, None, None, None]
        pos += decay * trans_amp[None]
    if jitter > 0.0:
        pos += rng.normal(0.0, jitter, size=pos.shape)
    if quat_delta is not None:
        ach[..., 3:] = qmul(quat_delta[None], ref[:, None, :, 3:])
    ach[..., 3:] = qnorm(ach[..., 3:])
    return ach


def rand_vec(rng, shape, max_norm):
    d = rng.normal(size=shape + (3,))
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d * rng.uniform(0.0, max_norm, shape)[..., None]


def rand_quat(rng, shape, max_angle):
    axis = rand_vec(rng, shape, 1.0)
    axis /= np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-12)
    theta = rng.uniform(0.0, max_angle, shape)
    return np.concatenate([axis * np.sin(theta / 2)[..., None], np.cos(theta / 2)[..., None]], axis=-1)


# ---------------------------------------------------------------- 1. 알려진 답 테스트

def check(name, got, expect, tol=1e-9):
    ok = bool(abs(float(got) - float(expect)) <= tol)
    RESULTS["unit_tests"].append({"name": name, "got": float(got), "expected": float(expect), "tol": tol, "pass": ok})
    return ok


def check_bool(name, got, expect):
    ok = bool(got) == bool(expect)
    RESULTS["unit_tests"].append({"name": name, "got": bool(got), "expected": bool(expect), "pass": ok})
    return ok


def unit_tests():
    ref = ref_traj()
    verts = [box_surface(N_VERT, np.random.default_rng(SEED + 1)) for _ in range(2)]
    ids = np.array([0, 1])
    ring = np.stack([0.05 * np.cos(np.linspace(0, 2 * np.pi, 36, endpoint=False)),
                     0.05 * np.sin(np.linspace(0, 2 * np.pi, 36, endpoint=False)),
                     np.zeros(36)], axis=-1)
    ring_verts = [ring, ring]
    oks = []

    # (1) 항등
    ach = compose(ref, 1)
    oks.append(check_bool("identity: SP-SR raw (b) = 1", sp_raw(ach, ref)[0][0], True))
    oks.append(check_bool("identity: SP-SR centered (a) = 1", sp_centered(ach, ref)[0][0], True))
    oks.append(check_bool("identity: SP-SR warp (c) = 1", sp_warp(ach, ref)[0][0], True))
    oks.append(check_bool("identity: MP-SR = 1", mp_sr(ach, ref, ids)[0], True))
    oks.append(check("identity: ADD-AUC = 1", add_auc(ach, ref, verts), 1.0))
    oks.append(check("identity: MPPE = 0 cm", mppe_cm(ach, ref), 0.0))
    oks.append(check("identity: RPE = 0 cm", rpe_cm(ach, ref), 0.0))
    far = np.full((5, 3), 0.5)
    rot_id = np.broadcast_to(np.eye(3), (5, 3, 3))
    oks.append(check("identity: PEN, hand points outside object = 0 cm",
                     pen_cm(far[:, None], rot_id, np.zeros((5, 3))), 0.0))
    oks.append(check("warp of identity returns reference positions",
                     np.abs(warp_start(compose(ref, 1), ref)[..., :3] - compose(ref, 1)[..., :3]).max(), 0.0))

    # (2) 상수 translation 오프셋 0.035 m (모든 바디 같은 방향)
    off = np.zeros((1, 2, 3))
    off[..., 0] = 0.035
    ach = compose(ref, 1, off=off)
    oks.append(check("const offset 0.035 m: SP-SR raw pos = 0.035", sp_raw(ach, ref)[1][0], 0.035))
    oks.append(check("const offset 0.035 m: SP-SR centered pos = 0", sp_centered(ach, ref)[1][0], 0.0))
    oks.append(check("const offset 0.035 m: SP-SR warp pos = 0", sp_warp(ach, ref)[1][0], 0.0))
    oks.append(check("const offset 0.035 m: ADD-AUC = 0.6875 (thresholds 1-3 cm fail, 4-9 cm pass)",
                     add_auc(ach, ref, verts), 0.6875))
    oks.append(check("const offset 0.035 m: MPPE = 3.5 cm", mppe_cm(ach, ref), 3.5))
    oks.append(check("const offset 0.035 m: RPE = 0 (both bodies move together)", rpe_cm(ach, ref), 0.0))
    oks.append(check_bool("const offset 0.035 m: MP-SR fails (> 0.03 m)", mp_sr(ach, ref, ids)[0], False))
    off2 = off * (0.02 / 0.035)
    oks.append(check_bool("const offset 0.02 m: MP-SR passes", mp_sr(compose(ref, 1, off=off2), ref, ids)[0], True))

    # (3) 상수 회전 오프셋 (세계 z축), 위치 변화 없음
    th = 0.3
    ach = compose(ref, 1, quat_delta=np.broadcast_to(quat_z(th), (1, 2, 4)).copy())
    _, pos_r, ang_r = sp_raw(ach, ref)
    oks.append(check("rot offset 0.3 rad: SP-SR raw rot = 0.3 (geodesic)", ang_r[0], th))
    oks.append(check("rot offset 0.3 rad: SP-SR raw pos = 0", pos_r[0], 0.0))
    add_ring = add_values(compose(ref, 1, quat_delta=np.broadcast_to(quat_z(th), (1, 2, 4)).copy()),
                          ref, ring_verts)
    oks.append(check("rot offset 0.3 rad: ADD on ring r=5 cm = 2 r sin(theta/2)",
                     add_ring.mean(), 2 * 0.05 * np.sin(th / 2)))
    oks.append(check("rot offset 0.3 rad: MPPE = 100 * 4*2r*sin(theta/2)/6 cm",
                     mppe_cm(ach, ref), 100 * (4 * 2 * 0.05 * np.sin(th / 2)) / 6))
    th2 = 0.6
    ach6 = compose(ref, 1, quat_delta=np.broadcast_to(quat_z(th2), (1, 2, 4)).copy())
    oks.append(check_bool("rot offset 0.6 rad: SP-SR raw fails (rot > 0.5)", sp_raw(ach6, ref)[0][0], False))
    oks.append(check_bool("rot offset 0.6 rad: SP-SR centered fails (rot not centered)", sp_centered(ach6, ref)[0][0], False))
    wa = warp_start(ach6, ref)
    # 허용오차 1e-6: 측지각 식의 sqrt(1 - d^2) 가 d ~ 1 근처에서 1e-8 수준 잡음을 만든다
    oks.append(check("rot offset 0.6 rad: warp removes the orientation error", sp_raw(wa, ref)[2][0], 0.0, tol=1e-6))
    oks.append(check_bool("rot offset 0.6 rad: warp creates a position error (rigid-warp artifact)",
                          sp_raw(wa, ref)[1][0] > 0.0, True))

    # (4) 선형 드리프트 (총 d, 시간에 선형). 시간 평균 = d/2
    for d, expect_raw, expect_cent in [(0.10, 0.05, None), (0.25, 0.125, 0.0625)]:
        dr = np.zeros((1, 2, 3))
        dr[..., 0] = d
        ach = compose(ref, 1, drift=dr)
        oks.append(check(f"drift {d} m: SP-SR raw pos = d/2", sp_raw(ach, ref)[1][0], expect_raw))
        if expect_cent is not None:
            oks.append(check(f"drift {d} m: SP-SR centered pos = d * mean|s-1/2| (about 0.25 d)",
                             sp_centered(ach, ref)[1][0], expect_cent, tol=2e-3))
    oks.append(check_bool("drift 0.25 m: centered (a) passes, raw (b) fails",
                          sp_centered(compose(ref, 1, drift=np.broadcast_to(np.array([0.25, 0, 0]), (1, 2, 3)).copy()), ref)[0][0]
                          and not sp_raw(compose(ref, 1, drift=np.broadcast_to(np.array([0.25, 0, 0]), (1, 2, 3)).copy()), ref)[0][0],
                          True))

    # (5) 과도 오프셋 o exp(-t/tau): 원시 평균은 작고, 첫 프레임 정렬은 과잉 보정으로 크게 남는다
    tau = T / 10
    tr = np.zeros((1, 2, 3))
    tr[..., 0] = 0.15
    ach = compose(ref, 1, trans_amp=tr, trans_tau=tau)
    s_decay = np.exp(-np.arange(T) / tau)
    oks.append(check("transient 0.15 m: SP-SR raw pos = 0.15 * mean(exp(-t/tau))",
                     sp_raw(ach, ref)[1][0], 0.15 * s_decay.mean()))
    oks.append(check("transient 0.15 m: SP-SR warp pos = 0.15 * mean(1 - exp(-t/tau))  (over-correction)",
                     sp_warp(ach, ref)[1][0], 0.15 * (1 - s_decay).mean()))
    oks.append(check_bool("transient 0.15 m: warp (c) fails while raw (b) passes",
                          (not sp_warp(ach, ref)[0][0]) and sp_raw(ach, ref)[0][0], True))

    # (6) 떨림 σ: ADD 프레임 값 = ||n_t||, 3차원 가우시안이므로 chi(3) 분포. SP-SR 원시 평균 = 1.596 σ
    sig = 0.01
    ach = compose(ref, 20, jitter=sig, rng=np.random.default_rng(SEED + 2))
    add = add_values(ach, ref, verts)
    frac_emp = np.array([np.mean(add < t) for t in ADD_THR])
    frac_ana = stats.chi(df=3).cdf(ADD_THR / sig)
    oks.append(check_bool("jitter 1 cm: ADD fractions match chi(3) CDF (max abs diff < 0.02)",
                          np.max(np.abs(frac_emp - frac_ana)) < 0.02, True))
    oks.append(check_bool("jitter 1 cm: SP-SR raw pos mean = sigma*sqrt(8/pi) (rel. err < 2%)",
                          abs(sp_raw(ach, ref)[1].mean() / (sig * np.sqrt(8 / np.pi)) - 1.0) < 0.02, True))

    # (7) PEN: 상자 안쪽 2 mm 점 1개와 바깥 점 1개 -> 평균 1 mm -> 0.1 cm. 회전·이동 후에도 같아야 함.
    hands = np.array([[0.028, 0.0, 0.0], [0.05, 0.0, 0.0]])
    rot0 = np.eye(3)[None]
    trans0 = np.zeros((1, 3))
    oks.append(check("PEN: one point 2 mm inside a face, one outside -> 0.1 cm",
                     pen_cm(hands[None], rot0, trans0), 0.1))
    center = np.array([[0.0, 0.0, 0.0]])
    oks.append(check("PEN: centre point of 6x6x10 cm box (depth 3 cm) -> 3.0 cm",
                     pen_cm(center[None], rot0, trans0), 3.0))
    rz90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    world = (hands @ rz90.T) + np.array([0.2, 0.0, 0.0])
    oks.append(check("PEN: rotated 90 deg and translated object gives the same value",
                     pen_cm(world[None], rz90[None], np.array([[0.2, 0.0, 0.0]])), 0.1))

    # (8) RPE: 바디 1만 x 방향으로 3 cm 이동 -> 상대 위치 오차 3 cm
    rb = ref.copy()
    ach_rpe = compose(rb, 1)
    ach_rpe[:, 0, 1, 0] += 0.03
    oks.append(check("RPE: one body shifted 3 cm -> 3.0 cm", rpe_cm(ach_rpe, ref), 3.0, tol=1e-9))

    # (9) 각도 기하: 회전 오차 각도는 기하 각 그대로 (0.3 rad, 20도 vs 40도)
    oks.append(check_bool("MP-SR: rot 20 deg passes", mp_sr(compose(ref, 1, quat_delta=np.broadcast_to(quat_z(np.radians(20)), (1, 2, 4)).copy()), ref, ids)[0], True))
    oks.append(check_bool("MP-SR: rot 40 deg fails", mp_sr(compose(ref, 1, quat_delta=np.broadcast_to(quat_z(np.radians(40)), (1, 2, 4)).copy()), ref, ids)[0], False))

    passed = sum(oks)
    RESULTS["unit_tests_summary"] = {"passed": int(passed), "total": len(oks)}
    return passed == len(oks)


# ---------------------------------------------------------------- 2. SP-SR 정의 비교

PERTURBATIONS = {
    "초기 과도 오프셋 0.15 m (tau 1 s)": dict(trans_amp=np.array([0.15, 0.0, 0.0]), trans_tau=T / 10),
    "지속 오프셋 0.15 m": dict(off=np.array([0.15, 0.0, 0.0])),
    "선형 드리프트 0.25 m": dict(drift=np.array([0.25, 0.0, 0.0])),
    "상수 회전 오프셋 0.6 rad (z축)": dict(quat_angle=0.6),
    "떨림 σ 0.08 m (위치)": dict(jitter=0.08),
}


def perturb(ref, spec, w, rng):
    kw = {}
    B = ref.shape[1]
    if "trans_amp" in spec:
        kw["trans_amp"] = np.broadcast_to(spec["trans_amp"], (w, B, 3)).copy()
        kw["trans_tau"] = spec["trans_tau"]
    if "off" in spec:
        kw["off"] = np.broadcast_to(spec["off"], (w, B, 3)).copy()
    if "drift" in spec:
        kw["drift"] = np.broadcast_to(spec["drift"], (w, B, 3)).copy()
    if "quat_angle" in spec:
        kw["quat_delta"] = np.broadcast_to(quat_z(spec["quat_angle"]), (w, B, 4)).copy()
    if "jitter" in spec:
        kw["jitter"] = spec["jitter"]
        kw["rng"] = rng
    return compose(ref, w, **kw)


def variant_table(ach, ref):
    row = {}
    for key, fn in [("a_centered", lambda: sp_centered(ach, ref)),
                    ("b_raw_chord", lambda: sp_raw(ach, ref)),
                    ("c_kit_warp_track3", lambda: sp_warp(ach, ref))]:
        ok, p, a = fn()
        row[key] = {"pos_m": float(p[0]), "rot_rad": float(a[0]), "pass": bool(ok[0])}
    ok, p, a = sp_centered(ach, ref, lag=True)
    row["diag_a_with_1step_ref_lag"] = {"pos_m": float(p[0]), "rot_rad": float(a[0]), "pass": bool(ok[0])}
    return row


def comparison_perturbations():
    ref = ref_traj()
    out = {}
    for name, spec in PERTURBATIONS.items():
        ach = perturb(ref, spec, 1, np.random.default_rng(SEED + 3))
        out[name] = variant_table(ach, ref)
    # 기준: 변형 없음
    out["항등 (기준)"] = variant_table(compose(ref, 1), ref)
    RESULTS["perturbations"] = out


def population_sr(name, n, off_max, drift_max, rot_max, jitter, seed):
    ref = ref_traj()
    rng = np.random.default_rng(seed)
    off = rand_vec(rng, (n, 2), off_max) if off_max > 0 else None
    drift = rand_vec(rng, (n, 2), drift_max) if drift_max > 0 else None
    qd = rand_quat(rng, (n, 2), rot_max) if rot_max > 0 else None
    ach = compose(ref, n, off=off, drift=drift, quat_delta=qd, jitter=jitter, rng=rng)
    ok_a, _, _ = sp_centered(ach, ref)
    ok_b, _, _ = sp_raw(ach, ref)
    ok_c, _, _ = sp_warp(ach, ref)
    return {
        "name": name,
        "episodes": n,
        "assumptions": {
            "constant_offset_norm_uniform_m": [0.0, off_max],
            "drift_total_norm_uniform_m": [0.0, drift_max],
            "rotation_offset_angle_uniform_rad": [0.0, rot_max],
            "jitter_sigma_m": jitter,
            "direction": "uniform on sphere, independent per body and per episode",
            "reference": "two bodies, min-jerk 15 cm motion, 10 s, 30 fps",
        },
        "SR_a_centered": float(ok_a.mean()),
        "SR_b_raw_chord_or_kit_track2": float(ok_b.mean()),
        "SR_c_kit_warp_track3": float(ok_c.mean()),
        "disagree_a_pass_b_fail": int(np.sum(ok_a & ~ok_b)),
        "disagree_b_pass_a_fail": int(np.sum(ok_b & ~ok_a)),
        "disagree_c_pass_b_fail": int(np.sum(ok_c & ~ok_b)),
        "disagree_b_pass_c_fail": int(np.sum(ok_b & ~ok_c)),
    }


def populations():
    specs = [
        ("완만 (오프셋 <= 5 cm, 드리프트 <= 5 cm)", 0.05, 0.05, 0.0, 0.002),
        ("넓음 (오프셋 <= 15 cm, 드리프트 <= 25 cm)", 0.15, 0.25, 0.0, 0.002),
        ("넓음 + 회전 오프셋 (<= 0.6 rad)", 0.15, 0.25, 0.6, 0.002),
    ]
    RESULTS["populations"] = [population_sr(nm, 500, o, d, r, j, SEED + 10 + i)
                              for i, (nm, o, d, r, j) in enumerate(specs)]


# ---------------------------------------------------------------- 3. 떨림 민감도

JITTER_GRID_MM = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100]


def jitter_curves():
    ref = ref_traj()
    verts = [box_surface(N_VERT, np.random.default_rng(SEED + 1)) for _ in range(2)]
    rows = []
    for i, mm in enumerate(JITTER_GRID_MM):
        sig = mm / 1000.0
        if sig == 0.0:
            ach = compose(ref, 40)
        else:
            ach = compose(ref, 40, jitter=sig, rng=np.random.default_rng(SEED + 100 + i))
        auc = add_auc(ach, ref, verts)
        auc_ana = trap9(stats.chi(df=3).cdf(ADD_THR / sig)) if sig > 0 else 1.0
        ok_b, p_b, _ = sp_raw(ach, ref)
        ok_a, p_a, _ = sp_centered(ach, ref)
        ok_c, p_c, _ = sp_warp(ach, ref)
        rows.append({
            "sigma_mm": mm,
            "ADD_AUC": auc,
            "ADD_AUC_analytic_chi3": float(auc_ana),
            "SP_SR_b_raw": float(ok_b.mean()),
            "SP_SR_a_centered": float(ok_a.mean()),
            "SP_SR_c_warp_track3": float(ok_c.mean()),
            "SP_pos_b_mean_m": float(p_b.mean()),
            "SP_pos_b_analytic_m": float(sig * np.sqrt(8 / np.pi)),
        })
    RESULTS["jitter"] = {
        "grid_mm": JITTER_GRID_MM,
        "episodes_per_sigma": 40,
        "note": "떨림은 위치에만 더한다. 회전 떨림은 넣지 않았다. ADD 해석 값은 ||n|| 의 chi(3) 분포, SP-SR 해석 값은 평균 노름 sigma*sqrt(8/pi).",
        "rows": rows,
    }


def figure():
    font_candidates = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                       "/System/Library/Fonts/AppleSDGothicNeo.ttc"]
    font_path = next((p for p in font_candidates if Path(p).exists()), None)
    if font_path:
        font_manager.fontManager.addfont(font_path)
        fam = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = [fam, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 9

    rows = RESULTS["jitter"]["rows"]
    x = np.array([r["sigma_mm"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    ax = axes[0]
    xs = np.linspace(0.5, 100, 200)
    ana = [trap9(stats.chi(df=3).cdf(ADD_THR / (s / 1000.0))) if s > 0 else 1.0 for s in xs]
    ax.plot(xs, ana, color="#8d8d8d", lw=1.2, label="해석 값 (chi(3))")
    ax.plot(x, [r["ADD_AUC"] for r in rows], "o-", color="#1f5fae", ms=4, label="키트 포트")
    ax.set_xlabel("떨림 σ (mm, 위치)")
    ax.set_ylabel("ADD-AUC (1–9 cm)")
    ax.set_ylim(0, 1.02)
    ax.set_title("AUC: 떨림에 따라 단조 감소")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(x, [r["SP_SR_b_raw"] for r in rows], "o-", color="#c95a24", ms=4, label="(b) 원시 (CHORD, 키트 트랙 2)")
    ax.plot(x, [r["SP_SR_a_centered"] for r in rows], "s--", color="#2a9d8f", ms=4, label="(a) 중심화 (SPIDER 양손)")
    ax.plot(x, [r["SP_SR_c_warp_track3"] for r in rows], "^:", color="#6a4c93", ms=4, label="(c) 첫 프레임 정렬 (키트 트랙 3)")
    ax.axvline(1000 * 0.10 / np.sqrt(8 / np.pi), color="#8d8d8d", lw=1, ls="-.")
    ax.set_xlabel("떨림 σ (mm, 위치)")
    ax.set_ylabel("SP-SR 성공 비율")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("SP-SR: (a), (b)는 63 mm 부근, (c)는 그보다 일찍 떨어짐")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=160)
    plt.close(fig)


def main():
    t0 = time.perf_counter()
    c0 = time.process_time()
    RESULTS["config"] = {
        "seed": SEED, "T_frames": T, "fps": FPS, "half_extents_m": HALF.tolist(),
        "add_thresholds_m": ADD_THR.round(3).tolist(), "sp_thresholds": [SP_POS, SP_ROT],
        "mp_thresholds": [MP_POS, MP_DEG], "n_vertices": N_VERT, "static_drift_m": STATIC_DRIFT,
        "kit_warp_flag": {"track_2": False, "track_3": True},
    }
    ok = unit_tests()
    comparison_perturbations()
    populations()
    jitter_curves()
    figure()
    RESULTS["timing"] = {"cpu_s": round(time.process_time() - c0, 2), "wall_s": round(time.perf_counter() - t0, 2)}
    OUT_JSON.write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2))
    s = RESULTS["unit_tests_summary"]
    print(f"unit tests {s['passed']}/{s['total']}  cpu {RESULTS['timing']['cpu_s']} s")
    if not ok:
        raise SystemExit("unit test failure")


if __name__ == "__main__":
    main()
