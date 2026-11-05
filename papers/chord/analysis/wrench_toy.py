"""CHORD 접촉 wrench 보상 toy (CPU, numpy, 단일 스레드). 실행: python3 analysis/wrench_toy.py

공식 코드(video_to_data @ 9cad120)의 식을 numpy로 옮겨 아래를 확인합니다.
- W1: 상자 뚜껑 예시(Sec. 3.1 오른쪽 그림). 같은 위치에서 법선만 돌릴 때 위치 보상·Eq. 3·코드 보상이 어떻게 변하는지
- W2: 법선은 맞고 위치만 밀릴 때 (표면을 따라 0–5 cm)
- W3: 힘 크기 불변성. 코드가 접촉력 방향만 쓰므로 크기를 0.1–10배로 바꿔도 지지값이 같은지
- W4: 마찰계수 mu(코드 0.1, 논문 미기재)에 따라 보상이 법선 오차에 얼마나 민감한지
- W5: Eq. 3(512방향 합의 exp)과 코드(방향별 exp의 평균)의 차이
- W6: 접촉 개수와 축소 force-closure 보상 r_fc, Ferrari–Canny epsilon
- W7: 기저 방향 512개 무작위 표본에 따른 보상 흔들림
- W8: 같은 body 안의 추가 접촉 (사람 지지값이 0인 방향은 코드가 벌하지 않음)
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MU, K_EDGES, NB, TOL, VAR = 0.1, 8, 512, 0.1, 0.1  # objective/tracking.yaml, commands_cfg.py
ACTIVE = 1e-3                                       # *_cmd_active / *_cur_active 임계값
HALF = np.array([0.10, 0.075, 0.05])                # 상자 반치수 (m)
RC = float(np.linalg.norm(HALF))                    # 경계 구 반지름 = object_mesh_radius 역할


def basis(nb=NB, seed=0):
    """sample_wrench_space_basis_scaled(rc=1.0): 6D 가우시안을 정규화 = 단위 구 균일."""
    b = np.random.default_rng(seed).standard_normal((nb, 6))
    return b / np.linalg.norm(b, axis=1, keepdims=True)


def tangent(n):
    """compute_tangent_basis (Frisvad 2012)."""
    nx, ny, nz = n
    s = 1.0 if nz >= 0 else -1.0
    a = -1.0 / (s + nz)
    b = nx * ny * a
    t1 = np.array([1 + s * nx * nx * a, s * b, -s * nx])
    t2 = np.array([b, s + ny * ny * a, -ny])
    return t1 / np.linalg.norm(t1), t2 / np.linalg.norm(t2)


def wrench_matrix(points, normals, mu=MU, k=K_EDGES, rc=RC):
    """compute_wrench_space: 단위 마찰원뿔 모서리 k개 + 법선 1개, 토크는 p x f / rc. (6, c*(k+1))."""
    th = np.linspace(0, 2 * np.pi, k + 1)[:-1]
    cols = []
    for p, n in zip(points, normals):
        n = n / np.linalg.norm(n)
        t1, t2 = tangent(n)
        e = n[None] + mu * (np.cos(th)[:, None] * t1 + np.sin(th)[:, None] * t2)
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        e = np.vstack([e, n])
        cols.append(np.hstack([e, np.cross(p, e) / rc]))
    return np.vstack(cols).T if cols else np.zeros((6, 0))


def support(W, B):
    """compute_wrench_space_support_function: max_col(B W), 0 이하는 0."""
    if W.shape[1] == 0:
        return np.zeros(len(B))
    return np.clip((B @ W).max(axis=1), 0.0, None)


def code_reward(sh, sr, tol=TOL, var=VAR):
    """contact_wrench_support_reward_jit의 한 손·한 body 값: 사람 활성 방향 평균, 양쪽 활성 칸만 exp."""
    cmd, cur = sh > ACTIVE, sr > ACTIVE
    loss = np.clip((1 - tol) * sh - sr, 0, None) ** 2 + np.clip(sr - (1 + tol) * sh, 0, None) ** 2
    return float(((cmd & cur) * np.exp(-loss / var)).sum() / max(cmd.sum(), 1e-6))


def eq3_reward(sh, sr, tol=TOL, var=VAR):
    """논문 Eq. 3: exp(-(||max(0,(1-b)sh-sr)||^2 + ||max(0,sr-(1+b)sh)||^2) / v)."""
    lo = np.clip((1 - tol) * sh - sr, 0, None)
    hi = np.clip(sr - (1 + tol) * sh, 0, None)
    return float(np.exp(-(lo @ lo + hi @ hi) / var))


def missed(sh, sr):
    cmd, cur = sh > ACTIVE, sr > ACTIVE
    return float((cmd & ~cur).sum() / max(cmd.sum(), 1e-6))


def loc_reward(ph, pr, sigma=0.02):
    """위치 기반 접촉 보상의 대표형(DexMachina류): exp(-||dp||^2 / sigma^2), sigma = 2 cm (가정)."""
    d = np.linalg.norm(np.asarray(ph) - np.asarray(pr), axis=-1)
    return float(np.mean(np.exp(-(d ** 2) / sigma ** 2)))


def rot(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


B = basis()
out = {"config": {"mu": MU, "edges": K_EDGES, "num_basis": NB, "tolerance": TOL, "var": VAR,
                  "box_half_extent_m": HALF.tolist(), "rc_m": round(RC, 4), "location_sigma_m": 0.02,
                  "cone_half_angle_deg": round(math.degrees(math.atan(MU)), 2)}}

# ---------- W1: 상자 뚜껑 — 같은 위치, 법선 회전 ----------
# 사람 엄지는 앞면 위쪽 모서리 아래(뚜껑 턱)에서 위로 민다: 법선(물체 안쪽) +z.
# 로봇은 같은 점에서 법선을 -x(앞면을 안쪽으로 미는 방향) 쪽으로 theta만큼 돌린다.
p_h = np.array([HALF[0], 0.0, HALF[2] - 0.01])
n_h = np.array([0.0, 0.0, 1.0])
sh1 = support(wrench_matrix([p_h], [n_h]), B)
rows = []
for deg in [0, 5, 10, 15, 20, 30, 45, 60, 75, 90]:
    n_r = rot([0, 1, 0], -math.radians(deg)) @ n_h  # +z → -x 방향으로 회전
    sr = support(wrench_matrix([p_h], [n_r]), B)
    rows.append({"normal_err_deg": deg, "location": round(loc_reward(p_h, p_h), 4),
                 "code": round(code_reward(sh1, sr), 4), "eq3": round(eq3_reward(sh1, sr), 4),
                 "missed_frac": round(missed(sh1, sr), 4)})
out["W1_normal_rotation_same_location"] = rows
out["W1_note"] = "90도에서 n_r = -x (앞면을 안쪽으로 밂). 위치 보상은 1.0 그대로, wrench 보상만 법선 차이를 잡는다"

# ---------- W2: 법선은 같고 위치만 이동 (윗면 위를 x 방향으로) ----------
p0 = np.array([0.0, 0.0, -HALF[2]])  # 아랫면 중앙, 위로 받침 (법선 +z)
sh2 = support(wrench_matrix([p0], [n_h]), B)
rows = []
for cm in [0, 0.5, 1, 2, 3, 5]:
    pr = p0 + np.array([cm / 100, 0, 0])
    sr = support(wrench_matrix([pr], [n_h]), B)
    rows.append({"offset_cm": cm, "location": round(loc_reward(p0, pr), 4),
                 "code": round(code_reward(sh2, sr), 4), "eq3": round(eq3_reward(sh2, sr), 4)})
out["W2_location_offset_same_normal"] = rows
out["W2_note"] = f"토크는 p x f / rc(rc = {RC:.3f} m)라서 1 cm 이동은 토크 성분을 약 {0.01 / RC:.3f}만 바꾼다"

# ---------- W3: 힘 크기 불변성 ----------
f_dir = np.array([0.03, -0.02, 1.0])
sr_ref = support(wrench_matrix([p0], [f_dir]), B)
rows = []
for scale in [0.1, 1.0, 10.0]:
    sr = support(wrench_matrix([p0], [scale * f_dir]), B)  # wrench_preprocess_jit: 힘 → 정규화된 방향
    rows.append({"force_scale": scale, "max_abs_support_diff": float(np.abs(sr - sr_ref).max())})
out["W3_force_magnitude_invariance"] = rows

# ---------- W4: 마찰계수 민감도 ----------
rows = []
for mu in [0.1, 0.3, 0.5, 1.0]:
    shm = support(wrench_matrix([p_h], [n_h], mu=mu), B)
    r = {}
    for deg in [5, 10, 20, 30]:
        n_r = rot([0, 1, 0], -math.radians(deg)) @ n_h
        r[f"code@{deg}deg"] = round(code_reward(shm, support(wrench_matrix([p_h], [n_r], mu=mu), B)), 4)
    rows.append({"mu": mu, "cone_half_angle_deg": round(math.degrees(math.atan(mu)), 1), **r})
out["W4_friction_sensitivity"] = rows

# ---------- W5: Eq. 3 vs 코드, 오차가 여러 방향에 퍼질 때 ----------
# 두 손가락 pinch (x = ±0.1 양쪽 면), 로봇은 둘 다 같은 각도로 법선 오차
ph2 = [np.array([HALF[0], 0, 0]), np.array([-HALF[0], 0, 0])]
nh2 = [np.array([-1.0, 0, 0]), np.array([1.0, 0, 0])]
shp = support(wrench_matrix(ph2, nh2), B)
rows = []
for deg in [0, 2, 5, 10, 15, 20, 30]:
    nr2 = [rot([0, 0, 1], math.radians(deg)) @ nh2[0], rot([0, 0, 1], -math.radians(deg)) @ nh2[1]]
    sr = support(wrench_matrix(ph2, nr2), B)
    lo = np.clip((1 - TOL) * shp - sr, 0, None)
    hi = np.clip(sr - (1 + TOL) * shp, 0, None)
    rows.append({"normal_err_deg": deg, "code": round(code_reward(shp, sr), 4),
                 "eq3": round(eq3_reward(shp, sr), 6), "sum_sq_violation": round(float(lo @ lo + hi @ hi), 4),
                 "frac_dirs_violating": round(float(((lo > 0) | (hi > 0)).mean()), 4)})
out["W5_eq3_vs_code_pinch"] = rows

# ---------- W6: 접촉 개수와 r_fc, Ferrari–Canny epsilon ----------
faces = {"+x": ([HALF[0], 0, 0], [-1, 0, 0]), "-x": ([-HALF[0], 0, 0], [1, 0, 0]),
         "+y": ([0, HALF[1], 0], [0, -1, 0]), "-y": ([0, -HALF[1], 0], [0, 1, 0]),
         "+z": ([0, 0, HALF[2]], [0, 0, -1]), "-z": ([0, 0, -HALF[2]], [0, 0, 1])}
configs = {
    "1 contact (-z)": ["-z"],
    "2 antipodal (x)": ["+x", "-x"],
    "3 (x pair + -z)": ["+x", "-x", "-z"],
    "4 (x pair + z pair)": ["+x", "-x", "+z", "-z"],
    "6 face centers": list(faces),
}
offs = {"+x": [0, 0.04, 0.03], "-x": [0, -0.04, -0.03], "+z": [0.05, 0.03, 0], "-z": [-0.05, -0.03, 0],
        "+y": [0.06, 0, 0.02], "-y": [-0.06, 0, -0.02]}
rows = []
for name, keys in configs.items():
    for mu in [0.1, 0.5]:
        for shifted in [False, True]:
            pts = [np.array(faces[k][0], float) + (np.array(offs[k]) if shifted else 0) for k in keys]
            nrm = [np.array(faces[k][1], float) for k in keys]
            s = support(wrench_matrix(pts, nrm, mu=mu), B)
            rows.append({"contacts": name, "mu": mu, "face_centers": not shifted,
                         "r_fc(eps=0.01)": round(float((s > 0.01).mean()), 4),
                         "ferrari_canny_min_support": round(float(s.min()), 4)})
out["W6_force_closure"] = rows
out["W6_note"] = ("면 중앙 접촉은 p x n = 0이라 순수 토크 방향을 만들지 못한다. 위치를 면 위에서 옮기면(shifted) 토크가 생긴다. "
                  "min support > 0이면 이 기저에서 force closure")

# ---------- W7: 기저 표본에 따른 흔들림 ----------
vals = {"code_W1_30deg": [], "code_W5_10deg": []}
n30 = rot([0, 1, 0], -math.radians(30)) @ n_h
nr10 = [rot([0, 0, 1], math.radians(10)) @ nh2[0], rot([0, 0, 1], -math.radians(10)) @ nh2[1]]
for seed in range(20):
    Bs = basis(seed=seed)
    vals["code_W1_30deg"].append(code_reward(support(wrench_matrix([p_h], [n_h]), Bs),
                                             support(wrench_matrix([p_h], [n30]), Bs)))
    vals["code_W5_10deg"].append(code_reward(support(wrench_matrix(ph2, nh2), Bs),
                                             support(wrench_matrix(ph2, nr10), Bs)))
out["W7_basis_seed_spread"] = {k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4),
                                   "min": round(float(np.min(v)), 4), "max": round(float(np.max(v)), 4)}
                               for k, v in vals.items()}

# ---------- W8: 같은 body 위의 추가 접촉 ----------
sh_single = support(wrench_matrix([p0], [n_h]), B)
extra_pts = [p0, np.array([HALF[0], 0, 0])]
extra_nrm = [n_h, np.array([-1.0, 0, 0])]
sr_extra = support(wrench_matrix(extra_pts, extra_nrm), B)
cmd = sh_single > ACTIVE
out["W8_extra_contact_same_body"] = {
    "code_reward_single": round(code_reward(sh_single, support(wrench_matrix([p0], [n_h]), B)), 4),
    "code_reward_with_extra_side_contact": round(code_reward(sh_single, sr_extra), 4),
    "eq3_with_extra": round(eq3_reward(sh_single, sr_extra), 6),
    "dirs_human_zero_robot_positive": int(((~cmd) & (sr_extra > ACTIVE)).sum()),
    "frac_human_active_dirs": round(float(cmd.mean()), 4),
}
out["W8_note"] = ("사람 지지값이 0인 방향에서 로봇 지지값이 생겨도 코드 보상은 그 방향을 세지 않는다. "
                  "unintended 벌칙은 사람 접촉이 없는 body에만 걸린다. 사람 활성 방향에서 지지값이 (1+tol)배를 넘으면 too_large로 깎인다")

# ---------- W9: 512개 표본 방향의 min support는 Ferrari–Canny epsilon의 상한 ----------
# 마찰 있는 점접촉 두 개(antipodal)는 두 점을 잇는 축(x) 둘레 토크를 만들지 못한다(p x f의 x성분 = 0).
W_anti = wrench_matrix(ph2, nh2)
e_tx = np.array([0, 0, 0, 1.0, 0, 0])
cosang = np.clip(B @ e_tx, -1, 1)
test = basis(nb=20000, seed=123)
cover = np.degrees(np.arccos(np.clip((test @ B.T).max(axis=1), -1, 1)))
out["W9_sampled_epsilon_vs_true"] = {
    "antipodal_tau_x_component_max_abs": float(np.abs(W_anti[3]).max()),
    "support_at_exact_tau_x": float(support(W_anti, e_tx[None])[0]),
    "support_at_exact_minus_tau_x": float(support(W_anti, -e_tx[None])[0]),
    "sampled_min_support_512": round(float(support(W_anti, B).min()), 4),
    "nearest_basis_angle_to_tau_x_deg": round(float(np.degrees(np.arccos(np.abs(cosang).max()))), 1),
    "basis_covering_angle_deg": {"median": round(float(np.median(cover)), 1),
                                 "p95": round(float(np.percentile(cover, 95)), 1),
                                 "max_of_20000": round(float(cover.max()), 1)},
}
out["W9_note"] = ("두 점접촉은 x축 토크 방향의 지지값이 정확히 0(force closure 아님)인데, 512개 무작위 방향의 최솟값은 양수라 "
                  "r_fc = 1, 표본 epsilon > 0이 된다. 6차원에서 512개 방향은 가장 가까운 방향까지 보통 수십 도 떨어져 있다")

(ROOT / "analysis" / "wrench_toy_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
for k, v in out.items():
    print(k, json.dumps(v, ensure_ascii=False)[:600])
