"""토이 T1: 픽셀 특징(cx, cy, Δcx, Δcy, Δw, Δh)이 물체의 접근 정보를 얼마나 담는지 검사한다.

중심 주장: 금속 좌표 없이 한 장의 RGB에서 얻은 중심과 크기 변화만으로 정책이 접근 동역학을 추정할 수 있다.
검사: 논문 코드(catchpolicy/pixel2catch.py get_object_random_pose, 480–539행)와 같은 탄도 궤적을 만들고,
      핀홀 카메라(cfg의 intrinsic_matrix, 카메라 위치 (-0.39, 0.01, 2.2))로 구 모양 물체를 투영한다.
      특징(시간 t와 t-1의 값을 이어 붙인 12차원)으로 두 목표를 회귀한다.
      목표 1: 물체가 z_catch에 닿기까지 남은 시간 τ (정책이 알아야 할 접근 시점)
      목표 2: 카메라에서 물체까지의 깊이 Z (크기 정보 없이 거리를 알 수 있는지)
      비교: 같은 회귀에 3D 상태(위치, 속도)를 넣은 오라클.
조건: A 물체 크기 고정(반지름 3.5 cm), B 크기 변동(2.5–4.5 cm), C 크기 변동 + 모서리 ±5 px 잡음 + 지연 0–2 스텝.

단순화(논문과 다른 점)
- 카메라 방향은 cfg의 쿼터니언 규약을 검증하지 못해, 카메라 위치에서 작업공간 중심 (0.6, 0, 0.9)을 바라보게 했다.
- 물체는 구로 가정하고 w = h = 2·f·r/Z 로 근사한다. 실제 코드의 원시 물체는 원뿔, 캡슐 등 10종이다.
- 눌림, 항력, 방출 직전 공중 정지(stay_vz)는 무시했다.
- 특징은 코드처럼 중심은 W, H로 나누고 Δ는 픽셀 단위로 둔다.

출력: analysis/toy_pixel_features_results.json. 실행: python3 analysis/toy_pixel_features.py (numpy, 수 초)
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_pixel_features_results.json"
G = 9.81
FX, FY, CX0, CY0 = 618.163, 618.468, 328.451, 246.513  # cfg intrinsic_matrix
W, H = 640, 480
CAM = np.array([-0.39, 0.01, 2.2])
TARGET = np.array([0.6, 0.0, 0.9])
ZCATCH_HZ = 30.0  # 정책 주기


def cam_R():
    f = TARGET - CAM
    f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    return np.stack([right, down, f], axis=1)  # 열: 카메라 x(오른쪽), y(아래), z(앞)


R = cam_R()


def project(P):
    pc = (P - CAM) @ R
    z = pc[..., 2]
    u = FX * pc[..., 0] / z + CX0
    v = FY * pc[..., 1] / z + CY0
    return u, v, z


def sample_traj(rng, n):
    """get_object_random_pose 와 같은 식(480–539행)으로 시작 위치, 속도를 뽑는다."""
    Xs = rng.random(n) * 0.5 + 2.4
    Ys = rng.random(n) * 0.6 - 0.3
    Zs = (2.0 * rng.random(n) - 1.0) * 0.05 + 0.8
    xT = rng.random(n) * 0.2 + 0.65
    yT = rng.random(n) * 0.7 - 0.35
    z_catch = rng.random(n) * 0.2 + 0.9
    peak = np.maximum(rng.random(n) * 0.3 + 1.6, np.maximum(Zs, z_catch) + 0.2)
    vz = np.sqrt(2 * G * (peak - Zs))
    T = vz / G + np.sqrt(2 * (peak - z_catch) / G)
    vx = (xT - Xs) / T
    vy = (yT - Ys) / T
    p0 = np.stack([Xs, Ys, Zs], 1)
    v0 = np.stack([vx, vy, vz], 1)
    return p0, v0, z_catch


def frames(p0, v0, z_catch, rng, radius, n_frames=12):
    """정책 주기(30 Hz)로 샘플. 공이 z_catch 위에 있는 동안의 프레임만 쓴다."""
    n = p0.shape[0]
    t_hit = None
    # 하강 중 z_catch에 닿는 시각
    disc = v0[:, 2] ** 2 + 2 * G * (p0[:, 2] - z_catch)
    t_hit = (v0[:, 2] + np.sqrt(np.maximum(disc, 0))) / G  # 하강 시각
    out = []
    for k in range(n_frames * 3):
        t = k / ZCATCH_HZ
        pos = p0 + v0 * t
        pos[:, 2] = p0[:, 2] + v0[:, 2] * t - 0.5 * G * t * t
        tau = t_hit - t
        ok = (tau > 0.05) & (pos[:, 2] > 0.2) & (pos[:, 0] > 0.3)
        u, v, z = project(pos)
        w = 2 * FX * radius / z
        hgt = 2 * FY * radius / z
        out.append((t, pos, tau, u, v, z, w, hgt, ok))
    return out


def build_features(p0, v0, z_catch, rng, radius, noise_px=0.0, lag_max=0):
    fr = frames(p0, v0, z_catch, rng, radius)
    rows_x, rows_tau, rows_z, rows_pos, valid = [], [], [], [], []
    n = p0.shape[0]
    # 시간별 특징을 쌓고, 각 샘플의 (t, t-1) 특징을 이어 붙인다
    feats = []
    for (t, pos, tau, u, v, z, w, hgt, ok) in fr:
        cxx = u.copy()
        cyy = v.copy()
        if noise_px > 0:
            # 바운딩 박스 모서리에 정수 잡음 (코드: randint(-5,5))
            for c in [0]:
                pass
            dx0 = rng.integers(-5, 5, size=n) * (noise_px / 5.0)
            dx1 = rng.integers(-5, 5, size=n) * (noise_px / 5.0)
            dy0 = rng.integers(-5, 5, size=n) * (noise_px / 5.0)
            dy1 = rng.integers(-5, 5, size=n) * (noise_px / 5.0)
            x0 = u - w / 2 + dx0
            x1 = u + w / 2 + dx1
            y0 = v - hgt / 2 + dy0
            y1 = v + hgt / 2 + dy1
            cxx = (x0 + x1) / 2
            cyy = (y0 + y1) / 2
            w = x1 - x0
            hgt = y1 - y0
        feats.append((cxx, cyy, w, hgt, pos, tau, z, ok))
    X, Y = [], []
    for k in range(1, len(feats)):
        cxk, cyk, wk, hk, pk, tauk, zk, okk = feats[k]
        cxp, cyp, wp, hp, _, _, _, okp = feats[k - 1]
        if lag_max > 0:
            lag = rng.integers(0, lag_max + 1, size=n)
            # 지연: 이전 프레임을 최대 lag_max 스텝까지 골라 씀
            idx = np.clip(k - lag, 0, None)
            cxk_l = np.array([feats[j][0][i] for i, j in enumerate(idx)])
            cyk_l = np.array([feats[j][1][i] for i, j in enumerate(idx)])
            wk_l = np.array([feats[j][2][i] for i, j in enumerate(idx)])
            hk_l = np.array([feats[j][3][i] for i, j in enumerate(idx)])
            cxk, cyk, wk, hk = cxk_l, cyk_l, wk_l, hk_l
        f = np.stack([cxk / W, cyk / H, cxk - cxp, cyk - cyp, wk - wp, hk - hp], 1)
        X.append(f)
        Y.append(np.stack([tauk, zk, pk[:, 0], pk[:, 1], pk[:, 2]], 1))
        valid.append(okk & okp)
    X = np.stack(X, 1)  # (n, T, 6)
    Y = np.stack(Y, 1)
    valid = np.stack(valid, 1)
    return X, Y, valid


def poly2(F):
    n, d = F.shape
    cols = [np.ones(n)] + [F[:, i] for i in range(d)]
    for i in range(d):
        for j in range(i, d):
            cols.append(F[:, i] * F[:, j])
    return np.stack(cols, 1)


def ridge_fit_eval(Xtr, ytr, Xte, yte, lam=1e-3):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    A = np.concatenate([np.ones((A.shape[0], 1)), A], 1)
    B = np.concatenate([np.ones((B.shape[0], 1)), B], 1)
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ ytr)
    pred = B @ w
    ss_res = np.sum((yte - pred) ** 2)
    ss_tot = np.sum((yte - yte.mean()) ** 2)
    return float(1 - ss_res / ss_tot), float(np.sqrt(np.mean((yte - pred) ** 2)))


def run(cond, rng, n_tr=3000, n_te=1000):
    radius_fn = cond["radius"]
    out = {}
    for split, n in [("tr", n_tr), ("te", n_te)]:
        p0, v0, zc = sample_traj(rng, n)
        r = radius_fn(rng, n)
        X, Y, valid = build_features(p0, v0, zc, rng, r, noise_px=cond["noise"], lag_max=cond["lag"])
        out[split] = (X, Y, valid)
    Xtr, Ytr, Vtr = out["tr"]
    Xte, Yte, Vte = out["te"]
    # 두 프레임(t-1, t)을 쌓아 한 샘플로 만든다. 시간 index는 같은 타임라인을 쓴다.
    def stack(X, Y, V):
        Fs, Ts = [], []
        for k in range(X.shape[1]):
            m = V[:, k]
            if k == 0:
                continue
            Fs.append(X[m, k - 1 if k - 1 >= 0 else k])
            Ts.append(Y[m, k])
        return np.concatenate(Fs), np.concatenate(Ts)
    # 특징은 (t-1, t) 둘 다 쓴다
    def pair(X, Y, V):
        Fs, Ts = [], []
        for k in range(1, X.shape[1]):
            m = V[:, k] & V[:, k - 1]
            Fs.append(np.concatenate([X[m, k - 1], X[m, k]], 1))
            Ts.append(Y[m, k])
        return np.concatenate(Fs), np.concatenate(Ts)
    Ftr, Ttr = pair(Xtr, Ytr, Vtr)
    Fte, Tte = pair(Xte, Yte, Vte)
    Ptr, Pte = poly2(Ftr), poly2(Fte)
    res = {}
    names = ["tau_s", "depth_m", "X_m", "Y_m", "Zobj_m"]
    for j, name in enumerate(["tau_s", "depth_m"]):
        r2, rmse = ridge_fit_eval(Ptr, Ttr[:, j], Pte, Tte[:, j])
        res[name] = {"r2": round(r2, 3), "RMSE": round(rmse, 4)}
    # 기준: 목표 분산 대비 평균만 쓰는 경우의 RMSE
    res["n_train"] = int(Ftr.shape[0])
    res["n_test"] = int(Fte.shape[0])
    return res


def oracle(rng, n_tr=3000, n_te=1000):
    """3D 위치, 속도와 반지름(알고 있다고 가정)으로 같은 목표를 회귀."""
    res = {}
    for split, n in [("tr", n_tr), ("te", n_te)]:
        p0, v0, zc = sample_traj(rng, n)
        feats = []
        taus, deps = [], []
        for k in range(1, 30):
            t = k / ZCATCH_HZ
            pos = p0 + v0 * t
            pos[:, 2] = p0[:, 2] + v0[:, 2] * t - 0.5 * G * t * t
            vel = v0.copy()
            vel[:, 2] = v0[:, 2] - G * t
            disc = v0[:, 2] ** 2 + 2 * G * (p0[:, 2] - zc)
            t_hit = (v0[:, 2] + np.sqrt(np.maximum(disc, 0))) / G
            tau = t_hit - t
            _, _, z = project(pos)
            ok = (tau > 0.05) & (pos[:, 2] > 0.2) & (pos[:, 0] > 0.3)
            feats.append((np.concatenate([pos, vel], 1), tau, z, ok))
        F = np.concatenate([f[0][f[3]] for f in feats])
        T = np.concatenate([np.stack([f[1][f[3]], f[2][f[3]]], 1) for f in feats])
        res[split] = (F, T)
    Ftr, Ttr = res["tr"]
    Fte, Tte = res["te"]
    out = {}
    for j, name in enumerate(["tau_s", "depth_m"]):
        r2, rmse = ridge_fit_eval(poly2(Ftr), Ttr[:, j], poly2(Fte), Tte[:, j])
        out[name] = {"r2": round(r2, 3), "RMSE": round(rmse, 4)}
    return out


def main():
    rng = np.random.default_rng(0)
    conds = {
        "A_fixed_size": {"radius": lambda r, n: np.full(n, 0.035), "noise": 0.0, "lag": 0},
        "B_size_varies": {"radius": lambda r, n: r.uniform(0.025, 0.045, n), "noise": 0.0, "lag": 0},
        "C_size_noise_lag": {"radius": lambda r, n: r.uniform(0.025, 0.045, n), "noise": 5.0, "lag": 2},
    }
    res = {}
    for name, c in conds.items():
        res[name] = run(c, np.random.default_rng(1))
    res["oracle_3d_state"] = oracle(np.random.default_rng(2))
    # 기준: 목표 평균만 쓰는 경우 R² = 0
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
