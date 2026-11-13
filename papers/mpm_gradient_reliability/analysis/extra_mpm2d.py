"""2차원 MLS-MPM 축소 구현과 손으로 쓴 전진 접선(forward tangent)으로 0.34 불일치의 원인을 구분하는 검증 (CPU numpy, 1스레드).

질문: 논문의 f64 AD/FD 불일치(20 스텝, 0.34)가 (H1) FD 섭동 크기(절단·반올림), (H2) 항복 분기 전환(섭동이 임계를 넘는 불연속),
(H3) SVD 간격 민감도(극분해 미분의 1/(s_j²−s_i²)) 중 무엇에서 오는지 소규모 모형에서 본다.

구현 범위(논문 대비 축소):
  - 2차원, 격자 40×40(dx = 1/40), 입자 400개(한 변 0.25 정사각 블록), 블록은 아래로 초기 속도를 받아 바닥에 닿는다.
    바닥과 좌·우 벽은 3칸 두께 sticky 경계(격자 속도 0). 중력 (0, −9.8). dt = 1e-3.
  - MLS-MPM(APIC, 2차 B-spline 가중치). 격자 질량이 입자 위치에 의존하므로 그 미분도 전파한다.
  - 고정 corotated 탄성(P = 2μ(F−R) + λ(J−1)J F^{−T}, R = 극분해 인자).
  - 소성: 로그 특이값 ε = log σ에서 ε1−ε2 > Y이면 차이를 Y로 되돌리는 복귀 사상(평균 로그 변형률 보존, 경화 없음).
  - 기울기: 매개변수 θ ∈ {E, Y} 방향 전진 접선을 손으로 전파한다(P2G, 격자 갱신, G2P, F 갱신, 복귀 사상, 극분해).
    손실 L_T = (1/N) Σ_p |x_p(T) − x*_p(T)|², x*는 E* = 1.2 E0, Y* = 1.2 Y0로 만든 목표 궤적.
  - 극분해 미분 두 가지 (K = Uᵀ dF V):
      general    : dR = U(Ω−Φ)Vᵀ, Ω_ij = (s_j K_ij + s_i K_ji)/(s_j²−s_i²), Φ_ij = (s_i K_ij + s_j K_ji)/(s_j²−s_i²)
      gauge_safe : dR = U M Vᵀ, M_ij = (K_ij − K_ji)/(s_i + s_j), M_ii = 0 (같은 식을 1/(s_j²−s_i²)를 거치지 않고 정리)
    둘 다 정확한 항등식이다. 간격 s_i ≈ s_j에서 general은 Ω와 Φ가 각각 1/간격으로 커져 서로 빼면서 반올림 오차가 남는다.
  - 복귀 사상이 바뀐 입자의 F 미분은 시험 F의 SVD(general 형)로 계산한다. 소성 입자의 간격은 Y만큼 벌어져 있어 이 항은 간격에 민감하지 않다.
  - 기준: 중심 차분, 상대 섭동 h = 1e-2 … 1e-12(θ0 기준), 같은 정밀도 안에서 계산한다.
  - 정밀도: f64와 f32(상태, 격자 누적, 접선 모두 같은 자료형). 격자 누적은 np.add.at(순차)이므로 f32의 누적 순서 효과가 실제로 들어간다.
  - 초기 F: 노이즈 0(정확한 F = I), 노이즈 1e-15와 1e-13(간격이 반올림 수준 근처), 노이즈 1e-2(간격 0.01 근처).

한계: 2차원, 입자 400개, 경화 없는 로그 변형률 항복, 접촉은 sticky 경계뿐(접촉 모형 없음), 논문과 같은 Taichi 커널이나 DPSI 설정이 아니다.
실행: python3 analysis/extra_mpm2d.py -> analysis/extra_mpm2d_results.json
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

NN = 40                       # 격자 노드 수 (dx = 1/NN)
DX = 1.0 / NN
DT = 1e-3
GRAV = np.array([0.0, -9.8])
NU = 0.3
MU_PER_E = 1.0 / (2.0 * (1.0 + NU))
LAM_PER_E = NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
SP = DX / 2.0                 # 입자 간격
VP = SP * SP                  # 초기 입자 부피
MP = 1.0 * VP                 # 질량 (밀도 1)
NB = 20                       # 블록 한 변 입자 수 -> N = 400
X_LEFT, Y_BOTTOM, V_INIT = 0.375, 0.12, -4.0
E0, Y0 = 5.0, 0.03           # 평가점: 탄성 계수, 항복(로그 특이값 차이 한계)
E_STAR_FAC, Y_STAR_FAC = 1.2, 1.2   # 목표 궤적의 매개변수 배율
HS = [10.0 ** (-k / 2.0) for k in range(1, 25)]   # 상대 섭동 10^-0.5 … 1e-12 (반 자릿수 간격)
TMAX = 40

IDX = np.arange(NN * NN)
IX_ALL, IY_ALL = IDX // NN, IDX % NN
BAND = (IX_ALL < 3) | (IX_ALL > NN - 4) | (IY_ALL < 3)
AR = np.arange(3)
EYE2 = np.eye(2)


# ---------------------------------------------------------------- 가중치와 극분해 미분

def wts(f):
    """2차 B-spline 가중치와 그 fx 미분. f는 (N,) 분수 좌표(기준 노드 기준 0.5 … 1.5)."""
    w = np.stack([0.5 * (1.5 - f) ** 2, 0.75 - (f - 1.0) ** 2, 0.5 * (f - 0.5) ** 2], axis=-1)
    dw = np.stack([-(1.5 - f), -2.0 * (f - 1.0), f - 0.5], axis=-1)
    return w, dw


def sv_offdiag_general(S, K):
    """Ω, Φ의 비대각 성분(식 Ω_ij = (s_j K_ij + s_i K_ji)/D, Φ_ij = (s_i K_ij + s_j K_ji)/D, D = s_j² − s_i²)."""
    s2 = S * S
    Si, Sj = S[:, :, None], S[:, None, :]
    D = s2[:, None, :] - s2[:, :, None]
    KT = np.swapaxes(K, -1, -2)
    with np.errstate(divide="ignore", invalid="ignore"):
        Om = (Sj * K + Si * KT) / D
        Ph = (Si * K + Sj * KT) / D
    Om[:, [0, 1], [0, 1]] = 0.0
    Ph[:, [0, 1], [0, 1]] = 0.0
    return Om, Ph


def dpolar(U, S, Vt, dF, mode):
    """극분해 인자 R = U Vᵀ의 방향 미분 dR. mode: 'general' 또는 'gauge_safe'."""
    K = np.swapaxes(U, -1, -2) @ dF @ np.swapaxes(Vt, -1, -2)
    if mode == "gauge_safe":
        M = (K - np.swapaxes(K, -1, -2)) / (S[:, :, None] + S[:, None, :])
        return U @ M @ Vt
    Om, Ph = sv_offdiag_general(S, K)
    return U @ (Om - Ph) @ Vt


# ---------------------------------------------------------------- 초기 상태와 시뮬레이션

def init_state(noise, seed, f):
    xs = X_LEFT + (np.arange(NB) + 0.5) * SP
    ys = Y_BOTTOM + (np.arange(NB) + 0.5) * SP
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    x = np.stack([X.ravel(), Y.ravel()], axis=1)
    v = np.zeros_like(x)
    v[:, 1] = V_INIT
    F = np.tile(EYE2, (x.shape[0], 1, 1))
    if noise > 0:
        rng = np.random.default_rng(seed)
        F = F + noise * rng.standard_normal(F.shape)
    return x.astype(f), v.astype(f), F.astype(f)


def simulate(E, Yth, f, plastic, x0, v0, F0, T, tangent=None, mode="gauge_safe", target=None, keep_x=False):
    """MLS-MPM 롤아웃. tangent in {None, 'E', 'Y'}이면 같이 전파한다.

    반환: 단계 t = 1 … T의 손실 L[t], 기울기 G[t](tangent일 때), 로그 목록.
    target: (T+1, N, 2) 목표 궤적(f64로 만든 것을 f로 바꿔 씀). 없으면 손실은 nan.
    keep_x: True이면 모든 단계 위치(T+1, N, 2)를 xs로 돌려준다(목표 궤적 생성용).
    """
    N = x0.shape[0]
    c = f
    mu, lam = c(E * MU_PER_E), c(E * LAM_PER_E)
    Yv = c(Yth) if plastic else c(np.inf)
    dmu, dlam, dY = c(0.0), c(0.0), c(0.0)
    if tangent == "E":
        dmu, dlam = c(MU_PER_E), c(LAM_PER_E)
    elif tangent == "Y":
        dY = c(1.0)
    tang = tangent is not None
    dt, dx_ = c(DT), c(DX)
    c4 = c(4.0 * VP * DT / DX ** 2)
    att = c(4.0 / DX ** 2)
    mp = c(MP)
    grav = GRAV.astype(f) * c(1.0)
    eye = EYE2.astype(f)

    x = x0.copy()
    v = v0.copy()
    C = np.zeros((N, 2, 2), f)
    F = F0.copy()
    if tang:
        dxp = np.zeros((N, 2), f)
        dvp = np.zeros((N, 2), f)
        dC = np.zeros((N, 2, 2), f)
        dF = np.zeros((N, 2, 2), f)
    prev = np.zeros(N, bool)
    out = {"L": np.full(T + 1, np.nan), "G": np.full(T + 1, np.nan),
           "gap_min": np.zeros(T), "gap_lt8": np.zeros(T, int), "gap_lt12": np.zeros(T, int),
           "dist_min": np.zeros(T), "nplast": np.zeros(T, int), "switch": np.zeros(T, int),
           "pl_hist": np.zeros((T, N), bool),
           "xs": [x.copy()] if keep_x else None}

    for t in range(T):
        # ---- 응력과 그 미분 (현재 F 기준)
        U, S, Vt = np.linalg.svd(F)
        Rm = U @ Vt
        J = S[:, 0] * S[:, 1]
        Finv = np.linalg.inv(F)
        FiT = np.swapaxes(Finv, -1, -2)
        cJ = ((J - 1) * J)[:, None, None]
        P = c(2.0) * mu * (F - Rm) + lam * cJ * FiT
        Ft_ = np.swapaxes(F, -1, -2)
        A = mp * C - c4 * (P @ Ft_)
        gap = (S[:, 0] - S[:, 1]) / S[:, 0]
        out["gap_min"][t] = gap.min()
        out["gap_lt8"][t] = int((gap < 1e-8).sum())
        out["gap_lt12"][t] = int((gap < 1e-12).sum())
        if tang:
            dRm = dpolar(U, S, Vt, dF, mode)
            dJ = J * np.einsum("nij,nji->n", Finv, dF)
            dP = (c(2.0) * mu * (dF - dRm) + c(2.0) * dmu * (F - Rm) + dlam * cJ * FiT
                  + lam * ((c(2.0) * J - 1) * dJ)[:, None, None] * FiT
                  - lam * cJ * (FiT @ np.swapaxes(dF, -1, -2) @ FiT))
            dA = mp * dC - c4 * (dP @ Ft_ + P @ np.swapaxes(dF, -1, -2))

        # ---- P2G
        xg = x / dx_
        base = np.floor(xg - 0.5).astype(np.int64)
        fx = xg - base
        wx, dwx = wts(fx[:, 0])
        wy, dwy = wts(fx[:, 1])
        w = wx[:, :, None] * wy[:, None, :]                       # (N,3,3)
        off = np.empty((N, 3, 3, 2), f)
        off[..., 0] = (AR[None, :, None] - fx[:, 0, None, None]) * dx_
        off[..., 1] = (AR[None, None, :] - fx[:, 1, None, None]) * dx_
        ix = base[:, 0, None] + AR
        iy = base[:, 1, None] + AR
        flat = (ix[:, :, None] * NN + iy[:, None, :]).reshape(-1)
        Aoff = np.einsum("nij,nabj->nabi", A, off)
        mom = w[..., None] * (mp * v[:, None, None, :] + Aoff)
        mg = np.zeros(NN * NN, f)
        mvg = np.zeros((NN * NN, 2), f)
        np.add.at(mg, flat, (w * mp).reshape(-1))
        np.add.at(mvg, flat, mom.reshape(-1, 2))
        if tang:
            dfx = dxp / dx_
            dw = (dwx[:, :, None] * wy[:, None, :]) * dfx[:, 0, None, None] \
                + (wx[:, :, None] * dwy[:, None, :]) * dfx[:, 1, None, None]
            dmom = dw[..., None] * (mp * v[:, None, None, :] + Aoff) \
                + w[..., None] * (mp * dvp[:, None, None, :]
                                  + np.einsum("nij,nabj->nabi", dA, off)
                                  - np.einsum("nij,nj->ni", A, dxp)[:, None, None, :])
            dmg = np.zeros(NN * NN, f)
            dmvg = np.zeros((NN * NN, 2), f)
            np.add.at(dmg, flat, (dw * mp).reshape(-1))
            np.add.at(dmvg, flat, dmom.reshape(-1, 2))

        # ---- 격자 갱신 (sticky 경계)
        msafe = np.where(mg > 0, mg, c(1.0))
        vg = mvg / msafe[:, None] + dt * grav
        vg[BAND] = 0.0
        if tang:
            dvg = (dmvg - (mvg / msafe[:, None]) * dmg[:, None]) / msafe[:, None]
            dvg[BAND] = 0.0

        # ---- G2P
        vn = vg[flat].reshape(N, 3, 3, 2)
        vnew = np.einsum("nab,nabi->ni", w, vn)
        Cn = att * np.einsum("nab,nabi,nabj->nij", w, vn, off)
        xn = x + dt * vnew
        if tang:
            dvn = dvg[flat].reshape(N, 3, 3, 2)
            dvnew = np.einsum("nab,nabi->ni", dw, vn) + np.einsum("nab,nabi->ni", w, dvn)
            dCn = att * (np.einsum("nab,nabi,nabj->nij", dw, vn, off)
                         + np.einsum("nab,nabi,nabj->nij", w, dvn, off)
                         - vnew[:, :, None] * dxp[:, None, :])
            dxn = dxp + dt * dvnew

        # ---- F 갱신과 복귀 사상
        Fa = (eye + dt * Cn) @ F
        U2, S2, Vt2 = np.linalg.svd(Fa)
        eps = np.log(S2)
        diff = eps[:, 0] - eps[:, 1]
        pl = diff > Yv
        out["dist_min"][t] = float(np.min(np.abs(diff - Yv))) if np.isfinite(Yv) else np.inf
        out["nplast"][t] = int(pl.sum())
        out["switch"][t] = int((pl != prev).sum())
        out["pl_hist"][t] = pl
        prev = pl
        Fn = Fa
        if tang:
            dFa = dt * (dCn @ F) + (eye + dt * Cn) @ dF
            dFn = dFa
        if pl.any():
            m_ = 0.5 * (eps[pl, 0] + eps[pl, 1])
            h_ = 0.5 * Yv
            e = np.exp(np.stack([m_ + h_, m_ - h_], axis=-1))     # (np, 2)
            Up, Vp = U2[pl], Vt2[pl]
            Fn = Fn.copy()
            Fn[pl] = np.einsum("nij,nj,njk->nik", Up, e, Vp)
            if tang:
                Kp = np.swapaxes(Up, -1, -2) @ dFa[pl] @ np.swapaxes(Vp, -1, -2)
                Sp = S2[pl]
                dep = np.stack([Kp[:, 0, 0] / Sp[:, 0], Kp[:, 1, 1] / Sp[:, 1]], axis=-1)
                dm = 0.5 * (dep[:, 0] + dep[:, 1])
                dh = 0.5 * dY
                de = e * np.stack([dm + dh, dm - dh], axis=-1)
                Om, Ph = sv_offdiag_general(Sp, Kp)
                Cm = Om * e[:, None, :] - e[:, :, None] * Ph
                Cm[:, [0, 1], [0, 1]] = de
                dFn = dFa.copy()
                dFn[pl] = Up @ Cm @ Vp

        # ---- 상태 갱신
        x, v, C, F = xn, vnew, Cn, Fn
        if tang:
            dxp, dvp, dC, dF = dxn, dvnew, dCn, dFn

        if keep_x:
            out["xs"].append(x.copy())
        # ---- 손실과 기울기 (단계 t+1)
        if target is not None:
            r = x - target[t + 1].astype(f)
            out["L"][t + 1] = float(np.mean(np.sum(r.astype(np.float64) ** 2, axis=1)))
            if tang:
                out["G"][t + 1] = float(2.0 / N * np.sum(r.astype(np.float64) * dxp.astype(np.float64)))
    return out


def target_traj(E, Yth, plastic, noise, seed, T):
    """목표 궤적 (T+1, N, 2), f64."""
    x0, v0, F0 = init_state(noise, seed, np.float64)
    o = simulate(E, Yth, np.float64, plastic, x0, v0, F0, T, keep_x=True)
    return np.stack(o["xs"])


# ---------------------------------------------------------------- 실험 드라이버

INITS = [("exact", 0.0, [0]), ("g1e-15", 1e-15, [1, 2]), ("g1e-13", 1e-13, [1, 2]), ("normal", 1e-2, [1, 2, 3, 4])]
MODES = ["general", "gauge_safe"]
PRECS = {"f64": np.float64, "f32": np.float32}


def _nan(a):
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(a, float).ravel()]


def run_combo(kind, noise, seed, prec, param, plastic, tgt):
    f = PRECS[prec]
    x0, v0, F0 = init_state(noise, seed, f)
    rec = {"init": kind, "seed": seed, "prec": prec, "param": param, "plastic": plastic}
    ad = {}
    for mode in MODES:
        o = simulate(E0, Y0, f, plastic, x0, v0, F0, TMAX, tangent=param, mode=mode, target=tgt)
        ad[mode] = o
        rec[f"AD_{mode}"] = _nan(o["G"][1:])
        if mode == "gauge_safe":
            rec["L"] = _nan(o["L"][1:])
            rec["gap_min"] = _nan(o["gap_min"])
            rec["gap_lt8"] = o["gap_lt8"].tolist()
            rec["gap_lt12"] = o["gap_lt12"].tolist()
            rec["nplast"] = o["nplast"].tolist()
            rec["switch"] = o["switch"].tolist()
            rec["dist_min"] = _nan(o["dist_min"])
    fd_rows, flips, mism = [], [], {m: [] for m in MODES}
    for h in HS:
        if param == "E":
            tp, tm = E0 * (1 + h), E0 * (1 - h)
            ep = simulate(tp, Y0, f, plastic, x0, v0, F0, TMAX, target=tgt, keep_x=False)
            em = simulate(tm, Y0, f, plastic, x0, v0, F0, TMAX, target=tgt, keep_x=False)
            den = tp - tm
        else:
            tp, tm = Y0 * (1 + h), Y0 * (1 - h)
            ep = simulate(E0, tp, f, True, x0, v0, F0, TMAX, target=tgt)
            em = simulate(E0, tm, f, True, x0, v0, F0, TMAX, target=tgt)
            den = tp - tm
        fd = (ep["L"][1:] - em["L"][1:]) / den
        fd_rows.append(_nan(fd))
        flips.append(int((ep["pl_hist"] != em["pl_hist"]).any(axis=0).sum()))
        for m in MODES:
            mism[m].append(_nan(np.abs(ad[m]["G"][1:] - fd) / np.abs(ad[m]["G"][1:])))
    rec["HS"] = HS
    rec["FD"] = fd_rows
    rec["flips"] = flips
    rec["mism_general"] = mism["general"]
    rec["mism_gauge_safe"] = mism["gauge_safe"]
    return rec


def polar_gap_check():
    """극분해 미분 단위 시험: 간격 g에서 두 식의 상대 오차 중앙값(f64, 중심 차분 h = 1e-6, 시드 20개)."""
    rng = np.random.default_rng(0)
    rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    polar = lambda F: np.linalg.svd(F)[0] @ np.linalg.svd(F)[2]
    out = {}
    for gap in [1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]:
        errs = {"general": [], "gauge_safe": []}
        for _ in range(20):
            F = rot(rng.uniform(0, 2 * np.pi)) @ np.diag([1.0, 1.0 - gap]) @ rot(rng.uniform(0, 2 * np.pi)).T
            dF = rng.standard_normal((2, 2))
            h = 1e-6
            ref = (polar(F + h * dF) - polar(F - h * dF)) / (2 * h)
            U, S, Vt = np.linalg.svd(F)
            for mode in errs:
                d = dpolar(U[None], S[None], Vt[None], dF[None], mode)[0]
                errs[mode].append(float(np.linalg.norm(d - ref) / np.linalg.norm(ref)))
        out[f"{gap:.0e}"] = {m: float(np.median(v)) for m, v in errs.items()}
    return out


def main():
    t_start = time.time()
    combos = []
    tgt_cache = {}
    for kind, noise, seeds in INITS:
        for seed in seeds:
            for plastic in (False, True):
                tgt_cache[(kind, seed, plastic)] = target_traj(E0 * E_STAR_FAC, Y0 * Y_STAR_FAC, plastic, noise, seed, TMAX)
            for prec in PRECS:
                for param, plastic in (("E", False), ("E", True), ("Y", True)):
                    tgt = tgt_cache[(kind, seed, plastic)]
                    rec = run_combo(kind, noise, seed, prec, param, plastic, tgt)
                    combos.append(rec)
                    print(f"{kind:>6} seed {seed} {prec} {param} plastic={int(plastic)}  "
                          f"t={time.time()-t_start:6.1f}s", flush=True)
    # f32 AD의 정확도: f64 AD 대비 (같은 시드, 같은 설정)
    key = lambda r: (r["init"], r["seed"], r["param"], r["plastic"])
    ad64 = {key(r): r for r in combos if r["prec"] == "f64"}
    for r in combos:
        if r["prec"] == "f32":
            ref = ad64[key(r)]
            for m in MODES:
                a32, a64 = np.array(r[f"AD_{m}"], float), np.array(ref[f"AD_{m}"], float)
                r[f"ad32_vs64_{m}"] = _nan(np.abs(a32 - a64) / np.abs(a64))
    out = {"polar_gap_check": polar_gap_check(),
           "settings": {"NN": NN, "DX": DX, "DT": DT, "N_particles": NB * NB, "E0": E0, "Y0": Y0,
                        "E_star_factor": E_STAR_FAC, "Y_star_factor": Y_STAR_FAC, "V_INIT": V_INIT,
                        "Y_BOTTOM": Y_BOTTOM, "TMAX": TMAX, "HS": HS, "INITS": [(k, n, s) for k, n, s in INITS],
                        "nu": NU, "gravity": GRAV.tolist(),
                        "note": "FD는 같은 정밀도 안에서 중심 차분. AD는 general/gauge_safe 두 극분해 미분."},
           "combos": combos}
    (HERE / "extra_mpm2d_results.json").write_text(json.dumps(out))
    print(f"done in {time.time()-t_start:.1f}s; wrote extra_mpm2d_results.json", flush=True)


def load():
    return json.loads((HERE / "extra_mpm2d_results.json").read_text())


def _arr(v):
    return np.array([np.nan if x is None else x for x in v], float)


def summarize(D):
    """표 출력: (1) 정상 간격 설정의 T × 정밀도 × 매개변수 × 극분해 식, (2) 간격 제어 설정, (3) 큰 섭동 h-곡선, (4) 단계별 로그."""
    HS = np.array(D["settings"]["HS"])
    C = D["combos"]
    big = (HS >= 1e-2) & (HS <= 0.32)
    TT = [2, 5, 10, 20, 30, 40]
    lines = []
    lines.append("## (1) 정상 간격(normal, 시드 4개): 시드 중앙값의 상대 불일치")
    lines.append("열: min = 전 h 중 최소, max = h ∈ [1e-2, 0.32] 중 최대. 식은 normal에서 두 식이 같다.")
    lines.append("prec param plast formula | " + " | ".join(f"T{t} min/max" for t in TT))
    for prec in ("f64", "f32"):
        for param, pl in (("E", False), ("E", True), ("Y", True)):
            for mode in ("general", "gauge_safe"):
                rs = [r for r in C if r["init"] == "normal" and r["prec"] == prec and r["param"] == param and r["plastic"] == pl]
                row = []
                for t in TT:
                    mn, mx = [], []
                    for r in rs:
                        m = _arr(r[f"mism_{mode}"])[:, t - 1]
                        ok = np.isfinite(m)
                        mn.append(np.min(m[ok]) if ok.any() else np.nan)
                        okb = ok & big
                        mx.append(np.max(m[okb]) if okb.any() else np.nan)
                    row.append(f"{np.nanmedian(mn):.1e}/{np.nanmedian(mx):.1e}")
                lines.append(f"{prec} {param} {int(pl)} {mode} | " + " | ".join(row))
    lines.append("")
    lines.append("## (2) 간격 제어 설정: AD 비유한(NaN) 개수와 f64 mismatch(T=20, 40, 최소 h)")
    lines.append("init | formula | NaN 실행 수 | T20 min | T40 min | 두 식의 AD 차 최대(f64, T20)")
    for init in ("exact", "g1e-15", "g1e-13", "normal"):
        for mode in ("general", "gauge_safe"):
            rs = [r for r in C if r["init"] == init and r["prec"] == "f64" and r["param"] == "E" and r["plastic"] is True]
            nan_runs = sum(int(not np.all(np.isfinite(_arr(r[f"AD_{mode}"])))) for r in rs)
            m20 = [np.nanmin(_arr(r[f"mism_{mode}"])[:, 19]) if np.isfinite(_arr(r[f"mism_{mode}"])[:, 19]).any() else np.nan for r in rs]
            m40 = [np.nanmin(_arr(r[f"mism_{mode}"])[:, 39]) if np.isfinite(_arr(r[f"mism_{mode}"])[:, 39]).any() else np.nan for r in rs]
            dd = []
            for r in rs:
                g, gg = _arr(r["AD_gauge_safe"])[19], _arr(r["AD_general"])[19]
                dd.append(abs(g - gg) / abs(g) if np.isfinite(gg) else np.nan)
            lines.append(f"{init} | {mode} | {nan_runs}/{len(rs)} | {np.nanmedian(m20):.1e} | {np.nanmedian(m40):.1e} | "
                         f"{np.nanmax(dd) if np.isfinite(dd).any() else float('nan'):.1e}")
    lines.append("")
    lines.append("## (3) 섭동 크기별 불일치 (f64, normal, 시드 중앙값, gauge_safe, T=20 / T=40)")
    sel = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
    lines.append("h = " + " ".join(f"{HS[i]:.0e}" for i in sel))
    for param, pl in (("E", False), ("E", True), ("Y", True)):
        rs = [r for r in C if r["init"] == "normal" and r["prec"] == "f64" and r["param"] == param and r["plastic"] == pl]
        for T in (20, 40):
            med = np.median(np.array([_arr(r["mism_gauge_safe"])[:, T - 1] for r in rs]), axis=0)
            lines.append(f"{param} plast{int(pl)} T{T}: " + " ".join(f"{med[i]:.1e}" for i in sel))
        fl = np.median(np.array([r["flips"] for r in rs]), axis=0)
        lines.append(f"{param} plast{int(pl)} flips: " + " ".join(f"{fl[i]:.0f}" for i in sel))
    lines.append("")
    lines.append("## (4) 단계별 로그 (normal, 시드 1, f64, plastic on)")
    for param in ("E", "Y"):
        r = [q for q in C if q["init"] == "normal" and q["prec"] == "f64" and q["param"] == param and q["plastic"] and q["seed"] == 1][0]
        for t in (5, 10, 15, 20, 25, 30, 35, 40):
            lines.append(f"{param} step {t}: min gap {r['gap_min'][t-1]:.2e}, switch {r['switch'][t-1]}, plastic {r['nplast'][t-1]}, min |diff-Y| {r['dist_min'][t-1]:.2e}")
    lines.append("")
    lines.append("## (5) 극분해 식 단위 시험 (간격별 상대 오차 중앙값, FD h=1e-6)")
    for g, v in D["polar_gap_check"].items():
        lines.append(f"gap {g}: general {v['general']:.1e}, gauge_safe {v['gauge_safe']:.1e}")
    text = "\n".join(lines)
    print(text)
    return text


def make_figure(D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HS = np.array(D["settings"]["HS"])
    C = D["combos"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    pg = D["polar_gap_check"]
    gaps = [float(g) for g in pg]
    a1.loglog(gaps, [pg[g]["general"] for g in pg], "o-", color="#c0392b", label="general (1/(s_j^2-s_i^2))")
    a1.loglog(gaps, [pg[g]["gauge_safe"] for g in pg], "s-", color="#1f5fae", label="gauge-safe (1/(s_i+s_j))")
    a1.set_xlabel("singular-value gap g = (s1 - s2)/s1")
    a1.set_ylabel("relative error of dR vs central FD")
    a1.set_title("(a) polar-derivative error vs gap (2x2, f64)")
    a1.invert_xaxis()
    a1.grid(alpha=0.3, which="both")
    a1.legend(fontsize=8)
    specs = [("E", False, "E, plasticity off", "#7f8c8d", "-"),
             ("E", True, "E, plasticity on", "#1f5fae", "--"),
             ("Y", True, "yield threshold Y, plasticity on", "#c0392b", "-")]
    for param, pl, lab, col, ls in specs:
        rs = [r for r in C if r["init"] == "normal" and r["prec"] == "f64" and r["param"] == param and r["plastic"] == pl]
        M = np.array([_arr(r["mism_gauge_safe"])[:, 19] for r in rs])
        med = np.nanmedian(M, axis=0)
        q1, q3 = np.nanpercentile(M, 25, axis=0), np.nanpercentile(M, 75, axis=0)
        ok = np.isfinite(med) & (med > 0)
        a2.loglog(HS[ok], med[ok], ls, color=col, label=lab)
        a2.fill_between(HS[ok], np.maximum(q1[ok], 1e-16), q3[ok], color=col, alpha=0.15)
    a2.axhline(0.34, color="k", lw=0.8, ls=":", label="paper f64, 20 steps (0.34)")
    a2.set_xlabel("relative perturbation h")
    a2.set_ylabel("|AD - FD| / |AD| at T = 20")
    a2.set_title("(b) rollout mismatch vs h (f64, 4 seeds)")
    a2.invert_xaxis()
    a2.grid(alpha=0.3, which="both")
    a2.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    out = HERE.parent / "report" / "figures" / "toy_mpm_mismatch.png"
    fig.savefig(out, dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    import sys
    if "--report" in sys.argv:
        D = load()
        summarize(D)
        make_figure(D)
    else:
        main()
