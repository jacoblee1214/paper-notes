"""토이: VISTA 식 (2)–(4), 명제 1·2의 구조를 numpy로 다시 만들어 동변성 오차를 잰다.

논문의 주장
  명제 1: 시각 토큰을 순열하면 융합 출력도 같은 순열이다 (식 2–4, 부록 A.3).
  명제 2: 센서 국소 특징이 전체 회전에서 변하지 않으면, 융합 뒤 구면 조화 계수에 D^l(R_t)를 곱하는 것만으로
          작업 공간 회전 g에 대해 π(g·O) = g·π(O)가 된다 (식 1, §3.4).
이 토이가 보는 것
  T1. 교차 주의 융합(식 2–4)이 토큰 순열과 가환하는지 (수치 오차).
  T2. 융합 방식이 교차 주의든 단순 연결이든, 보정 D^l(R_t) 뒤 출력이 g에 대해 동변인지.
      → 동변성은 융합 방식에 의존하지 않고, 센서 국소 특징이 불변이라는 가정에 의존한다.
  T3. 가정이 깨질 때(국소 특징이 세계 방향에 beta만큼 의존) 동변성 오차가 beta에 따라 어떻게 자라는지.
  T4. 위치 중심 이동(c_ws)이 회전 중심과 같을 때만 동변인지.
  T5. 헬피스 반구 토큰 수 P와 구면 조화 차원 산술.
설계 단서: 실제 e3nn 컨볼루션과 학습은 쓰지 않았다(설치하지 않음). l <= 2의 실수 구면 조화와 직접 구한 D^l로 대체한 축소판이다.
실행: python3 analysis/toy_equivariance.py   (numpy만, 수 초)
"""
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_equivariance_results.json"
rng = np.random.default_rng(0)


# ------------------------------------------------------------ 실수 구면 조화 (l <= 2), 정규화는 '구성 요소' 방식
def real_sh(x):
    """x: (M,3) 단위벡터 → (M,9). 순서: l=0, l=1(y,z,x), l=2(5개)."""
    X, Y, Z = x[:, 0], x[:, 1], x[:, 2]
    out = [np.ones_like(X),
           math.sqrt(3) * Y, math.sqrt(3) * Z, math.sqrt(3) * X,
           math.sqrt(15) * X * Y, math.sqrt(15) * Y * Z,
           math.sqrt(5) / 2 * (3 * Z**2 - 1), math.sqrt(15) * X * Z,
           math.sqrt(15) / 2 * (X**2 - Y**2)]
    return np.stack(out, axis=1)


L_SLICES = [slice(0, 1), slice(1, 4), slice(4, 9)]


def rand_rot(r=rng):
    q = r.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def wigner(R, pts=None):
    """Y(R x) = D(R) Y(x)를 만족하는 D(R) (9x9, l 블록 대각)를 점들에서 최소제곱으로 구한다."""
    pts = pts if pts is not None else _PTS
    A = real_sh(pts)            # (M,9)
    B = real_sh(pts @ R.T)      # (M,9) = Y(R x)
    Dt, *_ = np.linalg.lstsq(A, B, rcond=None)
    D = Dt.T
    mask = np.zeros((9, 9), bool)
    for s in L_SLICES:
        mask[s, s] = True
    assert np.abs(D[~mask]).max() < 1e-9, "D가 l 블록 대각이 아님"
    return D


_PTS = rng.normal(size=(60, 3))
_PTS /= np.linalg.norm(_PTS, axis=1, keepdims=True)


def fib_hemisphere(P):
    """z >= 0 반구 위의 거의 균일한 P개 점 (피보나치 격자). 센서 국소 프레임에 고정된 격자."""
    i = np.arange(P) + 0.5
    z = i / P
    phi = math.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z**2)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


# ------------------------------------------------------------ 융합 모듈 (식 2–4와 단순 연결)
class Fusion:
    def __init__(self, C=16, dh=8, seed=1):
        g = np.random.default_rng(seed)
        self.C, self.dh = C, dh
        self.Wq, self.Wk, self.Wu = (g.normal(size=(C, dh)) / math.sqrt(C) for _ in range(3))
        self.W1 = g.normal(size=(C + dh, C)) / math.sqrt(C + dh)
        self.Wc = g.normal(size=(2 * C, C)) / math.sqrt(2 * C)

    @staticmethod
    def ln(x):
        return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)

    def attn(self, V, T):
        q, k, u = V @ self.Wq, T @ self.Wk, T @ self.Wu
        s = q @ k.T / math.sqrt(self.dh)
        a = np.exp(s - s.max(1, keepdims=True))
        a /= a.sum(1, keepdims=True)
        m = a @ u
        h = np.tanh(np.concatenate([V, m], 1) @ self.W1)
        return self.ln(V + h)

    def concat(self, V, T):
        return self.ln(np.tanh(np.concatenate([V, T], 1) @ self.Wc))


def encode(fusion_fn, V, T, dirs, w):
    """융합 → 구면 조화 투영 c = sum_i f_i Y(s_i) w (식 10) → (C,9)."""
    F = fusion_fn(V, T)
    return F.T @ (real_sh(dirs) * w)


def readout_vector(c_corr, a):
    """보정된 계수의 l=1 블록에서 채널 가중합으로 3벡터를 만든다 (y,z,x 순서를 x,y,z로 되돌림)."""
    c1 = c_corr[:, 1:4]                   # (C,3) 순서 (y,z,x)
    v = a @ c1                            # (3,)
    return np.array([v[2], v[0], v[1]])   # (x,y,z)


def equiv_error(fusion_fn, beta=0.0, trials=200, C=16, P=96):
    dirs = fib_hemisphere(P)
    w = 1 / math.sqrt(P)
    g_ = np.random.default_rng(5)
    a = g_.normal(size=C)
    errs = []
    for _ in range(trials):
        V0 = g_.normal(size=(P, C))
        T0 = g_.normal(size=(P, C))
        R = rand_rot(g_)
        g = rand_rot(g_)
        # 세계 방향에 의존하는 성분: 센서 국소 특징에 R(손목 자세)의 일부가 새는 경우
        leak = g_.normal(size=(P, C, 9)) / 3
        def obs_feats(Rt):
            Vt = V0 + beta * (leak[..., :9] @ Rt.reshape(9))  # (P,C)
            return Vt, T0
        # 원 관측
        Va, Ta = obs_feats(R)
        c = encode(fusion_fn, Va, Ta, dirs, w)
        cc = (wigner(R) @ c.T).T          # c_corr = D(R_t) c  (채널 C, 계수 9)
        v0 = readout_vector(cc, a)
        # 회전된 관측: 자세 gR, 센서 국소 특징은 불변(beta=0) 또는 누설
        Vb, Tb = obs_feats(g @ R)
        c2 = encode(fusion_fn, Vb, Tb, dirs, w)
        cc2 = (wigner(g @ R) @ c2.T).T
        v1 = readout_vector(cc2, a)
        errs.append(np.linalg.norm(v1 - g @ v0) / (np.linalg.norm(v0) + 1e-12))
    return float(np.mean(errs)), float(np.max(errs))


def healpix_counts(nside=8):
    """HEALPix 링 구조에서 z >= 0 픽셀 수. z = 0 링은 경계라서 포함 여부에 따라 두 값이 나온다."""
    n = nside
    zs, cnt = [], []
    for i in range(1, 4 * n):
        if i < n:
            z, k = 1 - i**2 / (3 * n**2), 4 * i
        elif i <= 3 * n:
            z, k = 4 / 3 - 2 * i / (3 * n), 4 * n
        else:
            z, k = -1 + (4 * n - i)**2 / (3 * n**2), 4 * (4 * n - i)
        zs.append(z)
        cnt.append(k)
    zs, cnt = np.array(zs), np.array(cnt)
    assert cnt.sum() == 12 * n * n
    return int(cnt[zs > 1e-12].sum()), int(cnt[zs >= -1e-12].sum()), int(cnt.sum())


def main():
    R = {}
    fz = Fusion()
    # T1: 식 2–4 순열 가환
    P, C = 96, 16
    g = np.random.default_rng(2)
    V, T = g.normal(size=(P, C)), g.normal(size=(P, C))
    perm = g.permutation(P)
    d1 = np.abs(fz.attn(V[perm], T) - fz.attn(V, T)[perm]).max()
    d2 = np.abs(fz.concat(V[perm], T) - fz.concat(V, T)[perm]).max()
    # 반례: 시각 토큰만 순열하고 연결 모듈에 같은 위치 토큰 짝이 필요한 경우(T도 같이 순열하지 않으면 연결은 깨짐)
    R["T1_attention_permutation_max_abs_err"] = float(d1)
    R["T1_concat_permutation_max_abs_err_same_perm_on_visual_only"] = float(d2)
    # 해석: 연결은 시각 i와 촉각 i를 짝짓기 때문에 시각만 순열하면 출력이 달라진다
    R["T1_note"] = "concat은 인덱스 i끼리 짝짓는 연산이라 시각 토큰만 순열하면 순열 가환이 아님 (오차가 큼). 교차 주의는 촉각 토큰 전체를 보므로 가환."

    # T2: 동변성 오차 (beta = 0)
    for name, fn in [("attention", fz.attn), ("concat", fz.concat)]:
        mean_e, max_e = equiv_error(fn, beta=0.0)
        R[f"T2_equivariance_rel_err_{name}"] = {"mean": mean_e, "max": max_e}

    # T3: 가정 위반 정도 beta에 따른 오차
    betas = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
    R["T3_beta_sweep"] = {"betas": betas, "attention_mean_rel_err": [], "concat_mean_rel_err": []}
    for b in betas:
        R["T3_beta_sweep"]["attention_mean_rel_err"].append(equiv_error(fz.attn, beta=b, trials=100)[0])
        R["T3_beta_sweep"]["concat_mean_rel_err"].append(equiv_error(fz.concat, beta=b, trials=100)[0])

    # T4: 위치 중심 이동. 정책은 (p - c)를 입력으로 쓴다. 회전 g를 원점 중심으로 적용하면 (g p - c) != g (p - c)
    c = np.array([0.0, 0.0, 0.8])
    g4 = np.random.default_rng(7)
    p = g4.normal(size=3)
    errs_origin, errs_about_c, errs_yaw = [], [], []
    for _ in range(200):
        g = rand_rot(g4)
        errs_origin.append(np.linalg.norm((g @ p - c) - g @ (p - c)))        # 회전 중심 = 원점
        errs_about_c.append(np.linalg.norm((g @ (p - c) + c - c) - g @ (p - c)))  # 회전 중심 = c
    th = np.deg2rad(45)
    Rz = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    errs_yaw = float(np.linalg.norm((Rz @ p - c) - Rz @ (p - c)))             # C8 원소: 연직축 회전, c가 그 축 위
    R["T4_center_shift"] = {"mean_err_rotate_about_origin_I60like": float(np.mean(errs_origin)),
                            "mean_err_rotate_about_c": float(np.mean(errs_about_c)),
                            "err_yaw_rotation_about_z_axis_c_on_axis": errs_yaw}

    # T5: 차원과 토큰 수
    lo, hi, tot = healpix_counts(8)
    R["T5_healpix_nside8"] = {"tokens_z_gt_0": lo, "tokens_z_ge_0": hi, "total_pixels": tot,
                              "tactile_proj_logits_params_49xP": [49 * lo, 49 * hi]}
    R["T5_coeff_dims"] = {"(lmax+1)^2": 49, "so3_dim": 455, "channels_obs_out": 128,
                          "obs_coeff_numbers_per_frame": 128 * 455}
    # 공식 코드의 융합 모듈 파라미터 수(정적 계산): q,k,v: 1024->512, out: 512->1024, fuse: 2048->1024->1024, LN 1024
    d, inner = 1024, 8 * 64
    attn_params = 3 * (d * inner + inner) + (inner * d + d) + (2 * d * d + d) + (d * d + d) + 2 * d
    concat_params = 2 * d + (2 * d * d + d)  # LN(2048) + Linear(2048->1024)
    R["T5_fusion_params_static"] = {"cross_attention_block": attn_params, "concat_block": concat_params,
                                    "ratio": attn_params / concat_params}
    OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(R, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
