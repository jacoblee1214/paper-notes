"""논문 핵심 식을 CPU에서 수치로 확인한다(학습 없음, 수 초).

V1: 식 (21) OT 조건부 벡터장 = d/dt ψ_t(x0) (ψ_t 식 (22))  [정의 검사]
V2: 식 (19) VP 역방향 벡터장 = 식 (15) 일반 공식에 μ, σ를 대입한 값  [식 유도 검사]
V3: 정리 2. L_FM(v) - L_CFM(v)가 v에 상관없이 상수인지 (1차원, 두 데이터 점 혼합)
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("toy_results.json")
rng = np.random.default_rng(0)
res = {}

# V1: OT 벡터장 식 (21)과 ψ_t의 시간 미분 비교
sig_min = 0.01
x0 = rng.normal(size=1000)
x1 = rng.normal(size=1000) * 2 + 1
t = rng.uniform(0.05, 0.95, size=1000)
h = 1e-6
psi = lambda tt: (1 - (1 - sig_min) * tt) * x0 + tt * x1
dpsi_fd = (psi(t + h) - psi(t - h)) / (2 * h)
y = psi(t)  # ψ_t(x0)
u21 = (x1 - (1 - sig_min) * (y - t * x1) / (1 - (1 - sig_min) * t))
res["V1_ot_vf_max_abs_err"] = float(np.max(np.abs(u21 - dpsi_fd)))

# V2: 식 (19) 역방향 VP 벡터장 vs 식 (15)에 μ=α_{1-t}x1, σ=sqrt(1-α_{1-t}^2) 대입
def T(s, bmin=0.1, bmax=20.0):
    return s * bmin + 0.5 * s * s * (bmax - bmin)

def Tp(s, bmin=0.1, bmax=20.0):
    return bmin + s * (bmax - bmin)

tt = rng.uniform(0.05, 0.95, size=2000)
xx = rng.normal(size=2000) * 1.5
xq = rng.normal(size=2000)
s = 1 - tt
alpha = np.exp(-0.5 * T(s))
mu = lambda tau: np.exp(-0.5 * T(1 - tau)) * xq
sg = lambda tau: np.sqrt(1 - np.exp(-T(1 - tau)))
eps = 1e-6
dmu = (mu(tt + eps) - mu(tt - eps)) / (2 * eps)
dsg = (sg(tt + eps) - sg(tt - eps)) / (2 * eps)
u15 = dsg / sg(tt) * (xx - mu(tt)) + dmu
u19 = -(Tp(s) / 2) * (np.exp(-T(s)) * xx - np.exp(-0.5 * T(s)) * xq) / (1 - np.exp(-T(s)))
res["V2_vp_vf_max_abs_err"] = float(np.max(np.abs(u15 - u19)))

# V3: 정리 2. 1차원, 두 데이터 점 q = {-1, +1} 가중 0.3/0.7, OT 경로(σ_min=0.01)
xs = np.linspace(-6, 6, 2401)
ts = np.linspace(0.0, 1.0, 401)
ws = np.array([0.3, 0.7])
sites = np.array([-1.0, 1.0])

def comp(k, tau):
    mu_k = tau * sites[k]
    sd = 1 - (1 - sig_min) * tau
    return mu_k, sd

def gauss(x, m, sd):
    return np.exp(-0.5 * ((x[None, :] - m[:, None]) / sd[:, None]) ** 2) / (np.sqrt(2 * np.pi) * sd[:, None])

def gap(vf):
    L_fm = 0.0
    L_cfm = 0.0
    dt = ts[1] - ts[0]
    dx = xs[1] - xs[0]
    for tau in ts:
        mu_k = np.array([comp(k, tau)[0] for k in range(2)])
        sd = np.array([comp(k, tau)[1] for k in range(2)])
        dens_k = gauss(xs, mu_k, sd)  # 2 x Nx
        pk = ws[:, None] * dens_k      # q_k * p_t(x|x_k)
        pt = pk.sum(0)
        # 조건부 벡터장 u(x|x_k) = (x_k - (1-σ)x)/(1-(1-σ)τ) 을 x에서 평가
        ucond = (sites[:, None] - (1 - sig_min) * xs[None, :]) / (1 - (1 - sig_min) * tau)
        umarg = (pk * ucond).sum(0) / np.maximum(pt, 1e-300)
        v = vf(xs, tau)
        L_fm += np.sum((v - umarg) ** 2 * pt) * dx * dt
        L_cfm += np.sum(pk * (v[None, :] - ucond) ** 2) * dx * dt
    return L_fm - L_cfm

vfs = {
    "zero": lambda x, t_: np.zeros_like(x),
    "linear": lambda x, t_: 0.3 * x - 0.5 * t_,
    "nonlin": lambda x, t_: np.tanh(x + t_) - 0.2 * x * t_,
}
gaps = {name: gap(f) for name, f in vfs.items()}
res["V3_gap_L_FM_minus_L_CFM"] = {k: float(v) for k, v in gaps.items()}
res["V3_spread_of_gap"] = float(max(gaps.values()) - min(gaps.values()))
OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=2))
