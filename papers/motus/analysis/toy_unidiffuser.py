"""논문 3절·4.1절의 주장 검증용 CPU 토이: 독립 타임스텝(UniDiffuser형) 스케줄러 하나로 여러 조건부 모드가 나오는가.

설정: (o, a)가 상관 rho인 2차원 가우시안. 각 모달리티에 독립 rectified-flow 시간 tau_o, tau_a ~ U(0,1).
  - 'indep': 학습 때 tau_o, tau_a를 독립 샘플 (논문 Alg. 1 방식)
  - 'shared': tau_o = tau_a로 묶은 대조군 (결합 모드만 학습되는 경우)
모드별 검사 (해석적 정답 있음):
  - joint   : tau_o, tau_a를 함께 1→0    → 결합 분포 corr = rho
  - IDM     : o를 깨끗하게(tau_o=0) 고정, a만 노이즈에서 복원 → a|o ~ N(rho*o, 1-rho^2)
  - WM      : a를 깨끗하게(tau_a=0), o만 복원 → o|a ~ N(rho*a, 1-rho^2)
  - marginal: a를 노이즈(tau_a=1) 유지, o만 복원 → o ~ N(0,1)
실행: OMP_NUM_THREADS=2 python3 motus/analysis/toy_unidiffuser.py  (한 프로세스, 수십 초)
출력: analysis/toy_unidiffuser_results.json
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)
HERE = Path(__file__).resolve().parent
RHO = 0.8
SEEDS = [0, 1, 2]
STEPS = 3000
BATCH = 512
INTEG = 50


def sample_joint(n, g):
    x = torch.randn(n, 2, generator=g)
    o = x[:, 0]
    a = RHO * o + np.sqrt(1 - RHO ** 2) * x[:, 1]
    return o, a


class Net(nn.Module):
    """입력 6차원: (x_o, x_a, tau_o, tau_a, 0, 0), 출력 2차원: (v_o, v_a)."""

    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(6, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))


def train(mode, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for _ in range(STEPS):
        o0, a0 = sample_joint(BATCH, g)
        if mode == "indep":
            to = torch.rand(BATCH, generator=g)
            ta = torch.rand(BATCH, generator=g)
        else:
            to = torch.rand(BATCH, generator=g)
            ta = to.clone()
        eo, ea = torch.randn(BATCH, generator=g), torch.randn(BATCH, generator=g)
        xo = (1 - to) * o0 + to * eo
        xa = (1 - ta) * a0 + ta * ea
        vo_t, va_t = eo - o0, ea - a0
        vo, va = forward(net, xo, xa, to, ta)
        loss = ((vo - vo_t) ** 2).mean() + ((va - va_t) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def forward(net, xo, xa, to, ta):
    h = torch.stack([xo, xa, to, ta, torch.zeros_like(to), torch.zeros_like(to)], 1)
    v = net.f(h)
    return v[:, 0], v[:, 1]


@torch.no_grad()
def integrate(net, xo, xa, to_fixed, ta_fixed, free_o, free_a, n):
    """to_fixed/ta_fixed: 고정된 모달리티의 시간 (0이면 깨끗한 값을 그대로 둔다)."""
    xo = xo.clone()
    xa = xa.clone()
    ts = torch.linspace(1, 0, INTEG + 1)
    for k in range(INTEG):
        t_cur, t_nxt = ts[k], ts[k + 1]
        dt = t_nxt - t_cur
        to = torch.full((n,), float(t_cur)) if free_o else torch.full((n,), float(to_fixed))
        ta = torch.full((n,), float(t_cur)) if free_a else torch.full((n,), float(ta_fixed))
        vo, va = forward(net, xo, xa, to, ta)
        if free_o:
            xo = xo + dt * vo
        if free_a:
            xa = xa + dt * va
    return xo, xa


def evaluate(net, seed):
    g = torch.Generator().manual_seed(1000 + seed)
    n = 4000
    res = {}
    # joint
    xo, xa = integrate(net, torch.randn(n, generator=g), torch.randn(n, generator=g), 1.0, 1.0, True, True, n)
    res["joint_corr"] = float(np.corrcoef(xo.numpy(), xa.numpy())[0, 1])
    res["joint_corr_true"] = RHO
    # IDM: o clean (tau_o=0), a free. Condition on o drawn from N(0,1).
    o_clean = torch.randn(n, generator=g)
    _, a_idm = integrate(net, o_clean, torch.randn(n, generator=g), 0.0, 1.0, False, True, n)
    resid = a_idm - RHO * o_clean
    res["idm_resid_mean"] = float(resid.mean())
    res["idm_resid_var"] = float(resid.var())
    res["idm_resid_var_true"] = 1 - RHO ** 2
    # WM: a clean (tau_a=0), o free.
    a_clean = torch.randn(n, generator=g)
    o_wm, _ = integrate(net, torch.randn(n, generator=g), a_clean, 1.0, 0.0, True, False, n)
    resid = o_wm - RHO * a_clean
    res["wm_resid_mean"] = float(resid.mean())
    res["wm_resid_var"] = float(resid.var())
    res["wm_resid_var_true"] = 1 - RHO ** 2
    # marginal: a noisy (tau_a=1) fixed as noise, o free.
    o_marg, _ = integrate(net, torch.randn(n, generator=g), torch.randn(n, generator=g), 1.0, 1.0, True, False, n)
    res["marg_o_mean"] = float(o_marg.mean())
    res["marg_o_var"] = float(o_marg.var())
    return res


def summarize(rs):
    keys = rs[0].keys()
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rs])
        out[k] = {"mean": round(float(vals.mean()), 4), "sd_over_seeds": round(float(vals.std(ddof=1)), 4)}
    return out


if __name__ == "__main__":
    result = {"rho": RHO, "steps": STEPS, "batch": BATCH, "integration_steps": INTEG, "seeds": SEEDS}
    for mode in ["indep", "shared"]:
        rs = []
        for s in SEEDS:
            net = train(mode, s)
            rs.append(evaluate(net, s))
        result[mode] = summarize(rs)
        print(mode, json.dumps(result[mode], indent=1))
    with open(HERE / "toy_unidiffuser_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
