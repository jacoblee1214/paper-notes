"""FlowDPG 중심 주장의 작은 CPU 토이. 행동 2차원, 상태 1차원, 해석적 크리틱 Q(s,a) = -||a - c(s)||^2.

검사 (analysis/03_analysis.md §4):
  K1 식 (11) 이후의 정상점: 증류 + BC + 일관성 항의 속도 이동량 s(t) 닫힌 형태를 수치로 확인
  K2 근사 (i): ODE 민감도 Phi(1,t) 가 항등행렬에서 얼마나 먼가 (BC로 학습한 망, 속도장의 야코비안)
  K3 FlowDPG 대 ODE 역전파 DPG(BPTT) 대 BC: 생성 행동의 Q 이득과 데모 중심에서의 거리
  K4 크리틱 기울기에 잡음이 있을 때 적응 이동(정규화)과 원시 이동(raw alpha*g)의 차이

실행: python3 analysis/toy_dpg.py   (torch 필요, CPU 단일 스레드, 약 1–2분)  -> analysis/toy_dpg_results.json
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
OUT = Path(__file__).with_name("toy_dpg_results.json")
D = 2
SIG = 0.3          # 데모 표준편차
T_STEPS = 12       # 추론 오일러 단계 수 (v1 부록 표 3의 T = 12)
ALPHA, LAM_MAX, MU, EPS = 0.5, 0.5, 1.0, 1e-6   # v1 부록 표 3
t0 = time.time()


def center(s):
    """크리틱이 가리키는 목표 행동 c(s). 데모 중심(0,0)에서 약 2 sigma 떨어져 있다."""
    return torch.stack([0.5 + 0 * s, 0.5 * torch.sin(3 * s)], dim=-1)


def q_fn(s, a):
    return -((a - center(s)) ** 2).sum(-1)


def make_net():
    return torch.nn.Sequential(
        torch.nn.Linear(D + 2, 128), torch.nn.SiLU(),
        torch.nn.Linear(128, 128), torch.nn.SiLU(),
        torch.nn.Linear(128, D),
    )


def vel(net, x, t, s):
    return net(torch.cat([x, t, s], -1))


def sample(net, s, x0, steps=T_STEPS, grad=False):
    x = x0
    dt = 1.0 / steps
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for i in range(steps):
            t = torch.full_like(s, i * dt)
            x = x + dt * vel(net, x, t, s)
    return x


def demo_batch(n):
    s = torch.rand(n, 1) * 2 - 1
    x1 = SIG * torch.randn(n, D)
    return s, x1


def fm_parts(n):
    s, x1 = demo_batch(n)
    x0 = torch.randn(n, D)
    t = torch.rand(n, 1)
    xt = (1 - t) * x0 + t * x1
    return s, x0, x1, t, xt, x1 - x0


def train_bc(net, steps=1000, lr=2e-3, bs=256):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        s, x0, x1, t, xt, u = fm_parts(bs)
        loss = ((vel(net, xt, t, s) - u) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


def train_flowdpg(net, steps=1200, warm=300, lr=1e-3, bs=256, alpha=ALPHA, normalize=True, g_noise=0.0, cons=True):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for k in range(1, steps + 1):
        s, x0, x1, t, xt, u = fm_parts(bs)
        v = vel(net, xt, t, s)
        ahat = (xt + (1 - t) * v).detach()                         # 식 (4), no grad
        a_req = ahat.clone().requires_grad_(True)
        g = torch.autograd.grad(q_fn(s, a_req).sum(), a_req)[0]     # 식 (5)
        if g_noise > 0:
            g = g + g_noise * g.norm(dim=-1, keepdim=True) * torch.randn_like(g) / np.sqrt(D)
        if normalize:
            delta = alpha * u.norm(dim=-1, keepdim=True) / (g.norm(dim=-1, keepdim=True) + EPS) * g   # 식 (6)
        else:
            delta = alpha * g
        ustar = ahat + delta - x0                                   # 식 (7)
        lam = LAM_MAX * min(k / warm, 1.0)
        l_distill = ((v - ustar) ** 2).sum(-1)
        l_bc = ((v - u) ** 2).sum(-1)
        l_cons = ((1 - t[:, 0]) ** 2) * ((v - u) ** 2).sum(-1)      # = ||ahat - x1||^2 (식 (9)와 동치)
        loss = (lam * l_distill + (1 - lam) * l_bc + (MU * l_cons if cons else 0.0)).mean()   # 식 (8)
        opt.zero_grad()
        loss.backward()
        opt.step()


def train_bptt(net, eta, steps=1200, warm=300, lr=1e-3, bs=256):
    """바닐라 DPG: BC + eta * (-Q(s, ODE 끝점)) 를 12단계 오일러를 통해 역전파."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for k in range(1, steps + 1):
        s, x0, x1, t, xt, u = fm_parts(bs)
        l_bc = ((vel(net, xt, t, s) - u) ** 2).sum(-1).mean()
        xe = sample(net, s, torch.randn(bs, D), grad=True)
        l_q = -q_fn(s, xe).mean()
        e = eta * min(k / warm, 1.0)
        loss = l_bc + e * l_q
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def evaluate(net, n=8192, seed=123):
    gen = torch.Generator().manual_seed(seed)
    s = torch.rand(n, 1, generator=gen) * 2 - 1
    x0 = torch.randn(n, D, generator=gen)
    a = sample(net, s, x0)
    q = q_fn(s, a)
    dist_demo = a.norm(dim=-1)                       # 데모 중심 (0,0)에서의 거리
    return {
        "Q_mean": float(q.mean()),
        "dist_to_c_mean": float((a - center(s)).norm(dim=-1).mean()),
        "dist_from_demo_center": float(dist_demo.mean()),
        "frac_beyond_3sigma_of_demo": float((dist_demo > 3 * SIG * np.sqrt(2)).float().mean()),
        "spread_std": float(a.std(0).mean()),
    }


# ---- K1: 정상점 닫힌 형태. 스칼라 한 점에서 목표를 고정해 두고 v를 경사하강
def k1_stationary():
    out = []
    rng = np.random.default_rng(0)
    for t in (0.05, 0.25, 0.5, 0.75, 0.95):
        x0, x1 = rng.normal(size=D), SIG * rng.normal(size=D)
        u = x1 - x0
        xt = (1 - t) * x0 + t * x1
        ghat = np.array([1.0, 0.0])
        delta = ALPHA * np.linalg.norm(u) * ghat
        lam = LAM_MAX
        v = u.copy()
        for _ in range(20000):                    # 목표(v 분리)를 매번 갱신하는 고정점 반복
            vbar = v.copy()
            ahat = xt + (1 - t) * vbar
            ustar = ahat + delta - x0
            grad = 2 * lam * (v - ustar) + 2 * (1 - lam) * (v - u) + 2 * MU * (1 - t) ** 2 * (v - u)
            v = v - 0.02 * grad
        shift_num = float(np.dot(v - u, ghat) / np.linalg.norm(delta))     # 단위: ||Delta|| 배
        shift_formula = float(lam / (lam * t + (1 - lam) + MU * (1 - t) ** 2))
        out.append({"t": t, "shift_over_delta_numeric": round(shift_num, 5), "shift_over_delta_formula": round(shift_formula, 5)})
    return out


# ---- K2: Phi(1,t) 대 I
def k2_phi(net):
    out = []
    for t in (0.0, 0.25, 0.5, 0.75, 0.9):
        steps = max(1, int(np.ceil(T_STEPS * (1 - t))))
        dt = (1 - t) / steps
        fro, scale = [], []
        for sv in (-0.8, 0.0, 0.8):
            for _ in range(6):
                x = (1 - t) * torch.randn(1, D) + t * SIG * torch.randn(1, D)
                s = torch.tensor([[sv]])

                def flow(xin):
                    xc = xin
                    for i in range(steps):
                        tt = torch.full((1, 1), t + i * dt)
                        xc = xc + dt * vel(net, xc, tt, s)
                    return xc
                J = torch.autograd.functional.jacobian(flow, x).reshape(D, D)
                fro.append(float(torch.linalg.matrix_norm(J - torch.eye(D)) / np.sqrt(D)))
                scale.append(float(torch.linalg.svdvals(J).mean()))
        out.append({"t": t, "rel_dev_from_I": round(float(np.mean(fro)), 3), "mean_singular_value": round(float(np.mean(scale)), 3)})
    return out


def main():
    res = {}
    res["K1_stationary_shift"] = k1_stationary()
    # 같은 시드로 BC 사전학습을 공유
    seeds = [0, 1, 2]
    rows = {"BC": [], "FlowDPG": [], "FlowDPG_nocons": [], "BPTT_eta1": [], "BPTT_eta3": []}
    k4 = {"FlowDPG_normalized": {}, "raw_alpha_g": {}}
    phi = None
    for sd in seeds:
        torch.manual_seed(sd)
        base = make_net()
        train_bc(base)
        sd_base = {k: v.clone() for k, v in base.state_dict().items()}
        rows["BC"].append(evaluate(base))
        if sd == 0:
            phi = k2_phi(base)

        def fresh():
            n = make_net()
            n.load_state_dict(sd_base)
            return n
        n = fresh(); torch.manual_seed(100 + sd); train_flowdpg(n); rows["FlowDPG"].append(evaluate(n))
        n = fresh(); torch.manual_seed(100 + sd); train_flowdpg(n, cons=False); rows["FlowDPG_nocons"].append(evaluate(n))
        for eta, key in ((1.0, "BPTT_eta1"), (3.0, "BPTT_eta3")):
            n = fresh(); torch.manual_seed(100 + sd); train_bptt(n, eta); rows[key].append(evaluate(n))
        if sd == 0:
            for gn in (0.0, 3.0):
                for key, norm in (("FlowDPG_normalized", True), ("raw_alpha_g", False)):
                    n = fresh(); torch.manual_seed(100 + sd); train_flowdpg(n, normalize=norm, g_noise=gn)
                    k4[key][f"g_noise_{gn}"] = evaluate(n)
        print(f"seed {sd} done {time.time() - t0:.0f}s", flush=True)
    res["K2_phi_vs_identity"] = phi

    def agg(lst):
        keys = lst[0].keys()
        return {k: {"mean": round(float(np.mean([r[k] for r in lst])), 4), "sd": round(float(np.std([r[k] for r in lst], ddof=1)), 4)} for k in keys}
    res["K3_methods_3seeds"] = {k: agg(v) for k, v in rows.items()}
    res["K4_noisy_critic_gradient_seed0"] = k4
    res["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
