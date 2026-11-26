"""토이 T2: 작은 max-cut SDP(n=12 노드, 변수 78개)에서 COSMO식 ADMM을 풀어서 펼치고(deep unfolding),
반복마다 (rho_eq, rho_psd, alpha)를 학습한다 (torch CPU, float64, 약 1분).

논문 3절과 식 13의 가장 작은 판이다. 공식 코드는 쓰지 않고 논문 식 (3), (13)과 부록 D, E를 보고 새로 짰다.
  - 문제: min q^T x s.t. A x + s = b, s in {0}^n x PSD (svec), q = -svec(L/4), 식 (1), 부록 F.3
  - 정책: 개루프 일정(반복 k마다 로짓 3개). 논문은 잔차를 보는 폐루프 MLP이므로 이 토이는 더 약한 정책이다.
  - 손실: 식 13 (초기 잔차로 나눈 뒤 log, k=1..K 합), PSD 역전파는 DK 규칙(식 9–10)
  - 기준선: 기본값 (rho_eq=100, rho_psd=0.1, alpha=1.6), 격자 탐색한 상수, 잔차 균형 적응형(게이트 5)
  - 평가: 새 그래프 100개, 절대 허용오차 1e-3 (무한 노름, 원 문제 잔차)에 닿는 반복 수

실행: python3 analysis/toy_du_maxcut.py   → analysis/toy_du_maxcut_results.json
"""
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
OUT = Path(__file__).resolve().parent / "toy_du_maxcut_results.json"

NN = 12                       # 노드 수
NV = NN * (NN + 1) // 2       # svec 차원
K = 60                        # 펼침 길이
SIGMA = 1e-6
RHO_MIN, RHO_MAX = 1e-3, 1e3
TRIL = torch.tril_indices(NN, NN)
DIAG_IDX = torch.where(TRIL[0] == TRIL[1])[0]
SCALE = torch.where(TRIL[0] == TRIL[1], torch.tensor(1.0), torch.sqrt(torch.tensor(2.0)))


def svec(X):
    return X[..., TRIL[0], TRIL[1]] * SCALE


def smat(x):
    X = torch.zeros(*x.shape[:-1], NN, NN)
    X[..., TRIL[0], TRIL[1]] = x / SCALE
    return X + X.mT - torch.diag_embed(torch.diagonal(X, dim1=-2, dim2=-1))


class PsdDK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, S):
        lam, U = torch.linalg.eigh(S)
        ctx.save_for_backward(lam, U)
        return U @ torch.diag_embed(torch.clamp(lam, min=0)) @ U.mT

    @staticmethod
    def backward(ctx, G):
        lam, U = ctx.saved_tensors
        G = 0.5 * (G + G.mT)
        f = torch.clamp(lam, min=0)
        d = lam.unsqueeze(-2) - lam.unsqueeze(-1)
        same = d == 0
        L = torch.where(same, (lam.unsqueeze(-1) > 0).to(d.dtype),
                        (f.unsqueeze(-2) - f.unsqueeze(-1)) / torch.where(same, torch.ones_like(d), d))
        return U @ (L * (U.mT @ G @ U)) @ U.mT


def proj_autograd(S):
    lam, U = torch.linalg.eigh(S)
    return U @ torch.diag_embed(torch.clamp(lam, min=0)) @ U.mT


def make_graphs(B, seed):
    g = torch.Generator().manual_seed(seed)
    A = (torch.rand(B, NN, NN, generator=g) < 0.5).to(torch.float64)
    low = torch.tril(A, diagonal=-1)
    A = low + low.mT
    L = torch.diag_embed(A.sum(-1)) - A
    return -svec(0.25 * L)    # q: (B, NV)


# A = [A_eq; -I], b = [1; 0]: x 에 곱하는 연산을 직접 쓴다
def Ax(x):
    return torch.cat([x[..., DIAG_IDX], -x], dim=-1)


def ATy(y):
    ye, yp = y[..., :NN], y[..., NN:]
    out = -yp
    out = out.clone()
    out[..., DIAG_IDX] = out[..., DIAG_IDX] + ye
    return out


b_vec = torch.cat([torch.ones(NN), torch.zeros(NV)])
eq_mask = torch.cat([torch.ones(NN), torch.zeros(NV)])      # 영 원뿔 행


def unroll(q, rho_eq, rho_psd, alpha, n_iter, proj, residual_tol=None, adaptive=False):
    """rho_eq, rho_psd, alpha: (n_iter, B) 개루프 일정, 또는 adaptive=True 면 초기값 (B,)에서 잔차 균형으로 갱신.
    반환: 잔차 이력 (primal, dual) (n_iter+1, B), 수렴 반복 수 (residual_tol 가 주어지면)."""
    B = q.shape[0]
    x = torch.zeros(B, NV)
    s = torch.zeros(B, NN + NV)
    y = torch.zeros(B, NN + NV)
    prim_h, dual_h = [], []

    def resid(x, s, y):
        return (Ax(x) + s - b_vec), (q - ATy(y))

    p0, d0 = resid(x, s, y)
    prim_h.append(p0.pow(2).sum(-1))
    dual_h.append(d0.pow(2).sum(-1))
    conv_iter = torch.full((B,), -1)
    if adaptive:
        rho_p = rho_psd.clone()  # (B,) 초기값
        a_const = alpha
    for k in range(n_iter):
        if adaptive:
            re, rp, al = 1000.0 * rho_p, rho_p, a_const * torch.ones(B)
        else:
            re, rp, al = rho_eq[k], rho_psd[k], alpha[k]
        rho = torch.cat([re.unsqueeze(-1).expand(B, NN), rp.unsqueeze(-1).expand(B, NV)], dim=-1)
        xi = SIGMA * x - q + ATy(y + rho * (b_vec - s))
        # M = (sigma + rho_psd) I + rho_eq * A_eq^T A_eq (대각)
        Mdiag = SIGMA + rp.unsqueeze(-1).expand(B, NV).clone()
        Mdiag[..., DIAG_IDX] = Mdiag[..., DIAG_IDX] + re.unsqueeze(-1)
        x_t = xi / Mdiag
        z_t = b_vec - Ax(x_t)
        al_ = al.unsqueeze(-1)
        x_n = al_ * x_t + (1 - al_) * x
        s_rel = al_ * z_t + (1 - al_) * s
        v = s_rel + y / rho
        s_n = torch.cat([torch.zeros(B, NN), svec(proj(smat(v[..., NN:])))], dim=-1)
        y = y + rho * (s_rel - s_n)
        x, s = x_n, s_n
        p, d = resid(x, s, y)
        prim_h.append(p.pow(2).sum(-1))
        dual_h.append(d.pow(2).sum(-1))
        if residual_tol is not None:
            ok = (p.abs().amax(-1) <= residual_tol) & (d.abs().amax(-1) <= residual_tol) & (conv_iter < 0)
            conv_iter = torch.where(ok, torch.full_like(conv_iter, k + 1), conv_iter)
            if bool((conv_iter > 0).all()):
                break
        if adaptive:
            with torch.no_grad():
                pn = p.abs().amax(-1) / torch.maximum(torch.maximum(Ax(x).abs().amax(-1), s.abs().amax(-1)), b_vec.abs().amax().expand(B))
                dn = d.abs().amax(-1) / torch.maximum(q.abs().amax(-1), ATy(y).abs().amax(-1))
                new = rho_p * torch.sqrt(pn / (dn + 1e-12) + 1e-12)
                fire = (new > 5 * rho_p) | (new < rho_p / 5)
                rho_p = torch.where(fire, new.clamp(1e-6, 1e6), rho_p)
    return torch.stack(prim_h), torch.stack(dual_h), conv_iter


def logit_to_rho(l):
    return torch.exp(np.log(RHO_MIN) + (np.log(RHO_MAX) - np.log(RHO_MIN)) * torch.sigmoid(l))


def rho_to_logit(r):
    return torch.logit((torch.log(torch.tensor(r)) - np.log(RHO_MIN)) / (np.log(RHO_MAX) - np.log(RHO_MIN)))


def schedule(params, B):
    re = logit_to_rho(params["l_eq"]).unsqueeze(-1).expand(-1, B)
    rp = logit_to_rho(params["l_psd"]).unsqueeze(-1).expand(-1, B)
    al = (2 * torch.sigmoid(params["l_al"])).unsqueeze(-1).expand(-1, B)
    return re, rp, al


def loss_fn(q, params, proj):
    B = q.shape[0]
    re, rp, al = schedule(params, B)
    prim, dual, _ = unroll(q, re, rp, al, K, proj)
    # 식 13: 초기 잔차(분리, 분리된 분모)로 나눈 뒤 log, k=1..K
    t = prim[1:] / prim[:1].detach() + dual[1:] / dual[:1].detach()
    return torch.log(t + 1e-300).sum(0).mean()


def const_schedule(re0, rp0, al0, n, B):
    return (torch.full((n, B), re0), torch.full((n, B), rp0), torch.full((n, B), al0))


def median_iters(q, sched_fn, cap, adaptive_args=None):
    B = q.shape[0]
    with torch.no_grad():
        if adaptive_args is not None:
            _, _, it = unroll(q, None, torch.full((B,), adaptive_args[0]), adaptive_args[1], cap, proj_autograd,
                              residual_tol=1e-3, adaptive=True)
        else:
            re, rp, al = sched_fn(cap, B)
            _, _, it = unroll(q, re, rp, al, cap, proj_autograd, residual_tol=1e-3)
    ok = it > 0
    return it, int(ok.sum())


def summarize(it, n_ok, cap):
    it = it.clone()
    charged = torch.where(it > 0, it, torch.full_like(it, cap)).double()
    solved = it[it > 0].double()
    return {"solved": n_ok, "n": int(it.numel()),
            "median_charged": float(charged.median()),
            "median_solved_only": float(solved.median()) if solved.numel() else None,
            "q1_q3_solved_only": [float(solved.quantile(0.25)), float(solved.quantile(0.75))] if solved.numel() else None}


def main():
    t0 = time.time()
    cap = 3000
    grid_cap = 1000
    q_tr, q_val, q_te = make_graphs(100, 0), make_graphs(20, 1), make_graphs(100, 2)
    out = {"n_nodes": NN, "n_vars": NV, "K": K, "cap": cap, "tol_abs": 1e-3}

    # --- 기울기 점검: autograd(eigh)와 DK의 손실 기울기 (초기 일정)
    params0 = {"l_eq": rho_to_logit(100.0).repeat(K), "l_psd": rho_to_logit(0.1).repeat(K),
               "l_al": torch.logit(torch.tensor(0.8)).repeat(K)}
    g = {}
    for name, proj in (("dk", PsdDK.apply), ("autograd", proj_autograd)):
        p = {k: v.clone().requires_grad_(True) for k, v in params0.items()}
        loss_fn(q_tr[:5], p, proj).backward()
        g[name] = torch.cat([p[k].grad for k in ("l_eq", "l_psd", "l_al")])
    out["grad_check_initial"] = {"dk_finite": bool(torch.isfinite(g["dk"]).all()),
                                 "autograd_finite": bool(torch.isfinite(g["autograd"]).all()),
                                 "rel_diff": float((g["dk"] - g["autograd"]).norm() / g["dk"].norm())
                                 if torch.isfinite(g["autograd"]).all() else None}

    # --- 기준선
    base = {}
    it, ok = median_iters(q_te, lambda n, B: const_schedule(100.0, 0.1, 1.6, n, B), cap)
    base["default_const(100,0.1,1.6)"] = summarize(it, ok, cap)
    # 격자 탐색 (훈련 30개, 상한 1000, 실패는 상한으로 계산한 반복 수 중앙값)
    best, best_val = None, 1e18
    for re0, rp0, al0 in itertools.product([10, 100, 1000], [1e-2, 0.1, 1, 10], [1.0, 1.6, 1.9]):
        it, ok = median_iters(q_tr[:30], lambda n, B: const_schedule(re0, rp0, al0, n, B), grid_cap)
        val = float(torch.where(it > 0, it, torch.full_like(it, grid_cap)).double().median())
        if val < best_val:
            best_val, best = val, (re0, rp0, al0)
    it, ok = median_iters(q_te, lambda n, B: const_schedule(*best, n, B), cap)
    base["grid_tuned_const"] = {**summarize(it, ok, cap), "params(rho_eq,rho_psd,alpha)": list(best),
                                "train30_median_charged": best_val, "grid_size": 3 * 4 * 3, "grid_cap": grid_cap}
    it, ok = median_iters(q_te, None, cap, adaptive_args=(0.1, 1.6))
    base["adaptive_residual_balance_gate5"] = summarize(it, ok, cap)
    out["baselines_test"] = base

    # --- 학습
    params = {k: v.clone().requires_grad_(True) for k, v in params0.items()}
    params["l_al"].data = torch.logit(torch.tensor(0.8)).repeat(K)
    opt = torch.optim.Adam(params.values(), lr=3e-2)
    epochs, bs = 40, 5
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=3e-3)
    best_val_loss, best_params = 1e18, None
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(100, generator=torch.Generator().manual_seed(ep))
        tot = 0.0
        for i in range(0, 100, bs):
            opt.zero_grad()
            loss = loss_fn(q_tr[perm[i:i + bs]], params, PsdDK.apply)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params.values(), 1.0)
            opt.step()
            tot += float(loss.detach()) * bs / 100
        sched.step()
        with torch.no_grad():
            vl = float(loss_fn(q_val, params, PsdDK.apply))
        hist.append({"epoch": ep + 1, "train_loss": tot, "val_loss": vl})
        if vl < best_val_loss:
            best_val_loss, best_params = vl, {k: v.detach().clone() for k, v in params.items()}
    out["train_history_first_last"] = [hist[0], hist[-1]]
    out["best_val_loss"] = best_val_loss

    def learned_sched(n, B):
        re, rp, al = schedule(best_params, B)
        pad = n - K
        if pad > 0:  # K 이후에는 마지막 값을 유지
            re = torch.cat([re, re[-1:].expand(pad, B)])
            rp = torch.cat([rp, rp[-1:].expand(pad, B)])
            al = torch.cat([al, al[-1:].expand(pad, B)])
        return re, rp, al

    it, ok = median_iters(q_te, learned_sched, cap)
    out["learned_open_loop_test"] = summarize(it, ok, cap)
    re, rp, al = schedule(best_params, 1)
    out["learned_schedule_every_10"] = {"rho_eq": [float(x) for x in re[::10, 0]], "rho_psd": [float(x) for x in rp[::10, 0]],
                                        "alpha": [float(x) for x in al[::10, 0]]}
    # 학습 길이 K 이내 수렴 비율
    out["learned_converged_within_K"] = float(((it > 0) & (it <= K)).double().mean())
    # 학습 전 일정(기본값)을 같은 평가 틀에 넣은 값 (학습 효과 분리)
    out["seconds"] = time.time() - t0
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
