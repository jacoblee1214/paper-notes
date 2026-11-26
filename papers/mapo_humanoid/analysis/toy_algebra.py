"""토이 T1: MAPO 논문 4절 식과 부록 B 기울기 주장을 수치로 확인한다 (학습 없음, 수 초).

확인 항목
  a. 식 (3)–(6)의 항등식: KL_measured = KL_expert + KL_router + d_w·d_mu/σ² (순차 경로)
  b. 경로 의존성: 전문가 먼저 / 라우터 먼저 경로의 교차항 d_w·d_mu가 같은 갱신에서 부호까지 같은지
  c. MAPO의 두 비율(식 7)은 둘 다 옛 값에서 출발한다. 두 성분 변화의 합과 실제 평균 변화의 차(상호작용 항 Σ Δw_e Δμ_e)의 크기
  d. 갱신 직전(새 = 옛)에는 MAPO 기울기가 PPO 기울기와 정확히 같은지, 갱신 크기가 커지면 코사인이 어떻게 변하는지
  e. 부록 B.5 식 (25)–(26): 전문가 평균 기울기 블록이 w_old/w 배인지

모델: 가우시안 정책 N(Σ w_e μ_e(s), σ² I), 가중치는 소프트맥스 라우터, 전문가는 작은 MLP. 행동 차원 4, 상태 차원 8.
실행: python3 analysis/toy_algebra.py   (torch 필요, CPU, 수 초)
결과: analysis/toy_algebra_results.json
"""
import json
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
OUT = Path(__file__).resolve().parent / "toy_algebra_results.json"
RES = {}
SD, AD = 8, 4  # 상태, 행동 차원


class MoE(torch.nn.Module):
    def __init__(self, n, hid=16):
        super().__init__()
        self.router = torch.nn.Sequential(torch.nn.Linear(SD, hid), torch.nn.ELU(), torch.nn.Linear(hid, n))
        self.experts = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.Linear(SD, hid), torch.nn.ELU(), torch.nn.Linear(hid, AD)) for _ in range(n)])

    def parts(self, s):
        w = torch.softmax(self.router(s), -1)                 # B x N
        mu = torch.stack([e(s) for e in self.experts], 1)     # B x N x A
        return w, mu


def perturb(model, scale, gen, coherent=0.0):
    """모든 파라미터에 가우시안 잡음을 더한다. coherent > 0이면 전문가 파라미터 변화에 공통 성분을 섞는다."""
    with torch.no_grad():
        common = [torch.randn(p.shape, generator=gen) for p in model.experts[0].parameters()]
        for k, e in enumerate(model.experts):
            for i, p in enumerate(e.parameters()):
                p.add_(scale * ((1 - coherent) * torch.randn(p.shape, generator=gen) + coherent * common[i]))
        for p in model.router.parameters():
            p.add_(scale * torch.randn(p.shape, generator=gen))


def mean_of(w, mu):
    return (w.unsqueeze(-1) * mu).sum(1)


def check_a_b_c(n=6, scale=0.05, trials=200, seed=0):
    gen = torch.Generator().manual_seed(seed)
    s = torch.randn(512, SD, generator=gen)
    sigma = 0.5
    out = {"identity_max_abs_err": 0.0}
    sign_disagree, corr_x, corr_y, inter_ratio = 0, [], [], []
    inter_by_scale, kl_by_scale = {}, {}
    for sc in (0.01, 0.02, 0.05, 0.1):
        ratios, kls = [], []
        for t in range(trials // 4):
            old = MoE(n)
            new = MoE(n)
            new.load_state_dict(old.state_dict())
            perturb(new, sc, gen)
            with torch.no_grad():
                wo, muo = old.parts(s)
                wn, mun = new.parts(s)
                m_old, m_new = mean_of(wo, muo), mean_of(wn, mun)
                # 전문가 먼저 경로: π' 평균 = Σ w_old μ_new
                m_p = mean_of(wo, mun)
                d_mu = m_old - m_p
                d_w = m_p - m_new
                kl_meas = ((m_old - m_new) ** 2).sum(-1) / (2 * sigma ** 2)
                kls.append(float(kl_meas.mean()))
                kl_e = (d_mu ** 2).sum(-1) / (2 * sigma ** 2)
                kl_r = (d_w ** 2).sum(-1) / (2 * sigma ** 2)
                cross_e = (d_w * d_mu).sum(-1) / sigma ** 2
                out["identity_max_abs_err"] = max(out["identity_max_abs_err"], float((kl_meas - kl_e - kl_r - cross_e).abs().max()))
                # 라우터 먼저 경로: π'' 평균 = Σ w_new μ_old
                m_q = mean_of(wn, muo)
                d_w2 = m_old - m_q
                d_mu2 = m_q - m_new
                cross_r = (d_w2 * d_mu2).sum(-1) / sigma ** 2
                if sc == scale:
                    sign_disagree += int(((cross_e > 0) != (cross_r > 0)).sum())
                    corr_x.append(cross_e.numpy())
                    corr_y.append(cross_r.numpy())
                # MAPO 방식(둘 다 옛 값에서 출발)의 두 성분 변화와 상호작용 항
                c1 = m_q - m_old        # 라우터만 바꾼 평균 변화 (r_router의 분자)
                c2 = m_p - m_old        # 전문가만 바꾼 평균 변화 (r_expert의 분자)
                inter = (m_new - m_old) - c1 - c2
                ratios.append(float(inter.norm(dim=-1).mean() / (m_new - m_old).norm(dim=-1).mean()))
        inter_by_scale[str(sc)] = round(float(np.mean(ratios)), 4)
        kl_by_scale[str(sc)] = round(float(np.mean(kls)), 5)
    cx, cy = np.concatenate(corr_x), np.concatenate(corr_y)
    total = len(cx)
    out["path_dependence"] = {
        "n_states_total": total,
        "sign_disagree_fraction": round(sign_disagree / total, 4),
        "pearson_cross_expert_first_vs_router_first": round(float(np.corrcoef(cx, cy)[0, 1]), 4),
        "note": "교차항 부호가 경로(전문가 먼저/라우터 먼저)에 따라 달라지는 상태의 비율",
    }
    out["interaction_over_total_change_by_param_scale"] = inter_by_scale
    out["mean_measured_kl_by_param_scale"] = kl_by_scale   # PPO 신뢰 영역의 규모는 ε² = 0.04
    out["interaction_note"] = "MAPO의 두 성분 변화(옛 값 기준)의 합이 놓치는 상호작용 항 Σ Δw_e Δμ_e의 상대 크기. 갱신 크기에 거의 비례하면 2차 항이다."
    return out


def surrogate_grads(model, old_state, s, a, adv, eps, sigma, mapo, clip=True):
    """같은 샘플 (s, a)에서 PPO 또는 MAPO 대리 손실의 기울기를 반환한다."""
    old = MoE(model.router[-1].out_features)
    old.load_state_dict(old_state)
    with torch.no_grad():
        wo, muo = old.parts(s)
        m_old = mean_of(wo, muo)
    w, mu = model.parts(s)

    def logp(m):
        return -((a - m) ** 2).sum(-1) / (2 * sigma ** 2)

    lp_old = logp(m_old)
    if not mapo:
        r = torch.exp(logp(mean_of(w, mu)) - lp_old)
        loss = -(torch.minimum(r * adv, torch.clamp(r, 1 - eps, 1 + eps) * adv) if clip else r * adv).mean()
    else:
        r1 = torch.exp(logp(mean_of(w, muo)) - lp_old)    # 라우터만 새 값 (식 7, router)
        r2 = torch.exp(logp(mean_of(wo, mu)) - lp_old)    # 전문가만 새 값 (식 7, expert)
        f = lambda r: (torch.minimum(r * adv, torch.clamp(r, 1 - eps, 1 + eps) * adv) if clip else r * adv)
        loss = -(f(r1).mean() + f(r2).mean())
    model.zero_grad()
    loss.backward()
    gr = torch.cat([p.grad.flatten() for p in model.router.parameters()])
    ge = torch.cat([p.grad.flatten() for e in model.experts for p in e.parameters()])
    return gr.clone(), ge.clone()


def cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))


def check_d(n=6, seed=1):
    gen = torch.Generator().manual_seed(seed)
    sigma, eps = 0.5, 0.2
    s = torch.randn(1024, SD, generator=gen)
    rows = {}
    for sc in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1):
        res = {"router_cos": [], "expert_cos": [], "router_norm_ratio": [], "expert_norm_ratio": []}
        for t in range(10):
            model = MoE(n)
            old_state = {k: v.clone() for k, v in model.state_dict().items()}
            with torch.no_grad():
                wo, muo = model.parts(s)
                m_old = mean_of(wo, muo)
                a = m_old + sigma * torch.randn(s.shape[0], AD, generator=gen)
                adv = torch.randn(s.shape[0], generator=gen)
            perturb(model, sc, gen)
            grp, gep = surrogate_grads(model, old_state, s, a, adv, eps, sigma, mapo=False, clip=False)
            grm, gem = surrogate_grads(model, old_state, s, a, adv, eps, sigma, mapo=True, clip=False)
            res["router_cos"].append(cos(grp, grm))
            res["expert_cos"].append(cos(gep, gem))
            res["router_norm_ratio"].append(float(grm.norm() / grp.norm()))
            res["expert_norm_ratio"].append(float(gem.norm() / gep.norm()))
        rows[str(sc)] = {k: round(float(np.mean(v)), 5) for k, v in res.items()}
        rows[str(sc)]["router_cos_min"] = round(float(np.min(res["router_cos"])), 5)
        rows[str(sc)]["expert_cos_min"] = round(float(np.min(res["expert_cos"])), 5)
    return rows


def check_e(n=6, seed=2):
    """식 (25)–(26): ∇θk m = w_k ∂μ_k (PPO), w_old_k ∂μ_k (MAPO). 블록별 비율이 w_old_k / w_k인지 확인한다."""
    gen = torch.Generator().manual_seed(seed)
    s = torch.randn(1, SD, generator=gen)  # 한 상태에서 정확히 비교
    model = MoE(n)
    old_state = {k: v.clone() for k, v in model.state_dict().items()}
    old = MoE(n)
    old.load_state_dict(old_state)
    perturb(model, 0.1, gen)
    with torch.no_grad():
        wo, _ = old.parts(s)
    w, mu = model.parts(s)
    u = torch.randn(AD, generator=gen)       # 임의의 출력 방향 (∇m에 곱해 스칼라로 만듦)
    ratios = []
    for k, e in enumerate(model.experts):
        ps = list(e.parameters())
        g_ppo = torch.autograd.grad((w[0, k] * mu[0, k] * u).sum(), ps, retain_graph=True)
        g_mapo = torch.autograd.grad((wo[0, k] * mu[0, k] * u).sum(), ps, retain_graph=True)
        gp = torch.cat([g.flatten() for g in g_ppo])
        gm = torch.cat([g.flatten() for g in g_mapo])
        ratios.append({"k": k, "measured_ratio": round(float((gm.norm() / gp.norm())), 6),
                       "w_old_over_w": round(float((wo[0, k] / w[0, k]).detach()), 6), "cos": round(cos(gp, gm), 8)})
    return ratios


def main():
    RES["abc"] = check_a_b_c()
    RES["d_gradient_equivalence"] = check_d()
    RES["e_expert_block_scale"] = check_e()
    OUT.write_text(json.dumps(RES, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(RES, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
