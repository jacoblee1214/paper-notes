"""토이 T2: 작은 문맥형 연속 행동 과제에서 PPO(혼합 정책 클리핑)와 MAPO(식 7, 성분별 클리핑)를 비교한다.

목적: 논문의 두 핵심 주장을 장난감 규모에서 시험한다.
  (1) MoE 정책을 PPO로 학습할 때 d_w·d_mu 교차항(식 6)이 사소하지 않은 크기로 나타나고 전문가 수에 따라 달라지는가 (그림 2의 장난감 판).
  (2) 같은 하이퍼파라미터에서 MAPO가 PPO보다 높은 보상에 도달하는가.
이 토이는 휴머노이드 추종이 아니다. 시뮬레이터, 시간 의존성, 적응 학습률, 공유 인코더가 없다. 논문의 수치를 재현하지 않는다.

과제: 상태 s ∈ R^8, 행동 a ∈ R^4. 상태는 고정된 6개 방향 중 가장 많이 겹치는 방향(모드)에 따라 서로 다른 선형 목표 a*(s)를 갖는다.
  보상 r = −‖a − a*(s)‖². 한 단계짜리 문맥형 밴딧이라 감마·GAE가 없고, 이점은 r − V(s) (V는 작은 MLP).
정책: 소프트맥스 라우터(8→32→N) 와 전문가 N개(8→32→4)의 혼합 평균, 상태 비의존 학습 가능한 σ. 식 (7)의 두 비율을 그대로 쓴다.
방법: ppo, mapo, mapo_avg(두 손실을 1/2배, 논문의 Averaged-MAPO)
설정: 반복 150회, 반복당 상태 2048개, 미니배치 4개 x 에폭 4, Adam 3e-3, ε = 0.2, 엔트로피 계수 0.005, 시드 4개.
실행: python3 analysis/toy_moe_ppo.py   (torch CPU, 약 1–2분)
결과: analysis/toy_moe_ppo_results.json
(clip_fraction_a: ppo는 혼합 정책 비율, mapo는 라우터 비율. clip_fraction_b: mapo의 전문가 비율. 학습 전체 평균)
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)
OUT = Path(__file__).resolve().parent / "toy_moe_ppo_results.json"
SD, AD, H, MODES = 8, 4, 32, 6
EPS, ENT, LR = 0.2, 0.005, 3e-3
ITERS, BATCH, MB, EPOCHS = 150, 2048, 4, 4


class Task:
    def __init__(self, seed):
        g = torch.Generator().manual_seed(1000 + seed)
        self.u = torch.nn.functional.normalize(torch.randn(MODES, SD, generator=g), dim=-1)
        self.A = torch.randn(MODES, AD, SD, generator=g) * 0.6
        self.c = torch.randn(MODES, AD, generator=g) * 0.8

    def target(self, s):
        mode = (s @ self.u.T).argmax(-1)
        return torch.einsum("bij,bj->bi", self.A[mode], s) + self.c[mode]


class Policy(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.r1, self.r2 = torch.nn.Linear(SD, H), torch.nn.Linear(H, n)
        self.W1 = torch.nn.Parameter(torch.randn(n, SD, H) * 0.3)
        self.b1 = torch.nn.Parameter(torch.zeros(n, 1, H))
        self.W2 = torch.nn.Parameter(torch.randn(n, H, AD) * 0.2)
        self.b2 = torch.nn.Parameter(torch.zeros(n, 1, AD))
        self.log_std = torch.nn.Parameter(torch.full((AD,), math.log(0.7)))
        self.v1, self.v2 = torch.nn.Linear(SD, 64), torch.nn.Linear(64, 1)

    def parts(self, s):
        w = torch.softmax(self.r2(torch.nn.functional.elu(self.r1(s))), -1)             # B x N
        h = torch.nn.functional.elu(torch.einsum("bs,nsh->nbh", s, self.W1) + self.b1)
        mu = (torch.einsum("nbh,nha->nba", h, self.W2) + self.b2).permute(1, 0, 2)      # B x N x A
        return w, mu

    def value(self, s):
        return self.v2(torch.nn.functional.elu(self.v1(s))).squeeze(-1)

    def router_params(self):
        return list(self.r1.parameters()) + list(self.r2.parameters())


def mix(w, mu):
    return (w.unsqueeze(-1) * mu).sum(1)


def logp(a, m, std):
    return (-((a - m) ** 2) / (2 * std ** 2)).sum(-1)


def clip_active(r, adv):
    """클리핑 때문에 기울기가 0이 되는 표본 비율"""
    return float((torch.clamp(r, 1 - EPS, 1 + EPS) * adv < r * adv).float().mean())


def clipped(r, adv):
    return torch.minimum(r * adv, torch.clamp(r, 1 - EPS, 1 + EPS) * adv)


def run(method, n, seed, task_seed=0, measure=False):
    torch.manual_seed(seed)
    task = Task(task_seed)
    pol = Policy(n)
    opt = torch.optim.Adam(pol.parameters(), lr=LR)
    curve, over, under, kl_meas = [], [], [], []
    cf_a, cf_b = [], []
    thr = EPS ** 2
    for it in range(ITERS):
        s = torch.randn(BATCH, SD)
        with torch.no_grad():
            w_old, mu_old = pol.parts(s)
            m_old = mix(w_old, mu_old)
            std = pol.log_std.exp()
            a = m_old + std * torch.randn_like(m_old)
            rew = -((a - task.target(s)) ** 2).sum(-1)
            lp_old = logp(a, m_old, std)
            adv = rew - pol.value(s)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ret = rew
            err = ((m_old - task.target(s)) ** 2).sum(-1).mean().item()
        curve.append(err)
        for ep in range(EPOCHS):
            perm = torch.randperm(BATCH)
            for idx in perm.chunk(MB):
                sb = s[idx]
                w, mu = pol.parts(sb)
                std = pol.log_std.exp()
                if method == "ppo":
                    r = torch.exp(logp(a[idx], mix(w, mu), std) - lp_old[idx])
                    pg = -clipped(r, adv[idx]).mean()
                    cf_a.append(clip_active(r.detach(), adv[idx]))
                else:  # mapo / mapo_avg: 식 (7). 라우터는 r1, 전문가는 r2로만 기울기를 받는다
                    r_router = torch.exp(logp(a[idx], mix(w, mu_old[idx]), std) - lp_old[idx])
                    r_expert = torch.exp(logp(a[idx], mix(w_old[idx], mu), std) - lp_old[idx])
                    pg = -(clipped(r_router, adv[idx]).mean() + clipped(r_expert, adv[idx]).mean())
                    cf_a.append(clip_active(r_router.detach(), adv[idx]))
                    cf_b.append(clip_active(r_expert.detach(), adv[idx]))
                    if method == "mapo_avg":   # 논문 5절의 Averaged-MAPO: 두 손실에 1/2을 곱해 σ의 유효 엔트로피 계수를 유지
                        pg = 0.5 * pg
                ent = pol.log_std.sum()
                vloss = ((pol.value(sb) - ret[idx]) ** 2).mean()
                loss = pg - ENT * ent + 0.5 * vloss * 0.01
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
                opt.step()
        if measure:
            with torch.no_grad():
                w_new, mu_new = pol.parts(s)
                std = pol.log_std.exp()
                m_p = mix(w_old, mu_new)                      # 전문가 먼저 경로의 중간 정책
                d_mu = m_old - m_p
                d_w = m_p - mix(w_new, mu_new)
                cross = (d_w * d_mu).sum(-1) / std.pow(2).mean()   # σ 비등방성은 평균으로 근사
                over.append(float((cross > thr).float().mean()))
                under.append(float((cross < -thr).float().mean()))
                kl_meas.append(float((((m_old - mix(w_new, mu_new)) ** 2).sum(-1) / (2 * std.pow(2).mean())).mean()))
    return {"err_curve": curve, "over": over, "under": under, "kl": kl_meas,
            "clip_a": float(np.mean(cf_a)), "clip_b": float(np.mean(cf_b)) if cf_b else None}


def summarize(runs, last=20):
    final = [float(np.mean(r["err_curve"][-last:])) for r in runs]
    return final


def main():
    t0 = time.time()
    res = {"config": {"iters": ITERS, "batch": BATCH, "minibatches": MB, "epochs": EPOCHS, "lr": LR, "eps": EPS,
                      "modes": MODES, "obs_dim": SD, "act_dim": AD}}
    # (2) 같은 설정에서 PPO 대 MAPO, 전문가 6개, 시드 4개 (같은 과제, 다른 초기화·표본)
    seeds = [0, 1, 2, 3]
    perf = {}
    for method in ("ppo", "mapo", "mapo_avg"):
        runs = [run(method, 6, sd) for sd in seeds]
        perf[method] = {"final_err_per_seed": [round(x, 4) for x in summarize(runs)],
                        "final_err_mean": round(float(np.mean(summarize(runs))), 4),
                        "clip_fraction_a": round(float(np.mean([r["clip_a"] for r in runs])), 4),
                        "clip_fraction_b": None if runs[0]["clip_b"] is None else round(float(np.mean([r["clip_b"] for r in runs])), 4),
                        "curve_mean": [round(float(np.mean([r["err_curve"][i] for r in runs])), 4) for i in range(0, ITERS, 10)]}
    res["perf_N6"] = perf
    print("perf", json.dumps({k: (v["final_err_per_seed"], v["final_err_mean"]) for k, v in perf.items()}), round(time.time() - t0, 1), "s", flush=True)
    # 단일 정책(전문가 1개, 라우터 무의미) 기준선
    mono = [run("ppo", 1, sd) for sd in seeds[:2]]
    res["perf_N1_ppo"] = {"final_err_per_seed": [round(x, 4) for x in summarize(mono)],
                          "final_err_mean": round(float(np.mean(summarize(mono))), 4)}
    # (1) 교차항 크기: PPO 학습 중 전문가 수별 비율
    mis = {}
    for n in (2, 4, 8, 16):
        runs = [run("ppo", n, sd, measure=True) for sd in (0, 1)]
        mis[str(n)] = {"over_mean": round(float(np.mean([np.mean(r["over"][10:]) for r in runs])), 4),
                       "under_mean": round(float(np.mean([np.mean(r["under"][10:]) for r in runs])), 4),
                       "kl_mean": round(float(np.mean([np.mean(r["kl"][10:]) for r in runs])), 4)}
    res["misestimation_ppo_by_N"] = mis
    res["misestimation_note"] = ("반복마다 20회 갱신(4 미니배치 x 4 에폭)이 끝난 뒤 같은 상태 배치에서 교차항 d_w·d_mu/σ² > ε²(과대) 또는 < −ε²(과소)인 상태 비율. "
                                 "클리핑이 있는 PPO 학습 중 값이며 논문 그림 2(바닐라 정책 기울기)와 조건이 다르다. 초기 10회 반복은 제외.")
    res["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "perf_N6"}, indent=1, ensure_ascii=False))
    print(json.dumps(perf, ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()
