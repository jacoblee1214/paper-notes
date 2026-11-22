"""토이: 상태 조건부 소스 분포(LeaP식)가 조건부 행동 분포 근사에 주는 효과를 CPU에서 확인한다.

논문 §5.2의 핵심 주장("상태 적응형 분산이 핵심, 고정 분산은 효과 없음")을 가장 작은 설정으로 다시 본다.
  - 상태 s in [-1,1]^2, 행동 a in R^8 (논문의 청크 H*da를 줄인 것). 조건부 평균 f(s)는 고정 무작위 MLP.
  - 설정 A(단봉, 이분산): a = f(s) + sigma*(s) * eps,  sigma*(s) = 0.03 + 0.35 * sigmoid(4 s1)
  - 설정 B(쌍봉):          a = f(s) + 0.4 r m(s) u + 0.05 eps,  r = +-1 (같은 확률)
  - 설정 C(청크):          H=4 스텝 x 2차원 궤적 a_h = p(s) + (h/3 - 1/2) v(s) + 0.02 eps, |v(s)|가 상태에 따라 0.05-0.95.
                           소스는 논문처럼 (mu, sigma)를 H 스텝에 공유하고 스텝마다 독립 잡음을 뽑는다(식 2-3).
  - 생성기: 같은 MLP(은닉 128 x 3) 흐름 매칭, 조건은 s, 오일러 NFE=6 (논문 표 3과 같은 6)
  - 소스 변형: N(0,I) / mu(s)+고정 sigma=1 / mu(s)+고정 sigma=0.1 / mu(s)+고정 sigma=0.25 /
               mu(s)만(결정적) / LeaP식 mu(s), sigma(s)를 NLL+흐름 손실로 함께 학습
  - 지표: 상태별 에너지 거리(생성 64개 대 참 64개, 작을수록 좋음), 표본 평균 RMSE(f(s) 대비)
추정이 아니라 이 토이에서 나온 값이며, 로봇 정책 성공률과 직접 대응하지 않는다. 시드 2개, 약 1-2분.
실행: python3 toy_source_prior.py
"""
import json
import time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)
D, DS = 8, 2
STEPS, BATCH, HID = 2000, 256, 128
NFE_LIST = (1, 6)
VARIANTS = ["N(0,I)", "mu+sigma=1", "mu+sigma=0.1", "mu+sigma=0.25", "only mu", "LeaP (mu, sigma(s))"]


def make_world(setting, seed=123):
    g = torch.Generator().manual_seed(seed)
    W1 = torch.randn(DS, 16, generator=g)
    W2 = torch.randn(16, D, generator=g) / 4.0
    u = torch.randn(D, generator=g); u = u / u.norm()
    if setting == "C":
        Wp = torch.randn(16, 2, generator=g) / 4.0
        Wv = torch.randn(16, 2, generator=g)
        hh = (torch.arange(4).float() / 3 - 0.5)[None, :, None]
        def traj(s, noise=True):
            h = torch.tanh(s @ W1)
            p, v = h @ Wp, h @ Wv
            v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-6) * (0.05 + 0.9 * torch.sigmoid(4 * s[:, :1]))
            a = p[:, None, :] + hh * v[:, None, :]
            return a.reshape(s.shape[0], D)
        return (lambda s: traj(s)), (lambda s: traj(s) + 0.02 * torch.randn(s.shape[0], D))
    f = lambda s: torch.tanh(s @ W1) @ W2
    def sample(s):
        if setting == "A":
            sig = 0.03 + 0.35 * torch.sigmoid(4 * s[:, :1])
            return f(s) + sig * torch.randn(s.shape[0], D)
        r = (torch.randint(0, 2, (s.shape[0], 1)).float() * 2 - 1)
        m = 0.5 + 0.5 * (s[:, 1:2] + 1) / 2
        return f(s) + 0.4 * r * m * u + 0.05 * torch.randn(s.shape[0], D)
    return f, sample


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, HID), nn.SiLU(), nn.Linear(HID, HID), nn.SiLU(), nn.Linear(HID, HID), nn.SiLU(), nn.Linear(HID, o))


class Model(nn.Module):
    def __init__(self, variant, pd=D):
        super().__init__()
        self.v = variant
        self.pd = pd                      # 프라이어 출력 차원(설정 C는 2, 나머지는 D). pd < D면 스텝에 걸쳐 공유
        self.vel = mlp(D + 1 + DS, D)
        self.prior = mlp(DS, 2 * pd) if variant not in ("N(0,I)",) else None

    def source(self, s):
        """(z0, mu, logvar). 변형별 소스 분포에서 z0를 재매개화로 뽑는다."""
        if self.v == "N(0,I)":
            return torch.randn(s.shape[0], D), None, None
        out = self.prior(s)
        mu, lv = out[:, :self.pd], out[:, self.pd:]
        if self.pd < D:
            mu, lv = mu.repeat(1, D // self.pd), lv.repeat(1, D // self.pd)
        if self.v == "mu+sigma=1": lv = torch.zeros_like(lv)
        elif self.v == "mu+sigma=0.1": lv = torch.full_like(lv, 2 * np.log(0.1))
        elif self.v == "mu+sigma=0.25": lv = torch.full_like(lv, 2 * np.log(0.25))
        elif self.v == "only mu": lv = torch.full_like(lv, -40.0)      # sigma ~ 0 (결정적)
        else: lv = lv.clamp(-10, 5)
        return mu + torch.exp(0.5 * lv) * torch.randn_like(mu), mu, lv   # 잡음은 스텝마다 독립

    def v_field(self, x, t, s):
        return self.vel(torch.cat([x, t[:, None], s], 1))

    def loss(self, s, a):
        z0, mu, lv = self.source(s)
        t = torch.rand(s.shape[0])
        xt = (1 - t[:, None]) * z0 + t[:, None] * a
        l_flow = ((self.v_field(xt, t, s) - (a - z0)) ** 2).mean()
        l_nll = 0.0
        if mu is not None and self.v in ("mu+sigma=1", "mu+sigma=0.1", "mu+sigma=0.25", "only mu", "LeaP (mu, sigma(s))"):
            l_nll = 0.5 * (lv + (a - mu) ** 2 / lv.exp()).mean() if self.v == "LeaP (mu, sigma(s))" else 0.5 * ((a - mu) ** 2).mean()
        return l_flow + l_nll

    @torch.no_grad()
    def sample(self, s, nfe):
        x, _, _ = self.source(s)
        for k in range(nfe):
            t = torch.full((s.shape[0],), k / nfe)
            x = x + self.v_field(x, t, s) / nfe
        return x


def energy_distance(X, Y):
    """X, Y: (S, n, D) -> 상태별 에너지 거리 평균."""
    xy = torch.cdist(X, Y).mean((1, 2)); xx = torch.cdist(X, X).mean((1, 2)); yy = torch.cdist(Y, Y).mean((1, 2))
    return float((2 * xy - xx - yy).mean())


def run(setting, variant, seed):
    torch.manual_seed(seed)
    f, sample = make_world(setting)
    model = Model(variant, pd=2 if setting == "C" else D)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    for _ in range(STEPS):
        s = torch.rand(BATCH, DS) * 2 - 1
        loss = model.loss(s, sample(s))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    g = torch.Generator().manual_seed(999)
    S, n = 300, 64
    s = (torch.rand(S, DS, generator=g) * 2 - 1)
    sr = s[:, None, :].expand(S, n, DS).reshape(-1, DS)
    torch.manual_seed(1000 + seed)
    truth = sample(sr).reshape(S, n, D)
    res = {}
    for nfe in NFE_LIST:
        gen = model.sample(sr, nfe).reshape(S, n, D)
        res[f"nfe{nfe}"] = dict(energy=energy_distance(gen, truth),
                                mean_rmse=float(((gen.mean(1) - f(s)) ** 2).mean().sqrt()),
                                spread_ratio=float(gen.std(1).mean() / truth.std(1).mean()))
    with torch.no_grad():
        z0, _, _ = model.source(sr)
        res["source_to_target_dist"] = float((z0.reshape(S, n, D) - truth).norm(dim=-1).mean())
    return res


def main():
    t0 = time.time()
    out = {"config": dict(D=D, steps=STEPS, batch=BATCH, hidden=HID, seeds=[0, 1])}
    for setting in ("A", "B", "C"):
        out[setting] = {}
        for variant in VARIANTS:
            runs = [run(setting, variant, sd) for sd in (0, 1)]
            agg = {}
            for key in ("nfe1", "nfe6"):
                agg[key] = {m: round(float(np.mean([r[key][m] for r in runs])), 4) for m in ("energy", "mean_rmse", "spread_ratio")}
            agg["source_to_target_dist"] = round(float(np.mean([r["source_to_target_dist"] for r in runs])), 3)
            agg["energy_nfe6_per_seed"] = [round(r["nfe6"]["energy"], 4) for r in runs]
            out[setting][variant] = agg
            print(setting, variant, agg["nfe6"], agg["source_to_target_dist"], f"{time.time()-t0:.0f}s", flush=True)
    out["seconds"] = round(time.time() - t0)
    json.dump(out, open("toy_source_prior_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
