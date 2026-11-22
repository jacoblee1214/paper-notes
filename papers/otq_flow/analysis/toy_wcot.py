"""장난감 검증: 가중 한계분포 COT-CFM(wCOT)과 에너지 가중 FM(EWFM)을 2차원 two-moons 목표 분포에서 비교한다.
논문 Fig. 1의 주장(같은 가중치에서 wCOT가 2-스텝 샘플링 품질이 더 좋고 경로가 곧다)을 아주 작게 점검한다.
조건(상태)은 없다: 조건부 비용 증강(식 3~4)은 이 장난감에서 검증하지 않는다.
실행: python3 toy_wcot.py   (torch CPU, 약 2~4분)  결과: toy_wcot_results.json"""
import json, math, time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)
LAM = 2.0
WMAX = 10.0

def moons(n, rng):
    t = rng.uniform(0, math.pi, n // 2)
    a = np.stack([np.cos(t), np.sin(t)], 1)
    b = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], 1)
    x = np.concatenate([a, b]) + rng.normal(0, 0.08, (n, 2))
    return (x - np.array([0.5, 0.25])) * 2.0

def energy(x):               # 왼쪽 반(x<0)이 에너지 1, 나머지 0
    return (x[:, 0] < 0).float()

def weights(x):
    return torch.clamp(torch.exp(-LAM * energy(x)), max=WMAX)   # 논문 식 5, 에너지 낮을수록 큰 가중치

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))
    def forward(s, t, x):
        return s.f(torch.cat([x, t[:, None]], 1))

def sinkhorn_plan(C, p, eps, iters=100):
    m = C.shape[0]
    logK = -C / eps
    lu = torch.zeros(m); lv = torch.zeros(m)
    logk = torch.full((m,), -math.log(m)); logp = torch.log(p)
    for _ in range(iters):
        lu = logk - torch.logsumexp(logK + lv[None, :], 1)
        lv = logp - torch.logsumexp(logK + lu[:, None], 0)
    return torch.exp(logK + lu[:, None] + lv[None, :])

def train(method, data, seed, steps=3000, bs=256):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    net = Net(); opt = torch.optim.Adam(net.parameters(), 1e-3)
    n = data.shape[0]; ess = []
    for it in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g)
        a = data[idx]; z = torch.randn(bs, 2, generator=g)
        w = weights(a)
        if method == 'ewfm':       # 독립 결합 + 샘플 가중 손실
            wl = w / w.mean()
        else:                      # wCOT: 목표 질량 p_j ∝ w_j 인 엔트로피 OT 결합에서 (z, a) 표집
            p = w / w.sum()
            C = torch.cdist(z, a) ** 2
            plan = sinkhorn_plan(C, p, eps=0.05 * C.median().item())
            row = plan / plan.sum(1, keepdim=True)
            j = torch.multinomial(row, 1, generator=g).squeeze(1)
            a = a[j]; wl = torch.ones(bs)
            if it % 500 == 0: ess.append(len(set(j.tolist())))
        t = torch.rand(bs, generator=g)
        xt = t[:, None] * a + (1 - t[:, None]) * z
        loss = (wl * ((net(t, xt) - (a - z)) ** 2).sum(1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net, ess

@torch.no_grad()
def sample(net, z, nfe):
    x = z.clone()
    for k in range(nfe):
        t = torch.full((z.shape[0],), k / nfe)
        x = x + net(t, x) / nfe
    return x

def swd(x, y, rng, L=64):
    th = rng.normal(size=(L, 2)); th /= np.linalg.norm(th, axis=1, keepdims=True)
    th = torch.tensor(th, dtype=torch.float32)
    px = torch.sort(x @ th.T, 0)[0]; py = torch.sort(y @ th.T, 0)[0]
    return ((px - py) ** 2).mean().sqrt().item()

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    data = torch.tensor(moons(20000, rng), dtype=torch.float32)
    w = weights(data); p = (w / w.sum()).numpy()
    ref = data[torch.tensor(rng.choice(len(data), 4000, p=p))]       # 기울어진 목표에서의 기준 표본
    res = {'lambda': LAM, 'wmax': WMAX, 'frac_low_energy_target_tilted': float((energy(ref) == 0).float().mean()),
           'frac_low_energy_target_untilted': float((energy(data) == 0).float().mean()), 'runs': []}
    zs = torch.randn(4000, 2, generator=torch.Generator().manual_seed(123))
    for method in ['ewfm', 'wcot']:
        for seed in range(3):
            t0 = time.time()
            net, ess = train(method, data, seed)
            r = {'method': method, 'seed': seed, 'train_sec': round(time.time() - t0, 1)}
            for nfe in [1, 2, 3, 100]:
                x = sample(net, zs, nfe)
                r[f'swd_nfe{nfe}'] = round(swd(x, ref, np.random.default_rng(1)), 4)
                r[f'frac_good_nfe{nfe}'] = round(float((energy(x) == 0).float().mean()), 4)
            x2 = sample(net, zs, 2); x100 = sample(net, zs, 100)
            r['straightness_gap_nfe2_vs_100'] = round((x2 - x100).norm(dim=1).mean().item(), 4)
            if ess: r['distinct_targets_per_batch_256'] = ess
            res['runs'].append(r); print(r, flush=True)
    summ = {}
    for method in ['ewfm', 'wcot']:
        rs = [r for r in res['runs'] if r['method'] == method]
        summ[method] = {k: [round(float(np.mean([r[k] for r in rs])), 4), round(float(np.std([r[k] for r in rs])), 4)] for k in rs[0] if k.startswith(('swd', 'frac', 'straight'))}
    res['summary_mean_std_over_3_seeds'] = summ
    # 유효 표본 크기: 배치 256, 어드밴티지 ~ N(0, s^2) 일 때 ESS = 1/sum p^2 (clamp 없음/있음)
    ess_tab = {}
    rng2 = np.random.default_rng(0)
    for s in [0.1, 0.3, 1.0]:
        for lam in [0.4, 1.0, 2.0, 5.0]:
            vals = []; vals_c = []
            for _ in range(200):
                A = rng2.normal(0, s, 256); ww = np.exp(lam * A)
                vals.append(1 / ((ww / ww.sum()) ** 2).sum())
                wc = np.minimum(ww, 10.0); vals_c.append(1 / ((wc / wc.sum()) ** 2).sum())
            ess_tab[f'adv_std={s},lambda={lam}'] = {'ess_noclamp': round(float(np.mean(vals)), 1), 'ess_wmax10': round(float(np.mean(vals_c)), 1)}
    res['ess_table_batch256'] = ess_tab
    json.dump(res, open('toy_wcot_results.json', 'w'), indent=1, ensure_ascii=False)
