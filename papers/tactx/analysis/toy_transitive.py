"""토이 T1: 쌍 데이터만으로 세 센서가 한 잠재 공간에 정렬되는가 (논문 §4.2 주장 점검).

합성 접촉 s = (x, y, 반경 r, 종횡비 a)를 세 가짜 센서로 렌더링한다.
  D: 16x16 "깊이 영상" (가우시안 블롭), E: 15차원 벡터 (s의 고정 비선형 사상), F: 12x16 압력 격자 (넓은 커널 + 포화).
학습은 D-E, E-F 쌍만 쓰는 조건("transitive")과 D-E, E-F, F-D 쌍을 모두 쓰는 조건("all3")을 비교한다.
방법: recon (재구성+KL), l2 (재구성+KL+L2 당김), nce (InfoNCE만), tactx (재구성+KL+InfoNCE).
평가는 D-F 쌍을 한 번도 학습하지 않은 조건에서, 정답 대응이 있는 검증 접촉 D-F에 대해
  pos_cos: 정답 쌍 코사인, neg_cos: 무작위 쌍 코사인, top1: F 잠재로 D 잠재를 찾는 검색 정확도(정답 대응 사용),
  nn_cos: 논문 §4.2 방식(E 잠재 최근접으로 이어 붙인 D 잠재와 F 잠재의 코사인).
실행: python3 analysis/toy_transitive.py  → analysis/toy_transitive_results.json (torch CPU, 2분 안팎)
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

OUT = Path(__file__).resolve().parent / "toy_transitive_results.json"
torch.set_num_threads(4)
LAT = 16
N_TRAIN, N_VAL = 3000, 1000
STEPS, BATCH = 600, 128
TAU = 0.01
BETA = 0.1


def sample_contacts(n, g):
    x = torch.rand(n, generator=g) * 2 - 1
    y = torch.rand(n, generator=g) * 2 - 1
    r = 0.15 + 0.35 * torch.rand(n, generator=g)
    a = 0.6 + 0.8 * torch.rand(n, generator=g)
    return torch.stack([x, y, r, a], 1)


def render(s, kind, mix):
    x, y, r, a = s[:, 0:1], s[:, 1:2], s[:, 2:3], s[:, 3:4]
    if kind == "D":
        gx = torch.linspace(-1, 1, 16).view(1, 1, 16)
        gy = torch.linspace(-1, 1, 16).view(1, 16, 1)
        img = torch.exp(-(((gx - x.view(-1, 1, 1)) / (r.view(-1, 1, 1) * a.view(-1, 1, 1))) ** 2
                          + ((gy - y.view(-1, 1, 1)) / (r.view(-1, 1, 1) / a.view(-1, 1, 1))) ** 2))
        return img.reshape(len(s), -1) + 0.02 * torch.randn(len(s), 256)
    if kind == "F":
        gx = torch.linspace(-1, 1, 16).view(1, 1, 16)
        gy = torch.linspace(-1, 1, 12).view(1, 12, 1)
        img = torch.tanh(2.0 * torch.exp(-(((gx - x.view(-1, 1, 1)) / (1.4 * r.view(-1, 1, 1))) ** 2
                                           + ((gy - y.view(-1, 1, 1)) / (1.4 * r.view(-1, 1, 1))) ** 2)))
        return img.reshape(len(s), -1) + 0.02 * torch.randn(len(s), 192)
    z = torch.tanh(s @ mix[0] + mix[1]) @ mix[2]
    return z + 0.02 * torch.randn(len(s), 15)


DIMS = {"D": 256, "E": 15, "F": 192}


def mlp(i, o, h=96):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.ModuleDict({k: mlp(d, 2 * LAT) for k, d in DIMS.items()})
        self.dec = nn.ModuleDict({k: mlp(LAT, d) for k, d in DIMS.items()})

    def encode(self, k, x):
        mu, lv = self.enc[k](x).chunk(2, -1)
        return mu, lv.clamp(-10, 10)


def nt_xent(a, b, tau):
    """SimCLR NT_Xent와 같은 정의: 2N개 앵커, 부정 예시에 같은 센서 항목 포함."""
    n = len(a)
    z = torch.cat([a, b])
    sim = Fn.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / tau
    pos = torch.cat([torch.diag(sim, n), torch.diag(sim, -n)]).view(2 * n, 1)
    # 자기 자신과 양성쌍은 부정 예시에서 뺀다
    mask2 = torch.ones(2 * n, 2 * n, dtype=torch.bool)
    mask2.fill_diagonal_(False)
    idx = torch.arange(n)
    mask2[idx, idx + n] = False
    mask2[idx + n, idx] = False
    neg = sim[mask2].view(2 * n, -1)
    logits = torch.cat([pos, neg], 1)
    return Fn.cross_entropy(logits, torch.zeros(2 * n, dtype=torch.long))


def run(method, pairs, seed, data):
    torch.manual_seed(seed)
    m = Model()
    opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4)
    for step in range(STEPS):
        beta = BETA * min(1.0, step / (STEPS * 0.1))
        loss = 0.0
        for (p, q) in pairs:
            xp, xq = data[(p, q)]
            idx = torch.randint(0, len(xp), (BATCH,))
            a, b = xp[idx], xq[idx]
            mua, lva = m.encode(p, a)
            mub, lvb = m.encode(q, b)
            za = mua + torch.randn_like(mua) * (0.5 * lva).exp()
            zb = mub + torch.randn_like(mub) * (0.5 * lvb).exp()
            if method != "nce":
                rec = (Fn.l1_loss(m.dec[p](za), a) + Fn.l1_loss(m.dec[q](zb), b)
                       + Fn.l1_loss(m.dec[p](zb), a) + Fn.l1_loss(m.dec[q](za), b))
                kl = (-0.5 * (1 + lva - mua.pow(2) - lva.exp())).mean() + (-0.5 * (1 + lvb - mub.pow(2) - lvb.exp())).mean()
                loss = loss + rec + beta * kl
            if method in ("nce", "tactx"):
                loss = loss + nt_xent(mua, mub, TAU)
            if method == "l2":
                loss = loss + Fn.mse_loss(mua, mub)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m


@torch.no_grad()
def evaluate(m, val):
    mu = {k: m.encode(k, val[k])[0] for k in "DEF"}
    n = len(mu["D"])
    d, f = Fn.normalize(mu["D"], dim=1), Fn.normalize(mu["F"], dim=1)
    pos = (d * f).sum(1)
    perm = torch.roll(torch.arange(n), 1)
    neg = (d * f[perm]).sum(1)
    sim = d @ f.t()
    top1 = (sim.argmax(1) == torch.arange(n)).float().mean().item()
    # 논문 §4.2 방식: 별도 검증 D-E 집합과 E-F 집합 (같은 접촉이 아닌 서로 다른 접촉 집합)
    h = n // 2
    e_de = Fn.normalize(mu["E"][:h], dim=1)
    d_de = mu["D"][:h]
    e_ef = Fn.normalize(mu["E"][h:], dim=1)
    f_ef = Fn.normalize(mu["F"][h:], dim=1)
    nn_idx = (e_ef @ e_de.t()).argmax(1)
    nn_cos = (Fn.normalize(d_de[nn_idx], dim=1) * f_ef).sum(1)
    perm2 = torch.roll(torch.arange(len(nn_cos)), 1)
    nn_neg = (Fn.normalize(d_de[nn_idx], dim=1) * f_ef[perm2]).sum(1)
    return {"pos_cos": pos.mean().item(), "neg_cos": neg.mean().item(), "top1": top1,
            "nn_cos": nn_cos.mean().item(), "nn_cos_random_pairs": nn_neg.mean().item()}


def main():
    t0 = time.time()
    g = torch.Generator().manual_seed(123)
    mix = (torch.randn(4, 24, generator=g), torch.randn(24, generator=g) * 0.3, torch.randn(24, 15, generator=g) / math.sqrt(24))
    torch.manual_seed(7)
    s_tr = {p: sample_contacts(N_TRAIN, g) for p in ("DE", "EF", "FD")}
    data = {}
    for p, (a, b) in {"DE": ("D", "E"), "EF": ("E", "F"), "FD": ("F", "D")}.items():
        data[(a, b)] = (render(s_tr[p], a, mix), render(s_tr[p], b, mix))
    s_val = sample_contacts(N_VAL, g)
    val = {k: render(s_val, k, mix) for k in "DEF"}
    conditions = {"transitive": [("D", "E"), ("E", "F")], "all3": [("D", "E"), ("E", "F"), ("F", "D")]}
    res = {}
    for cond, pairs in conditions.items():
        for method in ("recon", "l2", "nce", "tactx"):
            rows = []
            for seed in range(2):
                m = run(method, pairs, seed, data)
                rows.append(evaluate(m, val))
            res[f"{cond}/{method}"] = {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))] for k in rows[0]}
            r = res[f"{cond}/{method}"]
            print(f"{cond:10s} {method:6s} pos {r['pos_cos'][0]:.3f} neg {r['neg_cos'][0]:.3f} top1 {r['top1'][0]:.3f} "
                  f"nn_cos {r['nn_cos'][0]:.3f} (무작위 {r['nn_cos_random_pairs'][0]:.3f})  [{time.time() - t0:.0f}s]")
    res["_config"] = {"latent": LAT, "n_train_per_pair": N_TRAIN, "n_val": N_VAL, "steps": STEPS, "batch": BATCH, "tau": TAU,
                      "beta_max": BETA, "seeds": 2, "note": "합성 센서, 한 가지 렌더링 설정. 논문 실험의 대체가 아니라 메커니즘 점검"}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
