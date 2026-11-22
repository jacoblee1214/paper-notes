"""토이 V1: 명령 토큰 정렬 손실과 검색(Recall@1)이 무엇을 재는지 보는 합성 실험.

X-WBC는 사람 전신·로봇 기준·희소 VR(5점) 세 명령 경로를 토큰으로 만들고, 같은 클립의 세 토큰 사이 쌍별 MSE(식 2)를 줄인다.
그림 3 오른쪽의 검색 Recall@1은 '같은 모션의 다른 경로 토큰이 가장 가까운가'를 잰다.
여기서는 PPO 대신 지도 회귀(행동 목표)로 정책 손실을 대신하고, 정렬 손실 유무에 따라 (a) 행동 오차, (b) 검색 Recall@1이 어떻게 달라지는지 본다.
실제 X-WBC의 재현이 아니라 메커니즘 확인용이다(로봇 3대, 잠재 24차원, torch CPU, 약 1분).

실행: python3 analysis/toy_alignment.py   → analysis/toy_alignment_results.json
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent / "toy_alignment_results.json"
torch.set_num_threads(2)
D, NR, PH = 24, 3, 6            # 잠재 차원, 로봇 수, 모션당 위상 수
HID, TOK, STEPS, BS = 96, 16, 2500, 256


def make_world(seed):
    g = torch.Generator().manual_seed(seed)
    A_h = torch.randn(D, 48, generator=g) / D ** 0.5           # 사람 전신 관측 (48차원)
    sel = torch.arange(0, 48, 4)                                   # 희소 VR: 전신의 일부 12개 성분만
    A_r = [torch.randn(D, 36, generator=g) / D ** 0.5 for _ in range(NR)]  # 로봇별 기준 모션 관측 (36차원)
    W_a = [torch.randn(D, 12, generator=g) / D ** 0.5 for _ in range(NR)]  # 로봇별 행동 (12차원)
    U_a = [torch.randn(10, 12, generator=g) / 10 ** 0.5 for _ in range(NR)]  # 로봇별 고유감각 영향
    return dict(A_h=A_h, sel=sel, A_r=A_r, W_a=W_a, U_a=U_a)


def sample_latent(m, phases, seed):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(m, D, generator=g)
    drift = 0.5 * torch.randn(m, D, generator=g)
    ph = torch.arange(phases).float() / (phases - 1)
    z = base[:, None, :] + ph[None, :, None] * drift[:, None, :]   # (m, phases, D)
    return z.reshape(-1, D), torch.arange(m).repeat_interleave(phases), ph.repeat(m)


def views(w, z, j):
    h = torch.tanh(z @ w["A_h"])
    v = h[:, w["sel"]]
    r = torch.tanh(z @ w["A_r"][j])
    return h, v, r


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, HID), nn.ReLU(), nn.Linear(HID, o))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_h, self.enc_v = mlp(48, TOK), mlp(12, TOK)                     # 로봇 공유
        self.enc_r = nn.ModuleList([mlp(36, TOK) for _ in range(NR)])         # 로봇별
        self.enc_p = nn.ModuleList([mlp(10, TOK) for _ in range(NR)])         # 고유감각, 로봇별
        self.core = nn.Sequential(nn.Linear(2 * TOK, HID), nn.ReLU(), nn.Linear(HID, HID), nn.ReLU())  # 공유 몸통
        self.dec = nn.ModuleList([nn.Linear(HID, 12) for _ in range(NR)])      # 로봇별 디코더

    def tokens(self, w, z, j):
        h, v, r = views(w, z, j)
        return F.normalize(self.enc_h(h), dim=-1), F.normalize(self.enc_v(v), dim=-1), F.normalize(self.enc_r[j](r), dim=-1)

    def act(self, tok, s, j):
        return self.dec[j](self.core(torch.cat([tok, self.enc_p[j](s)], -1)))


def target(w, z, s, j):
    return z @ w["W_a"][j] + s @ w["U_a"][j]


def train(w, mode, seed, lam):
    torch.manual_seed(seed)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    ztr, _, _ = sample_latent(600, PH, 1000 + seed)
    for step in range(STEPS):
        loss = 0.0
        for j in range(NR):
            idx = torch.randint(0, len(ztr), (BS,))
            z = ztr[idx]
            s = torch.randn(BS, 10)
            th, tv, tr = net.tokens(w, z, j)
            route = torch.randint(0, 3, (BS, 1))                               # 1:1:1 경로 샘플링
            tok = torch.where(route == 0, th, torch.where(route == 1, tv, tr))
            task = F.mse_loss(net.act(tok, s, j), target(w, z, s, j)) if mode != "align_only" else 0.0
            align = (F.mse_loss(th, tv) + F.mse_loss(th, tr) + F.mse_loss(tv, tr)) / 3
            loss = loss + (task if mode != "align_only" else 0.0) + (lam * align if mode != "task_only" else 0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


@torch.no_grad()
def evaluate(net, w):
    zte, mid, ph = sample_latent(96, PH, 7)        # 학습에 없던 96 모션 x 6 위상 = 576 갤러리 (우연 수준 0.17%, 1.04%)
    res = {}
    toks = [net.tokens(w, zte, j) for j in range(NR)]
    names = ["human", "vr", "robot"]

    def recall(q, g):
        sim = q @ g.T
        top = sim.argmax(1)
        same_phase = (top == torch.arange(len(zte))).float().mean().item()
        same_motion = (mid[top] == mid).float().mean().item()
        return 100 * same_phase, 100 * same_motion

    for a in range(3):
        for b in range(a + 1, 3):
            res[f"{names[a]}->{names[b]}"] = recall(toks[0][a], toks[0][b])
    res["robot0->robot1"] = recall(toks[0][2], toks[1][2])
    # 행동 오차 (경로별, 로봇 평균)
    err = {n: [] for n in names}
    s = torch.randn(len(zte), 10)
    for j in range(NR):
        t = target(w, zte, s, j)
        for k, n in enumerate(names):
            err[n].append(F.mse_loss(net.act(toks[j][k], s, j), t).item())
    res["action_mse"] = {n: float(np.mean(v)) for n, v in err.items()}
    res["token_std"] = float(torch.cat([toks[0][0], toks[0][1], toks[0][2]]).std(0).mean())
    return res


def main():
    t0 = time.time()
    w = make_world(0)
    modes = {"align_lam1": ("both", 1.0), "no_align": ("both", 0.0), "align_only": ("align_only", 1.0)}
    out = {}
    for name, (mode, lam) in modes.items():
        runs = [evaluate(train(w, mode, seed, lam), w) for seed in range(3)]
        agg = {}
        for key in runs[0]:
            if key == "action_mse":
                agg[key] = {n: [float(np.mean([r[key][n] for r in runs])), float(np.std([r[key][n] for r in runs]))] for n in runs[0][key]}
            elif key == "token_std":
                agg[key] = [float(np.mean([r[key] for r in runs])), float(np.std([r[key] for r in runs]))]
            else:
                agg[key] = {"same_phase": [float(np.mean([r[key][0] for r in runs])), float(np.std([r[key][0] for r in runs]))],
                            "same_motion": [float(np.mean([r[key][1] for r in runs])), float(np.std([r[key][1] for r in runs]))]}
        out[name] = agg
        print(name, f"{time.time() - t0:.0f}s", json.dumps(agg)[:300])
    out["chance"] = {"same_phase": 100 / 576, "same_motion": 100 / 96}
    out["seconds"] = time.time() - t0
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
