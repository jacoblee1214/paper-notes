"""토이 T2: 부록 A.5의 NDM 구조를 그대로 짜서 정책 미세조정 한 번의 계산량을 이 기기에서 잰다.

검증하려는 것: 부록 D의 "NDM과 정책 최적화 한 반복 약 3초(RTX 4090)"가 이 구조와 표 6 설정에서
CPU/MPS로도 같은 자릿수인가. 정확한 FK, 보상 항, 정책 구조는 논문에 없어 다음을 가정한다.
  - 정책: MLP 3층(은닉 256), 입력 = 상태 64 + 이전 행동 23 + 기준 미래 상태 단순 대체 벡터 64
  - 미분 가능 FK는 생략 (비용이 작다고 가정, 측정에 포함 안 됨)
  - 반복당 미니배치 수는 표 6 값을 그대로 쓰고 에폭 수는 1로 가정
실행: python3 toy_ndm_cost.py  (CPU 약 1분, MPS 사용 가능하면 추가)  -> toy_ndm_cost_results.json
"""
import json
import os
import time

import torch
import torch.nn as nn

torch.manual_seed(0)
S, A, H, M = 64, 23, 10, 5
DEC_IN = 64 + S + A


class NDM(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(S + A, 64)
        self.pos = nn.Parameter(torch.zeros(H - 1, 64))
        layer = nn.TransformerEncoderLayer(64, 4, 16, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.dec = nn.Sequential(nn.Linear(DEC_IN, 64), nn.ELU(), nn.Linear(64, 512), nn.ELU(), nn.Linear(512, 512))
        self.heads = nn.ModuleList([nn.Linear(512, S) for _ in range(M)])

    def forward(self, hist_s, hist_a, s, a):
        x = self.proj(torch.cat([hist_s, hist_a], -1)) + self.pos
        z = self.enc(x)[:, -1]
        u = self.dec(torch.cat([z, s, a], -1))
        m = torch.randint(0, M, (1,)).item()  # 한 머리를 무작위로 선택 (롤아웃 비용 근사)
        return s + self.heads[m](u)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(S + A + 64, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(),
                                 nn.Linear(256, A), nn.Tanh())

    def forward(self, s, a_prev, ref):
        return self.net(torch.cat([s, a_prev, ref], -1))


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


def bench(dev, B=512, Lw=8, Lpi=128, reps=2):
    ndm, pol = NDM().to(dev), Policy().to(dev)
    opt_w = torch.optim.Adam(ndm.parameters(), 2e-3)
    opt_p = torch.optim.Adam(pol.parameters(), 1e-4)
    npar = sum(p.numel() for p in ndm.parameters())
    # --- NDM 한 미니배치: 길이 Lw 자기회귀 롤아웃 (정답 행동 사용)
    def ndm_step():
        hs = torch.randn(B, H - 1, S, device=dev) * 0.1
        ha = torch.randn(B, H - 1, A, device=dev) * 0.1
        s = torch.randn(B, S, device=dev) * 0.1
        loss = 0.0
        for k in range(Lw):
            a = torch.randn(B, A, device=dev) * 0.1
            sn = ndm(hs, ha, s, a)
            loss = loss + (sn ** 2).mean()
            hs = torch.cat([hs[:, 1:], s[:, None]], 1)
            ha = torch.cat([ha[:, 1:], a[:, None]], 1)
            s = sn
        opt_w.zero_grad()
        (loss / Lw).backward()
        opt_w.step()

    # --- 정책 한 미니배치: 길이 Lpi 롤아웃 (NDM 고정, 정책과 NDM을 통해 역전파)
    def pol_step():
        hs = torch.randn(B, H - 1, S, device=dev) * 0.1
        ha = torch.randn(B, H - 1, A, device=dev) * 0.1
        s = torch.randn(B, S, device=dev) * 0.1
        ap = torch.zeros(B, A, device=dev)
        ref = torch.randn(B, 64, device=dev) * 0.1
        loss = 0.0
        for k in range(Lpi):
            a = pol(s, ap, ref)
            sn = ndm(hs, ha, s, a)
            loss = loss + ((sn - ref[:, :S]) ** 2).mean()
            hs = torch.cat([hs[:, 1:], s[:, None]], 1)
            ha = torch.cat([ha[:, 1:], a[:, None]], 1)
            s, ap = sn, a
        opt_p.zero_grad()
        (loss / Lpi).backward()
        opt_p.step()

    out = {"device": dev, "ndm_params": npar}
    for name, fn in (("ndm_minibatch_s", ndm_step), ("policy_minibatch_s", pol_step)):
        fn(); sync(dev)  # 워밍업
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        sync(dev)
        out[name] = (time.perf_counter() - t) / reps
    out["iter_estimate_s"] = 32 * out["ndm_minibatch_s"] + 2 * out["policy_minibatch_s"]
    return out


if __name__ == "__main__":
    res = []
    devs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for d in devs:
        t0 = time.time()
        r = bench(d)
        r["wall_s"] = time.time() - t0
        print(r)
        res.append(r)
    res.append({"torch": torch.__version__, "threads": torch.get_num_threads(),
                "paper_claim_iter_s": 3.0, "note": "정책 구조와 FK 생략 가정, 에폭 1회 가정"})
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "toy_ndm_cost_results.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
