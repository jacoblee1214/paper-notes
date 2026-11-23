"""부록 A.1이 적은 구조(6채널 입력 U-Net, 블록 출력 채널 (64,128,256,256), 잔차 블록 22개)를
랜덤 가중치로 만들어 파라미터 수와 한 스텝 시간을 잰다. 가중치 다운로드와 학습은 하지 않는다.

가정(논문 미기재): 입력 해상도 128 또는 256 정방형, 시간 임베딩은 채널 256의 사인 임베딩 + MLP,
다운샘플은 stride-2 합성곱, 업샘플은 최근접 + 합성곱, 어텐션 없음.
실행: python3 analysis/toy_unet_bench.py   (torch 필요, 장치는 MPS가 있으면 MPS와 CPU 둘 다, 약 1분)
결과: analysis/toy_unet_bench_results.json
"""
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent / "toy_unet_bench_results.json"


class Res(nn.Module):
    def __init__(self, cin, cout, temb):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(temb, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(e)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, ch=(64, 128, 256, 256), enc_blocks=2, mid_blocks=2, dec_blocks=3, temb=256):
        super().__init__()
        self.temb = temb
        self.mlp = nn.Sequential(nn.Linear(temb, temb), nn.SiLU(), nn.Linear(temb, temb))
        self.inp = nn.Conv2d(6, ch[0], 3, padding=1)
        self.enc, self.down = nn.ModuleList(), nn.ModuleList()
        skips, c = [ch[0]], ch[0]
        for i, co in enumerate(ch):
            for _ in range(enc_blocks):
                self.enc.append(Res(c, co, temb)); c = co; skips.append(c)
            if i < len(ch) - 1:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1)); skips.append(c)
        self.mid = nn.ModuleList([Res(c, c, temb) for _ in range(mid_blocks)])
        self.dec, self.up = nn.ModuleList(), nn.ModuleList()
        for i, co in reversed(list(enumerate(ch))):
            for _ in range(dec_blocks):
                self.dec.append(Res(c + skips.pop(), co, temb)); c = co
            if i > 0:
                self.up.append(nn.Conv2d(c, c, 3, padding=1))
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 3, 3, padding=1))
        self.enc_blocks, self.dec_blocks, self.nch = enc_blocks, dec_blocks, len(ch)

    def emb(self, t):
        half = self.temb // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[:, None].float() * f[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], -1))

    def forward(self, x_t, t, mixed):
        e = self.emb(t)
        h = self.inp(torch.cat([mixed, x_t], 1))
        hs, ei, di = [h], 0, 0
        for i in range(self.nch):
            for _ in range(self.enc_blocks):
                h = self.enc[ei](h, e); ei += 1; hs.append(h)
            if i < self.nch - 1:
                h = self.down[i](h); hs.append(h)
        for m in self.mid:
            h = m(h, e)
        for i in range(self.nch):
            for _ in range(self.dec_blocks):
                h = self.dec[di](torch.cat([h, hs.pop()], 1), e); di += 1
            if i < self.nch - 1:
                h = self.up[i](F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.out(h)


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


def bench(dev, res, bs=2, n=8):
    torch.manual_seed(0)
    m = UNet().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
    x = torch.randn(bs, 3, res, res, device=dev)
    mix = torch.randn(bs, 3, res, res, device=dev)
    t = torch.randint(0, 1000, (bs,), device=dev)
    for _ in range(2):
        loss = F.mse_loss(m(x, t, mix), x); opt.zero_grad(); loss.backward(); opt.step()
    sync(dev); t0 = time.time()
    for _ in range(n):
        loss = F.mse_loss(m(x, t, mix), x); opt.zero_grad(); loss.backward(); opt.step()
    sync(dev); train = (time.time() - t0) / n
    with torch.no_grad():
        x1, mix1, t1 = x[:1], mix[:1], t[:1]
        m(x1, t1, mix1); sync(dev); t0 = time.time()
        for _ in range(n):
            m(x1, t1, mix1)
        sync(dev); fwd = (time.time() - t0) / n
    return {"train_step_s_bs2": round(train, 3), "forward_s_bs1": round(fwd, 3)}


def main():
    m = UNet()
    n_res = len(m.enc) + len(m.mid) + len(m.dec)
    R = {"params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 2), "residual_blocks": n_res,
         "assumptions": "해상도·시간 임베딩·다운/업샘플 방식은 논문에 없어 가정", "bench": {}}
    devs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev in devs:
        for res in (128, 256):
            r = bench(dev, res)
            steps = 139800
            r["train_300_epochs_hours"] = round(r["train_step_s_bs2"] * steps / 3600, 1)
            r["sample_1000_steps_s"] = round(r["forward_s_bs1"] * 1000, 1)
            R["bench"][f"{dev}_{res}"] = r
            print(dev, res, r, flush=True)
    OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")
    print(R["params_M"], "M params,", n_res, "res blocks")


if __name__ == "__main__":
    main()
