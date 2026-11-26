"""토이 1: 현재 점과 목표 점을 어떻게 묶어 인코딩하면 상대 강체 변환을 읽어낼 수 있는가 (지도학습 대리 과제).

논문 3.2절과 표 2는 Paired Point Encoding(6D 점 쌍에 PointNet), Decoupled Point Encoding(두 PointNet을 따로),
MLP Point Encoding(점을 펼쳐 MLP)을 정책 성능으로 비교한다(학생 SR 0.600 / 0.203 / 0.057). 이 토이는 RL 없이
"입력만 보고 현재 → 목표 상대 변환(이동 3 + 회전)을 맞히는" 지도학습으로 세 인코더의 정보 차이만 본다.
RL 성능의 대리값이 아니며, 표 2의 수치를 재현하지도 않는다.

설정(가정, 논문에 없음): 점 N = 32개, 매 표본마다 새 형상(비등방 가우시안 점군), 현재 자세는 임의 회전·이동,
목표 자세는 현재에서 상대 회전(각도 0–180도)과 이동(±0.15 m). 평가는 학습에 쓰지 않은 형상 4096개.
세 인코더는 파라미터 수를 비슷하게 맞춘다. 시드 3개.

실행: python3 analysis/toy_pair_encoding.py  (torch CPU, 약 1–2분) → analysis/toy_pair_encoding_results.json
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "toy_pair_encoding_results.json"
N = 32
STEPS = 1500
BATCH = 256
EVAL = 4096


def rot_from_6d(x):
    a, b = x[..., :3], x[..., 3:6]
    e1 = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b = b - (e1 * b).sum(-1, keepdim=True) * e1
    e2 = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1)


def random_rot(b, gen, max_angle=np.pi):
    axis = torch.randn(b, 3, generator=gen)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    ang = torch.rand(b, generator=gen) * max_angle
    K = torch.zeros(b, 3, 3)
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]
    s, c = torch.sin(ang)[:, None, None], torch.cos(ang)[:, None, None]
    return torch.eye(3) + s * K + (1 - c) * (K @ K)


def make_batch(b, gen):
    scale = 0.03 + 0.05 * torch.rand(b, 1, 3, generator=gen)  # 비등방 형상, 3–8 cm
    base = torch.randn(b, N, 3, generator=gen) * scale
    R0 = random_rot(b, gen)
    t0 = (torch.rand(b, 1, 3, generator=gen) - 0.5) * 0.3
    cur = base @ R0.transpose(1, 2) + t0
    Rd = random_rot(b, gen)  # 상대 회전 0–180도
    td = (torch.rand(b, 1, 3, generator=gen) - 0.5) * 0.3
    center = cur.mean(1, keepdim=True)
    tgt = (cur - center) @ Rd.transpose(1, 2) + center + td
    return cur, tgt, Rd, td.squeeze(1)


class PointNet(nn.Module):
    def __init__(self, din, h, out):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Linear(h, 2 * h), nn.ReLU())
        self.out = nn.Linear(2 * h, out)

    def forward(self, x):  # [B, N, din], 평균-최대 혼합 풀링(채널 절반씩)
        f = self.mlp(x)
        h = f.shape[-1] // 2
        g = torch.cat([f[..., :h].amax(1), f[..., h:].mean(1)], dim=-1)
        return self.out(g)


class Model(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        head_in = 128
        if kind == "paired":
            self.enc = PointNet(6, 96, head_in)
        elif kind == "decoupled":
            self.enc_c, self.enc_t = PointNet(3, 96, head_in // 2), PointNet(3, 96, head_in // 2)
        else:
            self.enc = nn.Sequential(nn.Linear(N * 6, 256), nn.ReLU(), nn.Linear(256, head_in))
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(head_in, 256), nn.ReLU(), nn.Linear(256, 9))

    def forward(self, cur, tgt):
        if self.kind == "paired":
            z = self.enc(torch.cat([cur, tgt], -1))
        elif self.kind == "decoupled":
            z = torch.cat([self.enc_c(cur), self.enc_t(tgt)], -1)
        else:
            z = self.enc(torch.cat([cur, tgt], -1).flatten(1))
        o = self.head(z)
        return rot_from_6d(o[:, :6]), o[:, 6:]


def metrics(model, cur, tgt, Rd, td):
    with torch.no_grad():
        R, t = model(cur, tgt)
        cosang = ((R.transpose(1, 2) @ Rd).diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2
        ang = torch.rad2deg(torch.acos(cosang.clamp(-1, 1)))
        terr = (t - td).norm(dim=-1) * 100
    return float(ang.mean()), float(ang.median()), float(terr.mean()), float((ang < 10).float().mean())


def run(kind, seed):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(1000 + seed)
    model = Model(kind)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    for _ in range(STEPS):
        cur, tgt, Rd, td = make_batch(BATCH, gen)
        R, t = model(cur, tgt)
        loss = ((R - Rd) ** 2).mean() * 9 + ((t - td) ** 2).mean() * 100
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    ev = torch.Generator().manual_seed(999)
    cur, tgt, Rd, td = make_batch(EVAL, ev)
    m = metrics(model, cur, tgt, Rd, td)
    return {"params": n_par, "rot_err_mean_deg": m[0], "rot_err_median_deg": m[1], "trans_err_cm": m[2], "frac_rot_under_10deg": m[3]}


def main():
    t0 = time.time()
    res = {"setup": {"N": N, "steps": STEPS, "batch": BATCH, "eval_shapes": EVAL, "seeds": [0, 1, 2]}, "runs": {}, "summary": {}}
    for kind in ["paired", "decoupled", "mlp"]:
        rs = [run(kind, s) for s in range(3)]
        res["runs"][kind] = rs
        res["summary"][kind] = {k: round(float(np.mean([r[k] for r in rs])), 3) for k in rs[0]}
        res["summary"][kind]["rot_err_mean_deg_sd"] = round(float(np.std([r["rot_err_mean_deg"] for r in rs])), 3)
        print(kind, res["summary"][kind], f"{time.time() - t0:.0f}s", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
