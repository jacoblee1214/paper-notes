"""토이 T1: 타이피컬 셋 통계(노름, 등방성, 스펙트럼 백색성)는 어떤 이동을 잡아내는가.

논문 4.2절의 세 항(식은 논문 정의를 그대로 구현: C_norm, C_iso(블록 공분산 Frobenius), C_spec(주파수 구간 평균 파워의 분산))을
Dubins 규모(D = 4 x 16 x 16 = 1024)의 잡음에 적용한다. 보상은 선형 r(eps) = <eps, p> (p는 단위 벡터)로 두고,
알고리즘 1의 기울기 처리(성분 클램프 0.3, 노름 클립, 학습률 1.0, 10회)로 잡음을 움직인다.

묻는 것
  (a) 보상 방향 p 가 매끈한 저주파 무늬일 때와 무작위 방향일 때, 같은 이동량에서 세 통계가 얼마나 변하는가.
  (b) 노름 항만 쓸 때와 세 항을 모두 쓸 때(Dubins 표 3 계수 1.0, 0.5, 5.0) 보상 이득이 얼마나 줄고 통계가 어떻게 되는가.

주의: 선형 보상은 VLM 보상이 아니다. 이 토이는 "통계가 무엇을 보는가"를 보여 줄 뿐 실제 세계모델의 그럴듯함과 무관하다.
실행: python3 analysis/toy_typical_set.py   (torch 필요, CPU, 약 20초)
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)
OUT = Path(__file__).with_name("toy_typical_set_results.json")
C, H, W = 4, 16, 16
D = C * H * W
K = 16                 # 등방성 부분벡터 크기 (Dubins 표 3)
M = D // K
NPERM = 20
NB = 8                 # 스펙트럼 구간 수 (논문에 B 값 없음, 가정)

# 방사 주파수 구간 (FFT 격자)
fy = torch.fft.fftfreq(H).view(H, 1)
fx = torch.fft.fftfreq(W).view(1, W)
rad = torch.sqrt(fx ** 2 + fy ** 2)
edges = torch.linspace(0, rad.max() + 1e-6, NB + 1)
bin_id = torch.bucketize(rad, edges[1:-1])  # 0..NB-1


def c_norm(e):
    return -((e.norm() - math.sqrt(D)) ** 2)


def c_iso(e, gen):
    tot = 0.0
    for _ in range(NPERM):
        perm = torch.randperm(D, generator=gen)
        blocks = e.reshape(-1)[perm].reshape(M, K)
        S = blocks.T @ blocks / M
        tot = tot + ((S - torch.eye(K)) ** 2).sum() / K
    return -tot / NPERM


def c_spec(e):
    f = torch.fft.fft2(e.reshape(C, H, W), norm="ortho")
    p = (f.real ** 2 + f.imag ** 2).mean(dim=0)       # 채널 평균 파워 (H, W)
    bm = torch.stack([p[bin_id == b].mean() for b in range(NB)])
    return -bm.var(unbiased=False)


def stats_of(e, gen):
    with torch.no_grad():
        return {"norm_dev_sd": float((e.norm() - math.sqrt(D)) / 0.7071),
                "iso": float(-c_iso(e, gen)), "spec": float(-c_spec(e))}


def baseline(n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(n):
        e = torch.randn(D, generator=g)
        rows.append([float((e.norm() - math.sqrt(D)) / 0.7071), float(-c_iso(e, g)), float(-c_spec(e))])
    a = np.array(rows)
    return {"norm_dev_sd": [float(a[:, 0].mean()), float(a[:, 0].std())],
            "iso": [float(a[:, 1].mean()), float(a[:, 1].std())],
            "spec": [float(a[:, 2].mean()), float(a[:, 2].std())]}


def smooth_dir():
    """각 채널에 저주파 코사인 무늬 (주파수 1..2). 단위 벡터."""
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    p = torch.zeros(C, H, W)
    for c in range(C):
        p[c] = torch.cos(2 * math.pi * (1 * xx + (c % 2) * yy) / W + 0.3 * c) + 0.5 * torch.cos(2 * math.pi * 2 * yy / H)
    p = p.reshape(-1)
    return p / p.norm()


def random_dir(seed):
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(D, generator=g)
    return p / p.norm()


def run(direction, lam, beta, clip, iters=10, seed=0):
    """lam = (l1, l2, l3). 알고리즘 1: g = beta * grad r + grad C_pla, 클램프 0.3, 노름 클립, eps += lr * g."""
    g = torch.Generator().manual_seed(seed)
    e0 = torch.randn(D, generator=g)
    e = e0.clone()
    for _ in range(iters):
        x = e.clone().requires_grad_(True)
        r = (x * direction).sum()
        pla = 0.0
        if lam[0] > 0:
            pla = pla + lam[0] * c_norm(x)
        if lam[1] > 0:
            pla = pla + lam[1] * c_iso(x, g)
        if lam[2] > 0:
            pla = pla + lam[2] * c_spec(x)
        total = beta * r + pla if not isinstance(pla, float) else beta * r
        gr = torch.autograd.grad(total, x)[0]
        gr = gr.clamp(-0.3, 0.3)
        n = gr.norm()
        if n > clip:
            gr = gr * (clip / n)
        e = (e + 1.0 * gr).detach()
    st = stats_of(e, g)
    st["보상이득(sd)"] = float(((e - e0) * direction).sum())
    st["이동거리"] = float((e - e0).norm())
    st["이동/독립표본거리_%"] = 100 * st["이동거리"] / math.sqrt(2 * D)
    return st


def main():
    t0 = time.time()
    res = {"D": D, "K": K, "M": M, "NPERM": NPERM, "NB": NB, "기준(무작위 가우시안 200개)": baseline()}
    dirs = {"매끈한 저주파": smooth_dir(), "무작위 방향": random_dir(1)}
    settings = {"규제 없음": (0, 0, 0), "노름만(1.0)": (1.0, 0, 0), "세 항(1.0, 0.5, 5.0)": (1.0, 0.5, 5.0)}
    rows = []
    for dname, d in dirs.items():
        for clip in (1.0, 5.0):
            for sname, lam in settings.items():
                st = run(d, lam, beta=10.0, clip=clip)
                rows.append({"방향": dname, "노름클립": clip, "설정": sname, **st})
    res["rows"] = rows
    # 계수 민감도 (매끈한 방향, 클립 1.0): 세 항 전체 계수를 배수로
    sens = []
    for mult in (0.0, 0.2, 1.0, 5.0, 25.0):
        lam = tuple(mult * x for x in (1.0, 0.5, 5.0))
        st = run(dirs["매끈한 저주파"], lam, beta=10.0, clip=1.0)
        sens.append({"계수배수": mult, **st})
    res["계수민감도"] = sens
    res["걸린_시간_초"] = time.time() - t0
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT, "(%.1f s)" % res["걸린_시간_초"])
    for r in rows:
        print(r["방향"], r["노름클립"], r["설정"], {k: round(v, 3) for k, v in r.items() if isinstance(v, float)})
    print("기준", res["기준(무작위 가우시안 200개)"])


if __name__ == "__main__":
    main()
