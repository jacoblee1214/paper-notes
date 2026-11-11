"""작은 PINN 손실의 2차 미분(매개변수 방향 이차 도함수)에 AD/FD 검사 프로토콜을 적용 (CPU, torch 4스레드, 수십 초).

질문: 같은 손실의 이차 도함수 vᵀ∇²L v를 여러 방법으로 구했을 때, 기준(장정밀도 전진 접선 AD)과의 오차는
      차분 간격 h에 대해 U 곡선을 그리는가, 고원을 보이는가? 반복(초기값, 방향)의 산포와 비교하면 어떤가?

설정
  - 문제: u''(x) = −π² sin(πx), x ∈ [0, 1], u(0) = u(1) = 0 (해 u = sin(πx)).
  - 모델: u_θ(x) = W3 tanh(W2 tanh(W1 x + b1) + b2) + b3, 은닉 16 × 2, 매개변수 321개 (θ는 하나의 평탄 벡터).
  - 손실: L(θ) = mean_i [u_θ''(x_i) − f(x_i)]² + 10·(u_θ(0)² + u_θ(1)²), 배치 격자 64점 (고정).
    u'' 는 함수형 자동 미분(torch.func.grad 두 번)으로 얻는다. 즉 손실은 θ에 대해 미분 가능한 PINN 손실이다.
  - 추정량 (모두 방향 v 하나에 대한 vᵀ∇²L v):
      AD32   : 전진 접선 기준의 Hessian-벡터 곱을 float32로 (torch.func.jvp ∘ grad)
      FDg32  : [∇L(θ+hv) − ∇L(θ−hv)]·v / (2h), 기울기는 float32 자동 미분
      FDg64  : 같은 식을 float64로
      FDl32  : [L(θ+hv) − 2L(θ) + L(θ−hv)] / h², float32
      FDl64  : 같은 식을 float64로
  - 기준: float64 전진 접선 AD (jvp ∘ grad). 이 값을 참값으로 쓴다 (AD와 FD의 한계 비교용).
  - 반복: 초기값과 방향을 시드 20개로 바꾼다. 차분 간격 h = 1, 0.3, 10^(−1), …, 10^(−8).
  - 판정: 시드별 상대 오차 |추정 − 기준| / |기준|의 중앙값과 사분위 범위. U 곡선은 최소가 내부에 있고 양 끝이
    최소의 3배 이상인 경우, 고원은 h에 따라 오차가 1/3–3배 안에서 변하지 않는 경우로 본다.
가정·한계: 장난감 PINN 하나이며 학습된 상태가 아니라 무작위 초기값에서 검사한다. 기울기는 θ의 함수로만 계산한다.
  float64는 CPU에서 계산했다(MPS는 float64를 지원하지 않는다).
실행: python3 analysis/extra_pinn_hessian.py -> analysis/extra_pinn_hessian_results.json
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "4"

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import grad, jvp, vmap

torch.set_num_threads(4)
HERE = Path(__file__).resolve().parent

HID = 16
N_COLLOC = 64
LAM_BC = 10.0
N_SEEDS = 20
H_EXPONENTS = list(range(1, 9))          # h = 10^-1 ... 10^-8
DEVICE = "cpu"


def n_params():
    return HID * 1 + HID + HID * HID + HID + 1 * HID + 1


def unpack(th):
    i = 0
    W1 = th[i:i + HID].reshape(HID, 1); i += HID
    b1 = th[i:i + HID]; i += HID
    W2 = th[i:i + HID * HID].reshape(HID, HID); i += HID * HID
    b2 = th[i:i + HID]; i += HID
    W3 = th[i:i + HID].reshape(1, HID); i += HID
    b3 = th[i:i + 1]
    return W1, b1, W2, b2, W3, b3


def u_scalar(x, th):
    W1, b1, W2, b2, W3, b3 = unpack(th)
    h1 = torch.tanh(W1[:, 0] * x + b1)
    h2 = torch.tanh(W2 @ h1 + b2)
    return (W3 @ h2 + b3).squeeze()


du = grad(u_scalar, argnums=0)
d2u = grad(du, argnums=0)


def make_loss(xs):
    f = -(math.pi**2) * torch.sin(math.pi * xs)

    def loss(th):
        r = vmap(d2u, in_dims=(0, None))(xs, th) - f
        bc0 = u_scalar(torch.zeros((), dtype=th.dtype), th)
        bc1 = u_scalar(torch.ones((), dtype=th.dtype), th)
        return torch.mean(r**2) + LAM_BC * (bc0**2 + bc1**2)
    return loss


def init_theta(seed, dtype):
    g = torch.Generator().manual_seed(seed)
    # 층별 1/sqrt(fan_in) 정규분포 초기화
    sizes = [(HID, 1), (HID,), (HID, HID), (HID,), (1, HID), (1,)]
    parts = []
    for shape in sizes:
        fan_in = shape[-1] if len(shape) > 1 else HID
        parts.append(torch.randn(int(np.prod(shape)), generator=g, dtype=torch.float64) / math.sqrt(fan_in))
    return torch.cat(parts).to(dtype)


def hvp_ad(loss, th, v):
    """전진 접선 기준 Hessian-벡터 곱의 방향 성분: vᵀ H v (= (H v)·v)."""
    _, hv = jvp(grad(loss), (th,), (v,))
    return torch.dot(hv, v)


def fd_grad(loss, th, v, h):
    gp = grad(loss)(th + h * v)
    gm = grad(loss)(th - h * v)
    return torch.dot(gp - gm, v) / (2 * h)


def fd_loss(loss, th, v, h):
    return (loss(th + h * v) - 2 * loss(th) + loss(th - h * v)) / (h * h)


def main():
    t_all = time.perf_counter()
    xs64 = torch.linspace(0.0, 1.0, N_COLLOC, dtype=torch.float64)
    xs32 = xs64.float()
    loss64 = make_loss(xs64)
    loss32 = make_loss(xs32)

    hs = [1.0, 0.3] + [10.0 ** (-k) for k in H_EXPONENTS]
    per_seed = []
    ref_vals = []
    for seed in range(N_SEEDS):
        th64 = init_theta(seed, torch.float64)
        g = torch.Generator().manual_seed(1000 + seed)
        v64 = torch.randn(n_params(), generator=g, dtype=torch.float64)
        v64 = v64 / v64.norm()
        ref = float(hvp_ad(loss64, th64, v64))
        ref_vals.append(ref)
        th32, v32 = th64.float(), v64.float()
        ad32 = float(hvp_ad(loss32, th32, v32))
        est = {"AD32": {}, "FDg32": {}, "FDg64": {}, "FDl32": {}, "FDl64": {}}
        for h in hs:
            est["FDg32"][h] = float(fd_grad(loss32, th32, v32, torch.tensor(h, dtype=torch.float32)))
            est["FDl32"][h] = float(fd_loss(loss32, th32, v32, torch.tensor(h, dtype=torch.float32)))
            est["FDg64"][h] = float(fd_grad(loss64, th64, v64, torch.tensor(h, dtype=torch.float64)))
            est["FDl64"][h] = float(fd_loss(loss64, th64, v64, torch.tensor(h, dtype=torch.float64)))
        est["AD32"]["all"] = ad32
        rel = {}
        for name, d in est.items():
            if name == "AD32":
                rel[name] = {"all": abs(ad32 - ref) / abs(ref)}
            else:
                rel[name] = {str(h): abs(val - ref) / abs(ref) for h, val in d.items()}
        per_seed.append(dict(seed=seed, reference=ref, rel_error=rel))

    def agg(name, key):
        vals = np.array([s["rel_error"][name][key] for s in per_seed])
        return dict(median=float(np.median(vals)),
                    q25=float(np.percentile(vals, 25)), q75=float(np.percentile(vals, 75)))

    summary = {"AD32_float32_jvp": agg("AD32", "all")}
    shape = {}
    for name in ("FDg32", "FDl32", "FDg64", "FDl64"):
        rows = {}
        for h in hs:
            rows[f"{h:.0e}"] = agg(name, str(h))
        summary[name] = rows
        meds = np.array([rows[k]["median"] for k in rows])
        hs_arr = np.array(hs)
        imin = int(np.argmin(meds))
        left_big = meds[0] >= 3 * meds[imin]
        right_big = meds[-1] >= 3 * meds[imin]
        if 0 < imin < len(meds) - 1 and left_big and right_big:
            shape[name] = dict(verdict="U", h_star=float(hs_arr[imin]), min_median_rel_err=float(meds[imin]))
        elif meds.max() <= 3 * meds.min():
            shape[name] = dict(verdict="plateau", median_range=[float(meds.min()), float(meds.max())])
        else:
            shape[name] = dict(verdict="monotone_or_mixed", h_at_min=float(hs_arr[imin]),
                               min_median_rel_err=float(meds[imin]))

    ref_arr = np.array(ref_vals)
    out = dict(
        device=DEVICE,
        versions=dict(torch=torch.__version__, numpy=np.__version__, python=sys.version.split()[0]),
        threads=torch.get_num_threads(),
        mps_available=bool(torch.backends.mps.is_available()),
        setup=dict(problem="u'' = -pi^2 sin(pi x), u(0)=u(1)=0", hidden=HID, n_params=n_params(),
                   n_colloc=N_COLLOC, bc_weight=LAM_BC, n_seeds=N_SEEDS,
                   h_values=["1", "3e-1"] + [f"1e-{k}" for k in H_EXPONENTS],
                   reference="float64 forward-over-reverse (jvp of grad)",
                   direction="unit random vector per seed"),
        reference_stats=dict(mean=float(ref_arr.mean()), sd=float(ref_arr.std(ddof=1)),
                             min=float(ref_arr.min()), max=float(ref_arr.max())),
        summary=summary,
        shape_verdict=shape,
        per_seed=per_seed,
        total_wall_s=round(time.perf_counter() - t_all, 2),
    )
    (HERE / "extra_pinn_hessian_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("reference mean", out["reference_stats"]["mean"], "sd", out["reference_stats"]["sd"])
    print("AD32 median rel err", summary["AD32_float32_jvp"]["median"],
          "IQR", summary["AD32_float32_jvp"]["q25"], summary["AD32_float32_jvp"]["q75"])
    for name in ("FDg32", "FDl32", "FDg64", "FDl64"):
        print(name, shape[name])
        for k, v in summary[name].items():
            print(f"   h={k} median {v['median']:.3e}  IQR [{v['q25']:.3e}, {v['q75']:.3e}]")
    print("total wall", out["total_wall_s"], "s")


if __name__ == "__main__":
    main()
