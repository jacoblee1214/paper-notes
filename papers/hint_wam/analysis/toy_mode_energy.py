"""토이 T1: 숨은 정보 아래의 증류 (보충 J의 장난감 실험을 다시 만든 것).

관측 x ~ U(-1, 1), 숨은 c ∈ {-1, +1}(균등). 올바른 행동 a = c (1 + 0.5 x).
교사는 c를 보고 a를 출력한다(표본 하나). 학생은 c를 못 본다.
학생 f(x, z), z ~ N(0,1), MLP 2층 64 GELU.
  - 제곱오차 증류: 교사 표본 하나에 대한 MSE
  - 에너지 점수 증류: M = 4 학생 표본, (1/M) Σ|f_m - a| - (1/2M^2) Σ|f_m - f_m'|
성공률: 학생 표본이 올바른 두 행동 중 하나의 0.25 이내에 있는 비율(64개 x, x마다 512 표본, 3 시드).

실행: python3 -I analysis/toy_mode_energy.py   (CPU, 1-2분 이내)
출력: analysis/toy_mode_energy_results.json
한계: 논문 보충 J의 설정을 따랐지만 손실 세부(옵티마이저 값 등)는 논문과 같다고만 가정한다.
"""
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

torch.set_num_threads(2)
OUT = Path(__file__).resolve().parent / "toy_mode_energy_results.json"
STEPS, BATCH, LR, M, SEEDS = 3000, 256, 3e-3, 4, [0, 1, 2]


def net():
    return nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))


def sample_batch(n):
    x = torch.rand(n, 1) * 2 - 1
    c = torch.randint(0, 2, (n, 1)).float() * 2 - 1
    a = c * (1 + 0.5 * x)
    return x, a


def student_samples(f, x, m):
    z = torch.randn(x.shape[0], m, 1)
    xx = x[:, None, :].expand(-1, m, -1)
    inp = torch.cat([xx, z], dim=-1).reshape(-1, 2)
    return f(inp).reshape(x.shape[0], m)  # [n, m]


def success(f, n_x=64, n_s=512):
    xs = torch.linspace(-1, 1, n_x)[:, None]
    with torch.no_grad():
        s = student_samples(f, xs, n_s)  # [n_x, n_s]
        right = 1 + 0.5 * xs
        ok = ((s - right).abs() < 0.25) | ((s + right).abs() < 0.25)
        return float(ok.float().mean())


def train(method, seed):
    torch.manual_seed(seed)
    f = net()
    opt = torch.optim.Adam(f.parameters(), lr=LR)
    for _ in range(STEPS):
        x, a = sample_batch(BATCH)
        if method == "squared_error":
            pred = student_samples(f, x, 1)
            loss = ((pred - a) ** 2).mean()
        else:
            s = student_samples(f, x, M)  # [B, M]
            acc = (s - a).abs().mean(dim=1)  # [B]
            div = (s[:, :, None] - s[:, None, :]).abs().mean(dim=(1, 2))  # [B]
            loss = (acc - 0.5 * div).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return success(f)


def main():
    t0 = time.time()
    res = {"settings": {"steps": STEPS, "batch": BATCH, "lr": LR, "M": M, "seeds": SEEDS}}
    for method in ["squared_error", "energy_score"]:
        vals = [train(method, s) for s in SEEDS]
        res[method] = {"success_per_seed": [round(v, 4) for v in vals],
                       "mean": round(sum(vals) / len(vals), 4)}
        print(method, res[method])
    res["elapsed_s"] = round(time.time() - t0, 1)
    res["paper_reference (보충 J)"] = {"squared_error": 0.00, "energy_score": 0.92}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print("elapsed", res["elapsed_s"], "s")


if __name__ == "__main__":
    main()
