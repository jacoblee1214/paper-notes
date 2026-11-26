"""토이: 원시 행동 / 명목(nominal) 궤적 / 기록된 실현(logged) 궤적 중 무엇을 조건으로 줄 때 장면 결과를 얼마나 잘 맞히나.

논문 4.3절의 표 3(원시 행동 대 명목)과 표 4(logged 대 nominal)가 말하는 두 가지를 아주 작게 흉내 낸다.
  1) 명목 궤적은 원시 행동의 결정적 함수다(식 1). 정보량은 같고, 학습 쉬움(표본 효율)만 다를 수 있다.
  2) 기록된 실현 궤적은 접촉 결과(마찰에 의한 감속)를 담는다. 결과를 그대로 흘리는 셈이다.

모형(우리가 만든 것, 논문의 실험과 무관):
  - 1차원 이펙터가 속도 제한 추종기로 목표 a_t를 따라간다. 블록은 위치 xb에 있고, 이펙터가 닿으면 밀려간다.
  - 접촉 중 이펙터는 숨은 마찰 mu에 비례해 느려진다. 결과 y = 블록 최종 이동량.
  - 조건 후보: raw = 목표열 a, nominal = 접촉 없는 추종 결과 q, logged = 접촉 포함 추종 결과.
    세 조건 모두 블록 위치 xb(장면 문맥)는 함께 준다. mu는 아무도 모른다.
  - 예측기: 은닉 64 두 층 MLP. 학습 표본 수 N을 바꿔 학습 곡선을 본다.

주의: logged가 이기는 것은 구성상 당연하다(실현 궤적이 결과를 거의 결정). 이 토이는 방향의 직관을 보일 뿐
논문 수치의 근거가 아니다. 실행: python3 analysis/toy_leakage.py (torch, CPU, 약 1분) → analysis/toy_leakage_results.json
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent / "toy_leakage_results.json"
T, VMAX, R_CONTACT = 32, 0.035, 0.05


def simulate(n, rng):
    xb = rng.uniform(0.35, 0.75, n)
    mu = rng.uniform(0.2, 0.9, n)
    g = rng.uniform(0.5, 1.0, n)
    ramp = rng.integers(1, 6, n)               # 목표가 g에 도달하는 데 걸리는 스텝(짧을수록 추종기보다 앞선다)
    a = np.minimum(1.0, (np.arange(1, T + 1)[None, :] / ramp[:, None])) * g[:, None]
    qn = np.zeros((n, T + 1))
    qr = np.zeros((n, T + 1))
    b = xb.copy()
    for t in range(T):
        qn[:, t + 1] = qn[:, t] + np.clip(0.5 * (a[:, t] - qn[:, t]), -VMAX, VMAX)
        step = np.clip(0.5 * (a[:, t] - qr[:, t]), -VMAX, VMAX)
        touching = qr[:, t] + R_CONTACT >= b
        step = np.where(touching, step * (1 - 0.6 * mu), step)
        qr[:, t + 1] = qr[:, t] + step
        b = np.maximum(b, qr[:, t + 1] + R_CONTACT)
    y = b - xb
    return xb, a, qn[:, 1:], qr[:, 1:], y


def make_features(kind, xb, a, qn, qr):
    seq = {"raw": a, "nominal": qn, "logged": qr}[kind]
    return np.concatenate([xb[:, None], seq], axis=1).astype(np.float32)


def fit_eval(Xtr, ytr, Xte, yte, seed, epochs=600):
    torch.manual_seed(seed)
    mu_x, sd_x = Xtr.mean(0), Xtr.std(0) + 1e-6
    ym, ys = ytr.mean(), ytr.std() + 1e-6
    f = lambda X: torch.tensor((X - mu_x) / sd_x)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 64), torch.nn.Tanh(),
                              torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
    xt, yt = f(Xtr), torch.tensor(((ytr - ym) / ys).astype(np.float32))[:, None]
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(xt), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = net(f(Xte)).numpy()[:, 0] * ys + ym
    rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
    r2 = float(1 - np.sum((pred - yte) ** 2) / np.sum((yte - yte.mean()) ** 2))
    return rmse, r2


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    te = simulate(2000, rng)
    sizes, seeds = [32, 128, 512, 2048], [0, 1, 2]
    res = {"sizes": sizes, "seeds": seeds, "T": T, "metric": "test RMSE of block displacement (m)", "runs": {}}
    # 정보량 점검: 명목이 원시 행동의 함수임을 확인(같은 행동이면 같은 명목)
    xb, a, qn, qr, y = te
    res["info_check_corr_y_with_final_q"] = {
        "corr(y, final raw target)": float(np.corrcoef(y, a[:, -1])[0, 1]),
        "corr(y, final nominal q)": float(np.corrcoef(y, qn[:, -1])[0, 1]),
        "corr(y, final logged q)": float(np.corrcoef(y, qr[:, -1])[0, 1]),
    }
    res["y_sd"] = float(y.std())
    for N in sizes:
        for kind in ("raw", "nominal", "logged"):
            rmses, r2s = [], []
            for s in seeds:
                tr = simulate(N, np.random.default_rng(100 + s))
                Xtr, Xte = make_features(kind, *tr[:4]), make_features(kind, *te[:4])
                rmse, r2 = fit_eval(Xtr, tr[4], Xte, te[4], seed=s)
                rmses.append(rmse)
                r2s.append(r2)
            res["runs"][f"{kind}_N{N}"] = {"rmse_mean": float(np.mean(rmses)), "rmse_sd": float(np.std(rmses)),
                                          "r2_mean": float(np.mean(r2s))}
            print(f"N={N:5d} {kind:8s} rmse={np.mean(rmses):.4f}±{np.std(rmses):.4f} r2={np.mean(r2s):.3f}")
    res["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT, res["seconds"], "s")


if __name__ == "__main__":
    main()
