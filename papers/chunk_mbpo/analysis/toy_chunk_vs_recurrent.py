"""토이: 청크 월드 모델 대 한 스텝 재귀 모델 (논문 그림 5(b)와 3.3절 설계의 작은 검사).

시스템: 토크로 구동되는 감쇠 이중 진자(해석적 동역학, RK4, dt=0.05). 상태 표현은 [sin, cos, 각속도/3] 6차원(항등 인코더).
비교: (a) 한 스텝 MLP를 H스텝 펼친 손실로 학습한 뒤 재귀 롤아웃, (b) 같은 크기의 청크 MLP(입력 s0 + 행동 H개, 출력 s1..sH).
측정: 예측 스텝 k=1..H의 상태 MSE, 그리고 청크 모델이 s1을 예측할 때 미래 행동(a1..a_{H-1})에 의존하는 정도
      (참 동역학에서는 0, 재귀 모델은 구조상 0).
데이터 행동: iid 균등 / AR(1) 시간 상관(rho=0.9). 시드 3개, 은닉 128x2, 1500 스텝, CPU, 약 1–2분.
사용법: python3 analysis/toy_chunk_vs_recurrent.py          (기본 3개 설정, 약 1.5분)
        python3 analysis/toy_chunk_vs_recurrent.py long     (학습 5000 스텝 한 설정, 약 2분)   (torch 필요)
"""
import json, os, time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
H = 5
DT = 0.05


def dyn(x, u):
    """이중 진자 (질량 1, 길이 1, 중력 9.8, 점성 감쇠 0.3). x=(th1,th2,w1,w2), u=(tau1,tau2)."""
    th1, th2, w1, w2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    g, b = 9.8, 0.3
    c2, s2 = np.cos(th2), np.sin(th2)
    # 단순화한 균일 막대 없는 점질량 이중 진자
    m11 = 3 + 2 * c2
    m12 = 1 + c2
    m22 = np.ones_like(c2)
    h = -s2 * (2 * w1 * w2 + w2 ** 2)
    h2 = s2 * w1 ** 2
    G1 = 2 * g * np.sin(th1) + g * np.sin(th1 + th2)
    G2 = g * np.sin(th1 + th2)
    r1 = u[:, 0] - h - G1 - b * w1
    r2 = u[:, 1] - h2 - G2 - b * w2
    det = m11 * m22 - m12 ** 2
    a1 = (m22 * r1 - m12 * r2) / det
    a2 = (-m12 * r1 + m11 * r2) / det
    return np.stack([w1, w2, a1, a2], 1)


def step(x, u):
    k1 = dyn(x, u); k2 = dyn(x + DT / 2 * k1, u); k3 = dyn(x + DT / 2 * k2, u); k4 = dyn(x + DT * k3, u)
    return x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def feat(x):
    return np.stack([np.sin(x[:, 0]), np.cos(x[:, 0]), np.sin(x[:, 1]), np.cos(x[:, 1]), x[:, 2] / 3, x[:, 3] / 3], 1).astype(np.float32)


def gen(n, rng, corr):
    x = np.stack([rng.uniform(-np.pi, np.pi, n), rng.uniform(-np.pi, np.pi, n), rng.normal(0, 1, n), rng.normal(0, 1, n)], 1)
    U = np.zeros((n, H, 2))
    u = rng.uniform(-4, 4, (n, 2))
    for k in range(H):
        if corr:
            u = 0.9 * u + np.sqrt(1 - 0.81) * rng.uniform(-4, 4, (n, 2)) * 1.7
            u = np.clip(u, -6, 6)
        else:
            u = rng.uniform(-4, 4, (n, 2))
        U[:, k] = u
    S = [feat(x)]
    for k in range(H):
        x = step(x, U[:, k])
        S.append(feat(x))
    return torch.tensor(S[0]), torch.tensor(U.astype(np.float32) / 4), torch.tensor(np.stack(S[1:], 1))


def mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.Mish(), nn.Linear(h, h), nn.Mish(), nn.Linear(h, o))


class Chunk(nn.Module):
    def __init__(self):
        super().__init__(); self.f = mlp(6 + 2 * H, 6 * H)

    def forward(self, s0, U):
        return s0.unsqueeze(1) + self.f(torch.cat([s0, U.flatten(1)], 1)).view(-1, H, 6)  # 잔차형(재귀 모델과 맞춤)


class Rec(nn.Module):
    def __init__(self):
        super().__init__(); self.f = mlp(6 + 2, 6)

    def forward(self, s0, U):
        out, s = [], s0
        for k in range(H):
            s = s + self.f(torch.cat([s, U[:, k]], 1))
            out.append(s)
        return torch.stack(out, 1)


def train(model, data, steps, seed):
    torch.manual_seed(seed)
    s0, U, S = data
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    w = (0.5 ** torch.arange(H)).view(1, H, 1)  # TD-MPC2식 rho^t 가중
    for it in range(steps):
        idx = torch.randint(0, s0.shape[0], (256,))
        loss = (w * (model(s0[idx], U[idx]) - S[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model


def evaluate(model, test):
    s0, U, S = test
    with torch.no_grad():
        err = ((model(s0, U) - S) ** 2).mean((0, 2)).numpy()
        # 미래 행동 민감도: a1..a_{H-1}를 새로 뽑아 바꿨을 때 s1 예측 변화 (a0 고정)
        U2 = U.clone(); U2[:, 1:] = torch.rand_like(U2[:, 1:]) * 2 - 1
        d = ((model(s0, U)[:, 0] - model(s0, U2)[:, 0]) ** 2).mean().item()
        # 비교 기준: 현재 행동 a0를 바꿨을 때 s1 예측 변화
        U3 = U.clone(); U3[:, 0] = torch.rand_like(U3[:, 0]) * 2 - 1
        d0 = ((model(s0, U)[:, 0] - model(s0, U3)[:, 0]) ** 2).mean().item()
    return err, d, d0


def main(steps=1500, long=False):
    t0 = time.time()
    res = {}
    rng = np.random.default_rng(0)
    test_iid = gen(5000, rng, False)
    var = test_iid[2].var((0,)).mean().item()
    cfgs = ((False, 20000),) if long else ((False, 20000), (True, 20000), (False, 2000))
    for corr, n in cfgs:
        if True:
            key = f"{'AR1' if corr else 'iid'}_n{n}_steps{steps}"
            rows = {"chunk": [], "recurrent": []}
            leak = {"chunk": [], "recurrent": []}
            cur = {"chunk": [], "recurrent": []}
            for seed in range(3):
                rng = np.random.default_rng(100 + seed)
                data = gen(n, rng, corr)
                test = gen(5000, np.random.default_rng(999), corr)
                for name, cls in (("chunk", Chunk), ("recurrent", Rec)):
                    torch.manual_seed(seed)
                    m = train(cls(), data, steps, seed)
                    e, d, d0 = evaluate(m, test)
                    rows[name].append(e); leak[name].append(d); cur[name].append(d0)
            res[key] = {nm: dict(mse_by_step_mean=np.mean(rows[nm], 0).tolist(), mse_by_step_sd=np.std(rows[nm], 0).tolist(),
                                 future_action_sensitivity_mean=float(np.mean(leak[nm])), future_action_sensitivity_sd=float(np.std(leak[nm])),
                                 current_action_sensitivity_mean=float(np.mean(cur[nm])))
                        for nm in rows}
            res[key]["ratio_rec_over_chunk_by_step"] = (np.mean(rows["recurrent"], 0) / np.mean(rows["chunk"], 0)).tolist()
            print(key, "chunk", np.round(np.mean(rows["chunk"], 0) * 1e3, 3), "rec", np.round(np.mean(rows["recurrent"], 0) * 1e3, 3),
                  "leak chunk %.2e" % np.mean(leak["chunk"]), f"{time.time() - t0:.0f}s", flush=True)
    res["state_feature_variance_mean"] = var
    json.dump(res, open(os.path.join(HERE, "toy_chunk_vs_recurrent_results%s.json" % ("_long" if long else "")), "w"), indent=1)


if __name__ == "__main__":
    import sys
    if "long" in sys.argv:
        main(steps=5000, long=True)
    else:
        main()
