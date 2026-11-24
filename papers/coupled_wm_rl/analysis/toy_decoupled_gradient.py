"""토이: 분리(decoupled) 기울기가 국소 모델 단독 롤아웃보다 참 기울기와 더 잘 맞는가?
논문 부록 A.3 (G1에서 코사인 0.52 대 0.35 대 0.08)의 방향성만 저차원 토이로 확인한다. 논문 수치의 재현이 아니다.

설정(전부 가정): 참 동역학 f는 미분 가능한 비선형 8차원 상태, 2차원 행동. 정책은 tanh(W s). 보상은 -|s-g|^2.
- 전역 모델 f_fwd = 참 f (높은 정확도 가정). 국소 모델 f_phi = 작은 MLP를 1스텝 예측으로만 학습(다양한 학습 크기).
- 참 기울기: 참 f를 통과한 역전파.
- (A) 국소 단독: 국소 모델로 굴려 국소 모델 안에서 역전파(Dreamer식).
- (B) 분리(DMO): 전방은 참 f 궤적, 야코비안은 그 궤적의 상태에서 f_phi로 평가.
- (C) 분리 + 노이즈 전방: 전방 모델이 참 f에 작은 오차를 가진 경우.
측정: 초기 상태 64개에 대해 W의 기울기 벡터와 참 기울기의 코사인(평균), 상대 노름 오차. 호라이즌 H를 바꿔 본다.
실행: python3 analysis/toy_decoupled_gradient.py   (torch CPU, 2분 미만)
"""
import json, os, time
import torch

torch.manual_seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
ds, da = 8, 2
dt = 0.1
Bm = torch.randn(ds, da) * 0.8
Am = torch.randn(ds, ds) * 0.25
Am = Am - 0.8 * torch.eye(ds)
C = torch.randn(ds, ds) * 0.35
goal = torch.randn(ds)

def f_true(s, a):
    """비선형: 선형 + 곱셈 결합 + 사인(국소 모델이 장기로는 틀리기 쉬운 항)"""
    lin = s @ Am.T + a @ Bm.T
    nl = torch.sin(3.0 * (s @ C.T)) * 0.6 + 0.3 * torch.tanh(s) * torch.roll(torch.tanh(s), 1, -1)
    return s + dt * (lin + nl)

def reward(s):
    return -((s - goal) ** 2).sum(-1)

class Local(torch.nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(ds + da, h), torch.nn.SiLU(), torch.nn.Linear(h, h), torch.nn.SiLU(), torch.nn.Linear(h, ds))
    def forward(self, s, a):
        return s + self.net(torch.cat([s, a], -1))

def train_local(n_data, steps, h=64, seed=1):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(n_data, ds, generator=g) * 1.0
    a = torch.rand(n_data, da, generator=g) * 2 - 1
    y = f_true(s, a)
    m = Local(h)
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    for _ in range(steps):
        idx = torch.randint(0, n_data, (256,), generator=g)
        loss = ((m(s[idx], a[idx]) - y[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        se = torch.randn(4096, ds, generator=g); ae = torch.rand(4096, da, generator=g) * 2 - 1
        err = ((m(se, ae) - f_true(se, ae)) ** 2).mean().sqrt().item()
    return m, err

def policy(W, s):
    return torch.tanh(s @ W.T)

def ret_true(W, s0, H):
    s = s0; R = 0.0
    for t in range(H):
        a = policy(W, s); s = f_true(s, a); R = R + (0.97 ** t) * reward(s)
    return R.sum()

def ret_local(W, s0, H, m):
    s = s0; R = 0.0
    for t in range(H):
        a = policy(W, s); s = m(s, a); R = R + (0.97 ** t) * reward(s)
    return R.sum()

def ret_dmo(W, s0, H, m, fwd_noise=0.0, gen=None):
    """전방은 f_true(+노이즈) 상태를 쓰고, 기울기 경로는 f_phi의 야코비안을 거친다."""
    s = s0; R = 0.0
    for t in range(H):
        a = policy(W, s)
        with torch.no_grad():
            nxt = f_true(s, a)
            if fwd_noise > 0:
                nxt = nxt + fwd_noise * torch.randn(nxt.shape, generator=gen)
        loc = m(s, a)
        s = nxt + (loc - loc.detach())      # 값은 전방 모델, 야코비안은 국소 모델
        R = R + (0.97 ** t) * reward(s)
    return R.sum()

def grads(fn, W):
    W = W.clone().requires_grad_(True)
    out = fn(W)
    g, = torch.autograd.grad(out, W)
    return g.flatten()

def stat(fn_est, fn_true, W, n_init=64, seed=3):
    g = torch.Generator().manual_seed(seed)
    cos, rel = [], []
    for i in range(n_init):
        s0 = (torch.randn(1, ds, generator=g) * 1.0)
        gt = grads(lambda W_: fn_true(W_, s0), W)
        ge = grads(lambda W_: fn_est(W_, s0), W)
        cos.append(torch.dot(gt, ge) / (gt.norm() * ge.norm() + 1e-12))
        rel.append((ge - gt).norm() / (gt.norm() + 1e-12))
    return float(torch.stack(cos).mean()), float(torch.stack(rel).median())

t0 = time.time()
W = torch.randn(da, ds) * 0.5
res = {"설정": {"ds": ds, "da": da, "dt": dt, "초기 상태 수": 64, "정책 W 시드": 0}, "결과": []}
for label, n_data, steps, hid in [("정확(은닉 64, n=20000, 3000 스텝)", 20000, 3000, 64), ("중간(은닉 16, n=2000, 800 스텝)", 2000, 800, 16), ("부정확(은닉 8, n=300, 200 스텝)", 300, 200, 8)]:
    m, err = train_local(n_data, steps, hid)
    row = {"국소 모델": label, "1스텝 RMSE(분포 내)": round(err, 4), "호라이즌별": {}}
    for H in [4, 8, 16, 32, 64]:
        gen = torch.Generator().manual_seed(7)
        ca, ra = stat(lambda W_, s0: ret_local(W_, s0, H, m), lambda W_, s0: ret_true(W_, s0, H), W)
        cb, rb = stat(lambda W_, s0: ret_dmo(W_, s0, H, m), lambda W_, s0: ret_true(W_, s0, H), W)
        cc, rc = stat(lambda W_, s0: ret_dmo(W_, s0, H, m, 0.02, gen), lambda W_, s0: ret_true(W_, s0, H), W)
        row["호라이즌별"][str(H)] = {"A 국소 단독 코사인": round(ca, 3), "A 상대오차(중앙값)": round(ra, 3),
                                  "B 분리 코사인": round(cb, 3), "B 상대오차(중앙값)": round(rb, 3),
                                  "C 분리+전방노이즈0.02 코사인": round(cc, 3), "C 상대오차": round(rc, 3)}
    res["결과"].append(row)
    print(label, err, time.time() - t0, flush=True)
res["소요 초"] = round(time.time() - t0, 1)
with open(os.path.join(HERE, "toy_decoupled_gradient_results.json"), "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
