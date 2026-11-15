"""토이 2: 논문 3.2.5절 DreamZero-Flash의 방향 검사. 같은 시드, 같은 모델 구조, 같은 반복 수에서
  coupled(영상과 행동이 같은 t를 씀)와 decoupled(영상 t = 1 - η, η ~ Beta(7,1); 행동 t ~ U)을 비교하고, 대조군으로 독립 균등(indep)을 둔다.
추론은 영상을 부분 잡음 수준(t_v = 0.125, 논문의 E[t_v])에 고정하고 행동만 1단계 또는 4단계로 적분한다.

세계(가정): 영상 잠재 z1 ∈ R^4 ~ N(0, I), 행동 a1 = tanh(W z1) + 0.05 잡음 (W 고정, 비선형 역동역학 역할).
모델: MLP 입력 [z_t, a_t, t_v, t_a] → 속도 (v_z, v_a), 은닉 2층 × 128 tanh, Adam, 12000 반복, 배치 256.
평가: 테스트 행동의 MSE(a1 대비). 시드 5개씩, 평균과 표준오차, Welch t 검정.
한계: 장난감 세계이며 실제 영상 DiT와 무관하다. 방향만 본다. 결과를 논문 수치의 검증으로 쓰지 않는다.
실행: python3 analysis/toy_decoupled_noise.py  (numpy, 단일 스레드 약 3–4분) → analysis/toy_decoupled_noise_results.json
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("toy_decoupled_noise_results.json")
DZ, DA, H = 4, 2, 128
N_ITER, BATCH, LR = 12000, 256, 2e-3
SEEDS = [0, 1, 2, 3, 4]
W_TRUE = np.random.default_rng(123).standard_normal((DA, DZ)) * 0.8


def world(rng, n):
    z1 = rng.standard_normal((n, DZ))
    a1 = np.tanh(z1 @ W_TRUE.T) + 0.05 * rng.standard_normal((n, DA))
    return z1, a1


class MLP:
    def __init__(self, rng, din, dout):
        s = lambda a, b: rng.standard_normal((a, b)) / np.sqrt(a)
        self.p = [s(din, H), np.zeros(H), s(H, H), np.zeros(H), s(H, dout), np.zeros(dout)]
        self.m = [np.zeros_like(x) for x in self.p]
        self.v = [np.zeros_like(x) for x in self.p]
        self.t = 0

    def forward(self, X):
        W1, b1, W2, b2, W3, b3 = self.p
        h1 = np.tanh(X @ W1 + b1)
        h2 = np.tanh(h1 @ W2 + b2)
        return h2 @ W3 + b3, (X, h1, h2)

    def step(self, X, Y):
        out, (X0, h1, h2) = self.forward(X)
        g = 2 * (out - Y) / (len(X) * out.shape[1])
        W1, b1, W2, b2, W3, b3 = self.p
        gW3 = h2.T @ g; gb3 = g.sum(0)
        dh2 = (g @ W3.T) * (1 - h2 ** 2)
        gW2 = h1.T @ dh2; gb2 = dh2.sum(0)
        dh1 = (dh2 @ W2.T) * (1 - h1 ** 2)
        gW1 = X0.T @ dh1; gb1 = dh1.sum(0)
        grads = [gW1, gb1, gW2, gb2, gW3, gb3]
        self.t += 1
        for i, gr in enumerate(grads):
            self.m[i] = 0.9 * self.m[i] + 0.1 * gr
            self.v[i] = 0.999 * self.v[i] + 0.001 * gr * gr
            mh = self.m[i] / (1 - 0.9 ** self.t)
            vh = self.v[i] / (1 - 0.999 ** self.t)
            self.p[i] -= LR * mh / (np.sqrt(vh) + 1e-8)
        return float(((out - Y) ** 2).mean())


def build_batch(rng, scheme, n):
    z1, a1 = world(rng, n)
    z0 = rng.standard_normal((n, DZ)); a0 = rng.standard_normal((n, DA))
    if scheme == "coupled":
        t_v = rng.uniform(0, 1, (n, 1)); t_a = t_v.copy()
    elif scheme == "indep":  # 대조군: 두 시점 모두 U(0,1), 독립. Beta 편향 없이 대각선 바깥도 학습
        t_v = rng.uniform(0, 1, (n, 1)); t_a = rng.uniform(0, 1, (n, 1))
    else:  # decoupled: t_v = 1 - eta, eta ~ Beta(7, 1); t_a ~ U
        t_v = 1 - rng.beta(7, 1, (n, 1)); t_a = rng.uniform(0, 1, (n, 1))
    z_t = t_v * z1 + (1 - t_v) * z0
    a_t = t_a * a1 + (1 - t_a) * a0
    X = np.hstack([z_t, a_t, t_v, t_a])
    Y = np.hstack([z1 - z0, a1 - a0])
    return X, Y


def train(seed, scheme):
    rng = np.random.default_rng(1000 + seed)
    model = MLP(rng, DZ + DA + 2, DZ + DA)
    for _ in range(N_ITER):
        X, Y = build_batch(rng, scheme, BATCH)
        model.step(X, Y)
    return model


def act_mse(model, seed, t_v_inf, n_steps):
    """영상은 t_v_inf에 고정, 행동만 n_steps Euler 단계로 0 → 1 적분."""
    rng = np.random.default_rng(5000 + seed)
    n = 4000
    z1, a1 = world(rng, n)
    z0 = rng.standard_normal((n, DZ)); a = rng.standard_normal((n, DA))
    z_t = t_v_inf * z1 + (1 - t_v_inf) * z0
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_a = np.full((n, 1), i * dt)
        X = np.hstack([z_t, a, np.full((n, 1), t_v_inf), t_a])
        out, _ = model.forward(X)
        a = a + dt * out[:, DZ:]
    return float(((a - a1) ** 2).mean())


res = {"assumptions": [
    "세계: z1 ~ N(0, I_4), a1 = tanh(W z1) + 0.05 잡음 (W 고정, 시드 123)",
    "모델 MLP 2×128 tanh, Adam(lr 2e-3), 12000 반복, 배치 256 (시드마다 독립 초기화)",
    "추론: 영상 t_v = 0.125 고정 (논문 E[t_video] = 0.125), 행동 1단계 또는 4단계 Euler",
    "결과는 시드 5개의 행동 MSE 평균과 표준오차. 시드 간 분산만 반영하고 테스트 샘플 분산은 무시",
    "추가 기준: 영상이 깨끗할 때(t_v = 1) 1단계 행동 MSE. 주의: coupled는 t_v = t_a인 대각선만 학습하므로 (0.125, 0)이나 (1, 0) 같은 추론 상태는 외삽이다",
    "대조군 indep: t_v, t_a 모두 U(0,1) 독립. decoupled와의 차이가 Beta 편향 때문인지 대각선 외 학습 범위 때문인지 가른다"],
    "runs": {}, "tests": {}}
store = {}
for scheme in ("coupled", "decoupled", "indep"):
    for seed in SEEDS:
        model = train(seed, scheme)
        store.setdefault(scheme, []).append({
            "1step_video_0.125": act_mse(model, seed, 0.125, 1),
            "4step_video_0.125": act_mse(model, seed, 0.125, 4),
            "1step_video_clean": act_mse(model, seed, 1.0, 1),
        })
for scheme, rows in store.items():
    res["runs"][scheme] = {}
    for key in rows[0]:
        vals = np.array([r[key] for r in rows])
        res["runs"][scheme][key] = {"mean": round(float(vals.mean()), 4),
                                    "se": round(float(vals.std(ddof=1) / np.sqrt(len(vals))), 4),
                                    "per_seed": [round(float(v), 4) for v in vals]}
for key in rows[0]:
    for other in ("coupled", "indep"):
        a = np.array([r[key] for r in store[other]])
        b = np.array([r[key] for r in store["decoupled"]])
        t, p = stats.ttest_ind(b, a, equal_var=False)
        res["tests"].setdefault(key, {})[f"decoupled_minus_{other}"] = {"diff": round(float(b.mean() - a.mean()), 4), "t": round(float(t), 2), "p_two_sided": round(float(p), 4)}
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", OUT)
for k in rows[0]:
    print(k, res["runs"]["coupled"][k]["mean"], res["runs"]["decoupled"][k]["mean"], res["tests"][k])
