"""토이 2: 접촉 구간 업샘플링(FIRST의 가중 샘플링)이 단계별 행동 오차에 주는 효과.

설정(가정, 논문 값 아님):
  - 시연 한 편 T=400 스텝(100 Hz 가정). 접촉 시작 t_c 무작위(150–300).
  - PC(접촉 전 F=100 스텝), C(접촉 후 60 스텝), F(나머지 자유 운동).
  - 상태 s=(s1, s2)는 정규분포. 행동 a는 단계마다 다른 비선형 함수 + 잡음.
  - 정책: 무작위 Fourier 특징(D=150) + 가중 리지 회귀. 샘플 가중치 w_PC, w_C는 업샘플링의 기댓값 형태.
  - 평가: 단계별 테스트 MSE (노이즈 없는 목표값 기준).
출력: analysis/toy_upsampling_results.json. 실행 약 10초.
주의: 장난감이다. 결과의 방향(어느 수준에서 포화되는지)만 보고, 수치를 논문과 비교하지 않는다.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("toy_upsampling_results.json")
rng = np.random.default_rng(1)
F_PRE, L_C, T = 100, 60, 400
D = 150


def make_demos(n_demo):
    S, Y, PH = [], [], []
    for _ in range(n_demo):
        tc = rng.integers(150, 300)
        s = rng.normal(size=(T, 2))
        ph = np.zeros(T, dtype=int)  # 0 F, 1 PC, 2 C
        ph[tc - F_PRE:tc] = 1
        ph[tc:tc + L_C] = 2
        a = np.zeros((T, 1))
        f = ph == 0
        p = ph == 1
        c = ph == 2
        a[f, 0] = -0.5 * s[f, 0]
        a[p, 0] = -1.5 * s[p, 0] + 0.8 * np.sin(4 * s[p, 1])
        a[c, 0] = np.tanh(3 * s[c, 1]) - 0.5 * s[c, 0] * s[c, 1]
        a += rng.normal(0, 0.05, size=a.shape)
        S.append(s); Y.append(a[:, 0]); PH.append(ph)
    return np.concatenate(S), np.concatenate(Y), np.concatenate(PH)


def clean_target(s, ph):
    y = np.zeros(len(s))
    f, p, c = ph == 0, ph == 1, ph == 2
    y[f] = -0.5 * s[f, 0]
    y[p] = -1.5 * s[p, 0] + 0.8 * np.sin(4 * s[p, 1])
    y[c] = np.tanh(3 * s[c, 1]) - 0.5 * s[c, 0] * s[c, 1]
    return y


W_rff = rng.normal(size=(2, D))
b_rff = rng.uniform(0, 2 * np.pi, D)


def phi(s):
    return np.sqrt(2 / D) * np.cos(s @ W_rff + b_rff)


def wridge(X, y, w, lam=1e-2):
    A = X.T @ (X * w[:, None]) + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ (w * y))


S_tr, Y_tr, PH_tr = make_demos(20)
rng = np.random.default_rng(2)  # 테스트 집합용 시드
S_te, Y_te, PH_te = make_demos(30)
Y_true_te = clean_target(S_te, PH_te)
X_tr, X_te = phi(S_tr), phi(S_te)

res = {"setting": dict(F_pre=F_PRE, L_c=L_C, D=D, demos_train=20, demos_test=30, lam=1e-2),
       "n_train_by_phase": {int(k): int(v) for k, v in zip(*np.unique(PH_tr, return_counts=True))}}
rows = []
for scheme, wpc, wc in [("none", 1, 1)] + [("PC", m, 1) for m in (2, 5, 10, 20)] + \
        [("C_only", 1, m) for m in (2, 5, 10)] + [("PC+C", m, m) for m in (2, 5, 10)]:
    w = np.ones(len(PH_tr))
    w[PH_tr == 1] = wpc
    w[PH_tr == 2] = wc
    wf = wridge(X_tr, Y_tr, w)
    pred = X_te @ wf
    mse = {}
    for name, k in [("F", 0), ("PC", 1), ("C", 2)]:
        m = PH_te == k
        mse[name] = float(np.mean((pred[m] - Y_true_te[m]) ** 2))
    rows.append({"scheme": scheme, "w_PC": wpc, "w_C": wc, **{f"mse_{k}": round(v, 5) for k, v in mse.items()}})
res["rows"] = rows
OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
for r in rows:
    print(r)
