"""토이: BitVLA의 W1.58A8 선형층 양자화(논문 식 1–4)를 난수 가중치와 활성으로 확인한다.

실행: python3 analysis/toy_quant.py   → analysis/toy_quant_results.json (numpy, 1초 미만)

무엇을 확인하나
  Q1  식 (4)를 글자 그대로 쓴 Y = (127 α / β) Qw(W) Qa(x) 와 역양자화 계수 α β / 127 를 쓴 값을 비교한다.
      식 (1)–(2)의 정의(Qw = RoundClip(W/α), Qa = RoundClip(127x/β))에서 원래 값으로 돌아오려면
      곱해야 하는 계수는 α β / 127 이다. 두 계수의 비는 (127/β)^2 이다.
  Q2  W1.58A8의 상대 오차를 W만, A만 양자화한 경우와 나눠 본다(활성 이상치 채널 유무 포함).
  Q3  가중치 분포별 영(0) 비율과 엔트로피(비트/가중치). 1.58 = log2(3)은 균등 분포일 때의 상한이다.
  Q4  2비트 묶음(공개 코드 quantize_to_int2와 같은 구조)의 왕복 정확성과, 5개 삼진수를 한 바이트에 넣는
      묶음(3^5 = 243 <= 256)의 비트/가중치.

한계: 난수 가중치다. 학습된 BitNet 가중치의 분포와 이상치는 다를 수 있다. 결과는 실제 모델의 오차가 아니다.
공개 코드의 함수는 가져오거나 import하지 않았고, 같은 수식을 numpy로 새로 썼다.
(공식 코드: https://github.com/ustcwhy/BitVLA, 정적 읽기만 함)
"""
import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "toy_quant_results.json"
rng = np.random.default_rng(0)


def qw(W):
    alpha = np.abs(W).mean()
    return np.clip(np.round(W / alpha), -1, 1), alpha


def qa(X):
    beta = np.abs(X).max(axis=-1, keepdims=True)  # 토큰별 absmax (식 2의 β)
    return np.clip(np.round(127 * X / beta), -128, 127), beta


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def cos(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))


M, N, T = 512, 1024, 256
res = {}

# ---- Q1, Q2
cases = {}
for label, outlier in (("가우시안 활성", False), ("이상치 채널 1% x 30", True)):
    W = rng.normal(0, 0.02, size=(M, N))
    X = rng.normal(0, 1.0, size=(T, N))
    if outlier:
        idx = rng.choice(N, size=N // 100, replace=False)
        X[:, idx] *= 30.0
    Y0 = X @ W.T
    Wq, alpha = qw(W)
    Xq, beta = qa(X)
    core = Xq @ Wq.T                               # 정수 곱셈 부분
    y_correct = (alpha * beta / 127.0) * core       # α β / 127
    y_literal = (127.0 * alpha / beta) * core       # 식 (4) 그대로
    y_w_only = X @ (Wq * alpha).T
    y_a_only = (Xq * beta / 127.0) @ W.T
    ratio = (y_literal[np.abs(y_correct) > 1e-9] / y_correct[np.abs(y_correct) > 1e-9])
    cases[label] = {
        "rel_err_W1.58A8": rel(y_correct, Y0), "cos_W1.58A8": cos(y_correct, Y0),
        "rel_err_W1.58_only": rel(y_w_only, Y0), "rel_err_A8_only": rel(y_a_only, Y0),
        "rel_err_literal_eq4": rel(y_literal, Y0),
        "literal_over_correct_median": float(np.median(ratio)),
        "expected_(127/beta)^2_median": float(np.median((127.0 / beta) ** 2)),
        "alpha": float(alpha), "beta_median": float(np.median(beta)),
        "zero_fraction": float((Wq == 0).mean()),
    }
res["Q1_Q2_linear_layer"] = {"shape": {"out": M, "in": N, "tokens": T}, "cases": cases}

# ---- Q3 영 비율과 엔트로피
def entropy(p):
    p = np.array([x for x in p if x > 0])
    return float(-(p * np.log2(p)).sum())


dist = {}
for name, gen in (("가우시안", lambda n: rng.normal(0, 1, n)),
                  ("라플라스", lambda n: rng.laplace(0, 1, n)),
                  ("균등(-1,1)", lambda n: rng.uniform(-1, 1, n)),
                  ("학생 t(3)", lambda n: rng.standard_t(3, n))):
    W = gen(2_000_000)
    q, a = qw(W)
    p = [float((q == v).mean()) for v in (-1, 0, 1)]
    dist[name] = {"p(-1),p(0),p(+1)": [round(x, 4) for x in p], "entropy_bits": round(entropy(p), 4),
                  "log2(3)": round(math.log2(3), 4)}
res["Q3_ternary_stats"] = dist

# ---- Q4 묶음
def pack2(q):  # 공개 코드의 quantize_to_int2와 같은 구조: 값 + 1 을 2비트로, 4개를 한 바이트에
    u = (q.astype(np.int8) + 1).astype(np.uint8).ravel()
    pad = (-u.size) % 4
    u = np.concatenate([u, np.zeros(pad, dtype=np.uint8)]).reshape(-1, 4)
    return (u[:, 0] | (u[:, 1] << 2) | (u[:, 2] << 4) | (u[:, 3] << 6)).astype(np.uint8), pad


def unpack2(b, n):
    u = np.stack([b & 3, (b >> 2) & 3, (b >> 4) & 3, (b >> 6) & 3], axis=1).ravel()[:n]
    return u.astype(np.int8) - 1


def pack243(q):  # 5개 삼진수 -> 0..242 한 바이트
    u = (q.astype(np.int16) + 1).ravel()
    pad = (-u.size) % 5
    u = np.concatenate([u, np.zeros(pad, dtype=np.int16)]).reshape(-1, 5)
    w = np.array([1, 3, 9, 27, 81], dtype=np.int16)
    return (u * w).sum(axis=1).astype(np.uint8), pad


def unpack243(b, n):
    v = b.astype(np.int16)
    digits = []
    for _ in range(5):
        digits.append(v % 3)
        v = v // 3
    return (np.stack(digits, axis=1).ravel()[:n] - 1).astype(np.int8)


W = rng.normal(0, 0.02, size=(2560, 6913))  # 4로도 5로도 나누어떨어지지 않는 크기(패딩 경로 확인)
q, _ = qw(W)
b2, pad2 = pack2(q)
b5, pad5 = pack243(q)
n = q.size
res["Q4_packing"] = {
    "n_weights": int(n),
    "pack2_roundtrip_exact": bool(np.array_equal(unpack2(b2, n).reshape(q.shape), q.astype(np.int8))),
    "pack2_bits_per_weight": round(b2.size * 8 / n, 4),
    "pack243_roundtrip_exact": bool(np.array_equal(unpack243(b5, n).reshape(q.shape), q.astype(np.int8))),
    "pack243_bits_per_weight": round(b5.size * 8 / n, 4),
    "ideal_log2_3": round(math.log2(3), 4),
    "llm_linear_params_(공개 config 기준)": 2_084_044_800,
    "llm_linear_gb_2bit": round(2_084_044_800 * 2 / 8 / 1e9, 3),
    "llm_linear_gb_1.6bit(243묶음)": round(2_084_044_800 * 1.6 / 8 / 1e9, 3),
    "llm_linear_gb_1.58bit(이론)": round(2_084_044_800 * math.log2(3) / 8 / 1e9, 3),
}

OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
for label, c in cases.items():
    print(label, {k: round(v, 4) for k, v in c.items()})
print(json.dumps(res["Q3_ternary_stats"], ensure_ascii=False))
print(json.dumps(res["Q4_packing"], ensure_ascii=False))
