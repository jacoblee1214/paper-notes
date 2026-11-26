"""WAM-TTT 부록 A의 "선형 어텐션 증인"과 표 B.1의 inner-loop 설정을 소형 CPU 토이로 점검한다.

검사 (논문 값: 헤드 차원 d = 48, 고속 가중치 은닉 f_h = 128, inner SGD N = 1,
              inner LR 학습 0.1 / 시험 0.01, 식 3의 평균 정규화 1/(B L_h d)):
  T1. 식 A.2 닫힌 해, 식 A.3 외적(헤비안) 근사, softmax 어텐션 읽기를 비교한다.
      키가 등방(K^T K ≈ (L/d) I)일 때와 비등방(고유값이 완만히 감소)일 때를 나눈다.
  T2. 식 3을 한 번 SGD(η = 0.1, 0.01)로 갱신했을 때 읽기 출력이 얼마나 변하는지(선형과 2층 MLP)
      와, 목표 값에 코사인 0.9까지 가까워지는 데 필요한 단계 수를 센다. 값이 키와 무관한 임의 연상(random)과
      키의 선형 함수(structured, 일관된 사람→로봇 대응의 단순화)를 나눈다.

가정: 키/값/질의 원소는 평균 0, 분산 1에 가깝다(투영 뒤 정규화 가정). 사람 토큰 수 L_h는 논문에 없어 {64, 256, 1024}를 훑는다.
값은 키의 근처 질의에서 되찾아야 하는 연상 기억이다(질의 = 키 + 잡음). 한 시드의 토이이며 논문의 모델과 같지 않다.

실행: python3 analysis/toy_fast_weight.py  (CPU, 수십 초) → analysis/toy_fast_weight_results.json
"""
import json
import time
from pathlib import Path

import torch

torch.manual_seed(0)
D, FH = 48, 128
OUT = Path(__file__).resolve().parent / "toy_fast_weight_results.json"


def make_keys(L, aniso):
    z = torch.randn(L, D)
    if aniso:
        # 고유값이 1에서 0.01로 완만히 감소 (조건수 100), 평균 분산은 1로 맞춘다
        s = torch.logspace(0, -1, D)
        s = s / s.pow(2).mean().sqrt()
        q, _ = torch.linalg.qr(torch.randn(D, D))
        z = (z * s) @ q.T
    return z


def attention_readout(K, V, Q, temp):
    a = torch.softmax(Q @ K.T / temp, dim=-1)
    return a @ V


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def t1():
    res = []
    for aniso in (False, True):
        for L in (64, 256, 1024):
            K = make_keys(L, aniso)
            V = torch.randn(L, D)
            Q = K + 0.3 * torch.randn(L, D)
            # 같은 위치의 값을 되찾는 것이 목표이므로, 목표 읽기는 정답 값 V (연상 기억의 이상)
            temp = D ** 0.5
            att = attention_readout(K, V, Q, temp)
            # 닫힌 해 (식 A.2, 안정화를 위해 작은 릿지)
            W = V.T @ K @ torch.linalg.inv(K.T @ K + 1e-6 * torch.eye(D))
            heb = V.T @ K / L * D  # (d/L) V^T K  (식 A.3, 등방일 때 W* 와 같은 방향)
            res.append({
                "aniso": aniso, "L": L,
                "cos_closed_vs_target": cos(Q @ W.T, V),
                "cos_hebb_vs_target": cos(Q @ heb.T, V),
                "cos_softmax_vs_target": cos(att, V),
                "cos_closed_vs_hebb": cos(Q @ W.T, Q @ heb.T),
                "cos_closed_vs_softmax": cos(Q @ W.T, att),
                "cond_KtK": torch.linalg.cond(K.T @ K).item(),
            })
    return res


def lin_loss(W, K, V):
    return ((K @ W.T - V) ** 2).mean()  # 1/(L d) 정규화 (식 3, B = 1)


def mlp_forward(p, x):
    return torch.nn.functional.gelu(x @ p[0].T) @ p[1].T


def t2():
    res = []
    A = torch.randn(D, D) / D ** 0.5
    for L, mode in [(l, m) for m in ("random", "structured") for l in (64, 256, 1024)]:
        K = make_keys(L, True)
        V = torch.randn(L, D) if mode == "random" else K @ A.T
        Q = K + 0.3 * torch.randn(L, D)
        for kind in ("linear", "mlp"):
            for eta in (0.1, 0.01):
                if kind == "linear":
                    p = [torch.randn(D, D) / D ** 0.5]
                    f = lambda pp, x: x @ pp[0].T
                else:
                    p = [torch.randn(FH, D) / D ** 0.5, torch.randn(D, FH) / FH ** 0.5]
                    f = mlp_forward
                p = [t.clone().requires_grad_(True) for t in p]
                base = f(p, Q).detach()
                hist = []
                steps_to_cos = None
                for step in range(1, 2001):
                    loss = ((f(p, K) - V) ** 2).mean()
                    g = torch.autograd.grad(loss, p)
                    with torch.no_grad():
                        for t, gg in zip(p, g):
                            t -= eta * gg
                    if step in (1, 10, 100, 1000, 2000):
                        out = f(p, Q).detach()
                        hist.append({
                            "step": step,
                            "rel_readout_change": ((out - base).norm() / base.norm()).item(),
                            "cos_vs_target_V": cos(out, V),
                            "cos_delta_vs_target_V": cos(out - base, V),
                            "loss": loss.item(),
                        })
                    if steps_to_cos is None and step % 10 == 0 and cos(f(p, Q).detach(), V) > 0.9:
                        steps_to_cos = step
                res.append({"mode": mode, "L": L, "kind": kind, "eta": eta, "hist": hist, "steps_to_cos0.9": steps_to_cos})
    return res


if __name__ == "__main__":
    t0 = time.time()
    out = {"T1": t1(), "T2": t2(), "seconds": None}
    out["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=1))
    for r in out["T1"]:
        print("T1", r)
    for r in out["T2"]:
        h1 = r["hist"][0]
        print("T2", r["mode"], r["L"], r["kind"], r["eta"], "step1 rel_change=%.2e cos_delta=%.3f" % (h1["rel_readout_change"], h1["cos_delta_vs_target_V"]),
              "steps_to_cos0.9=", r["steps_to_cos0.9"])
    print("seconds", out["seconds"])
