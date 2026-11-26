"""토이 1: 새 모달리티를 넣는 두 방식이 초기 출력을 얼마나 흔드는가 (논문 RQ4, 식 1·2).

무작위 가중치의 작은 트랜스포머(8층)를 "사전학습 백본"의 대역으로 쓰고, 액션 토큰 출력이 새 토큰 때문에 바뀌는 크기를 잰다.
  concat : 힘 토큰을 시퀀스에 이어 붙여 기존 self-attention에 통과시킴 (논문의 X-VLA_AFM 방식)
  gca    : 4층마다 교차어텐션 잔차를 더함. 게이트 tanh(alpha) (alpha=0, 식 1) 또는 sigmoid(w·z̄ + b) (식 2, b = 닫힘 정도)
무작위 가중치이므로 절댓값은 예시이고, 구조적 성질(게이트 0이면 정확히 0, 식 2는 b에 따라 0이 아님)만 본다.
학습 성능에 대한 증거가 아니다. 실행: python3 analysis/toy_gca_init.py (CPU, 수 초)
"""
import json
from pathlib import Path

import torch
import torch.nn as nn

torch.manual_seed(0)
D, H, L, NA, NV, NF = 64, 4, 8, 8, 32, 8
OUT = Path(__file__).resolve().parent / "toy_gca_init_results.json"


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.att = nn.MultiheadAttention(D, H, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.att(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.ca = nn.ModuleList([nn.MultiheadAttention(D, H, batch_first=True) for _ in range(L // 4)])
        self.lnq = nn.ModuleList([nn.LayerNorm(D) for _ in range(L // 4)])
        self.wg = nn.Linear(D, 1)

    def forward(self, act, vis, force=None, mode="none", gate_bias=None):
        x = torch.cat([act, vis] + ([force] if mode == "concat" else []), dim=1)
        for i, b in enumerate(self.blocks):
            x = b(x)
            if mode == "gca" and i % 4 == 3:
                j = i // 4
                a = x[:, :NA]
                delta = self.ca[j](self.lnq[j](a), force, force, need_weights=False)[0]
                zbar = force.mean(1)
                g = torch.tanh(torch.zeros(1)) if gate_bias is None else torch.sigmoid(self.wg(zbar) + gate_bias)[:, None, :]
                x = torch.cat([a + g * delta, x[:, NA:]], dim=1)
        return x[:, :NA]


def main():
    net = Net().eval()
    res = {}
    rel = {"concat": [], "gca_tanh0": [], "gca_sigmoid_b-5": [], "gca_sigmoid_b-2": [], "gca_sigmoid_b0": []}
    with torch.no_grad():
        for _ in range(50):
            act, vis, force = torch.randn(1, NA, D), torch.randn(1, NV, D), torch.randn(1, NF, D)
            base = net(act, vis)
            n = base.norm()
            rel["concat"].append(float((net(act, vis, force, "concat") - base).norm() / n))
            rel["gca_tanh0"].append(float((net(act, vis, force, "gca") - base).norm() / n))
            for b, k in [(-5.0, "gca_sigmoid_b-5"), (-2.0, "gca_sigmoid_b-2"), (0.0, "gca_sigmoid_b0")]:
                rel[k].append(float((net(act, vis, force, "gca", b) - base).norm() / n))
    for k, v in rel.items():
        res[k] = {"mean_rel_change": sum(v) / len(v), "max_rel_change": max(v)}
        print(f"{k:18s} 평균 상대 변화 {res[k]['mean_rel_change']:.4f}  최대 {res[k]['max_rel_change']:.4f}")
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
