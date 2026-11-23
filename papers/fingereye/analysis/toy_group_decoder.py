"""토이: 그룹 조건 디코더(GroupDec)에서 토큰이 하나뿐인 그룹의 교차 어텐션이 무엇을 하는지 확인한다.

논문 식 (15)는 그룹마다 별도의 교차 어텐션을 두고 잔차를 평균한다. 공식 저장소의 구현을 정적으로 읽으면
고유감각(joint) 그룹과 손목(wrist) 그룹은 토큰이 하나씩이다. 키가 하나면 softmax 가중치는 항상 1이므로
그 그룹의 출력은 쿼리에 상관없이 같은 벡터다. 이를 무작위 가중치의 작은 모듈로 확인하고(코드는 import하지 않음),
논문 표 8의 차원으로 네 가지 인터페이스의 파라미터 수도 직접 센다.

실행: python3 analysis/toy_group_decoder.py  (torch, CPU, 몇 초)  → analysis/toy_group_decoder_results.json
"""
import json
from pathlib import Path

import torch
from torch import nn

torch.manual_seed(0)
OUT = Path(__file__).with_name("toy_group_decoder_results.json")
D, H, Tp = 512, 16, 16
R = {}

# 1) 키가 하나인 교차 어텐션은 쿼리와 무관한 상수 벡터를 낸다
attn = nn.MultiheadAttention(D, H, batch_first=True).eval()
q = torch.randn(1, Tp, D)
with torch.no_grad():
    for n_keys in (1, 2, 4):
        mem = torch.randn(1, n_keys, D)
        out, w = attn(q, mem, mem)
        spread = (out - out.mean(dim=1, keepdim=True)).abs().max().item()
        R[f"cross_attn_{n_keys}_key"] = {"max_abs_dev_across_16_queries": spread, "attn_weight_min": float(w.min()), "attn_weight_max": float(w.max())}

# 2) 세 그룹(FE 4토큰, wrist 1, joint 1) 잔차 평균에서 쿼리마다 달라지는 몫
joint = nn.MultiheadAttention(D, H, batch_first=True).eval()
wrist = nn.MultiheadAttention(D, H, batch_first=True).eval()
fe = nn.MultiheadAttention(D, H, batch_first=True).eval()
with torch.no_grad():
    mj, mw, mf = torch.randn(1, 1, D), torch.randn(1, 1, D), torch.randn(1, 4, D)
    dj, dw, df = joint(q, mj, mj)[0], wrist(q, mw, mw)[0], fe(q, mf, mf)[0]
    var = lambda t: float(t.var(dim=1).mean())
    R["group_residual_variance_across_queries"] = {"joint(1 token)": var(dj), "wrist(1 token)": var(dw), "fe(4 tokens)": var(df)}
    avg = torch.stack([dj, dw, df]).mean(0)
    R["group_mean_query_dependent_share"] = float(var(df / 3) / var(avg))
    R["fe_residual_weight_in_mean"] = 1 / 3
    # 평탄 디코더(토큰 6개 한 번에): 무작위 초기화에서 키 6개 어텐션 질량 평균
    flat = nn.MultiheadAttention(D, H, batch_first=True).eval()
    mem6 = torch.cat([mf, mw, mj], dim=1)
    _, w6 = flat(q, mem6, mem6, average_attn_weights=True)
    R["flat_6_token_init_attention_mass"] = {"FE(4)": float(w6[..., :4].sum(-1).mean()), "wrist": float(w6[..., 4].mean()), "joint": float(w6[..., 5].mean())}


# 3) 파라미터 수: 표 8의 차원(512, FF 2048, 4+4층)으로 네 인터페이스를 직접 센다
class EncLayer(nn.Module):
    def __init__(s):
        super().__init__()
        s.a = nn.MultiheadAttention(D, H)
        s.l1, s.l2 = nn.Linear(D, 2048), nn.Linear(2048, D)
        s.n1, s.n2 = nn.LayerNorm(D), nn.LayerNorm(D)


class DecLayer(nn.Module):
    def __init__(s, n_cross):
        super().__init__()
        s.sa = nn.MultiheadAttention(D, H)
        s.ca = nn.ModuleList([nn.MultiheadAttention(D, H) for _ in range(n_cross)])
        s.l1, s.l2 = nn.Linear(D, 2048), nn.Linear(2048, D)
        s.n = nn.ModuleList([nn.LayerNorm(D) for _ in range(2 + n_cross)])


def count(m):
    return sum(p.numel() for p in m.parameters())


proj = nn.Sequential(nn.Linear(2304, 2 * D), nn.LayerNorm(2 * D), nn.GELU(), nn.Linear(2 * D, D))
state = nn.Sequential(nn.Linear(8, D), nn.LayerNorm(D), nn.GELU(), nn.Linear(D, D))
head = nn.Sequential(nn.Linear(D, D // 2), nn.GELU(), nn.Linear(D // 2, 8))
base = 5 * count(proj) + 5 * D + Tp * D + count(state) + count(head) + 2 * D
cfg = {
    "NoEnc+FDec": base + 4 * count(DecLayer(1)),
    "GEnc+FDec": base + 4 * count(EncLayer()) + 4 * count(DecLayer(1)),
    "NoEnc+GDec": base + 4 * count(DecLayer(3)),
    "GEnc+GDec": base + 4 * count(EncLayer()) + 4 * count(DecLayer(3)),
}
R["param_counts"] = cfg
R["param_ratio_to_NoEnc+FDec"] = {k: round(v / cfg["NoEnc+FDec"], 3) for k, v in cfg.items()}
OUT.write_text(json.dumps(R, indent=1), encoding="utf-8")
print(json.dumps(R, indent=1))
