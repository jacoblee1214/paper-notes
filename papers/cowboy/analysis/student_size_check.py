"""토이 T2: 부록 E.5가 적은 학생 정책 크기를 torch로 세운다 (학습하지 않음, 한 번의 순전파와 역전파 시간만 잰다).

적힌 값: 토큰 폭 128(124 + 유형 4), 인코더 6층, 디코더 6층, 헤드 8, 피드포워드 1024,
점군 토큰 128개 + 상태 토큰 5개, 행동 질의 C개(논문에 C 값 없음; 1과 8로 둔다) + 보조 질의 1개,
점 인코더 MLP [3, 64, 64, 64] + 투영 64 -> 124, 상태 인코더 MLP [in, 128, 256, 124] 5개(입력 7, 6, 16, 16, 3),
행동 머리 -> 32, 보조 머리 -> 3.
반경 0.1, 이웃 64개의 집합 추상화(SetAbstraction) 그룹핑은 파라미터가 없어 세지 않는다.
논문 표 11의 상태 입력은 Franka 7, ARX 6, LEAP 16, LEAP 제어 목표 차이 16, 목표 물체 위치 3 이다.

실행: python3 analysis/student_size_check.py   (torch 필요, CPU에서 수 초; MPS가 있으면 같이 잰다)
결과: analysis/student_size_check_results.json
"""
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

OUT = Path(__file__).resolve().parent / "student_size_check_results.json"
D, TYPE, FF, HEADS, LAYERS = 128, 4, 1024, 8, 6


def mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Student(nn.Module):
    def __init__(self, n_action_queries=1):
        super().__init__()
        self.point_mlp = mlp([3, 64, 64, 64])
        self.point_proj = mlp([64, 124])
        self.state_enc = nn.ModuleList([mlp([i, 128, 256, 124]) for i in (7, 6, 16, 16, 3)])
        self.type_emb = nn.Embedding(8, TYPE)
        enc = nn.TransformerEncoderLayer(D, HEADS, FF, dropout=0.0, batch_first=True)
        dec = nn.TransformerDecoderLayer(D, HEADS, FF, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, LAYERS)
        self.decoder = nn.TransformerDecoder(dec, LAYERS)
        self.queries = nn.Parameter(torch.zeros(n_action_queries + 1, D))
        self.action_head = nn.Linear(D, 32)
        self.aux_head = nn.Linear(D, 3)
        self.c = n_action_queries

    def forward(self, tokens, states):
        # tokens: (B, 128, 64) 집합 추상화 뒤의 점 특징, states: 길이 5 리스트
        B = tokens.shape[0]
        pt = self.point_proj(tokens)                                   # (B, 128, 124)
        st = torch.stack([enc(s) for enc, s in zip(self.state_enc, states)], dim=1)  # (B, 5, 124)
        x = torch.cat([pt, st], dim=1)
        types = torch.cat([torch.zeros(128), torch.arange(1, 6).float()]).long().to(x.device)
        x = torch.cat([x, self.type_emb(types).unsqueeze(0).expand(B, -1, -1)], dim=-1)  # (B, 133, 128)
        mem = self.encoder(x)
        out = self.decoder(self.queries.unsqueeze(0).expand(B, -1, -1), mem)
        return self.action_head(out[:, : self.c]), self.aux_head(out[:, self.c:])


def count(m):
    return sum(p.numel() for p in m.parameters())


def main():
    res = {"d_model": D, "layers": LAYERS, "heads": HEADS, "ff": FF}
    for c in (1, 8):
        m = Student(c)
        res[f"params_C{c}"] = count(m)
        res[f"params_C{c}_transformer_only"] = count(m.encoder) + count(m.decoder)
        res[f"params_C{c}_encoders"] = count(m.point_mlp) + count(m.point_proj) + sum(count(e) for e in m.state_enc)
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    B = 64
    for dev in devices:
        torch.manual_seed(0)
        m = Student(1).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        tok = torch.randn(B, 128, 64, device=dev)
        sts = [torch.randn(B, i, device=dev) for i in (7, 6, 16, 16, 3)]
        times = []
        for it in range(6):
            t0 = time.perf_counter()
            a, aux = m(tok, sts)
            loss = a.pow(2).mean() + 0.1 * aux.pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if dev == "mps":
                torch.mps.synchronize()
            times.append(time.perf_counter() - t0)
        res[f"step_ms_batch{B}_{dev}"] = round(1000 * sum(times[2:]) / len(times[2:]), 1)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
