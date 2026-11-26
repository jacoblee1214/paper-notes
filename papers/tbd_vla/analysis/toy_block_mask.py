"""토이: 블록 확산의 이중 배치(doubled layout) 마스크가 정말 "블록 단위 자기회귀 + 블록 안 병렬"인지 검사한다.

논문 4.2절은 깨끗한 열 x0 와 오염된 열 xt 를 이어 붙여 한 번의 순전파로 모든 블록을 학습한다고 하고, 그림 2(B)에 마스크를 그린다.
이 스크립트는 공식 저장소(TBD-VLA/lerobot)의 허용 쌍 표(`_make_block_diffusion_training_mask`)를 numpy로 옮겨 적은 것이며
저장소 코드를 import하거나 실행하지 않는다. 저장소는 `git clone https://github.com/TBD-VLA/lerobot repo` 후
`git -C repo checkout 41c746ed487520cf6cc49eb767d090837691e2b2` 한 상태(분석 시점의 main)이고, 이 스크립트는 repo/를 읽지 않는다.

검사
 1. 정보 누수: 블록 j 를 예측하는 위치(j=0이면 앵커, 그 밖에는 clean_{j-1})의 출력이 정답 블록 clean_j 와
    이후 블록(clean_{>j}, noisy_{>j})을 바꿔도 변하지 않는다(2층 무작위 트랜스포머로 수치 확인).
 2. 오염 블록 noisy_j 를 바꾸면 블록 j 의 예측이 바뀐다. 블록 j+1 의 예측도 2층 이상 경로(noisy_j -> clean_{j-1} -> clean_j)로
    아주 조금 바뀐다. 이 경로의 정보는 이미 보이는 clean_j 의 부분집합이라 정답 누수는 아니지만, 추론에서 확정된 블록은
    마스크 비율 0 인 noisy_j 로 캐시되므로 학습(무작위 비율)과 입력 분포가 다르다.
 3. 순차 추론 동등성: 블록 j 까지만 이어 붙인 짧은 열(추론 시 상태)에서 같은 예측 위치의 출력이 긴 열의 출력과 같다.
 4. 논문 그림 2(B) 마스크(깨끗한 블록이 앵커를 본다)와 현재 main(보지 않는다)의 차이 크기: 같은 무작위 가중치에서 출력이 얼마나 다른가.
실행: python3 analysis/toy_block_mask.py   (torch CPU, 수 초). 결과는 toy_block_mask_results.json.
"""
import json
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent / "toy_block_mask_results.json"
P, B, NB, V = 5, 3, 4, 11          # 접두 길이, 블록 토큰 수, 블록 수, 어휘
PRE, ANC, NOI, CLN = 0, 1, 2, 3


def layout(n_blocks=NB):
    """[prefix | anchor | noisy_0..n-1 | clean_0..n-1] 의 (범주, 블록 번호, 위치 id)"""
    cat, blk, pos = [], [], []
    for i in range(P):
        cat.append(PRE); blk.append(-1); pos.append(i)
    for i in range(B):
        cat.append(ANC); blk.append(-1); pos.append(P + i)
    for b in range(n_blocks):
        for i in range(B):
            cat.append(NOI); blk.append(b); pos.append(P + B + b * B + i)
    for b in range(n_blocks):
        for i in range(B):
            cat.append(CLN); blk.append(b); pos.append(P + B + b * B + i)   # clean_b 는 noisy_b 와 같은 위치(RoPE 공유)
    return np.array(cat), np.array(blk), np.array(pos)


def allowed(cat, blk, clean_sees_anchor):
    L = len(cat)
    ci, cj = cat[:, None], cat[None, :]
    bi, bj = blk[:, None], blk[None, :]
    idx = np.arange(L)
    causal = idx[None, :] <= idx[:, None]
    a = ((ci == PRE) & (cj == PRE) & causal)
    a |= (ci == ANC) & (cj == PRE)
    a |= (ci == ANC) & (cj == ANC)
    a |= (ci == ANC) & (cj == NOI) & (bj == 0)
    a |= (ci == NOI) & (cj == PRE)
    a |= (ci == NOI) & (cj == NOI) & (bi == bj)
    a |= (ci == NOI) & (cj == CLN) & (bi > bj)
    a |= (ci == CLN) & (cj == PRE)
    if clean_sees_anchor:
        a |= (ci == CLN) & (cj == ANC)          # 논문 그림 2(B)와 2026-06 코드의 항
    a |= (ci == CLN) & (cj == NOI) & (bj == bi + 1)
    a |= (ci == CLN) & (cj == CLN) & (bi >= bj)
    return a


class Tiny(torch.nn.Module):
    def __init__(self, d=32, layers=2):
        super().__init__()
        self.emb = torch.nn.Embedding(V + 1, d)
        self.q = torch.nn.ModuleList([torch.nn.Linear(d, d) for _ in range(layers)])
        self.k = torch.nn.ModuleList([torch.nn.Linear(d, d) for _ in range(layers)])
        self.v = torch.nn.ModuleList([torch.nn.Linear(d, d) for _ in range(layers)])
        self.o = torch.nn.ModuleList([torch.nn.Linear(d, d) for _ in range(layers)])
        self.head = torch.nn.Linear(d, V)
        self.d = d

    def posenc(self, pos):
        i = torch.arange(self.d // 2, dtype=torch.float32)
        ang = torch.tensor(pos, dtype=torch.float32)[:, None] / (10.0 ** (i[None, :] * 4 / self.d))
        return torch.cat([ang.sin(), ang.cos()], dim=-1)

    def forward(self, ids, pos, mask):
        h = self.emb(ids) + self.posenc(pos)
        m = torch.tensor(mask)
        for q, k, v, o in zip(self.q, self.k, self.v, self.o):
            att = (q(h) @ k(h).T) / self.d ** 0.5
            att = att.masked_fill(~m, float("-inf")).softmax(-1)
            h = h + o(att @ v(h))
        return self.head(h)


def rand_ids(rng, cat, blk, n_blocks):
    clean = rng.integers(0, V - 1, size=(n_blocks, B))          # 정답 토큰
    mask_tok = V - 1                                            # 마스크 토큰 id
    noisy = clean.copy()
    for b in range(n_blocks):
        t = rng.random()
        noisy[b][rng.random(B) < t] = mask_tok
    ids = np.zeros(len(cat), dtype=np.int64)
    ids[cat == PRE] = rng.integers(0, V - 1, size=(cat == PRE).sum())
    ids[cat == ANC] = mask_tok
    ids[cat == NOI] = noisy.reshape(-1)
    ids[cat == CLN] = clean.reshape(-1)
    return ids, clean, noisy


def predictor_positions(cat, blk, j):
    if j == 0:
        return np.where(cat == ANC)[0]
    return np.where((cat == CLN) & (blk == j - 1))[0]


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    res = {"leak_max_abs_change": {}, "forward_check": {}, "seq_equiv_max_abs": {}, "anchor_term_effect": {}}
    n_trial = 40
    leak_clean_ge_j, leak_noisy_gt_j, leak_clean_gt_j_noisy_only = [], [], []
    infl_noisy_j_on_j, infl_noisy_j_on_later = [], []
    seq_dev, anchor_dev = [], []
    for trial in range(n_trial):
        net = Tiny()
        cat, blk, pos = layout()
        M = allowed(cat, blk, clean_sees_anchor=False)
        ids, clean, noisy = rand_ids(rng, cat, blk, NB)
        pos_t = pos
        with torch.no_grad():
            base = net(torch.tensor(ids), pos_t, M)
            for j in range(NB):
                pp = predictor_positions(cat, blk, j)
                # 1. 정답 블록 j 이후(clean_j.., noisy_{>j})를 바꿔도 변하지 않아야 한다
                ids2 = ids.copy()
                sel = ((cat == CLN) & (blk >= j)) | ((cat == NOI) & (blk > j))
                ids2[sel] = rng.integers(0, V - 1, size=sel.sum())
                out2 = net(torch.tensor(ids2), pos_t, M)
                leak_clean_ge_j.append(float((out2[pp] - base[pp]).abs().max()))
                # 2. noisy_j 를 바꾸면 블록 j 예측은 바뀌고, 블록 j+1 이후 예측(clean_j 이상)은 clean_j 가 noisy_{j+1} 를 보므로 j+1에서만 바뀐다
                ids3 = ids.copy()
                selj = (cat == NOI) & (blk == j)
                ids3[selj] = (ids3[selj] + 1) % (V - 1)
                out3 = net(torch.tensor(ids3), pos_t, M)
                infl_noisy_j_on_j.append(float((out3[pp] - base[pp]).abs().max()))
                if j + 1 < NB:
                    pp_next = predictor_positions(cat, blk, j + 1)
                    infl_noisy_j_on_later.append(float((out3[pp_next] - base[pp_next]).abs().max()))
                # 3. 순차 추론 동등성: [prefix | anchor | noisy_0..j | clean_0..j-1] 만 이어 붙인 짧은 열
                keep = (cat == PRE) | (cat == ANC) | ((cat == NOI) & (blk <= j)) | ((cat == CLN) & (blk <= j - 1))
                kidx = np.where(keep)[0]
                Mk = allowed(cat[kidx], blk[kidx], False)
                outk = net(torch.tensor(ids[kidx]), pos_t[kidx], Mk)
                # 짧은 열 안에서 예측 위치의 인덱스
                new_idx = {int(g): n for n, g in enumerate(kidx)}
                got = torch.stack([outk[new_idx[int(g)]] for g in pp])
                seq_dev.append(float((got - base[pp]).abs().max()))
                # 4. 그림 2(B) 마스크 대 main 마스크 (j >= 1 에서만 차이가 난다)
                M2 = allowed(cat, blk, clean_sees_anchor=True)
                out4 = net(torch.tensor(ids), pos_t, M2)
                anchor_dev.append((j, float((out4[pp] - base[pp]).abs().max())))
    res["leak_max_abs_change"] = {"clean_ge_j_and_noisy_gt_j_perturbed": max(leak_clean_ge_j), "n_checks": len(leak_clean_ge_j)}
    res["forward_check"] = {"noisy_j_changes_pred_j_mean_abs": float(np.mean(infl_noisy_j_on_j)), "min": float(np.min(infl_noisy_j_on_j)),
                            "noisy_j_changes_pred_j+1_mean_abs": float(np.mean(infl_noisy_j_on_later))}
    res["seq_equiv_max_abs"] = {"max_over_trials_blocks": max(seq_dev), "n_checks": len(seq_dev)}
    ad = {j: float(np.mean([v for jj, v in anchor_dev if jj == j])) for j in range(NB)}
    res["anchor_term_effect"] = {"mean_abs_output_change_by_block": ad,
                                 "note": "블록 0은 앵커가 예측하므로 같음. j>=1에서 달라지는 크기는 무작위 가중치에서의 값이며 학습된 모델의 영향은 아님"}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
