"""블록 인과 마스크의 효과를 같은 데이터에서 분리하는 장난감 검사 (torch, CPU 4스레드, 수 분 이내).

질문: 같은 행동 청크 데이터로 정책 구조(블록 인과 어텐션 마스크 유무)만 바꾸면 성공률이 얼마나 달라지는가?
      그 차이는 데이터 크기와 데이터의 의존 방향에 따라 어떻게 달라지는가?

설정
  - 관측 o ∈ R^8 ~ N(0, I). 행동 청크는 블록 4개(기지, 몸통, 팔, 손), 블록마다 H = 8 스텝 × 2차원 = 16차원.
  - 전문가 (결정적): 블록 k의 청크 a_k = tanh(G_k·o + R_k·a_{앞 블록}). 의존 순서는 두 가지.
      정방향 자료: 0 → 1 → 2 → 3 (앞 블록이 뒤 블록의 입력), 역방향 자료: 3 → 2 → 1 → 0.
  - 정책 (흐름 정합, 작은 트랜스포머): 토큰 5개 = 관측 토큰 + 블록 토큰 4개.
      블록 토큰은 잡음 섞인 청크 x_t^k = (1−t) z_k + t a_k를 넣는다 (블록 임베딩과 시간 임베딩 포함).
      출력은 블록별 속도 v = a − z. d_model 64, 헤드 4, 층 2, 드롭아웃 0.
      마스크 켬: 블록 k 토큰은 관측과 블록 j ≤ k의 토큰만 본다 (기지 → 손 순서의 블록 인과).
      마스크 끔: 모든 토큰이 모든 토큰을 본다.
  - 학습: Adam 1e-3, 배치 128, 3000 스텝, 데이터 크기 N ∈ {300, 3000}, 시드 3개.
  - 추론: 잡음에서 10단계 오일러 적분으로 청크를 만든다. 시험 2000개.
  - 성공: 청크의 모든 원소가 전문가와 0.2 이내 (오프라인 시험 성공률이며 폐루프 롤아웃이 아니다).
  - 같은 데이터 생성 시드를 마스크 켬/끔이 공유하므로 차이는 구조에서 온다.
가정·한계: 장난감 대응 함수이며 로봇 환경과 폐루프 동작을 모사하지 않는다. 성공은 오프라인 정확도 기준이다.
  논문 정책의 토큰 구성과 학습 목적을 그대로 따르지 않는다.
실행: python3 analysis/extra_block_causal.py -> analysis/extra_block_causal_results.json
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "4"

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
HERE = Path(__file__).resolve().parent

OBS_DIM = 8
N_BLOCKS = 4
BLOCK_DIM = 2
H = 8
BLOCK_SIZE = H * BLOCK_DIM          # 16
D_MODEL = 64
TOL = 0.2
TRAIN_STEPS = 3000
BATCH = 128
N_TEST = 2000
N_EULER = 10
SIZES = (300, 3000)
SEEDS = (0, 1, 2)
DATA_SEED = 12345


def make_expert(rng):
    G = [rng.standard_normal((BLOCK_SIZE, OBS_DIM)) / np.sqrt(OBS_DIM) for _ in range(N_BLOCKS)]
    R = [rng.standard_normal((BLOCK_SIZE, BLOCK_SIZE)) * 0.5 / np.sqrt(BLOCK_SIZE) for _ in range(N_BLOCKS)]
    return G, R


def expert_chunks(o, weights, order):
    """order: 블록 의존 순서 (앞 원소가 먼저 계산되어 뒤 원소의 입력이 된다). 반환 (N, 4, 16), 블록 인덱스 순."""
    G, R = weights
    out = [None] * N_BLOCKS
    prev = None
    for b in order:
        pre = o @ G[b].T
        if prev is not None:
            pre = pre + prev @ R[b].T
        out[b] = np.tanh(pre)
        prev = out[b]
    return np.stack(out, axis=1)


def make_data(n, weights, order, rng):
    o = rng.standard_normal((n, OBS_DIM))
    return o.astype(np.float32), expert_chunks(o, weights, order).astype(np.float32)


class FlowPolicy(torch.nn.Module):
    def __init__(self, mask_on):
        super().__init__()
        self.obs_proj = torch.nn.Linear(OBS_DIM, D_MODEL)
        self.time_proj = torch.nn.Linear(1, D_MODEL)
        self.blk_proj = torch.nn.Linear(BLOCK_SIZE, D_MODEL)
        self.blk_emb = torch.nn.Parameter(torch.randn(N_BLOCKS, D_MODEL) * 0.02)
        layer = torch.nn.TransformerEncoderLayer(D_MODEL, nhead=4, dim_feedforward=128,
                                                 dropout=0.0, batch_first=True)
        self.enc = torch.nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = torch.nn.Linear(D_MODEL, BLOCK_SIZE)
        m = torch.zeros(1 + N_BLOCKS, 1 + N_BLOCKS)
        for k in range(N_BLOCKS):
            for j in range(N_BLOCKS):
                if j > k:
                    m[k + 1, j + 1] = float("-inf")       # 블록 k는 블록 j > k를 보지 않는다
        self.mask_on = mask_on
        self.register_buffer("attn_mask", m)

    def forward(self, o, xt, t):
        # o: (n, 8), xt: (n, 4, 16), t: (n, 1)
        tok_obs = (self.obs_proj(o) + self.time_proj(t)).unsqueeze(1)
        tok_blk = self.blk_proj(xt) + self.blk_emb
        x = torch.cat([tok_obs, tok_blk], dim=1)
        y = self.enc(x, mask=self.attn_mask if self.mask_on else None)
        return self.head(y[:, 1:, :])


def train_policy(mask_on, o_tr, a_tr, seed):
    torch.manual_seed(seed)
    model = FlowPolicy(mask_on)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    Otr = torch.from_numpy(o_tr)
    Atr = torch.from_numpy(a_tr)
    t0 = time.perf_counter()
    loss = None
    for _ in range(TRAIN_STEPS):
        idx = rng.integers(0, Otr.shape[0], BATCH)
        a = Atr[idx]
        z = torch.from_numpy(rng.standard_normal((BATCH, N_BLOCKS, BLOCK_SIZE)).astype(np.float32))
        t = torch.from_numpy(rng.random((BATCH, 1)).astype(np.float32))
        tb = t.unsqueeze(1)
        xt = (1 - tb) * z + tb * a
        pred = model(Otr[idx], xt, t)
        loss = torch.mean((pred - (a - z)) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    wall = time.perf_counter() - t0
    return model.eval(), wall, float(loss.item())


def sample_chunks(model, o_te, seed):
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.standard_normal((o_te.shape[0], N_BLOCKS, BLOCK_SIZE)).astype(np.float32))
    o = torch.from_numpy(o_te)
    dt = 1.0 / N_EULER
    with torch.no_grad():
        for i in range(N_EULER):
            t = torch.full((o.shape[0], 1), i * dt)
            x = x + dt * model(o, x, t)
    return x.numpy()


def evaluate(model, o_te, a_te, seed):
    pred = sample_chunks(model, o_te, seed)
    err = np.abs(pred - a_te)
    chunk_ok = np.all(err.reshape(err.shape[0], -1) < TOL, axis=1)
    block_ok = np.all(err < TOL, axis=2)
    return dict(success=float(chunk_ok.mean()),
                block_success=[float(v) for v in block_ok.mean(axis=0)],
                mse=float(np.mean((pred - a_te) ** 2)))


def main():
    t_all = time.perf_counter()
    rng_w = np.random.default_rng(DATA_SEED)
    weights = make_expert(rng_w)
    orders = {"forward_chain_0to3": [0, 1, 2, 3], "reverse_chain_3to0": [3, 2, 1, 0]}
    results = {}
    for ename, order in orders.items():
        rng_te = np.random.default_rng(DATA_SEED + 1)
        o_te, a_te = make_data(N_TEST, weights, order, rng_te)
        results[ename] = {}
        for n in SIZES:
            results[ename][str(n)] = {}
            for mask_on in (True, False):
                key = "mask_on" if mask_on else "mask_off"
                runs = []
                for s in SEEDS:
                    rng_d = np.random.default_rng(DATA_SEED + 100 * s + n)
                    o_tr, a_tr = make_data(n, weights, order, rng_d)
                    model, wall, last = train_policy(mask_on, o_tr, a_tr, s)
                    r = evaluate(model, o_te, a_te, seed=s)
                    r.update(wall_s=round(wall, 2), final_train_loss=last)
                    runs.append(r)
                suc = np.array([r["success"] for r in runs])
                mse = np.array([r["mse"] for r in runs])
                results[ename][str(n)][key] = dict(
                    success_mean=float(suc.mean()), success_sd=float(suc.std(ddof=1)),
                    success_per_seed=[float(v) for v in suc],
                    mse_mean=float(mse.mean()), mse_sd=float(mse.std(ddof=1)),
                    block_success_mean=[float(np.mean([r["block_success"][b] for r in runs]))
                                        for b in range(N_BLOCKS)],
                    runs=runs,
                )
                print(f"{ename:20s} N={n:5d} {key:8s} success {suc.mean():.3f} ± {suc.std(ddof=1):.3f} "
                      f"mse {mse.mean():.4f} train wall {sum(r['wall_s'] for r in runs):.1f}s", flush=True)
    out = dict(
        device="cpu",
        versions=dict(torch=torch.__version__, numpy=np.__version__, python=sys.version.split()[0]),
        threads=torch.get_num_threads(),
        mps_available=bool(torch.backends.mps.is_available()),
        setup=dict(obs_dim=OBS_DIM, n_blocks=N_BLOCKS, block_dim=BLOCK_DIM, horizon=H, d_model=D_MODEL,
                   layers=2, heads=4, train_steps=TRAIN_STEPS, batch=BATCH, optimizer="Adam 1e-3",
                   euler_steps=N_EULER, tol=TOL, n_test=N_TEST, sizes=list(SIZES), seeds=list(SEEDS),
                   mask_rule="block k attends to obs and blocks j <= k (order 0 -> 3)"),
        results=results,
        total_wall_s=round(time.perf_counter() - t_all, 2),
    )
    (HERE / "extra_block_causal_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("total wall", out["total_wall_s"], "s")


if __name__ == "__main__":
    main()
