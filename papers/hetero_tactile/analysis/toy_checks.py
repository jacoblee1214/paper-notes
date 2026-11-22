"""CPU 토이 두 개 (각각 수 초). 결과는 toy_checks_results.json.

토이 1: 공식 코드의 `best_probe_model_state = probe_mlp.state_dict().copy()` 패턴이 스냅샷이 아니라는 것을 확인한다
        (dict.copy()는 얕은 복사라 텐서가 학습 중인 파라미터와 같은 저장소를 가리킨다).
토이 2: Procrustes 교차 센서 검색(논문 그림 3)이 에피소드별 대응이 없고 객체 정보만 공유되는 합성 임베딩에서도
        R@1/R@5가 어느 정도 나오는지 본다. 객체 정보만 공유하는 경우와 에피소드 정보도 공유하는 경우를 비교한다.

실행: python3 analysis/toy_checks.py   (torch는 토이 1에만 쓴다. 없으면 토이 1을 건너뛴다)
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("toy_checks_results.json")
R = {}

# ---------------------------------------------------------------- 토이 1
try:
    import copy
    import torch

    torch.manual_seed(0)
    net = torch.nn.Linear(4, 1)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    x, y = torch.randn(64, 4), torch.randn(64, 1)

    def step():
        opt.zero_grad()
        torch.nn.functional.mse_loss(net(x), y).backward()
        opt.step()

    step()
    snap_shallow = net.state_dict().copy()          # 공식 코드와 같은 패턴
    snap_deep = {k: v.detach().clone() for k, v in net.state_dict().items()}  # 올바른 스냅샷 (backbone 쪽 코드가 쓰는 방식)
    w_at_snapshot = net.weight.detach().clone()
    for _ in range(50):
        step()
    w_now = net.weight.detach().clone()
    R["toy1_state_dict_alias"] = {
        "weight_changed_after_50_steps": float((w_now - w_at_snapshot).abs().max()),
        "shallow_snapshot_equals_current": bool(torch.equal(snap_shallow["weight"], w_now)),
        "shallow_snapshot_equals_snapshot_time": bool(torch.equal(snap_shallow["weight"], w_at_snapshot)),
        "deep_snapshot_equals_snapshot_time": bool(torch.equal(snap_deep["weight"], w_at_snapshot)),
        "same_storage_as_live_param": bool(snap_shallow["weight"].data_ptr() == net.weight.data_ptr()),
        "meaning": "load_state_dict(shallow)는 아무 일도 하지 않는다: 헤드는 early-stop 시점(마지막 스텝)의 가중치로 남고, "
                   "인코더·트렁크만 best-val 스텝으로 되돌아간다.",
        "torch": torch.__version__,
    }
except ImportError:
    R["toy1_state_dict_alias"] = "torch 없음, 건너뜀"

# ---------------------------------------------------------------- 토이 2
D, N_EP, N_OBJ, N_WIN = 192, 99, 30, 8


def make_pair(rng, sigma, lam, n_ep=N_EP):
    """두 센서의 에피소드 임베딩. 객체 코드는 공유(센서 B는 고정 회전 Q), 에피소드 코드는 lam 만큼 공유, 창마다 독립 잡음."""
    obj_code = rng.normal(size=(N_OBJ, D))
    Q, _ = np.linalg.qr(rng.normal(size=(D, D)))
    obj = rng.integers(0, N_OBJ, n_ep)
    ep_code = rng.normal(size=(n_ep, D))
    base = obj_code[obj] + lam * ep_code
    a = np.stack([base + sigma * rng.normal(size=(n_ep, D)) for _ in range(N_WIN)]).mean(0)
    b = np.stack([base @ Q + sigma * rng.normal(size=(n_ep, D)) for _ in range(N_WIN)]).mean(0)
    # 센서마다 에피소드와 무관한 센서 특유 성분도 더한다
    a = a + 0.5 * rng.normal(size=(1, D))
    return a, b, obj


def procrustes(A, B):
    u, _, vt = np.linalg.svd(A.T @ B)
    return u @ vt


def recall(a, b, W):
    q = (a @ W)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    g = b / np.linalg.norm(b, axis=1, keepdims=True)
    sim = q @ g.T
    rank = (sim > np.diag(sim)[:, None]).sum(1)  # 정답보다 높은 후보 수
    return float((rank == 0).mean()), float((rank < 5).mean())


def run(sigma, lam, reps=40, seed=0):
    rng = np.random.default_rng(seed)
    r1, r5 = [], []
    for _ in range(reps):
        a, b, obj = make_pair(rng, sigma, lam)
        idx = rng.permutation(N_EP)
        fit, test = idx[: N_EP // 2], idx[N_EP // 2:]
        a_c, b_c = a[fit] - a[fit].mean(0), b[fit] - b[fit].mean(0)
        W = procrustes(a_c, b_c)
        x1, x5 = recall(a[test] - a[fit].mean(0), b[test] - b[fit].mean(0), W)
        r1.append(x1)
        r5.append(x5)
    return round(100 * float(np.mean(r1)), 1), round(100 * float(np.mean(r5)), 1)


tab = []
for lam in (0.0, 0.5, 1.0):
    for sigma in (1.5, 2.5, 3.5):
        r1, r5 = run(sigma, lam)
        tab.append({"episode_code_shared_lambda": lam, "window_noise_sigma": sigma, "R@1_pct": r1, "R@5_pct": r5})
R["toy2_procrustes_retrieval"] = {
    "setup": "에피소드 99개, 객체 30개(균등 무작위 배정), 창 8개 평균, d=192, 절반(50)으로 Procrustes를 맞추고 나머지 49개에서 검색 (갤러리 49), 40번 반복",
    "chance_pct_gallery49": {"R@1": round(100 / 49, 1), "R@5": round(500 / 49, 1)},
    "paper_fig3_reported_pct_gallery99": {"random_init": [3, 20], "MAE_only": [20, 61], "HTT": [41, 82]},
    "rows": tab,
    "caveat": "토이는 실제 임베딩이 아니다. 갤러리 크기(49 대 99)와 객체 분포가 논문과 다르다. 객체 정보만 공유하는 행(lambda=0)이 논문 수치 범위의 R@K를 만들 수 있는지만 본다.",
}
OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(R, ensure_ascii=False, indent=1))
