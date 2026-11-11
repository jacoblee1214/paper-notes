"""3차원 손 입자 집계 검사: 입자 밀도와 합·평균 집계가 교차 손 일반화에 주는 영향 (torch, CPU 4스레드, 수 분 이내).

질문: 2차원 장난감에서 본 밀도 효과(합 집계 최적 κ가 밀도에 묶임)가 3차원 손 기하에서도 나타나는가?
      학습 밀도가 시험 밀도와 다르고 손이 다를 때, 합 집계 모형과 평균 집계 모형은 어떻게 달라지는가?

설정
  - 손 3종 (MuJoCo MJCF를 문자열로 직접 작성, 외부 자산 없음): 2관절 긴 팔 (0.14·0.12 m), 3관절 팔 (0.10·0.10·0.08 m),
    2관절 짧은 팔 (0.18·0.09 m). 관절은 힌지이고 축 방향이 손마다 다르다. 관절 각은 에피소드마다 U(−1, 1) rad.
  - 입자: 각 캡슐(링크) 축을 따라 간격 h로 배치 (MuJoCo mj_forward의 geom_xpos와 geom_xmat에서 끝점 계산).
    h ∈ {0.010, 0.025, 0.060} m. 입자 수는 링크 길이 / h에 비례한다.
  - 물체: 점 30개 (구 반경 0.05 m 안). 물체 중심은 손 입자 중 하나 근처 (50%) 또는 작업 공간 안의 무작위 점 (50%).
  - 참 세계 (κ = 1, 둘 다 가정이며 논문 모형이 아님):
      S (합형): y = Σ_i Σ_j K(p_j − o_i)
      M (평균형): y = Σ_i mean_j K(p_j − o_i)
      K(d) = max(0, 1 − |d|/0.05)
  - 학습기 (작은 MLP, 같은 쌍 커널 g_θ: 3 → 32 → 32 → 1, SiLU):
      합 집계: ŷ = Σ_i Σ_j g_θ(p_j − o_i)      평균 집계: ŷ = Σ_i mean_j g_θ(p_j − o_i)
  - 학습: Adam 2e-3, 배치 64 에피소드, 3000 스텝, 시드 0. 학습 자료는 손 A, B만 쓴다.
      혼합 학습: 밀도 3종 모두 (에피소드 손당 200개씩)
      단일 밀도 학습: h = 0.025만
  - 시험: 손 C (학습에 없음), 밀도 3종, 손당 에피소드 200개, 상대 MSE = Σ(ŷ − y)² / Σy².
가정·한계: 쌍 커널이 거리 함수라서 학습기가 참 세계를 정확히 표현할 수 있다. 따라서 남는 오차는 주로 집계 방식과
  밀도에서 온다. 그래프 네트워크나 학습된 집계 함수는 쓰지 않았다. 손 기하는 MuJoCo로 얻지만 로봇 동역학은 없다.
실행: python3 analysis/extra_particle_3d.py -> analysis/extra_particle_3d_results.json
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "4"

import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

torch.set_num_threads(4)
HERE = Path(__file__).resolve().parent

DENSITIES = (0.010, 0.025, 0.060)
N_OBJ = 30
R_CONTACT = 0.05
N_EP = 200
TRAIN_STEPS = 3000
BATCH = 64
LR = 2e-3
SEED = 0


def hand_xml(lens, axes):
    """lens: 링크 길이 목록, axes: 관절 축 목록. 링크는 직전 관절 좌표계에서 x축으로 뻗는다."""
    body = ""
    close = ""
    for k, (L, ax) in enumerate(zip(lens, axes)):
        indent = "  " * (k + 1)
        body += (f'{indent}<body name="l{k}" pos="{0.0 if k == 0 else lens[k - 1]} 0 0">\n'
                 f'{indent}  <joint name="j{k}" type="hinge" axis="{ax}"/>\n'
                 f'{indent}  <geom type="capsule" fromto="0 0 0 {L} 0 0" size="0.01"/>\n')
        close = f"{indent}</body>\n" + close
    # 중첩 닫기: 바깥에서 안쪽으로 생성한 문자열을 정렬
    nested = body + close
    return f"<mujoco><worldbody>\n{nested}</worldbody></mujoco>"


HANDS = {
    "A_long2": hand_xml([0.14, 0.12], ["0 0 1", "0 1 0"]),
    "B_arm3": hand_xml([0.10, 0.10, 0.08], ["0 0 1", "0 1 0", "1 0 0"]),
    "C_short2": hand_xml([0.18, 0.09], ["1 0 0", "0 0 1"]),
}


def hand_particles(model, data, h):
    """모든 캡슐 축을 따라 간격 ~h로 점을 찍는다. 반환 (N, 3)."""
    mujoco.mj_forward(model, data)
    pts = []
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            continue
        center = data.geom_xpos[g].copy()
        axis = data.geom_xmat[g].reshape(3, 3)[:, 2].copy()
        half = model.geom_size[g][1]
        length = 2 * half
        n = max(2, int(round(length / h)) + 1)
        ts = np.linspace(-half, half, n)
        pts.append(center[None, :] + ts[:, None] * axis[None, :])
    return np.concatenate(pts, axis=0)


def sample_episode(rng, name, h, model, data):
    """에피소드 1개: (손 입자 (N,3), 물체 점 (M,3))."""
    data.qpos[:] = rng.uniform(-1.0, 1.0, size=model.nq)
    P = hand_particles(model, data, h)
    if rng.random() < 0.5:
        c = P[rng.integers(len(P))] + rng.uniform(-0.04, 0.04, 3)
    else:
        c = rng.uniform([-0.1, -0.2, -0.15], [0.25, 0.2, 0.15])
    O = c[None, :] + rng.uniform(-0.05, 0.05, size=(N_OBJ, 3))
    return P, O


def truth(P, O, world):
    d = np.linalg.norm(P[None, :, :] - O[:, None, :], axis=2)          # (M, N)
    K = np.clip(1 - d / R_CONTACT, 0, None)
    if world == "S":
        return float(K.sum())
    return float(K.mean(axis=1).sum())                                   # M: 물체 점마다 손 입자 평균


def build_set(rng, hands, densities, world, models_data):
    """반환: 패딩된 배열 (E, N_max, 3), 마스크 (E, N_max), 물체 (E, M, 3), 목표 (E,)."""
    eps = []
    for name in hands:
        model, data = models_data[name]
        for h in densities:
            for _ in range(N_EP):
                P, O = sample_episode(rng, name, h, model, data)
                eps.append((P, O, truth(P, O, world)))
    nmax = max(len(e[0]) for e in eps)
    E = len(eps)
    Pp = np.zeros((E, nmax, 3), dtype=np.float32)
    M = np.zeros((E, nmax), dtype=np.float32)
    O = np.zeros((E, N_OBJ, 3), dtype=np.float32)
    y = np.zeros(E, dtype=np.float32)
    dens = np.zeros(E, dtype=np.float32)
    for k, (P, Ob, t) in enumerate(eps):
        Pp[k, :len(P)] = P
        M[k, :len(P)] = 1.0
        O[k] = Ob
        y[k] = t
    return (torch.from_numpy(Pp), torch.from_numpy(M), torch.from_numpy(O), torch.from_numpy(y))


class PairKernel(torch.nn.Module):
    def __init__(self, agg):
        super().__init__()
        self.agg = agg
        self.g = torch.nn.Sequential(torch.nn.Linear(3, 32), torch.nn.SiLU(),
                                     torch.nn.Linear(32, 32), torch.nn.SiLU(),
                                     torch.nn.Linear(32, 1))

    def forward(self, Pp, M, O):
        # Pp: (B, N, 3), M: (B, N), O: (B, Mo, 3)
        rel = (O[:, :, None, :] - Pp[:, None, :, :]) / R_CONTACT          # (B, Mo, N, 3), 접촉 반경 단위
        g = self.g(rel).squeeze(-1)                                      # (B, Mo, N)
        g = g * M[:, None, :]
        if self.agg == "sum":
            per_obj = g.sum(dim=2)
        else:
            cnt = M.sum(dim=1).clamp(min=1.0)[:, None]
            per_obj = g.sum(dim=2) / cnt
        return per_obj.sum(dim=1)


def train_model(agg, data, seed):
    torch.manual_seed(seed)
    Pp, M, O, y = data
    net = PairKernel(agg)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed)
    E = Pp.shape[0]
    t0 = time.perf_counter()
    last = None
    for _ in range(TRAIN_STEPS):
        idx = torch.randint(0, E, (BATCH,), generator=g)
        pred = net(Pp[idx], M[idx], O[idx])
        loss = torch.mean((pred - y[idx]) ** 2) / torch.mean(y[idx] ** 2).clamp(min=1e-8)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.item())
    return net.eval(), time.perf_counter() - t0, last


def rel_mse(net, data):
    Pp, M, O, y = data
    with torch.no_grad():
        pred = net(Pp, M, O)
    return float(torch.sum((pred - y) ** 2) / torch.sum(y**2))


def main():
    t_all = time.perf_counter()
    models_data = {}
    for name, xml in HANDS.items():
        model = mujoco.MjModel.from_xml_string(xml)
        models_data[name] = (model, mujoco.MjData(model))
    hand_info = {}
    for name, (model, data) in models_data.items():
        counts = {}
        for h in DENSITIES:
            rng_c = np.random.default_rng(SEED)
            P, _ = sample_episode(rng_c, name, h, model, data)
            counts[str(h)] = int(len(P))
        hand_info[name] = dict(nq=int(model.nq), ngeom=int(model.ngeom), particles_per_density_example=counts)

    results = {"hands": hand_info, "cells": {}}
    trains = {"mixed_density": DENSITIES, "single_density_0.025": (0.025,)}
    for world in ("S", "M"):
        for tname, tdens in trains.items():
            rng_tr = np.random.default_rng(SEED + 1)
            train_data = build_set(rng_tr, ["A_long2", "B_arm3"], tdens, world, models_data)
            key = f"world_{world}__train_{tname}"
            results["cells"][key] = {}
            for agg in ("sum", "mean"):
                net, wall, last = train_model(agg, train_data, SEED)
                cell = {"train_wall_s": round(wall, 2), "final_train_loss_rel": last,
                        "train_episodes": int(train_data[0].shape[0]), "test": {}}
                for hname in ("A_long2", "B_arm3", "C_short2"):
                    for h in DENSITIES:
                        rng_te = np.random.default_rng(SEED + 7)
                        te = build_set(rng_te, [hname], (h,), world, models_data)
                        cell["test"][f"{hname}@{h}"] = rel_mse(net, te)
                results["cells"][key][agg] = cell
                c_vals = [cell["test"][f"C_short2@{h}"] for h in DENSITIES]
                print(f"world {world} train {tname:22s} agg {agg:4s} wall {wall:5.1f}s "
                      f"C_short2 rel-MSE by test density {[f'{v:.3f}' for v in c_vals]}", flush=True)

    results["device"] = "cpu"
    results["versions"] = dict(torch=torch.__version__, mujoco=mujoco.__version__, numpy=np.__version__,
                               python=sys.version.split()[0])
    results["threads"] = torch.get_num_threads()
    results["mps_available"] = bool(torch.backends.mps.is_available())
    results["setup"] = dict(densities=list(DENSITIES), n_obj=N_OBJ, r_contact=R_CONTACT, n_ep_per_cell=N_EP,
                            train_steps=TRAIN_STEPS, batch=BATCH, lr=LR, seed=SEED,
                            train_hands=["A_long2", "B_arm3"], test_hand="C_short2")
    results["total_wall_s"] = round(time.perf_counter() - t_all, 2)
    (HERE / "extra_particle_3d_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("total wall", results["total_wall_s"], "s")


if __name__ == "__main__":
    main()
