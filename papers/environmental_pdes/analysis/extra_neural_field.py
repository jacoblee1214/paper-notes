"""신경 장 대리 모형을 입자 필터의 측정 모형으로 쓰는 검증 (CPU, torch 4스레드, 수 분 이내).

질문: 격자 지도(보간) 대신 작은 MLP φ_θ(x, t; κ)를 측정 모형으로 쓰면 위치 오차가 어떻게 달라지는가?
      같은 MLP에 κ를 입력으로 넣었을 때 센서 자료만으로 κ를 복원할 수 있는가?

설정 (pde_toy.py의 장, 경로, 운동 잡음, 측정 잡음, 초기 입자 분포를 그대로 쓴다):
  - 학습 자료: pde_toy.solve(40, κ, "asym")의 0–120 s 스냅샷. κ ∈ {0.10, 0.12, 0.15, 0.25, 0.30, 0.40}.
    κ = 0.20 (필터와 시험에 쓰는 참값)은 학습에서 뺀다. 필터 지도와 같은 격자이므로 보간 오차만 본다.
  - MLP: 입력 (x/L, y/L, t/T, κ/0.2) → φ, 은닉 64 × 3, SiLU. Adam, 배치 4096, 6000 스텝, 고정 시드.
  - 필터: pde_toy.run_one과 같은 난수 순서의 입자 필터(N = 500, 체계적 재표본추출), 측정 모형만 교체.
    격자 지도 기준선(grid_match)은 pde_toy의 asym_match와 같은 값이 나와야 한다 (구현 검증).
  - κ 복원: 센서 자료와 참 경로(오라클)에서 κ 후보 46개에 대한 음의 로그우도 프로파일, 실행 30회.
가정·한계: 경로는 참 경로로 프로파일한다(필터 추정 경로가 아니다). 장과 시험 장은 같은 솔버 계열이며
  참값(80×80)은 학습 자료(40×40)보다 세밀하다. 학습 κ 6개로 κ 방향 보간을 본 것이며 외삽은 보지 않았다.
실행: python3 analysis/extra_neural_field.py -> analysis/extra_neural_field_results.json
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
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pde_toy as P  # noqa: E402

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "4"  # pde_toy 임포트가 1로 덮어쓰므로 되돌림

torch.set_num_threads(4)
torch.manual_seed(0)

L, T = P.L, P.T
N_PART = P.N_PART
N_RUNS = P.N_RUNS
SIG_S, SIG_O, SIG_INIT = P.SIG_S, P.SIG_O, P.SIG_INIT
N_MAP, N_TRUE = P.N_MAP, P.N_TRUE
KAPPA_REF = 0.2
KAPPA_TRAIN = (0.10, 0.12, 0.15, 0.25, 0.30, 0.40)
TRAIN_STEPS = 6000
BATCH = 4096
DEVICE = torch.device("cpu")


class FieldNet(torch.nn.Module):
    def __init__(self, width=64, depth=3):
        super().__init__()
        layers, d_in = [], 4
        for _ in range(depth):
            layers += [torch.nn.Linear(d_in, width), torch.nn.SiLU()]
            d_in = width
        layers.append(torch.nn.Linear(d_in, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def features(xy, t, kappa):
    """xy: (N, 2) [m], t: 시각 [s] (스칼라), kappa: 스칼라 -> (N, 4) 텐서."""
    n = xy.shape[0]
    f = np.empty((n, 4), dtype=np.float32)
    f[:, 0:2] = xy / L
    f[:, 2] = t / T
    f[:, 3] = kappa / KAPPA_REF
    return torch.from_numpy(f)


def build_training_set():
    """40×40 솔버 스냅샷을 (x, t, κ) → φ 자료로 펼친다."""
    n = N_MAP
    dx = L / n
    xs = (np.arange(n) + 0.5) * dx
    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    xy = np.stack([XX.ravel(), YY.ravel()], axis=1)
    feats, targs, info = [], [], {}
    for kap in KAPPA_TRAIN:
        snaps, inf = P.solve(n, kap, "asym")
        info[str(kap)] = inf
        for t in range(T + 1):
            feats.append(features(xy, float(t), kap).numpy())
            targs.append(snaps[t].ravel().astype(np.float32))
    X = torch.from_numpy(np.concatenate(feats))
    Y = torch.from_numpy(np.concatenate(targs)).unsqueeze(1)
    return X, Y, info


def train_net(X, Y):
    net = FieldNet().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_STEPS, eta_min=1e-4)
    g = torch.Generator().manual_seed(1)
    t0 = time.perf_counter()
    log = []
    for step in range(TRAIN_STEPS):
        idx = torch.randint(0, X.shape[0], (BATCH,), generator=g)
        loss = torch.mean((net(X[idx].to(DEVICE)) - Y[idx].to(DEVICE)) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == TRAIN_STEPS - 1:
            log.append(dict(step=step, mse=float(loss.item())))
    return net.eval(), time.perf_counter() - t0, log


class NeuralMeasure:
    """pde_toy.sample과 같은 인터페이스: 측정 예측 = φ_θ(p, t; κ̂)."""

    def __init__(self, net, kappa):
        self.net, self.kappa = net, kappa

    def __call__(self, t, p):
        with torch.no_grad():
            out = self.net(features(np.clip(p, 0, L), float(t), self.kappa).to(DEVICE))
        return out.numpy().ravel().astype(np.float64)


class GridMeasure:
    def __init__(self, snaps):
        self.snaps = snaps

    def __call__(self, t, p):
        return P.sample(self.snaps[t], p)


def run_pf(seed, truth, meas, d, p_path, sig_s=SIG_S):
    """pde_toy.run_one과 같은 난수 순서. meas(t, p) -> 측정 예측. 반환 (추정 궤적, 참 궤적, 센서 자료)."""
    rng = np.random.default_rng(P.RNG_BASE + seed)
    p_true = np.empty((T + 1, 2))
    p_true[0] = p_path[0]
    for t in range(1, T + 1):
        p_true[t] = np.clip(p_true[t - 1] + d[t - 1] + rng.normal(0, SIG_O, 2), 0, L)
    y = np.array([P.sample(truth[t], p_true[t:t + 1])[0] for t in range(T + 1)])
    y = y + rng.normal(0, sig_s, T + 1)

    p = np.clip(p_true[0] + rng.normal(0, SIG_INIT, (N_PART, 2)), 0, L)
    w = np.full(N_PART, 1.0 / N_PART)
    est = np.empty((T + 1, 2))
    est[0] = p.mean(axis=0)
    for t in range(1, T + 1):
        p = np.clip(p + d[t - 1] + rng.normal(0, SIG_O, (N_PART, 2)), 0, L)
        pred = meas(t, p)
        logw = np.log(w + 1e-300) - 0.5 * ((y[t] - pred) / sig_s) ** 2
        logw -= logw.max()
        w = np.exp(logw)
        w /= w.sum()
        est[t] = w @ p
        if 1.0 / np.sum(w**2) < N_PART / 2:
            u0 = (rng.random() + np.arange(N_PART)) / N_PART
            idx = np.minimum(np.searchsorted(np.cumsum(w), u0), N_PART - 1)
            p = p[idx]
            w = np.full(N_PART, 1.0 / N_PART)
    return est, p_true, y


def rmse_of(est, p_true):
    e = np.linalg.norm(est[1:] - p_true[1:], axis=1)
    return float(np.sqrt(np.mean(e**2))), float(e[-1])


def fidelity(net, grid40_snaps, truth80):
    """(a) 40×40 학습 격자의 κ=0.2 솔버 장 대비 MLP 오차, (b) 80×80 참값 대비 MLP와 격자 지도 오차."""
    out = {}
    n = N_MAP
    xs40 = (np.arange(n) + 0.5) * (L / n)
    XX, YY = np.meshgrid(xs40, xs40, indexing="ij")
    xy40 = np.stack([XX.ravel(), YY.ravel()], axis=1)
    m = NeuralMeasure(net, KAPPA_REF)
    n80 = N_TRUE
    xs80 = (np.arange(n80) + 0.5) * (L / n80)
    XX8, YY8 = np.meshgrid(xs80, xs80, indexing="ij")
    xy80 = np.stack([XX8.ravel(), YY8.ravel()], axis=1)
    for t in (0, 60, 120):
        pred40 = m(t, xy40)
        ref40 = grid40_snaps[t].ravel()
        pred80 = m(t, xy80)
        ref80 = truth80[t].ravel()
        grid_at80 = P.sample(grid40_snaps[t], xy80)
        out[f"t{t}"] = dict(
            mlp_vs_solver40_rms=float(np.sqrt(np.mean((pred40 - ref40) ** 2))),
            mlp_vs_truth80_rms=float(np.sqrt(np.mean((pred80 - ref80) ** 2))),
            grid40_vs_truth80_rms=float(np.sqrt(np.mean((grid_at80 - ref80) ** 2))),
            field_std_truth80=float(ref80.std()),
        )
    return out


def kappa_profile(net, y, p_true, kgrid):
    """센서 자료와 참 경로에서 κ̂별 음의 로그우도. 가우시안 잡음 SIG_S."""
    nll = np.empty(len(kgrid))
    for i, kap in enumerate(kgrid):
        m = NeuralMeasure(net, kap)
        pred = np.array([m(t, p_true[t:t + 1])[0] for t in range(T + 1)])
        nll[i] = 0.5 * np.sum(((y - pred) / SIG_S) ** 2)
    return nll


def main():
    t_all = time.perf_counter()
    p_path, d = P.path_disp()

    truth80, truth_info = P.solve(N_TRUE, P.KAPPA, "asym")
    grid40, grid_info = P.solve(N_MAP, P.KAPPA, "asym")

    X, Y, train_info = build_training_set()
    net, train_wall, train_log = train_net(X, Y)
    n_params = sum(p.numel() for p in net.parameters())

    fid = fidelity(net, grid40, truth80)

    configs = {
        "grid_match_k0.2": GridMeasure(grid40),
        "mlp_k0.2": NeuralMeasure(net, 0.2),
        "mlp_k0.1": NeuralMeasure(net, 0.1),
        "mlp_k0.15": NeuralMeasure(net, 0.15),
        "mlp_k0.3": NeuralMeasure(net, 0.3),
        "mlp_k0.4": NeuralMeasure(net, 0.4),
    }
    results, raw = {}, {}
    for name, meas in configs.items():
        rm, fe = [], []
        t0 = time.perf_counter()
        for s in range(N_RUNS):
            est, p_true, _ = run_pf(s, truth80, meas, d, p_path)
            r, f = rmse_of(est, p_true)
            rm.append(r)
            fe.append(f)
        rm, fe = np.array(rm), np.array(fe)
        raw[name] = rm
        results[name] = dict(
            rmse_mean=float(rm.mean()), rmse_sd=float(rm.std(ddof=1)), rmse_median=float(np.median(rm)),
            final_mean=float(fe.mean()), frac_rmse_gt_2m=float(np.mean(rm > 2.0)),
            wall_s=round(time.perf_counter() - t0, 2),
        )
    for name in configs:
        if name == "grid_match_k0.2":
            continue
        t_res = stats.ttest_rel(raw[name], raw["grid_match_k0.2"])
        results[name]["paired_vs_grid_match_p"] = float(t_res.pvalue)
        results[name]["paired_mean_diff_vs_grid"] = float(np.mean(raw[name] - raw["grid_match_k0.2"]))

    # κ 복원: 참 경로, 센서 자료 프로파일 (실행 30회 중 argmax 분포)
    kgrid = np.linspace(0.05, 0.50, 46)
    argmins = []
    prof_example = None
    t0 = time.perf_counter()
    for s in range(N_RUNS):
        _, p_true, y = run_pf(s, truth80, NeuralMeasure(net, 0.2), d, p_path)
        nll = kappa_profile(net, y, p_true, kgrid)
        argmins.append(float(kgrid[np.argmin(nll)]))
        if s == 0:
            prof_example = dict(kappa=[round(float(k), 3) for k in kgrid], nll=[round(float(v), 2) for v in nll])
    argmins = np.array(argmins)
    kappa_recovery = dict(
        argmax_median=float(np.median(argmins)),
        argmax_q25=float(np.percentile(argmins, 25)),
        argmax_q75=float(np.percentile(argmins, 75)),
        argmax_mean=float(argmins.mean()),
        argmax_sd=float(argmins.std(ddof=1)),
        frac_within_0p05=float(np.mean(np.abs(argmins - 0.2) <= 0.05)),
        wall_s=round(time.perf_counter() - t0, 2),
        example_run0_profile=prof_example,
    )

    out = dict(
        device=str(DEVICE),
        versions=dict(torch=torch.__version__, numpy=np.__version__, python=sys.version.split()[0]),
        threads=torch.get_num_threads(),
        mps_available=bool(torch.backends.mps.is_available()),
        setup=dict(
            kappa_train=list(KAPPA_TRAIN), kappa_heldout=KAPPA_REF, n_train_samples=int(X.shape[0]),
            net=dict(inputs=4, width=64, depth=3, activation="SiLU", n_params=n_params),
            train=dict(steps=TRAIN_STEPS, batch=BATCH, optimizer="Adam lr 3e-3 cosine", seed=0,
                       wall_s=round(train_wall, 2), final_mse=train_log[-1]["mse"], log=train_log),
            n_particles=N_PART, n_runs=N_RUNS, sig_odo=SIG_O, sig_sensor=SIG_S,
        ),
        fidelity=fid,
        pf_results=results,
        kappa_recovery=kappa_recovery,
        solver_train_info=train_info,
        total_wall_s=round(time.perf_counter() - t_all, 2),
    )
    (HERE / "extra_neural_field_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("device", DEVICE, "torch", torch.__version__, "params", n_params, "train wall", round(train_wall, 1), "s")
    print("train final mse", train_log[-1]["mse"])
    for k, v in fid.items():
        print("fidelity", k, {a: round(b, 4) for a, b in v.items()})
    for name, r in results.items():
        print(f"{name:18s} RMSE {r['rmse_mean']:.3f}±{r['rmse_sd']:.3f} (median {r['rmse_median']:.3f}) "
              f"final {r['final_mean']:.3f} >2m {r['frac_rmse_gt_2m']:.2f} "
              f"p_vs_grid {r.get('paired_vs_grid_match_p', float('nan')):.3g} wall {r['wall_s']} s")
    print("kappa argmax median", kappa_recovery["argmax_median"], "IQR",
          kappa_recovery["argmax_q25"], kappa_recovery["argmax_q75"])
    print("total wall", out["total_wall_s"], "s")


if __name__ == "__main__":
    main()
