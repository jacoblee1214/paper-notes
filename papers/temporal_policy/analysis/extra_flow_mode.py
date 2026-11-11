"""학습된 이력 조건부 흐름의 모드 붕괴 검사 (torch, CPU 4스레드, 수 분 이내).

질문: 두 봉우리 합성 데이터로 학습한 결정적 이력 흐름(ODE)은 두 모드 사이의 평균으로 붕괴하는가?
      확률적 샘플러(SDE, g = 0.1, 1.0)는 두 모드를 회복하는가? 결정적 샘플러의 끝점은 샘플링 시드마다 분산 0인가?

설정 (transport_toy.py의 상태 과정, 청크 길이 H = 16, 시차 d = 14, 보간 ε = 1, 두 봉우리 조건부 목표를 그대로 쓴다):
  - 이력 c: 단위분산 AR(1) (φ = 0.99)의 청크, 목표 x1 | c: 평균 A·c ± δ (δ = 0.5·linspace(−1, 1, 16)), 잡음 0.2.
    이력이 가리키는 모드와 같은 모드를 고를 확률이 1 − h. h = 0.5 (이력이 모드를 정하지 않음)가 주 설정,
    h = 0.02 (이력이 모드를 거의 정함)는 대조 설정.
  - 보간 x_t = (1−t)c + t·x1 + ε(1−t)√t·z. 학습 목표는 드리프트 (x1 − x_t)/(1−t)이며, 손실은 (1−t)²로 가중한 MSE.
  - 드리프트 망 b_θ(x_t, c, t): 입력 33 → 128 → 128 → 16, SiLU. Adam 1e-3, 코사인 감쇠, 배치 512, 6000 스텝.
    학습 시드 3개 (0, 1, 2), 각 설정.
  - 점수: 학습된 b에서 ∇log p = (t·b − (x_t − c)) / (ε² t (1−t)) (transport_toy의 식 5 항등식)로 계산한다.
    결정적 샘플러 g = 0은 b − ½ε²(1−t)²∇log p (코드와 같은 보정), 보정 없는 ODE도 함께 본다.
    SDE는 b + ½(g² − ε²(1−t)²)∇log p + g√dt·z. 시작 스텝 t = 0의 보정항은 0으로 둔다.
  - 평가: 이력 1000개, 단계 수 n ∈ {10, 16}. 시드 분산은 이력 200개에 샘플러 시드 5개.
  - 지표: 끝점의 모드 라벨 (미소 값 ±δ/2 기준 +, −, 사이), 조건부 평균 mu + (2p₊−1)δ 대비 RMS,
    가장 가까운 모드 중심까지의 거리 (per-dim 제곱), 샘플러 시드 간 끝점 SD.
가정·한계: 점수는 학습된 드리프트에서 식 5로 계산한다(오라클 점수를 쓰지 않음). 학습 데이터는 무한 샘플이다.
  조건부 목표는 가우시안 두 봉우리이며 실제 로봇 데이터가 아니다. 학습 손실은 transport_toy의 식과 같은 보간을 쓰지만
  논문의 학습 목적 함수와 같다고 보증하지 않는다.
실행: python3 analysis/extra_flow_mode.py -> analysis/extra_flow_mode_results.json
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import transport_toy as TT  # noqa: E402

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "4"  # transport_toy 임포트가 1로 덮어쓰므로 되돌림

torch.set_num_threads(4)
DEVICE = torch.device("cpu")

H = TT.H
EPS = TT.EPS
PHI = 0.99
M_AMP = 0.5
SIGMA = 0.2
TRAIN_STEPS = 6000
BATCH = 512
TRAIN_SEEDS = (0, 1, 2)
N_EVAL = 1000
N_SEED_HIST = 200
N_SAMPLER_SEEDS = 5
N_STEPS = (10, 16)
G_LIST = (0.1, 1.0)
H_LIST = (0.5, 0.02)

A, _ = TT.gauss_conditional(PHI)
S00, _, _ = TT.joint_blocks(PHI)
LC = np.linalg.cholesky(S00 + 1e-10 * np.eye(H))
DELTA = M_AMP * np.linspace(-1, 1, H)
DHAT = DELTA / np.linalg.norm(DELTA)
HALF = np.linalg.norm(DELTA) / 2


def sample_hist(rng, n):
    return rng.standard_normal((n, H)) @ LC.T


def p_plus_of(c, h):
    slope_sign = np.sign(c[:, -1] - c[:, 0])
    return np.where(slope_sign > 0, 1 - h, h)


def sample_target(rng, c, h):
    mu = c @ A.T
    plus = rng.random(len(c)) < p_plus_of(c, h)
    mode = np.where(plus, 1.0, -1.0)
    return mu + mode[:, None] * DELTA + SIGMA * rng.standard_normal(mu.shape)


class VNet(torch.nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2 * H + 1, width), torch.nn.SiLU(),
            torch.nn.Linear(width, width), torch.nn.SiLU(),
            torch.nn.Linear(width, H),
        )

    def forward(self, x, c, t):
        tt = torch.full((x.shape[0], 1), float(t), dtype=x.dtype)
        return self.net(torch.cat([x, c, tt], dim=1))


def train(h, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(1000 + seed)
    net = VNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_STEPS, eta_min=1e-5)
    t0 = time.perf_counter()
    last = None
    for _ in range(TRAIN_STEPS):
        c = sample_hist(rng, BATCH)
        x1 = sample_target(rng, c, h)
        t = rng.uniform(0.0, 0.95, BATCH)[:, None]
        z = rng.standard_normal(c.shape)
        xt = (1 - t) * c + t * x1 + EPS * (1 - t) * np.sqrt(t) * z
        target = (x1 - xt) / (1 - t)
        ct, xtt, tg = (torch.from_numpy(a).float() for a in (c, xt, target))
        tt = torch.from_numpy(t).float()
        pred = _forward_batch(net, xtt, ct, tt)
        loss = torch.mean(((pred - tg) * (1 - tt)) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        last = float(loss.item())
    return net.eval(), time.perf_counter() - t0, last


def _forward_batch(net, x, c, tt):
    return net.net(torch.cat([x, c, tt], dim=1))


def drift_np(net, x, c, t):
    with torch.no_grad():
        out = net(torch.from_numpy(x).float(), torch.from_numpy(c).float(), t)
    return out.double().numpy()


def score_from_drift(b, x, c, t):
    if t == 0.0:
        return np.zeros_like(x)
    return (t * b - (x - c)) / (EPS**2 * t * (1 - t))


def oracle_field(c, h):
    """오라클 (b, 점수) 함수. 두 봉우리 조건부 목표를 닫힌 식으로 계산한다 (transport_toy.tp_drift)."""
    mu = c @ A.T
    pp = p_plus_of(c, h)
    mus = np.stack([mu + DELTA, mu - DELTA], axis=1)
    logpi = np.log(np.stack([pp, 1 - pp], axis=1) + 1e-300)
    Sig = SIGMA**2 * np.eye(H)

    def f(x, t):
        if t == 0.0:
            return TT.tp_drift(x, c, 1e-12, mus, logpi, Sig)[0], np.zeros_like(x)
        b, sc = TT.tp_drift(x, c, t, mus, logpi, Sig, eps=EPS)
        return b, sc
    return f


def learned_field(net, c, source):
    """source='eq5': 학습 드리프트 + 식 5로 얻은 점수 (주 설정). source='oracle': 학습 드리프트 + 오라클 점수 (진단)."""
    orc = oracle_field(c, H_CURRENT[0]) if source == "oracle" else None

    def f(x, t):
        b = drift_np(net, x, c, t)
        if source == "oracle" and t > 0.0:
            _, sc = orc(x, t)
        else:
            sc = score_from_drift(b, x, c, t)
        return b, sc
    return f


H_CURRENT = [None]


def sample_endpoints(field, c, n, mode, g=None, rng=None):
    """field(x, t) -> (b, score). mode: 'ode_corr', 'ode_raw', 'sde'. 반환: 끝점 (이력 수, H)."""
    x = c.copy()
    dt = 1.0 / n
    for i in range(n):
        t = i * dt
        b, s = field(x, t)
        if mode == "ode_corr":
            b = b - 0.5 * (EPS * (1 - t)) ** 2 * (s if t > 0 else 0.0)
        elif mode == "sde":
            b = b + 0.5 * (g**2 - (EPS * (1 - t)) ** 2) * (s if t > 0 else 0.0)
        x = x + dt * b
        if mode == "sde":
            x = x + g * np.sqrt(dt) * rng.standard_normal(x.shape)
    return x


def labels(x, c):
    mu = c @ A.T
    proj = (x - mu) @ DHAT
    plus = proj > HALF
    minus = proj < -HALF
    return float(plus.mean()), float(minus.mean()), float((~(plus | minus)).mean())


def mode_dist(x, c):
    mu = c @ A.T
    dp = np.mean((x - (mu + DELTA)) ** 2, axis=1)
    dm = np.mean((x - (mu - DELTA)) ** 2, axis=1)
    return float(np.mean(np.minimum(dp, dm)))


def run_variants(field_for, c, c_seed, m_true, n, names):
    rows = {}
    for label, source, name in names:
        kind = "sde" if name.startswith("sde") else name
        g = float(name[len("sde_g"):]) if name.startswith("sde") else None
        if source == "oracle_full":
            fld = lambda x, t, c=c: oracle_field(c, H_CURRENT[0])(x, t)  # noqa: E731
            fld_seed = lambda x, t, c=c_seed: oracle_field(c, H_CURRENT[0])(x, t)  # noqa: E731
        else:
            fld = field_for(c, source)
            fld_seed = field_for(c_seed, source)
        x = sample_endpoints(fld, c, n, kind, g=g, rng=np.random.default_rng(4242 + n))
        fp, fm, fb = labels(x, c)
        reps = []
        for s_ in range(N_SAMPLER_SEEDS):
            rs = np.random.default_rng(9000 + s_)
            reps.append(sample_endpoints(fld_seed, c_seed, n, kind, g=g, rng=rs))
        reps = np.stack(reps)
        rows[f"n{n}_{label}"] = dict(
            frac_plus=fp, frac_minus=fm, frac_between=fb,
            rms_to_conditional_mean=float(np.sqrt(np.mean((x - m_true) ** 2))),
            mean_sq_dist_to_nearest_mode_center=mode_dist(x, c),
            sampler_seed_sd_per_dim=float(np.mean(reps.std(axis=0, ddof=1))),
        )
    return rows


MAIN_NAMES = [("ode_corr", "eq5", "ode_corr"), ("ode_raw", "eq5", "ode_raw"),
              ("sde_g0.1", "eq5", "sde_g0.1"), ("sde_g1.0", "eq5", "sde_g1.0")]
DIAG_NAMES = [("diag_ode_corr_oracle_score", "oracle", "ode_corr"),
              ("diag_sde_g1.0_oracle_score", "oracle", "sde_g1.0"),
              ("diag_sde_g0.1_oracle_score", "oracle", "sde_g0.1"),
              ("diag_oracle_full_sde_g1.0", "oracle_full", "sde_g1.0"),
              ("diag_oracle_full_ode_corr", "oracle_full", "ode_corr")]


def evaluate(net, h):
    H_CURRENT[0] = h
    rng_eval = np.random.default_rng(777)
    c = sample_hist(rng_eval, N_EVAL)
    mu = c @ A.T
    pp = p_plus_of(c, h)
    m_true = mu + (2 * pp - 1)[:, None] * DELTA
    truth_frac_plus = float(np.mean(pp))
    out = {"truth_frac_plus": truth_frac_plus, "truth_frac_minus": 1 - truth_frac_plus, "rows": {}}
    c_seed = c[:N_SEED_HIST]

    def field_for(cc, source):
        return learned_field(net, cc, "eq5" if source == "eq5" else "oracle")

    for n in N_STEPS:
        out["rows"].update(run_variants(field_for, c, c_seed, m_true, n, MAIN_NAMES))
        out["rows"].update(run_variants(field_for, c, c_seed, m_true, n, DIAG_NAMES))
    return out


def main():
    t_all = time.perf_counter()
    results = {}
    for h in H_LIST:
        per_seed = []
        for s in TRAIN_SEEDS:
            net, wall, last = train(h, s)
            ev = evaluate(net, h)
            ev["train_seed"] = s
            ev["train_wall_s"] = round(wall, 2)
            ev["final_loss"] = last
            per_seed.append(ev)
            r = ev["rows"]
            print(f"h={h} seed={s} train {wall:.1f}s loss {last:.4f} | "
                  f"n10 ode_corr between {r['n10_ode_corr']['frac_between']:.3f} "
                  f"+ {r['n10_ode_corr']['frac_plus']:.3f} - {r['n10_ode_corr']['frac_minus']:.3f} "
                  f"sde1.0 between {r['n10_sde_g1.0']['frac_between']:.3f} "
                  f"+ {r['n10_sde_g1.0']['frac_plus']:.3f} - {r['n10_sde_g1.0']['frac_minus']:.3f}")
        results[f"h={h}"] = dict(per_train_seed=per_seed)
    out = dict(
        device=str(DEVICE),
        versions=dict(torch=torch.__version__, numpy=np.__version__, python=sys.version.split()[0]),
        threads=torch.get_num_threads(),
        mps_available=bool(torch.backends.mps.is_available()),
        setup=dict(H=H, lag=TT.LAG, phi=PHI, delta_amp=M_AMP, sigma=SIGMA, eps=EPS,
                   train_steps=TRAIN_STEPS, batch=BATCH, optimizer="Adam 1e-3 cosine",
                   net="33-128-128-16 SiLU", train_seeds=list(TRAIN_SEEDS),
                   n_eval_histories=N_EVAL, n_seed_histories=N_SEED_HIST, n_sampler_seeds=N_SAMPLER_SEEDS,
                   n_steps=list(N_STEPS), g=list(G_LIST)),
        results=results,
        total_wall_s=round(time.perf_counter() - t_all, 2),
    )
    (HERE / "extra_flow_mode_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("total wall", out["total_wall_s"], "s")


if __name__ == "__main__":
    main()
