"""추가 검증 2 (contact_disassembly): 시드 분산과 초기조건 분산을 분리한 numpy MLP toy. 실행: python3 analysis/extra_seed_init_var.py

질문: 논문 시험은 초기조건 50개를 모든 시드에서 고정한다(04 D10). 이때 "1500 독립 rollouts" 가정으로 잡은 표준오차가
얼마나 과소(또는 과대) 추정되는가?

toy: 상태 s ∈ [−1, 1]², 참 행동 g(s) = sin(2 s1) + 0.5 s2. 정책 = MLP(2→16→1, tanh), 학습 데모 25개(잡음 0.05),
  Adam 800 스텝. 성공 = |π(s) − g(s)| < 0.10 인 초기조건(롤아웃 대신 한 스텝 판정으로 단순화).
한 번의 반복(replication)에서 시드 10개(데모와 초기화가 시드마다 다름)를 학습하고 두 방식으로 평가한다.
  고정 프로토콜  : 모든 시드가 같은 시험 초기조건 50개 (논문과 같음)
  재표집 프로토콜: 시드마다 새 초기조건 50개 (초기조건 분산까지 포함)
"참 SD"는 반복 100번에서 시드 10개 평균 성공률의 표준편차(시드와 초기조건을 모두 새로 뽑은 값)이다.
비교: 각 프로토콜의 "보고되는 표준오차"(시드 간 표준편차/√10) 평균 대 참 SD, 그리고 "1500 독립" 이항 표준오차.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N_SEEDS, N_INIT, N_DEMO, N_REP = 10, 50, 25, 100
TAU, H, STEPS, LR = 0.10, 16, 800, 0.01
FIXED_INIT_SEED = 999


def g(s):
    return np.sin(2 * s[:, 0]) + 0.5 * s[:, 1]


def train_policy(rng):
    s = rng.uniform(-1, 1, size=(N_DEMO, 2))
    y = g(s) + 0.05 * rng.normal(size=N_DEMO)
    W1 = rng.normal(0, 1 / np.sqrt(2), (2, H)); b1 = np.zeros(H)
    W2 = rng.normal(0, 1 / np.sqrt(H), (H, 1)); b2 = np.zeros(1)
    P = [W1, b1, W2, b2]
    M = [np.zeros_like(p) for p in P]; V = [np.zeros_like(p) for p in P]
    yy = y[:, None]
    for it in range(1, STEPS + 1):
        W1, b1, W2, b2 = P
        h = np.tanh(s @ W1 + b1)
        out = h @ W2 + b2
        go = 2 * (out - yy) / N_DEMO
        gW2, gb2 = h.T @ go, go.sum(0)
        gh = (go @ W2.T) * (1 - h ** 2)
        gW1, gb1 = s.T @ gh, gh.sum(0)
        Gs = [gW1, gb1, gW2, gb2]
        for i in range(4):
            M[i] = 0.9 * M[i] + 0.1 * Gs[i]
            V[i] = 0.999 * V[i] + 0.001 * Gs[i] ** 2
            P[i] = P[i] - LR * (M[i] / (1 - 0.9 ** it)) / (np.sqrt(V[i] / (1 - 0.999 ** it)) + 1e-8)
    return P


def success(P, s):
    W1, b1, W2, b2 = P
    pred = (np.tanh(s @ W1 + b1) @ W2 + b2)[:, 0]
    return (np.abs(pred - g(s)) < TAU).astype(float)


def main():
    fixed_init = np.random.default_rng(FIXED_INIT_SEED).uniform(-1, 1, size=(N_INIT, 2))
    rep_fixed_mean, rep_fixed_se, rep_resamp_mean, rep_resamp_se, rep_seed_rates = [], [], [], [], []
    se3_fixed, se3_resamp, mean3_fixed, mean3_resamp = [], [], [], []
    for r in range(N_REP):
        rng_r = np.random.default_rng(20000 + r)
        init_r = rng_r.uniform(-1, 1, size=(N_INIT, 2))          # 이 반복의 새 초기조건 (재표집 프로토콜)
        fixed_rates, resamp_rates = [], []
        for k in range(N_SEEDS):
            rng_k = np.random.default_rng(100000 * (r + 1) + k)
            P = train_policy(rng_k)
            fixed_rates.append(success(P, fixed_init).mean())
            resamp_rates.append(success(P, init_r).mean())
        fixed_rates, resamp_rates = np.array(fixed_rates), np.array(resamp_rates)
        rep_fixed_mean.append(fixed_rates.mean()); rep_fixed_se.append(fixed_rates.std(ddof=1) / np.sqrt(N_SEEDS))
        rep_resamp_mean.append(resamp_rates.mean()); rep_resamp_se.append(resamp_rates.std(ddof=1) / np.sqrt(N_SEEDS))
        rep_seed_rates.append(resamp_rates)
        mean3_fixed.append(fixed_rates[:3].mean()); mean3_resamp.append(resamp_rates[:3].mean())
        se3_fixed.append(fixed_rates[:3].std(ddof=1) / np.sqrt(3))
        se3_resamp.append(resamp_rates[:3].std(ddof=1) / np.sqrt(3))
    rep_fixed_mean, rep_resamp_mean = np.array(rep_fixed_mean), np.array(rep_resamp_mean)
    p_hat = float(rep_resamp_mean.mean())
    # 참 SD: 반복마다 시드와 초기조건을 모두 새로 뽑은 "전체" 평균의 SD (재표집 프로토콜의 반복 간 SD)
    true_sd = float(rep_resamp_mean.std(ddof=1))
    fixed_sd_across = float(rep_fixed_mean.std(ddof=1))            # 고정 초기조건의 반복 간 SD (시드만 변함)
    rep_fixed_se_mean = float(np.mean(rep_fixed_se))
    rep_resamp_se_mean = float(np.mean(rep_resamp_se))
    binom_1500 = float(np.sqrt(p_hat * (1 - p_hat) / 1500))
    se3_f, se3_r = float(np.mean(se3_fixed)), float(np.mean(se3_resamp))
    true_sd3_resamp = float(np.std(mean3_resamp, ddof=1))
    true_sd3_fixed = float(np.std(mean3_fixed, ddof=1))
    binom_500 = float(np.sqrt(p_hat * (1 - p_hat) / (N_SEEDS * N_INIT)))
    out = {
        "setup": {"n_replications": N_REP, "n_seeds": N_SEEDS, "n_init": N_INIT, "n_demo": N_DEMO,
                  "tau": TAU, "mlp": f"2→{H}→1 tanh, Adam {STEPS} steps, lr {LR}", "fixed_init_seed": FIXED_INIT_SEED,
                  "noise": 0.05, "truth": "g(s)=sin(2 s1)+0.5 s2"},
        "success_rate_mean": round(p_hat, 4),
        "true_sd_grand_mean_resampled_inits_and_seeds": round(true_sd, 4),
        "fixed_init_protocol": {
            "sd_across_replications_of_grand_mean(seed+init fixed)": round(fixed_sd_across, 4),
            "mean_reported_SE_(seed_sd/sqrt10)": round(rep_fixed_se_mean, 4),
            "ratio_reported_to_true_sd": round(rep_fixed_se_mean / true_sd, 3),
        },
        "resampled_init_protocol": {
            "mean_reported_SE_(seed_sd/sqrt10)": round(rep_resamp_se_mean, 4),
            "ratio_reported_to_true_sd": round(rep_resamp_se_mean / true_sd, 3),
        },
        "three_seed_version_first3": {
            "true_sd_fixed_init": round(true_sd3_fixed, 4), "fixed_init_reported_SE": round(se3_f, 4),
            "fixed_ratio_to_true_sd": round(se3_f / true_sd3_fixed, 3),
            "true_sd_resampled": round(true_sd3_resamp, 4), "resampled_reported_SE": round(se3_r, 4),
            "resampled_ratio_to_true_sd": round(se3_r / true_sd3_resamp, 3)},
        "naive_binomial_SE": {"N=1500": round(binom_1500, 4), "N=500": round(binom_500, 4),
                              "ratio_1500_to_true_sd": round(binom_1500 / true_sd, 3)},
    }
    # 같은 반복에서 "초기조건을 고정했을 때"와 "재표집했을 때"의 평균 차이 (고정 프로토콜의 편향 여부)
    out["mean_diff_fixed_minus_resampled"] = round(float(np.mean(rep_fixed_mean - rep_resamp_mean)), 4)
    path = ROOT / "analysis" / "extra_seed_init_var_results.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    plot(rep_fixed_mean, rep_resamp_mean, rep_fixed_se, rep_resamp_se, true_sd)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print("wrote", path)


def plot(fixed_mean, resamp_mean, fixed_se, resamp_se, true_sd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                              "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if Path(f).exists()), None)
    if font:
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 8.5, "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    bins = np.linspace(min(fixed_mean.min(), resamp_mean.min()) - 0.01,
                       max(fixed_mean.max(), resamp_mean.max()) + 0.01, 30)
    ax.hist(fixed_mean, bins=bins, alpha=0.6, color="#9a9a9f", label="고정 초기조건 (논문 방식)")
    ax.hist(resamp_mean, bins=bins, alpha=0.6, color="#2a78d6", label="초기조건 재표집")
    ax.set_xlabel("시드 10개 평균 성공률 (반복 100회)")
    ax.set_ylabel("반복 수")
    ax.set_title(f"반복 간 분산: 고정 SD {fixed_mean.std(ddof=1):.3f}, 재표집 SD {true_sd:.3f}", fontsize=8.5)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "report" / "figures" / "extra_seed_init_var.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
