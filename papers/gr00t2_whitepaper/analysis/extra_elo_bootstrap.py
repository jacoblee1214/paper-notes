"""블라인드 쌍 비교 Elo의 구간 폭이 과제 수와 과제당 시도 수에 따라 어떻게 변하는지 보는 합성 데이터 실험 (numpy, CPU).

백서 부록 D의 집계 방식을 따릅니다. 실제 데이터는 공개되지 않았으므로 모든 값은 합성입니다.
- 한 시도에서 4개 체크포인트가 같은 장면을 한 번씩 돌고, 쌍 6개가 매치 6개를 만듭니다.
- 매치는 Bradley-Terry 확률로 승패가 나고, 독립적으로 동점(기본 33%)이 나옵니다.
- 같은 과제의 체크포인트별 효과는 시도 사이에 공유됩니다(장면 효과). 과제 단위 재표본은 이 공유 변동을 잡습니다.
- 쌍·과제마다 가중치 합을 1로 맞춘 뒤(부록 식 7), 최대 우도 Bradley-Terry를 MM으로 풉니다. 기준점(anchor)에 가상 동점 1개를 둡니다.
- 구간은 과제를 복원추출하는 군집 부트스트랩(B = 500, 백분위 2.5–97.5)입니다.
- 동점 처리 두 가지를 비교합니다: 동점을 반 승으로 셈(백서 방식), 동점 매치를 버림.

실행: python3 analysis/extra_elo_bootstrap.py  →  analysis/extra_elo_bootstrap_results.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "extra_elo_bootstrap_results.json"

ARMS = ["GR00T 2", "GR00T 2 Turbo", "GR00T 1.7", "pi0.5"]
# 백서 표 13의 G2 기준 차이를 그대로 쓴 참값. Turbo는 표의 +12 (early checkpoint)를 씀.
ELO_TRUE = np.array([1600.0, 1588.0, 1505.0, 1364.0])
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
TAU = 0.33  # 동점 비율 (백서 V2 35%, V1 30%의 중간)
SIGMA_GRID = [0.0, 50.0, 100.0, 150.0]  # 과제·체크포인트 효과의 표준편차 (Elo 단위)
TASK_GRID = [4, 8, 16, 24, 38, 64]
TRIAL_GRID = [1, 4, 26]  # 26 = 백서 V2의 6,035 매치를 4개 체크포인트·38과제로 맞춘 근사치
TIE_MODES = ["half", "drop"]
N_DATA = 20  # 설정당 합성 데이터 수
B = 500  # 부트스트랩 재표본 수 (백서와 같음)
ANCHOR_ELO = 1500.0
N_ITER = 500
BASE_SEED = 20261112


def elo_gap(p, a, b):
    return 400.0 * np.log10(p[..., a] / p[..., b])


def simulate(rng, T, K, sigma):
    """Return per-(task, pair) fraction of wins for arm i (decisive or tie-half) and tie/decisive counts."""
    eff = ELO_TRUE[None, :] + sigma * rng.standard_normal((T, len(ARMS)))  # 과제·체크포인트 효과
    gi = np.array([i for i, _ in PAIRS])
    gj = np.array([j for _, j in PAIRS])
    gap = eff[:, gi] - eff[:, gj]  # (T, 6)
    p_win = 1.0 / (1.0 + 10.0 ** (-gap / 400.0))  # 쌍 i가 j를 이길 확률 (동점 아님)
    shape = (T, K, len(PAIRS))
    tie = rng.random(shape) < TAU
    win_i = rng.random(shape) < p_win[:, None, :]
    win_i &= ~tie
    lose = ~tie & ~win_i
    return win_i.sum(axis=1), lose.sum(axis=1), tie.sum(axis=1)  # each (T, 6)


def pair_stats(win_i, lose, tie, K, mode):
    """Per (task, pair): fraction f of i's wins, and availability n (1 or 0)."""
    if mode == "half":
        f = (win_i + 0.5 * tie) / K
        n = np.ones_like(f)
    else:  # drop: 동점 매치를 버리고 승패만 씀
        d = win_i + lose
        n = (d > 0).astype(float)
        f = np.where(d > 0, win_i / np.maximum(d, 1), 0.0)
    return f, n


def build_matrices(f, n):
    """Return W (T, 5, 5) and N (T, 5, 5) with arm 4 = anchor. W[t,i,j] = weight of i over j."""
    T = f.shape[0]
    W = np.zeros((T, 5, 5))
    N = np.zeros((T, 5, 5))
    for k, (i, j) in enumerate(PAIRS):
        W[:, i, j] = f[:, k] * n[:, k]
        W[:, j, i] = (1.0 - f[:, k]) * n[:, k]
        N[:, i, j] = n[:, k]
        N[:, j, i] = n[:, k]
    return W, N


def anchor_prior():
    """각 체크포인트에 기준점과의 가상 동점 1개 (가중치 1, 승 0.5 + 패 0.5)."""
    Wp = np.zeros((5, 5))
    Np = np.zeros((5, 5))
    for i in range(4):
        Wp[i, 4] = 0.5
        Wp[4, i] = 0.5
        Np[i, 4] = 1.0
        Np[4, i] = 1.0
    return Wp, Np


def fit_bt(W, N):
    """MM 반복(Hunter 2004). W, N: (B, 5, 5). 기준점 강도는 1로 고정. Return p (B, 5) and last change."""
    Bn = W.shape[0]
    p = np.ones((Bn, 5))
    Wsum = W.sum(axis=2)
    change = np.inf
    for _ in range(N_ITER):
        S = p[:, :, None] + p[:, None, :]
        D = (N / S).sum(axis=2)
        p_new = Wsum / D
        p_new[:, 4] = 1.0
        change = float(np.max(np.abs(np.log(p_new[:, :4]) - np.log(p[:, :4]))))
        p = p_new
    return p, change


def run_config(T, K, sigma, mode, seed):
    rng = np.random.default_rng(seed)
    Wp, Np = anchor_prior()
    widths_gp, widths_gg, est_gp, est_gg, excl_gp, excl_gg, cov_gp = [], [], [], [], [], [], []
    last_changes = []
    for _ in range(N_DATA):
        win_i, lose, tie = simulate(rng, T, K, sigma)
        f, n = pair_stats(win_i, lose, tie, K, mode)
        W, N = build_matrices(f, n)
        # 부트스트랩: 과제를 복원추출하고 과제 단위 행렬을 다시 합침 (가중치는 재표본에서 다시 계산한 셈)
        idx = rng.integers(0, T, size=(B, T))
        mult = np.zeros((B, T))
        np.add.at(mult, (np.arange(B)[:, None], idx), 1.0)
        mult = np.vstack([np.ones((1, T)), mult])  # 행 0 = 원자료
        Wb = np.einsum("bt,tij->bij", mult, W) + Wp
        Nb = np.einsum("bt,tij->bij", mult, N) + Np
        p, change = fit_bt(Wb, Nb)
        last_changes.append(change)
        g_gp = elo_gap(p, 0, 3)  # GR00T 2 - pi0.5
        g_gg = elo_gap(p, 0, 2)  # GR00T 2 - GR00T 1.7
        lo_gp, hi_gp = np.percentile(g_gp[1:], [2.5, 97.5])
        lo_gg, hi_gg = np.percentile(g_gg[1:], [2.5, 97.5])
        widths_gp.append(hi_gp - lo_gp)
        widths_gg.append(hi_gg - lo_gg)
        est_gp.append(g_gp[0])
        est_gg.append(g_gg[0])
        excl_gp.append(lo_gp > 0)
        excl_gg.append(lo_gg > 0)
        cov_gp.append(lo_gp <= 236.0 <= hi_gp)
    return {
        "median_width_G2_vs_pi05": float(np.median(widths_gp)),
        "median_width_G2_vs_GR17": float(np.median(widths_gg)),
        "mean_est_G2_vs_pi05": float(np.mean(est_gp)),
        "mean_est_G2_vs_GR17": float(np.mean(est_gg)),
        "power_excl0_G2_vs_pi05": float(np.mean(excl_gp)),
        "power_excl0_G2_vs_GR17": float(np.mean(excl_gg)),
        "coverage_of_236_G2_vs_pi05": float(np.mean(cov_gp)),
        "max_last_mm_change_log": float(np.max(last_changes)),
    }


def main():
    results = {
        "setup": {
            "arms": ARMS,
            "elo_true_offsets_from_G2": {a: float(ELO_TRUE[0] - e) for a, e in zip(ARMS, ELO_TRUE)},
            "tie_rate": TAU,
            "pairs_per_trial": len(PAIRS),
            "bootstrap_resamples": B,
            "datasets_per_config": N_DATA,
            "anchor": "fixed strength 1 (Elo 1500) with one virtual half-win/half-loss per checkpoint",
            "effect_sigma_elo_grid": SIGMA_GRID,
            "tasks_grid": TASK_GRID,
            "trials_per_task_grid": TRIAL_GRID,
            "note": "all data synthetic; trials per task of the whitepaper are not reported",
        },
        "configs": [],
    }
    cfg_id = 0
    for mode in TIE_MODES:
        for sigma in SIGMA_GRID:
            for K in TRIAL_GRID:
                for T in TASK_GRID:
                    seed = [BASE_SEED, cfg_id]
                    r = run_config(T, K, sigma, mode, seed)
                    results["configs"].append({"tie_mode": mode, "sigma_elo": sigma, "trials_per_task": K,
                                               "tasks": T, **r})
                    cfg_id += 1
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.name} with {len(results['configs'])} configs")


if __name__ == "__main__":
    main()
