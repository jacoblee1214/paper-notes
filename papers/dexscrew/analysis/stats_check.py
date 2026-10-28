"""Table I(nut-bolt), Table II(screwdriver) 요약 통계 점검 (CPU).

1) 재구성 가능성: 시행 10회의 progress ratio가 g% 단위(예: 12.5% = 8회전 중 k회)라고 가정할 때,
   보고된 (평균, 표준편차)를 만드는 10개 값의 조합이 존재하는지 전수 탐색한다 (GRIMMER식 점검).
2) 유의성: 핵심 비교(촉각+히스토리 vs 히스토리만 등)에 대해 요약 통계 기반 Welch t-test,
   그리고 재구성된 분포가 있으면 순열 검정(exact이 아니라 Monte Carlo 20만 회)을 계산한다.
실행: python3 analysis/stats_check.py → analysis/stats_check_results.json
"""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
N = 10
# (조건, 물체) -> (mean, sd); sd None 은 표에 없음
NUT = {
    ("none", "square"): (63.75, 33.05), ("none", "triangular"): (30.00, 39.62),
    ("none", "hexagonal"): (75.00, 29.46), ("none", "cross"): (63.75, 33.05),
    ("tactile", "square"): (62.50, 33.85), ("tactile", "triangular"): (66.25, 27.67),
    ("tactile", "hexagonal"): (75.00, 16.67), ("tactile", "cross"): (82.50, 10.54),
    ("hist", "square"): (87.50, 20.41), ("hist", "triangular"): (80.00, 32.91),
    ("hist", "hexagonal"): (85.00, 32.17), ("hist", "cross"): (100.00, 0.00),
    ("tactile+hist", "square"): (97.50, 7.91), ("tactile+hist", "triangular"): (96.25, 8.44),
    ("tactile+hist", "hexagonal"): (95.00, 10.54), ("tactile+hist", "cross"): (98.75, 3.95),
}
SCREW = {
    "direct_sim2real": (41.60, 26.21), "expert_replay": (50.80, 19.27),
    "none": (69.20, 35.25), "tactile": (67.63, 35.65),
    "hist": (87.50, 18.61), "tactile+hist": (95.00, 13.24),
}


def reconstruct(mean, sd, step, n=N, tol_m=0.005, tol_s=0.005):
    levels = np.arange(0, 100 + 1e-9, step)
    sols = []
    for combo in itertools.combinations_with_replacement(levels, n):
        a = np.array(combo)
        if abs(a.mean() - mean) <= tol_m + 1e-9 and abs(a.std(ddof=1) - sd) <= tol_s + 1e-9:
            sols.append(combo)
    return sols


def welch(a, b):
    (m1, s1), (m2, s2) = a, b
    t, p = stats.ttest_ind_from_stats(m1, s1, N, m2, s2, N, equal_var=False)
    return float(t), float(p)


def perm_test(x, y, iters=200_000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    obs = x.mean() - y.mean()
    pooled = np.concatenate([x, y])
    cnt = 0
    for _ in range(iters):
        rng.shuffle(pooled)
        if abs(pooled[:len(x)].mean() - pooled[len(x):].mean()) >= abs(obs) - 1e-12:
            cnt += 1
    return (cnt + 1) / (iters + 1)


def main():
    out = {"nut_reconstruction": {}, "nut_tests": {}, "screw_tests": {}}
    recon = {}
    for key, (m, s) in NUT.items():
        found = None
        for step in (12.5, 10.0, 25.0, 20.0):
            sols = reconstruct(m, s, step)
            if sols:
                found = {"step_pct": step, "n_solutions": len(sols), "example": list(sols[0])}
                recon[key] = sols
                break
        out["nut_reconstruction"]["/".join(key)] = found or "재구성 불가 (12.5/10/25/20% 단위)"

    for nut in ("square", "triangular", "hexagonal", "cross"):
        for a, b in (("tactile+hist", "hist"), ("tactile", "none"), ("hist", "none")):
            t, p = welch(NUT[(a, nut)], NUT[(b, nut)])
            entry = {"welch_t": round(t, 3), "welch_p": round(p, 4)}
            if (a, nut) in recon and (b, nut) in recon and len(recon[(a, nut)]) == 1 and len(recon[(b, nut)]) == 1:
                entry["perm_p(유일 재구성)"] = round(perm_test(recon[(a, nut)][0], recon[(b, nut)][0]), 4)
            out["nut_tests"][f"{nut}: {a} vs {b}"] = entry

    # 4개 너트를 합친 비교 (조건당 40회): 가능한 재구성 조합을 모두 돌려 p값의 범위를 보고한다
    nuts = ("square", "triangular", "hexagonal", "cross")
    a, b = "tactile+hist", "hist"
    p_perm, p_mw, means_b = [], [], []
    for combo_a in itertools.product(*[recon[(a, n)] for n in nuts]):
        xa = [v for sol in combo_a for v in sol]
        for combo_b in itertools.product(*[recon[(b, n)] for n in nuts]):
            xb = [v for sol in combo_b for v in sol]
            p_perm.append(perm_test(xa, xb, iters=20_000))
            p_mw.append(float(stats.mannwhitneyu(xa, xb).pvalue))
            means_b.append(float(np.mean(xb)))
    # 실패 꼬리: 시행별 progress < 75% 인 횟수 (Fisher exact)
    tail_a, tail_b, fisher_p = [], [], []
    for combo_a in itertools.product(*[recon[(a, n)] for n in nuts]):
        ka = int(sum(v < 75 for sol in combo_a for v in sol))
        for combo_b in itertools.product(*[recon[(b, n)] for n in nuts]):
            kb = int(sum(v < 75 for sol in combo_b for v in sol))
            tail_a.append(ka); tail_b.append(kb)
            fisher_p.append(float(stats.fisher_exact([[ka, 40 - ka], [kb, 40 - kb]]).pvalue))
    out["nut_tests"]["4종 합산: 진행률 75% 미만 시행 수 (촉각+히스토리 vs 히스토리)"] = {
        "tactile+hist": sorted(set(tail_a)), "hist": sorted(set(tail_b)),
        "fisher_p_range": [round(min(fisher_p), 4), round(max(fisher_p), 4)]}
    # 검정력: 표준편차 σ, 효과 δ 일 때 80% 검정력(α=0.05, 양측)에 필요한 조건당 시행 수 (정규근사)
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
    out["power_n_per_group"] = {f"sigma={sg}, delta={dl}": int(np.ceil(2 * (z * sg / dl) ** 2))
                                for sg, dl in ((20, 10), (15, 10), (20, 7.5), (13, 7.5))}

    out["nut_tests"][f"4종 합산 (n=40): {a} vs {b}"] = {
        "mean_a": round(float(np.mean(xa)), 2), "mean_b": round(means_b[0], 2),
        "n_reconstruction_combos": len(p_perm),
        "perm_p_range": [round(min(p_perm), 4), round(max(p_perm), 4)],
        "mannwhitney_p_range": [round(min(p_mw), 4), round(max(p_mw), 4)]}

    for a, b in (("tactile+hist", "hist"), ("hist", "none"), ("tactile", "none"), ("none", "expert_replay")):
        t, p = welch(SCREW[a], SCREW[b])
        out["screw_tests"][f"{a} vs {b}"] = {"welch_t": round(t, 3), "welch_p": round(p, 4)}

    (ROOT / "analysis" / "stats_check_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
