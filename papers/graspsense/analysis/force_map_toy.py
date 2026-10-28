"""힘 지도 재순위화(Eq. 5–6)와 조회 방식(Eq. 4 vs 최근접 꼭짓점)의 toy 검증 (CPU, numpy). 실행: python3 analysis/force_map_toy.py

공식 코드에 힘 지도와 재순위화가 없으므로(force_map_builder.py는 placeholder) 논문 Sec. III-C·D의 설명대로 만든 toy입니다.
F_base의 함수 꼴, k_m, F_clamp, 가산점 폭, α는 논문에 없어서 가정했습니다.

- T1: α = 0이면 S_F = min_k F(t_k)이고, F가 두께에 대해 단조 증가하면 선택은 "접촉점 최소 두께가 가장 큰 후보"와 같다.
      재료 계수 k_m과 지수 n을 바꿔도 선택이 바뀌지 않는지, α > 0이면 단위(스케일)에 따라 바뀌는지
- T2: 계획 단계(무게중심 좌표 보간, Eq. 4)와 실행 단계(최근접 꼭짓점 조회, Sec. III-E)의 허용 하중 차이가
      테두리 전이 구간에서 메쉬 해상도에 따라 얼마나 커지는지
- T3: 허용 하중을 σ t² (얇은 판 굽힘의 차원 분석)로 보면 Table I이 어떤 두께에 해당하는지
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)

L, A = 20, 36                      # 높이 층, 각 구간 (공식 코드 ForceMapBuilder 기본값)
M_TOP = 5                          # 실험 1: 상위 M = 5 후보 (Sec. IV-A)
N_FING = 5


def cup_thickness(L=L, A=A, t_wall=0.35e-3, t_rim=1.0e-3, t_base=1.5e-3, noise=0.04):
    """종이컵 비슷한 두께장 [m]: 아래 2층은 바닥, 위 2층은 말린 테두리, 나머지는 벽."""
    t = np.full((L, A), t_wall)
    t[:2] = t_base
    t[-2:] = t_rim
    return t * (1 + noise * rng.standard_normal((L, A)))


def force_map(t, k_m=1.0, n=2.0, bonus=True, sigma_layers=1.5, bonus_frac=0.3, f_clamp=None):
    """논문 Sec. III-C 4): F = min((F_base + Σ_s b_s) · k_m, F_clamp). F_base = t^n (꼴은 가정, t는 mm)."""
    f = (t / 1e-3) ** n
    if bonus:
        prof = f.mean(axis=1)
        peaks = [l for l in range(1, L - 1) if prof[l] > prof[l - 1] and prof[l] >= prof[l + 1]]
        peaks += [l for l in (0, L - 1) if prof[l] == prof.max()]
        ls = np.arange(L)
        f = f + sum(bonus_frac * prof[s] * np.exp(-0.5 * ((ls - s) / sigma_layers) ** 2) for s in peaks)[:, None]
    f = f * k_m
    if f_clamp is not None:
        f = np.minimum(f, f_clamp)
    return f


def sample_candidates(k=50):
    """후보마다 손가락 5개 접촉 셀 (l, a). 엄지 반대편 배치를 흉내 내어 같은 높이 근처에 모음."""
    cands = []
    for _ in range(k):
        l0 = rng.integers(0, L)
        a0 = rng.integers(0, A)
        ls = np.clip(l0 + rng.integers(-3, 4, N_FING), 0, L - 1)
        as_ = (a0 + np.array([0, 3, 6, 9, A // 2]) + rng.integers(-1, 2, N_FING)) % A
        cands.append((ls, as_))
    return cands


def score(fmap, cand, alpha):
    vals = fmap[cand[0], cand[1]]
    return vals.min() - alpha * vals.var()


def t1(trials=400):
    """T1: (a) α = 0이면 재료 계수 k_m을 바꿔도 top-1이 같은가, (b) 그 선택이 '접촉 최소 두께 최대'와 같은가,
    (c) α > 0이면 k_m만 바꿔도 선택이 바뀌는가. 후보는 50개 중 무작위 M = 5개."""
    kms = (0.35, 1.0, 10.0)
    alphas = (0.0, 0.01, 0.1, 1.0)
    res = {}
    for bonus_key, bonus in (("no_bonus", False), ("with_bonus", True)):
        for n in (1.0, 2.0, 3.0):
            same_across_km = {a: 0 for a in alphas}
            agree_thick = {a: 0 for a in alphas}
            for _ in range(trials):
                t = cup_thickness()
                cands = sample_candidates()
                top = [cands[i] for i in rng.choice(len(cands), M_TOP, replace=False)]
                ref = int(np.argmax([t[c[0], c[1]].min() for c in top]))
                for a in alphas:
                    picks = [int(np.argmax([score(force_map(t, k_m=k, n=n, bonus=bonus), c, a) for c in top]))
                             for k in kms]
                    same_across_km[a] += len(set(picks)) == 1
                    agree_thick[a] += picks[1] == ref
            res[f"{bonus_key}_n{n:g}"] = {
                "same_choice_across_k_m": {f"alpha{a:g}": round(v / trials, 3) for a, v in same_across_km.items()},
                "agree_with_max_min_thickness_k_m1": {f"alpha{a:g}": round(v / trials, 3) for a, v in agree_thick.items()},
            }
    return res


def t2():
    """T2: 높이 방향 1D. 벽 4.8 N → 테두리 41.2 N 전이가 폭 w(3 mm) 안에서 선형으로 일어남(Table I 종이컵 값).
    메쉬 꼭짓점 간격 h마다, 전이 구간 ±5 mm의 접촉점에서 선형 보간 값과 최근접 꼭짓점 값의 차이."""
    f_wall, f_rim, w = 4.8, 41.2, 3e-3
    z_tr = 0.0

    def f_true(z):
        return np.interp(z, [z_tr - w / 2, z_tr + w / 2], [f_wall, f_rim])

    out = {}
    zs = rng.uniform(-5e-3, 5e-3, 20000)
    for h in (0.5e-3, 1e-3, 2e-3, 5e-3, 10e-3):
        offset = rng.uniform(0, h)
        verts = np.arange(-30e-3 + offset, 30e-3, h)
        fv = f_true(verts)
        f_bary = np.interp(zs, verts, fv)                     # Eq. 4 (1D에서는 선형 보간)
        f_nn = fv[np.abs(zs[:, None] - verts[None, :]).argmin(axis=1)]
        rel = np.abs(f_nn - f_bary) / np.minimum(f_nn, f_bary)
        over = f_nn > 1.5 * f_bary                            # 실행 단계가 50% 넘게 관대하게 보는 경우
        out[f"spacing_{h * 1e3:g}mm"] = {
            "median_rel_diff": round(float(np.median(rel)), 3),
            "p95_rel_diff": round(float(np.quantile(rel, 0.95)), 3),
            "share_nn_over_1.5x_bary": round(float(over.mean()), 3),
        }
    return out


def t3():
    """T3: F ≈ C σ t² (얇은 판 점하중, C ~ 1)에서 Table I이 함의하는 두께. σ는 사전 지식의 대략값(확인 필요)."""
    sigma = {"paper cup": (30e6, 50e6), "plastic cup": (25e6, 35e6), "glass goblet": (30e6, 50e6)}
    table1 = {
        "paper cup": {"rim": 41.2, "wall": 4.8, "base": 88.8},
        "plastic cup": {"rim": 3.7, "wall": 0.3, "base": 8.0},
        "glass goblet": {"bowl": 69.0, "stem": 565.0, "base": 989.0},
    }
    out = {}
    for obj, zones in table1.items():
        lo, hi = sigma[obj]
        out[obj] = {z: [round(math.sqrt(F / hi) * 1e3, 2), round(math.sqrt(F / lo) * 1e3, 2)] for z, F in zones.items()}
        ratios = list(zones.values())
        out[obj]["zone1_over_zone2_force"] = round(ratios[0] / ratios[1], 2)
        out[obj]["implied_thickness_ratio_n2"] = round(math.sqrt(ratios[0] / ratios[1]), 2)
        out[obj]["implied_thickness_ratio_n3"] = round((ratios[0] / ratios[1]) ** (1 / 3), 2)
    out["unit"] = "mm, [sigma_hi, sigma_lo] 범위, C = 1"
    return out


def main() -> None:
    out = {"T1_ranking_invariance": t1(), "T2_lookup_mismatch": t2(), "T3_implied_thickness": t3()}
    path = ROOT / "analysis" / "force_map_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", path)


if __name__ == "__main__":
    main()
