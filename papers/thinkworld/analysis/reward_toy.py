"""세계 상상 보상(Eq. 6)과 GRPO 정규화 toy (CPU, numpy). 실행: python3 analysis/reward_toy.py

- R1: Eq. 6 motion 가중치(중앙값 이하 0.2 → 90분위에서 1, 선형)는 변화량의 '순위'만으로 정해진다.
      196 패치(14×14)에서 상위 10%·25%와 하위 50%가 받는 몫을 정확히 계산한다.
- R2: Table 2의 CoD on/off 차이(전체 패치, 상위 25% 움직임 패치)로 보상 차이를 추정한다.
- R3: 형식 오류 CoD 벌칙(그룹 최저보다 0.05 낮음)이 R2의 보상 차이와 비교해 얼마나 큰지,
      그리고 4개 후보 그룹 표준화에서 정상 후보들의 advantage 폭을 얼마나 줄이는지.
- R4: 보상에 잡음이 섞일 때 K = 4 그룹 표준화 advantage가 진짜 품질 순서를 얼마나 따르는지.
"""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P = 196  # DINOv3 ViT-H/16+ @ 224 → 14×14 patch (A.1)
K = 4    # GRPO 후보 수 (Sec. 4.1, A.6)


def motion_weights(change: np.ndarray) -> np.ndarray:
    """A.6: 중앙값 이하 0.2, 90분위에서 1까지 선형, 그 밖은 clip."""
    med, p90 = np.median(change), np.quantile(change, 0.9)
    w = 0.2 + 0.8 * (change - med) / max(p90 - med, 1e-12)
    return np.clip(w, 0.2, 1.0)


def main() -> None:
    rng = np.random.default_rng(0)
    out: dict = {}

    # R1: 가중치는 순위만의 함수 → 아무 분포로 뽑아도 같은 몫(분포 세 개로 확인)
    shares = {}
    for name, sampler in {"lognormal": lambda: rng.lognormal(0, 1, P), "uniform": lambda: rng.uniform(0, 1, P),
                          "exponential": lambda: rng.exponential(1, P)}.items():
        acc = []
        for _ in range(2000):
            c = sampler()
            w = motion_weights(c)
            order = np.argsort(-c)
            top10, top25, bot50 = order[: int(0.1 * P)], order[: int(0.25 * P)], order[P // 2:]
            wn = w / w.sum()
            acc.append([wn[top10].sum(), wn[top25].sum(), wn[bot50].sum()])
        m = np.mean(acc, axis=0)
        shares[name] = {"motion_term_top10": round(m[0], 3), "motion_term_top25": round(m[1], 3),
                        "motion_term_bottom50": round(m[2], 3)}
    s10, s25, sb50 = (shares["lognormal"][k] for k in ("motion_term_top10", "motion_term_top25", "motion_term_bottom50"))
    # Eq. 6 = ½·균등 평균 + ½·가중 평균
    out["R1_weight_shares"] = {
        "by_distribution": shares,
        "total_reward_share": {"top10": round(0.5 * 0.1 + 0.5 * s10, 3), "top25": round(0.5 * 0.25 + 0.5 * s25, 3),
                               "bottom50": round(0.5 * 0.5 + 0.5 * sb50, 3)},
        "uniform_reference": {"top10": 0.1, "top25": 0.25, "bottom50": 0.5},
    }

    # R2: Table 2에서 보상 차이 추정. 가정: Table 2의 '상위 25% 움직임 패치'는 보상과 같은 변화 순위로 고른 것,
    #     각 그룹 안에서 코사인 차이가 균일.
    gaps = {}
    for cond, d_all, d_top in (("Shuffle", 0.024, 0.071), ("Empty", 0.020, 0.079), ("CoD off", 0.018, 0.071)):
        d_rest = (d_all - 0.25 * d_top) / 0.75
        weighted = s25 * d_top + (1 - s25) * d_rest
        gaps[cond] = {"delta_rest_patches": round(d_rest, 4), "delta_motion_weighted": round(weighted, 4),
                      "reward_gap_eq6": round(0.5 * d_all + 0.5 * weighted, 4)}
    out["R2_reward_gap_from_table2"] = gaps
    gap = float(np.mean([g["reward_gap_eq6"] for g in gaps.values()]))

    # R3: 형식 오류 벌칙과 그룹 표준화
    r3 = {"malformed_penalty": 0.05, "mean_gap_CoD_vs_noCoD": round(gap, 4), "penalty_over_gap": round(0.05 / gap, 2)}
    for spread_frac in (0.25, 0.5, 1.0):
        s = spread_frac * gap  # 정상 후보 사이 진짜 보상 차이(표준편차)
        widths_clean, widths_mal = [], []
        for _ in range(20000):
            r = rng.normal(0, s, K)
            a = (r - r.mean()) / (r.std() + 1e-8)
            widths_clean.append(a.max() - a.min())
            r2 = r.copy()
            r2[-1] = r[:-1].min() - 0.05  # 후보 하나가 형식 오류
            a2 = (r2 - r2.mean()) / (r2.std() + 1e-8)
            widths_mal.append(a2[:-1].max() - a2[:-1].min())
        r3[f"valid_spread={spread_frac}x_gap"] = {
            "adv_range_valid_no_malformed": round(float(np.mean(widths_clean)), 3),
            "adv_range_valid_with_1_malformed": round(float(np.mean(widths_mal)), 3),
            "shrink_ratio": round(float(np.mean(widths_mal) / np.mean(widths_clean)), 3),
        }
    out["R3_malformed_penalty"] = r3

    # R4: 보상 잡음 대비 진짜 품질 순서의 보존
    r4 = {}
    for noise_ratio in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        corr, best_hit = [], []
        for _ in range(20000):
            q = rng.normal(0, 1, K)                      # 진짜 품질(단위: 후보 간 표준편차)
            r = q + rng.normal(0, noise_ratio, K)        # 관측 보상 = 품질 + 예측·특징 잡음
            a = (r - r.mean()) / (r.std() + 1e-8)
            corr.append(np.corrcoef(a, q)[0, 1])
            best_hit.append(np.argmax(a) == np.argmax(q))
        r4[f"noise/spread={noise_ratio}"] = {"corr_adv_quality": round(float(np.nanmean(corr)), 3),
                                             "P(top_adv_is_truly_best)": round(float(np.mean(best_hit)), 3)}
    r4["chance_P_best"] = round(1 / K, 3)
    r4["note"] = "그룹 표준화는 보상 차이의 절대 크기를 지우므로, 잡음/차이 비율만이 신호 품질을 정한다"
    out["R4_group_standardization"] = r4

    path = ROOT / "analysis" / "reward_toy_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("wrote", path)


if __name__ == "__main__":
    main()
