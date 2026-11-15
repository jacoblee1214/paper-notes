"""토이 1: 벤치마크 순위가 시도 수 잡음과 과제 부분집합 재표본에서 얼마나 뒤집히는가 (numpy, 단일 프로세스, 약 1분).

모형 (우리가 만든 것, 실제 LIBERO 정책이 아님):
- 과제 T=10개, 과제당 시도 S회 (LIBERO 한 스위트의 공식 설정 10 x 50).
- 정책 A의 과제별 성공률 p_At ~ Beta(평균 0.90, 집중도 10): 과제 간 이질성.
- 정책 B는 p_Bt = p_At + δ (δ는 평균 격차, 과제 전체에 같은 크기로 더함).
- 세 가지 평가 설계:
  (1) 독립: 두 정책이 서로 다른 초기 상태에서 평가 (비짝지음)
  (2) 짝지음: 두 정책이 같은 초기 상태 난수를 공유 (common random numbers)
  (3) 독립 + 과제 부분집합: 10개 과제 중 5개만 뽑아 평균 (과제 재표본)
- 순위 뒤집힘 확률 = P(관측 평균 차이의 부호가 참 부호와 다름), 동점은 절반으로 셈.
- 과제 부분집합 뒤집힘 = P(부분집합 순위 ≠ 전체 과제 순위). 참값과 무관하게 벤치마크 자체가 얼마나 흔들리는지.

실행: python3 -I analysis/toy_rank_flip.py  → analysis/toy_rank_flip_results.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "toy_rank_flip_results.json"

T = 10          # 과제 수 (LIBERO 한 스위트)
MEAN_A = 0.90   # 정책 A의 평균 성공률
KAPPA = 10.0    # Beta 집중도 (과제 간 이질성)
DELTAS = [0.0, 0.008, 0.02, 0.03, 0.05, 0.10]
S_LIST = [50, 100, 500]
REPS = 4000
SUBSET_K = 5
PHI = 0.2        # 짝지어도 초기 상태별로 결과가 갈리는 시도 비율 (가정, 측정값 아님)
rng = np.random.default_rng(20260615)


def flip_prob(d_obs, true_sign):
    """참 부호가 양수일 때 P(D<0) + 0.5 P(D=0); 참 부호 0이면 P(D>0)+0.5P(D=0)을 '뒤집힘'으로 본다."""
    if true_sign > 0:
        return float(np.mean(d_obs < 0) + 0.5 * np.mean(d_obs == 0))
    if true_sign < 0:
        return float(np.mean(d_obs > 0) + 0.5 * np.mean(d_obs == 0))
    return float(np.mean(d_obs != 0))  # δ=0: 어떤 0이 아닌 차이든 '잘못된 순위'


results = {"config": {"T": T, "mean_A": MEAN_A, "kappa": KAPPA, "deltas": DELTAS, "S_list": S_LIST,
                      "reps": REPS, "subset_k": SUBSET_K, "phi_discordance_assumed": PHI, "seed": 20260615},
           "grid": []}

for S in S_LIST:
    for delta in DELTAS:
        a = MEAN_A * KAPPA
        b = (1 - MEAN_A) * KAPPA
        pA = rng.beta(a, b, size=(REPS, T))
        pB = np.clip(pA + delta, 0.001, 0.999)
        pA = np.clip(pA, 0.001, 0.999)
        # 참 격차 (과제 평균의 기댓값 차이, 같은 표본에서 계산)
        true_gap = float(np.mean(pB - pA))
        true_sign = int(np.sign(true_gap)) if delta > 0 else 0

        # (1) 독립 평가
        xa = rng.binomial(S, pA)
        xb = rng.binomial(S, pB)
        d_ind = np.mean(xb - xa, axis=1) / S
        p_ind = flip_prob(d_ind, true_sign)

        # (2a) 짝지음 상한: 같은 초기 상태 난수 u를 공유하고 단조 결합 (불일치는 격차만큼만 생김, 낙관적 상한)
        u = rng.random((REPS, T, S), dtype=np.float32)
        sa = (u < pA[..., None]).sum(axis=2)
        sb = (u < pB[..., None]).sum(axis=2)
        d_pair_bound = np.mean(sb - sa, axis=1) / S
        p_pair_bound = flip_prob(d_pair_bound, true_sign)
        # (2b) 짝지음 + 불일치 비율 PHI: 시도의 PHI 비율에서는 B가 독립 난수로 성공/실패 (주변 확률은 그대로)
        flag = rng.random((REPS, T, S)) < PHI
        indep_b = rng.random((REPS, T, S), dtype=np.float32) < pB[..., None]
        xb_disc = np.where(flag, indep_b, u < pB[..., None])
        sb2 = xb_disc.sum(axis=2)
        d_pair = np.mean(sb2 - sa, axis=1) / S
        p_pair = flip_prob(d_pair, true_sign)
        disc_rate = float(np.mean(xb_disc != (u < pA[..., None])))  # 시도 단위 불일치 비율 (A와 B 결과가 다른 비율)
        del u, flag, indep_b, xb_disc

        # (3) 독립 평가 + 과제 부분집합 (5/10)
        masks = np.zeros((REPS, T), dtype=bool)
        idx = np.argsort(rng.random((REPS, T)), axis=1)[:, :SUBSET_K]
        np.put_along_axis(masks, idx, True, axis=1)
        d_sub = np.sum((xb - xa) * masks, axis=1) / (S * SUBSET_K)
        p_sub = flip_prob(d_sub, true_sign)
        # 부분집합 순위 ≠ 전체 과제 순위 (참 부호 기준이 아니라 같은 표본의 전체 평균 기준)
        p_sub_vs_full = float(np.mean(np.sign(d_sub) != np.sign(d_ind)))

        results["grid"].append({
            "S": S, "delta_pp": round(100 * delta, 1), "true_gap_pp": round(100 * true_gap, 2),
            "flip_independent": round(p_ind, 4),
            "flip_paired_discordance_0.2": round(p_pair, 4),
            "flip_paired_monotone_upper_bound": round(p_pair_bound, 4),
            "trial_discordance_rate_phi0.2": round(disc_rate, 4),
            "flip_task_subset_5of10": round(p_sub, 4),
            "subset_vs_full_sign_disagreement": round(p_sub_vs_full, 4),
        })
        print(f"S={S:4d} δ={100*delta:5.1f}pp  indep={p_ind:.3f}  paired_phi={p_pair:.3f}  paired_bound={p_pair_bound:.3f}  "
              f"subset={p_sub:.3f}  subset_vs_full={p_sub_vs_full:.3f}")

# 가장 작은 격차에서 80% 순위 정확도를 얻는 데 필요한 격차 (격자 보간 없이 가장 가까운 격자값)
summary = {}
for S in S_LIST:
    rows = [r for r in results["grid"] if r["S"] == S and r["delta_pp"] > 0]
    for key in ["flip_independent", "flip_paired_discordance_0.2", "flip_task_subset_5of10"]:
        ok = [r["delta_pp"] for r in rows if r[key] <= 0.2]
        summary[f"S{S}_{key}_min_delta_pp_for_80pct_correct_on_grid"] = min(ok) if ok else None
results["summary"] = summary
OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", OUT)
print(json.dumps(summary, indent=1))
