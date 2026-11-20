"""토이: 과제 내 정책 순서 일치(논문 표 3)가 시행 수 n=20의 표본 잡음만으로 얼마나 기대되는지.
실행: python3 analysis/toy_rank_noise.py -> analysis/toy_rank_noise_results.json (numpy, 약 수 초)
가정: 표 3의 실로봇 성공률을 참값 p로 둔다. 각 정책은 n=20 이항 시행으로 측정한다(논문이 n을 명시하지 않아 0.05 단위에서 추정).
세계모델 측정은 참값에서 평균 편차 mean_gap, 셀별 편차 sd_gap을 가진 정규 편차를 더한 뒤 n=20 이항으로 잰다.
"""
import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(20260301)
OUT = Path(__file__).resolve().parent / "toy_rank_noise_results.json"
N = 20
REPS = 20000

# 표 3에서 과제별 (real success) 3정책, (WM success) 3정책 -- recompute.py와 같은 값
real = np.array([
    [0.70, 0.85, 0.85], [0.40, 0.55, 0.75], [0.05, 0.15, 0.80], [0.20, 0.50, 0.80],
    [0.15, 0.50, 0.70], [0.30, 0.10, 0.75], [0.60, 0.70, 0.65]])
wm = np.array([
    [0.55, 0.70, 0.70], [0.20, 0.40, 0.65], [0.00, 0.05, 0.30], [0.00, 0.35, 0.65],
    [0.00, 0.00, 0.05], [0.20, 0.00, 0.50], [0.40, 0.55, 0.55]])

pairs = []
for t in range(real.shape[0]):
    for a in range(3):
        for b in range(a + 1, 3):
            if real[t, a] != real[t, b] and wm[t, a] != wm[t, b]:
                pairs.append((t, a, b))
# 실로봇 쌍 중 값이 같은 쌍은 순서가 정의되지 않으므로 제외 (recompute.py: 성공 기준 비동점 18쌍과 같아야 함)

def sign_match_rate(p_true_a, p_true_b, sims):
    """두 정책을 n=20 이항으로 각각 재고, 측정 순서가 참 순서와 같을 확률."""
    ka = rng.binomial(N, np.clip(p_true_a, 0, 1), size=sims)
    kb = rng.binomial(N, np.clip(p_true_b, 0, 1), size=sims)
    true_sign = np.sign(p_true_a - p_true_b)
    meas_sign = np.sign(ka - kb)
    decided = meas_sign != 0
    correct = (meas_sign == true_sign) & decided
    return correct.mean(), (~decided).mean()

res = {"n_pairs_nontied": len(pairs), "N_trials_assumed": N, "reps": REPS}

# (A) 실로봇 참값 기준: 쌍마다 n=20 측정 한 번으로 순서를 맞힐 확률 (잡음만)
pc = []; ties = []
for (t, a, b) in pairs:
    c, tie = sign_match_rate(real[t, a], real[t, b], REPS)
    pc.append(c); ties.append(tie)
res["A_noise_only_pair_correct_prob"] = {
    "per_pair": [round(float(x), 3) for x in pc],
    "mean": round(float(np.mean(pc)), 3),
    "expected_correct_out_of_pairs": round(float(np.sum(pc)), 2),
    "prob_all_pairs_correct_if_independent": round(float(np.prod(pc)), 4),
    "note": "독립 쌍 근사. 실제 쌍은 같은 과제 안에서 상관되어 있어 근사이다",
}

# (B) 세계모델 편차 모형: 실로봇 참값 + 평균 편차(-0.20) + 셀별 편차 sd 0.154 (표 3에서 추정)
gap_mean = -0.202; gap_sd = 0.154
agree_frac = []
all_ok = []
for rep_sims in range(1):
    pass
hits = np.zeros(REPS)
per_pair_b = []
for (t, a, b) in pairs:
    pa = np.clip(real[t, a] + gap_mean + gap_sd * rng.standard_normal(REPS), 0, 1)
    pb = np.clip(real[t, b] + gap_mean + gap_sd * rng.standard_normal(REPS), 0, 1)
    ka = rng.binomial(N, pa); kb = rng.binomial(N, pb)
    ms = np.sign(ka - kb); ts = np.sign(real[t, a] - real[t, b])
    ok = (ms == ts) & (ms != 0)
    per_pair_b.append(ok.mean())
    hits += ok
res["B_wm_model_pair_correct_prob"] = {
    "gap_mean": gap_mean, "gap_sd": gap_sd,
    "per_pair": [round(float(x), 3) for x in per_pair_b],
    "expected_correct_out_of_pairs": round(float(np.sum(per_pair_b)), 2),
    "note": "논문 관측 18쌍 모두 일치와 비교. 셀별 편차를 과제 간에 독립으로 두었다",
}

# 관측: recompute.py 결과 success_agree = 18 (비동점 18쌍 중 불일치 0)
res["observed_success_agree_nontied"] = {"agree": 18, "disagree": 0}
res["reading"] = (
    "A (참값=표 3 실로봇 성공률, 정책당 n=20 한 번): 18쌍 중 기대 16.1쌍이 맞고, 18쌍 모두 맞을 확률은 독립 근사에서 약 0.11이다. "
    "즉 20회 측정은 순서를 대체로 맞히지만 완벽하지 않다. "
    "B (세계모델 편차 평균 -0.20, 셀별 sd 0.154): 기대 14.0쌍이 맞고, 18쌍 모두 맞을 확률은 독립 근사에서 약 0.005이다. "
    "관측 18/18은 B 모형보다 높다. 평균 편차는 순서를 바꾸지 않으므로 셀별 sd가 순서 일치를 결정한다고 봐야 한다. "
    "쌍은 과제 안에서 상관되어 있고 짝 설계 여부는 논문이 적지 않았으므로, 이 확률은 근사값으로만 읽는다."
)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
print(json.dumps(res, ensure_ascii=False, indent=2))
