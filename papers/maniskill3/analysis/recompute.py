"""ManiSkill3 (arXiv:2410.00425 v2) 보고 수치 재계산. 학습·재현 없음, 표 값과 본문 값만 사용.

실행: python3 analysis/recompute.py  -> analysis/recompute_results.json
입력은 논문 본문과 표(Table II–IV, 본문 IV-D, III-B, 부록 XI-C)에 적힌 값뿐입니다.
"""
import json
import math
from pathlib import Path

from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def two_prop_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


res = {}

# 1) 본문 III-E / IV-D: 실물 cube picking, 3 runs x 8 평가 = 24회, 22 성공
k, n = 22, 24
lo, hi = wilson(k, n)
res["sim2real_cube"] = {"k": k, "n": n, "rate_exact": round(k / n, 4),
                        "reported": 0.916, "wilson95": [round(lo, 3), round(hi, 3)],
                        "note": "보고 91.6%는 22/24=91.67%의 절삭 표기. 실험 간 분산은 원자료 없음"}

# 2) Table IV: PerAct, 'best success over 80k steps', 평가 100 episodes (본문 IV-B)
#    최고 체크포인트 선택(best-of-training)이라 독립 평가치가 아님. 아래 CI는 상한 참고용.
perac = {
    "PushCube": {"no_aug_no_mv": 0.96, "aug_no_mv": 0.98, "aug_mv": 1.00},
    "StackCube": {"no_aug_no_mv": 0.38, "aug_no_mv": 0.47, "aug_mv": 0.55},
}
n_ep = 100
peract = {}
for task, d in perac.items():
    peract[task] = {}
    for key, p in d.items():
        k = round(p * n_ep)
        lo, hi = wilson(k, n_ep)
        peract[task][key] = {"k": k, "n": n_ep, "wilson95": [round(lo, 3), round(hi, 3)]}
z, pv = two_prop_z(38, 100, 55, 100)
res["peract"] = {"assumed_n_per_setting": n_ep, "rows": peract,
                 "stack_no_aug_vs_aug_mv": {"diff": 0.17, "z": round(z, 2), "p_two_sided": round(pv, 4)},
                 "push_no_aug_vs_aug_mv": None}
z2, p2 = two_prop_z(96, 100, 100, 100)
res["peract"]["push_no_aug_vs_aug_mv"] = {"diff": 0.04, "z": round(z2, 2), "p_two_sided": round(p2, 4)}

# 3) Table II / III: LfD 성공률. 평가 횟수 미보고. n=100을 가정한 '가능한 최대 불확실성' 예시만 계산
lfd = {
    "state": {"BC": [0.03, 0.81, 0.00, 0.00], "ACT": [1.00, 0.97, 0.97, 0.43], "DP": [1.00, 0.96, 0.99, 0.66]},
    "rgb": {"BC": [0.03, 0.81, 0.00, 0.00], "ACT": [0.98, 0.89, 0.80, 0.00], "DP": [1.00, 0.86, 0.81, 0.00]},
}
dup = [i for i, (a, b) in enumerate(zip(lfd["state"]["BC"], lfd["rgb"]["BC"])) if a == b]
res["lfd_1000demo_bc_identical_positions"] = dup  # 0..3 = PickCube, PushCube, StackCube, PegInsertion
res["lfd_note"] = "Table II(state)와 III(RGB)의 BC 1000 demos 행이 네 과제 모두 같음(0.03, 0.81, 0.00, 0.00). 복사 오류 가능성, 저자 확인 필요"
ex = {}
for task_i, task in enumerate(["PickCube", "PushCube", "StackCube", "PegInsertion"]):
    p = lfd["rgb"]["DP"][task_i]
    ex[task] = [round(x, 3) for x in wilson(round(p * 100), 100)]
res["lfd_rgb_dp_1000_wilson_if_n100"] = ex

# 4) 메모리 비율 (본문·부록 값 간 일관성)
res["memory_ratios"] = {
    "fig2_128env_isaac_over_ms3": round(14.1 / 3.5, 2),
    "appendix_min_camera_isaac_over_ms3": round(4.8 / 1.7, 2),
    "abstract_claim_range": "2-3x",
    "fig31_32_text_claim": 24,
    "note": "본문 III-B 3.5 대 14.1 GB는 약 4.0배, 부록 1.7 대 4.8 GB는 약 2.8배. 초록 2-3배와 맞고, 부록 XI-C의 '24x'는 위 두 비율과 맞지 않음",
}

# 5) 실로봇-시뮬 상관 0.9284 (real2sim, 4 과제). 점 개수가 명시되지 않아 n 가정별 p-value
r = 0.9284
pvals = {}
for n_pts in (4, 8):
    t = r * math.sqrt(n_pts - 2) / math.sqrt(1 - r * r)
    pvals[str(n_pts)] = round(2 * (1 - stats.t.cdf(abs(t), n_pts - 2)), 5)
res["real2sim_corr"] = {"r": r, "p_two_sided_by_n_points": pvals,
                        "note": "점 개수(과제 4 x 정책 2 = 8 가정 등) 미보고. 정책 간 평균이면 n=4가 맞음"}

# 6) 속도 주장: 15x (PPO state, GPU ~1 min vs ManiSkill2 ~15 min), 8x (RGB ~10 min)
res["speedup_claims"] = {"state_claim_x": 15, "state_ms3_min": 1, "state_implied_ms2_min": 15,
                         "rgb_claim_x": 8, "rgb_ms3_min": 10, "rgb_implied_ms2_min": 80,
                         "note": "그림 11에서 읽은 값이 아니라 본문 서술값. 그림 재판독은 하지 않음. v1은 '약 5x'(v1 결과 절)"}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
