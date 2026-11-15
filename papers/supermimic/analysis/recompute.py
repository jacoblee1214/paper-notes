"""SuperMimic 논문 수치 재계산과 일관성 검사 (numpy, scipy만 사용, 수 초).

입력: 논문 본문·부록 표에서 옮긴 숫자 (각 상수 옆에 출처 절·표 번호를 적었다).
출력: analysis/recompute_results.json  (실행: python3 analysis/recompute.py)

검사 항목
  1. 컴퓨트 비율 (§4.1 Compute accounting): wall-clock, GPU-hours, H100/L40 정규화 하한
  2. 데이터 비율 (§4.1): 611 h / 264.56 h
  3. 실패 수 (Table A.8, Figure 3a): 비율, Wilson·Clopper-Pearson 구간, Fisher 검정
     쌍체(같은 129,514 모션) 정보가 없으므로 McNemar 정확 검정의 가능한 범위를 o(두 정책 모두 실패한 수)에 대해 계산
  4. 표 합계 검사: Table A.7, A.8 (합계, 그룹 제약 "어떤 그룹에서도 SONIC보다 많이 실패하지 않음")
  5. 단계 검사 (Figure 7, §4.5, G.2): 672 = 3·(15·15−1), 650 + 15 + 7 = 672
  6. 학습 설정 (Table A.5, §3.2): 업데이트 대 데이터 비율 0.4가 코드 설정에서 나오는지
  7. 종료 기준 (Table A.2, A.6): 기울기 임계각
"""
import json
import math
from pathlib import Path

from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}

# ---- 1. 컴퓨트 (§4.1, 본문 "Compute accounting") -------------------------
SONIC_GPUS, SONIC_DAYS = 128, 7               # 본문: 128 L40 GPUs, 7 days
SONIC_H = SONIC_DAYS * 24                     # 168 h
SONIC_GPUH = SONIC_GPUS * SONIC_H             # 21,504 (본문 "about 21,000")
SM_1GPU_H, SM_2GPU_H = 3.0, 9.8               # 본문 Fig 1, Table A.8 (2 GPUs × 9.8 h)
res["compute"] = {
    "sonic_gpu_hours": SONIC_GPUH,
    "wallclock_ratio_1gpu": SONIC_H / SM_1GPU_H,            # 56 (>50× 주장과 일치)
    "wallclock_ratio_2gpu": SONIC_H / SM_2GPU_H,            # 17.1 (2 GPU는 벽시계 시간이 길어짐)
    "gpuhour_ratio_1gpu": SONIC_GPUH / SM_1GPU_H,           # 7168 (본문 "about 7200×")
    "gpuhour_ratio_2gpu": SONIC_GPUH / (2 * SM_2GPU_H),     # 1097 (2 GPU를 GPU-h로 계산해도 1000× 이상)
}
# H100 대 L40 정규화 하한: 본문 §4.1은 H100이 L40보다 tensor-core 5.5배, fp32 0.74배라고 함.
# 가장 유리한 가정(H100 1대가 L40 5.5대와 같다)으로 SuperMimic 3 h를 L40-h로 바꾸면:
res["compute"]["gpuhour_ratio_worst_case_h100_5p5x"] = SONIC_GPUH / (SM_1GPU_H * 5.5)   # 약 1303
res["compute"]["gpuhour_ratio_fp32_h100_0p74x"] = SONIC_GPUH / (SM_1GPU_H / 0.74)       # 약 5300
res["compute"]["sonic_gpuh_from_paper_21000_check"] = 21000 / SONIC_GPUH                   # 0.977

# ---- 2. 데이터 (§4.1) ---------------------------------------------------
res["data_ratio_611_over_264p56"] = 611 / 264.56           # 2.31 (본문 "about 2.3×")

# ---- 3. 실패 수 (Table A.8, Fig 3a) -----------------------------------
N = 129514
SONIC_F, SM1_F, SM2_F = 179, 25, 0
def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h
def cp(k, n):
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    return ci.low, ci.high
fail = {}
for name, k in [("sonic_released", SONIC_F), ("supermimic_1gpu_3h", SM1_F), ("supermimic_2gpu_9p8h", SM2_F)]:
    fail[name] = {
        "failures": k,
        "rate_pct": 100 * k / N,
        "wilson95_pct": [100 * x for x in wilson(k, N)],
        "clopper_pearson95_pct": [100 * x for x in cp(k, N)],
    }
fail["upper95_if_zero_rule_of_three_pct"] = 100 * 3 / N
res["failures"] = fail
res["failure_ratio_sonic_over_sm1"] = SONIC_F / SM1_F      # 7.16
# 독립 표본 가정의 Fisher 검정 (모션을 독립 추출로 취급; 벤치마크가 고정이므로 모집단 구간이 아님)
ft = stats.fisher_exact([[SM1_F, N - SM1_F], [SONIC_F, N - SONIC_F]])
res["fisher_independent_p"] = ft.pvalue
# 쌍체 McNemar 정확 검정 범위.  o = 두 정책이 모두 실패한 모션 수 (0..min(25,179)).
#   b = SONIC 실패 & SM 성공 = 179 - o,  c = SM 실패 & SONIC 성공 = 25 - o
# 그룹 제약: 그룹별로 SM 실패 <= SONIC 실패 (Table A.8 "In no group ...")는 o의 상한을 더 줄인다.
pvals = []
for o in range(0, SM1_F + 1):
    b, c = SONIC_F - o, SM1_F - o
    n_d = b + c
    if n_d == 0:
        continue
    pvals.append((o, stats.binomtest(c, n_d, 0.5).pvalue))
res["mcnemar_exact_p_range_over_overlap"] = {
    "max_p": max(p for _, p in pvals), "min_p": min(p for _, p in pvals),
    "note": "o 전 범위에서 p가 모두 매우 작음. 쌍체 정보 없이도 차이는 유의하다고 볼 수 있으나, 모션별 자료가 있어야 정확한 값을 준다.",
}

# ---- 4. 표 합계 (Table A.7, A.8) -------------------------------------------
TBL_A7 = {"raw": 142220, "after_name": 129667, "after_feas": 129514}
res["tableA7"] = {
    "name_filter_removed": TBL_A7["raw"] - TBL_A7["after_name"],           # 12,553 (본문 일치)
    "feas_filter_removed": TBL_A7["after_name"] - TBL_A7["after_feas"],    # 153 (본문 일치)
}
groups = [  # (이름, 모션 수, SONIC, SM 1GPU 3h, SM 2GPU 9.8h)  — Table A.8
    ("side rolls", 22, 22, 2, 0),
    ("flips 360/180/90", 44, 44, 1, 0),
    ("fire-in-the-hole", 20, 3, 3, 0),
    ("crossed legs", 10, 9, 0, 0),
    ("lying/rest/stand", 99, 28, 8, 0),
    ("broad jumps", 24, 15, 0, 0),
    ("dance routine", 8, 8, 0, 0),
    ("all other", 129287, 50, 11, 0),
]
tot_m = sum(g[1] for g in groups)
tot_s = sum(g[2] for g in groups)
tot_1 = sum(g[3] for g in groups)
tot_2 = sum(g[4] for g in groups)
res["tableA8"] = {
    "motions_sum": tot_m, "motions_expected": N,
    "sonic_sum": tot_s, "sm1_sum": tot_1, "sm2_sum": tot_2,
    "groups_where_sm1_exceeds_sonic": [g[0] for g in groups if g[3] > g[2]],
    "first_five_groups_motions": sum(g[1] for g in groups[:5]),          # 195
    "first_five_groups_sm1_failures": sum(g[3] for g in groups[:5]),    # 14
    "share_of_sm1_failures_first_five_pct": 100 * sum(g[3] for g in groups[:5]) / tot_1,  # 56%
}

# ---- 5. 단계 검사 (Figure 7, §4.5, G.2) ------------------------------------
step_total = 3 * (15 * 15 - 1)                 # 672 (세 평면, 원점 제외)
rl_better, dist_better, dist_fell = 650, 15, 7
res["step_test"] = {
    "total_expected_672": step_total,
    "rl_better": rl_better, "distill_better": dist_better, "distill_fell_commands": dist_fell,
    "sum_check": rl_better + dist_better + dist_fell,      # 672 -> 7은 "넘어짐"으로 따로 세는 것으로 보임
    "sign_test_p_two_sided_650_of_665": stats.binomtest(650, 650 + 15, 0.5).pvalue,
    "caveat": "672개 명령은 한 정책 쌍의 격자 응답이라 독립이 아니고, 학습 시드가 1개이므로 검정은 기술적 요약으로만 쓴다.",
}

# ---- 6. 학습 설정 일관성 (Table A.5, §3.2, 코드 wm_learner/policy_learner 기본값) --
num_envs, wm_nb, wm_bs, wm_h, replay_ratio = 4096, 10, 2048, 8, 16
res["update_to_data_ratio_from_code"] = num_envs / wm_nb / wm_bs / wm_h * replay_ratio   # 0.4 (Table A.5)
res["policy_horizon_check"] = {"H": 32, "imagined_batch": 4096, "wm_k": 8}

# ---- 7. 종료 기준 (Table A.2 본문, A.6 본문) -------------------------------
res["tilt_termination_deg"] = {
    "anchor_etilt_0p8_deg": math.degrees(math.acos(1 - 0.8)),   # 78.5 (본문 "about 78°")
    "velocity_1plus_uz_0p7_deg": math.degrees(math.acos(0.3)),  # 72.5 (본문 "about 73°")
}

# ---- 8. 미분(BPTT) 샘플 수와 시드 -----------------------------------------
res["seed_count_reported"] = {"figures": "단일 시드(Fig 3, 4의 값)", "step_test": 1, "text": "several random seeds (§4)"}

OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=2, ensure_ascii=False))
