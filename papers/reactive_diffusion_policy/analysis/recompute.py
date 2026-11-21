"""RDP (arXiv:2503.02881 v3, RSS 2025) 표 수치 재계산과 검정.

논문에는 시행별 점수가 없고 조건별 평균과 "All"만 있다. 그래서 다음과 같이 계산한다.
- 이진 성공(클램프, 들기)은 Wilson 95% 구간과 Fisher 정확 검정을 쓴다 (시행 10회 x 조건 2 또는 3).
- 연속 또는 {0, 0.5, 1} 점수는 시행별 분산을 모른다. [0, 1] 범위 가정에서 평균 m의 분산은
  최대 m(1-m)이므로, 이 최대 분산을 넣은 Welch 검정을 쓴다. 분산이 커질수록 p가 커지므로
  이 값은 가능한 모든 [0, 1] 분포에서 p의 상한이다(보수적 p_max).
- 표 간 일관성(All = 조건 평균, 점수 가능 값, 표 VI 대 표 II)을 점검한다.

실행: python3 analysis/recompute.py  (numpy, scipy, 수 초)
출력: analysis/recompute_results.json
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}

# ---- 원자료 (논문 본문과 부록 표; 열 순서: 조건 3개 + All) ----
# 표 II Peeling: [No perturb, Perturb before contact, Perturb after contact, All]
PEEL = {
    "DP": [0.56, 0.58, 0.19, 0.44],
    "DP w. tactile img.": [0.60, 0.49, 0.16, 0.41],
    "DP w. tactile emb.": [0.48, 0.55, 0.15, 0.39],
    "RDP (GelSight)": [0.98, 0.93, 0.80, 0.90],
    "RDP (MCTac)": [1.00, 0.84, 0.79, 0.88],
    "RDP (Force)": [0.99, 0.98, 0.88, 0.95],
}
# 표 III Wiping
WIPE = {
    "DP": [0.75, 0.70, 0.25, 0.57],
    "DP w. tactile emb.": [0.60, 0.75, 0.15, 0.50],
    "RDP (GelSight)": [0.85, 0.95, 0.50, 0.77],
    "RDP (Force)": [0.95, 0.85, 0.80, 0.87],
}
# 표 IV Bimanual lifting: 소프트 컵 (클램프 %, 들기 %, 점수), 하드 컵 (클램프, 들기, 점수), All 점수
# 10회 x 2 조건 = 20회. 들기 성공 수와 점수로 가능 여부를 검사한다.
LIFT = {
    "DP": dict(soft=(0, 0, 0.00), hard=(0, 0, 0.00), all=0.00),
    "DP w. tactile emb.": dict(soft=(1, 1, 0.10), hard=(2, 1, 0.05), all=0.08),
    "RDP (GelSight + MCTac)": dict(soft=(10, 10, 0.55), hard=(9, 8, 0.40), all=0.48),
    "RDP (Force)": dict(soft=(10, 9, 0.80), hard=(9, 9, 0.60), all=0.70),
}
# 표 VI (부록 G): RDP (Force) 절제, Peeling [No, before, after, All]
ABL = {
    "DP": [0.56, 0.58, 0.19, 0.44],
    "RDP w. only normal force": [0.47, 0.62, 0.36, 0.48],
    "RDP w. symmetric tokenizer": [0.75, 0.62, 0.36, 0.58],
    "RDP (default)": [0.99, 0.98, 0.88, 0.95],
}
# 표 V: 시도 수가 표에 없다. 조건당 10회로 가정한다.
CHUNK = {
    "DP w. tactile emb., chunk 8": (100, 0.15),
    "DP w. tactile emb., chunk 2": (20, 0.10),
    "DP w. tactile emb., chunk 8, tau 0.2": (30, 0.05),
    "DP w. tactile emb., chunk 8, tau 0.5": (0, 0.00),
    "DP w. tactile emb., chunk 8, tau 0.8": (100, 0.15),
    "RDP (GelSight), chunk 8": (100, 0.50),
}
N_TRIAL = 10          # 조건당 시행 수 (논문 §V-A-4: "10 trials for each test-time variation")

# ---- 1. 일관성 검사 ----
checks = {}
def mean_check(tab, name):
    out = []
    for k, v in tab.items():
        m = np.mean(v[:3])
        out.append((k, round(float(m), 3), v[3], bool(abs(round(float(m), 2) - v[3]) < 0.006)))
    return out
checks["peel_all_equals_mean_of_conditions"] = mean_check(PEEL, "peel")
checks["wipe_all_equals_mean_of_conditions"] = mean_check(WIPE, "wipe")
checks["abl_all_equals_mean_of_conditions"] = mean_check(ABL, "abl")
checks["abl_default_equals_table_II_force_row"] = ABL["RDP (default)"] == PEEL["RDP (Force)"]
# 점수 격자: 10회 평균이므로 0.05의 배수여야 한다 (점수 {0, 0.5, 1}일 때). 연속 점수인 Peeling은 제외
checks["wipe_condition_scores_on_0.05_grid"] = bool(all(abs(x * 20 - round(x * 20)) < 1e-9 for r in WIPE.values() for x in r[:3]))
# 들기 점수 가능성: L회 들기 성공, 점수 1은 들기 성공 중 압박 없는 경우, 0.5는 부분 압박.
# 합 S = 10 * score 는 0.5L 이상 L 이하의 0.5 배수여야 하고 L = 0이면 0이어야 한다.
lift_feas = {}
for name, d in LIFT.items():
    for cup in ("soft", "hard"):
        clamp, lift, score = d[cup]
        S = round(score * N_TRIAL * 2) / 2
        ok = (S == 0) if lift == 0 else (0.5 * lift <= S <= lift)
        lift_feas[f"{name} / {cup}"] = bool(ok)
checks["lift_scores_feasible_for_lift_counts"] = lift_feas
checks["lift_all_is_mean_of_two_cups"] = {k: (round((v["soft"][2] + v["hard"][2]) / 2, 2) == v["all"]) for k, v in LIFT.items()}
checks["abl_normal_force_drop_text"] = round(ABL["RDP (default)"][3], 2)  # 본문 0.95 -> 0.48, 0.58
res["consistency"] = checks

# ---- 2. 이진 성공: Wilson과 Fisher ----
def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(c - h, 3), round(c + h, 3))

binary = {}
for name, d in LIFT.items():
    clamp = d["soft"][0] + d["hard"][0]
    lift = d["soft"][1] + d["hard"][1]
    binary[name] = dict(clamp=f"{clamp}/20", clamp_wilson95=wilson(clamp, 20),
                        lift=f"{lift}/20", lift_wilson95=wilson(lift, 20))
res["binary_wilson"] = binary

def fisher(a, b, n=20):
    """a, b = 성공 수 (각 n회), 양측 Fisher 정확 검정"""
    _, p = stats.fisher_exact([[a, n - a], [b, n - b]])
    return float(f"{p:.3g}")

pairs = {
    "lift: RDP (Force) vs DP w. tactile emb.": (18, 2),
    "lift: RDP (GelSight + MCTac) vs DP w. tactile emb.": (18, 2),
    "lift: DP w. tactile emb. vs DP": (2, 0),
    "clamp: RDP (Force) vs DP w. tactile emb.": (19, 3),
    "clamp: RDP (GelSight + MCTac) vs DP w. tactile emb.": (19, 3),
}
res["binary_fisher"] = {k: fisher(*v) for k, v in pairs.items()}

# ---- 3. 점수 Welch 검정: 가능한 [0, 1] 분포 중 최대 분산을 쓴 상한 p ----
def welch_pmax(m1, n1, m2, n2):
    """각 조건의 분산을 m(1-m)로 두었을 때의 Welch 양측 p (가능한 분포 중 최대 분산 -> p의 상한)"""
    v1 = m1 * (1 - m1) * n1 / (n1 - 1)
    v2 = m2 * (1 - m2) * n2 / (n2 - 1)
    t, p = stats.ttest_ind_from_stats(m1, math.sqrt(v1), n1, m2, math.sqrt(v2), n2, equal_var=False)
    return float(f"{float(p):.3g}")

def welch_sd(m1, m2, sd, n1=N_TRIAL, n2=N_TRIAL):
    """시행별 SD를 가정했을 때 (참고용 시나리오)"""
    t, p = stats.ttest_ind_from_stats(m1, sd, n1, m2, sd, n2, equal_var=False)
    return float(f"{float(p):.3g}")

score = {}
# 조건별 비교 (n = 10): Peeling "perturb after contact", Wiping "perturb after contact"
for task, tab in (("Peeling", PEEL), ("Wiping", WIPE)):
    rdp_key = "RDP (Force)"
    base_keys = ["DP", "DP w. tactile emb."]
    for cond_idx, cond in enumerate(["no perturb", "perturb before contact", "perturb after contact"]):
        for b in base_keys:
            score[f"{task} {cond}: {rdp_key} vs {b}"] = welch_pmax(tab[rdp_key][cond_idx], N_TRIAL,
                                                                 tab[b][cond_idx], N_TRIAL)
    # 전체 평균 (n = 30 per method)
    for b in base_keys:
        score[f"{task} All: {rdp_key} vs {b} (n=30 each)"] = welch_pmax(tab[rdp_key][3], 30, tab[b][3], 30)
# Peeling 전 조건에서 RDP (GelSight)와 DP w. tactile emb.의 "after contact"
score["Peeling after contact: RDP (GelSight) vs DP w. tactile emb."] = welch_pmax(
    PEEL["RDP (GelSight)"][2], N_TRIAL, PEEL["DP w. tactile emb."][2], N_TRIAL)
res["score_welch_pmax"] = score

# 시나리오 SD: 시행별 SD를 0.2, 0.3으로 가정했을 때 Wiping 전체 평균 (n=30)
res["wipe_all_rdpF_vs_dp_sd_scenarios"] = {
    f"sd={sd}": welch_sd(WIPE["RDP (Force)"][3], WIPE["DP"][3], sd, 30, 30) for sd in (0.2, 0.3)
}

# ---- 4. "35% 이상 개선" 주장 점검 (표 II, III, IV의 All 점수, 최고 기준선 대비) ----
claim = {}
for task, tab, rdp, base_keys in (
        ("Peeling", PEEL, "RDP (Force)", ["DP", "DP w. tactile emb."]),
        ("Wiping", WIPE, "RDP (Force)", ["DP", "DP w. tactile emb."]),
        ("Lifting", {k: [0, 0, 0, v["all"]] for k, v in LIFT.items()}, "RDP (Force)", ["DP", "DP w. tactile emb."])):
    best = max(base_keys, key=lambda b: tab[b][3])
    r = tab[rdp][3]
    bb = tab[best][3]
    claim[task] = dict(best_baseline=best, best_all=bb, rdp_all=r,
                       abs_gain_points=round((r - bb) * 100, 1),
                       rel_gain_percent=round((r / bb - 1) * 100, 1) if bb > 0 else None)
res["claim_gt35_check"] = claim

# ---- 5. 표 V: 시도 수 가정 10회 (표에 없음) ----
res["chunk_table_v"] = {k: v for k, v in CHUNK.items()}
res["chunk_fisher_chunk8_vs_chunk2_assumed_n10"] = fisher(10, 2, n=10)
res["chunk_fisher_rdp_vs_dp_chunk8_assumed_n10"] = fisher(10, 10, n=10)

# ---- 6. 표 VI 대 표 II: 절제 점수 차이 ----
res["ablation_drop_check"] = dict(
    default_all=ABL["RDP (default)"][3],
    normal_force_only_all=ABL["RDP w. only normal force"][3],
    symmetric_tokenizer_all=ABL["RDP w. symmetric tokenizer"][3],
    p_max_default_vs_symmetric=welch_pmax(ABL["RDP (default)"][3], 30, ABL["RDP w. symmetric tokenizer"][3], 30),
    p_max_default_vs_normal_only=welch_pmax(ABL["RDP (default)"][3], 30, ABL["RDP w. only normal force"][3], 30),
)


# ---- 7. 추가 비교: 촉각 입력 효과, 센서 비교, 사용자 연구 (주장 점검용) ----
extra = {}
extra["peel All: DP vs DP w. tactile emb. (n=30 each)"] = welch_pmax(PEEL["DP"][3], 30, PEEL["DP w. tactile emb."][3], 30)
extra["wipe All: DP vs DP w. tactile emb. (n=30 each)"] = welch_pmax(WIPE["DP"][3], 30, WIPE["DP w. tactile emb."][3], 30)
extra["peel All: RDP (Force) vs RDP (GelSight) (n=30 each)"] = welch_pmax(PEEL["RDP (Force)"][3], 30, PEEL["RDP (GelSight)"][3], 30)
extra["wipe All: RDP (Force) vs RDP (GelSight) (n=30 each)"] = welch_pmax(WIPE["RDP (Force)"][3], 30, WIPE["RDP (GelSight)"][3], 30)
extra["peel All: RDP (GelSight) vs RDP (MCTac) (n=30 each)"] = welch_pmax(PEEL["RDP (GelSight)"][3], 30, PEEL["RDP (MCTac)"][3], 30)
extra["peel All: DP vs RDP (Force), 시나리오 시행별 SD=0.2 (참고용, 상한 아님)"] = welch_sd(PEEL["DP"][3], PEEL["RDP (Force)"][3], 0.2, 30, 30)
# 사용자 연구: 10 x 10 x 2 = 200 trials, 조건당 100회로 가정 (논문 부록 H). 안정 힘 비율 0.58 대 0.87.
def fisher_n(a, na, b, nb):
    _, p = stats.fisher_exact([[a, na - a], [b, nb - b]])
    return float(f"{p:.3g}")
extra["user study stable-force ratio 0.58 vs 0.87 (n=100 per arm, assumed)"] = fisher_n(58, 100, 87, 100)
res["extra_comparisons"] = extra

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
if __name__ == "__main__":
    print(json.dumps(res, ensure_ascii=False, indent=2))
