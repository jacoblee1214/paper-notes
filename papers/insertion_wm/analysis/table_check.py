"""InsertionWM 수치 재계산 (CPU, numpy·scipy, 수 초). 실행: python3 analysis/table_check.py

논문은 시도 횟수(trial)를 일부만 밝혔습니다. 그래서 다음과 같이 나눠 계산합니다.
- 확인된 것: 미지 객체 10개 × 100회 = 1000회 (본문 VI-A), 학습 객체 90개 × 100회 (본문 VI-A)
- 가정: 생성 모델(AutoMate) 7%의 시도 수는 밝혀지지 않았습니다. 그래서 100회와 1000회 두 가지로 봅니다.
- 쌍 설계 여부 불명. 쌍별 집계가 없어 McNemar는 쓰지 않습니다. 비쌍 검정은 근사입니다.
- 객체 단위 분산은 Fig. 5 막대를 눈으로 읽을 수 없어 가정(SD)으로 둡니다.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "table_check_results.json"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def holm(ps):
    ps = list(ps)
    order = np.argsort(ps)
    m = len(ps)
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * ps[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


res = {}

# T1. 헤드라인: 미지 객체 56% (1000회) 대 AutoMate 7%
k_ours = 560
n_ours = 1000
res["T1_unseen_ours"] = {"k": k_ours, "n": n_ours, "p": k_ours / n_ours,
                        "wilson95": wilson(k_ours, n_ours),
                        "note": "10개 × 100회 (본문 VI-A에서 확인)"}
for n_auto in (100, 1000):
    k_auto = round(0.07 * n_auto)
    res[f"T1_fisher_ours_vs_automate_n{n_auto}"] = {
        "k_auto": k_auto, "n_auto": n_auto, "p_auto": k_auto / n_auto,
        "wilson95_auto": wilson(k_auto, n_auto),
        "fisher_p_two_sided": fisher(k_ours, n_ours, k_auto, n_auto),
        "note": "AutoMate 시도 수 불명. 가정 n" + str(n_auto)}
# 같은 학습 객체 수에서 비교하면? 논문 Fig. 1에서 AutoMate 7% 점의 x는 약 15–25 사이. 우리 모델은 x=15에서 약 40%
res["T1_matched_size_note"] = ("Fig. 1: 우리 모델의 미지 객체 성공률은 학습 객체 5/15/45/75/90개에서 약 27/40/50/53/56%, "
                               "AutoMate 점 7%는 학습 객체 약 15–25개(정확한 x 불명)에 있습니다. 56%와 7%는 학습 규모가 다릅니다.")

# T2. 학습 객체 90개(seen) 55.0% (본문) 대 54.5% (Fig. 7 막대): 불일치 기록
res["T2_seen_90"] = {"text_pct": 55.0, "fig7_pct": 54.5, "n_trials": 9000,
                    "wilson95": wilson(round(0.55 * 9000), 9000),
                    "note": "본문 55.0%와 Fig. 7 54.5%가 다름. 시도 수 9000회(90 × 100) 가정"}
res["T2_unseen_ci_halfwidth_pp"] = round(100 * (wilson(560, 1000)[1] - 0.56), 2)

# T3. 전문가(specialist) 79.78% 대 AutoMate 70.31% (100개 객체)
# 쌍(객체별) 차이의 SD는 논문에 없음. 그래서 SD를 가정해 검정력을 본다.
diff = 79.78 - 70.31
sens = []
for sd in (10, 15, 20, 30):
    se = sd / math.sqrt(100)
    z = diff / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    sens.append({"assumed_sd_pp": sd, "se_pp": round(se, 2), "z": round(z, 2), "p_normal": float(p)})
res["T3_specialists_paired_sensitivity"] = {"diff_pp": round(diff, 2), "n_assemblies": 100, "rows": sens,
                                           "note": "객체 단위 쌍 차이 SD는 Fig. 5에서 추정 불가. 가정 SD에 따른 민감도"}
# 시도 단위로 보면 과신(over-precise). 객체를 단위로 봐야 함.
n_trials_spec = 100 * 5 * 100
res["T3_trial_level_naive"] = {"n_trials": n_trials_spec, "wilson95_ours": wilson(round(0.7978 * n_trials_spec), n_trials_spec),
                              "note": "시도 단위 Wilson은 객체 간 분산을 무시해 지나치게 좁음. 참고용"}

# T4. Fig. 7 절제 (생성 모델 90개, 시드 수 불명). 막대 값만 사용.
abl = {
    "sensors_generalist": {"proprio_only": 39.6, "depth_ours": 54.5, "depth_alt": 46.4},
    "curriculum_generalist": {"none": 52.1, "curriculum_ours": 54.5, "demonstrations": 56.5},
    "sensors_specialists": {"proprio_only": 55.2, "depth_ours": 79.8, "depth_alt": 72.7},
    "curriculum_specialists": {"none": 76.2, "curriculum_ours": 79.8, "curriculum_alt": 79.5},
}
abl_out = {}
for name, d in abl.items():
    keys = list(d.keys())
    base = d[keys[0]]
    row = {}
    for k in keys[1:]:
        diff = round(d[k] - base, 1)
        # 시도 수: 생성 모델 90×100=9000, 전문가 100×100×5=50000으로 가정한 이항 근사(시드 분산 무시)
        n = 9000 if "generalist" in name else 50000
        k1 = round(d[k] / 100 * n)
        k0 = round(base / 100 * n)
        row[k] = {"diff_pp_vs_" + keys[0]: diff, "fisher_p_naive": fisher(k1, n, k0, n)}
    abl_out[name] = row
res["T4_ablation_naive_binomial"] = abl_out
# Holm: 절제 생성 모델 비교 4개 (시드 분산 무시한 근사)
ps = []
labels = []
for name, row in abl_out.items():
    for k, v in row.items():
        ps.append(v["fisher_p_naive"])
        labels.append(f"{name}:{k}")
adj = holm(ps)
res["T4_holm_naive"] = {lab: {"p_raw": float(p), "p_holm": float(a)} for lab, p, a in zip(labels, ps, adj)}
res["T4_note"] = ("절제 값은 시드 수가 밝혀지지 않아(생성 모델 단일 실행으로 보임) 시드 분산이 빠져 있습니다. "
                  "naive 검정은 과신입니다. 차이 2.4pp(curriculum)는 시드 간 변동보다 작을 수 있습니다.")

# T5. 미세조정 (Fig. 6). 5 seed, 95% CI 음영. 30k 스텝 대 2M 스텝 기준선.
res["T5_finetune_note"] = ("Fig. 6 x축은 30k 환경 스텝까지. 특정 스크래치 기준선은 2M 스텝 값(점선). "
                           "두 곡선의 스텝 범위가 달라 '데이터 효율'은 30k 시점 비교로만 확인 가능. "
                           "1053과 1132에서는 미세조정도 낮게 머묾.")
# 눈으로 읽은 0스텝 미세조정 값(Fig. 6, 근사 ±0.1): 객체별 평균
read0 = {"1036": 0.95, "1041": 0.80, "1053": 0.02, "1079": 0.65, "1092": 0.75,
         "1102": 0.55, "1125": 0.60, "1129": 0.85, "1132": 0.00, "1136": 0.80}
res["T5_fig6_step0_read_mean"] = {"values": read0, "mean_read": round(float(np.mean(list(read0.values()))), 3),
                                 "note": "눈으로 읽은 값(±0.1). 본문 56%와 대략 같은 수준인지 확인용"}

# T6. 계산량 추정 (TD-MPC2 기본 설정: horizon 3, samples 512, iterations 6, pi trajs 24)
H, N, IT, PI = 3, 512, 6, 24
calls = IT * N * H + PI * H
for label, params in (("dynamics~0.8M 가정", 0.8e6), ("전체 5M 상한", 5e6)):
    flops = 2 * params * calls
    res.setdefault("T6_planning_cost_per_step", {})[label] = {
        "model_calls_per_step": calls, "gflop_per_step": round(flops / 1e9, 1)}
res["T6_note"] = ("논문에 샘플 수, 호라이즌, 반복 수가 없음. TD-MPC2 공개 기본값(commit 332f19e)을 가정한 추정. "
                  "MLP 호출만 셈. CNN 인코더와 메모리 이동은 제외.")

# T7. 객체 단위 불확실성: 미지 객체는 10개뿐이라 시도 단위 CI는 과신이다.
# 객체 간 SD를 가정(Fig. 5·6의 객체별 퍼짐을 눈으로 본 25–30pp)하고 10개 평균의 95% 구간을 계산한다.
obj = {}
for sd in (20, 25, 30):
    se = sd / math.sqrt(10)
    obj[f"sd{sd}"] = {"se_pp": round(se, 1), "ci95_halfwidth_pp": round(1.96 * se, 1),
                      "ci95_pct": [round(56 - 1.96 * se, 1), round(56 + 1.96 * se, 1)]}
res["T7_unseen_object_level_ci"] = {"n_objects": 10, "rows": obj,
                                   "note": "객체 간 SD는 논문에 없음. 가정 값. Fig. 6 0스텝 값(눈으로 읽음) SD는 아래 참고"}
v = np.array(list(read0.values()))
res["T7_fig6_step0_read_sd"] = round(float(v.std(ddof=1)), 3)

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
print(json.dumps({k: res[k] for k in ("T1_unseen_ours", "T2_unseen_ci_halfwidth_pp", "T6_planning_cost_per_step")},
                 ensure_ascii=False, indent=1))
print("wrote", OUT)
