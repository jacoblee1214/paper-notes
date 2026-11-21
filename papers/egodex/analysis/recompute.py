"""EgoDex (arXiv:2505.11709 v3) 보고 수치 재계산 검사. 순수 Python, CPU 수 초.

입력: 논문 Table 2–6 값을 직접 옮긴 상수 (paper/paper.txt, v3 기준).
출력: analysis/recompute_results.json
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- Table 2: 2초 horizon, (avg K=1,5,10 ; final K=1,5,10) ----
T2 = {
    "Dec+BC":      {"avg": [0.045, 0.045, 0.045], "final": [0.062, 0.062, 0.062]},
    "Dec+DDPM":    {"avg": [0.053, 0.044, 0.041], "final": [0.071, 0.050, 0.044]},
    "Dec+FM":      {"avg": [0.052, 0.042, 0.040], "final": [0.071, 0.049, 0.043]},
    "EncDec+BC":   {"avg": [0.044, 0.044, 0.044], "final": [0.060, 0.060, 0.060]},
    "EncDec+DDPM": {"avg": [0.052, 0.042, 0.039], "final": [0.071, 0.048, 0.043]},
    "EncDec+FM":   {"avg": [0.051, 0.041, 0.038], "final": [0.070, 0.047, 0.041]},
}
# ---- Table 3: Dec+BC, horizon (H=30,60,90) ----
T3 = {"avg": [0.031, 0.045, 0.053], "final": [0.049, 0.062, 0.069]}
# ---- Table 4: goal conditioning (Dec+BC vs Dec+BC w/ goal) ----
T4 = {"base": {"avg": 0.045, "final": 0.062}, "goal": {"avg": 0.035, "final": 0.029}}
# ---- Table 5: OOD, Dec+BC ----
T5_ID = {"avg": 0.045, "final": 0.062}
T5_OOD = {
    "Jigsaw Puzzle": (0.047, 0.065),
    "Tetra Board": (0.060, 0.082),
    "Knit Scarf": (0.064, 0.093),
    "Play Reversi": (0.065, 0.096),
    "Blowdry Hair": (0.083, 0.118),
    "Stamp Paper": (0.099, 0.162),
}
# ---- Table 6: EncDec+FM, horizon x K ----
T6 = {
    30: {"avg": [0.036, 0.028, 0.026], "final": [0.055, 0.037, 0.033]},
    60: {"avg": [0.051, 0.041, 0.038], "final": [0.070, 0.047, 0.041]},
    90: {"avg": [0.061, 0.050, 0.047], "final": [0.079, 0.053, 0.046]},
}


def rel(new, old):
    """상대 변화 (new - old) / old, 백분율."""
    return 100.0 * (new - old) / old


def red(better, worse):
    """개선율 = 1 - better/worse, 백분율."""
    return 100.0 * (1.0 - better / worse)


results = {"claims": [], "derived": {}, "consistency": {}, "sensitivity": {}}


def claim(key, text_value, recomputed, note):
    results["claims"].append({
        "key": key, "text_value": text_value,
        "recomputed": round(recomputed, 2), "note": note,
    })


# 1) 본문 Table 3 문장
claim("h1s_avg", 31, red(T3["avg"][0], T3["avg"][1]), "2s->1s 평균 거리 개선 (Table 3)")
claim("h1s_final", 21, red(T3["final"][0], T3["final"][1]), "2s->1s 최종 거리 개선 (Table 3)")
claim("h3s_avg", 18, rel(T3["avg"][2], T3["avg"][1]), "2s->3s 평균 거리 악화 (Table 3)")
claim("h3s_final", 11, rel(T3["final"][2], T3["final"][1]), "2s->3s 최종 거리 악화 (Table 3)")

# 2) Table 4 문장 (서로 다른 벤치마크: 예측 vs 역동역학 조건부)
claim("goal_avg", 22, red(T4["goal"]["avg"], T4["base"]["avg"]), "목표 이미지 조건 평균 개선 (Table 4)")
claim("goal_final", 53, red(T4["goal"]["final"], T4["base"]["final"]), "목표 이미지 조건 최종 개선 (Table 4)")

# 3) "BC가 K=1에서 확산/FM보다 약 15% 좋다"
bc_vs = {}
for dec in ["Dec", "EncDec"]:
    bc = T2[f"{dec}+BC"]["avg"][0]
    for rep in ["DDPM", "FM"]:
        other = T2[f"{dec}+{rep}"]["avg"][0]
        bc_vs[f"{dec}+BC vs {dec}+{rep} (avg, K=1)"] = round(red(bc, other), 1)
results["derived"]["bc_advantage_K1_avg_pct"] = bc_vs
claim("bc_k1_about15", 15, sum(bc_vs.values()) / len(bc_vs),
      "BC의 K=1 평균 거리 우위 (4개 비교 평균, 본문 '약 15%')")

# 4) "EncDec+FM이 K=5, K=10에서 최대 34% 우위" -- 비교 대상 불명확. 가능한 기준 모두 계산
fm10_avg = T2["EncDec+FM"]["avg"][2]
fm10_final = T2["EncDec+FM"]["final"][2]
cands = {}
for name, row in T2.items():
    if name == "EncDec+FM":
        continue
    cands[f"avg K=10 vs {name} K=1"] = round(red(fm10_avg, row["avg"][0]), 1)
    cands[f"final K=10 vs {name} K=1"] = round(red(fm10_final, row["final"][0]), 1)
results["derived"]["encdec_fm_k10_improvements_pct"] = cands
claim("fm_up_to_34", 34, red(fm10_final, T2["Dec+BC"]["final"][0]),
      "34%는 'final 거리, EncDec+FM K=10 vs Dec+BC K=1'(33.9%)에만 맞음. 모든 비교의 최댓값은 42.3%(final vs Dec+DDPM/FM K=1)")
results["derived"]["fm_34_match"] = "final K=10 vs Dec+BC K=1 = %.1f%%" % red(fm10_final, T2["Dec+BC"]["final"][0])

# 5) 샘플링 K에 따른 best-of-K 개선
kgain = {}
for name in ["Dec+DDPM", "Dec+FM", "EncDec+DDPM", "EncDec+FM"]:
    kgain[name] = {
        "avg_K1_to_K10_pct": round(red(T2[name]["avg"][2], T2[name]["avg"][0]), 1),
        "final_K1_to_K10_pct": round(red(T2[name]["final"][2], T2[name]["final"][0]), 1),
    }
results["derived"]["best_of_k_gain"] = kgain

# 6) OOD 대 ID (Dec+BC)
ood = {}
for k, (a, f) in T5_OOD.items():
    ood[k] = {"avg_vs_ID_pct": round(rel(a, T5_ID["avg"]), 1),
              "final_vs_ID_pct": round(rel(f, T5_ID["final"]), 1)}
results["derived"]["ood_vs_id_pct"] = ood

# 7) 표 사이 일관성
cons = {}
# Table 6 (H=60, EncDec+FM, K=1,5,10)과 Table 2 (EncDec+FM)
cons["table6_h60_equals_table2_encdecfm"] = (
    T6[60]["avg"] == T2["EncDec+FM"]["avg"] and T6[60]["final"] == T2["EncDec+FM"]["final"]
)
# Table 3 (Dec+BC, H=60) 과 Table 2 Dec+BC
cons["table3_h60_equals_table2_decbc"] = (T3["avg"][1] == T2["Dec+BC"]["avg"][0]
                                         and T3["final"][1] == T2["Dec+BC"]["final"][0])
# 단조성: K가 커질수록 감소 (확률 모델), H가 커질수록 증가
mono_k = all(T6[h]["avg"][i] >= T6[h]["avg"][i + 1] for h in T6 for i in range(2))
mono_h = all(T6[30]["avg"][i] <= T6[60]["avg"][i] <= T6[90]["avg"][i] for i in range(3))
cons["table6_monotone_in_K"] = mono_k
cons["table6_monotone_in_H"] = mono_h
# 최종 거리는 평균보다 항상 큰가 (Table 2, 6)
cons["final_ge_avg_all_rows"] = all(
    f >= a for row in T2.values() for a, f in zip(row["avg"], row["final"]))
results["consistency"].update(cons)

# 8) 데이터셋 통계 (본문과 README)
hrs_train, hrs_test, hrs_extra = 725, 7, 97
total_h = hrs_train + hrs_test + hrs_extra
frames = total_h * 3600 * 30
episodes = 338000
results["consistency"]["dataset"] = {
    "split_hours_sum": total_h,
    "paper_total_hours": 829,
    "test_share_of_train_plus_test_pct": round(100 * hrs_test / (hrs_train + hrs_test), 2),
    "frames_at_30hz_millions": round(frames / 1e6, 1),
    "paper_frames_millions": 90,
    "mean_episode_seconds": round(829 * 3600 / episodes, 2),
    "approx_test_episodes_from_7h": round(hrs_test * 3600 / (829 * 3600 / episodes)),
    "tasks_from_listing": 76 * 2 + 28 + 14,
    "paper_tasks": 194,
    "raw_over_compressed_ratio": round(500 / 2.0, 1),
    "paper_ratio_claim": 250,
    "repo_zip_total_GB": 5 * 300 + 16 + 200,
    "paper_disk_TB": 2.0,
}

# 9) 민감도: 테스트 분할 크기만 보면 얼마나 작은 차이를 가릴 수 있는지 (가정 명시)
# 가정: 테스트 에피소드 수 = 338,000 * 1%, 에피소드 단위 독립, 표준편차 sigma는 보고되지 않아 가정
n_eps = 338000 * 0.01
sens = {"assumed_test_episodes": n_eps, "assumed_sigma_m": [0.02, 0.03, 0.05], "rows": []}
for s in sens["assumed_sigma_m"]:
    se = s / math.sqrt(n_eps)
    mde = (1.96 + 0.8416) * math.sqrt(2) * s / math.sqrt(n_eps)  # 두 독립 평균, 양측 5%, 검정력 80%
    sens["rows"].append({"sigma_m": s, "SE_of_mean_m": round(se, 5), "MDE_80pct_m": round(mde, 5)})
results["sensitivity"] = sens

with open(os.path.join(HERE, "recompute_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(json.dumps(results["claims"], ensure_ascii=False, indent=1))
print(json.dumps(results["derived"]["encdec_fm_k10_improvements_pct"], indent=1))
print(results["derived"]["fm_34_match"])
print(json.dumps(results["consistency"], ensure_ascii=False, indent=1))
print(json.dumps(results["sensitivity"], indent=1))
