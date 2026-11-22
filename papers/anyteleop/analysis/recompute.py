"""AnyTeleop (arXiv:2307.04577v3 / RSS 2023) 보고 수치 재계산.

입력: 논문 표 II–V의 보고 값 (요약 통계만 있음, 원자료 없음).
출력: analysis/recompute_results.json, 표준출력 요약.
CPU 전용, 수 초. 설치 없이 python3 (numpy, scipy) 사용.
"""
import itertools
import json
import math
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "recompute_results.json")
results = {}

# ---------- 표 III: 시뮬 모방학습, 3 시드 x 100 시행, 평균 ± SD ----------
# (과제, 변형) -> {방법: (평균, SD)}; RL은 표 III 그대로
T3 = {
    ("Relocate", "Floating-Hand"): {"RL": (36.3, 15.3), "Baseline": (49.7, 18.3), "Ours": (53.7, 12.2)},
    ("Flip Mug", "Floating-Hand"): {"RL": (33.7, 15.0), "Baseline": (51.3, 34.7), "Ours": (47.3, 28.3)},
    ("Open Door", "Floating-Hand"): {"RL": (69.3, 38.0), "Baseline": (64.7, 14.7), "Ours": (73.3, 9.0)},
    ("Relocate", "Arm-Hand"): {"RL": (33.7, 29.3), "Baseline": (40.3, 36.7), "Ours": (70.0, 9.8)},
    ("Flip Mug", "Arm-Hand"): {"RL": (31.0, 28.7), "Baseline": (36.0, 32.4), "Ours": (53.7, 24.0)},
    ("Open Door", "Arm-Hand"): {"RL": (34.7, 31.7), "Baseline": (51.3, 30.7), "Ours": (79.7, 15.5)},
}
N_SEEDS = 3
N_TRIALS = 100


def welch(m1, s1, m2, s2, n=N_SEEDS):
    """두 집단 요약 통계로 Welch t 검정, 평균 차 95% CI (표본 SD, ddof=1)."""
    v1, v2 = s1 ** 2 / n, s2 ** 2 / n
    se = math.sqrt(v1 + v2)
    diff = m1 - m2
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n - 1) + v2 ** 2 / (n - 1))
    t = diff / se
    p = 2 * stats.t.sf(abs(t), df)
    tc = stats.t.ppf(0.975, df)
    return {"diff": round(diff, 2), "t": round(t, 3), "df": round(df, 2), "p": float(p),
            "ci95": [round(diff - tc * se, 2), round(diff + tc * se, 2)]}


def int_triples_consistent(mean, sd_rep, ddof):
    """3개 정수 백분율(0–100)로 보고된 평균(소수 1자리)과 SD(소수 1자리)를 만들 수 있는지 전수 검사."""
    target_sum_opts = [s for s in range(0, 301) if round(s / 3, 1) == round(mean, 1)]
    hits = []
    for a, b, c in itertools.combinations_with_replacement(range(0, 101), 3):
        if a + b + c not in target_sum_opts:
            continue
        x = np.array([a, b, c], dtype=float)
        sd = float(np.std(x, ddof=ddof))
        if round(sd, 1) == round(sd_rep, 1):
            hits.append((a, b, c))
    return hits


results["table3"] = {}
print("== 표 III 재계산 (3 시드, 시드당 100 시행)")
for (task, var), d in T3.items():
    key = f"{task}|{var}"
    row = {}
    for m in ["RL", "Baseline", "Ours"]:
        mean, sd = d[m]
        h1 = int_triples_consistent(mean, sd, 1)
        h0 = int_triples_consistent(mean, sd, 0)
        row[m] = {
            "mean": mean, "sd": sd,
            "int_consistent_ddof1": len(h1) > 0,
            "int_consistent_ddof0": len(h0) > 0,
            "example_ddof1": h1[0] if h1 else None,
            "example_ddof0": h0[0] if h0 else None,
        }
    row["Ours_vs_Baseline"] = welch(d["Ours"][0], d["Ours"][1], d["Baseline"][0], d["Baseline"][1])
    row["Ours_vs_RL"] = welch(d["Ours"][0], d["Ours"][1], d["RL"][0], d["RL"][1])
    results["table3"][key] = row
    b = row["Ours_vs_Baseline"]
    print(f"{key:28s} Ours-Base {b['diff']:+6.1f}  95%CI [{b['ci95'][0]:+.1f},{b['ci95'][1]:+.1f}]  p={b['p']:.3f}  "
          f"Ours-RL p={row['Ours_vs_RL']['p']:.4f}  정수일관(ddof1,ddof0)={row['Ours']['int_consistent_ddof1']},{row['Ours']['int_consistent_ddof0']}")

# 모든 셀의 정수 일관성 요약
incons = [(k, m) for k, r in results["table3"].items() for m in ["RL", "Baseline", "Ours"]
          if not (r[m]["int_consistent_ddof1"] or r[m]["int_consistent_ddof0"])]
results["table3_integer_inconsistent_cells"] = incons
print("정수 일관성 실패 셀:", incons if incons else "없음")

# 논문 본문 주장: 6 조건 중 5에서 AnyTeleop 평균이 높음 (Floating Flip Mug만 낮음)
wins = sum(1 for r in results["table3"].values() if r["Ours_vs_Baseline"]["diff"] > 0)
results["table3_ours_higher_than_baseline_count"] = wins
print(f"Ours 평균이 Baseline보다 높은 조건: {wins}/6")

# ---------- 표 IV: 실로봇 10 과제, AnyTeleop 10회 ----------
T4 = {
    "Pickup Box Object": (1.0, 0.9), "Pickup Fabric Toy": (1.0, 0.9), "Box Rotation": (0.6, 0.6),
    "Scissor Pickup": (0.8, 0.7), "Cup Stack": (0.9, 0.6), "Two Cup Stacking": (0.7, 0.3),
    "Pouring Cubes onto Plate": (0.7, 0.7), "Cup Into Plate": (1.0, 0.8), "Open Drawer": (1.0, 0.9),
    "Open Drawer and Pickup Object": (0.9, 0.6),
}
N4 = 10


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 3), round(c + h, 3)]


results["table4"] = {}
print("\n== 표 IV 재계산 (AnyTeleop 10회; 기준선 시도 수는 [54]에서 가져온 값이라 미상, 10으로 가정)")
wins = ties = losses = 0
for task, (ours, base) in T4.items():
    ko, kb = round(ours * N4), round(base * N4)
    _, p = stats.fisher_exact([[ko, N4 - ko], [kb, N4 - kb]])
    results["table4"][task] = {"ours_k": ko, "base_k": kb, "wilson_ours": wilson(ko, N4),
                               "fisher_p_two_sided": round(float(p), 3)}
    if ours > base: wins += 1
    elif ours == base: ties += 1
    else: losses += 1
    print(f"{task:30s} Ours {ko}/10  Base {kb}/10  Wilson(Ours) {wilson(ko, N4)}  Fisher p={p:.3f}")
ks_o = sum(round(v[0] * N4) for v in T4.values())
ks_b = sum(round(v[1] * N4) for v in T4.values())
results["table4_summary"] = {"ours_wins": wins, "ties": ties, "losses": losses,
                            "ours_total_k": ks_o, "base_total_k": ks_b, "total_trials": N4 * len(T4)}
print(f"우위 {wins}, 동률 {ties}, 열세 {losses}; 합계 성공 {ks_o}/100 대 {ks_b}/100 (기준선 n 가정 10)")
# 기준선 값이 0.1 단위인지 (n=10 가정과 일치하는지)
results["table4_baseline_multiple_of_0.1"] = all(abs(v[1] * 10 - round(v[1] * 10)) < 1e-9 for v in T4.values())
print("기준선 값이 10회 기준 정수 성공인지:", results["table4_baseline_multiple_of_0.1"])

# 검정력: 10회 vs 10회에서 차이 0.5를 80% 검정력으로 잡으려면 (근사, 정규근사 2표본 비율)
def n_needed(p1, p2, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)
results["table4_n_needed_for_0.4_gap_80pct"] = n_needed(0.9, 0.5)
print("차이 0.4(0.9 대 0.5)를 80% 검정력으로 잡는 팔당 시도 수(근사):", results["table4_n_needed_for_0.4_gap_80pct"])


# 잘못된 풀링 대조 (참고용): 300 시행을 독립으로 보면 과대 확신이 생김 (시드 분산 무시)
pool = {}
for (task, var), d in T3.items():
    k_o = round(d["Ours"][0] / 100 * 300)
    k_b = round(d["Baseline"][0] / 100 * 300)
    _, pp = stats.fisher_exact([[k_o, 300 - k_o], [k_b, 300 - k_b]])
    pool[f"{task}|{var}"] = {"ours_k_of_300": k_o, "base_k_of_300": k_b, "fisher_p_pooled_WRONG": float(pp)}
results["table3_pooled_wrong_check"] = pool
print("\n[참고] 300 시행 풀링 Fisher (잘못된 방식):", {k: f"{v['fisher_p_pooled_WRONG']:.1e}" for k, v in pool.items()})
# 표 IV 10과제 x 10회를 100 독립 시행으로 풀링 (과제 간 독립 아님, 상한 참고용)
_, p_pool4 = stats.fisher_exact([[86, 14], [70, 30]])
results["table4_pooled_100_fisher_p_UPPER_BOUND_ONLY"] = float(p_pool4)
print(f"[참고] 표 IV 풀링 86/100 대 70/100 Fisher p={p_pool4:.3f} (과제 간 독립 가정, 상한 참고용)")

# ---------- 표 II: 프로파일링 예산 점검 ----------
# 목표 주파수에서 프레임당 시간 예산: 25 Hz -> 40 ms, 120 Hz -> 8.33 ms
prof = {"Hand Pose (RGB)": {"desk": (26, 5), "lap": (34, 5), "budget_ms": 40},
        "Retargeting": {"desk": (9, 7), "lap": (10, 9), "budget_ms": 40},
        "Motion (120 Hz 목표)": {"desk": (8, 3), "lap": (11, 5), "budget_ms": 1000 / 120}}
results["table2"] = {}
print("\n== 표 II 예산 점검 (정규 근사로 예산 초과 비율 추정)")
for k, v in prof.items():
    r = {}
    for dev in ["desk", "lap"]:
        m, s = v[dev]
        over = float(stats.norm.sf((v["budget_ms"] - m) / s))
        r[dev] = {"mean_ms": m, "sd_ms": s, "frac_over_budget_normal_approx": round(over, 3)}
    results["table2"][k] = r
    print(f"{k:24s} 예산 {v['budget_ms']:.1f} ms | 데스크톱 초과 비율 ~{r['desk']['frac_over_budget_normal_approx']:.2f}, 노트북 ~{r['lap']['frac_over_budget_normal_approx']:.2f}")
results["table2_laptop_serial_sum_ms"] = 34 + 1 + 10
results["table2_desk_serial_sum_ms"] = 26 + 1 + 9
print("직렬 합(검출+융합+재타깃): 데스크톱", 26 + 1 + 9, "ms, 노트북", 34 + 1 + 10, "ms (25 Hz 예산 40 ms; 파이프라인이면 합이 아님)")

# ---------- 표 V: 카메라 구성 (단일 값, 시행 10회, 오차율 = 잘못 누른 키 / 전체 키) ----------
T5 = {"Single RGB": (109, 28.1), "Single RGB-D": (87, 21.8), "Two RGB-D": (74, 12.5)}
results["table5"] = {}
print("\n== 표 V 점검")
for name, (t, e) in T5.items():
    # 오차율이 k/N 형태인지 (N = 32 가정 시 가능한 정수 k, 절단·반올림 두 방식)
    cands = []
    for N in range(2, 201):
        for k in range(0, N + 1):
            if round(100 * k / N, 1) == e:
                cands.append((k, N, "round"))
            elif math.floor(100 * k / N * 10) / 10 == e:
                cands.append((k, N, "trunc"))
    small = [c for c in cands if c[1] <= 100][:6]
    results["table5"][name] = {"time_s": t, "error_pct": e, "k_over_N_examples_le100": small}
    print(f"{name:14s} 시간 {t} s, 오차 {e}%  가능한 k/N 예(N<=100): {small[:4]}")
results["table5_note"] = "시행 10회 단일 값, 표준편차 없음. 분모 키 수는 논문에 없음."

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
print("\n저장:", OUT)
