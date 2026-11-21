"""GR-1 (arXiv:2312.13139 v2) 보고 수치 재계산. 표준 라이브러리만 사용, 학습·재현 없음.

입력: 논문 표 값을 아래에 직접 옮겨 적음 (출처 표 번호 주석). 출력: recompute_results.json.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
N_CHAIN = 1000  # 논문 A.2: 1000 unique sequence instruction chains (성공률이 0.001 단위인지로 확인)

# Table 1: 연속 k개 성공률 (k=1..5), 보고된 Avg. Len.
T1 = {
    "ABCD→D": {
        "MCIL": ([0.373, 0.027, 0.002, 0.000, 0.000], 0.40),
        "RT-1": ([0.844, 0.617, 0.438, 0.323, 0.227], 2.45),
        "HULC": ([0.889, 0.733, 0.587, 0.475, 0.383], 3.06),
        "MT-R3M": ([0.752, 0.527, 0.375, 0.258, 0.163], 2.08),
        "GR-1": ([0.949, 0.896, 0.844, 0.789, 0.731], 4.21),
    },
    "ABC→D": {
        "MCIL": ([0.304, 0.013, 0.002, 0.000, 0.000], 0.31),
        "RT-1": ([0.533, 0.222, 0.094, 0.038, 0.013], 0.90),
        "HULC": ([0.418, 0.165, 0.057, 0.019, 0.011], 0.67),
        "MT-R3M": ([0.529, 0.234, 0.105, 0.043, 0.018], 0.93),
        "GR-1": ([0.854, 0.712, 0.596, 0.497, 0.401], 3.06),
    },
    "10% data": {
        "RT-1": ([0.249, 0.069, 0.015, 0.006, 0.000], 0.34),
        "HULC": ([0.668, 0.295, 0.103, 0.032, 0.013], 1.11),
        "MT-R3M": ([0.408, 0.146, 0.043, 0.014, 0.002], 0.61),
        "GR-1": ([0.778, 0.533, 0.332, 0.218, 0.139], 2.00),
    },
    "unseen lang": {
        "RT-1": ([0.494, 0.222, 0.086, 0.036, 0.017], 0.86),
        "HULC": ([0.715, 0.470, 0.308, 0.199, 0.130], 1.82),
        "MT-R3M": ([0.512, 0.249, 0.106, 0.040, 0.017], 0.92),
        "GR-1": ([0.764, 0.555, 0.381, 0.270, 0.196], 2.17),
    },
}

# Table 4: ablation (Pre-training 여부, Data 분할) -> (k=1..5, Avg. Len)
T4 = [
    ("no VP, no PT", "ABCD→D", [0.889, 0.775, 0.661, 0.549, 0.459], 3.33),
    ("VP only (no PT)", "ABCD→D", [0.918, 0.833, 0.761, 0.685, 0.619], 3.82),
    ("full GR-1", "ABCD→D", [0.949, 0.896, 0.844, 0.789, 0.731], 4.21),
    ("no VP, no PT", "ABC→D", [0.823, 0.609, 0.425, 0.318, 0.225], 2.40),
    ("VP only (no PT)", "ABC→D", [0.815, 0.651, 0.498, 0.392, 0.297], 2.65),
    ("full GR-1", "ABC→D", [0.854, 0.712, 0.596, 0.497, 0.401], 3.06),
    ("no VP, no PT", "10% data", [0.526, 0.288, 0.138, 0.061, 0.022], 1.04),
    ("VP only (no PT)", "10% data", [0.698, 0.415, 0.223, 0.133, 0.052], 1.52),
    ("full GR-1", "10% data", [0.778, 0.533, 0.332, 0.218, 0.139], 2.00),
]

# Table 5: 예측 step (Pre-training 없음, ABCD→D)
T5 = [
    (1, [0.895, 0.802, 0.710, 0.643, 0.562], 3.61),
    (3, [0.918, 0.833, 0.761, 0.685, 0.619], 3.82),
    (5, [0.909, 0.806, 0.719, 0.649, 0.583], 3.67),
]

# Table 2: 실로봇 성공률 (시도 횟수는 논문에 없음)
T2 = {
    "RT-1": {"seen": 0.27, "unseen_inst": 0.13, "unseen_cat": 0.00, "artic": 0.35},
    "MT-R3M": {"seen": 0.15, "unseen_inst": 0.13, "unseen_cat": 0.10, "artic": 0.30},
    "GR-1": {"seen": 0.79, "unseen_inst": 0.73, "unseen_cat": 0.30, "artic": 0.75},
}


def wilson(p, n, z=1.96):
    """Wilson 95% 신뢰구간."""
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def two_prop_z(p1, n1, p2, n2):
    """독립 가정 two-proportion z-test (양측 p). 같은 chain을 쓰면 양(+) 상관이라 이 검정은 보수적."""
    x1, x2 = p1 * n1, p2 * n2
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan")
    z = (p1 - p2) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p


def fisher_exact_two_sided(a, n1, c, n2):
    """2x2 Fisher exact (양측). a/n1 vs c/n2, 정수 카운트."""
    total_pos = a + c
    total = n1 + n2

    def prob(x):
        return math.comb(n1, x) * math.comb(n2, total_pos - x) / math.comb(total, total_pos)

    p_obs = prob(a)
    p_val = 0.0
    for x in range(max(0, total_pos - n2), min(n1, total_pos) + 1):
        px = prob(x)
        if px <= p_obs * (1 + 1e-9):
            p_val += px
    return min(1.0, p_val)


def main():
    out = {}

    # 1. Avg. Len 재계산 = sum_k P(>=k) (k=1..5). 2자리 반올림 일치 확인.
    avg_check = []
    for split, methods in T1.items():
        for m, (sr, reported) in methods.items():
            calc = sum(sr)
            ok = abs(calc - reported) <= 0.005 + 1e-9
            avg_check.append({"split": split, "method": m, "calc": round(calc, 4), "reported": reported, "match": ok})
    for name, split, sr, reported in T4:
        calc = sum(sr)
        avg_check.append({"split": f"T4 {split}", "method": name, "calc": round(calc, 4), "reported": reported,
                          "match": abs(calc - reported) <= 0.005 + 1e-9})
    for step, sr, reported in T5:
        calc = sum(sr)
        avg_check.append({"split": "T5 ABCD→D", "method": f"step {step}", "calc": round(calc, 4), "reported": reported,
                          "match": abs(calc - reported) <= 0.005 + 1e-9})
    out["avg_len_check"] = avg_check
    out["avg_len_mismatches"] = [r for r in avg_check if not r["match"]]

    # 2. 단조성: P(>=1) >= P(>=2) >= ... 과 0.001 단위(n=1000) 확인
    mono, grid = [], []
    all_rows = []
    for split, methods in T1.items():
        for m, (sr, _) in methods.items():
            all_rows.append((f"T1 {split}", m, sr))
    for name, split, sr, _ in T4:
        all_rows.append((f"T4 {split}", name, sr))
    for step, sr, _ in T5:
        all_rows.append(("T5 ABCD→D", f"step {step}", sr))
    for split, m, sr in all_rows:
        if any(sr[i] < sr[i + 1] for i in range(4)):
            mono.append((split, m))
        if any(abs(round(x * N_CHAIN) - x * N_CHAIN) > 1e-6 for x in sr):
            grid.append((split, m))
    out["monotonic_violations"] = mono
    out["non_integer_count_n1000_violations"] = grid

    # 3. ABCD→D: GR-1 vs 최고 baseline (k=1 two-proportion, Wilson CI)
    def ci_block(sr):
        return [{"k": k + 1, "p": p, "ci95": [round(x, 4) for x in wilson(p, N_CHAIN)]} for k, p in enumerate(sr)]

    out["wilson_GR1_ABCD"] = ci_block(T1["ABCD→D"]["GR-1"][0])
    out["wilson_HULC_ABCD"] = ci_block(T1["ABCD→D"]["HULC"][0])

    sig = []
    for split, base in [("ABCD→D", "HULC"), ("ABC→D", "RT-1"), ("10% data", "HULC"), ("unseen lang", "HULC")]:
        g = T1[split]["GR-1"][0][0]
        b = T1[split][base][0][0]
        z, p = two_prop_z(g, N_CHAIN, b, N_CHAIN)
        sig.append({"split": split, "GR1_k1": g, "baseline": base, "baseline_k1": b,
                    "diff": round(g - b, 3), "z": round(z, 2), "p_two_sided": p})
    out["k1_significance_independent_approx"] = sig

    # 4. Avg. Len 불확실성 상한: 연속 k의 SE 합 (상관 미지 -> 보수적 상한, Var(sum) <= (sum SE)^2)
    avg_se = []
    for split, base in [("ABCD→D", "HULC"), ("ABC→D", "RT-1"), ("10% data", "HULC")]:
        for m in ["GR-1", base]:
            sr = T1[split][m][0]
            se_upper = sum(math.sqrt(p * (1 - p) / N_CHAIN) for p in sr)
            avg_se.append({"split": split, "method": m, "avg_len": round(sum(sr), 3),
                           "avg_len_se_upper_bound": round(se_upper, 3),
                           "ci95_half_width_upper": round(1.96 * se_upper, 3)})
    out["avg_len_se_upper_bound"] = avg_se

    # 5. Table 7: 34개 작업별 성공률 평균 (논문 표에 있는 행 수 확인)
    T7 = [
        (94.9, 71.2, 51.6), (99.3, 99.6, 86.9), (98.5, 91.7, 50.6), (100, 100, 95.6), (82.9, 79.8, 74.7),
        (100, 93.8, 80.0), (97.8, 93.8, 56.4), (91.3, 89.1, 34.8), (99.4, 100, 94.2), (98.6, 81.9, 45.3),
        (97.7, 76.7, 36.5), (94.1, 72.0, 73.8), (99.2, 99.5, 90.7), (99.4, 98.6, 79.8), (97.1, 71.2, 66.7),
        (84.1, 72.7, 56.1), (99.5, 98.8, 91.3), (99.3, 100, 79.6), (100, 98.7, 95.6), (80.1, 45.7, 43.2),
        (61.8, 60.3, 50.0), (82.3, 77.8, 54.7), (97.1, 66.2, 64.9), (98.9, 98.6, 95.0), (95.3, 70.5, 61.5),
        (89.6, 84.5, 73.8), (97.0, 90.1, 54.7), (54.2, 49.3, 43.6), (100, 81.8, 50.0), (91.5, 79.1, 51.7),
        (100, 84.4, 100.0), (96.4, 70.4, 66.0), (53.6, 50.0, 33.9), (100, 100, 87.5),
    ]
    cols = list(zip(*T7))
    out["table7_rows"] = len(T7)
    out["table7_column_means"] = [round(sum(c) / len(c), 2) for c in cols]
    out["table7_note"] = "CALVIN 34개 작업 중 표에 있는 행 수와 비교. 평균은 작업 가중 평균이라 Table 1 값과 정확히 일치하지 않을 수 있음"

    # 6. 실로봇: 시도 횟수 미보고. 가정 n별 Wilson CI와 Fisher p (가정 값, 보고 값 아님)
    rr = []
    for n in [10, 20, 30, 50]:
        for cond in ["seen"]:
            a = round(T2["GR-1"][cond] * n)
            c = round(T2["RT-1"][cond] * n)
            p_fisher = fisher_exact_two_sided(a, n, c, n)
            rr.append({"assumed_n_per_arm": n, "GR1_k": a, "RT1_k": c,
                       "GR1_ci95": [round(x, 3) for x in wilson(a / n, n)],
                       "RT1_ci95": [round(x, 3) for x in wilson(c / n, n)],
                       "fisher_p_two_sided": round(p_fisher, 4)})
    out["real_robot_hypothetical_n"] = rr
    out["real_robot_note"] = "논문은 시도 횟수를 보고하지 않음. 위 n은 가정이며 반올림된 비율 기준 근사."

    # 7. 10% 데이터 비율 (논문: 66 traj x 34 tasks = 2244 / 22,966)
    out["ten_percent_share"] = round(66 * 34 / 22966, 4)

    path = os.path.join(HERE, "recompute_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({
        "avg_len_mismatches": out["avg_len_mismatches"],
        "monotonic_violations": out["monotonic_violations"],
        "non_integer_violations": out["non_integer_count_n1000_violations"],
        "k1_sig": out["k1_significance_independent_approx"],
        "table7_rows": out["table7_rows"],
        "table7_means": out["table7_column_means"],
        "ten_percent": out["ten_percent_share"],
        "real_robot": out["real_robot_hypothetical_n"],
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
