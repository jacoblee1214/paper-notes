"""DINO-WM (arXiv:2411.04983 v2) 보고 수치 재계산.

표준 라이브러리만 사용합니다. 입력은 논문 표의 보고값이고, SR은 시도 수 n으로
성공 횟수를 복원해 정확 검정을 합니다. CD는 시도 수 10이며 표준편차가 없어 검정하지 않습니다.
실행: python3 analysis/recompute.py  (수 초, 결과는 recompute_results.json)
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 논문 v2 Table 1 (SR는 n=50, CD는 n=10) -------------------------------
T1_SR_N = 50
T1_SR = {  # 환경: {방법: SR}
    "Maze":   {"IRIS": 0.74, "DreamerV3": 1.00, "TD-MPC2": 0.00, "Ours": 0.98},
    "Wall":   {"IRIS": 0.04, "DreamerV3": 1.00, "TD-MPC2": 0.00, "Ours": 0.96},
    "Reach":  {"IRIS": 0.18, "DreamerV3": 0.64, "TD-MPC2": 0.00, "Ours": 0.92},
    "PushT":  {"IRIS": 0.32, "DreamerV3": 0.30, "TD-MPC2": 0.00, "Ours": 0.90},
}
T1_CD_N = 10
T1_CD = {  # 환경: {방법: CD (낮을수록 좋음)}
    "Rope":     {"IRIS": 1.11, "DreamerV3": 2.49, "TD-MPC2": 2.52, "Ours": 0.41},
    "Granular": {"IRIS": 0.37, "DreamerV3": 1.05, "TD-MPC2": 1.21, "Ours": 0.26},
}

# ---- 논문 v2 Table 2 (인코더 비교, SR n=50, CD n=10) ----------------------
T2_SR = {
    "Maze":  {"R3M": 0.94, "ResNet": 0.98, "DINO CLS": 0.96, "DINO Patch (Ours)": 0.98},
    "Wall":  {"R3M": 0.34, "ResNet": 0.12, "DINO CLS": 0.58, "DINO Patch (Ours)": 0.96},
    "Reach": {"R3M": 0.40, "ResNet": 0.06, "DINO CLS": 0.60, "DINO Patch (Ours)": 0.92},
    "PushT": {"R3M": 0.42, "ResNet": 0.20, "DINO CLS": 0.44, "DINO Patch (Ours)": 0.90},
}
T2_CD = {
    "Rope":     {"R3M": 1.13, "ResNet": 1.08, "DINO CLS": 0.84, "DINO Patch (Ours)": 0.41},
    "Granular": {"R3M": 0.95, "ResNet": 0.90, "DINO CLS": 0.79, "DINO Patch (Ours)": 0.26},
}

# ---- 논문 v2 Table 3 (분포 밖 구성, SR n=50, CD n=10 가정) ----------------
T3_SR = {
    "WallRandom": {"IRIS": 0.06, "DreamerV3": 0.76, "R3M": 0.40, "ResNet": 0.40, "DINO CLS": 0.64, "Ours": 0.82},
    "PushObj":    {"IRIS": 0.14, "DreamerV3": 0.18, "R3M": 0.16, "ResNet": 0.14, "DINO CLS": 0.18, "Ours": 0.34},
}
T3_CD = {
    "GranularRandom": {"IRIS": 0.86, "DreamerV3": 1.53, "R3M": 1.12, "ResNet": 0.98, "DINO CLS": 1.36, "Ours": 0.63},
}

# ---- 논문 v2 Table 4 (LPIPS, 낮을수록 좋음) --------------------------------
T4_LPIPS = {
    "PushT":    {"R3M": 0.045, "ResNet": 0.063, "DINO CLS": 0.039, "AVDC": 0.046, "Ours": 0.007},
    "Wall":     {"R3M": 0.008, "ResNet": 0.002, "DINO CLS": 0.004, "AVDC": 0.030, "Ours": 0.0016},
    "Rope":     {"R3M": 0.023, "ResNet": 0.025, "DINO CLS": 0.029, "AVDC": 0.060, "Ours": 0.009},
    "Granular": {"R3M": 0.080, "ResNet": 0.080, "DINO CLS": 0.086, "AVDC": 0.106, "Ours": 0.035},
}

# ---- 논문 v2 Table 5 (PushT 데이터 크기, SR n=50 가정) --------------------
T5 = {"n=200": 0.08, "n=1000": 0.48, "n=5000": 0.72, "n=10000": 0.88, "n=18500": 0.92}

# ---- 논문 v2 Table 7 (PushT 디코더 손실 절제) -----------------------------
T7 = {"w/o decoder loss": 0.92, "with decoder loss": 0.80}


def log_comb(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_two_sided(a, b, c, d):
    """2x2 표 [[a,b],[c,d]]의 양측 Fisher 정확 검정 p값."""
    n1, n2 = a + b, c + d
    m = a + c
    total = n1 + n2
    lo, hi = max(0, m - n2), min(n1, m)

    def logp(x):
        return log_comb(n1, x) + log_comb(n2, m - x) - log_comb(total, m)

    p_obs = math.exp(logp(a))
    p = 0.0
    for x in range(lo, hi + 1):
        px = math.exp(logp(x))
        if px <= p_obs * (1 + 1e-7):
            p += px
    return min(1.0, p)


def welch_p(m1, s1, n1, m2, s2, n2):
    """Welch t 검정의 양측 p (정규 근사). SD가 있을 때만 쓰며, 여기서는 쓰지 않음."""
    se = math.sqrt(s1 * s1 / n1 + s2 * s2 / n2)
    t = (m1 - m2) / se
    return math.erfc(abs(t) / math.sqrt(2))


def wilson(k, n, z=1.959964):
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def main():
    out = {}

    # 1) 시도 수 복원: SR 값이 k/n의 정수 배인지
    grid = {}
    for env, row in {**T1_SR, **{k: v for k, v in T2_SR.items()}, **T3_SR}.items():
        pass
    checks = []
    for table in (T1_SR, T2_SR, T3_SR):
        for env, row in table.items():
            for m, v in row.items():
                k = round(v * T1_SR_N)
                ok = abs(k / T1_SR_N - v) < 1e-9
                checks.append({"env": env, "method": m, "sr": v, "k_of_50": k, "integer_on_grid": ok})
    out["sr_grid_check"] = {
        "n": T1_SR_N,
        "all_on_grid": all(c["integer_on_grid"] for c in checks),
        "n_values": len(checks),
    }

    # 2) Table 1: Ours 대 각 기준선 (Fisher 양측), 환경별 Wilson 95% CI
    t1 = {}
    for env, row in T1_SR.items():
        ko = round(row["Ours"] * T1_SR_N)
        entry = {"Ours_k": ko, "Ours_CI95": [round(x, 3) for x in wilson(ko, T1_SR_N)], "vs": {}}
        for m, v in row.items():
            if m == "Ours":
                continue
            k = round(v * T1_SR_N)
            entry["vs"][m] = {
                "k": k,
                "CI95": [round(x, 3) for x in wilson(k, T1_SR_N)],
                "fisher_p": round(fisher_two_sided(ko, T1_SR_N - ko, k, T1_SR_N - k), 4),
                "diff": round(row["Ours"] - v, 3),
            }
        t1[env] = entry
    out["table1_sr"] = t1

    # 3) Table 2 인코더 비교 (Ours = DINO Patch, Fisher 대 DINO CLS)
    t2 = {}
    for env, row in T2_SR.items():
        ko = round(row["DINO Patch (Ours)"] * T1_SR_N)
        dc = round(row["DINO CLS"] * T1_SR_N)
        t2[env] = {
            "Ours_vs_DINO_CLS_fisher_p": round(fisher_two_sided(ko, T1_SR_N - ko, dc, T1_SR_N - dc), 4),
            "Ours_k": ko, "DINO_CLS_k": dc,
        }
    out["table2_sr_vs_dinocls"] = t2

    # 4) Table 3 분포 밖 구성: Ours 대 최선 기준선 (Fisher)
    t3 = {}
    for env, row in T3_SR.items():
        ko = round(row["Ours"] * T1_SR_N)
        best_m = max((m for m in row if m != "Ours"), key=lambda m: row[m])
        kb = round(row[best_m] * T1_SR_N)
        t3[env] = {
            "best_baseline": best_m, "best_k": kb, "Ours_k": ko,
            "fisher_p_vs_best": round(fisher_two_sided(ko, T1_SR_N - ko, kb, T1_SR_N - kb), 4),
            "Ours_CI95": [round(x, 3) for x in wilson(ko, T1_SR_N)],
        }
    out["table3_generalization_sr"] = t3

    # 5) 상대 개선 계산 (초록의 "45% on average"와 "56%" 검증)
    rel = {}
    for env, row in T1_SR.items():
        best_base = max(v for m, v in row.items() if m != "Ours")
        rel[f"T1 SR {env}"] = round((row["Ours"] - best_base) / best_base * 100, 1) if best_base > 0 else None
    for env, row in T1_CD.items():
        best_base = min(v for m, v in row.items() if m != "Ours")
        rel[f"T1 CD reduction {env} (vs best baseline)"] = round((best_base - row["Ours"]) / best_base * 100, 1)
    for env, row in T4_LPIPS.items():
        best_base = min(v for m, v in row.items() if m != "Ours")
        rel[f"T4 LPIPS reduction {env}"] = round((best_base - row["Ours"]) / best_base * 100, 1)
    rel["T3 CD reduction GranularRandom"] = round(
        (min(v for m, v in T3_CD["GranularRandom"].items() if m != "Ours") - T3_CD["GranularRandom"]["Ours"])
        / min(v for m, v in T3_CD["GranularRandom"].items() if m != "Ours") * 100, 1)
    out["relative_improvements_pct"] = rel
    # "hardest" 정의 후보별 평균 (논문은 정의를 적지 않음)
    candidates = {
        "Rope+Granular CD reduction (T1)": [rel["T1 CD reduction Rope (vs best baseline)"], rel["T1 CD reduction Granular (vs best baseline)"]],
        "Reach+PushT SR gain (T1)": [rel["T1 SR Reach"], rel["T1 SR PushT"]],
        "Granular LPIPS reduction (T4) only": [rel["T4 LPIPS reduction Granular"]],
    }
    out["hardest_definition_candidates_pct"] = {
        k: {"values": v, "mean": round(sum(v) / len(v), 1)} for k, v in candidates.items()
    }

    # 6) 데이터 크기 스케일링 (Table 5): 단조성과 Table 1 PushT와의 간격
    keys = list(T5.keys())
    vals = [T5[k] for k in keys]
    out["table5_monotone_nondecreasing"] = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
    k18500 = round(T5["n=18500"] * T1_SR_N)
    k_t1 = round(T1_SR["PushT"]["Ours"] * T1_SR_N)
    out["table5_vs_table1_pushT"] = {
        "n18500_k": k18500, "table1_k": k_t1,
        "fisher_p": round(fisher_two_sided(k18500, T1_SR_N - k18500, k_t1, T1_SR_N - k_t1), 4),
        "note": "동일 설정이라면 같은 값이어야 하지만 0.92 대 0.90 (Table 1)",
    }
    k_wo = round(T7["w/o decoder loss"] * T1_SR_N)
    out["table7_vs_table1_pushT"] = {
        "w_o_decoder_loss_k": k_wo, "table1_k": k_t1,
        "fisher_p": round(fisher_two_sided(k_wo, T1_SR_N - k_wo, k_t1, T1_SR_N - k_t1), 4),
    }

    # 7) Table 6 (인과 마스크 절제) 셀 배치 추정: 행 w/o mask, with mask; 열 h=1,2,3
    # 텍스트상 h=1에서 둘은 같아야 하므로 0.76이 두 행 모두에 해당.
    out["table6_layout_inference"] = {
        "w_o_mask": {"h1": 0.76, "h2": 0.36, "h3": 0.08},
        "with_mask": {"h1": 0.76, "h2": 0.88, "h3": 0.92},
        "note": "PDF 추출 순서로 추정한 배치. 원문 그림으로 확인 필요",
    }

    # 8) CD 변동 검정 불가 확인: 표준편차가 보고되지 않음
    out["cd_testable"] = False
    out["cd_note"] = "CD는 n=10 평균만 보고되고 SD가 없어 Welch 검정 불가 (논문 전체에 SD/오차막대 없음)"

    # 9) 임의 시드 1개 가정의 한계: 논문에 시드 반복이 보고되지 않음
    out["seed_repeats_reported"] = False

    with open(os.path.join(HERE, "recompute_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 콘솔 요약
    print("SR grid (n=50) all on grid:", out["sr_grid_check"]["all_on_grid"], out["sr_grid_check"]["n_values"], "values")
    for env, e in t1.items():
        print(env, "Ours", e["Ours_k"], "/50 CI", e["Ours_CI95"])
        for m, r in e["vs"].items():
            print("   vs", m, "diff", r["diff"], "fisher p", r["fisher_p"])
    print("Table2 Ours vs DINO CLS:", {k: v["Ours_vs_DINO_CLS_fisher_p"] for k, v in t2.items()})
    print("Table3:", {k: (v["best_baseline"], v["fisher_p_vs_best"]) for k, v in t3.items()})
    print("relative:", rel)
    print("hardest candidates:", out["hardest_definition_candidates_pct"])
    print("table5 monotone:", out["table5_monotone_nondecreasing"], "vs table1:", out["table5_vs_table1_pushT"])
    print("table7 vs table1:", out["table7_vs_table1_pushT"])


if __name__ == "__main__":
    main()
