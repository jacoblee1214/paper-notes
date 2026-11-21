"""CoRA-VTLA (arXiv:2611.23963 v1) 논문 표 수치 재계산과 검정.

입력: 논문 본문과 표 1–3, 표 2에 적힌 값 (paper/paper.txt의 pdftotext 결과에서 옮김).
  - MuJoCo: 과제당 50회 시행, 3과제, 합계 150회 (논문 4.1절, 4.2절).
  - 실세계: 과제당 50회 시행, 3과제, 합계 150회 (논문 4.1절).
  - UniVTAC Insert Tube: 100개 고정 장면 (논문 4.2절).
  논문은 시드 반복을 보고하지 않습니다. 모든 검정은 한 번 학습한 정책의 시행 단위로 계산합니다.
  성공 횟수는 표의 백분율에 시행 수를 곱해 얻습니다. 모든 값이 정수로 떨어지는지 확인합니다.

실행: python3 analysis/recompute.py  -> analysis/recompute_results.json
"""
import json
import math
from pathlib import Path

from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
Z = 1.959964

MUJOCO_TASKS = ["Nail", "Plug", "Knob"]
MUJOCO_N = 50  # 과제당
REAL_TASKS = ["Cup", "Stamp", "Tube"]
REAL_N = 50  # 과제당
UNIVTAC_N = 100

# ---- 표 1 (MuJoCo, 과제별 %, UniVTAC %). 논문 Table 1 ----
T1 = {
    "Base": {"mj": [74, 0, 20], "mj_avg": 31.33, "univtac": 46},
    "Direct": {"mj": [68, 32, 50], "mj_avg": 50.00, "univtac": 51},
    "FTP-pi0.5": {"mj": [56, 4, 72], "mj_avg": 44.00, "univtac": 57},
    "CoRA-VTLA": {"mj": [78, 76, 66], "mj_avg": 73.33, "univtac": 67},
}

# ---- 표 2 (실세계, 과제별 %). 논문 Table 2 ----
T2 = {
    "Base": {"real": [38, 30, 30], "avg": 32.7},
    "Direct": {"real": [50, 48, 34], "avg": 44.0},
    "FTP-pi0.5": {"real": [46, 44, 44], "avg": 44.7},
    "CoRA-VTLA": {"real": [68, 76, 56], "avg": 66.7},
}

# ---- 표 3 (MuJoCo 절제, 과제별 %). 논문 Table 3 ----
T3 = {
    "CoRA-VTLA (full)": {"mj": [78, 76, 66], "avg": 73.33},
    "+ Command & Motion": {"mj": [76, 68, 60], "avg": 68.00},
    "w/o Memory Token": {"mj": [76, 48, 38], "avg": 54.00},
    "w/o Response Pretraining": {"mj": [76, 34, 52], "avg": 54.00},
    "w/o Gaze": {"mj": [26, 70, 52], "avg": 49.33},
    "w/o Semantic Branch": {"mj": [62, 54, 32], "avg": 49.33},
}


def to_counts(pcts, n):
    counts = []
    for v in pcts:
        k = v * n / 100
        assert abs(k - round(k)) < 1e-9, (v, n)  # 정수 횟수로 떨어져야 함
        counts.append(int(round(k)))
    return counts


def wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


def wald(k, n):
    p = k / n
    h = Z * math.sqrt(p * (1 - p) / n)
    return [round(100 * (p - h), 2), round(100 * (p + h), 2)]


def newcombe_diff(k1, n1, k2, n2):
    """두 독립 비율 차이(p2 - p1)의 Newcombe score 구간 (단순 Wilson 결합)."""
    l1, u1 = (x / 100 for x in wilson(k1, n1))
    l2, u2 = (x / 100 for x in wilson(k2, n2))
    p1, p2 = k1 / n1, k2 / n2
    d = p2 - p1
    lo = d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    hi = d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return [round(100 * lo, 2), round(100 * hi, 2)]


def two_prop_z(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (k2 / n2 - k1 / n1) / se
    return round(2 * (1 - stats.norm.cdf(abs(z))), 6)


def mcnemar_paired(b, c, n):
    """b: 제안법만 성공, c: 비교법만 성공. 정확 양측 McNemar와 쌍체 Wald 구간(차이 = (b-c)/n)."""
    m = b + c
    p_exact = min(1.0, 2 * stats.binom.cdf(min(b, c), m, 0.5))
    d = (b - c) / n
    var = (b + c - (b - c) ** 2 / n) / n ** 2
    h = Z * math.sqrt(var)
    return {
        "b_only_proposed": b, "c_only_comparator": c, "diff_pp": round(100 * d, 2),
        "exact_two_sided_p": round(p_exact, 6),
        "paired_wald_ci_pp": [round(100 * (d - h), 2), round(100 * (d + h), 2)],
    }


def main():
    res = {}

    # 1. 표 1 평균 재계산
    mj_avg_check, mj_counts = {}, {}
    for name, row in T1.items():
        cnt = to_counts(row["mj"], MUJOCO_N)
        mj_counts[name] = cnt
        total = sum(cnt)
        mj_avg_check[name] = {
            "successes_of_150": total,
            "pooled_pct": round(100 * total / 150, 2),
            "reported_avg": row["mj_avg"],
            "match": abs(round(100 * total / 150, 2) - row["mj_avg"]) < 0.01,
            "wilson95_pct": wilson(total, 150),
        }
    res["mujoco_table1_avg"] = mj_avg_check

    # 2. 실세계 평균 재계산과 CI (CoRA-VTLA 66.7%의 논문 CI 확인)
    real_check = {}
    for name, row in T2.items():
        cnt = to_counts(row["real"], REAL_N)
        total = sum(cnt)
        real_check[name] = {
            "successes_of_150": total,
            "pooled_pct": round(100 * total / 150, 2),
            "reported_avg": row["avg"],
            "match": abs(round(100 * total / 150, 1) - row["avg"]) < 0.06,
            "wilson95_pct": wilson(total, 150),
            "wald95_pct": wald(total, 150),
        }
    res["real_table2_avg"] = real_check
    res["real_cora_reported_ci"] = [59.3, 74.0]

    # 3. 표 3 절제 평균
    abl = {}
    for name, row in T3.items():
        cnt = to_counts(row["mj"], MUJOCO_N)
        abl[name] = {"successes_of_150": sum(cnt), "pooled_pct": round(100 * sum(cnt) / 150, 2),
                     "reported_avg": row["avg"],
                     "match": abs(round(100 * sum(cnt) / 150, 2) - row["avg"]) < 0.01}
    res["ablation_table3_avg"] = abl

    # 4. 실패 원인 회계 (본문 4.2절): Base 96 timeout + 7 safety, CoRA 36 timeout + 4 safety
    base_fail = 150 - sum(mj_counts["Base"])
    cora_fail = 150 - sum(mj_counts["CoRA-VTLA"])
    res["failure_accounting"] = {
        "base_failures": base_fail, "base_reported_timeouts_plus_safety": 96 + 7,
        "base_match": base_fail == 96 + 7,
        "cora_failures": cora_fail, "cora_reported_timeouts_plus_safety": 36 + 4,
        "cora_match": cora_fail == 36 + 4,
    }

    # 5. MuJoCo 비교 (쌍체 자료가 없으므로 비쌍체 근사만 계산)
    cora, direct, base, ftp = (sum(mj_counts[k]) for k in ["CoRA-VTLA", "Direct", "Base", "FTP-pi0.5"])
    res["mujoco_unpaired_approx"] = {
        "cora_vs_direct_diff_pp": round(100 * (cora - direct) / 150, 2),
        "cora_vs_direct_newcombe_ci_pp": newcombe_diff(direct, 150, cora, 150),
        "cora_vs_direct_z_p_two_sided": two_prop_z(direct, 150, cora, 150),
        "reported_paired_bootstrap_ci_pp": [14.67, 32.0],
        "reported_padj": 5.1e-6,
        "note": "쌍체 불일치 횟수가 본문에 없어 정확 McNemar는 계산하지 못했습니다.",
    }
    per_task = {}
    for i, t in enumerate(MUJOCO_TASKS):
        per_task[t] = {
            "cora_vs_direct_pp": T1["CoRA-VTLA"]["mj"][i] - T1["Direct"]["mj"][i],
            "cora_vs_base_pp": round(T1["CoRA-VTLA"]["mj"][i] - T1["Base"]["mj"][i], 2),
            "fisher_cora_vs_base_p": round(stats.fisher_exact(
                [[mj_counts["CoRA-VTLA"][i], 50 - mj_counts["CoRA-VTLA"][i]],
                 [mj_counts["Base"][i], 50 - mj_counts["Base"][i]]])[1], 8),
        }
    res["mujoco_per_task"] = per_task

    # 6. UniVTAC: 논문이 보고한 쌍체 불일치 횟수 (29 / 13: 대 Direct, 36 / 15: 대 Base)
    uv_cora, uv_direct, uv_base = 67, 51, 46
    res["univtac"] = {
        "cora_minus_direct_from_counts": 29 - 13, "cora_minus_direct_from_table": uv_cora - uv_direct,
        "cora_minus_base_from_counts": 36 - 15, "cora_minus_base_from_table": uv_cora - uv_base,
        "vs_direct": mcnemar_paired(29, 13, UNIVTAC_N),
        "vs_base": mcnemar_paired(36, 15, UNIVTAC_N),
        "reported_ci_vs_direct_pp": [4, 28], "reported_padj_vs_direct": 0.0195,
    }

    # 7. 실세계 비교 (쌍체 아님, 과제 합산 150회 기준)
    rc, rf, rd, rb = (sum(to_counts(T2[k]["real"], REAL_N)) for k in ["CoRA-VTLA", "FTP-pi0.5", "Direct", "Base"])
    res["real_unpaired_approx"] = {
        "cora_vs_ftp_diff_pp": round(100 * (rc - rf) / 150, 2),
        "cora_vs_ftp_fisher_p": round(stats.fisher_exact([[rc, 150 - rc], [rf, 150 - rf]])[1], 6),
        "cora_vs_direct_fisher_p": round(stats.fisher_exact([[rc, 150 - rc], [rd, 150 - rd]])[1], 6),
        "cora_vs_base_fisher_p": round(stats.fisher_exact([[rc, 150 - rc], [rb, 150 - rb]])[1], 8),
    }

    # 8. 시간 단위 점검 (본문 3절, 4.2절)
    res["timing_checks"] = {
        "k16_events_at_20hz_s": 16 / 20, "reported_s": 0.8,
        "600_commands_at_20hz_s": 600 / 20, "reported_timeout_s": 30,
    }

    # 9. 주의: 논문이 보고한 시각 분석 값은 재계산할 수 없습니다 (원자료 없음).
    res["unverifiable_reported"] = {
        "entropy_base_vs_direct": [0.838, 0.884, "p=0.0054, episode sign-flip, 24 frames"],
        "contact_share_L17": [1.62, 1.18],
        "gaze_semantic_share_L17": [2.77, 46.65, "all 24 frames increase, p=0.00049"],
        "real_attention_mean_share": [2.1, 7.5, "8 frames"],
        "real_tube_example_share": [2.6, 12.6],
        "memory_offset_scenes_success": [8, 16, 25],
        "memory_aligned_scenes_success": [16, 22, 25],
    }
    # sign-flip 최소 p값 (모든 프레임이 증가할 때) 은 에피소드 수에 의존한다. 에피소드 수가 보고되지 않는다.

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
