"""Token Steering (arXiv:2606.15021) 보고 수치 재계산.

검사 항목(번호는 03_analysis.md와 같다)
  V1  표 1 SIR(0.38, 1.00, 1.00, 0.94)과 그림 3b(88%, 14%)가 어떤 시도 수 n에서 정수 비율로 가능한가
  V2  표 2 성공률(10.0, 72.5, 16.7, 93.8)이 어떤 시도 수에서 가능한가 (스펀지 교환은 단계 진행률 k/(3n))
  V3  드로어 성공률 p=0.003, 완료 시간 p=0.020이 어떤 가정에서 나오는가 (Fisher, 순위합)
  V4  Wilson 95% 구간 (언어 따르기 10회, 드로어 가정 시도 수, 표 1·그림 3b 가정 시도 수)
  V5  조작 토큰 비율: 시퀀스 길이 8–36에서 창 w가 차지하는 비율, "25% 미만" 주장 검사
  V6  경로 효율 정의(부록 B.2 시작-끝 거리) 대 본문 4.1(시작-블록 거리) 수치 예시

실행: python3 analysis/recompute.py  (numpy, scipy, 수 초)
결과: analysis/recompute_results.json
"""
import json
from math import comb
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_results.json"
res = {}


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def feasible(target, decimals, nmax, denom_factor=1):
    """k/(denom_factor*n)이 target(소수 decimals 자리 반올림)과 같아지는 (n, k) 목록."""
    out = []
    for n in range(1, nmax + 1):
        d = denom_factor * n
        for k in range(0, d + 1):
            if round(k / d * 100, decimals) == target or abs(round(k / d, decimals + 2) - round(target / 100, decimals + 2)) < 1e-12:
                out.append((n, k))
    return out


# ---------------------------------------------------------------- V1
def v1():
    rows = {"SIR_w1": 0.38, "SIR_w2": 1.00, "SIR_w4": 1.00, "SIR_w6": 0.94}
    out = {}
    for name, v in rows.items():
        ok = [n for n in range(1, 41) if any(round(k / n, 2) == v for k in range(n + 1))]
        out[name] = ok
    common = [n for n in range(1, 41) if all(n in out[k] for k in out)]
    out["table1_common_n_upto40"] = common
    # 그림 3b: 100%, 88%, 88%, 14% (논문), 12% (프로젝트 페이지)
    f3 = {"88%": 88, "14%": 14, "12%": 12, "100%": 100}
    f3_ok = {}
    for name, v in f3.items():
        f3_ok[name] = [n for n in range(1, 61) if any(round(k / n * 100) == v for k in range(n + 1))]
    out["fig3b_feasible_n_upto60"] = f3_ok
    out["fig3b_common_n_88_14_upto60"] = [n for n in range(1, 61) if n in f3_ok["88%"] and n in f3_ok["14%"]]
    out["fig3b_common_n_88_12_upto60"] = [n for n in range(1, 61) if n in f3_ok["88%"] and n in f3_ok["12%"]]
    # 저자 공개 노트북(paper-analyses/expt1-figures.ipynb): 창 크기당 CSV 8개
    out["fig3b_b3_14pct_possible_with_n16"] = 14 in [round(k / 16 * 100) for k in range(17)]
    out["fig3b_b3_possible_values_n16"] = [round(k / 16 * 100, 1) for k in range(17)][:5]
    # w=1 대 w=2 Fisher (n=8, n=16 가정)
    out["fisher_w1_vs_w2"] = {
        "n8_3of8_vs_8of8": float(stats.fisher_exact([[3, 5], [8, 0]])[1]),
        "n16_6of16_vs_16of16": float(stats.fisher_exact([[6, 10], [16, 0]])[1]),
        "n16_w2_16of16_vs_w6_15of16": float(stats.fisher_exact([[16, 0], [15, 1]])[1]),
    }
    out["notebook_trials_per_window"] = 8
    out["SIR_values_possible_with_n8"] = [round(k / 8, 2) for k in range(9)]
    out["w6_0.94_possible_with_n8"] = 0.94 in out["SIR_values_possible_with_n8"]
    return out


# ---------------------------------------------------------------- V2
def v2():
    out = {}
    for name, v in {"drawer_auto": 10.0, "drawer_steer": 72.5}.items():
        ok = [(n, k) for n in range(1, 81) for k in range(n + 1) if round(k / n * 100, 1) == v]
        out[name] = {"n_list_upto80": sorted({n for n, _ in ok})}
    for name, v in {"swap_auto_progress": 16.7, "swap_steer_progress": 93.8}.items():
        # 평균 진행률 = S/(3n), S는 단계 합(0..3n)
        ok = [(n, s) for n in range(1, 61) for s in range(3 * n + 1) if round(s / (3 * n) * 100, 1) == v]
        out[name] = {"n_list_upto60": sorted({n for n, _ in ok}), "examples": ok[:6]}
    # 사용자 7명이 같은 횟수 m씩 했다면 합계 시도 7m
    out["drawer_72.5_requires_n_multiple_of_40_so_7m_needs"] = [m for m in range(1, 200) if round(29 * 7 * m / 40, 6) == int(29 * 7 * m / 40) and (7 * m) % 40 == 0][:3]
    return out


# ---------------------------------------------------------------- V3
def mwu_binary(k0, n0, k1, n1):
    a = np.r_[np.ones(k0), np.zeros(n0 - k0)]
    b = np.r_[np.ones(k1), np.zeros(n1 - k1)]
    r = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
    return float(r.pvalue)


def exact_ranksum_min_p(n1, n2):
    return 2.0 / comb(n1 + n2, n1)


def asym_ranksum_extreme_p(n1, n2):
    """한 집단이 전부 다른 집단보다 큰 완전 분리일 때 정규근사(동점 없음, 연속성 보정 없음) 양측 p."""
    u = n1 * n2
    mu = n1 * n2 / 2
    sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u - mu) / sd
    return float(2 * (1 - stats.norm.cdf(z)))


def v3():
    out = {"drawer_success_scenarios": []}
    for n0 in (10, 20, 30, 40):
        k0 = round(0.10 * n0)
        for n1, k1 in ((40, 29), (80, 58)):
            _, pf = stats.fisher_exact([[k0, n0 - k0], [k1, n1 - k1]])
            out["drawer_success_scenarios"].append({
                "auto": f"{k0}/{n0}", "steer": f"{k1}/{n1}",
                "fisher_p": float(pf), "mannwhitney_asym_p": mwu_binary(k0, n0, k1, n1)})
    out["reported_p_drawer_success"] = 0.003
    out["reported_p_drawer_time"] = 0.020
    out["reported_p_swap"] = "<0.001"
    # 시간 검정: 자율 성공 시행 수 a, 조향 성공 29(=72.5% of 40 가정)
    out["time_test_attainable"] = []
    for a in (1, 2, 3, 4):
        out["time_test_attainable"].append({
            "auto_successes": a, "steer_successes": 29,
            "exact_min_two_sided_p": exact_ranksum_min_p(a, 29),
            "asymptotic_extreme_p": asym_ranksum_extreme_p(a, 29)})
    # 7명 대 7명 사용자 단위 순위합의 정확 최소 p
    out["user_level_7v7_exact_min_p"] = exact_ranksum_min_p(7, 7)
    # 스펀지: 진행률 0, 1/3, 2/3, 1 값 시행 16개씩 가정한 예시 — 자율 평균 1/6, 조향 평균 15/16
    rng = np.random.default_rng(0)
    # 자율: 16회, 단계 합 8 (예: 8회는 1단계, 8회는 0단계)
    auto = np.array([1 / 3] * 8 + [0.0] * 8)
    # 조향: 16회, 단계 합 45 (15회 3단계, 1회 0단계)
    steer = np.array([1.0] * 15 + [0.0])
    r = stats.mannwhitneyu(auto, steer, alternative="two-sided", method="asymptotic")
    out["swap_example_16v16"] = {"auto_mean": float(auto.mean()), "steer_mean": float(steer.mean()),
                                 "mannwhitney_asym_p": float(r.pvalue)}
    return out


# ---------------------------------------------------------------- V4
def v4():
    cases = {
        "lang_green_cube_before_0of10": (0, 10), "lang_green_cube_after_10of10": (10, 10),
        "lang_bluerect_before_0of10": (0, 10), "lang_bluerect_after_10of10": (10, 10),
        "drawer_auto_4of40": (4, 40), "drawer_auto_1of10": (1, 10), "drawer_steer_29of40": (29, 40),
        "table1_w1_3of8": (3, 8), "table1_w1_6of16": (6, 16),
        "table1_w2_8of8": (8, 8), "table1_w6_15of16": (15, 16),
        "fig3b_b1_7of8": (7, 8), "fig3b_b3_1of7": (1, 7), "fig3b_b3_1of8": (1, 8),
    }
    out = {}
    for name, (k, n) in cases.items():
        lo, hi = wilson(k, n)
        out[name] = {"rate": k / n, "wilson95": [round(lo, 3), round(hi, 3)]}
    return out


# ---------------------------------------------------------------- V5
def v5():
    out = {"seq_len_range": [8, 36]}
    out["window_fraction"] = {}
    for w in (1, 2, 4, 6):
        out["window_fraction"][f"w{w}"] = {
            "min_if_len36": round(w / 36, 3), "max_if_len8": round(min(1.0, w / 8), 3)}
    out["w2_fraction_range"] = [round(2 / 36, 3), round(2 / 8, 3)]
    out["claim_less_than_25pct_trivially_true_if_unit_is_tokens_and_w2"] = 2 / 8 <= 0.25
    out["fig5c_reported_means"] = {"green_cube": 0.21, "blue_rect": 0.26}
    out["fig5c_blue_exceeds_25pct"] = 0.26 > 0.25
    return out


# ---------------------------------------------------------------- V6
def v6():
    """블록을 집어 그릇으로 돌아오는 궤적의 두 효율 정의 비교(손으로 만든 예시)."""
    # 그릇 위(0,0) 출발 → 블록(0.2,0)까지 직선 → 같은 길로 그릇 위로 복귀
    go = np.c_[np.linspace(0, 0.2, 50), np.zeros(50)]
    back = go[::-1]
    traj = np.vstack([go, back])
    path = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
    d_start_end = np.linalg.norm(traj[-1] - traj[0])
    d_start_block = np.linalg.norm(go[-1] - go[0])
    return {"path_length_m": float(path), "start_to_end_m": float(d_start_end),
            "start_to_block_m": float(d_start_block),
            "eff_B2_start_end_over_path": float(d_start_end / path),
            "eff_sec41_start_block_over_path": float(d_start_block / path),
            "note": "복귀 궤적에서 B.2 정의는 0, 본문 4.1 정의는 0.5. 보고된 MPE 0.56–0.72는 B.2 정의로는 나오기 어렵다."}


if __name__ == "__main__":
    res["V1_sir_feasibility"] = v1()
    res["V2_table2_feasibility"] = v2()
    res["V3_significance"] = v3()
    res["V4_wilson"] = v4()
    res["V5_token_fraction"] = v5()
    res["V6_path_efficiency_definition"] = v6()
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1)[:6000])
