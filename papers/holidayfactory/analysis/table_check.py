"""HolidayFactory 1.0(arXiv:2611.11789v1)의 표·본문 수치 재계산과 유의성 점검.

원자료(작업별 결과, 쌍 단위 결과)는 공개되지 않았다. 모든 검정은 논문에 보고된 집계 값에서 한다.
쌍 설계(같은 과제·같은 1차 자산을 공유)는 불일치 쌍 수를 알 수 없으므로, 실현 가능한 모든 분할에 대해
정확 McNemar p의 최솟값과 최댓값을 구간으로 보고한다.

실행: python3 analysis/table_check.py → analysis/table_check_results.json (1초 미만)
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom, fisher_exact, norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "table_check_results.json"
Z = norm.ppf(0.975)


def wilson(k, n):
    p = k / n
    den = 1 + Z**2 / n
    c = (p + Z**2 / (2 * n)) / den
    h = Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / den
    return [round(float(max(0.0, c - h)), 3), round(float(min(1.0, c + h)), 3)]


def fisher(k1, n1, k2, n2):
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(fisher_exact(table, alternative="two-sided")[1])


def mcnemar_exact(b, c):
    """불일치 쌍 b, c에 대한 정확 양측 McNemar p (이항 검정, 1/2)."""
    m = b + c
    if m == 0:
        return 1.0
    return float(min(1.0, 2 * binom.cdf(min(b, c), m, 0.5)))


def mcnemar_range(n, x, y):
    """쌍 n개에서 성공이 A는 x, B는 y개일 때 실현 가능한 (b, c) 분할 전체의 정확 McNemar p 범위.

    b = B 성공 & A 실패, c = A 성공 & B 실패, b - c = y - x.
    """
    ps = []
    for c in range(0, n + 1):
        b = c + (y - x)
        if b < 0 or c > x or b > y or b > n - x or c > n - y:
            continue
        ps.append((b, c, mcnemar_exact(b, c)))
    if not ps:
        return None
    lo = min(ps, key=lambda t: t[2])
    hi = max(ps, key=lambda t: t[2])
    return {"feasible_splits": len(ps), "p_min": round(lo[2], 5), "p_max": round(hi[2], 5),
            "b_c_at_p_min": [lo[0], lo[1]], "b_c_at_p_max": [hi[0], hi[1]]}


R = {}

# ---- Table 5: 배터리 들기. 방법당 20회(시간 간격 4개 x 5회). 주의: 파라미터는 CDMM으로 튜닝됨
R["table5_battery"] = {
    "cdmm_32iter": {"k": 20, "n": 20, "wilson95": wilson(20, 20)},
    "mujoco_15iter": {"k": 1, "n": 20, "wilson95": wilson(1, 20)},
    "mujoco_100iter": {"k": 0, "n": 20, "wilson95": wilson(0, 20)},
    "fisher_p_cdmm_vs_mujoco15": fisher(20, 20, 1, 20),
    "fisher_p_cdmm_vs_mujoco100": fisher(20, 20, 0, 20),
    "note": "파라미터(잡기 자세, 목표값)를 CDMM으로 튜닝했으므로 차이가 솔버만의 효과가 아니다.",
}

# ---- Table 10: 시각 형상 충실도 (VLM 판정 60건 = 자산 30개 x 2회 재배열). 자산 단위 독립성 가정이 필요하다.
R["table10_visual"] = {
    "mesh_detailed_realistic": {"k": 48, "n": 60, "wilson95": wilson(48, 60), "effective_n_assets": 30},
    "code_native_detailed_realistic": {"k": 0, "n": 60, "wilson95": wilson(0, 60)},
}

# ---- Table 11, 12: 검증 탈락과 콜라이더 재작성 비율
R["table11_verification"] = {
    "any_stage": {"k": 34, "n": 72, "pct": round(34 / 72 * 100, 1)},
    "conformance": {"k": 28, "n": 72, "pct": round(28 / 72 * 100, 1), "unique": 24},
    "collider": {"k": 10, "n": 72, "pct": round(10 / 72 * 100, 1), "unique": 4},
    "articulation": {"k": 2, "n": 72, "pct": round(2 / 72 * 100, 1), "unique": 0},
    "interaction": {"k": 2, "n": 72, "pct": round(2 / 72 * 100, 1), "unique": 0},
    "unique_sum_check": "unique 합 = 24 + 4 = 28, 탈락 34 중 6개는 두 단계 이상에서 탈락",
}
R["table12_recollider"] = {
    "colliders": {"k": 346, "n": 1597, "pct": round(346 / 1597 * 100, 1), "wilson95": wilson(346, 1597)},
    "assets": {"k": 39, "n": 72, "pct": round(39 / 72 * 100, 1)},
}

# ---- Table 13: 수정 피드백 비교 (같은 1차 자산 32개, 쌍 설계). 1차 실패 12개가 이후 결과를 좌우한다.
#      성공 수 (1차 20/32는 세 전략 공통). 독립 재생성 27/32, 거친 피드백 26/32, E RIS 32/32 (Success@3).
R["table13_revision_paired"] = {
    "first_round_success": {"k": 20, "n": 32, "pct": round(20 / 32 * 100, 1)},
    "success_at3": {
        "independent": {"k": 27, "n": 32, "wilson95": wilson(27, 32)},
        "coarse": {"k": 26, "n": 32, "wilson95": wilson(26, 32)},
        "eris": {"k": 32, "n": 32, "wilson95": wilson(32, 32)},
    },
    "mcnemar_eris_vs_independent": mcnemar_range(32, 27, 32),
    "mcnemar_eris_vs_coarse": mcnemar_range(32, 26, 32),
    "mcnemar_independent_vs_coarse": mcnemar_range(32, 27, 26),
    "recovery_at1": {"independent": "4/12", "coarse": "2/12", "eris": "11/12"},
    "fisher_recovery_at1_eris_11_12_vs_independent_4_12": round(fisher(11, 12, 4, 12), 4),
    "consistency": "Success@2, @3 값은 1차 20/32 + Recovery 분자로 모두 재현됨(0.75, 0.84 / 0.69, 0.81 / 0.97, 1.00)",
}

# ---- Table 14: 백본 5개에서 E RIS와 독립 재생성의 쌍 비교 (40쌍, 20과제 x 2시드; 과제 단위 군집 미반영)
t14 = {
    "claude-opus-5-5 (high)": (26, 40, 40),
    "claude-fable-5-1": (28, 38, 40),
    "gpt-6-astra": (26, 38, 40),
    "gpt-6.1-sol": (23, 37, 40),
    "gpt-6-luna": (17, 30, 40),
}
R["table14_backbones"] = {}
for name, (raw, eris, n) in t14.items():
    R["table14_backbones"][name] = {
        "raw_k": raw, "eris_k": eris, "n": n,
        "delta_pp": round((eris - raw) / n * 100, 1),
        "mcnemar_range": mcnemar_range(n, raw, eris),
    }

# ---- Table 15: 생성 효율. 1차 시도 횟수, 비용 대비 검산
R["table15_efficiency"] = {
    "one_shot_rounds_total": round(2.22 * 18, 1),
    "one_shot_check": "1회 시도 x 40단위 = 40 라운드 예상. 2.22 x 18 = 39.96으로 일치",
    "independent_rounds_total": round(3.81 * 26, 1),
    "eris_rounds_total": round(1.70 * 40, 1),
    "cost_per_round_usd": {"one_shot": round(2.59 / 2.22, 2), "independent": round(5.00 / 3.81, 2),
                           "eris": round(2.48 / 1.70, 2)},
    "eris_over_independent_cost_per_solved": round(2.48 / 5.00, 3),
    "reading": "E RIS의 해결당 비용 절감은 라운드 수 감소(3.81 -> 1.70)에서 나온다. 라운드당 비용은 오히려 늘었다.",
}

# ---- Figure 19 / Table 43: single tote transfer 평가 생성. 모델 4 x 3시행 = 12. 같은 모델의 두 조건은 쌍이지만 집계만 있다.
R["fig19_eval_generation"] = {
    "accept_direct_no_feedback": {"k": 6, "n": 12},
    "accept_direct_feedback": {"k": 12, "n": 12},
    "accept_questions_no_feedback": {"k": 10, "n": 12},
    "accept_questions_feedback": {"k": 11, "n": 12},
    "fisher_p_direct_nofb_vs_fb": round(fisher(6, 12, 12, 12), 4),
    "fisher_p_direct_nofb_vs_questions_nofb": round(fisher(6, 12, 10, 12), 4),
    "caveat": "시행 3회, 과제 1개, 기록 15개. 시행은 독립 무작위 시드가 아니라 반복 호출이다(E.3).",
}

# ---- Figure 14, Table 17/18: 생성 실험의 작은 표본 검정
R["fig14_isg_feedback"] = {
    "gen_then_verify_5_of_18_vs_feedback_16_of_18": round(fisher(5, 18, 16, 18), 6),
    "scene_only_1_of_10_vs_full_isg_5_of_10": round(fisher(1, 10, 5, 10), 4),
    "spec_gen_first_attempt_4_of_54_vs_5attempts_40_of_54": round(fisher(4, 54, 40, 54), 8),
}
R["table17_own_critic"] = {
    "gpt-6.1-sol_scene_accept_1_of_10_vs_claude-sonnet_7_of_10": round(fisher(1, 10, 7, 10), 4),
    "shared_opus_critic_3_of_10_vs_3_of_10": 1.0,
}

# ---- 서론 주장: 독립 가정 하의 연속 성공 확률
R["intro_claim_c20"] = {
    "p95_20_jobs": round(0.95 ** 20, 4),
    "p99_20_jobs": round(0.99 ** 20, 4),
    "paper_says": "95% -> 36%, 99% -> 82%",
}

# ---- Table 45: 국가별 로봇 설치 (2025). 합계는 5개 지역
inst = {"China": 354000, "EU": 60500, "Japan": 36200, "South Korea": 30000, "United States": 38400}
R["table45_installs"] = {"sum_5_regions": sum(inst.values()),
                         "note": "서론의 '2025년 600,000대 이상 설치'는 본문 IFR 인용이며 이 표 5개 지역 합(519,100)과 직접 비교되지 않는다."}

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2))
print("wrote", OUT)
