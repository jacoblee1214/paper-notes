"""TouchTwin 논문의 표 1–3 수치를 다시 계산하고 검정한다 (수 초, numpy·scipy).

입력: 논문 본문과 표에 적힌 값만 쓴다 (paper/paper.txt의 Table 1–3, 본문 수치).
출력: analysis/recompute_results.json

내용
  V1. 표 3 성공 횟수: Wilson·Clopper-Pearson 95% 구간, 조건 간 Fisher 정확 검정 (Holm 보정)
  V2. 표 3 grip 비율(0.31, 1.55, 0.99)과 본문의 확인 가능한 수치 (27.4/20.8 등)
  V3. 표 2 접촉 누락률 32.2% 대 13.1%: 표본 크기가 보고되지 않아 검정은 못 하고,
      프레임 수에 따른 필요 표본 크기만 계산한다
  V4. 일관성 검사: 초록·서론·표·그림 캡션 사이의 숫자 대조
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "recompute_results.json"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(c - h), float(c + h)


def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def fisher(k1, n1, k2, n2):
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(stats.fisher_exact(table, alternative="two-sided").pvalue)


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj


def v1_success_tables():
    # 표 3 (논문): held-out 기록 20회, 시뮬 테스트 36회
    conds = {
        "held_out": {"pose": (17, 20), "vision": (15, 20), "touch": (20, 20)},
        "sim_test": {"pose": (26, 36), "vision": (33, 36), "touch": (35, 36)},
    }
    out = {}
    for name, d in conds.items():
        rows = {}
        for c, (k, n) in d.items():
            rows[c] = {
                "k": k, "n": n, "rate": round(k / n, 4),
                "wilson95": [round(x, 4) for x in wilson(k, n)],
                "clopper_pearson95": [round(x, 4) for x in clopper_pearson(k, n)],
            }
        pairs = [("touch", "pose"), ("touch", "vision"), ("vision", "pose")]
        raw = [fisher(d[a][0], d[a][1], d[b][0], d[b][1]) for a, b in pairs]
        adj = holm(raw)
        rows["pairwise_fisher"] = [
            {"a": a, "b": b, "diff_pp": round(100 * (d[a][0] / d[a][1] - d[b][0] / d[b][1]), 1),
             "p_raw": round(p, 4), "p_holm": round(q, 4)}
            for (a, b), p, q in zip(pairs, raw, adj)
        ]
        out[name] = rows
    return out


def power_per_group(p1, p2, za=1.959964, zb=0.841621):
    """두 비율을 비교할 때 조건당 필요한 표본 수 (양측 5%, 검정력 80%, 근사식)."""
    pbar = (p1 + p2) / 2
    num = za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return int(np.ceil(num ** 2 / (p1 - p2) ** 2))


def v1_power():
    # 표 3 비교별 필요 시도 수 (관측 비율을 참값으로 가정한 근사)
    pairs = {
        "held_out_touch_vs_vision_20_vs_15": (1.00, 0.75),
        "held_out_touch_vs_pose_20_vs_17": (1.00, 0.85),
        "sim_test_touch_vs_pose_35_vs_26": (35 / 36, 26 / 36),
        "sim_test_touch_vs_vision_35_vs_33": (35 / 36, 33 / 36),
    }
    return {k: {"p1": round(a, 3), "p2": round(b, 3), "n_per_group_80pct": power_per_group(a, b)}
            for k, (a, b) in pairs.items()}


def v2_grip_and_text():
    # 표 3의 grip (× 기록) 비율과 본문 수치
    res = {
        "grip_ratio_table3": {"pose": 0.31, "vision": 1.55, "touch": 0.99},
        "mug_holding_force": {},
    }
    # 본문: 300 g 물을 넣으면 필요한 힘 27.4 N 대 빈 머그 20.8 N (+31%)
    ratio = 27.4 / 20.8
    res["mug_holding_force"]["ratio"] = round(ratio, 4)
    res["mug_holding_force"]["increase_pct"] = round(100 * (ratio - 1), 2)
    res["mug_holding_force"]["text_claim_pct"] = 31
    # 힘이 질량에 비례한다고 가정할 때 암시되는 빈 머그 질량 (m0): (m0 + 0.3)/m0 = ratio
    m0 = 0.3 / (ratio - 1)
    res["mug_holding_force"]["implied_empty_mass_kg_if_proportional"] = round(m0, 3)
    # 같은 가정에서 필요한 힘 20.8 N이 m0 g 의 몇 배인지 (질량 0.3–0.4 kg 머그의 무게는 2.9–3.9 N)
    res["mug_holding_force"]["note"] = (
        "머그 질량은 논문에 없음. 비례 모델이면 빈 머그가 약 0.95 kg이어야 함 (일반 세라믹 머그보다 큼). "
        "동적 하중과 마찰 모델이 섞여 있을 수 있어 확정은 못 함"
    )
    return res


def v3_contact_power():
    # 표 2: 접촉 누락률 32.2% 대 13.1%. 프레임 또는 클립 수가 없어 검정 불가.
    p1, p2 = 0.322, 0.131
    za, zb = 1.959964, 0.841621  # 양측 5%, 검정력 80%
    pbar = (p1 + p2) / 2
    n_per_group = ((za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
                   / (p1 - p2) ** 2)
    res = {
        "miss_rate": {"vision": p1, "touch": p2},
        "independent_frames_needed_per_group_80pct_power": int(np.ceil(n_per_group)),
        "note": "프레임 간 상관을 무시한 값. 클립 단위 무작위 효과를 고려하면 더 큼. 논문은 n을 보고하지 않음",
    }
    # 표 2의 평균 탐지 오차 1.01 s 대 0.61 s: SD와 n이 없어 검정 불가
    res["press_duration_error"] = {"vision_s": 1.01, "touch_s": 0.61, "test": "not possible (no SD, no n)"}
    res["f1"] = {"vision": 0.778, "touch": 0.890, "test": "not possible (no n, no precision/recall)"}
    # 가정 n에 따른 위험 구간 폭 (Wilson 95%)
    widths = {}
    for n in (200, 500, 1000):
        lo, hi = wilson(int(round(p2 * n)), n)
        widths[str(n)] = {"touch_rate_wilson95": [round(lo, 3), round(hi, 3)]}
    res["illustrative_wilson_if_n_frames"] = widths
    return res


def v4_consistency():
    # 초록·서론·그림 1의 수치를 표 값과 대조한다 (값은 논문 텍스트에서 옮김)
    checks = [
        ("초록·그림 1: 0.31× 등 grip 비율 = 표 3", "0.31, 1.55, 0.99", "일치"),
        ("초록: 20/20 (held-out), 35/36 (시뮬 테스트) = 표 3", "20/20, 35/36", "일치"),
        ("초록: 13% 대 32% = 표 2 (13.1%, 32.2%)", "13.1, 32.2", "일치 (반올림)"),
        ("본문 300 g 물: 27.4/20.8 N = +31%", round(100 * (27.4 / 20.8 - 1), 2), "불일치 소폭 (계산 +31.7%, 반올림 +32%)"),
        ("프로젝트 페이지: 'held-out objects' 대 논문 'held-out recordings'", "-", "문구 불일치 (대상이 물체인지 기록인지)"),
        ("논문 날짜 November 16, 2026 대 arXiv v1 제출 13 Nov 2026", "-", "제출 뒤 날짜 (사소)"),
    ]
    return [{"check": c, "value": v, "result": r} for c, v, r in checks]


def main():
    res = {
        "v1_success": v1_success_tables(),
        "v1_power": v1_power(),
        "v2_grip_and_text": v2_grip_and_text(),
        "v3_contact_power": v3_contact_power(),
        "v4_consistency": v4_consistency(),
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
