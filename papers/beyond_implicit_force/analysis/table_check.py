"""Table IV·본문 수치 재계산 (CPU, 단일 스레드). 실행: python3 analysis/table_check.py

- C1: 표 내부 일관성 (Overall = 두 stiffness 모드 평균, n_total 산술, 성공 수가 정수인지)
- C2: 변형 쌍별 Fisher 정확검정 + Holm 보정, 과제 합산 Cochran–Mantel–Haenszel(MH) 검정
- C3: Soft Bottle Press를 판별 문제로 다시 보기 (항상 누르는 정책 = 50%, Youden J)
- C4: Plug Insertion의 두 번째 시도 허용이 변형별로 얼마나 점수를 올리는지
- C5: 하이퍼파라미터를 ACT 공식 기본값(README)과 비교, chunk 길이(초), temporal ensembling 가중의 평균 명령 나이
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
V = ["Base ACT", "ACT + Torque", "ACT-o", "ACT-o + Torque"]

# Table IV (성공률 %, 시행 수)
BOTTLE = {"Base ACT": (100, 0), "ACT + Torque": (60, 80), "ACT-o": (50, 0), "ACT-o + Torque": (80, 70)}  # (low, high), 각 n=10
BOTTLE_OVERALL = {"Base ACT": 50, "ACT + Torque": 70, "ACT-o": 25, "ACT-o + Torque": 75}
PLUG = {"Base ACT": (20, 30), "ACT + Torque": (40, 70), "ACT-o": (0, 0), "ACT-o + Torque": (10, 50)}  # (1-step, 1 or 2), n=10
WIPE = {"Base ACT": 60, "ACT + Torque": 90, "ACT-o": 15, "ACT-o + Torque": 95}  # n=20


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def mh_test(tables):
    """2x2 표 목록 [(a, b, c, d)] = [(k1, n1-k1, k2, n2-k2)]에 대한 MH 검정(연속성 보정 없음)과 공통 OR."""
    num = den = var = 0.0
    or_n = or_d = 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        if n <= 1:
            continue
        r1, c1 = a + b, a + c
        num += a - r1 * c1 / n
        var += r1 * (c + d) * c1 * (b + d) / (n * n * (n - 1))
        or_n += a * d / n
        or_d += b * c / n
    chi2 = num * num / var if var > 0 else float("nan")
    return {"chi2": round(chi2, 3), "p": float(f"{stats.chi2.sf(chi2, 1):.2e}"),
            "common_OR": (round(or_n / or_d, 2) if or_d > 0 else "inf")}


def counts():
    """과제별 (성공, 시행) — Bottle은 두 모드 합산 n=20, Plug은 '1 OR 2-Steps' 주 지표 n=10, Wipe n=20."""
    out = {}
    for v in V:
        lo, hi = BOTTLE[v]
        out[v] = {"bottle": (lo // 10 + hi // 10, 20), "bottle_low": (lo // 10, 10), "bottle_high": (hi // 10, 10),
                  "plug_1step": (PLUG[v][0] // 10, 10), "plug": (PLUG[v][1] // 10, 10),
                  "wipe": (round(WIPE[v] * 20 / 100), 20)}
    return out


def main():
    out = {}
    C = counts()

    # C1 내부 일관성
    c1 = {}
    for v in V:
        lo, hi = BOTTLE[v]
        c1[v] = {"bottle_overall_is_mode_mean": (lo + hi) / 2 == BOTTLE_OVERALL[v],
                 "wipe_count_integer": abs(WIPE[v] * 20 / 100 - round(WIPE[v] * 20 / 100)) < 1e-9,
                 "plug_two_step_ge_one_step": PLUG[v][1] >= PLUG[v][0]}
    out["C1_internal_consistency"] = {"per_variant": c1,
                                      "n_total": {"bottle": 4 * 20, "plug": 4 * 10, "wipe": 4 * 20,
                                                  "foam_stop": 4 * 3 * 10, "paper": [80, 40, 80, 120]}}

    # C2 쌍별 검정
    comparisons = {
        "remove_cue (Base vs ACT-o)": ("Base ACT", "ACT-o"),
        "recover (ACT-o+τ vs ACT-o)": ("ACT-o + Torque", "ACT-o"),
        "improve_base (ACT+τ vs Base)": ("ACT + Torque", "Base ACT"),
        "explicit_only_vs_base (ACT-o+τ vs Base)": ("ACT-o + Torque", "Base ACT"),
        "hybrid_vs_explicit (ACT+τ vs ACT-o+τ)": ("ACT + Torque", "ACT-o + Torque"),
    }
    tasks = ["bottle", "plug", "wipe"]
    c2 = {}
    for name, (a, b) in comparisons.items():
        rows = {}
        ps = []
        for t in tasks:
            k1, n1 = C[a][t]
            k2, n2 = C[b][t]
            p = fisher(k1, n1, k2, n2)
            ps.append(p)
            rows[t] = {"a": [k1, n1], "b": [k2, n2], "fisher_p": float(f"{p:.3g}")}
        for t, padj in zip(tasks, holm(ps)):
            rows[t]["holm_p_within_comparison"] = float(f"{padj:.3g}")
        tabs = [(C[a][t][0], C[a][t][1] - C[a][t][0], C[b][t][0], C[b][t][1] - C[b][t][0]) for t in tasks]
        rows["MH_pooled_3_tasks"] = mh_test(tabs)
        c2[name] = rows
    # 고강성 모드만 따로 (논문이 강조한 판별)
    c2["bottle_high_only"] = {f"{a} vs {b}": float(f"{fisher(C[a]['bottle_high'][0], 10, C[b]['bottle_high'][0], 10):.3g}")
                              for a, b in [("ACT + Torque", "Base ACT"), ("ACT-o + Torque", "ACT-o"), ("ACT-o + Torque", "Base ACT")]}
    out["C2_pairwise_tests"] = c2
    out["C2b_wilson_95"] = {v: {t: wilson(*C[v][t]) for t in tasks} for v in V}

    # C3 Soft Bottle Press를 판별로: low에서 계속 누르고 high에서 물러나야 성공
    c3 = {}
    for v in V:
        lo, hi = (x / 100 for x in BOTTLE[v])
        c3[v] = {"overall": (lo + hi) / 2, "youden_J": round(lo + hi - 1, 2),
                 "interpretation": ("항상 누르는 정책과 같음" if hi == 0 and lo == 1 else
                                    "판별 신호 없음(J≤0)" if lo + hi - 1 <= 0 else "판별함")}
    c3["always_press_policy_overall"] = 0.5
    out["C3_bottle_as_discrimination"] = c3

    # C4 Plug 두 번째 시도
    out["C4_plug_second_attempt"] = {v: {"one_step": PLUG[v][0] / 100, "one_or_two": PLUG[v][1] / 100,
                                         "gain_from_retry_pp": PLUG[v][1] - PLUG[v][0]} for v in V}

    # C5 하이퍼파라미터
    act_default = {"kl_weight": 10, "chunk_size": 100, "hidden_dim": 512, "batch_size": 8,
                   "dim_feedforward": 3200, "num_epochs": 2000, "lr": 1e-5}
    paper = {"kl_weight": 10, "chunk_size": 100, "hidden_dim": 512, "batch_size": 16,
             "dim_feedforward": 3200, "num_epochs": 8000, "lr": 2e-5}
    hz = 50
    k = 0.01  # act/imitate_episodes.py:255
    idx = np.arange(paper["chunk_size"])  # 0 = 가장 오래된 예측(가장 큰 가중)
    w = np.exp(-k * idx)
    w /= w.sum()
    age_steps = (paper["chunk_size"] - 1) - idx  # 오래된 예측일수록 나이가 많음
    out["C5_hyperparams"] = {
        "act_readme_default": act_default, "paper_table3": paper,
        "differs": {kk: [act_default[kk], paper[kk]] for kk in act_default if act_default[kk] != paper[kk]},
        "chunk_seconds_at_50Hz": paper["chunk_size"] / hz,
        "temporal_agg_mean_command_age_ms_if_used": round(float((w * age_steps).sum()) / hz * 1000, 1),
        "note": "논문은 temporal ensembling 사용 여부를 적지 않음. 쓰지 않고 chunk를 그대로 실행하면 명령이 최대 2 s 동안 열린 루프",
    }

    path = ROOT / "analysis" / "table_check_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False)[:4000])
    print("wrote", path)


if __name__ == "__main__":
    main()
