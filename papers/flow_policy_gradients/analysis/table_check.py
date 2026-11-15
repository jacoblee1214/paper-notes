"""논문 본문과 부록의 보고 수치를 다시 계산하고 내부 일관성을 본다.

입력: 논문 v2 본문·부록에서 옮긴 수치(출처 절과 그림 번호를 주석으로 적었다). 그림 곡선은 값이 없어서 여기서 다루지 않는다.
계산:
  1. 평가 횟수와 보고 비율의 정합성: 보고 비율 x 평가 에피소드 수 n이 정수인가(없으면 n이 그 평가 단위가 아님)
  2. 이항 신뢰구간(Wilson 95%)과 정확 이항 검정(scipy)
  3. 10개 시드 평균 차이의 민감도: 시드 간 표준편차 s를 가정했을 때 z 값(시드별 값은 논문에 없음)
  4. 벽시계 시간 비율(23분 대 19분)과 v1·v2 문구 차이
  5. 스윕 범위 불일치(표 A.1 대 그림 캡션)
실행: python3 -I analysis/table_check.py  (scipy, 수 초)
결과: analysis/table_check_results.json
"""
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "table_check_results.json"


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def min_denominator(p_percent):
    """두 자리 백분율 값을 정확히 나타내는 가장 작은 평가 에피소드 수 n (정수 횟수)."""
    f = Fraction(p_percent).limit_denominator(100000) / 100  # 예: 73.76 -> 0.7376
    return f.denominator


def main():
    out = {}

    # 1. 부록 C.4: "base policy success rates (evaluated on 1,000 episodes)"
    base = {
        "Can FPO++ / vanilla FPO": 73.76,
        "Can ReinFlow": 76.30,
        "Can DPPO": 76.10,
        "Square DPPO": 37.60,
        "Square ReinFlow": 36.50,
        "Square FPO++ / vanilla FPO": 28.62,
    }
    rows = []
    for name, p in base.items():
        k_float = p / 100 * 1000
        lo, hi = wilson(k_float, 1000)
        rows.append({"name": name, "reported_pct": p, "count_if_n1000": k_float,
                     "is_integer_if_n1000": bool(abs(k_float - round(k_float)) < 1e-9),
                     "min_n_for_integer_count": min_denominator(p),
                     "wilson95_if_n1000": [round(lo, 4), round(hi, 4)]})
    out["base_policy_success_granularity"] = rows

    # Figure A.9 주석: 71.35%(zero), 10.00%(random), 96.11%(zero), 64.36%(random)
    a9 = {"Can base zero-sampling": 71.35, "Can base random-sampling": 10.00,
          "Can high-quality base zero-sampling": 96.11, "Can high-quality base random-sampling": 64.36}
    out["figure_a9_granularity"] = [{"name": k, "reported_pct": v, "min_n_for_integer_count": min_denominator(v)}
                                   for k, v in a9.items()]

    # 2. 절 4.4 / 부록 C.7: 10 시드 x 100 에피소드 = 1000 에피소드 (풀링)
    c7 = [("sparse, PPO trust region", 0.73), ("sparse, ASPO", 0.07),
          ("dense, PPO trust region", 0.81), ("dense, ASPO", 0.28)]
    pooled = []
    for name, p in c7:
        k = round(p * 1000)
        lo, hi = wilson(k, 1000)
        pooled.append({"name": name, "p": p, "count_of_1000": k, "wilson95": [round(lo, 4), round(hi, 4)]})
    out["c7_pooled_wilson"] = pooled

    # 민감도: 시드 평균 차이 d, 시드 간 표준편차 s 가정 -> Welch 식 근사 z = d / (s*sqrt(2/10))
    sens = []
    for name, d in [("sparse PPO - ASPO", 0.73 - 0.07), ("dense PPO - ASPO", 0.81 - 0.28)]:
        for s in [0.1, 0.2, 0.3]:
            z = d / (s * math.sqrt(2 / 10))
            p2 = 2 * (1 - stats.norm.cdf(abs(z)))
            sens.append({"comparison": name, "diff": round(d, 3), "assumed_seed_sd": s,
                         "z": round(z, 2), "p_two_sided_normal": float(p2)})
    # 같은 차이가 유의하지 않으려면 필요한 시드 간 표준편차 (z = 2 기준)
    need_sd = {name: round(d / (2 * math.sqrt(2 / 10)), 3)
               for name, d in [("sparse", 0.66), ("dense", 0.53)]}
    out["seed_sensitivity"] = {"rows": sens, "seed_sd_needed_for_z2": need_sd,
                               "note": "시드별 값이 없어 정확한 t 검정은 할 수 없다. 이 표는 가정 기반 근사다."}

    # 3. 절 4.5 / 부록 C.1: 벽시계 시간(G1 locomotion, L40S)
    out["wallclock"] = {"gaussian_ppo_min": 19, "fpo_pp_min": 23,
                        "ratio_fpo_to_gauss": round(23 / 19, 3),
                        "motion_tracking_factor_v2": 2, "motion_tracking_factor_v1": 3,
                        "note": "v1 한계 절은 '3x'(최대), v2는 '2x'로 바뀌었다. 바뀐 이유는 본문에 없다."}

    # 4. 스윕 범위 불일치
    out["sweep_mismatch"] = {
        "table_a1_clip_sweep": [0.03, 0.04, 0.05, 0.06],
        "fig2_and_fig4_caption_clip_values": [0.04, 0.05, 0.06],
        "fig2_caption_lr_sweep": ["1e-5", "1e-4", "3e-4"],
        "note": "표 A.1에는 0.03이 있으나 그림 캡션에는 없다. 그림 4 캡션은 세 값만 쓴다."}

    # 5. 표 A.1 / A.2 값 대 저장소 기본값(확인은 04 문서 참조)
    out["table_vs_repo_defaults"] = {
        "A.1 betas (0.9, 0.95)": "repo rl_cfg 기본값 (0.9, 0.999)",
        "A.2 learning rate 3e-4": "repo task_cfgs 기본값 1e-4 (motion 설정이 덮어쓰지 않음)",
        "A.2 learning epochs 5": "repo 기본값 16 (motion 설정이 덮어쓰지 않음)",
        "A.2 minibatches 4": "repo motion 설정 6",
        "A.2 flow steps 50": "repo 기본값 64 (motion 설정이 덮어쓰지 않음)",
        "A.1 samples per action 16": "repo Spot/H1/G1 설정 32",
        "B.2.2 rollout length 96": "repo motion 설정 num_steps_per_env 48",
    }


    # 6. v1 표 I (v2에서 삭제됨): 2.5분 춤 동작, 100 롤아웃, ± 는 롤아웃 간 표준편차라고 가정
    t1 = {"random_5": (34.7, 55.0), "random_50": (38.4, 26.6), "zero_5": (45.1, 27.4), "zero_50": (45.5, 23.2)}
    n = 100
    pairs = [("zero_5 - random_5", "zero_5", "random_5"), ("zero_50 - zero_5", "zero_50", "zero_5"),
             ("random_50 - random_5", "random_50", "random_5"), ("zero_5 - random_50", "zero_5", "random_50")]
    t1_rows = []
    for name, a_, b_ in pairs:
        ma, sa = t1[a_]
        mb, sb = t1[b_]
        tt = stats.ttest_ind_from_stats(ma, sa, n, mb, sb, n, equal_var=False)
        t1_rows.append({"comparison": name, "diff": round(ma - mb, 2), "welch_t": round(float(tt.statistic), 2),
                        "p_two_sided": float(tt.pvalue)})
    out["v1_table1_welch"] = {"values": t1, "n_rollouts_assumed": n, "rows": t1_rows,
                              "note": "v1 표 I만 해당. 표준편차 정의(롤아웃 간)는 논문에 없어 가정이다."}

    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in out["base_policy_success_granularity"]:
        print(f"{r['name']:<28} {r['reported_pct']:6.2f}%  count(n=1000)={r['count_if_n1000']:8.2f} "
              f"integer={r['is_integer_if_n1000']}  min_n={r['min_n_for_integer_count']}  wilson={r['wilson95_if_n1000']}")
    for r in out["figure_a9_granularity"]:
        print(f"{r['name']:<40} {r['reported_pct']:6.2f}%  min_n={r['min_n_for_integer_count']}")
    for r in pooled:
        print(f"C.7 {r['name']:<28} p={r['p']:.2f} k/1000={r['count_of_1000']} wilson={r['wilson95']}")
    for r in sens:
        print(f"sens {r['comparison']:<20} s={r['assumed_seed_sd']} z={r['z']} p={r['p_two_sided_normal']:.2g}")
    print("seed sd needed for z=2:", need_sd)
    for r in out["v1_table1_welch"]["rows"]:
        print(f"v1 T1 {r['comparison']:<24} diff={r['diff']:+.2f} t={r['welch_t']} p={r['p_two_sided']:.3f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
