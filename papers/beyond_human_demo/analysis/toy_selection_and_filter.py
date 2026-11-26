"""토이 두 개 (CPU, numpy/scipy, 수 초). 결과는 toy_selection_and_filter_results.json.

실행: python3 analysis/toy_selection_and_filter.py

S. 선택 편향 (식 4: k* = argmax SuccessRate(D(k)), 표 1의 G*는 그 최댓값):
   세 라운드(각 10회)의 참 성공률이 같다고 해도 최댓값은 참값보다 높게 나온다. 정확한 이항 분포로 그 크기를 계산한다.
   라운드 간 증가가 있는지는 Cochran-Armitage 추세 검정(점수 0, 1, 2)으로 본다.
   논문의 라운드별 성공 횟수: tomato 6, 6, 7 / marker 7, 6, 9 / wine 0, 0, 9 (표 1).

F. 공개 코드의 기본값을 옮겨 적은 산술 검사 (코드는 읽기만 했고 실행하지 않았다).
   값의 출처: https://github.com/Yuchen-Song/glide_code, 커밋 8ebb33a6dca12a2091facccdc45a1a6ffbb50e66
   (복제본 repo/ 는 `git clone <url> repo && git -C repo checkout <commit>` 로 다시 만든다. 이 스크립트는 repo/ 를 읽지 않는다.)
   F1 접시 이동 필터의 carry_differential_gain 0.03 때문에 높이 차 제한 0.8 cm가 작동하려면 조작자 입력의 높이 차가 얼마나 커야 하는가
   F2 접시 중점 수직 속도 상한 0.055 m/s에서 높이를 올리는 데 걸리는 최소 시간
   F3 마커 놓기 단계의 최소 소요 시간 (고정 지연과 손 열림 속도)
   F4 와인 병 기울기 증폭 (28도부터 1.85배, 135도 상한)이 조작자의 손목 각도를 어떻게 바꾸는가
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ROUNDS = {"tomato": (6, 6, 7), "marker": (7, 6, 9), "wine": (0, 0, 9)}
N = 10


def max_dist(p, n=N, r=3):
    """r개의 독립 Binomial(n, p) 의 최댓값 분포 (정확)."""
    k = np.arange(n + 1)
    cdf = stats.binom.cdf(k, n, p)
    cdf_prev = np.concatenate([[0.0], cdf[:-1]])
    return cdf ** r - cdf_prev ** r


def range_prob_ge(p, obs_range, n=N, r=3):
    """r개 Binomial(n, p) 의 (최대 - 최소) >= obs_range 일 확률 (정확, 3중 합)."""
    pm = stats.binom.pmf(np.arange(n + 1), n, p)
    tot = 0.0
    for a in range(n + 1):
        for b in range(n + 1):
            for c in range(n + 1):
                if max(a, b, c) - min(a, b, c) >= obs_range:
                    tot += pm[a] * pm[b] * pm[c]
    return float(tot)


def cochran_armitage(ks):
    ks = np.array(ks, float)
    ns = np.full(len(ks), float(N))
    s = np.arange(len(ks), dtype=float)
    p = ks.sum() / ns.sum()
    t = (s * (ks - ns * p)).sum()
    var = p * (1 - p) * ((ns * s * s).sum() - (ns * s).sum() ** 2 / ns.sum())
    z = t / np.sqrt(var)
    return float(z), float(2 * stats.norm.sf(abs(z)))


def part_s():
    out = {"bias_curve": {}, "tasks": {}}
    for p in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        d = max_dist(p)
        out["bias_curve"][str(p)] = dict(expected_max_rate=round(float((d * np.arange(N + 1)).sum() / N), 3),
                                          bias=round(float((d * np.arange(N + 1)).sum() / N - p), 3))
    for t, ks in ROUNDS.items():
        p_hat = sum(ks) / (3 * N)
        d = max_dist(p_hat)
        z, pz = cochran_armitage(ks)
        spread = max(ks) - min(ks)
        out["tasks"][t] = dict(
            rounds=list(ks), pooled_p=round(p_hat, 3),
            expected_max_rate_if_equal_rounds=round(float((d * np.arange(N + 1)).sum() / N), 3),
            p_best_of_3_ge_observed_max=round(float(d[max(ks):].sum()), 3),
            observed_spread=spread,
            p_spread_ge_observed_if_equal_rounds=round(range_prob_ge(p_hat, spread), 4),
            trend_z=round(z, 3), trend_p=round(pz, 4),
        )
    return out


def part_f():
    out = {}
    # F1: 접시 필터 (teleop_gripper_plate_glide.py PlateLiftTeleopFilter.filter_pair)
    gain, hdiff, sep = 0.03, 0.008, 0.02  # carry_differential_gain, carry_max_height_diff, carry_max_separation_delta
    out["F1_plate_differential"] = dict(
        carry_differential_gain=gain,
        height_clamp_activates_when_raw_height_diff_exceeds_m=round(hdiff / gain, 4),  # 0.8 cm / 0.03
        separation_clamp_activates_when_raw_xy_change_exceeds_m=round(sep / gain, 4),   # 2 cm / 0.03
        note="half_delta = gain * (raw_half_sep - init_half_sep); 클램프는 half_delta 에 걸린다. "
             "조작자 입력의 좌우 높이 차 d 는 로봇에서 gain * d 로 나타난다(클램프 전).",
        commanded_height_diff_for_raw_5cm_m=round(gain * 0.05, 4),
        commanded_height_diff_for_raw_10cm_m=round(gain * 0.10, 4),
    )
    # F2: 중점 수직 속도 상한 0.055 m/s. Fig. 3의 높이 변화(그림 판독 0.08 -> 0.30 m)는 판독값이다.
    for dz in (0.10, 0.22, 0.30):
        out.setdefault("F2_min_lift_time_s", {})[str(dz)] = round(dz / 0.055, 2)
    # F3: 마커 놓기. 열림 시작(닫힘 비율 0.70 이하) 전 지연 0.15 s, 열림 속도 상한 0.90 travel/s로 0.70 -> 0.20,
    #     그 뒤 닫힘 비율이 0.20 이하로 1.05 s(marker-release-retreat-s) 유지되면 놓기 단계가 끝난다.
    open_s = (0.70 - 0.20) / 0.90
    out["F3_marker_release_min_s"] = dict(
        gripper_release_delay_s=0.15, open_070_to_020_s=round(open_s, 2), retreat_hold_after_open_s=1.05,
        ee_pinned_at_least_s=0.55,
        total_s=round(0.15 + open_s + 1.05, 2),
        note="논문 표 4는 0.55 s 고정과 50%·20% 기준을 적었고, 1.05 s 유지 시간은 적지 않았다.",
    )
    # F4: 와인 병 기울기 증폭. robot = start + gain*(op - start) for op > start, 상한 135도
    start, g, cap, pour_start, carry_cap = 28.0, 1.85, 135.0, 36.0, 90.0
    def robot(op):
        return min(start + g * (op - start), cap) if op > start else op
    out["F4_wine_tilt_boost"] = dict(
        operator_deg_to_reach_pour_intent_36=round(start + (pour_start - start) / g, 1),
        operator_deg_to_reach_90=round(start + (carry_cap - start) / g, 1),
        operator_deg_to_hit_135_cap=round(start + (cap - start) / g, 1),
        samples={str(op): round(robot(op), 1) for op in (20, 28, 40, 60, 80, 90, 103)},
        note="분석 D의 로그: 조작자 손목 약 103도, 로봇 병 기울기 약 88도에서 멈춤(증폭 도입 전 90도 한계)."
             " 증폭 뒤에는 103도가 135도 상한에 걸림.",
    )
    return out


def main():
    res = {"S_selection": part_s(), "F_code_constants": part_f()}
    (ROOT / "analysis" / "toy_selection_and_filter_results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res["S_selection"]["tasks"], ensure_ascii=False, indent=1))
    print(json.dumps(res["F_code_constants"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
