"""PhysCoRe 논문 표 수치를 다시 계산하고 본문 주장과 맞는지 확인한다 (수 초, 표준 라이브러리 + scipy).

입력: 논문 v3의 표 값을 아래 상수로 옮겨 적었다 (pdftotext 결과 paper/v3.txt 기준).
출력: analysis/recompute_results.json

재현이 아니다. 학습, 시뮬레이션, 공식 코드 실행은 하지 않는다.
"""
import json
from pathlib import Path

from scipy import stats

OUT = Path(__file__).with_name("recompute_results.json")

# 표 2: 미래 예측 (elastic, elastoplastic). CD, Track, IoU, PSNR, SSIM, LPIPS
T2 = {
    "PGND":     {"elastic": [0.03420, 0.13175, 37.8, 10.74, 0.103, 0.416],
                 "elastoplastic": [0.04311, 0.13263, 23.9, 9.87, 0.081, 0.440]},
    "PhysTwin": {"elastic": [0.01617, 0.03016, 70.5, 14.15, 0.309, 0.199],
                 "elastoplastic": [0.00778, 0.01821, 63.1, 14.16, 0.285, 0.154]},
    "PhysCoRe": {"elastic": [0.00910, 0.02557, 68.2, 14.47, 0.302, 0.194],
                 "elastoplastic": [0.00541, 0.01670, 66.1, 15.02, 0.286, 0.130]},
}
METRICS = ["CD", "Track", "IoU", "PSNR", "SSIM", "LPIPS"]
HIGHER_IS_BETTER = {"IoU", "PSNR", "SSIM"}

# 표 3: RfD 절제. (CD, Track)
T3 = {
    "MfM":     {"elastic": (0.01047, 0.02723), "elastoplastic": (0.00658, 0.01981)},
    "MfM+RfD": {"elastic": (0.00910, 0.02557), "elastoplastic": (0.00541, 0.01670)},
}

# 표 4: 계획 Chamfer 거리 (mm)
T4 = {"rope": {"PGND": 48.8, "PhysTwin": 17.3, "PhysCoRe": 6.2},
      "towel": {"PGND": 30.6, "PhysTwin": 7.1, "PhysCoRe": 5.3}}

# 표 5: 실행 시간 (s). 식별/최적화, 테스트 롤아웃 (12개 평가 에피소드 평균)
T5 = {"PhysTwin": (930.0, 7.6), "PGND": (8280.0, 1.9), "PhysCoRe": (11.4, 8.5)}

N_EVAL_EPISODES = 12  # 표 5 설명: 12개 평가 에피소드 평균


def rel_change(base, new, higher_better=False):
    """기준 대비 상대 변화 (%). 낮을수록 좋은 지표는 감소율이 양수다."""
    if higher_better:
        return 100.0 * (new - base) / base
    return 100.0 * (base - new) / base


def main():
    res = {"table2_vs_phystwin": {}, "table2_vs_pgnd": {}, "table3_rfd": {}, "table4_planning": {},
           "table5_runtime": {}, "claim_checks": {}, "significance_bounds": {}}

    # 표 2: PhysCoRe 대 PhysTwin, 지표별 상대 변화 (양수 = PhysCoRe가 더 좋음)
    for grp in ["elastic", "elastoplastic"]:
        row = {}
        for i, m in enumerate(METRICS):
            base = T2["PhysTwin"][grp][i]
            new = T2["PhysCoRe"][grp][i]
            row[m] = round(rel_change(base, new, m in HIGHER_IS_BETTER), 2)
        res["table2_vs_phystwin"][grp] = row
        row2 = {}
        for i, m in enumerate(METRICS):
            base = T2["PGND"][grp][i]
            new = T2["PhysCoRe"][grp][i]
            row2[m] = round(rel_change(base, new, m in HIGHER_IS_BETTER), 2)
        res["table2_vs_pgnd"][grp] = row2

    # 본문 주장 대조 (5.2절)
    res["claim_checks"]["text_CD_elastic_pct"] = {
        "text": 43.7, "recomputed": res["table2_vs_phystwin"]["elastic"]["CD"]}
    res["claim_checks"]["text_CD_elastoplastic_pct"] = {
        "text": 30.5, "recomputed": res["table2_vs_phystwin"]["elastoplastic"]["CD"]}
    res["claim_checks"]["text_Track_elastic_pct"] = {
        "text": 15.2, "recomputed": res["table2_vs_phystwin"]["elastic"]["Track"]}
    res["claim_checks"]["text_Track_elastoplastic_pct"] = {
        "text": 8.3, "recomputed": res["table2_vs_phystwin"]["elastoplastic"]["Track"]}
    # 5.3절 RfD 절제
    for grp in ["elastic", "elastoplastic"]:
        a = T3["MfM"][grp][0]
        b = T3["MfM+RfD"][grp][0]
        res["table3_rfd"][grp] = {
            "CD_reduction_pct": round(rel_change(a, b), 2),
            "Track_reduction_pct": round(rel_change(T3["MfM"][grp][1], T3["MfM+RfD"][grp][1]), 2)}
    res["claim_checks"]["text_RfD_elastic_pct"] = {
        "text": 13.1, "recomputed": res["table3_rfd"]["elastic"]["CD_reduction_pct"]}
    res["claim_checks"]["text_RfD_elastoplastic_pct"] = {
        "text": 17.8, "recomputed": res["table3_rfd"]["elastoplastic"]["CD_reduction_pct"]}
    # 표 2의 PhysCoRe 행과 표 3의 MfM+RfD 행이 같은 값인지 확인 (일관성)
    res["claim_checks"]["table2_PhysCoRe_equals_table3_full"] = {
        "elastic": T2["PhysCoRe"]["elastic"][:2] == list(T3["MfM+RfD"]["elastic"]),
        "elastoplastic": T2["PhysCoRe"]["elastoplastic"][:2] == list(T3["MfM+RfD"]["elastoplastic"])}

    # 표 4: 계획. 상대 개선
    for task, d in T4.items():
        res["table4_planning"][task] = {
            "vs_PhysTwin_pct": round(rel_change(d["PhysTwin"], d["PhysCoRe"]), 1),
            "vs_PGND_pct": round(rel_change(d["PGND"], d["PhysCoRe"]), 1)}

    # 표 5: 실행 시간
    idt, tst = {}, {}
    for k, (ident, test) in T5.items():
        idt[k] = ident
        tst[k] = test
    res["table5_runtime"] = {
        "identification_speedup_vs_PhysTwin": round(idt["PhysTwin"] / idt["PhysCoRe"], 1),
        "identification_speedup_vs_PGND": round(idt["PGND"] / idt["PhysCoRe"], 1),
        "test_time_PhysCoRe_over_PhysTwin": round(tst["PhysCoRe"] / tst["PhysTwin"], 3),
        "test_time_PhysCoRe_over_PGND": round(tst["PhysCoRe"] / tst["PGND"], 2),
        "ms_per_frame_text": 110,
        "implied_predicted_frames_per_episode": round(tst["PhysCoRe"] / 0.110, 1),
    }

    # 유의성: 에피소드 단위 쌍체 자료가 없다. 대신 "이 정도 평균 차이가 p<0.05가 되려면
    # 에피소드별 차이의 SD가 얼마 이하여야 하는가"를 계산한다 (n은 12를 상한으로 가정).
    t_crit = stats.t.ppf(0.975, df=N_EVAL_EPISODES - 1)
    bounds = {}
    for grp in ["elastic", "elastoplastic"]:
        for i, m in enumerate(["CD", "Track"]):
            base = T2["PhysTwin"][grp][i]
            new = T2["PhysCoRe"][grp][i]
            d = base - new
            sd_max = d * (N_EVAL_EPISODES ** 0.5) / t_crit
            bounds[f"{grp}_{m}"] = {
                "mean_diff": round(d, 6),
                "max_sd_of_paired_diff_for_p_lt_0.05_n12": round(sd_max, 6),
                "max_sd_as_pct_of_PhysTwin_mean": round(100.0 * sd_max / base, 1)}
    res["significance_bounds"] = {"t_crit_df11_two_sided_05": round(float(t_crit), 4),
                                  "n_assumed": N_EVAL_EPISODES, "rows": bounds,
                                  "note": "표 2는 평균만 있고 에피소드별 값과 SD가 없어 검정 불가. 위 값은 필요 조건이다."}
    # 부호 검정 상한: 12개 모두 같은 방향이면 양측 p
    res["significance_bounds"]["sign_test_all_12_same_direction_two_sided_p"] = round(2 * 0.5 ** 12, 6)

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res["table2_vs_phystwin"], ensure_ascii=False, indent=1))
    print("table3", res["table3_rfd"])
    print("claims", json.dumps(res["claim_checks"], ensure_ascii=False))
    print("table4", res["table4_planning"])
    print("table5", res["table5_runtime"])
    print("bounds", json.dumps(res["significance_bounds"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
